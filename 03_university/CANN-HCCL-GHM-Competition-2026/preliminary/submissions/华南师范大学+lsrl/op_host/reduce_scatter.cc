/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>

#include <algorithm>
#include <cstdio>
#include <limits>
#include <vector>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

namespace {
constexpr uint32_t INTRA_SERVER_LAYER = 0;
constexpr uint32_t INTER_SERVER_LAYER = 1;
constexpr uint32_t SERVER_RANK_SIZE = 8;
constexpr uint32_t CHANNEL_NOTIFY_NUM = REDUCE_SCATTER_CHANNEL_NOTIFY_NUM;
constexpr uint32_t EXPECTED_RANK_SIZE = SERVER_RANK_SIZE * 2;

bool ContainsRank(const std::vector<uint32_t> &ranks, uint32_t rank)
{
    return std::binary_search(ranks.begin(), ranks.end(), rank);
}

HcclResult FillChannelDesc(HcclComm comm, uint32_t myRank, uint32_t remoteRank, uint32_t netLayer,
    HcclChannelDesc *channelDesc)
{
    CHK_PTR_NULL(channelDesc);

    CommLink *links = nullptr;
    uint32_t linkNum = 0;
    CHK_RET(HcclRankGraphGetLinks(comm, netLayer, myRank, remoteRank, &links, &linkNum));

    for (uint32_t linkIdx = 0; linkIdx < linkNum; ++linkIdx) {
        const CommLink &link = links[linkIdx];
        if (link.linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_UBC_CTP) {
            continue;
        }

        channelDesc->remoteRank = remoteRank;
        channelDesc->notifyNum = CHANNEL_NOTIFY_NUM;
        channelDesc->channelProtocol = link.linkAttr.linkProtocol;
        channelDesc->localEndpoint.protocol = link.srcEndpointDesc.protocol;
        channelDesc->localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
        channelDesc->localEndpoint.loc = link.srcEndpointDesc.loc;
        channelDesc->remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
        channelDesc->remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
        channelDesc->remoteEndpoint.loc = link.dstEndpointDesc.loc;
        return HCCL_SUCCESS;
    }

    HCCL_ERROR("No UBC_CTP link found from rank %u to rank %u on layer %u", myRank, remoteRank, netLayer);
    return HCCL_E_NOT_FOUND;
}

HcclResult GetIntraServerRanks(HcclComm comm, const OpParam &param, std::vector<uint32_t> *intraServerRanks)
{
    CHK_PTR_NULL(intraServerRanks);

    if (param.rankSize != EXPECTED_RANK_SIZE) {
        HCCL_ERROR("ReduceScatter requires the 2x8 topology, actual rank size is %u", param.rankSize);
        return HCCL_E_NOT_SUPPORT;
    }

    uint32_t *intraRanks = nullptr;
    uint32_t intraRankNum = 0;
    CHK_RET(HcclRankGraphGetRanksByLayer(comm, INTRA_SERVER_LAYER, &intraRanks, &intraRankNum));
    if (intraRanks == nullptr || intraRankNum != SERVER_RANK_SIZE) {
        HCCL_ERROR("Invalid intra-server topology, rank number is %u", intraRankNum);
        return HCCL_E_NOT_SUPPORT;
    }

    intraServerRanks->assign(intraRanks, intraRanks + intraRankNum);
    std::sort(intraServerRanks->begin(), intraServerRanks->end());
    if (!ContainsRank(*intraServerRanks, param.myRank)) {
        HCCL_ERROR("Current rank %u is absent from its intra-server topology", param.myRank);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult AcquireChannels(HcclComm comm, const OpParam &param, const std::vector<uint32_t> &intraServerRanks,
    AlgResourceCtx *resCtx)
{
    CHK_PTR_NULL(resCtx);

    const uint32_t channelNum = param.rankSize - 1;
    std::vector<HcclChannelDesc> channelDescs(channelNum);
    std::vector<ChannelHandle> channelHandles(channelNum);
    CHK_RET(HcclChannelDescInit(channelDescs.data(), channelNum));

    uint32_t channelIdx = 0;
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }

        const uint32_t netLayer =
            ContainsRank(intraServerRanks, remoteRank) ? INTRA_SERVER_LAYER : INTER_SERVER_LAYER;
        CHK_RET(FillChannelDesc(comm, param.myRank, remoteRank, netLayer, &channelDescs[channelIdx]));
        resCtx->channels[channelIdx].remoteRank = remoteRank;
        resCtx->channels[channelIdx].notifyNum = CHANNEL_NOTIFY_NUM;
        ++channelIdx;
    }

    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AICPU_TS, channelDescs.data(), channelNum,
        channelHandles.data()));
    for (uint32_t idx = 0; idx < channelNum; ++idx) {
        resCtx->channels[idx].handle = channelHandles[idx];
        CHK_RET(HcclChannelGetHcclBuffer(comm, channelHandles[idx], &resCtx->channels[idx].remoteCclMem.addr,
            &resCtx->channels[idx].remoteCclMem.size));
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    if (dataType != HCCL_DATA_TYPE_FP32 || op != HCCL_REDUCE_SUM) {
        HCCL_ERROR("Only FP32 SUM ReduceScatter is supported, dataType[%d], op[%d]", dataType, op);
        return HCCL_E_NOT_SUPPORT;
    }
    if (recvCount > std::numeric_limits<uint64_t>::max() / sizeof(float)) {
        HCCL_ERROR("recvCount[%llu] overflows the FP32 byte size", static_cast<unsigned long long>(recvCount));
        return HCCL_E_PARA;
    }

    // 构造算子参数
    OpParam param;
    const int tagLength = snprintf(param.tag, sizeof(param.tag), "%s", "hccl_custom_reduce_scatter");
    if (tagLength < 0 || static_cast<size_t>(tagLength) >= sizeof(param.tag)) {
        HCCL_ERROR("Failed to construct the ReduceScatter resource tag");
        return HCCL_E_INTERNAL;
    }
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;

    // 注册算子信息
    HcclDfxOpInfo dfxInfo;
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    // ==============================================
    // STEP 1: 解析拓扑信息
    // ==============================================
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));

    // ==============================================
    // STEP 2: 创建资源
    // ==============================================
    CommEngine aicpuTsEngine = CommEngine::COMM_ENGINE_AICPU_TS;
    CommEngine cpuTsEngine = CommEngine::COMM_ENGINE_CPU_TS;

    // ==============================================
    // STEP 2.1: 申请用于 Host/Device 同步的通信资源
    // ==============================================
    // 将用户传入的 stream 转换为 thread，并申请 Notify；同时导出为 AICPU 上可用的 thread
    CHK_RET(HcclThreadAcquireWithStream(comm, cpuTsEngine, stream, 1, &param.cpuThread));
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &param.cpuThread, aicpuTsEngine, &param.cpuThreadOnAicpu));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, aicpuTsEngine, &ctx, &size) == HCCL_SUCCESS) {
        // AICPU 资源已经存在，复用资源
        HCCL_INFO("Engine context already exists");
        param.resCtx = ctx;
        param.ctxSize = size;

        // Host 资源已经存在，复用资源
        void *hostCtx = nullptr;
        uint64_t hostCtxSize = 0;
        CHK_RET(HcclEngineCtxGet(comm, param.tag, cpuTsEngine, &hostCtx, &hostCtxSize));
        ThreadHandle *aicpuThread = static_cast<ThreadHandle *>(hostCtx);
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));
    } else {
        // Device 资源不存在，资源构建
        AlgResourceCtx resCtxHost{};

        // 从通信域获取 HCCL Buffer（Device上的内存，默认总大小400MB）
        void *cclBufferAddr;
        uint64_t cclBufferSize;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        // ==============================================
        // STEP 2.2: 申请资源Thread和Channel
        // ==============================================

        // Thread 0 schedules deterministic reduction.  Seven workers own the
        // intra-server Mesh links, one pipelines cross-server communication,
        // and eight pack strided input blocks for large-message slices.
        constexpr uint32_t threadNum = REDUCE_SCATTER_AICPU_THREAD_NUM;
        constexpr uint32_t notifyNumPerThread = REDUCE_SCATTER_THREAD_NOTIFY_NUM;

        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, notifyNumPerThread,
            resCtxHost.aicpuThreads));
        // Only thread 0 participates in Host/AICPU synchronization.
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThreads[0], cpuTsEngine,
            &param.aicpuThreadOnCpu));

        // 两个 rank 之间仅申请一个 channel。Server 内走 Mesh 层，跨 Server 走 Clos 层。
        std::vector<uint32_t> intraServerRanks;
        CHK_RET(GetIntraServerRanks(comm, param, &intraServerRanks));
        CHK_RET(AcquireChannels(comm, param, intraServerRanks, &resCtxHost));

        // ==============================================
        // STEP 2.3: 申请通信引擎上下文
        // ==============================================
        // 申请 AICPU 通信引擎上下文，存放 AlgResourceCtx 信息
        param.ctxSize = sizeof(AlgResourceCtx);
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, aicpuTsEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, aicpuTsEngine, param.tag, &resCtxHost, param.ctxSize, 0));
        // 申请 CPU 通信引擎上下文，存放 aicpuThread 句柄
        void *hostCtx = nullptr;
        uint64_t hostCtxSize = sizeof(ThreadHandle);
        const void *aicpuThreadPtr = static_cast<const void *>(&resCtxHost.aicpuThreads[0]);
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, cpuTsEngine, hostCtxSize, &hostCtx));
        CHK_RET(HcclEngineCtxCopy(comm, cpuTsEngine, param.tag, aicpuThreadPtr, hostCtxSize, 0));
    }

    // ==============================================
    // STEP 3: 下发 AICPU Kernel
    // ==============================================
    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}
