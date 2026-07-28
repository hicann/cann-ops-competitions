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

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 40;
constexpr uint32_t FIXED_RANK_SIZE = 16;
constexpr uint32_t RANKS_PER_SERVER = 8;
constexpr uint32_t AICPU_THREAD_NUM = 8;

HcclResult FindFirstLink(HcclComm comm, uint32_t srcRank, uint32_t dstRank, const CommLink *&selectedLink)
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));

    for (uint32_t layerIdx = 0; layerIdx < netLayerNum; ++layerIdx) {
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayers[layerIdx], srcRank, dstRank, &links, &linkNum));
        if (linkNum != 0) {
            selectedLink = &links[0];
            return HCCL_SUCCESS;
        }
    }

    HCCL_ERROR("No link found between rank[%u] and rank[%u]", srcRank, dstRank);
    return HCCL_E_INTERNAL;
}

HcclResult BuildChannelDesc(HcclComm comm, uint32_t srcRank, uint32_t dstRank, HcclChannelDesc &desc)
{
    const CommLink *link = nullptr;
    CHK_RET(FindFirstLink(comm, srcRank, dstRank, link));
    CHK_RET(HcclChannelDescInit(&desc, 1));
    desc.remoteRank = dstRank;
    desc.notifyNum = CHANNEL_NOTIFY_NUM;
    desc.channelProtocol = link->linkAttr.linkProtocol;
    desc.localEndpoint.protocol = link->srcEndpointDesc.protocol;
    desc.localEndpoint.commAddr = link->srcEndpointDesc.commAddr;
    desc.localEndpoint.loc = link->srcEndpointDesc.loc;
    desc.remoteEndpoint.protocol = link->dstEndpointDesc.protocol;
    desc.remoteEndpoint.commAddr = link->dstEndpointDesc.commAddr;
    desc.remoteEndpoint.loc = link->dstEndpointDesc.loc;
    return HCCL_SUCCESS;
}

HcclResult AcquireHierarchicalChannels(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtx)
{
    CHK_PRT_RET(param.rankSize != FIXED_RANK_SIZE || param.myRank >= param.rankSize,
        HCCL_ERROR("V8.1 requires 16 valid ranks, rankSize[%u], myRank[%u]", param.rankSize, param.myRank),
        HCCL_E_NOT_SUPPORT);
    std::vector<uint32_t> remoteRanks;
    remoteRanks.reserve(RANKS_PER_SERVER);
    const uint32_t serverBase = (param.myRank / RANKS_PER_SERVER) * RANKS_PER_SERVER;
    for (uint32_t localRank = 0; localRank < RANKS_PER_SERVER; ++localRank) {
        const uint32_t peerRank = serverBase + localRank;
        if (peerRank != param.myRank) {
            remoteRanks.push_back(peerRank);
        }
    }
    const uint32_t pairedRank
        = param.myRank < RANKS_PER_SERVER ? param.myRank + RANKS_PER_SERVER : param.myRank - RANKS_PER_SERVER;
    remoteRanks.push_back(pairedRank);
    std::vector<HcclChannelDesc> descs(remoteRanks.size());
    std::vector<ChannelHandle> handles(remoteRanks.size());
    for (size_t idx = 0; idx < remoteRanks.size(); ++idx) {
        CHK_RET(BuildChannelDesc(comm, param.myRank, remoteRanks[idx], descs[idx]));
    }
    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AICPU_TS, descs.data(), descs.size(), handles.data()));

    resCtx.channels.reserve(remoteRanks.size());
    for (size_t idx = 0; idx < remoteRanks.size(); ++idx) {
        void *remoteBuffer = nullptr;
        uint64_t remoteBufferSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, handles[idx], &remoteBuffer, &remoteBufferSize));
        resCtx.channels.push_back(
            ChannelInfo{remoteRanks[idx], CHANNEL_NOTIFY_NUM, handles[idx], {remoteBuffer, remoteBufferSize}});
    }
    HCCL_INFO("V8.1 hierarchical rank[%u] channels[%zu]", param.myRank, resCtx.channels.size());
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("Only HCCL_DATA_TYPE_FP32 is supported, dataType[%d]", static_cast<int>(dataType)),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(op != HCCL_REDUCE_SUM, HCCL_ERROR("Only HCCL_REDUCE_SUM is supported, op[%d]", static_cast<int>(op)),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(count > UINT64_MAX / sizeof(float), HCCL_ERROR("AllReduce count overflows byte size"), HCCL_E_PARA);
    if (count == 0) {
        return HCCL_SUCCESS;
    }

    // 构造算子参数
    OpParam param;
    sprintf(param.tag, "%s", "hccl_custom_allreduce_v8_1_direct_index");
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = count;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;

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
        CHK_PRT_RET(hostCtx == nullptr || hostCtxSize < sizeof(ThreadHandle),
            HCCL_ERROR("Invalid cached Host context, addr[%p], size[%llu]", hostCtx,
                static_cast<unsigned long long>(hostCtxSize)),
            HCCL_E_INTERNAL);
        ThreadHandle *aicpuThread = static_cast<ThreadHandle *>(hostCtx);
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));
    } else {
        // Device 资源不存在，资源构建
        AlgResourceCtx resCtxHost;

        // 从通信域获取 HCCL Buffer（Device上的内存，默认总大小400MB）
        void *cclBufferAddr;
        uint64_t cclBufferSize;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        // ==============================================
        // STEP 2.2: 申请资源Thread和Channel
        // ==============================================

        // 创建 AICPU_TS 通信引擎上的 thread 资源
        uint32_t threadNum = AICPU_THREAD_NUM;
        uint32_t notifyNumPerThread = 8;

        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, notifyNumPerThread, resCtxHost.threads.data()));
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        CHK_RET(AcquireHierarchicalChannels(comm, param, resCtxHost));

        // ==============================================
        // STEP 2.3: 申请通信引擎上下文
        // ==============================================
        // 申请 AICPU 通信引擎上下文，存放 AlgResourceCtx 信息
        std::vector<char> seq = resCtxHost.Serialize();
        uint64_t seqSize = seq.size();
        param.ctxSize = seqSize;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, aicpuTsEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, aicpuTsEngine, param.tag, seq.data(), seqSize, 0));
        // 申请 CPU 通信引擎上下文，存放 aicpuThread 句柄
        void *hostCtx = nullptr;
        uint64_t hostCtxSize = sizeof(ThreadHandle);
        const void *aicpuThreadPtr = static_cast<const void *>(&resCtxHost.aicpuThread);
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, cpuTsEngine, hostCtxSize, &hostCtx));
        CHK_RET(HcclEngineCtxCopy(comm, cpuTsEngine, param.tag, aicpuThreadPtr, hostCtxSize, 0));
    }

    // ==============================================
    // STEP 3: 下发 AICPU Kernel
    // ==============================================
    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}
