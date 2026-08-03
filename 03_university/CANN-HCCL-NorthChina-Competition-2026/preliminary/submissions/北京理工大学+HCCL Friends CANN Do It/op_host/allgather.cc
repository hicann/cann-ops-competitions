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

constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;

// CPU_TS contexts are host memory.  Store the handle directly instead of
// using HcclEngineCtxCopy: the latter does not reliably preserve CPU_TS thread
// handles in the simulator when the operator context is reused.
struct AicpuMainThreadCache {
    ThreadHandle thread;
};

HcclResult BuildChannelDesc(HcclComm comm, uint32_t srcRank, uint32_t dstRank, HcclChannelDesc &desc)
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerCount));
    CHK_PRT_RET(netLayers == nullptr || netLayerCount == 0,
        HCCL_ERROR("Rank graph contains no network layers"), HCCL_E_INTERNAL);

    CommLink selected{};
    bool found = false;
    for (uint32_t layerIndex = 0; layerIndex < netLayerCount && !found; ++layerIndex) {
        uint32_t linkCount = 0;
        CommLink *links = nullptr;
        CHK_RET(HcclRankGraphGetLinks(
            comm, netLayers[layerIndex], srcRank, dstRank, &links, &linkCount));
        for (uint32_t linkIndex = 0; linkIndex < linkCount; ++linkIndex) {
            if (links[linkIndex].linkAttr.linkProtocol == COMM_PROTOCOL_UBC_CTP) {
                selected = links[linkIndex];
                found = true;
                break;
            }
        }
    }
    CHK_PRT_RET(!found,
        HCCL_ERROR("No UBC_CTP link from rank[%u] to rank[%u] in [%u] network layers",
            srcRank, dstRank, netLayerCount), HCCL_E_INTERNAL);

    CHK_RET(HcclChannelDescInit(&desc, 1));
    desc.remoteRank = dstRank;
    desc.notifyNum = CHANNEL_NOTIFY_NUM;
    desc.channelProtocol = selected.linkAttr.linkProtocol;
    desc.localEndpoint.protocol = selected.srcEndpointDesc.protocol;
    desc.localEndpoint.commAddr = selected.srcEndpointDesc.commAddr;
    desc.localEndpoint.loc = selected.srcEndpointDesc.loc;
    desc.remoteEndpoint.protocol = selected.dstEndpointDesc.protocol;
    desc.remoteEndpoint.commAddr = selected.dstEndpointDesc.commAddr;
    desc.remoteEndpoint.loc = selected.dstEndpointDesc.loc;
    return HCCL_SUCCESS;
}

HcclResult AcquireChannels(HcclComm comm, const OpParam &param, AlgResourceCtx &ctx)
{
    if (param.rankSize <= 1) {
        return HCCL_SUCCESS;
    }
    const uint32_t peerCount = param.rankSize - 1;
    std::vector<HcclChannelDesc> descs(peerCount);
    std::vector<ChannelHandle> handles(peerCount);
    uint32_t index = 0;
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (rank == param.myRank) {
            continue;
        }
        CHK_RET(BuildChannelDesc(comm, param.myRank, rank, descs[index++]));
    }
    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AICPU_TS, descs.data(), peerCount, handles.data()));

    index = 0;
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (rank == param.myRank) {
            continue;
        }
        void *remoteBuffer = nullptr;
        uint64_t remoteBufferSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, handles[index], &remoteBuffer, &remoteBufferSize));
        ChannelInfo channel;
        channel.remoteRank = rank;
        channel.notifyNum = CHANNEL_NOTIFY_NUM;
        channel.handle = handles[index];
        channel.remoteCclMem = CommBuffer{remoteBuffer, remoteBufferSize};
        ctx.channels.push_back(channel);
        ++index;
    }
    return HCCL_SUCCESS;
}

} // namespace

HcclResult HcclAllGather(
    void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    // 构造算子参数
    OpParam param;
    sprintf(param.tag, "%s", "hccl_custom_allgather");
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = sendCount;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;

    // 注册算子信息
    HcclDfxOpInfo dfxInfo{};
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
        uint64_t hostCtxSize = sizeof(AicpuMainThreadCache);
        CHK_RET(HcclEngineCtxGet(comm, param.tag, cpuTsEngine, &hostCtx, &hostCtxSize));
        CHK_PRT_RET(hostCtx == nullptr || hostCtxSize < sizeof(AicpuMainThreadCache),
            HCCL_ERROR("Invalid cached AICPU main thread context"), HCCL_E_INTERNAL);
        auto *cache = static_cast<AicpuMainThreadCache *>(hostCtx);
        CHK_RET(HcclThreadExportToCommEngine(
            comm, 1, &cache->thread, cpuTsEngine, &param.aicpuThreadOnCpu));
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

        // Keep one control/copy stream plus one stream per peer.  The dedicated
        // control stream pipelines the next CCL slice while every physical
        // Mesh/Clos link remains active on its own worker stream.
        uint32_t threadNum = param.rankSize > 1 ? param.rankSize : 1;
        uint32_t notifyNumPerThread = param.rankSize > 1 ? param.rankSize : 1;

        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, notifyNumPerThread, resCtxHost.threads.data()));
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        CHK_RET(AcquireChannels(comm, param, resCtxHost));

        // 构建 remoteRank -> channel 下标映射，供递归加倍算法快速查找
        resCtxHost.rankToChannelIdx.assign(param.rankSize, INVALID_VALUE_RANKID);
        for (uint32_t i = 0; i < resCtxHost.channels.size(); i++) {
            uint32_t remoteRank = resCtxHost.channels[i].remoteRank;
            if (remoteRank < param.rankSize) {
                resCtxHost.rankToChannelIdx[remoteRank] = i;
            }
        }

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
        uint64_t hostCtxSize = sizeof(AicpuMainThreadCache);
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, cpuTsEngine, hostCtxSize, &hostCtx));
        CHK_PTR_NULL(hostCtx);
        auto *cache = static_cast<AicpuMainThreadCache *>(hostCtx);
        cache->thread = resCtxHost.aicpuThread;
    }

    // ==============================================
    // STEP 3: 下发 AICPU Kernel
    // ==============================================
    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}
