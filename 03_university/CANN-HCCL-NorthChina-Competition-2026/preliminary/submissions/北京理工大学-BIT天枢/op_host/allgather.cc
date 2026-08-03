/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdio>
#include <limits>
#include <vector>

#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>

#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"
#include "log.h"

namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 2; // 0: 数据就绪；1: 数据读取完成
constexpr uint32_t THREAD_NOTIFY_NUM = 3;  // 0: 二叉树启动/本地拷贝完成；1/2: 二叉树左右子树完成
constexpr uint32_t MAX_SUPPORTED_RANK_SIZE = 16;
constexpr uint32_t FP32_BYTES = sizeof(float);

HcclResult FindDirectLink(HcclComm comm, uint32_t srcRank, uint32_t dstRank, CommLink &selectedLink)
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    CHK_PTR_NULL(netLayers);

    for (uint32_t layerIdx = 0; layerIdx < netLayerNum; ++layerIdx) {
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayers[layerIdx], srcRank, dstRank, &links, &linkNum));
        if (linkNum == 0) {
            continue;
        }
        CHK_PTR_NULL(links);
        // 赛题拓扑约束为任意两个NPU之间仅一条物理链路，因此直接选择查询到的第一条链路。
        selectedLink = links[0];
        return HCCL_SUCCESS;
    }

    HCCL_ERROR("No physical link found between rank[%u] and rank[%u]", srcRank, dstRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult BuildAlgorithmResources(HcclComm comm, const OpParam &param, CommEngine engine, AlgResourceCtx &resCtx)
{
    void *cclBufferAddr = nullptr;
    uint64_t cclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    CHK_PTR_NULL(cclBufferAddr);
    CHK_PRT_RET(cclBufferSize == 0, HCCL_ERROR("HCCL buffer size is zero"), HCCL_E_MEMORY);
    resCtx.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};
    resCtx.chunkCapacity = cclBufferSize;

    const uint32_t peerCount = (param.rankSize > 0) ? (param.rankSize - 1) : 0;
    // 16-rank竞赛拓扑申请17个Thread：
    // threads[0]为主控/Bank准备，threads[1..15]固定负责15条Channel，
    // threads[16]独立完成本Rank input -> output[myRank]，避免该本地拷贝进入网络关键路径。
    // 非16-rank场景继续保留原资源规模。
    const uint32_t threadNum = (peerCount == 0)
        ? 1
        : (peerCount + ((param.rankSize == 16) ? 2U : 1U));
    resCtx.threads.resize(threadNum);
    CHK_RET(HcclThreadAcquire(comm, engine, threadNum, THREAD_NOTIFY_NUM, resCtx.threads.data()));
    resCtx.aicpuThread = resCtx.threads[0];

    if (peerCount == 0) {
        return HCCL_SUCCESS;
    }

    std::vector<HcclChannelDesc> channelDescs(peerCount);
    CHK_RET(HcclChannelDescInit(channelDescs.data(), peerCount));
    resCtx.channels.resize(peerCount);

    uint32_t channelIdx = 0;
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }

        CommLink link;
        CHK_RET(FindDirectLink(comm, param.myRank, remoteRank, link));
        HcclChannelDesc &desc = channelDescs[channelIdx];
        desc.remoteRank = remoteRank;
        desc.channelProtocol = link.linkAttr.linkProtocol;
        desc.localEndpoint = link.srcEndpointDesc;
        desc.remoteEndpoint = link.dstEndpointDesc;
        desc.notifyNum = CHANNEL_NOTIFY_NUM;

        resCtx.channels[channelIdx].remoteRank = remoteRank;
        resCtx.channels[channelIdx].notifyNum = CHANNEL_NOTIFY_NUM;
        ++channelIdx;
    }

    std::vector<ChannelHandle> channelHandles(peerCount);
    CHK_RET(HcclChannelAcquire(comm, engine, channelDescs.data(), peerCount, channelHandles.data()));

    for (uint32_t idx = 0; idx < peerCount; ++idx) {
        ChannelInfo &channel = resCtx.channels[idx];
        channel.handle = channelHandles[idx];
        CHK_RET(HcclChannelGetHcclBuffer(
            comm, channel.handle, &channel.remoteCclMem.addr, &channel.remoteCclMem.size));
        CHK_PTR_NULL(channel.remoteCclMem.addr);
        CHK_PRT_RET(channel.remoteCclMem.size == 0,
            HCCL_ERROR("Remote HCCL buffer of rank[%u] is empty", channel.remoteRank), HCCL_E_MEMORY);
        resCtx.chunkCapacity = std::min(resCtx.chunkCapacity, channel.remoteCclMem.size);
    }

    CHK_PRT_RET(resCtx.chunkCapacity == 0, HCCL_ERROR("Common chunk capacity is zero"), HCCL_E_MEMORY);
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
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("Only FP32 is supported, dataType[%d]", static_cast<int32_t>(dataType)), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(sendCount > (std::numeric_limits<uint64_t>::max() / FP32_BYTES),
        HCCL_ERROR("sendCount overflow, sendCount[%lu]", sendCount), HCCL_E_PARA);

    OpParam param;
    const int32_t tagRet = std::snprintf(param.tag, sizeof(param.tag), "%s", "hccl_custom_allgather_v26_large_triple_decoupled");
    CHK_PRT_RET(tagRet < 0 || static_cast<size_t>(tagRet) >= sizeof(param.tag),
        HCCL_ERROR("Failed to build operator tag"), HCCL_E_INTERNAL);
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = sendCount;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH] = {0};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_SUPPORTED_RANK_SIZE,
        HCCL_ERROR("Unsupported rankSize[%u], expected 1..%u", param.rankSize, MAX_SUPPORTED_RANK_SIZE), HCCL_E_PARA);
    const uint64_t sendBytes = sendCount * FP32_BYTES;
    CHK_PRT_RET(sendBytes > 0 && param.rankSize > (std::numeric_limits<uint64_t>::max() / sendBytes),
        HCCL_ERROR("AllGather output size overflow"), HCCL_E_PARA);

    const CommEngine aicpuTsEngine = CommEngine::COMM_ENGINE_AICPU_TS;
    const CommEngine cpuTsEngine = CommEngine::COMM_ENGINE_CPU_TS;

    CHK_RET(HcclThreadAcquireWithStream(comm, cpuTsEngine, stream, 1, &param.cpuThread));
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &param.cpuThread, aicpuTsEngine, &param.cpuThreadOnAicpu));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, aicpuTsEngine, &ctx, &size) == HCCL_SUCCESS) {
        HCCL_INFO("Reuse AllGather engine context");
        param.resCtx = ctx;
        param.ctxSize = size;

        void *hostCtx = nullptr;
        uint64_t hostCtxSize = 0;
        CHK_RET(HcclEngineCtxGet(comm, param.tag, cpuTsEngine, &hostCtx, &hostCtxSize));
        CHK_PRT_RET(hostCtxSize != sizeof(ThreadHandle),
            HCCL_ERROR("Unexpected host context size[%lu]", hostCtxSize), HCCL_E_INTERNAL);
        ThreadHandle *aicpuThread = static_cast<ThreadHandle *>(hostCtx);
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));
    } else {
        AlgResourceCtx resCtxHost;
        CHK_RET(BuildAlgorithmResources(comm, param, aicpuTsEngine, resCtxHost));
        CHK_RET(HcclThreadExportToCommEngine(
            comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        std::vector<char> seq = resCtxHost.Serialize();
        const uint64_t seqSize = seq.size();
        CHK_PRT_RET(seqSize == 0, HCCL_ERROR("Serialized resource context is empty"), HCCL_E_INTERNAL);
        param.ctxSize = seqSize;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, aicpuTsEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, aicpuTsEngine, param.tag, seq.data(), seqSize, 0));

        void *hostCtx = nullptr;
        const uint64_t hostCtxSize = sizeof(ThreadHandle);
        const void *aicpuThreadPtr = static_cast<const void *>(&resCtxHost.aicpuThread);
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, cpuTsEngine, hostCtxSize, &hostCtx));
        CHK_RET(HcclEngineCtxCopy(comm, cpuTsEngine, param.tag, aicpuThreadPtr, hostCtxSize, 0));
    }

    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}
