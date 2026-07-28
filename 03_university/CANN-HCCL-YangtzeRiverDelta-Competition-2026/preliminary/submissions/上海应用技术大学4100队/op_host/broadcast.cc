/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
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
// 0: root-server scatter data
// 1: root-server scatter ack
// 2: segment ready data
// 3: segment ready ack
constexpr uint32_t CHANNEL_NOTIFY_NUM = 4;

// 仍然只用 1 个 AICPU thread。
// 不用 thread notify，避免之前 checker / local notify 问题。
constexpr uint32_t THREAD_NUM = 1;
constexpr uint32_t NOTIFY_NUM_PER_THREAD = 1;

// 新 tag，避免复用 scatter-RD v1/v2 的 EngineCtx。
constexpr char BROADCAST_TAG[] = "hccl_custom_broadcast_hier8_v1";

constexpr uint32_t SERVER_RANK_NUM = 8;
constexpr uint32_t LAYER_INTRA_SERVER = 0;
constexpr uint32_t LAYER_INTER_SERVER = 1;

bool IsSameServer(uint32_t rankA, uint32_t rankB)
{
    return rankA / SERVER_RANK_NUM == rankB / SERVER_RANK_NUM;
}

HcclResult TryFillChannelDescByLayer(HcclComm comm, uint32_t myRank, uint32_t remoteRank,
    uint32_t layer, HcclChannelDesc &channelDesc, bool &filled)
{
    filled = false;

    CommLink *links = nullptr;
    uint32_t linkNum = 0;
    CHK_RET(HcclRankGraphGetLinks(comm, layer, myRank, remoteRank, &links, &linkNum));

    if (linkNum == 0 || links == nullptr) {
        return HCCL_SUCCESS;
    }

    const CommLink &link = links[0];

    channelDesc.remoteRank = remoteRank;
    channelDesc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
    channelDesc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
    channelDesc.localEndpoint.loc = link.srcEndpointDesc.loc;
    channelDesc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
    channelDesc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
    channelDesc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
    channelDesc.channelProtocol = link.linkAttr.linkProtocol;
    channelDesc.notifyNum = CHANNEL_NOTIFY_NUM;

    filled = true;
    return HCCL_SUCCESS;
}

HcclResult FillChannelDesc(HcclComm comm, uint32_t myRank, uint32_t remoteRank, const uint32_t *netLayers,
    uint32_t netLayerNum, HcclChannelDesc &channelDesc)
{
    CHK_RET(HcclChannelDescInit(&channelDesc, 1));

    const uint32_t preferredLayer = IsSameServer(myRank, remoteRank) ? LAYER_INTRA_SERVER : LAYER_INTER_SERVER;

    // Server 内通信优先 layer-0，跨 Server 通信优先 layer-1。
    for (uint32_t layerIndex = 0; layerIndex < netLayerNum; ++layerIndex) {
        if (netLayers[layerIndex] != preferredLayer) {
            continue;
        }

        bool filled = false;
        CHK_RET(TryFillChannelDescByLayer(
            comm, myRank, remoteRank, netLayers[layerIndex], channelDesc, filled));
        if (filled) {
            HCCL_INFO("Use preferred layer[%u] for channel myRank[%u] remoteRank[%u]",
                netLayers[layerIndex], myRank, remoteRank);
            return HCCL_SUCCESS;
        }
    }

    // fallback，保证功能正确性。
    for (uint32_t layerIndex = 0; layerIndex < netLayerNum; ++layerIndex) {
        if (netLayers[layerIndex] == preferredLayer) {
            continue;
        }

        bool filled = false;
        CHK_RET(TryFillChannelDescByLayer(
            comm, myRank, remoteRank, netLayers[layerIndex], channelDesc, filled));
        if (filled) {
            HCCL_INFO("Fallback to layer[%u] for channel myRank[%u] remoteRank[%u]",
                netLayers[layerIndex], myRank, remoteRank);
            return HCCL_SUCCESS;
        }
    }

    HCCL_ERROR("No communication link found, myRank[%u], remoteRank[%u], preferredLayer[%u]",
        myRank, remoteRank, preferredLayer);
    return HCCL_E_PARA;
}

HcclResult BuildChannels(HcclComm comm, CommEngine engine, uint32_t myRank, uint32_t rankSize,
    AlgResourceCtx &resourceCtx)
{
    if (rankSize <= 1) {
        return HCCL_SUCCESS;
    }

    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    if (netLayers == nullptr || netLayerNum == 0) {
        HCCL_ERROR("Rank graph has no valid network layer");
        return HCCL_E_PARA;
    }

    const uint32_t channelNum = rankSize - 1;
    std::vector<HcclChannelDesc> channelDescs(channelNum);
    std::vector<uint32_t> remoteRanks(channelNum);

    uint32_t channelIndex = 0;
    for (uint32_t remoteRank = 0; remoteRank < rankSize; ++remoteRank) {
        if (remoteRank == myRank) {
            continue;
        }

        CHK_RET(FillChannelDesc(
            comm, myRank, remoteRank, netLayers, netLayerNum, channelDescs[channelIndex]));
        remoteRanks[channelIndex] = remoteRank;
        ++channelIndex;
    }

    std::vector<ChannelHandle> channelHandles(channelNum);
    CHK_RET(HcclChannelAcquire(comm, engine, channelDescs.data(), channelNum, channelHandles.data()));

    resourceCtx.channels.resize(channelNum);
    for (uint32_t i = 0; i < channelNum; ++i) {
        ChannelInfo &channelInfo = resourceCtx.channels[i];
        channelInfo.remoteRank = remoteRanks[i];
        channelInfo.notifyNum = CHANNEL_NOTIFY_NUM;
        channelInfo.handle = channelHandles[i];

        CHK_RET(HcclChannelGetHcclBuffer(
            comm, channelInfo.handle, &channelInfo.remoteCclMem.addr, &channelInfo.remoteCclMem.size));
        if (channelInfo.remoteCclMem.addr == nullptr || channelInfo.remoteCclMem.size == 0) {
            HCCL_ERROR("Invalid remote HCCL buffer, remoteRank[%u]", channelInfo.remoteRank);
            return HCCL_E_PTR;
        }
    }

    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclBroadcast(
    void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(buf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param;
    const int tagRet = std::snprintf(param.tag, sizeof(param.tag), "%s", BROADCAST_TAG);
    if (tagRet < 0 || static_cast<size_t>(tagRet) >= sizeof(param.tag)) {
        HCCL_ERROR("Failed to construct broadcast tag");
        return HCCL_E_PARA;
    }

    param.inputPtr = buf;
    param.outputPtr = buf;
    param.count = count;
    param.dataType = dataType;
    param.root = root;
    param.opType = HcclCMDType::HCCL_CMD_BROADCAST;

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH] = {};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));

    if (param.rankSize == 0 || param.myRank >= param.rankSize || param.root >= param.rankSize) {
        HCCL_ERROR("Invalid rank information, myRank[%u], rankSize[%u], root[%u]",
            param.myRank, param.rankSize, param.root);
        return HCCL_E_PARA;
    }

    if (dataType != HCCL_DATA_TYPE_FP32) {
        HCCL_ERROR("Only float32 is required and supported by this competition implementation");
        return HCCL_E_PARA;
    }

    if (count == 0 || param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    const CommEngine aicpuTsEngine = CommEngine::COMM_ENGINE_AICPU_TS;
    const CommEngine cpuTsEngine = CommEngine::COMM_ENGINE_CPU_TS;

    CHK_RET(HcclThreadAcquireWithStream(comm, cpuTsEngine, stream, 1, &param.cpuThread));
    CHK_RET(HcclThreadExportToCommEngine(
        comm, 1, &param.cpuThread, aicpuTsEngine, &param.cpuThreadOnAicpu));

    void *ctx = nullptr;
    uint64_t size = 0;

    if (HcclEngineCtxGet(comm, param.tag, aicpuTsEngine, &ctx, &size) == HCCL_SUCCESS) {
        HCCL_INFO("Engine context already exists");

        param.resCtx = ctx;
        param.ctxSize = size;

        void *hostCtx = nullptr;
        uint64_t hostCtxSize = 0;
        CHK_RET(HcclEngineCtxGet(comm, param.tag, cpuTsEngine, &hostCtx, &hostCtxSize));
        if (hostCtx == nullptr || hostCtxSize < sizeof(ThreadHandle)) {
            HCCL_ERROR("Invalid cached host engine context");
            return HCCL_E_PTR;
        }

        ThreadHandle *aicpuThread = static_cast<ThreadHandle *>(hostCtx);
        CHK_RET(HcclThreadExportToCommEngine(
            comm, 1, aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));
    } else {
        AlgResourceCtx resCtxHost;

        CHK_RET(HcclGetHcclBuffer(comm, &resCtxHost.localBuffer.addr, &resCtxHost.localBuffer.size));
        if (resCtxHost.localBuffer.addr == nullptr || resCtxHost.localBuffer.size == 0) {
            HCCL_ERROR("Invalid local HCCL buffer");
            return HCCL_E_PTR;
        }

        resCtxHost.threads.resize(THREAD_NUM);
        CHK_RET(HcclThreadAcquire(
            comm, aicpuTsEngine, THREAD_NUM, NOTIFY_NUM_PER_THREAD, resCtxHost.threads.data()));
        resCtxHost.aicpuThread = resCtxHost.threads[0];

        CHK_RET(HcclThreadExportToCommEngine(
            comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        CHK_RET(BuildChannels(comm, aicpuTsEngine, param.myRank, param.rankSize, resCtxHost));

        std::vector<char> seq = resCtxHost.Serialize();
        if (seq.empty()) {
            HCCL_ERROR("Serialized resource context is empty");
            return HCCL_E_PARA;
        }

        param.ctxSize = seq.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, aicpuTsEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, aicpuTsEngine, param.tag, seq.data(), seq.size(), 0));

        void *hostCtx = nullptr;
        const uint64_t hostCtxSize = sizeof(ThreadHandle);
        const void *aicpuThreadPtr = static_cast<const void *>(&resCtxHost.aicpuThread);

        CHK_RET(HcclEngineCtxCreate(comm, param.tag, cpuTsEngine, hostCtxSize, &hostCtx));
        CHK_RET(HcclEngineCtxCopy(comm, cpuTsEngine, param.tag, aicpuThreadPtr, hostCtxSize, 0));
    }

    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}