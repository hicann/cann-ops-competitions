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
#include <cstdint>
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
constexpr uint32_t COMPETITION_RANK_SIZE = 16;
constexpr uint32_t LOCAL_RANK_SIZE = 8;
constexpr uint64_t SMALL_MESSAGE_BYTES = 1024ULL * 1024ULL;
constexpr uint32_t CHANNEL_NOTIFY_NUM = 2;
// Thread 0 drives the eight Clos peers in order. Threads 1..7 drive the seven
// server-local Full-Mesh links concurrently.
constexpr uint32_t THREAD_NUM = LOCAL_RANK_SIZE;
constexpr uint32_t THREAD_NOTIFY_NUM = LOCAL_RANK_SIZE - 1;

HcclResult GetChannelDesc(HcclComm comm, uint32_t localRank, uint32_t remoteRank, HcclChannelDesc &channelDesc)
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    CHK_PRT_RET(netLayers == nullptr || netLayerNum == 0,
        HCCL_ERROR("No topology layer found, localRank[%u], remoteRank[%u]", localRank, remoteRank), HCCL_E_PARA);

    for (uint32_t i = 0; i < netLayerNum; ++i) {
        CommLink *linkList = nullptr;
        uint32_t linkNum = 0;
        HcclResult ret = HcclRankGraphGetLinks(comm, netLayers[i], localRank, remoteRank, &linkList, &linkNum);
        if (ret != HCCL_SUCCESS || linkList == nullptr || linkNum == 0) {
            continue;
        }

        CHK_RET(HcclChannelDescInit(&channelDesc, 1));
        channelDesc.remoteRank = remoteRank;
        channelDesc.localEndpoint = linkList[0].srcEndpointDesc;
        channelDesc.remoteEndpoint = linkList[0].dstEndpointDesc;
        channelDesc.channelProtocol = linkList[0].linkAttr.linkProtocol;
        channelDesc.notifyNum = CHANNEL_NOTIFY_NUM;
        return HCCL_SUCCESS;
    }

    HCCL_ERROR("No available link found, localRank[%u], remoteRank[%u]", localRank, remoteRank);
    return HCCL_E_PARA;
}

HcclResult DiscoverTopology(HcclComm comm, uint32_t myRank, AlgResourceCtx &resCtx)
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    CHK_PRT_RET(netLayers == nullptr || netLayerNum < 2,
        HCCL_ERROR("The competition topology requires at least two layers, layerNum[%u]", netLayerNum), HCCL_E_PARA);
    const uint32_t localLayer = netLayers[0];
    const uint32_t outerLayer = netLayers[netLayerNum - 1];

    uint32_t *localRanks = nullptr;
    uint32_t localRankNum = 0;
    CHK_RET(HcclRankGraphGetRanksByLayer(comm, localLayer, &localRanks, &localRankNum));
    CHK_PRT_RET(localRanks == nullptr || localRankNum != LOCAL_RANK_SIZE,
        HCCL_ERROR("The Layer-0 group must contain 8 ranks, rankNum[%u]", localRankNum), HCCL_E_PARA);
    resCtx.localRanks.assign(localRanks, localRanks + localRankNum);
    std::sort(resCtx.localRanks.begin(), resCtx.localRanks.end());

    uint32_t *globalRanks = nullptr;
    uint32_t globalRankNum = 0;
    CHK_RET(HcclRankGraphGetRanksByLayer(comm, outerLayer, &globalRanks, &globalRankNum));
    CHK_PRT_RET(globalRanks == nullptr || globalRankNum != COMPETITION_RANK_SIZE,
        HCCL_ERROR("The outer layer must contain 16 ranks, rankNum[%u]", globalRankNum), HCCL_E_PARA);

    const auto localIt = std::find(resCtx.localRanks.begin(), resCtx.localRanks.end(), myRank);
    CHK_PRT_RET(localIt == resCtx.localRanks.end(), HCCL_ERROR("Rank[%u] is absent from its Layer-0 group", myRank),
        HCCL_E_PARA);

    resCtx.globalRanks.assign(globalRanks, globalRanks + globalRankNum);
    std::sort(resCtx.globalRanks.begin(), resCtx.globalRanks.end());

    std::vector<uint32_t> remoteServerRanks;
    remoteServerRanks.reserve(LOCAL_RANK_SIZE);
    for (uint32_t i = 0; i < globalRankNum; ++i) {
        if (std::find(resCtx.localRanks.begin(), resCtx.localRanks.end(), globalRanks[i]) == resCtx.localRanks.end()) {
            remoteServerRanks.push_back(globalRanks[i]);
        }
    }
    std::sort(remoteServerRanks.begin(), remoteServerRanks.end());
    CHK_PRT_RET(remoteServerRanks.size() != LOCAL_RANK_SIZE,
        HCCL_ERROR("The remote server must contain 8 ranks, rankNum[%lu]", remoteServerRanks.size()), HCCL_E_PARA);

    const uint32_t localIndex = static_cast<uint32_t>(std::distance(resCtx.localRanks.begin(), localIt));
    resCtx.interRank = remoteServerRanks[localIndex];
    return HCCL_SUCCESS;
}

std::vector<uint32_t> BuildPeerRanks(uint32_t myRank, const AlgResourceCtx &resCtx, bool smallMessage)
{
    std::vector<uint32_t> peers;
    if (smallMessage) {
        const auto localIt = std::find(resCtx.localRanks.begin(), resCtx.localRanks.end(), myRank);
        if (localIt == resCtx.localRanks.end()) {
            return peers;
        }
        const uint32_t localIndex = static_cast<uint32_t>(
            std::distance(resCtx.localRanks.begin(), localIt));
        for (uint32_t mask = 1; mask < LOCAL_RANK_SIZE; mask <<= 1) {
            peers.push_back(resCtx.localRanks[localIndex ^ mask]);
        }
        peers.push_back(resCtx.interRank);
        return peers;
    }

    peers.reserve(COMPETITION_RANK_SIZE - 1);
    for (uint32_t rank : resCtx.globalRanks) {
        if (rank != myRank) {
            peers.push_back(rank);
        }
    }
    return peers;
}

HcclResult CreateChannels(HcclComm comm, CommEngine engine, uint32_t myRank,
    bool smallMessage, AlgResourceCtx &resCtx)
{
    const std::vector<uint32_t> remoteRanks = BuildPeerRanks(myRank, resCtx, smallMessage);
    const uint32_t expectedChannelNum = smallMessage ? 4U : COMPETITION_RANK_SIZE - 1;
    CHK_PRT_RET(remoteRanks.size() != expectedChannelNum,
        HCCL_ERROR("Unexpected peer count, small[%d], peers[%lu]", smallMessage, remoteRanks.size()),
        HCCL_E_PARA);
    std::vector<HcclChannelDesc> channelDescs(remoteRanks.size());
    for (uint32_t i = 0; i < remoteRanks.size(); ++i) {
        CHK_RET(GetChannelDesc(comm, myRank, remoteRanks[i], channelDescs[i]));
    }

    std::vector<ChannelHandle> channelHandles(remoteRanks.size());
    CHK_RET(HcclChannelAcquire(
        comm, engine, channelDescs.data(), static_cast<uint32_t>(channelDescs.size()), channelHandles.data()));

    resCtx.channels.resize(remoteRanks.size());
    for (uint32_t i = 0; i < remoteRanks.size(); ++i) {
        void *remoteBuffer = nullptr;
        uint64_t remoteBufferSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, channelHandles[i], &remoteBuffer, &remoteBufferSize));
        CHK_PRT_RET(remoteBuffer == nullptr || remoteBufferSize == 0,
            HCCL_ERROR("Invalid remote HCCL buffer, remoteRank[%u]", remoteRanks[i]), HCCL_E_PARA);

        resCtx.channels[i].remoteRank = remoteRanks[i];
        resCtx.channels[i].notifyNum = CHANNEL_NOTIFY_NUM;
        resCtx.channels[i].handle = channelHandles[i];
        resCtx.channels[i].remoteCclMem = CommBuffer{remoteBuffer, remoteBufferSize};
    }
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
        HCCL_ERROR("Only FP32 is supported, dataType[%d]", static_cast<int32_t>(dataType)), HCCL_E_PARA);
    CHK_PRT_RET(op != HCCL_REDUCE_SUM, HCCL_ERROR("Only SUM is supported, reduceOp[%d]", static_cast<int32_t>(op)),
        HCCL_E_PARA);
    CHK_PRT_RET(count > std::numeric_limits<uint64_t>::max() / sizeof(float),
        HCCL_ERROR("Input size overflow, count[%lu]", count), HCCL_E_PARA);

    const uint64_t totalBytes = count * sizeof(float);
    const bool smallMessage = totalBytes <= SMALL_MESSAGE_BYTES;

    OpParam param;
    std::snprintf(param.tag, sizeof(param.tag), "%s",
        smallMessage ? "hccl_custom_allreduce_small_v14" : "hccl_custom_allreduce_direct_v14");
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = count;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH] = {0};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize != COMPETITION_RANK_SIZE || param.myRank >= param.rankSize,
        HCCL_ERROR("This implementation requires 16 ranks, rank[%u], rankSize[%u]", param.myRank, param.rankSize),
        HCCL_E_PARA);

    CommEngine aicpuTsEngine = CommEngine::COMM_ENGINE_AICPU_TS;
    CommEngine cpuTsEngine = CommEngine::COMM_ENGINE_CPU_TS;

    CHK_RET(HcclThreadAcquireWithStream(comm, cpuTsEngine, stream, 1, &param.cpuThread));
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &param.cpuThread, aicpuTsEngine, &param.cpuThreadOnAicpu));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, aicpuTsEngine, &ctx, &size) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = size;

        void *hostCtx = nullptr;
        uint64_t hostCtxSize = 0;
        CHK_RET(HcclEngineCtxGet(comm, param.tag, cpuTsEngine, &hostCtx, &hostCtxSize));
        CHK_PRT_RET(hostCtx == nullptr || hostCtxSize < sizeof(ThreadHandle),
            HCCL_ERROR("Invalid host engine context, size[%lu]", hostCtxSize), HCCL_E_PARA);
        ThreadHandle *aicpuThread = static_cast<ThreadHandle *>(hostCtx);
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));
    } else {
        AlgResourceCtx resCtxHost;

        void *cclBufferAddr = nullptr;
        uint64_t cclBufferSize = 0;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        CHK_PRT_RET(cclBufferAddr == nullptr || cclBufferSize == 0,
            HCCL_ERROR("Invalid local HCCL buffer, size[%lu]", cclBufferSize), HCCL_E_PARA);
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        const uint32_t threadNum = smallMessage ? 1U : THREAD_NUM;
        const uint32_t threadNotifyNum = smallMessage ? 1U : THREAD_NOTIFY_NUM;
        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(
            comm, aicpuTsEngine, threadNum, threadNotifyNum, resCtxHost.threads.data()));
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        CHK_RET(DiscoverTopology(comm, param.myRank, resCtxHost));
        CHK_RET(CreateChannels(comm, aicpuTsEngine, param.myRank, smallMessage, resCtxHost));

        std::vector<char> seq = resCtxHost.Serialize();
        uint64_t seqSize = seq.size();
        CHK_PRT_RET(seqSize == 0, HCCL_ERROR("Serialized resource context is empty"), HCCL_E_PARA);
        param.ctxSize = seqSize;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, aicpuTsEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, aicpuTsEngine, param.tag, seq.data(), seqSize, 0));

        void *hostCtx = nullptr;
        uint64_t hostCtxSize = sizeof(ThreadHandle);
        const void *aicpuThreadPtr = static_cast<const void *>(&resCtxHost.aicpuThread);
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, cpuTsEngine, hostCtxSize, &hostCtx));
        CHK_RET(HcclEngineCtxCopy(comm, cpuTsEngine, param.tag, aicpuThreadPtr, hostCtxSize, 0));
    }

    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}
