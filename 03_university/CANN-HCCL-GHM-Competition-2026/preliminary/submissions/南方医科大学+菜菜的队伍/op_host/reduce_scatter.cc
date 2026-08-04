/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <vector>

#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"
#include "log.h"

namespace {
constexpr uint32_t SERVER_FABRIC_LAYER = 0;
constexpr uint32_t CLUSTER_FABRIC_LAYER = 1;
constexpr uint32_t CHANNEL_NOTIFY_COUNT = 2;
constexpr uint32_t THREAD_NOTIFY_COUNT = yga_reduce_scatter::PEER_COUNT - 1;

HcclResult FillChannelDescriptor(
    HcclComm comm, uint32_t sourceRank, uint32_t targetRank, uint32_t layer, HcclChannelDesc &descriptor)
{
    CommLink *availableLinks = nullptr;
    uint32_t availableLinkCount = 0;
    CHK_RET(HcclRankGraphGetLinks(comm, layer, sourceRank, targetRank, &availableLinks, &availableLinkCount));
    CHK_PRT_RET(availableLinkCount == 0,
        HCCL_ERROR("No physical path from rank[%u] to rank[%u] on layer[%u]", sourceRank, targetRank, layer),
        HCCL_E_INTERNAL);

    CHK_RET(HcclChannelDescInit(&descriptor, 1));
    const CommLink &selectedLink = availableLinks[0];
    descriptor.remoteRank = targetRank;
    descriptor.notifyNum = CHANNEL_NOTIFY_COUNT;
    descriptor.channelProtocol = selectedLink.linkAttr.linkProtocol;
    descriptor.localEndpoint.protocol = selectedLink.srcEndpointDesc.protocol;
    descriptor.localEndpoint.commAddr = selectedLink.srcEndpointDesc.commAddr;
    descriptor.localEndpoint.loc = selectedLink.srcEndpointDesc.loc;
    descriptor.remoteEndpoint.protocol = selectedLink.dstEndpointDesc.protocol;
    descriptor.remoteEndpoint.commAddr = selectedLink.dstEndpointDesc.commAddr;
    descriptor.remoteEndpoint.loc = selectedLink.dstEndpointDesc.loc;
    return HCCL_SUCCESS;
}

HcclResult ReadCompetitionTopology(
    HcclComm comm, const OpParam &param, AlgResourceCtx &resources, std::vector<uint32_t> &serverRanks)
{
    uint32_t *rankData = nullptr;
    uint32_t rankCount = 0;
    CHK_RET(HcclRankGraphGetRanksByLayer(comm, SERVER_FABRIC_LAYER, &rankData, &rankCount));
    CHK_PRT_RET(rankCount != yga_reduce_scatter::RANKS_PER_SERVER,
        HCCL_ERROR("Expected 8 ranks in the local server, got[%u]", rankCount), HCCL_E_PARA);
    serverRanks.assign(rankData, rankData + rankCount);
    std::sort(serverRanks.begin(), serverRanks.end());

    rankData = nullptr;
    rankCount = 0;
    CHK_RET(HcclRankGraphGetRanksByLayer(comm, CLUSTER_FABRIC_LAYER, &rankData, &rankCount));
    CHK_PRT_RET(rankCount != yga_reduce_scatter::RANK_COUNT,
        HCCL_ERROR("Expected 16 ranks in the communication domain, got[%u]", rankCount), HCCL_E_PARA);
    resources.orderedRanks.assign(rankData, rankData + rankCount);
    std::sort(resources.orderedRanks.begin(), resources.orderedRanks.end());

    auto self = std::find(resources.orderedRanks.begin(), resources.orderedRanks.end(), param.myRank);
    CHK_PRT_RET(self == resources.orderedRanks.end(), HCCL_ERROR("Rank[%u] is absent from the topology", param.myRank),
        HCCL_E_INTERNAL);
    resources.selfIndex = static_cast<uint32_t>(std::distance(resources.orderedRanks.begin(), self));
    return HCCL_SUCCESS;
}

HcclResult OpenPeerPaths(
    HcclComm comm, const OpParam &param, const std::vector<uint32_t> &serverRanks, AlgResourceCtx &resources)
{
    std::vector<uint32_t> peerIndices;
    std::vector<uint32_t> peerLayers;
    peerIndices.reserve(yga_reduce_scatter::PEER_COUNT);
    peerLayers.reserve(yga_reduce_scatter::PEER_COUNT);

    for (uint32_t index = 0; index < resources.orderedRanks.size(); ++index) {
        if (index == resources.selfIndex) {
            continue;
        }
        const uint32_t peerRank = resources.orderedRanks[index];
        const bool sameServer = std::find(serverRanks.begin(), serverRanks.end(), peerRank) != serverRanks.end();
        peerIndices.push_back(index);
        peerLayers.push_back(sameServer ? SERVER_FABRIC_LAYER : CLUSTER_FABRIC_LAYER);
    }

    std::vector<HcclChannelDesc> descriptors(peerIndices.size());
    for (uint32_t i = 0; i < peerIndices.size(); ++i) {
        CHK_RET(FillChannelDescriptor(
            comm, param.myRank, resources.orderedRanks[peerIndices[i]], peerLayers[i], descriptors[i]));
    }

    std::vector<ChannelHandle> channels(peerIndices.size());
    CHK_RET(HcclChannelAcquire(
        comm, CommEngine::COMM_ENGINE_AICPU_TS, descriptors.data(), descriptors.size(), channels.data()));

    resources.peerPaths.resize(peerIndices.size());
    for (uint32_t i = 0; i < peerIndices.size(); ++i) {
        PeerPath &path = resources.peerPaths[i];
        path.rankIndex = peerIndices[i];
        path.rankId = resources.orderedRanks[path.rankIndex];
        path.fabricLayer = peerLayers[i];
        path.notifyCount = CHANNEL_NOTIFY_COUNT;
        path.channel = channels[i];
        CHK_RET(
            HcclChannelGetHcclBuffer(comm, path.channel, &path.remoteWorkspace.address, &path.remoteWorkspace.bytes));
    }
    return HCCL_SUCCESS;
}

HcclResult CreateAlgorithmResources(HcclComm comm, const OpParam &param, AlgResourceCtx &resources)
{
    std::vector<uint32_t> serverRanks;
    CHK_RET(ReadCompetitionTopology(comm, param, resources, serverRanks));
    CHK_RET(HcclGetHcclBuffer(comm, &resources.workspace.address, &resources.workspace.bytes));
    CHK_PRT_RET(resources.workspace.address == nullptr
                    || resources.workspace.bytes < yga_reduce_scatter::PEER_COUNT * yga_reduce_scatter::DMA_ALIGNMENT,
        HCCL_ERROR("HCCL workspace[%llu] is too small", static_cast<unsigned long long>(resources.workspace.bytes)),
        HCCL_E_INTERNAL);

    resources.workerThreads.resize(yga_reduce_scatter::PEER_COUNT);
    CHK_RET(HcclThreadAcquire(comm, CommEngine::COMM_ENGINE_AICPU_TS, resources.workerThreads.size(),
        THREAD_NOTIFY_COUNT, resources.workerThreads.data()));
    resources.aicpuThread = resources.workerThreads[0];
    CHK_RET(OpenPeerPaths(comm, param, serverRanks, resources));
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op,
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
    CHK_PRT_RET(recvCount > std::numeric_limits<uint64_t>::max()
                                / (yga_reduce_scatter::FP32_BYTES * yga_reduce_scatter::RANK_COUNT),
        HCCL_ERROR("recvCount[%llu] overflows the input extent", static_cast<unsigned long long>(recvCount)),
        HCCL_E_PARA);
    if (recvCount == 0) {
        return HCCL_SUCCESS;
    }

    OpParam param;
    const uint64_t outputBytes = recvCount * yga_reduce_scatter::FP32_BYTES;
    const char *resourceTag = outputBytes == yga_reduce_scatter::HALF_MEG_SLICE_BYTES
                                  ? "hccl_yga_rs_512_owner_tree_v7"
                                  : "hccl_yga_rs_lane_fusion_v3";
    const int tagLength = snprintf(param.tag, sizeof(param.tag), "%s", resourceTag);
    CHK_PRT_RET(tagLength < 0 || static_cast<uint32_t>(tagLength) >= sizeof(param.tag),
        HCCL_ERROR("Failed to create operation tag"), HCCL_E_INTERNAL);
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize != yga_reduce_scatter::RANK_COUNT,
        HCCL_ERROR("Expected rank size 16, got[%u]", param.rankSize), HCCL_E_PARA);

    const CommEngine aicpuEngine = CommEngine::COMM_ENGINE_AICPU_TS;
    const CommEngine hostEngine = CommEngine::COMM_ENGINE_CPU_TS;
    CHK_RET(HcclThreadAcquireWithStream(comm, hostEngine, stream, 1, &param.cpuThread));
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &param.cpuThread, aicpuEngine, &param.cpuThreadOnAicpu));

    void *deviceContext = nullptr;
    uint64_t deviceContextSize = 0;
    if (HcclEngineCtxGet(comm, param.tag, aicpuEngine, &deviceContext, &deviceContextSize) == HCCL_SUCCESS) {
        param.resCtx = deviceContext;
        param.ctxSize = deviceContextSize;

        void *hostContext = nullptr;
        uint64_t hostContextSize = 0;
        CHK_RET(HcclEngineCtxGet(comm, param.tag, hostEngine, &hostContext, &hostContextSize));
        CHK_PRT_RET(hostContextSize != sizeof(ThreadHandle),
            HCCL_ERROR("Unexpected host context size[%llu]", static_cast<unsigned long long>(hostContextSize)),
            HCCL_E_INTERNAL);
        ThreadHandle *deviceThread = static_cast<ThreadHandle *>(hostContext);
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, deviceThread, hostEngine, &param.aicpuThreadOnCpu));
    } else {
        AlgResourceCtx resources;
        CHK_RET(CreateAlgorithmResources(comm, param, resources));
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resources.aicpuThread, hostEngine, &param.aicpuThreadOnCpu));

        std::vector<char> encodedResources = resources.Serialize();
        param.ctxSize = encodedResources.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, aicpuEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, aicpuEngine, param.tag, encodedResources.data(), encodedResources.size(), 0));

        void *hostContext = nullptr;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, hostEngine, sizeof(ThreadHandle), &hostContext));
        CHK_RET(HcclEngineCtxCopy(comm, hostEngine, param.tag, &resources.aicpuThread, sizeof(ThreadHandle), 0));
    }

    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}
