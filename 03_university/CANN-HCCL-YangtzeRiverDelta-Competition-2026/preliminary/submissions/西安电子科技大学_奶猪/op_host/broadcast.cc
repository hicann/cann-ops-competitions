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
constexpr uint32_t CHANNEL_NOTIFY_NUM = 5;
constexpr uint32_t THREAD_NOTIFY_NUM = custom_broadcast::PEER_SIZE;
constexpr uint64_t MIN_CCL_BUFFER_SIZE = custom_broadcast::DATA_TYPE_SIZE;

HcclResult InitChannelDesc(
    HcclComm comm, uint32_t srcRank, uint32_t dstRank, uint32_t netLayer, HcclChannelDesc *desc)
{
    CommLink *links = nullptr;
    uint32_t linkNum = 0;
    CHK_RET(HcclRankGraphGetLinks(comm, netLayer, srcRank, dstRank, &links, &linkNum));
    CHK_PRT_RET(linkNum == 0,
        HCCL_ERROR("No link between rank[%u] and rank[%u] on layer[%u]", srcRank, dstRank, netLayer),
        HCCL_E_NOT_FOUND);
    CHK_PTR_NULL(links);

    CHK_RET(HcclChannelDescInit(desc, 1));
    for (uint32_t i = 0; i < linkNum; ++i) {
        const CommLink &link = links[i];
        if (link.linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_UBC_CTP) {
            continue;
        }
        desc->remoteRank = dstRank;
        desc->notifyNum = CHANNEL_NOTIFY_NUM;
        desc->channelProtocol = link.linkAttr.linkProtocol;
        desc->localEndpoint.protocol = link.srcEndpointDesc.protocol;
        desc->localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
        desc->localEndpoint.loc = link.srcEndpointDesc.loc;
        desc->remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
        desc->remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
        desc->remoteEndpoint.loc = link.dstEndpointDesc.loc;
        return HCCL_SUCCESS;
    }

    HCCL_ERROR("UBC_CTP link not found between rank[%u] and rank[%u] on layer[%u]", srcRank, dstRank, netLayer);
    return HCCL_E_NOT_FOUND;
}

HcclResult ParseTopology(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtx,
    std::vector<uint32_t> &localRanks)
{
    uint32_t *localRankData = nullptr;
    uint32_t localRankNum = 0;
    CHK_RET(HcclRankGraphGetRanksByLayer(
        comm, custom_broadcast::INTRA_NET_LAYER, &localRankData, &localRankNum));
    CHK_PRT_RET(localRankNum != custom_broadcast::LOCAL_RANK_SIZE,
        HCCL_ERROR("Unexpected layer-0 rank size[%u]", localRankNum), HCCL_E_PARA);
    CHK_PTR_NULL(localRankData);
    localRanks.assign(localRankData, localRankData + localRankNum);
    std::sort(localRanks.begin(), localRanks.end());
    CHK_PRT_RET(std::find(localRanks.begin(), localRanks.end(), param.myRank) == localRanks.end(),
        HCCL_ERROR("Rank[%u] is absent from its layer-0 topology", param.myRank), HCCL_E_INTERNAL);

    uint32_t *allRankData = nullptr;
    uint32_t allRankNum = 0;
    CHK_RET(HcclRankGraphGetRanksByLayer(
        comm, custom_broadcast::INTER_NET_LAYER, &allRankData, &allRankNum));
    CHK_PRT_RET(allRankNum != custom_broadcast::RANK_SIZE,
        HCCL_ERROR("Unexpected layer-1 rank size[%u]", allRankNum), HCCL_E_PARA);
    CHK_PTR_NULL(allRankData);
    resCtx.ranks.assign(allRankData, allRankData + allRankNum);
    std::sort(resCtx.ranks.begin(), resCtx.ranks.end());

    auto rankIt = std::find(resCtx.ranks.begin(), resCtx.ranks.end(), param.myRank);
    CHK_PRT_RET(rankIt == resCtx.ranks.end(),
        HCCL_ERROR("Rank[%u] is absent from layer-1 topology", param.myRank), HCCL_E_INTERNAL);
    resCtx.rankIndex = static_cast<uint32_t>(std::distance(resCtx.ranks.begin(), rankIt));
    return HCCL_SUCCESS;
}

HcclResult AcquireChannels(HcclComm comm, const OpParam &param, const std::vector<uint32_t> &localRanks,
    AlgResourceCtx &resCtx)
{
    std::vector<uint32_t> remoteRanks;
    std::vector<uint32_t> remoteIndices;
    std::vector<uint32_t> netLayers;
    remoteRanks.reserve(custom_broadcast::PEER_SIZE);
    remoteIndices.reserve(custom_broadcast::PEER_SIZE);
    netLayers.reserve(custom_broadcast::PEER_SIZE);

    uint32_t intraPeerNum = 0;
    uint32_t interPeerNum = 0;
    for (uint32_t rankIndex = 0; rankIndex < resCtx.ranks.size(); ++rankIndex) {
        const uint32_t rank = resCtx.ranks[rankIndex];
        if (rank == param.myRank) {
            continue;
        }
        const bool isLocal = std::find(localRanks.begin(), localRanks.end(), rank) != localRanks.end();
        remoteRanks.push_back(rank);
        remoteIndices.push_back(rankIndex);
        netLayers.push_back(isLocal ? custom_broadcast::INTRA_NET_LAYER : custom_broadcast::INTER_NET_LAYER);
        intraPeerNum += static_cast<uint32_t>(isLocal);
        interPeerNum += static_cast<uint32_t>(!isLocal);
    }
    CHK_PRT_RET(intraPeerNum != custom_broadcast::LOCAL_RANK_SIZE - 1 ||
            interPeerNum != custom_broadcast::LOCAL_RANK_SIZE,
        HCCL_ERROR("Unexpected peer split: intra[%u], inter[%u]", intraPeerNum, interPeerNum), HCCL_E_PARA);

    std::vector<HcclChannelDesc> descs(remoteRanks.size());
    for (uint32_t i = 0; i < remoteRanks.size(); ++i) {
        CHK_RET(InitChannelDesc(comm, param.myRank, remoteRanks[i], netLayers[i], &descs[i]));
    }

    std::vector<ChannelHandle> handles(remoteRanks.size());
    CHK_RET(HcclChannelAcquire(
        comm, CommEngine::COMM_ENGINE_AICPU_TS, descs.data(), descs.size(), handles.data()));

    resCtx.channels.resize(remoteRanks.size());
    for (uint32_t i = 0; i < remoteRanks.size(); ++i) {
        ChannelInfo &channel = resCtx.channels[i];
        channel.remoteRank = remoteRanks[i];
        channel.remoteRankIndex = remoteIndices[i];
        channel.netLayer = netLayers[i];
        channel.notifyNum = CHANNEL_NOTIFY_NUM;
        channel.handle = handles[i];
        void *remoteBuffer = nullptr;
        uint64_t remoteBufferSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, handles[i], &remoteBuffer, &remoteBufferSize));
        CHK_PTR_NULL(remoteBuffer);
        CHK_PRT_RET(remoteBufferSize < MIN_CCL_BUFFER_SIZE,
            HCCL_ERROR("Remote CCL buffer for rank[%u] is too small", remoteRanks[i]), HCCL_E_MEMORY);
        channel.remoteCclMem = CommBuffer{remoteBuffer, remoteBufferSize};
    }
    return HCCL_SUCCESS;
}

HcclResult BuildResources(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtx)
{
    std::vector<uint32_t> localRanks;
    CHK_RET(ParseTopology(comm, param, resCtx, localRanks));

    void *cclBuffer = nullptr;
    uint64_t cclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBuffer, &cclBufferSize));
    CHK_PTR_NULL(cclBuffer);
    CHK_PRT_RET(cclBufferSize < MIN_CCL_BUFFER_SIZE,
        HCCL_ERROR("Local CCL buffer size[%llu] is too small", static_cast<unsigned long long>(cclBufferSize)),
        HCCL_E_MEMORY);
    resCtx.localBuffer = CommBuffer{cclBuffer, cclBufferSize};

    resCtx.threads.resize(custom_broadcast::PEER_SIZE);
    CHK_RET(HcclThreadAcquire(comm, CommEngine::COMM_ENGINE_AICPU_TS, resCtx.threads.size(), THREAD_NOTIFY_NUM,
        resCtx.threads.data()));
    resCtx.aicpuThread = resCtx.threads[0];
    CHK_RET(AcquireChannels(comm, param, localRanks, resCtx));
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclBroadcast(
    void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(buf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("Unsupported data type[%d], only FP32 is accepted", static_cast<int32_t>(dataType)), HCCL_E_PARA);
    CHK_PRT_RET(count > std::numeric_limits<uint64_t>::max() / custom_broadcast::DATA_TYPE_SIZE,
        HCCL_ERROR("Element count[%llu] overflows byte calculation", static_cast<unsigned long long>(count)),
        HCCL_E_PARA);

    OpParam param;
    int tagLen = snprintf(param.tag, sizeof(param.tag), "%s", "hccl_custom_broadcast_stok_v1");
    CHK_PRT_RET(tagLen < 0 || static_cast<uint32_t>(tagLen) >= sizeof(param.tag),
        HCCL_ERROR("Failed to initialize operation tag"), HCCL_E_INTERNAL);
    param.inputPtr = buf;
    param.outputPtr = buf;
    param.count = count;
    param.dataType = dataType;
    param.root = root;
    param.opType = HcclCMDType::HCCL_CMD_BROADCAST;

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize != custom_broadcast::RANK_SIZE,
        HCCL_ERROR("Unexpected rank size[%u]", param.rankSize), HCCL_E_PARA);
    CHK_PRT_RET(root >= param.rankSize,
        HCCL_ERROR("Root rank[%u] is outside rank size[%u]", root, param.rankSize), HCCL_E_PARA);
    if (count == 0) {
        return HCCL_SUCCESS;
    }

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
        CHK_PRT_RET(hostCtxSize != sizeof(ThreadHandle),
            HCCL_ERROR("Unexpected host context size[%llu]", static_cast<unsigned long long>(hostCtxSize)),
            HCCL_E_INTERNAL);
        ThreadHandle *aicpuThread = static_cast<ThreadHandle *>(hostCtx);
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));
    } else {
        AlgResourceCtx resCtxHost;
        CHK_RET(BuildResources(comm, param, resCtxHost));
        CHK_RET(HcclThreadExportToCommEngine(
            comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        std::vector<char> seq = resCtxHost.Serialize();
        param.ctxSize = seq.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, aicpuTsEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, aicpuTsEngine, param.tag, seq.data(), seq.size(), 0));

        void *hostCtx = nullptr;
        const uint64_t hostCtxSize = sizeof(ThreadHandle);
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, cpuTsEngine, hostCtxSize, &hostCtx));
        CHK_RET(HcclEngineCtxCopy(
            comm, cpuTsEngine, param.tag, &resCtxHost.aicpuThread, hostCtxSize, 0));
    }

    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}
