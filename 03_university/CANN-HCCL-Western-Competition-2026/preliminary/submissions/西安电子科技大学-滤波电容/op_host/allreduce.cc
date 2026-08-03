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
#include <utility>
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
constexpr uint32_t AR_UNKNOWN_RANK = AR_RANKS;

struct HostResourceCtx {
    ThreadHandle aicpuThread = 0;
    uint64_t cclBufferBytes = 0;
    uint64_t chunkCapacity = 0;
};

HcclResult TraceFailure(uint32_t rank, const char *stage, const char *api, HcclResult ret)
{
    HCCL_ERROR("[AR_JUDGE][rank=%u][stage=%s][api=%s][ret=%d]", rank, stage, api,
        static_cast<int32_t>(ret));
    return ret;
}

#define AR_CHECK(rank, stage, call) \
    do { \
        const HcclResult arRet = (call); \
        if (arRet != HCCL_SUCCESS) { \
            return TraceFailure((rank), (stage), #call, arRet); \
        } \
    } while (0)

bool MatchesRankRange(const uint32_t *ranks, uint32_t rankNum, uint32_t firstRank, uint32_t expectedNum)
{
    if (ranks == nullptr || rankNum != expectedNum) {
        return false;
    }
    std::vector<uint32_t> sortedRanks(ranks, ranks + rankNum);
    std::sort(sortedRanks.begin(), sortedRanks.end());
    for (uint32_t index = 0; index < expectedNum; ++index) {
        if (sortedRanks[index] != firstRank + index) {
            return false;
        }
    }
    return true;
}

HcclResult BuildTopology(HcclComm comm, uint32_t myRank, AlgResourceCtx &resources,
    uint32_t &intraLayer, uint32_t &globalLayer)
{
    uint32_t *layers = nullptr;
    uint32_t layerNum = 0;
    AR_CHECK(myRank, "BUILD_TOPOLOGY", HcclRankGraphGetLayers(comm, &layers, &layerNum));
    if (layers == nullptr || layerNum == 0) {
        return HCCL_E_NOT_SUPPORT;
    }

    const uint32_t localFirstRank = (myRank / AR_RANKS_PER_SERVER) * AR_RANKS_PER_SERVER;
    bool foundIntraLayer = false;
    bool foundGlobalLayer = false;
    for (uint32_t layerIndex = 0; layerIndex < layerNum; ++layerIndex) {
        const uint32_t layer = layers[layerIndex];
        uint32_t *ranks = nullptr;
        uint32_t rankNum = 0;
        AR_CHECK(myRank, "BUILD_TOPOLOGY", HcclRankGraphGetRanksByLayer(comm, layer, &ranks, &rankNum));

        if (!foundIntraLayer && MatchesRankRange(ranks, rankNum, localFirstRank, AR_RANKS_PER_SERVER)) {
            intraLayer = layer;
            foundIntraLayer = true;
        }
        if (!foundGlobalLayer && MatchesRankRange(ranks, rankNum, 0, AR_RANKS)) {
            globalLayer = layer;
            foundGlobalLayer = true;
        }
    }
    if (!foundIntraLayer || !foundGlobalLayer || intraLayer == globalLayer) {
        return HCCL_E_NOT_SUPPORT;
    }

    for (uint32_t rank = 0; rank < AR_RANKS; ++rank) {
        const uint32_t server = rank / AR_RANKS_PER_SERVER;
        const uint32_t lane = rank % AR_RANKS_PER_SERVER;
        resources.serverOfRank[rank] = server;
        resources.laneOfRank[rank] = lane;
        resources.rankOfServerLane[server][lane] = rank;
    }
    resources.leaderRank[0] = 0;
    resources.leaderRank[1] = AR_RANKS_PER_SERVER;
    resources.rootLeader = 0;

    return HCCL_SUCCESS;
}

HcclResult BuildBufferLayout(uint64_t bufferBytes, bool useParallel, AlgResourceCtx &resources)
{
    uint64_t chunkCapacity = 0;
    uint64_t layoutBytes = 0;
    if (useParallel) {
        constexpr uint64_t layoutNumerator = 15;
        constexpr uint64_t layoutDenominator = 8;
        chunkCapacity = ((bufferBytes / layoutNumerator) * layoutDenominator +
            ((bufferBytes % layoutNumerator) * layoutDenominator) / layoutNumerator) &
            ~(AR_STRIPE_ALIGNMENT_BYTES - 1U);
        if (chunkCapacity < AR_STRIPE_ALIGNMENT_BYTES ||
            chunkCapacity > std::numeric_limits<uint64_t>::max() / layoutNumerator) {
            return HCCL_E_NOT_SUPPORT;
        }
        const uint64_t stagingBytes = (chunkCapacity / AR_RANKS_PER_SERVER) *
            AR_PARALLEL_PEER_NUM;
        resources.inputOffset = 0;
        resources.reductionOffset = chunkCapacity;
        resources.outputOffset = 0;
        layoutBytes = chunkCapacity + stagingBytes;
    } else {
        constexpr uint64_t regionCount = 2;
        chunkCapacity = (bufferBytes / regionCount) & ~(AR_STRIPE_ALIGNMENT_BYTES - 1U);
        if (chunkCapacity < AR_FLOAT_BYTES ||
            chunkCapacity > std::numeric_limits<uint64_t>::max() / regionCount) {
            return HCCL_E_NOT_SUPPORT;
        }
        resources.inputOffset = 0;
        resources.reductionOffset = 0;
        resources.outputOffset = chunkCapacity;
        layoutBytes = chunkCapacity * regionCount;
    }
    if (layoutBytes > bufferBytes) {
        return HCCL_E_INTERNAL;
    }

    resources.cclBufferBytes = bufferBytes;
    resources.chunkCapacity = chunkCapacity;
    return HCCL_SUCCESS;
}

HcclResult AcquireChannelDesc(HcclComm comm, uint32_t layer, uint32_t srcRank, uint32_t dstRank,
    uint32_t notifyNum, HcclChannelDesc &desc)
{
    CommLink *links = nullptr;
    uint32_t linkNum = 0;
    AR_CHECK(srcRank, "ACQUIRE_CHANNEL", HcclRankGraphGetLinks(comm, layer, srcRank, dstRank,
        &links, &linkNum));
    if (links == nullptr || linkNum == 0) {
        return HCCL_E_NOT_SUPPORT;
    }

    AR_CHECK(srcRank, "ACQUIRE_CHANNEL", HcclChannelDescInit(&desc, 1));
    const CommLink &link = links[0];
    desc.remoteRank = dstRank;
    desc.notifyNum = notifyNum;
    desc.channelProtocol = link.linkAttr.linkProtocol;
    desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
    desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
    desc.localEndpoint.loc = link.srcEndpointDesc.loc;
    desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
    desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
    desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
    return HCCL_SUCCESS;
}

HcclResult AcquireChannels(HcclComm comm, const OpParam &param, uint32_t intraLayer,
    uint32_t globalLayer, bool useStriped, bool useParallel, AlgResourceCtx &resources)
{
    const uint32_t server = resources.serverOfRank[param.myRank];
    const uint32_t leader = resources.leaderRank[server];
    const uint32_t lane = resources.laneOfRank[param.myRank];
    std::vector<std::pair<uint32_t, uint32_t>> requests;

    if (useParallel) {
        for (uint32_t peerLane = 0; peerLane < AR_RANKS_PER_SERVER; ++peerLane) {
            if (peerLane != lane) {
                requests.emplace_back(resources.rankOfServerLane[server][peerLane], intraLayer);
            }
        }
        requests.emplace_back(resources.rankOfServerLane[server ^ 1U][lane], globalLayer);
    } else if (useStriped) {
        for (uint32_t mask = 1; mask < AR_RANKS_PER_SERVER; mask <<= 1U) {
            const uint32_t peerLane = lane ^ mask;
            requests.emplace_back(resources.rankOfServerLane[server][peerLane], intraLayer);
        }
        requests.emplace_back(resources.rankOfServerLane[server ^ 1U][lane], globalLayer);
    } else {
        if (lane != 0) {
            const uint32_t parentLane = lane - (lane & (~lane + 1U));
            requests.emplace_back(resources.rankOfServerLane[server][parentLane], intraLayer);
        }
        for (uint32_t step = 1; step < AR_RANKS_PER_SERVER; step <<= 1U) {
            const uint32_t childLane = lane + step;
            if (lane % (step << 1U) == 0 && childLane < AR_RANKS_PER_SERVER) {
                requests.emplace_back(resources.rankOfServerLane[server][childLane], intraLayer);
            }
        }
        if (param.myRank == leader) {
            requests.emplace_back(resources.leaderRank[server ^ 1U], globalLayer);
        }
    }

    std::vector<HcclChannelDesc> descriptions(requests.size());
    std::vector<ChannelHandle> handles(requests.size());
    const uint32_t channelNotifyNum = useParallel ? AR_PARALLEL_CHANNEL_NOTIFY_NUM :
        AR_CHANNEL_NOTIFY_NUM;
    for (size_t index = 0; index < requests.size(); ++index) {
        HcclResult ret = AcquireChannelDesc(comm, requests[index].second, param.myRank,
            requests[index].first, channelNotifyNum, descriptions[index]);
        if (ret != HCCL_SUCCESS) {
            return TraceFailure(param.myRank, "ACQUIRE_CHANNEL", "AcquireChannelDesc", ret);
        }
    }

    AR_CHECK(param.myRank, "ACQUIRE_CHANNEL", HcclChannelAcquire(comm, COMM_ENGINE_AICPU_TS,
        descriptions.data(), static_cast<uint32_t>(descriptions.size()), handles.data()));

    uint64_t safeBufferBytes = resources.localBuffer.size;
    for (size_t index = 0; index < requests.size(); ++index) {
        void *remoteBuffer = nullptr;
        uint64_t remoteBufferSize = 0;
        AR_CHECK(param.myRank, "ACQUIRE_CHANNEL", HcclChannelGetHcclBuffer(comm, handles[index],
            &remoteBuffer, &remoteBufferSize));
        if (remoteBuffer == nullptr) {
            return HCCL_E_NOT_SUPPORT;
        }
        safeBufferBytes = std::min(safeBufferBytes, remoteBufferSize);

        ChannelInfo channel{};
        channel.remoteRank = requests[index].first;
        channel.notifyNum = channelNotifyNum;
        channel.handle = handles[index];
        channel.remoteCclMem = CommBuffer{remoteBuffer, remoteBufferSize};
        resources.channels.push_back(channel);
    }

    AR_CHECK(param.myRank, "ACQUIRE_CHANNEL", BuildBufferLayout(safeBufferBytes, useParallel,
        resources));
    const uint64_t requiredRemoteBytes = useParallel ?
        resources.reductionOffset + (resources.chunkCapacity / AR_RANKS_PER_SERVER) *
        AR_PARALLEL_PEER_NUM : resources.outputOffset + resources.chunkCapacity;
    for (const ChannelInfo &channel : resources.channels) {
        if (channel.remoteCclMem.size < requiredRemoteBytes) {
            return HCCL_E_NOT_SUPPORT;
        }
    }

    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    if (count == 0) {
        return HCCL_SUCCESS;
    }
    if (dataType != HCCL_DATA_TYPE_FP32 || op != HCCL_REDUCE_SUM ||
        count > std::numeric_limits<uint64_t>::max() / AR_FLOAT_BYTES) {
        return HCCL_E_NOT_SUPPORT;
    }
    const uint64_t totalDataBytes = count * AR_FLOAT_BYTES;
    const bool useStriped = totalDataBytes >= AR_STRIPED_THRESHOLD_BYTES;
    const bool useParallel = totalDataBytes >= AR_PARALLEL_THRESHOLD_BYTES;

    OpParam param{};
    std::snprintf(param.tag, sizeof(param.tag), "%s", useParallel ?
        "hccl_custom_allreduce_fullmesh_v5" : (useStriped ?
        "hccl_custom_allreduce_striped_v2" : "hccl_custom_allreduce_binomial_tree_v2"));
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = count;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;

    AR_CHECK(AR_UNKNOWN_RANK, "ENTRY", HcclGetRankId(comm, &param.myRank));
    AR_CHECK(param.myRank, "ENTRY", HcclGetRankSize(comm, &param.rankSize));
    if (param.rankSize != AR_RANKS || param.myRank >= AR_RANKS) {
        return HCCL_E_NOT_SUPPORT;
    }
    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
    AR_CHECK(param.myRank, "REGISTER", HcclGetCommName(comm, commName));
    AR_CHECK(param.myRank, "REGISTER", HcclDfxRegOpInfoByCommId(commName,
        reinterpret_cast<void *>(&dfxInfo)));

    constexpr CommEngine aicpuTsEngine = CommEngine::COMM_ENGINE_AICPU_TS;
    constexpr CommEngine cpuTsEngine = CommEngine::COMM_ENGINE_CPU_TS;
    AR_CHECK(param.myRank, "ACQUIRE_THREAD", HcclThreadAcquireWithStream(comm, cpuTsEngine, stream, 1,
        &param.cpuThread));
    AR_CHECK(param.myRank, "ACQUIRE_THREAD", HcclThreadExportToCommEngine(comm, 1, &param.cpuThread,
        aicpuTsEngine, &param.cpuThreadOnAicpu));

    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    uint64_t chunkCapacity = 0;
    if (HcclEngineCtxGet(comm, param.tag, aicpuTsEngine, &ctx, &ctxSize) == HCCL_SUCCESS) {
        if (ctx == nullptr || ctxSize == 0) {
            return HCCL_E_INTERNAL;
        }
        param.resCtx = ctx;
        param.ctxSize = ctxSize;

        void *hostCtx = nullptr;
        uint64_t hostCtxSize = 0;
        AR_CHECK(param.myRank, "ACQUIRE_THREAD", HcclEngineCtxGet(comm, param.tag, cpuTsEngine,
            &hostCtx, &hostCtxSize));
        if (hostCtx == nullptr || hostCtxSize != sizeof(HostResourceCtx)) {
            return HCCL_E_INTERNAL;
        }
        HostResourceCtx *hostResources = static_cast<HostResourceCtx *>(hostCtx);
        AR_CHECK(param.myRank, "ACQUIRE_THREAD", HcclThreadExportToCommEngine(comm, 1,
            &hostResources->aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));
        chunkCapacity = hostResources->chunkCapacity;
    } else {
        AlgResourceCtx resourcePlan{};
        uint32_t intraLayer = 0;
        uint32_t globalLayer = 0;
        AR_CHECK(param.myRank, "BUILD_TOPOLOGY", BuildTopology(comm, param.myRank, resourcePlan,
            intraLayer, globalLayer));

        void *cclBufferAddr = nullptr;
        uint64_t cclBufferSize = 0;
        AR_CHECK(param.myRank, "GET_CCL_BUFFER",
            HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        if (cclBufferAddr == nullptr) {
            return HCCL_E_NOT_SUPPORT;
        }
        resourcePlan.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        AlgResourceCtx resources = resourcePlan;
        const uint32_t threadNum = useParallel ? AR_PARALLEL_THREAD_NUM : 1U;
        const uint32_t notifyNum = useParallel ? AR_THREAD_NOTIFY_NUM : 1U;
        resources.threads.resize(threadNum);
        AR_CHECK(param.myRank, "ACQUIRE_THREAD", HcclThreadAcquire(comm, aicpuTsEngine, threadNum,
            notifyNum, resources.threads.data()));
        resources.aicpuThread = resources.threads[0];
        AR_CHECK(param.myRank, "ACQUIRE_THREAD", HcclThreadExportToCommEngine(comm, 1,
            &resources.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));
        AR_CHECK(param.myRank, "ACQUIRE_CHANNEL", AcquireChannels(comm, param, intraLayer,
            globalLayer, useStriped, useParallel, resources));
        chunkCapacity = resources.chunkCapacity;

        const std::vector<char> sequence = resources.Serialize();
        if (sequence.empty()) {
            return HCCL_E_INTERNAL;
        }
        param.ctxSize = sequence.size();
        AR_CHECK(param.myRank, "CREATE_CONTEXT", HcclEngineCtxCreate(comm, param.tag, aicpuTsEngine,
            param.ctxSize, &param.resCtx));
        AR_CHECK(param.myRank, "CREATE_CONTEXT", HcclEngineCtxCopy(comm, aicpuTsEngine, param.tag,
            sequence.data(), sequence.size(), 0));

        void *hostCtx = nullptr;
        const HostResourceCtx hostResources{resources.aicpuThread, resources.cclBufferBytes,
            resources.chunkCapacity};
        AR_CHECK(param.myRank, "CREATE_CONTEXT", HcclEngineCtxCreate(comm, param.tag, cpuTsEngine,
            sizeof(HostResourceCtx), &hostCtx));
        AR_CHECK(param.myRank, "CREATE_CONTEXT", HcclEngineCtxCopy(comm, cpuTsEngine, param.tag,
            &hostResources, sizeof(HostResourceCtx), 0));
    }

    if (chunkCapacity < AR_FLOAT_BYTES) {
        return HCCL_E_INTERNAL;
    }
    AR_CHECK(param.myRank, "KERNEL_LAUNCH", ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}
