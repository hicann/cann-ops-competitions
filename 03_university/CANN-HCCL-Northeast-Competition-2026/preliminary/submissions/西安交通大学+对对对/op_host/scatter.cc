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
constexpr char BASE_TAG[] = "hccl_scatter_v019_base";
constexpr char TOPOLOGY_TAG[] = "hccl_scatter_v019_topology";

template <typename Protocol>
bool IsScatterProtocol(Protocol protocol)
{
    return protocol == COMM_PROTOCOL_UBC_CTP || protocol == COMM_PROTOCOL_UBOE || protocol == COMM_PROTOCOL_ROCE;
}

HcclResult MakeScatterChannelDesc(HcclComm comm, uint32_t myRank, uint32_t peer, HcclChannelDesc &desc)
{
    uint32_t *layers = nullptr;
    uint32_t layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layers, &layerCount));
    for (uint32_t layer = 0; layer < layerCount; ++layer) {
        CommLink *links = nullptr;
        uint32_t linkCount = 0;
        const HcclResult result = HcclRankGraphGetLinks(comm, layers[layer], myRank, peer, &links, &linkCount);
        if (result != HCCL_SUCCESS || links == nullptr) {
            continue;
        }
        for (uint32_t index = 0; index < linkCount; ++index) {
            const CommLink &link = links[index];
            const auto protocol = link.linkAttr.linkProtocol;
            if (!IsScatterProtocol(protocol)) {
                continue;
            }
            HcclChannelDescInit(&desc, 1);
            desc.remoteRank = peer;
            desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
            desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
            desc.localEndpoint.loc = link.srcEndpointDesc.loc;
            desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
            desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
            desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
            desc.channelProtocol = protocol;
            desc.notifyNum = SCATTER_CHANNEL_NOTIFIES;
            return HCCL_SUCCESS;
        }
    }
    HCCL_ERROR("No AICPU_TS link from rank %u to rank %u", myRank, peer);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AcquireScatterChannels(HcclComm comm, const OpParam &param, AlgResourceCtx &ctx)
{
    std::vector<HcclChannelDesc> descs;
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer == param.myRank) {
            continue;
        }
        HcclChannelDesc desc{};
        CHK_RET(MakeScatterChannelDesc(comm, param.myRank, peer, desc));
        descs.push_back(desc);
    }
    if (descs.empty()) {
        return HCCL_SUCCESS;
    }
    const uint32_t count = static_cast<uint32_t>(descs.size());
    std::vector<ChannelHandle> handles(count);
    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AICPU_TS, descs.data(), count, handles.data()));
    ctx.channels.resize(count);
    for (uint32_t index = 0; index < count; ++index) {
        ChannelInfo &channel = ctx.channels[index];
        channel.remoteRank = descs[index].remoteRank;
        channel.notifyNum = descs[index].notifyNum;
        channel.channelProtocol = static_cast<uint32_t>(descs[index].channelProtocol);
        channel.handle = handles[index];
        CHK_RET(HcclChannelGetHcclBuffer(comm, channel.handle, &channel.remoteCclMem.addr, &channel.remoteCclMem.size));
        CHK_PRT_RET(channel.remoteCclMem.addr == nullptr || channel.remoteCclMem.size < sizeof(float),
            HCCL_ERROR("Empty Scatter communication buffer for rank %u", channel.remoteRank), HCCL_E_PARA);
    }
    return HCCL_SUCCESS;
}

HcclResult StoreContext(HcclComm comm, const char *tag, CommEngine engine,
    const AlgResourceCtx &ctx, void **address, uint64_t &size)
{
    std::vector<char> sequence = ctx.Serialize();
    size = sequence.size();
    CHK_RET(HcclEngineCtxCreate(comm, tag, engine, size, address));
    CHK_RET(HcclEngineCtxCopy(comm, engine, tag, sequence.data(), size, 0));
    return HCCL_SUCCESS;
}

HcclResult GetBaseContext(HcclComm comm, const OpParam &param, AlgResourceCtx &base)
{
    void *address = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, BASE_TAG, CommEngine::COMM_ENGINE_CPU_TS, &address, &size) == HCCL_SUCCESS) {
        CHK_PRT_RET(address == nullptr || size == 0,
            HCCL_ERROR("Incomplete cached Scatter base context"), HCCL_E_INTERNAL);
        CHK_PRT_RET(!base.DeSerialize(address, size),
            HCCL_ERROR("Invalid cached Scatter base context encoding"), HCCL_E_INTERNAL);
        CHK_PRT_RET(base.threads.size() != 1 || base.channels.size() != param.rankSize - 1,
            HCCL_ERROR("Invalid cached Scatter base topology"), HCCL_E_INTERNAL);
        return HCCL_SUCCESS;
    }

    CHK_RET(HcclGetHcclBuffer(comm, &base.localBuffer.addr, &base.localBuffer.size));
    const uint32_t workers = std::min(SCATTER_MAX_WORKERS, param.rankSize - 1);
    base.threads.resize(1);
    // Base stores only main; algorithm plans acquire their role-specific workers.
    // Reserve joins now. Index 0 is exclusively the Host -> Device handshake.
    CHK_RET(HcclThreadAcquire(comm, CommEngine::COMM_ENGINE_AICPU_TS, 1, workers + 1, base.threads.data()));
    base.aicpuThread = base.threads[0];
    // Acquire exactly one channel per peer once for the entire comm. Never
    // reacquire/export channels when the message size or root changes.
    CHK_RET(AcquireScatterChannels(comm, param, base));
    CHK_RET(StoreContext(comm, BASE_TAG, CommEngine::COMM_ENGINE_CPU_TS, base, &address, size));
    return HCCL_SUCCESS;
}

HcclResult DiscoverServerGroups(HcclComm comm, const OpParam &param, std::vector<uint32_t> &groups)
{
    groups.clear();
    uint32_t *layers = nullptr;
    uint32_t layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layers, &layerCount));
    CHK_PRT_RET(layerCount != 0 && layers == nullptr,
        HCCL_ERROR("Scatter topology returned an invalid layer list"), HCCL_E_INTERNAL);
    // Copy borrowed API arrays before making another topology query.
    std::vector<uint32_t> layerIds;
    if (layerCount != 0) {
        layerIds.assign(layers, layers + layerCount);
    }
    std::sort(layerIds.begin(), layerIds.end());
    for (uint32_t layer : layerIds) {
        uint32_t *ranks = nullptr;
        uint32_t rankCount = 0;
        CHK_RET(HcclRankGraphGetRanksByLayer(comm, layer, &ranks, &rankCount));
        if (rankCount != SCATTER_SERVER_RANKS) {
            continue;
        }
        CHK_PTR_NULL(ranks);
        std::vector<uint32_t> localRanks(ranks, ranks + rankCount);
        std::sort(localRanks.begin(), localRanks.end());
        CHK_PRT_RET(localRanks.back() >= param.rankSize ||
            std::adjacent_find(localRanks.begin(), localRanks.end()) != localRanks.end() ||
            !std::binary_search(localRanks.begin(), localRanks.end(), param.myRank),
            HCCL_ERROR("Scatter topology returned an invalid local rank instance"), HCCL_E_INTERNAL);
        // On the contest's two-server topology this API returns only our
        // server's eight members. The other server is their complement.
        // Do not query links between two other ranks: that pair need not be
        // visible in this process's rank graph. Our own channels were already
        // acquired for every peer using myRank -> peer in GetBaseContext.
        std::vector<uint32_t> candidate(param.rankSize, 1);
        for (uint32_t rank : localRanks) {
            candidate[rank] = 0;
        }
        // Canonical group zero always contains rank zero, irrespective of
        // rank numbering, this process's server, or the current scatter root.
        if (candidate[0] != 0) {
            for (uint32_t &group : candidate) {
                group = 1 - group;
            }
        }
        // Equivalent instances/layer orders yield the same canonical plan.
        // A local ambiguity must not silently select DIRECT while peers use
        // HYBRID. This is an error, not a collective fallback protocol.
        CHK_PRT_RET(!groups.empty() && groups != candidate,
            HCCL_ERROR("Ambiguous Scatter eight-rank server instances"), HCCL_E_NOT_SUPPORT);
        groups = candidate;
    }
    CHK_PRT_RET(groups.empty(),
        HCCL_ERROR("Scatter hybrid requires a local eight-rank server instance"), HCCL_E_NOT_SUPPORT);
    return HCCL_SUCCESS;
}

HcclResult GetServerGroups(HcclComm comm, const OpParam &param, std::vector<uint32_t> &groups)
{
    if (param.rankSize != SCATTER_SERVER_RANKS + SCATTER_REMOTE_COUNT) {
        return HCCL_SUCCESS;
    }
    void *address = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, TOPOLOGY_TAG, CommEngine::COMM_ENGINE_CPU_TS, &address, &size) == HCCL_SUCCESS) {
        CHK_PRT_RET(address == nullptr || size < sizeof(size_t),
            HCCL_ERROR("Incomplete cached Scatter server topology"), HCCL_E_INTERNAL);
        ScatterContextReader reader(address, size);
        CHK_PRT_RET(!reader.Read(groups),
            HCCL_ERROR("Invalid cached Scatter server topology encoding"), HCCL_E_INTERNAL);
        CHK_PRT_RET(groups.size() != param.rankSize || groups[0] != 0 ||
            std::count(groups.begin(), groups.end(), 0U) != SCATTER_SERVER_RANKS ||
            std::count(groups.begin(), groups.end(), 1U) != SCATTER_REMOTE_COUNT ||
            size != sizeof(size_t) + groups.size() * sizeof(uint32_t),
            HCCL_ERROR("Invalid cached Scatter server topology"), HCCL_E_INTERNAL);
        return HCCL_SUCCESS;
    }
    CHK_RET(DiscoverServerGroups(comm, param, groups));
    BinaryStream stream;
    // Cache only a successfully identified partition for the target 2x8 graph.
    stream << groups;
    std::vector<char> sequence;
    stream.Dump(sequence);
    CHK_RET(HcclEngineCtxCreate(comm, TOPOLOGY_TAG, CommEngine::COMM_ENGINE_CPU_TS, sequence.size(), &address));
    CHK_RET(HcclEngineCtxCopy(comm, CommEngine::COMM_ENGINE_CPU_TS, TOPOLOGY_TAG,
        sequence.data(), sequence.size(), 0));
    return HCCL_SUCCESS;
}

HcclResult FindScatterBuffer(const OpParam &param, const AlgResourceCtx &ctx, uint32_t rank, CommBuffer &buffer)
{
    if (rank == param.myRank) {
        buffer = ctx.localBuffer;
        return HCCL_SUCCESS;
    }
    for (const auto &channel : ctx.channels) {
        if (channel.remoteRank == rank) {
            buffer = channel.remoteCclMem;
            return HCCL_SUCCESS;
        }
    }
    HCCL_ERROR("Missing Scatter buffer for rank %u", rank);
    return HCCL_E_INTERNAL;
}

HcclResult SelectSmallTree(HcclComm comm, const OpParam &param, AlgResourceCtx &ctx)
{
    if (param.rankSize != SCATTER_SERVER_RANKS * 2) {
        return HCCL_SUCCESS;
    }
    std::vector<uint32_t> groups;
    CHK_RET(GetServerGroups(comm, param, groups));
    std::vector<uint32_t> ranks;
    ranks.reserve(param.rankSize);
    for (uint32_t server = 0; server < 2; ++server) {
        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            if (groups[rank] == server) {
                ranks.push_back(rank);
            }
        }
    }
    const uint64_t groupBytes = param.count * sizeof(float) * SCATTER_SMALL_GROUP_RANKS;
    for (uint32_t group = 0; group < SCATTER_SMALL_WORKERS; ++group) {
        const uint32_t leader = ScatterSmallGroupLeader(ranks, group, param.root);
        if (leader == param.root) {
            continue; // Complete root input capacity was checked by SelectAlgorithm.
        }
        CommBuffer buffer;
        CHK_RET(FindScatterBuffer(param, ctx, leader, buffer));
        CHK_PTR_NULL(buffer.addr);
        if (buffer.size < groupBytes) {
            // Every rank checks the same three leaders via its all-peer table.
            // Keep the original fan-out collectively, before enqueuing any work.
            return HCCL_SUCCESS;
        }
    }
    ctx.smallMode = SCATTER_SMALL_TREE;
    ctx.smallRanks = ranks;
    return HCCL_SUCCESS;
}

HcclResult SelectAlgorithm(HcclComm comm, const OpParam &param, AlgResourceCtx &ctx)
{
    if (param.rankSize == 1) {
        return HCCL_SUCCESS;
    }
    CommBuffer rootBuffer = ctx.localBuffer;
    if (param.myRank != param.root) {
        bool found = false;
        for (const auto &channel : ctx.channels) {
            if (channel.remoteRank == param.root) {
                rootBuffer = channel.remoteCclMem;
                found = true;
                break;
            }
        }
        CHK_PRT_RET(!found, HCCL_ERROR("Missing Scatter root channel"), HCCL_E_INTERNAL);
    }
    CHK_PTR_NULL(rootBuffer.addr);
    const uint64_t inputBytes = param.count * sizeof(float) * param.rankSize;
    if (inputBytes <= SCATTER_SMALL_INPUT_BYTES && inputBytes <= rootBuffer.size) {
        ctx.chunkBytes = 0;
        ctx.workerCount = ScatterSmallRootWorkerCount(param.rankSize, param.myRank, param.root);
        return SelectSmallTree(comm, param, ctx);
    }
    ctx.workerCount = std::min(SCATTER_MAX_WORKERS, param.rankSize - 1);
    ctx.chunkBytes = std::min(SCATTER_CHUNK_BYTES, rootBuffer.size / (2 * ctx.workerCount));
    ctx.chunkBytes = ctx.chunkBytes / sizeof(float) * sizeof(float);
    CHK_PRT_RET(ctx.chunkBytes == 0, HCCL_ERROR("Scatter root buffer cannot hold worker slots"), HCCL_E_PARA);
    if (param.count < 4) {
        return HCCL_SUCCESS;
    }
    std::vector<uint32_t> groups;
    // The same cached server partition is also used by the small tree.
    CHK_RET(GetServerGroups(comm, param, groups));
    if (groups.empty()) {
        return HCCL_SUCCESS;
    }
    std::vector<uint32_t> relayRanks;
    std::vector<uint32_t> remoteRanks;
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (groups[rank] != groups[param.root]) {
            remoteRanks.push_back(rank);
        } else if (rank != param.root && relayRanks.size() < SCATTER_RELAY_COUNT) {
            relayRanks.push_back(rank);
        }
    }
    CHK_PRT_RET(relayRanks.size() != SCATTER_RELAY_COUNT || remoteRanks.size() != SCATTER_REMOTE_COUNT,
        HCCL_ERROR("Invalid Scatter hybrid rank partition"), HCCL_E_INTERNAL);
    uint64_t hybridChunkBytes = ctx.chunkBytes;
    for (uint32_t relayIndex = 0; relayIndex < relayRanks.size(); ++relayIndex) {
        CommBuffer relayBuffer;
        CHK_RET(FindScatterBuffer(param, ctx, relayRanks[relayIndex], relayBuffer));
        CHK_PTR_NULL(relayBuffer.addr);
        const uint64_t relaySlots = 2ULL * ScatterRelayChildCount(relayIndex);
        hybridChunkBytes = std::min(hybridChunkBytes, relayBuffer.size / relaySlots);
    }
    hybridChunkBytes = hybridChunkBytes / sizeof(float) * sizeof(float);
    if (hybridChunkBytes == 0) {
        return HCCL_SUCCESS;
    }
    // Every rank observes the same root and seven relay capacities through its
    // all-peer channel table; no decision uses only this rank's local buffer.
    ctx.chunkBytes = hybridChunkBytes;
    ctx.routeMode = SCATTER_ROUTE_HYBRID;
    ctx.routeRoot = param.root;
    ctx.relayRanks = relayRanks;
    ctx.remoteRanks = remoteRanks;
    // Retain the original hybrid capacity as an upper bound. Push adds two
    // incoming slots at every peer; relay downstream slots remain separate.
    // All ranks use every peer's visible capacity, never a local protocol
    // decision, so the route and slot stride agree across the communicator.
    uint64_t pushChunkBytes = hybridChunkBytes;
    for (uint32_t relayIndex = 0; relayIndex < relayRanks.size(); ++relayIndex) {
        CommBuffer relayBuffer;
        CHK_RET(FindScatterBuffer(param, ctx, relayRanks[relayIndex], relayBuffer));
        CHK_PTR_NULL(relayBuffer.addr);
        const uint64_t relaySlots = ScatterRelayBufferSlots(SCATTER_ROUTE_HYBRID_PUSH, relayIndex);
        pushChunkBytes = std::min(pushChunkBytes, relayBuffer.size / relaySlots);
    }
    for (uint32_t remoteRank : remoteRanks) {
        CommBuffer remoteBuffer;
        CHK_RET(FindScatterBuffer(param, ctx, remoteRank, remoteBuffer));
        CHK_PTR_NULL(remoteBuffer.addr);
        pushChunkBytes = std::min(pushChunkBytes, remoteBuffer.size / 2);
    }
    pushChunkBytes = pushChunkBytes / sizeof(float) * sizeof(float);
    if (pushChunkBytes != 0) {
        ctx.chunkBytes = pushChunkBytes;
        ctx.routeMode = SCATTER_ROUTE_HYBRID_PUSH;
    }
    return HCCL_SUCCESS;
}

HcclResult GetAlgorithmContext(HcclComm comm, OpParam &param, AlgResourceCtx &ctx)
{
    const bool largeRoot = ctx.chunkBytes != 0 && param.myRank == param.root;
    const bool parallelSmallRoot = ctx.chunkBytes == 0 && param.myRank == param.root && ctx.workerCount != 0;
    const bool hybrid = ScatterIsHybridRoute(ctx.routeMode);
    const bool hybridRemote = hybrid &&
        std::binary_search(ctx.remoteRanks.begin(), ctx.remoteRanks.end(), param.myRank);
    const bool hybridRelay = hybrid &&
        std::binary_search(ctx.relayRanks.begin(), ctx.relayRanks.end(), param.myRank);
    char storageTag[TAG_LENGTH];
    int length;
    if (hybrid) {
        const char *role = largeRoot ? "root" : (hybridRemote ? "remote" : (hybridRelay ? "relay" : "local"));
        length = snprintf(storageTag, sizeof(storageTag), "hccl_scatter_v019_hybrid_%u_%s_%u_%llu",
            ctx.routeMode, role, param.root, static_cast<unsigned long long>(ctx.chunkBytes));
    } else if (ctx.chunkBytes == 0) {
        // Count/root-dependent capacity selection runs on every call. Mode is
        // part of the cache key; ranks are comm-static and leader roles use the
        // current param.root on Device, so no cached pointer/count/root is used.
        const char *role = param.myRank == param.root ? "root" : "recv";
        length = snprintf(storageTag, sizeof(storageTag), "hccl_scatter_v019_small_%u_%s_%u",
            ctx.smallMode, role, ctx.workerCount);
    } else {
        length = snprintf(storageTag, sizeof(storageTag), "hccl_scatter_v019_%s_%llu",
            largeRoot ? "large_root" : "large_recv",
            static_cast<unsigned long long>(ctx.chunkBytes));
    }
    CHK_PRT_RET(length < 0 || static_cast<size_t>(length) >= sizeof(storageTag),
        HCCL_ERROR("Scatter context tag is too long"), HCCL_E_INTERNAL);
    if (HcclEngineCtxGet(comm, storageTag, CommEngine::COMM_ENGINE_AICPU_TS,
        &param.resCtx, &param.ctxSize) == HCCL_SUCCESS) {
        CHK_PRT_RET(param.resCtx == nullptr || param.ctxSize == 0,
            HCCL_ERROR("Incomplete cached Scatter algorithm context"), HCCL_E_INTERNAL);
        return HCCL_SUCCESS;
    }
    if (largeRoot || parallelSmallRoot || hybridRemote) {
        // Communicator V2 grows its pool and returns the first N thread handles.
        // The original main keeps its reserved join notifications; new workers
        // need only their start notification at index 0.
        const uint32_t threadCount = parallelSmallRoot ? ScatterSmallAuxWorkerCount(ctx.workerCount, ctx.smallMode) + 1 :
            (largeRoot ? ctx.workerCount + 1 : 2);
        ctx.threads.resize(threadCount);
        CHK_RET(HcclThreadAcquire(comm, CommEngine::COMM_ENGINE_AICPU_TS,
            threadCount, 1, ctx.threads.data()));
        CHK_PRT_RET(ctx.threads[0] != ctx.aicpuThread,
            HCCL_ERROR("Scatter requires a stable AICPU_TS main thread during pool expansion"), HCCL_E_NOT_SUPPORT);
    }
    // Each plan has immutable device storage, so later host calls cannot
    // overwrite a context that a previously queued kernel still uses.
    CHK_RET(StoreContext(comm, storageTag, CommEngine::COMM_ENGINE_AICPU_TS,
        ctx, &param.resCtx, param.ctxSize));
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    OpParam param{};
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.myRank >= param.rankSize || root >= param.rankSize,
        HCCL_ERROR("Invalid Scatter root/rank"), HCCL_E_PARA);
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("This Scatter implementation supports FP32"), HCCL_E_PARA);
    CHK_PRT_RET(recvCount > std::numeric_limits<uint64_t>::max() / sizeof(float) / param.rankSize,
        HCCL_ERROR("Scatter input byte count overflows uint64_t"), HCCL_E_PARA);
    if (recvCount == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(recvBuf);
    if (param.myRank == root) {
        CHK_PTR_NULL(sendBuf);
    }
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.root = root;
    param.opType = HcclCMDType::HCCL_CMD_SCATTER;

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclThreadAcquireWithStream(comm, CommEngine::COMM_ENGINE_CPU_TS, stream, 1, &param.cpuThread));
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &param.cpuThread,
        CommEngine::COMM_ENGINE_AICPU_TS, &param.cpuThreadOnAicpu));
    AlgResourceCtx ctx{};
    CHK_RET(GetBaseContext(comm, param, ctx));
    CHK_RET(SelectAlgorithm(comm, param, ctx));
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &ctx.aicpuThread,
        CommEngine::COMM_ENGINE_CPU_TS, &param.aicpuThreadOnCpu));
    CHK_RET(GetAlgorithmContext(comm, param, ctx));
    const int tagLength = snprintf(param.tag, sizeof(param.tag), "hccl_scatter_v019_%s",
        ctx.chunkBytes == 0 ? (ctx.smallMode == SCATTER_SMALL_TREE ? "small_tree" : "small") :
        (ctx.routeMode == SCATTER_ROUTE_HYBRID_PUSH ? "hybrid_push" :
            (ScatterIsHybridRoute(ctx.routeMode) ? "hybrid" : "large")));
    CHK_PRT_RET(tagLength < 0 || static_cast<size_t>(tagLength) >= sizeof(param.tag),
        HCCL_ERROR("Scatter operation tag is too long"), HCCL_E_INTERNAL);
    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}
