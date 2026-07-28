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
#include <memory>
#include <queue>
#include <vector>

#include <ccu/ccu_launch.h>
#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>

#include "ccu_kernel.h"
#include "common.h"
#include "custom.h"
#include "exec_op.h"
#include "hccl.h"
#include "log.h"

namespace {

constexpr uint32_t CHANNEL_NOTIFY_COUNT = 8;
constexpr uint32_t CONTROL_NOTIFY_MASK_BITS = 2;
constexpr uint32_t NOTIFY_MASK_BITS = 16;
constexpr uint32_t PIPELINE_SIGNAL_CAPACITY =
    CHANNEL_NOTIFY_COUNT * NOTIFY_MASK_BITS - CONTROL_NOTIFY_MASK_BITS;
static_assert(CHANNEL_NOTIFY_COUNT * NOTIFY_MASK_BITS > CONTROL_NOTIFY_MASK_BITS);
constexpr uint32_t FIRST_SERVER_SIZE = 8;
constexpr uint32_t PIPELINE_TREE_MAX_CHILDREN = 2;
constexpr uint32_t DUAL_STRIPE_COUNT = 2;
constexpr uint64_t SMALL_MESSAGE_BYTES = 512ULL * 1024ULL;
// Platform-proven V25 granularity: 512 MiB needs 124 full chunks plus one
// tail, fitting in the 126 DATA bits left after address/token bootstrap.
constexpr uint64_t STATIC_PIPELINE_CHUNK_BYTES = 33ULL * 128ULL * 1024ULL;
static_assert(STATIC_PIPELINE_CHUNK_BYTES < MAX_DATA_SIZE);
// Attack build: finer dual-stripe chunking to cut the 2x8 16-hop pipeline-fill
// overhead (points 20/21). A 256 MiB stripe -> ~114 chunks (<=126 capacity).
constexpr uint64_t DUAL_STRIPE_CHUNK_BYTES = 3ULL * 768ULL * 1024ULL;
static_assert(DUAL_STRIPE_CHUNK_BYTES < MAX_DATA_SIZE);
constexpr uint32_t INTRA_SERVER_LAYER = 0;
constexpr uint32_t INTER_SERVER_LAYER = 1;
// Two independent main-to-worker gates let the two competition-sized chunks
// coexist without recording twice into one unconsumed ThreadNotify.
constexpr uint32_t HIERARCHICAL_WORKER_NOTIFY_COUNT = 2;
constexpr uint32_t BALANCED_FUSED_WORKER_NOTIFY_COUNT = 2;
constexpr uint32_t PARALLEL_WORKER_NOTIFY_COUNT = 2;
constexpr uint32_t RELAY_WORKER_NOTIFY_COUNT = 2;
constexpr uint32_t SMALL_DIRECT_WORKER_NOTIFY_COUNT = 1;
constexpr uint32_t DUAL_STRIPE_WORKER_NOTIFY_COUNT = 1;
constexpr bool SMALL_DUAL_DIE_PARALLEL = true;

struct CandidateLink {
    bool valid = false;
    CommLink link{};
    uint32_t layer = UINT32_MAX;
    uint32_t protocolPriority = UINT32_MAX;
    uint32_t hop = UINT32_MAX;
};

uint32_t GetProtocolPriority(CommProtocol protocol)
{
    if (protocol == COMM_PROTOCOL_UBC_CTP) {
        return 0;
    }
    if (protocol == COMM_PROTOCOL_UBC_TP) {
        return 1;
    }
    return UINT32_MAX;
}

bool IsBetterCandidate(uint32_t protocolPriority, uint32_t hop, uint32_t layer,
    const CandidateLink &current)
{
    if (!current.valid || protocolPriority != current.protocolPriority) {
        return !current.valid || protocolPriority < current.protocolPriority;
    }
    if (hop != current.hop) {
        return hop < current.hop;
    }
    return layer < current.layer;
}

HcclResult SelectCrossServerChannelDesc(
    HcclComm comm, uint32_t localRank, uint32_t remoteRank, HcclChannelDesc &channelDesc,
    EndpointAttrDieId &localDie)
{
    uint32_t *layers = nullptr;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layers, &layerNum));
    if (layerNum == 0 || layers == nullptr) {
        HCCL_ERROR("Broadcast RankGraph has no network layers");
        return HCCL_E_NOT_FOUND;
    }

    std::vector<uint32_t> netLayers(layers, layers + layerNum);
    std::sort(netLayers.begin(), netLayers.end());

    CandidateLink selected;
    for (uint32_t netLayer : netLayers) {
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        const HcclResult linkRet = HcclRankGraphGetLinks(
            comm, netLayer, localRank, remoteRank, &links, &linkNum);
        if (linkRet != HCCL_SUCCESS || linkNum == 0) {
            continue;
        }
        if (links == nullptr) {
            HCCL_ERROR("Broadcast RankGraph returned null links: src=%u dst=%u layer=%u count=%u",
                localRank, remoteRank, netLayer, linkNum);
            return HCCL_E_INTERNAL;
        }

        for (uint32_t linkIdx = 0; linkIdx < linkNum; ++linkIdx) {
            const CommLink &link = links[linkIdx];
            const uint32_t protocolPriority = GetProtocolPriority(link.linkAttr.linkProtocol);
            if (protocolPriority == UINT32_MAX) {
                continue;
            }

            if (!IsBetterCandidate(protocolPriority, link.linkAttr.hop, netLayer, selected)) {
                continue;
            }

            selected.valid = true;
            selected.link = link;
            selected.layer = netLayer;
            selected.protocolPriority = protocolPriority;
            selected.hop = link.linkAttr.hop;
        }
    }

    if (!selected.valid) {
        HCCL_ERROR("No cross-server CCU link from rank %u to rank %u", localRank, remoteRank);
        return HCCL_E_NOT_FOUND;
    }

    CHK_RET(HcclRankGraphGetEndpointInfo(comm, localRank, &selected.link.srcEndpointDesc,
        ENDPOINT_ATTR_DIE_ID, sizeof(localDie), &localDie));
    CHK_RET(HcclChannelDescInit(&channelDesc, 1));
    channelDesc.remoteRank = remoteRank;
    channelDesc.channelProtocol = selected.link.linkAttr.linkProtocol;
    channelDesc.notifyNum = CHANNEL_NOTIFY_COUNT;
    channelDesc.localEndpoint = selected.link.srcEndpointDesc;
    channelDesc.remoteEndpoint = selected.link.dstEndpointDesc;
    return HCCL_SUCCESS;
}

HcclResult SelectLayerChannelDesc(HcclComm comm, uint32_t localRank, uint32_t remoteRank,
    uint32_t targetLayer, HcclChannelDesc &channelDesc, EndpointAttrDieId &localDie)
{
    CommLink *links = nullptr;
    uint32_t linkNum = 0;
    const HcclResult linkRet = HcclRankGraphGetLinks(
        comm, targetLayer, localRank, remoteRank, &links, &linkNum);
    if (linkRet != HCCL_SUCCESS || linkNum == 0) {
        HCCL_ERROR("No CCU link from rank %u to rank %u on layer %u",
            localRank, remoteRank, targetLayer);
        return linkRet == HCCL_SUCCESS ? HCCL_E_NOT_FOUND : linkRet;
    }
    if (links == nullptr) {
        HCCL_ERROR("Broadcast RankGraph returned null links: src=%u dst=%u layer=%u count=%u",
            localRank, remoteRank, targetLayer, linkNum);
        return HCCL_E_INTERNAL;
    }

    CandidateLink selected;
    for (uint32_t linkIdx = 0; linkIdx < linkNum; ++linkIdx) {
        const CommLink &link = links[linkIdx];
        const uint32_t protocolPriority = GetProtocolPriority(link.linkAttr.linkProtocol);
        if (protocolPriority == UINT32_MAX ||
            !IsBetterCandidate(protocolPriority, link.linkAttr.hop, targetLayer, selected)) {
            continue;
        }
        selected.valid = true;
        selected.link = link;
        selected.layer = targetLayer;
        selected.protocolPriority = protocolPriority;
        selected.hop = link.linkAttr.hop;
    }
    if (!selected.valid) {
        HCCL_ERROR("No supported CCU protocol from rank %u to rank %u on layer %u",
            localRank, remoteRank, targetLayer);
        return HCCL_E_NOT_FOUND;
    }

    CHK_RET(HcclRankGraphGetEndpointInfo(comm, localRank, &selected.link.srcEndpointDesc,
        ENDPOINT_ATTR_DIE_ID, sizeof(localDie), &localDie));
    CHK_RET(HcclChannelDescInit(&channelDesc, 1));
    channelDesc.remoteRank = remoteRank;
    channelDesc.channelProtocol = selected.link.linkAttr.linkProtocol;
    channelDesc.notifyNum = CHANNEL_NOTIFY_COUNT;
    channelDesc.localEndpoint = selected.link.srcEndpointDesc;
    channelDesc.remoteEndpoint = selected.link.dstEndpointDesc;
    return HCCL_SUCCESS;
}

bool IsFirstServerRank(uint32_t rank)
{
    return rank < FIRST_SERVER_SIZE;
}

void AppendPartitionRanks(uint32_t begin, uint32_t end, uint32_t preferredRank,
    std::vector<uint32_t> &ranks)
{
    ranks.clear();
    if (preferredRank >= begin && preferredRank < end) {
        ranks.push_back(preferredRank);
    }
    for (uint32_t rank = begin; rank < end; ++rank) {
        if (rank != preferredRank) {
            ranks.push_back(rank);
        }
    }
}

HcclResult AddDirectedTreeEdge(uint32_t parent, uint32_t child,
    std::vector<uint32_t> &parents, std::vector<std::vector<uint32_t>> &children)
{
    if (parent >= parents.size() || child >= parents.size() || parent == child ||
        IsFirstServerRank(parent) == IsFirstServerRank(child)) {
        HCCL_ERROR("Invalid pipelined Broadcast edge %u -> %u", parent, child);
        return HCCL_E_INTERNAL;
    }
    if (parents[child] != INVALID_VALUE_RANKID) {
        HCCL_ERROR("Pipelined Broadcast rank %u has two parents", child);
        return HCCL_E_INTERNAL;
    }
    if (children[parent].size() >= PIPELINE_TREE_MAX_CHILDREN) {
        HCCL_ERROR("Pipelined Broadcast rank %u exceeds child limit %u",
            parent, PIPELINE_TREE_MAX_CHILDREN);
        return HCCL_E_INTERNAL;
    }
    parents[child] = parent;
    children[parent].push_back(child);
    return HCCL_SUCCESS;
}

HcclResult BuildPipelinedCrossServerTree(uint32_t rankSize, uint32_t root,
    std::vector<uint32_t> &parents, std::vector<std::vector<uint32_t>> &children)
{
    if (rankSize != 16 && rankSize != 12) {
        return HCCL_E_NOT_SUPPORT;
    }

    std::vector<uint32_t> firstServerRanks;
    std::vector<uint32_t> secondServerRanks;
    AppendPartitionRanks(0, FIRST_SERVER_SIZE, root, firstServerRanks);
    AppendPartitionRanks(FIRST_SERVER_SIZE, rankSize, root, secondServerRanks);

    const bool rootInFirstServer = IsFirstServerRank(root);
    const std::vector<uint32_t> &rootSide = rootInFirstServer ? firstServerRanks : secondServerRanks;
    const std::vector<uint32_t> &otherSide = rootInFirstServer ? secondServerRanks : firstServerRanks;
    const size_t pairedRanks = std::min(rootSide.size(), otherSide.size());

    std::vector<uint32_t> spine;
    spine.reserve(rankSize);
    for (size_t idx = 0; idx < pairedRanks; ++idx) {
        spine.push_back(rootSide[idx]);
        spine.push_back(otherSide[idx]);
    }

    size_t rootSideExtraBegin = pairedRanks;
    if (rootSide.size() > pairedRanks) {
        // Close the alternating spine with one rank from the root partition.
        spine.push_back(rootSide[pairedRanks]);
        rootSideExtraBegin = pairedRanks + 1;
    }

    if (spine.empty() || spine.front() != root) {
        HCCL_ERROR("Failed to start pipelined Broadcast tree at root %u", root);
        return HCCL_E_INTERNAL;
    }

    parents.assign(rankSize, INVALID_VALUE_RANKID);
    children.assign(rankSize, {});
    for (size_t idx = 1; idx < spine.size(); ++idx) {
        CHK_RET(AddDirectedTreeEdge(spine[idx - 1], spine[idx], parents, children));
    }

    std::vector<uint32_t> extraRanks;
    extraRanks.insert(extraRanks.end(), rootSide.begin() + rootSideExtraBegin, rootSide.end());
    extraRanks.insert(extraRanks.end(), otherSide.begin() + pairedRanks, otherSide.end());

    const std::vector<uint32_t> &minorityServerRanks =
        firstServerRanks.size() < secondServerRanks.size() ? firstServerRanks : secondServerRanks;
    if (extraRanks.size() > minorityServerRanks.size()) {
        HCCL_ERROR("Cannot balance pipelined Broadcast tree: extras=%zu minority=%zu",
            extraRanks.size(), minorityServerRanks.size());
        return HCCL_E_INTERNAL;
    }
    for (size_t idx = 0; idx < extraRanks.size(); ++idx) {
        CHK_RET(AddDirectedTreeEdge(minorityServerRanks[idx], extraRanks[idx], parents, children));
    }

    uint32_t edgeCount = 0;
    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        if (rank != root && parents[rank] == INVALID_VALUE_RANKID) {
            HCCL_ERROR("Pipelined Broadcast tree did not reach rank %u", rank);
            return HCCL_E_INTERNAL;
        }
        edgeCount += static_cast<uint32_t>(children[rank].size());
    }
    if (edgeCount != rankSize - 1) {
        HCCL_ERROR("Pipelined Broadcast tree has %u edges for %u ranks", edgeCount, rankSize);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult BuildDualStripePipelines(uint32_t rankSize, uint32_t root,
    std::vector<std::vector<uint32_t>> &stripeParents,
    std::vector<std::vector<uint32_t>> &stripeChildren)
{
    if (rankSize != 2 * FIRST_SERVER_SIZE || root != 0) {
        return HCCL_E_NOT_SUPPORT;
    }

    stripeParents.assign(
        DUAL_STRIPE_COUNT, std::vector<uint32_t>(rankSize, INVALID_VALUE_RANKID));
    stripeChildren.assign(
        DUAL_STRIPE_COUNT, std::vector<uint32_t>(rankSize, INVALID_VALUE_RANKID));
    std::vector<std::vector<uint32_t>> edgeOwners(
        rankSize, std::vector<uint32_t>(rankSize, INVALID_VALUE_RANKID));
    constexpr uint32_t REMOTE_SHIFTS[DUAL_STRIPE_COUNT] = {0, 2};

    for (uint32_t stripeIdx = 0; stripeIdx < DUAL_STRIPE_COUNT; ++stripeIdx) {
        std::vector<uint32_t> path;
        path.reserve(rankSize);
        for (uint32_t localIdx = 0; localIdx < FIRST_SERVER_SIZE; ++localIdx) {
            path.push_back(localIdx);
            path.push_back(FIRST_SERVER_SIZE +
                (localIdx + REMOTE_SHIFTS[stripeIdx]) % FIRST_SERVER_SIZE);
        }
        if (path.size() != rankSize || path.front() != root) {
            HCCL_ERROR("Invalid dual-stripe path: stripe=%u size=%zu root=%u",
                stripeIdx, path.size(), root);
            return HCCL_E_INTERNAL;
        }

        for (size_t pathIdx = 1; pathIdx < path.size(); ++pathIdx) {
            const uint32_t parent = path[pathIdx - 1];
            const uint32_t child = path[pathIdx];
            if (IsFirstServerRank(parent) == IsFirstServerRank(child) ||
                stripeParents[stripeIdx][child] != INVALID_VALUE_RANKID ||
                stripeChildren[stripeIdx][parent] != INVALID_VALUE_RANKID) {
                HCCL_ERROR("Invalid dual-stripe edge: stripe=%u parent=%u child=%u",
                    stripeIdx, parent, child);
                return HCCL_E_INTERNAL;
            }
            const uint32_t edgeBegin = std::min(parent, child);
            const uint32_t edgeEnd = std::max(parent, child);
            if (edgeOwners[edgeBegin][edgeEnd] != INVALID_VALUE_RANKID) {
                HCCL_ERROR("Dual-stripe paths reused edge %u-%u in stripes %u and %u",
                    edgeBegin, edgeEnd, edgeOwners[edgeBegin][edgeEnd], stripeIdx);
                return HCCL_E_INTERNAL;
            }
            edgeOwners[edgeBegin][edgeEnd] = stripeIdx;
            stripeParents[stripeIdx][child] = parent;
            stripeChildren[stripeIdx][parent] = child;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult BuildDistributedSmallCtpTree(uint32_t rankSize, uint32_t root,
    std::vector<uint32_t> &parents, std::vector<std::vector<uint32_t>> &children)
{
    const bool rootInFirstGroup = root < FIRST_SERVER_SIZE;
    std::vector<uint32_t> sameGroupRanks;
    std::vector<uint32_t> otherGroupRanks;
    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        if (rank == root) {
            continue;
        }
        if ((rank < FIRST_SERVER_SIZE) == rootInFirstGroup) {
            sameGroupRanks.push_back(rank);
        } else {
            otherGroupRanks.push_back(rank);
        }
    }
    if (otherGroupRanks.empty()) {
        HCCL_ERROR("Small CTP tree has no opposite-group rank: root=%u rankSize=%u", root, rankSize);
        return HCCL_E_INTERNAL;
    }

    parents.assign(rankSize, INVALID_VALUE_RANKID);
    children.assign(rankSize, {});
    for (uint32_t otherRank : otherGroupRanks) {
        parents[otherRank] = root;
        children[root].push_back(otherRank);
    }
    for (size_t rankIdx = 0; rankIdx < sameGroupRanks.size(); ++rankIdx) {
        const uint32_t parent = otherGroupRanks[rankIdx % otherGroupRanks.size()];
        const uint32_t child = sameGroupRanks[rankIdx];
        parents[child] = parent;
        children[parent].push_back(child);
    }

    uint32_t edgeCount = 0;
    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        edgeCount += static_cast<uint32_t>(children[rank].size());
        if (rank != root && parents[rank] == INVALID_VALUE_RANKID) {
            HCCL_ERROR("Small CTP tree rank %u has no parent", rank);
            return HCCL_E_INTERNAL;
        }
        if (children[rank].size() > CHANNEL_NOTIFY_COUNT) {
            HCCL_ERROR("Small CTP tree rank %u has too many children: %zu", rank,
                children[rank].size());
            return HCCL_E_INTERNAL;
        }
    }
    if (edgeCount != rankSize - 1) {
        HCCL_ERROR("Small CTP tree edge mismatch: edges=%u rankSize=%u", edgeCount, rankSize);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult BuildCompetitionTree(uint32_t rankSize, uint32_t root, bool usePipelinedTree,
    std::vector<uint32_t> &parents, std::vector<std::vector<uint32_t>> &children)
{
    if (usePipelinedTree) {
        if (rankSize == 4) {
            // Exact V25 large-message chain: one child per rank keeps the
            // per-chunk forwarding path fully pipelined.
            parents.assign(rankSize, INVALID_VALUE_RANKID);
            children.assign(rankSize, {});
            uint32_t parent = root;
            for (uint32_t step = 1; step < rankSize; ++step) {
                const uint32_t child = (root + step) % rankSize;
                parents[child] = parent;
                children[parent].push_back(child);
                parent = child;
            }
            return HCCL_SUCCESS;
        }
        return BuildPipelinedCrossServerTree(rankSize, root, parents, children);
    }

    if (rankSize == 4) {
        // V28b platform-proven 4x1 small-message star (point 22: 14 us).
        parents.assign(rankSize, INVALID_VALUE_RANKID);
        children.assign(rankSize, {});
        for (uint32_t rank = 0; rank < rankSize; ++rank) {
            if (rank == root) {
                continue;
            }
            parents[rank] = root;
            children[root].push_back(rank);
        }
        return HCCL_SUCCESS;
    }

    if (rankSize == 12 || rankSize == 16) {
        return BuildDistributedSmallCtpTree(rankSize, root, parents, children);
    }

    HCCL_ERROR("Unsupported competition Broadcast rankSize %u", rankSize);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult CreateTreeResource(HcclComm comm, const OpParam &param,
    bool usePipelinedTree, AlgResourceCtx &resource)
{
    std::vector<uint32_t> parents;
    std::vector<std::vector<uint32_t>> children;
    CHK_RET(BuildCompetitionTree(param.rankSize, param.root, usePipelinedTree, parents, children));

    const uint32_t parent = parents[param.myRank];
    std::vector<uint32_t> peers;
    if (parent != INVALID_VALUE_RANKID) {
        peers.push_back(parent);
    }
    peers.insert(peers.end(), children[param.myRank].begin(), children[param.myRank].end());
    if (peers.empty()) {
        HCCL_ERROR("Broadcast rank %u has no tree edge", param.myRank);
        return HCCL_E_INTERNAL;
    }

    std::vector<HcclChannelDesc> channelDescs(peers.size());
    std::vector<EndpointAttrDieId> localDies(peers.size(), UINT32_MAX);
    for (size_t channelIdx = 0; channelIdx < peers.size(); ++channelIdx) {
        CHK_RET(SelectCrossServerChannelDesc(comm, param.myRank, peers[channelIdx],
            channelDescs[channelIdx], localDies[channelIdx]));
    }
    for (size_t channelIdx = 1; channelIdx < localDies.size(); ++channelIdx) {
        if (localDies[channelIdx] != localDies[0]) {
            HCCL_ERROR("Broadcast cross-server links selected different local dies: rank=%u firstDie=%u die=%u",
                param.myRank, localDies[0], localDies[channelIdx]);
            return HCCL_E_UNAVAIL;
        }
    }

    std::vector<ChannelHandle> channels(peers.size());
    CHK_RET(HcclChannelAcquire(comm, COMM_ENGINE_CCU, channelDescs.data(),
        static_cast<uint32_t>(channelDescs.size()), channels.data()));

    auto kernelArg = std::make_shared<CcuKernelArgBase>();
    kernelArg->mode = BroadcastKernelMode::TREE;
    kernelArg->myRank = param.myRank;
    kernelArg->rankSize = param.rankSize;
    kernelArg->root = param.root;
    if (usePipelinedTree) {
        const auto sizeIter = SIZE_TABLE.find(param.dataType);
        if (sizeIter == SIZE_TABLE.end() || sizeIter->second == 0 ||
            param.count > UINT64_MAX / sizeIter->second) {
            HCCL_ERROR("Invalid Broadcast count or data type for static pipeline specialization");
            return HCCL_E_PARA;
        }
        const uint64_t totalBytes = param.count * sizeIter->second;
        const uint64_t fullChunks = totalBytes / STATIC_PIPELINE_CHUNK_BYTES;
        if (fullChunks > UINT32_MAX) {
            HCCL_ERROR("Broadcast static pipeline chunk count is too large: chunks=%llu",
                static_cast<unsigned long long>(fullChunks));
            return HCCL_E_NOT_SUPPORT;
        }
        kernelArg->pipelineFullChunks = static_cast<uint32_t>(fullChunks);
        kernelArg->pipelineChunkBytes = STATIC_PIPELINE_CHUNK_BYTES;
        kernelArg->pipelineTailBytes = totalBytes % STATIC_PIPELINE_CHUNK_BYTES;
        const uint64_t signalCount = fullChunks +
            (kernelArg->pipelineTailBytes != 0 ? 1ULL : 0ULL);
        if (signalCount > PIPELINE_SIGNAL_CAPACITY) {
            HCCL_ERROR("Broadcast static pipeline needs too many DATA notify bits: signals=%llu capacity=%u",
                static_cast<unsigned long long>(signalCount), PIPELINE_SIGNAL_CAPACITY);
            return HCCL_E_NOT_SUPPORT;
        }
    }
    size_t firstChildChannel = 0;
    if (parent != INVALID_VALUE_RANKID) {
        kernelArg->hasParent = 1;
        kernelArg->parentChannel = channels[0];
        firstChildChannel = 1;
    }
    for (size_t channelIdx = firstChildChannel; channelIdx < channels.size(); ++channelIdx) {
        if (kernelArg->childCount >= MAX_RANK_SIZE - 1) {
            HCCL_ERROR("Broadcast rank %u has too many children", param.myRank);
            return HCCL_E_PARA;
        }
        const uint32_t childRank = peers[channelIdx];
        kernelArg->childChannels[kernelArg->childCount] = channels[channelIdx];
        kernelArg->childNeedsChunkNotify[kernelArg->childCount] =
            children[childRank].empty() ? 0U : 1U;
        ++kernelArg->childCount;
    }

    CcuInsHandle ccuInstance = 0;
    uint32_t instanceNum = 1;
    CHK_RET(HcclCommQueryCcuIns(comm, &ccuInstance, &instanceNum));
    if (instanceNum != 1 || ccuInstance == 0) {
        HCCL_ERROR("Unexpected CCU instance count %u", instanceNum);
        return HCCL_E_UNAVAIL;
    }

    CHK_RET_CCU(HcommCcuKernelRegisterStart(ccuInstance));
    char kernelName[64]{};
    (void)snprintf(kernelName, sizeof(kernelName), "HcclBcastV29_%c_f%u_t%llu_r%u_k%u_d%u",
        usePipelinedTree ? 'P' : 'S', kernelArg->pipelineFullChunks,
        static_cast<unsigned long long>(kernelArg->pipelineTailBytes), param.root,
        param.myRank, localDies[0]);
    CcuKernelHandle kernelHandle = 0;
    const void *registerArgs[] = {kernelArg.get()};
    CHK_RET_CCU(HcommCcuKernelRegister(ccuInstance, 0, kernelName,
        reinterpret_cast<const void *>(ops_hccl::CcuKernel), registerArgs, 1, &kernelHandle));
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(ccuInstance));

    resource.ccuThread = param.cpuThread;
    resource.threads.clear();
    resource.ccuKernels.clear();
    resource.ccuKernels.push_back(kernelHandle);
    HCCL_INFO("Broadcast v29 %s tree rank=%u die=%u parent=%u children=%u full=%u tail=%llu",
        usePipelinedTree ? "static-pipeline" :
            (param.rankSize == 4 ? "small-star" : "small-distributed"),
        param.myRank, localDies[0], parent, kernelArg->childCount,
        kernelArg->pipelineFullChunks,
        static_cast<unsigned long long>(kernelArg->pipelineTailBytes));
    return HCCL_SUCCESS;
}

HcclResult CreateDualStripePipelineResource(
    HcclComm comm, const OpParam &param, AlgResourceCtx &resource)
{
    if (param.rankSize != 2 * FIRST_SERVER_SIZE || param.root != 0 ||
        param.myRank >= param.rankSize) {
        HCCL_ERROR("Invalid dual-stripe Broadcast configuration: rank=%u rankSize=%u root=%u",
            param.myRank, param.rankSize, param.root);
        return HCCL_E_PARA;
    }
    const auto sizeIter = SIZE_TABLE.find(param.dataType);
    if (sizeIter == SIZE_TABLE.end() || sizeIter->second == 0 ||
        param.count > UINT64_MAX / sizeIter->second) {
        return HCCL_E_PARA;
    }

    std::vector<std::vector<uint32_t>> stripeParents;
    std::vector<std::vector<uint32_t>> stripeChildren;
    CHK_RET(BuildDualStripePipelines(
        param.rankSize, param.root, stripeParents, stripeChildren));

    std::vector<uint32_t> peers;
    for (uint32_t stripeIdx = 0; stripeIdx < DUAL_STRIPE_COUNT; ++stripeIdx) {
        const uint32_t parent = stripeParents[stripeIdx][param.myRank];
        const uint32_t child = stripeChildren[stripeIdx][param.myRank];
        if (parent != INVALID_VALUE_RANKID &&
            std::find(peers.begin(), peers.end(), parent) == peers.end()) {
            peers.push_back(parent);
        }
        if (child != INVALID_VALUE_RANKID &&
            std::find(peers.begin(), peers.end(), child) == peers.end()) {
            peers.push_back(child);
        }
    }
    if (peers.empty()) {
        return HCCL_E_INTERNAL;
    }

    std::vector<HcclChannelDesc> channelDescs(peers.size());
    std::vector<EndpointAttrDieId> localDies(peers.size(), UINT32_MAX);
    for (size_t peerIdx = 0; peerIdx < peers.size(); ++peerIdx) {
        CHK_RET(SelectCrossServerChannelDesc(comm, param.myRank, peers[peerIdx],
            channelDescs[peerIdx], localDies[peerIdx]));
    }
    for (size_t peerIdx = 1; peerIdx < localDies.size(); ++peerIdx) {
        if (localDies[peerIdx] != localDies[0]) {
            HCCL_ERROR("Dual-stripe links selected different local dies: rank=%u first=%u current=%u",
                param.myRank, localDies[0], localDies[peerIdx]);
            return HCCL_E_UNAVAIL;
        }
    }

    std::vector<ChannelHandle> channels(peers.size());
    CHK_RET(HcclChannelAcquire(comm, COMM_ENGINE_CCU, channelDescs.data(),
        static_cast<uint32_t>(channelDescs.size()), channels.data()));
    const auto getChannel = [&peers, &channels](uint32_t peer) -> ChannelHandle {
        const auto peerIter = std::find(peers.begin(), peers.end(), peer);
        return peerIter == peers.end() ? 0 :
            channels[static_cast<size_t>(peerIter - peers.begin())];
    };

    const uint64_t firstStripeElements = param.count / DUAL_STRIPE_COUNT;
    const uint64_t stripeElements[DUAL_STRIPE_COUNT] = {
        firstStripeElements, param.count - firstStripeElements};
    std::shared_ptr<CcuKernelArgBase> kernelArgs[DUAL_STRIPE_COUNT];
    for (uint32_t stripeIdx = 0; stripeIdx < DUAL_STRIPE_COUNT; ++stripeIdx) {
        const uint64_t stripeBytes = stripeElements[stripeIdx] * sizeIter->second;
        const uint64_t fullChunks = stripeBytes / DUAL_STRIPE_CHUNK_BYTES;
        const uint64_t tailBytes = stripeBytes % DUAL_STRIPE_CHUNK_BYTES;
        const uint64_t signalCount = fullChunks + (tailBytes != 0 ? 1ULL : 0ULL);
        if (stripeBytes == 0 || fullChunks > UINT32_MAX ||
            signalCount > PIPELINE_SIGNAL_CAPACITY) {
            HCCL_ERROR("Invalid dual-stripe size: stripe=%u bytes=%llu signals=%llu",
                stripeIdx, static_cast<unsigned long long>(stripeBytes),
                static_cast<unsigned long long>(signalCount));
            return HCCL_E_NOT_SUPPORT;
        }

        auto kernelArg = std::make_shared<CcuKernelArgBase>();
        kernelArg->mode = BroadcastKernelMode::TREE;
        kernelArg->myRank = param.myRank;
        kernelArg->rankSize = param.rankSize;
        kernelArg->root = param.root;
        kernelArg->pipelineFullChunks = static_cast<uint32_t>(fullChunks);
        kernelArg->pipelineChunkBytes = DUAL_STRIPE_CHUNK_BYTES;
        kernelArg->pipelineTailBytes = tailBytes;
        const uint32_t parent = stripeParents[stripeIdx][param.myRank];
        const uint32_t child = stripeChildren[stripeIdx][param.myRank];
        if (parent != INVALID_VALUE_RANKID) {
            kernelArg->hasParent = 1;
            kernelArg->parentChannel = getChannel(parent);
        }
        if (child != INVALID_VALUE_RANKID) {
            kernelArg->childCount = 1;
            kernelArg->childChannels[0] = getChannel(child);
            kernelArg->childNeedsChunkNotify[0] =
                stripeChildren[stripeIdx][child] == INVALID_VALUE_RANKID ? 0U : 1U;
        }
        if ((parent != INVALID_VALUE_RANKID &&
             kernelArg->parentChannel == 0) ||
            (child != INVALID_VALUE_RANKID &&
             kernelArg->childChannels[0] == 0)) {
            return HCCL_E_INTERNAL;
        }
        kernelArgs[stripeIdx] = kernelArg;
    }

    CcuInsHandle ccuInstance = 0;
    uint32_t instanceNum = 1;
    CHK_RET(HcclCommQueryCcuIns(comm, &ccuInstance, &instanceNum));
    if (instanceNum != 1 || ccuInstance == 0) {
        return HCCL_E_UNAVAIL;
    }
    CHK_RET_CCU(HcommCcuKernelRegisterStart(ccuInstance));
    CcuKernelHandle kernelHandles[DUAL_STRIPE_COUNT]{};
    for (uint32_t stripeIdx = 0; stripeIdx < DUAL_STRIPE_COUNT; ++stripeIdx) {
        char kernelName[64]{};
        (void)snprintf(kernelName, sizeof(kernelName),
            "HcclB40D_s%u_f%u_t%llu_r%u_k%u_d%u", stripeIdx,
            kernelArgs[stripeIdx]->pipelineFullChunks,
            static_cast<unsigned long long>(kernelArgs[stripeIdx]->pipelineTailBytes),
            param.root, param.myRank, localDies[0]);
        const void *registerArgs[] = {kernelArgs[stripeIdx].get()};
        CHK_RET_CCU(HcommCcuKernelRegister(ccuInstance, 0, kernelName,
            reinterpret_cast<const void *>(ops_hccl::CcuKernel),
            registerArgs, 1, &kernelHandles[stripeIdx]));
    }
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(ccuInstance));

    resource.ccuThread = param.cpuThread;
    resource.ccuKernels = {kernelHandles[0], kernelHandles[1]};
    resource.threads.resize(1);
    CHK_RET(HcclThreadAcquire(comm, COMM_ENGINE_CCU, 1,
        1, resource.threads.data()));
    HCCL_INFO("Broadcast v40a dual-stripe rank=%u die=%u peers=%zu first=%llu second=%llu",
        param.myRank, localDies[0], peers.size(),
        static_cast<unsigned long long>(stripeElements[0] * sizeIter->second),
        static_cast<unsigned long long>(stripeElements[1] * sizeIter->second));
    return HCCL_SUCCESS;
}

HcclResult BuildScatterPeers(uint32_t rankSize, uint32_t myRank, std::vector<uint32_t> &peers)
{
    peers.clear();
    if (rankSize == 4) {
        for (uint32_t rank = 0; rank < rankSize; ++rank) {
            if (rank != myRank) {
                peers.push_back(rank);
            }
        }
        return HCCL_SUCCESS;
    }

    if (rankSize != 12 && rankSize != 16) {
        HCCL_ERROR("Unsupported scatter-allgather Broadcast rankSize %u", rankSize);
        return HCCL_E_NOT_SUPPORT;
    }

    const uint32_t begin = IsFirstServerRank(myRank) ? FIRST_SERVER_SIZE : 0;
    const uint32_t end = IsFirstServerRank(myRank) ? rankSize : FIRST_SERVER_SIZE;
    for (uint32_t rank = begin; rank < end; ++rank) {
        peers.push_back(rank);
    }
    return HCCL_SUCCESS;
}

HcclResult BuildHierarchicalIntraPeers(
    uint32_t rankSize, uint32_t myRank, std::vector<uint32_t> &peers)
{
    if ((rankSize != 12 && rankSize != 16) || myRank >= rankSize) {
        HCCL_ERROR("Invalid hierarchical intra rank configuration: rank=%u rankSize=%u",
            myRank, rankSize);
        return HCCL_E_PARA;
    }

    const uint32_t begin = IsFirstServerRank(myRank) ? 0 : FIRST_SERVER_SIZE;
    const uint32_t end = IsFirstServerRank(myRank) ? FIRST_SERVER_SIZE : rankSize;
    peers.clear();
    for (uint32_t rank = begin; rank < end; ++rank) {
        if (rank != myRank) {
            peers.push_back(rank);
        }
    }
    return peers.empty() ? HCCL_E_INTERNAL : HCCL_SUCCESS;
}

HcclResult BuildHierarchicalInterPeers(uint32_t rankSize, uint32_t root,
    uint32_t myRank, std::vector<uint32_t> &peers)
{
    if ((rankSize != 12 && rankSize != 16) || root >= FIRST_SERVER_SIZE ||
        myRank >= rankSize) {
        HCCL_ERROR("Invalid hierarchical inter rank configuration: rank=%u rankSize=%u root=%u",
            myRank, rankSize, root);
        return HCCL_E_PARA;
    }

    peers.clear();
    const uint32_t begin = IsFirstServerRank(myRank) ? FIRST_SERVER_SIZE : 0;
    const uint32_t end = IsFirstServerRank(myRank) ? rankSize : FIRST_SERVER_SIZE;
    for (uint32_t rank = begin; rank < end; ++rank) {
        peers.push_back(rank);
    }
    return HCCL_SUCCESS;
}

HcclResult AcquireLayerPeerChannels(HcclComm comm, const OpParam &param,
    const std::vector<uint32_t> &peers, uint32_t layer,
    std::vector<ChannelHandle> &channels, EndpointAttrDieId &localDie)
{
    if (peers.empty() || peers.size() > MAX_RANK_SIZE) {
        HCCL_ERROR("Invalid layer-%u peer count %zu for rank %u", layer, peers.size(), param.myRank);
        return HCCL_E_PARA;
    }

    std::vector<HcclChannelDesc> channelDescs(peers.size());
    std::vector<EndpointAttrDieId> localDies(peers.size(), UINT32_MAX);
    for (size_t peerIdx = 0; peerIdx < peers.size(); ++peerIdx) {
        CHK_RET(SelectLayerChannelDesc(comm, param.myRank, peers[peerIdx], layer,
            channelDescs[peerIdx], localDies[peerIdx]));
    }
    for (size_t peerIdx = 1; peerIdx < localDies.size(); ++peerIdx) {
        if (localDies[peerIdx] != localDies[0]) {
            HCCL_ERROR("Layer-%u links selected different local dies: rank=%u firstDie=%u die=%u",
                layer, param.myRank, localDies[0], localDies[peerIdx]);
            return HCCL_E_UNAVAIL;
        }
    }

    channels.resize(peers.size());
    CHK_RET(HcclChannelAcquire(comm, COMM_ENGINE_CCU, channelDescs.data(),
        static_cast<uint32_t>(channelDescs.size()), channels.data()));
    localDie = localDies[0];
    return HCCL_SUCCESS;
}

std::shared_ptr<CcuKernelArgBase> MakeSmallDirectTreeArg(const OpParam &param,
    const std::vector<ChannelHandle> &channels, bool hasParent)
{
    auto kernelArg = std::make_shared<CcuKernelArgBase>();
    kernelArg->mode = BroadcastKernelMode::TREE;
    kernelArg->myRank = param.myRank;
    kernelArg->rankSize = param.rankSize;
    kernelArg->root = param.root;
    if (hasParent) {
        kernelArg->hasParent = 1;
        kernelArg->parentChannel = channels[0];
        return kernelArg;
    }

    kernelArg->childCount = static_cast<uint32_t>(channels.size());
    for (size_t childIdx = 0; childIdx < channels.size(); ++childIdx) {
        kernelArg->childChannels[childIdx] = channels[childIdx];
    }
    return kernelArg;
}

HcclResult CreateSmallDualDieDirectResource(
    HcclComm comm, const OpParam &param, AlgResourceCtx &resource)
{
    if ((param.rankSize != 12 && param.rankSize != 16) ||
        param.root >= FIRST_SERVER_SIZE || param.myRank >= param.rankSize) {
        HCCL_ERROR("Invalid small dual-die Broadcast configuration: rank=%u rankSize=%u root=%u",
            param.myRank, param.rankSize, param.root);
        return HCCL_E_PARA;
    }

    const bool isRoot = param.myRank == param.root;
    std::vector<uint32_t> intraPeers;
    std::vector<uint32_t> interPeers;
    if (isRoot) {
        for (uint32_t rank = 0; rank < FIRST_SERVER_SIZE; ++rank) {
            if (rank != param.root) {
                intraPeers.push_back(rank);
            }
        }
        for (uint32_t rank = FIRST_SERVER_SIZE; rank < param.rankSize; ++rank) {
            interPeers.push_back(rank);
        }
    } else if (IsFirstServerRank(param.myRank)) {
        intraPeers.push_back(param.root);
    } else {
        interPeers.push_back(param.root);
    }

    std::vector<ChannelHandle> intraChannels;
    std::vector<ChannelHandle> interChannels;
    EndpointAttrDieId intraDie = UINT32_MAX;
    EndpointAttrDieId interDie = UINT32_MAX;
    if (!intraPeers.empty()) {
        CHK_RET(AcquireLayerPeerChannels(comm, param, intraPeers, INTRA_SERVER_LAYER,
            intraChannels, intraDie));
    }
    if (!interPeers.empty()) {
        CHK_RET(AcquireLayerPeerChannels(comm, param, interPeers, INTER_SERVER_LAYER,
            interChannels, interDie));
    }
    if (isRoot && (intraChannels.empty() || interChannels.empty() || intraDie == interDie)) {
        HCCL_ERROR("Small direct Broadcast needs two nonempty, distinct dies: rank=%u intra=%zu inter=%zu intraDie=%u interDie=%u",
            param.myRank, intraChannels.size(), interChannels.size(), intraDie, interDie);
        return HCCL_E_UNAVAIL;
    }

    std::shared_ptr<CcuKernelArgBase> intraArg;
    std::shared_ptr<CcuKernelArgBase> interArg;
    if (!intraChannels.empty()) {
        intraArg = MakeSmallDirectTreeArg(param, intraChannels, !isRoot);
    }
    if (!interChannels.empty()) {
        interArg = MakeSmallDirectTreeArg(param, interChannels, !isRoot);
    }

    CcuInsHandle ccuInstance = 0;
    uint32_t instanceNum = 1;
    CHK_RET(HcclCommQueryCcuIns(comm, &ccuInstance, &instanceNum));
    if (instanceNum != 1 || ccuInstance == 0) {
        HCCL_ERROR("Unexpected CCU instance count %u", instanceNum);
        return HCCL_E_UNAVAIL;
    }

    CHK_RET_CCU(HcommCcuKernelRegisterStart(ccuInstance));
    CcuKernelHandle intraKernel = 0;
    CcuKernelHandle interKernel = 0;
    char kernelName[64]{};
    if (intraArg != nullptr) {
        (void)snprintf(kernelName, sizeof(kernelName), "HcclBcastV31aSDI_r%u_k%u_d%u",
            param.root, param.myRank, intraDie);
        const void *registerArgs[] = {intraArg.get()};
        CHK_RET_CCU(HcommCcuKernelRegister(ccuInstance, 0, kernelName,
            reinterpret_cast<const void *>(ops_hccl::CcuKernel), registerArgs, 1,
            &intraKernel));
    }
    if (interArg != nullptr) {
        (void)snprintf(kernelName, sizeof(kernelName), "HcclBcastV31aSDX_r%u_k%u_d%u",
            param.root, param.myRank, interDie);
        const void *registerArgs[] = {interArg.get()};
        CHK_RET_CCU(HcommCcuKernelRegister(ccuInstance, 0, kernelName,
            reinterpret_cast<const void *>(ops_hccl::CcuKernel), registerArgs, 1,
            &interKernel));
    }
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(ccuInstance));

    resource.ccuThread = param.cpuThread;
    resource.ccuKernels.clear();
    resource.threads.clear();
    if (isRoot) {
        resource.ccuKernels = {intraKernel, interKernel};
        if (SMALL_DUAL_DIE_PARALLEL) {
            resource.threads.resize(1);
            CHK_RET(HcclThreadAcquire(comm, COMM_ENGINE_CCU, 1,
                SMALL_DIRECT_WORKER_NOTIFY_COUNT, resource.threads.data()));
        }
    } else {
        resource.ccuKernels.push_back(intraArg != nullptr ? intraKernel : interKernel);
    }
    HCCL_INFO("Broadcast v31a small direct rank=%u root=%u intraDie=%u interDie=%u intraPeers=%zu interPeers=%zu parallel=%u",
        param.myRank, param.root, intraDie, interDie, intraPeers.size(), interPeers.size(),
        SMALL_DUAL_DIE_PARALLEL ? 1U : 0U);
    return HCCL_SUCCESS;
}

std::shared_ptr<CcuKernelArgBase> MakePeerKernelArg(BroadcastKernelMode mode,
    const OpParam &param, const std::vector<uint32_t> &peers,
    const std::vector<ChannelHandle> &channels)
{
    auto kernelArg = std::make_shared<CcuKernelArgBase>();
    kernelArg->mode = mode;
    kernelArg->myRank = param.myRank;
    kernelArg->rankSize = param.rankSize;
    kernelArg->root = param.root;
    kernelArg->peerCount = static_cast<uint32_t>(peers.size());
    for (size_t peerIdx = 0; peerIdx < peers.size(); ++peerIdx) {
        kernelArg->peerChannels[peerIdx] = channels[peerIdx];
        kernelArg->peerRanks[peerIdx] = peers[peerIdx];
    }
    return kernelArg;
}

HcclResult CreateHierarchicalResource(
    HcclComm comm, const OpParam &param, AlgResourceCtx &resource)
{
    const bool rankOnRootServer = IsFirstServerRank(param.myRank);
    std::vector<uint32_t> intraPeers;
    std::vector<uint32_t> interPeers;
    if (rankOnRootServer) {
        CHK_RET(BuildHierarchicalIntraPeers(param.rankSize, param.myRank, intraPeers));
    }
    CHK_RET(BuildHierarchicalInterPeers(
        param.rankSize, param.root, param.myRank, interPeers));

    std::vector<ChannelHandle> intraChannels;
    std::vector<ChannelHandle> interChannels;
    EndpointAttrDieId intraDie = UINT32_MAX;
    EndpointAttrDieId interDie = UINT32_MAX;
    if (rankOnRootServer) {
        CHK_RET(AcquireLayerPeerChannels(comm, param, intraPeers, INTRA_SERVER_LAYER,
            intraChannels, intraDie));
    }
    CHK_RET(AcquireLayerPeerChannels(comm, param, interPeers, INTER_SERVER_LAYER,
        interChannels, interDie));
    if (rankOnRootServer && intraDie == interDie) {
        HCCL_ERROR("Hierarchical Broadcast expected distinct layer dies: rank=%u die=%u",
            param.myRank, intraDie);
        return HCCL_E_UNAVAIL;
    }

    auto interTransferArg = MakePeerKernelArg(
        BroadcastKernelMode::HIERARCHICAL_INTER_TRANSFER, param, interPeers, interChannels);
    std::shared_ptr<CcuKernelArgBase> intraScatterArg;
    std::shared_ptr<CcuKernelArgBase> intraAllgatherArg;
    if (rankOnRootServer) {
        intraScatterArg = MakePeerKernelArg(
            BroadcastKernelMode::HIERARCHICAL_INTRA_SCATTER, param, intraPeers, intraChannels);
        intraAllgatherArg = MakePeerKernelArg(
            BroadcastKernelMode::HIERARCHICAL_INTRA_ALLGATHER, param, intraPeers, intraChannels);
    }

    CcuInsHandle ccuInstance = 0;
    uint32_t instanceNum = 1;
    CHK_RET(HcclCommQueryCcuIns(comm, &ccuInstance, &instanceNum));
    if (instanceNum != 1 || ccuInstance == 0) {
        HCCL_ERROR("Unexpected CCU instance count %u", instanceNum);
        return HCCL_E_UNAVAIL;
    }

    CHK_RET_CCU(HcommCcuKernelRegisterStart(ccuInstance));
    CcuKernelHandle intraScatterKernel = 0;
    CcuKernelHandle interTransferKernel = 0;
    CcuKernelHandle intraAllgatherKernel = 0;
    char kernelName[64]{};

    if (rankOnRootServer) {
        (void)snprintf(kernelName, sizeof(kernelName), "HcclBcastV13IS_%u_die%u",
            param.myRank, intraDie);
        const void *intraScatterArgs[] = {intraScatterArg.get()};
        CHK_RET_CCU(HcommCcuKernelRegister(ccuInstance, 0, kernelName,
            reinterpret_cast<const void *>(ops_hccl::CcuHierarchicalKernel),
            intraScatterArgs, 1, &intraScatterKernel));
    }

    (void)snprintf(kernelName, sizeof(kernelName), "HcclBcastV13IT_%u_die%u",
        param.myRank, interDie);
    const void *interTransferArgs[] = {interTransferArg.get()};
    CHK_RET_CCU(HcommCcuKernelRegister(ccuInstance, 0, kernelName,
        reinterpret_cast<const void *>(ops_hccl::CcuHierarchicalKernel),
        interTransferArgs, 1, &interTransferKernel));

    if (rankOnRootServer) {
        (void)snprintf(kernelName, sizeof(kernelName), "HcclBcastV13IG_%u_die%u",
            param.myRank, intraDie);
        const void *intraAllgatherArgs[] = {intraAllgatherArg.get()};
        CHK_RET_CCU(HcommCcuKernelRegister(ccuInstance, 0, kernelName,
            reinterpret_cast<const void *>(ops_hccl::CcuHierarchicalKernel),
            intraAllgatherArgs, 1, &intraAllgatherKernel));
    }
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(ccuInstance));

    resource.ccuThread = param.cpuThread;
    resource.ccuKernels.clear();
    resource.threads.clear();
    if (rankOnRootServer) {
        resource.ccuKernels = {intraScatterKernel, interTransferKernel, intraAllgatherKernel};
        resource.threads.resize(1);
        CHK_RET(HcclThreadAcquire(comm, COMM_ENGINE_CCU, 1,
            HIERARCHICAL_WORKER_NOTIFY_COUNT, resource.threads.data()));
    } else {
        resource.ccuKernels.push_back(interTransferKernel);
    }
    HCCL_INFO("Broadcast v13 hierarchical rank=%u root=%u intraDie=%u interDie=%u intraPeers=%zu interPeers=%zu",
        param.myRank, param.root, intraDie, interDie, intraPeers.size(), interPeers.size());
    return HCCL_SUCCESS;
}

HcclResult CreateBalancedFusedResource(
    HcclComm comm, const OpParam &param, AlgResourceCtx &resource)
{
    if (param.rankSize != 12 || param.root != 0 ||
        param.myRank >= param.rankSize) {
        HCCL_ERROR("Invalid balanced-fused configuration: rank=%u rankSize=%u root=%u",
            param.myRank, param.rankSize, param.root);
        return HCCL_E_PARA;
    }

    const bool rankOnSourceServer = IsFirstServerRank(param.myRank);
    std::vector<uint32_t> intraPeers;
    std::vector<uint32_t> interPeers;
    CHK_RET(BuildHierarchicalIntraPeers(
        param.rankSize, param.myRank, intraPeers));
    CHK_RET(BuildHierarchicalInterPeers(
        param.rankSize, param.root, param.myRank, interPeers));

    std::vector<ChannelHandle> intraChannels;
    std::vector<ChannelHandle> interChannels;
    EndpointAttrDieId intraDie = UINT32_MAX;
    EndpointAttrDieId interDie = UINT32_MAX;
    CHK_RET(AcquireLayerPeerChannels(comm, param, intraPeers,
        INTRA_SERVER_LAYER, intraChannels, intraDie));
    CHK_RET(AcquireLayerPeerChannels(comm, param, interPeers,
        INTER_SERVER_LAYER, interChannels, interDie));
    if (intraDie == interDie) {
        HCCL_ERROR("Balanced-fused expected distinct layer dies: rank=%u die=%u",
            param.myRank, intraDie);
        return HCCL_E_UNAVAIL;
    }

    auto fusedArg = MakePeerKernelArg(
        BroadcastKernelMode::BALANCED_INTER_FUSED,
        param, interPeers, interChannels);
    std::shared_ptr<CcuKernelArgBase> firstArg;
    std::shared_ptr<CcuKernelArgBase> finalArg;
    if (rankOnSourceServer) {
        firstArg = MakePeerKernelArg(
            BroadcastKernelMode::HIERARCHICAL_INTRA_SCATTER,
            param, intraPeers, intraChannels);
        finalArg = MakePeerKernelArg(
            BroadcastKernelMode::HIERARCHICAL_INTRA_ALLGATHER,
            param, intraPeers, intraChannels);
    } else {
        const auto rootIter =
            std::find(interPeers.begin(), interPeers.end(), param.root);
        if (rootIter == interPeers.end()) {
            HCCL_ERROR("Balanced-fused remote rank %u has no root peer",
                param.myRank);
            return HCCL_E_INTERNAL;
        }
        const size_t rootPeerIdx =
            static_cast<size_t>(rootIter - interPeers.begin());
        const std::vector<uint32_t> rootPeer{param.root};
        const std::vector<ChannelHandle> rootChannel{
            interChannels[rootPeerIdx]};
        firstArg = MakePeerKernelArg(
            BroadcastKernelMode::BALANCED_INTER_SCATTER_RECV,
            param, rootPeer, rootChannel);
        finalArg = MakePeerKernelArg(
            BroadcastKernelMode::CROSS_FIRST_INTRA_ALLGATHER,
            param, intraPeers, intraChannels);
    }

    CcuInsHandle ccuInstance = 0;
    uint32_t instanceNum = 1;
    CHK_RET(HcclCommQueryCcuIns(comm, &ccuInstance, &instanceNum));
    if (instanceNum != 1 || ccuInstance == 0) {
        HCCL_ERROR("Unexpected CCU instance count %u", instanceNum);
        return HCCL_E_UNAVAIL;
    }

    CHK_RET_CCU(HcommCcuKernelRegisterStart(ccuInstance));
    CcuKernelHandle firstKernel = 0;
    CcuKernelHandle fusedKernel = 0;
    CcuKernelHandle finalKernel = 0;
    char kernelName[64]{};

    (void)snprintf(kernelName, sizeof(kernelName),
        "HcclBcastV33F0_r%u_d%u", param.myRank,
        rankOnSourceServer ? intraDie : interDie);
    const void *firstArgs[] = {firstArg.get()};
    const void *firstFunction = rankOnSourceServer ?
        reinterpret_cast<const void *>(ops_hccl::CcuHierarchicalKernel) :
        reinterpret_cast<const void *>(ops_hccl::CcuBalancedPlaneKernel);
    CHK_RET_CCU(HcommCcuKernelRegister(ccuInstance, 0, kernelName,
        firstFunction, firstArgs, 1, &firstKernel));

    (void)snprintf(kernelName, sizeof(kernelName),
        "HcclBcastV33F1_r%u_d%u", param.myRank, interDie);
    const void *fusedArgs[] = {fusedArg.get()};
    CHK_RET_CCU(HcommCcuKernelRegister(ccuInstance, 0, kernelName,
        reinterpret_cast<const void *>(ops_hccl::CcuBalancedPlaneKernel),
        fusedArgs, 1, &fusedKernel));

    (void)snprintf(kernelName, sizeof(kernelName),
        "HcclBcastV33F2_r%u_d%u", param.myRank, intraDie);
    const void *finalArgs[] = {finalArg.get()};
    CHK_RET_CCU(HcommCcuKernelRegister(ccuInstance, 0, kernelName,
        reinterpret_cast<const void *>(ops_hccl::CcuHierarchicalKernel),
        finalArgs, 1, &finalKernel));
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(ccuInstance));

    resource.ccuThread = param.cpuThread;
    resource.ccuKernels = {firstKernel, fusedKernel, finalKernel};
    resource.threads.resize(1);
    CHK_RET(HcclThreadAcquire(comm, COMM_ENGINE_CCU, 1,
        BALANCED_FUSED_WORKER_NOTIFY_COUNT, resource.threads.data()));
    HCCL_INFO("Broadcast v33 balanced-fused rank=%u intraDie=%u interDie=%u intraPeers=%zu interPeers=%zu",
        param.myRank, intraDie, interDie,
        intraPeers.size(), interPeers.size());
    return HCCL_SUCCESS;
}

HcclResult CreateParallelMeshNhrResource(
    HcclComm comm, const OpParam &param, AlgResourceCtx &resource)
{
    if (param.rankSize != 2 * FIRST_SERVER_SIZE || param.root != 0 ||
        param.myRank >= param.rankSize) {
        HCCL_ERROR("Invalid parallel Mesh-NHR Broadcast configuration: rank=%u rankSize=%u root=%u",
            param.myRank, param.rankSize, param.root);
        return HCCL_E_PARA;
    }

    std::vector<uint32_t> intraPeers;
    CHK_RET(BuildHierarchicalIntraPeers(param.rankSize, param.myRank, intraPeers));
    const uint32_t pairedRank = IsFirstServerRank(param.myRank) ?
        param.myRank + FIRST_SERVER_SIZE : param.myRank - FIRST_SERVER_SIZE;
    const std::vector<uint32_t> interPeers{pairedRank};

    std::vector<ChannelHandle> intraChannels;
    std::vector<ChannelHandle> interChannels;
    EndpointAttrDieId intraDie = UINT32_MAX;
    EndpointAttrDieId interDie = UINT32_MAX;
    CHK_RET(AcquireLayerPeerChannels(comm, param, intraPeers, INTRA_SERVER_LAYER,
        intraChannels, intraDie));
    CHK_RET(AcquireLayerPeerChannels(comm, param, interPeers, INTER_SERVER_LAYER,
        interChannels, interDie));
    if (intraDie == interDie) {
        HCCL_ERROR("Parallel Mesh-NHR Broadcast expected distinct layer dies: rank=%u die=%u",
            param.myRank, intraDie);
        return HCCL_E_UNAVAIL;
    }

    auto intraArg = MakePeerKernelArg(
        BroadcastKernelMode::PARALLEL_INTRA_SAG, param, intraPeers, intraChannels);
    auto interArg = MakePeerKernelArg(
        BroadcastKernelMode::TWO_SHOT_INTER_PAIRED, param, interPeers, interChannels);

    CcuInsHandle ccuInstance = 0;
    uint32_t instanceNum = 1;
    CHK_RET(HcclCommQueryCcuIns(comm, &ccuInstance, &instanceNum));
    if (instanceNum != 1 || ccuInstance == 0) {
        HCCL_ERROR("Unexpected CCU instance count %u", instanceNum);
        return HCCL_E_UNAVAIL;
    }

    CHK_RET_CCU(HcommCcuKernelRegisterStart(ccuInstance));
    CcuKernelHandle intraKernel = 0;
    CcuKernelHandle interKernel = 0;
    char kernelName[64]{};

    (void)snprintf(kernelName, sizeof(kernelName), "HcclBcastV20M_%u_die%u",
        param.myRank, intraDie);
    const void *intraArgs[] = {intraArg.get()};
    CHK_RET_CCU(HcommCcuKernelRegister(ccuInstance, 0, kernelName,
        reinterpret_cast<const void *>(ops_hccl::CcuHierarchicalKernel),
        intraArgs, 1, &intraKernel));

    (void)snprintf(kernelName, sizeof(kernelName), "HcclBcastV20N_%u_die%u",
        param.myRank, interDie);
    const void *interArgs[] = {interArg.get()};
    CHK_RET_CCU(HcommCcuKernelRegister(ccuInstance, 0, kernelName,
        reinterpret_cast<const void *>(ops_hccl::CcuTwoShotKernel),
        interArgs, 1, &interKernel));
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(ccuInstance));

    resource.ccuThread = param.cpuThread;
    resource.ccuKernels = {intraKernel, interKernel};
    resource.threads.resize(1);
    CHK_RET(HcclThreadAcquire(comm, COMM_ENGINE_CCU, 1,
        PARALLEL_WORKER_NOTIFY_COUNT, resource.threads.data()));
    HCCL_INFO("Broadcast v20 parallel Mesh-NHR rank=%u intraDie=%u interDie=%u pairedRank=%u",
        param.myRank, intraDie, interDie, pairedRank);
    return HCCL_SUCCESS;
}

HcclResult CreateRelayResource(
    HcclComm comm, const OpParam &param, AlgResourceCtx &resource)
{
    if ((param.rankSize != 12 && param.rankSize != 16) || param.root != 0) {
        HCCL_ERROR("Invalid relay configuration: rank=%u rankSize=%u root=%u",
            param.myRank, param.rankSize, param.root);
        return HCCL_E_PARA;
    }

    const bool rankOnSourceServer = IsFirstServerRank(param.myRank);
    std::vector<uint32_t> intraPeers;
    std::vector<uint32_t> interPeers;
    CHK_RET(BuildHierarchicalIntraPeers(param.rankSize, param.myRank, intraPeers));

    const uint32_t remoteRankCount = param.rankSize - FIRST_SERVER_SIZE;
    const uint32_t slicesPerRemoteRank = FIRST_SERVER_SIZE / remoteRankCount;
    if (rankOnSourceServer) {
        const uint32_t sliceIndex =
            (param.myRank + FIRST_SERVER_SIZE - param.root) % FIRST_SERVER_SIZE;
        interPeers.push_back(
            FIRST_SERVER_SIZE + sliceIndex / slicesPerRemoteRank);
    } else {
        const uint32_t firstSlice =
            (param.myRank - FIRST_SERVER_SIZE) * slicesPerRemoteRank;
        for (uint32_t sliceOffset = 0; sliceOffset < slicesPerRemoteRank; ++sliceOffset) {
            interPeers.push_back(
                (param.root + firstSlice + sliceOffset) % FIRST_SERVER_SIZE);
        }
    }

    std::vector<ChannelHandle> intraChannels;
    std::vector<ChannelHandle> interChannels;
    EndpointAttrDieId intraDie = UINT32_MAX;
    EndpointAttrDieId interDie = UINT32_MAX;
    CHK_RET(AcquireLayerPeerChannels(comm, param, intraPeers, INTRA_SERVER_LAYER,
        intraChannels, intraDie));
    CHK_RET(AcquireLayerPeerChannels(comm, param, interPeers, INTER_SERVER_LAYER,
        interChannels, interDie));
    if (intraDie == interDie) {
        HCCL_ERROR("Relay Broadcast expected distinct layer dies: rank=%u die=%u",
            param.myRank, intraDie);
        return HCCL_E_UNAVAIL;
    }

    std::shared_ptr<CcuKernelArgBase> intraScatterArg;
    if (rankOnSourceServer) {
        intraScatterArg = MakePeerKernelArg(
            BroadcastKernelMode::HIERARCHICAL_INTRA_SCATTER,
            param, intraPeers, intraChannels);
    }
    auto interRelayArg = MakePeerKernelArg(
        BroadcastKernelMode::RELAY_INTER_TRANSFER,
        param, interPeers, interChannels);
    auto intraAllgatherArg = MakePeerKernelArg(
        rankOnSourceServer ? BroadcastKernelMode::HIERARCHICAL_INTRA_ALLGATHER :
            BroadcastKernelMode::CROSS_FIRST_INTRA_ALLGATHER,
        param, intraPeers, intraChannels);

    CcuInsHandle ccuInstance = 0;
    uint32_t instanceNum = 1;
    CHK_RET(HcclCommQueryCcuIns(comm, &ccuInstance, &instanceNum));
    if (instanceNum != 1 || ccuInstance == 0) {
        HCCL_ERROR("Unexpected CCU instance count %u", instanceNum);
        return HCCL_E_UNAVAIL;
    }

    CHK_RET_CCU(HcommCcuKernelRegisterStart(ccuInstance));
    CcuKernelHandle intraScatterKernel = 0;
    CcuKernelHandle interRelayKernel = 0;
    CcuKernelHandle intraAllgatherKernel = 0;
    char kernelName[64]{};
    if (intraScatterArg != nullptr) {
        (void)snprintf(kernelName, sizeof(kernelName), "HcclBcastV16RS_%u_die%u",
            param.myRank, intraDie);
        const void *kernelArgs[] = {intraScatterArg.get()};
        CHK_RET_CCU(HcommCcuKernelRegister(ccuInstance, 0, kernelName,
            reinterpret_cast<const void *>(ops_hccl::CcuHierarchicalKernel),
            kernelArgs, 1, &intraScatterKernel));
    }
    (void)snprintf(kernelName, sizeof(kernelName), "HcclBcastV16RT_%u_die%u",
        param.myRank, interDie);
    const void *interKernelArgs[] = {interRelayArg.get()};
    CHK_RET_CCU(HcommCcuKernelRegister(ccuInstance, 0, kernelName,
        reinterpret_cast<const void *>(ops_hccl::CcuHierarchicalKernel),
        interKernelArgs, 1, &interRelayKernel));

    (void)snprintf(kernelName, sizeof(kernelName), "HcclBcastV16RG_%u_die%u",
        param.myRank, intraDie);
    const void *intraAllgatherArgs[] = {intraAllgatherArg.get()};
    CHK_RET_CCU(HcommCcuKernelRegister(ccuInstance, 0, kernelName,
        reinterpret_cast<const void *>(ops_hccl::CcuHierarchicalKernel),
        intraAllgatherArgs, 1, &intraAllgatherKernel));
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(ccuInstance));

    resource.ccuThread = param.cpuThread;
    resource.ccuKernels.clear();
    resource.threads.clear();
    if (rankOnSourceServer) {
        resource.ccuKernels = {
            intraScatterKernel, interRelayKernel, intraAllgatherKernel};
    } else {
        resource.ccuKernels = {interRelayKernel, intraAllgatherKernel};
    }
    resource.threads.resize(1);
    CHK_RET(HcclThreadAcquire(comm, COMM_ENGINE_CCU, 1,
        RELAY_WORKER_NOTIFY_COUNT, resource.threads.data()));
    HCCL_INFO("Broadcast v16 relay rank=%u intraDie=%u interDie=%u intraPeers=%zu interPeers=%zu",
        param.myRank, intraDie, interDie, intraPeers.size(), interPeers.size());
    return HCCL_SUCCESS;
}

HcclResult CreateScatterAllgatherResource(
    HcclComm comm, const OpParam &param, AlgResourceCtx &resource)
{
    std::vector<uint32_t> peers;
    CHK_RET(BuildScatterPeers(param.rankSize, param.myRank, peers));
    if (peers.empty() || peers.size() > MAX_RANK_SIZE) {
        HCCL_ERROR("Invalid scatter-allgather peer count %zu for rank %u", peers.size(), param.myRank);
        return HCCL_E_INTERNAL;
    }

    std::vector<HcclChannelDesc> channelDescs(peers.size());
    std::vector<EndpointAttrDieId> localDies(peers.size(), UINT32_MAX);
    for (size_t peerIdx = 0; peerIdx < peers.size(); ++peerIdx) {
        CHK_RET(SelectCrossServerChannelDesc(comm, param.myRank, peers[peerIdx],
            channelDescs[peerIdx], localDies[peerIdx]));
    }
    for (size_t peerIdx = 1; peerIdx < localDies.size(); ++peerIdx) {
        if (localDies[peerIdx] != localDies[0]) {
            HCCL_ERROR("Scatter-allgather links selected different local dies: rank=%u firstDie=%u die=%u",
                param.myRank, localDies[0], localDies[peerIdx]);
            return HCCL_E_UNAVAIL;
        }
    }

    std::vector<ChannelHandle> channels(peers.size());
    CHK_RET(HcclChannelAcquire(comm, COMM_ENGINE_CCU, channelDescs.data(),
        static_cast<uint32_t>(channelDescs.size()), channels.data()));

    auto kernelArg = std::make_shared<CcuKernelArgBase>();
    // v33 merge: the 4x1 large-message path now runs v27's chain pipeline.
    kernelArg->mode = param.rankSize == 4 ? BroadcastKernelMode::PIPELINED_CHAIN4 :
        BroadcastKernelMode::BIPARTITE_SCATTER_ALLGATHER;
    kernelArg->myRank = param.myRank;
    kernelArg->rankSize = param.rankSize;
    kernelArg->root = param.root;
    kernelArg->peerCount = static_cast<uint32_t>(peers.size());
    for (size_t peerIdx = 0; peerIdx < peers.size(); ++peerIdx) {
        kernelArg->peerChannels[peerIdx] = channels[peerIdx];
        kernelArg->peerRanks[peerIdx] = peers[peerIdx];
    }

    CcuInsHandle ccuInstance = 0;
    uint32_t instanceNum = 1;
    CHK_RET(HcclCommQueryCcuIns(comm, &ccuInstance, &instanceNum));
    if (instanceNum != 1 || ccuInstance == 0) {
        HCCL_ERROR("Unexpected CCU instance count %u", instanceNum);
        return HCCL_E_UNAVAIL;
    }

    CHK_RET_CCU(HcommCcuKernelRegisterStart(ccuInstance));
    char kernelName[64]{};
    (void)snprintf(kernelName, sizeof(kernelName), "HcclBcastV13%c_%u_die%u",
        param.rankSize == 4 ? 'C' : 'B', param.myRank, localDies[0]);
    CcuKernelHandle kernelHandle = 0;
    const void *registerArgs[] = {kernelArg.get()};
    CHK_RET_CCU(HcommCcuKernelRegister(ccuInstance, 0, kernelName,
        reinterpret_cast<const void *>(ops_hccl::CcuScatterAllgatherKernel), registerArgs, 1,
        &kernelHandle));
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(ccuInstance));

    resource.ccuThread = param.cpuThread;
    resource.threads.clear();
    resource.ccuKernels.clear();
    resource.ccuKernels.push_back(kernelHandle);
    HCCL_INFO("Broadcast v13 %s scatter-allgather rank=%u die=%u peers=%zu root=%u",
        param.rankSize == 4 ? "chain4" : "bipartite", param.myRank, localDies[0], peers.size(),
        param.root);
    return HCCL_SUCCESS;
}

} // namespace

HcclResult HcclBroadcast(
    void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(buf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param{};
    param.inputPtr = buf;
    param.outputPtr = buf;
    param.count = count;
    param.dataType = dataType;
    param.root = root;
    param.opType = HcclCMDType::HCCL_CMD_BROADCAST;

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    if (param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || root >= param.rankSize) {
        HCCL_ERROR("Invalid rank configuration: rankSize=%u root=%u", param.rankSize, root);
        return HCCL_E_PARA;
    }
    if (count == 0 || param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    const auto sizeIter = SIZE_TABLE.find(dataType);
    if (sizeIter == SIZE_TABLE.end()) {
        HCCL_ERROR("Unsupported Broadcast data type %d", static_cast<int32_t>(dataType));
        return HCCL_E_NOT_SUPPORT;
    }
    if (count > std::numeric_limits<uint64_t>::max() / sizeIter->second) {
        HCCL_ERROR("Broadcast byte count overflows uint64_t");
        return HCCL_E_PARA;
    }
    const uint64_t totalBytes = count * sizeIter->second;
    const bool useSmallDualDieDirect = totalBytes <= SMALL_MESSAGE_BYTES &&
        (param.rankSize == 12 || param.rankSize == 16) && param.root < FIRST_SERVER_SIZE;
    const bool useDualStripePipeline = totalBytes > SMALL_MESSAGE_BYTES &&
        param.rankSize == 2 * FIRST_SERVER_SIZE && param.root == 0;
    // v33 merge: 4x1 large messages run v27's chain4 pipeline (recovers 23/24).
    const bool useChain4 = totalBytes > SMALL_MESSAGE_BYTES && param.rankSize == 4;
    const bool useStaticPipeline = totalBytes > SMALL_MESSAGE_BYTES &&
        param.rankSize == 2 * FIRST_SERVER_SIZE &&
        !useDualStripePipeline;
    // V25 remains the platform-fastest path for 2x8 large messages.
    const bool useParallelMeshNhr = false;
    const bool useRelay = false;
    const bool useBalancedFused =
        totalBytes >= BROADCAST_LARGE_MESSAGE_THRESHOLD &&
        param.rankSize == 12 && param.root == 0;
    const bool useHierarchical = totalBytes >= BROADCAST_LARGE_MESSAGE_THRESHOLD &&
        param.rankSize == 12 && param.root < FIRST_SERVER_SIZE;
    const bool useScatterAllgather = totalBytes >= BROADCAST_LARGE_MESSAGE_THRESHOLD &&
        param.rankSize == 12;

    const char *algorithmTag = useSmallDualDieDirect ? "small_dualdie_parallel" :
        useDualStripePipeline ? "dual_stripe_pipeline" :
        useStaticPipeline ? "static_v25" :
        useParallelMeshNhr ? "parallel_mesh_nhr_2x8" :
        useRelay ? "relay_sag" :
        useBalancedFused ? "balanced_fused_v33" :
        useChain4 ? "chain4_pipeline16" :
        useHierarchical ? "hier_sag_w2" : !useScatterAllgather ? "tree" :
        "bipartite_sag";
    if (useDualStripePipeline || useStaticPipeline) {
        // Static CCU instructions embed the chunk count and tail length.
        (void)snprintf(param.tag, sizeof(param.tag),
            "hccl_custom_broadcast_v29_%s_count_%llu_root_%u", algorithmTag,
            static_cast<unsigned long long>(count), root);
    } else {
        (void)snprintf(param.tag, sizeof(param.tag), "hccl_custom_broadcast_v29_%s_root_%u",
            algorithmTag, root);
    }

    const uint32_t streamNotifyNum =
        (useSmallDualDieDirect && SMALL_DUAL_DIE_PARALLEL && param.myRank == param.root) ?
            SMALL_DIRECT_WORKER_NOTIFY_COUNT :
        useDualStripePipeline ? DUAL_STRIPE_WORKER_NOTIFY_COUNT :
        useParallelMeshNhr ? PARALLEL_WORKER_NOTIFY_COUNT :
        useRelay ? RELAY_WORKER_NOTIFY_COUNT :
        useBalancedFused ? BALANCED_FUSED_WORKER_NOTIFY_COUNT :
        (useHierarchical && param.myRank < FIRST_SERVER_SIZE) ?
            HIERARCHICAL_WORKER_NOTIFY_COUNT : 0;
    CHK_RET(HcclThreadAcquireWithStream(comm, COMM_ENGINE_CCU, stream, streamNotifyNum, &param.cpuThread));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, COMM_ENGINE_CCU, &ctx, &size) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = size;
    } else {
        AlgResourceCtx resource{};
        if (useSmallDualDieDirect) {
            CHK_RET(CreateSmallDualDieDirectResource(comm, param, resource));
        } else if (useDualStripePipeline) {
            CHK_RET(CreateDualStripePipelineResource(comm, param, resource));
        } else if (useStaticPipeline) {
            CHK_RET(CreateTreeResource(comm, param, true, resource));
        } else if (useParallelMeshNhr) {
            CHK_RET(CreateParallelMeshNhrResource(comm, param, resource));
        } else if (useRelay) {
            CHK_RET(CreateRelayResource(comm, param, resource));
        } else if (useBalancedFused) {
            CHK_RET(CreateBalancedFusedResource(comm, param, resource));
        } else if (useHierarchical) {
            CHK_RET(CreateHierarchicalResource(comm, param, resource));
        } else if (useChain4 || useScatterAllgather) {
            CHK_RET(CreateScatterAllgatherResource(comm, param, resource));
        } else {
            CHK_RET(CreateTreeResource(comm, param, false, resource));
        }

        std::vector<char> sequence = resource.Serialize();
        param.ctxSize = sequence.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, COMM_ENGINE_CCU, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, COMM_ENGINE_CCU, param.tag, sequence.data(), sequence.size(), 0));
    }

    CHK_RET(ops_hccl::ExecOp(param));
    return HCCL_SUCCESS;
}

// v30 marker: V29 hybrid paths plus a two-chunk 8+4 hierarchy window.
