/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>
#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>
#include <hcomm/hcomm_primitives.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>
#include "common.h"
#include "custom.h"
#include "exec_op.h"
#include "hccl.h"
#include "log.h"
#include "ccu_kernel.h"
namespace {
constexpr uint64_t COUNT_512KB_FP32 = 131072;
constexpr uint64_t COUNT_512MB_FP32 = 134217728;
constexpr uint64_t COUNT_400MB_PLUS_4B_FP32 = 104857601;
constexpr uint32_t TOPOLOGY_LAYER = 0;
constexpr uint32_t CROSS_GROUP_LAYER = 1;
struct DiscoveredTopologyGroup {
uint32_t topoInstId = 0;
std::vector<uint32_t> ranks;
};
HcclResult Phase0Fail(const char *stage, HcclResult result)
{
std::printf("[PHASE0_FAIL] stage=%s\n", stage);
return result;
}
HcclResult Phase0FailCcu(const char *stage, CcuResult result)
{
return Phase0Fail(stage, ConvertCcuToHccl(result));
}
bool IsSupportedElementCount(uint64_t count)
{
return count == COUNT_512KB_FP32 || count == COUNT_512MB_FP32 || count == COUNT_400MB_PLUS_4B_FP32;
}
HcclResult ValidateCompetitionInput(uint64_t count, HcclDataType dataType, HcclReduceOp op)
{
CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32,
    HCCL_ERROR("[HcclAllReduce] Unsupported data type [%d]; only FP32 is accepted", static_cast<int32_t>(dataType)),
    HCCL_E_NOT_SUPPORT);
CHK_PRT_RET(count > std::numeric_limits<uint64_t>::max() / sizeof(float),
    HCCL_ERROR("[HcclAllReduce] Data size overflow, count [%llu]", static_cast<unsigned long long>(count)),
    HCCL_E_PARA);
CHK_PRT_RET(!IsSupportedElementCount(count),
    HCCL_ERROR("[HcclAllReduce] Unsupported FP32 element count [%llu]; expected 131072, 134217728, or "
               "104857601",
        static_cast<unsigned long long>(count)),
    HCCL_E_NOT_SUPPORT);
CHK_PRT_RET(op != HCCL_REDUCE_SUM,
    HCCL_ERROR("[HcclAllReduce] Unsupported reduction op [%d]; only SUM is accepted", static_cast<int32_t>(op)),
    HCCL_E_NOT_SUPPORT);
return HCCL_SUCCESS;
}
bool GroupOrder(const DiscoveredTopologyGroup &lhs, const DiscoveredTopologyGroup &rhs)
{
if (lhs.ranks.size() != rhs.ranks.size()) {
    return lhs.ranks.size() > rhs.ranks.size();
}
if (lhs.ranks.front() != rhs.ranks.front()) {
    return lhs.ranks.front() < rhs.ranks.front();
}
return lhs.topoInstId < rhs.topoInstId;
}
HcclResult ClassifyExactTopology(
uint32_t rankSize, const std::vector<DiscoveredTopologyGroup> &groups, TopologyKind &topology)
{
topology = TopologyKind::UNSUPPORTED;
if (rankSize == 4 && groups.size() == 4
    && std::all_of(groups.begin(), groups.end(), [](const DiscoveredTopologyGroup &group) {
           return group.ranks.size() == 1;
       })) {
    topology = TopologyKind::TOPOLOGY_4X1;
} else if (rankSize == 12 && groups.size() == 2 && groups[0].ranks.size() == 8 && groups[1].ranks.size() == 4) {
    topology = TopologyKind::TOPOLOGY_8_PLUS_4;
} else if (rankSize == 16 && groups.size() == 2 && groups[0].ranks.size() == 8 && groups[1].ranks.size() == 8) {
    topology = TopologyKind::TOPOLOGY_2X8;
}
CHK_PRT_RET(topology == TopologyKind::UNSUPPORTED,
    HCCL_ERROR("[HcclAllReduce] Unsupported layer-0 topology layout: rankSize [%u], groupCount [%zu]", rankSize,
        groups.size()),
    HCCL_E_NOT_SUPPORT);
return HCCL_SUCCESS;
}
HcclResult DiscoverCanonicalTopology(
HcclComm comm, uint32_t currentRank, uint32_t rankSize, TopologyKind &topology, CanonicalRankGroupMap &rankMap)
{
CHK_PRT_RET(rankSize == 0 || rankSize > MAX_CANONICAL_RANKS || currentRank >= rankSize,
    HCCL_ERROR("[HcclAllReduce] Unsupported communicator rank metadata: rank [%u], rankSize [%u]", currentRank,
        rankSize),
    HCCL_E_NOT_SUPPORT);
uint32_t *instanceSizes = nullptr;
uint32_t instanceCount = 0;
CHK_RET(HcclRankGraphGetInstSizeListByLayer(comm, TOPOLOGY_LAYER, &instanceSizes, &instanceCount));
CHK_PRT_RET(instanceCount == 0 || instanceCount > MAX_CANONICAL_GROUPS || instanceSizes == nullptr,
    HCCL_ERROR("[HcclAllReduce] Invalid global layer-0 instance-size list, count [%u]", instanceCount),
    HCCL_E_NOT_SUPPORT);
std::vector<uint32_t> sortedInstanceSizes(instanceSizes, instanceSizes + instanceCount);
CHK_PRT_RET(std::any_of(sortedInstanceSizes.begin(), sortedInstanceSizes.end(), [rankSize](uint32_t size) {
                return size == 0 || size > rankSize;
            }),
    HCCL_ERROR("[HcclAllReduce] Invalid rank count in global layer-0 instance-size list"), HCCL_E_NOT_SUPPORT);
const uint64_t coveredRanks = std::accumulate(
    sortedInstanceSizes.begin(), sortedInstanceSizes.end(), static_cast<uint64_t>(0));
CHK_PRT_RET(coveredRanks != rankSize,
    HCCL_ERROR("[HcclAllReduce] Global layer-0 instances cover [%llu] ranks, communicator has [%u]",
        static_cast<unsigned long long>(coveredRanks), rankSize),
    HCCL_E_NOT_SUPPORT);
std::sort(sortedInstanceSizes.begin(), sortedInstanceSizes.end(), std::greater<uint32_t>());
uint32_t *localRanks = nullptr;
uint32_t localRankCount = 0;
CHK_RET(HcclRankGraphGetRanksByLayer(comm, TOPOLOGY_LAYER, &localRanks, &localRankCount));
CHK_PRT_RET(localRankCount == 0 || localRankCount > rankSize || localRanks == nullptr,
    HCCL_ERROR("[HcclAllReduce] Invalid current-rank layer-0 membership, count [%u]", localRankCount),
    HCCL_E_NOT_SUPPORT);
std::vector<uint32_t> localGroup(localRanks, localRanks + localRankCount);
std::sort(localGroup.begin(), localGroup.end());
CHK_PRT_RET(std::adjacent_find(localGroup.begin(), localGroup.end()) != localGroup.end()
                || localGroup.back() >= rankSize
                || !std::binary_search(localGroup.begin(), localGroup.end(), currentRank),
    HCCL_ERROR("[HcclAllReduce] Invalid current-rank layer-0 membership for rank [%u]", currentRank),
    HCCL_E_NOT_SUPPORT);
std::vector<DiscoveredTopologyGroup> groups;
groups.reserve(instanceCount);
if (rankSize == 4 && sortedInstanceSizes == std::vector<uint32_t>({1, 1, 1, 1})) {
    CHK_PRT_RET(localGroup.size() != 1,
        HCCL_ERROR("[HcclAllReduce] 4x1 topology requires singleton local layer-0 membership"),
        HCCL_E_NOT_SUPPORT);
    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        groups.push_back({rank, {rank}});
    }
} else if (instanceCount == 2) {
    std::vector<uint32_t> complement;
    complement.reserve(rankSize - localGroup.size());
    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        if (!std::binary_search(localGroup.begin(), localGroup.end(), rank)) {
            complement.push_back(rank);
        }
    }
    CHK_PRT_RET(complement.empty(),
        HCCL_ERROR("[HcclAllReduce] Two-group topology has an empty complementary layer-0 group"),
        HCCL_E_NOT_SUPPORT);
    groups.push_back({0, std::move(localGroup)});
    groups.push_back({1, std::move(complement)});
} else {
    HCCL_ERROR("[HcclAllReduce] Unsupported global layer-0 instance layout: rankSize [%u], instanceCount [%u]",
        rankSize, instanceCount);
    return HCCL_E_NOT_SUPPORT;
}
std::sort(groups.begin(), groups.end(), GroupOrder);
std::vector<uint32_t> reconstructedSizes;
reconstructedSizes.reserve(groups.size());
std::transform(groups.begin(), groups.end(), std::back_inserter(reconstructedSizes),
    [](const DiscoveredTopologyGroup &group) {
        return static_cast<uint32_t>(group.ranks.size());
    });
CHK_PRT_RET(reconstructedSizes != sortedInstanceSizes,
    HCCL_ERROR("[HcclAllReduce] Reconstructed layer-0 membership does not match global instance sizes"),
    HCCL_E_NOT_SUPPORT);
CHK_RET(ClassifyExactTopology(rankSize, groups, topology));
rankMap = CanonicalRankGroupMap{};
rankMap.rankCount = rankSize;
rankMap.groupCount = static_cast<uint32_t>(groups.size());
rankMap.canonicalToPhysical.fill(INVALID_VALUE_RANKID);
rankMap.physicalToCanonical.fill(INVALID_VALUE_RANKID);
rankMap.groupByCanonicalRank.fill(INVALID_VALUE_RANKID);
rankMap.groupLocalRank.fill(INVALID_VALUE_RANKID);
std::array<bool, MAX_CANONICAL_RANKS> rankSeen{};
uint32_t canonicalRank = 0;
for (uint32_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
    const DiscoveredTopologyGroup &group = groups[groupIndex];
    for (uint32_t localIndex = 0; localIndex < group.ranks.size(); ++localIndex) {
        const uint32_t physicalRank = group.ranks[localIndex];
        CHK_PRT_RET(rankSeen[physicalRank] || canonicalRank >= rankSize,
            HCCL_ERROR("[HcclAllReduce] Layer-0 topology rank membership is duplicated or oversized at rank [%u]",
                physicalRank),
            HCCL_E_NOT_SUPPORT);
        rankSeen[physicalRank] = true;
        rankMap.canonicalToPhysical[canonicalRank] = physicalRank;
        rankMap.physicalToCanonical[physicalRank] = canonicalRank;
        rankMap.groupByCanonicalRank[canonicalRank] = groupIndex;
        rankMap.groupLocalRank[canonicalRank] = localIndex;
        ++canonicalRank;
    }
}
CHK_PRT_RET(canonicalRank != rankSize
                || std::any_of(rankSeen.begin(), rankSeen.begin() + rankSize,
                    [](bool seen) {
                        return !seen;
                    }),
    HCCL_ERROR("[HcclAllReduce] Layer-0 topology covers [%u] unique ranks, communicator has [%u]", canonicalRank,
        rankSize),
    HCCL_E_NOT_SUPPORT);
CHK_PRT_RET(!custom_detail::IsValidRankMap(rankMap, topology),
    HCCL_ERROR("[HcclAllReduce] Canonical topology map failed consistency validation"), HCCL_E_INTERNAL);
return HCCL_SUCCESS;
}
bool RankMapsEqual(const CanonicalRankGroupMap &lhs, const CanonicalRankGroupMap &rhs)
{
if (lhs.rankCount != rhs.rankCount || lhs.groupCount != rhs.groupCount) {
    return false;
}
for (uint32_t rank = 0; rank < lhs.rankCount; ++rank) {
    if (lhs.canonicalToPhysical[rank] != rhs.canonicalToPhysical[rank]
        || lhs.physicalToCanonical[rank] != rhs.physicalToCanonical[rank]
        || lhs.groupByCanonicalRank[rank] != rhs.groupByCanonicalRank[rank]
        || lhs.groupLocalRank[rank] != rhs.groupLocalRank[rank]) {
        return false;
    }
}
return true;
}
HcclResult NormalizeEndpointDie(EndpointAttrDieId dieId, DieEndpoint &die)
{
if (dieId == 0) {
    die = DieEndpoint::DIE_0;
    return HCCL_SUCCESS;
}
if (dieId == 1) {
    die = DieEndpoint::DIE_1;
    return HCCL_SUCCESS;
}
HCCL_ERROR("[HcclAllReduce] Unsupported endpoint die ID [%u]; competition links require die 0 or 1", dieId);
return HCCL_E_NOT_SUPPORT;
}
HcclResult PopulatePeerEndpointMetadata(
HcclComm comm, uint32_t localRank, const CommLink &link, PeerChannelResource &peer)
{
peer.protocol = link.linkAttr.linkProtocol;
peer.localEndpoint = link.srcEndpointDesc;
peer.remoteEndpoint = link.dstEndpointDesc;
CHK_PRT_RET(peer.localEndpoint.protocol != peer.protocol || peer.remoteEndpoint.protocol != peer.protocol
                || peer.localEndpoint.commAddr.type == COMM_ADDR_TYPE_RESERVED
                || peer.remoteEndpoint.commAddr.type == COMM_ADDR_TYPE_RESERVED,
    HCCL_ERROR(
        "[HcclAllReduce] Peer [%u] link endpoint protocol/address metadata is inconsistent", peer.remoteRank),
    HCCL_E_NOT_SUPPORT);
CHK_PRT_RET(peer.localEndpoint.loc.locType != ENDPOINT_LOC_TYPE_DEVICE
                || peer.remoteEndpoint.loc.locType != ENDPOINT_LOC_TYPE_DEVICE,
    HCCL_ERROR("[HcclAllReduce] Peer [%u] link does not expose device endpoint locations", peer.remoteRank),
    HCCL_E_NOT_SUPPORT);
CHK_RET(HcclRankGraphGetEndpointInfo(comm, localRank, &peer.localEndpoint, ENDPOINT_ATTR_DIE_ID,
    sizeof(peer.localEndpointDieId), &peer.localEndpointDieId));
CHK_RET(HcclRankGraphGetEndpointInfo(comm, localRank, &peer.localEndpoint, ENDPOINT_ATTR_LOCATION,
    sizeof(peer.localEndpointLocation), &peer.localEndpointLocation));
peer.remoteEndpointLocation = static_cast<EndpointAttrLocation>(peer.remoteEndpoint.loc.locType);
CHK_PRT_RET(peer.localEndpointLocation == UINT32_MAX || peer.remoteEndpointLocation == UINT32_MAX,
    HCCL_ERROR("[HcclAllReduce] Peer [%u] link returned invalid endpoint location metadata", peer.remoteRank),
    HCCL_E_NOT_SUPPORT);
CHK_RET(NormalizeEndpointDie(peer.localEndpointDieId, peer.localDie));
// The current-rank API guarantees attributes only for the local endpoint. A remote Die is not needed to place a
// local CCU kernel, so keep it explicitly unavailable instead of inventing metadata.
peer.remoteEndpointDieId = UINT32_MAX;
peer.remoteDie = DieEndpoint::NOT_APPLICABLE;
return HCCL_SUCCESS;
}
HcclResult BuildRequiredPeerSet(
const AlgorithmPlan &plan, uint32_t localCanonical, uint32_t rankCount, CanonicalPeerSet &requiredPeers)
{
requiredPeers = CanonicalPeerSet{};
CHK_PRT_RET(localCanonical >= rankCount || rankCount > requiredPeers.mask.size(),
    HCCL_ERROR("[HcclAllReduce] Invalid canonical rank metadata for peer-set construction"), HCCL_E_PARA);
const bool supportedEightPlusFour
    = plan.topology == TopologyKind::TOPOLOGY_8_PLUS_4
      && (plan.family == AlgorithmFamily::DIRECT12_RSAG
          || plan.family == AlgorithmFamily::FULL_EXCHANGE_FIXED_TREE);
const bool supportedFourXOne
    = plan.topology == TopologyKind::TOPOLOGY_4X1
      && (plan.family == AlgorithmFamily::RING_RSAG
          || plan.family == AlgorithmFamily::FULL_EXCHANGE_FIXED_TREE);
const bool supportedTwoXEight
    = plan.topology == TopologyKind::TOPOLOGY_2X8
      && (plan.family == AlgorithmFamily::TOPOLOGY_BALANCED_DIRECT_RSAG
          || plan.family == AlgorithmFamily::BULK_PUSH_STAGED_REDUCE);
CHK_PRT_RET(!plan.implemented || (!supportedEightPlusFour && !supportedFourXOne && !supportedTwoXEight),
    HCCL_ERROR("[HcclAllReduce] Required peer set is not implemented for family [%u]",
        static_cast<uint32_t>(plan.family)),
    HCCL_E_NOT_SUPPORT);
if (plan.topology == TopologyKind::TOPOLOGY_4X1 && plan.family == AlgorithmFamily::RING_RSAG) {
    const uint32_t nextPeer = (localCanonical + 1) % rankCount;
    const uint32_t previousPeer = (localCanonical + rankCount - 1) % rankCount;
    CHK_PRT_RET(nextPeer == previousPeer || nextPeer == localCanonical || previousPeer == localCanonical,
        HCCL_ERROR("[HcclAllReduce] Invalid 4x1 Ring peer set"), HCCL_E_INTERNAL);
    requiredPeers.mask[nextPeer] = 1;
    requiredPeers.mask[previousPeer] = 1;
    requiredPeers.count = 2;
    return HCCL_SUCCESS;
}
for (uint32_t peerCanonical = 0; peerCanonical < rankCount; ++peerCanonical) {
    if (peerCanonical != localCanonical) {
        requiredPeers.mask[peerCanonical] = 1;
        ++requiredPeers.count;
    }
}
return HCCL_SUCCESS;
}
HcclResult AcquirePeerChannels(HcclComm comm, uint32_t localRank, TopologyKind topology,
const CanonicalRankGroupMap &rankMap, const CanonicalPeerSet &requiredPeers, AlgResourceCtx &resourceContext)
{
const uint32_t localCanonical = rankMap.physicalToCanonical[localRank];
CHK_PRT_RET(localCanonical >= rankMap.rankCount,
    HCCL_ERROR("[HcclAllReduce] Local rank [%u] is absent from canonical topology", localRank), HCCL_E_INTERNAL);
std::array<PeerChannelResource, MAX_PEER_CHANNELS> pendingPeers{};
std::array<HcclChannelDesc, MAX_PEER_CHANNELS> descriptors{};
std::array<ChannelHandle, MAX_PEER_CHANNELS> handles{};
uint32_t pendingPeerCount = 0;
for (uint32_t peerCanonical = 0; peerCanonical < rankMap.rankCount; ++peerCanonical) {
    if (!requiredPeers.mask[peerCanonical]) {
        continue;
    }
    PeerChannelResource peer;
    peer.canonicalPeer = peerCanonical;
    peer.remoteRank = rankMap.canonicalToPhysical[peerCanonical];
    peer.layer = custom_detail::ExpectedPeerLayer(topology, rankMap, localCanonical, peerCanonical);
    CHK_PRT_RET(peer.layer == FabricLayer::INVALID,
        HCCL_ERROR("[HcclAllReduce] Failed to derive the expected fabric layer for peer [%u]", peer.remoteRank),
        HCCL_E_INTERNAL);
    const uint32_t netLayer = peer.layer == FabricLayer::L0 ? TOPOLOGY_LAYER : CROSS_GROUP_LAYER;
    CommLink *links = nullptr;
    uint32_t linkNum = 0;
    CHK_RET(HcclRankGraphGetLinks(comm, netLayer, localRank, peer.remoteRank, &links, &linkNum));
    CHK_PRT_RET(linkNum == 0 || links == nullptr,
        HCCL_ERROR("[HcclAllReduce] Missing layer [%u] link from rank [%u] to peer [%u]", netLayer, localRank,
            peer.remoteRank),
        HCCL_E_NOT_SUPPORT);
    uint32_t supportedLinkCount = 0;
    CommLink selectedLink{};
    for (uint32_t linkIndex = 0; linkIndex < linkNum; ++linkIndex) {
        if (custom_detail::IsSupportedCcuProtocol(links[linkIndex].linkAttr.linkProtocol)) {
            // Some topology providers repeat descriptors for one physical Full-Mesh edge. Select one
            // descriptor deterministically and still acquire exactly one channel for this peer.
            if (supportedLinkCount == 0) {
                selectedLink = links[linkIndex];
            }
            ++supportedLinkCount;
        }
    }
    CHK_PRT_RET(supportedLinkCount == 0,
        HCCL_ERROR("[HcclAllReduce] No supported CCU link from rank [%u] to peer [%u] on layer [%u]",
            localRank, peer.remoteRank, netLayer),
        HCCL_E_NOT_SUPPORT);
    CHK_RET(PopulatePeerEndpointMetadata(comm, localRank, selectedLink, peer));
    CHK_PRT_RET(pendingPeerCount >= pendingPeers.size(),
        HCCL_ERROR("[HcclAllReduce] Peer descriptor count exceeds the fixed competition maximum"), HCCL_E_INTERNAL);
    for (uint32_t prior = 0; prior < pendingPeerCount; ++prior) {
        CHK_PRT_RET(pendingPeers[prior].remoteRank == peer.remoteRank
                        || pendingPeers[prior].canonicalPeer == peer.canonicalPeer,
            HCCL_ERROR("[HcclAllReduce] Duplicate peer rank or canonical rank [%u] before channel acquisition",
                peer.remoteRank),
            HCCL_E_INTERNAL);
    }
    HcclChannelDesc &descriptor = descriptors[pendingPeerCount];
    CHK_RET(HcclChannelDescInit(&descriptor, 1));
    descriptor.remoteRank = peer.remoteRank;
    descriptor.channelProtocol = peer.protocol;
    descriptor.localEndpoint = peer.localEndpoint;
    descriptor.remoteEndpoint = peer.remoteEndpoint;
    descriptor.notifyNum = CHANNEL_NOTIFY_COUNT;
    pendingPeers[pendingPeerCount++] = peer;
}
CHK_PRT_RET(pendingPeerCount != requiredPeers.count,
    HCCL_ERROR(
        "[HcclAllReduce] Prepared [%u] peer channels, expected [%u]", pendingPeerCount, requiredPeers.count),
    HCCL_E_INTERNAL);
CHK_RET(HcclChannelAcquire(comm, COMM_ENGINE_CCU, descriptors.data(), pendingPeerCount, handles.data()));
for (uint32_t index = 0; index < pendingPeerCount; ++index) {
    CHK_PRT_RET(handles[index] == 0,
        HCCL_ERROR("[HcclAllReduce] Batch CCU channel acquisition returned an invalid handle for peer [%u]",
            pendingPeers[index].remoteRank),
        HCCL_E_INTERNAL);
    for (uint32_t prior = 0; prior < index; ++prior) {
        CHK_PRT_RET(handles[prior] == handles[index],
            HCCL_ERROR("[HcclAllReduce] Batch CCU channel acquisition returned a duplicate handle for peer [%u]",
                pendingPeers[index].remoteRank),
            HCCL_E_INTERNAL);
    }
}
resourceContext.peerChannelCount = pendingPeerCount;
for (uint32_t index = 0; index < pendingPeerCount; ++index) {
    pendingPeers[index].handle = handles[index];
    resourceContext.peerChannels[index] = pendingPeers[index];
}
return HCCL_SUCCESS;
}
struct Direct12StaticArgs {
std::array<ops_hccl::Direct12DieStaticArg, 2> dieArgs{};
};
struct FullExchangeStaticArgs {
std::array<ops_hccl::FullExchangeDieStaticArg, 2> dieArgs{};
};
struct TwoXEightMultiRootStaticArgs {
std::array<ops_hccl::TwoXEightMultiRootStaticArg, 2> layerArgs{};
};
HcclResult BuildFullExchangeStaticArgs(const AlgResourceCtx &resourceContext, FullExchangeStaticArgs &args)
{
CHK_PRT_RET(resourceContext.rankMap.rankCount != DIRECT12_CANONICAL_LEAF_COUNT
                || resourceContext.localCanonicalRank >= DIRECT12_CANONICAL_LEAF_COUNT
                || resourceContext.localGroup >= 2,
    HCCL_ERROR("[HcclAllReduce] FullExchange requires the canonical 8+4 rank map"), HCCL_E_NOT_SUPPORT);
args = FullExchangeStaticArgs{};
for (uint32_t die = 0; die < args.dieArgs.size(); ++die) {
    args.dieArgs[die].canonicalRank = resourceContext.localCanonicalRank;
    args.dieArgs[die].canonicalGroup = resourceContext.localGroup;
}
std::array<uint32_t, 2> channelCounts{};
std::array<DieEndpoint, 2> layerDies = {
    DieEndpoint::NOT_APPLICABLE, DieEndpoint::NOT_APPLICABLE};
std::array<bool, DIRECT12_CANONICAL_LEAF_COUNT> peerSeen{};
for (uint32_t index = 0; index < resourceContext.peerChannelCount; ++index) {
    const PeerChannelResource &peer = resourceContext.peerChannels[index];
    CHK_PRT_RET(peer.handle == 0 || peer.canonicalPeer >= DIRECT12_CANONICAL_LEAF_COUNT,
        HCCL_ERROR("[HcclAllReduce] FullExchange static args found an invalid peer channel"),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(peer.localDie != DieEndpoint::DIE_0 && peer.localDie != DieEndpoint::DIE_1,
        HCCL_ERROR("[HcclAllReduce] FullExchange peer channel has no concrete local IO Die"),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(peer.canonicalPeer == resourceContext.localCanonicalRank || peerSeen[peer.canonicalPeer],
        HCCL_ERROR("[HcclAllReduce] FullExchange peer ranks must be unique and exclude the local rank"),
        HCCL_E_INTERNAL);
    peerSeen[peer.canonicalPeer] = true;
    uint32_t layerIndex = 0;
    if (peer.layer == FabricLayer::L0) {
        layerIndex = 0;
    } else if (peer.layer == FabricLayer::L1) {
        layerIndex = 1;
    } else {
        HCCL_ERROR("[HcclAllReduce] FullExchange peer channel has an invalid fabric layer");
        return HCCL_E_INTERNAL;
    }
    const uint32_t peerGroup = peer.canonicalPeer < 8 ? 0 : 1;
    const uint32_t expectedPeerGroup = layerIndex == 0
                                           ? resourceContext.localGroup
                                           : 1U - resourceContext.localGroup;
    CHK_PRT_RET(peerGroup != expectedPeerGroup,
        HCCL_ERROR("[HcclAllReduce] FullExchange peer rank does not belong to its fabric layer"),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(layerDies[layerIndex] != DieEndpoint::NOT_APPLICABLE
                    && layerDies[layerIndex] != peer.localDie,
        HCCL_ERROR("[HcclAllReduce] FullExchange layer [%u] spans both local IO Dies", layerIndex),
        HCCL_E_NOT_SUPPORT);
    layerDies[layerIndex] = peer.localDie;
    const uint32_t die = static_cast<uint32_t>(peer.localDie);
    CHK_PRT_RET(args.dieArgs[die].layer != FabricLayer::INVALID
                    && args.dieArgs[die].layer != peer.layer,
        HCCL_ERROR("[HcclAllReduce] FullExchange IO Die carries channels from both fabric layers"),
        HCCL_E_NOT_SUPPORT);
    args.dieArgs[die].layer = peer.layer;
    uint32_t &channelCount = channelCounts[die];
    CHK_PRT_RET(channelCount >= MAX_PEER_CHANNELS,
        HCCL_ERROR("[HcclAllReduce] FullExchange per-Die static args exceed their fixed capacity"),
        HCCL_E_INTERNAL);
    args.dieArgs[die].channels[channelCount] = peer.handle;
    args.dieArgs[die].peerCanonicalRanks[channelCount] = peer.canonicalPeer;
    ++channelCount;
}
const uint32_t localGroupSize = resourceContext.localGroup == 0 ? 8 : 4;
const uint32_t remoteGroupSize = resourceContext.localGroup == 0 ? 4 : 8;
CHK_PRT_RET(resourceContext.localGroup >= 2 || layerDies[0] == DieEndpoint::NOT_APPLICABLE
                || layerDies[1] == DieEndpoint::NOT_APPLICABLE || layerDies[0] == layerDies[1]
                || channelCounts[static_cast<uint32_t>(layerDies[0])] != localGroupSize - 1
                || channelCounts[static_cast<uint32_t>(layerDies[1])] != remoteGroupSize,
    HCCL_ERROR("[HcclAllReduce] FullExchange requires one complete layer on each IO Die"),
    HCCL_E_NOT_SUPPORT);
for (uint32_t die = 0; die < channelCounts.size(); ++die) {
    args.dieArgs[die].channelCount = channelCounts[die];
}
args.dieArgs[static_cast<uint32_t>(layerDies[0])].copyLocalInput = 1;
return HCCL_SUCCESS;
}
HcclResult BuildTwoXEightMultiRootStaticArgs(
const AlgResourceCtx &resourceContext, TwoXEightMultiRootStaticArgs &args)
{
args = TwoXEightMultiRootStaticArgs{};
args.layerArgs[0].canonicalRank = resourceContext.localCanonicalRank;
args.layerArgs[0].layer = FabricLayer::L0;
args.layerArgs[1].canonicalRank = resourceContext.localCanonicalRank;
args.layerArgs[1].layer = FabricLayer::L1;
std::array<DieEndpoint, 2> layerDies = {
    DieEndpoint::NOT_APPLICABLE, DieEndpoint::NOT_APPLICABLE};
for (uint32_t index = 0; index < resourceContext.peerChannelCount; ++index) {
    const PeerChannelResource &peer = resourceContext.peerChannels[index];
    CHK_PRT_RET(peer.handle == 0 || peer.canonicalPeer >= TWO_X_EIGHT_RANK_COUNT,
        HCCL_ERROR("[HcclAllReduce] 2x8 MultiRoot static args found an invalid peer channel"),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(peer.localDie != DieEndpoint::DIE_0 && peer.localDie != DieEndpoint::DIE_1,
        HCCL_ERROR("[HcclAllReduce] 2x8 MultiRoot peer channel has no concrete local IO Die"),
        HCCL_E_NOT_SUPPORT);
    uint32_t layerIndex = 0;
    if (peer.layer == FabricLayer::L0) {
        layerIndex = 0;
    } else if (peer.layer == FabricLayer::L1) {
        layerIndex = 1;
    } else {
        HCCL_ERROR("[HcclAllReduce] 2x8 MultiRoot peer channel has an invalid fabric layer");
        return HCCL_E_INTERNAL;
    }
    CHK_PRT_RET(layerDies[layerIndex] != DieEndpoint::NOT_APPLICABLE
                    && layerDies[layerIndex] != peer.localDie,
        HCCL_ERROR("[HcclAllReduce] 2x8 MultiRoot layer [%u] spans both local IO Dies", layerIndex),
        HCCL_E_NOT_SUPPORT);
    layerDies[layerIndex] = peer.localDie;
    ops_hccl::TwoXEightMultiRootStaticArg &arg = args.layerArgs[layerIndex];
    CHK_PRT_RET(arg.channelCount >= arg.channels.size(),
        HCCL_ERROR("[HcclAllReduce] 2x8 MultiRoot per-layer channel capacity exceeded"), HCCL_E_INTERNAL);
    arg.channels[arg.channelCount] = peer.handle;
    arg.peerCanonicalRanks[arg.channelCount] = peer.canonicalPeer;
    ++arg.channelCount;
}
CHK_PRT_RET(args.layerArgs[0].channelCount != 7 || args.layerArgs[1].channelCount != 8,
    HCCL_ERROR("[HcclAllReduce] 2x8 MultiRoot requires exactly seven L0 and eight L1 channels"),
    HCCL_E_NOT_SUPPORT);
CHK_PRT_RET(layerDies[0] == DieEndpoint::NOT_APPLICABLE || layerDies[1] == DieEndpoint::NOT_APPLICABLE
                || layerDies[0] == layerDies[1],
    HCCL_ERROR("[HcclAllReduce] 2x8 MultiRoot L0 and L1 kernels must bind different IO Dies"),
    HCCL_E_NOT_SUPPORT);
return HCCL_SUCCESS;
}
using RegisteredCcuKernel = CcuResult (*)(CcuKernelArg);
template <size_t KernelCount>
HcclResult RegisterStageKernels(HcclComm comm, const std::array<const char *, KernelCount> &names,
const std::array<RegisteredCcuKernel, KernelCount> &kernels,
const std::array<const void *, KernelCount> &staticArgs,
const std::array<CcuStageKernelId, KernelCount> &stageOrder, const char *familyName,
AlgResourceCtx &resourceContext)
{
constexpr uint32_t RESERVED_CCU_REGISTER_DIE_ID = 0;
CcuInsHandle instruction = 0;
uint32_t instructionCount = 0;
const HcclResult queryResult = HcclCommQueryCcuIns(comm, &instruction, &instructionCount);
if (queryResult != HCCL_SUCCESS) {
    return Phase0Fail("query_ccu_instruction", queryResult);
}
if (instruction == 0 || instructionCount != 1) {
    return Phase0Fail("query_ccu_instruction", HCCL_E_NOT_SUPPORT);
}
std::array<CcuKernelHandle, KernelCount> handles{};
CcuResult ccuResult = HcommCcuKernelRegisterStart(instruction);
if (ccuResult != CCU_SUCCESS) {
    return Phase0FailCcu("register_start", ccuResult);
}
CcuResult registrationResult = CCU_SUCCESS;
for (uint32_t index = 0; index < handles.size() && registrationResult == CCU_SUCCESS; ++index) {
    const void *kernelStaticArgs[] = {staticArgs[index]};
    registrationResult = HcommCcuKernelRegister(instruction, RESERVED_CCU_REGISTER_DIE_ID, names[index],
        reinterpret_cast<const void *>(kernels[index]), kernelStaticArgs, 1, &handles[index]);
    if (registrationResult != CCU_SUCCESS) {
        HCCL_ERROR("[HcclAllReduce] %s kernel registration failed: index [%u], name [%s], result [%d]",
            familyName, index, names[index], static_cast<int32_t>(registrationResult));
    }
}
ccuResult = HcommCcuKernelRegisterEnd(instruction);
if (registrationResult != CCU_SUCCESS) {
    return Phase0FailCcu("register_kernel", registrationResult);
}
if (ccuResult != CCU_SUCCESS) {
    return Phase0FailCcu("register_end", ccuResult);
}
for (uint32_t index = 0; index < handles.size(); ++index) {
    CHK_PRT_RET(handles[index] == 0,
        HCCL_ERROR("[HcclAllReduce] %s registration returned a zero handle", familyName), HCCL_E_INTERNAL);
    for (uint32_t prior = 0; prior < index; ++prior) {
        CHK_PRT_RET(handles[prior] == handles[index],
            HCCL_ERROR("[HcclAllReduce] %s registration returned duplicate handles", familyName),
            HCCL_E_INTERNAL);
    }
}
resourceContext.registeredStageCount = stageOrder.size();
for (uint32_t index = 0; index < stageOrder.size(); ++index) {
    resourceContext.registeredStages[index] = RegisteredStageResource{stageOrder[index], index, handles[index]};
}
return HCCL_SUCCESS;
}
template <typename StaticArg> void SetDirect12CanonicalMetadata(const AlgResourceCtx &resourceContext, StaticArg &arg)
{
arg.canonicalRank = resourceContext.localCanonicalRank;
arg.canonicalGroup = resourceContext.localGroup;
arg.canonicalGroupRank = resourceContext.localGroupRank;
}
HcclResult BuildDirect12StaticArgs(const AlgResourceCtx &resourceContext, Direct12StaticArgs &args)
{
CHK_PRT_RET(resourceContext.rankMap.rankCount != DIRECT12_CANONICAL_LEAF_COUNT
                || resourceContext.localCanonicalRank >= DIRECT12_CANONICAL_LEAF_COUNT
                || resourceContext.localGroup >= 2,
    HCCL_ERROR("[HcclAllReduce] Direct12 requires the canonical 8+4 rank map"), HCCL_E_NOT_SUPPORT);
args = Direct12StaticArgs{};
for (uint32_t die = 0; die < args.dieArgs.size(); ++die) {
    SetDirect12CanonicalMetadata(resourceContext, args.dieArgs[die]);
    CHK_PRT_RET(resourceContext.messageSize != MessageSizeKind::SIZE_512MB
                    && resourceContext.messageSize != MessageSizeKind::SIZE_400MB_PLUS_4B,
        HCCL_ERROR("[HcclAllReduce] Direct12 received an unsupported message size"), HCCL_E_NOT_SUPPORT);
    args.dieArgs[die].mode = resourceContext.messageSize == MessageSizeKind::SIZE_512MB
                                 ? DIRECT12_MODE_FRONTIER_512
                                 : DIRECT12_MODE_PREFIX_SCALAR_TAIL;
    args.dieArgs[die].scratchStrideBytes = resourceContext.tileBytes;
}
std::array<uint32_t, 2> l0ChannelCounts{};
std::array<uint32_t, 2> l1ChannelCounts{};
DieEndpoint l0Die = DieEndpoint::NOT_APPLICABLE;
DieEndpoint l1Die = DieEndpoint::NOT_APPLICABLE;
std::array<bool, DIRECT12_CANONICAL_LEAF_COUNT> peerSeen{};
for (uint32_t index = 0; index < resourceContext.peerChannelCount; ++index) {
    const PeerChannelResource &peer = resourceContext.peerChannels[index];
    CHK_PRT_RET(peer.handle == 0 || peer.canonicalPeer >= DIRECT12_CANONICAL_LEAF_COUNT,
        HCCL_ERROR("[HcclAllReduce] Direct12 static args found an invalid existing peer channel"),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(peer.localDie != DieEndpoint::DIE_0 && peer.localDie != DieEndpoint::DIE_1,
        HCCL_ERROR("[HcclAllReduce] Direct12 peer channel has no concrete local IO Die"),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(peer.canonicalPeer == resourceContext.localCanonicalRank || peerSeen[peer.canonicalPeer],
        HCCL_ERROR("[HcclAllReduce] Direct12 peer ranks must be unique and exclude the local rank"),
        HCCL_E_INTERNAL);
    peerSeen[peer.canonicalPeer] = true;
    const uint32_t die = static_cast<uint32_t>(peer.localDie);
    if (peer.layer == FabricLayer::L0) {
        CHK_PRT_RET((peer.canonicalPeer < 8 ? 0U : 1U) != resourceContext.localGroup,
            HCCL_ERROR("[HcclAllReduce] Direct12 L0 peer is outside the local canonical group"),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(l0Die != DieEndpoint::NOT_APPLICABLE && l0Die != peer.localDie,
            HCCL_ERROR("[HcclAllReduce] Direct12 L0 channels span both IO Dies"), HCCL_E_NOT_SUPPORT);
        l0Die = peer.localDie;
        uint32_t &l0ChannelCount = l0ChannelCounts[die];
        CHK_PRT_RET(l0ChannelCount >= MAX_PEER_CHANNELS,
            HCCL_ERROR("[HcclAllReduce] Direct12 L0 static args exceed their fixed capacity"), HCCL_E_INTERNAL);
        args.dieArgs[die].l0Channels[l0ChannelCount] = peer.handle;
        args.dieArgs[die].l0PeerCanonicalRanks[l0ChannelCount] = peer.canonicalPeer;
        ++l0ChannelCount;
    } else if (peer.layer == FabricLayer::L1) {
        CHK_PRT_RET((peer.canonicalPeer < 8 ? 0U : 1U) == resourceContext.localGroup,
            HCCL_ERROR("[HcclAllReduce] Direct12 L1 peer is inside the local canonical group"),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(l1Die != DieEndpoint::NOT_APPLICABLE && l1Die != peer.localDie,
            HCCL_ERROR("[HcclAllReduce] Direct12 L1 channels span both IO Dies"), HCCL_E_NOT_SUPPORT);
        l1Die = peer.localDie;
        uint32_t &l1ChannelCount = l1ChannelCounts[die];
        CHK_PRT_RET(l1ChannelCount >= MAX_PEER_CHANNELS,
            HCCL_ERROR("[HcclAllReduce] Direct12 L1 static args exceed their fixed capacity"), HCCL_E_INTERNAL);
        args.dieArgs[die].l1Channels[l1ChannelCount] = peer.handle;
        args.dieArgs[die].l1PeerCanonicalRanks[l1ChannelCount] = peer.canonicalPeer;
        ++l1ChannelCount;
    } else {
        HCCL_ERROR("[HcclAllReduce] Direct12 static args found an invalid peer layer");
        return HCCL_E_INTERNAL;
    }
}
CHK_PRT_RET(l0Die == DieEndpoint::NOT_APPLICABLE || l1Die == DieEndpoint::NOT_APPLICABLE || l0Die == l1Die,
    HCCL_ERROR("[HcclAllReduce] Direct12 L0 and L1 channels must occupy different local IO Dies"),
    HCCL_E_NOT_SUPPORT);
const uint32_t localGroupSize = resourceContext.localGroup == 0 ? 8 : 4;
const uint32_t remoteGroupSize = resourceContext.localGroup == 0 ? 4 : 8;
CHK_PRT_RET(resourceContext.localGroup >= 2
                || l0ChannelCounts[static_cast<uint32_t>(l0Die)] != localGroupSize - 1
                || l1ChannelCounts[static_cast<uint32_t>(l1Die)] != remoteGroupSize,
    HCCL_ERROR("[HcclAllReduce] Direct12 requires complete L0 and L1 canonical groups"),
    HCCL_E_NOT_SUPPORT);
for (uint32_t die = 0; die < l0ChannelCounts.size(); ++die) {
    args.dieArgs[die].l0ChannelCount = l0ChannelCounts[die];
    args.dieArgs[die].l1ChannelCount = l1ChannelCounts[die];
}
// The L0 Die owns the local canonical group and is the only writer of its local leaf.
args.dieArgs[static_cast<uint32_t>(l0Die)].copyLocalInput = 1;
return HCCL_SUCCESS;
}
HcclResult RegisterDirect12StageKernels(
HcclComm comm, const Direct12StaticArgs &args, AlgResourceCtx &resourceContext)
{
const bool prefixScalarTail = args.dieArgs[0].mode == DIRECT12_MODE_PREFIX_SCALAR_TAIL;
const std::array<const char *, 2> names = prefixScalarTail
    ? std::array<const char *, 2>{"Direct12PrefixScalarDie0Kernel", "Direct12PrefixScalarDie1Kernel"}
    : std::array<const char *, 2>{"Direct12Die0Kernel", "Direct12Die1Kernel"};
const RegisteredCcuKernel kernel = prefixScalarTail
    ? ops_hccl::Direct12PrefixScalarDieKernel
    : ops_hccl::Direct12DieKernel;
const std::array<RegisteredCcuKernel, 2> kernels = {kernel, kernel};
const std::array<const void *, 2> staticArgs = {&args.dieArgs[0], &args.dieArgs[1]};
return RegisterStageKernels(
    comm, names, kernels, staticArgs, DIRECT12_STAGE_ORDER, "Direct12", resourceContext);
}
HcclResult RegisterFullExchangeStageKernels(
HcclComm comm, const FullExchangeStaticArgs &args, AlgResourceCtx &resourceContext)
{
const std::array<const char *, 2> names = {"FullExchangeDie0Kernel", "FullExchangeDie1Kernel"};
const std::array<RegisteredCcuKernel, 2> kernels = {
    ops_hccl::FullExchangeDieKernel, ops_hccl::FullExchangeDieKernel};
const std::array<const void *, 2> staticArgs = {&args.dieArgs[0], &args.dieArgs[1]};
return RegisterStageKernels(
    comm, names, kernels, staticArgs, FULL_EXCHANGE_STAGE_ORDER, "FullExchange", resourceContext);
}
HcclResult RegisterTwoXEightMultiRootKernels(
HcclComm comm, const TwoXEightMultiRootStaticArgs &args, AlgResourceCtx &resourceContext)
{
const std::array<const char *, 2> names = {
    "TwoXEightMultiRootL0Kernel", "TwoXEightMultiRootL1Kernel"};
const std::array<RegisteredCcuKernel, 2> kernels = {
    ops_hccl::TwoXEightMultiRootKernel, ops_hccl::TwoXEightMultiRootKernel};
const std::array<const void *, 2> staticArgs = {&args.layerArgs[0], &args.layerArgs[1]};
return RegisterStageKernels(comm, names, kernels, staticArgs, TWO_X_EIGHT_MULTI_ROOT_STAGE_ORDER,
    "TwoXEightMultiRoot", resourceContext);
}
HcclResult RegisterTwoXEightBulkKernels(
HcclComm comm, const TwoXEightMultiRootStaticArgs &args, AlgResourceCtx &resourceContext)
{
const std::array<const char *, 2> names = {
    "TwoXEightBulkL0Kernel", "TwoXEightBulkL1Kernel"};
const std::array<RegisteredCcuKernel, 2> kernels = {
    ops_hccl::TwoXEightBulkKernel, ops_hccl::TwoXEightBulkKernel};
const std::array<const void *, 2> staticArgs = {&args.layerArgs[0], &args.layerArgs[1]};
return RegisterStageKernels(comm, names, kernels, staticArgs, TWO_X_EIGHT_BULK_STAGE_ORDER,
    "TwoXEightBulk", resourceContext);
}
template <typename StaticArg, size_t PeerCount>
HcclResult BuildFourXOneL1StaticArg(const AlgResourceCtx &resourceContext, StaticArg &arg,
const char *familyName, const std::array<uint32_t, PeerCount> &orderedPeers)
{
arg = StaticArg{};
arg.canonicalRank = resourceContext.localCanonicalRank;
CHK_PRT_RET(resourceContext.peerChannelCount != PeerCount || arg.channels.size() != PeerCount,
    HCCL_ERROR("[HcclAllReduce] 4x1 %s requires exactly [%zu] peer channels", familyName, PeerCount),
    HCCL_E_NOT_SUPPORT);
DieEndpoint localDie = DieEndpoint::NOT_APPLICABLE;
for (size_t slot = 0; slot < PeerCount; ++slot) {
    const PeerChannelResource *selectedPeer = nullptr;
    for (uint32_t index = 0; index < resourceContext.peerChannelCount; ++index) {
        if (resourceContext.peerChannels[index].canonicalPeer == orderedPeers[slot]) {
            selectedPeer = &resourceContext.peerChannels[index];
            break;
        }
    }
    CHK_PRT_RET(selectedPeer == nullptr,
        HCCL_ERROR("[HcclAllReduce] 4x1 %s is missing canonical peer [%u]", familyName,
            orderedPeers[slot]),
        HCCL_E_NOT_SUPPORT);
    const PeerChannelResource &peer = *selectedPeer;
    CHK_PRT_RET(peer.handle == 0 || peer.canonicalPeer >= FOUR_X_ONE_RANK_COUNT
                    || peer.layer != FabricLayer::L1,
        HCCL_ERROR("[HcclAllReduce] 4x1 %s requires valid L1 peer channels", familyName),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(peer.localDie != DieEndpoint::DIE_0 && peer.localDie != DieEndpoint::DIE_1,
        HCCL_ERROR("[HcclAllReduce] 4x1 peer channel has no concrete local IO Die"), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(localDie != DieEndpoint::NOT_APPLICABLE && localDie != peer.localDie,
        HCCL_ERROR("[HcclAllReduce] 4x1 L1 channels span both local IO Dies"), HCCL_E_NOT_SUPPORT);
    localDie = peer.localDie;
    arg.channels[slot] = peer.handle;
    arg.peerCanonicalRanks[slot] = peer.canonicalPeer;
    ++arg.channelCount;
}
return HCCL_SUCCESS;
}
HcclResult BuildFourXOneRingStaticArg(
const AlgResourceCtx &resourceContext, ops_hccl::FourXOneRingStaticArg &arg)
{
const uint32_t rank = resourceContext.localCanonicalRank;
const std::array<uint32_t, 2> peers = {
    (rank + 1) % FOUR_X_ONE_RANK_COUNT,
    (rank + FOUR_X_ONE_RANK_COUNT - 1) % FOUR_X_ONE_RANK_COUNT};
return BuildFourXOneL1StaticArg(resourceContext, arg, "RingRSAG", peers);
}
HcclResult BuildFourXOnePullStaticArgs(const AlgResourceCtx &resourceContext,
std::array<ops_hccl::FourXOnePullStaticArg, ACTIVE_WORKERS> &args)
{
std::array<uint32_t, FOUR_X_ONE_PEER_COUNT> peers{};
uint32_t peerCount = 0;
for (uint32_t peer = 0; peer < FOUR_X_ONE_RANK_COUNT; ++peer) {
    if (peer != resourceContext.localCanonicalRank) {
        peers[peerCount++] = peer;
    }
}
CHK_PRT_RET(peerCount != peers.size(),
    HCCL_ERROR("[HcclAllReduce] Invalid 4x1 pull peer set"), HCCL_E_INTERNAL);
for (uint32_t worker = 0; worker < args.size(); ++worker) {
    CHK_RET(BuildFourXOneL1StaticArg(resourceContext, args[worker], "Pull", peers));
    args[worker].worker = worker;
    args[worker].hasScalarTail = worker == 0
                                        && resourceContext.messageSize
                                               == MessageSizeKind::SIZE_400MB_PLUS_4B
                                    ? 1U
                                    : 0U;
}
return HCCL_SUCCESS;
}
HcclResult RegisterFourXOneRingKernels(
HcclComm comm, const ops_hccl::FourXOneRingStaticArg &arg, AlgResourceCtx &resourceContext)
{
const std::array<const char *, 2> names = {"FourXOneRingWorker0Kernel", "FourXOneRingWorker1Kernel"};
const std::array<RegisteredCcuKernel, 2> kernels = {
    ops_hccl::FourXOneRingKernel, ops_hccl::FourXOneRingKernel};
const std::array<const void *, 2> staticArgs = {&arg, &arg};
return RegisterStageKernels(
    comm, names, kernels, staticArgs, FOUR_X_ONE_RING_STAGE_ORDER, "FourXOneRing", resourceContext);
}
HcclResult RegisterFourXOnePullFullExchangeKernel(HcclComm comm,
const ops_hccl::FourXOnePullStaticArg &arg, AlgResourceCtx &resourceContext)
{
const std::array<const char *, 1> names = {"FourXOnePullFullExchangeKernel"};
const std::array<RegisteredCcuKernel, 1> kernels = {ops_hccl::FourXOnePullFullExchangeKernel};
const std::array<const void *, 1> staticArgs = {&arg};
return RegisterStageKernels(comm, names, kernels, staticArgs,
    FOUR_X_ONE_PULL_FULL_EXCHANGE_STAGE_ORDER, "FourXOnePullFullExchange", resourceContext);
}
HcclResult BuildResourceTag(char *tag, size_t tagSize, TopologyKind topology)
{
const int tagLength = std::snprintf(tag, tagSize, "hccl_custom_allreduce_f%u_s%u_p%u_%s",
    RESOURCE_CONTEXT_FRAMEWORK_VERSION, RESOURCE_CONTEXT_SCHEMA_VERSION, static_cast<uint32_t>(ACTIVE_PACKAGE),
    ops_hccl::GetTopologyName(topology));
CHK_PRT_RET(tagLength <= 0 || static_cast<size_t>(tagLength) >= tagSize,
    HCCL_ERROR("[HcclAllReduce] Failed to build versioned topology resource tag"), HCCL_E_INTERNAL);
return HCCL_SUCCESS;
}
HcclResult CheckCachedContext(HcclComm comm, const char *tag, uint64_t streamKey, uint32_t localRank,
const AlgorithmPlan &plan, const CanonicalRankGroupMap &rankMap, bool &found,
AlgResourceCtx &cachedContext, void *&cachedData, uint64_t &cachedSize)
{
found = false;
cachedData = nullptr;
cachedSize = 0;
const HcclResult getResult = HcclEngineCtxGet(comm, tag, COMM_ENGINE_CCU, &cachedData, &cachedSize);
if (getResult == HCCL_E_NOT_FOUND) {
    return HCCL_SUCCESS;
}
CHK_RET(getResult);
found = true;
CHK_PRT_RET(cachedData == nullptr || cachedSize == 0 || cachedSize > MAX_SERIALIZED_RESOURCE_CONTEXT_BYTES,
    HCCL_ERROR("[HcclAllReduce] Cached CCU context [%s] has invalid address or size [%llu]", tag,
        static_cast<unsigned long long>(cachedSize)),
    HCCL_E_PARA);
const char *cachedBytes = static_cast<const char *>(cachedData);
const std::vector<char> encoded(cachedBytes, cachedBytes + static_cast<size_t>(cachedSize));
const HcclResult decodeResult = cachedContext.DeSerialize(encoded);
CHK_PRT_RET(decodeResult != HCCL_SUCCESS,
    HCCL_ERROR("[HcclAllReduce] Cached CCU context [%s] failed checked decode, result [%d]", tag,
        static_cast<int32_t>(decodeResult)),
    decodeResult);
CHK_PRT_RET(cachedContext.streamKey != streamKey,
    HCCL_ERROR("[HcclAllReduce] Cached topology resources are bound to a different user stream"),
    HCCL_E_NOT_SUPPORT);
CHK_PRT_RET(cachedContext.package != plan.package || cachedContext.topology != plan.topology
                || cachedContext.messageSize != plan.messageSize || cachedContext.family != plan.family
                || cachedContext.tailPolicy != plan.tailPolicy
                || cachedContext.resourceProfile != plan.resourceProfile
                || cachedContext.workerCount != plan.streamCount
                || cachedContext.kernelHandleCount != plan.kernelHandleCount
                || cachedContext.tileBytes != plan.tileBytes || cachedContext.scratchBytes != plan.scratchBytes
                || cachedContext.localPhysicalRank != localRank
                || cachedContext.localCanonicalRank != rankMap.physicalToCanonical[localRank]
                || !RankMapsEqual(cachedContext.rankMap, rankMap),
    HCCL_ERROR("[HcclAllReduce] Cached CCU context [%s] does not match the selected cell", tag),
    HCCL_E_NOT_SUPPORT);
return HCCL_SUCCESS;
}
HcclResult CacheResourceContext(HcclComm comm, const char *tag, const AlgResourceCtx &resourceContext)
{
std::vector<char> encoded;
const HcclResult encodeResult = resourceContext.SerializeChecked(encoded);
CHK_PRT_RET(encodeResult != HCCL_SUCCESS || encoded.empty(),
    HCCL_ERROR("[HcclAllReduce] Failed to serialize the launch-ready CCU resource context"),
    encodeResult == HCCL_SUCCESS ? HCCL_E_INTERNAL : encodeResult);
void *cachedContext = nullptr;
const HcclResult createResult
    = HcclEngineCtxCreate(comm, tag, COMM_ENGINE_CCU, encoded.size(), &cachedContext);
if (createResult != HCCL_SUCCESS) {
    return createResult;
}
if (cachedContext == nullptr) {
    (void)HcclEngineCtxDestroy(comm, tag, COMM_ENGINE_CCU);
    HCCL_ERROR("[HcclAllReduce] Created CCU resource context has a null address");
    return HCCL_E_INTERNAL;
}
const HcclResult copyResult
    = HcclEngineCtxCopy(comm, COMM_ENGINE_CCU, tag, encoded.data(), encoded.size(), 0);
if (copyResult != HCCL_SUCCESS) {
    (void)HcclEngineCtxDestroy(comm, tag, COMM_ENGINE_CCU);
    return copyResult;
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
CHK_RET(ValidateCompetitionInput(count, dataType, op));
OpParam param;
param.inputPtr = sendBuf;
param.outputPtr = recvBuf;
param.count = count;
param.dataType = dataType;
param.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
param.reduceType = op;
CHK_RET(HcclGetRankId(comm, &param.myRank));
CHK_RET(HcclGetRankSize(comm, &param.rankSize));
CHK_PRT_RET(
    param.rankSize == 0 || param.rankSize > static_cast<uint32_t>(MAX_RANK_SIZE) || param.myRank >= param.rankSize,
    HCCL_ERROR("[HcclAllReduce] Invalid communicator rank metadata: rank [%u], rankSize [%u]", param.myRank,
        param.rankSize),
    HCCL_E_NOT_SUPPORT);
TopologyKind topology = TopologyKind::UNSUPPORTED;
CanonicalRankGroupMap rankMap;
CHK_RET(DiscoverCanonicalTopology(comm, param.myRank, param.rankSize, topology, rankMap));
const uint64_t dataBytes = count * sizeof(float);
AlgorithmPlan plan;
CHK_RET(ops_hccl::SelectAlgorithmPlan(param.rankSize, dataBytes, plan));
CHK_PRT_RET(plan.topology != topology,
    HCCL_ERROR("[HcclAllReduce] Selected plan topology [%u] conflicts with discovered topology [%u]",
        static_cast<uint32_t>(plan.topology), static_cast<uint32_t>(topology)),
    HCCL_E_NOT_SUPPORT);
if (!plan.implemented) {
    HCCL_WARNING("[HcclAllReduce] Selected package/topology/message cell is explicitly unsupported");
    return HCCL_E_NOT_SUPPORT;
}
ResourceProfile resourceProfile;
CHK_RET(ops_hccl::GetResourceProfile(plan.resourceProfile, resourceProfile));
CHK_PRT_RET(resourceProfile.engine != COMM_ENGINE_CCU || !resourceProfile.acquireAllPeerChannels,
    HCCL_ERROR("[HcclAllReduce] Only the all-peer CCU resource profile is supported"), HCCL_E_NOT_SUPPORT);
CHK_RET(BuildResourceTag(param.tag, sizeof(param.tag), topology));
HcclDfxOpInfo dfxInfo{};
dfxInfo.opType = static_cast<uint32_t>(param.opType);
dfxInfo.reduceOp = static_cast<uint32_t>(op);
dfxInfo.dataType = static_cast<uint32_t>(dataType);
dfxInfo.dataCount = count;
dfxInfo.engine = COMM_ENGINE_CCU;
dfxInfo.inputMemAddr = reinterpret_cast<uint64_t>(sendBuf);
dfxInfo.inputMemSize = dataBytes;
dfxInfo.outputMemAddr = reinterpret_cast<uint64_t>(recvBuf);
dfxInfo.outputMemSize = dataBytes;
const int dfxTagLength = std::snprintf(dfxInfo.algTag, sizeof(dfxInfo.algTag), "%s", param.tag);
CHK_PRT_RET(dfxTagLength <= 0 || static_cast<size_t>(dfxTagLength) >= sizeof(dfxInfo.algTag),
    HCCL_ERROR("[HcclAllReduce] Failed to build DFX algorithm tag"), HCCL_E_INTERNAL);
char commName[COMM_INDENTIFIER_MAX_LENGTH] = {};
CHK_RET(HcclGetCommName(comm, commName));
CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));
static_assert(sizeof(uintptr_t) <= sizeof(uint64_t), "stream keys require pointers of at most 64 bits");
const uint64_t streamKey = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(stream));
bool cachedContextFound = false;
AlgResourceCtx cachedContext;
void *cachedData = nullptr;
uint64_t cachedSize = 0;
CHK_RET(CheckCachedContext(
    comm, param.tag, streamKey, param.myRank, plan, rankMap, cachedContextFound,
    cachedContext, cachedData, cachedSize));
if (cachedContextFound) {
    ++cachedContext.invocationSequence;
    std::vector<char> updatedContext;
    CHK_RET(cachedContext.SerializeChecked(updatedContext));
    CHK_PRT_RET(updatedContext.size() != cachedSize,
        HCCL_ERROR("[HcclAllReduce] Cached Direct12 context size changed unexpectedly"), HCCL_E_INTERNAL);
    CHK_RET(HcclEngineCtxCopy(
        comm, COMM_ENGINE_CCU, param.tag, updatedContext.data(), updatedContext.size(), 0));
    param.resCtx = cachedData;
    param.ctxSize = cachedSize;
    return ops_hccl::ExecOp(param);
}
AlgResourceCtx resourceContext;
resourceContext.package = ACTIVE_PACKAGE;
resourceContext.topology = topology;
resourceContext.messageSize = plan.messageSize;
resourceContext.family = plan.family;
resourceContext.tailPolicy = plan.tailPolicy;
resourceContext.resourceProfile = plan.resourceProfile;
resourceContext.workerCount = plan.streamCount;
resourceContext.kernelHandleCount = plan.kernelHandleCount;
resourceContext.tileBytes = plan.tileBytes;
resourceContext.scratchBytes = plan.scratchBytes;
resourceContext.streamKey = streamKey;
resourceContext.rankMap = rankMap;
resourceContext.localPhysicalRank = param.myRank;
resourceContext.localCanonicalRank = rankMap.physicalToCanonical[param.myRank];
resourceContext.localGroup = rankMap.groupByCanonicalRank[resourceContext.localCanonicalRank];
resourceContext.localGroupRank = rankMap.groupLocalRank[resourceContext.localCanonicalRank];
CanonicalPeerSet requiredPeers;
CHK_RET(BuildRequiredPeerSet(
    plan, resourceContext.localCanonicalRank, resourceContext.rankMap.rankCount, requiredPeers));
resourceContext.requiredPeerCount = requiredPeers.count;
resourceContext.requiredPeerMask = requiredPeers.mask;
HcclResult phase0Result
    = HcclGetHcclBuffer(comm, &resourceContext.hcclBuffer.addr, &resourceContext.hcclBuffer.size);
if (phase0Result != HCCL_SUCCESS) {
    return Phase0Fail("hccl_buffer", phase0Result);
}
if (resourceContext.hcclBuffer.size < resourceContext.scratchBytes) {
    return Phase0Fail("hccl_buffer_size", HCCL_E_NOT_SUPPORT);
}
phase0Result = HcclThreadAcquireWithStream(
    comm, COMM_ENGINE_CCU, stream, resourceProfile.notifyNumPerThread, &resourceContext.mainThread);
if (phase0Result != HCCL_SUCCESS) {
    return Phase0Fail("main_thread_acquire", phase0Result);
}
if (resourceContext.mainThread == 0) {
    return Phase0Fail("main_thread_handle", HCCL_E_INTERNAL);
}
param.cpuThread = resourceContext.mainThread;
if (resourceContext.workerCount == 2) {
    const HcclResult slaveAcquire = HcclThreadAcquire(comm, COMM_ENGINE_CCU, 1, CELL_THREAD_NOTIFY_COUNT,
        resourceContext.slaveThreads.data());
    if (slaveAcquire != HCCL_SUCCESS) {
        return Phase0Fail("slave_thread_acquire", slaveAcquire);
    }
    resourceContext.slaveThreadCount = 1;
    if (resourceContext.slaveThreads[0] == 0 || resourceContext.slaveThreads[0] == resourceContext.mainThread) {
        return Phase0Fail("slave_thread_handle", HCCL_E_INTERNAL);
    }
}
phase0Result = AcquirePeerChannels(comm, param.myRank, topology, rankMap, requiredPeers, resourceContext);
if (phase0Result != HCCL_SUCCESS) {
    return Phase0Fail("peer_channel_acquire", phase0Result);
}
if (resourceContext.ValidateCanonicalStructure() != HCCL_SUCCESS) {
    return Phase0Fail("base_resource_context", HCCL_E_INTERNAL);
}
if (plan.family == AlgorithmFamily::DIRECT12_RSAG) {
    Direct12StaticArgs direct12Args;
    phase0Result = BuildDirect12StaticArgs(resourceContext, direct12Args);
    if (phase0Result != HCCL_SUCCESS) {
        return phase0Result;
    }
    phase0Result = RegisterDirect12StageKernels(comm, direct12Args, resourceContext);
    if (phase0Result != HCCL_SUCCESS) {
        return phase0Result;
    }
} else if (plan.family == AlgorithmFamily::FULL_EXCHANGE_FIXED_TREE
           && plan.topology == TopologyKind::TOPOLOGY_8_PLUS_4) {
    FullExchangeStaticArgs fullExchangeArgs;
    phase0Result = BuildFullExchangeStaticArgs(resourceContext, fullExchangeArgs);
    if (phase0Result != HCCL_SUCCESS) {
        return phase0Result;
    }
    phase0Result = RegisterFullExchangeStageKernels(comm, fullExchangeArgs, resourceContext);
    if (phase0Result != HCCL_SUCCESS) {
        return phase0Result;
    }
} else if (plan.family == AlgorithmFamily::FULL_EXCHANGE_FIXED_TREE
           && plan.topology == TopologyKind::TOPOLOGY_4X1) {
    std::array<ops_hccl::FourXOnePullStaticArg, ACTIVE_WORKERS> pullArgs;
    phase0Result = BuildFourXOnePullStaticArgs(resourceContext, pullArgs);
    if (phase0Result != HCCL_SUCCESS) {
        return phase0Result;
    }
    phase0Result = RegisterFourXOnePullFullExchangeKernel(comm, pullArgs[0], resourceContext);
    if (phase0Result != HCCL_SUCCESS) {
        return phase0Result;
    }
} else if (plan.family == AlgorithmFamily::RING_RSAG
           && plan.topology == TopologyKind::TOPOLOGY_4X1) {
    ops_hccl::FourXOneRingStaticArg ringArg;
    phase0Result = BuildFourXOneRingStaticArg(resourceContext, ringArg);
    if (phase0Result != HCCL_SUCCESS) {
        return phase0Result;
    }
    phase0Result = RegisterFourXOneRingKernels(comm, ringArg, resourceContext);
    if (phase0Result != HCCL_SUCCESS) {
        return phase0Result;
    }
} else if (plan.family == AlgorithmFamily::TOPOLOGY_BALANCED_DIRECT_RSAG
           && plan.topology == TopologyKind::TOPOLOGY_2X8) {
    TwoXEightMultiRootStaticArgs multiRootArgs;
    phase0Result = BuildTwoXEightMultiRootStaticArgs(resourceContext, multiRootArgs);
    if (phase0Result != HCCL_SUCCESS) {
        return phase0Result;
    }
    phase0Result = RegisterTwoXEightMultiRootKernels(comm, multiRootArgs, resourceContext);
    if (phase0Result != HCCL_SUCCESS) {
        return phase0Result;
    }
} else if (plan.family == AlgorithmFamily::BULK_PUSH_STAGED_REDUCE
           && plan.topology == TopologyKind::TOPOLOGY_2X8) {
    TwoXEightMultiRootStaticArgs bulkArgs;
    phase0Result = BuildTwoXEightMultiRootStaticArgs(resourceContext, bulkArgs);
    if (phase0Result != HCCL_SUCCESS) {
        return phase0Result;
    }
    phase0Result = RegisterTwoXEightBulkKernels(comm, bulkArgs, resourceContext);
    if (phase0Result != HCCL_SUCCESS) {
        return phase0Result;
    }
} else {
    return Phase0Fail("algorithm_family", HCCL_E_NOT_SUPPORT);
}
if (resourceContext.ValidateLaunchReady() != HCCL_SUCCESS) {
    return Phase0Fail("launch_ready_context", HCCL_E_INTERNAL);
}
CHK_RET(CacheResourceContext(comm, param.tag, resourceContext));
std::vector<char> launchContext;
CHK_RET(resourceContext.SerializeChecked(launchContext));
param.resCtx = launchContext.data();
param.ctxSize = launchContext.size();
return ops_hccl::ExecOp(param);
}
