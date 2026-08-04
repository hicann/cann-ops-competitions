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
#include <array>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

#include <ccu/ccu_launch.h>
#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>

#include "common.h"
#include "ccu_kernel.h"
#include "custom.h"
#include "exec_op.h"
#include "hccl.h"
#include "log.h"

namespace {

constexpr uint32_t NETWORK_LAYER_COUNT = 2;
constexpr uint32_t IO_DIE_COUNT = 2;
constexpr uint32_t RESOURCE_GROUP_COUNT = NETWORK_LAYER_COUNT * IO_DIE_COUNT;
constexpr uint32_t CHANNEL_NOTIFY_COUNT = 4;
constexpr uint32_t KERNEL_ARGUMENT_COUNT = 1;
constexpr uint32_t RANK_SIZE_8_PLUS_4 = 12;
constexpr uint32_t RANK_SIZE_2_X_8 = 16;
constexpr uint64_t FORMAL_SMALL_2X8_DATA_SIZE = 32768U;
constexpr uint64_t FORMAL_SMALL_8_PLUS_4_DATA_SIZE = 43692U;
constexpr uint64_t FORMAL_SMALL_8_PLUS_4_OUTPUT_SIZE = 524304U;
constexpr uint64_t FORMAL_SMALL_OUTPUT_SIZE = 512U * 1024U;
constexpr uint32_t DEFAULT_MAIN_NOTIFY_COUNT = 1;
constexpr uint32_t COPY_OVERLAP_MAIN_NOTIFY_COUNT = 2;
// CANN 9.1.0 ignores this reserved argument and derives the target Die from all Channel handles.
constexpr uint32_t RESERVED_DIE_ID = 0;

struct DieGroupResource {
    uint32_t layerId = 0;
    EndpointAttrDieId dieId = 0;
    std::vector<HcclChannelDesc> channelDescs;
    std::vector<ChannelHandle> channels;
    std::vector<uint32_t> peerRanks;
};

bool IsSupportedProtocol(const CommLink &link)
{
    return link.linkAttr.linkProtocol == CommProtocol::COMM_PROTOCOL_UBC_CTP;
}

uint32_t GetGroupIndex(uint32_t layerId, EndpointAttrDieId dieId)
{
    return layerId * IO_DIE_COUNT + static_cast<uint32_t>(dieId);
}

HcclResult QueryLocalEndpointDie(
    HcclComm comm, uint32_t myRank, const EndpointDesc &endpoint, EndpointAttrDieId &dieId)
{
    CHK_RET(HcclRankGraphGetEndpointInfo(
        comm, myRank, &endpoint, ENDPOINT_ATTR_DIE_ID, sizeof(EndpointAttrDieId), &dieId));
    if (static_cast<uint32_t>(dieId) >= IO_DIE_COUNT) {
        HCCL_ERROR("[HcclAllGather] Local endpoint returned invalid IO Die %u",
            static_cast<uint32_t>(dieId));
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult AppendPeerChannel(HcclComm comm, const OpParam &param, uint32_t remoteRank, uint32_t layerId,
    const CommLink &link, std::array<DieGroupResource, RESOURCE_GROUP_COUNT> &groups)
{
    if (layerId >= NETWORK_LAYER_COUNT) {
        HCCL_ERROR("[HcclAllGather] Unsupported network layer: %u", layerId);
        return HCCL_E_NOT_SUPPORT;
    }

    EndpointAttrDieId localDieId = 0;
    CHK_RET(QueryLocalEndpointDie(comm, param.myRank, link.srcEndpointDesc, localDieId));
    const uint32_t groupIndex = GetGroupIndex(layerId, localDieId);
    if (groupIndex >= groups.size()) {
        HCCL_ERROR("[HcclAllGather] Invalid layer/Die group: layer=%u die=%u", layerId,
            static_cast<uint32_t>(localDieId));
        return HCCL_E_INTERNAL;
    }

    HcclChannelDesc desc;
    CHK_RET(HcclChannelDescInit(&desc, 1));
    desc.remoteRank = remoteRank;
    desc.notifyNum = CHANNEL_NOTIFY_COUNT;
    desc.channelProtocol = link.linkAttr.linkProtocol;
    desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
    desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
    desc.localEndpoint.loc = link.srcEndpointDesc.loc;
    desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
    desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
    desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;

    DieGroupResource &group = groups[groupIndex];
    if (group.channelDescs.empty()) {
        group.layerId = layerId;
        group.dieId = localDieId;
    } else if (group.layerId != layerId || group.dieId != localDieId) {
        HCCL_ERROR("[HcclAllGather] Internal layer/Die group collision");
        return HCCL_E_INTERNAL;
    }
    group.channelDescs.push_back(desc);
    group.peerRanks.push_back(remoteRank);
    return HCCL_SUCCESS;
}

HcclResult BuildDieGroupChannels(
    HcclComm comm, const OpParam &param, std::array<DieGroupResource, RESOURCE_GROUP_COUNT> &groups)
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    if (netLayers == nullptr || netLayerNum == 0) {
        HCCL_ERROR("[HcclAllGather] RankGraph returned no network layers");
        return HCCL_E_NOT_FOUND;
    }

    // The RankGraph owns this memory. Copy it before making subsequent RankGraph calls.
    const std::vector<uint32_t> layerIds(netLayers, netLayers + netLayerNum);
    std::array<bool, NETWORK_LAYER_COUNT> layerPresent{};
    for (const uint32_t layerId : layerIds) {
        if (layerId >= NETWORK_LAYER_COUNT) {
            HCCL_ERROR("[HcclAllGather] The final problem supports only layer-0/layer-1, got layer %u",
                layerId);
            return HCCL_E_NOT_SUPPORT;
        }
        layerPresent[layerId] = true;
    }
    if (!layerPresent[0] || !layerPresent[1]) {
        HCCL_ERROR("[HcclAllGather] Expected both layer-0 and layer-1 in RankGraph");
        return HCCL_E_NOT_SUPPORT;
    }

    std::vector<bool> peerSeen(param.rankSize, false);
    peerSeen[param.myRank] = true;
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }

        bool found = false;
        // Prefer Mesh for same-server peers and fall back to Clos for cross-server peers.
        for (uint32_t layerId = 0; layerId < NETWORK_LAYER_COUNT && !found; ++layerId) {
            CommLink *links = nullptr;
            uint32_t linkCount = 0;
            CHK_RET(HcclRankGraphGetLinks(
                comm, layerId, param.myRank, remoteRank, &links, &linkCount));
            if (linkCount != 0 && links == nullptr) {
                HCCL_ERROR("[HcclAllGather] RankGraph returned null links for rank %u -> %u on layer-%u",
                    param.myRank, remoteRank, layerId);
                return HCCL_E_INTERNAL;
            }
            for (uint32_t linkIndex = 0; linkIndex < linkCount; ++linkIndex) {
                if (!IsSupportedProtocol(links[linkIndex])) {
                    continue;
                }
                if (peerSeen[remoteRank]) {
                    HCCL_ERROR("[HcclAllGather] Duplicate Channel request for peer %u", remoteRank);
                    return HCCL_E_INTERNAL;
                }
                CHK_RET(AppendPeerChannel(
                    comm, param, remoteRank, layerId, links[linkIndex], groups));
                peerSeen[remoteRank] = true;
                found = true;
                break;
            }
        }
        if (!found) {
            HCCL_ERROR("[HcclAllGather] No UBC_CTP link from rank %u to rank %u",
                param.myRank, remoteRank);
            return HCCL_E_NOT_FOUND;
        }
    }

    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (!peerSeen[rank]) {
            HCCL_ERROR("[HcclAllGather] No Channel request was generated for peer %u", rank);
            return HCCL_E_INTERNAL;
        }
    }

    // Channel connection is a peer-to-peer handshake. The local (layer, Die) grouping order is not
    // symmetric at the two ends of a link, so acquiring one group at a time can make the peers wait
    // in different batches. Acquire every peer once in remote-rank order, then distribute the
    // returned handles to Die-local kernel groups.
    std::vector<HcclChannelDesc> orderedDescs;
    std::vector<uint32_t> orderedGroupIndices;
    std::vector<uint32_t> orderedChannelIndices;
    orderedDescs.reserve(param.rankSize - 1);
    orderedGroupIndices.reserve(param.rankSize - 1);
    orderedChannelIndices.reserve(param.rankSize - 1);
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }
        bool placed = false;
        for (uint32_t groupIndex = 0; groupIndex < groups.size() && !placed; ++groupIndex) {
            const DieGroupResource &group = groups[groupIndex];
            for (uint32_t channelIndex = 0; channelIndex < group.peerRanks.size(); ++channelIndex) {
                if (group.peerRanks[channelIndex] != remoteRank) {
                    continue;
                }
                orderedDescs.push_back(group.channelDescs[channelIndex]);
                orderedGroupIndices.push_back(groupIndex);
                orderedChannelIndices.push_back(channelIndex);
                placed = true;
                break;
            }
        }
        if (!placed) {
            HCCL_ERROR("[HcclAllGather] Cannot place Channel descriptor for peer %u", remoteRank);
            return HCCL_E_INTERNAL;
        }
    }
    if (orderedDescs.size() != param.rankSize - 1) {
        HCCL_ERROR("[HcclAllGather] Expected %u ordered Channel descriptors, got %zu",
            param.rankSize - 1, orderedDescs.size());
        return HCCL_E_INTERNAL;
    }

    std::vector<ChannelHandle> orderedChannels(orderedDescs.size());
    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, orderedDescs.data(),
        static_cast<uint32_t>(orderedDescs.size()), orderedChannels.data()));

    for (DieGroupResource &group : groups) {
        group.channels.resize(group.channelDescs.size());
    }
    for (uint32_t orderedIndex = 0; orderedIndex < orderedChannels.size(); ++orderedIndex) {
        groups[orderedGroupIndices[orderedIndex]].channels[orderedChannelIndices[orderedIndex]] =
            orderedChannels[orderedIndex];
    }
    return HCCL_SUCCESS;
}

HcclResult PlaceLayerPeersFirst(
    std::array<DieGroupResource, RESOURCE_GROUP_COUNT> &groups, uint32_t layerId,
    const std::vector<uint32_t> &orderedPeers, uint32_t &selectedGroupIndex)
{
    selectedGroupIndex = RESOURCE_GROUP_COUNT;
    if (orderedPeers.empty()) {
        return HCCL_E_PARA;
    }

    for (uint32_t orderedIndex = 0; orderedIndex < orderedPeers.size(); ++orderedIndex) {
        const uint32_t peerRank = orderedPeers[orderedIndex];
        uint32_t foundGroupIndex = RESOURCE_GROUP_COUNT;
        uint32_t foundChannelIndex = MAX_RANK_SIZE;
        for (uint32_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
            DieGroupResource &group = groups[groupIndex];
            if (group.layerId != layerId) {
                continue;
            }
            const auto peerIter =
                std::find(group.peerRanks.begin(), group.peerRanks.end(), peerRank);
            if (peerIter == group.peerRanks.end()) {
                continue;
            }
            foundGroupIndex = groupIndex;
            foundChannelIndex =
                static_cast<uint32_t>(std::distance(group.peerRanks.begin(), peerIter));
            break;
        }
        if (foundGroupIndex == RESOURCE_GROUP_COUNT) {
            HCCL_ERROR("[HcclAllGather] Cannot place layer-%u subset peer %u",
                layerId, peerRank);
            return HCCL_E_INTERNAL;
        }
        if (selectedGroupIndex == RESOURCE_GROUP_COUNT) {
            selectedGroupIndex = foundGroupIndex;
        } else if (selectedGroupIndex != foundGroupIndex) {
            HCCL_ERROR("[HcclAllGather] Layer-%u subset spans multiple IO Dies", layerId);
            return HCCL_E_NOT_SUPPORT;
        }

        DieGroupResource &group = groups[selectedGroupIndex];
        if (orderedIndex >= group.peerRanks.size()) {
            return HCCL_E_INTERNAL;
        }
        std::swap(group.peerRanks[orderedIndex], group.peerRanks[foundChannelIndex]);
        std::swap(group.channelDescs[orderedIndex], group.channelDescs[foundChannelIndex]);
        std::swap(group.channels[orderedIndex], group.channels[foundChannelIndex]);
    }
    return HCCL_SUCCESS;
}

HcclResult PlaceLayerPeersByGroup(
    std::array<DieGroupResource, RESOURCE_GROUP_COUNT> &groups, uint32_t layerId,
    const std::vector<uint32_t> &orderedPeers,
    std::array<uint32_t, RESOURCE_GROUP_COUNT> &selectedCounts)
{
    selectedCounts.fill(0U);
    for (const uint32_t peerRank : orderedPeers) {
        uint32_t foundGroupIndex = RESOURCE_GROUP_COUNT;
        uint32_t foundChannelIndex = MAX_RANK_SIZE;
        for (uint32_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
            DieGroupResource &group = groups[groupIndex];
            if (group.layerId != layerId) {
                continue;
            }
            const auto peerIter =
                std::find(group.peerRanks.begin(), group.peerRanks.end(), peerRank);
            if (peerIter == group.peerRanks.end()) {
                continue;
            }
            foundGroupIndex = groupIndex;
            foundChannelIndex =
                static_cast<uint32_t>(std::distance(group.peerRanks.begin(), peerIter));
            break;
        }
        if (foundGroupIndex == RESOURCE_GROUP_COUNT) {
            HCCL_ERROR("[HcclAllGather] Cannot place layer-%u grouped peer %u",
                layerId, peerRank);
            return HCCL_E_INTERNAL;
        }
        DieGroupResource &group = groups[foundGroupIndex];
        const uint32_t prefixIndex = selectedCounts[foundGroupIndex]++;
        if (prefixIndex >= group.peerRanks.size()) {
            return HCCL_E_INTERNAL;
        }
        std::swap(group.peerRanks[prefixIndex], group.peerRanks[foundChannelIndex]);
        std::swap(group.channelDescs[prefixIndex], group.channelDescs[foundChannelIndex]);
        std::swap(group.channels[prefixIndex], group.channels[foundChannelIndex]);
    }
    return HCCL_SUCCESS;
}

HcclResult ConfigureGroupSafeRelay(const OpParam &param,
    std::array<DieGroupResource, RESOURCE_GROUP_COUNT> &groups,
    std::vector<uint32_t> &pairedRemoteRanks, uint32_t &pairGroupIndex,
    std::array<uint32_t, RESOURCE_GROUP_COUNT> &relayChannelCounts,
    std::array<std::array<uint32_t, MAX_RANK_SIZE>, RESOURCE_GROUP_COUNT> &relaySourceRanks0,
    std::array<std::array<uint32_t, MAX_RANK_SIZE>, RESOURCE_GROUP_COUNT> &relaySourceRanks1,
    std::array<uint32_t, RESOURCE_GROUP_COUNT> &relaySourceCounts,
    uint32_t &pairRoundMask)
{
    pairedRemoteRanks.clear();
    pairGroupIndex = RESOURCE_GROUP_COUNT;
    relayChannelCounts.fill(0U);
    for (auto &ranks : relaySourceRanks0) {
        ranks.fill(MAX_RANK_SIZE);
    }
    for (auto &ranks : relaySourceRanks1) {
        ranks.fill(MAX_RANK_SIZE);
    }
    relaySourceCounts.fill(0U);
    pairRoundMask = 0U;
    if (param.rankSize != RANK_SIZE_8_PLUS_4 &&
        param.rankSize != RANK_SIZE_2_X_8) {
        return HCCL_SUCCESS;
    }

    std::vector<uint32_t> localRanks = {param.myRank};
    std::vector<uint32_t> remoteRanks;
    for (const DieGroupResource &group : groups) {
        if (group.layerId == 0) {
            localRanks.insert(localRanks.end(), group.peerRanks.begin(), group.peerRanks.end());
        } else if (group.layerId == 1) {
            remoteRanks.insert(remoteRanks.end(), group.peerRanks.begin(), group.peerRanks.end());
        }
    }
    std::sort(localRanks.begin(), localRanks.end());
    std::sort(remoteRanks.begin(), remoteRanks.end());
    const bool valid2x8 =
        param.rankSize == RANK_SIZE_2_X_8 &&
        localRanks.size() == 8U && remoteRanks.size() == 8U;
    const bool valid8Plus4 =
        param.rankSize == RANK_SIZE_8_PLUS_4 &&
        ((localRanks.size() == 8U && remoteRanks.size() == 4U) ||
         (localRanks.size() == 4U && remoteRanks.size() == 8U));
    if ((!valid2x8 && !valid8Plus4) ||
        std::adjacent_find(localRanks.begin(), localRanks.end()) != localRanks.end() ||
        std::adjacent_find(remoteRanks.begin(), remoteRanks.end()) != remoteRanks.end()) {
        HCCL_INFO("[HcclAllGather] Communicator is not a supported group-safe split; "
            "keep the direct path");
        return HCCL_SUCCESS;
    }

    const auto localIter = std::find(localRanks.begin(), localRanks.end(), param.myRank);
    if (localIter == localRanks.end()) {
        return HCCL_E_INTERNAL;
    }
    const uint32_t localIndex =
        static_cast<uint32_t>(std::distance(localRanks.begin(), localIter));

    std::vector<uint32_t> relayTargets;
    if (param.rankSize == RANK_SIZE_2_X_8) {
        pairedRemoteRanks.push_back(remoteRanks[localIndex]);
        pairRoundMask = 1U;
        for (const uint32_t rank : localRanks) {
            if (rank != param.myRank) {
                relayTargets.push_back(rank);
            }
        }
    } else if (localRanks.size() == 8U) {
        pairedRemoteRanks.push_back(remoteRanks[localIndex % 4U]);
        pairRoundMask = localIndex < 4U ? 1U : 2U;
        const uint32_t halfBegin = localIndex / 4U * 4U;
        for (uint32_t index = halfBegin; index < halfBegin + 4U; ++index) {
            if (index != localIndex) {
                relayTargets.push_back(localRanks[index]);
            }
        }
    } else {
        pairedRemoteRanks.push_back(remoteRanks[localIndex]);
        pairedRemoteRanks.push_back(remoteRanks[localIndex + 4U]);
        pairRoundMask = 3U;
        for (const uint32_t rank : localRanks) {
            if (rank != param.myRank) {
                relayTargets.push_back(rank);
            }
        }
    }
    const size_t expectedRelayTargetCount =
        param.rankSize == RANK_SIZE_2_X_8 ? 7U : 3U;
    if (relayTargets.size() != expectedRelayTargetCount ||
        (pairedRemoteRanks.size() != 1U && pairedRemoteRanks.size() != 2U)) {
        return HCCL_E_INTERNAL;
    }

    CHK_RET(PlaceLayerPeersFirst(
        groups, 1U, pairedRemoteRanks, pairGroupIndex));
    CHK_RET(PlaceLayerPeersByGroup(
        groups, 0U, relayTargets, relayChannelCounts));
    for (uint32_t groupIndex = 0; groupIndex < RESOURCE_GROUP_COUNT; ++groupIndex) {
        const DieGroupResource &group = groups[groupIndex];
        const uint32_t relayCount = relayChannelCounts[groupIndex];
        if (group.layerId != 0U || relayCount == 0U) {
            continue;
        }
        relaySourceCounts[groupIndex] =
            param.rankSize == RANK_SIZE_2_X_8 ||
            localRanks.size() == 8U ? 1U : 2U;
        for (uint32_t channelIndex = 0; channelIndex < relayCount; ++channelIndex) {
            const auto peerIter = std::find(
                localRanks.begin(), localRanks.end(), group.peerRanks[channelIndex]);
            if (peerIter == localRanks.end()) {
                return HCCL_E_INTERNAL;
            }
            const uint32_t peerIndex =
                static_cast<uint32_t>(std::distance(localRanks.begin(), peerIter));
            if (param.rankSize == RANK_SIZE_2_X_8) {
                relaySourceRanks0[groupIndex][channelIndex] =
                    remoteRanks[peerIndex];
            } else if (localRanks.size() == 8U) {
                relaySourceRanks0[groupIndex][channelIndex] =
                    remoteRanks[peerIndex % 4U];
            } else {
                relaySourceRanks0[groupIndex][channelIndex] =
                    remoteRanks[peerIndex];
                relaySourceRanks1[groupIndex][channelIndex] =
                    remoteRanks[peerIndex + 4U];
            }
        }
    }
    HCCL_INFO("[HcclAllGather] group-safe relay rank[%u], rankSize[%u], localSize[%zu], "
        "pairCount[%zu], relayCount[%zu]",
        param.myRank, param.rankSize, localRanks.size(),
        pairedRemoteRanks.size(), relayTargets.size());
    return HCCL_SUCCESS;
}

HcclResult RegisterStagedKernels(HcclComm comm, uint32_t localRank,
    uint32_t rankSize,
    const std::array<DieGroupResource, RESOURCE_GROUP_COUNT> &groups, uint32_t pairedRemoteRank,
    const std::vector<uint32_t> &asymmetricPairedRanks, uint32_t asymmetricPairGroupIndex,
    const std::array<uint32_t, RESOURCE_GROUP_COUNT> &relayChannelCounts,
    const std::array<std::array<uint32_t, MAX_RANK_SIZE>, RESOURCE_GROUP_COUNT> &relaySourceRanks0,
    const std::array<std::array<uint32_t, MAX_RANK_SIZE>, RESOURCE_GROUP_COUNT> &relaySourceRanks1,
    const std::array<uint32_t, RESOURCE_GROUP_COUNT> &relaySourceCounts,
    uint32_t asymmetricPairRoundMask,
    AlgResourceCtx &resourceContext)
{
    CcuInsHandle instructionHandle{0};
    uint32_t instructionCount = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &instructionHandle, &instructionCount));
    if (instructionCount != 1) {
        HCCL_ERROR("[HcclAllGather] Expected one CCU instruction context, got %u", instructionCount);
        return HCCL_E_INTERNAL;
    }

    const std::array<void *, NETWORK_LAYER_COUNT> kernelFunctions = {
        reinterpret_cast<void *>(ops_hccl::CcuStagedLayer0),
        reinterpret_cast<void *>(ops_hccl::CcuStagedLayer1),
    };
    std::array<CcuKernelInfo, RESOURCE_GROUP_COUNT> kernelInfo;
    CcuKernelInfo small4x1PullInfo{};
    bool hasSmall4x1PullInfo = false;
    for (uint32_t groupIndex = 0; groupIndex < RESOURCE_GROUP_COUNT; ++groupIndex) {
        const DieGroupResource &group = groups[groupIndex];
        if (group.channels.empty()) {
            continue;
        }
        if (group.layerId >= NETWORK_LAYER_COUNT ||
            static_cast<uint32_t>(group.dieId) >= IO_DIE_COUNT ||
            group.channels.size() >= MAX_RANK_SIZE ||
            group.channels.size() != group.peerRanks.size()) {
            HCCL_ERROR("[HcclAllGather] Invalid group-%u channel mapping", groupIndex);
            return HCCL_E_INTERNAL;
        }

        CcuKernelInfo &info = kernelInfo[groupIndex];
        const int written = std::snprintf(info.kernelFuncName, sizeof(info.kernelFuncName),
            "CcuAllGatherStagedL%uD%u", group.layerId, static_cast<uint32_t>(group.dieId));
        if (written <= 0 || static_cast<size_t>(written) >= sizeof(info.kernelFuncName)) {
            HCCL_ERROR("[HcclAllGather] Failed to set kernel name for group-%u", groupIndex);
            return HCCL_E_INTERNAL;
        }
        info.kernelFunc = kernelFunctions[group.layerId];

        auto kernelArg = std::make_shared<ops_hccl::CcuKernelArgDirect>();
        kernelArg->layerId = group.layerId;
        kernelArg->localRank = localRank;
        kernelArg->channelCount =
            static_cast<uint32_t>(group.channels.size());
        if (!asymmetricPairedRanks.empty() &&
            groupIndex == asymmetricPairGroupIndex) {
            kernelArg->pairChannelCount =
                static_cast<uint32_t>(asymmetricPairedRanks.size());
            if ((asymmetricPairRoundMask & 1U) != 0U) {
                kernelArg->asymmetricPairRound0Channel = 0U;
            }
            if ((asymmetricPairRoundMask & 2U) != 0U) {
                kernelArg->asymmetricPairRound1Channel =
                    asymmetricPairedRanks.size() == 2U ? 1U : 0U;
            }
            kernelArg->asymmetricPairLargeSide =
                asymmetricPairedRanks.size() == 1U;
        } else if (pairedRemoteRank < MAX_RANK_SIZE &&
            group.layerId == 1U && !group.peerRanks.empty() &&
            group.peerRanks[0] == pairedRemoteRank) {
            kernelArg->pairChannelCount = 1U;
        }
        kernelArg->relayChannelCount = relayChannelCounts[groupIndex];
        kernelArg->relaySourceCount = relaySourceCounts[groupIndex];
        for (uint32_t channelIndex = 0;
            channelIndex < kernelArg->relayChannelCount; ++channelIndex) {
            kernelArg->relaySourceRanks0[channelIndex] =
                relaySourceRanks0[groupIndex][channelIndex];
            kernelArg->relaySourceRanks1[channelIndex] =
                relaySourceRanks1[groupIndex][channelIndex];
        }
        kernelArg->asymmetricLayer0 =
            !asymmetricPairedRanks.empty() && group.layerId == 0U;
        if (kernelArg->pairChannelCount > kernelArg->channelCount ||
            kernelArg->relayChannelCount > kernelArg->channelCount ||
            (kernelArg->relayChannelCount != 0U &&
             (kernelArg->relaySourceCount == 0U ||
              kernelArg->relaySourceCount > 2U)) ||
            (kernelArg->pairChannelCount != 0U && group.layerId != 1U) ||
            (kernelArg->relayChannelCount != 0U && group.layerId != 0U)) {
            HCCL_ERROR("[HcclAllGather] Invalid group-%u subset counts", groupIndex);
            return HCCL_E_INTERNAL;
        }
        for (uint32_t channelIndex = 0; channelIndex < kernelArg->channelCount; ++channelIndex) {
            kernelArg->channels[channelIndex] = group.channels[channelIndex];
            kernelArg->peerRanks[channelIndex] = group.peerRanks[channelIndex];
        }
        info.setKernelArg(kernelArg);

        if (rankSize == 4U && group.layerId == 1U) {
            if (hasSmall4x1PullInfo || kernelArg->channelCount != 3U) {
                HCCL_ERROR("[HcclAllGather] Invalid 4x1 layer-1 Channel mapping");
                return HCCL_E_INTERNAL;
            }
            const int smallNameWritten = std::snprintf(
                small4x1PullInfo.kernelFuncName,
                sizeof(small4x1PullInfo.kernelFuncName),
                "%s", "CcuAllGatherSmall4x1DirectPull");
            if (smallNameWritten <= 0 ||
                static_cast<size_t>(smallNameWritten) >=
                    sizeof(small4x1PullInfo.kernelFuncName)) {
                HCCL_ERROR("[HcclAllGather] Failed to set 4x1 Pull kernel name");
                return HCCL_E_INTERNAL;
            }
            small4x1PullInfo.kernelFunc =
                reinterpret_cast<void *>(ops_hccl::CcuSmall4x1DirectPull);
            auto smallKernelArg =
                std::make_shared<ops_hccl::CcuKernelArgDirect>();
            smallKernelArg->layerId = 1U;
            smallKernelArg->localRank = localRank;
            smallKernelArg->channelCount = kernelArg->channelCount;
            for (uint32_t channelIndex = 0;
                channelIndex < smallKernelArg->channelCount; ++channelIndex) {
                if (group.peerRanks[channelIndex] >= rankSize ||
                    group.peerRanks[channelIndex] == localRank) {
                    HCCL_ERROR(
                        "[HcclAllGather] Invalid 4x1 Pull peer rank %u",
                        group.peerRanks[channelIndex]);
                    return HCCL_E_INTERNAL;
                }
                smallKernelArg->channels[channelIndex] =
                    group.channels[channelIndex];
                smallKernelArg->peerRanks[channelIndex] =
                    group.peerRanks[channelIndex];
            }
            small4x1PullInfo.setKernelArg(smallKernelArg);
            hasSmall4x1PullInfo = true;
        }
    }
    if (rankSize == 4U && !hasSmall4x1PullInfo) {
        HCCL_ERROR("[HcclAllGather] Missing 4x1 layer-1 resources");
        return HCCL_E_INTERNAL;
    }

    CcuResult ccuResult = HcommCcuKernelRegisterStart(instructionHandle);
    if (ccuResult != CCU_SUCCESS) {
        HCCL_ERROR("[HcclAllGather] Kernel registration start failed, ccuRet=%d",
            static_cast<int32_t>(ccuResult));
        return ConvertCcuToHccl(ccuResult);
    }

    resourceContext.ccuKernels.clear();
    resourceContext.kernelLayers.clear();
    resourceContext.small4x1PullKernel = 0;
    resourceContext.pairKernelIndex = MAX_RANK_SIZE;
    resourceContext.pairedRemoteRank = pairedRemoteRank;
    resourceContext.relayKernelIndices.clear();
    resourceContext.asymmetricPairedRanks = asymmetricPairedRanks;
    resourceContext.asymmetricPairRoundMask = asymmetricPairRoundMask;
    for (uint32_t groupIndex = 0; groupIndex < RESOURCE_GROUP_COUNT; ++groupIndex) {
        const DieGroupResource &group = groups[groupIndex];
        if (group.channels.empty()) {
            continue;
        }

        CcuKernelInfo &info = kernelInfo[groupIndex];
        CcuKernelHandle kernelHandle = 0;
        const void *kernelArgs[] = {info.kernelArg};
        ccuResult = HcommCcuKernelRegister(instructionHandle, RESERVED_DIE_ID, info.kernelFuncName,
            info.kernelFunc, kernelArgs, KERNEL_ARGUMENT_COUNT, &kernelHandle);
        if (ccuResult != CCU_SUCCESS) {
            HCCL_ERROR("[HcclAllGather] Layer-%u die-%u staged kernel registration failed, ccuRet=%d",
                group.layerId, static_cast<uint32_t>(group.dieId),
                static_cast<int32_t>(ccuResult));
            const CcuResult endResult = HcommCcuKernelRegisterEnd(instructionHandle);
            if (endResult != CCU_SUCCESS) {
                HCCL_ERROR("[HcclAllGather] Cleanup registration end failed, ccuRet=%d",
                    static_cast<int32_t>(endResult));
            }
            return ConvertCcuToHccl(ccuResult);
        }
        const uint32_t kernelIndex = static_cast<uint32_t>(resourceContext.ccuKernels.size());
        resourceContext.ccuKernels.push_back(kernelHandle);
        resourceContext.kernelLayers.push_back(group.layerId);
        const bool ownsPairChannels =
            groupIndex == asymmetricPairGroupIndex ||
            (pairedRemoteRank < MAX_RANK_SIZE && group.layerId == 1 &&
             !group.peerRanks.empty() && group.peerRanks[0] == pairedRemoteRank);
        if (ownsPairChannels) {
            if (resourceContext.pairKernelIndex != MAX_RANK_SIZE) {
                HCCL_ERROR("[HcclAllGather] Paired Channel was assigned to multiple kernels");
                const CcuResult endResult = HcommCcuKernelRegisterEnd(instructionHandle);
                if (endResult != CCU_SUCCESS) {
                    HCCL_ERROR("[HcclAllGather] Cleanup registration end failed, ccuRet=%d",
                        static_cast<int32_t>(endResult));
                }
                return HCCL_E_INTERNAL;
            }
            resourceContext.pairKernelIndex = kernelIndex;
        }
        if (relayChannelCounts[groupIndex] != 0U) {
            resourceContext.relayKernelIndices.push_back(kernelIndex);
        }
    }

    if (hasSmall4x1PullInfo) {
        const void *smallKernelArgs[] = {small4x1PullInfo.kernelArg};
        ccuResult = HcommCcuKernelRegister(
            instructionHandle, RESERVED_DIE_ID,
            small4x1PullInfo.kernelFuncName,
            small4x1PullInfo.kernelFunc, smallKernelArgs,
            KERNEL_ARGUMENT_COUNT,
            &resourceContext.small4x1PullKernel);
        if (ccuResult != CCU_SUCCESS) {
            HCCL_ERROR(
                "[HcclAllGather] Dedicated 4x1 Pull kernel registration failed, ccuRet=%d",
                static_cast<int32_t>(ccuResult));
            const CcuResult endResult =
                HcommCcuKernelRegisterEnd(instructionHandle);
            if (endResult != CCU_SUCCESS) {
                HCCL_ERROR(
                    "[HcclAllGather] Cleanup registration end failed, ccuRet=%d",
                    static_cast<int32_t>(endResult));
            }
            resourceContext.small4x1PullKernel = 0;
            return ConvertCcuToHccl(ccuResult);
        }
    }

    ccuResult = HcommCcuKernelRegisterEnd(instructionHandle);
    if (ccuResult != CCU_SUCCESS) {
        HCCL_ERROR("[HcclAllGather] Kernel registration end failed, ccuRet=%d",
            static_cast<int32_t>(ccuResult));
        return ConvertCcuToHccl(ccuResult);
    }
    if ((pairedRemoteRank < MAX_RANK_SIZE ||
         !asymmetricPairedRanks.empty()) &&
        resourceContext.pairKernelIndex == MAX_RANK_SIZE) {
        HCCL_ERROR("[HcclAllGather] No layer-1 kernel owns the paired Channel");
        return HCCL_E_INTERNAL;
    }
    if (!asymmetricPairedRanks.empty() &&
        resourceContext.relayKernelIndices.empty()) {
        HCCL_ERROR("[HcclAllGather] No layer-0 kernel owns the relay Channel subset");
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterSmall2x8PullKernelsOnly(
    HcclComm comm, uint32_t localRank, uint32_t rankSize,
    const std::array<DieGroupResource, RESOURCE_GROUP_COUNT> &groups,
    Small2x8PullResourceCtx &resourceContext)
{
    if (rankSize != RANK_SIZE_2_X_8) {
        return HCCL_E_PARA;
    }

    CcuInsHandle instructionHandle{0};
    uint32_t instructionCount = 0;
    CHK_RET(HcclCommQueryCcuIns(
        comm, &instructionHandle, &instructionCount));
    if (instructionCount != 1U) {
        HCCL_ERROR(
            "[HcclAllGather] Expected one CCU instruction context, got %u",
            instructionCount);
        return HCCL_E_INTERNAL;
    }

    std::array<CcuKernelInfo, RESOURCE_GROUP_COUNT> kernelInfo;
    std::array<bool, RESOURCE_GROUP_COUNT> hasKernelInfo{};
    uint32_t groupCount = 0U;
    uint32_t layer0Count = 0U;
    uint32_t layer1Count = 0U;
    for (uint32_t groupIndex = 0;
        groupIndex < RESOURCE_GROUP_COUNT; ++groupIndex) {
        const DieGroupResource &group = groups[groupIndex];
        if (group.channels.empty()) {
            continue;
        }
        if (group.layerId >= NETWORK_LAYER_COUNT ||
            group.channels.size() >= MAX_RANK_SIZE ||
            group.channels.size() != group.peerRanks.size()) {
            HCCL_ERROR(
                "[HcclAllGather] Invalid small 2x8 group-%u",
                groupIndex);
            return HCCL_E_INTERNAL;
        }

        CcuKernelInfo &info = kernelInfo[groupIndex];
        const int written = std::snprintf(
            info.kernelFuncName, sizeof(info.kernelFuncName),
            "CcuAllGatherSmall2x8OnlyL%uD%u", group.layerId,
            static_cast<uint32_t>(group.dieId));
        if (written <= 0 ||
            static_cast<size_t>(written) >=
                sizeof(info.kernelFuncName)) {
            return HCCL_E_INTERNAL;
        }
        info.kernelFunc = group.layerId == 0U ?
            reinterpret_cast<void *>(
                ops_hccl::CcuSmall2x8Layer0DirectPull) :
            reinterpret_cast<void *>(
                ops_hccl::CcuSmall2x8Layer1DirectPull);

        auto kernelArg =
            std::make_shared<ops_hccl::CcuKernelArgDirect>();
        kernelArg->layerId = group.layerId;
        kernelArg->localRank = localRank;
        kernelArg->channelCount =
            static_cast<uint32_t>(group.channels.size());
        for (uint32_t channelIndex = 0;
            channelIndex < kernelArg->channelCount; ++channelIndex) {
            if (group.peerRanks[channelIndex] >= rankSize ||
                group.peerRanks[channelIndex] == localRank) {
                return HCCL_E_INTERNAL;
            }
            kernelArg->channels[channelIndex] =
                group.channels[channelIndex];
            kernelArg->peerRanks[channelIndex] =
                group.peerRanks[channelIndex];
        }
        info.setKernelArg(kernelArg);
        hasKernelInfo[groupIndex] = true;
        ++groupCount;
        layer0Count += group.layerId == 0U ? 1U : 0U;
        layer1Count += group.layerId == 1U ? 1U : 0U;
    }
    if (groupCount != 2U || layer0Count != 1U ||
        layer1Count != 1U) {
        HCCL_ERROR(
            "[HcclAllGather] Expected one group per layer for small 2x8, got groups=%u layer0=%u layer1=%u",
            groupCount, layer0Count, layer1Count);
        return HCCL_E_INTERNAL;
    }

    CcuResult ccuResult =
        HcommCcuKernelRegisterStart(instructionHandle);
    if (ccuResult != CCU_SUCCESS) {
        return ConvertCcuToHccl(ccuResult);
    }
    resourceContext.kernels.clear();
    resourceContext.layers.clear();
    for (uint32_t groupIndex = 0;
        groupIndex < RESOURCE_GROUP_COUNT; ++groupIndex) {
        if (!hasKernelInfo[groupIndex]) {
            continue;
        }
        CcuKernelInfo &info = kernelInfo[groupIndex];
        const void *kernelArgs[] = {info.kernelArg};
        CcuKernelHandle kernelHandle = 0;
        ccuResult = HcommCcuKernelRegister(
            instructionHandle, RESERVED_DIE_ID,
            info.kernelFuncName, info.kernelFunc, kernelArgs,
            KERNEL_ARGUMENT_COUNT, &kernelHandle);
        if (ccuResult != CCU_SUCCESS) {
            HcommCcuKernelRegisterEnd(instructionHandle);
            resourceContext.kernels.clear();
            resourceContext.layers.clear();
            return ConvertCcuToHccl(ccuResult);
        }
        resourceContext.kernels.push_back(kernelHandle);
        resourceContext.layers.push_back(
            groups[groupIndex].layerId);
    }
    ccuResult = HcommCcuKernelRegisterEnd(instructionHandle);
    if (ccuResult != CCU_SUCCESS) {
        return ConvertCcuToHccl(ccuResult);
    }
    return resourceContext.kernels.size() == 2U ?
        HCCL_SUCCESS : HCCL_E_INTERNAL;
}

HcclResult Register8Plus4PullKernelsOnly(
    HcclComm comm, uint32_t localRank, uint32_t rankSize,
    const std::array<DieGroupResource, RESOURCE_GROUP_COUNT> &groups,
    Small8Plus4PullResourceCtx &resourceContext, bool largeMessage)
{
    if (rankSize != RANK_SIZE_8_PLUS_4) {
        return HCCL_E_PARA;
    }
    uint32_t groupCount = 0U;
    uint32_t peerCount = 0U;
    uint32_t layer0Count = 0U;
    uint32_t layer1Count = 0U;
    uint32_t localCopyGroup = RESOURCE_GROUP_COUNT;
    for (uint32_t groupIndex = 0;
        groupIndex < RESOURCE_GROUP_COUNT; ++groupIndex) {
        const DieGroupResource &group = groups[groupIndex];
        if (group.channels.empty()) {
            continue;
        }
        if (group.channels.size() != group.peerRanks.size() ||
            group.channels.size() >= 16U ||
            group.layerId >= NETWORK_LAYER_COUNT) {
            return HCCL_E_INTERNAL;
        }
        ++groupCount;
        peerCount += static_cast<uint32_t>(group.channels.size());
        layer0Count += group.layerId == 0U ? 1U : 0U;
        layer1Count += group.layerId == 1U ? 1U : 0U;
        if (group.layerId == 1U &&
            localCopyGroup == RESOURCE_GROUP_COUNT) {
            localCopyGroup = groupIndex;
        }
    }
    if (groupCount != 2U || peerCount != 11U ||
        layer0Count != 1U || layer1Count != 1U ||
        localCopyGroup == RESOURCE_GROUP_COUNT) {
        HCCL_ERROR(
            "[HcclAllGather] Expected one group per layer/11 peers for small 8+4");
        return HCCL_E_INTERNAL;
    }

    CcuInsHandle instructionHandle{0};
    uint32_t instructionCount = 0;
    CHK_RET(HcclCommQueryCcuIns(
        comm, &instructionHandle, &instructionCount));
    if (instructionCount != 1U) {
        return HCCL_E_INTERNAL;
    }

    std::array<CcuKernelInfo, RESOURCE_GROUP_COUNT> kernelInfo;
    std::array<bool, RESOURCE_GROUP_COUNT> hasKernelInfo{};
    for (uint32_t groupIndex = 0;
        groupIndex < RESOURCE_GROUP_COUNT; ++groupIndex) {
        const DieGroupResource &group = groups[groupIndex];
        if (group.channels.empty()) {
            continue;
        }
        const bool ownsLocalCopy =
            groupIndex == localCopyGroup;
        CcuKernelInfo &info = kernelInfo[groupIndex];
        const int written = std::snprintf(
            info.kernelFuncName, sizeof(info.kernelFuncName),
            largeMessage ? "CcuAllGatherLarge8Plus4Only%sL%uD%u" :
                "CcuAllGatherSmall8Plus4Only%sL%uD%u",
            ownsLocalCopy ? "Owner" : "Peer",
            group.layerId, static_cast<uint32_t>(group.dieId));
        if (written <= 0 ||
            static_cast<size_t>(written) >=
                sizeof(info.kernelFuncName)) {
            return HCCL_E_INTERNAL;
        }
        info.kernelFunc = largeMessage ?
            (ownsLocalCopy ?
                reinterpret_cast<void *>(
                    ops_hccl::CcuLarge8Plus4Layer1DirectPull) :
                reinterpret_cast<void *>(
                    ops_hccl::CcuLarge8Plus4Layer0DirectPull)) :
            (ownsLocalCopy ?
                reinterpret_cast<void *>(
                    ops_hccl::CcuSmall2x8Layer1DirectPull) :
                reinterpret_cast<void *>(
                    ops_hccl::CcuSmall2x8Layer0DirectPull));

        auto kernelArg =
            std::make_shared<ops_hccl::CcuKernelArgDirect>();
        // These two stage-free functions differ only in LocalCopy ownership.
        // Channel handles, not this specialization marker, determine the Die.
        kernelArg->layerId = ownsLocalCopy ? 1U : 0U;
        kernelArg->localRank = localRank;
        kernelArg->channelCount =
            static_cast<uint32_t>(group.channels.size());
        for (uint32_t channelIndex = 0;
            channelIndex < kernelArg->channelCount; ++channelIndex) {
            const uint32_t peerRank =
                group.peerRanks[channelIndex];
            if (peerRank >= rankSize || peerRank == localRank) {
                return HCCL_E_INTERNAL;
            }
            kernelArg->channels[channelIndex] =
                group.channels[channelIndex];
            kernelArg->peerRanks[channelIndex] = peerRank;
        }
        info.setKernelArg(kernelArg);
        hasKernelInfo[groupIndex] = true;
    }

    CcuResult ccuResult =
        HcommCcuKernelRegisterStart(instructionHandle);
    if (ccuResult != CCU_SUCCESS) {
        return ConvertCcuToHccl(ccuResult);
    }
    resourceContext.kernels.clear();
    resourceContext.layers.clear();
    for (uint32_t groupIndex = 0;
        groupIndex < RESOURCE_GROUP_COUNT; ++groupIndex) {
        if (!hasKernelInfo[groupIndex]) {
            continue;
        }
        CcuKernelInfo &info = kernelInfo[groupIndex];
        const void *kernelArgs[] = {info.kernelArg};
        CcuKernelHandle kernelHandle = 0;
        ccuResult = HcommCcuKernelRegister(
            instructionHandle, RESERVED_DIE_ID,
            info.kernelFuncName, info.kernelFunc, kernelArgs,
            KERNEL_ARGUMENT_COUNT, &kernelHandle);
        if (ccuResult != CCU_SUCCESS) {
            HcommCcuKernelRegisterEnd(instructionHandle);
            resourceContext.kernels.clear();
            resourceContext.layers.clear();
            return ConvertCcuToHccl(ccuResult);
        }
        resourceContext.kernels.push_back(kernelHandle);
        resourceContext.layers.push_back(
            groups[groupIndex].layerId);
    }
    ccuResult = HcommCcuKernelRegisterEnd(instructionHandle);
    if (ccuResult != CCU_SUCCESS) {
        return ConvertCcuToHccl(ccuResult);
    }
    return resourceContext.kernels.size() == 2U ?
        HCCL_SUCCESS : HCCL_E_INTERNAL;
}

HcclResult AcquireAuxiliaryThreads(
    HcclComm comm, const OpParam &param, AlgResourceCtx &resourceContext)
{
    bool hasLayer0 = false;
    bool hasLayer1 = false;
    for (const uint32_t layerId : resourceContext.kernelLayers) {
        hasLayer0 = hasLayer0 || layerId == 0;
        hasLayer1 = hasLayer1 || layerId == 1;
    }
    resourceContext.slaveThreads.clear();
    // Mixed-layer communicators use the auxiliary Thread for layer-1. A pure
    // layer-1 communicator uses it only for Channel-free local-copy overlap.
    const bool mixedLayers = hasLayer0 && hasLayer1;
    const bool pureClos =
        !hasLayer0 && hasLayer1 && resourceContext.ccuKernels.size() == 1;
    uint32_t threadCount = 0;
    if (mixedLayers) {
        // slave[0] owns every layer-1 Channel. The two large mixed topologies
        // additionally use slave[1] only for a Channel-free local copy.
        threadCount =
            (param.rankSize == RANK_SIZE_8_PLUS_4 ||
             param.rankSize == RANK_SIZE_2_X_8) ? 2U : 1U;
    } else if (pureClos) {
        threadCount = 1U;
    }
    if (threadCount == 0) {
        return HCCL_SUCCESS;
    }

    std::vector<ThreadHandle> slaveThreads(threadCount);
    CHK_RET(HcclThreadAcquire(
        comm, CommEngine::COMM_ENGINE_CCU, threadCount, 1, slaveThreads.data()));
    resourceContext.slaveThreads = slaveThreads;
    return HCCL_SUCCESS;
}

HcclResult CreateSmall2x8PullResourceContextFromGroups(
    HcclComm comm, const OpParam &param,
    const std::array<DieGroupResource, RESOURCE_GROUP_COUNT> &groups,
    Small2x8PullResourceCtx &resourceContext)
{
    CHK_RET(RegisterSmall2x8PullKernelsOnly(
        comm, param.myRank, param.rankSize, groups,
        resourceContext));
    std::vector<ThreadHandle> slaveThreads(1U);
    CHK_RET(HcclThreadAcquire(
        comm, CommEngine::COMM_ENGINE_CCU, 1U, 1U,
        slaveThreads.data()));
    resourceContext.slaveThreads = slaveThreads;
    return HCCL_SUCCESS;
}

HcclResult Create8Plus4PullResourceContextFromGroups(
    HcclComm comm, const OpParam &param,
    const std::array<DieGroupResource, RESOURCE_GROUP_COUNT> &groups,
    Small8Plus4PullResourceCtx &resourceContext, bool largeMessage)
{
    CHK_RET(Register8Plus4PullKernelsOnly(
        comm, param.myRank, param.rankSize, groups,
        resourceContext, largeMessage));
    std::vector<ThreadHandle> slaveThreads(1U);
    CHK_RET(HcclThreadAcquire(
        comm, CommEngine::COMM_ENGINE_CCU, 1U, 1U,
        slaveThreads.data()));
    resourceContext.slaveThreads = slaveThreads;
    return HCCL_SUCCESS;
}

bool IsCanonicalTwoGroupShape(
    const std::array<DieGroupResource, RESOURCE_GROUP_COUNT> &groups)
{
    uint32_t layer0Count = 0U;
    uint32_t layer1Count = 0U;
    for (const DieGroupResource &group : groups) {
        if (group.channels.empty()) {
            continue;
        }
        layer0Count += group.layerId == 0U ? 1U : 0U;
        layer1Count += group.layerId == 1U ? 1U : 0U;
    }
    return layer0Count == 1U && layer1Count == 1U;
}

HcclResult CreateResourceContextFromGroups(
    HcclComm comm, const OpParam &param,
    std::array<DieGroupResource, RESOURCE_GROUP_COUNT> &groups,
    AlgResourceCtx &resourceContext)
{
    uint32_t pairedRemoteRank = MAX_RANK_SIZE;
    std::vector<uint32_t> asymmetricPairedRanks;
    uint32_t asymmetricPairGroupIndex = RESOURCE_GROUP_COUNT;
    std::array<uint32_t, RESOURCE_GROUP_COUNT> relayChannelCounts{};
    std::array<std::array<uint32_t, MAX_RANK_SIZE>, RESOURCE_GROUP_COUNT>
        relaySourceRanks0{};
    std::array<std::array<uint32_t, MAX_RANK_SIZE>, RESOURCE_GROUP_COUNT>
        relaySourceRanks1{};
    std::array<uint32_t, RESOURCE_GROUP_COUNT> relaySourceCounts{};
    uint32_t asymmetricPairRoundMask = 0U;
    CHK_RET(ConfigureGroupSafeRelay(param, groups, asymmetricPairedRanks,
        asymmetricPairGroupIndex, relayChannelCounts, relaySourceRanks0,
        relaySourceRanks1, relaySourceCounts, asymmetricPairRoundMask));
    CHK_RET(RegisterStagedKernels(comm, param.myRank, param.rankSize,
        groups, pairedRemoteRank,
        asymmetricPairedRanks, asymmetricPairGroupIndex, relayChannelCounts,
        relaySourceRanks0, relaySourceRanks1, relaySourceCounts,
        asymmetricPairRoundMask, resourceContext));
    CHK_RET(AcquireAuxiliaryThreads(comm, param, resourceContext));
    return HCCL_SUCCESS;
}

HcclResult CreateResourceContext(
    HcclComm comm, const OpParam &param, AlgResourceCtx &resourceContext)
{
    if (param.rankSize == 1) {
        resourceContext.ccuKernels.clear();
        resourceContext.kernelLayers.clear();
        resourceContext.small4x1PullKernel = 0;
        resourceContext.slaveThreads.clear();
        resourceContext.pairKernelIndex = MAX_RANK_SIZE;
        resourceContext.pairedRemoteRank = MAX_RANK_SIZE;
        resourceContext.relayKernelIndices.clear();
        resourceContext.asymmetricPairedRanks.clear();
        resourceContext.asymmetricPairRoundMask = 0U;
        return HCCL_SUCCESS;
    }

    std::array<DieGroupResource, RESOURCE_GROUP_COUNT> groups;
    CHK_RET(BuildDieGroupChannels(comm, param, groups));
    return CreateResourceContextFromGroups(
        comm, param, groups, resourceContext);
}

} // namespace

HcclResult HcclAllGather(
    void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm,
    aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param{};
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = sendCount;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    if (param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || param.myRank >= param.rankSize) {
        HCCL_ERROR("[HcclAllGather] Invalid rank information: rank=%u rankSize=%u",
            param.myRank, param.rankSize);
        return HCCL_E_PARA;
    }
    const auto sizeIter = SIZE_TABLE.find(dataType);
    if (sizeIter == SIZE_TABLE.end()) {
        HCCL_ERROR("[HcclAllGather] Unsupported data type: %d", static_cast<int32_t>(dataType));
        return HCCL_E_NOT_SUPPORT;
    }
    const uint64_t dataTypeSize = sizeIter->second;
    const bool isFormalSmall2x8 =
        param.rankSize == RANK_SIZE_2_X_8 &&
        sendCount <=
            std::numeric_limits<uint64_t>::max() / dataTypeSize &&
        sendCount * dataTypeSize ==
            FORMAL_SMALL_2X8_DATA_SIZE &&
        FORMAL_SMALL_2X8_DATA_SIZE * RANK_SIZE_2_X_8 ==
            FORMAL_SMALL_OUTPUT_SIZE;
    const bool isFormalSmall8Plus4 =
        param.rankSize == RANK_SIZE_8_PLUS_4 &&
        sendCount <=
            std::numeric_limits<uint64_t>::max() / dataTypeSize &&
        sendCount * dataTypeSize ==
            FORMAL_SMALL_8_PLUS_4_DATA_SIZE &&
        FORMAL_SMALL_8_PLUS_4_DATA_SIZE * RANK_SIZE_8_PLUS_4 ==
            FORMAL_SMALL_8_PLUS_4_OUTPUT_SIZE;
    const char *contextTag = isFormalSmall2x8 ?
        "hccl_final_v064_2x8_adaptive_twogroup_pull" :
        (isFormalSmall8Plus4 ?
            "hccl_final_v064_8plus4_adaptive_twogroup_pull" :
            "hccl_final_v052_4x1_small_pull_postsync");
    const int written = std::snprintf(
        param.tag, sizeof(param.tag), "%s", contextTag);
    if (written <= 0 ||
        static_cast<size_t>(written) >= sizeof(param.tag)) {
        HCCL_ERROR("[HcclAllGather] Failed to set operator tag");
        return HCCL_E_INTERNAL;
    }

    const CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;
    // The stream-bound main CCU thread runs layer 0, or all kernels for a single-layer communicator.
    const uint32_t mainNotifyCount =
        (param.rankSize == RANK_SIZE_8_PLUS_4 ||
         param.rankSize == RANK_SIZE_2_X_8) ?
        COPY_OVERLAP_MAIN_NOTIFY_COUNT : DEFAULT_MAIN_NOTIFY_COUNT;
    CHK_RET(HcclThreadAcquireWithStream(
        comm, ccuEngine, stream, mainNotifyCount, &param.cpuThread));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, ccuEngine, &ctx, &size) == HCCL_SUCCESS) {
        HCCL_INFO("Engine context already exists");
        param.resCtx = ctx;
        param.ctxSize = size;
    } else {
        AlgResourceCtx resCtxHost;
        if (isFormalSmall2x8 || isFormalSmall8Plus4) {
            std::array<DieGroupResource, RESOURCE_GROUP_COUNT> groups;
            CHK_RET(BuildDieGroupChannels(comm, param, groups));
            if (IsCanonicalTwoGroupShape(groups)) {
                if (isFormalSmall2x8) {
                    Small2x8PullResourceCtx smallResCtx;
                    CHK_RET(CreateSmall2x8PullResourceContextFromGroups(
                        comm, param, groups, smallResCtx));
                    resCtxHost.ccuKernels = smallResCtx.kernels;
                    resCtxHost.kernelLayers = smallResCtx.layers;
                    resCtxHost.slaveThreads = smallResCtx.slaveThreads;
                    resCtxHost.small2x8DirectPull = 1U;
                } else {
                    Small8Plus4PullResourceCtx smallResCtx;
                    CHK_RET(Create8Plus4PullResourceContextFromGroups(
                        comm, param, groups, smallResCtx,
                        false));
                    resCtxHost.ccuKernels = smallResCtx.kernels;
                    resCtxHost.kernelLayers = smallResCtx.layers;
                    resCtxHost.slaveThreads = smallResCtx.slaveThreads;
                    resCtxHost.small8Plus4DirectPull = 1U;
                }
            } else {
                CHK_RET(CreateResourceContextFromGroups(
                    comm, param, groups, resCtxHost));
            }
        } else {
            CHK_RET(CreateResourceContext(
                comm, param, resCtxHost));
        }
        const std::vector<char> seq = resCtxHost.Serialize();
        const uint64_t seqSize = seq.size();
        param.ctxSize = seqSize;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, ccuEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, ccuEngine, param.tag, seq.data(), seqSize, 0));
    }

    CHK_RET(ops_hccl::ExecOp(param));
    return HCCL_SUCCESS;
}
