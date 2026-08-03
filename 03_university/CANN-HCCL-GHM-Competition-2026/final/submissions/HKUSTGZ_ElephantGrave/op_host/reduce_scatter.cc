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
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>
#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "exec_op.h"
#include "ccu_kernel.h"

namespace {

constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;
constexpr uint32_t MAX_NETWORK_GROUPS = 2;
constexpr uint32_t MAX_GROUP_SOURCE_COUNT = 8;
constexpr uint32_t MAX_ASYNC_STRIPES = 4;
constexpr uint32_t MAX_COMBINE_STRIPES = 8;
constexpr uint32_t HYBRID_MS_KERNEL_MODE = MAX_ASYNC_STRIPES + 1;
constexpr uint32_t HIERARCHICAL_KERNEL_MODE = HYBRID_MS_KERNEL_MODE + 1;
constexpr uint32_t WIDE_LANE_KERNEL_MODE_BASE = HIERARCHICAL_KERNEL_MODE;
constexpr uint32_t WIDE_LANE_WORKSPACE_BANKS = 2;
constexpr uint64_t SMALL_OUTPUT_THRESHOLD = 1U * 1024U * 1024U;
constexpr uint64_t MS_SLICE_BYTES = 4096U;
constexpr uint64_t MS_CHECKER_BUCKET_CAP = 640U * 1024U;
constexpr uint64_t MS_PARALLEL_LOOPS = 16U;
constexpr uint64_t MS_LOOP_BYTES = MS_SLICE_BYTES * MS_PARALLEL_LOOPS;
constexpr uint64_t RANK4_PACKED_MS_OUTPUT_BYTES = 2U * MS_LOOP_BYTES;
constexpr uint64_t MATE_RANK16_THRESHOLD = 16ULL * 1024ULL * 1024ULL;
constexpr uint64_t MATE_RANK12_THRESHOLD = 32ULL * 1024ULL * 1024ULL;
constexpr uint64_t MATE_ALIGNMENT = 4096U;
constexpr uint32_t WRITE_REDUCE_16_NODE_SIZE = 8U;
constexpr uint32_t WRITE_REDUCE_16_LOCAL_PEERS
    = WRITE_REDUCE_16_NODE_SIZE - 1U;
constexpr uint32_t WRITE_REDUCE_CROSS_STRIPES = 8U;
constexpr uint64_t RANK16_OWNER_PULL_512M_RECV_COUNT = 8388608U;
constexpr uint64_t RANK16_OWNER_PULL_TAIL_RECV_COUNT = 6553600U;

constexpr bool IsOfficialRank16OwnerPullCount(uint64_t recvCount)
{
    return recvCount == RANK16_OWNER_PULL_512M_RECV_COUNT
        || recvCount == RANK16_OWNER_PULL_TAIL_RECV_COUNT;
}

struct LayerChannelGroup {
    uint32_t layer{0};
    bool isIntra{false};
    std::vector<HcclChannelDesc> descs;
    std::vector<ChannelHandle> handles;
};

struct MateNodeLayout {
    uint32_t localBegin{0};
    uint32_t localSize{0};
    uint32_t remoteBegin{0};
    uint32_t remoteSize{0};
    uint32_t localIndex{0};
};

bool IsPreferredProtocol(CommProtocol protocol)
{
    return protocol == CommProtocol::COMM_PROTOCOL_UBC_CTP || protocol == CommProtocol::COMM_PROTOCOL_UBC_TP;
}

int32_t EndpointIoDieHint(const EndpointDesc &endpoint)
{
    if (endpoint.loc.locType != EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE
        || endpoint.commAddr.type != CommAddrType::COMM_ADDR_TYPE_EID) {
        return -1;
    }

    // Competition topology assigns eight endpoint EIDs per NPU.  Endpoints
    // 1..3 are on IO die 0, 4..7 on IO die 1, and endpoint 8 (Clos) on die 0.
    const uint16_t endpointValue
        = (static_cast<uint16_t>(endpoint.commAddr.eid[COMM_ADDR_EID_LEN - 2]) << 8)
        | endpoint.commAddr.eid[COMM_ADDR_EID_LEN - 1];
    if (endpointValue == 0) {
        return -1;
    }
    const uint16_t endpointOrdinal = static_cast<uint16_t>((endpointValue - 1U) % 8U + 1U);
    return endpointOrdinal >= 4U && endpointOrdinal <= 7U ? 1 : 0;
}

HcclResult MakeChannelDesc(const CommLink &link, uint32_t remoteRank, HcclChannelDesc &desc)
{
    CHK_RET(HcclChannelDescInit(&desc, 1));
    desc.remoteRank = remoteRank;
    desc.notifyNum = CHANNEL_NOTIFY_NUM;
    desc.channelProtocol = link.linkAttr.linkProtocol;
    desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
    desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
    desc.localEndpoint.loc = link.srcEndpointDesc.loc;
    desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
    desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
    desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
    return HCCL_SUCCESS;
}

HcclResult FindLayerChannel(HcclComm comm, uint32_t myRank, uint32_t remoteRank, const std::vector<uint32_t> &layers,
    int32_t preferredIoDieOverride, uint32_t &selectedLayer, bool &selectedIntra, HcclChannelDesc &selectedDesc)
{
    constexpr std::array<CommProtocol, 2> PROTOCOL_ORDER = {
        CommProtocol::COMM_PROTOCOL_UBC_CTP,
        CommProtocol::COMM_PROTOCOL_UBC_TP,
    };

    struct LayerMembership {
        uint32_t layer{0};
        std::vector<uint32_t> ranks;
    };
    std::vector<LayerMembership> candidates;
    candidates.reserve(layers.size());
    for (uint32_t layer : layers) {
        uint32_t *layerRanks = nullptr;
        uint32_t rankCount = 0;
        CHK_RET(HcclRankGraphGetRanksByLayer(comm, layer, &layerRanks, &rankCount));
        LayerMembership candidate;
        candidate.layer = layer;
        candidate.ranks.assign(layerRanks, layerRanks + rankCount);
        candidates.push_back(std::move(candidate));
    }

    // A network layer is a routing reachability layer, not an intra/inter
    // execution group. Search every layer for the requested IO die before
    // accepting a fallback route from an earlier layer.
    for (uint32_t routePass = 0; routePass < 2; ++routePass) {
        for (const auto &candidate : candidates) {
            if (std::find(candidate.ranks.begin(), candidate.ranks.end(), remoteRank)
                == candidate.ranks.end()) {
                continue;
            }
            CommLink *links = nullptr;
            uint32_t linkCount = 0;
            const HcclResult linkResult
                = HcclRankGraphGetLinks(comm, candidate.layer, myRank, remoteRank, &links, &linkCount);
            if (linkResult != HCCL_SUCCESS || links == nullptr || linkCount == 0) {
                continue;
            }
        for (CommProtocol expected : PROTOCOL_ORDER) {
            for (uint32_t i = 0; i < linkCount; ++i) {
                if (links[i].linkAttr.linkProtocol != expected
                    || !IsPreferredProtocol(links[i].linkAttr.linkProtocol)) {
                    continue;
                }
                const bool sameServer = links[i].srcEndpointDesc.loc.locType
                        == EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE
                    && links[i].dstEndpointDesc.loc.locType == EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE
                    && links[i].srcEndpointDesc.loc.device.serverIdx
                        == links[i].dstEndpointDesc.loc.device.serverIdx;
                const int32_t preferredDie
                    = preferredIoDieOverride >= 0 ? preferredIoDieOverride : (sameServer ? 1 : 0);
                const int32_t candidateDie = EndpointIoDieHint(links[i].srcEndpointDesc);
                if (routePass == 0 && candidateDie != preferredDie) {
                    continue;
                }
                CHK_RET(MakeChannelDesc(links[i], remoteRank, selectedDesc));
                selectedLayer = candidate.layer;
                selectedIntra = sameServer;
                return HCCL_SUCCESS;
            }
        }
        }
    }

    HCCL_ERROR("[FindLayerChannel] no CCU link from rank %u to rank %u", myRank, remoteRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult BuildLayerGroups(HcclComm comm, const OpParam &param, const HierarchyRankPlan *hierarchyPlan,
    const MateNodeLayout *mateLayout, std::vector<LayerChannelGroup> &groups)
{
    uint32_t *netLayers = nullptr;
    uint32_t layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &layerCount));
    CHK_PRT_RET(layerCount == 0, HCCL_ERROR("[BuildLayerGroups] rank graph has no network layer"), HCCL_E_NOT_FOUND);
    const std::vector<uint32_t> layers(netLayers, netLayers + layerCount);

    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }

        uint32_t selectedLayer = 0;
        bool selectedIntra = false;
        const bool hierarchyIntra = hierarchyPlan != nullptr
            && remoteRank >= hierarchyPlan->localBegin
            && remoteRank < hierarchyPlan->localBegin + hierarchyPlan->localSize;
        const bool mateIntra = mateLayout != nullptr
            && remoteRank >= mateLayout->localBegin
            && remoteRank < mateLayout->localBegin + mateLayout->localSize;
        const bool logicalGrouping = hierarchyPlan != nullptr || mateLayout != nullptr;
        const bool logicalIntra = hierarchyPlan != nullptr ? hierarchyIntra : mateIntra;
        const int32_t preferredIoDie = logicalGrouping ? (logicalIntra ? 1 : 0) : -1;
        HcclChannelDesc desc;
        CHK_RET(FindLayerChannel(
            comm, param.myRank, remoteRank, layers, preferredIoDie, selectedLayer, selectedIntra, desc));
        if (logicalGrouping) {
            // Communicator membership, not endpoint routing metadata, defines
            // which phase consumes this peer.
            selectedIntra = logicalIntra;
        }

        auto groupIt = std::find_if(groups.begin(), groups.end(),
            [selectedLayer, selectedIntra, logicalGrouping](const LayerChannelGroup &group) {
            return logicalGrouping ? group.isIntra == selectedIntra : group.layer == selectedLayer;
        });
        if (groupIt == groups.end()) {
            groups.push_back(LayerChannelGroup{});
            groups.back().layer = selectedLayer;
            groups.back().isIntra = selectedIntra;
            groupIt = groups.end() - 1;
        } else if (!logicalGrouping) {
            CHK_PRT_RET(groupIt->isIntra != selectedIntra,
                HCCL_ERROR("[BuildLayerGroups] layer %u mixes intra- and inter-node channels", selectedLayer),
                HCCL_E_NOT_SUPPORT);
        }
        groupIt->descs.push_back(desc);
    }

    std::sort(groups.begin(), groups.end(), [](const LayerChannelGroup &lhs, const LayerChannelGroup &rhs) {
        if (lhs.isIntra != rhs.isIntra) {
            return lhs.isIntra;
        }
        return lhs.layer < rhs.layer;
    });
    CHK_PRT_RET(groups.empty() && param.rankSize > 1, HCCL_ERROR("[BuildLayerGroups] no channel group was created"),
        HCCL_E_NOT_FOUND);
    CHK_PRT_RET(groups.size() > MAX_NETWORK_GROUPS,
        HCCL_ERROR(
            "[BuildLayerGroups] expected at most %u active network layers, got %zu", MAX_NETWORK_GROUPS, groups.size()),
        HCCL_E_NOT_SUPPORT);

    if (hierarchyPlan != nullptr) {
        CHK_PRT_RET(groups.size() != 2 || !groups[0].isIntra || groups[1].isIntra,
            HCCL_ERROR("[BuildLayerGroups] hierarchical schedule requires one intra and one inter execution group"),
            HCCL_E_NOT_SUPPORT);
        std::vector<HcclChannelDesc> sparseCross;
        for (const auto &desc : groups[1].descs) {
            for (uint32_t i = 0; i < hierarchyPlan->crossPeerCount; ++i) {
                if (desc.remoteRank == hierarchyPlan->crossPeers[i]) {
                    sparseCross.push_back(desc);
                    break;
                }
            }
        }
        CHK_PRT_RET(sparseCross.size() != hierarchyPlan->crossPeerCount,
            HCCL_ERROR("[BuildLayerGroups] could not build all sparse hierarchy edges"), HCCL_E_NOT_FOUND);
        groups[1].descs.swap(sparseCross);
    }
    if (mateLayout != nullptr) {
        CHK_PRT_RET(groups.size() != 2 || !groups[0].isIntra || groups[1].isIntra,
            HCCL_ERROR("[BuildLayerGroups] MATE requires one intra and one inter execution group"),
            HCCL_E_NOT_SUPPORT);
        CHK_PRT_RET(groups[0].descs.size() != mateLayout->localSize - 1U
                || groups[1].descs.size() != mateLayout->remoteSize,
            HCCL_ERROR("[BuildLayerGroups] MATE peer counts do not match node layout"),
            HCCL_E_NOT_SUPPORT);
    }

    for (auto &group : groups) {
        group.handles.resize(group.descs.size());
        CHK_PRT_RET(group.descs.size() > MAX_GROUP_SOURCE_COUNT,
            HCCL_ERROR("[BuildLayerGroups] layer %u has %zu peers, CCU supports at most %u", group.layer,
                group.descs.size(), MAX_GROUP_SOURCE_COUNT),
            HCCL_E_NOT_SUPPORT);
        CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, group.descs.data(),
            static_cast<uint32_t>(group.descs.size()), group.handles.data()));
    }
    return HCCL_SUCCESS;
}

HcclResult AcquireThreads(HcclComm comm, const OpParam &param, size_t groupCount, AlgResourceCtx &resource)
{
    const size_t threadCount = std::max<size_t>(1, groupCount);
    resource.threads.resize(threadCount);
    resource.threads[0] = param.cpuThread;
    resource.ccuThread = param.cpuThread;
    if (threadCount > 1) {
        CHK_RET(HcclThreadAcquire(
            comm, CommEngine::COMM_ENGINE_CCU, static_cast<uint32_t>(threadCount - 1), 1, &resource.threads[1]));
    }
    return HCCL_SUCCESS;
}

struct MateByteRange {
    uint64_t offset{0};
    uint64_t length{0};
};

bool BuildMateNodeLayout(uint32_t rankSize, uint32_t myRank, MateNodeLayout &layout)
{
    if (myRank >= rankSize) {
        return false;
    }
    if (rankSize == 16) {
        layout.localBegin = myRank < 8 ? 0 : 8;
        layout.localSize = 8;
        layout.remoteBegin = myRank < 8 ? 8 : 0;
        layout.remoteSize = 8;
    } else if (rankSize == 12) {
        if (myRank < 8) {
            layout.localBegin = 0;
            layout.localSize = 8;
            layout.remoteBegin = 8;
            layout.remoteSize = 4;
        } else {
            layout.localBegin = 8;
            layout.localSize = 4;
            layout.remoteBegin = 0;
            layout.remoteSize = 8;
        }
    } else {
        return false;
    }
    layout.localIndex = myRank - layout.localBegin;
    return true;
}

MateByteRange SplitMateRange(uint64_t elementCount, uint32_t partCount, uint32_t partIndex)
{
    const uint64_t base = elementCount / partCount;
    const uint64_t extra = elementCount % partCount;
    const uint64_t beginElements
        = static_cast<uint64_t>(partIndex) * base + std::min<uint64_t>(partIndex, extra);
    const uint64_t partElements = base + static_cast<uint64_t>(partIndex < extra);
    return MateByteRange{beginElements * sizeof(float), partElements * sizeof(float)};
}

bool AlignMateSize(uint64_t value, uint64_t &aligned)
{
    if (value > std::numeric_limits<uint64_t>::max() - (MATE_ALIGNMENT - 1U)) {
        return false;
    }
    aligned = (value + MATE_ALIGNMENT - 1U) & ~(MATE_ALIGNMENT - 1U);
    return true;
}

struct HybridWorkspacePlan {
    MateByteRange mateRange;
    MateByteRange directRange;
    uint64_t mailboxStride{0};
    uint64_t mailboxBytes{0};
    uint64_t wideWorkspaceBase{0};
    uint64_t wideStride{0};
    std::array<uint64_t, DIRECT_TILE_COUNT> directTileLengths{};
    uint64_t required{0};
};

bool ScaleMateElements(uint64_t elementCount, uint32_t numerator,
    uint32_t denominator, uint64_t &scaled)
{
    const uint64_t whole = elementCount / denominator;
    if (whole > std::numeric_limits<uint64_t>::max() / numerator) {
        return false;
    }
    scaled = whole * numerator
        + ((elementCount % denominator) * numerator) / denominator;
    return true;
}

bool BuildDirectWideTiles(uint64_t directBytes,
    std::array<uint64_t, DIRECT_TILE_COUNT> &lengths, uint64_t &workspaceStride)
{
    constexpr uint64_t DATA_TYPE_SIZE = sizeof(float);
    if (DIRECT_TILE_COUNT != 3U || directBytes % DATA_TYPE_SIZE != 0
        || directBytes < DIRECT_TILE_COUNT * MATE_ALIGNMENT) {
        return false;
    }
    const uint64_t tileBytes
        = (directBytes / DIRECT_TILE_COUNT / MATE_ALIGNMENT) * MATE_ALIGNMENT;
    if (tileBytes == 0 || tileBytes > directBytes / 2U) {
        return false;
    }
    lengths = {tileBytes, tileBytes, directBytes - 2U * tileBytes};
    const uint64_t maxLength = *std::max_element(lengths.begin(), lengths.end());
    return lengths[2] != 0 && AlignMateSize(maxLength, workspaceStride)
        && workspaceStride != 0;
}

bool CalculateHybridWorkspace(uint64_t elementCount, uint32_t rankSize,
    uint32_t localSize, uint32_t remoteSize, HybridWorkspacePlan &plan)
{
    if (localSize < 2U || localSize > MATE_MAX_PEERS
        || remoteSize == 0 || remoteSize > MATE_MAX_PEERS
        || elementCount > std::numeric_limits<uint64_t>::max() / sizeof(float)) {
        return false;
    }
    const uint32_t numerator = rankSize == 16 ? 3U : (rankSize == 12 ? 1U : 0U);
    const uint32_t denominator = rankSize == 16 ? 8U : (rankSize == 12 ? 4U : 0U);
    if (numerator == 0 || denominator == 0) {
        return false;
    }
    uint64_t mateElements = 0;
    if (!ScaleMateElements(elementCount, numerator, denominator, mateElements)) {
        return false;
    }
    constexpr uint64_t ALIGN_ELEMENTS = MATE_ALIGNMENT / sizeof(float);
    mateElements = (mateElements / ALIGN_ELEMENTS) * ALIGN_ELEMENTS;
    if (mateElements == 0 || mateElements >= elementCount) {
        return false;
    }
    plan.mateRange = MateByteRange{0, mateElements * sizeof(float)};
    plan.directRange = MateByteRange{
        plan.mateRange.length, (elementCount - mateElements) * sizeof(float)};

    const uint64_t maxShardElements
        = mateElements / localSize + static_cast<uint64_t>(mateElements % localSize != 0);
    if (maxShardElements > std::numeric_limits<uint64_t>::max() / sizeof(float)
        || !AlignMateSize(maxShardElements * sizeof(float), plan.mailboxStride)
        || plan.mailboxStride > std::numeric_limits<uint64_t>::max() / remoteSize) {
        return false;
    }
    plan.mailboxBytes = plan.mailboxStride * remoteSize;
    plan.wideWorkspaceBase = plan.mailboxBytes;
    if (!BuildDirectWideTiles(plan.directRange.length,
            plan.directTileLengths, plan.wideStride)) {
        return false;
    }
    const uint32_t maxSourceCount = std::max(localSize, remoteSize);
    const uint32_t wideSlotCount
        = WIDE_LANE_WORKSPACE_BANKS * (maxSourceCount - 1U);
    if (wideSlotCount == 0
        || plan.wideStride > std::numeric_limits<uint64_t>::max() / wideSlotCount) {
        return false;
    }
    const uint64_t wideBytes = plan.wideStride * wideSlotCount;
    if (plan.mailboxBytes > std::numeric_limits<uint64_t>::max() - wideBytes) {
        return false;
    }
    plan.required = plan.mailboxBytes + wideBytes;
    return true;
}

struct Hetero12WorkspacePlan {
    uint64_t outputBytes{0};
    uint64_t helperShardBytes{0};
    uint64_t prefixBytes{0};
    uint64_t suffixBytes{0};
    uint64_t h8MailboxBase{0};
    uint64_t h8MailboxBytes{0};
    uint64_t h8WideWorkspaceBase{0};
    uint64_t h8Required{0};
    uint64_t h4PrefixWorkspaceBase{0};
    uint64_t h4PrefixWorkspaceBytes{0};
    uint64_t h4SuffixWorkspaceBase{0};
    uint64_t h4SuffixWorkspaceBytes{0};
    uint64_t h4Required{0};
    uint64_t fullWideStride{0};
    uint64_t prefixWideStride{0};
    uint64_t suffixWideStride{0};
    std::array<uint64_t, DIRECT_TILE_COUNT> fullTileLengths{};
    std::array<uint64_t, DIRECT_TILE_COUNT> prefixTileLengths{};
    std::array<uint64_t, DIRECT_TILE_COUNT> suffixTileLengths{};
};

bool MultiplyMateSize(uint64_t value, uint64_t factor, uint64_t &product)
{
    if (factor != 0 && value > std::numeric_limits<uint64_t>::max() / factor) {
        return false;
    }
    product = value * factor;
    return true;
}

bool AddMateSize(uint64_t lhs, uint64_t rhs, uint64_t &sum)
{
    if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
        return false;
    }
    sum = lhs + rhs;
    return true;
}

bool CalculateHetero12Workspace(uint64_t elementCount, Hetero12WorkspacePlan &plan)
{
    constexpr uint64_t ALIGN_ELEMENTS = MATE_ALIGNMENT / sizeof(float);
    constexpr uint64_t OWN_PIECE_COUNT = 12U;
    constexpr uint64_t H8_HELPER_COUNT = 4U;
    constexpr uint64_t H4_HELPER_COUNT = 8U;
    constexpr uint64_t H8_WIDE_SLOT_COUNT
        = WIDE_LANE_WORKSPACE_BANKS * (H8_HELPER_COUNT - 1U);
    constexpr uint64_t H4_PREFIX_WIDE_SLOT_COUNT
        = WIDE_LANE_WORKSPACE_BANKS * (H8_HELPER_COUNT - 1U);
    constexpr uint64_t H4_SUFFIX_WIDE_SLOT_COUNT
        = WIDE_LANE_WORKSPACE_BANKS * (H4_HELPER_COUNT - 1U);

    if (elementCount == 0
        || elementCount > std::numeric_limits<uint64_t>::max() / sizeof(float)) {
        return false;
    }
    const uint64_t helperShardElements
        = (elementCount / (OWN_PIECE_COUNT * ALIGN_ELEMENTS)) * ALIGN_ELEMENTS;
    if (helperShardElements == 0
        || helperShardElements > std::numeric_limits<uint64_t>::max() / sizeof(float)) {
        return false;
    }
    uint64_t prefixElements = 0;
    if (!MultiplyMateSize(helperShardElements, H4_HELPER_COUNT, prefixElements)
        || prefixElements >= elementCount) {
        return false;
    }

    plan.outputBytes = elementCount * sizeof(float);
    plan.helperShardBytes = helperShardElements * sizeof(float);
    plan.prefixBytes = prefixElements * sizeof(float);
    plan.suffixBytes = plan.outputBytes - plan.prefixBytes;
    if (!BuildDirectWideTiles(
            plan.outputBytes, plan.fullTileLengths, plan.fullWideStride)
        || !BuildDirectWideTiles(
            plan.prefixBytes, plan.prefixTileLengths, plan.prefixWideStride)
        || !BuildDirectWideTiles(
            plan.suffixBytes, plan.suffixTileLengths, plan.suffixWideStride)
        || plan.prefixWideStride < plan.suffixWideStride
        || !AlignMateSize(plan.outputBytes, plan.h8MailboxBase)
        || !MultiplyMateSize(
            plan.helperShardBytes, H8_HELPER_COUNT, plan.h8MailboxBytes)
        || !AddMateSize(
            plan.h8MailboxBase, plan.h8MailboxBytes, plan.h8WideWorkspaceBase)) {
        return false;
    }

    uint64_t h8WideBytes = 0;
    if (!MultiplyMateSize(plan.fullWideStride, H8_WIDE_SLOT_COUNT, h8WideBytes)
        || !AddMateSize(plan.h8WideWorkspaceBase, h8WideBytes, plan.h8Required)) {
        return false;
    }

    plan.h4PrefixWorkspaceBase = 0;
    if (!MultiplyMateSize(plan.prefixWideStride, H4_PREFIX_WIDE_SLOT_COUNT,
            plan.h4PrefixWorkspaceBytes)) {
        return false;
    }
    plan.h4SuffixWorkspaceBase = plan.h4PrefixWorkspaceBytes;
    if (!MultiplyMateSize(plan.suffixWideStride, H4_SUFFIX_WIDE_SLOT_COUNT,
            plan.h4SuffixWorkspaceBytes)
        || !AddMateSize(plan.h4SuffixWorkspaceBase,
            plan.h4SuffixWorkspaceBytes, plan.h4Required)) {
        return false;
    }
    return true;
}

bool ShouldUseMate(const OpParam &param, uint64_t localBufferSize)
{
    const uint64_t outputBytes = param.count * sizeof(float);
    if ((param.rankSize == 16 && outputBytes < MATE_RANK16_THRESHOLD)
        || (param.rankSize == 12 && outputBytes < MATE_RANK12_THRESHOLD)
        || (param.rankSize != 16 && param.rankSize != 12)) {
        return false;
    }

    HybridWorkspacePlan plan;
    if (param.rankSize == 16) {
        return CalculateHybridWorkspace(param.count, param.rankSize, 8, 8, plan)
            && plan.required <= localBufferSize;
    }

    // Rank-12 dual push writes directly to recvBuf and does not consume the
    // former one-way-MATE workspace.  The common gate is retained only to
    // request the logical intra/cross channel grouping and two CCU threads.
    return true;
}

constexpr uint64_t CompactLowBits(uint32_t bitCount)
{
    return (uint64_t{1} << bitCount) - 1U;
}

uint64_t EncodeCompactParallel(uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum)
{
    return ((repeatNum & CompactLowBits(7)) << 55U)
        | ((repeatLoopIndex & CompactLowBits(7)) << 48U)
        | ((totalLoopNum & CompactLowBits(7)) << 41U);
}

std::array<uint64_t, 4> BuildCompactMsGoSize(uint64_t size)
{
    const uint64_t mainLoopCount = size / MS_LOOP_BYTES;
    const uint64_t mainBytes = mainLoopCount * MS_LOOP_BYTES;
    const uint64_t tailBytes = size - mainBytes;
    const uint64_t fullTailSlices = tailBytes / MS_SLICE_BYTES;
    const uint64_t residual = tailBytes - fullTailSlices * MS_SLICE_BYTES;
    uint64_t parallelParam = 0;
    uint64_t tailSliceBytes = 0;
    if (fullTailSlices != 0 && residual == 0) {
        parallelParam = EncodeCompactParallel(fullTailSlices - 1U, 0, 1);
        tailSliceBytes = MS_SLICE_BYTES;
    } else if (fullTailSlices == 0 && residual != 0) {
        parallelParam = EncodeCompactParallel(0, 0, 1);
        tailSliceBytes = residual;
    } else if (fullTailSlices != 0 && residual != 0) {
        parallelParam = EncodeCompactParallel(fullTailSlices - 1U, 1, 2);
        tailSliceBytes = residual;
    }
    return {mainBytes, mainLoopCount, parallelParam, tailSliceBytes};
}

bool BuildWriteReduce4x1Stripes(uint64_t elementCount,
    std::array<uint64_t, ops_hccl::WRITE_REDUCE_4X1_STRIPES> &offsets,
    std::array<uint64_t, ops_hccl::WRITE_REDUCE_4X1_STRIPES> &lengths)
{
    constexpr uint64_t DATA_TYPE_SIZE = sizeof(float);
    constexpr uint64_t ALIGN_ELEMENTS = MATE_ALIGNMENT / DATA_TYPE_SIZE;
    if (elementCount < ops_hccl::WRITE_REDUCE_4X1_STRIPES
        || elementCount > std::numeric_limits<uint64_t>::max() / DATA_TYPE_SIZE) {
        return false;
    }

    const uint64_t average
        = elementCount / ops_hccl::WRITE_REDUCE_4X1_STRIPES;
    const uint64_t alignedElements
        = (average / ALIGN_ELEMENTS) * ALIGN_ELEMENTS;
    uint64_t nextOffsetElements = 0;
    for (uint32_t stripe = 0;
        stripe < ops_hccl::WRITE_REDUCE_4X1_STRIPES; ++stripe) {
        offsets[stripe] = nextOffsetElements * DATA_TYPE_SIZE;
        uint64_t stripeElements = 0;
        if (alignedElements != 0) {
            stripeElements
                = stripe + 1U == ops_hccl::WRITE_REDUCE_4X1_STRIPES
                ? elementCount - nextOffsetElements
                : alignedElements;
        } else {
            const uint64_t base
                = elementCount / ops_hccl::WRITE_REDUCE_4X1_STRIPES;
            const uint64_t extra
                = elementCount % ops_hccl::WRITE_REDUCE_4X1_STRIPES;
            stripeElements
                = base + static_cast<uint64_t>(stripe < extra);
        }
        if (stripeElements == 0
            || nextOffsetElements > elementCount - stripeElements) {
            return false;
        }
        lengths[stripe] = stripeElements * DATA_TYPE_SIZE;
        nextOffsetElements += stripeElements;
    }
    return nextOffsetElements == elementCount;
}

bool BuildWriteReduceDualStripes(uint64_t elementCount, uint32_t stripeCount,
    std::array<uint64_t, ops_hccl::WRITE_REDUCE_DUAL_MAX_CHANNELS> &offsets,
    std::array<uint64_t, ops_hccl::WRITE_REDUCE_DUAL_MAX_CHANNELS> &lengths)
{
    constexpr uint64_t DATA_TYPE_SIZE = sizeof(float);
    constexpr uint64_t ALIGN_ELEMENTS = MATE_ALIGNMENT / DATA_TYPE_SIZE;
    if (stripeCount == 0
        || stripeCount > ops_hccl::WRITE_REDUCE_DUAL_MAX_CHANNELS
        || elementCount < stripeCount
        || elementCount > std::numeric_limits<uint64_t>::max()
            / DATA_TYPE_SIZE) {
        return false;
    }

    offsets.fill(0);
    lengths.fill(0);
    const uint64_t average = elementCount / stripeCount;
    const uint64_t alignedElements
        = (average / ALIGN_ELEMENTS) * ALIGN_ELEMENTS;
    uint64_t nextOffsetElements = 0;
    for (uint32_t stripe = 0; stripe < stripeCount; ++stripe) {
        offsets[stripe] = nextOffsetElements * DATA_TYPE_SIZE;
        uint64_t stripeElements = 0;
        if (alignedElements != 0) {
            stripeElements = stripe + 1U == stripeCount
                ? elementCount - nextOffsetElements
                : alignedElements;
        } else {
            const uint64_t base = elementCount / stripeCount;
            const uint64_t extra = elementCount % stripeCount;
            stripeElements
                = base + static_cast<uint64_t>(stripe < extra);
        }
        if (stripeElements == 0
            || nextOffsetElements > elementCount - stripeElements) {
            return false;
        }
        lengths[stripe] = stripeElements * DATA_TYPE_SIZE;
        nextOffsetElements += stripeElements;
    }
    return nextOffsetElements == elementCount;
}

bool FillWriteReduceDualArg(const OpParam &param,
    const MateNodeLayout &layout, const LayerChannelGroup &group,
    uint32_t stripeCount, bool initializeRange,
    uint64_t phaseOffset, uint64_t phaseLength,
    const std::array<uint64_t, ops_hccl::WRITE_REDUCE_DUAL_MAX_CHANNELS>
        &stripeOffsets,
    const std::array<uint64_t, ops_hccl::WRITE_REDUCE_DUAL_MAX_CHANNELS>
        &stripeLengths,
    ops_hccl::CcuKernelArgWriteReduceDual &arg)
{
    const uint32_t expectedChannels = group.isIntra
        ? layout.localSize - 1U
        : layout.remoteSize;
    const uint64_t outputBytes = param.count * sizeof(float);
    if ((param.rankSize != 12U && param.rankSize != 16U)
        || layout.localSize < 2U
        || layout.localSize > WRITE_REDUCE_CROSS_STRIPES
        || layout.remoteSize == 0U
        || layout.remoteSize > WRITE_REDUCE_CROSS_STRIPES
        || layout.localSize + layout.remoteSize != param.rankSize
        || stripeCount == 0U
        || stripeCount > ops_hccl::WRITE_REDUCE_DUAL_MAX_CHANNELS
        || group.handles.size() != expectedChannels
        || group.descs.size() != expectedChannels
        || phaseLength == 0 || phaseOffset > outputBytes
        || phaseLength > outputBytes - phaseOffset) {
        return false;
    }

    arg.dataType = param.dataType;
    arg.reduceOp = param.reduceType;
    arg.myRank = param.myRank;
    arg.rankSize = param.rankSize;
    arg.stripeCount = stripeCount;
    arg.initializeRange = initializeRange;
    arg.outputBytes = outputBytes;
    arg.phaseOffset = phaseOffset;
    arg.phaseLength = phaseLength;
    arg.channelCount = expectedChannels;
    arg.stripeOffsets = stripeOffsets;
    arg.stripeLengths = stripeLengths;

    std::array<bool, MAX_RANK_SIZE> seenTarget{};
    for (uint32_t channel = 0; channel < expectedChannels; ++channel) {
        const uint32_t target = group.descs[channel].remoteRank;
        const bool targetIsLocal = target >= layout.localBegin
            && target < layout.localBegin + layout.localSize;
        const bool targetIsRemote = target >= layout.remoteBegin
            && target < layout.remoteBegin + layout.remoteSize;
        if (target >= param.rankSize || target == param.myRank
            || seenTarget[target]
            || (group.isIntra ? !targetIsLocal : !targetIsRemote)) {
            return false;
        }
        seenTarget[target] = true;
        arg.channels[channel] = group.handles[channel];
        arg.targetRanks[channel] = target;
        if (group.isIntra) {
            const uint32_t targetLocal = target - layout.localBegin;
            arg.sourceIndicesAtTarget[channel]
                = layout.localIndex < targetLocal
                ? layout.localIndex
                : layout.localIndex - 1U;
        } else {
            arg.sourceIndicesAtTarget[channel] = layout.localIndex;
        }
        if (arg.sourceIndicesAtTarget[channel] >= stripeCount) {
            return false;
        }
    }
    return true;
}

bool FillOwnerPullDualArg(const OpParam &param,
    const MateNodeLayout &layout, const LayerChannelGroup &group,
    uint32_t stripeCount, bool initializeRange,
    uint64_t phaseOffset, uint64_t phaseLength,
    const std::array<uint64_t, ops_hccl::WRITE_REDUCE_DUAL_MAX_CHANNELS>
        &stripeOffsets,
    const std::array<uint64_t, ops_hccl::WRITE_REDUCE_DUAL_MAX_CHANNELS>
        &stripeLengths,
    ops_hccl::CcuKernelArgWriteReduceDual &arg)
{
    if (param.rankSize != 16U
        || !FillWriteReduceDualArg(param, layout, group, stripeCount,
            initializeRange, phaseOffset, phaseLength, stripeOffsets,
            stripeLengths, arg)) {
        return false;
    }

    // FillWriteReduceDualArg numbers an outgoing source in each target's
    // peer list. Owner-pull runs at the target, so local channels instead
    // number the remote source in this owner's peer list. Rank-16 cross uses
    // a symmetric 8x8 Latin residue.
    std::array<bool, ops_hccl::WRITE_REDUCE_DUAL_MAX_CHANNELS>
        seenResidue{};
    for (uint32_t channel = 0; channel < arg.channelCount; ++channel) {
        const uint32_t source = arg.targetRanks[channel];
        uint32_t residue = 0U;
        if (group.isIntra) {
            const uint32_t sourceLocal = source - layout.localBegin;
            residue = sourceLocal < layout.localIndex
                ? sourceLocal : sourceLocal - 1U;
        } else {
            const uint32_t sourceRemote = source - layout.remoteBegin;
            residue = (layout.localIndex + sourceRemote) % stripeCount;
        }
        if (residue >= stripeCount || seenResidue[residue]) {
            return false;
        }
        seenResidue[residue] = true;
        arg.sourceIndicesAtTarget[channel] = residue;
    }
    return true;
}

bool FillRank16DualArg(bool ownerPull, const OpParam &param,
    const MateNodeLayout &layout, const LayerChannelGroup &group,
    uint32_t stripeCount, bool initializeRange,
    uint64_t phaseOffset, uint64_t phaseLength,
    const std::array<uint64_t, ops_hccl::WRITE_REDUCE_DUAL_MAX_CHANNELS>
        &stripeOffsets,
    const std::array<uint64_t, ops_hccl::WRITE_REDUCE_DUAL_MAX_CHANNELS>
        &stripeLengths,
    ops_hccl::CcuKernelArgWriteReduceDual &arg)
{
    if (ownerPull) {
        return FillOwnerPullDualArg(param, layout, group, stripeCount,
            initializeRange, phaseOffset, phaseLength, stripeOffsets,
            stripeLengths, arg);
    }
    return FillWriteReduceDualArg(param, layout, group, stripeCount,
        initializeRange, phaseOffset, phaseLength, stripeOffsets,
        stripeLengths, arg);
}

HcclResult RegisterWriteReduce16Kernels(HcclComm comm, const OpParam &param,
    const std::vector<LayerChannelGroup> &groups,
    const MateNodeLayout &layout, bool ownerPull, AlgResourceCtx &resource)
{
    CHK_PRT_RET(param.rankSize != 2U * WRITE_REDUCE_16_NODE_SIZE
            || groups.size() != 2U
            || layout.localSize != WRITE_REDUCE_16_NODE_SIZE
            || layout.remoteSize != WRITE_REDUCE_16_NODE_SIZE,
        HCCL_ERROR("[RegisterWriteReduce16Kernels] invalid rank-16 layout"),
        HCCL_E_INTERNAL);

    const LayerChannelGroup *intraGroup = nullptr;
    const LayerChannelGroup *crossGroup = nullptr;
    for (const auto &group : groups) {
        if (group.isIntra) {
            intraGroup = &group;
        } else {
            crossGroup = &group;
        }
    }
    CHK_PRT_RET(intraGroup == nullptr || crossGroup == nullptr
            || intraGroup->handles.size() != WRITE_REDUCE_16_LOCAL_PEERS
            || crossGroup->handles.size() != WRITE_REDUCE_16_NODE_SIZE,
        HCCL_ERROR(
            "[RegisterWriteReduce16Kernels] missing 7-way local or 8-way cross group"),
        HCCL_E_INTERNAL);

    constexpr uint64_t ALIGN_ELEMENTS = MATE_ALIGNMENT / sizeof(float);
    const uint64_t halfElements = param.count / 2U;
    const uint64_t phaseAElements
        = (halfElements / ALIGN_ELEMENTS) * ALIGN_ELEMENTS;
    const uint64_t phaseBElements = param.count - phaseAElements;
    CHK_PRT_RET(phaseAElements < WRITE_REDUCE_16_NODE_SIZE
            || phaseBElements < WRITE_REDUCE_16_NODE_SIZE,
        HCCL_ERROR(
            "[RegisterWriteReduce16Kernels] output is too small for two phases"),
        HCCL_E_INTERNAL);
    const uint64_t phaseABytes = phaseAElements * sizeof(float);
    const uint64_t phaseBBytes = phaseBElements * sizeof(float);

    std::array<uint64_t, ops_hccl::WRITE_REDUCE_DUAL_MAX_CHANNELS>
        localAOffsets{};
    std::array<uint64_t, ops_hccl::WRITE_REDUCE_DUAL_MAX_CHANNELS>
        localALengths{};
    std::array<uint64_t, ops_hccl::WRITE_REDUCE_DUAL_MAX_CHANNELS>
        crossAOffsets{};
    std::array<uint64_t, ops_hccl::WRITE_REDUCE_DUAL_MAX_CHANNELS>
        crossALengths{};
    std::array<uint64_t, ops_hccl::WRITE_REDUCE_DUAL_MAX_CHANNELS>
        localBOffsets{};
    std::array<uint64_t, ops_hccl::WRITE_REDUCE_DUAL_MAX_CHANNELS>
        localBLengths{};
    std::array<uint64_t, ops_hccl::WRITE_REDUCE_DUAL_MAX_CHANNELS>
        crossBOffsets{};
    std::array<uint64_t, ops_hccl::WRITE_REDUCE_DUAL_MAX_CHANNELS>
        crossBLengths{};
    CHK_PRT_RET(
        !BuildWriteReduceDualStripes(phaseAElements,
            WRITE_REDUCE_16_LOCAL_PEERS, localAOffsets, localALengths)
            || !BuildWriteReduceDualStripes(phaseAElements,
                ownerPull ? 1U : WRITE_REDUCE_16_NODE_SIZE,
                crossAOffsets, crossALengths)
            || !BuildWriteReduceDualStripes(phaseBElements,
                WRITE_REDUCE_16_LOCAL_PEERS, localBOffsets, localBLengths)
            || !BuildWriteReduceDualStripes(phaseBElements,
                ownerPull ? 1U : WRITE_REDUCE_16_NODE_SIZE,
                crossBOffsets, crossBLengths),
        HCCL_ERROR(
            "[RegisterWriteReduce16Kernels] invalid aligned stripe partition"),
        HCCL_E_INTERNAL);

    std::array<std::shared_ptr<ops_hccl::CcuKernelArgWriteReduceDual>, 4>
        kernelArgs = {
            std::make_shared<ops_hccl::CcuKernelArgWriteReduceDual>(),
            std::make_shared<ops_hccl::CcuKernelArgWriteReduceDual>(),
            std::make_shared<ops_hccl::CcuKernelArgWriteReduceDual>(),
            std::make_shared<ops_hccl::CcuKernelArgWriteReduceDual>(),
        };
    CHK_PRT_RET(
        !FillRank16DualArg(ownerPull, param, layout, *intraGroup,
            WRITE_REDUCE_16_LOCAL_PEERS, true,
            0, phaseABytes, localAOffsets, localALengths, *kernelArgs[0])
            || !FillRank16DualArg(ownerPull, param, layout, *crossGroup,
                WRITE_REDUCE_CROSS_STRIPES, true,
                phaseABytes, phaseBBytes, crossBOffsets, crossBLengths,
                *kernelArgs[1])
            || !FillRank16DualArg(ownerPull, param, layout, *intraGroup,
                WRITE_REDUCE_16_LOCAL_PEERS, false,
                phaseABytes, phaseBBytes, localBOffsets, localBLengths,
                *kernelArgs[2])
            || !FillRank16DualArg(ownerPull, param, layout, *crossGroup,
                WRITE_REDUCE_CROSS_STRIPES, false,
                0, phaseABytes, crossAOffsets, crossALengths,
                *kernelArgs[3]),
        HCCL_ERROR(
            "[RegisterWriteReduce16Kernels] could not build phase arguments"),
        HCCL_E_INTERNAL);

    constexpr std::array<const char *, 4> KERNEL_NAMES = {
        "CcuReduceScatterWriteReduce16LocalA",
        "CcuReduceScatterWriteReduce16CrossB",
        "CcuReduceScatterWriteReduce16LocalB",
        "CcuReduceScatterWriteReduce16CrossA",
    };
    constexpr std::array<const char *, 4> OWNER_PULL_KERNEL_NAMES = {
        "CcuReduceScatterOwnerPull16LocalA",
        "CcuReduceScatterOwnerPull16CrossB",
        "CcuReduceScatterOwnerPull16LocalB",
        "CcuReduceScatterOwnerPull16CrossA",
    };
    constexpr std::array<uint32_t, 4> KERNEL_MODES = {
        WRITE_REDUCE_16_LOCAL_A_MODE,
        WRITE_REDUCE_16_CROSS_B_MODE,
        WRITE_REDUCE_16_LOCAL_B_MODE,
        WRITE_REDUCE_16_CROSS_A_MODE,
    };
    std::array<CcuKernelInfo, 4> kernelInfos{};
    resource.ccuKernels.resize(kernelInfos.size());
    resource.stripeCounts.resize(kernelInfos.size());
    resource.combineKernel = 0;
    for (uint32_t kernel = 0; kernel < kernelInfos.size(); ++kernel) {
        std::snprintf(kernelInfos[kernel].kernelFuncName,
            sizeof(kernelInfos[kernel].kernelFuncName), "%s",
            ownerPull
                ? OWNER_PULL_KERNEL_NAMES[kernel] : KERNEL_NAMES[kernel]);
        kernelInfos[kernel].kernelFunc = ownerPull
            ? reinterpret_cast<void *>(
                ops_hccl::CcuReduceScatterOwnerPullDualKernel)
            : reinterpret_cast<void *>(
                ops_hccl::CcuReduceScatterWriteReduceDualKernel);
        kernelInfos[kernel].setKernelArg(kernelArgs[kernel]);
        resource.stripeCounts[kernel] = KERNEL_MODES[kernel];
    }

    CcuInsHandle instance{0};
    uint32_t instanceCount = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &instance, &instanceCount));
    CHK_PRT_RET(instanceCount != 1,
        HCCL_ERROR(
            "[RegisterWriteReduce16Kernels] expected one CCU instance, got %u",
            instanceCount),
        HCCL_E_INTERNAL);
    CHK_RET_CCU(HcommCcuKernelRegisterStart(instance));
    for (uint32_t kernel = 0; kernel < kernelInfos.size(); ++kernel) {
        const void *args[] = {kernelInfos[kernel].kernelArg};
        CHK_RET_CCU(HcommCcuKernelRegister(instance, 0,
            kernelInfos[kernel].kernelFuncName, kernelInfos[kernel].kernelFunc,
            args, 1, &resource.ccuKernels[kernel]));
    }
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(instance));
    return HCCL_SUCCESS;
}

HcclResult RegisterWriteReduce12Kernels(HcclComm comm, const OpParam &param,
    const std::vector<LayerChannelGroup> &groups,
    const MateNodeLayout &layout, AlgResourceCtx &resource)
{
    CHK_PRT_RET(param.rankSize != 12U || groups.size() != 2U
            || layout.localSize + layout.remoteSize != param.rankSize
            || !((layout.localSize == 8U && layout.remoteSize == 4U)
                || (layout.localSize == 4U && layout.remoteSize == 8U)),
        HCCL_ERROR("[RegisterWriteReduce12Kernels] invalid rank-12 layout"),
        HCCL_E_INTERNAL);

    const LayerChannelGroup *intraGroup = nullptr;
    const LayerChannelGroup *crossGroup = nullptr;
    for (const auto &group : groups) {
        if (group.isIntra) {
            intraGroup = &group;
        } else {
            crossGroup = &group;
        }
    }
    const uint32_t localStripeCount = layout.localSize - 1U;
    CHK_PRT_RET(intraGroup == nullptr || crossGroup == nullptr
            || intraGroup->handles.size() != localStripeCount
            || crossGroup->handles.size() != layout.remoteSize,
        HCCL_ERROR(
            "[RegisterWriteReduce12Kernels] local/cross channel counts do not match layout"),
        HCCL_E_INTERNAL);

    constexpr uint64_t ALIGN_ELEMENTS = MATE_ALIGNMENT / sizeof(float);
    const uint64_t halfElements = param.count / 2U;
    const uint64_t phaseAElements
        = (halfElements / ALIGN_ELEMENTS) * ALIGN_ELEMENTS;
    const uint64_t phaseBElements = param.count - phaseAElements;
    CHK_PRT_RET(phaseAElements < WRITE_REDUCE_CROSS_STRIPES
            || phaseBElements < WRITE_REDUCE_CROSS_STRIPES,
        HCCL_ERROR(
            "[RegisterWriteReduce12Kernels] output is too small for two phases"),
        HCCL_E_INTERNAL);
    const uint64_t phaseABytes = phaseAElements * sizeof(float);
    const uint64_t phaseBBytes = phaseBElements * sizeof(float);

    std::array<uint64_t, ops_hccl::WRITE_REDUCE_DUAL_MAX_CHANNELS>
        localAOffsets{};
    std::array<uint64_t, ops_hccl::WRITE_REDUCE_DUAL_MAX_CHANNELS>
        localALengths{};
    std::array<uint64_t, ops_hccl::WRITE_REDUCE_DUAL_MAX_CHANNELS>
        crossAOffsets{};
    std::array<uint64_t, ops_hccl::WRITE_REDUCE_DUAL_MAX_CHANNELS>
        crossALengths{};
    std::array<uint64_t, ops_hccl::WRITE_REDUCE_DUAL_MAX_CHANNELS>
        localBOffsets{};
    std::array<uint64_t, ops_hccl::WRITE_REDUCE_DUAL_MAX_CHANNELS>
        localBLengths{};
    std::array<uint64_t, ops_hccl::WRITE_REDUCE_DUAL_MAX_CHANNELS>
        crossBOffsets{};
    std::array<uint64_t, ops_hccl::WRITE_REDUCE_DUAL_MAX_CHANNELS>
        crossBLengths{};
    CHK_PRT_RET(
        !BuildWriteReduceDualStripes(phaseAElements,
            localStripeCount, localAOffsets, localALengths)
            || !BuildWriteReduceDualStripes(phaseAElements,
                WRITE_REDUCE_CROSS_STRIPES,
                crossAOffsets, crossALengths)
            || !BuildWriteReduceDualStripes(phaseBElements,
                localStripeCount, localBOffsets, localBLengths)
            || !BuildWriteReduceDualStripes(phaseBElements,
                WRITE_REDUCE_CROSS_STRIPES,
                crossBOffsets, crossBLengths),
        HCCL_ERROR(
            "[RegisterWriteReduce12Kernels] invalid aligned stripe partition"),
        HCCL_E_INTERNAL);

    std::array<std::shared_ptr<ops_hccl::CcuKernelArgWriteReduceDual>, 4>
        kernelArgs = {
            std::make_shared<ops_hccl::CcuKernelArgWriteReduceDual>(),
            std::make_shared<ops_hccl::CcuKernelArgWriteReduceDual>(),
            std::make_shared<ops_hccl::CcuKernelArgWriteReduceDual>(),
            std::make_shared<ops_hccl::CcuKernelArgWriteReduceDual>(),
        };
    CHK_PRT_RET(
        !FillWriteReduceDualArg(param, layout, *intraGroup,
            localStripeCount, true, 0, phaseABytes,
            localAOffsets, localALengths, *kernelArgs[0])
            || !FillWriteReduceDualArg(param, layout, *crossGroup,
                WRITE_REDUCE_CROSS_STRIPES, true,
                phaseABytes, phaseBBytes,
                crossBOffsets, crossBLengths, *kernelArgs[1])
            || !FillWriteReduceDualArg(param, layout, *intraGroup,
                localStripeCount, false,
                phaseABytes, phaseBBytes,
                localBOffsets, localBLengths, *kernelArgs[2])
            || !FillWriteReduceDualArg(param, layout, *crossGroup,
                WRITE_REDUCE_CROSS_STRIPES, false,
                0, phaseABytes,
                crossAOffsets, crossALengths, *kernelArgs[3]),
        HCCL_ERROR(
            "[RegisterWriteReduce12Kernels] could not build phase arguments"),
        HCCL_E_INTERNAL);

    constexpr std::array<const char *, 4> KERNEL_NAMES = {
        "CcuReduceScatterWriteReduce12LocalA",
        "CcuReduceScatterWriteReduce12CrossB",
        "CcuReduceScatterWriteReduce12LocalB",
        "CcuReduceScatterWriteReduce12CrossA",
    };
    constexpr std::array<uint32_t, 4> KERNEL_MODES = {
        WRITE_REDUCE_12_LOCAL_A_MODE,
        WRITE_REDUCE_12_CROSS_B_MODE,
        WRITE_REDUCE_12_LOCAL_B_MODE,
        WRITE_REDUCE_12_CROSS_A_MODE,
    };
    std::array<CcuKernelInfo, 4> kernelInfos{};
    resource.ccuKernels.resize(kernelInfos.size());
    resource.stripeCounts.resize(kernelInfos.size());
    resource.combineKernel = 0;
    for (uint32_t kernel = 0; kernel < kernelInfos.size(); ++kernel) {
        std::snprintf(kernelInfos[kernel].kernelFuncName,
            sizeof(kernelInfos[kernel].kernelFuncName), "%s",
            KERNEL_NAMES[kernel]);
        kernelInfos[kernel].kernelFunc = reinterpret_cast<void *>(
            ops_hccl::CcuReduceScatterWriteReduceDualKernel);
        kernelInfos[kernel].setKernelArg(kernelArgs[kernel]);
        resource.stripeCounts[kernel] = KERNEL_MODES[kernel];
    }

    CcuInsHandle instance{0};
    uint32_t instanceCount = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &instance, &instanceCount));
    CHK_PRT_RET(instanceCount != 1,
        HCCL_ERROR(
            "[RegisterWriteReduce12Kernels] expected one CCU instance, got %u",
            instanceCount),
        HCCL_E_INTERNAL);
    CHK_RET_CCU(HcommCcuKernelRegisterStart(instance));
    for (uint32_t kernel = 0; kernel < kernelInfos.size(); ++kernel) {
        const void *args[] = {kernelInfos[kernel].kernelArg};
        CHK_RET_CCU(HcommCcuKernelRegister(instance, 0,
            kernelInfos[kernel].kernelFuncName,
            kernelInfos[kernel].kernelFunc, args, 1,
            &resource.ccuKernels[kernel]));
    }
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(instance));
    return HCCL_SUCCESS;
}

[[maybe_unused]] HcclResult RegisterHetero12Kernels(
    HcclComm comm, const OpParam &param,
    const std::vector<LayerChannelGroup> &groups, const MateNodeLayout &layout,
    AlgResourceCtx &resource)
{
    CHK_PRT_RET(param.rankSize != 12 || groups.size() != 2,
        HCCL_ERROR("[RegisterHetero12Kernels] expected rank-12 with two channel groups"),
        HCCL_E_INTERNAL);
    const LayerChannelGroup *intraGroup = nullptr;
    const LayerChannelGroup *crossGroup = nullptr;
    for (const auto &group : groups) {
        if (group.isIntra) {
            intraGroup = &group;
        } else {
            crossGroup = &group;
        }
    }
    CHK_PRT_RET(intraGroup == nullptr || crossGroup == nullptr,
        HCCL_ERROR("[RegisterHetero12Kernels] missing intra or inter group"),
        HCCL_E_INTERNAL);
    const bool isH8 = layout.localSize == 8U && layout.remoteSize == 4U;
    const bool isH4 = layout.localSize == 4U && layout.remoteSize == 8U;
    CHK_PRT_RET(!isH8 && !isH4,
        HCCL_ERROR("[RegisterHetero12Kernels] invalid asymmetric node layout"),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(intraGroup->handles.size() != layout.localSize - 1U
            || crossGroup->handles.size() != layout.remoteSize,
        HCCL_ERROR("[RegisterHetero12Kernels] channel counts do not match node layout"),
        HCCL_E_INTERNAL);

    Hetero12WorkspacePlan workspace;
    CHK_PRT_RET(!CalculateHetero12Workspace(param.count, workspace)
            || (isH8 ? workspace.h8Required : workspace.h4Required)
                > resource.localBuffer.size,
        HCCL_ERROR("[RegisterHetero12Kernels] workspace exceeds HCCL buffer"),
        HCCL_E_INTERNAL);

    std::array<ChannelHandle, MATE_MAX_PEERS> orderedCrossChannels{};
    for (uint32_t peerIndex = 0; peerIndex < layout.remoteSize; ++peerIndex) {
        const uint32_t peerRank = layout.remoteBegin + peerIndex;
        bool found = false;
        for (size_t channel = 0; channel < crossGroup->descs.size(); ++channel) {
            if (crossGroup->descs[channel].remoteRank == peerRank) {
                orderedCrossChannels[peerIndex] = crossGroup->handles[channel];
                found = true;
                break;
            }
        }
        CHK_PRT_RET(!found,
            HCCL_ERROR("[RegisterHetero12Kernels] ordered cross channel is missing"),
            HCCL_E_INTERNAL);
    }

    constexpr uint32_t H8_OWN_PIECES = 12U;
    constexpr uint32_t H8_HELPER_PIECES = 4U;
    constexpr uint32_t H4_HELPER_PIECES = 8U;
    const uint64_t outputBytes = workspace.outputBytes;
    std::vector<CcuKernelInfo> kernelInfos(4);
    resource.ccuKernels.resize(kernelInfos.size());
    resource.stripeCounts.resize(kernelInfos.size());
    resource.combineKernel = 0;

    if (isH8) {
        auto mateLocalArg = std::make_shared<ops_hccl::CcuKernelArgMateLocal>();
        mateLocalArg->dataType = param.dataType;
        mateLocalArg->reduceOp = param.reduceType;
        mateLocalArg->channelCount
            = static_cast<uint32_t>(intraGroup->handles.size());
        mateLocalArg->ownPieceCount = H8_OWN_PIECES;
        mateLocalArg->pieceCount = H8_OWN_PIECES + H8_HELPER_PIECES;
        for (size_t channel = 0; channel < intraGroup->handles.size(); ++channel) {
            mateLocalArg->channels[channel] = intraGroup->handles[channel];
        }
        for (uint32_t piece = 0; piece < H8_OWN_PIECES; ++piece) {
            const MateByteRange ownPiece
                = SplitMateRange(param.count, H8_OWN_PIECES, piece);
            mateLocalArg->inputOffsets[piece]
                = static_cast<uint64_t>(param.myRank) * outputBytes + ownPiece.offset;
            mateLocalArg->outputOffsets[piece] = ownPiece.offset;
            mateLocalArg->pieceLengths[piece] = ownPiece.length;
        }
        for (uint32_t target = 0; target < H8_HELPER_PIECES; ++target) {
            const uint32_t piece = H8_OWN_PIECES + target;
            mateLocalArg->inputOffsets[piece]
                = static_cast<uint64_t>(layout.remoteBegin + target) * outputBytes
                + static_cast<uint64_t>(layout.localIndex)
                    * workspace.helperShardBytes;
            mateLocalArg->outputOffsets[piece]
                = workspace.h8MailboxBase
                + static_cast<uint64_t>(target) * workspace.helperShardBytes;
            mateLocalArg->pieceLengths[piece] = workspace.helperShardBytes;
        }
        std::snprintf(kernelInfos[0].kernelFuncName,
            sizeof(kernelInfos[0].kernelFuncName),
            "CcuReduceScatterHetero12H8MateLocal");
        kernelInfos[0].kernelFunc = reinterpret_cast<void *>(
            ops_hccl::CcuReduceScatterMateLocalExpandedKernel);
        kernelInfos[0].setKernelArg(mateLocalArg);
        resource.stripeCounts[0] = HETERO12_H8_MATE_LOCAL_MODE;

        auto directCrossArg = std::make_shared<ops_hccl::CcuKernelArgDirect>();
        directCrossArg->dataType = param.dataType;
        directCrossArg->reduceOp = param.reduceType;
        directCrossArg->channelCount = layout.remoteSize;
        directCrossArg->inputOffset
            = static_cast<uint64_t>(param.myRank) * outputBytes;
        directCrossArg->outputOffset = 0;
        directCrossArg->length = outputBytes;
        directCrossArg->workspaceOffset = workspace.h8WideWorkspaceBase;
        directCrossArg->workspaceStride = workspace.fullWideStride;
        directCrossArg->tileLengths = workspace.fullTileLengths;
        for (uint32_t peer = 0; peer < layout.remoteSize; ++peer) {
            directCrossArg->channels[peer] = orderedCrossChannels[peer];
        }
        std::snprintf(kernelInfos[1].kernelFuncName,
            sizeof(kernelInfos[1].kernelFuncName),
            "CcuReduceScatterHetero12H8CrossScratch");
        kernelInfos[1].kernelFunc = reinterpret_cast<void *>(
            ops_hccl::CcuReduceScatterDirectCrossScratchKernel);
        kernelInfos[1].setKernelArg(directCrossArg);
        resource.stripeCounts[1] = HETERO12_H8_DIRECT_CROSS_MODE;

        auto combineArg = std::make_shared<ops_hccl::CcuKernelArgCombine>();
        combineArg->channelCount
            = static_cast<uint32_t>(intraGroup->handles.size());
        combineArg->dataType = param.dataType;
        combineArg->reduceOp = param.reduceType;
        combineArg->stripeCount = 1;
        combineArg->compactArgs = true;
        combineArg->staticStripeLength = outputBytes;
        for (size_t channel = 0; channel < intraGroup->handles.size(); ++channel) {
            combineArg->channels[channel] = intraGroup->handles[channel];
        }
        std::snprintf(kernelInfos[2].kernelFuncName,
            sizeof(kernelInfos[2].kernelFuncName),
            "CcuReduceScatterHetero12H8Combine");
        kernelInfos[2].kernelFunc = reinterpret_cast<void *>(
            ops_hccl::CcuReduceScatterPinnedCombineKernel);
        kernelInfos[2].setKernelArg(combineArg);
        resource.stripeCounts[2] = HETERO12_H8_COMBINE_MODE;

        auto publisherArg = std::make_shared<ops_hccl::CcuKernelArgPublisher>();
        publisherArg->channelCount = layout.remoteSize;
        for (uint32_t peer = 0; peer < layout.remoteSize; ++peer) {
            publisherArg->channels[peer] = orderedCrossChannels[peer];
        }
        std::snprintf(kernelInfos[3].kernelFuncName,
            sizeof(kernelInfos[3].kernelFuncName),
            "CcuReduceScatterHetero12H8Publisher");
        kernelInfos[3].kernelFunc = reinterpret_cast<void *>(
            ops_hccl::CcuReduceScatterPublisherKernel);
        kernelInfos[3].setKernelArg(publisherArg);
        resource.stripeCounts[3] = HETERO12_H8_PUBLISH_MODE;
    } else {
        auto localPrefixArg = std::make_shared<ops_hccl::CcuKernelArgDirect>();
        localPrefixArg->dataType = param.dataType;
        localPrefixArg->reduceOp = param.reduceType;
        localPrefixArg->channelCount
            = static_cast<uint32_t>(intraGroup->handles.size());
        localPrefixArg->inputOffset
            = static_cast<uint64_t>(param.myRank) * outputBytes;
        localPrefixArg->outputOffset = 0;
        localPrefixArg->length = workspace.prefixBytes;
        localPrefixArg->workspaceOffset = workspace.h4PrefixWorkspaceBase;
        localPrefixArg->workspaceStride = workspace.prefixWideStride;
        localPrefixArg->tileLengths = workspace.prefixTileLengths;
        for (size_t channel = 0; channel < intraGroup->handles.size(); ++channel) {
            localPrefixArg->channels[channel] = intraGroup->handles[channel];
        }
        std::snprintf(kernelInfos[0].kernelFuncName,
            sizeof(kernelInfos[0].kernelFuncName),
            "CcuReduceScatterHetero12H4LocalPrefix");
        kernelInfos[0].kernelFunc = reinterpret_cast<void *>(
            ops_hccl::CcuReduceScatterDirectLocalInitKernel);
        kernelInfos[0].setKernelArg(localPrefixArg);
        resource.stripeCounts[0] = HETERO12_H4_LOCAL_PREFIX_MODE;

        auto directCrossArg = std::make_shared<ops_hccl::CcuKernelArgDirect>();
        directCrossArg->dataType = param.dataType;
        directCrossArg->reduceOp = param.reduceType;
        directCrossArg->channelCount = layout.remoteSize;
        directCrossArg->inputOffset
            = static_cast<uint64_t>(param.myRank) * outputBytes
            + workspace.prefixBytes;
        directCrossArg->outputOffset = workspace.prefixBytes;
        directCrossArg->length = workspace.suffixBytes;
        directCrossArg->workspaceOffset = workspace.h4SuffixWorkspaceBase;
        directCrossArg->workspaceStride = workspace.suffixWideStride;
        directCrossArg->tileLengths = workspace.suffixTileLengths;
        for (uint32_t peer = 0; peer < layout.remoteSize; ++peer) {
            directCrossArg->channels[peer] = orderedCrossChannels[peer];
        }
        std::snprintf(kernelInfos[1].kernelFuncName,
            sizeof(kernelInfos[1].kernelFuncName),
            "CcuReduceScatterHetero12H4CrossSuffix");
        kernelInfos[1].kernelFunc = reinterpret_cast<void *>(
            ops_hccl::CcuReduceScatterDirectCrossKernel);
        kernelInfos[1].setKernelArg(directCrossArg);
        resource.stripeCounts[1] = HETERO12_H4_DIRECT_CROSS_MODE;

        auto localSuffixArg = std::make_shared<ops_hccl::CcuKernelArgDirect>();
        localSuffixArg->dataType = param.dataType;
        localSuffixArg->reduceOp = param.reduceType;
        localSuffixArg->channelCount
            = static_cast<uint32_t>(intraGroup->handles.size());
        localSuffixArg->inputOffset
            = static_cast<uint64_t>(param.myRank) * outputBytes
            + workspace.prefixBytes;
        localSuffixArg->outputOffset = workspace.prefixBytes;
        localSuffixArg->length = workspace.suffixBytes;
        localSuffixArg->workspaceOffset = workspace.h4PrefixWorkspaceBase;
        localSuffixArg->workspaceStride = workspace.suffixWideStride;
        localSuffixArg->tileLengths = workspace.suffixTileLengths;
        for (size_t channel = 0; channel < intraGroup->handles.size(); ++channel) {
            localSuffixArg->channels[channel] = intraGroup->handles[channel];
        }
        std::snprintf(kernelInfos[2].kernelFuncName,
            sizeof(kernelInfos[2].kernelFuncName),
            "CcuReduceScatterHetero12H4LocalSuffix");
        kernelInfos[2].kernelFunc = reinterpret_cast<void *>(
            ops_hccl::CcuReduceScatterDirectLocalKernel);
        kernelInfos[2].setKernelArg(localSuffixArg);
        resource.stripeCounts[2] = HETERO12_H4_LOCAL_SUFFIX_MODE;

        auto mateCrossArg = std::make_shared<ops_hccl::CcuKernelArgMateCross>();
        mateCrossArg->dataType = param.dataType;
        mateCrossArg->reduceOp = param.reduceType;
        mateCrossArg->channelCount = layout.remoteSize;
        mateCrossArg->remotePartialOffset
            = workspace.h8MailboxBase
            + static_cast<uint64_t>(layout.localIndex)
                * workspace.helperShardBytes;
        for (uint32_t helper = 0; helper < H4_HELPER_PIECES; ++helper) {
            mateCrossArg->channels[helper] = orderedCrossChannels[helper];
            mateCrossArg->outputOffsets[helper]
                = static_cast<uint64_t>(helper) * workspace.helperShardBytes;
            mateCrossArg->shardLengths[helper] = workspace.helperShardBytes;
        }
        std::snprintf(kernelInfos[3].kernelFuncName,
            sizeof(kernelInfos[3].kernelFuncName),
            "CcuReduceScatterHetero12H4MateCross");
        kernelInfos[3].kernelFunc = reinterpret_cast<void *>(
            ops_hccl::CcuReduceScatterMateCrossKernel);
        kernelInfos[3].setKernelArg(mateCrossArg);
        resource.stripeCounts[3] = HETERO12_H4_MATE_CROSS_MODE;
    }

    CcuInsHandle instance{0};
    uint32_t instanceCount = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &instance, &instanceCount));
    CHK_PRT_RET(instanceCount != 1,
        HCCL_ERROR("[RegisterHetero12Kernels] expected one CCU instance, got %u",
            instanceCount),
        HCCL_E_INTERNAL);
    CHK_RET_CCU(HcommCcuKernelRegisterStart(instance));
    for (size_t kernel = 0; kernel < kernelInfos.size(); ++kernel) {
        const void *kernelArgs[] = {kernelInfos[kernel].kernelArg};
        CHK_RET_CCU(HcommCcuKernelRegister(instance, 0,
            kernelInfos[kernel].kernelFuncName, kernelInfos[kernel].kernelFunc,
            kernelArgs, 1, &resource.ccuKernels[kernel]));
    }
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(instance));
    return HCCL_SUCCESS;
}

HcclResult RegisterHybridKernels(HcclComm comm, const OpParam &param,
    const std::vector<LayerChannelGroup> &groups, AlgResourceCtx &resource)
{
    CHK_PRT_RET(groups.size() != 2, HCCL_ERROR("[RegisterMateKernels] expected two channel groups"),
        HCCL_E_INTERNAL);
    const LayerChannelGroup *intraGroup = nullptr;
    const LayerChannelGroup *crossGroup = nullptr;
    for (const auto &group : groups) {
        if (group.isIntra) {
            intraGroup = &group;
        } else {
            crossGroup = &group;
        }
    }
    CHK_PRT_RET(intraGroup == nullptr || crossGroup == nullptr,
        HCCL_ERROR("[RegisterMateKernels] missing intra or inter group"), HCCL_E_INTERNAL);

    MateNodeLayout layout;
    CHK_PRT_RET(!BuildMateNodeLayout(param.rankSize, param.myRank, layout),
        HCCL_ERROR("[RegisterMateKernels] unsupported rank layout"), HCCL_E_INTERNAL);
    CHK_PRT_RET(intraGroup->handles.size() != layout.localSize - 1U
            || crossGroup->handles.size() != layout.remoteSize,
        HCCL_ERROR("[RegisterMateKernels] channel counts do not match node layout"), HCCL_E_INTERNAL);

    HybridWorkspacePlan workspace;
    CHK_PRT_RET(!CalculateHybridWorkspace(
                    param.count, param.rankSize, layout.localSize, layout.remoteSize, workspace)
            || workspace.required > resource.localBuffer.size,
        HCCL_ERROR("[RegisterMateKernels] MATE workspace exceeds HCCL buffer"), HCCL_E_INTERNAL);

    HybridWorkspacePlan remoteWorkspace;
    CHK_PRT_RET(!CalculateHybridWorkspace(
                    param.count, param.rankSize, layout.remoteSize, layout.localSize, remoteWorkspace),
        HCCL_ERROR("[RegisterMateKernels] remote MATE layout is invalid"), HCCL_E_INTERNAL);

    const uint64_t outputBytes = param.count * sizeof(float);

    std::array<ChannelHandle, MATE_MAX_PEERS> orderedCrossChannels{};
    for (uint32_t helper = 0; helper < layout.remoteSize; ++helper) {
        const uint32_t helperRank = layout.remoteBegin + helper;
        bool found = false;
        for (size_t channel = 0; channel < crossGroup->descs.size(); ++channel) {
            if (crossGroup->descs[channel].remoteRank == helperRank) {
                orderedCrossChannels[helper] = crossGroup->handles[channel];
                found = true;
                break;
            }
        }
        CHK_PRT_RET(!found, HCCL_ERROR("[RegisterMateKernels] remote helper channel is missing"),
            HCCL_E_INTERNAL);
    }

    std::vector<CcuKernelInfo> kernelInfos(4);
    resource.ccuKernels.resize(kernelInfos.size());
    resource.stripeCounts.resize(kernelInfos.size());
    resource.combineKernel = 0;

    const uint64_t mateElements = workspace.mateRange.length / sizeof(float);
    const MateByteRange localShard = SplitMateRange(
        mateElements, layout.localSize, layout.localIndex);
    auto mateLocalArg = std::make_shared<ops_hccl::CcuKernelArgMateLocal>();
    mateLocalArg->dataType = param.dataType;
    mateLocalArg->reduceOp = param.reduceType;
    mateLocalArg->channelCount = static_cast<uint32_t>(intraGroup->handles.size());
    mateLocalArg->ownPieceCount = layout.localSize;
    mateLocalArg->pieceCount = layout.localSize + layout.remoteSize;
    CHK_PRT_RET(mateLocalArg->pieceCount > ops_hccl::MATE_MAX_PIECES,
        HCCL_ERROR("[RegisterMateKernels] too many target-major pieces"), HCCL_E_INTERNAL);
    for (size_t channel = 0; channel < intraGroup->handles.size(); ++channel) {
        mateLocalArg->channels[channel] = intraGroup->handles[channel];
    }
    for (uint32_t piece = 0; piece < layout.localSize; ++piece) {
        const MateByteRange ownPiece
            = SplitMateRange(mateElements, layout.localSize, piece);
        mateLocalArg->inputOffsets[piece]
            = static_cast<uint64_t>(param.myRank) * outputBytes + ownPiece.offset;
        mateLocalArg->outputOffsets[piece] = ownPiece.offset;
        mateLocalArg->pieceLengths[piece] = ownPiece.length;
    }
    for (uint32_t target = 0; target < layout.remoteSize; ++target) {
        const uint32_t piece = layout.localSize + target;
        mateLocalArg->inputOffsets[piece]
            = static_cast<uint64_t>(layout.remoteBegin + target) * outputBytes
            + localShard.offset;
        mateLocalArg->outputOffsets[piece]
            = static_cast<uint64_t>(target) * workspace.mailboxStride;
        mateLocalArg->pieceLengths[piece] = localShard.length;
    }
    std::snprintf(kernelInfos[0].kernelFuncName, sizeof(kernelInfos[0].kernelFuncName),
        "CcuReduceScatterHybridMateLocal");
    kernelInfos[0].kernelFunc
        = reinterpret_cast<void *>(ops_hccl::CcuReduceScatterMateLocalKernel);
    kernelInfos[0].setKernelArg(mateLocalArg);
    resource.stripeCounts[0] = HYBRID_MATE_LOCAL_MODE;

    auto directCrossArg = std::make_shared<ops_hccl::CcuKernelArgDirect>();
    directCrossArg->dataType = param.dataType;
    directCrossArg->reduceOp = param.reduceType;
    directCrossArg->channelCount = layout.remoteSize;
    directCrossArg->inputOffset
        = static_cast<uint64_t>(param.myRank) * outputBytes + workspace.directRange.offset;
    directCrossArg->outputOffset = workspace.directRange.offset;
    directCrossArg->length = workspace.directRange.length;
    directCrossArg->workspaceOffset = workspace.wideWorkspaceBase;
    directCrossArg->workspaceStride = workspace.wideStride;
    directCrossArg->tileLengths = workspace.directTileLengths;
    for (uint32_t helper = 0; helper < layout.remoteSize; ++helper) {
        directCrossArg->channels[helper] = orderedCrossChannels[helper];
    }
    std::snprintf(kernelInfos[1].kernelFuncName, sizeof(kernelInfos[1].kernelFuncName),
        "CcuReduceScatterHybridDirectCrossWide");
    kernelInfos[1].kernelFunc
        = reinterpret_cast<void *>(ops_hccl::CcuReduceScatterDirectCrossKernel);
    kernelInfos[1].setKernelArg(directCrossArg);
    resource.stripeCounts[1] = HYBRID_DIRECT_CROSS_MODE;

    auto directLocalArg = std::make_shared<ops_hccl::CcuKernelArgDirect>();
    directLocalArg->dataType = param.dataType;
    directLocalArg->reduceOp = param.reduceType;
    directLocalArg->channelCount = static_cast<uint32_t>(intraGroup->handles.size());
    directLocalArg->inputOffset
        = static_cast<uint64_t>(param.myRank) * outputBytes + workspace.directRange.offset;
    directLocalArg->outputOffset = workspace.directRange.offset;
    directLocalArg->length = workspace.directRange.length;
    directLocalArg->workspaceOffset = workspace.wideWorkspaceBase;
    directLocalArg->workspaceStride = workspace.wideStride;
    directLocalArg->tileLengths = workspace.directTileLengths;
    for (size_t channel = 0; channel < intraGroup->handles.size(); ++channel) {
        directLocalArg->channels[channel] = intraGroup->handles[channel];
    }
    std::snprintf(kernelInfos[2].kernelFuncName, sizeof(kernelInfos[2].kernelFuncName),
        "CcuReduceScatterHybridDirectLocalWide");
    kernelInfos[2].kernelFunc
        = reinterpret_cast<void *>(ops_hccl::CcuReduceScatterDirectLocalKernel);
    kernelInfos[2].setKernelArg(directLocalArg);
    resource.stripeCounts[2] = HYBRID_DIRECT_LOCAL_MODE;

    auto mateCrossArg = std::make_shared<ops_hccl::CcuKernelArgMateCross>();
    mateCrossArg->dataType = param.dataType;
    mateCrossArg->reduceOp = param.reduceType;
    mateCrossArg->channelCount = layout.remoteSize;
    mateCrossArg->remotePartialOffset
        = static_cast<uint64_t>(layout.localIndex) * remoteWorkspace.mailboxStride;
    for (uint32_t helper = 0; helper < layout.remoteSize; ++helper) {
        const MateByteRange remoteShard = SplitMateRange(
            workspace.mateRange.length / sizeof(float), layout.remoteSize, helper);
        mateCrossArg->channels[helper] = orderedCrossChannels[helper];
        mateCrossArg->outputOffsets[helper] = remoteShard.offset;
        mateCrossArg->shardLengths[helper] = remoteShard.length;
    }
    std::snprintf(kernelInfos[3].kernelFuncName, sizeof(kernelInfos[3].kernelFuncName),
        "CcuReduceScatterHybridMateCross");
    kernelInfos[3].kernelFunc
        = reinterpret_cast<void *>(ops_hccl::CcuReduceScatterMateCrossKernel);
    kernelInfos[3].setKernelArg(mateCrossArg);
    resource.stripeCounts[3] = HYBRID_MATE_CROSS_MODE;

    CcuInsHandle instance{0};
    uint32_t instanceCount = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &instance, &instanceCount));
    CHK_PRT_RET(instanceCount != 1,
        HCCL_ERROR("[RegisterMateKernels] expected one CCU instance, got %u", instanceCount),
        HCCL_E_INTERNAL);
    CHK_RET_CCU(HcommCcuKernelRegisterStart(instance));
    for (size_t kernel = 0; kernel < kernelInfos.size(); ++kernel) {
        const void *kernelArgs[] = {kernelInfos[kernel].kernelArg};
        CHK_RET_CCU(HcommCcuKernelRegister(instance, 0, kernelInfos[kernel].kernelFuncName,
            kernelInfos[kernel].kernelFunc, kernelArgs, 1, &resource.ccuKernels[kernel]));
    }
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(instance));
    return HCCL_SUCCESS;
}

HcclResult RegisterHierarchyKernels(HcclComm comm, const OpParam &param,
    const std::vector<LayerChannelGroup> &groups, const HierarchyRankPlan &plan, AlgResourceCtx &resource)
{
    CHK_PRT_RET(groups.size() != 2 || !groups[0].isIntra || groups[1].isIntra,
        HCCL_ERROR("[RegisterHierarchyKernels] invalid layer ordering"), HCCL_E_INTERNAL);

    std::vector<CcuKernelInfo> kernelInfos(2);
    auto localArg = std::make_shared<ops_hccl::CcuKernelArgNodeLocal>();
    localArg->dataType = param.dataType;
    localArg->reduceOp = param.reduceType;
    localArg->targetCount = plan.targetCount;
    localArg->channelCount = static_cast<uint32_t>(groups[0].handles.size());
    for (size_t i = 0; i < groups[0].handles.size(); ++i) {
        localArg->channels[i] = groups[0].handles[i];
    }
    std::snprintf(kernelInfos[0].kernelFuncName, sizeof(kernelInfos[0].kernelFuncName),
        "CcuReduceScatterNodeLocal");
    kernelInfos[0].kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuReduceScatterNodeLocalKernel);
    kernelInfos[0].setKernelArg(localArg);

    auto crossArg = std::make_shared<ops_hccl::CcuKernelArgCrossFinalize>();
    crossArg->dataType = param.dataType;
    crossArg->reduceOp = param.reduceType;
    crossArg->remoteSlot = plan.remoteSlot;
    crossArg->channelCount = static_cast<uint32_t>(groups[1].handles.size());
    crossArg->ownerChannelIndex0 = crossArg->channelCount;
    crossArg->ownerChannelIndex1 = crossArg->channelCount;
    for (size_t i = 0; i < groups[1].handles.size(); ++i) {
        crossArg->channels[i] = groups[1].handles[i];
        if (groups[1].descs[i].remoteRank == plan.ownerPeers[0]) {
            crossArg->ownerChannelIndex0 = static_cast<uint32_t>(i);
        }
        if (groups[1].descs[i].remoteRank == plan.ownerPeers[1]) {
            crossArg->ownerChannelIndex1 = static_cast<uint32_t>(i);
        }
    }
    CHK_PRT_RET(crossArg->ownerChannelIndex0 >= crossArg->channelCount
            || crossArg->ownerChannelIndex1 >= crossArg->channelCount,
        HCCL_ERROR("[RegisterHierarchyKernels] owner channel is missing"), HCCL_E_INTERNAL);
    std::snprintf(kernelInfos[1].kernelFuncName, sizeof(kernelInfos[1].kernelFuncName),
        "CcuReduceScatterCrossFinalize");
    kernelInfos[1].kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuReduceScatterCrossFinalizeKernel);
    kernelInfos[1].setKernelArg(crossArg);

    CcuInsHandle instance{0};
    uint32_t instanceCount = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &instance, &instanceCount));
    CHK_PRT_RET(instanceCount != 1,
        HCCL_ERROR("[RegisterHierarchyKernels] expected one CCU instance, got %u", instanceCount), HCCL_E_INTERNAL);

    resource.ccuKernels.resize(2);
    resource.stripeCounts.assign(2, HIERARCHICAL_KERNEL_MODE);
    resource.combineKernel = 0;
    CHK_RET_CCU(HcommCcuKernelRegisterStart(instance));
    for (size_t i = 0; i < kernelInfos.size(); ++i) {
        const void *kernelArgs[] = {kernelInfos[i].kernelArg};
        CHK_RET_CCU(HcommCcuKernelRegister(instance, 0, kernelInfos[i].kernelFuncName, kernelInfos[i].kernelFunc,
            kernelArgs, 1, &resource.ccuKernels[i]));
    }
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(instance));
    return HCCL_SUCCESS;
}

HcclResult RegisterKernels(
    HcclComm comm, const OpParam &param, const std::vector<LayerChannelGroup> &groups,
    const HierarchyRankPlan *hierarchyPlan, const MateNodeLayout *mateLayout, AlgResourceCtx &resource)
{
    if (groups.empty()) {
        return HCCL_SUCCESS;
    }
    if (hierarchyPlan != nullptr) {
        return RegisterHierarchyKernels(comm, param, groups, *hierarchyPlan, resource);
    }
    if (mateLayout != nullptr) {
        if (param.rankSize == 12) {
            return RegisterWriteReduce12Kernels(
                comm, param, groups, *mateLayout, resource);
        }
        if (param.rankSize == 16) {
            return RegisterWriteReduce16Kernels(
                comm, param, groups, *mateLayout,
                IsOfficialRank16OwnerPullCount(param.count), resource);
        }
        return RegisterHybridKernels(comm, param, groups, resource);
    }

    size_t includeSelfGroup = 0;
    if (groups.size() == 2 && groups[1].handles.size() < groups[0].handles.size()) {
        includeSelfGroup = 1;
    }
    std::vector<CcuKernelInfo> kernelInfos(groups.size());
    resource.stripeCounts.resize(groups.size());
    const uint64_t outputBytes = param.count * sizeof(float);
    const bool smallOutput = outputBytes <= SMALL_OUTPUT_THRESHOLD;
    const bool dualLarge = groups.size() == 2 && !smallOutput;
    const bool writeReduce4x1 = param.rankSize == 4U && groups.size() == 1U
        && groups[0].handles.size() == ops_hccl::WRITE_REDUCE_4X1_STRIPES
        && !smallOutput;
    std::array<uint64_t, ops_hccl::WRITE_REDUCE_4X1_STRIPES>
        writeReduceOffsets{};
    std::array<uint64_t, ops_hccl::WRITE_REDUCE_4X1_STRIPES>
        writeReduceLengths{};
    CHK_PRT_RET(writeReduce4x1
            && !BuildWriteReduce4x1Stripes(
                param.count, writeReduceOffsets, writeReduceLengths),
        HCCL_ERROR("[RegisterKernels] invalid 4x1 WriteReduce stripe partition"),
        HCCL_E_INTERNAL);
    for (size_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
        auto &info = kernelInfos[groupIndex];
        if (writeReduce4x1) {
            auto writeReduceArg
                = std::make_shared<ops_hccl::CcuKernelArgWriteReduce4x1>();
            writeReduceArg->dataType = param.dataType;
            writeReduceArg->reduceOp = param.reduceType;
            writeReduceArg->myRank = param.myRank;
            writeReduceArg->outputBytes = outputBytes;
            writeReduceArg->channelCount
                = static_cast<uint32_t>(groups[groupIndex].handles.size());
            writeReduceArg->stripeOffsets = writeReduceOffsets;
            writeReduceArg->stripeLengths = writeReduceLengths;
            std::array<bool, 4> seenTarget{};
            for (uint32_t channel = 0;
                channel < writeReduceArg->channelCount; ++channel) {
                const uint32_t target
                    = groups[groupIndex].descs[channel].remoteRank;
                CHK_PRT_RET(target >= param.rankSize || target == param.myRank
                        || seenTarget[target],
                    HCCL_ERROR(
                        "[RegisterKernels] invalid 4x1 WriteReduce target %u",
                        target),
                    HCCL_E_INTERNAL);
                seenTarget[target] = true;
                writeReduceArg->channels[channel]
                    = groups[groupIndex].handles[channel];
                writeReduceArg->targetRanks[channel] = target;
                writeReduceArg->sourceIndicesAtTarget[channel]
                    = param.myRank < target ? param.myRank : param.myRank - 1U;
            }
            std::snprintf(info.kernelFuncName,
                sizeof(info.kernelFuncName),
                "CcuReduceScatterWriteReduce4x1");
            info.kernelFunc = reinterpret_cast<void *>(
                ops_hccl::CcuReduceScatterWriteReduce4x1Kernel);
            info.setKernelArg(writeReduceArg);
            resource.stripeCounts[groupIndex] = WRITE_REDUCE_4X1_MODE;
            continue;
        }
        int nameLen = std::snprintf(
            info.kernelFuncName, sizeof(info.kernelFuncName), "CcuReduceScatterPartialL%u", groups[groupIndex].layer);
        CHK_PRT_RET(nameLen <= 0 || static_cast<size_t>(nameLen) >= sizeof(info.kernelFuncName),
            HCCL_ERROR("[RegisterKernels] failed to build kernel name"), HCCL_E_INTERNAL);
        info.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuReduceScatterPartialKernel);

        auto kernelArg = std::make_shared<ops_hccl::CcuKernelArgPartialReduce>();
        kernelArg->dataType = param.dataType;
        kernelArg->outputDataType = param.dataType;
        kernelArg->reduceOp = param.reduceType;
        kernelArg->includeSelf = groupIndex == includeSelfGroup;
        kernelArg->channelCount = static_cast<uint32_t>(groups[groupIndex].handles.size());
        const uint64_t rankSquare = static_cast<uint64_t>(param.rankSize) * param.rankSize;
        const uint64_t msPrefixCap = (MS_CHECKER_BUCKET_CAP / rankSquare) * MS_SLICE_BYTES;
        // Small messages fit completely in the checker-safe MS budget.  This
        // removes HBM lane workspaces while preserving the proven one-kernel
        // (4x1) or concurrent two-partial workflow.
        kernelArg->enableMsPrefix = smallOutput && msPrefixCap != 0;
        // Large dual-die cases use all sources in one read wave.  The runtime
        // mode also carries sourceCount so it can reserve exactly two banks
        // rather than pessimistically allocating seven lanes for every group.
        kernelArg->enableWideLane = dualLarge;
        kernelArg->enableRank4PackedMs = param.rankSize == 4U
            && groups.size() == 1U
            && groups[0].handles.size() == 3U
            && groupIndex == includeSelfGroup
            && outputBytes == RANK4_PACKED_MS_OUTPUT_BYTES;
        kernelArg->compactArgs = kernelArg->enableMsPrefix;
        kernelArg->stripeCount = smallOutput || kernelArg->enableWideLane
            ? 0
            : static_cast<uint32_t>(
                std::min<uint64_t>(std::min(MAX_ASYNC_STRIPES, kernelArg->channelCount), param.count));
        const uint32_t sourceCount
            = kernelArg->channelCount + static_cast<uint32_t>(kernelArg->includeSelf);
        if (kernelArg->compactArgs) {
            kernelArg->staticSliceOffset = outputBytes * param.myRank;
            kernelArg->staticMsGoSize = BuildCompactMsGoSize(outputBytes);
            resource.stripeCounts[groupIndex] = kernelArg->includeSelf
                ? COMPACT_MS_OUTPUT_MODE
                : COMPACT_MS_SCRATCH_MODE;
        } else {
            resource.stripeCounts[groupIndex] = kernelArg->enableWideLane
                ? WIDE_LANE_KERNEL_MODE_BASE + sourceCount
                : (kernelArg->enableMsPrefix ? HYBRID_MS_KERNEL_MODE : kernelArg->stripeCount);
        }
        for (size_t channelIndex = 0; channelIndex < groups[groupIndex].handles.size(); ++channelIndex) {
            kernelArg->channels[channelIndex] = groups[groupIndex].handles[channelIndex];
        }
        CHK_PRT_RET(sourceCount > MAX_GROUP_SOURCE_COUNT,
            HCCL_ERROR("[RegisterKernels] layer %u has too many reduce sources", groups[groupIndex].layer),
            HCCL_E_NOT_SUPPORT);
        info.setKernelArg(kernelArg);
    }

    CcuInsHandle instance{0};
    uint32_t instanceCount = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &instance, &instanceCount));
    CHK_PRT_RET(instanceCount != 1, HCCL_ERROR("[RegisterKernels] expected one CCU instance, got %u", instanceCount),
        HCCL_E_INTERNAL);

    resource.ccuKernels.resize(groups.size());
    CHK_RET_CCU(HcommCcuKernelRegisterStart(instance));
    for (size_t i = 0; i < kernelInfos.size(); ++i) {
        const void *kernelArgs[] = {kernelInfos[i].kernelArg};
        // dieId is reserved. The translator selects the IO die from the channels.
        CHK_RET_CCU(HcommCcuKernelRegister(instance, 0, kernelInfos[i].kernelFuncName, kernelInfos[i].kernelFunc,
            kernelArgs, 1, &resource.ccuKernels[i]));
    }
    std::shared_ptr<ops_hccl::CcuKernelArgCombine> combineArg;
    if (groups.size() == 2) {
        combineArg = std::make_shared<ops_hccl::CcuKernelArgCombine>();
        combineArg->channelCount = 0;
        combineArg->dataType = param.dataType;
        combineArg->reduceOp = param.reduceType;
        combineArg->stripeCount = outputBytes <= SMALL_OUTPUT_THRESHOLD
            ? 1
            : static_cast<uint32_t>(std::min<uint64_t>(MAX_COMBINE_STRIPES, param.count));
        combineArg->compactArgs = smallOutput;
        combineArg->staticStripeLength = outputBytes;
        const void *combineArgs[] = {combineArg.get()};
        CHK_RET_CCU(HcommCcuKernelRegister(instance, 0, "CcuReduceScatterCombine",
            reinterpret_cast<void *>(ops_hccl::CcuReduceScatterCombineKernel), combineArgs, 1,
            &resource.combineKernel));
    }
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(instance));
    return HCCL_SUCCESS;
}

template <size_t N>
bool LoadCachedCommId(const void *ctx, uint64_t ctxSize, char (&commId)[N])
{
    constexpr uint64_t CACHE_BYTES = sizeof(uint64_t) + N;
    if (ctx == nullptr || ctxSize < CACHE_BYTES || ctxSize > std::numeric_limits<size_t>::max()) {
        return false;
    }
    const char *cache = static_cast<const char *>(ctx) + static_cast<size_t>(ctxSize - CACHE_BYTES);
    uint64_t magic = 0;
    std::memcpy(&magic, cache, sizeof(magic));
    if (magic != COMM_ID_CACHE_MAGIC) {
        return false;
    }
    std::memcpy(commId, cache + sizeof(magic), N);
    return commId[0] != '\0' && std::memchr(commId, '\0', N) != nullptr;
}

} // namespace

HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    CHK_PRT_RET(
        dataType != HCCL_DATA_TYPE_FP32, HCCL_ERROR("[HcclReduceScatter] only FP32 is supported"), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(op != HCCL_REDUCE_SUM, HCCL_ERROR("[HcclReduceScatter] only SUM is supported"), HCCL_E_NOT_SUPPORT);

    // 构造算子参数
    OpParam param;
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;

    // ==============================================
    // STEP 1: 解析拓扑信息
    // ==============================================
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || param.myRank >= param.rankSize,
        HCCL_ERROR("[HcclReduceScatter] invalid rank %u/%u", param.myRank, param.rankSize), HCCL_E_PARA);
    int tagLen = 0;
    if (param.rankSize == 4U
        && recvCount == RANK4_PACKED_MS_OUTPUT_BYTES / sizeof(float)) {
        tagLen = std::snprintf(param.tag, sizeof(param.tag),
            "hccl_custom_reducescatter_v12v_rank4_packed_ms_cid_%llu",
            static_cast<unsigned long long>(recvCount));
    } else if (param.rankSize == 12) {
        tagLen = std::snprintf(param.tag, sizeof(param.tag),
            "hccl_custom_reducescatter_v12t_rank12_alt_cid_%llu",
            static_cast<unsigned long long>(recvCount));
    } else if (param.rankSize == 16
        && IsOfficialRank16OwnerPullCount(recvCount)) {
        tagLen = std::snprintf(param.tag, sizeof(param.tag),
            "hccl_custom_reducescatter_v12bs_rank16_owner_pull_p1_cid_%llu",
            static_cast<unsigned long long>(recvCount));
    } else if (param.rankSize == 16) {
        tagLen = std::snprintf(param.tag, sizeof(param.tag),
            "hccl_custom_reducescatter_v12t_rank16_alt_cid_%llu",
            static_cast<unsigned long long>(recvCount));
    } else {
        tagLen = std::snprintf(param.tag, sizeof(param.tag),
            "hccl_custom_reducescatter_v12t_rank4_epoch3_cid_%llu",
            static_cast<unsigned long long>(recvCount));
    }
    CHK_PRT_RET(tagLen <= 0 || static_cast<size_t>(tagLen) >= sizeof(param.tag),
        HCCL_ERROR("[HcclReduceScatter] failed to build resource tag"), HCCL_E_INTERNAL);

    CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;
    void *ctx = nullptr;
    uint64_t size = 0;
    const bool contextExists
        = recvCount != 0 && HcclEngineCtxGet(comm, param.tag, ccuEngine, &ctx, &size) == HCCL_SUCCESS;

    // DFX registration remains per invocation. Only the immutable
    // communicator-id lookup is served from this communicator-owned context;
    // an old, short, or malformed context safely falls back to the public
    // communicator-name query.
    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
    if (!contextExists || !LoadCachedCommId(ctx, size, commName)) {
        CHK_RET(HcclGetCommName(comm, commName));
    }
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));
    if (recvCount == 0) {
        return HCCL_SUCCESS;
    }

    constexpr uint64_t DATA_TYPE_SIZE = sizeof(float);
    CHK_PRT_RET(recvCount > std::numeric_limits<uint64_t>::max() / DATA_TYPE_SIZE,
        HCCL_ERROR("[HcclReduceScatter] output byte size overflows"), HCCL_E_PARA);
    const uint64_t outputBytes = recvCount * DATA_TYPE_SIZE;
    CHK_PRT_RET(outputBytes > std::numeric_limits<uint64_t>::max() / param.rankSize,
        HCCL_ERROR("[HcclReduceScatter] input byte size overflows"), HCCL_E_PARA);
    const uint64_t inputBytes = outputBytes * param.rankSize;
    // V5 full hierarchy concentrated every destination on one cross channel
    // (about 10.7 GB/s in all four measured large cases).  V6 deliberately
    // keeps both direct IO-die groups active and widens source concurrency.
    const HierarchyRankPlan *hierarchyPlan = nullptr;

    // ==============================================
    // STEP 2: 创建资源
    // ==============================================
    // ==============================================
    // STEP 2.1: 申请用于 Host/Device 同步的通信资源
    // ==============================================
    // 将用户传入的 stream 转换为 CCU 通信引擎中的 thread，并申请 1 个 notify
    CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, stream, 1, &param.cpuThread));

    if (contextExists) {
        // CCU 资源已经存在，复用资源
        HCCL_INFO("Engine context already exists");
        param.resCtx = ctx;
        param.ctxSize = size;
    } else {
        // Device 资源不存在，资源构建
        AlgResourceCtx resCtxHost;
        std::copy_n(commName, COMM_INDENTIFIER_MAX_LENGTH, resCtxHost.cachedCommId.begin());

        // 从通信域获取 HCCL Buffer（Device上的内存，默认总大小400MB）
        void *cclBufferAddr;
        uint64_t cclBufferSize;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};
        CHK_RET_CCU(HcommCcuGetMemToken(
            reinterpret_cast<uint64_t>(cclBufferAddr), cclBufferSize, &resCtxHost.localBufferToken));
        resCtxHost.cachedInputAddress = reinterpret_cast<uint64_t>(sendBuf);
        resCtxHost.cachedInputSize = inputBytes;
        CHK_RET_CCU(HcommCcuGetMemToken(
            resCtxHost.cachedInputAddress, resCtxHost.cachedInputSize, &resCtxHost.cachedInputToken));
        resCtxHost.cachedOutputAddress = reinterpret_cast<uint64_t>(recvBuf);
        resCtxHost.cachedOutputSize = outputBytes;
        CHK_RET_CCU(HcommCcuGetMemToken(
            resCtxHost.cachedOutputAddress, resCtxHost.cachedOutputSize, &resCtxHost.cachedOutputToken));

        // ==============================================
        // STEP 2.2: 每个 peer 只申请一个 Channel，再按网络层分给不同 CCU/IO die
        // ==============================================
        MateNodeLayout mateLayout;
        const MateNodeLayout *mateLayoutPtr = nullptr;
        if (ShouldUseMate(param, cclBufferSize)) {
            CHK_PRT_RET(!BuildMateNodeLayout(param.rankSize, param.myRank, mateLayout),
                HCCL_ERROR("[HcclReduceScatter] could not build MATE rank layout"), HCCL_E_INTERNAL);
            mateLayoutPtr = &mateLayout;
        }
        std::vector<LayerChannelGroup> layerGroups;
        if (param.rankSize > 1) {
            CHK_RET(BuildLayerGroups(comm, param, hierarchyPlan, mateLayoutPtr, layerGroups));
        }
        CHK_RET(AcquireThreads(comm, param, layerGroups.size(), resCtxHost));
        CHK_RET(RegisterKernels(comm, param, layerGroups, hierarchyPlan, mateLayoutPtr, resCtxHost));

        // ==============================================
        // STEP 2.3: 申请通信引擎上下文
        // ==============================================
        // 申请 CCU 通信引擎上下文，存放 AlgResourceCtx 信息
        std::vector<char> seq = resCtxHost.Serialize();
        uint64_t seqSize = seq.size();
        param.ctxSize = seqSize;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, ccuEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, ccuEngine, param.tag, seq.data(), seqSize, 0));
    }

    // ==============================================
    // STEP 3: 下发 CCU Kernel
    // ==============================================
    CHK_RET(ops_hccl::ExecOp(param));
    return HCCL_SUCCESS;
}
