/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CUSTOM_H
#define OPS_HCCL_CUSTOM_H

#include <cstring>
#include <type_traits>
#include <vector>
#include <hccl/hccl_rank_graph.h>

#include "common.h"

struct AlgResourceCtx;

namespace ops_hccl {
constexpr uint64_t DIRECT_CTX_MAGIC = 0x5343435544495231ULL;
constexpr uint32_t DIRECT_CTX_VERSION = 51;
// V20: queue 2x8 helper Pull immediately after preparation on the root die.
// Compress root8 zero offers; retain baseline Small, ratios and DONE/RELEASE.
// Select by cached topology and runtime root, never by test number or call order.
constexpr bool V13_LARGE_COPY_FOUR_BY_ONE = true;
constexpr bool V13_LARGE_COPY_OTHERS = true;
constexpr bool V13_SMALL_ROOT_SPECIALIZATION = true;
constexpr bool V13_BATCH_OFFERS = true;
constexpr bool V13_ROOT4_PREFIXES = true;
constexpr uint64_t ELEMENT_BYTES = sizeof(float);
constexpr uint32_t DIRECT_MAX_DIE_GROUPS = 2;
// Retain the existing per-rank size boundary; no new performance-optimal
// threshold is implied by the Pull implementation.
constexpr uint64_t DIRECT_SMALL_MAX_BYTES = 1024ULL * 1024;
static_assert(DIRECT_SMALL_MAX_BYTES <= MAX_DATA_SIZE, "Small Direct must fit in one transfer");
static_assert(ELEMENT_BYTES == 4, "This Scatter supports FP32 elements");
static_assert(MAX_RANK_SIZE < 32, "Topology masks use uint32_t");
static_assert(MAX_RANK_SIZE <= 16, "Root gaps and suffix use at most 16 event bits");
static_assert(MAX_DATA_SIZE > 0 && MAX_DATA_SIZE % ELEMENT_BYTES == 0,
    "Direct transfers must contain whole FP32 elements");

enum class MessageClass : uint32_t { EMPTY, SMALL, LARGE };

inline MessageClass ClassifyMessage(uint64_t count)
{
    if (count == 0) {
        return MessageClass::EMPTY;
    }
    return count <= DIRECT_SMALL_MAX_BYTES / ELEMENT_BYTES ? MessageClass::SMALL : MessageClass::LARGE;
}

inline bool UseSmallDirect(const OpParam &param)
{
    // A collective-wide condition: no per-rank buffer capacity or die count.
    return ClassifyMessage(param.count) == MessageClass::SMALL;
}

enum class TopologyKind : uint32_t { UNKNOWN, TWO_BY_EIGHT, FOUR_BY_ONE, EIGHT_PLUS_FOUR, FOUR_BY_THREE };
enum class TopologyEvidence : uint32_t { NONE, LAYER_INSTANCES, LOCAL_MESH_INSTANCE, SINGLETON_CASE };
enum class TopologyReason : uint32_t {
    NOT_QUERIED, RECOGNIZED, UNSUPPORTED_RANK_SIZE, QUERY_FAILED, INVALID_METADATA,
    NO_SERVER_MESH, AMBIGUOUS, CAPACITY_LIMIT, QUERY_EXCEPTION
};
enum class PathClass : uint32_t { UNKNOWN, LOCAL_MESH, REMOTE_CLOS };

// Storage bounds, not hardware limits. Larger descriptions remain UNKNOWN and
// never change the collective's wire protocol or canonical channel selection.
constexpr uint32_t MAX_TOPOLOGY_LAYERS = 8;
constexpr uint32_t MAX_TOPOLOGY_SUBINSTANCES = 64;
constexpr uint32_t TOPO_TYPE_VALID = 1;
constexpr uint32_t TOPO_MEMBERS_VALID = 2;
constexpr uint32_t TOPO_SIZES_VALID = 4;

inline uint32_t RankBit(uint32_t rank)
{
    return rank < MAX_RANK_SIZE ? (uint32_t{1} << rank) : 0;
}

inline uint32_t AllRankMask(uint32_t rankSize)
{
    return rankSize > 0 && rankSize <= MAX_RANK_SIZE ? (uint32_t{1} << rankSize) - 1 : 0;
}

inline uint32_t RankMaskSize(uint32_t mask)
{
    uint32_t size = 0;
    while (mask != 0) {
        mask &= mask - 1;
        ++size;
    }
    return size;
}

inline TopologyKind ShapeFromLocalSize(uint32_t rankSize, uint32_t localSize)
{
    if (rankSize == 16 && localSize == 8) {
        return TopologyKind::TWO_BY_EIGHT;
    }
    if (rankSize == 4 && localSize == 1) {
        return TopologyKind::FOUR_BY_ONE;
    }
    if (rankSize == 12 && (localSize == 4 || localSize == 8)) {
        return TopologyKind::EIGHT_PLUS_FOUR;
    }
    if (rankSize == 12 && localSize == 3) {
        return TopologyKind::FOUR_BY_THREE;
    }
    return TopologyKind::UNKNOWN;
}

// The caller supplies a sorted copy; raw provider order is retained separately.
inline TopologyKind ShapeFromSizes(uint32_t rankSize, const uint32_t *sizes, uint32_t sizeCount)
{
    if (rankSize == 16 && sizeCount == 2 && sizes[0] == 8 && sizes[1] == 8) {
        return TopologyKind::TWO_BY_EIGHT;
    }
    if (rankSize == 4 && sizeCount == 4 && sizes[0] == 1 && sizes[1] == 1 &&
        sizes[2] == 1 && sizes[3] == 1) {
        return TopologyKind::FOUR_BY_ONE;
    }
    if (rankSize == 12 && sizeCount == 2 && sizes[0] == 4 && sizes[1] == 8) {
        return TopologyKind::EIGHT_PLUS_FOUR;
    }
    if (rankSize == 12 && sizeCount == 4 && sizes[0] == 3 && sizes[1] == 3 &&
        sizes[2] == 3 && sizes[3] == 3) {
        return TopologyKind::FOUR_BY_THREE;
    }
    return TopologyKind::UNKNOWN;
}

struct TopologyLayerFacts {
    uint32_t layer = INVALID_VALUE_RANKID;
    CommTopo type = COMM_TOPO_RESERVED;
    uint32_t valid = 0;
    int32_t typeResult = HCCL_E_NOT_SUPPORT;
    int32_t membersResult = HCCL_E_NOT_SUPPORT;
    int32_t sizesResult = HCCL_E_NOT_SUPPORT;
    int32_t detailResult = HCCL_E_NOT_SUPPORT;
    uint32_t memberMask = 0;
    uint32_t instanceCount = 0;
    uint32_t instanceSizes[MAX_RANK_SIZE]{}; // Actual provider order, not inferred sizes.
    uint32_t meshMemberMask = 0;
    uint32_t closMemberMask = 0;
    TopologyReason issue = TopologyReason::NOT_QUERIED;

    void Reset()
    {
        layer = INVALID_VALUE_RANKID;
        type = COMM_TOPO_RESERVED;
        valid = 0;
        typeResult = HCCL_E_NOT_SUPPORT;
        membersResult = HCCL_E_NOT_SUPPORT;
        sizesResult = HCCL_E_NOT_SUPPORT;
        detailResult = HCCL_E_NOT_SUPPORT;
        memberMask = 0;
        instanceCount = 0;
        for (auto &size : instanceSizes) {
            size = 0;
        }
        meshMemberMask = 0;
        closMemberMask = 0;
        issue = TopologyReason::NOT_QUERIED;
    }
};

struct TopologyFacts {
    TopologyKind kind = TopologyKind::UNKNOWN;
    TopologyEvidence evidence = TopologyEvidence::NONE;
    TopologyReason reason = TopologyReason::NOT_QUERIED;
    uint32_t localMemberMask = 0;
    uint32_t meshLayer = INVALID_VALUE_RANKID;
    uint32_t closLayer = INVALID_VALUE_RANKID;
    uint32_t serverCount = 0;
    uint32_t serverSizes[MAX_RANK_SIZE]{}; // Populated only from a matching global size query.
    uint32_t layerCount = 0;
    TopologyLayerFacts layers[MAX_TOPOLOGY_LAYERS]{};

    // Platform GCC 13.3 reported a gimple_add_tmp_var ICE at a TopologyFacts{}
    // assignment. Reset in place without a nested aggregate temporary,
    // preserving the declared defaults, including all nonzero sentinels.
    void Reset(TopologyReason resetReason = TopologyReason::NOT_QUERIED)
    {
        kind = TopologyKind::UNKNOWN;
        evidence = TopologyEvidence::NONE;
        reason = resetReason;
        localMemberMask = 0;
        meshLayer = INVALID_VALUE_RANKID;
        closLayer = INVALID_VALUE_RANKID;
        serverCount = 0;
        for (auto &size : serverSizes) {
            size = 0;
        }
        layerCount = 0;
        for (auto &entry : layers) {
            entry.Reset();
        }
    }

    bool Valid(uint32_t rankSize, uint32_t myRank) const
    {
        const uint32_t all = AllRankMask(rankSize);
        if (all == 0 || myRank >= rankSize || layerCount > MAX_TOPOLOGY_LAYERS ||
            serverCount > MAX_RANK_SIZE || static_cast<uint32_t>(kind) > 4 ||
            static_cast<uint32_t>(evidence) > 3 || static_cast<uint32_t>(reason) > 8 ||
            (localMemberMask & ~all) != 0) {
            return false;
        }
        bool foundMeshLayer = meshLayer == INVALID_VALUE_RANKID;
        bool foundClosLayer = closLayer == INVALID_VALUE_RANKID;
        bool foundMeshMembers = false;
        for (uint32_t i = 0; i < layerCount; ++i) {
            const auto &entry = layers[i];
            if (entry.instanceCount > MAX_RANK_SIZE || (entry.valid & ~uint32_t{7}) != 0 ||
                static_cast<uint32_t>(entry.issue) > 8 ||
                ((entry.memberMask | entry.meshMemberMask | entry.closMemberMask) & ~all) != 0) {
                return false;
            }
            if (i != 0 && layers[i - 1].layer >= entry.layer) {
                return false;
            }
            if ((entry.valid & TOPO_MEMBERS_VALID) != 0 && (entry.memberMask & RankBit(myRank)) == 0) {
                return false;
            }
            if ((entry.valid & TOPO_SIZES_VALID) != 0) {
                uint64_t sum = 0;
                for (uint32_t j = 0; j < entry.instanceCount; ++j) {
                    if (entry.instanceSizes[j] == 0 || entry.instanceSizes[j] > rankSize) {
                        return false;
                    }
                    sum += entry.instanceSizes[j];
                }
                if (sum != rankSize) {
                    return false;
                }
            }
            foundMeshLayer = foundMeshLayer || entry.layer == meshLayer;
            foundClosLayer = foundClosLayer || entry.layer == closLayer;
            foundMeshMembers = foundMeshMembers ||
                (entry.layer == meshLayer && entry.meshMemberMask == localMemberMask);
        }
        if (!foundMeshLayer || !foundClosLayer) {
            return false;
        }
        if (kind == TopologyKind::UNKNOWN) {
            return localMemberMask == 0 && evidence == TopologyEvidence::NONE && serverCount == 0;
        }
        if (reason != TopologyReason::RECOGNIZED || evidence == TopologyEvidence::NONE ||
            (localMemberMask & RankBit(myRank)) == 0 ||
            ShapeFromLocalSize(rankSize, RankMaskSize(localMemberMask)) != kind) {
            return false;
        }
        if (evidence == TopologyEvidence::SINGLETON_CASE) {
            return kind == TopologyKind::FOUR_BY_ONE && localMemberMask == RankBit(myRank) && serverCount == 0 &&
                closLayer != INVALID_VALUE_RANKID;
        }
        if (meshLayer == INVALID_VALUE_RANKID || !foundMeshMembers) {
            return false;
        }
        if (evidence == TopologyEvidence::LAYER_INSTANCES) {
            return ShapeFromSizes(rankSize, serverSizes, serverCount) == kind;
        }
        return serverCount == 0;
    }
};

inline bool UseFourByOneCompact(const TopologyFacts &topology, uint32_t rankSize)
{
    return rankSize == 4 && topology.kind == TopologyKind::FOUR_BY_ONE;
}

inline bool UseFourByThreeCompact(const TopologyFacts &topology, uint32_t rankSize)
{
    return rankSize == 12 && topology.kind == TopologyKind::FOUR_BY_THREE;
}

inline bool UseEightSideRootPhase(const TopologyFacts &topology, uint32_t rankSize,
    uint32_t myRank, uint32_t groupCount)
{
    const uint32_t mask = topology.localMemberMask;
    return rankSize == 12 && myRank < rankSize && topology.kind == TopologyKind::EIGHT_PLUS_FOUR &&
        groupCount == DIRECT_MAX_DIE_GROUPS && RankMaskSize(mask) == 8 &&
        (mask & ~AllRankMask(rankSize)) == 0 && (mask & RankBit(myRank)) != 0;
}

inline bool RootOnEightSide(const TopologyFacts &topology, uint32_t rankSize, uint32_t root)
{
    if (rankSize != 12 || root >= rankSize || topology.kind != TopologyKind::EIGHT_PLUS_FOUR) {
        return false;
    }
    const uint32_t localSize = RankMaskSize(topology.localMemberMask);
    const bool rootLocal = (topology.localMemberMask & RankBit(root)) != 0;
    return rootLocal ? localSize == 8 : localSize == 4;
}

struct PeerLinkInfo {
    uint32_t valid = 0;
    uint32_t peer = INVALID_VALUE_RANKID;
    uint32_t layer = INVALID_VALUE_RANKID;
    uint32_t dieId = INVALID_VALUE_RANKID;
    uint32_t hop = 0;
    PathClass pathClass = PathClass::UNKNOWN;
    EndpointDesc localEndpoint{};
    EndpointDesc remoteEndpoint{};
    ChannelHandle channel = 0;
};

// Root authorizes a layout and exact local group; actual supplier descriptors
// partition output with root-filled gaps. No remote server mask is guessed.
enum class PullLayout : uint64_t {
    DIRECT = 0, TWO_BY_EIGHT_PREFIXES, EIGHT_PLUS_FOUR_BANDS, FOUR_BY_THREE_PREFIXES, EIGHT_PLUS_FOUR_PREFIXES
};

inline bool UsesEarlyOffer(PullLayout layout)
{
    return layout == PullLayout::TWO_BY_EIGHT_PREFIXES || layout == PullLayout::FOUR_BY_THREE_PREFIXES ||
        layout == PullLayout::EIGHT_PLUS_FOUR_PREFIXES;
}

constexpr uint32_t CHANNEL_NOTIFY_COUNT = 8;
constexpr uint32_t DIRECT_THREAD_NOTIFY_COUNT = 1;
constexpr uint32_t DIRECT_THREAD_START = 0;
constexpr uint32_t DIRECT_THREAD_DONE = 0;
constexpr uint32_t PULL_NOTIFY = 0;
constexpr uint16_t PULL_META_MASK = 0x3;
constexpr uint16_t PULL_ROOT_META_MASK = 0xf;
constexpr uint16_t PULL_DONE_MASK = 0x100;
constexpr uint16_t ROOT_RELEASE_MASK = 0x200;
constexpr uint32_t OFFER_NOTIFY = 1;
constexpr uint16_t OFFER_META_MASK = 0x7f;
constexpr uint16_t OFFER_READY_MASK = 0x80;
// Every nonroot pair publishes META, including zero-byte offers. The root8
// fused entry writes only BYTES=0 with the complete mask; every N12 Large
// receiver reads BYTES first and initializes zero descriptors locally.
// Prefix layouts publish complete offers before waiting and send READY for each nonzero piece,
// even if root authorization rejects its prefetch. Legacy 8+4 bands retain
// post-prefetch META without READY. Receivers drain READY based on the
// descriptor's type and raw bytes, independently of payload authorization.
// Root releases a frame only after every rank has finished reading and drained
// its notifications, protecting scratch and channel XNs across root changes.
// Root-channel XNs 0..3: base, token, layout, mask.
// V16 starts reads before the full offer drain only after topology/header
// authorization: 12-rank DIRECT or an exact 2x8 mask with a unique prefix supplier.
// Other-channel XNs 0..6: offset, bytes, negative end, address, token,
// negative expected layout and negative expected root mask. XN 7 is unused.
// No XN/HBM Load or Store is used: the VM checker does not model XN addresses
// as ordinary data slices. Scratch contains payload only, at relative offset 0.
constexpr uint64_t PULL_PAYLOAD_OFFSET = 0;
static_assert((PULL_ROOT_META_MASK & PULL_DONE_MASK) == 0 &&
    ((PULL_ROOT_META_MASK | PULL_DONE_MASK) & ROOT_RELEASE_MASK) == 0, "Distinct root notification bits");
static_assert((OFFER_META_MASK & OFFER_READY_MASK) == 0, "Separate descriptor and payload readiness bits");

struct PullKernelArg {
    uint32_t myRank = 0;
    uint32_t rankSize = 0;
    uint32_t staticRoot = INVALID_VALUE_RANKID; // Per-root PullGroup and OfferGroup specializations.
    uint32_t peerCount = 0;
    uint32_t peers[MAX_RANK_SIZE]{};
    ChannelHandle channels[MAX_RANK_SIZE]{};
};
// Registration-only wrapper; the serialized context and runtime GT args stay unchanged.
struct PrefixRegArg {
    PullKernelArg kernel;
    uint32_t expectedRootMask = 0;
};
struct PullPeerArg {
    ChannelHandle channel = 0;
    uint32_t myRank = 0;
    uint32_t rankSize = 0;
};
struct SupplyPiece {
    uint64_t offset = 0;
    uint64_t bytes = 0;
    uint64_t negativeEnd = 0; // -(offset + bytes), modulo 2^64.
    uint64_t bufferOffset = 0; // Relative to the original HCCL allocation.
    uint64_t inputOffset = 0; // Relative to root's whole input.
};
struct SupplyPlan {
    PullLayout layout = PullLayout::DIRECT;
    uint32_t rootMask = 0;
    SupplyPiece pieces[MAX_RANK_SIZE]{};
};

// Registration/execution share this structural decision for each possible root.
// It never enters the serialized context or depends on this call's message size.
struct HelperPreparationPlan {
    uint32_t target = INVALID_VALUE_RANKID;
    uint32_t rootGroup = INVALID_VALUE_RANKID;
    uint32_t targetGroup = INVALID_VALUE_RANKID;
};
inline bool BuildHelperPreparationPlan(const AlgResourceCtx &ctx, uint32_t root, HelperPreparationPlan &plan);

enum SmallRootArg : uint32_t {
    SR_INPUT, SR_INPUT_TOKEN, SR_COPY_SOURCE, SR_OUTPUT, SR_OUTPUT_TOKEN, SR_BUFFER, SR_BUFFER_TOKEN,
    SR_COPY_BYTES, SR_STAGING, SR_DO_COPY, SR_ARG_COUNT
};
// Legacy SR_DO_COPY: 0 = none, 1 = after DONE, 2 = safe copy immediately after META.
enum DirectRootArg : uint32_t { DR_INPUT, DR_INPUT_TOKEN, DR_ARG_COUNT };
enum DirectRootCopyArg : uint32_t {
    DRC_INPUT, DRC_INPUT_TOKEN, DRC_SOURCE, DRC_OUTPUT, DRC_OUTPUT_TOKEN, DRC_BYTES, DRC_ARG_COUNT
};
enum SmallPeerArg : uint32_t { SP_OUTPUT, SP_OUTPUT_TOKEN, SP_SOURCE_OFFSET, SP_BYTES, SP_ARG_COUNT };
enum RootPublishArg : uint32_t { RP_INPUT, RP_TOKEN, RP_LAYOUT, RP_MASK, RP_ARG_COUNT };
enum RootPhaseArg : uint32_t {
    RPH_INPUT, RPH_TOKEN, RPH_LAYOUT, RPH_MASK, RPH_COPY_SOURCE, RPH_OUTPUT, RPH_OUTPUT_TOKEN,
    RPH_COPY_BYTES, RPH_DO_COPY, RPH_ARG_COUNT
};
enum PrefetchArg : uint32_t {
    // EXPECTED fields contain negatives, modulo 2^64, for CCU equality checks.
    PF_BUFFER, PF_BUFFER_TOKEN, PF_EXPECTED_LAYOUT, PF_EXPECTED_MASK,
    PF_INPUT_OFFSET, PF_BUFFER_OFFSET, PF_BYTES, PF_ARG_COUNT
};
enum OfferArg : uint32_t {
    // EXPECTED fields use the same negative representation as PrefetchArg.
    OF_OFFSET, OF_BYTES, OF_NEGATIVE_END, OF_ADDRESS, OF_TOKEN,
    OF_EXPECTED_LAYOUT, OF_EXPECTED_MASK, OF_ARG_COUNT
};
// Batched zero offers plus at most one nonzero prefix, sharing immutable fields.
// Every peer still receives the complete seven-field V12 descriptor.
enum BatchOfferArg : uint32_t {
    BG_ADDRESS, BG_TOKEN, BG_EXPECTED_LAYOUT, BG_EXPECTED_MASK,
    BG_TARGET, BG_BYTES, BG_NEGATIVE_END, BG_ARG_COUNT
};
// The first seven values mirror BG, but these replacement slots always use HP8,
// including zero-byte supply. Only the root-containing group consumes root META.
enum HelperPreparationArg : uint32_t {
    HP_ADDRESS, HP_TOKEN, HP_NEGATIVE_LAYOUT, HP_NEGATIVE_MASK, HP_TARGET, HP_BYTES,
    HP_NEGATIVE_END, HP_SOURCE_OFFSET, HP_ARG_COUNT
};
enum GroupTaskArg : uint32_t {
    GT_OUTPUT, GT_OUTPUT_TOKEN, GT_SOURCE_OFFSET, GT_BYTES, GT_ARG_COUNT
};
enum CopyArg : uint32_t {
    COPY_SOURCE, COPY_SOURCE_TOKEN, COPY_OUTPUT, COPY_OUTPUT_TOKEN, COPY_BUFFER, COPY_BUFFER_TOKEN,
    COPY_BYTES, COPY_STAGING, COPY_ARG_COUNT
};
// Keep each launch within one SQE's argument payload. The SDK also supports
// longer argument lists; this protocol deliberately needs no continuation SQE.
constexpr uint32_t SINGLE_SQE_ARG_COUNT = 13;
static_assert(SR_ARG_COUNT <= SINGLE_SQE_ARG_COUNT && SP_ARG_COUNT <= SINGLE_SQE_ARG_COUNT &&
    RP_ARG_COUNT <= SINGLE_SQE_ARG_COUNT && PF_ARG_COUNT <= SINGLE_SQE_ARG_COUNT &&
    OF_ARG_COUNT <= SINGLE_SQE_ARG_COUNT && GT_ARG_COUNT <= SINGLE_SQE_ARG_COUNT &&
    COPY_ARG_COUNT <= SINGLE_SQE_ARG_COUNT && DR_ARG_COUNT <= SINGLE_SQE_ARG_COUNT &&
    DRC_ARG_COUNT <= SINGLE_SQE_ARG_COUNT && BG_ARG_COUNT <= SINGLE_SQE_ARG_COUNT &&
    HP_ARG_COUNT <= SINGLE_SQE_ARG_COUNT && RPH_ARG_COUNT <= SINGLE_SQE_ARG_COUNT,
    "Each Scatter task uses one argument SQE");
struct DirectGroup {
    uint32_t dieId = INVALID_VALUE_RANKID;
    PullKernelArg kernelArg;
    CcuKernelHandle smallRoot = 0;
    CcuKernelHandle directRoot = 0;
    CcuKernelHandle directRootCopy = 0;
    CcuKernelHandle publish = 0;
    CcuKernelHandle rootWait = 0;
    CcuKernelHandle rootRelease = 0;
};
inline PullLayout RootPullLayout(const TopologyFacts &topology)
{
    switch (topology.kind) {
        case TopologyKind::TWO_BY_EIGHT:
            return PullLayout::TWO_BY_EIGHT_PREFIXES;
        case TopologyKind::EIGHT_PLUS_FOUR:
            return RankMaskSize(topology.localMemberMask) == 4 ?
                (V13_ROOT4_PREFIXES ? PullLayout::EIGHT_PLUS_FOUR_PREFIXES : PullLayout::EIGHT_PLUS_FOUR_BANDS) :
                PullLayout::DIRECT;
        case TopologyKind::FOUR_BY_THREE:
            return PullLayout::FOUR_BY_THREE_PREFIXES;
        default:
            return PullLayout::DIRECT;
    }
}
} // namespace ops_hccl

struct CommBuffer {
    void *addr = nullptr;
    uint64_t size = 0;
};
struct AlgResourceCtx {
    uint64_t magic = ops_hccl::DIRECT_CTX_MAGIC;
    uint32_t version = ops_hccl::DIRECT_CTX_VERSION;
    uint32_t myRank = 0;
    uint32_t rankSize = 0;
    uint32_t groupCount = 0;
    uint32_t workerReady = 0;
    ThreadHandle worker = 0;
    CommBuffer localBuffer;
    ops_hccl::TopologyFacts topology;
    ops_hccl::PeerLinkInfo peerLinks[MAX_RANK_SIZE]{};
    ops_hccl::DirectGroup groups[ops_hccl::DIRECT_MAX_DIE_GROUPS]{};
    CcuKernelHandle smallPeers[MAX_RANK_SIZE]{};
    CcuKernelHandle captures[MAX_RANK_SIZE]{};
    CcuKernelHandle prefetches[MAX_RANK_SIZE]{};
    CcuKernelHandle rootAcks[MAX_RANK_SIZE]{};
    CcuKernelHandle offerSends[MAX_RANK_SIZE]{};
    CcuKernelHandle offerReadies[MAX_RANK_SIZE]{};
    CcuKernelHandle offerGroups[MAX_RANK_SIZE][ops_hccl::DIRECT_MAX_DIE_GROUPS]{};
    CcuKernelHandle pullGroups[MAX_RANK_SIZE][ops_hccl::DIRECT_MAX_DIE_GROUPS]{};
    CcuKernelHandle copyKernel = 0;

    bool Valid() const
    {
        using namespace ops_hccl;
        if (magic != DIRECT_CTX_MAGIC || version != DIRECT_CTX_VERSION || rankSize == 0 ||
            rankSize > MAX_RANK_SIZE || myRank >= rankSize || groupCount > DIRECT_MAX_DIE_GROUPS ||
            workerReady > 1 || (workerReady == 0) != (worker == 0) || copyKernel == 0 ||
            localBuffer.addr == nullptr || localBuffer.size < ELEMENT_BYTES ||
            !topology.Valid(rankSize, myRank)) {
            return false;
        }
        uint32_t members = 0;
        uint32_t dies = 0;
        for (uint32_t g = 0; g < groupCount; ++g) {
            const auto &group = groups[g];
            const auto &arg = group.kernelArg;
            if (group.dieId >= DIRECT_MAX_DIE_GROUPS || (dies & (1U << group.dieId)) != 0 ||
                arg.myRank != myRank || arg.rankSize != rankSize || arg.peerCount == 0 ||
                arg.peerCount >= rankSize || group.smallRoot == 0 || group.publish == 0 ||
                group.rootWait == 0 || group.rootRelease == 0) {
                return false;
            }
            if (V13_SMALL_ROOT_SPECIALIZATION && (group.directRoot == 0 || group.directRootCopy == 0)) {
                return false;
            }
            dies |= 1U << group.dieId;
            for (uint32_t i = 0; i < arg.peerCount; ++i) {
                const uint32_t peer = arg.peers[i];
                if (peer >= rankSize || peer == myRank || (members & RankBit(peer)) != 0) {
                    return false;
                }
                const auto &link = peerLinks[peer];
                if (link.valid != 1 || link.peer != peer || link.dieId != group.dieId ||
                    link.channel == 0 || arg.channels[i] != link.channel ||
                    smallPeers[peer] == 0 || captures[peer] == 0 || prefetches[peer] == 0 ||
                    rootAcks[peer] == 0 || offerSends[peer] == 0 || offerReadies[peer] == 0) {
                    return false;
                }
                members |= RankBit(peer);
            }
            for (uint32_t root = 0; root < rankSize; ++root) {
                if (root == myRank) {
                    continue;
                }
                if (pullGroups[root][g] == 0) {
                    return false;
                }
                bool needsOfferSlot = arg.peerCount > 1 || arg.peers[0] != root;
                if (!needsOfferSlot && V13_BATCH_OFFERS) {
                    HelperPreparationPlan preparation;
                    needsOfferSlot = BuildHelperPreparationPlan(*this, root, preparation);
                }
                if (V13_BATCH_OFFERS && needsOfferSlot && offerGroups[root][g] == 0) {
                    return false;
                }
            }
        }
        return members == (AllRankMask(rankSize) & ~RankBit(myRank));
    }
    std::vector<char> Serialize() const
    {
        std::vector<char> result(sizeof(AlgResourceCtx));
        std::memcpy(result.data(), this, sizeof(AlgResourceCtx));
        return result;
    }
    bool DeSerialize(const void *data, uint64_t size)
    {
        if (data == nullptr || size != sizeof(AlgResourceCtx)) {
            return false;
        }
        std::memcpy(this, data, sizeof(AlgResourceCtx));
        return Valid();
    }
};

namespace ops_hccl {
inline bool BuildHelperPreparationPlan(const AlgResourceCtx &ctx, uint32_t root, HelperPreparationPlan &plan)
{
    const uint32_t mask = ctx.topology.localMemberMask;
    if (!V13_BATCH_OFFERS || ctx.rankSize != 16 || ctx.myRank >= ctx.rankSize || root >= ctx.rankSize ||
        root == ctx.myRank || ctx.topology.kind != TopologyKind::TWO_BY_EIGHT ||
        ctx.groupCount != DIRECT_MAX_DIE_GROUPS || RankMaskSize(mask) != 8 ||
        (mask & ~AllRankMask(ctx.rankSize)) != 0 || (mask & RankBit(ctx.myRank)) == 0 ||
        (mask & RankBit(root)) == 0 || ctx.localBuffer.addr == nullptr ||
        ctx.localBuffer.size <= PULL_PAYLOAD_OFFSET) {
        return false;
    }
    uint32_t helperOrdinal = 0;
    uint32_t remoteCount = 0;
    uint32_t remoteRanks[MAX_RANK_SIZE]{};
    for (uint32_t rank = 0; rank < ctx.rankSize; ++rank) {
        if ((mask & RankBit(rank)) == 0) {
            remoteRanks[remoteCount++] = rank;
        } else if (rank != root && rank < ctx.myRank) {
            ++helperOrdinal;
        }
    }
    if (remoteCount != 8 || helperOrdinal >= 7) {
        return false;
    }
    const uint32_t target = remoteRanks[helperOrdinal];
    const auto &rootLink = ctx.peerLinks[root];
    const auto &targetLink = ctx.peerLinks[target];
    if (rootLink.valid != 1 || targetLink.valid != 1 || rootLink.peer != root || targetLink.peer != target ||
        rootLink.channel == 0 || targetLink.channel == 0 || rootLink.hop != 1 ||
        rootLink.dieId >= DIRECT_MAX_DIE_GROUPS || targetLink.dieId >= DIRECT_MAX_DIE_GROUPS ||
        rootLink.dieId == targetLink.dieId) {
        return false;
    }
    HelperPreparationPlan candidate;
    candidate.target = target;
    for (uint32_t g = 0; g < ctx.groupCount; ++g) {
        const auto &group = ctx.groups[g];
        const auto &arg = group.kernelArg;
        if (group.dieId >= DIRECT_MAX_DIE_GROUPS || arg.peerCount == 0 || arg.peerCount >= ctx.rankSize) {
            return false;
        }
        for (uint32_t i = 0; i < arg.peerCount; ++i) {
            const uint32_t peer = arg.peers[i];
            if (peer != root && peer != target) {
                continue;
            }
            const auto &link = ctx.peerLinks[peer];
            uint32_t &index = peer == root ? candidate.rootGroup : candidate.targetGroup;
            if (index != INVALID_VALUE_RANKID || group.dieId != link.dieId || arg.channels[i] != link.channel) {
                return false;
            }
            index = g;
        }
    }
    if (candidate.rootGroup == INVALID_VALUE_RANKID || candidate.targetGroup == INVALID_VALUE_RANKID ||
        candidate.rootGroup == candidate.targetGroup) {
        return false;
    }
    plan = candidate;
    return true;
}
} // namespace ops_hccl
static_assert(std::is_trivially_copyable<AlgResourceCtx>::value, "Engine context must be byte-copyable");
#endif // OPS_HCCL_CUSTOM_H
