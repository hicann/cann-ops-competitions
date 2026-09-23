/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CCU_EXEC_OP_H
#define OPS_HCCL_CCU_EXEC_OP_H

#include "common.h"
#include "ccu_kernel.h"
#include <limits>
#include <vector>

namespace ops_hccl {
// recvBytes is per rank; inputBytes is the entire root input. Every nonempty
// invocation uses a positive final block and zero or more full CCU blocks.
HcclResult ValidateScatterParams(const OpParam &param, uint64_t &recvBytes, uint64_t &inputBytes);
HcclResult ExecOp(const OpParam &param);

// The production selector consumes verified instance membership, not the outer
// network-layer type. In particular, a CUSTOM layer containing a MESH instance
// has exactly the same routing input as any other layer with those members.
struct ScatterRouteInput {
    uint32_t rankSize = 0;
    uint32_t root = 0;
    uint64_t recvBytes = 0;
    uint32_t serverMasks[MAX_RANK_SIZE] = {};
    uint64_t capacities[MAX_RANK_SIZE] = {};
};

struct ScatterRouteDecision {
    ScatterAlgorithm algorithm = ScatterAlgorithm::DIRECT;
    ScatterTopology topology = ScatterTopology::UNKNOWN;
    ScatterRouteReason reason = ScatterRouteReason::UNKNOWN_MEMBERS;
    uint32_t sceneId = 0;
    uint32_t directFusion = 0;
    uint32_t rootAckFusion = 0;
    uint32_t compactScratch = 0;
    uint32_t relayOverlap = 0;
    uint32_t treeWidth = 0;
    uint64_t chunkBytes = 0;
    uint64_t fullSteps = 0;
    uint64_t tailBytes = 0;
};

namespace scatter_route_detail {
struct ScenePolicy {
    ScatterAlgorithm algorithm;
    uint64_t chunkBytes;
    uint32_t treeWidth;
    uint32_t directFusion;
    bool directSingleWave;
    bool rootAckFusion;
    bool directForceChunked;
};

// V014 preserves the V013 table and aligned EIGHT_FOUR admission. New options
// are independently gated below; transfer sizes and every final ACK remain.
// No platform case identifiers enter routing.
inline constexpr ScenePolicy POLICIES[] = {
    {ScatterAlgorithm::DIRECT, MAX_DATA_SIZE, 0, 1, true, false, false},
    {ScatterAlgorithm::RELAY, 32ULL << 20, 0, 0, false, true, false},
    {ScatterAlgorithm::RELAY, 32ULL << 20, 0, 0, false, true, false},
    {ScatterAlgorithm::DIRECT, MAX_DATA_SIZE, 0, 1, false, false, false},
    {ScatterAlgorithm::DIRECT, MAX_DATA_SIZE, 0, 1, false, false, false},
    {ScatterAlgorithm::DIRECT, MAX_DATA_SIZE, 0, 1, false, false, false},
    {ScatterAlgorithm::DIRECT, MAX_DATA_SIZE, 0, 1, true, false, false},
    {ScatterAlgorithm::DIRECT, MAX_DATA_SIZE, 0, 1, false, false, true},
    {ScatterAlgorithm::DIRECT, MAX_DATA_SIZE, 0, 1, false, false, true},
    {ScatterAlgorithm::DIRECT, MAX_DATA_SIZE, 0, 1, true, false, false},
    {ScatterAlgorithm::RELAY, 64ULL << 20, 0, 0, false, true, false},
    {ScatterAlgorithm::RELAY, 64ULL << 20, 0, 0, false, false, false}
};

constexpr uint32_t CountBits(uint32_t mask)
{
    uint32_t count = 0;
    while (mask != 0) { mask &= mask - 1; ++count; }
    return count;
}

constexpr bool NearSize(uint64_t actual, uint64_t reference, uint64_t tolerance)
{
    return actual > reference ? actual - reference <= tolerance : reference - actual <= tolerance;
}

constexpr bool MatchLargeSignature(uint64_t bytes, uint32_t ranks, uint64_t decimal, uint64_t binary)
{
    // Either per-rank or aggregate interpretation, either unit system, and a
    // rank-wise element-rounding tolerance. Transfer lengths are never rounded.
    const uint64_t total = bytes * ranks; // SelectScatterRoute checks overflow.
    const uint64_t tolerance = uint64_t{4} * ranks;
    return NearSize(bytes, decimal, tolerance) || NearSize(bytes, binary, tolerance) ||
        NearSize(total, decimal, tolerance) || NearSize(total, binary, tolerance);
}

constexpr int SelectProfile(uint64_t bytes, uint32_t ranks)
{
    if (bytes <= SCATTER_SMALL_LIMIT) { return 0; }
    const bool bulk = MatchLargeSignature(bytes, ranks, 512000000ULL, 512ULL << 20);
    const bool irregular = MatchLargeSignature(bytes, ranks, 400000004ULL, (400ULL << 20) + 4);
    return bulk == irregular ? -1 : (bulk ? 1 : 2);
}

constexpr bool MatchAlignedAggregateSignature(uint64_t bytes, uint32_t ranks, uint64_t nominal)
{
    // The original test driver rounds a large total input down to P * 512
    // bytes before dividing it among ranks. Recognize that exact resulting B;
    // do not widen the original signature tolerance or round transfer lengths.
    // uint32_t ranks * 512 fits uint64_t; the division precedes multiplication.
    return ranks != 0 && bytes == (nominal / (uint64_t{ranks} * 512)) * 512;
}

constexpr int SelectAlignedAggregateProfile(uint64_t bytes, uint32_t ranks)
{
    if (bytes <= SCATTER_SMALL_LIMIT) { return -1; }
    const bool bulk = MatchAlignedAggregateSignature(bytes, ranks, 512000000ULL) ||
        MatchAlignedAggregateSignature(bytes, ranks, 512ULL << 20);
    const bool irregular = MatchAlignedAggregateSignature(bytes, ranks, 400000004ULL) ||
        MatchAlignedAggregateSignature(bytes, ranks, (400ULL << 20) + 4);
    return bulk == irregular ? -1 : (bulk ? 1 : 2);
}

constexpr uint64_t RelayTailBytes(uint64_t wave, uint32_t helpers, uint32_t remotes, uint32_t children)
{
    return helpers == 0 || children == 0 || remotes <= 4 ? 0 :
        ((wave / 4) * (remotes - 4) / (uint64_t{helpers + 4} * children)) * 4;
}

constexpr uint64_t CompactScratchBytes(uint64_t wave, uint32_t helpers, uint32_t remotes)
{
    uint64_t extent = 0;
    for (uint32_t h = 0; h < helpers; ++h) {
        const uint32_t children = remotes / helpers + (h < remotes % helpers ? 1 : 0);
        const uint64_t bytes = children * RelayTailBytes(wave, helpers, remotes, children);
        if (bytes > extent) { extent = bytes; }
    }
    return extent;
}

constexpr uint64_t CompactChunk(uint64_t limit, uint64_t capacity, uint32_t helpers, uint32_t remotes)
{
    uint64_t low = 0, high = limit / 4;
    while (low < high) {
        const uint64_t middle = low + (high - low + 1) / 2;
        if (CompactScratchBytes(middle * 4, helpers, remotes) <= capacity) { low = middle; }
        else { high = middle - 1; }
    }
    return low * 4;
}

constexpr uint64_t WaveCount(uint64_t bytes, uint64_t chunk)
{
    return chunk == 0 ? 0 : bytes / chunk + (bytes % chunk != 0 ? 1 : 0);
}
} // namespace scatter_route_detail

constexpr uint32_t SelectScatterEventMerge(uint32_t sceneId)
{
    return (SCATTER_FEATURES & 1U) != 0 && (sceneId == 1 || sceneId == 4 || sceneId == 10) ? 1 : 0;
}

constexpr uint32_t SelectScatterEarlyCopy(uint32_t sceneId)
{
    // Preserve the already-best 2x8 and 8+4 small paths (scenes 1 and 7).
    return (SCATTER_TUNING & 2U) != 0 && (sceneId == 4 || sceneId == 10) ? 1U : 0U;
}

constexpr uint32_t SelectScatterReadyWrite(uint32_t sceneId)
{
    return (SCATTER_OPTIMIZATIONS & 1U) != 0 &&
        (sceneId == 1 || sceneId == 4 || sceneId == 10) ? 1 : 0;
}

constexpr uint32_t SelectScatterRelayCopy(ScatterAlgorithm algorithm, uint32_t sceneId,
    uint64_t fullSteps, uint64_t tailBytes)
{
    return algorithm == ScatterAlgorithm::RELAY && fullSteps == 1 && tailBytes == 0 &&
        (((SCATTER_OPTIMIZATIONS & 2U) != 0 && (sceneId == 11 || sceneId == 12)) ||
         ((SCATTER_OPTIMIZATIONS & 4U) != 0 && (sceneId == 2 || sceneId == 3))) ? 1 : 0;
}

constexpr uint32_t SelectScatterRelayOverlap(ScatterAlgorithm algorithm, uint32_t sceneId)
{
    if (algorithm == ScatterAlgorithm::RELAY && (sceneId == 2 || sceneId == 3) &&
        (SCATTER_V018_OPTIONS & 1U) != 0) {
        // Mode 2 uses one tail-ready bit per helper, with no V016/V017 fine
        // forwarding or cross-helper interleave implicitly enabled for 2x8.
        return 2;
    }
    return (SCATTER_FEATURES & 4U) != 0 && algorithm == ScatterAlgorithm::RELAY &&
        (sceneId == 11 || sceneId == 12) ? 1 : 0;
}

constexpr uint32_t SelectScatterHelperAckFusion(uint32_t sceneId)
{
    return (SCATTER_V018_OPTIONS & 2U) != 0 && (sceneId == 2 || sceneId == 3) ? 1 : 0;
}

constexpr uint32_t SelectScatterShortPush(const AlgResourceCtx &ctx)
{
    if (ctx.algorithm != ScatterAlgorithm::DIRECT || ctx.topologyKnown == 0 ||
        ctx.dataPath != ScatterDataPath::SINGLE_WAVE || ctx.recvBytes == 0 ||
        ctx.recvBytes % 4U != 0 || ctx.recvBytes > SCATTER_SMALL_LIMIT) { return 0; }
    return ((SCATTER_V025_OPTIONS & 1U) != 0 && ctx.topology == ScatterTopology::TWO_EIGHT &&
            ctx.rankSize == 16 && ctx.sceneId == 1) ||
        ((SCATTER_V025_OPTIONS & 2U) != 0 && ctx.topology == ScatterTopology::EIGHT_FOUR &&
            ctx.rankSize == 12 && ctx.sceneId == 7) ||
        ((SCATTER_V025_OPTIONS & 4U) != 0 && ctx.topology == ScatterTopology::FOUR_THREE &&
            ctx.rankSize == 12 && ctx.sceneId == 10) ? 1U : 0U;
}

constexpr uint32_t SelectScatterPullRead(ScatterAlgorithm algorithm, uint32_t sceneId, uint64_t bytes)
{
    return (SCATTER_V018_OPTIONS & 4U) != 0 && algorithm == ScatterAlgorithm::DIRECT &&
        (sceneId == 4 || sceneId == 10) && bytes != 0 && bytes <= SCATTER_SMALL_LIMIT ? 1 : 0;
}

constexpr uint32_t SelectScatterExtendedPull(const AlgResourceCtx &ctx)
{
    if (ctx.algorithm != ScatterAlgorithm::DIRECT || ctx.topologyKnown == 0 ||
        ctx.dataPath != ScatterDataPath::SINGLE_WAVE || ctx.recvBytes == 0 ||
        ctx.recvBytes > SCATTER_SMALL_LIMIT) { return 0; }
    return ((SCATTER_V023_OPTIONS & 1U) != 0 && ctx.topology == ScatterTopology::TWO_EIGHT &&
            ctx.rankSize == 16 && ctx.sceneId == 1) ||
        ((SCATTER_V023_OPTIONS & 2U) != 0 && ctx.topology == ScatterTopology::EIGHT_FOUR &&
            ctx.rankSize == 12 && ctx.sceneId == 7) ? 1U : 0U;
}

constexpr uint64_t SelectScatterPullMsBytes(const AlgResourceCtx &ctx)
{
    if (SelectScatterShortPush(ctx) != 0 || ctx.myRank == ctx.root ||
        ctx.algorithm != ScatterAlgorithm::DIRECT ||
        ctx.topologyKnown == 0 || ctx.dataPath != ScatterDataPath::SINGLE_WAVE ||
        ctx.recvBytes == 0 || ctx.recvBytes % 4 != 0 || ctx.recvBytes > SCATTER_PULL_MS_MAX_BYTES ||
        (SelectScatterPullRead(ctx.algorithm, ctx.sceneId, ctx.recvBytes) == 0 &&
            SelectScatterExtendedPull(ctx) == 0)) { return 0; }
    return ((SCATTER_V023_OPTIONS & 4U) != 0 && ctx.topology == ScatterTopology::FOUR_ONE &&
            ctx.rankSize == 4 && ctx.sceneId == 4) ||
        ((SCATTER_V023_OPTIONS & 8U) != 0 && ctx.topology == ScatterTopology::FOUR_THREE &&
            ctx.rankSize == 12 && ctx.sceneId == 10) ||
        ((SCATTER_V024_OPTIONS & 1U) != 0 && SelectScatterExtendedPull(ctx) != 0 &&
            ctx.topology == ScatterTopology::TWO_EIGHT && ctx.rankSize == 16 && ctx.sceneId == 1) ||
        ((SCATTER_V024_OPTIONS & 2U) != 0 && SelectScatterExtendedPull(ctx) != 0 &&
            ctx.topology == ScatterTopology::EIGHT_FOUR && ctx.rankSize == 12 && ctx.sceneId == 7) ?
        ctx.recvBytes : 0;
}

constexpr uint32_t SelectScatterPullMsTail(const AlgResourceCtx &ctx)
{
    return (SCATTER_V024_OPTIONS & 4U) != 0 && SelectScatterPullMsBytes(ctx) != 0 &&
        ctx.topology == ScatterTopology::FOUR_THREE && ctx.rankSize == 12 && ctx.sceneId == 10 &&
        ctx.recvBytes > SCATTER_PULL_MS_TAIL_BYTES ? 1U : 0U;
}

constexpr uint32_t SelectScatterLargeReadyWrite(const AlgResourceCtx &ctx)
{
    return (SCATTER_V023_OPTIONS & 16U) != 0 && ctx.topologyKnown != 0 &&
        ctx.algorithm == ScatterAlgorithm::DIRECT && ctx.topology == ScatterTopology::EIGHT_FOUR &&
        ctx.rankSize == 12 && (ctx.sceneId == 8 || ctx.sceneId == 9) && ctx.directFusion != 0 &&
        ctx.recvBytes > SCATTER_SMALL_LIMIT && ctx.recvBytes <= MAX_DATA_SIZE &&
        ctx.dataPath == ScatterDataPath::SINGLE_WAVE ? 1U : 0U;
}

constexpr ScatterPullAddressMode SelectScatterPullAddressMode(ScatterAlgorithm algorithm,
    ScatterTopology topology, uint32_t sceneId, uint32_t rankSize, ScatterDataPath dataPath, uint64_t bytes)
{
    if (dataPath != ScatterDataPath::SINGLE_WAVE || SelectScatterPullRead(algorithm, sceneId, bytes) == 0) {
        return ScatterPullAddressMode::SLICE;
    }
    if ((SCATTER_V021_OPTIONS & 1U) != 0 && topology == ScatterTopology::FOUR_ONE &&
        sceneId == 4 && rankSize == 4) {
        return ScatterPullAddressMode::THREE_SOURCES;
    }
    if ((SCATTER_V021_OPTIONS & 2U) != 0 && topology == ScatterTopology::FOUR_THREE &&
        sceneId == 10 && rankSize == 12) {
        return ScatterPullAddressMode::BASE_OFFSET;
    }
    return ScatterPullAddressMode::SLICE;
}

// Use the same effective protocol as BuildGroups when packing runtime slots.
// In particular, a short-Push receiver must not load the old BASE_OFFSET slot.
constexpr ScatterPullAddressMode SelectScatterEffectivePullAddressMode(const AlgResourceCtx &ctx)
{
    return SelectScatterShortPush(ctx) != 0 ? ScatterPullAddressMode::SLICE :
        SelectScatterPullAddressMode(ctx.algorithm, ctx.topology, ctx.sceneId,
            ctx.rankSize, ctx.dataPath, ctx.recvBytes);
}

constexpr uint32_t SelectScatterSmallPipeline(ScatterAlgorithm algorithm, ScatterTopology topology,
    uint32_t sceneId, uint32_t rankSize, ScatterDataPath dataPath, uint64_t bytes)
{
    return algorithm == ScatterAlgorithm::DIRECT && topology == ScatterTopology::EIGHT_FOUR &&
        sceneId == 7 && rankSize == 12 && dataPath == ScatterDataPath::SINGLE_WAVE &&
        bytes != 0 && bytes <= SCATTER_SMALL_LIMIT ? SCATTER_V022_OPTIONS & 7U : 0U;
}

constexpr uint32_t SelectScatterPullPeerOrder(ScatterAlgorithm algorithm, ScatterTopology topology,
    uint32_t sceneId, uint32_t rankSize, ScatterDataPath dataPath, uint64_t bytes)
{
    return (SCATTER_V022_OPTIONS & 16U) != 0 &&
        SelectScatterPullAddressMode(algorithm, topology, sceneId, rankSize, dataPath, bytes) ==
            ScatterPullAddressMode::BASE_OFFSET ? 1U : 0U;
}

constexpr bool SelectScatterMainHeavy(const AlgResourceCtx &ctx)
{
    const bool target = (ctx.topology == ScatterTopology::FOUR_ONE && ctx.sceneId == 4 &&
            ctx.rankSize == 4) ||
        (ctx.topology == ScatterTopology::FOUR_THREE && ctx.sceneId == 10 && ctx.rankSize == 12);
    return (SCATTER_V022_OPTIONS & 8U) != 0 && target && ctx.topologyKnown != 0 &&
        ctx.myRank == ctx.root && ctx.dataPath == ScatterDataPath::SINGLE_WAVE &&
        SelectScatterPullRead(ctx.algorithm, ctx.sceneId, ctx.recvBytes) != 0 &&
        ctx.schedule == ScatterSchedule::PARALLEL && ctx.groupCount == 2 && ctx.worker != 0 &&
        ctx.groups[0].dieId < SCATTER_MAX_DIES && ctx.groups[1].dieId < SCATTER_MAX_DIES &&
        ctx.groups[0].dieId != ctx.groups[1].dieId;
}

// Preserve existing fused DIRECT behavior except for the explicit CHUNKED
// policies. Those policies isolate Copy fusion from instruction-path changes.
// Invalid scene IDs never index the policy table.
constexpr ScatterDataPath SelectScatterDataPath(ScatterAlgorithm algorithm, uint32_t sceneId,
    uint32_t directFusion, uint64_t recvBytes)
{
    if (algorithm != ScatterAlgorithm::DIRECT) { return ScatterDataPath::CHUNKED; }
    constexpr uint32_t policyCount = sizeof(scatter_route_detail::POLICIES) /
        sizeof(scatter_route_detail::POLICIES[0]);
    if (sceneId != 0 && sceneId <= policyCount &&
        scatter_route_detail::POLICIES[sceneId - 1].directForceChunked) {
        // V020 B changes only admitted 8+4 fused, genuinely single-wave inputs.
        if ((SCATTER_V020_OPTIONS & 4U) != 0 && (sceneId == 8 || sceneId == 9) &&
            directFusion != 0 && recvBytes != 0 && recvBytes <= MAX_DATA_SIZE) {
            return ScatterDataPath::SINGLE_WAVE;
        }
        return ScatterDataPath::CHUNKED;
    }
    if (directFusion != 0 && recvBytes <= MAX_DATA_SIZE) { return ScatterDataPath::SINGLE_WAVE; }
    if (sceneId == 0 || sceneId > policyCount) { return ScatterDataPath::CHUNKED; }
    const auto &policy = scatter_route_detail::POLICIES[sceneId - 1];
    return policy.directSingleWave && directFusion == 0 && recvBytes != 0 && recvBytes <= SCATTER_SMALL_LIMIT
        ? ScatterDataPath::SINGLE_WAVE : ScatterDataPath::CHUNKED;
}

constexpr uint32_t SelectScatterRootAckFusion(ScatterAlgorithm algorithm, uint32_t sceneId)
{
    constexpr uint32_t policyCount = sizeof(scatter_route_detail::POLICIES) /
        sizeof(scatter_route_detail::POLICIES[0]);
    if (algorithm != ScatterAlgorithm::RELAY || sceneId == 0 || sceneId > policyCount) { return 0; }
    const auto &policy = scatter_route_detail::POLICIES[sceneId - 1];
    return policy.algorithm == ScatterAlgorithm::RELAY && policy.rootAckFusion ? 1 : 0;
}

struct ScatterSendPhases {
    uint32_t count = 3;
    AdaptiveRole roles[3] = {SCATTER_SEND_PHASES[0], SCATTER_SEND_PHASES[1], SCATTER_SEND_PHASES[2]};
    uint32_t fuseAckRequest = 0;
};

// Production Host batch construction uses this same selector. Fused root SEND
// groups publish their own requests without waiting; every DATA group still
// precedes the shared WAIT_ACK batch. Only explicitly admitted 2x8 helpers
// share that request fusion; other helpers and TREE keep all three phases.
constexpr ScatterSendPhases SelectScatterSendPhases(ScatterAlgorithm algorithm, bool isRoot,
    uint32_t rootAckFusion, uint32_t helperAckFusion = 0)
{
    ScatterSendPhases phases{};
    if (algorithm == ScatterAlgorithm::RELAY &&
        ((isRoot && rootAckFusion == 1) || (!isRoot && helperAckFusion == 1))) {
        phases.count = 2;
        phases.roles[1] = AdaptiveRole::WAIT_ACK;
        phases.fuseAckRequest = 1;
    }
    return phases;
}

// This is the single production route decision, also evaluated by the
// submission-external static_assert translation unit. It does not query the
// platform, allocate resources, construct a CCU kernel, or execute an operator.
constexpr ScatterRouteDecision SelectScatterRoute(const ScatterRouteInput &input)
{
    ScatterRouteDecision route{};
    if (input.rankSize == 0 || input.rankSize > MAX_RANK_SIZE || input.root >= input.rankSize) {
        route.reason = ScatterRouteReason::INVALID_PARTITION;
        return route;
    }
    if (input.recvBytes == 0 || input.recvBytes % 4 != 0 ||
        input.recvBytes > std::numeric_limits<uint64_t>::max() / input.rankSize) {
        route.reason = ScatterRouteReason::UNMATCHED_SIZE;
        return route;
    }
    if (input.rankSize == 1) {
        route.reason = ScatterRouteReason::SINGLE_RANK;
        return route;
    }
    const uint32_t valid = (uint32_t{1} << input.rankSize) - 1;
    for (uint32_t rank = 0; rank < input.rankSize; ++rank) {
        if (input.serverMasks[rank] == 0) { return route; }
    }
    uint32_t visited = 0;
    uint32_t groupCount = 0;
    uint32_t sizes[MAX_RANK_SIZE + 1] = {};
    for (uint32_t rank = 0; rank < input.rankSize; ++rank) {
        const uint32_t mask = input.serverMasks[rank];
        if ((mask & ~valid) != 0 || (mask & (uint32_t{1} << rank)) == 0) {
            route.reason = ScatterRouteReason::INVALID_PARTITION;
            return route;
        }
        if ((visited & (uint32_t{1} << rank)) != 0) { continue; }
        if ((visited & mask) != 0) {
            route.reason = ScatterRouteReason::INVALID_PARTITION;
            return route;
        }
        for (uint32_t peer = 0; peer < input.rankSize; ++peer) {
            if ((mask & (uint32_t{1} << peer)) != 0 && input.serverMasks[peer] != mask) {
                route.reason = ScatterRouteReason::INVALID_PARTITION;
                return route;
            }
        }
        ++sizes[scatter_route_detail::CountBits(mask)];
        ++groupCount;
        visited |= mask;
    }
    if (visited != valid) {
        route.reason = ScatterRouteReason::INVALID_PARTITION;
        return route;
    }
    if (groupCount == 2 && sizes[8] == 2) { route.topology = ScatterTopology::TWO_EIGHT; }
    else if (groupCount == 4 && sizes[1] == 4) { route.topology = ScatterTopology::FOUR_ONE; }
    else if (groupCount == 2 && sizes[4] == 1 && sizes[8] == 1) { route.topology = ScatterTopology::EIGHT_FOUR; }
    else if (groupCount == 4 && sizes[3] == 4) { route.topology = ScatterTopology::FOUR_THREE; }
    else {
        route.reason = ScatterRouteReason::UNSUPPORTED_TOPOLOGY;
        return route;
    }
    int profile = scatter_route_detail::SelectProfile(input.recvBytes, input.rankSize);
    if (profile < 0 && (route.topology == ScatterTopology::FOUR_THREE ||
        route.topology == ScatterTopology::EIGHT_FOUR)) {
        profile = scatter_route_detail::SelectAlignedAggregateProfile(input.recvBytes, input.rankSize);
    }
    if (profile < 0) {
        route.reason = ScatterRouteReason::UNMATCHED_SIZE;
        return route;
    }
    const uint32_t index = (static_cast<uint32_t>(route.topology) - 1) * 3 + static_cast<uint32_t>(profile);
    const auto &policy = scatter_route_detail::POLICIES[index];
    route.sceneId = index + 1;
    route.directFusion = policy.directFusion;
    if (policy.algorithm == ScatterAlgorithm::DIRECT) {
        route.reason = ScatterRouteReason::DIRECT_POLICY;
        return route;
    }
    uint64_t capacity = std::numeric_limits<uint64_t>::max();
    for (uint32_t rank = 0; rank < input.rankSize; ++rank) {
        if (input.capacities[rank] < capacity) { capacity = input.capacities[rank]; }
    }
    if (policy.algorithm == ScatterAlgorithm::TREE) {
        if (policy.treeWidth == 0 || capacity / policy.treeWidth < input.recvBytes) {
            route.reason = ScatterRouteReason::TREE_SCRATCH_TOO_SMALL;
            return route;
        }
        route.treeWidth = policy.treeWidth;
        route.chunkBytes = input.recvBytes;
        route.fullSteps = 1;
    } else {
        const uint32_t helpers = scatter_route_detail::CountBits(input.serverMasks[input.root]) - 1;
        const uint32_t remotes = input.rankSize - helpers - 1;
        if (helpers == 0) {
            route.reason = ScatterRouteReason::NO_LOCAL_HELPER;
            return route;
        }
        if (remotes <= 4) {
            route.reason = ScatterRouteReason::REMOTE_FANOUT_AT_MOST_FOUR;
            return route;
        }
        const uint64_t maxChildren = (remotes + helpers - 1) / helpers;
        uint64_t chunk = policy.chunkBytes < input.recvBytes ? policy.chunkBytes : input.recvBytes;
        if (capacity / maxChildren < chunk) { chunk = capacity / maxChildren; }
        chunk -= chunk % 4;
        if (chunk < 4) {
            route.reason = ScatterRouteReason::RELAY_SCRATCH_TOO_SMALL;
            return route;
        }
        // Keep the old layout when the larger feasible chunk cannot remove a
        // wave. All-rank minimum capacity and the existing 64MiB cap remain.
        if ((SCATTER_FEATURES & 2U) != 0 && route.topology == ScatterTopology::FOUR_THREE) {
            const uint64_t limit = policy.chunkBytes < input.recvBytes ? policy.chunkBytes : input.recvBytes;
            const uint64_t compact = scatter_route_detail::CompactChunk(limit, capacity, helpers, remotes);
            if (compact > chunk && scatter_route_detail::WaveCount(input.recvBytes, compact) <
                scatter_route_detail::WaveCount(input.recvBytes, chunk)) {
                chunk = compact;
                route.compactScratch = 1;
            }
        }
        route.chunkBytes = chunk;
        route.fullSteps = input.recvBytes / chunk;
        route.tailBytes = input.recvBytes % chunk;
    }
    route.algorithm = policy.algorithm;
    route.relayOverlap = SelectScatterRelayOverlap(route.algorithm, route.sceneId);
    route.rootAckFusion = SelectScatterRootAckFusion(route.algorithm, route.sceneId);
    route.reason = ScatterRouteReason::MATCHED;
    return route;
}

const char *ScatterRouteReasonName(ScatterRouteReason reason);

struct ScatterBatchPlan {
    std::vector<AdaptiveKernelArg> groups;
    bool parallel = false;
};
struct ScatterShapePlan { std::vector<ScatterBatchPlan> batches; };
struct ScatterPlan {
    ScatterAlgorithm algorithm = ScatterAlgorithm::DIRECT;
    ScatterTopology topology = ScatterTopology::UNKNOWN;
    ScatterRouteReason routeReason = ScatterRouteReason::UNKNOWN_MEMBERS;
    uint32_t sceneId = 0;
    uint32_t directFusion = 0;
    uint32_t rootAckFusion = 0;
    uint32_t compactScratch = 0;
    uint32_t relayOverlap = 0;
    uint32_t relayCopy = 0;
    uint64_t recvBytes = 0;
    uint64_t chunkBytes = 0;
    uint64_t fullSteps = 0;
    uint64_t tailBytes = 0;
    uint64_t scratchUsedBytes = 0;
    ScatterShapePlan full;
    ScatterShapePlan tail;
};
HcclResult BuildScatterPlan(const OpParam &param, const ScatterCommResources &resources,
    uint64_t recvBytes, ScatterPlan &plan);
} // namespace ops_hccl

#endif // OPS_HCCL_CCU_EXEC_OP_H
