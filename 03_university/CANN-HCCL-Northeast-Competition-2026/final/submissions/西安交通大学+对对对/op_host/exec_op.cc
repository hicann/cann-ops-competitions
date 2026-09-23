/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <limits>
#include <algorithm>
#include <array>
#include <vector>

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>

#include "log.h"
#include "custom.h"
#include "exec_op.h"
#include "ccu_kernel.h"

namespace ops_hccl {
namespace {
bool ValidAddressRange(const void *ptr, uint64_t bytes)
{
    const uint64_t addr = reinterpret_cast<uintptr_t>(ptr);
    return bytes <= std::numeric_limits<uintptr_t>::max() - addr;
}

uint32_t CountPeers(uint32_t mask)
{
    uint32_t count = 0;
    while (mask != 0) {
        mask &= mask - 1;
        ++count;
    }
    return count;
}

struct PlanEdge {
    uint32_t from = 0;
    uint32_t to = 0;
    ScatterTransfer transfer{};
};

std::vector<std::vector<uint32_t>> PartitionMembers(const ScatterRouteInput &input)
{
    // SelectScatterRoute already checked that every rank reports the same
    // disjoint partition. Only expand those masks into ordered member lists.
    std::vector<std::vector<uint32_t>> servers;
    uint32_t visited = 0;
    for (uint32_t rank = 0; rank < input.rankSize; ++rank) {
        if ((visited & (uint32_t{1} << rank)) != 0) { continue; }
        const uint32_t mask = input.serverMasks[rank];
        std::vector<uint32_t> members;
        for (uint32_t peer = 0; peer < input.rankSize; ++peer) {
            if ((mask & (uint32_t{1} << peer)) != 0) { members.push_back(peer); }
        }
        servers.push_back(std::move(members));
        visited |= mask;
    }
    return servers;
}

uint16_t ReadyMask(ScatterBuffer destination)
{
    return destination == ScatterBuffer::OUTPUT ? 3U : 12U;
}

void AddEdge(std::vector<PlanEdge> &edges, uint32_t from, uint32_t to, ScatterBuffer source,
    ScatterBuffer destination, uint64_t sourceOffset, uint64_t destinationOffset, uint64_t bytes)
{
    if (bytes == 0) { return; }
    PlanEdge edge{};
    edge.from = from;
    edge.to = to;
    edge.transfer.source = source;
    edge.transfer.destination = destination;
    edge.transfer.sourceOffset = sourceOffset;
    edge.transfer.destinationOffset = destinationOffset;
    edge.transfer.bytes = bytes;
    edges.push_back(edge);
}

std::vector<PlanEdge> EdgesFor(const std::vector<PlanEdge> &edges, uint32_t rank, bool outgoing)
{
    std::vector<PlanEdge> result;
    for (const auto &edge : edges) {
        if ((outgoing ? edge.from : edge.to) == rank) { result.push_back(edge); }
    }
    return result;
}

HcclResult MakeSendBatch(const std::vector<PlanEdge> &edges, const ScatterCommResources &resources,
    ScatterBatchPlan &batch, bool tailFirst = false)
{
    std::vector<uint32_t> peers;
    for (const auto &edge : edges) {
        if (std::find(peers.begin(), peers.end(), edge.to) == peers.end()) { peers.push_back(edge.to); }
    }
    const bool known = std::all_of(peers.begin(), peers.end(), [&resources](uint32_t peer) {
        return resources.dies[peer] < SCATTER_MAX_DIES;
    });
    std::vector<uint32_t> groupDies;
    for (const uint32_t peer : peers) {
        if (resources.channels[peer] == 0) { return HCCL_E_PARA; }
        size_t group = batch.groups.size();
        if (known) {
            const auto found = std::find(groupDies.begin(), groupDies.end(), resources.dies[peer]);
            if (found != groupDies.end()) { group = static_cast<size_t>(found - groupDies.begin()); }
        }
        if (group == batch.groups.size()) {
            batch.groups.emplace_back();
            groupDies.push_back(resources.dies[peer]);
        }
        auto &arg = batch.groups[group];
        const uint32_t index = arg.peerCount++;
        if (arg.peerCount >= MAX_RANK_SIZE) { return HCCL_E_PARA; }
        arg.channels[index] = resources.channels[peer];
        arg.peers[index] = peer;
        const uint32_t firstTransfer = arg.transferCount;
        for (const auto &edge : edges) {
            if (edge.to != peer) { continue; }
            if (arg.transferCount == SCATTER_MAX_TRANSFERS) { return HCCL_E_PARA; }
            auto &transfer = arg.transfers[arg.transferCount++];
            transfer = edge.transfer;
            transfer.channelIndex = index;
            arg.readyMasks[index] |= ReadyMask(transfer.destination);
        }
        if (tailFirst) {
            // Preserve peer/die order and each tail's original order. Only
            // move this peer's forwarding payload ahead of its own output;
            // regrouping the original edges would also change scheduling.
            std::stable_partition(arg.transfers + firstTransfer, arg.transfers + arg.transferCount,
                [](const ScatterTransfer &transfer) {
                    return transfer.destination == ScatterBuffer::SCRATCH;
                });
        }
    }
    batch.parallel = known && batch.groups.size() == 2 && resources.worker != 0;
    return HCCL_SUCCESS;
}

HcclResult AddSendAndDrain(const std::vector<PlanEdge> &edges, const ScatterCommResources &resources,
    ScatterAlgorithm algorithm, bool isRoot, uint32_t rootAckFusion, ScatterShapePlan &shape,
    uint32_t relayOverlap = 0, uint32_t helperAckFusion = 0)
{
    if (edges.empty()) { return HCCL_SUCCESS; }
    ScatterBatchPlan send;
    // Overlap mode 1 preserves 4x3; mode 2 admits 2x8 whole-helper release.
    // Only root-to-helper payload order changes; helper sends keep edge order.
    const bool tailFirst = isRoot && algorithm == ScatterAlgorithm::RELAY && relayOverlap != 0;
    CHK_RET(MakeSendBatch(edges, resources, send, tailFirst));
    // Finish every DATA group before any WAIT_ACK group. Fused root sends
    // publish each request after its peer's DONE, so no second request batch
    // may be emitted. Unfused sends retain the separate requests; 2x8 helpers
    // opt in separately, without moving any WAIT_ACK ahead of DATA groups.
    const auto phases = SelectScatterSendPhases(algorithm, isRoot, rootAckFusion, helperAckFusion);
    for (uint32_t phase = 0; phase < phases.count; ++phase) {
        const AdaptiveRole role = phases.roles[phase];
        ScatterBatchPlan batch = send;
        for (auto &arg : batch.groups) {
            arg.role = role;
            arg.fuseAckRequest = role == AdaptiveRole::SEND_DATA ? phases.fuseAckRequest : 0;
            if (role == AdaptiveRole::SEND_DATA && isRoot && relayOverlap != 0) {
                uint32_t helperMask = 0;
                for (uint32_t i = 0; i < arg.transferCount; ++i) {
                    if (arg.transfers[i].destination == ScatterBuffer::SCRATCH) {
                        arg.earlyTail = relayOverlap == 1 && (SCATTER_OPTIMIZATIONS & 8U) != 0 ? 2 : 1;
                        helperMask |= 1U << arg.transfers[i].channelIndex;
                    }
                }
                // MakeSendBatch already groups channels by local die. A single
                // helper has no cross-helper blocking to remove. Unknown dies
                // remain singleton groups and therefore retain their old order.
                arg.interleaveTails = relayOverlap == 1 && (SCATTER_TUNING & 1U) != 0 && arg.earlyTail == 2 &&
                    (helperMask & (helperMask - 1U)) != 0 ? 1U : 0U;
            }
            if (role != AdaptiveRole::SEND_DATA) {
                arg.transferCount = 0;
                std::fill(std::begin(arg.transfers), std::end(arg.transfers), ScatterTransfer{});
                std::fill(std::begin(arg.readyMasks), std::end(arg.readyMasks), uint16_t{0});
            }
        }
        shape.batches.push_back(std::move(batch));
    }
    return HCCL_SUCCESS;
}

HcclResult AddReceive(const std::vector<PlanEdge> &incoming, const ScatterCommResources &resources,
    bool deferred, ScatterShapePlan &shape, AdaptiveRole role = AdaptiveRole::RECEIVE)
{
    if (incoming.empty()) { return HCCL_SUCCESS; }
    const uint32_t parent = incoming.front().from;
    if (resources.channels[parent] == 0) { return HCCL_E_PARA; }
    AdaptiveKernelArg arg{};
    arg.role = role;
    arg.peerCount = 1;
    arg.channels[0] = resources.channels[parent];
    arg.peers[0] = parent;
    arg.deferAck = deferred ? 1 : 0;
    for (const auto &edge : incoming) {
        if (edge.from != parent) { return HCCL_E_PARA; }
        arg.readyMasks[0] |= ReadyMask(edge.transfer.destination);
    }
    ScatterBatchPlan batch;
    batch.groups.push_back(arg);
    shape.batches.push_back(std::move(batch));
    return HCCL_SUCCESS;
}

HcclResult BuildOverlapShape(const OpParam &param, const ScatterCommResources &resources,
    const std::vector<PlanEdge> &first, const std::vector<PlanEdge> &second,
    uint32_t rootAckFusion, ScatterShapePlan &shape, uint32_t relayOverlap = 1, uint32_t helperAckFusion = 0)
{
    if (param.myRank == param.root) {
        return AddSendAndDrain(EdgesFor(first, param.myRank, true), resources,
            ScatterAlgorithm::RELAY, true, rootAckFusion, shape, relayOverlap);
    }
    const auto incoming = EdgesFor(first, param.myRank, false);
    const auto forwarding = EdgesFor(second, param.myRank, true);
    const auto tail = EdgesFor(second, param.myRank, false);
    if (incoming.empty()) { return HCCL_E_PARA; }
    if (!forwarding.empty()) {
        // Publish both output/scratch, then consume only TAIL_COMPLETE. The
        // upstream channel is never mixed with downstream dies in one kernel.
        const bool fineProtocol = relayOverlap == 1 && (SCATTER_OPTIMIZATIONS & 8U) != 0;
        const uint32_t parent = incoming.front().from;
        const uint32_t upstreamDie = resources.dies[parent];
        const bool fineForward = fineProtocol && forwarding.size() > 1 && upstreamDie < SCATTER_MAX_DIES &&
            std::all_of(forwarding.begin(), forwarding.end(), [&resources, upstreamDie](const PlanEdge &edge) {
                return resources.dies[edge.to] == upstreamDie;
            });
        CHK_RET(AddReceive(incoming, resources, false, shape,
            fineForward ? AdaptiveRole::PUBLISH_READY : AdaptiveRole::RECEIVE_TAIL));
        uint16_t tailMask = 0;
        uint32_t tailCount = 0;
        for (const auto &edge : incoming) {
            if (edge.transfer.destination != ScatterBuffer::SCRATCH) { continue; }
            if (++tailCount > 15) { return HCCL_E_PARA; }
            tailMask |= static_cast<uint16_t>(1U << tailCount);
        }
        if (fineProtocol) { shape.batches.back().groups[0].tailWaitMask = tailMask; }
        const size_t sendBegin = shape.batches.size();
        CHK_RET(AddSendAndDrain(forwarding, resources, ScatterAlgorithm::RELAY, false, 0, shape, 0, helperAckFusion));
        if (fineForward) {
            // This admission leaves a single known-die send group. Publish
            // addresses in the existing receive kernel, then forward each
            // child as its own tail arrives, within the existing send kernel.
            for (auto &arg : shape.batches[sendBegin].groups) {
                arg.tailChannel = resources.channels[parent];
                for (uint32_t i = 0; i < arg.transferCount; ++i) {
                    uint32_t ordinal = 0;
                    for (const auto &edge : incoming) {
                        if (edge.transfer.destination != ScatterBuffer::SCRATCH) { continue; }
                        ++ordinal;
                        if (edge.transfer.destinationOffset == arg.transfers[i].sourceOffset &&
                            edge.transfer.bytes == arg.transfers[i].bytes) {
                            arg.tailWaitMasks[i] = static_cast<uint16_t>(1U << ordinal);
                        }
                    }
                    if (arg.tailWaitMasks[i] == 0) { return HCCL_E_PARA; }
                }
            }
        }
    } else if (!tail.empty()) {
        // Return after publishing the root destination, without waiting for
        // its prefix. The helper can now deliver concurrently to a disjoint
        // output interval. Separate kernels also work with unknown/different dies.
        CHK_RET(AddReceive(incoming, resources, false, shape, AdaptiveRole::PUBLISH_READY));
        CHK_RET(AddReceive(tail, resources, false, shape));
    } else {
        return AddReceive(incoming, resources, false, shape);
    }
    // All downstream reads/writes and ACKs precede this final root ACK. This
    // drains root output completion too, and protects single-wave scratch reuse.
    AdaptiveKernelArg finish{};
    finish.role = AdaptiveRole::FINISH_RECEIVE;
    finish.peerCount = 1;
    finish.peers[0] = incoming.front().from;
    finish.channels[0] = resources.channels[finish.peers[0]];
    ScatterBatchPlan batch;
    batch.groups.push_back(finish);
    shape.batches.push_back(std::move(batch));
    return shape.batches.size() <= SCATTER_MAX_BATCHES ? HCCL_SUCCESS : HCCL_E_PARA;
}

HcclResult BuildLocalShape(const OpParam &param, const ScatterCommResources &resources,
    const std::vector<PlanEdge> &first, const std::vector<PlanEdge> &second,
    const std::array<ScatterTransfer, MAX_RANK_SIZE> &localCopies, ScatterAlgorithm algorithm,
    uint32_t rootAckFusion, ScatterShapePlan &shape, uint32_t helperAckFusion = 0)
{
    if (param.myRank == param.root) {
        return AddSendAndDrain(EdgesFor(first, param.myRank, true), resources,
            algorithm, true, rootAckFusion, shape);
    }
    const auto incoming = EdgesFor(first, param.myRank, false);
    const auto forwarding = EdgesFor(second, param.myRank, true);
    CHK_RET(AddReceive(incoming, resources, !forwarding.empty(), shape));
    if (forwarding.empty()) {
        return AddReceive(EdgesFor(second, param.myRank, false), resources, false, shape);
    }
    if (incoming.empty()) { return HCCL_E_PARA; }
    CHK_RET(AddSendAndDrain(forwarding, resources, algorithm, false, rootAckFusion, shape, 0, helperAckFusion));
    if (localCopies[param.myRank].bytes != 0) {
        AdaptiveKernelArg copy{};
        copy.role = AdaptiveRole::LOCAL_COPY;
        copy.transferCount = 1;
        copy.transfers[0] = localCopies[param.myRank];
        ScatterBatchPlan batch;
        batch.groups.push_back(copy);
        shape.batches.push_back(std::move(batch));
    }
    AdaptiveKernelArg ack{};
    ack.role = AdaptiveRole::ACK_ONLY;
    ack.peerCount = 1;
    ack.peers[0] = incoming.front().from;
    ack.channels[0] = resources.channels[ack.peers[0]];
    ScatterBatchPlan batch;
    batch.groups.push_back(ack);
    shape.batches.push_back(std::move(batch));
    return shape.batches.size() <= SCATTER_MAX_BATCHES ? HCCL_SUCCESS : HCCL_E_PARA;
}

HcclResult BuildTree(const OpParam &param, const ScatterCommResources &resources,
    const std::vector<std::vector<uint32_t>> &servers, uint32_t width, uint64_t bytes,
    ScatterShapePlan &shape)
{
    std::vector<PlanEdge> first;
    std::vector<PlanEdge> second;
    std::array<ScatterTransfer, MAX_RANK_SIZE> copies{};
    for (const auto &server : servers) {
        for (size_t begin = 0; begin < server.size(); begin += width) {
            const size_t end = std::min(begin + width, server.size());
            auto groupBegin = server.begin() + begin;
            auto groupEnd = server.begin() + end;
            const bool hasRoot = std::find(groupBegin, groupEnd, param.root) != groupEnd;
            const uint32_t leader = hasRoot ? param.root : server[begin];
            const bool forward = !hasRoot && end - begin > 1;
            for (size_t i = begin; i < end; ++i) {
                const uint32_t rank = server[i];
                if (rank == param.root) { continue; }
                const uint64_t slot = (i - begin) * bytes;
                AddEdge(first, param.root, forward ? leader : rank, ScatterBuffer::INPUT,
                    forward ? ScatterBuffer::SCRATCH : ScatterBuffer::OUTPUT,
                    rank * bytes, forward ? slot : 0, bytes);
                if (!forward) { continue; }
                if (rank != leader) {
                    AddEdge(second, leader, rank, ScatterBuffer::SCRATCH, ScatterBuffer::OUTPUT, slot, 0, bytes);
                } else {
                    copies[leader].source = ScatterBuffer::SCRATCH;
                    copies[leader].destination = ScatterBuffer::OUTPUT;
                    copies[leader].sourceOffset = slot;
                    copies[leader].bytes = bytes;
                }
            }
        }
    }
    return BuildLocalShape(param, resources, first, second, copies, ScatterAlgorithm::TREE, 0, shape);
}

HcclResult BuildRelayWave(const OpParam &param, const ScatterCommResources &resources,
    const std::vector<uint32_t> &helpers, const std::vector<uint32_t> &remotes,
    uint64_t stride, uint64_t chunk, uint64_t wave, uint32_t rootAckFusion, ScatterShapePlan &shape,
    uint32_t compactScratch, uint32_t relayOverlap, uint32_t helperAckFusion = 0)
{
    std::vector<PlanEdge> first;
    std::vector<PlanEdge> second;
    std::array<ScatterTransfer, MAX_RANK_SIZE> copies{};
    std::vector<uint32_t> childCounts(helpers.size());
    std::vector<uint64_t> scratchEnds(helpers.size());
    for (size_t i = 0; i < remotes.size(); ++i) { ++childCounts[i % helpers.size()]; }
    for (const uint32_t helper : helpers) {
        AddEdge(first, param.root, helper, ScatterBuffer::INPUT, ScatterBuffer::OUTPUT,
            helper * stride, 0, wave);
    }
    for (size_t i = 0; i < remotes.size(); ++i) {
        const size_t helperIndex = i % helpers.size();
        const uint32_t helper = helpers[helperIndex];
        const uint32_t remote = remotes[i];
        // wave <= 64MiB and r <= 15, so this integer product cannot overflow.
        // Balance helper's own B plus forwarded bytes against root's 4b Clos.
        const uint64_t tail = scatter_route_detail::RelayTailBytes(wave,
            static_cast<uint32_t>(helpers.size()), static_cast<uint32_t>(remotes.size()), childCounts[helperIndex]);
        const uint64_t prefix = wave - tail;
        const uint64_t slot = compactScratch != 0 ? scratchEnds[helperIndex] : (i / helpers.size()) * chunk;
        scratchEnds[helperIndex] += tail;
        AddEdge(first, param.root, remote, ScatterBuffer::INPUT, ScatterBuffer::OUTPUT,
            remote * stride, 0, prefix);
        AddEdge(first, param.root, helper, ScatterBuffer::INPUT, ScatterBuffer::SCRATCH,
            remote * stride + prefix, slot, tail);
        AddEdge(second, helper, remote, ScatterBuffer::SCRATCH, ScatterBuffer::OUTPUT, slot, prefix, tail);
    }
    if (relayOverlap != 0) {
        return BuildOverlapShape(param, resources, first, second, rootAckFusion, shape, relayOverlap, helperAckFusion);
    }
    return BuildLocalShape(param, resources, first, second, copies, ScatterAlgorithm::RELAY,
        rootAckFusion, shape, helperAckFusion);
}

bool ValidShape(const ScatterShape &shape, ThreadHandle worker)
{
    if (shape.batchCount == 0 || shape.batchCount > SCATTER_MAX_BATCHES) { return false; }
    for (uint32_t i = 0; i < shape.batchCount; ++i) {
        const auto &batch = shape.batches[i];
        if (batch.kernelCount == 0 || batch.kernelCount >= MAX_RANK_SIZE || batch.parallel > 1 ||
            (batch.parallel != 0 && (batch.kernelCount != 2 || worker == 0))) { return false; }
        for (uint32_t j = 0; j < batch.kernelCount; ++j) {
            if (batch.kernels[j] == 0) { return false; }
        }
    }
    return true;
}

HcclResult LoadContext(const OpParam &param, uint64_t recvBytes, AlgResourceCtx &ctx)
{
    CHK_PTR_NULL(param.resCtx);
    if (param.ctxSize != sizeof(ctx)) {
        return HCCL_E_PARA;
    }
    std::memcpy(&ctx, param.resCtx, sizeof(ctx));
    if (ctx.magic != SCATTER_CTX_MAGIC || ctx.version != SCATTER_CTX_VERSION || ctx.root != param.root ||
        ctx.myRank != param.myRank || ctx.rankSize != param.rankSize ||
        ctx.recvBytes != recvBytes || ctx.sceneId > 12 || ctx.directFusion > 1 || ctx.rootAckFusion > 1 ||
        ctx.compactScratch > 1 || ctx.relayOverlap != SelectScatterRelayOverlap(ctx.algorithm, ctx.sceneId) ||
        (ctx.compactScratch != 0 && ((SCATTER_FEATURES & 2U) == 0 || ctx.algorithm != ScatterAlgorithm::RELAY ||
            (ctx.sceneId != 11 && ctx.sceneId != 12))) ||
        static_cast<uint32_t>(ctx.routeReason) > static_cast<uint32_t>(ScatterRouteReason::SINGLE_RANK) ||
        static_cast<uint32_t>(ctx.topology) > static_cast<uint32_t>(ScatterTopology::FOUR_THREE) ||
        ctx.groupCount >= MAX_RANK_SIZE || ctx.topologyKnown > 1 ||
        (ctx.dataPath != ScatterDataPath::SINGLE_WAVE && ctx.dataPath != ScatterDataPath::CHUNKED) ||
        (ctx.schedule != ScatterSchedule::PARALLEL && ctx.schedule != ScatterSchedule::SEQUENTIAL)) {
        return HCCL_E_PARA;
    }

    if (ctx.rootAckFusion != SelectScatterRootAckFusion(ctx.algorithm, ctx.sceneId) ||
        ctx.relayCopy != SelectScatterRelayCopy(ctx.algorithm, ctx.sceneId, ctx.fullSteps, ctx.tailBytes) ||
        ctx.dataPath != SelectScatterDataPath(ctx.algorithm, ctx.sceneId, ctx.directFusion, recvBytes)) {
        return HCCL_E_PARA;
    }
    if (ctx.sceneId == 0) {
        if (ctx.directFusion != 0) { return HCCL_E_PARA; }
    } else {
        const auto &policy = scatter_route_detail::POLICIES[ctx.sceneId - 1];
        if (ctx.directFusion != policy.directFusion ||
            (ctx.algorithm != ScatterAlgorithm::DIRECT && ctx.algorithm != policy.algorithm)) {
            return HCCL_E_PARA;
        }
    }

    if (ctx.algorithm != ScatterAlgorithm::DIRECT) {
        if ((ctx.algorithm != ScatterAlgorithm::TREE && ctx.algorithm != ScatterAlgorithm::RELAY) ||
            ctx.routeReason != ScatterRouteReason::MATCHED || ctx.sceneId == 0 ||
            ctx.chunkBytes == 0 || ctx.chunkBytes > MAX_DATA_SIZE ||
            ctx.chunkBytes % 4 != 0 || ctx.fullSteps != recvBytes / ctx.chunkBytes ||
            ctx.tailBytes != recvBytes % ctx.chunkBytes || ctx.fullSteps == 0 ||
            ctx.scratchAddress == 0 || ctx.scratchBytes == 0 ||
            ctx.worker == param.cpuThread || !ValidShape(ctx.fullShape, ctx.worker) ||
            (ctx.tailBytes != 0 && !ValidShape(ctx.tailShape, ctx.worker)) ||
            (param.myRank == param.root ? ctx.copyKernel == 0 : ctx.copyKernel != 0)) {
            return HCCL_E_PARA;
        }
        return HCCL_SUCCESS;
    }

    const uint32_t validRanks = (uint32_t{1} << param.rankSize) - 1;
    const bool isRoot = param.myRank == param.root;
    const uint32_t expected = isRoot ? validRanks & ~(uint32_t{1} << param.root)
                                     : uint32_t{1} << param.root;
    if ((ctx.localRankMask & ~validRanks) != 0 ||
        (ctx.localRankMask & (uint32_t{1} << param.myRank)) == 0) {
        return HCCL_E_PARA;
    }
    uint32_t peers = 0;
    uint32_t dies = 0;
    for (uint32_t i = 0; i < ctx.groupCount; ++i) {
        const auto &group = ctx.groups[i];
        const uint32_t count = CountPeers(group.peerMask);
        if (group.kernel == 0 || group.peerMask == 0 || (group.peerMask & ~expected) != 0 ||
            (group.peerMask & peers) != 0 || group.localPeerCount > count || group.remotePeerCount > count ||
            group.localPeerCount + group.remotePeerCount > count) {
            return HCCL_E_PARA;
        }
        peers |= group.peerMask;
        if (ctx.topologyKnown != 0 && isRoot &&
            (group.localPeerCount != CountPeers(group.peerMask & ctx.localRankMask) ||
             group.remotePeerCount != CountPeers(group.peerMask & ~ctx.localRankMask))) {
            return HCCL_E_PARA;
        }
        if (ctx.schedule == ScatterSchedule::PARALLEL && isRoot) {
            if (group.dieId >= SCATTER_MAX_DIES || (dies & (uint32_t{1} << group.dieId)) != 0) {
                return HCCL_E_PARA;
            }
            dies |= uint32_t{1} << group.dieId;
        }
    }
    if (peers != expected || (!isRoot && (ctx.groupCount != 1 || ctx.copyKernel != 0 || ctx.worker != 0)) ||
        (isRoot && ctx.copyKernel == 0) ||
        (ctx.schedule == ScatterSchedule::PARALLEL && isRoot && ctx.groupCount > SCATTER_MAX_DIES)) {
        return HCCL_E_PARA;
    }
    const bool needsWorker = isRoot && ctx.schedule == ScatterSchedule::PARALLEL && ctx.groupCount == 2;
    if ((needsWorker && (ctx.worker == 0 || ctx.worker == param.cpuThread)) ||
        (!needsWorker && ctx.worker != 0)) {
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchGroups(const OpParam &param, const AlgResourceCtx &ctx, const uint64_t *args)
{
    if (ctx.schedule == ScatterSchedule::PARALLEL && ctx.groupCount == 2) {
        // Only admitted small Pull scenes put the higher-priority group on
        // main. Keep logical copyOwner and channel/die bindings with each group.
        // START still orders worker work after preceding user work; DONE joins
        // its complete kernel, including any Copy, before subsequent user work.
        const uint32_t workerGroup = SelectScatterMainHeavy(ctx) ? 1U : 0U;
        const uint32_t mainGroup = 1U - workerGroup;
        CHK_RET(HcommThreadNotifyRecordOnThread(param.cpuThread, ctx.worker, SCATTER_THREAD_START));
        CHK_RET(HcommThreadNotifyWaitOnThread(ctx.worker, SCATTER_THREAD_START, CUSTOM_TIMEOUT));
        CHK_RET_CCU(HcommCcuKernelLaunch(ctx.worker, ctx.groups[workerGroup].kernel, args, SCATTER_TASK_ARG_COUNT));
        CHK_RET(HcommThreadNotifyRecordOnThread(ctx.worker, param.cpuThread, SCATTER_THREAD_DONE));
        CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread, ctx.groups[mainGroup].kernel, args, SCATTER_TASK_ARG_COUNT));
        CHK_RET(HcommThreadNotifyWaitOnThread(param.cpuThread, SCATTER_THREAD_DONE, CUSTOM_TIMEOUT));
        return HCCL_SUCCESS;
    }
    // Unknown die or unavailable worker: same V002 protocol on one thread.
    // Each kernel still loops over its data without further Host launches.
    for (uint32_t i = 0; i < ctx.groupCount; ++i) {
        CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread, ctx.groups[i].kernel, args, SCATTER_TASK_ARG_COUNT));
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchShape(const OpParam &param, const AlgResourceCtx &ctx,
    const ScatterShape &shape, const uint64_t *args)
{
    for (uint32_t phase = 0; phase < shape.batchCount; ++phase) {
        const auto &batch = shape.batches[phase];
        if (batch.parallel != 0) {
            CHK_RET(HcommThreadNotifyRecordOnThread(param.cpuThread, ctx.worker, SCATTER_THREAD_START));
            CHK_RET(HcommThreadNotifyWaitOnThread(ctx.worker, SCATTER_THREAD_START, CUSTOM_TIMEOUT));
            CHK_RET_CCU(HcommCcuKernelLaunch(ctx.worker, batch.kernels[0], args, ADAPTIVE_ARG_COUNT));
            CHK_RET(HcommThreadNotifyRecordOnThread(ctx.worker, param.cpuThread, SCATTER_THREAD_DONE));
            CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread, batch.kernels[1], args, ADAPTIVE_ARG_COUNT));
            CHK_RET(HcommThreadNotifyWaitOnThread(param.cpuThread, SCATTER_THREAD_DONE, CUSTOM_TIMEOUT));
        } else {
            for (uint32_t group = 0; group < batch.kernelCount; ++group) {
                CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread, batch.kernels[group], args, ADAPTIVE_ARG_COUNT));
            }
        }
        // All local groups join before resources of a later registration
        // round are reused. DATA, REQUEST_ACK and WAIT_ACK remain separate in fallback.
    }
    return HCCL_SUCCESS;
}

bool Overlaps(uint64_t first, uint64_t firstBytes, uint64_t second, uint64_t secondBytes)
{
    return firstBytes != 0 && secondBytes != 0 && first < second + secondBytes && second < first + firstBytes;
}

HcclResult ExecAdaptive(const OpParam &param, const AlgResourceCtx &ctx, uint64_t inputBytes,
    uint64_t output, uint64_t outputToken)
{
    const bool isRoot = param.myRank == param.root;
    const uint64_t input = isRoot ? reinterpret_cast<uintptr_t>(param.inputPtr) : 0;
    uint64_t inputToken = 0;
    uint64_t scratchToken = 0;
    if (isRoot) { CHK_RET_CCU(HcommCcuGetMemToken(input, inputBytes, &inputToken)); }
    if (ctx.scratchUsedBytes != 0) {
        if (ctx.scratchUsedBytes > ctx.scratchBytes ||
            ctx.scratchUsedBytes > std::numeric_limits<uintptr_t>::max() - ctx.scratchAddress ||
            Overlaps(ctx.scratchAddress, ctx.scratchUsedBytes, output, ctx.recvBytes) ||
            (isRoot && Overlaps(ctx.scratchAddress, ctx.scratchUsedBytes, input, inputBytes))) {
            return HCCL_E_PARA;
        }
        CHK_RET_CCU(HcommCcuGetMemToken(ctx.scratchAddress, ctx.scratchUsedBytes, &scratchToken));
    }
    // Slot 7 is the invocation's Copy flag; slots 8..12 remain padding.
    // Every role still loads all 13 contiguous SQE arguments.
    const bool fusedCopy = isRoot && ctx.relayCopy != 0 &&
        !Overlaps(output, ctx.recvBytes, input, inputBytes);
    uint64_t args[ADAPTIVE_ARG_COUNT] = {
        input, inputToken, output, outputToken, ctx.scratchAddress, scratchToken, 0, fusedCopy ? 1ULL : 0ULL
    };
    HCCL_DEBUG("Scatter RELAY rank=%u root=%u scene=%u relayCopy=%u fusedCopy=%u opt=%u",
        param.myRank, param.root, ctx.sceneId, ctx.relayCopy, fusedCopy ? 1U : 0U, SCATTER_OPTIMIZATIONS);
    for (uint64_t wave = 0; wave < ctx.fullSteps; ++wave) {
        args[ADAPTIVE_WAVE_OFFSET] = wave * ctx.chunkBytes;
        CHK_RET(LaunchShape(param, ctx, ctx.fullShape, args));
    }
    if (ctx.tailBytes != 0) {
        args[ADAPTIVE_WAVE_OFFSET] = ctx.fullSteps * ctx.chunkBytes;
        CHK_RET(LaunchShape(param, ctx, ctx.tailShape, args));
    }
    if (isRoot) {
        const uint64_t rootInput = input + param.root * ctx.recvBytes;
        if (!fusedCopy && rootInput != output) {
            // Root has drained all forwarders' final ACKs before writing its
            // own output. It may safely alias another rank's original slice.
            uint64_t copyArgs[SCATTER_TASK_ARG_COUNT] = {
                rootInput, inputToken, output, outputToken, ctx.recvBytes,
                (ctx.recvBytes - 1) / MAX_DATA_SIZE, (ctx.recvBytes - 1) % MAX_DATA_SIZE + 1, 0, rootInput
            };
            CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread, ctx.copyKernel, copyArgs, SCATTER_TASK_ARG_COUNT));
        }
    }
    return HCCL_SUCCESS;
}

uint64_t ScratchExtent(const ScatterShapePlan &shape)
{
    uint64_t extent = 0;
    for (const auto &batch : shape.batches) {
        for (const auto &arg : batch.groups) {
            for (uint32_t i = 0; i < arg.transferCount; ++i) {
                const auto &transfer = arg.transfers[i];
                if (transfer.source == ScatterBuffer::SCRATCH) {
                    extent = std::max(extent, transfer.sourceOffset + transfer.bytes);
                }
            }
        }
    }
    return extent;
}
} // namespace

const char *ScatterRouteReasonName(ScatterRouteReason reason)
{
    switch (reason) {
        case ScatterRouteReason::MATCHED: return "matched";
        case ScatterRouteReason::UNKNOWN_MEMBERS: return "unknown-members";
        case ScatterRouteReason::INVALID_PARTITION: return "invalid-partition";
        case ScatterRouteReason::UNSUPPORTED_TOPOLOGY: return "unsupported-topology";
        case ScatterRouteReason::UNMATCHED_SIZE: return "unmatched-size";
        case ScatterRouteReason::TREE_SCRATCH_TOO_SMALL: return "tree-scratch-too-small";
        case ScatterRouteReason::NO_LOCAL_HELPER: return "no-local-helper";
        case ScatterRouteReason::REMOTE_FANOUT_AT_MOST_FOUR: return "remote-fanout-at-most-four";
        case ScatterRouteReason::RELAY_SCRATCH_TOO_SMALL: return "relay-scratch-too-small";
        case ScatterRouteReason::DIRECT_POLICY: return "direct-policy";
        case ScatterRouteReason::SINGLE_RANK: return "single-rank";
        default: return "invalid-reason";
    }
}

HcclResult BuildScatterPlan(const OpParam &param, const ScatterCommResources &resources,
    uint64_t recvBytes, ScatterPlan &plan)
{
    uint64_t expectedBytes = 0;
    uint64_t totalBytes = 0;
    CHK_RET(ValidateScatterParams(param, expectedBytes, totalBytes));
    if (recvBytes == 0 || recvBytes != expectedBytes || resources.magic != SCATTER_CTX_MAGIC ||
        resources.version != SCATTER_CTX_VERSION || resources.myRank != param.myRank ||
        resources.rankSize != param.rankSize) { return HCCL_E_PARA; }
    plan = ScatterPlan{};
    plan.recvBytes = recvBytes;
    ScatterRouteInput input{};
    input.rankSize = param.rankSize;
    input.root = param.root;
    input.recvBytes = recvBytes;
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        input.serverMasks[rank] = resources.serverMasks[rank];
        input.capacities[rank] = resources.capacities[rank];
    }
    const ScatterRouteDecision route = SelectScatterRoute(input);
    plan.algorithm = route.algorithm;
    plan.topology = route.topology;
    plan.routeReason = route.reason;
    plan.sceneId = route.sceneId;
    plan.directFusion = route.directFusion;
    plan.rootAckFusion = route.rootAckFusion;
    plan.compactScratch = route.compactScratch;
    plan.relayOverlap = route.relayOverlap;
    plan.chunkBytes = route.chunkBytes;
    plan.fullSteps = route.fullSteps;
    plan.tailBytes = route.tailBytes;
    if (route.algorithm == ScatterAlgorithm::DIRECT) { return HCCL_SUCCESS; }

    if (route.algorithm == ScatterAlgorithm::TREE) {
        CHK_RET(BuildTree(param, resources, PartitionMembers(input), route.treeWidth, recvBytes, plan.full));
    } else {
        const uint32_t rootMask = resources.serverMasks[param.root];
        std::vector<uint32_t> helpers;
        std::vector<uint32_t> remotes;
        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            if (rank == param.root) { continue; }
            ((rootMask & (uint32_t{1} << rank)) != 0 ? helpers : remotes).push_back(rank);
        }
        CHK_RET(BuildRelayWave(param, resources, helpers, remotes, recvBytes,
            plan.chunkBytes, plan.chunkBytes, plan.rootAckFusion, plan.full, plan.compactScratch, plan.relayOverlap,
            SelectScatterHelperAckFusion(plan.sceneId)));
        if (plan.tailBytes != 0) {
            CHK_RET(BuildRelayWave(param, resources, helpers, remotes, recvBytes,
                plan.chunkBytes, plan.tailBytes, plan.rootAckFusion, plan.tail, plan.compactScratch, plan.relayOverlap,
                SelectScatterHelperAckFusion(plan.sceneId)));
        }
    }
    if ((SCATTER_V019_OPTIONS & 2U) != 0 && plan.algorithm == ScatterAlgorithm::RELAY &&
        (plan.sceneId == 2 || plan.sceneId == 3)) {
        for (auto *shape : {&plan.full, &plan.tail}) {
            for (auto &batch : shape->batches) {
                for (auto &arg : batch.groups) {
                    arg.readyWrite = arg.role == AdaptiveRole::SEND_DATA && arg.peerCount > 1 ? 1 : 0;
                }
            }
        }
    }
    const bool tailPriority = ((SCATTER_V020_OPTIONS & 1U) != 0 &&
        (plan.sceneId == 11 || plan.sceneId == 12)) || ((SCATTER_V020_OPTIONS & 2U) != 0 &&
        (plan.sceneId == 2 || plan.sceneId == 3));
    if (tailPriority && param.myRank == param.root && plan.algorithm == ScatterAlgorithm::RELAY &&
        plan.relayOverlap != 0 && plan.fullSteps == 1 && plan.tailBytes == 0) {
        for (auto &batch : plan.full.batches) {
            for (auto &arg : batch.groups) {
                if (arg.role != AdaptiveRole::SEND_DATA || arg.earlyTail == 0) { continue; }
                // Both output and scratch metadata must exist for a delayed peer.
                for (uint32_t peer = 0; peer < arg.peerCount; ++peer) {
                    if (arg.readyMasks[peer] == 15U) { arg.deferHelperOutput = 1; }
                }
            }
        }
    }
    plan.relayCopy = SelectScatterRelayCopy(plan.algorithm, plan.sceneId, plan.fullSteps, plan.tailBytes);
    if (param.myRank == param.root && plan.relayCopy != 0) {
        if (plan.full.batches.empty() || plan.full.batches.front().groups.empty()) { return HCCL_E_PARA; }
        auto &owner = plan.full.batches.front().groups.back();
        if (owner.role != AdaptiveRole::SEND_DATA) { return HCCL_E_PARA; }
        owner.copyOwner = 1;
        owner.copySourceOffset = param.root * recvBytes;
        owner.copyBytes = recvBytes;
    }
    plan.scratchUsedBytes = std::max(ScratchExtent(plan.full), ScratchExtent(plan.tail));
    if (plan.full.batches.empty() || plan.full.batches.size() > SCATTER_MAX_BATCHES ||
        (plan.tailBytes != 0 && (plan.tail.batches.empty() || plan.tail.batches.size() > SCATTER_MAX_BATCHES)) ||
        plan.scratchUsedBytes > resources.scratchBytes) { return HCCL_E_PARA; }
    return HCCL_SUCCESS;
}

HcclResult ValidateScatterParams(const OpParam &param, uint64_t &recvBytes, uint64_t &inputBytes)
{
    if (param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || param.myRank >= param.rankSize ||
        param.root >= param.rankSize) {
        return HCCL_E_PARA;
    }
    const auto type = SIZE_TABLE.find(param.dataType);
    if (type == SIZE_TABLE.end()) {
        return HCCL_E_NOT_SUPPORT;
    }
    if (param.count > std::numeric_limits<uint64_t>::max() / type->second) {
        return HCCL_E_PARA;
    }
    recvBytes = param.count * type->second;
    if (recvBytes > std::numeric_limits<uint64_t>::max() / param.rankSize) {
        return HCCL_E_PARA;
    }
    inputBytes = recvBytes * param.rankSize;
    if (recvBytes == 0) {
        return HCCL_SUCCESS;
    }

    CHK_PTR_NULL(param.outputPtr);
    if (!ValidAddressRange(param.outputPtr, recvBytes)) {
        return HCCL_E_PARA;
    }
    // sendBuf has meaning only on root. Non-root ranks may pass nullptr.
    if (param.myRank == param.root) {
        CHK_PTR_NULL(param.inputPtr);
        if (!ValidAddressRange(param.inputPtr, inputBytes)) {
            return HCCL_E_PARA;
        }
        const uint64_t src = reinterpret_cast<uintptr_t>(param.inputPtr) + param.root * recvBytes;
        const uint64_t dst = reinterpret_cast<uintptr_t>(param.outputPtr);
        // Exact in-place is a no-op; the CCU LocalCopy primitive does not
        // provide memmove semantics for partially overlapping ranges.
        if (src != dst && src < dst + recvBytes && dst < src + recvBytes) {
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ExecOp(const OpParam &param)
{
    uint64_t recvBytes = 0;
    uint64_t inputBytes = 0;
    CHK_RET(ValidateScatterParams(param, recvBytes, inputBytes));
    if (recvBytes == 0) {
        return HCCL_SUCCESS;
    }
    AlgResourceCtx ctx{};
    CHK_RET(LoadContext(param, recvBytes, ctx));

    const uint64_t output = reinterpret_cast<uintptr_t>(param.outputPtr);
    uint64_t outputToken = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(output, recvBytes, &outputToken));
    if (ctx.algorithm != ScatterAlgorithm::DIRECT) {
        return ExecAdaptive(param, ctx, inputBytes, output, outputToken);
    }

    // A positive final block makes the single-block path just one batch of
    // writes. The preceding full blocks, if any, execute inside CCU. Exact
    // multiples never enqueue a zero-byte transfer, including B == 256MiB.
    uint64_t args[SCATTER_TASK_ARG_COUNT] = {
        0, 0, output, outputToken, recvBytes,
        (recvBytes - 1) / MAX_DATA_SIZE, (recvBytes - 1) % MAX_DATA_SIZE + 1, 0, 0
    };

    const ScatterPullAddressMode pullAddressMode = SelectScatterEffectivePullAddressMode(ctx);
    if (param.myRank != param.root) {
        if (pullAddressMode == ScatterPullAddressMode::BASE_OFFSET) {
            // ValidateScatterParams already checked recvBytes * rankSize.
            // The base belongs to this invocation, not a cached user buffer.
            args[SCATTER_LOOP_COUNT] = param.myRank * recvBytes;
        }
        CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread, ctx.groups[0].kernel, args, SCATTER_TASK_ARG_COUNT));
        return HCCL_SUCCESS;
    }

    const uint64_t input = reinterpret_cast<uintptr_t>(param.inputPtr);
    uint64_t inputToken = 0;
    // The root accesses every rank's slice, so the token query covers the
    // entire input, not just a single rank's recvCount.
    CHK_RET_CCU(HcommCcuGetMemToken(input, inputBytes, &inputToken));

    args[SCATTER_SRC_ADDR] = input;
    args[SCATTER_SRC_TOKEN] = inputToken;
    const uint64_t rootInput = input + param.root * recvBytes;
    const bool fusedCopy = ctx.directFusion != 0 && ctx.groupCount != 0 &&
        !Overlaps(output, recvBytes, input, inputBytes);
    HCCL_DEBUG("Scatter DIRECT rank=%u root=%u scene=%u B=%llu directFusion=%u fusedCopy=%u dataPath=%u",
        param.myRank, param.root, ctx.sceneId, static_cast<unsigned long long>(recvBytes), ctx.directFusion,
        fusedCopy ? 1U : 0U, static_cast<uint32_t>(ctx.dataPath));
    args[SCATTER_COPY_ENABLED] = fusedCopy ? 1 : 0;
    args[SCATTER_ROOT_SRC_ADDR] = rootInput;
    if (pullAddressMode == ScatterPullAddressMode::THREE_SOURCES) {
        // A shared global mapping for every die group: the three non-root
        // ranks in ascending rank order. Never use a group's local peer index.
        constexpr uint32_t sourceSlots[] = {SCATTER_SRC_ADDR, SCATTER_LOOP_COUNT, SCATTER_LAST_BYTES};
        uint32_t ordinal = 0;
        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            if (rank == param.root) { continue; }
            args[sourceSlots[ordinal++]] = input + rank * recvBytes;
        }
    }
    CHK_RET(LaunchGroups(param, ctx, args));

    // All groups, including the worker, precede this copy on the current user
    // stream. recvBuf may alias a different rank's source slice, so waiting
    // only for root's own group would not be sufficient.
    if (!fusedCopy && rootInput != output) {
        args[SCATTER_SRC_ADDR] = rootInput;
        CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread, ctx.copyKernel, args, SCATTER_TASK_ARG_COUNT));
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
