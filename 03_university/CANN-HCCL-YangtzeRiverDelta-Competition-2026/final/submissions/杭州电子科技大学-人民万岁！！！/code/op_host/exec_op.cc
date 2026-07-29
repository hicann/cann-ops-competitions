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
#include <limits>

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>

#include "log.h"
#include "custom.h"
#include "exec_op.h"

namespace ops_hccl {
namespace {
constexpr uint64_t DIRECT_CHUNK_SIZE = 128ULL * 1024ULL * 1024ULL;
constexpr uint64_t CHAIN_SEGMENT_SIZE = 16ULL * 1024ULL * 1024ULL;
constexpr uint64_t CHAIN_TINY_TAIL_SIZE = 4ULL * 1024ULL;
constexpr uint64_t CUT_THROUGH_SIZE_512M = 512ULL * 1024ULL * 1024ULL;
constexpr uint64_t CUT_THROUGH_SIZE_400M_PLUS_4 = 400ULL * 1024ULL * 1024ULL + 4ULL;
constexpr uint64_t ASYMMETRIC_DIRECT_512K_SIZE = 512ULL * 1024ULL;
constexpr uint64_t RANK12_WEIGHTED_SLICE_ALIGNMENT = 512ULL;
constexpr uint32_t RANK12_RANK_SIZE = 12;
constexpr uint32_t RANK12_LARGE_SERVER_RANKS = 8;
constexpr uint32_t RANK12_LARGE_SERVER_WEIGHT = 3;
constexpr uint32_t RANK12_SMALL_SERVER_WEIGHT = 4;
constexpr uint32_t RANK12_TOTAL_WEIGHT = 40;
constexpr uint32_t OWNER_READY_THREAD_NOTIFY_IDX = 1;

static_assert(RANK12_LARGE_SERVER_RANKS * RANK12_LARGE_SERVER_WEIGHT +
    (RANK12_RANK_SIZE - RANK12_LARGE_SERVER_RANKS) * RANK12_SMALL_SERVER_WEIGHT ==
    RANK12_TOTAL_WEIGHT, "Rank-12 SAG weights must add up to the declared total");

struct DataSlice {
    uint64_t offset;
    uint64_t size;
};

HcclResult PreSyncThreads(const std::vector<ThreadHandle> &threads)
{
    if (threads.size() <= 1) {
        return HCCL_SUCCESS;
    }

    for (uint32_t threadIdx = 1; threadIdx < threads.size(); threadIdx++) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[0], threads[threadIdx], 0)));
    }
    for (uint32_t threadIdx = 1; threadIdx < threads.size(); threadIdx++) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(threads[threadIdx], 0, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult PostSyncThreads(const std::vector<ThreadHandle> &threads)
{
    if (threads.size() <= 1) {
        return HCCL_SUCCESS;
    }

    for (uint32_t threadIdx = 1; threadIdx < threads.size(); threadIdx++) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[0], threadIdx - 1, CUSTOM_TIMEOUT)));
    }
    for (uint32_t threadIdx = 1; threadIdx < threads.size(); threadIdx++) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(threads[threadIdx], threads[0], threadIdx - 1)));
    }
    return HCCL_SUCCESS;
}

std::vector<DataSlice> BuildDataSlices(uint64_t count, uint64_t typeSize, uint32_t rankSize)
{
    const uint64_t baseCount = count / rankSize;
    const uint64_t remainder = count % rankSize;
    const uint32_t firstLargerRank = rankSize - static_cast<uint32_t>(remainder);

    std::vector<DataSlice> slices(rankSize);
    uint64_t offset = 0;
    for (uint32_t rank = 0; rank < rankSize; rank++) {
        const uint64_t sliceCount = baseCount + ((remainder != 0 && rank >= firstLargerRank) ? 1 : 0);
        slices[rank] = DataSlice{offset, sliceCount * typeSize};
        offset += slices[rank].size;
    }
    return slices;
}

std::vector<DataSlice> BuildRank12WeightedSlices(uint64_t dataSize)
{
    const uint64_t alignedBlockCount = dataSize / RANK12_WEIGHTED_SLICE_ALIGNMENT;
    const uint64_t tailSize = dataSize % RANK12_WEIGHTED_SLICE_ALIGNMENT;

    std::vector<DataSlice> slices(RANK12_RANK_SIZE);
    uint32_t cumulativeWeight = 0;
    uint64_t offset = 0;
    for (uint32_t rank = 0; rank < RANK12_RANK_SIZE; rank++) {
        cumulativeWeight += rank < RANK12_LARGE_SERVER_RANKS ?
            RANK12_LARGE_SERVER_WEIGHT : RANK12_SMALL_SERVER_WEIGHT;
        uint64_t sliceEnd = alignedBlockCount * cumulativeWeight /
            RANK12_TOTAL_WEIGHT * RANK12_WEIGHTED_SLICE_ALIGNMENT;
        if (rank + 1 == RANK12_RANK_SIZE) {
            sliceEnd += tailSize;
        }
        slices[rank] = DataSlice{offset, sliceEnd - offset};
        offset = sliceEnd;
    }
    return slices;
}

std::vector<DataSlice> BuildChainSegments(uint64_t dataSize)
{
    std::vector<DataSlice> segments;
    uint64_t segmentOffset = 0;
    while (segmentOffset < dataSize) {
        const uint64_t remainingSize = dataSize - segmentOffset;
        uint64_t segmentSize = std::min(CHAIN_SEGMENT_SIZE, remainingSize);
        // Preserve v0.3B2's proven tail merge: a tiny tail is transferred as
        // part of the preceding full segment, avoiding a separate CCU launch.
        if (remainingSize > CHAIN_SEGMENT_SIZE &&
            remainingSize - CHAIN_SEGMENT_SIZE <= CHAIN_TINY_TAIL_SIZE) {
            segmentSize = remainingSize;
        }
        segments.push_back(DataSlice{segmentOffset, segmentSize});
        segmentOffset += segmentSize;
    }
    return segments;
}

HcclResult LaunchScatter(const AlgResourceCtx &resCtx, uint64_t baseAddr, uint64_t token,
    const std::vector<DataSlice> &slices)
{
    CHK_RET(PreSyncThreads(resCtx.threads));
    for (uint32_t kernelIdx = 0; kernelIdx < resCtx.ccuKernels.size(); kernelIdx++) {
        const std::vector<uint32_t> &peers = resCtx.peersPerKernel[kernelIdx];
        std::vector<uint64_t> taskArgs;
        taskArgs.reserve(2 + 2 * peers.size());
        taskArgs.push_back(baseAddr);
        taskArgs.push_back(token);
        for (const uint32_t peer : peers) {
            CHK_PRT_RET(peer >= slices.size(), HCCL_ERROR("Invalid scatter peer %u", peer), HCCL_E_INTERNAL);
            taskArgs.push_back(slices[peer].offset);
            taskArgs.push_back(slices[peer].size);
        }

        CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[kernelIdx], resCtx.ccuKernels[kernelIdx],
            taskArgs.data(), taskArgs.size()));
    }
    CHK_RET(PostSyncThreads(resCtx.threads));
    return HCCL_SUCCESS;
}

HcclResult LaunchAllGather(const AlgResourceCtx &resCtx, uint64_t baseAddr, uint64_t token,
    const DataSlice &localSlice)
{
    const std::vector<uint64_t> taskArgs = {baseAddr, token, localSlice.offset, localSlice.size};
    CHK_RET(PreSyncThreads(resCtx.threads));
    for (uint32_t kernelIdx = 0; kernelIdx < resCtx.allGatherKernels.size(); kernelIdx++) {
        CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[kernelIdx], resCtx.allGatherKernels[kernelIdx],
            taskArgs.data(), taskArgs.size()));
    }
    CHK_RET(PostSyncThreads(resCtx.threads));
    return HCCL_SUCCESS;
}

HcclResult LaunchOwnerReadySag(const AlgResourceCtx &resCtx, uint64_t baseAddr, uint64_t token,
    const std::vector<DataSlice> &slices, uint32_t myRank, uint32_t root)
{
    CHK_PRT_RET(resCtx.threads.empty() || resCtx.threads.size() != resCtx.ccuKernels.size() ||
        resCtx.threads.size() != resCtx.allGatherKernels.size() ||
        resCtx.threads.size() != resCtx.peersPerKernel.size() || myRank >= slices.size(),
        HCCL_ERROR("Owner-ready SAG resource shape is invalid"), HCCL_E_INTERNAL);

    uint32_t rootChannelCount = 0;
    uint32_t rootGroupIdx = INVALID_VALUE_RANKID;
    if (myRank != root) {
        for (uint32_t groupIdx = 0; groupIdx < resCtx.peersPerKernel.size(); groupIdx++) {
            for (const uint32_t peer : resCtx.peersPerKernel[groupIdx]) {
                if (peer == root) {
                    rootChannelCount++;
                    rootGroupIdx = groupIdx;
                }
            }
        }
        CHK_PRT_RET(rootChannelCount != 1 || rootGroupIdx != 0,
            HCCL_ERROR("Owner-ready SAG root channel must be unique and mapped to thread 0"),
            HCCL_E_INTERNAL);
    }

    CHK_RET(PreSyncThreads(resCtx.threads));
    for (uint32_t kernelIdx = 0; kernelIdx < resCtx.ccuKernels.size(); kernelIdx++) {
        const std::vector<uint32_t> &peers = resCtx.peersPerKernel[kernelIdx];
        std::vector<uint64_t> scatterArgs;
        scatterArgs.reserve(2 + 2 * peers.size());
        scatterArgs.push_back(baseAddr);
        scatterArgs.push_back(token);
        for (const uint32_t peer : peers) {
            CHK_PRT_RET(peer >= slices.size(), HCCL_ERROR("Invalid owner-ready scatter peer %u", peer),
                HCCL_E_INTERNAL);
            scatterArgs.push_back(slices[peer].offset);
            scatterArgs.push_back(slices[peer].size);
        }
        CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[kernelIdx], resCtx.ccuKernels[kernelIdx],
            scatterArgs.data(), scatterArgs.size()));
    }

    if (myRank != root) {
        // The group containing root was promoted to thread 0 during resource
        // construction.  Its LeanScatter kernel returns only after this rank's
        // owner slice is locally visible.  Release every auxiliary group from
        // that single completion edge instead of joining all Scatter threads.
        // Index 1 is dedicated to this edge.  Initial thread PreSync uses index
        // 0, so even an arbitrarily delayed auxiliary thread cannot observe two
        // Records on one notify before its matching Wait.
        for (uint32_t threadIdx = 1; threadIdx < resCtx.threads.size(); threadIdx++) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                resCtx.threads[0], resCtx.threads[threadIdx], OWNER_READY_THREAD_NOTIFY_IDX)));
        }
        for (uint32_t threadIdx = 1; threadIdx < resCtx.threads.size(); threadIdx++) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                resCtx.threads[threadIdx], OWNER_READY_THREAD_NOTIFY_IDX, CUSTOM_TIMEOUT)));
        }
    }

    const DataSlice &localSlice = slices[myRank];
    const std::vector<uint64_t> allGatherArgs = {
        baseAddr, token, localSlice.offset, localSlice.size};
    for (uint32_t kernelIdx = 0; kernelIdx < resCtx.allGatherKernels.size(); kernelIdx++) {
        CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[kernelIdx], resCtx.allGatherKernels[kernelIdx],
            allGatherArgs.data(), allGatherArgs.size()));
    }
    CHK_RET(PostSyncThreads(resCtx.threads));
    return HCCL_SUCCESS;
}

HcclResult LaunchPipelinedChain(
    const AlgResourceCtx &resCtx, uint64_t baseAddr, uint64_t token, uint64_t dataSize)
{
    constexpr uint32_t variantCount = static_cast<uint32_t>(ChainKernelVariant::COUNT);
    CHK_PRT_RET(resCtx.threads.size() != 1 || resCtx.ccuKernels.size() != variantCount,
        HCCL_ERROR("Pipelined chain requires one thread and %u static kernels", variantCount),
        HCCL_E_INTERNAL);

    const std::vector<DataSlice> segments = BuildChainSegments(dataSize);
    CHK_PRT_RET(segments.empty() || segments.back().offset + segments.back().size != dataSize,
        HCCL_ERROR("Pipelined chain segments do not cover the input buffer"), HCCL_E_INTERNAL);

    const auto launchSingle = [&](ChainKernelVariant variant, const DataSlice &segment) -> HcclResult {
        const uint32_t kernelIdx = static_cast<uint32_t>(variant);
        const std::vector<uint64_t> taskArgs = {baseAddr, token, segment.offset, segment.size};
        return ConvertCcuToHccl(HcommCcuKernelLaunch(
            resCtx.threads[0], resCtx.ccuKernels[kernelIdx], taskArgs.data(), taskArgs.size()));
    };
    const auto launchPair = [&](ChainKernelVariant variant, const DataSlice &first,
                                    const DataSlice &second) -> HcclResult {
        const uint32_t kernelIdx = static_cast<uint32_t>(variant);
        const std::vector<uint64_t> taskArgs = {
            baseAddr, token, first.offset, first.size, second.offset, second.size};
        return ConvertCcuToHccl(HcommCcuKernelLaunch(
            resCtx.threads[0], resCtx.ccuKernels[kernelIdx], taskArgs.data(), taskArgs.size()));
    };

    const size_t segmentCount = segments.size();
    if (segmentCount == 1) {
        return launchSingle(ChainKernelVariant::SINGLE, segments[0]);
    }
    if (segmentCount == 2) {
        CHK_RET(launchSingle(ChainKernelVariant::FIRST, segments[0]));
        return launchSingle(ChainKernelVariant::LAST, segments[1]);
    }

    const size_t pairCount = segmentCount / 2;
    const bool hasTail = (segmentCount % 2) != 0;
    // The contest shapes use either 16 pairs (512 MiB) or 12 pairs plus one
    // merged tail (400 MiB + 4 B).  Each pair owns one bit for the whole
    // broadcast, so there is no bit reuse and therefore no ACK dependency.
    const bool supportsUniqueReady = pairCount <= CHAIN_UNIQUE_PAIR_SLOT_COUNT &&
        (!hasTail || pairCount == CHAIN_UNIQUE_TAIL_SLOT);
    if (supportsUniqueReady) {
        const uint32_t pairBase = static_cast<uint32_t>(ChainKernelVariant::UNIQUE_PAIR_SLOT_BASE);
        for (size_t pairSlot = 0; pairSlot < pairCount; pairSlot++) {
            const auto variant = static_cast<ChainKernelVariant>(pairBase + pairSlot);
            CHK_RET(launchPair(variant, segments[2 * pairSlot], segments[2 * pairSlot + 1]));
        }
        if (hasTail) {
            CHK_RET(launchSingle(ChainKernelVariant::UNIQUE_TAIL_SLOT12, segments.back()));
        }
        const DataSlice drainArgs{0, dataSize};
        return launchSingle(ChainKernelVariant::UNIQUE_DRAIN, drainArgs);
    }

    // Conservative fallback for non-contest odd shapes that have no static
    // tail-bit kernel.  The proven one-segment READY/ACK protocol is slower but
    // preserves general correctness without ever reusing an unacknowledged bit.
    CHK_RET(launchSingle(ChainKernelVariant::FIRST, segments[0]));
    for (size_t segmentIdx = 1; segmentIdx + 1 < segmentCount; segmentIdx++) {
        CHK_RET(launchSingle(ChainKernelVariant::MIDDLE, segments[segmentIdx]));
    }
    return launchSingle(ChainKernelVariant::LAST, segments.back());
}

HcclResult LaunchSegmentCutThroughChain(
    const AlgResourceCtx &resCtx, uint64_t baseAddr, uint64_t token, uint64_t dataSize)
{
    CHK_PRT_RET(resCtx.threads.size() != 1 || resCtx.ccuKernels.size() != CUT_THROUGH_KERNEL_COUNT,
        HCCL_ERROR("Segment cut-through chain requires one thread and %u static kernels",
            CUT_THROUGH_KERNEL_COUNT), HCCL_E_INTERNAL);
    CHK_PRT_RET(dataSize != CUT_THROUGH_SIZE_512M && dataSize != CUT_THROUGH_SIZE_400M_PLUS_4,
        HCCL_ERROR("Unsupported segment cut-through data size %llu",
            static_cast<unsigned long long>(dataSize)), HCCL_E_INTERNAL);

    const std::vector<DataSlice> segments = BuildChainSegments(dataSize);
    const size_t expectedSegmentCount = dataSize == CUT_THROUGH_SIZE_512M ? 32 : 25;
    CHK_PRT_RET(segments.size() != expectedSegmentCount || segments.empty() ||
        segments.back().offset + segments.back().size != dataSize,
        HCCL_ERROR("Cut-through segment count/coverage is invalid"), HCCL_E_INTERNAL);
    for (size_t segmentIdx = 0; segmentIdx < segments.size(); segmentIdx++) {
        const uint64_t expectedOffset = segmentIdx == 0 ? 0 :
            segments[segmentIdx - 1].offset + segments[segmentIdx - 1].size;
        CHK_PRT_RET(segments[segmentIdx].offset != expectedOffset || segments[segmentIdx].size == 0,
            HCCL_ERROR("Cut-through segment %zu is not contiguous", segmentIdx), HCCL_E_INTERNAL);
    }

    const size_t pairCount = segments.size() / 2;
    const bool hasTail = (segments.size() % 2) != 0;
    CHK_PRT_RET(pairCount == 0 || pairCount > CUT_THROUGH_PAIR_COUNT ||
        (hasTail && segments.size() - 1 != CUT_THROUGH_TAIL_SEGMENT),
        HCCL_ERROR("Unsupported cut-through pair/tail shape"), HCCL_E_INTERNAL);

    for (size_t pairSlot = 0; pairSlot < pairCount; pairSlot++) {
        const DataSlice &first = segments[2 * pairSlot];
        const DataSlice &second = segments[2 * pairSlot + 1];
        const std::vector<uint64_t> taskArgs = {
            baseAddr, token, first.offset, first.size, second.offset, second.size};
        CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[0], resCtx.ccuKernels[pairSlot],
            taskArgs.data(), taskArgs.size()));
    }

    if (hasTail) {
        const DataSlice &tail = segments.back();
        const std::vector<uint64_t> tailArgs = {baseAddr, token, tail.offset, tail.size};
        CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[0],
            resCtx.ccuKernels[CUT_THROUGH_TAIL_KERNEL], tailArgs.data(), tailArgs.size()));
    }

    const std::vector<uint64_t> drainArgs = {baseAddr, token, 0, dataSize};
    CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[0],
        resCtx.ccuKernels[CUT_THROUGH_DRAIN_KERNEL], drainArgs.data(), drainArgs.size()));
    return HCCL_SUCCESS;
}

HcclResult LaunchQuad4Final5Chain(
    const AlgResourceCtx &resCtx, uint64_t baseAddr, uint64_t token, uint64_t dataSize)
{
    CHK_PRT_RET(resCtx.threads.size() != 1 || resCtx.ccuKernels.size() != QUAD4_RESOURCE_COUNT,
        HCCL_ERROR("Quad4 Final5 chain requires one thread and %u static kernels",
            QUAD4_RESOURCE_COUNT), HCCL_E_INTERNAL);
    CHK_PRT_RET(dataSize != CUT_THROUGH_SIZE_512M && dataSize != CUT_THROUGH_SIZE_400M_PLUS_4,
        HCCL_ERROR("Unsupported Quad4 Final5 data size %llu",
            static_cast<unsigned long long>(dataSize)), HCCL_E_INTERNAL);

    const std::vector<DataSlice> segments = BuildChainSegments(dataSize);
    const size_t expectedSegmentCount = dataSize == CUT_THROUGH_SIZE_512M ? 32 : 25;
    CHK_PRT_RET(segments.size() != expectedSegmentCount || segments.empty() ||
        segments.back().offset + segments.back().size != dataSize,
        HCCL_ERROR("Quad4 segment count/coverage is invalid"), HCCL_E_INTERNAL);
    for (size_t segmentIdx = 0; segmentIdx < segments.size(); segmentIdx++) {
        const uint64_t expectedOffset = segmentIdx == 0 ? 0 :
            segments[segmentIdx - 1].offset + segments[segmentIdx - 1].size;
        CHK_PRT_RET(segments[segmentIdx].offset != expectedOffset || segments[segmentIdx].size == 0,
            HCCL_ERROR("Quad4 segment %zu is not contiguous", segmentIdx), HCCL_E_INTERNAL);
    }

    const bool is512M = dataSize == CUT_THROUGH_SIZE_512M;
    const size_t quadCount = is512M ? QUAD4_KERNEL_COUNT :
        (segments.size() - FINAL5_SEGMENTS_PER_KERNEL) / QUAD4_SEGMENTS_PER_KERNEL;
    const size_t coveredSegments = quadCount * QUAD4_SEGMENTS_PER_KERNEL +
        (is512M ? 0 : FINAL5_SEGMENTS_PER_KERNEL);
    CHK_PRT_RET(quadCount == 0 || quadCount > QUAD4_KERNEL_COUNT ||
        coveredSegments != segments.size(),
        HCCL_ERROR("Unsupported Quad4 Final5 shape"), HCCL_E_INTERNAL);

    for (size_t quadSlot = 0; quadSlot < quadCount; quadSlot++) {
        std::vector<uint64_t> taskArgs;
        taskArgs.reserve(2 + 2 * QUAD4_SEGMENTS_PER_KERNEL);
        taskArgs.push_back(baseAddr);
        taskArgs.push_back(token);
        for (size_t segmentInQuad = 0; segmentInQuad < QUAD4_SEGMENTS_PER_KERNEL; segmentInQuad++) {
            const DataSlice &segment =
                segments[QUAD4_SEGMENTS_PER_KERNEL * quadSlot + segmentInQuad];
            taskArgs.push_back(segment.offset);
            taskArgs.push_back(segment.size);
        }
        CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[0], resCtx.ccuKernels[quadSlot],
            taskArgs.data(), taskArgs.size()));
    }

    if (!is512M) {
        std::vector<uint64_t> finalArgs;
        finalArgs.reserve(2 + 2 * FINAL5_SEGMENTS_PER_KERNEL);
        finalArgs.push_back(baseAddr);
        finalArgs.push_back(token);
        const size_t finalFirstSegment = quadCount * QUAD4_SEGMENTS_PER_KERNEL;
        for (size_t finalSegment = 0; finalSegment < FINAL5_SEGMENTS_PER_KERNEL; finalSegment++) {
            const DataSlice &segment = segments[finalFirstSegment + finalSegment];
            finalArgs.push_back(segment.offset);
            finalArgs.push_back(segment.size);
        }
        CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[0],
            resCtx.ccuKernels[FINAL5_DRAIN_KERNEL], finalArgs.data(), finalArgs.size()));
        return HCCL_SUCCESS;
    }

    const std::vector<uint64_t> drainArgs = {baseAddr, token, 0, dataSize};
    CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[0],
        resCtx.ccuKernels[QUAD4_DRAIN_KERNEL], drainArgs.data(), drainArgs.size()));
    return HCCL_SUCCESS;
}

HcclResult LaunchPersistentRolling2_400(
    const AlgResourceCtx &resCtx, uint64_t baseAddr, uint64_t token, uint64_t dataSize)
{
    CHK_PRT_RET(resCtx.threads.size() != 1 || resCtx.ccuKernels.size() != 1 ||
        !resCtx.allGatherKernels.empty(),
        HCCL_ERROR("Persistent Rolling-2 resource shape is invalid"), HCCL_E_INTERNAL);
    CHK_PRT_RET(dataSize != CUT_THROUGH_SIZE_400M_PLUS_4,
        HCCL_ERROR("Unsupported Persistent Rolling-2 data size %llu",
            static_cast<unsigned long long>(dataSize)), HCCL_E_INTERNAL);

    // Cell offsets, sizes and READY locations are embedded in the static CCU
    // graph; only invocation-specific memory identity crosses the Host boundary.
    const std::vector<uint64_t> taskArgs = {baseAddr, token};
    CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[0], resCtx.ccuKernels[0],
        taskArgs.data(), taskArgs.size()));
    return HCCL_SUCCESS;
}

HcclResult LaunchPersistentRolling2_512(
    const AlgResourceCtx &resCtx, uint64_t baseAddr, uint64_t token, uint64_t dataSize)
{
    CHK_PRT_RET(resCtx.threads.size() != 1 || resCtx.ccuKernels.size() != 1 ||
        !resCtx.allGatherKernels.empty(),
        HCCL_ERROR("Persistent 512 MiB Rolling-2 resource shape is invalid"), HCCL_E_INTERNAL);
    CHK_PRT_RET(dataSize != CUT_THROUGH_SIZE_512M,
        HCCL_ERROR("Unsupported Persistent 512 MiB Rolling-2 data size %llu",
            static_cast<unsigned long long>(dataSize)), HCCL_E_INTERNAL);

    // All 64 offsets, fixed 8 MiB sizes and READY locations are embedded in
    // the static graph; only the per-invocation memory identity is dynamic.
    const std::vector<uint64_t> taskArgs = {baseAddr, token};
    CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[0], resCtx.ccuKernels[0],
        taskArgs.data(), taskArgs.size()));
    return HCCL_SUCCESS;
}

HcclResult LaunchAsymmetricDirect512K2Arg(
    const AlgResourceCtx &resCtx, uint64_t baseAddr, uint64_t token, uint64_t dataSize)
{
    CHK_PRT_RET(resCtx.threads.empty() || resCtx.threads.size() != resCtx.ccuKernels.size() ||
        !resCtx.allGatherKernels.empty(),
        HCCL_ERROR("512 KiB asymmetric resource shape is invalid"), HCCL_E_INTERNAL);
    CHK_PRT_RET(dataSize != ASYMMETRIC_DIRECT_512K_SIZE,
        HCCL_ERROR("Unsupported 512 KiB asymmetric data size %llu",
            static_cast<unsigned long long>(dataSize)), HCCL_E_INTERNAL);

    const std::vector<uint64_t> taskArgs = {baseAddr, token};
    // Match the generic Direct path when one rank owns multiple topology/die
    // kernel groups.  Every auxiliary thread must be released before launch
    // and joined before the collective returns.
    CHK_RET(PreSyncThreads(resCtx.threads));
    for (uint32_t kernelIdx = 0; kernelIdx < resCtx.ccuKernels.size(); kernelIdx++) {
        CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[kernelIdx], resCtx.ccuKernels[kernelIdx],
            taskArgs.data(), taskArgs.size()));
    }
    CHK_RET(PostSyncThreads(resCtx.threads));
    return HCCL_SUCCESS;
}

HcclResult LaunchAsymmetricDirect512KRegistered(
    const AlgResourceCtx &resCtx, uint64_t baseAddr, uint64_t token, uint64_t dataSize)
{
    CHK_PRT_RET(resCtx.threads.empty() || resCtx.threads.size() != resCtx.ccuKernels.size() ||
        resCtx.registeredFastKernels.size() != resCtx.threads.size() ||
        !resCtx.allGatherKernels.empty(),
        HCCL_ERROR("Registered 512 KiB asymmetric resource shape is invalid"), HCCL_E_INTERNAL);
    CHK_PRT_RET(dataSize != ASYMMETRIC_DIRECT_512K_SIZE,
        HCCL_ERROR("Unsupported registered asymmetric data size %llu",
            static_cast<unsigned long long>(dataSize)), HCCL_E_INTERNAL);

    const bool registeredIdentityMatches = baseAddr == resCtx.registeredBaseAddr &&
        token == resCtx.registeredToken;
    if (registeredIdentityMatches) {
        CHK_RET(PreSyncThreads(resCtx.threads));
        for (uint32_t kernelIdx = 0; kernelIdx < resCtx.registeredFastKernels.size(); kernelIdx++) {
            CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[kernelIdx],
                resCtx.registeredFastKernels[kernelIdx], nullptr, 0));
        }
        CHK_RET(PostSyncThreads(resCtx.threads));
        return HCCL_SUCCESS;
    }

    // Engine contexts may be reused by a later invocation with a different
    // user buffer or token.  Fall back to the validated two-argument graph;
    // never execute a graph containing stale memory identity.
    return LaunchAsymmetricDirect512K2Arg(resCtx, baseAddr, token, dataSize);
}
} // namespace

HcclResult ExecOp(const OpParam &param)
{
    // 反序列化
    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> seq(ctx, ctx + param.ctxSize);
    AlgResourceCtx resCtx;
    resCtx.DeSerialize(seq);

    CHK_PRT_RET(resCtx.threads.empty() || resCtx.ccuKernels.empty(), HCCL_ERROR("CCU resource is empty"),
        HCCL_E_INTERNAL);
    const bool singleThreadMultiKernel = resCtx.algorithm == BroadcastAlgorithm::PIPELINED_CHAIN ||
        resCtx.algorithm == BroadcastAlgorithm::SEGMENT_CUT_THROUGH_CHAIN ||
        resCtx.algorithm == BroadcastAlgorithm::QUAD4_ROLLING2_FINAL5_CHAIN;
    CHK_PRT_RET(!singleThreadMultiKernel &&
        resCtx.threads.size() != resCtx.ccuKernels.size(),
        HCCL_ERROR("Thread count %zu does not match kernel count %zu", resCtx.threads.size(),
            resCtx.ccuKernels.size()),
        HCCL_E_INTERNAL);
    if (resCtx.algorithm == BroadcastAlgorithm::SCATTER_ALLGATHER ||
        resCtx.algorithm == BroadcastAlgorithm::LEAN_SCATTER_ALLGATHER ||
        resCtx.algorithm == BroadcastAlgorithm::OWNER_READY_SCATTER_ALLGATHER) {
        CHK_PRT_RET(resCtx.allGatherKernels.size() != resCtx.threads.size(),
            HCCL_ERROR("AllGather kernel count %zu does not match thread count %zu",
                resCtx.allGatherKernels.size(), resCtx.threads.size()),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(resCtx.peersPerKernel.size() != resCtx.ccuKernels.size(),
            HCCL_ERROR("Peer group count %zu does not match kernel count %zu", resCtx.peersPerKernel.size(),
                resCtx.ccuKernels.size()),
            HCCL_E_INTERNAL);
        if (resCtx.algorithm == BroadcastAlgorithm::LEAN_SCATTER_ALLGATHER ||
            resCtx.algorithm == BroadcastAlgorithm::OWNER_READY_SCATTER_ALLGATHER) {
            const bool validLeanRank = param.rankSize == 12 ||
                (V05B_ENABLE_RANK16_LEAN_SAG && param.rankSize == 16);
            CHK_PRT_RET(!validLeanRank,
                HCCL_ERROR("Lean-SAG does not support rank size %u", param.rankSize), HCCL_E_INTERNAL);
        }
    }
    if (resCtx.algorithm == BroadcastAlgorithm::PIPELINED_CHAIN) {
        constexpr uint32_t variantCount = static_cast<uint32_t>(ChainKernelVariant::COUNT);
        CHK_PRT_RET(resCtx.threads.size() != 1 || resCtx.ccuKernels.size() != variantCount ||
            !resCtx.allGatherKernels.empty(),
            HCCL_ERROR("Pipelined chain resource shape is invalid"), HCCL_E_INTERNAL);
    }
    if (resCtx.algorithm == BroadcastAlgorithm::SEGMENT_CUT_THROUGH_CHAIN) {
        CHK_PRT_RET(resCtx.threads.size() != 1 ||
            resCtx.ccuKernels.size() != CUT_THROUGH_KERNEL_COUNT || !resCtx.allGatherKernels.empty(),
            HCCL_ERROR("Segment cut-through chain resource shape is invalid"), HCCL_E_INTERNAL);
    }
    if (resCtx.algorithm == BroadcastAlgorithm::QUAD4_ROLLING2_FINAL5_CHAIN) {
        CHK_PRT_RET(resCtx.threads.size() != 1 ||
            resCtx.ccuKernels.size() != QUAD4_RESOURCE_COUNT || !resCtx.allGatherKernels.empty(),
            HCCL_ERROR("Quad4 Final5 chain resource shape is invalid"), HCCL_E_INTERNAL);
    }
    if (resCtx.algorithm == BroadcastAlgorithm::PERSISTENT_ROLLING2_400) {
        CHK_PRT_RET(resCtx.threads.size() != 1 || resCtx.ccuKernels.size() != 1 ||
            !resCtx.allGatherKernels.empty(),
            HCCL_ERROR("Persistent Rolling-2 chain resource shape is invalid"), HCCL_E_INTERNAL);
    }
    if (resCtx.algorithm == BroadcastAlgorithm::PERSISTENT_ROLLING2_512) {
        CHK_PRT_RET(resCtx.threads.size() != 1 || resCtx.ccuKernels.size() != 1 ||
            !resCtx.allGatherKernels.empty(),
            HCCL_ERROR("Persistent 512 MiB Rolling-2 chain resource shape is invalid"),
            HCCL_E_INTERNAL);
    }
    if (resCtx.algorithm == BroadcastAlgorithm::ASYMMETRIC_DIRECT_512K_REGISTERED) {
        CHK_PRT_RET(resCtx.registeredFastKernels.size() != resCtx.threads.size(),
            HCCL_ERROR("Registered fast kernel count %zu does not match thread count %zu",
                resCtx.registeredFastKernels.size(), resCtx.threads.size()), HCCL_E_INTERNAL);
    }

    // Engine Context可跨调用复用，但主Thread必须绑定本次传入的stream。
    resCtx.threads[0] = param.cpuThread;

    const auto typeSizeIt = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(typeSizeIt == SIZE_TABLE.end(), HCCL_ERROR("Unsupported data type %d", param.dataType),
        HCCL_E_NOT_SUPPORT);
    const uint64_t typeSize = typeSizeIt->second;
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / typeSize,
        HCCL_ERROR("Broadcast data size overflows uint64"), HCCL_E_PARA);

    const uint64_t dataSize = param.count * typeSize;
    if (dataSize == 0) {
        return HCCL_SUCCESS;
    }
    if (resCtx.algorithm == BroadcastAlgorithm::LEAN_SCATTER_ALLGATHER ||
        resCtx.algorithm == BroadcastAlgorithm::OWNER_READY_SCATTER_ALLGATHER) {
        CHK_PRT_RET(typeSize != sizeof(float) ||
            (dataSize != CUT_THROUGH_SIZE_512M && dataSize != CUT_THROUGH_SIZE_400M_PLUS_4),
            HCCL_ERROR("Lean-SAG only supports the two 12-rank contest large-message cases"),
            HCCL_E_INTERNAL);
    }
    if (resCtx.algorithm == BroadcastAlgorithm::ASYMMETRIC_PULL_512K_2ARG) {
        CHK_PRT_RET(resCtx.threads.size() != resCtx.ccuKernels.size() ||
            !resCtx.registeredFastKernels.empty() || !resCtx.allGatherKernels.empty(),
            HCCL_ERROR("512 KiB pull resource shape is invalid"), HCCL_E_INTERNAL);
    }
    if (resCtx.algorithm == BroadcastAlgorithm::PERSISTENT_ROLLING2_400) {
        CHK_PRT_RET(typeSize != sizeof(float) || param.rankSize != 4 ||
            dataSize != CUT_THROUGH_SIZE_400M_PLUS_4,
            HCCL_ERROR("Persistent Rolling-2 only supports the 4-rank 400 MiB + 4 B case"),
            HCCL_E_INTERNAL);
    }
    if (resCtx.algorithm == BroadcastAlgorithm::PERSISTENT_ROLLING2_512) {
        CHK_PRT_RET(typeSize != sizeof(float) || param.rankSize != 4 ||
            dataSize != CUT_THROUGH_SIZE_512M,
            HCCL_ERROR("Persistent Rolling-2 only supports the 4-rank 512 MiB case"),
            HCCL_E_INTERNAL);
    }

    const uint64_t baseAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    uint64_t token = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(baseAddr, dataSize, &token));

    if (resCtx.algorithm == BroadcastAlgorithm::SCATTER_ALLGATHER ||
        resCtx.algorithm == BroadcastAlgorithm::LEAN_SCATTER_ALLGATHER ||
        resCtx.algorithm == BroadcastAlgorithm::OWNER_READY_SCATTER_ALLGATHER) {
        const bool useRank12WeightedSlices =
            resCtx.algorithm == BroadcastAlgorithm::OWNER_READY_SCATTER_ALLGATHER &&
            param.rankSize == RANK12_RANK_SIZE && param.root < RANK12_LARGE_SERVER_RANKS &&
            typeSize == sizeof(float) &&
            (dataSize == CUT_THROUGH_SIZE_512M || dataSize == CUT_THROUGH_SIZE_400M_PLUS_4);
        const std::vector<DataSlice> slices = useRank12WeightedSlices ?
            BuildRank12WeightedSlices(dataSize) :
            BuildDataSlices(param.count, typeSize, param.rankSize);
        CHK_PRT_RET(slices.empty() || slices.back().offset + slices.back().size != dataSize,
            HCCL_ERROR("Scatter/AllGather slices do not cover the input buffer"), HCCL_E_INTERNAL);
        if (resCtx.algorithm == BroadcastAlgorithm::OWNER_READY_SCATTER_ALLGATHER) {
            CHK_RET(LaunchOwnerReadySag(resCtx, baseAddr, token, slices, param.myRank, param.root));
        } else {
            CHK_RET(LaunchScatter(resCtx, baseAddr, token, slices));
            CHK_RET(LaunchAllGather(resCtx, baseAddr, token, slices[param.myRank]));
        }
        return HCCL_SUCCESS;
    }

    if (resCtx.algorithm == BroadcastAlgorithm::PIPELINED_CHAIN) {
        return LaunchPipelinedChain(resCtx, baseAddr, token, dataSize);
    }

    if (resCtx.algorithm == BroadcastAlgorithm::SEGMENT_CUT_THROUGH_CHAIN) {
        return LaunchSegmentCutThroughChain(resCtx, baseAddr, token, dataSize);
    }

    if (resCtx.algorithm == BroadcastAlgorithm::QUAD4_ROLLING2_FINAL5_CHAIN) {
        return LaunchQuad4Final5Chain(resCtx, baseAddr, token, dataSize);
    }

    if (resCtx.algorithm == BroadcastAlgorithm::PERSISTENT_ROLLING2_400) {
        return LaunchPersistentRolling2_400(resCtx, baseAddr, token, dataSize);
    }

    if (resCtx.algorithm == BroadcastAlgorithm::PERSISTENT_ROLLING2_512) {
        return LaunchPersistentRolling2_512(resCtx, baseAddr, token, dataSize);
    }

    if (resCtx.algorithm == BroadcastAlgorithm::ASYMMETRIC_DIRECT_512K_2ARG) {
        CHK_PRT_RET(typeSize != sizeof(float) ||
            (param.rankSize != 4 && param.rankSize != 12 && param.rankSize != 16),
            HCCL_ERROR("512 KiB asymmetric path supports only float32 rank 4/12/16"), HCCL_E_INTERNAL);
        return LaunchAsymmetricDirect512K2Arg(resCtx, baseAddr, token, dataSize);
    }

    if (resCtx.algorithm == BroadcastAlgorithm::ASYMMETRIC_DIRECT_512K_REGISTERED) {
        CHK_PRT_RET(typeSize != sizeof(float) ||
            (param.rankSize != 4 && param.rankSize != 12 && param.rankSize != 16),
            HCCL_ERROR("Registered 512 KiB asymmetric path supports only float32 rank 4/12/16"),
            HCCL_E_INTERNAL);
        return LaunchAsymmetricDirect512KRegistered(resCtx, baseAddr, token, dataSize);
    }

    if (resCtx.algorithm == BroadcastAlgorithm::ASYMMETRIC_PULL_512K_2ARG) {
        CHK_PRT_RET(typeSize != sizeof(float) ||
            (param.rankSize != 4 && param.rankSize != 12 && param.rankSize != 16),
            HCCL_ERROR("512 KiB asymmetric pull supports only float32 rank 4/12/16"),
            HCCL_E_INTERNAL);
        return LaunchAsymmetricDirect512K2Arg(resCtx, baseAddr, token, dataSize);
    }

    uint64_t processedSize = 0;
    while (processedSize < dataSize) {
        const uint64_t chunkSize = std::min(DIRECT_CHUNK_SIZE, dataSize - processedSize);
        const uint64_t chunkAddr = baseAddr + processedSize;
        std::vector<uint64_t> taskArgs = {chunkAddr, token, chunkSize};

        CHK_RET(PreSyncThreads(resCtx.threads));
        for (uint32_t kernelIdx = 0; kernelIdx < resCtx.ccuKernels.size(); kernelIdx++) {
            CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[kernelIdx], resCtx.ccuKernels[kernelIdx],
                taskArgs.data(), taskArgs.size()));
        }
        CHK_RET(PostSyncThreads(resCtx.threads));

        processedSize += chunkSize;
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
