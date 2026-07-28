/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "custom.h"
#include "exec_op.h"
#include "log.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <vector>

namespace {
constexpr uint32_t RS_DATA_NOTIFY = 0;
constexpr uint32_t RS_ACK_NOTIFY = 1;
constexpr uint32_t AG_READY_NOTIFY = 2;
constexpr uint32_t AG_ACK_NOTIFY = 3;
constexpr uint32_t ROUND_DONE_NOTIFY = 4;
constexpr uint64_t HCCL_MIN_SLICE_ALIGN = 128;
constexpr uint64_t CROSS_BFLY_BYTES = 512ULL * 1024ULL;
constexpr uint64_t MIDDLE_512M_BYTES = 512ULL * 1024ULL * 1024ULL;
constexpr uint64_t MIDDLE_512M_SLICE_BYTES = MIDDLE_512M_BYTES / custom_allreduce::RANK_SIZE;
constexpr uint64_t LARGE_400M4B_BYTES = 400ULL * 1024ULL * 1024ULL + custom_allreduce::DATA_TYPE_SIZE;
constexpr uint64_t LARGE_400M4B_COMMON_SLICE_BYTES = 25ULL * 1024ULL * 1024ULL;
constexpr uint64_t LARGE_400M4B_LAST_SLICE_BYTES
    = LARGE_400M4B_COMMON_SLICE_BYTES + custom_allreduce::DATA_TYPE_SIZE;
constexpr uint32_t SERVER_CROSS_RANK_MASK = 8;
constexpr uint32_t CROSS_ACCUMULATOR_LANE_COUNT = 2;
constexpr uint32_t CROSS_ACCUMULATOR_LANE_SIZE
    = custom_allreduce::LOCAL_RANK_SIZE / CROSS_ACCUMULATOR_LANE_COUNT;
constexpr uint32_t COMPACT_SAME_SERVER_SLOT_COUNT = custom_allreduce::LOCAL_RANK_SIZE - 1;
constexpr uint32_t COMPACT_CROSS_ACCUMULATOR_SLOT = COMPACT_SAME_SERVER_SLOT_COUNT;
constexpr uint32_t COMPACT_CROSS_LANE_SLOT_COUNT
    = COMPACT_CROSS_ACCUMULATOR_SLOT + CROSS_ACCUMULATOR_LANE_COUNT;
constexpr uint32_t CROSS_LANE_READY_THREAD_NOTIFY = custom_allreduce::PEER_SIZE - 1;
constexpr uint32_t PIPELINE_WORKER_COUNT = 2;
constexpr uint32_t SAME_SERVER_PIPELINE_SHARD_COUNT = CROSS_ACCUMULATOR_LANE_SIZE;
constexpr uint32_t SAME_SERVER_SHARD_NOTIFY_BASE = ROUND_DONE_NOTIFY + 1;
constexpr uint32_t PIPELINE_CHANNEL_NOTIFY_NUM
    = SAME_SERVER_SHARD_NOTIFY_BASE + SAME_SERVER_PIPELINE_SHARD_COUNT;
constexpr uint32_t PIPELINE_WORKER_START_NOTIFY = custom_allreduce::PEER_SIZE - 1;
constexpr uint32_t WORKER_SECOND_DONE_NOTIFY = custom_allreduce::LOCAL_RANK_SIZE;
constexpr uint32_t CHANNEL_WORKER_DONE_NOTIFY = custom_allreduce::PEER_SIZE - 1;
constexpr uint32_t CROSS_END_THREAD_NOTIFY_BASE = custom_allreduce::PEER_SIZE;
constexpr uint32_t FUSED_ROUND_BEGIN_THREAD_NOTIFY = custom_allreduce::PEER_SIZE;

using custom_allreduce::RankSlice;
using custom_allreduce::SegmentDesc;
using RankSlices = std::array<RankSlice, custom_allreduce::RANK_SIZE>;
using RoundSegments = std::array<SegmentDesc, custom_allreduce::RANK_SIZE>;

struct WorkSlice {
    uint64_t offset = 0;
    uint64_t count = 0;
};

void *OffsetAddr(void *addr, uint64_t offset)
{
    return static_cast<void *>(static_cast<uint8_t *>(addr) + offset);
}

HcclResult CheckChannelBuffer(const ChannelInfo &channel, uint64_t offset, uint64_t bytes)
{
    CHK_PTR_NULL(channel.remoteCclMem.addr);
    CHK_PRT_RET(offset > channel.remoteCclMem.size || bytes > channel.remoteCclMem.size - offset,
        HCCL_ERROR("Remote CCL buffer for rank %u is too small: size=%llu, offset=%llu, required=%llu",
            channel.remoteRank, static_cast<unsigned long long>(channel.remoteCclMem.size),
            static_cast<unsigned long long>(offset), static_cast<unsigned long long>(bytes)),
        HCCL_E_MEMORY);
    return HCCL_SUCCESS;
}

HcclResult BatchWriteAndNotify(ThreadHandle thread, const ChannelInfo &channel, void *dst, const void *src,
    uint64_t bytes, uint32_t notifyIdx = NOTIFY_IDX_DATA_SIGNAL)
{
    HcommBatchTransferDesc descs[2] = {};
    descs[0].transType = HCOMM_TRANSFER_TYPE_WRITE;
    descs[0].transferInfo.write.len = bytes;
    descs[0].transferInfo.write.dst = dst;
    descs[0].transferInfo.write.src = const_cast<void *>(src);
    descs[1].transType = HCOMM_TRANSFER_TYPE_NOTIFY_RECORD;
    descs[1].transferInfo.notifyRecord.notifyIdx = notifyIdx;
    return static_cast<HcclResult>(HcommBatchTransferOnThread(thread, channel.handle, descs, 2));
}

HcclResult BatchWriteReduceAndNotify(ThreadHandle thread, const ChannelInfo &channel, void *dst, const void *src,
    uint64_t count, uint32_t notifyIdx)
{
    HcommBatchTransferDesc descs[2] = {};
    descs[0].transType = HCOMM_TRANSFER_TYPE_WRITE_REDUCE;
    descs[0].transferInfo.reduce.count = count;
    descs[0].transferInfo.reduce.dst = dst;
    descs[0].transferInfo.reduce.src = const_cast<void *>(src);
    descs[0].transferInfo.reduce.reduceOp = HCOMM_REDUCE_SUM;
    descs[0].transferInfo.reduce.dataType = HCOMM_DATA_TYPE_FP32;
    descs[1].transType = HCOMM_TRANSFER_TYPE_NOTIFY_RECORD;
    descs[1].transferInfo.notifyRecord.notifyIdx = notifyIdx;
    return static_cast<HcclResult>(HcommBatchTransferOnThread(thread, channel.handle, descs, 2));
}

HcclResult BatchReadAndNotify(ThreadHandle thread, const ChannelInfo &channel, void *dst, const void *src,
    uint64_t bytes, uint32_t notifyIdx)
{
    HcommBatchTransferDesc descs[2] = {};
    descs[0].transType = HCOMM_TRANSFER_TYPE_READ;
    descs[0].transferInfo.read.len = bytes;
    descs[0].transferInfo.read.dst = dst;
    descs[0].transferInfo.read.src = const_cast<void *>(src);
    descs[1].transType = HCOMM_TRANSFER_TYPE_NOTIFY_RECORD;
    descs[1].transferInfo.notifyRecord.notifyIdx = notifyIdx;
    return static_cast<HcclResult>(HcommBatchTransferOnThread(thread, channel.handle, descs, 2));
}

bool HasTwoServerCrossButterflyChannels(const OpParam &param, const AlgResourceCtx &resCtx)
{
    constexpr uint32_t masks[] = {SERVER_CROSS_RANK_MASK, SERVER_CROSS_RANK_MASK ^ 1U,
        SERVER_CROSS_RANK_MASK ^ 2U, SERVER_CROSS_RANK_MASK ^ 4U};
    if (resCtx.channels.size() != sizeof(masks) / sizeof(masks[0])) {
        return false;
    }
    for (uint32_t step = 0; step < sizeof(masks) / sizeof(masks[0]); ++step) {
        if (resCtx.channels[step].remoteRankIndex != (param.myRank ^ masks[step])) {
            return false;
        }
    }
    return true;
}

bool ShouldUseTwoServerCrossButterfly512K(const OpParam &param, const AlgResourceCtx &resCtx)
{
    const uint64_t totalBytes = param.count * custom_allreduce::DATA_TYPE_SIZE;
    const uint64_t slotBytes = custom_allreduce::AlignUp(totalBytes, HCCL_MIN_SLICE_ALIGN);
    return param.rankSize == custom_allreduce::RANK_SIZE && totalBytes == CROSS_BFLY_BYTES &&
        resCtx.localBuffer.size >= 4ULL * slotBytes && HasTwoServerCrossButterflyChannels(param, resCtx);
}

HcclResult RunTwoServerCrossButterfly512K(const OpParam &param, const AlgResourceCtx &resCtx)
{
    const ThreadHandle thread = resCtx.threads[0];
    const uint64_t totalBytes = param.count * custom_allreduce::DATA_TYPE_SIZE;
    const uint64_t slotBytes = custom_allreduce::AlignUp(totalBytes, HCCL_MIN_SLICE_ALIGN);
    void *accumAddr = param.inputPtr;

    constexpr uint32_t masks[] = {SERVER_CROSS_RANK_MASK, SERVER_CROSS_RANK_MASK ^ 1U,
        SERVER_CROSS_RANK_MASK ^ 2U, SERVER_CROSS_RANK_MASK ^ 4U};
    for (uint32_t stepIdx = 0; stepIdx + 1U < sizeof(masks) / sizeof(masks[0]); ++stepIdx) {
        const ChannelInfo &channel = resCtx.channels[stepIdx];

        const uint64_t scratchOffset = static_cast<uint64_t>(stepIdx) * slotBytes;
        void *scratchAddr = OffsetAddr(resCtx.localBuffer.addr, scratchOffset);
        CHK_RET(CheckChannelBuffer(channel, scratchOffset, totalBytes));
        CHK_RET(BatchWriteAndNotify(
            thread, channel, OffsetAddr(channel.remoteCclMem.addr, scratchOffset), accumAddr, totalBytes));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
            thread, scratchAddr, accumAddr, param.count, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
        accumAddr = scratchAddr;
    }

    constexpr uint32_t finalStep = 3;
    const ChannelInfo &finalChannel = resCtx.channels[finalStep];
    const uint64_t remoteAccumOffset = static_cast<uint64_t>(finalStep - 1U) * slotBytes;
    CHK_RET(CheckChannelBuffer(finalChannel, remoteAccumOffset, totalBytes));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, finalChannel.handle, RS_DATA_NOTIFY)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, finalChannel.handle, RS_DATA_NOTIFY, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommReadOnThread(thread, finalChannel.handle, param.outputPtr,
        OffsetAddr(finalChannel.remoteCclMem.addr, remoteAccumOffset), totalBytes)));
    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
        thread, param.outputPtr, accumAddr, param.count, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    return HCCL_SUCCESS;
}

HcclResult BeginParallel(const std::vector<ThreadHandle> &threads, uint32_t activeThreadCount)
{
    CHK_PRT_RET(activeThreadCount == 0 || activeThreadCount > threads.size(),
        HCCL_ERROR("Invalid active thread count[%u] for pool[%llu]", activeThreadCount,
            static_cast<unsigned long long>(threads.size())),
        HCCL_E_INTERNAL);
    for (uint32_t i = 1; i < activeThreadCount; ++i) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[0], threads[i], 0)));
    }
    for (uint32_t i = 1; i < activeThreadCount; ++i) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[i], 0, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult BeginParallel(const std::vector<ThreadHandle> &threads)
{
    return BeginParallel(threads, static_cast<uint32_t>(threads.size()));
}

HcclResult EndParallel(const std::vector<ThreadHandle> &threads, uint32_t activeThreadCount)
{
    CHK_PRT_RET(activeThreadCount == 0 || activeThreadCount > threads.size(),
        HCCL_ERROR("Invalid active thread count[%u] for pool[%llu]", activeThreadCount,
            static_cast<unsigned long long>(threads.size())),
        HCCL_E_INTERNAL);
    for (uint32_t i = 1; i < activeThreadCount; ++i) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[0], i - 1, CUSTOM_TIMEOUT)));
    }
    for (uint32_t i = 1; i < activeThreadCount; ++i) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[i], threads[0], i - 1)));
    }
    return HCCL_SUCCESS;
}

HcclResult EndParallel(const std::vector<ThreadHandle> &threads)
{
    return EndParallel(threads, static_cast<uint32_t>(threads.size()));
}

HcclResult ValidateCompact512KResources(const OpParam &param, const AlgResourceCtx &resCtx)
{
    CHK_PRT_RET(resCtx.rankIndex != param.myRank || resCtx.rankIndex >= custom_allreduce::RANK_SIZE,
        HCCL_ERROR("Invalid compact 512KiB rank index[%u]", resCtx.rankIndex), HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.ranks.size() != custom_allreduce::RANK_SIZE || resCtx.threads.size() != 1 ||
            resCtx.channels.size() != 4,
        HCCL_ERROR("Invalid compact 512KiB resources: ranks[%llu], threads[%llu], channels[%llu]",
            static_cast<unsigned long long>(resCtx.ranks.size()),
            static_cast<unsigned long long>(resCtx.threads.size()),
            static_cast<unsigned long long>(resCtx.channels.size())),
        HCCL_E_INTERNAL);
    CHK_PTR_NULL(resCtx.localBuffer.addr);
    for (const ChannelInfo &channel : resCtx.channels) {
        CHK_PRT_RET(channel.notifyNum < 2,
            HCCL_ERROR("Compact channel to rank[%u] has insufficient notify resources", channel.remoteRank),
            HCCL_E_INTERNAL);
        CHK_PTR_NULL(channel.remoteCclMem.addr);
    }
    return HCCL_SUCCESS;
}

HcclResult ValidateResources(const AlgResourceCtx &resCtx)
{
    CHK_PRT_RET(resCtx.rankIndex >= custom_allreduce::RANK_SIZE,
        HCCL_ERROR("Invalid global rank index[%u]", resCtx.rankIndex), HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.ranks.size() != custom_allreduce::RANK_SIZE,
        HCCL_ERROR("Unexpected rank vector size[%llu]", static_cast<unsigned long long>(resCtx.ranks.size())),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.threads.size() != custom_allreduce::PEER_SIZE,
        HCCL_ERROR("Unexpected thread count[%llu]", static_cast<unsigned long long>(resCtx.threads.size())),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.channels.size() != custom_allreduce::PEER_SIZE,
        HCCL_ERROR("Unexpected channel count[%llu]", static_cast<unsigned long long>(resCtx.channels.size())),
        HCCL_E_INTERNAL);
    CHK_PTR_NULL(resCtx.localBuffer.addr);

    uint32_t expectedRemoteIndex = 0;
    for (const ChannelInfo &channel : resCtx.channels) {
        if (expectedRemoteIndex == resCtx.rankIndex) {
            ++expectedRemoteIndex;
        }
        CHK_PRT_RET(channel.remoteRankIndex != expectedRemoteIndex,
            HCCL_ERROR("Unexpected channel order: got[%u], expected[%u]", channel.remoteRankIndex, expectedRemoteIndex),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(channel.notifyNum < 5,
            HCCL_ERROR("Channel to rank[%u] has insufficient notify resources", channel.remoteRank), HCCL_E_INTERNAL);
        CHK_PTR_NULL(channel.remoteCclMem.addr);
        ++expectedRemoteIndex;
    }
    return HCCL_SUCCESS;
}

uint32_t CompactSourceSlot(uint32_t sourceRank, uint32_t ownerRank)
{
    return sourceRank < ownerRank ? sourceRank : sourceRank - 1;
}

bool IsSameServer(uint32_t lhsRank, uint32_t rhsRank)
{
    return (lhsRank & SERVER_CROSS_RANK_MASK) == (rhsRank & SERVER_CROSS_RANK_MASK);
}

uint32_t CompactSameServerSourceSlot(uint32_t sourceRank, uint32_t ownerRank)
{
    const uint32_t sourceLocalIndex = sourceRank & (custom_allreduce::LOCAL_RANK_SIZE - 1U);
    const uint32_t ownerLocalIndex = ownerRank & (custom_allreduce::LOCAL_RANK_SIZE - 1U);
    return sourceLocalIndex < ownerLocalIndex ? sourceLocalIndex : sourceLocalIndex - 1U;
}

WorkSlice GetWorkSlice(uint64_t count, uint32_t worker, uint32_t workerCount)
{
    const uint64_t base = count / workerCount;
    const uint64_t remainder = count % workerCount;
    WorkSlice work;
    work.count = base + static_cast<uint64_t>(worker < remainder);
    work.offset = static_cast<uint64_t>(worker) * base + std::min<uint64_t>(worker, remainder);
    return work;
}

SegmentDesc GetEqualPipelineShard(uint64_t totalBytes, uint32_t shard)
{
    if (shard >= CROSS_ACCUMULATOR_LANE_SIZE) {
        return {};
    }
    const uint64_t shardBytes = custom_allreduce::AlignDown(
        totalBytes / CROSS_ACCUMULATOR_LANE_SIZE, custom_allreduce::PREFERRED_ALIGN);
    const uint64_t offsetBytes = static_cast<uint64_t>(shard) * shardBytes;
    const uint64_t validBytes = shard + 1U == CROSS_ACCUMULATOR_LANE_SIZE
        ? totalBytes - offsetBytes
        : shardBytes;
    return SegmentDesc{offsetBytes, validBytes};
}

SegmentDesc GetSameServerPipelineShard(uint64_t totalBytes, uint32_t shard)
{
    if (shard >= CROSS_ACCUMULATOR_LANE_SIZE) {
        return {};
    }
    if (totalBytes == MIDDLE_512M_SLICE_BYTES) {
        constexpr uint64_t offsets[] = {0, 8ULL * 1024 * 1024, 20ULL * 1024 * 1024, 28ULL * 1024 * 1024};
        constexpr uint64_t sizes[] = {
            8ULL * 1024 * 1024, 12ULL * 1024 * 1024, 8ULL * 1024 * 1024, 4ULL * 1024 * 1024};
        return SegmentDesc{offsets[shard], sizes[shard]};
    }
    return GetEqualPipelineShard(totalBytes, shard);
}

HcclResult ValidateSlotLayout(uint64_t bufferSize, uint64_t slotStride, uint32_t ownerRank)
{
    CHK_PRT_RET(slotStride == 0 || slotStride > bufferSize / custom_allreduce::PEER_SIZE,
        HCCL_ERROR("CCL buffer for owner[%u] cannot hold 15 slots with stride[%llu]", ownerRank,
            static_cast<unsigned long long>(slotStride)),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

RankSlices BuildRankSlices(uint64_t totalBytes)
{
    RankSlices slices{};
    for (uint32_t rank = 0; rank < custom_allreduce::RANK_SIZE; ++rank) {
        slices[rank] = custom_allreduce::BuildRankSlice(totalBytes, rank);
    }
    return slices;
}

uint64_t GetRoundCount(const RankSlices &slices, uint64_t tileBytes)
{
    return custom_allreduce::DivUp(slices[0].validBytes, tileBytes);
}

RoundSegments BuildRoundSegments(const RankSlices &slices, uint64_t round, uint64_t roundCount, uint64_t tileBytes)
{
    RoundSegments segments{};
    for (uint32_t rank = 0; rank < custom_allreduce::RANK_SIZE; ++rank) {
        segments[rank] = custom_allreduce::BuildSegment(slices[rank], round, roundCount, tileBytes);
    }
    return segments;
}

uint64_t MaxSliceBytes(const RankSlices &slices)
{
    uint64_t result = 0;
    for (const RankSlice &slice : slices) {
        result = std::max(result, slice.validBytes);
    }
    return result;
}

uint64_t OwnerBufferSize(const AlgResourceCtx &resCtx, uint32_t ownerRank)
{
    if (ownerRank == resCtx.rankIndex) {
        return resCtx.localBuffer.size;
    }
    for (const ChannelInfo &channel : resCtx.channels) {
        if (channel.remoteRankIndex == ownerRank) {
            return channel.remoteCclMem.size;
        }
    }
    return 0;
}

bool TileFits(const AlgResourceCtx &resCtx, const RankSlices &slices, uint64_t tileBytes)
{
    if (tileBytes == 0) {
        return false;
    }
    const uint64_t roundCount = GetRoundCount(slices, tileBytes);
    if (roundCount == 0) {
        return false;
    }
    for (uint64_t round = 0; round < roundCount; ++round) {
        const RoundSegments segments = BuildRoundSegments(slices, round, roundCount, tileBytes);
        for (uint32_t owner = 0; owner < custom_allreduce::RANK_SIZE; ++owner) {
            const uint64_t stride
                = custom_allreduce::AlignUp(segments[owner].validBytes, custom_allreduce::PREFERRED_ALIGN);
            const uint64_t bufferSize = OwnerBufferSize(resCtx, owner);
            if (stride == 0 || bufferSize == 0 || stride > bufferSize / custom_allreduce::PEER_SIZE) {
                return false;
            }
        }
    }
    return true;
}

bool CompactCrossLaneSlotsFit(uint64_t bufferSize, uint64_t slotStride)
{
    return slotStride != 0 && slotStride <= bufferSize / COMPACT_CROSS_LANE_SLOT_COUNT;
}

bool CompactCrossLaneTileFits(const AlgResourceCtx &resCtx, const RankSlices &slices, uint64_t tileBytes)
{
    if (tileBytes == 0) {
        return false;
    }
    const uint64_t roundCount = GetRoundCount(slices, tileBytes);
    if (roundCount == 0) {
        return false;
    }
    for (uint64_t round = 0; round < roundCount; ++round) {
        const RoundSegments segments = BuildRoundSegments(slices, round, roundCount, tileBytes);
        for (uint32_t owner = 0; owner < custom_allreduce::RANK_SIZE; ++owner) {
            const uint64_t stride
                = custom_allreduce::AlignUp(segments[owner].validBytes, custom_allreduce::PREFERRED_ALIGN);
            if (!CompactCrossLaneSlotsFit(OwnerBufferSize(resCtx, owner), stride)) {
                return false;
            }
        }
    }
    return true;
}

bool IsCrossLaneAccumulatorMessage(uint64_t totalBytes)
{
    return totalBytes == MIDDLE_512M_BYTES || totalBytes == LARGE_400M4B_BYTES;
}

bool ShouldUseCrossLaneAccumulator(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t totalBytes, uint64_t tileBytes,
    uint32_t workerCount)
{
    const bool validTile = totalBytes == MIDDLE_512M_BYTES
        ? (tileBytes == custom_allreduce::TILE_16_MIB || tileBytes == MIDDLE_512M_SLICE_BYTES)
        : (totalBytes == LARGE_400M4B_BYTES && tileBytes == LARGE_400M4B_LAST_SLICE_BYTES);
    if (param.rankSize != custom_allreduce::RANK_SIZE || !IsCrossLaneAccumulatorMessage(totalBytes) ||
        !validTile || workerCount == 0 ||
        workerCount >= custom_allreduce::LOCAL_RANK_SIZE ||
        resCtx.workerThreads.size() != PIPELINE_WORKER_COUNT ||
        !CompactCrossLaneSlotsFit(resCtx.localBuffer.size, tileBytes)) {
        return false;
    }
    for (const ChannelInfo &channel : resCtx.channels) {
        if (channel.notifyNum < PIPELINE_CHANNEL_NOTIFY_NUM ||
            !CompactCrossLaneSlotsFit(channel.remoteCclMem.size, tileBytes)) {
            return false;
        }
    }
    return true;
}

uint64_t FixedTileBytes(custom_allreduce::AlgorithmType algorithm)
{
    switch (algorithm) {
        case custom_allreduce::AlgorithmType::Direct16MiB:
            return custom_allreduce::TILE_16_MIB;
        case custom_allreduce::AlgorithmType::Direct8MiB:
            return custom_allreduce::TILE_8_MIB;
        case custom_allreduce::AlgorithmType::Direct5MiB:
            return custom_allreduce::TILE_5_MIB;
        case custom_allreduce::AlgorithmType::Direct4MiB:
            return custom_allreduce::TILE_4_MIB;
        default:
            return 0;
    }
}

uint64_t MaxRuntimeTileBytes(const AlgResourceCtx &resCtx, const RankSlices &slices)
{
    uint64_t result = MaxSliceBytes(slices);
    for (uint32_t owner = 0; owner < custom_allreduce::RANK_SIZE; ++owner) {
        const uint64_t bufferSize = OwnerBufferSize(resCtx, owner);
        if (bufferSize == 0) {
            return 0;
        }
        const uint64_t ownerLimit
            = custom_allreduce::AlignDown(bufferSize / custom_allreduce::PEER_SIZE, custom_allreduce::PREFERRED_ALIGN);
        result = std::min(result, ownerLimit);
    }
    return result;
}

uint64_t SelectTileBytes(
    custom_allreduce::AlgorithmType algorithm, const AlgResourceCtx &resCtx, const RankSlices &slices,
    uint64_t totalBytes)
{
    const uint64_t whole = MaxSliceBytes(slices);
    if (algorithm == custom_allreduce::AlgorithmType::DirectWhole) {
        return TileFits(resCtx, slices, whole) ? whole : 0;
    }
    const uint64_t fixed = FixedTileBytes(algorithm);
    if (fixed != 0) {
        return TileFits(resCtx, slices, fixed) ? fixed : 0;
    }
    if (algorithm != custom_allreduce::AlgorithmType::DirectAuto) {
        return 0;
    }

    const uint64_t expectedWhole = totalBytes == MIDDLE_512M_BYTES
        ? MIDDLE_512M_SLICE_BYTES
        : LARGE_400M4B_LAST_SLICE_BYTES;
    if (IsCrossLaneAccumulatorMessage(totalBytes) && whole == expectedWhole &&
        CompactCrossLaneTileFits(resCtx, slices, whole)) {
        return whole;
    }
    if (totalBytes == MIDDLE_512M_BYTES && TileFits(resCtx, slices, custom_allreduce::TILE_16_MIB)) {
        return custom_allreduce::TILE_16_MIB;
    }
    const uint64_t runtimeMaxTile = MaxRuntimeTileBytes(resCtx, slices);
    const std::array<uint64_t, 6> candidates = {whole, runtimeMaxTile, custom_allreduce::TILE_16_MIB,
        custom_allreduce::TILE_8_MIB, custom_allreduce::TILE_5_MIB, custom_allreduce::TILE_4_MIB};
    for (uint64_t candidate : candidates) {
        if (candidate <= whole && TileFits(resCtx, slices, candidate)) {
            return candidate;
        }
    }
    return 0;
}

HcclResult ExchangeContributions(const OpParam &param, const AlgResourceCtx &resCtx, const RoundSegments &segments)
{
    CHK_RET(BeginParallel(resCtx.threads));
    for (uint32_t i = 0; i < resCtx.channels.size(); ++i) {
        const ChannelInfo &channel = resCtx.channels[i];
        const ThreadHandle thread = resCtx.threads[i];
        const uint32_t owner = channel.remoteRankIndex;
        const SegmentDesc &segment = segments[owner];
        const uint64_t slotStride = custom_allreduce::AlignUp(segment.validBytes, custom_allreduce::PREFERRED_ALIGN);
        CHK_RET(ValidateSlotLayout(channel.remoteCclMem.size, slotStride, owner));
        const uint32_t slot = CompactSourceSlot(resCtx.rankIndex, owner);
        void *dst = static_cast<uint8_t *>(channel.remoteCclMem.addr) + static_cast<uint64_t>(slot) * slotStride;
        const void *src = static_cast<const uint8_t *>(param.inputPtr) + segment.offsetBytes;
        CHK_RET(BatchWriteAndNotify(thread, channel, dst, src, segment.validBytes, RS_DATA_NOTIFY));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, channel.handle, RS_DATA_NOTIFY, CUSTOM_TIMEOUT)));
    }
    CHK_RET(EndParallel(resCtx.threads));
    return HCCL_SUCCESS;
}

HcclResult ReduceSourcesInRankOrder(
    const OpParam &param, const AlgResourceCtx &resCtx, const SegmentDesc &segment, uint64_t slotStride,
    uint32_t workerCount)
{
    CHK_RET(ValidateSlotLayout(resCtx.localBuffer.size, slotStride, resCtx.rankIndex));
    const uint64_t totalCount = segment.validBytes / custom_allreduce::DATA_TYPE_SIZE;
    uint8_t *localBuffer = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    const uint8_t *input = static_cast<const uint8_t *>(param.inputPtr) + segment.offsetBytes;
    uint8_t *output = static_cast<uint8_t *>(param.outputPtr) + segment.offsetBytes;

    CHK_RET(BeginParallel(resCtx.threads, workerCount));
    for (uint32_t worker = 0; worker < workerCount; ++worker) {
        const WorkSlice work = GetWorkSlice(totalCount, worker, workerCount);
        if (work.count == 0) {
            continue;
        }
        const uint64_t byteOffset = work.offset * custom_allreduce::DATA_TYPE_SIZE;
        const uint64_t workBytes = work.count * custom_allreduce::DATA_TYPE_SIZE;
        const ThreadHandle thread = resCtx.threads[worker];
        void *dst = nullptr;
        if (resCtx.rankIndex == 0) {
            dst = output + byteOffset;
            const void *source0 = input + byteOffset;
            if (dst != source0) {
                CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, dst, source0, workBytes)));
            }
        } else {
            dst = localBuffer + byteOffset; // source rank 0 is always compact slot 0.
        }

        for (uint32_t source = 1; source < custom_allreduce::RANK_SIZE; ++source) {
            const void *src = nullptr;
            if (source == resCtx.rankIndex) {
                src = input + byteOffset;
            } else {
                const uint32_t slot = CompactSourceSlot(source, resCtx.rankIndex);
                src = localBuffer + static_cast<uint64_t>(slot) * slotStride + byteOffset;
            }
            CHK_RET(static_cast<HcclResult>(
                HcommLocalReduceOnThread(thread, dst, src, work.count, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
        }

        if (resCtx.rankIndex != 0) {
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, output + byteOffset, dst, workBytes)));
        }
    }
    CHK_RET(EndParallel(resCtx.threads, workerCount));
    return HCCL_SUCCESS;
}

HcclResult ReleaseReduceScatterSlots(const AlgResourceCtx &resCtx)
{
    CHK_RET(BeginParallel(resCtx.threads));
    for (uint32_t i = 0; i < resCtx.channels.size(); ++i) {
        const ChannelInfo &channel = resCtx.channels[i];
        const ThreadHandle thread = resCtx.threads[i];
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, channel.handle, RS_ACK_NOTIFY)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, channel.handle, RS_ACK_NOTIFY, CUSTOM_TIMEOUT)));
    }
    CHK_RET(EndParallel(resCtx.threads));
    return HCCL_SUCCESS;
}

HcclResult PublishOwnerResult(const OpParam &param, const AlgResourceCtx &resCtx, const SegmentDesc &segment)
{
    if (resCtx.rankIndex != 0) {
        return HCCL_SUCCESS; // The source-0 slot is already the reduced result.
    }
    const void *src = static_cast<const uint8_t *>(param.outputPtr) + segment.offsetBytes;
    CHK_RET(static_cast<HcclResult>(
        HcommLocalCopyOnThread(resCtx.threads[0], resCtx.localBuffer.addr, src, segment.validBytes)));
    return HCCL_SUCCESS;
}

HcclResult AllGatherRound(const OpParam &param, const AlgResourceCtx &resCtx, const RoundSegments &segments)
{
    CHK_RET(BeginParallel(resCtx.threads));
    for (uint32_t i = 0; i < resCtx.channels.size(); ++i) {
        const ChannelInfo &channel = resCtx.channels[i];
        const ThreadHandle thread = resCtx.threads[i];
        const SegmentDesc &segment = segments[channel.remoteRankIndex];
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, channel.handle, AG_READY_NOTIFY)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, channel.handle, AG_READY_NOTIFY, CUSTOM_TIMEOUT)));
        void *dst = static_cast<uint8_t *>(param.outputPtr) + segment.offsetBytes;
        CHK_RET(BatchReadAndNotify(thread, channel, dst, channel.remoteCclMem.addr, segment.validBytes, AG_ACK_NOTIFY));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, channel.handle, AG_ACK_NOTIFY, CUSTOM_TIMEOUT)));
    }
    CHK_RET(EndParallel(resCtx.threads));
    return HCCL_SUCCESS;
}

HcclResult AllGatherRoundFromExtraAccumulator(
    const OpParam &param, const AlgResourceCtx &resCtx, const RoundSegments &segments, uint32_t accumulatorSlot)
{
    CHK_RET(BeginParallel(resCtx.threads));
    for (uint32_t i = 0; i < resCtx.channels.size(); ++i) {
        const ChannelInfo &channel = resCtx.channels[i];
        const ThreadHandle thread = resCtx.threads[i];
        const SegmentDesc &segment = segments[channel.remoteRankIndex];
        const uint64_t remoteStride
            = custom_allreduce::AlignUp(segment.validBytes, custom_allreduce::PREFERRED_ALIGN);
        const uint64_t accumulatorOffset = static_cast<uint64_t>(accumulatorSlot) * remoteStride;
        CHK_RET(CheckChannelBuffer(channel, accumulatorOffset, segment.validBytes));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, channel.handle, AG_READY_NOTIFY)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, channel.handle, AG_READY_NOTIFY, CUSTOM_TIMEOUT)));
        void *dst = static_cast<uint8_t *>(param.outputPtr) + segment.offsetBytes;
        const void *src = static_cast<const uint8_t *>(channel.remoteCclMem.addr) + accumulatorOffset;
        CHK_RET(BatchReadAndNotify(thread, channel, dst, src, segment.validBytes, AG_ACK_NOTIFY));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, channel.handle, AG_ACK_NOTIFY, CUSTOM_TIMEOUT)));
    }
    CHK_RET(EndParallel(resCtx.threads));
    return HCCL_SUCCESS;
}

HcclResult FusedReleaseAllGatherAndRoundCompletion(
    const OpParam &param, const AlgResourceCtx &resCtx, const RoundSegments &segments, uint32_t accumulatorSlot)
{
    for (uint32_t i = 1; i < resCtx.threads.size(); ++i) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resCtx.threads[0], resCtx.threads[i], FUSED_ROUND_BEGIN_THREAD_NOTIFY)));
    }
    for (uint32_t i = 1; i < resCtx.threads.size(); ++i) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            resCtx.threads[i], FUSED_ROUND_BEGIN_THREAD_NOTIFY, CUSTOM_TIMEOUT)));
    }
    for (uint32_t i = 0; i < resCtx.channels.size(); ++i) {
        const ChannelInfo &channel = resCtx.channels[i];
        const ThreadHandle thread = resCtx.threads[i];
        const SegmentDesc &segment = segments[channel.remoteRankIndex];
        const uint64_t remoteStride
            = custom_allreduce::AlignUp(segment.validBytes, custom_allreduce::PREFERRED_ALIGN);
        const uint64_t accumulatorOffset = static_cast<uint64_t>(accumulatorSlot) * remoteStride;
        CHK_RET(CheckChannelBuffer(channel, accumulatorOffset, segment.validBytes));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, channel.handle, RS_ACK_NOTIFY)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, channel.handle, RS_ACK_NOTIFY, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, channel.handle, AG_READY_NOTIFY)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, channel.handle, AG_READY_NOTIFY, CUSTOM_TIMEOUT)));
        void *dst = static_cast<uint8_t *>(param.outputPtr) + segment.offsetBytes;
        const void *src = static_cast<const uint8_t *>(channel.remoteCclMem.addr) + accumulatorOffset;
        CHK_RET(BatchReadAndNotify(thread, channel, dst, src, segment.validBytes, AG_ACK_NOTIFY));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, channel.handle, AG_ACK_NOTIFY, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, channel.handle, ROUND_DONE_NOTIFY)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, channel.handle, ROUND_DONE_NOTIFY, CUSTOM_TIMEOUT)));
    }
    CHK_RET(EndParallel(resCtx.threads));
    return HCCL_SUCCESS;
}

HcclResult SynchronizeRoundCompletion(const AlgResourceCtx &resCtx)
{
    CHK_RET(BeginParallel(resCtx.threads));
    for (uint32_t i = 0; i < resCtx.channels.size(); ++i) {
        const ChannelInfo &channel = resCtx.channels[i];
        const ThreadHandle thread = resCtx.threads[i];
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, channel.handle, ROUND_DONE_NOTIFY)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, channel.handle, ROUND_DONE_NOTIFY, CUSTOM_TIMEOUT)));
    }
    CHK_RET(EndParallel(resCtx.threads));
    return HCCL_SUCCESS;
}

HcclResult ExchangeAndReduceCrossLaneAccumulator(const OpParam &param, const AlgResourceCtx &resCtx,
    const RoundSegments &segments, const SegmentDesc &localSegment, uint64_t slotStride, uint32_t workerCount)
{
    std::array<uint32_t, custom_allreduce::LOCAL_RANK_SIZE - 1> sameChannelIndices{};
    std::array<uint32_t, custom_allreduce::LOCAL_RANK_SIZE> crossChannelIndices{};
    std::array<uint32_t, custom_allreduce::LOCAL_RANK_SIZE> crossChannelByLocalIndex{};
    crossChannelByLocalIndex.fill(INVALID_VALUE_RANKID);
    uint32_t sameChannelCount = 0;
    uint32_t crossChannelCount = 0;
    for (uint32_t channelIndex = 0; channelIndex < resCtx.channels.size(); ++channelIndex) {
        if (IsSameServer(resCtx.rankIndex, resCtx.channels[channelIndex].remoteRankIndex)) {
            CHK_PRT_RET(sameChannelCount >= sameChannelIndices.size(),
                HCCL_ERROR("Too many same-server channels for rank[%u]", resCtx.rankIndex), HCCL_E_INTERNAL);
            sameChannelIndices[sameChannelCount++] = channelIndex;
        } else {
            CHK_PRT_RET(crossChannelCount >= crossChannelIndices.size(),
                HCCL_ERROR("Too many cross-server channels for rank[%u]", resCtx.rankIndex), HCCL_E_INTERNAL);
            crossChannelIndices[crossChannelCount++] = channelIndex;
            const uint32_t remoteLocalIndex
                = resCtx.channels[channelIndex].remoteRankIndex & (custom_allreduce::LOCAL_RANK_SIZE - 1U);
            CHK_PRT_RET(crossChannelByLocalIndex[remoteLocalIndex] != INVALID_VALUE_RANKID,
                HCCL_ERROR("Duplicate cross-server local index[%u] for rank[%u]", remoteLocalIndex,
                    resCtx.rankIndex),
                HCCL_E_INTERNAL);
            crossChannelByLocalIndex[remoteLocalIndex] = channelIndex;
        }
    }
    CHK_PRT_RET(sameChannelCount != sameChannelIndices.size() ||
        crossChannelCount != crossChannelIndices.size() || workerCount > sameChannelCount,
        HCCL_ERROR("Rank[%u] has invalid same/cross channel counts[%u/%u] or worker count[%u]", resCtx.rankIndex,
            sameChannelCount, crossChannelCount, workerCount),
        HCCL_E_INTERNAL);
    for (uint32_t remoteLocalIndex = 0; remoteLocalIndex < crossChannelByLocalIndex.size(); ++remoteLocalIndex) {
        CHK_PRT_RET(crossChannelByLocalIndex[remoteLocalIndex] == INVALID_VALUE_RANKID,
            HCCL_ERROR("Missing cross-server local index[%u] for rank[%u]", remoteLocalIndex, resCtx.rankIndex),
            HCCL_E_INTERNAL);
    }

    uint8_t *localBuffer = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    const uint8_t *localInput = static_cast<const uint8_t *>(param.inputPtr) + localSegment.offsetBytes;
    const bool batchSameServerShards
        = param.count * custom_allreduce::DATA_TYPE_SIZE == LARGE_400M4B_BYTES;

    CHK_RET(BeginParallel(resCtx.threads));
    for (const ThreadHandle workerThread : resCtx.workerThreads) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resCtx.threads[0], workerThread, PIPELINE_WORKER_START_NOTIFY)));
    }
    for (const ThreadHandle workerThread : resCtx.workerThreads) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            workerThread, PIPELINE_WORKER_START_NOTIFY, CUSTOM_TIMEOUT)));
    }
    for (uint32_t sameIndex = 0; sameIndex < sameChannelCount; ++sameIndex) {
        const uint32_t i = sameChannelIndices[sameIndex];
        const ChannelInfo &channel = resCtx.channels[i];
        const ThreadHandle thread = resCtx.threads[i];
        const uint32_t owner = channel.remoteRankIndex;
        const SegmentDesc &segment = segments[owner];
        const uint64_t remoteStride
            = custom_allreduce::AlignUp(segment.validBytes, custom_allreduce::PREFERRED_ALIGN);
        CHK_PRT_RET(!CompactCrossLaneSlotsFit(channel.remoteCclMem.size, remoteStride),
            HCCL_ERROR("Remote CCL buffer for owner[%u] cannot hold compact cross-lane slots", owner),
            HCCL_E_MEMORY);
        const uint32_t slot = CompactSameServerSourceSlot(resCtx.rankIndex, owner);
        std::array<HcommBatchTransferDesc, 2U * SAME_SERVER_PIPELINE_SHARD_COUNT> batchDescs{};
        for (uint32_t shard = 0; shard < CROSS_ACCUMULATOR_LANE_SIZE; ++shard) {
            const SegmentDesc shardSlice = GetSameServerPipelineShard(segment.validBytes, shard);
            const uint64_t shardOffset = shardSlice.offsetBytes;
            const uint64_t shardBytes = shardSlice.validBytes;
            CHK_PRT_RET(shardBytes == 0 || shardBytes % custom_allreduce::DATA_TYPE_SIZE != 0 ||
                shardOffset + shardBytes > segment.validBytes,
                HCCL_ERROR("Invalid same-server shard[%u] for owner[%u]", shard, owner), HCCL_E_INTERNAL);
            const void *src
                = static_cast<const uint8_t *>(param.inputPtr) + segment.offsetBytes + shardOffset;
            void *dst = static_cast<uint8_t *>(channel.remoteCclMem.addr) +
                static_cast<uint64_t>(slot) * remoteStride + shardOffset;
            if (batchSameServerShards) {
                HcommBatchTransferDesc &writeDesc = batchDescs[2U * shard];
                writeDesc.transType = HCOMM_TRANSFER_TYPE_WRITE;
                writeDesc.transferInfo.write.len = shardBytes;
                writeDesc.transferInfo.write.dst = dst;
                writeDesc.transferInfo.write.src = const_cast<void *>(src);
                HcommBatchTransferDesc &notifyDesc = batchDescs[2U * shard + 1U];
                notifyDesc.transType = HCOMM_TRANSFER_TYPE_NOTIFY_RECORD;
                notifyDesc.transferInfo.notifyRecord.notifyIdx = SAME_SERVER_SHARD_NOTIFY_BASE + shard;
            } else {
                CHK_RET(BatchWriteAndNotify(
                    thread, channel, dst, src, shardBytes, SAME_SERVER_SHARD_NOTIFY_BASE + shard));
            }
        }
        if (batchSameServerShards) {
            CHK_RET(static_cast<HcclResult>(HcommBatchTransferOnThread(
                thread, channel.handle, batchDescs.data(), static_cast<uint32_t>(batchDescs.size()))));
        }
    }

    const uint32_t sourceLocalIndex = resCtx.rankIndex & (custom_allreduce::LOCAL_RANK_SIZE - 1U);
    const uint32_t sourceLane = sourceLocalIndex % CROSS_ACCUMULATOR_LANE_COUNT;
    const uint32_t sourceLanePosition = sourceLocalIndex / CROSS_ACCUMULATOR_LANE_COUNT;
    for (uint32_t phase = 0; phase < CROSS_ACCUMULATOR_LANE_SIZE; ++phase) {
        if (phase != 0) {
            for (uint32_t remoteLocalIndex = 0; remoteLocalIndex < custom_allreduce::LOCAL_RANK_SIZE;
                ++remoteLocalIndex) {
                const uint32_t i = crossChannelByLocalIndex[remoteLocalIndex];
                const ChannelInfo &channel = resCtx.channels[i];
                const ThreadHandle thread = resCtx.threads[i];
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                    thread, CROSS_LANE_READY_THREAD_NOTIFY, CUSTOM_TIMEOUT)));
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyRecordOnThread(thread, channel.handle, AG_READY_NOTIFY)));
            }
            for (uint32_t crossIndex = 0; crossIndex < crossChannelCount; ++crossIndex) {
                const uint32_t i = crossChannelIndices[crossIndex];
                const ChannelInfo &channel = resCtx.channels[i];
                const ThreadHandle thread = resCtx.threads[i];
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyWaitOnThread(thread, channel.handle, AG_READY_NOTIFY, CUSTOM_TIMEOUT)));
            }
        }

        for (uint32_t crossIndex = 0; crossIndex < crossChannelCount; ++crossIndex) {
            const uint32_t i = crossChannelIndices[crossIndex];
            const ChannelInfo &channel = resCtx.channels[i];
            const ThreadHandle thread = resCtx.threads[i];
            const uint32_t owner = channel.remoteRankIndex;
            const SegmentDesc &segment = segments[owner];
            const uint64_t remoteStride
                = custom_allreduce::AlignUp(segment.validBytes, custom_allreduce::PREFERRED_ALIGN);
            const uint32_t shardIndex = (sourceLanePosition + phase) % CROSS_ACCUMULATOR_LANE_SIZE;
            const SegmentDesc shardSlice = GetEqualPipelineShard(segment.validBytes, shardIndex);
            const uint64_t shardOffset = shardSlice.offsetBytes;
            const uint64_t shardBytes = shardSlice.validBytes;
            const void *src = static_cast<const uint8_t *>(param.inputPtr) + segment.offsetBytes + shardOffset;
            CHK_PRT_RET(shardBytes == 0 || shardBytes % custom_allreduce::DATA_TYPE_SIZE != 0 ||
                shardOffset + shardBytes > segment.validBytes,
                HCCL_ERROR("Invalid cross-server shard[%u] for owner[%u]", shardIndex, owner), HCCL_E_INTERNAL);
            CHK_PRT_RET(!CompactCrossLaneSlotsFit(channel.remoteCclMem.size, remoteStride),
                HCCL_ERROR("Remote CCL buffer for owner[%u] cannot hold lane accumulators", owner),
                HCCL_E_MEMORY);
            void *dst = static_cast<uint8_t *>(channel.remoteCclMem.addr) +
                static_cast<uint64_t>(COMPACT_CROSS_ACCUMULATOR_SLOT + sourceLane) * remoteStride + shardOffset;
            if (phase == 0) {
                CHK_RET(BatchWriteAndNotify(thread, channel, dst, src, shardBytes, RS_DATA_NOTIFY));
            } else {
                CHK_RET(BatchWriteReduceAndNotify(thread, channel, dst, src,
                    shardBytes / custom_allreduce::DATA_TYPE_SIZE, RS_DATA_NOTIFY));
            }
        }

        for (uint32_t remoteLocalIndex = 0; remoteLocalIndex < custom_allreduce::LOCAL_RANK_SIZE;
            ++remoteLocalIndex) {
            const uint32_t i = crossChannelByLocalIndex[remoteLocalIndex];
            const ChannelInfo &channel = resCtx.channels[i];
            const ThreadHandle thread = resCtx.threads[i];
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyWaitOnThread(thread, channel.handle, RS_DATA_NOTIFY, CUSTOM_TIMEOUT)));
            if (phase + 1U < CROSS_ACCUMULATOR_LANE_SIZE) {
                const uint32_t nextLocalIndex = (remoteLocalIndex + custom_allreduce::LOCAL_RANK_SIZE -
                    CROSS_ACCUMULATOR_LANE_COUNT) % custom_allreduce::LOCAL_RANK_SIZE;
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(thread,
                    resCtx.threads[crossChannelByLocalIndex[nextLocalIndex]], CROSS_LANE_READY_THREAD_NOTIFY)));
            }
        }
    }

    const bool useFusedRound
        = param.count * custom_allreduce::DATA_TYPE_SIZE == LARGE_400M4B_BYTES;
    CHK_PRT_RET(workerCount != PIPELINE_WORKER_COUNT,
        HCCL_ERROR("Cross-lane shard pipeline requires %u workers, got[%u]", PIPELINE_WORKER_COUNT, workerCount),
        HCCL_E_INTERNAL);
    if (useFusedRound) {
        std::array<ThreadHandle, custom_allreduce::LOCAL_RANK_SIZE + 1U> completionThreads{};
        uint32_t completionThreadCount = 1;
        completionThreads[0] = resCtx.threads[0];
        for (uint32_t crossIndex = 0; crossIndex < crossChannelCount; ++crossIndex) {
            const uint32_t channelIndex = crossChannelIndices[crossIndex];
            if (channelIndex != 0) {
                completionThreads[completionThreadCount++] = resCtx.threads[channelIndex];
            }
        }
        for (uint32_t i = 1; i < completionThreadCount; ++i) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(completionThreads[0],
                CROSS_END_THREAD_NOTIFY_BASE + i - 1U, CUSTOM_TIMEOUT)));
        }
        for (uint32_t i = 1; i < completionThreadCount; ++i) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(completionThreads[i],
                completionThreads[0], CROSS_END_THREAD_NOTIFY_BASE + i - 1U)));
        }
    } else {
        CHK_RET(EndParallel(resCtx.threads));
    }
    const ThreadHandle worker0 = resCtx.workerThreads[0];
    const ThreadHandle worker1 = resCtx.workerThreads[1];

    const uint32_t sameAccumulatorRank = resCtx.channels[sameChannelIndices[0]].remoteRankIndex;
    const uint32_t sameAccumulatorSlot = CompactSameServerSourceSlot(sameAccumulatorRank, resCtx.rankIndex);
    for (uint32_t shard = 0; shard < CROSS_ACCUMULATOR_LANE_SIZE; ++shard) {
        const ThreadHandle thread = resCtx.workerThreads[shard % PIPELINE_WORKER_COUNT];
        for (uint32_t sameIndex = 0; sameIndex < sameChannelCount; ++sameIndex) {
            const ChannelInfo &channel = resCtx.channels[sameChannelIndices[sameIndex]];
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyWaitOnThread(
                    thread, channel.handle, SAME_SERVER_SHARD_NOTIFY_BASE + shard, CUSTOM_TIMEOUT)));
        }
        const SegmentDesc shardSlice = GetSameServerPipelineShard(localSegment.validBytes, shard);
        const uint64_t byteOffset = shardSlice.offsetBytes;
        CHK_PRT_RET(shardSlice.validBytes == 0 ||
            shardSlice.validBytes % custom_allreduce::DATA_TYPE_SIZE != 0 ||
            byteOffset + shardSlice.validBytes > localSegment.validBytes,
            HCCL_ERROR("Invalid local reduction shard[%u]", shard), HCCL_E_INTERNAL);
        const uint64_t shardCount = shardSlice.validBytes / custom_allreduce::DATA_TYPE_SIZE;
        void *dst = localBuffer + static_cast<uint64_t>(sameAccumulatorSlot) * slotStride + byteOffset;
        for (uint32_t sameIndex = 1; sameIndex < sameChannelCount; ++sameIndex) {
            const uint32_t sourceRank = resCtx.channels[sameChannelIndices[sameIndex]].remoteRankIndex;
            const uint32_t sourceSlot = CompactSameServerSourceSlot(sourceRank, resCtx.rankIndex);
            const void *src = localBuffer + static_cast<uint64_t>(sourceSlot) * slotStride + byteOffset;
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
                thread, dst, src, shardCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
        }
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
            thread, dst, localInput + byteOffset, shardCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    }

    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(resCtx.threads[0], worker0, 0)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(resCtx.threads[0], worker1, 0)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(worker0, 0, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(worker1, 0, CUSTOM_TIMEOUT)));

    const uint64_t crossAccumulatorOffset
        = static_cast<uint64_t>(COMPACT_CROSS_ACCUMULATOR_SLOT) * slotStride;
    uint8_t *output = static_cast<uint8_t *>(param.outputPtr) + localSegment.offsetBytes;
    for (uint32_t shard = 0; shard < CROSS_ACCUMULATOR_LANE_SIZE; ++shard) {
        const ThreadHandle thread = resCtx.workerThreads[shard % PIPELINE_WORKER_COUNT];
        const SegmentDesc shardSlice = GetSameServerPipelineShard(localSegment.validBytes, shard);
        const uint64_t byteOffset = shardSlice.offsetBytes;
        const uint64_t shardBytes = shardSlice.validBytes;
        const uint64_t shardCount = shardBytes / custom_allreduce::DATA_TYPE_SIZE;
        void *crossAccumulator = localBuffer + crossAccumulatorOffset + byteOffset;
        const void *laneAccumulator = localBuffer +
            static_cast<uint64_t>(COMPACT_CROSS_ACCUMULATOR_SLOT + 1U) * slotStride + byteOffset;
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
            thread, crossAccumulator, laneAccumulator, shardCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
        const void *sameAccumulator
            = localBuffer + static_cast<uint64_t>(sameAccumulatorSlot) * slotStride + byteOffset;
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
            thread, crossAccumulator, sameAccumulator, shardCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(thread, output + byteOffset, crossAccumulator, shardBytes)));
    }

    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyRecordOnThread(worker1, worker0, WORKER_SECOND_DONE_NOTIFY)));
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyWaitOnThread(worker0, WORKER_SECOND_DONE_NOTIFY, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyRecordOnThread(worker0, resCtx.threads[0], CHANNEL_WORKER_DONE_NOTIFY)));
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyWaitOnThread(resCtx.threads[0], CHANNEL_WORKER_DONE_NOTIFY, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult RunCrossLaneAccumulator(const OpParam &param, const AlgResourceCtx &resCtx,
    const RankSlices &slices, uint64_t tileBytes, uint32_t workerCount)
{
    const uint64_t totalBytes = param.count * custom_allreduce::DATA_TYPE_SIZE;
    const uint64_t roundCount = GetRoundCount(slices, tileBytes);
    for (uint64_t round = 0; round < roundCount; ++round) {
        const RoundSegments segments = BuildRoundSegments(slices, round, roundCount, tileBytes);
        const SegmentDesc &localSegment = segments[resCtx.rankIndex];
        const uint64_t localStride
            = custom_allreduce::AlignUp(localSegment.validBytes, custom_allreduce::PREFERRED_ALIGN);
        CHK_RET(ExchangeAndReduceCrossLaneAccumulator(
            param, resCtx, segments, localSegment, localStride, workerCount));
        if (totalBytes == LARGE_400M4B_BYTES) {
            CHK_RET(FusedReleaseAllGatherAndRoundCompletion(
                param, resCtx, segments, COMPACT_CROSS_ACCUMULATOR_SLOT));
        } else {
            CHK_RET(ReleaseReduceScatterSlots(resCtx));
            CHK_RET(AllGatherRoundFromExtraAccumulator(
                param, resCtx, segments, COMPACT_CROSS_ACCUMULATOR_SLOT));
            CHK_RET(SynchronizeRoundCompletion(resCtx));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RunOneShot16(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t totalBytes, uint32_t workerCount)
{
    const uint64_t slotStride = custom_allreduce::AlignUp(totalBytes, custom_allreduce::PREFERRED_ALIGN);
    CHK_RET(ValidateSlotLayout(resCtx.localBuffer.size, slotStride, resCtx.rankIndex));
    for (const ChannelInfo &channel : resCtx.channels) {
        CHK_RET(ValidateSlotLayout(channel.remoteCclMem.size, slotStride, channel.remoteRankIndex));
    }

    RoundSegments segments{};
    for (SegmentDesc &segment : segments) {
        segment.validBytes = totalBytes;
    }
    CHK_RET(ExchangeContributions(param, resCtx, segments));
    CHK_RET(ReduceSourcesInRankOrder(param, resCtx, SegmentDesc{0, totalBytes}, slotStride, workerCount));
    CHK_RET(ReleaseReduceScatterSlots(resCtx));
    return HCCL_SUCCESS;
}

HcclResult RunDirectRsAg16(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t totalBytes, custom_allreduce::AlgorithmType algorithm,
    uint32_t workerCount)
{
    const RankSlices slices = BuildRankSlices(totalBytes);
    uint64_t coveredBytes = 0;
    for (const RankSlice &slice : slices) {
        CHK_PRT_RET(slice.offsetBytes != coveredBytes || slice.validBytes % custom_allreduce::DATA_TYPE_SIZE != 0,
            HCCL_ERROR("Invalid rank slice at offset[%llu] with bytes[%llu]",
                static_cast<unsigned long long>(slice.offsetBytes), static_cast<unsigned long long>(slice.validBytes)),
            HCCL_E_INTERNAL);
        coveredBytes += slice.validBytes;
    }
    CHK_PRT_RET(coveredBytes != totalBytes,
        HCCL_ERROR("Rank slices cover[%llu] bytes, expected[%llu]", static_cast<unsigned long long>(coveredBytes),
            static_cast<unsigned long long>(totalBytes)),
        HCCL_E_INTERNAL);

    const uint64_t tileBytes = SelectTileBytes(algorithm, resCtx, slices, totalBytes);
    CHK_PRT_RET(tileBytes == 0,
        HCCL_ERROR("No feasible tile for algorithm[%u] and CCL buffer[%llu]", static_cast<uint32_t>(algorithm),
            static_cast<unsigned long long>(resCtx.localBuffer.size)),
        HCCL_E_INTERNAL);
    const uint64_t roundCount = GetRoundCount(slices, tileBytes);
    const uint32_t effectiveWorkerCount = workerCount == 0
        ? (totalBytes <= custom_allreduce::SMALL_THRESHOLD_BYTES ? 1U :
            (IsCrossLaneAccumulatorMessage(totalBytes) ? PIPELINE_WORKER_COUNT : 4U))
        : workerCount;

    if (ShouldUseCrossLaneAccumulator(param, resCtx, totalBytes, tileBytes, effectiveWorkerCount)) {
        return RunCrossLaneAccumulator(param, resCtx, slices, tileBytes, effectiveWorkerCount);
    }

    for (uint64_t round = 0; round < roundCount; ++round) {
        const RoundSegments segments = BuildRoundSegments(slices, round, roundCount, tileBytes);
        const SegmentDesc &localSegment = segments[resCtx.rankIndex];
        const uint64_t localStride
            = custom_allreduce::AlignUp(localSegment.validBytes, custom_allreduce::PREFERRED_ALIGN);
        CHK_RET(ExchangeContributions(param, resCtx, segments));
        CHK_RET(ReduceSourcesInRankOrder(param, resCtx, localSegment, localStride, effectiveWorkerCount));
        CHK_RET(PublishOwnerResult(param, resCtx, localSegment));
        CHK_RET(ReleaseReduceScatterSlots(resCtx));
        CHK_RET(AllGatherRound(param, resCtx, segments));
        CHK_RET(SynchronizeRoundCompletion(resCtx));
    }
    return HCCL_SUCCESS;
}
} // namespace

namespace ops_hccl {
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    CHK_PRT_RET(param.dataType != HCCL_DATA_TYPE_FP32 || param.reduceType != HCCL_REDUCE_SUM,
        HCCL_ERROR("ExecOp only supports FP32 SUM"), HCCL_E_PARA);
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / custom_allreduce::DATA_TYPE_SIZE,
        HCCL_ERROR("Element count overflows byte calculation"), HCCL_E_PARA);
    const uint64_t totalBytes = param.count * custom_allreduce::DATA_TYPE_SIZE;
    if (param.rankSize == custom_allreduce::RANK_SIZE && totalBytes == CROSS_BFLY_BYTES) {
        CHK_RET(ValidateCompact512KResources(param, resCtx));
        CHK_PRT_RET(!ShouldUseTwoServerCrossButterfly512K(param, resCtx),
            HCCL_ERROR("Compact 512KiB butterfly resources are incomplete"), HCCL_E_INTERNAL);
        return RunTwoServerCrossButterfly512K(param, resCtx);
    }
    CHK_RET(ValidateResources(resCtx));

    const custom_allreduce::AlgorithmType algorithm = custom_allreduce::DecodeAlgorithm(param.root);
    const uint32_t configuredWorkerCount = custom_allreduce::DecodeWorkerCount(param.root);
    CHK_PRT_RET(configuredWorkerCount > resCtx.threads.size(),
        HCCL_ERROR("Invalid reduce worker count[%u]", configuredWorkerCount), HCCL_E_PARA);
    if (algorithm == custom_allreduce::AlgorithmType::OneShot16) {
        CHK_PRT_RET(totalBytes > custom_allreduce::SMALL_THRESHOLD_BYTES,
            HCCL_ERROR("OneShot16 received oversized message[%llu]", static_cast<unsigned long long>(totalBytes)),
            HCCL_E_PARA);
        const uint32_t effectiveWorkerCount = configuredWorkerCount == 0 ? 1U : configuredWorkerCount;
        return RunOneShot16(param, resCtx, totalBytes, effectiveWorkerCount);
    }
    return RunDirectRsAg16(param, resCtx, totalBytes, algorithm, configuredWorkerCount);
}
} // namespace ops_hccl
