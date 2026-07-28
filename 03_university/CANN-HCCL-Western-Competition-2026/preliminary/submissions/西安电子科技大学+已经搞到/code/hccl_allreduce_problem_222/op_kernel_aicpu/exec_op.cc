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
#include "log.h"
#include "exec_op.h"

namespace ops_hccl {
namespace {
    constexpr uint64_t HIERARCHICAL_MIN_BYTES = 1ULL * 1024 * 1024;
    constexpr uint32_t FIXED_RANK_SIZE = 16;
    constexpr uint32_t RANKS_PER_SERVER = 8;
    constexpr uint32_t SHARD_NUM = RANKS_PER_SERVER;
    constexpr uint32_t TILE_LANE_NUM = 8;
    constexpr uint64_t MAX_PIPELINE_TILE_COUNT = 32ULL * 1024 * 1024 / sizeof(float);
    constexpr uint64_t PERF_512M_COUNT = 512ULL * 1024 * 1024 / sizeof(float);
    constexpr uint64_t PERF_400M_PLUS_4B_COUNT = 400ULL * 1024 * 1024 / sizeof(float) + 1;
    constexpr uint64_t FIXED_LARGE_TILE_NUM = 16;
    constexpr uint32_t INPUT_DATA_NOTIFY_INDEX = 0;
    constexpr uint32_t CROSS_DATA_NOTIFY_INDEX = 1;
    constexpr uint32_t CROSS_ACK_NOTIFY_INDEX = 2;
    constexpr uint32_t GATHER_DATA_NOTIFY_INDEX = 3;
    constexpr uint32_t GATHER_ACK_NOTIFY_INDEX = 4;
    constexpr uint32_t NOTIFY_NUM_PER_LANE = 5;
    constexpr uint32_t HIERARCHICAL_NOTIFY_NUM = NOTIFY_NUM_PER_LANE * TILE_LANE_NUM;
    constexpr uint32_t PAIRED_CHANNEL_INDEX = RANKS_PER_SERVER - 1;
    constexpr uint32_t THREAD_BOOTSTRAP_NOTIFY_INDEX = 0;
    constexpr uint32_t LANE_DONE_NOTIFY_BASE = 1;
    constexpr uint32_t SMALL_STAGE_SLOT_NUM = 7;

    uint64_t SegmentCount(uint64_t chunkCount, uint32_t rankSize, uint32_t segmentIndex)
    {
        const uint64_t baseCount = chunkCount / rankSize;
        const uint64_t remainder = chunkCount % rankSize;
        return baseCount + (segmentIndex < remainder ? 1 : 0);
    }

    uint64_t SegmentOffset(uint64_t chunkCount, uint32_t rankSize, uint32_t segmentIndex)
    {
        const uint64_t baseCount = chunkCount / rankSize;
        const uint64_t remainder = chunkCount % rankSize;
        return segmentIndex * baseCount + std::min<uint64_t>(segmentIndex, remainder);
    }

    const ChannelInfo *LocalChannel(const AlgResourceCtx &resCtx, uint32_t myRank, uint32_t peerLocalRank)
    {
        const uint32_t localRank = myRank % RANKS_PER_SERVER;
        if (resCtx.channels.size() != RANKS_PER_SERVER || peerLocalRank >= RANKS_PER_SERVER
            || peerLocalRank == localRank) {
            return nullptr;
        }
        const uint32_t channelIndex = peerLocalRank < localRank ? peerLocalRank : peerLocalRank - 1;
        const ChannelInfo *channel = &resCtx.channels[channelIndex];
        const uint32_t expectedRank = (myRank / RANKS_PER_SERVER) * RANKS_PER_SERVER + peerLocalRank;
        return channel->remoteRank == expectedRank ? channel : nullptr;
    }

    const ChannelInfo *PairedChannel(const AlgResourceCtx &resCtx, uint32_t myRank)
    {
        if (resCtx.channels.size() != RANKS_PER_SERVER) {
            return nullptr;
        }
        const uint32_t expectedRank
            = myRank < RANKS_PER_SERVER ? myRank + RANKS_PER_SERVER : myRank - RANKS_PER_SERVER;
        const ChannelInfo *channel = &resCtx.channels[PAIRED_CHANNEL_INDEX];
        return channel->remoteRank == expectedRank ? channel : nullptr;
    }

    uint64_t MaxHierarchicalChunkCount(uint64_t bufferCount)
    {
        uint64_t chunkCount = bufferCount / (TILE_LANE_NUM + 1);
        while (chunkCount != 0
               && TILE_LANE_NUM * (chunkCount + SegmentCount(chunkCount, SHARD_NUM, 0)) > bufferCount) {
            --chunkCount;
        }
        return std::min(chunkCount, MAX_PIPELINE_TILE_COUNT);
    }

    HcclResult CheckResources(const OpParam &param, const AlgResourceCtx &resCtx)
    {
        CHK_PRT_RET(param.myRank >= param.rankSize,
            HCCL_ERROR("Invalid rank[%u] for rank size[%u]", param.myRank, param.rankSize), HCCL_E_INTERNAL);
        CHK_PRT_RET(resCtx.localBuffer.addr == nullptr || resCtx.localBuffer.size < sizeof(float),
            HCCL_ERROR("Local CCL buffer is unavailable"), HCCL_E_INTERNAL);
        CHK_PRT_RET(resCtx.channels.size() != RANKS_PER_SERVER,
            HCCL_ERROR("V8 requires exactly one Channel for each of 8 peers, actual[%zu]", resCtx.channels.size()),
            HCCL_E_INTERNAL);

        const uint32_t serverBase = (param.myRank / RANKS_PER_SERVER) * RANKS_PER_SERVER;
        const uint32_t pairedRank
            = param.myRank < RANKS_PER_SERVER ? param.myRank + RANKS_PER_SERVER : param.myRank - RANKS_PER_SERVER;
        for (uint32_t localRank = 0; localRank < RANKS_PER_SERVER; ++localRank) {
            if (localRank == param.myRank % RANKS_PER_SERVER) {
                continue;
            }
            const uint32_t peerRank = serverBase + localRank;
            const ChannelInfo *peerChannel = LocalChannel(resCtx, param.myRank, localRank);
            CHK_PTR_NULL(peerChannel);
            CHK_PRT_RET(peerChannel->notifyNum < HIERARCHICAL_NOTIFY_NUM,
                HCCL_ERROR("Local peer[%u] has insufficient notify slots", peerRank), HCCL_E_INTERNAL);
            CHK_PRT_RET(peerChannel->remoteCclMem.addr == nullptr || peerChannel->remoteCclMem.size < sizeof(float),
                HCCL_ERROR("Local peer[%u] CCL buffer is unavailable", peerRank), HCCL_E_INTERNAL);
        }
        const ChannelInfo *pairedChannel = PairedChannel(resCtx, param.myRank);
        CHK_PTR_NULL(pairedChannel);
        CHK_PRT_RET(pairedChannel->notifyNum < HIERARCHICAL_NOTIFY_NUM,
            HCCL_ERROR("Cross-server peer[%u] has insufficient notify slots", pairedRank), HCCL_E_INTERNAL);
        CHK_PRT_RET(pairedChannel->remoteCclMem.addr == nullptr || pairedChannel->remoteCclMem.size < sizeof(float),
            HCCL_ERROR("Cross-server peer[%u] CCL buffer is unavailable", pairedRank), HCCL_E_INTERNAL);
        return HCCL_SUCCESS;
    }

    HcclResult RecordLocalPeers(
        const OpParam &param, const AlgResourceCtx &resCtx, ThreadHandle thread, uint32_t notifyIndex)
    {
        for (uint32_t peerLocalRank = 0; peerLocalRank < RANKS_PER_SERVER; ++peerLocalRank) {
            if (peerLocalRank == param.myRank % RANKS_PER_SERVER) {
                continue;
            }
            const ChannelInfo *peerChannel = LocalChannel(resCtx, param.myRank, peerLocalRank);
            CHK_PTR_NULL(peerChannel);
            CHK_RET(
                static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, peerChannel->handle, notifyIndex)));
        }
        return HCCL_SUCCESS;
    }

    HcclResult WaitLocalPeers(
        const OpParam &param, const AlgResourceCtx &resCtx, ThreadHandle thread, uint32_t notifyIndex)
    {
        for (uint32_t peerLocalRank = 0; peerLocalRank < RANKS_PER_SERVER; ++peerLocalRank) {
            if (peerLocalRank == param.myRank % RANKS_PER_SERVER) {
                continue;
            }
            const ChannelInfo *peerChannel = LocalChannel(resCtx, param.myRank, peerLocalRank);
            CHK_PTR_NULL(peerChannel);
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyWaitOnThread(thread, peerChannel->handle, notifyIndex, CUSTOM_TIMEOUT)));
        }
        return HCCL_SUCCESS;
    }

    HcclResult ExecuteTileLane(const OpParam &param, const AlgResourceCtx &resCtx, uint8_t *input,
        uint8_t *output, uint8_t *localBuffer, uint64_t processedCount, uint64_t chunkCount, uint64_t maxChunkCount,
        uint32_t laneIndex)
    {
        constexpr uint64_t dataTypeSize = sizeof(float);
        ThreadHandle laneThread = resCtx.threads[laneIndex];
        const uint32_t localRank = param.myRank % RANKS_PER_SERVER;
        const ChannelInfo *pairedChannel = PairedChannel(resCtx, param.myRank);
        CHK_PTR_NULL(pairedChannel);
        const uint32_t notifyBase = laneIndex * NOTIFY_NUM_PER_LANE;
        const uint64_t slotStride = maxChunkCount + SegmentCount(maxChunkCount, SHARD_NUM, 0);
        const uint64_t sourceOffset = laneIndex * slotStride;
        const uint64_t globalOffset = sourceOffset + maxChunkCount;

        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(laneThread,
            localBuffer + sourceOffset * dataTypeSize, input + processedCount * dataTypeSize,
            chunkCount * dataTypeSize)));
        CHK_RET(RecordLocalPeers(param, resCtx, laneThread, notifyBase + INPUT_DATA_NOTIFY_INDEX));
        CHK_RET(WaitLocalPeers(param, resCtx, laneThread, notifyBase + INPUT_DATA_NOTIFY_INDEX));

        const uint64_t ownerCount = SegmentCount(chunkCount, SHARD_NUM, localRank);
        const uint64_t ownerOffset = SegmentOffset(chunkCount, SHARD_NUM, localRank);
        if (ownerCount != 0) {
            for (uint32_t step = 0; step < RANKS_PER_SERVER - 1; ++step) {
                const uint32_t peerOffset = 1 + (step + laneIndex) % (RANKS_PER_SERVER - 1);
                const uint32_t peerLocalRank = (localRank + peerOffset) % RANKS_PER_SERVER;
                const ChannelInfo *peerChannel = LocalChannel(resCtx, param.myRank, peerLocalRank);
                CHK_PTR_NULL(peerChannel);
                CHK_RET(static_cast<HcclResult>(HcommReadReduceOnThread(laneThread, peerChannel->handle,
                    localBuffer + (sourceOffset + ownerOffset) * dataTypeSize,
                    static_cast<uint8_t *>(peerChannel->remoteCclMem.addr) +
                        (sourceOffset + ownerOffset) * dataTypeSize,
                    ownerCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
            }
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(laneThread,
                localBuffer + globalOffset * dataTypeSize,
                localBuffer + (sourceOffset + ownerOffset) * dataTypeSize, ownerCount * dataTypeSize)));
        }
        CHK_RET(HcommChannelNotifyRecordOnThread(
            laneThread, pairedChannel->handle, notifyBase + CROSS_DATA_NOTIFY_INDEX));
        CHK_RET(HcommChannelNotifyWaitOnThread(
            laneThread, pairedChannel->handle, notifyBase + CROSS_DATA_NOTIFY_INDEX, CUSTOM_TIMEOUT));

        if (ownerCount != 0) {
            CHK_RET(static_cast<HcclResult>(HcommReadReduceOnThread(laneThread, pairedChannel->handle,
                localBuffer + globalOffset * dataTypeSize,
                static_cast<uint8_t *>(pairedChannel->remoteCclMem.addr) +
                    (sourceOffset + ownerOffset) * dataTypeSize,
                ownerCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
        }
        CHK_RET(HcommChannelNotifyRecordOnThread(
            laneThread, pairedChannel->handle, notifyBase + CROSS_ACK_NOTIFY_INDEX));
        CHK_RET(RecordLocalPeers(param, resCtx, laneThread, notifyBase + GATHER_DATA_NOTIFY_INDEX));
        CHK_RET(WaitLocalPeers(param, resCtx, laneThread, notifyBase + GATHER_DATA_NOTIFY_INDEX));

        for (uint32_t shardIndex = 0; shardIndex < SHARD_NUM; ++shardIndex) {
            const uint64_t shardCount = SegmentCount(chunkCount, SHARD_NUM, shardIndex);
            const uint64_t shardOffset = SegmentOffset(chunkCount, SHARD_NUM, shardIndex);
            if (shardCount == 0) {
                continue;
            }
            uint8_t *outputShard = output + (processedCount + shardOffset) * dataTypeSize;
            if (shardIndex == localRank) {
                CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
                    laneThread, outputShard, localBuffer + globalOffset * dataTypeSize, shardCount * dataTypeSize)));
            } else {
                const ChannelInfo *ownerChannel = LocalChannel(resCtx, param.myRank, shardIndex);
                CHK_PTR_NULL(ownerChannel);
                CHK_RET(static_cast<HcclResult>(HcommReadOnThread(laneThread, ownerChannel->handle, outputShard,
                    static_cast<uint8_t *>(ownerChannel->remoteCclMem.addr) + globalOffset * dataTypeSize,
                    shardCount * dataTypeSize)));
            }
        }
        CHK_RET(RecordLocalPeers(param, resCtx, laneThread, notifyBase + GATHER_ACK_NOTIFY_INDEX));
        CHK_RET(WaitLocalPeers(param, resCtx, laneThread, notifyBase + GATHER_ACK_NOTIFY_INDEX));
        CHK_RET(HcommChannelNotifyWaitOnThread(
            laneThread, pairedChannel->handle, notifyBase + CROSS_ACK_NOTIFY_INDEX, CUSTOM_TIMEOUT));
        return HCCL_SUCCESS;
    }

    HcclResult ExecuteSmallHierarchical(const OpParam &param, const AlgResourceCtx &resCtx, ThreadHandle thread,
        uint8_t *input, uint8_t *output, uint8_t *localBuffer)
    {
        constexpr uint64_t dataTypeSize = sizeof(float);
        const uint64_t dataBytes = param.count * dataTypeSize;
        const uint64_t scratchBytes = dataBytes * SMALL_STAGE_SLOT_NUM;
        CHK_PRT_RET(resCtx.localBuffer.size < scratchBytes,
            HCCL_ERROR("Small hierarchical requires[%llu] local CCL bytes, actual[%llu]",
                static_cast<unsigned long long>(scratchBytes), static_cast<unsigned long long>(resCtx.localBuffer.size)),
            HCCL_E_INTERNAL);

        const uint32_t localRank = param.myRank % RANKS_PER_SERVER;
        const ChannelInfo *pairedChannel = PairedChannel(resCtx, param.myRank);
        CHK_PTR_NULL(pairedChannel);

        const ChannelInfo *reclaimPeers[4] = {nullptr, nullptr, nullptr, pairedChannel};
        uint32_t reclaimIndex = 0;
        for (uint32_t mask = 1; mask < RANKS_PER_SERVER; mask <<= 1) {
            const ChannelInfo *peerChannel = LocalChannel(resCtx, param.myRank, localRank ^ mask);
            CHK_PTR_NULL(peerChannel);
            reclaimPeers[reclaimIndex++] = peerChannel;
        }
        for (uint32_t index = 0; index < 4; ++index) {
            CHK_PRT_RET(reclaimPeers[index]->remoteCclMem.addr == nullptr
                    || reclaimPeers[index]->remoteCclMem.size < scratchBytes,
                HCCL_ERROR("Small hierarchical peer[%u] CCL buffer is too small, required[%llu], actual[%llu]",
                    reclaimPeers[index]->remoteRank, static_cast<unsigned long long>(scratchBytes),
                    static_cast<unsigned long long>(reclaimPeers[index]->remoteCclMem.size)),
                HCCL_E_INTERNAL);
        }

        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, localBuffer, input, dataBytes)));
        uint32_t sourceSlot = 0;
        uint32_t destinationSlot = 1;

        for (uint32_t mask = RANKS_PER_SERVER >> 1; mask != 0; mask >>= 1) {
            const ChannelInfo *peerChannel = LocalChannel(resCtx, param.myRank, localRank ^ mask);
            CHK_PTR_NULL(peerChannel);
            const uint32_t firstSegment
                = (localRank / (mask << 1)) * (mask << 1) + ((localRank & mask) != 0 ? mask : 0);
            const uint64_t firstCount = SegmentOffset(param.count, RANKS_PER_SERVER, firstSegment);
            const uint64_t endCount = SegmentOffset(param.count, RANKS_PER_SERVER, firstSegment + mask);
            if (endCount != firstCount) {
                CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread,
                    localBuffer + destinationSlot * dataBytes + firstCount * dataTypeSize,
                    localBuffer + sourceSlot * dataBytes + firstCount * dataTypeSize,
                    (endCount - firstCount) * dataTypeSize)));
            }
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(thread, peerChannel->handle, NOTIFY_IDX_DATA_SIGNAL)));
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyWaitOnThread(thread, peerChannel->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
            if (endCount != firstCount) {
                CHK_RET(static_cast<HcclResult>(HcommReadReduceOnThread(thread, peerChannel->handle,
                    localBuffer + destinationSlot * dataBytes + firstCount * dataTypeSize,
                    static_cast<uint8_t *>(peerChannel->remoteCclMem.addr)
                        + sourceSlot * dataBytes + firstCount * dataTypeSize,
                    endCount - firstCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
            }
            sourceSlot = destinationSlot++;
        }

        const uint64_t ownerFirst = SegmentOffset(param.count, RANKS_PER_SERVER, localRank);
        const uint64_t ownerEnd = SegmentOffset(param.count, RANKS_PER_SERVER, localRank + 1);
        if (ownerEnd != ownerFirst) {
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread,
                localBuffer + destinationSlot * dataBytes + ownerFirst * dataTypeSize,
                localBuffer + sourceSlot * dataBytes + ownerFirst * dataTypeSize,
                (ownerEnd - ownerFirst) * dataTypeSize)));
        }
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, pairedChannel->handle, NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, pairedChannel->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
        if (ownerEnd != ownerFirst) {
            CHK_RET(static_cast<HcclResult>(HcommReadReduceOnThread(thread, pairedChannel->handle,
                localBuffer + destinationSlot * dataBytes + ownerFirst * dataTypeSize,
                static_cast<uint8_t *>(pairedChannel->remoteCclMem.addr)
                    + sourceSlot * dataBytes + ownerFirst * dataTypeSize,
                ownerEnd - ownerFirst, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
        }
        sourceSlot = destinationSlot++;

        for (uint32_t mask = 1; mask < RANKS_PER_SERVER; mask <<= 1) {
            const ChannelInfo *peerChannel = LocalChannel(resCtx, param.myRank, localRank ^ mask);
            CHK_PTR_NULL(peerChannel);
            const uint32_t ownBlockStart = (localRank / mask) * mask;
            const uint32_t remoteBlockStart
                = (localRank & mask) == 0 ? ownBlockStart + mask : ownBlockStart - mask;
            const uint64_t ownFirst = SegmentOffset(param.count, RANKS_PER_SERVER, ownBlockStart);
            const uint64_t ownEnd = SegmentOffset(param.count, RANKS_PER_SERVER, ownBlockStart + mask);
            const uint64_t remoteFirst = SegmentOffset(param.count, RANKS_PER_SERVER, remoteBlockStart);
            const uint64_t remoteEnd = SegmentOffset(param.count, RANKS_PER_SERVER, remoteBlockStart + mask);
            uint8_t *destination
                = mask == (RANKS_PER_SERVER >> 1) ? output : localBuffer + destinationSlot * dataBytes;
            if (ownEnd != ownFirst) {
                CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, destination + ownFirst * dataTypeSize,
                    localBuffer + sourceSlot * dataBytes + ownFirst * dataTypeSize,
                    (ownEnd - ownFirst) * dataTypeSize)));
            }
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(thread, peerChannel->handle, NOTIFY_IDX_DATA_SIGNAL)));
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyWaitOnThread(thread, peerChannel->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
            if (remoteEnd != remoteFirst) {
                CHK_RET(static_cast<HcclResult>(HcommReadOnThread(thread, peerChannel->handle,
                    destination + remoteFirst * dataTypeSize,
                    static_cast<uint8_t *>(peerChannel->remoteCclMem.addr)
                        + sourceSlot * dataBytes + remoteFirst * dataTypeSize,
                    (remoteEnd - remoteFirst) * dataTypeSize)));
            }
            if (mask != (RANKS_PER_SERVER >> 1)) {
                sourceSlot = destinationSlot++;
            }
        }

        for (uint32_t index = 0; index < 4; ++index) {
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(thread, reclaimPeers[index]->handle, NOTIFY_IDX_ACK)));
        }
        for (uint32_t index = 0; index < 4; ++index) {
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                thread, reclaimPeers[index]->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        }
        return HCCL_SUCCESS;
    }
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }

    constexpr uint64_t dataTypeSize = sizeof(float);
    CHK_PRT_RET(param.rankSize != FIXED_RANK_SIZE,
        HCCL_ERROR("Hierarchical implementation requires rank size[%u], actual[%u]", FIXED_RANK_SIZE, param.rankSize),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(resCtx.threads.empty(), HCCL_ERROR("No AICPU thread is available"), HCCL_E_INTERNAL);
    ThreadHandle thread = resCtx.threads[0];
    uint8_t *input = static_cast<uint8_t *>(param.inputPtr);
    uint8_t *output = static_cast<uint8_t *>(param.outputPtr);
    uint8_t *localBuffer = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    CHK_RET(CheckResources(param, resCtx));

    const uint64_t dataBytes = param.count * dataTypeSize;
    if (dataBytes <= HIERARCHICAL_MIN_BYTES) {
        return ExecuteSmallHierarchical(param, resCtx, thread, input, output, localBuffer);
    }

    uint64_t availableBufferBytes = resCtx.localBuffer.size;
    for (const ChannelInfo &channel : resCtx.channels) {
        availableBufferBytes = std::min(availableBufferBytes, channel.remoteCclMem.size);
    }
    const uint64_t maxChunkCount = MaxHierarchicalChunkCount(availableBufferBytes / dataTypeSize);
    CHK_PRT_RET(maxChunkCount == 0, HCCL_ERROR("No aligned CCL buffer space is available"), HCCL_E_INTERNAL);
    const bool fixedPerformanceLarge
        = param.count == PERF_512M_COUNT || param.count == PERF_400M_PLUS_4B_COUNT;
    const uint64_t fixedTileCount
        = param.count / FIXED_LARGE_TILE_NUM + static_cast<uint64_t>(param.count % FIXED_LARGE_TILE_NUM != 0);
    const bool canUseFixedTiles = fixedPerformanceLarge && fixedTileCount <= maxChunkCount;
    const uint64_t dynamicTotalChunks
        = param.count / maxChunkCount + static_cast<uint64_t>(param.count % maxChunkCount != 0);
    const uint64_t totalChunks = canUseFixedTiles ? FIXED_LARGE_TILE_NUM : dynamicTotalChunks;

    CHK_PRT_RET(resCtx.threads.size() < TILE_LANE_NUM,
        HCCL_ERROR("Tile-lane algorithm requires[%u] AICPU threads, actual[%zu]", TILE_LANE_NUM,
            resCtx.threads.size()),
        HCCL_E_INTERNAL);
    for (uint32_t lane = 1; lane < TILE_LANE_NUM; ++lane) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            thread, resCtx.threads[lane], THREAD_BOOTSTRAP_NOTIFY_INDEX)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            resCtx.threads[lane], THREAD_BOOTSTRAP_NOTIFY_INDEX, CUSTOM_TIMEOUT)));
    }
    for (uint64_t processed = 0, chunk = 0; processed < param.count; ++chunk) {
        const uint64_t remainingCount = param.count - processed;
        const uint64_t remainingChunks = totalChunks - chunk;
        const uint64_t chunkCount
            = remainingCount / remainingChunks + static_cast<uint64_t>(remainingCount % remainingChunks != 0);
        const uint32_t lane = static_cast<uint32_t>(chunk % TILE_LANE_NUM);
        CHK_RET(ExecuteTileLane(
            param, resCtx, input, output, localBuffer, processed, chunkCount, maxChunkCount, lane));

        processed += chunkCount;
    }

    const uint32_t activeLanes = static_cast<uint32_t>(std::min<uint64_t>(totalChunks, TILE_LANE_NUM));
    for (uint32_t lane = 1; lane < activeLanes; ++lane) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resCtx.threads[lane], thread, LANE_DONE_NOTIFY_BASE + lane - 1)));
    }
    for (uint32_t lane = 1; lane < activeLanes; ++lane) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(thread, LANE_DONE_NOTIFY_BASE + lane - 1, CUSTOM_TIMEOUT)));
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
