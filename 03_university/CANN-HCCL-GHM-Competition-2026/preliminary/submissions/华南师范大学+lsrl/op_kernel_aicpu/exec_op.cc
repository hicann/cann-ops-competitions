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

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace {
constexpr uint32_t SERVER_RANK_SIZE = 8;
constexpr uint32_t MAIN_THREAD_INDEX = 0;
constexpr uint32_t WORKER_THREAD_FIRST_INDEX = MAIN_THREAD_INDEX + 1;
constexpr uint32_t WORKER_THREAD_NUM = SERVER_RANK_SIZE - 1;
constexpr uint32_t CROSS_PIPELINE_THREAD_INDEX = WORKER_THREAD_FIRST_INDEX + WORKER_THREAD_NUM;
constexpr uint32_t PACK_THREAD_FIRST_INDEX = CROSS_PIPELINE_THREAD_INDEX + 1;
constexpr uint32_t PACK_THREAD_NUM = SERVER_RANK_SIZE;
constexpr uint64_t SMALL_MESSAGE_BYTES = 1024 * 1024;
constexpr uint64_t PIPELINE_MIN_MESSAGE_BYTES = 8 * 1024 * 1024;
constexpr uint64_t PIPELINE_SLICE_BYTES = 8 * 1024 * 1024;
constexpr uint64_t PIPELINE_SLICE_COUNT = PIPELINE_SLICE_BYTES / sizeof(float);
constexpr uint64_t PARALLEL_PACK_MIN_SLICE_BYTES = 4 * 1024;
constexpr uint32_t TREE_PIPELINE_SLICE_NUM = SERVER_RANK_SIZE - 1;
constexpr uint32_t TREE_PIPELINE_BANK_NUM = TREE_PIPELINE_SLICE_NUM + 1;
constexpr uint32_t TREE_NOTIFY_NUM_PER_SLICE = 2;

// Cross-server channel notification indices.
constexpr uint32_t NOTIFY_CROSS_BUFFER_READY = 0;
constexpr uint32_t NOTIFY_CROSS_REDUCE_DONE = 1;
constexpr uint32_t NOTIFY_PAIR_SUM_RELEASED = 2;

// Intra-server channel notification indices.
constexpr uint32_t NOTIFY_RECEIVE_SLOT_READY = 0;
constexpr uint32_t NOTIFY_LOCAL_DATA_READY = 1;
constexpr uint32_t NOTIFY_LOCAL_DATA_CONSUMED = 2;

// Notify indices 0..7 drive packing and Mesh workers.  Index 8 is dedicated
// to the cross-server pipeline thread.
constexpr uint32_t NOTIFY_PIPELINE_STAGE_READY = SERVER_RANK_SIZE;

constexpr uint32_t PAIR_SUM_SLOT_NUM = SERVER_RANK_SIZE;
constexpr uint32_t CROSS_SEND_SLOT_BASE = PAIR_SUM_SLOT_NUM;
constexpr uint32_t CROSS_SEND_SLOT_NUM = SERVER_RANK_SIZE;
constexpr uint32_t LOCAL_BUFFER_SLOT_NUM = PAIR_SUM_SLOT_NUM + CROSS_SEND_SLOT_NUM;

static_assert(REDUCE_SCATTER_AICPU_THREAD_NUM == PACK_THREAD_FIRST_INDEX + PACK_THREAD_NUM,
    "ReduceScatter thread layout must reserve control, Mesh, cross-server, and packing threads");
static_assert(REDUCE_SCATTER_THREAD_NOTIFY_NUM > NOTIFY_PIPELINE_STAGE_READY,
    "ReduceScatter requires a dedicated cross-server pipeline notify");
struct SliceDesc {
    uint64_t offsetCount = 0;
    uint64_t count = 0;
    uint64_t bytes = 0;
    uint64_t bufferOffsetBytes = 0;
    uint64_t crossBufferOffsetBytes = 0;
};

const ChannelInfo *FindChannel(const AlgResourceCtx &resCtx, uint32_t localRank, uint32_t remoteRank)
{
    if (remoteRank >= REDUCE_SCATTER_RANK_SIZE || remoteRank == localRank) {
        return nullptr;
    }

    // AcquireChannels stores channels in ascending remote-rank order while
    // omitting the local rank.  Use that layout directly on the AICPU hot path.
    const uint32_t channelIndex = remoteRank < localRank ? remoteRank : remoteRank - 1;
    if (channelIndex < REDUCE_SCATTER_CHANNEL_NUM && resCtx.channels[channelIndex].remoteRank == remoteRank) {
        return &resCtx.channels[channelIndex];
    }

    // Keep a defensive fallback in case a future resource builder changes the order.
    for (const ChannelInfo &channel : resCtx.channels) {
        if (channel.remoteRank == remoteRank) {
            return &channel;
        }
    }
    return nullptr;
}

void *AddOffset(void *base, uint64_t offset)
{
    return static_cast<void *>(static_cast<char *>(base) + offset);
}

uint32_t ReceiveSlot(uint32_t targetLocalIndex, uint32_t sourceLocalIndex)
{
    return sourceLocalIndex < targetLocalIndex ? sourceLocalIndex : sourceLocalIndex - 1;
}

uint32_t PeerLocalIndex(uint32_t peerOrdinal, uint32_t localIndex)
{
    return peerOrdinal < localIndex ? peerOrdinal : peerOrdinal + 1;
}

uint32_t MultiplyByGfPrimitive(uint32_t value)
{
    // GF(2^3), primitive polynomial x^3 + x + 1.  Multiplication by x
    // permutes the seven non-zero three-bit vectors.
    return ((value << 1) & (SERVER_RANK_SIZE - 1)) ^ ((value & (SERVER_RANK_SIZE / 2)) != 0 ? 3U : 0U);
}

uint32_t PhysicalFromLogical(uint32_t logicalIndex, uint32_t layoutId)
{
    if (layoutId == 0) {
        return logicalIndex;
    }

    // The three basis vectors are d, x*d, x^2*d in GF(2^3).  For every
    // non-zero d they form a basis, and across layoutId 1..7 each tree round
    // therefore uses every physical xor link exactly once.
    const uint32_t basis0 = layoutId;
    const uint32_t basis1 = MultiplyByGfPrimitive(basis0);
    const uint32_t basis2 = MultiplyByGfPrimitive(basis1);
    return ((logicalIndex & 4U) != 0 ? basis0 : 0U) ^
        ((logicalIndex & 2U) != 0 ? basis1 : 0U) ^
        ((logicalIndex & 1U) != 0 ? basis2 : 0U);
}

uint32_t LogicalFromPhysical(uint32_t physicalIndex, uint32_t layoutId)
{
    for (uint32_t logicalIndex = 0; logicalIndex < SERVER_RANK_SIZE; ++logicalIndex) {
        if (PhysicalFromLogical(logicalIndex, layoutId) == physicalIndex) {
            return logicalIndex;
        }
    }
    return INVALID_VALUE_RANKID;
}

HcclResult CopyOrReduce(ThreadHandle thread, void *dst, const void *src, uint64_t count, uint64_t bytes,
    HcommDataType dataType, HcommReduceOp reduceOp, bool isFirstOperand)
{
    if (isFirstOperand) {
        CHK_RET(HcommLocalCopyOnThread(thread, dst, src, bytes));
    } else {
        CHK_RET(HcommLocalReduceOnThread(thread, dst, src, count, dataType, reduceOp));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceOutputSegment(const OpParam &param, ThreadHandle thread, uint32_t localIndex, void *pairSum,
    void *crossSend, uint64_t sliceCount, uint64_t sliceBytes, uint64_t outputOffsetBytes,
    uint32_t segmentIndex, HcommDataType dataType, HcommReduceOp reduceOp)
{
    // Split by element rather than by source rank.  Every thread owns a
    // disjoint output interval, while each individual element is still reduced
    // in the fixed source-rank order 0..7.
    const uint64_t baseSegmentCount = sliceCount / SERVER_RANK_SIZE;
    const uint64_t remainder = sliceCount % SERVER_RANK_SIZE;
    const uint64_t segmentCount = baseSegmentCount + (segmentIndex < remainder ? 1 : 0);
    if (segmentCount == 0) {
        return HCCL_SUCCESS;
    }

    const uint64_t segmentOffsetCount = segmentIndex * baseSegmentCount + std::min<uint64_t>(segmentIndex,
        remainder);
    const uint64_t segmentOffsetBytes = segmentOffsetCount * sizeof(float);
    const uint64_t segmentBytes = segmentCount * sizeof(float);
    void *output = AddOffset(param.outputPtr, outputOffsetBytes + segmentOffsetBytes);

    for (uint32_t sourceLocalIndex = 0; sourceLocalIndex < SERVER_RANK_SIZE; ++sourceLocalIndex) {
        const void *source = nullptr;
        if (sourceLocalIndex == localIndex) {
            source = AddOffset(pairSum, localIndex * sliceBytes + segmentOffsetBytes);
        } else {
            const uint32_t receiveSlot = ReceiveSlot(localIndex, sourceLocalIndex);
            source = AddOffset(crossSend, receiveSlot * sliceBytes + segmentOffsetBytes);
        }
        CHK_RET(CopyOrReduce(thread, output, source, segmentCount, segmentBytes, dataType, reduceOp,
            sourceLocalIndex == 0));
    }
    return HCCL_SUCCESS;
}

SliceDesc MakeSlice(uint64_t offsetCount, uint64_t count, uint64_t bufferOffsetBytes)
{
    SliceDesc slice{};
    slice.offsetCount = offsetCount;
    slice.count = count;
    slice.bytes = count * sizeof(float);
    slice.bufferOffsetBytes = bufferOffsetBytes;
    slice.crossBufferOffsetBytes = bufferOffsetBytes + PAIR_SUM_SLOT_NUM * slice.bytes;
    return slice;
}

HcclResult PackInputSlice(const OpParam &param, ThreadHandle thread, uint32_t localServerStart,
    uint32_t remoteServerStart, void *pairSum, void *crossSend, const SliceDesc &slice, uint64_t totalBytes,
    uint32_t layoutId)
{
    const uint64_t offsetBytes = slice.offsetCount * sizeof(float);
    if (layoutId == 0 && offsetBytes == 0 && slice.bytes == totalBytes) {
        // A complete output slice contains eight contiguous blocks for either
        // server.  Coalesce sixteen small copies into two large DMA tasks.
        const uint64_t serverHalfBytes = SERVER_RANK_SIZE * totalBytes;
        CHK_RET(HcommLocalCopyOnThread(thread, pairSum,
            AddOffset(param.inputPtr, localServerStart * totalBytes), serverHalfBytes));
        CHK_RET(HcommLocalCopyOnThread(thread, crossSend,
            AddOffset(param.inputPtr, remoteServerStart * totalBytes), serverHalfBytes));
        return HCCL_SUCCESS;
    }

    for (uint32_t logicalBlockIndex = 0; logicalBlockIndex < SERVER_RANK_SIZE; ++logicalBlockIndex) {
        const uint32_t physicalBlockIndex = PhysicalFromLogical(logicalBlockIndex, layoutId);
        const uint32_t localOutputRank = localServerStart + physicalBlockIndex;
        const uint32_t remoteOutputRank = remoteServerStart + physicalBlockIndex;
        CHK_RET(HcommLocalCopyOnThread(thread, AddOffset(pairSum, logicalBlockIndex * slice.bytes),
            AddOffset(param.inputPtr, localOutputRank * totalBytes + offsetBytes), slice.bytes));
        CHK_RET(HcommLocalCopyOnThread(thread, AddOffset(crossSend, logicalBlockIndex * slice.bytes),
            AddOffset(param.inputPtr, remoteOutputRank * totalBytes + offsetBytes), slice.bytes));
    }
    return HCCL_SUCCESS;
}

HcclResult PackInputSliceParallel(const OpParam &param, const AlgResourceCtx &resCtx, ThreadHandle coordinator,
    uint32_t localServerStart, uint32_t remoteServerStart, void *pairSum, void *crossSend,
    const SliceDesc &slice, uint64_t totalBytes, uint32_t layoutId)
{
    const uint64_t offsetBytes = slice.offsetCount * sizeof(float);
    if (layoutId == 0 && offsetBytes == 0 && slice.bytes == totalBytes) {
        // Two full server halves are already contiguous.  Large coalesced DMA
        // tasks are preferable to splitting them into sixteen smaller tasks.
        return PackInputSlice(param, coordinator, localServerStart, remoteServerStart, pairSum, crossSend, slice,
            totalBytes, layoutId);
    }

    // A sliced server half is strided by totalBytes in the user input.  Give
    // each output-rank block its own thread so the sixteen required copies no
    // longer serialize on the cross-server pipeline thread.
    for (uint32_t blockIndex = 0; blockIndex < PACK_THREAD_NUM; ++blockIndex) {
        const ThreadHandle packThread = resCtx.aicpuThreads[PACK_THREAD_FIRST_INDEX + blockIndex];
        const uint32_t physicalBlockIndex = PhysicalFromLogical(blockIndex, layoutId);
        const uint32_t localOutputRank = localServerStart + physicalBlockIndex;
        const uint32_t remoteOutputRank = remoteServerStart + physicalBlockIndex;

        CHK_RET(HcommThreadNotifyRecordOnThread(coordinator, packThread, blockIndex));
        CHK_RET(HcommThreadNotifyWaitOnThread(packThread, blockIndex, CUSTOM_TIMEOUT));
        CHK_RET(HcommLocalCopyOnThread(packThread, AddOffset(pairSum, blockIndex * slice.bytes),
            AddOffset(param.inputPtr, localOutputRank * totalBytes + offsetBytes), slice.bytes));
        CHK_RET(HcommLocalCopyOnThread(packThread, AddOffset(crossSend, blockIndex * slice.bytes),
            AddOffset(param.inputPtr, remoteOutputRank * totalBytes + offsetBytes), slice.bytes));
        CHK_RET(HcommThreadNotifyRecordOnThread(packThread, coordinator, blockIndex));
    }
    for (uint32_t blockIndex = 0; blockIndex < PACK_THREAD_NUM; ++blockIndex) {
        CHK_RET(HcommThreadNotifyWaitOnThread(coordinator, blockIndex, CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
}

HcclResult RunCrossServerReduce(ThreadHandle thread, const ChannelInfo &pairedChannel, void *crossSend,
    const SliceDesc &slice, HcommDataType dataType, HcommReduceOp reduceOp)
{
    const uint64_t requiredPairSumBytes = slice.bufferOffsetBytes + PAIR_SUM_SLOT_NUM * slice.bytes;
    if (pairedChannel.remoteCclMem.size < requiredPairSumBytes) {
        HCCL_ERROR("Cross-server CCL buffer is too small for rank %u", pairedChannel.remoteRank);
        return HCCL_E_INTERNAL;
    }

    // Both paired ranks initialize this bank before either side reduces into it.
    CHK_RET(HcommChannelNotifyRecordOnThread(thread, pairedChannel.handle, NOTIFY_CROSS_BUFFER_READY));
    CHK_RET(HcommChannelNotifyWaitOnThread(thread, pairedChannel.handle, NOTIFY_CROSS_BUFFER_READY, CUSTOM_TIMEOUT));
    CHK_RET(HcommWriteReduceOnThread(thread, pairedChannel.handle,
        AddOffset(pairedChannel.remoteCclMem.addr, slice.bufferOffsetBytes), crossSend,
        SERVER_RANK_SIZE * slice.count, dataType, reduceOp));
    CHK_RET(HcommChannelNotifyRecordOnThread(thread, pairedChannel.handle, NOTIFY_CROSS_REDUCE_DONE));
    CHK_RET(HcommChannelNotifyWaitOnThread(thread, pairedChannel.handle, NOTIFY_CROSS_REDUCE_DONE, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult ReleasePairSumBank(ThreadHandle thread, const ChannelInfo &pairedChannel)
{
    CHK_RET(HcommChannelNotifyRecordOnThread(thread, pairedChannel.handle, NOTIFY_PAIR_SUM_RELEASED));
    CHK_RET(HcommChannelNotifyWaitOnThread(thread, pairedChannel.handle, NOTIFY_PAIR_SUM_RELEASED,
        CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult RunTreeLocalReduceScatter(const OpParam &param, const AlgResourceCtx &resCtx, ThreadHandle mainThread,
    uint32_t localServerStart, uint32_t localIndex, void *pairSum, uint64_t sliceCount, uint64_t sliceBytes,
    uint64_t remoteBufferOffsetBytes, uint64_t outputOffsetBytes, uint32_t layoutId,
    uint32_t readyNotifyIndex, uint32_t doneNotifyIndex, HcommDataType dataType, HcommReduceOp reduceOp)
{
    const uint32_t logicalIndex = LogicalFromPhysical(localIndex, layoutId);
    if (logicalIndex >= SERVER_RANK_SIZE) {
        HCCL_ERROR("Invalid tree layout %u for local index %u", layoutId, localIndex);
        return HCCL_E_INTERNAL;
    }

    // A fixed xor tree uses only three communication rounds for eight ranks.
    // Each destination has exactly one writer in a round, so every FP32 sum has
    // a deterministic operand order independent of transfer completion order.
    for (uint32_t blockCount = SERVER_RANK_SIZE / 2; blockCount > 0; blockCount /= 2) {
        const uint32_t partnerLogicalIndex = logicalIndex ^ blockCount;
        const uint32_t partnerLocalIndex = PhysicalFromLogical(partnerLogicalIndex, layoutId);
        const uint32_t partnerRank = localServerStart + partnerLocalIndex;
        const ChannelInfo *partnerChannel = FindChannel(resCtx, localServerStart + localIndex, partnerRank);
        if (partnerChannel == nullptr) {
            HCCL_ERROR("Missing tree partner channel for rank %u", partnerRank);
            return HCCL_E_INTERNAL;
        }

        const uint32_t groupBase = logicalIndex & ~((blockCount * 2) - 1);
        const uint32_t sendBlock = groupBase + (logicalIndex & blockCount ? 0 : blockCount);
        const uint64_t sendOffset = sendBlock * sliceBytes;

        // A cross-server pair only confirms its own pairSum.  Before this
        // intra-server tree edge may reduce into the partner buffer, both tree
        // peers must additionally confirm that their pairSum initialization
        // (including the cross-server reduce) has completed.
        CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, partnerChannel->handle, readyNotifyIndex));
        CHK_RET(HcommChannelNotifyWaitOnThread(mainThread, partnerChannel->handle, readyNotifyIndex,
            CUSTOM_TIMEOUT));
        CHK_RET(HcommWriteReduceOnThread(mainThread, partnerChannel->handle,
            AddOffset(partnerChannel->remoteCclMem.addr, remoteBufferOffsetBytes + sendOffset),
            AddOffset(pairSum, sendOffset),
            blockCount * sliceCount, dataType, reduceOp));
        CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, partnerChannel->handle, doneNotifyIndex));
        CHK_RET(HcommChannelNotifyWaitOnThread(mainThread, partnerChannel->handle, doneNotifyIndex,
            CUSTOM_TIMEOUT));
    }

    CHK_RET(HcommLocalCopyOnThread(mainThread, AddOffset(param.outputPtr, outputOffsetBytes),
        AddOffset(pairSum, logicalIndex * sliceBytes), sliceBytes));
    return HCCL_SUCCESS;
}

HcclResult RunParallelLocalReduceScatter(const OpParam &param, const AlgResourceCtx &resCtx, ThreadHandle mainThread,
    uint32_t localServerStart, uint32_t localIndex, void *pairSum, void *crossSend, uint64_t sliceCount,
    uint64_t sliceBytes, uint64_t offsetBytes, uint64_t remoteBufferOffsetBytes, uint64_t requiredBufferBytes,
    bool waitForDataConsumed, HcommDataType dataType,
    HcommReduceOp reduceOp)
{
    // The main thread owns receive readiness and reduction scheduling.  First,
    // advertise that all seven receive slots can overwrite crossSend.
    const ChannelInfo *peerChannels[SERVER_RANK_SIZE] = {};
    for (uint32_t peerOrdinal = 0; peerOrdinal < WORKER_THREAD_NUM; ++peerOrdinal) {
        const uint32_t targetLocalIndex = PeerLocalIndex(peerOrdinal, localIndex);
        const uint32_t targetRank = localServerStart + targetLocalIndex;
        const ChannelInfo *targetChannel = FindChannel(resCtx, localServerStart + localIndex, targetRank);
        if (targetChannel == nullptr ||
            targetChannel->remoteCclMem.size < remoteBufferOffsetBytes + requiredBufferBytes) {
            HCCL_ERROR("Invalid intra-server channel for target rank %u", targetRank);
            return HCCL_E_INTERNAL;
        }
        peerChannels[targetLocalIndex] = targetChannel;
        CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, targetChannel->handle, NOTIFY_RECEIVE_SLOT_READY));
    }

    // Each worker owns exactly one physical Mesh peer.  Their writes target
    // disjoint remote slots, which makes all seven links independently runnable.
    for (uint32_t peerOrdinal = 0; peerOrdinal < WORKER_THREAD_NUM; ++peerOrdinal) {
        const uint32_t targetLocalIndex = PeerLocalIndex(peerOrdinal, localIndex);
        const ChannelInfo *targetChannel = peerChannels[targetLocalIndex];
        const uint32_t targetReceiveSlot = ReceiveSlot(targetLocalIndex, localIndex);
        const ThreadHandle workerThread = resCtx.aicpuThreads[WORKER_THREAD_FIRST_INDEX + peerOrdinal];

        CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, workerThread, peerOrdinal));
        CHK_RET(HcommThreadNotifyWaitOnThread(workerThread, peerOrdinal, CUSTOM_TIMEOUT));
        CHK_RET(HcommChannelNotifyWaitOnThread(workerThread, targetChannel->handle, NOTIFY_RECEIVE_SLOT_READY,
            CUSTOM_TIMEOUT));
        CHK_RET(HcommWriteOnThread(workerThread, targetChannel->handle,
            AddOffset(targetChannel->remoteCclMem.addr,
                remoteBufferOffsetBytes + (CROSS_SEND_SLOT_BASE + targetReceiveSlot) * sliceBytes),
            AddOffset(pairSum, targetLocalIndex * sliceBytes), sliceBytes));
        CHK_RET(HcommChannelNotifyRecordOnThread(workerThread, targetChannel->handle, NOTIFY_LOCAL_DATA_READY));
        if (waitForDataConsumed) {
            CHK_RET(HcommChannelNotifyWaitOnThread(workerThread, targetChannel->handle, NOTIFY_LOCAL_DATA_CONSUMED,
                CUSTOM_TIMEOUT));
        }
        CHK_RET(HcommThreadNotifyRecordOnThread(workerThread, mainThread, peerOrdinal));
    }

    if (!waitForDataConsumed) {
        // Unique staging banks cannot be overwritten by a following slice.
        // Wait until every incoming block is visible before releasing the same
        // seven workers for the deterministic, element-partitioned reduction.
        for (uint32_t sourceLocalIndex = 0; sourceLocalIndex < SERVER_RANK_SIZE; ++sourceLocalIndex) {
            if (sourceLocalIndex == localIndex) {
                continue;
            }
            const uint32_t sourceRank = localServerStart + sourceLocalIndex;
            const ChannelInfo *sourceChannel = peerChannels[sourceLocalIndex];
            if (sourceChannel == nullptr) {
                HCCL_ERROR("Missing intra-server channel for source rank %u", sourceRank);
                return HCCL_E_INTERNAL;
            }
            CHK_RET(HcommChannelNotifyWaitOnThread(mainThread, sourceChannel->handle, NOTIFY_LOCAL_DATA_READY,
                CUSTOM_TIMEOUT));
        }

        // Match the completion records from the outbound Mesh work before
        // reusing each worker's thread notify for its reduction phase.
        for (uint32_t peerOrdinal = 0; peerOrdinal < WORKER_THREAD_NUM; ++peerOrdinal) {
            CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, peerOrdinal, CUSTOM_TIMEOUT));
        }

        // Segment 0 stays on the main thread; segments 1..7 run on the seven
        // workers.  Output intervals never overlap, and every segment retains
        // source-rank order, so FP32 results remain deterministic.
        for (uint32_t peerOrdinal = 0; peerOrdinal < WORKER_THREAD_NUM; ++peerOrdinal) {
            const ThreadHandle workerThread = resCtx.aicpuThreads[WORKER_THREAD_FIRST_INDEX + peerOrdinal];
            CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, workerThread, peerOrdinal));
            CHK_RET(HcommThreadNotifyWaitOnThread(workerThread, peerOrdinal, CUSTOM_TIMEOUT));
            CHK_RET(ReduceOutputSegment(param, workerThread, localIndex, pairSum, crossSend, sliceCount,
                sliceBytes, offsetBytes, peerOrdinal + 1, dataType, reduceOp));
            CHK_RET(HcommThreadNotifyRecordOnThread(workerThread, mainThread, peerOrdinal));
        }
        CHK_RET(ReduceOutputSegment(param, mainThread, localIndex, pairSum, crossSend, sliceCount, sliceBytes,
            offsetBytes, 0, dataType, reduceOp));
        for (uint32_t peerOrdinal = 0; peerOrdinal < WORKER_THREAD_NUM; ++peerOrdinal) {
            CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, peerOrdinal, CUSTOM_TIMEOUT));
        }
        return HCCL_SUCCESS;
    }

    // A reused staging bank needs per-source consumption acknowledgements.
    // Keep its final FP32 reduction on the main thread so the acknowledgement
    // remains ordered after the corresponding buffer read.
    for (uint32_t sourceLocalIndex = 0; sourceLocalIndex < SERVER_RANK_SIZE; ++sourceLocalIndex) {
        const bool isFirstOperand = sourceLocalIndex == 0;
        if (sourceLocalIndex == localIndex) {
            CHK_RET(CopyOrReduce(mainThread, AddOffset(param.outputPtr, offsetBytes),
                AddOffset(pairSum, localIndex * sliceBytes), sliceCount, sliceBytes, dataType, reduceOp,
                isFirstOperand));
            continue;
        }

        const uint32_t sourceRank = localServerStart + sourceLocalIndex;
        const ChannelInfo *sourceChannel = peerChannels[sourceLocalIndex];
        if (sourceChannel == nullptr) {
            HCCL_ERROR("Missing intra-server channel for source rank %u", sourceRank);
            return HCCL_E_INTERNAL;
        }
        const uint32_t receiveSlot = ReceiveSlot(localIndex, sourceLocalIndex);
        CHK_RET(HcommChannelNotifyWaitOnThread(mainThread, sourceChannel->handle, NOTIFY_LOCAL_DATA_READY,
            CUSTOM_TIMEOUT));
        CHK_RET(CopyOrReduce(mainThread, AddOffset(param.outputPtr, offsetBytes),
            AddOffset(crossSend, receiveSlot * sliceBytes), sliceCount, sliceBytes, dataType, reduceOp,
            isFirstOperand));
        if (waitForDataConsumed) {
            CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, sourceChannel->handle, NOTIFY_LOCAL_DATA_CONSUMED));
        }
    }

    // Join every outbound write before advancing the main control thread.
    for (uint32_t peerOrdinal = 0; peerOrdinal < WORKER_THREAD_NUM; ++peerOrdinal) {
        CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, peerOrdinal, CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
}

HcclResult PrepareCrossServerSlice(const OpParam &param, const AlgResourceCtx &resCtx, ThreadHandle thread,
    uint32_t localServerStart, uint32_t remoteServerStart, const ChannelInfo &pairedChannel, const SliceDesc &slice,
    uint64_t totalBytes, HcommDataType dataType, HcommReduceOp reduceOp, bool parallelPacking,
    uint32_t layoutId)
{
    void *pairSum = AddOffset(resCtx.localBuffer.addr, slice.bufferOffsetBytes);
    void *crossSend = AddOffset(resCtx.localBuffer.addr, slice.crossBufferOffsetBytes);
    if (parallelPacking && slice.bytes >= PARALLEL_PACK_MIN_SLICE_BYTES) {
        CHK_RET(PackInputSliceParallel(param, resCtx, thread, localServerStart, remoteServerStart, pairSum,
            crossSend, slice, totalBytes, layoutId));
    } else {
        CHK_RET(PackInputSlice(param, thread, localServerStart, remoteServerStart, pairSum, crossSend, slice,
            totalBytes, layoutId));
    }
    CHK_RET(RunCrossServerReduce(thread, pairedChannel, crossSend, slice, dataType, reduceOp));
    return HCCL_SUCCESS;
}

HcclResult RunLargeLocalSlice(const OpParam &param, const AlgResourceCtx &resCtx, ThreadHandle mainThread,
    uint32_t localServerStart, uint32_t localIndex, const SliceDesc &slice, HcommDataType dataType,
    HcommReduceOp reduceOp, bool waitForDataConsumed)
{
    void *pairSum = AddOffset(resCtx.localBuffer.addr, slice.bufferOffsetBytes);
    void *crossSend = AddOffset(resCtx.localBuffer.addr, slice.crossBufferOffsetBytes);
    const uint64_t requiredBufferBytes = LOCAL_BUFFER_SLOT_NUM * slice.bytes;
    return RunParallelLocalReduceScatter(param, resCtx, mainThread, localServerStart, localIndex, pairSum, crossSend,
        slice.count, slice.bytes, slice.offsetCount * sizeof(float), slice.bufferOffsetBytes, requiredBufferBytes,
        waitForDataConsumed, dataType, reduceOp);
}

HcclResult QueuePipelinedCrossServerSlice(const OpParam &param, const AlgResourceCtx &resCtx, ThreadHandle mainThread,
    ThreadHandle crossThread, uint32_t localServerStart, uint32_t remoteServerStart, const ChannelInfo &pairedChannel,
    const SliceDesc &slice, uint64_t totalBytes, HcommDataType dataType, HcommReduceOp reduceOp)
{
    // The main thread only releases this worker after the preceding cross-server
    // stage completed.  Consequently channel notify 0/1 remain strictly ordered.
    CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, crossThread, NOTIFY_PIPELINE_STAGE_READY));
    CHK_RET(HcommThreadNotifyWaitOnThread(crossThread, NOTIFY_PIPELINE_STAGE_READY, CUSTOM_TIMEOUT));
    CHK_RET(PrepareCrossServerSlice(param, resCtx, crossThread, localServerStart, remoteServerStart, pairedChannel,
        slice, totalBytes, dataType, reduceOp, true, 0));
    CHK_RET(HcommThreadNotifyRecordOnThread(crossThread, mainThread, NOTIFY_PIPELINE_STAGE_READY));
    return HCCL_SUCCESS;
}

SliceDesc MakePipelineSlice(uint64_t offsetCount, uint64_t count, uint32_t sliceIndex,
    uint64_t pingPongBankBytes, bool useUniqueBanks)
{
    const uint64_t bufferOffsetBytes = useUniqueBanks ? LOCAL_BUFFER_SLOT_NUM * offsetCount * sizeof(float) :
        (sliceIndex % 2) * pingPongBankBytes;
    return MakeSlice(offsetCount, count, bufferOffsetBytes);
}

SliceDesc MakeConcurrentTreeSlice(uint64_t offsetCount, uint64_t count, uint32_t sliceIndex,
    uint64_t bankBytes)
{
    SliceDesc slice = MakeSlice(offsetCount, count, sliceIndex * bankBytes);
    slice.crossBufferOffsetBytes = (sliceIndex + 1) * bankBytes;
    return slice;
}

[[maybe_unused]] HcclResult RunConcurrentTreeReduceScatter(const OpParam &param, const AlgResourceCtx &resCtx,
    ThreadHandle mainThread, uint32_t localServerStart, uint32_t remoteServerStart, uint32_t localIndex,
    const ChannelInfo &pairedChannel, uint64_t totalBytes, HcommDataType dataType, HcommReduceOp reduceOp)
{
    // Seven equal slices let the three-round recursive-halving trees run on
    // seven independent AICPU threads.  An eighth bank is the rolling
    // cross-send source; after cross stage N completes, that bank becomes the
    // immutable pair-sum bank of tree N+1.
    const uint64_t baseSliceCount = param.count / TREE_PIPELINE_SLICE_NUM;
    const uint64_t remainder = param.count % TREE_PIPELINE_SLICE_NUM;
    const uint64_t maxSliceCount = baseSliceCount + (remainder != 0 ? 1 : 0);
    const uint64_t sliceCapacity = resCtx.localBuffer.size /
        (TREE_PIPELINE_BANK_NUM * PAIR_SUM_SLOT_NUM * sizeof(float));
    if (maxSliceCount == 0 || maxSliceCount > sliceCapacity) {
        HCCL_ERROR("HCCL buffer is too small for concurrent tree staging");
        return HCCL_E_PARA;
    }

    const uint64_t bankBytes = PAIR_SUM_SLOT_NUM * maxSliceCount * sizeof(float);
    SliceDesc slices[TREE_PIPELINE_SLICE_NUM] = {};
    uint64_t offsetCount = 0;
    for (uint32_t sliceIndex = 0; sliceIndex < TREE_PIPELINE_SLICE_NUM; ++sliceIndex) {
        const uint64_t sliceCount = baseSliceCount + (sliceIndex < remainder ? 1 : 0);
        slices[sliceIndex] = MakeConcurrentTreeSlice(offsetCount, sliceCount, sliceIndex, bankBytes);
        offsetCount += sliceCount;
    }

    const ThreadHandle crossThread = resCtx.aicpuThreads[CROSS_PIPELINE_THREAD_INDEX];

    // Prepare slice 0 on the control thread.  Its tree and the remaining six
    // cross stages can then start concurrently.
    CHK_RET(PrepareCrossServerSlice(param, resCtx, mainThread, localServerStart, remoteServerStart, pairedChannel,
        slices[0], totalBytes, dataType, reduceOp, true, 1));
    CHK_RET(HcommThreadNotifyRecordOnThread(mainThread,
        resCtx.aicpuThreads[WORKER_THREAD_FIRST_INDEX], 0));

    CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, crossThread, NOTIFY_PIPELINE_STAGE_READY));
    CHK_RET(HcommThreadNotifyWaitOnThread(crossThread, NOTIFY_PIPELINE_STAGE_READY, CUSTOM_TIMEOUT));
    for (uint32_t sliceIndex = 1; sliceIndex < TREE_PIPELINE_SLICE_NUM; ++sliceIndex) {
        const uint32_t layoutId = sliceIndex + 1;
        CHK_RET(PrepareCrossServerSlice(param, resCtx, crossThread, localServerStart, remoteServerStart,
            pairedChannel, slices[sliceIndex], totalBytes, dataType, reduceOp, true, layoutId));
        CHK_RET(HcommThreadNotifyRecordOnThread(crossThread,
            resCtx.aicpuThreads[WORKER_THREAD_FIRST_INDEX + sliceIndex], sliceIndex));
    }
    CHK_RET(HcommThreadNotifyRecordOnThread(crossThread, mainThread, NOTIFY_PIPELINE_STAGE_READY));

    // Each slice uses a different GF(2^3) basis.  In every tree round, the
    // seven slices therefore select all seven xor peers exactly once, spreading
    // the large WriteReduce transfers over the complete Mesh.
    for (uint32_t sliceIndex = 0; sliceIndex < TREE_PIPELINE_SLICE_NUM; ++sliceIndex) {
        const ThreadHandle treeThread =
            resCtx.aicpuThreads[WORKER_THREAD_FIRST_INDEX + sliceIndex];
        const uint32_t layoutId = sliceIndex + 1;
        const uint32_t readyNotifyIndex = sliceIndex * TREE_NOTIFY_NUM_PER_SLICE;
        const uint32_t doneNotifyIndex = readyNotifyIndex + 1;
        CHK_RET(HcommThreadNotifyWaitOnThread(treeThread, sliceIndex, CUSTOM_TIMEOUT));
        CHK_RET(RunTreeLocalReduceScatter(param, resCtx, treeThread, localServerStart, localIndex,
            AddOffset(resCtx.localBuffer.addr, slices[sliceIndex].bufferOffsetBytes),
            slices[sliceIndex].count, slices[sliceIndex].bytes,
            slices[sliceIndex].bufferOffsetBytes, slices[sliceIndex].offsetCount * sizeof(float),
            layoutId, readyNotifyIndex, doneNotifyIndex, dataType, reduceOp));
        CHK_RET(HcommThreadNotifyRecordOnThread(treeThread, mainThread, sliceIndex));
    }

    CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, NOTIFY_PIPELINE_STAGE_READY, CUSTOM_TIMEOUT));
    for (uint32_t sliceIndex = 0; sliceIndex < TREE_PIPELINE_SLICE_NUM; ++sliceIndex) {
        CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, sliceIndex, CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
}

HcclResult RunPipelinedLargeReduceScatter(const OpParam &param, const AlgResourceCtx &resCtx, ThreadHandle mainThread,
    uint32_t localServerStart, uint32_t remoteServerStart, uint32_t localIndex, const ChannelInfo &pairedChannel,
    uint64_t totalBytes, HcommDataType dataType, HcommReduceOp reduceOp)
{
    // The performance inputs fit in the CCL buffer in full.  Give every 8 MB
    // output slice its own staging bank, so Clos for slice N+1 can overlap Mesh
    // for slice N without any per-slice buffer-reuse dependency.
    const uint64_t uniqueBankBytes = LOCAL_BUFFER_SLOT_NUM * totalBytes;
    const bool useUniqueBanks = uniqueBankBytes <= resCtx.localBuffer.size;
    const uint64_t pingPongSliceCapacity =
        resCtx.localBuffer.size / (2 * LOCAL_BUFFER_SLOT_NUM * sizeof(float));
    if (!useUniqueBanks && pingPongSliceCapacity == 0) {
        HCCL_ERROR("HCCL buffer is too small for pipelined ReduceScatter staging");
        return HCCL_E_PARA;
    }

    const uint64_t sliceCapacity = useUniqueBanks ? std::min(PIPELINE_SLICE_COUNT, param.count) :
        pingPongSliceCapacity;
    const uint64_t pingPongBankBytes =
        LOCAL_BUFFER_SLOT_NUM * pingPongSliceCapacity * sizeof(float);
    const uint64_t firstSliceCount = useUniqueBanks ? sliceCapacity :
        (param.count <= pingPongSliceCapacity * 2 ? (param.count + 1) / 2 :
            std::min(pingPongSliceCapacity, param.count));
    SliceDesc current = MakePipelineSlice(0, firstSliceCount, 0, pingPongBankBytes, useUniqueBanks);
    const ThreadHandle crossThread = resCtx.aicpuThreads[CROSS_PIPELINE_THREAD_INDEX];

    CHK_RET(PrepareCrossServerSlice(param, resCtx, mainThread, localServerStart, remoteServerStart, pairedChannel,
        current, totalBytes, dataType, reduceOp, true, 0));

    uint64_t nextOffsetCount = current.offsetCount + current.count;
    uint32_t nextSliceIndex = 1;
    bool hasNextSlice = nextOffsetCount < param.count;
    SliceDesc next{};
    if (hasNextSlice) {
        next = MakePipelineSlice(nextOffsetCount, std::min(sliceCapacity, param.count - nextOffsetCount),
            nextSliceIndex, pingPongBankBytes, useUniqueBanks);
        CHK_RET(QueuePipelinedCrossServerSlice(param, resCtx, mainThread, crossThread, localServerStart,
            remoteServerStart, pairedChannel, next, totalBytes, dataType, reduceOp));
    }

    CHK_RET(RunLargeLocalSlice(param, resCtx, mainThread, localServerStart, localIndex, current, dataType, reduceOp,
        !useUniqueBanks));

    while (hasNextSlice) {
        const uint64_t followingOffsetCount = next.offsetCount + next.count;
        const bool hasFollowingSlice = followingOffsetCount < param.count;

        // The next cross-server stage will reuse the current bank only after
        // both paired ranks completed all Mesh reads from it.
        if (hasFollowingSlice && !useUniqueBanks) {
            CHK_RET(ReleasePairSumBank(mainThread, pairedChannel));
        }
        CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, NOTIFY_PIPELINE_STAGE_READY, CUSTOM_TIMEOUT));

        current = next;
        nextOffsetCount = followingOffsetCount;
        ++nextSliceIndex;
        hasNextSlice = hasFollowingSlice;
        if (hasNextSlice) {
            next = MakePipelineSlice(nextOffsetCount, std::min(sliceCapacity, param.count - nextOffsetCount),
                nextSliceIndex, pingPongBankBytes, useUniqueBanks);
            CHK_RET(QueuePipelinedCrossServerSlice(param, resCtx, mainThread, crossThread, localServerStart,
                remoteServerStart, pairedChannel, next, totalBytes, dataType, reduceOp));
        }

        CHK_RET(RunLargeLocalSlice(param, resCtx, mainThread, localServerStart, localIndex, current, dataType,
            reduceOp, !useUniqueBanks));
    }
    return HCCL_SUCCESS;
}
} // namespace

namespace ops_hccl {
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    if (param.dataType != HCCL_DATA_TYPE_FP32 || param.reduceType != HCCL_REDUCE_SUM) {
        HCCL_ERROR("Only FP32 SUM ReduceScatter is supported");
        return HCCL_E_NOT_SUPPORT;
    }
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    if (param.rankSize != REDUCE_SCATTER_RANK_SIZE || param.myRank >= REDUCE_SCATTER_RANK_SIZE ||
        param.count > std::numeric_limits<uint64_t>::max() / (sizeof(float) * param.rankSize)) {
        HCCL_ERROR("Invalid ReduceScatter parameters");
        return HCCL_E_PARA;
    }

    const uint64_t totalBytes = param.count * sizeof(float);
    const uint64_t maxSliceCount = resCtx.localBuffer.size / LOCAL_BUFFER_SLOT_NUM / sizeof(float);
    if (maxSliceCount == 0) {
        HCCL_ERROR("HCCL buffer is too small for ReduceScatter staging");
        return HCCL_E_PARA;
    }

    const uint32_t localServerStart = (param.myRank / SERVER_RANK_SIZE) * SERVER_RANK_SIZE;
    const uint32_t remoteServerStart = localServerStart == 0 ? SERVER_RANK_SIZE : 0;
    const uint32_t localIndex = param.myRank - localServerStart;
    const uint32_t pairedRank = remoteServerStart + localIndex;
    const ThreadHandle mainThread = resCtx.aicpuThreads[MAIN_THREAD_INDEX];
    const HcommDataType hcommDataType = static_cast<HcommDataType>(param.dataType);
    const HcommReduceOp hcommReduceOp = static_cast<HcommReduceOp>(param.reduceType);
    const ChannelInfo *pairedChannel = FindChannel(resCtx, param.myRank, pairedRank);
    if (pairedChannel == nullptr) {
        HCCL_ERROR("Missing cross-server channel for paired rank %u", pairedRank);
        return HCCL_E_INTERNAL;
    }

    if (totalBytes >= PIPELINE_MIN_MESSAGE_BYTES) {
        return RunPipelinedLargeReduceScatter(param, resCtx, mainThread, localServerStart, remoteServerStart,
            localIndex, *pairedChannel, totalBytes, hcommDataType, hcommReduceOp);
    }

    for (uint64_t offsetCount = 0; offsetCount < param.count;) {
        const uint64_t sliceCount = std::min(maxSliceCount, param.count - offsetCount);
        const SliceDesc slice = MakeSlice(offsetCount, sliceCount, 0);
        void *pairSum = resCtx.localBuffer.addr;
        CHK_RET(PrepareCrossServerSlice(param, resCtx, mainThread, localServerStart, remoteServerStart,
            *pairedChannel, slice, totalBytes, hcommDataType, hcommReduceOp, false, 0));

        if (totalBytes <= SMALL_MESSAGE_BYTES) {
            CHK_RET(RunTreeLocalReduceScatter(param, resCtx, mainThread, localServerStart, localIndex, pairSum,
                slice.count, slice.bytes, slice.bufferOffsetBytes, slice.offsetCount * sizeof(float), 0,
                NOTIFY_RECEIVE_SLOT_READY, NOTIFY_LOCAL_DATA_READY, hcommDataType, hcommReduceOp));
        } else {
            CHK_RET(RunLargeLocalSlice(param, resCtx, mainThread, localServerStart, localIndex, slice,
                hcommDataType, hcommReduceOp, true));
        }

        // The paired rank may overwrite pairSum for the next slice only after this
        // rank has completed all local uses of the current slice.
        if (offsetCount + sliceCount < param.count) {
            CHK_RET(ReleasePairSumBank(mainThread, *pairedChannel));
        }
        offsetCount += sliceCount;
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
