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

#include <algorithm>

namespace {

// Coordinator notify 0 is reserved for the Host -> AICPU kernel start handshake.
constexpr uint32_t BANK_FREE_NOTIFY_BEGIN = 1;
constexpr uint32_t SLAVE_THREAD_DONE_NOTIFY_BEGIN =
    BANK_FREE_NOTIFY_BEGIN + custom_rs::PIPELINE_BUFFER_NUM;
constexpr uint32_t SLAVE_THREAD_START_NOTIFY = 0;
constexpr uint32_t COMM_BANK_FREE_NOTIFY_BEGIN = 1;
constexpr uint32_t SLAVE_THREAD_NUM =
    custom_rs::COMM_THREAD_NUM + custom_rs::REDUCE_THREAD_NUM;
constexpr uint32_t FIFTEEN_SLOT_PASS_NUM = 3;
constexpr uint32_t GROUP_DONE_NOTIFY_BEGIN = 1;

static_assert(SLAVE_THREAD_DONE_NOTIFY_BEGIN + SLAVE_THREAD_NUM <=
    custom_rs::DIRECT_THREAD_NOTIFY_NUM,
    "Coordinator slave-thread completion notify indices exceed the acquired notify count");
static_assert(COMM_BANK_FREE_NOTIFY_BEGIN + custom_rs::PIPELINE_BUFFER_NUM <=
    custom_rs::DIRECT_THREAD_NOTIFY_NUM,
    "Communication bank-free notify indices exceed the acquired notify count");
static_assert(GROUP_DONE_NOTIFY_BEGIN +
    FIFTEEN_SLOT_PASS_NUM * custom_rs::REDUCE_GROUP_NUM <=
    custom_rs::DIRECT_THREAD_NOTIFY_NUM,
    "Grouped first-fold completion notify indices exceed the acquired notify count");

void *OffsetAddress(void *base, uint64_t offset)
{
    return static_cast<void *>(static_cast<uint8_t *>(base) + offset);
}

HcclResult StartSlaveThreads(const AlgResourceCtx &resCtx)
{
    ThreadHandle coordinator = resCtx.threads[custom_rs::COORDINATOR_THREAD_INDEX];
    for (uint32_t workerIndex = 0; workerIndex < SLAVE_THREAD_NUM; ++workerIndex) {
        ThreadHandle worker = resCtx.threads[custom_rs::COMM_THREAD_BEGIN + workerIndex];
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            coordinator, worker, SLAVE_THREAD_START_NOTIFY)));
        // Every slave thread starts with a LocalWait controlled by Thread 0.
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            worker, SLAVE_THREAD_START_NOTIFY, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult FinishSlaveThreads(const AlgResourceCtx &resCtx)
{
    ThreadHandle coordinator = resCtx.threads[custom_rs::COORDINATOR_THREAD_INDEX];
    for (uint32_t workerIndex = 0; workerIndex < SLAVE_THREAD_NUM; ++workerIndex) {
        ThreadHandle worker = resCtx.threads[custom_rs::COMM_THREAD_BEGIN + workerIndex];
        // Every slave thread ends with a LocalRecord back to Thread 0.
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(worker, coordinator,
            SLAVE_THREAD_DONE_NOTIFY_BEGIN + workerIndex)));
    }
    for (uint32_t workerIndex = 0; workerIndex < SLAVE_THREAD_NUM; ++workerIndex) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(coordinator,
            SLAVE_THREAD_DONE_NOTIFY_BEGIN + workerIndex, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult NotifyPeerReady(ThreadHandle thread, const ChannelInfo &channel)
{
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, custom_rs::NOTIFY_READY)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        thread, channel.handle, custom_rs::NOTIFY_READY, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult NotifyAllPeersReady(ThreadHandle thread, const AlgResourceCtx &resCtx)
{
    // Queue every READY record before any READY wait. All ranks follow the
    // same ordering, so no rank can block before notifying the other peers.
    for (const auto &channel : resCtx.channels) {
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
            thread, channel.handle, custom_rs::NOTIFY_READY)));
    }
    for (const auto &channel : resCtx.channels) {
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            thread, channel.handle, custom_rs::NOTIFY_READY, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult ExecuteRecursiveHalving(const OpParam &param, const AlgResourceCtx &resCtx,
    ThreadHandle thread, uint64_t tileOffset, uint64_t tileBytes, uint64_t sliceBytes)
{
    const uint64_t requiredBufferBytes = tileBytes * custom_rs::RECURSIVE_HALVING_BUFFER_SLOT_NUM;
    uint64_t usableBufferSize = resCtx.localBuffer.size;
    for (uint32_t round = 0; round < custom_rs::RECURSIVE_HALVING_ROUND_NUM; ++round) {
        const uint32_t expectedRank = custom_rs::GetRecursiveHalvingPeer(param.myRank, round);
        const uint32_t channelIdx = custom_rs::GetChannelIndex(param.myRank, expectedRank);
        const ChannelInfo &channel = resCtx.channels[channelIdx];
        CHK_PRT_RET(channel.remoteRank != expectedRank,
            HCCL_ERROR("Unexpected recursive-halving channel rank[%u], expected[%u]",
                channel.remoteRank, expectedRank), HCCL_E_INTERNAL);
        CHK_PRT_RET(channel.remoteCclMem.addr == nullptr,
            HCCL_ERROR("Remote HCCL buffer of rank[%u] is null", channel.remoteRank), HCCL_E_PTR);
        usableBufferSize = std::min(usableBufferSize, channel.remoteCclMem.size);
    }
    CHK_PRT_RET(usableBufferSize < requiredBufferBytes,
        HCCL_ERROR("HCCL buffer[%llu] is smaller than recursive-halving requirement[%llu]",
            static_cast<unsigned long long>(usableBufferSize),
            static_cast<unsigned long long>(requiredBufferBytes)), HCCL_E_INTERNAL);

    if (tileOffset == 0 && tileBytes == sliceBytes) {
        // The pure small-data case is contiguous and keeps the original
        // one-copy critical path.
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread,
            resCtx.localBuffer.addr, param.inputPtr, tileBytes * custom_rs::RANK_SIZE)));
    } else {
        // A tail tile is strided in sendBuf: target slices are separated by
        // the full sliceBytes. Pack the sixteen tails into contiguous scratch.
        for (uint32_t targetRank = 0; targetRank < custom_rs::RANK_SIZE; ++targetRank) {
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread,
                OffsetAddress(resCtx.localBuffer.addr, static_cast<uint64_t>(targetRank) * tileBytes),
                OffsetAddress(param.inputPtr,
                    static_cast<uint64_t>(targetRank) * sliceBytes + tileOffset),
                tileBytes)));
        }
    }

    uint32_t activeStart = 0;
    for (uint32_t round = 0; round < custom_rs::RECURSIVE_HALVING_ROUND_NUM; ++round) {
        const uint32_t halfSlices = custom_rs::GetRecursiveHalvingMask(round);
        const bool keepUpperHalf = (param.myRank & halfSlices) != 0;
        const uint32_t keepStart = activeStart + (keepUpperHalf ? halfSlices : 0);
        const uint64_t exchangeBytes = static_cast<uint64_t>(halfSlices) * tileBytes;
        const uint32_t peerRank = custom_rs::GetRecursiveHalvingPeer(param.myRank, round);
        const ChannelInfo &channel = resCtx.channels[custom_rs::GetChannelIndex(param.myRank, peerRank)];

        // Peers retain opposite halves. The local range being updated is disjoint
        // from the range that the peer concurrently reads from this rank.
        CHK_RET(NotifyPeerReady(thread, channel));
        CHK_RET(static_cast<HcclResult>(HcommReadReduceOnThread(thread, channel.handle,
            OffsetAddress(resCtx.localBuffer.addr, static_cast<uint64_t>(keepStart) * tileBytes),
            OffsetAddress(channel.remoteCclMem.addr, static_cast<uint64_t>(keepStart) * tileBytes),
            exchangeBytes / sizeof(float), HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
        activeStart = keepStart;
    }

    CHK_PRT_RET(activeStart != param.myRank,
        HCCL_ERROR("Unexpected recursive-halving result slice[%u], expected[%u]", activeStart, param.myRank),
        HCCL_E_INTERNAL);
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread,
        OffsetAddress(param.outputPtr, tileOffset),
        OffsetAddress(resCtx.localBuffer.addr, static_cast<uint64_t>(activeStart) * tileBytes), tileBytes)));
    return HCCL_SUCCESS;
}

HcclResult DispatchAllToAllExchange(const OpParam &param, const AlgResourceCtx &resCtx,
    uint32_t bufferIndex, uint64_t tileIndex, uint64_t tileOffset, uint64_t tileBytes,
    uint64_t sliceBytes, uint64_t bankSlotStride, uint64_t tileSlotStride)
{
    ThreadHandle coordinator = resCtx.threads[custom_rs::COORDINATOR_THREAD_INDEX];

    const bool reuseBuffer = tileIndex >= custom_rs::PIPELINE_BUFFER_NUM;
    if (reuseBuffer) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(coordinator,
            BANK_FREE_NOTIFY_BEGIN + bufferIndex, CUSTOM_TIMEOUT)));
        CHK_RET(NotifyAllPeersReady(coordinator, resCtx));
        // Communication threads can otherwise run ahead of Thread 0 and
        // overwrite a reused bank before Reduce has released it.
        for (uint32_t threadIndex = 0; threadIndex < custom_rs::COMM_THREAD_NUM; ++threadIndex) {
            ThreadHandle commThread = resCtx.threads[custom_rs::COMM_THREAD_BEGIN + threadIndex];
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                coordinator, commThread, COMM_BANK_FREE_NOTIFY_BEGIN + bufferIndex)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                commThread, COMM_BANK_FREE_NOTIFY_BEGIN + bufferIndex, CUSTOM_TIMEOUT)));
        }
    }

    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }
        const uint32_t channelIdx = custom_rs::GetChannelIndex(param.myRank, remoteRank);
        const ChannelInfo &channel = resCtx.channels[channelIdx];
        ThreadHandle commThread = resCtx.threads[
            custom_rs::GetCommThreadIndex(param.myRank, remoteRank)];

        const uint64_t inputOffset = static_cast<uint64_t>(remoteRank) * sliceBytes + tileOffset;
        const uint64_t remoteOffset =
            custom_rs::ScratchOffset(bufferIndex, param.myRank, bankSlotStride, tileSlotStride);
        CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(commThread, channel.handle,
            OffsetAddress(channel.remoteCclMem.addr, remoteOffset), OffsetAddress(param.inputPtr, inputOffset),
            tileBytes)));
    }

    // On the shared Clos thread, queue all eight writes before their records.
    // Mesh threads each carry only one peer and remain fully parallel.
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }
        const uint32_t channelIdx = custom_rs::GetChannelIndex(param.myRank, remoteRank);
        const ChannelInfo &channel = resCtx.channels[channelIdx];
        ThreadHandle commThread = resCtx.threads[
            custom_rs::GetCommThreadIndex(param.myRank, remoteRank)];
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
            commThread, channel.handle, custom_rs::NOTIFY_DATA_BEGIN + bufferIndex)));
    }

    return HCCL_SUCCESS;
}

HcclResult DispatchFifteenSlotAllToAll(const OpParam &param, const AlgResourceCtx &resCtx,
    uint32_t dataNotifyIdx, uint64_t tileOffset, uint64_t tileBytes,
    uint64_t sliceBytes, uint64_t bankOffset, uint64_t slotStride)
{
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }
        const uint32_t channelIdx = custom_rs::GetChannelIndex(param.myRank, remoteRank);
        const ChannelInfo &channel = resCtx.channels[channelIdx];
        ThreadHandle commThread = resCtx.threads[
            custom_rs::GetCommThreadIndex(param.myRank, remoteRank)];

        const uint64_t inputOffset =
            static_cast<uint64_t>(remoteRank) * sliceBytes + tileOffset;
        // This path stores only the fifteen remote contributions. At the
        // destination, source ranks are compacted into slots [0, 15).
        const uint32_t remoteSlot = custom_rs::GetChannelIndex(remoteRank, param.myRank);
        const uint64_t remoteOffset =
            bankOffset + static_cast<uint64_t>(remoteSlot) * slotStride;
        CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(commThread, channel.handle,
            OffsetAddress(channel.remoteCclMem.addr, remoteOffset),
            OffsetAddress(param.inputPtr, inputOffset), tileBytes)));
    }

    // Preserve the proven baseline ordering: queue all writes first, then
    // queue their DATA records on the same per-peer communication threads.
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }
        const uint32_t channelIdx = custom_rs::GetChannelIndex(param.myRank, remoteRank);
        const ChannelInfo &channel = resCtx.channels[channelIdx];
        ThreadHandle commThread = resCtx.threads[
            custom_rs::GetCommThreadIndex(param.myRank, remoteRank)];
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
            commThread, channel.handle, dataNotifyIdx)));
    }
    return HCCL_SUCCESS;
}

HcclResult ReducePair(const AlgResourceCtx &resCtx, ThreadHandle thread, uint32_t bufferIndex,
    uint32_t left, uint32_t right, uint64_t tileCount,
    uint64_t bankSlotStride, uint64_t tileSlotStride)
{
    void *dst = OffsetAddress(
        resCtx.localBuffer.addr,
        custom_rs::ScratchOffset(bufferIndex, left, bankSlotStride, tileSlotStride));
    const void *src = OffsetAddress(
        resCtx.localBuffer.addr,
        custom_rs::ScratchOffset(bufferIndex, right, bankSlotStride, tileSlotStride));
    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
        thread, dst, src, tileCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    return HCCL_SUCCESS;
}

HcclResult ReduceFold(const AlgResourceCtx &resCtx, ThreadHandle thread, uint32_t bufferIndex,
    uint32_t left, uint32_t right, uint32_t slotCount, uint64_t tileCount,
    uint64_t bankSlotStride, uint64_t tileSlotStride)
{
    void *dst = OffsetAddress(
        resCtx.localBuffer.addr,
        custom_rs::ScratchOffset(bufferIndex, left, bankSlotStride, tileSlotStride));
    const void *src = OffsetAddress(
        resCtx.localBuffer.addr,
        custom_rs::ScratchOffset(bufferIndex, right, bankSlotStride, tileSlotStride));
    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
        thread, dst, src, tileCount * slotCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    return HCCL_SUCCESS;
}

HcclResult ReduceFifteenSlotFold(const AlgResourceCtx &resCtx, ThreadHandle thread,
    uint64_t bankOffset, uint32_t left, uint32_t right, uint32_t slotCount,
    uint64_t physicalSlotCount, uint64_t slotStride)
{
    void *dst = OffsetAddress(resCtx.localBuffer.addr,
        bankOffset + static_cast<uint64_t>(left) * slotStride);
    const void *src = OffsetAddress(resCtx.localBuffer.addr,
        bankOffset + static_cast<uint64_t>(right) * slotStride);
    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
        thread, dst, src, physicalSlotCount * slotCount,
        HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    return HCCL_SUCCESS;
}

const ChannelInfo &GetFifteenSlotSourceChannel(const OpParam &param,
    const AlgResourceCtx &resCtx, uint32_t slotIndex)
{
    const uint32_t sourceRank =
        slotIndex < param.myRank ? slotIndex : slotIndex + 1;
    return resCtx.channels[custom_rs::GetChannelIndex(param.myRank, sourceRank)];
}

uint32_t GetGroupDoneNotifyIndex(uint32_t passIndex, uint32_t groupIndex)
{
    return GROUP_DONE_NOTIFY_BEGIN +
        passIndex * custom_rs::REDUCE_GROUP_NUM + groupIndex;
}

HcclResult DispatchFifteenSlotFirstFoldGroups(const OpParam &param,
    const AlgResourceCtx &resCtx, uint32_t dataNotifyIdx, uint32_t passIndex,
    uint64_t bankOffset, uint64_t physicalSlotCount, uint64_t slotStride)
{
    ThreadHandle reduceMain = resCtx.threads[custom_rs::REDUCE_MAIN_THREAD_INDEX];
    // Split the original contiguous seven-pair fold into two large groups:
    // group 0 handles pairs [0, 4), group 1 handles pairs [4, 7).
    // A group starts once only its own source channels are ready.
    constexpr uint32_t groupBegin[custom_rs::REDUCE_GROUP_NUM] = {0, 4};
    constexpr uint32_t groupPairCount[custom_rs::REDUCE_GROUP_NUM] = {4, 3};
    for (uint32_t groupIndex = 0;
        groupIndex < custom_rs::REDUCE_GROUP_NUM; ++groupIndex) {
        const uint32_t begin = groupBegin[groupIndex];
        const uint32_t pairCount = groupPairCount[groupIndex];
        ThreadHandle worker = resCtx.threads[
            custom_rs::REDUCE_GROUP_THREAD_BEGIN + groupIndex];

        for (uint32_t pair = 0; pair < pairCount; ++pair) {
            const ChannelInfo &leftChannel =
                GetFifteenSlotSourceChannel(param, resCtx, begin + pair);
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                worker, leftChannel.handle, dataNotifyIdx, CUSTOM_TIMEOUT)));
        }
        for (uint32_t pair = 0; pair < pairCount; ++pair) {
            const ChannelInfo &rightChannel =
                GetFifteenSlotSourceChannel(param, resCtx, begin + 8 + pair);
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                worker, rightChannel.handle, dataNotifyIdx, CUSTOM_TIMEOUT)));
        }

        CHK_RET(ReduceFifteenSlotFold(resCtx, worker, bankOffset,
            begin, begin + 8, pairCount, physicalSlotCount, slotStride));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            worker, reduceMain,
            GetGroupDoneNotifyIndex(passIndex, groupIndex))));
    }
    return HCCL_SUCCESS;
}

HcclResult DispatchFifteenSlotReduce(const OpParam &param, const AlgResourceCtx &resCtx,
    uint32_t dataNotifyIdx, uint32_t bankIndex, uint32_t passIndex,
    uint64_t tileOffset, uint64_t tileBytes, uint64_t sliceBytes,
    uint64_t bankOffset, uint64_t slotStride, bool releaseBank)
{
    ThreadHandle coordinator = resCtx.threads[custom_rs::COORDINATOR_THREAD_INDEX];
    ThreadHandle reduceMain = resCtx.threads[custom_rs::REDUCE_MAIN_THREAD_INDEX];
    const uint64_t tileCount = tileBytes / sizeof(float);
    const uint64_t physicalSlotCount = slotStride / sizeof(float);
    const uint64_t selfInputOffset =
        static_cast<uint64_t>(param.myRank) * sliceBytes + tileOffset;

    // recvBuf acts as the sixteenth slot, so scratch only needs the fifteen
    // remote contributions. This copy overlaps the independent A2A writes.
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(reduceMain,
        OffsetAddress(param.outputPtr, tileOffset),
        OffsetAddress(param.inputPtr, selfInputOffset), tileBytes)));

    CHK_PRT_RET(passIndex >= FIFTEEN_SLOT_PASS_NUM,
        HCCL_ERROR("Invalid fifteen-slot pass index[%u]", passIndex), HCCL_E_INTERNAL);
    CHK_RET(DispatchFifteenSlotFirstFoldGroups(param, resCtx, dataNotifyIdx,
        passIndex, bankOffset, physicalSlotCount, slotStride));

    // Slot 7 is not part of the first seven-pair fold. Main waits for it and
    // for both grouped folds before entering the dependent 8 -> 4 layer.
    const ChannelInfo &slot7Channel =
        GetFifteenSlotSourceChannel(param, resCtx, 7);
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(reduceMain,
        slot7Channel.handle, dataNotifyIdx, CUSTOM_TIMEOUT)));
    for (uint32_t groupIndex = 0;
        groupIndex < custom_rs::REDUCE_GROUP_NUM; ++groupIndex) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            reduceMain, GetGroupDoneNotifyIndex(passIndex, groupIndex),
            CUSTOM_TIMEOUT)));
    }

    // The remaining folds stay as the original three large contiguous calls.
    CHK_RET(ReduceFifteenSlotFold(resCtx, reduceMain, bankOffset, 0, 4, 4,
        physicalSlotCount, slotStride));
    CHK_RET(ReduceFifteenSlotFold(resCtx, reduceMain, bankOffset, 0, 2, 2,
        physicalSlotCount, slotStride));
    CHK_RET(ReduceFifteenSlotFold(resCtx, reduceMain, bankOffset, 0, 1, 1,
        physicalSlotCount, slotStride));

    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(reduceMain,
        OffsetAddress(param.outputPtr, tileOffset),
        OffsetAddress(resCtx.localBuffer.addr, bankOffset), tileCount,
        HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    if (releaseBank) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            reduceMain, coordinator, BANK_FREE_NOTIFY_BEGIN + bankIndex)));
    }
    return HCCL_SUCCESS;
}

HcclResult WaitFifteenSlotBankReusable(const AlgResourceCtx &resCtx, uint32_t bufferIndex)
{
    ThreadHandle coordinator = resCtx.threads[custom_rs::COORDINATOR_THREAD_INDEX];
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        coordinator, BANK_FREE_NOTIFY_BEGIN + bufferIndex, CUSTOM_TIMEOUT)));
    CHK_RET(NotifyAllPeersReady(coordinator, resCtx));

    // Keep the proven baseline reuse ordering: the destination first releases
    // its bank, all ranks exchange READY, then Thread 0 opens the local gate on
    // every communication stream before the reused-bank writes are submitted.
    for (uint32_t threadIndex = 0; threadIndex < custom_rs::COMM_THREAD_NUM; ++threadIndex) {
        ThreadHandle commThread = resCtx.threads[custom_rs::COMM_THREAD_BEGIN + threadIndex];
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            coordinator, commThread, COMM_BANK_FREE_NOTIFY_BEGIN + bufferIndex)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            commThread, COMM_BANK_FREE_NOTIFY_BEGIN + bufferIndex, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult DispatchDeterministicTreeReduce(const OpParam &param, const AlgResourceCtx &resCtx,
    uint32_t bufferIndex, uint64_t tileIndex, uint64_t tileNum, uint64_t tileOffset,
    uint64_t tileBytes, uint64_t bankSlotStride, uint64_t tileSlotStride)
{
    ThreadHandle coordinator = resCtx.threads[custom_rs::COORDINATOR_THREAD_INDEX];
    ThreadHandle reduceMain = resCtx.threads[custom_rs::REDUCE_THREAD_BEGIN];
    const uint64_t tileCount = tileBytes / sizeof(float);

    const uint64_t sliceBytes = param.count * sizeof(float);
    const uint64_t selfInputOffset = static_cast<uint64_t>(param.myRank) * sliceBytes + tileOffset;
    const uint64_t selfScratchOffset =
        custom_rs::ScratchOffset(bufferIndex, param.myRank, bankSlotStride, tileSlotStride);
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(reduceMain,
        OffsetAddress(resCtx.localBuffer.addr, selfScratchOffset),
        OffsetAddress(param.inputPtr, selfInputOffset), tileBytes)));

    // Reduce Main has already passed its one-time Thread-0 start gate. It can
    // now consume the matching Channel DATA notifications directly.
    for (const auto &channel : resCtx.channels) {
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(reduceMain,
            channel.handle, custom_rs::NOTIFY_DATA_BEGIN + bufferIndex, CUSTOM_TIMEOUT)));
    }

    if (tileSlotStride == tileBytes) {
        // The current tile uses a compact per-tile stride. Fold contiguous
        // halves in four deterministic operations, including a large tail tile.
        CHK_RET(ReduceFold(resCtx, reduceMain, bufferIndex, 0, 8, 8,
            tileCount, bankSlotStride, tileSlotStride));
        CHK_RET(ReduceFold(resCtx, reduceMain, bufferIndex, 0, 4, 4,
            tileCount, bankSlotStride, tileSlotStride));
        CHK_RET(ReduceFold(resCtx, reduceMain, bufferIndex, 0, 2, 2,
            tileCount, bankSlotStride, tileSlotStride));
        CHK_RET(ReduceFold(resCtx, reduceMain, bufferIndex, 0, 1, 1,
            tileCount, bankSlotStride, tileSlotStride));
    } else {
        // An unaligned large tile keeps the fixed slot stride. Preserve the
        // balanced deterministic tree with fifteen padding-safe operations.
        for (uint32_t step = 1; step < custom_rs::RANK_SIZE; step *= 2) {
            for (uint32_t left = 0; left < custom_rs::RANK_SIZE; left += step * 2) {
                CHK_RET(ReducePair(resCtx, reduceMain, bufferIndex, left, left + step,
                    tileCount, bankSlotStride, tileSlotStride));
            }
        }
    }

    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(reduceMain,
        OffsetAddress(param.outputPtr, tileOffset),
        OffsetAddress(resCtx.localBuffer.addr,
            custom_rs::ScratchOffset(bufferIndex, 0, bankSlotStride, tileSlotStride)),
        tileBytes)));
    if (tileIndex + custom_rs::PIPELINE_BUFFER_NUM < tileNum) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(reduceMain, coordinator,
            BANK_FREE_NOTIFY_BEGIN + bufferIndex)));
    }
    return HCCL_SUCCESS;
}

} // namespace

namespace ops_hccl {
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(param.rankSize != custom_rs::RANK_SIZE,
        HCCL_ERROR("Unexpected rankSize[%u]", param.rankSize), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(param.myRank >= param.rankSize,
        HCCL_ERROR("Invalid rank[%u] for rankSize[%u]", param.myRank, param.rankSize), HCCL_E_PARA);
    CHK_PRT_RET(param.dataType != HCCL_DATA_TYPE_FP32 || param.reduceType != HCCL_REDUCE_SUM,
        HCCL_ERROR("Only FP32 SUM is supported"), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(resCtx.localBuffer.addr == nullptr,
        HCCL_ERROR("Local HCCL buffer is null"), HCCL_E_PTR);

    CHK_PRT_RET(resCtx.threads.size() != custom_rs::DIRECT_THREAD_NUM,
        HCCL_ERROR("Unexpected hybrid thread count[%zu]", resCtx.threads.size()), HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.channels.size() != custom_rs::DIRECT_CHANNEL_NUM,
        HCCL_ERROR("Unexpected hybrid channel count[%zu]", resCtx.channels.size()), HCCL_E_INTERNAL);

    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }
        const uint32_t channelIdx = custom_rs::GetChannelIndex(param.myRank, remoteRank);
        CHK_PRT_RET(resCtx.channels[channelIdx].remoteRank != remoteRank,
            HCCL_ERROR("Unexpected channel rank[%u], expected[%u]",
                resCtx.channels[channelIdx].remoteRank, remoteRank),
            HCCL_E_INTERNAL);
    }

    const uint64_t sliceBytes = param.count * sizeof(float);
    if (custom_rs::UseRecursiveHalving(sliceBytes)) {
        // Preserve the original small-data critical path on Thread 0. The
        // additional reduce thread has no tasks in this case.
        return ExecuteRecursiveHalving(param, resCtx,
            resCtx.threads[custom_rs::COORDINATOR_THREAD_INDEX], 0, sliceBytes, sliceBytes);
    }

    uint64_t usableBufferSize = resCtx.localBuffer.size;
    for (const auto &channel : resCtx.channels) {
        CHK_PRT_RET(channel.remoteCclMem.addr == nullptr,
            HCCL_ERROR("Remote HCCL buffer of rank[%u] is null", channel.remoteRank), HCCL_E_PTR);
        usableBufferSize = std::min(usableBufferSize, channel.remoteCclMem.size);
    }

    // If all fifteen remote slices fit after per-tile 4 KiB alignment, use
    // three resident banks with a 40% / 40% / 20% split. The first two passes
    // carry most of the traffic, while the smaller final pass shortens the
    // exposed reduce tail. No bank is reused. Channel notify indices 1, 2 and 0
    // are used once each, avoiding both extra channel resources and multi-record.
    const uint64_t threePassTile0Bytes = custom_rs::AlignDown(
        sliceBytes * 2 / 5, custom_rs::MIN_SLICE_ALIGN);
    const uint64_t threePassTile1Bytes = threePassTile0Bytes;
    const uint64_t threePassTile2Bytes =
        sliceBytes - threePassTile0Bytes - threePassTile1Bytes;
    const uint64_t threePassSlot0Stride =
        (threePassTile0Bytes + custom_rs::MIN_SLICE_ALIGN - 1) /
            custom_rs::MIN_SLICE_ALIGN * custom_rs::MIN_SLICE_ALIGN;
    const uint64_t threePassSlot1Stride = threePassSlot0Stride;
    const uint64_t threePassSlot2Stride =
        (threePassTile2Bytes + custom_rs::MIN_SLICE_ALIGN - 1) /
            custom_rs::MIN_SLICE_ALIGN * custom_rs::MIN_SLICE_ALIGN;
    const uint64_t threePassBank1Offset =
        custom_rs::DIRECT_CHANNEL_NUM * threePassSlot0Stride;
    const uint64_t threePassBank2Offset = threePassBank1Offset +
        custom_rs::DIRECT_CHANNEL_NUM * threePassSlot1Stride;
    const uint64_t threePassScratchBytes = threePassBank2Offset +
        custom_rs::DIRECT_CHANNEL_NUM * threePassSlot2Stride;
    const bool useThreePass = threePassTile0Bytes > 0 && threePassTile2Bytes > 0 &&
        threePassSlot0Stride <= custom_rs::MAX_TRANSFER_BYTES &&
        threePassSlot2Stride <= custom_rs::MAX_TRANSFER_BYTES &&
        threePassScratchBytes <= usableBufferSize;
    if (useThreePass) {
        CHK_RET(StartSlaveThreads(resCtx));
        CHK_RET(DispatchFifteenSlotAllToAll(param, resCtx, custom_rs::NOTIFY_DATA_BEGIN,
            0, threePassTile0Bytes, sliceBytes, 0, threePassSlot0Stride));
        CHK_RET(DispatchFifteenSlotReduce(param, resCtx, custom_rs::NOTIFY_DATA_BEGIN, 0,
            0, 0, threePassTile0Bytes, sliceBytes, 0, threePassSlot0Stride, false));

        CHK_RET(DispatchFifteenSlotAllToAll(param, resCtx, custom_rs::NOTIFY_DATA_BEGIN + 1,
            threePassTile0Bytes, threePassTile1Bytes, sliceBytes,
            threePassBank1Offset, threePassSlot1Stride));
        CHK_RET(DispatchFifteenSlotReduce(param, resCtx, custom_rs::NOTIFY_DATA_BEGIN + 1, 1,
            1, threePassTile0Bytes, threePassTile1Bytes, sliceBytes,
            threePassBank1Offset, threePassSlot1Stride, false));

        CHK_RET(DispatchFifteenSlotAllToAll(param, resCtx, custom_rs::NOTIFY_READY,
            threePassTile0Bytes + threePassTile1Bytes, threePassTile2Bytes, sliceBytes,
            threePassBank2Offset, threePassSlot2Stride));
        CHK_RET(DispatchFifteenSlotReduce(param, resCtx, custom_rs::NOTIFY_READY, 2,
            2, threePassTile0Bytes + threePassTile1Bytes, threePassTile2Bytes, sliceBytes,
            threePassBank2Offset, threePassSlot2Stride, false));
        CHK_RET(FinishSlaveThreads(resCtx));
        return HCCL_SUCCESS;
    }


    // For larger slices, use two equal fifteen-slot banks. This expands each
    // full tile from roughly 12.5 MiB to 13.33 MiB in the default 400 MiB HCCL
    // buffer, reducing the 512 MiB case's final tile from about 7 MiB to about
    // 5.33 MiB. Every bank stride is rounded down to 4 KiB; the final tile uses
    // its own rounded-up 4 KiB stride so all four contiguous folds remain valid.
    const uint64_t fifteenSlotStride = custom_rs::AlignDown(
        usableBufferSize / (custom_rs::PIPELINE_BUFFER_NUM * custom_rs::DIRECT_CHANNEL_NUM),
        custom_rs::MIN_SLICE_ALIGN);
    const uint64_t fifteenSlotRemainder =
        sliceBytes > 2 * fifteenSlotStride ? sliceBytes - 2 * fifteenSlotStride : 0;
    const uint64_t fifteenSlotTailStride = fifteenSlotRemainder == 0 ? 0 :
        (fifteenSlotRemainder + custom_rs::MIN_SLICE_ALIGN - 1) /
            custom_rs::MIN_SLICE_ALIGN * custom_rs::MIN_SLICE_ALIGN;
    const uint64_t fifteenSlotBank1Offset =
        custom_rs::DIRECT_CHANNEL_NUM * fifteenSlotStride;
    const bool useFifteenSlotPipeline = fifteenSlotRemainder > 0 &&
        fifteenSlotStride <= custom_rs::MAX_TRANSFER_BYTES &&
        fifteenSlotTailStride <= fifteenSlotStride;
    if (useFifteenSlotPipeline) {
        CHK_RET(StartSlaveThreads(resCtx));

        CHK_RET(DispatchFifteenSlotAllToAll(param, resCtx, custom_rs::NOTIFY_DATA_BEGIN, 0,
            fifteenSlotStride, sliceBytes, 0, fifteenSlotStride));
        CHK_RET(DispatchFifteenSlotReduce(param, resCtx, custom_rs::NOTIFY_DATA_BEGIN, 0, 0,
            0, fifteenSlotStride, sliceBytes, 0, fifteenSlotStride, true));

        CHK_RET(DispatchFifteenSlotAllToAll(param, resCtx, custom_rs::NOTIFY_DATA_BEGIN + 1,
            fifteenSlotStride,
            fifteenSlotStride, sliceBytes, fifteenSlotBank1Offset, fifteenSlotStride));
        CHK_RET(DispatchFifteenSlotReduce(param, resCtx, custom_rs::NOTIFY_DATA_BEGIN + 1, 1,
            1, fifteenSlotStride, fifteenSlotStride, sliceBytes,
            fifteenSlotBank1Offset, fifteenSlotStride, false));

        CHK_RET(WaitFifteenSlotBankReusable(resCtx, 0));
        CHK_RET(DispatchFifteenSlotAllToAll(param, resCtx, custom_rs::NOTIFY_DATA_BEGIN,
            2 * fifteenSlotStride,
            fifteenSlotRemainder, sliceBytes, 0, fifteenSlotTailStride));
        CHK_RET(DispatchFifteenSlotReduce(param, resCtx, custom_rs::NOTIFY_DATA_BEGIN, 0,
            2, 2 * fifteenSlotStride, fifteenSlotRemainder, sliceBytes,
            0, fifteenSlotTailStride, false));

        CHK_RET(FinishSlaveThreads(resCtx));
        return HCCL_SUCCESS;
    }

    const uint64_t slotStride = std::min(custom_rs::MAX_TRANSFER_BYTES,
        custom_rs::AlignDown(usableBufferSize / custom_rs::SCRATCH_SLOT_NUM, custom_rs::MIN_SLICE_ALIGN));
    CHK_PRT_RET(slotStride < sizeof(float),
        HCCL_ERROR("HCCL buffer[%llu] is too small", static_cast<unsigned long long>(usableBufferSize)),
        HCCL_E_INTERNAL);

    uint64_t largeTileNum = 0;
    uint64_t remainingBytes = sliceBytes;
    while (!custom_rs::UseRecursiveHalving(remainingBytes)) {
        const uint64_t tileBytes = std::min(slotStride, remainingBytes);
        ++largeTileNum;
        remainingBytes -= tileBytes;
    }

    CHK_RET(StartSlaveThreads(resCtx));

    uint64_t tileOffset = 0;
    for (uint64_t tileIndex = 0; tileIndex < largeTileNum; ++tileIndex) {
        const uint64_t tileBytes = std::min(slotStride, sliceBytes - tileOffset);
        const uint32_t bufferIndex =
            static_cast<uint32_t>(tileIndex % custom_rs::PIPELINE_BUFFER_NUM);
        CHK_PRT_RET(tileBytes % sizeof(float) != 0,
            HCCL_ERROR("Tile bytes[%llu] is not FP32 aligned", static_cast<unsigned long long>(tileBytes)),
            HCCL_E_INTERNAL);
        const uint64_t tileSlotStride = custom_rs::UseCompactTileStride(tileBytes)
            ? tileBytes
            : slotStride;
        CHK_RET(DispatchAllToAllExchange(param, resCtx, bufferIndex, tileIndex,
            tileOffset, tileBytes, sliceBytes, slotStride, tileSlotStride));
        CHK_RET(DispatchDeterministicTreeReduce(param, resCtx, bufferIndex, tileIndex,
            largeTileNum, tileOffset, tileBytes, slotStride, tileSlotStride));
        tileOffset += tileBytes;
    }

    const uint64_t tailBytes = sliceBytes - tileOffset;
    if (tailBytes > 0) {
        CHK_RET(ExecuteRecursiveHalving(param, resCtx,
            resCtx.threads[custom_rs::REDUCE_THREAD_BEGIN],
            tileOffset, tailBytes, sliceBytes));
    }

    CHK_RET(FinishSlaveThreads(resCtx));

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
