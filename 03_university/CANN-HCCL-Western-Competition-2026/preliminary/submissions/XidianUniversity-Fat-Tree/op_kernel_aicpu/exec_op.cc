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

namespace ops_hccl {
namespace {
constexpr uint32_t RANK_SIZE = 16;
constexpr uint32_t PEER_NUM = RANK_SIZE - 1;
constexpr uint32_t SERVER_RANK_SIZE = 8;
constexpr uint32_t SMALL_STEP_NUM = 4;

constexpr uint64_t SMALL_DATA_BYTES = 1ULL * 1024 * 1024;
constexpr uint32_t SMALL_MASKS[SMALL_STEP_NUM] = {8, 9, 10, 12};
constexpr uint32_t SMALL_NOTIFY_DATA = 0;

// Large Latin uses four independent channel-notify ranges. Reusing one id for both seed and done can let the
// checker match the wrong Record/Wait pair because channel notify is a one-bit token, not a counter.
constexpr uint32_t LARGE_NOTIFY_SEED_BASE = 0;
constexpr uint32_t LARGE_NOTIFY_DONE_BASE = PEER_NUM;
constexpr uint32_t LARGE_NOTIFY_REDUCE_BARRIER = 2 * PEER_NUM;
constexpr uint32_t LARGE_NOTIFY_AG_BARRIER = 2 * PEER_NUM + 1;
constexpr uint32_t WORKER_NOTIFY_START = 0;
constexpr uint64_t ALIGN_BYTES = 16ULL * 1024;
constexpr uint64_t ALIGN_COUNT = ALIGN_BYTES / sizeof(float);

struct DataRange {
    uint64_t offset = 0;
    uint64_t count = 0;
};

void *PtrOffset(void *ptr, uint64_t offset)
{
    return static_cast<void *>(static_cast<uint8_t *>(ptr) + offset);
}

const void *ConstPtrOffset(const void *ptr, uint64_t offset)
{
    return static_cast<const void *>(static_cast<const uint8_t *>(ptr) + offset);
}

uint64_t AlignDown(uint64_t value, uint64_t align)
{
    return value / align * align;
}

uint64_t AlignUp(uint64_t value, uint64_t align)
{
    return (value + align - 1) / align * align;
}

uint64_t GetMinCclBufferSize(const AlgResourceCtx &resCtx)
{
    return resCtx.minCclBufferSize;
}

DataRange GetBalancedRange(uint64_t totalCount, uint32_t partNum, uint32_t partIdx)
{
    uint64_t baseCount = totalCount / partNum;
    uint64_t extraCount = totalCount % partNum;
    DataRange range;
    range.offset = static_cast<uint64_t>(partIdx) * baseCount + std::min<uint64_t>(partIdx, extraCount);
    range.count = baseCount + (partIdx < extraCount ? 1 : 0);
    return range;
}

// Split a range evenly in 16KB units; only the final sub-16KB remainder stays on the last part.
DataRange GetAlignedBalancedRange(uint64_t totalCount, uint32_t partNum, uint32_t partIdx)
{
    uint64_t alignedTotalCount = AlignDown(totalCount, ALIGN_COUNT);
    uint64_t alignedUnitNum = alignedTotalCount / ALIGN_COUNT;
    if (alignedUnitNum < partNum) {
        return GetBalancedRange(totalCount, partNum, partIdx);
    }

    uint64_t baseUnitNum = alignedUnitNum / partNum;
    uint64_t extraUnitNum = alignedUnitNum % partNum;
    uint64_t unitOffset = static_cast<uint64_t>(partIdx) * baseUnitNum +
        std::min<uint64_t>(partIdx, extraUnitNum);
    uint64_t unitCount = baseUnitNum + (partIdx < extraUnitNum ? 1 : 0);
    DataRange range;
    range.offset = unitOffset * ALIGN_COUNT;
    range.count = unitCount * ALIGN_COUNT;
    if (partIdx + 1 == partNum) {
        range.count += totalCount - alignedTotalCount;
    }
    return range;
}

DataRange GetOwnerRange(uint64_t totalCount, uint32_t ownerRank)
{
    return GetAlignedBalancedRange(totalCount, RANK_SIZE, ownerRank);
}

DataRange GetLaneRange(uint64_t ownerCount, uint32_t laneIdx)
{
    return GetAlignedBalancedRange(ownerCount, PEER_NUM, laneIdx);
}

uint32_t GetPeerRank(uint32_t ownerRank, uint32_t peerIdx)
{
    return peerIdx < ownerRank ? peerIdx : peerIdx + 1;
}

uint32_t GetPeerIndex(uint32_t ownerRank, uint32_t peerRank)
{
    return peerRank < ownerRank ? peerRank : peerRank - 1;
}

bool IsInterServer(uint32_t rankA, uint32_t rankB)
{
    return rankA / SERVER_RANK_SIZE != rankB / SERVER_RANK_SIZE;
}

HcclResult GetLargeChannel(
    const OpParam &param, const AlgResourceCtx &resCtx, uint32_t remoteRank, const ChannelInfo *&channel)
{
    CHK_PRT_RET(remoteRank >= RANK_SIZE || remoteRank == param.myRank,
        HCCL_ERROR("Invalid remote rank[%u] for rank[%u]", remoteRank, param.myRank), HCCL_E_INTERNAL);
    uint32_t channelIdx = GetPeerIndex(param.myRank, remoteRank);
    CHK_PRT_RET(channelIdx >= resCtx.channelNum || resCtx.channels[channelIdx].remoteRank != remoteRank,
        HCCL_ERROR("Channel mapping mismatch, remoteRank[%u], channelIdx[%u]", remoteRank, channelIdx),
        HCCL_E_INTERNAL);
    channel = &resCtx.channels[channelIdx];
    return HCCL_SUCCESS;
}

HcclResult ReleaseWorker(ThreadHandle mainThread, ThreadHandle workerThread)
{
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyRecordOnThread(mainThread, workerThread, WORKER_NOTIFY_START)));
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyWaitOnThread(workerThread, WORKER_NOTIFY_START, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult RecordWorkerDone(ThreadHandle workerThread, ThreadHandle mainThread, uint32_t workerIdx)
{
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyRecordOnThread(workerThread, mainThread, workerIdx)));
    return HCCL_SUCCESS;
}

HcclResult WaitAllWorkers(const AlgResourceCtx &resCtx)
{
    for (uint32_t workerIdx = 0; workerIdx < PEER_NUM; workerIdx++) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(resCtx.aicpuThread, workerIdx, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

// Validate small-data resources and return the CCL slot size used by each CLOS step.
HcclResult PrepareSmallClos(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t &slotBytes)
{
    CHK_PRT_RET(resCtx.threadNum != RANK_SIZE || resCtx.channelNum != PEER_NUM,
        HCCL_ERROR("Invalid small resources, threads[%u], channels[%u]", resCtx.threadNum, resCtx.channelNum),
        HCCL_E_INTERNAL);
    uint64_t dataBytes = param.count * sizeof(float);
    slotBytes = AlignUp(dataBytes, ALIGN_BYTES);
    CHK_PRT_RET(slotBytes == 0 || slotBytes > GetMinCclBufferSize(resCtx) / SMALL_STEP_NUM,
        HCCL_ERROR("Small CCL buffer is too small, slotBytes[%llu]", static_cast<unsigned long long>(slotBytes)),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

// Run the in-place small-data path; output already contains the local input, so only CCL slots are refreshed.
HcclResult RunSmallClosReadReduceInPlace(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t slotBytes)
{
    uint64_t dataBytes = param.count * sizeof(float);
    ThreadHandle thread = resCtx.aicpuThread;
    for (uint32_t step = 0; step < SMALL_STEP_NUM; step++) {
        uint32_t remoteRank = param.myRank ^ SMALL_MASKS[step];
        const ChannelInfo *channel = nullptr;
        CHK_RET(GetLargeChannel(param, resCtx, remoteRank, channel));

        uint64_t slotOffset = static_cast<uint64_t>(step) * slotBytes;
        void *localSend = PtrOffset(resCtx.localBuffer.addr, slotOffset);
        const void *remoteSend = ConstPtrOffset(channel->remoteCclMem.addr, slotOffset);

        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, localSend, param.outputPtr, dataBytes)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, channel->handle, SMALL_NOTIFY_DATA)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            thread, channel->handle, SMALL_NOTIFY_DATA, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(HcommReadReduceOnThread(thread, channel->handle, param.outputPtr,
            remoteSend, param.count, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    }
    return HCCL_SUCCESS;
}

// Run the out-of-place small-data path; the first step seeds both output and the first CCL slot from input.
HcclResult RunSmallClosReadReduceOutOfPlace(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t slotBytes)
{
    uint64_t dataBytes = param.count * sizeof(float);
    ThreadHandle thread = resCtx.aicpuThread;
    for (uint32_t step = 0; step < SMALL_STEP_NUM; step++) {
        uint32_t remoteRank = param.myRank ^ SMALL_MASKS[step];
        const ChannelInfo *channel = nullptr;
        CHK_RET(GetLargeChannel(param, resCtx, remoteRank, channel));

        uint64_t slotOffset = static_cast<uint64_t>(step) * slotBytes;
        void *localSend = PtrOffset(resCtx.localBuffer.addr, slotOffset);
        const void *remoteSend = ConstPtrOffset(channel->remoteCclMem.addr, slotOffset);

        if (step == 0) {
            CHK_RET(static_cast<HcclResult>(
                HcommLocalCopyOnThread(thread, param.outputPtr, param.inputPtr, dataBytes)));
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, localSend, param.inputPtr, dataBytes)));
        } else {
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, localSend, param.outputPtr, dataBytes)));
        }
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, channel->handle, SMALL_NOTIFY_DATA)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            thread, channel->handle, SMALL_NOTIFY_DATA, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(HcommReadReduceOnThread(thread, channel->handle, param.outputPtr,
            remoteSend, param.count, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    }
    return HCCL_SUCCESS;
}

// Run small data through four CLOS-heavy Recursive Doubling levels using remote ReadReduce.
HcclResult RunSmallClosReadReduce(const OpParam &param, const AlgResourceCtx &resCtx)
{
    uint64_t slotBytes = 0;
    CHK_RET(PrepareSmallClos(param, resCtx, slotBytes));
    if (param.inputPtr == param.outputPtr) {
        return RunSmallClosReadReduceInPlace(param, resCtx, slotBytes);
    }
    return RunSmallClosReadReduceOutOfPlace(param, resCtx, slotBytes);
}

HcclResult RunLatinLane(
    const OpParam &param, const AlgResourceCtx &resCtx, uint32_t workerIdx, ThreadHandle thread)
{
    uint32_t localLaneIdx = workerIdx;
    const ChannelInfo &ownerChannel = resCtx.channels[workerIdx];
    uint32_t remoteOwner = ownerChannel.remoteRank;
    uint32_t sourceIdxAtOwner = GetPeerIndex(remoteOwner, param.myRank);
    DataRange remoteOwnerRange = GetOwnerRange(param.count, remoteOwner);

    for (uint32_t round = 0; round < PEER_NUM; round++) {
        uint32_t localSourceIdx = (localLaneIdx + PEER_NUM - round) % PEER_NUM;
        uint32_t localSourceRank = GetPeerRank(param.myRank, localSourceIdx);
        const ChannelInfo *sourceChannel = nullptr;
        CHK_RET(GetLargeChannel(param, resCtx, localSourceRank, sourceChannel));

        // Seed the token for this local lane before waiting for the remote owner token. Every rank does this first,
        // so round zero cannot form a wait cycle. Later rounds are released only after this lane's prior write is done.
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
            thread, sourceChannel->handle, LARGE_NOTIFY_SEED_BASE + localLaneIdx)));

        uint32_t remoteLaneIdx = (sourceIdxAtOwner + round) % PEER_NUM;
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread, ownerChannel.handle,
            LARGE_NOTIFY_SEED_BASE + remoteLaneIdx, CUSTOM_TIMEOUT)));

        DataRange remoteLaneRange = GetLaneRange(remoteOwnerRange.count, remoteLaneIdx);
        if (remoteLaneRange.count != 0) {
            uint64_t inputOffset = (remoteOwnerRange.offset + remoteLaneRange.offset) * sizeof(float);
            void *remoteDst = PtrOffset(ownerChannel.remoteCclMem.addr, remoteLaneRange.offset * sizeof(float));
            CHK_RET(static_cast<HcclResult>(HcommWriteReduceOnThread(thread, ownerChannel.handle, remoteDst,
                ConstPtrOffset(param.inputPtr, inputOffset), remoteLaneRange.count, HCOMM_DATA_TYPE_FP32,
                HCOMM_REDUCE_SUM)));
        }
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
            thread, ownerChannel.handle, LARGE_NOTIFY_DONE_BASE + remoteLaneIdx)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread, sourceChannel->handle,
            LARGE_NOTIFY_DONE_BASE + localLaneIdx, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult QueueOwnerReduce(const OpParam &param, const AlgResourceCtx &resCtx)
{
    DataRange localOwnerRange = GetOwnerRange(param.count, param.myRank);
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resCtx.aicpuThread, resCtx.localBuffer.addr,
        ConstPtrOffset(param.inputPtr, localOwnerRange.offset * sizeof(float)),
        localOwnerRange.count * sizeof(float))));

    for (uint32_t pass = 0; pass < 2; pass++) {
        for (uint32_t workerIdx = 0; workerIdx < PEER_NUM; workerIdx++) {
            bool interServer = IsInterServer(param.myRank, resCtx.channels[workerIdx].remoteRank);
            if ((pass == 0 && !interServer) || (pass == 1 && interServer)) {
                continue;
            }
            ThreadHandle worker = resCtx.threads[workerIdx + 1];
            CHK_RET(ReleaseWorker(resCtx.aicpuThread, worker));
            CHK_RET(RunLatinLane(param, resCtx, workerIdx, worker));
            CHK_RET(RecordWorkerDone(worker, resCtx.aicpuThread, workerIdx));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult QueueOwnerAllGather(const OpParam &param, const AlgResourceCtx &resCtx)
{
    DataRange localOwnerRange = GetOwnerRange(param.count, param.myRank);
    for (uint32_t pass = 0; pass < 2; pass++) {
        for (uint32_t workerIdx = 0; workerIdx < PEER_NUM; workerIdx++) {
            ThreadHandle worker = resCtx.threads[workerIdx + 1];
            const ChannelInfo &channel = resCtx.channels[workerIdx];
            bool interServer = IsInterServer(param.myRank, channel.remoteRank);
            if ((pass == 0 && !interServer) || (pass == 1 && interServer)) {
                continue;
            }
            DataRange remoteOwnerRange = GetOwnerRange(param.count, channel.remoteRank);
            CHK_RET(ReleaseWorker(resCtx.aicpuThread, worker));

            // Fold the reduce-completion barrier into AG: only the peer that owns this block must be ready before
            // this read starts. CLOS peers are released first to hide their longer startup latency.
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(worker, channel.handle, LARGE_NOTIFY_REDUCE_BARRIER)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                worker, channel.handle, LARGE_NOTIFY_REDUCE_BARRIER, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(HcommReadOnThread(worker, channel.handle,
                PtrOffset(param.outputPtr, remoteOwnerRange.offset * sizeof(float)), channel.remoteCclMem.addr,
                remoteOwnerRange.count * sizeof(float))));

            // Keep the AG done handshake so this rank does not finish while a peer may still be reading its CCL block.
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(worker, channel.handle, LARGE_NOTIFY_AG_BARRIER)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                worker, channel.handle, LARGE_NOTIFY_AG_BARRIER, CUSTOM_TIMEOUT)));
            CHK_RET(RecordWorkerDone(worker, resCtx.aicpuThread, workerIdx));
        }
    }
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resCtx.aicpuThread,
        PtrOffset(param.outputPtr, localOwnerRange.offset * sizeof(float)), resCtx.localBuffer.addr,
        localOwnerRange.count * sizeof(float))));
    return HCCL_SUCCESS;
}

HcclResult RunOwnerBlockLatin(const OpParam &param, const AlgResourceCtx &resCtx)
{
    CHK_PRT_RET(resCtx.threadNum != RANK_SIZE || resCtx.channelNum != PEER_NUM,
        HCCL_ERROR("Invalid large resources, threads[%u], channels[%u]", resCtx.threadNum, resCtx.channelNum),
        HCCL_E_INTERNAL);
    uint64_t maxOwnerCount = 0;
    for (uint32_t ownerRank = 0; ownerRank < RANK_SIZE; ownerRank++) {
        maxOwnerCount = std::max(maxOwnerCount, GetOwnerRange(param.count, ownerRank).count);
    }
    CHK_PRT_RET(maxOwnerCount > GetMinCclBufferSize(resCtx) / sizeof(float),
        HCCL_ERROR("Owner block is too large, ownerCount[%llu]", static_cast<unsigned long long>(maxOwnerCount)),
        HCCL_E_INTERNAL);

    CHK_RET(QueueOwnerReduce(param, resCtx));
    CHK_RET(WaitAllWorkers(resCtx));

    CHK_RET(QueueOwnerAllGather(param, resCtx));
    CHK_RET(WaitAllWorkers(resCtx));
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(param.rankSize != RANK_SIZE, HCCL_ERROR("Unsupported rankSize[%u]", param.rankSize), HCCL_E_PARA);
    CHK_PRT_RET(param.dataType != HCCL_DATA_TYPE_FP32 || param.reduceType != HCCL_REDUCE_SUM,
        HCCL_ERROR("Only float32 sum is supported"), HCCL_E_PARA);

    uint64_t dataBytes = param.count * sizeof(float);
    if (dataBytes <= SMALL_DATA_BYTES) {
        return RunSmallClosReadReduce(param, resCtx);
    }
    return RunOwnerBlockLatin(param, resCtx);
}
} // namespace ops_hccl
