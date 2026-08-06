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
#include <cstdint>
#include <limits>

#include "custom.h"
#include "log.h"
#include "exec_op.h"

namespace ops_hccl {
namespace {
constexpr uint64_t CCL_SLOT_ALIGNMENT = 256;
constexpr uint64_t SMALL_INPUT_BYTES = 1ULL * 1024 * 1024;
constexpr uint64_t BALANCED_INPUT_THRESHOLD_BYTES = 400ULL * 1024 * 1024;
constexpr uint64_t INTERLEAVED_ORDER_MIN_INPUT_BYTES = 64ULL * 1024 * 1024;
constexpr uint32_t CHANNEL_NOTIFY_READY = 0;
constexpr uint32_t CHANNEL_NOTIFY_DONE = 1;
constexpr uint32_t CHANNEL_NOTIFY_NUM_PER_BANK = 2;
constexpr uint32_t WORKER_START_NOTIFY = 0;
constexpr uint32_t TREE_ROUND1_START_NOTIFY = 1;
constexpr uint32_t TREE_ROUND2_START_NOTIFY = 2;
constexpr uint32_t TREE_ROUND3_START_NOTIFY = 3;
constexpr uint32_t TREE_ROUND4_START_NOTIFY = 4;
constexpr uint32_t TREE_DONE_NOTIFY = 14;
constexpr uint32_t PARALLEL_BANK_NUM = 2;
constexpr uint32_t CONTROL_THREAD_INDEX = 0;
constexpr uint32_t REDUCE_THREAD_BASE = 1;
constexpr uint32_t WORKER_THREAD_BASE = REDUCE_THREAD_BASE + PARALLEL_BANK_NUM;

enum class AlgorithmMode {
    PAPER2_SMALL,
    SINGLE_GLOBAL_TREE,
    BALANCED_GLOBAL_TREE,
};

HcclResult SelectAlgorithmMode(const OpParam &param, uint64_t recvBytes, AlgorithmMode &mode)
{
    if (recvBytes > std::numeric_limits<uint64_t>::max() / param.rankSize) {
        return HCCL_E_PARA;
    }
    const uint64_t inputBytes = recvBytes * param.rankSize;
    if (inputBytes < SMALL_INPUT_BYTES) {
        mode = AlgorithmMode::PAPER2_SMALL;
        return HCCL_SUCCESS;
    }
    mode = inputBytes >= BALANCED_INPUT_THRESHOLD_BYTES ?
        AlgorithmMode::BALANCED_GLOBAL_TREE : AlgorithmMode::SINGLE_GLOBAL_TREE;
    return HCCL_SUCCESS;
}

void *Offset(void *base, uint64_t offset)
{
    return static_cast<void *>(static_cast<uint8_t *>(base) + offset);
}

const void *Offset(const void *base, uint64_t offset)
{
    return static_cast<const void *>(static_cast<const uint8_t *>(base) + offset);
}

HcclResult CheckRange(uint64_t offset, uint64_t bytes, uint64_t size, const char *bufferName)
{
    if (offset > size || bytes > size - offset) {
        HCCL_ERROR("%s range overflow, offset[%llu], bytes[%llu], size[%llu]", bufferName,
            static_cast<unsigned long long>(offset), static_cast<unsigned long long>(bytes),
            static_cast<unsigned long long>(size));
        return HCCL_E_MEMORY;
    }
    return HCCL_SUCCESS;
}

bool IsPowerOfTwo(uint32_t value)
{
    return value != 0 && (value & (value - 1)) == 0;
}

const ChannelInfo *FindChannel(const AlgResourceCtx &resCtx, uint32_t remoteRank)
{
    for (const ChannelInfo &channel : resCtx.channels) {
        if (channel.remoteRank == remoteRank) {
            return &channel;
        }
    }
    return nullptr;
}

bool UseInterleavedOrder(const OpParam &param, uint64_t recvBytes)
{
    return param.rankSize == 16 && recvBytes * param.rankSize >= INTERLEAVED_ORDER_MIN_INPUT_BYTES;
}

uint32_t PhysicalToLogicalRank(uint32_t physicalRank, uint32_t rankSize, bool interleaved)
{
    if (!interleaved) {
        return physicalRank;
    }
    return 2 * (physicalRank % (rankSize / 2)) + physicalRank / (rankSize / 2);
}

uint32_t LogicalToPhysicalRank(uint32_t logicalRank, uint32_t rankSize, bool interleaved)
{
    if (!interleaved) {
        return logicalRank;
    }
    return logicalRank / 2 + (logicalRank % 2) * (rankSize / 2);
}

uint32_t ContiguousPhysicalBlocks(
    uint32_t logicalStart, uint32_t blockOffset, uint32_t blockCount, uint32_t rankSize, bool interleaved)
{
    uint32_t contiguousBlocks = 1;
    uint32_t previousPhysicalRank = LogicalToPhysicalRank(
        (logicalStart + blockOffset) % rankSize, rankSize, interleaved);
    while (blockOffset + contiguousBlocks < blockCount) {
        const uint32_t physicalRank = LogicalToPhysicalRank(
            (logicalStart + blockOffset + contiguousBlocks) % rankSize, rankSize, interleaved);
        if (physicalRank != previousPhysicalRank + 1) {
            break;
        }
        previousPhysicalRank = physicalRank;
        ++contiguousBlocks;
    }
    return contiguousBlocks;
}

HcclResult CopyPaperBlocks(ThreadHandle thread, void *destination, const void *source, uint32_t logicalStart,
    uint32_t blockCount, uint32_t rankSize, uint64_t blockBytes, bool interleaved)
{
    uint32_t blockOffset = 0;
    while (blockOffset < blockCount) {
        const uint32_t physicalStart = LogicalToPhysicalRank(
            (logicalStart + blockOffset) % rankSize, rankSize, interleaved);
        const uint32_t contiguousBlocks =
            ContiguousPhysicalBlocks(logicalStart, blockOffset, blockCount, rankSize, interleaved);
        const uint64_t copyBytes = static_cast<uint64_t>(contiguousBlocks) * blockBytes;
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread,
            Offset(destination, static_cast<uint64_t>(blockOffset) * blockBytes),
            Offset(source, static_cast<uint64_t>(physicalStart) * blockBytes), copyBytes)));
        blockOffset += contiguousBlocks;
    }
    return HCCL_SUCCESS;
}

HcclResult QueueFirstPaperRound(const OpParam &param, const ChannelInfo &outgoing, ThreadHandle thread,
    uint32_t logicalStart, uint32_t skip, uint64_t recvBytes, bool interleaved)
{
    uint32_t blockOffset = 0;
    while (blockOffset < skip) {
        const uint32_t physicalStart = LogicalToPhysicalRank(
            (logicalStart + blockOffset) % param.rankSize, param.rankSize, interleaved);
        const uint32_t contiguousBlocks =
            ContiguousPhysicalBlocks(logicalStart, blockOffset, skip, param.rankSize, interleaved);
        const uint64_t writeCount = static_cast<uint64_t>(contiguousBlocks) * param.count;
        void *destination = Offset(outgoing.remoteCclMem.addr, static_cast<uint64_t>(blockOffset) * recvBytes);
        const void *source = Offset(param.inputPtr, static_cast<uint64_t>(physicalStart) * recvBytes);
        blockOffset += contiguousBlocks;
        CHK_RET(static_cast<HcclResult>(HcommWriteReduceOnThread(thread, outgoing.handle, destination, source,
            writeCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    }
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, outgoing.handle, CHANNEL_NOTIFY_DONE)));
    return HCCL_SUCCESS;
}

HcclResult QueuePaperRound(const OpParam &param, const AlgResourceCtx &resCtx, ThreadHandle thread, uint32_t skip,
    uint64_t recvBytes, bool firstRound, bool interleaved)
{
    const uint32_t myLogicalRank = PhysicalToLogicalRank(param.myRank, param.rankSize, interleaved);
    const uint32_t toLogicalRank = (myLogicalRank + skip) % param.rankSize;
    const uint32_t fromLogicalRank = (myLogicalRank + param.rankSize - skip) % param.rankSize;
    const uint32_t toRank = LogicalToPhysicalRank(toLogicalRank, param.rankSize, interleaved);
    const uint32_t fromRank = LogicalToPhysicalRank(fromLogicalRank, param.rankSize, interleaved);
    const ChannelInfo *outgoing = FindChannel(resCtx, toRank);
    const ChannelInfo *incoming = FindChannel(resCtx, fromRank);
    if (outgoing == nullptr || incoming == nullptr) {
        HCCL_ERROR("Missing paper channel, rank[%u], skip[%u], to[%u], from[%u]", param.myRank, skip, toRank,
            fromRank);
        return HCCL_E_INTERNAL;
    }

    const uint64_t sendBytes = static_cast<uint64_t>(skip) * recvBytes;
    if (sendBytes > outgoing->remoteCclMem.size) {
        HCCL_ERROR("Remote CCL buffer is too small, rank[%u], required[%llu], size[%llu]", toRank,
            static_cast<unsigned long long>(sendBytes),
            static_cast<unsigned long long>(outgoing->remoteCclMem.size));
        return HCCL_E_MEMORY;
    }

    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, incoming->handle, CHANNEL_NOTIFY_READY)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        thread, outgoing->handle, CHANNEL_NOTIFY_READY, CUSTOM_TIMEOUT)));
    if (firstRound) {
        const uint32_t logicalStart = (myLogicalRank + skip) % param.rankSize;
        CHK_RET(QueueFirstPaperRound(param, *outgoing, thread, logicalStart, skip, recvBytes, interleaved));
    } else {
        const uint64_t sendCount = static_cast<uint64_t>(skip) * param.count;
        const void *source = Offset(resCtx.localBuffer.addr, sendBytes);
        CHK_RET(static_cast<HcclResult>(HcommWriteReduceOnThread(thread, outgoing->handle,
            outgoing->remoteCclMem.addr, source, sendCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, outgoing->handle, CHANNEL_NOTIFY_DONE)));
    }
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        thread, incoming->handle, CHANNEL_NOTIFY_DONE, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult ExecPaper2Small(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t recvBytes)
{
    if (!IsPowerOfTwo(param.rankSize) || resCtx.threads.size() != 1) {
        HCCL_ERROR("Invalid paper2 small resources, rankSize[%u], threads[%zu]", param.rankSize,
            resCtx.threads.size());
        return HCCL_E_PARA;
    }
    ThreadHandle thread = resCtx.threads[0];
    if (param.rankSize == 1) {
        return static_cast<HcclResult>(HcommLocalCopyOnThread(thread, param.outputPtr, param.inputPtr, recvBytes));
    }

    const uint32_t initialBlockCount = param.rankSize / 2;
    const uint64_t initialBytes = static_cast<uint64_t>(initialBlockCount) * recvBytes;
    if (initialBytes > resCtx.localBuffer.size) {
        HCCL_ERROR("Local CCL buffer is too small, required[%llu], size[%llu]",
            static_cast<unsigned long long>(initialBytes),
            static_cast<unsigned long long>(resCtx.localBuffer.size));
        return HCCL_E_MEMORY;
    }
    const bool interleaved = UseInterleavedOrder(param, recvBytes);
    const uint32_t myLogicalRank = PhysicalToLogicalRank(param.myRank, param.rankSize, interleaved);
    CHK_RET(CopyPaperBlocks(thread, resCtx.localBuffer.addr, param.inputPtr, myLogicalRank, initialBlockCount,
        param.rankSize, recvBytes, interleaved));

    bool firstRound = true;
    for (uint32_t skip = initialBlockCount; skip > 0; skip /= 2) {
        CHK_RET(QueuePaperRound(param, resCtx, thread, skip, recvBytes, firstRound, interleaved));
        firstRound = false;
    }
    return static_cast<HcclResult>(
        HcommLocalCopyOnThread(thread, param.outputPtr, resCtx.localBuffer.addr, recvBytes));
}

HcclResult QueueChannelWrite(const ChannelInfo &channel, ThreadHandle thread, void *destination,
    const void *source, uint64_t bytes, uint32_t notifyBase)
{
    if (notifyBase + CHANNEL_NOTIFY_DONE >= channel.notifyNum) {
        HCCL_ERROR("Channel notify range overflow, base[%u], notifyNum[%u]", notifyBase, channel.notifyNum);
        return HCCL_E_INTERNAL;
    }
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, notifyBase + CHANNEL_NOTIFY_READY)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        thread, channel.handle, notifyBase + CHANNEL_NOTIFY_READY, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(
        HcommWriteOnThread(thread, channel.handle, destination, source, bytes)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, notifyBase + CHANNEL_NOTIFY_DONE)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        thread, channel.handle, notifyBase + CHANNEL_NOTIFY_DONE, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult QueueGlobalMeshWorker(const OpParam &param, const ChannelInfo &channel, ThreadHandle thread,
    uint64_t recvBytes, uint64_t processedBytes, uint64_t bankBase, uint64_t slotStride, uint64_t chunkBytes,
    uint32_t notifyBase)
{
    const uint64_t inputOffset = static_cast<uint64_t>(channel.remoteRank) * recvBytes + processedBytes;
    const uint64_t remoteSlotOffset = bankBase + static_cast<uint64_t>(param.myRank) * slotStride;
    CHK_RET(CheckRange(remoteSlotOffset, chunkBytes, channel.remoteCclMem.size, "remote global-mesh slot"));
    return QueueChannelWrite(channel, thread, Offset(channel.remoteCclMem.addr, remoteSlotOffset),
        Offset(param.inputPtr, inputOffset), chunkBytes, notifyBase);
}

HcclResult QueueTreeReduceTask(const OpParam &param, const AlgResourceCtx &resCtx, ThreadHandle thread,
    uint32_t targetLogicalRank, uint32_t sourceLogicalRank, uint64_t bankBase, uint64_t slotStride,
    uint64_t chunkCount)
{
    const uint32_t targetRank = (param.myRank + targetLogicalRank) % param.rankSize;
    const uint32_t sourceRank = (param.myRank + sourceLogicalRank) % param.rankSize;
    const uint64_t targetOffset = bankBase + static_cast<uint64_t>(targetRank) * slotStride;
    const uint64_t sourceOffset = bankBase + static_cast<uint64_t>(sourceRank) * slotStride;
    return static_cast<HcclResult>(HcommLocalReduceOnThread(thread, Offset(resCtx.localBuffer.addr, targetOffset),
        Offset(resCtx.localBuffer.addr, sourceOffset), chunkCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM));
}

HcclResult QueueGlobalTreeReduce(const OpParam &param, const AlgResourceCtx &resCtx, ThreadHandle coordinator,
    uint32_t workerBase, uint64_t processedBytes, uint64_t bankBase, uint64_t slotStride, uint64_t chunkBytes)
{
    if (param.rankSize != 16 || workerBase + param.rankSize - 1 > resCtx.threads.size()) {
        HCCL_ERROR("Unsupported v7 global tree, rankSize[%u], workerBase[%u], threads[%zu]", param.rankSize,
            workerBase, resCtx.threads.size());
        return HCCL_E_NOT_SUPPORT;
    }
    ThreadHandle workers[15];
    for (uint32_t workerIdx = 0; workerIdx < param.rankSize - 1; ++workerIdx) {
        workers[workerIdx] = resCtx.threads[workerBase + workerIdx];
    }
    const uint64_t chunkCount = chunkBytes / sizeof(float);
    for (uint32_t taskIdx = 0; taskIdx < 8; ++taskIdx) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            coordinator, workers[taskIdx], TREE_ROUND1_START_NOTIFY)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            workers[taskIdx], TREE_ROUND1_START_NOTIFY, CUSTOM_TIMEOUT)));
        CHK_RET(QueueTreeReduceTask(param, resCtx, workers[taskIdx], 2 * taskIdx, 2 * taskIdx + 1,
            bankBase, slotStride, chunkCount));
    }

    const uint32_t round2Targets[4] = {0, 2, 4, 6};
    const uint32_t round2Sources[4] = {1, 3, 5, 7};
    for (uint32_t taskIdx = 0; taskIdx < 4; ++taskIdx) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            workers[round2Sources[taskIdx]], workers[round2Targets[taskIdx]], TREE_ROUND2_START_NOTIFY)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            workers[round2Targets[taskIdx]], TREE_ROUND2_START_NOTIFY, CUSTOM_TIMEOUT)));
        CHK_RET(QueueTreeReduceTask(param, resCtx, workers[round2Targets[taskIdx]], 4 * taskIdx,
            4 * taskIdx + 2, bankBase, slotStride, chunkCount));
    }

    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyRecordOnThread(workers[2], workers[0], TREE_ROUND3_START_NOTIFY)));
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyRecordOnThread(workers[6], workers[4], TREE_ROUND3_START_NOTIFY)));
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyWaitOnThread(workers[0], TREE_ROUND3_START_NOTIFY, CUSTOM_TIMEOUT)));
    CHK_RET(QueueTreeReduceTask(param, resCtx, workers[0], 0, 4, bankBase, slotStride, chunkCount));
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyWaitOnThread(workers[4], TREE_ROUND3_START_NOTIFY, CUSTOM_TIMEOUT)));
    CHK_RET(QueueTreeReduceTask(param, resCtx, workers[4], 8, 12, bankBase, slotStride, chunkCount));

    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyRecordOnThread(workers[4], workers[0], TREE_ROUND4_START_NOTIFY)));
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyWaitOnThread(workers[0], TREE_ROUND4_START_NOTIFY, CUSTOM_TIMEOUT)));
    CHK_RET(QueueTreeReduceTask(param, resCtx, workers[0], 0, 8, bankBase, slotStride, chunkCount));
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyRecordOnThread(workers[0], coordinator, TREE_DONE_NOTIFY)));
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyWaitOnThread(coordinator, TREE_DONE_NOTIFY, CUSTOM_TIMEOUT)));

    const uint64_t rootOffset = bankBase + static_cast<uint64_t>(param.myRank) * slotStride;
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(coordinator, Offset(param.outputPtr, processedBytes),
        Offset(resCtx.localBuffer.addr, rootOffset), chunkBytes)));
    return HCCL_SUCCESS;
}

uint64_t GetStagingBufferSize(const AlgResourceCtx &resCtx)
{
    uint64_t stagingBufferSize = resCtx.localBuffer.size;
    for (const ChannelInfo &channel : resCtx.channels) {
        stagingBufferSize = std::min(stagingBufferSize, channel.remoteCclMem.size);
    }
    return stagingBufferSize;
}

HcclResult QueueBank(const OpParam &param, const AlgResourceCtx &resCtx, uint32_t bankIdx, uint64_t recvBytes,
    uint64_t processedBytes, uint64_t slotStride, uint64_t bankStride, uint64_t chunkBytes)
{
    const uint32_t peerNum = param.rankSize - 1;
    const uint32_t workerBase = WORKER_THREAD_BASE + bankIdx * peerNum;
    const uint32_t channelNotifyBase = bankIdx * CHANNEL_NOTIFY_NUM_PER_BANK;
    const uint64_t bankBase = static_cast<uint64_t>(bankIdx) * bankStride;
    ThreadHandle controlThread = resCtx.threads[CONTROL_THREAD_INDEX];
    ThreadHandle reduceThread = resCtx.threads[REDUCE_THREAD_BASE + bankIdx];

    const uint64_t ownInputOffset = static_cast<uint64_t>(param.myRank) * recvBytes + processedBytes;
    const uint64_t ownSlotOffset = bankBase + static_cast<uint64_t>(param.myRank) * slotStride;
    CHK_RET(CheckRange(ownSlotOffset, chunkBytes, resCtx.localBuffer.size, "local balanced-mesh slot"));
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(controlThread,
        Offset(resCtx.localBuffer.addr, ownSlotOffset), Offset(param.inputPtr, ownInputOffset), chunkBytes)));

    for (uint32_t peerIdx = 0; peerIdx < peerNum; ++peerIdx) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            controlThread, resCtx.threads[workerBase + peerIdx], WORKER_START_NOTIFY)));
    }
    for (uint32_t peerIdx = 0; peerIdx < peerNum; ++peerIdx) {
        ThreadHandle worker = resCtx.threads[workerBase + peerIdx];
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(worker, WORKER_START_NOTIFY, CUSTOM_TIMEOUT)));
        CHK_RET(QueueGlobalMeshWorker(param, resCtx.channels[peerIdx], worker, recvBytes, processedBytes,
            bankBase, slotStride, chunkBytes, channelNotifyBase));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(worker, reduceThread, peerIdx)));
    }
    for (uint32_t peerIdx = 0; peerIdx < peerNum; ++peerIdx) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(reduceThread, peerIdx, CUSTOM_TIMEOUT)));
    }

    CHK_RET(QueueGlobalTreeReduce(
        param, resCtx, reduceThread, workerBase, processedBytes, bankBase, slotStride, chunkBytes));
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyRecordOnThread(reduceThread, controlThread, bankIdx)));
    return HCCL_SUCCESS;
}

HcclResult WaitBank(const AlgResourceCtx &resCtx, uint32_t bankIdx)
{
    return static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        resCtx.threads[CONTROL_THREAD_INDEX], bankIdx, CUSTOM_TIMEOUT));
}

HcclResult ExecSingleGlobalMesh(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t recvBytes)
{
    const uint32_t peerNum = param.rankSize - 1;
    if (resCtx.channels.size() != peerNum || resCtx.threads.size() != peerNum + CONTROL_THREAD_INDEX + 1) {
        HCCL_ERROR("Single FullMesh resource mismatch: channels[%zu], threads[%zu], rankSize[%u]",
            resCtx.channels.size(), resCtx.threads.size(), param.rankSize);
        return HCCL_E_INTERNAL;
    }

    uint64_t slotStride = GetStagingBufferSize(resCtx) / param.rankSize;
    slotStride = slotStride / CCL_SLOT_ALIGNMENT * CCL_SLOT_ALIGNMENT;
    if (recvBytes > slotStride) {
        HCCL_ERROR("Single FullMesh buffer is too small, recvBytes[%llu], slotStride[%llu]",
            static_cast<unsigned long long>(recvBytes), static_cast<unsigned long long>(slotStride));
        return HCCL_E_MEMORY;
    }

    ThreadHandle mainThread = resCtx.threads[CONTROL_THREAD_INDEX];
    const uint64_t ownInputOffset = static_cast<uint64_t>(param.myRank) * recvBytes;
    const uint64_t ownSlotOffset = static_cast<uint64_t>(param.myRank) * slotStride;
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread,
        Offset(resCtx.localBuffer.addr, ownSlotOffset), Offset(param.inputPtr, ownInputOffset), recvBytes)));

    for (uint32_t peerIdx = 0; peerIdx < peerNum; ++peerIdx) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            mainThread, resCtx.threads[peerIdx + 1], WORKER_START_NOTIFY)));
    }
    for (uint32_t peerIdx = 0; peerIdx < peerNum; ++peerIdx) {
        ThreadHandle worker = resCtx.threads[peerIdx + 1];
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(worker, WORKER_START_NOTIFY, CUSTOM_TIMEOUT)));
        CHK_RET(QueueGlobalMeshWorker(
            param, resCtx.channels[peerIdx], worker, recvBytes, 0, 0, slotStride, recvBytes, 0));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(worker, mainThread, peerIdx)));
    }
    for (uint32_t peerIdx = 0; peerIdx < peerNum; ++peerIdx) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(mainThread, peerIdx, CUSTOM_TIMEOUT)));
    }

    return QueueGlobalTreeReduce(param, resCtx, mainThread, 1, 0, 0, slotStride, recvBytes);
}

HcclResult ExecBalancedGlobalMesh(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t recvBytes)
{
    const uint32_t peerNum = param.rankSize - 1;
    const uint32_t threadNum = WORKER_THREAD_BASE + PARALLEL_BANK_NUM * peerNum;
    if (resCtx.channels.size() != peerNum || resCtx.threads.size() != threadNum) {
        HCCL_ERROR("Balanced FullMesh resource mismatch: channels[%zu], threads[%zu], rankSize[%u]",
            resCtx.channels.size(), resCtx.threads.size(), param.rankSize);
        return HCCL_E_INTERNAL;
    }

    const uint64_t stagingBufferSize = GetStagingBufferSize(resCtx);
    uint64_t slotStride = stagingBufferSize / (PARALLEL_BANK_NUM * static_cast<uint64_t>(param.rankSize));
    slotStride = slotStride / CCL_SLOT_ALIGNMENT * CCL_SLOT_ALIGNMENT;
    if (slotStride < sizeof(float)) {
        HCCL_ERROR("CCL buffer is too small for balanced FullMesh, size[%llu], rankSize[%u]",
            static_cast<unsigned long long>(stagingBufferSize), param.rankSize);
        return HCCL_E_MEMORY;
    }
    const uint64_t bankStride = static_cast<uint64_t>(param.rankSize) * slotStride;
    CHK_RET(CheckRange(bankStride, bankStride, stagingBufferSize, "balanced FullMesh banks"));

    const uint64_t firstHalfBytes = param.count / PARALLEL_BANK_NUM * sizeof(float);
    const uint64_t secondHalfBytes = recvBytes - firstHalfBytes;
    uint64_t processedHalfBytes = 0;
    while (processedHalfBytes < firstHalfBytes) {
        const uint64_t chunkBytes = std::min(slotStride, firstHalfBytes - processedHalfBytes);
        CHK_RET(QueueBank(
            param, resCtx, 0, recvBytes, processedHalfBytes, slotStride, bankStride, chunkBytes));
        CHK_RET(QueueBank(param, resCtx, 1, recvBytes, firstHalfBytes + processedHalfBytes, slotStride,
            bankStride, chunkBytes));
        CHK_RET(WaitBank(resCtx, 0));
        CHK_RET(WaitBank(resCtx, 1));
        processedHalfBytes += chunkBytes;
    }

    if (secondHalfBytes > firstHalfBytes) {
        const uint64_t tailBytes = secondHalfBytes - firstHalfBytes;
        CHK_RET(QueueBank(param, resCtx, 1, recvBytes, firstHalfBytes + processedHalfBytes, slotStride,
            bankStride, tailBytes));
        CHK_RET(WaitBank(resCtx, 1));
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    if (param.dataType != HCCL_DATA_TYPE_FP32 || param.reduceType != HCCL_REDUCE_SUM) {
        HCCL_ERROR("Only FP32 SUM is supported");
        return HCCL_E_NOT_SUPPORT;
    }
    if (param.rankSize == 0 || param.myRank >= param.rankSize || resCtx.threads.empty()) {
        HCCL_ERROR("Invalid rank or thread resource");
        return HCCL_E_PARA;
    }
    if (param.count > std::numeric_limits<uint64_t>::max() / sizeof(float)) {
        HCCL_ERROR("Receive count overflow");
        return HCCL_E_PARA;
    }

    const uint64_t recvBytes = param.count * sizeof(float);
    if (recvBytes == 0) {
        return HCCL_SUCCESS;
    }
    if (param.rankSize == 1) {
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(resCtx.threads[0], param.outputPtr, param.inputPtr, recvBytes)));
        return HCCL_SUCCESS;
    }
    if (recvBytes > std::numeric_limits<uint64_t>::max() / param.rankSize) {
        HCCL_ERROR("Invalid global FullMesh ReduceScatter shape, rankSize[%u], recvBytes[%llu]", param.rankSize,
            static_cast<unsigned long long>(recvBytes));
        return HCCL_E_PARA;
    }

    AlgorithmMode mode = AlgorithmMode::SINGLE_GLOBAL_TREE;
    CHK_RET(SelectAlgorithmMode(param, recvBytes, mode));
    if (mode == AlgorithmMode::PAPER2_SMALL) {
        return ExecPaper2Small(param, resCtx, recvBytes);
    }
    if (mode == AlgorithmMode::SINGLE_GLOBAL_TREE) {
        return ExecSingleGlobalMesh(param, resCtx, recvBytes);
    }
    return ExecBalancedGlobalMesh(param, resCtx, recvBytes);
}
} // namespace ops_hccl
