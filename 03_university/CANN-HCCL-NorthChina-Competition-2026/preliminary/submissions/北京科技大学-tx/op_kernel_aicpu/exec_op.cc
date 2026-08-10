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
#include <limits>
#include <vector>

namespace ops_hccl {
namespace {
constexpr uint32_t EXPECTED_RANK_SIZE = 16;
constexpr uint32_t RANKS_PER_SERVER = 8;
constexpr uint32_t THREAD_NUM = 16;
constexpr uint32_t PEER_NUM = 15;
constexpr uint32_t CHANNEL_NUM = 15;

constexpr uint64_t MIN_SLICE_ALIGN = 128;
constexpr uint64_t MAX_TRANSFER_BYTES =
    256ULL * 1024 * 1024;
constexpr uint64_t SMALL_BLOCK_LIMIT_BYTES =
    512ULL * 1024;
constexpr uint64_t PARALLEL_STAGE_MIN_BYTES =
    1ULL * 1024 * 1024;

// thread0.notify[0] is reserved by the fixed Host/Device gate.
// Every worker has exactly one START producer and at most two DONE producers.
constexpr uint32_t THREAD_NOTIFY_START = 0;
constexpr uint32_t THREAD_NOTIFY_DONE_LEFT = 1;
constexpr uint32_t THREAD_NOTIFY_DONE_RIGHT = 2;
constexpr uint32_t MAIN_NOTIFY_ROOT_DONE =
    THREAD_NOTIFY_DONE_LEFT;

// The small-message path chains its four fixed Channel owners through a
// dedicated Thread Notify. Keeping this separate from the large-path tree
// preserves one producer/one consumer semantics across repeated calls.
constexpr uint32_t THREAD_NOTIFY_SMALL_CHAIN = 3;

// The large-message path uses t0 and t1 as two LocalCopy lanes.  t0 is the
// only producer of t1.notify[4].  The join is consumed before t1 releases the
// existing START tree, so no Channel owner can observe a partially staged
// source.
constexpr uint32_t THREAD_NOTIFY_STAGE_JOIN = 4;

// Each Channel uses one stable meaning per notification index.
constexpr uint32_t CHANNEL_NOTIFY_READY = NOTIFY_IDX_ACK;
constexpr uint32_t CHANNEL_NOTIFY_DONE =
    NOTIFY_IDX_DATA_SIGNAL;

uint64_t AlignDown(uint64_t value, uint64_t alignment)
{
    return value / alignment * alignment;
}

uint32_t ExpectedRemoteRank(
    uint32_t myRank,
    uint32_t channelIdx)
{
    if (myRank >= EXPECTED_RANK_SIZE ||
        channelIdx >= CHANNEL_NUM) {
        return INVALID_VALUE_RANKID;
    }
    return channelIdx < myRank
        ? channelIdx
        : channelIdx + 1U;
}

uint32_t ChannelIndexForRemoteRank(
    uint32_t myRank,
    uint32_t remoteRank)
{
    if (myRank >= EXPECTED_RANK_SIZE ||
        remoteRank >= EXPECTED_RANK_SIZE ||
        remoteRank == myRank) {
        return CHANNEL_NUM;
    }
    return remoteRank < myRank
        ? remoteRank
        : remoteRank - 1U;
}

HcclResult ThreadRecord(
    ThreadHandle src,
    ThreadHandle dst,
    uint32_t notifyIdx)
{
    return static_cast<HcclResult>(
        HcommThreadNotifyRecordOnThread(
            src,
            dst,
            notifyIdx));
}

HcclResult ThreadWait(
    ThreadHandle thread,
    uint32_t notifyIdx)
{
    return static_cast<HcclResult>(
        HcommThreadNotifyWaitOnThread(
            thread,
            notifyIdx,
            CUSTOM_TIMEOUT));
}

HcclResult ChannelRecord(
    ThreadHandle owner,
    const ChannelInfo &channel,
    uint32_t notifyIdx)
{
    return static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(
            owner,
            channel.handle,
            notifyIdx));
}

HcclResult ChannelWait(
    ThreadHandle owner,
    const ChannelInfo &channel,
    uint32_t notifyIdx)
{
    return static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(
            owner,
            channel.handle,
            notifyIdx,
            CUSTOM_TIMEOUT));
}

HcclResult ChannelRead(
    ThreadHandle owner,
    const ChannelInfo &channel,
    void *dst,
    const void *src,
    uint64_t bytes)
{
    CHK_PRT_RET(bytes == 0 || bytes > MAX_TRANSFER_BYTES,
        HCCL_ERROR(
            "AllGather invalid Read bytes[%llu]",
            static_cast<unsigned long long>(bytes)),
        HCCL_E_PARA);
    return static_cast<HcclResult>(
        HcommReadOnThread(
            owner,
            channel.handle,
            dst,
            src,
            bytes));
}

HcclResult LocalCopy(
    ThreadHandle thread,
    void *dst,
    const void *src,
    uint64_t bytes)
{
    CHK_PRT_RET(bytes == 0 || bytes > MAX_TRANSFER_BYTES,
        HCCL_ERROR(
            "AllGather invalid LocalCopy bytes[%llu]",
            static_cast<unsigned long long>(bytes)),
        HCCL_E_PARA);
    return static_cast<HcclResult>(
        HcommLocalCopyOnThread(
            thread,
            dst,
            src,
            bytes));
}

HcclResult CheckResources(
    const OpParam &param,
    const AlgResourceCtx &resCtx)
{
    CHK_PRT_RET(
        param.rankSize != EXPECTED_RANK_SIZE ||
            param.myRank >= param.rankSize,
        HCCL_ERROR(
            "AllGather invalid rank[%u], rankSize[%u]",
            param.myRank,
            param.rankSize),
        HCCL_E_PARA);
    CHK_PRT_RET(resCtx.threads.size() != THREAD_NUM,
        HCCL_ERROR(
            "AllGather thread count[%zu], expected[%u]",
            resCtx.threads.size(),
            THREAD_NUM),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.aicpuThread != resCtx.threads[0],
        HCCL_ERROR("AllGather controller Thread mismatch"),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.channels.size() != CHANNEL_NUM,
        HCCL_ERROR(
            "AllGather Channel count[%zu], expected[%u]",
            resCtx.channels.size(),
            CHANNEL_NUM),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(
        resCtx.localBuffer.addr == nullptr ||
            resCtx.localBuffer.size == 0,
        HCCL_ERROR("AllGather invalid local CCL buffer"),
        HCCL_E_INTERNAL);

    for (uint32_t idx = 0; idx < CHANNEL_NUM; ++idx) {
        const ChannelInfo &channel = resCtx.channels[idx];
        const uint32_t expectedRank =
            ExpectedRemoteRank(param.myRank, idx);
        CHK_PRT_RET(
            channel.remoteRank != expectedRank ||
                channel.remoteRank >= param.rankSize ||
                channel.remoteRank == param.myRank,
            HCCL_ERROR(
                "AllGather Channel[%u] remoteRank[%u], expected[%u]",
                idx,
                channel.remoteRank,
                expectedRank),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(
            channel.notifyNum < 2 ||
                channel.remoteCclMem.addr == nullptr ||
                channel.remoteCclMem.size == 0,
            HCCL_ERROR(
                "AllGather invalid Channel[%u] resources",
                idx),
            HCCL_E_INTERNAL);
    }

    return HCCL_SUCCESS;
}

uint64_t GetCommonCclBufferSize(
    const AlgResourceCtx &resCtx)
{
    uint64_t commonSize = resCtx.localBuffer.size;
    for (const ChannelInfo &channel : resCtx.channels) {
        commonSize =
            std::min(commonSize, channel.remoteCclMem.size);
    }
    return commonSize;
}

uint64_t SelectSliceCapacity(
    const AlgResourceCtx &resCtx,
    uint32_t dataTypeSize)
{
    uint64_t capacity = std::min(
        GetCommonCclBufferSize(resCtx),
        MAX_TRANSFER_BYTES);
    capacity = AlignDown(capacity, MIN_SLICE_ALIGN);
    capacity = AlignDown(capacity, dataTypeSize);
    return capacity;
}

/*
 * Stage one source slice with two LocalCopy lanes and then release the
 * existing worker tree:
 *
 *   t0: Record START(t1) -> copy first half -> Record JOIN(t1)
 *   t1: Wait START       -> copy second half -> Wait JOIN
 *       -> release t2,t3
 *
 * The two copies target disjoint ranges.  t1 cannot release any reader until
 * both ranges are complete.  The remaining START tree is unchanged:
 * t1 -> t2,t3; t2 -> t4,t5; ...; t7 -> t14,t15.
 */
HcclResult EnqueueParallelStageAndStartTree(
    const std::vector<ThreadHandle> &threads,
    void *localSource,
    const void *input,
    uint64_t sliceBytes,
    uint32_t dataTypeSize)
{
    CHK_PRT_RET(
        threads.size() != THREAD_NUM ||
            localSource == nullptr ||
            input == nullptr ||
            dataTypeSize == 0,
        HCCL_ERROR(
            "AllGather invalid parallel-stage resources, threads[%zu] "
            "sliceBytes[%llu] typeSize[%u]",
            threads.size(),
            static_cast<unsigned long long>(sliceBytes),
            dataTypeSize),
        HCCL_E_INTERNAL);

    uint64_t firstBytes =
        AlignDown(sliceBytes / 2, MIN_SLICE_ALIGN);
    firstBytes = AlignDown(firstBytes, dataTypeSize);
    const uint64_t secondBytes = sliceBytes - firstBytes;
    CHK_PRT_RET(
        firstBytes == 0 ||
            secondBytes == 0 ||
            firstBytes > MAX_TRANSFER_BYTES ||
            secondBytes > MAX_TRANSFER_BYTES,
        HCCL_ERROR(
            "AllGather invalid parallel-stage split, slice[%llu] "
            "first[%llu] second[%llu]",
            static_cast<unsigned long long>(sliceBytes),
            static_cast<unsigned long long>(firstBytes),
            static_cast<unsigned long long>(secondBytes)),
        HCCL_E_INTERNAL);

    const ThreadHandle mainThread = threads[0];
    const ThreadHandle rootWorker = threads[1];
    CHK_RET(ThreadRecord(
        mainThread,
        rootWorker,
        THREAD_NOTIFY_START));
    CHK_RET(LocalCopy(
        mainThread,
        localSource,
        input,
        firstBytes));
    CHK_RET(ThreadWait(
        rootWorker,
        THREAD_NOTIFY_START));
    CHK_RET(LocalCopy(
        rootWorker,
        static_cast<uint8_t *>(localSource) + firstBytes,
        static_cast<const uint8_t *>(input) + firstBytes,
        secondBytes));
    CHK_RET(ThreadRecord(
        mainThread,
        rootWorker,
        THREAD_NOTIFY_STAGE_JOIN));
    CHK_RET(ThreadWait(
        rootWorker,
        THREAD_NOTIFY_STAGE_JOIN));

    for (uint32_t worker = 1; worker <= PEER_NUM; ++worker) {
        if (worker != 1U) {
            CHK_RET(ThreadWait(
                threads[worker],
                THREAD_NOTIFY_START));
        }

        const uint32_t leftChild = worker * 2;
        const uint32_t rightChild = leftChild + 1;
        if (leftChild <= PEER_NUM) {
            CHK_RET(ThreadRecord(
                threads[worker],
                threads[leftChild],
                THREAD_NOTIFY_START));
        }
        if (rightChild <= PEER_NUM) {
            CHK_RET(ThreadRecord(
                threads[worker],
                threads[rightChild],
                THREAD_NOTIFY_START));
        }
    }

    return HCCL_SUCCESS;
}

// Unchanged v11 START tree for non-performance residual slices.
HcclResult EnqueueStartTree(
    const std::vector<ThreadHandle> &threads)
{
    CHK_RET(ThreadRecord(
        threads[0],
        threads[1],
        THREAD_NOTIFY_START));

    for (uint32_t worker = 1; worker <= PEER_NUM; ++worker) {
        CHK_RET(ThreadWait(
            threads[worker],
            THREAD_NOTIFY_START));

        const uint32_t leftChild = worker * 2;
        const uint32_t rightChild = leftChild + 1;
        if (leftChild <= PEER_NUM) {
            CHK_RET(ThreadRecord(
                threads[worker],
                threads[leftChild],
                THREAD_NOTIFY_START));
        }
        if (rightChild <= PEER_NUM) {
            CHK_RET(ThreadRecord(
                threads[worker],
                threads[rightChild],
                THREAD_NOTIFY_START));
        }
    }

    return HCCL_SUCCESS;
}

// Every parent Wait is appended after that parent's Channel DONE Wait.
// Every child Record is appended after that child's Channel DONE Wait.
HcclResult EnqueueCompletionTree(
    const std::vector<ThreadHandle> &threads)
{
    for (uint32_t parent = 1; parent <= PEER_NUM; ++parent) {
        const uint32_t leftChild = parent * 2;
        const uint32_t rightChild = leftChild + 1;
        if (leftChild <= PEER_NUM) {
            CHK_RET(ThreadWait(
                threads[parent],
                THREAD_NOTIFY_DONE_LEFT));
        }
        if (rightChild <= PEER_NUM) {
            CHK_RET(ThreadWait(
                threads[parent],
                THREAD_NOTIFY_DONE_RIGHT));
        }
    }

    for (uint32_t child = 2; child <= PEER_NUM; ++child) {
        const uint32_t parent = child / 2;
        const uint32_t parentNotify =
            (child == parent * 2)
            ? THREAD_NOTIFY_DONE_LEFT
            : THREAD_NOTIFY_DONE_RIGHT;
        CHK_RET(ThreadRecord(
            threads[child],
            threads[parent],
            parentNotify));
    }

    CHK_RET(ThreadRecord(
        threads[1],
        threads[0],
        MAIN_NOTIFY_ROOT_DONE));
    return HCCL_SUCCESS;
}

HcclResult EnqueuePeerReads(
    const OpParam &param,
    const AlgResourceCtx &resCtx,
    uint64_t blockBytes,
    uint64_t sliceOffset,
    uint64_t sliceBytes)
{
    for (uint32_t idx = 0; idx < PEER_NUM; ++idx) {
        const ChannelInfo &channel = resCtx.channels[idx];
        const ThreadHandle owner = resCtx.threads[idx + 1];

        CHK_RET(ChannelRecord(
            owner,
            channel,
            CHANNEL_NOTIFY_READY));
        CHK_RET(ChannelWait(
            owner,
            channel,
            CHANNEL_NOTIFY_READY));

        uint8_t *peerOutput =
            static_cast<uint8_t *>(param.outputPtr) +
            static_cast<uint64_t>(channel.remoteRank) * blockBytes +
            sliceOffset;
        CHK_RET(ChannelRead(
            owner,
            channel,
            peerOutput,
            channel.remoteCclMem.addr,
            sliceBytes));

        CHK_RET(ChannelRecord(
            owner,
            channel,
            CHANNEL_NOTIFY_DONE));
        CHK_RET(ChannelWait(
            owner,
            channel,
            CHANNEL_NOTIFY_DONE));
    }

    return HCCL_SUCCESS;
}

/*
 * Latency path for B <= 512 KiB.
 *
 * The first three XOR rounds gather the eight ranks of one Server into the
 * local CCL buffer in local-rank order. The fourth (xor 8) Read writes the
 * remote Server half directly to its final output range; only the local half
 * needs a final LocalCopy.
 *
 * A Channel keeps the same owner as the large Direct-Mesh path:
 * channels[i] is always operated by threads[i + 1]. A dedicated Thread
 * Notify forms the dependency chain
 *
 *   t0 -> owner(xor 1) -> owner(xor 2) -> owner(xor 4)
 *      -> owner(xor 8) -> t0.
 *
 * READY publishes the source produced by the preceding round. DONE proves
 * both endpoints have finished reading before either advances or reuses a
 * Channel notification in the next invocation.
 */
HcclResult ExecSmallRecursiveDoubling(
    const OpParam &param,
    const AlgResourceCtx &resCtx,
    uint64_t blockBytes)
{
    CHK_PRT_RET(
        blockBytes == 0 ||
            blockBytes > SMALL_BLOCK_LIMIT_BYTES,
        HCCL_ERROR(
            "AllGather invalid small block bytes[%llu]",
            static_cast<unsigned long long>(blockBytes)),
        HCCL_E_PARA);

    const uint64_t serverBytes =
        static_cast<uint64_t>(RANKS_PER_SERVER) * blockBytes;
    CHK_PRT_RET(
        serverBytes == 0 ||
            serverBytes > MAX_TRANSFER_BYTES ||
            serverBytes > resCtx.localBuffer.size,
        HCCL_ERROR(
            "AllGather small Server bytes[%llu] exceed local CCL[%llu]",
            static_cast<unsigned long long>(serverBytes),
            static_cast<unsigned long long>(
                resCtx.localBuffer.size)),
        HCCL_E_INTERNAL);

    const ThreadHandle mainThread = resCtx.threads[0];
    uint8_t *localCcl =
        static_cast<uint8_t *>(resCtx.localBuffer.addr);
    const uint32_t localRank =
        param.myRank % RANKS_PER_SERVER;
    const uint32_t localServerBase =
        param.myRank / RANKS_PER_SERVER * RANKS_PER_SERVER;
    const uint32_t remoteServerBase =
        localServerBase ^ RANKS_PER_SERVER;

    CHK_RET(LocalCopy(
        mainThread,
        localCcl +
            static_cast<uint64_t>(localRank) * blockBytes,
        param.inputPtr,
        blockBytes));

    ThreadHandle predecessor = mainThread;
    for (uint32_t width = 1;
         width < RANKS_PER_SERVER;
         width <<= 1U) {
        const uint32_t remoteRank = param.myRank ^ width;
        const uint32_t channelIdx =
            ChannelIndexForRemoteRank(
                param.myRank,
                remoteRank);
        CHK_PRT_RET(channelIdx >= CHANNEL_NUM,
            HCCL_ERROR(
                "AllGather invalid small Channel rank[%u] remote[%u]",
                param.myRank,
                remoteRank),
            HCCL_E_INTERNAL);

        const ChannelInfo &channel =
            resCtx.channels[channelIdx];
        const ThreadHandle owner =
            resCtx.threads[channelIdx + 1U];
        const uint32_t remoteLocalStart =
            (localRank ^ width) & ~(width - 1U);
        const uint64_t remoteOffset =
            static_cast<uint64_t>(remoteLocalStart) *
            blockBytes;
        const uint64_t roundBytes =
            static_cast<uint64_t>(width) * blockBytes;

        CHK_PRT_RET(
            remoteOffset > channel.remoteCclMem.size ||
                roundBytes >
                    channel.remoteCclMem.size - remoteOffset,
            HCCL_ERROR(
                "AllGather small round exceeds remote CCL, rank[%u] "
                "remote[%u] offset[%llu] bytes[%llu] ccl[%llu]",
                param.myRank,
                remoteRank,
                static_cast<unsigned long long>(remoteOffset),
                static_cast<unsigned long long>(roundBytes),
                static_cast<unsigned long long>(
                    channel.remoteCclMem.size)),
            HCCL_E_INTERNAL);

        CHK_RET(ThreadRecord(
            predecessor,
            owner,
            THREAD_NOTIFY_SMALL_CHAIN));
        CHK_RET(ThreadWait(
            owner,
            THREAD_NOTIFY_SMALL_CHAIN));
        CHK_RET(ChannelRecord(
            owner,
            channel,
            CHANNEL_NOTIFY_READY));
        CHK_RET(ChannelWait(
            owner,
            channel,
            CHANNEL_NOTIFY_READY));
        CHK_RET(ChannelRead(
            owner,
            channel,
            localCcl + remoteOffset,
            static_cast<const uint8_t *>(
                channel.remoteCclMem.addr) +
                remoteOffset,
            roundBytes));
        CHK_RET(ChannelRecord(
            owner,
            channel,
            CHANNEL_NOTIFY_DONE));
        CHK_RET(ChannelWait(
            owner,
            channel,
            CHANNEL_NOTIFY_DONE));
        predecessor = owner;
    }

    const uint32_t pairedRank =
        param.myRank ^ RANKS_PER_SERVER;
    const uint32_t pairedChannelIdx =
        ChannelIndexForRemoteRank(
            param.myRank,
            pairedRank);
    CHK_PRT_RET(pairedChannelIdx >= CHANNEL_NUM,
        HCCL_ERROR(
            "AllGather invalid paired Channel rank[%u] remote[%u]",
            param.myRank,
            pairedRank),
        HCCL_E_INTERNAL);

    const ChannelInfo &pairedChannel =
        resCtx.channels[pairedChannelIdx];
    const ThreadHandle pairedOwner =
        resCtx.threads[pairedChannelIdx + 1U];
    CHK_PRT_RET(
        serverBytes > pairedChannel.remoteCclMem.size,
        HCCL_ERROR(
            "AllGather paired CCL[%llu] smaller than Server bytes[%llu]",
            static_cast<unsigned long long>(
                pairedChannel.remoteCclMem.size),
            static_cast<unsigned long long>(serverBytes)),
        HCCL_E_INTERNAL);

    CHK_RET(ThreadRecord(
        predecessor,
        pairedOwner,
        THREAD_NOTIFY_SMALL_CHAIN));
    CHK_RET(ThreadWait(
        pairedOwner,
        THREAD_NOTIFY_SMALL_CHAIN));
    CHK_RET(ChannelRecord(
        pairedOwner,
        pairedChannel,
        CHANNEL_NOTIFY_READY));
    CHK_RET(ChannelWait(
        pairedOwner,
        pairedChannel,
        CHANNEL_NOTIFY_READY));

    uint8_t *remoteServerOutput =
        static_cast<uint8_t *>(param.outputPtr) +
        static_cast<uint64_t>(remoteServerBase) *
            blockBytes;
    CHK_RET(ChannelRead(
        pairedOwner,
        pairedChannel,
        remoteServerOutput,
        pairedChannel.remoteCclMem.addr,
        serverBytes));
    CHK_RET(ChannelRecord(
        pairedOwner,
        pairedChannel,
        CHANNEL_NOTIFY_DONE));
    CHK_RET(ChannelWait(
        pairedOwner,
        pairedChannel,
        CHANNEL_NOTIFY_DONE));

    CHK_RET(ThreadRecord(
        pairedOwner,
        mainThread,
        THREAD_NOTIFY_SMALL_CHAIN));
    CHK_RET(ThreadWait(
        mainThread,
        THREAD_NOTIFY_SMALL_CHAIN));

    uint8_t *localServerOutput =
        static_cast<uint8_t *>(param.outputPtr) +
        static_cast<uint64_t>(localServerBase) *
            blockBytes;
    CHK_RET(LocalCopy(
        mainThread,
        localServerOutput,
        localCcl,
        serverBytes));
    return HCCL_SUCCESS;
}

HcclResult ExecDirectRead(
    const OpParam &param,
    const AlgResourceCtx &resCtx,
    uint32_t dataTypeSize,
    uint64_t blockBytes)
{
    const uint64_t sliceCapacity =
        SelectSliceCapacity(resCtx, dataTypeSize);
    CHK_PRT_RET(sliceCapacity < dataTypeSize,
        HCCL_ERROR(
            "AllGather CCL buffer too small, commonBytes[%llu] typeSize[%u]",
            static_cast<unsigned long long>(
                GetCommonCclBufferSize(resCtx)),
            dataTypeSize),
        HCCL_E_INTERNAL);

    const ThreadHandle mainThread = resCtx.threads[0];
    uint8_t *localSource =
        static_cast<uint8_t *>(resCtx.localBuffer.addr);
    uint64_t processedBytes = 0;

    while (processedBytes < blockBytes) {
        const uint64_t sliceBytes = std::min(
            sliceCapacity,
            blockBytes - processedBytes);
        const uint8_t *input =
            static_cast<const uint8_t *>(param.inputPtr) +
            processedBytes;

        if (sliceBytes >= PARALLEL_STAGE_MIN_BYTES) {
            // t0 and t1 stage disjoint halves concurrently.  t1 releases the
            // START tree only after its JOIN Wait proves both halves complete.
            CHK_RET(EnqueueParallelStageAndStartTree(
                resCtx.threads,
                localSource,
                input,
                sliceBytes,
                dataTypeSize));
        } else {
            CHK_RET(LocalCopy(
                mainThread,
                localSource,
                input,
                sliceBytes));
            CHK_RET(EnqueueStartTree(resCtx.threads));
        }

        // The self copy overlaps all peer Reads after the START release.
        uint8_t *selfOutput =
            static_cast<uint8_t *>(param.outputPtr) +
            static_cast<uint64_t>(param.myRank) * blockBytes +
            processedBytes;
        CHK_RET(LocalCopy(
            mainThread,
            selfOutput,
            input,
            sliceBytes));

        CHK_RET(EnqueuePeerReads(
            param,
            resCtx,
            blockBytes,
            processedBytes,
            sliceBytes));
        CHK_RET(EnqueueCompletionTree(resCtx.threads));

        // This Wait closes all 15 Channel DONE handshakes. Only then may
        // the next slice overwrite localSource or reuse any Notify index.
        CHK_RET(ThreadWait(
            mainThread,
            MAIN_NOTIFY_ROOT_DONE));
        processedBytes += sliceBytes;
    }

    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(
    const OpParam &param,
    const AlgResourceCtx &resCtx)
{
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }

    const auto typeIt = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(typeIt == SIZE_TABLE.end(),
        HCCL_ERROR(
            "Unsupported AllGather data type[%d]",
            static_cast<int>(param.dataType)),
        HCCL_E_PARA);
    const uint32_t dataTypeSize = typeIt->second;
    CHK_PRT_RET(
        dataTypeSize != static_cast<uint32_t>(sizeof(float)),
        HCCL_ERROR(
            "AllGather requires FP32 size[%zu], actual[%u]",
            sizeof(float),
            dataTypeSize),
        HCCL_E_PARA);

    CHK_PRT_RET(
        param.count >
            std::numeric_limits<uint64_t>::max() / dataTypeSize,
        HCCL_ERROR(
            "AllGather byte-size overflow, count[%llu] typeSize[%u]",
            static_cast<unsigned long long>(param.count),
            dataTypeSize),
        HCCL_E_PARA);
    const uint64_t blockBytes = param.count * dataTypeSize;

    CHK_PRT_RET(
        param.rankSize == 0 ||
            blockBytes >
                std::numeric_limits<uint64_t>::max() /
                    param.rankSize,
        HCCL_ERROR(
            "AllGather output-size overflow, blockBytes[%llu] rankSize[%u]",
            static_cast<unsigned long long>(blockBytes),
            param.rankSize),
        HCCL_E_PARA);

    CHK_RET(CheckResources(param, resCtx));
    if (blockBytes <= SMALL_BLOCK_LIMIT_BYTES) {
        return ExecSmallRecursiveDoubling(
            param,
            resCtx,
            blockBytes);
    }
    return ExecDirectRead(
        param,
        resCtx,
        dataTypeSize,
        blockBytes);
}
} // namespace ops_hccl
