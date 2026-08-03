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
constexpr uint32_t SLAVE_THREAD_START_NOTIFY_INDEX = 0;
constexpr uint32_t MAIN_THREAD_BARRIER_NOTIFY_BASE = 1;
constexpr uint32_t MAX_ALLGATHER_RANK_SIZE = 16;
constexpr uint32_t FULL_BUFFER_EXIT_BARRIER_NOTIFY_INDEX = NOTIFY_IDX_ACK;
constexpr uint32_t BUFFER_BANK_MASK = 1;
constexpr uint32_t PREVIOUS_FULL_BUFFER_FLAG = 2;
constexpr uint64_t SMALL_MESSAGE_OUTPUT_THRESHOLD_BYTES = 1ULL << 20;
constexpr uint64_t LARGE_READ_PULL_OUTPUT_THRESHOLD_BYTES = 64ULL << 20;
constexpr uint64_t BINARY_FANIN_OUTPUT_THRESHOLD_BYTES = 64ULL << 20;
constexpr uint64_t MAX_READ_PULL_CHUNK_BYTES = 256ULL * 1024 * 1024;
constexpr uint64_t PACKED_RMA_ALIGNMENT_BYTES = 128;
constexpr uint32_t BINARY_FANIN_HALF_COUNT = 2;
constexpr uint32_t BINARY_FANIN_HALF_RANKS = 8;

HcclResult GetDataTypeSize(HcclDataType dataType, uint64_t &dataTypeSize)
{
    switch (dataType) {
        case HCCL_DATA_TYPE_INT8:
        case HCCL_DATA_TYPE_UINT8:
        case HCCL_DATA_TYPE_HIF8:
        case HCCL_DATA_TYPE_FP8E4M3:
        case HCCL_DATA_TYPE_FP8E5M2:
        case HCCL_DATA_TYPE_FP8E8M0:
            dataTypeSize = 1;
            return HCCL_SUCCESS;
        case HCCL_DATA_TYPE_INT16:
        case HCCL_DATA_TYPE_FP16:
        case HCCL_DATA_TYPE_UINT16:
        case HCCL_DATA_TYPE_BFP16:
            dataTypeSize = 2;
            return HCCL_SUCCESS;
        case HCCL_DATA_TYPE_INT32:
        case HCCL_DATA_TYPE_FP32:
        case HCCL_DATA_TYPE_UINT32:
            dataTypeSize = 4;
            return HCCL_SUCCESS;
        case HCCL_DATA_TYPE_INT64:
        case HCCL_DATA_TYPE_UINT64:
        case HCCL_DATA_TYPE_FP64:
            dataTypeSize = 8;
            return HCCL_SUCCESS;
        case HCCL_DATA_TYPE_INT128:
            dataTypeSize = 16;
            return HCCL_SUCCESS;
        default:
            HCCL_ERROR("Unsupported AllGather data type: %d", static_cast<int32_t>(dataType));
            return HCCL_E_PARA;
    }
}

bool CheckedMultiply(uint64_t lhs, uint64_t rhs, uint64_t &product)
{
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return false;
    }
    product = lhs * rhs;
    return true;
}

uint64_t PackedRankAlignmentOffset(uint32_t rank, uint64_t totalBytes)
{
    if (totalBytes % PACKED_RMA_ALIGNMENT_BYTES == 0) {
        return 0;
    }
    return ((totalBytes % PACKED_RMA_ALIGNMENT_BYTES) * rank) %
        PACKED_RMA_ALIGNMENT_BYTES;
}

bool CheckedAddressOffset(const void *base, uint64_t offset, const void *&address)
{
    const uintptr_t baseAddress = reinterpret_cast<uintptr_t>(base);
    if (offset > std::numeric_limits<uintptr_t>::max() - baseAddress) {
        return false;
    }
    address = reinterpret_cast<const void *>(baseAddress + static_cast<uintptr_t>(offset));
    return true;
}

bool CheckedAddressOffset(void *base, uint64_t offset, void *&address)
{
    const void *constAddress = nullptr;
    if (!CheckedAddressOffset(static_cast<const void *>(base), offset, constAddress)) {
        return false;
    }
    address = const_cast<void *>(constAddress);
    return true;
}

HcclResult QueueBinaryFanInHalfCopy(ThreadHandle leaderThread, const CommBuffer &localWorkingBuffer,
    void *outputPtr, uint32_t binaryFanInHalf, uint64_t binaryFanInHalfBytes)
{
    uint64_t binaryFanInHalfOffset = 0;
    CHK_PRT_RET(!CheckedMultiply(binaryFanInHalf, binaryFanInHalfBytes, binaryFanInHalfOffset),
        HCCL_ERROR("AllGather binary fan-in half offset overflows"), HCCL_E_PARA);
    const void *halfSource = nullptr;
    void *halfOutput = nullptr;
    CHK_PRT_RET(!CheckedAddressOffset(localWorkingBuffer.addr, binaryFanInHalfOffset, halfSource) ||
            !CheckedAddressOffset(outputPtr, binaryFanInHalfOffset, halfOutput),
        HCCL_ERROR("AllGather binary fan-in half address overflows"), HCCL_E_PARA);
    CHK_RET(HcommLocalCopyOnThread(
        leaderThread, halfOutput, halfSource, binaryFanInHalfBytes));
    return HCCL_SUCCESS;
}

HcclResult RunDisseminationBarrier(const OpParam &param, const AlgResourceCtx &resCtx,
    const uint32_t channelIndexByRemoteRank[MAX_ALLGATHER_RANK_SIZE], uint32_t notifyIndex)
{
    const ThreadHandle thread = resCtx.threads[0];
    for (uint32_t distance = 1; distance < param.rankSize; distance <<= 1) {
        const uint32_t sendRank = (param.myRank + distance) % param.rankSize;
        const uint32_t receiveRank =
            (param.myRank + param.rankSize - distance) % param.rankSize;
        const uint32_t sendChannelIndex = channelIndexByRemoteRank[sendRank];
        const uint32_t receiveChannelIndex = channelIndexByRemoteRank[receiveRank];
        CHK_PRT_RET(sendChannelIndex >= resCtx.channels.size() ||
                receiveChannelIndex >= resCtx.channels.size(),
            HCCL_ERROR("AllGather full-buffer barrier channel is missing at distance %u", distance),
            HCCL_E_NOT_FOUND);
        const ChannelInfo &sendChannel = resCtx.channels[sendChannelIndex];
        const ChannelInfo &receiveChannel = resCtx.channels[receiveChannelIndex];
        CHK_RET(HcommChannelNotifyRecordOnThread(
            thread, sendChannel.handle, notifyIndex));
        CHK_RET(HcommChannelNotifyWaitOnThread(
            thread, receiveChannel.handle, notifyIndex, CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
}

HcclResult RunFullBufferExitBarrier(const OpParam &param, const AlgResourceCtx &resCtx,
    const uint32_t channelIndexByRemoteRank[MAX_ALLGATHER_RANK_SIZE])
{
    return RunDisseminationBarrier(
        param, resCtx, channelIndexByRemoteRank, FULL_BUFFER_EXIT_BARRIER_NOTIFY_INDEX);
}

HcclResult RunRecursiveDoublingSmallMessage(
    const OpParam &param, const AlgResourceCtx &resCtx,
    const uint32_t channelIndexByRemoteRank[MAX_ALLGATHER_RANK_SIZE], const CommBuffer &localBank,
    uint64_t bankOffset, uint32_t dataNotifyIndex, uint64_t totalBytes, uint64_t outputBytes)
{
    const ThreadHandle thread = resCtx.threads[0];
    uint64_t localRankOffset = 0;
    CHK_PRT_RET(!CheckedMultiply(param.myRank, totalBytes, localRankOffset),
        HCCL_ERROR("AllGather recursive-doubling local rank offset overflows"), HCCL_E_PARA);
    void *localRankSlot = nullptr;
    CHK_PRT_RET(!CheckedAddressOffset(localBank.addr, localRankOffset, localRankSlot),
        HCCL_ERROR("AllGather recursive-doubling local rank address overflows"), HCCL_E_PARA);
    CHK_RET(HcommLocalCopyOnThread(thread, localRankSlot, param.inputPtr, totalBytes));

    for (uint32_t step = 0; step < 4; ++step) {
        const uint32_t ranksInBlock = 1U << step;
        const uint32_t partnerRank = param.myRank ^ (1U << step);
        const uint32_t channelIndex = channelIndexByRemoteRank[partnerRank];
        CHK_PRT_RET(channelIndex >= resCtx.channels.size(),
            HCCL_ERROR("AllGather recursive-doubling channel to rank %u is missing", partnerRank),
            HCCL_E_NOT_FOUND);
        const ChannelInfo &partnerChannel = resCtx.channels[channelIndex];

        uint64_t blockBytes = 0;
        CHK_PRT_RET(!CheckedMultiply(ranksInBlock, totalBytes, blockBytes),
            HCCL_ERROR("AllGather recursive-doubling block size overflows"), HCCL_E_PARA);
        const uint32_t localBlockRank = param.myRank & ~(ranksInBlock - 1U);
        uint64_t localBlockOffset = 0;
        CHK_PRT_RET(!CheckedMultiply(localBlockRank, totalBytes, localBlockOffset),
            HCCL_ERROR("AllGather recursive-doubling block offset overflows"), HCCL_E_PARA);
        CHK_PRT_RET(localBlockOffset > outputBytes || blockBytes > outputBytes - localBlockOffset,
            HCCL_ERROR("AllGather recursive-doubling local block exceeds CCL buffer"), HCCL_E_PARA);
        CHK_PRT_RET(bankOffset > partnerChannel.remoteCclMem.size ||
                outputBytes > partnerChannel.remoteCclMem.size - bankOffset,
            HCCL_ERROR("AllGather recursive-doubling remote CCL buffer for rank %u is too small", partnerRank),
            HCCL_E_PARA);

        const void *localBlock = nullptr;
        void *remoteBankAddr = nullptr;
        void *remoteBlock = nullptr;
        CHK_PRT_RET(!CheckedAddressOffset(localBank.addr, localBlockOffset, localBlock) ||
                !CheckedAddressOffset(partnerChannel.remoteCclMem.addr, bankOffset, remoteBankAddr) ||
                !CheckedAddressOffset(remoteBankAddr, localBlockOffset, remoteBlock),
            HCCL_ERROR("AllGather recursive-doubling block address overflows"), HCCL_E_PARA);

        CHK_RET(HcommWriteOnThread(thread, partnerChannel.handle, remoteBlock, localBlock, blockBytes));
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, partnerChannel.handle, dataNotifyIndex));
        CHK_RET(HcommChannelNotifyWaitOnThread(
            thread, partnerChannel.handle, dataNotifyIndex, CUSTOM_TIMEOUT));
    }

    CHK_RET(HcommLocalCopyOnThread(thread, param.outputPtr, localBank.addr, outputBytes));
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }

    uint64_t dataTypeSize = 0;
    CHK_RET(GetDataTypeSize(param.dataType, dataTypeSize));

    uint64_t totalBytes = 0;
    CHK_PRT_RET(!CheckedMultiply(param.count, dataTypeSize, totalBytes),
        HCCL_ERROR("AllGather byte count overflows: count %llu, element size %llu",
            static_cast<unsigned long long>(param.count), static_cast<unsigned long long>(dataTypeSize)),
        HCCL_E_PARA);
    uint64_t outputBytes = 0;
    CHK_PRT_RET(!CheckedMultiply(param.rankSize, totalBytes, outputBytes),
        HCCL_ERROR("AllGather output byte count overflows: ranks %u, slice bytes %llu", param.rankSize,
            static_cast<unsigned long long>(totalBytes)),
        HCCL_E_PARA);

    CHK_PRT_RET(param.rankSize == 0 || param.myRank >= param.rankSize,
        HCCL_ERROR("Invalid rank metadata: rank %u of %u", param.myRank, param.rankSize), HCCL_E_PARA);
    CHK_PRT_RET(param.rankSize > MAX_ALLGATHER_RANK_SIZE,
        HCCL_ERROR("AllGather rank size %u exceeds the fixed 2x8 competition topology", param.rankSize),
        HCCL_E_PARA);
    CHK_PTR_NULL(param.inputPtr);
    CHK_PTR_NULL(param.outputPtr);

    const uint32_t peerCount = param.rankSize - 1;
    CHK_PRT_RET(resCtx.channels.size() != peerCount,
        HCCL_ERROR("AllGather channel cardinality mismatch: %zu != %u", resCtx.channels.size(), peerCount),
        HCCL_E_PARA);
    if (param.rankSize == 1) {
        CHK_PRT_RET(resCtx.threads.size() != 1,
            HCCL_ERROR("Single-rank AllGather requires exactly one thread, got %zu", resCtx.threads.size()),
            HCCL_E_PARA);
        CHK_RET(HcommLocalCopyOnThread(resCtx.threads[0], param.outputPtr, param.inputPtr, totalBytes));
        return HCCL_SUCCESS;
    }

    const uint32_t expectedThreadCount = peerCount + 1;
    CHK_PRT_RET(resCtx.threads.size() != expectedThreadCount,
        HCCL_ERROR("AllGather thread cardinality mismatch: %zu != %u", resCtx.threads.size(),
            expectedThreadCount),
        HCCL_E_PARA);
    CHK_PTR_NULL(resCtx.localBuffer.addr);
    CHK_PRT_RET(param.root >= 4,
        HCCL_ERROR("Invalid CCL buffer epoch %u", param.root), HCCL_E_PARA);
    const uint32_t bufferBank = param.root & BUFFER_BANK_MASK;
    const bool previousUsedFullBuffer = (param.root & PREVIOUS_FULL_BUFFER_FLAG) != 0;
    const uint64_t bankSize = resCtx.localBuffer.size / 2;
    const bool useFullBufferDirect = outputBytes > bankSize;
    const uint64_t workingBufferSize = useFullBufferDirect ? resCtx.localBuffer.size : bankSize;
    const uint64_t workingBufferOffset = useFullBufferDirect ? 0 : bufferBank * bankSize;
    const uint32_t dataNotifyIndex = NOTIFY_IDX_DATA_SIGNAL + bufferBank;
    void *localWorkingBufferAddr = nullptr;
    CHK_PRT_RET(workingBufferSize == 0 ||
            !CheckedAddressOffset(resCtx.localBuffer.addr, workingBufferOffset, localWorkingBufferAddr),
        HCCL_ERROR("Invalid local CCL working buffer for bank %u", bufferBank), HCCL_E_PARA);
    const CommBuffer localWorkingBuffer = CommBuffer{localWorkingBufferAddr, workingBufferSize};
    const uint64_t rankStridedSliceCapacity = localWorkingBuffer.size / param.rankSize;
    const uint64_t alignedRankStridedSliceCapacity =
        rankStridedSliceCapacity - (rankStridedSliceCapacity % dataTypeSize);
    const bool useLargeReadPull = param.rankSize == MAX_ALLGATHER_RANK_SIZE &&
        outputBytes >= LARGE_READ_PULL_OUTPUT_THRESHOLD_BYTES;
    const bool usePackedReadPull = useLargeReadPull &&
        totalBytes > alignedRankStridedSliceCapacity;
    const bool useDedicatedPackedControl = usePackedReadPull;
    const uint32_t workerThreadOffset = useDedicatedPackedControl ? 1 : 0;
    const uint32_t firstWorkerChannelIndex = useDedicatedPackedControl ? 0 : 1;
    const uint32_t completionNotifyIndexOffset = useDedicatedPackedControl ? 1 : 0;
    const uint64_t packedSliceCapacity =
        std::min(localWorkingBuffer.size, MAX_READ_PULL_CHUNK_BYTES);
    const uint64_t sliceCapacity =
        usePackedReadPull ? packedSliceCapacity : rankStridedSliceCapacity;
    const uint64_t alignedSliceCapacity = sliceCapacity - (sliceCapacity % dataTypeSize);
    CHK_PRT_RET(alignedSliceCapacity == 0,
        HCCL_ERROR("HCCL buffer %llu cannot hold one %llu-byte element for each of %u ranks",
            static_cast<unsigned long long>(localWorkingBuffer.size),
            static_cast<unsigned long long>(dataTypeSize), param.rankSize),
        HCCL_E_PARA);
    const bool precomputeCompletionTree = param.rankSize == MAX_ALLGATHER_RANK_SIZE &&
        totalBytes <= alignedSliceCapacity &&
        (useLargeReadPull || outputBytes >= BINARY_FANIN_OUTPUT_THRESHOLD_BYTES);

    uint32_t channelIndexByRemoteRank[MAX_ALLGATHER_RANK_SIZE];
    uint32_t binaryFanInChannelIndices[BINARY_FANIN_HALF_COUNT][MAX_ALLGATHER_RANK_SIZE];
    uint32_t binaryFanInChannelCounts[BINARY_FANIN_HALF_COUNT] = {0, 0};
    uint32_t binaryFanInNodePositions[MAX_ALLGATHER_RANK_SIZE];
    for (uint32_t index = 0; index < MAX_ALLGATHER_RANK_SIZE; ++index) {
        channelIndexByRemoteRank[index] = peerCount;
        binaryFanInNodePositions[index] = peerCount;
    }

    uint32_t seenRemoteRanks = 1U << param.myRank;
    for (uint32_t channelIndex = 0; channelIndex < peerCount; ++channelIndex) {
        const ChannelInfo &channel = resCtx.channels[channelIndex];
        CHK_PRT_RET(channel.remoteRank >= param.rankSize || channel.remoteRank == param.myRank,
            HCCL_ERROR("Invalid or duplicate AllGather channel rank %u at index %u", channel.remoteRank,
                channelIndex),
            HCCL_E_PARA);
        const uint32_t remoteRankBit = 1U << channel.remoteRank;
        CHK_PRT_RET((seenRemoteRanks & remoteRankBit) != 0,
            HCCL_ERROR("Invalid or duplicate AllGather channel rank %u at index %u", channel.remoteRank,
                channelIndex),
            HCCL_E_PARA);
        seenRemoteRanks |= remoteRankBit;
        channelIndexByRemoteRank[channel.remoteRank] = channelIndex;
        if (precomputeCompletionTree && channelIndex > 0) {
            const uint32_t binaryFanInHalf = channel.remoteRank / BINARY_FANIN_HALF_RANKS;
            const uint32_t nodePosition = binaryFanInChannelCounts[binaryFanInHalf]++;
            binaryFanInChannelIndices[binaryFanInHalf][nodePosition] = channelIndex;
            binaryFanInNodePositions[channelIndex] = nodePosition;
        }
        CHK_PRT_RET(channel.notifyNum < DIRECT_CHANNEL_NOTIFY_COUNT,
            HCCL_ERROR("AllGather channel for rank %u has only %u notifies", channel.remoteRank, channel.notifyNum),
            HCCL_E_PARA);
        CHK_PTR_NULL(channel.remoteCclMem.addr);
        CHK_PRT_RET(workingBufferOffset > channel.remoteCclMem.size ||
                workingBufferSize > channel.remoteCclMem.size - workingBufferOffset,
            HCCL_ERROR("Remote CCL buffer for rank %u cannot hold selected working buffer", channel.remoteRank),
            HCCL_E_PARA);
    }

    const bool useRecursiveDoubling = param.rankSize == MAX_ALLGATHER_RANK_SIZE &&
        outputBytes <= SMALL_MESSAGE_OUTPUT_THRESHOLD_BYTES && outputBytes <= localWorkingBuffer.size;
    if (useRecursiveDoubling) {
        return RunRecursiveDoublingSmallMessage(
            param, resCtx, channelIndexByRemoteRank, localWorkingBuffer,
            workingBufferOffset, dataNotifyIndex, totalBytes, outputBytes);
    }

    const ThreadHandle mainThread = resCtx.threads[0];
    uint32_t binaryFanInRootChannelIndices[BINARY_FANIN_HALF_COUNT] = {peerCount, peerCount};
    ThreadHandle binaryFanInParentThreads[MAX_ALLGATHER_RANK_SIZE];
    uint32_t binaryFanInChildChannelIndices[MAX_ALLGATHER_RANK_SIZE][2];
    if (precomputeCompletionTree) {
        for (uint32_t channelIndex = 0; channelIndex < MAX_ALLGATHER_RANK_SIZE; ++channelIndex) {
            binaryFanInParentThreads[channelIndex] = mainThread;
            binaryFanInChildChannelIndices[channelIndex][0] = peerCount;
            binaryFanInChildChannelIndices[channelIndex][1] = peerCount;
        }
        for (uint32_t binaryFanInHalf = 0;
                binaryFanInHalf < BINARY_FANIN_HALF_COUNT; ++binaryFanInHalf) {
            const uint32_t nodeCount = binaryFanInChannelCounts[binaryFanInHalf];
            if (nodeCount > 0) {
                binaryFanInRootChannelIndices[binaryFanInHalf] =
                    binaryFanInChannelIndices[binaryFanInHalf][0];
            }
            for (uint32_t nodePosition = 0; nodePosition < nodeCount; ++nodePosition) {
                const uint32_t channelIndex =
                    binaryFanInChannelIndices[binaryFanInHalf][nodePosition];
                if (nodePosition > 0) {
                    const uint32_t parentPosition = (nodePosition - 1) / 2;
                    const uint32_t parentChannelIndex =
                        binaryFanInChannelIndices[binaryFanInHalf][parentPosition];
                    binaryFanInParentThreads[channelIndex] = resCtx.threads[parentChannelIndex];
                }
                for (uint32_t childOffset = 0; childOffset < 2; ++childOffset) {
                    const uint32_t childPosition = 2 * nodePosition + childOffset + 1;
                    if (childPosition < nodeCount) {
                        binaryFanInChildChannelIndices[channelIndex][childOffset] =
                            binaryFanInChannelIndices[binaryFanInHalf][childPosition];
                    }
                }
            }
        }
    }
    uint64_t processedBytes = 0;
    while (processedBytes < totalBytes) {
        const uint64_t sliceSize = std::min(totalBytes - processedBytes, alignedSliceCapacity);
        const bool useContiguousOutput = !usePackedReadPull &&
            processedBytes == 0 && sliceSize == totalBytes;
        const bool useBinaryFanInOutput = !useLargeReadPull && useContiguousOutput &&
            param.rankSize == MAX_ALLGATHER_RANK_SIZE &&
            outputBytes >= BINARY_FANIN_OUTPUT_THRESHOLD_BYTES;
        const bool useReadPullBinaryOutput = useLargeReadPull && useContiguousOutput;
        const bool useCompletionTree = useBinaryFanInOutput || useReadPullBinaryOutput;
        const uint32_t mainBinaryFanInHalf = param.myRank / BINARY_FANIN_HALF_RANKS;
        const uint32_t readPullMainHalf =
            resCtx.channels[0].remoteRank / BINARY_FANIN_HALF_RANKS;
        uint64_t binaryFanInHalfBytes = 0;
        uint32_t binaryFanInRootNotifyIndices[BINARY_FANIN_HALF_COUNT];
        for (uint32_t binaryFanInHalf = 0; binaryFanInHalf < BINARY_FANIN_HALF_COUNT; ++binaryFanInHalf) {
            binaryFanInRootNotifyIndices[binaryFanInHalf] = 0;
        }
        if (useCompletionTree) {
            CHK_PRT_RET(!CheckedMultiply(BINARY_FANIN_HALF_RANKS, sliceSize, binaryFanInHalfBytes) ||
                    binaryFanInHalfBytes * BINARY_FANIN_HALF_COUNT != outputBytes,
                HCCL_ERROR("AllGather binary fan-in half size is inconsistent"), HCCL_E_PARA);
            for (uint32_t binaryFanInHalf = 0; binaryFanInHalf < BINARY_FANIN_HALF_COUNT; ++binaryFanInHalf) {
                const uint32_t rootChannelIndex = binaryFanInRootChannelIndices[binaryFanInHalf];
                CHK_PRT_RET(rootChannelIndex >= peerCount,
                    HCCL_ERROR("AllGather binary fan-in root is missing for half %u", binaryFanInHalf),
                    HCCL_E_NOT_FOUND);
                binaryFanInRootNotifyIndices[binaryFanInHalf] =
                    MAIN_THREAD_BARRIER_NOTIFY_BASE + rootChannelIndex - 1;
            }
        }
        uint64_t chunkBufferBytes = 0;
        const uint64_t chunkBufferRanks = usePackedReadPull ? 1 : param.rankSize;
        CHK_PRT_RET(!CheckedMultiply(chunkBufferRanks, sliceSize, chunkBufferBytes),
            HCCL_ERROR("AllGather chunk buffer size overflows"), HCCL_E_PARA);
        CHK_PRT_RET(chunkBufferBytes > localWorkingBuffer.size,
            HCCL_ERROR("AllGather chunk %llu exceeds local HCCL buffer %llu",
                static_cast<unsigned long long>(chunkBufferBytes),
                static_cast<unsigned long long>(localWorkingBuffer.size)),
            HCCL_E_PARA);

        uint64_t localRankOffset = 0;
        if (usePackedReadPull) {
            localRankOffset = PackedRankAlignmentOffset(param.myRank, totalBytes);
        }
        CHK_PRT_RET((!usePackedReadPull &&
                !CheckedMultiply(param.myRank, sliceSize, localRankOffset)) ||
                localRankOffset > localWorkingBuffer.size ||
                sliceSize > localWorkingBuffer.size - localRankOffset,
            HCCL_ERROR("AllGather local rank offset overflows"), HCCL_E_PARA);
        const void *inputChunk = nullptr;
        void *localRankSlot = nullptr;
        CHK_PRT_RET(!CheckedAddressOffset(param.inputPtr, processedBytes, inputChunk) ||
                !CheckedAddressOffset(localWorkingBuffer.addr, localRankOffset, localRankSlot),
            HCCL_ERROR("AllGather local copy address overflows"), HCCL_E_PARA);
        CHK_RET(HcommLocalCopyOnThread(mainThread, localRankSlot, inputChunk, sliceSize));

        if (useLargeReadPull) {
            CHK_RET(RunDisseminationBarrier(param, resCtx, channelIndexByRemoteRank, NOTIFY_IDX_ACK));
        }

        for (uint32_t channelIndex = firstWorkerChannelIndex;
                channelIndex < peerCount; ++channelIndex) {
            CHK_RET(HcommThreadNotifyRecordOnThread(
                mainThread, resCtx.threads[channelIndex + workerThreadOffset],
                SLAVE_THREAD_START_NOTIFY_INDEX));
        }

        for (uint32_t channelIndex = 0; channelIndex < peerCount; ++channelIndex) {
            const ChannelInfo &channel = resCtx.channels[channelIndex];
            const ThreadHandle workerThread =
                resCtx.threads[channelIndex + workerThreadOffset];
            if (channelIndex >= firstWorkerChannelIndex) {
                CHK_RET(HcommThreadNotifyWaitOnThread(
                    workerThread, SLAVE_THREAD_START_NOTIFY_INDEX, CUSTOM_TIMEOUT));
            }
            if (!useLargeReadPull &&
                    ((useFullBufferDirect && !previousUsedFullBuffer) || processedBytes != 0)) {
                CHK_RET(HcommChannelNotifyRecordOnThread(workerThread, channel.handle, NOTIFY_IDX_ACK));
                CHK_RET(HcommChannelNotifyWaitOnThread(
                    workerThread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
            }

            if (useLargeReadPull) {
                uint64_t remoteRankOffset = 0;
                void *remoteBankAddr = nullptr;
                void *localPeerSlot = nullptr;
                void *remotePeerSlot = nullptr;
                CHK_PRT_RET(!CheckedAddressOffset(
                                channel.remoteCclMem.addr, workingBufferOffset, remoteBankAddr),
                    HCCL_ERROR("AllGather remote READ bank address overflows for rank %u", channel.remoteRank),
                    HCCL_E_PARA);
                if (usePackedReadPull) {
                    remoteRankOffset = PackedRankAlignmentOffset(channel.remoteRank, totalBytes);
                }
                CHK_PRT_RET((!usePackedReadPull &&
                        !CheckedMultiply(channel.remoteRank, sliceSize, remoteRankOffset)) ||
                        remoteRankOffset > workingBufferSize ||
                        sliceSize > workingBufferSize - remoteRankOffset,
                    HCCL_ERROR("AllGather remote READ rank offset overflows for rank %u", channel.remoteRank),
                    HCCL_E_PARA);
                CHK_PRT_RET(
                        !CheckedAddressOffset(localWorkingBuffer.addr, remoteRankOffset, localPeerSlot) ||
                        !CheckedAddressOffset(remoteBankAddr, remoteRankOffset, remotePeerSlot),
                    HCCL_ERROR("AllGather remote READ address overflows for rank %u", channel.remoteRank),
                    HCCL_E_PARA);
                if (useContiguousOutput) {
                    CHK_RET(HcommReadOnThread(
                        workerThread, channel.handle, localPeerSlot, remotePeerSlot, sliceSize));
                } else {
                    uint64_t outputRankOffset = 0;
                    CHK_PRT_RET(!CheckedMultiply(channel.remoteRank, totalBytes, outputRankOffset) ||
                            outputRankOffset > outputBytes || processedBytes > outputBytes - outputRankOffset ||
                            sliceSize > outputBytes - outputRankOffset - processedBytes,
                        HCCL_ERROR("AllGather direct READ output offset overflows for rank %u", channel.remoteRank),
                        HCCL_E_PARA);
                    void *peerOutputChunk = nullptr;
                    CHK_PRT_RET(!CheckedAddressOffset(
                                    param.outputPtr, outputRankOffset + processedBytes, peerOutputChunk),
                        HCCL_ERROR("AllGather direct READ output address overflows for rank %u", channel.remoteRank),
                        HCCL_E_PARA);
                    CHK_RET(HcommReadOnThread(
                        workerThread, channel.handle, peerOutputChunk, remotePeerSlot, sliceSize));
                }
            } else {
                void *remoteBankAddr = nullptr;
                void *remoteRankSlot = nullptr;
                CHK_PRT_RET(!CheckedAddressOffset(
                                channel.remoteCclMem.addr, workingBufferOffset, remoteBankAddr) ||
                        !CheckedAddressOffset(remoteBankAddr, param.myRank * sliceSize, remoteRankSlot),
                    HCCL_ERROR("Remote HCCL rank slot address overflows for rank %u", channel.remoteRank),
                    HCCL_E_PARA);
                CHK_RET(HcommWriteOnThread(
                    workerThread, channel.handle, remoteRankSlot, localRankSlot, sliceSize));
                CHK_RET(HcommChannelNotifyRecordOnThread(
                    workerThread, channel.handle, dataNotifyIndex));
            }
            const uint32_t binaryFanInHalf = channel.remoteRank / BINARY_FANIN_HALF_RANKS;
            const bool waitForDataOnBinaryRoot = useBinaryFanInOutput &&
                channelIndex == 0 &&
                binaryFanInHalf != mainBinaryFanInHalf;
            if (!useLargeReadPull && !waitForDataOnBinaryRoot) {
                CHK_RET(HcommChannelNotifyWaitOnThread(
                    workerThread, channel.handle, dataNotifyIndex, CUSTOM_TIMEOUT));
            }

            if (useCompletionTree && channelIndex > 0) {
                const uint32_t nodePosition = binaryFanInNodePositions[channelIndex];
                for (uint32_t binaryFanInChildOffset = 1;
                        binaryFanInChildOffset <= 2; ++binaryFanInChildOffset) {
                    const uint32_t childChannelIndex =
                        binaryFanInChildChannelIndices[channelIndex][binaryFanInChildOffset - 1];
                    if (childChannelIndex >= peerCount) {
                        continue;
                    }
                    const uint32_t childNotifyIndex =
                        MAIN_THREAD_BARRIER_NOTIFY_BASE + childChannelIndex - 1;
                    CHK_RET(HcommThreadNotifyWaitOnThread(
                        workerThread, childNotifyIndex, CUSTOM_TIMEOUT));
                }
                if (nodePosition == 0) {
                    if (useBinaryFanInOutput && binaryFanInHalf != mainBinaryFanInHalf) {
                        const ChannelInfo &mainChannel = resCtx.channels[0];
                        if (mainChannel.remoteRank / BINARY_FANIN_HALF_RANKS == binaryFanInHalf) {
                            CHK_RET(HcommChannelNotifyWaitOnThread(
                                workerThread, mainChannel.handle, dataNotifyIndex, CUSTOM_TIMEOUT));
                        }
                        CHK_RET(QueueBinaryFanInHalfCopy(
                            workerThread, localWorkingBuffer, param.outputPtr,
                            binaryFanInHalf, binaryFanInHalfBytes));
                    } else if (useReadPullBinaryOutput && binaryFanInHalf != readPullMainHalf) {
                        CHK_RET(QueueBinaryFanInHalfCopy(
                            workerThread, localWorkingBuffer, param.outputPtr,
                            binaryFanInHalf, binaryFanInHalfBytes));
                    }
                }
            }

            if (!useContiguousOutput && !useLargeReadPull) {
                uint64_t localPeerOffset = 0;
                uint64_t outputRankOffset = 0;
                CHK_PRT_RET(!CheckedMultiply(channel.remoteRank, sliceSize, localPeerOffset) ||
                        !CheckedMultiply(channel.remoteRank, totalBytes, outputRankOffset) ||
                        outputRankOffset > outputBytes - processedBytes ||
                        sliceSize > outputBytes - outputRankOffset - processedBytes,
                    HCCL_ERROR("AllGather parallel output offset overflows for rank %u", channel.remoteRank),
                    HCCL_E_PARA);
                const void *localPeerSlot = nullptr;
                void *peerOutputChunk = nullptr;
                CHK_PRT_RET(!CheckedAddressOffset(localWorkingBuffer.addr, localPeerOffset, localPeerSlot) ||
                        !CheckedAddressOffset(param.outputPtr, outputRankOffset + processedBytes, peerOutputChunk),
                    HCCL_ERROR("AllGather parallel output address overflows for rank %u", channel.remoteRank),
                    HCCL_E_PARA);
                CHK_RET(HcommLocalCopyOnThread(workerThread, peerOutputChunk, localPeerSlot, sliceSize));
            }

            if (channelIndex >= firstWorkerChannelIndex) {
                const uint32_t mainNotifyIndex = channelIndex + completionNotifyIndexOffset;
                const ThreadHandle completionTargetThread =
                    useCompletionTree ? binaryFanInParentThreads[channelIndex] : mainThread;
                CHK_RET(HcommThreadNotifyRecordOnThread(
                    workerThread, completionTargetThread, mainNotifyIndex));
            }
        }

        if (!useContiguousOutput) {
            uint64_t selfOutputRankOffset = 0;
            CHK_PRT_RET(!CheckedMultiply(param.myRank, totalBytes, selfOutputRankOffset) ||
                    selfOutputRankOffset > outputBytes - processedBytes ||
                    sliceSize > outputBytes - selfOutputRankOffset - processedBytes,
                HCCL_ERROR("AllGather self output offset overflows"), HCCL_E_PARA);
            void *selfOutputChunk = nullptr;
            CHK_PRT_RET(!CheckedAddressOffset(
                            param.outputPtr, selfOutputRankOffset + processedBytes, selfOutputChunk),
                HCCL_ERROR("AllGather self output address overflows"), HCCL_E_PARA);
            CHK_RET(HcommLocalCopyOnThread(mainThread, selfOutputChunk, localRankSlot, sliceSize));
        }

        if (useReadPullBinaryOutput) {
            CHK_RET(HcommThreadNotifyWaitOnThread(
                mainThread, binaryFanInRootNotifyIndices[readPullMainHalf], CUSTOM_TIMEOUT));
            CHK_RET(QueueBinaryFanInHalfCopy(
                mainThread, localWorkingBuffer, param.outputPtr,
                readPullMainHalf, binaryFanInHalfBytes));
            CHK_RET(HcommThreadNotifyWaitOnThread(
                mainThread, binaryFanInRootNotifyIndices[1 - readPullMainHalf], CUSTOM_TIMEOUT));
        } else if (useBinaryFanInOutput) {
            CHK_RET(HcommThreadNotifyWaitOnThread(
                mainThread, binaryFanInRootNotifyIndices[mainBinaryFanInHalf], CUSTOM_TIMEOUT));
            CHK_RET(QueueBinaryFanInHalfCopy(
                mainThread, localWorkingBuffer, param.outputPtr,
                mainBinaryFanInHalf, binaryFanInHalfBytes));
            CHK_RET(HcommThreadNotifyWaitOnThread(
                mainThread, binaryFanInRootNotifyIndices[1 - mainBinaryFanInHalf], CUSTOM_TIMEOUT));
        } else {
            for (uint32_t channelIndex = firstWorkerChannelIndex;
                    channelIndex < peerCount; ++channelIndex) {
                const uint32_t mainNotifyIndex = channelIndex + completionNotifyIndexOffset;
                CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, mainNotifyIndex, CUSTOM_TIMEOUT));
            }
            if (useContiguousOutput) {
                CHK_RET(HcommLocalCopyOnThread(
                    mainThread, param.outputPtr, localWorkingBuffer.addr, outputBytes));
            }
        }
        processedBytes += sliceSize;
        if (useLargeReadPull) {
            CHK_RET(RunDisseminationBarrier(param, resCtx, channelIndexByRemoteRank, dataNotifyIndex));
        }
    }

    if (useFullBufferDirect && !useLargeReadPull) {
        CHK_RET(RunFullBufferExitBarrier(param, resCtx, channelIndexByRemoteRank));
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
