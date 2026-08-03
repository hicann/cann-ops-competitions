/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include <array>
#include <limits>
#include <vector>

namespace {
constexpr uint32_t PIPELINE_BANK_NUM = 2;
constexpr uint64_t PIPELINE_MIN_OVERFLOW_BYTES = 4ULL * 1024ULL * 1024ULL;

void *AddOffset(void *addr, uint64_t offset)
{
    return static_cast<void *>(static_cast<uint8_t *>(addr) + offset);
}

bool IsPowerOfTwo(uint32_t value)
{
    return value != 0 && (value & (value - 1)) == 0;
}

uint32_t GetLog2(uint32_t value)
{
    uint32_t result = 0;
    while (value > 1) {
        value >>= 1;
        ++result;
    }
    return result;
}

void *GetSlotAddr(void *buffer, uint64_t bankOffset, uint32_t slotIdx, uint64_t slotStride)
{
    return AddOffset(buffer, bankOffset + static_cast<uint64_t>(slotIdx) * slotStride);
}

uint32_t GetRemoteSlotIndex(uint32_t rank, uint32_t localRank)
{
    return rank < localRank ? rank : rank - 1;
}

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return alignment == 0 ? value : (value + alignment - 1) / alignment * alignment;
}

HcclResult ValidateWorkerIndices(const std::vector<ThreadHandle> &threads,
    const std::vector<uint32_t> &workerIndices)
{
    for (uint32_t i = 0; i < workerIndices.size(); ++i) {
        CHK_PRT_RET(workerIndices[i] == 0 || workerIndices[i] >= threads.size(),
            HCCL_ERROR("Invalid worker thread index[%u]", workerIndices[i]), HCCL_E_INTERNAL);
        CHK_PRT_RET(std::find(workerIndices.begin(), workerIndices.begin() + i, workerIndices[i]) !=
                workerIndices.begin() + i,
            HCCL_ERROR("Duplicate worker thread index[%u]", workerIndices[i]), HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult JoinWorkerThreads(const std::vector<ThreadHandle> &threads,
    const std::vector<uint32_t> &workerIndices)
{
    for (uint32_t threadIdx : workerIndices) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(threads[0], threadIdx - 1, CUSTOM_TIMEOUT)));
    }
    for (uint32_t threadIdx : workerIndices) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(threads[threadIdx], threads[0], threadIdx - 1)));
    }
    return HCCL_SUCCESS;
}

int32_t FindRankIndex(const std::vector<uint32_t> &ranks, uint32_t rank)
{
    const auto iter = std::find(ranks.begin(), ranks.end(), rank);
    return iter == ranks.end() ? -1 : static_cast<int32_t>(std::distance(ranks.begin(), iter));
}

int32_t FindChannelIndex(const std::vector<ChannelInfo> &channels, uint32_t remoteRank)
{
    for (uint32_t i = 0; i < channels.size(); ++i) {
        if (channels[i].remoteRank == remoteRank) {
            return static_cast<int32_t>(i);
        }
    }
    return -1;
}

HcclResult ExchangeOneSlice(ThreadHandle thread, const ChannelInfo &channel, void *remoteDst, void *localSrc,
    uint64_t size)
{
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(thread, channel.handle, remoteDst, localSrc, size)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult ExchangeOneSliceReduce(ThreadHandle thread, const ChannelInfo &channel, void *remoteDst, void *localSrc,
    uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp)
{
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(
        HcommWriteReduceOnThread(thread, channel.handle, remoteDst, localSrc, count, dataType, reduceOp)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult ExchangeTwoSlices(ThreadHandle thread, const ChannelInfo &channel, void *remoteDst0, void *localSrc0,
    void *remoteDst1, void *localSrc1, uint64_t size)
{
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(thread, channel.handle, remoteDst0, localSrc0, size)));
    CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(thread, channel.handle, remoteDst1, localSrc1, size)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult SyncMainToWorker(ThreadHandle mainThread, ThreadHandle workerThread)
{
    if (mainThread == workerThread) {
        return HCCL_SUCCESS;
    }
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(mainThread, workerThread, 0)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(workerThread, 0, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult SyncMainToWorkers(const std::vector<ThreadHandle> &threads,
    const std::vector<uint32_t> &workerIndices)
{
    for (uint32_t threadIdx : workerIndices) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[0], threads[threadIdx], 0)));
    }
    for (uint32_t threadIdx : workerIndices) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[threadIdx], 0, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult JoinWorker(const std::vector<ThreadHandle> &threads, uint32_t workerIdx)
{
    if (workerIdx == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(workerIdx >= threads.size(), HCCL_ERROR("Invalid worker thread index[%u]", workerIdx),
        HCCL_E_INTERNAL);
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyWaitOnThread(threads[0], workerIdx - 1, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyRecordOnThread(threads[workerIdx], threads[0], workerIdx - 1)));
    return HCCL_SUCCESS;
}

HcclResult ValidateResources(const OpParam &param, const AlgResourceCtx &resCtx, uint32_t dataTypeSize)
{
    CHK_PRT_RET(param.rankSize == 0 || param.myRank >= param.rankSize,
        HCCL_ERROR("Invalid rank information: rank[%u], rankSize[%u]", param.myRank, param.rankSize), HCCL_E_PARA);
    CHK_PRT_RET(resCtx.threads.empty(), HCCL_ERROR("No AICPU thread resource"), HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.localBuffer.addr == nullptr || resCtx.localBuffer.size == 0,
        HCCL_ERROR("Invalid local CCL buffer"), HCCL_E_INTERNAL);
    CHK_PRT_RET(param.rankSize > 1 &&
            resCtx.hierarchyMode != custom_reducescatter::kHierarchySmallNhr &&
            resCtx.threads.size() != resCtx.channels.size(),
        HCCL_ERROR("Thread/channel count mismatch: threads[%zu], channels[%zu]", resCtx.threads.size(),
            resCtx.channels.size()), HCCL_E_INTERNAL);
    for (uint32_t i = 0; i < resCtx.channels.size(); ++i) {
        CHK_PRT_RET(resCtx.channels[i].remoteRank == param.myRank,
            HCCL_ERROR("Channel unexpectedly targets local rank[%u]", param.myRank), HCCL_E_INTERNAL);
        for (uint32_t j = 0; j < i; ++j) {
            CHK_PRT_RET(resCtx.channels[i].remoteRank == resCtx.channels[j].remoteRank,
                HCCL_ERROR("Multiple channels target remote rank[%u]", resCtx.channels[i].remoteRank),
                HCCL_E_INTERNAL);
        }
    }
    if (resCtx.hierarchyMode == custom_reducescatter::kHierarchySmallNhr) {
        const uint32_t localRankSize = static_cast<uint32_t>(resCtx.group0Ranks.size());
        CHK_PRT_RET(!IsPowerOfTwo(param.rankSize) || localRankSize == 0 ||
                resCtx.group1Ranks.size() != localRankSize || localRankSize * 2 != param.rankSize ||
                resCtx.localGroupIndex > 1 || resCtx.localRankIndex >= localRankSize || resCtx.threads.size() != 1 ||
                resCtx.channels.size() != GetLog2(param.rankSize),
            HCCL_ERROR("Invalid small NHR resources for rank[%u]", param.myRank), HCCL_E_INTERNAL);
        const std::vector<uint32_t> &localRanks =
            resCtx.localGroupIndex == 0 ? resCtx.group0Ranks : resCtx.group1Ranks;
        const std::vector<uint32_t> &otherRanks =
            resCtx.localGroupIndex == 0 ? resCtx.group1Ranks : resCtx.group0Ranks;
        CHK_PRT_RET(localRanks[resCtx.localRankIndex] != param.myRank ||
                otherRanks[resCtx.localRankIndex] != resCtx.peerRank,
            HCCL_ERROR("Invalid small NHR rank mapping for rank[%u]", param.myRank), HCCL_E_INTERNAL);

        const uint32_t algRank = resCtx.localGroupIndex * localRankSize + resCtx.localRankIndex;
        for (uint32_t mask = param.rankSize / 2; mask > 0; mask >>= 1) {
            const uint32_t peerAlgRank = algRank ^ mask;
            const uint32_t peerRank = peerAlgRank < localRankSize ? resCtx.group0Ranks[peerAlgRank] :
                resCtx.group1Ranks[peerAlgRank - localRankSize];
            CHK_PRT_RET(FindChannelIndex(resCtx.channels, peerRank) < 0,
                HCCL_ERROR("Small NHR channel for rank[%u] is missing", peerRank), HCCL_E_INTERNAL);
        }
    } else if (resCtx.hierarchyMode == custom_reducescatter::kHierarchyTwoServer) {
        const uint32_t localRankSize = static_cast<uint32_t>(resCtx.group0Ranks.size());
        const std::vector<uint32_t> &localRanks =
            resCtx.localGroupIndex == 0 ? resCtx.group0Ranks : resCtx.group1Ranks;
        const std::vector<uint32_t> &otherRanks =
            resCtx.localGroupIndex == 0 ? resCtx.group1Ranks : resCtx.group0Ranks;
        CHK_PRT_RET(resCtx.localGroupIndex > 1 || localRankSize <= 1 || resCtx.group1Ranks.size() != localRankSize ||
                localRankSize * 2 != param.rankSize || resCtx.localRankIndex >= localRankSize ||
                localRanks[resCtx.localRankIndex] != param.myRank ||
                otherRanks[resCtx.localRankIndex] != resCtx.peerRank || resCtx.channels.size() != localRankSize,
            HCCL_ERROR("Invalid two-server hierarchy for rank[%u]", param.myRank), HCCL_E_INTERNAL);
    } else if (resCtx.hierarchyMode == custom_reducescatter::kHierarchyDisabled) {
        CHK_PRT_RET(param.rankSize > 1 && resCtx.channels.size() != param.rankSize - 1,
            HCCL_ERROR("Flat resource count mismatch: rankSize[%u], channels[%zu]", param.rankSize,
                resCtx.channels.size()), HCCL_E_INTERNAL);
    } else {
        HCCL_ERROR("Unsupported hierarchy mode[%u]", resCtx.hierarchyMode);
        return HCCL_E_INTERNAL;
    }
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / param.rankSize ||
            param.count * param.rankSize > std::numeric_limits<uint64_t>::max() / dataTypeSize,
        HCCL_ERROR("ReduceScatter data size overflow"), HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult ExecSmallNhr(const OpParam &param, const AlgResourceCtx &resCtx, uint32_t dataTypeSize,
    uint64_t minCclBufferSize)
{
    const uint64_t sliceSize = param.count * dataTypeSize;
    const uint64_t totalSize = sliceSize * param.rankSize;
    CHK_PRT_RET(totalSize > minCclBufferSize,
        HCCL_ERROR("CCL buffer[%llu] is too small for small NHR data[%llu]",
            static_cast<unsigned long long>(minCclBufferSize), static_cast<unsigned long long>(totalSize)),
        HCCL_E_MEMORY);

    const uint32_t localRankSize = static_cast<uint32_t>(resCtx.group0Ranks.size());
    const uint32_t algRank = resCtx.localGroupIndex * localRankSize + resCtx.localRankIndex;
    std::vector<uint32_t> orderedRanks;
    orderedRanks.reserve(param.rankSize);
    orderedRanks.insert(orderedRanks.end(), resCtx.group0Ranks.begin(), resCtx.group0Ranks.end());
    orderedRanks.insert(orderedRanks.end(), resCtx.group1Ranks.begin(), resCtx.group1Ranks.end());

    bool identityOrder = true;
    for (uint32_t rankIdx = 0; rankIdx < param.rankSize; ++rankIdx) {
        if (orderedRanks[rankIdx] != rankIdx) {
            identityOrder = false;
            break;
        }
    }
    if (identityOrder) {
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(resCtx.threads[0], resCtx.localBuffer.addr, param.inputPtr, totalSize)));
    } else {
        for (uint32_t rankIdx = 0; rankIdx < param.rankSize; ++rankIdx) {
            void *source = AddOffset(param.inputPtr, static_cast<uint64_t>(orderedRanks[rankIdx]) * sliceSize);
            void *destination = AddOffset(resCtx.localBuffer.addr, static_cast<uint64_t>(rankIdx) * sliceSize);
            CHK_RET(static_cast<HcclResult>(
                HcommLocalCopyOnThread(resCtx.threads[0], destination, source, sliceSize)));
        }
    }

    uint32_t rangeStart = 0;
    uint32_t rangeBlocks = param.rankSize;
    for (uint32_t mask = param.rankSize / 2; mask > 0; mask >>= 1) {
        const uint32_t halfBlocks = rangeBlocks / 2;
        const uint32_t peerAlgRank = algRank ^ mask;
        const uint32_t peerRank = orderedRanks[peerAlgRank];
        const int32_t channelIdx = FindChannelIndex(resCtx.channels, peerRank);
        CHK_PRT_RET(channelIdx < 0, HCCL_ERROR("Small NHR channel for rank[%u] is missing", peerRank),
            HCCL_E_INTERNAL);

        uint32_t sendStart = rangeStart;
        if ((algRank & mask) == 0) {
            sendStart += halfBlocks;
        } else {
            rangeStart += halfBlocks;
        }
        const ChannelInfo &channel = resCtx.channels[static_cast<uint32_t>(channelIdx)];
        void *localSrc = AddOffset(resCtx.localBuffer.addr, static_cast<uint64_t>(sendStart) * sliceSize);
        void *remoteDst = AddOffset(channel.remoteCclMem.addr, static_cast<uint64_t>(sendStart) * sliceSize);
        CHK_RET(ExchangeOneSliceReduce(resCtx.threads[0], channel, remoteDst, localSrc,
            static_cast<uint64_t>(halfBlocks) * param.count, static_cast<HcommDataType>(param.dataType),
            static_cast<HcommReduceOp>(param.reduceType)));
        rangeBlocks = halfBlocks;
    }

    CHK_PRT_RET(rangeBlocks != 1 || rangeStart != algRank,
        HCCL_ERROR("Small NHR final range mismatch: start[%u], blocks[%u], algRank[%u]",
            rangeStart, rangeBlocks, algRank), HCCL_E_INTERNAL);
    void *result = AddOffset(resCtx.localBuffer.addr, static_cast<uint64_t>(rangeStart) * sliceSize);
    CHK_RET(static_cast<HcclResult>(
        HcommLocalCopyOnThread(resCtx.threads[0], param.outputPtr, result, sliceSize)));
    return HCCL_SUCCESS;
}

HcclResult ReduceLocalPartial(const OpParam &param, const AlgResourceCtx &resCtx,
    const std::vector<uint32_t> &localRanks, uint32_t destinationRank, uint32_t slotBase,
    uint64_t processedCount, uint64_t sliceCount, uint64_t sliceSize, uint64_t bankOffset,
    uint64_t slotStride, uint32_t dataTypeSize)
{
    void *partial = GetSlotAddr(resCtx.localBuffer.addr, bankOffset, slotBase + resCtx.localRankIndex, slotStride);
    for (uint32_t sourceIdx = 0; sourceIdx < localRanks.size(); ++sourceIdx) {
        void *source = nullptr;
        if (sourceIdx == resCtx.localRankIndex) {
            const uint64_t inputCountOffset = static_cast<uint64_t>(destinationRank) * param.count + processedCount;
            source = AddOffset(param.inputPtr, inputCountOffset * dataTypeSize);
        } else {
            source = GetSlotAddr(resCtx.localBuffer.addr, bankOffset, slotBase + sourceIdx, slotStride);
        }
        if (sourceIdx == 0) {
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resCtx.threads[0], partial, source, sliceSize)));
        } else {
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(resCtx.threads[0], partial, source, sliceCount,
                static_cast<HcommDataType>(param.dataType), static_cast<HcommReduceOp>(param.reduceType))));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceCclBufferTree(const OpParam &param, const AlgResourceCtx &resCtx, void *output,
    uint64_t sliceCount, uint64_t sliceSize)
{
    constexpr uint32_t optimizedRankSize = 16;
    if (param.rankSize == optimizedRankSize) {
        if (sliceSize <= custom_reducescatter::kSliceAlignment) {
            for (uint32_t stride = 1; stride < optimizedRankSize; stride *= 2) {
                for (uint32_t dstRank = 0; dstRank + stride < optimizedRankSize; dstRank += stride * 2) {
                    const uint32_t srcRank = dstRank + stride;
                    void *destination =
                        AddOffset(resCtx.localBuffer.addr, static_cast<uint64_t>(dstRank) * sliceSize);
                    void *source = AddOffset(resCtx.localBuffer.addr, static_cast<uint64_t>(srcRank) * sliceSize);
                    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(resCtx.threads[0], destination,
                        source, sliceCount, static_cast<HcommDataType>(param.dataType),
                        static_cast<HcommReduceOp>(param.reduceType))));
                }
            }
            CHK_RET(static_cast<HcclResult>(
                HcommLocalCopyOnThread(resCtx.threads[0], output, resCtx.localBuffer.addr, sliceSize)));
            return HCCL_SUCCESS;
        }

        constexpr uint32_t expectedThreadNum = optimizedRankSize - 1;
        CHK_PRT_RET(resCtx.threads.size() < expectedThreadNum,
            HCCL_ERROR("Insufficient tree-reduce threads[%zu]", resCtx.threads.size()), HCCL_E_INTERNAL);
        constexpr uint32_t firstRoundThreadNum = optimizedRankSize / 2;
        std::vector<uint32_t> firstRoundWorkers;
        firstRoundWorkers.reserve(firstRoundThreadNum - 1);
        for (uint32_t threadIdx = 1; threadIdx < firstRoundThreadNum; ++threadIdx) {
            firstRoundWorkers.push_back(threadIdx);
        }
        CHK_RET(ValidateWorkerIndices(resCtx.threads, firstRoundWorkers));
        CHK_RET(SyncMainToWorkers(resCtx.threads, firstRoundWorkers));
        for (uint32_t dstRank = 0; dstRank < optimizedRankSize; dstRank += 2) {
            const uint32_t srcRank = dstRank + 1;
            const uint32_t threadIdx = dstRank / 2;
            void *destination = AddOffset(resCtx.localBuffer.addr, static_cast<uint64_t>(dstRank) * sliceSize);
            void *source = AddOffset(resCtx.localBuffer.addr, static_cast<uint64_t>(srcRank) * sliceSize);
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(resCtx.threads[threadIdx], destination,
                source, sliceCount, static_cast<HcommDataType>(param.dataType),
                static_cast<HcommReduceOp>(param.reduceType))));
        }

        // Preserve the same reduction tree while replacing full level barriers with direct dependencies.
        constexpr uint32_t dependencyNotifyBase = 12;
        uint32_t notifyIdx = dependencyNotifyBase;
        for (uint32_t stride = 2; stride < optimizedRankSize; stride *= 2, ++notifyIdx) {
            for (uint32_t dstRank = 0; dstRank + stride < optimizedRankSize; dstRank += stride * 2) {
                const uint32_t srcRank = dstRank + stride;
                const uint32_t dstThreadIdx = dstRank / 2;
                const uint32_t srcThreadIdx = srcRank / 2;
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                    resCtx.threads[srcThreadIdx], resCtx.threads[dstThreadIdx], notifyIdx)));
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                    resCtx.threads[dstThreadIdx], notifyIdx, CUSTOM_TIMEOUT)));
                void *destination =
                    AddOffset(resCtx.localBuffer.addr, static_cast<uint64_t>(dstRank) * sliceSize);
                void *source = AddOffset(resCtx.localBuffer.addr, static_cast<uint64_t>(srcRank) * sliceSize);
                CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(resCtx.threads[dstThreadIdx], destination,
                    source, sliceCount, static_cast<HcommDataType>(param.dataType),
                    static_cast<HcommReduceOp>(param.reduceType))));
            }
        }
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(resCtx.threads[0], output, resCtx.localBuffer.addr, sliceSize)));
        return HCCL_SUCCESS;
    }

    std::vector<uint32_t> workerIndices;
    workerIndices.reserve(resCtx.threads.size() > 0 ? resCtx.threads.size() - 1 : 0);
    for (uint32_t stride = 1; stride < param.rankSize; stride *= 2) {
        uint32_t activeThreadNum = 0;
        for (uint32_t dstRank = 0; dstRank + stride < param.rankSize; dstRank += stride * 2) {
            ++activeThreadNum;
        }
        CHK_PRT_RET(activeThreadNum == 0 || activeThreadNum > resCtx.threads.size(),
            HCCL_ERROR("Invalid tree-reduce thread count[%u] for stride[%u]", activeThreadNum, stride),
            HCCL_E_INTERNAL);

        workerIndices.clear();
        for (uint32_t threadIdx = 1; threadIdx < activeThreadNum; ++threadIdx) {
            workerIndices.push_back(threadIdx);
        }
        CHK_RET(ValidateWorkerIndices(resCtx.threads, workerIndices));
        CHK_RET(SyncMainToWorkers(resCtx.threads, workerIndices));

        uint32_t threadIdx = 0;
        for (uint32_t dstRank = 0; dstRank + stride < param.rankSize; dstRank += stride * 2) {
            const uint32_t srcRank = dstRank + stride;
            void *destination = AddOffset(resCtx.localBuffer.addr, static_cast<uint64_t>(dstRank) * sliceSize);
            void *source = AddOffset(resCtx.localBuffer.addr, static_cast<uint64_t>(srcRank) * sliceSize);
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(resCtx.threads[threadIdx], destination,
                source, sliceCount, static_cast<HcommDataType>(param.dataType),
                static_cast<HcommReduceOp>(param.reduceType))));
            ++threadIdx;
        }
        CHK_RET(JoinWorkerThreads(resCtx.threads, workerIndices));
    }
    CHK_RET(static_cast<HcclResult>(
        HcommLocalCopyOnThread(resCtx.threads[0], output, resCtx.localBuffer.addr, sliceSize)));
    return HCCL_SUCCESS;
}

struct FlatReadyTreePlan {
    std::array<uint32_t, 8> pairOwners{};
    std::array<uint32_t, 8> signalThreads{};
    std::array<uint32_t, 8> pairResultSlots{};
};

HcclResult BuildFlatReadyTreePlan(const OpParam &param, const AlgResourceCtx &resCtx,
    FlatReadyTreePlan &plan)
{
    constexpr uint32_t rankSize = 16;
    constexpr uint32_t threadNum = rankSize - 1;
    const uint32_t invalidThreadIdx = std::numeric_limits<uint32_t>::max();
    CHK_PRT_RET(param.rankSize != rankSize || resCtx.threads.size() != threadNum ||
            resCtx.channels.size() != threadNum,
        HCCL_ERROR("Invalid producer-ready resources: rankSize[%u], threads[%zu], channels[%zu]",
            param.rankSize, resCtx.threads.size(), resCtx.channels.size()), HCCL_E_INTERNAL);

    std::array<uint32_t, rankSize> producerThreads{};
    producerThreads.fill(invalidThreadIdx);
    for (uint32_t channelIdx = 0; channelIdx < resCtx.channels.size(); ++channelIdx) {
        const uint32_t remoteRank = resCtx.channels[channelIdx].remoteRank;
        CHK_PRT_RET(remoteRank >= rankSize || remoteRank == param.myRank ||
                producerThreads[remoteRank] != invalidThreadIdx,
            HCCL_ERROR("Invalid producer channel[%u] for rank[%u]", channelIdx, remoteRank), HCCL_E_INTERNAL);
        producerThreads[remoteRank] = channelIdx;
    }
    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        CHK_PRT_RET(rank != param.myRank && producerThreads[rank] == invalidThreadIdx,
            HCCL_ERROR("Producer channel for rank[%u] is missing", rank), HCCL_E_INTERNAL);
    }

    plan.signalThreads.fill(invalidThreadIdx);
    std::array<bool, threadNum> coveredThreads{};
    for (uint32_t pairIdx = 0; pairIdx < plan.pairOwners.size(); ++pairIdx) {
        const uint32_t evenRank = pairIdx * 2;
        const uint32_t oddRank = evenRank + 1;
        uint32_t ownerThreadIdx = invalidThreadIdx;
        uint32_t signalThreadIdx = invalidThreadIdx;
        uint32_t resultRank = evenRank;
        if (evenRank == param.myRank) {
            ownerThreadIdx = producerThreads[oddRank];
            resultRank = oddRank;
        } else if (oddRank == param.myRank) {
            ownerThreadIdx = producerThreads[evenRank];
        } else {
            ownerThreadIdx = producerThreads[evenRank];
            signalThreadIdx = producerThreads[oddRank];
        }

        CHK_PRT_RET(ownerThreadIdx >= threadNum || coveredThreads[ownerThreadIdx],
            HCCL_ERROR("Invalid owner thread[%u] for pair[%u]", ownerThreadIdx, pairIdx), HCCL_E_INTERNAL);
        plan.pairOwners[pairIdx] = ownerThreadIdx;
        plan.pairResultSlots[pairIdx] = GetRemoteSlotIndex(resultRank, param.myRank);
        CHK_PRT_RET(plan.pairResultSlots[pairIdx] >= threadNum,
            HCCL_ERROR("Invalid result slot[%u] for pair[%u]", plan.pairResultSlots[pairIdx], pairIdx),
            HCCL_E_INTERNAL);
        coveredThreads[ownerThreadIdx] = true;
        if (signalThreadIdx != invalidThreadIdx) {
            CHK_PRT_RET(signalThreadIdx >= threadNum || coveredThreads[signalThreadIdx],
                HCCL_ERROR("Invalid signal thread[%u] for pair[%u]", signalThreadIdx, pairIdx), HCCL_E_INTERNAL);
            plan.signalThreads[pairIdx] = signalThreadIdx;
            coveredThreads[signalThreadIdx] = true;
        }
    }
    for (uint32_t threadIdx = 0; threadIdx < threadNum; ++threadIdx) {
        CHK_PRT_RET(!coveredThreads[threadIdx], HCCL_ERROR("Producer thread[%u] is not covered", threadIdx),
            HCCL_E_INTERNAL);
    }
    CHK_PRT_RET(plan.pairOwners[0] != 0,
        HCCL_ERROR("Producer-ready root owner[%u] is not thread 0", plan.pairOwners[0]), HCCL_E_INTERNAL);
    CHK_PRT_RET(plan.pairResultSlots[0] != 0,
        HCCL_ERROR("Producer-ready root slot[%u] is not slot 0", plan.pairResultSlots[0]), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult ReduceCclBufferProducerReady(const OpParam &param, const AlgResourceCtx &resCtx, void *output,
    uint64_t sliceCount, uint64_t sliceSize, const FlatReadyTreePlan &plan)
{
    const uint32_t invalidThreadIdx = std::numeric_limits<uint32_t>::max();
    constexpr uint32_t readyNotifyIdx = 11;
    for (uint32_t pairIdx = 0; pairIdx < plan.pairOwners.size(); ++pairIdx) {
        const uint32_t signalThreadIdx = plan.signalThreads[pairIdx];
        if (signalThreadIdx == invalidThreadIdx) {
            continue;
        }
        const uint32_t ownerThreadIdx = plan.pairOwners[pairIdx];
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resCtx.threads[signalThreadIdx], resCtx.threads[ownerThreadIdx], readyNotifyIdx)));
    }
    for (uint32_t pairIdx = 0; pairIdx < plan.pairOwners.size(); ++pairIdx) {
        if (plan.signalThreads[pairIdx] == invalidThreadIdx) {
            continue;
        }
        const uint32_t ownerThreadIdx = plan.pairOwners[pairIdx];
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(resCtx.threads[ownerThreadIdx], readyNotifyIdx, CUSTOM_TIMEOUT)));
    }
    for (uint32_t pairIdx = 0; pairIdx < plan.pairOwners.size(); ++pairIdx) {
        const uint32_t dstRank = pairIdx * 2;
        const uint32_t srcRank = dstRank + 1;
        void *destination = AddOffset(resCtx.localBuffer.addr, static_cast<uint64_t>(dstRank) * sliceSize);
        void *source = AddOffset(resCtx.localBuffer.addr, static_cast<uint64_t>(srcRank) * sliceSize);
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(resCtx.threads[plan.pairOwners[pairIdx]],
            destination, source, sliceCount, static_cast<HcommDataType>(param.dataType),
            static_cast<HcommReduceOp>(param.reduceType))));
    }

    // Each owner already represents one fixed even/odd pair; keep the remaining p3 tree unchanged.
    constexpr uint32_t rankSize = 16;
    uint32_t notifyIdx = 12;
    for (uint32_t stride = 2; stride < rankSize; stride *= 2, ++notifyIdx) {
        for (uint32_t dstRank = 0; dstRank + stride < rankSize; dstRank += stride * 2) {
            const uint32_t srcRank = dstRank + stride;
            const uint32_t dstThreadIdx = plan.pairOwners[dstRank / 2];
            const uint32_t srcThreadIdx = plan.pairOwners[srcRank / 2];
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                resCtx.threads[srcThreadIdx], resCtx.threads[dstThreadIdx], notifyIdx)));
            CHK_RET(static_cast<HcclResult>(
                HcommThreadNotifyWaitOnThread(resCtx.threads[dstThreadIdx], notifyIdx, CUSTOM_TIMEOUT)));
            void *destination = AddOffset(resCtx.localBuffer.addr, static_cast<uint64_t>(dstRank) * sliceSize);
            void *source = AddOffset(resCtx.localBuffer.addr, static_cast<uint64_t>(srcRank) * sliceSize);
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(resCtx.threads[dstThreadIdx], destination,
                source, sliceCount, static_cast<HcommDataType>(param.dataType),
                static_cast<HcommReduceOp>(param.reduceType))));
        }
    }
    CHK_RET(static_cast<HcclResult>(
        HcommLocalCopyOnThread(resCtx.threads[0], output, resCtx.localBuffer.addr, sliceSize)));
    return HCCL_SUCCESS;
}

HcclResult ReduceCompactCclBufferProducerReady(const OpParam &param, const AlgResourceCtx &resCtx, void *output,
    void *localInput, uint64_t sliceCount, uint64_t sliceSize, const FlatReadyTreePlan &plan)
{
    const uint32_t invalidThreadIdx = std::numeric_limits<uint32_t>::max();
    constexpr uint32_t readyNotifyIdx = 11;
    for (uint32_t pairIdx = 0; pairIdx < plan.pairOwners.size(); ++pairIdx) {
        const uint32_t signalThreadIdx = plan.signalThreads[pairIdx];
        if (signalThreadIdx == invalidThreadIdx) {
            continue;
        }
        const uint32_t ownerThreadIdx = plan.pairOwners[pairIdx];
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resCtx.threads[signalThreadIdx], resCtx.threads[ownerThreadIdx], readyNotifyIdx)));
    }
    for (uint32_t pairIdx = 0; pairIdx < plan.pairOwners.size(); ++pairIdx) {
        if (plan.signalThreads[pairIdx] == invalidThreadIdx) {
            continue;
        }
        const uint32_t ownerThreadIdx = plan.pairOwners[pairIdx];
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(resCtx.threads[ownerThreadIdx], readyNotifyIdx, CUSTOM_TIMEOUT)));
    }

    // The self contribution has no scratch slot, so its pair reduces directly into the partner slot.
    for (uint32_t pairIdx = 0; pairIdx < plan.pairOwners.size(); ++pairIdx) {
        const uint32_t evenRank = pairIdx * 2;
        const uint32_t oddRank = evenRank + 1;
        void *destination = AddOffset(resCtx.localBuffer.addr,
            static_cast<uint64_t>(plan.pairResultSlots[pairIdx]) * sliceSize);
        void *source = localInput;
        if (evenRank != param.myRank && oddRank != param.myRank) {
            source = AddOffset(resCtx.localBuffer.addr,
                static_cast<uint64_t>(GetRemoteSlotIndex(oddRank, param.myRank)) * sliceSize);
        }
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(resCtx.threads[plan.pairOwners[pairIdx]],
            destination, source, sliceCount, static_cast<HcommDataType>(param.dataType),
            static_cast<HcommReduceOp>(param.reduceType))));
    }

    constexpr uint32_t rankSize = 16;
    uint32_t notifyIdx = 12;
    for (uint32_t stride = 2; stride < rankSize; stride *= 2, ++notifyIdx) {
        for (uint32_t dstRank = 0; dstRank + stride < rankSize; dstRank += stride * 2) {
            const uint32_t srcRank = dstRank + stride;
            const uint32_t dstPairIdx = dstRank / 2;
            const uint32_t srcPairIdx = srcRank / 2;
            const uint32_t dstThreadIdx = plan.pairOwners[dstPairIdx];
            const uint32_t srcThreadIdx = plan.pairOwners[srcPairIdx];
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                resCtx.threads[srcThreadIdx], resCtx.threads[dstThreadIdx], notifyIdx)));
            CHK_RET(static_cast<HcclResult>(
                HcommThreadNotifyWaitOnThread(resCtx.threads[dstThreadIdx], notifyIdx, CUSTOM_TIMEOUT)));
            void *destination = AddOffset(resCtx.localBuffer.addr,
                static_cast<uint64_t>(plan.pairResultSlots[dstPairIdx]) * sliceSize);
            void *source = AddOffset(resCtx.localBuffer.addr,
                static_cast<uint64_t>(plan.pairResultSlots[srcPairIdx]) * sliceSize);
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(resCtx.threads[dstThreadIdx], destination,
                source, sliceCount, static_cast<HcommDataType>(param.dataType),
                static_cast<HcommReduceOp>(param.reduceType))));
        }
    }
    void *result = AddOffset(resCtx.localBuffer.addr,
        static_cast<uint64_t>(plan.pairResultSlots[0]) * sliceSize);
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resCtx.threads[0], output, result, sliceSize)));
    return HCCL_SUCCESS;
}

HcclResult ExecFlat(const OpParam &param, const AlgResourceCtx &resCtx, uint32_t dataTypeSize,
    uint64_t minCclBufferSize)
{
    constexpr uint32_t optimizedRankSize = 16;
    constexpr uint32_t optimizedThreadNum = optimizedRankSize - 1;
    const bool supportsProducerReady = param.rankSize == optimizedRankSize &&
        resCtx.threads.size() == optimizedThreadNum && resCtx.channels.size() == optimizedThreadNum;
    const uint64_t outputBytes = param.count * dataTypeSize;
    const bool useCompactSlots = supportsProducerReady && outputBytes > minCclBufferSize;
    const uint32_t slotNum = useCompactSlots ? optimizedThreadNum : param.rankSize;
    const uint64_t maxSliceCount =
        custom_reducescatter::MaxSliceCount(minCclBufferSize, slotNum, dataTypeSize);
    CHK_PRT_RET(maxSliceCount == 0,
        HCCL_ERROR("CCL buffer[%llu] is too small for slotNum[%u]",
            static_cast<unsigned long long>(minCclBufferSize), slotNum), HCCL_E_MEMORY);

    uint64_t processedCount = 0;
    std::vector<uint32_t> workerIndices;
    for (uint32_t threadIdx = 1; threadIdx < resCtx.threads.size(); ++threadIdx) {
        workerIndices.push_back(threadIdx);
    }
    CHK_RET(ValidateWorkerIndices(resCtx.threads, workerIndices));
    FlatReadyTreePlan readyPlan;
    if (supportsProducerReady) {
        CHK_RET(BuildFlatReadyTreePlan(param, resCtx, readyPlan));
    }
    while (processedCount < param.count) {
        const uint64_t sliceCount = std::min(maxSliceCount, param.count - processedCount);
        const uint64_t sliceSize = sliceCount * dataTypeSize;
        const uint64_t localInputOffset = static_cast<uint64_t>(param.myRank) * param.count + processedCount;
        void *localInput = AddOffset(param.inputPtr, localInputOffset * dataTypeSize);
        const bool useProducerReady = supportsProducerReady &&
            sliceSize > custom_reducescatter::kSliceAlignment;
        if (useCompactSlots) {
            CHK_RET(SyncMainToWorkers(resCtx.threads, workerIndices));
        } else if (useProducerReady) {
            void *localSlot = AddOffset(resCtx.localBuffer.addr, static_cast<uint64_t>(param.myRank) * sliceSize);
            CHK_RET(SyncMainToWorkers(resCtx.threads, workerIndices));
            const uint32_t copyThreadIdx = readyPlan.pairOwners[param.myRank / 2];
            CHK_RET(static_cast<HcclResult>(
                HcommLocalCopyOnThread(resCtx.threads[copyThreadIdx], localSlot, localInput, sliceSize)));
        } else {
            void *localSlot = AddOffset(resCtx.localBuffer.addr, static_cast<uint64_t>(param.myRank) * sliceSize);
            CHK_RET(static_cast<HcclResult>(
                HcommLocalCopyOnThread(resCtx.threads[0], localSlot, localInput, sliceSize)));
            CHK_RET(SyncMainToWorkers(resCtx.threads, workerIndices));
        }

        for (uint32_t channelIdx = 0; channelIdx < resCtx.channels.size(); ++channelIdx) {
            const ChannelInfo &channel = resCtx.channels[channelIdx];
            const uint64_t inputCountOffset = static_cast<uint64_t>(channel.remoteRank) * param.count + processedCount;
            void *input = AddOffset(param.inputPtr, inputCountOffset * dataTypeSize);
            const uint32_t remoteSlot = useCompactSlots ?
                GetRemoteSlotIndex(param.myRank, channel.remoteRank) : param.myRank;
            void *remoteDst = AddOffset(channel.remoteCclMem.addr, static_cast<uint64_t>(remoteSlot) * sliceSize);
            CHK_RET(ExchangeOneSlice(resCtx.threads[channelIdx], channel, remoteDst, input, sliceSize));
        }
        void *output = AddOffset(param.outputPtr, processedCount * dataTypeSize);
        if (useCompactSlots) {
            CHK_RET(ReduceCompactCclBufferProducerReady(
                param, resCtx, output, localInput, sliceCount, sliceSize, readyPlan));
        } else if (useProducerReady) {
            CHK_RET(ReduceCclBufferProducerReady(param, resCtx, output, sliceCount, sliceSize, readyPlan));
        } else {
            CHK_RET(JoinWorkerThreads(resCtx.threads, workerIndices));
            CHK_RET(ReduceCclBufferTree(param, resCtx, output, sliceCount, sliceSize));
        }
        processedCount += sliceCount;
    }
    return HCCL_SUCCESS;
}

struct TwoServerChunk {
    uint64_t processedCount = 0;
    uint64_t sliceCount = 0;
    uint64_t sliceSize = 0;
    uint64_t bankOffset = 0;
    bool valid = false;
};

struct LocalChannelPlan {
    uint32_t channelIdx = 0;
    uint32_t destination0 = 0;
    uint32_t destination1 = 0;
};

HcclResult PrepareLocalPartials(const OpParam &param, const AlgResourceCtx &resCtx,
    const std::vector<uint32_t> &localRanks, const std::vector<uint32_t> &localWorkerIndices,
    const std::vector<LocalChannelPlan> &localChannelPlans, const TwoServerChunk &chunk, uint64_t slotStride,
    uint32_t dataTypeSize)
{
    const uint32_t localRankSize = static_cast<uint32_t>(localRanks.size());
    CHK_RET(SyncMainToWorkers(resCtx.threads, localWorkerIndices));
    for (const LocalChannelPlan &plan : localChannelPlans) {
        const ChannelInfo &channel = resCtx.channels[plan.channelIdx];
        void *input0 = AddOffset(param.inputPtr,
            (static_cast<uint64_t>(plan.destination0) * param.count + chunk.processedCount) * dataTypeSize);
        void *input1 = AddOffset(param.inputPtr,
            (static_cast<uint64_t>(plan.destination1) * param.count + chunk.processedCount) * dataTypeSize);
        void *remoteDst0 = GetSlotAddr(channel.remoteCclMem.addr, chunk.bankOffset,
            resCtx.localRankIndex, slotStride);
        void *remoteDst1 = GetSlotAddr(channel.remoteCclMem.addr, chunk.bankOffset,
            localRankSize + resCtx.localRankIndex, slotStride);
        CHK_RET(ExchangeTwoSlices(resCtx.threads[plan.channelIdx], channel, remoteDst0, input0, remoteDst1, input1,
            chunk.sliceSize));
    }
    CHK_RET(JoinWorkerThreads(resCtx.threads, localWorkerIndices));

    const uint32_t destination0 = resCtx.group0Ranks[resCtx.localRankIndex];
    const uint32_t destination1 = resCtx.group1Ranks[resCtx.localRankIndex];
    CHK_RET(ReduceLocalPartial(param, resCtx, localRanks, destination0, 0, chunk.processedCount, chunk.sliceCount,
        chunk.sliceSize, chunk.bankOffset, slotStride, dataTypeSize));
    CHK_RET(ReduceLocalPartial(param, resCtx, localRanks, destination1, localRankSize, chunk.processedCount,
        chunk.sliceCount, chunk.sliceSize, chunk.bankOffset, slotStride, dataTypeSize));
    return HCCL_SUCCESS;
}

HcclResult StartPeerExchange(const OpParam &param, const AlgResourceCtx &resCtx, uint32_t peerChannelIdx,
    const TwoServerChunk &chunk, uint64_t slotStride)
{
    const uint32_t localRankSize = static_cast<uint32_t>(resCtx.group0Ranks.size());
    const ChannelInfo &peerChannel = resCtx.channels[peerChannelIdx];
    ThreadHandle peerThread = resCtx.threads[peerChannelIdx];
    CHK_RET(SyncMainToWorker(resCtx.threads[0], peerThread));
    void *partial0 = GetSlotAddr(resCtx.localBuffer.addr, chunk.bankOffset,
        resCtx.localRankIndex, slotStride);
    void *partial1 = GetSlotAddr(resCtx.localBuffer.addr, chunk.bankOffset,
        localRankSize + resCtx.localRankIndex, slotStride);
    void *sendPartial = resCtx.localGroupIndex == 0 ? partial1 : partial0;
    const uint32_t resultSlot = resCtx.localGroupIndex == 0 ? localRankSize + resCtx.localRankIndex :
        resCtx.localRankIndex;
    void *remoteDst = GetSlotAddr(peerChannel.remoteCclMem.addr, chunk.bankOffset, resultSlot, slotStride);
    return ExchangeOneSliceReduce(peerThread, peerChannel, remoteDst, sendPartial, chunk.sliceCount,
        static_cast<HcommDataType>(param.dataType), static_cast<HcommReduceOp>(param.reduceType));
}

HcclResult FinishPeerExchange(const OpParam &param, const AlgResourceCtx &resCtx, uint32_t peerChannelIdx,
    const TwoServerChunk &chunk, uint64_t slotStride, uint32_t dataTypeSize)
{
    const uint32_t localRankSize = static_cast<uint32_t>(resCtx.group0Ranks.size());
    CHK_RET(JoinWorker(resCtx.threads, peerChannelIdx));
    void *partial0 = GetSlotAddr(resCtx.localBuffer.addr, chunk.bankOffset,
        resCtx.localRankIndex, slotStride);
    void *partial1 = GetSlotAddr(resCtx.localBuffer.addr, chunk.bankOffset,
        localRankSize + resCtx.localRankIndex, slotStride);
    void *output = AddOffset(param.outputPtr, chunk.processedCount * dataTypeSize);
    void *result = resCtx.localGroupIndex == 0 ? partial0 : partial1;
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resCtx.threads[0], output, result, chunk.sliceSize)));
    return HCCL_SUCCESS;
}

HcclResult ExecTwoServer(const OpParam &param, const AlgResourceCtx &resCtx, uint32_t dataTypeSize,
    uint64_t minCclBufferSize)
{
    const std::vector<uint32_t> &localRanks =
        resCtx.localGroupIndex == 0 ? resCtx.group0Ranks : resCtx.group1Ranks;
    const uint32_t localRankSize = static_cast<uint32_t>(localRanks.size());
    const uint32_t slotNum = localRankSize * 2;
    const uint64_t singleBankMaxSliceCount =
        custom_reducescatter::MaxSliceCount(minCclBufferSize, slotNum, dataTypeSize);
    CHK_PRT_RET(singleBankMaxSliceCount == 0,
        HCCL_ERROR("CCL buffer[%llu] is too small for hierarchy slotNum[%u]",
            static_cast<unsigned long long>(minCclBufferSize), slotNum), HCCL_E_MEMORY);

    const int32_t peerChannelIdx = FindChannelIndex(resCtx.channels, resCtx.peerRank);
    CHK_PRT_RET(peerChannelIdx < 0, HCCL_ERROR("Peer channel for rank[%u] is missing", resCtx.peerRank),
        HCCL_E_INTERNAL);
    const uint32_t peerThreadIdx = static_cast<uint32_t>(peerChannelIdx);
    std::vector<uint32_t> localWorkerIndices;
    std::vector<LocalChannelPlan> localChannelPlans;
    localWorkerIndices.reserve(localRankSize - 1);
    localChannelPlans.reserve(localRankSize - 1);
    uint32_t localChannelNum = 0;
    for (uint32_t channelIdx = 0; channelIdx < resCtx.channels.size(); ++channelIdx) {
        if (channelIdx == peerThreadIdx) {
            continue;
        }
        ++localChannelNum;
        if (channelIdx != 0) {
            localWorkerIndices.push_back(channelIdx);
        }
        const ChannelInfo &channel = resCtx.channels[channelIdx];
        const int32_t remoteLocalIndex = FindRankIndex(localRanks, channel.remoteRank);
        CHK_PRT_RET(remoteLocalIndex < 0,
            HCCL_ERROR("Local channel rank[%u] is outside the local group", channel.remoteRank), HCCL_E_INTERNAL);
        localChannelPlans.push_back(LocalChannelPlan{channelIdx, resCtx.group0Ranks[remoteLocalIndex],
            resCtx.group1Ranks[remoteLocalIndex]});
    }
    CHK_PRT_RET(localChannelNum != localRankSize - 1,
        HCCL_ERROR("Invalid local channel count[%u], expected[%u]", localChannelNum, localRankSize - 1),
        HCCL_E_INTERNAL);
    CHK_RET(ValidateWorkerIndices(resCtx.threads, localWorkerIndices));

    const uint64_t pipelineMaxSliceCount = custom_reducescatter::MaxSliceCount(
        minCclBufferSize, slotNum * PIPELINE_BANK_NUM, dataTypeSize);
    const uint64_t overflowCount =
        param.count > singleBankMaxSliceCount ? param.count - singleBankMaxSliceCount : 0;
    const uint64_t overflowBytes = overflowCount * dataTypeSize;
    const bool enablePipeline = pipelineMaxSliceCount > 0 && overflowBytes > PIPELINE_MIN_OVERFLOW_BYTES;
    const uint32_t bankNum = enablePipeline ? PIPELINE_BANK_NUM : 1;
    uint64_t maxSliceCount = singleBankMaxSliceCount;
    uint64_t pipelineChunkNum = 1;
    if (enablePipeline) {
        pipelineChunkNum = param.count / pipelineMaxSliceCount +
            static_cast<uint64_t>(param.count % pipelineMaxSliceCount != 0);
        const uint64_t averageSliceCount = param.count / pipelineChunkNum +
            static_cast<uint64_t>(param.count % pipelineChunkNum != 0);
        const uint64_t alignCount = std::max<uint64_t>(1, custom_reducescatter::kSliceAlignment / dataTypeSize);
        maxSliceCount = AlignUp(averageSliceCount, alignCount);
    }
    const uint64_t slotStride = maxSliceCount * dataTypeSize;
    const uint64_t bankStride = static_cast<uint64_t>(slotNum) * slotStride;
    HCCL_INFO("Two-server pipeline[%d], bankNum[%u], chunkNum[%llu], maxSliceCount[%llu], overflowBytes[%llu]",
        enablePipeline, bankNum, static_cast<unsigned long long>(pipelineChunkNum),
        static_cast<unsigned long long>(maxSliceCount), static_cast<unsigned long long>(overflowBytes));

    uint64_t processedCount = 0;
    uint32_t chunkIdx = 0;
    TwoServerChunk pendingChunk;
    while (processedCount < param.count) {
        const uint64_t sliceCount = std::min(maxSliceCount, param.count - processedCount);
        TwoServerChunk currentChunk{processedCount, sliceCount, sliceCount * dataTypeSize,
            static_cast<uint64_t>(chunkIdx % bankNum) * bankStride, true};
        if (!enablePipeline) {
            CHK_RET(PrepareLocalPartials(param, resCtx, localRanks, localWorkerIndices, localChannelPlans,
                currentChunk, slotStride, dataTypeSize));
            CHK_RET(StartPeerExchange(param, resCtx, peerThreadIdx, currentChunk, slotStride));
            CHK_RET(FinishPeerExchange(param, resCtx, peerThreadIdx, currentChunk, slotStride, dataTypeSize));
        } else {
            CHK_RET(PrepareLocalPartials(param, resCtx, localRanks, localWorkerIndices, localChannelPlans,
                currentChunk, slotStride, dataTypeSize));
            if (pendingChunk.valid) {
                CHK_RET(FinishPeerExchange(param, resCtx, peerThreadIdx, pendingChunk, slotStride, dataTypeSize));
            }
            CHK_RET(StartPeerExchange(param, resCtx, peerThreadIdx, currentChunk, slotStride));
            pendingChunk = currentChunk;
        }
        processedCount += sliceCount;
        ++chunkIdx;
    }
    if (enablePipeline && pendingChunk.valid) {
        CHK_RET(FinishPeerExchange(param, resCtx, peerThreadIdx, pendingChunk, slotStride, dataTypeSize));
    }
    return HCCL_SUCCESS;
}
} // namespace

namespace ops_hccl {
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    HCCL_INFO("Executing AICPU Kernel on Ascend NPU");

    const auto dataTypeIter = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(dataTypeIter == SIZE_TABLE.end(), HCCL_ERROR("Unsupported data type[%d]", param.dataType),
        HCCL_E_NOT_SUPPORT);
    const uint32_t dataTypeSize = dataTypeIter->second;
    CHK_RET(ValidateResources(param, resCtx, dataTypeSize));

    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    if (param.rankSize == 1) {
        return static_cast<HcclResult>(
            HcommLocalCopyOnThread(resCtx.threads[0], param.outputPtr, param.inputPtr, param.count * dataTypeSize));
    }

    uint64_t minCclBufferSize = resCtx.localBuffer.size;
    for (const ChannelInfo &channel : resCtx.channels) {
        CHK_PRT_RET(channel.remoteRank >= param.rankSize || channel.remoteCclMem.addr == nullptr,
            HCCL_ERROR("Invalid channel for remote rank[%u]", channel.remoteRank), HCCL_E_INTERNAL);
        minCclBufferSize = std::min(minCclBufferSize, channel.remoteCclMem.size);
    }

    if (resCtx.hierarchyMode == custom_reducescatter::kHierarchySmallNhr) {
        return ExecSmallNhr(param, resCtx, dataTypeSize, minCclBufferSize);
    }
    if (resCtx.hierarchyMode == custom_reducescatter::kHierarchyTwoServer) {
        return ExecTwoServer(param, resCtx, dataTypeSize, minCclBufferSize);
    }
    return ExecFlat(param, resCtx, dataTypeSize, minCclBufferSize);
}
} // namespace ops_hccl
