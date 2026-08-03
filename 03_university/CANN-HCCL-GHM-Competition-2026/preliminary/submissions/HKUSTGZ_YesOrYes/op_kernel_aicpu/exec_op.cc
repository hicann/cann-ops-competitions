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
#include <array>
#include <cstdint>

namespace ops_hccl {
namespace {
    constexpr uint64_t HCCL_MIN_SLICE_ALIGN = 128;
    constexpr uint64_t MAX_TRANSFER_BYTES = 256ULL * 1024ULL * 1024ULL;
    constexpr uint32_t PIPELINE_BANK_COUNT = 2;
    constexpr uint32_t PARALLEL_REDUCE_THREAD_NUM = CUSTOM_PARALLEL_REDUCE_EXTRA_THREAD_NUM + 1;
    constexpr uint32_t NOTIFY_NUM_PER_BANK = PARALLEL_REDUCE_THREAD_NUM + 1;
    constexpr uint32_t COMPETITION_RANK_COUNT = 2 * CUSTOM_COMPETITION_GROUP_RANK_NUM;

    struct ReduceStripe {
        ThreadHandle thread;
        uint64_t offsetCount;
        uint64_t count;
    };

    void *OffsetAddress(void *base, uint64_t offset)
    {
        return static_cast<void *>(static_cast<uint8_t *>(base) + offset);
    }

    uint32_t RemoteSlotIndex(uint32_t sourceRank, uint32_t receiverRank)
    {
        return sourceRank < receiverRank ? sourceRank : sourceRank - 1;
    }

    bool IsSmallRecursiveHalving(const OpParam &param)
    {
        return param.rankSize == COMPETITION_RANK_COUNT
            && param.count == CUSTOM_COMPETITION_512KB_RECV_COUNT;
    }

    bool IsRollingStripe512MB(const OpParam &param)
    {
        return param.rankSize == COMPETITION_RANK_COUNT
            && param.count == CUSTOM_COMPETITION_512MB_RECV_COUNT;
    }

    bool IsLargeTreePipeline(const OpParam &param)
    {
        return param.rankSize == COMPETITION_RANK_COUNT
            && param.count > CUSTOM_LARGE_MESSAGE_THRESHOLD_BYTES / sizeof(float);
    }

    uint32_t TreeDataNotifyIndex(uint32_t bankIndex)
    {
        return bankIndex;
    }

    uint32_t TreeAckNotifyIndex(uint32_t bankIndex)
    {
        return PIPELINE_BANK_COUNT + bankIndex;
    }

    uint32_t TreeLevelNotifyIndex(uint32_t bankIndex, uint32_t level)
    {
        // A thread notify is a state rather than an accumulating counter.
        // Two chunks are submitted before the first reduction completes, so
        // every in-flight bank needs an independent three-level tree signal.
        return 1 + bankIndex * 3 + level;
    }

    uint32_t TreeDoneNotifyIndex(uint32_t bankIndex)
    {
        // Main-thread notify 0 is reserved for the Host/AICPU handshake.
        return 1 + bankIndex;
    }

    uint32_t AckNotifyIndex(uint32_t bankIndex)
    {
        return bankIndex * NOTIFY_NUM_PER_BANK;
    }

    uint32_t DataNotifyIndex(uint32_t bankIndex, uint32_t stripeIndex)
    {
        return AckNotifyIndex(bankIndex) + 1 + stripeIndex;
    }

    uint32_t StripeDoneNotifyIndex(uint32_t bankIndex, uint32_t stripeIndex)
    {
        // Index 0 is reserved for the start barrier. Give each in-flight bank
        // a disjoint completion notify so two fast reduction workers cannot
        // publish consecutive records before the main thread consumes the
        // first one. This keeps the protocol correct for non-counting notifies.
        return 1 + bankIndex * (PARALLEL_REDUCE_THREAD_NUM - 1) + (stripeIndex - 1);
    }

    uint64_t BankSlotOffset(
        uint32_t bankIndex, uint32_t slotIndex, uint64_t slotStride, uint64_t remoteRankCount)
    {
        return (static_cast<uint64_t>(bankIndex) * remoteRankCount + slotIndex) * slotStride;
    }

    using ReduceStripeArray = std::array<ReduceStripe, PARALLEL_REDUCE_THREAD_NUM>;

    HcclResult BuildReduceStripes(
        const OpParam &param, const AlgResourceCtx &resCtx, uint64_t chunkCount, ReduceStripeArray &stripes)
    {
        constexpr uint64_t dataTypeSize = sizeof(float);
        const uint64_t reduceAlignCount = HCCL_MIN_SLICE_ALIGN / dataTypeSize;
        const uint64_t commonStripeCount
            = chunkCount / PARALLEL_REDUCE_THREAD_NUM / reduceAlignCount * reduceAlignCount;
        CHK_PRT_RET(commonStripeCount == 0,
            HCCL_ERROR("Parallel reduction chunk is too small, count %llu",
                static_cast<unsigned long long>(chunkCount)),
            HCCL_E_INTERNAL);

        for (uint32_t stripeIdx = 0; stripeIdx < PARALLEL_REDUCE_THREAD_NUM; ++stripeIdx) {
            const uint64_t offsetCount = static_cast<uint64_t>(stripeIdx) * commonStripeCount;
            const uint64_t stripeCount = stripeIdx + 1 == PARALLEL_REDUCE_THREAD_NUM
                ? chunkCount - offsetCount
                : commonStripeCount;
            const uint32_t threadIdx = stripeIdx == 0 ? 0 : param.rankSize + stripeIdx - 1;
            CHK_PRT_RET(threadIdx >= resCtx.threads.size() || stripeCount == 0,
                HCCL_ERROR("Invalid reduction stripe %u, thread %u/%zu, count %llu",
                    stripeIdx, threadIdx, resCtx.threads.size(),
                    static_cast<unsigned long long>(stripeCount)),
                HCCL_E_INTERNAL);
            stripes[stripeIdx] = ReduceStripe{resCtx.threads[threadIdx], offsetCount, stripeCount};
        }
        return HCCL_SUCCESS;
    }

    HcclResult ThreadSyncBefore(
        const FixedResourceList<ThreadHandle, CUSTOM_MAX_THREAD_NUM> &threads, uint32_t threadCount)
    {
        CHK_PRT_RET(threadCount == 0 || threadCount > threads.size(),
            HCCL_ERROR("[ThreadSyncBefore] Invalid thread count %u/%zu", threadCount, threads.size()), HCCL_E_PARA);
        for (uint32_t threadIdx = 1; threadIdx < threadCount; ++threadIdx) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[0], threads[threadIdx], 0)));
        }
        for (uint32_t threadIdx = 1; threadIdx < threadCount; ++threadIdx) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[threadIdx], 0, CUSTOM_TIMEOUT)));
        }
        return HCCL_SUCCESS;
    }

    HcclResult ThreadSyncAfter(
        const FixedResourceList<ThreadHandle, CUSTOM_MAX_THREAD_NUM> &threads, uint32_t threadCount)
    {
        CHK_PRT_RET(threadCount == 0 || threadCount > threads.size(),
            HCCL_ERROR("[ThreadSyncAfter] Invalid thread count %u/%zu", threadCount, threads.size()), HCCL_E_PARA);
        // notify 0 on the main thread is reserved for Host/Device synchronization.
        for (uint32_t threadIdx = 1; threadIdx < threadCount; ++threadIdx) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[0], threadIdx, CUSTOM_TIMEOUT)));
        }
        for (uint32_t threadIdx = 1; threadIdx < threadCount; ++threadIdx) {
            CHK_RET(
                static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[threadIdx], threads[0], threadIdx)));
        }
        return HCCL_SUCCESS;
    }

    HcclResult ExecSmallRecursiveHalving(const OpParam &param, const AlgResourceCtx &resCtx)
    {
        constexpr uint64_t dataTypeSize = sizeof(float);
        const uint64_t sliceBytes = param.count * dataTypeSize;
        const uint64_t requiredBufferBytes = static_cast<uint64_t>(param.rankSize) * sliceBytes;
        CHK_PRT_RET(resCtx.localBuffer.size < requiredBufferBytes,
            HCCL_ERROR("[ExecSmallRecursiveHalving] HCCL buffer %llu is smaller than required %llu",
                static_cast<unsigned long long>(resCtx.localBuffer.size),
                static_cast<unsigned long long>(requiredBufferBytes)),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(resCtx.channels.size() != CUSTOM_SMALL_RH_STEP_NUM,
            HCCL_ERROR("[ExecSmallRecursiveHalving] Expected %u channels, got %zu",
                CUSTOM_SMALL_RH_STEP_NUM, resCtx.channels.size()),
            HCCL_E_INTERNAL);

        ThreadHandle thread = resCtx.threads[0];
        uint8_t *scratch = static_cast<uint8_t *>(resCtx.localBuffer.addr);
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(thread, scratch, param.inputPtr, requiredBufferBytes)));

        for (uint32_t step = 0; step < CUSTOM_SMALL_RH_STEP_NUM; ++step) {
            const uint32_t delta = CUSTOM_SMALL_RH_PARTNER_DELTAS[step];
            const uint32_t partner = param.myRank ^ delta;
            const ChannelInfo &channel = resCtx.channels[step];
            CHK_PRT_RET(channel.remoteRank != partner
                    || channel.remoteCclMem.addr == nullptr
                    || channel.remoteCclMem.size < requiredBufferBytes,
                HCCL_ERROR("[ExecSmallRecursiveHalving] Invalid step %u channel for rank %u",
                    step, partner),
                HCCL_E_INTERNAL);

            const uint32_t blockStart = (param.myRank / (2U * delta)) * (2U * delta);
            const uint32_t readSliceStart =
                (param.myRank & delta) == 0U ? blockStart : blockStart + delta;
            const uint64_t transferCount = static_cast<uint64_t>(delta) * param.count;

            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(thread, channel.handle, 1)));
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyWaitOnThread(thread, channel.handle, 1, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(HcommReadReduceOnThread(thread, channel.handle,
                OffsetAddress(scratch, static_cast<uint64_t>(readSliceStart) * sliceBytes),
                OffsetAddress(channel.remoteCclMem.addr, static_cast<uint64_t>(readSliceStart) * sliceBytes),
                transferCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
        }

        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
            thread, param.outputPtr,
            OffsetAddress(scratch, static_cast<uint64_t>(param.myRank) * sliceBytes),
            sliceBytes)));
        return HCCL_SUCCESS;
    }

    HcclResult BuildLargeTreeChannelOrder(const OpParam &param, const AlgResourceCtx &resCtx,
        std::array<uint32_t, COMPETITION_RANK_COUNT - 1> &channelOrder)
    {
        uint32_t orderSize = 0;
        const uint32_t localGroup = param.myRank / CUSTOM_COMPETITION_GROUP_RANK_NUM;
        // Interleave one high-bandwidth Clos peer and one local full-mesh peer
        // with the same device slot. All channels own independent queues, but
        // this order avoids submitting either fabric as one long host-side run.
        for (uint32_t slot = 0; slot < CUSTOM_COMPETITION_GROUP_RANK_NUM; ++slot) {
            for (uint32_t channelIdx = 0; channelIdx < resCtx.channels.size(); ++channelIdx) {
                const ChannelInfo &channel = resCtx.channels[channelIdx];
                const bool crossGroup =
                    channel.remoteRank / CUSTOM_COMPETITION_GROUP_RANK_NUM != localGroup;
                if (crossGroup
                    && channel.remoteRank % CUSTOM_COMPETITION_GROUP_RANK_NUM == slot) {
                    channelOrder[orderSize++] = channelIdx;
                }
            }
            for (uint32_t channelIdx = 0; channelIdx < resCtx.channels.size(); ++channelIdx) {
                const ChannelInfo &channel = resCtx.channels[channelIdx];
                const bool sameGroup =
                    channel.remoteRank / CUSTOM_COMPETITION_GROUP_RANK_NUM == localGroup;
                if (sameGroup && channel.remoteRank != param.myRank
                    && channel.remoteRank % CUSTOM_COMPETITION_GROUP_RANK_NUM == slot) {
                    channelOrder[orderSize++] = channelIdx;
                }
            }
        }
        CHK_PRT_RET(orderSize != COMPETITION_RANK_COUNT - 1,
            HCCL_ERROR("[BuildLargeTreeChannelOrder] Expected %u peers, got %u",
                COMPETITION_RANK_COUNT - 1, orderSize),
            HCCL_E_INTERNAL);
        return HCCL_SUCCESS;
    }

    HcclResult SubmitLargeTreeChunk(const OpParam &param, const AlgResourceCtx &resCtx,
        const std::array<uint32_t, COMPETITION_RANK_COUNT - 1> &channelOrder,
        uint64_t processedCount, uint64_t chunkCount, uint32_t bankIndex,
        uint64_t slotStride, bool waitForReuse)
    {
        constexpr uint64_t dataTypeSize = sizeof(float);
        constexpr uint64_t remoteRankCount = COMPETITION_RANK_COUNT - 1;
        const uint64_t chunkBytes = chunkCount * dataTypeSize;
        for (uint32_t orderIdx = 0; orderIdx < channelOrder.size(); ++orderIdx) {
            const uint32_t channelIdx = channelOrder[orderIdx];
            const ChannelInfo &channel = resCtx.channels[channelIdx];
            const ThreadHandle sendThread = resCtx.threads[channelIdx + 1];
            if (waitForReuse) {
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                    sendThread, channel.handle, TreeAckNotifyIndex(bankIndex), CUSTOM_TIMEOUT)));
            }

            const uint64_t inputOffset =
                (static_cast<uint64_t>(channel.remoteRank) * param.count + processedCount) * dataTypeSize;
            const void *source = OffsetAddress(param.inputPtr, inputOffset);
            const uint32_t remoteSlot = RemoteSlotIndex(param.myRank, channel.remoteRank);
            void *remoteDestination = OffsetAddress(channel.remoteCclMem.addr,
                BankSlotOffset(bankIndex, remoteSlot, slotStride, remoteRankCount));
            CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(
                sendThread, channel.handle, remoteDestination, source, chunkBytes)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                sendThread, channel.handle, TreeDataNotifyIndex(bankIndex))));
        }
        return HCCL_SUCCESS;
    }

    HcclResult ReduceLargeTreeChunk(const OpParam &param, const AlgResourceCtx &resCtx,
        const std::array<const ChannelInfo *, COMPETITION_RANK_COUNT> &channelByRank,
        uint64_t processedCount, uint64_t chunkCount, uint32_t bankIndex,
        uint64_t slotStride)
    {
        constexpr uint64_t dataTypeSize = sizeof(float);
        constexpr uint32_t pairCount = COMPETITION_RANK_COUNT / 2;
        constexpr uint32_t treeThreadBase = COMPETITION_RANK_COUNT;
        constexpr uint64_t remoteRankCount = COMPETITION_RANK_COUNT - 1;
        const HcommDataType dataType = static_cast<HcommDataType>(param.dataType);
        const HcommReduceOp reduceOp = static_cast<HcommReduceOp>(param.reduceType);
        ThreadHandle mainThread = resCtx.threads[0];
        void *output = OffsetAddress(param.outputPtr, processedCount * dataTypeSize);

        std::array<void *, COMPETITION_RANK_COUNT> sourceBuffers{};
        for (uint32_t sourceRank = 0; sourceRank < COMPETITION_RANK_COUNT; ++sourceRank) {
            if (sourceRank == param.myRank) {
                const uint64_t ownOffset =
                    (static_cast<uint64_t>(param.myRank) * param.count + processedCount) * dataTypeSize;
                sourceBuffers[sourceRank] = OffsetAddress(param.inputPtr, ownOffset);
            } else {
                const uint32_t slotIndex = RemoteSlotIndex(sourceRank, param.myRank);
                sourceBuffers[sourceRank] = OffsetAddress(resCtx.localBuffer.addr,
                    BankSlotOffset(bankIndex, slotIndex, slotStride, remoteRankCount));
            }
        }

        std::array<void *, pairCount> accumulators{};
        for (uint32_t pairIdx = 0; pairIdx < pairCount; ++pairIdx) {
            const uint32_t leftRank = 2 * pairIdx;
            const uint32_t rightRank = leftRank + 1;
            const ThreadHandle worker = resCtx.threads[treeThreadBase + pairIdx];
            if (leftRank != param.myRank) {
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                    worker, channelByRank[leftRank]->handle,
                    TreeDataNotifyIndex(bankIndex), CUSTOM_TIMEOUT)));
            }
            if (rightRank != param.myRank) {
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                    worker, channelByRank[rightRank]->handle,
                    TreeDataNotifyIndex(bankIndex), CUSTOM_TIMEOUT)));
            }

            if (pairIdx == 0) {
                CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
                    worker, output, sourceBuffers[leftRank], chunkCount * dataTypeSize)));
                CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
                    worker, output, sourceBuffers[rightRank], chunkCount, dataType, reduceOp)));
                accumulators[pairIdx] = output;
            } else {
                // Exactly one original input contribution can be read-only.
                // Use a received CCL slot as the in-place tree accumulator.
                const bool leftWritable = leftRank != param.myRank;
                void *destination = leftWritable ? sourceBuffers[leftRank] : sourceBuffers[rightRank];
                const void *source = leftWritable ? sourceBuffers[rightRank] : sourceBuffers[leftRank];
                CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
                    worker, destination, source, chunkCount, dataType, reduceOp)));
                accumulators[pairIdx] = destination;
            }
        }

        // Direct worker-to-worker edges preserve arrival-driven execution:
        // 8 independent pairs collapse to 4, then 2, then the output root.
        for (uint32_t groupIdx = 0; groupIdx < 4; ++groupIdx) {
            const uint32_t leftPair = 2 * groupIdx;
            const uint32_t rightPair = leftPair + 1;
            const ThreadHandle destinationWorker = resCtx.threads[treeThreadBase + leftPair];
            const ThreadHandle donorWorker = resCtx.threads[treeThreadBase + rightPair];
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                donorWorker, destinationWorker, TreeLevelNotifyIndex(bankIndex, 0))));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                destinationWorker, TreeLevelNotifyIndex(bankIndex, 0), CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
                destinationWorker, accumulators[leftPair], accumulators[rightPair],
                chunkCount, dataType, reduceOp)));
        }
        for (uint32_t groupIdx = 0; groupIdx < 2; ++groupIdx) {
            const uint32_t leftPair = 4 * groupIdx;
            const uint32_t rightPair = leftPair + 2;
            const ThreadHandle destinationWorker = resCtx.threads[treeThreadBase + leftPair];
            const ThreadHandle donorWorker = resCtx.threads[treeThreadBase + rightPair];
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                donorWorker, destinationWorker, TreeLevelNotifyIndex(bankIndex, 1))));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                destinationWorker, TreeLevelNotifyIndex(bankIndex, 1), CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
                destinationWorker, accumulators[leftPair], accumulators[rightPair],
                chunkCount, dataType, reduceOp)));
        }
        const ThreadHandle rootWorker = resCtx.threads[treeThreadBase];
        const ThreadHandle finalDonorWorker = resCtx.threads[treeThreadBase + 4];
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            finalDonorWorker, rootWorker, TreeLevelNotifyIndex(bankIndex, 2))));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(
                rootWorker, TreeLevelNotifyIndex(bankIndex, 2), CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
            rootWorker, accumulators[0], accumulators[4], chunkCount, dataType, reduceOp)));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(
                rootWorker, mainThread, TreeDoneNotifyIndex(bankIndex))));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(
                mainThread, TreeDoneNotifyIndex(bankIndex), CUSTOM_TIMEOUT)));

        // The complete tree has consumed every receive slot in this bank.
        for (const ChannelInfo &channel : resCtx.channels) {
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                mainThread, channel.handle, TreeAckNotifyIndex(bankIndex))));
        }
        return HCCL_SUCCESS;
    }

    HcclResult ExecLargeTreePipeline(const OpParam &param, const AlgResourceCtx &resCtx,
        const std::array<const ChannelInfo *, COMPETITION_RANK_COUNT> &channelByRank)
    {
        constexpr uint64_t dataTypeSize = sizeof(float);
        constexpr uint64_t remoteRankCount = COMPETITION_RANK_COUNT - 1;
        uint64_t commonBufferSize = resCtx.localBuffer.size;
        for (const ChannelInfo &channel : resCtx.channels) {
            commonBufferSize = std::min(commonBufferSize, channel.remoteCclMem.size);
        }
        const uint64_t slotStride = commonBufferSize / (remoteRankCount * PIPELINE_BANK_COUNT)
            / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
        const uint64_t maxChunkCount = std::min(slotStride, MAX_TRANSFER_BYTES) / dataTypeSize;
        CHK_PRT_RET(maxChunkCount == 0,
            HCCL_ERROR("[ExecLargeTreePipeline] CCL buffer is too small: %llu",
                static_cast<unsigned long long>(commonBufferSize)),
            HCCL_E_INTERNAL);

        const uint64_t loopCount =
            param.count / maxChunkCount + (param.count % maxChunkCount != 0);
        const uint64_t balancedChunkCount =
            param.count / loopCount + (param.count % loopCount != 0);
        CHK_PRT_RET(loopCount == 0,
            HCCL_ERROR("[ExecLargeTreePipeline] Invalid loop count"), HCCL_E_INTERNAL);

        std::array<uint32_t, COMPETITION_RANK_COUNT - 1> channelOrder{};
        CHK_RET(BuildLargeTreeChannelOrder(param, resCtx, channelOrder));
        CHK_RET(ThreadSyncBefore(resCtx.threads, CUSTOM_LARGE_TREE_THREAD_NUM));

        const uint64_t firstChunkCount = std::min(balancedChunkCount, param.count);
        CHK_RET(SubmitLargeTreeChunk(
            param, resCtx, channelOrder, 0, firstChunkCount, 0, slotStride, false));
        if (loopCount > 1) {
            const uint64_t secondOffset = balancedChunkCount;
            const uint64_t secondCount =
                std::min(balancedChunkCount, param.count - secondOffset);
            CHK_RET(SubmitLargeTreeChunk(
                param, resCtx, channelOrder, secondOffset, secondCount, 1, slotStride, false));
        }

        for (uint64_t loopIndex = 0; loopIndex < loopCount; ++loopIndex) {
            const uint64_t processedCount = loopIndex * balancedChunkCount;
            const uint64_t chunkCount =
                std::min(balancedChunkCount, param.count - processedCount);
            const uint32_t bankIndex = loopIndex % PIPELINE_BANK_COUNT;
            CHK_RET(ReduceLargeTreeChunk(
                param, resCtx, channelByRank, processedCount, chunkCount, bankIndex, slotStride));

            const uint64_t nextLoopIndex = loopIndex + PIPELINE_BANK_COUNT;
            if (nextLoopIndex < loopCount) {
                const uint64_t nextOffset = nextLoopIndex * balancedChunkCount;
                const uint64_t nextCount =
                    std::min(balancedChunkCount, param.count - nextOffset);
                CHK_RET(SubmitLargeTreeChunk(
                    param, resCtx, channelOrder, nextOffset, nextCount, bankIndex,
                    slotStride, true));
            }
        }

        // Wait for the final outbound use of every bank. Reuse waits consumed
        // older ACK generations on the dedicated send queues.
        const uint32_t usedBankCount =
            loopCount < PIPELINE_BANK_COUNT ? static_cast<uint32_t>(loopCount) : PIPELINE_BANK_COUNT;
        for (uint32_t bankIndex = 0; bankIndex < usedBankCount; ++bankIndex) {
            for (const ChannelInfo &channel : resCtx.channels) {
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                    resCtx.threads[0], channel.handle,
                    TreeAckNotifyIndex(bankIndex), CUSTOM_TIMEOUT)));
            }
        }
        return HCCL_SUCCESS;
    }

    uint32_t RollingDataNotifyIndex(uint32_t stripeIdx)
    {
        return stripeIdx;
    }

    uint32_t RollingReuseAckNotifyIndex(uint32_t stripeIdx)
    {
        return CUSTOM_ROLLING_STRIPE_NUM + stripeIdx;
    }

    constexpr uint32_t ROLLING_FINAL_ACK_NOTIFY_INDEX = 2 * CUSTOM_ROLLING_STRIPE_NUM;

    HcclResult ExecRollingStripe512MB(const OpParam &param, const AlgResourceCtx &resCtx,
        const std::array<const ChannelInfo *, COMPETITION_RANK_COUNT> &channelByRank)
    {
        constexpr uint64_t dataTypeSize = sizeof(float);
        constexpr uint32_t remoteRankCount = COMPETITION_RANK_COUNT - 1;
        constexpr uint32_t activeThreadCount
            = COMPETITION_RANK_COUNT - 1 + CUSTOM_ROLLING_STRIPE_NUM;
        const uint64_t chunkCount = param.count / 2;
        const uint64_t stripeCount = chunkCount / CUSTOM_ROLLING_STRIPE_NUM;
        const uint64_t stripeBytes = stripeCount * dataTypeSize;

        uint64_t commonBufferSize = resCtx.localBuffer.size;
        for (const ChannelInfo &channel : resCtx.channels) {
            commonBufferSize = std::min(commonBufferSize, channel.remoteCclMem.size);
        }
        const uint64_t slotStride = commonBufferSize / remoteRankCount
            / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
        CHK_PRT_RET(param.count % 2 != 0 || chunkCount % CUSTOM_ROLLING_STRIPE_NUM != 0
                || slotStride < chunkCount * dataTypeSize,
            HCCL_ERROR("[ExecRollingStripe512MB] Invalid count %llu or slot stride %llu",
                static_cast<unsigned long long>(param.count),
                static_cast<unsigned long long>(slotStride)),
            HCCL_E_INTERNAL);

        ThreadHandle mainThread = resCtx.threads[0];
        CHK_RET(ThreadSyncBefore(resCtx.threads, activeThreadCount));

        // Interleave Clos and intra-server channel setup so both fabrics begin
        // receiving work early. Each peer keeps one dedicated queue.
        std::array<uint32_t, remoteRankCount> channelOrder{};
        uint32_t orderSize = 0;
        const uint32_t localGroup = param.myRank / CUSTOM_COMPETITION_GROUP_RANK_NUM;
        for (uint32_t slot = 0; slot < CUSTOM_COMPETITION_GROUP_RANK_NUM; ++slot) {
            for (uint32_t channelIdx = 0; channelIdx < resCtx.channels.size(); ++channelIdx) {
                const ChannelInfo &channel = resCtx.channels[channelIdx];
                const bool crossGroup
                    = channel.remoteRank / CUSTOM_COMPETITION_GROUP_RANK_NUM != localGroup;
                const uint32_t localSlot = channel.remoteRank % CUSTOM_COMPETITION_GROUP_RANK_NUM;
                if (crossGroup && localSlot == slot) {
                    channelOrder[orderSize++] = channelIdx;
                }
            }
            for (uint32_t channelIdx = 0; channelIdx < resCtx.channels.size(); ++channelIdx) {
                const ChannelInfo &channel = resCtx.channels[channelIdx];
                const bool sameGroup
                    = channel.remoteRank / CUSTOM_COMPETITION_GROUP_RANK_NUM == localGroup;
                const uint32_t localSlot = channel.remoteRank % CUSTOM_COMPETITION_GROUP_RANK_NUM;
                if (sameGroup && localSlot == slot && channel.remoteRank != param.myRank) {
                    channelOrder[orderSize++] = channelIdx;
                }
            }
        }
        CHK_PRT_RET(orderSize != remoteRankCount,
            HCCL_ERROR("[ExecRollingStripe512MB] Invalid peer order size %u", orderSize), HCCL_E_INTERNAL);

        for (uint32_t orderIdx = 0; orderIdx < orderSize; ++orderIdx) {
            const uint32_t channelIdx = channelOrder[orderIdx];
            const ChannelInfo &channel = resCtx.channels[channelIdx];
            ThreadHandle sendThread = resCtx.threads[channelIdx + 1];
            const uint32_t remoteSlot = RemoteSlotIndex(param.myRank, channel.remoteRank);
            void *remoteBase = OffsetAddress(
                channel.remoteCclMem.addr, static_cast<uint64_t>(remoteSlot) * slotStride);

            const uint64_t firstChunkBase
                = static_cast<uint64_t>(channel.remoteRank) * param.count;

            // Chunk 0 — per‑stripe writes so the receiver can start
            // reducing stripe 0 while the sender still writes stripes 1‑7.
            for (uint32_t stripeIdx = 0; stripeIdx < CUSTOM_ROLLING_STRIPE_NUM; ++stripeIdx) {
                const uint64_t stripeElementOffset
                    = firstChunkBase + stripeIdx * stripeCount;
                void *stripeDst = OffsetAddress(
                    remoteBase, static_cast<uint64_t>(stripeIdx) * stripeBytes);
                CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(sendThread, channel.handle,
                    stripeDst,
                    OffsetAddress(param.inputPtr, stripeElementOffset * dataTypeSize),
                    stripeCount * dataTypeSize)));
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                    sendThread, channel.handle, RollingDataNotifyIndex(stripeIdx))));
            }

            const uint64_t secondChunkBase = firstChunkBase + chunkCount;

            // Chunk 1 — wait for each stripe’s reuse ACK individually
            // so that the write of stripe 0 already overlaps the
            // receiver’s still‑in‑progress chunk‑0 reduce of stripes 1‑7.
            for (uint32_t stripeIdx = 0; stripeIdx < CUSTOM_ROLLING_STRIPE_NUM; ++stripeIdx) {
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(sendThread,
                    channel.handle, RollingReuseAckNotifyIndex(stripeIdx), CUSTOM_TIMEOUT)));
                const uint64_t stripeElementOffset
                    = secondChunkBase + stripeIdx * stripeCount;
                void *stripeDst = OffsetAddress(
                    remoteBase, static_cast<uint64_t>(stripeIdx) * stripeBytes);
                CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(sendThread, channel.handle,
                    stripeDst,
                    OffsetAddress(param.inputPtr, stripeElementOffset * dataTypeSize),
                    stripeCount * dataTypeSize)));
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                    sendThread, channel.handle, RollingDataNotifyIndex(stripeIdx))));
            }
        }

        std::array<ThreadHandle, CUSTOM_ROLLING_STRIPE_NUM> reduceThreads{
            mainThread,
            resCtx.threads[param.rankSize],
            resCtx.threads[param.rankSize + 1],
            resCtx.threads[param.rankSize + 2],
            resCtx.threads[param.rankSize + 3],
            resCtx.threads[param.rankSize + 4],
            resCtx.threads[param.rankSize + 5],
            resCtx.threads[param.rankSize + 6]};
        for (uint32_t stripeIdx = 0; stripeIdx < CUSTOM_ROLLING_STRIPE_NUM; ++stripeIdx) {
            ThreadHandle reduceThread = reduceThreads[stripeIdx];
            for (uint32_t chunkIdx = 0; chunkIdx < 2; ++chunkIdx) {
                const uint64_t outputElementOffset
                    = chunkIdx * chunkCount + stripeIdx * stripeCount;
                void *output = OffsetAddress(param.outputPtr, outputElementOffset * dataTypeSize);
                const uint64_t ownInputOffset
                    = (static_cast<uint64_t>(param.myRank) * param.count + outputElementOffset) * dataTypeSize;
                CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(reduceThread, output,
                    OffsetAddress(param.inputPtr, ownInputOffset), stripeBytes)));
                for (uint32_t sourceRank = 0; sourceRank < param.rankSize; ++sourceRank) {
                    if (sourceRank == param.myRank) {
                        continue;
                    }
                    const ChannelInfo &channel = *channelByRank[sourceRank];
                    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(reduceThread,
                        channel.handle, RollingDataNotifyIndex(stripeIdx), CUSTOM_TIMEOUT)));
                    const uint32_t localSlot = RemoteSlotIndex(sourceRank, param.myRank);
                    const void *source = OffsetAddress(resCtx.localBuffer.addr,
                        static_cast<uint64_t>(localSlot) * slotStride + stripeIdx * stripeBytes);
                    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(reduceThread, output, source,
                        stripeCount, static_cast<HcommDataType>(param.dataType),
                        static_cast<HcommReduceOp>(param.reduceType))));
                    if (chunkIdx == 0) {
                        // Release one complete source slot as soon as this
                        // worker has consumed its stripe. The sender reuses the
                        // slot only after all four stripe ACKs, then transfers
                        // the second 16 MB chunk as one contiguous DMA.
                        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(reduceThread,
                            channel.handle, RollingReuseAckNotifyIndex(stripeIdx))));
                    }
                }
            }
            if (stripeIdx > 0) {
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                    reduceThread, mainThread, stripeIdx)));
            }
        }

        for (uint32_t stripeIdx = 1; stripeIdx < CUSTOM_ROLLING_STRIPE_NUM; ++stripeIdx) {
            CHK_RET(static_cast<HcclResult>(
                HcommThreadNotifyWaitOnThread(mainThread, stripeIdx, CUSTOM_TIMEOUT)));
        }
        for (const ChannelInfo &channel : resCtx.channels) {
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                mainThread, channel.handle, ROLLING_FINAL_ACK_NOTIFY_INDEX)));
        }
        for (const ChannelInfo &channel : resCtx.channels) {
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                mainThread, channel.handle, ROLLING_FINAL_ACK_NOTIFY_INDEX, CUSTOM_TIMEOUT)));
        }
        return HCCL_SUCCESS;
    }

    HcclResult ValidateResource(const OpParam &param, const AlgResourceCtx &resCtx)
    {
        const bool useSmallRecursiveHalving = IsSmallRecursiveHalving(param);
        const bool useRollingStripe512MB = IsRollingStripe512MB(param);
        const bool useLargeTreePipeline = IsLargeTreePipeline(param);
        const bool useParallelReduce
            = param.count > CUSTOM_LARGE_MESSAGE_THRESHOLD_BYTES / sizeof(float);
        const uint32_t expectedChannelNum = param.rankSize <= 1
            ? 0
            : (useSmallRecursiveHalving ? CUSTOM_SMALL_RH_STEP_NUM : param.rankSize - 1);
        const uint32_t smallWorkerNum = std::min(CUSTOM_SMALL_COMM_WORKER_NUM, expectedChannelNum);
        const uint32_t expectedThreadNum = param.rankSize <= 1
            ? 1
            : (useSmallRecursiveHalving
                ? CUSTOM_SMALL_RH_THREAD_NUM
                : (useRollingStripe512MB || useLargeTreePipeline
                    ? CUSTOM_LARGE_TREE_THREAD_NUM
                    : 1 + smallWorkerNum));
        CHK_PRT_RET(param.rankSize == 0 || param.myRank >= param.rankSize,
            HCCL_ERROR("[ValidateResource] Invalid rank %u/%u", param.myRank, param.rankSize), HCCL_E_PARA);
        CHK_PRT_RET(param.rankSize > COMPETITION_RANK_COUNT,
            HCCL_ERROR("[ValidateResource] Competition kernel supports at most %u ranks, got %u",
                COMPETITION_RANK_COUNT, param.rankSize), HCCL_E_NOT_SUPPORT);
        CHK_PRT_RET(resCtx.threads.size() != expectedThreadNum,
            HCCL_ERROR("[ValidateResource] Expected %u threads, got %zu", expectedThreadNum, resCtx.threads.size()),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(resCtx.channels.size() != expectedChannelNum,
            HCCL_ERROR("[ValidateResource] Expected %u channels, got %zu", expectedChannelNum, resCtx.channels.size()),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(resCtx.localBuffer.addr == nullptr || resCtx.localBuffer.size < sizeof(float),
            HCCL_ERROR("[ValidateResource] Invalid local HCCL buffer"), HCCL_E_INTERNAL);

        std::array<bool, COMPETITION_RANK_COUNT> rankSeen{};
        rankSeen[param.myRank] = true;
        for (const ChannelInfo &channel : resCtx.channels) {
            CHK_PRT_RET(channel.remoteRank >= param.rankSize || rankSeen[channel.remoteRank],
                HCCL_ERROR("[ValidateResource] Invalid or duplicate remote rank %u", channel.remoteRank),
                HCCL_E_INTERNAL);
            const uint32_t lastNotifyIndex = useSmallRecursiveHalving
                ? CUSTOM_SMALL_RH_CHANNEL_NOTIFY_NUM - 1
                : (useRollingStripe512MB
                    ? CUSTOM_ROLLING_CHANNEL_NOTIFY_NUM - 1
                    : (useLargeTreePipeline
                    ? CUSTOM_LARGE_TREE_CHANNEL_NOTIFY_NUM - 1
                    : (useParallelReduce
                        ? DataNotifyIndex(PIPELINE_BANK_COUNT - 1, PARALLEL_REDUCE_THREAD_NUM - 1)
                        : DataNotifyIndex(0, 0))));
            CHK_PRT_RET(channel.notifyNum <= lastNotifyIndex || channel.remoteCclMem.addr == nullptr
                            || channel.remoteCclMem.size < sizeof(float),
                HCCL_ERROR("[ValidateResource] Invalid channel resource for rank %u", channel.remoteRank),
                HCCL_E_INTERNAL);
            rankSeen[channel.remoteRank] = true;
        }
        return HCCL_SUCCESS;
    }
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    CHK_PRT_RET(param.dataType != HCCL_DATA_TYPE_FP32 || param.reduceType != HCCL_REDUCE_SUM,
        HCCL_ERROR("Only FP32 SUM is supported"), HCCL_E_NOT_SUPPORT);
    CHK_RET(ValidateResource(param, resCtx));

    constexpr uint64_t dataTypeSize = sizeof(float);
    ThreadHandle mainThread = resCtx.threads[0];
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }

    if (param.rankSize == 1) {
        const uint64_t dataBytes = param.count * dataTypeSize;
        CHK_RET(
            static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread, param.outputPtr, param.inputPtr, dataBytes)));
        return HCCL_SUCCESS;
    }

    std::array<const ChannelInfo *, COMPETITION_RANK_COUNT> channelByRank{};
    for (const ChannelInfo &channel : resCtx.channels) {
        channelByRank[channel.remoteRank] = &channel;
    }

    if (IsSmallRecursiveHalving(param)) {
        return ExecSmallRecursiveHalving(param, resCtx);
    }
    if (IsRollingStripe512MB(param)) {
        return ExecRollingStripe512MB(param, resCtx, channelByRank);
    }
    if (IsLargeTreePipeline(param)) {
        return ExecLargeTreePipeline(param, resCtx, channelByRank);
    }

    // The local contribution is copied directly to output, so the receive buffer
    // contains only remote-source slots. Large messages split every source slot
    // into two banks; while bank k is reduced, communication threads can fill
    // bank k+1 without waiting for the main reduction stream.
    uint64_t commonBufferSize = resCtx.localBuffer.size;
    for (const ChannelInfo &channel : resCtx.channels) {
        commonBufferSize = std::min(commonBufferSize, channel.remoteCclMem.size);
    }
    const uint64_t remoteRankCount = param.rankSize - 1;
    const bool useDoubleBuffer
        = param.count > CUSTOM_LARGE_MESSAGE_THRESHOLD_BYTES / dataTypeSize;
    const bool useParallelReduce = useDoubleBuffer;
    const uint32_t bankCount = useDoubleBuffer ? PIPELINE_BANK_COUNT : 1;
    const uint64_t slotStride = commonBufferSize / (remoteRankCount * bankCount)
        / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
    CHK_PRT_RET(slotStride < dataTypeSize,
        HCCL_ERROR("HCCL buffer is too small for %u ranks, size %llu", param.rankSize,
            static_cast<unsigned long long>(commonBufferSize)),
        HCCL_E_INTERNAL);

    const uint64_t maxChunkBytes = std::min(slotStride, MAX_TRANSFER_BYTES);
    const uint64_t maxChunkCount = maxChunkBytes / dataTypeSize;
    CHK_PRT_RET(maxChunkCount == 0, HCCL_ERROR("Computed chunk size is zero"), HCCL_E_INTERNAL);

    const uint64_t loopCount = param.count / maxChunkCount + (param.count % maxChunkCount != 0);
    // Equal-sized loops avoid a very small tail transfer for the 512 MB case.
    const uint64_t balancedChunkCount = param.count / loopCount + (param.count % loopCount != 0);
    const uint32_t sendWorkerCount = useParallelReduce
        ? static_cast<uint32_t>(remoteRankCount)
        : std::min(CUSTOM_SMALL_COMM_WORKER_NUM, static_cast<uint32_t>(remoteRankCount));
    const uint32_t sendThreadEnd = sendWorkerCount + 1;
    const uint32_t activeThreadEnd = useParallelReduce
        ? static_cast<uint32_t>(resCtx.threads.size())
        : sendThreadEnd;

    // On the competition's guarded 16-rank layout, start the eight cross-group
    // transfers first and bind them one-to-one to the eight compact workers.
    // The seven same-group transfers then reuse seven of those queues. Generic
    // rank layouts keep the original ascending channel order and round-robin
    // assignment.
    const bool useCompetitionSmallSchedule = !useParallelReduce
        && param.rankSize == 2 * CUSTOM_COMPETITION_GROUP_RANK_NUM
        && sendWorkerCount == CUSTOM_COMPETITION_GROUP_RANK_NUM;
    std::array<uint32_t, COMPETITION_RANK_COUNT - 1> sendChannelOrder{};
    uint32_t sendChannelOrderSize = 0;
    if (useCompetitionSmallSchedule) {
        const uint32_t localGroup = param.myRank / CUSTOM_COMPETITION_GROUP_RANK_NUM;
        for (uint32_t pass = 0; pass < 2; ++pass) {
            for (uint32_t channelIdx = 0; channelIdx < resCtx.channels.size(); ++channelIdx) {
                const uint32_t remoteGroup
                    = resCtx.channels[channelIdx].remoteRank / CUSTOM_COMPETITION_GROUP_RANK_NUM;
                const bool isCrossGroup = remoteGroup != localGroup;
                if (isCrossGroup == (pass == 0)) {
                    sendChannelOrder[sendChannelOrderSize++] = channelIdx;
                }
            }
        }
    } else {
        for (uint32_t channelIdx = 0; channelIdx < resCtx.channels.size(); ++channelIdx) {
            sendChannelOrder[sendChannelOrderSize++] = channelIdx;
        }
    }

    // The Checker and the AICPU+TS execution model require an explicit start
    // edge for every non-main stream. This is the minimum legal front barrier;
    // partially connected variants leave channel task queues unreachable.
    CHK_RET(ThreadSyncBefore(resCtx.threads, activeThreadEnd));

    uint64_t processedCount = 0;
    uint64_t loopIndex = 0;
    while (processedCount < param.count) {
        const uint64_t chunkCount = std::min(balancedChunkCount, param.count - processedCount);
        const uint64_t chunkBytes = chunkCount * dataTypeSize;
        const uint32_t bankIndex = useDoubleBuffer ? loopIndex % PIPELINE_BANK_COUNT : 0;

        const uint64_t ownInputOffset
            = (static_cast<uint64_t>(param.myRank) * param.count + processedCount) * dataTypeSize;
        void *output = OffsetAddress(param.outputPtr, processedCount * dataTypeSize);
        const void *ownInput = OffsetAddress(param.inputPtr, ownInputOffset);
        ReduceStripeArray reduceStripes{};
        if (!useParallelReduce) {
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread, output, ownInput, chunkBytes)));
        } else {
            CHK_RET(BuildReduceStripes(param, resCtx, chunkCount, reduceStripes));
        }

        // Every receive slot is unused when the operation starts, so the first
        // chunk can be written immediately. Later chunks wait until the receiver
        // has reduced the previous contents and released the slot.
        for (uint32_t orderIdx = 0; orderIdx < sendChannelOrderSize; ++orderIdx) {
            const uint32_t channelIdx = sendChannelOrder[orderIdx];
            const ChannelInfo &channel = resCtx.channels[channelIdx];
            // Small messages reuse a compact set of queues in round-robin
            // channel order. Each queue remains internally ordered, while the
            // independent queues retain enough peer-level communication overlap.
            const uint32_t sendWorkerIndex = useCompetitionSmallSchedule
                ? channel.remoteRank % CUSTOM_COMPETITION_GROUP_RANK_NUM
                : channelIdx % sendWorkerCount;
            ThreadHandle sendThread = resCtx.threads[sendWorkerIndex + 1];
            // Both banks are initially free. A bank is reused only after the
            // receiver has completed the reduction from two chunks earlier.
            if ((!useDoubleBuffer && loopIndex > 0)
                || (useDoubleBuffer && loopIndex >= PIPELINE_BANK_COUNT)) {
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyWaitOnThread(
                        sendThread, channel.handle, AckNotifyIndex(bankIndex), CUSTOM_TIMEOUT)));
            }

            const uint64_t inputOffset
                = (static_cast<uint64_t>(channel.remoteRank) * param.count + processedCount) * dataTypeSize;
            const void *source = OffsetAddress(param.inputPtr, inputOffset);
            const uint32_t remoteSlotIndex = RemoteSlotIndex(param.myRank, channel.remoteRank);
            void *remoteSlot = OffsetAddress(channel.remoteCclMem.addr,
                BankSlotOffset(bankIndex, remoteSlotIndex, slotStride, remoteRankCount));
            if (!useParallelReduce) {
                // Keep the transfer and DATA-ready signal as two stream-ordered
                // tasks. The competition runtime does not expose the fused
                // write-with-notify primitive, while the explicit pair preserves
                // the same write-before-notify dependency on this send thread.
                CHK_RET(static_cast<HcclResult>(
                    HcommWriteOnThread(sendThread, channel.handle, remoteSlot, source, chunkBytes)));
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                    sendThread, channel.handle, DataNotifyIndex(bankIndex, 0))));
            } else {
                // Transfer the complete peer chunk as one contiguous DMA. The
                // stripe-0 signal and the following stream-ordered signals publish
                // the same completed chunk to four reduction workers.
                // Compared with four independent transfers this preserves the
                // two-bank communication/reduction pipeline while cutting the
                // transport task count and improving large-transfer efficiency.
                CHK_RET(static_cast<HcclResult>(
                    HcommWriteOnThread(sendThread, channel.handle, remoteSlot, source, chunkBytes)));
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                    sendThread, channel.handle, DataNotifyIndex(bankIndex, 0))));
                for (uint32_t stripeIdx = 1; stripeIdx < reduceStripes.size(); ++stripeIdx) {
                    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                        sendThread, channel.handle, DataNotifyIndex(bankIndex, stripeIdx))));
                }
            }
        }

        // Small messages consume every source on the main stream immediately.
        // Large-message workers consume four disjoint stripes independently.
        // Peer-level communication and the alternate bank still overlap with
        // reduction of the completed contiguous chunk.
        const bool bankWillBeReused = useDoubleBuffer && loopIndex + PIPELINE_BANK_COUNT < loopCount;
        const bool isFinalDoubleBufferChunk = useDoubleBuffer && loopIndex + 1 == loopCount;
        if (!useParallelReduce) {
            for (uint32_t sourceRank = 0; sourceRank < param.rankSize; ++sourceRank) {
                if (sourceRank == param.myRank) {
                    continue;
                }
                const ChannelInfo &channel = *channelByRank[sourceRank];
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                    mainThread, channel.handle, DataNotifyIndex(bankIndex, 0), CUSTOM_TIMEOUT)));
                const uint32_t localSlotIndex = RemoteSlotIndex(sourceRank, param.myRank);
                const void *sourceSlot = OffsetAddress(resCtx.localBuffer.addr,
                    BankSlotOffset(bankIndex, localSlotIndex, slotStride, remoteRankCount));
                CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(mainThread, output, sourceSlot, chunkCount,
                    static_cast<HcommDataType>(param.dataType), static_cast<HcommReduceOp>(param.reduceType))));
            }
        } else {
            // Every stripe preserves the same ascending source-rank reduction
            // order. Each stripe waits on a distinct channel notification, so
            // the write-to-reduce dependency is direct and no local fan-out
            // synchronization cycle is introduced.
            for (uint32_t stripeIdx = 0; stripeIdx < reduceStripes.size(); ++stripeIdx) {
                const ReduceStripe &stripe = reduceStripes[stripeIdx];
                const uint64_t stripeOffsetBytes = stripe.offsetCount * dataTypeSize;
                void *stripeOutput = OffsetAddress(output, stripeOffsetBytes);
                CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(stripe.thread, stripeOutput,
                    OffsetAddress(const_cast<void *>(ownInput), stripeOffsetBytes),
                    stripe.count * dataTypeSize)));
                for (uint32_t sourceRank = 0; sourceRank < param.rankSize; ++sourceRank) {
                    if (sourceRank == param.myRank) {
                        continue;
                    }
                    const ChannelInfo &channel = *channelByRank[sourceRank];
                    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                        stripe.thread, channel.handle, DataNotifyIndex(bankIndex, stripeIdx), CUSTOM_TIMEOUT)));
                    const uint32_t localSlotIndex = RemoteSlotIndex(sourceRank, param.myRank);
                    const void *sourceStripe = OffsetAddress(resCtx.localBuffer.addr,
                        BankSlotOffset(bankIndex, localSlotIndex, slotStride, remoteRankCount) + stripeOffsetBytes);
                    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(stripe.thread,
                        stripeOutput, sourceStripe, stripe.count,
                        static_cast<HcommDataType>(param.dataType),
                        static_cast<HcommReduceOp>(param.reduceType))));
                }
            }

            // Acknowledge receive-bank reuse only after every stripe is complete.
            for (uint32_t stripeIdx = 1; stripeIdx < reduceStripes.size(); ++stripeIdx) {
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                    mainThread, StripeDoneNotifyIndex(bankIndex, stripeIdx), CUSTOM_TIMEOUT)));
            }
            for (uint32_t stripeIdx = 1; stripeIdx < reduceStripes.size(); ++stripeIdx) {
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                    reduceStripes[stripeIdx].thread, mainThread,
                    StripeDoneNotifyIndex(bankIndex, stripeIdx))));
            }
        }

        if ((!useDoubleBuffer && loopIndex + 1 < loopCount)
            || bankWillBeReused || isFinalDoubleBufferChunk) {
            for (uint32_t sourceRank = 0; sourceRank < param.rankSize; ++sourceRank) {
                if (sourceRank == param.myRank) {
                    continue;
                }
                const ChannelInfo &channel = *channelByRank[sourceRank];
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                    mainThread, channel.handle, AckNotifyIndex(bankIndex))));
            }
        }

        processedCount += chunkCount;
        ++loopIndex;
    }

    if (!useDoubleBuffer) {
        CHK_RET(ThreadSyncAfter(resCtx.threads, sendThreadEnd));
    } else {
        // Do not join a send stream locally after it has waited on a remote ACK:
        // the Checker resolves local notifications before cross-rank ones, so
        // that dependency order would be cyclic. Instead, every receiver ACKs
        // the final bank after reducing it. Waiting for all final ACKs here is a
        // direct cross-rank completion barrier and safely closes input lifetime.
        const uint32_t finalBankIndex = (loopCount - 1) % PIPELINE_BANK_COUNT;
        for (uint32_t sourceRank = 0; sourceRank < param.rankSize; ++sourceRank) {
            if (sourceRank == param.myRank) {
                continue;
            }
            const ChannelInfo &channel = *channelByRank[sourceRank];
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                mainThread, channel.handle, AckNotifyIndex(finalBankIndex), CUSTOM_TIMEOUT)));
        }
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
