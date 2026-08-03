/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>

#include "custom.h"
#include "log.h"
#include "exec_op.h"

namespace {
constexpr uint32_t COMPETITION_RANK_SIZE = 16;
constexpr uint32_t REMOTE_PEER_COUNT = COMPETITION_RANK_SIZE - 1;
constexpr uint32_t CHANNEL_NOTIFY_READY = 0;
constexpr uint32_t CHANNEL_NOTIFY_DATA = 1;
constexpr uint32_t WORKER_NOTIFY_START = 0;
constexpr uint64_t SMALL_INPUT_MAX_BYTES = 512ULL * 1024;
constexpr uint64_t STRIPE_ALIGNMENT_ELEMENTS = 1024; // 4 KiB for FP32

struct StripeRange {
    uint64_t offset;
    uint64_t count;
};

const ChannelInfo *ChannelForRank(const OpParam &param, const AlgResourceCtx &resources, uint32_t remoteRank)
{
    if (remoteRank >= param.rankSize || remoteRank == param.myRank) {
        return nullptr;
    }
    const uint32_t channelIdx = remoteRank < param.myRank ? remoteRank : remoteRank - 1;
    if (channelIdx >= resources.channels.size() || resources.channels[channelIdx].remoteRank != remoteRank) {
        return nullptr;
    }
    return &resources.channels[channelIdx];
}

StripeRange GetStripe(uint64_t totalCount, uint32_t stripe)
{
    const uint64_t average = totalCount / REMOTE_PEER_COUNT;
    const uint64_t aligned = (average / STRIPE_ALIGNMENT_ELEMENTS) * STRIPE_ALIGNMENT_ELEMENTS;
    if (aligned != 0) {
        const uint64_t offset = static_cast<uint64_t>(stripe) * aligned;
        const uint64_t count = stripe + 1 == REMOTE_PEER_COUNT ? totalCount - offset : aligned;
        return StripeRange{offset, count};
    }

    const uint64_t base = totalCount / REMOTE_PEER_COUNT;
    const uint64_t remainder = totalCount % REMOTE_PEER_COUNT;
    const uint64_t offset = static_cast<uint64_t>(stripe) * base +
        (stripe < remainder ? stripe : remainder);
    const uint64_t count = base + (stripe < remainder ? 1 : 0);
    return StripeRange{offset, count};
}

HcclResult RunRecursiveHalving(const OpParam &param, const AlgResourceCtx &resources,
    uint64_t blockBytes, uint64_t inputBytes)
{
    ThreadHandle mainThread = resources.threads[0];
    CHK_RET(static_cast<HcclResult>(
        HcommLocalCopyOnThread(mainThread, resources.localBuffer.addr, param.inputPtr, inputBytes)));

    for (uint32_t mask = param.rankSize >> 1; mask != 0; mask >>= 1) {
        const uint32_t peer = param.myRank ^ mask;
        const ChannelInfo *channel = ChannelForRank(param, resources, peer);
        CHK_PRT_RET(channel == nullptr,
            HCCL_ERROR("Missing channel for recursive-halving peer[%u]", peer), HCCL_E_INTERNAL);

        const uint32_t sendBlockStart = peer & ~(mask - 1);
        const uint64_t elementOffset = static_cast<uint64_t>(sendBlockStart) * param.count;
        const uint64_t transferCount = static_cast<uint64_t>(mask) * param.count;
        const void *src = static_cast<const void *>(
            static_cast<const uint8_t *>(resources.localBuffer.addr) + elementOffset * sizeof(float));
        void *dst = static_cast<void *>(
            static_cast<uint8_t *>(channel->remoteCclMem.addr) + elementOffset * sizeof(float));

        // Both endpoints post READY before waiting. DATA is recorded only
        // after WriteReduce on the same thread, and consumed before the next
        // round reuses the one-bit notify.
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
            mainThread, channel->handle, CHANNEL_NOTIFY_READY)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            mainThread, channel->handle, CHANNEL_NOTIFY_READY, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(HcommWriteReduceOnThread(mainThread, channel->handle, dst, src,
            transferCount, static_cast<HcommDataType>(param.dataType),
            static_cast<HcommReduceOp>(param.reduceType))));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
            mainThread, channel->handle, CHANNEL_NOTIFY_DATA)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            mainThread, channel->handle, CHANNEL_NOTIFY_DATA, CUSTOM_TIMEOUT)));
    }

    const void *result = static_cast<const void *>(static_cast<const uint8_t *>(resources.localBuffer.addr) +
        static_cast<uint64_t>(param.myRank) * blockBytes);
    CHK_RET(static_cast<HcclResult>(
        HcommLocalCopyOnThread(mainThread, param.outputPtr, result, blockBytes)));
    return HCCL_SUCCESS;
}

HcclResult RunLatinWriteReduce(const OpParam &param, const AlgResourceCtx &resources, uint64_t blockBytes)
{
    ThreadHandle mainThread = resources.threads[0];
    const void *selfContribution = static_cast<const void *>(static_cast<const uint8_t *>(param.inputPtr) +
        static_cast<uint64_t>(param.myRank) * blockBytes);
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
        mainThread, resources.localBuffer.addr, selfContribution, blockBytes)));

    for (uint32_t round = 0; round < REMOTE_PEER_COUNT; ++round) {
        // Release every peer worker before queuing a wait on main. Each worker
        // owns one channel for the complete collective.
        for (uint32_t channelIdx = 0; channelIdx < resources.channels.size(); ++channelIdx) {
            ThreadHandle worker = resources.threads[channelIdx + 1];
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                mainThread, worker, WORKER_NOTIFY_START)));
        }

        for (uint32_t channelIdx = 0; channelIdx < resources.channels.size(); ++channelIdx) {
            const ChannelInfo &channel = resources.channels[channelIdx];
            ThreadHandle worker = resources.threads[channelIdx + 1];
            const uint32_t targetRank = channel.remoteRank;
            const uint32_t sourceIndexAtTarget = param.myRank < targetRank ? param.myRank : param.myRank - 1;
            const uint32_t stripe = (sourceIndexAtTarget + REMOTE_PEER_COUNT - round) % REMOTE_PEER_COUNT;
            const StripeRange range = GetStripe(param.count, stripe);
            CHK_PRT_RET(range.count == 0 || range.offset > param.count || range.count > param.count - range.offset,
                HCCL_ERROR("Invalid stripe[%u]: offset[%llu], count[%llu], total[%llu]", stripe,
                    static_cast<unsigned long long>(range.offset), static_cast<unsigned long long>(range.count),
                    static_cast<unsigned long long>(param.count)),
                HCCL_E_INTERNAL);

            const uint64_t sourceOffsetElements = static_cast<uint64_t>(targetRank) * param.count + range.offset;
            const void *src = static_cast<const void *>(
                static_cast<const uint8_t *>(param.inputPtr) + sourceOffsetElements * sizeof(float));
            void *dst = static_cast<void *>(
                static_cast<uint8_t *>(channel.remoteCclMem.addr) + range.offset * sizeof(float));

            CHK_RET(static_cast<HcclResult>(
                HcommThreadNotifyWaitOnThread(worker, WORKER_NOTIFY_START, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                worker, channel.handle, CHANNEL_NOTIFY_READY)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                worker, channel.handle, CHANNEL_NOTIFY_READY, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(HcommWriteReduceOnThread(worker, channel.handle, dst, src,
                range.count, static_cast<HcommDataType>(param.dataType),
                static_cast<HcommReduceOp>(param.reduceType))));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                worker, channel.handle, CHANNEL_NOTIFY_DATA)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                worker, channel.handle, CHANNEL_NOTIFY_DATA, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                worker, mainThread, channelIdx + 1)));
        }

        for (uint32_t channelIdx = 0; channelIdx < resources.channels.size(); ++channelIdx) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                mainThread, channelIdx + 1, CUSTOM_TIMEOUT)));
        }
    }

    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
        mainThread, param.outputPtr, resources.localBuffer.addr, blockBytes)));
    return HCCL_SUCCESS;
}
} // namespace

namespace ops_hccl {
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    HCCL_INFO("ReduceScatter from-scratch plan: rank[%u/%u], recvCount[%llu]", param.myRank, param.rankSize,
        static_cast<unsigned long long>(param.count));

    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(param.rankSize != COMPETITION_RANK_SIZE || param.myRank >= param.rankSize,
        HCCL_ERROR("Invalid competition rank metadata: myRank[%u], rankSize[%u]", param.myRank, param.rankSize),
        HCCL_E_PARA);
    CHK_PRT_RET(param.dataType != HCCL_DATA_TYPE_FP32 || param.reduceType != HCCL_REDUCE_SUM,
        HCCL_ERROR("Only FP32 SUM is supported"), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(param.count > UINT64_MAX / sizeof(float) / param.rankSize,
        HCCL_ERROR("Input byte calculation overflow for recvCount[%llu]",
            static_cast<unsigned long long>(param.count)),
        HCCL_E_PARA);
    CHK_PRT_RET(resCtx.localBuffer.addr == nullptr,
        HCCL_ERROR("Local HCCL buffer is null"), HCCL_E_PTR);
    CHK_PRT_RET(resCtx.threads.size() != COMPETITION_RANK_SIZE,
        HCCL_ERROR("Unexpected thread count[%zu], expected[%u]", resCtx.threads.size(), COMPETITION_RANK_SIZE),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.channels.size() != REMOTE_PEER_COUNT,
        HCCL_ERROR("Unexpected channel count[%zu], expected[%u]", resCtx.channels.size(), REMOTE_PEER_COUNT),
        HCCL_E_INTERNAL);

    const uint64_t blockBytes = param.count * sizeof(float);
    const uint64_t inputBytes = blockBytes * param.rankSize;
    uint64_t usableCclBytes = resCtx.localBuffer.size;
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }
        const ChannelInfo *channel = ChannelForRank(param, resCtx, remoteRank);
        CHK_PRT_RET(channel == nullptr || channel->remoteCclMem.addr == nullptr,
            HCCL_ERROR("Missing channel or remote CCL buffer for rank[%u]", remoteRank), HCCL_E_INTERNAL);
        usableCclBytes = usableCclBytes < channel->remoteCclMem.size ?
            usableCclBytes : channel->remoteCclMem.size;
    }

    if (inputBytes <= SMALL_INPUT_MAX_BYTES) {
        CHK_PRT_RET(inputBytes > usableCclBytes,
            HCCL_ERROR("Small-message input[%llu] exceeds usable CCL buffer[%llu]",
                static_cast<unsigned long long>(inputBytes), static_cast<unsigned long long>(usableCclBytes)),
            HCCL_E_MEMORY);
        HCCL_INFO("Select high-bit-first recursive halving for inputBytes[%llu]",
            static_cast<unsigned long long>(inputBytes));
        return RunRecursiveHalving(param, resCtx, blockBytes, inputBytes);
    }

    CHK_PRT_RET(blockBytes > usableCclBytes,
        HCCL_ERROR("Large-message accumulator[%llu] exceeds usable CCL buffer[%llu]",
            static_cast<unsigned long long>(blockBytes), static_cast<unsigned long long>(usableCclBytes)),
        HCCL_E_MEMORY);
    HCCL_INFO("Select 15-round Latin striped WriteReduce for inputBytes[%llu], accumulatorBytes[%llu]",
        static_cast<unsigned long long>(inputBytes), static_cast<unsigned long long>(blockBytes));
    return RunLatinWriteReduce(param, resCtx, blockBytes);
}
} // namespace ops_hccl
