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
#include <cstdint>
#include <vector>

namespace ops_hccl {
namespace {
constexpr uint64_t FP32_SIZE = sizeof(float);
constexpr uint64_t HCCL_MIN_SLICE_ALIGN = 128;
constexpr uint64_t MAX_SLICE_SIZE = 256ULL * 1024ULL * 1024ULL;

void *AddByteOffset(void *address, uint64_t offset)
{
    return static_cast<void *>(static_cast<uint8_t *>(address) + offset);
}

HcclResult ThreadSyncBefore(const std::vector<ThreadHandle> &threads)
{
    for (uint32_t index = 1; index < threads.size(); ++index) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(threads[0], threads[index], 0)));
    }
    for (uint32_t index = 1; index < threads.size(); ++index) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(threads[index], 0, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult ThreadSyncAfter(const std::vector<ThreadHandle> &threads)
{
    for (uint32_t index = 1; index < threads.size(); ++index) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(threads[0], index - 1, CUSTOM_TIMEOUT)));
    }
    for (uint32_t index = 1; index < threads.size(); ++index) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(threads[index], threads[0], index - 1)));
    }
    return HCCL_SUCCESS;
}

HcclResult ValidateResources(const OpParam &param, const AlgResourceCtx &resCtx)
{
    if (param.rankSize == 0 || param.myRank >= param.rankSize) {
        HCCL_ERROR("Invalid rank information: rank %u, rank size %u", param.myRank, param.rankSize);
        return HCCL_E_INTERNAL;
    }
    if (resCtx.localBuffer.addr == nullptr) {
        HCCL_ERROR("Local HCCL buffer is null");
        return HCCL_E_PTR;
    }

    const uint32_t expectedPeerCount = param.rankSize - 1;
    if (resCtx.threads.size() != (expectedPeerCount == 0 ? 1U : expectedPeerCount) ||
        resCtx.channels.size() != expectedPeerCount) {
        HCCL_ERROR("Unexpected resource count: threads %zu, channels %zu, peers %u",
            resCtx.threads.size(), resCtx.channels.size(), expectedPeerCount);
        return HCCL_E_INTERNAL;
    }

    std::vector<bool> rankSeen(param.rankSize, false);
    rankSeen[param.myRank] = true;
    for (const ChannelInfo &channel : resCtx.channels) {
        if (channel.remoteRank >= param.rankSize || rankSeen[channel.remoteRank] ||
            channel.remoteCclMem.addr == nullptr) {
            HCCL_ERROR("Invalid channel for remote rank %u", channel.remoteRank);
            return HCCL_E_INTERNAL;
        }
        rankSeen[channel.remoteRank] = true;
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
    CHK_RET(ValidateResources(param, resCtx));

    if (param.count == 0) {
        return HCCL_SUCCESS;
    }

    const uint64_t outputSize = param.count * FP32_SIZE;
    const ThreadHandle mainThread = resCtx.threads[0];
    if (param.rankSize == 1) {
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(mainThread, param.outputPtr, param.inputPtr, outputSize)));
        return HCCL_SUCCESS;
    }

    // 每轮的本地 HCCL Buffer 由两部分组成：
    //   [rankSize 个发送分片][rankSize - 1 个远端读取临时分片]
    // 网络读取并行落入不同临时分片，规避并发归约写同一地址造成的不确定性。
    const uint64_t localSliceCount = static_cast<uint64_t>(param.rankSize) + resCtx.channels.size();
    // 所有 rank 都取整个通信域最小的 HCCL Buffer，再按同一布局切分，保证即使各 rank
    // Buffer 大小不同，发送区偏移和每轮分片大小仍完全一致。
    uint64_t minBufferSize = resCtx.localBuffer.size;
    for (const ChannelInfo &channel : resCtx.channels) {
        minBufferSize = std::min(minBufferSize, channel.remoteCclMem.size);
    }
    uint64_t maxSliceSize = minBufferSize / localSliceCount;
    maxSliceSize = std::min(maxSliceSize, MAX_SLICE_SIZE);
    maxSliceSize = maxSliceSize / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
    const uint64_t maxCountPerLoop = maxSliceSize / FP32_SIZE;
    if (maxCountPerLoop == 0) {
        HCCL_ERROR("HCCL buffer is too small for ReduceScatter");
        return HCCL_E_INTERNAL;
    }

    // 通道按 remoteRank 建索引，后续始终按 rank 0..N-1 的固定顺序归约。
    std::vector<uint32_t> channelIndexByRank(param.rankSize, INVALID_VALUE_RANKID);
    for (uint32_t channelIndex = 0; channelIndex < resCtx.channels.size(); ++channelIndex) {
        channelIndexByRank[resCtx.channels[channelIndex].remoteRank] = channelIndex;
    }

    // Reuse the pointer table across chunks instead of allocating it on every loop.
    std::vector<void *> sourceBlocks(param.rankSize, nullptr);
    const bool singleLoop = param.count <= maxCountPerLoop;
    uint64_t processedCount = 0;
    while (processedCount < param.count) {
        const uint64_t sliceCount = std::min(maxCountPerLoop, param.count - processedCount);
        const uint64_t sliceSize = sliceCount * FP32_SIZE;

        // A one-loop input is already laid out as contiguous destination shards.
        // Stage it with one task to avoid 16 separate local-copy submissions.
        if (singleLoop) {
            CHK_RET(static_cast<HcclResult>(
                HcommLocalCopyOnThread(mainThread, resCtx.localBuffer.addr, param.inputPtr,
                    outputSize * static_cast<uint64_t>(param.rankSize))));
        } else {
            // Multi-loop chunks are strided in the user input and must be staged separately.
            for (uint32_t destinationRank = 0; destinationRank < param.rankSize; ++destinationRank) {
                void *localStage = AddByteOffset(
                    resCtx.localBuffer.addr, static_cast<uint64_t>(destinationRank) * sliceSize);
                const uint64_t inputOffset =
                    static_cast<uint64_t>(destinationRank) * outputSize + processedCount * FP32_SIZE;
                const void *userInput = AddByteOffset(param.inputPtr, inputOffset);
                CHK_RET(static_cast<HcclResult>(
                    HcommLocalCopyOnThread(mainThread, localStage, userInput, sliceSize)));
            }
        }

        // 本地搬运完成后再放行各通信 Thread。
        CHK_RET(ThreadSyncBefore(resCtx.threads));

        // 每条 Thread 从一个对端并行读取本 rank 所需的分片。
        for (uint32_t channelIndex = 0; channelIndex < resCtx.channels.size(); ++channelIndex) {
            const ChannelInfo &channel = resCtx.channels[channelIndex];
            const ThreadHandle thread = resCtx.threads[channelIndex];
            const void *remoteStage = AddByteOffset(
                channel.remoteCclMem.addr, static_cast<uint64_t>(param.myRank) * sliceSize);
            void *localScratch = AddByteOffset(resCtx.localBuffer.addr,
                (static_cast<uint64_t>(param.rankSize) + channelIndex) * sliceSize);

            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK)));
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(
                HcommReadOnThread(thread, channel.handle, localScratch, remoteStage, sliceSize)));
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
        }

        // 等待全部远端读取以及对端对本地发送区的读取完成，随后才能归约并复用 Buffer。
        CHK_RET(ThreadSyncAfter(resCtx.threads));

        sourceBlocks[param.myRank] = AddByteOffset(
            resCtx.localBuffer.addr, static_cast<uint64_t>(param.myRank) * sliceSize);
        for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
            if (remoteRank == param.myRank) {
                continue;
            }
            const uint32_t channelIndex = channelIndexByRank[remoteRank];
            if (channelIndex == INVALID_VALUE_RANKID) {
                HCCL_ERROR("Missing channel for rank %u", remoteRank);
                return HCCL_E_INTERNAL;
            }
            sourceBlocks[remoteRank] = AddByteOffset(resCtx.localBuffer.addr,
                (static_cast<uint64_t>(param.rankSize) + channelIndex) * sliceSize);
        }

        // 固定以 rank 0 为初值，再严格按 1..N-1 顺序归约，确保浮点结果可重复。
        void *accumulator = sourceBlocks[0];
        for (uint32_t sourceRank = 1; sourceRank < param.rankSize; ++sourceRank) {
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(mainThread, accumulator,
                sourceBlocks[sourceRank], sliceCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
        }

        void *userOutput = AddByteOffset(param.outputPtr, processedCount * FP32_SIZE);
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(mainThread, userOutput, accumulator, sliceSize)));
        processedCount += sliceCount;
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
