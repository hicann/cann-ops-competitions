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
#include <unordered_map>

namespace ops_hccl {
namespace {
constexpr uint64_t HCCL_MIN_SLICE_ALIGN = 128;
constexpr uint64_t ASSUMED_CCL_BUFFER_SIZE = 400ULL * 1024 * 1024;
constexpr uint64_t RECURSIVE_DOUBLING_MAX_DATA_SIZE = 1ULL * 1024 * 1024;

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

uint64_t GetMinCclBufferSize(const AlgResourceCtx &resCtx)
{
    uint64_t minBufferSize = resCtx.localBuffer.size;
    for (const ChannelInfo &channel : resCtx.channels) {
        if (channel.remoteCclMem.addr == nullptr || channel.remoteCclMem.size == 0) {
            return 0;
        }
        minBufferSize = std::min(minBufferSize, channel.remoteCclMem.size);
    }
    return minBufferSize;
}

HcclResult WriteBlockOnThread(ThreadHandle thread, const ChannelInfo &channel, uint8_t *localCclBase,
    uint64_t blockOffset, uint64_t blockBytes)
{
    CHK_PRT_RET(channel.remoteCclMem.addr == nullptr || channel.remoteCclMem.size < blockOffset + blockBytes,
        HCCL_ERROR("Remote CCL buffer too small for rank[%u], offset[%llu] bytes[%llu]", channel.remoteRank,
            static_cast<unsigned long long>(blockOffset), static_cast<unsigned long long>(blockBytes)),
        HCCL_E_INTERNAL);

    uint8_t *localBlock = localCclBase + blockOffset;
    uint8_t *remoteBlock = static_cast<uint8_t *>(channel.remoteCclMem.addr) + blockOffset;
    CHK_RET(static_cast<HcclResult>(
        HcommWriteOnThread(thread, channel.handle, remoteBlock, localBlock, blockBytes)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult ReadBlockOnThread(ThreadHandle thread, const ChannelInfo &channel, void *destination,
    uint64_t remoteOffset, uint64_t blockBytes)
{
    CHK_PRT_RET(channel.remoteCclMem.addr == nullptr || remoteOffset > channel.remoteCclMem.size ||
            blockBytes > channel.remoteCclMem.size - remoteOffset,
        HCCL_ERROR("Remote CCL buffer too small for rank[%u], offset[%llu] bytes[%llu]", channel.remoteRank,
            static_cast<unsigned long long>(remoteOffset), static_cast<unsigned long long>(blockBytes)),
        HCCL_E_INTERNAL);

    const uint8_t *remoteBlock = static_cast<const uint8_t *>(channel.remoteCclMem.addr) + remoteOffset;
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(
        HcommReadOnThread(thread, channel.handle, destination, remoteBlock, blockBytes)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult RecursiveDoublingAllGather(
    const OpParam &param, const AlgResourceCtx &resCtx, uint8_t *localCclBase, uint64_t sliceBytes)
{
    const ThreadHandle thread = resCtx.aicpuThread;
    for (uint32_t distance = 1; distance < param.rankSize; distance <<= 1) {
        const uint32_t partnerRank = param.myRank ^ distance;
        const ChannelInfo *channel = FindChannel(resCtx, partnerRank);
        CHK_PRT_RET(channel == nullptr,
            HCCL_ERROR("No channel from rank[%u] to recursive-doubling partner[%u]", param.myRank, partnerRank),
            HCCL_E_INTERNAL);

        // Before this stage each rank owns a contiguous block of `distance` rank slices.
        const uint32_t blockStartRank = param.myRank & ~(distance - 1);
        const uint64_t blockOffset = static_cast<uint64_t>(blockStartRank) * sliceBytes;
        const uint64_t blockBytes = static_cast<uint64_t>(distance) * sliceBytes;
        CHK_RET(WriteBlockOnThread(thread, *channel, localCclBase, blockOffset, blockBytes));
    }
    return HCCL_SUCCESS;
}

HcclResult ReleaseWorkers(const std::vector<ThreadHandle> &threads)
{
    for (uint32_t i = 1; i < threads.size(); ++i) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[0], threads[i], 0)));
    }
    for (uint32_t i = 1; i < threads.size(); ++i) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[i], 0, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult JoinWorkers(const std::vector<ThreadHandle> &threads)
{
    for (uint32_t i = 1; i < threads.size(); ++i) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[0], i, CUSTOM_TIMEOUT)));
    }
    for (uint32_t i = 1; i < threads.size(); ++i) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[i], threads[0], i)));
    }
    return HCCL_SUCCESS;
}

HcclResult CopyRecursiveDoublingResultToOutput(
    const OpParam &param, ThreadHandle thread, uint8_t *localCclBase, uint64_t dataSize)
{
    const uint64_t outputBytes = static_cast<uint64_t>(param.rankSize) * dataSize;
    CHK_RET(static_cast<HcclResult>(
        HcommLocalCopyOnThread(thread, param.outputPtr, localCclBase, outputBytes)));
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }

    const auto sizeIter = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(sizeIter == SIZE_TABLE.end(), HCCL_ERROR("Unsupported data type[%d]", param.dataType),
        HCCL_E_NOT_SUPPORT);

    const uint64_t dataTypeSize = sizeIter->second;
    const uint64_t dataSize = param.count * dataTypeSize;
    const ThreadHandle thread = resCtx.aicpuThread;

    if (param.rankSize == 1) {
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, param.outputPtr, param.inputPtr, dataSize)));
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(resCtx.localBuffer.addr == nullptr || resCtx.localBuffer.size == 0,
        HCCL_ERROR("Invalid local CCL buffer"), HCCL_E_INTERNAL);

    const uint64_t minCclBufferSize = GetMinCclBufferSize(resCtx);
    CHK_PRT_RET(minCclBufferSize == 0, HCCL_ERROR("Invalid remote CCL buffer"), HCCL_E_INTERNAL);

    const bool useRecursiveDoubling = param.rankSize == 16 && IsPowerOfTwo(param.rankSize) &&
        dataSize <= RECURSIVE_DOUBLING_MAX_DATA_SIZE;

    // Recursive doubling needs one slot per rank. The large-message pull path stages only this rank's input,
    // so it can use the whole CCL buffer and needs only two chunks for the official 512 MiB case.
    const uint64_t usableCclBufferSize = std::min(minCclBufferSize, ASSUMED_CCL_BUFFER_SIZE);
    uint64_t maxSliceBytes = useRecursiveDoubling ? usableCclBufferSize / param.rankSize : usableCclBufferSize;
    maxSliceBytes = maxSliceBytes / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN;
    CHK_PRT_RET(maxSliceBytes < dataTypeSize,
        HCCL_ERROR("CCL buffer too small, size[%llu] rankSize[%u]",
            static_cast<unsigned long long>(usableCclBufferSize), param.rankSize),
        HCCL_E_INTERNAL);

    const uint64_t maxSliceCount = maxSliceBytes / dataTypeSize;

    CHK_PRT_RET(resCtx.channels.size() != param.rankSize - 1,
        HCCL_ERROR("Channel count mismatch, actual[%zu] expected[%u]", resCtx.channels.size(), param.rankSize - 1),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.threads.size() != resCtx.channels.size() + 1,
        HCCL_ERROR("Control/worker/channel mismatch, threads[%zu] channels[%zu]", resCtx.threads.size(),
            resCtx.channels.size()),
        HCCL_E_INTERNAL);

    uint64_t processedCount = 0;
    while (processedCount < param.count) {
        const uint64_t sliceCount = std::min(maxSliceCount, param.count - processedCount);
        const uint64_t sliceBytes = sliceCount * dataTypeSize;
        uint8_t *localCclBase = static_cast<uint8_t *>(resCtx.localBuffer.addr);
        const uint64_t localRankOffset = useRecursiveDoubling ?
            static_cast<uint64_t>(param.myRank) * sliceBytes : 0;
        uint8_t *localSlice = localCclBase + localRankOffset;
        uint8_t *inputSlice = static_cast<uint8_t *>(param.inputPtr) + processedCount * dataTypeSize;

        // Publish this rank's slice before releasing readers on all peer channels.
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, localSlice, inputSlice, sliceBytes)));

        if (useRecursiveDoubling) {
            // Rank 0-7/8-15 map to the two servers: XOR 1/2/4 stays local, XOR 8 crosses servers once.
            CHK_RET(RecursiveDoublingAllGather(param, resCtx, localCclBase, sliceBytes));
            CHK_RET(CopyRecursiveDoublingResultToOutput(param, thread, localCclBase, dataSize));
        } else {
            CHK_RET(ReleaseWorkers(resCtx.threads));

            // Pull every remote slice directly into its final output slot. Seven Mesh reads and eight Clos reads
            // execute together, avoiding the previous remote-CCL-to-output copy of all 15 remote slices.
            for (uint32_t i = 0; i < resCtx.channels.size(); ++i) {
                const ChannelInfo &channel = resCtx.channels[i];
                uint8_t *destination = static_cast<uint8_t *>(param.outputPtr) +
                    static_cast<uint64_t>(channel.remoteRank) * dataSize + processedCount * dataTypeSize;
                CHK_RET(ReadBlockOnThread(resCtx.threads[i + 1], channel, destination, 0, sliceBytes));
            }

            // The local output copy overlaps the 15 remote reads on their independent worker Threads.
            uint8_t *localDestination = static_cast<uint8_t *>(param.outputPtr) +
                static_cast<uint64_t>(param.myRank) * dataSize + processedCount * dataTypeSize;
            CHK_RET(static_cast<HcclResult>(
                HcommLocalCopyOnThread(thread, localDestination, inputSlice, sliceBytes)));
            CHK_RET(JoinWorkers(resCtx.threads));
        }

        processedCount += sliceCount;
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
