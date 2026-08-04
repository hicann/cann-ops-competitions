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
#include <limits>

#include "custom.h"
#include "log.h"
#include "exec_op.h"

namespace ops_hccl {
namespace {
constexpr uint64_t MAX_READ_BYTES = 256ULL * 1024ULL * 1024ULL;
constexpr uint64_t SMALL_MESSAGE_BYTES = 512ULL * 1024ULL;
constexpr uint32_t RANKS_PER_SERVER = 8;
constexpr uint32_t COMPETITION_RANK_SIZE = 2 * RANKS_PER_SERVER;

HcclResult ThreadSyncBefore(const std::vector<ThreadHandle> &threads)
{
    // Thread 0 is control-only for the large path. Notify 0 starts all
    // Channel workers after the current source slot has been staged.
    for (uint32_t idx = 1; idx < threads.size(); ++idx) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(threads[0], threads[idx], 0)));
    }
    for (uint32_t idx = 1; idx < threads.size(); ++idx) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(threads[idx], 0, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult ThreadSyncAfter(const std::vector<ThreadHandle> &threads)
{
    // Main Notify 0 is used by the fixed Host->AICPU handshake. Use a
    // dedicated, one-producer completion Notify for every worker.
    for (uint32_t idx = 1; idx < threads.size(); ++idx) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(threads[0], idx, CUSTOM_TIMEOUT)));
    }
    for (uint32_t idx = 1; idx < threads.size(); ++idx) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(threads[idx], threads[0], idx)));
    }
    return HCCL_SUCCESS;
}

HcclResult ScheduleBoundedLocalCopy(
    ThreadHandle thread, void *dst, const void *src, uint64_t bytes)
{
    uint64_t processedBytes = 0;
    while (processedBytes < bytes) {
        const uint64_t copyBytes =
            std::min(MAX_READ_BYTES, bytes - processedBytes);
        void *copyDst = static_cast<void *>(
            static_cast<uint8_t *>(dst) + processedBytes);
        const void *copySrc = static_cast<const void *>(
            static_cast<const uint8_t *>(src) + processedBytes);
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(thread, copyDst, copySrc, copyBytes)));
        processedBytes += copyBytes;
    }
    return HCCL_SUCCESS;
}

const ChannelInfo *FindChannelByRank(
    const std::vector<ChannelInfo> &channels, uint32_t remoteRank)
{
    for (const ChannelInfo &channel : channels) {
        if (channel.remoteRank == remoteRank) {
            return &channel;
        }
    }
    return nullptr;
}

HcclResult ScheduleSmallRecursiveLocal(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t dataSize)
{
    // For latency-dominated small messages, build the local 8-rank block with
    // three recursive-doubling exchanges on thread 0. This removes six worker
    // start/completion pairs and four intra-server Channel protocols compared
    // with v010. exp012b additionally evaluates whether four local thread
    // notifications cost more than overlapping the final local copy.
    const uint32_t serverId = param.myRank / RANKS_PER_SERVER;
    const uint32_t serverBaseRank = serverId * RANKS_PER_SERVER;
    const uint32_t localRank = param.myRank % RANKS_PER_SERVER;
    const uint32_t crossServerBaseRank =
        (1U - serverId) * RANKS_PER_SERVER;
    const uint32_t crossRank = crossServerBaseRank + localRank;
    const uint64_t serverBlockBytes = dataSize * RANKS_PER_SERVER;

    CHK_PRT_RET(resCtx.localBuffer.addr == nullptr ||
            resCtx.localBuffer.size < serverBlockBytes,
        HCCL_ERROR("Local CCL buffer is invalid, addr=%p size=%llu required=%llu",
            resCtx.localBuffer.addr,
            static_cast<unsigned long long>(resCtx.localBuffer.size),
            static_cast<unsigned long long>(serverBlockBytes)),
        HCCL_E_PARA);

    void *localInputSlot = static_cast<void *>(
        static_cast<uint8_t *>(resCtx.localBuffer.addr) +
        static_cast<uint64_t>(localRank) * dataSize);
    CHK_RET(static_cast<HcclResult>(
        HcommLocalCopyOnThread(
            resCtx.threads[0], localInputSlot, param.inputPtr, dataSize)));

    // At step k, each local rank owns one aligned block of 2^k ranks and
    // reads the adjacent block from localRank xor 2^k. Distinct partners are
    // used at xor 1, 2 and 4, so every peer still has at most one Channel.
    for (uint32_t groupRankCount = 1;
         groupRankCount < RANKS_PER_SERVER; groupRankCount <<= 1U) {
        const uint32_t remoteLocalRank = localRank ^ groupRankCount;
        const uint32_t remoteRank = serverBaseRank + remoteLocalRank;
        const uint32_t remoteBlockBaseLocalRank =
            remoteLocalRank & ~(groupRankCount - 1U);
        const uint64_t exchangeBytes =
            static_cast<uint64_t>(groupRankCount) * dataSize;

        const ChannelInfo *channel =
            FindChannelByRank(resCtx.channels, remoteRank);
        CHK_PRT_RET(channel == nullptr,
            HCCL_ERROR("No channel found for recursive-local rank %u", remoteRank),
            HCCL_E_NOT_FOUND);
        CHK_PRT_RET(channel->remoteCclMem.addr == nullptr ||
                channel->remoteCclMem.size < serverBlockBytes,
            HCCL_ERROR("Remote CCL buffer is invalid for rank %u, addr=%p size=%llu required=%llu",
                remoteRank, channel->remoteCclMem.addr,
                static_cast<unsigned long long>(channel->remoteCclMem.size),
                static_cast<unsigned long long>(serverBlockBytes)),
            HCCL_E_PARA);

        void *localReceiveBlock = static_cast<void *>(
            static_cast<uint8_t *>(resCtx.localBuffer.addr) +
            static_cast<uint64_t>(remoteBlockBaseLocalRank) * dataSize);
        void *remoteSourceBlock = static_cast<void *>(
            static_cast<uint8_t *>(channel->remoteCclMem.addr) +
            static_cast<uint64_t>(remoteBlockBaseLocalRank) * dataSize);

        // ACK proves the peer has completed its current block. DATA proves
        // both sides have finished reading before either can return and reuse
        // its CCL input slot in the next collective invocation.
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(
                resCtx.threads[0], channel->handle, NOTIFY_IDX_ACK)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(
                resCtx.threads[0], channel->handle,
                NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(
            HcommReadOnThread(
                resCtx.threads[0], channel->handle, localReceiveBlock,
                remoteSourceBlock, exchangeBytes)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(
                resCtx.threads[0], channel->handle, NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(
                resCtx.threads[0], channel->handle,
                NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    }

    // Copy the whole local-server half inline on thread 0. Compared with v011,
    // this deliberately removes the worker start/wait and completion pair.
    // The tradeoff is that the local 8D copy no longer overlaps the Clos Read.
    void *localServerOutput = static_cast<void *>(
        static_cast<uint8_t *>(param.outputPtr) +
        static_cast<uint64_t>(serverBaseRank) * dataSize);
    CHK_RET(static_cast<HcclResult>(
        HcommLocalCopyOnThread(
            resCtx.threads[0], localServerOutput,
            resCtx.localBuffer.addr, serverBlockBytes)));

    const ChannelInfo *crossChannel = FindChannelByRank(resCtx.channels, crossRank);
    CHK_PRT_RET(crossChannel == nullptr,
        HCCL_ERROR("No channel found for paired cross-server rank %u", crossRank),
        HCCL_E_NOT_FOUND);
    CHK_PRT_RET(crossChannel->remoteCclMem.addr == nullptr ||
            crossChannel->remoteCclMem.size < serverBlockBytes,
        HCCL_ERROR("Cross-server CCL buffer is invalid for rank %u, addr=%p size=%llu required=%llu",
            crossRank, crossChannel->remoteCclMem.addr,
            static_cast<unsigned long long>(crossChannel->remoteCclMem.size),
            static_cast<unsigned long long>(serverBlockBytes)),
        HCCL_E_PARA);

    void *crossServerOutput = static_cast<void *>(
        static_cast<uint8_t *>(param.outputPtr) +
        static_cast<uint64_t>(crossServerBaseRank) * dataSize);

    // Both paired ranks reach ACK only after their local server blocks and
    // local output halves are complete. Read the peer's contiguous eight-rank
    // block directly into the disjoint final output half.
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(
            resCtx.threads[0], crossChannel->handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(
            resCtx.threads[0], crossChannel->handle,
            NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(
        HcommReadOnThread(
            resCtx.threads[0], crossChannel->handle, crossServerOutput,
            crossChannel->remoteCclMem.addr, serverBlockBytes)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(
            resCtx.threads[0], crossChannel->handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(
            resCtx.threads[0], crossChannel->handle,
            NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    // Both output halves share thread 0, so the fixed Host completion record
    // is naturally ordered after this function's final DATA wait.
    return HCCL_SUCCESS;
}

HcclResult ScheduleReadDirectExchange(
    const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t processedBytes, uint64_t slotOffset, uint64_t sliceSize,
    uint64_t dataSize)
{
    for (uint32_t idx = 0; idx < resCtx.channels.size(); ++idx) {
        const ChannelInfo &channel = resCtx.channels[idx];
        // Thread 0 stays free for staging/control. Channel idx is permanently
        // scheduled on worker idx + 1 throughout the large invocation.
        const ThreadHandle thread = resCtx.threads[idx + 1];
        CHK_PRT_RET(channel.remoteRank >= param.rankSize,
            HCCL_ERROR("Invalid remote rank %u", channel.remoteRank), HCCL_E_PARA);
        CHK_PRT_RET(channel.remoteCclMem.addr == nullptr ||
                slotOffset > channel.remoteCclMem.size ||
                sliceSize > channel.remoteCclMem.size - slotOffset,
            HCCL_ERROR("Remote CCL buffer is invalid, addr=%p size=%llu offset=%llu required=%llu",
                channel.remoteCclMem.addr,
                static_cast<unsigned long long>(channel.remoteCclMem.size),
                static_cast<unsigned long long>(slotOffset),
                static_cast<unsigned long long>(sliceSize)),
            HCCL_E_PARA);

        void *remoteRankOutputAddr = static_cast<void *>(
            static_cast<uint8_t *>(param.outputPtr) +
            static_cast<uint64_t>(channel.remoteRank) * dataSize + processedBytes);

        // Per-channel strict one-Record/one-Wait protocol:
        // ACK -> Read -> DATA_DONE. ACK means the peer has finished staging
        // its current chunk. DATA_DONE means this rank has finished reading
        // the peer CCL buffer, so the peer may reuse it for the next chunk.
        // No second Record can reach a Notify
        // before the matching Wait consumes and resets the previous one.
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(
                thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        uint64_t readBytesDone = 0;
        while (readBytesDone < sliceSize) {
            const uint64_t readBytes =
                std::min(MAX_READ_BYTES, sliceSize - readBytesDone);
            void *readDst = static_cast<void *>(
                static_cast<uint8_t *>(remoteRankOutputAddr) + readBytesDone);
            const void *readSrc = static_cast<const void *>(
                static_cast<const uint8_t *>(channel.remoteCclMem.addr) +
                slotOffset + readBytesDone);
            CHK_RET(static_cast<HcclResult>(
                HcommReadOnThread(
                    thread, channel.handle, readDst, readSrc, readBytes)));
            readBytesDone += readBytes;
        }
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(
                thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(
                thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));

    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    const auto typeSizeIter = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(typeSizeIter == SIZE_TABLE.end(),
        HCCL_ERROR("Unsupported data type %d", static_cast<int>(param.dataType)),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(param.rankSize == 0 || param.myRank >= param.rankSize,
        HCCL_ERROR("Invalid rank info: rank=%u rankSize=%u", param.myRank, param.rankSize),
        HCCL_E_PARA);

    const uint64_t typeSize = typeSizeIter->second;
    CHK_PRT_RET(typeSize == 0 ||
            param.count > std::numeric_limits<uint64_t>::max() / typeSize,
        HCCL_ERROR("Input byte size overflows, count=%llu typeSize=%llu",
            static_cast<unsigned long long>(param.count),
            static_cast<unsigned long long>(typeSize)),
        HCCL_E_PARA);
    const uint64_t dataSize = param.count * typeSize;
    if (dataSize == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(param.inputPtr);
    CHK_PTR_NULL(param.outputPtr);
    CHK_PRT_RET(dataSize > std::numeric_limits<uint64_t>::max() / param.rankSize,
        HCCL_ERROR("Output byte size overflows, dataSize=%llu rankSize=%u",
            static_cast<unsigned long long>(dataSize), param.rankSize),
        HCCL_E_PARA);

    CHK_PRT_RET(resCtx.threads.empty(),
        HCCL_ERROR("No AICPU thread acquired"), HCCL_E_INTERNAL);

    if (param.rankSize == 1) {
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(
                resCtx.threads[0], param.outputPtr, param.inputPtr, dataSize)));
        return HCCL_SUCCESS;
    }

    const uint32_t peerCount = param.rankSize - 1;
    CHK_PRT_RET(resCtx.threads.size() != param.rankSize ||
            resCtx.channels.size() != peerCount,
        HCCL_ERROR("Resource count mismatch: threads=%zu channels=%zu peers=%u",
            resCtx.threads.size(), resCtx.channels.size(), peerCount),
        HCCL_E_INTERNAL);
    CHK_PTR_NULL(resCtx.localBuffer.addr);

    if (param.rankSize == COMPETITION_RANK_SIZE &&
        dataSize <= SMALL_MESSAGE_BYTES) {
        HCCL_INFO("AllGather algorithm=recursive-local-inline-copy dataSize=%llu",
            static_cast<unsigned long long>(dataSize));
        CHK_RET(ScheduleSmallRecursiveLocal(param, resCtx, dataSize));
        return HCCL_SUCCESS;
    }

    // Preserve v012's two-round layout. The only large-path change is that
    // thread 0 is control/copy-only and all 15 Channels use workers 1..15.
    uint64_t maxSliceBytes =
        std::min(MAX_READ_BYTES, resCtx.localBuffer.size);
    for (const ChannelInfo &channel : resCtx.channels) {
        CHK_PTR_NULL(channel.remoteCclMem.addr);
        maxSliceBytes =
            std::min(maxSliceBytes, channel.remoteCclMem.size);
    }
    maxSliceBytes = (maxSliceBytes / typeSize) * typeSize;
    CHK_PRT_RET(maxSliceBytes == 0,
        HCCL_ERROR("Local or remote CCL buffer is empty"),
        HCCL_E_PARA);

    HCCL_INFO("AllGather algorithm=large-control-copy-overlap dataSize=%llu maxSlice=%llu",
        static_cast<unsigned long long>(dataSize),
        static_cast<unsigned long long>(maxSliceBytes));

    uint64_t processedBytes = 0;
    while (processedBytes < dataSize) {
        const uint64_t sliceSize =
            std::min(maxSliceBytes, dataSize - processedBytes);
        const void *inputAddr = static_cast<const void *>(
            static_cast<const uint8_t *>(param.inputPtr) + processedBytes);

        CHK_RET(ScheduleBoundedLocalCopy(
            resCtx.threads[0], resCtx.localBuffer.addr,
            inputAddr, sliceSize));

        CHK_RET(ThreadSyncBefore(resCtx.threads));
        CHK_RET(ScheduleReadDirectExchange(
            param, resCtx, processedBytes, 0,
            sliceSize, dataSize));

        // Start Records are already ordered after the input staging copy.
        // Copy this rank's final slice on control thread 0 while all 15
        // workers execute network Reads, then queue completion Waits.
        void *localRankOutputAddr = static_cast<void *>(
            static_cast<uint8_t *>(param.outputPtr) +
            static_cast<uint64_t>(param.myRank) * dataSize + processedBytes);
        CHK_RET(ScheduleBoundedLocalCopy(
            resCtx.threads[0], localRankOutputAddr,
            resCtx.localBuffer.addr, sliceSize));

        CHK_RET(ThreadSyncAfter(resCtx.threads));
        processedBytes += sliceSize;
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
