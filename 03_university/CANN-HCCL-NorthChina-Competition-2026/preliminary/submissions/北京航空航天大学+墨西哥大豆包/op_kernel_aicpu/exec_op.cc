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

// 新增
#include <algorithm>
#include <vector>
// 结束

// 新增
namespace {
constexpr uint32_t SERVER_RANK_SIZE = 8;
constexpr uint32_t EXPECTED_RANK_SIZE = 16;
constexpr uint32_t NOTIFY_IDX_ACK = 0;
constexpr uint32_t NOTIFY_IDX_DATA_SIGNAL = 1;
// 删除
// // Direct all-to-all has lower setup overhead only for genuinely tiny messages.
// // At 512 KiB the hierarchical path uses half as many channels and is already
// // preferable on the fixed 2 x 8 topology.
// 新增
// Keep the original direct path for tiny messages. The 512 KiB performance case
// uses fused direct push, while larger cases use direct pull into recvBuf.
// 结束
constexpr uint64_t SMALL_DATA_SIZE = 4ULL * 1024ULL;
// 新增
constexpr uint64_t MEDIUM_DATA_SIZE = 512ULL * 1024ULL;
// 结束
constexpr uint64_t MIN_SLICE_ALIGN = 128;
constexpr uint64_t MAX_CHUNK_SIZE = 256ULL * 1024ULL * 1024ULL;

uint8_t *AddOffset(void *address, uint64_t offset)
{
    return static_cast<uint8_t *>(address) + offset;
}

// 删除
// const uint8_t *AddOffset(const void *address, uint64_t offset)
// {
//     return static_cast<const uint8_t *>(address) + offset;
// }

uint64_t AlignDown(uint64_t value, uint64_t align)
{
    return value / align * align;
}

HcclResult StartWorker(ThreadHandle controlThread, ThreadHandle workerThread)
{
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(controlThread, workerThread, 0)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(workerThread, 0, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult WaitWorker(ThreadHandle controlThread, ThreadHandle workerThread, uint32_t notifyIdx)
{
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(controlThread, notifyIdx, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(workerThread, controlThread, notifyIdx)));
    return HCCL_SUCCESS;
}

HcclResult WriteToPeer(ThreadHandle thread, const ChannelInfo &channel, void *remoteDst, const void *localSrc,
                       uint64_t size)
{
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(thread, channel.handle, remoteDst, localSrc, size)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

// 新增
HcclResult WriteToPeerNoAck(ThreadHandle thread, const ChannelInfo &channel, void *remoteDst,
                            const void *localSrc, uint64_t size)
{
    CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(thread, channel.handle, remoteDst, localSrc, size)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}
// 结束

HcclResult LaunchChannelWrite(const AlgResourceCtx &resCtx, ThreadHandle controlThread, const ChannelInfo &channel,
                              void *remoteDst, const void *localSrc, uint64_t size)
{
    CHK_PRT_RET(channel.threadIdx >= resCtx.threads.size(),
        HCCL_ERROR("Invalid thread index %u for peer %u", channel.threadIdx, channel.remoteRank), HCCL_E_INTERNAL);
    ThreadHandle workerThread = resCtx.threads[channel.threadIdx];
    CHK_RET(StartWorker(controlThread, workerThread));
    CHK_RET(WriteToPeer(workerThread, channel, remoteDst, localSrc, size));
    return HCCL_SUCCESS;
}

HcclResult WaitChannelWrite(const AlgResourceCtx &resCtx, ThreadHandle controlThread, const ChannelInfo &channel)
{
    CHK_PRT_RET(channel.threadIdx >= resCtx.threads.size(),
        HCCL_ERROR("Invalid thread index %u for peer %u", channel.threadIdx, channel.remoteRank), HCCL_E_INTERNAL);
    return WaitWorker(controlThread, resCtx.threads[channel.threadIdx], channel.threadIdx);
}

// 新增
HcclResult ReadFromPeer(ThreadHandle thread, const ChannelInfo &channel, void *localDst, const void *remoteSrc,
                        uint64_t size)
{
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommReadOnThread(thread, channel.handle, localDst, remoteSrc, size)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult LaunchChannelRead(const AlgResourceCtx &resCtx, ThreadHandle controlThread, const ChannelInfo &channel,
                             void *localDst, const void *remoteSrc, uint64_t size)
{
    static_cast<void>(controlThread);
    CHK_PRT_RET(channel.threadIdx >= resCtx.threads.size(),
        HCCL_ERROR("Invalid thread index %u for peer %u", channel.threadIdx, channel.remoteRank), HCCL_E_INTERNAL);
    ThreadHandle workerThread = resCtx.threads[channel.threadIdx];
    // 删除
    // CHK_RET(StartWorker(controlThread, workerThread));
    CHK_RET(ReadFromPeer(workerThread, channel, localDst, remoteSrc, size));
    return HCCL_SUCCESS;
}

HcclResult StartAllChannelWorkers(const AlgResourceCtx &resCtx, ThreadHandle controlThread)
{
    for (const ChannelInfo &channel : resCtx.channels) {
        CHK_PRT_RET(channel.threadIdx >= resCtx.threads.size(),
            HCCL_ERROR("Invalid thread index %u for peer %u", channel.threadIdx, channel.remoteRank),
            HCCL_E_INTERNAL);
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            controlThread, resCtx.threads[channel.threadIdx], 0)));
    }
    for (const ChannelInfo &channel : resCtx.channels) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            resCtx.threads[channel.threadIdx], 0, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult WaitAllChannelWorkers(const AlgResourceCtx &resCtx, ThreadHandle controlThread)
{
    for (const ChannelInfo &channel : resCtx.channels) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(controlThread, channel.threadIdx, CUSTOM_TIMEOUT)));
    }
    for (const ChannelInfo &channel : resCtx.channels) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resCtx.threads[channel.threadIdx], controlThread, channel.threadIdx)));
    }
    return HCCL_SUCCESS;
}

[[maybe_unused]] HcclResult WaitAllChannelWorkersRecordFirst(
    const AlgResourceCtx &resCtx, ThreadHandle controlThread)
{
    // Submit completion records after each worker's communication tasks, then
    // submit the matching waits on the control thread. This preserves one-to-one
    // notify use and avoids putting all control waits ahead of their producers.
    for (const ChannelInfo &channel : resCtx.channels) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resCtx.threads[channel.threadIdx], controlThread, channel.threadIdx)));
    }
    for (const ChannelInfo &channel : resCtx.channels) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(controlThread, channel.threadIdx, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}
// 结束

HcclResult CopyChunkToOutputRange(const OpParam &param, const AlgResourceCtx &resCtx, ThreadHandle copyThread,
                                  uint32_t sourceRankBegin, uint32_t sourceRankEnd, uint64_t bankBase,
                                  uint64_t processedCount, uint64_t chunkCount, uint32_t dataTypeSize)
{
    uint64_t chunkBytes = chunkCount * dataTypeSize;
    uint64_t fullInputBytes = param.count * dataTypeSize;
    for (uint32_t sourceRank = sourceRankBegin; sourceRank < sourceRankEnd; ++sourceRank) {
        const void *src = AddOffset(resCtx.localBuffer.addr, bankBase + sourceRank * chunkBytes);
        void *dst = AddOffset(param.outputPtr, sourceRank * fullInputBytes + processedCount * dataTypeSize);
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(copyThread, dst, src, chunkBytes)));
    }
    return HCCL_SUCCESS;
}

HcclResult CopyChunkToOutput(const OpParam &param, const AlgResourceCtx &resCtx, ThreadHandle copyThread,
                             uint64_t bankBase, uint64_t processedCount, uint64_t chunkCount,
                             uint32_t dataTypeSize)
{
    return CopyChunkToOutputRange(param, resCtx, copyThread, 0, param.rankSize, bankBase, processedCount,
                                  chunkCount, dataTypeSize);
}

HcclResult LaunchCopyChunkToOutputRange(const OpParam &param, const AlgResourceCtx &resCtx,
                                        ThreadHandle controlThread, ThreadHandle copyThread,
                                        uint32_t sourceRankBegin, uint32_t sourceRankEnd, uint64_t bankBase,
                                        uint64_t processedCount, uint64_t chunkCount, uint32_t dataTypeSize)
{
    CHK_RET(StartWorker(controlThread, copyThread));
    return CopyChunkToOutputRange(param, resCtx, copyThread, sourceRankBegin, sourceRankEnd, bankBase,
                                  processedCount, chunkCount, dataTypeSize);
}

HcclResult WaitCopyChunk(ThreadHandle controlThread, ThreadHandle copyThread, uint32_t notifyIdx)
{
    return WaitWorker(controlThread, copyThread, notifyIdx);
}

HcclResult RunDirectAllGather(const OpParam &param, const AlgResourceCtx &resCtx, uint32_t dataTypeSize)
{
    ThreadHandle controlThread = resCtx.threads[0];
    uint64_t scratchBound = AlignDown(resCtx.localBuffer.size / param.rankSize, MIN_SLICE_ALIGN);
    uint64_t maxChunkBytes = std::min(MAX_CHUNK_SIZE, scratchBound);
    CHK_PRT_RET(maxChunkBytes == 0, HCCL_ERROR("CCL buffer is too small"), HCCL_E_INTERNAL);

    uint64_t maxChunkCount = maxChunkBytes / dataTypeSize;
    uint64_t processedCount = 0;
    while (processedCount < param.count) {
        uint64_t chunkCount = std::min(maxChunkCount, param.count - processedCount);
        uint64_t chunkBytes = chunkCount * dataTypeSize;
        void *localSlice = AddOffset(resCtx.localBuffer.addr, param.myRank * chunkBytes);
        const void *inputSlice = AddOffset(param.inputPtr, processedCount * dataTypeSize);
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(controlThread, localSlice, inputSlice, chunkBytes)));

        for (const ChannelInfo &channel : resCtx.channels) {
            void *remoteDst = AddOffset(channel.remoteCclMem.addr, param.myRank * chunkBytes);
            CHK_RET(LaunchChannelWrite(resCtx, controlThread, channel, remoteDst, localSlice, chunkBytes));
        }
        for (const ChannelInfo &channel : resCtx.channels) {
            CHK_RET(WaitChannelWrite(resCtx, controlThread, channel));
        }

        CHK_RET(CopyChunkToOutput(param, resCtx, controlThread, 0, processedCount, chunkCount, dataTypeSize));
        processedCount += chunkCount;
    }
    return HCCL_SUCCESS;
}

// 新增
[[maybe_unused]] HcclResult RunFusedDirectAllGather(
    const OpParam &param, const AlgResourceCtx &resCtx, uint32_t dataTypeSize)
{
    ThreadHandle controlThread = resCtx.threads[0];
    uint64_t scratchBound = AlignDown(resCtx.localBuffer.size / param.rankSize, MIN_SLICE_ALIGN);
    uint64_t maxChunkBytes = std::min(MAX_CHUNK_SIZE, scratchBound);
    CHK_PRT_RET(maxChunkBytes == 0, HCCL_ERROR("CCL buffer is too small"), HCCL_E_INTERNAL);

    uint64_t maxChunkCount = maxChunkBytes / dataTypeSize;
    uint64_t fullInputBytes = param.count * dataTypeSize;
    uint64_t processedCount = 0;
    while (processedCount < param.count) {
        uint64_t chunkCount = std::min(maxChunkCount, param.count - processedCount);
        uint64_t chunkBytes = chunkCount * dataTypeSize;
        uint64_t processedBytes = processedCount * dataTypeSize;
        void *localSlice = AddOffset(resCtx.localBuffer.addr, param.myRank * chunkBytes);
        const void *inputSlice = AddOffset(param.inputPtr, processedBytes);
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(controlThread, localSlice, inputSlice, chunkBytes)));
        CHK_RET(StartAllChannelWorkers(resCtx, controlThread));

        for (const ChannelInfo &channel : resCtx.channels) {
            CHK_PRT_RET(channel.remoteRank >= param.rankSize,
                HCCL_ERROR("Invalid remote rank %u", channel.remoteRank), HCCL_E_INTERNAL);
            CHK_PRT_RET(channel.threadIdx >= resCtx.threads.size(),
                HCCL_ERROR("Invalid thread index %u", channel.threadIdx), HCCL_E_INTERNAL);
            ThreadHandle workerThread = resCtx.threads[channel.threadIdx];
            void *remoteDst = AddOffset(channel.remoteCclMem.addr, param.myRank * chunkBytes);
            const void *peerScratch = AddOffset(resCtx.localBuffer.addr, channel.remoteRank * chunkBytes);
            void *peerOutput = AddOffset(param.outputPtr,
                channel.remoteRank * fullInputBytes + processedBytes);
            // 删除
            // CHK_RET(StartWorker(controlThread, workerThread));
            CHK_RET(WriteToPeer(workerThread, channel, remoteDst, localSlice, chunkBytes));
            CHK_RET(static_cast<HcclResult>(
                HcommLocalCopyOnThread(workerThread, peerOutput, peerScratch, chunkBytes)));
        }

        void *selfOutput = AddOffset(param.outputPtr, param.myRank * fullInputBytes + processedBytes);
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(controlThread, selfOutput, inputSlice, chunkBytes)));

        // 删除
        // for (const ChannelInfo &channel : resCtx.channels) {
        //     CHK_RET(WaitChannelWrite(resCtx, controlThread, channel));
        // }
        CHK_RET(WaitAllChannelWorkers(resCtx, controlThread));
        processedCount += chunkCount;
    }
    return HCCL_SUCCESS;
}

HcclResult RunDirectPullAllGather(const OpParam &param, const AlgResourceCtx &resCtx, uint32_t dataTypeSize)
{
    ThreadHandle controlThread = resCtx.threads[0];
    uint64_t scratchBound = resCtx.localBuffer.size;
    for (const ChannelInfo &channel : resCtx.channels) {
        scratchBound = std::min(scratchBound, channel.remoteCclMem.size);
    }
    uint64_t maxChunkBytes = std::min(MAX_CHUNK_SIZE, AlignDown(scratchBound, MIN_SLICE_ALIGN));
    CHK_PRT_RET(maxChunkBytes == 0, HCCL_ERROR("CCL buffer is too small for direct pull"), HCCL_E_INTERNAL);

    uint64_t maxChunkCount = maxChunkBytes / dataTypeSize;
    uint64_t fullInputBytes = param.count * dataTypeSize;
    uint64_t processedCount = 0;
    while (processedCount < param.count) {
        uint64_t chunkCount = std::min(maxChunkCount, param.count - processedCount);
        uint64_t chunkBytes = chunkCount * dataTypeSize;
        uint64_t processedBytes = processedCount * dataTypeSize;
        void *localScratch = resCtx.localBuffer.addr;
        const void *inputSlice = AddOffset(param.inputPtr, processedBytes);
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(controlThread, localScratch, inputSlice, chunkBytes)));
        CHK_RET(StartAllChannelWorkers(resCtx, controlThread));

        for (const ChannelInfo &channel : resCtx.channels) {
            CHK_PRT_RET(channel.remoteRank >= param.rankSize,
                HCCL_ERROR("Invalid remote rank %u", channel.remoteRank), HCCL_E_INTERNAL);
            void *peerOutput = AddOffset(param.outputPtr,
                channel.remoteRank * fullInputBytes + processedBytes);
            CHK_RET(LaunchChannelRead(
                resCtx, controlThread, channel, peerOutput, channel.remoteCclMem.addr, chunkBytes));
        }

        void *selfOutput = AddOffset(param.outputPtr, param.myRank * fullInputBytes + processedBytes);
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(controlThread, selfOutput, inputSlice, chunkBytes)));

        // 删除
        // for (const ChannelInfo &channel : resCtx.channels) {
        //     CHK_RET(WaitChannelWrite(resCtx, controlThread, channel));
        // }
        CHK_RET(WaitAllChannelWorkers(resCtx, controlThread));
        processedCount += chunkCount;
    }
    return HCCL_SUCCESS;
}

HcclResult RunMediumDirectPullAllGather(const OpParam &param, const AlgResourceCtx &resCtx,
                                        uint32_t dataTypeSize)
{
    ThreadHandle controlThread = resCtx.threads[0];
    uint64_t dataSize = param.count * dataTypeSize;
    CHK_PRT_RET(resCtx.threads.size() < SERVER_RANK_SIZE + 1,
        HCCL_ERROR("Insufficient worker threads for medium direct pull"), HCCL_E_INTERNAL);
    CHK_PRT_RET(dataSize > resCtx.localBuffer.size,
        HCCL_ERROR("Local CCL buffer is too small for medium direct pull"), HCCL_E_INTERNAL);

    uint32_t serverId = param.myRank / SERVER_RANK_SIZE;
    uint32_t localId = param.myRank % SERVER_RANK_SIZE;
    const ChannelInfo *crossChannels[SERVER_RANK_SIZE] = {};
    const ChannelInfo *localChannels[SERVER_RANK_SIZE] = {};

    for (const ChannelInfo &channel : resCtx.channels) {
        CHK_PRT_RET(channel.remoteRank >= param.rankSize,
            HCCL_ERROR("Invalid remote rank %u", channel.remoteRank), HCCL_E_INTERNAL);
        CHK_PRT_RET(dataSize > channel.remoteCclMem.size,
            HCCL_ERROR("Remote CCL buffer is too small for rank %u", channel.remoteRank), HCCL_E_INTERNAL);
        uint32_t remoteLocalId = channel.remoteRank % SERVER_RANK_SIZE;
        if (channel.remoteRank / SERVER_RANK_SIZE == serverId) {
            localChannels[remoteLocalId] = &channel;
        } else {
            crossChannels[remoteLocalId] = &channel;
        }
    }

    for (uint32_t workerIdx = 0; workerIdx < SERVER_RANK_SIZE; ++workerIdx) {
        CHK_PRT_RET(crossChannels[workerIdx] == nullptr,
            HCCL_ERROR("Missing cross-server channel for local rank %u", workerIdx), HCCL_E_INTERNAL);
        CHK_PRT_RET(workerIdx != localId && localChannels[workerIdx] == nullptr,
            HCCL_ERROR("Missing local channel for local rank %u", workerIdx), HCCL_E_INTERNAL);
    }

    // The 512 KiB performance case always fits in one CCL-buffer slice. Publish
    // it once, then use eight workers. Every worker reads one cross-server peer;
    // seven workers subsequently read one local peer on the same thread.
    CHK_RET(static_cast<HcclResult>(
        HcommLocalCopyOnThread(controlThread, resCtx.localBuffer.addr, param.inputPtr, dataSize)));

    // Start only eight communication workers instead of all fifteen.
    for (uint32_t workerIdx = 0; workerIdx < SERVER_RANK_SIZE; ++workerIdx) {
        ThreadHandle workerThread = resCtx.threads[workerIdx + 1];
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(controlThread, workerThread, 0)));
    }
    for (uint32_t workerIdx = 0; workerIdx < SERVER_RANK_SIZE; ++workerIdx) {
        ThreadHandle workerThread = resCtx.threads[workerIdx + 1];
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(workerThread, 0, CUSTOM_TIMEOUT)));
    }

    for (uint32_t workerIdx = 0; workerIdx < SERVER_RANK_SIZE; ++workerIdx) {
        ThreadHandle workerThread = resCtx.threads[workerIdx + 1];
        const ChannelInfo &crossChannel = *crossChannels[workerIdx];
        void *crossOutput = AddOffset(param.outputPtr, crossChannel.remoteRank * dataSize);
        CHK_RET(ReadFromPeer(
            workerThread, crossChannel, crossOutput, crossChannel.remoteCclMem.addr, dataSize));

        if (workerIdx != localId) {
            const ChannelInfo &localChannel = *localChannels[workerIdx];
            void *localOutput = AddOffset(param.outputPtr, localChannel.remoteRank * dataSize);
            CHK_RET(ReadFromPeer(
                workerThread, localChannel, localOutput, localChannel.remoteCclMem.addr, dataSize));
        }
    }

    void *selfOutput = AddOffset(param.outputPtr, param.myRank * dataSize);
    CHK_RET(static_cast<HcclResult>(
        HcommLocalCopyOnThread(controlThread, selfOutput, param.inputPtr, dataSize)));

    // Record worker completion before submitting the matching control waits.
    for (uint32_t workerIdx = 0; workerIdx < SERVER_RANK_SIZE; ++workerIdx) {
        ThreadHandle workerThread = resCtx.threads[workerIdx + 1];
        uint32_t notifyIdx = workerIdx + 1;
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(workerThread, controlThread, notifyIdx)));
    }
    for (uint32_t workerIdx = 0; workerIdx < SERVER_RANK_SIZE; ++workerIdx) {
        uint32_t notifyIdx = workerIdx + 1;
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(controlThread, notifyIdx, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}
// 结束

// 新增
bool CanRunLowLatencyDirectPush(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t dataSize)
{
    if (dataSize != MEDIUM_DATA_SIZE || param.rankSize != EXPECTED_RANK_SIZE ||
        resCtx.channels.size() != param.rankSize - 1 || resCtx.threads.size() < param.rankSize) {
        return false;
    }

    uint64_t scratchSize = dataSize * param.rankSize;
    if (resCtx.localBuffer.size < scratchSize) {
        return false;
    }
    for (const ChannelInfo &channel : resCtx.channels) {
        if (channel.remoteRank >= param.rankSize || channel.threadIdx >= resCtx.threads.size() ||
            channel.remoteCclMem.size < scratchSize) {
            return false;
        }
    }
    return true;
}

HcclResult RunLowLatencyDirectPush(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t dataSize)
{
    ThreadHandle controlThread = resCtx.threads[0];
    uint64_t selfOffset = static_cast<uint64_t>(param.myRank) * dataSize;
    void *selfOutput = AddOffset(param.outputPtr, selfOffset);
    CHK_RET(static_cast<HcclResult>(
        HcommLocalCopyOnThread(controlThread, selfOutput, param.inputPtr, dataSize)));

    uint32_t localServerId = param.myRank / SERVER_RANK_SIZE;
    for (uint32_t phase = 0; phase < 2; ++phase) {
        bool submitCrossServer = phase == 0;
        for (const ChannelInfo &channel : resCtx.channels) {
            bool isCrossServer = channel.remoteRank / SERVER_RANK_SIZE != localServerId;
            if (isCrossServer != submitCrossServer) {
                continue;
            }

            ThreadHandle workerThread = resCtx.threads[channel.threadIdx];
            uint64_t localRankOffset = static_cast<uint64_t>(param.myRank) * dataSize;
            uint64_t peerRankOffset = static_cast<uint64_t>(channel.remoteRank) * dataSize;
            void *remoteDst = AddOffset(channel.remoteCclMem.addr, localRankOffset);
            const void *peerScratch = AddOffset(resCtx.localBuffer.addr, peerRankOffset);
            void *peerOutput = AddOffset(param.outputPtr, peerRankOffset);
            CHK_RET(WriteToPeerNoAck(workerThread, channel, remoteDst, param.inputPtr, dataSize));
            CHK_RET(static_cast<HcclResult>(
                HcommLocalCopyOnThread(workerThread, peerOutput, peerScratch, dataSize)));
        }
    }

    CHK_RET(WaitAllChannelWorkersRecordFirst(resCtx, controlThread));
    return HCCL_SUCCESS;
}
// 结束

// 新增
const ChannelInfo *FindChannelByRank(const AlgResourceCtx &resCtx, uint32_t remoteRank)
{
    for (const ChannelInfo &channel : resCtx.channels) {
        if (channel.remoteRank == remoteRank) {
            return &channel;
        }
    }
    return nullptr;
}

bool CanRunRecursiveDoublingAllGather(const OpParam &param, const AlgResourceCtx &resCtx,
                                      uint64_t dataSize)
{
    if (param.rankSize != EXPECTED_RANK_SIZE || resCtx.channels.size() != param.rankSize - 1) {
        return false;
    }

    uint64_t outputSize = dataSize * param.rankSize;
    if (resCtx.localBuffer.size < outputSize) {
        return false;
    }

    // 删除
    // for (uint32_t distance = 1; distance < SERVER_RANK_SIZE; distance <<= 1) {
    // 新增
    for (uint32_t distance = 1; distance < param.rankSize; distance <<= 1) {
    // 结束
        uint32_t peerRank = param.myRank ^ distance;
        const ChannelInfo *channel = FindChannelByRank(resCtx, peerRank);
        if (channel == nullptr || channel->remoteCclMem.size < outputSize) {
            return false;
        }
    }
    return true;
}

HcclResult RunRecursiveDoublingAllGather(const OpParam &param, const AlgResourceCtx &resCtx,
                                         uint32_t dataTypeSize)
{
    ThreadHandle controlThread = resCtx.threads[0];
    uint64_t dataSize = param.count * dataTypeSize;
    // 删除
    // uint64_t outputSize = dataSize * param.rankSize;
    void *selfScratch = AddOffset(resCtx.localBuffer.addr, param.myRank * dataSize);
    CHK_RET(static_cast<HcclResult>(
        HcommLocalCopyOnThread(controlThread, selfScratch, param.inputPtr, dataSize)));

    // 删除
    // for (uint32_t distance = 1; distance < param.rankSize; distance <<= 1) {
    // 新增
    for (uint32_t distance = 1; distance < SERVER_RANK_SIZE; distance <<= 1) {
    // 结束
        uint32_t peerRank = param.myRank ^ distance;
        const ChannelInfo *channel = FindChannelByRank(resCtx, peerRank);
        CHK_PRT_RET(channel == nullptr,
            HCCL_ERROR("Missing recursive-doubling channel for peer %u", peerRank), HCCL_E_INTERNAL);

        uint32_t localGroupBegin = param.myRank & ~(distance - 1);
        uint64_t groupOffset = static_cast<uint64_t>(localGroupBegin) * dataSize;
        uint64_t groupSize = static_cast<uint64_t>(distance) * dataSize;
        const void *localSrc = AddOffset(resCtx.localBuffer.addr, groupOffset);
        void *remoteDst = AddOffset(channel->remoteCclMem.addr, groupOffset);
        CHK_RET(WriteToPeer(controlThread, *channel, remoteDst, localSrc, groupSize));
    }

    // 删除
    // CHK_RET(static_cast<HcclResult>(
    //     HcommLocalCopyOnThread(controlThread, param.outputPtr, resCtx.localBuffer.addr, outputSize)));
    // 新增
    uint32_t crossPeerRank = param.myRank ^ SERVER_RANK_SIZE;
    const ChannelInfo *crossChannel = FindChannelByRank(resCtx, crossPeerRank);
    CHK_PRT_RET(crossChannel == nullptr,
        HCCL_ERROR("Missing recursive-doubling cross-server channel for peer %u", crossPeerRank),
        HCCL_E_INTERNAL);

    uint32_t localServerBegin = param.myRank / SERVER_RANK_SIZE * SERVER_RANK_SIZE;
    uint32_t remoteServerBegin = crossPeerRank / SERVER_RANK_SIZE * SERVER_RANK_SIZE;
    uint64_t serverDataSize = static_cast<uint64_t>(SERVER_RANK_SIZE) * dataSize;
    uint64_t remoteServerOffset = static_cast<uint64_t>(remoteServerBegin) * dataSize;
    const void *remoteSrc = AddOffset(crossChannel->remoteCclMem.addr, remoteServerOffset);
    void *remoteOutput = AddOffset(param.outputPtr, remoteServerOffset);
    uint64_t localServerOffset = static_cast<uint64_t>(localServerBegin) * dataSize;
    const void *localServerSrc = AddOffset(resCtx.localBuffer.addr, localServerOffset);
    void *localServerOutput = AddOffset(param.outputPtr, localServerOffset);

    // 删除
    // CHK_RET(ReadFromPeer(controlThread, *crossChannel, remoteOutput, remoteSrc, serverDataSize));
    // CHK_RET(static_cast<HcclResult>(
    //     HcommLocalCopyOnThread(controlThread, localServerOutput, localServerSrc, serverDataSize)));
    // 新增
    CHK_PRT_RET(crossChannel->threadIdx >= resCtx.threads.size(),
        HCCL_ERROR("Invalid copy thread index %u", crossChannel->threadIdx), HCCL_E_INTERNAL);
    ThreadHandle localCopyThread = resCtx.threads[crossChannel->threadIdx];
    CHK_RET(StartWorker(controlThread, localCopyThread));
    CHK_RET(static_cast<HcclResult>(
        HcommLocalCopyOnThread(localCopyThread, localServerOutput, localServerSrc, serverDataSize)));
    CHK_RET(ReadFromPeer(controlThread, *crossChannel, remoteOutput, remoteSrc, serverDataSize));
    CHK_RET(WaitWorker(controlThread, localCopyThread, crossChannel->threadIdx));
    // 结束
    // 结束
    return HCCL_SUCCESS;
}
// 结束

HcclResult LaunchMeshChunk(const OpParam &param, const AlgResourceCtx &resCtx,
                           const std::vector<const ChannelInfo *> &meshChannels, ThreadHandle controlThread,
                           uint64_t bankBase, uint64_t processedCount, uint64_t chunkCount,
                           uint32_t dataTypeSize)
{
    uint64_t chunkBytes = chunkCount * dataTypeSize;
    void *localSlice = AddOffset(resCtx.localBuffer.addr, bankBase + param.myRank * chunkBytes);
    const void *inputSlice = AddOffset(param.inputPtr, processedCount * dataTypeSize);
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(controlThread, localSlice, inputSlice, chunkBytes)));

    for (const ChannelInfo *channel : meshChannels) {
        void *remoteDst = AddOffset(channel->remoteCclMem.addr, bankBase + param.myRank * chunkBytes);
        CHK_RET(LaunchChannelWrite(resCtx, controlThread, *channel, remoteDst, localSlice, chunkBytes));
    }
    return HCCL_SUCCESS;
}

HcclResult WaitMeshChunk(const AlgResourceCtx &resCtx, const std::vector<const ChannelInfo *> &meshChannels,
                         ThreadHandle controlThread)
{
    for (const ChannelInfo *channel : meshChannels) {
        CHK_RET(WaitChannelWrite(resCtx, controlThread, *channel));
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchCrossChunk(const OpParam &param, const AlgResourceCtx &resCtx, const ChannelInfo &crossChannel,
                            ThreadHandle controlThread, uint64_t bankBase, uint64_t chunkCount,
                            uint32_t dataTypeSize)
{
    uint64_t chunkBytes = chunkCount * dataTypeSize;
    uint32_t serverBaseRank = param.myRank / SERVER_RANK_SIZE * SERVER_RANK_SIZE;
    uint64_t aggregateOffset = bankBase + serverBaseRank * chunkBytes;
    uint64_t aggregateBytes = SERVER_RANK_SIZE * chunkBytes;
    const void *localSrc = AddOffset(resCtx.localBuffer.addr, aggregateOffset);
    void *remoteDst = AddOffset(crossChannel.remoteCclMem.addr, aggregateOffset);
    return LaunchChannelWrite(resCtx, controlThread, crossChannel, remoteDst, localSrc, aggregateBytes);
}

HcclResult RunHierarchicalAllGather(const OpParam &param, const AlgResourceCtx &resCtx, uint32_t dataTypeSize)
{
    ThreadHandle controlThread = resCtx.threads[0];
    uint32_t serverId = param.myRank / SERVER_RANK_SIZE;
    uint32_t crossPeer = param.myRank ^ SERVER_RANK_SIZE;
    std::vector<const ChannelInfo *> meshChannels;
    const ChannelInfo *crossChannel = nullptr;
    std::vector<const ChannelInfo *> copyThreadChannels;

    for (const ChannelInfo &channel : resCtx.channels) {
        if (channel.remoteRank / SERVER_RANK_SIZE == serverId) {
            meshChannels.push_back(&channel);
        } else if (channel.remoteRank == crossPeer) {
            crossChannel = &channel;
        } else if (copyThreadChannels.size() < 2) {
            // Channels to the other server except the same local rank are not
            // used by the hierarchical algorithm. Reuse two of their worker
            // threads to copy one server's output each.
            copyThreadChannels.push_back(&channel);
        }
    }

    CHK_PRT_RET(meshChannels.size() != SERVER_RANK_SIZE - 1 || crossChannel == nullptr ||
                    copyThreadChannels.size() != 2,
        HCCL_ERROR("Invalid 8x2 topology resources, mesh channel num %u",
            static_cast<uint32_t>(meshChannels.size())), HCCL_E_INTERNAL);
    for (const ChannelInfo *copyThreadChannel : copyThreadChannels) {
        CHK_PRT_RET(copyThreadChannel->threadIdx >= resCtx.threads.size(),
            HCCL_ERROR("Invalid copy thread index %u", copyThreadChannel->threadIdx), HCCL_E_INTERNAL);
    }
    ThreadHandle copyThreads[2] = {resCtx.threads[copyThreadChannels[0]->threadIdx],
                                   resCtx.threads[copyThreadChannels[1]->threadIdx]};
    uint32_t copyNotifyIdx[2] = {copyThreadChannels[0]->threadIdx, copyThreadChannels[1]->threadIdx};

    auto LaunchOutputCopy =
        [&](uint64_t bankBase, uint64_t processedCount, uint64_t chunkCount) -> HcclResult {
        for (uint32_t copyIdx = 0; copyIdx < 2; ++copyIdx) {
            uint32_t sourceRankBegin = copyIdx * SERVER_RANK_SIZE;
            uint32_t sourceRankEnd = sourceRankBegin + SERVER_RANK_SIZE;
            CHK_RET(LaunchCopyChunkToOutputRange(
                param, resCtx, controlThread, copyThreads[copyIdx], sourceRankBegin, sourceRankEnd,
                bankBase, processedCount, chunkCount, dataTypeSize));
        }
        return HCCL_SUCCESS;
    };

    auto WaitOutputCopy = [&]() -> HcclResult {
        for (uint32_t copyIdx = 0; copyIdx < 2; ++copyIdx) {
            CHK_RET(WaitCopyChunk(controlThread, copyThreads[copyIdx], copyNotifyIdx[copyIdx]));
        }
        return HCCL_SUCCESS;
    };

    uint64_t maxChunkBytes = AlignDown(resCtx.localBuffer.size / (2 * param.rankSize), MIN_SLICE_ALIGN);
    maxChunkBytes = std::min(MAX_CHUNK_SIZE, maxChunkBytes);
    CHK_PRT_RET(maxChunkBytes == 0, HCCL_ERROR("CCL buffer is too small for double buffer"), HCCL_E_INTERNAL);
    uint64_t maxChunkCount = maxChunkBytes / dataTypeSize;
    uint64_t bankStride = maxChunkBytes * param.rankSize;
    uint64_t loopCount = param.count / maxChunkCount + static_cast<uint64_t>(param.count % maxChunkCount != 0);

    auto GetProcessedCount = [maxChunkCount](uint64_t loop) { return loop * maxChunkCount; };
    auto GetChunkCount = [&param, maxChunkCount](uint64_t loop) {
        uint64_t processedCount = loop * maxChunkCount;
        return std::min(maxChunkCount, param.count - processedCount);
    };
    auto GetBankBase = [bankStride](uint64_t loop) { return (loop & 1ULL) * bankStride; };

    uint64_t firstChunkCount = GetChunkCount(0);
    CHK_RET(LaunchMeshChunk(
        param, resCtx, meshChannels, controlThread, GetBankBase(0), 0, firstChunkCount, dataTypeSize));
    CHK_RET(WaitMeshChunk(resCtx, meshChannels, controlThread));

    for (uint64_t loop = 1; loop < loopCount; ++loop) {
        uint64_t previousLoop = loop - 1;
        uint64_t previousChunkCount = GetChunkCount(previousLoop);
        CHK_RET(LaunchCrossChunk(param, resCtx, *crossChannel, controlThread, GetBankBase(previousLoop),
                                 previousChunkCount, dataTypeSize));

        // The current mesh operation reuses the bank from two iterations ago.
        // Its asynchronous output copy must finish before that bank is written.
        if (loop >= 2) {
            CHK_RET(WaitOutputCopy());
        }

        uint64_t processedCount = GetProcessedCount(loop);
        uint64_t chunkCount = GetChunkCount(loop);
        CHK_RET(LaunchMeshChunk(param, resCtx, meshChannels, controlThread, GetBankBase(loop), processedCount,
                                chunkCount, dataTypeSize));

        CHK_RET(WaitChannelWrite(resCtx, controlThread, *crossChannel));
        CHK_RET(LaunchOutputCopy(GetBankBase(previousLoop), GetProcessedCount(previousLoop),
                                 previousChunkCount));
        CHK_RET(WaitMeshChunk(resCtx, meshChannels, controlThread));
    }

    uint64_t lastLoop = loopCount - 1;
    uint64_t lastChunkCount = GetChunkCount(lastLoop);
    CHK_RET(LaunchCrossChunk(param, resCtx, *crossChannel, controlThread, GetBankBase(lastLoop),
                             lastChunkCount, dataTypeSize));
    CHK_RET(WaitChannelWrite(resCtx, controlThread, *crossChannel));
    if (loopCount >= 2) {
        CHK_RET(WaitOutputCopy());
    }
    CHK_RET(LaunchOutputCopy(GetBankBase(lastLoop), GetProcessedCount(lastLoop), lastChunkCount));
    CHK_RET(WaitOutputCopy());
    return HCCL_SUCCESS;
}
} // namespace
// 结束

namespace ops_hccl {
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    HCCL_INFO("Executing AICPU Kernel on Ascend NPU");

    // 删除
    // // TODO: 算法任务编排

    // 新增
    auto dataTypeIter = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(dataTypeIter == SIZE_TABLE.end(), HCCL_ERROR("Unsupported data type"), HCCL_E_PARA);
    uint32_t dataTypeSize = dataTypeIter->second;
    uint64_t dataSize = param.count * dataTypeSize;

    CHK_PRT_RET(resCtx.threads.empty(), HCCL_ERROR("No AICPU TS thread resource"), HCCL_E_INTERNAL);
    if (dataSize == 0) {
        return HCCL_SUCCESS;
    }
    if (param.rankSize == 1) {
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(resCtx.threads[0], param.outputPtr, param.inputPtr, dataSize)));
        return HCCL_SUCCESS;
    }

    if (dataSize <= SMALL_DATA_SIZE || param.rankSize != EXPECTED_RANK_SIZE) {
        CHK_RET(RunDirectAllGather(param, resCtx, dataTypeSize));
    // 删除
    // } else {
    // 新增
    } else if (resCtx.channels.size() != param.rankSize - 1) {
        CHK_RET(RunHierarchicalAllGather(param, resCtx, dataTypeSize));
    } else if (dataSize <= MEDIUM_DATA_SIZE) {
        // 删除
        // CHK_RET(RunMediumDirectPullAllGather(param, resCtx, dataTypeSize));
        // 新增
        // 删除
        // if (CanRunRecursiveDoublingAllGather(param, resCtx, dataSize)) {
        //     CHK_RET(RunRecursiveDoublingAllGather(param, resCtx, dataTypeSize));
        // } else {
        //     CHK_RET(RunMediumDirectPullAllGather(param, resCtx, dataTypeSize));
        // }
        // 新增
        if (CanRunLowLatencyDirectPush(param, resCtx, dataSize)) {
            CHK_RET(RunLowLatencyDirectPush(param, resCtx, dataSize));
        } else if (CanRunRecursiveDoublingAllGather(param, resCtx, dataSize)) {
            CHK_RET(RunRecursiveDoublingAllGather(param, resCtx, dataTypeSize));
        } else {
            CHK_RET(RunMediumDirectPullAllGather(param, resCtx, dataTypeSize));
        }
        // 结束
        // 结束
    } else {
        CHK_RET(RunDirectPullAllGather(param, resCtx, dataTypeSize));
    // 结束
    }
    // 结束

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
