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

namespace ops_hccl {
namespace {
constexpr uint32_t ALGORITHM_TWO_DIMENSION = 1;
constexpr uint32_t COMPETITION_RANK_SIZE = 16;
constexpr uint32_t SERVER_RANK_SIZE = 8;
constexpr uint32_t FLAT_THREAD_NUM = 2;
constexpr uint32_t TWO_DIMENSION_THREAD_NUM = SERVER_RANK_SIZE;
constexpr uint32_t PIPELINE_MAX_TILE_NUM = 13;
constexpr uint64_t DEFAULT_SLICE_TILE_SIZE = 8ULL * 1024 * 1024;
constexpr uint64_t DEEP_PIPELINE_TILE_SIZE = 4ULL * 1024 * 1024;
constexpr uint64_t FAST_TAIL_THRESHOLD = 1ULL * 1024 * 1024;
constexpr uint64_t DEEP_PIPELINE_DATA_LIMIT = 400ULL * 1024 * 1024 + sizeof(float);
constexpr uint32_t NOTIFY_STAGE_READY_BASE = 0;
constexpr uint32_t NOTIFY_ALLGATHER_READY_BASE = PIPELINE_MAX_TILE_NUM;
constexpr uint32_t NOTIFY_TAIL_READY = PIPELINE_MAX_TILE_NUM * 2;
constexpr uint32_t NOTIFY_CHUNK_DONE = PIPELINE_MAX_TILE_NUM * 2 + 1;

struct DataSlice {
    uint64_t offset = 0;
    uint64_t size = 0;
};

const ChannelInfo *FindChannel(const AlgResourceCtx &resCtx, uint32_t remoteRank)
{
    for (const ChannelInfo &channel : resCtx.channels) {
        if (channel.remoteRank == remoteRank) {
            return &channel;
        }
    }
    return nullptr;
}

uint32_t GetServerBase(uint32_t rank)
{
    return rank / SERVER_RANK_SIZE * SERVER_RANK_SIZE;
}

DataSlice CalcSlice(uint64_t chunkSize, uint32_t localIndex)
{
    const uint64_t chunkCount = chunkSize / sizeof(float);
    const uint64_t baseCount = chunkCount / SERVER_RANK_SIZE;
    const uint64_t remainder = chunkCount % SERVER_RANK_SIZE;
    const uint64_t prefixRemainder = localIndex < remainder ? localIndex : remainder;
    const uint64_t sliceCount = baseCount + (localIndex < remainder ? 1 : 0);
    return DataSlice{
        (localIndex * baseCount + prefixRemainder) * sizeof(float),
        sliceCount * sizeof(float),
    };
}

HcclResult ThreadSyncBefore(const std::vector<ThreadHandle> &threads)
{
    for (uint32_t idx = 1; idx < threads.size(); ++idx) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[0], threads[idx], 1)));
    }
    for (uint32_t idx = 1; idx < threads.size(); ++idx) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(threads[idx], 1, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult ThreadSyncAfter(const std::vector<ThreadHandle> &threads)
{
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

HcclResult WriteOrNotify(ThreadHandle thread, const ChannelInfo &channel, void *dst, const void *src,
    uint64_t size, uint32_t notifyIdx)
{
    if (size == 0) {
        // 零长度任务的发送端和接收端都会跳过，禁止留下无人消费的Notify。
        return HCCL_SUCCESS;
    }
    return static_cast<HcclResult>(
        HcommWriteWithNotifyOnThread(thread, channel.handle, dst, src, size, notifyIdx));
}

HcclResult ExecFlat(const OpParam &param, const AlgResourceCtx &resCtx)
{
    if (resCtx.threads.size() != FLAT_THREAD_NUM) {
        HCCL_ERROR("[ExecFlat] Invalid thread count %zu, expected %u",
            resCtx.threads.size(), FLAT_THREAD_NUM);
        return HCCL_E_INTERNAL;
    }

    const bool isRoot = param.myRank == param.root;
    const ChannelInfo *rootChannel = nullptr;
    uint64_t chunkCapacity = 0;
    if (isRoot) {
        if (resCtx.channels.size() != param.rankSize - 1) {
            HCCL_ERROR("[ExecFlat] Root rank %u has %zu channels, expected %u",
                param.root, resCtx.channels.size(), param.rankSize - 1);
            return HCCL_E_INTERNAL;
        }
        chunkCapacity = resCtx.localBuffer.size;
    } else {
        rootChannel = FindChannel(resCtx, param.root);
        if (rootChannel == nullptr) {
            HCCL_ERROR("[ExecFlat] Channel to root rank %u was not found", param.root);
            return HCCL_E_NOT_FOUND;
        }
        chunkCapacity = rootChannel->remoteCclMem.size;
    }
    if (chunkCapacity == 0) {
        HCCL_ERROR("[ExecFlat] Root HCCL buffer size is zero");
        return HCCL_E_INTERNAL;
    }

    const uint64_t dataSize = param.count * sizeof(float);
    const ThreadHandle thread = resCtx.threads[0];
    uint8_t *userBufferBase = static_cast<uint8_t *>(isRoot ? param.inputPtr : param.outputPtr);
    for (uint64_t offset = 0; offset < dataSize;) {
        const uint64_t remaining = dataSize - offset;
        const uint64_t chunkSize = remaining < chunkCapacity ? remaining : chunkCapacity;
        void *userBuffer = static_cast<void *>(userBufferBase + offset);

        if (isRoot) {
            CHK_RET(static_cast<HcclResult>(
                HcommLocalCopyOnThread(thread, resCtx.localBuffer.addr, userBuffer, chunkSize)));
            // Copy 完成后同时唤醒两个通信队列，将15个接收方均匀分摊到两个 Thread。
            CHK_RET(ThreadSyncBefore(resCtx.threads));
            for (size_t channelIdx = 0; channelIdx < resCtx.channels.size(); ++channelIdx) {
                const ChannelInfo &channel = resCtx.channels[channelIdx];
                const ThreadHandle channelThread = resCtx.threads[channelIdx % FLAT_THREAD_NUM];
                CHK_RET(static_cast<HcclResult>(
                    HcommChannelNotifyRecordOnThread(channelThread, channel.handle, NOTIFY_IDX_ACK)));
            }
            // 先放行所有接收端，再等待完成，避免同一 Thread 上不同接收方被串行化。
            for (size_t channelIdx = 0; channelIdx < resCtx.channels.size(); ++channelIdx) {
                const ChannelInfo &channel = resCtx.channels[channelIdx];
                const ThreadHandle channelThread = resCtx.threads[channelIdx % FLAT_THREAD_NUM];
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                    channelThread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
            }
            CHK_RET(ThreadSyncAfter(resCtx.threads));
        } else {
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                thread, rootChannel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(HcommReadOnThread(
                thread, rootChannel->handle, userBuffer, rootChannel->remoteCclMem.addr, chunkSize)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
                thread, rootChannel->handle, NOTIFY_IDX_DATA_SIGNAL)));
        }
        offset += chunkSize;
    }
    return HCCL_SUCCESS;
}

DataSlice CalcSliceTile(const DataSlice &slice, uint64_t tileIndex, uint64_t tileSize)
{
    const uint64_t tileOffset = tileIndex * tileSize;
    if (tileOffset >= slice.size) {
        return DataSlice{slice.offset + slice.size, 0};
    }

    const uint64_t remaining = slice.size - tileOffset;
    const uint64_t currentTileSize = remaining < tileSize ? remaining : tileSize;
    return DataSlice{slice.offset + tileOffset, currentTileSize};
}

uint64_t CalcCompactCclOffset(const DataSlice &slice, const DataSlice &tile)
{
    return tile.offset - slice.offset;
}

uint64_t CalcTileCount(uint64_t chunkSize, uint64_t tileSize)
{
    const uint64_t maxSliceSize = CalcSlice(chunkSize, 0).size;
    return maxSliceSize == 0 ? 0 : (maxSliceSize - 1) / tileSize + 1;
}

uint64_t CalcCompactChunkCapacity(uint64_t bufferCapacity, uint64_t tileSize)
{
    const uint64_t notifyLimitedSliceCapacity = tileSize * PIPELINE_MAX_TILE_NUM;
    const uint64_t sliceCapacity = bufferCapacity < notifyLimitedSliceCapacity ?
        bufferCapacity : notifyLimitedSliceCapacity;
    return sliceCapacity * SERVER_RANK_SIZE;
}

HcclResult PublishOwnedTileReady(
    const OpParam &param, const AlgResourceCtx &resCtx, ThreadHandle ownerThread, uint32_t notifyIdx)
{
    const uint32_t serverBase = GetServerBase(param.myRank);
    for (uint32_t localIdx = 0; localIdx < SERVER_RANK_SIZE; ++localIdx) {
        const uint32_t remoteRank = serverBase + localIdx;
        // root 的用户 Buffer 已经持有完整输入，不需要参与 AllGather。
        if (remoteRank == param.myRank || remoteRank == param.root) {
            continue;
        }

        const ChannelInfo *channel = FindChannel(resCtx, remoteRank);
        if (channel == nullptr) {
            HCCL_ERROR("[PublishOwnedTileReady] Channel to rank %u was not found", remoteRank);
            return HCCL_E_NOT_FOUND;
        }
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
            ownerThread, channel->handle, notifyIdx)));
    }
    return HCCL_SUCCESS;
}

HcclResult ScheduleRootScatterTile(const OpParam &param, const AlgResourceCtx &resCtx,
    uint8_t *rootUserChunk, uint64_t chunkSize, uint64_t tileIndex, uint64_t tileSize)
{
    if (param.myRank != param.root) {
        return HCCL_SUCCESS;
    }
    const uint32_t stageNotifyIdx = NOTIFY_STAGE_READY_BASE + static_cast<uint32_t>(tileIndex);

    const uint32_t rootServerBase = GetServerBase(param.root);
    for (uint32_t localIdx = 0; localIdx < SERVER_RANK_SIZE; ++localIdx) {
        const uint32_t ownerRank = rootServerBase + localIdx;
        const DataSlice slice = CalcSlice(chunkSize, localIdx);
        const DataSlice tile = CalcSliceTile(slice, tileIndex, tileSize);
        if (tile.size == 0) {
            continue;
        }
        const uint64_t cclOffset = CalcCompactCclOffset(slice, tile);
        const ThreadHandle sliceThread = resCtx.threads[localIdx];

        if (ownerRank == param.root) {
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
                sliceThread,
                static_cast<uint8_t *>(resCtx.localBuffer.addr) + cclOffset,
                rootUserChunk + tile.offset,
                tile.size)));
            continue;
        }

        const ChannelInfo *channel = FindChannel(resCtx, ownerRank);
        if (channel == nullptr) {
            HCCL_ERROR("[ScheduleRootScatterTile] Channel to rank %u was not found", ownerRank);
            return HCCL_E_NOT_FOUND;
        }
        CHK_RET(WriteOrNotify(
            sliceThread,
            *channel,
            static_cast<uint8_t *>(channel->remoteCclMem.addr) + cclOffset,
            rootUserChunk + tile.offset,
            tile.size,
            stageNotifyIdx));
    }
    return HCCL_SUCCESS;
}

HcclResult ScheduleOwnedTile(const OpParam &param, const AlgResourceCtx &resCtx,
    uint8_t *userChunk, uint64_t chunkSize, uint64_t tileIndex, uint64_t tileSize)
{
    const uint32_t rootServerBase = GetServerBase(param.root);
    const uint32_t myServerBase = GetServerBase(param.myRank);
    const uint32_t myLocalIndex = param.myRank % SERVER_RANK_SIZE;
    const ThreadHandle ownerThread = resCtx.threads[myLocalIndex];
    const DataSlice ownedSlice = CalcSlice(chunkSize, myLocalIndex);
    const DataSlice ownedTile = CalcSliceTile(ownedSlice, tileIndex, tileSize);
    if (ownedTile.size == 0) {
        return HCCL_SUCCESS;
    }
    const uint64_t cclOffset = CalcCompactCclOffset(ownedSlice, ownedTile);
    const uint32_t stageNotifyIdx = NOTIFY_STAGE_READY_BASE + static_cast<uint32_t>(tileIndex);
    const uint32_t allGatherNotifyIdx = NOTIFY_ALLGATHER_READY_BASE + static_cast<uint32_t>(tileIndex);

    const uint32_t crossRank = param.myRank ^ SERVER_RANK_SIZE;
    const ChannelInfo *crossChannel = FindChannel(resCtx, crossRank);
    if (crossChannel == nullptr) {
        HCCL_ERROR("[ScheduleOwnedTile] Channel to cross rank %u was not found", crossRank);
        return HCCL_E_NOT_FOUND;
    }

    if (param.myRank != param.root) {
        const uint32_t sourceRank = myServerBase == rootServerBase ? param.root : crossRank;
        const ChannelInfo *sourceChannel = FindChannel(resCtx, sourceRank);
        if (sourceChannel == nullptr) {
            HCCL_ERROR("[ScheduleOwnedTile] Channel to source rank %u was not found", sourceRank);
            return HCCL_E_NOT_FOUND;
        }
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            ownerThread, sourceChannel->handle, stageNotifyIdx, CUSTOM_TIMEOUT)));
    }

    // 本Server的读端一旦发现当前Tile就绪即可工作，不再等待其他Owner。
    CHK_RET(PublishOwnedTileReady(param, resCtx, ownerThread, allGatherNotifyIdx));

    if (myServerBase == rootServerBase) {
        CHK_RET(WriteOrNotify(
            ownerThread,
            *crossChannel,
            static_cast<uint8_t *>(crossChannel->remoteCclMem.addr) + cclOffset,
            static_cast<uint8_t *>(resCtx.localBuffer.addr) + cclOffset,
            ownedTile.size,
            stageNotifyIdx));
    }

    if (param.myRank != param.root && ownedTile.size > 0) {
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
            ownerThread,
            userChunk + ownedTile.offset,
            static_cast<uint8_t *>(resCtx.localBuffer.addr) + cclOffset,
            ownedTile.size)));
    }
    return HCCL_SUCCESS;
}

HcclResult ScheduleRemoteTileReads(const OpParam &param, const AlgResourceCtx &resCtx,
    uint8_t *userChunk, uint64_t chunkSize, uint64_t tileIndex, uint64_t tileSize)
{
    if (param.myRank == param.root) {
        return HCCL_SUCCESS;
    }

    const uint32_t serverBase = GetServerBase(param.myRank);
    const uint32_t allGatherNotifyIdx = NOTIFY_ALLGATHER_READY_BASE + static_cast<uint32_t>(tileIndex);
    const uint32_t myLocalIndex = param.myRank % SERVER_RANK_SIZE;
    for (uint32_t localIdx = 0; localIdx < SERVER_RANK_SIZE; ++localIdx) {
        if (localIdx == myLocalIndex) {
            continue;
        }

        const DataSlice ownerSlice = CalcSlice(chunkSize, localIdx);
        const DataSlice tile = CalcSliceTile(ownerSlice, tileIndex, tileSize);
        if (tile.size == 0) {
            continue;
        }
        const uint64_t cclOffset = CalcCompactCclOffset(ownerSlice, tile);
        const uint32_t ownerRank = serverBase + localIdx;
        const ChannelInfo *channel = FindChannel(resCtx, ownerRank);
        if (channel == nullptr) {
            HCCL_ERROR("[ScheduleRemoteTileReads] Channel to rank %u was not found", ownerRank);
            return HCCL_E_NOT_FOUND;
        }

        const ThreadHandle readThread = resCtx.threads[localIdx];
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            readThread, channel->handle, allGatherNotifyIdx, CUSTOM_TIMEOUT)));
        if (tile.size > 0) {
            CHK_RET(static_cast<HcclResult>(HcommReadOnThread(
                readThread,
                channel->handle,
                userChunk + tile.offset,
                static_cast<uint8_t *>(channel->remoteCclMem.addr) + cclOffset,
                tile.size)));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult SchedulePipelinedTile(const OpParam &param, const AlgResourceCtx &resCtx,
    uint8_t *userChunk, uint64_t chunkSize, uint64_t tileIndex, uint64_t tileSize)
{
    CHK_RET(ScheduleRootScatterTile(param, resCtx, userChunk, chunkSize, tileIndex, tileSize));
    CHK_RET(ScheduleOwnedTile(param, resCtx, userChunk, chunkSize, tileIndex, tileSize));
    CHK_RET(ScheduleRemoteTileReads(param, resCtx, userChunk, chunkSize, tileIndex, tileSize));
    return HCCL_SUCCESS;
}

HcclResult ChunkCompletionBarrier(const OpParam &param, const AlgResourceCtx &resCtx)
{
    const uint32_t rootServerBase = GetServerBase(param.root);
    const uint32_t myServerBase = GetServerBase(param.myRank);
    const uint32_t relay = param.root ^ SERVER_RANK_SIZE;

    if (param.myRank == param.root) {
        for (uint32_t localIdx = 0; localIdx < SERVER_RANK_SIZE; ++localIdx) {
            const uint32_t childRank = rootServerBase + localIdx;
            if (childRank == param.root) {
                continue;
            }
            const ChannelInfo *channel = FindChannel(resCtx, childRank);
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                resCtx.threads[0], channel->handle, NOTIFY_CHUNK_DONE, CUSTOM_TIMEOUT)));
        }
        const ChannelInfo *relayChannel = FindChannel(resCtx, relay);
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            resCtx.threads[0], relayChannel->handle, NOTIFY_CHUNK_DONE, CUSTOM_TIMEOUT)));
    } else if (param.myRank == relay) {
        for (uint32_t localIdx = 0; localIdx < SERVER_RANK_SIZE; ++localIdx) {
            const uint32_t childRank = myServerBase + localIdx;
            if (childRank == relay) {
                continue;
            }
            const ChannelInfo *channel = FindChannel(resCtx, childRank);
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                resCtx.threads[0], channel->handle, NOTIFY_CHUNK_DONE, CUSTOM_TIMEOUT)));
        }
        const ChannelInfo *rootChannel = FindChannel(resCtx, param.root);
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
            resCtx.threads[0], rootChannel->handle, NOTIFY_CHUNK_DONE)));
    } else {
        const uint32_t parent = myServerBase == rootServerBase ? param.root : relay;
        const ChannelInfo *parentChannel = FindChannel(resCtx, parent);
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(
            resCtx.threads[0], parentChannel->handle, NOTIFY_CHUNK_DONE)));
    }
    return HCCL_SUCCESS;
}

// 极小尾块不值得再启动完整的Scatter-Cross-AllGather。
// root先扇出到本Server和对端relay，再由relay在远端Server内扇出。
HcclResult ExecFastTail(const OpParam &param, const AlgResourceCtx &resCtx,
    uint8_t *userTail, uint64_t tailSize)
{
    const uint32_t rootServerBase = GetServerBase(param.root);
    const uint32_t myServerBase = GetServerBase(param.myRank);
    const uint32_t relay = param.root ^ SERVER_RANK_SIZE;
    const ThreadHandle thread = resCtx.threads[0];

    if (param.myRank == param.root) {
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(thread, resCtx.localBuffer.addr, userTail, tailSize)));
        for (uint32_t localIdx = 0; localIdx < SERVER_RANK_SIZE; ++localIdx) {
            const uint32_t childRank = rootServerBase + localIdx;
            if (childRank == param.root) {
                continue;
            }
            const ChannelInfo *channel = FindChannel(resCtx, childRank);
            if (channel == nullptr) {
                HCCL_ERROR("[ExecFastTail] Channel to local child rank %u was not found", childRank);
                return HCCL_E_NOT_FOUND;
            }
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(thread, channel->handle, NOTIFY_TAIL_READY)));
        }
        const ChannelInfo *relayChannel = FindChannel(resCtx, relay);
        if (relayChannel == nullptr) {
            HCCL_ERROR("[ExecFastTail] Channel to relay rank %u was not found", relay);
            return HCCL_E_NOT_FOUND;
        }
        CHK_RET(WriteOrNotify(thread, *relayChannel, relayChannel->remoteCclMem.addr,
            resCtx.localBuffer.addr, tailSize, NOTIFY_TAIL_READY));
    } else if (param.myRank == relay) {
        const ChannelInfo *rootChannel = FindChannel(resCtx, param.root);
        if (rootChannel == nullptr) {
            HCCL_ERROR("[ExecFastTail] Channel to root rank %u was not found", param.root);
            return HCCL_E_NOT_FOUND;
        }
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            thread, rootChannel->handle, NOTIFY_TAIL_READY, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(thread, userTail, resCtx.localBuffer.addr, tailSize)));
        for (uint32_t localIdx = 0; localIdx < SERVER_RANK_SIZE; ++localIdx) {
            const uint32_t childRank = myServerBase + localIdx;
            if (childRank == relay) {
                continue;
            }
            const ChannelInfo *channel = FindChannel(resCtx, childRank);
            if (channel == nullptr) {
                HCCL_ERROR("[ExecFastTail] Channel to remote child rank %u was not found", childRank);
                return HCCL_E_NOT_FOUND;
            }
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(thread, channel->handle, NOTIFY_TAIL_READY)));
        }
    } else {
        const uint32_t parent = myServerBase == rootServerBase ? param.root : relay;
        const ChannelInfo *parentChannel = FindChannel(resCtx, parent);
        if (parentChannel == nullptr) {
            HCCL_ERROR("[ExecFastTail] Channel to parent rank %u was not found", parent);
            return HCCL_E_NOT_FOUND;
        }
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            thread, parentChannel->handle, NOTIFY_TAIL_READY, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(HcommReadOnThread(
            thread, parentChannel->handle, userTail, parentChannel->remoteCclMem.addr, tailSize)));
    }
    return ChunkCompletionBarrier(param, resCtx);
}

HcclResult ExecTwoDimension(const OpParam &param, const AlgResourceCtx &resCtx)
{
    if (param.rankSize != COMPETITION_RANK_SIZE || resCtx.threads.size() != TWO_DIMENSION_THREAD_NUM ||
        resCtx.channels.size() != SERVER_RANK_SIZE) {
        HCCL_ERROR("[ExecTwoDimension] Invalid resources: rankSize %u, threads %zu, channels %zu",
            param.rankSize, resCtx.threads.size(), resCtx.channels.size());
        return HCCL_E_INTERNAL;
    }

    const uint64_t alignedBufferCapacity = resCtx.localBuffer.size / sizeof(float) * sizeof(float);
    if (alignedBufferCapacity == 0) {
        HCCL_ERROR("[ExecTwoDimension] HCCL buffer size is zero");
        return HCCL_E_INTERNAL;
    }

    const uint64_t dataSize = param.count * sizeof(float);
    const bool useDeepPipeline = dataSize <= DEEP_PIPELINE_DATA_LIMIT;
    const uint64_t tileSize = useDeepPipeline ? DEEP_PIPELINE_TILE_SIZE : DEFAULT_SLICE_TILE_SIZE;
    const uint64_t compactChunkCapacity = CalcCompactChunkCapacity(alignedBufferCapacity, tileSize);
    const uint64_t deepPipelineChunkCapacity = 400ULL * 1024 * 1024;
    const uint64_t chunkCapacity = useDeepPipeline && compactChunkCapacity > deepPipelineChunkCapacity ?
        deepPipelineChunkCapacity : compactChunkCapacity;
    if (chunkCapacity == 0) {
        HCCL_ERROR("[ExecTwoDimension] Compact chunk capacity is zero");
        return HCCL_E_INTERNAL;
    }
    uint8_t *userBufferBase = static_cast<uint8_t *>(
        param.myRank == param.root ? param.inputPtr : param.outputPtr);
    for (uint64_t offset = 0; offset < dataSize;) {
        const uint64_t remaining = dataSize - offset;
        if (remaining <= FAST_TAIL_THRESHOLD) {
            CHK_RET(ExecFastTail(param, resCtx, userBufferBase + offset, remaining));
            offset += remaining;
            continue;
        }
        const uint64_t chunkSize = remaining < chunkCapacity ? remaining : chunkCapacity;
        uint8_t *userChunk = userBufferBase + offset;

        const uint64_t maxSliceSize = CalcSlice(chunkSize, 0).size;
        if (maxSliceSize > alignedBufferCapacity) {
            HCCL_ERROR("[ExecTwoDimension] Slice size %llu exceeds HCCL buffer capacity %llu",
                static_cast<unsigned long long>(maxSliceSize),
                static_cast<unsigned long long>(alignedBufferCapacity));
            return HCCL_E_INTERNAL;
        }

        // 每个Rank只紧凑保存自己的Slice，用户Buffer仍使用Chunk内的全局Slice偏移。
        // 不同Tile占用Slice内的独立区间，可连续下发；整个Chunk结束时再统一汇合。
        const uint64_t tileCount = CalcTileCount(chunkSize, tileSize);
        if (tileCount > PIPELINE_MAX_TILE_NUM) {
            HCCL_ERROR("[ExecTwoDimension] Tile count %llu exceeds notify capacity %u",
                static_cast<unsigned long long>(tileCount), PIPELINE_MAX_TILE_NUM);
            return HCCL_E_INTERNAL;
        }
        // Checker只会自动启动Thread 0，必须由主Thread显式放行其余7条工作队列。
        // 这里只在Chunk入口同步一次，Tile之间仍保持连续流水。
        CHK_RET(ThreadSyncBefore(resCtx.threads));
        for (uint64_t tileIndex = 0; tileIndex < tileCount; ++tileIndex) {
            CHK_RET(SchedulePipelinedTile(param, resCtx, userChunk, chunkSize, tileIndex, tileSize));
        }
        CHK_RET(ThreadSyncAfter(resCtx.threads));
        CHK_RET(ChunkCompletionBarrier(param, resCtx));
        offset += chunkSize;
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    HCCL_INFO("Executing AICPU Kernel on Ascend NPU");

    if (param.count == 0 || param.rankSize == 1) {
        return HCCL_SUCCESS;
    }
    if (resCtx.threads.empty()) {
        HCCL_ERROR("[ExecOp] No AICPU thread is available");
        return HCCL_E_INTERNAL;
    }
    if (resCtx.algorithmMode == ALGORITHM_TWO_DIMENSION) {
        return ExecTwoDimension(param, resCtx);
    }
    return ExecFlat(param, resCtx);
}
} // namespace ops_hccl
