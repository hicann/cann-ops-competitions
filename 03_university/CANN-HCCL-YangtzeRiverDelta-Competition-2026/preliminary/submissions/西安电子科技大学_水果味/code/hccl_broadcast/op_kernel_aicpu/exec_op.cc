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

namespace {

constexpr uint32_t LAYER_0 = 0;
constexpr uint32_t EXPECTED_LOCAL_RANK_SIZE = 8;
constexpr uint32_t EXPECTED_RANK_SIZE = 2 * EXPECTED_LOCAL_RANK_SIZE;
constexpr uint32_t CHANNEL_NOTIFY_IDX_ACK = 0;
constexpr uint32_t CHANNEL_NOTIFY_IDX_DATA = 1;
constexpr uint64_t FLAT_BROADCAST_THRESHOLD = 64 * 1024;
constexpr uint64_t MAX_TILE_SIZE = 256 * 1024 * 1024;
constexpr uint64_t ALIGN_COUNT_FP32 = 32; // 32个float32为128B
constexpr uint64_t MIN_SHARDED_TILE_COUNT = EXPECTED_LOCAL_RANK_SIZE * ALIGN_COUNT_FP32;

enum class TransferMode : uint32_t {
    SEND,
    RECEIVE,
    EXCHANGE,
};

struct TransferTask {
    const ChannelInfo *channel = nullptr;
    const void *src = nullptr;
    void *dst = nullptr;
    uint64_t size = 0;
    TransferMode mode = TransferMode::SEND;
};

struct ChunkInfo {
    uint64_t offset = 0;
    uint64_t size = 0;
};

void *AddOffset(void *ptr, uint64_t offset)
{
    return static_cast<void *>(static_cast<uint8_t *>(ptr) + offset);
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

HcclResult StartParallelThreads(const std::vector<ThreadHandle> &threads, uint32_t activeThreadNum)
{
    CHK_PRT_RET(activeThreadNum == 0 || activeThreadNum > threads.size(),
        HCCL_ERROR("Invalid active thread num[%u], acquired thread num[%zu]", activeThreadNum, threads.size()),
        HCCL_E_INTERNAL);

    for (uint32_t i = 1; i < activeThreadNum; ++i) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[0], threads[i], 0)));
    }
    for (uint32_t i = 1; i < activeThreadNum; ++i) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[i], 0, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult FinishParallelThreads(const std::vector<ThreadHandle> &threads, uint32_t activeThreadNum)
{
    // notify[0]由Host/AICPU同步使用，因此从thread回主thread使用notify[1..14]。
    for (uint32_t i = 1; i < activeThreadNum; ++i) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[0], i, CUSTOM_TIMEOUT)));
    }
    for (uint32_t i = 1; i < activeThreadNum; ++i) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[i], threads[0], i)));
    }
    return HCCL_SUCCESS;
}

HcclResult RunTransferTask(ThreadHandle thread, const TransferTask &task)
{
    CHK_PTR_NULL(task.channel);
    if (task.mode == TransferMode::SEND) {
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, task.channel->handle, CHANNEL_NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(
            HcommWriteOnThread(thread, task.channel->handle, task.dst, task.src, task.size)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, task.channel->handle, CHANNEL_NOTIFY_IDX_DATA)));
        return HCCL_SUCCESS;
    }

    if (task.mode == TransferMode::RECEIVE) {
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, task.channel->handle, CHANNEL_NOTIFY_IDX_ACK)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, task.channel->handle, CHANNEL_NOTIFY_IDX_DATA, CUSTOM_TIMEOUT)));
        return HCCL_SUCCESS;
    }

    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, task.channel->handle, CHANNEL_NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, task.channel->handle, CHANNEL_NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(
        HcommWriteOnThread(thread, task.channel->handle, task.dst, task.src, task.size)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, task.channel->handle, CHANNEL_NOTIFY_IDX_DATA)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, task.channel->handle, CHANNEL_NOTIFY_IDX_DATA, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult RunParallelTasks(const AlgResourceCtx &resCtx, const std::vector<TransferTask> &tasks)
{
    CHK_PRT_RET(tasks.empty(), HCCL_ERROR("Transfer task list is empty"), HCCL_E_INTERNAL);
    uint32_t activeThreadNum = static_cast<uint32_t>(tasks.size());
    CHK_RET(StartParallelThreads(resCtx.threads, activeThreadNum));
    for (uint32_t i = 0; i < activeThreadNum; ++i) {
        CHK_RET(RunTransferTask(resCtx.threads[i], tasks[i]));
    }
    CHK_RET(FinishParallelThreads(resCtx.threads, activeThreadNum));
    return HCCL_SUCCESS;
}

uint64_t GetMinCclBufferSize(const AlgResourceCtx &resCtx)
{
    uint64_t minSize = resCtx.localBuffer.size;
    for (const ChannelInfo &channel : resCtx.channels) {
        minSize = std::min(minSize, channel.remoteCclMem.size);
    }
    return minSize;
}

HcclResult BuildChunks(uint64_t tileCount, uint32_t localRankSize, std::vector<ChunkInfo> &chunks)
{
    CHK_PRT_RET(localRankSize != EXPECTED_LOCAL_RANK_SIZE,
        HCCL_ERROR("Unexpected local rank size[%u]", localRankSize), HCCL_E_INTERNAL);

    uint64_t regularCount = tileCount / localRankSize;
    regularCount = regularCount / ALIGN_COUNT_FP32 * ALIGN_COUNT_FP32;
    CHK_PRT_RET(regularCount == 0,
        HCCL_ERROR("Tile count[%llu] is too small for sharded broadcast", static_cast<unsigned long long>(tileCount)),
        HCCL_E_INTERNAL);

    chunks.resize(localRankSize);
    uint64_t offsetCount = 0;
    for (uint32_t i = 0; i < localRankSize; ++i) {
        uint64_t chunkCount = i == localRankSize - 1 ? tileCount - offsetCount : regularCount;
        chunks[i].offset = offsetCount * sizeof(float);
        chunks[i].size = chunkCount * sizeof(float);
        offsetCount += chunkCount;
    }
    return HCCL_SUCCESS;
}

HcclResult RunFlatBroadcast(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t dataSize)
{
    if (param.myRank == param.root) {
        std::vector<TransferTask> tasks;
        tasks.reserve(resCtx.channels.size());
        for (const ChannelInfo &channel : resCtx.channels) {
            tasks.push_back(TransferTask{&channel, param.inputPtr, channel.remoteCclMem.addr, dataSize,
                TransferMode::SEND});
        }
        CHK_RET(RunParallelTasks(resCtx, tasks));
        return HCCL_SUCCESS;
    }

    const ChannelInfo *rootChannel = FindChannel(resCtx, param.root);
    CHK_PRT_RET(rootChannel == nullptr,
        HCCL_ERROR("Channel to root[%u] was not found on rank[%u]", param.root, param.myRank), HCCL_E_INTERNAL);
    std::vector<TransferTask> tasks = {
        TransferTask{rootChannel, nullptr, resCtx.localBuffer.addr, dataSize, TransferMode::RECEIVE}};
    CHK_RET(RunParallelTasks(resCtx, tasks));
    CHK_RET(static_cast<HcclResult>(
        HcommLocalCopyOnThread(resCtx.threads[0], param.outputPtr, resCtx.localBuffer.addr, dataSize)));
    return HCCL_SUCCESS;
}

HcclResult RunScatterPhase(const OpParam &param, const AlgResourceCtx &resCtx,
    const std::vector<ChunkInfo> &chunks, uint64_t userOffset)
{
    if (param.myRank == param.root) {
        std::vector<TransferTask> tasks;
        tasks.reserve(resCtx.channels.size());
        for (const ChannelInfo &channel : resCtx.channels) {
            CHK_PRT_RET(channel.remoteLocalIndex >= chunks.size(),
                HCCL_ERROR("Invalid remote local index[%u]", channel.remoteLocalIndex), HCCL_E_INTERNAL);
            const ChunkInfo &chunk = chunks[channel.remoteLocalIndex];
            const void *src = AddOffset(param.inputPtr, userOffset + chunk.offset);
            void *dst = AddOffset(channel.remoteCclMem.addr, chunk.offset);
            tasks.push_back(TransferTask{&channel, src, dst, chunk.size, TransferMode::SEND});
        }
        return RunParallelTasks(resCtx, tasks);
    }

    const ChannelInfo *rootChannel = FindChannel(resCtx, param.root);
    CHK_PRT_RET(rootChannel == nullptr,
        HCCL_ERROR("Channel to root[%u] was not found on rank[%u]", param.root, param.myRank), HCCL_E_INTERNAL);
    const ChunkInfo &ownChunk = chunks[resCtx.localIndex];
    std::vector<TransferTask> tasks = {TransferTask{rootChannel, nullptr,
        AddOffset(resCtx.localBuffer.addr, ownChunk.offset), ownChunk.size, TransferMode::RECEIVE}};
    return RunParallelTasks(resCtx, tasks);
}

HcclResult RunIntraServerAllGather(const OpParam &param, const AlgResourceCtx &resCtx,
    const std::vector<ChunkInfo> &chunks, uint64_t userOffset)
{
    const ChunkInfo &ownChunk = chunks[resCtx.localIndex];
    const void *src = param.myRank == param.root ? AddOffset(param.inputPtr, userOffset + ownChunk.offset)
                                                 : AddOffset(resCtx.localBuffer.addr, ownChunk.offset);

    std::vector<TransferTask> tasks;
    tasks.reserve(resCtx.localRankSize - 1);
    for (const ChannelInfo &channel : resCtx.channels) {
        if (channel.netLayer != LAYER_0) {
            continue;
        }
        void *dst = AddOffset(channel.remoteCclMem.addr, ownChunk.offset);
        tasks.push_back(TransferTask{&channel, src, dst, ownChunk.size, TransferMode::EXCHANGE});
    }
    CHK_PRT_RET(tasks.size() != resCtx.localRankSize - 1,
        HCCL_ERROR("Layer-0 channel num[%zu] is not expected[%u]", tasks.size(), resCtx.localRankSize - 1),
        HCCL_E_INTERNAL);
    return RunParallelTasks(resCtx, tasks);
}

HcclResult RunShardedBroadcast(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t minCclBufferSize)
{
    uint64_t maxTileSize = std::min(MAX_TILE_SIZE, minCclBufferSize);
    maxTileSize = maxTileSize / sizeof(float) * sizeof(float);
    CHK_PRT_RET(maxTileSize == 0, HCCL_ERROR("HCCL buffer is too small"), HCCL_E_INTERNAL);

    uint64_t maxTileCount = maxTileSize / sizeof(float);
    CHK_PRT_RET(maxTileCount < 2 * MIN_SHARDED_TILE_COUNT,
        HCCL_ERROR("HCCL buffer cannot hold the minimum sharded tile"), HCCL_E_INTERNAL);
    uint64_t processedCount = 0;
    while (processedCount < param.count) {
        uint64_t remainingCount = param.count - processedCount;
        uint64_t tileCount = std::min(maxTileCount, remainingCount);
        // 避免大数据最后只剩不足8个128B分片的极小尾块。
        if (remainingCount > maxTileCount && remainingCount - maxTileCount < MIN_SHARDED_TILE_COUNT) {
            tileCount = remainingCount - MIN_SHARDED_TILE_COUNT;
        }
        uint64_t userOffset = processedCount * sizeof(float);
        uint64_t tileSize = tileCount * sizeof(float);
        std::vector<ChunkInfo> chunks;
        CHK_RET(BuildChunks(tileCount, resCtx.localRankSize, chunks));
        CHK_RET(RunScatterPhase(param, resCtx, chunks, userOffset));
        CHK_RET(RunIntraServerAllGather(param, resCtx, chunks, userOffset));

        if (param.myRank != param.root) {
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resCtx.threads[0],
                AddOffset(param.outputPtr, userOffset), resCtx.localBuffer.addr, tileSize)));
        }
        processedCount += tileCount;
    }
    return HCCL_SUCCESS;
}

} // namespace

namespace ops_hccl {
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    HCCL_INFO("Executing custom Broadcast AICPU Kernel on Ascend NPU");
    CHK_PTR_NULL(param.inputPtr);
    CHK_PTR_NULL(param.outputPtr);
    CHK_PTR_NULL(resCtx.localBuffer.addr);
    CHK_PRT_RET(param.dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("Broadcast only supports float32, dataType[%d]", param.dataType), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(param.rankSize != EXPECTED_RANK_SIZE || resCtx.localRankSize != EXPECTED_LOCAL_RANK_SIZE,
        HCCL_ERROR("Unexpected topology, rankSize[%u], localRankSize[%u]", param.rankSize, resCtx.localRankSize),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(param.root >= param.rankSize,
        HCCL_ERROR("Root[%u] is out of rankSize[%u]", param.root, param.rankSize), HCCL_E_PARA);
    CHK_PRT_RET(resCtx.localIndex >= resCtx.localRankSize,
        HCCL_ERROR("Invalid local index[%u]", resCtx.localIndex), HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.threads.size() < param.rankSize - 1 || resCtx.channels.size() != param.rankSize - 1,
        HCCL_ERROR("Invalid resource, threads[%zu], channels[%zu]", resCtx.threads.size(), resCtx.channels.size()),
        HCCL_E_INTERNAL);

    uint64_t dataSize = param.count * sizeof(float);
    if (dataSize == 0) {
        return HCCL_SUCCESS;
    }
    uint64_t minCclBufferSize = GetMinCclBufferSize(resCtx);
    CHK_PRT_RET(minCclBufferSize == 0, HCCL_ERROR("HCCL buffer size is zero"), HCCL_E_INTERNAL);

    if (dataSize <= FLAT_BROADCAST_THRESHOLD) {
        CHK_PRT_RET(dataSize > minCclBufferSize,
            HCCL_ERROR("Data size[%llu] exceeds HCCL buffer size[%llu]", static_cast<unsigned long long>(dataSize),
                static_cast<unsigned long long>(minCclBufferSize)),
            HCCL_E_INTERNAL);
        return RunFlatBroadcast(param, resCtx, dataSize);
    }
    return RunShardedBroadcast(param, resCtx, minCclBufferSize);
}
} // namespace ops_hccl
