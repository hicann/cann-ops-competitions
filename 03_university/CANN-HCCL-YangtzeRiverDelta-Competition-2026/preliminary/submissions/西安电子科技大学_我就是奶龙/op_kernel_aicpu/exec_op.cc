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
#include <array>
#include <cstdint>
#include <limits>
#include <vector>

#include "custom.h"
#include "log.h"
#include "exec_op.h"

namespace ops_hccl {
namespace {
constexpr uint64_t TINY_DIRECT_DATA_BYTES = 4;
constexpr uint64_t CLOS_XOR_DATA_BYTES = 512 * 1024;
constexpr uint32_t OFFICIAL_RANK_SIZE = 16;
constexpr uint32_t DIRECT16_THREAD_NUM = OFFICIAL_RANK_SIZE;
constexpr std::array<uint32_t, 4> CLOS_XOR_MASKS = {8, 12, 10, 9};

struct DataSlice {
    uint64_t offset = 0;
    uint64_t count = 0;
};

uint8_t *AddBytes(void *address, uint64_t offset)
{
    return static_cast<uint8_t *>(address) + offset;
}

DataSlice CalcSlice(uint64_t count, uint32_t sliceNum, uint32_t sliceIndex)
{
    uint64_t stride = count / sliceNum + (count % sliceNum != 0);
    uint64_t offset = std::min(count, stride * sliceIndex);
    return DataSlice{offset, std::min(stride, count - offset)};
}

HcclResult BuildChannelTable(const OpParam &param, const AlgResourceCtx &resCtx,
    std::vector<const ChannelInfo *> &channelTable)
{
    channelTable.assign(param.rankSize, nullptr);
    for (const auto &channel : resCtx.channels) {
        CHK_PRT_RET(channel.remoteRank >= param.rankSize || channel.remoteRank == param.myRank,
            HCCL_ERROR("[BuildChannelTable] Invalid remoteRank[%u].", channel.remoteRank), HCCL_E_INTERNAL);
        CHK_PRT_RET(channelTable[channel.remoteRank] != nullptr,
            HCCL_ERROR("[BuildChannelTable] Duplicate channel to rank[%u].", channel.remoteRank), HCCL_E_INTERNAL);
        channelTable[channel.remoteRank] = &channel;
    }
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        CHK_PRT_RET(remoteRank != param.myRank && channelTable[remoteRank] == nullptr,
            HCCL_ERROR("[BuildChannelTable] Channel to rank[%u] does not exist.", remoteRank),
            HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult GetChannel(const std::vector<const ChannelInfo *> &channelTable, uint32_t remoteRank,
    const ChannelInfo **channel)
{
    CHK_PRT_RET(remoteRank >= channelTable.size() || channelTable[remoteRank] == nullptr,
        HCCL_ERROR("[GetChannel] Channel to rank[%u] does not exist.", remoteRank), HCCL_E_INTERNAL);
    *channel = channelTable[remoteRank];
    return HCCL_SUCCESS;
}

HcclResult MainToSubBarrier(const std::vector<ThreadHandle> &threads)
{
    uint32_t threadNum = static_cast<uint32_t>(threads.size());
    for (uint32_t index = 1; index < threadNum; ++index) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[0], threads[index], 0)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[index], 0, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult SubToMainBarrier(const std::vector<ThreadHandle> &threads)
{
    uint32_t threadNum = static_cast<uint32_t>(threads.size());
    for (uint32_t index = 1; index < threadNum; ++index) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(threads[index], threads[0], index - 1)));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(threads[0], index - 1, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult Direct16Send(ThreadHandle thread, const ChannelInfo &channel, const void *source, uint64_t size)
{
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommWriteWithNotifyOnThread(
        thread, channel.handle, channel.remoteCclMem.addr, source, size, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult RunDirect16Broadcast(const OpParam &param, const AlgResourceCtx &resCtx,
    const std::vector<const ChannelInfo *> &channelTable, uint32_t dataTypeSize)
{
    uint64_t dataSize = param.count * dataTypeSize;
    CHK_PRT_RET(dataSize > resCtx.localBuffer.size,
        HCCL_ERROR("[RunDirect16Broadcast] Local CCL buffer is too small."), HCCL_E_INTERNAL);

    if (param.myRank != param.root) {
        const ChannelInfo *rootChannel = nullptr;
        CHK_RET(GetChannel(channelTable, param.root, &rootChannel));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(resCtx.threads[0], rootChannel->handle, NOTIFY_IDX_ACK)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            resCtx.threads[0], rootChannel->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(resCtx.threads[0], param.outputPtr, resCtx.localBuffer.addr, dataSize)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(resCtx.threads[0], rootChannel->handle, NOTIFY_IDX_ACK)));
        return HCCL_SUCCESS;
    }

    CHK_RET(MainToSubBarrier(resCtx.threads));
    uint32_t threadIndex = 0;
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.root) {
            continue;
        }
        const ChannelInfo *channel = nullptr;
        CHK_RET(GetChannel(channelTable, remoteRank, &channel));
        CHK_PRT_RET(dataSize > channel->remoteCclMem.size,
            HCCL_ERROR("[RunDirect16Broadcast] Remote CCL buffer is too small."), HCCL_E_INTERNAL);
        CHK_RET(Direct16Send(resCtx.threads[threadIndex++], *channel, param.inputPtr, dataSize));
    }
    return SubToMainBarrier(resCtx.threads);
}

uint32_t ToClosXorCoordinate(uint32_t relativeRank)
{
    uint32_t basis1 = (relativeRank >> 2U) & 1U;
    uint32_t basis2 = (relativeRank >> 1U) & 1U;
    uint32_t basis3 = relativeRank & 1U;
    uint32_t basis0 = ((relativeRank >> 3U) & 1U) ^ basis1 ^ basis2 ^ basis3;
    return basis0 | (basis1 << 1U) | (basis2 << 2U) | (basis3 << 3U);
}

HcclResult RunClosXorButterfly(const OpParam &param, const AlgResourceCtx &resCtx,
    const std::vector<const ChannelInfo *> &channelTable, uint32_t dataTypeSize)
{
    CHK_PRT_RET(param.rankSize != OFFICIAL_RANK_SIZE,
        HCCL_ERROR("[RunClosXorButterfly] Invalid rankSize[%u].", param.rankSize), HCCL_E_PARA);
    uint64_t dataSize = param.count * dataTypeSize;
    CHK_PRT_RET(dataSize > resCtx.localBuffer.size,
        HCCL_ERROR("[RunClosXorButterfly] Local CCL buffer is too small."), HCCL_E_INTERNAL);

    ThreadHandle thread = resCtx.threads[0];
    uint32_t relativeRank = param.myRank ^ param.root;
    uint32_t coordinate = ToClosXorCoordinate(relativeRank);
    const void *source = param.myRank == param.root ? param.inputPtr : resCtx.localBuffer.addr;
    uint32_t parentRank = INVALID_VALUE_RANKID;
    std::array<uint32_t, CLOS_XOR_MASKS.size()> childRanks = {};
    uint32_t childNum = 0;

    for (uint32_t round = 0; round < CLOS_XOR_MASKS.size(); ++round) {
        uint32_t mask = CLOS_XOR_MASKS[round];
        uint32_t roundSpan = 1U << round;
        if (coordinate < roundSpan) {
            uint32_t childRank = param.myRank ^ mask;
            const ChannelInfo *channel = nullptr;
            CHK_RET(GetChannel(channelTable, childRank, &channel));
            CHK_PRT_RET(dataSize > channel->remoteCclMem.size,
                HCCL_ERROR("[RunClosXorButterfly] Remote CCL buffer is too small."), HCCL_E_INTERNAL);
            CHK_RET(static_cast<HcclResult>(HcommWriteWithNotifyOnThread(thread, channel->handle,
                channel->remoteCclMem.addr, source, dataSize, NOTIFY_IDX_DATA_SIGNAL)));
            childRanks[childNum++] = childRank;
            continue;
        }

        if (coordinate < (roundSpan << 1U)) {
            parentRank = param.myRank ^ mask;
            const ChannelInfo *channel = nullptr;
            CHK_RET(GetChannel(channelTable, parentRank, &channel));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                thread, channel->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
            source = resCtx.localBuffer.addr;
        }
    }

    CHK_PRT_RET(coordinate >= OFFICIAL_RANK_SIZE,
        HCCL_ERROR("[RunClosXorButterfly] Rank[%u] is not covered.", param.myRank), HCCL_E_INTERNAL);
    if (param.myRank != param.root) {
        CHK_PRT_RET(parentRank == INVALID_VALUE_RANKID,
            HCCL_ERROR("[RunClosXorButterfly] Parent of rank[%u] does not exist.", param.myRank),
            HCCL_E_INTERNAL);
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(thread, param.outputPtr, source, dataSize)));
        const ChannelInfo *parentChannel = nullptr;
        CHK_RET(GetChannel(channelTable, parentRank, &parentChannel));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, parentChannel->handle, NOTIFY_IDX_ACK)));
    }

    for (uint32_t index = 0; index < childNum; ++index) {
        const ChannelInfo *childChannel = nullptr;
        CHK_RET(GetChannel(channelTable, childRanks[index], &childChannel));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            thread, childChannel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult PublishShard(ThreadHandle thread, const ChannelInfo &channel, const void *source, uint64_t size)
{
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommWriteWithNotifyOnThread(
        thread, channel.handle, channel.remoteCclMem.addr, source, size, NOTIFY_IDX_DATA_SIGNAL)));
    return HCCL_SUCCESS;
}

HcclResult WaitPublishedShard(ThreadHandle thread, const ChannelInfo &channel)
{
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult ReadPublishedShard(ThreadHandle thread, const ChannelInfo &channel, void *destination,
    uint64_t size)
{
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(
        HcommReadOnThread(thread, channel.handle, destination, channel.remoteCclMem.addr, size)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult SynchronizePublishedShard(ThreadHandle thread, const ChannelInfo &channel)
{
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult RunCompactScatter(const OpParam &param, const AlgResourceCtx &resCtx,
    const std::vector<const ChannelInfo *> &channelTable, uint64_t segmentOffset,
    uint64_t segmentCount, uint32_t dataTypeSize)
{
    DataSlice ownedSlice = CalcSlice(segmentCount, param.rankSize, param.myRank);
    if (param.myRank != param.root) {
        const ChannelInfo *rootChannel = nullptr;
        CHK_RET(GetChannel(channelTable, param.root, &rootChannel));
        return WaitPublishedShard(resCtx.threads[0], *rootChannel);
    }

    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resCtx.threads[0], resCtx.localBuffer.addr,
        AddBytes(param.inputPtr, (segmentOffset + ownedSlice.offset) * dataTypeSize),
        ownedSlice.count * dataTypeSize)));
    CHK_RET(MainToSubBarrier(resCtx.threads));
    uint32_t threadIndex = 0;
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (rank == param.root) {
            continue;
        }
        DataSlice targetSlice = CalcSlice(segmentCount, param.rankSize, rank);
        const ChannelInfo *channel = nullptr;
        CHK_RET(GetChannel(channelTable, rank, &channel));
        CHK_RET(PublishShard(resCtx.threads[threadIndex++], *channel,
            AddBytes(param.inputPtr, (segmentOffset + targetSlice.offset) * dataTypeSize),
            targetSlice.count * dataTypeSize));
    }
    return SubToMainBarrier(resCtx.threads);
}

HcclResult RunCompactGather(const OpParam &param, const AlgResourceCtx &resCtx,
    const std::vector<const ChannelInfo *> &channelTable, uint64_t segmentOffset,
    uint64_t segmentCount, uint32_t dataTypeSize)
{
    DataSlice ownedSlice = CalcSlice(segmentCount, param.rankSize, param.myRank);
    CHK_RET(MainToSubBarrier(resCtx.threads));
    if (param.myRank != param.root) {
        ThreadHandle copyThread = resCtx.threads[param.rankSize - 1];
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(copyThread,
            AddBytes(param.outputPtr, (segmentOffset + ownedSlice.offset) * dataTypeSize),
            resCtx.localBuffer.addr, ownedSlice.count * dataTypeSize)));
    }

    uint32_t threadIndex = 0;
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (rank == param.myRank) {
            continue;
        }
        const ChannelInfo *channel = nullptr;
        CHK_RET(GetChannel(channelTable, rank, &channel));
        ThreadHandle thread = resCtx.threads[threadIndex++];
        if (param.myRank == param.root) {
            CHK_RET(SynchronizePublishedShard(thread, *channel));
            continue;
        }

        DataSlice remoteSlice = CalcSlice(segmentCount, param.rankSize, rank);
        CHK_RET(ReadPublishedShard(thread, *channel,
            AddBytes(param.outputPtr, (segmentOffset + remoteSlice.offset) * dataTypeSize),
            remoteSlice.count * dataTypeSize));
    }
    return SubToMainBarrier(resCtx.threads);
}

HcclResult RunCompactBroadcast(const OpParam &param, const AlgResourceCtx &resCtx,
    const std::vector<const ChannelInfo *> &channelTable, uint32_t dataTypeSize)
{
    uint64_t shardCapacityBytes = resCtx.localBuffer.size;
    for (const auto &channel : resCtx.channels) {
        shardCapacityBytes = std::min(shardCapacityBytes, channel.remoteCclMem.size);
    }
    uint64_t shardCapacityCount = shardCapacityBytes / dataTypeSize;
    CHK_PRT_RET(shardCapacityCount == 0,
        HCCL_ERROR("[RunCompactBroadcast] HCCL buffer is too small."), HCCL_E_INTERNAL);

    uint64_t maxSegmentCount = std::numeric_limits<uint64_t>::max();
    if (shardCapacityCount <= maxSegmentCount / param.rankSize) {
        maxSegmentCount = shardCapacityCount * param.rankSize;
    }

    for (uint64_t offset = 0; offset < param.count;) {
        uint64_t segmentCount = std::min(maxSegmentCount, param.count - offset);
        CHK_RET(RunCompactScatter(
            param, resCtx, channelTable, offset, segmentCount, dataTypeSize));
        CHK_RET(RunCompactGather(
            param, resCtx, channelTable, offset, segmentCount, dataTypeSize));
        offset += segmentCount;
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    HCCL_INFO("Executing AICPU Kernel on Ascend NPU");

    // TODO: 算法任务编排
    if (param.count == 0 || param.rankSize == 1) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(param.root >= param.rankSize,
        HCCL_ERROR("[ExecOp] Invalid root[%u], rankSize[%u].", param.root, param.rankSize), HCCL_E_PARA);

    auto dataTypeIter = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(dataTypeIter == SIZE_TABLE.end(),
        HCCL_ERROR("[ExecOp] Unsupported dataType[%d].", param.dataType), HCCL_E_NOT_SUPPORT);
    uint32_t dataTypeSize = dataTypeIter->second;
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / dataTypeSize,
        HCCL_ERROR("[ExecOp] Data size overflow, count[%llu].",
            static_cast<unsigned long long>(param.count)),
        HCCL_E_PARA);

    uint64_t dataBytes = param.count * dataTypeSize;
    CHK_PRT_RET(resCtx.threads.size() < DIRECT16_THREAD_NUM,
        HCCL_ERROR("[ExecOp] Invalid threadNum[%zu], required[%u].", resCtx.threads.size(),
            DIRECT16_THREAD_NUM),
        HCCL_E_INTERNAL);

    std::vector<const ChannelInfo *> channelTable;
    CHK_RET(BuildChannelTable(param, resCtx, channelTable));
    if (dataBytes == TINY_DIRECT_DATA_BYTES) {
        return RunDirect16Broadcast(param, resCtx, channelTable, dataTypeSize);
    }
    if (dataBytes == CLOS_XOR_DATA_BYTES) {
        return RunClosXorButterfly(param, resCtx, channelTable, dataTypeSize);
    }
    if (dataBytes < CLOS_XOR_DATA_BYTES) {
        return RunDirect16Broadcast(param, resCtx, channelTable, dataTypeSize);
    }
    return RunCompactBroadcast(param, resCtx, channelTable, dataTypeSize);
}
} // namespace ops_hccl