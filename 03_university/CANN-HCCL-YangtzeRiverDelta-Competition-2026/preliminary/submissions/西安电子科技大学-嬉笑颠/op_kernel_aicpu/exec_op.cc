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
#include <cstdint>
#include <limits>
#include <vector>

#include "custom.h"
#include "log.h"
#include "exec_op.h"

namespace ops_hccl {
namespace {
constexpr uint64_t DIRECT_DATA_BYTES = 512 * 1024;

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

    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (rank == param.myRank) {
            continue;
        }
        CHK_PRT_RET(channelTable[rank] == nullptr,
            HCCL_ERROR("[BuildChannelTable] Channel to rank[%u] does not exist.", rank), HCCL_E_INTERNAL);
    }
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
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[0], index - 1, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult SendSlice(ThreadHandle thread, const ChannelInfo &channel, void *remoteDestination,
    const void *localSource, uint64_t size)
{
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    if (size != 0) {
        CHK_RET(static_cast<HcclResult>(
            HcommWriteOnThread(thread, channel.handle, remoteDestination, localSource, size)));
    }
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    return HCCL_SUCCESS;
}

HcclResult ReceiveSlice(ThreadHandle thread, const ChannelInfo &channel)
{
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult ReadSlice(ThreadHandle thread, const ChannelInfo &channel, void *localDestination,
    const void *remoteSource, uint64_t size)
{
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    if (size != 0) {
        CHK_RET(static_cast<HcclResult>(
            HcommReadOnThread(thread, channel.handle, localDestination, remoteSource, size)));
    }
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult SynchronizeSlice(ThreadHandle thread, const ChannelInfo &channel)
{
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult RunDirectBroadcast(const OpParam &param, const AlgResourceCtx &resCtx,
    const std::vector<const ChannelInfo *> &channelTable, uint64_t segmentOffset, uint64_t segmentCount,
    uint32_t dataTypeSize)
{
    uint64_t segmentSize = segmentCount * dataTypeSize;
    void *segmentBuffer = AddBytes(param.outputPtr, segmentOffset * dataTypeSize);
    if (param.myRank == param.root) {
        CHK_RET(MainToSubBarrier(resCtx.threads));
        uint32_t threadIndex = 0;
        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            if (rank == param.myRank) {
                continue;
            }
            const ChannelInfo &channel = *channelTable[rank];
            CHK_RET(SendSlice(resCtx.threads[threadIndex++], channel, channel.remoteCclMem.addr, segmentBuffer,
                segmentSize));
        }
        CHK_RET(SubToMainBarrier(resCtx.threads));
        return HCCL_SUCCESS;
    }

    const ChannelInfo &channel = *channelTable[param.root];
    CHK_RET(ReceiveSlice(resCtx.threads[0], channel));
    CHK_RET(static_cast<HcclResult>(
        HcommLocalCopyOnThread(resCtx.threads[0], segmentBuffer, resCtx.localBuffer.addr, segmentSize)));
    return HCCL_SUCCESS;
}

HcclResult RunScatter(const OpParam &param, const AlgResourceCtx &resCtx,
    const std::vector<const ChannelInfo *> &channelTable, uint64_t segmentOffset, uint64_t segmentCount,
    uint32_t dataTypeSize)
{
    DataSlice ownedSlice = CalcSlice(segmentCount, param.rankSize, param.myRank);
    if (param.myRank != param.root) {
        const ChannelInfo &channel = *channelTable[param.root];
        CHK_RET(ReceiveSlice(resCtx.threads[0], channel));
        return HCCL_SUCCESS;
    }

    if (ownedSlice.count != 0) {
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resCtx.threads[0],
            AddBytes(resCtx.localBuffer.addr, ownedSlice.offset * dataTypeSize),
            AddBytes(param.inputPtr, (segmentOffset + ownedSlice.offset) * dataTypeSize),
            ownedSlice.count * dataTypeSize)));
    }
    CHK_RET(MainToSubBarrier(resCtx.threads));
    uint32_t threadIndex = 0;
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (rank == param.myRank) {
            continue;
        }
        DataSlice targetSlice = CalcSlice(segmentCount, param.rankSize, rank);
        const ChannelInfo &channel = *channelTable[rank];
        CHK_RET(SendSlice(resCtx.threads[threadIndex++], channel,
            AddBytes(channel.remoteCclMem.addr, targetSlice.offset * dataTypeSize),
            AddBytes(param.inputPtr, (segmentOffset + targetSlice.offset) * dataTypeSize),
            targetSlice.count * dataTypeSize));
    }
    CHK_RET(SubToMainBarrier(resCtx.threads));
    return HCCL_SUCCESS;
}

HcclResult RunAllGather(const OpParam &param, const AlgResourceCtx &resCtx,
    const std::vector<const ChannelInfo *> &channelTable, uint64_t segmentOffset, uint64_t segmentCount,
    uint32_t dataTypeSize)
{
    DataSlice ownedSlice = CalcSlice(segmentCount, param.rankSize, param.myRank);
    void *ownedData = AddBytes(resCtx.localBuffer.addr, ownedSlice.offset * dataTypeSize);
    CHK_RET(MainToSubBarrier(resCtx.threads));

    if (param.myRank != param.root && ownedSlice.count != 0) {
        ThreadHandle copyThread = resCtx.threads[param.rankSize - 1];
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(copyThread,
            AddBytes(param.outputPtr, (segmentOffset + ownedSlice.offset) * dataTypeSize), ownedData,
            ownedSlice.count * dataTypeSize)));
    }

    uint32_t threadIndex = 0;
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (rank == param.myRank) {
            continue;
        }
        const ChannelInfo &channel = *channelTable[rank];
        ThreadHandle thread = resCtx.threads[threadIndex++];
        if (param.myRank == param.root) {
            CHK_RET(SynchronizeSlice(thread, channel));
            continue;
        }

        DataSlice remoteSlice = CalcSlice(segmentCount, param.rankSize, rank);
        CHK_RET(ReadSlice(thread, channel,
            AddBytes(param.outputPtr, (segmentOffset + remoteSlice.offset) * dataTypeSize),
            AddBytes(channel.remoteCclMem.addr, remoteSlice.offset * dataTypeSize),
            remoteSlice.count * dataTypeSize));
    }
    CHK_RET(SubToMainBarrier(resCtx.threads));
    return HCCL_SUCCESS;
}

HcclResult RunSegmentedBroadcast(const OpParam &param, const AlgResourceCtx &resCtx,
    const std::vector<const ChannelInfo *> &channelTable, uint32_t dataTypeSize)
{
    uint64_t maxSegmentSize = resCtx.localBuffer.size;
    for (const auto &channel : resCtx.channels) {
        maxSegmentSize = std::min(maxSegmentSize, channel.remoteCclMem.size);
    }
    uint64_t maxSegmentCount = maxSegmentSize / dataTypeSize;
    CHK_PRT_RET(maxSegmentCount == 0,
        HCCL_ERROR("[RunSegmentedBroadcast] HCCL buffer is too small."), HCCL_E_INTERNAL);

    for (uint64_t offset = 0; offset < param.count;) {
        uint64_t remainingCount = param.count - offset;
        uint64_t remainingSize = remainingCount * dataTypeSize;
        if (remainingSize <= DIRECT_DATA_BYTES && remainingCount <= maxSegmentCount) {
            CHK_RET(RunDirectBroadcast(
                param, resCtx, channelTable, offset, remainingCount, dataTypeSize));
            break;
        }

        uint64_t segmentCount = std::min(maxSegmentCount, remainingCount);
        CHK_RET(RunScatter(param, resCtx, channelTable, offset, segmentCount, dataTypeSize));
        CHK_RET(RunAllGather(param, resCtx, channelTable, offset, segmentCount, dataTypeSize));
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

    uint32_t threadNum = std::max(1U, param.rankSize);
    CHK_PRT_RET(resCtx.threads.size() < threadNum,
        HCCL_ERROR("[ExecOp] Invalid threadNum[%zu], required[%u].", resCtx.threads.size(), threadNum),
        HCCL_E_INTERNAL);

    std::vector<const ChannelInfo *> channelTable;
    CHK_RET(BuildChannelTable(param, resCtx, channelTable));
    return RunSegmentedBroadcast(param, resCtx, channelTable, dataTypeSize);
}
} // namespace ops_hccl
