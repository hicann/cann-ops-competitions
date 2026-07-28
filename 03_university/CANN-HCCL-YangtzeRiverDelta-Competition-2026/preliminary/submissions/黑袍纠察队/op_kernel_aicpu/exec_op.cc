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

#include <hcomm/hcomm_primitives.h>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace {
constexpr uint64_t DIRECT_PULL_THRESHOLD = 1024ULL * 1024ULL;

struct Slice {
    uint64_t offset = 0;
    uint64_t size = 0;
};

const ChannelInfo *FindChannelByRank(const AlgResourceCtx &resCtx, uint32_t remoteRank)
{
    for (const auto &channel : resCtx.channels) {
        if (channel.remoteRank == remoteRank) {
            return &channel;
        }
    }
    return nullptr;
}

HcclResult CalcSlices(uint64_t dataSize, uint32_t rankSize, uint64_t typeSize, std::vector<Slice> &slices)
{
    CHK_PRT_RET(rankSize == 0 || typeSize == 0 || dataSize % typeSize != 0,
        HCCL_ERROR("Invalid slice parameters"), HCCL_E_PARA);

    slices.assign(rankSize, Slice{});
    const uint64_t elementCount = dataSize / typeSize;
    const uint64_t elementsPerRank = elementCount / rankSize;
    const uint64_t remainder = elementCount % rankSize;
    uint64_t offset = 0;
    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        const uint64_t sliceElements = elementsPerRank + (rank < remainder ? 1 : 0);
        slices[rank] = Slice{offset, sliceElements * typeSize};
        offset += slices[rank].size;
    }
    return HCCL_SUCCESS;
}

HcclResult StartWorker(const AlgResourceCtx &resCtx, uint32_t channelIndex)
{
    if (channelIndex == 0) {
        return HCCL_SUCCESS;
    }
    return static_cast<HcclResult>(
        HcommThreadNotifyRecordOnThread(resCtx.aicpuThread, resCtx.threads[channelIndex], 0));
}

HcclResult WaitWorkerStart(const AlgResourceCtx &resCtx, uint32_t channelIndex)
{
    if (channelIndex == 0) {
        return HCCL_SUCCESS;
    }
    return static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resCtx.threads[channelIndex], 0, CUSTOM_TIMEOUT));
}

HcclResult FinishWorker(const AlgResourceCtx &resCtx, uint32_t channelIndex)
{
    if (channelIndex == 0) {
        return HCCL_SUCCESS;
    }
    return static_cast<HcclResult>(
        HcommThreadNotifyRecordOnThread(resCtx.threads[channelIndex], resCtx.aicpuThread, channelIndex - 1));
}

HcclResult WaitWorkerFinish(const AlgResourceCtx &resCtx, uint32_t channelIndex)
{
    if (channelIndex == 0) {
        return HCCL_SUCCESS;
    }
    return static_cast<HcclResult>(
        HcommThreadNotifyWaitOnThread(resCtx.aicpuThread, channelIndex - 1, CUSTOM_TIMEOUT));
}

HcclResult WriteSliceAndNotify(ThreadHandle thread, const ChannelInfo &channel, const char *localBuffer,
    char *remoteBuffer, const Slice &slice)
{
    if (slice.size != 0) {
        CHK_RET(HcommWriteOnThread(
            thread, channel.handle, remoteBuffer + slice.offset, localBuffer + slice.offset, slice.size));
        CHK_RET(HcommChannelFenceOnThread(thread, channel.handle));
    }
    return static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL));
}

HcclResult RunDirectPull(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t totalBytes, uint64_t chunkSize)
{
    ThreadHandle mainThread = resCtx.aicpuThread;
    char *localBuffer = static_cast<char *>(resCtx.localBuffer.addr);
    char *userBuffer = static_cast<char *>(param.outputPtr);

    if (param.myRank == param.root) {
        for (uint64_t offset = 0; offset < totalBytes; offset += chunkSize) {
            const uint64_t currentBytes = std::min(chunkSize, totalBytes - offset);
            CHK_RET(HcommLocalCopyOnThread(mainThread, localBuffer, userBuffer + offset, currentBytes));
            for (const auto &channel : resCtx.channels) {
                CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, channel.handle, NOTIFY_IDX_DATA_SIGNAL));
            }
            for (const auto &channel : resCtx.channels) {
                CHK_RET(HcommChannelNotifyWaitOnThread(
                    mainThread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
            }
        }
        return HCCL_SUCCESS;
    }

    const ChannelInfo *rootChannel = FindChannelByRank(resCtx, param.root);
    CHK_PRT_RET(rootChannel == nullptr || rootChannel->remoteCclMem.addr == nullptr,
        HCCL_ERROR("No valid channel to root rank[%u]", param.root), HCCL_E_INTERNAL);
    const char *remoteRootBuffer = static_cast<const char *>(rootChannel->remoteCclMem.addr);
    for (uint64_t offset = 0; offset < totalBytes; offset += chunkSize) {
        const uint64_t currentBytes = std::min(chunkSize, totalBytes - offset);
        CHK_RET(HcommChannelNotifyWaitOnThread(
            mainThread, rootChannel->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
        CHK_RET(HcommReadOnThread(mainThread, rootChannel->handle, localBuffer, remoteRootBuffer, currentBytes));
        CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, rootChannel->handle, NOTIFY_IDX_ACK));
        CHK_RET(HcommLocalCopyOnThread(mainThread, userBuffer + offset, localBuffer, currentBytes));
    }
    return HCCL_SUCCESS;
}

HcclResult RunRootPipeline(const OpParam &param, const AlgResourceCtx &resCtx, const std::vector<Slice> &slices)
{
    const char *localBuffer = static_cast<const char *>(resCtx.localBuffer.addr);
    for (uint32_t index = 1; index < resCtx.channels.size(); ++index) {
        CHK_RET(StartWorker(resCtx, index));
    }

    for (uint32_t index = 0; index < resCtx.channels.size(); ++index) {
        const ChannelInfo &channel = resCtx.channels[index];
        ThreadHandle thread = resCtx.threads[index];
        CHK_RET(WaitWorkerStart(resCtx, index));
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
        char *remoteBuffer = static_cast<char *>(channel.remoteCclMem.addr);
        CHK_RET(WriteSliceAndNotify(thread, channel, localBuffer, remoteBuffer, slices[channel.remoteRank]));
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
        CHK_RET(WriteSliceAndNotify(thread, channel, localBuffer, remoteBuffer, slices[param.root]));
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
        CHK_RET(FinishWorker(resCtx, index));
    }

    for (uint32_t index = 1; index < resCtx.channels.size(); ++index) {
        CHK_RET(WaitWorkerFinish(resCtx, index));
    }
    return HCCL_SUCCESS;
}

HcclResult RunReceiverAllGather(
    const OpParam &param, const AlgResourceCtx &resCtx, const std::vector<Slice> &slices)
{
    const Slice &localSlice = slices[param.myRank];
    char *localBuffer = static_cast<char *>(resCtx.localBuffer.addr);

    for (uint32_t index = 1; index < resCtx.channels.size(); ++index) {
        CHK_RET(StartWorker(resCtx, index));
    }

    for (uint32_t index = 0; index < resCtx.channels.size(); ++index) {
        const ChannelInfo &channel = resCtx.channels[index];
        ThreadHandle thread = resCtx.threads[index];
        CHK_RET(WaitWorkerStart(resCtx, index));
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK));
        if (channel.remoteRank == param.root) {
            CHK_RET(HcommChannelNotifyWaitOnThread(
                thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
            CHK_RET(FinishWorker(resCtx, index));
            continue;
        }

        char *remoteBuffer = static_cast<char *>(channel.remoteCclMem.addr);
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
        if (localSlice.size != 0) {
            CHK_RET(HcommWriteOnThread(thread, channel.handle, remoteBuffer + localSlice.offset,
                localBuffer + localSlice.offset, localSlice.size));
            CHK_RET(HcommChannelFenceOnThread(thread, channel.handle));
        }
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL));
        CHK_RET(HcommChannelNotifyWaitOnThread(
            thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
        CHK_RET(FinishWorker(resCtx, index));
    }

    for (uint32_t index = 1; index < resCtx.channels.size(); ++index) {
        CHK_RET(WaitWorkerFinish(resCtx, index));
    }
    return HCCL_SUCCESS;
}

HcclResult RunTwoShot(
    const OpParam &param, const AlgResourceCtx &resCtx, uint64_t totalBytes, uint64_t chunkSize, uint64_t typeSize)
{
    ThreadHandle mainThread = resCtx.aicpuThread;
    char *localBuffer = static_cast<char *>(resCtx.localBuffer.addr);
    char *userBuffer = static_cast<char *>(param.outputPtr);
    const ChannelInfo *rootChannel = param.myRank == param.root ? nullptr : FindChannelByRank(resCtx, param.root);
    CHK_PRT_RET(param.myRank != param.root && rootChannel == nullptr,
        HCCL_ERROR("No channel to root rank[%u]", param.root), HCCL_E_INTERNAL);

    for (uint64_t offset = 0; offset < totalBytes; offset += chunkSize) {
        const uint64_t currentBytes = std::min(chunkSize, totalBytes - offset);
        std::vector<Slice> slices;
        CHK_RET(CalcSlices(currentBytes, param.rankSize, typeSize, slices));

        if (param.myRank == param.root) {
            CHK_RET(HcommLocalCopyOnThread(mainThread, localBuffer, userBuffer + offset, currentBytes));
            CHK_RET(RunRootPipeline(param, resCtx, slices));
            continue;
        }

        CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, rootChannel->handle, NOTIFY_IDX_ACK));
        CHK_RET(HcommChannelNotifyWaitOnThread(
            mainThread, rootChannel->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
        CHK_RET(RunReceiverAllGather(param, resCtx, slices));
        CHK_RET(HcommLocalCopyOnThread(mainThread, userBuffer + offset, localBuffer, currentBytes));
        CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, rootChannel->handle, NOTIFY_IDX_ACK));
    }
    return HCCL_SUCCESS;
}
} // namespace

namespace ops_hccl {
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    CHK_PRT_RET(param.myRank == INVALID_VALUE_RANKID || param.myRank >= param.rankSize ||
            param.root >= param.rankSize,
        HCCL_ERROR("Invalid rank info, myRank[%u], root[%u], rankSize[%u]", param.myRank, param.root, param.rankSize),
        HCCL_E_PARA);
    CHK_PTR_NULL(param.inputPtr);
    CHK_PTR_NULL(param.outputPtr);

    const auto sizeIt = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(sizeIt == SIZE_TABLE.end(), HCCL_ERROR("Unsupported data type[%d]", param.dataType), HCCL_E_PARA);
    const uint64_t typeSize = sizeIt->second;
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / typeSize,
        HCCL_ERROR("Broadcast data size overflow"), HCCL_E_PARA);
    const uint64_t totalBytes = param.count * typeSize;
    if (totalBytes == 0 || param.rankSize <= 1) {
        return HCCL_SUCCESS;
    }

    CHK_PTR_NULL(resCtx.localBuffer.addr);
    CHK_PRT_RET(resCtx.channels.size() != param.rankSize - 1 || resCtx.threads.size() != param.rankSize - 1,
        HCCL_ERROR("Incomplete broadcast resources"), HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.maxSliceSize < typeSize || resCtx.maxSliceSize > resCtx.localBuffer.size,
        HCCL_ERROR("Invalid HCCL buffer size"), HCCL_E_INTERNAL);
    for (const auto &channel : resCtx.channels) {
        CHK_PRT_RET(channel.remoteRank >= param.rankSize || channel.remoteCclMem.addr == nullptr ||
                channel.remoteCclMem.size < resCtx.maxSliceSize,
            HCCL_ERROR("Invalid channel resource"), HCCL_E_INTERNAL);
    }

    const uint64_t chunkSize = resCtx.maxSliceSize - resCtx.maxSliceSize % typeSize;
    CHK_PRT_RET(chunkSize == 0, HCCL_ERROR("HCCL buffer cannot hold one element"), HCCL_E_INTERNAL);
    if (totalBytes <= DIRECT_PULL_THRESHOLD) {
        return RunDirectPull(param, resCtx, totalBytes, chunkSize);
    }
    return RunTwoShot(param, resCtx, totalBytes, chunkSize, typeSize);
}
} // namespace ops_hccl
