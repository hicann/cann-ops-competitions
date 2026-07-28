/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

// HCCL 西部赛区初赛 — hccl_allreduce
// 接手：瞿鑫鹏 (24069100103)  西电 CS  2026-07-16
// 基线：submission-92058，两级 AllReduce，功能用例全部通过

#include "custom.h"
#include "log.h"
#include "exec_op.h"

namespace ops_hccl {
namespace {
constexpr uint32_t ROOT_RANK = 0;
constexpr uint32_t RANKS_PER_SERVER = 8;
constexpr uint64_t MAX_CHUNK_SIZE = 64ULL * 1024 * 1024;
constexpr uint64_t ONE_SHOT_LIMIT = 1ULL * 1024 * 1024;

struct SliceInfo {
    uint64_t offset;
    uint64_t count;
};

const ChannelInfo *FindChannel(const AlgResourceCtx &resource, uint32_t remoteRank)
{
    for (const ChannelInfo &channel : resource.channels) {
        if (channel.remoteRank == remoteRank) {
            return &channel;
        }
    }
    return nullptr;
}

HcclResult ReducePeer(const OpParam &param, const AlgResourceCtx &resource, const ChannelInfo &channel,
    uint64_t chunkCount, uint32_t dataTypeSize)
{
    ThreadHandle thread = resource.aicpuThread;
    for (uint64_t offset = 0; offset < param.count; offset += chunkCount) {
        const uint64_t currentCount = std::min(chunkCount, param.count - offset);
        void *output = static_cast<void *>(static_cast<uint8_t *>(param.outputPtr) + offset * dataTypeSize);

        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(thread, output, resource.localBuffer.addr,
            currentCount, static_cast<HcommDataType>(param.dataType), static_cast<HcommReduceOp>(param.reduceType))));
    }
    return HCCL_SUCCESS;
}

HcclResult ReducePeersChunkMajor(const OpParam &param, const AlgResourceCtx &resource,
    const std::vector<const ChannelInfo *> &channels, uint64_t chunkCount, uint32_t dataTypeSize)
{
    ThreadHandle thread = resource.aicpuThread;
    for (uint64_t offset = 0; offset < param.count; offset += chunkCount) {
        const uint64_t currentCount = std::min(chunkCount, param.count - offset);
        void *output = static_cast<uint8_t *>(param.outputPtr) + offset * dataTypeSize;

        for (const ChannelInfo *channel : channels) {
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(thread, channel->handle, NOTIFY_IDX_ACK)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                thread, channel->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(thread, output, resource.localBuffer.addr,
                currentCount, static_cast<HcommDataType>(param.dataType),
                static_cast<HcommReduceOp>(param.reduceType))));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CopyInputToOutput(const OpParam &param, ThreadHandle thread, uint64_t chunkCount,
    uint32_t dataTypeSize)
{
    if (param.inputPtr == param.outputPtr) {
        return HCCL_SUCCESS;
    }

    for (uint64_t offset = 0; offset < param.count; offset += chunkCount) {
        const uint64_t currentCount = std::min(chunkCount, param.count - offset);
        const uint64_t currentBytes = currentCount * dataTypeSize;
        const void *input = static_cast<const void *>(
            static_cast<const uint8_t *>(param.inputPtr) + offset * dataTypeSize);
        void *output = static_cast<void *>(
            static_cast<uint8_t *>(param.outputPtr) + offset * dataTypeSize);
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, output, input, currentBytes)));
    }
    return HCCL_SUCCESS;
}

HcclResult SendToRoot(const OpParam &param, const AlgResourceCtx &resource, const ChannelInfo &channel,
    uint64_t chunkCount, uint32_t dataTypeSize)
{
    ThreadHandle thread = resource.aicpuThread;
    for (uint64_t offset = 0; offset < param.count; offset += chunkCount) {
        const uint64_t currentCount = std::min(chunkCount, param.count - offset);
        const uint64_t currentBytes = currentCount * dataTypeSize;
        const void *input = static_cast<const void *>(
            static_cast<const uint8_t *>(param.inputPtr) + offset * dataTypeSize);

        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(
            HcommWriteOnThread(thread, channel.handle, channel.remoteCclMem.addr, input, currentBytes)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    }
    return HCCL_SUCCESS;
}

HcclResult BroadcastToPeer(const OpParam &param, const AlgResourceCtx &resource, const ChannelInfo &channel,
    uint64_t chunkCount, uint32_t dataTypeSize)
{
    ThreadHandle thread = resource.aicpuThread;
    for (uint64_t offset = 0; offset < param.count; offset += chunkCount) {
        const uint64_t currentCount = std::min(chunkCount, param.count - offset);
        const uint64_t currentBytes = currentCount * dataTypeSize;
        const void *output = static_cast<const void *>(
            static_cast<const uint8_t *>(param.outputPtr) + offset * dataTypeSize);

        CHK_RET(static_cast<HcclResult>(
            HcommWriteOnThread(thread, channel.handle, channel.remoteCclMem.addr, output, currentBytes)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult BroadcastPeersChunkMajor(const OpParam &param, const AlgResourceCtx &resource,
    const std::vector<const ChannelInfo *> &channels, uint64_t chunkCount, uint32_t dataTypeSize)
{
    ThreadHandle thread = resource.aicpuThread;
    for (uint64_t offset = 0; offset < param.count; offset += chunkCount) {
        const uint64_t currentCount = std::min(chunkCount, param.count - offset);
        const uint64_t currentBytes = currentCount * dataTypeSize;
        const void *output = static_cast<const uint8_t *>(param.outputPtr) + offset * dataTypeSize;

        for (const ChannelInfo *channel : channels) {
            CHK_RET(static_cast<HcclResult>(
                HcommWriteOnThread(thread, channel->handle, channel->remoteCclMem.addr, output, currentBytes)));
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(thread, channel->handle, NOTIFY_IDX_DATA_SIGNAL)));
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyWaitOnThread(thread, channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ReceiveFromRoot(const OpParam &param, const AlgResourceCtx &resource, const ChannelInfo &channel,
    uint64_t chunkCount, uint32_t dataTypeSize)
{
    ThreadHandle thread = resource.aicpuThread;
    for (uint64_t offset = 0; offset < param.count; offset += chunkCount) {
        const uint64_t currentCount = std::min(chunkCount, param.count - offset);
        const uint64_t currentBytes = currentCount * dataTypeSize;
        void *output = static_cast<void *>(static_cast<uint8_t *>(param.outputPtr) + offset * dataTypeSize);

        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(thread, output, resource.localBuffer.addr, currentBytes)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK)));
    }
    return HCCL_SUCCESS;
}

SliceInfo GetSlice(uint64_t count, uint32_t sliceNum, uint32_t sliceIdx)
{
    const uint64_t baseCount = count / sliceNum;
    const uint64_t remainder = count % sliceNum;
    return SliceInfo{sliceIdx * baseCount + std::min<uint64_t>(sliceIdx, remainder),
        baseCount + (sliceIdx < remainder ? 1 : 0)};
}

void *AddBytes(void *ptr, uint64_t offset)
{
    return static_cast<uint8_t *>(ptr) + offset;
}

HcclResult GetOptimizedChannels(const OpParam &param, const AlgResourceCtx &resource,
    std::vector<const ChannelInfo *> &localChannels, const ChannelInfo *&pairedChannel)
{
    const uint32_t localStart = param.myRank / RANKS_PER_SERVER * RANKS_PER_SERVER;
    const uint32_t localEnd = std::min(localStart + RANKS_PER_SERVER, param.rankSize);
    for (uint32_t rank = localStart; rank < localEnd; ++rank) {
        if (rank == param.myRank) {
            continue;
        }
        const ChannelInfo *channel = FindChannel(resource, rank);
        CHK_PRT_RET(channel == nullptr, HCCL_ERROR("Channel to local rank[%u] not found", rank), HCCL_E_INTERNAL);
        localChannels.push_back(channel);
    }

    pairedChannel = nullptr;
    if (param.rankSize == 2 * RANKS_PER_SERVER) {
        const uint32_t pairedRank = param.myRank < RANKS_PER_SERVER
            ? param.myRank + RANKS_PER_SERVER
            : param.myRank - RANKS_PER_SERVER;
        pairedChannel = FindChannel(resource, pairedRank);
        CHK_PRT_RET(pairedChannel == nullptr,
            HCCL_ERROR("Channel to paired rank[%u] not found", pairedRank), HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult StartLocalThreads(const AlgResourceCtx &resource, uint32_t channelNum)
{
    for (uint32_t idx = 0; idx < channelNum; ++idx) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resource.aicpuThread, resource.threads[idx + 1], 0)));
    }
    return HCCL_SUCCESS;
}

HcclResult WaitLocalThreads(const AlgResourceCtx &resource, uint32_t channelNum)
{
    for (uint32_t idx = 0; idx < channelNum; ++idx) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            resource.aicpuThread, idx + 1, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult FinishLocalTransfer(ThreadHandle thread, ThreadHandle mainThread, uint32_t notifyIdx)
{
    return static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(thread, mainThread, notifyIdx));
}

HcclResult CrossServerReduce(const OpParam &param, const AlgResourceCtx &resource,
    const ChannelInfo *pairedChannel, void *output, uint64_t count, uint32_t dataTypeSize)
{
    if (pairedChannel == nullptr) {
        return HCCL_SUCCESS;
    }

    ThreadHandle thread = resource.aicpuThread;
    const uint64_t bytes = count * dataTypeSize;
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, pairedChannel->handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        thread, pairedChannel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    if (bytes != 0) {
        CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(
            thread, pairedChannel->handle, pairedChannel->remoteCclMem.addr, output, bytes)));
    }
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, pairedChannel->handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        thread, pairedChannel->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    if (count != 0) {
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(thread, output, resource.localBuffer.addr, count,
            static_cast<HcommDataType>(param.dataType), static_cast<HcommReduceOp>(param.reduceType))));
    }
    return HCCL_SUCCESS;
}

HcclResult RunOneShot(const OpParam &param, const AlgResourceCtx &resource,
    const std::vector<const ChannelInfo *> &localChannels, const ChannelInfo *pairedChannel,
    uint64_t slotStride, uint32_t dataTypeSize)
{
    ThreadHandle mainThread = resource.aicpuThread;
    const uint32_t localStart = param.myRank / RANKS_PER_SERVER * RANKS_PER_SERVER;
    const uint32_t myLocalRank = param.myRank - localStart;
    const uint64_t bytes = param.count * dataTypeSize;

    CHK_RET(StartLocalThreads(resource, localChannels.size()));
    for (uint32_t idx = 0; idx < localChannels.size(); ++idx) {
        const ChannelInfo &channel = *localChannels[idx];
        ThreadHandle thread = resource.threads[idx + 1];
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(thread, 0, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(thread, channel.handle,
            AddBytes(channel.remoteCclMem.addr, myLocalRank * slotStride), param.inputPtr, bytes)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
        CHK_RET(FinishLocalTransfer(thread, mainThread, idx + 1));
    }
    CHK_RET(WaitLocalThreads(resource, localChannels.size()));

    if (param.inputPtr != param.outputPtr) {
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(mainThread, param.outputPtr, param.inputPtr, bytes)));
    }
    for (const ChannelInfo *channel : localChannels) {
        const uint32_t peerLocalRank = channel->remoteRank - localStart;
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(mainThread, param.outputPtr,
            AddBytes(resource.localBuffer.addr, peerLocalRank * slotStride), param.count,
            static_cast<HcommDataType>(param.dataType), static_cast<HcommReduceOp>(param.reduceType))));
    }

    CHK_RET(CrossServerReduce(param, resource, pairedChannel, param.outputPtr, param.count, dataTypeSize));
    return HCCL_SUCCESS;
}

HcclResult RunReduceScatterChunk(const OpParam &param, const AlgResourceCtx &resource,
    const std::vector<const ChannelInfo *> &localChannels, uint32_t localStart, uint32_t localRankSize,
    uint32_t myLocalRank, uint64_t roundOffset, uint64_t chunkCount, uint64_t slotStride,
    uint32_t dataTypeSize, uint64_t &receiveCount)
{
    ThreadHandle mainThread = resource.aicpuThread;
    const SliceInfo mySlice = GetSlice(param.count, localRankSize, myLocalRank);
    receiveCount = roundOffset < mySlice.count ? std::min(chunkCount, mySlice.count - roundOffset) : 0;

    CHK_RET(StartLocalThreads(resource, localChannels.size()));
    for (uint32_t idx = 0; idx < localChannels.size(); ++idx) {
        const ChannelInfo &channel = *localChannels[idx];
        const uint32_t peerLocalRank = channel.remoteRank - localStart;
        const SliceInfo peerSlice = GetSlice(param.count, localRankSize, peerLocalRank);
        const uint64_t sendCount = roundOffset < peerSlice.count
            ? std::min(chunkCount, peerSlice.count - roundOffset)
            : 0;
        const uint64_t sendBytes = sendCount * dataTypeSize;
        const void *sendPtr = AddBytes(param.inputPtr, (peerSlice.offset + roundOffset) * dataTypeSize);
        ThreadHandle thread = resource.threads[idx + 1];

        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(thread, 0, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        if (sendBytes != 0) {
            CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(thread, channel.handle,
                AddBytes(channel.remoteCclMem.addr, myLocalRank * slotStride), sendPtr, sendBytes)));
        }
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
        CHK_RET(FinishLocalTransfer(thread, mainThread, idx + 1));
    }
    CHK_RET(WaitLocalThreads(resource, localChannels.size()));

    if (receiveCount != 0) {
        const uint64_t receiveBytes = receiveCount * dataTypeSize;
        const uint64_t outputOffset = (mySlice.offset + roundOffset) * dataTypeSize;
        void *output = AddBytes(param.outputPtr, outputOffset);
        const void *input = AddBytes(param.inputPtr, outputOffset);
        if (output != input) {
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread, output, input, receiveBytes)));
        }
        for (const ChannelInfo *channel : localChannels) {
            const uint32_t peerLocalRank = channel->remoteRank - localStart;
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(mainThread, output,
                AddBytes(resource.localBuffer.addr, peerLocalRank * slotStride), receiveCount,
                static_cast<HcommDataType>(param.dataType), static_cast<HcommReduceOp>(param.reduceType))));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RunAllGatherChunk(const OpParam &param, const AlgResourceCtx &resource,
    const std::vector<const ChannelInfo *> &localChannels, uint32_t localStart, uint32_t localRankSize,
    uint32_t myLocalRank, uint64_t roundOffset, uint64_t receiveCount, uint64_t slotStride,
    uint32_t dataTypeSize)
{
    ThreadHandle mainThread = resource.aicpuThread;
    const SliceInfo mySlice = GetSlice(param.count, localRankSize, myLocalRank);
    const uint64_t sendBytes = receiveCount * dataTypeSize;
    const void *sendPtr = AddBytes(param.outputPtr, (mySlice.offset + roundOffset) * dataTypeSize);

    CHK_RET(StartLocalThreads(resource, localChannels.size()));
    for (uint32_t idx = 0; idx < localChannels.size(); ++idx) {
        const ChannelInfo &channel = *localChannels[idx];
        ThreadHandle thread = resource.threads[idx + 1];
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(thread, 0, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        if (sendBytes != 0) {
            CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(thread, channel.handle,
                AddBytes(channel.remoteCclMem.addr, myLocalRank * slotStride), sendPtr, sendBytes)));
        }
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
        CHK_RET(FinishLocalTransfer(thread, mainThread, idx + 1));
    }
    CHK_RET(WaitLocalThreads(resource, localChannels.size()));

    for (const ChannelInfo *channel : localChannels) {
        const uint32_t peerLocalRank = channel->remoteRank - localStart;
        const SliceInfo peerSlice = GetSlice(param.count, localRankSize, peerLocalRank);
        const uint64_t peerCount = roundOffset < peerSlice.count
            ? std::min(slotStride / dataTypeSize, peerSlice.count - roundOffset)
            : 0;
        if (peerCount == 0) {
            continue;
        }
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread,
            AddBytes(param.outputPtr, (peerSlice.offset + roundOffset) * dataTypeSize),
            AddBytes(resource.localBuffer.addr, peerLocalRank * slotStride), peerCount * dataTypeSize)));
    }
    return HCCL_SUCCESS;
}

HcclResult RunTwoShot(const OpParam &param, const AlgResourceCtx &resource,
    const std::vector<const ChannelInfo *> &localChannels, const ChannelInfo *pairedChannel,
    uint64_t slotStride, uint64_t chunkCount, uint32_t dataTypeSize)
{
    const uint32_t localStart = param.myRank / RANKS_PER_SERVER * RANKS_PER_SERVER;
    const uint32_t localRankSize = std::min(localStart + RANKS_PER_SERVER, param.rankSize) - localStart;
    const uint32_t myLocalRank = param.myRank - localStart;
    const uint64_t maxSliceCount = (param.count + localRankSize - 1) / localRankSize;

    for (uint64_t roundOffset = 0; roundOffset < maxSliceCount; roundOffset += chunkCount) {
        uint64_t receiveCount = 0;
        CHK_RET(RunReduceScatterChunk(param, resource, localChannels, localStart, localRankSize,
            myLocalRank, roundOffset, chunkCount, slotStride, dataTypeSize, receiveCount));

        const SliceInfo mySlice = GetSlice(param.count, localRankSize, myLocalRank);
        void *output = AddBytes(param.outputPtr, (mySlice.offset + roundOffset) * dataTypeSize);
        CHK_RET(CrossServerReduce(param, resource, pairedChannel, output, receiveCount, dataTypeSize));

        CHK_RET(RunAllGatherChunk(param, resource, localChannels, localStart, localRankSize,
            myLocalRank, roundOffset, receiveCount, slotStride, dataTypeSize));
    }
    return HCCL_SUCCESS;
}

HcclResult ExecOptimized(const OpParam &param, const AlgResourceCtx &resource, uint32_t dataTypeSize)
{
    const uint32_t localStart = param.myRank / RANKS_PER_SERVER * RANKS_PER_SERVER;
    const uint32_t localRankSize = std::min(localStart + RANKS_PER_SERVER, param.rankSize) - localStart;
    CHK_PRT_RET(resource.threads.size() < localRankSize,
        HCCL_ERROR("Insufficient threads[%u], expected[%u]",
            static_cast<uint32_t>(resource.threads.size()), localRankSize), HCCL_E_INTERNAL);

    std::vector<const ChannelInfo *> localChannels;
    const ChannelInfo *pairedChannel = nullptr;
    CHK_RET(GetOptimizedChannels(param, resource, localChannels, pairedChannel));
    CHK_PRT_RET(localChannels.size() + 1 != localRankSize,
        HCCL_ERROR("Invalid local channel count[%u]", static_cast<uint32_t>(localChannels.size())), HCCL_E_INTERNAL);

    uint64_t slotStride = std::min(MAX_CHUNK_SIZE, resource.localBuffer.size / localRankSize);
    slotStride = slotStride / dataTypeSize * dataTypeSize;
    const uint64_t chunkCount = slotStride / dataTypeSize;
    CHK_PRT_RET(chunkCount == 0, HCCL_ERROR("HCCL buffer is too small"), HCCL_E_INTERNAL);
    for (const ChannelInfo *channel : localChannels) {
        CHK_PRT_RET(channel->remoteCclMem.size < slotStride * localRankSize,
            HCCL_ERROR("Remote HCCL buffer on rank[%u] is too small", channel->remoteRank), HCCL_E_INTERNAL);
    }
    if (pairedChannel != nullptr) {
        CHK_PRT_RET(pairedChannel->remoteCclMem.size < slotStride,
            HCCL_ERROR("Paired rank HCCL buffer is too small"), HCCL_E_INTERNAL);
    }

    const uint64_t totalBytes = param.count * dataTypeSize;
    if (totalBytes <= ONE_SHOT_LIMIT && totalBytes <= slotStride) {
        return RunOneShot(param, resource, localChannels, pairedChannel, slotStride, dataTypeSize);
    }
    return RunTwoShot(param, resource, localChannels, pairedChannel, slotStride, chunkCount, dataTypeSize);
}
}

HcclResult ExecHierarchical(const OpParam &param, const AlgResourceCtx &resCtx)
{
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }

    auto dataTypeIt = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(dataTypeIt == SIZE_TABLE.end(), HCCL_ERROR("Unsupported data type[%d]", param.dataType),
        HCCL_E_NOT_SUPPORT);
    const uint32_t dataTypeSize = dataTypeIt->second;
    const uint64_t chunkBytes = std::min(MAX_CHUNK_SIZE, resCtx.localBuffer.size);
    const uint64_t chunkCount = chunkBytes / dataTypeSize;
    CHK_PRT_RET(chunkCount == 0, HCCL_ERROR("HCCL buffer is smaller than one element"), HCCL_E_INTERNAL);

    const uint32_t localLeader = param.myRank / RANKS_PER_SERVER * RANKS_PER_SERVER;
    const bool isLeader = param.myRank == localLeader;
    if (isLeader || param.rankSize <= 1) {
        CHK_RET(CopyInputToOutput(param, resCtx.aicpuThread, chunkCount, dataTypeSize));
    }
    if (param.rankSize <= 1) {
        return HCCL_SUCCESS;
    }

    if (param.rankSize > RANKS_PER_SERVER) {
        if (!isLeader) {
            const ChannelInfo *channel = FindChannel(resCtx, localLeader);
            CHK_PRT_RET(channel == nullptr, HCCL_ERROR("Channel to local leader not found"), HCCL_E_INTERNAL);
            CHK_RET(SendToRoot(param, resCtx, *channel, chunkCount, dataTypeSize));
            CHK_RET(ReceiveFromRoot(param, resCtx, *channel, chunkCount, dataTypeSize));
            return HCCL_SUCCESS;
        }

        const uint32_t localEnd = std::min(localLeader + RANKS_PER_SERVER, param.rankSize);
        std::vector<const ChannelInfo *> localChannels;
        for (uint32_t remoteRank = localLeader; remoteRank < localEnd; ++remoteRank) {
            if (remoteRank == localLeader) {
                continue;
            }
            const ChannelInfo *channel = FindChannel(resCtx, remoteRank);
            CHK_PRT_RET(channel == nullptr, HCCL_ERROR("Channel to rank[%u] not found", remoteRank), HCCL_E_INTERNAL);
            localChannels.push_back(channel);
        }
        CHK_RET(ReducePeersChunkMajor(param, resCtx, localChannels, chunkCount, dataTypeSize));

        const uint32_t peerLeader = localLeader == ROOT_RANK ? RANKS_PER_SERVER : ROOT_RANK;
        const ChannelInfo *leaderChannel = FindChannel(resCtx, peerLeader);
        CHK_PRT_RET(leaderChannel == nullptr, HCCL_ERROR("Channel to peer leader not found"), HCCL_E_INTERNAL);
        if (localLeader == ROOT_RANK) {
            CHK_RET(ReducePeer(param, resCtx, *leaderChannel, chunkCount, dataTypeSize));
            CHK_RET(BroadcastToPeer(param, resCtx, *leaderChannel, chunkCount, dataTypeSize));
        } else {
            OpParam leaderParam = param;
            leaderParam.inputPtr = param.outputPtr;
            CHK_RET(SendToRoot(leaderParam, resCtx, *leaderChannel, chunkCount, dataTypeSize));
            CHK_RET(ReceiveFromRoot(param, resCtx, *leaderChannel, chunkCount, dataTypeSize));
        }

        CHK_RET(BroadcastPeersChunkMajor(param, resCtx, localChannels, chunkCount, dataTypeSize));
    } else if (param.myRank == ROOT_RANK) {
        std::vector<const ChannelInfo *> channels;
        for (uint32_t remoteRank = 1; remoteRank < param.rankSize; ++remoteRank) {
            const ChannelInfo *channel = FindChannel(resCtx, remoteRank);
            CHK_PRT_RET(channel == nullptr, HCCL_ERROR("Channel to rank[%u] not found", remoteRank), HCCL_E_INTERNAL);
            channels.push_back(channel);
        }
        CHK_RET(ReducePeersChunkMajor(param, resCtx, channels, chunkCount, dataTypeSize));
        CHK_RET(BroadcastPeersChunkMajor(param, resCtx, channels, chunkCount, dataTypeSize));
    } else {
        const ChannelInfo *channel = FindChannel(resCtx, ROOT_RANK);
        CHK_PRT_RET(channel == nullptr, HCCL_ERROR("Channel to root not found"), HCCL_E_INTERNAL);
        CHK_RET(SendToRoot(param, resCtx, *channel, chunkCount, dataTypeSize));
        CHK_RET(ReceiveFromRoot(param, resCtx, *channel, chunkCount, dataTypeSize));
    }

    return HCCL_SUCCESS;
}

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }

    auto dataTypeIt = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(dataTypeIt == SIZE_TABLE.end(), HCCL_ERROR("Unsupported data type[%d]", param.dataType),
        HCCL_E_NOT_SUPPORT);
    const uint32_t dataTypeSize = dataTypeIt->second;

    if (param.rankSize <= 1) {
        const uint64_t chunkCount = std::max<uint64_t>(1, MAX_CHUNK_SIZE / dataTypeSize);
        return CopyInputToOutput(param, resCtx.aicpuThread, chunkCount, dataTypeSize);
    }
    if (param.rankSize <= RANKS_PER_SERVER || param.rankSize == 2 * RANKS_PER_SERVER) {
        return ExecOptimized(param, resCtx, dataTypeSize);
    }
    return ExecHierarchical(param, resCtx);
}
} // namespace ops_hccl
