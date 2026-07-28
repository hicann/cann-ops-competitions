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

namespace ops_hccl {
namespace {
constexpr uint64_t MAX_TRANSFER_CHUNK_SIZE = 256ULL * 1024ULL * 1024ULL;
constexpr uint64_t FAST_WRITE_WITH_NOTIFY_LIMIT = 512ULL * 1024ULL;
constexpr uint64_t DIRECT_BROADCAST_MAX_SIZE = 512ULL * 1024ULL;
constexpr uint32_t NOTIFY_IDX_WORKER_READY = 0;
constexpr uint32_t NOTIFY_IDX_WORKER_DONE_BASE = 2;
constexpr uint32_t NOTIFY_IDX_ROOT_SLICE = 2;

struct DataSlice {
    uint64_t offset;
    uint64_t size;
};

struct TransferSlice {
    void *dst;
    void *src;
    uint64_t size;
};

uint32_t GetDataTypeSize(HcclDataType dataType)
{
    if (dataType == HCCL_DATA_TYPE_FP32) {
        return sizeof(float);
    }
    return 0;
}

DataSlice GetRankSlice(const OpParam &param, uint32_t rank, uint32_t typeSize)
{
    uint64_t sliceCount = param.count / param.rankSize + (param.count % param.rankSize != 0 ? 1 : 0);
    uint64_t startCount = std::min(param.count, sliceCount * rank);
    uint64_t endCount = std::min(param.count, startCount + sliceCount);
    return DataSlice{startCount * typeSize, (endCount - startCount) * typeSize};
}

DataSlice GetOwnerSlice(const OpParam &param, uint32_t rank, uint32_t typeSize)
{
    if (rank == param.root || param.rankSize <= 1) {
        return DataSlice{0, 0};
    }
    uint32_t ownerCount = param.rankSize - 1;
    uint32_t ownerIndex = rank < param.root ? rank : rank - 1;
    uint64_t sliceCount = param.count / ownerCount + (param.count % ownerCount != 0 ? 1 : 0);
    uint64_t startCount = std::min(param.count, sliceCount * ownerIndex);
    uint64_t endCount = std::min(param.count, startCount + sliceCount);
    return DataSlice{startCount * typeSize, (endCount - startCount) * typeSize};
}

void *OffsetPtr(void *ptr, uint64_t offset)
{
    return static_cast<void *>(static_cast<uint8_t *>(ptr) + offset);
}

const ChannelInfo *FindChannel(const AlgResourceCtx &resCtx, uint32_t remoteRank)
{
    for (const auto &channel : resCtx.channels) {
        if (channel.remoteRank == remoteRank) {
            return &channel;
        }
    }
    return nullptr;
}

HcclResult CheckParam(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t &dataSize, uint64_t &chunkSize)
{
    CHK_PTR_NULL(param.inputPtr);
    CHK_PTR_NULL(param.outputPtr);
    CHK_PRT_RET(resCtx.threads.empty(), HCCL_ERROR("AICPU thread resource is empty"), HCCL_E_INTERNAL);
    uint32_t typeSize = GetDataTypeSize(param.dataType);
    CHK_PRT_RET(typeSize == 0, HCCL_ERROR("Unsupported dataType[%d]", static_cast<int32_t>(param.dataType)), HCCL_E_PARA);
    CHK_PRT_RET(param.root >= param.rankSize, HCCL_ERROR("Invalid root[%u], rankSize[%u]", param.root, param.rankSize),
        HCCL_E_PARA);
    CHK_PRT_RET(param.count > UINT64_MAX / typeSize,
        HCCL_ERROR("Data size overflow, count[%llu], typeSize[%u]", static_cast<unsigned long long>(param.count),
            typeSize),
        HCCL_E_PARA);

    dataSize = param.count * typeSize;
    uint64_t resourceChunkSize = resCtx.localBuffer.size == 0 ? MAX_TRANSFER_CHUNK_SIZE : resCtx.localBuffer.size;
    chunkSize = std::min(dataSize, std::min(resourceChunkSize, MAX_TRANSFER_CHUNK_SIZE));
    CHK_PRT_RET(chunkSize == 0, HCCL_ERROR("Invalid transfer chunk size 0"), HCCL_E_INTERNAL);
    bool useTwoShot = dataSize > DIRECT_BROADCAST_MAX_SIZE;
    size_t expectedChannelNum = (param.myRank == param.root || useTwoShot) ? param.rankSize - 1 : 1;
    CHK_PRT_RET(resCtx.channels.size() != expectedChannelNum,
        HCCL_ERROR("Channel count[%zu] does not match expected count[%zu]", resCtx.channels.size(),
            expectedChannelNum),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.threads.size() != expectedChannelNum,
        HCCL_ERROR("Thread count[%zu] does not match expected count[%zu]", resCtx.threads.size(),
            expectedChannelNum),
        HCCL_E_INTERNAL);
    for (const auto &channel : resCtx.channels) {
        CHK_PRT_RET(channel.remoteRank >= param.rankSize || channel.remoteRank == param.myRank,
            HCCL_ERROR("Invalid channel remoteRank[%u], myRank[%u], rankSize[%u]", channel.remoteRank,
                param.myRank, param.rankSize),
            HCCL_E_INTERNAL);
        CHK_PTR_NULL(channel.remoteUserMem.addr);
        CHK_PRT_RET(channel.remoteUserMem.size < dataSize,
            HCCL_ERROR("Remote user memory too small, remoteRank[%u], memSize[%llu], dataSize[%llu]",
                channel.remoteRank, static_cast<unsigned long long>(channel.remoteUserMem.size),
                static_cast<unsigned long long>(dataSize)),
            HCCL_E_INTERNAL);
        uint32_t requiredNotifyIdx = useTwoShot ? NOTIFY_IDX_ROOT_SLICE : NOTIFY_IDX_DATA_SIGNAL;
        CHK_PRT_RET(channel.notifyNum <= requiredNotifyIdx,
            HCCL_ERROR("Invalid channel notifyNum[%u], remoteRank[%u]", channel.notifyNum, channel.remoteRank),
            HCCL_E_INTERNAL);
    }
    if (param.myRank != param.root) {
        CHK_PRT_RET(resCtx.channels[0].remoteRank != param.root,
            HCCL_ERROR("Root channel is not assigned to the main thread, remoteRank[%u], root[%u]",
                resCtx.channels[0].remoteRank, param.root),
            HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult PreSyncWorkers(const std::vector<ThreadHandle> &threads)
{
    ThreadHandle mainThread = threads[0];
    for (size_t workerIdx = 1; workerIdx < threads.size(); ++workerIdx) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(mainThread, threads[workerIdx], NOTIFY_IDX_WORKER_READY)));
    }
    for (size_t workerIdx = 1; workerIdx < threads.size(); ++workerIdx) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(threads[workerIdx], NOTIFY_IDX_WORKER_READY, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult PostSyncWorkers(const std::vector<ThreadHandle> &threads)
{
    ThreadHandle mainThread = threads[0];
    for (size_t workerIdx = 1; workerIdx < threads.size(); ++workerIdx) {
        uint32_t doneNotifyIdx = NOTIFY_IDX_WORKER_DONE_BASE + workerIdx - 1;
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(mainThread, doneNotifyIdx, CUSTOM_TIMEOUT)));
    }
    for (size_t workerIdx = 1; workerIdx < threads.size(); ++workerIdx) {
        uint32_t doneNotifyIdx = NOTIFY_IDX_WORKER_DONE_BASE + workerIdx - 1;
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(threads[workerIdx], mainThread, doneNotifyIdx)));
    }
    return HCCL_SUCCESS;
}

HcclResult WriteSlicesAndSignal(ThreadHandle thread, const ChannelInfo &channel, const TransferSlice *slices,
    uint32_t sliceNum, uint64_t chunkSize, uint32_t remoteNotifyIdx)
{
    CHK_PTR_NULL(slices);
    CHK_PRT_RET(sliceNum == 0, HCCL_ERROR("Transfer slice count is 0"), HCCL_E_PARA);
    if (sliceNum == 1 && slices[0].size <= FAST_WRITE_WITH_NOTIFY_LIMIT) {
        return static_cast<HcclResult>(HcommWriteWithNotifyOnThread(
            thread, channel.handle, slices[0].dst, slices[0].src, slices[0].size, remoteNotifyIdx));
    }

    std::vector<HcommBatchTransferDesc> transferDescs;
    uint64_t transferDescNum = 1;
    for (uint32_t sliceIdx = 0; sliceIdx < sliceNum; ++sliceIdx) {
        CHK_PTR_NULL(slices[sliceIdx].dst);
        CHK_PTR_NULL(slices[sliceIdx].src);
        CHK_PRT_RET(slices[sliceIdx].size == 0, HCCL_ERROR("Transfer slice[%u] size is 0", sliceIdx),
            HCCL_E_PARA);
        transferDescNum += (slices[sliceIdx].size - 1) / chunkSize + 1;
    }
    CHK_PRT_RET(transferDescNum >= UINT32_MAX,
        HCCL_ERROR("Too many batch transfer descriptors[%llu]",
            static_cast<unsigned long long>(transferDescNum)),
        HCCL_E_PARA);
    transferDescs.reserve(static_cast<size_t>(transferDescNum));
    for (uint32_t sliceIdx = 0; sliceIdx < sliceNum; ++sliceIdx) {
        const TransferSlice &slice = slices[sliceIdx];
        for (uint64_t offset = 0; offset < slice.size; offset += chunkSize) {
            HcommBatchTransferDesc writeDesc{};
            writeDesc.transType = HCOMM_TRANSFER_TYPE_WRITE;
            writeDesc.transferInfo.write.len = std::min(chunkSize, slice.size - offset);
            writeDesc.transferInfo.write.dst = OffsetPtr(slice.dst, offset);
            writeDesc.transferInfo.write.src = OffsetPtr(slice.src, offset);
            transferDescs.push_back(writeDesc);
        }
    }

    HcommBatchTransferDesc notifyDesc{};
    notifyDesc.transType = HCOMM_TRANSFER_TYPE_NOTIFY_RECORD;
    notifyDesc.transferInfo.notifyRecord.notifyIdx = remoteNotifyIdx;
    transferDescs.push_back(notifyDesc);
    return static_cast<HcclResult>(HcommBatchTransferOnThread(
        thread, channel.handle, transferDescs.data(), static_cast<uint32_t>(transferDescs.size())));
}

HcclResult WriteAndSignal(ThreadHandle thread, const ChannelInfo &channel, void *dst, void *src, uint64_t dataSize,
    uint64_t chunkSize, uint32_t remoteNotifyIdx)
{
    TransferSlice slice{dst, src, dataSize};
    return WriteSlicesAndSignal(thread, channel, &slice, 1, chunkSize, remoteNotifyIdx);
}

HcclResult RunDirectRoot(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t dataSize, uint64_t chunkSize)
{
    CHK_RET(PreSyncWorkers(resCtx.threads));
    for (size_t channelIdx = 0; channelIdx < resCtx.channels.size(); ++channelIdx) {
        const ChannelInfo &channel = resCtx.channels[channelIdx];
        ThreadHandle thread = resCtx.threads[channelIdx];
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        CHK_RET(WriteAndSignal(
            thread, channel, channel.remoteUserMem.addr, param.inputPtr, dataSize, chunkSize,
            NOTIFY_IDX_DATA_SIGNAL));
    }
    CHK_RET(PostSyncWorkers(resCtx.threads));
    return HCCL_SUCCESS;
}

HcclResult RunDirectPeer(const OpParam &param, const AlgResourceCtx &resCtx)
{
    const ChannelInfo *rootChannel = FindChannel(resCtx, param.root);
    CHK_PTR_NULL(rootChannel);
    ThreadHandle thread = resCtx.threads[0];
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, rootChannel->handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, rootChannel->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

bool CanUseDistributedStaging(const OpParam &param, const AlgResourceCtx &resCtx, uint32_t typeSize)
{
    if (param.rankSize <= 1 || resCtx.localBuffer.addr == nullptr) {
        return false;
    }
    uint64_t ownerCount = param.rankSize - 1;
    uint64_t maxSliceCount = param.count / ownerCount + (param.count % ownerCount != 0 ? 1 : 0);
    uint64_t maxSliceSize = maxSliceCount * typeSize;
    if (maxSliceSize == 0 || resCtx.localBuffer.size < maxSliceSize) {
        return false;
    }
    for (const auto &channel : resCtx.channels) {
        if (channel.remoteCclMem.addr == nullptr || channel.remoteCclMem.size < maxSliceSize) {
            return false;
        }
    }
    return true;
}

HcclResult RunDistributedRoot(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t chunkSize,
    uint32_t typeSize)
{
    CHK_RET(PreSyncWorkers(resCtx.threads));
    for (size_t channelIdx = 0; channelIdx < resCtx.channels.size(); ++channelIdx) {
        const ChannelInfo &channel = resCtx.channels[channelIdx];
        ThreadHandle thread = resCtx.threads[channelIdx];
        DataSlice ownerSlice = GetOwnerSlice(param, channel.remoteRank, typeSize);
        CHK_PRT_RET(ownerSlice.size == 0,
            HCCL_ERROR("Invalid owner slice, remoteRank[%u]", channel.remoteRank), HCCL_E_INTERNAL);
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        CHK_RET(WriteAndSignal(thread, channel, channel.remoteCclMem.addr,
            OffsetPtr(param.inputPtr, ownerSlice.offset), ownerSlice.size, chunkSize,
            NOTIFY_IDX_DATA_SIGNAL));
    }
    CHK_RET(PostSyncWorkers(resCtx.threads));
    return HCCL_SUCCESS;
}

HcclResult RunDistributedOwner(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t chunkSize,
    uint32_t typeSize)
{
    const ChannelInfo &rootChannel = resCtx.channels[0];
    ThreadHandle mainThread = resCtx.threads[0];
    DataSlice localSlice = GetOwnerSlice(param, param.myRank, typeSize);
    CHK_PRT_RET(localSlice.size == 0, HCCL_ERROR("Invalid local owner slice"), HCCL_E_INTERNAL);

    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(mainThread, rootChannel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        mainThread, rootChannel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread,
        OffsetPtr(param.outputPtr, localSlice.offset), resCtx.localBuffer.addr, localSlice.size)));

    // The local copy is ordered before these records on the main thread, so
    // every worker observes a complete owner slice in the shared CCL buffer.
    CHK_RET(PreSyncWorkers(resCtx.threads));
    for (size_t channelIdx = 1; channelIdx < resCtx.channels.size(); ++channelIdx) {
        const ChannelInfo &channel = resCtx.channels[channelIdx];
        ThreadHandle thread = resCtx.threads[channelIdx];
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        CHK_RET(WriteAndSignal(thread, channel,
            OffsetPtr(channel.remoteUserMem.addr, localSlice.offset), resCtx.localBuffer.addr,
            localSlice.size, chunkSize, NOTIFY_IDX_DATA_SIGNAL));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    }
    CHK_RET(PostSyncWorkers(resCtx.threads));
    return HCCL_SUCCESS;
}

HcclResult RunTwoShotRoot(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t chunkSize,
    uint32_t typeSize)
{
    CHK_RET(PreSyncWorkers(resCtx.threads));
    DataSlice rootSlice = GetRankSlice(param, param.root, typeSize);
    for (size_t channelIdx = 0; channelIdx < resCtx.channels.size(); ++channelIdx) {
        const ChannelInfo &channel = resCtx.channels[channelIdx];
        ThreadHandle thread = resCtx.threads[channelIdx];
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));

        DataSlice remoteSlice = GetRankSlice(param, channel.remoteRank, typeSize);
        TransferSlice remoteRankSlice{OffsetPtr(channel.remoteUserMem.addr, remoteSlice.offset),
            OffsetPtr(param.inputPtr, remoteSlice.offset), remoteSlice.size};
        CHK_RET(WriteSlicesAndSignal(
            thread, channel, &remoteRankSlice, 1, chunkSize, NOTIFY_IDX_DATA_SIGNAL));

        // Let the peer fan out its own slice while the root slice continues on
        // this channel. The second notify protects peer completion.
        TransferSlice rootRankSlice{OffsetPtr(channel.remoteUserMem.addr, rootSlice.offset),
            OffsetPtr(param.inputPtr, rootSlice.offset), rootSlice.size};
        CHK_RET(WriteSlicesAndSignal(
            thread, channel, &rootRankSlice, 1, chunkSize, NOTIFY_IDX_ROOT_SLICE));
    }
    CHK_RET(PostSyncWorkers(resCtx.threads));
    return HCCL_SUCCESS;
}

HcclResult RunTwoShotPeer(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t chunkSize,
    uint32_t typeSize)
{
    const ChannelInfo &rootChannel = resCtx.channels[0];
    ThreadHandle mainThread = resCtx.threads[0];
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(mainThread, rootChannel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(mainThread, rootChannel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));

    CHK_RET(PreSyncWorkers(resCtx.threads));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        mainThread, rootChannel.handle, NOTIFY_IDX_ROOT_SLICE, CUSTOM_TIMEOUT)));
    DataSlice localSlice = GetRankSlice(param, param.myRank, typeSize);
    for (size_t channelIdx = 1; channelIdx < resCtx.channels.size(); ++channelIdx) {
        const ChannelInfo &channel = resCtx.channels[channelIdx];
        ThreadHandle thread = resCtx.threads[channelIdx];
        CHK_RET(WriteAndSignal(thread, channel, OffsetPtr(channel.remoteUserMem.addr, localSlice.offset),
            OffsetPtr(param.inputPtr, localSlice.offset), localSlice.size, chunkSize,
            NOTIFY_IDX_DATA_SIGNAL));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    }
    CHK_RET(PostSyncWorkers(resCtx.threads));
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    HCCL_INFO("Executing AICPU Kernel on Ascend NPU");

    if (param.count == 0 || param.rankSize <= 1) {
        return HCCL_SUCCESS;
    }

    uint64_t dataSize = 0;
    uint64_t chunkSize = 0;
    CHK_RET(CheckParam(param, resCtx, dataSize, chunkSize));
    if (dataSize > DIRECT_BROADCAST_MAX_SIZE) {
        uint32_t typeSize = GetDataTypeSize(param.dataType);
        bool useDistributedStaging = CanUseDistributedStaging(param, resCtx, typeSize);
        if (useDistributedStaging && param.myRank == param.root) {
            CHK_RET(RunDistributedRoot(param, resCtx, chunkSize, typeSize));
        } else if (useDistributedStaging) {
            CHK_RET(RunDistributedOwner(param, resCtx, chunkSize, typeSize));
        } else if (param.myRank == param.root) {
            CHK_RET(RunTwoShotRoot(param, resCtx, chunkSize, typeSize));
        } else {
            CHK_RET(RunTwoShotPeer(param, resCtx, chunkSize, typeSize));
        }
    } else if (param.myRank == param.root) {
        CHK_RET(RunDirectRoot(param, resCtx, dataSize, chunkSize));
    } else {
        CHK_RET(RunDirectPeer(param, resCtx));
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
