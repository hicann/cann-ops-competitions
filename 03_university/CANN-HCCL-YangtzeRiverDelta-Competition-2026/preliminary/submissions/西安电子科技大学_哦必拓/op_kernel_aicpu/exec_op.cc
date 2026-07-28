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
constexpr uint32_t MAIN_THREAD_INDEX = 0;
constexpr uint32_t FIRST_LOCAL_WORKER_INDEX = 1;
constexpr uint32_t CROSS_WORKER_INDEX = 8;
constexpr uint32_t LOCAL_PEER_COUNT = BROADCAST_SERVER_RANKS - 1;
constexpr uint32_t INGRESS_SLOT = BROADCAST_SERVER_RANKS - 1;
constexpr uint32_t MAX_PIPELINE_PIECES = 2;
constexpr uint64_t BALANCED_TREE_MIN_BYTES = 64ULL * 1024;
constexpr uint64_t BALANCED_TREE_MAX_BYTES = 1ULL * 1024 * 1024;
constexpr uint64_t STRIPED_MIN_BYTES = 32ULL * 1024 * 1024;
constexpr uint64_t DOUBLE_PIECE_MIN_BYTES = 256ULL * 1024 * 1024;
constexpr uint64_t MAX_TRANSFER_PIECE_BYTES = 64ULL * 1024 * 1024;
constexpr uint32_t DIRECT_PEER_COUNT = BROADCAST_RANK_SIZE - 1;
constexpr uint32_t DIRECT_SCATTER_READY_NOTIFY = 0;
constexpr uint32_t DIRECT_SCATTER_DATA_NOTIFY = 1;
constexpr uint32_t DIRECT_SCATTER_ACK_NOTIFY = 2;
constexpr uint32_t DIRECT_AG_READY_NOTIFY = 3;
constexpr uint32_t DIRECT_AG_ACK_NOTIFY = 4;

struct Segment {
    uint64_t countOffset;
    uint64_t count;
};

uint32_t DataTypeSize(HcclDataType dataType)
{
    switch (dataType) {
        case HCCL_DATA_TYPE_INT8:
        case HCCL_DATA_TYPE_UINT8:
        case HCCL_DATA_TYPE_HIF8:
        case HCCL_DATA_TYPE_FP8E4M3:
        case HCCL_DATA_TYPE_FP8E5M2:
        case HCCL_DATA_TYPE_FP8E8M0:
            return 1;
        case HCCL_DATA_TYPE_INT16:
        case HCCL_DATA_TYPE_FP16:
        case HCCL_DATA_TYPE_UINT16:
        case HCCL_DATA_TYPE_BFP16:
            return 2;
        case HCCL_DATA_TYPE_INT32:
        case HCCL_DATA_TYPE_FP32:
        case HCCL_DATA_TYPE_UINT32:
            return 4;
        case HCCL_DATA_TYPE_INT64:
        case HCCL_DATA_TYPE_UINT64:
        case HCCL_DATA_TYPE_FP64:
            return 8;
        case HCCL_DATA_TYPE_INT128:
            return 16;
        default:
            return 0;
    }
}

void *AddOffset(void *base, uint64_t offset)
{
    return static_cast<void *>(static_cast<uint8_t *>(base) + offset);
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

HcclResult RecordChannel(ThreadHandle thread, const ChannelInfo &channel, uint32_t notifyIdx)
{
    return static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, channel.handle, notifyIdx));
}

HcclResult WaitChannel(ThreadHandle thread, const ChannelInfo &channel, uint32_t notifyIdx)
{
    return static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, notifyIdx, CUSTOM_TIMEOUT));
}

HcclResult BatchWriteAndRecord(ThreadHandle thread, const ChannelInfo &channel, void *dst,
    const void *src, uint64_t bytes, uint32_t notifyIdx)
{
    HcommBatchTransferDesc descs[2] = {};
    descs[0].transType = HCOMM_TRANSFER_TYPE_WRITE;
    descs[0].transferInfo.write.dst = dst;
    descs[0].transferInfo.write.src = const_cast<void *>(src);
    descs[0].transferInfo.write.len = bytes;
    descs[1].transType = HCOMM_TRANSFER_TYPE_NOTIFY_RECORD;
    descs[1].transferInfo.notifyRecord.notifyIdx = notifyIdx;
    return static_cast<HcclResult>(HcommBatchTransferOnThread(thread, channel.handle, descs, 2));
}

HcclResult BatchReadAndRecord(ThreadHandle thread, const ChannelInfo &channel, void *dst,
    const void *src, uint64_t bytes, uint32_t notifyIdx)
{
    HcommBatchTransferDesc descs[2] = {};
    descs[0].transType = HCOMM_TRANSFER_TYPE_READ;
    descs[0].transferInfo.read.dst = dst;
    descs[0].transferInfo.read.src = const_cast<void *>(src);
    descs[0].transferInfo.read.len = bytes;
    descs[1].transType = HCOMM_TRANSFER_TYPE_NOTIFY_RECORD;
    descs[1].transferInfo.notifyRecord.notifyIdx = notifyIdx;
    return static_cast<HcclResult>(HcommBatchTransferOnThread(thread, channel.handle, descs, 2));
}

HcclResult StartWorker(ThreadHandle coordinator, ThreadHandle worker)
{
    return static_cast<HcclResult>(
        HcommThreadNotifyRecordOnThread(coordinator, worker, BROADCAST_WORKER_START_NOTIFY));
}

HcclResult WaitWorkerStart(ThreadHandle worker)
{
    return static_cast<HcclResult>(
        HcommThreadNotifyWaitOnThread(worker, BROADCAST_WORKER_START_NOTIFY, CUSTOM_TIMEOUT));
}

HcclResult SignalWorkerComplete(ThreadHandle worker, ThreadHandle coordinator, uint32_t notifyIdx)
{
    return static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(worker, coordinator, notifyIdx));
}

HcclResult JoinWorker(ThreadHandle coordinator, uint32_t notifyIdx)
{
    return static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(coordinator, notifyIdx, CUSTOM_TIMEOUT));
}

HcclResult SignalPieceReady(ThreadHandle coordinator, ThreadHandle worker, uint32_t notifyIdx)
{
    return static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(coordinator, worker, notifyIdx));
}

HcclResult WaitPieceReady(ThreadHandle worker, uint32_t notifyIdx)
{
    return static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(worker, notifyIdx, CUSTOM_TIMEOUT));
}

HcclResult ValidateChannel(const ChannelInfo &channel, uint32_t signalBase)
{
    CHK_PRT_RET(signalBase > channel.notifyNum ||
            BROADCAST_SIGNALS_PER_DIRECTION > channel.notifyNum - signalBase,
        HCCL_ERROR("Channel to rank[%u] has only [%u] notifies for signal base[%u]",
            channel.remoteRank, channel.notifyNum, signalBase),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult ValidateDirectChannel(const ChannelInfo &channel)
{
    CHK_PRT_RET(channel.notifyNum <= DIRECT_AG_ACK_NOTIFY,
        HCCL_ERROR("Channel to rank[%u] has only [%u] notifies for Direct AllGather",
            channel.remoteRank, channel.notifyNum),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult ValidateLocalWindow(const CommBuffer &buffer, uint64_t offset, uint64_t bytes)
{
    CHK_PRT_RET(offset > buffer.size || bytes > buffer.size - offset,
        HCCL_ERROR("Local CCL window overflow, offset[%llu], bytes[%llu], capacity[%llu]",
            static_cast<unsigned long long>(offset), static_cast<unsigned long long>(bytes),
            static_cast<unsigned long long>(buffer.size)),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult ValidateRemoteWindow(const ChannelInfo &channel, uint64_t offset, uint64_t bytes)
{
    CHK_PRT_RET(offset > channel.remoteCclMem.size || bytes > channel.remoteCclMem.size - offset,
        HCCL_ERROR("Remote CCL window overflow, rank[%u], offset[%llu], bytes[%llu], capacity[%llu]",
            channel.remoteRank, static_cast<unsigned long long>(offset),
            static_cast<unsigned long long>(bytes),
            static_cast<unsigned long long>(channel.remoteCclMem.size)),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult SendPiece(ThreadHandle thread, const ChannelInfo &channel, const void *src, uint64_t remoteOffset,
    uint64_t bytes, uint32_t signalBase = 0)
{
    CHK_RET(ValidateChannel(channel, signalBase));
    CHK_RET(ValidateRemoteWindow(channel, remoteOffset, bytes));
    CHK_RET(WaitChannel(thread, channel, signalBase));
    CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(thread, channel.handle,
        AddOffset(channel.remoteCclMem.addr, remoteOffset), const_cast<void *>(src), bytes)));
    CHK_RET(RecordChannel(thread, channel, signalBase + 1));
    return WaitChannel(thread, channel, signalBase + 2);
}

HcclResult ReceivePiece(ThreadHandle thread, const ChannelInfo &channel, void *dst, const CommBuffer &localBuffer,
    uint64_t localOffset, uint64_t bytes, uint32_t signalBase = 0)
{
    CHK_RET(ValidateChannel(channel, signalBase));
    CHK_RET(ValidateLocalWindow(localBuffer, localOffset, bytes));
    CHK_RET(RecordChannel(thread, channel, signalBase));
    CHK_RET(WaitChannel(thread, channel, signalBase + 1));
    CHK_RET(static_cast<HcclResult>(
        HcommLocalCopyOnThread(thread, dst, AddOffset(localBuffer.addr, localOffset), bytes)));
    return RecordChannel(thread, channel, signalBase + 2);
}

HcclResult ExchangePiece(ThreadHandle thread, const ChannelInfo &channel, const void *outgoingSrc,
    uint64_t remoteOffset, uint64_t outgoingBytes, void *incomingDst, const CommBuffer &localBuffer,
    uint64_t localOffset, uint64_t incomingBytes, uint32_t outgoingBase, uint32_t incomingBase)
{
    CHK_RET(ValidateChannel(channel, outgoingBase));
    CHK_RET(ValidateChannel(channel, incomingBase));
    CHK_RET(ValidateRemoteWindow(channel, remoteOffset, outgoingBytes));
    CHK_RET(ValidateLocalWindow(localBuffer, localOffset, incomingBytes));

    // Both peers publish incoming-ready before either waits for outgoing-ready.
    CHK_RET(RecordChannel(thread, channel, incomingBase));
    CHK_RET(WaitChannel(thread, channel, outgoingBase));
    CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(thread, channel.handle,
        AddOffset(channel.remoteCclMem.addr, remoteOffset), const_cast<void *>(outgoingSrc), outgoingBytes)));
    CHK_RET(RecordChannel(thread, channel, outgoingBase + 1));
    CHK_RET(WaitChannel(thread, channel, incomingBase + 1));
    CHK_RET(static_cast<HcclResult>(
        HcommLocalCopyOnThread(thread, incomingDst, AddOffset(localBuffer.addr, localOffset), incomingBytes)));
    CHK_RET(RecordChannel(thread, channel, incomingBase + 2));
    return WaitChannel(thread, channel, outgoingBase + 2);
}

HcclResult SendRange(ThreadHandle thread, const ChannelInfo &channel, void *buffer, const Segment &segment,
    uint64_t pieceCount, uint64_t remoteOffset, uint32_t dataTypeSize, uint32_t signalBase = 0)
{
    for (uint64_t offset = 0; offset < segment.count; offset += pieceCount) {
        uint64_t count = std::min(pieceCount, segment.count - offset);
        uint64_t byteOffset = (segment.countOffset + offset) * dataTypeSize;
        CHK_RET(SendPiece(thread, channel, AddOffset(buffer, byteOffset), remoteOffset,
            count * dataTypeSize, signalBase));
    }
    return HCCL_SUCCESS;
}

HcclResult ReceiveRange(ThreadHandle thread, const ChannelInfo &channel, void *buffer,
    const CommBuffer &localBuffer, const Segment &segment, uint64_t pieceCount, uint64_t localOffset,
    uint32_t dataTypeSize, uint32_t signalBase = 0)
{
    for (uint64_t offset = 0; offset < segment.count; offset += pieceCount) {
        uint64_t count = std::min(pieceCount, segment.count - offset);
        uint64_t byteOffset = (segment.countOffset + offset) * dataTypeSize;
        CHK_RET(ReceivePiece(thread, channel, AddOffset(buffer, byteOffset), localBuffer, localOffset,
            count * dataTypeSize, signalBase));
    }
    return HCCL_SUCCESS;
}

Segment SplitSegment(uint64_t count, uint32_t segmentIdx)
{
    uint64_t baseCount = count / BROADCAST_SERVER_RANKS;
    uint64_t extraCount = count % BROADCAST_SERVER_RANKS;
    Segment segment;
    segment.count = baseCount + (segmentIdx < extraCount ? 1 : 0);
    segment.countOffset = baseCount * segmentIdx + std::min<uint64_t>(segmentIdx, extraCount);
    return segment;
}

Segment SplitDirectSlice(uint64_t count, uint32_t rank)
{
    uint64_t baseCount = count / BROADCAST_RANK_SIZE;
    Segment slice{baseCount * rank, baseCount};
    if (rank == BROADCAST_RANK_SIZE - 1) {
        slice.count += count % BROADCAST_RANK_SIZE;
    }
    return slice;
}

uint32_t DirectPeerRank(uint32_t myRank, uint32_t peerIdx)
{
    return peerIdx < myRank ? peerIdx : peerIdx + 1;
}

Segment SplitPiece(const Segment &segment, uint32_t pieceIdx, uint32_t pieceCount)
{
    if (pieceCount == 1) {
        return segment;
    }
    uint64_t firstCount = segment.count / 2 + segment.count % 2;
    if (pieceIdx == 0) {
        return Segment{segment.countOffset, firstCount};
    }
    return Segment{segment.countOffset + firstCount, segment.count - firstCount};
}

uint32_t PhysicalRank(uint32_t serverBase, uint32_t rootLocalRank, uint32_t logicalRank)
{
    return serverBase + (rootLocalRank + logicalRank) % BROADCAST_SERVER_RANKS;
}

uint32_t SegmentIndex(uint32_t rank, uint32_t rootLocalRank)
{
    uint32_t localRank = rank % BROADCAST_SERVER_RANKS;
    return (localRank + BROADCAST_SERVER_RANKS - rootLocalRank) % BROADCAST_SERVER_RANKS;
}

uint32_t IncomingSlot(uint32_t sourceSegment, uint32_t destinationSegment)
{
    return sourceSegment < destinationSegment ? sourceSegment : sourceSegment - 1;
}

uint64_t CommonSlotStride(const AlgResourceCtx &resCtx, uint32_t slotCount, uint32_t dataTypeSize)
{
    uint64_t commonCapacity = resCtx.localBuffer.size;
    for (const ChannelInfo &channel : resCtx.channels) {
        commonCapacity = std::min(commonCapacity, channel.remoteCclMem.size);
    }
    uint64_t stride = commonCapacity / slotCount;
    return stride / dataTypeSize * dataTypeSize;
}

HcclResult BinomialBroadcast(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t pieceCount,
    uint32_t dataTypeSize)
{
    ThreadHandle thread = resCtx.aicpuThread;
    uint32_t logicalRank = (param.myRank + param.rankSize - param.root) % param.rankSize;
    for (uint64_t countOffset = 0; countOffset < param.count; countOffset += pieceCount) {
        Segment piece{countOffset, std::min(pieceCount, param.count - countOffset)};
        for (uint32_t distance = 1; distance < param.rankSize; distance <<= 1U) {
            if (logicalRank < distance) {
                uint32_t remoteLogicalRank = logicalRank + distance;
                if (remoteLogicalRank < param.rankSize) {
                    uint32_t remoteRank = (param.root + remoteLogicalRank) % param.rankSize;
                    const ChannelInfo *channel = FindChannel(resCtx, remoteRank);
                    CHK_PRT_RET(channel == nullptr,
                        HCCL_ERROR("Channel to rank[%u] not found", remoteRank), HCCL_E_INTERNAL);
                    CHK_RET(SendRange(thread, *channel, param.inputPtr, piece, piece.count, 0, dataTypeSize));
                }
            } else if (logicalRank < distance * 2U) {
                uint32_t remoteLogicalRank = logicalRank - distance;
                uint32_t remoteRank = (param.root + remoteLogicalRank) % param.rankSize;
                const ChannelInfo *channel = FindChannel(resCtx, remoteRank);
                CHK_PRT_RET(channel == nullptr,
                    HCCL_ERROR("Channel to rank[%u] not found", remoteRank), HCCL_E_INTERNAL);
                CHK_RET(ReceiveRange(
                    thread, *channel, param.inputPtr, resCtx.localBuffer, piece, piece.count, 0, dataTypeSize));
            }
            if (distance > param.rankSize / 2) {
                break;
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult BalancedThreeFanoutBroadcast(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t totalBytes)
{
    CHK_PRT_RET(resCtx.threads.size() != BROADCAST_THREAD_COUNT,
        HCCL_ERROR("Unexpected thread count[%zu]", resCtx.threads.size()), HCCL_E_INTERNAL);
    CHK_RET(ValidateLocalWindow(resCtx.localBuffer, 0, totalBytes));

    uint32_t rootServerBase = (param.root / BROADCAST_SERVER_RANKS) * BROADCAST_SERVER_RANKS;
    uint32_t remoteServerBase = rootServerBase ^ BROADCAST_SERVER_RANKS;
    uint32_t rootLocalRank = param.root % BROADCAST_SERVER_RANKS;
    uint32_t localProxy = PhysicalRank(rootServerBase, rootLocalRank, 4);
    uint32_t remoteProxy0 = PhysicalRank(remoteServerBase, rootLocalRank, 0);
    uint32_t remoteProxy4 = PhysicalRank(remoteServerBase, rootLocalRank, 4);
    ThreadHandle coordinator = resCtx.threads[MAIN_THREAD_INDEX];

    if (param.myRank == param.root) {
        uint32_t firstWaveRanks[3] = {localProxy, remoteProxy0, remoteProxy4};
        uint32_t secondWaveRanks[3] = {
            PhysicalRank(rootServerBase, rootLocalRank, 1),
            PhysicalRank(rootServerBase, rootLocalRank, 2),
            PhysicalRank(rootServerBase, rootLocalRank, 3)};
        const ChannelInfo *firstWave[3] = {};
        const ChannelInfo *secondWave[3] = {};
        for (uint32_t idx = 0; idx < 3; idx++) {
            firstWave[idx] = FindChannel(resCtx, firstWaveRanks[idx]);
            secondWave[idx] = FindChannel(resCtx, secondWaveRanks[idx]);
            CHK_PRT_RET(firstWave[idx] == nullptr || secondWave[idx] == nullptr,
                HCCL_ERROR("Balanced-tree root channel missing for worker[%u]", idx), HCCL_E_INTERNAL);
            CHK_RET(ValidateChannel(*firstWave[idx], 0));
            CHK_RET(ValidateChannel(*secondWave[idx], 0));
            CHK_RET(ValidateRemoteWindow(*firstWave[idx], 0, totalBytes));
            CHK_RET(ValidateRemoteWindow(*secondWave[idx], 0, totalBytes));
        }

        for (uint32_t idx = 0; idx < 3; idx++) {
            ThreadHandle worker = resCtx.threads[FIRST_LOCAL_WORKER_INDEX + idx];
            CHK_RET(StartWorker(coordinator, worker));
            CHK_RET(WaitWorkerStart(worker));
            CHK_RET(SendPiece(worker, *firstWave[idx], param.inputPtr, 0, totalBytes));
            CHK_RET(SendPiece(worker, *secondWave[idx], param.inputPtr, 0, totalBytes));
            CHK_RET(SignalWorkerComplete(worker, coordinator, idx + 1));
        }
        for (uint32_t idx = 0; idx < 3; idx++) {
            CHK_RET(JoinWorker(coordinator, idx + 1));
        }
        return HCCL_SUCCESS;
    }

    uint32_t myServerBase = (param.myRank / BROADCAST_SERVER_RANKS) * BROADCAST_SERVER_RANKS;
    uint32_t myLogicalRank = SegmentIndex(param.myRank, rootLocalRank);
    bool isProxy = param.myRank == localProxy || param.myRank == remoteProxy0 || param.myRank == remoteProxy4;
    uint32_t parentRank = INVALID_VALUE_RANKID;
    uint32_t childRanks[3] = {};

    if (isProxy) {
        parentRank = param.root;
        uint32_t childBase = myLogicalRank == 4 ? 5 : 1;
        for (uint32_t idx = 0; idx < 3; idx++) {
            childRanks[idx] = PhysicalRank(myServerBase, rootLocalRank, childBase + idx);
        }
    } else if (myServerBase == rootServerBase && myLogicalRank >= 1 && myLogicalRank <= 3) {
        parentRank = param.root;
    } else if (myServerBase == rootServerBase && myLogicalRank >= 5) {
        parentRank = localProxy;
    } else if (myServerBase == remoteServerBase && myLogicalRank >= 1 && myLogicalRank <= 3) {
        parentRank = remoteProxy0;
    } else if (myServerBase == remoteServerBase && myLogicalRank >= 5) {
        parentRank = remoteProxy4;
    }

    const ChannelInfo *parentChannel = FindChannel(resCtx, parentRank);
    CHK_PRT_RET(parentRank == INVALID_VALUE_RANKID || parentChannel == nullptr,
        HCCL_ERROR("Balanced-tree parent channel missing for rank[%u]", param.myRank), HCCL_E_INTERNAL);
    CHK_RET(ValidateChannel(*parentChannel, 0));

    const ChannelInfo *childChannels[3] = {};
    if (isProxy) {
        for (uint32_t idx = 0; idx < 3; idx++) {
            childChannels[idx] = FindChannel(resCtx, childRanks[idx]);
            CHK_PRT_RET(childChannels[idx] == nullptr,
                HCCL_ERROR("Balanced-tree child channel to rank[%u] missing", childRanks[idx]),
                HCCL_E_INTERNAL);
            CHK_RET(ValidateChannel(*childChannels[idx], 0));
            CHK_RET(ValidateRemoteWindow(*childChannels[idx], 0, totalBytes));
        }
    }

    CHK_RET(ReceivePiece(coordinator, *parentChannel, param.inputPtr, resCtx.localBuffer, 0, totalBytes));
    if (!isProxy) {
        return HCCL_SUCCESS;
    }

    for (uint32_t idx = 0; idx < 3; idx++) {
        ThreadHandle worker = resCtx.threads[FIRST_LOCAL_WORKER_INDEX + idx];
        CHK_RET(StartWorker(coordinator, worker));
        CHK_RET(WaitWorkerStart(worker));
        CHK_RET(SendPiece(worker, *childChannels[idx], param.inputPtr, 0, totalBytes));
        CHK_RET(SignalWorkerComplete(worker, coordinator, idx + 1));
    }
    for (uint32_t idx = 0; idx < 3; idx++) {
        CHK_RET(JoinWorker(coordinator, idx + 1));
    }
    return HCCL_SUCCESS;
}

HcclResult StartFlatSends(const OpParam &param, const AlgResourceCtx &resCtx, uint32_t serverBase,
    uint32_t excludedRank, const Segment &segment, uint64_t pieceCount, uint32_t dataTypeSize,
    uint32_t &workerCount)
{
    const ChannelInfo *channels[LOCAL_PEER_COUNT] = {};
    workerCount = 0;
    for (uint32_t localRank = 0; localRank < BROADCAST_SERVER_RANKS; localRank++) {
        uint32_t remoteRank = serverBase + localRank;
        if (remoteRank == excludedRank) {
            continue;
        }
        channels[workerCount] = FindChannel(resCtx, remoteRank);
        CHK_PRT_RET(channels[workerCount] == nullptr,
            HCCL_ERROR("Channel to rank[%u] not found", remoteRank), HCCL_E_INTERNAL);
        CHK_RET(ValidateChannel(*channels[workerCount], 0));
        workerCount++;
    }

    ThreadHandle coordinator = resCtx.threads[MAIN_THREAD_INDEX];
    for (uint32_t idx = 0; idx < workerCount; idx++) {
        ThreadHandle worker = resCtx.threads[FIRST_LOCAL_WORKER_INDEX + idx];
        CHK_RET(StartWorker(coordinator, worker));
        CHK_RET(WaitWorkerStart(worker));
        CHK_RET(SendRange(worker, *channels[idx], param.inputPtr, segment, pieceCount, 0, dataTypeSize));
        CHK_RET(SignalWorkerComplete(worker, coordinator, idx + 1));
    }
    return HCCL_SUCCESS;
}

HcclResult JoinWorkers(ThreadHandle coordinator, uint32_t workerCount)
{
    for (uint32_t idx = 0; idx < workerCount; idx++) {
        CHK_RET(JoinWorker(coordinator, idx + 1));
    }
    return HCCL_SUCCESS;
}

HcclResult GroupedFlatBroadcast(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t pieceCount,
    uint32_t dataTypeSize)
{
    CHK_PRT_RET(resCtx.threads.size() != BROADCAST_THREAD_COUNT,
        HCCL_ERROR("Unexpected thread count[%zu]", resCtx.threads.size()), HCCL_E_INTERNAL);

    uint32_t rootServerBase = (param.root / BROADCAST_SERVER_RANKS) * BROADCAST_SERVER_RANKS;
    uint32_t mirrorRank = param.root ^ BROADCAST_SERVER_RANKS;
    Segment message{0, param.count};
    ThreadHandle coordinator = resCtx.threads[MAIN_THREAD_INDEX];

    if (param.myRank == param.root) {
        uint32_t workerCount = 0;
        CHK_RET(StartFlatSends(param, resCtx, rootServerBase, param.root, message, pieceCount,
            dataTypeSize, workerCount));
        const ChannelInfo *mirrorChannel = FindChannel(resCtx, mirrorRank);
        CHK_PRT_RET(mirrorChannel == nullptr,
            HCCL_ERROR("Channel to mirror rank[%u] not found", mirrorRank), HCCL_E_INTERNAL);
        CHK_RET(SendRange(coordinator, *mirrorChannel, param.inputPtr, message, pieceCount, 0, dataTypeSize));
        return JoinWorkers(coordinator, workerCount);
    }

    if (param.myRank == mirrorRank) {
        const ChannelInfo *rootChannel = FindChannel(resCtx, param.root);
        CHK_PRT_RET(rootChannel == nullptr,
            HCCL_ERROR("Channel to root rank[%u] not found", param.root), HCCL_E_INTERNAL);
        CHK_RET(ReceiveRange(coordinator, *rootChannel, param.inputPtr, resCtx.localBuffer, message,
            pieceCount, 0, dataTypeSize));
        uint32_t workerCount = 0;
        uint32_t serverBase = (param.myRank / BROADCAST_SERVER_RANKS) * BROADCAST_SERVER_RANKS;
        CHK_RET(StartFlatSends(param, resCtx, serverBase, mirrorRank, message, pieceCount,
            dataTypeSize, workerCount));
        return JoinWorkers(coordinator, workerCount);
    }

    uint32_t myServerBase = (param.myRank / BROADCAST_SERVER_RANKS) * BROADCAST_SERVER_RANKS;
    uint32_t sourceRank = myServerBase == rootServerBase ? param.root : mirrorRank;
    const ChannelInfo *sourceChannel = FindChannel(resCtx, sourceRank);
    CHK_PRT_RET(sourceChannel == nullptr,
        HCCL_ERROR("Channel to source rank[%u] not found", sourceRank), HCCL_E_INTERNAL);
    return ReceiveRange(coordinator, *sourceChannel, param.inputPtr, resCtx.localBuffer, message,
        pieceCount, 0, dataTypeSize);
}

HcclResult BeginDirectParallel(const AlgResourceCtx &resCtx)
{
    ThreadHandle coordinator = resCtx.threads[MAIN_THREAD_INDEX];
    for (uint32_t peerIdx = 1; peerIdx < DIRECT_PEER_COUNT; peerIdx++) {
        CHK_RET(StartWorker(coordinator, resCtx.threads[peerIdx]));
    }
    for (uint32_t peerIdx = 1; peerIdx < DIRECT_PEER_COUNT; peerIdx++) {
        CHK_RET(WaitWorkerStart(resCtx.threads[peerIdx]));
    }
    return HCCL_SUCCESS;
}

HcclResult EndDirectParallel(const AlgResourceCtx &resCtx)
{
    ThreadHandle coordinator = resCtx.threads[MAIN_THREAD_INDEX];
    for (uint32_t peerIdx = 1; peerIdx < DIRECT_PEER_COUNT; peerIdx++) {
        CHK_RET(SignalWorkerComplete(resCtx.threads[peerIdx], coordinator, peerIdx));
    }
    for (uint32_t peerIdx = 1; peerIdx < DIRECT_PEER_COUNT; peerIdx++) {
        CHK_RET(JoinWorker(coordinator, peerIdx));
    }
    return HCCL_SUCCESS;
}

HcclResult DirectScatter(const OpParam &param, const AlgResourceCtx &resCtx, uint32_t dataTypeSize)
{
    Segment mySlice = SplitDirectSlice(param.count, param.myRank);
    if (param.myRank == param.root) {
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(resCtx.threads[MAIN_THREAD_INDEX],
            resCtx.localBuffer.addr,
            AddOffset(param.inputPtr, mySlice.countOffset * dataTypeSize),
            mySlice.count * dataTypeSize)));
    }

    CHK_RET(BeginDirectParallel(resCtx));
    if (param.myRank == param.root) {
        for (uint32_t peerIdx = 0; peerIdx < DIRECT_PEER_COUNT; peerIdx++) {
            uint32_t peerRank = DirectPeerRank(param.myRank, peerIdx);
            const ChannelInfo *channel = FindChannel(resCtx, peerRank);
            CHK_PRT_RET(channel == nullptr,
                HCCL_ERROR("Direct scatter channel to rank[%u] not found", peerRank), HCCL_E_INTERNAL);
            CHK_RET(ValidateDirectChannel(*channel));
            Segment slice = SplitDirectSlice(param.count, peerRank);
            CHK_RET(ValidateRemoteWindow(*channel, 0, slice.count * dataTypeSize));
            ThreadHandle thread = resCtx.threads[peerIdx];
            CHK_RET(WaitChannel(thread, *channel, DIRECT_SCATTER_READY_NOTIFY));
            CHK_RET(BatchWriteAndRecord(thread, *channel, channel->remoteCclMem.addr,
                AddOffset(param.inputPtr, slice.countOffset * dataTypeSize),
                slice.count * dataTypeSize, DIRECT_SCATTER_DATA_NOTIFY));
            CHK_RET(WaitChannel(thread, *channel, DIRECT_SCATTER_ACK_NOTIFY));
        }
    } else {
        uint32_t rootPeerIdx = param.root < param.myRank ? param.root : param.root - 1;
        const ChannelInfo *rootChannel = FindChannel(resCtx, param.root);
        CHK_PRT_RET(rootChannel == nullptr,
            HCCL_ERROR("Direct scatter root channel to rank[%u] not found", param.root), HCCL_E_INTERNAL);
        CHK_RET(ValidateDirectChannel(*rootChannel));
        ThreadHandle thread = resCtx.threads[rootPeerIdx];
        CHK_RET(RecordChannel(thread, *rootChannel, DIRECT_SCATTER_READY_NOTIFY));
        CHK_RET(WaitChannel(thread, *rootChannel, DIRECT_SCATTER_DATA_NOTIFY));
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread,
            AddOffset(param.inputPtr, mySlice.countOffset * dataTypeSize),
            resCtx.localBuffer.addr, mySlice.count * dataTypeSize)));
        CHK_RET(RecordChannel(thread, *rootChannel, DIRECT_SCATTER_ACK_NOTIFY));
    }
    return EndDirectParallel(resCtx);
}

HcclResult DirectAllGather(const OpParam &param, const AlgResourceCtx &resCtx, uint32_t dataTypeSize)
{
    CHK_RET(BeginDirectParallel(resCtx));
    for (uint32_t peerIdx = 0; peerIdx < DIRECT_PEER_COUNT; peerIdx++) {
        uint32_t peerRank = DirectPeerRank(param.myRank, peerIdx);
        const ChannelInfo *channel = FindChannel(resCtx, peerRank);
        CHK_PRT_RET(channel == nullptr,
            HCCL_ERROR("Direct AllGather channel to rank[%u] not found", peerRank), HCCL_E_INTERNAL);
        CHK_RET(ValidateDirectChannel(*channel));
        Segment slice = SplitDirectSlice(param.count, peerRank);
        if (param.myRank != param.root) {
            CHK_RET(ValidateRemoteWindow(*channel, 0, slice.count * dataTypeSize));
        }
        ThreadHandle thread = resCtx.threads[peerIdx];
        CHK_RET(RecordChannel(thread, *channel, DIRECT_AG_READY_NOTIFY));
        CHK_RET(WaitChannel(thread, *channel, DIRECT_AG_READY_NOTIFY));
        if (param.myRank == param.root) {
            CHK_RET(RecordChannel(thread, *channel, DIRECT_AG_ACK_NOTIFY));
        } else {
            CHK_RET(BatchReadAndRecord(thread, *channel,
                AddOffset(param.inputPtr, slice.countOffset * dataTypeSize),
                channel->remoteCclMem.addr, slice.count * dataTypeSize, DIRECT_AG_ACK_NOTIFY));
        }
        CHK_RET(WaitChannel(thread, *channel, DIRECT_AG_ACK_NOTIFY));
    }
    return EndDirectParallel(resCtx);
}

HcclResult DirectReadAllGatherBroadcast(const OpParam &param, const AlgResourceCtx &resCtx,
    uint32_t dataTypeSize)
{
    CHK_PRT_RET(resCtx.threads.size() != DIRECT_PEER_COUNT,
        HCCL_ERROR("Unexpected Direct thread count[%zu]", resCtx.threads.size()), HCCL_E_INTERNAL);
    Segment mySlice = SplitDirectSlice(param.count, param.myRank);
    CHK_RET(ValidateLocalWindow(resCtx.localBuffer, 0, mySlice.count * dataTypeSize));
    CHK_RET(DirectScatter(param, resCtx, dataTypeSize));
    return DirectAllGather(param, resCtx, dataTypeSize);
}

bool DirectReadAllGatherFits(const OpParam &param, const AlgResourceCtx &resCtx, uint32_t dataTypeSize)
{
    uint64_t maxSliceBytes = SplitDirectSlice(param.count, BROADCAST_RANK_SIZE - 1).count * dataTypeSize;
    if (resCtx.localBuffer.size < maxSliceBytes) {
        return false;
    }
    for (const ChannelInfo &channel : resCtx.channels) {
        if (channel.remoteCclMem.size < maxSliceBytes || channel.notifyNum <= DIRECT_AG_ACK_NOTIFY) {
            return false;
        }
    }
    return true;
}

HcclResult ValidateExchange(const AlgResourceCtx &resCtx, const ChannelInfo &channel,
    uint32_t mySegmentIdx, uint32_t peerSegmentIdx, const Segment &outgoing, const Segment &incoming,
    uint64_t slotStrideBytes, uint32_t dataTypeSize)
{
    uint32_t outgoingBase = mySegmentIdx < peerSegmentIdx ? 0 : BROADCAST_SIGNALS_PER_DIRECTION;
    uint32_t incomingBase = mySegmentIdx < peerSegmentIdx ? BROADCAST_SIGNALS_PER_DIRECTION : 0;
    uint64_t remoteOffset = static_cast<uint64_t>(IncomingSlot(mySegmentIdx, peerSegmentIdx)) * slotStrideBytes;
    uint64_t localOffset = static_cast<uint64_t>(IncomingSlot(peerSegmentIdx, mySegmentIdx)) * slotStrideBytes;
    CHK_RET(ValidateChannel(channel, outgoingBase));
    CHK_RET(ValidateChannel(channel, incomingBase));
    CHK_RET(ValidateRemoteWindow(channel, remoteOffset, outgoing.count * dataTypeSize));
    return ValidateLocalWindow(resCtx.localBuffer, localOffset, incoming.count * dataTypeSize);
}

HcclResult RunExchange(ThreadHandle worker, const OpParam &param, const AlgResourceCtx &resCtx,
    const ChannelInfo &channel, uint32_t mySegmentIdx, uint32_t peerSegmentIdx,
    const Segment &outgoing, const Segment &incoming, uint64_t slotStrideBytes, uint32_t dataTypeSize)
{
    uint32_t outgoingBase = mySegmentIdx < peerSegmentIdx ? 0 : BROADCAST_SIGNALS_PER_DIRECTION;
    uint32_t incomingBase = mySegmentIdx < peerSegmentIdx ? BROADCAST_SIGNALS_PER_DIRECTION : 0;
    uint64_t remoteOffset = static_cast<uint64_t>(IncomingSlot(mySegmentIdx, peerSegmentIdx)) * slotStrideBytes;
    uint64_t localOffset = static_cast<uint64_t>(IncomingSlot(peerSegmentIdx, mySegmentIdx)) * slotStrideBytes;
    void *outgoingPtr = AddOffset(param.inputPtr, outgoing.countOffset * dataTypeSize);
    void *incomingPtr = AddOffset(param.inputPtr, incoming.countOffset * dataTypeSize);
    return ExchangePiece(worker, channel, outgoingPtr, remoteOffset, outgoing.count * dataTypeSize,
        incomingPtr, resCtx.localBuffer, localOffset, incoming.count * dataTypeSize,
        outgoingBase, incomingBase);
}

bool PipelinePiecesFit(const OpParam &param, uint32_t pipelinePieceCount, uint64_t slotStrideBytes,
    uint32_t dataTypeSize)
{
    for (uint32_t segmentIdx = 0; segmentIdx < BROADCAST_SERVER_RANKS; segmentIdx++) {
        Segment segment = SplitSegment(param.count, segmentIdx);
        for (uint32_t pieceIdx = 0; pieceIdx < pipelinePieceCount; pieceIdx++) {
            Segment piece = SplitPiece(segment, pieceIdx, pipelinePieceCount);
            if (piece.count == 0 || piece.count > slotStrideBytes / dataTypeSize) {
                return false;
            }
        }
    }
    return true;
}

HcclResult RootStripedPipeline(const OpParam &param, const AlgResourceCtx &resCtx,
    uint32_t pipelinePieceCount, uint64_t slotStrideBytes, uint32_t dataTypeSize,
    uint32_t rootServerBase, uint32_t remoteServerBase, uint32_t rootLocalRank)
{
    const ChannelInfo *localChannels[LOCAL_PEER_COUNT] = {};
    uint32_t peerSegments[LOCAL_PEER_COUNT] = {};
    Segment rootSegment = SplitSegment(param.count, 0);
    for (uint32_t idx = 0; idx < LOCAL_PEER_COUNT; idx++) {
        uint32_t peerSegment = idx + 1;
        uint32_t peerRank = PhysicalRank(rootServerBase, rootLocalRank, peerSegment);
        localChannels[idx] = FindChannel(resCtx, peerRank);
        peerSegments[idx] = peerSegment;
        CHK_PRT_RET(localChannels[idx] == nullptr,
            HCCL_ERROR("Root local channel to rank[%u] not found", peerRank), HCCL_E_INTERNAL);
        CHK_RET(ValidateChannel(*localChannels[idx], 0));
        CHK_RET(ValidateChannel(*localChannels[idx], BROADCAST_SIGNALS_PER_DIRECTION));
        for (uint32_t pieceIdx = 0; pieceIdx < pipelinePieceCount; pieceIdx++) {
            Segment scatterPiece = SplitPiece(SplitSegment(param.count, peerSegment),
                pieceIdx, pipelinePieceCount);
            Segment rootPiece = SplitPiece(rootSegment, pieceIdx, pipelinePieceCount);
            CHK_RET(ValidateRemoteWindow(*localChannels[idx], INGRESS_SLOT * slotStrideBytes,
                scatterPiece.count * dataTypeSize));
            CHK_RET(ValidateRemoteWindow(*localChannels[idx],
                static_cast<uint64_t>(IncomingSlot(0, peerSegment)) * slotStrideBytes,
                rootPiece.count * dataTypeSize));
        }
    }

    uint32_t mirrorRank = PhysicalRank(remoteServerBase, rootLocalRank, 0);
    const ChannelInfo *crossChannel = FindChannel(resCtx, mirrorRank);
    CHK_PRT_RET(crossChannel == nullptr,
        HCCL_ERROR("Root cross channel to rank[%u] not found", mirrorRank), HCCL_E_INTERNAL);
    CHK_RET(ValidateChannel(*crossChannel, 0));
    for (uint32_t pieceIdx = 0; pieceIdx < pipelinePieceCount; pieceIdx++) {
        Segment rootPiece = SplitPiece(rootSegment, pieceIdx, pipelinePieceCount);
        CHK_RET(ValidateRemoteWindow(*crossChannel, INGRESS_SLOT * slotStrideBytes,
            rootPiece.count * dataTypeSize));
    }

    ThreadHandle coordinator = resCtx.threads[MAIN_THREAD_INDEX];
    for (uint32_t idx = 0; idx < LOCAL_PEER_COUNT; idx++) {
        ThreadHandle worker = resCtx.threads[FIRST_LOCAL_WORKER_INDEX + idx];
        CHK_RET(StartWorker(coordinator, worker));
        CHK_RET(WaitWorkerStart(worker));
        for (uint32_t pieceIdx = 0; pieceIdx < pipelinePieceCount; pieceIdx++) {
            Segment scatterPiece = SplitPiece(SplitSegment(param.count, peerSegments[idx]),
                pieceIdx, pipelinePieceCount);
            Segment rootPiece = SplitPiece(rootSegment, pieceIdx, pipelinePieceCount);
            CHK_RET(SendPiece(worker, *localChannels[idx],
                AddOffset(param.inputPtr, scatterPiece.countOffset * dataTypeSize),
                INGRESS_SLOT * slotStrideBytes, scatterPiece.count * dataTypeSize, 0));
            CHK_RET(SendPiece(worker, *localChannels[idx],
                AddOffset(param.inputPtr, rootPiece.countOffset * dataTypeSize),
                static_cast<uint64_t>(IncomingSlot(0, peerSegments[idx])) * slotStrideBytes,
                rootPiece.count * dataTypeSize, BROADCAST_SIGNALS_PER_DIRECTION));
        }
        CHK_RET(SignalWorkerComplete(worker, coordinator, idx + 1));
    }

    ThreadHandle crossWorker = resCtx.threads[CROSS_WORKER_INDEX];
    CHK_RET(StartWorker(coordinator, crossWorker));
    CHK_RET(WaitWorkerStart(crossWorker));
    for (uint32_t pieceIdx = 0; pieceIdx < pipelinePieceCount; pieceIdx++) {
        Segment rootPiece = SplitPiece(rootSegment, pieceIdx, pipelinePieceCount);
        CHK_RET(SendPiece(crossWorker, *crossChannel,
            AddOffset(param.inputPtr, rootPiece.countOffset * dataTypeSize),
            INGRESS_SLOT * slotStrideBytes, rootPiece.count * dataTypeSize, 0));
    }
    CHK_RET(SignalWorkerComplete(crossWorker, coordinator, CROSS_WORKER_INDEX));

    for (uint32_t idx = 1; idx <= LOCAL_PEER_COUNT; idx++) {
        CHK_RET(JoinWorker(coordinator, idx));
    }
    return JoinWorker(coordinator, CROSS_WORKER_INDEX);
}

HcclResult SourceOwnerStripedPipeline(const OpParam &param, const AlgResourceCtx &resCtx,
    uint32_t pipelinePieceCount, uint64_t slotStrideBytes, uint32_t dataTypeSize,
    uint32_t rootServerBase, uint32_t rootLocalRank, uint32_t mySegmentIdx)
{
    const ChannelInfo *localChannels[LOCAL_PEER_COUNT] = {};
    uint32_t peerSegments[LOCAL_PEER_COUNT] = {};
    uint32_t rootWorkerSlot = INVALID_VALUE_RANKID;
    uint32_t workerSlot = 0;
    for (uint32_t peerSegment = 0; peerSegment < BROADCAST_SERVER_RANKS; peerSegment++) {
        if (peerSegment == mySegmentIdx) {
            continue;
        }
        uint32_t peerRank = PhysicalRank(rootServerBase, rootLocalRank, peerSegment);
        localChannels[workerSlot] = FindChannel(resCtx, peerRank);
        peerSegments[workerSlot] = peerSegment;
        CHK_PRT_RET(localChannels[workerSlot] == nullptr,
            HCCL_ERROR("Source-owner local channel to rank[%u] not found", peerRank), HCCL_E_INTERNAL);
        if (peerSegment == 0) {
            rootWorkerSlot = workerSlot;
            CHK_RET(ValidateChannel(*localChannels[workerSlot], BROADCAST_SIGNALS_PER_DIRECTION));
            for (uint32_t pieceIdx = 0; pieceIdx < pipelinePieceCount; pieceIdx++) {
                Segment rootPiece = SplitPiece(SplitSegment(param.count, 0), pieceIdx, pipelinePieceCount);
                CHK_RET(ValidateLocalWindow(resCtx.localBuffer,
                    static_cast<uint64_t>(IncomingSlot(0, mySegmentIdx)) * slotStrideBytes,
                    rootPiece.count * dataTypeSize));
            }
        } else {
            for (uint32_t pieceIdx = 0; pieceIdx < pipelinePieceCount; pieceIdx++) {
                Segment outgoing = SplitPiece(SplitSegment(param.count, mySegmentIdx),
                    pieceIdx, pipelinePieceCount);
                Segment incoming = SplitPiece(SplitSegment(param.count, peerSegment),
                    pieceIdx, pipelinePieceCount);
                CHK_RET(ValidateExchange(resCtx, *localChannels[workerSlot], mySegmentIdx,
                    peerSegment, outgoing, incoming, slotStrideBytes, dataTypeSize));
            }
        }
        workerSlot++;
    }
    CHK_PRT_RET(rootWorkerSlot == INVALID_VALUE_RANKID,
        HCCL_ERROR("Source-owner root worker was not mapped"), HCCL_E_INTERNAL);

    uint32_t rootRank = PhysicalRank(rootServerBase, rootLocalRank, 0);
    const ChannelInfo *rootChannel = FindChannel(resCtx, rootRank);
    CHK_PRT_RET(rootChannel == nullptr,
        HCCL_ERROR("Scatter channel to root rank[%u] not found", rootRank), HCCL_E_INTERNAL);
    CHK_RET(ValidateChannel(*rootChannel, 0));
    Segment mySegment = SplitSegment(param.count, mySegmentIdx);
    for (uint32_t pieceIdx = 0; pieceIdx < pipelinePieceCount; pieceIdx++) {
        Segment piece = SplitPiece(mySegment, pieceIdx, pipelinePieceCount);
        CHK_RET(ValidateLocalWindow(resCtx.localBuffer, INGRESS_SLOT * slotStrideBytes,
            piece.count * dataTypeSize));
    }

    uint32_t mirrorRank = param.myRank ^ BROADCAST_SERVER_RANKS;
    const ChannelInfo *crossChannel = FindChannel(resCtx, mirrorRank);
    CHK_PRT_RET(crossChannel == nullptr,
        HCCL_ERROR("Source-owner cross channel to rank[%u] not found", mirrorRank), HCCL_E_INTERNAL);
    CHK_RET(ValidateChannel(*crossChannel, 0));
    for (uint32_t pieceIdx = 0; pieceIdx < pipelinePieceCount; pieceIdx++) {
        Segment piece = SplitPiece(mySegment, pieceIdx, pipelinePieceCount);
        CHK_RET(ValidateRemoteWindow(*crossChannel, INGRESS_SLOT * slotStrideBytes,
            piece.count * dataTypeSize));
    }

    ThreadHandle coordinator = resCtx.threads[MAIN_THREAD_INDEX];
    for (uint32_t idx = 0; idx < LOCAL_PEER_COUNT; idx++) {
        ThreadHandle worker = resCtx.threads[FIRST_LOCAL_WORKER_INDEX + idx];
        CHK_RET(StartWorker(coordinator, worker));
        CHK_RET(WaitWorkerStart(worker));
        uint32_t peerSegment = peerSegments[idx];
        if (peerSegment == 0) {
            for (uint32_t pieceIdx = 0; pieceIdx < pipelinePieceCount; pieceIdx++) {
                Segment rootPiece = SplitPiece(SplitSegment(param.count, 0), pieceIdx, pipelinePieceCount);
                CHK_RET(ReceivePiece(worker, *localChannels[idx],
                    AddOffset(param.inputPtr, rootPiece.countOffset * dataTypeSize), resCtx.localBuffer,
                    static_cast<uint64_t>(IncomingSlot(0, mySegmentIdx)) * slotStrideBytes,
                    rootPiece.count * dataTypeSize, BROADCAST_SIGNALS_PER_DIRECTION));
            }
        } else {
            for (uint32_t pieceIdx = 0; pieceIdx < pipelinePieceCount; pieceIdx++) {
                uint32_t readyNotify = pieceIdx == 0 ? BROADCAST_PIECE0_READY_NOTIFY :
                    BROADCAST_PIECE1_READY_NOTIFY;
                CHK_RET(WaitPieceReady(worker, readyNotify));
                Segment outgoing = SplitPiece(mySegment, pieceIdx, pipelinePieceCount);
                Segment incoming = SplitPiece(SplitSegment(param.count, peerSegment),
                    pieceIdx, pipelinePieceCount);
                CHK_RET(RunExchange(worker, param, resCtx, *localChannels[idx], mySegmentIdx,
                    peerSegment, outgoing, incoming, slotStrideBytes, dataTypeSize));
            }
        }
        CHK_RET(SignalWorkerComplete(worker, coordinator, idx + 1));
    }

    ThreadHandle crossWorker = resCtx.threads[CROSS_WORKER_INDEX];
    CHK_RET(StartWorker(coordinator, crossWorker));
    CHK_RET(WaitWorkerStart(crossWorker));
    for (uint32_t pieceIdx = 0; pieceIdx < pipelinePieceCount; pieceIdx++) {
        uint32_t readyNotify = pieceIdx == 0 ? BROADCAST_PIECE0_READY_NOTIFY :
            BROADCAST_PIECE1_READY_NOTIFY;
        CHK_RET(WaitPieceReady(crossWorker, readyNotify));
        Segment piece = SplitPiece(mySegment, pieceIdx, pipelinePieceCount);
        CHK_RET(SendPiece(crossWorker, *crossChannel,
            AddOffset(param.inputPtr, piece.countOffset * dataTypeSize),
            INGRESS_SLOT * slotStrideBytes, piece.count * dataTypeSize, 0));
    }
    CHK_RET(SignalWorkerComplete(crossWorker, coordinator, CROSS_WORKER_INDEX));

    for (uint32_t pieceIdx = 0; pieceIdx < pipelinePieceCount; pieceIdx++) {
        Segment piece = SplitPiece(mySegment, pieceIdx, pipelinePieceCount);
        CHK_RET(ReceivePiece(coordinator, *rootChannel,
            AddOffset(param.inputPtr, piece.countOffset * dataTypeSize), resCtx.localBuffer,
            INGRESS_SLOT * slotStrideBytes, piece.count * dataTypeSize, 0));
        uint32_t readyNotify = pieceIdx == 0 ? BROADCAST_PIECE0_READY_NOTIFY :
            BROADCAST_PIECE1_READY_NOTIFY;
        for (uint32_t idx = 0; idx < LOCAL_PEER_COUNT; idx++) {
            if (idx != rootWorkerSlot) {
                CHK_RET(SignalPieceReady(coordinator,
                    resCtx.threads[FIRST_LOCAL_WORKER_INDEX + idx], readyNotify));
            }
        }
        CHK_RET(SignalPieceReady(coordinator, crossWorker, readyNotify));
    }

    for (uint32_t idx = 1; idx <= LOCAL_PEER_COUNT; idx++) {
        CHK_RET(JoinWorker(coordinator, idx));
    }
    return JoinWorker(coordinator, CROSS_WORKER_INDEX);
}

HcclResult RemoteOwnerStripedPipeline(const OpParam &param, const AlgResourceCtx &resCtx,
    uint32_t pipelinePieceCount, uint64_t slotStrideBytes, uint32_t dataTypeSize,
    uint32_t remoteServerBase, uint32_t rootLocalRank, uint32_t mySegmentIdx)
{
    const ChannelInfo *localChannels[LOCAL_PEER_COUNT] = {};
    uint32_t peerSegments[LOCAL_PEER_COUNT] = {};
    uint32_t workerSlot = 0;
    for (uint32_t peerSegment = 0; peerSegment < BROADCAST_SERVER_RANKS; peerSegment++) {
        if (peerSegment == mySegmentIdx) {
            continue;
        }
        uint32_t peerRank = PhysicalRank(remoteServerBase, rootLocalRank, peerSegment);
        localChannels[workerSlot] = FindChannel(resCtx, peerRank);
        peerSegments[workerSlot] = peerSegment;
        CHK_PRT_RET(localChannels[workerSlot] == nullptr,
            HCCL_ERROR("Remote-owner local channel to rank[%u] not found", peerRank), HCCL_E_INTERNAL);
        for (uint32_t pieceIdx = 0; pieceIdx < pipelinePieceCount; pieceIdx++) {
            Segment outgoing = SplitPiece(SplitSegment(param.count, mySegmentIdx),
                pieceIdx, pipelinePieceCount);
            Segment incoming = SplitPiece(SplitSegment(param.count, peerSegment),
                pieceIdx, pipelinePieceCount);
            CHK_RET(ValidateExchange(resCtx, *localChannels[workerSlot], mySegmentIdx,
                peerSegment, outgoing, incoming, slotStrideBytes, dataTypeSize));
        }
        workerSlot++;
    }

    uint32_t mirrorRank = param.myRank ^ BROADCAST_SERVER_RANKS;
    const ChannelInfo *crossChannel = FindChannel(resCtx, mirrorRank);
    CHK_PRT_RET(crossChannel == nullptr,
        HCCL_ERROR("Remote-owner cross channel to rank[%u] not found", mirrorRank), HCCL_E_INTERNAL);
    CHK_RET(ValidateChannel(*crossChannel, 0));
    Segment mySegment = SplitSegment(param.count, mySegmentIdx);
    for (uint32_t pieceIdx = 0; pieceIdx < pipelinePieceCount; pieceIdx++) {
        Segment piece = SplitPiece(mySegment, pieceIdx, pipelinePieceCount);
        CHK_RET(ValidateLocalWindow(resCtx.localBuffer, INGRESS_SLOT * slotStrideBytes,
            piece.count * dataTypeSize));
    }

    ThreadHandle coordinator = resCtx.threads[MAIN_THREAD_INDEX];
    for (uint32_t idx = 0; idx < LOCAL_PEER_COUNT; idx++) {
        ThreadHandle worker = resCtx.threads[FIRST_LOCAL_WORKER_INDEX + idx];
        CHK_RET(StartWorker(coordinator, worker));
        CHK_RET(WaitWorkerStart(worker));
        for (uint32_t pieceIdx = 0; pieceIdx < pipelinePieceCount; pieceIdx++) {
            uint32_t readyNotify = pieceIdx == 0 ? BROADCAST_PIECE0_READY_NOTIFY :
                BROADCAST_PIECE1_READY_NOTIFY;
            CHK_RET(WaitPieceReady(worker, readyNotify));
            Segment outgoing = SplitPiece(mySegment, pieceIdx, pipelinePieceCount);
            Segment incoming = SplitPiece(SplitSegment(param.count, peerSegments[idx]),
                pieceIdx, pipelinePieceCount);
            CHK_RET(RunExchange(worker, param, resCtx, *localChannels[idx], mySegmentIdx,
                peerSegments[idx], outgoing, incoming, slotStrideBytes, dataTypeSize));
        }
        CHK_RET(SignalWorkerComplete(worker, coordinator, idx + 1));
    }

    for (uint32_t pieceIdx = 0; pieceIdx < pipelinePieceCount; pieceIdx++) {
        Segment piece = SplitPiece(mySegment, pieceIdx, pipelinePieceCount);
        CHK_RET(ReceivePiece(coordinator, *crossChannel,
            AddOffset(param.inputPtr, piece.countOffset * dataTypeSize), resCtx.localBuffer,
            INGRESS_SLOT * slotStrideBytes, piece.count * dataTypeSize, 0));
        uint32_t readyNotify = pieceIdx == 0 ? BROADCAST_PIECE0_READY_NOTIFY :
            BROADCAST_PIECE1_READY_NOTIFY;
        for (uint32_t idx = 0; idx < LOCAL_PEER_COUNT; idx++) {
            CHK_RET(SignalPieceReady(coordinator,
                resCtx.threads[FIRST_LOCAL_WORKER_INDEX + idx], readyNotify));
        }
    }
    return JoinWorkers(coordinator, LOCAL_PEER_COUNT);
}

HcclResult StripedPipelineBroadcast(const OpParam &param, const AlgResourceCtx &resCtx,
    uint32_t pipelinePieceCount, uint64_t slotStrideBytes, uint32_t dataTypeSize)
{
    CHK_PRT_RET(resCtx.threads.size() != BROADCAST_THREAD_COUNT,
        HCCL_ERROR("Unexpected thread count[%zu]", resCtx.threads.size()), HCCL_E_INTERNAL);
    CHK_PRT_RET(pipelinePieceCount == 0 || pipelinePieceCount > MAX_PIPELINE_PIECES,
        HCCL_ERROR("Invalid pipeline piece count[%u]", pipelinePieceCount), HCCL_E_INTERNAL);

    uint32_t rootServerBase = (param.root / BROADCAST_SERVER_RANKS) * BROADCAST_SERVER_RANKS;
    uint32_t remoteServerBase = rootServerBase ^ BROADCAST_SERVER_RANKS;
    uint32_t myServerBase = (param.myRank / BROADCAST_SERVER_RANKS) * BROADCAST_SERVER_RANKS;
    uint32_t rootLocalRank = param.root % BROADCAST_SERVER_RANKS;
    uint32_t mySegmentIdx = SegmentIndex(param.myRank, rootLocalRank);

    if (param.myRank == param.root) {
        return RootStripedPipeline(param, resCtx, pipelinePieceCount, slotStrideBytes,
            dataTypeSize, rootServerBase, remoteServerBase, rootLocalRank);
    }
    if (myServerBase == rootServerBase) {
        return SourceOwnerStripedPipeline(param, resCtx, pipelinePieceCount, slotStrideBytes,
            dataTypeSize, rootServerBase, rootLocalRank, mySegmentIdx);
    }
    return RemoteOwnerStripedPipeline(param, resCtx, pipelinePieceCount, slotStrideBytes,
        dataTypeSize, remoteServerBase, rootLocalRank, mySegmentIdx);
}

HcclResult ScatterRootSegmentsFallback(const OpParam &param, const AlgResourceCtx &resCtx,
    uint32_t serverBase, uint32_t rootLocalRank, uint64_t pieceCount, uint32_t dataTypeSize)
{
    const ChannelInfo *channels[LOCAL_PEER_COUNT] = {};
    Segment segments[LOCAL_PEER_COUNT] = {};
    for (uint32_t idx = 0; idx < LOCAL_PEER_COUNT; idx++) {
        uint32_t segmentIdx = idx + 1;
        uint32_t remoteRank = PhysicalRank(serverBase, rootLocalRank, segmentIdx);
        channels[idx] = FindChannel(resCtx, remoteRank);
        CHK_PRT_RET(channels[idx] == nullptr,
            HCCL_ERROR("Scatter channel to rank[%u] not found", remoteRank), HCCL_E_INTERNAL);
        segments[idx] = SplitSegment(param.count, segmentIdx);
    }

    ThreadHandle coordinator = resCtx.threads[MAIN_THREAD_INDEX];
    for (uint32_t idx = 0; idx < LOCAL_PEER_COUNT; idx++) {
        ThreadHandle worker = resCtx.threads[FIRST_LOCAL_WORKER_INDEX + idx];
        CHK_RET(StartWorker(coordinator, worker));
        CHK_RET(WaitWorkerStart(worker));
        CHK_RET(SendRange(worker, *channels[idx], param.inputPtr, segments[idx],
            pieceCount, 0, dataTypeSize));
        CHK_RET(SignalWorkerComplete(worker, coordinator, idx + 1));
    }
    return JoinWorkers(coordinator, LOCAL_PEER_COUNT);
}

HcclResult StartLocalExchangeFallback(const OpParam &param, const AlgResourceCtx &resCtx,
    uint32_t serverBase, uint32_t rootServerBase, uint32_t rootLocalRank, uint32_t mySegmentIdx,
    uint64_t pieceCount, uint64_t slotStrideBytes, uint32_t dataTypeSize)
{
    const ChannelInfo *channels[LOCAL_PEER_COUNT] = {};
    uint32_t peerSegments[LOCAL_PEER_COUNT] = {};
    uint32_t workerSlot = 0;
    for (uint32_t peerSegment = 0; peerSegment < BROADCAST_SERVER_RANKS; peerSegment++) {
        if (peerSegment == mySegmentIdx) {
            continue;
        }
        uint32_t peerRank = PhysicalRank(serverBase, rootLocalRank, peerSegment);
        channels[workerSlot] = FindChannel(resCtx, peerRank);
        CHK_PRT_RET(channels[workerSlot] == nullptr,
            HCCL_ERROR("Local exchange channel to rank[%u] not found", peerRank), HCCL_E_INTERNAL);
        peerSegments[workerSlot] = peerSegment;
        workerSlot++;
    }

    bool rootServer = serverBase == rootServerBase;
    Segment mySegment = SplitSegment(param.count, mySegmentIdx);
    ThreadHandle coordinator = resCtx.threads[MAIN_THREAD_INDEX];
    for (uint32_t idx = 0; idx < LOCAL_PEER_COUNT; idx++) {
        uint32_t peerSegment = peerSegments[idx];
        Segment peerRange = SplitSegment(param.count, peerSegment);
        uint64_t sendSlot = static_cast<uint64_t>(IncomingSlot(mySegmentIdx, peerSegment)) * slotStrideBytes;
        uint64_t receiveSlot = static_cast<uint64_t>(IncomingSlot(peerSegment, mySegmentIdx)) * slotStrideBytes;
        ThreadHandle worker = resCtx.threads[FIRST_LOCAL_WORKER_INDEX + idx];

        CHK_RET(StartWorker(coordinator, worker));
        CHK_RET(WaitWorkerStart(worker));
        if (rootServer && mySegmentIdx == 0) {
            CHK_RET(SendRange(worker, *channels[idx], param.inputPtr, mySegment,
                pieceCount, sendSlot, dataTypeSize));
        } else if (rootServer && peerSegment == 0) {
            CHK_RET(ReceiveRange(worker, *channels[idx], param.inputPtr, resCtx.localBuffer,
                peerRange, pieceCount, receiveSlot, dataTypeSize));
        } else if (mySegmentIdx < peerSegment) {
            CHK_RET(SendRange(worker, *channels[idx], param.inputPtr, mySegment,
                pieceCount, sendSlot, dataTypeSize));
            CHK_RET(ReceiveRange(worker, *channels[idx], param.inputPtr, resCtx.localBuffer,
                peerRange, pieceCount, receiveSlot, dataTypeSize));
        } else {
            CHK_RET(ReceiveRange(worker, *channels[idx], param.inputPtr, resCtx.localBuffer,
                peerRange, pieceCount, receiveSlot, dataTypeSize));
            CHK_RET(SendRange(worker, *channels[idx], param.inputPtr, mySegment,
                pieceCount, sendSlot, dataTypeSize));
        }
        CHK_RET(SignalWorkerComplete(worker, coordinator, idx + 1));
    }
    return HCCL_SUCCESS;
}

HcclResult SafeSegmentStripedBroadcast(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t pieceCount, uint64_t slotStrideBytes, uint32_t dataTypeSize)
{
    uint32_t rootServerBase = (param.root / BROADCAST_SERVER_RANKS) * BROADCAST_SERVER_RANKS;
    uint32_t myServerBase = (param.myRank / BROADCAST_SERVER_RANKS) * BROADCAST_SERVER_RANKS;
    uint32_t rootLocalRank = param.root % BROADCAST_SERVER_RANKS;
    uint32_t mySegmentIdx = SegmentIndex(param.myRank, rootLocalRank);
    Segment mySegment = SplitSegment(param.count, mySegmentIdx);
    ThreadHandle coordinator = resCtx.threads[MAIN_THREAD_INDEX];
    uint32_t mirrorRank = param.myRank ^ BROADCAST_SERVER_RANKS;
    const ChannelInfo *mirrorChannel = FindChannel(resCtx, mirrorRank);
    CHK_PRT_RET(mirrorChannel == nullptr,
        HCCL_ERROR("Channel to mirror rank[%u] not found", mirrorRank), HCCL_E_INTERNAL);

    if (myServerBase == rootServerBase) {
        if (param.myRank == param.root) {
            CHK_RET(ScatterRootSegmentsFallback(
                param, resCtx, rootServerBase, rootLocalRank, pieceCount, dataTypeSize));
        } else {
            const ChannelInfo *rootChannel = FindChannel(resCtx, param.root);
            CHK_PRT_RET(rootChannel == nullptr,
                HCCL_ERROR("Scatter channel to root rank[%u] not found", param.root), HCCL_E_INTERNAL);
            CHK_RET(ReceiveRange(coordinator, *rootChannel, param.inputPtr, resCtx.localBuffer,
                mySegment, pieceCount, 0, dataTypeSize));
        }

        CHK_RET(StartLocalExchangeFallback(param, resCtx, myServerBase, rootServerBase,
            rootLocalRank, mySegmentIdx, pieceCount, slotStrideBytes, dataTypeSize));
        CHK_RET(SendRange(coordinator, *mirrorChannel, param.inputPtr, mySegment,
            pieceCount, 0, dataTypeSize));
        return JoinWorkers(coordinator, LOCAL_PEER_COUNT);
    }

    CHK_RET(ReceiveRange(coordinator, *mirrorChannel, param.inputPtr, resCtx.localBuffer,
        mySegment, pieceCount, 0, dataTypeSize));
    CHK_RET(StartLocalExchangeFallback(param, resCtx, myServerBase, rootServerBase,
        rootLocalRank, mySegmentIdx, pieceCount, slotStrideBytes, dataTypeSize));
    return JoinWorkers(coordinator, LOCAL_PEER_COUNT);
}
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    if (param.count == 0 || param.rankSize <= 1) {
        return HCCL_SUCCESS;
    }

    uint32_t dataTypeSize = DataTypeSize(param.dataType);
    CHK_PRT_RET(dataTypeSize == 0, HCCL_ERROR("Unsupported dataType[%d]", param.dataType), HCCL_E_PARA);
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / dataTypeSize,
        HCCL_ERROR("Broadcast byte count overflow, count[%llu], dataTypeSize[%u]",
            static_cast<unsigned long long>(param.count), dataTypeSize),
        HCCL_E_PARA);

    uint64_t totalBytes = param.count * dataTypeSize;
    uint64_t regularPieceCount = std::min<uint64_t>(MAX_TRANSFER_PIECE_BYTES / dataTypeSize,
        resCtx.localBuffer.size / dataTypeSize);
    CHK_PRT_RET(regularPieceCount == 0,
        HCCL_ERROR("CCL buffer too small, capacity[%llu], dataTypeSize[%u]",
            static_cast<unsigned long long>(resCtx.localBuffer.size), dataTypeSize),
        HCCL_E_INTERNAL);

    if (param.rankSize == BROADCAST_RANK_SIZE && totalBytes >= STRIPED_MIN_BYTES) {
        if (DirectReadAllGatherFits(param, resCtx, dataTypeSize)) {
            return DirectReadAllGatherBroadcast(param, resCtx, dataTypeSize);
        }

        uint64_t slotStrideBytes = CommonSlotStride(resCtx, BROADCAST_SERVER_RANKS, dataTypeSize);
        CHK_PRT_RET(slotStrideBytes == 0,
            HCCL_ERROR("CCL buffer cannot hold eight striped slots"), HCCL_E_INTERNAL);
        uint32_t pipelinePieceCount = totalBytes >= DOUBLE_PIECE_MIN_BYTES ? 2 : 1;
        if (PipelinePiecesFit(param, pipelinePieceCount, slotStrideBytes, dataTypeSize)) {
            return StripedPipelineBroadcast(param, resCtx, pipelinePieceCount,
                slotStrideBytes, dataTypeSize);
        }

        uint64_t fallbackStrideBytes = CommonSlotStride(resCtx, LOCAL_PEER_COUNT, dataTypeSize);
        uint64_t fallbackPieceCount = std::min<uint64_t>(MAX_TRANSFER_PIECE_BYTES / dataTypeSize,
            fallbackStrideBytes / dataTypeSize);
        CHK_PRT_RET(fallbackPieceCount == 0,
            HCCL_ERROR("CCL buffer cannot hold fallback striped slots"), HCCL_E_INTERNAL);
        return SafeSegmentStripedBroadcast(param, resCtx, fallbackPieceCount,
            fallbackStrideBytes, dataTypeSize);
    }

    if (param.rankSize == BROADCAST_RANK_SIZE && totalBytes > BALANCED_TREE_MAX_BYTES) {
        return GroupedFlatBroadcast(param, resCtx, regularPieceCount, dataTypeSize);
    }
    if (param.rankSize == BROADCAST_RANK_SIZE && totalBytes >= BALANCED_TREE_MIN_BYTES) {
        // Change this selector to BinomialBroadcast for a one-line 512 KiB rollback.
        return BalancedThreeFanoutBroadcast(param, resCtx, totalBytes);
    }
    return BinomialBroadcast(param, resCtx, regularPieceCount, dataTypeSize);
}
} // namespace ops_hccl