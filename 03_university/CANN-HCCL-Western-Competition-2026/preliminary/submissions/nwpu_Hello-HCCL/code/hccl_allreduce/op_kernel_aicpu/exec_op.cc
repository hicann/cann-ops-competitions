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

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {
constexpr uint32_t EXPECTED_RANK_SIZE = 16;
constexpr uint32_t EXPECTED_LOCAL_RANK_SIZE = 8;
constexpr uint32_t EXPECTED_SERVER_SIZE = 2;
constexpr uint32_t EXPECTED_THREAD_NUM = 16;
constexpr uint32_t LOCAL_WORKER_NUM = EXPECTED_LOCAL_RANK_SIZE - 1;
constexpr uint32_t GLOBAL_WORKER_NUM = EXPECTED_RANK_SIZE - 1;
constexpr uint32_t BUTTERFLY_ROUND_NUM = 4;
constexpr uint32_t LOCAL_RABENSEIFNER_ROUND_NUM = 3;
constexpr uint32_t TWO_TWO_TWO_SIDE_ROUND_NUM = 2;
constexpr uint32_t REQUIRED_CHANNEL_NOTIFY_NUM = 2;
constexpr uint64_t BUFFER_ALIGNMENT = 16ULL * 1024ULL;
constexpr uint64_t MAX_PRIMITIVE_BYTES = 256ULL * 1024ULL * 1024ULL;
constexpr uint64_t LARGE_MESSAGE_THRESHOLD_BYTES = 32ULL * 1024ULL * 1024ULL;
constexpr uint64_t PIPELINE_TILE_BYTES = 256ULL * 1024ULL * 1024ULL;
static_assert(GROUPED_RS_REDUCE_GROUP_NUM == 3 || GROUPED_RS_REDUCE_GROUP_NUM == 4,
    "Grouped Reduce-Scatter supports exactly 3 or 4 groups");

using ChannelMap = std::array<const ChannelInfo *, EXPECTED_RANK_SIZE>;
using PeerOrder = std::array<uint32_t, GLOBAL_WORKER_NUM>;

struct BufferLayout {
    uint64_t meshSlotStride = 0;
    uint64_t meshBytes = 0;
    uint64_t pairOffset = 0;
    uint64_t pairCapacity = 0;
};

struct GlobalMeshLayout {
    uint64_t slotStride = 0;
};

HcclResult CheckTask(int32_t ret, const char *phase, uint32_t rank, uint64_t stage,
    uint32_t peerRank, uint32_t threadIndex, uint64_t amount)
{
    if (ret == HCCL_SUCCESS) {
        return HCCL_SUCCESS;
    }
    HCCL_ERROR("Primitive failed, phase[%s], rank[%u], stage[%llu], peer[%u], thread[%u], "
               "amount[%llu], ret[%d]",
        phase, rank, static_cast<unsigned long long>(stage), peerRank, threadIndex,
        static_cast<unsigned long long>(amount), ret);
    return static_cast<HcclResult>(ret);
}

bool IsSupportedProtocol(CommProtocol protocol)
{
    return protocol == COMM_PROTOCOL_UBC_CTP || protocol == COMM_PROTOCOL_UBC_TP
        || protocol == COMM_PROTOCOL_UBOE;
}

void *AddOffset(void *base, uint64_t offset)
{
    return static_cast<void *>(static_cast<uint8_t *>(base) + offset);
}

const void *AddOffset(const void *base, uint64_t offset)
{
    return static_cast<const void *>(static_cast<const uint8_t *>(base) + offset);
}

uint64_t AlignDown(uint64_t value, uint64_t alignment)
{
    return value - value % alignment;
}

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    if (value == 0) {
        return alignment;
    }
    const uint64_t remainder = value % alignment;
    return remainder == 0 ? value : value + alignment - remainder;
}

uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return value / divisor + (value % divisor != 0 ? 1 : 0);
}

void SplitRange(uint64_t totalCount, uint32_t partIndex, uint32_t partNum,
    uint64_t &partCount, uint64_t &partOffset)
{
    const uint64_t baseCount = totalCount / partNum;
    const uint64_t remainder = totalCount % partNum;
    partCount = baseCount + (partIndex < remainder ? 1 : 0);
    partOffset = baseCount * partIndex + std::min<uint64_t>(partIndex, remainder);
}

void GetPartInterval(uint64_t totalCount, uint32_t firstPart, uint32_t partNum,
    uint64_t &intervalCount, uint64_t &intervalOffset)
{
    const uint64_t baseCount = totalCount / EXPECTED_LOCAL_RANK_SIZE;
    const uint64_t remainder = totalCount % EXPECTED_LOCAL_RANK_SIZE;
    intervalOffset = baseCount * firstPart + std::min<uint64_t>(firstPart, remainder);
    const uint32_t endPart = firstPart + partNum;
    const uint64_t endOffset = baseCount * endPart + std::min<uint64_t>(endPart, remainder);
    intervalCount = endOffset - intervalOffset;
}

uint32_t GetPeerLocalIndex(uint32_t localRankIndex, uint32_t workerIndex)
{
    uint32_t peerLocalIndex = workerIndex - 1;
    if (peerLocalIndex >= localRankIndex) {
        ++peerLocalIndex;
    }
    return peerLocalIndex;
}

bool IsLocalRank(const AlgResourceCtx &resCtx, uint32_t rank)
{
    return std::find(resCtx.localRanks.begin(), resCtx.localRanks.end(), rank)
        != resCtx.localRanks.end();
}

PeerOrder BuildSendPeerOrder(const OpParam &param, const AlgResourceCtx &resCtx)
{
    PeerOrder order{};
    uint32_t index = 0;
    for (uint32_t rank = 0; rank < EXPECTED_RANK_SIZE; ++rank) {
        if (rank != param.myRank && !IsLocalRank(resCtx, rank)) {
            order[index++] = rank;
        }
    }
    for (uint32_t rank : resCtx.localRanks) {
        if (rank != param.myRank) {
            order[index++] = rank;
        }
    }
    return order;
}

PeerOrder BuildReceivePeerOrder(const OpParam &param, const AlgResourceCtx &resCtx)
{
    PeerOrder order{};
    uint32_t index = 0;
    for (uint32_t rank : resCtx.localRanks) {
        if (rank != param.myRank) {
            order[index++] = rank;
        }
    }
    for (uint32_t rank = 0; rank < EXPECTED_RANK_SIZE; ++rank) {
        if (rank != param.myRank && !IsLocalRank(resCtx, rank)) {
            order[index++] = rank;
        }
    }
    return order;
}

uint32_t GetCompactSourceSlot(uint32_t ownerRank, uint32_t sourceRank)
{
    return sourceRank < ownerRank ? sourceRank : sourceRank - 1;
}

uint32_t GetButterflyPeerRank(const AlgResourceCtx &resCtx, uint32_t round)
{
    if (round < 3) {
        const uint32_t peerLocalIndex = resCtx.localRankIndex ^ (1U << round);
        return resCtx.localRanks[peerLocalIndex];
    }
    return resCtx.pairRank;
}

HcclResult ValidateResourceContext(
    const OpParam &param, const AlgResourceCtx &resCtx, ChannelMap &channelByRank, uint64_t &usableBufferSize)
{
    CHK_PTR_NULL(param.inputPtr);
    CHK_PTR_NULL(param.outputPtr);
    if (param.rankSize != EXPECTED_RANK_SIZE || param.myRank >= param.rankSize) {
        HCCL_ERROR("Invalid rank metadata, rank[%u], rankSize[%u]", param.myRank, param.rankSize);
        return HCCL_E_PARA;
    }
    if (param.dataType != HCCL_DATA_TYPE_FP32 || param.reduceType != HCCL_REDUCE_SUM) {
        HCCL_ERROR("Unsupported operation, rank[%u], dataType[%d], reduceOp[%d]", param.myRank,
            static_cast<int32_t>(param.dataType), static_cast<int32_t>(param.reduceType));
        return HCCL_E_NOT_SUPPORT;
    }
    if (param.count > std::numeric_limits<uint64_t>::max() / sizeof(float)) {
        HCCL_ERROR("FP32 byte size overflow, rank[%u], count[%llu]", param.myRank,
            static_cast<unsigned long long>(param.count));
        return HCCL_E_PARA;
    }
    if (resCtx.schemaVersion != ALG_RESOURCE_CTX_SCHEMA_VERSION) {
        HCCL_ERROR("Context schema mismatch, rank[%u], actual[%u], expected[%u]", param.myRank,
            resCtx.schemaVersion, ALG_RESOURCE_CTX_SCHEMA_VERSION);
        return HCCL_E_INTERNAL;
    }
    if (resCtx.threads.size() != EXPECTED_THREAD_NUM || resCtx.aicpuThread != resCtx.threads[0]) {
        HCCL_ERROR("Invalid thread context, rank[%u], threadNum[%zu], main[%llu]", param.myRank,
            resCtx.threads.size(), static_cast<unsigned long long>(resCtx.aicpuThread));
        return HCCL_E_INTERNAL;
    }
    for (uint32_t index = 0; index < EXPECTED_THREAD_NUM; ++index) {
        if (resCtx.threads[index] == 0) {
            HCCL_ERROR("Invalid thread handle, rank[%u], threadIndex[%u]", param.myRank, index);
            return HCCL_E_UNAVAIL;
        }
    }
    if (resCtx.localRanks.size() != EXPECTED_LOCAL_RANK_SIZE
        || resCtx.localRankIndex >= EXPECTED_LOCAL_RANK_SIZE
        || resCtx.localRanks[resCtx.localRankIndex] != param.myRank
        || resCtx.serverIndex >= EXPECTED_SERVER_SIZE) {
        HCCL_ERROR("Invalid topology context, rank[%u], localRankNum[%zu], localIndex[%u], serverIndex[%u]",
            param.myRank, resCtx.localRanks.size(), resCtx.localRankIndex, resCtx.serverIndex);
        return HCCL_E_INTERNAL;
    }
    if (resCtx.pairRank >= param.rankSize
        || std::find(resCtx.localRanks.begin(), resCtx.localRanks.end(), resCtx.pairRank)
            != resCtx.localRanks.end()) {
        HCCL_ERROR("Invalid pair rank, rank[%u], pairRank[%u]", param.myRank, resCtx.pairRank);
        return HCCL_E_INTERNAL;
    }
    if (resCtx.localBuffer.addr == nullptr || resCtx.localBuffer.size == 0 || resCtx.usableBufferSize == 0) {
        HCCL_ERROR("Invalid local buffer, rank[%u], addr[%p], size[%llu], usable[%llu]", param.myRank,
            resCtx.localBuffer.addr, static_cast<unsigned long long>(resCtx.localBuffer.size),
            static_cast<unsigned long long>(resCtx.usableBufferSize));
        return HCCL_E_UNAVAIL;
    }
    if (resCtx.channels.size() != param.rankSize - 1) {
        HCCL_ERROR("Invalid channel count, rank[%u], actual[%zu], expected[%u]", param.myRank,
            resCtx.channels.size(), param.rankSize - 1);
        return HCCL_E_INTERNAL;
    }

    channelByRank.fill(nullptr);
    usableBufferSize = std::min(resCtx.localBuffer.size, resCtx.usableBufferSize);
    for (const ChannelInfo &channel : resCtx.channels) {
        if (channel.remoteRank >= param.rankSize || channel.remoteRank == param.myRank
            || channelByRank[channel.remoteRank] != nullptr) {
            HCCL_ERROR("Invalid or duplicate channel rank, rank[%u], peer[%u]", param.myRank,
                channel.remoteRank);
            return HCCL_E_INTERNAL;
        }
        if (channel.handle == 0 || channel.notifyNum < REQUIRED_CHANNEL_NOTIFY_NUM
            || channel.remoteCclMem.addr == nullptr || channel.remoteCclMem.size == 0) {
            HCCL_ERROR("Invalid channel resource, rank[%u], peer[%u], handle[%llu], notifys[%u], "
                       "addr[%p], size[%llu]",
                param.myRank, channel.remoteRank, static_cast<unsigned long long>(channel.handle),
                channel.notifyNum, channel.remoteCclMem.addr,
                static_cast<unsigned long long>(channel.remoteCclMem.size));
            return HCCL_E_UNAVAIL;
        }
        if (!IsSupportedProtocol(channel.protocol)) {
            HCCL_ERROR("Unsupported channel protocol, rank[%u], peer[%u], protocol[%d]", param.myRank,
                channel.remoteRank, static_cast<int32_t>(channel.protocol));
            return HCCL_E_NOT_SUPPORT;
        }
        channelByRank[channel.remoteRank] = &channel;
        usableBufferSize = std::min(usableBufferSize, channel.remoteCclMem.size);
    }
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (rank != param.myRank && channelByRank[rank] == nullptr) {
            HCCL_ERROR("Missing channel, rank[%u], peer[%u]", param.myRank, rank);
            return HCCL_E_NOT_FOUND;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CalculateBufferLayout(uint32_t rank, uint64_t usableBufferSize,
    uint64_t maxMeshShardCount, BufferLayout &layout)
{
    if (maxMeshShardCount > std::numeric_limits<uint64_t>::max() / sizeof(float)) {
        return HCCL_E_PARA;
    }
    const uint64_t shardBytes = maxMeshShardCount * sizeof(float);
    layout.meshSlotStride = AlignUp(std::max<uint64_t>(shardBytes, sizeof(float)), BUFFER_ALIGNMENT);
    if (layout.meshSlotStride > usableBufferSize / EXPECTED_LOCAL_RANK_SIZE) {
        HCCL_ERROR("Mesh slot does not fit, rank[%u], usable[%llu], shard[%llu], stride[%llu]", rank,
            static_cast<unsigned long long>(usableBufferSize),
            static_cast<unsigned long long>(shardBytes),
            static_cast<unsigned long long>(layout.meshSlotStride));
        return HCCL_E_MEMORY;
    }
    layout.meshBytes = layout.meshSlotStride * EXPECTED_LOCAL_RANK_SIZE;
    layout.pairOffset = layout.meshBytes;
    layout.pairCapacity = AlignDown(
        std::min(usableBufferSize - layout.meshBytes, MAX_PRIMITIVE_BYTES), sizeof(float));
    if (layout.pairCapacity < sizeof(float)) {
        HCCL_ERROR("Pair inbox does not fit, rank[%u], usable[%llu], meshBytes[%llu]", rank,
            static_cast<unsigned long long>(usableBufferSize),
            static_cast<unsigned long long>(layout.meshBytes));
        return HCCL_E_MEMORY;
    }
    return HCCL_SUCCESS;
}

HcclResult CalculateGlobalMeshLayout(uint32_t rank, uint64_t usableBufferSize,
    uint64_t maxTileCount, GlobalMeshLayout &layout)
{
    const uint64_t maxShardCount = CeilDiv(maxTileCount, EXPECTED_RANK_SIZE);
    if (maxShardCount > std::numeric_limits<uint64_t>::max() / sizeof(float)) {
        return HCCL_E_PARA;
    }
    layout.slotStride = AlignUp(
        std::max<uint64_t>(maxShardCount * sizeof(float), sizeof(float)), BUFFER_ALIGNMENT);
    if (layout.slotStride > usableBufferSize / GLOBAL_WORKER_NUM) {
        HCCL_ERROR("Global Mesh slots do not fit, rank[%u], usable[%llu], maxTile[%llu], "
                   "slotStride[%llu]",
            rank, static_cast<unsigned long long>(usableBufferSize),
            static_cast<unsigned long long>(maxTileCount),
            static_cast<unsigned long long>(layout.slotStride));
        return HCCL_E_MEMORY;
    }
    return HCCL_SUCCESS;
}

HcclResult CopyInSegments(const OpParam &param, ThreadHandle thread, void *dst, const void *src,
    uint64_t bytes, const char *phase)
{
    if (dst == src || bytes == 0) {
        return HCCL_SUCCESS;
    }
    for (uint64_t offset = 0, segment = 0; offset < bytes; ++segment) {
        const uint64_t currentBytes = std::min(MAX_PRIMITIVE_BYTES, bytes - offset);
        CHK_RET(CheckTask(HcommLocalCopyOnThread(thread, AddOffset(dst, offset), AddOffset(src, offset),
                              currentBytes),
            phase, param.myRank, segment, INVALID_VALUE_RANKID, 0, currentBytes));
        offset += currentBytes;
    }
    return HCCL_SUCCESS;
}

HcclResult BeginWorkerPhase(
    const OpParam &param, const AlgResourceCtx &resCtx, const char *phase, uint64_t stage,
    uint32_t workerNum)
{
    const ThreadHandle mainThread = resCtx.threads[0];
    for (uint32_t workerIndex = 1; workerIndex <= workerNum; ++workerIndex) {
        CHK_RET(CheckTask(HcommThreadNotifyWaitOnThread(
                              resCtx.threads[workerIndex], 0, CUSTOM_TIMEOUT),
            phase, param.myRank, stage, INVALID_VALUE_RANKID, workerIndex, 0));
    }
    for (uint32_t workerIndex = 1; workerIndex <= workerNum; ++workerIndex) {
        CHK_RET(CheckTask(HcommThreadNotifyRecordOnThread(
                              mainThread, resCtx.threads[workerIndex], 0),
            phase, param.myRank, stage, INVALID_VALUE_RANKID, workerIndex, 0));
    }
    return HCCL_SUCCESS;
}

HcclResult RecordWorkerDone(const OpParam &param, const AlgResourceCtx &resCtx,
    const char *phase, uint64_t stage, uint32_t workerIndex)
{
    return CheckTask(HcommThreadNotifyRecordOnThread(
                         resCtx.threads[workerIndex], resCtx.threads[0], workerIndex),
        phase, param.myRank, stage, INVALID_VALUE_RANKID, workerIndex, 0);
}

HcclResult WaitWorkerDone(
    const OpParam &param, const AlgResourceCtx &resCtx, const char *phase, uint64_t stage,
    uint32_t workerNum)
{
    const ThreadHandle mainThread = resCtx.threads[0];
    for (uint32_t workerIndex = 1; workerIndex <= workerNum; ++workerIndex) {
        CHK_RET(CheckTask(HcommThreadNotifyWaitOnThread(
                              mainThread, workerIndex, CUSTOM_TIMEOUT),
            phase, param.myRank, stage, INVALID_VALUE_RANKID, workerIndex, 0));
    }
    return HCCL_SUCCESS;
}

HcclResult RunButterflyBarrier(const OpParam &param, const AlgResourceCtx &resCtx,
    const ChannelMap &channelByRank, uint64_t stage, const char *phase)
{
    const ThreadHandle mainThread = resCtx.threads[0];
    for (uint32_t round = 0; round < BUTTERFLY_ROUND_NUM; ++round) {
        const uint32_t peerRank = GetButterflyPeerRank(resCtx, round);
        const ChannelInfo *channel = channelByRank[peerRank];
        if (channel == nullptr) {
            return HCCL_E_NOT_FOUND;
        }
        CHK_RET(CheckTask(HcommChannelNotifyRecordOnThread(
                              mainThread, channel->handle, NOTIFY_IDX_ACK),
            phase, param.myRank, stage + round, peerRank, 0, round));
        CHK_RET(CheckTask(HcommChannelNotifyWaitOnThread(
                              mainThread, channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT),
            phase, param.myRank, stage + round, peerRank, 0, round));
    }
    return HCCL_SUCCESS;
}

HcclResult RunLocalButterflyBarrier(const OpParam &param, const AlgResourceCtx &resCtx,
    const ChannelMap &channelByRank, uint64_t stage, const char *phase)
{
    const ThreadHandle mainThread = resCtx.threads[0];
    for (uint32_t round = 0; round < LOCAL_RABENSEIFNER_ROUND_NUM; ++round) {
        const uint32_t peerLocalIndex = resCtx.localRankIndex ^ (1U << round);
        const uint32_t peerRank = resCtx.localRanks[peerLocalIndex];
        const ChannelInfo *channel = channelByRank[peerRank];
        if (channel == nullptr) {
            return HCCL_E_NOT_FOUND;
        }
        CHK_RET(CheckTask(HcommChannelNotifyRecordOnThread(
                              mainThread, channel->handle, NOTIFY_IDX_ACK),
            phase, param.myRank, stage + round, peerRank, 0, round));
        CHK_RET(CheckTask(HcommChannelNotifyWaitOnThread(
                              mainThread, channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT),
            phase, param.myRank, stage + round, peerRank, 0, round));
    }
    return HCCL_SUCCESS;
}

HcclResult RunLocalScopedDirectBarrier(const OpParam &param, const AlgResourceCtx &resCtx,
    const ChannelMap &channelByRank, uint64_t stage, const char *phase)
{
    const ThreadHandle mainThread = resCtx.threads[0];
    std::array<uint32_t, LOCAL_RABENSEIFNER_ROUND_NUM> peerRanks{};
    for (uint32_t round = 0; round < LOCAL_RABENSEIFNER_ROUND_NUM; ++round) {
        const uint32_t peerLocalIndex = resCtx.localRankIndex ^ (1U << round);
        peerRanks[round] = resCtx.localRanks[peerLocalIndex];
    }

    for (uint32_t round = 0; round < LOCAL_RABENSEIFNER_ROUND_NUM; ++round) {
        const ChannelInfo *channel = channelByRank[peerRanks[round]];
        if (channel == nullptr) {
            return HCCL_E_NOT_FOUND;
        }
        CHK_RET(CheckTask(HcommChannelNotifyRecordOnThread(
                              mainThread, channel->handle, NOTIFY_IDX_ACK),
            phase, param.myRank, stage + round, peerRanks[round], 0, round));
    }

    for (uint32_t round = 0; round < LOCAL_RABENSEIFNER_ROUND_NUM; ++round) {
        const ChannelInfo *channel = channelByRank[peerRanks[round]];
        CHK_RET(CheckTask(HcommChannelNotifyWaitOnThread(
                              mainThread, channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT),
            phase, param.myRank, stage + round, peerRanks[round], 0, round));
    }
    return HCCL_SUCCESS;
}

HcclResult ScheduleParallelMeshSends(const OpParam &param, const AlgResourceCtx &resCtx,
    const ChannelMap &channelByRank, uint64_t stage, const char *phase, void *bufferBase,
    uint64_t totalCount, uint64_t slotStride, bool isGather)
{
    if (totalCount == 0) {
        return HCCL_SUCCESS;
    }
    CHK_RET(BeginWorkerPhase(param, resCtx, phase, stage, LOCAL_WORKER_NUM));
    const uint32_t localIndex = resCtx.localRankIndex;
    for (uint32_t workerIndex = 1; workerIndex <= LOCAL_WORKER_NUM; ++workerIndex) {
        const uint32_t peerLocalIndex = GetPeerLocalIndex(localIndex, workerIndex);
        const uint32_t peerRank = resCtx.localRanks[peerLocalIndex];
        const ChannelInfo *channel = channelByRank[peerRank];
        if (channel == nullptr) {
            return HCCL_E_NOT_FOUND;
        }
        const uint32_t sendPartIndex = isGather ? localIndex : peerLocalIndex;
        uint64_t sendCount = 0;
        uint64_t sendOffset = 0;
        SplitRange(totalCount, sendPartIndex, EXPECTED_LOCAL_RANK_SIZE, sendCount, sendOffset);
        const uint64_t sendBytes = sendCount * sizeof(float);
        if (sendBytes > slotStride) {
            HCCL_ERROR("Mesh send exceeds slot, rank[%u], stage[%llu], bytes[%llu], stride[%llu]",
                param.myRank, static_cast<unsigned long long>(stage),
                static_cast<unsigned long long>(sendBytes),
                static_cast<unsigned long long>(slotStride));
            return HCCL_E_MEMORY;
        }
        const ThreadHandle worker = resCtx.threads[workerIndex];
        if (sendBytes > 0) {
            const void *sendSrc = AddOffset(bufferBase, sendOffset * sizeof(float));
            void *remoteDst = AddOffset(channel->remoteCclMem.addr,
                static_cast<uint64_t>(localIndex) * slotStride);
            CHK_RET(CheckTask(HcommWriteOnThread(
                                  worker, channel->handle, remoteDst, sendSrc, sendBytes),
                "mesh_write", param.myRank, stage, peerRank, workerIndex, sendBytes));
        }
        CHK_RET(CheckTask(HcommChannelNotifyRecordOnThread(
                              worker, channel->handle, NOTIFY_IDX_DATA_SIGNAL),
            "mesh_data_record", param.myRank, stage, peerRank, workerIndex, sendBytes));
        CHK_RET(RecordWorkerDone(param, resCtx, phase, stage, workerIndex));
    }
    return HCCL_SUCCESS;
}

HcclResult CompleteParallelMesh(const OpParam &param, const AlgResourceCtx &resCtx,
    const ChannelMap &channelByRank, uint64_t stage, const char *phase, void *bufferBase,
    uint64_t totalCount, uint64_t slotStride, bool isGather)
{
    if (totalCount == 0) {
        return HCCL_SUCCESS;
    }
    CHK_RET(WaitWorkerDone(param, resCtx, phase, stage, LOCAL_WORKER_NUM));
    const ThreadHandle mainThread = resCtx.threads[0];
    const uint32_t localIndex = resCtx.localRankIndex;

    for (uint32_t sourceLocalIndex = 0; sourceLocalIndex < EXPECTED_LOCAL_RANK_SIZE;
         ++sourceLocalIndex) {
        if (sourceLocalIndex == localIndex) {
            continue;
        }
        const uint32_t peerRank = resCtx.localRanks[sourceLocalIndex];
        const ChannelInfo *channel = channelByRank[peerRank];
        const uint32_t recvPartIndex = isGather ? sourceLocalIndex : localIndex;
        uint64_t recvCount = 0;
        uint64_t recvOffset = 0;
        SplitRange(totalCount, recvPartIndex, EXPECTED_LOCAL_RANK_SIZE, recvCount, recvOffset);
        CHK_RET(CheckTask(HcommChannelNotifyWaitOnThread(
                              mainThread, channel->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT),
            "mesh_data_wait", param.myRank, stage, peerRank, 0, recvCount));
        if (recvCount > 0) {
            void *dst = AddOffset(bufferBase, recvOffset * sizeof(float));
            const void *src = AddOffset(resCtx.localBuffer.addr,
                static_cast<uint64_t>(sourceLocalIndex) * slotStride);
            if (isGather) {
                CHK_RET(CheckTask(HcommLocalCopyOnThread(
                                      mainThread, dst, src, recvCount * sizeof(float)),
                    "mesh_copy", param.myRank, stage, peerRank, 0, recvCount));
            } else {
                CHK_RET(CheckTask(HcommLocalReduceOnThread(mainThread, dst, src, recvCount,
                                      HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM),
                    "mesh_reduce", param.myRank, stage, peerRank, 0, recvCount));
            }
        }
        CHK_RET(CheckTask(HcommChannelNotifyRecordOnThread(
                              mainThread, channel->handle, NOTIFY_IDX_ACK),
            "mesh_ack_record", param.myRank, stage, peerRank, 0, recvCount));
    }
    for (uint32_t peerLocalIndex = 0; peerLocalIndex < EXPECTED_LOCAL_RANK_SIZE;
         ++peerLocalIndex) {
        if (peerLocalIndex == localIndex) {
            continue;
        }
        const uint32_t peerRank = resCtx.localRanks[peerLocalIndex];
        const ChannelInfo *channel = channelByRank[peerRank];
        CHK_RET(CheckTask(HcommChannelNotifyWaitOnThread(
                              mainThread, channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT),
            "mesh_ack_wait", param.myRank, stage, peerRank, 0, 0));
    }
    return HCCL_SUCCESS;
}

HcclResult ScheduleGlobalMeshSends(const OpParam &param, const AlgResourceCtx &resCtx,
    const ChannelMap &channelByRank, uint64_t stage, const char *phase,
    const void *bufferBase, uint64_t totalCount, uint64_t slotStride, bool isGather,
    bool trackWorkerDone)
{
    if (totalCount == 0) {
        return HCCL_SUCCESS;
    }
    CHK_RET(BeginWorkerPhase(param, resCtx, phase, stage, GLOBAL_WORKER_NUM));
    const PeerOrder peerOrder = BuildSendPeerOrder(param, resCtx);
    for (uint32_t workerIndex = 1; workerIndex <= GLOBAL_WORKER_NUM; ++workerIndex) {
        const uint32_t peerRank = peerOrder[workerIndex - 1];
        const ChannelInfo *channel = channelByRank[peerRank];
        if (channel == nullptr) {
            return HCCL_E_NOT_FOUND;
        }
        const uint32_t sendPartIndex = isGather ? param.myRank : peerRank;
        uint64_t sendCount = 0;
        uint64_t sendOffset = 0;
        SplitRange(totalCount, sendPartIndex, EXPECTED_RANK_SIZE, sendCount, sendOffset);
        const uint64_t sendBytes = sendCount * sizeof(float);
        if (sendBytes > slotStride) {
            HCCL_ERROR("Global Mesh send exceeds slot, rank[%u], stage[%llu], peer[%u], "
                       "bytes[%llu], stride[%llu]",
                param.myRank, static_cast<unsigned long long>(stage), peerRank,
                static_cast<unsigned long long>(sendBytes),
                static_cast<unsigned long long>(slotStride));
            return HCCL_E_MEMORY;
        }
        const ThreadHandle worker = resCtx.threads[workerIndex];
        if (sendBytes > 0) {
            const uint32_t remoteSlot = GetCompactSourceSlot(peerRank, param.myRank);
            void *remoteDst = AddOffset(channel->remoteCclMem.addr,
                static_cast<uint64_t>(remoteSlot) * slotStride);
            CHK_RET(CheckTask(HcommWriteOnThread(worker, channel->handle, remoteDst,
                                      AddOffset(bufferBase, sendOffset * sizeof(float)), sendBytes),
                "global_mesh_write", param.myRank, stage, peerRank, workerIndex, sendBytes));
        }
        CHK_RET(CheckTask(HcommChannelNotifyRecordOnThread(
                              worker, channel->handle, NOTIFY_IDX_DATA_SIGNAL),
            "global_mesh_data_record", param.myRank, stage, peerRank, workerIndex, sendBytes));
        if (trackWorkerDone) {
            CHK_RET(RecordWorkerDone(param, resCtx, phase, stage, workerIndex));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CompleteSerialGlobalReduceScatter(const OpParam &param,
    const AlgResourceCtx &resCtx, const ChannelMap &channelByRank, uint64_t stage,
    void *outputBase, uint64_t totalCount, uint64_t slotStride)
{
    if (totalCount == 0) {
        return HCCL_SUCCESS;
    }
    const ThreadHandle mainThread = resCtx.threads[0];
    const PeerOrder receiveOrder = BuildReceivePeerOrder(param, resCtx);
    for (uint32_t sourceIndex = 0; sourceIndex < GLOBAL_WORKER_NUM; ++sourceIndex) {
        const uint32_t sourceRank = receiveOrder[sourceIndex];
        const ChannelInfo *channel = channelByRank[sourceRank];
        uint64_t recvCount = 0;
        uint64_t recvOffset = 0;
        SplitRange(totalCount, param.myRank, EXPECTED_RANK_SIZE, recvCount, recvOffset);
        CHK_RET(CheckTask(HcommChannelNotifyWaitOnThread(
                              mainThread, channel->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT),
            "global_mesh_data_wait", param.myRank, stage, sourceRank, 0, recvCount));
        if (recvCount > 0) {
            void *dst = AddOffset(outputBase, recvOffset * sizeof(float));
            const uint32_t localSlot = GetCompactSourceSlot(param.myRank, sourceRank);
            const void *src = AddOffset(resCtx.localBuffer.addr,
                static_cast<uint64_t>(localSlot) * slotStride);
            CHK_RET(CheckTask(HcommLocalReduceOnThread(mainThread, dst, src, recvCount,
                                  HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM),
                "global_mesh_reduce", param.myRank, stage, sourceRank, 0, recvCount));
        }
    }

    return RunButterflyBarrier(
        param, resCtx, channelByRank, stage, "global_mesh_rs_barrier");
}

HcclResult CompleteGroupedGlobalReduceScatter(const OpParam &param,
    const AlgResourceCtx &resCtx, const ChannelMap &channelByRank, uint64_t stage,
    void *outputBase, uint64_t totalCount, uint64_t slotStride)
{
    if (totalCount == 0) {
        return HCCL_SUCCESS;
    }
    constexpr const char *PHASE = "global_mesh_grouped_rs";
    uint64_t ownCount = 0;
    uint64_t ownOffset = 0;
    SplitRange(totalCount, param.myRank, EXPECTED_RANK_SIZE, ownCount, ownOffset);
    void *ownOutput = AddOffset(outputBase, ownOffset * sizeof(float));
    const PeerOrder receiveOrder = BuildReceivePeerOrder(param, resCtx);
    std::array<uint32_t, GROUPED_RS_REDUCE_GROUP_NUM> accumulatorSlots{};

    for (uint32_t group = 0; group < GROUPED_RS_REDUCE_GROUP_NUM; ++group) {
        const uint32_t leaderIndex = group + 1;
        const ThreadHandle leader = resCtx.threads[leaderIndex];
        bool hasAccumulator = false;
        for (uint32_t sourceIndex = group; sourceIndex < GLOBAL_WORKER_NUM;
             sourceIndex += GROUPED_RS_REDUCE_GROUP_NUM) {
            const uint32_t sourceRank = receiveOrder[sourceIndex];
            const ChannelInfo *channel = channelByRank[sourceRank];
            if (channel == nullptr) {
                return HCCL_E_NOT_FOUND;
            }
            const uint32_t sourceSlot = GetCompactSourceSlot(param.myRank, sourceRank);
            CHK_RET(CheckTask(HcommChannelNotifyWaitOnThread(leader, channel->handle,
                                  NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT),
                "grouped_rs_data_wait", param.myRank, stage, sourceRank, leaderIndex,
                ownCount));
            if (!hasAccumulator) {
                accumulatorSlots[group] = sourceSlot;
                hasAccumulator = true;
                continue;
            }
            if (ownCount > 0) {
                void *accumulator = AddOffset(resCtx.localBuffer.addr,
                    static_cast<uint64_t>(accumulatorSlots[group]) * slotStride);
                const void *source = AddOffset(resCtx.localBuffer.addr,
                    static_cast<uint64_t>(sourceSlot) * slotStride);
                CHK_RET(CheckTask(HcommLocalReduceOnThread(leader, accumulator, source,
                                      ownCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM),
                    "grouped_rs_partial_reduce", param.myRank, stage, sourceRank,
                    leaderIndex, ownCount));
            }
        }
        if (!hasAccumulator) {
            HCCL_ERROR("Empty grouped RS source group, rank[%u], group[%u]", param.myRank,
                group);
            return HCCL_E_INTERNAL;
        }
        CHK_RET(RecordWorkerDone(param, resCtx, PHASE, stage, leaderIndex));
    }

    CHK_RET(WaitWorkerDone(
        param, resCtx, PHASE, stage, GROUPED_RS_REDUCE_GROUP_NUM));
    for (uint32_t group = 0; group < GROUPED_RS_REDUCE_GROUP_NUM; ++group) {
        if (ownCount > 0) {
            const void *partial = AddOffset(resCtx.localBuffer.addr,
                static_cast<uint64_t>(accumulatorSlots[group]) * slotStride);
            CHK_RET(CheckTask(HcommLocalReduceOnThread(resCtx.threads[0], ownOutput,
                                  partial, ownCount, HCOMM_DATA_TYPE_FP32,
                                  HCOMM_REDUCE_SUM),
                "grouped_rs_final_reduce", param.myRank, stage, INVALID_VALUE_RANKID,
                0, ownCount));
        }
    }
    return RunButterflyBarrier(
        param, resCtx, channelByRank, stage, "global_mesh_grouped_rs_barrier");
}

HcclResult RunGlobalAllGatherWorkers(const OpParam &param, const AlgResourceCtx &resCtx,
    const ChannelMap &channelByRank, uint64_t stage, void *outputBase,
    uint64_t totalCount, uint64_t slotStride)
{
    if (totalCount == 0) {
        return HCCL_SUCCESS;
    }
    constexpr const char *PHASE = "global_mesh_ag_workers";
    uint64_t ownCount = 0;
    uint64_t ownOffset = 0;
    SplitRange(totalCount, param.myRank, EXPECTED_RANK_SIZE, ownCount, ownOffset);
    const uint64_t ownBytes = ownCount * sizeof(float);
    if (ownBytes > slotStride) {
        HCCL_ERROR("Global All-Gather shard exceeds slot, rank[%u], stage[%llu], "
                   "bytes[%llu], stride[%llu]",
            param.myRank, static_cast<unsigned long long>(stage),
            static_cast<unsigned long long>(ownBytes),
            static_cast<unsigned long long>(slotStride));
        return HCCL_E_MEMORY;
    }

    CHK_RET(BeginWorkerPhase(param, resCtx, PHASE, stage, GLOBAL_WORKER_NUM));
    const PeerOrder peerOrder = BuildSendPeerOrder(param, resCtx);
    for (uint32_t workerIndex = 1; workerIndex <= GLOBAL_WORKER_NUM; ++workerIndex) {
        const uint32_t peerRank = peerOrder[workerIndex - 1];
        const ChannelInfo *channel = channelByRank[peerRank];
        if (channel == nullptr) {
            return HCCL_E_NOT_FOUND;
        }
        uint64_t peerCount = 0;
        uint64_t peerOffset = 0;
        SplitRange(totalCount, peerRank, EXPECTED_RANK_SIZE, peerCount, peerOffset);
        const ThreadHandle worker = resCtx.threads[workerIndex];

        if (ownBytes > 0) {
            const uint32_t remoteSlot = GetCompactSourceSlot(peerRank, param.myRank);
            void *remoteDst = AddOffset(channel->remoteCclMem.addr,
                static_cast<uint64_t>(remoteSlot) * slotStride);
            const void *sendSrc = AddOffset(outputBase, ownOffset * sizeof(float));
            CHK_RET(CheckTask(HcommWriteOnThread(
                                  worker, channel->handle, remoteDst, sendSrc, ownBytes),
                "global_mesh_ag_write", param.myRank, stage, peerRank, workerIndex, ownBytes));
        }
        CHK_RET(CheckTask(HcommChannelNotifyRecordOnThread(
                              worker, channel->handle, NOTIFY_IDX_DATA_SIGNAL),
            "global_mesh_ag_data_record", param.myRank, stage, peerRank, workerIndex, ownBytes));
        CHK_RET(CheckTask(HcommChannelNotifyWaitOnThread(
                              worker, channel->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT),
            "global_mesh_ag_data_wait", param.myRank, stage, peerRank, workerIndex, peerCount));
        if (peerCount > 0) {
            const uint32_t localSlot = GetCompactSourceSlot(param.myRank, peerRank);
            const void *src = AddOffset(resCtx.localBuffer.addr,
                static_cast<uint64_t>(localSlot) * slotStride);
            void *dst = AddOffset(outputBase, peerOffset * sizeof(float));
            CHK_RET(CheckTask(HcommLocalCopyOnThread(
                                  worker, dst, src, peerCount * sizeof(float)),
                "global_mesh_ag_copy", param.myRank, stage, peerRank, workerIndex, peerCount));
        }
        CHK_RET(RecordWorkerDone(param, resCtx, PHASE, stage, workerIndex));
    }

    CHK_RET(WaitWorkerDone(param, resCtx, PHASE, stage, GLOBAL_WORKER_NUM));
    return RunButterflyBarrier(
        param, resCtx, channelByRank, stage, "global_mesh_ag_barrier");
}

HcclResult RunSerialMesh(const OpParam &param, const AlgResourceCtx &resCtx,
    const ChannelMap &channelByRank, uint64_t stage, void *bufferBase, uint64_t totalCount,
    uint64_t slotStride, bool isGather)
{
    if (totalCount == 0) {
        return HCCL_SUCCESS;
    }
    const ThreadHandle thread = resCtx.threads[0];
    const uint32_t localIndex = resCtx.localRankIndex;
    for (uint32_t peerLocalIndex = 0; peerLocalIndex < EXPECTED_LOCAL_RANK_SIZE;
         ++peerLocalIndex) {
        if (peerLocalIndex == localIndex) {
            continue;
        }
        const uint32_t peerRank = resCtx.localRanks[peerLocalIndex];
        const ChannelInfo *channel = channelByRank[peerRank];
        const uint32_t sendPartIndex = isGather ? localIndex : peerLocalIndex;
        uint64_t sendCount = 0;
        uint64_t sendOffset = 0;
        SplitRange(totalCount, sendPartIndex, EXPECTED_LOCAL_RANK_SIZE, sendCount, sendOffset);
        const uint64_t sendBytes = sendCount * sizeof(float);
        if (sendBytes > 0) {
            void *remoteDst = AddOffset(channel->remoteCclMem.addr,
                static_cast<uint64_t>(localIndex) * slotStride);
            CHK_RET(CheckTask(HcommWriteOnThread(thread, channel->handle, remoteDst,
                                  AddOffset(bufferBase, sendOffset * sizeof(float)), sendBytes),
                "small_mesh_write", param.myRank, stage, peerRank, 0, sendBytes));
        }
        CHK_RET(CheckTask(HcommChannelNotifyRecordOnThread(
                              thread, channel->handle, NOTIFY_IDX_DATA_SIGNAL),
            "small_mesh_data_record", param.myRank, stage, peerRank, 0, sendBytes));
    }

    for (uint32_t sourceLocalIndex = 0; sourceLocalIndex < EXPECTED_LOCAL_RANK_SIZE;
         ++sourceLocalIndex) {
        if (sourceLocalIndex == localIndex) {
            continue;
        }
        const uint32_t peerRank = resCtx.localRanks[sourceLocalIndex];
        const ChannelInfo *channel = channelByRank[peerRank];
        const uint32_t recvPartIndex = isGather ? sourceLocalIndex : localIndex;
        uint64_t recvCount = 0;
        uint64_t recvOffset = 0;
        SplitRange(totalCount, recvPartIndex, EXPECTED_LOCAL_RANK_SIZE, recvCount, recvOffset);
        CHK_RET(CheckTask(HcommChannelNotifyWaitOnThread(
                              thread, channel->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT),
            "small_mesh_data_wait", param.myRank, stage, peerRank, 0, recvCount));
        if (recvCount > 0) {
            void *dst = AddOffset(bufferBase, recvOffset * sizeof(float));
            const void *src = AddOffset(resCtx.localBuffer.addr,
                static_cast<uint64_t>(sourceLocalIndex) * slotStride);
            if (isGather) {
                CHK_RET(CheckTask(HcommLocalCopyOnThread(
                                      thread, dst, src, recvCount * sizeof(float)),
                    "small_mesh_copy", param.myRank, stage, peerRank, 0, recvCount));
            } else {
                CHK_RET(CheckTask(HcommLocalReduceOnThread(thread, dst, src, recvCount,
                                      HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM),
                    "small_mesh_reduce", param.myRank, stage, peerRank, 0, recvCount));
            }
        }
        CHK_RET(CheckTask(HcommChannelNotifyRecordOnThread(
                              thread, channel->handle, NOTIFY_IDX_ACK),
            "small_mesh_ack_record", param.myRank, stage, peerRank, 0, recvCount));
    }
    for (uint32_t peerLocalIndex = 0; peerLocalIndex < EXPECTED_LOCAL_RANK_SIZE;
         ++peerLocalIndex) {
        if (peerLocalIndex == localIndex) {
            continue;
        }
        const ChannelInfo *channel = channelByRank[resCtx.localRanks[peerLocalIndex]];
        CHK_RET(CheckTask(HcommChannelNotifyWaitOnThread(
                              thread, channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT),
            "small_mesh_ack_wait", param.myRank, stage, channel->remoteRank, 0, 0));
    }
    return HCCL_SUCCESS;
}

HcclResult RunPairAllReduce(const OpParam &param, const AlgResourceCtx &resCtx,
    const ChannelMap &channelByRank, const BufferLayout &layout, void *bufferBase,
    uint64_t count)
{
    if (count == 0) {
        return HCCL_SUCCESS;
    }
    const ChannelInfo *channel = channelByRank[resCtx.pairRank];
    const ThreadHandle thread = resCtx.threads[0];
    const uint64_t capacityCount = layout.pairCapacity / sizeof(float);
    for (uint64_t offset = 0, segment = 0; offset < count; ++segment) {
        const uint64_t currentCount = std::min(capacityCount, count - offset);
        const uint64_t currentBytes = currentCount * sizeof(float);
        void *remoteDst = AddOffset(channel->remoteCclMem.addr, layout.pairOffset);
        CHK_RET(CheckTask(HcommWriteOnThread(thread, channel->handle, remoteDst,
                                  AddOffset(bufferBase, offset * sizeof(float)), currentBytes),
            "small_pair_write", param.myRank, segment, resCtx.pairRank, 0, currentBytes));
        CHK_RET(CheckTask(HcommChannelNotifyRecordOnThread(
                              thread, channel->handle, NOTIFY_IDX_DATA_SIGNAL),
            "small_pair_data_record", param.myRank, segment, resCtx.pairRank, 0, currentBytes));
        CHK_RET(CheckTask(HcommChannelNotifyWaitOnThread(
                              thread, channel->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT),
            "small_pair_data_wait", param.myRank, segment, resCtx.pairRank, 0, currentBytes));
        CHK_RET(CheckTask(HcommLocalReduceOnThread(thread,
                                  AddOffset(bufferBase, offset * sizeof(float)),
                                  AddOffset(resCtx.localBuffer.addr, layout.pairOffset), currentCount,
                                  HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM),
            "small_pair_reduce", param.myRank, segment, resCtx.pairRank, 0, currentCount));
        CHK_RET(CheckTask(HcommChannelNotifyRecordOnThread(
                              thread, channel->handle, NOTIFY_IDX_ACK),
            "small_pair_ack_record", param.myRank, segment, resCtx.pairRank, 0, currentCount));
        CHK_RET(CheckTask(HcommChannelNotifyWaitOnThread(
                              thread, channel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT),
            "small_pair_ack_wait", param.myRank, segment, resCtx.pairRank, 0, currentCount));
        offset += currentCount;
    }
    return HCCL_SUCCESS;
}

HcclResult RunTinyAllReduce(const OpParam &param, const AlgResourceCtx &resCtx,
    const ChannelMap &channelByRank, uint64_t usableBufferSize)
{
    BufferLayout layout{};
    CHK_RET(CalculateBufferLayout(param.myRank, usableBufferSize,
        CeilDiv(param.count, EXPECTED_LOCAL_RANK_SIZE), layout));
    CHK_RET(CopyInSegments(param, resCtx.threads[0], param.outputPtr, param.inputPtr,
        param.count * sizeof(float), "small_input_copy"));

    CHK_RET(RunSerialMesh(param, resCtx, channelByRank, 1, param.outputPtr, param.count,
        layout.meshSlotStride, false));
    uint64_t ownCount = 0;
    uint64_t ownOffset = 0;
    SplitRange(param.count, resCtx.localRankIndex, EXPECTED_LOCAL_RANK_SIZE, ownCount, ownOffset);
    CHK_RET(RunPairAllReduce(param, resCtx, channelByRank, layout,
        AddOffset(param.outputPtr, ownOffset * sizeof(float)), ownCount));
    CHK_RET(RunSerialMesh(param, resCtx, channelByRank, 3, param.outputPtr, param.count,
        layout.meshSlotStride, true));
    return HCCL_SUCCESS;
}

HcclResult RunButterflyAllReduce(const OpParam &param, const AlgResourceCtx &resCtx,
    const ChannelMap &channelByRank, uint64_t usableBufferSize)
{
    const uint64_t dataBytes = param.count * sizeof(float);
    const uint64_t alignedDataBytes = AlignUp(dataBytes, BUFFER_ALIGNMENT);
    if (alignedDataBytes > usableBufferSize / BUTTERFLY_ROUND_NUM) {
        HCCL_WARNING("Butterfly scratch does not fit, rank[%u], bytes[%llu], aligned[%llu], "
                     "usable[%llu]; use hierarchical fallback",
            param.myRank, static_cast<unsigned long long>(dataBytes),
            static_cast<unsigned long long>(alignedDataBytes),
            static_cast<unsigned long long>(usableBufferSize));
        return RunTinyAllReduce(param, resCtx, channelByRank, usableBufferSize);
    }

    const ThreadHandle mainThread = resCtx.threads[0];
    CHK_RET(CopyInSegments(param, mainThread, param.outputPtr, param.inputPtr,
        dataBytes, "butterfly_input_copy"));
    for (uint32_t round = 0; round < BUTTERFLY_ROUND_NUM; ++round) {
        const uint32_t peerRank = GetButterflyPeerRank(resCtx, round);
        const ChannelInfo *channel = channelByRank[peerRank];
        if (channel == nullptr) {
            return HCCL_E_NOT_FOUND;
        }
        const uint64_t roundOffset = static_cast<uint64_t>(round) * alignedDataBytes;
        void *remoteDst = AddOffset(channel->remoteCclMem.addr, roundOffset);
        CHK_RET(CheckTask(HcommWriteOnThread(mainThread, channel->handle,
                                  remoteDst, param.outputPtr, dataBytes),
            "butterfly_write", param.myRank, round, peerRank, 0, dataBytes));
        CHK_RET(CheckTask(HcommChannelNotifyRecordOnThread(
                              mainThread, channel->handle, NOTIFY_IDX_DATA_SIGNAL),
            "butterfly_data_record", param.myRank, round, peerRank, 0, dataBytes));
        CHK_RET(CheckTask(HcommChannelNotifyWaitOnThread(
                              mainThread, channel->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT),
            "butterfly_data_wait", param.myRank, round, peerRank, 0, dataBytes));
        CHK_RET(CheckTask(HcommLocalReduceOnThread(mainThread, param.outputPtr,
                                  AddOffset(resCtx.localBuffer.addr, roundOffset), param.count,
                                  HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM),
            "butterfly_reduce", param.myRank, round, peerRank, 0, param.count));
    }
    return RunButterflyBarrier(
        param, resCtx, channelByRank, 40, "butterfly_final_barrier");
}

HcclResult RunLocalRabenseifnerAllReduce(const OpParam &param,
    const AlgResourceCtx &resCtx, const ChannelMap &channelByRank,
    uint64_t usableBufferSize)
{
    const uint64_t dataBytes = param.count * sizeof(float);
    const uint64_t alignedDataBytes = AlignUp(dataBytes, BUFFER_ALIGNMENT);
    uint64_t maxShardCount = 0;
    uint64_t ignoredOffset = 0;
    SplitRange(param.count, 0, EXPECTED_LOCAL_RANK_SIZE, maxShardCount, ignoredOffset);
    const uint64_t pairOffset = LOCAL_RABENSEIFNER_ROUND_NUM * alignedDataBytes;
    const uint64_t pairBytes = maxShardCount * sizeof(float);
    constexpr uint32_t RABENSEIFNER_SCRATCH_SLOT_NUM =
        LOCAL_RABENSEIFNER_ROUND_NUM * 2 + 1;
    if (alignedDataBytes > usableBufferSize / RABENSEIFNER_SCRATCH_SLOT_NUM
        || pairBytes > alignedDataBytes) {
        HCCL_WARNING("Rabenseifner scratch does not fit, rank[%u]; use Butterfly fallback",
            param.myRank);
        return RunButterflyAllReduce(param, resCtx, channelByRank, usableBufferSize);
    }

    const ThreadHandle thread = resCtx.threads[0];
    const uint32_t localIndex = resCtx.localRankIndex;
    CHK_RET(CopyInSegments(param, thread, param.outputPtr, param.inputPtr,
        dataBytes, "rabenseifner_input_copy"));

    constexpr std::array<uint32_t, LOCAL_RABENSEIFNER_ROUND_NUM> RS_MASKS = {4, 2, 1};
    for (uint32_t round = 0; round < LOCAL_RABENSEIFNER_ROUND_NUM; ++round) {
        const uint32_t mask = RS_MASKS[round];
        const uint32_t peerLocalIndex = localIndex ^ mask;
        const uint32_t peerRank = resCtx.localRanks[peerLocalIndex];
        const ChannelInfo *channel = channelByRank[peerRank];
        if (channel == nullptr) {
            return HCCL_E_NOT_FOUND;
        }
        const uint32_t groupBegin = localIndex & ~(2U * mask - 1U);
        const uint32_t keepPartBegin = groupBegin + ((localIndex & mask) == 0 ? 0 : mask);
        const uint32_t sendPartBegin = keepPartBegin ^ mask;
        uint64_t keepCount = 0;
        uint64_t keepOffset = 0;
        uint64_t sendCount = 0;
        uint64_t sendOffset = 0;
        GetPartInterval(param.count, keepPartBegin, mask, keepCount, keepOffset);
        GetPartInterval(param.count, sendPartBegin, mask, sendCount, sendOffset);
        const uint64_t scratchOffset = static_cast<uint64_t>(round) * alignedDataBytes;
        const uint64_t sendBytes = sendCount * sizeof(float);
        if (sendBytes > alignedDataBytes || keepCount * sizeof(float) > alignedDataBytes) {
            return HCCL_E_MEMORY;
        }
        CHK_RET(CheckTask(HcommWriteOnThread(thread, channel->handle,
                                  AddOffset(channel->remoteCclMem.addr, scratchOffset),
                                  AddOffset(param.outputPtr, sendOffset * sizeof(float)), sendBytes),
            "rabenseifner_rs_write", param.myRank, round, peerRank, 0, sendBytes));
        CHK_RET(CheckTask(HcommChannelNotifyRecordOnThread(
                              thread, channel->handle, NOTIFY_IDX_DATA_SIGNAL),
            "rabenseifner_rs_data_record", param.myRank, round, peerRank, 0, sendBytes));
        CHK_RET(CheckTask(HcommChannelNotifyWaitOnThread(
                              thread, channel->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT),
            "rabenseifner_rs_data_wait", param.myRank, round, peerRank, 0, keepCount));
        CHK_RET(CheckTask(HcommLocalReduceOnThread(thread,
                                  AddOffset(param.outputPtr, keepOffset * sizeof(float)),
                                  AddOffset(resCtx.localBuffer.addr, scratchOffset), keepCount,
                                  HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM),
            "rabenseifner_rs_reduce", param.myRank, round, peerRank, 0, keepCount));
    }

    uint64_t ownCount = 0;
    uint64_t ownOffset = 0;
    SplitRange(param.count, localIndex, EXPECTED_LOCAL_RANK_SIZE, ownCount, ownOffset);
    const ChannelInfo *pairChannel = channelByRank[resCtx.pairRank];
    if (pairChannel == nullptr) {
        return HCCL_E_NOT_FOUND;
    }
    const uint64_t ownBytes = ownCount * sizeof(float);
    void *ownOutput = AddOffset(param.outputPtr, ownOffset * sizeof(float));
    CHK_RET(CheckTask(HcommWriteOnThread(thread, pairChannel->handle,
                              AddOffset(pairChannel->remoteCclMem.addr, pairOffset),
                              ownOutput, ownBytes),
        "rabenseifner_pair_write", param.myRank, 3, resCtx.pairRank, 0, ownBytes));
    CHK_RET(CheckTask(HcommChannelNotifyRecordOnThread(
                          thread, pairChannel->handle, NOTIFY_IDX_DATA_SIGNAL),
        "rabenseifner_pair_data_record", param.myRank, 3, resCtx.pairRank, 0, ownBytes));
    CHK_RET(CheckTask(HcommChannelNotifyWaitOnThread(
                          thread, pairChannel->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT),
        "rabenseifner_pair_data_wait", param.myRank, 3, resCtx.pairRank, 0, ownCount));
    CHK_RET(CheckTask(HcommLocalReduceOnThread(thread, ownOutput,
                              AddOffset(resCtx.localBuffer.addr, pairOffset), ownCount,
                              HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM),
        "rabenseifner_pair_reduce", param.myRank, 3, resCtx.pairRank, 0, ownCount));
    CHK_RET(CheckTask(HcommChannelNotifyRecordOnThread(
                          thread, pairChannel->handle, NOTIFY_IDX_ACK),
        "rabenseifner_pair_ack_record", param.myRank, 3, resCtx.pairRank, 0, ownCount));
    CHK_RET(CheckTask(HcommChannelNotifyWaitOnThread(
                          thread, pairChannel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT),
        "rabenseifner_pair_ack_wait", param.myRank, 3, resCtx.pairRank, 0, ownCount));

    constexpr std::array<uint32_t, LOCAL_RABENSEIFNER_ROUND_NUM> AG_MASKS = {1, 2, 4};
    for (uint32_t round = 0; round < LOCAL_RABENSEIFNER_ROUND_NUM; ++round) {
        const uint32_t mask = AG_MASKS[round];
        const uint32_t peerLocalIndex = localIndex ^ mask;
        const uint32_t peerRank = resCtx.localRanks[peerLocalIndex];
        const ChannelInfo *channel = channelByRank[peerRank];
        if (channel == nullptr) {
            return HCCL_E_NOT_FOUND;
        }
        const uint32_t sendPartBegin = localIndex & ~(mask - 1U);
        const uint32_t recvPartBegin = peerLocalIndex & ~(mask - 1U);
        uint64_t sendCount = 0;
        uint64_t sendOffset = 0;
        uint64_t recvCount = 0;
        uint64_t recvOffset = 0;
        GetPartInterval(param.count, sendPartBegin, mask, sendCount, sendOffset);
        GetPartInterval(param.count, recvPartBegin, mask, recvCount, recvOffset);
        const uint64_t scratchOffset =
            static_cast<uint64_t>(LOCAL_RABENSEIFNER_ROUND_NUM + 1 + round) * alignedDataBytes;
        const uint64_t sendBytes = sendCount * sizeof(float);
        if (sendBytes > alignedDataBytes || recvCount * sizeof(float) > alignedDataBytes) {
            return HCCL_E_MEMORY;
        }
        CHK_RET(CheckTask(HcommWriteOnThread(thread, channel->handle,
                                  AddOffset(channel->remoteCclMem.addr, scratchOffset),
                                  AddOffset(param.outputPtr, sendOffset * sizeof(float)), sendBytes),
            "rabenseifner_ag_write", param.myRank, 4 + round, peerRank, 0, sendBytes));
        CHK_RET(CheckTask(HcommChannelNotifyRecordOnThread(
                              thread, channel->handle, NOTIFY_IDX_DATA_SIGNAL),
            "rabenseifner_ag_data_record", param.myRank, 4 + round, peerRank, 0, sendBytes));
        CHK_RET(CheckTask(HcommChannelNotifyWaitOnThread(
                              thread, channel->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT),
            "rabenseifner_ag_data_wait", param.myRank, 4 + round, peerRank, 0, recvCount));
        CHK_RET(CheckTask(HcommLocalCopyOnThread(thread,
                              AddOffset(param.outputPtr, recvOffset * sizeof(float)),
                              AddOffset(resCtx.localBuffer.addr, scratchOffset),
                              recvCount * sizeof(float)),
            "rabenseifner_ag_copy", param.myRank, 4 + round, peerRank, 0, recvCount));
    }
    return RunLocalButterflyBarrier(
        param, resCtx, channelByRank, 70, "rabenseifner_local_final_barrier");
}

HcclResult RunLocalTwoTwoTwoAllReduce(const OpParam &param,
    const AlgResourceCtx &resCtx, const ChannelMap &channelByRank,
    uint64_t usableBufferSize)
{
    const uint64_t dataBytes = param.count * sizeof(float);
    const uint64_t alignedDataBytes = AlignUp(dataBytes, BUFFER_ALIGNMENT);
    constexpr uint32_t TWO_TWO_TWO_SCRATCH_SLOT_NUM = 6;
    if (alignedDataBytes > usableBufferSize / TWO_TWO_TWO_SCRATCH_SLOT_NUM) {
        HCCL_WARNING("2+2+2 scratch does not fit, rank[%u]; use Butterfly fallback",
            param.myRank);
        return RunButterflyAllReduce(param, resCtx, channelByRank, usableBufferSize);
    }

    const ThreadHandle thread = resCtx.threads[0];
    const uint32_t localIndex = resCtx.localRankIndex;

    constexpr std::array<uint32_t, TWO_TWO_TWO_SIDE_ROUND_NUM> RS_MASKS = {4, 2};
    for (uint32_t round = 0; round < TWO_TWO_TWO_SIDE_ROUND_NUM; ++round) {
        const uint32_t mask = RS_MASKS[round];
        const uint32_t peerLocalIndex = localIndex ^ mask;
        const uint32_t peerRank = resCtx.localRanks[peerLocalIndex];
        const ChannelInfo *channel = channelByRank[peerRank];
        if (channel == nullptr) {
            return HCCL_E_NOT_FOUND;
        }
        const uint32_t groupBegin = localIndex & ~(2U * mask - 1U);
        const uint32_t keepPartBegin = groupBegin + ((localIndex & mask) == 0 ? 0 : mask);
        const uint32_t sendPartBegin = keepPartBegin ^ mask;
        uint64_t keepCount = 0;
        uint64_t keepOffset = 0;
        uint64_t sendCount = 0;
        uint64_t sendOffset = 0;
        GetPartInterval(param.count, keepPartBegin, mask, keepCount, keepOffset);
        GetPartInterval(param.count, sendPartBegin, mask, sendCount, sendOffset);
        const uint64_t scratchOffset = static_cast<uint64_t>(round) * alignedDataBytes;
        const uint64_t sendBytes = sendCount * sizeof(float);
        if (sendBytes > alignedDataBytes || keepCount * sizeof(float) > alignedDataBytes) {
            return HCCL_E_MEMORY;
        }

        const void *sendSrc = (round == 0) ? param.inputPtr : param.outputPtr;
        CHK_RET(CheckTask(HcommWriteOnThread(thread, channel->handle,
                                AddOffset(channel->remoteCclMem.addr, scratchOffset),
                                AddOffset(sendSrc, sendOffset * sizeof(float)), sendBytes),
            "two_two_two_rs_write", param.myRank, round, peerRank, 0, sendBytes));
        CHK_RET(CheckTask(HcommChannelNotifyRecordOnThread(
                            thread, channel->handle, NOTIFY_IDX_DATA_SIGNAL),
            "two_two_two_rs_data_record", param.myRank, round, peerRank, 0, sendBytes));

        if (round == 0) {
            CHK_RET(CopyInSegments(param, thread,
                AddOffset(param.outputPtr, keepOffset * sizeof(float)),
                AddOffset(param.inputPtr, keepOffset * sizeof(float)),
                keepCount * sizeof(float), "two_two_two_keep_copy"));
        }

        CHK_RET(CheckTask(HcommChannelNotifyWaitOnThread(
                            thread, channel->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT),
            "two_two_two_rs_data_wait", param.myRank, round, peerRank, 0, keepCount));
        CHK_RET(CheckTask(HcommLocalReduceOnThread(thread,
                                AddOffset(param.outputPtr, keepOffset * sizeof(float)),
                                AddOffset(resCtx.localBuffer.addr, scratchOffset), keepCount,
                                HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM),
            "two_two_two_rs_reduce", param.myRank, round, peerRank, 0, keepCount));
    }

    const uint32_t activePartBegin = localIndex & ~1U;
    uint64_t activeCount = 0;
    uint64_t activeOffset = 0;
    GetPartInterval(param.count, activePartBegin, 2, activeCount, activeOffset);
    const uint64_t activeBytes = activeCount * sizeof(float);
    if (activeBytes > alignedDataBytes) {
        return HCCL_E_MEMORY;
    }
    const uint32_t localPeerRank = resCtx.localRanks[localIndex ^ 1U];
    const ChannelInfo *localChannel = channelByRank[localPeerRank];
    if (localChannel == nullptr) {
        return HCCL_E_NOT_FOUND;
    }
    constexpr uint32_t LOCAL_FULL_REDUCE_SLOT = 2;
    const uint64_t localReduceOffset =
        static_cast<uint64_t>(LOCAL_FULL_REDUCE_SLOT) * alignedDataBytes;
    void *activeOutput = AddOffset(param.outputPtr, activeOffset * sizeof(float));
    CHK_RET(CheckTask(HcommWriteOnThread(thread, localChannel->handle,
                              AddOffset(localChannel->remoteCclMem.addr, localReduceOffset),
                              activeOutput, activeBytes),
        "two_two_two_local_write", param.myRank, 2, localPeerRank, 0, activeBytes));
    CHK_RET(CheckTask(HcommChannelNotifyRecordOnThread(
                          thread, localChannel->handle, NOTIFY_IDX_DATA_SIGNAL),
        "two_two_two_local_data_record", param.myRank, 2, localPeerRank, 0, activeBytes));
    CHK_RET(CheckTask(HcommChannelNotifyWaitOnThread(
                          thread, localChannel->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT),
        "two_two_two_local_data_wait", param.myRank, 2, localPeerRank, 0, activeCount));
    CHK_RET(CheckTask(HcommLocalReduceOnThread(thread, activeOutput,
                              AddOffset(resCtx.localBuffer.addr, localReduceOffset), activeCount,
                              HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM),
        "two_two_two_local_reduce", param.myRank, 2, localPeerRank, 0, activeCount));

    const ChannelInfo *pairChannel = channelByRank[resCtx.pairRank];
    if (pairChannel == nullptr) {
        return HCCL_E_NOT_FOUND;
    }
    constexpr uint32_t PAIR_FULL_REDUCE_SLOT = 3;
    const uint64_t pairOffset =
        static_cast<uint64_t>(PAIR_FULL_REDUCE_SLOT) * alignedDataBytes;
    CHK_RET(CheckTask(HcommWriteOnThread(thread, pairChannel->handle,
                              AddOffset(pairChannel->remoteCclMem.addr, pairOffset),
                              activeOutput, activeBytes),
        "two_two_two_pair_write", param.myRank, 3, resCtx.pairRank, 0, activeBytes));
    CHK_RET(CheckTask(HcommChannelNotifyRecordOnThread(
                          thread, pairChannel->handle, NOTIFY_IDX_DATA_SIGNAL),
        "two_two_two_pair_data_record", param.myRank, 3, resCtx.pairRank, 0, activeBytes));
    CHK_RET(CheckTask(HcommChannelNotifyWaitOnThread(
                          thread, pairChannel->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT),
        "two_two_two_pair_data_wait", param.myRank, 3, resCtx.pairRank, 0, activeCount));
    CHK_RET(CheckTask(HcommLocalReduceOnThread(thread, activeOutput,
                              AddOffset(resCtx.localBuffer.addr, pairOffset), activeCount,
                              HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM),
        "two_two_two_pair_reduce", param.myRank, 3, resCtx.pairRank, 0, activeCount));
    CHK_RET(CheckTask(HcommChannelNotifyRecordOnThread(
                          thread, pairChannel->handle, NOTIFY_IDX_ACK),
        "two_two_two_pair_ack_record", param.myRank, 3, resCtx.pairRank, 0, activeCount));

    constexpr std::array<uint32_t, TWO_TWO_TWO_SIDE_ROUND_NUM> AG_MASKS = {2, 4};
    for (uint32_t round = 0; round < TWO_TWO_TWO_SIDE_ROUND_NUM; ++round) {
        const uint32_t mask = AG_MASKS[round];
        const uint32_t peerLocalIndex = localIndex ^ mask;
        const uint32_t peerRank = resCtx.localRanks[peerLocalIndex];
        const ChannelInfo *channel = channelByRank[peerRank];
        if (channel == nullptr) {
            return HCCL_E_NOT_FOUND;
        }
        const uint32_t sendPartBegin = localIndex & ~(mask - 1U);
        const uint32_t recvPartBegin = peerLocalIndex & ~(mask - 1U);
        uint64_t sendCount = 0;
        uint64_t sendOffset = 0;
        uint64_t recvCount = 0;
        uint64_t recvOffset = 0;
        GetPartInterval(param.count, sendPartBegin, mask, sendCount, sendOffset);
        GetPartInterval(param.count, recvPartBegin, mask, recvCount, recvOffset);
        const uint64_t scratchOffset =
            static_cast<uint64_t>(4 + round) * alignedDataBytes;
        const uint64_t sendBytes = sendCount * sizeof(float);
        if (sendBytes > alignedDataBytes || recvCount * sizeof(float) > alignedDataBytes) {
            return HCCL_E_MEMORY;
        }
        CHK_RET(CheckTask(HcommWriteOnThread(thread, channel->handle,
                                  AddOffset(channel->remoteCclMem.addr, scratchOffset),
                                  AddOffset(param.outputPtr, sendOffset * sizeof(float)), sendBytes),
            "two_two_two_ag_write", param.myRank, 4 + round, peerRank, 0, sendBytes));
        CHK_RET(CheckTask(HcommChannelNotifyRecordOnThread(
                              thread, channel->handle, NOTIFY_IDX_DATA_SIGNAL),
            "two_two_two_ag_data_record", param.myRank, 4 + round, peerRank, 0, sendBytes));
        CHK_RET(CheckTask(HcommChannelNotifyWaitOnThread(
                              thread, channel->handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT),
            "two_two_two_ag_data_wait", param.myRank, 4 + round, peerRank, 0, recvCount));
        CHK_RET(CheckTask(HcommLocalCopyOnThread(thread,
                              AddOffset(param.outputPtr, recvOffset * sizeof(float)),
                              AddOffset(resCtx.localBuffer.addr, scratchOffset),
                              recvCount * sizeof(float)),
            "two_two_two_ag_copy", param.myRank, 4 + round, peerRank, 0, recvCount));
    }

    CHK_RET(CheckTask(HcommChannelNotifyWaitOnThread(
                          thread, pairChannel->handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT),
        "two_two_two_pair_ack_wait", param.myRank, 3, resCtx.pairRank, 0, activeCount));

    return RunLocalScopedDirectBarrier(
        param, resCtx, channelByRank, 90, "two_two_two_local_final_barrier");
}

struct TileRange {
    uint64_t offset = 0;
    uint64_t count = 0;
    uint64_t ownOffset = 0;
    uint64_t ownCount = 0;
};

TileRange GetTileRange(
    uint64_t totalCount, uint64_t tileCapacityCount, uint64_t tileIndex, uint32_t localRankIndex)
{
    TileRange tile{};
    tile.offset = tileIndex * tileCapacityCount;
    tile.count = std::min(tileCapacityCount, totalCount - tile.offset);
    SplitRange(tile.count, localRankIndex, EXPECTED_LOCAL_RANK_SIZE,
        tile.ownCount, tile.ownOffset);
    return tile;
}

HcclResult InitializeOwnShard(const OpParam &param, const AlgResourceCtx &resCtx,
    const TileRange &tile, const char *phase)
{
    const uint64_t globalOwnOffset = tile.offset + tile.ownOffset;
    return CopyInSegments(param, resCtx.threads[0],
        AddOffset(param.outputPtr, globalOwnOffset * sizeof(float)),
        AddOffset(param.inputPtr, globalOwnOffset * sizeof(float)),
        tile.ownCount * sizeof(float), phase);
}

HcclResult ScheduleTileMesh(const OpParam &param, const AlgResourceCtx &resCtx,
    const ChannelMap &channelByRank, const BufferLayout &layout, const TileRange &tile,
    uint64_t stage, bool isGather)
{
    void *base = isGather ? AddOffset(param.outputPtr, tile.offset * sizeof(float))
                          : AddOffset(param.inputPtr, tile.offset * sizeof(float));
    return ScheduleParallelMeshSends(param, resCtx, channelByRank, stage,
        isGather ? "tile_mesh_ag" : "tile_mesh_rs", base, tile.count,
        layout.meshSlotStride, isGather);
}

HcclResult CompleteTileMesh(const OpParam &param, const AlgResourceCtx &resCtx,
    const ChannelMap &channelByRank, const BufferLayout &layout, const TileRange &tile,
    uint64_t stage, bool isGather)
{
    void *outputBase = AddOffset(param.outputPtr, tile.offset * sizeof(float));
    return CompleteParallelMesh(param, resCtx, channelByRank, stage,
        isGather ? "tile_mesh_ag" : "tile_mesh_rs", outputBase, tile.count,
        layout.meshSlotStride, isGather);
}

HcclResult RunTilePairAllReduce(const OpParam &param, const AlgResourceCtx &resCtx,
    const ChannelMap &channelByRank, const BufferLayout &layout, const TileRange &tile)
{
    const uint64_t globalOwnOffset = tile.offset + tile.ownOffset;
    return RunPairAllReduce(param, resCtx, channelByRank, layout,
        AddOffset(param.outputPtr, globalOwnOffset * sizeof(float)), tile.ownCount);
}

HcclResult RunTiledHierarchicalAllReduce(const OpParam &param, const AlgResourceCtx &resCtx,
    const ChannelMap &channelByRank, uint64_t usableBufferSize)
{
    const uint64_t tileCapacityCount = PIPELINE_TILE_BYTES / sizeof(float);
    const uint64_t tileNum = CeilDiv(param.count, tileCapacityCount);
    const uint64_t maxTileCount = std::min(param.count, tileCapacityCount);
    BufferLayout layout{};
    CHK_RET(CalculateBufferLayout(param.myRank, usableBufferSize,
        CeilDiv(maxTileCount, EXPECTED_LOCAL_RANK_SIZE), layout));

    HCCL_INFO("Tiled hierarchical allreduce, rank[%u], count[%llu], tileNum[%llu], "
              "tileCapacity[%llu], meshSlot[%llu], pairCapacity[%llu]",
        param.myRank, static_cast<unsigned long long>(param.count),
        static_cast<unsigned long long>(tileNum),
        static_cast<unsigned long long>(tileCapacityCount),
        static_cast<unsigned long long>(layout.meshSlotStride),
        static_cast<unsigned long long>(layout.pairCapacity));

    TileRange previous{};
    for (uint64_t tileIndex = 0; tileIndex < tileNum; ++tileIndex) {
        const TileRange current = GetTileRange(
            param.count, tileCapacityCount, tileIndex, resCtx.localRankIndex);
        CHK_RET(InitializeOwnShard(param, resCtx, current, "tile_own_copy"));
        const uint64_t rsStage = 10 + tileIndex;
        CHK_RET(ScheduleTileMesh(
            param, resCtx, channelByRank, layout, current, rsStage, false));
        if (tileIndex > 0) {
            CHK_RET(RunTilePairAllReduce(param, resCtx, channelByRank, layout, previous));
        }
        CHK_RET(CompleteTileMesh(
            param, resCtx, channelByRank, layout, current, rsStage, false));
        previous = current;
    }

    const TileRange first = GetTileRange(
        param.count, tileCapacityCount, 0, resCtx.localRankIndex);
    if (tileNum == 1) {
        CHK_RET(RunTilePairAllReduce(param, resCtx, channelByRank, layout, previous));
        CHK_RET(ScheduleTileMesh(param, resCtx, channelByRank, layout, first, 100, true));
        CHK_RET(CompleteTileMesh(param, resCtx, channelByRank, layout, first, 100, true));
        return HCCL_SUCCESS;
    }

    CHK_RET(ScheduleTileMesh(param, resCtx, channelByRank, layout, first, 100, true));
    CHK_RET(RunTilePairAllReduce(param, resCtx, channelByRank, layout, previous));
    CHK_RET(CompleteTileMesh(param, resCtx, channelByRank, layout, first, 100, true));
    for (uint64_t tileIndex = 1; tileIndex < tileNum; ++tileIndex) {
        const TileRange current = GetTileRange(
            param.count, tileCapacityCount, tileIndex, resCtx.localRankIndex);
        const uint64_t agStage = 100 + tileIndex;
        CHK_RET(ScheduleTileMesh(
            param, resCtx, channelByRank, layout, current, agStage, true));
        CHK_RET(CompleteTileMesh(
            param, resCtx, channelByRank, layout, current, agStage, true));
    }
    return HCCL_SUCCESS;
}

HcclResult RunGlobalTiledMeshAllReduce(const OpParam &param, const AlgResourceCtx &resCtx,
    const ChannelMap &channelByRank, uint64_t usableBufferSize)
{
    const uint64_t slotCapacityBytes = AlignDown(
        usableBufferSize / GLOBAL_WORKER_NUM, BUFFER_ALIGNMENT);
    if (slotCapacityBytes < sizeof(float)
        || slotCapacityBytes / sizeof(float)
            > std::numeric_limits<uint64_t>::max() / EXPECTED_RANK_SIZE) {
        HCCL_ERROR("Global Mesh tile capacity is invalid, rank[%u], usable[%llu], slot[%llu]",
            param.myRank, static_cast<unsigned long long>(usableBufferSize),
            static_cast<unsigned long long>(slotCapacityBytes));
        return HCCL_E_MEMORY;
    }
    const uint64_t tileCapacityCount =
        slotCapacityBytes / sizeof(float) * EXPECTED_RANK_SIZE;
    const uint64_t maxTileCount = std::min(param.count, tileCapacityCount);
    GlobalMeshLayout layout{};
    CHK_RET(CalculateGlobalMeshLayout(
        param.myRank, usableBufferSize, maxTileCount, layout));

    const uint64_t tileNum = CeilDiv(param.count, tileCapacityCount);
    HCCL_INFO("Global tiled Mesh allreduce, rank[%u], count[%llu], tileNum[%llu], "
              "tileCapacity[%llu], slotStride[%llu]",
        param.myRank, static_cast<unsigned long long>(param.count),
        static_cast<unsigned long long>(tileNum),
        static_cast<unsigned long long>(tileCapacityCount),
        static_cast<unsigned long long>(layout.slotStride));

    for (uint64_t tileIndex = 0; tileIndex < tileNum; ++tileIndex) {
        const uint64_t tileOffset = tileIndex * tileCapacityCount;
        const uint64_t tileCount = std::min(tileCapacityCount, param.count - tileOffset);
        uint64_t ownCount = 0;
        uint64_t ownOffset = 0;
        SplitRange(tileCount, param.myRank, EXPECTED_RANK_SIZE, ownCount, ownOffset);
        const uint64_t globalOwnOffset = tileOffset + ownOffset;
        const uint64_t rsStage = 200 + tileIndex * 2;
        const void *inputBase = AddOffset(param.inputPtr, tileOffset * sizeof(float));
        void *outputBase = AddOffset(param.outputPtr, tileOffset * sizeof(float));
        CHK_RET(ScheduleGlobalMeshSends(param, resCtx, channelByRank, rsStage,
            "global_mesh_rs", inputBase, tileCount, layout.slotStride, false, false));
        CHK_RET(CopyInSegments(param, resCtx.threads[0],
            AddOffset(param.outputPtr, globalOwnOffset * sizeof(float)),
            AddOffset(param.inputPtr, globalOwnOffset * sizeof(float)),
            ownCount * sizeof(float), "global_mesh_own_copy"));
        if (tileNum == 1) {
            CHK_RET(CompleteSerialGlobalReduceScatter(param, resCtx, channelByRank,
                rsStage, outputBase, tileCount, layout.slotStride));
        } else {
            CHK_RET(CompleteGroupedGlobalReduceScatter(param, resCtx, channelByRank,
                rsStage, outputBase, tileCount, layout.slotStride));
        }

        const uint64_t agStage = rsStage + 1;
        CHK_RET(RunGlobalAllGatherWorkers(param, resCtx, channelByRank, agStage,
            outputBase, tileCount, layout.slotStride));
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    ChannelMap channelByRank{};
    uint64_t usableBufferSize = 0;
    CHK_RET(ValidateResourceContext(param, resCtx, channelByRank, usableBufferSize));
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    const uint64_t dataBytes = param.count * sizeof(float);
    if (dataBytes <= SMALL_BUTTERFLY_THRESHOLD_BYTES) {
        if (dataBytes == SMALL_BUTTERFLY_THRESHOLD_BYTES) {
            return RunLocalTwoTwoTwoAllReduce(
                param, resCtx, channelByRank, usableBufferSize);
        }
        if (dataBytes >= SMALL_RABENSEIFNER_MIN_BYTES) {
            return RunLocalRabenseifnerAllReduce(param, resCtx, channelByRank, usableBufferSize);
        }
        return RunButterflyAllReduce(param, resCtx, channelByRank, usableBufferSize);
    }
    if (dataBytes >= LARGE_MESSAGE_THRESHOLD_BYTES) {
        return RunGlobalTiledMeshAllReduce(param, resCtx, channelByRank, usableBufferSize);
    }
    return RunTiledHierarchicalAllReduce(param, resCtx, channelByRank, usableBufferSize);
}
} // namespace ops_hccl