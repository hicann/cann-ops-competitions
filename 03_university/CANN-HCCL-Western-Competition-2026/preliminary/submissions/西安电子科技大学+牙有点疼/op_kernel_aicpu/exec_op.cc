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
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace ops_hccl {
namespace {
constexpr uint64_t HCCL_MIN_SLICE_ALIGN = 128;
constexpr uint64_t MAX_SLICE_BYTES = 256ULL * 1024ULL * 1024ULL;
constexpr uint32_t TWO_SERVER_RANK_SIZE = 16;
constexpr uint32_t RANKS_PER_SERVER = 8;
constexpr uint32_t TWO_SERVER_SLOT_NUM = RANKS_PER_SERVER + 1;
constexpr uint64_t TWO_SERVER_ALGO_MIN_BYTES = 64ULL * 1024ULL;
constexpr uint64_t TAIL_READ_REDUCE_MAX_BYTES = 1ULL * 1024ULL * 1024ULL;
constexpr uint64_t SMALL_HIERARCHICAL_ALLREDUCE_BYTES = 512ULL * 1024ULL;
constexpr uint32_t LARGE_HIERARCHICAL_STRIPE_NUM = RANKS_PER_SERVER - 1;
constexpr uint32_t LARGE_HIERARCHICAL_MESH_LANE_NUM = 2;
constexpr uint32_t LARGE_HIERARCHICAL_MESH_ROUND_NUM =
    (LARGE_HIERARCHICAL_STRIPE_NUM + LARGE_HIERARCHICAL_MESH_LANE_NUM - 1) /
    LARGE_HIERARCHICAL_MESH_LANE_NUM;
constexpr uint32_t LARGE_HIERARCHICAL_CCL_SLOT_NUM = 2;
constexpr uint32_t SERVER_BARRIER_NOTIFY_IDX = 2;
constexpr uint32_t HIER_INPUT_SLOT_IDX = 0;
constexpr uint32_t HIER_RESULT_SLOT_IDX = 1;
constexpr uint64_t HIERARCHICAL_ALGO_MIN_BYTES = TAIL_READ_REDUCE_MAX_BYTES + 1;

struct ExchangeEntry {
    const ChannelInfo *channel = nullptr;
    ThreadHandle thread = 0;
};

struct SyncThread {
    ThreadHandle thread = 0;
    uint32_t mainNotifyIdx = 0;
};

struct ChunkRange {
    uint64_t offsetCount = 0;
    uint64_t count = 0;
};

struct ReadPlan {
    uint64_t remoteOffset = 0;
    void *localAddr = nullptr;
};

struct VariableReadPlan {
    uint64_t remoteOffset = 0;
    void *localAddr = nullptr;
    uint64_t bytes = 0;
};

struct VariableWriteReducePlan {
    uint64_t remoteOffset = 0;
    const void *localAddr = nullptr;
    uint64_t count = 0;
};

struct TailSlotLayout {
    uint64_t inputOffset = 0;
    uint64_t resultOffset = 0;
};

bool GetDataTypeSize(HcclDataType dataType, uint32_t &size)
{
    switch (dataType) {
        case HCCL_DATA_TYPE_INT8:
        case HCCL_DATA_TYPE_UINT8:
        case HCCL_DATA_TYPE_HIF8:
        case HCCL_DATA_TYPE_FP8E4M3:
        case HCCL_DATA_TYPE_FP8E5M2:
        case HCCL_DATA_TYPE_FP8E8M0:
            size = 1;
            return true;
        case HCCL_DATA_TYPE_INT16:
        case HCCL_DATA_TYPE_UINT16:
        case HCCL_DATA_TYPE_FP16:
        case HCCL_DATA_TYPE_BFP16:
            size = 2;
            return true;
        case HCCL_DATA_TYPE_INT32:
        case HCCL_DATA_TYPE_UINT32:
        case HCCL_DATA_TYPE_FP32:
            size = 4;
            return true;
        case HCCL_DATA_TYPE_INT64:
        case HCCL_DATA_TYPE_UINT64:
        case HCCL_DATA_TYPE_FP64:
            size = 8;
            return true;
        case HCCL_DATA_TYPE_INT128:
            size = 16;
            return true;
        default:
            size = 0;
            return false;
    }
}

bool ToHcommDataType(HcclDataType dataType, HcommDataType &hcommDataType)
{
    uint32_t dataTypeSize = 0;
    if (!GetDataTypeSize(dataType, dataTypeSize)) {
        hcommDataType = HCOMM_DATA_TYPE_RESERVED;
        return false;
    }
    hcommDataType = static_cast<HcommDataType>(dataType);
    return true;
}

bool ToHcommReduceOp(HcclReduceOp reduceOp, HcommReduceOp &hcommReduceOp)
{
    switch (reduceOp) {
        case HCCL_REDUCE_SUM:
            hcommReduceOp = HCOMM_REDUCE_SUM;
            return true;
        case HCCL_REDUCE_PROD:
            hcommReduceOp = HCOMM_REDUCE_PROD;
            return true;
        case HCCL_REDUCE_MAX:
            hcommReduceOp = HCOMM_REDUCE_MAX;
            return true;
        case HCCL_REDUCE_MIN:
            hcommReduceOp = HCOMM_REDUCE_MIN;
            return true;
        default:
            hcommReduceOp = HCOMM_REDUCE_RESERVED;
            return false;
    }
}

void *PtrAdd(void *ptr, uint64_t offset)
{
    return static_cast<void *>(static_cast<uint8_t *>(ptr) + offset);
}

const void *PtrAddConst(const void *ptr, uint64_t offset)
{
    return static_cast<const void *>(static_cast<const uint8_t *>(ptr) + offset);
}

uint64_t AlignDown(uint64_t value, uint64_t align)
{
    return value / align * align;
}

uint64_t AlignUp(uint64_t value, uint64_t align)
{
    if (align == 0 || value == 0) {
        return value;
    }
    return (value + align - 1) / align * align;
}

uint64_t GetAlignedBufferOffset(const CommBuffer &buffer, uint64_t align)
{
    if (align == 0) {
        return 0;
    }
    const uint64_t misalign = reinterpret_cast<uintptr_t>(buffer.addr) % align;
    return misalign == 0 ? 0 : align - misalign;
}

uint32_t GetLocalRank(uint32_t rank)
{
    return rank % RANKS_PER_SERVER;
}

uint32_t GetServerStart(uint32_t rank)
{
    return rank / RANKS_PER_SERVER * RANKS_PER_SERVER;
}

bool IsTwoServerTopology(const OpParam &param)
{
    return param.rankSize == TWO_SERVER_RANK_SIZE;
}

bool ShouldUseHierarchicalTwoServerAlgo(const OpParam &param, uint64_t totalBytes)
{
    return IsTwoServerTopology(param) && totalBytes >= HIERARCHICAL_ALGO_MIN_BYTES;
}

bool ShouldUseTwoServerAlgo(const OpParam &param, uint64_t totalBytes)
{
    return IsTwoServerTopology(param) && totalBytes >= TWO_SERVER_ALGO_MIN_BYTES;
}

bool ShouldUseSmallHierarchicalAlgo(const OpParam &param, uint64_t totalBytes)
{
    return IsTwoServerTopology(param) && totalBytes == SMALL_HIERARCHICAL_ALLREDUCE_BYTES;
}

ChunkRange GetChunkRange(uint64_t sliceCount, uint32_t localRank)
{
    const uint64_t baseCount = sliceCount / RANKS_PER_SERVER;
    const uint64_t remainder = sliceCount % RANKS_PER_SERVER;
    ChunkRange range;
    range.offsetCount = baseCount * localRank + std::min<uint64_t>(localRank, remainder);
    range.count = baseCount + (localRank < remainder ? 1 : 0);
    return range;
}

ChunkRange GetPartitionRange(uint64_t totalCount, uint32_t partNum, uint32_t partIdx)
{
    if (partNum == 0 || partIdx >= partNum) {
        return ChunkRange{};
    }
    const uint64_t baseCount = totalCount / partNum;
    const uint64_t remainder = totalCount % partNum;
    ChunkRange range;
    range.offsetCount = baseCount * partIdx + std::min<uint64_t>(partIdx, remainder);
    range.count = baseCount + (partIdx < remainder ? 1 : 0);
    return range;
}

ChunkRange GetStripeRange(uint64_t chunkCount, uint32_t stripeIdx)
{
    const uint64_t baseCount = chunkCount / LARGE_HIERARCHICAL_STRIPE_NUM;
    const uint64_t remainder = chunkCount % LARGE_HIERARCHICAL_STRIPE_NUM;
    ChunkRange range;
    range.offsetCount = baseCount * stripeIdx + std::min<uint64_t>(stripeIdx, remainder);
    range.count = baseCount + (stripeIdx < remainder ? 1 : 0);
    return range;
}

uint32_t GetPackedContributionIndex(uint32_t sourceLocalRank, uint32_t ownerLocalRank)
{
    return sourceLocalRank < ownerLocalRank ? sourceLocalRank : sourceLocalRank - 1;
}

bool IsBufferRangeValid(const CommBuffer &buffer, uint64_t offset, uint64_t bytes)
{
    return offset <= buffer.size && bytes <= buffer.size - offset;
}

bool BuildTailSlotLayout(const CommBuffer &buffer, uint64_t tailBytes, TailSlotLayout &layout)
{
    const uint64_t baseOffset = GetAlignedBufferOffset(buffer, HCCL_MIN_SLICE_ALIGN);
    layout.inputOffset = baseOffset;
    layout.resultOffset = baseOffset + AlignUp(tailBytes, HCCL_MIN_SLICE_ALIGN);
    return IsBufferRangeValid(buffer, layout.inputOffset, tailBytes) &&
           IsBufferRangeValid(buffer, layout.resultOffset, tailBytes);
}

bool ShouldUseTailReadReduce(uint64_t remainingCount, uint64_t maxSliceCount, uint32_t dataTypeSize,
    const CommBuffer &localBuffer, TailSlotLayout &layout)
{
    if (remainingCount >= maxSliceCount) {
        return false;
    }

    const uint64_t remainingBytes = remainingCount * dataTypeSize;
    if (remainingBytes == 0 || remainingBytes > TAIL_READ_REDUCE_MAX_BYTES) {
        return false;
    }
    return BuildTailSlotLayout(localBuffer, remainingBytes, layout);
}

std::vector<uint32_t> GetAllPeers(const OpParam &param)
{
    std::vector<uint32_t> peers;
    peers.reserve(param.rankSize - 1);
    for (uint32_t rank = 0; rank < param.rankSize; rank++) {
        if (rank != param.myRank) {
            peers.push_back(rank);
        }
    }
    return peers;
}

std::vector<uint32_t> GetLocalServerPeers(uint32_t myRank)
{
    std::vector<uint32_t> peers;
    peers.reserve(RANKS_PER_SERVER - 1);
    const uint32_t serverStart = myRank / RANKS_PER_SERVER * RANKS_PER_SERVER;
    for (uint32_t rank = serverStart; rank < serverStart + RANKS_PER_SERVER; rank++) {
        if (rank != myRank) {
            peers.push_back(rank);
        }
    }
    return peers;
}

uint32_t GetCrossServerPeer(uint32_t myRank)
{
    return (myRank + RANKS_PER_SERVER) % TWO_SERVER_RANK_SIZE;
}

const ChannelInfo *FindChannel(const AlgResourceCtx &resCtx, uint32_t remoteRank, size_t &channelIdx)
{
    for (size_t idx = 0; idx < resCtx.channels.size(); idx++) {
        if (resCtx.channels[idx].remoteRank == remoteRank) {
            channelIdx = idx;
            return &resCtx.channels[idx];
        }
    }
    return nullptr;
}

ThreadHandle GetThreadForChannel(const AlgResourceCtx &resCtx, size_t channelIdx)
{
    (void)channelIdx;
    return resCtx.threads[0];
}

ThreadHandle GetWorkerThreadForChannel(const AlgResourceCtx &resCtx, size_t channelIdx, uint32_t &mainNotifyIdx)
{
    if (resCtx.threads.size() > 1) {
        const size_t workerIdx = channelIdx % (resCtx.threads.size() - 1) + 1;
        mainNotifyIdx = static_cast<uint32_t>(workerIdx - 1);
        return resCtx.threads[workerIdx];
    }
    mainNotifyIdx = 0;
    return resCtx.threads[0];
}

HcclResult BuildExchangeEntries(const AlgResourceCtx &resCtx, const std::vector<uint32_t> &peerRanks,
    std::vector<ExchangeEntry> &entries, std::vector<SyncThread> &syncThreads)
{
    entries.clear();
    entries.reserve(peerRanks.size());
    syncThreads.clear();
    syncThreads.push_back(SyncThread{resCtx.threads[0], 0});

    for (uint32_t peerRank : peerRanks) {
        size_t channelIdx = 0;
        const ChannelInfo *channel = FindChannel(resCtx, peerRank, channelIdx);
        if (channel == nullptr) {
            HCCL_ERROR("[BuildExchangeEntries] no channel to peer rank[%u]", peerRank);
            return HCCL_E_PARA;
        }

        const ThreadHandle commThread = GetThreadForChannel(resCtx, channelIdx);
        const auto syncIt = std::find_if(syncThreads.begin(), syncThreads.end(),
            [commThread](const SyncThread &syncThread) { return syncThread.thread == commThread; });
        if (commThread != resCtx.threads[0] && syncIt == syncThreads.end()) {
            syncThreads.push_back(SyncThread{commThread, 0});
        }
        entries.push_back(ExchangeEntry{channel, commThread});
    }
    return HCCL_SUCCESS;
}

HcclResult BuildWorkerExchangeEntries(const AlgResourceCtx &resCtx, const std::vector<uint32_t> &peerRanks,
    std::vector<ExchangeEntry> &entries, std::vector<SyncThread> &syncThreads)
{
    entries.clear();
    entries.reserve(peerRanks.size());
    syncThreads.clear();
    syncThreads.push_back(SyncThread{resCtx.threads[0], 0});

    for (uint32_t peerRank : peerRanks) {
        size_t channelIdx = 0;
        const ChannelInfo *channel = FindChannel(resCtx, peerRank, channelIdx);
        if (channel == nullptr) {
            HCCL_ERROR("[BuildWorkerExchangeEntries] no channel to peer rank[%u]", peerRank);
            return HCCL_E_PARA;
        }

        uint32_t mainNotifyIdx = 0;
        const ThreadHandle commThread = GetWorkerThreadForChannel(resCtx, channelIdx, mainNotifyIdx);
        const auto syncIt = std::find_if(syncThreads.begin(), syncThreads.end(),
            [commThread](const SyncThread &syncThread) { return syncThread.thread == commThread; });
        if (commThread != resCtx.threads[0] && syncIt == syncThreads.end()) {
            syncThreads.push_back(SyncThread{commThread, mainNotifyIdx});
        }
        entries.push_back(ExchangeEntry{channel, commThread});
    }
    return HCCL_SUCCESS;
}

HcclResult AssignUnusedWorkerThread(const AlgResourceCtx &resCtx,
    const std::vector<ExchangeEntry> &occupiedEntries, ExchangeEntry &entry, std::vector<SyncThread> &syncThreads)
{
    for (size_t workerIdx = 1; workerIdx < resCtx.threads.size(); workerIdx++) {
        const ThreadHandle candidate = resCtx.threads[workerIdx];
        const bool isOccupied = std::any_of(occupiedEntries.begin(), occupiedEntries.end(),
            [candidate](const ExchangeEntry &occupied) { return occupied.thread == candidate; });
        if (isOccupied) {
            continue;
        }
        entry.thread = candidate;
        syncThreads.clear();
        syncThreads.push_back(SyncThread{resCtx.threads[0], 0});
        syncThreads.push_back(SyncThread{candidate, static_cast<uint32_t>(workerIdx - 1)});
        return HCCL_SUCCESS;
    }
    HCCL_ERROR("[AssignUnusedWorkerThread] no worker is available for the independent exchange axis");
    return HCCL_E_PARA;
}

HcclResult ServerBarrier(const OpParam &param, const AlgResourceCtx &resCtx)
{
    const ThreadHandle thread = resCtx.threads[0];
    const uint32_t leaderRank = GetServerStart(param.myRank);
    if (param.myRank == leaderRank) {
        const std::vector<uint32_t> peers = GetLocalServerPeers(param.myRank);
        for (uint32_t peerRank : peers) {
            size_t channelIdx = 0;
            const ChannelInfo *channel = FindChannel(resCtx, peerRank, channelIdx);
            if (channel == nullptr) {
                HCCL_ERROR("[ServerBarrier] no channel from leader rank[%u] to peer rank[%u]", leaderRank, peerRank);
                return HCCL_E_PARA;
            }
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                thread, channel->handle, SERVER_BARRIER_NOTIFY_IDX, CUSTOM_TIMEOUT)));
        }
        for (uint32_t peerRank : peers) {
            size_t channelIdx = 0;
            const ChannelInfo *channel = FindChannel(resCtx, peerRank, channelIdx);
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(thread, channel->handle, SERVER_BARRIER_NOTIFY_IDX)));
        }
        return HCCL_SUCCESS;
    }

    size_t channelIdx = 0;
    const ChannelInfo *channel = FindChannel(resCtx, leaderRank, channelIdx);
    if (channel == nullptr) {
        HCCL_ERROR("[ServerBarrier] no channel from rank[%u] to leader rank[%u]", param.myRank, leaderRank);
        return HCCL_E_PARA;
    }
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel->handle, SERVER_BARRIER_NOTIFY_IDX)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel->handle, SERVER_BARRIER_NOTIFY_IDX, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult ThreadSyncBefore(const std::vector<SyncThread> &threads)
{
    for (size_t i = 1; i < threads.size(); i++) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(threads[0].thread, threads[i].thread, 0)));
    }
    for (size_t i = 1; i < threads.size(); i++) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[i].thread, 0, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult ThreadSyncAfter(const std::vector<SyncThread> &threads)
{
    for (size_t i = 1; i < threads.size(); i++) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[0].thread, threads[i].mainNotifyIdx,
            CUSTOM_TIMEOUT)));
    }
    for (size_t i = 1; i < threads.size(); i++) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            threads[i].thread, threads[0].thread, threads[i].mainNotifyIdx)));
    }
    return HCCL_SUCCESS;
}

HcclResult WriteReduceExchangeAsync(const ExchangeEntry &entry, uint64_t remoteOffset, const void *localAddr,
    uint64_t count, uint32_t dataTypeSize, HcommDataType dataType, HcommReduceOp reduceOp)
{
    const ChannelInfo &channel = *entry.channel;
    CHK_PTR_NULL(channel.remoteCclMem.addr);
    CHK_PTR_NULL(localAddr);
    if (dataTypeSize != 0 && count > std::numeric_limits<uint64_t>::max() / dataTypeSize) {
        return HCCL_E_PARA;
    }
    const uint64_t bytes = count * dataTypeSize;
    if (!IsBufferRangeValid(channel.remoteCclMem, remoteOffset, bytes)) {
        HCCL_ERROR("[WriteReduceExchangeAsync] remote buffer is too small, peerRank[%u], remoteOffset[%llu], "
                   "bytes[%llu], bufferSize[%llu]",
            channel.remoteRank, static_cast<unsigned long long>(remoteOffset),
            static_cast<unsigned long long>(bytes), static_cast<unsigned long long>(channel.remoteCclMem.size));
        return HCCL_E_MEMORY;
    }

    void *remoteAddr = PtrAdd(channel.remoteCclMem.addr, remoteOffset);
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(entry.thread, channel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(entry.thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(
        HcommWriteReduceOnThread(entry.thread, channel.handle, remoteAddr, localAddr, count, dataType, reduceOp)));
    CHK_RET(static_cast<HcclResult>(HcommChannelFenceOnThread(entry.thread, channel.handle)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(entry.thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        entry.thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult ReadWithEntries(const std::vector<ExchangeEntry> &entries, const std::vector<SyncThread> &syncThreads,
    const std::vector<ReadPlan> &readPlans, uint64_t bytes)
{
    if (entries.empty()) {
        return HCCL_SUCCESS;
    }
    if (readPlans.size() != entries.size()) {
        HCCL_ERROR("[ReadWithEntries] read plan size[%llu] does not match entries[%llu]",
            static_cast<unsigned long long>(readPlans.size()), static_cast<unsigned long long>(entries.size()));
        return HCCL_E_PARA;
    }

    CHK_RET(ThreadSyncBefore(syncThreads));
    for (size_t idx = 0; idx < entries.size(); idx++) {
        const ExchangeEntry &entry = entries[idx];
        const ChannelInfo &channel = *entry.channel;
        const ReadPlan &plan = readPlans[idx];
        CHK_PTR_NULL(channel.remoteCclMem.addr);
        CHK_PTR_NULL(plan.localAddr);
        if (!IsBufferRangeValid(channel.remoteCclMem, plan.remoteOffset, bytes)) {
            HCCL_ERROR("[ReadWithEntries] remote buffer is too small, peerRank[%u], remoteOffset[%llu], "
                       "bytes[%llu], bufferSize[%llu]",
                channel.remoteRank, static_cast<unsigned long long>(plan.remoteOffset), static_cast<unsigned long long>(bytes),
                static_cast<unsigned long long>(channel.remoteCclMem.size));
            return HCCL_E_MEMORY;
        }

        const void *remoteAddr = PtrAddConst(channel.remoteCclMem.addr, plan.remoteOffset);
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(entry.thread, channel.handle, NOTIFY_IDX_ACK)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(entry.thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(
            HcommReadOnThread(entry.thread, channel.handle, plan.localAddr, remoteAddr, bytes)));
        CHK_RET(static_cast<HcclResult>(HcommChannelFenceOnThread(entry.thread, channel.handle)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(entry.thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(entry.thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    }
    CHK_RET(ThreadSyncAfter(syncThreads));
    return HCCL_SUCCESS;
}

HcclResult ReadReadyVariableWithEntries(const std::vector<ExchangeEntry> &entries,
    const std::vector<SyncThread> &syncThreads, const std::vector<VariableReadPlan> &readPlans,
    bool waitForPeerReady)
{
    if (entries.empty()) {
        return HCCL_SUCCESS;
    }
    if (readPlans.size() != entries.size()) {
        HCCL_ERROR("[ReadReadyVariableWithEntries] read plan size[%llu] does not match entries[%llu]",
            static_cast<unsigned long long>(readPlans.size()), static_cast<unsigned long long>(entries.size()));
        return HCCL_E_PARA;
    }

    CHK_RET(ThreadSyncBefore(syncThreads));
    for (size_t idx = 0; idx < entries.size(); idx++) {
        const ExchangeEntry &entry = entries[idx];
        const ChannelInfo &channel = *entry.channel;
        const VariableReadPlan &plan = readPlans[idx];
        CHK_PTR_NULL(channel.remoteCclMem.addr);
        CHK_PTR_NULL(plan.localAddr);
        if (!IsBufferRangeValid(channel.remoteCclMem, plan.remoteOffset, plan.bytes)) {
            HCCL_ERROR("[ReadReadyVariableWithEntries] remote buffer is too small, peerRank[%u], "
                       "remoteOffset[%llu], bytes[%llu], bufferSize[%llu]",
                channel.remoteRank, static_cast<unsigned long long>(plan.remoteOffset),
                static_cast<unsigned long long>(plan.bytes),
                static_cast<unsigned long long>(channel.remoteCclMem.size));
            return HCCL_E_MEMORY;
        }

        const void *remoteAddr = PtrAddConst(channel.remoteCclMem.addr, plan.remoteOffset);
        if (waitForPeerReady) {
            CHK_RET(static_cast<HcclResult>(
                HcommChannelNotifyRecordOnThread(entry.thread, channel.handle, NOTIFY_IDX_ACK)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
                entry.thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        }
        CHK_RET(static_cast<HcclResult>(
            HcommReadOnThread(entry.thread, channel.handle, plan.localAddr, remoteAddr, plan.bytes)));
        CHK_RET(static_cast<HcclResult>(HcommChannelFenceOnThread(entry.thread, channel.handle)));
    }
    CHK_RET(ThreadSyncAfter(syncThreads));
    return HCCL_SUCCESS;
}

HcclResult WriteReduceTwoLaneWithEntries(const std::vector<ExchangeEntry> &entries,
    const std::vector<SyncThread> &syncThreads, const std::vector<VariableWriteReducePlan> &firstPlans,
    const std::vector<VariableWriteReducePlan> &secondPlans, bool initializeSecondLane,
    uint32_t dataTypeSize, HcommDataType dataType, HcommReduceOp reduceOp)
{
    if (entries.empty()) {
        return HCCL_SUCCESS;
    }
    if (firstPlans.size() != entries.size() || secondPlans.size() != entries.size()) {
        HCCL_ERROR("[WriteReduceTwoLaneWithEntries] plan size does not match entries[%llu]",
            static_cast<unsigned long long>(entries.size()));
        return HCCL_E_PARA;
    }

    CHK_RET(ThreadSyncBefore(syncThreads));
    for (size_t idx = 0; idx < entries.size(); idx++) {
        const ExchangeEntry &entry = entries[idx];
        const ChannelInfo &channel = *entry.channel;
        const VariableWriteReducePlan &first = firstPlans[idx];
        const VariableWriteReducePlan &second = secondPlans[idx];
        CHK_PTR_NULL(channel.remoteCclMem.addr);
        CHK_PTR_NULL(first.localAddr);
        CHK_PTR_NULL(second.localAddr);
        if ((dataTypeSize != 0 && first.count > std::numeric_limits<uint64_t>::max() / dataTypeSize) ||
            (dataTypeSize != 0 && second.count > std::numeric_limits<uint64_t>::max() / dataTypeSize)) {
            return HCCL_E_PARA;
        }
        const uint64_t firstBytes = first.count * dataTypeSize;
        const uint64_t secondBytes = second.count * dataTypeSize;
        if (!IsBufferRangeValid(channel.remoteCclMem, first.remoteOffset, firstBytes) ||
            !IsBufferRangeValid(channel.remoteCclMem, second.remoteOffset, secondBytes)) {
            HCCL_ERROR("[WriteReduceTwoLaneWithEntries] remote buffer is too small, peerRank[%u]",
                channel.remoteRank);
            return HCCL_E_MEMORY;
        }

        if (first.count != 0) {
            void *remoteAddr = PtrAdd(channel.remoteCclMem.addr, first.remoteOffset);
            CHK_RET(static_cast<HcclResult>(HcommWriteReduceOnThread(
                entry.thread, channel.handle, remoteAddr, first.localAddr, first.count, dataType, reduceOp)));
        }
        if (second.count != 0) {
            void *remoteAddr = PtrAdd(channel.remoteCclMem.addr, second.remoteOffset);
            if (initializeSecondLane) {
                CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(
                    entry.thread, channel.handle, remoteAddr, second.localAddr, secondBytes)));
            } else {
                CHK_RET(static_cast<HcclResult>(HcommWriteReduceOnThread(
                    entry.thread, channel.handle, remoteAddr, second.localAddr, second.count, dataType, reduceOp)));
            }
        }
        CHK_RET(static_cast<HcclResult>(HcommChannelFenceOnThread(entry.thread, channel.handle)));
    }
    CHK_RET(ThreadSyncAfter(syncThreads));
    return HCCL_SUCCESS;
}

HcclResult ReadReduceWithEntries(const std::vector<ExchangeEntry> &entries,
    const std::vector<SyncThread> &syncThreads, const std::vector<uint64_t> &remoteOffsets, void *dstAddr,
    uint64_t count, uint32_t dataTypeSize, HcommDataType dataType, HcommReduceOp reduceOp)
{
    if (entries.empty()) {
        return HCCL_SUCCESS;
    }
    if (remoteOffsets.size() != entries.size()) {
        HCCL_ERROR("[ReadReduceWithEntries] remote offset size[%llu] does not match entries[%llu]",
            static_cast<unsigned long long>(remoteOffsets.size()), static_cast<unsigned long long>(entries.size()));
        return HCCL_E_PARA;
    }

    const uint64_t bytes = count * dataTypeSize;
    CHK_PTR_NULL(dstAddr);
    CHK_RET(ThreadSyncBefore(syncThreads));
    for (size_t idx = 0; idx < entries.size(); idx++) {
        const ExchangeEntry &entry = entries[idx];
        const ChannelInfo &channel = *entry.channel;
        const uint64_t remoteOffset = remoteOffsets[idx];
        CHK_PTR_NULL(channel.remoteCclMem.addr);
        if (!IsBufferRangeValid(channel.remoteCclMem, remoteOffset, bytes)) {
            HCCL_ERROR("[ReadReduceWithEntries] remote buffer is too small, peerRank[%u], remoteOffset[%llu], "
                       "bytes[%llu], bufferSize[%llu]",
                channel.remoteRank, static_cast<unsigned long long>(remoteOffset),
                static_cast<unsigned long long>(bytes), static_cast<unsigned long long>(channel.remoteCclMem.size));
            return HCCL_E_MEMORY;
        }

        const void *remoteAddr = PtrAddConst(channel.remoteCclMem.addr, remoteOffset);
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(entry.thread, channel.handle, NOTIFY_IDX_ACK)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(entry.thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(
            HcommReadReduceOnThread(entry.thread, channel.handle, dstAddr, remoteAddr, count, dataType, reduceOp)));
        CHK_RET(static_cast<HcclResult>(HcommChannelFenceOnThread(entry.thread, channel.handle)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(entry.thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(entry.thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    }
    CHK_RET(ThreadSyncAfter(syncThreads));
    return HCCL_SUCCESS;
}

HcclResult ReduceSlotsToSlot(ThreadHandle thread, void *buffer, uint32_t slotCount, uint32_t dstSlotIdx,
    uint64_t sliceBytes, uint64_t sliceCount, HcommDataType dataType, HcommReduceOp reduceOp)
{
    void *dstSlot = PtrAdd(buffer, static_cast<uint64_t>(dstSlotIdx) * sliceBytes);
    for (uint32_t slotIdx = 0; slotIdx < slotCount; slotIdx++) {
        if (slotIdx == dstSlotIdx) {
            continue;
        }
        const void *srcSlot = PtrAddConst(buffer, static_cast<uint64_t>(slotIdx) * sliceBytes);
        CHK_RET(static_cast<HcclResult>(
            HcommLocalReduceOnThread(thread, dstSlot, srcSlot, sliceCount, dataType, reduceOp)));
    }
    return HCCL_SUCCESS;
}

HcclResult PrepareTailSlots(ThreadHandle thread, const OpParam &param, const AlgResourceCtx &resCtx,
    const TailSlotLayout &layout, uint64_t userOffset, uint64_t tailBytes)
{
    const void *inputTail = PtrAddConst(param.inputPtr, userOffset);
    void *inputSlot = PtrAdd(resCtx.localBuffer.addr, layout.inputOffset);
    void *resultSlot = PtrAdd(resCtx.localBuffer.addr, layout.resultOffset);

    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, inputSlot, inputTail, tailBytes)));
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, resultSlot, inputTail, tailBytes)));
    return HCCL_SUCCESS;
}

HcclResult RunTwoServerTailReadReduce(const OpParam &param, const AlgResourceCtx &resCtx,
    const std::vector<ExchangeEntry> &localEntries, const std::vector<SyncThread> &localSyncThreads,
    const std::vector<ExchangeEntry> &crossEntries, const std::vector<SyncThread> &crossSyncThreads,
    const TailSlotLayout &layout, uint64_t processedCount, uint64_t tailCount, uint32_t dataTypeSize,
    HcommDataType hcommDataType, HcommReduceOp hcommReduceOp)
{
    const ThreadHandle thread = resCtx.threads[0];
    const uint64_t userOffset = processedCount * dataTypeSize;
    const uint64_t tailBytes = tailCount * dataTypeSize;
    void *resultSlot = PtrAdd(resCtx.localBuffer.addr, layout.resultOffset);
    void *outputTail = PtrAdd(param.outputPtr, userOffset);

    CHK_RET(PrepareTailSlots(thread, param, resCtx, layout, userOffset, tailBytes));

    std::vector<uint64_t> localRemoteOffsets;
    localRemoteOffsets.reserve(localEntries.size());
    for (const ExchangeEntry &entry : localEntries) {
        TailSlotLayout remoteLayout;
        if (!BuildTailSlotLayout(entry.channel->remoteCclMem, tailBytes, remoteLayout)) {
            HCCL_ERROR("[RunTwoServerTailReadReduce] remote tail buffer is too small, peerRank[%u]",
                entry.channel->remoteRank);
            return HCCL_E_MEMORY;
        }
        localRemoteOffsets.push_back(remoteLayout.inputOffset);
    }
    CHK_RET(ReadReduceWithEntries(localEntries, localSyncThreads, localRemoteOffsets, resultSlot, tailCount,
        dataTypeSize, hcommDataType, hcommReduceOp));

    // Cross-server peers read this immutable local partial while resultSlot is reduced in place.
    void *inputSlot = PtrAdd(resCtx.localBuffer.addr, layout.inputOffset);
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, inputSlot, resultSlot, tailBytes)));

    std::vector<uint64_t> crossRemoteOffsets;
    crossRemoteOffsets.reserve(crossEntries.size());
    for (const ExchangeEntry &entry : crossEntries) {
        TailSlotLayout remoteLayout;
        if (!BuildTailSlotLayout(entry.channel->remoteCclMem, tailBytes, remoteLayout)) {
            HCCL_ERROR("[RunTwoServerTailReadReduce] remote tail buffer is too small, peerRank[%u]",
                entry.channel->remoteRank);
            return HCCL_E_MEMORY;
        }
        crossRemoteOffsets.push_back(remoteLayout.inputOffset);
    }
    CHK_RET(ReadReduceWithEntries(crossEntries, crossSyncThreads, crossRemoteOffsets, resultSlot, tailCount,
        dataTypeSize, hcommDataType, hcommReduceOp));

    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, outputTail, resultSlot, tailBytes)));
    return HCCL_SUCCESS;
}

HcclResult RunFlatAllGatherReduce(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t maxSliceCount,
    uint32_t dataTypeSize, HcommDataType hcommDataType, HcommReduceOp hcommReduceOp)
{
    const ThreadHandle thread = resCtx.threads[0];
    const std::vector<uint32_t> allPeers = GetAllPeers(param);
    std::vector<ExchangeEntry> peerEntries;
    std::vector<SyncThread> peerSyncThreads;
    CHK_RET(BuildExchangeEntries(resCtx, allPeers, peerEntries, peerSyncThreads));

    const uint64_t localBaseOffset = GetAlignedBufferOffset(resCtx.localBuffer, HCCL_MIN_SLICE_ALIGN);
    uint64_t processedCount = 0;
    while (processedCount < param.count) {
        const uint64_t remainingCount = param.count - processedCount;
        const uint64_t sliceCount = std::min(maxSliceCount, remainingCount);
        const uint64_t sliceBytes = sliceCount * dataTypeSize;
        const uint64_t sliceSlotStride = AlignUp(sliceBytes, HCCL_MIN_SLICE_ALIGN);
        const uint64_t userOffset = processedCount * dataTypeSize;
        void *localSlot =
            PtrAdd(resCtx.localBuffer.addr, localBaseOffset + static_cast<uint64_t>(param.myRank) * sliceSlotStride);
        const void *inputChunk = PtrAddConst(param.inputPtr, userOffset);
        void *outputChunk = PtrAdd(param.outputPtr, userOffset);

        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, localSlot, inputChunk, sliceBytes)));

        std::vector<ReadPlan> peerReadPlans;
        peerReadPlans.reserve(peerEntries.size());
        for (const ExchangeEntry &entry : peerEntries) {
            const uint64_t localSlotOffset =
                localBaseOffset + static_cast<uint64_t>(entry.channel->remoteRank) * sliceSlotStride;
            const uint64_t remoteSlotOffset =
                GetAlignedBufferOffset(entry.channel->remoteCclMem, HCCL_MIN_SLICE_ALIGN) +
                static_cast<uint64_t>(entry.channel->remoteRank) * sliceSlotStride;
            peerReadPlans.push_back(ReadPlan{remoteSlotOffset, PtrAdd(resCtx.localBuffer.addr, localSlotOffset)});
        }
        CHK_RET(ReadWithEntries(peerEntries, peerSyncThreads, peerReadPlans, sliceBytes));

        void *selfSlot =
            PtrAdd(resCtx.localBuffer.addr, localBaseOffset + static_cast<uint64_t>(param.myRank) * sliceSlotStride);
        CHK_RET(ReduceSlotsToSlot(thread, PtrAdd(resCtx.localBuffer.addr, localBaseOffset), param.rankSize,
            param.myRank, sliceSlotStride, sliceCount, hcommDataType, hcommReduceOp));
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, outputChunk, selfSlot, sliceBytes)));

        processedCount += sliceCount;
    }
    return HCCL_SUCCESS;
}

HcclResult RunTwoServerAllReduce(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t maxSliceCount,
    uint32_t dataTypeSize, HcommDataType hcommDataType, HcommReduceOp hcommReduceOp)
{
    const ThreadHandle thread = resCtx.threads[0];
    const uint32_t localSlotIdx = GetLocalRank(param.myRank);
    const uint32_t crossSlotIdx = RANKS_PER_SERVER;
    const std::vector<uint32_t> localPeers = GetLocalServerPeers(param.myRank);
    const std::vector<uint32_t> crossPeer = {GetCrossServerPeer(param.myRank)};
    std::vector<ExchangeEntry> localEntries;
    std::vector<SyncThread> localSyncThreads;
    CHK_RET(BuildExchangeEntries(resCtx, localPeers, localEntries, localSyncThreads));
    std::vector<ExchangeEntry> crossEntries;
    std::vector<SyncThread> crossSyncThreads;
    CHK_RET(BuildExchangeEntries(resCtx, crossPeer, crossEntries, crossSyncThreads));

    const uint64_t localBaseOffset = GetAlignedBufferOffset(resCtx.localBuffer, HCCL_MIN_SLICE_ALIGN);
    uint64_t processedCount = 0;
    while (processedCount < param.count) {
        const uint64_t remainingCount = param.count - processedCount;
        const uint64_t sliceCount = std::min(maxSliceCount, remainingCount);
        const uint64_t sliceBytes = sliceCount * dataTypeSize;
        const uint64_t sliceSlotStride = AlignUp(sliceBytes, HCCL_MIN_SLICE_ALIGN);
        const uint64_t userOffset = processedCount * dataTypeSize;
        TailSlotLayout tailLayout;
        if (ShouldUseTailReadReduce(remainingCount, maxSliceCount, dataTypeSize, resCtx.localBuffer,
                tailLayout)) {
            CHK_RET(RunTwoServerTailReadReduce(param, resCtx, localEntries, localSyncThreads, crossEntries,
                crossSyncThreads, tailLayout, processedCount, sliceCount, dataTypeSize, hcommDataType, hcommReduceOp));
            processedCount += sliceCount;
            continue;
        }

        const void *inputChunk = PtrAddConst(param.inputPtr, userOffset);
        void *outputChunk = PtrAdd(param.outputPtr, userOffset);
        void *localSlot = PtrAdd(resCtx.localBuffer.addr, localBaseOffset + static_cast<uint64_t>(localSlotIdx) *
            sliceSlotStride);

        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, localSlot, inputChunk, sliceBytes)));

        // Server内先得到8卡局部归约结果，避免后续跨Server all-to-all。
        std::vector<ReadPlan> localReadPlans;
        localReadPlans.reserve(localEntries.size());
        for (const ExchangeEntry &entry : localEntries) {
            const uint64_t rankSlotOffset =
                localBaseOffset + static_cast<uint64_t>(GetLocalRank(entry.channel->remoteRank)) * sliceSlotStride;
            const uint64_t remoteRankSlotOffset =
                GetAlignedBufferOffset(entry.channel->remoteCclMem, HCCL_MIN_SLICE_ALIGN) +
                static_cast<uint64_t>(GetLocalRank(entry.channel->remoteRank)) * sliceSlotStride;
            localReadPlans.push_back(ReadPlan{remoteRankSlotOffset, PtrAdd(resCtx.localBuffer.addr, rankSlotOffset)});
        }
        CHK_RET(ReadWithEntries(localEntries, localSyncThreads, localReadPlans, sliceBytes));
        CHK_RET(ReduceSlotsToSlot(thread, PtrAdd(resCtx.localBuffer.addr, localBaseOffset), RANKS_PER_SERVER,
            localSlotIdx, sliceSlotStride, sliceCount, hcommDataType, hcommReduceOp));

        // 每个rank只和另一台Server的同local-id rank交换一次局部结果。
        std::vector<ReadPlan> crossReadPlans;
        crossReadPlans.reserve(crossEntries.size());
        for (const ExchangeEntry &entry : crossEntries) {
            const uint64_t crossRemoteBaseOffset =
                GetAlignedBufferOffset(entry.channel->remoteCclMem, HCCL_MIN_SLICE_ALIGN);
            crossReadPlans.push_back(ReadPlan{crossRemoteBaseOffset + static_cast<uint64_t>(localSlotIdx) *
                sliceSlotStride, PtrAdd(resCtx.localBuffer.addr, localBaseOffset + static_cast<uint64_t>(crossSlotIdx) *
                sliceSlotStride)});
        }
        CHK_RET(ReadWithEntries(crossEntries, crossSyncThreads, crossReadPlans, sliceBytes));
        const void *crossSlot =
            PtrAddConst(resCtx.localBuffer.addr, localBaseOffset + static_cast<uint64_t>(crossSlotIdx) *
                sliceSlotStride);
        CHK_RET(static_cast<HcclResult>(
            HcommLocalReduceOnThread(thread, localSlot, crossSlot, sliceCount, hcommDataType, hcommReduceOp)));

        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, outputChunk, localSlot, sliceBytes)));
        processedCount += sliceCount;
    }
    return HCCL_SUCCESS;
}

HcclResult RunSmallPairReadReduce(ThreadHandle thread, const ChannelInfo &channel, void *localDst,
    const void *remoteSrc, uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp)
{
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(
        HcommReadReduceOnThread(thread, channel.handle, localDst, remoteSrc, count, dataType, reduceOp)));
    CHK_RET(static_cast<HcclResult>(HcommChannelFenceOnThread(thread, channel.handle)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL)));
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
        thread, channel.handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult RunSmallHierarchicalAllReduce(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t totalBytes,
    HcommDataType hcommDataType, HcommReduceOp hcommReduceOp)
{
    const ThreadHandle thread = resCtx.threads[0];
    const uint32_t localRank = GetLocalRank(param.myRank);
    const uint32_t crossServerStart = (param.myRank / RANKS_PER_SERVER ^ 1U) * RANKS_PER_SERVER;
    const uint64_t slotStride = AlignUp(totalBytes, HCCL_MIN_SLICE_ALIGN);
    const uint64_t localBaseOffset = GetAlignedBufferOffset(resCtx.localBuffer, HCCL_MIN_SLICE_ALIGN);
    if (!IsBufferRangeValid(resCtx.localBuffer, localBaseOffset, slotStride * 2)) {
        HCCL_ERROR("[RunSmallHierarchicalAllReduce] local buffer is too small, offset[%llu], bytes[%llu], size[%llu]",
            static_cast<unsigned long long>(localBaseOffset), static_cast<unsigned long long>(slotStride * 2),
            static_cast<unsigned long long>(resCtx.localBuffer.size));
        return HCCL_E_MEMORY;
    }
    for (const ChannelInfo &channel : resCtx.channels) {
        CHK_PTR_NULL(channel.remoteCclMem.addr);
        const uint64_t remoteBaseOffset = GetAlignedBufferOffset(channel.remoteCclMem, HCCL_MIN_SLICE_ALIGN);
        if (!IsBufferRangeValid(channel.remoteCclMem, remoteBaseOffset, slotStride * 2)) {
            HCCL_ERROR("[RunSmallHierarchicalAllReduce] remote buffer is too small, peerRank[%u], offset[%llu], "
                       "bytes[%llu], size[%llu]",
                channel.remoteRank, static_cast<unsigned long long>(remoteBaseOffset),
                static_cast<unsigned long long>(slotStride * 2),
                static_cast<unsigned long long>(channel.remoteCclMem.size));
            return HCCL_E_MEMORY;
        }
    }

    void *slots[2] = {PtrAdd(resCtx.localBuffer.addr, localBaseOffset),
        PtrAdd(resCtx.localBuffer.addr, localBaseOffset + slotStride)};
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, slots[0], param.inputPtr, totalBytes)));

    // Parity coloring maps all four logical hypercube dimensions to the wider Clos. Full-message
    // recursive doubling cuts the serial communication depth from eight rounds to four; ping-pong slots
    // keep every remote source immutable until both peers have completed the current read-reduce.
    constexpr uint32_t masks[] = {RANKS_PER_SERVER, 4, 2, 1};
    uint32_t currentSlotIdx = 0;
    for (uint32_t mask : masks) {
        const uint32_t peerRank = crossServerStart + (localRank ^ (mask & (RANKS_PER_SERVER - 1)));
        size_t channelIdx = 0;
        const ChannelInfo *channel = FindChannel(resCtx, peerRank, channelIdx);
        if (channel == nullptr) {
            HCCL_ERROR("[RunSmallHierarchicalAllReduce] no channel to recursive-doubling peer rank[%u]", peerRank);
            return HCCL_E_PARA;
        }

        const uint32_t nextSlotIdx = currentSlotIdx ^ 1U;
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(thread, slots[nextSlotIdx], slots[currentSlotIdx], totalBytes)));
        const uint64_t remoteBaseOffset = GetAlignedBufferOffset(channel->remoteCclMem, HCCL_MIN_SLICE_ALIGN);
        const void *remoteSrc = PtrAddConst(
            channel->remoteCclMem.addr, remoteBaseOffset + static_cast<uint64_t>(currentSlotIdx) * slotStride);
        CHK_RET(RunSmallPairReadReduce(thread, *channel, slots[nextSlotIdx], remoteSrc, param.count,
            hcommDataType, hcommReduceOp));
        currentSlotIdx = nextSlotIdx;
    }

    CHK_RET(static_cast<HcclResult>(
        HcommLocalCopyOnThread(thread, param.outputPtr, slots[currentSlotIdx], totalBytes)));
    return HCCL_SUCCESS;
}

HcclResult RunStripedMeshReduceScatter(const OpParam &param, const AlgResourceCtx &resCtx,
    const std::vector<ExchangeEntry> &localEntries, const std::vector<SyncThread> &localSyncThreads,
    const void *inputBase, uint64_t groupCount, uint64_t firstLaneRelativeOffset,
    uint64_t secondLaneRelativeOffset, bool useNaturalChunkOffset, uint32_t dataTypeSize,
    HcommDataType hcommDataType, HcommReduceOp hcommReduceOp, void *&localAggregate)
{
    const ThreadHandle thread = resCtx.threads[0];
    const uint32_t localRank = GetLocalRank(param.myRank);
    const ChunkRange chunk = GetChunkRange(groupCount, localRank);
    if (chunk.count == 0) {
        return HCCL_E_PARA;
    }
    const uint64_t localBaseOffset = GetAlignedBufferOffset(resCtx.localBuffer, HCCL_MIN_SLICE_ALIGN);
    const uint64_t aggregateRelativeOffset = firstLaneRelativeOffset +
        (useNaturalChunkOffset ? chunk.offsetCount * dataTypeSize : 0);
    const uint64_t secondAggregateRelativeOffset = secondLaneRelativeOffset +
        (useNaturalChunkOffset ? chunk.offsetCount * dataTypeSize : 0);
    const uint64_t chunkBytes = chunk.count * dataTypeSize;
    if (!IsBufferRangeValid(resCtx.localBuffer, localBaseOffset + aggregateRelativeOffset, chunkBytes) ||
        !IsBufferRangeValid(resCtx.localBuffer, localBaseOffset + secondAggregateRelativeOffset, chunkBytes)) {
        return HCCL_E_MEMORY;
    }
    localAggregate = PtrAdd(resCtx.localBuffer.addr, localBaseOffset + aggregateRelativeOffset);
    void *secondAggregate = PtrAdd(resCtx.localBuffer.addr, localBaseOffset + secondAggregateRelativeOffset);
    const void *selfContribution = PtrAddConst(inputBase, chunk.offsetCount * dataTypeSize);
    if (localAggregate != selfContribution) {
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(thread, localAggregate, selfContribution, chunkBytes)));
    }
    CHK_RET(ServerBarrier(param, resCtx));

    for (uint32_t round = 0; round < LARGE_HIERARCHICAL_MESH_ROUND_NUM; round++) {
        const uint32_t firstStripeRound = round * LARGE_HIERARCHICAL_MESH_LANE_NUM;
        const uint32_t secondStripeRound = firstStripeRound + 1;
        std::vector<VariableWriteReducePlan> firstPlans;
        std::vector<VariableWriteReducePlan> secondPlans;
        firstPlans.reserve(localEntries.size());
        secondPlans.reserve(localEntries.size());
        for (const ExchangeEntry &entry : localEntries) {
            const uint32_t peerLocalRank = GetLocalRank(entry.channel->remoteRank);
            const ChunkRange peerChunk = GetChunkRange(groupCount, peerLocalRank);
            const uint32_t packedSourceIdx = GetPackedContributionIndex(localRank, peerLocalRank);
            const uint32_t firstStripeIdx =
                (packedSourceIdx + firstStripeRound) % LARGE_HIERARCHICAL_STRIPE_NUM;
            const ChunkRange firstStripe = GetStripeRange(peerChunk.count, firstStripeIdx);
            const uint64_t remoteBaseOffset =
                GetAlignedBufferOffset(entry.channel->remoteCclMem, HCCL_MIN_SLICE_ALIGN);
            const uint64_t firstRemoteAggregateOffset = remoteBaseOffset + firstLaneRelativeOffset +
                (useNaturalChunkOffset ? peerChunk.offsetCount * dataTypeSize : 0);
            const uint64_t firstLocalStripeOffset =
                (peerChunk.offsetCount + firstStripe.offsetCount) * dataTypeSize;
            firstPlans.push_back(VariableWriteReducePlan{
                firstRemoteAggregateOffset + firstStripe.offsetCount * dataTypeSize,
                PtrAddConst(inputBase, firstLocalStripeOffset), firstStripe.count});

            if (secondStripeRound < LARGE_HIERARCHICAL_STRIPE_NUM) {
                const uint32_t secondStripeIdx =
                    (packedSourceIdx + secondStripeRound) % LARGE_HIERARCHICAL_STRIPE_NUM;
                const ChunkRange secondStripe = GetStripeRange(peerChunk.count, secondStripeIdx);
                const uint64_t secondRemoteAggregateOffset = remoteBaseOffset + secondLaneRelativeOffset +
                    (useNaturalChunkOffset ? peerChunk.offsetCount * dataTypeSize : 0);
                const uint64_t secondLocalStripeOffset =
                    (peerChunk.offsetCount + secondStripe.offsetCount) * dataTypeSize;
                secondPlans.push_back(VariableWriteReducePlan{
                    secondRemoteAggregateOffset + secondStripe.offsetCount * dataTypeSize,
                    PtrAddConst(inputBase, secondLocalStripeOffset), secondStripe.count});
            } else {
                secondPlans.push_back(VariableWriteReducePlan{remoteBaseOffset,
                    PtrAddConst(inputBase, peerChunk.offsetCount * dataTypeSize), 0});
            }
        }
        CHK_RET(WriteReduceTwoLaneWithEntries(localEntries, localSyncThreads, firstPlans, secondPlans,
            round == 0, dataTypeSize, hcommDataType, hcommReduceOp));
        CHK_RET(ServerBarrier(param, resCtx));
    }
    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
        thread, localAggregate, secondAggregate, chunk.count, hcommDataType, hcommReduceOp)));
    CHK_RET(ServerBarrier(param, resCtx));
    return HCCL_SUCCESS;
}

HcclResult RunHierarchicalTwoServerAllReduce(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t maxSliceCount, uint32_t dataTypeSize, HcommDataType hcommDataType, HcommReduceOp hcommReduceOp)
{
    (void)maxSliceCount;
    const ThreadHandle thread = resCtx.threads[0];
    const uint32_t localRank = GetLocalRank(param.myRank);
    const uint32_t serverIdx = param.myRank / RANKS_PER_SERVER;
    const std::vector<uint32_t> localPeers = GetLocalServerPeers(param.myRank);
    const std::vector<uint32_t> crossPeer = {GetCrossServerPeer(param.myRank)};
    std::vector<ExchangeEntry> localEntries;
    std::vector<SyncThread> localSyncThreads;
    CHK_RET(BuildWorkerExchangeEntries(resCtx, localPeers, localEntries, localSyncThreads));
    std::vector<ExchangeEntry> crossEntries;
    std::vector<SyncThread> crossSyncThreads;
    CHK_RET(BuildWorkerExchangeEntries(resCtx, crossPeer, crossEntries, crossSyncThreads));
    if (crossEntries.size() != 1) {
        return HCCL_E_PARA;
    }
    CHK_RET(AssignUnusedWorkerThread(resCtx, localEntries, crossEntries[0], crossSyncThreads));

    const uint64_t alignCount = std::max<uint64_t>(1, HCCL_MIN_SLICE_ALIGN / dataTypeSize);
    uint64_t firstCount = AlignDown(param.count / 3, alignCount);
    if (firstCount == 0 || firstCount >= param.count) {
        firstCount = param.count / 3;
    }
    const uint64_t secondCount = param.count - firstCount;
    const uint64_t firstBytes = firstCount * dataTypeSize;
    const ChunkRange firstMaxChunk = GetChunkRange(firstCount, 0);
    const uint64_t firstStride = AlignUp(firstMaxChunk.count * dataTypeSize, HCCL_MIN_SLICE_ALIGN);
    const ChunkRange secondServer0 = GetPartitionRange(secondCount, 2, 0);
    const ChunkRange secondServer1 = GetPartitionRange(secondCount, 2, 1);
    const uint64_t secondSourceStride = AlignUp(
        std::max(secondServer0.count, secondServer1.count) * dataTypeSize, HCCL_MIN_SLICE_ALIGN);
    const ChunkRange secondMaxChunk =
        GetChunkRange(std::max(secondServer0.count, secondServer1.count), 0);
    const uint64_t secondAggregateStride =
        AlignUp(secondMaxChunk.count * dataTypeSize, HCCL_MIN_SLICE_ALIGN);
    const uint64_t firstSecondLaneRelative = firstStride;
    const uint64_t secondSourceRelative = firstSecondLaneRelative + firstStride;
    const uint64_t secondAggregateRelative = secondSourceRelative + secondSourceStride;
    const uint64_t secondAggregateSecondLaneRelative = secondAggregateRelative + secondAggregateStride;
    const uint64_t requiredBytes = secondAggregateSecondLaneRelative + secondAggregateStride;
    const uint64_t localBaseOffset = GetAlignedBufferOffset(resCtx.localBuffer, HCCL_MIN_SLICE_ALIGN);
    if (!IsBufferRangeValid(resCtx.localBuffer, localBaseOffset, requiredBytes)) {
        HCCL_ERROR("[RunHierarchicalTwoServerAllReduce] workspace is too small, requiredBytes[%llu], size[%llu]",
            static_cast<unsigned long long>(requiredBytes),
            static_cast<unsigned long long>(resCtx.localBuffer.size));
        return HCCL_E_MEMORY;
    }
    for (const ChannelInfo &channel : resCtx.channels) {
        const uint64_t remoteBaseOffset = GetAlignedBufferOffset(channel.remoteCclMem, HCCL_MIN_SLICE_ALIGN);
        if (!IsBufferRangeValid(channel.remoteCclMem, remoteBaseOffset, requiredBytes)) {
            return HCCL_E_MEMORY;
        }
    }

    const ChunkRange mySecondServer = GetPartitionRange(secondCount, 2, serverIdx);
    const ChunkRange peerSecondServer = GetPartitionRange(secondCount, 2, 1 - serverIdx);
    void *secondSource = PtrAdd(resCtx.localBuffer.addr, localBaseOffset + secondSourceRelative);
    const void *mySecondInput = PtrAddConst(
        param.inputPtr, firstBytes + mySecondServer.offsetCount * dataTypeSize);
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
        thread, secondSource, mySecondInput, mySecondServer.count * dataTypeSize)));

    const uint64_t crossRemoteBase =
        GetAlignedBufferOffset(crossEntries[0].channel->remoteCclMem, HCCL_MIN_SLICE_ALIGN);

    // Phase 1: Mesh-RS(first axis) runs in parallel with Clos-RS(second axis).
    CHK_RET(ThreadSyncBefore(crossSyncThreads));
    const void *peerSecondInput = PtrAddConst(
        param.inputPtr, firstBytes + peerSecondServer.offsetCount * dataTypeSize);
    CHK_RET(WriteReduceExchangeAsync(crossEntries[0], crossRemoteBase + secondSourceRelative,
        peerSecondInput, peerSecondServer.count, dataTypeSize, hcommDataType, hcommReduceOp));
    void *firstAggregate = nullptr;
    CHK_RET(RunStripedMeshReduceScatter(param, resCtx, localEntries, localSyncThreads, param.inputPtr,
        firstCount, 0, firstSecondLaneRelative, false, dataTypeSize, hcommDataType, hcommReduceOp,
        firstAggregate));
    CHK_RET(ThreadSyncAfter(crossSyncThreads));

    // Phase 2: Clos-RS(first axis) runs in parallel with Mesh-RS(second axis).
    const ChunkRange firstChunk = GetChunkRange(firstCount, localRank);
    const ChunkRange peerFirstServer = GetPartitionRange(firstChunk.count, 2, 1 - serverIdx);
    CHK_RET(ThreadSyncBefore(crossSyncThreads));
    CHK_RET(WriteReduceExchangeAsync(crossEntries[0],
        crossRemoteBase + peerFirstServer.offsetCount * dataTypeSize,
        PtrAddConst(firstAggregate, peerFirstServer.offsetCount * dataTypeSize), peerFirstServer.count,
        dataTypeSize, hcommDataType, hcommReduceOp));
    void *secondAggregate = nullptr;
    CHK_RET(RunStripedMeshReduceScatter(param, resCtx, localEntries, localSyncThreads, secondSource,
        mySecondServer.count, secondAggregateRelative, secondAggregateSecondLaneRelative, false,
        dataTypeSize, hcommDataType, hcommReduceOp, secondAggregate));
    CHK_RET(ThreadSyncAfter(crossSyncThreads));

    std::vector<ExchangeEntry> phaseEntries = localEntries;
    phaseEntries.push_back(crossEntries[0]);
    std::vector<SyncThread> phaseSyncThreads = localSyncThreads;
    phaseSyncThreads.push_back(crossSyncThreads[1]);

    // Phase 3: Clos-AG(first axis) runs in parallel with Mesh-AG(second axis).
    const ChunkRange secondChunk = GetChunkRange(mySecondServer.count, localRank);
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread,
        PtrAdd(secondSource, secondChunk.offsetCount * dataTypeSize), secondAggregate,
        secondChunk.count * dataTypeSize)));
    std::vector<VariableReadPlan> phase3Plans;
    phase3Plans.reserve(phaseEntries.size());
    for (const ExchangeEntry &entry : phaseEntries) {
        const uint64_t remoteBase =
            GetAlignedBufferOffset(entry.channel->remoteCclMem, HCCL_MIN_SLICE_ALIGN);
        if (entry.channel->remoteRank == crossPeer[0]) {
            phase3Plans.push_back(VariableReadPlan{remoteBase + peerFirstServer.offsetCount * dataTypeSize,
                PtrAdd(firstAggregate, peerFirstServer.offsetCount * dataTypeSize),
                peerFirstServer.count * dataTypeSize});
        } else {
            const uint32_t peerLocalRank = GetLocalRank(entry.channel->remoteRank);
            const ChunkRange peerChunk = GetChunkRange(mySecondServer.count, peerLocalRank);
            const uint64_t chunkOffset = peerChunk.offsetCount * dataTypeSize;
            phase3Plans.push_back(VariableReadPlan{remoteBase + secondAggregateRelative,
                PtrAdd(secondSource, chunkOffset), peerChunk.count * dataTypeSize});
        }
    }
    // Both RS axes have already published immutable CCL ranges. No per-channel rendezvous is needed here.
    CHK_RET(ReadReadyVariableWithEntries(phaseEntries, phaseSyncThreads, phase3Plans, false));

    // Phase 4: Mesh-AG(first axis) runs in parallel with Clos-AG(second axis), directly into user output.
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread,
        PtrAdd(param.outputPtr, firstChunk.offsetCount * dataTypeSize), firstAggregate,
        firstChunk.count * dataTypeSize)));
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread,
        PtrAdd(param.outputPtr, firstBytes + mySecondServer.offsetCount * dataTypeSize), secondSource,
        mySecondServer.count * dataTypeSize)));
    std::vector<VariableReadPlan> phase4Plans;
    phase4Plans.reserve(phaseEntries.size());
    for (const ExchangeEntry &entry : phaseEntries) {
        const uint64_t remoteBase =
            GetAlignedBufferOffset(entry.channel->remoteCclMem, HCCL_MIN_SLICE_ALIGN);
        if (entry.channel->remoteRank == crossPeer[0]) {
            phase4Plans.push_back(VariableReadPlan{remoteBase + secondSourceRelative,
                PtrAdd(param.outputPtr, firstBytes + peerSecondServer.offsetCount * dataTypeSize),
                peerSecondServer.count * dataTypeSize});
        } else {
            const uint32_t peerLocalRank = GetLocalRank(entry.channel->remoteRank);
            const ChunkRange peerChunk = GetChunkRange(firstCount, peerLocalRank);
            phase4Plans.push_back(VariableReadPlan{remoteBase,
                PtrAdd(param.outputPtr, peerChunk.offsetCount * dataTypeSize), peerChunk.count * dataTypeSize});
        }
    }
    // The start ACK publishes phase 3 to every source peer. This is the final phase, so a completion ACK
    // would only delay local return after the fenced reads have already produced the user output.
    CHK_RET(ReadReadyVariableWithEntries(phaseEntries, phaseSyncThreads, phase4Plans, true));
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    HCCL_INFO("Executing AICPU Kernel on Ascend NPU");

    CHK_PTR_NULL(param.inputPtr);
    CHK_PTR_NULL(param.outputPtr);
    CHK_PTR_NULL(resCtx.localBuffer.addr);
    if (resCtx.threads.empty()) {
        HCCL_ERROR("[ExecOp] no AICPU thread resource");
        return HCCL_E_PARA;
    }
    if (param.rankSize == 0 || param.myRank >= param.rankSize) {
        HCCL_ERROR("[ExecOp] invalid rank info, myRank[%u], rankSize[%u]", param.myRank, param.rankSize);
        return HCCL_E_PARA;
    }

    uint32_t dataTypeSize = 0;
    if (!GetDataTypeSize(param.dataType, dataTypeSize)) {
        HCCL_ERROR("[ExecOp] unsupported dataType[%d]", param.dataType);
        return HCCL_E_NOT_SUPPORT;
    }
    if (dataTypeSize != 0 && param.count > std::numeric_limits<uint64_t>::max() / dataTypeSize) {
        HCCL_ERROR("[ExecOp] count[%llu] overflows byte size", static_cast<unsigned long long>(param.count));
        return HCCL_E_PARA;
    }
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }

    HcommDataType hcommDataType = HCOMM_DATA_TYPE_RESERVED;
    HcommReduceOp hcommReduceOp = HCOMM_REDUCE_RESERVED;
    if (!ToHcommDataType(param.dataType, hcommDataType) || !ToHcommReduceOp(param.reduceType, hcommReduceOp)) {
        HCCL_ERROR("[ExecOp] unsupported dataType[%d] or reduce op[%d]", param.dataType, param.reduceType);
        return HCCL_E_NOT_SUPPORT;
    }

    const ThreadHandle thread = resCtx.threads[0];
    const uint64_t totalBytes = param.count * dataTypeSize;
    if (param.rankSize == 1) {
        if (param.inputPtr != param.outputPtr) {
            CHK_RET(
                static_cast<HcclResult>(HcommLocalCopyOnThread(thread, param.outputPtr, param.inputPtr, totalBytes)));
        }
        return HCCL_SUCCESS;
    }
    if (resCtx.channels.size() < static_cast<size_t>(param.rankSize - 1)) {
        HCCL_ERROR("[ExecOp] channel resource is insufficient, channels[%llu], rankSize[%u]",
            static_cast<unsigned long long>(resCtx.channels.size()), param.rankSize);
        return HCCL_E_PARA;
    }

    if (ShouldUseSmallHierarchicalAlgo(param, totalBytes)) {
        return RunSmallHierarchicalAllReduce(param, resCtx, totalBytes, hcommDataType, hcommReduceOp);
    }

    uint64_t slotNum = param.rankSize;
    if (ShouldUseHierarchicalTwoServerAlgo(param, totalBytes)) {
        slotNum = LARGE_HIERARCHICAL_CCL_SLOT_NUM;
    } else if (ShouldUseTwoServerAlgo(param, totalBytes)) {
        slotNum = TWO_SERVER_SLOT_NUM;
    }
    const uint64_t alignedBaseOffset = GetAlignedBufferOffset(resCtx.localBuffer, HCCL_MIN_SLICE_ALIGN);
    if (alignedBaseOffset >= resCtx.localBuffer.size) {
        HCCL_ERROR("[ExecOp] HCCL buffer cannot satisfy aligned base, bufferSize[%llu], alignedBaseOffset[%llu]",
            static_cast<unsigned long long>(resCtx.localBuffer.size),
            static_cast<unsigned long long>(alignedBaseOffset));
        return HCCL_E_MEMORY;
    }
    uint64_t maxSlotBytes = (resCtx.localBuffer.size - alignedBaseOffset) / slotNum;
    if (maxSlotBytes >= HCCL_MIN_SLICE_ALIGN) {
        maxSlotBytes = AlignDown(maxSlotBytes, HCCL_MIN_SLICE_ALIGN);
    }
    maxSlotBytes = std::min(maxSlotBytes, MAX_SLICE_BYTES);
    uint64_t maxSliceCount = maxSlotBytes / dataTypeSize;
    if (ShouldUseHierarchicalTwoServerAlgo(param, totalBytes)) {
        maxSliceCount *= RANKS_PER_SERVER;
    }
    if (maxSliceCount == 0) {
        HCCL_ERROR("[ExecOp] HCCL buffer is too small, bufferSize[%llu], rankSize[%u], dataTypeSize[%u]",
            static_cast<unsigned long long>(resCtx.localBuffer.size), param.rankSize, dataTypeSize);
        return HCCL_E_MEMORY;
    }

    if (ShouldUseHierarchicalTwoServerAlgo(param, totalBytes)) {
        return RunHierarchicalTwoServerAllReduce(param, resCtx, maxSliceCount, dataTypeSize, hcommDataType,
            hcommReduceOp);
    }
    if (ShouldUseTwoServerAlgo(param, totalBytes)) {
        return RunTwoServerAllReduce(param, resCtx, maxSliceCount, dataTypeSize, hcommDataType, hcommReduceOp);
    }
    return RunFlatAllGatherReduce(param, resCtx, maxSliceCount, dataTypeSize, hcommDataType, hcommReduceOp);
}
} // namespace ops_hccl
