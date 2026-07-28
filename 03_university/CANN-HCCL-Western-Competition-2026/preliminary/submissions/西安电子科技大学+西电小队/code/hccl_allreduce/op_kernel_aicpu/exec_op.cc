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

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace {
constexpr uint32_t COMPETITION_RANK_SIZE = 16;
constexpr uint32_t LOCAL_RANK_SIZE = 8;
constexpr uint64_t FP32_BYTES = sizeof(float);
constexpr uint64_t SMALL_MESSAGE_BYTES = 1024ULL * 1024ULL;
constexpr uint64_t TEST_512_MIB_BYTES = 512ULL * 1024ULL * 1024ULL;
constexpr uint64_t TEST_400_MIB_PLUS_4_BYTES = 400ULL * 1024ULL * 1024ULL + FP32_BYTES;
constexpr uint64_t DIRECT_WINDOW_BYTES = 256ULL * 1024ULL * 1024ULL;
constexpr uint64_t DIRECT_WINDOW_COUNT = DIRECT_WINDOW_BYTES / FP32_BYTES;
constexpr uint64_t DIRECT_MAX_SHARD_BYTES = DIRECT_WINDOW_BYTES / COMPETITION_RANK_SIZE;
constexpr uint64_t DIRECT_FINAL_OFFSET = (COMPETITION_RANK_SIZE - 1) * DIRECT_MAX_SHARD_BYTES;
constexpr uint64_t DIRECT_WORKSPACE_BYTES = DIRECT_FINAL_OFFSET + DIRECT_MAX_SHARD_BYTES;
constexpr uint32_t DIRECT_REDUCE_PARTS = 4;
constexpr uint64_t AGGREGATE_REGION_BYTES = 64ULL * 1024ULL * 1024ULL;
constexpr uint64_t STAGING_REGION_BYTES = 256ULL * 1024ULL * 1024ULL;
constexpr uint64_t WORKSPACE_BYTES = AGGREGATE_REGION_BYTES + STAGING_REGION_BYTES;
constexpr uint32_t NOTIFY_READY = NOTIFY_IDX_ACK;
constexpr uint32_t NOTIFY_DATA = NOTIFY_IDX_DATA_SIGNAL;

struct BufferLayout {
    uint8_t *workspace = nullptr;
    uint8_t *aggregate = nullptr;
    uint8_t *staging = nullptr;
    uint64_t workspaceBytes = 0;
    uint64_t aggregateBytes = 0;
    uint64_t stagingBytes = 0;
};

struct PipelinePlan {
    uint32_t localRank = 0;
    uint32_t server = 0;
    uint64_t firstCount = 0;
    uint64_t secondCount = 0;
    uint64_t secondBase = 0;
    uint64_t firstShardCount = 0;
    uint64_t firstShardBytes = 0;
    uint64_t firstPairBytes = 0;
    uint64_t ownSecondBlockOffset = 0;
    uint64_t ownSecondBlockCount = 0;
    uint64_t ownSecondBlockBytes = 0;
    uint64_t ownSecondShardOffset = 0;
    uint64_t ownSecondShardCount = 0;
    uint64_t ownSecondShardBytes = 0;
    uint64_t phase1InterOffset = 0;
    uint64_t phase2InterOffset = 0;
    uint64_t assemblySecondOffset = 0;
};

struct DirectRankPlan {
    uint32_t myGlobalRank = 0;
    std::array<uint32_t, LOCAL_RANK_SIZE> remoteRanks{};
    std::array<uint32_t, LOCAL_RANK_SIZE> remoteGlobalRanks{};
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

HcclResult GetLocalRankIndex(const AlgResourceCtx &resCtx, uint32_t myRank, uint32_t &localRank)
{
    for (uint32_t i = 0; i < resCtx.localRanks.size(); ++i) {
        if (resCtx.localRanks[i] == myRank) {
            localRank = i;
            return HCCL_SUCCESS;
        }
    }
    HCCL_ERROR("Rank[%u] is absent from the Layer-0 rank list", myRank);
    return HCCL_E_PARA;
}

HcclResult GetGlobalRankIndex(const AlgResourceCtx &resCtx, uint32_t myRank, uint32_t &globalRank)
{
    for (uint32_t i = 0; i < resCtx.globalRanks.size(); ++i) {
        if (resCtx.globalRanks[i] == myRank) {
            globalRank = i;
            return HCCL_SUCCESS;
        }
    }
    HCCL_ERROR("Rank[%u] is absent from the global rank list", myRank);
    return HCCL_E_PARA;
}

bool IsLocalPeer(const AlgResourceCtx &resCtx, uint32_t rank)
{
    return std::find(resCtx.localRanks.begin(), resCtx.localRanks.end(), rank) != resCtx.localRanks.end();
}

HcclResult BuildDirectRankPlan(const OpParam &param, const AlgResourceCtx &resCtx, DirectRankPlan &plan)
{
    CHK_RET(GetGlobalRankIndex(resCtx, param.myRank, plan.myGlobalRank));
    uint32_t localRank = 0;
    CHK_RET(GetLocalRankIndex(resCtx, param.myRank, localRank));

    std::array<uint32_t, LOCAL_RANK_SIZE> remoteRanks{};
    std::array<uint32_t, LOCAL_RANK_SIZE> remoteGlobals{};
    uint32_t remoteNum = 0;
    for (uint32_t globalRank = 0; globalRank < COMPETITION_RANK_SIZE; ++globalRank) {
        const uint32_t rank = resCtx.globalRanks[globalRank];
        if (IsLocalPeer(resCtx, rank)) {
            continue;
        }
        CHK_PRT_RET(remoteNum >= LOCAL_RANK_SIZE,
            HCCL_ERROR("Too many remote ranks"), HCCL_E_PARA);
        remoteRanks[remoteNum] = rank;
        remoteGlobals[remoteNum] = globalRank;
        ++remoteNum;
    }
    CHK_PRT_RET(remoteNum != LOCAL_RANK_SIZE,
        HCCL_ERROR("Expected eight remote ranks, found[%u]", remoteNum), HCCL_E_PARA);

    // The lexicographically first server advances by +round and the other by
    // -round.  Thus every Clos operation has its matching peer in the same
    // round instead of waiting for a row/column wavefront.
    const bool firstServer = resCtx.localRanks.front() < remoteRanks.front();
    for (uint32_t round = 0; round < LOCAL_RANK_SIZE; ++round) {
        const uint32_t remoteIndex = firstServer
            ? (localRank + round) % LOCAL_RANK_SIZE
            : (localRank + LOCAL_RANK_SIZE - round) % LOCAL_RANK_SIZE;
        plan.remoteRanks[round] = remoteRanks[remoteIndex];
        plan.remoteGlobalRanks[round] = remoteGlobals[remoteIndex];
    }
    return HCCL_SUCCESS;
}

uint64_t GetPartitionCount(uint64_t count, uint32_t partition, uint32_t partitionNum)
{
    const uint64_t base = count / partitionNum;
    return base + (partition < count % partitionNum ? 1 : 0);
}

uint64_t GetPartitionOffset(uint64_t count, uint32_t partition, uint32_t partitionNum)
{
    const uint64_t base = count / partitionNum;
    return static_cast<uint64_t>(partition) * base + std::min<uint64_t>(partition, count % partitionNum);
}

uint64_t ScaleCount(uint64_t count, uint64_t numerator, uint64_t denominator)
{
    // Split the multiplication to avoid overflowing when count is large.
    return (count / denominator) * numerator + ((count % denominator) * numerator) / denominator;
}

uint32_t CompactPeerSlot(uint32_t senderLocalRank, uint32_t targetLocalRank)
{
    return senderLocalRank < targetLocalRank ? senderLocalRank : senderLocalRank - 1;
}

HcclResult GetBufferLayout(const AlgResourceCtx &resCtx, BufferLayout &layout)
{
    CHK_PRT_RET(resCtx.localBuffer.addr == nullptr || resCtx.localBuffer.size < WORKSPACE_BYTES,
        HCCL_ERROR("Invalid local HCCL buffer, size[%lu]", resCtx.localBuffer.size), HCCL_E_PARA);

    for (const ChannelInfo &channel : resCtx.channels) {
        CHK_PRT_RET(channel.remoteCclMem.addr == nullptr || channel.remoteCclMem.size < WORKSPACE_BYTES
                || channel.notifyNum < 2,
            HCCL_ERROR("Invalid channel resource, remoteRank[%u], bufferSize[%lu], notifyNum[%u]",
                channel.remoteRank, channel.remoteCclMem.size, channel.notifyNum),
            HCCL_E_PARA);
    }

    layout.workspace = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    layout.workspaceBytes = WORKSPACE_BYTES;
    layout.aggregate = layout.workspace;
    layout.aggregateBytes = AGGREGATE_REGION_BYTES;
    layout.staging = layout.aggregate + layout.aggregateBytes;
    layout.stagingBytes = STAGING_REGION_BYTES;
    return HCCL_SUCCESS;
}

uint8_t *GetRemoteWorkspace(const ChannelInfo &channel)
{
    return static_cast<uint8_t *>(channel.remoteCclMem.addr);
}

HcclResult StartWorker(ThreadHandle mainThread, ThreadHandle worker)
{
    CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, worker, 0));
    CHK_RET(HcommThreadNotifyWaitOnThread(worker, 0, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult FinishWorker(ThreadHandle worker, ThreadHandle mainThread, uint32_t joinIndex)
{
    CHK_RET(HcommThreadNotifyRecordOnThread(worker, mainThread, joinIndex));
    return HCCL_SUCCESS;
}

HcclResult JoinWorkers(ThreadHandle mainThread, uint32_t workerNum)
{
    for (uint32_t i = 0; i < workerNum; ++i) {
        CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, i, CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchFirstPairReductions(const AlgResourceCtx &resCtx, const BufferLayout &layout,
    const PipelinePlan &plan, ThreadHandle mainThread, uint8_t *first0, uint8_t *first1,
    uint32_t &workerNum)
{
    workerNum = static_cast<uint32_t>(std::min<uint64_t>(
        LOCAL_RANK_SIZE - 1, plan.firstShardCount));
    CHK_PRT_RET(workerNum == 0, HCCL_ERROR("No worker available for first-segment reduction"), HCCL_E_PARA);

    // Every worker owns a disjoint element interval.  Inside that interval it
    // consumes senders in the same rank order as the serial implementation,
    // so floating-point results remain deterministic.
    for (uint32_t workerIndex = 0; workerIndex < workerNum; ++workerIndex) {
        const uint64_t partOffset = GetPartitionOffset(plan.firstShardCount, workerIndex, workerNum);
        const uint64_t partCount = GetPartitionCount(plan.firstShardCount, workerIndex, workerNum);
        const uint64_t partBytes = partOffset * FP32_BYTES;
        ThreadHandle worker = resCtx.threads[workerIndex + 1];
        CHK_RET(StartWorker(mainThread, worker));
        for (uint32_t senderLocal = 0; senderLocal < LOCAL_RANK_SIZE; ++senderLocal) {
            if (senderLocal == plan.localRank) {
                continue;
            }
            const uint64_t slotOffset =
                static_cast<uint64_t>(CompactPeerSlot(senderLocal, plan.localRank)) * plan.firstPairBytes;
            CHK_RET(HcommLocalReduceOnThread(worker, first0 + partBytes,
                layout.workspace + slotOffset + partBytes, partCount,
                HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM));
            CHK_RET(HcommLocalReduceOnThread(worker, first1 + partBytes,
                layout.workspace + slotOffset + plan.firstShardBytes + partBytes, partCount,
                HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM));
        }
        CHK_RET(FinishWorker(worker, mainThread, workerIndex));
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchSecondShardReductions(const AlgResourceCtx &resCtx, const BufferLayout &layout,
    const PipelinePlan &plan, ThreadHandle mainThread, uint8_t *second, uint32_t &workerNum)
{
    workerNum = static_cast<uint32_t>(std::min<uint64_t>(
        LOCAL_RANK_SIZE - 1, plan.ownSecondShardCount));
    CHK_PRT_RET(workerNum == 0, HCCL_ERROR("No worker available for second-segment reduction"), HCCL_E_PARA);

    for (uint32_t workerIndex = 0; workerIndex < workerNum; ++workerIndex) {
        const uint64_t partOffset = GetPartitionOffset(plan.ownSecondShardCount, workerIndex, workerNum);
        const uint64_t partCount = GetPartitionCount(plan.ownSecondShardCount, workerIndex, workerNum);
        const uint64_t partBytes = partOffset * FP32_BYTES;
        ThreadHandle worker = resCtx.threads[workerIndex + 1];
        CHK_RET(StartWorker(mainThread, worker));
        for (uint32_t senderLocal = 0; senderLocal < LOCAL_RANK_SIZE; ++senderLocal) {
            if (senderLocal == plan.localRank) {
                continue;
            }
            const uint64_t slotOffset =
                static_cast<uint64_t>(CompactPeerSlot(senderLocal, plan.localRank)) * plan.ownSecondShardBytes;
            CHK_RET(HcommLocalReduceOnThread(worker, second + partBytes,
                layout.workspace + slotOffset + partBytes, partCount,
                HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM));
        }
        CHK_RET(FinishWorker(worker, mainThread, workerIndex));
    }
    return HCCL_SUCCESS;
}

HcclResult ChannelExchangeDone(ThreadHandle thread, ChannelHandle channel)
{
    CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel, NOTIFY_DATA));
    CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel, NOTIFY_DATA, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult ChannelReady(ThreadHandle thread, ChannelHandle channel)
{
    CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel, NOTIFY_READY));
    CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel, NOTIFY_READY, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult ValidateDirectResources(const AlgResourceCtx &resCtx)
{
    CHK_PRT_RET(resCtx.localBuffer.addr == nullptr || resCtx.localBuffer.size < DIRECT_WORKSPACE_BYTES,
        HCCL_ERROR("Direct-exchange workspace is too small, size[%lu], required[%lu]",
            resCtx.localBuffer.size, DIRECT_WORKSPACE_BYTES),
        HCCL_E_PARA);
    for (const ChannelInfo &channel : resCtx.channels) {
        CHK_PRT_RET(channel.remoteCclMem.addr == nullptr
                || channel.remoteCclMem.size < DIRECT_WORKSPACE_BYTES || channel.notifyNum < 2,
            HCCL_ERROR("Invalid direct channel, rank[%u], bufferSize[%lu], notifyNum[%u]",
                channel.remoteRank, channel.remoteCclMem.size, channel.notifyNum),
            HCCL_E_PARA);
    }
    return HCCL_SUCCESS;
}

HcclResult DirectReduceScatterExchange(const OpParam &param, const AlgResourceCtx &resCtx,
    ThreadHandle mainThread, const DirectRankPlan &rankPlan, uint64_t windowBase,
    uint64_t windowCount, bool reuseWorkspace)
{
    auto *input = static_cast<uint8_t *>(param.inputPtr);
    uint32_t localWorkerIndex = 0;

    // Seven Full-Mesh transfers run on threads 1..7.  Every sender writes into
    // a sender-specific slot at the owner, so no two links can overwrite each
    // other even though they run concurrently.
    for (uint32_t targetGlobal = 0; targetGlobal < COMPETITION_RANK_SIZE; ++targetGlobal) {
        if (targetGlobal == rankPlan.myGlobalRank) {
            continue;
        }
        const uint32_t targetRank = resCtx.globalRanks[targetGlobal];
        if (!IsLocalPeer(resCtx, targetRank)) {
            continue;
        }
        const ChannelInfo *channel = FindChannel(resCtx, targetRank);
        CHK_PRT_RET(channel == nullptr,
            HCCL_ERROR("Direct Mesh channel missing, rank[%u]", targetRank), HCCL_E_PARA);
        const uint64_t targetOffset = GetPartitionOffset(
            windowCount, targetGlobal, COMPETITION_RANK_SIZE);
        const uint64_t targetBytes = GetPartitionCount(
            windowCount, targetGlobal, COMPETITION_RANK_SIZE) * FP32_BYTES;
        const uint32_t senderSlot = CompactPeerSlot(rankPlan.myGlobalRank, targetGlobal);
        ThreadHandle worker = resCtx.threads[localWorkerIndex + 1];
        CHK_RET(StartWorker(mainThread, worker));
        if (reuseWorkspace) {
            CHK_RET(ChannelReady(worker, channel->handle));
        }
        CHK_RET(HcommWriteOnThread(worker, channel->handle,
            GetRemoteWorkspace(*channel) + static_cast<uint64_t>(senderSlot) * DIRECT_MAX_SHARD_BYTES,
            input + (windowBase + targetOffset) * FP32_BYTES, targetBytes));
        CHK_RET(ChannelExchangeDone(worker, channel->handle));
        CHK_RET(FinishWorker(worker, mainThread, localWorkerIndex));
        ++localWorkerIndex;
    }
    CHK_PRT_RET(localWorkerIndex != LOCAL_RANK_SIZE - 1,
        HCCL_ERROR("Expected seven local direct peers, found[%u]", localWorkerIndex), HCCL_E_PARA);

    // One Clos injection path is about eight times one Mesh link, so thread 0
    // drives the eight remote peers in deterministic order while the seven
    // local links are in flight.
    for (uint32_t round = 0; round < LOCAL_RANK_SIZE; ++round) {
        const uint32_t targetRank = rankPlan.remoteRanks[round];
        const uint32_t targetGlobal = rankPlan.remoteGlobalRanks[round];
        const ChannelInfo *channel = FindChannel(resCtx, targetRank);
        CHK_PRT_RET(channel == nullptr,
            HCCL_ERROR("Direct Clos channel missing, rank[%u]", targetRank), HCCL_E_PARA);
        const uint64_t targetOffset = GetPartitionOffset(
            windowCount, targetGlobal, COMPETITION_RANK_SIZE);
        const uint64_t targetBytes = GetPartitionCount(
            windowCount, targetGlobal, COMPETITION_RANK_SIZE) * FP32_BYTES;
        const uint32_t senderSlot = CompactPeerSlot(rankPlan.myGlobalRank, targetGlobal);
        if (reuseWorkspace) {
            CHK_RET(ChannelReady(mainThread, channel->handle));
        }
        CHK_RET(HcommWriteOnThread(mainThread, channel->handle,
            GetRemoteWorkspace(*channel) + static_cast<uint64_t>(senderSlot) * DIRECT_MAX_SHARD_BYTES,
            input + (windowBase + targetOffset) * FP32_BYTES, targetBytes));
        CHK_RET(ChannelExchangeDone(mainThread, channel->handle));
    }
    CHK_RET(JoinWorkers(mainThread, localWorkerIndex));
    return HCCL_SUCCESS;
}

HcclResult DirectReduceOwnShard(const OpParam &param, const AlgResourceCtx &resCtx,
    ThreadHandle mainThread, const DirectRankPlan &rankPlan, uint64_t windowBase, uint64_t windowCount)
{
    auto *input = static_cast<uint8_t *>(param.inputPtr);
    auto *output = static_cast<uint8_t *>(param.outputPtr);
    auto *workspace = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    const uint64_t ownOffset = GetPartitionOffset(
        windowCount, rankPlan.myGlobalRank, COMPETITION_RANK_SIZE);
    const uint64_t ownCount = GetPartitionCount(
        windowCount, rankPlan.myGlobalRank, COMPETITION_RANK_SIZE);
    const uint64_t ownBytes = ownCount * FP32_BYTES;
    uint8_t *ownOutput = output + (windowBase + ownOffset) * FP32_BYTES;
    CHK_PRT_RET(ownBytes > DIRECT_MAX_SHARD_BYTES,
        HCCL_ERROR("Direct shard exceeds slot, bytes[%lu], capacity[%lu]",
            ownBytes, DIRECT_MAX_SHARD_BYTES),
        HCCL_E_PARA);

    // The copy is ordered before the worker-start records on mainThread.
    CHK_RET(HcommLocalCopyOnThread(mainThread, ownOutput,
        input + (windowBase + ownOffset) * FP32_BYTES, ownBytes));

    const uint32_t partNum = static_cast<uint32_t>(
        std::min<uint64_t>(DIRECT_REDUCE_PARTS, ownCount));
    CHK_PRT_RET(partNum == 0, HCCL_ERROR("Direct shard is empty"), HCCL_E_PARA);
    const uint32_t workerNum = partNum - 1;

    // Parts 1..P-1 execute on workers.  Each worker owns disjoint elements and
    // consumes the fifteen senders in the same global-rank order.
    for (uint32_t part = 1; part < partNum; ++part) {
        const uint64_t partOffset = GetPartitionOffset(ownCount, part, partNum);
        const uint64_t partCount = GetPartitionCount(ownCount, part, partNum);
        const uint64_t partBytes = partOffset * FP32_BYTES;
        ThreadHandle worker = resCtx.threads[part];
        CHK_RET(StartWorker(mainThread, worker));
        for (uint32_t senderGlobal = 0; senderGlobal < COMPETITION_RANK_SIZE; ++senderGlobal) {
            if (senderGlobal == rankPlan.myGlobalRank) {
                continue;
            }
            const uint32_t senderSlot = CompactPeerSlot(senderGlobal, rankPlan.myGlobalRank);
            CHK_RET(HcommLocalReduceOnThread(worker, ownOutput + partBytes,
                workspace + static_cast<uint64_t>(senderSlot) * DIRECT_MAX_SHARD_BYTES + partBytes,
                partCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM));
        }
        CHK_RET(FinishWorker(worker, mainThread, part - 1));
    }

    // Part 0 runs on the coordinator concurrently with the worker parts.
    const uint64_t mainPartCount = GetPartitionCount(ownCount, 0, partNum);
    for (uint32_t senderGlobal = 0; senderGlobal < COMPETITION_RANK_SIZE; ++senderGlobal) {
        if (senderGlobal == rankPlan.myGlobalRank) {
            continue;
        }
        const uint32_t senderSlot = CompactPeerSlot(senderGlobal, rankPlan.myGlobalRank);
        CHK_RET(HcommLocalReduceOnThread(mainThread, ownOutput,
            workspace + static_cast<uint64_t>(senderSlot) * DIRECT_MAX_SHARD_BYTES,
            mainPartCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM));
    }
    CHK_RET(JoinWorkers(mainThread, workerNum));

    // All peers expose their final shard at the same fixed remote address.
    CHK_RET(HcommLocalCopyOnThread(mainThread,
        workspace + DIRECT_FINAL_OFFSET, ownOutput, ownBytes));
    return HCCL_SUCCESS;
}

HcclResult DirectAllGather(const OpParam &param, const AlgResourceCtx &resCtx,
    ThreadHandle mainThread, const DirectRankPlan &rankPlan, uint64_t windowBase, uint64_t windowCount)
{
    auto *output = static_cast<uint8_t *>(param.outputPtr);
    uint32_t localWorkerIndex = 0;

    for (uint32_t ownerGlobal = 0; ownerGlobal < COMPETITION_RANK_SIZE; ++ownerGlobal) {
        if (ownerGlobal == rankPlan.myGlobalRank) {
            continue;
        }
        const uint32_t ownerRank = resCtx.globalRanks[ownerGlobal];
        if (!IsLocalPeer(resCtx, ownerRank)) {
            continue;
        }
        const ChannelInfo *channel = FindChannel(resCtx, ownerRank);
        CHK_PRT_RET(channel == nullptr,
            HCCL_ERROR("Direct gather Mesh channel missing, rank[%u]", ownerRank), HCCL_E_PARA);
        const uint64_t ownerOffset = GetPartitionOffset(
            windowCount, ownerGlobal, COMPETITION_RANK_SIZE);
        const uint64_t ownerBytes = GetPartitionCount(
            windowCount, ownerGlobal, COMPETITION_RANK_SIZE) * FP32_BYTES;
        ThreadHandle worker = resCtx.threads[localWorkerIndex + 1];
        CHK_RET(StartWorker(mainThread, worker));
        // READY is recorded only after the owner's final copy, so the following
        // read cannot observe a partially reduced shard.
        CHK_RET(ChannelReady(worker, channel->handle));
        CHK_RET(HcommReadOnThread(worker, channel->handle,
            output + (windowBase + ownerOffset) * FP32_BYTES,
            GetRemoteWorkspace(*channel) + DIRECT_FINAL_OFFSET, ownerBytes));
        CHK_RET(FinishWorker(worker, mainThread, localWorkerIndex));
        ++localWorkerIndex;
    }

    for (uint32_t round = 0; round < LOCAL_RANK_SIZE; ++round) {
        const uint32_t ownerRank = rankPlan.remoteRanks[round];
        const uint32_t ownerGlobal = rankPlan.remoteGlobalRanks[round];
        const ChannelInfo *channel = FindChannel(resCtx, ownerRank);
        CHK_PRT_RET(channel == nullptr,
            HCCL_ERROR("Direct gather Clos channel missing, rank[%u]", ownerRank), HCCL_E_PARA);
        const uint64_t ownerOffset = GetPartitionOffset(
            windowCount, ownerGlobal, COMPETITION_RANK_SIZE);
        const uint64_t ownerBytes = GetPartitionCount(
            windowCount, ownerGlobal, COMPETITION_RANK_SIZE) * FP32_BYTES;
        CHK_RET(ChannelReady(mainThread, channel->handle));
        CHK_RET(HcommReadOnThread(mainThread, channel->handle,
            output + (windowBase + ownerOffset) * FP32_BYTES,
            GetRemoteWorkspace(*channel) + DIRECT_FINAL_OFFSET, ownerBytes));
    }
    CHK_RET(JoinWorkers(mainThread, localWorkerIndex));
    return HCCL_SUCCESS;
}

HcclResult ExecuteDirectExchange(const OpParam &param, const AlgResourceCtx &resCtx,
    ThreadHandle mainThread)
{
    CHK_RET(ValidateDirectResources(resCtx));
    DirectRankPlan rankPlan;
    CHK_RET(BuildDirectRankPlan(param, resCtx, rankPlan));

    uint64_t windowBase = 0;
    while (windowBase < param.count) {
        const uint64_t windowCount = std::min<uint64_t>(
            DIRECT_WINDOW_COUNT, param.count - windowBase);
        // Always perform the pre-write READY/ACK barrier, including window 0.
        // The HCCL workspace persists across operator invocations; without this
        // barrier a fast rank could overwrite the previous call's final shard
        // while a slower peer is still reading it.
        CHK_RET(DirectReduceScatterExchange(param, resCtx, mainThread, rankPlan,
            windowBase, windowCount, true));
        CHK_RET(DirectReduceOwnShard(param, resCtx, mainThread, rankPlan,
            windowBase, windowCount));
        CHK_RET(DirectAllGather(param, resCtx, mainThread, rankPlan,
            windowBase, windowCount));
        windowBase += windowCount;
    }
    return HCCL_SUCCESS;
}

HcclResult BuildPipelinePlan(const OpParam &param, const AlgResourceCtx &resCtx, const BufferLayout &layout,
    PipelinePlan &plan)
{
    CHK_RET(GetLocalRankIndex(resCtx, param.myRank, plan.localRank));
    plan.server = param.myRank < resCtx.interRank ? 0U : 1U;

    // The two opposite-direction flows are balanced near 1:2 because one Clos
    // path is about 8x one Mesh path.  512 MiB uses 21:43 to leave a 1 MiB
    // workspace guard; 400 MiB+4 B can use the balanced 1:2 split directly.
    const uint64_t totalBytes = param.count * FP32_BYTES;
    uint64_t firstTarget = 0;
    if (totalBytes == TEST_512_MIB_BYTES) {
        firstTarget = ScaleCount(param.count, 21, 64);
    } else if (totalBytes == TEST_400_MIB_PLUS_4_BYTES) {
        firstTarget = param.count / 3;
    } else {
        firstTarget = ScaleCount(param.count, 5, 16);
    }
    plan.firstCount = (firstTarget / COMPETITION_RANK_SIZE) * COMPETITION_RANK_SIZE;
    plan.secondCount = param.count - plan.firstCount;
    plan.secondBase = plan.firstCount;
    plan.firstShardCount = plan.firstCount / COMPETITION_RANK_SIZE;
    plan.firstShardBytes = plan.firstShardCount * FP32_BYTES;
    plan.firstPairBytes = 2 * plan.firstShardBytes;

    const uint32_t ownBlockPartition = plan.server * LOCAL_RANK_SIZE;
    plan.ownSecondBlockOffset = GetPartitionOffset(plan.secondCount, ownBlockPartition, COMPETITION_RANK_SIZE);
    const uint64_t ownBlockEnd = GetPartitionOffset(
        plan.secondCount, ownBlockPartition + LOCAL_RANK_SIZE, COMPETITION_RANK_SIZE);
    plan.ownSecondBlockCount = ownBlockEnd - plan.ownSecondBlockOffset;
    plan.ownSecondBlockBytes = plan.ownSecondBlockCount * FP32_BYTES;

    const uint32_t ownShard = ownBlockPartition + plan.localRank;
    plan.ownSecondShardOffset = GetPartitionOffset(plan.secondCount, ownShard, COMPETITION_RANK_SIZE);
    plan.ownSecondShardCount = GetPartitionCount(plan.secondCount, ownShard, COMPETITION_RANK_SIZE);
    plan.ownSecondShardBytes = plan.ownSecondShardCount * FP32_BYTES;

    plan.phase1InterOffset = (LOCAL_RANK_SIZE - 1) * plan.firstPairBytes;
    plan.phase2InterOffset = (LOCAL_RANK_SIZE - 1) * plan.ownSecondShardBytes;
    plan.assemblySecondOffset = plan.firstPairBytes;

    CHK_PRT_RET(plan.firstShardCount == 0,
        HCCL_ERROR("Pipeline first segment is empty, count[%lu]", param.count), HCCL_E_PARA);
    CHK_PRT_RET(plan.phase1InterOffset > layout.workspaceBytes
            || plan.ownSecondBlockBytes > layout.workspaceBytes - plan.phase1InterOffset,
        HCCL_ERROR("Phase-1 workspace overflow, mesh[%lu], inter[%lu], capacity[%lu]",
            plan.phase1InterOffset, plan.ownSecondBlockBytes, layout.workspaceBytes),
        HCCL_E_PARA);
    CHK_PRT_RET(plan.phase2InterOffset > layout.workspaceBytes
            || plan.firstShardBytes > layout.workspaceBytes - plan.phase2InterOffset,
        HCCL_ERROR("Phase-2 workspace overflow, mesh[%lu], inter[%lu], capacity[%lu]",
            plan.phase2InterOffset, plan.firstShardBytes, layout.workspaceBytes),
        HCCL_E_PARA);
    CHK_PRT_RET(plan.assemblySecondOffset > layout.workspaceBytes
            || plan.ownSecondBlockBytes > layout.workspaceBytes - plan.assemblySecondOffset,
        HCCL_ERROR("Assembly workspace overflow, first[%lu], second[%lu], capacity[%lu]",
            plan.assemblySecondOffset, plan.ownSecondBlockBytes, layout.workspaceBytes),
        HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult ExecuteRecursiveDoublingStep(const OpParam &param, const AlgResourceCtx &resCtx, ThreadHandle thread,
    const BufferLayout &layout, uint32_t partner, uint64_t totalBytes, uint64_t stagingOffset)
{
    const ChannelInfo *channel = FindChannel(resCtx, partner);
    CHK_PRT_RET(channel == nullptr, HCCL_ERROR("Recursive-doubling channel to rank[%u] not found", partner),
        HCCL_E_PARA);
    CHK_PRT_RET(stagingOffset > layout.stagingBytes || totalBytes > layout.stagingBytes - stagingOffset,
        HCCL_ERROR("Small-message staging slot is too small, offset[%lu], bytes[%lu], capacity[%lu]",
            stagingOffset, totalBytes, layout.stagingBytes),
        HCCL_E_PARA);

    CHK_RET(HcommWriteOnThread(thread, channel->handle,
        GetRemoteWorkspace(*channel) + layout.aggregateBytes + stagingOffset, layout.aggregate, totalBytes));
    CHK_RET(ChannelExchangeDone(thread, channel->handle));
    CHK_RET(HcommLocalReduceOnThread(
        thread, layout.aggregate, layout.staging + stagingOffset, param.count, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM));
    return HCCL_SUCCESS;
}

HcclResult ExecuteRecursiveDoubling(
    const OpParam &param, const AlgResourceCtx &resCtx, ThreadHandle thread, const BufferLayout &layout)
{
    const uint64_t totalBytes = param.count * FP32_BYTES;
    CHK_PRT_RET(totalBytes > layout.aggregateBytes,
        HCCL_ERROR("Small-message buffer is too small, bytes[%lu], capacity[%lu]", totalBytes,
            layout.aggregateBytes),
        HCCL_E_PARA);

    CHK_RET(HcommLocalCopyOnThread(thread, layout.aggregate, param.inputPtr, totalBytes));
    uint32_t localRank = 0;
    CHK_RET(GetLocalRankIndex(resCtx, param.myRank, localRank));
    uint32_t stepIndex = 0;
    for (uint32_t mask = 1; mask < LOCAL_RANK_SIZE; mask <<= 1) {
        CHK_RET(ExecuteRecursiveDoublingStep(param, resCtx, thread, layout,
            resCtx.localRanks[localRank ^ mask], totalBytes,
            static_cast<uint64_t>(stepIndex) * SMALL_MESSAGE_BYTES));
        ++stepIndex;
    }
    CHK_RET(ExecuteRecursiveDoublingStep(param, resCtx, thread, layout, resCtx.interRank, totalBytes,
        static_cast<uint64_t>(stepIndex) * SMALL_MESSAGE_BYTES));
    CHK_RET(HcommLocalCopyOnThread(thread, param.outputPtr, layout.aggregate, totalBytes));
    return HCCL_SUCCESS;
}

HcclResult Phase1ParallelReduceScatter(const OpParam &param, const AlgResourceCtx &resCtx,
    const BufferLayout &layout, const PipelinePlan &plan, ThreadHandle mainThread)
{
    auto *input = static_cast<uint8_t *>(param.inputPtr);
    auto *output = static_cast<uint8_t *>(param.outputPtr);
    uint32_t workerIndex = 0;

    // First segment: seven Full-Mesh links send the two slices owned by each target local rank.
    for (uint32_t targetLocal = 0; targetLocal < LOCAL_RANK_SIZE; ++targetLocal) {
        if (targetLocal == plan.localRank) {
            continue;
        }
        const uint32_t targetRank = resCtx.localRanks[targetLocal];
        const ChannelInfo *channel = FindChannel(resCtx, targetRank);
        CHK_PRT_RET(channel == nullptr, HCCL_ERROR("Phase-1 Mesh channel missing, rank[%u]", targetRank),
            HCCL_E_PARA);
        ThreadHandle worker = resCtx.threads[workerIndex + 1];
        CHK_RET(StartWorker(mainThread, worker));
        const uint32_t remoteSlot = CompactPeerSlot(plan.localRank, targetLocal);
        uint8_t *remoteScratch = GetRemoteWorkspace(*channel) + remoteSlot * plan.firstPairBytes;
        CHK_RET(HcommWriteOnThread(worker, channel->handle, remoteScratch,
            input + static_cast<uint64_t>(targetLocal) * plan.firstShardBytes, plan.firstShardBytes));
        CHK_RET(HcommWriteOnThread(worker, channel->handle, remoteScratch + plan.firstShardBytes,
            input + static_cast<uint64_t>(LOCAL_RANK_SIZE + targetLocal) * plan.firstShardBytes,
            plan.firstShardBytes));
        CHK_RET(ChannelExchangeDone(worker, channel->handle));
        CHK_RET(FinishWorker(worker, mainThread, workerIndex));
        ++workerIndex;
    }

    // Second segment: the corresponding rank in the other server sends the block retained by this server.
    const ChannelInfo *interChannel = FindChannel(resCtx, resCtx.interRank);
    CHK_PRT_RET(interChannel == nullptr,
        HCCL_ERROR("Phase-1 Clos channel missing, rank[%u]", resCtx.interRank), HCCL_E_PARA);
    const uint32_t remoteServer = 1U - plan.server;
    const uint32_t remoteBlockPartition = remoteServer * LOCAL_RANK_SIZE;
    const uint64_t remoteBlockOffset = GetPartitionOffset(
        plan.secondCount, remoteBlockPartition, COMPETITION_RANK_SIZE);
    const uint64_t remoteBlockEnd = GetPartitionOffset(
        plan.secondCount, remoteBlockPartition + LOCAL_RANK_SIZE, COMPETITION_RANK_SIZE);
    const uint64_t remoteBlockBytes = (remoteBlockEnd - remoteBlockOffset) * FP32_BYTES;
    CHK_RET(HcommWriteOnThread(mainThread, interChannel->handle,
        GetRemoteWorkspace(*interChannel) + plan.phase1InterOffset,
        input + (plan.secondBase + remoteBlockOffset) * FP32_BYTES, remoteBlockBytes));
    CHK_RET(ChannelExchangeDone(mainThread, interChannel->handle));
    CHK_RET(JoinWorkers(mainThread, workerIndex));

    // Reduce in a fixed sender-local-rank order after every transfer has completed.
    const uint64_t ownFirst0 = static_cast<uint64_t>(plan.localRank) * plan.firstShardBytes;
    const uint64_t ownFirst1 = static_cast<uint64_t>(LOCAL_RANK_SIZE + plan.localRank) * plan.firstShardBytes;
    CHK_RET(HcommLocalCopyOnThread(mainThread, output + ownFirst0, input + ownFirst0, plan.firstShardBytes));
    CHK_RET(HcommLocalCopyOnThread(mainThread, output + ownFirst1, input + ownFirst1, plan.firstShardBytes));
    uint32_t reduceWorkerNum = 0;
    CHK_RET(LaunchFirstPairReductions(resCtx, layout, plan, mainThread,
        output + ownFirst0, output + ownFirst1, reduceWorkerNum));

    const uint64_t ownSecondOutput = (plan.secondBase + plan.ownSecondBlockOffset) * FP32_BYTES;
    CHK_RET(HcommLocalCopyOnThread(mainThread, output + ownSecondOutput, input + ownSecondOutput,
        plan.ownSecondBlockBytes));
    CHK_RET(HcommLocalReduceOnThread(mainThread, output + ownSecondOutput,
        layout.workspace + plan.phase1InterOffset, plan.ownSecondBlockCount,
        HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM));
    CHK_RET(JoinWorkers(mainThread, reduceWorkerNum));
    return HCCL_SUCCESS;
}

HcclResult Phase2ParallelReduceScatter(const OpParam &param, const AlgResourceCtx &resCtx,
    const BufferLayout &layout, const PipelinePlan &plan, ThreadHandle mainThread)
{
    auto *output = static_cast<uint8_t *>(param.outputPtr);
    const uint32_t blockPartition = plan.server * LOCAL_RANK_SIZE;
    uint32_t workerIndex = 0;

    // Second segment: reduce-scatter the server block over all seven local Mesh links.
    for (uint32_t targetLocal = 0; targetLocal < LOCAL_RANK_SIZE; ++targetLocal) {
        if (targetLocal == plan.localRank) {
            continue;
        }
        const uint32_t targetRank = resCtx.localRanks[targetLocal];
        const ChannelInfo *channel = FindChannel(resCtx, targetRank);
        CHK_PRT_RET(channel == nullptr, HCCL_ERROR("Phase-2 Mesh channel missing, rank[%u]", targetRank),
            HCCL_E_PARA);
        const uint32_t targetShard = blockPartition + targetLocal;
        const uint64_t targetShardOffset = GetPartitionOffset(
            plan.secondCount, targetShard, COMPETITION_RANK_SIZE);
        const uint64_t targetShardBytes = GetPartitionCount(
            plan.secondCount, targetShard, COMPETITION_RANK_SIZE) * FP32_BYTES;
        ThreadHandle worker = resCtx.threads[workerIndex + 1];
        CHK_RET(StartWorker(mainThread, worker));
        // Phase 2 reuses the Phase-1 scratch region. Do not overwrite a peer until it has completed its local
        // deterministic reduction from that region.
        CHK_RET(ChannelReady(worker, channel->handle));
        const uint32_t remoteSlot = CompactPeerSlot(plan.localRank, targetLocal);
        CHK_RET(HcommWriteOnThread(worker, channel->handle,
            GetRemoteWorkspace(*channel) + static_cast<uint64_t>(remoteSlot) * targetShardBytes,
            output + (plan.secondBase + targetShardOffset) * FP32_BYTES, targetShardBytes));
        CHK_RET(ChannelExchangeDone(worker, channel->handle));
        CHK_RET(FinishWorker(worker, mainThread, workerIndex));
        ++workerIndex;
    }

    // First segment: exchange the slice retained by the opposite server.
    const ChannelInfo *interChannel = FindChannel(resCtx, resCtx.interRank);
    CHK_PRT_RET(interChannel == nullptr,
        HCCL_ERROR("Phase-2 Clos channel missing, rank[%u]", resCtx.interRank), HCCL_E_PARA);
    const uint32_t remoteServer = 1U - plan.server;
    const uint32_t remoteFirstShard = remoteServer * LOCAL_RANK_SIZE + plan.localRank;
    const uint64_t remotePhase2Offset = (LOCAL_RANK_SIZE - 1) *
        GetPartitionCount(plan.secondCount, remoteFirstShard, COMPETITION_RANK_SIZE) * FP32_BYTES;
    CHK_RET(ChannelReady(mainThread, interChannel->handle));
    CHK_RET(HcommWriteOnThread(mainThread, interChannel->handle,
        GetRemoteWorkspace(*interChannel) + remotePhase2Offset,
        output + static_cast<uint64_t>(remoteFirstShard) * plan.firstShardBytes, plan.firstShardBytes));
    CHK_RET(ChannelExchangeDone(mainThread, interChannel->handle));
    CHK_RET(JoinWorkers(mainThread, workerIndex));

    const uint64_t ownSecondOffset = (plan.secondBase + plan.ownSecondShardOffset) * FP32_BYTES;
    uint32_t reduceWorkerNum = 0;
    CHK_RET(LaunchSecondShardReductions(resCtx, layout, plan, mainThread,
        output + ownSecondOffset, reduceWorkerNum));

    const uint32_t ownFirstShard = plan.server * LOCAL_RANK_SIZE + plan.localRank;
    const uint64_t ownFirstOffset = static_cast<uint64_t>(ownFirstShard) * plan.firstShardBytes;
    CHK_RET(HcommLocalReduceOnThread(mainThread, output + ownFirstOffset,
        layout.workspace + plan.phase2InterOffset, plan.firstShardCount,
        HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM));
    CHK_RET(JoinWorkers(mainThread, reduceWorkerNum));

    // Store the two final shards in the remotely addressable CCL workspace for the two AllGather phases.
    CHK_RET(HcommLocalCopyOnThread(mainThread,
        layout.workspace + static_cast<uint64_t>(plan.server) * plan.firstShardBytes,
        output + ownFirstOffset, plan.firstShardBytes));
    const uint64_t secondRelativeOffset = plan.ownSecondShardOffset - plan.ownSecondBlockOffset;
    CHK_RET(HcommLocalCopyOnThread(mainThread,
        layout.workspace + plan.assemblySecondOffset + secondRelativeOffset * FP32_BYTES,
        output + ownSecondOffset, plan.ownSecondShardBytes));
    return HCCL_SUCCESS;
}

HcclResult Phase3ParallelAllGather(const AlgResourceCtx &resCtx, const BufferLayout &layout,
    const PipelinePlan &plan, ThreadHandle mainThread)
{
    const uint32_t blockPartition = plan.server * LOCAL_RANK_SIZE;
    uint32_t workerIndex = 0;

    // Second segment: gather the seven local final shards into one contiguous server block.
    for (uint32_t peerLocal = 0; peerLocal < LOCAL_RANK_SIZE; ++peerLocal) {
        if (peerLocal == plan.localRank) {
            continue;
        }
        const uint32_t peerRank = resCtx.localRanks[peerLocal];
        const ChannelInfo *channel = FindChannel(resCtx, peerRank);
        CHK_PRT_RET(channel == nullptr, HCCL_ERROR("Phase-3 Mesh channel missing, rank[%u]", peerRank),
            HCCL_E_PARA);
        const uint32_t peerShard = blockPartition + peerLocal;
        const uint64_t peerShardOffset = GetPartitionOffset(
            plan.secondCount, peerShard, COMPETITION_RANK_SIZE);
        const uint64_t peerRelativeOffset = peerShardOffset - plan.ownSecondBlockOffset;
        const uint64_t peerShardBytes = GetPartitionCount(
            plan.secondCount, peerShard, COMPETITION_RANK_SIZE) * FP32_BYTES;
        ThreadHandle worker = resCtx.threads[workerIndex + 1];
        CHK_RET(StartWorker(mainThread, worker));
        CHK_RET(ChannelReady(worker, channel->handle));
        CHK_RET(HcommReadOnThread(worker, channel->handle,
            layout.workspace + plan.assemblySecondOffset + peerRelativeOffset * FP32_BYTES,
            GetRemoteWorkspace(*channel) + plan.assemblySecondOffset + peerRelativeOffset * FP32_BYTES,
            peerShardBytes));
        CHK_RET(FinishWorker(worker, mainThread, workerIndex));
        ++workerIndex;
    }

    // First segment: obtain the final shard owned by the other server in this corresponding-rank pair.
    const ChannelInfo *interChannel = FindChannel(resCtx, resCtx.interRank);
    CHK_PRT_RET(interChannel == nullptr,
        HCCL_ERROR("Phase-3 Clos channel missing, rank[%u]", resCtx.interRank), HCCL_E_PARA);
    const uint32_t remoteServer = 1U - plan.server;
    CHK_RET(ChannelReady(mainThread, interChannel->handle));
    CHK_RET(HcommReadOnThread(mainThread, interChannel->handle,
        layout.workspace + static_cast<uint64_t>(remoteServer) * plan.firstShardBytes,
        GetRemoteWorkspace(*interChannel) + static_cast<uint64_t>(remoteServer) * plan.firstShardBytes,
        plan.firstShardBytes));
    CHK_RET(JoinWorkers(mainThread, workerIndex));
    return HCCL_SUCCESS;
}

HcclResult Phase4ParallelAllGather(const OpParam &param, const AlgResourceCtx &resCtx,
    const BufferLayout &layout, const PipelinePlan &plan, ThreadHandle mainThread)
{
    auto *output = static_cast<uint8_t *>(param.outputPtr);
    uint32_t workerIndex = 0;

    // First segment: every local peer exposes a two-server shard pair.
    for (uint32_t peerLocal = 0; peerLocal < LOCAL_RANK_SIZE; ++peerLocal) {
        if (peerLocal == plan.localRank) {
            continue;
        }
        const uint32_t peerRank = resCtx.localRanks[peerLocal];
        const ChannelInfo *channel = FindChannel(resCtx, peerRank);
        CHK_PRT_RET(channel == nullptr, HCCL_ERROR("Phase-4 Mesh channel missing, rank[%u]", peerRank),
            HCCL_E_PARA);
        ThreadHandle worker = resCtx.threads[workerIndex + 1];
        CHK_RET(StartWorker(mainThread, worker));
        CHK_RET(ChannelReady(worker, channel->handle));
        CHK_RET(HcommReadOnThread(worker, channel->handle,
            output + static_cast<uint64_t>(peerLocal) * plan.firstShardBytes,
            GetRemoteWorkspace(*channel), plan.firstShardBytes));
        CHK_RET(HcommReadOnThread(worker, channel->handle,
            output + static_cast<uint64_t>(LOCAL_RANK_SIZE + peerLocal) * plan.firstShardBytes,
            GetRemoteWorkspace(*channel) + plan.firstShardBytes, plan.firstShardBytes));
        CHK_RET(FinishWorker(worker, mainThread, workerIndex));
        ++workerIndex;
    }

    // Second segment: exchange the assembled server blocks across Clos while the Mesh reads are in flight.
    const ChannelInfo *interChannel = FindChannel(resCtx, resCtx.interRank);
    CHK_PRT_RET(interChannel == nullptr,
        HCCL_ERROR("Phase-4 Clos channel missing, rank[%u]", resCtx.interRank), HCCL_E_PARA);
    const uint32_t remoteServer = 1U - plan.server;
    const uint32_t remoteBlockPartition = remoteServer * LOCAL_RANK_SIZE;
    const uint64_t remoteBlockOffset = GetPartitionOffset(
        plan.secondCount, remoteBlockPartition, COMPETITION_RANK_SIZE);
    const uint64_t remoteBlockEnd = GetPartitionOffset(
        plan.secondCount, remoteBlockPartition + LOCAL_RANK_SIZE, COMPETITION_RANK_SIZE);
    const uint64_t remoteBlockBytes = (remoteBlockEnd - remoteBlockOffset) * FP32_BYTES;
    CHK_RET(ChannelReady(mainThread, interChannel->handle));
    CHK_RET(HcommReadOnThread(mainThread, interChannel->handle,
        output + (plan.secondBase + remoteBlockOffset) * FP32_BYTES,
        GetRemoteWorkspace(*interChannel) + plan.assemblySecondOffset, remoteBlockBytes));

    const uint64_t ownFirst0 = static_cast<uint64_t>(plan.localRank) * plan.firstShardBytes;
    const uint64_t ownFirst1 = static_cast<uint64_t>(LOCAL_RANK_SIZE + plan.localRank) * plan.firstShardBytes;
    CHK_RET(HcommLocalCopyOnThread(mainThread, output + ownFirst0, layout.workspace, plan.firstShardBytes));
    CHK_RET(HcommLocalCopyOnThread(mainThread, output + ownFirst1,
        layout.workspace + plan.firstShardBytes, plan.firstShardBytes));
    CHK_RET(HcommLocalCopyOnThread(mainThread,
        output + (plan.secondBase + plan.ownSecondBlockOffset) * FP32_BYTES,
        layout.workspace + plan.assemblySecondOffset, plan.ownSecondBlockBytes));
    CHK_RET(JoinWorkers(mainThread, workerIndex));
    return HCCL_SUCCESS;
}

HcclResult ExecuteOfficialPipeline(const OpParam &param, const AlgResourceCtx &resCtx,
    const BufferLayout &layout, ThreadHandle mainThread)
{
    PipelinePlan plan;
    CHK_RET(BuildPipelinePlan(param, resCtx, layout, plan));
    CHK_RET(Phase1ParallelReduceScatter(param, resCtx, layout, plan, mainThread));
    CHK_RET(Phase2ParallelReduceScatter(param, resCtx, layout, plan, mainThread));
    CHK_RET(Phase3ParallelAllGather(resCtx, layout, plan, mainThread));
    CHK_RET(Phase4ParallelAllGather(param, resCtx, layout, plan, mainThread));
    return HCCL_SUCCESS;
}
} // namespace

namespace ops_hccl {
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    HCCL_INFO("Executing adaptive direct-exchange AllReduce on AICPU, rank[%u/%u], count[%lu]",
        param.myRank, param.rankSize, param.count);

    CHK_PTR_NULL(param.inputPtr);
    CHK_PTR_NULL(param.outputPtr);
    CHK_PRT_RET(param.dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("Only FP32 is supported, dataType[%d]", static_cast<int32_t>(param.dataType)), HCCL_E_PARA);
    CHK_PRT_RET(param.reduceType != HCCL_REDUCE_SUM,
        HCCL_ERROR("Only SUM is supported, reduceOp[%d]", static_cast<int32_t>(param.reduceType)), HCCL_E_PARA);
    CHK_PRT_RET(param.rankSize != COMPETITION_RANK_SIZE || param.myRank >= param.rankSize,
        HCCL_ERROR("This implementation requires 16 ranks, rank[%u], rankSize[%u]", param.myRank,
            param.rankSize),
        HCCL_E_PARA);
    const uint64_t totalBytes = param.count * FP32_BYTES;
    const bool smallMessage = totalBytes <= SMALL_MESSAGE_BYTES;
    CHK_PRT_RET(resCtx.localRanks.size() != LOCAL_RANK_SIZE
            || resCtx.globalRanks.size() != COMPETITION_RANK_SIZE
            || resCtx.interRank == INVALID_VALUE_RANKID,
        HCCL_ERROR("Invalid topology, localRanks[%lu], globalRanks[%lu], interRank[%u]",
            resCtx.localRanks.size(), resCtx.globalRanks.size(), resCtx.interRank),
        HCCL_E_PARA);
    if (smallMessage) {
        CHK_PRT_RET(resCtx.threads.empty() || resCtx.channels.size() != 4,
            HCCL_ERROR("Invalid small resources, threads[%lu], channels[%lu]",
                resCtx.threads.size(), resCtx.channels.size()),
            HCCL_E_PARA);
    } else {
        CHK_PRT_RET(resCtx.threads.size() < LOCAL_RANK_SIZE
                || resCtx.channels.size() != COMPETITION_RANK_SIZE - 1,
            HCCL_ERROR("Invalid large resources, threads[%lu], channels[%lu]",
                resCtx.threads.size(), resCtx.channels.size()),
            HCCL_E_PARA);
    }

    if (param.count == 0) {
        return HCCL_SUCCESS;
    }

    ThreadHandle mainThread = resCtx.threads[0];
    if (totalBytes <= SMALL_MESSAGE_BYTES) {
        BufferLayout layout;
        CHK_RET(GetBufferLayout(resCtx, layout));
        return ExecuteRecursiveDoubling(param, resCtx, mainThread, layout);
    }
    if (totalBytes == TEST_512_MIB_BYTES || totalBytes == TEST_400_MIB_PLUS_4_BYTES) {
        return ExecuteDirectExchange(param, resCtx, mainThread);
    }
    BufferLayout layout;
    CHK_RET(GetBufferLayout(resCtx, layout));
    return ExecuteOfficialPipeline(param, resCtx, layout, mainThread);
}
} // namespace ops_hccl
