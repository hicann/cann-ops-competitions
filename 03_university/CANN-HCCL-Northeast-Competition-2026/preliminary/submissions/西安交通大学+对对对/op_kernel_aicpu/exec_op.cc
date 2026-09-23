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
#include <limits>

namespace ops_hccl {
namespace {
constexpr uint64_t MAX_CHUNK_BYTES = SCATTER_CHUNK_BYTES;
constexpr uint32_t MAX_WORKERS = SCATTER_MAX_WORKERS;
constexpr uint32_t SLOT_COUNT = 2;
constexpr uint32_t WORKER_START_NOTIFY = 0;
constexpr uint32_t REMOTE_TAIL_DONE_NOTIFY = 1;

uint32_t RankIndex(const std::vector<uint32_t> &ranks, uint32_t rank)
{
    for (uint32_t index = 0; index < ranks.size(); ++index) {
        if (ranks[index] == rank) {
            return index;
        }
    }
    return INVALID_VALUE_RANKID;
}

uint32_t RootWorker(const OpParam &param, const AlgResourceCtx &resCtx, uint32_t peer)
{
    const uint32_t peerIndex = peer < param.root ? peer : peer - 1;
    return peerIndex % resCtx.workerCount;
}

uint64_t ChunkCount(uint64_t bytes, uint64_t chunkBytes)
{
    return bytes == 0 ? 0 : (bytes - 1) / chunkBytes + 1;
}

const ChannelInfo *FindChannel(const AlgResourceCtx &resCtx, uint32_t peer)
{
    for (const auto &channel : resCtx.channels) {
        if (channel.remoteRank == peer) {
            return &channel;
        }
    }
    return nullptr;
}

HcclResult CopyRootOutput(const OpParam &param, ThreadHandle thread, uint64_t recvBytes)
{
    const auto *source = static_cast<const char *>(param.inputPtr) +
        static_cast<uint64_t>(param.root) * recvBytes;
    auto *output = static_cast<char *>(param.outputPtr);
    if (source == output) {
        return HCCL_SUCCESS;
    }
    for (uint64_t offset = 0; offset < recvBytes;) {
        const uint64_t bytes = std::min(MAX_CHUNK_BYTES, recvBytes - offset);
        CHK_RET(HcommLocalCopyOnThread(thread, output + offset, source + offset, bytes));
        offset += bytes;
    }
    return HCCL_SUCCESS;
}

HcclResult ValidateHybridPlan(const OpParam &param, const AlgResourceCtx &resCtx)
{
    CHK_PRT_RET(resCtx.routeRoot != param.root || param.rankSize != SCATTER_SERVER_RANKS * 2 ||
        param.count < 4 || resCtx.chunkBytes == 0 || resCtx.workerCount != param.rankSize - 1 ||
        resCtx.relayRanks.size() != SCATTER_RELAY_COUNT ||
        resCtx.remoteRanks.size() != SCATTER_REMOTE_COUNT,
        HCCL_ERROR("Scatter hybrid route configuration is invalid"), HCCL_E_PARA);
    CHK_PRT_RET(!std::is_sorted(resCtx.relayRanks.begin(), resCtx.relayRanks.end()) ||
        !std::is_sorted(resCtx.remoteRanks.begin(), resCtx.remoteRanks.end()),
        HCCL_ERROR("Scatter hybrid rank vectors must use canonical order"), HCCL_E_PARA);
    std::vector<bool> seen(param.rankSize, false);
    seen[param.root] = true;
    for (const auto rank : resCtx.relayRanks) {
        CHK_PRT_RET(rank >= param.rankSize || seen[rank],
            HCCL_ERROR("Scatter relay ranks must be distinct and exclude root"), HCCL_E_PARA);
        seen[rank] = true;
    }
    for (const auto rank : resCtx.remoteRanks) {
        CHK_PRT_RET(rank >= param.rankSize || seen[rank],
            HCCL_ERROR("Scatter remote ranks must be distinct and exclude root/relays"), HCCL_E_PARA);
        seen[rank] = true;
    }
    return HCCL_SUCCESS;
}

HcclResult ValidateRelayResources(const OpParam &param, const AlgResourceCtx &resCtx)
{
    const uint32_t relayIndex = RankIndex(resCtx.relayRanks, param.myRank);
    if (relayIndex != INVALID_VALUE_RANKID) {
        const uint32_t childCount = ScatterRelayChildCount(relayIndex);
        const uint64_t relayBytes = resCtx.chunkBytes *
            ScatterRelayBufferSlots(resCtx.routeMode, relayIndex);
        CHK_PTR_NULL(resCtx.localBuffer.addr);
        CHK_PRT_RET(resCtx.localBuffer.size < relayBytes,
            HCCL_ERROR("Scatter relay buffer cannot hold its child slot pairs"), HCCL_E_PARA);
        for (uint32_t child = 0; child < childCount; ++child) {
            const uint32_t remoteIndex = ScatterRelayRemoteIndex(relayIndex, child);
            const ChannelInfo *channel = FindChannel(resCtx, resCtx.remoteRanks[remoteIndex]);
            CHK_PTR_NULL(channel);
            CHK_PRT_RET(channel->handle == 0 || channel->notifyNum < SLOT_COUNT * 2,
                HCCL_ERROR("Scatter relay child channel is invalid"), HCCL_E_PARA);
        }
    }
    const uint32_t remoteIndex = RankIndex(resCtx.remoteRanks, param.myRank);
    if (remoteIndex != INVALID_VALUE_RANKID) {
        const uint32_t sourceRelay = ScatterRemoteRelayIndex(remoteIndex);
        const uint64_t relayBytes = resCtx.chunkBytes *
            ScatterRelayBufferSlots(resCtx.routeMode, sourceRelay);
        const ChannelInfo *channel = FindChannel(resCtx, resCtx.relayRanks[sourceRelay]);
        CHK_PTR_NULL(channel);
        CHK_PRT_RET(channel->handle == 0 || channel->notifyNum < SLOT_COUNT * 2,
            HCCL_ERROR("Scatter remote relay channel is invalid"), HCCL_E_PARA);
        CHK_PTR_NULL(channel->remoteCclMem.addr);
        CHK_PRT_RET(channel->remoteCclMem.size < relayBytes,
            HCCL_ERROR("Scatter remote relay buffer cannot hold its child slot pairs"), HCCL_E_PARA);
    }
    return HCCL_SUCCESS;
}

HcclResult ValidateSmallTreeResources(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t recvBytes)
{
    CHK_PRT_RET(param.rankSize != SCATTER_SERVER_RANKS * 2 ||
        resCtx.smallRanks.size() != param.rankSize,
        HCCL_ERROR("Scatter small tree requires four groups of four ranks"), HCCL_E_PARA);
    std::vector<bool> seen(param.rankSize, false);
    for (uint32_t index = 0; index < param.rankSize; ++index) {
        const uint32_t rank = resCtx.smallRanks[index];
        CHK_PRT_RET(rank >= param.rankSize || seen[rank],
            HCCL_ERROR("Scatter small groups must partition all ranks exactly once"), HCCL_E_PARA);
        seen[rank] = true;
        CHK_PRT_RET(index % SCATTER_SERVER_RANKS != 0 && rank <= resCtx.smallRanks[index - 1],
            HCCL_ERROR("Scatter small server members must be sorted"), HCCL_E_PARA);
    }
    CHK_PRT_RET(resCtx.smallRanks[0] != 0,
        HCCL_ERROR("Scatter small server order must be canonical"), HCCL_E_PARA);
    // Validate the complete table: leaves now read from leaders, and leaders
    // need child channels as well as root. Do not validate only the root edge.
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        const uint32_t group = ScatterSmallGroupIndex(resCtx.smallRanks, rank);
        const uint32_t leader = ScatterSmallGroupLeader(resCtx.smallRanks, group, param.root);
        const uint64_t requiredBytes = rank == param.root ? recvBytes * param.rankSize :
            (rank == leader ? recvBytes * SCATTER_SMALL_GROUP_RANKS : 0);
        CommBuffer buffer = resCtx.localBuffer;
        if (rank != param.myRank) {
            const ChannelInfo *channel = FindChannel(resCtx, rank);
            CHK_PTR_NULL(channel);
            CHK_PRT_RET(channel->handle == 0 || channel->notifyNum < 2,
                HCCL_ERROR("Scatter small tree channel or notifications are invalid"), HCCL_E_PARA);
            buffer = channel->remoteCclMem;
        }
        if (requiredBytes != 0) {
            CHK_PTR_NULL(buffer.addr);
            CHK_PRT_RET(buffer.size < requiredBytes,
                HCCL_ERROR("Scatter small tree staging capacity is insufficient"), HCCL_E_PARA);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ValidateResources(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t recvBytes)
{
    CHK_PRT_RET(resCtx.aicpuThread == 0 || resCtx.threads.empty() ||
        resCtx.threads[0] != resCtx.aicpuThread,
        HCCL_ERROR("Scatter main thread is missing or inconsistent"), HCCL_E_PARA);
    if (param.rankSize == 1) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(resCtx.channels.size() != param.rankSize - 1,
        HCCL_ERROR("Scatter requires one cached channel per peer"), HCCL_E_PARA);
    CHK_PRT_RET(resCtx.routeMode != SCATTER_ROUTE_DIRECT && !ScatterIsHybridRoute(resCtx.routeMode),
        HCCL_ERROR("Scatter route mode is invalid"), HCCL_E_PARA);
    const bool hybrid = ScatterIsHybridRoute(resCtx.routeMode);
    const bool receiverStaging = resCtx.routeMode == SCATTER_ROUTE_HYBRID_PUSH;
    if (hybrid) {
        CHK_RET(ValidateHybridPlan(param, resCtx));
    }

    const bool small = resCtx.chunkBytes == 0;
    uint64_t stagingBytes = recvBytes * param.rankSize;
    uint32_t requiredNotifies = 2;
    size_t requiredThreads = 1;
    if (small) {
        const uint32_t expectedWorkers = ScatterSmallRootWorkerCount(param.rankSize, param.myRank, param.root);
        CHK_PRT_RET(resCtx.workerCount != expectedWorkers || resCtx.routeMode != SCATTER_ROUTE_DIRECT ||
            stagingBytes > SCATTER_SMALL_INPUT_BYTES ||
            (resCtx.smallMode != SCATTER_SMALL_FANOUT && resCtx.smallMode != SCATTER_SMALL_TREE) ||
            (resCtx.smallMode == SCATTER_SMALL_FANOUT && !resCtx.smallRanks.empty()),
            HCCL_ERROR("Scatter small-input configuration is invalid"), HCCL_E_PARA);
        requiredThreads += ScatterSmallAuxWorkerCount(expectedWorkers, resCtx.smallMode);
    } else {
        CHK_PRT_RET(resCtx.workerCount == 0 || resCtx.workerCount > MAX_WORKERS ||
            resCtx.workerCount > param.rankSize - 1 || resCtx.chunkBytes > MAX_CHUNK_BYTES ||
            resCtx.chunkBytes % sizeof(float) != 0,
            HCCL_ERROR("Scatter pipeline configuration is invalid"), HCCL_E_PARA);
        stagingBytes = resCtx.chunkBytes * SLOT_COUNT * resCtx.workerCount;
        requiredNotifies = receiverStaging ? SCATTER_CHANNEL_NOTIFIES : SLOT_COUNT * 2;
        if (param.myRank == param.root) {
            requiredThreads += resCtx.workerCount;
        } else if (hybrid && RankIndex(resCtx.remoteRanks, param.myRank) != INVALID_VALUE_RANKID) {
            // The prefix uses main while the relay tail uses a separate TS flow.
            requiredThreads += 1;
        }
    }
    CHK_PRT_RET(resCtx.threads.size() != requiredThreads,
        HCCL_ERROR("Scatter thread count does not match its selected algorithm"), HCCL_E_PARA);
    for (size_t i = 0; i < resCtx.threads.size(); ++i) {
        CHK_PRT_RET(resCtx.threads[i] == 0,
            HCCL_ERROR("Scatter thread handle is invalid"), HCCL_E_PARA);
        for (size_t j = 0; j < i; ++j) {
            CHK_PRT_RET(resCtx.threads[i] == resCtx.threads[j],
                HCCL_ERROR("Scatter worker threads must be distinct"), HCCL_E_PARA);
        }
    }

    if (small && resCtx.smallMode == SCATTER_SMALL_TREE) {
        return ValidateSmallTreeResources(param, resCtx, recvBytes);
    }
    // Check all resources needed by this call before submitting a READY or fork.
    if (param.myRank == param.root) {
        CHK_PTR_NULL(resCtx.localBuffer.addr);
        CHK_PRT_RET(resCtx.localBuffer.size < stagingBytes,
            HCCL_ERROR("Scatter root staging buffer is too small"), HCCL_E_PARA);
    } else if (receiverStaging) {
        CHK_PTR_NULL(resCtx.localBuffer.addr);
        CHK_PRT_RET(resCtx.localBuffer.size < resCtx.chunkBytes * SLOT_COUNT,
            HCCL_ERROR("Scatter receiver buffer cannot hold incoming slots"), HCCL_E_PARA);
    }
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer == param.myRank || (param.myRank != param.root && peer != param.root)) {
            continue;
        }
        const ChannelInfo *channel = FindChannel(resCtx, peer);
        CHK_PTR_NULL(channel);
        CHK_PRT_RET(channel->handle == 0 || channel->notifyNum < requiredNotifies,
            HCCL_ERROR("Scatter channel handle or notify count is invalid"), HCCL_E_PARA);
        if (param.myRank != param.root) {
            CHK_PTR_NULL(channel->remoteCclMem.addr);
            CHK_PRT_RET(channel->remoteCclMem.size < stagingBytes,
                HCCL_ERROR("Scatter root remote staging buffer is too small"), HCCL_E_PARA);
        } else if (receiverStaging) {
            const uint32_t relayIndex = RankIndex(resCtx.relayRanks, peer);
            const uint32_t slots = relayIndex == INVALID_VALUE_RANKID ? SLOT_COUNT :
                ScatterRelayBufferSlots(resCtx.routeMode, relayIndex);
            CHK_PTR_NULL(channel->remoteCclMem.addr);
            CHK_PRT_RET(channel->remoteCclMem.size < resCtx.chunkBytes * slots,
                HCCL_ERROR("Scatter peer buffer cannot hold receiver staging layout"), HCCL_E_PARA);
        }
    }
    if (hybrid) {
        CHK_RET(ValidateRelayResources(param, resCtx));
    }
    return HCCL_SUCCESS;
}

HcclResult ScheduleSmallPeerGroup(const OpParam &param, const AlgResourceCtx &resCtx,
    ThreadHandle thread, uint32_t group)
{
    // Publish every READY in this group before waiting for any peer's ACK.
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer != param.root && RootWorker(param, resCtx, peer) == group) {
            CHK_RET(HcommChannelNotifyRecordOnThread(thread, FindChannel(resCtx, peer)->handle, 0));
        }
    }
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer != param.root && RootWorker(param, resCtx, peer) == group) {
            CHK_RET(HcommChannelNotifyWaitOnThread(thread, FindChannel(resCtx, peer)->handle,
                1, CUSTOM_TIMEOUT));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ScheduleSmallWorker(const OpParam &param, const AlgResourceCtx &resCtx, uint32_t worker)
{
    const ThreadHandle thread = resCtx.threads[worker + 1];
    CHK_RET(HcommThreadNotifyWaitOnThread(thread, WORKER_START_NOTIFY, CUSTOM_TIMEOUT));
    CHK_RET(ScheduleSmallPeerGroup(param, resCtx, thread, worker));
    return static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(thread, resCtx.aicpuThread, worker + 1));
}

HcclResult SmallScatter(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t recvBytes)
{
    const ThreadHandle mainThread = resCtx.aicpuThread;
    if (param.myRank != param.root) {
        const ChannelInfo &channel = *FindChannel(resCtx, param.root);
        const auto *source = static_cast<const char *>(channel.remoteCclMem.addr) +
            static_cast<uint64_t>(param.myRank) * recvBytes;
        CHK_RET(HcommChannelNotifyWaitOnThread(mainThread, channel.handle, 0, CUSTOM_TIMEOUT));
        CHK_RET(HcommReadOnThread(mainThread, channel.handle, param.outputPtr, source, recvBytes));
        CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, channel.handle, 1));
        return HCCL_SUCCESS;
    }

    // One staging copy makes all destination slices available concurrently.
    CHK_RET(HcommLocalCopyOnThread(mainThread, resCtx.localBuffer.addr, param.inputPtr,
        recvBytes * param.rankSize));
    if (resCtx.workerCount != 0) {
        const uint32_t auxCount = ScatterSmallAuxWorkerCount(resCtx.workerCount);
        // Workers cannot publish READY until the complete staging copy finishes.
        for (uint32_t worker = 0; worker < auxCount; ++worker) {
            CHK_RET(HcommThreadNotifyRecordOnThread(mainThread,
                resCtx.threads[worker + 1], WORKER_START_NOTIFY));
        }
        for (uint32_t worker = 0; worker < auxCount; ++worker) {
            CHK_RET(ScheduleSmallWorker(param, resCtx, worker));
        }
        // Keep four logical groups. Main owns the final, three-peer group and
        // needs no START/DONE handshake with itself. Other groups are queued first.
        CHK_RET(ScheduleSmallPeerGroup(param, resCtx, mainThread, auxCount));
        for (uint32_t worker = 0; worker < auxCount; ++worker) {
            CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, worker + 1, CUSTOM_TIMEOUT));
        }
        // All peer reads complete before root output or the next buffer reuse.
        return CopyRootOutput(param, mainThread, recvBytes);
    }
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer != param.root) {
            CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, FindChannel(resCtx, peer)->handle, 0));
        }
    }
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer != param.root) {
            CHK_RET(HcommChannelNotifyWaitOnThread(mainThread, FindChannel(resCtx, peer)->handle,
                1, CUSTOM_TIMEOUT));
        }
    }
    return CopyRootOutput(param, mainThread, recvBytes);
}

HcclResult ScheduleSmallTreeChildren(const AlgResourceCtx &resCtx, ThreadHandle thread,
    uint32_t group, uint32_t leader)
{
    const uint32_t begin = group * SCATTER_SMALL_GROUP_RANKS;
    for (uint32_t index = begin; index < begin + SCATTER_SMALL_GROUP_RANKS; ++index) {
        const uint32_t peer = resCtx.smallRanks[index];
        if (peer != leader) {
            CHK_RET(HcommChannelNotifyRecordOnThread(thread, FindChannel(resCtx, peer)->handle, 0));
        }
    }
    // Publish all three READYs before waiting for any child. Every edge uses
    // READY -> Read -> ACK on the same channel, including when root changes.
    for (uint32_t index = begin; index < begin + SCATTER_SMALL_GROUP_RANKS; ++index) {
        const uint32_t peer = resCtx.smallRanks[index];
        if (peer != leader) {
            CHK_RET(HcommChannelNotifyWaitOnThread(thread, FindChannel(resCtx, peer)->handle,
                1, CUSTOM_TIMEOUT));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CopySmallLeaderOutput(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t recvBytes, uint32_t localIndex)
{
    const auto *source = static_cast<const char *>(resCtx.localBuffer.addr) +
        static_cast<uint64_t>(localIndex) * recvBytes;
    if (source == param.outputPtr) {
        return HCCL_SUCCESS;
    }
    const auto sourceAddress = reinterpret_cast<std::uintptr_t>(source);
    const auto outputAddress = reinterpret_cast<std::uintptr_t>(param.outputPtr);
    const bool overlap = sourceAddress < outputAddress ? outputAddress - sourceAddress < recvBytes :
        sourceAddress - outputAddress < recvBytes;
    if (overlap) {
        // LocalCopy need not provide memmove semantics. After child ACKs, read
        // this slice from root instead; root still retains it until our final ACK.
        const ChannelInfo &rootChannel = *FindChannel(resCtx, param.root);
        const auto *rootSource = static_cast<const char *>(rootChannel.remoteCclMem.addr) +
            static_cast<uint64_t>(param.myRank) * recvBytes;
        return static_cast<HcclResult>(HcommReadOnThread(resCtx.aicpuThread, rootChannel.handle,
            param.outputPtr, rootSource, recvBytes));
    }
    return static_cast<HcclResult>(HcommLocalCopyOnThread(resCtx.aicpuThread,
        param.outputPtr, source, recvBytes));
}

HcclResult SmallTreeLeaderReceive(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t recvBytes, uint32_t group)
{
    const ThreadHandle thread = resCtx.aicpuThread;
    const ChannelInfo &rootChannel = *FindChannel(resCtx, param.root);
    const auto *rootBuffer = static_cast<const char *>(rootChannel.remoteCclMem.addr);
    auto *localBuffer = static_cast<char *>(resCtx.localBuffer.addr);
    const uint32_t begin = group * SCATTER_SMALL_GROUP_RANKS;
    const uint32_t end = begin + SCATTER_SMALL_GROUP_RANKS;
    CHK_RET(HcommChannelNotifyWaitOnThread(thread, rootChannel.handle, 0, CUSTOM_TIMEOUT));
    for (uint32_t index = begin; index < end;) {
        uint32_t next = index + 1;
        // Merge only consecutive GLOBAL ranks; server members may be interleaved.
        while (next < end && resCtx.smallRanks[next] == resCtx.smallRanks[next - 1] + 1) {
            ++next;
        }
        CHK_RET(HcommReadOnThread(thread, rootChannel.handle,
            localBuffer + static_cast<uint64_t>(index - begin) * recvBytes,
            rootBuffer + static_cast<uint64_t>(resCtx.smallRanks[index]) * recvBytes,
            static_cast<uint64_t>(next - index) * recvBytes));
        index = next;
    }
    CHK_RET(ScheduleSmallTreeChildren(resCtx, thread, group, param.myRank));
    // No child can still read the packed buffer when an aliased output is written.
    const uint32_t localIndex = RankIndex(resCtx.smallRanks, param.myRank) - begin;
    CHK_RET(CopySmallLeaderOutput(param, resCtx, recvBytes, localIndex));
    // Aggregate completion protects root CCL until the entire subtree is done.
    return static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, rootChannel.handle, 1));
}

HcclResult ScheduleSmallTreeWorker(const OpParam &param, const AlgResourceCtx &resCtx,
    uint32_t rootGroup, uint32_t worker)
{
    const ThreadHandle thread = resCtx.threads[worker + 1];
    CHK_RET(HcommThreadNotifyWaitOnThread(thread, WORKER_START_NOTIFY, CUSTOM_TIMEOUT));
    // Start every other group before waiting: the three leaders still receive
    // and forward their data concurrently, while root uses only one auxiliary TS.
    for (uint32_t group = 0; group < SCATTER_SMALL_WORKERS; ++group) {
        if (group != rootGroup) {
            const uint32_t leader = ScatterSmallGroupLeader(resCtx.smallRanks, group, param.root);
            const ChannelInfo &channel = *FindChannel(resCtx, leader);
            CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, 0));
        }
    }
    for (uint32_t group = 0; group < SCATTER_SMALL_WORKERS; ++group) {
        if (group != rootGroup) {
            const uint32_t leader = ScatterSmallGroupLeader(resCtx.smallRanks, group, param.root);
            const ChannelInfo &channel = *FindChannel(resCtx, leader);
            CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, 1, CUSTOM_TIMEOUT));
        }
    }
    return static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(thread, resCtx.aicpuThread, worker + 1));
}

HcclResult SmallTreeScatter(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t recvBytes)
{
    const ThreadHandle mainThread = resCtx.aicpuThread;
    const uint32_t group = ScatterSmallGroupIndex(resCtx.smallRanks, param.myRank);
    const uint32_t leader = ScatterSmallGroupLeader(resCtx.smallRanks, group, param.root);
    if (param.myRank != param.root) {
        if (param.myRank == leader) {
            return SmallTreeLeaderReceive(param, resCtx, recvBytes, group);
        }
        const ChannelInfo &channel = *FindChannel(resCtx, leader);
        const uint32_t slice = leader == param.root ? param.myRank :
            RankIndex(resCtx.smallRanks, param.myRank) - group * SCATTER_SMALL_GROUP_RANKS;
        const auto *source = static_cast<const char *>(channel.remoteCclMem.addr) +
            static_cast<uint64_t>(slice) * recvBytes;
        CHK_RET(HcommChannelNotifyWaitOnThread(mainThread, channel.handle, 0, CUSTOM_TIMEOUT));
        CHK_RET(HcommReadOnThread(mainThread, channel.handle, param.outputPtr, source, recvBytes));
        return static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(mainThread, channel.handle, 1));
    }
    CHK_RET(HcommLocalCopyOnThread(mainThread, resCtx.localBuffer.addr, param.inputPtr,
        recvBytes * param.rankSize));
    const uint32_t auxCount = ScatterSmallAuxWorkerCount(resCtx.workerCount, resCtx.smallMode);
    for (uint32_t worker = 0; worker < auxCount; ++worker) {
        CHK_RET(HcommThreadNotifyRecordOnThread(mainThread,
            resCtx.threads[worker + 1], WORKER_START_NOTIFY));
    }
    for (uint32_t worker = 0; worker < auxCount; ++worker) {
        CHK_RET(ScheduleSmallTreeWorker(param, resCtx, group, worker));
    }
    CHK_RET(ScheduleSmallTreeChildren(resCtx, mainThread, group, param.root));
    for (uint32_t worker = 0; worker < auxCount; ++worker) {
        CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, worker + 1, CUSTOM_TIMEOUT));
    }
    return CopyRootOutput(param, mainThread, recvBytes);
}

HcclResult LargeReceive(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t recvBytes)
{
    const ChannelInfo &channel = *FindChannel(resCtx, param.root);
    const uint32_t peerIndex = param.myRank < param.root ? param.myRank : param.myRank - 1;
    const uint32_t worker = peerIndex % resCtx.workerCount;
    const auto *remoteBuffer = static_cast<const char *>(channel.remoteCclMem.addr);
    auto *output = static_cast<char *>(param.outputPtr);
    uint64_t chunkIndex = 0;
    for (uint64_t offset = 0; offset < recvBytes; ++chunkIndex) {
        const uint32_t slot = static_cast<uint32_t>(chunkIndex % SLOT_COUNT);
        const uint64_t slotOffset = (static_cast<uint64_t>(worker) * SLOT_COUNT + slot) * resCtx.chunkBytes;
        const uint64_t bytes = std::min(resCtx.chunkBytes, recvBytes - offset);
        CHK_RET(HcommChannelNotifyWaitOnThread(resCtx.aicpuThread, channel.handle, slot * 2, CUSTOM_TIMEOUT));
        CHK_RET(HcommReadOnThread(resCtx.aicpuThread, channel.handle, output + offset,
            remoteBuffer + slotOffset, bytes));
        CHK_RET(HcommChannelNotifyRecordOnThread(resCtx.aicpuThread, channel.handle, slot * 2 + 1));
        offset += bytes;
    }
    return HCCL_SUCCESS;
}

HcclResult ScheduleWorker(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t recvBytes, uint32_t worker)
{
    const ThreadHandle thread = resCtx.threads[worker + 1];
    CHK_RET(HcommThreadNotifyWaitOnThread(thread, WORKER_START_NOTIFY, CUSTOM_TIMEOUT));
    const auto *input = static_cast<const char *>(param.inputPtr);
    auto *localBuffer = static_cast<char *>(resCtx.localBuffer.addr);
    const uint64_t chunkCount = (recvBytes - 1) / resCtx.chunkBytes + 1;
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer == param.root) {
            continue;
        }
        const uint32_t peerIndex = peer < param.root ? peer : peer - 1;
        if (peerIndex % resCtx.workerCount != worker) {
            continue;
        }
        const ChannelInfo &channel = *FindChannel(resCtx, peer);
        const uint64_t peerOffset = static_cast<uint64_t>(peer) * recvBytes;
        uint64_t chunkIndex = 0;
        for (uint64_t offset = 0; offset < recvBytes; ++chunkIndex) {
            const uint32_t slot = static_cast<uint32_t>(chunkIndex % SLOT_COUNT);
            // Two credits permit staging the next chunk while the current read
            // is outstanding. Reuse a slot only after its earlier read is ACKed.
            if (chunkIndex >= SLOT_COUNT) {
                CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, slot * 2 + 1, CUSTOM_TIMEOUT));
            }
            const uint64_t slotOffset = (static_cast<uint64_t>(worker) * SLOT_COUNT + slot) * resCtx.chunkBytes;
            const uint64_t bytes = std::min(resCtx.chunkBytes, recvBytes - offset);
            CHK_RET(HcommLocalCopyOnThread(thread, localBuffer + slotOffset, input + peerOffset + offset, bytes));
            CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, slot * 2));
            offset += bytes;
        }
        // Drain both outstanding slots before another peer can use this lane.
        const uint64_t firstPending = chunkCount > SLOT_COUNT ? chunkCount - SLOT_COUNT : 0;
        for (uint64_t chunk = firstPending; chunk < chunkCount; ++chunk) {
            const uint32_t slot = static_cast<uint32_t>(chunk % SLOT_COUNT);
            CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, slot * 2 + 1, CUSTOM_TIMEOUT));
        }
    }
    return static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(thread, resCtx.aicpuThread, worker + 1));
}

HcclResult LargeScatter(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t recvBytes)
{
    if (param.myRank != param.root) {
        return LargeReceive(param, resCtx, recvBytes);
    }
    // Fork every TS worker before joining any. Scheduling these streams does
    // not create CPU threads; their explicit handles carry the parallel work.
    for (uint32_t worker = 0; worker < resCtx.workerCount; ++worker) {
        CHK_RET(HcommThreadNotifyRecordOnThread(resCtx.aicpuThread, resCtx.threads[worker + 1], WORKER_START_NOTIFY));
    }
    for (uint32_t worker = 0; worker < resCtx.workerCount; ++worker) {
        CHK_RET(ScheduleWorker(param, resCtx, recvBytes, worker));
    }
    for (uint32_t worker = 0; worker < resCtx.workerCount; ++worker) {
        CHK_RET(HcommThreadNotifyWaitOnThread(resCtx.aicpuThread, worker + 1, CUSTOM_TIMEOUT));
    }
    // All remote reads finish before an in-place root output can overwrite input.
    return CopyRootOutput(param, resCtx.aicpuThread, recvBytes);
}

struct HybridRange {
    uint32_t sourceRank;
    uint64_t sourceOffset;
    uint64_t bytes;
};

std::vector<HybridRange> MakeHybridRanges(const OpParam &param, const AlgResourceCtx &resCtx,
    uint32_t peer, uint64_t recvBytes)
{
    // Round only the tail down to complete FP32 elements. The prefix owns the
    // remainder, so the two output regions cover recvCount without overlap.
    const uint32_t relayIndex = RankIndex(resCtx.relayRanks, peer);
    if (relayIndex != INVALID_VALUE_RANKID) {
        // Stage all child tails before this relay's own slice. Each child owns
        // a distinct weighted tail and an independent downstream slot pair.
        const uint32_t childCount = ScatterRelayChildCount(relayIndex);
        std::vector<HybridRange> ranges;
        ranges.reserve(childCount + 1);
        for (uint32_t child = 0; child < childCount; ++child) {
            const uint32_t remoteIndex = ScatterRelayRemoteIndex(relayIndex, child);
            const uint64_t tailBytes = ScatterTailCount(param.count, remoteIndex) * sizeof(float);
            ranges.push_back({resCtx.remoteRanks[remoteIndex], recvBytes - tailBytes, tailBytes});
        }
        ranges.push_back({peer, 0, recvBytes});
        return ranges;
    }
    const uint32_t remoteIndex = RankIndex(resCtx.remoteRanks, peer);
    if (remoteIndex != INVALID_VALUE_RANKID) {
        const uint64_t tailBytes = ScatterTailCount(param.count, remoteIndex) * sizeof(float);
        return {{peer, 0, recvBytes - tailBytes}};
    }
    return {{peer, 0, recvBytes}};
}

HcclResult ScheduleHybridWorker(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t recvBytes, uint32_t worker)
{
    const ThreadHandle thread = resCtx.threads[worker + 1];
    CHK_RET(HcommThreadNotifyWaitOnThread(thread, WORKER_START_NOTIFY, CUSTOM_TIMEOUT));
    const auto *input = static_cast<const char *>(param.inputPtr);
    auto *localBuffer = static_cast<char *>(resCtx.localBuffer.addr);
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer == param.root || RootWorker(param, resCtx, peer) != worker) {
            continue;
        }
        const ChannelInfo &channel = *FindChannel(resCtx, peer);
        const bool push = ScatterUsePush(resCtx.routeMode, channel);
        if (push) {
            // A receiver can enter this call after its own previous work has
            // drained, while other ranks are still finishing that call. Do not
            // overwrite its CCL until it explicitly grants the new invocation.
            CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle,
                SCATTER_PUSH_INIT_NOTIFY, CUSTOM_TIMEOUT));
        }
        const auto ranges = MakeHybridRanges(param, resCtx, peer, recvBytes);
        uint64_t chunkIndex = 0;
        for (const auto &range : ranges) {
            const uint64_t sourceBase = static_cast<uint64_t>(range.sourceRank) * recvBytes + range.sourceOffset;
            // Keep a single credit sequence across all child tails and own slice;
            // a segment can end in a partial chunk or an odd numbered slot.
            for (uint64_t offset = 0; offset < range.bytes; ++chunkIndex) {
                const uint32_t slot = static_cast<uint32_t>(chunkIndex % SLOT_COUNT);
                if (chunkIndex >= SLOT_COUNT) {
                    CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle,
                        slot * 2 + 1, CUSTOM_TIMEOUT));
                }
                const uint64_t slotOffset = (static_cast<uint64_t>(worker) * SLOT_COUNT + slot) *
                    resCtx.chunkBytes;
                const uint64_t bytes = std::min(resCtx.chunkBytes, range.bytes - offset);
                if (push) {
                    auto *incoming = static_cast<char *>(channel.remoteCclMem.addr) +
                        static_cast<uint64_t>(slot) * resCtx.chunkBytes;
                    CHK_RET(HcommWriteWithNotifyOnThread(thread, channel.handle, incoming,
                        input + sourceBase + offset, bytes, slot * 2));
                } else {
                    // Unreviewed protocols retain the original first-hop Read
                    // path. Both endpoints use this established channel's protocol.
                    CHK_RET(HcommLocalCopyOnThread(thread, localBuffer + slotOffset,
                        input + sourceBase + offset, bytes));
                    CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, slot * 2));
                }
                offset += bytes;
            }
        }
        const uint64_t firstPending = chunkIndex > SLOT_COUNT ? chunkIndex - SLOT_COUNT : 0;
        for (uint64_t chunk = firstPending; chunk < chunkIndex; ++chunk) {
            const uint32_t slot = static_cast<uint32_t>(chunk % SLOT_COUNT);
            CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, slot * 2 + 1, CUSTOM_TIMEOUT));
        }
    }
    return static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(thread, resCtx.aicpuThread, worker + 1));
}

HcclResult ReceiveHybridRange(const OpParam &param, const AlgResourceCtx &resCtx,
    ThreadHandle thread, uint32_t sourceRank, uint32_t lane, uint64_t outputOffset, uint64_t bytes)
{
    const ChannelInfo &channel = *FindChannel(resCtx, sourceRank);
    const bool push = sourceRank == param.root && ScatterUsePush(resCtx.routeMode, channel);
    const auto *remoteBuffer = static_cast<const char *>(channel.remoteCclMem.addr);
    auto *output = static_cast<char *>(param.outputPtr);
    uint64_t chunkIndex = 0;
    for (uint64_t offset = 0; offset < bytes; ++chunkIndex) {
        const uint32_t slot = static_cast<uint32_t>(chunkIndex % SLOT_COUNT);
        const uint64_t slotOffset = (static_cast<uint64_t>(lane) * SLOT_COUNT + slot) * resCtx.chunkBytes;
        const uint64_t chunkBytes = std::min(resCtx.chunkBytes, bytes - offset);
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, slot * 2, CUSTOM_TIMEOUT));
        if (push) {
            const auto *incoming = static_cast<const char *>(resCtx.localBuffer.addr) +
                static_cast<uint64_t>(slot) * resCtx.chunkBytes;
            CHK_RET(HcommLocalCopyOnThread(thread, output + outputOffset + offset, incoming, chunkBytes));
        } else {
            CHK_RET(HcommReadOnThread(thread, channel.handle, output + outputOffset + offset,
                remoteBuffer + slotOffset, chunkBytes));
        }
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, slot * 2 + 1));
        offset += chunkBytes;
    }
    return HCCL_SUCCESS;
}

HcclResult HybridRemoteReceive(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t recvBytes, uint32_t remoteIndex)
{
    const ThreadHandle mainThread = resCtx.aicpuThread;
    const ThreadHandle tailThread = resCtx.threads[1];
    const uint64_t tailBytes = ScatterTailCount(param.count, remoteIndex) * sizeof(float);
    const uint64_t prefixBytes = recvBytes - tailBytes;
    CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, tailThread, WORKER_START_NOTIFY));
    CHK_RET(HcommThreadNotifyWaitOnThread(tailThread, WORKER_START_NOTIFY, CUSTOM_TIMEOUT));
    CHK_RET(ReceiveHybridRange(param, resCtx, tailThread,
        resCtx.relayRanks[ScatterRemoteRelayIndex(remoteIndex)],
        ScatterChildLane(resCtx.routeMode, ScatterRemoteChildIndex(remoteIndex)), prefixBytes, tailBytes));
    CHK_RET(HcommThreadNotifyRecordOnThread(tailThread, mainThread, REMOTE_TAIL_DONE_NOTIFY));
    // Enqueue the complete tail flow on its own TS thread, then the root prefix
    // on main. They execute concurrently and write disjoint output regions.
    CHK_RET(ReceiveHybridRange(param, resCtx, mainThread, param.root,
        RootWorker(param, resCtx, param.myRank), 0, prefixBytes));
    return static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(mainThread,
        REMOTE_TAIL_DONE_NOTIFY, CUSTOM_TIMEOUT));
}

HcclResult RelayReadUpstream(const AlgResourceCtx &resCtx, const ChannelInfo &rootChannel,
    uint32_t rootWorker, uint64_t chunkIndex, uint64_t totalChunks, void *destination, uint64_t bytes)
{
    const uint32_t slot = static_cast<uint32_t>(chunkIndex % SLOT_COUNT);
    const uint64_t slotOffset = (static_cast<uint64_t>(rootWorker) * SLOT_COUNT + slot) * resCtx.chunkBytes;
    const auto *source = static_cast<const char *>(rootChannel.remoteCclMem.addr) + slotOffset;
    CHK_RET(HcommChannelNotifyWaitOnThread(resCtx.aicpuThread, rootChannel.handle,
        slot * 2, CUSTOM_TIMEOUT));
    if (ScatterUsePush(resCtx.routeMode, rootChannel)) {
        const auto *incoming = static_cast<const char *>(resCtx.localBuffer.addr) +
            static_cast<uint64_t>(slot) * resCtx.chunkBytes;
        CHK_RET(HcommLocalCopyOnThread(resCtx.aicpuThread, destination, incoming, bytes));
    } else {
        CHK_RET(HcommReadOnThread(resCtx.aicpuThread, rootChannel.handle, destination, source, bytes));
    }
    if (chunkIndex + 1 < totalChunks) {
        // Free the incoming slot after LocalCopy, or root's slot after Read.
        // Downstream readers use disjoint child slots in either path.
        CHK_RET(HcommChannelNotifyRecordOnThread(resCtx.aicpuThread, rootChannel.handle, slot * 2 + 1));
    }
    return HCCL_SUCCESS;
}

HcclResult HybridRelayReceive(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t recvBytes, uint32_t relayIndex)
{
    const ThreadHandle thread = resCtx.aicpuThread;
    const ChannelInfo &rootChannel = *FindChannel(resCtx, param.root);
    const uint32_t rootWorker = RootWorker(param, resCtx, param.myRank);
    const uint32_t childCount = ScatterRelayChildCount(relayIndex);
    // At most two children; zero-length tails consume no slots or notifications.
    uint64_t childTailBytes[2] = {};
    uint64_t childTailChunks[2] = {};
    uint64_t totalChunks = ChunkCount(recvBytes, resCtx.chunkBytes);
    for (uint32_t child = 0; child < childCount; ++child) {
        const uint32_t remoteIndex = ScatterRelayRemoteIndex(relayIndex, child);
        childTailBytes[child] = ScatterTailCount(param.count, remoteIndex) * sizeof(float);
        childTailChunks[child] = ChunkCount(childTailBytes[child], resCtx.chunkBytes);
        // Each segment rounds up independently, including a partial final slot.
        totalChunks += childTailChunks[child];
    }
    auto *localBuffer = static_cast<char *>(resCtx.localBuffer.addr);
    uint64_t upstreamChunk = 0;
    for (uint32_t child = 0; child < childCount; ++child) {
        const uint32_t remoteIndex = ScatterRelayRemoteIndex(relayIndex, child);
        const ChannelInfo &leafChannel = *FindChannel(resCtx, resCtx.remoteRanks[remoteIndex]);
        const uint64_t tailBytes = childTailBytes[child];
        uint64_t childChunk = 0;
        for (uint64_t offset = 0; offset < tailBytes; ++childChunk, ++upstreamChunk) {
            const uint32_t slot = static_cast<uint32_t>(childChunk % SLOT_COUNT);
            if (childChunk >= SLOT_COUNT) {
                CHK_RET(HcommChannelNotifyWaitOnThread(thread, leafChannel.handle,
                    slot * 2 + 1, CUSTOM_TIMEOUT));
            }
            const uint64_t slotOffset = (static_cast<uint64_t>(ScatterChildLane(resCtx.routeMode, child)) *
                SLOT_COUNT + slot) * resCtx.chunkBytes;
            const uint64_t bytes = std::min(resCtx.chunkBytes, tailBytes - offset);
            CHK_RET(RelayReadUpstream(resCtx, rootChannel, rootWorker, upstreamChunk,
                totalChunks, localBuffer + slotOffset, bytes));
            // Each child's two slots are independent; issue READY after this
            // chunk, without waiting for the rest of any child's tail.
            CHK_RET(HcommChannelNotifyRecordOnThread(thread, leafChannel.handle, slot * 2));
            offset += bytes;
        }
    }
    auto *output = static_cast<char *>(param.outputPtr);
    for (uint64_t offset = 0; offset < recvBytes; ++upstreamChunk) {
        const uint64_t bytes = std::min(resCtx.chunkBytes, recvBytes - offset);
        CHK_RET(RelayReadUpstream(resCtx, rootChannel, rootWorker, upstreamChunk,
            totalChunks, output + offset, bytes));
        offset += bytes;
    }
    for (uint32_t child = 0; child < childCount; ++child) {
        const uint32_t remoteIndex = ScatterRelayRemoteIndex(relayIndex, child);
        const ChannelInfo &leafChannel = *FindChannel(resCtx, resCtx.remoteRanks[remoteIndex]);
        const uint64_t tailChunks = childTailChunks[child];
        const uint64_t firstPending = tailChunks > SLOT_COUNT ? tailChunks - SLOT_COUNT : 0;
        for (uint64_t chunk = firstPending; chunk < tailChunks; ++chunk) {
            const uint32_t slot = static_cast<uint32_t>(chunk % SLOT_COUNT);
            CHK_RET(HcommChannelNotifyWaitOnThread(thread, leafChannel.handle, slot * 2 + 1, CUSTOM_TIMEOUT));
        }
    }
    // Hold exactly the final root ACK until all child flows are drained.
    // Root's worker join therefore still protects all input and relay slots,
    // including an in-place root output and the next invocation's buffer reuse.
    const uint32_t finalSlot = static_cast<uint32_t>((totalChunks - 1) % SLOT_COUNT);
    return static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread,
        rootChannel.handle, finalSlot * 2 + 1));
}

HcclResult HybridScatter(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t recvBytes)
{
    if (param.myRank != param.root) {
        const ChannelInfo &rootChannel = *FindChannel(resCtx, param.root);
        if (ScatterUsePush(resCtx.routeMode, rootChannel)) {
            // Dedicated INIT avoids a next-call credit overwriting an unconsumed
            // final ACK from the previous call. Publish before any READY wait.
            CHK_RET(HcommChannelNotifyRecordOnThread(resCtx.aicpuThread, rootChannel.handle,
                SCATTER_PUSH_INIT_NOTIFY));
        }
        const uint32_t remoteIndex = RankIndex(resCtx.remoteRanks, param.myRank);
        if (remoteIndex != INVALID_VALUE_RANKID) {
            return HybridRemoteReceive(param, resCtx, recvBytes, remoteIndex);
        }
        const uint32_t relayIndex = RankIndex(resCtx.relayRanks, param.myRank);
        if (relayIndex != INVALID_VALUE_RANKID) {
            return HybridRelayReceive(param, resCtx, recvBytes, relayIndex);
        }
        return LargeReceive(param, resCtx, recvBytes);
    }
    for (uint32_t worker = 0; worker < resCtx.workerCount; ++worker) {
        CHK_RET(HcommThreadNotifyRecordOnThread(resCtx.aicpuThread,
            resCtx.threads[worker + 1], WORKER_START_NOTIFY));
    }
    for (uint32_t worker = 0; worker < resCtx.workerCount; ++worker) {
        CHK_RET(ScheduleHybridWorker(param, resCtx, recvBytes, worker));
    }
    for (uint32_t worker = 0; worker < resCtx.workerCount; ++worker) {
        CHK_RET(HcommThreadNotifyWaitOnThread(resCtx.aicpuThread, worker + 1, CUSTOM_TIMEOUT));
    }
    return CopyRootOutput(param, resCtx.aicpuThread, recvBytes);
}
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    CHK_PRT_RET(param.opType != HcclCMDType::HCCL_CMD_SCATTER || param.dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("Scatter requires the FP32 Scatter operation"), HCCL_E_PARA);
    CHK_PRT_RET(param.rankSize == 0 || param.myRank >= param.rankSize || param.root >= param.rankSize,
        HCCL_ERROR("Invalid Scatter rank information"), HCCL_E_PARA);
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / sizeof(float) / param.rankSize,
        HCCL_ERROR("Scatter input byte count overflows uint64_t"), HCCL_E_PARA);

    const uint64_t recvBytes = param.count * sizeof(float);
    if (recvBytes == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(param.outputPtr);
    if (param.myRank == param.root) {
        CHK_PTR_NULL(param.inputPtr);
    }
    CHK_RET(ValidateResources(param, resCtx, recvBytes));
    if (param.rankSize == 1) {
        return CopyRootOutput(param, resCtx.aicpuThread, recvBytes);
    }
    if (resCtx.chunkBytes == 0) {
        return resCtx.smallMode == SCATTER_SMALL_TREE ? SmallTreeScatter(param, resCtx, recvBytes) :
            SmallScatter(param, resCtx, recvBytes);
    }
    if (ScatterIsHybridRoute(resCtx.routeMode)) {
        return HybridScatter(param, resCtx, recvBytes);
    }
    return LargeScatter(param, resCtx, recvBytes);
}
} // namespace ops_hccl
