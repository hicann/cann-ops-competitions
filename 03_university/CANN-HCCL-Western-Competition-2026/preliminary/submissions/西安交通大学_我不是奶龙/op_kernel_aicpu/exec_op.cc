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
constexpr uint64_t MAX_SINGLE_TASK_BYTES = 256ULL * 1024 * 1024;
constexpr uint64_t MESH_MESSAGE_THRESHOLD_BYTES = 256ULL * 1024;
constexpr uint64_t SMALL_ONE_SHOT_MAX_BYTES = 1ULL * 1024 * 1024;
constexpr uint64_t HALVING_DOUBLING_MESSAGE_BYTES = 512ULL * 1024;
constexpr uint64_t LARGE_MESH_CHUNK_THRESHOLD_BYTES = 64ULL * 1024 * 1024;
constexpr uint64_t SLOT_ALIGNMENT_BYTES = 128;
constexpr uint32_t DATA_NOTIFY = 0;
constexpr uint32_t ACK_NOTIFY = 1;
constexpr uint32_t READY_NOTIFY = 2;
constexpr uint32_t HD_ALL_GATHER_NOTIFY = 3;
constexpr uint32_t SLAVE_RS_START_NOTIFY = 0;
constexpr uint32_t SLAVE_AG_START_NOTIFY = 1;
constexpr uint32_t CONTEST_RANK_SIZE = 16;
constexpr uint32_t SERVER_SIZE = 8;
constexpr uint32_t LOCAL_PEER_NUM = SERVER_SIZE - 1;
constexpr uint32_t MESH_THREAD_NUM = 7;
constexpr uint32_t MESH_SLAVE_NUM = MESH_THREAD_NUM - 1;
constexpr uint32_t CROSS_AG_THREAD_BEGIN = MESH_THREAD_NUM;
constexpr uint32_t CROSS_AG_THREAD_NUM = LOCAL_PEER_NUM;
constexpr uint32_t COPY_THREAD_INDEX = CROSS_AG_THREAD_BEGIN + CROSS_AG_THREAD_NUM;
constexpr uint32_t CONTEST_THREAD_NUM = COPY_THREAD_INDEX + 1;
constexpr uint32_t LARGE_SLOT_NUM = 2;
constexpr uint32_t SMALL_SLOT_NUM = SERVER_SIZE + 1;
constexpr uint64_t FP32_ALIGNMENT_COUNT = SLOT_ALIGNMENT_BYTES / sizeof(float);

using LocalPeerRanks = std::array<uint32_t, SERVER_SIZE - 1>;
using LocalChannels = std::array<const ChannelInfo *, SERVER_SIZE - 1>;
using CrossPeerRanks = std::array<uint32_t, SERVER_SIZE - 1>;
using CrossChannels = std::array<const ChannelInfo *, SERVER_SIZE - 1>;
using ChannelTable = std::array<const ChannelInfo *, CONTEST_RANK_SIZE>;

uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return value / divisor + (value % divisor != 0 ? 1 : 0);
}

uint64_t AlignDown(uint64_t value, uint64_t alignment)
{
    return value - value % alignment;
}

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return CeilDiv(value, alignment) * alignment;
}

HcclResult BuildChannelTable(
    const AlgResourceCtx &resCtx, ChannelTable &channelTable, uint64_t &bufferBytes)
{
    CHK_PRT_RET(resCtx.localBuffer.addr == nullptr, HCCL_ERROR("Local CCL buffer is null"), HCCL_E_INTERNAL);
    channelTable.fill(nullptr);
    bufferBytes = resCtx.localBuffer.size;
    for (const auto &channel : resCtx.channels) {
        CHK_PRT_RET(channel.remoteCclMem.addr == nullptr,
            HCCL_ERROR("Remote CCL buffer for rank[%u] is null", channel.remoteRank), HCCL_E_INTERNAL);
        CHK_PRT_RET(channel.remoteRank >= channelTable.size(),
            HCCL_ERROR("Remote rank[%u] exceeds the contest rank table", channel.remoteRank), HCCL_E_INTERNAL);
        CHK_PRT_RET(channelTable[channel.remoteRank] != nullptr,
            HCCL_ERROR("Duplicate channel to rank[%u]", channel.remoteRank),
            HCCL_E_INTERNAL);
        channelTable[channel.remoteRank] = &channel;
        bufferBytes = std::min(bufferBytes, channel.remoteCclMem.size);
    }
    CHK_PRT_RET(bufferBytes < 2 * sizeof(float), HCCL_ERROR("CCL buffer is too small, size[%llu]",
        static_cast<unsigned long long>(bufferBytes)), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

const ChannelInfo *FindChannel(const ChannelTable &channelTable, uint32_t remoteRank)
{
    return remoteRank < channelTable.size() ? channelTable[remoteRank] : nullptr;
}

uint64_t SliceCount(uint64_t windowCount, uint32_t slice)
{
    return windowCount / SERVER_SIZE + (slice < windowCount % SERVER_SIZE ? 1 : 0);
}

uint64_t SliceOffset(uint64_t windowCount, uint32_t slice)
{
    return slice * (windowCount / SERVER_SIZE) + std::min<uint64_t>(slice, windowCount % SERVER_SIZE);
}

uint64_t AlignedBoundary(uint64_t total, uint32_t index, uint32_t parts)
{
    if (index == 0) {
        return 0;
    }
    if (index >= parts) {
        return total;
    }

    uint64_t quotient = total / parts;
    uint64_t remainder = total % parts;
    uint64_t ideal = static_cast<uint64_t>(index) * quotient +
        CeilDiv(static_cast<uint64_t>(index) * remainder, parts);
    return std::min(total, AlignUp(ideal, FP32_ALIGNMENT_COUNT));
}

uint64_t MeshSliceOffset(uint64_t windowCount, uint32_t slice)
{
    return AlignedBoundary(windowCount, slice, SERVER_SIZE);
}

uint64_t MeshSliceCount(uint64_t windowCount, uint32_t slice)
{
    return MeshSliceOffset(windowCount, slice + 1) - MeshSliceOffset(windowCount, slice);
}

uint64_t MeshStripeOffset(uint64_t sliceCount, uint32_t stripe)
{
    return AlignedBoundary(sliceCount, stripe, LOCAL_PEER_NUM);
}

uint64_t MeshStripeCount(uint64_t sliceCount, uint32_t stripe)
{
    return MeshStripeOffset(sliceCount, stripe + 1) - MeshStripeOffset(sliceCount, stripe);
}

uint64_t DualPathLocalPrefixCount(uint64_t sliceCount)
{
    return AlignedBoundary(sliceCount, LOCAL_PEER_NUM, CONTEST_RANK_SIZE - 1);
}

uint8_t *SlotAddress(uint8_t *buffer, uint64_t slotStrideBytes, uint32_t slot)
{
    return buffer + slot * slotStrideBytes;
}

HcclResult WriteAndRecordData(
    ThreadHandle thread, const ChannelInfo &channel, void *remoteDst, const void *localSrc, uint64_t bytes)
{
    if (bytes > 0) {
        CHK_RET(static_cast<HcclResult>(HcommWriteOnThread(thread, channel.handle, remoteDst, localSrc, bytes)));
    }
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, DATA_NOTIFY)));
    return HCCL_SUCCESS;
}

HcclResult WriteWithNotify(
    ThreadHandle thread, const ChannelInfo &channel, void *remoteDst, const void *localSrc, uint64_t bytes)
{
    if (bytes > 0) {
        CHK_RET(static_cast<HcclResult>(
            HcommWriteOnThread(thread, channel.handle, remoteDst, localSrc, bytes)));
    }
    return static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, DATA_NOTIFY));
}

HcclResult FusedWriteWithNotify(
    ThreadHandle thread, const ChannelInfo &channel, void *remoteDst, const void *localSrc,
    uint64_t bytes, uint32_t notifyIndex)
{
    if (bytes == 0) {
        return static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, channel.handle, notifyIndex));
    }
    return static_cast<HcclResult>(HcommWriteWithNotifyOnThread(
        thread, channel.handle, remoteDst, localSrc, bytes, notifyIndex));
}

HcclResult WriteReduceWithNotify(ThreadHandle thread, const ChannelInfo &channel, void *remoteDst,
    const void *localSrc, uint64_t count)
{
    if (count > 0) {
        CHK_RET(static_cast<HcclResult>(HcommWriteReduceOnThread(thread, channel.handle,
            remoteDst, localSrc, count, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    }
    return static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, DATA_NOTIFY));
}

HcclResult WaitChannelData(ThreadHandle thread, const ChannelInfo &channel)
{
    return static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, DATA_NOTIFY, CUSTOM_TIMEOUT));
}

HcclResult RecordChannelAck(ThreadHandle thread, const ChannelInfo &channel)
{
    return static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, channel.handle, ACK_NOTIFY));
}

HcclResult WaitChannelAck(ThreadHandle thread, const ChannelInfo &channel)
{
    return static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, ACK_NOTIFY, CUSTOM_TIMEOUT));
}

HcclResult PairBarrier(ThreadHandle thread, const ChannelInfo &channel)
{
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, READY_NOTIFY)));
    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyWaitOnThread(thread, channel.handle, READY_NOTIFY, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult AllChannelBarrier(ThreadHandle thread, const std::vector<ChannelInfo> &channels)
{
    // Record every edge before waiting on any edge. This avoids a wait cycle
    // when different ranks enumerate their peers in a different order.
    for (const auto &channel : channels) {
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(thread, channel.handle, READY_NOTIFY)));
    }
    for (const auto &channel : channels) {
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyWaitOnThread(thread, channel.handle, READY_NOTIFY, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult WriteAndSynchronize(ThreadHandle thread, const ChannelInfo &channel, void *remoteDst,
    const void *localSrc, uint64_t bytes)
{
    CHK_RET(WriteAndRecordData(thread, channel, remoteDst, localSrc, bytes));
    CHK_RET(WaitChannelData(thread, channel));
    return HCCL_SUCCESS;
}

HcclResult Acknowledge(ThreadHandle thread, const ChannelInfo &recordChannel, const ChannelInfo &waitChannel)
{
    CHK_RET(RecordChannelAck(thread, recordChannel));
    CHK_RET(WaitChannelAck(thread, waitChannel));
    return HCCL_SUCCESS;
}

uint64_t RecursiveWindowCapacity(uint64_t bufferBytes)
{
    uint64_t capacityBytes = std::min(MAX_SINGLE_TASK_BYTES, bufferBytes / 2);
    return capacityBytes / sizeof(float);
}

HcclResult ExecRecursiveDoubling(const OpParam &param, const AlgResourceCtx &resCtx,
    const ChannelTable &channelTable, uint64_t bufferBytes)
{
    ThreadHandle thread = resCtx.aicpuThread;
    uint64_t capacity = RecursiveWindowCapacity(bufferBytes);
    CHK_PRT_RET(capacity == 0, HCCL_ERROR("No CCL buffer capacity for recursive doubling"), HCCL_E_INTERNAL);

    uint8_t *input = static_cast<uint8_t *>(param.inputPtr);
    uint8_t *output = static_cast<uint8_t *>(param.outputPtr);
    uint8_t *accumulator = static_cast<uint8_t *>(resCtx.localBuffer.addr);

    for (uint64_t base = 0; base < param.count; base += capacity) {
        uint64_t windowCount = std::min(capacity, param.count - base);
        uint64_t windowBytes = windowCount * sizeof(float);
        uint8_t *staging = accumulator + windowBytes;

        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(thread, accumulator, input + base * sizeof(float), windowBytes)));

        for (uint32_t distance = 1; distance < param.rankSize; distance <<= 1U) {
            uint32_t partner = param.myRank ^ distance;
            const ChannelInfo *channel = FindChannel(channelTable, partner);
            CHK_PRT_RET(channel == nullptr, HCCL_ERROR("Missing channel to rank[%u]", partner), HCCL_E_INTERNAL);

            // The ready handshake prevents a new partner from overwriting the
            // single staging slot before the previous local reduction ends.
            CHK_RET(Acknowledge(thread, *channel, *channel));
            void *remoteStaging = static_cast<uint8_t *>(channel->remoteCclMem.addr) + windowBytes;
            CHK_RET(WriteAndSynchronize(thread, *channel, remoteStaging, accumulator, windowBytes));
            CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(
                thread, accumulator, staging, windowCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
        }

        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(thread, output + base * sizeof(float), accumulator, windowBytes)));
    }

    // The Mesh implementation can follow a small-message call while reusing
    // the same EngineCtx. Make every exposed CCL buffer reusable only after
    // all peers have completed their final output copy.
    CHK_RET(AllChannelBarrier(thread, resCtx.channels));
    return HCCL_SUCCESS;
}

HcclResult BuildMeshResources(const OpParam &param,
    const ChannelTable &channelTable, LocalPeerRanks &peerRanks,
    LocalChannels &localChannels, const ChannelInfo *&crossChannel)
{
    uint32_t serverBase = (param.myRank / SERVER_SIZE) * SERVER_SIZE;
    uint32_t index = 0;
    for (uint32_t localPeer = 0; localPeer < SERVER_SIZE; ++localPeer) {
        uint32_t peer = serverBase + localPeer;
        if (peer == param.myRank) {
            continue;
        }
        const ChannelInfo *channel = FindChannel(channelTable, peer);
        CHK_PRT_RET(channel == nullptr, HCCL_ERROR("Missing layer-0 Mesh channel to rank[%u]", peer),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(channel->notifyNum <= READY_NOTIFY,
            HCCL_ERROR("Channel to rank[%u] has only [%u] notifies", peer, channel->notifyNum), HCCL_E_INTERNAL);
        peerRanks[index] = peer;
        localChannels[index] = channel;
        ++index;
    }
    CHK_PRT_RET(index != SERVER_SIZE - 1, HCCL_ERROR("Invalid local Mesh peer count[%u]", index),
        HCCL_E_INTERNAL);

    uint32_t crossRank = param.myRank ^ SERVER_SIZE;
    crossChannel = FindChannel(channelTable, crossRank);
    CHK_PRT_RET(crossChannel == nullptr, HCCL_ERROR("Missing layer-1 Clos channel to rank[%u]", crossRank),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(crossChannel->notifyNum <= READY_NOTIFY,
        HCCL_ERROR("Cross channel to rank[%u] has only [%u] notifies", crossRank, crossChannel->notifyNum),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult BuildCrossAgResources(const OpParam &param, const ChannelTable &channelTable,
    CrossPeerRanks &peerRanks, CrossChannels &channels)
{
    uint32_t localRank = param.myRank % SERVER_SIZE;
    uint32_t remoteServerBase = ((param.myRank / SERVER_SIZE) ^ 1U) * SERVER_SIZE;
    uint32_t index = 0;
    for (uint32_t remoteLocalRank = 0; remoteLocalRank < SERVER_SIZE; ++remoteLocalRank) {
        if (remoteLocalRank == localRank) {
            continue;
        }
        uint32_t peer = remoteServerBase + remoteLocalRank;
        const ChannelInfo *channel = FindChannel(channelTable, peer);
        CHK_PRT_RET(channel == nullptr, HCCL_ERROR("Missing layer-1 AG channel to rank[%u]", peer),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(channel->notifyNum <= READY_NOTIFY,
            HCCL_ERROR("AG channel to rank[%u] has only [%u] notifies", peer, channel->notifyNum),
            HCCL_E_INTERNAL);
        peerRanks[index] = peer;
        channels[index] = channel;
        ++index;
    }
    CHK_PRT_RET(index != LOCAL_PEER_NUM, HCCL_ERROR("Invalid cross AG peer count[%u]", index),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

uint64_t MeshMaximumSlotStride(uint64_t bufferBytes)
{
    uint64_t stride = AlignDown(bufferBytes / SERVER_SIZE, SLOT_ALIGNMENT_BYTES);
    return std::min(stride, MAX_SINGLE_TASK_BYTES);
}

HcclResult RecordSlaveStart(ThreadHandle mainThread, const AlgResourceCtx &resCtx, uint32_t notifyIndex)
{
    for (uint32_t worker = 1; worker < MESH_THREAD_NUM; ++worker) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            mainThread, resCtx.threads[worker], notifyIndex)));
    }
    return HCCL_SUCCESS;
}

HcclResult WaitSlaveDone(ThreadHandle mainThread)
{
    for (uint32_t worker = 1; worker <= MESH_SLAVE_NUM; ++worker) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(mainThread, worker, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult RecordCrossStart(
    ThreadHandle mainThread, const AlgResourceCtx &resCtx, uint32_t notifyIndex)
{
    for (uint32_t worker = CROSS_AG_THREAD_BEGIN; worker < COPY_THREAD_INDEX; ++worker) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            mainThread, resCtx.threads[worker], notifyIndex)));
    }
    return HCCL_SUCCESS;
}

HcclResult WaitCrossDone(ThreadHandle mainThread)
{
    for (uint32_t worker = CROSS_AG_THREAD_BEGIN; worker < COPY_THREAD_INDEX; ++worker) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(mainThread, worker, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult RecordCopyStart(ThreadHandle mainThread, const AlgResourceCtx &resCtx, uint32_t notifyIndex)
{
    return static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        mainThread, resCtx.threads[COPY_THREAD_INDEX], notifyIndex));
}

HcclResult WaitCopyDone(ThreadHandle mainThread)
{
    return static_cast<HcclResult>(
        HcommThreadNotifyWaitOnThread(mainThread, COPY_THREAD_INDEX, CUSTOM_TIMEOUT));
}

HcclResult ScheduleCopyWorker(ThreadHandle mainThread, ThreadHandle copyThread,
    uint32_t startNotifyIndex, void *dst, const void *src, uint64_t bytes)
{
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyWaitOnThread(copyThread, startNotifyIndex, CUSTOM_TIMEOUT)));
    if (bytes > 0 && static_cast<const void *>(dst) != src) {
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(copyThread, dst, src, bytes)));
    }
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        copyThread, mainThread, COPY_THREAD_INDEX)));
    return HCCL_SUCCESS;
}

HcclResult WaitAllLocalData(ThreadHandle mainThread, const LocalChannels &channels)
{
    for (const ChannelInfo *channel : channels) {
        CHK_RET(WaitChannelData(mainThread, *channel));
    }
    return HCCL_SUCCESS;
}

HcclResult RecordAllLocalAck(ThreadHandle mainThread, const LocalChannels &channels)
{
    for (const ChannelInfo *channel : channels) {
        CHK_RET(RecordChannelAck(mainThread, *channel));
    }
    return HCCL_SUCCESS;
}

HcclResult LocalMeshBarrier(ThreadHandle mainThread, const LocalChannels &channels)
{
    // AllGather writes directly into sender-indexed slots. In particular,
    // owners 0, 1 and 2 overwrite the slots used by peers for cross-server
    // accumulator, receive and backup storage. Record every local edge first,
    // then wait for all seven peers to finish the cross-server phase.
    for (const ChannelInfo *channel : channels) {
        CHK_RET(static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(mainThread, channel->handle, READY_NOTIFY)));
    }
    for (const ChannelInfo *channel : channels) {
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            mainThread, channel->handle, READY_NOTIFY, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult ScheduleSlaveWindow(ThreadHandle mainThread, ThreadHandle workerThread, uint32_t workerIndex,
    const ChannelInfo &channel, uint32_t senderLocalRank, uint32_t targetLocalRank, uint8_t *input,
    uint8_t *localBuffer, uint64_t base, uint64_t windowCount, uint64_t slotStrideBytes)
{
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyWaitOnThread(workerThread, SLAVE_RS_START_NOTIFY, CUSTOM_TIMEOUT)));

    uint64_t rsCount = SliceCount(windowCount, targetLocalRank);
    uint64_t rsOffset = SliceOffset(windowCount, targetLocalRank);
    void *remoteRsSlot =
        static_cast<uint8_t *>(channel.remoteCclMem.addr) + senderLocalRank * slotStrideBytes;
    CHK_RET(WriteAndRecordData(workerThread, channel, remoteRsSlot,
        input + (base + rsOffset) * sizeof(float), rsCount * sizeof(float)));
    CHK_RET(WaitChannelAck(workerThread, channel));

    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyWaitOnThread(workerThread, SLAVE_AG_START_NOTIFY, CUSTOM_TIMEOUT)));

    uint64_t agCount = SliceCount(windowCount, senderLocalRank);
    uint8_t *localResult = SlotAddress(localBuffer, slotStrideBytes, senderLocalRank);
    void *remoteAgSlot =
        static_cast<uint8_t *>(channel.remoteCclMem.addr) + senderLocalRank * slotStrideBytes;
    CHK_RET(WriteAndRecordData(
        workerThread, channel, remoteAgSlot, localResult, agCount * sizeof(float)));
    CHK_RET(WaitChannelAck(workerThread, channel));

    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyRecordOnThread(workerThread, mainThread, workerIndex)));
    return HCCL_SUCCESS;
}

HcclResult CopyMeshWindowToOutput(ThreadHandle mainThread, uint8_t *output, uint8_t *localBuffer,
    uint64_t base, uint64_t windowCount, uint64_t slotStrideBytes)
{
    uint64_t maxSliceCount = CeilDiv(windowCount, SERVER_SIZE);
    uint64_t logicalSlotBytes = maxSliceCount * sizeof(float);
    if (windowCount % SERVER_SIZE == 0 && logicalSlotBytes == slotStrideBytes) {
        return static_cast<HcclResult>(HcommLocalCopyOnThread(
            mainThread, output + base * sizeof(float), localBuffer, windowCount * sizeof(float)));
    }

    for (uint32_t slice = 0; slice < SERVER_SIZE; ++slice) {
        uint64_t count = SliceCount(windowCount, slice);
        if (count == 0) {
            continue;
        }
        uint64_t offset = SliceOffset(windowCount, slice);
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread,
            output + (base + offset) * sizeof(float), SlotAddress(localBuffer, slotStrideBytes, slice),
            count * sizeof(float))));
    }
    return HCCL_SUCCESS;
}

__attribute__((noinline)) HcclResult ExecDirectMeshHierarchical(const OpParam &param, const AlgResourceCtx &resCtx,
    const ChannelTable &channelTable, uint64_t bufferBytes)
{
    CHK_PRT_RET(resCtx.threads.size() < MESH_THREAD_NUM,
        HCCL_ERROR("Direct Mesh needs [%u] threads, only [%llu] available", MESH_THREAD_NUM,
            static_cast<unsigned long long>(resCtx.threads.size())), HCCL_E_INTERNAL);

    LocalPeerRanks peerRanks{};
    LocalChannels localChannels{};
    const ChannelInfo *crossChannel = nullptr;
    CHK_RET(BuildMeshResources(param, channelTable, peerRanks, localChannels, crossChannel));

    uint64_t maximumStrideBytes = MeshMaximumSlotStride(bufferBytes);
    CHK_PRT_RET(maximumStrideBytes < sizeof(float),
        HCCL_ERROR("CCL buffer is too small for eight Mesh slots, size[%llu]",
            static_cast<unsigned long long>(bufferBytes)), HCCL_E_INTERNAL);
    uint64_t slotCapacity = maximumStrideBytes / sizeof(float);
    // Besides the per-link slices, the output fast path emits one local-copy
    // task for the complete window. Keep that task within the engine's
    // documented 256 MiB single-task limit as well.
    uint64_t windowCapacity = std::min(
        slotCapacity * SERVER_SIZE, MAX_SINGLE_TASK_BYTES / sizeof(float));
    CHK_PRT_RET(windowCapacity == 0, HCCL_ERROR("Direct Mesh window capacity is zero"), HCCL_E_INTERNAL);

    ThreadHandle mainThread = resCtx.aicpuThread;
    uint8_t *input = static_cast<uint8_t *>(param.inputPtr);
    uint8_t *output = static_cast<uint8_t *>(param.outputPtr);
    uint8_t *localBuffer = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    uint32_t localRank = param.myRank % SERVER_SIZE;
    uint32_t serverIndex = param.myRank / SERVER_SIZE;

    uint64_t windowsLeft = CeilDiv(param.count, windowCapacity);
    uint64_t remaining = param.count;
    uint64_t base = 0;
    while (windowsLeft > 0) {
        // Balance all windows so 400 MiB + 4 B does not create a 4-byte tail
        // collective after an otherwise full-size first window.
        uint64_t windowCount = CeilDiv(remaining, windowsLeft);
        uint64_t maxSliceCount = CeilDiv(windowCount, SERVER_SIZE);
        uint64_t slotStrideBytes = AlignUp(maxSliceCount * sizeof(float), SLOT_ALIGNMENT_BYTES);
        CHK_PRT_RET(slotStrideBytes > maximumStrideBytes || slotStrideBytes * SERVER_SIZE > bufferBytes,
            HCCL_ERROR("Mesh window exceeds CCL buffer, window[%llu] stride[%llu] buffer[%llu]",
                static_cast<unsigned long long>(windowCount),
                static_cast<unsigned long long>(slotStrideBytes),
                static_cast<unsigned long long>(bufferBytes)), HCCL_E_INTERNAL);

        uint64_t ownedCount = SliceCount(windowCount, localRank);
        uint64_t ownedOffset = SliceOffset(windowCount, localRank);
        uint8_t *ownedSlot = SlotAddress(localBuffer, slotStrideBytes, localRank);
        if (ownedCount > 0) {
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread, ownedSlot,
                input + (base + ownedOffset) * sizeof(float), ownedCount * sizeof(float))));
        }

        // T0 and T1..T6 launch all seven direct writes concurrently. The
        // workers also enqueue their later AllGather program behind AG_START.
        CHK_RET(RecordSlaveStart(mainThread, resCtx, SLAVE_RS_START_NOTIFY));
        for (uint32_t worker = 1; worker < MESH_THREAD_NUM; ++worker) {
            uint32_t targetLocalRank = peerRanks[worker] % SERVER_SIZE;
            CHK_RET(ScheduleSlaveWindow(mainThread, resCtx.threads[worker], worker,
                *localChannels[worker], localRank, targetLocalRank, input, localBuffer, base, windowCount,
                slotStrideBytes));
        }

        uint32_t mainTargetLocalRank = peerRanks[0] % SERVER_SIZE;
        uint64_t mainRsCount = SliceCount(windowCount, mainTargetLocalRank);
        uint64_t mainRsOffset = SliceOffset(windowCount, mainTargetLocalRank);
        void *mainRemoteRsSlot =
            static_cast<uint8_t *>(localChannels[0]->remoteCclMem.addr) + localRank * slotStrideBytes;
        CHK_RET(WriteAndRecordData(mainThread, *localChannels[0], mainRemoteRsSlot,
            input + (base + mainRsOffset) * sizeof(float), mainRsCount * sizeof(float)));

        CHK_RET(WaitAllLocalData(mainThread, localChannels));

        // Every contribution has a stable sender-indexed slot. Reducing only
        // on T0 in increasing sender order makes the floating-point DAG fully
        // independent of transfer completion order.
        if (ownedCount > 0) {
            uint8_t *accumulator = SlotAddress(localBuffer, slotStrideBytes, 0);
            for (uint32_t sender = 1; sender < SERVER_SIZE; ++sender) {
                CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(mainThread, accumulator,
                    SlotAddress(localBuffer, slotStrideBytes, sender), ownedCount, HCOMM_DATA_TYPE_FP32,
                    HCOMM_REDUCE_SUM)));
            }
        }

        // Release all seven remote writers only after every staging slot has
        // been consumed. Record every ACK before T0 waits for its own peer.
        CHK_RET(RecordAllLocalAck(mainThread, localChannels));
        CHK_RET(WaitChannelAck(mainThread, *localChannels[0]));

        // The next cross-server write targets slot1, which is also a local RS
        // staging slot. Both corresponding ranks must finish local reduction
        // before either side reuses it.
        CHK_RET(PairBarrier(mainThread, *crossChannel));

        uint8_t *partialSlot = SlotAddress(localBuffer, slotStrideBytes, 0);
        uint8_t *crossReceiveSlot = SlotAddress(localBuffer, slotStrideBytes, 1);
        uint8_t *serverOneBackup = SlotAddress(localBuffer, slotStrideBytes, 2);
        if (serverIndex == 1 && ownedCount > 0) {
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
                mainThread, serverOneBackup, partialSlot, ownedCount * sizeof(float))));
        }

        void *remoteCrossSlot =
            static_cast<uint8_t *>(crossChannel->remoteCclMem.addr) + slotStrideBytes;
        CHK_RET(WriteAndRecordData(mainThread, *crossChannel, remoteCrossSlot, partialSlot,
            ownedCount * sizeof(float)));
        CHK_RET(WaitChannelData(mainThread, *crossChannel));

        // Both servers explicitly evaluate P(server0) + P(server1), including
        // operand direction, so even NaN payload and signed-zero behavior is
        // identical at the two owners.
        if (ownedCount > 0) {
            if (serverIndex == 0) {
                CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(mainThread, partialSlot,
                    crossReceiveSlot, ownedCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
            } else {
                CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
                    mainThread, partialSlot, crossReceiveSlot, ownedCount * sizeof(float))));
                CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(mainThread, partialSlot,
                    serverOneBackup, ownedCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
            }
        }
        CHK_RET(RecordChannelAck(mainThread, *crossChannel));
        CHK_RET(WaitChannelAck(mainThread, *crossChannel));

        if (localRank != 0 && ownedCount > 0) {
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
                mainThread, ownedSlot, partialSlot, ownedCount * sizeof(float))));
        }

        // No local rank may begin its direct AllGather until every local rank
        // has stopped using slots 0..2 for the cross-server reduction.
        CHK_RET(LocalMeshBarrier(mainThread, localChannels));

        // All seven peers now receive this rank's final owned slice directly
        // into the physical slot indexed by the owner rank.
        CHK_RET(RecordSlaveStart(mainThread, resCtx, SLAVE_AG_START_NOTIFY));
        void *mainRemoteAgSlot =
            static_cast<uint8_t *>(localChannels[0]->remoteCclMem.addr) + localRank * slotStrideBytes;
        CHK_RET(WriteAndRecordData(mainThread, *localChannels[0], mainRemoteAgSlot, ownedSlot,
            ownedCount * sizeof(float)));

        CHK_RET(WaitAllLocalData(mainThread, localChannels));
        CHK_RET(CopyMeshWindowToOutput(
            mainThread, output, localBuffer, base, windowCount, slotStrideBytes));

        // Delay AG ACK until the output copy finishes. Consequently no peer
        // can start reusing any of our eight CCL slots in the next window.
        CHK_RET(RecordAllLocalAck(mainThread, localChannels));
        CHK_RET(WaitChannelAck(mainThread, *localChannels[0]));
        CHK_RET(WaitSlaveDone(mainThread));

        base += windowCount;
        remaining -= windowCount;
        --windowsLeft;
    }
    return HCCL_SUCCESS;
}

HcclResult PairWriteReduceRangeStep(ThreadHandle thread, const ChannelInfo &channel,
    uint32_t senderLocalRank, uint32_t targetLocalRank, uint8_t *input, uint64_t base,
    uint64_t targetOffset, uint64_t rangeOffset, uint64_t rangeCount, uint32_t step)
{
    CHK_RET(Acknowledge(thread, channel, channel));

    uint32_t senderOrdinal = senderLocalRank < targetLocalRank ? senderLocalRank : senderLocalRank - 1;
    uint32_t stripe = (senderOrdinal + step) % LOCAL_PEER_NUM;
    uint64_t stripeCount = MeshStripeCount(rangeCount, stripe);
    uint64_t stripeOffset = MeshStripeOffset(rangeCount, stripe);

    void *remoteDst = static_cast<uint8_t *>(channel.remoteCclMem.addr) +
        (rangeOffset + stripeOffset) * sizeof(float);
    const void *localSrc = input +
        (base + targetOffset + rangeOffset + stripeOffset) * sizeof(float);
    CHK_RET(WriteReduceWithNotify(thread, channel, remoteDst, localSrc, stripeCount));
    CHK_RET(WaitChannelData(thread, channel));
    return HCCL_SUCCESS;
}

HcclResult PairDirectReadRange(ThreadHandle thread, const ChannelInfo &channel, uint8_t *output,
    uint64_t outputOffset, uint64_t remoteResultOffsetBytes, uint64_t remoteElementOffset,
    uint64_t count)
{
    CHK_RET(Acknowledge(thread, channel, channel));

    if (count > 0) {
        const void *remoteSrc = static_cast<uint8_t *>(channel.remoteCclMem.addr) +
            remoteResultOffsetBytes + remoteElementOffset * sizeof(float);
        CHK_RET(static_cast<HcclResult>(HcommReadOnThread(thread, channel.handle,
            output + outputOffset * sizeof(float), remoteSrc, count * sizeof(float))));
    }

    CHK_RET(static_cast<HcclResult>(
        HcommChannelNotifyRecordOnThread(thread, channel.handle, DATA_NOTIFY)));
    CHK_RET(WaitChannelData(thread, channel));
    return HCCL_SUCCESS;
}

HcclResult ScheduleMeshChunkWorker(ThreadHandle mainThread, ThreadHandle workerThread,
    uint32_t workerIndex, const ChannelInfo &channel, uint32_t senderLocalRank,
    uint32_t targetLocalRank, uint8_t *input, uint8_t *output, uint64_t base,
    uint64_t windowCount, uint64_t remoteResultOffsetBytes)
{
    uint64_t peerCount = MeshSliceCount(windowCount, targetLocalRank);
    uint64_t peerOffset = MeshSliceOffset(windowCount, targetLocalRank);
    uint64_t localPrefixCount = DualPathLocalPrefixCount(peerCount);
    for (uint32_t step = 0; step < LOCAL_PEER_NUM; ++step) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(workerThread, SLAVE_RS_START_NOTIFY, CUSTOM_TIMEOUT)));
        CHK_RET(PairWriteReduceRangeStep(workerThread, channel, senderLocalRank, targetLocalRank,
            input, base, peerOffset, 0, localPrefixCount, step));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(workerThread, mainThread, workerIndex)));
    }

    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyWaitOnThread(workerThread, SLAVE_AG_START_NOTIFY, CUSTOM_TIMEOUT)));
    CHK_RET(PairDirectReadRange(workerThread, channel, output, base + peerOffset,
        remoteResultOffsetBytes, 0, localPrefixCount));
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyRecordOnThread(workerThread, mainThread, workerIndex)));
    return HCCL_SUCCESS;
}

HcclResult ScheduleCrossRsAgWorker(ThreadHandle mainThread, ThreadHandle workerThread,
    uint32_t workerIndex, const ChannelInfo &channel, uint32_t senderLocalRank,
    uint32_t ownerLocalRank, uint8_t *input, uint8_t *output, uint64_t base,
    uint64_t windowCount, uint64_t remoteResultOffsetBytes)
{
    uint64_t peerCount = MeshSliceCount(windowCount, ownerLocalRank);
    uint64_t peerOffset = MeshSliceOffset(windowCount, ownerLocalRank);
    uint64_t localPrefixCount = DualPathLocalPrefixCount(peerCount);
    uint64_t crossSuffixCount = peerCount - localPrefixCount;
    for (uint32_t step = 0; step < LOCAL_PEER_NUM; ++step) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(workerThread, SLAVE_RS_START_NOTIFY, CUSTOM_TIMEOUT)));
        CHK_RET(PairWriteReduceRangeStep(workerThread, channel, senderLocalRank, ownerLocalRank,
            input, base, peerOffset, localPrefixCount, crossSuffixCount, step));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(workerThread, mainThread, workerIndex)));
    }

    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyWaitOnThread(workerThread, SLAVE_AG_START_NOTIFY, CUSTOM_TIMEOUT)));
    CHK_RET(PairDirectReadRange(workerThread, channel, output,
        base + peerOffset + localPrefixCount, remoteResultOffsetBytes,
        localPrefixCount, peerCount - localPrefixCount));
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyRecordOnThread(workerThread, mainThread, workerIndex)));
    return HCCL_SUCCESS;
}

uint64_t MeshChunkMaximumStride(uint64_t bufferBytes)
{
    return std::min(AlignDown(bufferBytes / LARGE_SLOT_NUM, SLOT_ALIGNMENT_BYTES),
        MAX_SINGLE_TASK_BYTES);
}

HcclResult ExecMeshChunkHierarchical(const OpParam &param, const AlgResourceCtx &resCtx,
    const ChannelTable &channelTable, uint64_t bufferBytes)
{
    CHK_PRT_RET(resCtx.threads.size() < CONTEST_THREAD_NUM,
        HCCL_ERROR("MeshChunk needs [%u] threads, only [%llu] available", CONTEST_THREAD_NUM,
            static_cast<unsigned long long>(resCtx.threads.size())), HCCL_E_INTERNAL);

    LocalPeerRanks peerRanks{};
    LocalChannels localChannels{};
    CrossPeerRanks crossPeerRanks{};
    CrossChannels crossChannels{};
    const ChannelInfo *crossChannel = nullptr;
    CHK_RET(BuildMeshResources(param, channelTable, peerRanks, localChannels, crossChannel));
    CHK_RET(BuildCrossAgResources(param, channelTable, crossPeerRanks, crossChannels));

    uint64_t maximumStrideBytes = MeshChunkMaximumStride(bufferBytes);
    CHK_PRT_RET(maximumStrideBytes < sizeof(float),
        HCCL_ERROR("CCL buffer is too small for MeshChunk, size[%llu]",
            static_cast<unsigned long long>(bufferBytes)), HCCL_E_INTERNAL);
    uint64_t slotCapacity = maximumStrideBytes / sizeof(float);
    CHK_PRT_RET(slotCapacity > std::numeric_limits<uint64_t>::max() / SERVER_SIZE,
        HCCL_ERROR("MeshChunk window capacity overflows"), HCCL_E_INTERNAL);
    uint64_t windowCapacity = slotCapacity * SERVER_SIZE;

    ThreadHandle mainThread = resCtx.aicpuThread;
    uint8_t *input = static_cast<uint8_t *>(param.inputPtr);
    uint8_t *output = static_cast<uint8_t *>(param.outputPtr);
    uint8_t *localBuffer = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    uint32_t localRank = param.myRank % SERVER_SIZE;
    uint32_t serverIndex = param.myRank / SERVER_SIZE;

    uint64_t windowsLeft = CeilDiv(param.count, windowCapacity);
    uint64_t remaining = param.count;
    uint64_t base = 0;
    while (windowsLeft > 0) {
        uint64_t targetWindowCount = CeilDiv(remaining, windowsLeft);
        uint64_t windowCount = windowsLeft > 1 ?
            AlignUp(targetWindowCount, FP32_ALIGNMENT_COUNT) : targetWindowCount;
        uint64_t maxSliceCount = CeilDiv(windowCount, SERVER_SIZE);
        uint64_t slotStrideBytes = AlignUp(maxSliceCount * sizeof(float), SLOT_ALIGNMENT_BYTES);
        CHK_PRT_RET(slotStrideBytes > maximumStrideBytes ||
                slotStrideBytes > bufferBytes / LARGE_SLOT_NUM,
            HCCL_ERROR("MeshChunk window exceeds CCL buffer, window[%llu] stride[%llu] buffer[%llu]",
                static_cast<unsigned long long>(windowCount),
                static_cast<unsigned long long>(slotStrideBytes),
                static_cast<unsigned long long>(bufferBytes)), HCCL_E_INTERNAL);

        uint64_t ownedCount = MeshSliceCount(windowCount, localRank);
        uint64_t ownedOffset = MeshSliceOffset(windowCount, localRank);
        uint8_t *partialSlot = localBuffer;
        uint8_t *crossReceiveSlot = localBuffer + slotStrideBytes;
        uint8_t *resultSlot = localBuffer + serverIndex * slotStrideBytes;
        if (ownedCount > 0) {
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(mainThread, partialSlot,
                input + (base + ownedOffset) * sizeof(float), ownedCount * sizeof(float))));
        }

        uint64_t remoteResultOffsetBytes = serverIndex * slotStrideBytes;
        uint64_t crossRemoteResultOffsetBytes = (serverIndex ^ 1U) * slotStrideBytes;
        // Enqueue the first main-to-worker records before the workers' first
        // wait tasks so the task graph has an executable root on every rank.
        CHK_RET(RecordSlaveStart(mainThread, resCtx, SLAVE_RS_START_NOTIFY));
        CHK_RET(RecordCrossStart(mainThread, resCtx, SLAVE_RS_START_NOTIFY));
        for (uint32_t worker = 1; worker < MESH_THREAD_NUM; ++worker) {
            uint32_t targetLocalRank = peerRanks[worker] % SERVER_SIZE;
            CHK_RET(ScheduleMeshChunkWorker(mainThread, resCtx.threads[worker], worker,
                *localChannels[worker], localRank, targetLocalRank, input, output, base,
                windowCount, remoteResultOffsetBytes));
        }
        for (uint32_t worker = 0; worker < CROSS_AG_THREAD_NUM; ++worker) {
            uint32_t threadIndex = CROSS_AG_THREAD_BEGIN + worker;
            uint32_t ownerLocalRank = crossPeerRanks[worker] % SERVER_SIZE;
            CHK_RET(ScheduleCrossRsAgWorker(mainThread, resCtx.threads[threadIndex], threadIndex,
                *crossChannels[worker], localRank, ownerLocalRank, input, output, base,
                windowCount, crossRemoteResultOffsetBytes));
        }

        uint32_t mainTargetLocalRank = peerRanks[0] % SERVER_SIZE;
        uint64_t mainPeerCount = MeshSliceCount(windowCount, mainTargetLocalRank);
        uint64_t mainPeerOffset = MeshSliceOffset(windowCount, mainTargetLocalRank);
        uint64_t mainLocalPrefixCount = DualPathLocalPrefixCount(mainPeerCount);
        for (uint32_t step = 0; step < LOCAL_PEER_NUM; ++step) {
            if (step > 0) {
                CHK_RET(RecordSlaveStart(mainThread, resCtx, SLAVE_RS_START_NOTIFY));
                CHK_RET(RecordCrossStart(mainThread, resCtx, SLAVE_RS_START_NOTIFY));
            }
            CHK_RET(PairWriteReduceRangeStep(mainThread, *localChannels[0], localRank,
                mainTargetLocalRank, input, base, mainPeerOffset, 0, mainLocalPrefixCount, step));
            CHK_RET(WaitSlaveDone(mainThread));
            CHK_RET(WaitCrossDone(mainThread));
        }

        // Both corresponding ranks finish their local MeshChunk before either
        // side overwrites the cross receive slot.
        CHK_RET(PairBarrier(mainThread, *crossChannel));
        void *remoteCrossReceive =
            static_cast<uint8_t *>(crossChannel->remoteCclMem.addr) + slotStrideBytes;
        CHK_RET(WriteWithNotify(mainThread, *crossChannel, remoteCrossReceive,
            partialSlot, ownedCount * sizeof(float)));
        CHK_RET(WaitChannelData(mainThread, *crossChannel));

        // Server 0 keeps P0+P1 in slot 0. Server 1 keeps the same ordered
        // expression in slot 1 by reducing P1 into the received P0.
        if (ownedCount > 0) {
            if (serverIndex == 0) {
                CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(mainThread,
                    partialSlot, crossReceiveSlot, ownedCount, HCOMM_DATA_TYPE_FP32,
                    HCOMM_REDUCE_SUM)));
            } else {
                CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(mainThread,
                    crossReceiveSlot, partialSlot, ownedCount, HCOMM_DATA_TYPE_FP32,
                    HCOMM_REDUCE_SUM)));
            }
        }
        CHK_RET(RecordChannelAck(mainThread, *crossChannel));
        CHK_RET(WaitChannelAck(mainThread, *crossChannel));

        // Each non-owned slice is split at an aligned 7:8 boundary. Seven
        // layer-0 reads fetch the prefixes while seven layer-1 reads fetch
        // the suffixes from the identical result copies on the other server.
        // T14 copies this rank's own result concurrently.
        CHK_RET(RecordSlaveStart(mainThread, resCtx, SLAVE_AG_START_NOTIFY));
        CHK_RET(RecordCrossStart(mainThread, resCtx, SLAVE_AG_START_NOTIFY));
        CHK_RET(RecordCopyStart(mainThread, resCtx, SLAVE_AG_START_NOTIFY));
        CHK_RET(ScheduleCopyWorker(mainThread, resCtx.threads[COPY_THREAD_INDEX],
            SLAVE_AG_START_NOTIFY, output + (base + ownedOffset) * sizeof(float),
            resultSlot, ownedCount * sizeof(float)));
        CHK_RET(PairDirectReadRange(mainThread, *localChannels[0], output,
            base + mainPeerOffset, remoteResultOffsetBytes, 0, mainLocalPrefixCount));
        CHK_RET(WaitSlaveDone(mainThread));
        CHK_RET(WaitCrossDone(mainThread));
        CHK_RET(WaitCopyDone(mainThread));

        base += windowCount;
        remaining -= windowCount;
        --windowsLeft;
    }
    return HCCL_SUCCESS;
}

HcclResult PairOneShotWrite(ThreadHandle thread, const ChannelInfo &channel,
    uint32_t senderLocalRank, const void *input, uint64_t bytes, uint64_t slotStrideBytes)
{
    CHK_RET(Acknowledge(thread, channel, channel));
    void *remoteDst = static_cast<uint8_t *>(channel.remoteCclMem.addr) +
        senderLocalRank * slotStrideBytes;
    CHK_RET(WriteWithNotify(thread, channel, remoteDst, input, bytes));
    CHK_RET(WaitChannelData(thread, channel));
    return HCCL_SUCCESS;
}

__attribute__((noinline)) HcclResult ExecHierarchicalHalvingDoubling(const OpParam &param, const AlgResourceCtx &resCtx,
    const ChannelTable &channelTable, uint64_t bufferBytes)
{
    uint64_t dataBytes = param.count * sizeof(float);
    CHK_PRT_RET(dataBytes != HALVING_DOUBLING_MESSAGE_BYTES || param.count % CONTEST_RANK_SIZE != 0,
        HCCL_ERROR("Halving-Doubling expects exactly 512 KiB divisible by sixteen, bytes[%llu] count[%llu]",
            static_cast<unsigned long long>(dataBytes),
            static_cast<unsigned long long>(param.count)), HCCL_E_PARA);

    // Layout: one full accumulator followed by three round-private
    // ReduceScatter staging ranges (D/2 + D/4 + D/8) and one D/8 staging
    // range for the fused turn. The complete layout consumes exactly 2D.
    CHK_PRT_RET(dataBytes > bufferBytes / 2,
        HCCL_ERROR("Halving-Doubling needs a two-message CCL buffer, bytes[%llu] buffer[%llu]",
            static_cast<unsigned long long>(dataBytes),
            static_cast<unsigned long long>(bufferBytes)), HCCL_E_INTERNAL);

    ThreadHandle thread = resCtx.aicpuThread;
    uint8_t *input = static_cast<uint8_t *>(param.inputPtr);
    uint8_t *output = static_cast<uint8_t *>(param.outputPtr);
    uint8_t *localBuffer = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    uint8_t *accumulator = localBuffer;

    CHK_RET(static_cast<HcclResult>(
        HcommLocalCopyOnThread(thread, accumulator, input, dataBytes)));

    uint64_t rangeOffset = 0;
    uint64_t rangeCount = param.count;
    uint64_t stagingOffsetBytes = dataBytes;
    // The first dimension is the high-bandwidth Clos edge. Moving half of the
    // payload across servers first halves the traffic carried by the three
    // lower-bandwidth layer-0 dimensions.
    constexpr std::array<uint32_t, 3> REDUCE_SCATTER_MASKS = {8, 4, 2};
    for (uint32_t mask : REDUCE_SCATTER_MASKS) {
        uint64_t halfCount = rangeCount / 2;
        uint64_t halfBytes = halfCount * sizeof(float);
        bool keepUpperHalf = (param.myRank & mask) != 0;
        uint64_t keepOffset = rangeOffset + (keepUpperHalf ? halfCount : 0);
        uint64_t sendOffset = rangeOffset + (keepUpperHalf ? 0 : halfCount);

        uint32_t partnerRank = param.myRank ^ mask;
        const ChannelInfo *channel = FindChannel(channelTable, partnerRank);
        CHK_PRT_RET(channel == nullptr,
            HCCL_ERROR("Missing Halving-Doubling channel to rank[%u]", partnerRank), HCCL_E_INTERNAL);
        CHK_PRT_RET(channel->notifyNum <= HD_ALL_GATHER_NOTIFY,
            HCCL_ERROR("Halving-Doubling channel to rank[%u] has only [%u] notifies",
                partnerRank, channel->notifyNum), HCCL_E_INTERNAL);

        void *remoteStaging = static_cast<uint8_t *>(channel->remoteCclMem.addr) + stagingOffsetBytes;
        CHK_RET(FusedWriteWithNotify(thread, *channel, remoteStaging,
            accumulator + sendOffset * sizeof(float), halfBytes, DATA_NOTIFY));
        CHK_RET(WaitChannelData(thread, *channel));
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(thread,
            accumulator + keepOffset * sizeof(float), localBuffer + stagingOffsetBytes,
            halfCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));

        rangeOffset = keepOffset;
        rangeCount = halfCount;
        stagingOffsetBytes += halfBytes;
    }

    // Fuse the final mask-1 ReduceScatter round with the first mask-1
    // AllGather round. After the first three dimensions, both ranks in a
    // mask-1 pair own the same D/8 logical range and hold complementary
    // eight-rank partial sums. Exchanging and reducing the complete range
    // gives both ranks the final D/8 result, so the reverse mask-1 exchange
    // becomes unnecessary. Compared with the separate 32 KiB RS and AG
    // rounds, this keeps the same 64 KiB wire volume while removing one data
    // communication phase. A delayed ACK below closes the staging lifetime
    // across repeated collectives.
    constexpr uint32_t FUSED_TURN_MASK = 1;
    uint32_t fusedPartnerRank = param.myRank ^ FUSED_TURN_MASK;
    const ChannelInfo *fusedChannel = FindChannel(channelTable, fusedPartnerRank);
    CHK_PRT_RET(fusedChannel == nullptr,
        HCCL_ERROR("Missing fused Halving-Doubling channel to rank[%u]", fusedPartnerRank),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(fusedChannel->notifyNum <= HD_ALL_GATHER_NOTIFY,
        HCCL_ERROR("Fused Halving-Doubling channel to rank[%u] has only [%u] notifies",
            fusedPartnerRank, fusedChannel->notifyNum), HCCL_E_INTERNAL);

    uint64_t fusedBytes = rangeCount * sizeof(float);
    void *remoteFusedStaging =
        static_cast<uint8_t *>(fusedChannel->remoteCclMem.addr) + stagingOffsetBytes;
    CHK_RET(FusedWriteWithNotify(thread, *fusedChannel, remoteFusedStaging,
        accumulator + rangeOffset * sizeof(float), fusedBytes, DATA_NOTIFY));
    CHK_RET(WaitChannelData(thread, *fusedChannel));
    // Every rank uses one fixed partner and one fixed reduction DAG, so task
    // completion order cannot change the result of a repeated invocation.
    // For finite FP32 values, the peers also produce the same bitwise sum even
    // though their local/remote operand directions are reversed. Keeping the
    // result in the primary accumulator prevents the next invocation's remote
    // staging writes from corrupting this invocation's final output copy.
    CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(thread,
        accumulator + rangeOffset * sizeof(float), localBuffer + stagingOffsetBytes,
        rangeCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    // DATA Wait only proves that the incoming write has completed; it does
    // not prove that the peer has consumed its own staging range. Record our
    // completion now, then defer the matching Wait until after AllGather and
    // the output copy. This prevents a faster parity subgroup in the next
    // invocation from overwriting a slower peer's fused-reduction source.
    CHK_RET(RecordChannelAck(thread, *fusedChannel));

    // Use a notify reserved exclusively for this AllGather. This prevents a
    // fast rank from colliding with an unconsumed one-bit notification when a
    // later collective selects a different algorithm and notify protocol.
    constexpr std::array<uint32_t, 3> ALL_GATHER_MASKS = {2, 4, 8};
    for (uint32_t mask : ALL_GATHER_MASKS) {
        uint32_t partnerRank = param.myRank ^ mask;
        const ChannelInfo *channel = FindChannel(channelTable, partnerRank);
        CHK_PRT_RET(channel == nullptr,
            HCCL_ERROR("Missing Halving-Doubling channel to rank[%u]", partnerRank), HCCL_E_INTERNAL);

        void *remoteDst = static_cast<uint8_t *>(channel->remoteCclMem.addr) +
            rangeOffset * sizeof(float);
        CHK_RET(FusedWriteWithNotify(thread, *channel, remoteDst,
            accumulator + rangeOffset * sizeof(float), rangeCount * sizeof(float), HD_ALL_GATHER_NOTIFY));
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            thread, channel->handle, HD_ALL_GATHER_NOTIFY, CUSTOM_TIMEOUT)));

        if ((param.myRank & mask) != 0) {
            rangeOffset -= rangeCount;
        }
        rangeCount *= 2;
    }

    CHK_PRT_RET(rangeOffset != 0 || rangeCount != param.count,
        HCCL_ERROR("Halving-Doubling reconstructed an invalid range, offset[%llu] count[%llu]",
            static_cast<unsigned long long>(rangeOffset),
            static_cast<unsigned long long>(rangeCount)), HCCL_E_INTERNAL);
    CHK_RET(static_cast<HcclResult>(
        HcommLocalCopyOnThread(thread, output, accumulator, dataBytes)));
    CHK_RET(WaitChannelAck(thread, *fusedChannel));
    return HCCL_SUCCESS;
}

HcclResult ScheduleOneShotWorker(ThreadHandle mainThread, ThreadHandle workerThread,
    uint32_t workerIndex, const ChannelInfo &channel, uint32_t senderLocalRank,
    const void *input, uint64_t bytes, uint64_t slotStrideBytes)
{
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyWaitOnThread(workerThread, SLAVE_RS_START_NOTIFY, CUSTOM_TIMEOUT)));
    CHK_RET(PairOneShotWrite(workerThread, channel, senderLocalRank, input, bytes,
        slotStrideBytes));
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyRecordOnThread(workerThread, mainThread, workerIndex)));
    return HCCL_SUCCESS;
}

HcclResult ExecOneShotHierarchical(const OpParam &param, const AlgResourceCtx &resCtx,
    const ChannelTable &channelTable, uint64_t bufferBytes)
{
    CHK_PRT_RET(resCtx.threads.size() < MESH_THREAD_NUM,
        HCCL_ERROR("OneShot needs [%u] threads, only [%llu] available", MESH_THREAD_NUM,
            static_cast<unsigned long long>(resCtx.threads.size())), HCCL_E_INTERNAL);

    LocalPeerRanks peerRanks{};
    LocalChannels localChannels{};
    const ChannelInfo *crossChannel = nullptr;
    CHK_RET(BuildMeshResources(param, channelTable, peerRanks, localChannels, crossChannel));

    uint64_t dataBytes = param.count * sizeof(float);
    uint64_t slotStrideBytes = AlignUp(dataBytes, SLOT_ALIGNMENT_BYTES);
    CHK_PRT_RET(slotStrideBytes > bufferBytes / SMALL_SLOT_NUM,
        HCCL_ERROR("OneShot needs nine slots, bytes[%llu] stride[%llu] buffer[%llu]",
            static_cast<unsigned long long>(dataBytes),
            static_cast<unsigned long long>(slotStrideBytes),
            static_cast<unsigned long long>(bufferBytes)), HCCL_E_INTERNAL);

    ThreadHandle mainThread = resCtx.aicpuThread;
    uint8_t *input = static_cast<uint8_t *>(param.inputPtr);
    uint8_t *output = static_cast<uint8_t *>(param.outputPtr);
    uint8_t *localBuffer = static_cast<uint8_t *>(resCtx.localBuffer.addr);
    uint32_t localRank = param.myRank % SERVER_SIZE;
    uint32_t serverIndex = param.myRank / SERVER_SIZE;

    // Preserve this rank's contribution when AllReduce is invoked in-place.
    // The fixed sender slot is otherwise unused locally during OneShot.
    const uint8_t *localInput = input;
    if (input == output) {
        uint8_t *localInputSlot = SlotAddress(localBuffer, slotStrideBytes, localRank);
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(mainThread, localInputSlot, input, dataBytes)));
        localInput = localInputSlot;
    }

    CHK_RET(RecordSlaveStart(mainThread, resCtx, SLAVE_RS_START_NOTIFY));
    for (uint32_t worker = 1; worker < MESH_THREAD_NUM; ++worker) {
        CHK_RET(ScheduleOneShotWorker(mainThread, resCtx.threads[worker], worker,
            *localChannels[worker], localRank, localInput, dataBytes, slotStrideBytes));
    }
    CHK_RET(PairOneShotWrite(mainThread, *localChannels[0], localRank, localInput,
        dataBytes, slotStrideBytes));
    CHK_RET(WaitSlaveDone(mainThread));

    // Every rank evaluates the local partial in local-rank order 0..7.
    const void *rankZeroSrc = localRank == 0 ? static_cast<const void *>(localInput) :
        static_cast<const void *>(SlotAddress(localBuffer, slotStrideBytes, 0));
    CHK_RET(static_cast<HcclResult>(
        HcommLocalCopyOnThread(mainThread, output, rankZeroSrc, dataBytes)));
    for (uint32_t sender = 1; sender < SERVER_SIZE; ++sender) {
        const void *src = sender == localRank ? static_cast<const void *>(localInput) :
            static_cast<const void *>(SlotAddress(localBuffer, slotStrideBytes, sender));
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(mainThread, output,
            src, param.count, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    }

    uint8_t *crossReceive = SlotAddress(localBuffer, slotStrideBytes, SERVER_SIZE);
    CHK_RET(PairBarrier(mainThread, *crossChannel));
    void *remoteCrossReceive = static_cast<uint8_t *>(crossChannel->remoteCclMem.addr) +
        SERVER_SIZE * slotStrideBytes;
    CHK_RET(WriteWithNotify(mainThread, *crossChannel, remoteCrossReceive, output, dataBytes));
    CHK_RET(WaitChannelData(mainThread, *crossChannel));
    if (serverIndex == 0) {
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(mainThread, output,
            crossReceive, param.count, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
    } else {
        CHK_RET(static_cast<HcclResult>(HcommLocalReduceOnThread(mainThread, crossReceive,
            output, param.count, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM)));
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(mainThread, output, crossReceive, dataBytes)));
    }
    CHK_RET(RecordChannelAck(mainThread, *crossChannel));
    CHK_RET(WaitChannelAck(mainThread, *crossChannel));
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(param.dataType != HCCL_DATA_TYPE_FP32 || param.reduceType != HCCL_REDUCE_SUM,
        HCCL_ERROR("Unsupported AllReduce type or operation"), HCCL_E_PARA);

    if (param.rankSize == 1) {
        return static_cast<HcclResult>(HcommLocalCopyOnThread(
            resCtx.aicpuThread, param.outputPtr, param.inputPtr, param.count * sizeof(float)));
    }

    CHK_PRT_RET(param.rankSize > CONTEST_RANK_SIZE,
        HCCL_ERROR("Rank size[%u] exceeds the contest implementation", param.rankSize), HCCL_E_PARA);

    ChannelTable channelTable{};
    uint64_t bufferBytes = 0;
    CHK_RET(BuildChannelTable(resCtx, channelTable, bufferBytes));

    uint64_t dataBytes = param.count * sizeof(float);
    if (param.rankSize == CONTEST_RANK_SIZE) {
        if (dataBytes >= LARGE_MESH_CHUNK_THRESHOLD_BYTES) {
            return ExecMeshChunkHierarchical(param, resCtx, channelTable, bufferBytes);
        }
        if (dataBytes == HALVING_DOUBLING_MESSAGE_BYTES) {
            return ExecHierarchicalHalvingDoubling(param, resCtx, channelTable, bufferBytes);
        }
        if (dataBytes >= MESH_MESSAGE_THRESHOLD_BYTES && dataBytes <= SMALL_ONE_SHOT_MAX_BYTES) {
            return ExecOneShotHierarchical(param, resCtx, channelTable, bufferBytes);
        }
        if (dataBytes >= MESH_MESSAGE_THRESHOLD_BYTES) {
            return ExecDirectMeshHierarchical(param, resCtx, channelTable, bufferBytes);
        }
    }

    return ExecRecursiveDoubling(param, resCtx, channelTable, bufferBytes);
}
} // namespace ops_hccl
