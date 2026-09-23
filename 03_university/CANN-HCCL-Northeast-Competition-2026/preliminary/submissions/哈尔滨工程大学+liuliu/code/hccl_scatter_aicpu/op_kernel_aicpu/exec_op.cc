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
#include "custom.h"
#include "log.h"
#include "exec_op.h"

namespace ops_hccl {
namespace {
const ChannelInfo *FindPeer(const AlgResourceCtx &ctx, uint32_t rank)
{
    for (const ChannelInfo &channel : ctx.channels) {
        if (channel.remoteRank == rank) {
            return &channel;
        }
    }
    return nullptr;
}

uint64_t ChunkCapacity(const AlgResourceCtx &ctx, const ChannelInfo &channel)
{
    const uint64_t capacity = std::min(SCATTER_CHUNK_BYTES,
        std::min(ctx.localBuffer.size, channel.remoteCclMem.size));
    return capacity - capacity % sizeof(float);
}

uint64_t AlignBufferBytes(uint64_t bytes)
{
    const uint64_t alignment = bytes >= SCATTER_BUFFER_ALIGNMENT ? SCATTER_BUFFER_ALIGNMENT : sizeof(float);
    return bytes - bytes % alignment;
}

uint64_t RootSlotSize(uint64_t rootBufferBytes, uint32_t rankSize)
{
    return AlignBufferBytes(rootBufferBytes / (rankSize - 1));
}

struct PipeShape {
    uint64_t chunkBytes;
    uint32_t slots;
};

// Both sides use the root's slot size and the receiver's buffer size, in that order.
PipeShape GetPipeShape(uint64_t rootSlotBytes, uint64_t receiverBufferBytes, uint64_t bytes)
{
    const uint64_t available = std::min(rootSlotBytes, receiverBufferBytes);
    const uint32_t slots = available >= 2 * sizeof(float) ? 2 : 1;
    const uint64_t chunk = AlignBufferBytes(std::min(SCATTER_PIPE_CHUNK_BYTES, available / slots));
    // One-chunk messages need only one credit and one data notification.
    return PipeShape{chunk, bytes <= chunk ? 1U : slots};
}

HcclResult SendPipe(ThreadHandle thread, const ChannelInfo &channel, uint8_t *scratch,
    const uint8_t *source, uint64_t bytes, const PipeShape &shape)
{
    if (shape.slots == 1) {
        for (uint64_t offset = 0; offset < bytes;) {
            const uint64_t length = std::min(shape.chunkBytes, bytes - offset);
            CHK_RET(HcommLocalCopyOnThread(thread, scratch, source + offset, length));
            CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, SCATTER_NOTIFY_READY,
                CUSTOM_TIMEOUT));
            CHK_RET(HcommWriteOnThread(thread, channel.handle, channel.remoteCclMem.addr, scratch, length));
            CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, SCATTER_NOTIFY_DATA));
            CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, SCATTER_NOTIFY_CONSUMED,
                CUSTOM_TIMEOUT));
            offset += length;
        }
        return HCCL_SUCCESS;
    }
    uint32_t slot = 0;
    for (uint64_t offset = 0; offset < bytes;) {
        const uint64_t length = std::min(shape.chunkBytes, bytes - offset);
        const uint64_t slotOffset = static_cast<uint64_t>(slot) * shape.chunkBytes;
        // A credit means this remote slot is free, including at the first iteration.
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, slot, CUSTOM_TIMEOUT));
        CHK_RET(HcommLocalCopyOnThread(thread, scratch + slotOffset, source + offset, length));
        CHK_RET(HcommWriteOnThread(thread, channel.handle,
            static_cast<uint8_t *>(channel.remoteCclMem.addr) + slotOffset, scratch + slotOffset, length));
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, SCATTER_PIPE_DATA_BASE + slot));
        offset += length;
        slot = (slot + 1) % shape.slots;
    }
    // Consume the final credits: no residual notification or outstanding receiver copy.
    for (uint32_t i = 0; i < shape.slots; ++i) {
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, i, CUSTOM_TIMEOUT));
    }
    // Receiver must not advertise next invocation's credits before the old ones are drained.
    CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, SCATTER_PIPE_RELEASE));
    return HCCL_SUCCESS;
}

HcclResult ReceivePipe(ThreadHandle thread, const ChannelInfo &channel, uint8_t *scratch,
    uint8_t *output, uint64_t bytes, const PipeShape &shape)
{
    if (shape.slots == 1) {
        for (uint64_t offset = 0; offset < bytes;) {
            const uint64_t length = std::min(shape.chunkBytes, bytes - offset);
            CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, SCATTER_NOTIFY_READY));
            CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, SCATTER_NOTIFY_DATA,
                CUSTOM_TIMEOUT));
            CHK_RET(HcommLocalCopyOnThread(thread, output + offset, scratch, length));
            CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, SCATTER_NOTIFY_CONSUMED));
            offset += length;
        }
        return HCCL_SUCCESS;
    }
    for (uint32_t slot = 0; slot < shape.slots; ++slot) {
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, slot));
    }
    uint32_t slot = 0;
    for (uint64_t offset = 0; offset < bytes;) {
        const uint64_t length = std::min(shape.chunkBytes, bytes - offset);
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, SCATTER_PIPE_DATA_BASE + slot,
            CUSTOM_TIMEOUT));
        CHK_RET(HcommLocalCopyOnThread(thread, output + offset,
            scratch + static_cast<uint64_t>(slot) * shape.chunkBytes, length));
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, slot));
        offset += length;
        slot = (slot + 1) % shape.slots;
    }
    CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, SCATTER_PIPE_RELEASE, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult ExecPipelined(const OpParam &param, const AlgResourceCtx &ctx, uint64_t bytes)
{
    const ThreadHandle mainThread = ctx.aicpuThread;
    if (param.myRank != param.root) {
        const ChannelInfo *channel = FindPeer(ctx, param.root);
        if (channel == nullptr) {
            return HCCL_E_PARA;
        }
        const uint64_t rootSlot = RootSlotSize(channel->remoteCclMem.size, param.rankSize);
        const PipeShape shape = GetPipeShape(rootSlot, ctx.localBuffer.size, bytes);
        if (shape.chunkBytes == 0) {
            return HCCL_E_PARA;
        }
        return ReceivePipe(mainThread, *channel, static_cast<uint8_t *>(ctx.localBuffer.addr),
            static_cast<uint8_t *>(param.outputPtr), bytes, shape);
    }

    CHK_PTR_NULL(param.inputPtr);
    if (ctx.threads.size() < param.rankSize) {
        return HCCL_E_PARA;
    }
    const uint64_t rootSlot = RootSlotSize(ctx.localBuffer.size, param.rankSize);
    const auto *input = static_cast<const uint8_t *>(param.inputPtr);
    std::vector<const ChannelInfo *> peers;
    std::vector<PipeShape> shapes;
    // Validate every peer before recording any of the worker tasks.
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer == param.root) {
            continue;
        }
        const ChannelInfo *channel = FindPeer(ctx, peer);
        if (channel == nullptr || channel->remoteCclMem.addr == nullptr) {
            return HCCL_E_PARA;
        }
        const PipeShape shape = GetPipeShape(rootSlot, channel->remoteCclMem.size, bytes);
        if (shape.chunkBytes == 0) {
            return HCCL_E_PARA;
        }
        peers.push_back(channel);
        shapes.push_back(shape);
    }

    for (uint32_t i = 1; i < param.rankSize; ++i) {
        CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, ctx.threads[i], 0));
    }
    for (uint32_t i = 1; i < param.rankSize; ++i) {
        const ThreadHandle worker = ctx.threads[i];
        const ChannelInfo &channel = *peers[i - 1];
        auto *scratch = static_cast<uint8_t *>(ctx.localBuffer.addr) + static_cast<uint64_t>(i - 1) * rootSlot;
        CHK_RET(HcommThreadNotifyWaitOnThread(worker, 0, CUSTOM_TIMEOUT));
        CHK_RET(SendPipe(worker, channel, scratch,
            input + static_cast<uint64_t>(channel.remoteRank) * bytes, bytes, shapes[i - 1]));
        CHK_RET(HcommThreadNotifyRecordOnThread(worker, mainThread, i));
    }
    for (uint32_t i = 1; i < param.rankSize; ++i) {
        CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, i, CUSTOM_TIMEOUT));
    }
    // Do not modify root's output until all source slices have been consumed.
    const auto *ownSlice = input + static_cast<uint64_t>(param.root) * bytes;
    if (ownSlice != param.outputPtr) {
        CHK_RET(HcommLocalCopyOnThread(mainThread, param.outputPtr, ownSlice, bytes));
    }
    return HCCL_SUCCESS;
}

// V5 uses the fixed two-server competition topology, discovered by Host queries.
// The root sends 4/11 of seven remote slices through its seven local peers.
// V11 keeps the routing and uses one private source slot per root worker.
struct MixedPlan {
    uint64_t chunkBytes = 0;
    uint64_t relayBytes = 0;
    std::vector<uint32_t> helper;
    std::vector<uint32_t> target;
};

bool BuildMixedPlan(const OpParam &param, const AlgResourceCtx &ctx, uint64_t bytes, MixedPlan &plan)
{
    constexpr uint32_t ranks = 16;
    constexpr uint32_t serverRanks = 8;
    constexpr uint64_t minRankBytes = 16ULL * 1024 * 1024;
    if (param.rankSize != ranks || bytes < minRankBytes || ctx.serverGroup.size() != ranks ||
        ctx.channels.size() != ranks - 1 || ctx.threads.size() < ranks) {
        return false;
    }
    std::vector<uint32_t> helpers;
    std::vector<uint32_t> remotes;
    const uint32_t rootGroup = ctx.serverGroup[param.root];
    for (uint32_t rank = 0; rank < ranks; ++rank) {
        if (ctx.serverGroup[rank] > 1) {
            return false;
        }
        if (ctx.serverGroup[rank] == rootGroup) {
            if (rank != param.root) {
                helpers.push_back(rank);
            }
        } else {
            remotes.push_back(rank);
        }
    }
    if (helpers.size() != serverRanks - 1 || remotes.size() != serverRanks) {
        return false;
    }
    uint64_t minBufferBytes = ctx.localBuffer.size;
    for (const ChannelInfo &channel : ctx.channels) {
        minBufferBytes = std::min(minBufferBytes, channel.remoteCclMem.size);
    }
    // Preserve V10's eligibility and root allocation. Each worker now uses one
    // source slot twice as large; receivers retain two independent destination slots.
    const uint64_t baseChunk = AlignBufferBytes(std::min(SCATTER_PIPE_CHUNK_BYTES, minBufferBytes / 30));
    if (baseChunk < 64ULL * 1024) {
        return false;
    }
    plan.chunkBytes = 2 * baseChunk;
    // Split in whole FP32 elements, without overflowing count * 4 / 11.
    const uint64_t relayCount = (param.count / 11) * 4 + ((param.count % 11) * 4) / 11;
    plan.relayBytes = relayCount * sizeof(float);
    plan.helper.assign(ranks, INVALID_VALUE_RANKID);
    plan.target.assign(ranks, INVALID_VALUE_RANKID);
    for (uint32_t i = 0; i < serverRanks - 1; ++i) {
        plan.helper[remotes[i]] = helpers[i];
        plan.target[helpers[i]] = remotes[i];
    }
    return plan.relayBytes > 0 && plan.relayBytes < bytes;
}

HcclResult MixedSend(ThreadHandle thread, const ChannelInfo &channel, uint8_t *scratch,
    const uint8_t *source, uint64_t bytes, uint64_t chunk, uint64_t remoteOffset)
{
    auto *remote = static_cast<uint8_t *>(channel.remoteCclMem.addr) + remoteOffset;
    uint32_t slot = 0;
    for (uint64_t offset = 0; offset < bytes;) {
        const uint64_t length = std::min(chunk, bytes - offset);
        const uint64_t slotOffset = static_cast<uint64_t>(slot) * chunk;
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, slot, CUSTOM_TIMEOUT));
        // This worker orders each Write before the next LocalCopy, so its source
        // slot can be reused. Remote destination slots still require their credits.
        CHK_RET(HcommLocalCopyOnThread(thread, scratch, source + offset, length));
        CHK_RET(HcommWriteOnThread(thread, channel.handle, remote + slotOffset, scratch, length));
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, slot));
        offset += length;
        slot ^= 1U;
    }
    // Drain both slots, including the unused initial credit for one-chunk segments.
    for (uint32_t i = 0; i < 2; ++i) {
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, i, CUSTOM_TIMEOUT));
    }
    CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, SCATTER_PIPE_RELEASE));
    return HCCL_SUCCESS;
}

HcclResult MixedReceive(ThreadHandle thread, const ChannelInfo &channel, uint8_t *scratch,
    uint8_t *output, uint64_t bytes, uint64_t chunk)
{
    for (uint32_t slot = 0; slot < 2; ++slot) {
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, slot));
    }
    uint32_t slot = 0;
    for (uint64_t offset = 0; offset < bytes;) {
        const uint64_t length = std::min(chunk, bytes - offset);
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, slot, CUSTOM_TIMEOUT));
        CHK_RET(HcommLocalCopyOnThread(thread, output + offset,
            scratch + static_cast<uint64_t>(slot) * chunk, length));
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, slot));
        offset += length;
        slot ^= 1U;
    }
    CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, SCATTER_PIPE_RELEASE, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult MixedForward(ThreadHandle thread, const ChannelInfo &fromRoot, const ChannelInfo &toRemote,
    uint8_t *scratch, uint64_t bytes, uint64_t chunk)
{
    // Remote's first two slots belong to its direct root connection.
    auto *remote = static_cast<uint8_t *>(toRemote.remoteCclMem.addr) + 2 * chunk;
    for (uint32_t slot = 0; slot < 2; ++slot) {
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, fromRoot.handle, slot));
    }
    uint32_t slot = 0;
    for (uint64_t offset = 0; offset < bytes;) {
        const uint64_t length = std::min(chunk, bytes - offset);
        const uint64_t slotOffset = static_cast<uint64_t>(slot) * chunk;
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, fromRoot.handle, slot, CUSTOM_TIMEOUT));
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, toRemote.handle, slot, CUSTOM_TIMEOUT));
        // Forward directly from the registered receive slot, without a second staging copy.
        CHK_RET(HcommWriteOnThread(thread, toRemote.handle, remote + slotOffset, scratch + slotOffset, length));
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, toRemote.handle, slot));
        // Root can reuse this slot only after the forwarding write has consumed its source.
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, fromRoot.handle, slot));
        offset += length;
        slot ^= 1U;
    }
    for (uint32_t i = 0; i < 2; ++i) {
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, toRemote.handle, i, CUSTOM_TIMEOUT));
    }
    CHK_RET(HcommChannelNotifyRecordOnThread(thread, toRemote.handle, SCATTER_PIPE_RELEASE));
    CHK_RET(HcommChannelNotifyWaitOnThread(thread, fromRoot.handle, SCATTER_PIPE_RELEASE, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult ExecMixed(const OpParam &param, const AlgResourceCtx &ctx, uint64_t bytes, const MixedPlan &plan)
{
    const ThreadHandle mainThread = ctx.aicpuThread;
    auto *local = static_cast<uint8_t *>(ctx.localBuffer.addr);
    auto *output = static_cast<uint8_t *>(param.outputPtr);
    const uint64_t chunk = plan.chunkBytes;
    // Validate the complete channel map before submitting any communication tasks.
    std::vector<const ChannelInfo *> peers(param.rankSize, nullptr);
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (rank == param.myRank) {
            continue;
        }
        peers[rank] = FindPeer(ctx, rank);
        if (peers[rank] == nullptr || peers[rank]->remoteCclMem.addr == nullptr ||
            peers[rank]->notifyNum < SCATTER_CHANNEL_NOTIFY_NUM) {
            return HCCL_E_PARA;
        }
    }
    if (param.myRank == param.root) {
        CHK_PTR_NULL(param.inputPtr);
        const auto *input = static_cast<const uint8_t *>(param.inputPtr);
        for (uint32_t i = 1; i < param.rankSize; ++i) {
            CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, ctx.threads[i], 0));
        }
        uint32_t workerIndex = 1;
        for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
            if (peer == param.root) {
                continue;
            }
            const ThreadHandle worker = ctx.threads[workerIndex];
            auto *scratch = local + static_cast<uint64_t>(workerIndex - 1) * chunk;
            CHK_RET(HcommThreadNotifyWaitOnThread(worker, 0, CUSTOM_TIMEOUT));
            const uint32_t target = plan.target[peer];
            if (target != INVALID_VALUE_RANKID) {
                // Relay first, then helper's own slice on the same root-helper channel.
                CHK_RET(MixedSend(worker, *peers[peer], scratch,
                    input + static_cast<uint64_t>(target) * bytes, plan.relayBytes, chunk, 0));
                CHK_RET(MixedSend(worker, *peers[peer], scratch,
                    input + static_cast<uint64_t>(peer) * bytes, bytes, chunk, 0));
            } else {
                const uint64_t offset = plan.helper[peer] == INVALID_VALUE_RANKID ? 0 : plan.relayBytes;
                CHK_RET(MixedSend(worker, *peers[peer], scratch,
                    input + static_cast<uint64_t>(peer) * bytes + offset, bytes - offset, chunk, 0));
            }
            CHK_RET(HcommThreadNotifyRecordOnThread(worker, mainThread, workerIndex));
            ++workerIndex;
        }
        for (uint32_t i = 1; i < param.rankSize; ++i) {
            CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, i, CUSTOM_TIMEOUT));
        }
        const auto *ownSlice = input + static_cast<uint64_t>(param.root) * bytes;
        if (ownSlice != output) {
            CHK_RET(HcommLocalCopyOnThread(mainThread, output, ownSlice, bytes));
        }
        return HCCL_SUCCESS;
    }

    const ChannelInfo &rootChannel = *peers[param.root];
    const uint32_t target = plan.target[param.myRank];
    if (target != INVALID_VALUE_RANKID) {
        CHK_RET(MixedForward(mainThread, rootChannel, *peers[target], local, plan.relayBytes, chunk));
        return MixedReceive(mainThread, rootChannel, local, output, bytes, chunk);
    }
    const uint32_t helper = plan.helper[param.myRank];
    if (helper == INVALID_VALUE_RANKID) {
        return MixedReceive(mainThread, rootChannel, local, output, bytes, chunk);
    }

    // A remote rank receives disjoint output ranges concurrently on two workers.
    const ThreadHandle directThread = ctx.threads[1];
    const ThreadHandle relayThread = ctx.threads[2];
    CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, directThread, 0));
    CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, relayThread, 0));
    CHK_RET(HcommThreadNotifyWaitOnThread(directThread, 0, CUSTOM_TIMEOUT));
    CHK_RET(MixedReceive(directThread, rootChannel, local, output + plan.relayBytes,
        bytes - plan.relayBytes, chunk));
    CHK_RET(HcommThreadNotifyRecordOnThread(directThread, mainThread, 1));
    CHK_RET(HcommThreadNotifyWaitOnThread(relayThread, 0, CUSTOM_TIMEOUT));
    CHK_RET(MixedReceive(relayThread, *peers[helper], local + 2 * chunk, output, plan.relayBytes, chunk));
    CHK_RET(HcommThreadNotifyRecordOnThread(relayThread, mainThread, 2));
    CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, 1, CUSTOM_TIMEOUT));
    CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, 2, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}


bool RangesDisjoint(const void *first, uint64_t firstBytes, const void *second, uint64_t secondBytes)
{
    const uintptr_t firstStart = reinterpret_cast<uintptr_t>(first);
    const uintptr_t secondStart = reinterpret_cast<uintptr_t>(second);
    const uintptr_t maxAddress = std::numeric_limits<uintptr_t>::max();
    if (firstBytes > maxAddress - firstStart || secondBytes > maxAddress - secondStart) {
        return false;
    }
    return firstStart + firstBytes <= secondStart || secondStart + secondBytes <= firstStart;
}


// V4: one input staging copy, then each receiver independently reads its own slice.
// Gate on total input bytes so this covers the 512KB case under either size convention.
bool UseSmallRead(const OpParam &param, const AlgResourceCtx &ctx, uint64_t bytes)
{
    constexpr uint64_t maxTotalBytes = 8ULL * 1024 * 1024;
    if (param.rankSize <= 1 || bytes < SCATTER_PARALLEL_THRESHOLD ||
        bytes > maxTotalBytes / param.rankSize || ctx.channels.size() != param.rankSize - 1) {
        return false;
    }
    // All ranks see the same minimum through the existing full peer channel list.
    uint64_t minBufferBytes = ctx.localBuffer.size;
    for (const ChannelInfo &channel : ctx.channels) {
        minBufferBytes = std::min(minBufferBytes, channel.remoteCclMem.size);
    }
    return bytes * param.rankSize <= minBufferBytes;
}

HcclResult ExecSmallRead(const OpParam &param, const AlgResourceCtx &ctx, uint64_t bytes)
{
    constexpr uint64_t fanout = 4;
    const ThreadHandle thread = ctx.aicpuThread;
    auto *local = static_cast<uint8_t *>(ctx.localBuffer.addr);
    const uint64_t totalBytes = bytes * param.rankSize;
    const uint64_t relativeRank = (static_cast<uint64_t>(param.myRank) + param.rankSize - param.root) %
        param.rankSize;
    std::vector<const ChannelInfo *> children;
    const ChannelInfo *parent = nullptr;
    const ChannelInfo *rootChannel = nullptr;
    for (uint64_t child = relativeRank * fanout + 1;
         child < param.rankSize && child <= relativeRank * fanout + fanout; ++child) {
        const uint32_t rank = static_cast<uint32_t>((child + param.root) % param.rankSize);
        const ChannelInfo *channel = FindPeer(ctx, rank);
        if (channel == nullptr || channel->notifyNum < SCATTER_CHANNEL_NOTIFY_NUM) {
            return HCCL_E_PARA;
        }
        children.push_back(channel);
    }
    if (param.myRank != param.root) {
        const uint64_t parentRelative = (relativeRank - 1) / fanout;
        const uint32_t parentRank = static_cast<uint32_t>((parentRelative + param.root) % param.rankSize);
        parent = FindPeer(ctx, parentRank);
        rootChannel = FindPeer(ctx, param.root);
        if (parent == nullptr || parent->notifyNum < SCATTER_CHANNEL_NOTIFY_NUM || rootChannel == nullptr ||
            rootChannel->remoteCclMem.addr == nullptr || rootChannel->remoteCclMem.size < totalBytes) {
            return HCCL_E_PARA;
        }
    }

    if (param.myRank == param.root) {
        CHK_PTR_NULL(param.inputPtr);
        if (param.inputPtr != local) {
            CHK_RET(HcommLocalCopyOnThread(thread, local, param.inputPtr, totalBytes));
        }
    } else {
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, parent->handle, SCATTER_NOTIFY_DATA, CUSTOM_TIMEOUT));
    }
    // Transitive notification ordering preserves root staging visibility.
    for (const ChannelInfo *child : children) {
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, child->handle, SCATTER_NOTIFY_DATA));
    }

    const auto *ownSlice = param.myRank == param.root ?
        local + static_cast<uint64_t>(param.root) * bytes : local;
    const bool rootCopyEarly = param.myRank == param.root && ownSlice != param.outputPtr &&
        RangesDisjoint(param.outputPtr, bytes, local, ctx.localBuffer.size);
    if (param.myRank != param.root) {
        const auto *source = static_cast<const uint8_t *>(rootChannel->remoteCclMem.addr) +
            static_cast<uint64_t>(param.myRank) * bytes;
        CHK_RET(HcommReadOnThread(thread, rootChannel->handle, local, source, bytes));
        if (local != param.outputPtr) {
            CHK_RET(HcommLocalCopyOnThread(thread, param.outputPtr, local, bytes));
        }
    } else if (rootCopyEarly) {
        // Input is already fully staged: this is safe even if output aliases user input.
        CHK_RET(HcommLocalCopyOnThread(thread, param.outputPtr, ownSlice, bytes));
    }
    for (const ChannelInfo *child : children) {
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, child->handle, SCATTER_NOTIFY_CONSUMED,
            CUSTOM_TIMEOUT));
    }
    if (param.myRank != param.root) {
        // The subtree ACK certifies every descendant read and output copy has completed.
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, parent->handle, SCATTER_NOTIFY_CONSUMED));
    } else if (ownSlice != param.outputPtr && !rootCopyEarly) {
        CHK_RET(HcommLocalCopyOnThread(thread, param.outputPtr, ownSlice, bytes));
    }
    return HCCL_SUCCESS;
}

} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    if (param.rankSize == 0 || param.myRank >= param.rankSize || param.root >= param.rankSize ||
        param.dataType != HCCL_DATA_TYPE_FP32 ||
        param.count > std::numeric_limits<uint64_t>::max() / sizeof(float) / param.rankSize) {
        return HCCL_E_PARA;
    }
    const uint64_t bytes = param.count * sizeof(float);
    if (bytes == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(param.outputPtr);
    CHK_PTR_NULL(resCtx.localBuffer.addr);
    if (UseSmallRead(param, resCtx, bytes)) {
        return ExecSmallRead(param, resCtx, bytes);
    }
    MixedPlan mixedPlan;
    if (BuildMixedPlan(param, resCtx, bytes, mixedPlan)) {
        return ExecMixed(param, resCtx, bytes, mixedPlan);
    }
    if (bytes >= SCATTER_PARALLEL_THRESHOLD && param.rankSize > 1) {
        const ChannelInfo *rootChannel = FindPeer(resCtx, param.root);
        if (param.myRank != param.root && rootChannel == nullptr) {
            return HCCL_E_PARA;
        }
        const uint64_t rootBufferBytes = param.myRank == param.root ? resCtx.localBuffer.size :
            rootChannel->remoteCclMem.size;
        if (RootSlotSize(rootBufferBytes, param.rankSize) >= sizeof(float)) {
            return ExecPipelined(param, resCtx, bytes);
        }
    }
    const ThreadHandle thread = resCtx.aicpuThread;
    auto *output = static_cast<uint8_t *>(param.outputPtr);

    if (param.myRank == param.root) {
        CHK_PTR_NULL(param.inputPtr);
        const auto *input = static_cast<const uint8_t *>(param.inputPtr);
        for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
            if (peer == param.root) {
                continue;
            }
            const ChannelInfo *channel = FindPeer(resCtx, peer);
            if (channel == nullptr) {
                HCCL_ERROR("Scatter: missing channel for rank %u", peer);
                return HCCL_E_PARA;
            }
            const uint64_t capacity = ChunkCapacity(resCtx, *channel);
            if (capacity == 0 || channel->remoteCclMem.addr == nullptr) {
                return HCCL_E_PARA;
            }
            for (uint64_t offset = 0; offset < bytes;) {
                const uint64_t length = std::min(capacity, bytes - offset);
                // Only registered HCCL buffers participate in remote writes.
                CHK_RET(HcommLocalCopyOnThread(thread, resCtx.localBuffer.addr,
                    input + static_cast<uint64_t>(peer) * bytes + offset, length));
                CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel->handle, SCATTER_NOTIFY_READY,
                    CUSTOM_TIMEOUT));
                CHK_RET(HcommWriteOnThread(thread, channel->handle, channel->remoteCclMem.addr,
                    resCtx.localBuffer.addr, length));
                CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel->handle, SCATTER_NOTIFY_DATA));
                // Receiver has copied the chunk out before either staging buffer is reused.
                CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel->handle, SCATTER_NOTIFY_CONSUMED,
                    CUSTOM_TIMEOUT));
                offset += length;
            }
        }
        // Defer root's output until all remote source slices have been read.
        const auto *ownSlice = input + static_cast<uint64_t>(param.root) * bytes;
        if (ownSlice != output) {
            CHK_RET(HcommLocalCopyOnThread(thread, output, ownSlice, bytes));
        }
    } else {
        const ChannelInfo *channel = FindPeer(resCtx, param.root);
        if (channel == nullptr) {
            return HCCL_E_PARA;
        }
        const uint64_t capacity = ChunkCapacity(resCtx, *channel);
        if (capacity == 0) {
            return HCCL_E_PARA;
        }
        for (uint64_t offset = 0; offset < bytes;) {
            const uint64_t length = std::min(capacity, bytes - offset);
            CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel->handle, SCATTER_NOTIFY_READY));
            CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel->handle, SCATTER_NOTIFY_DATA,
                CUSTOM_TIMEOUT));
            CHK_RET(HcommLocalCopyOnThread(thread, output + offset, resCtx.localBuffer.addr, length));
            CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel->handle, SCATTER_NOTIFY_CONSUMED));
            offset += length;
        }
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
