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

namespace ops_hccl {
namespace {
// A compact, one-copy path avoids per-peer worker startup for small inputs.
constexpr uint64_t COMPACT_INPUT_LIMIT = 8ULL * 1024 * 1024;

bool UseCompactInput(uint64_t bytes, uint32_t ranks, uint64_t rootBufferSize)
{
    return ranks != 0 && bytes <= std::min(rootBufferSize, COMPACT_INPUT_LIMIT) / ranks;
}

uint64_t SlotStride(uint64_t bufferSize, uint32_t ranks)
{
    if (ranks == 0) {
        return 0;
    }
    return std::min<uint64_t>(bufferSize / (SCATTER_SLOT_COUNT * uint64_t(ranks)), 16ULL * 1024 * 1024)
        / sizeof(float) * sizeof(float);
}

uint64_t BalancedChunk(uint64_t bytes, uint64_t localSize, uint64_t remoteSize, uint32_t ranks)
{
    const uint64_t capacity = SlotStride(std::min(localSize, remoteSize), ranks);
    if (capacity == 0 || bytes == 0) {
        return 0;
    }
    const uint64_t rounds = 1 + (bytes - 1) / capacity;
    const uint64_t elements = bytes / sizeof(float);
    return (1 + (elements - 1) / rounds) * sizeof(float);
}

HcclResult CompactLane(const OpParam &param, const AlgResourceCtx &ctx, ThreadHandle thread,
    size_t lane, size_t lanes, uint64_t bytes)
{
    for (size_t i = lane; i < ctx.channels.size(); i += lanes) {
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, ctx.channels[i].handle, 0));
    }
    const auto ownInput = static_cast<const char *>(param.inputPtr) + uint64_t(param.root) * bytes;
    if (lane == 0 && ownInput != param.outputPtr) {
        CHK_RET(HcommLocalCopyOnThread(thread, param.outputPtr, ownInput, bytes));
    }
    for (size_t i = lane; i < ctx.channels.size(); i += lanes) {
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, ctx.channels[i].handle, SCATTER_SLOT_COUNT, CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
}

HcclResult SendCompact(const OpParam &param, const AlgResourceCtx &ctx, uint64_t bytes)
{
    const auto thread = ctx.aicpuThread;
    CHK_RET(HcommLocalCopyOnThread(thread, ctx.localBuffer.addr, param.inputPtr, bytes * param.rankSize));
    // Four groups bound each lane's serial notify chain; the main thread is a lane.
    const size_t lanes = ctx.channels.size() >= 8 && ctx.threads.size() >= 4 ? 4 : 1;
    for (size_t lane = 1; lane < lanes; ++lane) {
        const auto worker = ctx.threads[lane];
        CHK_RET(HcommThreadNotifyRecordOnThread(thread, worker, 0));
        CHK_RET(HcommThreadNotifyWaitOnThread(worker, 0, CUSTOM_TIMEOUT));
        CHK_RET(CompactLane(param, ctx, worker, lane, lanes, bytes));
        CHK_RET(HcommThreadNotifyRecordOnThread(worker, thread, static_cast<uint32_t>(lane)));
    }
    CHK_RET(CompactLane(param, ctx, thread, 0, lanes, bytes));
    for (size_t lane = 1; lane < lanes; ++lane) {
        CHK_RET(HcommThreadNotifyWaitOnThread(thread, static_cast<uint32_t>(lane), CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
}

HcclResult SendPeer(const OpParam &param, const AlgResourceCtx &ctx, size_t lane, uint64_t bytes)
{
    const auto &channel = ctx.channels[lane];
    const auto thread = ctx.threads[lane + 1];
    const uint64_t chunk = BalancedChunk(bytes, ctx.localBuffer.size, channel.remoteCclMem.size, param.rankSize);
    CHK_PRT_RET(chunk == 0, HCCL_ERROR("Scatter buffer is too small"), HCCL_E_PARA);
    const uint64_t stride = SlotStride(channel.remoteCclMem.size, param.rankSize);
    const uint64_t base = uint64_t(channel.remoteRank) * SCATTER_SLOT_COUNT * stride;
    const auto input = static_cast<const char *>(param.inputPtr) + uint64_t(channel.remoteRank) * bytes;
    const auto staging = static_cast<char *>(channel.remoteCclMem.addr) + base;
    uint64_t step = 0;
    for (uint64_t offset = 0; offset < bytes; ++step) {
        const uint32_t slot = step % SCATTER_SLOT_COUNT;
        // Even the first write waits until the receiver releases its local slot.
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, SCATTER_SLOT_COUNT + slot, CUSTOM_TIMEOUT));
        const uint64_t length = std::min(chunk, bytes - offset);
        CHK_RET(HcommWriteOnThread(thread, channel.handle, staging + slot * stride, input + offset, length));
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, slot));
        offset += length;
    }
    // Drain both slots before returning control to the caller or another op.
    for (uint32_t slot = 0; slot < std::min<uint64_t>(step, SCATTER_SLOT_COUNT); ++slot) {
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, SCATTER_SLOT_COUNT + slot, CUSTOM_TIMEOUT));
    }
    CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, 5));
    return HCCL_SUCCESS;
}

HcclResult ReceiveRoot(const OpParam &param, const AlgResourceCtx &ctx, uint64_t bytes)
{
    const ChannelInfo *channel = nullptr;
    for (const auto &candidate : ctx.channels) {
        if (candidate.remoteRank == param.root) {
            channel = &candidate;
            break;
        }
    }
    CHK_PTR_NULL(channel);
    if (UseCompactInput(bytes, param.rankSize, channel->remoteCclMem.size)) {
        const auto remote = static_cast<const char *>(channel->remoteCclMem.addr) + uint64_t(param.myRank) * bytes;
        CHK_RET(HcommChannelNotifyWaitOnThread(ctx.aicpuThread, channel->handle, 0, CUSTOM_TIMEOUT));
        CHK_RET(HcommReadOnThread(ctx.aicpuThread, channel->handle, param.outputPtr, remote, bytes));
        CHK_RET(HcommChannelNotifyRecordOnThread(ctx.aicpuThread, channel->handle, SCATTER_SLOT_COUNT));
        return HCCL_SUCCESS;
    }
    const uint64_t chunk = BalancedChunk(bytes, ctx.localBuffer.size, channel->remoteCclMem.size, param.rankSize);
    CHK_PRT_RET(chunk == 0, HCCL_ERROR("Scatter buffer is too small"), HCCL_E_PARA);
    const uint64_t stride = SlotStride(ctx.localBuffer.size, param.rankSize);
    const uint64_t base = uint64_t(param.myRank) * SCATTER_SLOT_COUNT * stride;
    const auto local = static_cast<const char *>(ctx.localBuffer.addr) + base;
    const auto output = static_cast<char *>(param.outputPtr);
    const auto thread = ctx.aicpuThread;
    const uint64_t rounds = 1 + (bytes - 1) / chunk;
    for (uint32_t slot = 0; slot < std::min<uint64_t>(rounds, SCATTER_SLOT_COUNT); ++slot) {
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel->handle, SCATTER_SLOT_COUNT + slot));
    }
    uint64_t step = 0;
    for (uint64_t offset = 0; offset < bytes; ++step) {
        const uint32_t slot = step % SCATTER_SLOT_COUNT;
        const uint64_t length = std::min(chunk, bytes - offset);
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel->handle, slot, CUSTOM_TIMEOUT));
        CHK_RET(HcommLocalCopyOnThread(thread, output + offset, local + slot * stride, length));
        // Release this receive slot only after its output copy completes.
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel->handle, SCATTER_SLOT_COUNT + slot));
        offset += length;
    }
    CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel->handle, 5, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}
// A rank-pair route is valid for every root. Contiguous halves are the preferred
// placement for the contest's 2 x 8 topology; other placements only affect cost.
const ChannelInfo *ChannelTo(const AlgResourceCtx &ctx, uint32_t peer)
{
    for (const auto &channel : ctx.channels) {
        if (channel.remoteRank == peer) {
            return &channel;
        }
    }
    return nullptr;
}

uint32_t CompactLeader(uint32_t group, uint32_t root)
{
    // Four contiguous ranks share a server in the contest placement.
    // The root's group chooses its next member so every leader is non-root.
    return group == root / 4 ? group * 4 + (root + 1) % 4 : group * 4;
}

HcclResult CompactTree(const OpParam &param, const AlgResourceCtx &ctx, uint64_t bytes)
{
    // Channel 0 publishes readiness, 1 acknowledges completed reads.
    // Keep acknowledgements separate from large-path initial credits (2/3):
    // a leaf may enqueue the next collective before its parent consumes its ack.
    const auto thread = ctx.aicpuThread;
    if (param.myRank == param.root) {
        CHK_RET(HcommLocalCopyOnThread(thread, ctx.localBuffer.addr, param.inputPtr, bytes * param.rankSize));
        // Release the three larger groups before the root's three-peer group.
        for (uint32_t i = 1; i <= 4; ++i) {
            const auto leader = ChannelTo(ctx, CompactLeader((param.root / 4 + i) % 4, param.root));
            CHK_PTR_NULL(leader);
            CHK_RET(HcommChannelNotifyRecordOnThread(thread, leader->handle, 0));
        }
        const auto own = static_cast<const char *>(param.inputPtr) + uint64_t(param.root) * bytes;
        if (own != param.outputPtr) {
            CHK_RET(HcommLocalCopyOnThread(thread, param.outputPtr, own, bytes));
        }
        for (uint32_t i = 1; i <= 4; ++i) {
            const auto leader = ChannelTo(ctx, CompactLeader((param.root / 4 + i) % 4, param.root));
            CHK_PTR_NULL(leader);
            CHK_RET(HcommChannelNotifyWaitOnThread(thread, leader->handle, 1, CUSTOM_TIMEOUT));
        }
        return HCCL_SUCCESS;
    }

    const uint32_t group = param.myRank / 4;
    const uint32_t leaderRank = CompactLeader(group, param.root);
    const bool isLeader = param.myRank == leaderRank;
    const auto parent = ChannelTo(ctx, isLeader ? param.root : leaderRank);
    const auto root = ChannelTo(ctx, param.root);
    CHK_PTR_NULL(parent);
    CHK_PTR_NULL(root);
    CHK_RET(HcommChannelNotifyWaitOnThread(thread, parent->handle, 0, CUSTOM_TIMEOUT));
    if (isLeader) {
        for (uint32_t peer = group * 4; peer < group * 4 + 4; ++peer) {
            if (peer == param.root || peer == leaderRank) {
                continue;
            }
            const auto child = ChannelTo(ctx, peer);
            CHK_PTR_NULL(child);
            CHK_RET(HcommChannelNotifyRecordOnThread(thread, child->handle, 0));
        }
    }
    // The tree carries only notifications. Every rank reads its own slice
    // directly from root; no intermediate scratch or data copy is added.
    const auto remote = static_cast<const char *>(root->remoteCclMem.addr) + uint64_t(param.myRank) * bytes;
    CHK_RET(HcommReadOnThread(thread, root->handle, param.outputPtr, remote, bytes));
    if (isLeader) {
        for (uint32_t peer = group * 4; peer < group * 4 + 4; ++peer) {
            if (peer == param.root || peer == leaderRank) {
                continue;
            }
            const auto child = ChannelTo(ctx, peer);
            CHK_PTR_NULL(child);
            CHK_RET(HcommChannelNotifyWaitOnThread(thread, child->handle, 1, CUSTOM_TIMEOUT));
        }
    }
    // A leader releases root only after its read and every child read.
    CHK_RET(HcommChannelNotifyRecordOnThread(thread, parent->handle, 1));
    return HCCL_SUCCESS;
}

uint64_t RelayCapacity(const AlgResourceCtx &ctx)
{
    uint64_t capacity = ctx.localBuffer.size;
    for (const auto &channel : ctx.channels) {
        capacity = std::min(capacity, channel.remoteCclMem.size);
    }
    // A near frame holds C own bytes plus a 4/11 prefix. Two frames fit.
    // C in 44-byte units makes the two frames exactly 120 bytes per unit.
    // Preserve the established fallback for very small CCL allocations.
    return capacity < 176 ? 0 : std::min<uint64_t>(capacity / 120 * 44, 32ULL * 1024 * 1024);
}

uint64_t RelayPrefix(uint64_t length)
{
    return ((length / sizeof(float) * 4 + 10) / 11) * sizeof(float);
}

// Keep full-size steady-state transfers; only taper a sizable final block.
// One extra round reduces its exposed output-copy tail without splitting every
// block. All participants derive the same schedule from bytes and chunk.
bool SplitRelayTail(uint64_t bytes, uint64_t chunk)
{
    const uint64_t tail = 1 + (bytes - 1) % chunk;
    return tail >= 4ULL * 1024 * 1024;
}

uint64_t RelayLength(uint64_t bytes, uint64_t chunk, uint64_t offset)
{
    const uint64_t tailStart = (bytes - 1) / chunk * chunk;
    if (offset == tailStart && SplitRelayTail(bytes, chunk)) {
        const uint64_t elements = (bytes - offset) / sizeof(float);
        return ((elements * 3 + 3) / 4) * sizeof(float);
    }
    return std::min(chunk, bytes - offset);
}

uint64_t RelayTransferRounds(uint64_t bytes, uint64_t chunk)
{
    return 1 + (bytes - 1) / chunk + (SplitRelayTail(bytes, chunk) ? 1 : 0);
}

HcclResult InitialCredits(ThreadHandle thread, ChannelHandle channel, uint64_t rounds)
{
    for (uint32_t slot = 0; slot < std::min<uint64_t>(rounds, SCATTER_SLOT_COUNT); ++slot) {
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel, SCATTER_SLOT_COUNT + slot));
    }
    return HCCL_SUCCESS;
}

HcclResult DrainCredits(ThreadHandle thread, ChannelHandle channel, uint64_t rounds)
{
    for (uint32_t slot = 0; slot < std::min<uint64_t>(rounds, SCATTER_SLOT_COUNT); ++slot) {
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel, SCATTER_SLOT_COUNT + slot, CUSTOM_TIMEOUT));
    }
    // A receiver must not publish next-call credits before this drain finishes.
    // Index 5 is reused only after the final prefix-ready has been consumed.
    CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel, 5));
    return HCCL_SUCCESS;
}

HcclResult SendSplitPeer(const OpParam &param, const AlgResourceCtx &ctx, size_t lane,
    uint64_t bytes, uint64_t chunk)
{
    const auto &channel = ctx.channels[lane];
    const auto thread = ctx.threads[lane + 1];
    const uint32_t peer = channel.remoteRank;
    const uint32_t partner = peer ^ 8U;
    const bool near = peer / 8 == param.root / 8;
    const uint64_t stride = near ? chunk + RelayPrefix(chunk) : chunk;
    const uint64_t recvBase = !near && partner != param.root ? RelayPrefix(chunk) : 0;
    const auto input = static_cast<const char *>(param.inputPtr);
    const auto remote = static_cast<char *>(channel.remoteCclMem.addr);
    uint64_t step = 0;
    for (uint64_t offset = 0; offset < bytes; ++step) {
        const uint32_t slot = step % SCATTER_SLOT_COUNT;
        const uint64_t length = RelayLength(bytes, chunk, offset);
        const uint64_t prefix = RelayPrefix(length);
        const uint64_t skip = !near && partner != param.root ? prefix : 0;
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, SCATTER_SLOT_COUNT + slot, CUSTOM_TIMEOUT));
        if (near) {
            // Publish the relay prefix before sending the local-rank slice.
            // Separate notifications let forwarding overlap the following write.
            CHK_RET(HcommWriteOnThread(thread, channel.handle, remote + slot * stride + chunk,
                input + uint64_t(partner) * bytes + offset, prefix));
            CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, 2 * SCATTER_SLOT_COUNT + slot));
        }
        if (length > skip) {
            CHK_RET(HcommWriteOnThread(thread, channel.handle, remote + slot * stride + recvBase,
                input + uint64_t(peer) * bytes + offset + skip, length - skip));
        }
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, slot));
        offset += length;
    }
    return DrainCredits(thread, channel.handle, step);
}

HcclResult ReceivePart(const OpParam &param, const AlgResourceCtx &ctx, const ChannelInfo &channel,
    ThreadHandle thread, uint64_t bytes, uint64_t chunk, bool prefixPart, bool split)
{
    // Fixed prefix reservation separates independent producers even on tails.
    const uint64_t recvBase = split && !prefixPart ? RelayPrefix(chunk) : 0;
    const auto local = static_cast<const char *>(ctx.localBuffer.addr) + recvBase;
    const auto output = static_cast<char *>(param.outputPtr);
    CHK_RET(InitialCredits(thread, channel.handle, RelayTransferRounds(bytes, chunk)));
    uint64_t step = 0;
    for (uint64_t offset = 0; offset < bytes; ++step) {
        const uint32_t slot = step % SCATTER_SLOT_COUNT;
        const uint64_t length = RelayLength(bytes, chunk, offset);
        const uint64_t prefix = split ? RelayPrefix(length) : 0;
        const uint64_t skip = prefixPart ? 0 : prefix;
        const uint64_t received = prefixPart ? prefix : length - skip;
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, slot, CUSTOM_TIMEOUT));
        if (received != 0) {
            CHK_RET(HcommLocalCopyOnThread(thread, output + offset + skip, local + slot * chunk, received));
        }
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, SCATTER_SLOT_COUNT + slot));
        offset += length;
    }
    CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, 5, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

// The main lane receives prefix-ready and explicitly starts forwarding.
// Both root-channel waits stay on that lane; the worker only uses its partner.
HcclResult ForwardSplit(const AlgResourceCtx &ctx, const ChannelInfo &partner,
    uint64_t bytes, uint64_t chunk)
{
    const auto worker = ctx.threads[1];
    const auto local = static_cast<const char *>(ctx.localBuffer.addr);
    const auto remote = static_cast<char *>(partner.remoteCclMem.addr);
    const uint64_t stride = chunk + RelayPrefix(chunk);
    uint64_t step = 0;
    for (uint64_t offset = 0; offset < bytes; ++step) {
        const uint32_t slot = step % SCATTER_SLOT_COUNT;
        const uint64_t length = RelayLength(bytes, chunk, offset);
        CHK_RET(HcommThreadNotifyWaitOnThread(worker, 0, CUSTOM_TIMEOUT));
        CHK_RET(HcommChannelNotifyWaitOnThread(worker, partner.handle, SCATTER_SLOT_COUNT + slot, CUSTOM_TIMEOUT));
        CHK_RET(HcommWriteOnThread(worker, partner.handle, remote + slot * chunk,
            local + slot * stride + chunk, RelayPrefix(length)));
        CHK_RET(HcommChannelNotifyRecordOnThread(worker, partner.handle, slot));
        // Keep slot completion notifications separate from the final join.
        CHK_RET(HcommThreadNotifyRecordOnThread(worker, ctx.aicpuThread, 1 + slot));
        offset += length;
    }
    CHK_RET(DrainCredits(worker, partner.handle, step));
    CHK_RET(HcommThreadNotifyRecordOnThread(worker, ctx.aicpuThread, 3));
    return HCCL_SUCCESS;
}

HcclResult ReceiveSplit(const OpParam &param, const AlgResourceCtx &ctx,
    uint64_t bytes, uint64_t chunk)
{
    const auto root = ChannelTo(ctx, param.root);
    CHK_PTR_NULL(root);
    const auto thread = ctx.aicpuThread;
    const uint32_t partnerRank = param.myRank ^ 8U;
    const bool near = param.myRank / 8 == param.root / 8;
    if (!near) {
        const bool split = partnerRank != param.root;
        if (split) {
            const auto partner = ChannelTo(ctx, partnerRank);
            CHK_PTR_NULL(partner);
            const auto worker = ctx.threads[1];
            CHK_RET(HcommThreadNotifyRecordOnThread(thread, worker, 0));
            CHK_RET(HcommThreadNotifyWaitOnThread(worker, 0, CUSTOM_TIMEOUT));
            CHK_RET(ReceivePart(param, ctx, *partner, worker, bytes, chunk, true, true));
            CHK_RET(HcommThreadNotifyRecordOnThread(worker, thread, 1));
        }
        CHK_RET(ReceivePart(param, ctx, *root, thread, bytes, chunk, false, split));
        if (split) {
            CHK_RET(HcommThreadNotifyWaitOnThread(thread, 1, CUSTOM_TIMEOUT));
        }
        return HCCL_SUCCESS;
    }
    const auto partner = ChannelTo(ctx, partnerRank);
    CHK_PTR_NULL(partner);
    const auto local = static_cast<const char *>(ctx.localBuffer.addr);
    const auto worker = ctx.threads[1];
    const uint64_t stride = chunk + RelayPrefix(chunk);
    CHK_RET(ForwardSplit(ctx, *partner, bytes, chunk));
    const auto output = static_cast<char *>(param.outputPtr);
    CHK_RET(InitialCredits(thread, root->handle, RelayTransferRounds(bytes, chunk)));
    uint64_t step = 0;
    for (uint64_t offset = 0; offset < bytes; ++step) {
        const uint32_t slot = step % SCATTER_SLOT_COUNT;
        const uint64_t length = RelayLength(bytes, chunk, offset);
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, root->handle, 2 * SCATTER_SLOT_COUNT + slot, CUSTOM_TIMEOUT));
        CHK_RET(HcommThreadNotifyRecordOnThread(thread, worker, 0));
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, root->handle, slot, CUSTOM_TIMEOUT));
        CHK_RET(HcommLocalCopyOnThread(thread, output + offset, local + slot * stride, length));
        CHK_RET(HcommThreadNotifyWaitOnThread(thread, 1 + slot, CUSTOM_TIMEOUT));
        // Release only after both the local output copy and forwarding write.
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, root->handle, SCATTER_SLOT_COUNT + slot));
        offset += length;
    }
    CHK_RET(HcommChannelNotifyWaitOnThread(thread, root->handle, 5, CUSTOM_TIMEOUT));
    CHK_RET(HcommThreadNotifyWaitOnThread(thread, 3, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(param.dataType != HCCL_DATA_TYPE_FP32 || param.rankSize == 0 || param.root >= param.rankSize,
        HCCL_ERROR("Invalid Scatter parameters"), HCCL_E_PARA);
    const uint64_t bytes = param.count * sizeof(float);
    const auto rootChannel = param.myRank == param.root ? nullptr : ChannelTo(resCtx, param.root);
    if (param.myRank != param.root) {
        CHK_PTR_NULL(rootChannel);
    }
    const uint64_t rootCapacity = rootChannel ? rootChannel->remoteCclMem.size : resCtx.localBuffer.size;
    const bool compact = UseCompactInput(bytes, param.rankSize, rootCapacity);
    if (param.rankSize == 16 && compact) {
        return CompactTree(param, resCtx, bytes);
    }
    const uint64_t relayCapacity = param.rankSize == 16 && !compact ? RelayCapacity(resCtx) : 0;
    const bool relay = relayCapacity != 0 && resCtx.threads.size() >= param.rankSize;
    const uint64_t relayRounds = relay ? 1 + (bytes - 1) / relayCapacity : 0;
    const uint64_t relayChunk = relay ? (1 + (param.count - 1) / relayRounds) * sizeof(float) : 0;
    if (param.myRank != param.root) {
        if (relay) {
            return ReceiveSplit(param, resCtx, bytes, relayChunk);
        }
        return ReceiveRoot(param, resCtx, bytes);
    }
    if (UseCompactInput(bytes, param.rankSize, resCtx.localBuffer.size)) {
        return SendCompact(param, resCtx, bytes);
    }
    CHK_PRT_RET(resCtx.threads.size() < resCtx.channels.size() + 1,
        HCCL_ERROR("Insufficient Scatter threads"), HCCL_E_PARA);
    // All worker starts are ordered after the host/device handshake.
    for (size_t i = 0; i < resCtx.channels.size(); ++i) {
        CHK_RET(HcommThreadNotifyRecordOnThread(resCtx.aicpuThread, resCtx.threads[i + 1], 0));
        CHK_RET(HcommThreadNotifyWaitOnThread(resCtx.threads[i + 1], 0, CUSTOM_TIMEOUT));
        if (relay) {
            CHK_RET(SendSplitPeer(param, resCtx, i, bytes, relayChunk));
        } else {
            CHK_RET(SendPeer(param, resCtx, i, bytes));
        }
        CHK_RET(HcommThreadNotifyRecordOnThread(resCtx.threads[i + 1], resCtx.aicpuThread,
            static_cast<uint32_t>(i + 1)));
    }
    const auto ownInput = static_cast<const char *>(param.inputPtr) + uint64_t(param.root) * bytes;
    if (ownInput != param.outputPtr) {
        CHK_RET(HcommLocalCopyOnThread(resCtx.aicpuThread, param.outputPtr, ownInput, bytes));
    }
    for (size_t i = 0; i < resCtx.channels.size(); ++i) {
        CHK_RET(HcommThreadNotifyWaitOnThread(resCtx.aicpuThread, static_cast<uint32_t>(i + 1), CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
