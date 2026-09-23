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
#include <array>
#include <limits>

namespace ops_hccl {
namespace {
    const ChannelInfo *FindChannel(const AlgResourceCtx &ctx, uint32_t peer)
    {
        for (const auto &channel : ctx.channels) {
            if (channel.remoteRank == peer) {
                return &channel;
            }
        }
        return nullptr;
    }

    uint64_t BalancedChunk(uint64_t bytes, uint64_t capacity)
    {
        const uint64_t maxSlice = bytes <= 64 * 1024 * 1024 ? 16 * 1024 * 1024 : SCATTER_LARGE_MAX_SLICE;
        const uint64_t limit = std::min(capacity, maxSlice);
        if (limit == 0 || bytes == 0) {
            return 0;
        }
        const uint64_t rounds = bytes / limit + (bytes % limit != 0);
        const uint64_t size = bytes / rounds + (bytes % rounds != 0);
        return std::min(limit, (size + 127) / 128 * 128);
    }

    HcclResult WriteReady(
        ThreadHandle thread, const ChannelInfo &channel, void *dst, const void *src, uint64_t bytes, uint32_t slot)
    {
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, 2 * slot, CUSTOM_TIMEOUT));
        const auto result = HcommWriteWithNotifyOnThread(thread, channel.handle, dst, src, bytes, 2 * slot + 1);
        if (result == HCCL_E_NOT_SUPPORT) {
            CHK_RET(HcommWriteOnThread(thread, channel.handle, dst, src, bytes));
            CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, 2 * slot + 1));
        } else {
            CHK_RET(result);
        }
        return HCCL_SUCCESS;
    }

    // Two-level radix-four scatter. XOR with root makes every root position valid
    // while each subtree still covers four contiguous physical rank slices.
    // Intermediate ranks read their whole subtree once, then serve its leaves.
    // Parent ACK is delayed until all descendants release this call's CCL data.
    HcclResult ExecTree(const OpParam &param, const AlgResourceCtx &ctx, uint64_t bytes)
    {
        const uint32_t relative = param.myRank ^ param.root;
        const bool leader = (relative & 3) == 0;
        const ThreadHandle main = ctx.aicpuThread;
        const ChannelInfo *parent = nullptr;
        auto *stage = static_cast<uint8_t *>(ctx.localBuffer.addr);
        if (relative == 0) {
            CHK_RET(HcommLocalCopyOnThread(main, stage, param.inputPtr, 16 * bytes));
        } else {
            const uint32_t parentRank = leader ? param.root : (param.myRank & ~3U) | (param.root & 3);
            parent = FindChannel(ctx, parentRank);
            CHK_PTR_NULL(parent);
            const uint64_t first = leader ? param.myRank & ~3U : param.myRank;
            const uint64_t length = (leader ? 4 : 1) * bytes;
            void *dest = leader ? stage + first * bytes : param.outputPtr;
            CHK_RET(HcommChannelNotifyWaitOnThread(main, parent->handle, 0, CUSTOM_TIMEOUT));
            CHK_RET(HcommReadOnThread(main, parent->handle, dest,
                static_cast<const uint8_t *>(parent->remoteCclMem.addr) + first * bytes, length));
        }
        std::array<const ChannelInfo *, 6> children{};
        uint32_t count = 0;
        if (relative == 0) {
            // Start the larger subtrees before the root's three leaves.
            for (uint32_t delta : {4U, 8U, 12U}) {
                children[count] = FindChannel(ctx, param.root ^ delta);
                CHK_PTR_NULL(children[count]);
                ++count;
            }
        }
        if (leader) {
            for (uint32_t delta : {1U, 2U, 3U}) {
                children[count] = FindChannel(ctx, param.myRank ^ delta);
                CHK_PTR_NULL(children[count]);
                ++count;
            }
            for (uint32_t i = 0; i < count; ++i)
                CHK_RET(HcommChannelNotifyRecordOnThread(main, children[i]->handle, 0));
            const auto *own = relative == 0 ? static_cast<const uint8_t *>(param.inputPtr) + param.root * bytes
                                            : stage + param.myRank * bytes;
            if (own != param.outputPtr)
                CHK_RET(HcommLocalCopyOnThread(main, param.outputPtr, own, bytes));
            for (uint32_t i = 0; i < count; ++i)
                CHK_RET(HcommChannelNotifyWaitOnThread(main, children[i]->handle, 1, CUSTOM_TIMEOUT));
        }
        if (parent != nullptr)
            CHK_RET(HcommChannelNotifyRecordOnThread(main, parent->handle, 1));
        return HCCL_SUCCESS;
    }

    // Four root queues prepare disjoint immutable regions. Each can publish its
    // region immediately, overlapping packing, direct reads and ACK collection.
    HcclResult ExecFanout(const OpParam &param, const AlgResourceCtx &ctx, uint64_t bytes)
    {
        const ThreadHandle main = ctx.aicpuThread;
        if (param.myRank != param.root) {
            const auto *root = FindChannel(ctx, param.root);
            CHK_PTR_NULL(root);
            CHK_RET(HcommChannelNotifyWaitOnThread(main, root->handle, 0, CUSTOM_TIMEOUT));
            CHK_RET(HcommReadOnThread(main, root->handle, param.outputPtr,
                static_cast<const uint8_t *>(root->remoteCclMem.addr) + param.myRank * bytes, bytes));
            CHK_RET(HcommChannelNotifyRecordOnThread(main, root->handle, 1));
            return HCCL_SUCCESS;
        }
        constexpr uint32_t groups = 4;
        for (uint32_t group = 1; group < groups; ++group) {
            CHK_RET(HcommThreadNotifyRecordOnThread(main, ctx.threads[group], 0));
            CHK_RET(HcommThreadNotifyWaitOnThread(ctx.threads[group], 0, CUSTOM_TIMEOUT));
        }
        for (uint32_t group = 0; group < groups; ++group) {
            const ThreadHandle thread = ctx.threads[group];
            uint32_t first = (param.root & ~3U) ^ (4 * group);
            uint32_t last = first + 4;
            if (first == param.root)
                ++first;
            if (last == param.root + 1)
                --last;
            CHK_RET(HcommLocalCopyOnThread(thread, static_cast<uint8_t *>(ctx.localBuffer.addr) + first * bytes,
                static_cast<const uint8_t *>(param.inputPtr) + first * bytes, (last - first) * bytes));
            for (uint32_t peer = first; peer < last; ++peer) {
                if (peer == param.root)
                    continue;
                const auto *channel = FindChannel(ctx, peer);
                CHK_PTR_NULL(channel);
                CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel->handle, 0));
            }
            if (group == 0) {
                const auto *own = static_cast<const uint8_t *>(param.inputPtr) + param.root * bytes;
                if (own != param.outputPtr)
                    CHK_RET(HcommLocalCopyOnThread(main, param.outputPtr, own, bytes));
            }
            for (uint32_t peer = first; peer < last; ++peer) {
                if (peer == param.root)
                    continue;
                const auto *channel = FindChannel(ctx, peer);
                CHK_PTR_NULL(channel);
                CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel->handle, 1, CUSTOM_TIMEOUT));
            }
            if (group != 0)
                CHK_RET(HcommThreadNotifyRecordOnThread(thread, main, group));
        }
        for (uint32_t group = 1; group < groups; ++group)
            CHK_RET(HcommThreadNotifyWaitOnThread(main, group, CUSTOM_TIMEOUT));
        return HCCL_SUCCESS;
    }

    // READY credits are sent only for slots that will be written again. A separate
    // CONSUMED joins the receiver's copy and forwarding before root may finish.
    // Own output and forwarding each have two independent slots. Both paths can
    // consume a chunk while root fills the next one; only reused slots get credits.
    HcclResult ExecRelay(const OpParam &param, const AlgResourceCtx &ctx, uint64_t bytes)
    {
        const bool rootLocal = std::binary_search(ctx.localRanks.begin(), ctx.localRanks.end(), param.root);
        std::array<uint32_t, 7> relays{};
        std::array<uint32_t, 8> targets{};
        uint32_t relayCount = 0, targetCount = 0;
        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            const bool local = std::binary_search(ctx.localRanks.begin(), ctx.localRanks.end(), rank);
            if (local == rootLocal) {
                if (rank != param.root) {
                    if (relayCount == relays.size())
                        return HCCL_E_PARA;
                    relays[relayCount++] = rank;
                }
            } else {
                if (targetCount == targets.size())
                    return HCCL_E_PARA;
                targets[targetCount++] = rank;
            }
        }
        if (relayCount != relays.size() || targetCount != targets.size())
            return HCCL_E_PARA;
        // Balance one local link carrying (1+f) slices against root's shared
        // Clos port carrying (8-7f) slices at four times the link bandwidth.
        // 1+f = (8-7f)/4 gives f=4/11; retain whole FP32 elements.
        const uint64_t relayBytes = (bytes / 4) / 11 * 4 * 4;
        const ThreadHandle main = ctx.aicpuThread;
        if (param.myRank == param.root) {
            for (uint32_t i = 0; i < ctx.channels.size(); ++i) {
                const auto &c = ctx.channels[i];
                const ThreadHandle worker = ctx.threads[i + 1];
                CHK_RET(HcommThreadNotifyRecordOnThread(main, worker, 0));
                CHK_RET(HcommThreadNotifyWaitOnThread(worker, 0, CUSTOM_TIMEOUT));
                const auto relay = std::find(relays.begin(), relays.end(), c.remoteRank);
                const uint64_t slotSize = c.remoteCclMem.size / (relay != relays.end() ? 4 : 2) / 128 * 128;
                const uint64_t chunkSize = BalancedChunk(bytes, slotSize);
                if (chunkSize == 0)
                    return HCCL_E_PARA;
                auto *stage = static_cast<uint8_t *>(c.remoteCclMem.addr);
                const auto *input = static_cast<const uint8_t *>(param.inputPtr);
                if (relay != relays.end()) {
                    const uint32_t target = targets[relay - relays.begin()];
                    const uint64_t rounds = bytes / chunkSize + (bytes % chunkSize != 0);
                    const uint64_t relayChunk = (relayBytes / rounds + (relayBytes % rounds != 0) + 127) / 128 * 128;
                    uint64_t forwarded = 0;
                    uint64_t chunk = 0;
                    for (uint64_t offset = 0; offset < bytes; ++chunk) {
                        const uint32_t slot = chunk % 2;
                        const uint64_t len = std::min(chunkSize, bytes - offset);
                        if (forwarded < relayBytes) {
                            const uint64_t part = std::min(relayChunk, relayBytes - forwarded);
                            CHK_RET(WriteReady(worker, c, stage + (2 + slot) * slotSize,
                                input + target * bytes + forwarded, part, 2 + slot));
                            forwarded += part;
                        }
                        CHK_RET(WriteReady(
                            worker, c, stage + slot * slotSize, input + c.remoteRank * bytes + offset, len, slot));
                        offset += len;
                    }
                } else {
                    uint64_t offset = c.remoteRank == targets.back() ? 0 : relayBytes;
                    uint64_t chunk = 0;
                    while (offset < bytes) {
                        const uint64_t len = std::min(chunkSize, bytes - offset);
                        const uint32_t slot = chunk % 2;
                        CHK_RET(WriteReady(
                            worker, c, stage + slot * slotSize, input + c.remoteRank * bytes + offset, len, slot));
                        offset += len;
                        ++chunk;
                    }
                }
            }
            const auto *own = static_cast<const uint8_t *>(param.inputPtr) + param.root * bytes;
            if (own != param.outputPtr)
                CHK_RET(HcommLocalCopyOnThread(main, param.outputPtr, own, bytes));
            for (uint32_t i = 0; i < ctx.channels.size(); ++i) {
                CHK_RET(HcommChannelNotifyWaitOnThread(
                    main, ctx.channels[i].handle, SCATTER_NOTIFY_CONSUMED, CUSTOM_TIMEOUT));
            }
            return HCCL_SUCCESS;
        }
        const auto *root = FindChannel(ctx, param.root);
        CHK_PTR_NULL(root);
        const auto relay = std::find(relays.begin(), relays.end(), param.myRank);
        const uint64_t slotSize = ctx.localBuffer.size / (relay != relays.end() ? 4 : 2) / 128 * 128;
        const uint64_t chunkSize = BalancedChunk(bytes, slotSize);
        if (chunkSize == 0)
            return HCCL_E_PARA;
        auto *stage = static_cast<uint8_t *>(ctx.localBuffer.addr);
        auto *output = static_cast<uint8_t *>(param.outputPtr);
        uint32_t join = 0;
        if (relay != relays.end()) {
            const auto *target = FindChannel(ctx, targets[relay - relays.begin()]);
            CHK_PTR_NULL(target);
            join = static_cast<uint32_t>(target - ctx.channels.data()) + 1;
            const ThreadHandle worker = ctx.threads[join];
            CHK_RET(HcommThreadNotifyRecordOnThread(main, worker, 0));
            CHK_RET(HcommThreadNotifyWaitOnThread(worker, 0, CUSTOM_TIMEOUT));
            const uint64_t rounds = bytes / chunkSize + (bytes % chunkSize != 0);
            const uint64_t relayChunk = (relayBytes / rounds + (relayBytes % rounds != 0) + 127) / 128 * 128;
            const uint64_t forwardRounds = relayBytes / relayChunk + (relayBytes % relayChunk != 0);
            for (uint32_t slot = 0; slot < std::min<uint64_t>(forwardRounds, 2); ++slot)
                CHK_RET(HcommChannelNotifyRecordOnThread(worker, root->handle, 2 * (2 + slot)));
            for (uint64_t offset = 0, chunk = 0; offset < relayBytes; ++chunk) {
                const uint32_t slot = chunk % 2;
                CHK_RET(HcommChannelNotifyWaitOnThread(worker, root->handle, 2 * (2 + slot) + 1, CUSTOM_TIMEOUT));
                CHK_RET(HcommChannelNotifyRecordOnThread(worker, target->handle, 0));
                CHK_RET(HcommChannelNotifyWaitOnThread(worker, target->handle, 1, CUSTOM_TIMEOUT));
                offset += std::min(relayChunk, relayBytes - offset);
                if (chunk + 2 < forwardRounds)
                    CHK_RET(HcommChannelNotifyRecordOnThread(worker, root->handle, 2 * (2 + slot)));
            }
            CHK_RET(HcommThreadNotifyRecordOnThread(worker, main, join));
            for (uint32_t slot = 0; slot < std::min<uint64_t>(rounds, 2); ++slot)
                CHK_RET(HcommChannelNotifyRecordOnThread(main, root->handle, 2 * slot));
            for (uint64_t offset = 0, chunk = 0; offset < bytes; ++chunk) {
                const uint32_t slot = chunk % 2;
                const uint64_t len = std::min(chunkSize, bytes - offset);
                CHK_RET(HcommChannelNotifyWaitOnThread(main, root->handle, 2 * slot + 1, CUSTOM_TIMEOUT));
                CHK_RET(HcommLocalCopyOnThread(main, output + offset, stage + slot * slotSize, len));
                offset += len;
                if (chunk + 2 < rounds)
                    CHK_RET(HcommChannelNotifyRecordOnThread(main, root->handle, 2 * slot));
            }
        } else {
            if (param.myRank != targets.back()) {
                const auto index = std::find(targets.begin(), targets.end(), param.myRank) - targets.begin();
                const auto *source = FindChannel(ctx, relays[index]);
                CHK_PTR_NULL(source);
                join = static_cast<uint32_t>(source - ctx.channels.data()) + 1;
                const ThreadHandle worker = ctx.threads[join];
                const uint64_t remoteSlot = source->remoteCclMem.size / 4 / 128 * 128;
                const uint64_t sourceChunk = BalancedChunk(bytes, remoteSlot);
                if (sourceChunk == 0)
                    return HCCL_E_PARA;
                const uint64_t rounds = bytes / sourceChunk + (bytes % sourceChunk != 0);
                const uint64_t relayChunk = (relayBytes / rounds + (relayBytes % rounds != 0) + 127) / 128 * 128;
                CHK_RET(HcommThreadNotifyRecordOnThread(main, worker, 0));
                CHK_RET(HcommThreadNotifyWaitOnThread(worker, 0, CUSTOM_TIMEOUT));
                for (uint64_t offset = 0, chunk = 0; offset < relayBytes; ++chunk) {
                    const uint64_t len = std::min(relayChunk, relayBytes - offset);
                    CHK_RET(HcommChannelNotifyWaitOnThread(worker, source->handle, 0, CUSTOM_TIMEOUT));
                    CHK_RET(HcommReadOnThread(worker, source->handle, output + offset,
                        static_cast<const uint8_t *>(source->remoteCclMem.addr) + (2 + chunk % 2) * remoteSlot, len));
                    CHK_RET(HcommChannelNotifyRecordOnThread(worker, source->handle, 1));
                    offset += len;
                }
                CHK_RET(HcommThreadNotifyRecordOnThread(worker, main, join));
            }
            uint64_t offset = join == 0 ? 0 : relayBytes;
            const uint64_t count = (bytes - offset) / chunkSize + ((bytes - offset) % chunkSize != 0);
            for (uint32_t slot = 0; slot < std::min<uint64_t>(count, 2); ++slot) {
                CHK_RET(HcommChannelNotifyRecordOnThread(main, root->handle, 2 * slot));
            }
            for (uint64_t chunk = 0; offset < bytes; ++chunk) {
                const uint32_t slot = chunk % 2;
                const uint64_t len = std::min(chunkSize, bytes - offset);
                CHK_RET(HcommChannelNotifyWaitOnThread(main, root->handle, 2 * slot + 1, CUSTOM_TIMEOUT));
                CHK_RET(HcommLocalCopyOnThread(main, output + offset, stage + slot * slotSize, len));
                if (chunk + 2 < count)
                    CHK_RET(HcommChannelNotifyRecordOnThread(main, root->handle, 2 * slot));
                offset += len;
            }
        }
        if (join != 0)
            CHK_RET(HcommThreadNotifyWaitOnThread(main, join, CUSTOM_TIMEOUT));
        CHK_RET(HcommChannelNotifyRecordOnThread(main, root->handle, SCATTER_NOTIFY_CONSUMED));
        return HCCL_SUCCESS;
    }

    // Direct push fallback uses the receiver's capacity, with no root staging allocation.
    HcclResult ExecPush(const OpParam &param, const AlgResourceCtx &ctx, uint64_t bytes)
    {
        const bool sender = param.myRank == param.root;
        const ThreadHandle main = ctx.aicpuThread;
        std::array<uint32_t, 15> order{};
        const bool grouped = sender && param.rankSize == 16 && ctx.localRanks.size() == 8;
        if (grouped) {
            std::array<uint32_t, 7> local{};
            std::array<uint32_t, 8> remote{};
            uint32_t localCount = 0, remoteCount = 0;
            for (uint32_t i = 0; i < ctx.channels.size(); ++i) {
                if (std::binary_search(ctx.localRanks.begin(), ctx.localRanks.end(), ctx.channels[i].remoteRank)) {
                    if (localCount == local.size())
                        return HCCL_E_PARA;
                    local[localCount++] = i;
                } else {
                    if (remoteCount == remote.size())
                        return HCCL_E_PARA;
                    remote[remoteCount++] = i;
                }
            }
            if (localCount != local.size() || remoteCount != remote.size())
                return HCCL_E_PARA;
            // Short transfers favor finishing fan-out with a fast Clos transfer.
            // Larger transfers interleave mesh and Clos work to start both early.
            if (bytes <= 64 * 1024) {
                std::copy(local.begin(), local.end(), order.begin());
                std::copy(remote.begin(), remote.end(), order.begin() + local.size());
            } else {
                for (uint32_t i = 0; i < local.size(); ++i) {
                    order[2 * i] = local[i];
                    order[2 * i + 1] = remote[i];
                }
                order.back() = remote.back();
            }
        }
        for (uint32_t step = 0; step < ctx.channels.size(); ++step) {
            const uint32_t i = grouped ? order[step] : step;
            const auto &channel = ctx.channels[i];
            if (!sender && channel.remoteRank != param.root) {
                continue;
            }
            const ThreadHandle thread = sender ? ctx.threads[i + 1] : main;
            const uint64_t capacity = sender ? channel.remoteCclMem.size : ctx.localBuffer.size;
            const uint64_t chunkSize = BalancedChunk(bytes, capacity);
            if (chunkSize == 0) {
                return HCCL_E_PARA;
            }
            const uint64_t chunks = bytes / chunkSize + (bytes % chunkSize != 0);
            const uint32_t depth = chunks > 1 && capacity / chunkSize >= 2 ? 2 : 1;
            if (sender) {
                CHK_RET(HcommThreadNotifyRecordOnThread(main, thread, 0));
                CHK_RET(HcommThreadNotifyWaitOnThread(thread, 0, CUSTOM_TIMEOUT));
            } else {
                for (uint32_t slot = 0; slot < depth; ++slot) {
                    CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, 2 * slot));
                }
            }
            uint64_t chunk = 0;
            for (uint64_t offset = 0; offset < bytes; ++chunk) {
                const uint32_t slot = chunk % depth;
                const uint64_t length = std::min(chunkSize, bytes - offset);
                if (sender) {
                    const auto *input
                        = static_cast<const uint8_t *>(param.inputPtr) + channel.remoteRank * bytes + offset;
                    auto *remote = static_cast<uint8_t *>(channel.remoteCclMem.addr) + slot * chunkSize;
                    CHK_RET(WriteReady(thread, channel, remote, input, length, slot));
                } else {
                    auto *output = static_cast<uint8_t *>(param.outputPtr) + offset;
                    const auto *local = static_cast<const uint8_t *>(ctx.localBuffer.addr) + slot * chunkSize;
                    CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, 2 * slot + 1, CUSTOM_TIMEOUT));
                    CHK_RET(HcommLocalCopyOnThread(thread, output, local, length));
                    if (chunk + depth < chunks) {
                        CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, 2 * slot));
                    }
                }
                offset += length;
            }
            if (!sender) {
                CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel.handle, SCATTER_NOTIFY_CONSUMED));
            }
        }
        if (sender) {
            const auto *own = static_cast<const uint8_t *>(param.inputPtr) + param.root * bytes;
            if (own != param.outputPtr) {
                CHK_RET(HcommLocalCopyOnThread(main, param.outputPtr, own, bytes));
            }
            // Receiver completion also proves the worker's last DATA was published.
            for (uint32_t step = 0; step < ctx.channels.size(); ++step) {
                const uint32_t i = grouped ? order[step] : step;
                CHK_RET(HcommChannelNotifyWaitOnThread(
                    main, ctx.channels[i].handle, SCATTER_NOTIFY_CONSUMED, CUSTOM_TIMEOUT));
            }
        }
        return HCCL_SUCCESS;
    }
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    const auto type = SIZE_TABLE.find(param.dataType);
    if (type == SIZE_TABLE.end() || param.rankSize == 0 || param.root >= param.rankSize
        || param.myRank >= param.rankSize || resCtx.threads.size() != param.rankSize
        || param.count > std::numeric_limits<uint64_t>::max() / type->second / param.rankSize) {
        return HCCL_E_PARA;
    }
    const uint64_t bytes = param.count * type->second;
    if (bytes == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(param.outputPtr);
    const ThreadHandle mainThread = resCtx.aicpuThread;
    if (param.myRank == param.root) {
        CHK_PTR_NULL(param.inputPtr);
    }
    if (param.rankSize == 1) {
        if (param.inputPtr != param.outputPtr) {
            CHK_RET(HcommLocalCopyOnThread(mainThread, param.outputPtr, param.inputPtr, bytes));
        }
        return HCCL_SUCCESS;
    }
    if (resCtx.channels.size() != param.rankSize - 1 || resCtx.localBuffer.addr == nullptr) {
        return HCCL_E_PARA;
    }
    if (bytes >= SCATTER_RELAY_THRESHOLD) {
        if (param.rankSize == 16 && resCtx.localRanks.size() == 8) {
            return ExecRelay(param, resCtx, bytes);
        }
        return ExecPush(param, resCtx, bytes);
    }
    const uint64_t inputBytes = bytes * param.rankSize;
    if (param.rankSize == 16 && inputBytes <= SCATTER_PACK_LIMIT) {
        uint64_t capacity = resCtx.localBuffer.size;
        for (const auto &channel : resCtx.channels)
            capacity = std::min(capacity, channel.remoteCclMem.size);
        // Every rank sees all CCL capacities, so routing agrees even if buffers differ.
        if (inputBytes <= capacity)
            return bytes <= 64 * 1024 ? ExecTree(param, resCtx, bytes) : ExecFanout(param, resCtx, bytes);
    }
    const bool sender = param.myRank == param.root;
    if (sender && inputBytes <= SCATTER_PACK_LIMIT && inputBytes <= resCtx.localBuffer.size) {
        // Packed data remains immutable until every remote read completes. Receivers own the transfer
        // threads, so root needs no per-peer workers: publish all readiness before waiting for any ACK.
        const auto &group = resCtx.localRanks;
        const bool grouped = param.rankSize == 16 && group.size() == 8;
        const bool split = grouped && inputBytes >= 1024 * 1024 && group.back() - group.front() == 7
                           && (group.front() == 0 || group.front() == 8);
        // The root Clos port is shared by eight readers. Release them first. With contiguous groups,
        // prepare their immutable region first so remote reads overlap preparation of the local group.
        for (uint32_t phase = 0; phase < (grouped ? 2U : 1U); ++phase) {
            if (split || phase == 0) {
                uint64_t first = 0;
                uint64_t last = param.rankSize;
                if (split) {
                    first = phase == 0 ? (group.front() == 0 ? 8 : 0) : group.front();
                    last = first + 8;
                }
                // Exclude root's own slice only when it lies at the edge of this contiguous range.
                if (first == param.root) {
                    ++first;
                }
                if (last == param.root + 1) {
                    --last;
                }
                CHK_RET(
                    HcommLocalCopyOnThread(mainThread, static_cast<uint8_t *>(resCtx.localBuffer.addr) + first * bytes,
                        static_cast<const uint8_t *>(param.inputPtr) + first * bytes, (last - first) * bytes));
            }
            for (const auto &channel : resCtx.channels) {
                const bool local = grouped && std::binary_search(group.begin(), group.end(), channel.remoteRank);
                if (!grouped || local == (phase == 1)) {
                    CHK_RET(HcommChannelNotifyRecordOnThread(mainThread, channel.handle, 0));
                }
            }
        }
        const auto *own = static_cast<const uint8_t *>(param.inputPtr) + param.root * bytes;
        if (own != param.outputPtr) {
            CHK_RET(HcommLocalCopyOnThread(mainThread, param.outputPtr, own, bytes));
        }
        for (const auto &channel : resCtx.channels) {
            CHK_RET(HcommChannelNotifyWaitOnThread(mainThread, channel.handle, 1, CUSTOM_TIMEOUT));
        }
        return HCCL_SUCCESS;
    }
    for (uint32_t i = 0; i < resCtx.channels.size(); ++i) {
        const auto &channel = resCtx.channels[i];
        if (param.myRank != param.root && channel.remoteRank != param.root) {
            continue;
        }
        // A receiver has no independent peer work to fan out: the main thread already owns the launch dependency.
        const ThreadHandle worker = sender ? resCtx.threads[i + 1] : mainThread;
        const uint64_t rootBufferSize = sender ? resCtx.localBuffer.size : channel.remoteCclMem.size;
        const uint64_t slotSize = ScatterSlotSize(rootBufferSize, param.rankSize);
        const bool packed = inputBytes <= SCATTER_PACK_LIMIT && inputBytes <= rootBufferSize;
        const uint64_t maxChunk = std::min(SCATTER_MAX_SLICE, packed ? bytes : slotSize);
        if (maxChunk == 0 || channel.remoteCclMem.addr == nullptr) {
            return HCCL_E_PARA;
        }
        // Use the minimum number of rounds allowed by capacity, then balance their sizes.
        // This avoids a fixed 16MiB split and avoids a large first round followed by a tiny tail.
        const uint64_t rounds = bytes / maxChunk + (bytes % maxChunk != 0);
        const uint64_t balanced = bytes / rounds + (bytes % rounds != 0);
        const uint64_t aligned = (balanced + SCATTER_SLICE_ALIGN - 1) / SCATTER_SLICE_ALIGN * SCATTER_SLICE_ALIGN;
        const uint64_t chunkSize = std::min(maxChunk, aligned);
        // Do not enqueue transfers ahead of the launcher's host-stream dependency.
        if (sender) {
            CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, worker, 0));
            CHK_RET(HcommThreadNotifyWaitOnThread(worker, 0, CUSTOM_TIMEOUT));
        }
        // A remote read lands directly in the user's output. Only root CCL needs staging.
        // Receiver's acknowledgement protects that staging slot until the read completes.
        for (uint64_t offset = 0; offset < bytes;) {
            const uint64_t length = std::min(chunkSize, bytes - offset);
            if (sender) {
                auto *stage = static_cast<uint8_t *>(resCtx.localBuffer.addr)
                              + (packed ? channel.remoteRank * bytes + offset : i * slotSize);
                if (!packed) {
                    const auto *input
                        = static_cast<const uint8_t *>(param.inputPtr) + channel.remoteRank * bytes + offset;
                    CHK_RET(HcommLocalCopyOnThread(worker, stage, input, length));
                }
                CHK_RET(HcommChannelNotifyRecordOnThread(worker, channel.handle, 0));
                CHK_RET(HcommChannelNotifyWaitOnThread(worker, channel.handle, 1, CUSTOM_TIMEOUT));
            } else {
                // Root lists peers in ascending order with itself removed.
                const uint32_t rootSlot = param.myRank < param.root ? param.myRank : param.myRank - 1;
                const auto *remote = static_cast<const uint8_t *>(channel.remoteCclMem.addr)
                                     + (packed ? param.myRank * bytes + offset : rootSlot * slotSize);
                auto *output = static_cast<uint8_t *>(param.outputPtr) + offset;
                CHK_RET(HcommChannelNotifyWaitOnThread(worker, channel.handle, 0, CUSTOM_TIMEOUT));
                CHK_RET(HcommReadOnThread(worker, channel.handle, output, remote, length));
                CHK_RET(HcommChannelNotifyRecordOnThread(worker, channel.handle, 1));
            }
            offset += length;
        }
        if (sender) {
            CHK_RET(HcommThreadNotifyRecordOnThread(worker, mainThread, i + 1));
        }
    }
    // Release all network workers first; root's own copy can run concurrently with peer transfers.
    if (param.myRank == param.root) {
        auto *ownSlice = static_cast<uint8_t *>(param.inputPtr) + param.root * bytes;
        if (ownSlice != param.outputPtr) {
            CHK_RET(HcommLocalCopyOnThread(mainThread, param.outputPtr, ownSlice, bytes));
        }
    }
    // Join only after releasing every worker, keeping independent peer transfers concurrent.
    for (uint32_t i = 0; i < resCtx.channels.size(); ++i) {
        if (sender) {
            CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, i + 1, CUSTOM_TIMEOUT));
        }
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
