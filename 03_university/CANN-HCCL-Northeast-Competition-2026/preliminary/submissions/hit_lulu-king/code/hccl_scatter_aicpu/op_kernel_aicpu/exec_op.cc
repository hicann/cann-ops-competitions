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


namespace {
const ChannelInfo *FindPeer(const AlgResourceCtx &ctx, uint32_t peer)
{
    for (const auto &ch : ctx.channels) if (ch.remoteRank == peer) return &ch;
    return nullptr;
}

HcclResult ReadRange(ThreadHandle thread, const ChannelInfo &ch, void *dst, const void *src, uint64_t bytes)
{
    constexpr uint64_t LIMIT = 256ULL * 1024 * 1024;
    for (uint64_t offset = 0; offset < bytes;) {
        const uint64_t length = std::min(LIMIT, bytes - offset);
        CHK_RET(static_cast<HcclResult>(HcommReadOnThread(thread, ch.handle,
            static_cast<uint8_t *>(dst) + offset, static_cast<const uint8_t *>(src) + offset, length)));
        offset += length;
    }
    return HCCL_SUCCESS;
}

HcclResult RelayScatter(const OpParam &param, const AlgResourceCtx &ctx, uint64_t bytes, uint64_t elementBytes)
{
    CHK_PRT_RET(ctx.helperForRank.size() != param.rankSize || ctx.threads.size() != 2,
        HCCL_ERROR("Relay plan/resources mismatch"), HCCL_E_INTERNAL);
    const ThreadHandle main = ctx.threads[0], worker = ctx.threads[1];
    // Remote channel slots: root DONE=0/READY=1; relay DATA=0/READY-or-ACK=1.
    // Local thread slots: start=0, ping/pong=1/2, worker completion=3.
    const uint64_t tail = (param.count / 4) * elementBytes;
    const uint64_t direct = bytes - tail;
    if (param.myRank == param.root) {
        CHK_PRT_RET(ctx.channels.size() != param.rankSize - 1, HCCL_ERROR("Missing root peers"), HCCL_E_INTERNAL);
        // Main thread's entry wait is ordered after root's stream producer.
        for (const auto &ch : ctx.channels)
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(main, ch.handle, 1)));
        const uint8_t *self = static_cast<const uint8_t *>(param.inputPtr) + param.root * bytes;
        const uintptr_t input = reinterpret_cast<uintptr_t>(param.inputPtr);
        const uintptr_t output = reinterpret_cast<uintptr_t>(param.outputPtr);
        const bool overlap = output >= input ? output - input < bytes * param.rankSize : input - output < bytes;
        if (!overlap)
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(main, param.outputPtr, self, bytes)));
        for (const auto &ch : ctx.channels)
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(main, ch.handle, 0, CUSTOM_TIMEOUT)));
        if (overlap && param.outputPtr != self)
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(main, param.outputPtr, self, bytes)));
        return HCCL_SUCCESS;
    }

    const ChannelInfo *root = FindPeer(ctx, param.root);
    CHK_PTR_NULL(root);
    CHK_PRT_RET(root->remoteOutput.addr == nullptr || root->remoteOutput.size < bytes * param.rankSize,
        HCCL_ERROR("Missing root input"), HCCL_E_PARA);
    CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(main, root->handle, 1, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(main, worker, 0)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(worker, 0, CUSTOM_TIMEOUT)));
    const uint8_t *input = static_cast<const uint8_t *>(root->remoteOutput.addr);
    const uint32_t helper = ctx.helperForRank[param.myRank];

    if (helper != INVALID_VALUE_RANKID) {
        const ChannelInfo *relay = FindPeer(ctx, helper);
        CHK_PTR_NULL(relay);
        // Receiver stream is ready before permitting any relay writes into output.
        CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(worker, relay->handle, 1)));
        for (uint64_t offset = 0; offset < tail; offset += std::min(RELAY_CHUNK_BYTES, tail - offset)) {
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(worker, relay->handle, 0, CUSTOM_TIMEOUT)));
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(worker, relay->handle, 1)));
        }
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(worker, main, 3)));
        // Separate root channel is owned only by main; relay channel only by worker.
        CHK_RET(ReadRange(main, *root, param.outputPtr, input + param.myRank * bytes, direct));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(main, 3, CUSTOM_TIMEOUT)));
    } else {
        CHK_PRT_RET(ctx.localBuffer.addr == nullptr || ctx.localBuffer.size < 2 * RELAY_CHUNK_BYTES,
            HCCL_ERROR("Insufficient double buffer"), HCCL_E_PARA);
        uint64_t sequence = 0;
        for (uint32_t dstRank = 0; dstRank < param.rankSize; ++dstRank) {
            if (ctx.helperForRank[dstRank] != param.myRank) continue;
            const ChannelInfo *dst = FindPeer(ctx, dstRank);
            CHK_PTR_NULL(dst);
            CHK_PRT_RET(dst->remoteOutput.addr == nullptr || dst->remoteOutput.size < bytes,
                HCCL_ERROR("Missing relay destination"), HCCL_E_PARA);
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(worker, dst->handle, 1, CUSTOM_TIMEOUT)));
            for (uint64_t offset = 0; offset < tail;) {
                const uint64_t length = std::min(RELAY_CHUNK_BYTES, tail - offset);
                const uint32_t slot = static_cast<uint32_t>(sequence % 2);
                void *scratch = static_cast<uint8_t *>(ctx.localBuffer.addr) + slot * RELAY_CHUNK_BYTES;
                if (sequence >= 2)
                    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(main, slot + 1, CUSTOM_TIMEOUT)));
                CHK_RET(ReadRange(main, *root, scratch, input + dstRank * bytes + direct + offset, length));
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(main, worker, slot + 1)));
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(worker, slot + 1, CUSTOM_TIMEOUT)));
                CHK_RET(static_cast<HcclResult>(HcommWriteWithNotifyOnThread(worker, dst->handle,
                    static_cast<uint8_t *>(dst->remoteOutput.addr) + direct + offset, scratch, length, 0)));
                CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(worker, dst->handle, 1, CUSTOM_TIMEOUT)));
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(worker, main, slot + 1)));
                offset += length;
                ++sequence;
            }
        }
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(worker, main, 3)));
        // Own output read overlaps the tail of forwarding instead of delaying its start.
        CHK_RET(ReadRange(main, *root, param.outputPtr, input + param.myRank * bytes, bytes));
        for (uint64_t pending = sequence > 2 ? sequence - 2 : 0; pending < sequence; ++pending)
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(main,
                static_cast<uint32_t>(pending % 2) + 1, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(main, 3, CUSTOM_TIMEOUT)));
    }
    // Helper DONE covers its own output and all relayed chunks; receiver DONE covers both paths.
    return static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(main, root->handle, 0));
}
} // namespace

namespace ops_hccl {
HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    CHK_PRT_RET(param.rankSize == 0 || param.myRank >= param.rankSize || param.root >= param.rankSize,
        HCCL_ERROR("Invalid pull rank"), HCCL_E_PARA);
    if (param.count == 0) return HCCL_SUCCESS;
    CHK_PTR_NULL(param.outputPtr);
    CHK_PRT_RET(resCtx.threads.empty(), HCCL_ERROR("Missing thread"), HCCL_E_INTERNAL);
    auto type = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(type == SIZE_TABLE.end() || type->second == 0,
        HCCL_ERROR("Invalid type"), HCCL_E_PARA);
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / type->second,
        HCCL_ERROR("Size overflow"), HCCL_E_PARA);
    uint64_t bytes = param.count * type->second;
    CHK_PRT_RET(bytes > std::numeric_limits<uint64_t>::max() / param.rankSize,
        HCCL_ERROR("Input overflow"), HCCL_E_PARA);
    uint64_t total = bytes * param.rankSize;
    ThreadHandle thread = resCtx.threads[0];
    if (param.myRank == param.root) CHK_PTR_NULL(param.inputPtr);
    if (param.rankSize == 1) {
        return static_cast<HcclResult>(HcommLocalCopyOnThread(thread, param.outputPtr, param.inputPtr, bytes));
    }
    // v2 field names retained: root registers INPUT, each other rank registers OUTPUT.
    void *local = param.myRank == param.root ? param.inputPtr : param.outputPtr;
    uint64_t required = param.myRank == param.root ? total : bytes;
    CHK_PRT_RET(resCtx.registeredOutput.addr != local || resCtx.registeredOutput.size < required,
        HCCL_ERROR("Cached registration mismatch; use a fresh communicator"), HCCL_E_PARA);
    if (!resCtx.helperForRank.empty()) return RelayScatter(param, resCtx, bytes, type->second);
    constexpr uint32_t DONE = 0;
    if (param.myRank == param.root) {
        CHK_PRT_RET(resCtx.channels.size() != param.rankSize - 1,
            HCCL_ERROR("Missing receivers"), HCCL_E_INTERNAL);
        const uint8_t *self = static_cast<const uint8_t *>(param.inputPtr) + param.root * bytes;
        const uintptr_t inputAddr = reinterpret_cast<uintptr_t>(param.inputPtr);
        const uintptr_t outputAddr = reinterpret_cast<uintptr_t>(param.outputPtr);
        // Subtraction-based interval test avoids overflowing address + length.
        const bool overlap = outputAddr >= inputAddr ? outputAddr - inputAddr < total
                                                    : inputAddr - outputAddr < bytes;
        const bool deferSelfCopy = overlap && param.outputPtr != self;
        if (!deferSelfCopy && param.outputPtr != self) {
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, param.outputPtr, self, bytes)));
        }
        // Remote memory and channels are ready before this kernel is launched,
        // so receivers can start the one-sided reads immediately. DONE keeps the
        // root's input alive until every receiver has completed its read.
        for (const auto &ch : resCtx.channels) {
            CHK_RET(static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread, ch.handle, DONE, CUSTOM_TIMEOUT)));
        }
        // If recvBuf overlaps another part of sendBuf, defer the local copy until
        // all remote readers finish so it cannot overwrite their source data.
        if (deferSelfCopy) {
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, param.outputPtr, self, bytes)));
        }
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(resCtx.channels.size() != 1 || resCtx.channels[0].remoteRank != param.root,
        HCCL_ERROR("Missing root"), HCCL_E_INTERNAL);
    const auto &ch = resCtx.channels[0];
    CHK_PRT_RET(ch.remoteOutput.addr == nullptr || ch.remoteOutput.size < total,
        HCCL_ERROR("Root input not registered"), HCCL_E_PARA);
    const uint8_t *src = static_cast<const uint8_t *>(ch.remoteOutput.addr) + param.myRank * bytes;
    uint8_t *dst = static_cast<uint8_t *>(param.outputPtr);
    constexpr uint64_t CHUNK = 256ULL * 1024 * 1024;
    for (uint64_t offset = 0; offset < bytes;) {
        uint64_t length = std::min(CHUNK, bytes - offset);
        CHK_RET(static_cast<HcclResult>(HcommReadOnThread(thread, ch.handle, dst + offset, src + offset, length)));
        offset += length;
    }
    // Same-thread Read -> completion notify follows the official AICPU P2P example.
    return static_cast<HcclResult>(HcommChannelNotifyRecordOnThread(thread, ch.handle, DONE));
}
} // namespace ops_hccl
