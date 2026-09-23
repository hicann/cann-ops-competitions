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
    constexpr uint32_t FORWARD_READY_NOTIFY = 1;
    constexpr uint32_t LOCAL_B_READY_NOTIFY = 0;
    constexpr uint32_t ALL_CHANNELS = (1U << SCATTER_CHANNELS) - 1;
    constexpr uint64_t SMALL_MESSAGE_BYTES = 1024ULL * 1024;
    constexpr uint32_t SMALL_SEND_GROUPS = SCATTER_WIDTH / 2;

    void *At(void *ptr, uint64_t offset)
    {
        return static_cast<uint8_t *>(ptr) + offset;
    }

    uint32_t ChannelFor(const AlgResourceCtx &ctx, uint32_t peer)
    {
        return peer < ctx.myRank ? peer : peer - 1;
    }

    uint32_t Column(const std::array<uint32_t, SCATTER_WIDTH> &ranks, uint32_t rank)
    {
        return static_cast<uint32_t>(std::find(ranks.begin(), ranks.end(), rank) - ranks.begin());
    }

    const void *Input(const OpParam &param, uint32_t rank, uint64_t offset)
    {
        return At(param.inputPtr, rank * (param.count * sizeof(float)) + offset);
    }

    HcclResult StartWorkers(const AlgResourceCtx &ctx, uint32_t mask)
    {
        for (uint32_t ch = 0; ch < SCATTER_CHANNELS; ++ch) {
            const ThreadHandle thread = ctx.threads[ch + 1];
            if ((mask & (1U << ch)) == 0 || thread == ctx.aicpuThread) {
                continue;
            }
            CHK_RET(HcommThreadNotifyRecordOnThread(ctx.aicpuThread, thread, 0));
            CHK_RET(HcommThreadNotifyWaitOnThread(thread, 0, CUSTOM_TIMEOUT));
        }
        return HCCL_SUCCESS;
    }

    HcclResult JoinWorkers(const AlgResourceCtx &ctx, uint32_t mask)
    {
        for (uint32_t ch = 0; ch < SCATTER_CHANNELS; ++ch) {
            const ThreadHandle thread = ctx.threads[ch + 1];
            if ((mask & (1U << ch)) == 0 || thread == ctx.aicpuThread) {
                continue;
            }
            CHK_RET(HcommThreadNotifyRecordOnThread(thread, ctx.aicpuThread, ch + 1));
            CHK_RET(HcommThreadNotifyWaitOnThread(ctx.aicpuThread, ch + 1, CUSTOM_TIMEOUT));
        }
        return HCCL_SUCCESS;
    }

    HcclResult Receive(const AlgResourceCtx &ctx, uint32_t ch)
    {
        CHK_RET(HcommChannelNotifyRecordOnThread(ctx.threads[ch + 1], ctx.channels[ch].handle, NOTIFY_IDX_ACK));
        CHK_RET(HcommChannelNotifyWaitOnThread(
            ctx.threads[ch + 1], ctx.channels[ch].handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT));
        return HCCL_SUCCESS;
    }

    HcclResult BeginSend(const AlgResourceCtx &ctx, uint32_t ch)
    {
        return static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(
            ctx.threads[ch + 1], ctx.channels[ch].handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT));
    }

    HcclResult EndSend(const AlgResourceCtx &ctx, uint32_t ch)
    {
        return static_cast<HcclResult>(
            HcommChannelNotifyRecordOnThread(ctx.threads[ch + 1], ctx.channels[ch].handle, NOTIFY_IDX_DATA_SIGNAL));
    }

    HcclResult Write(const AlgResourceCtx &ctx, uint32_t ch, const void *src, uint64_t offset, uint64_t bytes)
    {
        const auto &peer = ctx.channels[ch];
        if (offset > peer.remoteCclMem.size || bytes > peer.remoteCclMem.size - offset
            || bytes > SCATTER_MAX_TRANSFER) {
            return HCCL_E_PARA;
        }
        if (bytes != 0) {
            CHK_RET(
                HcommWriteOnThread(ctx.threads[ch + 1], peer.handle, At(peer.remoteCclMem.addr, offset), src, bytes));
        }
        return HCCL_SUCCESS;
    }

    HcclResult CopyResult(const OpParam &param, const AlgResourceCtx &ctx, const ScatterSlice &s)
    {
        if (param.myRank == param.root) {
            const void *src = Input(param, param.root, s.offset);
            void *dst = At(param.outputPtr, s.offset);
            if (src != dst) {
                CHK_RET(HcommLocalCopyOnThread(ctx.aicpuThread, dst, src, s.aBytes + s.bBytes));
            }
        } else {
            CHK_RET(
                HcommLocalCopyOnThread(ctx.aicpuThread, At(param.outputPtr, s.offset), ctx.localBuffer.addr, s.aBytes));
            if (s.bBytes != 0) {
                CHK_RET(HcommLocalCopyOnThread(ctx.aicpuThread, At(param.outputPtr, s.offset + s.aBytes),
                    At(ctx.localBuffer.addr, s.BOffset()), s.bBytes));
            }
        }
        return HCCL_SUCCESS;
    }

    struct DirectPlan {
        uint32_t workerMask = 0;
        ThreadHandle copyThread = 0;
    };

    DirectPlan AssignDirectSenders(const AlgResourceCtx &ctx, AlgResourceCtx &directCtx)
    {
        DirectPlan plan{};
        std::array<ThreadHandle, SMALL_SEND_GROUPS> localThreads{};
        std::array<ThreadHandle, SMALL_SEND_GROUPS> crossThreads{};
        uint32_t localIndex = 0;
        for (uint32_t peer : ctx.localRanks) {
            if (peer == ctx.myRank) {
                continue;
            }
            const uint32_t ch = ChannelFor(ctx, peer);
            const uint32_t group = localIndex % SMALL_SEND_GROUPS;
            if (localIndex < SMALL_SEND_GROUPS) {
                localThreads[group] = ctx.threads[ch + 1];
                plan.workerMask |= 1U << ch;
            }
            directCtx.threads[ch + 1] = localThreads[group];
            ++localIndex;
        }
        for (uint32_t col = 0; col < SCATTER_WIDTH; ++col) {
            const uint32_t ch = ChannelFor(ctx, ctx.remoteRanks[col]);
            const uint32_t group = col % SMALL_SEND_GROUPS;
            if (col < SMALL_SEND_GROUPS) {
                crossThreads[group] = col == 0 ? ctx.aicpuThread : ctx.threads[ch + 1];
                plan.workerMask |= 1U << ch;
            }
            directCtx.threads[ch + 1] = crossThreads[group];
        }
        plan.copyThread = localThreads[SMALL_SEND_GROUPS - 1];
        return plan;
    }

    bool CanOverlapRootCopy(const OpParam &param)
    {
        if (param.outputPtr == Input(param, param.root, 0)) {
            return true;
        }
        const uintptr_t input = reinterpret_cast<uintptr_t>(param.inputPtr);
        const uintptr_t output = reinterpret_cast<uintptr_t>(param.outputPtr);
        const uint64_t bytes = param.count * sizeof(float);
        return output < input ? input - output >= bytes : output - input >= bytes * param.rankSize;
    }

    HcclResult DirectScatter(const OpParam &param, const AlgResourceCtx &ctx)
    {
        AlgResourceCtx directCtx = ctx;
        const bool isRoot = param.myRank == param.root;
        DirectPlan plan{};
        if (isRoot) {
            plan = AssignDirectSenders(ctx, directCtx);
            CHK_RET(StartWorkers(directCtx, plan.workerMask));
        } else {
            directCtx.threads.fill(ctx.aicpuThread);
        }
        ScatterSlice s{};
        s.aBytes = param.count * sizeof(float);
        if (isRoot) {
            for (uint32_t ch = 0; ch < SCATTER_CHANNELS; ++ch) {
                CHK_RET(BeginSend(directCtx, ch));
                CHK_RET(Write(directCtx, ch, Input(param, ctx.channels[ch].remoteRank, 0), 0, s.aBytes));
                CHK_RET(EndSend(directCtx, ch));
            }
            const bool overlapCopy = CanOverlapRootCopy(param);
            if (overlapCopy) {
                AlgResourceCtx copyCtx = directCtx;
                copyCtx.aicpuThread = plan.copyThread;
                CHK_RET(CopyResult(param, copyCtx, s));
            }
            CHK_RET(JoinWorkers(directCtx, plan.workerMask));
            if (overlapCopy) {
                return HCCL_SUCCESS;
            }
        } else {
            CHK_RET(Receive(directCtx, ChannelFor(ctx, param.root)));
        }
        return CopyResult(param, directCtx, s);
    }

    uint32_t ActiveChannels(const OpParam &param, const AlgResourceCtx &ctx, bool hasB)
    {
        if (param.myRank == param.root) {
            if (hasB) {
                return ALL_CHANNELS;
            }
            uint32_t mask = 1U << ChannelFor(ctx, ctx.remoteRanks[ctx.localIndex]);
            for (uint32_t peer : ctx.localRanks) {
                if (peer != param.root) {
                    mask |= 1U << ChannelFor(ctx, peer);
                }
            }
            return mask;
        }
        const uint32_t paired = ctx.remoteRanks[ctx.localIndex];
        uint32_t mask = 1U << ChannelFor(ctx, paired);
        if (hasB || Column(ctx.localRanks, param.root) < SCATTER_WIDTH) {
            mask |= 1U << ChannelFor(ctx, param.root);
        }
        return mask;
    }

    HcclResult SendFromRoot(const OpParam &param, const AlgResourceCtx &ctx, const ScatterSlice &s, uint32_t mask)
    {
        for (uint32_t ch = 0; ch < SCATTER_CHANNELS; ++ch) {
            if ((mask & (1U << ch)) == 0) {
                continue;
            }
            const uint32_t peer = ctx.channels[ch].remoteRank;
            const uint32_t col = Column(ctx.localRanks, peer);
            CHK_RET(BeginSend(ctx, ch));
            if (col < SCATTER_WIDTH) {
                CHK_RET(Write(ctx, ch, Input(param, peer, s.offset), 0, s.aBytes));
                CHK_RET(Write(ctx, ch, Input(param, ctx.remoteRanks[col], s.offset), s.aStride, s.aBytes));
                // Release A forwarding before sending B on this local link.
                CHK_RET(EndSend(ctx, ch));
                if (s.bBytes == 0) {
                    continue;
                }
            } else if (peer == ctx.remoteRanks[ctx.localIndex]) {
                CHK_RET(Write(ctx, ch, Input(param, peer, s.offset), 0, s.aBytes));
            }
            CHK_RET(Write(ctx, ch, Input(param, peer, s.offset + s.aBytes), s.BOffset(), s.bBytes));
            if (col < SCATTER_WIDTH) {
                CHK_RET(HcommChannelNotifyRecordOnThread(
                    ctx.threads[ch + 1], ctx.channels[ch].handle, LOCAL_B_READY_NOTIFY));
            } else {
                CHK_RET(EndSend(ctx, ch));
            }
        }
        return HCCL_SUCCESS;
    }

    HcclResult RelayFromSource(const OpParam &param, const AlgResourceCtx &ctx, const ScatterSlice &s)
    {
        const uint32_t rootCh = ChannelFor(ctx, param.root);
        const uint32_t crossCh = ChannelFor(ctx, ctx.remoteRanks[ctx.localIndex]);
        CHK_RET(Receive(ctx, rootCh));
        CHK_RET(
            HcommThreadNotifyRecordOnThread(ctx.threads[rootCh + 1], ctx.threads[crossCh + 1], FORWARD_READY_NOTIFY));
        if (s.bBytes != 0) {
            CHK_RET(HcommChannelNotifyWaitOnThread(
                ctx.threads[rootCh + 1], ctx.channels[rootCh].handle, LOCAL_B_READY_NOTIFY, CUSTOM_TIMEOUT));
        }
        CHK_RET(HcommThreadNotifyWaitOnThread(ctx.threads[crossCh + 1], FORWARD_READY_NOTIFY, CUSTOM_TIMEOUT));
        CHK_RET(BeginSend(ctx, crossCh));
        CHK_RET(Write(ctx, crossCh, At(ctx.localBuffer.addr, s.aStride), 0, s.aBytes));
        return EndSend(ctx, crossCh);
    }

    HcclResult BalancedScatter(const OpParam &param, const AlgResourceCtx &ctx)
    {
        const uint64_t maxCount = ScatterChunkCount(ctx.localBuffer.size);
        if (maxCount == 0) {
            return HCCL_E_MEMORY;
        }
        AlgResourceCtx runCtx = ctx;
        const bool sourceServer = Column(ctx.localRanks, param.root) < SCATTER_WIDTH;
        const uint32_t paired = ctx.remoteRanks[ctx.localIndex];
        if (param.myRank == param.root) {
            runCtx.threads[ChannelFor(ctx, paired) + 1] = ctx.aicpuThread;
        } else if (paired == param.root) {
            runCtx.threads.fill(ctx.aicpuThread);
        }
        uint64_t processed = 0;
        while (processed < param.count) {
            const uint64_t count = std::min(maxCount, param.count - processed);
            ScatterSlice s{};
            s.offset = processed * sizeof(float);
            s.aBytes = ScatterRelayCount(count) * sizeof(float);
            s.bBytes = count * sizeof(float) - s.aBytes;
            s.aStride = ScatterAlignUp(s.aBytes);
            s.bStride = ScatterAlignUp(s.bBytes);
            if (s.BOffset() + s.bStride > ctx.localBuffer.size) {
                return HCCL_E_INTERNAL;
            }
            const uint32_t mask = ActiveChannels(param, runCtx, s.bBytes != 0);
            CHK_RET(StartWorkers(runCtx, mask));
            if (param.myRank == param.root) {
                CHK_RET(SendFromRoot(param, runCtx, s, mask));
            } else if (sourceServer) {
                CHK_RET(RelayFromSource(param, runCtx, s));
            } else {
                CHK_RET(Receive(runCtx, ChannelFor(ctx, paired)));
                if (paired != param.root && s.bBytes != 0) {
                    CHK_RET(Receive(runCtx, ChannelFor(ctx, param.root)));
                }
            }
            CHK_RET(JoinWorkers(runCtx, mask));
            CHK_RET(CopyResult(param, runCtx, s));
            processed += count;
        }
        return HCCL_SUCCESS;
    }
} // namespace

HcclResult ValidateScatter(const OpParam &param, const AlgResourceCtx &ctx)
{
    if (ctx.magic != SCATTER_CONTEXT_MAGIC || ctx.rankSize != param.rankSize || ctx.myRank != param.myRank
        || param.myRank >= param.rankSize || param.root >= param.rankSize || param.dataType != HCCL_DATA_TYPE_FP32
        || ctx.aicpuThread != ctx.threads[0]) {
        return HCCL_E_PARA;
    }
    if (param.count > std::numeric_limits<uint64_t>::max() / sizeof(float) / param.rankSize) {
        return HCCL_E_PARA;
    }
    if (param.count != 0) {
        CHK_PTR_NULL(param.outputPtr);
        if (param.myRank == param.root) {
            CHK_PTR_NULL(param.inputPtr);
        }
    }
    if (param.rankSize == 1) {
        return HCCL_SUCCESS;
    }
    if (param.rankSize != SCATTER_RANKS || ctx.localIndex >= SCATTER_WIDTH
        || ctx.localRanks[ctx.localIndex] != param.myRank || ctx.localBuffer.size < sizeof(float)) {
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_PTR_NULL(ctx.localBuffer.addr);
    std::array<bool, SCATTER_RANKS> present{};
    for (uint32_t col = 0; col < SCATTER_WIDTH; ++col) {
        for (uint32_t rank : {ctx.localRanks[col], ctx.remoteRanks[col]}) {
            if (rank >= SCATTER_RANKS || present[rank]) {
                return HCCL_E_PARA;
            }
            present[rank] = true;
        }
    }
    if (!std::is_sorted(ctx.localRanks.begin(), ctx.localRanks.end())
        || !std::is_sorted(ctx.remoteRanks.begin(), ctx.remoteRanks.end())) {
        return HCCL_E_PARA;
    }
    uint32_t ch = 0;
    for (uint32_t peer = 0; peer < SCATTER_RANKS; ++peer) {
        if (peer == param.myRank) {
            continue;
        }
        const auto &channel = ctx.channels[ch++];
        if (channel.remoteRank != peer || channel.notifyNum < 2 || channel.remoteCclMem.addr == nullptr
            || channel.remoteCclMem.size != ctx.localBuffer.size) {
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &ctx)
{
    CHK_RET(ValidateScatter(param, ctx));
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    if (param.rankSize == 1) {
        uint64_t offset = 0;
        const uint64_t bytes = param.count * sizeof(float);
        while (offset < bytes && param.inputPtr != param.outputPtr) {
            const uint64_t len = std::min(SCATTER_MAX_TRANSFER, bytes - offset);
            CHK_RET(
                HcommLocalCopyOnThread(ctx.aicpuThread, At(param.outputPtr, offset), At(param.inputPtr, offset), len));
            offset += len;
        }
        return HCCL_SUCCESS;
    }
    const uint64_t bytes = param.count * sizeof(float);
    if (bytes <= SMALL_MESSAGE_BYTES && bytes <= ctx.localBuffer.size) {
        return DirectScatter(param, ctx);
    }
    return BalancedScatter(param, ctx);
}
} // namespace ops_hccl
