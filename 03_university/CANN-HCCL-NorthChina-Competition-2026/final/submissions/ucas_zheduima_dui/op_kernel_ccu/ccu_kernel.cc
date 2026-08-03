/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <vector>

#include "common.h"
#include "custom.h"
#include "ccu_kernel.h"
#include "log.h"

#define RETURN_IF_CCU_ERROR(call) \
    do { \
        CcuResult ccuCallResult = (call); \
        if (ccuCallResult != CCU_SUCCESS) { \
            return ccuCallResult; \
        } \
    } while (0)

namespace ops_hccl {

namespace ccu = ::AscendC::ccu;

namespace {

    constexpr uint32_t REMOTE_OUTPUT_VAR_ID = 1;
    constexpr uint32_t REMOTE_TOKEN_VAR_ID = 2;
    constexpr uint32_t CHANNEL_NOTIFY_IDX = 0;
    // notify 位分配（slot 0）：地址/token 交换用 bit1|bit2（仅首轮或 VA 变化时）；
    // 每次 launch 的 PostSync 为普通 NotifyRecord，在 bit3 与 bit4 之间按 launch 奇偶轮换，
    // 保证同一 bit 的 record/wait 中间至少隔一次完整 launch（防多打一坍缩）。
    constexpr uint32_t EXCHANGE_MASK = (1U << REMOTE_OUTPUT_VAR_ID) | (1U << REMOTE_TOKEN_VAR_ID);
    constexpr uint16_t POST_MASK_EVEN = 1U << 3;
    constexpr uint16_t POST_MASK_ODD = 1U << 4;

    struct DirectPushContext {
        const CcuKernelArgAllGather *arg = nullptr;
        ccu::Variable input;
        ccu::Variable output;
        ccu::Variable inputToken;
        ccu::Variable outputToken;
        ccu::Variable dataSize;
        ccu::Variable outputOffset;
        ccu::Variable mode;
        ccu::Variable collectPartSize; // A：服内收集 + phase-2 转发
        ccu::Variable directPartSize;  // D：phase-1 扁平直推
        ccu::Variable relayPartSize;   // S2：写 pair + 服内广播
        ccu::Variable localBlockOffset;
        ccu::Variable firstCall;
        ccu::Variable syncParity;
        std::vector<ccu::Variable> remoteOutput;
        std::vector<ccu::Variable> remoteToken;
        ccu::Event event;
    };

    CcuResult InitResources(DirectPushContext &ctx)
    {
        ctx.remoteOutput.resize(ctx.arg->channelCount);
        ctx.remoteToken.resize(ctx.arg->channelCount);
        for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
            ctx.remoteOutput[channelIdx]
                = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], REMOTE_OUTPUT_VAR_ID);
            ctx.remoteToken[channelIdx]
                = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], REMOTE_TOKEN_VAR_ID);
        }
        return CCU_SUCCESS;
    }

    CcuResult LoadTaskArgs(DirectPushContext &ctx)
    {
        uint32_t argIdx = 0;
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.input, argIdx++));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.output, argIdx++));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.inputToken, argIdx++));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.outputToken, argIdx++));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.dataSize, argIdx++));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.outputOffset, argIdx++));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.mode, argIdx++));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.collectPartSize, argIdx++));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.directPartSize, argIdx++));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.relayPartSize, argIdx++));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.localBlockOffset, argIdx++));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.firstCall, argIdx++));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.syncParity, argIdx++));
        return CCU_SUCCESS;
    }

    // 首轮显式交换本端 recvBuf 地址与 token；后续轮由 PostSync 捎带，无需本步骤
    CcuResult ExchangeRemoteOutput(DirectPushContext &ctx)
    {
        for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
            RETURN_IF_CCU_ERROR(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx], ctx.output,
                REMOTE_OUTPUT_VAR_ID, CHANNEL_NOTIFY_IDX, 1U << REMOTE_OUTPUT_VAR_ID));
            RETURN_IF_CCU_ERROR(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx], ctx.outputToken,
                REMOTE_TOKEN_VAR_ID, CHANNEL_NOTIFY_IDX, 1U << REMOTE_TOKEN_VAR_ID));
        }
        for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
            RETURN_IF_CCU_ERROR(
                ccu::NotifyWait(ctx.arg->channels[channelIdx], CHANNEL_NOTIFY_IDX, EXCHANGE_MASK));
        }
        return CCU_SUCCESS;
    }

    // 后同步：普通 NotifyRecord/Wait，等待对端写齐 = 确认对端本阶段数据已全部落盘。
    // 地址/token 只在首轮或 VA 变化时经 ExchangeRemoteOutput 刷新（harness buffer 同址复用），
    // notify 位由调用方按 launch 奇偶指定，保证同一 bit 的 record/wait 严格交替（防多打一坍缩）
    CcuResult PostSync(DirectPushContext &ctx, uint16_t syncMask)
    {
        for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
            RETURN_IF_CCU_ERROR(
                ccu::NotifyRecord(ctx.arg->channels[channelIdx], CHANNEL_NOTIFY_IDX, syncMask));
        }
        for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
            RETURN_IF_CCU_ERROR(ccu::NotifyWait(ctx.arg->channels[channelIdx], CHANNEL_NOTIFY_IDX, syncMask));
        }
        return CCU_SUCCESS;
    }

    // 信用后移（小消息）：本轮只 record，wait 推迟到下一轮开头——信号飞行时间被整轮覆盖，
    // 每轮省一次完整 RT。上一轮 notify 位与本论奇偶相反。
    CcuResult PostRecordOnly(DirectPushContext &ctx, uint16_t syncMask)
    {
        for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
            RETURN_IF_CCU_ERROR(
                ccu::NotifyRecord(ctx.arg->channels[channelIdx], CHANNEL_NOTIFY_IDX, syncMask));
        }
        return CCU_SUCCESS;
    }

    CcuResult PostWaitOnly(DirectPushContext &ctx, uint16_t syncMask)
    {
        for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
            RETURN_IF_CCU_ERROR(ccu::NotifyWait(ctx.arg->channels[channelIdx], CHANNEL_NOTIFY_IDX, syncMask));
        }
        return CCU_SUCCESS;
    }

    uint16_t GetCompleteMask(uint32_t operationCount)
    {
        return (operationCount >= 16U) ? 0xFFFFU : static_cast<uint16_t>((1U << operationCount) - 1U);
    }

    // 扁平直推（整写版）：小消息与 4*1 拓扑（单波时延最优）
    CcuResult DirectTransferWhole(DirectPushContext &ctx)
    {
        ccu::LocalAddr src;
        src.addr = ctx.input;
        src.token = ctx.inputToken;

        std::vector<ccu::RemoteAddr> remoteDst(ctx.arg->channelCount);
        for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
            remoteDst[channelIdx].addr = ctx.remoteOutput[channelIdx];
            remoteDst[channelIdx].addr += ctx.outputOffset;
            remoteDst[channelIdx].token = ctx.remoteToken[channelIdx];
            const uint16_t mask = static_cast<uint16_t>(1U << channelIdx);
            RETURN_IF_CCU_ERROR(
                ccu::Write(ctx.arg->channels[channelIdx], remoteDst[channelIdx], src, ctx.dataSize, ctx.event, mask));
        }

        uint32_t operationCount = ctx.arg->channelCount;
        if (ctx.arg->copySelf != 0) {
            ccu::LocalAddr localDst;
            localDst.addr = ctx.output;
            localDst.addr += ctx.outputOffset;
            localDst.token = ctx.outputToken;
            const uint16_t localMask = static_cast<uint16_t>(1U << operationCount);
            RETURN_IF_CCU_ERROR(ccu::LocalCopy(localDst, src, ctx.dataSize, ctx.event, localMask));
            ++operationCount;
        }

        RETURN_IF_CCU_ERROR(ccu::EventWait(ctx.event, GetCompleteMask(operationCount)));
        return CCU_SUCCESS;
    }

    // 4*1 递归倍增 round-1：与 r^1 交换 own 切片 + own 本地落盘
    CcuResult RoundOneTransfer(DirectPushContext &ctx)
    {
        ccu::LocalAddr src;
        src.addr = ctx.input;
        src.token = ctx.inputToken;

        ccu::RemoteAddr dst;
        dst.addr = ctx.remoteOutput[ctx.arg->pairChannelIdx];
        dst.addr += ctx.outputOffset;
        dst.token = ctx.remoteToken[ctx.arg->pairChannelIdx];
        RETURN_IF_CCU_ERROR(
            ccu::Write(ctx.arg->channels[ctx.arg->pairChannelIdx], dst, src, ctx.dataSize, ctx.event, 1));

        ccu::LocalAddr localDst;
        localDst.addr = ctx.output;
        localDst.addr += ctx.outputOffset;
        localDst.token = ctx.outputToken;
        RETURN_IF_CCU_ERROR(ccu::LocalCopy(localDst, src, ctx.dataSize, ctx.event, 2));
        RETURN_IF_CCU_ERROR(ccu::EventWait(ctx.event, 3));
        return CCU_SUCCESS;
    }

    // 4*1 递归倍增 round-2：把相邻两片（pairBaseOffset 起，各 dataSize）分两条并行写给 r^2，
    // 源为本端 recvBuf（own 与 r^1 的切片已于 round-1 落盘）。
    // 注：拆两条 S 写以排除单条 2S 大写的速率损失
    CcuResult RoundTwoTransfer(DirectPushContext &ctx)
    {
        ccu::LocalAddr src;
        src.addr = ctx.output;
        src.addr += ctx.collectPartSize; // pairBaseOffset
        src.token = ctx.outputToken;

        ccu::RemoteAddr dst;
        dst.addr = ctx.remoteOutput[ctx.arg->pairChannelIdxR2];
        dst.addr += ctx.collectPartSize; // pairBaseOffset
        dst.token = ctx.remoteToken[ctx.arg->pairChannelIdxR2];

        RETURN_IF_CCU_ERROR(
            ccu::Write(ctx.arg->channels[ctx.arg->pairChannelIdxR2], dst, src, ctx.dataSize, ctx.event, 1));

        ccu::LocalAddr srcSecond;
        srcSecond.addr = ctx.output;
        srcSecond.addr += ctx.collectPartSize;
        srcSecond.addr += ctx.dataSize;
        srcSecond.token = ctx.outputToken;

        ccu::RemoteAddr dstSecond;
        dstSecond.addr = ctx.remoteOutput[ctx.arg->pairChannelIdxR2];
        dstSecond.addr += ctx.collectPartSize;
        dstSecond.addr += ctx.dataSize;
        dstSecond.token = ctx.remoteToken[ctx.arg->pairChannelIdxR2];

        RETURN_IF_CCU_ERROR(
            ccu::Write(ctx.arg->channels[ctx.arg->pairChannelIdxR2], dstSecond, srcSecond, ctx.dataSize, ctx.event, 2));
        RETURN_IF_CCU_ERROR(ccu::EventWait(ctx.event, 3));
        return CCU_SUCCESS;
    }

    CcuResult DirectTransfer(DirectPushContext &ctx)
    {
        return DirectTransferWhole(ctx);
    }

    // Phase-1（layer-0）：服内 allgather 前 X = A+D 字节（A 供 phase-2 转发，D 供服内分发）
    CcuResult PhaseOneLocal(DirectPushContext &ctx)
    {
        ccu::Variable gatherSize;
        gatherSize = ctx.collectPartSize;
        gatherSize += ctx.directPartSize;

        ccu::LocalAddr src;
        src.addr = ctx.input;
        src.token = ctx.inputToken;

        uint32_t operationCount = 0;
        for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
            ccu::RemoteAddr dst;
            dst.addr = ctx.remoteOutput[channelIdx];
            dst.addr += ctx.outputOffset;
            dst.token = ctx.remoteToken[channelIdx];
            RETURN_IF_CCU_ERROR(ccu::Write(ctx.arg->channels[channelIdx], dst, src, gatherSize, ctx.event,
                static_cast<uint16_t>(1U << operationCount)));
            ++operationCount;
        }
        ccu::LocalAddr localDst;
        localDst.addr = ctx.output;
        localDst.addr += ctx.outputOffset;
        localDst.token = ctx.outputToken;
        RETURN_IF_CCU_ERROR(ccu::LocalCopy(
            localDst, src, gatherSize, ctx.event, static_cast<uint16_t>(1U << operationCount)));
        ++operationCount;
        RETURN_IF_CCU_ERROR(ccu::EventWait(ctx.event, GetCompleteMask(operationCount)));
        return CCU_SUCCESS;
    }

    // Phase-1（layer-1）：D 段扁平直推全部远端对端 + 尾段 S2 写 pair（由 pair 服内广播）
    CcuResult PhaseOneRemote(DirectPushContext &ctx)
    {
        // D 段直推
        ccu::LocalAddr srcDirect;
        srcDirect.addr = ctx.input;
        srcDirect.addr += ctx.collectPartSize;
        srcDirect.token = ctx.inputToken;

        uint32_t operationCount = 0;
        for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
            ccu::RemoteAddr dst;
            dst.addr = ctx.remoteOutput[channelIdx];
            dst.addr += ctx.outputOffset;
            dst.addr += ctx.collectPartSize;
            dst.token = ctx.remoteToken[channelIdx];
            RETURN_IF_CCU_ERROR(ccu::Write(ctx.arg->channels[channelIdx], dst, srcDirect, ctx.directPartSize,
                ctx.event, static_cast<uint16_t>(1U << operationCount)));
            ++operationCount;
        }
        // 尾段 S2 → pair
        ccu::Variable relayOffset;
        relayOffset = ctx.collectPartSize;
        relayOffset += ctx.directPartSize;

        ccu::LocalAddr srcRelay;
        srcRelay.addr = ctx.input;
        srcRelay.addr += relayOffset;
        srcRelay.token = ctx.inputToken;

        ccu::RemoteAddr pairDst;
        pairDst.addr = ctx.remoteOutput[ctx.arg->pairChannelIdx];
        pairDst.addr += ctx.outputOffset;
        pairDst.addr += relayOffset;
        pairDst.token = ctx.remoteToken[ctx.arg->pairChannelIdx];
        RETURN_IF_CCU_ERROR(ccu::Write(ctx.arg->channels[ctx.arg->pairChannelIdx], pairDst, srcRelay,
            ctx.relayPartSize, ctx.event, static_cast<uint16_t>(1U << operationCount)));
        ++operationCount;
        // own 尾段本地落盘（phase-2 服内广播以本端 recvBuf 为源）
        ccu::LocalAddr localRelayDst;
        localRelayDst.addr = ctx.output;
        localRelayDst.addr += ctx.outputOffset;
        localRelayDst.addr += relayOffset;
        localRelayDst.token = ctx.outputToken;
        RETURN_IF_CCU_ERROR(ccu::LocalCopy(localRelayDst, srcRelay, ctx.relayPartSize, ctx.event,
            static_cast<uint16_t>(1U << operationCount)));
        ++operationCount;
        RETURN_IF_CCU_ERROR(ccu::EventWait(ctx.event, GetCompleteMask(operationCount)));
        return CCU_SUCCESS;
    }

    // Phase-2（layer-1）：把服内收集的全部 A 段转发给指定目标（pair/多目标）
    CcuResult PhaseTwoRemote(DirectPushContext &ctx)
    {
        uint32_t operationCount = 0;
        for (uint32_t targetIdx = 0; targetIdx < ctx.arg->stageTwoTargetCount; ++targetIdx) {
            const uint32_t channelIdx = ctx.arg->stageTwoTargetChannelIdx[targetIdx];
            ccu::Variable rankOffset;
            rankOffset = ctx.localBlockOffset;
            for (uint32_t rankIdx = 0; rankIdx < ctx.arg->localRankCount; ++rankIdx) {
                ccu::LocalAddr src;
                src.addr = ctx.output;
                src.addr += rankOffset;
                src.token = ctx.outputToken;

                ccu::RemoteAddr dst;
                dst.addr = ctx.remoteOutput[channelIdx];
                dst.addr += rankOffset;
                dst.token = ctx.remoteToken[channelIdx];

                RETURN_IF_CCU_ERROR(ccu::Write(ctx.arg->channels[channelIdx], dst, src, ctx.collectPartSize,
                    ctx.event, static_cast<uint16_t>(1U << operationCount)));
                ++operationCount;
                rankOffset += ctx.dataSize;
            }
        }
        if (operationCount != 0) {
            RETURN_IF_CCU_ERROR(ccu::EventWait(ctx.event, GetCompleteMask(operationCount)));
        }
        return CCU_SUCCESS;
    }

    // Phase-2（layer-0）：服内广播 own 尾段 + 各 owned-remote 的尾段
    // rankOffset 为切片起始偏移，尾段在切片内的偏移为 A + D
    CcuResult BroadcastRelayPart(DirectPushContext &ctx, ccu::Variable rankOffset, uint32_t &operationCount)
    {
        ccu::Variable relayOffset;
        relayOffset = ctx.collectPartSize;
        relayOffset += ctx.directPartSize;

        ccu::LocalAddr src;
        src.addr = ctx.output;
        src.addr += rankOffset;
        src.addr += relayOffset;
        src.token = ctx.outputToken;

        for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
            ccu::RemoteAddr dst;
            dst.addr = ctx.remoteOutput[channelIdx];
            dst.addr += rankOffset;
            dst.addr += relayOffset;
            dst.token = ctx.remoteToken[channelIdx];
            RETURN_IF_CCU_ERROR(ccu::Write(ctx.arg->channels[channelIdx], dst, src, ctx.relayPartSize, ctx.event,
                static_cast<uint16_t>(1U << operationCount)));
            ++operationCount;
        }
        return CCU_SUCCESS;
    }

    CcuResult PhaseTwoLocal(DirectPushContext &ctx)
    {
        uint32_t operationCount = 0;
        // own 尾段（从本端 recvBuf 取，phase-1 已由 LocalCopy 落盘）
        RETURN_IF_CCU_ERROR(BroadcastRelayPart(ctx, ctx.outputOffset, operationCount));
        // 各 owned-remote 尾段（phase-1 经 Clos 到达本端 recvBuf）
        for (uint32_t ownedIdx = 0; ownedIdx < ctx.arg->ownedRemoteCount; ++ownedIdx) {
            ccu::Variable ownedOffset;
            ownedOffset = 0;
            for (uint32_t offsetIdx = 0; offsetIdx < ctx.arg->ownedRemoteRanks[ownedIdx]; ++offsetIdx) {
                ownedOffset += ctx.dataSize;
            }
            RETURN_IF_CCU_ERROR(BroadcastRelayPart(ctx, ownedOffset, operationCount));
        }
        RETURN_IF_CCU_ERROR(ccu::EventWait(ctx.event, GetCompleteMask(operationCount)));
        return CCU_SUCCESS;
    }

    CcuResult Transfer(DirectPushContext &ctx)
    {
        CCU_IF(ctx.mode == TRANSFER_PHASE_ONE)
        {
            if (ctx.arg->isLocalLayer != 0) {
                RETURN_IF_CCU_ERROR(PhaseOneLocal(ctx));
            } else {
                RETURN_IF_CCU_ERROR(PhaseOneRemote(ctx));
            }
        }
        CCU_ELSE
        {
            CCU_IF(ctx.mode == TRANSFER_PHASE_TWO)
            {
                if (ctx.arg->isLocalLayer != 0) {
                    RETURN_IF_CCU_ERROR(PhaseTwoLocal(ctx));
                } else {
                    RETURN_IF_CCU_ERROR(PhaseTwoRemote(ctx));
                }
            }
            CCU_ELSE
            {
                RETURN_IF_CCU_ERROR(DirectTransfer(ctx));
            }
        }
        return CCU_SUCCESS;
    }

} // namespace

CcuResult CcuAllGatherDirectPush(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllGather *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount == 0 || kernelArg->channelCount >= MAX_RANK_SIZE) {
        HCCL_ERROR("[AllGather] Invalid CCU kernel argument");
        return CCU_E_PARA;
    }

    DirectPushContext ctx;
    ctx.arg = kernelArg;
    RETURN_IF_CCU_ERROR(InitResources(ctx));
    RETURN_IF_CCU_ERROR(LoadTaskArgs(ctx));
    CCU_IF(ctx.firstCall != 0)
    {
        RETURN_IF_CCU_ERROR(ExchangeRemoteOutput(ctx));
    }
    CCU_IF(ctx.mode == TRANSFER_FLAT)
    {
        // 小消息扁平整写：信用后移——先等上一轮（奇偶相反位）的 post，再传数据，
        // 本轮只 record 不 wait；首轮（firstCall）无上一轮可等
        CCU_IF(ctx.firstCall == 0)
        {
            CCU_IF(ctx.syncParity == 0)
            {
                RETURN_IF_CCU_ERROR(PostWaitOnly(ctx, POST_MASK_ODD));
            }
            CCU_ELSE
            {
                RETURN_IF_CCU_ERROR(PostWaitOnly(ctx, POST_MASK_EVEN));
            }
        }
        RETURN_IF_CCU_ERROR(DirectTransfer(ctx));
        CCU_IF(ctx.syncParity == 0)
        {
            RETURN_IF_CCU_ERROR(PostRecordOnly(ctx, POST_MASK_EVEN));
        }
        CCU_ELSE
        {
            RETURN_IF_CCU_ERROR(PostRecordOnly(ctx, POST_MASK_ODD));
        }
    }
    CCU_ELSE
    {
        RETURN_IF_CCU_ERROR(Transfer(ctx));
        CCU_IF(ctx.syncParity == 0)
        {
            RETURN_IF_CCU_ERROR(PostSync(ctx, POST_MASK_EVEN));
        }
        CCU_ELSE
        {
            RETURN_IF_CCU_ERROR(PostSync(ctx, POST_MASK_ODD));
        }
    }
    return CCU_SUCCESS;
}

namespace {

    // 极简扁平 kernel 的 taskArgs（10 个）：与 CcuAllGatherFlat 的 LoadArg 一一对应
    CcuResult FlatLoadTaskArgs(DirectPushContext &ctx)
    {
        uint32_t argIdx = 0;
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.input, argIdx++));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.output, argIdx++));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.inputToken, argIdx++));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.outputToken, argIdx++));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.dataSize, argIdx++));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.outputOffset, argIdx++));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.mode, argIdx++));
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.collectPartSize, argIdx++)); // round-2：pairBaseOffset
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.firstCall, argIdx++));       // 1 = 本轮先显式交换地址/token
        RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.syncParity, argIdx++));
        return CCU_SUCCESS;
    }

} // namespace

// 极简扁平 kernel：小消息与 4*1 拓扑；分层 kernel 之外的独立注册，指令数最小化
CcuResult CcuAllGatherFlat(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllGather *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount == 0 || kernelArg->channelCount >= MAX_RANK_SIZE) {
        HCCL_ERROR("[AllGather] Invalid CCU kernel argument");
        return CCU_E_PARA;
    }

    DirectPushContext ctx;
    ctx.arg = kernelArg;
    RETURN_IF_CCU_ERROR(InitResources(ctx));
    RETURN_IF_CCU_ERROR(FlatLoadTaskArgs(ctx));
    CCU_IF(ctx.firstCall != 0)
    {
        RETURN_IF_CCU_ERROR(ExchangeRemoteOutput(ctx));
    }
    CCU_IF(ctx.mode == TRANSFER_ROUND_FUSED)
    {
        // 4*1 递归倍增单 launch 版：round1（r^1，own S）→ 内部 barrier →
        // round2（r^2，相邻两片 2S）→ 收尾 barrier；两个 barrier 分用 even/odd 位
        RETURN_IF_CCU_ERROR(RoundOneTransfer(ctx));
        RETURN_IF_CCU_ERROR(PostSync(ctx, POST_MASK_EVEN));
        RETURN_IF_CCU_ERROR(RoundTwoTransfer(ctx));
        RETURN_IF_CCU_ERROR(PostSync(ctx, POST_MASK_ODD));
    }
    CCU_ELSE
    {
        // 小消息扁平整写：信用后移——先等上一轮（奇偶相反位）的 post，再传数据，
        // 本轮只 record 不 wait；首轮（firstCall）无上一轮可等
        CCU_IF(ctx.firstCall == 0)
        {
            CCU_IF(ctx.syncParity == 0)
            {
                RETURN_IF_CCU_ERROR(PostWaitOnly(ctx, POST_MASK_ODD));
            }
            CCU_ELSE
            {
                RETURN_IF_CCU_ERROR(PostWaitOnly(ctx, POST_MASK_EVEN));
            }
        }
        RETURN_IF_CCU_ERROR(DirectTransferWhole(ctx));
        CCU_IF(ctx.syncParity == 0)
        {
            RETURN_IF_CCU_ERROR(PostRecordOnly(ctx, POST_MASK_EVEN));
        }
        CCU_ELSE
        {
            RETURN_IF_CCU_ERROR(PostRecordOnly(ctx, POST_MASK_ODD));
        }
    }
    return CCU_SUCCESS;
}

} // namespace ops_hccl

#undef RETURN_IF_CCU_ERROR
