/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>

#include <ccu/ccu_primitives.hpp>

#include "ccu_kernel.h"
#include "custom.h"
#include "log.h"

namespace ops_hccl {
namespace ccu = ::AscendC::ccu;

#define CCU_CHK_RET(call) \
    do { \
        CcuResult ccuRet = static_cast<CcuResult>(call); \
        if (ccuRet != CCU_SUCCESS) { \
            HCCL_ERROR("[%s] CCU call failed[%d]", __func__, ccuRet); \
            return ccuRet; \
        } \
    } while (0)

namespace {
    constexpr uint16_t OUTPUT_XN_ID = 1;
    constexpr uint16_t TOKEN_XN_ID = 2;
    constexpr uint16_t DATA_DONE_ID0 = 3;
    constexpr uint16_t ALLGATHER_DONE_ID0 = 4;
    constexpr uint16_t DATA_DONE_ID1 = 5;
    constexpr uint16_t ALLGATHER_DONE_ID1 = 6;

    struct BroadcastContext {
        const CcuKernelArgBroadcast *arg = nullptr;
        std::vector<ccu::Variable> remoteOutput;
        std::vector<ccu::Variable> remoteToken;
        ccu::Variable outputAddr;
        ccu::Variable outputToken;
        ccu::Variable dataBytes;
        ccu::Variable dataOffset;
        ccu::Variable normalSliceBytes;
        ccu::Variable lastSliceBytes;
        ccu::Variable dataBytes1;
        ccu::Variable normalSliceBytes1;
        ccu::Variable lastSliceBytes1;
        ccu::Event event;
    };

    uint32_t GroupChannelIndex(const BroadcastContext &ctx, uint32_t groupPeer)
    {
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            if (ctx.arg->peerRanks[i] == groupPeer) {
                return i;
            }
        }
        return ctx.arg->channelCount;
    }

    CcuResult InitResources(BroadcastContext &ctx)
    {
        if (ctx.arg == nullptr || ctx.arg->channelCount == 0 || ctx.arg->channelCount > MAX_RANK_SIZE) {
            return CCU_E_PARA;
        }
        ctx.remoteOutput.resize(ctx.arg->channelCount);
        ctx.remoteToken.resize(ctx.arg->channelCount);
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            ctx.remoteOutput[i] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[i], OUTPUT_XN_ID);
            ctx.remoteToken[i] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[i], TOKEN_XN_ID);
        }
        return CCU_SUCCESS;
    }

    CcuResult LoadArgs(BroadcastContext &ctx)
    {
        uint32_t argId = 0;
        CCU_CHK_RET(ccu::LoadArg(ctx.outputAddr, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.outputToken, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.dataBytes, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.dataOffset, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.normalSliceBytes, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.lastSliceBytes, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.dataBytes1, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.normalSliceBytes1, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.lastSliceBytes1, argId++));
        return CCU_SUCCESS;
    }

    CcuResult PreSync(BroadcastContext &ctx)
    {
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            CCU_CHK_RET(ccu::WriteVariableWithNotify(
                ctx.arg->channels[i], ctx.outputAddr, OUTPUT_XN_ID, ctx.arg->ckeIndex, 1 << OUTPUT_XN_ID));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(
                ctx.arg->channels[i], ctx.outputToken, TOKEN_XN_ID, ctx.arg->ckeIndex, 1 << TOKEN_XN_ID));
        }
        constexpr uint16_t ADDRESS_MASK = (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[i], ctx.arg->ckeIndex, ADDRESS_MASK));
        }
        return CCU_SUCCESS;
    }

    CcuResult WriteOne(BroadcastContext &ctx, uint32_t channelIndex, ccu::Variable &remoteAddr, ccu::Variable &offset,
        ccu::Variable &bytes, uint16_t eventMask)
    {
        ccu::LocalAddr src;
        src.addr = ctx.outputAddr;
        src.addr += ctx.dataOffset;
        src.addr += offset;
        src.token = ctx.outputToken;
        ccu::RemoteAddr dst;
        dst.addr = remoteAddr;
        dst.addr += ctx.dataOffset;
        dst.addr += offset;
        dst.token = ctx.remoteToken[channelIndex];
        CCU_IF(bytes != 0)
        {
            CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIndex], dst, src, bytes, ctx.event, eventMask));
        }
        CCU_IF(bytes == 0)
        {
            CCU_CHK_RET(ccu::EventRecord(ctx.event, eventMask));
        }
        return CCU_SUCCESS;
    }

    CcuResult RunDirect(BroadcastContext &ctx, uint16_t doneId)
    {
        if (ctx.arg->groupRank == ctx.arg->groupRoot) {
            ccu::Variable zeroOffset;
            zeroOffset = 0;
            uint16_t eventMask = 0;
            for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
                const uint16_t peerMask = static_cast<uint16_t>(1U << (channelIndex + 1));
                CCU_CHK_RET(
                    WriteOne(ctx, channelIndex, ctx.remoteOutput[channelIndex], zeroOffset, ctx.dataBytes, peerMask));
                eventMask |= peerMask;
            }
            CCU_CHK_RET(ccu::EventWait(ctx.event, eventMask));
            for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
                CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[channelIndex], ctx.arg->ckeIndex, 1 << doneId));
            }
        } else {
            const uint32_t rootChannel = GroupChannelIndex(ctx, ctx.arg->groupRoot);
            if (rootChannel >= ctx.arg->channelCount) {
                return CCU_SUCCESS;
            }
            CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[rootChannel], ctx.arg->ckeIndex, 1 << doneId));
        }
        return CCU_SUCCESS;
    }

    CcuResult RunScatter(BroadcastContext &ctx, uint16_t doneId)
    {
        if (ctx.arg->groupRank != ctx.arg->groupRoot) {
            const uint32_t rootChannel = GroupChannelIndex(ctx, ctx.arg->groupRoot);
            if (rootChannel >= ctx.arg->channelCount) {
                return CCU_SUCCESS;
            }
            return ccu::NotifyWait(ctx.arg->channels[rootChannel], ctx.arg->ckeIndex, 1 << doneId);
        }

        uint16_t eventMask = 0;
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            const uint32_t peer = ctx.arg->peerRanks[channelIndex];
            ccu::Variable sliceOffset;
            sliceOffset = 0;
            for (uint32_t i = 0; i < peer; ++i) {
                sliceOffset += ctx.normalSliceBytes;
            }
            ccu::Variable &sliceBytes = peer + 1 == ctx.arg->groupSize ? ctx.lastSliceBytes : ctx.normalSliceBytes;
            const uint16_t peerMask = static_cast<uint16_t>(1U << (channelIndex + 1));
            CCU_CHK_RET(
                WriteOne(ctx, channelIndex, ctx.remoteOutput[channelIndex], sliceOffset, sliceBytes, peerMask));
            eventMask |= peerMask;
        }
        if (eventMask != 0) {
            CCU_CHK_RET(ccu::EventWait(ctx.event, eventMask));
            for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
                CCU_CHK_RET(
                    ccu::NotifyRecord(ctx.arg->channels[channelIndex], ctx.arg->ckeIndex, 1 << doneId));
            }
        }
        return CCU_SUCCESS;
    }

    CcuResult RunAllGather(BroadcastContext &ctx, uint16_t doneId)
    {
        ccu::Variable ownOffset;
        ownOffset = 0;
        for (uint32_t rank = 0; rank < ctx.arg->groupRank; ++rank) {
            ownOffset += ctx.normalSliceBytes;
        }
        ccu::Variable &ownBytes
            = ctx.arg->groupRank + 1 == ctx.arg->groupSize ? ctx.lastSliceBytes : ctx.normalSliceBytes;
        uint16_t eventMask = 0;
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            const uint16_t peerMask = static_cast<uint16_t>(1U << (channelIndex + 1));
            CCU_CHK_RET(WriteOne(ctx, channelIndex, ctx.remoteOutput[channelIndex], ownOffset, ownBytes, peerMask));
            eventMask |= peerMask;
        }
        if (eventMask != 0) {
            CCU_CHK_RET(ccu::EventWait(ctx.event, eventMask));
            for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
                CCU_CHK_RET(
                    ccu::NotifyRecord(ctx.arg->channels[channelIndex], ctx.arg->ckeIndex, 1 << doneId));
            }
        }
        return CCU_SUCCESS;
    }

    CcuResult RunComplete(BroadcastContext &ctx, uint16_t doneId)
    {
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            CCU_CHK_RET(
                ccu::NotifyWait(ctx.arg->channels[channelIndex], ctx.arg->ckeIndex, 1 << doneId));
        }
        return CCU_SUCCESS;
    }

    uint32_t OwnerIndex(uint32_t rank, uint32_t root)
    {
        return rank < root ? rank : rank - 1;
    }

    CcuResult RunRootExcludedScatter(BroadcastContext &ctx, uint16_t doneId)
    {
        if (ctx.arg->groupRank != ctx.arg->groupRoot) {
            const uint32_t rootChannel = GroupChannelIndex(ctx, ctx.arg->groupRoot);
            if (rootChannel >= ctx.arg->channelCount) {
                return CCU_E_INTERNAL;
            }
            return ccu::NotifyWait(ctx.arg->channels[rootChannel], ctx.arg->ckeIndex, 1 << doneId);
        }

        const uint32_t ownerCount = ctx.arg->groupSize - 1;
        uint16_t eventMask = 0;
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            const uint32_t owner = OwnerIndex(ctx.arg->peerRanks[channelIndex], ctx.arg->groupRoot);
            ccu::Variable sliceOffset;
            sliceOffset = 0;
            for (uint32_t index = 0; index < owner; ++index) {
                sliceOffset += ctx.normalSliceBytes;
            }
            ccu::Variable &sliceBytes = owner + 1 == ownerCount ? ctx.lastSliceBytes : ctx.normalSliceBytes;
            const uint16_t peerMask = static_cast<uint16_t>(1U << (channelIndex + 1));
            CCU_CHK_RET(
                WriteOne(ctx, channelIndex, ctx.remoteOutput[channelIndex], sliceOffset, sliceBytes, peerMask));
            eventMask |= peerMask;
        }
        CCU_CHK_RET(ccu::EventWait(ctx.event, eventMask));
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[channelIndex], ctx.arg->ckeIndex, 1 << doneId));
        }
        return CCU_SUCCESS;
    }

    CcuResult RunRootExcludedAllGather(BroadcastContext &ctx, uint16_t doneId)
    {
        if (ctx.arg->groupRank == ctx.arg->groupRoot) {
            return CCU_SUCCESS;
        }
        const uint32_t ownerCount = ctx.arg->groupSize - 1;
        const uint32_t owner = OwnerIndex(ctx.arg->groupRank, ctx.arg->groupRoot);
        ccu::Variable ownOffset;
        ownOffset = 0;
        for (uint32_t index = 0; index < owner; ++index) {
            ownOffset += ctx.normalSliceBytes;
        }
        ccu::Variable &ownBytes = owner + 1 == ownerCount ? ctx.lastSliceBytes : ctx.normalSliceBytes;
        uint16_t eventMask = 0;
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            if (ctx.arg->peerRanks[channelIndex] == ctx.arg->groupRoot) {
                continue;
            }
            const uint16_t peerMask = static_cast<uint16_t>(1U << (channelIndex + 1));
            CCU_CHK_RET(WriteOne(ctx, channelIndex, ctx.remoteOutput[channelIndex], ownOffset, ownBytes, peerMask));
            eventMask |= peerMask;
        }
        CCU_CHK_RET(ccu::EventWait(ctx.event, eventMask));
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            if (ctx.arg->peerRanks[channelIndex] != ctx.arg->groupRoot) {
                CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[channelIndex], ctx.arg->ckeIndex, 1 << doneId));
            }
        }
        return CCU_SUCCESS;
    }

    CcuResult RunRootExcludedComplete(BroadcastContext &ctx, uint16_t doneId)
    {
        if (ctx.arg->groupRank == ctx.arg->groupRoot) {
            return CCU_SUCCESS;
        }
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            if (ctx.arg->peerRanks[channelIndex] != ctx.arg->groupRoot) {
                CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIndex], ctx.arg->ckeIndex, 1 << doneId));
            }
        }
        return CCU_SUCCESS;
    }

    CcuResult RunTree(BroadcastContext &ctx)
    {
        ccu::Variable zeroOffset;
        zeroOffset = 0;
        uint16_t eventMask = 0;
        for (uint32_t opIndex = 0; opIndex < ctx.arg->opCount; ++opIndex) {
            const CcuBroadcastOpDesc &op = ctx.arg->ops[opIndex];
            if (op.channelIndex >= ctx.arg->channelCount) {
                return CCU_E_INTERNAL;
            }
            if (op.kind == static_cast<uint32_t>(CcuBroadcastOpKind::RECV)) {
                CCU_CHK_RET(
                    ccu::NotifyWait(ctx.arg->channels[op.channelIndex], ctx.arg->ckeIndex, 1 << DATA_DONE_ID0));
            } else if (op.kind == static_cast<uint32_t>(CcuBroadcastOpKind::SEND)) {
                const uint16_t opMask = static_cast<uint16_t>(1U << (opIndex + 1));
                CCU_CHK_RET(WriteOne(
                    ctx, op.channelIndex, ctx.remoteOutput[op.channelIndex], zeroOffset, ctx.dataBytes, opMask));
                eventMask |= opMask;
            } else {
                return CCU_E_PARA;
            }
        }
        if (eventMask != 0) {
            CCU_CHK_RET(ccu::EventWait(ctx.event, eventMask));
        }
        for (uint32_t opIndex = 0; opIndex < ctx.arg->opCount; ++opIndex) {
            const CcuBroadcastOpDesc &op = ctx.arg->ops[opIndex];
            if (op.kind == static_cast<uint32_t>(CcuBroadcastOpKind::SEND)) {
                CCU_CHK_RET(
                    ccu::NotifyRecord(ctx.arg->channels[op.channelIndex], ctx.arg->ckeIndex, 1 << DATA_DONE_ID0));
            }
        }
        return CCU_SUCCESS;
    }

    CcuResult RunPlan(BroadcastContext &ctx, CcuBroadcastPlanKind plan, uint16_t dataDoneId,
        uint16_t allGatherDoneId)
    {
        if (plan == CcuBroadcastPlanKind::DIRECT || plan == CcuBroadcastPlanKind::GLOBAL_DIRECT) {
            return RunDirect(ctx, dataDoneId);
        }
        if (plan == CcuBroadcastPlanKind::SCATTER_GATHER) {
            CCU_CHK_RET(RunScatter(ctx, dataDoneId));
            CCU_CHK_RET(RunAllGather(ctx, allGatherDoneId));
            return RunComplete(ctx, allGatherDoneId);
        }
        if (plan == CcuBroadcastPlanKind::GLOBAL_SCATTER) {
            return RunScatter(ctx, dataDoneId);
        }
        if (plan == CcuBroadcastPlanKind::GLOBAL_ALLGATHER) {
            return RunAllGather(ctx, allGatherDoneId);
        }
        if (plan == CcuBroadcastPlanKind::GLOBAL_COMPLETE) {
            return RunComplete(ctx, allGatherDoneId);
        }
        if (plan == CcuBroadcastPlanKind::ROOT_EXCLUDED_SCATTER_GATHER) {
            CCU_CHK_RET(RunRootExcludedScatter(ctx, dataDoneId));
            CCU_CHK_RET(RunRootExcludedAllGather(ctx, allGatherDoneId));
            return RunRootExcludedComplete(ctx, allGatherDoneId);
        }
        if (plan == CcuBroadcastPlanKind::TREE) {
            return RunTree(ctx);
        }
        return CCU_E_PARA;
    }

} // namespace

CcuResult CcuKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgBroadcast *>(arg);
    if (kernelArg == nullptr || kernelArg->root >= kernelArg->rankSize || kernelArg->groupSize < 2
        || kernelArg->groupRank >= kernelArg->groupSize || kernelArg->groupRoot >= kernelArg->groupSize
        || kernelArg->opCount > CCU_BROADCAST_MAX_OPS) {
        return CCU_E_PARA;
    }

    BroadcastContext ctx;
    ctx.arg = kernelArg;
    CCU_CHK_RET(InitResources(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    const auto plan = static_cast<CcuBroadcastPlanKind>(kernelArg->planKind);
    // Scatter publishes the user address once per launch.  The following
    // AllGather kernels are ordered after Scatter on the same launch stream,
    // so repeating the address exchange would only add fixed latency.
    if (plan != CcuBroadcastPlanKind::GLOBAL_ALLGATHER && plan != CcuBroadcastPlanKind::GLOBAL_COMPLETE) {
        CCU_CHK_RET(PreSync(ctx));
    }
    CCU_CHK_RET(RunPlan(ctx, plan, DATA_DONE_ID0, ALLGATHER_DONE_ID0));
    CCU_IF(ctx.dataBytes1 != 0)
    {
        ctx.dataOffset += ctx.dataBytes;
        ctx.dataBytes = ctx.dataBytes1;
        ctx.normalSliceBytes = ctx.normalSliceBytes1;
        ctx.lastSliceBytes = ctx.lastSliceBytes1;
        CCU_CHK_RET(RunPlan(ctx, plan, DATA_DONE_ID1, ALLGATHER_DONE_ID1));
    }
    return CCU_SUCCESS;
}
} // namespace ops_hccl
