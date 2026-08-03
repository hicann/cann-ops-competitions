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

#include "ccu_kernel.h"
#include "log.h"

#define CCU_CHK_RET(call) \
    do { \
        const CcuResult ccuRet = static_cast<CcuResult>(call); \
        if (UNLIKELY(ccuRet != CCU_SUCCESS)) { \
            HCCL_ERROR("[%s] call trace: ccuRet -> %d", __func__, ccuRet); \
            return ccuRet; \
        } \
    } while (0)

namespace ops_hccl {
namespace {
constexpr uint32_t INPUT_XN_ID = 0;
constexpr uint32_t TOKEN_XN_ID = 1;
constexpr uint32_t CHANNEL_NOTIFY_IDX = 0;
constexpr uint16_t FINAL_BARRIER_MASK = 1U << 4;
constexpr uint16_t OWN_COPY_MASK = 1U << 15;

struct PullContext {
    const CcuKernelArgReduceScatterPull *arg = nullptr;
    ccu::Variable inputAddr;
    ccu::Variable scratchAddr;
    ccu::Variable inputToken;
    ccu::Variable scratchToken;
    ccu::Variable rankSliceBytes;
    ccu::Variable processedBytes;
    ccu::Variable sliceBytes;
    ccu::Variable inputOffset;
    std::vector<ccu::Variable> remoteInputAddr;
    std::vector<ccu::Variable> remoteInputToken;
    ccu::Event transferEvent;
};

struct ReduceContext {
    const CcuKernelArgReduceScatterReduce *arg = nullptr;
    ccu::Variable inputAddr;
    ccu::Variable outputAddr;
    ccu::Variable scratchAddr;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable scratchToken;
    ccu::Variable rankSliceBytes;
    ccu::Variable processedBytes;
    ccu::Variable sliceBytes;
    ccu::Variable tileBytes;
    ccu::Variable loopCounter;
    ccu::Variable lastTileBytes;
    ccu::Variable retainedPartialAddr;
    ccu::Variable mergeFirstBytes;
    ccu::Variable mergeSecondBytes;
    ccu::Event reduceEvent;
};

struct FusedContext {
    const CcuKernelArgReduceScatterPull *arg = nullptr;
    ccu::Variable inputAddr;
    ccu::Variable outputAddr;
    ccu::Variable scratchAddr;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable scratchToken;
    ccu::Variable rankSliceBytes;
    ccu::Variable processedBytes;
    ccu::Variable sliceBytes;
    ccu::Variable inputOffset;
    std::vector<ccu::Variable> remoteInputAddr;
    std::vector<ccu::Variable> remoteInputToken;
    ccu::Event transferEvent;
    ccu::Event reduceEvent;
};

struct StripeContext {
    const CcuKernelArgReduceScatterStripe *arg = nullptr;
    ccu::Variable inputAddr;
    ccu::Variable outputAddr;
    ccu::Variable scratchAddr;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable scratchToken;
    ccu::Variable rankSliceBytes;
    ccu::Variable processedBytes;
    ccu::Variable sliceBytes;
    ccu::Variable primaryStripeBytes;
    ccu::Variable secondaryStripeBytes;
    ccu::Variable primaryTailStripeBytes;
    ccu::Variable secondaryTailStripeBytes;
    ccu::Variable transferBytes;
    ccu::Variable inputOffset;
    ccu::Variable partialBaseAddr;
    ccu::Variable groupOffset;
    ccu::Variable groupBytes;
    std::vector<ccu::Variable> stripeOffsets;
    std::vector<ccu::Variable> remoteInputAddr;
    std::vector<ccu::Variable> remoteInputToken;
    ccu::LocalAddr ownContribution;
    ccu::RemoteAddr remoteInput;
    ccu::LocalAddr partial;
    ccu::Event transferEvent;
};

struct GroupFlatContext {
    const CcuKernelArgReduceScatterStripe *arg = nullptr;
    ccu::Variable inputAddr;
    ccu::Variable outputAddr;
    ccu::Variable scratchAddr;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable scratchToken;
    ccu::Variable rankSliceBytes;
    ccu::Variable processedBytes;
    ccu::Variable sliceBytes;
    ccu::Variable tileBytes;
    ccu::Variable loopCounter;
    ccu::Variable lastTileBytes;
    ccu::Variable inputOffset;
    ccu::Variable groupScratchAddr;
    ccu::Variable retainedPartialAddr;
    std::vector<ccu::Variable> remoteInputAddr;
    std::vector<ccu::Variable> remoteInputToken;
    ccu::Event transferEvent;
    ccu::Event reduceEvent;
};

CcuResult Barrier(const CcuKernelArgBase &arg, uint16_t mask)
{
    for (uint32_t channelIdx = 0; channelIdx < arg.channelCount; ++channelIdx) {
        CCU_CHK_RET(ccu::NotifyRecord(arg.channels[channelIdx], CHANNEL_NOTIFY_IDX, mask));
    }
    for (uint32_t channelIdx = 0; channelIdx < arg.channelCount; ++channelIdx) {
        CCU_CHK_RET(ccu::NotifyWait(arg.channels[channelIdx], CHANNEL_NOTIFY_IDX, mask));
    }
    return CCU_SUCCESS;
}

bool IsOneBit(uint16_t mask)
{
    return mask != 0 && (mask & static_cast<uint16_t>(mask - 1)) == 0;
}

CcuResult ValidatePullArg(const CcuKernelArgReduceScatterPull *arg, bool fused)
{
    const bool validRankSize = arg != nullptr &&
        (arg->rankSize == 4 || arg->rankSize == 12 || arg->rankSize == MAX_RANK_SIZE);
    if (!validRankSize || arg->rankId >= arg->rankSize || arg->channelCount == 0 ||
        arg->channelCount >= arg->rankSize || arg->dataType != HCCL_DATA_TYPE_FP32 ||
        arg->reduceOp != HCCL_REDUCE_SUM || !IsOneBit(arg->addrMask) || !IsOneBit(arg->tokenMask) ||
        (arg->addrMask & arg->tokenMask) != 0 ||
        ((arg->addrMask | arg->tokenMask) & FINAL_BARRIER_MASK) != 0) {
        HCCL_ERROR("[ValidatePullArg] invalid kernel argument");
        return CCU_E_PARA;
    }
    if ((fused && (arg->rankSize != 4 || arg->channelCount != 3)) ||
        (!fused && arg->rankSize == 4 && arg->channelCount != 3)) {
        HCCL_ERROR("[ValidatePullArg] invalid topology for kernel mode");
        return CCU_E_PARA;
    }

    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        const uint32_t remoteRank = arg->remoteRanks[channelIdx];
        if (remoteRank >= arg->rankSize || remoteRank == arg->rankId) {
            HCCL_ERROR("[ValidatePullArg] invalid remote rank");
            return CCU_E_PARA;
        }
        for (uint32_t previous = 0; previous < channelIdx; ++previous) {
            if (arg->remoteRanks[previous] == remoteRank) {
                HCCL_ERROR("[ValidatePullArg] duplicate remote rank");
                return CCU_E_PARA;
            }
        }
    }
    return CCU_SUCCESS;
}

CcuResult ValidateReduceArg(const CcuKernelArgReduceScatterReduce *arg)
{
    if (arg == nullptr || (arg->rankSize != 12 && arg->rankSize != MAX_RANK_SIZE) ||
        arg->rankId >= arg->rankSize || arg->dataType != HCCL_DATA_TYPE_FP32 ||
        arg->reduceOp != HCCL_REDUCE_SUM) {
        HCCL_ERROR("[ValidateReduceArg] invalid kernel argument");
        return CCU_E_PARA;
    }
    return CCU_SUCCESS;
}

template <typename T> CcuResult PrepareRemoteInputs(T &ctx)
{
    ctx.remoteInputAddr.resize(ctx.arg->channelCount);
    ctx.remoteInputToken.resize(ctx.arg->channelCount);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        ctx.remoteInputAddr[channelIdx] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], INPUT_XN_ID);
        ctx.remoteInputToken[channelIdx] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], TOKEN_XN_ID);
    }
    return CCU_SUCCESS;
}

CcuResult LoadPullArgs(PullContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.inputAddr, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratchAddr, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratchToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.rankSliceBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.processedBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceBytes, argId++));
    return CCU_SUCCESS;
}

CcuResult LoadReduceArgs(ReduceContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.inputAddr, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputAddr, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratchAddr, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratchToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.rankSliceBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.processedBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceBytes, argId++));
    return CCU_SUCCESS;
}

CcuResult LoadFusedArgs(FusedContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.inputAddr, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputAddr, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratchAddr, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratchToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.rankSliceBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.processedBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceBytes, argId++));
    return CCU_SUCCESS;
}

CcuResult LoadStripeArgs(StripeContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.inputAddr, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputAddr, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratchAddr, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratchToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.rankSliceBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.processedBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.primaryStripeBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.secondaryStripeBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.primaryTailStripeBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.secondaryTailStripeBytes, argId++));
    return CCU_SUCCESS;
}

CcuResult LoadGroupFlatArgs(GroupFlatContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.inputAddr, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputAddr, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratchAddr, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratchToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.rankSliceBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.processedBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.tileBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.loopCounter, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.lastTileBytes, argId++));
    return CCU_SUCCESS;
}

CcuResult LoadBatchReduceArgs(ReduceContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.inputAddr, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputAddr, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratchAddr, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratchToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.rankSliceBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.processedBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.tileBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.loopCounter, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.lastTileBytes, argId++));
    return CCU_SUCCESS;
}

CcuResult LoadDirectMergeArgs(ReduceContext &ctx)
{
    CCU_CHK_RET(LoadReduceArgs(ctx));
    uint32_t argId = 9;
    CCU_CHK_RET(ccu::LoadArg(ctx.mergeFirstBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.mergeSecondBytes, argId++));
    return CCU_SUCCESS;
}

template <typename T> CcuResult PublishInput(T &ctx)
{
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx], ctx.inputAddr,
            INPUT_XN_ID, CHANNEL_NOTIFY_IDX, ctx.arg->addrMask));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx], ctx.inputToken,
            TOKEN_XN_ID, CHANNEL_NOTIFY_IDX, ctx.arg->tokenMask));
    }
    return CCU_SUCCESS;
}

template <typename T> CcuResult WaitInput(T &ctx)
{
    const uint16_t readyMask = ctx.arg->addrMask | ctx.arg->tokenMask;
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        CCU_CHK_RET(ccu::NotifyWait(
            ctx.arg->channels[channelIdx], CHANNEL_NOTIFY_IDX, readyMask));
    }
    return CCU_SUCCESS;
}

template <typename T> CcuResult ExchangeInput(T &ctx)
{
    CCU_CHK_RET(PublishInput(ctx));
    CCU_CHK_RET(WaitInput(ctx));
    return CCU_SUCCESS;
}

template <typename T> CcuResult StartRemoteReads(T &ctx)
{
    ccu::LocalAddr scratch;
    scratch.addr = ctx.scratchAddr;
    scratch.token = ctx.scratchToken;
    uint32_t scratchPiece = 0;
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        ccu::RemoteAddr remoteInput;
        remoteInput.addr = ctx.remoteInputAddr[channelIdx];
        remoteInput.addr += ctx.inputOffset;
        remoteInput.token = ctx.remoteInputToken[channelIdx];

        const uint32_t targetScratchPiece = ctx.arg->remoteRanks[channelIdx];
        while (scratchPiece < targetScratchPiece) {
            scratch.addr += ctx.sliceBytes;
            ++scratchPiece;
        }

        const uint16_t eventMask = static_cast<uint16_t>(uint32_t{1} << channelIdx);
        CCU_CHK_RET(ccu::Read(ctx.arg->channels[channelIdx], scratch, remoteInput,
            ctx.sliceBytes, ctx.transferEvent, eventMask));
    }
    return CCU_SUCCESS;
}

template <typename T> CcuResult PrepareInputOffset(T &ctx)
{
    ctx.inputOffset = ctx.processedBytes;
    for (uint32_t rank = 0; rank < ctx.arg->rankId; ++rank) {
        ctx.inputOffset += ctx.rankSliceBytes;
    }
    return CCU_SUCCESS;
}

template <typename T> CcuResult WaitRemoteReads(T &ctx, uint16_t extraMask = 0)
{
    const uint16_t remoteMask =
        static_cast<uint16_t>((uint32_t{1} << ctx.arg->channelCount) - 1);
    const uint16_t allMask = static_cast<uint16_t>(remoteMask | extraMask);
    CCU_CHK_RET(ccu::EventWait(ctx.transferEvent, allMask));
    return CCU_SUCCESS;
}

template <typename T> CcuResult StartOwnContribution(T &ctx)
{
    if (!ctx.arg->prepareOwnContribution) {
        return CCU_SUCCESS;
    }

    ccu::LocalAddr ownContribution;
    ownContribution.addr = ctx.inputAddr;
    ownContribution.addr += ctx.inputOffset;
    ownContribution.token = ctx.inputToken;

    ccu::LocalAddr ownScratch;
    ownScratch.addr = ctx.scratchAddr;
    for (uint32_t rank = 0; rank < ctx.arg->rankId; ++rank) {
        ownScratch.addr += ctx.sliceBytes;
    }
    ownScratch.token = ctx.scratchToken;
    CCU_CHK_RET(ccu::LocalCopy(
        ownScratch, ownContribution, ctx.sliceBytes, ctx.transferEvent, OWN_COPY_MASK));
    return CCU_SUCCESS;
}

CcuResult StartStripeOwnCopy(StripeContext &ctx)
{
    ctx.ownContribution.addr = ctx.inputAddr;
    ctx.ownContribution.addr += ctx.inputOffset;
    ctx.ownContribution.token = ctx.inputToken;

    ctx.partial.addr = ctx.partialBaseAddr;
    ctx.partial.token = ctx.outputToken;
    CCU_CHK_RET(ccu::LocalCopy(
        ctx.partial, ctx.ownContribution, ctx.groupBytes, ctx.transferEvent, OWN_COPY_MASK));
    return CCU_SUCCESS;
}

CcuResult TransferStripe(StripeContext &ctx, uint32_t channelIdx,
    uint32_t stripeIndex, bool reduce)
{
    const ccu::Variable &stripeBytes = ctx.arg->groupIndex == 0 ?
        ctx.primaryStripeBytes : ctx.secondaryStripeBytes;
    const ccu::Variable &tailStripeBytes = ctx.arg->groupIndex == 0 ?
        ctx.primaryTailStripeBytes : ctx.secondaryTailStripeBytes;
    const ccu::Variable &stripeOffset = ctx.stripeOffsets[stripeIndex];

    if (stripeIndex + 1 == ctx.arg->channelCount) {
        ctx.transferBytes = tailStripeBytes;
    } else {
        ctx.transferBytes = stripeBytes;
    }

    ctx.remoteInput.addr = ctx.remoteInputAddr[channelIdx];
    ctx.remoteInput.addr += ctx.inputOffset;
    ctx.remoteInput.addr += stripeOffset;
    ctx.remoteInput.token = ctx.remoteInputToken[channelIdx];

    ctx.partial.addr = ctx.partialBaseAddr;
    ctx.partial.addr += stripeOffset;
    ctx.partial.token = ctx.arg->writeOutput ? ctx.outputToken : ctx.scratchToken;

    const uint16_t eventMask = static_cast<uint16_t>(uint32_t{1} << channelIdx);
    if (reduce) {
        CCU_CHK_RET(ccu::ReadReduce(ctx.arg->channels[channelIdx], ctx.partial, ctx.remoteInput,
            ctx.transferBytes, ctx.arg->dataType, ctx.arg->reduceOp,
            ctx.transferEvent, eventMask));
    } else {
        CCU_CHK_RET(ccu::Read(ctx.arg->channels[channelIdx], ctx.partial, ctx.remoteInput,
            ctx.transferBytes, ctx.transferEvent, eventMask));
    }
    return CCU_SUCCESS;
}

CcuResult PrepareStripeTransfers(StripeContext &ctx)
{
    CCU_CHK_RET(PrepareInputOffset(ctx));
    ctx.groupOffset = 0;
    ctx.groupBytes = ctx.sliceBytes;
    if (ctx.arg->rankSize == 4 && !ctx.arg->finalBarrier) {
        ccu::Variable primaryGroupBytes;
        primaryGroupBytes = ctx.primaryTailStripeBytes;
        for (uint32_t stripe = 1; stripe < ctx.arg->channelCount; ++stripe) {
            primaryGroupBytes += ctx.primaryStripeBytes;
        }
        if (ctx.arg->groupIndex == 0) {
            ctx.groupBytes = primaryGroupBytes;
        } else {
            ctx.groupOffset = primaryGroupBytes;
            ctx.groupBytes = ctx.secondaryTailStripeBytes;
            for (uint32_t stripe = 1; stripe < ctx.arg->channelCount; ++stripe) {
                ctx.groupBytes += ctx.secondaryStripeBytes;
            }
        }
        ctx.inputOffset += ctx.groupOffset;
    }

    ctx.partialBaseAddr = ctx.arg->writeOutput ? ctx.outputAddr : ctx.scratchAddr;
    if (ctx.arg->writeOutput) {
        ctx.partialBaseAddr += ctx.processedBytes;
        ctx.partialBaseAddr += ctx.groupOffset;
    }

    const ccu::Variable &stripeBytes = ctx.arg->groupIndex == 0 ?
        ctx.primaryStripeBytes : ctx.secondaryStripeBytes;
    ctx.stripeOffsets.resize(ctx.arg->channelCount);
    ctx.stripeOffsets[0] = 0;
    for (uint32_t stripe = 1; stripe < ctx.arg->channelCount; ++stripe) {
        ctx.stripeOffsets[stripe] = ctx.stripeOffsets[stripe - 1];
        ctx.stripeOffsets[stripe] += stripeBytes;
    }
    return CCU_SUCCESS;
}

CcuResult RunStripedGroup(StripeContext &ctx)
{
    CCU_CHK_RET(PrepareStripeTransfers(ctx));
    if (ctx.arg->includeOwnContribution) {
        CCU_CHK_RET(StartStripeOwnCopy(ctx));
    }
    CCU_CHK_RET(ExchangeInput(ctx));
    if (ctx.arg->includeOwnContribution) {
        CCU_CHK_RET(ccu::EventWait(ctx.transferEvent, OWN_COPY_MASK));
    }

    const uint32_t channelCount = ctx.arg->channelCount;
    const uint16_t roundMask =
        static_cast<uint16_t>((uint32_t{1} << channelCount) - 1);
    for (uint32_t step = 0; step < channelCount; ++step) {
        for (uint32_t channelIdx = 0; channelIdx < channelCount; ++channelIdx) {
            const uint32_t stripeIndex = (channelIdx + step) % channelCount;
            const bool reduce = ctx.arg->includeOwnContribution || step != 0;
            CCU_CHK_RET(TransferStripe(ctx, channelIdx, stripeIndex, reduce));
        }
        CCU_CHK_RET(ccu::EventWait(ctx.transferEvent, roundMask));
    }
    return CCU_SUCCESS;
}

CcuResult PrepareGroupScratch(GroupFlatContext &ctx)
{
    ctx.groupScratchAddr = ctx.scratchAddr;
    if (ctx.arg->groupIndex != 0) {
        const uint32_t scratchPiece = ctx.arg->rankSize - ctx.arg->channelCount;
        for (uint32_t piece = 0; piece < scratchPiece; ++piece) {
            ctx.groupScratchAddr += ctx.tileBytes;
        }
    }
    return CCU_SUCCESS;
}

CcuResult StartGroupOwnContribution(GroupFlatContext &ctx)
{
    if (!ctx.arg->includeOwnContribution) {
        return CCU_SUCCESS;
    }

    ccu::LocalAddr ownContribution;
    ownContribution.addr = ctx.inputAddr;
    ownContribution.addr += ctx.inputOffset;
    ownContribution.token = ctx.inputToken;

    ccu::LocalAddr ownScratch;
    if (ctx.arg->rankSize == MAX_RANK_SIZE) {
        ownScratch.addr = ctx.outputAddr;
        ownScratch.addr += ctx.processedBytes;
        ownScratch.token = ctx.outputToken;
    } else {
        ownScratch.addr = ctx.groupScratchAddr;
        ownScratch.token = ctx.scratchToken;
    }
    CCU_CHK_RET(ccu::LocalCopy(
        ownScratch, ownContribution, ctx.sliceBytes, ctx.transferEvent, OWN_COPY_MASK));
    return CCU_SUCCESS;
}

CcuResult StartOneGroupRead(GroupFlatContext &ctx, uint32_t channelIdx,
    ccu::LocalAddr &localDestination)
{
    ccu::RemoteAddr remoteInput;
    remoteInput.addr = ctx.remoteInputAddr[channelIdx];
    remoteInput.addr += ctx.inputOffset;
    remoteInput.token = ctx.remoteInputToken[channelIdx];

    const uint16_t eventMask = static_cast<uint16_t>(uint32_t{1} << channelIdx);
    CCU_CHK_RET(ccu::Read(ctx.arg->channels[channelIdx], localDestination, remoteInput,
        ctx.sliceBytes, ctx.transferEvent, eventMask));
    return CCU_SUCCESS;
}

CcuResult StartGroupReads(GroupFlatContext &ctx)
{
    uint32_t firstScratchChannel = 0;
    if (ctx.arg->rankSize == MAX_RANK_SIZE && !ctx.arg->includeOwnContribution) {
        ccu::LocalAddr retained;
        retained.addr = ctx.retainedPartialAddr;
        retained.token = ctx.scratchToken;
        CCU_CHK_RET(StartOneGroupRead(ctx, 0, retained));
        firstScratchChannel = 1;
    }

    ccu::LocalAddr localScratch;
    localScratch.addr = ctx.groupScratchAddr;
    if (ctx.arg->rankSize != MAX_RANK_SIZE && ctx.arg->includeOwnContribution) {
        localScratch.addr += ctx.sliceBytes;
    }
    localScratch.token = ctx.scratchToken;

    for (uint32_t channelIdx = firstScratchChannel;
         channelIdx < ctx.arg->channelCount; ++channelIdx) {
        CCU_CHK_RET(StartOneGroupRead(ctx, channelIdx, localScratch));
        if (channelIdx + 1 < ctx.arg->channelCount) {
            localScratch.addr += ctx.sliceBytes;
        }
    }
    return CCU_SUCCESS;
}

CcuResult ReduceGroupScratch(GroupFlatContext &ctx)
{
    ccu::LocalAddr scratch;
    scratch.addr = ctx.groupScratchAddr;
    scratch.token = ctx.scratchToken;

    uint32_t remainPieces = 0;
    if (ctx.arg->rankSize == MAX_RANK_SIZE) {
        remainPieces = ctx.arg->channelCount -
            (ctx.arg->includeOwnContribution ? 0U : 1U);
    } else {
        remainPieces = ctx.arg->channelCount +
            (ctx.arg->includeOwnContribution ? 1U : 0U);
    }
    while (remainPieces > 1) {
        const uint32_t reducePieces = remainPieces / 2;
        ccu::Variable reduceBytes;
        reduceBytes = ctx.sliceBytes;
        for (uint32_t piece = 1; piece < reducePieces; ++piece) {
            reduceBytes += ctx.sliceBytes;
        }
        ccu::LocalAddr source;
        source.addr = scratch.addr;
        source.addr += reduceBytes;
        if (remainPieces % 2 != 0) {
            source.addr += ctx.sliceBytes;
        }
        source.token = scratch.token;
        CCU_CHK_RET(ccu::LocalReduce(
            scratch, source, reduceBytes, ctx.arg->dataType, ctx.arg->reduceOp, ctx.reduceEvent));
        CCU_CHK_RET(ccu::EventWait(ctx.reduceEvent));
        remainPieces -= reducePieces;
    }
    return CCU_SUCCESS;
}

CcuResult CopyGroupScratchToOutput(GroupFlatContext &ctx)
{
    ccu::LocalAddr output;
    output.addr = ctx.outputAddr;
    output.addr += ctx.processedBytes;
    output.token = ctx.outputToken;

    ccu::LocalAddr scratch;
    scratch.addr = ctx.groupScratchAddr;
    scratch.token = ctx.scratchToken;
    CCU_CHK_RET(ccu::LocalCopy(
        output, scratch, ctx.sliceBytes, ctx.reduceEvent));
    CCU_CHK_RET(ccu::EventWait(ctx.reduceEvent));
    return CCU_SUCCESS;
}

CcuResult PrepareRetainedPartials(GroupFlatContext &ctx)
{
    ctx.retainedPartialAddr = ctx.scratchAddr;
    for (uint32_t piece = 0; piece < ctx.arg->rankSize; ++piece) {
        ctx.retainedPartialAddr += ctx.tileBytes;
    }
    return CCU_SUCCESS;
}

CcuResult CopyGroupScratchToRetention(GroupFlatContext &ctx)
{
    ccu::LocalAddr retained;
    retained.addr = ctx.retainedPartialAddr;
    retained.token = ctx.scratchToken;

    ccu::LocalAddr scratch;
    scratch.addr = ctx.groupScratchAddr;
    scratch.token = ctx.scratchToken;
    CCU_CHK_RET(ccu::LocalCopy(
        retained, scratch, ctx.sliceBytes, ctx.reduceEvent));
    CCU_CHK_RET(ccu::EventWait(ctx.reduceEvent));
    return CCU_SUCCESS;
}

CcuResult AddGroupScratchToOutput(GroupFlatContext &ctx)
{
    ccu::LocalAddr output;
    output.addr = ctx.outputAddr;
    output.addr += ctx.processedBytes;
    output.token = ctx.outputToken;

    ccu::LocalAddr scratch;
    scratch.addr = ctx.groupScratchAddr;
    scratch.token = ctx.scratchToken;
    CCU_CHK_RET(ccu::LocalReduce(
        output, scratch, ctx.sliceBytes, ctx.arg->dataType, ctx.arg->reduceOp, ctx.reduceEvent));
    CCU_CHK_RET(ccu::EventWait(ctx.reduceEvent));
    return CCU_SUCCESS;
}

CcuResult AddGroupScratchToRetention(GroupFlatContext &ctx)
{
    ccu::LocalAddr retained;
    retained.addr = ctx.retainedPartialAddr;
    retained.token = ctx.scratchToken;

    ccu::LocalAddr scratch;
    scratch.addr = ctx.groupScratchAddr;
    scratch.token = ctx.scratchToken;
    CCU_CHK_RET(ccu::LocalReduce(
        retained, scratch, ctx.sliceBytes, ctx.arg->dataType, ctx.arg->reduceOp, ctx.reduceEvent));
    CCU_CHK_RET(ccu::EventWait(ctx.reduceEvent));
    return CCU_SUCCESS;
}

CcuResult RunGroupFlatTile(GroupFlatContext &ctx)
{
    CCU_CHK_RET(PrepareInputOffset(ctx));
    CCU_CHK_RET(StartGroupOwnContribution(ctx));
    CCU_CHK_RET(StartGroupReads(ctx));
    const uint16_t ownMask = ctx.arg->includeOwnContribution ? OWN_COPY_MASK : 0;
    CCU_CHK_RET(WaitRemoteReads(ctx, ownMask));
    CCU_CHK_RET(ReduceGroupScratch(ctx));
    if (ctx.arg->rankSize == MAX_RANK_SIZE && ctx.arg->writeOutput) {
        CCU_CHK_RET(AddGroupScratchToOutput(ctx));
    } else if (ctx.arg->rankSize == MAX_RANK_SIZE) {
        CCU_CHK_RET(AddGroupScratchToRetention(ctx));
    } else if (ctx.arg->writeOutput) {
        CCU_CHK_RET(CopyGroupScratchToOutput(ctx));
    } else {
        CCU_CHK_RET(CopyGroupScratchToRetention(ctx));
    }
    return CCU_SUCCESS;
}

CcuResult RunGroupFlatBatch(GroupFlatContext &ctx)
{
    CCU_CHK_RET(PublishInput(ctx));
    CCU_CHK_RET(PrepareGroupScratch(ctx));
    if (!ctx.arg->writeOutput) {
        CCU_CHK_RET(PrepareRetainedPartials(ctx));
    }
    CCU_CHK_RET(WaitInput(ctx));
    ccu::Variable counterStep;
    counterStep = 1;
    CCU_DO {
        ctx.loopCounter = ctx.loopCounter + counterStep;
        ctx.sliceBytes = ctx.tileBytes;
        CCU_IF(ctx.loopCounter == UINT64_MAX) {
            ctx.sliceBytes = ctx.lastTileBytes;
        }
        CCU_CHK_RET(RunGroupFlatTile(ctx));
        ctx.processedBytes = ctx.processedBytes + ctx.sliceBytes;
        if (!ctx.arg->writeOutput) {
            ctx.retainedPartialAddr = ctx.retainedPartialAddr + ctx.tileBytes;
        }
    } CCU_WHILE(ctx.loopCounter != UINT64_MAX);
    return CCU_SUCCESS;
}

template <typename T> CcuResult ReduceScratch(T &ctx, uint32_t remainPieces)
{
    ccu::LocalAddr scratch;
    scratch.addr = ctx.scratchAddr;
    scratch.token = ctx.scratchToken;

    while (remainPieces > 1) {
        const uint32_t reducePieces = remainPieces / 2;
        ccu::Variable reduceBytes;
        reduceBytes = ctx.sliceBytes;
        for (uint32_t piece = 1; piece < reducePieces; ++piece) {
            reduceBytes += ctx.sliceBytes;
        }
        ccu::LocalAddr source;
        source.addr = scratch.addr;
        source.addr += reduceBytes;
        if (remainPieces % 2 != 0) {
            source.addr += ctx.sliceBytes;
        }
        source.token = scratch.token;
        CCU_CHK_RET(ccu::LocalReduce(
            scratch, source, reduceBytes, ctx.arg->dataType, ctx.arg->reduceOp, ctx.reduceEvent));
        CCU_CHK_RET(ccu::EventWait(ctx.reduceEvent));
        remainPieces -= reducePieces;
    }
    return CCU_SUCCESS;
}

CcuResult PrepareBatchReduce(ReduceContext &ctx)
{
    ctx.retainedPartialAddr = ctx.scratchAddr;
    for (uint32_t piece = 0; piece < ctx.arg->rankSize; ++piece) {
        ctx.retainedPartialAddr += ctx.tileBytes;
    }
    return CCU_SUCCESS;
}

CcuResult AddRetainedPartialToOutput(ReduceContext &ctx)
{
    ccu::LocalAddr output;
    output.addr = ctx.outputAddr;
    output.addr += ctx.processedBytes;
    output.token = ctx.outputToken;

    ccu::LocalAddr retained;
    retained.addr = ctx.retainedPartialAddr;
    retained.token = ctx.scratchToken;
    CCU_CHK_RET(ccu::LocalReduce(
        output, retained, ctx.sliceBytes, ctx.arg->dataType, ctx.arg->reduceOp, ctx.reduceEvent));
    CCU_CHK_RET(ccu::EventWait(ctx.reduceEvent));
    return CCU_SUCCESS;
}

CcuResult RunBatchReduce(ReduceContext &ctx)
{
    CCU_CHK_RET(PrepareBatchReduce(ctx));
    ccu::Variable counterStep;
    counterStep = 1;
    CCU_DO {
        ctx.loopCounter = ctx.loopCounter + counterStep;
        ctx.sliceBytes = ctx.tileBytes;
        CCU_IF(ctx.loopCounter == UINT64_MAX) {
            ctx.sliceBytes = ctx.lastTileBytes;
        }
        CCU_CHK_RET(AddRetainedPartialToOutput(ctx));
        ctx.processedBytes = ctx.processedBytes + ctx.sliceBytes;
        ctx.retainedPartialAddr = ctx.retainedPartialAddr + ctx.tileBytes;
    } CCU_WHILE(ctx.loopCounter != UINT64_MAX);
    return CCU_SUCCESS;
}

CcuResult MergeDirectStripePartials(ReduceContext &ctx)
{
    ccu::LocalAddr output;
    output.addr = ctx.outputAddr;
    output.addr += ctx.processedBytes;
    output.token = ctx.outputToken;

    ccu::LocalAddr secondaryPartial;
    secondaryPartial.addr = ctx.scratchAddr;
    secondaryPartial.token = ctx.scratchToken;
    CCU_CHK_RET(ccu::LocalReduce(output, secondaryPartial, ctx.mergeFirstBytes,
        ctx.arg->dataType, ctx.arg->reduceOp, ctx.reduceEvent));
    CCU_CHK_RET(ccu::EventWait(ctx.reduceEvent));
    CCU_IF(ctx.mergeSecondBytes != 0) {
        output.addr += ctx.mergeFirstBytes;
        secondaryPartial.addr += ctx.mergeFirstBytes;
        CCU_CHK_RET(ccu::LocalReduce(output, secondaryPartial, ctx.mergeSecondBytes,
            ctx.arg->dataType, ctx.arg->reduceOp, ctx.reduceEvent));
        CCU_CHK_RET(ccu::EventWait(ctx.reduceEvent));
    }
    return CCU_SUCCESS;
}

template <typename T> CcuResult CopyScratchToOutput(T &ctx)
{
    ccu::LocalAddr output;
    output.addr = ctx.outputAddr;
    output.addr += ctx.processedBytes;
    output.token = ctx.outputToken;

    ccu::LocalAddr scratch;
    scratch.addr = ctx.scratchAddr;
    scratch.token = ctx.scratchToken;
    CCU_CHK_RET(ccu::LocalCopy(
        output, scratch, ctx.sliceBytes, ctx.reduceEvent));
    CCU_CHK_RET(ccu::EventWait(ctx.reduceEvent));
    return CCU_SUCCESS;
}
} // namespace

CcuResult CcuReduceScatterPullKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceScatterPull *>(arg);
    CCU_CHK_RET(ValidatePullArg(kernelArg, false));

    PullContext ctx;
    ctx.arg = kernelArg;
    CCU_CHK_RET(PrepareRemoteInputs(ctx));
    CCU_CHK_RET(LoadPullArgs(ctx));
    CCU_CHK_RET(PrepareInputOffset(ctx));
    CCU_CHK_RET(StartOwnContribution(ctx));
    CCU_CHK_RET(ExchangeInput(ctx));
    CCU_CHK_RET(StartRemoteReads(ctx));
    const uint16_t ownMask = kernelArg->prepareOwnContribution ? OWN_COPY_MASK : 0;
    CCU_CHK_RET(WaitRemoteReads(ctx, ownMask));
    if (kernelArg->rankSize == 12 || kernelArg->rankSize == MAX_RANK_SIZE) {
        CCU_CHK_RET(Barrier(*kernelArg, FINAL_BARRIER_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterReduceKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceScatterReduce *>(arg);
    CCU_CHK_RET(ValidateReduceArg(kernelArg));

    ReduceContext ctx;
    ctx.arg = kernelArg;
    CCU_CHK_RET(LoadReduceArgs(ctx));

    CCU_CHK_RET(ReduceScratch(ctx, ctx.arg->rankSize));
    CCU_CHK_RET(CopyScratchToOutput(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterReleaseKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceScatterPull *>(arg);
    CCU_CHK_RET(ValidatePullArg(kernelArg, false));
    CCU_CHK_RET(Barrier(*kernelArg, FINAL_BARRIER_MASK));
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterFusedKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceScatterPull *>(arg);
    CCU_CHK_RET(ValidatePullArg(kernelArg, true));

    FusedContext ctx;
    ctx.arg = kernelArg;
    CCU_CHK_RET(PrepareRemoteInputs(ctx));
    CCU_CHK_RET(LoadFusedArgs(ctx));
    CCU_CHK_RET(PrepareInputOffset(ctx));
    CCU_CHK_RET(StartOwnContribution(ctx));
    CCU_CHK_RET(ExchangeInput(ctx));
    CCU_CHK_RET(StartRemoteReads(ctx));
    CCU_CHK_RET(WaitRemoteReads(ctx, OWN_COPY_MASK));
    CCU_CHK_RET(ReduceScratch(ctx, ctx.arg->rankSize));
    CCU_CHK_RET(CopyScratchToOutput(ctx));
    CCU_CHK_RET(Barrier(*ctx.arg, FINAL_BARRIER_MASK));
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterStripeKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceScatterStripe *>(arg);
    const bool fused = kernelArg != nullptr && kernelArg->rankSize == 4 &&
        kernelArg->finalBarrier;
    CCU_CHK_RET(ValidatePullArg(kernelArg, fused));
    const bool validFused = fused && kernelArg->groupIndex == 0 &&
        kernelArg->includeOwnContribution && kernelArg->writeOutput && kernelArg->finalBarrier;
    const bool validRank4Split = !fused && kernelArg->rankSize == 4 &&
        kernelArg->groupIndex < 2 && kernelArg->includeOwnContribution &&
        kernelArg->writeOutput && !kernelArg->finalBarrier;
    const bool validSplit = !fused && kernelArg->rankSize != 4 && kernelArg->groupIndex < 2 &&
        kernelArg->includeOwnContribution == (kernelArg->groupIndex == 0) &&
        kernelArg->writeOutput == (kernelArg->groupIndex == 0) && !kernelArg->finalBarrier;
    if (!validFused && !validRank4Split && !validSplit) {
        HCCL_ERROR("[CcuReduceScatterStripeKernel] invalid stripe role");
        return CCU_E_PARA;
    }

    if (fused || kernelArg->rankSize == 4) {
        StripeContext ctx;
        ctx.arg = kernelArg;
        CCU_CHK_RET(PrepareRemoteInputs(ctx));
        CCU_CHK_RET(LoadStripeArgs(ctx));
        CCU_CHK_RET(RunStripedGroup(ctx));
        if (kernelArg->finalBarrier) {
            CCU_CHK_RET(Barrier(*kernelArg, FINAL_BARRIER_MASK));
        }
        return CCU_SUCCESS;
    }

    GroupFlatContext ctx;
    ctx.arg = kernelArg;
    CCU_CHK_RET(PrepareRemoteInputs(ctx));
    CCU_CHK_RET(LoadGroupFlatArgs(ctx));
    CCU_CHK_RET(RunGroupFlatBatch(ctx));
    CCU_CHK_RET(Barrier(*kernelArg, FINAL_BARRIER_MASK));
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterStripeReduceKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceScatterReduce *>(arg);
    CCU_CHK_RET(ValidateReduceArg(kernelArg));

    ReduceContext ctx;
    ctx.arg = kernelArg;
    CCU_CHK_RET(LoadBatchReduceArgs(ctx));
    CCU_CHK_RET(RunBatchReduce(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterDirectStripeKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceScatterStripe *>(arg);
    CCU_CHK_RET(ValidatePullArg(kernelArg, false));
    const bool validRankSize = kernelArg->rankSize == 12 || kernelArg->rankSize == MAX_RANK_SIZE;
    const bool validGroup = validRankSize && kernelArg->directStripe && kernelArg->groupIndex < 2 &&
        kernelArg->includeOwnContribution == (kernelArg->groupIndex == 0) &&
        kernelArg->writeOutput == (kernelArg->groupIndex == 0) &&
        !kernelArg->finalBarrier;
    if (!validGroup) {
        HCCL_ERROR("[CcuReduceScatterDirectStripeKernel] invalid direct-stripe role");
        return CCU_E_PARA;
    }

    StripeContext ctx;
    ctx.arg = kernelArg;
    CCU_CHK_RET(PrepareRemoteInputs(ctx));
    CCU_CHK_RET(LoadStripeArgs(ctx));
    CCU_CHK_RET(RunStripedGroup(ctx));
    CCU_CHK_RET(Barrier(*kernelArg, FINAL_BARRIER_MASK));
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterDirectMergeKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceScatterReduce *>(arg);
    CCU_CHK_RET(ValidateReduceArg(kernelArg));
    if (kernelArg->rankSize != 12 && kernelArg->rankSize != MAX_RANK_SIZE) {
        HCCL_ERROR("[CcuReduceScatterDirectMergeKernel] invalid rank size");
        return CCU_E_PARA;
    }

    ReduceContext ctx;
    ctx.arg = kernelArg;
    CCU_CHK_RET(LoadDirectMergeArgs(ctx));
    CCU_CHK_RET(MergeDirectStripePartials(ctx));
    return CCU_SUCCESS;
}
} // namespace ops_hccl
