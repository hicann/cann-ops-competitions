/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel.h"

#ifndef CCU_CHK_RET
#define CCU_CHK_RET(call) \
    do { \
        CcuResult ccuRet = (call); \
        if (ccuRet != CCU_SUCCESS) { \
            return ccuRet; \
        } \
    } while (0)
#endif

namespace ops_hccl {
namespace {
constexpr uint32_t OUTPUT_XN_ID = 1;
constexpr uint32_t TOKEN_XN_ID = 2;
constexpr uint32_t CHANNEL_NOTIFY_INDEX = 0;
constexpr uint32_t POST_SYNC_ID = 3;
constexpr uint32_t NHR_STEP_SYNC_ID = 4;

constexpr uint64_t SetBits(uint16_t end)
{
    return (uint64_t{1} << (end + 1)) - uint64_t{1};
}

uint64_t GetLoopParam(uint64_t loopCtxId, uint64_t gsaOffset, uint64_t loopIterNum)
{
    constexpr uint16_t CTX_END_BIT = 8;
    constexpr uint16_t CTX_SHIFT = 45;
    constexpr uint16_t GSA_END_BIT = 32;
    constexpr uint16_t GSA_SHIFT = 13;
    constexpr uint16_t LOOP_END_BIT = 13;
    return ((loopCtxId & SetBits(CTX_END_BIT)) << CTX_SHIFT) |
        ((gsaOffset & SetBits(GSA_END_BIT)) << GSA_SHIFT) | (loopIterNum & SetBits(LOOP_END_BIT));
}

uint64_t GetParallelParam(uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum)
{
    constexpr uint16_t REPEAT_END_BIT = 7;
    constexpr uint16_t REPEAT_SHIFT = 55;
    constexpr uint16_t REPEAT_LOOP_END_BIT = 7;
    constexpr uint16_t REPEAT_LOOP_SHIFT = 48;
    constexpr uint16_t TOTAL_LOOP_END_BIT = 7;
    constexpr uint16_t TOTAL_LOOP_SHIFT = 41;
    return ((repeatNum & SetBits(REPEAT_END_BIT)) << REPEAT_SHIFT) |
        ((repeatLoopIndex & SetBits(REPEAT_LOOP_END_BIT)) << REPEAT_LOOP_SHIFT) |
        ((totalLoopNum & SetBits(TOTAL_LOOP_END_BIT)) << TOTAL_LOOP_SHIFT);
}

uint64_t GetOffsetParam(uint64_t gsaOffset, uint64_t msOffset, uint64_t ckeOffset)
{
    constexpr uint16_t GSA_END_BIT = 32;
    constexpr uint16_t GSA_SHIFT = 21;
    constexpr uint16_t MS_END_BIT = 11;
    constexpr uint16_t MS_SHIFT = 10;
    constexpr uint16_t CKE_END_BIT = 10;
    return ((gsaOffset & SetBits(GSA_END_BIT)) << GSA_SHIFT) |
        ((msOffset & SetBits(MS_END_BIT)) << MS_SHIFT) | (ckeOffset & SetBits(CKE_END_BIT));
}

void InitGroupCopyResources(AllGatherContext &ctx)
{
    if (!ctx.resourceAllocated) {
        ctx.copyConfig.msInterleave = CCU_MS_INTERLEAVE;
        ctx.copyConfig.loopCount = CCU_MS_LOCAL_COPY_LOOP_COUNT;
        ctx.copyConfig.memSlice = CCU_LOCAL_COPY_MS_PER_LOOP * CCU_MS_SIZE;
        ctx.copyResource.eventCount = ctx.copyConfig.loopCount;
        ctx.copyResource.completedEvent = ccu::Array<ccu::Event>(ctx.copyResource.eventCount);
        ctx.copyResource.bufCount = ctx.copyConfig.loopCount * ctx.copyConfig.msInterleave;
        ctx.copyResource.ccuBuf = ccu::Array<ccu::CcuBuffer>(ctx.copyResource.bufCount);
        ctx.resourceAllocated = true;
    }

    const std::string loopType = "localcopy";
    if (!ctx.IsLoopEntityRegistered(loopType)) {
        ctx.CreateLoopEntity(loopType);
        auto &entity = ctx.loopMap[loopType];
        for (uint32_t index = 0; index < 2; ++index) {
            const uint32_t bufBase = index * ctx.copyConfig.msInterleave;
            const ccu::Event loopEvent = ctx.copyResource.completedEvent[index];
            entity.body[index].reset(new ccu::Func(
                [&ctx, index, bufBase, loopEvent]() {
                    ccu::LocalCopy(ctx.copyResource.ccuBuf[bufBase], ctx.copyLoopSrc[index],
                        ctx.copyLoopLength[index], loopEvent, 1);
                    ccu::EventWait(loopEvent, 1);
                    ccu::LocalCopy(ctx.copyLoopDst[index], ctx.copyResource.ccuBuf[bufBase],
                        ctx.copyLoopLength[index], loopEvent, 1);
                    ccu::EventWait(loopEvent, 1);
                }));
            entity.loops[index].reset(new ccu::Loop(entity.loopParam[index], *entity.body[index]));
        }
    }
}

CcuResult GroupCopy(AllGatherContext &ctx, ccu::LocalAddr dst, ccu::LocalAddr src, GroupOpSizeVars goSize)
{
    InitGroupCopyResources(ctx);
    auto &loops = ctx.loopMap["localcopy"];

    CCU_IF(goSize.addrOffset != 0)
    {
        ccu::Variable loopParam;
        loopParam = GetLoopParam(0, ctx.copyConfig.memSlice * ctx.copyConfig.loopCount, 0);
        loopParam += goSize.loopParam;
        ccu::Variable fixedSliceSize;
        fixedSliceSize = ctx.copyConfig.memSlice;

        ctx.copyLoopSrc[0].addr = src.addr;
        ctx.copyLoopSrc[0].token = src.token;
        ctx.copyLoopDst[0].addr = dst.addr;
        ctx.copyLoopDst[0].token = dst.token;
        ctx.copyLoopLength[0] = fixedSliceSize;
        loops.loopParam[0] = loopParam;

        ccu::Variable parallelConfig;
        parallelConfig = GetParallelParam(ctx.copyConfig.loopCount - 1, 0, 1);
        ccu::Variable offsetConfig;
        offsetConfig = GetOffsetParam(ctx.copyConfig.memSlice, ctx.copyConfig.msInterleave, 1);
        std::vector<ccu::Loop> groupLoops{*loops.loops[0]};
        ccu::LoopGroup group(parallelConfig, offsetConfig, ctx.copyConfig.loopCount, groupLoops);
    }

    CCU_IF(goSize.parallelParam != 0)
    {
        src.addr += goSize.addrOffset;
        dst.addr += goSize.addrOffset;
        ctx.copyLoopSrc[0].addr = src.addr;
        ctx.copyLoopSrc[0].token = src.token;
        ctx.copyLoopDst[0].addr = dst.addr;
        ctx.copyLoopDst[0].token = dst.token;
        ctx.copyLoopLength[0] = goSize.residual;

        src.addr += goSize.residual;
        dst.addr += goSize.residual;
        ccu::Variable fixedSliceSize;
        fixedSliceSize = ctx.copyConfig.memSlice;
        ctx.copyLoopSrc[1].addr = src.addr;
        ctx.copyLoopSrc[1].token = src.token;
        ctx.copyLoopDst[1].addr = dst.addr;
        ctx.copyLoopDst[1].token = dst.token;
        ctx.copyLoopLength[1] = fixedSliceSize;

        ccu::Variable loopConfig0;
        ccu::Variable loopConfig1;
        loopConfig0 = GetLoopParam(0, 0, 1);
        loopConfig1 = GetLoopParam(0, 0, 1);
        loops.loopParam[0] = loopConfig0;
        loops.loopParam[1] = loopConfig1;
        ccu::Variable offsetConfig;
        offsetConfig = GetOffsetParam(ctx.copyConfig.memSlice, ctx.copyConfig.msInterleave, 1);
        std::vector<ccu::Loop> groupLoops{*loops.loops[0], *loops.loops[1]};
        ccu::LoopGroup group(goSize.parallelParam, offsetConfig, ctx.copyConfig.loopCount, groupLoops);
    }
    return CCU_SUCCESS;
}

CcuResult InitResource(AllGatherContext &ctx)
{
    if (ctx.arg == nullptr || ctx.arg->rankSize < 2 || ctx.arg->rankSize > MAX_RANK_SIZE ||
        ctx.arg->rankId >= ctx.arg->rankSize || ctx.arg->channelCount == 0 ||
        ctx.arg->channelCount >= ctx.arg->rankSize) {
        return CCU_E_PARA;
    }
    if (ctx.arg->hierarchicalGrid) {
        const bool rankShapeValid = ctx.arg->physical8Plus4 ?
            ctx.arg->localRankCount + ctx.arg->interRankCount == ctx.arg->rankSize :
            ctx.arg->localRankCount * ctx.arg->interRankCount == ctx.arg->rankSize;
        if (ctx.arg->localRankCount < 2 || ctx.arg->interRankCount < 2 ||
            !rankShapeValid) {
            return CCU_E_PARA;
        }
        if (ctx.arg->role == AllGatherKernelRole::INTRA_SERVER) {
            if (ctx.arg->localPeerCount == 0 ||
                ctx.arg->localPeerCount != ctx.arg->channelCount ||
                ctx.arg->localPeerCount >= ctx.arg->localRankCount ||
                (!ctx.arg->purePhysicalGrid && !ctx.arg->physical8Plus4 &&
                    ctx.arg->localPeerCount + 1 != ctx.arg->localRankCount)) {
                return CCU_E_PARA;
            }
        } else if (ctx.arg->role == AllGatherKernelRole::INTER_SERVER) {
            if (ctx.arg->localPeerCount != 0 ||
                ctx.arg->channelCount >= ctx.arg->interRankCount ||
                (!ctx.arg->purePhysicalGrid && !ctx.arg->physical8Plus4 &&
                    ctx.arg->channelCount + 1 != ctx.arg->interRankCount)) {
                return CCU_E_PARA;
            }
        } else {
            return CCU_E_PARA;
        }
        if (ctx.arg->physical8Plus4 &&
            (ctx.arg->gatewaySourceCount > 2 ||
             (ctx.arg->hasGatewayTarget &&
              ctx.arg->gatewayTargetChannelIndex >= ctx.arg->channelCount))) {
            return CCU_E_PARA;
        }
        uint32_t phaseTwoOperationCount = 0;
        if (ctx.arg->role == AllGatherKernelRole::INTRA_SERVER) {
            phaseTwoOperationCount = ctx.arg->physical8Plus4 ?
                ctx.arg->localPeerCount * ctx.arg->gatewaySourceCount :
                ctx.arg->localPeerCount * ctx.arg->interRankCount;
        } else if (ctx.arg->role == AllGatherKernelRole::INTER_SERVER) {
            phaseTwoOperationCount = ctx.arg->physical8Plus4 ? 0 :
                ctx.arg->localRankCount * (ctx.arg->interRankCount - 1);
        } else {
            phaseTwoOperationCount = ctx.arg->localPeerCount * ctx.arg->interRankCount +
                (ctx.arg->channelCount - ctx.arg->localPeerCount) * ctx.arg->localRankCount;
        }
        if (phaseTwoOperationCount > 16) {
            return CCU_E_PARA;
        }
        if (ctx.arg->interRankCount == 2 &&
            ctx.arg->role == AllGatherKernelRole::INTER_SERVER &&
            (ctx.arg->pairedRank >= ctx.arg->rankSize ||
                ctx.arg->pairedChannelIndex >= ctx.arg->channelCount)) {
            return CCU_E_PARA;
        }
    }
    if (ctx.arg->useNhr) {
        if (ctx.arg->hierarchicalGrid || ctx.arg->rankSize != 4 ||
            ctx.arg->channelCount + 1 != ctx.arg->rankSize ||
            ctx.arg->nhrStepCount == 0 || ctx.arg->nhrStepCount > 4 ||
            ctx.arg->nhrTxOffsets[ctx.arg->nhrStepCount] > MAX_RANK_SIZE) {
            return CCU_E_PARA;
        }
        for (uint32_t step = 0; step < ctx.arg->nhrStepCount; ++step) {
            if (ctx.arg->nhrToRanks[step] >= ctx.arg->rankSize ||
                ctx.arg->nhrFromRanks[step] >= ctx.arg->rankSize ||
                ctx.arg->nhrToChannelIndices[step] >= ctx.arg->channelCount ||
                ctx.arg->nhrFromChannelIndices[step] >= ctx.arg->channelCount ||
                ctx.arg->nhrTxOffsets[step] >= ctx.arg->nhrTxOffsets[step + 1]) {
                return CCU_E_PARA;
            }
        }
    }

    ctx.output.resize(ctx.arg->rankSize);
    ctx.outputToken.resize(ctx.arg->rankSize);
    bool peerSeen[MAX_RANK_SIZE]{};
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        const uint32_t peerRank = ctx.arg->peerRanks[channelIndex];
        if (peerRank >= ctx.arg->rankSize || peerRank == ctx.arg->rankId || peerSeen[peerRank]) {
            return CCU_E_PARA;
        }
        peerSeen[peerRank] = true;
        ctx.output[peerRank] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIndex], OUTPUT_XN_ID);
        ctx.outputToken[peerRank] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIndex], TOKEN_XN_ID);
    }
    return CCU_SUCCESS;
}

CcuResult LoadArgs(AllGatherContext &ctx)
{
    uint32_t argIndex = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.output[ctx.arg->rankId], argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputToken, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputToken[ctx.arg->rankId], argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentRankOutputOffset, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceSize, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localCopyNeeded, argIndex++));
    if (!ctx.arg->latencyOptimizedBarrier) {
        CCU_CHK_RET(ccu::LoadArg(ctx.goSize.addrOffset, argIndex++));
        CCU_CHK_RET(ccu::LoadArg(ctx.goSize.loopParam, argIndex++));
        CCU_CHK_RET(ccu::LoadArg(ctx.goSize.parallelParam, argIndex++));
        CCU_CHK_RET(ccu::LoadArg(ctx.goSize.residual, argIndex++));
    }
    if (ctx.arg->hierarchicalGrid || ctx.arg->useNhr) {
        CCU_CHK_RET(ccu::LoadArg(ctx.dataSize, argIndex++));
    }
    if (ctx.arg->hierarchicalGrid) {
        CCU_CHK_RET(ccu::LoadArg(ctx.partOffset, argIndex++));
        CCU_CHK_RET(ccu::LoadArg(ctx.otherPartOffset, argIndex++));
        CCU_CHK_RET(ccu::LoadArg(ctx.otherPartSize, argIndex++));
        CCU_CHK_RET(ccu::LoadArg(ctx.executionMode, argIndex++));
    }
    return CCU_SUCCESS;
}

CcuResult PairedPreSync(AllGatherContext &ctx)
{
    const ChannelHandle pairedChannel = ctx.arg->channels[ctx.arg->pairedChannelIndex];
    CCU_CHK_RET(ccu::WriteVariableWithNotify(pairedChannel,
        ctx.output[ctx.arg->rankId], OUTPUT_XN_ID, CHANNEL_NOTIFY_INDEX, 1U << OUTPUT_XN_ID));
    CCU_CHK_RET(ccu::WriteVariableWithNotify(pairedChannel,
        ctx.outputToken[ctx.arg->rankId], TOKEN_XN_ID, CHANNEL_NOTIFY_INDEX, 1U << TOKEN_XN_ID));
    constexpr uint16_t ADDRESS_READY_MASK = (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
    CCU_CHK_RET(ccu::NotifyWait(pairedChannel, CHANNEL_NOTIFY_INDEX, ADDRESS_READY_MASK));
    return CCU_SUCCESS;
}

CcuResult SendAddressInfo(AllGatherContext &ctx)
{
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIndex],
            ctx.output[ctx.arg->rankId], OUTPUT_XN_ID, CHANNEL_NOTIFY_INDEX, 1U << OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIndex],
            ctx.outputToken[ctx.arg->rankId], TOKEN_XN_ID, CHANNEL_NOTIFY_INDEX, 1U << TOKEN_XN_ID));
    }
    return CCU_SUCCESS;
}

CcuResult WaitAddressInfo(AllGatherContext &ctx)
{
    constexpr uint16_t ADDRESS_READY_MASK = (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIndex], CHANNEL_NOTIFY_INDEX, ADDRESS_READY_MASK));
    }
    return CCU_SUCCESS;
}

bool UsesPairedHierarchySync(const AllGatherContext &ctx)
{
    return ctx.arg->hierarchicalGrid && ctx.arg->interRankCount == 2 &&
        ctx.arg->role == AllGatherKernelRole::INTER_SERVER;
}

CcuResult PreSync(AllGatherContext &ctx)
{
    if (UsesPairedHierarchySync(ctx)) {
        return PairedPreSync(ctx);
    }
    CCU_CHK_RET(SendAddressInfo(ctx));
    return WaitAddressInfo(ctx);
}

CcuResult WaitAddressForChannel(AllGatherContext &ctx, uint32_t channelIndex)
{
    constexpr uint16_t ADDRESS_READY_MASK = (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
    return ccu::NotifyWait(
        ctx.arg->channels[channelIndex], CHANNEL_NOTIFY_INDEX, ADDRESS_READY_MASK);
}

CcuResult WaitAddressForPhaseOneChannel(AllGatherContext &ctx, uint32_t channelIndex)
{
    if (UsesPairedHierarchySync(ctx) && channelIndex == ctx.arg->pairedChannelIndex) {
        return CCU_SUCCESS;
    }
    return WaitAddressForChannel(ctx, channelIndex);
}

CcuResult DrainPhysicalGatewayAddressInfo(AllGatherContext &ctx)
{
    if (!ctx.arg->physical8Plus4 || ctx.arg->role != AllGatherKernelRole::INTER_SERVER) {
        return CCU_SUCCESS;
    }
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        if (ctx.arg->hasGatewayTarget &&
            channelIndex == ctx.arg->gatewayTargetChannelIndex) {
            continue;
        }
        CCU_CHK_RET(WaitAddressForChannel(ctx, channelIndex));
    }
    return CCU_SUCCESS;
}

CcuResult PrepareHierarchicalPhaseOne(AllGatherContext &ctx)
{
    if (UsesPairedHierarchySync(ctx)) {
        return PairedPreSync(ctx);
    }
    return SendAddressInfo(ctx);
}

CcuResult ExecuteDirectAllGather(AllGatherContext &ctx)
{
    ccu::LocalAddr src;
    src.addr = ctx.input;
    src.token = ctx.inputToken;

    ccu::LocalAddr localDst;
    localDst.addr = ctx.output[ctx.arg->rankId];
    localDst.addr += ctx.currentRankOutputOffset;
    localDst.token = ctx.outputToken[ctx.arg->rankId];

    uint32_t completionCount = ctx.arg->channelCount;
    if (ctx.arg->copyLocal && ctx.arg->latencyOptimizedBarrier) {
        const uint16_t localMask = static_cast<uint16_t>(uint32_t{1} << completionCount);
        CCU_IF(ctx.localCopyNeeded != 0)
        {
            CCU_CHK_RET(ccu::LocalCopy(localDst, src, ctx.sliceSize, ctx.event, localMask));
        }
        CCU_ELSE
        {
            CCU_CHK_RET(ccu::EventRecord(ctx.event, localMask));
        }
    }
    if (ctx.arg->copyLocal) {
        ++completionCount;
    }

    constexpr uint16_t ADDRESS_READY_MASK = (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        CCU_CHK_RET(ccu::NotifyWait(
            ctx.arg->channels[channelIndex], CHANNEL_NOTIFY_INDEX, ADDRESS_READY_MASK));
        const uint32_t peerRank = ctx.arg->peerRanks[channelIndex];
        ccu::RemoteAddr remoteDst;
        remoteDst.addr = ctx.output[peerRank];
        remoteDst.addr += ctx.currentRankOutputOffset;
        remoteDst.token = ctx.outputToken[peerRank];
        const uint16_t channelMask = static_cast<uint16_t>(uint32_t{1} << channelIndex);
        CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIndex], remoteDst, src,
            ctx.sliceSize, ctx.event, channelMask));
    }

    if (ctx.arg->copyLocal && !ctx.arg->latencyOptimizedBarrier) {
        const uint16_t localMask = static_cast<uint16_t>(uint32_t{1} << ctx.arg->channelCount);
        CCU_IF(ctx.localCopyNeeded != 0)
        {
            CCU_CHK_RET(GroupCopy(ctx, localDst, src, ctx.goSize));
        }
        CCU_CHK_RET(ccu::EventRecord(ctx.event, localMask));
    }

    const uint16_t totalMask = static_cast<uint16_t>((uint32_t{1} << completionCount) - 1U);
    CCU_CHK_RET(ccu::EventWait(ctx.event, totalMask));
    return CCU_SUCCESS;
}

CcuResult InitRankOutputOffsets(AllGatherContext &ctx)
{
    ctx.rankOutputOffsets.resize(ctx.arg->rankSize);
    ctx.rankOutputOffsets[0] = 0;
    for (uint32_t rank = 1; rank < ctx.arg->rankSize; ++rank) {
        ctx.rankOutputOffsets[rank] = ctx.rankOutputOffsets[rank - 1];
        ctx.rankOutputOffsets[rank] += ctx.dataSize;
    }
    return CCU_SUCCESS;
}

CcuResult ExecuteDirectNhr(AllGatherContext &ctx)
{
    CCU_CHK_RET(InitRankOutputOffsets(ctx));

    ccu::LocalAddr localSrc;
    localSrc.addr = ctx.input;
    localSrc.token = ctx.inputToken;
    ccu::LocalAddr localDst;
    localDst.addr = ctx.output[ctx.arg->rankId];
    localDst.addr += ctx.currentRankOutputOffset;
    localDst.token = ctx.outputToken[ctx.arg->rankId];
    const bool localCopyPending = ctx.arg->copyLocal;

    for (uint32_t step = 0; step < ctx.arg->nhrStepCount; ++step) {
        const uint32_t toRank = ctx.arg->nhrToRanks[step];
        const uint32_t toChannelIndex = ctx.arg->nhrToChannelIndices[step];
        uint32_t operationCount = 0;
        for (uint32_t txIndex = ctx.arg->nhrTxOffsets[step];
            txIndex < ctx.arg->nhrTxOffsets[step + 1]; ++txIndex) {
            const uint32_t sourceRank = ctx.arg->nhrTxRanks[txIndex];
            ccu::LocalAddr src;
            if (step == 0 && sourceRank == ctx.arg->rankId) {
                src.addr = ctx.input;
                src.token = ctx.inputToken;
            } else {
                src.addr = ctx.output[ctx.arg->rankId];
                src.addr += ctx.rankOutputOffsets[sourceRank];
                src.token = ctx.outputToken[ctx.arg->rankId];
            }

            ccu::RemoteAddr remoteDst;
            remoteDst.addr = ctx.output[toRank];
            remoteDst.addr += ctx.rankOutputOffsets[sourceRank];
            remoteDst.token = ctx.outputToken[toRank];
            const uint16_t signalMask = static_cast<uint16_t>(uint32_t{1} << operationCount);
            CCU_CHK_RET(ccu::Write(ctx.arg->channels[toChannelIndex], remoteDst, src,
                ctx.sliceSize, ctx.event, signalMask));
            ++operationCount;
        }
        if (step == 0 && localCopyPending) {
            CCU_IF(ctx.localCopyNeeded != 0)
            {
                CCU_CHK_RET(GroupCopy(ctx, localDst, localSrc, ctx.goSize));
            }
        }
        const uint16_t totalMask = static_cast<uint16_t>((uint32_t{1} << operationCount) - 1U);
        CCU_CHK_RET(ccu::EventWait(ctx.event, totalMask));

        if (step + 1 != ctx.arg->nhrStepCount) {
            constexpr uint16_t STEP_SYNC_MASK = 1U << NHR_STEP_SYNC_ID;
            CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[toChannelIndex],
                CHANNEL_NOTIFY_INDEX, STEP_SYNC_MASK));
            CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[ctx.arg->nhrFromChannelIndices[step]],
                CHANNEL_NOTIFY_INDEX, STEP_SYNC_MASK));
        }
    }
    return CCU_SUCCESS;
}

CcuResult ExecuteHierarchicalPhaseOne(AllGatherContext &ctx)
{
    ccu::LocalAddr currentSrc;
    currentSrc.addr = ctx.input;
    currentSrc.addr += ctx.partOffset;
    currentSrc.token = ctx.inputToken;

    ccu::LocalAddr localDst;
    localDst.addr = ctx.output[ctx.arg->rankId];
    localDst.addr += ctx.currentRankOutputOffset;
    localDst.addr += ctx.partOffset;
    localDst.token = ctx.outputToken[ctx.arg->rankId];

    uint32_t operationCount = 0;
    CCU_IF(ctx.sliceSize != 0)
    {
        if (ctx.arg->role == AllGatherKernelRole::INTRA_SERVER) {
            for (uint32_t channelIndex = 0; channelIndex < ctx.arg->localPeerCount; ++channelIndex) {
                CCU_CHK_RET(WaitAddressForPhaseOneChannel(ctx, channelIndex));
                const uint32_t peerRank = ctx.arg->peerRanks[channelIndex];
                ccu::RemoteAddr remoteDst;
                remoteDst.addr = ctx.output[peerRank];
                remoteDst.addr += ctx.currentRankOutputOffset;
                remoteDst.addr += ctx.partOffset;
                remoteDst.token = ctx.outputToken[peerRank];
                const uint16_t signalMask = static_cast<uint16_t>(uint32_t{1} << operationCount);
                CCU_CHK_RET(ccu::Write(
                    ctx.arg->channels[channelIndex], remoteDst, currentSrc,
                    ctx.sliceSize, ctx.event, signalMask));
                ++operationCount;
            }
        } else if (ctx.arg->role == AllGatherKernelRole::INTER_SERVER) {
            const uint32_t firstChannelIndex = ctx.arg->physical8Plus4 ?
                ctx.arg->gatewayTargetChannelIndex : 0;
            const uint32_t channelCount = ctx.arg->physical8Plus4 ?
                (ctx.arg->hasGatewayTarget ? 1U : 0U) : ctx.arg->channelCount;
            for (uint32_t relativeIndex = 0; relativeIndex < channelCount; ++relativeIndex) {
                const uint32_t channelIndex = firstChannelIndex + relativeIndex;
                CCU_CHK_RET(WaitAddressForPhaseOneChannel(ctx, channelIndex));
                const uint32_t peerRank = ctx.arg->peerRanks[channelIndex];
                ccu::RemoteAddr remoteDst;
                remoteDst.addr = ctx.output[peerRank];
                remoteDst.addr += ctx.currentRankOutputOffset;
                remoteDst.addr += ctx.partOffset;
                remoteDst.token = ctx.outputToken[peerRank];
                const uint16_t signalMask = static_cast<uint16_t>(uint32_t{1} << operationCount);
                CCU_CHK_RET(ccu::Write(
                    ctx.arg->channels[channelIndex], remoteDst, currentSrc,
                    ctx.sliceSize, ctx.event, signalMask));
                ++operationCount;
            }
        } else {
            return CCU_E_PARA;
        }
    }

    CCU_CHK_RET(DrainPhysicalGatewayAddressInfo(ctx));

    if (ctx.arg->copyLocal) {
        CCU_IF(ctx.localCopyNeeded != 0)
        {
            CCU_CHK_RET(GroupCopy(ctx, localDst, currentSrc, ctx.goSize));
        }
        const uint16_t localMask = static_cast<uint16_t>(uint32_t{1} << operationCount);
        CCU_CHK_RET(ccu::EventRecord(ctx.event, localMask));
        ++operationCount;
    }
    if (operationCount != 0) {
        const uint16_t totalMask = static_cast<uint16_t>((uint32_t{1} << operationCount) - 1U);
        CCU_CHK_RET(ccu::EventWait(ctx.event, totalMask));
    }
    return CCU_SUCCESS;
}

CcuResult ExecuteHierarchicalPhaseTwo(AllGatherContext &ctx)
{
    CCU_CHK_RET(InitRankOutputOffsets(ctx));
    uint32_t operationCount = 0;
    if (ctx.arg->role == AllGatherKernelRole::INTER_SERVER && !ctx.arg->physical8Plus4) {
        const uint32_t firstInterChannel = 0;
        const uint32_t interChannelCount = ctx.arg->channelCount;
        for (uint32_t interIndex = 0; interIndex < interChannelCount; ++interIndex) {
            const uint32_t channelIndex = firstInterChannel + interIndex;
            const uint32_t peerRank = ctx.arg->peerRanks[channelIndex];
            for (uint32_t localIndex = 0; localIndex < ctx.arg->localRankCount; ++localIndex) {
                const uint32_t sourceRank = ctx.arg->localRanks[localIndex];
                ccu::LocalAddr src;
                src.addr = ctx.output[ctx.arg->rankId];
                src.addr += ctx.rankOutputOffsets[sourceRank];
                src.addr += ctx.partOffset;
                src.token = ctx.outputToken[ctx.arg->rankId];

                ccu::RemoteAddr remoteDst;
                remoteDst.addr = ctx.output[peerRank];
                remoteDst.addr += ctx.rankOutputOffsets[sourceRank];
                remoteDst.addr += ctx.partOffset;
                remoteDst.token = ctx.outputToken[peerRank];
                const uint16_t signalMask = static_cast<uint16_t>(uint32_t{1} << operationCount);
                CCU_CHK_RET(ccu::Write(
                    ctx.arg->channels[channelIndex], remoteDst, src, ctx.sliceSize, ctx.event, signalMask));
                ++operationCount;
            }
        }
    }
    if (ctx.arg->role == AllGatherKernelRole::INTRA_SERVER) {
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->localPeerCount; ++channelIndex) {
            const uint32_t peerRank = ctx.arg->peerRanks[channelIndex];
            const uint32_t sourceCount = ctx.arg->physical8Plus4 ?
                ctx.arg->gatewaySourceCount : ctx.arg->interRankCount;
            for (uint32_t sourceIndex = 0; sourceIndex < sourceCount; ++sourceIndex) {
                const uint32_t sourceRank = ctx.arg->physical8Plus4 ?
                    ctx.arg->gatewaySourceRanks[sourceIndex] : ctx.arg->interRanks[sourceIndex];
                ccu::LocalAddr src;
                src.addr = ctx.output[ctx.arg->rankId];
                src.addr += ctx.rankOutputOffsets[sourceRank];
                src.addr += ctx.partOffset;
                src.token = ctx.outputToken[ctx.arg->rankId];

                ccu::RemoteAddr remoteDst;
                remoteDst.addr = ctx.output[peerRank];
                remoteDst.addr += ctx.rankOutputOffsets[sourceRank];
                remoteDst.addr += ctx.partOffset;
                remoteDst.token = ctx.outputToken[peerRank];
                const uint16_t signalMask = static_cast<uint16_t>(uint32_t{1} << operationCount);
                CCU_CHK_RET(ccu::Write(
                    ctx.arg->channels[channelIndex], remoteDst, src, ctx.sliceSize, ctx.event, signalMask));
                ++operationCount;
            }
        }
    } else if (ctx.arg->role != AllGatherKernelRole::INTER_SERVER) {
        return CCU_E_PARA;
    }

    if (operationCount != 0) {
        const uint16_t totalMask = static_cast<uint16_t>((uint32_t{1} << operationCount) - 1U);
        CCU_CHK_RET(ccu::EventWait(ctx.event, totalMask));
    }
    return CCU_SUCCESS;
}

CcuResult ExecuteAllGather(AllGatherContext &ctx)
{
    if (ctx.arg->useNhr) {
        return ExecuteDirectNhr(ctx);
    }
    if (!ctx.arg->hierarchicalGrid) {
        return ExecuteDirectAllGather(ctx);
    }
    const bool rankShapeValid = ctx.arg->physical8Plus4 ?
        ctx.arg->localRankCount + ctx.arg->interRankCount == ctx.arg->rankSize :
        ctx.arg->localRankCount * ctx.arg->interRankCount == ctx.arg->rankSize;
    if (!rankShapeValid) {
        return CCU_E_PARA;
    }
    CCU_IF(ctx.executionMode == static_cast<uint64_t>(AllGatherExecutionMode::HIERARCHICAL_PHASE_ONE))
    {
        CCU_CHK_RET(ExecuteHierarchicalPhaseOne(ctx));
    }
    CCU_ELSE
    {
        CCU_CHK_RET(ExecuteHierarchicalPhaseTwo(ctx));
    }
    return CCU_SUCCESS;
}

CcuResult FullPeerPostSync(AllGatherContext &ctx)
{
    constexpr uint16_t POST_SYNC_MASK = 1U << POST_SYNC_ID;
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        CCU_CHK_RET(
            ccu::NotifyRecord(ctx.arg->channels[channelIndex], CHANNEL_NOTIFY_INDEX, POST_SYNC_MASK));
    }
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIndex], CHANNEL_NOTIFY_INDEX, POST_SYNC_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult GetChannelIndexForPeer(const AllGatherContext &ctx, uint32_t peerRank, uint32_t &channelIndex)
{
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        if (ctx.arg->peerRanks[index] == peerRank) {
            channelIndex = index;
            return CCU_SUCCESS;
        }
    }
    return CCU_E_PARA;
}

CcuResult DisseminationPostSync(AllGatherContext &ctx)
{
    constexpr uint16_t POST_SYNC_MASK = 1U << POST_SYNC_ID;
    for (uint32_t distance = 1; distance < ctx.arg->rankSize; distance <<= 1) {
        const uint32_t sendPeer = (ctx.arg->rankId + distance) % ctx.arg->rankSize;
        const uint32_t recvPeer =
            (ctx.arg->rankId + ctx.arg->rankSize - distance) % ctx.arg->rankSize;
        uint32_t sendChannelIndex = 0;
        uint32_t recvChannelIndex = 0;
        CCU_CHK_RET(GetChannelIndexForPeer(ctx, sendPeer, sendChannelIndex));
        CCU_CHK_RET(GetChannelIndexForPeer(ctx, recvPeer, recvChannelIndex));
        CCU_CHK_RET(
            ccu::NotifyRecord(ctx.arg->channels[sendChannelIndex], CHANNEL_NOTIFY_INDEX, POST_SYNC_MASK));
        CCU_CHK_RET(
            ccu::NotifyWait(ctx.arg->channels[recvChannelIndex], CHANNEL_NOTIFY_INDEX, POST_SYNC_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult PairedPostSync(AllGatherContext &ctx)
{
    constexpr uint16_t POST_SYNC_MASK = 1U << POST_SYNC_ID;
    const ChannelHandle pairedChannel = ctx.arg->channels[ctx.arg->pairedChannelIndex];
    CCU_CHK_RET(ccu::NotifyRecord(pairedChannel, CHANNEL_NOTIFY_INDEX, POST_SYNC_MASK));
    CCU_CHK_RET(ccu::NotifyWait(pairedChannel, CHANNEL_NOTIFY_INDEX, POST_SYNC_MASK));
    return CCU_SUCCESS;
}

CcuResult TwoServerLocalPostSync(AllGatherContext &ctx)
{
    constexpr uint32_t SERVER_RANK_COUNT = 8;
    constexpr uint16_t POST_SYNC_MASK = 1U << POST_SYNC_ID;
    const uint32_t serverBase = (ctx.arg->rankId / SERVER_RANK_COUNT) * SERVER_RANK_COUNT;
    const uint32_t localRank = ctx.arg->rankId % SERVER_RANK_COUNT;
    for (uint32_t distance = 1; distance < SERVER_RANK_COUNT; distance <<= 1) {
        const uint32_t sendPeer = serverBase + (localRank + distance) % SERVER_RANK_COUNT;
        const uint32_t recvPeer =
            serverBase + (localRank + SERVER_RANK_COUNT - distance) % SERVER_RANK_COUNT;
        uint32_t sendChannelIndex = 0;
        uint32_t recvChannelIndex = 0;
        CCU_CHK_RET(GetChannelIndexForPeer(ctx, sendPeer, sendChannelIndex));
        CCU_CHK_RET(GetChannelIndexForPeer(ctx, recvPeer, recvChannelIndex));
        CCU_CHK_RET(
            ccu::NotifyRecord(ctx.arg->channels[sendChannelIndex], CHANNEL_NOTIFY_INDEX, POST_SYNC_MASK));
        CCU_CHK_RET(
            ccu::NotifyWait(ctx.arg->channels[recvChannelIndex], CHANNEL_NOTIFY_INDEX, POST_SYNC_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult TwoServerCrossPostSync(AllGatherContext &ctx)
{
    constexpr uint32_t SERVER_RANK_COUNT = 8;
    constexpr uint32_t XOR_MASKS[] = {0, 1, 2, 4};
    constexpr uint16_t POST_SYNC_MASK = 1U << POST_SYNC_ID;
    const uint32_t remoteServerBase = ctx.arg->rankId < SERVER_RANK_COUNT ? SERVER_RANK_COUNT : 0;
    const uint32_t localRank = ctx.arg->rankId % SERVER_RANK_COUNT;
    for (uint32_t xorMask : XOR_MASKS) {
        const uint32_t peerRank = remoteServerBase + (localRank ^ xorMask);
        uint32_t channelIndex = 0;
        CCU_CHK_RET(GetChannelIndexForPeer(ctx, peerRank, channelIndex));
        CCU_CHK_RET(
            ccu::NotifyRecord(ctx.arg->channels[channelIndex], CHANNEL_NOTIFY_INDEX, POST_SYNC_MASK));
        CCU_CHK_RET(
            ccu::NotifyWait(ctx.arg->channels[channelIndex], CHANNEL_NOTIFY_INDEX, POST_SYNC_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult PostSync(AllGatherContext &ctx)
{
    if (ctx.arg->hierarchicalGrid && ctx.arg->interRankCount == 2 &&
        ctx.arg->role == AllGatherKernelRole::INTER_SERVER) {
        return PairedPostSync(ctx);
    }
    if (ctx.arg->hierarchicalGrid && ctx.arg->role == AllGatherKernelRole::INTRA_SERVER) {
        return FullPeerPostSync(ctx);
    }
    // Every direct-small kernel owns a symmetric peer group.  Recording to
    // all peers and then waiting for all peers is a one-round group barrier;
    // the topology-specific dissemination schedules below add 3-4 serial
    // network round trips and are only worthwhile for non-latency paths.
    if (ctx.arg->latencyOptimizedBarrier) {
        return FullPeerPostSync(ctx);
    }
    if (!ctx.arg->hierarchicalGrid && ctx.arg->rankSize == 16 && ctx.arg->channelCount == 7) {
        return TwoServerLocalPostSync(ctx);
    }
    if (!ctx.arg->hierarchicalGrid && ctx.arg->rankSize == 16 && ctx.arg->channelCount == 8) {
        return TwoServerCrossPostSync(ctx);
    }
    if (ctx.arg->channelCount == ctx.arg->rankSize - 1) {
        return DisseminationPostSync(ctx);
    }
    return FullPeerPostSync(ctx);
}
} // namespace

CcuResult CcuKernel(CcuKernelArg arg)
{
    AllGatherContext ctx;
    ctx.arg = static_cast<AllGatherKernelArg *>(arg);
    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    if (ctx.arg->hierarchicalGrid) {
        CCU_IF(ctx.executionMode == static_cast<uint64_t>(AllGatherExecutionMode::HIERARCHICAL_PHASE_ONE))
        {
            CCU_CHK_RET(PrepareHierarchicalPhaseOne(ctx));
        }
    } else if (ctx.arg->useNhr) {
        CCU_CHK_RET(PreSync(ctx));
    } else {
        CCU_CHK_RET(SendAddressInfo(ctx));
    }
    CCU_CHK_RET(ExecuteAllGather(ctx));
    CCU_CHK_RET(PostSync(ctx));
    return CcuResult::CCU_SUCCESS;
}

} // namespace ops_hccl
