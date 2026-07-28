/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <hcomm/hcomm_primitives.h>

#include <array>
#include <memory>
#include <vector>

#include "ccu_kernel.h"
#include "log.h"

#ifndef CCU_CHK_RET
#define CCU_CHK_RET(expr) \
    do { \
        CcuResult _ccuRet = (expr); \
        if (_ccuRet != CCU_SUCCESS) { \
            return _ccuRet; \
        } \
    } while (0)
#endif

namespace ops_hccl {
namespace {

    constexpr uint32_t INPUT_ADDR_XN_ID = 0;
    constexpr uint32_t INPUT_TOKEN_XN_ID = 1;
    constexpr uint32_t OUTPUT_ADDR_XN_ID = 2;
    constexpr uint32_t OUTPUT_TOKEN_XN_ID = 3;
    constexpr uint32_t CKE_IDX = 0;
    constexpr uint16_t RS_READY_MASK = 1U << 4;
    constexpr uint16_t AG_DONE_MASK = 1U << 5;
    constexpr uint32_t MISSION_PARTICIPANTS = 2;
    constexpr uint32_t GROUP_REDUCE_INTERLEAVE = 8;
    constexpr uint32_t GROUP_REDUCE_LOOP_COUNT = 16;
    constexpr uint64_t GROUP_REDUCE_SLICE_BYTES = 4096;

    struct DirectContext {
        const CcuKernelArgDirect *arg = nullptr;

        ccu::Variable inputAddr;
        ccu::Variable outputAddr;
        ccu::Variable inputToken;
        ccu::Variable outputToken;
        ccu::Variable scratchAddr;
        ccu::Variable scratchToken;
        ccu::Variable normalSliceBytes;
        ccu::Variable lastSliceBytes;
        ccu::Variable inPlace;
        ccu::Variable runtimePhase;
        ccu::Variable groupAddrOffset;
        ccu::Variable groupLoopParam;
        ccu::Variable groupParallelParam;
        ccu::Variable groupResidual;
        std::vector<ccu::Variable> peerInputAddr;
        std::vector<ccu::Variable> peerInputToken;
        std::vector<ccu::Variable> peerOutputAddr;
        std::vector<ccu::Variable> peerOutputToken;
        ccu::Event ioEvent;
        ccu::Event reduceEvent;
    };

    ccu::Variable MulByStatic(const ccu::Variable &value, uint32_t multiplier)
    {
        ccu::Variable result;
        result = 0;
        for (uint32_t i = 0; i < multiplier; ++i) {
            result += value;
        }
        return result;
    }

    ccu::LocalAddr ScratchSlot(const DirectContext &ctx, uint32_t slot, const ccu::Variable &sliceBytes)
    {
        ccu::LocalAddr addr;
        addr.addr = ctx.scratchAddr;
        addr.addr += MulByStatic(sliceBytes, slot);
        addr.token = ctx.scratchToken;
        return addr;
    }

    CcuResult InitResources(DirectContext &ctx)
    {
        const auto *arg = ctx.arg;
        if (arg == nullptr || arg->channelCount == 0 || arg->channelCount >= MAX_RANK_SIZE) {
            HCCL_ERROR("[CcuDirectAllReduceKernel] invalid kernel arg/channel count");
            return CCU_E_PARA;
        }

        ctx.peerInputAddr.resize(arg->channelCount);
        ctx.peerInputToken.resize(arg->channelCount);
        ctx.peerOutputAddr.resize(arg->channelCount);
        ctx.peerOutputToken.resize(arg->channelCount);
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            ctx.peerInputAddr[i] = ccu::GetResByChannel<ccu::Variable>(arg->channels[i], INPUT_ADDR_XN_ID);
            ctx.peerInputToken[i] = ccu::GetResByChannel<ccu::Variable>(arg->channels[i], INPUT_TOKEN_XN_ID);
            ctx.peerOutputAddr[i] = ccu::GetResByChannel<ccu::Variable>(arg->channels[i], OUTPUT_ADDR_XN_ID);
            ctx.peerOutputToken[i] = ccu::GetResByChannel<ccu::Variable>(arg->channels[i], OUTPUT_TOKEN_XN_ID);
        }
        return CCU_SUCCESS;
    }

    CcuResult LoadBaseArgs(DirectContext &ctx)
    {
        uint32_t argId = 0;
        CCU_CHK_RET(ccu::LoadArg(ctx.inputAddr, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.outputAddr, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.inputToken, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.outputToken, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.scratchAddr, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.scratchToken, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.normalSliceBytes, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.lastSliceBytes, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.inPlace, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.runtimePhase, argId++));
        return CCU_SUCCESS;
    }

    CcuResult LoadGroupArgs(DirectContext &ctx)
    {
        uint32_t argId = 10U;
        CCU_CHK_RET(ccu::LoadArg(ctx.groupAddrOffset, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.groupLoopParam, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.groupParallelParam, argId++));
        CCU_CHK_RET(ccu::LoadArg(ctx.groupResidual, argId++));
        return CCU_SUCCESS;
    }

    uint64_t SetBits(uint16_t end)
    {
        return (uint64_t{1} << (end + 1U)) - uint64_t{1};
    }

    uint64_t GetLoopParam(uint64_t loopCtxId, uint64_t gsaOffset, uint64_t loopIterNum)
    {
        return ((loopCtxId & SetBits(8)) << 45U) | ((gsaOffset & SetBits(32)) << 13U) | (loopIterNum & SetBits(13));
    }

    uint64_t GetParallelParam(uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum)
    {
        return ((repeatNum & SetBits(7)) << 55U) | ((repeatLoopIndex & SetBits(7)) << 48U)
               | ((totalLoopNum & SetBits(7)) << 41U);
    }

    uint64_t GetOffsetParam(uint64_t gsaOffset, uint64_t msOffset, uint64_t ckeOffset)
    {
        return ((gsaOffset & SetBits(32)) << 21U) | ((msOffset & SetBits(11)) << 10U) | (ckeOffset & SetBits(10));
    }

    struct GroupReduceVars {
        ccu::LocalAddr dst[2];
        std::array<std::vector<ccu::RemoteAddr>, 2> remoteSrc;
        ccu::LocalAddr localSrc[2];
        ccu::Variable len[2];
    };

    CcuResult GroupReduceSingleDie(
        DirectContext &ctx, const ccu::Variable &sliceOffset, const ccu::Variable &localSliceBytes)
    {
        const auto *arg = ctx.arg;
        if (arg->channelCount + 1U > GROUP_REDUCE_INTERLEAVE) {
            HCCL_ERROR("[CCU_V003] GroupReduce exceeds one interleave group");
            return CCU_E_NOT_SUPPORT;
        }

        ccu::Array<ccu::Event> completedEvents(GROUP_REDUCE_LOOP_COUNT);
        ccu::Array<ccu::CcuBuffer> ccuBuffers(GROUP_REDUCE_LOOP_COUNT * GROUP_REDUCE_INTERLEAVE);
        GroupReduceVars vars;
        ccu::Variable loopConfig[2];
        std::unique_ptr<ccu::Func> bodies[2];
        std::unique_ptr<ccu::Loop> loops[2];

        for (uint32_t index = 0; index < 2U; ++index) {
            vars.remoteSrc[index].resize(arg->channelCount);
            const uint32_t bufferBase = index * GROUP_REDUCE_INTERLEAVE;
            const ccu::Event loopEvent = completedEvents[index];
            bodies[index].reset(new ccu::Func([&ctx, &vars, &ccuBuffers, index, bufferBase, loopEvent]() {
                const uint32_t channelCount = ctx.arg->channelCount;
                for (uint32_t i = 0; i < channelCount; ++i) {
                    ccu::Read(ctx.arg->channels[i], ccuBuffers[bufferBase + i], vars.remoteSrc[index][i],
                        vars.len[index], loopEvent, static_cast<uint16_t>(1U << i));
                }
                ccu::LocalCopy(ccuBuffers[bufferBase + channelCount], vars.localSrc[index], vars.len[index], loopEvent,
                    static_cast<uint16_t>(1U << channelCount));
                ccu::EventWait(loopEvent, static_cast<uint16_t>((1U << (channelCount + 1U)) - 1U));
                ccu::LocalReduce(&ccuBuffers[bufferBase], channelCount + 1U, HCCL_DATA_TYPE_FP32, HCCL_DATA_TYPE_FP32,
                    HCCL_REDUCE_SUM, vars.len[index], loopEvent, 1U);
                ccu::EventWait(loopEvent, 1U);
                ccu::LocalCopy(vars.dst[index], ccuBuffers[bufferBase], vars.len[index], loopEvent, 1U);
                ccu::EventWait(loopEvent, 1U);
            }));
            loops[index].reset(new ccu::Loop(loopConfig[index], *bodies[index]));
        }

        ccu::LocalAddr localInput;
        localInput.addr = ctx.inputAddr;
        localInput.addr += sliceOffset;
        localInput.token = ctx.inputToken;

        ccu::LocalAddr localOutput;
        localOutput.addr = ctx.outputAddr;
        localOutput.addr += sliceOffset;
        localOutput.token = ctx.outputToken;

        std::vector<ccu::RemoteAddr> peerInputs(arg->channelCount);
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            peerInputs[i].addr = ctx.peerInputAddr[i];
            peerInputs[i].addr += sliceOffset;
            peerInputs[i].token = ctx.peerInputToken[i];
        }

        ccu::Variable loopParam;
        ccu::Variable parallelConfig;
        ccu::Variable offsetConfig;
        ccu::Variable sliceBytes;

        CCU_IF(ctx.groupAddrOffset != 0)
        {
            loopParam = GetLoopParam(0U, GROUP_REDUCE_SLICE_BYTES * GROUP_REDUCE_LOOP_COUNT, 0U);
            loopParam += ctx.groupLoopParam;
            sliceBytes = GROUP_REDUCE_SLICE_BYTES;
            for (uint32_t i = 0; i < arg->channelCount; ++i) {
                vars.remoteSrc[0][i].addr = peerInputs[i].addr;
                vars.remoteSrc[0][i].token = peerInputs[i].token;
            }
            vars.localSrc[0].addr = localInput.addr;
            vars.localSrc[0].token = localInput.token;
            vars.dst[0].addr = localOutput.addr;
            vars.dst[0].token = localOutput.token;
            vars.len[0] = sliceBytes;
            parallelConfig = GetParallelParam(GROUP_REDUCE_LOOP_COUNT - 1U, 0U, 1U);
            offsetConfig = GetOffsetParam(GROUP_REDUCE_SLICE_BYTES, GROUP_REDUCE_INTERLEAVE, 1U);
            loopConfig[0] = loopParam;
            std::vector<ccu::Loop> groupLoops{*loops[0]};
            ccu::LoopGroup group(parallelConfig, offsetConfig, GROUP_REDUCE_LOOP_COUNT, groupLoops);
        }

        CCU_IF(ctx.groupParallelParam != 0)
        {
            for (uint32_t i = 0; i < arg->channelCount; ++i) {
                peerInputs[i].addr += ctx.groupAddrOffset;
            }
            localInput.addr += ctx.groupAddrOffset;
            localOutput.addr += ctx.groupAddrOffset;

            for (uint32_t i = 0; i < arg->channelCount; ++i) {
                vars.remoteSrc[0][i].addr = peerInputs[i].addr;
                vars.remoteSrc[0][i].token = peerInputs[i].token;
            }
            vars.localSrc[0].addr = localInput.addr;
            vars.localSrc[0].token = localInput.token;
            vars.dst[0].addr = localOutput.addr;
            vars.dst[0].token = localOutput.token;
            vars.len[0] = ctx.groupResidual;

            for (uint32_t i = 0; i < arg->channelCount; ++i) {
                peerInputs[i].addr += ctx.groupResidual;
            }
            localInput.addr += ctx.groupResidual;
            localOutput.addr += ctx.groupResidual;
            sliceBytes = GROUP_REDUCE_SLICE_BYTES;
            for (uint32_t i = 0; i < arg->channelCount; ++i) {
                vars.remoteSrc[1][i].addr = peerInputs[i].addr;
                vars.remoteSrc[1][i].token = peerInputs[i].token;
            }
            vars.localSrc[1].addr = localInput.addr;
            vars.localSrc[1].token = localInput.token;
            vars.dst[1].addr = localOutput.addr;
            vars.dst[1].token = localOutput.token;
            vars.len[1] = sliceBytes;
            loopConfig[0] = GetLoopParam(0U, 0U, 1U);
            loopConfig[1] = GetLoopParam(0U, 0U, 1U);
            offsetConfig = GetOffsetParam(GROUP_REDUCE_SLICE_BYTES, GROUP_REDUCE_INTERLEAVE, 1U);
            std::vector<ccu::Loop> groupLoops{*loops[0], *loops[1]};
            ccu::LoopGroup group(ctx.groupParallelParam, offsetConfig, GROUP_REDUCE_LOOP_COUNT, groupLoops);
        }
        return CCU_SUCCESS;
    }

    CcuResult PreSyncInput(DirectContext &ctx)
    {
        const auto *arg = ctx.arg;
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            CCU_CHK_RET(ccu::WriteVariableWithNotify(
                arg->channels[i], ctx.inputAddr, INPUT_ADDR_XN_ID, CKE_IDX, 1U << INPUT_ADDR_XN_ID));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(
                arg->channels[i], ctx.inputToken, INPUT_TOKEN_XN_ID, CKE_IDX, 1U << INPUT_TOKEN_XN_ID));
        }
        constexpr uint16_t inputMask = (1U << INPUT_ADDR_XN_ID) | (1U << INPUT_TOKEN_XN_ID);
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_IDX, inputMask));
        }
        return CCU_SUCCESS;
    }

    CcuResult PreSyncOutput(DirectContext &ctx)
    {
        const auto *arg = ctx.arg;
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            CCU_CHK_RET(ccu::WriteVariableWithNotify(
                arg->channels[i], ctx.outputAddr, OUTPUT_ADDR_XN_ID, CKE_IDX, 1U << OUTPUT_ADDR_XN_ID));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(
                arg->channels[i], ctx.outputToken, OUTPUT_TOKEN_XN_ID, CKE_IDX, 1U << OUTPUT_TOKEN_XN_ID));
        }
        constexpr uint16_t outputMask = (1U << OUTPUT_ADDR_XN_ID) | (1U << OUTPUT_TOKEN_XN_ID);
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_IDX, outputMask));
        }
        return CCU_SUCCESS;
    }

    CcuResult ReadOwnerContributions(
        DirectContext &ctx, const ccu::Variable &sliceOffset, const ccu::Variable &localSliceBytes)
    {
        const auto *arg = ctx.arg;
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            ccu::RemoteAddr src;
            src.addr = ctx.peerInputAddr[i];
            src.addr += sliceOffset;
            src.token = ctx.peerInputToken[i];

            ccu::LocalAddr dst = ScratchSlot(ctx, arg->scratchSlots[i], localSliceBytes);
            CCU_CHK_RET(
                ccu::Read(arg->channels[i], dst, src, localSliceBytes, ctx.ioEvent, static_cast<uint16_t>(1U << i)));
        }
        const uint16_t mask = static_cast<uint16_t>((1U << arg->channelCount) - 1U);
        CCU_CHK_RET(ccu::EventWait(ctx.ioEvent, mask));
        return CCU_SUCCESS;
    }

    CcuResult BuildPartial(DirectContext &ctx, const ccu::Variable &sliceOffset, const ccu::Variable &localSliceBytes)
    {
        const auto *arg = ctx.arg;
        if (arg->includeLocal) {
            ccu::LocalAddr input;
            input.addr = ctx.inputAddr;
            input.addr += sliceOffset;
            input.token = ctx.inputToken;

            ccu::LocalAddr output;
            output.addr = ctx.outputAddr;
            output.addr += sliceOffset;
            output.token = ctx.outputToken;

            CCU_IF(ctx.inPlace == 0)
            {
                CCU_CHK_RET(ccu::LocalCopy(output, input, localSliceBytes, ctx.reduceEvent, 1));
                CCU_CHK_RET(ccu::EventWait(ctx.reduceEvent, 1));
            }

            for (uint32_t i = 0; i < arg->channelCount; ++i) {
                ccu::LocalAddr src = ScratchSlot(ctx, arg->scratchSlots[i], localSliceBytes);
                CCU_CHK_RET(ccu::LocalReduce(
                    output, src, localSliceBytes, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.reduceEvent, 1));
                CCU_CHK_RET(ccu::EventWait(ctx.reduceEvent, 1));
            }
        } else {
            ccu::LocalAddr partial = ScratchSlot(ctx, arg->scratchSlots[0], localSliceBytes);
            for (uint32_t i = 1; i < arg->channelCount; ++i) {
                ccu::LocalAddr src = ScratchSlot(ctx, arg->scratchSlots[i], localSliceBytes);
                CCU_CHK_RET(ccu::LocalReduce(
                    partial, src, localSliceBytes, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.reduceEvent, 1));
                CCU_CHK_RET(ccu::EventWait(ctx.reduceEvent, 1));
            }
        }
        return CCU_SUCCESS;
    }

    CcuResult MergePartials(DirectContext &ctx, const ccu::Variable &sliceOffset, const ccu::Variable &localSliceBytes)
    {
        if (ctx.arg->activeDieCount != MISSION_PARTICIPANTS || !ctx.arg->includeLocal) {
            return CCU_SUCCESS;
        }

        ccu::LocalAddr output;
        output.addr = ctx.outputAddr;
        output.addr += sliceOffset;
        output.token = ctx.outputToken;

        ccu::LocalAddr other = ScratchSlot(ctx, ctx.arg->otherPartialScratchSlot, localSliceBytes);
        CCU_CHK_RET(
            ccu::LocalReduce(output, other, localSliceBytes, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.reduceEvent, 1));
        CCU_CHK_RET(ccu::EventWait(ctx.reduceEvent, 1));
        return CCU_SUCCESS;
    }

    CcuResult ChannelBarrier(DirectContext &ctx, uint16_t mask)
    {
        const auto *arg = ctx.arg;
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            CCU_CHK_RET(ccu::NotifyRecord(arg->channels[i], CKE_IDX, mask));
        }
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_IDX, mask));
        }
        return CCU_SUCCESS;
    }

    CcuResult ReadPeerResults(DirectContext &ctx)
    {
        const auto *arg = ctx.arg;
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            ccu::Variable peerOffset = MulByStatic(ctx.normalSliceBytes, arg->peerRanks[i]);
            ccu::Variable peerSliceBytes;
            if (arg->peerRanks[i] == arg->rankSize - 1) {
                peerSliceBytes = ctx.lastSliceBytes;
            } else {
                peerSliceBytes = ctx.normalSliceBytes;
            }

            ccu::RemoteAddr src;
            src.addr = ctx.peerOutputAddr[i];
            src.addr += peerOffset;
            src.token = ctx.peerOutputToken[i];

            ccu::LocalAddr dst;
            dst.addr = ctx.outputAddr;
            dst.addr += peerOffset;
            dst.token = ctx.outputToken;

            CCU_CHK_RET(
                ccu::Read(arg->channels[i], dst, src, peerSliceBytes, ctx.ioEvent, static_cast<uint16_t>(1U << i)));
        }
        const uint16_t mask = static_cast<uint16_t>((1U << arg->channelCount) - 1U);
        CCU_CHK_RET(ccu::EventWait(ctx.ioEvent, mask));
        return CCU_SUCCESS;
    }

} // namespace

CcuResult CcuDirectAllReduceKernel(CcuKernelArg kernelArg)
{
    auto *arg = static_cast<CcuKernelArgDirect *>(kernelArg);
    DirectContext ctx;
    ctx.arg = arg;

    CCU_CHK_RET(InitResources(ctx));
    CCU_CHK_RET(LoadBaseArgs(ctx));

    ccu::Variable localSliceBytes;
    if (arg->myRank == arg->rankSize - 1) {
        localSliceBytes = ctx.lastSliceBytes;
    } else {
        localSliceBytes = ctx.normalSliceBytes;
    }
    ccu::Variable sliceOffset = MulByStatic(ctx.normalSliceBytes, arg->myRank);

    if (arg->phase == DirectKernelPhase::MERGE) {
        CCU_IF(localSliceBytes != 0)
        {
            CCU_CHK_RET(MergePartials(ctx, sliceOffset, localSliceBytes));
        }
        return CCU_SUCCESS;
    }

    if (arg->phase == DirectKernelPhase::FUSED_SINGLE_DIE) {
        CCU_CHK_RET(LoadGroupArgs(ctx));
        CCU_CHK_RET(PreSyncInput(ctx));
        CCU_CHK_RET(PreSyncOutput(ctx));
        CCU_IF(localSliceBytes != 0)
        {
            CCU_CHK_RET(GroupReduceSingleDie(ctx, sliceOffset, localSliceBytes));
        }
        CCU_CHK_RET(ChannelBarrier(ctx, RS_READY_MASK));
        CCU_CHK_RET(ReadPeerResults(ctx));
        CCU_CHK_RET(ChannelBarrier(ctx, AG_DONE_MASK));
        return CCU_SUCCESS;
    }

    if (arg->phase == DirectKernelPhase::DIRECT_SINGLE_DIE) {
        CCU_CHK_RET(PreSyncInput(ctx));
        CCU_CHK_RET(PreSyncOutput(ctx));
        CCU_IF(localSliceBytes != 0)
        {
            CCU_CHK_RET(ReadOwnerContributions(ctx, sliceOffset, localSliceBytes));
            CCU_CHK_RET(BuildPartial(ctx, sliceOffset, localSliceBytes));
        }
        CCU_CHK_RET(ChannelBarrier(ctx, RS_READY_MASK));
        CCU_CHK_RET(ReadPeerResults(ctx));
        CCU_CHK_RET(ChannelBarrier(ctx, AG_DONE_MASK));
        return CCU_SUCCESS;
    }

    CCU_IF(ctx.runtimePhase == 0)
    {
        CCU_CHK_RET(PreSyncInput(ctx));
        CCU_IF(localSliceBytes != 0)
        {
            CCU_CHK_RET(ReadOwnerContributions(ctx, sliceOffset, localSliceBytes));
            CCU_CHK_RET(BuildPartial(ctx, sliceOffset, localSliceBytes));
        }
    }
    CCU_ELSE
    {
        CCU_CHK_RET(PreSyncOutput(ctx));
        CCU_CHK_RET(ChannelBarrier(ctx, RS_READY_MASK));
        CCU_CHK_RET(ReadPeerResults(ctx));
        CCU_CHK_RET(ChannelBarrier(ctx, AG_DONE_MASK));
    }
    return CCU_SUCCESS;
}

namespace {

// This is the online-proven V026 owner-shard GroupReduce body, isolated from
// the V006 common kernel entry.  The 2x8 channel split supplies at most eight
// inputs per die (local+7 or eight remote), so the unused two-batch research
// path is deliberately not part of the submitted kernel.
CcuResult GroupReduceOwnerPartialIsolated(
    DirectContext &ctx, const ccu::Variable &sliceOffset, const ccu::Variable &localSliceBytes)
{
    const auto *arg = ctx.arg;
    const uint32_t inputCount = arg->channelCount + (arg->includeLocal ? 1U : 0U);
    if (inputCount == 0U || inputCount > GROUP_REDUCE_INTERLEAVE) {
        HCCL_ERROR("[CCU_V027] unsupported isolated GroupReduce input count[%u]", inputCount);
        return CCU_E_NOT_SUPPORT;
    }

    ccu::Array<ccu::Event> completedEvents(GROUP_REDUCE_LOOP_COUNT);
    ccu::Array<ccu::CcuBuffer> ccuBuffers(GROUP_REDUCE_LOOP_COUNT * GROUP_REDUCE_INTERLEAVE);

    GroupReduceVars vars;
    ccu::Variable loopConfig[2];
    std::unique_ptr<ccu::Func> bodies[2];
    std::unique_ptr<ccu::Loop> loops[2];

    for (uint32_t index = 0; index < 2U; ++index) {
        vars.remoteSrc[index].resize(arg->channelCount);
        const uint32_t bufferBase = index * GROUP_REDUCE_INTERLEAVE;
        const ccu::Event loopEvent = completedEvents[index];
        bodies[index].reset(new ccu::Func([&ctx, &vars, &ccuBuffers, index, bufferBase, loopEvent, inputCount]() {
            const uint32_t channelCount = ctx.arg->channelCount;
            for (uint32_t i = 0; i < channelCount; ++i) {
                ccu::Read(ctx.arg->channels[i], ccuBuffers[bufferBase + i], vars.remoteSrc[index][i],
                    vars.len[index], loopEvent, static_cast<uint16_t>(1U << i));
            }
            if (ctx.arg->includeLocal) {
                ccu::LocalCopy(ccuBuffers[bufferBase + channelCount], vars.localSrc[index], vars.len[index], loopEvent,
                    static_cast<uint16_t>(1U << channelCount));
            }
            ccu::EventWait(loopEvent, static_cast<uint16_t>((1U << inputCount) - 1U));
            ccu::LocalReduce(&ccuBuffers[bufferBase], inputCount, HCCL_DATA_TYPE_FP32, HCCL_DATA_TYPE_FP32,
                HCCL_REDUCE_SUM, vars.len[index], loopEvent, 1U);
            ccu::EventWait(loopEvent, 1U);
            ccu::LocalCopy(vars.dst[index], ccuBuffers[bufferBase], vars.len[index], loopEvent, 1U);
            ccu::EventWait(loopEvent, 1U);
        }));
        loops[index].reset(new ccu::Loop(loopConfig[index], *bodies[index]));
    }

    ccu::LocalAddr localInput;
    if (arg->includeLocal) {
        localInput.addr = ctx.inputAddr;
        localInput.addr += sliceOffset;
        localInput.token = ctx.inputToken;
    }

    ccu::LocalAddr destination;
    if (arg->includeLocal) {
        destination.addr = ctx.outputAddr;
        destination.addr += sliceOffset;
        destination.token = ctx.outputToken;
    } else {
        destination = ScratchSlot(ctx, arg->scratchSlots[0], localSliceBytes);
    }

    std::vector<ccu::RemoteAddr> peerInputs(arg->channelCount);
    for (uint32_t i = 0; i < arg->channelCount; ++i) {
        peerInputs[i].addr = ctx.peerInputAddr[i];
        peerInputs[i].addr += sliceOffset;
        peerInputs[i].token = ctx.peerInputToken[i];
    }

    ccu::Variable loopParam;
    ccu::Variable parallelConfig;
    ccu::Variable offsetConfig;
    ccu::Variable sliceBytes;

    CCU_IF(ctx.groupAddrOffset != 0)
    {
        loopParam = GetLoopParam(0U, GROUP_REDUCE_SLICE_BYTES * GROUP_REDUCE_LOOP_COUNT, 0U);
        loopParam += ctx.groupLoopParam;
        sliceBytes = GROUP_REDUCE_SLICE_BYTES;
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            vars.remoteSrc[0][i] = peerInputs[i];
        }
        if (arg->includeLocal) {
            vars.localSrc[0] = localInput;
        }
        vars.dst[0] = destination;
        vars.len[0] = sliceBytes;
        parallelConfig = GetParallelParam(GROUP_REDUCE_LOOP_COUNT - 1U, 0U, 1U);
        offsetConfig = GetOffsetParam(GROUP_REDUCE_SLICE_BYTES, GROUP_REDUCE_INTERLEAVE, 1U);
        loopConfig[0] = loopParam;
        std::vector<ccu::Loop> groupLoops{*loops[0]};
        ccu::LoopGroup group(parallelConfig, offsetConfig, GROUP_REDUCE_LOOP_COUNT, groupLoops);
    }

    CCU_IF(ctx.groupParallelParam != 0)
    {
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            peerInputs[i].addr += ctx.groupAddrOffset;
        }
        if (arg->includeLocal) {
            localInput.addr += ctx.groupAddrOffset;
        }
        destination.addr += ctx.groupAddrOffset;

        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            vars.remoteSrc[0][i] = peerInputs[i];
        }
        if (arg->includeLocal) {
            vars.localSrc[0] = localInput;
        }
        vars.dst[0] = destination;
        vars.len[0] = ctx.groupResidual;

        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            peerInputs[i].addr += ctx.groupResidual;
        }
        if (arg->includeLocal) {
            localInput.addr += ctx.groupResidual;
        }
        destination.addr += ctx.groupResidual;
        sliceBytes = GROUP_REDUCE_SLICE_BYTES;
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            vars.remoteSrc[1][i] = peerInputs[i];
        }
        if (arg->includeLocal) {
            vars.localSrc[1] = localInput;
        }
        vars.dst[1] = destination;
        vars.len[1] = sliceBytes;
        loopConfig[0] = GetLoopParam(0U, 0U, 1U);
        loopConfig[1] = GetLoopParam(0U, 0U, 1U);
        offsetConfig = GetOffsetParam(GROUP_REDUCE_SLICE_BYTES, GROUP_REDUCE_INTERLEAVE, 1U);
        std::vector<ccu::Loop> groupLoops{*loops[0], *loops[1]};
        ccu::LoopGroup group(ctx.groupParallelParam, offsetConfig, GROUP_REDUCE_LOOP_COUNT, groupLoops);
    }
    return CCU_SUCCESS;
}

CcuResult PushOwnerResultIsolated(
    DirectContext &ctx, const ccu::Variable &sliceOffset, const ccu::Variable &localSliceBytes)
{
    const auto *arg = ctx.arg;
    ccu::LocalAddr source;
    source.addr = ctx.outputAddr;
    source.addr += sliceOffset;
    source.token = ctx.outputToken;

    for (uint32_t i = 0; i < arg->channelCount; ++i) {
        ccu::RemoteAddr destination;
        destination.addr = ctx.peerOutputAddr[i];
        destination.addr += sliceOffset;
        destination.token = ctx.peerOutputToken[i];
        CCU_CHK_RET(ccu::Write(
            arg->channels[i], destination, source, localSliceBytes, ctx.ioEvent, static_cast<uint16_t>(1U << i)));
    }
    const uint16_t mask = static_cast<uint16_t>((1U << arg->channelCount) - 1U);
    CCU_CHK_RET(ccu::EventWait(ctx.ioEvent, mask));
    return CCU_SUCCESS;
}

} // namespace

CcuResult CcuSingleGroupDirectPushAllReduceKernel(CcuKernelArg kernelArg)
{
    auto *arg = static_cast<CcuKernelArgDirect *>(kernelArg);
    if (arg == nullptr || arg->rankSize != 4U || arg->activeDieCount != 1U || !arg->includeLocal
        || arg->channelCount != 3U) {
        return CCU_E_PARA;
    }

    DirectContext ctx;
    ctx.arg = arg;
    CCU_CHK_RET(InitResources(ctx));
    CCU_CHK_RET(LoadBaseArgs(ctx));
    CCU_CHK_RET(LoadGroupArgs(ctx));

    ccu::Variable localSliceBytes;
    if (arg->myRank == arg->rankSize - 1U) {
        localSliceBytes = ctx.lastSliceBytes;
    } else {
        localSliceBytes = ctx.normalSliceBytes;
    }
    ccu::Variable sliceOffset = MulByStatic(ctx.normalSliceBytes, arg->myRank);

    CCU_CHK_RET(PreSyncInput(ctx));
    CCU_CHK_RET(PreSyncOutput(ctx));
    CCU_IF(localSliceBytes != 0)
    {
        CCU_CHK_RET(GroupReduceSingleDie(ctx, sliceOffset, localSliceBytes));
        CCU_CHK_RET(PushOwnerResultIsolated(ctx, sliceOffset, localSliceBytes));
    }
    CCU_CHK_RET(ChannelBarrier(ctx, AG_DONE_MASK));
    return CCU_SUCCESS;
}

CcuResult CcuDualGroupPushAllReduceKernel(CcuKernelArg kernelArg)
{
    auto *arg = static_cast<CcuKernelArgDirect *>(kernelArg);
    if (arg == nullptr || arg->rankSize != 16U || arg->activeDieCount != MISSION_PARTICIPANTS) {
        return CCU_E_PARA;
    }

    DirectContext ctx;
    ctx.arg = arg;
    CCU_CHK_RET(InitResources(ctx));
    CCU_CHK_RET(LoadBaseArgs(ctx));
    CCU_CHK_RET(LoadGroupArgs(ctx));

    ccu::Variable localSliceBytes;
    if (arg->myRank == arg->rankSize - 1U) {
        localSliceBytes = ctx.lastSliceBytes;
    } else {
        localSliceBytes = ctx.normalSliceBytes;
    }
    ccu::Variable sliceOffset = MulByStatic(ctx.normalSliceBytes, arg->myRank);

    CCU_IF(ctx.runtimePhase == 0)
    {
        CCU_CHK_RET(PreSyncInput(ctx));
        CCU_CHK_RET(GroupReduceOwnerPartialIsolated(ctx, sliceOffset, localSliceBytes));
        CCU_CHK_RET(ccu::EventRecord(ctx.ioEvent, 1U));
        CCU_CHK_RET(ccu::EventWait(ctx.ioEvent, 1U));
    }
    CCU_ELSE
    {
        CCU_CHK_RET(PreSyncOutput(ctx));
        CCU_CHK_RET(PushOwnerResultIsolated(ctx, sliceOffset, localSliceBytes));
        CCU_CHK_RET(ChannelBarrier(ctx, AG_DONE_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult CcuEightPlusFourDirectPushAllReduceKernel(CcuKernelArg kernelArg)
{
    auto *arg = static_cast<CcuKernelArgDirect *>(kernelArg);
    if (arg == nullptr || arg->rankSize != 12U || arg->activeDieCount != MISSION_PARTICIPANTS) {
        return CCU_E_PARA;
    }

    DirectContext ctx;
    ctx.arg = arg;
    CCU_CHK_RET(InitResources(ctx));
    CCU_CHK_RET(LoadBaseArgs(ctx));
    CCU_CHK_RET(LoadGroupArgs(ctx));

    ccu::Variable localSliceBytes;
    if (arg->myRank == arg->rankSize - 1U) {
        localSliceBytes = ctx.lastSliceBytes;
    } else {
        localSliceBytes = ctx.normalSliceBytes;
    }
    ccu::Variable sliceOffset = MulByStatic(ctx.normalSliceBytes, arg->myRank);

    CCU_IF(ctx.runtimePhase == 0)
    {
        CCU_CHK_RET(PreSyncInput(ctx));
        CCU_CHK_RET(ReadOwnerContributions(ctx, sliceOffset, localSliceBytes));
        CCU_CHK_RET(BuildPartial(ctx, sliceOffset, localSliceBytes));
        CCU_CHK_RET(ccu::EventRecord(ctx.ioEvent, 1U));
        CCU_CHK_RET(ccu::EventWait(ctx.ioEvent, 1U));
    }
    CCU_ELSE
    {
        CCU_CHK_RET(PreSyncOutput(ctx));
        CCU_CHK_RET(PushOwnerResultIsolated(ctx, sliceOffset, localSliceBytes));
        CCU_CHK_RET(ChannelBarrier(ctx, AG_DONE_MASK));
    }
    return CCU_SUCCESS;
}

// Keep every V030 common helper and exported kernel above this point in the
// same source order.  The point-16 tree is deliberately appended so the code
// layout of all established entries is not perturbed by a helper inserted in
// the common section.
namespace {

    CcuResult BuildPartialTreeLayoutSafeV041(
        DirectContext &ctx, const ccu::Variable &sliceOffset, const ccu::Variable &localSliceBytes)
    {
        const auto *arg = ctx.arg;
        if (arg->channelCount == 0U || arg->channelCount >= MAX_RANK_SIZE) {
            return CCU_E_PARA;
        }

        ccu::LocalAddr output;
        if (arg->includeLocal) {
            ccu::LocalAddr input;
            input.addr = ctx.inputAddr;
            input.addr += sliceOffset;
            input.token = ctx.inputToken;

            output.addr = ctx.outputAddr;
            output.addr += sliceOffset;
            output.token = ctx.outputToken;

            CCU_IF(ctx.inPlace == 0)
            {
                CCU_CHK_RET(ccu::LocalCopy(output, input, localSliceBytes, ctx.reduceEvent, 1U));
                CCU_CHK_RET(ccu::EventWait(ctx.reduceEvent, 1U));
            }
        }

        for (uint32_t stride = 1U; stride < arg->channelCount; stride *= 2U) {
            uint16_t roundMask = 0U;
            uint32_t eventIndex = 0U;
            for (uint32_t base = 0U; base + stride < arg->channelCount; base += stride * 2U) {
                ccu::LocalAddr destination = ScratchSlot(ctx, arg->scratchSlots[base], localSliceBytes);
                ccu::LocalAddr source = ScratchSlot(ctx, arg->scratchSlots[base + stride], localSliceBytes);
                const uint16_t eventMask = static_cast<uint16_t>(1U << eventIndex++);
                CCU_CHK_RET(ccu::LocalReduce(destination, source, localSliceBytes, HCCL_DATA_TYPE_FP32,
                    HCCL_REDUCE_SUM, ctx.reduceEvent, eventMask));
                roundMask = static_cast<uint16_t>(roundMask | eventMask);
            }
            if (roundMask != 0U) {
                CCU_CHK_RET(ccu::EventWait(ctx.reduceEvent, roundMask));
            }
        }

        if (arg->includeLocal) {
            ccu::LocalAddr remotePartial = ScratchSlot(ctx, arg->scratchSlots[0], localSliceBytes);
            CCU_CHK_RET(ccu::LocalReduce(
                output, remotePartial, localSliceBytes, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.reduceEvent, 1U));
            CCU_CHK_RET(ccu::EventWait(ctx.reduceEvent, 1U));
        }
        return CCU_SUCCESS;
    }

} // namespace

CcuResult CcuEightPlusFourTreeDirectPushAllReduceKernelV041(CcuKernelArg kernelArg)
{
    auto *arg = static_cast<CcuKernelArgDirect *>(kernelArg);
    if (arg == nullptr || arg->rankSize != 12U || arg->activeDieCount != MISSION_PARTICIPANTS) {
        return CCU_E_PARA;
    }

    DirectContext ctx;
    ctx.arg = arg;
    CCU_CHK_RET(InitResources(ctx));
    CCU_CHK_RET(LoadBaseArgs(ctx));
    CCU_CHK_RET(LoadGroupArgs(ctx));

    ccu::Variable localSliceBytes;
    if (arg->myRank == arg->rankSize - 1U) {
        localSliceBytes = ctx.lastSliceBytes;
    } else {
        localSliceBytes = ctx.normalSliceBytes;
    }
    ccu::Variable sliceOffset = MulByStatic(ctx.normalSliceBytes, arg->myRank);

    CCU_IF(ctx.runtimePhase == 0)
    {
        CCU_CHK_RET(PreSyncInput(ctx));
        CCU_CHK_RET(ReadOwnerContributions(ctx, sliceOffset, localSliceBytes));
        CCU_CHK_RET(BuildPartialTreeLayoutSafeV041(ctx, sliceOffset, localSliceBytes));
        CCU_CHK_RET(ccu::EventRecord(ctx.ioEvent, 1U));
        CCU_CHK_RET(ccu::EventWait(ctx.ioEvent, 1U));
    }
    CCU_ELSE
    {
        CCU_CHK_RET(PreSyncOutput(ctx));
        CCU_CHK_RET(PushOwnerResultIsolated(ctx, sliceOffset, localSliceBytes));
        CCU_CHK_RET(ChannelBarrier(ctx, AG_DONE_MASK));
    }
    return CCU_SUCCESS;
}

// The exact-P14 NHR implementation is appended after the complete V041
// translation-unit prefix.  This preserves the established V030 entries and
// the V041 exact-P16 entry byte-for-byte and in the same source order.
namespace {

    constexpr uint16_t P14_NHR_READY_MASK_V042 = 1U << 6;
    constexpr uint16_t P14_NHR_RS_DONE_MASK_V042 = 1U << 7;
    constexpr uint16_t P14_NHR_AG_DONE_MASK_V042 = 1U << 8;

    uint32_t ChannelIndexForP14PeerV042(const CcuKernelArgDirect *arg, uint32_t peerRank)
    {
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            if (arg->peerRanks[i] == peerRank) {
                return i;
            }
        }
        return MAX_RANK_SIZE;
    }

    ccu::Variable P14NhrSliceBytesV042(const DirectContext &ctx, uint32_t sliceId)
    {
        return sliceId == ctx.arg->rankSize - 1U ? ctx.lastSliceBytes : ctx.normalSliceBytes;
    }

    ccu::Variable P14NhrSliceOffsetV042(const DirectContext &ctx, uint32_t sliceId)
    {
        return MulByStatic(ctx.normalSliceBytes, sliceId);
    }

    CcuResult P14NhrLocalCopySlicesV042(DirectContext &ctx, const std::array<uint32_t, 2> &sliceIds)
    {
        CCU_IF(ctx.inPlace == 0)
        {
            for (uint32_t i = 0; i < sliceIds.size(); ++i) {
                ccu::Variable sliceBytes = P14NhrSliceBytesV042(ctx, sliceIds[i]);
                ccu::Variable sliceOffset = P14NhrSliceOffsetV042(ctx, sliceIds[i]);
                ccu::LocalAddr source;
                source.addr = ctx.inputAddr;
                source.addr += sliceOffset;
                source.token = ctx.inputToken;
                ccu::LocalAddr destination;
                destination.addr = ctx.outputAddr;
                destination.addr += sliceOffset;
                destination.token = ctx.outputToken;
                CCU_IF(sliceBytes != 0)
                {
                    CCU_CHK_RET(
                        ccu::LocalCopy(destination, source, sliceBytes, ctx.ioEvent, static_cast<uint16_t>(1U << i)));
                }
                CCU_ELSE
                {
                    CCU_CHK_RET(ccu::EventRecord(ctx.ioEvent, static_cast<uint16_t>(1U << i)));
                }
            }
            CCU_CHK_RET(ccu::EventWait(ctx.ioEvent, (1U << sliceIds.size()) - 1U));
        }
        return CCU_SUCCESS;
    }

    CcuResult P14NhrWriteSlicesV042(DirectContext &ctx, uint32_t toRank,
        const std::array<uint32_t, 2> &sliceIds, uint32_t sliceCount, bool reduce, bool sourceIsInput)
    {
        const uint32_t channelIndex = ChannelIndexForP14PeerV042(ctx.arg, toRank);
        if (channelIndex == MAX_RANK_SIZE || sliceCount == 0U || sliceCount > sliceIds.size()) {
            return CCU_E_PARA;
        }

        for (uint32_t i = 0; i < sliceCount; ++i) {
            ccu::Variable sliceBytes = P14NhrSliceBytesV042(ctx, sliceIds[i]);
            ccu::Variable sliceOffset = P14NhrSliceOffsetV042(ctx, sliceIds[i]);
            ccu::LocalAddr source;
            source.addr = sourceIsInput ? ctx.inputAddr : ctx.outputAddr;
            source.addr += sliceOffset;
            source.token = sourceIsInput ? ctx.inputToken : ctx.outputToken;
            ccu::RemoteAddr destination;
            destination.addr = ctx.peerOutputAddr[channelIndex];
            destination.addr += sliceOffset;
            destination.token = ctx.peerOutputToken[channelIndex];
            const uint16_t eventMask = static_cast<uint16_t>(1U << i);
            CCU_IF(sliceBytes != 0)
            {
                if (reduce) {
                    CCU_CHK_RET(ccu::WriteReduce(ctx.arg->channels[channelIndex], destination, source, sliceBytes,
                        HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.ioEvent, eventMask));
                } else {
                    CCU_CHK_RET(ccu::Write(
                        ctx.arg->channels[channelIndex], destination, source, sliceBytes, ctx.ioEvent, eventMask));
                }
            }
            CCU_ELSE
            {
                CCU_CHK_RET(ccu::EventRecord(ctx.ioEvent, eventMask));
            }
        }
        CCU_CHK_RET(ccu::EventWait(ctx.ioEvent, static_cast<uint16_t>((1U << sliceCount) - 1U)));
        return CCU_SUCCESS;
    }

    CcuResult P14NhrReduceStepV042(DirectContext &ctx, uint32_t toRank, uint32_t fromRank,
        const std::array<uint32_t, 2> &sliceIds, uint32_t sliceCount, bool sourceIsInput, bool needsReady)
    {
        const uint32_t toIndex = ChannelIndexForP14PeerV042(ctx.arg, toRank);
        const uint32_t fromIndex = ChannelIndexForP14PeerV042(ctx.arg, fromRank);
        if (toIndex == MAX_RANK_SIZE || fromIndex == MAX_RANK_SIZE) {
            return CCU_E_PARA;
        }
        if (needsReady) {
            CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[fromIndex], CKE_IDX, P14_NHR_READY_MASK_V042));
            CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[toIndex], CKE_IDX, P14_NHR_READY_MASK_V042));
        }
        CCU_CHK_RET(P14NhrWriteSlicesV042(ctx, toRank, sliceIds, sliceCount, true, sourceIsInput));
        CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[toIndex], CKE_IDX, P14_NHR_RS_DONE_MASK_V042));
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[fromIndex], CKE_IDX, P14_NHR_RS_DONE_MASK_V042));
        return CCU_SUCCESS;
    }

    CcuResult P14NhrGatherStepV042(DirectContext &ctx, uint32_t toRank, uint32_t fromRank,
        const std::array<uint32_t, 2> &sliceIds, uint32_t sliceCount)
    {
        const uint32_t toIndex = ChannelIndexForP14PeerV042(ctx.arg, toRank);
        const uint32_t fromIndex = ChannelIndexForP14PeerV042(ctx.arg, fromRank);
        if (toIndex == MAX_RANK_SIZE || fromIndex == MAX_RANK_SIZE) {
            return CCU_E_PARA;
        }
        CCU_CHK_RET(P14NhrWriteSlicesV042(ctx, toRank, sliceIds, sliceCount, false, false));
        CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[toIndex], CKE_IDX, P14_NHR_AG_DONE_MASK_V042));
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[fromIndex], CKE_IDX, P14_NHR_AG_DONE_MASK_V042));
        return CCU_SUCCESS;
    }

    CcuResult RunExactP14NhrV042(DirectContext &ctx)
    {
        if (ctx.arg->rankSize != 4U || ctx.arg->channelCount != 3U) {
            return CCU_E_PARA;
        }

        const uint32_t rank = ctx.arg->myRank;
        const uint32_t plusOne = (rank + 1U) % 4U;
        const uint32_t plusTwo = (rank + 2U) % 4U;
        const uint32_t minusOne = (rank + 3U) % 4U;
        CCU_CHK_RET(P14NhrLocalCopySlicesV042(ctx, {rank, plusTwo}));
        CCU_CHK_RET(PreSyncOutput(ctx));
        CCU_CHK_RET(P14NhrReduceStepV042(ctx, minusOne, plusOne, {minusOne, plusOne}, 2U, true, false));
        CCU_CHK_RET(P14NhrReduceStepV042(ctx, plusTwo, plusTwo, {plusTwo, 0U}, 1U, false, true));
        CCU_CHK_RET(P14NhrGatherStepV042(ctx, plusTwo, plusTwo, {rank, 0U}, 1U));
        CCU_CHK_RET(P14NhrGatherStepV042(ctx, plusOne, minusOne, {rank, plusTwo}, 2U));
        return CCU_SUCCESS;
    }

} // namespace

CcuResult CcuExactP14NhrAllReduceKernelV042(CcuKernelArg kernelArg)
{
    auto *arg = static_cast<CcuKernelArgDirect *>(kernelArg);
    DirectContext ctx;
    ctx.arg = arg;
    CCU_CHK_RET(InitResources(ctx));
    CCU_CHK_RET(LoadBaseArgs(ctx));
    return RunExactP14NhrV042(ctx);
}

namespace {
    // Exact P13-only variant of the address/token exchange. Every peer gets
    // all four variable updates before one 0xf wait, replacing the two
    // independent 0x3/0xc waits used by generic entries.
    CcuResult PreSyncInputOutputCompactV051(DirectContext &ctx)
    {
        const auto *arg = ctx.arg;
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            CCU_CHK_RET(ccu::WriteVariableWithNotify(
                arg->channels[i], ctx.inputAddr, INPUT_ADDR_XN_ID, CKE_IDX, 1U << INPUT_ADDR_XN_ID));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(
                arg->channels[i], ctx.inputToken, INPUT_TOKEN_XN_ID, CKE_IDX, 1U << INPUT_TOKEN_XN_ID));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(
                arg->channels[i], ctx.outputAddr, OUTPUT_ADDR_XN_ID, CKE_IDX, 1U << OUTPUT_ADDR_XN_ID));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(
                arg->channels[i], ctx.outputToken, OUTPUT_TOKEN_XN_ID, CKE_IDX, 1U << OUTPUT_TOKEN_XN_ID));
        }
        constexpr uint16_t allAddressMask = (1U << INPUT_ADDR_XN_ID) | (1U << INPUT_TOKEN_XN_ID)
                                            | (1U << OUTPUT_ADDR_XN_ID) | (1U << OUTPUT_TOKEN_XN_ID);
        for (uint32_t i = 0; i < arg->channelCount; ++i) {
            CCU_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_IDX, allAddressMask));
        }
        return CCU_SUCCESS;
    }
} // namespace

CcuResult CcuExactP13CompactHandshakeAllReduceKernelV051(CcuKernelArg kernelArg)
{
    auto *arg = static_cast<CcuKernelArgDirect *>(kernelArg);
    if (arg == nullptr || arg->rankSize != 4U || arg->activeDieCount != 1U || !arg->includeLocal
        || arg->channelCount != 3U) {
        return CCU_E_PARA;
    }

    DirectContext ctx;
    ctx.arg = arg;
    CCU_CHK_RET(InitResources(ctx));
    CCU_CHK_RET(LoadBaseArgs(ctx));
    CCU_CHK_RET(LoadGroupArgs(ctx));

    ccu::Variable localSliceBytes;
    if (arg->myRank == arg->rankSize - 1U) {
        localSliceBytes = ctx.lastSliceBytes;
    } else {
        localSliceBytes = ctx.normalSliceBytes;
    }
    ccu::Variable sliceOffset = MulByStatic(ctx.normalSliceBytes, arg->myRank);

    CCU_CHK_RET(PreSyncInputOutputCompactV051(ctx));
    CCU_IF(localSliceBytes != 0)
    {
        CCU_CHK_RET(GroupReduceSingleDie(ctx, sliceOffset, localSliceBytes));
        CCU_CHK_RET(PushOwnerResultIsolated(ctx, sliceOffset, localSliceBytes));
    }
    CCU_CHK_RET(ChannelBarrier(ctx, AG_DONE_MASK));
    return CCU_SUCCESS;
}

// Exact P16-only entry. It retains V041's Mesh/Clos owner shards, local tree,
// partial merge, launch count and data operations. The only changed control
// dependency is a single combined address/token exchange before phase zero.
CcuResult CcuEightPlusFourTreeDirectPushCompactPreSyncAllReduceKernelV065(CcuKernelArg kernelArg)
{
    auto *arg = static_cast<CcuKernelArgDirect *>(kernelArg);
    if (arg == nullptr || arg->rankSize != 12U || arg->activeDieCount != MISSION_PARTICIPANTS) {
        return CCU_E_PARA;
    }

    DirectContext ctx;
    ctx.arg = arg;
    CCU_CHK_RET(InitResources(ctx));
    CCU_CHK_RET(LoadBaseArgs(ctx));
    CCU_CHK_RET(LoadGroupArgs(ctx));

    ccu::Variable localSliceBytes;
    if (arg->myRank == arg->rankSize - 1U) {
        localSliceBytes = ctx.lastSliceBytes;
    } else {
        localSliceBytes = ctx.normalSliceBytes;
    }
    ccu::Variable sliceOffset = MulByStatic(ctx.normalSliceBytes, arg->myRank);

    CCU_IF(ctx.runtimePhase == 0)
    {
        // V051's 0xf wait publishes and consumes every address/token pair in
        // this phase. Phase one then needs no second output-address wait.
        CCU_CHK_RET(PreSyncInputOutputCompactV051(ctx));
        CCU_CHK_RET(ReadOwnerContributions(ctx, sliceOffset, localSliceBytes));
        CCU_CHK_RET(BuildPartialTreeLayoutSafeV041(ctx, sliceOffset, localSliceBytes));
        CCU_CHK_RET(ccu::EventRecord(ctx.ioEvent, 1U));
        CCU_CHK_RET(ccu::EventWait(ctx.ioEvent, 1U));
    }
    CCU_ELSE
    {
        CCU_CHK_RET(PushOwnerResultIsolated(ctx, sliceOffset, localSliceBytes));
        CCU_CHK_RET(ChannelBarrier(ctx, AG_DONE_MASK));
    }
    return CCU_SUCCESS;
}
// Exact P11 helper appended after all existing entries. It preserves the two
// SelectWindow payloads and five host launches, replacing only local serial
// reduction with a tree inside each owner shard.
namespace {

    CcuResult BuildPartialTreeExactP11V067(
        DirectContext &ctx, const ccu::Variable &sliceOffset, const ccu::Variable &localSliceBytes)
    {
        const auto *arg = ctx.arg;
        if (arg->channelCount == 0U || arg->channelCount >= MAX_RANK_SIZE) {
            return CCU_E_PARA;
        }

        ccu::LocalAddr output;
        if (arg->includeLocal) {
            ccu::LocalAddr input;
            input.addr = ctx.inputAddr;
            input.addr += sliceOffset;
            input.token = ctx.inputToken;
            output.addr = ctx.outputAddr;
            output.addr += sliceOffset;
            output.token = ctx.outputToken;
            CCU_IF(ctx.inPlace == 0)
            {
                CCU_CHK_RET(ccu::LocalCopy(output, input, localSliceBytes, ctx.reduceEvent, 1U));
                CCU_CHK_RET(ccu::EventWait(ctx.reduceEvent, 1U));
            }
        }

        for (uint32_t stride = 1U; stride < arg->channelCount; stride *= 2U) {
            uint16_t roundMask = 0U;
            uint32_t eventIndex = 0U;
            for (uint32_t base = 0U; base + stride < arg->channelCount; base += stride * 2U) {
                ccu::LocalAddr destination = ScratchSlot(ctx, arg->scratchSlots[base], localSliceBytes);
                ccu::LocalAddr source = ScratchSlot(ctx, arg->scratchSlots[base + stride], localSliceBytes);
                const uint16_t eventMask = static_cast<uint16_t>(1U << eventIndex++);
                CCU_CHK_RET(ccu::LocalReduce(destination, source, localSliceBytes, HCCL_DATA_TYPE_FP32,
                    HCCL_REDUCE_SUM, ctx.reduceEvent, eventMask));
                roundMask = static_cast<uint16_t>(roundMask | eventMask);
            }
            if (roundMask != 0U) {
                CCU_CHK_RET(ccu::EventWait(ctx.reduceEvent, roundMask));
            }
        }

        if (arg->includeLocal) {
            ccu::LocalAddr remotePartial = ScratchSlot(ctx, arg->scratchSlots[0], localSliceBytes);
            CCU_CHK_RET(ccu::LocalReduce(
                output, remotePartial, localSliceBytes, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.reduceEvent, 1U));
            CCU_CHK_RET(ccu::EventWait(ctx.reduceEvent, 1U));
        }
        return CCU_SUCCESS;
    }

} // namespace

CcuResult CcuTwoByEightTreeDirectPushAllReduceKernelV067(CcuKernelArg kernelArg)
{
    auto *arg = static_cast<CcuKernelArgDirect *>(kernelArg);
    if (arg == nullptr || arg->rankSize != 16U || arg->activeDieCount != MISSION_PARTICIPANTS) {
        return CCU_E_PARA;
    }

    DirectContext ctx;
    ctx.arg = arg;
    CCU_CHK_RET(InitResources(ctx));
    CCU_CHK_RET(LoadBaseArgs(ctx));
    CCU_CHK_RET(LoadGroupArgs(ctx));

    ccu::Variable localSliceBytes;
    if (arg->myRank == arg->rankSize - 1U) {
        localSliceBytes = ctx.lastSliceBytes;
    } else {
        localSliceBytes = ctx.normalSliceBytes;
    }
    ccu::Variable sliceOffset = MulByStatic(ctx.normalSliceBytes, arg->myRank);

    CCU_IF(ctx.runtimePhase == 0)
    {
        CCU_CHK_RET(PreSyncInput(ctx));
        CCU_CHK_RET(ReadOwnerContributions(ctx, sliceOffset, localSliceBytes));
        CCU_CHK_RET(BuildPartialTreeExactP11V067(ctx, sliceOffset, localSliceBytes));
        CCU_CHK_RET(ccu::EventRecord(ctx.ioEvent, 1U));
        CCU_CHK_RET(ccu::EventWait(ctx.ioEvent, 1U));
    }
    CCU_ELSE
    {
        CCU_CHK_RET(PreSyncOutput(ctx));
        CCU_CHK_RET(PushOwnerResultIsolated(ctx, sliceOffset, localSliceBytes));
        CCU_CHK_RET(ChannelBarrier(ctx, AG_DONE_MASK));
    }
    return CCU_SUCCESS;
}
// Append the exact-P12 implementation after the complete V067
// translation-unit prefix.  Keep a dedicated copy of the tree helper instead
// of sharing V041's helper: sharing changes the compiler's inlining decision
// and perturbs the already-proven exact-P16 kernel layout.
namespace {

    CcuResult BuildPartialTreeExactP12V068(
        DirectContext &ctx, const ccu::Variable &sliceOffset, const ccu::Variable &localSliceBytes)
    {
        const auto *arg = ctx.arg;
        if (arg->channelCount == 0U || arg->channelCount >= MAX_RANK_SIZE) {
            return CCU_E_PARA;
        }

        ccu::LocalAddr output;
        if (arg->includeLocal) {
            ccu::LocalAddr input;
            input.addr = ctx.inputAddr;
            input.addr += sliceOffset;
            input.token = ctx.inputToken;

            output.addr = ctx.outputAddr;
            output.addr += sliceOffset;
            output.token = ctx.outputToken;

            CCU_IF(ctx.inPlace == 0)
            {
                CCU_CHK_RET(ccu::LocalCopy(output, input, localSliceBytes, ctx.reduceEvent, 1U));
                CCU_CHK_RET(ccu::EventWait(ctx.reduceEvent, 1U));
            }
        }

        for (uint32_t stride = 1U; stride < arg->channelCount; stride *= 2U) {
            uint16_t roundMask = 0U;
            uint32_t eventIndex = 0U;
            for (uint32_t base = 0U; base + stride < arg->channelCount; base += stride * 2U) {
                ccu::LocalAddr destination = ScratchSlot(ctx, arg->scratchSlots[base], localSliceBytes);
                ccu::LocalAddr source = ScratchSlot(ctx, arg->scratchSlots[base + stride], localSliceBytes);
                const uint16_t eventMask = static_cast<uint16_t>(1U << eventIndex++);
                CCU_CHK_RET(ccu::LocalReduce(destination, source, localSliceBytes, HCCL_DATA_TYPE_FP32,
                    HCCL_REDUCE_SUM, ctx.reduceEvent, eventMask));
                roundMask = static_cast<uint16_t>(roundMask | eventMask);
            }
            if (roundMask != 0U) {
                CCU_CHK_RET(ccu::EventWait(ctx.reduceEvent, roundMask));
            }
        }

        if (arg->includeLocal) {
            ccu::LocalAddr remotePartial = ScratchSlot(ctx, arg->scratchSlots[0], localSliceBytes);
            CCU_CHK_RET(ccu::LocalReduce(
                output, remotePartial, localSliceBytes, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.reduceEvent, 1U));
            CCU_CHK_RET(ccu::EventWait(ctx.reduceEvent, 1U));
        }
        return CCU_SUCCESS;
    }

} // namespace

CcuResult CcuTwoByEightTreeDirectPushAllReduceKernelV068(CcuKernelArg kernelArg)
{
    auto *arg = static_cast<CcuKernelArgDirect *>(kernelArg);
    if (arg == nullptr || arg->rankSize != 16U || arg->activeDieCount != MISSION_PARTICIPANTS) {
        return CCU_E_PARA;
    }

    DirectContext ctx;
    ctx.arg = arg;
    CCU_CHK_RET(InitResources(ctx));
    CCU_CHK_RET(LoadBaseArgs(ctx));
    // Retain the proven 14-argument dual-die launch ABI.  The four loop
    // arguments are deliberately unused by the coarse full-slice tree.
    CCU_CHK_RET(LoadGroupArgs(ctx));

    ccu::Variable localSliceBytes;
    if (arg->myRank == arg->rankSize - 1U) {
        localSliceBytes = ctx.lastSliceBytes;
    } else {
        localSliceBytes = ctx.normalSliceBytes;
    }
    ccu::Variable sliceOffset = MulByStatic(ctx.normalSliceBytes, arg->myRank);

    CCU_IF(ctx.runtimePhase == 0)
    {
        CCU_CHK_RET(PreSyncInput(ctx));
        CCU_CHK_RET(ReadOwnerContributions(ctx, sliceOffset, localSliceBytes));
        CCU_CHK_RET(BuildPartialTreeExactP12V068(ctx, sliceOffset, localSliceBytes));
        CCU_CHK_RET(ccu::EventRecord(ctx.ioEvent, 1U));
        CCU_CHK_RET(ccu::EventWait(ctx.ioEvent, 1U));
    }
    CCU_ELSE
    {
        CCU_CHK_RET(PreSyncOutput(ctx));
        CCU_CHK_RET(PushOwnerResultIsolated(ctx, sliceOffset, localSliceBytes));
        CCU_CHK_RET(ChannelBarrier(ctx, AG_DONE_MASK));
    }
    return CCU_SUCCESS;
}


} // namespace ops_hccl
