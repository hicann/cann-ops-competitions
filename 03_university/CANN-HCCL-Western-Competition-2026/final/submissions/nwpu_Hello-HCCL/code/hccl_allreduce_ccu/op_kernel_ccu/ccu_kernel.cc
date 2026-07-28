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

namespace ccu = ::AscendC::ccu;

#define CCU_CHK_RET(call)                         \
    do {                                          \
        const CcuResult ccuRet = (call);          \
        if (ccuRet != CcuResult::CCU_SUCCESS) {   \
            return ccuRet;                        \
        }                                         \
    } while (0)

namespace ops_hccl {
namespace {

constexpr uint32_t INPUT_ADDR_XN = 0;
constexpr uint32_t OUTPUT_ADDR_XN = 1;
constexpr uint32_t INPUT_TOKEN_XN = 2;
constexpr uint32_t OUTPUT_TOKEN_XN = 3;
constexpr uint32_t CHANNEL_NOTIFY_IDX = 0;
constexpr uint16_t INPUT_ADDR_BIT = uint16_t{1} << INPUT_ADDR_XN;
constexpr uint16_t OUTPUT_ADDR_BIT = uint16_t{1} << OUTPUT_ADDR_XN;
constexpr uint16_t INPUT_TOKEN_BIT = uint16_t{1} << INPUT_TOKEN_XN;
constexpr uint16_t OUTPUT_TOKEN_BIT = uint16_t{1} << OUTPUT_TOKEN_XN;
constexpr uint16_t ADDRESS_READY_MASK =
    INPUT_ADDR_BIT | OUTPUT_ADDR_BIT | INPUT_TOKEN_BIT | OUTPUT_TOKEN_BIT;
constexpr uint16_t FINISH_BIT = uint16_t{1} << 4;

uint16_t CountMask(uint32_t count)
{
    return count == 16 ? static_cast<uint16_t>(0xFFFFu)
                       : static_cast<uint16_t>((uint32_t{1} << count) - 1);
}

struct KernelContext {
    const AllReduceKernelArg *arg = nullptr;
    ccu::Variable myInput;
    ccu::Variable myOutput;
    ccu::Variable myInputToken;
    ccu::Variable myOutputToken;
    ccu::Variable scratchAddr;
    ccu::Variable scratchToken;
    ccu::Variable mySliceOffset;
    ccu::Variable mySliceBytes;
    ccu::Variable inPlace;
    ccu::Variable needsHandshake;
    ccu::Variable phase;
    std::vector<ccu::Variable> peerInput;
    std::vector<ccu::Variable> peerOutput;
    std::vector<ccu::Variable> peerInputToken;
    std::vector<ccu::Variable> peerOutputToken;
};

ccu::LocalAddr LocalAddressWithOffset(ccu::Variable base, ccu::Variable token,
    ccu::Variable unitOffset, uint32_t units)
{
    ccu::LocalAddr result;
    result.addr = base;
    result.token = token;
    for (uint32_t idx = 0; idx < units; ++idx) {
        result.addr += unitOffset;
    }
    return result;
}

ccu::LocalAddr LocalOutputSlice(const KernelContext &ctx)
{
    ccu::LocalAddr output;
    output.addr = ctx.myOutput;
    output.addr += ctx.mySliceOffset;
    output.token = ctx.myOutputToken;
    return output;
}

ccu::LocalAddr LocalInputSlice(const KernelContext &ctx)
{
    ccu::LocalAddr input;
    input.addr = ctx.myInput;
    input.addr += ctx.mySliceOffset;
    input.token = ctx.myInputToken;
    return input;
}

CcuResult InitContext(KernelContext &ctx)
{
    const AllReduceKernelArg &arg = *ctx.arg;
    ctx.peerInput.resize(arg.channelCount);
    ctx.peerOutput.resize(arg.channelCount);
    ctx.peerInputToken.resize(arg.channelCount);
    ctx.peerOutputToken.resize(arg.channelCount);
    for (uint32_t idx = 0; idx < arg.channelCount; ++idx) {
        ctx.peerInput[idx] =
            ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], INPUT_ADDR_XN);
        ctx.peerOutput[idx] =
            ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], OUTPUT_ADDR_XN);
        ctx.peerInputToken[idx] =
            ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], INPUT_TOKEN_XN);
        ctx.peerOutputToken[idx] =
            ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], OUTPUT_TOKEN_XN);
    }

    uint32_t taskArg = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.myInput, taskArg++));
    CCU_CHK_RET(ccu::LoadArg(ctx.myOutput, taskArg++));
    CCU_CHK_RET(ccu::LoadArg(ctx.myInputToken, taskArg++));
    CCU_CHK_RET(ccu::LoadArg(ctx.myOutputToken, taskArg++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratchAddr, taskArg++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratchToken, taskArg++));
    CCU_CHK_RET(ccu::LoadArg(ctx.mySliceOffset, taskArg++));
    CCU_CHK_RET(ccu::LoadArg(ctx.mySliceBytes, taskArg++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inPlace, taskArg++));
    CCU_CHK_RET(ccu::LoadArg(ctx.needsHandshake, taskArg++));
    CCU_CHK_RET(ccu::LoadArg(ctx.phase, taskArg++));
    (void)ctx.inPlace;
    return CCU_SUCCESS;
}

CcuResult PublishAddresses(const KernelContext &ctx)
{
    const AllReduceKernelArg &arg = *ctx.arg;
    for (uint32_t idx = 0; idx < arg.channelCount; ++idx) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[idx],
            ctx.myInput, INPUT_ADDR_XN, CHANNEL_NOTIFY_IDX, INPUT_ADDR_BIT));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[idx],
            ctx.myOutput, OUTPUT_ADDR_XN, CHANNEL_NOTIFY_IDX, OUTPUT_ADDR_BIT));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[idx],
            ctx.myInputToken, INPUT_TOKEN_XN, CHANNEL_NOTIFY_IDX, INPUT_TOKEN_BIT));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[idx],
            ctx.myOutputToken, OUTPUT_TOKEN_XN, CHANNEL_NOTIFY_IDX, OUTPUT_TOKEN_BIT));
    }
    return CCU_SUCCESS;
}

CcuResult WaitForAddresses(const KernelContext &ctx)
{
    const AllReduceKernelArg &arg = *ctx.arg;
    for (uint32_t idx = 0; idx < arg.channelCount; ++idx) {
        CCU_CHK_RET(ccu::NotifyWait(
            arg.channels[idx], CHANNEL_NOTIFY_IDX, ADDRESS_READY_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult PairwiseLocalReduce(KernelContext &ctx, std::vector<ccu::LocalAddr> &input)
{
    const AllReduceKernelArg &arg = *ctx.arg;
    std::vector<ccu::LocalAddr> live = input;
    while (live.size() > 1) {
        const uint32_t liveCount = static_cast<uint32_t>(live.size());
        const uint32_t pairCount = liveCount / 2;
        const bool hasOddTail = (liveCount % 2) != 0;
        ccu::Event levelEvent;
        for (uint32_t pairIdx = 0; pairIdx < pairCount; ++pairIdx) {
            CCU_CHK_RET(ccu::LocalReduce(live[pairIdx],
                live[pairCount + pairIdx], ctx.mySliceBytes, arg.dataType,
                arg.reduceType, levelEvent,
                static_cast<uint16_t>(uint32_t{1} << pairIdx)));
        }
        CCU_CHK_RET(ccu::EventWait(levelEvent, CountMask(pairCount)));

        std::vector<ccu::LocalAddr> next(
            live.begin(), live.begin() + pairCount);
        if (hasOddTail) {
            next.push_back(live.back());
        }
        live.swap(next);
    }
    return CCU_SUCCESS;
}

CcuResult ReduceGroupPartial(KernelContext &ctx)
{
    const AllReduceKernelArg &arg = *ctx.arg;
    CCU_IF(ctx.mySliceBytes != 0)
    {
        std::vector<ccu::LocalAddr> scratch(arg.sourceCount);
        ccu::LocalAddr nextScratch;
        nextScratch.addr = ctx.scratchAddr;
        nextScratch.token = ctx.scratchToken;
        for (uint32_t slot = 0; slot < arg.scratchBaseSlot; ++slot) {
            nextScratch.addr += ctx.mySliceBytes;
        }
        for (uint32_t sourceIdx = 0; sourceIdx < arg.channelCount; ++sourceIdx) {
            scratch[sourceIdx].addr = nextScratch.addr;
            scratch[sourceIdx].token = nextScratch.token;
            nextScratch.addr += ctx.mySliceBytes;
        }

        ccu::Event readEvent;
        for (uint32_t channelIdx = 0; channelIdx < arg.channelCount; ++channelIdx) {
            ccu::RemoteAddr remoteInput;
            remoteInput.addr = ctx.peerInput[channelIdx];
            remoteInput.addr += ctx.mySliceOffset;
            remoteInput.token = ctx.peerInputToken[channelIdx];
            const uint16_t mask =
                static_cast<uint16_t>(uint32_t{1} << channelIdx);
            CCU_CHK_RET(ccu::Read(arg.channels[channelIdx], scratch[channelIdx],
                remoteInput, ctx.mySliceBytes, readEvent, mask));
        }
        if (arg.includesLocalRank != 0) {
            const uint32_t localSourceIndex = arg.channelCount;
            scratch[localSourceIndex].addr = ctx.myInput;
            scratch[localSourceIndex].addr += ctx.mySliceOffset;
            scratch[localSourceIndex].token = ctx.myInputToken;
        }
        CCU_CHK_RET(ccu::EventWait(readEvent, CountMask(arg.channelCount)));
        CCU_CHK_RET(PairwiseLocalReduce(ctx, scratch));
    }
    return CCU_SUCCESS;
}

CcuResult ReduceGroupPartialOutputRoot(KernelContext &ctx, bool outputRootUsesLocalInput)
{
    const AllReduceKernelArg &arg = *ctx.arg;
    CCU_IF(ctx.mySliceBytes != 0)
    {
        std::vector<ccu::LocalAddr> reductionInputs(arg.sourceCount);
        const ccu::LocalAddr output = LocalOutputSlice(ctx);
        reductionInputs[0].addr = output.addr;
        reductionInputs[0].token = output.token;
        ccu::Event readEvent;
        if (outputRootUsesLocalInput) {
            ccu::LocalAddr nextScratch;
            nextScratch.addr = ctx.scratchAddr;
            nextScratch.token = ctx.scratchToken;
            for (uint32_t slot = 0; slot < arg.scratchBaseSlot; ++slot) {
                nextScratch.addr += ctx.mySliceBytes;
            }
            for (uint32_t channelIdx = 0; channelIdx < arg.channelCount; ++channelIdx) {
                reductionInputs[channelIdx + 1].addr = nextScratch.addr;
                reductionInputs[channelIdx + 1].token = nextScratch.token;
                ccu::RemoteAddr remoteInput;
                remoteInput.addr = ctx.peerInput[channelIdx];
                remoteInput.addr += ctx.mySliceOffset;
                remoteInput.token = ctx.peerInputToken[channelIdx];
                CCU_CHK_RET(ccu::Read(arg.channels[channelIdx],
                    reductionInputs[channelIdx + 1], remoteInput, ctx.mySliceBytes,
                    readEvent, static_cast<uint16_t>(uint32_t{1} << channelIdx)));
                nextScratch.addr += ctx.mySliceBytes;
            }
        } else {
            ccu::RemoteAddr rootRemoteInput;
            rootRemoteInput.addr = ctx.peerInput[0];
            rootRemoteInput.addr += ctx.mySliceOffset;
            rootRemoteInput.token = ctx.peerInputToken[0];
            CCU_CHK_RET(ccu::Read(arg.channels[0], reductionInputs[0],
                rootRemoteInput, ctx.mySliceBytes, readEvent, 1));

            ccu::LocalAddr nextScratch;
            nextScratch.addr = ctx.scratchAddr;
            nextScratch.token = ctx.scratchToken;
            for (uint32_t slot = 0; slot < arg.scratchBaseSlot; ++slot) {
                nextScratch.addr += ctx.mySliceBytes;
            }
            for (uint32_t channelIdx = 1; channelIdx < arg.channelCount; ++channelIdx) {
                reductionInputs[channelIdx].addr = nextScratch.addr;
                reductionInputs[channelIdx].token = nextScratch.token;
                ccu::RemoteAddr remoteInput;
                remoteInput.addr = ctx.peerInput[channelIdx];
                remoteInput.addr += ctx.mySliceOffset;
                remoteInput.token = ctx.peerInputToken[channelIdx];
                CCU_CHK_RET(ccu::Read(arg.channels[channelIdx],
                    reductionInputs[channelIdx], remoteInput, ctx.mySliceBytes,
                    readEvent, static_cast<uint16_t>(uint32_t{1} << channelIdx)));
                nextScratch.addr += ctx.mySliceBytes;
            }
            const ccu::LocalAddr localInput = LocalInputSlice(ctx);
            reductionInputs[arg.channelCount].addr = localInput.addr;
            reductionInputs[arg.channelCount].token = localInput.token;
        }
        CCU_CHK_RET(ccu::EventWait(readEvent, CountMask(arg.channelCount)));
        CCU_CHK_RET(PairwiseLocalReduce(ctx, reductionInputs));
    }
    return CCU_SUCCESS;
}

CcuResult CombineGroupPartialsOutputRoot(KernelContext &ctx)
{
    const AllReduceKernelArg &arg = *ctx.arg;
    CCU_IF(ctx.mySliceBytes != 0)
    {
        ccu::LocalAddr output = LocalOutputSlice(ctx);
        ccu::Event reduceEvent;
        for (uint32_t groupIdx = 0; groupIdx < arg.groupCount; ++groupIdx) {
            if (groupIdx == arg.groupIndex) {
                continue;
            }
            ccu::LocalAddr nextPartial = LocalAddressWithOffset(ctx.scratchAddr,
                ctx.scratchToken, ctx.mySliceBytes, arg.partialSlotIndices[groupIdx]);
            CCU_CHK_RET(ccu::LocalReduce(output, nextPartial, ctx.mySliceBytes,
                arg.dataType, arg.reduceType, reduceEvent, 1));
            CCU_CHK_RET(ccu::EventWait(reduceEvent, 1));
        }
    }
    return CCU_SUCCESS;
}

CcuResult BroadcastAndRecord(KernelContext &ctx)
{
    const AllReduceKernelArg &arg = *ctx.arg;
    CCU_IF(ctx.mySliceBytes != 0)
    {
        ccu::LocalAddr localOutput = LocalOutputSlice(ctx);
        ccu::Event writeEvent;
        for (uint32_t idx = 0; idx < arg.channelCount; ++idx) {
            ccu::RemoteAddr remoteOutput;
            remoteOutput.addr = ctx.peerOutput[idx];
            remoteOutput.addr += ctx.mySliceOffset;
            remoteOutput.token = ctx.peerOutputToken[idx];
            CCU_CHK_RET(ccu::Write(arg.channels[idx], remoteOutput, localOutput,
                ctx.mySliceBytes, writeEvent,
                static_cast<uint16_t>(uint32_t{1} << idx)));
        }
        CCU_CHK_RET(ccu::EventWait(writeEvent, CountMask(arg.channelCount)));
    }
    for (uint32_t idx = 0; idx < arg.channelCount; ++idx) {
        CCU_CHK_RET(ccu::NotifyRecord(
            arg.channels[idx], CHANNEL_NOTIFY_IDX, FINISH_BIT));
    }
    return CCU_SUCCESS;
}

CcuResult WaitForFinish(const KernelContext &ctx)
{
    const AllReduceKernelArg &arg = *ctx.arg;
    for (uint32_t idx = 0; idx < arg.channelCount; ++idx) {
        CCU_CHK_RET(ccu::NotifyWait(
            arg.channels[idx], CHANNEL_NOTIFY_IDX, FINISH_BIT));
    }
    return CCU_SUCCESS;
}

constexpr uint16_t SMALL4_READ_DONE_BIT = uint16_t{1} << 5;

struct Small4KernelContext {
    const Small4KernelArg *arg = nullptr;
    ccu::Variable inputAddr;
    ccu::Variable outputAddr;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable scratchAddr;
    ccu::Variable scratchToken;
    ccu::Variable bytes;
    std::vector<ccu::Variable> peerInput;
    std::vector<ccu::Variable> peerInputToken;
};

CcuResult InitSmall4Context(Small4KernelContext &ctx)
{
    const Small4KernelArg &arg = *ctx.arg;
    ctx.peerInput.resize(arg.channelCount);
    ctx.peerInputToken.resize(arg.channelCount);
    for (uint32_t idx = 0; idx < arg.channelCount; ++idx) {
        ctx.peerInput[idx] =
            ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], INPUT_ADDR_XN);
        ctx.peerInputToken[idx] =
            ccu::GetResByChannel<ccu::Variable>(arg.channels[idx], INPUT_TOKEN_XN);
    }
    uint32_t taskArg = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.inputAddr, taskArg++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputAddr, taskArg++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputToken, taskArg++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputToken, taskArg++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratchAddr, taskArg++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratchToken, taskArg++));
    CCU_CHK_RET(ccu::LoadArg(ctx.bytes, taskArg++));
    return CCU_SUCCESS;
}

CcuResult PublishSmall4Input(const Small4KernelContext &ctx)
{
    const Small4KernelArg &arg = *ctx.arg;
    constexpr uint16_t inputReadyMask = INPUT_ADDR_BIT | INPUT_TOKEN_BIT;
    for (uint32_t idx = 0; idx < arg.channelCount; ++idx) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[idx],
            ctx.inputAddr, INPUT_ADDR_XN, CHANNEL_NOTIFY_IDX, INPUT_ADDR_BIT));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg.channels[idx],
            ctx.inputToken, INPUT_TOKEN_XN, CHANNEL_NOTIFY_IDX, INPUT_TOKEN_BIT));
    }
    for (uint32_t idx = 0; idx < arg.channelCount; ++idx) {
        CCU_CHK_RET(ccu::NotifyWait(
            arg.channels[idx], CHANNEL_NOTIFY_IDX, inputReadyMask));
    }
    return CCU_SUCCESS;
}

CcuResult RunSmall4AllPairs(Small4KernelContext &ctx)
{
    const Small4KernelArg &arg = *ctx.arg;
    CCU_IF(ctx.bytes != 0)
    {
        std::vector<ccu::LocalAddr> remoteCopies(arg.channelCount);
        ccu::LocalAddr output;
        output.addr = ctx.outputAddr;
        output.token = ctx.outputToken;
        ccu::LocalAddr nextScratch;
        nextScratch.addr = ctx.scratchAddr;
        nextScratch.token = ctx.scratchToken;

        ccu::Event readEvent;
        for (uint32_t idx = 0; idx < arg.channelCount; ++idx) {
            if (arg.inPlace == 0 && idx == 0) {
                remoteCopies[idx] = output;
            } else {
                remoteCopies[idx] = nextScratch;
                nextScratch.addr += ctx.bytes;
            }
            ccu::RemoteAddr remoteInput;
            remoteInput.addr = ctx.peerInput[idx];
            remoteInput.token = ctx.peerInputToken[idx];
            CCU_CHK_RET(ccu::Read(arg.channels[idx], remoteCopies[idx],
                remoteInput, ctx.bytes, readEvent,
                static_cast<uint16_t>(uint32_t{1} << idx)));
        }
        CCU_CHK_RET(ccu::EventWait(readEvent, CountMask(arg.channelCount)));

        // All ranks finish reading before any in-place rank overwrites input.
        for (uint32_t idx = 0; idx < arg.channelCount; ++idx) {
            CCU_CHK_RET(ccu::NotifyRecord(
                arg.channels[idx], CHANNEL_NOTIFY_IDX, SMALL4_READ_DONE_BIT));
        }
        for (uint32_t idx = 0; idx < arg.channelCount; ++idx) {
            CCU_CHK_RET(ccu::NotifyWait(
                arg.channels[idx], CHANNEL_NOTIFY_IDX, SMALL4_READ_DONE_BIT));
        }

        ccu::LocalAddr a = output;
        ccu::LocalAddr b;
        ccu::LocalAddr c;
        ccu::LocalAddr d;
        if (arg.inPlace != 0) {
            b = remoteCopies[0];
            c = remoteCopies[1];
            d = remoteCopies[2];
        } else {
            b = remoteCopies[1];
            c = remoteCopies[2];
            d.addr = ctx.inputAddr;
            d.token = ctx.inputToken;
        }

        ccu::Event firstLevel;
        CCU_CHK_RET(ccu::LocalReduce(a, c, ctx.bytes, arg.dataType,
            arg.reduceType, firstLevel, 1));
        CCU_CHK_RET(ccu::LocalReduce(b, d, ctx.bytes, arg.dataType,
            arg.reduceType, firstLevel, 2));
        CCU_CHK_RET(ccu::EventWait(firstLevel, 3));
        ccu::Event secondLevel;
        CCU_CHK_RET(ccu::LocalReduce(a, b, ctx.bytes, arg.dataType,
            arg.reduceType, secondLevel, 1));
        CCU_CHK_RET(ccu::EventWait(secondLevel, 1));
    }
    return CCU_SUCCESS;
}

} // namespace

CcuResult CcuPhasedMeshAllReduceKernel(CcuKernelArg kernelArg)
{
    auto *arg = static_cast<AllReduceKernelArg *>(kernelArg);
    if (arg == nullptr || arg->rankSize < 2 || arg->rankSize > MAX_RANK_SIZE
        || arg->rankId >= arg->rankSize || arg->channelCount == 0
        || arg->channelCount >= arg->rankSize || arg->groupCount == 0
        || arg->groupCount > MAX_CCU_DIE_GROUPS || arg->groupIndex >= arg->groupCount
        || arg->includesLocalRank > 1
        || arg->sourceCount != arg->channelCount + arg->includesLocalRank
        || arg->sourceCount == 0 || arg->sourceCount > MAX_RANK_SIZE
        || arg->scratchBaseSlot + arg->sourceCount > arg->rankSize
        || (arg->groupCount == 1 && arg->includesLocalRank == 0)
        || arg->outputRootUsesLocalInput > 1) {
        return CCU_E_PARA;
    }
    for (uint32_t idx = 0; idx < arg->channelCount; ++idx) {
        if (arg->remoteRanks[idx] >= arg->rankSize
            || arg->remoteRanks[idx] == arg->rankId) {
            return CCU_E_PARA;
        }
        for (uint32_t previous = 0; previous < idx; ++previous) {
            if (arg->remoteRanks[previous] == arg->remoteRanks[idx]) {
                return CCU_E_PARA;
            }
        }
    }
    if (arg->scratchBaseSlot + arg->channelCount > arg->rankSize - 1) {
        return CCU_E_PARA;
    }
    for (uint32_t idx = 0; idx < arg->groupCount; ++idx) {
        if (arg->partialSlotIndices[idx] >= arg->rankSize - 1) {
            return CCU_E_PARA;
        }
    }

    KernelContext ctx;
    ctx.arg = arg;
    CCU_CHK_RET(InitContext(ctx));

    if (arg->groupCount == 1) {
        CCU_IF(ctx.phase == MESH_PHASE_SINGLE_DIE_ALL)
        {
            CCU_IF(ctx.needsHandshake != 0)
            {
                CCU_CHK_RET(PublishAddresses(ctx));
                CCU_CHK_RET(WaitForAddresses(ctx));
            }
            CCU_CHK_RET(ReduceGroupPartialOutputRoot(ctx, arg->outputRootUsesLocalInput != 0));
            CCU_CHK_RET(BroadcastAndRecord(ctx));
            CCU_CHK_RET(WaitForFinish(ctx));
        }
    } else {
        CCU_IF(ctx.phase == MESH_PHASE_PUBLISH)
        {
            CCU_CHK_RET(PublishAddresses(ctx));
        }
        CCU_IF(ctx.phase == MESH_PHASE_REDUCE_PARTIAL)
        {
            CCU_IF(ctx.needsHandshake != 0)
            {
                CCU_CHK_RET(WaitForAddresses(ctx));
            }
            if (arg->includesLocalRank != 0) {
                CCU_CHK_RET(ReduceGroupPartialOutputRoot(ctx, arg->outputRootUsesLocalInput != 0));
            } else {
                CCU_CHK_RET(ReduceGroupPartial(ctx));
            }
        }
        if (arg->groupIndex == 0) {
            CCU_IF(ctx.phase == MESH_PHASE_COMBINE)
            {
                CCU_CHK_RET(CombineGroupPartialsOutputRoot(ctx));
            }
        }
        CCU_IF(ctx.phase == MESH_PHASE_FINISH)
        {
            CCU_CHK_RET(BroadcastAndRecord(ctx));
            CCU_CHK_RET(WaitForFinish(ctx));
        }
    }
    return CCU_SUCCESS;
}

CcuResult CcuSingleDieOutputRootAllReduceKernel(CcuKernelArg kernelArg)
{
    auto *outputRootArg = static_cast<OutputRootKernelArg *>(kernelArg);
    if (outputRootArg == nullptr) {
        return CCU_E_PARA;
    }
    AllReduceKernelArg *arg = outputRootArg;
    if (arg->rankSize < 2 || arg->rankSize > MAX_RANK_SIZE
        || arg->rankId >= arg->rankSize || arg->channelCount == 0
        || arg->channelCount >= arg->rankSize || arg->groupCount != 1
        || arg->groupIndex != 0 || arg->includesLocalRank != 1
        || arg->sourceCount != arg->channelCount + 1
        || arg->sourceCount == 0 || arg->sourceCount > MAX_RANK_SIZE
        || arg->scratchBaseSlot + arg->channelCount > arg->rankSize - 1
        || outputRootArg->outputRootUsesLocalInput > 1) {
        return CCU_E_PARA;
    }
    for (uint32_t idx = 0; idx < arg->channelCount; ++idx) {
        if (arg->remoteRanks[idx] >= arg->rankSize
            || arg->remoteRanks[idx] == arg->rankId) {
            return CCU_E_PARA;
        }
        for (uint32_t previous = 0; previous < idx; ++previous) {
            if (arg->remoteRanks[previous] == arg->remoteRanks[idx]) {
                return CCU_E_PARA;
            }
        }
    }

    KernelContext ctx;
    ctx.arg = arg;
    CCU_CHK_RET(InitContext(ctx));
    CCU_CHK_RET(PublishAddresses(ctx));
    CCU_CHK_RET(WaitForAddresses(ctx));
    CCU_CHK_RET(ReduceGroupPartialOutputRoot(ctx,
        outputRootArg->outputRootUsesLocalInput != 0));
    CCU_CHK_RET(BroadcastAndRecord(ctx));
    CCU_CHK_RET(WaitForFinish(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuPhasedMeshAllReduceSequentialScratchReuseKernel(CcuKernelArg kernelArg)
{
    auto *sequentialArg = static_cast<SequentialScratchReuseKernelArg *>(kernelArg);
    if (sequentialArg == nullptr) {
        return CCU_E_PARA;
    }
    AllReduceKernelArg *arg = sequentialArg;
    if (arg->rankSize < 2 || arg->rankSize > MAX_RANK_SIZE
        || arg->rankId >= arg->rankSize || arg->channelCount == 0
        || arg->channelCount >= arg->rankSize || arg->groupCount != MAX_CCU_DIE_GROUPS
        || arg->groupIndex >= arg->groupCount || arg->includesLocalRank > 1
        || arg->sourceCount != arg->channelCount + arg->includesLocalRank
        || arg->sourceCount == 0 || arg->sourceCount > MAX_RANK_SIZE
        || sequentialArg->scratchSlotCount < 2
        || sequentialArg->scratchSlotCount > arg->rankSize - 1
        || sequentialArg->outputRootUsesLocalInput > 1
        || arg->partialSlotIndices[0] != 1 || arg->partialSlotIndices[1] != 0
        || (arg->groupIndex == 0 &&
            (arg->includesLocalRank != 1 || arg->scratchBaseSlot != 1))
        || (arg->groupIndex == 1 &&
            (arg->includesLocalRank != 0 || arg->scratchBaseSlot != 0))
        || arg->scratchBaseSlot + arg->channelCount > sequentialArg->scratchSlotCount) {
        return CCU_E_PARA;
    }
    for (uint32_t idx = 0; idx < arg->channelCount; ++idx) {
        if (arg->remoteRanks[idx] >= arg->rankSize
            || arg->remoteRanks[idx] == arg->rankId) {
            return CCU_E_PARA;
        }
        for (uint32_t previous = 0; previous < idx; ++previous) {
            if (arg->remoteRanks[previous] == arg->remoteRanks[idx]) {
                return CCU_E_PARA;
            }
        }
    }

    KernelContext ctx;
    ctx.arg = arg;
    CCU_CHK_RET(InitContext(ctx));

    CCU_IF(ctx.phase == MESH_PHASE_PUBLISH)
    {
        CCU_CHK_RET(PublishAddresses(ctx));
    }
    CCU_IF(ctx.phase == MESH_PHASE_REDUCE_PARTIAL)
    {
        CCU_IF(ctx.needsHandshake != 0)
        {
            CCU_CHK_RET(WaitForAddresses(ctx));
        }
        if (arg->groupIndex == 0) {
            CCU_CHK_RET(ReduceGroupPartialOutputRoot(ctx,
                sequentialArg->outputRootUsesLocalInput != 0));
        } else {
            CCU_CHK_RET(ReduceGroupPartial(ctx));
        }
    }
    if (arg->groupIndex == 0) {
        CCU_IF(ctx.phase == MESH_PHASE_COMBINE)
        {
            CCU_CHK_RET(CombineGroupPartialsOutputRoot(ctx));
        }
    }
    CCU_IF(ctx.phase == MESH_PHASE_FINISH)
    {
        CCU_CHK_RET(BroadcastAndRecord(ctx));
        CCU_CHK_RET(WaitForFinish(ctx));
    }
    return CCU_SUCCESS;
}

CcuResult CcuSmall4AllPairsKernel(CcuKernelArg kernelArg)
{
    auto *arg = static_cast<Small4KernelArg *>(kernelArg);
    if (arg == nullptr || arg->channelCount != 3 || arg->inPlace > 1
        || arg->dataType != HCCL_DATA_TYPE_FP32
        || arg->reduceType != HCCL_REDUCE_SUM) {
        return CCU_E_PARA;
    }

    Small4KernelContext ctx;
    ctx.arg = arg;
    CCU_CHK_RET(InitSmall4Context(ctx));
    CCU_CHK_RET(PublishSmall4Input(ctx));
    CCU_CHK_RET(RunSmall4AllPairs(ctx));
    return CCU_SUCCESS;
}

} // namespace ops_hccl

#undef CCU_CHK_RET
