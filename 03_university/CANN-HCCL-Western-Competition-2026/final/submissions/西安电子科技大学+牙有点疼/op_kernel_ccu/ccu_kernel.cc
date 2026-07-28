/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <ccu/ccu_primitives.hpp>
#include <ccu/ccu_types.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "ccu_kernel.h"
#include "custom.h"

namespace ccu = ::AscendC::ccu;

#define CCU_KERNEL_CHK_RET(call) \
    do { \
        CcuResult ccuKernelRet = (call); \
        if (ccuKernelRet != CCU_SUCCESS) { \
            return ccuKernelRet; \
        } \
    } while (0)

namespace ops_hccl {
namespace {
constexpr uint32_t INPUT_XN_ID = 0;
constexpr uint32_t TOKEN_XN_ID = 1;
constexpr uint32_t OUTPUT_XN_ID = 2;
constexpr uint32_t REDUCE_SYNC_ID = 3;
constexpr uint32_t CKE_INDEX = 0;
constexpr uint32_t DIE_COUNT = 2;
constexpr uint64_t CCU_MS_SIZE = 4096;
constexpr uint64_t CCU_MS_INTERLEAVE = 8;
constexpr uint32_t REDUCE_LOOP_COUNT = 16;

struct GroupOpSizeVars {
    ccu::Variable addrOffset;
    ccu::Variable loopParam;
    ccu::Variable parallelParam;
    ccu::Variable residual;
    ccu::Variable useMs;
};

constexpr uint64_t SetBits(uint16_t end)
{
    return (uint64_t{1} << (end + 1)) - uint64_t{1};
}

uint64_t GetLoopParam(uint64_t loopCtxId, uint64_t gsaOffset, uint64_t loopIterNum)
{
    constexpr uint16_t CTX_ID_BIT_NUM = 8;
    constexpr uint16_t CTX_ID_SHIFT_BIT = 45;
    constexpr uint16_t GSA_BIT_NUM = 32;
    constexpr uint16_t GSA_SHIFT_BIT = 13;
    constexpr uint16_t LOOP_NUM_BIT_NUM = 13;
    return ((loopCtxId & SetBits(CTX_ID_BIT_NUM)) << CTX_ID_SHIFT_BIT) |
        ((gsaOffset & SetBits(GSA_BIT_NUM)) << GSA_SHIFT_BIT) |
        (loopIterNum & SetBits(LOOP_NUM_BIT_NUM));
}

uint64_t GetParallelParam(uint64_t repeatNum, uint64_t repeatLoopIndex,
    uint64_t totalLoopNum)
{
    constexpr uint16_t REPEAT_BIT_NUM = 7;
    constexpr uint16_t REPEAT_NUM_SHIFT_BIT = 55;
    constexpr uint16_t REPEAT_LOOP_BIT_NUM = 7;
    constexpr uint16_t REPEAT_LOOP_SHIFT_BIT = 48;
    constexpr uint16_t TOTAL_LOOP_BIT_NUM = 7;
    constexpr uint16_t TOTAL_LOOP_SHIFT_BIT = 41;
    return ((repeatNum & SetBits(REPEAT_BIT_NUM)) << REPEAT_NUM_SHIFT_BIT) |
        ((repeatLoopIndex & SetBits(REPEAT_LOOP_BIT_NUM)) << REPEAT_LOOP_SHIFT_BIT) |
        ((totalLoopNum & SetBits(TOTAL_LOOP_BIT_NUM)) << TOTAL_LOOP_SHIFT_BIT);
}

uint64_t GetOffsetParam(uint64_t gsaOffset, uint64_t msOffset, uint64_t ckeOffset)
{
    constexpr uint16_t GSA_BIT_NUM = 32;
    constexpr uint16_t GSA_SHIFT_BIT = 21;
    constexpr uint16_t MS_BIT_NUM = 11;
    constexpr uint16_t MS_SHIFT_BIT = 10;
    constexpr uint16_t CKE_BIT_NUM = 10;
    return ((gsaOffset & SetBits(GSA_BIT_NUM)) << GSA_SHIFT_BIT) |
        ((msOffset & SetBits(MS_BIT_NUM)) << MS_SHIFT_BIT) |
        (ckeOffset & SetBits(CKE_BIT_NUM));
}

struct AllReduceContext {
    const AllReduceKernelArg *arg = nullptr;

    ccu::Variable myInput;
    ccu::Variable myOutput;
    ccu::Variable myToken;
    ccu::Variable peerInput[MAX_RANK_SIZE];
    ccu::Variable peerToken[MAX_RANK_SIZE];
    ccu::Variable peerOutput[MAX_RANK_SIZE];
    ccu::Variable scratch;
    ccu::Variable sliceOffset;
    ccu::Variable sliceSize;
    ccu::Variable phase;
    GroupOpSizeVars reduceSize;

    ccu::LocalAddr groupData[MAX_RANK_SIZE];
    ccu::RemoteAddr remoteInput[MAX_RANK_SIZE];
    ccu::RemoteAddr remoteOutput[MAX_RANK_SIZE];
    ccu::LocalAddr localInput;
    ccu::LocalAddr localOutput;
    ccu::LocalAddr partial[2];
    ccu::Event event;
    ccu::Array<ccu::Event> loopEvents{0};
    ccu::Array<ccu::CcuBuffer> loopBuffers{0};
    std::unique_ptr<ccu::Func> reduceBodies[2];
    std::unique_ptr<ccu::Loop> reduceLoops[2];
    ccu::Variable reduceLoopParam[2];
    ccu::LocalAddr loopDst[2];
    ccu::LocalAddr loopSources[2][MAX_RANK_SIZE];
    ccu::Variable loopLen[2];
    uint32_t loopBufferStride = 0;
    bool reduceLoopsReady = false;
};

CcuResult InitResource(AllReduceContext &ctx)
{
    const auto *arg = ctx.arg;
    if (arg == nullptr || arg->rankSize == 0 || arg->rankSize > MAX_RANK_SIZE ||
        arg->dieId >= DIE_COUNT || (arg->channelCount == 0 && !arg->includeLocal) ||
        arg->channelCount + static_cast<uint32_t>(arg->includeLocal) > MAX_RANK_SIZE) {
        return CCU_E_PARA;
    }
    for (uint32_t i = 0; i < arg->channelCount; ++i) {
        ctx.peerInput[i] = ccu::GetResByChannel<ccu::Variable>(arg->channels[i], INPUT_XN_ID);
        ctx.peerToken[i] = ccu::GetResByChannel<ccu::Variable>(arg->channels[i], TOKEN_XN_ID);
        if (arg->writeOutput) {
            ctx.peerOutput[i] = ccu::GetResByChannel<ccu::Variable>(arg->channels[i], OUTPUT_XN_ID);
        }
    }
    return CCU_SUCCESS;
}

CcuResult LoadArgs(AllReduceContext &ctx)
{
    uint32_t argId = 0;
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.myInput, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.myOutput, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.myToken, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.scratch, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.sliceOffset, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.sliceSize, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.phase, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.reduceSize.addrOffset, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.reduceSize.loopParam, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.reduceSize.parallelParam, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.reduceSize.residual, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.reduceSize.useMs, argId++));
    return CCU_SUCCESS;
}

CcuResult PreSync(AllReduceContext &ctx)
{
    const auto *arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; ++i) {
        CCU_KERNEL_CHK_RET(ccu::WriteVariableWithNotify(
            arg->channels[i], ctx.myInput, INPUT_XN_ID, CKE_INDEX, 1U << INPUT_XN_ID));
        CCU_KERNEL_CHK_RET(ccu::WriteVariableWithNotify(
            arg->channels[i], ctx.myToken, TOKEN_XN_ID, CKE_INDEX, 1U << TOKEN_XN_ID));
        if (arg->writeOutput) {
            CCU_KERNEL_CHK_RET(ccu::WriteVariableWithNotify(
                arg->channels[i], ctx.myOutput, OUTPUT_XN_ID, CKE_INDEX, 1U << OUTPUT_XN_ID));
        }
    }
    uint16_t inputMask = (1U << INPUT_XN_ID) | (1U << TOKEN_XN_ID);
    if (arg->writeOutput) {
        inputMask |= 1U << OUTPUT_XN_ID;
    }
    for (uint32_t i = 0; i < arg->channelCount; ++i) {
        CCU_KERNEL_CHK_RET(ccu::NotifyWait(arg->channels[i], CKE_INDEX, inputMask));
    }

    return CCU_SUCCESS;
}

CcuResult CreateReduceLoops(AllReduceContext &ctx, uint32_t sourceCount)
{
    if (sourceCount == 0 || sourceCount > CCU_MS_INTERLEAVE) {
        return CCU_E_PARA;
    }
    if (!ctx.reduceLoopsReady) {
        ctx.loopBufferStride = sourceCount;
        ctx.loopEvents = ccu::Array<ccu::Event>(REDUCE_LOOP_COUNT);
        ctx.loopBuffers = ccu::Array<ccu::CcuBuffer>(
            REDUCE_LOOP_COUNT * ctx.loopBufferStride);
        ctx.reduceLoopsReady = true;
    } else if (sourceCount > ctx.loopBufferStride) {
        return CCU_E_PARA;
    }

    for (uint32_t index = 0; index < 2; ++index) {
        const uint32_t bufferBase = index * ctx.loopBufferStride;
        ccu::Event loopEvent = ctx.loopEvents[index];
        ctx.reduceBodies[index].reset(new ccu::Func(
            [&ctx, index, bufferBase, loopEvent, sourceCount]() {
                for (uint32_t i = 0; i < sourceCount; ++i) {
                    (void)ccu::LocalCopy(ctx.loopBuffers[bufferBase + i],
                        ctx.loopSources[index][i], ctx.loopLen[index], loopEvent,
                        static_cast<uint16_t>(1U << i));
                }
                (void)ccu::EventWait(loopEvent,
                    static_cast<uint16_t>((1U << sourceCount) - 1U));
                if (sourceCount > 1) {
                    (void)ccu::LocalReduce(&ctx.loopBuffers[bufferBase], sourceCount,
                        ctx.arg->dataType, ctx.arg->dataType, ctx.arg->reduceOp,
                        ctx.loopLen[index], loopEvent, 1);
                    (void)ccu::EventWait(loopEvent, 1);
                }
                (void)ccu::LocalCopy(ctx.loopDst[index], ctx.loopBuffers[bufferBase],
                    ctx.loopLen[index], loopEvent, 1);
                (void)ccu::EventWait(loopEvent, 1);
            }));
        ctx.reduceLoops[index].reset(
            new ccu::Loop(ctx.reduceLoopParam[index], *ctx.reduceBodies[index]));
    }
    return CCU_SUCCESS;
}

CcuResult ReduceWithMs(AllReduceContext &ctx, ccu::LocalAddr dst,
    ccu::LocalAddr *sources, uint32_t sourceCount)
{
    CCU_KERNEL_CHK_RET(CreateReduceLoops(ctx, sourceCount));

    ccu::LocalAddr reduceDst;
    reduceDst.addr = dst.addr;
    reduceDst.token = dst.token;
    ccu::LocalAddr reduceSources[MAX_RANK_SIZE];
    for (uint32_t i = 0; i < sourceCount; ++i) {
        reduceSources[i].addr = sources[i].addr;
        reduceSources[i].token = sources[i].token;
    }

    CCU_IF(ctx.reduceSize.loopParam != 0) {
        ccu::Variable loopParam;
        loopParam = GetLoopParam(0, CCU_MS_SIZE * REDUCE_LOOP_COUNT, 0);
        loopParam += ctx.reduceSize.loopParam;
        ctx.loopDst[0].addr = reduceDst.addr;
        ctx.loopDst[0].token = reduceDst.token;
        for (uint32_t i = 0; i < sourceCount; ++i) {
            ctx.loopSources[0][i].addr = reduceSources[i].addr;
            ctx.loopSources[0][i].token = reduceSources[i].token;
        }
        ctx.loopLen[0] = CCU_MS_SIZE;

        ccu::Variable parallelParam;
        parallelParam = GetParallelParam(REDUCE_LOOP_COUNT - 1, 0, 1);
        ccu::Variable offsetParam;
        offsetParam = GetOffsetParam(CCU_MS_SIZE, ctx.loopBufferStride, 1);
        ctx.reduceLoopParam[0] = loopParam;
        std::vector<ccu::Loop> loops{*ctx.reduceLoops[0]};
        ccu::LoopGroup group(parallelParam, offsetParam, 1, loops);
    }

    CCU_IF(ctx.reduceSize.parallelParam != 0) {
        reduceDst.addr += ctx.reduceSize.addrOffset;
        for (uint32_t i = 0; i < sourceCount; ++i) {
            reduceSources[i].addr += ctx.reduceSize.addrOffset;
        }

        ctx.loopDst[0].addr = reduceDst.addr;
        ctx.loopDst[0].token = reduceDst.token;
        for (uint32_t i = 0; i < sourceCount; ++i) {
            ctx.loopSources[0][i].addr = reduceSources[i].addr;
            ctx.loopSources[0][i].token = reduceSources[i].token;
        }
        ctx.loopLen[0] = ctx.reduceSize.residual;

        reduceDst.addr += ctx.reduceSize.residual;
        for (uint32_t i = 0; i < sourceCount; ++i) {
            reduceSources[i].addr += ctx.reduceSize.residual;
        }
        ctx.loopDst[1].addr = reduceDst.addr;
        ctx.loopDst[1].token = reduceDst.token;
        for (uint32_t i = 0; i < sourceCount; ++i) {
            ctx.loopSources[1][i].addr = reduceSources[i].addr;
            ctx.loopSources[1][i].token = reduceSources[i].token;
        }
        ctx.loopLen[1] = CCU_MS_SIZE;

        ccu::Variable loopParam0;
        loopParam0 = GetLoopParam(0, 0, 1);
        ccu::Variable loopParam1;
        loopParam1 = GetLoopParam(0, 0, 1);
        ccu::Variable offsetParam;
        offsetParam = GetOffsetParam(CCU_MS_SIZE, ctx.loopBufferStride, 1);
        ctx.reduceLoopParam[0] = loopParam0;
        ctx.reduceLoopParam[1] = loopParam1;
        std::vector<ccu::Loop> loops{*ctx.reduceLoops[0], *ctx.reduceLoops[1]};
        ccu::LoopGroup group(ctx.reduceSize.parallelParam, offsetParam, 2, loops);
    }
    return CCU_SUCCESS;
}

CcuResult ReduceSources(AllReduceContext &ctx, ccu::LocalAddr result,
    ccu::LocalAddr *sources, uint32_t sourceCount)
{
    if (sourceCount == 0 || sourceCount > MAX_RANK_SIZE) {
        return CCU_E_PARA;
    }

    uint32_t processed = sourceCount < CCU_MS_INTERLEAVE ?
        sourceCount : static_cast<uint32_t>(CCU_MS_INTERLEAVE);
    CCU_KERNEL_CHK_RET(ReduceWithMs(ctx, result, sources, processed));
    while (processed < sourceCount) {
        ccu::LocalAddr stageSources[CCU_MS_INTERLEAVE];
        stageSources[0].addr = result.addr;
        stageSources[0].token = result.token;
        uint32_t stageCount = 1;
        while (stageCount < CCU_MS_INTERLEAVE && processed < sourceCount) {
            stageSources[stageCount].addr = sources[processed].addr;
            stageSources[stageCount].token = sources[processed].token;
            ++stageCount;
            ++processed;
        }
        CCU_KERNEL_CHK_RET(ReduceWithMs(ctx, result, stageSources, stageCount));
    }
    return CCU_SUCCESS;
}

CcuResult ReduceSourcesWithHbm(AllReduceContext &ctx, ccu::LocalAddr *sources,
    uint32_t sourceCount)
{
    if (sourceCount == 0 || sourceCount > MAX_RANK_SIZE) {
        return CCU_E_PARA;
    }

    uint32_t remain = sourceCount;
    while (remain > 1) {
        const uint32_t reducePieces = remain / 2;
        const uint32_t sourceIndex = remain - reducePieces;
        ccu::Variable reduceLength;
        reduceLength = ctx.sliceSize;
        for (uint32_t i = 1; i < reducePieces; ++i) {
            reduceLength += ctx.sliceSize;
        }
        CCU_KERNEL_CHK_RET(ccu::LocalReduce(sources[0], sources[sourceIndex],
            reduceLength, ctx.arg->dataType, ctx.arg->reduceOp, ctx.event, 1));
        CCU_KERNEL_CHK_RET(ccu::EventWait(ctx.event, 1));
        remain -= reducePieces;
    }
    return CCU_SUCCESS;
}

CcuResult ReduceGroup(AllReduceContext &ctx)
{
    const auto *arg = ctx.arg;
    ccu::Variable scratchOffset;
    scratchOffset = 0;
    for (uint32_t i = 0; i < arg->scratchSlotBase; ++i) {
        scratchOffset += ctx.sliceSize;
    }

    for (uint32_t i = 0; i < arg->channelCount; ++i) {
        ctx.groupData[i].addr = ctx.scratch;
        ctx.groupData[i].addr += scratchOffset;
        ctx.groupData[i].token = ctx.myToken;
        scratchOffset += ctx.sliceSize;

        ctx.remoteInput[i].addr = ctx.peerInput[i];
        ctx.remoteInput[i].addr += ctx.sliceOffset;
        ctx.remoteInput[i].token = ctx.peerToken[i];
        CCU_KERNEL_CHK_RET(ccu::Read(arg->channels[i], ctx.groupData[i], ctx.remoteInput[i],
            ctx.sliceSize, ctx.event, static_cast<uint16_t>(1U << i)));
    }
    uint32_t groupSize = arg->channelCount;
    if (arg->includeLocal) {
        ctx.localInput.addr = ctx.myInput;
        ctx.localInput.addr += ctx.sliceOffset;
        ctx.localInput.token = ctx.myToken;
        ctx.groupData[groupSize].addr = ctx.localInput.addr;
        ctx.groupData[groupSize].token = ctx.localInput.token;
        CCU_IF(ctx.reduceSize.useMs == 0) {
            ctx.groupData[groupSize].addr = ctx.scratch;
            ctx.groupData[groupSize].addr += scratchOffset;
            ctx.groupData[groupSize].token = ctx.myToken;
            CCU_KERNEL_CHK_RET(ccu::LocalCopy(ctx.groupData[groupSize], ctx.localInput,
                ctx.sliceSize, ctx.event, static_cast<uint16_t>(1U << groupSize)));
        }
        ++groupSize;
    }

    const uint16_t peerMask = static_cast<uint16_t>((1U << arg->channelCount) - 1U);
    if (arg->includeLocal) {
        const uint16_t groupMask = static_cast<uint16_t>((1U << groupSize) - 1U);
        CCU_IF(ctx.reduceSize.useMs != 0) {
            CCU_KERNEL_CHK_RET(ccu::EventWait(ctx.event, peerMask));
        } CCU_ELSE {
            CCU_KERNEL_CHK_RET(ccu::EventWait(ctx.event, groupMask));
        }
    } else {
        CCU_KERNEL_CHK_RET(ccu::EventWait(ctx.event, peerMask));
    }

    CCU_IF(ctx.reduceSize.useMs != 0) {
        ccu::LocalAddr result;
        if (arg->directBroadcast || arg->dieId == 0) {
            result.addr = ctx.myOutput;
            result.addr += ctx.sliceOffset;
        } else {
            ccu::Variable resultOffset;
            resultOffset = 0;
            for (uint32_t i = 0; i < arg->partialSlot[arg->dieId]; ++i) {
                resultOffset += ctx.sliceSize;
            }
            result.addr = ctx.scratch;
            result.addr += resultOffset;
        }
        result.token = ctx.myToken;
        CCU_KERNEL_CHK_RET(ReduceSources(
            ctx, result, ctx.groupData, groupSize));
    } CCU_ELSE {
        CCU_KERNEL_CHK_RET(ReduceSourcesWithHbm(
            ctx, ctx.groupData, groupSize));
    }
    return CCU_SUCCESS;
}

CcuResult BroadcastResult(AllReduceContext &ctx, ccu::LocalAddr source,
    bool copyToLocalOutput)
{
    const auto *arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; ++i) {
        ctx.remoteOutput[i].addr = ctx.peerOutput[i];
        ctx.remoteOutput[i].addr += ctx.sliceOffset;
        ctx.remoteOutput[i].token = ctx.peerToken[i];
        CCU_KERNEL_CHK_RET(ccu::Write(arg->channels[i], ctx.remoteOutput[i],
            source, ctx.sliceSize, ctx.event, static_cast<uint16_t>(1U << i)));
    }
    uint16_t mask = static_cast<uint16_t>((1U << arg->channelCount) - 1U);
    if (copyToLocalOutput) {
        ctx.localOutput.addr = ctx.myOutput;
        ctx.localOutput.addr += ctx.sliceOffset;
        ctx.localOutput.token = ctx.myToken;
        CCU_KERNEL_CHK_RET(ccu::LocalCopy(ctx.localOutput, source, ctx.sliceSize,
            ctx.event, static_cast<uint16_t>(1U << arg->channelCount)));
        mask |= static_cast<uint16_t>(1U << arg->channelCount);
    }
    return ccu::EventWait(ctx.event, mask);
}

CcuResult BroadcastGroupResult(AllReduceContext &ctx)
{
    CCU_IF(ctx.reduceSize.useMs != 0) {
        ctx.localOutput.addr = ctx.myOutput;
        ctx.localOutput.addr += ctx.sliceOffset;
        ctx.localOutput.token = ctx.myToken;
        CCU_KERNEL_CHK_RET(BroadcastResult(ctx, ctx.localOutput, false));
    } CCU_ELSE {
        CCU_KERNEL_CHK_RET(BroadcastResult(ctx, ctx.groupData[0], true));
    }
    return CCU_SUCCESS;
}

CcuResult CombinePublishedPartials(AllReduceContext &ctx)
{
    constexpr uint32_t remoteDie = 1;
    ccu::Variable partialOffset;
    partialOffset = 0;
    for (uint32_t i = 0; i < ctx.arg->partialSlot[remoteDie]; ++i) {
        partialOffset += ctx.sliceSize;
    }
    ctx.partial[remoteDie].addr = ctx.scratch;
    ctx.partial[remoteDie].addr += partialOffset;
    ctx.partial[remoteDie].token = ctx.myToken;

    CCU_IF(ctx.reduceSize.useMs != 0) {
        ctx.localOutput.addr = ctx.myOutput;
        ctx.localOutput.addr += ctx.sliceOffset;
        ctx.localOutput.token = ctx.myToken;
        CCU_KERNEL_CHK_RET(ccu::LocalReduce(ctx.localOutput, ctx.partial[remoteDie],
            ctx.sliceSize, ctx.arg->dataType, ctx.arg->reduceOp, ctx.event, 1));
    } CCU_ELSE {
        ccu::Variable localPartialOffset;
        localPartialOffset = 0;
        for (uint32_t i = 0; i < ctx.arg->partialSlot[0]; ++i) {
            localPartialOffset += ctx.sliceSize;
        }
        ctx.partial[0].addr = ctx.scratch;
        ctx.partial[0].addr += localPartialOffset;
        ctx.partial[0].token = ctx.myToken;
        CCU_KERNEL_CHK_RET(ccu::LocalReduce(ctx.partial[0], ctx.partial[remoteDie],
            ctx.sliceSize, ctx.arg->dataType, ctx.arg->reduceOp, ctx.event, 1));
    }
    CCU_KERNEL_CHK_RET(ccu::EventWait(ctx.event, 1));
    CCU_KERNEL_CHK_RET(ccu::EventRecord(ctx.event, 1));
    return ccu::EventWait(ctx.event, 1);
}

CcuResult BroadcastCombinedPartial(AllReduceContext &ctx)
{
    CCU_IF(ctx.reduceSize.useMs != 0) {
        ctx.localOutput.addr = ctx.myOutput;
        ctx.localOutput.addr += ctx.sliceOffset;
        ctx.localOutput.token = ctx.myToken;
        CCU_KERNEL_CHK_RET(BroadcastResult(ctx, ctx.localOutput, false));
    } CCU_ELSE {
        ccu::Variable partialOffset;
        partialOffset = 0;
        for (uint32_t i = 0; i < ctx.arg->partialSlot[0]; ++i) {
            partialOffset += ctx.sliceSize;
        }
        ctx.partial[0].addr = ctx.scratch;
        ctx.partial[0].addr += partialOffset;
        ctx.partial[0].token = ctx.myToken;
        CCU_KERNEL_CHK_RET(BroadcastResult(
            ctx, ctx.partial[0], ctx.arg->dieId == 0));
    }
    return CCU_SUCCESS;
}

CcuResult PostSync(AllReduceContext &ctx, uint32_t syncId)
{
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        CCU_KERNEL_CHK_RET(
            ccu::NotifyRecord(ctx.arg->channels[i], CKE_INDEX, 1U << syncId));
    }
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        CCU_KERNEL_CHK_RET(
            ccu::NotifyWait(ctx.arg->channels[i], CKE_INDEX, 1U << syncId));
    }
    return CCU_SUCCESS;
}

} // namespace

CcuResult CcuKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<AllReduceKernelArg *>(arg);
    if (kernelArg == nullptr || kernelArg->dataType != HCCL_DATA_TYPE_FP32 ||
        kernelArg->reduceOp != HCCL_REDUCE_SUM) {
        return CCU_E_PARA;
    }

    AllReduceContext ctx;
    ctx.arg = kernelArg;
    CCU_KERNEL_CHK_RET(LoadArgs(ctx));
    if (kernelArg->combineOnly) {
        CCU_IF(ctx.sliceSize != 0) {
            CCU_KERNEL_CHK_RET(CombinePublishedPartials(ctx));
        }
        return CCU_SUCCESS;
    }
    CCU_KERNEL_CHK_RET(InitResource(ctx));
    CCU_KERNEL_CHK_RET(PreSync(ctx));
    CCU_IF(ctx.phase == 0) {
        CCU_IF(ctx.sliceSize != 0) {
            CCU_KERNEL_CHK_RET(ReduceGroup(ctx));
            if (kernelArg->directBroadcast) {
                CCU_KERNEL_CHK_RET(BroadcastGroupResult(ctx));
            }
        }
    } CCU_ELSE {
        CCU_IF(ctx.sliceSize != 0) {
            CCU_KERNEL_CHK_RET(BroadcastCombinedPartial(ctx));
        }
    }
    CCU_KERNEL_CHK_RET(PostSync(ctx, REDUCE_SYNC_ID));
    return CCU_SUCCESS;
}
} // namespace ops_hccl

#undef CCU_KERNEL_CHK_RET
