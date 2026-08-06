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
#include <hcomm/hcomm_primitives.h>

#include "ccu_kernel.h"
#include "custom.h"

namespace final_small512 {
namespace ccu = ::AscendC::ccu;

namespace ops_hccl {
namespace {
#define RS2_CCU_CHK_RET(expression) \
    do { \
        const CcuResult result = (expression); \
        if (result != CCU_SUCCESS) { \
            return result; \
        } \
    } while (0)

constexpr uint32_t INPUT_XN_ID = 0;
constexpr uint32_t TOKEN_XN_ID = 1;
constexpr uint32_t POST_SYNC_ID = 2;
constexpr uint32_t RING_STAGE_READY_ID = 3;
constexpr uint32_t RING_STAGE_DONE_ID = 4;
constexpr uint32_t CKE_IDX = 0;
constexpr uint64_t KERNEL_MODE_FINALIZE = 1;
constexpr uint64_t KERNEL_MODE_DIRECT_OUTPUT = 2;

struct KernelContext {
    const CcuKernelArgReduceScatterTree *arg = nullptr;
    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable scratch;
    ccu::Variable scratchToken;
    ccu::Variable targetInputOffset;
    ccu::Variable outputOffset;
    ccu::Variable chunkBytes;
    ccu::Variable mode;
    ccu::Variable resultCount;
    std::vector<ccu::Variable> resultOffsets;
    std::vector<ccu::Variable> peerInputs;
    std::vector<ccu::Variable> peerTokens;
    ccu::Event event;
};

CcuResult InitResources(KernelContext &ctx)
{
    ctx.peerInputs.resize(ctx.arg->channelCount);
    ctx.peerTokens.resize(ctx.arg->channelCount);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        ctx.peerInputs[channelIdx] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], INPUT_XN_ID);
        ctx.peerTokens[channelIdx] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], TOKEN_XN_ID);
    }
    return CCU_SUCCESS;
}

CcuResult LoadArgs(KernelContext &ctx)
{
    uint32_t argId = 0;
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.output, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.inputToken, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.outputToken, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.scratch, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.scratchToken, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.targetInputOffset, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.outputOffset, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.chunkBytes, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.mode, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.resultCount, argId++));
    ctx.resultOffsets.resize(MAX_GLOBAL_RESULT_COUNT);
    for (uint32_t resultIdx = 0; resultIdx < MAX_GLOBAL_RESULT_COUNT; ++resultIdx) {
        RS2_CCU_CHK_RET(ccu::LoadArg(ctx.resultOffsets[resultIdx], argId++));
    }
    return CCU_SUCCESS;
}
CcuResult PreSync(KernelContext &ctx)
{
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx], ctx.input, INPUT_XN_ID,
            CKE_IDX, static_cast<uint16_t>(1U << INPUT_XN_ID)));
        RS2_CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx], ctx.inputToken, TOKEN_XN_ID,
            CKE_IDX, static_cast<uint16_t>(1U << TOKEN_XN_ID)));
    }

    constexpr uint16_t waitMask = static_cast<uint16_t>((1U << INPUT_XN_ID) | (1U << TOKEN_XN_ID));
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIdx], CKE_IDX, waitMask));
    }
    return CCU_SUCCESS;
}

CcuResult PostSync(KernelContext &ctx)
{
    constexpr uint16_t postMask = static_cast<uint16_t>(1U << POST_SYNC_ID);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[channelIdx], CKE_IDX, postMask));
    }
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIdx], CKE_IDX, postMask));
    }
    return CCU_SUCCESS;
}

CcuResult ReadPieces(KernelContext &ctx, std::vector<ccu::LocalAddr> &pieces)
{
    ccu::Variable scratchOffset;
    scratchOffset = 0;
    for (uint32_t slotIdx = 0; slotIdx < ctx.arg->scratchSlotBase; ++slotIdx) {
        scratchOffset += ctx.chunkBytes;
    }

    std::vector<ccu::RemoteAddr> remoteInputs(ctx.arg->channelCount);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        pieces[channelIdx].addr = ctx.scratch;
        pieces[channelIdx].addr += scratchOffset;
        pieces[channelIdx].token = ctx.scratchToken;

        remoteInputs[channelIdx].addr = ctx.peerInputs[channelIdx];
        remoteInputs[channelIdx].addr += ctx.targetInputOffset;
        remoteInputs[channelIdx].token = ctx.peerTokens[channelIdx];
        scratchOffset += ctx.chunkBytes;
    }

    uint16_t readMask = 0;
    if (ctx.arg->includeLocalInput != 0) {
        const uint32_t localPieceIdx = ctx.arg->channelCount;
        pieces[localPieceIdx].addr = ctx.scratch;
        pieces[localPieceIdx].addr += scratchOffset;
        pieces[localPieceIdx].token = ctx.scratchToken;

        ccu::LocalAddr localInput;
        localInput.addr = ctx.input;
        localInput.addr += ctx.targetInputOffset;
        localInput.token = ctx.inputToken;
        const uint16_t localMask = static_cast<uint16_t>(1U << localPieceIdx);
        RS2_CCU_CHK_RET(ccu::LocalCopy(pieces[localPieceIdx], localInput, ctx.chunkBytes, ctx.event, localMask));
        readMask = static_cast<uint16_t>(readMask | localMask);
    }

    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        const uint16_t channelMask = static_cast<uint16_t>(1U << channelIdx);
        RS2_CCU_CHK_RET(ccu::Read(ctx.arg->channels[channelIdx], pieces[channelIdx], remoteInputs[channelIdx],
            ctx.chunkBytes, ctx.event, channelMask));
        readMask = static_cast<uint16_t>(readMask | channelMask);
    }
    if (readMask != 0) {
        RS2_CCU_CHK_RET(ccu::EventWait(ctx.event, readMask));
    }
    return CCU_SUCCESS;
}

CcuResult ReducePieces(KernelContext &ctx, std::vector<ccu::LocalAddr> &pieces)
{
    uint32_t remainingPieces = static_cast<uint32_t>(pieces.size());
    while (remainingPieces > 1) {
        const uint32_t reducePieces = remainingPieces / 2;
        const uint32_t sourceIdx = remainingPieces - reducePieces;

        ccu::Variable reduceBytes;
        reduceBytes = ctx.chunkBytes;
        for (uint32_t pieceIdx = 1; pieceIdx < reducePieces; ++pieceIdx) {
            reduceBytes += ctx.chunkBytes;
        }
        RS2_CCU_CHK_RET(ccu::LocalReduce(pieces[0], pieces[sourceIdx], reduceBytes, HCCL_DATA_TYPE_FP32,
            HCCL_REDUCE_SUM, ctx.event));
        RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));
        remainingPieces -= reducePieces;
    }
    return CCU_SUCCESS;
}

CcuResult WriteOutput(KernelContext &ctx, const ccu::LocalAddr &reducedData, bool initializeOutput)
{
    ccu::LocalAddr output;
    output.addr = ctx.output;
    output.addr += ctx.outputOffset;
    output.token = ctx.outputToken;
    if (initializeOutput) {
        RS2_CCU_CHK_RET(ccu::LocalCopy(output, reducedData, ctx.chunkBytes, ctx.event));
    } else {
        RS2_CCU_CHK_RET(ccu::LocalReduce(output, reducedData, ctx.chunkBytes, HCCL_DATA_TYPE_FP32,
            HCCL_REDUCE_SUM, ctx.event));
    }
    RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));
    return CCU_SUCCESS;
}

CcuResult FinalizeResults(KernelContext &ctx)
{
    ccu::LocalAddr output;
    output.addr = ctx.output;
    output.addr += ctx.outputOffset;
    output.token = ctx.outputToken;

    ccu::LocalAddr partial;
    partial.addr = ctx.scratch;
    partial.addr += ctx.resultOffsets[0];
    partial.token = ctx.scratchToken;
    RS2_CCU_CHK_RET(ccu::LocalCopy(output, partial, ctx.chunkBytes, ctx.event));
    RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));

    CCU_IF(ctx.resultCount != static_cast<uint64_t>(1)) {
        partial.addr = ctx.scratch;
        partial.addr += ctx.resultOffsets[1];
        partial.token = ctx.scratchToken;
        RS2_CCU_CHK_RET(ccu::LocalReduce(output, partial, ctx.chunkBytes, HCCL_DATA_TYPE_FP32,
            HCCL_REDUCE_SUM, ctx.event));
        RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));

        CCU_IF(ctx.resultCount != static_cast<uint64_t>(2)) {
            partial.addr = ctx.scratch;
            partial.addr += ctx.resultOffsets[2];
            partial.token = ctx.scratchToken;
            RS2_CCU_CHK_RET(ccu::LocalReduce(output, partial, ctx.chunkBytes, HCCL_DATA_TYPE_FP32,
                HCCL_REDUCE_SUM, ctx.event));
            RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));

            CCU_IF(ctx.resultCount != static_cast<uint64_t>(3)) {
                partial.addr = ctx.scratch;
                partial.addr += ctx.resultOffsets[3];
                partial.token = ctx.scratchToken;
                RS2_CCU_CHK_RET(ccu::LocalReduce(output, partial, ctx.chunkBytes, HCCL_DATA_TYPE_FP32,
                    HCCL_REDUCE_SUM, ctx.event));
                RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));
            }
        }
    }
    return CCU_SUCCESS;
}
} // namespace

CcuResult CcuReduceScatterTreeKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceScatterTree *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount == 0) {
        return CCU_E_PARA;
    }

    KernelContext ctx;
    ctx.arg = kernelArg;
    RS2_CCU_CHK_RET(InitResources(ctx));
    RS2_CCU_CHK_RET(LoadArgs(ctx));
    CCU_IF(ctx.mode == KERNEL_MODE_FINALIZE) {
        RS2_CCU_CHK_RET(FinalizeResults(ctx));
    } CCU_ELSE {
        RS2_CCU_CHK_RET(PreSync(ctx));

        const uint32_t pieceCount = kernelArg->channelCount + kernelArg->includeLocalInput;
        std::vector<ccu::LocalAddr> pieces(pieceCount);
        RS2_CCU_CHK_RET(ReadPieces(ctx, pieces));
        RS2_CCU_CHK_RET(ReducePieces(ctx, pieces));
        if (kernelArg->writeOutput != 0) {
            RS2_CCU_CHK_RET(WriteOutput(ctx, pieces[0], kernelArg->initializeOutput != 0));
        } else {
            CCU_IF(ctx.mode == KERNEL_MODE_DIRECT_OUTPUT) {
                RS2_CCU_CHK_RET(WriteOutput(ctx, pieces[0], kernelArg->scratchSlotBase == 0));
            }
        }
        RS2_CCU_CHK_RET(PostSync(ctx));
    }
    return CCU_SUCCESS;
}

namespace {
struct SmallKernelContext {
    const CcuKernelArgReduceScatterSmall *arg = nullptr;
    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable scratch;
    ccu::Variable scratchToken;
    ccu::Variable targetInputOffset;
    ccu::Variable outputOffset;
    ccu::Variable chunkBytes;
    std::vector<ccu::Variable> peerInputs;
    std::vector<ccu::Variable> peerTokens;
    ccu::Event event;
};

CcuResult InitSmallResources(SmallKernelContext &ctx)
{
    ctx.peerInputs.resize(ctx.arg->channelCount);
    ctx.peerTokens.resize(ctx.arg->channelCount);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        ctx.peerInputs[channelIdx] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], INPUT_XN_ID);
        ctx.peerTokens[channelIdx] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], TOKEN_XN_ID);
    }
    return CCU_SUCCESS;
}

CcuResult LoadSmallArgs(SmallKernelContext &ctx)
{
    uint32_t argId = 0;
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.output, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.inputToken, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.outputToken, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.scratch, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.scratchToken, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.targetInputOffset, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.outputOffset, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.chunkBytes, argId++));
    return CCU_SUCCESS;
}

CcuResult SmallPreSync(SmallKernelContext &ctx)
{
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx], ctx.input, INPUT_XN_ID,
            CKE_IDX, static_cast<uint16_t>(1U << INPUT_XN_ID)));
        RS2_CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx], ctx.inputToken, TOKEN_XN_ID,
            CKE_IDX, static_cast<uint16_t>(1U << TOKEN_XN_ID)));
    }
    constexpr uint16_t waitMask = static_cast<uint16_t>((1U << INPUT_XN_ID) | (1U << TOKEN_XN_ID));
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIdx], CKE_IDX, waitMask));
    }
    return CCU_SUCCESS;
}

CcuResult SmallReadPieces(SmallKernelContext &ctx, std::vector<ccu::LocalAddr> &pieces)
{
    ccu::Variable scratchOffset;
    scratchOffset = 0;
    for (uint32_t slotIdx = 0; slotIdx < ctx.arg->scratchSlotBase; ++slotIdx) {
        scratchOffset += ctx.chunkBytes;
    }

    std::vector<ccu::RemoteAddr> remoteInputs(ctx.arg->channelCount);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        pieces[channelIdx].addr = ctx.scratch;
        pieces[channelIdx].addr += scratchOffset;
        pieces[channelIdx].token = ctx.scratchToken;
        remoteInputs[channelIdx].addr = ctx.peerInputs[channelIdx];
        remoteInputs[channelIdx].addr += ctx.targetInputOffset;
        remoteInputs[channelIdx].token = ctx.peerTokens[channelIdx];
        scratchOffset += ctx.chunkBytes;
    }

    uint16_t readMask = 0;
    if (ctx.arg->includeLocalInput != 0) {
        const uint32_t localPieceIdx = ctx.arg->channelCount;
        pieces[localPieceIdx].addr = ctx.scratch;
        pieces[localPieceIdx].addr += scratchOffset;
        pieces[localPieceIdx].token = ctx.scratchToken;
        ccu::LocalAddr localInput;
        localInput.addr = ctx.input;
        localInput.addr += ctx.targetInputOffset;
        localInput.token = ctx.inputToken;
        const uint16_t localMask = static_cast<uint16_t>(1U << localPieceIdx);
        RS2_CCU_CHK_RET(ccu::LocalCopy(pieces[localPieceIdx], localInput, ctx.chunkBytes, ctx.event, localMask));
        readMask = static_cast<uint16_t>(readMask | localMask);
    }
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        const uint16_t channelMask = static_cast<uint16_t>(1U << channelIdx);
        RS2_CCU_CHK_RET(ccu::Read(ctx.arg->channels[channelIdx], pieces[channelIdx], remoteInputs[channelIdx],
            ctx.chunkBytes, ctx.event, channelMask));
        readMask = static_cast<uint16_t>(readMask | channelMask);
    }
    return ccu::EventWait(ctx.event, readMask);
}

CcuResult SmallReducePieces(SmallKernelContext &ctx, std::vector<ccu::LocalAddr> &pieces)
{
    uint32_t remainingPieces = static_cast<uint32_t>(pieces.size());
    while (remainingPieces > 1) {
        const uint32_t reducePieces = remainingPieces / 2;
        const uint32_t sourceIdx = remainingPieces - reducePieces;
        ccu::Variable reduceBytes;
        reduceBytes = ctx.chunkBytes;
        for (uint32_t pieceIdx = 1; pieceIdx < reducePieces; ++pieceIdx) {
            reduceBytes += ctx.chunkBytes;
        }
        RS2_CCU_CHK_RET(ccu::LocalReduce(pieces[0], pieces[sourceIdx], reduceBytes, HCCL_DATA_TYPE_FP32,
            HCCL_REDUCE_SUM, ctx.event));
        RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));
        remainingPieces -= reducePieces;
    }
    return CCU_SUCCESS;
}

CcuResult SmallWriteOutput(SmallKernelContext &ctx, const ccu::LocalAddr &reducedData)
{
    ccu::LocalAddr output;
    output.addr = ctx.output;
    output.addr += ctx.outputOffset;
    output.token = ctx.outputToken;
    RS2_CCU_CHK_RET(ccu::LocalCopy(output, reducedData, ctx.chunkBytes, ctx.event));
    return ccu::EventWait(ctx.event);
}

CcuResult SmallPostSync(SmallKernelContext &ctx)
{
    constexpr uint16_t postMask = static_cast<uint16_t>(1U << POST_SYNC_ID);
    if (ctx.arg->barrierRoundCount != 0) {
        for (uint32_t roundIdx = 0; roundIdx < ctx.arg->barrierRoundCount; ++roundIdx) {
            RS2_CCU_CHK_RET(ccu::NotifyRecord(
                ctx.arg->channels[ctx.arg->barrierSendChannelIndices[roundIdx]], CKE_IDX, postMask));
            RS2_CCU_CHK_RET(ccu::NotifyWait(
                ctx.arg->channels[ctx.arg->barrierRecvChannelIndices[roundIdx]], CKE_IDX, postMask));
        }
        return CCU_SUCCESS;
    }
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[channelIdx], CKE_IDX, postMask));
    }
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIdx], CKE_IDX, postMask));
    }
    return CCU_SUCCESS;
}
} // namespace

CcuResult CcuReduceScatterSmallKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceScatterSmall *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount == 0) {
        return CCU_E_PARA;
    }
    const uint32_t pieceCount = kernelArg->channelCount + kernelArg->includeLocalInput;
    if (pieceCount < 2 || pieceCount > 8) {
        return CCU_E_NOT_SUPPORT;
    }

    SmallKernelContext ctx;
    ctx.arg = kernelArg;
    RS2_CCU_CHK_RET(InitSmallResources(ctx));
    RS2_CCU_CHK_RET(LoadSmallArgs(ctx));
    RS2_CCU_CHK_RET(SmallPreSync(ctx));
    std::vector<ccu::LocalAddr> pieces(pieceCount);
    RS2_CCU_CHK_RET(SmallReadPieces(ctx, pieces));
    RS2_CCU_CHK_RET(SmallReducePieces(ctx, pieces));
    if (kernelArg->writeOutput != 0) {
        RS2_CCU_CHK_RET(SmallWriteOutput(ctx, pieces[0]));
    }
    RS2_CCU_CHK_RET(SmallPostSync(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterSmallFinalizeKernel(CcuKernelArg)
{
    ccu::Variable output;
    ccu::Variable outputToken;
    ccu::Variable scratch;
    ccu::Variable scratchToken;
    ccu::Variable partialOffset;
    ccu::Variable chunkBytes;
    RS2_CCU_CHK_RET(ccu::LoadArg(output, 0));
    RS2_CCU_CHK_RET(ccu::LoadArg(outputToken, 1));
    RS2_CCU_CHK_RET(ccu::LoadArg(scratch, 2));
    RS2_CCU_CHK_RET(ccu::LoadArg(scratchToken, 3));
    RS2_CCU_CHK_RET(ccu::LoadArg(partialOffset, 4));
    RS2_CCU_CHK_RET(ccu::LoadArg(chunkBytes, 5));

    ccu::LocalAddr destination;
    destination.addr = output;
    destination.token = outputToken;
    ccu::LocalAddr source;
    source.addr = scratch;
    source.addr += partialOffset;
    source.token = scratchToken;
    ccu::Event event;
    RS2_CCU_CHK_RET(ccu::LocalReduce(destination, source, chunkBytes, HCCL_DATA_TYPE_FP32,
        HCCL_REDUCE_SUM, event));
    return ccu::EventWait(event);
}

namespace {
constexpr uint32_t HALF_RING_LANE_COUNT = 2;

struct RingLaneContext {
    ChannelHandle nextChannel = 0;
    ChannelHandle prevChannel = 0;
    uint16_t readyMask = 0;
    uint16_t doneMask = 0;
    ccu::Variable chunkOffset;
    ccu::Variable chunkBytes;
    ccu::Event event;
};

struct RingKernelContext {
    const CcuKernelArgReduceScatterRing *arg = nullptr;
    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable recvBytes;
    ccu::Variable nextInput;
    ccu::Variable nextInputToken;
    RingLaneContext lanes[HALF_RING_LANE_COUNT];
};

CcuResult InitRingResources(RingKernelContext &ctx)
{
    ctx.nextInput = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[0], INPUT_XN_ID);
    ctx.nextInputToken = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[0], TOKEN_XN_ID);
    for (uint32_t laneIdx = 0; laneIdx < HALF_RING_LANE_COUNT; ++laneIdx) {
        RingLaneContext &lane = ctx.lanes[laneIdx];
        lane.nextChannel = ctx.arg->channels[0];
        lane.prevChannel = ctx.arg->channels[1];
        lane.readyMask = static_cast<uint16_t>(1U << (RING_STAGE_READY_ID + laneIdx * 2));
        lane.doneMask = static_cast<uint16_t>(1U << (RING_STAGE_DONE_ID + laneIdx * 2));
    }
    return CCU_SUCCESS;
}

CcuResult LoadRingArgs(RingKernelContext &ctx)
{
    uint32_t argId = 0;
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.output, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.inputToken, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.outputToken, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.recvBytes, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.lanes[0].chunkBytes, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.lanes[1].chunkBytes, argId++));
    ctx.lanes[0].chunkOffset = static_cast<uint64_t>(0);
    ctx.lanes[1].chunkOffset = ctx.lanes[0].chunkBytes;
    return CCU_SUCCESS;
}

CcuResult RingPreSync(RingKernelContext &ctx)
{
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx], ctx.input, INPUT_XN_ID,
            CKE_IDX, static_cast<uint16_t>(1U << INPUT_XN_ID)));
        RS2_CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx], ctx.inputToken, TOKEN_XN_ID,
            CKE_IDX, static_cast<uint16_t>(1U << TOKEN_XN_ID)));
    }
    constexpr uint16_t waitMask = static_cast<uint16_t>((1U << INPUT_XN_ID) | (1U << TOKEN_XN_ID));
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIdx], CKE_IDX, waitMask));
    }
    return CCU_SUCCESS;
}

CcuResult RingStageRecordReady(RingLaneContext &lane)
{
    return ccu::NotifyRecord(lane.prevChannel, CKE_IDX, lane.readyMask);
}

CcuResult RingStageWaitReady(RingLaneContext &lane)
{
    return ccu::NotifyWait(lane.nextChannel, CKE_IDX, lane.readyMask);
}

CcuResult RingStageWrite(RingKernelContext &ctx, RingLaneContext &lane, uint32_t blockIdx)
{
    ccu::Variable blockOffset;
    blockOffset = lane.chunkOffset;
    for (uint32_t block = 0; block < blockIdx; ++block) {
        blockOffset += ctx.recvBytes;
    }

    ccu::LocalAddr source;
    source.addr = ctx.input;
    source.addr += blockOffset;
    source.token = ctx.inputToken;
    ccu::RemoteAddr destination;
    destination.addr = ctx.nextInput;
    destination.addr += blockOffset;
    destination.token = ctx.nextInputToken;
    return ccu::WriteReduce(lane.nextChannel, destination, source, lane.chunkBytes,
        HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, lane.event);
}

CcuResult RingStageRecordDone(RingLaneContext &lane)
{
    return ccu::NotifyRecord(lane.nextChannel, CKE_IDX, lane.doneMask);
}

CcuResult RingStageWaitDone(RingLaneContext &lane)
{
    return ccu::NotifyWait(lane.prevChannel, CKE_IDX, lane.doneMask);
}

CcuResult RingWriteOutput(RingKernelContext &ctx, RingLaneContext &lane)
{
    ccu::Variable blockOffset;
    blockOffset = lane.chunkOffset;
    for (uint32_t block = 0; block < ctx.arg->rankId; ++block) {
        blockOffset += ctx.recvBytes;
    }

    ccu::LocalAddr source;
    source.addr = ctx.input;
    source.addr += blockOffset;
    source.token = ctx.inputToken;
    ccu::LocalAddr destination;
    destination.addr = ctx.output;
    destination.addr += lane.chunkOffset;
    destination.token = ctx.outputToken;
    return ccu::LocalCopy(destination, source, lane.chunkBytes, lane.event);
}

CcuResult RingPostSync(RingKernelContext &ctx)
{
    constexpr uint16_t postMask = static_cast<uint16_t>(1U << POST_SYNC_ID);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[channelIdx], CKE_IDX, postMask));
    }
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIdx], CKE_IDX, postMask));
    }
    return CCU_SUCCESS;
}
} // namespace

CcuResult CcuReduceScatterHalfRingKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceScatterRing *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount != 2 || kernelArg->rankSize != 4) {
        return CCU_E_PARA;
    }
    RingKernelContext ctx;
    ctx.arg = kernelArg;
    RS2_CCU_CHK_RET(InitRingResources(ctx));
    RS2_CCU_CHK_RET(LoadRingArgs(ctx));
    RS2_CCU_CHK_RET(RingPreSync(ctx));
    for (uint32_t stage = 0; stage < kernelArg->rankSize - 1; ++stage) {
        for (uint32_t laneIdx = 0; laneIdx < HALF_RING_LANE_COUNT; ++laneIdx) {
            RS2_CCU_CHK_RET(RingStageRecordReady(ctx.lanes[laneIdx]));
        }
        for (uint32_t laneIdx = 0; laneIdx < HALF_RING_LANE_COUNT; ++laneIdx) {
            RS2_CCU_CHK_RET(RingStageWaitReady(ctx.lanes[laneIdx]));
        }
        for (uint32_t laneIdx = 0; laneIdx < HALF_RING_LANE_COUNT; ++laneIdx) {
            RS2_CCU_CHK_RET(RingStageWrite(ctx, ctx.lanes[laneIdx], kernelArg->sendBlockIndices[stage]));
        }
        for (uint32_t laneIdx = 0; laneIdx < HALF_RING_LANE_COUNT; ++laneIdx) {
            RS2_CCU_CHK_RET(ccu::EventWait(ctx.lanes[laneIdx].event));
        }
        for (uint32_t laneIdx = 0; laneIdx < HALF_RING_LANE_COUNT; ++laneIdx) {
            RS2_CCU_CHK_RET(RingStageRecordDone(ctx.lanes[laneIdx]));
        }
        for (uint32_t laneIdx = 0; laneIdx < HALF_RING_LANE_COUNT; ++laneIdx) {
            RS2_CCU_CHK_RET(RingStageWaitDone(ctx.lanes[laneIdx]));
        }
    }
    for (uint32_t laneIdx = 0; laneIdx < HALF_RING_LANE_COUNT; ++laneIdx) {
        RS2_CCU_CHK_RET(RingWriteOutput(ctx, ctx.lanes[laneIdx]));
    }
    for (uint32_t laneIdx = 0; laneIdx < HALF_RING_LANE_COUNT; ++laneIdx) {
        RS2_CCU_CHK_RET(ccu::EventWait(ctx.lanes[laneIdx].event));
    }
    RS2_CCU_CHK_RET(RingPostSync(ctx));
    return CCU_SUCCESS;
}

} // namespace ops_hccl
} // namespace final_small512
#undef RS2_CCU_CHK_RET

namespace final_scratch_lite {
namespace ccu = ::AscendC::ccu;

namespace ops_hccl {
namespace {
#define RS2_CCU_CHK_RET(expression) \
    do { \
        const CcuResult result = (expression); \
        if (result != CCU_SUCCESS) { \
            return result; \
        } \
    } while (0)

constexpr uint32_t INPUT_XN_ID = 0;
constexpr uint32_t TOKEN_XN_ID = 1;
constexpr uint32_t POST_SYNC_ID = 2;
constexpr uint32_t CKE_IDX = 0;
constexpr uint64_t KERNEL_MODE_FINALIZE = 1;
constexpr uint64_t KERNEL_MODE_DIRECT_OUTPUT = 2;

struct KernelContext {
    const CcuKernelArgReduceScatterTree *arg = nullptr;
    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable scratch;
    ccu::Variable scratchToken;
    ccu::Variable targetInputOffset;
    ccu::Variable outputOffset;
    ccu::Variable chunkBytes;
    ccu::Variable mode;
    ccu::Variable resultCount;
    std::vector<ccu::Variable> resultOffsets;
    std::vector<ccu::Variable> peerInputs;
    std::vector<ccu::Variable> peerTokens;
    ccu::Event event;
};

CcuResult InitResources(KernelContext &ctx)
{
    ctx.peerInputs.resize(ctx.arg->channelCount);
    ctx.peerTokens.resize(ctx.arg->channelCount);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        ctx.peerInputs[channelIdx] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], INPUT_XN_ID);
        ctx.peerTokens[channelIdx] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], TOKEN_XN_ID);
    }
    return CCU_SUCCESS;
}

CcuResult LoadArgs(KernelContext &ctx)
{
    uint32_t argId = 0;
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.output, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.inputToken, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.outputToken, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.scratch, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.scratchToken, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.targetInputOffset, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.outputOffset, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.chunkBytes, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.mode, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.resultCount, argId++));
    ctx.resultOffsets.resize(MAX_GLOBAL_RESULT_COUNT);
    for (uint32_t resultIdx = 0; resultIdx < MAX_GLOBAL_RESULT_COUNT; ++resultIdx) {
        RS2_CCU_CHK_RET(ccu::LoadArg(ctx.resultOffsets[resultIdx], argId++));
    }
    return CCU_SUCCESS;
}
CcuResult PreSync(KernelContext &ctx)
{
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx], ctx.input, INPUT_XN_ID,
            CKE_IDX, static_cast<uint16_t>(1U << INPUT_XN_ID)));
        RS2_CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx], ctx.inputToken, TOKEN_XN_ID,
            CKE_IDX, static_cast<uint16_t>(1U << TOKEN_XN_ID)));
    }

    constexpr uint16_t waitMask = static_cast<uint16_t>((1U << INPUT_XN_ID) | (1U << TOKEN_XN_ID));
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIdx], CKE_IDX, waitMask));
    }
    return CCU_SUCCESS;
}

CcuResult PostSync(KernelContext &ctx)
{
    constexpr uint16_t postMask = static_cast<uint16_t>(1U << POST_SYNC_ID);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[channelIdx], CKE_IDX, postMask));
    }
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIdx], CKE_IDX, postMask));
    }
    return CCU_SUCCESS;
}

CcuResult ReadPieces(KernelContext &ctx, std::vector<ccu::LocalAddr> &pieces)
{
    ccu::Variable scratchOffset;
    scratchOffset = 0;
    for (uint32_t slotIdx = 0; slotIdx < ctx.arg->scratchSlotBase; ++slotIdx) {
        scratchOffset += ctx.chunkBytes;
    }

    std::vector<ccu::RemoteAddr> remoteInputs(ctx.arg->channelCount);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        pieces[channelIdx].addr = ctx.scratch;
        pieces[channelIdx].addr += scratchOffset;
        pieces[channelIdx].token = ctx.scratchToken;

        remoteInputs[channelIdx].addr = ctx.peerInputs[channelIdx];
        remoteInputs[channelIdx].addr += ctx.targetInputOffset;
        remoteInputs[channelIdx].token = ctx.peerTokens[channelIdx];
        scratchOffset += ctx.chunkBytes;
    }

    uint16_t readMask = 0;
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        const uint16_t channelMask = static_cast<uint16_t>(1U << channelIdx);
        RS2_CCU_CHK_RET(ccu::Read(ctx.arg->channels[channelIdx], pieces[channelIdx], remoteInputs[channelIdx],
            ctx.chunkBytes, ctx.event, channelMask));
        readMask = static_cast<uint16_t>(readMask | channelMask);
    }
    if (readMask != 0) {
        RS2_CCU_CHK_RET(ccu::EventWait(ctx.event, readMask));
    }
    return CCU_SUCCESS;
}

CcuResult ReducePieces(KernelContext &ctx, std::vector<ccu::LocalAddr> &pieces)
{
    uint32_t remainingPieces = static_cast<uint32_t>(pieces.size());
    while (remainingPieces > 1) {
        const uint32_t reducePieces = remainingPieces / 2;
        const uint32_t sourceIdx = remainingPieces - reducePieces;

        ccu::Variable reduceBytes;
        reduceBytes = ctx.chunkBytes;
        for (uint32_t pieceIdx = 1; pieceIdx < reducePieces; ++pieceIdx) {
            reduceBytes += ctx.chunkBytes;
        }
        RS2_CCU_CHK_RET(ccu::LocalReduce(pieces[0], pieces[sourceIdx], reduceBytes, HCCL_DATA_TYPE_FP32,
            HCCL_REDUCE_SUM, ctx.event));
        RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));
        remainingPieces -= reducePieces;
    }
    if (ctx.arg->includeLocalInput != 0) {
        ccu::LocalAddr localInput;
        localInput.addr = ctx.input;
        localInput.addr += ctx.targetInputOffset;
        localInput.token = ctx.inputToken;
        RS2_CCU_CHK_RET(ccu::LocalReduce(pieces[0], localInput, ctx.chunkBytes, HCCL_DATA_TYPE_FP32,
            HCCL_REDUCE_SUM, ctx.event));
        RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));
    }
    return CCU_SUCCESS;
}

CcuResult WriteOutput(KernelContext &ctx, const ccu::LocalAddr &reducedData, bool initializeOutput)
{
    ccu::LocalAddr output;
    output.addr = ctx.output;
    output.addr += ctx.outputOffset;
    output.token = ctx.outputToken;
    if (initializeOutput) {
        RS2_CCU_CHK_RET(ccu::LocalCopy(output, reducedData, ctx.chunkBytes, ctx.event));
    } else {
        RS2_CCU_CHK_RET(ccu::LocalReduce(output, reducedData, ctx.chunkBytes, HCCL_DATA_TYPE_FP32,
            HCCL_REDUCE_SUM, ctx.event));
    }
    RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));
    return CCU_SUCCESS;
}

CcuResult FinalizeResults(KernelContext &ctx)
{
    ccu::LocalAddr output;
    output.addr = ctx.output;
    output.addr += ctx.outputOffset;
    output.token = ctx.outputToken;

    ccu::LocalAddr partial;
    CCU_IF(ctx.resultCount != static_cast<uint64_t>(0)) {
        partial.addr = ctx.scratch;
        partial.addr += ctx.resultOffsets[0];
        partial.token = ctx.scratchToken;
        RS2_CCU_CHK_RET(ccu::LocalReduce(output, partial, ctx.chunkBytes, HCCL_DATA_TYPE_FP32,
            HCCL_REDUCE_SUM, ctx.event));
        RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));

        CCU_IF(ctx.resultCount != static_cast<uint64_t>(1)) {
        partial.addr = ctx.scratch;
        partial.addr += ctx.resultOffsets[1];
        partial.token = ctx.scratchToken;
        RS2_CCU_CHK_RET(ccu::LocalReduce(output, partial, ctx.chunkBytes, HCCL_DATA_TYPE_FP32,
            HCCL_REDUCE_SUM, ctx.event));
        RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));

            CCU_IF(ctx.resultCount != static_cast<uint64_t>(2)) {
                partial.addr = ctx.scratch;
                partial.addr += ctx.resultOffsets[2];
                partial.token = ctx.scratchToken;
                RS2_CCU_CHK_RET(ccu::LocalReduce(output, partial, ctx.chunkBytes, HCCL_DATA_TYPE_FP32,
                    HCCL_REDUCE_SUM, ctx.event));
                RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));

                CCU_IF(ctx.resultCount != static_cast<uint64_t>(3)) {
                    partial.addr = ctx.scratch;
                    partial.addr += ctx.resultOffsets[3];
                    partial.token = ctx.scratchToken;
                    RS2_CCU_CHK_RET(ccu::LocalReduce(output, partial, ctx.chunkBytes, HCCL_DATA_TYPE_FP32,
                        HCCL_REDUCE_SUM, ctx.event));
                    RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));
                }
            }
        }
    }
    return CCU_SUCCESS;
}
} // namespace

CcuResult CcuReduceScatterTreeKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceScatterTree *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount == 0) {
        return CCU_E_PARA;
    }

    KernelContext ctx;
    ctx.arg = kernelArg;
    RS2_CCU_CHK_RET(InitResources(ctx));
    RS2_CCU_CHK_RET(LoadArgs(ctx));
    CCU_IF(ctx.mode == KERNEL_MODE_FINALIZE) {
        RS2_CCU_CHK_RET(FinalizeResults(ctx));
    } CCU_ELSE {
        RS2_CCU_CHK_RET(PreSync(ctx));

        const uint32_t pieceCount = kernelArg->channelCount;
        std::vector<ccu::LocalAddr> pieces(pieceCount);
        RS2_CCU_CHK_RET(ReadPieces(ctx, pieces));
        RS2_CCU_CHK_RET(ReducePieces(ctx, pieces));
        if (kernelArg->writeOutput != 0) {
            RS2_CCU_CHK_RET(WriteOutput(ctx, pieces[0], kernelArg->initializeOutput != 0));
        } else {
            CCU_IF(ctx.mode == KERNEL_MODE_DIRECT_OUTPUT) {
                RS2_CCU_CHK_RET(WriteOutput(ctx, pieces[0], kernelArg->scratchSlotBase == 0));
            }
        }
        RS2_CCU_CHK_RET(PostSync(ctx));
    }
    return CCU_SUCCESS;
}

} // namespace ops_hccl
} // namespace final_scratch_lite
#undef RS2_CCU_CHK_RET

namespace final_epoch_cache {
namespace ccu = ::AscendC::ccu;

namespace ops_hccl {
namespace {
#define RS2_CCU_CHK_RET(expression) \
    do { \
        const CcuResult result = (expression); \
        if (result != CCU_SUCCESS) { \
            return result; \
        } \
    } while (0)

constexpr uint32_t INPUT_XN_ID = 0;
constexpr uint32_t TOKEN_XN_ID = 1;
constexpr uint32_t POST_SYNC_ID = 2;
constexpr uint32_t CKE_IDX = 0;
constexpr uint64_t KERNEL_MODE_FINALIZE = 1;
constexpr uint64_t KERNEL_MODE_DIRECT_OUTPUT = 2;

struct KernelContext {
    const CcuKernelArgReduceScatterTree *arg = nullptr;
    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable scratch;
    ccu::Variable scratchToken;
    ccu::Variable targetInputOffset;
    ccu::Variable outputOffset;
    ccu::Variable chunkBytes;
    ccu::Variable mode;
    ccu::Variable resultCount;
    std::vector<ccu::Variable> resultOffsets;
    std::vector<ccu::Variable> peerInputs;
    std::vector<ccu::Variable> peerTokens;
    ccu::Event event;
};

CcuResult InitResources(KernelContext &ctx)
{
    ctx.peerInputs.resize(ctx.arg->channelCount);
    ctx.peerTokens.resize(ctx.arg->channelCount);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        ctx.peerInputs[channelIdx] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], INPUT_XN_ID);
        ctx.peerTokens[channelIdx] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], TOKEN_XN_ID);
    }
    return CCU_SUCCESS;
}

CcuResult LoadArgs(KernelContext &ctx)
{
    uint32_t argId = 0;
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.output, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.inputToken, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.outputToken, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.scratch, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.scratchToken, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.targetInputOffset, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.outputOffset, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.chunkBytes, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.mode, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.resultCount, argId++));
    ctx.resultOffsets.resize(MAX_GLOBAL_RESULT_COUNT);
    for (uint32_t resultIdx = 0; resultIdx < MAX_GLOBAL_RESULT_COUNT; ++resultIdx) {
        RS2_CCU_CHK_RET(ccu::LoadArg(ctx.resultOffsets[resultIdx], argId++));
    }
    return CCU_SUCCESS;
}
CcuResult PreSync(KernelContext &ctx)
{
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx], ctx.input, INPUT_XN_ID,
            CKE_IDX, static_cast<uint16_t>(1U << INPUT_XN_ID)));
        RS2_CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx], ctx.inputToken, TOKEN_XN_ID,
            CKE_IDX, static_cast<uint16_t>(1U << TOKEN_XN_ID)));
    }

    constexpr uint16_t waitMask = static_cast<uint16_t>((1U << INPUT_XN_ID) | (1U << TOKEN_XN_ID));
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIdx], CKE_IDX, waitMask));
    }
    return CCU_SUCCESS;
}

CcuResult PostSync(KernelContext &ctx)
{
    constexpr uint16_t postMask = static_cast<uint16_t>(1U << POST_SYNC_ID);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[channelIdx], CKE_IDX, postMask));
    }
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIdx], CKE_IDX, postMask));
    }
    return CCU_SUCCESS;
}

CcuResult ReadPieces(KernelContext &ctx, std::vector<ccu::LocalAddr> &pieces)
{
    ccu::Variable scratchOffset;
    scratchOffset = 0;
    for (uint32_t slotIdx = 0; slotIdx < ctx.arg->scratchSlotBase; ++slotIdx) {
        scratchOffset += ctx.chunkBytes;
    }

    std::vector<ccu::RemoteAddr> remoteInputs(ctx.arg->channelCount);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        pieces[channelIdx].addr = ctx.scratch;
        pieces[channelIdx].addr += scratchOffset;
        pieces[channelIdx].token = ctx.scratchToken;

        remoteInputs[channelIdx].addr = ctx.peerInputs[channelIdx];
        remoteInputs[channelIdx].addr += ctx.targetInputOffset;
        remoteInputs[channelIdx].token = ctx.peerTokens[channelIdx];
        scratchOffset += ctx.chunkBytes;
    }

    uint16_t readMask = 0;
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        const uint16_t channelMask = static_cast<uint16_t>(1U << channelIdx);
        RS2_CCU_CHK_RET(ccu::Read(ctx.arg->channels[channelIdx], pieces[channelIdx], remoteInputs[channelIdx],
            ctx.chunkBytes, ctx.event, channelMask));
        readMask = static_cast<uint16_t>(readMask | channelMask);
    }
    if (readMask != 0) {
        RS2_CCU_CHK_RET(ccu::EventWait(ctx.event, readMask));
    }
    return CCU_SUCCESS;
}

CcuResult ReducePieces(KernelContext &ctx, std::vector<ccu::LocalAddr> &pieces)
{
    uint32_t remainingPieces = static_cast<uint32_t>(pieces.size());
    while (remainingPieces > 1) {
        const uint32_t reducePieces = remainingPieces / 2;
        const uint32_t sourceIdx = remainingPieces - reducePieces;

        ccu::Variable reduceBytes;
        reduceBytes = ctx.chunkBytes;
        for (uint32_t pieceIdx = 1; pieceIdx < reducePieces; ++pieceIdx) {
            reduceBytes += ctx.chunkBytes;
        }
        RS2_CCU_CHK_RET(ccu::LocalReduce(pieces[0], pieces[sourceIdx], reduceBytes, HCCL_DATA_TYPE_FP32,
            HCCL_REDUCE_SUM, ctx.event));
        RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));
        remainingPieces -= reducePieces;
    }
    if (ctx.arg->includeLocalInput != 0) {
        ccu::LocalAddr localInput;
        localInput.addr = ctx.input;
        localInput.addr += ctx.targetInputOffset;
        localInput.token = ctx.inputToken;
        RS2_CCU_CHK_RET(ccu::LocalReduce(pieces[0], localInput, ctx.chunkBytes, HCCL_DATA_TYPE_FP32,
            HCCL_REDUCE_SUM, ctx.event));
        RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));
    }
    return CCU_SUCCESS;
}

CcuResult WriteOutput(KernelContext &ctx, const ccu::LocalAddr &reducedData, bool initializeOutput)
{
    ccu::LocalAddr output;
    output.addr = ctx.output;
    output.addr += ctx.outputOffset;
    output.token = ctx.outputToken;
    if (initializeOutput) {
        RS2_CCU_CHK_RET(ccu::LocalCopy(output, reducedData, ctx.chunkBytes, ctx.event));
    } else {
        RS2_CCU_CHK_RET(ccu::LocalReduce(output, reducedData, ctx.chunkBytes, HCCL_DATA_TYPE_FP32,
            HCCL_REDUCE_SUM, ctx.event));
    }
    RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));
    return CCU_SUCCESS;
}

CcuResult FinalizeResults(KernelContext &ctx)
{
    ccu::LocalAddr output;
    output.addr = ctx.output;
    output.addr += ctx.outputOffset;
    output.token = ctx.outputToken;

    ccu::LocalAddr partial;
    CCU_IF(ctx.resultCount != static_cast<uint64_t>(0)) {
        partial.addr = ctx.scratch;
        partial.addr += ctx.resultOffsets[0];
        partial.token = ctx.scratchToken;
        RS2_CCU_CHK_RET(ccu::LocalReduce(output, partial, ctx.chunkBytes, HCCL_DATA_TYPE_FP32,
            HCCL_REDUCE_SUM, ctx.event));
        RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));

        CCU_IF(ctx.resultCount != static_cast<uint64_t>(1)) {
        partial.addr = ctx.scratch;
        partial.addr += ctx.resultOffsets[1];
        partial.token = ctx.scratchToken;
        RS2_CCU_CHK_RET(ccu::LocalReduce(output, partial, ctx.chunkBytes, HCCL_DATA_TYPE_FP32,
            HCCL_REDUCE_SUM, ctx.event));
        RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));

            CCU_IF(ctx.resultCount != static_cast<uint64_t>(2)) {
                partial.addr = ctx.scratch;
                partial.addr += ctx.resultOffsets[2];
                partial.token = ctx.scratchToken;
                RS2_CCU_CHK_RET(ccu::LocalReduce(output, partial, ctx.chunkBytes, HCCL_DATA_TYPE_FP32,
                    HCCL_REDUCE_SUM, ctx.event));
                RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));

                CCU_IF(ctx.resultCount != static_cast<uint64_t>(3)) {
                    partial.addr = ctx.scratch;
                    partial.addr += ctx.resultOffsets[3];
                    partial.token = ctx.scratchToken;
                    RS2_CCU_CHK_RET(ccu::LocalReduce(output, partial, ctx.chunkBytes, HCCL_DATA_TYPE_FP32,
                        HCCL_REDUCE_SUM, ctx.event));
                    RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));
                }
            }
        }
    }
    return CCU_SUCCESS;
}
} // namespace

CcuResult CcuReduceScatterTreeKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceScatterTree *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount == 0) {
        return CCU_E_PARA;
    }

    KernelContext ctx;
    ctx.arg = kernelArg;
    RS2_CCU_CHK_RET(InitResources(ctx));
    RS2_CCU_CHK_RET(LoadArgs(ctx));
    CCU_IF(ctx.mode == KERNEL_MODE_FINALIZE) {
        RS2_CCU_CHK_RET(FinalizeResults(ctx));
    } CCU_ELSE {
        RS2_CCU_CHK_RET(PreSync(ctx));

        const uint32_t pieceCount = kernelArg->channelCount;
        std::vector<ccu::LocalAddr> pieces(pieceCount);
        RS2_CCU_CHK_RET(ReadPieces(ctx, pieces));
        RS2_CCU_CHK_RET(ReducePieces(ctx, pieces));
        if (kernelArg->writeOutput != 0) {
            RS2_CCU_CHK_RET(WriteOutput(ctx, pieces[0], kernelArg->initializeOutput != 0));
        } else {
            CCU_IF(ctx.mode == KERNEL_MODE_DIRECT_OUTPUT) {
                RS2_CCU_CHK_RET(WriteOutput(ctx, pieces[0], kernelArg->scratchSlotBase == 0));
            }
        }
        RS2_CCU_CHK_RET(PostSync(ctx));
    }
    return CCU_SUCCESS;
}

} // namespace ops_hccl
} // namespace final_epoch_cache
#undef RS2_CCU_CHK_RET

namespace final_direct_pipeline {
namespace ccu = ::AscendC::ccu;

namespace ops_hccl {
namespace {
#define RS2_CCU_CHK_RET(expression) \
    do { \
        const CcuResult result = (expression); \
        if (result != CCU_SUCCESS) { \
            return result; \
        } \
    } while (0)

constexpr uint32_t INPUT_XN_ID = 0;
constexpr uint32_t TOKEN_XN_ID = 1;
constexpr uint32_t POST_SYNC_ID = 2;
constexpr uint32_t CKE_IDX = 0;
constexpr uint64_t KERNEL_MODE_FINALIZE = 1;
constexpr uint64_t KERNEL_MODE_DIRECT_OUTPUT = 2;

struct KernelContext {
    const CcuKernelArgReduceScatterTree *arg = nullptr;
    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable scratch;
    ccu::Variable scratchToken;
    ccu::Variable targetInputOffset;
    ccu::Variable outputOffset;
    ccu::Variable chunkBytes;
    ccu::Variable mode;
    ccu::Variable resultCount;
    std::vector<ccu::Variable> resultOffsets;
    std::vector<ccu::Variable> peerInputs;
    std::vector<ccu::Variable> peerTokens;
    ccu::Event event;
};

CcuResult InitResources(KernelContext &ctx)
{
    ctx.peerInputs.resize(ctx.arg->channelCount);
    ctx.peerTokens.resize(ctx.arg->channelCount);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        ctx.peerInputs[channelIdx] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], INPUT_XN_ID);
        ctx.peerTokens[channelIdx] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], TOKEN_XN_ID);
    }
    return CCU_SUCCESS;
}

CcuResult LoadArgs(KernelContext &ctx)
{
    uint32_t argId = 0;
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.output, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.inputToken, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.outputToken, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.scratch, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.scratchToken, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.targetInputOffset, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.outputOffset, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.chunkBytes, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.mode, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.resultCount, argId++));
    ctx.resultOffsets.resize(MAX_GLOBAL_RESULT_COUNT);
    for (uint32_t resultIdx = 0; resultIdx < MAX_GLOBAL_RESULT_COUNT; ++resultIdx) {
        RS2_CCU_CHK_RET(ccu::LoadArg(ctx.resultOffsets[resultIdx], argId++));
    }
    return CCU_SUCCESS;
}
CcuResult PreSync(KernelContext &ctx)
{
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx], ctx.input, INPUT_XN_ID,
            CKE_IDX, static_cast<uint16_t>(1U << INPUT_XN_ID)));
        RS2_CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx], ctx.inputToken, TOKEN_XN_ID,
            CKE_IDX, static_cast<uint16_t>(1U << TOKEN_XN_ID)));
    }

    constexpr uint16_t waitMask = static_cast<uint16_t>((1U << INPUT_XN_ID) | (1U << TOKEN_XN_ID));
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIdx], CKE_IDX, waitMask));
    }
    return CCU_SUCCESS;
}

CcuResult PostSync(KernelContext &ctx)
{
    constexpr uint16_t postMask = static_cast<uint16_t>(1U << POST_SYNC_ID);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[channelIdx], CKE_IDX, postMask));
    }
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIdx], CKE_IDX, postMask));
    }
    return CCU_SUCCESS;
}

CcuResult ReadPieces(KernelContext &ctx, std::vector<ccu::LocalAddr> &pieces)
{
    ccu::Variable scratchOffset;
    scratchOffset = 0;
    for (uint32_t slotIdx = 0; slotIdx < ctx.arg->scratchSlotBase; ++slotIdx) {
        scratchOffset += ctx.chunkBytes;
    }

    std::vector<ccu::RemoteAddr> remoteInputs(ctx.arg->channelCount);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        pieces[channelIdx].addr = ctx.scratch;
        pieces[channelIdx].addr += scratchOffset;
        pieces[channelIdx].token = ctx.scratchToken;

        remoteInputs[channelIdx].addr = ctx.peerInputs[channelIdx];
        remoteInputs[channelIdx].addr += ctx.targetInputOffset;
        remoteInputs[channelIdx].token = ctx.peerTokens[channelIdx];
        scratchOffset += ctx.chunkBytes;
    }

    uint16_t readMask = 0;
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        const uint16_t channelMask = static_cast<uint16_t>(1U << channelIdx);
        RS2_CCU_CHK_RET(ccu::Read(ctx.arg->channels[channelIdx], pieces[channelIdx], remoteInputs[channelIdx],
            ctx.chunkBytes, ctx.event, channelMask));
        readMask = static_cast<uint16_t>(readMask | channelMask);
    }
    if (readMask != 0) {
        RS2_CCU_CHK_RET(ccu::EventWait(ctx.event, readMask));
    }
    return CCU_SUCCESS;
}

CcuResult IssueFirstLayerPairReadyReads(KernelContext &ctx, std::vector<ccu::LocalAddr> &pieces,
    std::vector<uint16_t> &readyMasks)
{
    ccu::Variable scratchOffset;
    scratchOffset = 0;
    for (uint32_t slotIdx = 0; slotIdx < ctx.arg->scratchSlotBase; ++slotIdx) {
        scratchOffset += ctx.chunkBytes;
    }

    std::vector<ccu::RemoteAddr> remoteInputs(ctx.arg->channelCount);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        pieces[channelIdx].addr = ctx.scratch;
        pieces[channelIdx].addr += scratchOffset;
        pieces[channelIdx].token = ctx.scratchToken;

        remoteInputs[channelIdx].addr = ctx.peerInputs[channelIdx];
        remoteInputs[channelIdx].addr += ctx.targetInputOffset;
        remoteInputs[channelIdx].token = ctx.peerTokens[channelIdx];
        scratchOffset += ctx.chunkBytes;
    }

    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        const uint16_t channelMask = static_cast<uint16_t>(1U << channelIdx);
        RS2_CCU_CHK_RET(ccu::Read(ctx.arg->channels[channelIdx], pieces[channelIdx], remoteInputs[channelIdx],
            ctx.chunkBytes, ctx.event, channelMask));
        readyMasks[channelIdx] = channelMask;
    }
    return CCU_SUCCESS;
}

CcuResult ReduceRemainingPieces(KernelContext &ctx, std::vector<ccu::LocalAddr> &pieces,
    uint32_t remainingPieces)
{
    while (remainingPieces > 1) {
        const uint32_t reducePieces = remainingPieces / 2;
        const uint32_t sourceIdx = remainingPieces - reducePieces;

        ccu::Variable reduceBytes;
        reduceBytes = ctx.chunkBytes;
        for (uint32_t pieceIdx = 1; pieceIdx < reducePieces; ++pieceIdx) {
            reduceBytes += ctx.chunkBytes;
        }
        RS2_CCU_CHK_RET(ccu::LocalReduce(pieces[0], pieces[sourceIdx], reduceBytes, HCCL_DATA_TYPE_FP32,
            HCCL_REDUCE_SUM, ctx.event));
        RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));
        remainingPieces -= reducePieces;
    }
    return CCU_SUCCESS;
}

CcuResult ReduceLocalInput(KernelContext &ctx, ccu::LocalAddr &reducedPiece)
{
    if (ctx.arg->includeLocalInput == 0) {
        return CCU_SUCCESS;
    }
    ccu::LocalAddr localInput;
    localInput.addr = ctx.input;
    localInput.addr += ctx.targetInputOffset;
    localInput.token = ctx.inputToken;
    RS2_CCU_CHK_RET(ccu::LocalReduce(reducedPiece, localInput, ctx.chunkBytes, HCCL_DATA_TYPE_FP32,
        HCCL_REDUCE_SUM, ctx.event));
    RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));
    return CCU_SUCCESS;
}

CcuResult ReduceFirstLayerPairReady(KernelContext &ctx, std::vector<ccu::LocalAddr> &pieces,
    std::vector<uint16_t> &readyMasks)
{
    uint32_t remainingPieces = static_cast<uint32_t>(pieces.size());
    if (remainingPieces > 1) {
        const uint32_t reducePieces = remainingPieces / 2;
        const uint32_t sourceIdx = remainingPieces - reducePieces;
        for (uint32_t pieceIdx = 0; pieceIdx < reducePieces; ++pieceIdx) {
            const uint32_t sourcePieceIdx = sourceIdx + pieceIdx;
            const uint16_t inputMask = static_cast<uint16_t>(
                readyMasks[pieceIdx] | readyMasks[sourcePieceIdx]);
            RS2_CCU_CHK_RET(ccu::EventWait(ctx.event, inputMask));
            const uint16_t outputMask = static_cast<uint16_t>(1U << pieceIdx);
            RS2_CCU_CHK_RET(ccu::LocalReduce(pieces[pieceIdx], pieces[sourcePieceIdx], ctx.chunkBytes,
                HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.event, outputMask));
            readyMasks[pieceIdx] = outputMask;
        }
        remainingPieces -= reducePieces;
    }

    uint16_t firstLayerReadyMask = 0;
    for (uint32_t pieceIdx = 0; pieceIdx < remainingPieces; ++pieceIdx) {
        firstLayerReadyMask = static_cast<uint16_t>(firstLayerReadyMask | readyMasks[pieceIdx]);
    }
    if (firstLayerReadyMask != 0) {
        RS2_CCU_CHK_RET(ccu::EventWait(ctx.event, firstLayerReadyMask));
    }
    RS2_CCU_CHK_RET(ReduceRemainingPieces(ctx, pieces, remainingPieces));
    return ReduceLocalInput(ctx, pieces[0]);
}

CcuResult ReducePieces(KernelContext &ctx, std::vector<ccu::LocalAddr> &pieces)
{
    RS2_CCU_CHK_RET(ReduceRemainingPieces(ctx, pieces, static_cast<uint32_t>(pieces.size())));
    return ReduceLocalInput(ctx, pieces[0]);
}

CcuResult WriteOutput(KernelContext &ctx, const ccu::LocalAddr &reducedData, bool initializeOutput)
{
    ccu::LocalAddr output;
    output.addr = ctx.output;
    output.addr += ctx.outputOffset;
    output.token = ctx.outputToken;
    if (initializeOutput) {
        RS2_CCU_CHK_RET(ccu::LocalCopy(output, reducedData, ctx.chunkBytes, ctx.event));
    } else {
        RS2_CCU_CHK_RET(ccu::LocalReduce(output, reducedData, ctx.chunkBytes, HCCL_DATA_TYPE_FP32,
            HCCL_REDUCE_SUM, ctx.event));
    }
    RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));
    return CCU_SUCCESS;
}

CcuResult FinalizeResults(KernelContext &ctx)
{
    ccu::LocalAddr output;
    output.addr = ctx.output;
    output.addr += ctx.outputOffset;
    output.token = ctx.outputToken;

    ccu::LocalAddr partial;
    CCU_IF(ctx.resultCount != static_cast<uint64_t>(0)) {
        partial.addr = ctx.scratch;
        partial.addr += ctx.resultOffsets[0];
        partial.token = ctx.scratchToken;
        RS2_CCU_CHK_RET(ccu::LocalReduce(output, partial, ctx.chunkBytes, HCCL_DATA_TYPE_FP32,
            HCCL_REDUCE_SUM, ctx.event));
        RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));

        CCU_IF(ctx.resultCount != static_cast<uint64_t>(1)) {
        partial.addr = ctx.scratch;
        partial.addr += ctx.resultOffsets[1];
        partial.token = ctx.scratchToken;
        RS2_CCU_CHK_RET(ccu::LocalReduce(output, partial, ctx.chunkBytes, HCCL_DATA_TYPE_FP32,
            HCCL_REDUCE_SUM, ctx.event));
        RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));

            CCU_IF(ctx.resultCount != static_cast<uint64_t>(2)) {
                partial.addr = ctx.scratch;
                partial.addr += ctx.resultOffsets[2];
                partial.token = ctx.scratchToken;
                RS2_CCU_CHK_RET(ccu::LocalReduce(output, partial, ctx.chunkBytes, HCCL_DATA_TYPE_FP32,
                    HCCL_REDUCE_SUM, ctx.event));
                RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));

                CCU_IF(ctx.resultCount != static_cast<uint64_t>(3)) {
                    partial.addr = ctx.scratch;
                    partial.addr += ctx.resultOffsets[3];
                    partial.token = ctx.scratchToken;
                    RS2_CCU_CHK_RET(ccu::LocalReduce(output, partial, ctx.chunkBytes, HCCL_DATA_TYPE_FP32,
                        HCCL_REDUCE_SUM, ctx.event));
                    RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));
                }
            }
        }
    }
    return CCU_SUCCESS;
}
} // namespace

CcuResult CcuReduceScatterTreeKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceScatterTree *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount == 0) {
        return CCU_E_PARA;
    }

    KernelContext ctx;
    ctx.arg = kernelArg;
    RS2_CCU_CHK_RET(InitResources(ctx));
    RS2_CCU_CHK_RET(LoadArgs(ctx));
    CCU_IF(ctx.mode == KERNEL_MODE_FINALIZE) {
        RS2_CCU_CHK_RET(FinalizeResults(ctx));
    } CCU_ELSE {
        RS2_CCU_CHK_RET(PreSync(ctx));

        const uint32_t pieceCount = kernelArg->channelCount;
        std::vector<ccu::LocalAddr> pieces(pieceCount);
        if (kernelArg->useFirstLayerPairReady != 0) {
            std::vector<uint16_t> readyMasks(pieceCount);
            RS2_CCU_CHK_RET(IssueFirstLayerPairReadyReads(ctx, pieces, readyMasks));
            RS2_CCU_CHK_RET(ReduceFirstLayerPairReady(ctx, pieces, readyMasks));
        } else {
            RS2_CCU_CHK_RET(ReadPieces(ctx, pieces));
            RS2_CCU_CHK_RET(ReducePieces(ctx, pieces));
        }
        if (kernelArg->writeOutput != 0) {
            RS2_CCU_CHK_RET(WriteOutput(ctx, pieces[0], kernelArg->initializeOutput != 0));
        } else {
            CCU_IF(ctx.mode == KERNEL_MODE_DIRECT_OUTPUT) {
                RS2_CCU_CHK_RET(WriteOutput(ctx, pieces[0], kernelArg->scratchSlotBase == 0));
            }
        }
        RS2_CCU_CHK_RET(PostSync(ctx));
    }
    return CCU_SUCCESS;
}

} // namespace ops_hccl
} // namespace final_direct_pipeline
#undef RS2_CCU_CHK_RET

namespace final_adaptive41 {
namespace ccu = ::AscendC::ccu;

namespace ops_hccl {
namespace {
#define RS2_CCU_CHK_RET(expression) \
    do { \
        const CcuResult result = (expression); \
        if (result != CCU_SUCCESS) { \
            return result; \
        } \
    } while (0)

constexpr uint32_t INPUT_XN_ID = 0;
constexpr uint32_t TOKEN_XN_ID = 1;
constexpr uint32_t POST_SYNC_ID = 2;
constexpr uint32_t RING_STAGE_READY_ID = 3;
constexpr uint32_t RING_STAGE_DONE_ID = 4;
constexpr uint32_t CKE_IDX = 0;
constexpr uint64_t KERNEL_MODE_FINALIZE = 1;
constexpr uint64_t KERNEL_MODE_DIRECT_OUTPUT = 2;

struct KernelContext {
    const CcuKernelArgReduceScatterTree *arg = nullptr;
    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable scratch;
    ccu::Variable scratchToken;
    ccu::Variable targetInputOffset;
    ccu::Variable outputOffset;
    ccu::Variable chunkBytes;
    ccu::Variable mode;
    ccu::Variable resultCount;
    std::vector<ccu::Variable> resultOffsets;
    std::vector<ccu::Variable> peerInputs;
    std::vector<ccu::Variable> peerTokens;
    ccu::Event event;
};

CcuResult InitResources(KernelContext &ctx)
{
    ctx.peerInputs.resize(ctx.arg->channelCount);
    ctx.peerTokens.resize(ctx.arg->channelCount);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        ctx.peerInputs[channelIdx] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], INPUT_XN_ID);
        ctx.peerTokens[channelIdx] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], TOKEN_XN_ID);
    }
    return CCU_SUCCESS;
}

CcuResult LoadArgs(KernelContext &ctx)
{
    uint32_t argId = 0;
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.output, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.inputToken, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.outputToken, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.scratch, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.scratchToken, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.targetInputOffset, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.outputOffset, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.chunkBytes, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.mode, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.resultCount, argId++));
    ctx.resultOffsets.resize(MAX_GLOBAL_RESULT_COUNT);
    for (uint32_t resultIdx = 0; resultIdx < MAX_GLOBAL_RESULT_COUNT; ++resultIdx) {
        RS2_CCU_CHK_RET(ccu::LoadArg(ctx.resultOffsets[resultIdx], argId++));
    }
    return CCU_SUCCESS;
}
CcuResult PreSync(KernelContext &ctx)
{
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx], ctx.input, INPUT_XN_ID,
            CKE_IDX, static_cast<uint16_t>(1U << INPUT_XN_ID)));
        RS2_CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx], ctx.inputToken, TOKEN_XN_ID,
            CKE_IDX, static_cast<uint16_t>(1U << TOKEN_XN_ID)));
    }

    constexpr uint16_t waitMask = static_cast<uint16_t>((1U << INPUT_XN_ID) | (1U << TOKEN_XN_ID));
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIdx], CKE_IDX, waitMask));
    }
    return CCU_SUCCESS;
}

CcuResult PostSync(KernelContext &ctx)
{
    constexpr uint16_t postMask = static_cast<uint16_t>(1U << POST_SYNC_ID);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[channelIdx], CKE_IDX, postMask));
    }
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIdx], CKE_IDX, postMask));
    }
    return CCU_SUCCESS;
}

CcuResult ReadPieces(KernelContext &ctx, std::vector<ccu::LocalAddr> &pieces)
{
    ccu::Variable scratchOffset;
    scratchOffset = 0;
    for (uint32_t slotIdx = 0; slotIdx < ctx.arg->scratchSlotBase; ++slotIdx) {
        scratchOffset += ctx.chunkBytes;
    }

    std::vector<ccu::RemoteAddr> remoteInputs(ctx.arg->channelCount);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        pieces[channelIdx].addr = ctx.scratch;
        pieces[channelIdx].addr += scratchOffset;
        pieces[channelIdx].token = ctx.scratchToken;

        remoteInputs[channelIdx].addr = ctx.peerInputs[channelIdx];
        remoteInputs[channelIdx].addr += ctx.targetInputOffset;
        remoteInputs[channelIdx].token = ctx.peerTokens[channelIdx];
        scratchOffset += ctx.chunkBytes;
    }

    if (ctx.arg->includeLocalInput != 0) {
        const uint32_t localPieceIdx = ctx.arg->channelCount;
        pieces[localPieceIdx].addr = ctx.scratch;
        pieces[localPieceIdx].addr += scratchOffset;
        pieces[localPieceIdx].token = ctx.scratchToken;

        ccu::LocalAddr localInput;
        localInput.addr = ctx.input;
        localInput.addr += ctx.targetInputOffset;
        localInput.token = ctx.inputToken;
        RS2_CCU_CHK_RET(ccu::LocalCopy(pieces[localPieceIdx], localInput, ctx.chunkBytes, ctx.event));
        RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));
    }

    uint16_t readMask = 0;
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        const uint16_t channelMask = static_cast<uint16_t>(1U << channelIdx);
        RS2_CCU_CHK_RET(ccu::Read(ctx.arg->channels[channelIdx], pieces[channelIdx], remoteInputs[channelIdx],
            ctx.chunkBytes, ctx.event, channelMask));
        readMask = static_cast<uint16_t>(readMask | channelMask);
    }
    if (readMask != 0) {
        RS2_CCU_CHK_RET(ccu::EventWait(ctx.event, readMask));
    }
    return CCU_SUCCESS;
}

CcuResult ReducePieces(KernelContext &ctx, std::vector<ccu::LocalAddr> &pieces)
{
    uint32_t remainingPieces = static_cast<uint32_t>(pieces.size());
    while (remainingPieces > 1) {
        const uint32_t reducePieces = remainingPieces / 2;
        const uint32_t sourceIdx = remainingPieces - reducePieces;

        ccu::Variable reduceBytes;
        reduceBytes = ctx.chunkBytes;
        for (uint32_t pieceIdx = 1; pieceIdx < reducePieces; ++pieceIdx) {
            reduceBytes += ctx.chunkBytes;
        }
        RS2_CCU_CHK_RET(ccu::LocalReduce(pieces[0], pieces[sourceIdx], reduceBytes, HCCL_DATA_TYPE_FP32,
            HCCL_REDUCE_SUM, ctx.event));
        RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));
        remainingPieces -= reducePieces;
    }
    return CCU_SUCCESS;
}

CcuResult WriteOutput(KernelContext &ctx, const ccu::LocalAddr &reducedData, bool initializeOutput)
{
    ccu::LocalAddr output;
    output.addr = ctx.output;
    output.addr += ctx.outputOffset;
    output.token = ctx.outputToken;
    if (initializeOutput) {
        RS2_CCU_CHK_RET(ccu::LocalCopy(output, reducedData, ctx.chunkBytes, ctx.event));
    } else {
        RS2_CCU_CHK_RET(ccu::LocalReduce(output, reducedData, ctx.chunkBytes, HCCL_DATA_TYPE_FP32,
            HCCL_REDUCE_SUM, ctx.event));
    }
    RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));
    return CCU_SUCCESS;
}

CcuResult FinalizeResults(KernelContext &ctx)
{
    ccu::LocalAddr output;
    output.addr = ctx.output;
    output.addr += ctx.outputOffset;
    output.token = ctx.outputToken;

    ccu::LocalAddr partial;
    partial.addr = ctx.scratch;
    partial.addr += ctx.resultOffsets[0];
    partial.token = ctx.scratchToken;
    RS2_CCU_CHK_RET(ccu::LocalCopy(output, partial, ctx.chunkBytes, ctx.event));
    RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));

    CCU_IF(ctx.resultCount != static_cast<uint64_t>(1)) {
        partial.addr = ctx.scratch;
        partial.addr += ctx.resultOffsets[1];
        partial.token = ctx.scratchToken;
        RS2_CCU_CHK_RET(ccu::LocalReduce(output, partial, ctx.chunkBytes, HCCL_DATA_TYPE_FP32,
            HCCL_REDUCE_SUM, ctx.event));
        RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));

        CCU_IF(ctx.resultCount != static_cast<uint64_t>(2)) {
            partial.addr = ctx.scratch;
            partial.addr += ctx.resultOffsets[2];
            partial.token = ctx.scratchToken;
            RS2_CCU_CHK_RET(ccu::LocalReduce(output, partial, ctx.chunkBytes, HCCL_DATA_TYPE_FP32,
                HCCL_REDUCE_SUM, ctx.event));
            RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));

            CCU_IF(ctx.resultCount != static_cast<uint64_t>(3)) {
                partial.addr = ctx.scratch;
                partial.addr += ctx.resultOffsets[3];
                partial.token = ctx.scratchToken;
                RS2_CCU_CHK_RET(ccu::LocalReduce(output, partial, ctx.chunkBytes, HCCL_DATA_TYPE_FP32,
                    HCCL_REDUCE_SUM, ctx.event));
                RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));
            }
        }
    }
    return CCU_SUCCESS;
}
} // namespace

CcuResult CcuReduceScatterTreeKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceScatterTree *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount == 0) {
        return CCU_E_PARA;
    }

    KernelContext ctx;
    ctx.arg = kernelArg;
    RS2_CCU_CHK_RET(InitResources(ctx));
    RS2_CCU_CHK_RET(LoadArgs(ctx));
    CCU_IF(ctx.mode == KERNEL_MODE_FINALIZE) {
        RS2_CCU_CHK_RET(FinalizeResults(ctx));
    } CCU_ELSE {
        RS2_CCU_CHK_RET(PreSync(ctx));

        const uint32_t pieceCount = kernelArg->channelCount + kernelArg->includeLocalInput;
        std::vector<ccu::LocalAddr> pieces(pieceCount);
        RS2_CCU_CHK_RET(ReadPieces(ctx, pieces));
        RS2_CCU_CHK_RET(ReducePieces(ctx, pieces));
        if (kernelArg->writeOutput != 0) {
            RS2_CCU_CHK_RET(WriteOutput(ctx, pieces[0], kernelArg->initializeOutput != 0));
        } else {
            CCU_IF(ctx.mode == KERNEL_MODE_DIRECT_OUTPUT) {
                RS2_CCU_CHK_RET(WriteOutput(ctx, pieces[0], kernelArg->scratchSlotBase == 0));
            }
        }
        RS2_CCU_CHK_RET(PostSync(ctx));
    }
    return CCU_SUCCESS;
}

namespace {
struct RingKernelContext {
    const CcuKernelArgReduceScatterRing *arg = nullptr;
    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable recvBytes;
    ccu::Variable chunkOffset;
    ccu::Variable chunkBytes;
    ccu::Variable firstSliceBytes;
    ccu::Variable secondSliceBytes;
    ccu::Variable nextInput;
    ccu::Variable nextInputToken;
    ccu::Event event;
};

CcuResult InitRingResources(RingKernelContext &ctx)
{
    ctx.nextInput = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[0], INPUT_XN_ID);
    ctx.nextInputToken = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[0], TOKEN_XN_ID);
    return CCU_SUCCESS;
}

CcuResult LoadRingArgs(RingKernelContext &ctx)
{
    uint32_t argId = 0;
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.output, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.inputToken, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.outputToken, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.recvBytes, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.chunkOffset, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.chunkBytes, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.firstSliceBytes, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.secondSliceBytes, argId++));
    return CCU_SUCCESS;
}

CcuResult RingPreSync(RingKernelContext &ctx)
{
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx], ctx.input, INPUT_XN_ID,
            CKE_IDX, static_cast<uint16_t>(1U << INPUT_XN_ID)));
        RS2_CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx], ctx.inputToken, TOKEN_XN_ID,
            CKE_IDX, static_cast<uint16_t>(1U << TOKEN_XN_ID)));
    }
    constexpr uint16_t waitMask = static_cast<uint16_t>((1U << INPUT_XN_ID) | (1U << TOKEN_XN_ID));
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIdx], CKE_IDX, waitMask));
    }
    return CCU_SUCCESS;
}

CcuResult RingStage(RingKernelContext &ctx, uint32_t blockIdx)
{
    constexpr uint16_t readyMask = static_cast<uint16_t>(1U << RING_STAGE_READY_ID);
    constexpr uint16_t doneMask = static_cast<uint16_t>(1U << RING_STAGE_DONE_ID);
    const ChannelHandle nextChannel = ctx.arg->channels[0];
    const ChannelHandle prevChannel = ctx.arg->channels[1];
    RS2_CCU_CHK_RET(ccu::NotifyRecord(prevChannel, CKE_IDX, readyMask));
    RS2_CCU_CHK_RET(ccu::NotifyWait(nextChannel, CKE_IDX, readyMask));

    ccu::Variable blockOffset;
    blockOffset = ctx.chunkOffset;
    for (uint32_t block = 0; block < blockIdx; ++block) {
        blockOffset += ctx.recvBytes;
    }

    ccu::LocalAddr source;
    source.addr = ctx.input;
    source.addr += blockOffset;
    source.token = ctx.inputToken;
    ccu::RemoteAddr destination;
    destination.addr = ctx.nextInput;
    destination.addr += blockOffset;
    destination.token = ctx.nextInputToken;

    CCU_IF(ctx.firstSliceBytes != static_cast<uint64_t>(0)) {
        RS2_CCU_CHK_RET(ccu::WriteReduce(nextChannel, destination, source, ctx.firstSliceBytes,
            HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.event, 1));
    } CCU_ELSE {
        RS2_CCU_CHK_RET(ccu::EventRecord(ctx.event, 1));
    }
    source.addr += ctx.firstSliceBytes;
    destination.addr += ctx.firstSliceBytes;
    CCU_IF(ctx.secondSliceBytes != static_cast<uint64_t>(0)) {
        RS2_CCU_CHK_RET(ccu::WriteReduce(nextChannel, destination, source, ctx.secondSliceBytes,
            HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.event, 2));
    } CCU_ELSE {
        RS2_CCU_CHK_RET(ccu::EventRecord(ctx.event, 2));
    }
    RS2_CCU_CHK_RET(ccu::EventWait(ctx.event, 3));

    RS2_CCU_CHK_RET(ccu::NotifyRecord(nextChannel, CKE_IDX, doneMask));
    RS2_CCU_CHK_RET(ccu::NotifyWait(prevChannel, CKE_IDX, doneMask));
    return CCU_SUCCESS;
}

CcuResult RingWriteOutput(RingKernelContext &ctx)
{
    ccu::Variable blockOffset;
    blockOffset = ctx.chunkOffset;
    for (uint32_t block = 0; block < ctx.arg->rankId; ++block) {
        blockOffset += ctx.recvBytes;
    }
    ccu::LocalAddr source;
    source.addr = ctx.input;
    source.addr += blockOffset;
    source.token = ctx.inputToken;
    ccu::LocalAddr destination;
    destination.addr = ctx.output;
    destination.addr += ctx.chunkOffset;
    destination.token = ctx.outputToken;
    RS2_CCU_CHK_RET(ccu::LocalCopy(destination, source, ctx.chunkBytes, ctx.event));
    RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));
    return CCU_SUCCESS;
}

CcuResult RingPostSync(RingKernelContext &ctx)
{
    constexpr uint16_t postMask = static_cast<uint16_t>(1U << POST_SYNC_ID);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[channelIdx], CKE_IDX, postMask));
    }
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIdx], CKE_IDX, postMask));
    }
    return CCU_SUCCESS;
}
} // namespace

CcuResult CcuReduceScatterRingKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgReduceScatterRing *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount != 2 || kernelArg->rankSize != 4) {
        return CCU_E_PARA;
    }
    RingKernelContext ctx;
    ctx.arg = kernelArg;
    RS2_CCU_CHK_RET(InitRingResources(ctx));
    RS2_CCU_CHK_RET(LoadRingArgs(ctx));
    RS2_CCU_CHK_RET(RingPreSync(ctx));
    for (uint32_t stage = 0; stage < kernelArg->rankSize - 1; ++stage) {
        RS2_CCU_CHK_RET(RingStage(ctx, kernelArg->sendBlockIndices[stage]));
    }
    RS2_CCU_CHK_RET(RingWriteOutput(ctx));
    RS2_CCU_CHK_RET(RingPostSync(ctx));
    return CCU_SUCCESS;
}

namespace {
constexpr uint32_t EIGHT_PLUS_FOUR_GROUP_SIZE = 4;
constexpr uint32_t EIGHT_PLUS_FOUR_GROUP_COUNT = 3;
constexpr uint32_t EIGHT_PLUS_FOUR_PAIR_PHASE = 0;
constexpr uint32_t EIGHT_PLUS_FOUR_REMOTE_PHASE = 1;

struct EightPlusFourPreReduceContext {
    const CcuKernelArgEightPlusFourPreReduce *arg = nullptr;
    ccu::Variable input;
    ccu::Variable inputToken;
    ccu::Variable scratch;
    ccu::Variable scratchToken;
    ccu::Variable recvBytes;
    ccu::Variable chunkOffset;
    ccu::Variable chunkBytes;
    std::vector<ccu::Variable> peerInputs;
    std::vector<ccu::Variable> peerTokens;
    ccu::Event event;
};

CcuResult InitEightPlusFourPreReduce(EightPlusFourPreReduceContext &ctx)
{
    ctx.peerInputs.resize(ctx.arg->channelCount);
    ctx.peerTokens.resize(ctx.arg->channelCount);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        ctx.peerInputs[channelIdx] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], INPUT_XN_ID);
        ctx.peerTokens[channelIdx] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], TOKEN_XN_ID);
    }
    uint32_t argId = 0;
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.inputToken, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.scratch, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.scratchToken, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.recvBytes, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.chunkOffset, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.chunkBytes, argId++));
    return CCU_SUCCESS;
}

CcuResult EightPlusFourPreReducePreSync(EightPlusFourPreReduceContext &ctx)
{
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx], ctx.input, INPUT_XN_ID,
            CKE_IDX, static_cast<uint16_t>(1U << INPUT_XN_ID)));
        RS2_CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx], ctx.inputToken, TOKEN_XN_ID,
            CKE_IDX, static_cast<uint16_t>(1U << TOKEN_XN_ID)));
    }
    constexpr uint16_t waitMask = static_cast<uint16_t>((1U << INPUT_XN_ID) | (1U << TOKEN_XN_ID));
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIdx], CKE_IDX, waitMask));
    }
    return CCU_SUCCESS;
}

CcuResult EightPlusFourPreReducePostSync(EightPlusFourPreReduceContext &ctx)
{
    constexpr uint16_t postMask = static_cast<uint16_t>(1U << POST_SYNC_ID);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[channelIdx], CKE_IDX, postMask));
    }
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIdx], CKE_IDX, postMask));
    }
    return CCU_SUCCESS;
}

void SetEightPlusFourBlockOffset(ccu::Variable &blockOffset, const ccu::Variable &chunkOffset,
    const ccu::Variable &recvBytes, uint32_t blockIdx)
{
    blockOffset = chunkOffset;
    for (uint32_t block = 0; block < blockIdx; ++block) {
        blockOffset += recvBytes;
    }
}

CcuResult PreReduceEightPlusFourBlock(EightPlusFourPreReduceContext &ctx, uint32_t blockIdx)
{
    ccu::Variable blockOffset;
    SetEightPlusFourBlockOffset(blockOffset, ctx.chunkOffset, ctx.recvBytes, blockIdx);

    std::vector<ccu::LocalAddr> pieces(EIGHT_PLUS_FOUR_GROUP_SIZE);
    ccu::Variable scratchOffset;
    scratchOffset = 0;
    for (uint32_t pieceIdx = 0; pieceIdx < EIGHT_PLUS_FOUR_GROUP_SIZE; ++pieceIdx) {
        pieces[pieceIdx].addr = ctx.scratch;
        pieces[pieceIdx].addr += scratchOffset;
        pieces[pieceIdx].token = ctx.scratchToken;
        scratchOffset += ctx.chunkBytes;
    }

    ccu::LocalAddr localInput;
    localInput.addr = ctx.input;
    localInput.addr += blockOffset;
    localInput.token = ctx.inputToken;
    RS2_CCU_CHK_RET(ccu::LocalCopy(pieces[EIGHT_PLUS_FOUR_GROUP_SIZE - 1], localInput,
        ctx.chunkBytes, ctx.event, 1));

    uint16_t readMask = 1;
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        ccu::RemoteAddr remoteInput;
        remoteInput.addr = ctx.peerInputs[channelIdx];
        remoteInput.addr += blockOffset;
        remoteInput.token = ctx.peerTokens[channelIdx];
        const uint16_t channelMask = static_cast<uint16_t>(1U << (channelIdx + 1));
        RS2_CCU_CHK_RET(ccu::Read(ctx.arg->channels[channelIdx], pieces[channelIdx], remoteInput,
            ctx.chunkBytes, ctx.event, channelMask));
        readMask = static_cast<uint16_t>(readMask | channelMask);
    }
    RS2_CCU_CHK_RET(ccu::EventWait(ctx.event, readMask));

    ccu::Variable firstLevelBytes;
    firstLevelBytes = ctx.chunkBytes;
    firstLevelBytes += ctx.chunkBytes;
    RS2_CCU_CHK_RET(ccu::LocalReduce(pieces[0], pieces[2], firstLevelBytes, HCCL_DATA_TYPE_FP32,
        HCCL_REDUCE_SUM, ctx.event));
    RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));
    RS2_CCU_CHK_RET(ccu::LocalReduce(pieces[0], pieces[1], ctx.chunkBytes, HCCL_DATA_TYPE_FP32,
        HCCL_REDUCE_SUM, ctx.event));
    RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));
    RS2_CCU_CHK_RET(ccu::LocalCopy(localInput, pieces[0], ctx.chunkBytes, ctx.event));
    RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));
    return CCU_SUCCESS;
}

struct EightPlusFourExchangeContext {
    const CcuKernelArgEightPlusFourExchange *arg = nullptr;
    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable recvBytes;
    ccu::Variable chunkOffset;
    ccu::Variable chunkBytes;
    std::vector<ccu::Variable> remoteInputs;
    std::vector<ccu::Variable> remoteInputTokens;
    ccu::Event event;
};

CcuResult InitEightPlusFourExchange(EightPlusFourExchangeContext &ctx)
{
    ctx.remoteInputs.resize(ctx.arg->channelCount);
    ctx.remoteInputTokens.resize(ctx.arg->channelCount);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        ctx.remoteInputs[channelIdx] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], INPUT_XN_ID);
        ctx.remoteInputTokens[channelIdx] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], TOKEN_XN_ID);
    }
    uint32_t argId = 0;
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.output, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.inputToken, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.outputToken, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.recvBytes, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.chunkOffset, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.chunkBytes, argId++));
    return CCU_SUCCESS;
}

CcuResult EightPlusFourExchangePreSync(EightPlusFourExchangeContext &ctx)
{
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx], ctx.input, INPUT_XN_ID,
            CKE_IDX, static_cast<uint16_t>(1U << INPUT_XN_ID)));
        RS2_CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx], ctx.inputToken, TOKEN_XN_ID,
            CKE_IDX, static_cast<uint16_t>(1U << TOKEN_XN_ID)));
    }
    constexpr uint16_t waitMask = static_cast<uint16_t>((1U << INPUT_XN_ID) | (1U << TOKEN_XN_ID));
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIdx], CKE_IDX, waitMask));
    }
    return CCU_SUCCESS;
}

CcuResult WriteReduceEightPlusFourBlock(EightPlusFourExchangeContext &ctx, uint32_t channelIdx,
    uint32_t blockIdx, uint16_t eventMask)
{
    ccu::Variable blockOffset;
    SetEightPlusFourBlockOffset(blockOffset, ctx.chunkOffset, ctx.recvBytes, blockIdx);
    ccu::LocalAddr source;
    source.addr = ctx.input;
    source.addr += blockOffset;
    source.token = ctx.inputToken;
    ccu::RemoteAddr destination;
    destination.addr = ctx.remoteInputs[channelIdx];
    destination.addr += blockOffset;
    destination.token = ctx.remoteInputTokens[channelIdx];
    return ccu::WriteReduce(ctx.arg->channels[channelIdx], destination, source, ctx.chunkBytes,
        HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.event, eventMask);
}

CcuResult EightPlusFourExchangePostSync(EightPlusFourExchangeContext &ctx)
{
    constexpr uint16_t postMask = static_cast<uint16_t>(1U << POST_SYNC_ID);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[channelIdx], CKE_IDX, postMask));
    }
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        RS2_CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIdx], CKE_IDX, postMask));
    }
    return CCU_SUCCESS;
}

CcuResult RunEightPlusFourPairExchange(EightPlusFourExchangeContext &ctx)
{
    if (ctx.arg->groupIndex == 0) {
        RS2_CCU_CHK_RET(WriteReduceEightPlusFourBlock(ctx, 0,
            ctx.arg->groupRank + EIGHT_PLUS_FOUR_GROUP_SIZE, 1));
        RS2_CCU_CHK_RET(ccu::EventWait(ctx.event, 1));
        RS2_CCU_CHK_RET(WriteReduceEightPlusFourBlock(ctx, 0,
            ctx.arg->groupRank + 2 * EIGHT_PLUS_FOUR_GROUP_SIZE, 1));
        RS2_CCU_CHK_RET(ccu::EventWait(ctx.event, 1));
    } else {
        RS2_CCU_CHK_RET(WriteReduceEightPlusFourBlock(ctx, 0, ctx.arg->groupRank, 1));
        RS2_CCU_CHK_RET(ccu::EventWait(ctx.event, 1));
    }
    return EightPlusFourExchangePostSync(ctx);
}

CcuResult CopyEightPlusFourOutput(EightPlusFourExchangeContext &ctx)
{
    const uint32_t blockIdx = ctx.arg->groupRank + ctx.arg->groupIndex * EIGHT_PLUS_FOUR_GROUP_SIZE;
    ccu::Variable blockOffset;
    SetEightPlusFourBlockOffset(blockOffset, ctx.chunkOffset, ctx.recvBytes, blockIdx);
    ccu::LocalAddr source;
    source.addr = ctx.input;
    source.addr += blockOffset;
    source.token = ctx.inputToken;
    ccu::LocalAddr destination;
    destination.addr = ctx.output;
    destination.addr += ctx.chunkOffset;
    destination.token = ctx.outputToken;
    RS2_CCU_CHK_RET(ccu::LocalCopy(destination, source, ctx.chunkBytes, ctx.event));
    RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));
    return CCU_SUCCESS;
}

CcuResult RunEightPlusFourRemoteExchange(EightPlusFourExchangeContext &ctx)
{
    if (ctx.arg->groupIndex == 1) {
        RS2_CCU_CHK_RET(WriteReduceEightPlusFourBlock(ctx, 0,
            ctx.arg->groupRank + 2 * EIGHT_PLUS_FOUR_GROUP_SIZE, 1));
        RS2_CCU_CHK_RET(ccu::EventWait(ctx.event, 1));
    } else if (ctx.arg->groupIndex == 2) {
        RS2_CCU_CHK_RET(WriteReduceEightPlusFourBlock(ctx, 0, ctx.arg->groupRank, 1));
        RS2_CCU_CHK_RET(WriteReduceEightPlusFourBlock(ctx, 1,
            ctx.arg->groupRank + EIGHT_PLUS_FOUR_GROUP_SIZE, 2));
        RS2_CCU_CHK_RET(ccu::EventWait(ctx.event, 3));
    }
    RS2_CCU_CHK_RET(EightPlusFourExchangePostSync(ctx));
    return CopyEightPlusFourOutput(ctx);
}
} // namespace

CcuResult CcuReduceScatterEightPlusFourPreReduceKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgEightPlusFourPreReduce *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount != EIGHT_PLUS_FOUR_GROUP_SIZE - 1 ||
        kernelArg->groupRank >= EIGHT_PLUS_FOUR_GROUP_SIZE) {
        return CCU_E_PARA;
    }
    EightPlusFourPreReduceContext ctx;
    ctx.arg = kernelArg;
    RS2_CCU_CHK_RET(InitEightPlusFourPreReduce(ctx));
    RS2_CCU_CHK_RET(EightPlusFourPreReducePreSync(ctx));
    for (uint32_t blockGroup = 0; blockGroup < EIGHT_PLUS_FOUR_GROUP_COUNT; ++blockGroup) {
        RS2_CCU_CHK_RET(PreReduceEightPlusFourBlock(ctx,
            kernelArg->groupRank + blockGroup * EIGHT_PLUS_FOUR_GROUP_SIZE));
    }
    RS2_CCU_CHK_RET(EightPlusFourPreReducePostSync(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterEightPlusFourExchangeKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgEightPlusFourExchange *>(arg);
    if (kernelArg == nullptr || kernelArg->groupIndex >= EIGHT_PLUS_FOUR_GROUP_COUNT ||
        kernelArg->groupRank >= EIGHT_PLUS_FOUR_GROUP_SIZE || kernelArg->channelCount == 0 ||
        kernelArg->channelCount > 2) {
        return CCU_E_PARA;
    }
    EightPlusFourExchangeContext ctx;
    ctx.arg = kernelArg;
    RS2_CCU_CHK_RET(InitEightPlusFourExchange(ctx));
    RS2_CCU_CHK_RET(EightPlusFourExchangePreSync(ctx));
    if (kernelArg->phase == EIGHT_PLUS_FOUR_PAIR_PHASE && kernelArg->groupIndex < 2 &&
        kernelArg->channelCount == 1) {
        return RunEightPlusFourPairExchange(ctx);
    }
    if (kernelArg->phase == EIGHT_PLUS_FOUR_REMOTE_PHASE &&
        ((kernelArg->groupIndex < 2 && kernelArg->channelCount == 1) ||
        (kernelArg->groupIndex == 2 && kernelArg->channelCount == 2))) {
        return RunEightPlusFourRemoteExchange(ctx);
    }
    return CCU_E_PARA;
}

namespace {
constexpr uint16_t EDGE_STAGE_ZERO_MASK = static_cast<uint16_t>(1U << 2);
constexpr uint16_t EDGE_STAGE_ONE_MASK = static_cast<uint16_t>(1U << 3);

struct EightPlusFourRingEdgeContext {
    const CcuKernelArgEightPlusFourRingEdge *arg = nullptr;
    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable recvBytes;
    ccu::Variable chunkOffset;
    ccu::Variable chunkBytes;
    ccu::Variable firstSliceBytes;
    ccu::Variable secondSliceBytes;
    ccu::Variable blockOffset;
    ccu::Variable remoteInput;
    ccu::Variable remoteInputToken;
    ccu::Event event;
};

CcuResult InitEightPlusFourRingEdge(EightPlusFourRingEdgeContext &ctx)
{
    ctx.remoteInput = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[0], INPUT_XN_ID);
    ctx.remoteInputToken = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[0], TOKEN_XN_ID);
    uint32_t argId = 0;
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.output, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.inputToken, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.outputToken, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.recvBytes, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.chunkOffset, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.chunkBytes, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.firstSliceBytes, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.secondSliceBytes, argId++));
    RS2_CCU_CHK_RET(ccu::LoadArg(ctx.blockOffset, argId++));
    return CCU_SUCCESS;
}

CcuResult PrepareEightPlusFourRingEdge(CcuKernelArg arg, EightPlusFourRingEdgeContext &ctx)
{
    auto *kernelArg = static_cast<CcuKernelArgEightPlusFourRingEdge *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount != 1 ||
        kernelArg->groupIndex >= EIGHT_PLUS_FOUR_GROUP_COUNT ||
        kernelArg->groupRank >= EIGHT_PLUS_FOUR_GROUP_SIZE) {
        return CCU_E_PARA;
    }
    ctx.arg = kernelArg;
    return InitEightPlusFourRingEdge(ctx);
}

CcuResult RunEightPlusFourRingEdgeInit(EightPlusFourRingEdgeContext &ctx)
{
    RS2_CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[0], ctx.input, INPUT_XN_ID,
        CKE_IDX, static_cast<uint16_t>(1U << INPUT_XN_ID)));
    RS2_CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[0], ctx.inputToken, TOKEN_XN_ID,
        CKE_IDX, static_cast<uint16_t>(1U << TOKEN_XN_ID)));
    constexpr uint16_t waitMask = static_cast<uint16_t>((1U << INPUT_XN_ID) | (1U << TOKEN_XN_ID));
    return ccu::NotifyWait(ctx.arg->channels[0], CKE_IDX, waitMask);
}

CcuResult SendEightPlusFourRingBlock(EightPlusFourRingEdgeContext &ctx, uint16_t doneMask)
{
    ccu::LocalAddr source;
    source.addr = ctx.input;
    source.addr += ctx.blockOffset;
    source.token = ctx.inputToken;
    ccu::RemoteAddr destination;
    destination.addr = ctx.remoteInput;
    destination.addr += ctx.blockOffset;
    destination.token = ctx.remoteInputToken;

    CCU_IF(ctx.firstSliceBytes != static_cast<uint64_t>(0)) {
        RS2_CCU_CHK_RET(ccu::WriteReduce(ctx.arg->channels[0], destination, source, ctx.firstSliceBytes,
            HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.event, 1));
    } CCU_ELSE {
        RS2_CCU_CHK_RET(ccu::EventRecord(ctx.event, 1));
    }
    source.addr += ctx.firstSliceBytes;
    destination.addr += ctx.firstSliceBytes;
    CCU_IF(ctx.secondSliceBytes != static_cast<uint64_t>(0)) {
        RS2_CCU_CHK_RET(ccu::WriteReduce(ctx.arg->channels[0], destination, source, ctx.secondSliceBytes,
            HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.event, 2));
    } CCU_ELSE {
        RS2_CCU_CHK_RET(ccu::EventRecord(ctx.event, 2));
    }
    RS2_CCU_CHK_RET(ccu::EventWait(ctx.event, 3));
    return ccu::NotifyRecord(ctx.arg->channels[0], CKE_IDX, doneMask);
}

CcuResult CopyEightPlusFourRingOutput(EightPlusFourRingEdgeContext &ctx)
{
    const uint32_t blockIdx = ctx.arg->groupRank + ctx.arg->groupIndex * EIGHT_PLUS_FOUR_GROUP_SIZE;
    ccu::Variable outputBlockOffset;
    SetEightPlusFourBlockOffset(outputBlockOffset, ctx.chunkOffset, ctx.recvBytes, blockIdx);
    ccu::LocalAddr source;
    source.addr = ctx.input;
    source.addr += outputBlockOffset;
    source.token = ctx.inputToken;
    ccu::LocalAddr destination;
    destination.addr = ctx.output;
    destination.addr += ctx.chunkOffset;
    destination.token = ctx.outputToken;
    RS2_CCU_CHK_RET(ccu::LocalCopy(destination, source, ctx.chunkBytes, ctx.event));
    RS2_CCU_CHK_RET(ccu::EventWait(ctx.event));
    return CCU_SUCCESS;
}
} // namespace

CcuResult CcuReduceScatterEightPlusFourEdgeInitKernel(CcuKernelArg arg)
{
    EightPlusFourRingEdgeContext ctx;
    RS2_CCU_CHK_RET(PrepareEightPlusFourRingEdge(arg, ctx));
    return RunEightPlusFourRingEdgeInit(ctx);
}

CcuResult CcuReduceScatterEightPlusFourEdgeSendStageZeroKernel(CcuKernelArg arg)
{
    EightPlusFourRingEdgeContext ctx;
    RS2_CCU_CHK_RET(PrepareEightPlusFourRingEdge(arg, ctx));
    return SendEightPlusFourRingBlock(ctx, EDGE_STAGE_ZERO_MASK);
}

CcuResult CcuReduceScatterEightPlusFourEdgeWaitStageZeroKernel(CcuKernelArg arg)
{
    EightPlusFourRingEdgeContext ctx;
    RS2_CCU_CHK_RET(PrepareEightPlusFourRingEdge(arg, ctx));
    return ccu::NotifyWait(ctx.arg->channels[0], CKE_IDX, EDGE_STAGE_ZERO_MASK);
}

CcuResult CcuReduceScatterEightPlusFourEdgeSendStageOneKernel(CcuKernelArg arg)
{
    EightPlusFourRingEdgeContext ctx;
    RS2_CCU_CHK_RET(PrepareEightPlusFourRingEdge(arg, ctx));
    return SendEightPlusFourRingBlock(ctx, EDGE_STAGE_ONE_MASK);
}

CcuResult CcuReduceScatterEightPlusFourEdgeWaitStageOneKernel(CcuKernelArg arg)
{
    EightPlusFourRingEdgeContext ctx;
    RS2_CCU_CHK_RET(PrepareEightPlusFourRingEdge(arg, ctx));
    return ccu::NotifyWait(ctx.arg->channels[0], CKE_IDX, EDGE_STAGE_ONE_MASK);
}

CcuResult CcuReduceScatterEightPlusFourEdgeCopyOutputKernel(CcuKernelArg arg)
{
    EightPlusFourRingEdgeContext ctx;
    RS2_CCU_CHK_RET(PrepareEightPlusFourRingEdge(arg, ctx));
    return CopyEightPlusFourRingOutput(ctx);
}

} // namespace ops_hccl
} // namespace final_adaptive41
#undef RS2_CCU_CHK_RET
