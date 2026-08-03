/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ===========================================================================
// ReduceScatter CCU Kernel
//   Mesh: Read(peer_INPUT → CCL) + LocalReduce (CCL → output)
//   RH:   WriteReduce(my_INPUT → peer_INPUT), 数据原地累积, 无需 CCL
// ===========================================================================

#include "ccu_primitives.hpp"
#include "ccu_types.h"

#include "ccu_kernel.h"

#define CCU_CHK_RET(call) \
    do { \
        CcuResult __ret = (call); \
        if (__ret != CCU_SUCCESS) { \
            return __ret; \
        } \
    } while (0)

namespace ccu = ::AscendC::ccu;

namespace ops_hccl {

constexpr uint16_t RS_SELF_OUTPUT_EVENT_MASK = 1U << 15;
constexpr uint16_t RS_PIPELINE_MERGE_EVENT_MASK = 1U << 14;

// ===========================================================================
// Mesh Kernel: Read + LocalReduce (CCL-based, checker verified)
// ===========================================================================
struct ReduceScatterCtx {
    const ReduceScatterKernelArg *arg;

    ccu::Variable peerInputVa[RS_MAX_RANK_SIZE];
    ccu::Variable peerToken[RS_MAX_RANK_SIZE];

    ccu::Variable myInputVa;
    ccu::Variable myOutputVa;
    ccu::Variable myInputToken;
    ccu::Variable myOutputToken;
    ccu::Variable myCclVa;
    ccu::Variable myCclToken;
    ccu::Variable sliceBytes;
    ccu::Variable sliceByteOff;
    ccu::Variable blockBytes;
    ccu::Variable currentPartialVa;
    ccu::Variable previousPartialVa;
    ccu::Variable previousMergeBytes;
    ccu::Variable previousOutputOff;
    ccu::Variable doPreSync;

    ccu::LocalAddr myInputAddr;
    ccu::LocalAddr myOutputAddr;
    ccu::LocalAddr scratchAddr[RS_MAX_RANK_SIZE];
    ccu::RemoteAddr remoteInputAddr[RS_MAX_RANK_SIZE];

    ccu::Event event;
};

static CcuResult InitResource(ReduceScatterCtx &ctx)
{
    const auto *arg = ctx.arg;
    uint32_t channelIdx = 0;
    for (uint32_t peerId = 0; peerId < arg->rankSize; peerId++) {
        if (peerId == arg->rankId)
            continue;
        if (channelIdx < arg->channelCount) {
            ctx.peerInputVa[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], RS_INPUT_XN_ID);
            ctx.peerToken[peerId] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], RS_TOKEN_XN_ID);
            channelIdx++;
        }
    }
    return CCU_SUCCESS;
}

static CcuResult LoadArgsBase(ReduceScatterCtx &ctx)
{
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.myInputVa, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.myOutputVa, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.myInputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.myOutputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.myCclVa, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.myCclToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceByteOff, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.blockBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentPartialVa, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.previousPartialVa, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.previousMergeBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.previousOutputOff, argId++));
    return CCU_SUCCESS;
}

static CcuResult LoadArgsPipeline(ReduceScatterCtx &ctx)
{
    CCU_CHK_RET(LoadArgsBase(ctx));
    uint32_t argId = 13;
    CCU_CHK_RET(ccu::LoadArg(ctx.doPreSync, argId++));
    return CCU_SUCCESS;
}

static void StartPipelineMerge(ReduceScatterCtx &ctx)
{
    CCU_IF(ctx.previousMergeBytes != 0)
    {
        CCU_IF(ctx.currentPartialVa != 0)
        {
            ccu::LocalAddr dst;
            dst.addr = ctx.myOutputVa;
            dst.addr = dst.addr + ctx.previousOutputOff;
            dst.token = ctx.myOutputToken;

            ccu::LocalAddr src;
            src.addr = ctx.previousPartialVa;
            src.token = ctx.myCclToken;

            ccu::LocalReduce(dst, src, ctx.previousMergeBytes, ctx.arg->dataType, ctx.arg->reduceOp, ctx.event,
                RS_PIPELINE_MERGE_EVENT_MASK);
        }
    }
}

static void FinishPipelineMerge(ReduceScatterCtx &ctx)
{
    CCU_IF(ctx.previousMergeBytes != 0)
    {
        CCU_IF(ctx.currentPartialVa != 0)
        {
            ccu::EventWait(ctx.event, RS_PIPELINE_MERGE_EVENT_MASK);
        }
    }
}

static void PreSync(ReduceScatterCtx &ctx)
{
    const auto *arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::WriteVariableWithNotify(
            arg->channels[i], ctx.myInputVa, RS_INPUT_XN_ID, RS_CKE_IDX_0, 1 << RS_INPUT_XN_ID);
        ccu::WriteVariableWithNotify(
            arg->channels[i], ctx.myInputToken, RS_TOKEN_XN_ID, RS_CKE_IDX_0, 1 << RS_TOKEN_XN_ID);
    }
    uint32_t allBit = (1 << RS_INPUT_XN_ID) | (1 << RS_TOKEN_XN_ID);
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], RS_CKE_IDX_0, allBit);
    }
}

static void PostSyncChannels(const ReduceScatterKernelArg *arg)
{
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyRecord(arg->channels[i], RS_CKE_IDX_0, 1 << RS_POST_SYNC_ID);
    }
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], RS_CKE_IDX_0, 1 << RS_POST_SYNC_ID);
    }
}

static void PostSync(ReduceScatterCtx &ctx)
{
    PostSyncChannels(ctx.arg);
}

static void BuildAddresses(ReduceScatterCtx &ctx)
{
    const auto *arg = ctx.arg;

    ccu::Variable ownBlockOff;
    ownBlockOff = 0;
    for (uint32_t i = 0; i < arg->rankId; i++) {
        ownBlockOff = ownBlockOff + ctx.blockBytes;
    }

    ctx.myInputAddr.addr = ctx.myInputVa;
    ctx.myInputAddr.addr = ctx.myInputAddr.addr + ownBlockOff;
    ctx.myInputAddr.addr = ctx.myInputAddr.addr + ctx.sliceByteOff;
    ctx.myInputAddr.token = ctx.myInputToken;

    ctx.myOutputAddr.addr = ctx.myOutputVa;
    ctx.myOutputAddr.addr = ctx.myOutputAddr.addr + ctx.sliceByteOff;
    ctx.myOutputAddr.token = ctx.myOutputToken;

    ccu::Variable scratchOff;
    scratchOff = 0;
    uint32_t channelIdx = 0;
    for (uint32_t r = 0; r < arg->rankSize; r++) {
        ctx.scratchAddr[r].addr = ctx.myCclVa;
        ctx.scratchAddr[r].addr = ctx.scratchAddr[r].addr + scratchOff;
        ctx.scratchAddr[r].token = ctx.myCclToken;
        scratchOff = scratchOff + ctx.sliceBytes;

        if (r == arg->rankId)
            continue;
        if (channelIdx < arg->channelCount) {
            ctx.remoteInputAddr[r].addr = ctx.peerInputVa[r];
            ctx.remoteInputAddr[r].addr = ctx.remoteInputAddr[r].addr + ownBlockOff;
            ctx.remoteInputAddr[r].addr = ctx.remoteInputAddr[r].addr + ctx.sliceByteOff;
            ctx.remoteInputAddr[r].token = ctx.peerToken[r];
            channelIdx++;
        }
    }
}

static void TreeReduceScratch(ReduceScatterCtx &ctx, uint32_t pieceCount)
{
    const auto *arg = ctx.arg;
    uint32_t remain = pieceCount;
    while (remain > 1) {
        uint32_t reducePieces = remain / 2;
        uint32_t srcIdx = remain - reducePieces;
        ccu::Variable reduceLen;
        reduceLen = ctx.sliceBytes;
        for (uint32_t i = 1; i < reducePieces; i++) {
            reduceLen = reduceLen + ctx.sliceBytes;
        }
        ccu::LocalReduce(
            ctx.scratchAddr[0], ctx.scratchAddr[srcIdx], reduceLen, arg->dataType, arg->reduceOp, ctx.event, 1);
        ccu::EventWait(ctx.event, 1);
        remain -= reducePieces;
    }
}

static void ReduceRemotePiecesToPartial(ReduceScatterCtx &ctx)
{
    const auto *arg = ctx.arg;
    ccu::LocalAddr partialSum;
    partialSum.addr = ctx.currentPartialVa;
    partialSum.token = ctx.myCclToken;

    uint32_t channelIdx = 0;
    uint32_t pieceCount = 0;
    uint32_t scratchPieceCount = 0;
    uint16_t validMask = 0;
    for (uint32_t r = 0; r < arg->rankSize; r++) {
        if (r == arg->rankId)
            continue;
        if (channelIdx >= arg->channelCount)
            break;

        const uint16_t mask = static_cast<uint16_t>(1U << pieceCount);
        if (pieceCount == 0) {
            ccu::Read(arg->channels[channelIdx], partialSum, ctx.remoteInputAddr[r], ctx.sliceBytes, ctx.event, mask);
        } else {
            ccu::Read(arg->channels[channelIdx], ctx.scratchAddr[scratchPieceCount], ctx.remoteInputAddr[r],
                ctx.sliceBytes, ctx.event, mask);
            scratchPieceCount++;
        }
        validMask |= mask;
        pieceCount++;
        channelIdx++;
    }

    ccu::EventWait(ctx.event, validMask);
    TreeReduceScratch(ctx, scratchPieceCount);
    if (scratchPieceCount > 0) {
        ccu::LocalReduce(partialSum, ctx.scratchAddr[0], ctx.sliceBytes, arg->dataType, arg->reduceOp, ctx.event, 1);
        ccu::EventWait(ctx.event, 1);
    }
}

static CcuResult DataPhase(ReduceScatterCtx &ctx)
{
    const auto *arg = ctx.arg;

    CCU_IF(ctx.sliceBytes != 0)
    {
        if (arg->channelCount == arg->rankSize - 1) {
            uint32_t channelIdx = 0;
            uint16_t validMask = 0;

            bool includeSelf = arg->initOutput;
            if (includeSelf) {
                validMask = (1 << arg->rankId);
                ccu::LocalCopy(
                    ctx.scratchAddr[arg->rankId], ctx.myInputAddr, ctx.sliceBytes, ctx.event, 1 << arg->rankId);
            }
            for (uint32_t r = 0; r < arg->rankSize; r++) {
                const uint16_t mask = 1 << r;
                if (r == arg->rankId)
                    continue;
                if (channelIdx < arg->channelCount) {
                    ccu::Read(arg->channels[channelIdx], ctx.scratchAddr[r], ctx.remoteInputAddr[r], ctx.sliceBytes,
                        ctx.event, mask);
                    validMask |= mask;
                    channelIdx++;
                }
            }
            ccu::EventWait(ctx.event, validMask);

            TreeReduceScratch(ctx, arg->rankSize);
            if (!arg->skipOutput) {
                if (arg->reduceToOutput) {
                    ccu::LocalReduce(ctx.myOutputAddr, ctx.scratchAddr[0], ctx.sliceBytes, arg->dataType, arg->reduceOp,
                        ctx.event, 1);
                } else {
                    ccu::LocalCopy(ctx.myOutputAddr, ctx.scratchAddr[0], ctx.sliceBytes, ctx.event, 1);
                }
                ccu::EventWait(ctx.event, 1);
            }
        } else if (arg->skipOutput) {
            CCU_IF(ctx.currentPartialVa != 0)
            {
                ReduceRemotePiecesToPartial(ctx);
            }
            CCU_IF(ctx.currentPartialVa == 0)
            {
                uint32_t channelIdx = 0;
                uint32_t pieceCount = 0;
                uint16_t validMask = 0;
                for (uint32_t r = 0; r < arg->rankSize; r++) {
                    if (r == arg->rankId)
                        continue;
                    if (channelIdx >= arg->channelCount)
                        break;
                    const uint16_t mask = static_cast<uint16_t>(1U << pieceCount);
                    ccu::Read(arg->channels[channelIdx], ctx.scratchAddr[pieceCount], ctx.remoteInputAddr[r],
                        ctx.sliceBytes, ctx.event, mask);
                    validMask |= mask;
                    pieceCount++;
                    channelIdx++;
                }
                ccu::EventWait(ctx.event, validMask);
                TreeReduceScratch(ctx, pieceCount);
            }
        } else {
            uint32_t channelIdx = 0;
            uint32_t pieceCount = 0;
            uint16_t validMask = 0;
            uint16_t selfOutputMask = 0;
            const bool initOutputDirect = arg->initOutput && !arg->reduceToOutput;

            if (initOutputDirect) {
                selfOutputMask = RS_SELF_OUTPUT_EVENT_MASK;
                ccu::LocalCopy(ctx.myOutputAddr, ctx.myInputAddr, ctx.sliceBytes, ctx.event, selfOutputMask);
            } else if (arg->initOutput) {
                const uint16_t mask = 1 << pieceCount;
                ccu::LocalCopy(ctx.scratchAddr[pieceCount], ctx.myInputAddr, ctx.sliceBytes, ctx.event, mask);
                validMask |= mask;
                pieceCount++;
            }
            for (uint32_t r = 0; r < arg->rankSize; r++) {
                if (r == arg->rankId)
                    continue;
                if (channelIdx >= arg->channelCount)
                    break;
                const uint16_t mask = static_cast<uint16_t>(1U << pieceCount);
                ccu::Read(arg->channels[channelIdx], ctx.scratchAddr[pieceCount], ctx.remoteInputAddr[r],
                    ctx.sliceBytes, ctx.event, mask);
                validMask |= mask;
                pieceCount++;
                channelIdx++;
            }
            ccu::EventWait(ctx.event, validMask);

            TreeReduceScratch(ctx, pieceCount);
            if (initOutputDirect) {
                ccu::EventWait(ctx.event, selfOutputMask);
                if (pieceCount > 0) {
                    ccu::LocalReduce(ctx.myOutputAddr, ctx.scratchAddr[0], ctx.sliceBytes, arg->dataType, arg->reduceOp,
                        ctx.event, 1);
                    ccu::EventWait(ctx.event, 1);
                }
            } else if (arg->reduceToOutput) {
                ccu::LocalReduce(
                    ctx.myOutputAddr, ctx.scratchAddr[0], ctx.sliceBytes, arg->dataType, arg->reduceOp, ctx.event, 1);
                ccu::EventWait(ctx.event, 1);
            } else {
                ccu::LocalCopy(ctx.myOutputAddr, ctx.scratchAddr[0], ctx.sliceBytes, ctx.event, 1);
                ccu::EventWait(ctx.event, 1);
            }
        }
    }

    return CCU_SUCCESS;
}

CcuResult CcuKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<ReduceScatterKernelArg *>(arg);

    if (kernelArg->channelCount == 0)
        return CCU_E_PARA;

    ReduceScatterCtx ctx;
    ctx.arg = kernelArg;

    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgsBase(ctx));
    StartPipelineMerge(ctx);
    PreSync(ctx);
    BuildAddresses(ctx);
    CCU_CHK_RET(DataPhase(ctx));
    FinishPipelineMerge(ctx);
    PostSync(ctx);

    return CCU_SUCCESS;
}

CcuResult CcuKernelPipeline(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<ReduceScatterKernelArg *>(arg);

    if (kernelArg->channelCount == 0)
        return CCU_E_PARA;

    ReduceScatterCtx ctx;
    ctx.arg = kernelArg;

    CCU_CHK_RET(InitResource(ctx));
    CCU_CHK_RET(LoadArgsPipeline(ctx));
    StartPipelineMerge(ctx);
    CCU_IF(ctx.doPreSync != 0)
    {
        PreSync(ctx);
    }
    BuildAddresses(ctx);
    CCU_CHK_RET(DataPhase(ctx));
    FinishPipelineMerge(ctx);

    return CCU_SUCCESS;
}

struct Small4x1Ctx {
    const ReduceScatterKernelArg *arg;

    ccu::Variable peerInputVa[3];
    ccu::Variable peerToken[3];
    ccu::Variable inputVa;
    ccu::Variable outputVa;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable cclVa;
    ccu::Variable cclToken;
    ccu::Variable sliceBytes;
    ccu::Variable sliceByteOff;
    ccu::Variable blockBytes;

    ccu::LocalAddr selfAddr;
    ccu::LocalAddr outputAddr;
    ccu::LocalAddr scratch[3];
    ccu::RemoteAddr remoteAddr[3];
    ccu::Event event;
};

static CcuResult Small4x1Init(Small4x1Ctx &ctx)
{
    const auto *arg = ctx.arg;
    if (arg->rankSize != 4 || arg->channelCount != 3) {
        return CCU_E_PARA;
    }
    for (uint32_t i = 0; i < 3; i++) {
        ctx.peerInputVa[i] = ccu::GetResByChannel<ccu::Variable>(arg->channels[i], RS_INPUT_XN_ID);
        ctx.peerToken[i] = ccu::GetResByChannel<ccu::Variable>(arg->channels[i], RS_TOKEN_XN_ID);
    }
    return CCU_SUCCESS;
}

static CcuResult Small4x1LoadArgs(Small4x1Ctx &ctx)
{
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.inputVa, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputVa, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.cclVa, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.cclToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceByteOff, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.blockBytes, argId++));
    return CCU_SUCCESS;
}

static void Small4x1PreSync(Small4x1Ctx &ctx)
{
    const auto *arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::WriteVariableWithNotify(
            arg->channels[i], ctx.inputVa, RS_INPUT_XN_ID, RS_CKE_IDX_0, 1 << RS_INPUT_XN_ID);
        ccu::WriteVariableWithNotify(
            arg->channels[i], ctx.inputToken, RS_TOKEN_XN_ID, RS_CKE_IDX_0, 1 << RS_TOKEN_XN_ID);
    }
    const uint32_t allBit = (1 << RS_INPUT_XN_ID) | (1 << RS_TOKEN_XN_ID);
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], RS_CKE_IDX_0, allBit);
    }
}

static void Small4x1BuildAddresses(Small4x1Ctx &ctx)
{
    const auto *arg = ctx.arg;

    ccu::Variable ownBlockOff;
    ownBlockOff = 0;
    for (uint32_t i = 0; i < arg->rankId; i++) {
        ownBlockOff = ownBlockOff + ctx.blockBytes;
    }

    ctx.selfAddr.addr = ctx.inputVa;
    ctx.selfAddr.addr = ctx.selfAddr.addr + ownBlockOff;
    ctx.selfAddr.addr = ctx.selfAddr.addr + ctx.sliceByteOff;
    ctx.selfAddr.token = ctx.inputToken;

    ctx.outputAddr.addr = ctx.outputVa;
    ctx.outputAddr.addr = ctx.outputAddr.addr + ctx.sliceByteOff;
    ctx.outputAddr.token = ctx.outputToken;

    ccu::Variable scratchOff;
    scratchOff = 0;
    for (uint32_t i = 0; i < 3; i++) {
        ctx.scratch[i].addr = ctx.cclVa;
        ctx.scratch[i].addr = ctx.scratch[i].addr + scratchOff;
        ctx.scratch[i].token = ctx.cclToken;
        scratchOff = scratchOff + ctx.sliceBytes;

        ctx.remoteAddr[i].addr = ctx.peerInputVa[i];
        ctx.remoteAddr[i].addr = ctx.remoteAddr[i].addr + ownBlockOff;
        ctx.remoteAddr[i].addr = ctx.remoteAddr[i].addr + ctx.sliceByteOff;
        ctx.remoteAddr[i].token = ctx.peerToken[i];
    }
}

static void Small4x1DataPhase(Small4x1Ctx &ctx)
{
    const auto *arg = ctx.arg;
    constexpr uint16_t selfMask = RS_SELF_OUTPUT_EVENT_MASK;
    uint16_t peerMask = 0;

    ccu::LocalCopy(ctx.outputAddr, ctx.selfAddr, ctx.sliceBytes, ctx.event, selfMask);
    for (uint32_t i = 0; i < 3; i++) {
        const uint16_t mask = static_cast<uint16_t>(1U << i);
        ccu::Read(arg->channels[i], ctx.scratch[i], ctx.remoteAddr[i], ctx.sliceBytes, ctx.event, mask);
        peerMask |= mask;
    }
    ccu::EventWait(ctx.event, peerMask);

    ccu::LocalReduce(ctx.scratch[0], ctx.scratch[2], ctx.sliceBytes, arg->dataType, arg->reduceOp, ctx.event, 1);
    ccu::EventWait(ctx.event, 1);
    ccu::LocalReduce(ctx.scratch[0], ctx.scratch[1], ctx.sliceBytes, arg->dataType, arg->reduceOp, ctx.event, 1);
    ccu::EventWait(ctx.event, 1);

    ccu::EventWait(ctx.event, selfMask);
    ccu::LocalReduce(ctx.outputAddr, ctx.scratch[0], ctx.sliceBytes, arg->dataType, arg->reduceOp, ctx.event, 1);
    ccu::EventWait(ctx.event, 1);
}

CcuResult CcuKernelSmall4x1(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<ReduceScatterKernelArg *>(arg);

    Small4x1Ctx ctx;
    ctx.arg = kernelArg;
    CCU_CHK_RET(Small4x1Init(ctx));
    CCU_CHK_RET(Small4x1LoadArgs(ctx));
    Small4x1PreSync(ctx);
    Small4x1BuildAddresses(ctx);
    CCU_IF(ctx.sliceBytes != 0)
    {
        Small4x1DataPhase(ctx);
    }
    PostSyncChannels(kernelArg);
    return CCU_SUCCESS;
}

struct MergeScratchCtx {
    const ReduceScatterKernelArg *arg;

    ccu::Variable myCclVa;
    ccu::Variable myOutputVa;
    ccu::Variable myCclToken;
    ccu::Variable myOutputToken;
    ccu::Variable sliceBytes;
    ccu::Variable secondScratchOff;
    ccu::Variable sliceByteOff;
    ccu::Variable doPostSync;

    ccu::Event event;
};

static CcuResult MergeLoadArgs(MergeScratchCtx &ctx)
{
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.myCclVa, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.myOutputVa, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.myCclToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.myOutputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.secondScratchOff, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceByteOff, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.doPostSync, argId++));
    return CCU_SUCCESS;
}

CcuResult CcuKernelMergeScratch(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<ReduceScatterKernelArg *>(arg);

    if (kernelArg->channelCount > 0) {
        auto dieAnchor = ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[0], RS_INPUT_XN_ID);
        (void)dieAnchor;
    }

    MergeScratchCtx ctx;
    ctx.arg = kernelArg;

    CCU_CHK_RET(MergeLoadArgs(ctx));

    CCU_IF(ctx.sliceBytes != 0)
    {
        ccu::LocalAddr outDst;
        outDst.addr = ctx.myOutputVa;
        outDst.addr = outDst.addr + ctx.sliceByteOff;
        outDst.token = ctx.myOutputToken;

        ccu::LocalAddr firstSum;
        firstSum.addr = ctx.myCclVa;
        firstSum.token = ctx.myCclToken;

        CCU_IF(ctx.secondScratchOff == 0)
        {
            ccu::LocalReduce(outDst, firstSum, ctx.sliceBytes, kernelArg->dataType, kernelArg->reduceOp, ctx.event, 1);
        }

        CCU_IF(ctx.secondScratchOff != 0)
        {
            ccu::LocalAddr secondSum;
            secondSum.addr = ctx.myCclVa;
            secondSum.addr = secondSum.addr + ctx.secondScratchOff;
            secondSum.token = ctx.myCclToken;

            ccu::LocalReduce(outDst, secondSum, ctx.sliceBytes, kernelArg->dataType, kernelArg->reduceOp, ctx.event, 1);
        }

        ccu::EventWait(ctx.event, 1);
        CCU_IF(ctx.doPostSync != 0)
        {
            PostSyncChannels(kernelArg);
        }
    }

    CCU_IF(ctx.sliceBytes == 0)
    {
        CCU_IF(ctx.doPostSync != 0)
        {
            PostSyncChannels(kernelArg);
        }
    }

    return CCU_SUCCESS;
}

constexpr uint32_t RS_GDRR_MAX_GROUPS = 4;
constexpr uint32_t RS_GDRR_MAX_GROUP_SIZE = 4;

struct GroupDrrCtx {
    const ReduceScatterKernelArg *arg;
    ccu::Variable peerInputVa[RS_MAX_RANK_SIZE];
    ccu::Variable peerToken[RS_MAX_RANK_SIZE];
    ccu::Variable peerBlockVa[RS_MAX_RANK_SIZE];
    ccu::Variable inputVa;
    ccu::Variable inputToken;
    ccu::Variable outputVa;
    ccu::Variable outputToken;
    ccu::Variable cclVa;
    ccu::Variable cclToken;
    ccu::Variable blockBytes;
    ccu::Variable targetOff;
    ccu::Variable scratchOff;
    ccu::Variable chunkBytes[RS_GDRR_MAX_GROUPS];
    ccu::Variable lastChunkBytes[RS_GDRR_MAX_GROUPS];
    ccu::Variable chunkOff[RS_GDRR_MAX_GROUPS][RS_GDRR_MAX_GROUP_SIZE];
    ccu::LocalAddr groupBase[RS_GDRR_MAX_GROUPS];
    ccu::LocalAddr dst[RS_GDRR_MAX_GROUPS][RS_GDRR_MAX_GROUP_SIZE];
    ccu::RemoteAddr src[RS_MAX_RANK_SIZE];
    ccu::LocalAddr selfSrc;
    ccu::LocalAddr primaryBase;
    ccu::Event event;
};

static uint32_t GetGroupDrrMaxGroupSize(const GroupDrrCtx &ctx)
{
    uint32_t maxGroupSize = ctx.arg->stepOffset;
    if (maxGroupSize == 0 || maxGroupSize > RS_GDRR_MAX_GROUP_SIZE) {
        maxGroupSize = RS_GDRR_MAX_GROUP_SIZE;
    }
    return maxGroupSize;
}

static uint32_t GetGroupDrrGroupCount(uint32_t channelCount, uint32_t maxGroupSize)
{
    return (channelCount + maxGroupSize - 1U) / maxGroupSize;
}

static uint32_t GetGroupDrrGroupStart(uint32_t groupIdx, uint32_t maxGroupSize)
{
    return groupIdx * maxGroupSize;
}

static uint32_t GetGroupDrrGroupSize(uint32_t channelCount, uint32_t groupIdx, uint32_t maxGroupSize)
{
    const uint32_t start = GetGroupDrrGroupStart(groupIdx, maxGroupSize);
    const uint32_t remain = channelCount - start;
    return (remain < maxGroupSize) ? remain : maxGroupSize;
}

static CcuResult GroupDrrInit(GroupDrrCtx &ctx)
{
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        ctx.peerInputVa[i] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[i], RS_INPUT_XN_ID);
        ctx.peerToken[i] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[i], RS_TOKEN_XN_ID);
    }
    return CCU_SUCCESS;
}

static CcuResult GroupDrrLoadArgs(GroupDrrCtx &ctx)
{
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.inputVa, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.inputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputVa, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.cclVa, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.cclToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.blockBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.targetOff, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.scratchOff, argId++));
    for (uint32_t g = 0; g < RS_GDRR_MAX_GROUPS; g++) {
        CCU_CHK_RET(ccu::LoadArg(ctx.chunkBytes[g], argId++));
    }
    for (uint32_t g = 0; g < RS_GDRR_MAX_GROUPS; g++) {
        CCU_CHK_RET(ccu::LoadArg(ctx.lastChunkBytes[g], argId++));
    }
    return CCU_SUCCESS;
}

static void GroupDrrPreSync(GroupDrrCtx &ctx)
{
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        ccu::WriteVariableWithNotify(
            ctx.arg->channels[i], ctx.inputVa, RS_INPUT_XN_ID, RS_CKE_IDX_0, 1 << RS_INPUT_XN_ID);
        ccu::WriteVariableWithNotify(
            ctx.arg->channels[i], ctx.inputToken, RS_TOKEN_XN_ID, RS_CKE_IDX_0, 1 << RS_TOKEN_XN_ID);
    }
    constexpr uint32_t allBit = (1 << RS_INPUT_XN_ID) | (1 << RS_TOKEN_XN_ID);
    for (uint32_t i = 0; i < ctx.arg->channelCount; i++) {
        ccu::NotifyWait(ctx.arg->channels[i], RS_CKE_IDX_0, allBit);
    }
}

static void GroupDrrPostSync(GroupDrrCtx &ctx)
{
    PostSyncChannels(ctx.arg);
}

static void GroupDrrBuildAddresses(GroupDrrCtx &ctx)
{
    ccu::Variable ownBlockOff;
    ownBlockOff = 0;
    for (uint32_t i = 0; i < ctx.arg->rankId; i++) {
        ownBlockOff = ownBlockOff + ctx.blockBytes;
    }

    if (ctx.arg->reduceToOutput) {
        ctx.primaryBase.addr = ctx.outputVa;
        ctx.primaryBase.token = ctx.outputToken;
    } else {
        ctx.primaryBase.addr = ctx.cclVa;
        ctx.primaryBase.addr = ctx.primaryBase.addr + ctx.targetOff;
        ctx.primaryBase.token = ctx.cclToken;
    }

    const uint32_t maxGroupSize = GetGroupDrrMaxGroupSize(ctx);
    const uint32_t groupCount = GetGroupDrrGroupCount(ctx.arg->channelCount, maxGroupSize);
    for (uint32_t g = 0; g < groupCount; g++) {
        if (g == 0) {
            ctx.groupBase[g] = ctx.primaryBase;
        } else {
            ctx.groupBase[g].addr = ctx.cclVa;
            ctx.groupBase[g].addr = ctx.groupBase[g].addr + ctx.scratchOff;
            for (uint32_t prev = 1; prev < g; prev++) {
                ctx.groupBase[g].addr = ctx.groupBase[g].addr + ctx.blockBytes;
            }
            ctx.groupBase[g].token = ctx.cclToken;
        }

        const uint32_t groupSize = GetGroupDrrGroupSize(ctx.arg->channelCount, g, maxGroupSize);
        for (uint32_t c = 0; c < groupSize; c++) {
            if (c == 0) {
                ctx.chunkOff[g][c] = 0;
            } else {
                ctx.chunkOff[g][c] = ctx.chunkOff[g][c - 1] + ctx.chunkBytes[g];
            }
            ctx.dst[g][c].addr = ctx.groupBase[g].addr;
            ctx.dst[g][c].addr = ctx.dst[g][c].addr + ctx.chunkOff[g][c];
            ctx.dst[g][c].token = ctx.groupBase[g].token;
        }
    }

    const uint32_t n = ctx.arg->channelCount;
    for (uint32_t i = 0; i < n; i++) {
        ctx.peerBlockVa[i] = ctx.peerInputVa[i] + ownBlockOff;
        ctx.src[i].token = ctx.peerToken[i];
    }

    ctx.selfSrc.addr = ctx.inputVa;
    ctx.selfSrc.addr = ctx.selfSrc.addr + ownBlockOff;
    ctx.selfSrc.token = ctx.inputToken;
}

static uint16_t GroupDrrIssueGroup(GroupDrrCtx &ctx, uint32_t groupIdx, uint32_t groupSize, uint32_t round,
    uint32_t channelStart)
{
    if (round >= groupSize) {
        return 0;
    }

    uint16_t validMask = 0;
    for (uint32_t i = 0; i < groupSize; i++) {
        const uint32_t channelIdx = channelStart + i;
        const uint32_t chunk = (i + round) % groupSize;
        ctx.src[channelIdx].addr = ctx.peerBlockVa[channelIdx];
        ctx.src[channelIdx].addr = ctx.src[channelIdx].addr + ctx.chunkOff[groupIdx][chunk];
        const uint16_t mask = static_cast<uint16_t>(1U << (channelStart + chunk));
        ccu::Variable &len
            = (chunk + 1 == groupSize) ? ctx.lastChunkBytes[groupIdx] : ctx.chunkBytes[groupIdx];
        if (round == 0) {
            ccu::Read(ctx.arg->channels[channelIdx], ctx.dst[groupIdx][chunk], ctx.src[channelIdx], len, ctx.event,
                mask);
        } else {
            ccu::ReadReduce(ctx.arg->channels[channelIdx], ctx.dst[groupIdx][chunk], ctx.src[channelIdx], len,
                ctx.arg->dataType, ctx.arg->reduceOp, ctx.event, mask);
        }
        validMask |= mask;
    }
    return validMask;
}

static CcuResult GroupDrrData(GroupDrrCtx &ctx)
{
    const uint32_t maxGroupSize = GetGroupDrrMaxGroupSize(ctx);
    const uint32_t groupCount = GetGroupDrrGroupCount(ctx.arg->channelCount, maxGroupSize);
    const uint32_t rounds
        = (ctx.arg->channelCount < maxGroupSize) ? ctx.arg->channelCount : maxGroupSize;

    for (uint32_t round = 0; round < rounds; round++) {
        uint16_t waitMask = 0;
        for (uint32_t g = 0; g < groupCount; g++) {
            const uint32_t channelStart = GetGroupDrrGroupStart(g, maxGroupSize);
            const uint32_t groupSize = GetGroupDrrGroupSize(ctx.arg->channelCount, g, maxGroupSize);
            waitMask |= GroupDrrIssueGroup(ctx, g, groupSize, round, channelStart);
        }
        ccu::EventWait(ctx.event, waitMask);
    }

    for (uint32_t g = 1; g < groupCount; g++) {
        ccu::LocalReduce(ctx.primaryBase, ctx.groupBase[g], ctx.blockBytes, ctx.arg->dataType, ctx.arg->reduceOp,
            ctx.event, 1);
        ccu::EventWait(ctx.event, 1);
    }

    if (ctx.arg->initOutput) {
        ccu::LocalReduce(
            ctx.primaryBase, ctx.selfSrc, ctx.blockBytes, ctx.arg->dataType, ctx.arg->reduceOp, ctx.event, 1);
        ccu::EventWait(ctx.event, 1);
    }
    return CCU_SUCCESS;
}

CcuResult CcuKernelGroupDrr(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<ReduceScatterKernelArg *>(arg);
    const uint32_t maxGroupSize = (kernelArg->stepOffset == 0) ? RS_GDRR_MAX_GROUP_SIZE : kernelArg->stepOffset;
    if (kernelArg->channelCount == 0 || kernelArg->channelCount > 8 || maxGroupSize == 0
        || maxGroupSize > RS_GDRR_MAX_GROUP_SIZE
        || GetGroupDrrGroupCount(kernelArg->channelCount, maxGroupSize) > RS_GDRR_MAX_GROUPS) {
        return CCU_E_PARA;
    }

    GroupDrrCtx ctx;
    ctx.arg = kernelArg;
    CCU_CHK_RET(GroupDrrInit(ctx));
    CCU_CHK_RET(GroupDrrLoadArgs(ctx));
    GroupDrrPreSync(ctx);
    GroupDrrBuildAddresses(ctx);
    CCU_CHK_RET(GroupDrrData(ctx));
    GroupDrrPostSync(ctx);
    return CCU_SUCCESS;
}

// ===========================================================================
// RH Kernel: WriteReduce(my_INPUT → peer_INPUT), 数据原地累积, 无需 CCL
// ===========================================================================

struct RhCtx {
    const ReduceScatterKernelArg *arg;

    ccu::Variable myInputVa;
    ccu::Variable myInputToken;
    ccu::Variable outputVa;
    ccu::Variable outputToken;
    ccu::Variable sliceBytes;
    ccu::Variable sliceByteOff;
    ccu::Variable blockBytes;

    ccu::Variable peerInputVa[RS_MAX_RANK_SIZE];
    ccu::Variable peerToken[RS_MAX_RANK_SIZE];

    ccu::Event event;
};

static CcuResult RhInitResource(RhCtx &ctx)
{
    const auto *arg = ctx.arg;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ctx.peerInputVa[i] = ccu::GetResByChannel<ccu::Variable>(arg->channels[i], RS_INPUT_XN_ID);
        ctx.peerToken[i] = ccu::GetResByChannel<ccu::Variable>(arg->channels[i], RS_TOKEN_XN_ID);
    }
    return CCU_SUCCESS;
}

static CcuResult RhLoadArgs(RhCtx &ctx)
{
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.myInputVa, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputVa, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.myInputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sliceByteOff, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.blockBytes, argId++));
    return CCU_SUCCESS;
}

static void RhPreSync(RhCtx &ctx)
{
    const auto *arg = ctx.arg;
    if (arg->channelCount == 0)
        return;
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::WriteVariableWithNotify(
            arg->channels[i], ctx.myInputVa, RS_INPUT_XN_ID, RS_CKE_IDX_0, 1 << RS_INPUT_XN_ID);
        ccu::WriteVariableWithNotify(
            arg->channels[i], ctx.myInputToken, RS_TOKEN_XN_ID, RS_CKE_IDX_0, 1 << RS_TOKEN_XN_ID);
    }
    uint32_t allBit = (1 << RS_INPUT_XN_ID) | (1 << RS_TOKEN_XN_ID);
    for (uint32_t i = 0; i < arg->channelCount; i++) {
        ccu::NotifyWait(arg->channels[i], RS_CKE_IDX_0, allBit);
    }
}

static void RhPostSync(RhCtx &ctx)
{
    const auto *arg = ctx.arg;
    if (arg->channelCount == 0)
        return;
    const uint32_t lastStep = arg->channelCount - 1;
    ccu::NotifyRecord(arg->channels[lastStep], RS_CKE_IDX_0, 1 << RS_POST_SYNC_ID);
    ccu::NotifyWait(arg->channels[lastStep], RS_CKE_IDX_0, 1 << RS_POST_SYNC_ID);
}

static void RhWriteReduceContiguous(RhCtx &ctx, uint32_t step, uint64_t sendBlk, uint64_t xferBlks)
{
    const auto *arg = ctx.arg;

    ccu::Variable xferBytes;
    xferBytes = 0;
    for (uint64_t i = 0; i < xferBlks; i++) {
        xferBytes = xferBytes + ctx.blockBytes;
    }
    ccu::Variable sendOffBytes;
    sendOffBytes = 0;
    for (uint64_t i = 0; i < sendBlk; i++) {
        sendOffBytes = sendOffBytes + ctx.blockBytes;
    }

    ccu::RemoteAddr peerKeep;
    peerKeep.addr = ctx.peerInputVa[step];
    peerKeep.addr = peerKeep.addr + sendOffBytes;
    peerKeep.addr = peerKeep.addr + ctx.sliceByteOff;
    peerKeep.token = ctx.peerToken[step];

    ccu::LocalAddr mySend;
    mySend.addr = ctx.myInputVa;
    mySend.addr = mySend.addr + sendOffBytes;
    mySend.addr = mySend.addr + ctx.sliceByteOff;
    mySend.token = ctx.myInputToken;

    ccu::WriteReduce(arg->channels[step], peerKeep, mySend, xferBytes, arg->dataType, arg->reduceOp, ctx.event, 1);
    ccu::EventWait(ctx.event, 1);
}

static void RhWriteReduceSliced(RhCtx &ctx, uint32_t step, uint64_t sendBlk, uint64_t xferBlks)
{
    const auto *arg = ctx.arg;
    uint16_t waitMask = 0;

    for (uint64_t blk = 0; blk < xferBlks; blk++) {
        const uint16_t mask = 1 << blk;

        ccu::Variable sendOffBytes;
        sendOffBytes = 0;
        for (uint64_t i = 0; i < sendBlk + blk; i++) {
            sendOffBytes = sendOffBytes + ctx.blockBytes;
        }

        ccu::RemoteAddr peerKeep;
        peerKeep.addr = ctx.peerInputVa[step];
        peerKeep.addr = peerKeep.addr + sendOffBytes;
        peerKeep.addr = peerKeep.addr + ctx.sliceByteOff;
        peerKeep.token = ctx.peerToken[step];

        ccu::LocalAddr mySend;
        mySend.addr = ctx.myInputVa;
        mySend.addr = mySend.addr + sendOffBytes;
        mySend.addr = mySend.addr + ctx.sliceByteOff;
        mySend.token = ctx.myInputToken;

        ccu::WriteReduce(
            arg->channels[step], peerKeep, mySend, ctx.sliceBytes, arg->dataType, arg->reduceOp, ctx.event, mask);
        waitMask |= mask;
    }
    ccu::EventWait(ctx.event, waitMask);
}

static void RhCopyResult(RhCtx &ctx)
{
    const auto *arg = ctx.arg;

    if (arg->initOutput) {
        ccu::Variable ownBlockOff;
        ownBlockOff = 0;
        for (uint32_t i = 0; i < arg->rankId; i++) {
            ownBlockOff = ownBlockOff + ctx.blockBytes;
        }

        ccu::LocalAddr inSrc;
        inSrc.addr = ctx.myInputVa;
        inSrc.addr = inSrc.addr + ownBlockOff;
        inSrc.addr = inSrc.addr + ctx.sliceByteOff;
        inSrc.token = ctx.myInputToken;

        ccu::LocalAddr outDst;
        outDst.addr = ctx.outputVa;
        outDst.addr = outDst.addr + ctx.sliceByteOff;
        outDst.token = ctx.outputToken;

        ccu::LocalCopy(outDst, inSrc, ctx.sliceBytes, ctx.event, 1);
        ccu::EventWait(ctx.event, 1);
    }
}

static CcuResult RhDataPhase(RhCtx &ctx, bool sliced)
{
    const auto *arg = ctx.arg;

    CCU_IF(ctx.sliceBytes != 0)
    {
        for (uint32_t step = 0; step < arg->channelCount; step++) {
            uint64_t maskVal = arg->rankSize >> (step + arg->stepOffset + 1);
            uint32_t g = arg->rankId;
            bool keepLow = ((g & maskVal) == 0);

            uint64_t off = 0;
            for (uint64_t m = arg->rankSize >> 1; m > maskVal; m >>= 1) {
                if (!((g & m) == 0))
                    off += m;
            }
            uint64_t sendBlk = keepLow ? (off + maskVal) : off;
            uint64_t xferBlks = maskVal;

            if (step + arg->stepOffset != 0) {
                ccu::NotifyRecord(arg->channels[step], RS_CKE_IDX_0, RS_STEP_PRE_SYNC_BIT);
                ccu::NotifyWait(arg->channels[step], RS_CKE_IDX_0, RS_STEP_PRE_SYNC_BIT);
            }

            // WriteReduce: my_INPUT.send → peer_INPUT.keep
            if (sliced) {
                RhWriteReduceSliced(ctx, step, sendBlk, xferBlks);
            } else {
                RhWriteReduceContiguous(ctx, step, sendBlk, xferBlks);
            }

            // The last step is synchronized by RhPostSync on the same channel.
            if (step + 1 < arg->channelCount) {
                ccu::NotifyRecord(arg->channels[step], RS_CKE_IDX_0, RS_STEP_POST_SYNC_BIT);
                ccu::NotifyWait(arg->channels[step], RS_CKE_IDX_0, RS_STEP_POST_SYNC_BIT);
            }
        }
    }

    return CCU_SUCCESS;
}

CcuResult CcuKernelRH(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<ReduceScatterKernelArg *>(arg);

    RhCtx ctx;
    ctx.arg = kernelArg;

    CCU_CHK_RET(RhInitResource(ctx));
    CCU_CHK_RET(RhLoadArgs(ctx));
    RhPreSync(ctx);
    CCU_CHK_RET(RhDataPhase(ctx, false));
    RhPostSync(ctx);
    RhCopyResult(ctx);

    return CCU_SUCCESS;
}

CcuResult CcuKernelRHSliced(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<ReduceScatterKernelArg *>(arg);

    RhCtx ctx;
    ctx.arg = kernelArg;

    CCU_CHK_RET(RhInitResource(ctx));
    CCU_CHK_RET(RhLoadArgs(ctx));
    RhPreSync(ctx);
    CCU_CHK_RET(RhDataPhase(ctx, true));
    RhPostSync(ctx);
    RhCopyResult(ctx);

    return CCU_SUCCESS;
}

} // namespace ops_hccl
