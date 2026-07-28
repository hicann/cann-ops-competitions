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

#include "ccu_kernel.h"

namespace ops_hccl {
namespace {

constexpr uint32_t ROOT_RANK = 0;
constexpr uint32_t INPUT_XN_ID = 0;
constexpr uint32_t OUTPUT_XN_ID = 1;
constexpr uint32_t TOKEN_XN_ID = 2;
constexpr uint32_t CKE_IDX_0 = 0;
constexpr uint32_t POST_SYNC_ID = 3;
constexpr uint32_t RECURSIVE_DOUBLING_READY_ID = 4;
constexpr uint32_t ALLGATHER_STEP_READY_ID = 5;
constexpr uint32_t HIER_READ_READY_ID = 6;
constexpr uint32_t HIER_FUSED_RS_DONE_ID = 7;
constexpr uint32_t SMALL_2D_INPUT_READ_DONE_ID = 4;
constexpr uint32_t SMALL_2D_PARTIAL_READY_ID = 5;
constexpr uint32_t SMALL_2D_PARTIAL_READ_DONE_ID = 6;
constexpr uint32_t SMALL_2D_COLUMN_AG_DONE_ID = 7;
constexpr uint64_t ALGORITHM_ROOT = 0;
constexpr uint64_t ALGORITHM_FULL_MESH = 1;
constexpr uint64_t ALGORITHM_RECURSIVE_DOUBLING = 2;
constexpr uint64_t ALGORITHM_SMALL_2D_BEST = 3;
constexpr uint64_t ALGORITHM_NHR_WRITE_REDUCE = 4;
constexpr uint64_t ALGORITHM_PAR_MESH_RS_INPUT = 6;
constexpr uint64_t ALGORITHM_PAR_MESH_RS_OUTPUT = 7;
constexpr uint64_t ALGORITHM_PAR_MESH_AG = 8;
constexpr uint64_t ALGORITHM_PAR_CLOS_RS = 9;
constexpr uint64_t ALGORITHM_PAR_CLOS_AG = 10;
constexpr uint64_t ALGORITHM_PAR_CLOS_RS_OUTPUT = 11;
constexpr uint64_t ALGORITHM_PAR_MESH_RS_AG = 12;
constexpr uint64_t ALGORITHM_PAR_CLOS_RS_AG = 13;
constexpr uint64_t ALGORITHM_MIXED_RADIX_12 = 20;
// The XOR path must not reuse a CKE bit while different rank pairs can be at
// different rounds.  Otherwise a fast pair can consume an old notification as
// the next round's notification and eventually create a cyclic wait graph.
constexpr uint32_t XOR_RS1_DONE_BIT = 6;
constexpr uint32_t XOR_RS2_READY_BIT = 7;
constexpr uint32_t XOR_RS2_DONE_BIT = 8;
constexpr uint32_t XOR_RS3_READY_BIT = 9;
constexpr uint32_t XOR_RS3_DONE_BIT = 10;
constexpr uint32_t XOR_RS4_READY_BIT = 11;
constexpr uint32_t XOR_RS4_DONE_BIT = 12;
constexpr uint32_t XOR_AG1_DONE_BIT = 13;
constexpr uint32_t XOR_AG2_DONE_BIT = 14;
constexpr uint32_t XOR_AG3_DONE_BIT = 15;
constexpr uint32_t MIXED_RADIX3_DONE_BIT = 10;
constexpr uint32_t MIXED_RADIX2_DONE_BIT = 11;
constexpr uint32_t MIXED_RADIX1_DONE_BIT = 12;
constexpr uint32_t MIXED_AG1_DONE_BIT = 13;
constexpr uint32_t MIXED_AG2_DONE_BIT = 14;

struct RootAllReduceContext {
    const CcuKernelArgAllReduce *arg = nullptr;
    std::vector<ccu::Variable> input;
    std::vector<ccu::Variable> output;
    std::vector<ccu::Variable> token;
    ccu::Variable sliceSize;
    ccu::Variable isInPlace;
    ccu::Variable ownerOffset;
    ccu::Variable ownerSize;
    ccu::Variable algorithmType;
    ccu::Variable scratch;
    ccu::Variable scratchToken;
    ccu::Variable pipelineTileSize;
    ccu::Variable pipelineTailSize;
    ccu::Variable fourGlobalSliceSize;
    ccu::Event event;
};

uint32_t ChannelIndex(uint32_t localRank, uint32_t peerRank)
{
    return peerRank < localRank ? peerRank : peerRank - 1;
}

CcuResult InitResources(RootAllReduceContext &ctx)
{
    if (ctx.arg->rankSize <= 1 || ctx.arg->rankSize > MAX_RANK_SIZE ||
        ctx.arg->channelCount != ctx.arg->rankSize - 1) {
        HCCL_ERROR("[CcuKernel] invalid rank/channel count: rankSize=%u channelCount=%u",
            ctx.arg->rankSize, ctx.arg->channelCount);
        return CCU_E_PARA;
    }

    ctx.input.resize(ctx.arg->rankSize);
    ctx.output.resize(ctx.arg->rankSize);
    ctx.token.resize(ctx.arg->rankSize);
    for (uint32_t peer = 0; peer < ctx.arg->rankSize; ++peer) {
        if (peer == ctx.arg->rankId) {
            continue;
        }
        uint32_t channelIndex = ChannelIndex(ctx.arg->rankId, peer);
        ctx.input[peer] = ccu::GetResByChannel<ccu::Variable>(
            ctx.arg->channels[channelIndex], INPUT_XN_ID);
        ctx.output[peer] = ccu::GetResByChannel<ccu::Variable>(
            ctx.arg->channels[channelIndex], OUTPUT_XN_ID);
        ctx.token[peer] = ccu::GetResByChannel<ccu::Variable>(
            ctx.arg->channels[channelIndex], TOKEN_XN_ID);
    }
    return CCU_SUCCESS;
}

CcuResult LoadArgs(RootAllReduceContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHECK_RET(ccu::LoadArg(ctx.input[ctx.arg->rankId], argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.output[ctx.arg->rankId], argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.token[ctx.arg->rankId], argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.sliceSize, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.isInPlace, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.ownerOffset, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.ownerSize, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.algorithmType, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.scratch, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.scratchToken, argId++));
    return CCU_SUCCESS;
}

CcuResult CopyInputToOutput(RootAllReduceContext &ctx)
{
    ccu::LocalAddr src;
    src.addr = ctx.input[ctx.arg->rankId];
    src.token = ctx.token[ctx.arg->rankId];

    ccu::LocalAddr dst;
    dst.addr = ctx.output[ctx.arg->rankId];
    dst.token = ctx.token[ctx.arg->rankId];

    CCU_IF(ctx.isInPlace == 0)
    {
        CCU_CHECK_RET(ccu::LocalCopy(dst, src, ctx.sliceSize, ctx.event, 1));
    }
    CCU_IF(ctx.isInPlace != 0)
    {
        CCU_CHECK_RET(ccu::EventRecord(ctx.event, 1));
    }
    CCU_CHECK_RET(ccu::EventWait(ctx.event, 1));
    return CCU_SUCCESS;
}

CcuResult RootPreSync(RootAllReduceContext &ctx)
{
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        CCU_CHECK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[i],
            ctx.output[ctx.arg->rankId], OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID));
        CCU_CHECK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[i],
            ctx.token[ctx.arg->rankId], TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID));
    }

    constexpr uint32_t allBits = (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        CCU_CHECK_RET(ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, allBits));
    }
    return CCU_SUCCESS;
}

CcuResult FullMeshPreSync(RootAllReduceContext &ctx)
{
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        CCU_CHECK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[i],
            ctx.input[ctx.arg->rankId], INPUT_XN_ID, CKE_IDX_0, 1 << INPUT_XN_ID));
        CCU_CHECK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[i],
            ctx.output[ctx.arg->rankId], OUTPUT_XN_ID, CKE_IDX_0, 1 << OUTPUT_XN_ID));
        CCU_CHECK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[i],
            ctx.token[ctx.arg->rankId], TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID));
    }

    constexpr uint32_t allBits = (1 << INPUT_XN_ID) |
        (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        CCU_CHECK_RET(ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, allBits));
    }
    return CCU_SUCCESS;
}

CcuResult ReduceAndBroadcast(RootAllReduceContext &ctx)
{
    if (ctx.arg->rankId != ROOT_RANK) {
        return CCU_SUCCESS;
    }

    ccu::LocalAddr localOutput;
    localOutput.addr = ctx.output[ROOT_RANK];
    localOutput.token = ctx.token[ROOT_RANK];

    for (uint32_t peer = 1; peer < ctx.arg->rankSize; ++peer) {
        uint32_t channelIndex = ChannelIndex(ROOT_RANK, peer);
        ccu::RemoteAddr remoteInput;
        remoteInput.addr = ctx.output[peer];
        remoteInput.token = ctx.token[peer];
        CCU_CHECK_RET(ccu::ReadReduce(ctx.arg->channels[channelIndex], localOutput,
            remoteInput, ctx.sliceSize, ctx.arg->dataType, ctx.arg->reduceOp, ctx.event, 1));
        CCU_CHECK_RET(ccu::EventWait(ctx.event, 1));
    }

    for (uint32_t peer = 1; peer < ctx.arg->rankSize; ++peer) {
        uint32_t channelIndex = ChannelIndex(ROOT_RANK, peer);
        ccu::RemoteAddr remoteOutput;
        remoteOutput.addr = ctx.output[peer];
        remoteOutput.token = ctx.token[peer];
        CCU_CHECK_RET(ccu::Write(ctx.arg->channels[channelIndex], remoteOutput,
            localOutput, ctx.sliceSize, ctx.event, 1));
        CCU_CHECK_RET(ccu::EventWait(ctx.event, 1));
    }
    return CCU_SUCCESS;
}

CcuResult CopyOwnerInputToOutput(RootAllReduceContext &ctx)
{
    ccu::LocalAddr src;
    src.addr = ctx.input[ctx.arg->rankId];
    src.addr += ctx.ownerOffset;
    src.token = ctx.token[ctx.arg->rankId];

    ccu::LocalAddr dst;
    dst.addr = ctx.output[ctx.arg->rankId];
    dst.addr += ctx.ownerOffset;
    dst.token = ctx.token[ctx.arg->rankId];

    CCU_IF(ctx.isInPlace == 0)
    {
        CCU_CHECK_RET(ccu::LocalCopy(dst, src, ctx.ownerSize, ctx.event, 1));
    }
    CCU_IF(ctx.isInPlace != 0)
    {
        CCU_CHECK_RET(ccu::EventRecord(ctx.event, 1));
    }
    CCU_CHECK_RET(ccu::EventWait(ctx.event, 1));
    return CCU_SUCCESS;
}

CcuResult DoublingAllGather16(RootAllReduceContext &ctx)
{
    // The host guarantees equal owner sizes for every 16-rank FullMesh slice.
    // Disseminate the owner results in four recursive-doubling rounds. After
    // round mask, every rank owns the aligned group containing 2*mask ranks.
    for (uint32_t mask = 1; mask < ctx.arg->rankSize; mask <<= 1) {
        const uint32_t peer = ctx.arg->rankId ^ mask;
        const uint32_t channelIndex = ChannelIndex(ctx.arg->rankId, peer);

        const uint32_t blockStartRank = ctx.arg->rankId & ~(mask - 1U);
        ccu::Variable blockOffset;
        blockOffset = 0;
        for (uint32_t i = 0; i < blockStartRank; ++i) {
            blockOffset += ctx.ownerSize;
        }
        ccu::Variable blockSize;
        blockSize = 0;
        for (uint32_t i = 0; i < mask; ++i) {
            blockSize += ctx.ownerSize;
        }

        ccu::LocalAddr localBlock;
        localBlock.addr = ctx.output[ctx.arg->rankId];
        localBlock.addr += blockOffset;
        localBlock.token = ctx.token[ctx.arg->rankId];

        ccu::RemoteAddr remoteOutput;
        remoteOutput.addr = ctx.output[peer];
        remoteOutput.addr += blockOffset;
        remoteOutput.token = ctx.token[peer];
        CCU_CHECK_RET(ccu::Write(ctx.arg->channels[channelIndex], remoteOutput,
            localBlock, blockSize, ctx.event, 1));
        CCU_CHECK_RET(ccu::EventWait(ctx.event, 1));

        // Local completion does not prove that the partner's opposite write
        // has completed. Exchange a ready bit before using the received block.
        CCU_CHECK_RET(ccu::NotifyRecord(ctx.arg->channels[channelIndex], CKE_IDX_0,
            1 << ALLGATHER_STEP_READY_ID));
        CCU_CHECK_RET(ccu::NotifyWait(ctx.arg->channels[channelIndex], CKE_IDX_0,
            1 << ALLGATHER_STEP_READY_ID));
    }
    return CCU_SUCCESS;
}

CcuResult FullMeshReduceScatterAndAllGather(RootAllReduceContext &ctx)
{
    ccu::LocalAddr localOwnerSlice;
    localOwnerSlice.addr = ctx.output[ctx.arg->rankId];
    localOwnerSlice.addr += ctx.ownerOffset;
    localOwnerSlice.token = ctx.token[ctx.arg->rankId];

    // Cyclic peer order makes every step a permutation: no step sends all
    // readers to the same source rank.  The order is fixed for each owner,
    // so floating-point reduction remains deterministic across executions.
    for (uint32_t step = 1; step < ctx.arg->rankSize; ++step) {
        const uint32_t peer = (ctx.arg->rankId + step) % ctx.arg->rankSize;
        const uint32_t channelIndex = ChannelIndex(ctx.arg->rankId, peer);
        ccu::RemoteAddr remoteInput;
        remoteInput.addr = ctx.input[peer];
        remoteInput.addr += ctx.ownerOffset;
        remoteInput.token = ctx.token[peer];
        CCU_CHECK_RET(ccu::ReadReduce(ctx.arg->channels[channelIndex], localOwnerSlice,
            remoteInput, ctx.ownerSize, ctx.arg->dataType, ctx.arg->reduceOp, ctx.event, 1));
        CCU_CHECK_RET(ccu::EventWait(ctx.event, 1));
    }

    if (ctx.arg->rankSize != 16) {
        // Keep the proven c923b7f schedule for every topology except 2x8.
        // This isolates the experiment to performance points 11/12 and
        // protects the existing 4-rank and 12-rank large-message results.
        for (uint32_t step = 1; step < ctx.arg->rankSize; ++step) {
            const uint32_t peer = (ctx.arg->rankId + step) % ctx.arg->rankSize;
            const uint32_t channelIndex = ChannelIndex(ctx.arg->rankId, peer);
            ccu::RemoteAddr remoteOutput;
            remoteOutput.addr = ctx.output[peer];
            remoteOutput.addr += ctx.ownerOffset;
            remoteOutput.token = ctx.token[peer];
            CCU_CHECK_RET(ccu::Write(ctx.arg->channels[channelIndex], remoteOutput,
                localOwnerSlice, ctx.ownerSize, ctx.event, 1));
            CCU_CHECK_RET(ccu::EventWait(ctx.event, 1));
        }
        return CCU_SUCCESS;
    }

    return DoublingAllGather16(ctx);
}

CcuResult Mesh16ScratchReduceScatterAndAllGather(RootAllReduceContext &ctx)
{
    std::vector<ccu::LocalAddr> scratchSlots(ctx.arg->rankSize);
    ccu::Variable scratchOffset;
    scratchOffset = 0;

    // Fetch this rank's owner slice from every physical rank into disjoint
    // local slots. Network completion order cannot affect the reduction tree.
    for (uint32_t rank = 0; rank < ctx.arg->rankSize; ++rank) {
        scratchSlots[rank].addr = ctx.scratch;
        scratchSlots[rank].addr += scratchOffset;
        scratchSlots[rank].token = ctx.scratchToken;
        scratchOffset += ctx.ownerSize;
        const uint32_t eventBit = 1U << rank;
        if (rank == ctx.arg->rankId) {
            ccu::LocalAddr localInput;
            localInput.addr = ctx.input[ctx.arg->rankId];
            localInput.addr += ctx.ownerOffset;
            localInput.token = ctx.token[ctx.arg->rankId];
            CCU_CHECK_RET(ccu::LocalCopy(scratchSlots[rank], localInput,
                ctx.ownerSize, ctx.event, eventBit));
        } else {
            const uint32_t channelIndex = ChannelIndex(ctx.arg->rankId, rank);
            ccu::RemoteAddr remoteInput;
            remoteInput.addr = ctx.input[rank];
            remoteInput.addr += ctx.ownerOffset;
            remoteInput.token = ctx.token[rank];
            CCU_CHECK_RET(ccu::Read(ctx.arg->channels[channelIndex], scratchSlots[rank],
                remoteInput, ctx.ownerSize, ctx.event, eventBit));
        }
    }
    const uint32_t allReadBits = (1U << ctx.arg->rankSize) - 1U;
    CCU_CHECK_RET(ccu::EventWait(ctx.event, allReadBits));

    // Pairwise tree copied from the official Mesh Mem2Mem structure. Slot
    // layout is fixed by physical rank, so floating-point order is stable.
    uint32_t remainingPieces = ctx.arg->rankSize;
    while (remainingPieces > 1) {
        const uint32_t reducePieces = remainingPieces / 2;
        const uint32_t srcIndex = remainingPieces - reducePieces;
        ccu::Variable reduceBytes;
        reduceBytes = 0;
        for (uint32_t i = 0; i < reducePieces; ++i) {
            reduceBytes += ctx.ownerSize;
        }
        CCU_CHECK_RET(ccu::LocalReduce(scratchSlots[0], scratchSlots[srcIndex],
            reduceBytes, ctx.arg->dataType, ctx.arg->reduceOp, ctx.event, 1));
        CCU_CHECK_RET(ccu::EventWait(ctx.event, 1));
        remainingPieces -= reducePieces;
    }

    ccu::LocalAddr localOwnerSlice;
    localOwnerSlice.addr = ctx.output[ctx.arg->rankId];
    localOwnerSlice.addr += ctx.ownerOffset;
    localOwnerSlice.token = ctx.token[ctx.arg->rankId];
    CCU_CHECK_RET(ccu::LocalCopy(localOwnerSlice, scratchSlots[0],
        ctx.ownerSize, ctx.event, 1));
    CCU_CHECK_RET(ccu::EventWait(ctx.event, 1));

    return DoublingAllGather16(ctx);
}

CcuResult NhrAllGather12(RootAllReduceContext &ctx)
{
    constexpr uint32_t rankSize = 12;
    constexpr uint32_t stepCount = 4;

    // Official NHR dissemination for a non-power-of-two rank count.  The
    // four steps send 1, 1, 3 and 6 owner slices respectively.  Total bytes
    // are identical to direct FullMesh AllGather, while synchronization drops
    // from eleven peer waits to four propagation rounds.
    for (uint32_t step = 0; step < stepCount; ++step) {
        const uint32_t delta = 1U << (stepCount - 1U - step);
        const uint32_t sendTo = (ctx.arg->rankId + delta) % rankSize;
        const uint32_t recvFrom =
            (ctx.arg->rankId + rankSize - delta) % rankSize;
        const uint32_t sendChannelIndex = ChannelIndex(ctx.arg->rankId, sendTo);
        const uint32_t recvChannelIndex = ChannelIndex(ctx.arg->rankId, recvFrom);
        const uint32_t sliceCount = (rankSize - 1U + delta) / (delta << 1U);
        const uint32_t sliceStride = delta << 1U;

        uint32_t sliceIndex = ctx.arg->rankId;
        for (uint32_t i = 0; i < sliceCount; ++i) {
            ccu::Variable sliceOffset;
            sliceOffset = 0;
            for (uint32_t owner = 0; owner < sliceIndex; ++owner) {
                sliceOffset += ctx.ownerSize;
            }

            ccu::LocalAddr localSlice;
            localSlice.addr = ctx.output[ctx.arg->rankId];
            localSlice.addr += sliceOffset;
            localSlice.token = ctx.token[ctx.arg->rankId];

            ccu::RemoteAddr remoteSlice;
            remoteSlice.addr = ctx.output[sendTo];
            remoteSlice.addr += sliceOffset;
            remoteSlice.token = ctx.token[sendTo];
            CCU_CHECK_RET(ccu::Write(ctx.arg->channels[sendChannelIndex], remoteSlice,
                localSlice, ctx.ownerSize, ctx.event, 1U << i));

            sliceIndex = (sliceIndex + rankSize - sliceStride) % rankSize;
        }
        const uint32_t stepMask = (1U << sliceCount) - 1U;
        CCU_CHECK_RET(ccu::EventWait(ctx.event, stepMask));

        // Preserve the proven B-28/B-11 dependency on all four NHR rounds.
        // A-29's final-round elision correlated with a clear point-18
        // regression and is unrelated to the successful 2x8 parallel path.
        CCU_CHECK_RET(ccu::NotifyRecord(ctx.arg->channels[sendChannelIndex], CKE_IDX_0,
            1 << ALLGATHER_STEP_READY_ID));
        CCU_CHECK_RET(ccu::NotifyWait(ctx.arg->channels[recvChannelIndex], CKE_IDX_0,
            1 << ALLGATHER_STEP_READY_ID));
    }
    return CCU_SUCCESS;
}

CcuResult Mesh12ScratchReduceScatterAndAllGather(RootAllReduceContext &ctx)
{
    std::vector<ccu::LocalAddr> scratchSlots(ctx.arg->rankSize);
    ccu::Variable scratchOffset;
    scratchOffset = 0;

    // This is the same deterministic Scratch layout used by the official
    // Mesh Mem2Mem kernel: one slot per physical rank, with all transfers
    // issued before a single aggregate wait.
    for (uint32_t rank = 0; rank < ctx.arg->rankSize; ++rank) {
        scratchSlots[rank].addr = ctx.scratch;
        scratchSlots[rank].addr += scratchOffset;
        scratchSlots[rank].token = ctx.scratchToken;
        scratchOffset += ctx.ownerSize;

        const uint32_t eventBit = 1U << rank;
        if (rank == ctx.arg->rankId) {
            ccu::LocalAddr localInput;
            localInput.addr = ctx.input[ctx.arg->rankId];
            localInput.addr += ctx.ownerOffset;
            localInput.token = ctx.token[ctx.arg->rankId];
            CCU_CHECK_RET(ccu::LocalCopy(scratchSlots[rank], localInput,
                ctx.ownerSize, ctx.event, eventBit));
        } else {
            const uint32_t channelIndex = ChannelIndex(ctx.arg->rankId, rank);
            ccu::RemoteAddr remoteInput;
            remoteInput.addr = ctx.input[rank];
            remoteInput.addr += ctx.ownerOffset;
            remoteInput.token = ctx.token[rank];
            CCU_CHECK_RET(ccu::Read(ctx.arg->channels[channelIndex], scratchSlots[rank],
                remoteInput, ctx.ownerSize, ctx.event, eventBit));
        }
    }
    constexpr uint32_t allReadBits = (1U << 12) - 1U;
    CCU_CHECK_RET(ccu::EventWait(ctx.event, allReadBits));

    // Reduce fixed physical-rank slots in the official pairwise progression:
    // 12 -> 6 -> 3 -> 2 -> 1. Network completion order cannot change FP32
    // addition order, so repeated executions remain deterministic.
    uint32_t remainingPieces = ctx.arg->rankSize;
    while (remainingPieces > 1) {
        const uint32_t reducePieces = remainingPieces / 2;
        const uint32_t srcIndex = remainingPieces - reducePieces;
        ccu::Variable reduceBytes;
        reduceBytes = 0;
        for (uint32_t i = 0; i < reducePieces; ++i) {
            reduceBytes += ctx.ownerSize;
        }
        CCU_CHECK_RET(ccu::LocalReduce(scratchSlots[0], scratchSlots[srcIndex],
            reduceBytes, ctx.arg->dataType, ctx.arg->reduceOp, ctx.event, 1));
        CCU_CHECK_RET(ccu::EventWait(ctx.event, 1));
        remainingPieces -= reducePieces;
    }

    ccu::LocalAddr localOwnerSlice;
    localOwnerSlice.addr = ctx.output[ctx.arg->rankId];
    localOwnerSlice.addr += ctx.ownerOffset;
    localOwnerSlice.token = ctx.token[ctx.arg->rankId];
    CCU_CHECK_RET(ccu::LocalCopy(localOwnerSlice, scratchSlots[0],
        ctx.ownerSize, ctx.event, 1));
    CCU_CHECK_RET(ccu::EventWait(ctx.event, 1));

    return NhrAllGather12(ctx);
}

CcuResult MixedRadixPairSync(
    RootAllReduceContext &ctx, uint32_t peer, uint32_t bit)
{
    const uint32_t channel = ChannelIndex(ctx.arg->rankId, peer);
    CCU_CHECK_RET(ccu::NotifyRecord(
        ctx.arg->channels[channel], CKE_IDX_0, 1U << bit));
    CCU_CHECK_RET(ccu::NotifyWait(
        ctx.arg->channels[channel], CKE_IDX_0, 1U << bit));
    return CCU_SUCCESS;
}

CcuResult MixedRadixTripleSync(RootAllReduceContext &ctx,
    const uint32_t peers[2], uint32_t bit)
{
    for (uint32_t i = 0; i < 2U; ++i) {
        const uint32_t channel =
            ChannelIndex(ctx.arg->rankId, peers[i]);
        CCU_CHECK_RET(ccu::NotifyRecord(
            ctx.arg->channels[channel], CKE_IDX_0, 1U << bit));
    }
    for (uint32_t i = 0; i < 2U; ++i) {
        const uint32_t channel =
            ChannelIndex(ctx.arg->rankId, peers[i]);
        CCU_CHECK_RET(ccu::NotifyWait(
            ctx.arg->channels[channel], CKE_IDX_0, 1U << bit));
    }
    return CCU_SUCCESS;
}

CcuResult MixedRadix12AllReduce(RootAllReduceContext &ctx)
{
    constexpr uint32_t rankSize = 12;
    constexpr uint32_t ownersPerHalf = 6;
    constexpr uint32_t ownersPerQuarter = 3;
    ccu::Variable offsets[rankSize + 1];
    offsets[0] = 0;
    for (uint32_t i = 1; i <= rankSize; ++i) {
        offsets[i] = offsets[i - 1] + ctx.ownerSize;
    }

    const uint32_t rank = ctx.arg->rankId;
    const uint32_t radix3 = rank / 4U;
    const uint32_t radix2 = (rank / 2U) & 1U;
    const uint32_t radix1 = rank & 1U;
    const uint32_t halfOwner = radix1 * ownersPerHalf;
    const uint32_t quarterOwner =
        halfOwner + radix2 * ownersPerQuarter;
    const uint32_t finalOwner = quarterOwner + radix3;

    ccu::Variable halfSize;
    halfSize = offsets[ownersPerHalf];
    ccu::Variable quarterSize;
    quarterSize = offsets[ownersPerQuarter];

    ccu::LocalAddr localHalf;
    localHalf.addr = ctx.output[rank];
    localHalf.addr += offsets[halfOwner];
    localHalf.token = ctx.token[rank];
    ccu::LocalAddr inputHalf;
    inputHalf.addr = ctx.input[rank];
    inputHalf.addr += offsets[halfOwner];
    inputHalf.token = ctx.token[rank];
    CCU_IF(ctx.isInPlace == 0)
    {
        CCU_CHECK_RET(ccu::LocalCopy(
            localHalf, inputHalf, halfSize, ctx.event, 1U));
    }
    CCU_IF(ctx.isInPlace != 0)
    {
        CCU_CHECK_RET(ccu::EventRecord(ctx.event, 1U));
    }
    CCU_CHECK_RET(ccu::EventWait(ctx.event, 1U));

    const uint32_t peerRadix1 = rank ^ 1U;
    ccu::RemoteAddr remoteInputHalf;
    remoteInputHalf.addr = ctx.input[peerRadix1];
    remoteInputHalf.addr += offsets[halfOwner];
    remoteInputHalf.token = ctx.token[peerRadix1];
    CCU_CHECK_RET(ccu::ReadReduce(
        ctx.arg->channels[ChannelIndex(rank, peerRadix1)],
        localHalf, remoteInputHalf, halfSize, ctx.arg->dataType,
        ctx.arg->reduceOp, ctx.event, 1U));
    CCU_CHECK_RET(ccu::EventWait(ctx.event, 1U));

    const uint32_t peerRadix2 = rank ^ 2U;
    CCU_CHECK_RET(MixedRadixPairSync(
        ctx, peerRadix2, MIXED_RADIX3_DONE_BIT));
    ccu::LocalAddr localQuarter;
    localQuarter.addr = ctx.output[rank];
    localQuarter.addr += offsets[quarterOwner];
    localQuarter.token = ctx.token[rank];
    ccu::RemoteAddr remoteQuarter;
    remoteQuarter.addr = ctx.output[peerRadix2];
    remoteQuarter.addr += offsets[quarterOwner];
    remoteQuarter.token = ctx.token[peerRadix2];
    CCU_CHECK_RET(ccu::ReadReduce(
        ctx.arg->channels[ChannelIndex(rank, peerRadix2)],
        localQuarter, remoteQuarter, quarterSize, ctx.arg->dataType,
        ctx.arg->reduceOp, ctx.event, 1U));
    CCU_CHECK_RET(ccu::EventWait(ctx.event, 1U));

    uint32_t peer0 =
        ((radix3 + 1U) % 3U) * 4U + (rank % 4U);
    uint32_t peer1 =
        ((radix3 + 2U) % 3U) * 4U + (rank % 4U);
    if (peer1 < peer0) {
        const uint32_t tmp = peer0;
        peer0 = peer1;
        peer1 = tmp;
    }
    const uint32_t peers[2] = {peer0, peer1};

    CCU_CHECK_RET(MixedRadixTripleSync(
        ctx, peers, MIXED_RADIX2_DONE_BIT));

    ccu::LocalAddr scratch0;
    scratch0.addr = ctx.scratch;
    scratch0.token = ctx.scratchToken;
    ccu::LocalAddr scratch1;
    scratch1.addr = ctx.scratch;
    scratch1.addr += ctx.ownerSize;
    scratch1.token = ctx.scratchToken;
    ccu::LocalAddr scratchSlots[2] = {scratch0, scratch1};
    for (uint32_t i = 0; i < 2U; ++i) {
        const uint32_t peer = peers[i];
        ccu::RemoteAddr remoteOwner;
        remoteOwner.addr = ctx.output[peer];
        remoteOwner.addr += offsets[finalOwner];
        remoteOwner.token = ctx.token[peer];
        CCU_CHECK_RET(ccu::Read(
            ctx.arg->channels[ChannelIndex(rank, peer)],
            scratchSlots[i], remoteOwner, ctx.ownerSize,
            ctx.event, 1U << i));
    }
    CCU_CHECK_RET(ccu::EventWait(ctx.event, 0x3U));
    ccu::LocalAddr localOwner;
    localOwner.addr = ctx.output[rank];
    localOwner.addr += offsets[finalOwner];
    localOwner.token = ctx.token[rank];
    CCU_CHECK_RET(ccu::LocalReduce(
        localOwner, scratch0, ctx.ownerSize,
        ctx.arg->dataType, ctx.arg->reduceOp, ctx.event, 1U));
    CCU_CHECK_RET(ccu::EventWait(ctx.event, 1U));
    CCU_CHECK_RET(ccu::LocalReduce(
        localOwner, scratch1, ctx.ownerSize,
        ctx.arg->dataType, ctx.arg->reduceOp, ctx.event, 1U));
    CCU_CHECK_RET(ccu::EventWait(ctx.event, 1U));
    CCU_CHECK_RET(MixedRadixTripleSync(
        ctx, peers, MIXED_RADIX1_DONE_BIT));

    for (uint32_t i = 0; i < 2U; ++i) {
        const uint32_t peer = peers[i];
        ccu::RemoteAddr remoteAgOwner;
        remoteAgOwner.addr = ctx.output[peer];
        remoteAgOwner.addr += offsets[finalOwner];
        remoteAgOwner.token = ctx.token[peer];
        CCU_CHECK_RET(ccu::Write(
            ctx.arg->channels[ChannelIndex(rank, peer)],
            remoteAgOwner, localOwner, ctx.ownerSize,
            ctx.event, 1U << i));
    }
    CCU_CHECK_RET(ccu::EventWait(ctx.event, 0x3U));
    CCU_CHECK_RET(MixedRadixTripleSync(
        ctx, peers, MIXED_AG1_DONE_BIT));

    ccu::LocalAddr localAgQuarter;
    localAgQuarter.addr = ctx.output[rank];
    localAgQuarter.addr += offsets[quarterOwner];
    localAgQuarter.token = ctx.token[rank];
    ccu::RemoteAddr remoteAgQuarter;
    remoteAgQuarter.addr = ctx.output[peerRadix2];
    remoteAgQuarter.addr += offsets[quarterOwner];
    remoteAgQuarter.token = ctx.token[peerRadix2];
    CCU_CHECK_RET(ccu::Write(
        ctx.arg->channels[ChannelIndex(rank, peerRadix2)],
        remoteAgQuarter, localAgQuarter, quarterSize,
        ctx.event, 1U));
    CCU_CHECK_RET(ccu::EventWait(ctx.event, 1U));
    CCU_CHECK_RET(MixedRadixPairSync(
        ctx, peerRadix2, MIXED_AG2_DONE_BIT));

    ccu::LocalAddr localAgHalf;
    localAgHalf.addr = ctx.output[rank];
    localAgHalf.addr += offsets[halfOwner];
    localAgHalf.token = ctx.token[rank];
    ccu::RemoteAddr remoteAgHalf;
    remoteAgHalf.addr = ctx.output[peerRadix1];
    remoteAgHalf.addr += offsets[halfOwner];
    remoteAgHalf.token = ctx.token[peerRadix1];
    CCU_CHECK_RET(ccu::Write(
        ctx.arg->channels[ChannelIndex(rank, peerRadix1)],
        remoteAgHalf, localAgHalf, halfSize,
        ctx.event, 1U));
    CCU_CHECK_RET(ccu::EventWait(ctx.event, 1U));
    return CCU_SUCCESS;
}

uint32_t HighestPowerOfTwo(uint32_t value)
{
    uint32_t result = 1;
    while ((result << 1) <= value) {
        result <<= 1;
    }
    return result;
}

uint32_t ActiveRankToPhysical(uint32_t activeRank, uint32_t remainder)
{
    return activeRank < remainder ? activeRank * 2 + 1 : activeRank + remainder;
}

void AddUniquePeer(std::vector<uint32_t> &peers, uint32_t peer)
{
    for (uint32_t existing : peers) {
        if (existing == peer) {
            return;
        }
    }
    peers.push_back(peer);
}

std::vector<uint32_t> GetRecursiveDoublingPeers(const RootAllReduceContext &ctx)
{
    std::vector<uint32_t> peers;
    const uint32_t rankSize = ctx.arg->rankSize;
    const uint32_t rank = ctx.arg->rankId;
    const uint32_t powerOfTwo = HighestPowerOfTwo(rankSize);
    const uint32_t remainder = rankSize - powerOfTwo;

    bool active = true;
    uint32_t activeRank = rank;
    if (rank < 2 * remainder) {
        AddUniquePeer(peers, rank ^ 1);
        if ((rank & 1) == 0) {
            active = false;
        } else {
            activeRank = rank / 2;
        }
    } else {
        activeRank = rank - remainder;
    }

    if (active) {
        for (uint32_t mask = 1; mask < powerOfTwo; mask <<= 1) {
            const uint32_t peerActiveRank = activeRank ^ mask;
            AddUniquePeer(peers, ActiveRankToPhysical(peerActiveRank, remainder));
        }
    }
    return peers;
}

CcuResult RecursiveDoublingPreSync(RootAllReduceContext &ctx,
    const std::vector<uint32_t> &peers)
{
    for (uint32_t peer : peers) {
        const uint32_t channelIndex = ChannelIndex(ctx.arg->rankId, peer);
        CCU_CHECK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIndex],
            ctx.scratch, INPUT_XN_ID, CKE_IDX_0, 1 << INPUT_XN_ID));
        CCU_CHECK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIndex],
            ctx.scratchToken, TOKEN_XN_ID, CKE_IDX_0, 1 << TOKEN_XN_ID));
    }

    constexpr uint32_t allBits = (1 << INPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t peer : peers) {
        const uint32_t channelIndex = ChannelIndex(ctx.arg->rankId, peer);
        CCU_CHECK_RET(ccu::NotifyWait(ctx.arg->channels[channelIndex], CKE_IDX_0, allBits));
    }
    return CCU_SUCCESS;
}

CcuResult RecursiveDoublingPairSync(RootAllReduceContext &ctx, uint32_t peer)
{
    const uint32_t channelIndex = ChannelIndex(ctx.arg->rankId, peer);
    CCU_CHECK_RET(ccu::NotifyRecord(ctx.arg->channels[channelIndex], CKE_IDX_0,
        1 << RECURSIVE_DOUBLING_READY_ID));
    CCU_CHECK_RET(ccu::NotifyWait(ctx.arg->channels[channelIndex], CKE_IDX_0,
        1 << RECURSIVE_DOUBLING_READY_ID));
    return CCU_SUCCESS;
}

CcuResult PublishRecursiveDoublingSnapshot(RootAllReduceContext &ctx, uint32_t slot)
{
    ccu::LocalAddr src;
    src.addr = ctx.output[ctx.arg->rankId];
    src.token = ctx.token[ctx.arg->rankId];

    ccu::LocalAddr dst;
    dst.addr = ctx.scratch;
    for (uint32_t i = 0; i < slot; ++i) {
        dst.addr += ctx.sliceSize;
    }
    dst.token = ctx.scratchToken;

    CCU_CHECK_RET(ccu::LocalCopy(dst, src, ctx.sliceSize, ctx.event, 1));
    CCU_CHECK_RET(ccu::EventWait(ctx.event, 1));
    return CCU_SUCCESS;
}

CcuResult ReduceRecursiveDoublingSnapshot(RootAllReduceContext &ctx,
    uint32_t peer, uint32_t slot)
{
    const uint32_t channelIndex = ChannelIndex(ctx.arg->rankId, peer);
    ccu::LocalAddr localOutput;
    localOutput.addr = ctx.output[ctx.arg->rankId];
    localOutput.token = ctx.token[ctx.arg->rankId];

    ccu::RemoteAddr remoteSnapshot;
    remoteSnapshot.addr = ctx.input[peer];
    for (uint32_t i = 0; i < slot; ++i) {
        remoteSnapshot.addr += ctx.sliceSize;
    }
    remoteSnapshot.token = ctx.token[peer];

    CCU_CHECK_RET(ccu::ReadReduce(ctx.arg->channels[channelIndex], localOutput,
        remoteSnapshot, ctx.sliceSize, ctx.arg->dataType, ctx.arg->reduceOp, ctx.event, 1));
    CCU_CHECK_RET(ccu::EventWait(ctx.event, 1));
    return CCU_SUCCESS;
}

CcuResult ReadRecursiveDoublingSnapshot(RootAllReduceContext &ctx,
    uint32_t peer, uint32_t slot)
{
    const uint32_t channelIndex = ChannelIndex(ctx.arg->rankId, peer);
    ccu::LocalAddr localOutput;
    localOutput.addr = ctx.output[ctx.arg->rankId];
    localOutput.token = ctx.token[ctx.arg->rankId];

    ccu::RemoteAddr remoteSnapshot;
    remoteSnapshot.addr = ctx.input[peer];
    for (uint32_t i = 0; i < slot; ++i) {
        remoteSnapshot.addr += ctx.sliceSize;
    }
    remoteSnapshot.token = ctx.token[peer];

    CCU_CHECK_RET(ccu::Read(ctx.arg->channels[channelIndex], localOutput,
        remoteSnapshot, ctx.sliceSize, ctx.event, 1));
    CCU_CHECK_RET(ccu::EventWait(ctx.event, 1));
    return CCU_SUCCESS;
}

CcuResult RecursiveDoublingPostSync(RootAllReduceContext &ctx,
    const std::vector<uint32_t> &peers)
{
    for (uint32_t peer : peers) {
        const uint32_t channelIndex = ChannelIndex(ctx.arg->rankId, peer);
        CCU_CHECK_RET(ccu::NotifyRecord(ctx.arg->channels[channelIndex], CKE_IDX_0,
            1 << POST_SYNC_ID));
    }
    for (uint32_t peer : peers) {
        const uint32_t channelIndex = ChannelIndex(ctx.arg->rankId, peer);
        CCU_CHECK_RET(ccu::NotifyWait(ctx.arg->channels[channelIndex], CKE_IDX_0,
            1 << POST_SYNC_ID));
    }
    return CCU_SUCCESS;
}

CcuResult Small2DPeerSync(RootAllReduceContext &ctx,
    const std::vector<uint32_t> &peers, uint32_t syncId)
{
    for (uint32_t peer : peers) {
        const uint32_t channelIndex = ChannelIndex(ctx.arg->rankId, peer);
        CCU_CHECK_RET(ccu::NotifyRecord(ctx.arg->channels[channelIndex], CKE_IDX_0,
            1U << syncId));
    }
    for (uint32_t peer : peers) {
        const uint32_t channelIndex = ChannelIndex(ctx.arg->rankId, peer);
        CCU_CHECK_RET(ccu::NotifyWait(ctx.arg->channels[channelIndex], CKE_IDX_0,
            1U << syncId));
    }
    return CCU_SUCCESS;
}

std::vector<uint32_t> GetSmall2DPeers(
    const RootAllReduceContext &ctx, bool localPeersOnly)
{
    std::vector<uint32_t> peers;
    const uint32_t rank = ctx.arg->rankId;
    const uint32_t groupSize = ctx.arg->rankSize == 16 ? 8U : 4U;
    const uint32_t groupCount = ctx.arg->rankSize / groupSize;
    const uint32_t groupBase = (rank / groupSize) * groupSize;
    const uint32_t lane = rank % groupSize;

    for (uint32_t localLane = 0; localLane < groupSize; ++localLane) {
        const uint32_t peer = groupBase + localLane;
        if (peer != rank) {
            peers.push_back(peer);
        }
    }
    if (!localPeersOnly) {
        for (uint32_t group = 0; group < groupCount; ++group) {
            const uint32_t peer = group * groupSize + lane;
            if (peer != rank) {
                peers.push_back(peer);
            }
        }
    }
    return peers;
}

CcuResult Small2DPreSync(RootAllReduceContext &ctx,
    const std::vector<uint32_t> &peers)
{
    for (uint32_t peer : peers) {
        const uint32_t channelIndex = ChannelIndex(ctx.arg->rankId, peer);
        CCU_CHECK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIndex],
            ctx.input[ctx.arg->rankId], INPUT_XN_ID, CKE_IDX_0, 1U << INPUT_XN_ID));
        CCU_CHECK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIndex],
            ctx.output[ctx.arg->rankId], OUTPUT_XN_ID, CKE_IDX_0, 1U << OUTPUT_XN_ID));
        CCU_CHECK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIndex],
            ctx.token[ctx.arg->rankId], TOKEN_XN_ID, CKE_IDX_0, 1U << TOKEN_XN_ID));
    }

    constexpr uint32_t allBits = (1U << INPUT_XN_ID) |
        (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
    for (uint32_t peer : peers) {
        const uint32_t channelIndex = ChannelIndex(ctx.arg->rankId, peer);
        CCU_CHECK_RET(ccu::NotifyWait(ctx.arg->channels[channelIndex], CKE_IDX_0, allBits));
    }
    return CCU_SUCCESS;
}

CcuResult Small2DBestAllReduce(RootAllReduceContext &ctx)
{
    constexpr uint32_t groupSize = 4;
    constexpr uint32_t groupCount = 3;
    if (ctx.arg->rankSize != groupSize * groupCount) {
        return CCU_SUCCESS;
    }
    const uint32_t rank = ctx.arg->rankId;
    const uint32_t groupIndex = rank / groupSize;
    const uint32_t groupBase = groupIndex * groupSize;
    const uint32_t lane = rank % groupSize;
    const std::vector<uint32_t> localPeers = GetSmall2DPeers(ctx, true);
    const std::vector<uint32_t> allPeers = GetSmall2DPeers(ctx, false);
    std::vector<uint32_t> crossPeers;
    for (uint32_t group = 0; group < groupCount; ++group) {
        const uint32_t peer = group * groupSize + lane;
        if (peer != rank) {
            crossPeers.push_back(peer);
        }
    }
    CCU_CHECK_RET(Small2DPreSync(ctx, allPeers));

    std::vector<ccu::LocalAddr> scratchSlots(groupSize);
    ccu::Variable scratchOffset;
    scratchOffset = 0;
    for (uint32_t slot = 0; slot < groupSize; ++slot) {
        scratchSlots[slot].addr = ctx.scratch;
        scratchSlots[slot].addr += scratchOffset;
        scratchSlots[slot].token = ctx.scratchToken;
        scratchOffset += ctx.sliceSize;

        const uint32_t sourceRank = groupBase + slot;
        const uint32_t eventBit = 1U << slot;
        if (sourceRank == rank) {
            ccu::LocalAddr localInput;
            localInput.addr = ctx.input[rank];
            localInput.token = ctx.token[rank];
            CCU_CHECK_RET(ccu::LocalCopy(scratchSlots[slot], localInput,
                ctx.sliceSize, ctx.event, eventBit));
        } else {
            ccu::RemoteAddr remoteInput;
            remoteInput.addr = ctx.input[sourceRank];
            remoteInput.token = ctx.token[sourceRank];
            CCU_CHECK_RET(ccu::Read(
                ctx.arg->channels[ChannelIndex(rank, sourceRank)],
                scratchSlots[slot], remoteInput, ctx.sliceSize, ctx.event, eventBit));
        }
    }
    CCU_CHECK_RET(ccu::EventWait(ctx.event, 0xFU));
    CCU_CHECK_RET(Small2DPeerSync(
        ctx, localPeers, SMALL_2D_INPUT_READ_DONE_ID));

    ccu::Variable twoSliceSize;
    twoSliceSize = ctx.sliceSize;
    twoSliceSize += ctx.sliceSize;
    CCU_CHECK_RET(ccu::LocalReduce(scratchSlots[0], scratchSlots[2],
        twoSliceSize, ctx.arg->dataType, ctx.arg->reduceOp, ctx.event, 1U));
    CCU_CHECK_RET(ccu::EventWait(ctx.event, 1U));
    CCU_CHECK_RET(ccu::LocalReduce(scratchSlots[0], scratchSlots[1],
        ctx.sliceSize, ctx.arg->dataType, ctx.arg->reduceOp, ctx.event, 1U));
    CCU_CHECK_RET(ccu::EventWait(ctx.event, 1U));

    if (groupIndex != 0) {
        CCU_CHECK_RET(ccu::LocalCopy(scratchSlots[groupIndex], scratchSlots[0],
            ctx.sliceSize, ctx.event, 1U));
        CCU_CHECK_RET(ccu::EventWait(ctx.event, 1U));
    }

    ccu::LocalAddr localOutput;
    localOutput.addr = ctx.output[rank];
    localOutput.token = ctx.token[rank];
    CCU_CHECK_RET(ccu::LocalCopy(localOutput, scratchSlots[groupIndex],
        ctx.sliceSize, ctx.event, 1U));
    CCU_CHECK_RET(ccu::EventWait(ctx.event, 1U));
    CCU_CHECK_RET(Small2DPeerSync(
        ctx, crossPeers, SMALL_2D_PARTIAL_READY_ID));

    uint32_t remoteReadBits = 0;
    for (uint32_t group = 0; group < groupCount; ++group) {
        if (group == groupIndex) {
            continue;
        }
        const uint32_t peer = group * groupSize + lane;
        const uint32_t eventBit = 1U << group;
        ccu::RemoteAddr remotePartial;
        remotePartial.addr = ctx.output[peer];
        remotePartial.token = ctx.token[peer];
        CCU_CHECK_RET(ccu::Read(
            ctx.arg->channels[ChannelIndex(rank, peer)],
            scratchSlots[group], remotePartial, ctx.sliceSize, ctx.event, eventBit));
        remoteReadBits |= eventBit;
    }
    CCU_CHECK_RET(ccu::EventWait(ctx.event, remoteReadBits));
    CCU_CHECK_RET(Small2DPeerSync(
        ctx, crossPeers, SMALL_2D_PARTIAL_READ_DONE_ID));

    CCU_CHECK_RET(ccu::LocalReduce(scratchSlots[0], scratchSlots[1],
        ctx.sliceSize, ctx.arg->dataType, ctx.arg->reduceOp, ctx.event, 1U));
    CCU_CHECK_RET(ccu::EventWait(ctx.event, 1U));
    CCU_CHECK_RET(ccu::LocalReduce(scratchSlots[0], scratchSlots[2],
        ctx.sliceSize, ctx.arg->dataType, ctx.arg->reduceOp, ctx.event, 1U));
    CCU_CHECK_RET(ccu::EventWait(ctx.event, 1U));
    CCU_CHECK_RET(ccu::LocalCopy(localOutput, scratchSlots[0],
        ctx.sliceSize, ctx.event, 1U));
    CCU_CHECK_RET(ccu::EventWait(ctx.event, 1U));
    CCU_CHECK_RET(RecursiveDoublingPostSync(ctx, allPeers));
    return CCU_SUCCESS;
}

uint64_t Small2DChunkSize(uint32_t ownerIndex)
{
    constexpr uint64_t totalElements = (512U * 1024U) / sizeof(float);
    constexpr uint64_t baseElements = totalElements / 12U;
    constexpr uint32_t extraOwners = totalElements % 12U;
    return (baseElements + (ownerIndex < extraOwners ? 1U : 0U)) * sizeof(float);
}

uint64_t Small2DChunkOffset(uint32_t ownerIndex)
{
    uint64_t offset = 0;
    for (uint32_t owner = 0; owner < ownerIndex; ++owner) {
        offset += Small2DChunkSize(owner);
    }
    return offset;
}

[[maybe_unused]] CcuResult Small2DTwoShotAllReduce(RootAllReduceContext &ctx)
{
    constexpr uint32_t axis0Size = 4;
    constexpr uint32_t axis1Size = 3;
    // CCU_IF is a generated runtime branch: its C++ body is still visited
    // while every baseline kernel is registered. Do not construct 12-rank
    // channel references for the 4/16-rank baseline contexts.
    if (ctx.arg->rankSize != axis0Size * axis1Size) {
        return CCU_SUCCESS;
    }
    const uint32_t rank = ctx.arg->rankId;
    const uint32_t axis0 = rank % axis0Size;
    const uint32_t axis1 = rank / axis0Size;
    const std::vector<uint32_t> rowPeers = GetSmall2DPeers(ctx, true);
    const std::vector<uint32_t> allPeers = GetSmall2DPeers(ctx, false);
    std::vector<uint32_t> columnPeers;
    for (uint32_t row = 0; row < axis1Size; ++row) {
        const uint32_t peer = row * axis0Size + axis0;
        if (peer != rank) {
            columnPeers.push_back(peer);
        }
    }
    CCU_CHECK_RET(Small2DPreSync(ctx, allPeers));

    // Owner order is axis0-major: (x,0), (x,1), (x,2). This makes the
    // first-axis ReduceScatter region contiguous while keeping one unique
    // final owner for every physical rank.
    const uint32_t regionFirstOwner = axis0 * axis1Size;
    const uint64_t regionOffsetValue = Small2DChunkOffset(regionFirstOwner);
    uint64_t regionSizeValue = 0;
    for (uint32_t y = 0; y < axis1Size; ++y) {
        regionSizeValue += Small2DChunkSize(regionFirstOwner + y);
    }
    ccu::Variable regionOffset;
    regionOffset = regionOffsetValue;
    ccu::Variable regionSize;
    regionSize = regionSizeValue;

    // Step 1: four-way row ReduceScatter. Every source is snapshotted into a
    // disjoint slot, then consumed by a fixed x=0,1,2,3 reduction tree.
    std::vector<ccu::LocalAddr> rowSlots(axis0Size);
    ccu::Variable rowSlotOffset;
    rowSlotOffset = 0;
    for (uint32_t sourceX = 0; sourceX < axis0Size; ++sourceX) {
        rowSlots[sourceX].addr = ctx.scratch;
        rowSlots[sourceX].addr += rowSlotOffset;
        rowSlots[sourceX].token = ctx.scratchToken;
        rowSlotOffset += regionSize;

        const uint32_t sourceRank = axis1 * axis0Size + sourceX;
        const uint32_t eventBit = 1U << sourceX;
        if (sourceRank == rank) {
            ccu::LocalAddr localInput;
            localInput.addr = ctx.input[rank];
            localInput.addr += regionOffset;
            localInput.token = ctx.token[rank];
            CCU_CHECK_RET(ccu::LocalCopy(rowSlots[sourceX], localInput,
                regionSize, ctx.event, eventBit));
        } else {
            ccu::RemoteAddr remoteInput;
            remoteInput.addr = ctx.input[sourceRank];
            remoteInput.addr += regionOffset;
            remoteInput.token = ctx.token[sourceRank];
            CCU_CHECK_RET(ccu::Read(
                ctx.arg->channels[ChannelIndex(rank, sourceRank)],
                rowSlots[sourceX], remoteInput, regionSize, ctx.event, eventBit));
        }
    }
    CCU_CHECK_RET(ccu::EventWait(ctx.event, 0xFU));
    CCU_CHECK_RET(Small2DPeerSync(
        ctx, rowPeers, SMALL_2D_INPUT_READ_DONE_ID));

    ccu::Variable twoRegionSize;
    twoRegionSize = regionSize;
    twoRegionSize += regionSize;
    CCU_CHECK_RET(ccu::LocalReduce(rowSlots[0], rowSlots[2],
        twoRegionSize, ctx.arg->dataType, ctx.arg->reduceOp, ctx.event, 1U));
    CCU_CHECK_RET(ccu::EventWait(ctx.event, 1U));
    CCU_CHECK_RET(ccu::LocalReduce(rowSlots[0], rowSlots[1],
        regionSize, ctx.arg->dataType, ctx.arg->reduceOp, ctx.event, 1U));
    CCU_CHECK_RET(ccu::EventWait(ctx.event, 1U));

    ccu::LocalAddr localRegion;
    localRegion.addr = ctx.output[rank];
    localRegion.addr += regionOffset;
    localRegion.token = ctx.token[rank];
    CCU_CHECK_RET(ccu::LocalCopy(
        localRegion, rowSlots[0], regionSize, ctx.event, 1U));
    CCU_CHECK_RET(ccu::EventWait(ctx.event, 1U));
    CCU_CHECK_RET(Small2DPeerSync(
        ctx, columnPeers, SMALL_2D_PARTIAL_READY_ID));

    // Step 2: three-way column ReduceScatter. Rank (x,y) retains owner
    // x*3+y. Slots are again indexed by physical y and reduced in y order.
    const uint32_t finalOwner = regionFirstOwner + axis1;
    const uint64_t finalOffsetValue = Small2DChunkOffset(finalOwner);
    const uint64_t finalSizeValue = Small2DChunkSize(finalOwner);
    ccu::Variable finalOffset;
    finalOffset = finalOffsetValue;
    ccu::Variable finalSize;
    finalSize = finalSizeValue;
    std::vector<ccu::LocalAddr> columnSlots(axis1Size);
    ccu::Variable columnSlotOffset;
    columnSlotOffset = 0;
    for (uint32_t sourceY = 0; sourceY < axis1Size; ++sourceY) {
        columnSlots[sourceY].addr = ctx.scratch;
        columnSlots[sourceY].addr += columnSlotOffset;
        columnSlots[sourceY].token = ctx.scratchToken;
        columnSlotOffset += finalSize;

        const uint32_t sourceRank = sourceY * axis0Size + axis0;
        const uint32_t eventBit = 1U << sourceY;
        if (sourceRank == rank) {
            ccu::LocalAddr localPartial;
            localPartial.addr = ctx.output[rank];
            localPartial.addr += finalOffset;
            localPartial.token = ctx.token[rank];
            CCU_CHECK_RET(ccu::LocalCopy(columnSlots[sourceY], localPartial,
                finalSize, ctx.event, eventBit));
        } else {
            ccu::RemoteAddr remotePartial;
            remotePartial.addr = ctx.output[sourceRank];
            remotePartial.addr += finalOffset;
            remotePartial.token = ctx.token[sourceRank];
            CCU_CHECK_RET(ccu::Read(
                ctx.arg->channels[ChannelIndex(rank, sourceRank)],
                columnSlots[sourceY], remotePartial, finalSize, ctx.event, eventBit));
        }
    }
    CCU_CHECK_RET(ccu::EventWait(ctx.event, 0x7U));
    CCU_CHECK_RET(Small2DPeerSync(
        ctx, columnPeers, SMALL_2D_PARTIAL_READ_DONE_ID));
    CCU_CHECK_RET(ccu::LocalReduce(columnSlots[0], columnSlots[1],
        finalSize, ctx.arg->dataType, ctx.arg->reduceOp, ctx.event, 1U));
    CCU_CHECK_RET(ccu::EventWait(ctx.event, 1U));
    CCU_CHECK_RET(ccu::LocalReduce(columnSlots[0], columnSlots[2],
        finalSize, ctx.arg->dataType, ctx.arg->reduceOp, ctx.event, 1U));
    CCU_CHECK_RET(ccu::EventWait(ctx.event, 1U));

    ccu::LocalAddr localFinal;
    localFinal.addr = ctx.output[rank];
    localFinal.addr += finalOffset;
    localFinal.token = ctx.token[rank];
    CCU_CHECK_RET(ccu::LocalCopy(
        localFinal, columnSlots[0], finalSize, ctx.event, 1U));
    CCU_CHECK_RET(ccu::EventWait(ctx.event, 1U));

    // Reverse Step 2: gather the three final owners inside each column.
    uint32_t columnWriteBits = 0;
    for (uint32_t peer : columnPeers) {
        const uint32_t eventBit = 1U << (peer / axis0Size);
        ccu::RemoteAddr remoteFinal;
        remoteFinal.addr = ctx.output[peer];
        remoteFinal.addr += finalOffset;
        remoteFinal.token = ctx.token[peer];
        CCU_CHECK_RET(ccu::Write(ctx.arg->channels[ChannelIndex(rank, peer)],
            remoteFinal, localFinal, finalSize, ctx.event, eventBit));
        columnWriteBits |= eventBit;
    }
    CCU_CHECK_RET(ccu::EventWait(ctx.event, columnWriteBits));
    CCU_CHECK_RET(Small2DPeerSync(
        ctx, columnPeers, SMALL_2D_COLUMN_AG_DONE_ID));

    // Reverse Step 1: every x lane now owns its complete three-slice region.
    // Concurrent row writes target non-overlapping regions.
    uint32_t rowWriteBits = 0;
    for (uint32_t peer : rowPeers) {
        const uint32_t eventBit = 1U << (peer % axis0Size);
        ccu::RemoteAddr remoteRegion;
        remoteRegion.addr = ctx.output[peer];
        remoteRegion.addr += regionOffset;
        remoteRegion.token = ctx.token[peer];
        CCU_CHECK_RET(ccu::Write(ctx.arg->channels[ChannelIndex(rank, peer)],
            remoteRegion, localRegion, regionSize, ctx.event, eventBit));
        rowWriteBits |= eventBit;
    }
    CCU_CHECK_RET(ccu::EventWait(ctx.event, rowWriteBits));
    CCU_CHECK_RET(RecursiveDoublingPostSync(ctx, allPeers));
    return CCU_SUCCESS;
}

CcuResult RecursiveDoublingAllReduce(RootAllReduceContext &ctx)
{
    const uint32_t rankSize = ctx.arg->rankSize;
    const uint32_t rank = ctx.arg->rankId;
    const uint32_t powerOfTwo = HighestPowerOfTwo(rankSize);
    const uint32_t remainder = rankSize - powerOfTwo;
    const std::vector<uint32_t> peers = GetRecursiveDoublingPeers(ctx);

    CCU_CHECK_RET(CopyInputToOutput(ctx));
    CCU_CHECK_RET(RecursiveDoublingPreSync(ctx, peers));

    bool active = true;
    uint32_t activeRank = rank;
    uint32_t roundSlotBase = 0;
    if (rank < 2 * remainder) {
        const uint32_t peer = rank ^ 1;
        CCU_CHECK_RET(PublishRecursiveDoublingSnapshot(ctx, 0));
        CCU_CHECK_RET(RecursiveDoublingPairSync(ctx, peer));
        if ((rank & 1) == 0) {
            active = false;
        } else {
            CCU_CHECK_RET(ReduceRecursiveDoublingSnapshot(ctx, peer, 0));
            activeRank = rank / 2;
        }
        roundSlotBase = 1;
    } else {
        activeRank = rank - remainder;
        roundSlotBase = remainder == 0 ? 0 : 1;
    }

    if (active) {
        uint32_t round = 0;
        for (uint32_t mask = 1; mask < powerOfTwo; mask <<= 1, ++round) {
            const uint32_t peerActiveRank = activeRank ^ mask;
            const uint32_t peer = ActiveRankToPhysical(peerActiveRank, remainder);
            const uint32_t slot = roundSlotBase + round;
            CCU_CHECK_RET(PublishRecursiveDoublingSnapshot(ctx, slot));
            CCU_CHECK_RET(RecursiveDoublingPairSync(ctx, peer));
            CCU_CHECK_RET(ReduceRecursiveDoublingSnapshot(ctx, peer, slot));
        }
    }

    if (rank < 2 * remainder) {
        const uint32_t peer = rank ^ 1;
        constexpr uint32_t finalSlot = 4;
        if ((rank & 1) != 0) {
            CCU_CHECK_RET(PublishRecursiveDoublingSnapshot(ctx, finalSlot));
            CCU_CHECK_RET(RecursiveDoublingPairSync(ctx, peer));
        } else {
            CCU_CHECK_RET(RecursiveDoublingPairSync(ctx, peer));
            CCU_CHECK_RET(ReadRecursiveDoublingSnapshot(ctx, peer, finalSlot));
        }
    }

    CCU_CHECK_RET(RecursiveDoublingPostSync(ctx, peers));
    return CCU_SUCCESS;
}

CcuResult PostSync(RootAllReduceContext &ctx)
{
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        CCU_CHECK_RET(ccu::NotifyRecord(
            ctx.arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        CCU_CHECK_RET(ccu::NotifyWait(
            ctx.arg->channels[i], CKE_IDX_0, 1 << POST_SYNC_ID));
    }
    return CCU_SUCCESS;
}

CcuResult ParallelMeshReduceScatter(RootAllReduceContext &ctx, bool sourceOutput)
{
    std::vector<ccu::LocalAddr> scratchSlots(ctx.arg->rankSize);
    ccu::Variable scratchOffset;
    scratchOffset = 0;
    for (uint32_t rank = 0; rank < ctx.arg->rankSize; ++rank) {
        scratchSlots[rank].addr = ctx.scratch;
        scratchSlots[rank].addr += scratchOffset;
        scratchSlots[rank].token = ctx.scratchToken;
        scratchOffset += ctx.ownerSize;
    }

    for (uint32_t rank = 0; rank < ctx.arg->rankSize; ++rank) {
        const uint32_t eventBit = 1U << rank;
        if (rank == ctx.arg->rankId) {
            ccu::LocalAddr src;
            src.addr = sourceOutput ? ctx.output[rank] : ctx.input[rank];
            src.addr += ctx.ownerOffset;
            src.token = ctx.token[rank];
            CCU_CHECK_RET(ccu::LocalCopy(
                scratchSlots[rank], src, ctx.ownerSize, ctx.event, eventBit));
        } else {
            const uint32_t channel = ChannelIndex(ctx.arg->rankId, rank);
            ccu::RemoteAddr src;
            src.addr = sourceOutput ? ctx.output[rank] : ctx.input[rank];
            src.addr += ctx.ownerOffset;
            src.token = ctx.token[rank];
            CCU_CHECK_RET(ccu::Read(ctx.arg->channels[channel], scratchSlots[rank],
                src, ctx.ownerSize, ctx.event, eventBit));
        }
    }
    CCU_CHECK_RET(ccu::EventWait(ctx.event, (1U << ctx.arg->rankSize) - 1U));

    // Every local input region must remain readable until all eight owners
    // have taken their snapshot.  This fixed barrier also closes the checker
    // dependency before any owner writes Output.
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        CCU_CHECK_RET(ccu::NotifyRecord(
            ctx.arg->channels[i], CKE_IDX_0, 1U << HIER_READ_READY_ID));
    }
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        CCU_CHECK_RET(ccu::NotifyWait(
            ctx.arg->channels[i], CKE_IDX_0, 1U << HIER_READ_READY_ID));
    }

    uint32_t remaining = ctx.arg->rankSize;
    while (remaining > 1) {
        const uint32_t reducePieces = remaining / 2;
        const uint32_t srcIndex = remaining - reducePieces;
        ccu::Variable reduceBytes;
        reduceBytes = 0;
        for (uint32_t i = 0; i < reducePieces; ++i) {
            reduceBytes += ctx.ownerSize;
        }
        CCU_CHECK_RET(ccu::LocalReduce(scratchSlots[0], scratchSlots[srcIndex],
            reduceBytes, ctx.arg->dataType, ctx.arg->reduceOp, ctx.event, 1));
        CCU_CHECK_RET(ccu::EventWait(ctx.event, 1));
        remaining -= reducePieces;
    }

    ccu::LocalAddr dst;
    dst.addr = ctx.output[ctx.arg->rankId];
    dst.addr += ctx.ownerOffset;
    dst.token = ctx.token[ctx.arg->rankId];
    CCU_CHECK_RET(ccu::LocalCopy(
        dst, scratchSlots[0], ctx.ownerSize, ctx.event, 1));
    CCU_CHECK_RET(ccu::EventWait(ctx.event, 1));
    return CCU_SUCCESS;
}

CcuResult ParallelMeshAllGather(RootAllReduceContext &ctx)
{
    ccu::LocalAddr src;
    src.addr = ctx.output[ctx.arg->rankId];
    src.addr += ctx.ownerOffset;
    src.token = ctx.token[ctx.arg->rankId];
    for (uint32_t peer = 0; peer < ctx.arg->rankSize; ++peer) {
        const uint32_t bit = 1U << peer;
        if (peer == ctx.arg->rankId) {
            CCU_CHECK_RET(ccu::EventRecord(ctx.event, bit));
            continue;
        }
        const uint32_t channel = ChannelIndex(ctx.arg->rankId, peer);
        ccu::RemoteAddr dst;
        dst.addr = ctx.output[peer];
        dst.addr += ctx.ownerOffset;
        dst.token = ctx.token[peer];
        CCU_CHECK_RET(ccu::Write(
            ctx.arg->channels[channel], dst, src, ctx.ownerSize, ctx.event, bit));
    }
    CCU_CHECK_RET(ccu::EventWait(ctx.event, (1U << ctx.arg->rankSize) - 1U));
    return CCU_SUCCESS;
}

CcuResult ParallelClosReduceScatter(RootAllReduceContext &ctx, bool sourceOutput)
{
    const uint32_t peer = ctx.arg->rankId ^ 1U;
    constexpr uint32_t channel = 0;
    const uint32_t keepOwner = ctx.arg->rankId;
    const uint32_t sendOwner = keepOwner ^ 1U;

    ccu::Variable keepOffset;
    keepOffset = 0;
    if (keepOwner != 0) {
        keepOffset += ctx.ownerSize;
    }
    ccu::Variable sendOffset;
    sendOffset = 0;
    if (sendOwner != 0) {
        sendOffset += ctx.ownerSize;
    }

    if (!sourceOutput) {
        ccu::LocalAddr keepSrc;
        keepSrc.addr = ctx.input[ctx.arg->rankId];
        keepSrc.addr += keepOffset;
        keepSrc.token = ctx.token[ctx.arg->rankId];
        ccu::LocalAddr keepDst;
        keepDst.addr = ctx.output[ctx.arg->rankId];
        keepDst.addr += keepOffset;
        keepDst.token = ctx.token[ctx.arg->rankId];
        CCU_CHECK_RET(ccu::LocalCopy(
            keepDst, keepSrc, ctx.ownerSize, ctx.event, 1));
        CCU_CHECK_RET(ccu::EventWait(ctx.event, 1));
    }

    // The peer must initialize its retained half before the incoming
    // WriteReduce.  Rank order and the local-then-remote add order are fixed.
    CCU_CHECK_RET(ccu::NotifyRecord(
        ctx.arg->channels[channel], CKE_IDX_0, 1U << HIER_READ_READY_ID));
    CCU_CHECK_RET(ccu::NotifyWait(
        ctx.arg->channels[channel], CKE_IDX_0, 1U << HIER_READ_READY_ID));

    ccu::LocalAddr sendSrc;
    sendSrc.addr = sourceOutput ?
        ctx.output[ctx.arg->rankId] : ctx.input[ctx.arg->rankId];
    sendSrc.addr += sendOffset;
    sendSrc.token = ctx.token[ctx.arg->rankId];
    ccu::RemoteAddr remoteKeep;
    remoteKeep.addr = ctx.output[peer];
    remoteKeep.addr += sendOffset;
    remoteKeep.token = ctx.token[peer];
    CCU_CHECK_RET(ccu::WriteReduce(ctx.arg->channels[channel], remoteKeep,
        sendSrc, ctx.ownerSize, ctx.arg->dataType, ctx.arg->reduceOp,
        ctx.event, 1));
    CCU_CHECK_RET(ccu::EventWait(ctx.event, 1));
    return CCU_SUCCESS;
}

CcuResult ParallelClosAllGather(RootAllReduceContext &ctx)
{
    const uint32_t peer = ctx.arg->rankId ^ 1U;
    constexpr uint32_t channel = 0;
    ccu::Variable keepOffset;
    keepOffset = 0;
    if (ctx.arg->rankId != 0) {
        keepOffset += ctx.ownerSize;
    }

    ccu::LocalAddr src;
    src.addr = ctx.output[ctx.arg->rankId];
    src.addr += keepOffset;
    src.token = ctx.token[ctx.arg->rankId];
    ccu::RemoteAddr dst;
    dst.addr = ctx.output[peer];
    dst.addr += keepOffset;
    dst.token = ctx.token[peer];
    CCU_CHECK_RET(ccu::Write(
        ctx.arg->channels[channel], dst, src, ctx.ownerSize, ctx.event, 1));
    CCU_CHECK_RET(ccu::EventWait(ctx.event, 1));
    return CCU_SUCCESS;
}

} // namespace

CcuResult CcuKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllReduce *>(arg);
    RootAllReduceContext ctx;
    ctx.arg = kernelArg;

    CCU_CHECK_RET(InitResources(ctx));
    CCU_CHECK_RET(LoadArgs(ctx));
    CCU_IF(ctx.algorithmType == ALGORITHM_ROOT)
    {
        CCU_CHECK_RET(CopyInputToOutput(ctx));
        CCU_CHECK_RET(RootPreSync(ctx));
        CCU_CHECK_RET(ReduceAndBroadcast(ctx));
        CCU_CHECK_RET(PostSync(ctx));
    }
    CCU_IF(ctx.algorithmType == ALGORITHM_FULL_MESH)
    {
        CCU_CHECK_RET(FullMeshPreSync(ctx));
        CCU_CHECK_RET(CopyOwnerInputToOutput(ctx));
        CCU_CHECK_RET(FullMeshReduceScatterAndAllGather(ctx));
        CCU_CHECK_RET(PostSync(ctx));
    }
    CCU_IF(ctx.algorithmType == ALGORITHM_RECURSIVE_DOUBLING)
    {
        CCU_CHECK_RET(RecursiveDoublingAllReduce(ctx));
    }
    CCU_IF(ctx.algorithmType == ALGORITHM_SMALL_2D_BEST)
    {
        CCU_CHECK_RET(Small2DBestAllReduce(ctx));
    }
    return CCU_SUCCESS;
}

// Pre-compute 16 slice offsets at compile-time-unrolled loop count.
// Slice k is at offset k * ownerSize in the linear output buffer.
static void PrecomputeSliceOffsets(ccu::Variable (&offsets)[16],
    const ccu::Variable &ownerSize)
{
    offsets[0] = 0;
    offsets[1] = offsets[0] + ownerSize;
    offsets[2] = offsets[1] + ownerSize;
    offsets[3] = offsets[2] + ownerSize;
    offsets[4] = offsets[3] + ownerSize;
    offsets[5] = offsets[4] + ownerSize;
    offsets[6] = offsets[5] + ownerSize;
    offsets[7] = offsets[6] + ownerSize;
    offsets[8] = offsets[7] + ownerSize;
    offsets[9] = offsets[8] + ownerSize;
    offsets[10] = offsets[9] + ownerSize;
    offsets[11] = offsets[10] + ownerSize;
    offsets[12] = offsets[11] + ownerSize;
    offsets[13] = offsets[12] + ownerSize;
    offsets[14] = offsets[13] + ownerSize;
    offsets[15] = offsets[14] + ownerSize;
}

// The first recursive-halving round keeps only one half of the local slice.
// Initialize that retained half in output; the discarded half is sent
// directly from input in round 1.  Later rounds operate only on the retained,
// already-reduced output region.  This removes half of the NHR pre-copy bytes
// without changing network traffic or the floating-point reduction order.
CcuResult NhrInitializeRetainedHalf(RootAllReduceContext &ctx)
{
    const uint32_t retainedOwner = (ctx.arg->rankId & 8U) == 0 ? 0U : 8U;
    ccu::Variable retainedOffset;
    retainedOffset = 0;
    for (uint32_t owner = 0; owner < retainedOwner; ++owner) {
        retainedOffset += ctx.ownerSize;
    }

    ccu::LocalAddr dst;
    dst.addr = ctx.output[ctx.arg->rankId];
    dst.addr += retainedOffset;
    dst.token = ctx.token[ctx.arg->rankId];
    ccu::LocalAddr src;
    src.addr = ctx.input[ctx.arg->rankId];
    src.addr += retainedOffset;
    src.token = ctx.token[ctx.arg->rankId];

    CCU_IF(ctx.isInPlace == 0)
    {
        CCU_CHECK_RET(ccu::LocalCopy(dst, src, ctx.pipelineTileSize, ctx.event, 1));
    }
    CCU_IF(ctx.isInPlace != 0)
    {
        CCU_CHECK_RET(ccu::EventRecord(ctx.event, 1));
    }
    CCU_CHECK_RET(ccu::EventWait(ctx.event, 1));
    return CCU_SUCCESS;
}

CcuResult CcuMesh16ScratchKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllReduce *>(arg);
    if (kernelArg->rankSize != 16) {
        HCCL_ERROR("[CcuMesh16ScratchKernel] expected 16 ranks, got %u", kernelArg->rankSize);
        return CCU_E_PARA;
    }

    RootAllReduceContext ctx;
    ctx.arg = kernelArg;
    CCU_CHECK_RET(InitResources(ctx));
    CCU_CHECK_RET(LoadArgs(ctx));

    ccu::Variable nhrSliceOffset[16];
    CCU_IF(ctx.algorithmType == ALGORITHM_NHR_WRITE_REDUCE)
    {
        CCU_CHECK_RET(ccu::LoadArg(ctx.pipelineTileSize, 10));
        CCU_CHECK_RET(ccu::LoadArg(ctx.pipelineTailSize, 11));
        CCU_CHECK_RET(ccu::LoadArg(ctx.fourGlobalSliceSize, 12));
        PrecomputeSliceOffsets(nhrSliceOffset, ctx.ownerSize);
        CCU_CHECK_RET(NhrInitializeRetainedHalf(ctx));
    }

    CCU_CHECK_RET(FullMeshPreSync(ctx));
    CCU_IF(ctx.algorithmType == ALGORITHM_FULL_MESH)
    {
        CCU_CHECK_RET(Mesh16ScratchReduceScatterAndAllGather(ctx));
    }
    CCU_IF(ctx.algorithmType == ALGORITHM_NHR_WRITE_REDUCE)
    {
        const uint32_t rid = ctx.arg->rankId;

        // RS round 1: XOR 8, send 8 owners.
        {
            constexpr uint32_t mask = 8;
            const uint32_t peer = rid ^ mask;
            const uint32_t channel = ChannelIndex(rid, peer);
            const uint32_t sendOwner = (rid & mask) == 0 ? mask : 0;
            ccu::LocalAddr src;
            // This half is discarded locally after round 1, so read it
            // directly from input instead of pre-copying it to output.
            src.addr = ctx.input[rid];
            src.addr += nhrSliceOffset[sendOwner];
            src.token = ctx.token[rid];
            ccu::RemoteAddr dst;
            dst.addr = ctx.output[peer];
            dst.addr += nhrSliceOffset[sendOwner];
            dst.token = ctx.token[peer];
            CCU_CHECK_RET(ccu::WriteReduce(ctx.arg->channels[channel], dst, src,
                ctx.pipelineTileSize, ctx.arg->dataType, ctx.arg->reduceOp, ctx.event, 1));
            CCU_CHECK_RET(ccu::EventWait(ctx.event, 1));
            CCU_CHECK_RET(ccu::NotifyRecord(ctx.arg->channels[channel], CKE_IDX_0,
                1U << XOR_RS1_DONE_BIT));
            CCU_CHECK_RET(ccu::NotifyWait(ctx.arg->channels[channel], CKE_IDX_0,
                1U << XOR_RS1_DONE_BIT));
        }

        // RS round 2: XOR 4, send 4 owners.
        {
            constexpr uint32_t mask = 4;
            const uint32_t peer = rid ^ mask;
            const uint32_t channel = ChannelIndex(rid, peer);
            const uint32_t groupBase = rid & ~(2U * mask - 1U);
            const uint32_t sendOwner = groupBase + (((rid & mask) == 0) ? mask : 0);
            CCU_CHECK_RET(ccu::NotifyRecord(ctx.arg->channels[channel], CKE_IDX_0,
                1U << XOR_RS2_READY_BIT));
            CCU_CHECK_RET(ccu::NotifyWait(ctx.arg->channels[channel], CKE_IDX_0,
                1U << XOR_RS2_READY_BIT));
            ccu::LocalAddr src;
            src.addr = ctx.output[rid];
            src.addr += nhrSliceOffset[sendOwner];
            src.token = ctx.token[rid];
            ccu::RemoteAddr dst;
            dst.addr = ctx.output[peer];
            dst.addr += nhrSliceOffset[sendOwner];
            dst.token = ctx.token[peer];
            CCU_CHECK_RET(ccu::WriteReduce(ctx.arg->channels[channel], dst, src,
                ctx.pipelineTailSize, ctx.arg->dataType, ctx.arg->reduceOp, ctx.event, 1));
            CCU_CHECK_RET(ccu::EventWait(ctx.event, 1));
            CCU_CHECK_RET(ccu::NotifyRecord(ctx.arg->channels[channel], CKE_IDX_0,
                1U << XOR_RS2_DONE_BIT));
            CCU_CHECK_RET(ccu::NotifyWait(ctx.arg->channels[channel], CKE_IDX_0,
                1U << XOR_RS2_DONE_BIT));
        }

        // RS round 3: XOR 2, send 2 owners.
        {
            constexpr uint32_t mask = 2;
            const uint32_t peer = rid ^ mask;
            const uint32_t channel = ChannelIndex(rid, peer);
            const uint32_t groupBase = rid & ~(2U * mask - 1U);
            const uint32_t sendOwner = groupBase + (((rid & mask) == 0) ? mask : 0);
            CCU_CHECK_RET(ccu::NotifyRecord(ctx.arg->channels[channel], CKE_IDX_0,
                1U << XOR_RS3_READY_BIT));
            CCU_CHECK_RET(ccu::NotifyWait(ctx.arg->channels[channel], CKE_IDX_0,
                1U << XOR_RS3_READY_BIT));
            ccu::LocalAddr src;
            src.addr = ctx.output[rid];
            src.addr += nhrSliceOffset[sendOwner];
            src.token = ctx.token[rid];
            ccu::RemoteAddr dst;
            dst.addr = ctx.output[peer];
            dst.addr += nhrSliceOffset[sendOwner];
            dst.token = ctx.token[peer];
            CCU_CHECK_RET(ccu::WriteReduce(ctx.arg->channels[channel], dst, src,
                ctx.fourGlobalSliceSize, ctx.arg->dataType, ctx.arg->reduceOp, ctx.event, 1));
            CCU_CHECK_RET(ccu::EventWait(ctx.event, 1));
            CCU_CHECK_RET(ccu::NotifyRecord(ctx.arg->channels[channel], CKE_IDX_0,
                1U << XOR_RS3_DONE_BIT));
            CCU_CHECK_RET(ccu::NotifyWait(ctx.arg->channels[channel], CKE_IDX_0,
                1U << XOR_RS3_DONE_BIT));
        }

        // RS round 4: XOR 1, send one owner.
        {
            constexpr uint32_t mask = 1;
            const uint32_t peer = rid ^ mask;
            const uint32_t channel = ChannelIndex(rid, peer);
            const uint32_t groupBase = rid & ~(2U * mask - 1U);
            const uint32_t sendOwner = groupBase + (((rid & mask) == 0) ? mask : 0);
            CCU_CHECK_RET(ccu::NotifyRecord(ctx.arg->channels[channel], CKE_IDX_0,
                1U << XOR_RS4_READY_BIT));
            CCU_CHECK_RET(ccu::NotifyWait(ctx.arg->channels[channel], CKE_IDX_0,
                1U << XOR_RS4_READY_BIT));
            ccu::LocalAddr src;
            src.addr = ctx.output[rid];
            src.addr += nhrSliceOffset[sendOwner];
            src.token = ctx.token[rid];
            ccu::RemoteAddr dst;
            dst.addr = ctx.output[peer];
            dst.addr += nhrSliceOffset[sendOwner];
            dst.token = ctx.token[peer];
            CCU_CHECK_RET(ccu::WriteReduce(ctx.arg->channels[channel], dst, src,
                ctx.ownerSize, ctx.arg->dataType, ctx.arg->reduceOp, ctx.event, 1));
            CCU_CHECK_RET(ccu::EventWait(ctx.event, 1));
            CCU_CHECK_RET(ccu::NotifyRecord(ctx.arg->channels[channel], CKE_IDX_0,
                1U << XOR_RS4_DONE_BIT));
            CCU_CHECK_RET(ccu::NotifyWait(ctx.arg->channels[channel], CKE_IDX_0,
                1U << XOR_RS4_DONE_BIT));
        }

        // Reverse doubling AG.
        {
            constexpr uint32_t mask = 1;
            const uint32_t peer = rid ^ mask;
            const uint32_t channel = ChannelIndex(rid, peer);
            const uint32_t sendOwner = rid;
            ccu::LocalAddr src;
            src.addr = ctx.output[rid];
            src.addr += nhrSliceOffset[sendOwner];
            src.token = ctx.token[rid];
            ccu::RemoteAddr dst;
            dst.addr = ctx.output[peer];
            dst.addr += nhrSliceOffset[sendOwner];
            dst.token = ctx.token[peer];
            CCU_CHECK_RET(ccu::Write(ctx.arg->channels[channel], dst, src,
                ctx.ownerSize, ctx.event, 1));
            CCU_CHECK_RET(ccu::EventWait(ctx.event, 1));
            CCU_CHECK_RET(ccu::NotifyRecord(ctx.arg->channels[channel], CKE_IDX_0,
                1U << XOR_AG1_DONE_BIT));
            CCU_CHECK_RET(ccu::NotifyWait(ctx.arg->channels[channel], CKE_IDX_0,
                1U << XOR_AG1_DONE_BIT));
        }
        {
            constexpr uint32_t mask = 2;
            const uint32_t peer = rid ^ mask;
            const uint32_t channel = ChannelIndex(rid, peer);
            const uint32_t sendOwner = rid & ~(mask - 1U);
            ccu::LocalAddr src;
            src.addr = ctx.output[rid];
            src.addr += nhrSliceOffset[sendOwner];
            src.token = ctx.token[rid];
            ccu::RemoteAddr dst;
            dst.addr = ctx.output[peer];
            dst.addr += nhrSliceOffset[sendOwner];
            dst.token = ctx.token[peer];
            CCU_CHECK_RET(ccu::Write(ctx.arg->channels[channel], dst, src,
                ctx.fourGlobalSliceSize, ctx.event, 1));
            CCU_CHECK_RET(ccu::EventWait(ctx.event, 1));
            CCU_CHECK_RET(ccu::NotifyRecord(ctx.arg->channels[channel], CKE_IDX_0,
                1U << XOR_AG2_DONE_BIT));
            CCU_CHECK_RET(ccu::NotifyWait(ctx.arg->channels[channel], CKE_IDX_0,
                1U << XOR_AG2_DONE_BIT));
        }
        {
            constexpr uint32_t mask = 4;
            const uint32_t peer = rid ^ mask;
            const uint32_t channel = ChannelIndex(rid, peer);
            const uint32_t sendOwner = rid & ~(mask - 1U);
            ccu::LocalAddr src;
            src.addr = ctx.output[rid];
            src.addr += nhrSliceOffset[sendOwner];
            src.token = ctx.token[rid];
            ccu::RemoteAddr dst;
            dst.addr = ctx.output[peer];
            dst.addr += nhrSliceOffset[sendOwner];
            dst.token = ctx.token[peer];
            CCU_CHECK_RET(ccu::Write(ctx.arg->channels[channel], dst, src,
                ctx.pipelineTailSize, ctx.event, 1));
            CCU_CHECK_RET(ccu::EventWait(ctx.event, 1));
            CCU_CHECK_RET(ccu::NotifyRecord(ctx.arg->channels[channel], CKE_IDX_0,
                1U << XOR_AG3_DONE_BIT));
            CCU_CHECK_RET(ccu::NotifyWait(ctx.arg->channels[channel], CKE_IDX_0,
                1U << XOR_AG3_DONE_BIT));
        }
        {
            constexpr uint32_t mask = 8;
            const uint32_t peer = rid ^ mask;
            const uint32_t channel = ChannelIndex(rid, peer);
            const uint32_t sendOwner = rid & ~(mask - 1U);
            ccu::LocalAddr src;
            src.addr = ctx.output[rid];
            src.addr += nhrSliceOffset[sendOwner];
            src.token = ctx.token[rid];
            ccu::RemoteAddr dst;
            dst.addr = ctx.output[peer];
            dst.addr += nhrSliceOffset[sendOwner];
            dst.token = ctx.token[peer];
            CCU_CHECK_RET(ccu::Write(ctx.arg->channels[channel], dst, src,
                ctx.pipelineTileSize, ctx.event, 1));
            CCU_CHECK_RET(ccu::EventWait(ctx.event, 1));
        }
    }
    CCU_CHECK_RET(PostSync(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuMesh12ScratchKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllReduce *>(arg);
    if (kernelArg->rankSize != 12) {
        HCCL_ERROR("[CcuMesh12ScratchKernel] expected 12 ranks, got %u", kernelArg->rankSize);
        return CCU_E_PARA;
    }

    RootAllReduceContext ctx;
    ctx.arg = kernelArg;
    CCU_CHECK_RET(InitResources(ctx));
    CCU_CHECK_RET(LoadArgs(ctx));
    CCU_CHECK_RET(FullMeshPreSync(ctx));
    CCU_IF(ctx.algorithmType == ALGORITHM_FULL_MESH)
    {
        CCU_CHECK_RET(Mesh12ScratchReduceScatterAndAllGather(ctx));
    }
    CCU_IF(ctx.algorithmType == ALGORITHM_MIXED_RADIX_12)
    {
        CCU_CHECK_RET(MixedRadix12AllReduce(ctx));
    }
    CCU_CHECK_RET(PostSync(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuMesh4ScratchKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllReduce *>(arg);
    if (kernelArg->rankSize != 4) {
        HCCL_ERROR("[CcuMesh4ScratchKernel] expected 4 ranks, got %u", kernelArg->rankSize);
        return CCU_E_PARA;
    }

    RootAllReduceContext ctx;
    ctx.arg = kernelArg;
    CCU_CHECK_RET(InitResources(ctx));
    CCU_CHECK_RET(LoadArgs(ctx));
    CCU_CHECK_RET(FullMeshPreSync(ctx));
    CCU_CHECK_RET(Mesh16ScratchReduceScatterAndAllGather(ctx));
    CCU_CHECK_RET(PostSync(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuParallel2x8MeshKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllReduce *>(arg);
    if (kernelArg->rankSize != 8 || kernelArg->channelCount != 7) {
        HCCL_ERROR("[CcuParallel2x8MeshKernel] invalid local group: ranks=%u channels=%u",
            kernelArg->rankSize, kernelArg->channelCount);
        return CCU_E_PARA;
    }
    RootAllReduceContext ctx;
    ctx.arg = kernelArg;
    CCU_CHECK_RET(InitResources(ctx));
    CCU_CHECK_RET(LoadArgs(ctx));
    CCU_CHECK_RET(FullMeshPreSync(ctx));
    CCU_IF(ctx.algorithmType == ALGORITHM_PAR_MESH_RS_INPUT)
    {
        CCU_CHECK_RET(ParallelMeshReduceScatter(ctx, false));
    }
    CCU_IF(ctx.algorithmType == ALGORITHM_PAR_MESH_RS_OUTPUT)
    {
        CCU_CHECK_RET(ParallelMeshReduceScatter(ctx, true));
    }
    CCU_IF(ctx.algorithmType == ALGORITHM_PAR_MESH_AG)
    {
        CCU_CHECK_RET(ParallelMeshAllGather(ctx));
    }
    CCU_IF(ctx.algorithmType == ALGORITHM_PAR_MESH_RS_AG)
    {
        // Every rank only publishes its own completed owner and incoming AG
        // blocks target disjoint owner ranges, so no group barrier is needed
        // between the local RS completion and this rank's AG writes.
        CCU_CHECK_RET(ParallelMeshReduceScatter(ctx, true));
        CCU_CHECK_RET(ParallelMeshAllGather(ctx));
    }
    CCU_CHECK_RET(PostSync(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuParallel2x8ClosKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllReduce *>(arg);
    if (kernelArg->rankSize != 2 || kernelArg->rankId >= 2 ||
        kernelArg->channelCount != 1) {
        HCCL_ERROR("[CcuParallel2x8ClosKernel] invalid pair: rank=%u channels=%u",
            kernelArg->rankId, kernelArg->channelCount);
        return CCU_E_PARA;
    }
    RootAllReduceContext ctx;
    ctx.arg = kernelArg;
    CCU_CHECK_RET(InitResources(ctx));
    CCU_CHECK_RET(LoadArgs(ctx));
    CCU_CHECK_RET(FullMeshPreSync(ctx));
    CCU_IF(ctx.algorithmType == ALGORITHM_PAR_CLOS_RS)
    {
        CCU_CHECK_RET(ParallelClosReduceScatter(ctx, false));
    }
    CCU_IF(ctx.algorithmType == ALGORITHM_PAR_CLOS_RS_OUTPUT)
    {
        CCU_CHECK_RET(ParallelClosReduceScatter(ctx, true));
    }
    CCU_IF(ctx.algorithmType == ALGORITHM_PAR_CLOS_AG)
    {
        CCU_CHECK_RET(ParallelClosAllGather(ctx));
    }
    CCU_IF(ctx.algorithmType == ALGORITHM_PAR_CLOS_RS_AG)
    {
        CCU_CHECK_RET(ParallelClosReduceScatter(ctx, true));
        // EventWait proves only completion of this rank's outgoing
        // WriteReduce. Exchange a pair-done bit before using the owner that
        // the opposite rank reduced into.
        CCU_CHECK_RET(ccu::NotifyRecord(ctx.arg->channels[0], CKE_IDX_0,
            1U << HIER_FUSED_RS_DONE_ID));
        CCU_CHECK_RET(ccu::NotifyWait(ctx.arg->channels[0], CKE_IDX_0,
            1U << HIER_FUSED_RS_DONE_ID));
        CCU_CHECK_RET(ParallelClosAllGather(ctx));
    }
    CCU_CHECK_RET(PostSync(ctx));
    return CCU_SUCCESS;
}

} // namespace ops_hccl