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

#include <hcomm/hcomm_primitives.h>
#include <ccu/ccu_primitives.hpp>

#include "ccu_kernel.h"
#include "custom.h"

namespace ccu = ::AscendC::ccu;

namespace ops_hccl {
namespace {

constexpr uint32_t KERNEL_MAX_RANK_SIZE = 16;
constexpr uint32_t INPUT_ADDR_XN_ID = 0;
constexpr uint32_t INPUT_TOKEN_XN_ID = 1;
constexpr uint32_t CHANNEL_NOTIFY_INDEX = 0;

constexpr uint16_t INPUT_ADDR_READY_MASK = static_cast<uint16_t>(1U << INPUT_ADDR_XN_ID);
constexpr uint16_t INPUT_TOKEN_READY_MASK = static_cast<uint16_t>(1U << INPUT_TOKEN_XN_ID);
constexpr uint16_t PRE_SYNC_MASK =
    static_cast<uint16_t>(INPUT_ADDR_READY_MASK | INPUT_TOKEN_READY_MASK);
constexpr uint16_t POST_SYNC_MASK = static_cast<uint16_t>(1U << 2U);

constexpr uint32_t TREE_STRIDES[] = {1U, 2U, 4U, 8U};

static_assert(MAX_RANK_SIZE == KERNEL_MAX_RANK_SIZE, "CCU kernel and host rank limits must match");

#define OPS_CCU_RETURN_IF_ERROR(call) \
    do { \
        const CcuResult result = (call); \
        if (result != CCU_SUCCESS) { \
            return result; \
        } \
    } while (false)

struct ReduceScatterContext {
    const CcuReduceScatterKernelArg *arg = nullptr;

    ccu::Variable inputAddr[KERNEL_MAX_RANK_SIZE];
    ccu::Variable inputToken[KERNEL_MAX_RANK_SIZE];
    ccu::Variable outputAddr;
    ccu::Variable outputToken;
    ccu::Variable scratchAddr;
    ccu::Variable scratchToken;
    ccu::Variable currentInputOffset;
    ccu::Variable currentOutputOffset;
    ccu::Variable inputRepeatStride;
    ccu::Variable outputRepeatStride;
    ccu::Variable sliceSize;
    ccu::Variable repeatNum;

    ccu::LocalAddr localInput;
    ccu::RemoteAddr remoteInput[KERNEL_MAX_RANK_SIZE];
    ccu::LocalAddr scratch[KERNEL_MAX_RANK_SIZE];
    ccu::LocalAddr output;
    ccu::Event event;
};

constexpr uint16_t EventBit(uint32_t bit)
{
    return static_cast<uint16_t>(uint32_t{1} << bit);
}

constexpr uint16_t AllRankEventMask(uint32_t rankSize)
{
    // The shift is deliberately performed in 32 bits: rankSize == 16 must
    // produce 0xffff instead of overflowing a uint16_t intermediate.
    return static_cast<uint16_t>((uint32_t{1} << rankSize) - uint32_t{1});
}

CcuResult InitResource(ReduceScatterContext &ctx)
{
    const CcuReduceScatterKernelArg *arg = ctx.arg;
    if (arg == nullptr || arg->rankSize < 2 || arg->rankSize > KERNEL_MAX_RANK_SIZE
        || arg->rankId >= arg->rankSize || arg->channelCount != arg->rankSize - 1) {
        return CCU_E_PARA;
    }

    uint16_t peerMask = EventBit(arg->rankId);
    for (uint32_t channelIndex = 0; channelIndex < arg->channelCount; ++channelIndex) {
        const uint32_t peerRank = arg->peerRanks[channelIndex];
        if (peerRank >= arg->rankSize || peerRank == arg->rankId) {
            return CCU_E_PARA;
        }
        const uint16_t peerBit = EventBit(peerRank);
        if ((peerMask & peerBit) != 0) {
            return CCU_E_PARA;
        }
        peerMask = static_cast<uint16_t>(peerMask | peerBit);

        const ChannelHandle channel = arg->channels[channelIndex];
        ctx.inputAddr[peerRank] = ccu::GetResByChannel<ccu::Variable>(channel, INPUT_ADDR_XN_ID);
        ctx.inputToken[peerRank] = ccu::GetResByChannel<ccu::Variable>(channel, INPUT_TOKEN_XN_ID);
    }
    if (peerMask != AllRankEventMask(static_cast<uint32_t>(arg->rankSize))) {
        return CCU_E_PARA;
    }
    return CCU_SUCCESS;
}

CcuResult LoadArgs(ReduceScatterContext &ctx)
{
    const uint32_t selfRank = ctx.arg->rankId;
    uint32_t argId = 0;

    // Keep this order exactly aligned with ExecOp's twelve task arguments.
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.inputAddr[selfRank], argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.outputAddr, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.inputToken[selfRank], argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.outputToken, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.scratchAddr, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.scratchToken, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.currentInputOffset, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.currentOutputOffset, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.inputRepeatStride, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.outputRepeatStride, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.sliceSize, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.repeatNum, argId++));
    return CCU_SUCCESS;
}

CcuResult PreSync(const ReduceScatterContext &ctx)
{
    const CcuReduceScatterKernelArg *arg = ctx.arg;
    const uint32_t selfRank = arg->rankId;

    for (uint32_t channelIndex = 0; channelIndex < arg->channelCount; ++channelIndex) {
        const ChannelHandle channel = arg->channels[channelIndex];
        OPS_CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(channel, ctx.inputAddr[selfRank],
            INPUT_ADDR_XN_ID, CHANNEL_NOTIFY_INDEX, INPUT_ADDR_READY_MASK));
        OPS_CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(channel, ctx.inputToken[selfRank],
            INPUT_TOKEN_XN_ID, CHANNEL_NOTIFY_INDEX, INPUT_TOKEN_READY_MASK));
    }

    for (uint32_t channelIndex = 0; channelIndex < arg->channelCount; ++channelIndex) {
        OPS_CCU_RETURN_IF_ERROR(
            ccu::NotifyWait(arg->channels[channelIndex], CHANNEL_NOTIFY_INDEX, PRE_SYNC_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult InitAddresses(ReduceScatterContext &ctx)
{
    const CcuReduceScatterKernelArg *arg = ctx.arg;
    const uint32_t rankSize = static_cast<uint32_t>(arg->rankSize);

    ctx.localInput.addr = ctx.inputAddr[arg->rankId];
    ctx.localInput.addr += ctx.currentInputOffset;
    ctx.localInput.token = ctx.inputToken[arg->rankId];

    ctx.output.addr = ctx.outputAddr;
    ctx.output.addr += ctx.currentOutputOffset;
    ctx.output.token = ctx.outputToken;

    ccu::Variable scratchOffset;
    scratchOffset = 0;
    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        ctx.scratch[rank].addr = ctx.scratchAddr;
        ctx.scratch[rank].addr += scratchOffset;
        ctx.scratch[rank].token = ctx.scratchToken;
        scratchOffset = scratchOffset + ctx.sliceSize;

        if (rank != arg->rankId) {
            ctx.remoteInput[rank].addr = ctx.inputAddr[rank];
            ctx.remoteInput[rank].addr += ctx.currentInputOffset;
            ctx.remoteInput[rank].token = ctx.inputToken[rank];
        }
    }
    return CCU_SUCCESS;
}

CcuResult StageRankSlices(ReduceScatterContext &ctx)
{
    const CcuReduceScatterKernelArg *arg = ctx.arg;
    const uint16_t selfMask = EventBit(arg->rankId);
    OPS_CCU_RETURN_IF_ERROR(
        ccu::LocalCopy(ctx.scratch[arg->rankId], ctx.localInput, ctx.sliceSize, ctx.event, selfMask));

    for (uint32_t channelIndex = 0; channelIndex < arg->channelCount; ++channelIndex) {
        const uint32_t peerRank = arg->peerRanks[channelIndex];
        OPS_CCU_RETURN_IF_ERROR(ccu::Read(arg->channels[channelIndex], ctx.scratch[peerRank],
            ctx.remoteInput[peerRank], ctx.sliceSize, ctx.event, EventBit(peerRank)));
    }

    OPS_CCU_RETURN_IF_ERROR(
        ccu::EventWait(ctx.event, AllRankEventMask(static_cast<uint32_t>(arg->rankSize))));
    return CCU_SUCCESS;
}

CcuResult ReduceRankSlices(ReduceScatterContext &ctx)
{
    const CcuReduceScatterKernelArg *arg = ctx.arg;
    const uint32_t rankSize = static_cast<uint32_t>(arg->rankSize);

    if (rankSize > 8U) {
        // Reduce two contiguous halves per round. One wide operation replaces
        // several disjoint LocalReduce instructions while keeping a fixed tree.
        uint32_t remainingPieces = rankSize;
        ccu::Variable reduceBytes;
        while (remainingPieces > 1U) {
            const uint32_t reducePieces = remainingPieces / 2U;
            const uint32_t sourceIndex = remainingPieces - reducePieces;
            reduceBytes = ctx.sliceSize;
            for (uint32_t piece = 1; piece < reducePieces; ++piece) {
                reduceBytes += ctx.sliceSize;
            }

            constexpr uint16_t reduceMask = EventBit(0);
            OPS_CCU_RETURN_IF_ERROR(ccu::LocalReduce(ctx.scratch[0], ctx.scratch[sourceIndex],
                reduceBytes, arg->dataType, arg->reduceOp, ctx.event, reduceMask));
            OPS_CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.event, reduceMask));
            remainingPieces -= reducePieces;
        }
    } else {
        // Small rank groups keep the existing parallel tree.
        for (uint32_t stride : TREE_STRIDES) {
            if (stride >= rankSize) {
                break;
            }

            uint16_t roundMask = 0;
            for (uint32_t base = 0; base + stride < rankSize; base += 2U * stride) {
                const uint16_t eventMask = EventBit(base);
                OPS_CCU_RETURN_IF_ERROR(ccu::LocalReduce(ctx.scratch[base], ctx.scratch[base + stride],
                    ctx.sliceSize, arg->dataType, arg->reduceOp, ctx.event, eventMask));
                roundMask = static_cast<uint16_t>(roundMask | eventMask);
            }
            OPS_CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.event, roundMask));
        }
    }

    constexpr uint16_t outputCopyMask = EventBit(0);
    OPS_CCU_RETURN_IF_ERROR(
        ccu::LocalCopy(ctx.output, ctx.scratch[0], ctx.sliceSize, ctx.event, outputCopyMask));
    OPS_CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.event, outputCopyMask));
    return CCU_SUCCESS;
}

CcuResult DoChunk(ReduceScatterContext &ctx)
{
    CCU_IF(ctx.sliceSize != 0) {
        OPS_CCU_RETURN_IF_ERROR(StageRankSlices(ctx));
        OPS_CCU_RETURN_IF_ERROR(ReduceRankSlices(ctx));
    }
    return CCU_SUCCESS;
}

void AdvanceAddresses(ReduceScatterContext &ctx)
{
    const CcuReduceScatterKernelArg *arg = ctx.arg;
    const uint32_t rankSize = static_cast<uint32_t>(arg->rankSize);

    ctx.localInput.addr += ctx.inputRepeatStride;
    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        if (rank != arg->rankId) {
            ctx.remoteInput[rank].addr += ctx.inputRepeatStride;
        }
    }
    ctx.output.addr += ctx.outputRepeatStride;
}

CcuResult DoRepeat(ReduceScatterContext &ctx)
{
    OPS_CCU_RETURN_IF_ERROR(InitAddresses(ctx));

    ccu::Variable one;
    one = 1;
    CCU_DO {
        // ExecOp initializes repeatNum to UINT64_MAX - K. Incrementing
        // before the work makes this do-while body execute exactly K times.
        ctx.repeatNum = ctx.repeatNum + one;
        OPS_CCU_RETURN_IF_ERROR(DoChunk(ctx));
        AdvanceAddresses(ctx);
    }
    CCU_WHILE(ctx.repeatNum != UINT64_MAX);

    return CCU_SUCCESS;
}

CcuResult PostSync(const ReduceScatterContext &ctx)
{
    const CcuReduceScatterKernelArg *arg = ctx.arg;
    for (uint32_t channelIndex = 0; channelIndex < arg->channelCount; ++channelIndex) {
        OPS_CCU_RETURN_IF_ERROR(
            ccu::NotifyRecord(arg->channels[channelIndex], CHANNEL_NOTIFY_INDEX, POST_SYNC_MASK));
    }
    for (uint32_t channelIndex = 0; channelIndex < arg->channelCount; ++channelIndex) {
        OPS_CCU_RETURN_IF_ERROR(
            ccu::NotifyWait(arg->channels[channelIndex], CHANNEL_NOTIFY_INDEX, POST_SYNC_MASK));
    }
    return CCU_SUCCESS;
}

struct SplitReduceScatterContext {
    const CcuReduceScatterKernelArg *arg = nullptr;

    ccu::Variable inputAddr[KERNEL_MAX_RANK_SIZE];
    ccu::Variable inputToken[KERNEL_MAX_RANK_SIZE];
    ccu::Variable outputAddr;
    ccu::Variable outputToken;
    ccu::Variable scratchAddr;
    ccu::Variable scratchToken;
    ccu::Variable currentInputOffset;
    ccu::Variable sliceSize;

    ccu::LocalAddr localInput;
    ccu::RemoteAddr remoteInput[KERNEL_MAX_RANK_SIZE];
    ccu::LocalAddr scratch[KERNEL_MAX_RANK_SIZE];
    ccu::LocalAddr output;
    ccu::Event event;
};

CcuResult InitSplitResource(SplitReduceScatterContext &ctx, bool isWorker)
{
    const CcuReduceScatterKernelArg *arg = ctx.arg;
    if (arg == nullptr || arg->rankSize < 2 || arg->rankSize > KERNEL_MAX_RANK_SIZE
        || arg->rankId >= arg->rankSize || arg->channelCount > arg->rankSize - 1
        || (isWorker && arg->channelCount == 0)
        || (!isWorker && arg->hasWorker == 0 && arg->channelCount != arg->rankSize - 1)) {
        return CCU_E_PARA;
    }

    uint16_t peerMask = EventBit(arg->rankId);
    for (uint32_t channelIndex = 0; channelIndex < arg->channelCount; ++channelIndex) {
        const uint32_t peerRank = arg->peerRanks[channelIndex];
        if (peerRank >= arg->rankSize || peerRank == arg->rankId) {
            return CCU_E_PARA;
        }
        const uint16_t peerBit = EventBit(peerRank);
        if ((peerMask & peerBit) != 0) {
            return CCU_E_PARA;
        }
        peerMask = static_cast<uint16_t>(peerMask | peerBit);

        const ChannelHandle channel = arg->channels[channelIndex];
        ctx.inputAddr[peerRank] = ccu::GetResByChannel<ccu::Variable>(channel, INPUT_ADDR_XN_ID);
        ctx.inputToken[peerRank] = ccu::GetResByChannel<ccu::Variable>(channel, INPUT_TOKEN_XN_ID);
    }
    return CCU_SUCCESS;
}

CcuResult LoadSplitArgs(SplitReduceScatterContext &ctx)
{
    const uint32_t selfRank = ctx.arg->rankId;
    uint32_t argId = 0;

    // Keep this order exactly aligned with ExecOp's eight split-kernel task arguments.
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.inputAddr[selfRank], argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.outputAddr, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.inputToken[selfRank], argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.outputToken, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.scratchAddr, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.scratchToken, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.currentInputOffset, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.sliceSize, argId++));
    return CCU_SUCCESS;
}

CcuResult SplitPreSync(const SplitReduceScatterContext &ctx)
{
    const CcuReduceScatterKernelArg *arg = ctx.arg;
    const uint32_t selfRank = arg->rankId;

    for (uint32_t channelIndex = 0; channelIndex < arg->channelCount; ++channelIndex) {
        const ChannelHandle channel = arg->channels[channelIndex];
        OPS_CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(channel, ctx.inputAddr[selfRank],
            INPUT_ADDR_XN_ID, CHANNEL_NOTIFY_INDEX, INPUT_ADDR_READY_MASK));
        OPS_CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(channel, ctx.inputToken[selfRank],
            INPUT_TOKEN_XN_ID, CHANNEL_NOTIFY_INDEX, INPUT_TOKEN_READY_MASK));
    }
    for (uint32_t channelIndex = 0; channelIndex < arg->channelCount; ++channelIndex) {
        OPS_CCU_RETURN_IF_ERROR(
            ccu::NotifyWait(arg->channels[channelIndex], CHANNEL_NOTIFY_INDEX, PRE_SYNC_MASK));
    }
    return CCU_SUCCESS;
}

void InitSplitAddresses(SplitReduceScatterContext &ctx)
{
    const CcuReduceScatterKernelArg *arg = ctx.arg;

    ctx.localInput.addr = ctx.inputAddr[arg->rankId];
    ctx.localInput.addr += ctx.currentInputOffset;
    ctx.localInput.token = ctx.inputToken[arg->rankId];

    ctx.output.addr = ctx.outputAddr;
    ctx.output.token = ctx.outputToken;

    ccu::Variable scratchOffset;
    scratchOffset = 0;
    for (uint32_t rank = 0; rank < arg->rankSize; ++rank) {
        ctx.scratch[rank].addr = ctx.scratchAddr;
        ctx.scratch[rank].addr += scratchOffset;
        ctx.scratch[rank].token = ctx.scratchToken;
        scratchOffset = scratchOffset + ctx.sliceSize;
    }

    for (uint32_t channelIndex = 0; channelIndex < arg->channelCount; ++channelIndex) {
        const uint32_t peerRank = arg->peerRanks[channelIndex];
        ctx.remoteInput[peerRank].addr = ctx.inputAddr[peerRank];
        ctx.remoteInput[peerRank].addr += ctx.currentInputOffset;
        ctx.remoteInput[peerRank].token = ctx.inputToken[peerRank];
    }
}

CcuResult StageSplitGroup(SplitReduceScatterContext &ctx, bool includeSelf)
{
    const CcuReduceScatterKernelArg *arg = ctx.arg;
    uint16_t eventMask = 0;

    CCU_IF(ctx.sliceSize != 0) {
        if (includeSelf) {
            const uint16_t selfMask = EventBit(arg->rankId);
            OPS_CCU_RETURN_IF_ERROR(ccu::LocalCopy(
                ctx.scratch[arg->rankId], ctx.localInput, ctx.sliceSize, ctx.event, selfMask));
            eventMask = static_cast<uint16_t>(eventMask | selfMask);
        }

        for (uint32_t channelIndex = 0; channelIndex < arg->channelCount; ++channelIndex) {
            const uint32_t peerRank = arg->peerRanks[channelIndex];
            const uint16_t peerMask = EventBit(peerRank);
            OPS_CCU_RETURN_IF_ERROR(ccu::Read(arg->channels[channelIndex], ctx.scratch[peerRank],
                ctx.remoteInput[peerRank], ctx.sliceSize, ctx.event, peerMask));
            eventMask = static_cast<uint16_t>(eventMask | peerMask);
        }
        OPS_CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.event, eventMask));
    }
    return CCU_SUCCESS;
}

CcuResult ReduceSplitTree(SplitReduceScatterContext &ctx)
{
    const CcuReduceScatterKernelArg *arg = ctx.arg;

    CCU_IF(ctx.sliceSize != 0) {
        for (uint32_t stride : TREE_STRIDES) {
            if (stride >= arg->rankSize) {
                break;
            }

            uint16_t roundMask = 0;
            for (uint32_t base = 0; base + stride < arg->rankSize; base += 2U * stride) {
                const uint16_t reduceMask = EventBit(base);
                OPS_CCU_RETURN_IF_ERROR(ccu::LocalReduce(ctx.scratch[base],
                    ctx.scratch[base + stride], ctx.sliceSize, arg->dataType, arg->reduceOp,
                    ctx.event, reduceMask));
                roundMask = static_cast<uint16_t>(roundMask | reduceMask);
            }
            OPS_CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.event, roundMask));
        }

        constexpr uint16_t outputCopyMask = EventBit(0);
        OPS_CCU_RETURN_IF_ERROR(
            ccu::LocalCopy(ctx.output, ctx.scratch[0], ctx.sliceSize, ctx.event, outputCopyMask));
        OPS_CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.event, outputCopyMask));
    }
    return CCU_SUCCESS;
}

CcuResult SplitPostSync(const SplitReduceScatterContext &ctx)
{
    const CcuReduceScatterKernelArg *arg = ctx.arg;
    for (uint32_t channelIndex = 0; channelIndex < arg->channelCount; ++channelIndex) {
        OPS_CCU_RETURN_IF_ERROR(
            ccu::NotifyRecord(arg->channels[channelIndex], CHANNEL_NOTIFY_INDEX, POST_SYNC_MASK));
    }
    for (uint32_t channelIndex = 0; channelIndex < arg->channelCount; ++channelIndex) {
        OPS_CCU_RETURN_IF_ERROR(
            ccu::NotifyWait(arg->channels[channelIndex], CHANNEL_NOTIFY_INDEX, POST_SYNC_MASK));
    }
    return CCU_SUCCESS;
}

constexpr uint32_t TWO_D_SERVER_COUNT = 2;
constexpr uint32_t TWO_D_LOCAL_RANK_SIZE = 8;
constexpr uint32_t TWO_D_MESH_PEER_COUNT = TWO_D_LOCAL_RANK_SIZE - 1;
constexpr uint32_t TWO_D_NHR_PEER_COUNT = 1;
constexpr uint32_t TWO_D_MESH_A_OP_COUNT =
    TWO_D_SERVER_COUNT * TWO_D_MESH_PEER_COUNT;

struct ReduceScatter2DTaskArgs {
    ccu::Variable inputAddr;
    ccu::Variable outputAddr;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable scratchAddr;
    ccu::Variable scratchToken;
    ccu::Variable recvBytes;
    ccu::Variable splitABytes;
    ccu::Variable aLaneBytes;
    ccu::Variable aLastLaneBytes;
    ccu::Variable bLaneBytes;
    ccu::Variable bLastLaneBytes;
};

struct ReduceScatter2DBaseContext {
    const CcuReduceScatter2DKernelArg *arg = nullptr;
    ReduceScatter2DTaskArgs task;
    ccu::Variable remoteBaseAddr[TWO_D_MESH_PEER_COUNT];
    ccu::Variable remoteBaseToken[TWO_D_MESH_PEER_COUNT];
    ccu::Event event;
};

uint32_t Find2DTopologyIndex(const CcuReduceScatter2DKernelArg *arg, uint32_t rank)
{
    for (uint32_t index = 0; index < KERNEL_MAX_RANK_SIZE; ++index) {
        if (arg->topologyRanks[index] == rank) {
            return index;
        }
    }
    return KERNEL_MAX_RANK_SIZE;
}

CcuResult Init2DResource(ReduceScatter2DBaseContext &ctx, bool isMesh)
{
    const CcuReduceScatter2DKernelArg *arg = ctx.arg;
    const uint32_t expectedChannelCount =
        isMesh ? TWO_D_MESH_PEER_COUNT : TWO_D_NHR_PEER_COUNT;
    if (arg == nullptr || arg->rankSize != KERNEL_MAX_RANK_SIZE
        || arg->rankId >= arg->rankSize || arg->serverIndex >= TWO_D_SERVER_COUNT
        || arg->localIndex >= TWO_D_LOCAL_RANK_SIZE
        || arg->channelCount != expectedChannelCount) {
        return CCU_E_PARA;
    }

    uint16_t topologyMask = 0;
    for (uint32_t index = 0; index < KERNEL_MAX_RANK_SIZE; ++index) {
        const uint32_t topologyRank = arg->topologyRanks[index];
        if (topologyRank >= KERNEL_MAX_RANK_SIZE) {
            return CCU_E_PARA;
        }
        const uint16_t rankBit = EventBit(topologyRank);
        if ((topologyMask & rankBit) != 0) {
            return CCU_E_PARA;
        }
        topologyMask = static_cast<uint16_t>(topologyMask | rankBit);
    }
    const uint32_t selfTopologyIndex =
        arg->serverIndex * TWO_D_LOCAL_RANK_SIZE + arg->localIndex;
    if (arg->topologyRanks[selfTopologyIndex] != arg->rankId) {
        return CCU_E_PARA;
    }

    uint16_t peerMask = 0;
    for (uint32_t channelIndex = 0; channelIndex < arg->channelCount; ++channelIndex) {
        const uint32_t peerRank = arg->peerRanks[channelIndex];
        if (peerRank >= arg->rankSize || peerRank == arg->rankId) {
            return CCU_E_PARA;
        }
        const uint16_t peerBit = EventBit(peerRank);
        if ((peerMask & peerBit) != 0) {
            return CCU_E_PARA;
        }
        peerMask = static_cast<uint16_t>(peerMask | peerBit);

        const uint32_t peerTopologyIndex = Find2DTopologyIndex(arg, peerRank);
        if (peerTopologyIndex >= KERNEL_MAX_RANK_SIZE) {
            return CCU_E_PARA;
        }
        if (isMesh) {
            if (peerTopologyIndex / TWO_D_LOCAL_RANK_SIZE != arg->serverIndex
                || peerTopologyIndex % TWO_D_LOCAL_RANK_SIZE == arg->localIndex) {
                return CCU_E_PARA;
            }
        } else {
            const uint32_t pairedTopologyIndex =
                (TWO_D_SERVER_COUNT - 1U - arg->serverIndex) * TWO_D_LOCAL_RANK_SIZE
                + arg->localIndex;
            if (peerRank != arg->topologyRanks[pairedTopologyIndex]) {
                return CCU_E_PARA;
            }
        }

        const ChannelHandle channel = arg->channels[channelIndex];
        ctx.remoteBaseAddr[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(channel, INPUT_ADDR_XN_ID);
        ctx.remoteBaseToken[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(channel, INPUT_TOKEN_XN_ID);
    }
    return CCU_SUCCESS;
}

CcuResult Load2DTaskArgs(ReduceScatter2DBaseContext &ctx)
{
    uint32_t argId = 0;
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.task.inputAddr, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.task.outputAddr, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.task.inputToken, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.task.outputToken, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.task.scratchAddr, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.task.scratchToken, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.task.recvBytes, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.task.splitABytes, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.task.aLaneBytes, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.task.aLastLaneBytes, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.task.bLaneBytes, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.task.bLastLaneBytes, argId++));
    return CCU_SUCCESS;
}

CcuResult TwoDPreSync(const ReduceScatter2DBaseContext &ctx, bool publishScratch)
{
    const CcuReduceScatter2DKernelArg *arg = ctx.arg;
    if (publishScratch) {
        for (uint32_t channelIndex = 0; channelIndex < arg->channelCount; ++channelIndex) {
            const ChannelHandle channel = arg->channels[channelIndex];
            OPS_CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(channel, ctx.task.scratchAddr,
                INPUT_ADDR_XN_ID, CHANNEL_NOTIFY_INDEX, INPUT_ADDR_READY_MASK));
            OPS_CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(channel, ctx.task.scratchToken,
                INPUT_TOKEN_XN_ID, CHANNEL_NOTIFY_INDEX, INPUT_TOKEN_READY_MASK));
        }
    } else {
        for (uint32_t channelIndex = 0; channelIndex < arg->channelCount; ++channelIndex) {
            const ChannelHandle channel = arg->channels[channelIndex];
            OPS_CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(channel, ctx.task.inputAddr,
                INPUT_ADDR_XN_ID, CHANNEL_NOTIFY_INDEX, INPUT_ADDR_READY_MASK));
            OPS_CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(channel, ctx.task.inputToken,
                INPUT_TOKEN_XN_ID, CHANNEL_NOTIFY_INDEX, INPUT_TOKEN_READY_MASK));
        }
    }
    for (uint32_t channelIndex = 0; channelIndex < arg->channelCount; ++channelIndex) {
        OPS_CCU_RETURN_IF_ERROR(
            ccu::NotifyWait(arg->channels[channelIndex], CHANNEL_NOTIFY_INDEX, PRE_SYNC_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult TwoDPostSync(const ReduceScatter2DBaseContext &ctx)
{
    const CcuReduceScatter2DKernelArg *arg = ctx.arg;
    for (uint32_t channelIndex = 0; channelIndex < arg->channelCount; ++channelIndex) {
        OPS_CCU_RETURN_IF_ERROR(
            ccu::NotifyRecord(arg->channels[channelIndex], CHANNEL_NOTIFY_INDEX, POST_SYNC_MASK));
    }
    for (uint32_t channelIndex = 0; channelIndex < arg->channelCount; ++channelIndex) {
        OPS_CCU_RETURN_IF_ERROR(
            ccu::NotifyWait(arg->channels[channelIndex], CHANNEL_NOTIFY_INDEX, POST_SYNC_MASK));
    }
    return CCU_SUCCESS;
}

void BuildRepeatedOffset(ccu::Variable &offset, const ccu::Variable &stride, uint32_t count)
{
    offset = 0;
    for (uint32_t index = 0; index < count; ++index) {
        offset = offset + stride;
    }
}

void BuildLaneOffsets(ccu::Variable (&offsets)[TWO_D_MESH_PEER_COUNT],
    const ccu::Variable &laneBytes)
{
    offsets[0] = 0;
    for (uint32_t lane = 1; lane < TWO_D_MESH_PEER_COUNT; ++lane) {
        offsets[lane] = offsets[lane - 1] + laneBytes;
    }
}

void BuildBBytes(ccu::Variable &bBytes, const ReduceScatter2DTaskArgs &task)
{
    bBytes = task.bLastLaneBytes;
    for (uint32_t lane = 1; lane < TWO_D_MESH_PEER_COUNT; ++lane) {
        bBytes = bBytes + task.bLaneBytes;
    }
}

struct MeshAStage1Context {
    ReduceScatter2DBaseContext base;
    ccu::Variable targetInputOffset[TWO_D_SERVER_COUNT];
    ccu::Variable laneOffset[TWO_D_MESH_PEER_COUNT];
    ccu::LocalAddr localInput[TWO_D_SERVER_COUNT];
    ccu::LocalAddr scratchA[TWO_D_SERVER_COUNT];
    ccu::LocalAddr reduceDst[TWO_D_MESH_A_OP_COUNT];
    ccu::RemoteAddr reduceSrc[TWO_D_MESH_A_OP_COUNT];
};

void InitMeshAStage1Addresses(MeshAStage1Context &ctx)
{
    const CcuReduceScatter2DKernelArg *arg = ctx.base.arg;
    ReduceScatter2DTaskArgs &task = ctx.base.task;
    BuildLaneOffsets(ctx.laneOffset, task.aLaneBytes);

    for (uint32_t server = 0; server < TWO_D_SERVER_COUNT; ++server) {
        const uint32_t targetRank =
            arg->topologyRanks[server * TWO_D_LOCAL_RANK_SIZE + arg->localIndex];
        BuildRepeatedOffset(
            ctx.targetInputOffset[server], task.recvBytes, targetRank);

        ctx.localInput[server].addr = task.inputAddr;
        ctx.localInput[server].addr += ctx.targetInputOffset[server];
        ctx.localInput[server].token = task.inputToken;

        if (server == arg->serverIndex) {
            // This rank consumes its own-server partial in NHR stage 2.  Seed
            // the final output here so that stage 2 can reduce into it
            // directly instead of copying the whole A part out of scratch.
            ctx.scratchA[server].addr = task.outputAddr;
            ctx.scratchA[server].token = task.outputToken;
        } else {
            // The paired server still reads this partial remotely in stage 2.
            ctx.scratchA[server].addr = task.scratchAddr;
            if (server != 0) {
                ctx.scratchA[server].addr += task.splitABytes;
            }
            ctx.scratchA[server].token = task.scratchToken;
        }
    }
}

CcuResult RunMeshAStage1(MeshAStage1Context &ctx)
{
    const CcuReduceScatter2DKernelArg *arg = ctx.base.arg;
    ReduceScatter2DTaskArgs &task = ctx.base.task;

    CCU_IF(task.splitABytes != 0) {
        OPS_CCU_RETURN_IF_ERROR(ccu::LocalCopy(ctx.scratchA[0], ctx.localInput[0],
            task.splitABytes, ctx.base.event, EventBit(0)));
        OPS_CCU_RETURN_IF_ERROR(ccu::LocalCopy(ctx.scratchA[1], ctx.localInput[1],
            task.splitABytes, ctx.base.event, EventBit(1)));
        OPS_CCU_RETURN_IF_ERROR(
            ccu::EventWait(ctx.base.event, static_cast<uint16_t>(EventBit(0) | EventBit(1))));

        for (uint32_t step = 0; step < TWO_D_MESH_PEER_COUNT; ++step) {
            uint16_t stepMask = 0;
            for (uint32_t channelIndex = 0;
                channelIndex < TWO_D_MESH_PEER_COUNT; ++channelIndex) {
                const uint32_t lane =
                    (channelIndex + step) % TWO_D_MESH_PEER_COUNT;
                for (uint32_t server = 0; server < TWO_D_SERVER_COUNT; ++server) {
                    const uint32_t opIndex =
                        channelIndex * TWO_D_SERVER_COUNT + server;
                    const uint16_t opMask = EventBit(opIndex);

                    ctx.reduceDst[opIndex].addr = ctx.scratchA[server].addr;
                    ctx.reduceDst[opIndex].addr += ctx.laneOffset[lane];
                    ctx.reduceDst[opIndex].token =
                        ctx.scratchA[server].token;

                    ctx.reduceSrc[opIndex].addr =
                        ctx.base.remoteBaseAddr[channelIndex];
                    ctx.reduceSrc[opIndex].addr += ctx.targetInputOffset[server];
                    ctx.reduceSrc[opIndex].addr += ctx.laneOffset[lane];
                    ctx.reduceSrc[opIndex].token =
                        ctx.base.remoteBaseToken[channelIndex];

                    if (lane + 1U == TWO_D_MESH_PEER_COUNT) {
                        OPS_CCU_RETURN_IF_ERROR(ccu::ReadReduce(
                            arg->channels[channelIndex], ctx.reduceDst[opIndex],
                            ctx.reduceSrc[opIndex], task.aLastLaneBytes, arg->dataType,
                            arg->reduceOp, ctx.base.event, opMask));
                    } else {
                        OPS_CCU_RETURN_IF_ERROR(ccu::ReadReduce(
                            arg->channels[channelIndex], ctx.reduceDst[opIndex],
                            ctx.reduceSrc[opIndex], task.aLaneBytes, arg->dataType,
                            arg->reduceOp, ctx.base.event, opMask));
                    }
                    stepMask = static_cast<uint16_t>(stepMask | opMask);
                }
            }
            OPS_CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.base.event, stepMask));
        }
    }
    return CCU_SUCCESS;
}

struct NhrBStage1Context {
    ReduceScatter2DBaseContext base;
    ccu::Variable bBytes;
    ccu::Variable targetInputOffset[TWO_D_LOCAL_RANK_SIZE];
    ccu::LocalAddr localInput[TWO_D_LOCAL_RANK_SIZE];
    ccu::LocalAddr scratchB[TWO_D_LOCAL_RANK_SIZE];
    ccu::RemoteAddr remoteInput[TWO_D_LOCAL_RANK_SIZE];
};

void InitNhrBStage1Addresses(NhrBStage1Context &ctx)
{
    const CcuReduceScatter2DKernelArg *arg = ctx.base.arg;
    const ReduceScatter2DTaskArgs &task = ctx.base.task;
    BuildBBytes(ctx.bBytes, task);

    ccu::Variable scratchOffset;
    scratchOffset = task.splitABytes + task.splitABytes;
    for (uint32_t localIndex = 0; localIndex < TWO_D_LOCAL_RANK_SIZE; ++localIndex) {
        const uint32_t targetRank =
            arg->topologyRanks[arg->serverIndex * TWO_D_LOCAL_RANK_SIZE + localIndex];
        BuildRepeatedOffset(
            ctx.targetInputOffset[localIndex], task.recvBytes, targetRank);
        ctx.targetInputOffset[localIndex] =
            ctx.targetInputOffset[localIndex] + task.splitABytes;

        ctx.localInput[localIndex].addr = task.inputAddr;
        ctx.localInput[localIndex].addr += ctx.targetInputOffset[localIndex];
        ctx.localInput[localIndex].token = task.inputToken;

        if (localIndex == arg->localIndex) {
            // Mesh stage 2 consumes this rank's local-index partial.  Put it
            // at the final B output offset and remove the later full copy.
            ctx.scratchB[localIndex].addr = task.outputAddr;
            ctx.scratchB[localIndex].addr += task.splitABytes;
            ctx.scratchB[localIndex].token = task.outputToken;
        } else {
            // Other local ranks still read this partial remotely in stage 2.
            ctx.scratchB[localIndex].addr = task.scratchAddr;
            ctx.scratchB[localIndex].addr += scratchOffset;
            ctx.scratchB[localIndex].token = task.scratchToken;
        }

        ctx.remoteInput[localIndex].addr = ctx.base.remoteBaseAddr[0];
        ctx.remoteInput[localIndex].addr += ctx.targetInputOffset[localIndex];
        ctx.remoteInput[localIndex].token = ctx.base.remoteBaseToken[0];

        scratchOffset = scratchOffset + ctx.bBytes;
    }
}

CcuResult RunNhrBStage1(NhrBStage1Context &ctx)
{
    const CcuReduceScatter2DKernelArg *arg = ctx.base.arg;
    constexpr uint16_t allTargetMask =
        static_cast<uint16_t>((uint32_t{1} << TWO_D_LOCAL_RANK_SIZE) - uint32_t{1});

    CCU_IF(ctx.bBytes != 0) {
        for (uint32_t localIndex = 0; localIndex < TWO_D_LOCAL_RANK_SIZE; ++localIndex) {
            OPS_CCU_RETURN_IF_ERROR(ccu::LocalCopy(ctx.scratchB[localIndex],
                ctx.localInput[localIndex], ctx.bBytes, ctx.base.event,
                EventBit(localIndex)));
        }
        OPS_CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.base.event, allTargetMask));

        for (uint32_t localIndex = 0; localIndex < TWO_D_LOCAL_RANK_SIZE; ++localIndex) {
            OPS_CCU_RETURN_IF_ERROR(ccu::ReadReduce(arg->channels[0],
                ctx.scratchB[localIndex], ctx.remoteInput[localIndex], ctx.bBytes,
                arg->dataType, arg->reduceOp, ctx.base.event, EventBit(localIndex)));
        }
        OPS_CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.base.event, allTargetMask));
    }
    return CCU_SUCCESS;
}

struct MeshBStage2Context {
    ReduceScatter2DBaseContext base;
    ccu::Variable bBytes;
    ccu::Variable scratchTargetOffset;
    ccu::Variable laneOffset[TWO_D_MESH_PEER_COUNT];
    ccu::LocalAddr output;
    ccu::LocalAddr reduceDst[TWO_D_MESH_PEER_COUNT];
    ccu::RemoteAddr reduceSrc[TWO_D_MESH_PEER_COUNT];
};

void InitMeshBStage2Addresses(MeshBStage2Context &ctx)
{
    const CcuReduceScatter2DKernelArg *arg = ctx.base.arg;
    const ReduceScatter2DTaskArgs &task = ctx.base.task;
    BuildBBytes(ctx.bBytes, task);
    BuildLaneOffsets(ctx.laneOffset, task.bLaneBytes);

    ctx.scratchTargetOffset = task.splitABytes + task.splitABytes;
    for (uint32_t localIndex = 0; localIndex < arg->localIndex; ++localIndex) {
        ctx.scratchTargetOffset = ctx.scratchTargetOffset + ctx.bBytes;
    }

    ctx.output.addr = task.outputAddr;
    ctx.output.addr += task.splitABytes;
    ctx.output.token = task.outputToken;
}

CcuResult RunMeshBStage2(MeshBStage2Context &ctx)
{
    const CcuReduceScatter2DKernelArg *arg = ctx.base.arg;
    const ReduceScatter2DTaskArgs &task = ctx.base.task;
    constexpr uint16_t allPeerMask =
        static_cast<uint16_t>((uint32_t{1} << TWO_D_MESH_PEER_COUNT) - uint32_t{1});

    CCU_IF(ctx.bBytes != 0) {
        for (uint32_t step = 0; step < TWO_D_MESH_PEER_COUNT; ++step) {
            for (uint32_t channelIndex = 0;
                channelIndex < TWO_D_MESH_PEER_COUNT; ++channelIndex) {
                const uint32_t lane =
                    (channelIndex + step) % TWO_D_MESH_PEER_COUNT;
                ctx.reduceDst[channelIndex].addr = ctx.output.addr;
                ctx.reduceDst[channelIndex].addr += ctx.laneOffset[lane];
                ctx.reduceDst[channelIndex].token = task.outputToken;

                ctx.reduceSrc[channelIndex].addr =
                    ctx.base.remoteBaseAddr[channelIndex];
                ctx.reduceSrc[channelIndex].addr += ctx.scratchTargetOffset;
                ctx.reduceSrc[channelIndex].addr += ctx.laneOffset[lane];
                ctx.reduceSrc[channelIndex].token =
                    ctx.base.remoteBaseToken[channelIndex];

                if (lane + 1U == TWO_D_MESH_PEER_COUNT) {
                    OPS_CCU_RETURN_IF_ERROR(ccu::ReadReduce(
                        arg->channels[channelIndex], ctx.reduceDst[channelIndex],
                        ctx.reduceSrc[channelIndex], task.bLastLaneBytes, arg->dataType,
                        arg->reduceOp, ctx.base.event, EventBit(channelIndex)));
                } else {
                    OPS_CCU_RETURN_IF_ERROR(ccu::ReadReduce(
                        arg->channels[channelIndex], ctx.reduceDst[channelIndex],
                        ctx.reduceSrc[channelIndex], task.bLaneBytes, arg->dataType,
                        arg->reduceOp, ctx.base.event, EventBit(channelIndex)));
                }
            }
            OPS_CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.base.event, allPeerMask));
        }
    }
    return CCU_SUCCESS;
}

struct NhrAStage2Context {
    ReduceScatter2DBaseContext base;
    ccu::Variable scratchTargetOffset;
    ccu::LocalAddr output;
    ccu::RemoteAddr remoteScratch;
};

void InitNhrAStage2Addresses(NhrAStage2Context &ctx)
{
    const CcuReduceScatter2DKernelArg *arg = ctx.base.arg;
    const ReduceScatter2DTaskArgs &task = ctx.base.task;
    BuildRepeatedOffset(
        ctx.scratchTargetOffset, task.splitABytes, arg->serverIndex);

    ctx.output.addr = task.outputAddr;
    ctx.output.token = task.outputToken;

    ctx.remoteScratch.addr = ctx.base.remoteBaseAddr[0];
    ctx.remoteScratch.addr += ctx.scratchTargetOffset;
    ctx.remoteScratch.token = ctx.base.remoteBaseToken[0];
}

CcuResult RunNhrAStage2(NhrAStage2Context &ctx)
{
    const CcuReduceScatter2DKernelArg *arg = ctx.base.arg;
    ReduceScatter2DTaskArgs &task = ctx.base.task;

    CCU_IF(task.splitABytes != 0) {
        OPS_CCU_RETURN_IF_ERROR(ccu::ReadReduce(arg->channels[0], ctx.output,
            ctx.remoteScratch, task.splitABytes, arg->dataType, arg->reduceOp,
            ctx.base.event, EventBit(0)));
        OPS_CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.base.event, EventBit(0)));
    }
    return CCU_SUCCESS;
}

constexpr uint32_t LATIN_4X1_RANK_COUNT = 4;
constexpr uint32_t LATIN_4X1_PEER_COUNT = LATIN_4X1_RANK_COUNT - 1;
constexpr uint16_t LATIN_4X1_EVENT_MASK =
    static_cast<uint16_t>((uint32_t{1} << LATIN_4X1_PEER_COUNT) - uint32_t{1});

struct Latin4x1TaskArgs {
    ccu::Variable inputAddr;
    ccu::Variable outputAddr;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable currentInputOffset;
    ccu::Variable recvBytes;
    ccu::Variable laneBytes;
    ccu::Variable lastLaneBytes;
};

struct Latin4x1Context {
    const CcuReduceScatterKernelArg *arg = nullptr;
    Latin4x1TaskArgs task;
    ccu::Variable remoteInputAddr[LATIN_4X1_PEER_COUNT];
    ccu::Variable remoteInputToken[LATIN_4X1_PEER_COUNT];
    ccu::Variable laneOffset[LATIN_4X1_PEER_COUNT];
    ccu::LocalAddr localInput;
    ccu::LocalAddr output;
    ccu::LocalAddr reduceDst[LATIN_4X1_PEER_COUNT];
    ccu::RemoteAddr reduceSrc[LATIN_4X1_PEER_COUNT];
    ccu::Event event;
};

CcuResult InitLatin4x1Resource(Latin4x1Context &ctx)
{
    const CcuReduceScatterKernelArg *arg = ctx.arg;
    if (arg == nullptr || arg->rankSize != LATIN_4X1_RANK_COUNT ||
        arg->rankId >= LATIN_4X1_RANK_COUNT ||
        arg->channelCount != LATIN_4X1_PEER_COUNT) {
        return CCU_E_PARA;
    }

    uint16_t peerMask = EventBit(arg->rankId);
    for (uint32_t channelIndex = 0;
        channelIndex < LATIN_4X1_PEER_COUNT; ++channelIndex) {
        const uint32_t peerRank = arg->peerRanks[channelIndex];
        if (peerRank >= LATIN_4X1_RANK_COUNT || peerRank == arg->rankId) {
            return CCU_E_PARA;
        }
        const uint16_t peerBit = EventBit(peerRank);
        if ((peerMask & peerBit) != 0) {
            return CCU_E_PARA;
        }
        peerMask = static_cast<uint16_t>(peerMask | peerBit);

        const ChannelHandle channel = arg->channels[channelIndex];
        ctx.remoteInputAddr[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(channel, INPUT_ADDR_XN_ID);
        ctx.remoteInputToken[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(channel, INPUT_TOKEN_XN_ID);
    }
    return peerMask == AllRankEventMask(LATIN_4X1_RANK_COUNT) ?
        CCU_SUCCESS : CCU_E_PARA;
}

CcuResult LoadLatin4x1Args(Latin4x1Context &ctx)
{
    uint32_t argId = 0;
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.task.inputAddr, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.task.outputAddr, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.task.inputToken, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.task.outputToken, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.task.currentInputOffset, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.task.recvBytes, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.task.laneBytes, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.task.lastLaneBytes, argId++));
    return CCU_SUCCESS;
}

CcuResult Latin4x1PreSync(const Latin4x1Context &ctx)
{
    const CcuReduceScatterKernelArg *arg = ctx.arg;
    for (uint32_t channelIndex = 0;
        channelIndex < LATIN_4X1_PEER_COUNT; ++channelIndex) {
        const ChannelHandle channel = arg->channels[channelIndex];
        OPS_CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(channel,
            ctx.task.inputAddr, INPUT_ADDR_XN_ID, CHANNEL_NOTIFY_INDEX,
            INPUT_ADDR_READY_MASK));
        OPS_CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(channel,
            ctx.task.inputToken, INPUT_TOKEN_XN_ID, CHANNEL_NOTIFY_INDEX,
            INPUT_TOKEN_READY_MASK));
    }
    for (uint32_t channelIndex = 0;
        channelIndex < LATIN_4X1_PEER_COUNT; ++channelIndex) {
        OPS_CCU_RETURN_IF_ERROR(ccu::NotifyWait(
            arg->channels[channelIndex], CHANNEL_NOTIFY_INDEX, PRE_SYNC_MASK));
    }
    return CCU_SUCCESS;
}

void InitLatin4x1Addresses(Latin4x1Context &ctx)
{
    ctx.localInput.addr = ctx.task.inputAddr;
    ctx.localInput.addr += ctx.task.currentInputOffset;
    ctx.localInput.token = ctx.task.inputToken;

    ctx.output.addr = ctx.task.outputAddr;
    ctx.output.token = ctx.task.outputToken;

    ctx.laneOffset[0] = 0;
    for (uint32_t lane = 1; lane < LATIN_4X1_PEER_COUNT; ++lane) {
        ctx.laneOffset[lane] =
            ctx.laneOffset[lane - 1] + ctx.task.laneBytes;
    }
}

CcuResult RunLatin4x1(Latin4x1Context &ctx)
{
    const CcuReduceScatterKernelArg *arg = ctx.arg;
    CCU_IF(ctx.task.recvBytes != 0) {
        OPS_CCU_RETURN_IF_ERROR(ccu::LocalCopy(ctx.output, ctx.localInput,
            ctx.task.recvBytes, ctx.event, EventBit(0)));
        OPS_CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.event, EventBit(0)));

        for (uint32_t step = 0; step < LATIN_4X1_PEER_COUNT; ++step) {
            for (uint32_t channelIndex = 0;
                channelIndex < LATIN_4X1_PEER_COUNT; ++channelIndex) {
                const uint32_t lane =
                    (channelIndex + step) % LATIN_4X1_PEER_COUNT;
                ctx.reduceDst[channelIndex].addr = ctx.task.outputAddr;
                ctx.reduceDst[channelIndex].addr += ctx.laneOffset[lane];
                ctx.reduceDst[channelIndex].token = ctx.task.outputToken;

                ctx.reduceSrc[channelIndex].addr =
                    ctx.remoteInputAddr[channelIndex];
                ctx.reduceSrc[channelIndex].addr +=
                    ctx.task.currentInputOffset;
                ctx.reduceSrc[channelIndex].addr += ctx.laneOffset[lane];
                ctx.reduceSrc[channelIndex].token =
                    ctx.remoteInputToken[channelIndex];

                if (lane + 1U == LATIN_4X1_PEER_COUNT) {
                    OPS_CCU_RETURN_IF_ERROR(ccu::ReadReduce(
                        arg->channels[channelIndex], ctx.reduceDst[channelIndex],
                        ctx.reduceSrc[channelIndex], ctx.task.lastLaneBytes,
                        arg->dataType, arg->reduceOp, ctx.event,
                        EventBit(channelIndex)));
                } else {
                    OPS_CCU_RETURN_IF_ERROR(ccu::ReadReduce(
                        arg->channels[channelIndex], ctx.reduceDst[channelIndex],
                        ctx.reduceSrc[channelIndex], ctx.task.laneBytes,
                        arg->dataType, arg->reduceOp, ctx.event,
                        EventBit(channelIndex)));
                }
            }
            OPS_CCU_RETURN_IF_ERROR(
                ccu::EventWait(ctx.event, LATIN_4X1_EVENT_MASK));
        }
    }
    return CCU_SUCCESS;
}

CcuResult Latin4x1PostSync(const Latin4x1Context &ctx)
{
    const CcuReduceScatterKernelArg *arg = ctx.arg;
    for (uint32_t channelIndex = 0;
        channelIndex < LATIN_4X1_PEER_COUNT; ++channelIndex) {
        OPS_CCU_RETURN_IF_ERROR(ccu::NotifyRecord(
            arg->channels[channelIndex], CHANNEL_NOTIFY_INDEX, POST_SYNC_MASK));
    }
    for (uint32_t channelIndex = 0;
        channelIndex < LATIN_4X1_PEER_COUNT; ++channelIndex) {
        OPS_CCU_RETURN_IF_ERROR(ccu::NotifyWait(
            arg->channels[channelIndex], CHANNEL_NOTIFY_INDEX, POST_SYNC_MASK));
    }
    return CCU_SUCCESS;
}

constexpr uint32_t ASYM_RANK_SIZE = 12;
constexpr uint32_t ASYM_SERVER_COUNT = 2;
constexpr uint32_t ASYM_LARGE_RANK_COUNT = 8;
constexpr uint32_t ASYM_SMALL_RANK_COUNT = 4;
constexpr uint32_t ASYM_MAX_CHANNEL_COUNT = 8;
constexpr uint32_t ASYM_MAX_LOCAL_TARGET_COUNT = 8;
constexpr uint32_t ASYM_MAX_A_TARGET_COUNT = 3;

struct ReduceScatterAsymTaskArgs {
    ccu::Variable inputAddr;
    ccu::Variable outputAddr;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable scratchAddr;
    ccu::Variable scratchToken;
    ccu::Variable recvBytes;
    ccu::Variable splitABytes;
    ccu::Variable aLaneBytes;
    ccu::Variable aLastLaneBytes;
    ccu::Variable bCrossLaneBytes;
    ccu::Variable bCrossLastLaneBytes;
    ccu::Variable bMeshLaneBytes;
    ccu::Variable bMeshLastLaneBytes;
};

struct ReduceScatterAsymBaseContext {
    const CcuReduceScatterAsymKernelArg *arg = nullptr;
    ReduceScatterAsymTaskArgs task;
    ccu::Variable remoteBaseAddr[ASYM_MAX_CHANNEL_COUNT];
    ccu::Variable remoteBaseToken[ASYM_MAX_CHANNEL_COUNT];
    ccu::Event event;
};

uint32_t AsymServerOffset(uint32_t serverIndex)
{
    return serverIndex == 0 ? 0U : ASYM_LARGE_RANK_COUNT;
}

uint32_t FindAsymTopologyIndex(const CcuReduceScatterAsymKernelArg *arg, uint32_t rank)
{
    for (uint32_t index = 0; index < ASYM_RANK_SIZE; ++index) {
        if (arg->topologyRanks[index] == rank) {
            return index;
        }
    }
    return ASYM_RANK_SIZE;
}

uint32_t FindAsymChannelIndex(const CcuReduceScatterAsymKernelArg *arg, uint32_t peerRank)
{
    for (uint32_t index = 0; index < arg->channelCount; ++index) {
        if (arg->peerRanks[index] == peerRank) {
            return index;
        }
    }
    return arg->channelCount;
}

constexpr uint16_t AsymChannelMask(uint32_t channelCount)
{
    return static_cast<uint16_t>((uint32_t{1} << channelCount) - uint32_t{1});
}

CcuResult InitAsymResource(ReduceScatterAsymBaseContext &ctx, bool isMesh)
{
    const CcuReduceScatterAsymKernelArg *arg = ctx.arg;
    if (arg == nullptr || arg->rankSize != ASYM_RANK_SIZE ||
        arg->rankId >= arg->rankSize || arg->serverIndex >= ASYM_SERVER_COUNT ||
        arg->localIndex >= arg->localRankCount ||
        arg->localRankCount + arg->crossRankCount != ASYM_RANK_SIZE) {
        return CCU_E_PARA;
    }
    const bool onLargeServer =
        arg->serverIndex == 0 && arg->localRankCount == ASYM_LARGE_RANK_COUNT &&
        arg->crossRankCount == ASYM_SMALL_RANK_COUNT;
    const bool onSmallServer =
        arg->serverIndex == 1 && arg->localRankCount == ASYM_SMALL_RANK_COUNT &&
        arg->crossRankCount == ASYM_LARGE_RANK_COUNT;
    const uint32_t expectedChannelCount =
        isMesh ? arg->localRankCount - 1U : arg->crossRankCount;
    if ((!onLargeServer && !onSmallServer) ||
        expectedChannelCount == 0 || expectedChannelCount > ASYM_MAX_CHANNEL_COUNT ||
        arg->channelCount != expectedChannelCount) {
        return CCU_E_PARA;
    }

    uint16_t topologyMask = 0;
    for (uint32_t index = 0; index < ASYM_RANK_SIZE; ++index) {
        const uint32_t topologyRank = arg->topologyRanks[index];
        if (topologyRank >= ASYM_RANK_SIZE) {
            return CCU_E_PARA;
        }
        const uint16_t rankBit = EventBit(topologyRank);
        if ((topologyMask & rankBit) != 0) {
            return CCU_E_PARA;
        }
        topologyMask = static_cast<uint16_t>(topologyMask | rankBit);
    }

    const uint32_t selfTopologyIndex =
        AsymServerOffset(arg->serverIndex) + arg->localIndex;
    if (arg->topologyRanks[selfTopologyIndex] != arg->rankId) {
        return CCU_E_PARA;
    }
    const uint32_t expectedProxyRank = arg->serverIndex == 0
        ? arg->topologyRanks[ASYM_LARGE_RANK_COUNT + arg->localIndex / 2U]
        : arg->topologyRanks[arg->localIndex];
    if (arg->remoteProxyRank != expectedProxyRank) {
        return CCU_E_PARA;
    }

    uint16_t peerMask = 0;
    for (uint32_t channelIndex = 0; channelIndex < arg->channelCount; ++channelIndex) {
        const uint32_t peerRank = arg->peerRanks[channelIndex];
        if (peerRank >= ASYM_RANK_SIZE || peerRank == arg->rankId) {
            return CCU_E_PARA;
        }
        const uint16_t peerBit = EventBit(peerRank);
        if ((peerMask & peerBit) != 0) {
            return CCU_E_PARA;
        }
        peerMask = static_cast<uint16_t>(peerMask | peerBit);

        const uint32_t peerTopologyIndex = FindAsymTopologyIndex(arg, peerRank);
        if (peerTopologyIndex >= ASYM_RANK_SIZE) {
            return CCU_E_PARA;
        }
        const uint32_t peerServerIndex =
            peerTopologyIndex < ASYM_LARGE_RANK_COUNT ? 0U : 1U;
        if ((isMesh && peerServerIndex != arg->serverIndex) ||
            (!isMesh && peerServerIndex == arg->serverIndex)) {
            return CCU_E_PARA;
        }

        const ChannelHandle channel = arg->channels[channelIndex];
        ctx.remoteBaseAddr[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(channel, INPUT_ADDR_XN_ID);
        ctx.remoteBaseToken[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(channel, INPUT_TOKEN_XN_ID);
    }
    if (!isMesh &&
        FindAsymChannelIndex(arg, arg->remoteProxyRank) >= arg->channelCount) {
        return CCU_E_PARA;
    }
    return CCU_SUCCESS;
}

CcuResult LoadAsymTaskArgs(ReduceScatterAsymBaseContext &ctx)
{
    uint32_t argId = 0;
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.task.inputAddr, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.task.outputAddr, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.task.inputToken, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.task.outputToken, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.task.scratchAddr, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.task.scratchToken, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.task.recvBytes, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.task.splitABytes, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.task.aLaneBytes, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.task.aLastLaneBytes, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.task.bCrossLaneBytes, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.task.bCrossLastLaneBytes, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.task.bMeshLaneBytes, argId++));
    OPS_CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.task.bMeshLastLaneBytes, argId++));
    return CCU_SUCCESS;
}

CcuResult AsymPreSync(const ReduceScatterAsymBaseContext &ctx, bool publishScratch)
{
    const CcuReduceScatterAsymKernelArg *arg = ctx.arg;
    for (uint32_t channelIndex = 0; channelIndex < arg->channelCount; ++channelIndex) {
        const ChannelHandle channel = arg->channels[channelIndex];
        const ccu::Variable &addr =
            publishScratch ? ctx.task.scratchAddr : ctx.task.inputAddr;
        const ccu::Variable &token =
            publishScratch ? ctx.task.scratchToken : ctx.task.inputToken;
        OPS_CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(channel, addr,
            INPUT_ADDR_XN_ID, CHANNEL_NOTIFY_INDEX, INPUT_ADDR_READY_MASK));
        OPS_CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(channel, token,
            INPUT_TOKEN_XN_ID, CHANNEL_NOTIFY_INDEX, INPUT_TOKEN_READY_MASK));
    }
    for (uint32_t channelIndex = 0; channelIndex < arg->channelCount; ++channelIndex) {
        OPS_CCU_RETURN_IF_ERROR(
            ccu::NotifyWait(arg->channels[channelIndex], CHANNEL_NOTIFY_INDEX, PRE_SYNC_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult AsymPostSync(const ReduceScatterAsymBaseContext &ctx)
{
    const CcuReduceScatterAsymKernelArg *arg = ctx.arg;
    for (uint32_t channelIndex = 0; channelIndex < arg->channelCount; ++channelIndex) {
        OPS_CCU_RETURN_IF_ERROR(
            ccu::NotifyRecord(arg->channels[channelIndex], CHANNEL_NOTIFY_INDEX, POST_SYNC_MASK));
    }
    for (uint32_t channelIndex = 0; channelIndex < arg->channelCount; ++channelIndex) {
        OPS_CCU_RETURN_IF_ERROR(
            ccu::NotifyWait(arg->channels[channelIndex], CHANNEL_NOTIFY_INDEX, POST_SYNC_MASK));
    }
    return CCU_SUCCESS;
}

void BuildAsymLaneOffsets(ccu::Variable (&offsets)[ASYM_MAX_CHANNEL_COUNT],
    const ccu::Variable &laneBytes, uint32_t laneCount)
{
    offsets[0] = 0;
    for (uint32_t lane = 1; lane < laneCount; ++lane) {
        offsets[lane] = offsets[lane - 1] + laneBytes;
    }
}

void BuildAsymBBytes(ccu::Variable &bBytes,
    const ReduceScatterAsymBaseContext &base)
{
    bBytes = base.task.bCrossLastLaneBytes;
    for (uint32_t lane = 1; lane < base.arg->crossRankCount; ++lane) {
        bBytes = bBytes + base.task.bCrossLaneBytes;
    }
}

struct AsymMeshAStage1Context {
    ReduceScatterAsymBaseContext base;
    uint32_t targetCount = 0;
    uint32_t targetTopologyIndex[ASYM_MAX_A_TARGET_COUNT] = {};
    ccu::Variable targetInputOffset[ASYM_MAX_A_TARGET_COUNT];
    ccu::Variable laneOffset[ASYM_MAX_CHANNEL_COUNT];
    ccu::LocalAddr localInput[ASYM_MAX_A_TARGET_COUNT];
    ccu::LocalAddr partial[ASYM_MAX_A_TARGET_COUNT];
    ccu::LocalAddr reduceDst[ASYM_MAX_CHANNEL_COUNT];
    ccu::RemoteAddr reduceSrc[ASYM_MAX_CHANNEL_COUNT];
};

void InitAsymMeshAStage1Addresses(AsymMeshAStage1Context &ctx)
{
    const CcuReduceScatterAsymKernelArg *arg = ctx.base.arg;
    ReduceScatterAsymTaskArgs &task = ctx.base.task;
    const uint32_t localPeerCount = arg->localRankCount - 1U;
    BuildAsymLaneOffsets(ctx.laneOffset, task.aLaneBytes, localPeerCount);

    ctx.targetCount = 1;
    ctx.targetTopologyIndex[0] =
        AsymServerOffset(arg->serverIndex) + arg->localIndex;
    if (arg->serverIndex == 0 && arg->localIndex < ASYM_SMALL_RANK_COUNT) {
        ctx.targetTopologyIndex[ctx.targetCount++] =
            ASYM_LARGE_RANK_COUNT + arg->localIndex;
    } else if (arg->serverIndex == 1) {
        ctx.targetTopologyIndex[ctx.targetCount++] = arg->localIndex * 2U;
        ctx.targetTopologyIndex[ctx.targetCount++] = arg->localIndex * 2U + 1U;
    }

    for (uint32_t target = 0; target < ctx.targetCount; ++target) {
        const uint32_t targetRank =
            arg->topologyRanks[ctx.targetTopologyIndex[target]];
        BuildRepeatedOffset(
            ctx.targetInputOffset[target], task.recvBytes, targetRank);
        ctx.localInput[target].addr = task.inputAddr;
        ctx.localInput[target].addr += ctx.targetInputOffset[target];
        ctx.localInput[target].token = task.inputToken;

        if (target == 0) {
            ctx.partial[target].addr = task.outputAddr;
            ctx.partial[target].token = task.outputToken;
        } else {
            ccu::Variable scratchTargetOffset;
            BuildRepeatedOffset(scratchTargetOffset, task.splitABytes,
                ctx.targetTopologyIndex[target]);
            ctx.partial[target].addr = task.scratchAddr;
            ctx.partial[target].addr += scratchTargetOffset;
            ctx.partial[target].token = task.scratchToken;
        }
    }
}

CcuResult RunAsymMeshAStage1(AsymMeshAStage1Context &ctx)
{
    const CcuReduceScatterAsymKernelArg *arg = ctx.base.arg;
    ReduceScatterAsymTaskArgs &task = ctx.base.task;
    const uint32_t localPeerCount = arg->localRankCount - 1U;
    const uint16_t allPeerMask = AsymChannelMask(localPeerCount);

    CCU_IF(task.splitABytes != 0) {
        for (uint32_t target = 0; target < ctx.targetCount; ++target) {
            OPS_CCU_RETURN_IF_ERROR(ccu::LocalCopy(ctx.partial[target],
                ctx.localInput[target], task.splitABytes, ctx.base.event, EventBit(0)));
            OPS_CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.base.event, EventBit(0)));

            for (uint32_t step = 0; step < localPeerCount; ++step) {
                for (uint32_t channelIndex = 0;
                    channelIndex < localPeerCount; ++channelIndex) {
                    const uint32_t lane =
                        (channelIndex + step) % localPeerCount;
                    ctx.reduceDst[channelIndex].addr = ctx.partial[target].addr;
                    ctx.reduceDst[channelIndex].addr += ctx.laneOffset[lane];
                    ctx.reduceDst[channelIndex].token = ctx.partial[target].token;
                    ctx.reduceSrc[channelIndex].addr =
                        ctx.base.remoteBaseAddr[channelIndex];
                    ctx.reduceSrc[channelIndex].addr +=
                        ctx.targetInputOffset[target];
                    ctx.reduceSrc[channelIndex].addr += ctx.laneOffset[lane];
                    ctx.reduceSrc[channelIndex].token =
                        ctx.base.remoteBaseToken[channelIndex];
                    const ccu::Variable &laneBytes =
                        lane + 1U == localPeerCount
                        ? task.aLastLaneBytes : task.aLaneBytes;
                    OPS_CCU_RETURN_IF_ERROR(ccu::ReadReduce(
                        arg->channels[channelIndex], ctx.reduceDst[channelIndex],
                        ctx.reduceSrc[channelIndex], laneBytes, arg->dataType,
                        arg->reduceOp, ctx.base.event, EventBit(channelIndex)));
                }
                OPS_CCU_RETURN_IF_ERROR(
                    ccu::EventWait(ctx.base.event, allPeerMask));
            }
        }
    }
    return CCU_SUCCESS;
}

struct AsymNhrBStage1Context {
    ReduceScatterAsymBaseContext base;
    ccu::Variable bBytes;
    ccu::Variable targetInputOffset;
    ccu::Variable laneOffset[ASYM_MAX_CHANNEL_COUNT];
    ccu::LocalAddr localInput;
    ccu::LocalAddr partial;
    ccu::LocalAddr reduceDst[ASYM_MAX_CHANNEL_COUNT];
    ccu::RemoteAddr reduceSrc[ASYM_MAX_CHANNEL_COUNT];
};

void InitAsymNhrBStage1Addresses(AsymNhrBStage1Context &ctx)
{
    const CcuReduceScatterAsymKernelArg *arg = ctx.base.arg;
    const ReduceScatterAsymTaskArgs &task = ctx.base.task;
    BuildAsymBBytes(ctx.bBytes, ctx.base);
    BuildAsymLaneOffsets(
        ctx.laneOffset, task.bCrossLaneBytes, arg->crossRankCount);

    const uint32_t selfTopologyIndex =
        AsymServerOffset(arg->serverIndex) + arg->localIndex;
    const uint32_t targetRank = arg->topologyRanks[selfTopologyIndex];
    BuildRepeatedOffset(ctx.targetInputOffset, task.recvBytes, targetRank);
    ctx.targetInputOffset = ctx.targetInputOffset + task.splitABytes;
    ctx.localInput.addr = task.inputAddr;
    ctx.localInput.addr += ctx.targetInputOffset;
    ctx.localInput.token = task.inputToken;
    ctx.partial.addr = task.outputAddr;
    ctx.partial.addr += task.splitABytes;
    ctx.partial.token = task.outputToken;
}

CcuResult RunAsymNhrBStage1(AsymNhrBStage1Context &ctx)
{
    const CcuReduceScatterAsymKernelArg *arg = ctx.base.arg;
    const ReduceScatterAsymTaskArgs &task = ctx.base.task;
    const uint16_t allPeerMask = AsymChannelMask(arg->crossRankCount);

    CCU_IF(ctx.bBytes != 0) {
        OPS_CCU_RETURN_IF_ERROR(ccu::LocalCopy(ctx.partial,
            ctx.localInput, ctx.bBytes, ctx.base.event, EventBit(0)));
        OPS_CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.base.event, EventBit(0)));

        for (uint32_t step = 0; step < arg->crossRankCount; ++step) {
            for (uint32_t channelIndex = 0;
                channelIndex < arg->crossRankCount; ++channelIndex) {
                const uint32_t lane =
                    (channelIndex + step) % arg->crossRankCount;
                ctx.reduceDst[channelIndex].addr = ctx.partial.addr;
                ctx.reduceDst[channelIndex].addr += ctx.laneOffset[lane];
                ctx.reduceDst[channelIndex].token = ctx.partial.token;
                ctx.reduceSrc[channelIndex].addr =
                    ctx.base.remoteBaseAddr[channelIndex];
                ctx.reduceSrc[channelIndex].addr += ctx.targetInputOffset;
                ctx.reduceSrc[channelIndex].addr += ctx.laneOffset[lane];
                ctx.reduceSrc[channelIndex].token =
                    ctx.base.remoteBaseToken[channelIndex];
                const ccu::Variable &laneBytes =
                    lane + 1U == arg->crossRankCount
                    ? task.bCrossLastLaneBytes : task.bCrossLaneBytes;
                OPS_CCU_RETURN_IF_ERROR(ccu::ReadReduce(
                    arg->channels[channelIndex], ctx.reduceDst[channelIndex],
                    ctx.reduceSrc[channelIndex], laneBytes, arg->dataType,
                    arg->reduceOp, ctx.base.event, EventBit(channelIndex)));
            }
            OPS_CCU_RETURN_IF_ERROR(
                ccu::EventWait(ctx.base.event, allPeerMask));
        }
    }
    return CCU_SUCCESS;
}

struct AsymMeshBStage2Context {
    ReduceScatterAsymBaseContext base;
    ccu::Variable bBytes;
    ccu::Variable targetInputOffset;
    ccu::Variable laneOffset[ASYM_MAX_CHANNEL_COUNT];
    ccu::LocalAddr output;
    ccu::LocalAddr reduceDst[ASYM_MAX_CHANNEL_COUNT];
    ccu::RemoteAddr reduceSrc[ASYM_MAX_CHANNEL_COUNT];
};

void InitAsymMeshBStage2Addresses(AsymMeshBStage2Context &ctx)
{
    const CcuReduceScatterAsymKernelArg *arg = ctx.base.arg;
    const ReduceScatterAsymTaskArgs &task = ctx.base.task;
    BuildAsymBBytes(ctx.bBytes, ctx.base);
    const uint32_t selfTopologyIndex =
        AsymServerOffset(arg->serverIndex) + arg->localIndex;
    const uint32_t targetRank = arg->topologyRanks[selfTopologyIndex];
    BuildRepeatedOffset(ctx.targetInputOffset, task.recvBytes, targetRank);
    ctx.targetInputOffset = ctx.targetInputOffset + task.splitABytes;
    BuildAsymLaneOffsets(ctx.laneOffset, task.bMeshLaneBytes,
        arg->localRankCount - 1U);

    ctx.output.addr = task.outputAddr;
    ctx.output.addr += task.splitABytes;
    ctx.output.token = task.outputToken;
}

CcuResult RunAsymMeshBStage2(AsymMeshBStage2Context &ctx)
{
    const CcuReduceScatterAsymKernelArg *arg = ctx.base.arg;
    const ReduceScatterAsymTaskArgs &task = ctx.base.task;
    const uint32_t localPeerCount = arg->localRankCount - 1U;
    const uint16_t allPeerMask = AsymChannelMask(localPeerCount);

    CCU_IF(ctx.bBytes != 0) {
        for (uint32_t step = 0; step < localPeerCount; ++step) {
            for (uint32_t channelIndex = 0;
                channelIndex < localPeerCount; ++channelIndex) {
                const uint32_t lane =
                    (channelIndex + step) % localPeerCount;
                ctx.reduceDst[channelIndex].addr = ctx.output.addr;
                ctx.reduceDst[channelIndex].addr += ctx.laneOffset[lane];
                ctx.reduceDst[channelIndex].token = task.outputToken;
                ctx.reduceSrc[channelIndex].addr =
                    ctx.base.remoteBaseAddr[channelIndex];
                ctx.reduceSrc[channelIndex].addr +=
                    ctx.targetInputOffset;
                ctx.reduceSrc[channelIndex].addr += ctx.laneOffset[lane];
                ctx.reduceSrc[channelIndex].token =
                    ctx.base.remoteBaseToken[channelIndex];
                const ccu::Variable &laneBytes =
                    lane + 1U == localPeerCount
                    ? task.bMeshLastLaneBytes : task.bMeshLaneBytes;
                OPS_CCU_RETURN_IF_ERROR(ccu::ReadReduce(
                    arg->channels[channelIndex], ctx.reduceDst[channelIndex],
                    ctx.reduceSrc[channelIndex], laneBytes, arg->dataType,
                    arg->reduceOp, ctx.base.event, EventBit(channelIndex)));
            }
            OPS_CCU_RETURN_IF_ERROR(
                ccu::EventWait(ctx.base.event, allPeerMask));
        }
    }
    return CCU_SUCCESS;
}

struct AsymNhrAStage2Context {
    ReduceScatterAsymBaseContext base;
    uint32_t proxyChannelIndex = 0;
    ccu::Variable scratchTargetOffset;
    ccu::LocalAddr output;
    ccu::RemoteAddr remoteScratch;
};

void InitAsymNhrAStage2Addresses(AsymNhrAStage2Context &ctx)
{
    const CcuReduceScatterAsymKernelArg *arg = ctx.base.arg;
    const ReduceScatterAsymTaskArgs &task = ctx.base.task;
    ctx.proxyChannelIndex =
        FindAsymChannelIndex(arg, arg->remoteProxyRank);
    const uint32_t selfTopologyIndex =
        AsymServerOffset(arg->serverIndex) + arg->localIndex;
    BuildRepeatedOffset(
        ctx.scratchTargetOffset, task.splitABytes, selfTopologyIndex);
    ctx.output.addr = task.outputAddr;
    ctx.output.token = task.outputToken;
    ctx.remoteScratch.addr =
        ctx.base.remoteBaseAddr[ctx.proxyChannelIndex];
    ctx.remoteScratch.addr += ctx.scratchTargetOffset;
    ctx.remoteScratch.token =
        ctx.base.remoteBaseToken[ctx.proxyChannelIndex];
}

CcuResult RunAsymNhrAStage2(AsymNhrAStage2Context &ctx)
{
    const CcuReduceScatterAsymKernelArg *arg = ctx.base.arg;
    ReduceScatterAsymTaskArgs &task = ctx.base.task;
    CCU_IF(task.splitABytes != 0) {
        OPS_CCU_RETURN_IF_ERROR(ccu::ReadReduce(
            arg->channels[ctx.proxyChannelIndex], ctx.output,
            ctx.remoteScratch, task.splitABytes, arg->dataType,
            arg->reduceOp, ctx.base.event, EventBit(0)));
        OPS_CCU_RETURN_IF_ERROR(
            ccu::EventWait(ctx.base.event, EventBit(0)));
    }
    return CCU_SUCCESS;
}

} // namespace

CcuResult CcuKernel(CcuKernelArg arg)
{
    if (arg == nullptr) {
        return CCU_E_PTR;
    }

    ReduceScatterContext ctx;
    ctx.arg = static_cast<CcuReduceScatterKernelArg *>(arg);

    OPS_CCU_RETURN_IF_ERROR(InitResource(ctx));
    OPS_CCU_RETURN_IF_ERROR(LoadArgs(ctx));
    OPS_CCU_RETURN_IF_ERROR(PreSync(ctx));
    OPS_CCU_RETURN_IF_ERROR(DoRepeat(ctx));
    OPS_CCU_RETURN_IF_ERROR(PostSync(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterMainKernel(CcuKernelArg arg)
{
    if (arg == nullptr) {
        return CCU_E_PTR;
    }

    SplitReduceScatterContext ctx;
    ctx.arg = static_cast<CcuReduceScatterKernelArg *>(arg);

    OPS_CCU_RETURN_IF_ERROR(InitSplitResource(ctx, false));
    OPS_CCU_RETURN_IF_ERROR(LoadSplitArgs(ctx));
    OPS_CCU_RETURN_IF_ERROR(SplitPreSync(ctx));
    InitSplitAddresses(ctx);
    OPS_CCU_RETURN_IF_ERROR(StageSplitGroup(ctx, true));
    OPS_CCU_RETURN_IF_ERROR(SplitPostSync(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterWorkerKernel(CcuKernelArg arg)
{
    if (arg == nullptr) {
        return CCU_E_PTR;
    }

    SplitReduceScatterContext ctx;
    ctx.arg = static_cast<CcuReduceScatterKernelArg *>(arg);

    OPS_CCU_RETURN_IF_ERROR(InitSplitResource(ctx, true));
    OPS_CCU_RETURN_IF_ERROR(LoadSplitArgs(ctx));
    OPS_CCU_RETURN_IF_ERROR(SplitPreSync(ctx));
    InitSplitAddresses(ctx);
    OPS_CCU_RETURN_IF_ERROR(StageSplitGroup(ctx, false));
    OPS_CCU_RETURN_IF_ERROR(SplitPostSync(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterReduceKernel(CcuKernelArg arg)
{
    if (arg == nullptr) {
        return CCU_E_PTR;
    }

    SplitReduceScatterContext ctx;
    ctx.arg = static_cast<CcuReduceScatterKernelArg *>(arg);

    // Resolving the main-layer channels keeps this local-only reduction on
    // the same IO Die as the main staging kernel.
    OPS_CCU_RETURN_IF_ERROR(InitSplitResource(ctx, false));
    OPS_CCU_RETURN_IF_ERROR(LoadSplitArgs(ctx));
    InitSplitAddresses(ctx);
    OPS_CCU_RETURN_IF_ERROR(ReduceSplitTree(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatter2DMeshAStage1Kernel(CcuKernelArg arg)
{
    if (arg == nullptr) {
        return CCU_E_PTR;
    }

    MeshAStage1Context ctx;
    ctx.base.arg = static_cast<CcuReduceScatter2DKernelArg *>(arg);
    OPS_CCU_RETURN_IF_ERROR(Init2DResource(ctx.base, true));
    OPS_CCU_RETURN_IF_ERROR(Load2DTaskArgs(ctx.base));
    OPS_CCU_RETURN_IF_ERROR(TwoDPreSync(ctx.base, false));
    InitMeshAStage1Addresses(ctx);
    OPS_CCU_RETURN_IF_ERROR(RunMeshAStage1(ctx));
    OPS_CCU_RETURN_IF_ERROR(TwoDPostSync(ctx.base));
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatter2DNhrBStage1Kernel(CcuKernelArg arg)
{
    if (arg == nullptr) {
        return CCU_E_PTR;
    }

    NhrBStage1Context ctx;
    ctx.base.arg = static_cast<CcuReduceScatter2DKernelArg *>(arg);
    OPS_CCU_RETURN_IF_ERROR(Init2DResource(ctx.base, false));
    OPS_CCU_RETURN_IF_ERROR(Load2DTaskArgs(ctx.base));
    OPS_CCU_RETURN_IF_ERROR(TwoDPreSync(ctx.base, false));
    InitNhrBStage1Addresses(ctx);
    OPS_CCU_RETURN_IF_ERROR(RunNhrBStage1(ctx));
    OPS_CCU_RETURN_IF_ERROR(TwoDPostSync(ctx.base));
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatter2DMeshBStage2Kernel(CcuKernelArg arg)
{
    if (arg == nullptr) {
        return CCU_E_PTR;
    }

    MeshBStage2Context ctx;
    ctx.base.arg = static_cast<CcuReduceScatter2DKernelArg *>(arg);
    OPS_CCU_RETURN_IF_ERROR(Init2DResource(ctx.base, true));
    OPS_CCU_RETURN_IF_ERROR(Load2DTaskArgs(ctx.base));
    OPS_CCU_RETURN_IF_ERROR(TwoDPreSync(ctx.base, true));
    InitMeshBStage2Addresses(ctx);
    OPS_CCU_RETURN_IF_ERROR(RunMeshBStage2(ctx));
    OPS_CCU_RETURN_IF_ERROR(TwoDPostSync(ctx.base));
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatter2DNhrAStage2Kernel(CcuKernelArg arg)
{
    if (arg == nullptr) {
        return CCU_E_PTR;
    }

    NhrAStage2Context ctx;
    ctx.base.arg = static_cast<CcuReduceScatter2DKernelArg *>(arg);
    OPS_CCU_RETURN_IF_ERROR(Init2DResource(ctx.base, false));
    OPS_CCU_RETURN_IF_ERROR(Load2DTaskArgs(ctx.base));
    OPS_CCU_RETURN_IF_ERROR(TwoDPreSync(ctx.base, true));
    InitNhrAStage2Addresses(ctx);
    OPS_CCU_RETURN_IF_ERROR(RunNhrAStage2(ctx));
    OPS_CCU_RETURN_IF_ERROR(TwoDPostSync(ctx.base));
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatter4x1LatinKernel(CcuKernelArg arg)
{
    if (arg == nullptr) {
        return CCU_E_PTR;
    }

    Latin4x1Context ctx;
    ctx.arg = static_cast<CcuReduceScatterKernelArg *>(arg);

    OPS_CCU_RETURN_IF_ERROR(InitLatin4x1Resource(ctx));
    OPS_CCU_RETURN_IF_ERROR(LoadLatin4x1Args(ctx));
    OPS_CCU_RETURN_IF_ERROR(Latin4x1PreSync(ctx));
    InitLatin4x1Addresses(ctx);
    OPS_CCU_RETURN_IF_ERROR(RunLatin4x1(ctx));
    OPS_CCU_RETURN_IF_ERROR(Latin4x1PostSync(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterAsymMeshAStage1Kernel(CcuKernelArg arg)
{
    if (arg == nullptr) {
        return CCU_E_PTR;
    }
    AsymMeshAStage1Context ctx;
    ctx.base.arg = static_cast<CcuReduceScatterAsymKernelArg *>(arg);
    OPS_CCU_RETURN_IF_ERROR(InitAsymResource(ctx.base, true));
    OPS_CCU_RETURN_IF_ERROR(LoadAsymTaskArgs(ctx.base));
    OPS_CCU_RETURN_IF_ERROR(AsymPreSync(ctx.base, false));
    InitAsymMeshAStage1Addresses(ctx);
    OPS_CCU_RETURN_IF_ERROR(RunAsymMeshAStage1(ctx));
    OPS_CCU_RETURN_IF_ERROR(AsymPostSync(ctx.base));
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterAsymNhrBStage1Kernel(CcuKernelArg arg)
{
    if (arg == nullptr) {
        return CCU_E_PTR;
    }
    AsymNhrBStage1Context ctx;
    ctx.base.arg = static_cast<CcuReduceScatterAsymKernelArg *>(arg);
    OPS_CCU_RETURN_IF_ERROR(InitAsymResource(ctx.base, false));
    OPS_CCU_RETURN_IF_ERROR(LoadAsymTaskArgs(ctx.base));
    OPS_CCU_RETURN_IF_ERROR(AsymPreSync(ctx.base, false));
    InitAsymNhrBStage1Addresses(ctx);
    OPS_CCU_RETURN_IF_ERROR(RunAsymNhrBStage1(ctx));
    OPS_CCU_RETURN_IF_ERROR(AsymPostSync(ctx.base));
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterAsymMeshBStage2Kernel(CcuKernelArg arg)
{
    if (arg == nullptr) {
        return CCU_E_PTR;
    }
    AsymMeshBStage2Context ctx;
    ctx.base.arg = static_cast<CcuReduceScatterAsymKernelArg *>(arg);
    OPS_CCU_RETURN_IF_ERROR(InitAsymResource(ctx.base, true));
    OPS_CCU_RETURN_IF_ERROR(LoadAsymTaskArgs(ctx.base));
    OPS_CCU_RETURN_IF_ERROR(AsymPreSync(ctx.base, false));
    InitAsymMeshBStage2Addresses(ctx);
    OPS_CCU_RETURN_IF_ERROR(RunAsymMeshBStage2(ctx));
    OPS_CCU_RETURN_IF_ERROR(AsymPostSync(ctx.base));
    return CCU_SUCCESS;
}

CcuResult CcuReduceScatterAsymNhrAStage2Kernel(CcuKernelArg arg)
{
    if (arg == nullptr) {
        return CCU_E_PTR;
    }
    AsymNhrAStage2Context ctx;
    ctx.base.arg = static_cast<CcuReduceScatterAsymKernelArg *>(arg);
    OPS_CCU_RETURN_IF_ERROR(InitAsymResource(ctx.base, false));
    OPS_CCU_RETURN_IF_ERROR(LoadAsymTaskArgs(ctx.base));
    OPS_CCU_RETURN_IF_ERROR(AsymPreSync(ctx.base, true));
    InitAsymNhrAStage2Addresses(ctx);
    OPS_CCU_RETURN_IF_ERROR(RunAsymNhrAStage2(ctx));
    OPS_CCU_RETURN_IF_ERROR(AsymPostSync(ctx.base));
    return CCU_SUCCESS;
}

#undef OPS_CCU_RETURN_IF_ERROR

} // namespace ops_hccl
