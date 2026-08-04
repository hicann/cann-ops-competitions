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

#include <ccu/ccu_primitives.hpp>

#include "ccu_kernel.h"

namespace ccu = ::AscendC::ccu;

namespace ops_hccl {
namespace {
constexpr uint32_t INPUT_ADDR_XN_ID = 0;
constexpr uint32_t INPUT_TOKEN_XN_ID = 1;
constexpr uint32_t PRE_SYNC_NOTIFY_IDX = 0;
constexpr uint16_t INPUT_ADDR_BIT = 1U << INPUT_ADDR_XN_ID;
constexpr uint16_t INPUT_TOKEN_BIT = 1U << INPUT_TOKEN_XN_ID;
constexpr uint16_t PRE_SYNC_MASK = INPUT_ADDR_BIT | INPUT_TOKEN_BIT;
constexpr uint16_t POST_SYNC_BIT = 1U << 2;

struct ReduceScatterContext {
    ccu::Variable inputBase;
    ccu::Variable inputToken[MAX_RANK_SIZE];
    ccu::Variable remoteInputBase[MAX_RANK_SIZE];
    ccu::Variable outputBase;
    ccu::Variable outputToken;
    ccu::Variable scratchBase;
    ccu::Variable scratchToken;
    ccu::Variable inputSliceOffset;
    ccu::Variable outputOffset;
    ccu::Variable chunkBytes;
    ChannelHandle channelByRank[MAX_RANK_SIZE]{};
    ccu::LocalAddr localInput;
    ccu::RemoteAddr remoteInput[MAX_RANK_SIZE];
    ccu::LocalAddr scratch[MAX_RANK_SIZE];
    ccu::LocalAddr output;
    ccu::Event gatherEvent;
    ccu::Event reduceEvent;
};

CcuResult BindRemoteResources(ReduceScatterContext &ctx, const ReduceScatterKernelArg &kernelArg)
{
    uint32_t channelIndex = 0;
    for (uint32_t peerRank = 0; peerRank < kernelArg.rankSize; ++peerRank) {
        if (peerRank == kernelArg.rankId) {
            continue;
        }
        if (channelIndex >= kernelArg.channelCount) {
            return CCU_E_INTERNAL;
        }
        const ChannelHandle channel = kernelArg.channels[channelIndex++];
        ctx.channelByRank[peerRank] = channel;
        ctx.remoteInputBase[peerRank] =
            ccu::GetResByChannel<ccu::Variable>(channel, INPUT_ADDR_XN_ID);
        ctx.inputToken[peerRank] =
            ccu::GetResByChannel<ccu::Variable>(channel, INPUT_TOKEN_XN_ID);
    }
    return channelIndex == kernelArg.channelCount ? CCU_SUCCESS : CCU_E_INTERNAL;
}

CcuResult LoadTaskArgs(ReduceScatterContext &ctx, uint32_t rankId)
{
    constexpr uint32_t TOTAL_TASK_ARGS = 9;
    uint32_t argId = 0;
    CcuResult ret = ccu::LoadArg(ctx.inputBase, argId++);
    if (ret != CCU_SUCCESS) {
        return ret;
    }
    ret = ccu::LoadArg(ctx.inputToken[rankId], argId++);
    if (ret != CCU_SUCCESS) {
        return ret;
    }
    ret = ccu::LoadArg(ctx.outputBase, argId++);
    if (ret != CCU_SUCCESS) {
        return ret;
    }
    ret = ccu::LoadArg(ctx.outputToken, argId++);
    if (ret != CCU_SUCCESS) {
        return ret;
    }
    ret = ccu::LoadArg(ctx.scratchBase, argId++);
    if (ret != CCU_SUCCESS) {
        return ret;
    }
    ret = ccu::LoadArg(ctx.scratchToken, argId++);
    if (ret != CCU_SUCCESS) {
        return ret;
    }
    ret = ccu::LoadArg(ctx.inputSliceOffset, argId++);
    if (ret != CCU_SUCCESS) {
        return ret;
    }
    ret = ccu::LoadArg(ctx.outputOffset, argId++);
    if (ret != CCU_SUCCESS) {
        return ret;
    }
    ret = ccu::LoadArg(ctx.chunkBytes, argId++);
    if (ret != CCU_SUCCESS) {
        return ret;
    }
    if (argId != TOTAL_TASK_ARGS) {
        return CCU_E_INTERNAL;
    }
    return CCU_SUCCESS;
}

CcuResult PreSync(ReduceScatterContext &ctx, const ReduceScatterKernelArg &kernelArg)
{
    for (uint32_t channelIndex = 0; channelIndex < kernelArg.channelCount; ++channelIndex) {
        CcuResult ret = ccu::WriteVariableWithNotify(kernelArg.channels[channelIndex], ctx.inputBase,
            INPUT_ADDR_XN_ID, PRE_SYNC_NOTIFY_IDX, INPUT_ADDR_BIT);
        if (ret != CCU_SUCCESS) {
            return ret;
        }
        ret = ccu::WriteVariableWithNotify(kernelArg.channels[channelIndex], ctx.inputToken[kernelArg.rankId],
            INPUT_TOKEN_XN_ID, PRE_SYNC_NOTIFY_IDX, INPUT_TOKEN_BIT);
        if (ret != CCU_SUCCESS) {
            return ret;
        }
    }
    for (uint32_t channelIndex = 0; channelIndex < kernelArg.channelCount; ++channelIndex) {
        const CcuResult ret =
            ccu::NotifyWait(kernelArg.channels[channelIndex], PRE_SYNC_NOTIFY_IDX, PRE_SYNC_MASK);
        if (ret != CCU_SUCCESS) {
            return ret;
        }
    }
    return CCU_SUCCESS;
}

void PrepareAddresses(ReduceScatterContext &ctx, const ReduceScatterKernelArg &kernelArg)
{
    ctx.localInput.addr = ctx.inputBase;
    ctx.localInput.addr += ctx.inputSliceOffset;
    ctx.localInput.token = ctx.inputToken[kernelArg.rankId];

    ccu::Variable scratchOffset;
    scratchOffset = 0;
    for (uint32_t rank = 0; rank < kernelArg.rankSize; ++rank) {
        ctx.scratch[rank].addr = ctx.scratchBase;
        ctx.scratch[rank].addr += scratchOffset;
        ctx.scratch[rank].token = ctx.scratchToken;
        scratchOffset += ctx.chunkBytes;
    }

    for (uint32_t peerRank = 0; peerRank < kernelArg.rankSize; ++peerRank) {
        if (peerRank == kernelArg.rankId) {
            continue;
        }
        ctx.remoteInput[peerRank].addr = ctx.remoteInputBase[peerRank];
        ctx.remoteInput[peerRank].addr += ctx.inputSliceOffset;
        ctx.remoteInput[peerRank].token = ctx.inputToken[peerRank];
    }

    ctx.output.addr = ctx.outputBase;
    ctx.output.addr += ctx.outputOffset;
    ctx.output.token = ctx.outputToken;
}

CcuResult GatherAllContributions(ReduceScatterContext &ctx, const ReduceScatterKernelArg &kernelArg)
{
    for (uint32_t peerRank = 0; peerRank < kernelArg.rankSize; ++peerRank) {
        const uint16_t eventBit = static_cast<uint16_t>(1U << peerRank);
        CcuResult ret = CCU_SUCCESS;
        if (peerRank == kernelArg.rankId) {
            ret = ccu::LocalCopy(
                ctx.scratch[peerRank], ctx.localInput, ctx.chunkBytes, ctx.gatherEvent, eventBit);
        } else {
            ret = ccu::Read(ctx.channelByRank[peerRank], ctx.scratch[peerRank],
                ctx.remoteInput[peerRank], ctx.chunkBytes, ctx.gatherEvent, eventBit);
        }
        if (ret != CCU_SUCCESS) {
            return ret;
        }
    }

    const uint16_t allRanksMask = kernelArg.rankSize == 16
        ? static_cast<uint16_t>(0xFFFF)
        : static_cast<uint16_t>((1U << kernelArg.rankSize) - 1U);
    return ccu::EventWait(ctx.gatherEvent, allRanksMask);
}

CcuResult ReduceInFixedOrder(ReduceScatterContext &ctx, const ReduceScatterKernelArg &kernelArg)
{
    uint32_t remainingPieces = kernelArg.rankSize;
    while (remainingPieces > 1) {
        const uint32_t reducePieces = remainingPieces / 2;
        const uint32_t sourceIndex = remainingPieces - reducePieces;
        ccu::Variable reduceBytes;
        reduceBytes = ctx.chunkBytes;
        for (uint32_t index = 1; index < reducePieces; ++index) {
            reduceBytes += ctx.chunkBytes;
        }
        CcuResult ret = ccu::LocalReduce(ctx.scratch[0], ctx.scratch[sourceIndex], reduceBytes,
            HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, ctx.reduceEvent, 1);
        if (ret != CCU_SUCCESS) {
            return ret;
        }
        ret = ccu::EventWait(ctx.reduceEvent, 1);
        if (ret != CCU_SUCCESS) {
            return ret;
        }
        remainingPieces -= reducePieces;
    }

    CcuResult ret = ccu::LocalCopy(ctx.output, ctx.scratch[0], ctx.chunkBytes, ctx.reduceEvent, 1);
    if (ret != CCU_SUCCESS) {
        return ret;
    }
    return ccu::EventWait(ctx.reduceEvent, 1);
}

CcuResult PostSync(const ReduceScatterKernelArg &kernelArg)
{
    for (uint32_t channelIndex = 0; channelIndex < kernelArg.channelCount; ++channelIndex) {
        const CcuResult ret =
            ccu::NotifyRecord(kernelArg.channels[channelIndex], PRE_SYNC_NOTIFY_IDX, POST_SYNC_BIT);
        if (ret != CCU_SUCCESS) {
            return ret;
        }
    }
    for (uint32_t channelIndex = 0; channelIndex < kernelArg.channelCount; ++channelIndex) {
        const CcuResult ret =
            ccu::NotifyWait(kernelArg.channels[channelIndex], PRE_SYNC_NOTIFY_IDX, POST_SYNC_BIT);
        if (ret != CCU_SUCCESS) {
            return ret;
        }
    }
    return CCU_SUCCESS;
}
} // namespace

CcuResult CcuReduceScatterKernel(CcuKernelArg arg)
{
    if (arg == nullptr) {
        return CCU_E_PTR;
    }
    const auto *kernelArg = static_cast<const ReduceScatterKernelArg *>(arg);
    if (kernelArg->rankSize == 0 || kernelArg->rankSize > MAX_RANK_SIZE ||
        kernelArg->rankId >= kernelArg->rankSize || kernelArg->channelCount != kernelArg->rankSize - 1) {
        return CCU_E_PARA;
    }

    ReduceScatterContext ctx;
    CcuResult ret = BindRemoteResources(ctx, *kernelArg);
    if (ret != CCU_SUCCESS) {
        return ret;
    }
    ret = LoadTaskArgs(ctx, kernelArg->rankId);
    if (ret != CCU_SUCCESS) {
        return ret;
    }
    ret = PreSync(ctx, *kernelArg);
    if (ret != CCU_SUCCESS) {
        return ret;
    }

    CcuResult dataPathRet = CCU_SUCCESS;
    CCU_IF(ctx.chunkBytes != 0) {
        PrepareAddresses(ctx, *kernelArg);
        dataPathRet = GatherAllContributions(ctx, *kernelArg);
        if (dataPathRet == CCU_SUCCESS) {
            dataPathRet = ReduceInFixedOrder(ctx, *kernelArg);
        }
    }
    if (dataPathRet != CCU_SUCCESS) {
        return dataPathRet;
    }
    return PostSync(*kernelArg);
}
} // namespace ops_hccl
