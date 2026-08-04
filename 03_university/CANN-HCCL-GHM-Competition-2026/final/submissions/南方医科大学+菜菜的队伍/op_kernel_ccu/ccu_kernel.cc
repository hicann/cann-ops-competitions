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

#include "custom.h"
#include "ccu_kernel.h"

#define RETURN_IF_CCU_FAIL(expr)     \
    do {                             \
        CcuResult ccuCheckRet = (expr); \
        if (ccuCheckRet != CCU_SUCCESS) { \
            return ccuCheckRet;      \
        }                            \
    } while (0)

namespace ops_hccl {
namespace fabric = ::AscendC::ccu;
CcuResult CcuInboxExchange(CcuKernelArg arg)
{
    constexpr uint32_t kSlotAddrVar = 0;
    constexpr uint32_t kSlotTokenVar = 1;
    constexpr uint32_t kAddressNotify = 0;
    constexpr uint32_t kCompletionNotify = 1;
    constexpr uint16_t kAddressMask = (1U << kSlotAddrVar) | (1U << kSlotTokenVar);
    constexpr uint16_t kCompletionMask = 1U;

    auto *kernelArg = static_cast<ExchangeKernelSpec *>(arg);
    if (kernelArg == nullptr || kernelArg->laneCount == 0 || kernelArg->laneCount >= MAX_RANK_SIZE) {
        return CCU_E_PARA;
    }

    std::vector<fabric::Variable> peerInputAddr(kernelArg->laneCount);
    std::vector<fabric::Variable> localSlotAddr(kernelArg->laneCount);
    for (uint32_t channelIdx = 0; channelIdx < kernelArg->laneCount; ++channelIdx) {
        if (fabric::LoadArg(peerInputAddr[channelIdx], channelIdx) != CCU_SUCCESS ||
            fabric::LoadArg(localSlotAddr[channelIdx], kernelArg->laneCount + channelIdx) != CCU_SUCCESS) {
            return CCU_E_INTERNAL;
        }
    }
    fabric::Variable inputToken;
    fabric::Variable scratchToken;
    fabric::Variable chunkBytes;
    const uint32_t scalarArg = 2 * kernelArg->laneCount;
    if (fabric::LoadArg(inputToken, scalarArg) != CCU_SUCCESS ||
        fabric::LoadArg(scratchToken, scalarArg + 1) != CCU_SUCCESS ||
        fabric::LoadArg(chunkBytes, scalarArg + 2) != CCU_SUCCESS) {
        return CCU_E_INTERNAL;
    }

    std::vector<fabric::Variable> peerSlotAddr(kernelArg->laneCount);
    std::vector<fabric::Variable> peerSlotToken(kernelArg->laneCount);
    for (uint32_t channelIdx = 0; channelIdx < kernelArg->laneCount; ++channelIdx) {
        peerSlotAddr[channelIdx] =
            fabric::GetResByChannel<fabric::Variable>(kernelArg->lanes[channelIdx], kSlotAddrVar);
        peerSlotToken[channelIdx] =
            fabric::GetResByChannel<fabric::Variable>(kernelArg->lanes[channelIdx], kSlotTokenVar);
        if (fabric::WriteVariableWithNotify(kernelArg->lanes[channelIdx], localSlotAddr[channelIdx],
                kSlotAddrVar, kAddressNotify, 1U << kSlotAddrVar) != CCU_SUCCESS ||
            fabric::WriteVariableWithNotify(kernelArg->lanes[channelIdx], scratchToken,
                kSlotTokenVar, kAddressNotify, 1U << kSlotTokenVar) != CCU_SUCCESS) {
            return CCU_E_INTERNAL;
        }
    }
    for (uint32_t channelIdx = 0; channelIdx < kernelArg->laneCount; ++channelIdx) {
        if (fabric::NotifyWait(kernelArg->lanes[channelIdx], kAddressNotify, kAddressMask) != CCU_SUCCESS) {
            return CCU_E_INTERNAL;
        }
    }

    fabric::Event completion;
    uint16_t completionMask = 0;
    for (uint32_t channelIdx = 0; channelIdx < kernelArg->laneCount; ++channelIdx) {
        const uint16_t bit = static_cast<uint16_t>(1U << channelIdx);
        fabric::LocalAddr source;
        source.addr = peerInputAddr[channelIdx];
        source.token = inputToken;
        fabric::RemoteAddr destination;
        destination.addr = peerSlotAddr[channelIdx];
        destination.token = peerSlotToken[channelIdx];
        if (fabric::Write(kernelArg->lanes[channelIdx], destination, source, chunkBytes,
                completion, bit) != CCU_SUCCESS) {
            return CCU_E_INTERNAL;
        }
        completionMask = static_cast<uint16_t>(completionMask | bit);
    }
    if (fabric::EventWait(completion, completionMask) != CCU_SUCCESS) {
        return CCU_E_INTERNAL;
    }

    for (uint32_t channelIdx = 0; channelIdx < kernelArg->laneCount; ++channelIdx) {
        if (fabric::NotifyRecord(kernelArg->lanes[channelIdx], kCompletionNotify,
                kCompletionMask) != CCU_SUCCESS) {
            return CCU_E_INTERNAL;
        }
    }
    for (uint32_t channelIdx = 0; channelIdx < kernelArg->laneCount; ++channelIdx) {
        if (fabric::NotifyWait(kernelArg->lanes[channelIdx], kCompletionNotify,
                kCompletionMask) != CCU_SUCCESS) {
            return CCU_E_INTERNAL;
        }
    }
    return CCU_SUCCESS;
}

CcuResult CcuInboxFold(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<MergeKernelSpec *>(arg);
    if (kernelArg == nullptr || kernelArg->laneCount != 0 || kernelArg->remoteCount == 0 ||
        kernelArg->remoteCount >= MAX_RANK_SIZE) {
        return CCU_E_PARA;
    }

    std::vector<fabric::Variable> slotAddr(kernelArg->remoteCount);
    for (uint32_t slotIdx = 0; slotIdx < kernelArg->remoteCount; ++slotIdx) {
        if (fabric::LoadArg(slotAddr[slotIdx], slotIdx) != CCU_SUCCESS) {
            return CCU_E_INTERNAL;
        }
    }
    fabric::Variable outputAddr;
    fabric::Variable outputToken;
    fabric::Variable scratchToken;
    fabric::Variable chunkBytes;
    if (fabric::LoadArg(outputAddr, kernelArg->remoteCount) != CCU_SUCCESS ||
        fabric::LoadArg(outputToken, kernelArg->remoteCount + 1) != CCU_SUCCESS ||
        fabric::LoadArg(scratchToken, kernelArg->remoteCount + 2) != CCU_SUCCESS ||
        fabric::LoadArg(chunkBytes, kernelArg->remoteCount + 3) != CCU_SUCCESS) {
        return CCU_E_INTERNAL;
    }

    fabric::LocalAddr output;
    output.addr = outputAddr;
    output.token = outputToken;
    for (uint32_t slotIdx = 0; slotIdx < kernelArg->remoteCount; ++slotIdx) {
        fabric::LocalAddr source;
        source.addr = slotAddr[slotIdx];
        source.token = scratchToken;
        fabric::Event completion;
        if (fabric::LocalReduce(output, source, chunkBytes, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM,
                completion, 1U) != CCU_SUCCESS ||
            fabric::EventWait(completion, 1U) != CCU_SUCCESS) {
            return CCU_E_INTERNAL;
        }
    }
    return CCU_SUCCESS;
}

CcuResult CcuQuartetDeposit(CcuKernelArg arg)
{
    constexpr uint32_t kOutputAddrVar = 0;
    constexpr uint32_t kOutputTokenVar = 1;
    constexpr uint32_t kAddressNotify = 0;
    constexpr uint32_t kRoundDoneNotifyBase = 1;
    constexpr uint16_t kAddressMask = 3U;
    constexpr uint16_t kSyncMask = 1U;
    constexpr uint32_t kShardCount = 3;

    auto *kernelArg = static_cast<ExchangeKernelSpec *>(arg);
    if (kernelArg == nullptr || kernelArg->worldSize != 4 || kernelArg->selfRank >= 4 ||
        kernelArg->laneCount != 3 || kernelArg->resultBytes == 0 ||
        kernelArg->stripeOffsets[0] != 0 ||
        kernelArg->stripeOffsets[1] != kernelArg->stripeBytes[0] ||
        kernelArg->stripeOffsets[2] != kernelArg->stripeBytes[0] + kernelArg->stripeBytes[1] ||
        kernelArg->stripeOffsets[2] + kernelArg->stripeBytes[2] != kernelArg->resultBytes) {
        return CCU_E_PARA;
    }

    std::vector<fabric::Variable> localSourceBase(kernelArg->laneCount);
    for (uint32_t channelIdx = 0; channelIdx < kernelArg->laneCount; ++channelIdx) {
        if (kernelArg->remoteRanks[channelIdx] >= 4 || kernelArg->remoteRanks[channelIdx] == kernelArg->selfRank ||
            fabric::LoadArg(localSourceBase[channelIdx], channelIdx) != CCU_SUCCESS) {
            return CCU_E_PARA;
        }
    }
    const uint32_t scalarArg = kernelArg->laneCount;
    fabric::Variable outputAddr;
    fabric::Variable inputToken;
    fabric::Variable outputToken;
    fabric::Variable chunkBytes;
    if (fabric::LoadArg(outputAddr, scalarArg) != CCU_SUCCESS ||
        fabric::LoadArg(inputToken, scalarArg + 1) != CCU_SUCCESS ||
        fabric::LoadArg(outputToken, scalarArg + 2) != CCU_SUCCESS ||
        fabric::LoadArg(chunkBytes, scalarArg + 3) != CCU_SUCCESS) {
        return CCU_E_INTERNAL;
    }
    (void)chunkBytes;

    std::vector<fabric::Variable> peerOutputAddr(kernelArg->laneCount);
    std::vector<fabric::Variable> peerOutputToken(kernelArg->laneCount);
    for (uint32_t channelIdx = 0; channelIdx < kernelArg->laneCount; ++channelIdx) {
        const ChannelHandle channel = kernelArg->lanes[channelIdx];
        peerOutputAddr[channelIdx] =
            fabric::GetResByChannel<fabric::Variable>(channel, kOutputAddrVar);
        peerOutputToken[channelIdx] =
            fabric::GetResByChannel<fabric::Variable>(channel, kOutputTokenVar);
        if (fabric::WriteVariableWithNotify(channel, outputAddr, kOutputAddrVar,
                kAddressNotify, 1U << kOutputAddrVar) != CCU_SUCCESS ||
            fabric::WriteVariableWithNotify(channel, outputToken, kOutputTokenVar,
                kAddressNotify, 1U << kOutputTokenVar) != CCU_SUCCESS) {
            return CCU_E_INTERNAL;
        }
    }
    for (uint32_t channelIdx = 0; channelIdx < kernelArg->laneCount; ++channelIdx) {
        if (fabric::NotifyWait(kernelArg->lanes[channelIdx], kAddressNotify, kAddressMask) != CCU_SUCCESS) {
            return CCU_E_INTERNAL;
        }
    }

    fabric::Variable transferBytes[kShardCount];
    for (uint32_t shard = 0; shard < kShardCount; ++shard) {
        transferBytes[shard] = kernelArg->stripeBytes[shard];
    }
    fabric::Variable addressOffset;
    for (uint32_t round = 0; round < kShardCount; ++round) {
        fabric::Event completion;
        uint16_t completionMask = 0;
        for (uint32_t channelIdx = 0; channelIdx < kernelArg->laneCount; ++channelIdx) {
            const uint32_t targetRank = kernelArg->remoteRanks[channelIdx];
            const uint32_t sourcePosition =
                kernelArg->selfRank < targetRank ? kernelArg->selfRank : kernelArg->selfRank - 1U;
            const uint32_t shard = (sourcePosition + round) % kShardCount;
            addressOffset = kernelArg->stripeOffsets[shard];
            fabric::LocalAddr source;
            source.addr = localSourceBase[channelIdx];
            source.addr += addressOffset;
            source.token = inputToken;
            fabric::RemoteAddr destination;
            destination.addr = peerOutputAddr[channelIdx];
            destination.addr += addressOffset;
            destination.token = peerOutputToken[channelIdx];
            const uint16_t bit = static_cast<uint16_t>(1U << channelIdx);
            if (fabric::WriteReduce(kernelArg->lanes[channelIdx], destination, source,
                    transferBytes[shard], HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, completion, bit) != CCU_SUCCESS) {
                return CCU_E_INTERNAL;
            }
            completionMask = static_cast<uint16_t>(completionMask | bit);
        }
        if (fabric::EventWait(completion, completionMask) != CCU_SUCCESS) {
            return CCU_E_INTERNAL;
        }
        for (uint32_t channelIdx = 0; channelIdx < kernelArg->laneCount; ++channelIdx) {
            if (fabric::NotifyRecord(kernelArg->lanes[channelIdx], kRoundDoneNotifyBase + round, kSyncMask) !=
                CCU_SUCCESS) {
                return CCU_E_INTERNAL;
            }
        }
        for (uint32_t channelIdx = 0; channelIdx < kernelArg->laneCount; ++channelIdx) {
            if (fabric::NotifyWait(kernelArg->lanes[channelIdx], kRoundDoneNotifyBase + round, kSyncMask) !=
                CCU_SUCCESS) {
                return CCU_E_INTERNAL;
            }
        }
    }
    return CCU_SUCCESS;
}

namespace tiny_tile_graph {

constexpr uint32_t kInputAddrVar = 0U;
constexpr uint32_t kInputTokenVar = 1U;
constexpr uint32_t kPreSyncNotify = 0U;
constexpr uint16_t kInputMask = 3U;
constexpr uint16_t kReadyBit = 1U;
constexpr uint32_t kTileBytes = 4096U;
constexpr uint32_t kMaxInterleave = 8U;
constexpr uint32_t kR4Parallel = 16U;
constexpr uint64_t kLoopByteUnit = 8192ULL;
constexpr uint64_t kRepeatUnit = uint64_t{1} << 55U;
constexpr uint64_t kSingleLoopUnit = uint64_t{1} << 41U;
constexpr uint64_t kAddressStepUnit = uint64_t{1} << 21U;
constexpr uint64_t kBufferStepUnit = 1024ULL;

struct Context {
    const TinyMeshKernelSpec *arg = nullptr;
    fabric::Variable inputAddr;
    fabric::Variable outputAddr;
    fabric::Variable stagingBase;
    fabric::Variable inputToken;
    fabric::Variable outputToken;
    fabric::Variable scratchToken;
    fabric::Variable phase;
    std::vector<fabric::Variable> peerInputAddr;
    std::vector<fabric::Variable> peerInputToken;
    fabric::Event event;
};

CcuResult InitContext(Context &ctx)
{
    const auto *arg = ctx.arg;
    if (arg == nullptr || arg->worldSize == 0U || arg->worldSize > MAX_RANK_SIZE || arg->selfRank >= arg->worldSize ||
        arg->laneCount == 0U || arg->laneCount >= MAX_RANK_SIZE || arg->resultBytes == 0U ||
        arg->blockBytes != kTileBytes || arg->resultBytes % arg->blockBytes != 0U) {
        return CCU_E_PARA;
    }
    if (fabric::LoadArg(ctx.inputAddr, 0U) != CCU_SUCCESS ||
        fabric::LoadArg(ctx.outputAddr, 1U) != CCU_SUCCESS ||
        fabric::LoadArg(ctx.stagingBase, 2U) != CCU_SUCCESS ||
        fabric::LoadArg(ctx.inputToken, 3U) != CCU_SUCCESS ||
        fabric::LoadArg(ctx.outputToken, 4U) != CCU_SUCCESS ||
        fabric::LoadArg(ctx.scratchToken, 5U) != CCU_SUCCESS ||
        fabric::LoadArg(ctx.phase, 6U) != CCU_SUCCESS) {
        return CCU_E_INTERNAL;
    }
    ctx.peerInputAddr.resize(arg->laneCount);
    ctx.peerInputToken.resize(arg->laneCount);
    for (uint32_t channel = 0; channel < arg->laneCount; ++channel) {
        ctx.peerInputAddr[channel] =
            fabric::GetResByChannel<fabric::Variable>(arg->lanes[channel], kInputAddrVar);
        ctx.peerInputToken[channel] =
            fabric::GetResByChannel<fabric::Variable>(arg->lanes[channel], kInputTokenVar);
    }
    return CCU_SUCCESS;
}

CcuResult ExchangeInput(const Context &ctx)
{
    const auto *arg = ctx.arg;
    for (uint32_t channel = 0; channel < arg->laneCount; ++channel) {
        RETURN_IF_CCU_FAIL(fabric::WriteVariableWithNotify(
            arg->lanes[channel], ctx.inputAddr, kInputAddrVar, kPreSyncNotify, 1U << kInputAddrVar));
        RETURN_IF_CCU_FAIL(fabric::WriteVariableWithNotify(
            arg->lanes[channel], ctx.inputToken, kInputTokenVar, kPreSyncNotify, 1U << kInputTokenVar));
    }
    for (uint32_t channel = 0; channel < arg->laneCount; ++channel) {
        RETURN_IF_CCU_FAIL(fabric::NotifyWait(arg->lanes[channel], kPreSyncNotify, kInputMask));
    }
    return CCU_SUCCESS;
}

void BuildOwnerAddresses(Context &ctx, fabric::LocalAddr &localInput,
    std::vector<fabric::RemoteAddr> &remoteInputs)
{
    const auto *arg = ctx.arg;
    fabric::Variable ownerOffset;
    ownerOffset = static_cast<uint64_t>(arg->selfRank) * arg->resultBytes;
    localInput.addr = ctx.inputAddr;
    localInput.addr += ownerOffset;
    localInput.token = ctx.inputToken;
    remoteInputs.resize(arg->laneCount);
    for (uint32_t channel = 0; channel < arg->laneCount; ++channel) {
        remoteInputs[channel].addr = ctx.peerInputAddr[channel];
        remoteInputs[channel].addr += ownerOffset;
        remoteInputs[channel].token = ctx.peerInputToken[channel];
    }
}

CcuResult RunR4(Context &ctx)
{
    const auto *arg = ctx.arg;
    if (arg->worldSize != 4U || arg->laneCount != 3U || arg->tileParallelism != kR4Parallel ||
        arg->resultBytes != 128U * 1024U) {
        return CCU_E_PARA;
    }
    fabric::LocalAddr localInput;
    std::vector<fabric::RemoteAddr> remoteInputs;
    BuildOwnerAddresses(ctx, localInput, remoteInputs);
    fabric::Variable bankOffset;
    bankOffset = static_cast<uint64_t>(kR4Parallel) * kTileBytes;
    fabric::Variable ownerOffset;
    ownerOffset = static_cast<uint64_t>(arg->selfRank) * arg->resultBytes;
    fabric::LocalAddr secondInput;
    secondInput.addr = ctx.inputAddr;
    secondInput.addr += ownerOffset;
    secondInput.addr += bankOffset;
    secondInput.token = ctx.inputToken;
    fabric::LocalAddr firstOutput;
    firstOutput.addr = ctx.outputAddr;
    firstOutput.token = ctx.outputToken;
    fabric::LocalAddr secondOutput;
    secondOutput.addr = ctx.outputAddr;
    secondOutput.addr += bankOffset;
    secondOutput.token = ctx.outputToken;
    std::vector<fabric::RemoteAddr> secondRemotes(arg->laneCount);
    for (uint32_t channel = 0; channel < arg->laneCount; ++channel) {
        secondRemotes[channel].addr = ctx.peerInputAddr[channel];
        secondRemotes[channel].addr += ownerOffset;
        secondRemotes[channel].addr += bankOffset;
        secondRemotes[channel].token = ctx.peerInputToken[channel];
    }

    fabric::Array<fabric::Event> events(kR4Parallel);
    fabric::Array<fabric::CcuBuffer> buffers(5U * kR4Parallel);
    fabric::CcuBuffer lowerBuffers[4] =
        {buffers[0U], buffers[kR4Parallel], buffers[2U * kR4Parallel], buffers[3U * kR4Parallel]};
    fabric::CcuBuffer upperBuffers[4] = {buffers[kR4Parallel], buffers[2U * kR4Parallel],
        buffers[3U * kR4Parallel], buffers[4U * kR4Parallel]};
    fabric::Variable blockBytes;
    blockBytes = kTileBytes;
    fabric::Func body([&]() {
        for (uint32_t lane = 0U; lane < remoteInputs.size(); ++lane) {
            (void)fabric::Read(arg->lanes[lane], lowerBuffers[lane], remoteInputs[lane], blockBytes,
                events[0], static_cast<uint16_t>(1U << lane));
        }
        (void)fabric::LocalCopy(lowerBuffers[3], localInput, blockBytes, events[0], 0x8U);
        (void)fabric::EventWait(events[0], 0xFU);
        (void)fabric::LocalReduce(lowerBuffers, 4U, arg->scalarType, arg->scalarType,
            arg->combineOp, blockBytes, events[0], 0x1U);
        (void)fabric::EventWait(events[0], 0x1U);
        (void)fabric::LocalCopy(firstOutput, lowerBuffers[0], blockBytes, events[0], 0x1U);

        for (uint32_t lane = 0U; lane < secondRemotes.size(); ++lane) {
            (void)fabric::Read(arg->lanes[lane], upperBuffers[lane], secondRemotes[lane], blockBytes,
                events[0], static_cast<uint16_t>(0x10U << lane));
        }
        (void)fabric::LocalCopy(upperBuffers[3], secondInput, blockBytes, events[0], 0x80U);
        (void)fabric::EventWait(events[0], 0xF0U);
        (void)fabric::LocalReduce(upperBuffers, 4U, arg->scalarType, arg->scalarType,
            arg->combineOp, blockBytes, events[0], 0x10U);
        (void)fabric::EventWait(events[0], 0x10U);
        (void)fabric::LocalCopy(secondOutput, upperBuffers[0], blockBytes, events[0], 0x10U);
        (void)fabric::EventWait(events[0], 0x11U);
    });
    fabric::Variable loopParam;
    fabric::Loop loop(loopParam, body);
    loopParam = (2ULL * kR4Parallel * kTileBytes) * kLoopByteUnit + 1U;
    fabric::Variable parallelParam;
    parallelParam = static_cast<uint64_t>(kR4Parallel - 1U) * kRepeatUnit + kSingleLoopUnit;
    fabric::Variable offsetParam;
    offsetParam = static_cast<uint64_t>(kTileBytes) * kAddressStepUnit + kBufferStepUnit + 1U;
    std::vector<fabric::Loop> loops{loop};
    fabric::LoopGroup group(parallelParam, offsetParam, kR4Parallel, loops);
    RETURN_IF_CCU_FAIL(fabric::EventRecord(events[0], kReadyBit));
    return fabric::EventWait(events[0], kReadyBit);
}

CcuResult RunGenericPartial(Context &ctx)
{
    const auto *arg = ctx.arg;
    fabric::LocalAddr localInput;
    std::vector<fabric::RemoteAddr> remoteInputs;
    BuildOwnerAddresses(ctx, localInput, remoteInputs);
    fabric::LocalAddr partial;
    partial.addr = arg->isLeader != 0U ? ctx.outputAddr : ctx.stagingBase;
    partial.token = arg->isLeader != 0U ? ctx.outputToken : ctx.scratchToken;
    const uint32_t sourceCount = arg->laneCount + arg->isLeader;
    if (arg->worldSize != 16U || sourceCount != kMaxInterleave || arg->tileParallelism == 0U ||
        arg->resultBytes != static_cast<uint64_t>(arg->tileParallelism) * kTileBytes) {
        return CCU_E_PARA;
    }

    fabric::Array<fabric::Event> events(arg->tileParallelism);
    fabric::Array<fabric::CcuBuffer> buffers(kMaxInterleave * arg->tileParallelism);
    fabric::Variable blockBytes;
    blockBytes = kTileBytes;
    constexpr uint16_t kAllSources = 0xFFU;
    fabric::Func body([&]() {
        for (uint32_t channel = 0; channel < remoteInputs.size(); ++channel) {
            (void)fabric::Read(arg->lanes[channel], buffers[channel], remoteInputs[channel], blockBytes,
                events[0], static_cast<uint16_t>(1U << channel));
        }
        if (arg->isLeader != 0U) {
            const uint32_t localSlot = arg->laneCount;
            (void)fabric::LocalCopy(buffers[localSlot], localInput, blockBytes, events[0],
                static_cast<uint16_t>(1U << localSlot));
        }
        (void)fabric::EventWait(events[0], kAllSources);
        (void)fabric::LocalReduce(buffers.data(), kMaxInterleave, arg->scalarType, arg->scalarType,
            arg->combineOp, blockBytes, events[0], kReadyBit);
        (void)fabric::EventWait(events[0], kReadyBit);
        (void)fabric::LocalCopy(partial, buffers[0], blockBytes, events[0], kReadyBit);
        (void)fabric::EventWait(events[0], kReadyBit);
    });
    fabric::Variable loopParam;
    fabric::Loop loop(loopParam, body);
    loopParam = arg->resultBytes * kLoopByteUnit + 1U;
    fabric::Variable parallelParam;
    parallelParam = static_cast<uint64_t>(arg->tileParallelism - 1U) * kRepeatUnit + kSingleLoopUnit;
    fabric::Variable offsetParam;
    offsetParam = static_cast<uint64_t>(kTileBytes) * kAddressStepUnit +
        static_cast<uint64_t>(kMaxInterleave) * kBufferStepUnit + 1U;
    std::vector<fabric::Loop> loops{loop};
    fabric::LoopGroup group(parallelParam, offsetParam, arg->tileParallelism, loops);
    RETURN_IF_CCU_FAIL(fabric::EventRecord(events[0], kReadyBit));
    return fabric::EventWait(events[0], kReadyBit);
}

CcuResult RunMerge(Context &ctx)
{
    fabric::LocalAddr output;
    output.addr = ctx.outputAddr;
    output.token = ctx.outputToken;
    fabric::LocalAddr scratch;
    scratch.addr = ctx.stagingBase;
    scratch.token = ctx.scratchToken;
    fabric::Variable bytes;
    bytes = ctx.arg->resultBytes;
    RETURN_IF_CCU_FAIL(fabric::LocalReduce(output, scratch, bytes, ctx.arg->scalarType,
        ctx.arg->combineOp, ctx.event, kReadyBit));
    return fabric::EventWait(ctx.event, kReadyBit);
}

} // namespace tiny_tile_graph

CcuResult CcuTinyTileGraph(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<TinyMeshKernelSpec *>(arg);
    tiny_tile_graph::Context ctx;
    ctx.arg = kernelArg;
    RETURN_IF_CCU_FAIL(tiny_tile_graph::InitContext(ctx));
    CCU_IF(ctx.phase == static_cast<uint64_t>(MeshStage::PARTIAL))
    {
        RETURN_IF_CCU_FAIL(tiny_tile_graph::ExchangeInput(ctx));
        if (kernelArg->worldSize == 4U) {
            RETURN_IF_CCU_FAIL(tiny_tile_graph::RunR4(ctx));
        } else {
            RETURN_IF_CCU_FAIL(tiny_tile_graph::RunGenericPartial(ctx));
        }
    }
    CCU_IF(ctx.phase == static_cast<uint64_t>(MeshStage::MERGE))
    {
        RETURN_IF_CCU_FAIL(tiny_tile_graph::RunMerge(ctx));
    }
    return CCU_SUCCESS;
}

namespace capacity_inbox_graph {

constexpr uint32_t kInputAddressVar = 0U;
constexpr uint32_t kInputTokenVar = 1U;
constexpr uint32_t kInputReadyNotify = 0U;
constexpr uint16_t kInputReadyMask = 3U;

struct Context {
    const CapacityInboxSpec *arg = nullptr;
    fabric::Variable inputAddr;
    fabric::Variable stagingBase;
    fabric::Variable outputAddr;
    fabric::Variable inputToken;
    fabric::Variable scratchToken;
    fabric::Variable outputToken;
    std::vector<fabric::Variable> peerInputAddr;
    std::vector<fabric::Variable> peerInputToken;
};

CcuResult LoadContext(Context &ctx)
{
    const auto *arg = ctx.arg;
    if (arg == nullptr || arg->worldSize == 0U || arg->worldSize > MAX_RANK_SIZE ||
        arg->selfRank >= arg->worldSize || arg->laneCount == 0U || arg->laneCount >= MAX_RANK_SIZE ||
        arg->resultBytes == 0U || arg->slotPitch < arg->resultBytes || (arg->slotPitch & 511U) != 0U ||
        arg->slotTotal == 0U || arg->slotTotal > arg->worldSize - 1U) {
        return CCU_E_PARA;
    }
    if (fabric::LoadArg(ctx.inputAddr, 0U) != CCU_SUCCESS ||
        fabric::LoadArg(ctx.stagingBase, 1U) != CCU_SUCCESS ||
        fabric::LoadArg(ctx.outputAddr, 2U) != CCU_SUCCESS ||
        fabric::LoadArg(ctx.inputToken, 3U) != CCU_SUCCESS ||
        fabric::LoadArg(ctx.scratchToken, 4U) != CCU_SUCCESS ||
        fabric::LoadArg(ctx.outputToken, 5U) != CCU_SUCCESS) {
        return CCU_E_INTERNAL;
    }
    ctx.peerInputAddr.resize(arg->laneCount);
    ctx.peerInputToken.resize(arg->laneCount);
    for (uint32_t channel = 0; channel < arg->laneCount; ++channel) {
        if (arg->remoteRanks[channel] >= arg->worldSize || arg->remoteRanks[channel] == arg->selfRank) {
            return CCU_E_PARA;
        }
        ctx.peerInputAddr[channel] =
            fabric::GetResByChannel<fabric::Variable>(arg->lanes[channel], kInputAddressVar);
        ctx.peerInputToken[channel] =
            fabric::GetResByChannel<fabric::Variable>(arg->lanes[channel], kInputTokenVar);
    }
    return CCU_SUCCESS;
}

CcuResult AnnounceInput(Context &ctx, uint32_t channel)
{
    const auto *arg = ctx.arg;
    RETURN_IF_CCU_FAIL(fabric::WriteVariableWithNotify(arg->lanes[channel], ctx.inputAddr,
        kInputAddressVar, kInputReadyNotify, 1U << kInputAddressVar));
    return fabric::WriteVariableWithNotify(arg->lanes[channel], ctx.inputToken,
        kInputTokenVar, kInputReadyNotify, 1U << kInputTokenVar);
}

CcuResult Rendezvous(Context &ctx)
{
    const auto *arg = ctx.arg;
    for (uint32_t channel = 0; channel < arg->laneCount; ++channel) {
        RETURN_IF_CCU_FAIL(AnnounceInput(ctx, channel));
    }
    for (uint32_t channel = 0; channel < arg->laneCount; ++channel) {
        RETURN_IF_CCU_FAIL(fabric::NotifyWait(arg->lanes[channel], kInputReadyNotify, kInputReadyMask));
    }
    return CCU_SUCCESS;
}

void SetRemoteOwner(const Context &ctx, uint32_t channel, const fabric::Variable &ownerOffset,
    fabric::RemoteAddr &source)
{
    source.addr = ctx.peerInputAddr[channel];
    source.addr += ownerOffset;
    source.token = ctx.peerInputToken[channel];
}

CcuResult LaunchInboxReads(Context &ctx, const fabric::Variable &ownerOffset,
    const fabric::Variable &bytes, fabric::Event &completion, uint16_t &completionMask)
{
    const auto *arg = ctx.arg;
    completionMask = 0U;
    for (uint32_t channel = 0; channel < arg->laneCount; ++channel) {
        const uint32_t slot = arg->inboxSlots[channel];
        if (slot >= arg->slotTotal) {
            continue;
        }
        fabric::Variable slotOffset;
        slotOffset = static_cast<uint64_t>(slot) * arg->slotPitch;
        fabric::LocalAddr destination;
        destination.addr = ctx.stagingBase;
        destination.addr += slotOffset;
        destination.token = ctx.scratchToken;
        fabric::RemoteAddr source;
        SetRemoteOwner(ctx, channel, ownerOffset, source);
        const uint16_t bit = static_cast<uint16_t>(1U << channel);
        RETURN_IF_CCU_FAIL(fabric::Read(
            arg->lanes[channel], destination, source, bytes, completion, bit));
        completionMask = static_cast<uint16_t>(completionMask | bit);
    }
    return CCU_SUCCESS;
}

CcuResult FoldDirectSources(Context &ctx, const fabric::Variable &ownerOffset,
    const fabric::Variable &bytes)
{
    const auto *arg = ctx.arg;
    for (uint32_t channel = 0; channel < arg->laneCount; ++channel) {
        if (arg->fusedOrder[channel] == MAX_RANK_SIZE) {
            continue;
        }
        fabric::LocalAddr destination;
        destination.addr = ctx.outputAddr;
        destination.token = ctx.outputToken;
        fabric::RemoteAddr source;
        SetRemoteOwner(ctx, channel, ownerOffset, source);
        fabric::Event reduced;
        RETURN_IF_CCU_FAIL(fabric::ReadReduce(arg->lanes[channel], destination, source, bytes,
            arg->scalarType, arg->combineOp, reduced, 1U));
        RETURN_IF_CCU_FAIL(fabric::EventWait(reduced, 1U));
    }
    return CCU_SUCCESS;
}

CcuResult RunCapacityPull(Context &ctx)
{
    const auto *arg = ctx.arg;
    RETURN_IF_CCU_FAIL(Rendezvous(ctx));
    fabric::Variable bytes;
    bytes = arg->resultBytes;
    fabric::Variable ownerOffset;
    ownerOffset = static_cast<uint64_t>(arg->selfRank) * arg->resultBytes;
    fabric::Event inboxReads;
    uint16_t inboxMask = 0U;
    RETURN_IF_CCU_FAIL(LaunchInboxReads(ctx, ownerOffset, bytes, inboxReads, inboxMask));
    RETURN_IF_CCU_FAIL(FoldDirectSources(ctx, ownerOffset, bytes));
    if (inboxMask != 0U) {
        RETURN_IF_CCU_FAIL(fabric::EventWait(inboxReads, inboxMask));
    }
    return CCU_SUCCESS;
}

} // namespace capacity_inbox_graph

CcuResult CcuCapacityInbox(CcuKernelArg arg)
{
    capacity_inbox_graph::Context ctx;
    ctx.arg = static_cast<CapacityInboxSpec *>(arg);
    RETURN_IF_CCU_FAIL(capacity_inbox_graph::LoadContext(ctx));
    RETURN_IF_CCU_FAIL(capacity_inbox_graph::RunCapacityPull(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuRankOrderFold(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CapacityFoldSpec *>(arg);
    if (kernelArg == nullptr || kernelArg->laneCount != 0U || kernelArg->worldSize == 0U ||
        kernelArg->worldSize > MAX_RANK_SIZE || kernelArg->selfRank >= kernelArg->worldSize ||
        kernelArg->resultBytes == 0U || kernelArg->slotPitch < kernelArg->resultBytes ||
        (kernelArg->slotPitch & 511U) != 0U || kernelArg->slotTotal == 0U ||
        kernelArg->slotTotal > kernelArg->worldSize - 1U) {
        return CCU_E_PARA;
    }
    fabric::Variable outputAddr;
    fabric::Variable stagingBase;
    fabric::Variable outputToken;
    fabric::Variable scratchToken;
    if (fabric::LoadArg(outputAddr, 0U) != CCU_SUCCESS ||
        fabric::LoadArg(stagingBase, 1U) != CCU_SUCCESS ||
        fabric::LoadArg(outputToken, 2U) != CCU_SUCCESS ||
        fabric::LoadArg(scratchToken, 3U) != CCU_SUCCESS) {
        return CCU_E_INTERNAL;
    }
    fabric::LocalAddr output;
    output.addr = outputAddr;
    output.token = outputToken;
    for (uint32_t sourceRank = 0; sourceRank < kernelArg->worldSize; ++sourceRank) {
        if (sourceRank == kernelArg->selfRank) {
            continue;
        }
        const uint32_t slot = kernelArg->slotForRank[sourceRank];
        if (slot >= kernelArg->slotTotal) {
            continue;
        }
        fabric::Variable slotOffset;
        slotOffset = static_cast<uint64_t>(slot) * kernelArg->slotPitch;
        fabric::LocalAddr source;
        source.addr = stagingBase;
        source.addr += slotOffset;
        source.token = scratchToken;
        fabric::Event reduced;
        fabric::Variable bytes;
        bytes = kernelArg->resultBytes;
        RETURN_IF_CCU_FAIL(fabric::LocalReduce(output, source, bytes, kernelArg->scalarType,
            kernelArg->combineOp, reduced, 1U));
        RETURN_IF_CCU_FAIL(fabric::EventWait(reduced, 1U));
    }
    return CCU_SUCCESS;
}

} // namespace ops_hccl

#undef RETURN_IF_CCU_FAIL
