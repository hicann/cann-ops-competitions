/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include <hcomm/hcomm_primitives.h>

#include <cstdint>
#include <vector>

#include "ccu_kernel.h"
#include "custom.h"
#include "log.h"

namespace ops_hccl {
namespace ccu = ::AscendC::ccu;

#define CCU_RETURN_IF_ERROR(call) \
    do { \
        const CcuResult result = (call); \
        if (result != CCU_SUCCESS) { \
            return result; \
        } \
    } while (0)

namespace {
constexpr uint32_t PARAM_NOTIFY_INDEX = 0;
constexpr uint32_t SCATTER_NOTIFY_INDEX = 1;
constexpr uint32_t ALLGATHER_NOTIFY_INDEX = 2;

struct BroadcastContext {
    const CcuKernelArgBroadcast *arg{nullptr};
    ccu::Variable root;
    ccu::Variable phase;
    std::vector<ccu::Variable> localBases;
    std::vector<ccu::Variable> localTokens;
    std::vector<std::vector<ccu::Variable>> remoteBases;
    std::vector<std::vector<ccu::Variable>> remoteTokens;
    ccu::Variable addressOffset;
    ccu::Variable transferBytes;
    ccu::LocalAddr localSource;
    ccu::RemoteAddr remoteDestination;
    ccu::Event dataEvent;
};

CcuResult InitContext(BroadcastContext &ctx)
{
    const auto *arg = ctx.arg;
    ctx.localBases.resize(arg->windowCount);
    ctx.localTokens.resize(arg->windowCount);
    ctx.remoteBases.resize(arg->windowCount);
    ctx.remoteTokens.resize(arg->windowCount);
    uint32_t argId = 0;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.root, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.phase, argId++));
    for (uint32_t window = 0; window < arg->windowCount; ++window) {
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.localBases[window], argId++));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.localTokens[window], argId++));
        ctx.remoteBases[window].resize(arg->channelCount);
        ctx.remoteTokens[window].resize(arg->channelCount);
        const uint32_t baseVarIndex = 2 * window;
        const uint32_t tokenVarIndex = baseVarIndex + 1;
        for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
            ctx.remoteBases[window][channel] =
                ccu::GetResByChannel<ccu::Variable>(arg->channels[channel], baseVarIndex);
            ctx.remoteTokens[window][channel] =
                ccu::GetResByChannel<ccu::Variable>(arg->channels[channel], tokenVarIndex);
        }
    }
    return CCU_SUCCESS;
}

CcuResult ExchangeChannelWindows(BroadcastContext &ctx, uint32_t channelIndex)
{
    const auto *arg = ctx.arg;
    const ChannelHandle channel = arg->channels[channelIndex];
    uint16_t readyMask = 0;
    for (uint32_t window = 0; window < arg->windowCount; ++window) {
        const uint32_t baseVarIndex = 2 * window;
        const uint32_t tokenVarIndex = baseVarIndex + 1;
        const uint16_t baseBit = static_cast<uint16_t>(1U << baseVarIndex);
        const uint16_t tokenBit = static_cast<uint16_t>(1U << tokenVarIndex);
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
            channel, ctx.localBases[window], baseVarIndex, PARAM_NOTIFY_INDEX, baseBit));
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
            channel, ctx.localTokens[window], tokenVarIndex, PARAM_NOTIFY_INDEX, tokenBit));
        readyMask = static_cast<uint16_t>(readyMask | baseBit | tokenBit);
    }
    return ccu::NotifyWait(channel, PARAM_NOTIFY_INDEX, readyMask);
}

CcuResult OneShotPreSync(BroadcastContext &ctx)
{
    const auto *arg = ctx.arg;
    for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
        CCU_IF(ctx.root == arg->rankId)
        {
            CCU_RETURN_IF_ERROR(ExchangeChannelWindows(ctx, channel));
        }
        CCU_IF(ctx.root == arg->peerRanks[channel])
        {
            CCU_RETURN_IF_ERROR(ExchangeChannelWindows(ctx, channel));
        }
    }
    return CCU_SUCCESS;
}

CcuResult AllPeerPreSync(BroadcastContext &ctx)
{
    for (uint32_t channel = 0; channel < ctx.arg->channelCount; ++channel) {
        CCU_RETURN_IF_ERROR(ExchangeChannelWindows(ctx, channel));
    }
    return CCU_SUCCESS;
}

CcuResult RunOneShot(BroadcastContext &ctx)
{
    const auto *arg = ctx.arg;
    CCU_RETURN_IF_ERROR(OneShotPreSync(ctx));
    CCU_IF(ctx.root == arg->rankId)
    {
        ctx.transferBytes = arg->windowBytes[0];
        const uint16_t eventMask = static_cast<uint16_t>((1U << arg->channelCount) - 1U);
        for (uint32_t channelIndex = 0; channelIndex < arg->channelCount; ++channelIndex) {
            const uint16_t bit = static_cast<uint16_t>(1U << channelIndex);
            ctx.localSource.addr = ctx.localBases[0];
            ctx.localSource.token = ctx.localTokens[0];
            ctx.remoteDestination.addr = ctx.remoteBases[0][channelIndex];
            ctx.remoteDestination.token = ctx.remoteTokens[0][channelIndex];
            CCU_RETURN_IF_ERROR(ccu::Write(arg->channels[channelIndex], ctx.remoteDestination,
                ctx.localSource, ctx.transferBytes, ctx.dataEvent, bit));
        }
        CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.dataEvent, eventMask));
        for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
            CCU_RETURN_IF_ERROR(ccu::NotifyRecord(arg->channels[channel], SCATTER_NOTIFY_INDEX, 1));
        }
    }
    for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
        CCU_IF(ctx.root == arg->peerRanks[channel])
        {
            CCU_RETURN_IF_ERROR(ccu::NotifyWait(arg->channels[channel], SCATTER_NOTIFY_INDEX, 1));
        }
    }
    return CCU_SUCCESS;
}

CcuResult WaitForScatterFromRoot(BroadcastContext &ctx, uint32_t window)
{
    const auto *arg = ctx.arg;
    const uint16_t windowBit = static_cast<uint16_t>(1U << window);
    for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
        CCU_IF(ctx.root == arg->peerRanks[channel])
        {
            CCU_RETURN_IF_ERROR(ccu::NotifyWait(
                arg->channels[channel], SCATTER_NOTIFY_INDEX, windowBit));
        }
    }
    return CCU_SUCCESS;
}

CcuResult ScatterWindow(BroadcastContext &ctx, uint32_t window)
{
    const auto *arg = ctx.arg;
    const uint16_t windowBit = static_cast<uint16_t>(1U << window);
    CCU_IF(ctx.root == arg->rankId)
    {
        const uint16_t eventMask = static_cast<uint16_t>((1U << arg->channelCount) - 1U);
        for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
            const uint16_t bit = static_cast<uint16_t>(1U << channel);
            ctx.localSource.addr = ctx.localBases[window];
            ctx.addressOffset = arg->peerSliceOffsets[window][channel];
            ctx.localSource.addr += ctx.addressOffset;
            ctx.localSource.token = ctx.localTokens[window];
            ctx.remoteDestination.addr = ctx.remoteBases[window][channel];
            ctx.remoteDestination.addr += ctx.addressOffset;
            ctx.remoteDestination.token = ctx.remoteTokens[window][channel];
            ctx.transferBytes = arg->peerSliceBytes[window][channel];
            CCU_RETURN_IF_ERROR(ccu::Write(arg->channels[channel], ctx.remoteDestination,
                ctx.localSource, ctx.transferBytes, ctx.dataEvent, bit));
        }
        CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.dataEvent, eventMask));
        for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
            CCU_RETURN_IF_ERROR(ccu::NotifyRecord(
                arg->channels[channel], SCATTER_NOTIFY_INDEX, windowBit));
        }
    }
    return WaitForScatterFromRoot(ctx, window);
}

CcuResult AllGatherWindow(BroadcastContext &ctx, uint32_t window)
{
    const auto *arg = ctx.arg;
    const uint16_t windowBit = static_cast<uint16_t>(1U << window);
    ctx.localSource.addr = ctx.localBases[window];
    ctx.addressOffset = arg->ownerSliceOffsets[window];
    ctx.localSource.addr += ctx.addressOffset;
    ctx.localSource.token = ctx.localTokens[window];
    ctx.transferBytes = arg->ownerSliceBytes[window];

    // A non-root rank must not write its slice back to the root, because the root
    // already owns the complete input buffer.  Expand every possible root into a
    // distinct static branch so the event mask contains only writes that really
    // execute; recording dummy local events leaves unmatched posts in the checker.
    for (uint32_t candidateRoot = 0; candidateRoot < arg->rankSize; ++candidateRoot) {
        uint16_t eventMask = 0;
        for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
            if (arg->peerRanks[channel] != candidateRoot) {
                eventMask = static_cast<uint16_t>(eventMask | (1U << channel));
            }
        }
        if (eventMask == 0) {
            continue;
        }
        CCU_IF(ctx.root == candidateRoot)
        {
            for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
                if (arg->peerRanks[channel] == candidateRoot) {
                    continue;
                }
                const uint16_t bit = static_cast<uint16_t>(1U << channel);
                ctx.remoteDestination.addr = ctx.remoteBases[window][channel];
                ctx.remoteDestination.addr += ctx.addressOffset;
                ctx.remoteDestination.token = ctx.remoteTokens[window][channel];
                CCU_RETURN_IF_ERROR(ccu::Write(arg->channels[channel], ctx.remoteDestination,
                    ctx.localSource, ctx.transferBytes, ctx.dataEvent, bit));
            }
            CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.dataEvent, eventMask));
        }
    }

    for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
        CCU_IF(ctx.root != arg->peerRanks[channel])
        {
            CCU_RETURN_IF_ERROR(ccu::NotifyRecord(
                arg->channels[channel], ALLGATHER_NOTIFY_INDEX, windowBit));
        }
    }
    CCU_IF(ctx.root != arg->rankId)
    {
        for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
            CCU_RETURN_IF_ERROR(ccu::NotifyWait(
                arg->channels[channel], ALLGATHER_NOTIFY_INDEX, windowBit));
        }
    }
    return CCU_SUCCESS;
}

CcuResult RunScatterPhase(BroadcastContext &ctx)
{
    CCU_RETURN_IF_ERROR(AllPeerPreSync(ctx));
    for (uint32_t window = 0; window < ctx.arg->windowCount; ++window) {
        CCU_RETURN_IF_ERROR(ScatterWindow(ctx, window));
    }
    return CCU_SUCCESS;
}

CcuResult RunAllGatherPhase(BroadcastContext &ctx)
{
    for (uint32_t window = 0; window < ctx.arg->windowCount; ++window) {
        CCU_RETURN_IF_ERROR(AllGatherWindow(ctx, window));
    }
    return CCU_SUCCESS;
}
} // namespace

CcuResult CcuKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgBroadcast *>(arg);
    if (kernelArg == nullptr || kernelArg->rankSize < 2 || kernelArg->rankSize > MAX_RANK_SIZE
        || kernelArg->rankId >= kernelArg->rankSize || kernelArg->channelCount == 0
        || kernelArg->channelCount >= kernelArg->rankSize || kernelArg->channelCount > 15
        || kernelArg->kernelCount == 0 || kernelArg->kernelCount > 2
        || kernelArg->kernelIndex >= kernelArg->kernelCount || kernelArg->windowCount == 0
        || kernelArg->windowCount > BROADCAST_MAX_WINDOWS) {
        HCCL_ERROR("[BroadcastV3] invalid CCU kernel arguments");
        return CCU_E_PARA;
    }

    BroadcastContext ctx;
    ctx.arg = kernelArg;
    CCU_RETURN_IF_ERROR(InitContext(ctx));
    if (kernelArg->mode == static_cast<uint32_t>(BroadcastMode::ONE_SHOT)) {
        return RunOneShot(ctx);
    }
    if (kernelArg->mode == static_cast<uint32_t>(BroadcastMode::SCATTER_ALLGATHER)) {
        CCU_IF(ctx.phase == static_cast<uint32_t>(BroadcastPhase::SCATTER))
        {
            CCU_RETURN_IF_ERROR(RunScatterPhase(ctx));
        }
        CCU_IF(ctx.phase == static_cast<uint32_t>(BroadcastPhase::ALLGATHER))
        {
            CCU_RETURN_IF_ERROR(RunAllGatherPhase(ctx));
        }
        return CCU_SUCCESS;
    }
    HCCL_ERROR("[BroadcastV3] invalid mode[%u]", kernelArg->mode);
    return CCU_E_PARA;
}
} // namespace ops_hccl
