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

#include <ccu/ccu_types.h>

#include "custom.h"

namespace ccu = ::AscendC::ccu;

namespace ops_hccl {

struct CcuAllGatherKernelContext {
    const CcuAllGatherKernelArg *arg = nullptr;

    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable rankOffset;
    ccu::Variable partnerOffset;
    ccu::Variable sliceOffset0;
    ccu::Variable sliceSize0;
    ccu::Variable sliceOffset1;
    ccu::Variable sliceSize1;
    ccu::Variable mode;

    std::vector<ccu::Variable> remoteOutputs;
    std::vector<ccu::Variable> remoteTokens;

    ccu::Event event0;
    ccu::Event event1;
};

struct CcuSmallPullKernelContext {
    const CcuSmallPullKernelArg *arg = nullptr;

    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable rankOffset;
    ccu::Variable dataSize;

    std::vector<ccu::Variable> remoteTokens;
    ccu::Event event;
};

// Steady-state small-message kernel.  The first invocation has already
// published each peer's input token into TOKEN_XN_ID and populated the local
// rank's output slot.  The timed warm path therefore needs only the final
// output address/token and the per-rank byte count.
struct CcuSmallPullWarmKernelContext {
    const CcuSmallPullKernelArg *arg = nullptr;

    ccu::Variable output;
    ccu::Variable outputToken;
    ccu::Variable dataSize;

    std::vector<ccu::Variable> remoteTokens;
    ccu::Event event;
};

} // namespace ops_hccl


namespace ops_hccl {
namespace {

constexpr uint32_t OUTPUT_XN_ID = 1;
constexpr uint32_t TOKEN_XN_ID = 2;
constexpr uint32_t CHANNEL_NOTIFY_INDEX = 0;
constexpr uint16_t OUTPUT_READY_MASK = static_cast<uint16_t>(1U << OUTPUT_XN_ID);
constexpr uint16_t TOKEN_READY_MASK = static_cast<uint16_t>(1U << TOKEN_XN_ID);
constexpr uint16_t POST_SYNC_MASK = static_cast<uint16_t>(1U << 3);
constexpr uint16_t RD4_STAGE0_READY_MASK = static_cast<uint16_t>(1U << 4);
constexpr uint32_t EVENT_BIT_COUNT = 16;
constexpr uint32_t SMALL_DIRECT_MAX_CHANNELS = 8;

#define RETURN_IF_CCU_ERROR(call) \
    do { \
        CcuResult result = (call); \
        if (result != CCU_SUCCESS) { \
            return result; \
        } \
    } while (0)

static CcuResult InitResources(CcuAllGatherKernelContext &ctx)
{
    ctx.remoteOutputs.reserve(ctx.arg->channelCount);
    ctx.remoteTokens.reserve(ctx.arg->channelCount);
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        ctx.remoteOutputs.emplace_back(
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIndex], OUTPUT_XN_ID));
        ctx.remoteTokens.emplace_back(
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIndex], TOKEN_XN_ID));
    }
    return CCU_SUCCESS;
}

static CcuResult LoadTaskArgs(CcuAllGatherKernelContext &ctx)
{
    uint32_t argIndex = 0;
    RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.input, argIndex++));
    RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.output, argIndex++));
    RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.inputToken, argIndex++));
    RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.outputToken, argIndex++));
    RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.rankOffset, argIndex++));
    RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.partnerOffset, argIndex++));
    RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.sliceOffset0, argIndex++));
    RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.sliceSize0, argIndex++));
    RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.sliceOffset1, argIndex++));
    RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.sliceSize1, argIndex++));
    RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.mode, argIndex++));
    return CCU_SUCCESS;
}

static CcuResult ExchangeRemoteOutputAll(CcuAllGatherKernelContext &ctx)
{
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        RETURN_IF_CCU_ERROR(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIndex], ctx.output,
            OUTPUT_XN_ID, CHANNEL_NOTIFY_INDEX, OUTPUT_READY_MASK));
        RETURN_IF_CCU_ERROR(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIndex], ctx.outputToken,
            TOKEN_XN_ID, CHANNEL_NOTIFY_INDEX, TOKEN_READY_MASK));
    }

    constexpr uint16_t readyMask = OUTPUT_READY_MASK | TOKEN_READY_MASK;
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        RETURN_IF_CCU_ERROR(
            ccu::NotifyWait(ctx.arg->channels[channelIndex], CHANNEL_NOTIFY_INDEX, readyMask));
    }
    return CCU_SUCCESS;
}

static CcuResult PublishRemoteOutputOne(CcuAllGatherKernelContext &ctx,
    uint32_t channelIndex)
{
    RETURN_IF_CCU_ERROR(ccu::WriteVariableWithNotify(
        ctx.arg->channels[channelIndex], ctx.output, OUTPUT_XN_ID,
        CHANNEL_NOTIFY_INDEX, OUTPUT_READY_MASK));
    RETURN_IF_CCU_ERROR(ccu::WriteVariableWithNotify(
        ctx.arg->channels[channelIndex], ctx.outputToken, TOKEN_XN_ID,
        CHANNEL_NOTIFY_INDEX, TOKEN_READY_MASK));
    return CCU_SUCCESS;
}

static CcuResult WaitRemoteOutputOne(CcuAllGatherKernelContext &ctx,
    uint32_t channelIndex)
{
    constexpr uint16_t readyMask =
        OUTPUT_READY_MASK | TOKEN_READY_MASK;
    RETURN_IF_CCU_ERROR(ccu::NotifyWait(
        ctx.arg->channels[channelIndex], CHANNEL_NOTIFY_INDEX, readyMask));
    return CCU_SUCCESS;
}

static CcuResult ExchangeRemoteOutputOne(CcuAllGatherKernelContext &ctx,
    uint32_t channelIndex)
{
    RETURN_IF_CCU_ERROR(PublishRemoteOutputOne(ctx, channelIndex));
    RETURN_IF_CCU_ERROR(WaitRemoteOutputOne(ctx, channelIndex));
    return CCU_SUCCESS;
}

static CcuResult PostSyncAll(CcuAllGatherKernelContext &ctx)
{
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        RETURN_IF_CCU_ERROR(
            ccu::NotifyRecord(ctx.arg->channels[channelIndex], CHANNEL_NOTIFY_INDEX, POST_SYNC_MASK));
    }
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        RETURN_IF_CCU_ERROR(
            ccu::NotifyWait(ctx.arg->channels[channelIndex], CHANNEL_NOTIFY_INDEX, POST_SYNC_MASK));
    }
    return CCU_SUCCESS;
}

static CcuResult InitSmallPullResources(CcuSmallPullKernelContext &ctx)
{
    ctx.remoteTokens.reserve(ctx.arg->channelCount);
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        ctx.remoteTokens.emplace_back(
            ccu::GetResByChannel<ccu::Variable>(
                ctx.arg->channels[channelIndex], TOKEN_XN_ID));
    }
    return CCU_SUCCESS;
}

static CcuResult LoadSmallPullTaskArgs(CcuSmallPullKernelContext &ctx)
{
    uint32_t argIndex = 0;
    RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.input, argIndex++));
    RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.output, argIndex++));
    RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.inputToken, argIndex++));
    RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.outputToken, argIndex++));
    RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.rankOffset, argIndex++));
    RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.dataSize, argIndex++));
    return CCU_SUCCESS;
}

static CcuResult PublishSmallPullInputToken(CcuSmallPullKernelContext &ctx)
{
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        RETURN_IF_CCU_ERROR(ccu::WriteVariableWithNotify(
            ctx.arg->channels[channelIndex], ctx.inputToken, TOKEN_XN_ID,
            CHANNEL_NOTIFY_INDEX, TOKEN_READY_MASK));
    }
    return CCU_SUCCESS;
}

static CcuResult WaitSmallPullInputToken(CcuSmallPullKernelContext &ctx)
{
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        RETURN_IF_CCU_ERROR(ccu::NotifyWait(
            ctx.arg->channels[channelIndex], CHANNEL_NOTIFY_INDEX,
            TOKEN_READY_MASK));
    }
    return CCU_SUCCESS;
}

static CcuResult RunSmallPull(CcuSmallPullKernelContext &ctx)
{
    // Remote input addresses are static Channel resources. Publish the one
    // dynamic input token while the independent self copy is in flight.
    RETURN_IF_CCU_ERROR(PublishSmallPullInputToken(ctx));

    uint16_t eventMask = 0;
    if (ctx.arg->directCopyLocal != 0) {
        ccu::LocalAddr localSource;
        localSource.addr = ctx.input;
        localSource.token = ctx.inputToken;

        ccu::LocalAddr localDestination;
        localDestination.addr = ctx.output;
        localDestination.addr += ctx.rankOffset;
        localDestination.token = ctx.outputToken;

        const uint16_t localMask =
            static_cast<uint16_t>(1U << ctx.arg->channelCount);
        eventMask = static_cast<uint16_t>(eventMask | localMask);
        RETURN_IF_CCU_ERROR(ccu::LocalCopy(
            localDestination, localSource, ctx.dataSize, ctx.event, localMask));
    }

    RETURN_IF_CCU_ERROR(WaitSmallPullInputToken(ctx));

    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        ccu::LocalAddr destination;
        destination.addr = ctx.output;
        // peerRanks[] is registration-time metadata. Emit a finite ADD chain
        // using the already loaded dataSize Variable; this avoids allocating a
        // separate Address resource for every peer.
        for (uint32_t rankStep = 0;
             rankStep < ctx.arg->peerRanks[channelIndex]; ++rankStep) {
            destination.addr += ctx.dataSize;
        }
        destination.token = ctx.outputToken;

        ccu::RemoteAddr source;
        source.addr = ctx.arg->registeredRemoteInputs[channelIndex];
        source.token = ctx.remoteTokens[channelIndex];

        const uint16_t remoteMask =
            static_cast<uint16_t>(1U << channelIndex);
        eventMask = static_cast<uint16_t>(eventMask | remoteMask);
        RETURN_IF_CCU_ERROR(ccu::Read(
            ctx.arg->channels[channelIndex], destination, source,
            ctx.dataSize, ctx.event, remoteMask));
    }

    RETURN_IF_CCU_ERROR(ccu::EventWait(ctx.event, eventMask));
    return CCU_SUCCESS;
}

static CcuResult InitSmallPullWarmResources(
    CcuSmallPullWarmKernelContext &ctx)
{
    ctx.remoteTokens.reserve(ctx.arg->channelCount);
    for (uint32_t channelIndex = 0;
         channelIndex < ctx.arg->channelCount; ++channelIndex) {
        ctx.remoteTokens.emplace_back(
            ccu::GetResByChannel<ccu::Variable>(
                ctx.arg->channels[channelIndex], TOKEN_XN_ID));
    }
    return CCU_SUCCESS;
}

static CcuResult LoadSmallPullWarmTaskArgs(
    CcuSmallPullWarmKernelContext &ctx)
{
    uint32_t argIndex = 0;
    RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.output, argIndex++));
    RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.outputToken, argIndex++));
    RETURN_IF_CCU_ERROR(ccu::LoadArg(ctx.dataSize, argIndex++));
    return CCU_SUCCESS;
}

static CcuResult RunSmallPullWarm(CcuSmallPullWarmKernelContext &ctx)
{
    uint16_t eventMask = 0;
    for (uint32_t channelIndex = 0;
         channelIndex < ctx.arg->channelCount; ++channelIndex) {
        ccu::LocalAddr destination;
        destination.addr = ctx.output;
        for (uint32_t rankStep = 0;
             rankStep < ctx.arg->peerRanks[channelIndex]; ++rankStep) {
            destination.addr += ctx.dataSize;
        }
        destination.token = ctx.outputToken;

        ccu::RemoteAddr source;
        source.addr = ctx.arg->registeredRemoteInputs[channelIndex];
        source.token = ctx.remoteTokens[channelIndex];

        const uint16_t remoteMask =
            static_cast<uint16_t>(1U << channelIndex);
        eventMask = static_cast<uint16_t>(eventMask | remoteMask);
        RETURN_IF_CCU_ERROR(ccu::Read(
            ctx.arg->channels[channelIndex], destination, source,
            ctx.dataSize, ctx.event, remoteMask));
    }

    RETURN_IF_CCU_ERROR(ccu::EventWait(ctx.event, eventMask));
    return CCU_SUCCESS;
}

static CcuResult PostSyncOne(CcuAllGatherKernelContext &ctx, uint32_t channelIndex)
{
    RETURN_IF_CCU_ERROR(
        ccu::NotifyRecord(ctx.arg->channels[channelIndex], CHANNEL_NOTIFY_INDEX, POST_SYNC_MASK));
    RETURN_IF_CCU_ERROR(
        ccu::NotifyWait(ctx.arg->channels[channelIndex], CHANNEL_NOTIFY_INDEX, POST_SYNC_MASK));
    return CCU_SUCCESS;
}

static CcuResult PairSync(CcuAllGatherKernelContext &ctx,
    uint32_t channelIndex, uint16_t mask)
{
    RETURN_IF_CCU_ERROR(
        ccu::NotifyRecord(ctx.arg->channels[channelIndex], CHANNEL_NOTIFY_INDEX, mask));
    RETURN_IF_CCU_ERROR(
        ccu::NotifyWait(ctx.arg->channels[channelIndex], CHANNEL_NOTIFY_INDEX, mask));
    return CCU_SUCCESS;
}

static CcuResult IssueDirectSlice(CcuAllGatherKernelContext &ctx, ccu::Variable &sliceOffset,
    ccu::Variable &sliceSize, ccu::Event &event)
{
    ccu::LocalAddr source;
    source.addr = ctx.input;
    source.addr += sliceOffset;
    source.token = ctx.inputToken;

    uint32_t eventIndex = 0;
    CCU_IF(sliceSize != 0)
    {
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            ccu::RemoteAddr destination;
            destination.addr = ctx.remoteOutputs[channelIndex];
            destination.addr += ctx.rankOffset;
            destination.addr += sliceOffset;
            destination.token = ctx.remoteTokens[channelIndex];

            const uint16_t eventMask = static_cast<uint16_t>(1U << eventIndex++);
            RETURN_IF_CCU_ERROR(ccu::Write(ctx.arg->channels[channelIndex], destination, source,
                sliceSize, event, eventMask));
        }

        if (ctx.arg->directCopyLocal != 0) {
            ccu::LocalAddr localDestination;
            localDestination.addr = ctx.output;
            localDestination.addr += ctx.rankOffset;
            localDestination.addr += sliceOffset;
            localDestination.token = ctx.outputToken;

            const uint16_t localMask = static_cast<uint16_t>(1U << eventIndex);
            RETURN_IF_CCU_ERROR(
                ccu::LocalCopy(localDestination, source, sliceSize, event, localMask));
        }
    }
    return CCU_SUCCESS;
}

static CcuResult WaitDirectSlice(CcuAllGatherKernelContext &ctx, ccu::Variable &sliceSize,
    ccu::Event &event)
{
    uint16_t totalMask = 0;
    const uint32_t eventCount = ctx.arg->channelCount + ctx.arg->directCopyLocal;
    for (uint32_t eventIndex = 0; eventIndex < eventCount; ++eventIndex) {
        totalMask = static_cast<uint16_t>(totalMask | static_cast<uint16_t>(1U << eventIndex));
    }

    CCU_IF(sliceSize != 0)
    {
        RETURN_IF_CCU_ERROR(ccu::EventWait(event, totalMask));
    }
    return CCU_SUCCESS;
}

static CcuResult RunDirect(CcuAllGatherKernelContext &ctx)
{
    RETURN_IF_CCU_ERROR(ExchangeRemoteOutputAll(ctx));
    RETURN_IF_CCU_ERROR(
        IssueDirectSlice(ctx, ctx.sliceOffset0, ctx.sliceSize0, ctx.event0));
    RETURN_IF_CCU_ERROR(
        IssueDirectSlice(ctx, ctx.sliceOffset1, ctx.sliceSize1, ctx.event1));
    RETURN_IF_CCU_ERROR(WaitDirectSlice(ctx, ctx.sliceSize0, ctx.event0));
    RETURN_IF_CCU_ERROR(WaitDirectSlice(ctx, ctx.sliceSize1, ctx.event1));
    RETURN_IF_CCU_ERROR(PostSyncAll(ctx));
    return CCU_SUCCESS;
}

static CcuResult PublishRd4RemoteOutputs(CcuAllGatherKernelContext &ctx)
{
    RETURN_IF_CCU_ERROR(
        PublishRemoteOutputOne(ctx, ctx.arg->rd4Xor1ChannelIndex));
    RETURN_IF_CCU_ERROR(
        PublishRemoteOutputOne(ctx, ctx.arg->rd4Xor2ChannelIndex));
    return CCU_SUCCESS;
}

static CcuResult RunRd4(CcuAllGatherKernelContext &ctx)
{
    // Reuse the source and remote destination after Stage0 completes.  This
    // keeps one fewer LocalAddr and one fewer RemoteAddr than the passed RD4.
    ccu::LocalAddr source;
    source.addr = ctx.input;
    source.token = ctx.inputToken;

    ccu::LocalAddr localDestination;
    localDestination.addr = ctx.output;
    localDestination.addr += ctx.rankOffset;
    localDestination.token = ctx.outputToken;

    ccu::RemoteAddr remoteDestination;

    constexpr uint16_t localMask = static_cast<uint16_t>(1U << 0);
    constexpr uint16_t stage0RemoteMask = static_cast<uint16_t>(1U << 1);
    constexpr uint16_t stage0Mask = localMask | stage0RemoteMask;
    constexpr uint16_t stage1Mask = static_cast<uint16_t>(1U << 0);

    // Start the self copy before any control wait.  Publish both peer states
    // before waiting so xor2 state exchange overlaps Stage0.
    RETURN_IF_CCU_ERROR(ccu::LocalCopy(localDestination, source,
        ctx.sliceSize0, ctx.event0, localMask));
    RETURN_IF_CCU_ERROR(PublishRd4RemoteOutputs(ctx));

    const uint32_t xor1ChannelIndex = ctx.arg->rd4Xor1ChannelIndex;
    RETURN_IF_CCU_ERROR(
        WaitRemoteOutputOne(ctx, xor1ChannelIndex));
    remoteDestination.addr = ctx.remoteOutputs[xor1ChannelIndex];
    remoteDestination.addr += ctx.rankOffset;
    remoteDestination.token = ctx.remoteTokens[xor1ChannelIndex];
    RETURN_IF_CCU_ERROR(ccu::Write(ctx.arg->channels[xor1ChannelIndex],
        remoteDestination, source, ctx.sliceSize0, ctx.event0,
        stage0RemoteMask));
    RETURN_IF_CCU_ERROR(ccu::EventWait(ctx.event0, stage0Mask));
    RETURN_IF_CCU_ERROR(
        PairSync(ctx, xor1ChannelIndex, RD4_STAGE0_READY_MASK));

    const uint32_t xor2ChannelIndex = ctx.arg->rd4Xor2ChannelIndex;
    RETURN_IF_CCU_ERROR(
        WaitRemoteOutputOne(ctx, xor2ChannelIndex));

    source.addr = ctx.output;
    source.addr += ctx.partnerOffset;
    source.token = ctx.outputToken;
    remoteDestination.addr = ctx.remoteOutputs[xor2ChannelIndex];
    remoteDestination.addr += ctx.partnerOffset;
    remoteDestination.token = ctx.remoteTokens[xor2ChannelIndex];
    RETURN_IF_CCU_ERROR(ccu::Write(ctx.arg->channels[xor2ChannelIndex],
        remoteDestination, source, ctx.sliceSize1, ctx.event1, stage1Mask));
    RETURN_IF_CCU_ERROR(ccu::EventWait(ctx.event1, stage1Mask));

    // Functional probe: the outgoing xor2 write is complete here.  The
    // symmetric schedule is checked by the platform before this terminal
    // pair barrier is considered redundant.
    return CCU_SUCCESS;
}

static CcuResult RunColumnLayer1(CcuAllGatherKernelContext &ctx)
{
    const uint32_t channelIndex = ctx.arg->columnPeerChannelIndex;
    RETURN_IF_CCU_ERROR(ExchangeRemoteOutputOne(ctx, channelIndex));

    ccu::LocalAddr source;
    source.addr = ctx.input;
    source.addr += ctx.sliceOffset0;
    source.token = ctx.inputToken;

    uint16_t totalMask = 0;
    CCU_IF(ctx.sliceSize0 != 0)
    {
        ccu::RemoteAddr destination;
        destination.addr = ctx.remoteOutputs[channelIndex];
        destination.addr += ctx.rankOffset;
        destination.addr += ctx.sliceOffset0;
        destination.token = ctx.remoteTokens[channelIndex];

        constexpr uint16_t remoteMask = 1U << 0;
        totalMask = static_cast<uint16_t>(totalMask | remoteMask);
        RETURN_IF_CCU_ERROR(ccu::Write(ctx.arg->channels[channelIndex], destination, source,
            ctx.sliceSize0, ctx.event0, remoteMask));

        if (ctx.arg->columnCopyLocal != 0) {
            ccu::LocalAddr localDestination;
            localDestination.addr = ctx.output;
            localDestination.addr += ctx.rankOffset;
            localDestination.addr += ctx.sliceOffset0;
            localDestination.token = ctx.outputToken;

            constexpr uint16_t localMask = 1U << 1;
            totalMask = static_cast<uint16_t>(totalMask | localMask);
            RETURN_IF_CCU_ERROR(ccu::LocalCopy(
                localDestination, source, ctx.sliceSize0, ctx.event0, localMask));
        }

        RETURN_IF_CCU_ERROR(ccu::EventWait(ctx.event0, totalMask));
    }

    RETURN_IF_CCU_ERROR(PostSyncOne(ctx, channelIndex));
    return CCU_SUCCESS;
}

static CcuResult RunColumnLayer0(CcuAllGatherKernelContext &ctx)
{
    RETURN_IF_CCU_ERROR(ExchangeRemoteOutputAll(ctx));

    ccu::LocalAddr ownSource;
    ownSource.addr = ctx.output;
    ownSource.addr += ctx.rankOffset;
    ownSource.addr += ctx.sliceOffset0;
    ownSource.token = ctx.outputToken;

    ccu::LocalAddr partnerSource;
    partnerSource.addr = ctx.output;
    partnerSource.addr += ctx.partnerOffset;
    partnerSource.addr += ctx.sliceOffset0;
    partnerSource.token = ctx.outputToken;

    uint32_t eventIndex = 0;
    uint16_t totalMask = 0;
    CCU_IF(ctx.sliceSize0 != 0)
    {
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            ccu::RemoteAddr ownDestination;
            ownDestination.addr = ctx.remoteOutputs[channelIndex];
            ownDestination.addr += ctx.rankOffset;
            ownDestination.addr += ctx.sliceOffset0;
            ownDestination.token = ctx.remoteTokens[channelIndex];

            const uint16_t ownMask = static_cast<uint16_t>(1U << eventIndex++);
            totalMask = static_cast<uint16_t>(totalMask | ownMask);
            RETURN_IF_CCU_ERROR(ccu::Write(ctx.arg->channels[channelIndex], ownDestination,
                ownSource, ctx.sliceSize0, ctx.event0, ownMask));

            ccu::RemoteAddr partnerDestination;
            partnerDestination.addr = ctx.remoteOutputs[channelIndex];
            partnerDestination.addr += ctx.partnerOffset;
            partnerDestination.addr += ctx.sliceOffset0;
            partnerDestination.token = ctx.remoteTokens[channelIndex];

            const uint16_t partnerMask = static_cast<uint16_t>(1U << eventIndex++);
            totalMask = static_cast<uint16_t>(totalMask | partnerMask);
            RETURN_IF_CCU_ERROR(ccu::Write(ctx.arg->channels[channelIndex], partnerDestination,
                partnerSource, ctx.sliceSize0, ctx.event0, partnerMask));
        }

        RETURN_IF_CCU_ERROR(ccu::EventWait(ctx.event0, totalMask));
    }

    RETURN_IF_CCU_ERROR(PostSyncAll(ctx));
    return CCU_SUCCESS;
}

static CcuResult RunKernel(CcuKernelArg kernelArg)
{
    auto *arg = static_cast<CcuAllGatherKernelArg *>(kernelArg);
    if (arg == nullptr || arg->channelCount == 0 || arg->channelCount >= MAX_RANK_SIZE ||
        arg->columnPeerChannelIndex >= arg->channelCount ||
        arg->rd4Xor1ChannelIndex >= arg->channelCount ||
        arg->rd4Xor2ChannelIndex >= arg->channelCount ||
        arg->channelCount + arg->directCopyLocal > EVENT_BIT_COUNT ||
        2U * arg->channelCount > EVENT_BIT_COUNT) {
        return CCU_E_PARA;
    }

    CcuAllGatherKernelContext ctx;
    ctx.arg = arg;
    RETURN_IF_CCU_ERROR(InitResources(ctx));
    RETURN_IF_CCU_ERROR(LoadTaskArgs(ctx));

    CCU_IF(ctx.mode == static_cast<uint64_t>(AllGatherKernelMode::DIRECT))
    {
        RETURN_IF_CCU_ERROR(RunDirect(ctx));
    }

    CCU_IF(ctx.mode == static_cast<uint64_t>(AllGatherKernelMode::RD4_XOR_TWO_STAGE))
    {
        RETURN_IF_CCU_ERROR(RunRd4(ctx));
    }

    if (arg->layer == static_cast<uint32_t>(AllGatherKernelLayer::LAYER0)) {
        CCU_IF(ctx.mode == static_cast<uint64_t>(AllGatherKernelMode::COLUMN_LAYER0))
        {
            RETURN_IF_CCU_ERROR(RunColumnLayer0(ctx));
        }
    } else {
        CCU_IF(ctx.mode == static_cast<uint64_t>(AllGatherKernelMode::COLUMN_LAYER1))
        {
            RETURN_IF_CCU_ERROR(RunColumnLayer1(ctx));
        }
    }

    return CCU_SUCCESS;
}

static CcuResult RunSmallPullKernel(CcuKernelArg kernelArg)
{
    auto *arg = static_cast<CcuSmallPullKernelArg *>(kernelArg);
    if (arg == nullptr || arg->channelCount == 0 ||
        arg->channelCount > SMALL_DIRECT_MAX_CHANNELS ||
        arg->directCopyLocal > 1 ||
        arg->channelCount + arg->directCopyLocal > EVENT_BIT_COUNT) {
        return CCU_E_PARA;
    }

    CcuSmallPullKernelContext ctx;
    ctx.arg = arg;
    RETURN_IF_CCU_ERROR(InitSmallPullResources(ctx));
    RETURN_IF_CCU_ERROR(LoadSmallPullTaskArgs(ctx));
    return RunSmallPull(ctx);
}

static CcuResult RunSmallPullWarmKernel(CcuKernelArg kernelArg)
{
    auto *arg = static_cast<CcuSmallPullKernelArg *>(kernelArg);
    if (arg == nullptr || arg->channelCount == 0 ||
        arg->channelCount > SMALL_DIRECT_MAX_CHANNELS) {
        return CCU_E_PARA;
    }

    CcuSmallPullWarmKernelContext ctx;
    ctx.arg = arg;
    RETURN_IF_CCU_ERROR(InitSmallPullWarmResources(ctx));
    RETURN_IF_CCU_ERROR(LoadSmallPullWarmTaskArgs(ctx));
    return RunSmallPullWarm(ctx);
}

} // namespace

CcuResult CcuAllGatherLayer0Kernel(CcuKernelArg arg)
{
    return RunKernel(arg);
}

CcuResult CcuAllGatherLayer1Kernel(CcuKernelArg arg)
{
    return RunKernel(arg);
}

CcuResult CcuSmallPullLayer0Kernel(CcuKernelArg arg)
{
    return RunSmallPullKernel(arg);
}

CcuResult CcuSmallPullLayer1Kernel(CcuKernelArg arg)
{
    return RunSmallPullKernel(arg);
}

CcuResult CcuSmallPullWarmLayer0Kernel(CcuKernelArg arg)
{
    return RunSmallPullWarmKernel(arg);
}

CcuResult CcuSmallPullWarmLayer1Kernel(CcuKernelArg arg)
{
    return RunSmallPullWarmKernel(arg);
}

} // namespace ops_hccl

#undef RETURN_IF_CCU_ERROR

namespace ops_hccl {
namespace {

constexpr uint32_t STAGE_OUTPUT_XN_ID = 1;
constexpr uint32_t STAGE_TOKEN_XN_ID = 2;
constexpr uint32_t STAGE_CHANNEL_NOTIFY_INDEX = 0;
constexpr uint16_t STAGE_OUTPUT_READY_MASK =
    static_cast<uint16_t>(1U << STAGE_OUTPUT_XN_ID);
constexpr uint16_t STAGE_TOKEN_READY_MASK =
    static_cast<uint16_t>(1U << STAGE_TOKEN_XN_ID);
constexpr uint16_t STAGE_POST_SYNC_MASK = static_cast<uint16_t>(1U << 3);
constexpr uint16_t STAGE_PARTNER_READY_MASK = static_cast<uint16_t>(1U << 4);
constexpr uint16_t L0_OWN_EVENT_MASK = 0x007F;
constexpr uint16_t L0_RELAY_EVENT_MASK = 0x3F80;
constexpr uint16_t L0_BOTH_EVENT_MASK = 0x3FFF;
constexpr uint16_t L1_PREFIX_EVENT_MASK = 0x0101;
constexpr uint16_t L1_DATA_EVENT_MASK = 0x01FF;

#define RETURN_IF_STAGE_ERROR(call) \
    do { \
        CcuResult result = (call); \
        if (result != CCU_SUCCESS) { \
            return result; \
        } \
    } while (0)

struct CcuM1Layer0StageContext {
    const CcuM1StageKernelArg *arg = nullptr;

    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable rankOffset;
    ccu::Variable partnerOffset;
    ccu::Variable ownOffset;
    ccu::Variable ownSize;
    ccu::Variable relayOffset;
    ccu::Variable relaySize;
    ccu::Variable exchangeRemoteState;
    ccu::Variable finalPostSync;

    std::vector<ccu::Variable> remoteOutputs;
    std::vector<ccu::Variable> remoteTokens;
    ccu::Event event;
};

struct CcuM1Layer1StageContext {
    const CcuM1StageKernelArg *arg = nullptr;

    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable rankOffset;
    ccu::Variable chunkOffset;
    ccu::Variable chunkSize;
    ccu::Variable directOffset;
    ccu::Variable directSize;
    ccu::Variable exchangeRemoteState;
    ccu::Variable partnerReady;
    ccu::Variable finalPostSync;

    std::vector<ccu::Variable> remoteOutputs;
    std::vector<ccu::Variable> remoteTokens;
    ccu::Event event;
};

template <typename Context>
CcuResult InitStageRemoteResources(Context &ctx)
{
    ctx.remoteOutputs.reserve(ctx.arg->channelCount);
    ctx.remoteTokens.reserve(ctx.arg->channelCount);
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        ctx.remoteOutputs.emplace_back(ccu::GetResByChannel<ccu::Variable>(
            ctx.arg->channels[channelIndex], STAGE_OUTPUT_XN_ID));
        ctx.remoteTokens.emplace_back(ccu::GetResByChannel<ccu::Variable>(
            ctx.arg->channels[channelIndex], STAGE_TOKEN_XN_ID));
    }
    return CCU_SUCCESS;
}

template <typename Context>
CcuResult ExchangeStageRemoteOutputAll(Context &ctx)
{
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        RETURN_IF_STAGE_ERROR(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIndex],
            ctx.output, STAGE_OUTPUT_XN_ID, STAGE_CHANNEL_NOTIFY_INDEX,
            STAGE_OUTPUT_READY_MASK));
        RETURN_IF_STAGE_ERROR(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIndex],
            ctx.outputToken, STAGE_TOKEN_XN_ID, STAGE_CHANNEL_NOTIFY_INDEX,
            STAGE_TOKEN_READY_MASK));
    }

    constexpr uint16_t readyMask = STAGE_OUTPUT_READY_MASK | STAGE_TOKEN_READY_MASK;
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        RETURN_IF_STAGE_ERROR(ccu::NotifyWait(ctx.arg->channels[channelIndex],
            STAGE_CHANNEL_NOTIFY_INDEX, readyMask));
    }
    return CCU_SUCCESS;
}

template <typename Context>
CcuResult StagePostSyncAll(Context &ctx)
{
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        RETURN_IF_STAGE_ERROR(ccu::NotifyRecord(ctx.arg->channels[channelIndex],
            STAGE_CHANNEL_NOTIFY_INDEX, STAGE_POST_SYNC_MASK));
    }
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        RETURN_IF_STAGE_ERROR(ccu::NotifyWait(ctx.arg->channels[channelIndex],
            STAGE_CHANNEL_NOTIFY_INDEX, STAGE_POST_SYNC_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult LoadLayer0StageArgs(CcuM1Layer0StageContext &ctx)
{
    uint32_t argIndex = 0;
    RETURN_IF_STAGE_ERROR(ccu::LoadArg(ctx.input, argIndex++));
    RETURN_IF_STAGE_ERROR(ccu::LoadArg(ctx.output, argIndex++));
    RETURN_IF_STAGE_ERROR(ccu::LoadArg(ctx.inputToken, argIndex++));
    RETURN_IF_STAGE_ERROR(ccu::LoadArg(ctx.outputToken, argIndex++));
    RETURN_IF_STAGE_ERROR(ccu::LoadArg(ctx.rankOffset, argIndex++));
    RETURN_IF_STAGE_ERROR(ccu::LoadArg(ctx.partnerOffset, argIndex++));
    RETURN_IF_STAGE_ERROR(ccu::LoadArg(ctx.ownOffset, argIndex++));
    RETURN_IF_STAGE_ERROR(ccu::LoadArg(ctx.ownSize, argIndex++));
    RETURN_IF_STAGE_ERROR(ccu::LoadArg(ctx.relayOffset, argIndex++));
    RETURN_IF_STAGE_ERROR(ccu::LoadArg(ctx.relaySize, argIndex++));
    RETURN_IF_STAGE_ERROR(ccu::LoadArg(ctx.exchangeRemoteState, argIndex++));
    RETURN_IF_STAGE_ERROR(ccu::LoadArg(ctx.finalPostSync, argIndex++));
    return CCU_SUCCESS;
}

CcuResult LoadLayer1StageArgs(CcuM1Layer1StageContext &ctx)
{
    uint32_t argIndex = 0;
    RETURN_IF_STAGE_ERROR(ccu::LoadArg(ctx.input, argIndex++));
    RETURN_IF_STAGE_ERROR(ccu::LoadArg(ctx.output, argIndex++));
    RETURN_IF_STAGE_ERROR(ccu::LoadArg(ctx.inputToken, argIndex++));
    RETURN_IF_STAGE_ERROR(ccu::LoadArg(ctx.outputToken, argIndex++));
    RETURN_IF_STAGE_ERROR(ccu::LoadArg(ctx.rankOffset, argIndex++));
    RETURN_IF_STAGE_ERROR(ccu::LoadArg(ctx.chunkOffset, argIndex++));
    RETURN_IF_STAGE_ERROR(ccu::LoadArg(ctx.chunkSize, argIndex++));
    RETURN_IF_STAGE_ERROR(ccu::LoadArg(ctx.directOffset, argIndex++));
    RETURN_IF_STAGE_ERROR(ccu::LoadArg(ctx.directSize, argIndex++));
    RETURN_IF_STAGE_ERROR(ccu::LoadArg(ctx.exchangeRemoteState, argIndex++));
    RETURN_IF_STAGE_ERROR(ccu::LoadArg(ctx.partnerReady, argIndex++));
    RETURN_IF_STAGE_ERROR(ccu::LoadArg(ctx.finalPostSync, argIndex++));
    return CCU_SUCCESS;
}

CcuResult IssueLayer0Own(CcuM1Layer0StageContext &ctx)
{
    ccu::LocalAddr source;
    source.addr = ctx.input;
    source.addr += ctx.ownOffset;
    source.token = ctx.inputToken;

    CCU_IF(ctx.ownSize != 0)
    {
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            ccu::RemoteAddr destination;
            destination.addr = ctx.remoteOutputs[channelIndex];
            destination.addr += ctx.rankOffset;
            destination.addr += ctx.ownOffset;
            destination.token = ctx.remoteTokens[channelIndex];

            const uint16_t eventMask = static_cast<uint16_t>(1U << channelIndex);
            RETURN_IF_STAGE_ERROR(ccu::Write(ctx.arg->channels[channelIndex], destination,
                source, ctx.ownSize, ctx.event, eventMask));
        }
    }
    return CCU_SUCCESS;
}

CcuResult IssueLayer0Relay(CcuM1Layer0StageContext &ctx)
{
    ccu::LocalAddr source;
    source.addr = ctx.output;
    source.addr += ctx.partnerOffset;
    source.addr += ctx.relayOffset;
    source.token = ctx.outputToken;

    CCU_IF(ctx.relaySize != 0)
    {
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            ccu::RemoteAddr destination;
            destination.addr = ctx.remoteOutputs[channelIndex];
            destination.addr += ctx.partnerOffset;
            destination.addr += ctx.relayOffset;
            destination.token = ctx.remoteTokens[channelIndex];

            const uint16_t eventMask =
                static_cast<uint16_t>(1U << (ctx.arg->channelCount + channelIndex));
            RETURN_IF_STAGE_ERROR(ccu::Write(ctx.arg->channels[channelIndex], destination,
                source, ctx.relaySize, ctx.event, eventMask));
        }
    }
    return CCU_SUCCESS;
}

CcuResult WaitLayer0Data(CcuM1Layer0StageContext &ctx)
{
    // The three waits are mutually exclusive at runtime. Keeping the masks
    // immediate prevents a stage from waiting on an Event bit that was never
    // produced by a Write in that launch.
    CCU_IF(ctx.ownSize != 0)
    {
        CCU_IF(ctx.relaySize == 0)
        {
            RETURN_IF_STAGE_ERROR(ccu::EventWait(ctx.event, L0_OWN_EVENT_MASK));
        }
        CCU_IF(ctx.relaySize != 0)
        {
            RETURN_IF_STAGE_ERROR(ccu::EventWait(ctx.event, L0_BOTH_EVENT_MASK));
        }
    }
    CCU_IF(ctx.ownSize == 0)
    {
        CCU_IF(ctx.relaySize != 0)
        {
            RETURN_IF_STAGE_ERROR(ccu::EventWait(ctx.event, L0_RELAY_EVENT_MASK));
        }
    }
    return CCU_SUCCESS;
}

CcuResult RunLayer0Stage(CcuM1Layer0StageContext &ctx)
{
    CCU_IF(ctx.exchangeRemoteState != 0)
    {
        RETURN_IF_STAGE_ERROR(ExchangeStageRemoteOutputAll(ctx));
    }

    RETURN_IF_STAGE_ERROR(IssueLayer0Own(ctx));
    RETURN_IF_STAGE_ERROR(IssueLayer0Relay(ctx));
    RETURN_IF_STAGE_ERROR(WaitLayer0Data(ctx));

    CCU_IF(ctx.finalPostSync != 0)
    {
        RETURN_IF_STAGE_ERROR(StagePostSyncAll(ctx));
    }
    return CCU_SUCCESS;
}

CcuResult Layer1PartnerReady(CcuM1Layer1StageContext &ctx)
{
    const uint32_t partnerIndex = ctx.arg->partnerChannelIndex;
    RETURN_IF_STAGE_ERROR(ccu::NotifyRecord(ctx.arg->channels[partnerIndex],
        STAGE_CHANNEL_NOTIFY_INDEX, STAGE_PARTNER_READY_MASK));
    RETURN_IF_STAGE_ERROR(ccu::NotifyWait(ctx.arg->channels[partnerIndex],
        STAGE_CHANNEL_NOTIFY_INDEX, STAGE_PARTNER_READY_MASK));
    return CCU_SUCCESS;
}

CcuResult RunLayer1Data(CcuM1Layer1StageContext &ctx)
{
    ccu::LocalAddr partnerSource;
    partnerSource.addr = ctx.input;
    partnerSource.addr += ctx.chunkOffset;
    partnerSource.token = ctx.inputToken;

    ccu::LocalAddr directSource;
    directSource.addr = ctx.input;
    directSource.addr += ctx.directOffset;
    directSource.token = ctx.inputToken;

    const uint32_t partnerIndex = ctx.arg->partnerChannelIndex;
    ccu::RemoteAddr partnerDestination;
    partnerDestination.addr = ctx.remoteOutputs[partnerIndex];
    partnerDestination.addr += ctx.rankOffset;
    partnerDestination.addr += ctx.chunkOffset;
    partnerDestination.token = ctx.remoteTokens[partnerIndex];
    constexpr uint16_t partnerMask = 1U << 0;
    RETURN_IF_STAGE_ERROR(ccu::Write(ctx.arg->channels[partnerIndex],
        partnerDestination, partnerSource, ctx.chunkSize, ctx.event, partnerMask));

    // Prefix-first mode encodes the relay prefix with directSize == 0.  Only
    // the symmetric proxy receives that prefix.  A suffix launch has
    // chunkSize == directSize and sends the same disjoint suffix to all eight
    // remote ranks, including the proxy above.
    CCU_IF(ctx.directSize != 0)
    {
        uint32_t nonpartnerEventIndex = 1;
        for (uint32_t channelIndex = 0;
             channelIndex < ctx.arg->channelCount; ++channelIndex) {
            if (channelIndex == partnerIndex) {
                continue;
            }

            ccu::RemoteAddr destination;
            destination.addr = ctx.remoteOutputs[channelIndex];
            destination.addr += ctx.rankOffset;
            destination.addr += ctx.directOffset;
            destination.token = ctx.remoteTokens[channelIndex];

            const uint16_t directMask =
                static_cast<uint16_t>(1U << nonpartnerEventIndex++);
            RETURN_IF_STAGE_ERROR(ccu::Write(ctx.arg->channels[channelIndex],
                destination, directSource, ctx.directSize, ctx.event, directMask));
        }
    }

    ccu::LocalAddr localDestination;
    localDestination.addr = ctx.output;
    localDestination.addr += ctx.rankOffset;
    localDestination.addr += ctx.chunkOffset;
    localDestination.token = ctx.outputToken;
    constexpr uint16_t localMask = 1U << 8;
    RETURN_IF_STAGE_ERROR(ccu::LocalCopy(localDestination, partnerSource,
        ctx.chunkSize, ctx.event, localMask));

    CCU_IF(ctx.directSize == 0)
    {
        RETURN_IF_STAGE_ERROR(ccu::EventWait(ctx.event, L1_PREFIX_EVENT_MASK));
    }
    CCU_IF(ctx.directSize != 0)
    {
        RETURN_IF_STAGE_ERROR(ccu::EventWait(ctx.event, L1_DATA_EVENT_MASK));
    }

    CCU_IF(ctx.partnerReady != 0)
    {
        RETURN_IF_STAGE_ERROR(Layer1PartnerReady(ctx));
    }
    return CCU_SUCCESS;
}

CcuResult RunLayer1Stage(CcuM1Layer1StageContext &ctx)
{
    CCU_IF(ctx.exchangeRemoteState != 0)
    {
        RETURN_IF_STAGE_ERROR(ExchangeStageRemoteOutputAll(ctx));
    }

    CCU_IF(ctx.chunkSize != 0)
    {
        RETURN_IF_STAGE_ERROR(RunLayer1Data(ctx));
    }

    CCU_IF(ctx.finalPostSync != 0)
    {
        RETURN_IF_STAGE_ERROR(StagePostSyncAll(ctx));
    }
    return CCU_SUCCESS;
}

} // namespace

CcuResult CcuM1Layer0StageKernel(CcuKernelArg kernelArg)
{
    auto *arg = static_cast<CcuM1StageKernelArg *>(kernelArg);
    if (arg == nullptr || arg->channelCount != 7) {
        return CCU_E_PARA;
    }

    CcuM1Layer0StageContext ctx;
    ctx.arg = arg;
    RETURN_IF_STAGE_ERROR(InitStageRemoteResources(ctx));
    RETURN_IF_STAGE_ERROR(LoadLayer0StageArgs(ctx));
    return RunLayer0Stage(ctx);
}

CcuResult CcuM1Layer1StageKernel(CcuKernelArg kernelArg)
{
    auto *arg = static_cast<CcuM1StageKernelArg *>(kernelArg);
    if (arg == nullptr || arg->channelCount != 8 ||
        arg->partnerChannelIndex >= arg->channelCount) {
        return CCU_E_PARA;
    }

    CcuM1Layer1StageContext ctx;
    ctx.arg = arg;
    RETURN_IF_STAGE_ERROR(InitStageRemoteResources(ctx));
    RETURN_IF_STAGE_ERROR(LoadLayer1StageArgs(ctx));
    return RunLayer1Stage(ctx);
}

} // namespace ops_hccl

#undef RETURN_IF_STAGE_ERROR


namespace ops_hccl {
namespace {

constexpr uint32_t A84_OUTPUT_XN_ID = 1;
constexpr uint32_t A84_TOKEN_XN_ID = 2;
constexpr uint32_t A84_CHANNEL_NOTIFY_INDEX = 0;
constexpr uint16_t A84_OUTPUT_READY_MASK =
    static_cast<uint16_t>(1U << A84_OUTPUT_XN_ID);
constexpr uint16_t A84_TOKEN_READY_MASK =
    static_cast<uint16_t>(1U << A84_TOKEN_XN_ID);
constexpr uint16_t A84_POST_SYNC_MASK = static_cast<uint16_t>(1U << 3);
constexpr uint16_t A84_PREFIX_READY_MASK = static_cast<uint16_t>(1U << 4);

#define RETURN_IF_A84_ERROR(call) \
    do { \
        CcuResult result = (call); \
        if (result != CCU_SUCCESS) { \
            return result; \
        } \
    } while (0)

struct A84TaskContext {
    ccu::Variable input;
    ccu::Variable output;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable dataSize;
    ccu::Variable rankOffset;
    ccu::Variable rangeOffset;
    ccu::Variable rangeSize;
    ccu::Variable prefixSize;
    ccu::Variable exchangeRemoteState;
    ccu::Variable finalPostSync;
    ccu::Variable mode;
};

struct A84Layer0Context : A84TaskContext {
    const CcuA84PrefixLayer0KernelArg *arg = nullptr;
    std::vector<ccu::Variable> remoteOutputs;
    std::vector<ccu::Variable> remoteTokens;
    ccu::Event event;
};

struct A84Layer1Context : A84TaskContext {
    const CcuA84PrefixLayer1KernelArg *arg = nullptr;
    std::vector<ccu::Variable> remoteOutputs;
    std::vector<ccu::Variable> remoteTokens;
    ccu::Event event;
};

template <typename Context>
CcuResult LoadA84TaskArgs(Context &ctx)
{
    uint32_t argIndex = 0;
    RETURN_IF_A84_ERROR(ccu::LoadArg(ctx.input, argIndex++));
    RETURN_IF_A84_ERROR(ccu::LoadArg(ctx.output, argIndex++));
    RETURN_IF_A84_ERROR(ccu::LoadArg(ctx.inputToken, argIndex++));
    RETURN_IF_A84_ERROR(ccu::LoadArg(ctx.outputToken, argIndex++));
    RETURN_IF_A84_ERROR(ccu::LoadArg(ctx.dataSize, argIndex++));
    RETURN_IF_A84_ERROR(ccu::LoadArg(ctx.rankOffset, argIndex++));
    RETURN_IF_A84_ERROR(ccu::LoadArg(ctx.rangeOffset, argIndex++));
    RETURN_IF_A84_ERROR(ccu::LoadArg(ctx.rangeSize, argIndex++));
    RETURN_IF_A84_ERROR(ccu::LoadArg(ctx.prefixSize, argIndex++));
    RETURN_IF_A84_ERROR(ccu::LoadArg(ctx.exchangeRemoteState, argIndex++));
    RETURN_IF_A84_ERROR(ccu::LoadArg(ctx.finalPostSync, argIndex++));
    RETURN_IF_A84_ERROR(ccu::LoadArg(ctx.mode, argIndex++));
    return CCU_SUCCESS;
}

template <typename Context>
CcuResult InitA84RemoteResources(Context &ctx)
{
    ctx.remoteOutputs.reserve(ctx.arg->channelCount);
    ctx.remoteTokens.reserve(ctx.arg->channelCount);
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        ctx.remoteOutputs.emplace_back(ccu::GetResByChannel<ccu::Variable>(
            ctx.arg->channels[channelIndex], A84_OUTPUT_XN_ID));
        ctx.remoteTokens.emplace_back(ccu::GetResByChannel<ccu::Variable>(
            ctx.arg->channels[channelIndex], A84_TOKEN_XN_ID));
    }
    return CCU_SUCCESS;
}

template <typename Context>
CcuResult ExchangeA84RemoteOutputAll(Context &ctx)
{
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        RETURN_IF_A84_ERROR(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIndex],
            ctx.output, A84_OUTPUT_XN_ID, A84_CHANNEL_NOTIFY_INDEX,
            A84_OUTPUT_READY_MASK));
        RETURN_IF_A84_ERROR(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIndex],
            ctx.outputToken, A84_TOKEN_XN_ID, A84_CHANNEL_NOTIFY_INDEX,
            A84_TOKEN_READY_MASK));
    }
    constexpr uint16_t readyMask = A84_OUTPUT_READY_MASK | A84_TOKEN_READY_MASK;
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        RETURN_IF_A84_ERROR(ccu::NotifyWait(ctx.arg->channels[channelIndex],
            A84_CHANNEL_NOTIFY_INDEX, readyMask));
    }
    return CCU_SUCCESS;
}

template <typename Context>
CcuResult PostSyncA84All(Context &ctx)
{
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        RETURN_IF_A84_ERROR(ccu::NotifyRecord(ctx.arg->channels[channelIndex],
            A84_CHANNEL_NOTIFY_INDEX, A84_POST_SYNC_MASK));
    }
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        RETURN_IF_A84_ERROR(ccu::NotifyWait(ctx.arg->channels[channelIndex],
            A84_CHANNEL_NOTIFY_INDEX, A84_POST_SYNC_MASK));
    }
    return CCU_SUCCESS;
}

uint16_t MakeA84EventMask(uint32_t eventCount)
{
    uint16_t mask = 0;
    for (uint32_t index = 0; index < eventCount; ++index) {
        mask = static_cast<uint16_t>(mask | static_cast<uint16_t>(1U << index));
    }
    return mask;
}

void AddA84RankStride(ccu::Address &address, ccu::Variable &dataSize, uint32_t rank)
{
    for (uint32_t rankStep = 0; rankStep < rank; ++rankStep) {
        address += dataSize;
    }
}

CcuResult RunA84Layer0Own(A84Layer0Context &ctx)
{
    ccu::LocalAddr source;
    source.addr = ctx.input;
    source.addr += ctx.rangeOffset;
    source.token = ctx.inputToken;

    uint32_t eventIndex = 0;
    CCU_IF(ctx.rangeSize != 0)
    {
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            ccu::RemoteAddr destination;
            destination.addr = ctx.remoteOutputs[channelIndex];
            destination.addr += ctx.rankOffset;
            destination.addr += ctx.rangeOffset;
            destination.token = ctx.remoteTokens[channelIndex];
            const uint16_t mask = static_cast<uint16_t>(1U << eventIndex++);
            RETURN_IF_A84_ERROR(ccu::Write(ctx.arg->channels[channelIndex], destination,
                source, ctx.rangeSize, ctx.event, mask));
        }

        if (ctx.arg->copyLocal != 0) {
            ccu::LocalAddr localDestination;
            localDestination.addr = ctx.output;
            localDestination.addr += ctx.rankOffset;
            localDestination.addr += ctx.rangeOffset;
            localDestination.token = ctx.outputToken;
            const uint16_t localMask = static_cast<uint16_t>(1U << eventIndex++);
            RETURN_IF_A84_ERROR(ccu::LocalCopy(localDestination, source,
                ctx.rangeSize, ctx.event, localMask));
        }
        RETURN_IF_A84_ERROR(ccu::EventWait(ctx.event, MakeA84EventMask(eventIndex)));
    }
    return CCU_SUCCESS;
}

CcuResult RunA84Layer0Relay(A84Layer0Context &ctx)
{
    uint32_t eventIndex = 0;
    CCU_IF(ctx.prefixSize != 0)
    {
        for (uint32_t sourceIndex = 0;
             sourceIndex < ctx.arg->relaySourceCount; ++sourceIndex) {
            const uint32_t sourceRank = ctx.arg->relaySourceRanks[sourceIndex];
            ccu::LocalAddr source;
            source.addr = ctx.output;
            AddA84RankStride(source.addr, ctx.dataSize, sourceRank);
            source.token = ctx.outputToken;

            for (uint32_t channelIndex = 0;
                 channelIndex < ctx.arg->channelCount; ++channelIndex) {
                ccu::RemoteAddr destination;
                destination.addr = ctx.remoteOutputs[channelIndex];
                AddA84RankStride(destination.addr, ctx.dataSize, sourceRank);
                destination.token = ctx.remoteTokens[channelIndex];
                const uint16_t mask = static_cast<uint16_t>(1U << eventIndex++);
                RETURN_IF_A84_ERROR(ccu::Write(ctx.arg->channels[channelIndex],
                    destination, source, ctx.prefixSize, ctx.event, mask));
            }
        }
        if (ctx.arg->relaySourceCount != 0) {
            RETURN_IF_A84_ERROR(ccu::EventWait(
                ctx.event, MakeA84EventMask(eventIndex)));
        }
    }
    return CCU_SUCCESS;
}

CcuResult RunA84Layer0(A84Layer0Context &ctx)
{
    CCU_IF(ctx.exchangeRemoteState != 0)
    {
        RETURN_IF_A84_ERROR(ExchangeA84RemoteOutputAll(ctx));
    }
    CCU_IF(ctx.mode == static_cast<uint64_t>(A84PrefixKernelMode::LAYER0_OWN))
    {
        RETURN_IF_A84_ERROR(RunA84Layer0Own(ctx));
    }
    CCU_IF(ctx.mode == static_cast<uint64_t>(A84PrefixKernelMode::LAYER0_RELAY))
    {
        RETURN_IF_A84_ERROR(RunA84Layer0Relay(ctx));
    }
    CCU_IF(ctx.finalPostSync != 0)
    {
        RETURN_IF_A84_ERROR(PostSyncA84All(ctx));
    }
    return CCU_SUCCESS;
}

CcuResult RunA84Layer1Prefix(A84Layer1Context &ctx)
{
    const uint32_t proxyIndex = ctx.arg->outgoingProxyChannelIndex;
    ccu::LocalAddr source;
    source.addr = ctx.input;
    source.token = ctx.inputToken;

    ccu::RemoteAddr destination;
    destination.addr = ctx.remoteOutputs[proxyIndex];
    destination.addr += ctx.rankOffset;
    destination.token = ctx.remoteTokens[proxyIndex];

    uint32_t eventCount = 1;
    constexpr uint16_t prefixMask = 1U << 0;
    RETURN_IF_A84_ERROR(ccu::Write(ctx.arg->channels[proxyIndex], destination,
        source, ctx.prefixSize, ctx.event, prefixMask));
    if (ctx.arg->copyLocal != 0) {
        ccu::LocalAddr localDestination;
        localDestination.addr = ctx.output;
        localDestination.addr += ctx.rankOffset;
        localDestination.token = ctx.outputToken;
        const uint16_t localMask = static_cast<uint16_t>(1U << eventCount++);
        RETURN_IF_A84_ERROR(ccu::LocalCopy(localDestination, source,
            ctx.prefixSize, ctx.event, localMask));
    }
    RETURN_IF_A84_ERROR(ccu::EventWait(ctx.event, MakeA84EventMask(eventCount)));

    // EventWait makes the outgoing Prefix visible before the remote bit4.
    RETURN_IF_A84_ERROR(ccu::NotifyRecord(ctx.arg->channels[proxyIndex],
        A84_CHANNEL_NOTIFY_INDEX, A84_PREFIX_READY_MASK));
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        if ((ctx.arg->incomingProxyMask & static_cast<uint32_t>(1U << channelIndex)) == 0) {
            continue;
        }
        RETURN_IF_A84_ERROR(ccu::NotifyWait(ctx.arg->channels[channelIndex],
            A84_CHANNEL_NOTIFY_INDEX, A84_PREFIX_READY_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult RunA84Layer1Suffix(A84Layer1Context &ctx)
{
    ccu::LocalAddr source;
    source.addr = ctx.input;
    source.addr += ctx.rangeOffset;
    source.token = ctx.inputToken;

    CCU_IF(ctx.rangeSize != 0)
    {
        uint32_t eventCount = ctx.arg->channelCount;
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            ccu::RemoteAddr destination;
            destination.addr = ctx.remoteOutputs[channelIndex];
            destination.addr += ctx.rankOffset;
            destination.addr += ctx.rangeOffset;
            destination.token = ctx.remoteTokens[channelIndex];
            const uint16_t mask = static_cast<uint16_t>(1U << channelIndex);
            RETURN_IF_A84_ERROR(ccu::Write(ctx.arg->channels[channelIndex], destination,
                source, ctx.rangeSize, ctx.event, mask));
        }
        if (ctx.arg->copyLocal != 0) {
            ccu::LocalAddr localDestination;
            localDestination.addr = ctx.output;
            localDestination.addr += ctx.rankOffset;
            localDestination.addr += ctx.rangeOffset;
            localDestination.token = ctx.outputToken;
            const uint16_t localMask = static_cast<uint16_t>(1U << eventCount++);
            RETURN_IF_A84_ERROR(ccu::LocalCopy(localDestination, source,
                ctx.rangeSize, ctx.event, localMask));
        }
        RETURN_IF_A84_ERROR(ccu::EventWait(
            ctx.event, MakeA84EventMask(eventCount)));
    }
    return CCU_SUCCESS;
}

CcuResult RunA84Layer1(A84Layer1Context &ctx)
{
    CCU_IF(ctx.exchangeRemoteState != 0)
    {
        RETURN_IF_A84_ERROR(ExchangeA84RemoteOutputAll(ctx));
    }
    CCU_IF(ctx.mode == static_cast<uint64_t>(A84PrefixKernelMode::LAYER1_PREFIX))
    {
        RETURN_IF_A84_ERROR(RunA84Layer1Prefix(ctx));
    }
    CCU_IF(ctx.mode == static_cast<uint64_t>(A84PrefixKernelMode::LAYER1_SUFFIX))
    {
        RETURN_IF_A84_ERROR(RunA84Layer1Suffix(ctx));
    }
    CCU_IF(ctx.finalPostSync != 0)
    {
        RETURN_IF_A84_ERROR(PostSyncA84All(ctx));
    }
    return CCU_SUCCESS;
}

} // namespace

CcuResult CcuA84PrefixLayer0Kernel(CcuKernelArg kernelArg)
{
    auto *arg = static_cast<CcuA84PrefixLayer0KernelArg *>(kernelArg);
    if (arg == nullptr || (arg->channelCount != 3 && arg->channelCount != 7) ||
        arg->myRank >= 12 || arg->copyLocal > 1 || arg->relaySourceCount > 2 ||
        arg->channelCount * arg->relaySourceCount > 16) {
        return CCU_E_PARA;
    }
    for (uint32_t index = 0; index < arg->relaySourceCount; ++index) {
        if (arg->relaySourceRanks[index] >= 12) {
            return CCU_E_PARA;
        }
    }

    A84Layer0Context ctx;
    ctx.arg = arg;
    RETURN_IF_A84_ERROR(InitA84RemoteResources(ctx));
    RETURN_IF_A84_ERROR(LoadA84TaskArgs(ctx));
    return RunA84Layer0(ctx);
}

CcuResult CcuA84PrefixLayer1Kernel(CcuKernelArg kernelArg)
{
    auto *arg = static_cast<CcuA84PrefixLayer1KernelArg *>(kernelArg);
    if (arg == nullptr || (arg->channelCount != 4 && arg->channelCount != 8) ||
        arg->outgoingProxyChannelIndex >= arg->channelCount || arg->copyLocal > 1 ||
        (arg->incomingProxyMask >> arg->channelCount) != 0) {
        return CCU_E_PARA;
    }

    A84Layer1Context ctx;
    ctx.arg = arg;
    RETURN_IF_A84_ERROR(InitA84RemoteResources(ctx));
    RETURN_IF_A84_ERROR(LoadA84TaskArgs(ctx));
    return RunA84Layer1(ctx);
}

} // namespace ops_hccl

#undef RETURN_IF_A84_ERROR
