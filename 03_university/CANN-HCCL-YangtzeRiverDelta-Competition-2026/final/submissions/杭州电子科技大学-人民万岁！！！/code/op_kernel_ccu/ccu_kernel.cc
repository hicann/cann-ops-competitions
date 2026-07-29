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
namespace ccu = ::AscendC::ccu;

constexpr uint32_t BUFFER_XN_ID = 0;
constexpr uint32_t TOKEN_XN_ID = 1;
constexpr uint32_t PRE_SYNC_NOTIFY_IDX = 0;
constexpr uint32_t POST_SYNC_NOTIFY_IDX = 1;
constexpr uint16_t BUFFER_READY_MASK = 1U << BUFFER_XN_ID;
constexpr uint16_t TOKEN_READY_MASK = 1U << TOKEN_XN_ID;
constexpr uint16_t SCATTER_SYNC_MASK = 1U;
constexpr uint16_t ALLGATHER_SYNC_MASK = 1U << 1;
constexpr uint16_t ASYM_DIRECT_DONE_MASK = 1U;
constexpr uint16_t ASYM_PULL_READ_DONE_ACK_MASK = 1U;
constexpr uint16_t ASYM_PULL_READ_EVENT_MASK = 1U;
constexpr uint16_t CHAIN_READY_MASK = 1U;
constexpr uint16_t CHAIN_ACK_MASK = 1U << 1;
constexpr uint16_t CHAIN_FINAL_MASK = 1U << 2;
// A pair has two independent READY/ACK slots.  The slots are deliberately
// separated so that two records can be outstanding without ever recording the
// same notify bit twice before its wait consumes it.
constexpr uint16_t CHAIN_PAIR_READY0_MASK = 1U << 0;
constexpr uint16_t CHAIN_PAIR_ACK0_MASK = 1U << 1;
constexpr uint16_t CHAIN_PAIR_READY1_MASK = 1U << 3;
constexpr uint16_t CHAIN_PAIR_ACK1_MASK = 1U << 4;
// PRE_SYNC uses bits 0 and 1 on notify index 0 and consumes both before data
// transfer starts.  Bit 2 is therefore an independent, reusable-broadcast-safe
// completion signal for the final reverse drain.
constexpr uint16_t CHAIN_UNIQUE_FINAL_MASK = 1U << 2;
constexpr uint64_t ASYMMETRIC_DIRECT_512K_SIZE = 512ULL * 1024ULL;

#define CCU_CHECK_RET(call) \
    do { \
        const CcuResult ccuRet = static_cast<CcuResult>(call); \
        if (ccuRet != CCU_SUCCESS) { \
            return ccuRet; \
        } \
    } while (0)

struct DirectBroadcastContext {
    CcuKernelArgBroadcast *arg;
    std::vector<ccu::Variable> remoteBuffer;
    std::vector<ccu::Variable> remoteToken;
    ccu::Variable localBuffer;
    ccu::Variable localToken;
    ccu::Variable dataSize;
    ccu::Variable localOffset;
    std::vector<ccu::Variable> sliceOffsets;
    std::vector<ccu::Variable> sliceSizes;
    ccu::Event event;
};

static CcuResult InitResources(DirectBroadcastContext &ctx)
{
    if (ctx.arg->channelCount == 0) {
        return CcuResult::CCU_E_INTERNAL;
    }

    ctx.remoteBuffer.reserve(ctx.arg->channelCount);
    ctx.remoteToken.reserve(ctx.arg->channelCount);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; channelIdx++) {
        ctx.remoteBuffer.push_back(
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], BUFFER_XN_ID));
        ctx.remoteToken.push_back(
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], TOKEN_XN_ID));
    }
    return CCU_SUCCESS;
}

static CcuResult LoadTaskArgs(DirectBroadcastContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHECK_RET(ccu::LoadArg(ctx.localBuffer, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.localToken, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.dataSize, argId++));
    return CCU_SUCCESS;
}

// v0.14A removes the third task argument for the only fixed small-message
// case in the contest.  Assigning the immutable size while recording the
// kernel keeps the launch ABI at {buffer, token}; the communication protocol
// itself remains byte-for-byte identical to the validated asymmetric path.
static CcuResult LoadAsymmetricDirect512K2Args(DirectBroadcastContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHECK_RET(ccu::LoadArg(ctx.localBuffer, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.localToken, argId++));
    ctx.dataSize = ASYMMETRIC_DIRECT_512K_SIZE;
    return CCU_SUCCESS;
}

// The zero-task-argument graph is valid only for the exact memory identity
// captured during resource construction.  The Host performs that guard; this
// loader merely materializes the registered immediates and fixed byte count.
static CcuResult LoadAsymmetricDirect512KRegistered(DirectBroadcastContext &ctx)
{
    ctx.localBuffer = ctx.arg->registeredBaseAddr;
    ctx.localToken = ctx.arg->registeredToken;
    ctx.dataSize = ASYMMETRIC_DIRECT_512K_SIZE;
    return CCU_SUCCESS;
}

// Non-root never issues a DMA in the asymmetric star graph and therefore has
// no use for the fixed byte count.  It still initializes buffer/token before
// every collective because the root consumes the two remote XN values in that
// same collective.  This is intentionally not a cross-invocation cache.
static CcuResult LoadAsymmetricDirect512KRegisteredNonRoot(DirectBroadcastContext &ctx)
{
    ctx.localBuffer = ctx.arg->registeredBaseAddr;
    ctx.localToken = ctx.arg->registeredToken;
    return CCU_SUCCESS;
}

static CcuResult LoadScatterTaskArgs(DirectBroadcastContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHECK_RET(ccu::LoadArg(ctx.localBuffer, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.localToken, argId++));
    ctx.sliceOffsets.reserve(ctx.arg->channelCount);
    ctx.sliceSizes.reserve(ctx.arg->channelCount);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; channelIdx++) {
        ctx.sliceOffsets.emplace_back();
        CCU_CHECK_RET(ccu::LoadArg(ctx.sliceOffsets.back(), argId++));
        ctx.sliceSizes.emplace_back();
        CCU_CHECK_RET(ccu::LoadArg(ctx.sliceSizes.back(), argId++));
    }
    return CCU_SUCCESS;
}

static CcuResult LoadAllGatherTaskArgs(DirectBroadcastContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHECK_RET(ccu::LoadArg(ctx.localBuffer, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.localToken, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.localOffset, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.dataSize, argId++));
    return CCU_SUCCESS;
}

static CcuResult LoadChainTaskArgs(DirectBroadcastContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHECK_RET(ccu::LoadArg(ctx.localBuffer, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.localToken, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.localOffset, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.dataSize, argId++));
    return CCU_SUCCESS;
}

static CcuResult LoadChainPairTaskArgs(DirectBroadcastContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHECK_RET(ccu::LoadArg(ctx.localBuffer, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.localToken, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.localOffset, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.dataSize, argId++));
    ccu::Variable pairOffset;
    ccu::Variable pairDataSize;
    CCU_CHECK_RET(ccu::LoadArg(pairOffset, argId++));
    CCU_CHECK_RET(ccu::LoadArg(pairDataSize, argId++));
    ctx.sliceOffsets.push_back(pairOffset);
    ctx.sliceSizes.push_back(pairDataSize);
    return CCU_SUCCESS;
}

static CcuResult LoadChainQuadTaskArgs(DirectBroadcastContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHECK_RET(ccu::LoadArg(ctx.localBuffer, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.localToken, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.localOffset, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.dataSize, argId++));
    ctx.sliceOffsets.reserve(QUAD4_SEGMENTS_PER_KERNEL - 1);
    ctx.sliceSizes.reserve(QUAD4_SEGMENTS_PER_KERNEL - 1);
    for (uint32_t sliceIdx = 1; sliceIdx < QUAD4_SEGMENTS_PER_KERNEL; sliceIdx++) {
        ctx.sliceOffsets.emplace_back();
        CCU_CHECK_RET(ccu::LoadArg(ctx.sliceOffsets.back(), argId++));
        ctx.sliceSizes.emplace_back();
        CCU_CHECK_RET(ccu::LoadArg(ctx.sliceSizes.back(), argId++));
    }
    return CCU_SUCCESS;
}

static CcuResult LoadChainFiveTaskArgs(DirectBroadcastContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHECK_RET(ccu::LoadArg(ctx.localBuffer, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.localToken, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.localOffset, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.dataSize, argId++));
    ctx.sliceOffsets.reserve(FINAL5_SEGMENTS_PER_KERNEL - 1);
    ctx.sliceSizes.reserve(FINAL5_SEGMENTS_PER_KERNEL - 1);
    for (uint32_t sliceIdx = 1; sliceIdx < FINAL5_SEGMENTS_PER_KERNEL; sliceIdx++) {
        ctx.sliceOffsets.emplace_back();
        CCU_CHECK_RET(ccu::LoadArg(ctx.sliceOffsets.back(), argId++));
        ctx.sliceSizes.emplace_back();
        CCU_CHECK_RET(ccu::LoadArg(ctx.sliceSizes.back(), argId++));
    }
    return CCU_SUCCESS;
}

static CcuResult LoadPersistentTaskArgs(DirectBroadcastContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHECK_RET(ccu::LoadArg(ctx.localBuffer, argId++));
    CCU_CHECK_RET(ccu::LoadArg(ctx.localToken, argId++));
    return CCU_SUCCESS;
}

static CcuResult PreSync(DirectBroadcastContext &ctx)
{
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; channelIdx++) {
        CCU_CHECK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx], ctx.localBuffer, BUFFER_XN_ID,
            PRE_SYNC_NOTIFY_IDX, BUFFER_READY_MASK));
        CCU_CHECK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx], ctx.localToken, TOKEN_XN_ID,
            PRE_SYNC_NOTIFY_IDX, TOKEN_READY_MASK));
    }

    constexpr uint16_t allReadyMask = BUFFER_READY_MASK | TOKEN_READY_MASK;
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; channelIdx++) {
        CCU_CHECK_RET(ccu::NotifyWait(ctx.arg->channels[channelIdx], PRE_SYNC_NOTIFY_IDX, allReadyMask));
    }
    return CCU_SUCCESS;
}

static CcuResult AsymmetricDirectPreSync(DirectBroadcastContext &ctx)
{
    constexpr uint16_t allReadyMask = BUFFER_READY_MASK | TOKEN_READY_MASK;
    if (ctx.arg->rankId == ctx.arg->rootId) {
        // Root only needs each receiver's destination address and token.  It
        // does not publish its source metadata because no peer reads from it.
        for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; channelIdx++) {
            CCU_CHECK_RET(ccu::NotifyWait(
                ctx.arg->channels[channelIdx], PRE_SYNC_NOTIFY_IDX, allReadyMask));
        }
        return CCU_SUCCESS;
    }

    if (ctx.arg->channelCount != 1) {
        return CcuResult::CCU_E_INTERNAL;
    }
    CCU_CHECK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[0], ctx.localBuffer, BUFFER_XN_ID,
        PRE_SYNC_NOTIFY_IDX, BUFFER_READY_MASK));
    CCU_CHECK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[0], ctx.localToken, TOKEN_XN_ID,
        PRE_SYNC_NOTIFY_IDX, TOKEN_READY_MASK));
    return CCU_SUCCESS;
}

static CcuResult BroadcastData(DirectBroadcastContext &ctx)
{
    if (ctx.arg->rankId != ctx.arg->rootId) {
        return CCU_SUCCESS;
    }

    ccu::LocalAddr src;
    src.addr = ctx.localBuffer;
    src.token = ctx.localToken;

    uint16_t sendMask = 0;
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; channelIdx++) {
        ccu::RemoteAddr dst;
        dst.addr = ctx.remoteBuffer[channelIdx];
        dst.token = ctx.remoteToken[channelIdx];

        const uint16_t channelMask = static_cast<uint16_t>(1U << channelIdx);
        CCU_CHECK_RET(
            ccu::Write(ctx.arg->channels[channelIdx], dst, src, ctx.dataSize, ctx.event, channelMask));
        sendMask = static_cast<uint16_t>(sendMask | channelMask);
    }

    CCU_CHECK_RET(ccu::EventWait(ctx.event, sendMask));
    return CCU_SUCCESS;
}

static CcuResult ScatterData(DirectBroadcastContext &ctx)
{
    if (ctx.arg->rankId != ctx.arg->rootId) {
        return CCU_SUCCESS;
    }

    uint16_t sendMask = 0;
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; channelIdx++) {
        ccu::LocalAddr src;
        src.addr = ctx.localBuffer;
        src.addr += ctx.sliceOffsets[channelIdx];
        src.token = ctx.localToken;

        ccu::RemoteAddr dst;
        dst.addr = ctx.remoteBuffer[channelIdx];
        dst.addr += ctx.sliceOffsets[channelIdx];
        dst.token = ctx.remoteToken[channelIdx];

        const uint16_t channelMask = static_cast<uint16_t>(1U << channelIdx);
        CCU_CHECK_RET(
            ccu::Write(ctx.arg->channels[channelIdx], dst, src, ctx.sliceSizes[channelIdx], ctx.event, channelMask));
        sendMask = static_cast<uint16_t>(sendMask | channelMask);
    }
    CCU_CHECK_RET(ccu::EventWait(ctx.event, sendMask));
    return CCU_SUCCESS;
}

static CcuResult AllGatherData(DirectBroadcastContext &ctx)
{
    if (ctx.arg->leanHasRootChannel > 1) {
        return CcuResult::CCU_E_INTERNAL;
    }
    const bool skipRootChannel = V05B_ENABLE_SKIP_ROOT_DMA &&
        ctx.arg->rankId != ctx.arg->rootId && ctx.arg->leanHasRootChannel == 1;
    if (skipRootChannel && ctx.arg->leanRootChannelIdx >= ctx.arg->channelCount) {
        return CcuResult::CCU_E_INTERNAL;
    }

    ccu::LocalAddr src;
    src.addr = ctx.localBuffer;
    src.addr += ctx.localOffset;
    src.token = ctx.localToken;

    uint16_t sendMask = 0;
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; channelIdx++) {
        // The root already owns the complete source buffer.  Omitting this DMA
        // is safe because PhaseSync below still includes the root channel and
        // therefore preserves the collective completion condition.
        if (skipRootChannel && channelIdx == ctx.arg->leanRootChannelIdx) {
            continue;
        }
        ccu::RemoteAddr dst;
        dst.addr = ctx.remoteBuffer[channelIdx];
        dst.addr += ctx.localOffset;
        dst.token = ctx.remoteToken[channelIdx];

        const uint16_t channelMask = static_cast<uint16_t>(1U << channelIdx);
        CCU_CHECK_RET(ccu::Write(ctx.arg->channels[channelIdx], dst, src, ctx.dataSize, ctx.event, channelMask));
        sendMask = static_cast<uint16_t>(sendMask | channelMask);
    }
    // A topology/die group may contain only the root channel.  After the
    // optimization it has no DMA event to consume; EventWait(mask=0) is not a
    // valid no-op on all CCU toolchains, so avoid emitting it.
    if (sendMask != 0) {
        CCU_CHECK_RET(ccu::EventWait(ctx.event, sendMask));
    }
    return CCU_SUCCESS;
}

static CcuResult PhaseSync(DirectBroadcastContext &ctx, uint16_t mask)
{
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; channelIdx++) {
        CCU_CHECK_RET(ccu::NotifyRecord(ctx.arg->channels[channelIdx], POST_SYNC_NOTIFY_IDX, mask));
    }
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; channelIdx++) {
        CCU_CHECK_RET(ccu::NotifyWait(ctx.arg->channels[channelIdx], POST_SYNC_NOTIFY_IDX, mask));
    }
    return CCU_SUCCESS;
}

static CcuResult AsymmetricDirectComplete(DirectBroadcastContext &ctx)
{
    if (ctx.arg->rankId == ctx.arg->rootId) {
        // BroadcastData has already consumed every DMA completion event, so a
        // receiver observing DATA_DONE is guaranteed to see its final data.
        for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; channelIdx++) {
            CCU_CHECK_RET(ccu::NotifyRecord(
                ctx.arg->channels[channelIdx], POST_SYNC_NOTIFY_IDX, ASYM_DIRECT_DONE_MASK));
        }
        return CCU_SUCCESS;
    }

    if (ctx.arg->channelCount != 1) {
        return CcuResult::CCU_E_INTERNAL;
    }
    return static_cast<CcuResult>(ccu::NotifyWait(
        ctx.arg->channels[0], POST_SYNC_NOTIFY_IDX, ASYM_DIRECT_DONE_MASK));
}

// Root publishes only its source identity.  Every receiver performs the
// 512 KiB Read on its sole root channel, waits for the local DMA event and
// acknowledges completion.  Root waits for every ACK before its stream may
// release the source buffer, preserving full Broadcast completion semantics.
static CcuResult AsymmetricPull512K(DirectBroadcastContext &ctx)
{
    constexpr uint16_t sourceReadyMask = BUFFER_READY_MASK | TOKEN_READY_MASK;
    if (ctx.arg->rankId == ctx.arg->rootId) {
        for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; channelIdx++) {
            CCU_CHECK_RET(ccu::WriteVariableWithNotify(
                ctx.arg->channels[channelIdx], ctx.localBuffer, BUFFER_XN_ID,
                PRE_SYNC_NOTIFY_IDX, BUFFER_READY_MASK));
            CCU_CHECK_RET(ccu::WriteVariableWithNotify(
                ctx.arg->channels[channelIdx], ctx.localToken, TOKEN_XN_ID,
                PRE_SYNC_NOTIFY_IDX, TOKEN_READY_MASK));
        }
        for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; channelIdx++) {
            CCU_CHECK_RET(ccu::NotifyWait(
                ctx.arg->channels[channelIdx], POST_SYNC_NOTIFY_IDX,
                ASYM_PULL_READ_DONE_ACK_MASK));
        }
        return CCU_SUCCESS;
    }

    if (ctx.arg->channelCount != 1) {
        return CcuResult::CCU_E_INTERNAL;
    }
    CCU_CHECK_RET(ccu::NotifyWait(
        ctx.arg->channels[0], PRE_SYNC_NOTIFY_IDX, sourceReadyMask));

    ccu::LocalAddr dst;
    dst.addr = ctx.localBuffer;
    dst.token = ctx.localToken;
    ccu::RemoteAddr src;
    src.addr = ctx.remoteBuffer[0];
    src.token = ctx.remoteToken[0];
    CCU_CHECK_RET(ccu::Read(
        ctx.arg->channels[0], dst, src, ctx.dataSize, ctx.event,
        ASYM_PULL_READ_EVENT_MASK));
    CCU_CHECK_RET(ccu::EventWait(ctx.event, ASYM_PULL_READ_EVENT_MASK));
    CCU_CHECK_RET(ccu::NotifyRecord(
        ctx.arg->channels[0], POST_SYNC_NOTIFY_IDX,
        ASYM_PULL_READ_DONE_ACK_MASK));
    return CCU_SUCCESS;
}

// Lean-SAG replaces the full scatter barrier with one READY edge per rank.
// The root records only after all root->peer slice writes have completed.
// A non-root group waits only when that group owns the single channel to root;
// groups without root can return immediately and are fenced by Host-side
// PostSyncThreads before AllGather is launched.
static CcuResult LeanScatterReady(DirectBroadcastContext &ctx)
{
    if (ctx.arg->rankId == ctx.arg->rootId) {
        for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; channelIdx++) {
            CCU_CHECK_RET(ccu::NotifyRecord(
                ctx.arg->channels[channelIdx], POST_SYNC_NOTIFY_IDX, SCATTER_SYNC_MASK));
        }
        return CCU_SUCCESS;
    }

    if (ctx.arg->leanHasRootChannel > 1) {
        return CcuResult::CCU_E_INTERNAL;
    }
    if (ctx.arg->leanHasRootChannel == 0) {
        return CCU_SUCCESS;
    }
    if (ctx.arg->leanRootChannelIdx >= ctx.arg->channelCount) {
        return CcuResult::CCU_E_INTERNAL;
    }
    return static_cast<CcuResult>(ccu::NotifyWait(
        ctx.arg->channels[ctx.arg->leanRootChannelIdx], POST_SYNC_NOTIFY_IDX, SCATTER_SYNC_MASK));
}

static CcuResult ChainTransfer(DirectBroadcastContext &ctx)
{
    if (ctx.arg->hasNext != 0 && ctx.arg->chainIsFirst == 0) {
        CCU_CHECK_RET(ccu::NotifyWait(
            ctx.arg->channels[ctx.arg->nextChannelIdx], POST_SYNC_NOTIFY_IDX, CHAIN_ACK_MASK));
    }

    if (ctx.arg->hasPrev != 0) {
        CCU_CHECK_RET(ccu::NotifyWait(
            ctx.arg->channels[ctx.arg->prevChannelIdx], POST_SYNC_NOTIFY_IDX, CHAIN_READY_MASK));
        // ACK means READY was consumed. Send it before forwarding to overlap adjacent Clos links.
        if (ctx.arg->chainIsLast == 0) {
            CCU_CHECK_RET(ccu::NotifyRecord(
                ctx.arg->channels[ctx.arg->prevChannelIdx], POST_SYNC_NOTIFY_IDX, CHAIN_ACK_MASK));
        }
    }

    if (ctx.arg->hasNext != 0) {
        ccu::LocalAddr src;
        src.addr = ctx.localBuffer;
        src.addr += ctx.localOffset;
        src.token = ctx.localToken;

        ccu::RemoteAddr dst;
        dst.addr = ctx.remoteBuffer[ctx.arg->nextChannelIdx];
        dst.addr += ctx.localOffset;
        dst.token = ctx.remoteToken[ctx.arg->nextChannelIdx];

        const uint16_t eventMask = static_cast<uint16_t>(1U << ctx.arg->nextChannelIdx);
        CCU_CHECK_RET(ccu::Write(ctx.arg->channels[ctx.arg->nextChannelIdx], dst, src, ctx.dataSize,
            ctx.event, eventMask));
        CCU_CHECK_RET(ccu::EventWait(ctx.event, eventMask));
        CCU_CHECK_RET(ccu::NotifyRecord(
            ctx.arg->channels[ctx.arg->nextChannelIdx], POST_SYNC_NOTIFY_IDX, CHAIN_READY_MASK));
    }

    // The final segment deliberately has no ACK; FINAL drains the chain without leaving a stale bit.
    if (ctx.arg->chainIsLast != 0) {
        if (ctx.arg->hasNext != 0) {
            CCU_CHECK_RET(ccu::NotifyWait(
                ctx.arg->channels[ctx.arg->nextChannelIdx], POST_SYNC_NOTIFY_IDX, CHAIN_FINAL_MASK));
        }
        if (ctx.arg->hasPrev != 0) {
            CCU_CHECK_RET(ccu::NotifyRecord(
                ctx.arg->channels[ctx.arg->prevChannelIdx], POST_SYNC_NOTIFY_IDX, CHAIN_FINAL_MASK));
        }
    }
    return CCU_SUCCESS;
}

static CcuResult WaitPairAcks(DirectBroadcastContext &ctx)
{
    if (ctx.arg->hasNext == 0) {
        return CCU_SUCCESS;
    }

    const ChannelHandle nextChannel = ctx.arg->channels[ctx.arg->nextChannelIdx];
    CCU_CHECK_RET(ccu::NotifyWait(nextChannel, POST_SYNC_NOTIFY_IDX, CHAIN_PAIR_ACK0_MASK));
    CCU_CHECK_RET(ccu::NotifyWait(nextChannel, POST_SYNC_NOTIFY_IDX, CHAIN_PAIR_ACK1_MASK));
    return CCU_SUCCESS;
}

static CcuResult WaitPairReadyAndAck(DirectBroadcastContext &ctx)
{
    if (ctx.arg->hasPrev == 0) {
        return CCU_SUCCESS;
    }

    const ChannelHandle prevChannel = ctx.arg->channels[ctx.arg->prevChannelIdx];
    CCU_CHECK_RET(ccu::NotifyWait(prevChannel, POST_SYNC_NOTIFY_IDX, CHAIN_PAIR_READY0_MASK));
    CCU_CHECK_RET(ccu::NotifyWait(prevChannel, POST_SYNC_NOTIFY_IDX, CHAIN_PAIR_READY1_MASK));
    // Both READY bits have now been consumed and reset.  ACK the two slots
    // before forwarding so the upstream rank can prepare the next pair while
    // this rank's two DMA writes are in flight.
    CCU_CHECK_RET(ccu::NotifyRecord(prevChannel, POST_SYNC_NOTIFY_IDX, CHAIN_PAIR_ACK0_MASK));
    CCU_CHECK_RET(ccu::NotifyRecord(prevChannel, POST_SYNC_NOTIFY_IDX, CHAIN_PAIR_ACK1_MASK));
    return CCU_SUCCESS;
}

static CcuResult WritePair(DirectBroadcastContext &ctx, const ccu::Variable &firstOffset,
    const ccu::Variable &firstSize, const ccu::Variable &secondOffset, const ccu::Variable &secondSize)
{
    if (ctx.arg->hasNext == 0) {
        return CCU_SUCCESS;
    }

    ccu::LocalAddr firstSrc;
    firstSrc.addr = ctx.localBuffer;
    firstSrc.addr += firstOffset;
    firstSrc.token = ctx.localToken;

    ccu::RemoteAddr firstDst;
    firstDst.addr = ctx.remoteBuffer[ctx.arg->nextChannelIdx];
    firstDst.addr += firstOffset;
    firstDst.token = ctx.remoteToken[ctx.arg->nextChannelIdx];

    ccu::LocalAddr secondSrc;
    secondSrc.addr = ctx.localBuffer;
    secondSrc.addr += secondOffset;
    secondSrc.token = ctx.localToken;

    ccu::RemoteAddr secondDst;
    secondDst.addr = ctx.remoteBuffer[ctx.arg->nextChannelIdx];
    secondDst.addr += secondOffset;
    secondDst.token = ctx.remoteToken[ctx.arg->nextChannelIdx];

    constexpr uint16_t firstEventMask = 1U;
    constexpr uint16_t secondEventMask = 1U << 1;
    constexpr uint16_t pairEventMask = firstEventMask | secondEventMask;
    const ChannelHandle nextChannel = ctx.arg->channels[ctx.arg->nextChannelIdx];

    // Submit both non-overlapping transfers before waiting.  This is the data
    // plane difference from v0.3C: the second Write is no longer blocked by an
    // EventWait for the first Write.
    CCU_CHECK_RET(ccu::Write(nextChannel, firstDst, firstSrc, firstSize, ctx.event, firstEventMask));
    CCU_CHECK_RET(ccu::Write(nextChannel, secondDst, secondSrc, secondSize, ctx.event, secondEventMask));
    CCU_CHECK_RET(ccu::EventWait(ctx.event, pairEventMask));
    CCU_CHECK_RET(ccu::NotifyRecord(nextChannel, POST_SYNC_NOTIFY_IDX, CHAIN_PAIR_READY0_MASK));
    CCU_CHECK_RET(ccu::NotifyRecord(nextChannel, POST_SYNC_NOTIFY_IDX, CHAIN_PAIR_READY1_MASK));
    return CCU_SUCCESS;
}

static CcuResult DrainFinalPair(DirectBroadcastContext &ctx)
{
    // Consume the final pair's ACK bits so no slot is left armed in a reusable
    // Engine Context.  FINAL then propagates completion from the chain tail
    // back to root, exactly like the proven single-segment protocol.
    CCU_CHECK_RET(WaitPairAcks(ctx));
    if (ctx.arg->hasNext != 0) {
        CCU_CHECK_RET(ccu::NotifyWait(
            ctx.arg->channels[ctx.arg->nextChannelIdx], POST_SYNC_NOTIFY_IDX, CHAIN_FINAL_MASK));
    }
    if (ctx.arg->hasPrev != 0) {
        CCU_CHECK_RET(ccu::NotifyRecord(
            ctx.arg->channels[ctx.arg->prevChannelIdx], POST_SYNC_NOTIFY_IDX, CHAIN_FINAL_MASK));
    }
    return CCU_SUCCESS;
}

template <bool PairIsFirst, bool PairIsLast>
static CcuResult ChainTransferPair(DirectBroadcastContext &ctx, const ccu::Variable &firstOffset,
    const ccu::Variable &firstSize, const ccu::Variable &secondOffset, const ccu::Variable &secondSize)
{
    if constexpr (!PairIsFirst) {
        // The previous pair's ACKs prove both READY slots were consumed before
        // the same two bits are reused by this pair.
        CCU_CHECK_RET(WaitPairAcks(ctx));
    }

    CCU_CHECK_RET(WaitPairReadyAndAck(ctx));
    CCU_CHECK_RET(WritePair(ctx, firstOffset, firstSize, secondOffset, secondSize));

    if constexpr (PairIsLast) {
        CCU_CHECK_RET(DrainFinalPair(ctx));
    }
    return CCU_SUCCESS;
}

static CcuResult ChainTransferTailAfterPair(DirectBroadcastContext &ctx)
{
    // The previous pair used both slots.  Consume both ACKs before reusing
    // slot0 for the odd final segment.
    CCU_CHECK_RET(WaitPairAcks(ctx));

    if (ctx.arg->hasPrev != 0) {
        const ChannelHandle prevChannel = ctx.arg->channels[ctx.arg->prevChannelIdx];
        CCU_CHECK_RET(ccu::NotifyWait(prevChannel, POST_SYNC_NOTIFY_IDX, CHAIN_PAIR_READY0_MASK));
        CCU_CHECK_RET(ccu::NotifyRecord(prevChannel, POST_SYNC_NOTIFY_IDX, CHAIN_PAIR_ACK0_MASK));
    }

    if (ctx.arg->hasNext != 0) {
        ccu::LocalAddr src;
        src.addr = ctx.localBuffer;
        src.addr += ctx.localOffset;
        src.token = ctx.localToken;

        ccu::RemoteAddr dst;
        dst.addr = ctx.remoteBuffer[ctx.arg->nextChannelIdx];
        dst.addr += ctx.localOffset;
        dst.token = ctx.remoteToken[ctx.arg->nextChannelIdx];

        constexpr uint16_t eventMask = 1U;
        const ChannelHandle nextChannel = ctx.arg->channels[ctx.arg->nextChannelIdx];
        CCU_CHECK_RET(ccu::Write(nextChannel, dst, src, ctx.dataSize, ctx.event, eventMask));
        CCU_CHECK_RET(ccu::EventWait(ctx.event, eventMask));
        CCU_CHECK_RET(ccu::NotifyRecord(nextChannel, POST_SYNC_NOTIFY_IDX, CHAIN_PAIR_READY0_MASK));
        CCU_CHECK_RET(ccu::NotifyWait(nextChannel, POST_SYNC_NOTIFY_IDX, CHAIN_PAIR_ACK0_MASK));
        CCU_CHECK_RET(ccu::NotifyWait(nextChannel, POST_SYNC_NOTIFY_IDX, CHAIN_FINAL_MASK));
    }

    if (ctx.arg->hasPrev != 0) {
        CCU_CHECK_RET(ccu::NotifyRecord(
            ctx.arg->channels[ctx.arg->prevChannelIdx], POST_SYNC_NOTIFY_IDX, CHAIN_FINAL_MASK));
    }
    return CCU_SUCCESS;
}

template <bool PairIsFirst, bool PairIsLast>
static CcuResult RunChainPairKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgBroadcast *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }
    if (kernelArg->channelCount == 0 ||
        (kernelArg->hasPrev == 0 && kernelArg->hasNext == 0) ||
        (kernelArg->hasPrev != 0 && kernelArg->prevChannelIdx >= kernelArg->channelCount) ||
        (kernelArg->hasNext != 0 && kernelArg->nextChannelIdx >= kernelArg->channelCount)) {
        return CcuResult::CCU_E_INTERNAL;
    }

    DirectBroadcastContext ctx;
    ctx.arg = kernelArg;
    CCU_CHECK_RET(InitResources(ctx));
    CCU_CHECK_RET(LoadChainPairTaskArgs(ctx));
    if (ctx.sliceOffsets.size() != 1 || ctx.sliceSizes.size() != 1) {
        return CcuResult::CCU_E_INTERNAL;
    }
    if constexpr (PairIsFirst) {
        CCU_CHECK_RET(PreSync(ctx));
    }
    CCU_CHECK_RET((ChainTransferPair<PairIsFirst, PairIsLast>(
        ctx, ctx.localOffset, ctx.dataSize, ctx.sliceOffsets[0], ctx.sliceSizes[0])));
    return CCU_SUCCESS;
}

static CcuResult RunChainPairTailKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgBroadcast *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }
    if (kernelArg->channelCount == 0 ||
        (kernelArg->hasPrev == 0 && kernelArg->hasNext == 0) ||
        (kernelArg->hasPrev != 0 && kernelArg->prevChannelIdx >= kernelArg->channelCount) ||
        (kernelArg->hasNext != 0 && kernelArg->nextChannelIdx >= kernelArg->channelCount)) {
        return CcuResult::CCU_E_INTERNAL;
    }

    DirectBroadcastContext ctx;
    ctx.arg = kernelArg;
    CCU_CHECK_RET(InitResources(ctx));
    CCU_CHECK_RET(LoadChainTaskArgs(ctx));
    CCU_CHECK_RET(ChainTransferTailAfterPair(ctx));
    return CCU_SUCCESS;
}

static bool IsValidUniqueReadyMask(uint32_t mask)
{
    return mask != 0 && mask <= 0xFFFFU && (mask & (mask - 1)) == 0;
}

static CcuResult WaitUniqueReady(DirectBroadcastContext &ctx)
{
    if (ctx.arg->hasPrev == 0) {
        return CCU_SUCCESS;
    }

    const uint16_t readyMask = static_cast<uint16_t>(ctx.arg->chainReadyMask);
    return static_cast<CcuResult>(ccu::NotifyWait(
        ctx.arg->channels[ctx.arg->prevChannelIdx], POST_SYNC_NOTIFY_IDX, readyMask));
}

static CcuResult WriteUniquePair(DirectBroadcastContext &ctx, const ccu::Variable &firstOffset,
    const ccu::Variable &firstSize, const ccu::Variable &secondOffset, const ccu::Variable &secondSize)
{
    if (ctx.arg->hasNext == 0) {
        return CCU_SUCCESS;
    }

    ccu::LocalAddr firstSrc;
    firstSrc.addr = ctx.localBuffer;
    firstSrc.addr += firstOffset;
    firstSrc.token = ctx.localToken;

    ccu::RemoteAddr firstDst;
    firstDst.addr = ctx.remoteBuffer[ctx.arg->nextChannelIdx];
    firstDst.addr += firstOffset;
    firstDst.token = ctx.remoteToken[ctx.arg->nextChannelIdx];

    ccu::LocalAddr secondSrc;
    secondSrc.addr = ctx.localBuffer;
    secondSrc.addr += secondOffset;
    secondSrc.token = ctx.localToken;

    ccu::RemoteAddr secondDst;
    secondDst.addr = ctx.remoteBuffer[ctx.arg->nextChannelIdx];
    secondDst.addr += secondOffset;
    secondDst.token = ctx.remoteToken[ctx.arg->nextChannelIdx];

    constexpr uint16_t firstEventMask = 1U;
    constexpr uint16_t secondEventMask = 1U << 1;
    constexpr uint16_t pairEventMask = firstEventMask | secondEventMask;
    const ChannelHandle nextChannel = ctx.arg->channels[ctx.arg->nextChannelIdx];
    const uint16_t readyMask = static_cast<uint16_t>(ctx.arg->chainReadyMask);

    CCU_CHECK_RET(ccu::Write(nextChannel, firstDst, firstSrc, firstSize, ctx.event, firstEventMask));
    CCU_CHECK_RET(ccu::Write(nextChannel, secondDst, secondSrc, secondSize, ctx.event, secondEventMask));
    CCU_CHECK_RET(ccu::EventWait(ctx.event, pairEventMask));
    CCU_CHECK_RET(ccu::NotifyRecord(nextChannel, POST_SYNC_NOTIFY_IDX, readyMask));
    return CCU_SUCCESS;
}

static CcuResult WriteUniqueTail(DirectBroadcastContext &ctx)
{
    if (ctx.arg->hasNext == 0) {
        return CCU_SUCCESS;
    }

    ccu::LocalAddr src;
    src.addr = ctx.localBuffer;
    src.addr += ctx.localOffset;
    src.token = ctx.localToken;

    ccu::RemoteAddr dst;
    dst.addr = ctx.remoteBuffer[ctx.arg->nextChannelIdx];
    dst.addr += ctx.localOffset;
    dst.token = ctx.remoteToken[ctx.arg->nextChannelIdx];

    constexpr uint16_t eventMask = 1U;
    const ChannelHandle nextChannel = ctx.arg->channels[ctx.arg->nextChannelIdx];
    const uint16_t readyMask = static_cast<uint16_t>(ctx.arg->chainReadyMask);
    CCU_CHECK_RET(ccu::Write(nextChannel, dst, src, ctx.dataSize, ctx.event, eventMask));
    CCU_CHECK_RET(ccu::EventWait(ctx.event, eventMask));
    CCU_CHECK_RET(ccu::NotifyRecord(nextChannel, POST_SYNC_NOTIFY_IDX, readyMask));
    return CCU_SUCCESS;
}

static CcuResult DrainUniqueChain(CcuKernelArgBroadcast *kernelArg)
{
    if (kernelArg->hasNext != 0) {
        CCU_CHECK_RET(ccu::NotifyWait(
            kernelArg->channels[kernelArg->nextChannelIdx], PRE_SYNC_NOTIFY_IDX, CHAIN_UNIQUE_FINAL_MASK));
    }
    if (kernelArg->hasPrev != 0) {
        CCU_CHECK_RET(ccu::NotifyRecord(
            kernelArg->channels[kernelArg->prevChannelIdx], PRE_SYNC_NOTIFY_IDX, CHAIN_UNIQUE_FINAL_MASK));
    }
    return CCU_SUCCESS;
}

static bool IsValidCutThroughReady(uint32_t notifyIdx, uint32_t mask)
{
    return (notifyIdx == 1 || notifyIdx == 2) && IsValidUniqueReadyMask(mask);
}

static CcuResult WaitCutThroughReady(
    DirectBroadcastContext &ctx, uint32_t notifyIdx, uint32_t readyMask)
{
    if (ctx.arg->hasPrev == 0) {
        return CCU_SUCCESS;
    }
    return static_cast<CcuResult>(ccu::NotifyWait(
        ctx.arg->channels[ctx.arg->prevChannelIdx], notifyIdx, static_cast<uint16_t>(readyMask)));
}

static CcuResult SubmitCutThroughWrite(DirectBroadcastContext &ctx, const ccu::Variable &offset,
    const ccu::Variable &size, uint16_t eventMask)
{
    if (ctx.arg->hasNext == 0) {
        return CCU_SUCCESS;
    }

    ccu::LocalAddr src;
    src.addr = ctx.localBuffer;
    src.addr += offset;
    src.token = ctx.localToken;

    ccu::RemoteAddr dst;
    dst.addr = ctx.remoteBuffer[ctx.arg->nextChannelIdx];
    dst.addr += offset;
    dst.token = ctx.remoteToken[ctx.arg->nextChannelIdx];

    return static_cast<CcuResult>(ccu::Write(ctx.arg->channels[ctx.arg->nextChannelIdx],
        dst, src, size, ctx.event, eventMask));
}

static CcuResult FinishCutThroughWrite(DirectBroadcastContext &ctx, uint32_t notifyIdx,
    uint32_t readyMask, uint16_t eventMask)
{
    if (ctx.arg->hasNext == 0) {
        return CCU_SUCCESS;
    }

    CCU_CHECK_RET(ccu::EventWait(ctx.event, eventMask));
    CCU_CHECK_RET(ccu::NotifyRecord(ctx.arg->channels[ctx.arg->nextChannelIdx], notifyIdx,
        static_cast<uint16_t>(readyMask)));
    return CCU_SUCCESS;
}

static CcuResult TransferCutThroughPair(DirectBroadcastContext &ctx,
    const ccu::Variable &firstOffset, const ccu::Variable &firstSize,
    const ccu::Variable &secondOffset, const ccu::Variable &secondSize)
{
    constexpr uint16_t firstEventMask = 1U;
    constexpr uint16_t secondEventMask = 1U << 1;

    // The second upstream READY may arrive while the first local DMA is in
    // flight.  Both writes are submitted before either EventWait, preserving
    // the proven two-DMA queue window while forwarding each 16 MiB segment as
    // soon as its own local write completes.
    CCU_CHECK_RET(WaitCutThroughReady(
        ctx, ctx.arg->cutReadyNotifyIdx0, ctx.arg->cutReadyMask0));
    CCU_CHECK_RET(SubmitCutThroughWrite(ctx, firstOffset, firstSize, firstEventMask));
    CCU_CHECK_RET(WaitCutThroughReady(
        ctx, ctx.arg->cutReadyNotifyIdx1, ctx.arg->cutReadyMask1));
    CCU_CHECK_RET(SubmitCutThroughWrite(ctx, secondOffset, secondSize, secondEventMask));
    CCU_CHECK_RET(FinishCutThroughWrite(
        ctx, ctx.arg->cutReadyNotifyIdx0, ctx.arg->cutReadyMask0, firstEventMask));
    CCU_CHECK_RET(FinishCutThroughWrite(
        ctx, ctx.arg->cutReadyNotifyIdx1, ctx.arg->cutReadyMask1, secondEventMask));
    return CCU_SUCCESS;
}

static CcuResult TransferCutThroughRollingQuad(DirectBroadcastContext &ctx,
    const ccu::Variable &offset0, const ccu::Variable &size0,
    const ccu::Variable &offset1, const ccu::Variable &size1,
    const ccu::Variable &offset2, const ccu::Variable &size2,
    const ccu::Variable &offset3, const ccu::Variable &size3)
{
    constexpr uint16_t event0 = 1U;
    constexpr uint16_t event1 = 1U << 1;

    // Fuse four segments into one static graph, but retain the proven two-DMA
    // capacity.  An event slot is consumed before reuse and every completed
    // 16 MiB cell is announced immediately instead of waiting for the whole
    // 64 MiB quad to arrive.
    CCU_CHECK_RET(WaitCutThroughReady(
        ctx, ctx.arg->cutReadyNotifyIdx0, ctx.arg->cutReadyMask0));
    CCU_CHECK_RET(SubmitCutThroughWrite(ctx, offset0, size0, event0));
    CCU_CHECK_RET(WaitCutThroughReady(
        ctx, ctx.arg->cutReadyNotifyIdx1, ctx.arg->cutReadyMask1));
    CCU_CHECK_RET(SubmitCutThroughWrite(ctx, offset1, size1, event1));
    CCU_CHECK_RET(FinishCutThroughWrite(
        ctx, ctx.arg->cutReadyNotifyIdx0, ctx.arg->cutReadyMask0, event0));

    CCU_CHECK_RET(WaitCutThroughReady(
        ctx, ctx.arg->cutReadyNotifyIdx2, ctx.arg->cutReadyMask2));
    CCU_CHECK_RET(SubmitCutThroughWrite(ctx, offset2, size2, event0));
    CCU_CHECK_RET(FinishCutThroughWrite(
        ctx, ctx.arg->cutReadyNotifyIdx1, ctx.arg->cutReadyMask1, event1));

    CCU_CHECK_RET(WaitCutThroughReady(
        ctx, ctx.arg->cutReadyNotifyIdx3, ctx.arg->cutReadyMask3));
    CCU_CHECK_RET(SubmitCutThroughWrite(ctx, offset3, size3, event1));

    CCU_CHECK_RET(FinishCutThroughWrite(
        ctx, ctx.arg->cutReadyNotifyIdx2, ctx.arg->cutReadyMask2, event0));
    CCU_CHECK_RET(FinishCutThroughWrite(
        ctx, ctx.arg->cutReadyNotifyIdx3, ctx.arg->cutReadyMask3, event1));
    return CCU_SUCCESS;
}

static CcuResult TransferFinalRollingFiveAndDrain(DirectBroadcastContext &ctx,
    const ccu::Variable &offset0, const ccu::Variable &size0,
    const ccu::Variable &offset1, const ccu::Variable &size1,
    const ccu::Variable &offset2, const ccu::Variable &size2,
    const ccu::Variable &offset3, const ccu::Variable &size3,
    const ccu::Variable &offset4, const ccu::Variable &size4)
{
    constexpr uint16_t event0 = 1U;
    constexpr uint16_t event1 = 1U << 1;

    CCU_CHECK_RET(WaitCutThroughReady(
        ctx, ctx.arg->cutReadyNotifyIdx0, ctx.arg->cutReadyMask0));
    CCU_CHECK_RET(SubmitCutThroughWrite(ctx, offset0, size0, event0));
    CCU_CHECK_RET(WaitCutThroughReady(
        ctx, ctx.arg->cutReadyNotifyIdx1, ctx.arg->cutReadyMask1));
    CCU_CHECK_RET(SubmitCutThroughWrite(ctx, offset1, size1, event1));
    CCU_CHECK_RET(FinishCutThroughWrite(
        ctx, ctx.arg->cutReadyNotifyIdx0, ctx.arg->cutReadyMask0, event0));

    CCU_CHECK_RET(WaitCutThroughReady(
        ctx, ctx.arg->cutReadyNotifyIdx2, ctx.arg->cutReadyMask2));
    CCU_CHECK_RET(SubmitCutThroughWrite(ctx, offset2, size2, event0));
    CCU_CHECK_RET(FinishCutThroughWrite(
        ctx, ctx.arg->cutReadyNotifyIdx1, ctx.arg->cutReadyMask1, event1));

    CCU_CHECK_RET(WaitCutThroughReady(
        ctx, ctx.arg->cutReadyNotifyIdx3, ctx.arg->cutReadyMask3));
    CCU_CHECK_RET(SubmitCutThroughWrite(ctx, offset3, size3, event1));
    CCU_CHECK_RET(FinishCutThroughWrite(
        ctx, ctx.arg->cutReadyNotifyIdx2, ctx.arg->cutReadyMask2, event0));

    CCU_CHECK_RET(WaitCutThroughReady(
        ctx, ctx.arg->cutReadyNotifyIdx4, ctx.arg->cutReadyMask4));
    CCU_CHECK_RET(SubmitCutThroughWrite(ctx, offset4, size4, event0));
    CCU_CHECK_RET(FinishCutThroughWrite(
        ctx, ctx.arg->cutReadyNotifyIdx3, ctx.arg->cutReadyMask3, event1));
    CCU_CHECK_RET(FinishCutThroughWrite(
        ctx, ctx.arg->cutReadyNotifyIdx4, ctx.arg->cutReadyMask4, event0));
    return DrainUniqueChain(ctx.arg);
}

template <uint32_t SegmentIdx>
static CcuResult StartPersistentRollingSegment(DirectBroadcastContext &ctx, uint16_t eventMask)
{
    static_assert(SegmentIdx < V13_PERSISTENT_SEGMENT_COUNT,
        "Persistent rolling segment is out of range");
    constexpr uint32_t notifyIdx = 1U + SegmentIdx / 16U;
    constexpr uint32_t readyMask = 1U << (SegmentIdx % 16U);

    ccu::Variable offset;
    ccu::Variable size;
    offset = static_cast<uint64_t>(SegmentIdx) * V13_PERSISTENT_SEGMENT_SIZE;
    if constexpr (SegmentIdx + 1U == V13_PERSISTENT_SEGMENT_COUNT) {
        size = V13_PERSISTENT_TAIL_SIZE;
    } else {
        size = V13_PERSISTENT_SEGMENT_SIZE;
    }

    CCU_CHECK_RET(WaitCutThroughReady(ctx, notifyIdx, readyMask));
    CCU_CHECK_RET(SubmitCutThroughWrite(ctx, offset, size, eventMask));
    return CCU_SUCCESS;
}

template <uint32_t SegmentIdx>
static CcuResult FinishPersistentRollingSegment(DirectBroadcastContext &ctx, uint16_t eventMask)
{
    static_assert(SegmentIdx < V13_PERSISTENT_SEGMENT_COUNT,
        "Persistent rolling completion is out of range");
    constexpr uint32_t notifyIdx = 1U + SegmentIdx / 16U;
    constexpr uint32_t readyMask = 1U << (SegmentIdx % 16U);
    return FinishCutThroughWrite(ctx, notifyIdx, readyMask, eventMask);
}

template <uint32_t SegmentIdx>
static CcuResult ContinuePersistentRolling2(DirectBroadcastContext &ctx)
{
    if constexpr (SegmentIdx < V13_PERSISTENT_SEGMENT_COUNT) {
        constexpr uint16_t eventMask = 1U << (SegmentIdx % 2U);
        CCU_CHECK_RET((FinishPersistentRollingSegment<SegmentIdx - 2U>(ctx, eventMask)));
        CCU_CHECK_RET((StartPersistentRollingSegment<SegmentIdx>(ctx, eventMask)));
        return ContinuePersistentRolling2<SegmentIdx + 1U>(ctx);
    }
    return CCU_SUCCESS;
}

static CcuResult RunPersistentRolling2_400(CcuKernelArg arg)
{
    static_assert(V13_PERSISTENT_SEGMENT_COUNT >= 2U,
        "Persistent Rolling-2 requires at least two segments");
    static_assert(V13_PERSISTENT_CHANNEL_NOTIFY_NUM >=
        1U + (V13_PERSISTENT_SEGMENT_COUNT + 15U) / 16U,
        "Persistent channel notify count is too small");

    auto *kernelArg = static_cast<CcuKernelArgBroadcast *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }
    if (kernelArg->channelCount == 0 ||
        (kernelArg->hasPrev == 0 && kernelArg->hasNext == 0) ||
        (kernelArg->hasPrev != 0 && kernelArg->prevChannelIdx >= kernelArg->channelCount) ||
        (kernelArg->hasNext != 0 && kernelArg->nextChannelIdx >= kernelArg->channelCount)) {
        return CcuResult::CCU_E_INTERNAL;
    }

    DirectBroadcastContext ctx;
    ctx.arg = kernelArg;
    CCU_CHECK_RET(InitResources(ctx));
    CCU_CHECK_RET(LoadPersistentTaskArgs(ctx));
    CCU_CHECK_RET(PreSync(ctx));

    constexpr uint16_t event0 = 1U << 0;
    constexpr uint16_t event1 = 1U << 1;
    CCU_CHECK_RET((StartPersistentRollingSegment<0>(ctx, event0)));
    CCU_CHECK_RET((StartPersistentRollingSegment<1>(ctx, event1)));
    CCU_CHECK_RET((ContinuePersistentRolling2<2>(ctx)));
    CCU_CHECK_RET((FinishPersistentRollingSegment<V13_PERSISTENT_SEGMENT_COUNT - 2U>(ctx, event0)));
    CCU_CHECK_RET((FinishPersistentRollingSegment<V13_PERSISTENT_SEGMENT_COUNT - 1U>(ctx, event1)));
    return DrainUniqueChain(ctx.arg);
}

template <uint32_t SegmentIdx>
static CcuResult StartPersistentRolling512Segment(
    DirectBroadcastContext &ctx, uint16_t eventMask)
{
    static_assert(SegmentIdx < V18A_PERSISTENT_512_SEGMENT_COUNT,
        "Persistent 512 MiB rolling segment is out of range");
    constexpr uint32_t notifyIdx = 1U + SegmentIdx / 16U;
    constexpr uint32_t readyMask = 1U << (SegmentIdx % 16U);

    ccu::Variable offset;
    ccu::Variable size;
    offset = static_cast<uint64_t>(SegmentIdx) * V18A_PERSISTENT_512_SEGMENT_SIZE;
    size = V18A_PERSISTENT_512_SEGMENT_SIZE;

    CCU_CHECK_RET(WaitCutThroughReady(ctx, notifyIdx, readyMask));
    CCU_CHECK_RET(SubmitCutThroughWrite(ctx, offset, size, eventMask));
    return CCU_SUCCESS;
}

template <uint32_t SegmentIdx>
static CcuResult FinishPersistentRolling512Segment(
    DirectBroadcastContext &ctx, uint16_t eventMask)
{
    static_assert(SegmentIdx < V18A_PERSISTENT_512_SEGMENT_COUNT,
        "Persistent 512 MiB rolling completion is out of range");
    constexpr uint32_t notifyIdx = 1U + SegmentIdx / 16U;
    constexpr uint32_t readyMask = 1U << (SegmentIdx % 16U);
    return FinishCutThroughWrite(ctx, notifyIdx, readyMask, eventMask);
}

template <uint32_t SegmentIdx>
static CcuResult ContinuePersistentRolling2_512(DirectBroadcastContext &ctx)
{
    if constexpr (SegmentIdx < V18A_PERSISTENT_512_SEGMENT_COUNT) {
        constexpr uint16_t eventMask = 1U << (SegmentIdx % 2U);
        CCU_CHECK_RET((FinishPersistentRolling512Segment<SegmentIdx - 2U>(ctx, eventMask)));
        CCU_CHECK_RET((StartPersistentRolling512Segment<SegmentIdx>(ctx, eventMask)));
        return ContinuePersistentRolling2_512<SegmentIdx + 1U>(ctx);
    }
    return CCU_SUCCESS;
}

static CcuResult RunPersistentRolling2_512(CcuKernelArg arg)
{
    static_assert(V18A_PERSISTENT_512_SEGMENT_COUNT >= 2U,
        "Persistent 512 MiB Rolling-2 requires at least two segments");
    static_assert(V18A_PERSISTENT_512_CHANNEL_NOTIFY_NUM >=
        1U + (V18A_PERSISTENT_512_SEGMENT_COUNT + 15U) / 16U,
        "Persistent 512 MiB channel notify count is too small");

    auto *kernelArg = static_cast<CcuKernelArgBroadcast *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }
    if (kernelArg->channelCount == 0 ||
        (kernelArg->hasPrev == 0 && kernelArg->hasNext == 0) ||
        (kernelArg->hasPrev != 0 && kernelArg->prevChannelIdx >= kernelArg->channelCount) ||
        (kernelArg->hasNext != 0 && kernelArg->nextChannelIdx >= kernelArg->channelCount)) {
        return CcuResult::CCU_E_INTERNAL;
    }

    DirectBroadcastContext ctx;
    ctx.arg = kernelArg;
    CCU_CHECK_RET(InitResources(ctx));
    CCU_CHECK_RET(LoadPersistentTaskArgs(ctx));
    CCU_CHECK_RET(PreSync(ctx));

    constexpr uint16_t event0 = 1U << 0;
    constexpr uint16_t event1 = 1U << 1;
    CCU_CHECK_RET((StartPersistentRolling512Segment<0>(ctx, event0)));
    CCU_CHECK_RET((StartPersistentRolling512Segment<1>(ctx, event1)));
    CCU_CHECK_RET((ContinuePersistentRolling2_512<2>(ctx)));
    CCU_CHECK_RET((FinishPersistentRolling512Segment<
        V18A_PERSISTENT_512_SEGMENT_COUNT - 2U>(ctx, event0)));
    CCU_CHECK_RET((FinishPersistentRolling512Segment<
        V18A_PERSISTENT_512_SEGMENT_COUNT - 1U>(ctx, event1)));
    return DrainUniqueChain(ctx.arg);
}

static CcuResult TransferCutThroughTail(DirectBroadcastContext &ctx)
{
    constexpr uint16_t eventMask = 1U;
    CCU_CHECK_RET(WaitCutThroughReady(
        ctx, ctx.arg->cutReadyNotifyIdx0, ctx.arg->cutReadyMask0));
    CCU_CHECK_RET(SubmitCutThroughWrite(ctx, ctx.localOffset, ctx.dataSize, eventMask));
    CCU_CHECK_RET(FinishCutThroughWrite(
        ctx, ctx.arg->cutReadyNotifyIdx0, ctx.arg->cutReadyMask0, eventMask));
    return CCU_SUCCESS;
}

CcuResult CcuDirectBroadcastKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgBroadcast *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }

    DirectBroadcastContext ctx;
    ctx.arg = kernelArg;

    CCU_CHECK_RET(InitResources(ctx));
    CCU_CHECK_RET(LoadTaskArgs(ctx));
    CCU_CHECK_RET(PreSync(ctx));
    CCU_CHECK_RET(BroadcastData(ctx));
    CCU_CHECK_RET(PhaseSync(ctx, SCATTER_SYNC_MASK));
    return CCU_SUCCESS;
}

CcuResult CcuAsymmetricDirectBroadcastKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgBroadcast *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }

    DirectBroadcastContext ctx;
    ctx.arg = kernelArg;

    CCU_CHECK_RET(InitResources(ctx));
    CCU_CHECK_RET(LoadTaskArgs(ctx));
    CCU_CHECK_RET(AsymmetricDirectPreSync(ctx));
    CCU_CHECK_RET(BroadcastData(ctx));
    CCU_CHECK_RET(AsymmetricDirectComplete(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuAsymmetricDirect512K2ArgBroadcastKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgBroadcast *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }

    DirectBroadcastContext ctx;
    ctx.arg = kernelArg;

    CCU_CHECK_RET(InitResources(ctx));
    CCU_CHECK_RET(LoadAsymmetricDirect512K2Args(ctx));
    CCU_CHECK_RET(AsymmetricDirectPreSync(ctx));
    CCU_CHECK_RET(BroadcastData(ctx));
    CCU_CHECK_RET(AsymmetricDirectComplete(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuAsymmetricDirect512KRegisteredBroadcastKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgBroadcast *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }

    DirectBroadcastContext ctx;
    ctx.arg = kernelArg;

    CCU_CHECK_RET(InitResources(ctx));
    CCU_CHECK_RET(LoadAsymmetricDirect512KRegistered(ctx));
    CCU_CHECK_RET(AsymmetricDirectPreSync(ctx));
    CCU_CHECK_RET(BroadcastData(ctx));
    CCU_CHECK_RET(AsymmetricDirectComplete(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuAsymmetricDirect512KRegisteredNonRootBroadcastKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgBroadcast *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }
    if (kernelArg->rankId == kernelArg->rootId || kernelArg->channelCount != 1) {
        return CCU_E_INTERNAL;
    }

    DirectBroadcastContext ctx;
    ctx.arg = kernelArg;
    CCU_CHECK_RET(LoadAsymmetricDirect512KRegisteredNonRoot(ctx));
    CCU_CHECK_RET(AsymmetricDirectPreSync(ctx));
    CCU_CHECK_RET(AsymmetricDirectComplete(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuAsymmetricPull512K2ArgBroadcastKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgBroadcast *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount == 0 ||
        kernelArg->channelCount > MAX_RANK_SIZE - 1) {
        return CcuResult::CCU_E_INTERNAL;
    }

    DirectBroadcastContext ctx;
    ctx.arg = kernelArg;
    CCU_CHECK_RET(InitResources(ctx));
    CCU_CHECK_RET(LoadAsymmetricDirect512K2Args(ctx));
    CCU_CHECK_RET(AsymmetricPull512K(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuScatterBroadcastKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgBroadcast *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }

    DirectBroadcastContext ctx;
    ctx.arg = kernelArg;

    CCU_CHECK_RET(InitResources(ctx));
    CCU_CHECK_RET(LoadScatterTaskArgs(ctx));
    CCU_CHECK_RET(PreSync(ctx));
    CCU_CHECK_RET(ScatterData(ctx));
    CCU_CHECK_RET(PhaseSync(ctx, SCATTER_SYNC_MASK));
    return CCU_SUCCESS;
}

CcuResult CcuAllGatherBroadcastKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgBroadcast *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }

    DirectBroadcastContext ctx;
    ctx.arg = kernelArg;

    CCU_CHECK_RET(InitResources(ctx));
    CCU_CHECK_RET(LoadAllGatherTaskArgs(ctx));
    CCU_CHECK_RET(PreSync(ctx));
    CCU_CHECK_RET(AllGatherData(ctx));
    CCU_CHECK_RET(PhaseSync(ctx, ALLGATHER_SYNC_MASK));
    return CCU_SUCCESS;
}

CcuResult CcuLeanScatterBroadcastKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgBroadcast *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }

    DirectBroadcastContext ctx;
    ctx.arg = kernelArg;
    CCU_CHECK_RET(InitResources(ctx));
    CCU_CHECK_RET(LoadScatterTaskArgs(ctx));
    CCU_CHECK_RET(PreSync(ctx));
    CCU_CHECK_RET(ScatterData(ctx));
    CCU_CHECK_RET(LeanScatterReady(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuLeanAllGatherBroadcastKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgBroadcast *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }

    DirectBroadcastContext ctx;
    ctx.arg = kernelArg;
    CCU_CHECK_RET(InitResources(ctx));
    CCU_CHECK_RET(LoadAllGatherTaskArgs(ctx));
    CCU_CHECK_RET(AllGatherData(ctx));
    CCU_CHECK_RET(PhaseSync(ctx, ALLGATHER_SYNC_MASK));
    return CCU_SUCCESS;
}

CcuResult CcuPipelinedChainBroadcastKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgBroadcast *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }
    if (kernelArg->channelCount == 0 ||
        (kernelArg->hasPrev == 0 && kernelArg->hasNext == 0) ||
        (kernelArg->hasPrev != 0 && kernelArg->prevChannelIdx >= kernelArg->channelCount) ||
        (kernelArg->hasNext != 0 && kernelArg->nextChannelIdx >= kernelArg->channelCount)) {
        return CcuResult::CCU_E_INTERNAL;
    }

    DirectBroadcastContext ctx;
    ctx.arg = kernelArg;

    CCU_CHECK_RET(InitResources(ctx));
    CCU_CHECK_RET(LoadChainTaskArgs(ctx));
    if (ctx.arg->chainIsFirst != 0) {
        CCU_CHECK_RET(PreSync(ctx));
    }
    CCU_CHECK_RET(ChainTransfer(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuPipelinedChainPairFirstBroadcastKernel(CcuKernelArg arg)
{
    return RunChainPairKernel<true, false>(arg);
}

CcuResult CcuPipelinedChainPairMiddleBroadcastKernel(CcuKernelArg arg)
{
    return RunChainPairKernel<false, false>(arg);
}

CcuResult CcuPipelinedChainPairLastBroadcastKernel(CcuKernelArg arg)
{
    return RunChainPairKernel<false, true>(arg);
}

CcuResult CcuPipelinedChainPairTailBroadcastKernel(CcuKernelArg arg)
{
    return RunChainPairTailKernel(arg);
}

CcuResult CcuUniquePairBroadcastKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgBroadcast *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }
    if (kernelArg->channelCount == 0 || !IsValidUniqueReadyMask(kernelArg->chainReadyMask) ||
        (kernelArg->hasPrev == 0 && kernelArg->hasNext == 0) ||
        (kernelArg->hasPrev != 0 && kernelArg->prevChannelIdx >= kernelArg->channelCount) ||
        (kernelArg->hasNext != 0 && kernelArg->nextChannelIdx >= kernelArg->channelCount)) {
        return CcuResult::CCU_E_INTERNAL;
    }

    DirectBroadcastContext ctx;
    ctx.arg = kernelArg;
    CCU_CHECK_RET(InitResources(ctx));
    CCU_CHECK_RET(LoadChainPairTaskArgs(ctx));
    if (ctx.sliceOffsets.size() != 1 || ctx.sliceSizes.size() != 1) {
        return CcuResult::CCU_E_INTERNAL;
    }
    if (kernelArg->chainIsFirst != 0) {
        CCU_CHECK_RET(PreSync(ctx));
    }
    CCU_CHECK_RET(WaitUniqueReady(ctx));
    CCU_CHECK_RET(WriteUniquePair(
        ctx, ctx.localOffset, ctx.dataSize, ctx.sliceOffsets[0], ctx.sliceSizes[0]));
    return CCU_SUCCESS;
}

CcuResult CcuUniqueTailBroadcastKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgBroadcast *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }
    if (kernelArg->channelCount == 0 || !IsValidUniqueReadyMask(kernelArg->chainReadyMask) ||
        (kernelArg->hasPrev == 0 && kernelArg->hasNext == 0) ||
        (kernelArg->hasPrev != 0 && kernelArg->prevChannelIdx >= kernelArg->channelCount) ||
        (kernelArg->hasNext != 0 && kernelArg->nextChannelIdx >= kernelArg->channelCount)) {
        return CcuResult::CCU_E_INTERNAL;
    }

    DirectBroadcastContext ctx;
    ctx.arg = kernelArg;
    CCU_CHECK_RET(InitResources(ctx));
    CCU_CHECK_RET(LoadChainTaskArgs(ctx));
    CCU_CHECK_RET(WaitUniqueReady(ctx));
    CCU_CHECK_RET(WriteUniqueTail(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuCutThroughPairBroadcastKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgBroadcast *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }
    if (kernelArg->channelCount == 0 ||
        (kernelArg->hasPrev == 0 && kernelArg->hasNext == 0) ||
        (kernelArg->hasPrev != 0 && kernelArg->prevChannelIdx >= kernelArg->channelCount) ||
        (kernelArg->hasNext != 0 && kernelArg->nextChannelIdx >= kernelArg->channelCount) ||
        !IsValidCutThroughReady(kernelArg->cutReadyNotifyIdx0, kernelArg->cutReadyMask0) ||
        !IsValidCutThroughReady(kernelArg->cutReadyNotifyIdx1, kernelArg->cutReadyMask1) ||
        (kernelArg->cutReadyNotifyIdx0 == kernelArg->cutReadyNotifyIdx1 &&
            kernelArg->cutReadyMask0 == kernelArg->cutReadyMask1)) {
        return CcuResult::CCU_E_INTERNAL;
    }

    DirectBroadcastContext ctx;
    ctx.arg = kernelArg;
    CCU_CHECK_RET(InitResources(ctx));
    CCU_CHECK_RET(LoadChainPairTaskArgs(ctx));
    if (ctx.sliceOffsets.size() != 1 || ctx.sliceSizes.size() != 1) {
        return CcuResult::CCU_E_INTERNAL;
    }
    if (kernelArg->chainIsFirst != 0) {
        CCU_CHECK_RET(PreSync(ctx));
    }
    CCU_CHECK_RET(TransferCutThroughPair(
        ctx, ctx.localOffset, ctx.dataSize, ctx.sliceOffsets[0], ctx.sliceSizes[0]));
    return CCU_SUCCESS;
}

CcuResult CcuQuad4Rolling2BroadcastKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgBroadcast *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }
    const bool duplicateReady =
        (kernelArg->cutReadyNotifyIdx0 == kernelArg->cutReadyNotifyIdx1 &&
            kernelArg->cutReadyMask0 == kernelArg->cutReadyMask1) ||
        (kernelArg->cutReadyNotifyIdx0 == kernelArg->cutReadyNotifyIdx2 &&
            kernelArg->cutReadyMask0 == kernelArg->cutReadyMask2) ||
        (kernelArg->cutReadyNotifyIdx0 == kernelArg->cutReadyNotifyIdx3 &&
            kernelArg->cutReadyMask0 == kernelArg->cutReadyMask3) ||
        (kernelArg->cutReadyNotifyIdx1 == kernelArg->cutReadyNotifyIdx2 &&
            kernelArg->cutReadyMask1 == kernelArg->cutReadyMask2) ||
        (kernelArg->cutReadyNotifyIdx1 == kernelArg->cutReadyNotifyIdx3 &&
            kernelArg->cutReadyMask1 == kernelArg->cutReadyMask3) ||
        (kernelArg->cutReadyNotifyIdx2 == kernelArg->cutReadyNotifyIdx3 &&
            kernelArg->cutReadyMask2 == kernelArg->cutReadyMask3);
    if (kernelArg->channelCount == 0 ||
        (kernelArg->hasPrev == 0 && kernelArg->hasNext == 0) ||
        (kernelArg->hasPrev != 0 && kernelArg->prevChannelIdx >= kernelArg->channelCount) ||
        (kernelArg->hasNext != 0 && kernelArg->nextChannelIdx >= kernelArg->channelCount) ||
        !IsValidCutThroughReady(kernelArg->cutReadyNotifyIdx0, kernelArg->cutReadyMask0) ||
        !IsValidCutThroughReady(kernelArg->cutReadyNotifyIdx1, kernelArg->cutReadyMask1) ||
        !IsValidCutThroughReady(kernelArg->cutReadyNotifyIdx2, kernelArg->cutReadyMask2) ||
        !IsValidCutThroughReady(kernelArg->cutReadyNotifyIdx3, kernelArg->cutReadyMask3) ||
        duplicateReady) {
        return CcuResult::CCU_E_INTERNAL;
    }

    DirectBroadcastContext ctx;
    ctx.arg = kernelArg;
    CCU_CHECK_RET(InitResources(ctx));
    CCU_CHECK_RET(LoadChainQuadTaskArgs(ctx));
    if (ctx.sliceOffsets.size() != QUAD4_SEGMENTS_PER_KERNEL - 1 ||
        ctx.sliceSizes.size() != QUAD4_SEGMENTS_PER_KERNEL - 1) {
        return CcuResult::CCU_E_INTERNAL;
    }
    if (kernelArg->chainIsFirst != 0) {
        CCU_CHECK_RET(PreSync(ctx));
    }
    CCU_CHECK_RET(TransferCutThroughRollingQuad(ctx,
        ctx.localOffset, ctx.dataSize,
        ctx.sliceOffsets[0], ctx.sliceSizes[0],
        ctx.sliceOffsets[1], ctx.sliceSizes[1],
        ctx.sliceOffsets[2], ctx.sliceSizes[2]));
    return CCU_SUCCESS;
}

CcuResult CcuFinalRolling5DrainBroadcastKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgBroadcast *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }
    const bool duplicateReady =
        (kernelArg->cutReadyNotifyIdx0 == kernelArg->cutReadyNotifyIdx1 &&
            kernelArg->cutReadyMask0 == kernelArg->cutReadyMask1) ||
        (kernelArg->cutReadyNotifyIdx0 == kernelArg->cutReadyNotifyIdx2 &&
            kernelArg->cutReadyMask0 == kernelArg->cutReadyMask2) ||
        (kernelArg->cutReadyNotifyIdx0 == kernelArg->cutReadyNotifyIdx3 &&
            kernelArg->cutReadyMask0 == kernelArg->cutReadyMask3) ||
        (kernelArg->cutReadyNotifyIdx0 == kernelArg->cutReadyNotifyIdx4 &&
            kernelArg->cutReadyMask0 == kernelArg->cutReadyMask4) ||
        (kernelArg->cutReadyNotifyIdx1 == kernelArg->cutReadyNotifyIdx2 &&
            kernelArg->cutReadyMask1 == kernelArg->cutReadyMask2) ||
        (kernelArg->cutReadyNotifyIdx1 == kernelArg->cutReadyNotifyIdx3 &&
            kernelArg->cutReadyMask1 == kernelArg->cutReadyMask3) ||
        (kernelArg->cutReadyNotifyIdx1 == kernelArg->cutReadyNotifyIdx4 &&
            kernelArg->cutReadyMask1 == kernelArg->cutReadyMask4) ||
        (kernelArg->cutReadyNotifyIdx2 == kernelArg->cutReadyNotifyIdx3 &&
            kernelArg->cutReadyMask2 == kernelArg->cutReadyMask3) ||
        (kernelArg->cutReadyNotifyIdx2 == kernelArg->cutReadyNotifyIdx4 &&
            kernelArg->cutReadyMask2 == kernelArg->cutReadyMask4) ||
        (kernelArg->cutReadyNotifyIdx3 == kernelArg->cutReadyNotifyIdx4 &&
            kernelArg->cutReadyMask3 == kernelArg->cutReadyMask4);
    if (kernelArg->channelCount == 0 || kernelArg->chainIsLast == 0 ||
        (kernelArg->hasPrev == 0 && kernelArg->hasNext == 0) ||
        (kernelArg->hasPrev != 0 && kernelArg->prevChannelIdx >= kernelArg->channelCount) ||
        (kernelArg->hasNext != 0 && kernelArg->nextChannelIdx >= kernelArg->channelCount) ||
        !IsValidCutThroughReady(kernelArg->cutReadyNotifyIdx0, kernelArg->cutReadyMask0) ||
        !IsValidCutThroughReady(kernelArg->cutReadyNotifyIdx1, kernelArg->cutReadyMask1) ||
        !IsValidCutThroughReady(kernelArg->cutReadyNotifyIdx2, kernelArg->cutReadyMask2) ||
        !IsValidCutThroughReady(kernelArg->cutReadyNotifyIdx3, kernelArg->cutReadyMask3) ||
        !IsValidCutThroughReady(kernelArg->cutReadyNotifyIdx4, kernelArg->cutReadyMask4) ||
        duplicateReady) {
        return CcuResult::CCU_E_INTERNAL;
    }

    DirectBroadcastContext ctx;
    ctx.arg = kernelArg;
    CCU_CHECK_RET(InitResources(ctx));
    CCU_CHECK_RET(LoadChainFiveTaskArgs(ctx));
    if (ctx.sliceOffsets.size() != FINAL5_SEGMENTS_PER_KERNEL - 1 ||
        ctx.sliceSizes.size() != FINAL5_SEGMENTS_PER_KERNEL - 1) {
        return CcuResult::CCU_E_INTERNAL;
    }
    CCU_CHECK_RET(TransferFinalRollingFiveAndDrain(ctx,
        ctx.localOffset, ctx.dataSize,
        ctx.sliceOffsets[0], ctx.sliceSizes[0],
        ctx.sliceOffsets[1], ctx.sliceSizes[1],
        ctx.sliceOffsets[2], ctx.sliceSizes[2],
        ctx.sliceOffsets[3], ctx.sliceSizes[3]));
    return CCU_SUCCESS;
}

CcuResult CcuPersistentRolling2_400BroadcastKernel(CcuKernelArg arg)
{
    return RunPersistentRolling2_400(arg);
}

CcuResult CcuPersistentRolling2_512BroadcastKernel(CcuKernelArg arg)
{
    return RunPersistentRolling2_512(arg);
}

CcuResult CcuCutThroughTailBroadcastKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgBroadcast *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }
    if (kernelArg->channelCount == 0 ||
        (kernelArg->hasPrev == 0 && kernelArg->hasNext == 0) ||
        (kernelArg->hasPrev != 0 && kernelArg->prevChannelIdx >= kernelArg->channelCount) ||
        (kernelArg->hasNext != 0 && kernelArg->nextChannelIdx >= kernelArg->channelCount) ||
        !IsValidCutThroughReady(kernelArg->cutReadyNotifyIdx0, kernelArg->cutReadyMask0)) {
        return CcuResult::CCU_E_INTERNAL;
    }

    DirectBroadcastContext ctx;
    ctx.arg = kernelArg;
    CCU_CHECK_RET(InitResources(ctx));
    CCU_CHECK_RET(LoadChainTaskArgs(ctx));
    CCU_CHECK_RET(TransferCutThroughTail(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuUniqueDrainBroadcastKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgBroadcast *>(arg);
    if (kernelArg == nullptr) {
        return CCU_E_PTR;
    }
    if (kernelArg->channelCount == 0 ||
        (kernelArg->hasPrev == 0 && kernelArg->hasNext == 0) ||
        (kernelArg->hasPrev != 0 && kernelArg->prevChannelIdx >= kernelArg->channelCount) ||
        (kernelArg->hasNext != 0 && kernelArg->nextChannelIdx >= kernelArg->channelCount)) {
        return CcuResult::CCU_E_INTERNAL;
    }

    // Keep a conventional four-argument launch shape.  This avoids relying on
    // undocumented zero-argument launch behavior; the loaded values are not
    // used by the reverse completion drain.
    DirectBroadcastContext ctx;
    ctx.arg = kernelArg;
    CCU_CHECK_RET(LoadChainTaskArgs(ctx));
    CCU_CHECK_RET(DrainUniqueChain(kernelArg));
    return CCU_SUCCESS;
}

#undef CCU_CHECK_RET

} // namespace ops_hccl
