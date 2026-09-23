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

constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0;
constexpr int POST_SYNC_ID = 3;
constexpr uint16_t ROOT_COPY_MASK = 0x8000U;

#define CCU_RETURN_IF_ERROR(expr)        \
    do {                                 \
        CcuResult _ret = (expr);         \
        if (_ret != CCU_SUCCESS) {       \
            return _ret;                 \
        }                                \
    } while (0)

void AddRankOffset(ccu::LocalAddr &addr, const ccu::Variable &sliceStride, uint32_t rank)
{
    for (uint32_t i = 0; i < rank; ++i) {
        addr.addr += sliceStride;
    }
}

CcuResult InitRemoteResources(ScatterContext &ctx)
{
    // V30.5 resource compaction: slot 0 is the local runtime address/token;
    // slot (channelIdx + 1) is the remote XN pair for that channel.  The old
    // rankSize-sized vectors materialized 24 XN variables for every 12-rank
    // single-channel relay kernel even though only one peer was referenced.
    ctx.output.resize(static_cast<size_t>(ctx.arg->channelCount) + 1U);
    ctx.token.resize(static_cast<size_t>(ctx.arg->channelCount) + 1U);

    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        const uint32_t peer = ctx.arg->peerRanks[channelIdx];
        if (peer >= ctx.arg->rankSize || peer == ctx.arg->rankId) {
            return CcuResult::CCU_E_PARA;
        }
        const size_t slot = static_cast<size_t>(channelIdx) + 1U;
        ctx.output[slot] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], OUTPUT_XN_ID);
        ctx.token[slot] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], TOKEN_XN_ID);
    }
    return CCU_SUCCESS;
}

CcuResult LoadDirectArgs(ScatterContext &ctx)
{
    uint32_t argId = 0;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.input, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.output[0], argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.inputToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.token[0], argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.sliceStride, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.windowOffset, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.sliceSize, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.rootCopyEnabled, argId++));
    return CCU_SUCCESS;
}

CcuResult LoadRelayArgs(ScatterContext &ctx)
{
    uint32_t argId = 0;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.input, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.output[0], argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.relayBuffer, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.inputToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.token[0], argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.relayBufferToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.sliceStride, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.directBytes, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.relayBytes, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.relayOffset, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.relayChunkBytes, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.scratchOffset, argId++));
    // V18: rank stride, window displacement and window length are distinct.
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.windowOffset, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.sliceSize, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.rootCopyEnabled, argId++));
    return CCU_SUCCESS;
}

// V21 safe 2x8 path: exact V17.4 symmetric address/token and completion
// handshake. It is intentionally kept separate from the optimized directed
// protocol so rankSize=16 can recover the empirically stable behavior.
CcuResult LegacyPreSyncMemory(ScatterContext &ctx, const ccu::Variable &localAddr, const ccu::Variable &localToken)
{
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
            ctx.arg->channels[i], localAddr, OUTPUT_XN_ID, CKE_IDX_0, 1U << OUTPUT_XN_ID));
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
            ctx.arg->channels[i], localToken, TOKEN_XN_ID, CKE_IDX_0, 1U << TOKEN_XN_ID));
    }
    const uint32_t allBit = (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, allBit));
    }
    return CCU_SUCCESS;
}

CcuResult LegacyPostSync(ScatterContext &ctx)
{
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        CCU_RETURN_IF_ERROR(ccu::NotifyRecord(ctx.arg->channels[i], CKE_IDX_0, 1U << POST_SYNC_ID));
    }
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, 1U << POST_SYNC_ID));
    }
    return CCU_SUCCESS;
}

CcuResult LegacyDoDirectScatter(ScatterContext &ctx)
{
    if (ctx.arg->rankId != ctx.arg->rootId) { return CCU_SUCCESS; }
    uint16_t totalMask = 0;
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        const uint32_t peer = ctx.arg->peerRanks[channelIdx];
        ccu::LocalAddr src;
        src.addr = ctx.input;
        AddRankOffset(src, ctx.sliceStride, peer);
        src.addr += ctx.windowOffset;
        src.token = ctx.inputToken;
        ccu::RemoteAddr dst;
        dst.addr = ctx.output[static_cast<size_t>(channelIdx) + 1U];
        dst.addr += ctx.windowOffset;
        dst.token = ctx.token[static_cast<size_t>(channelIdx) + 1U];
        const uint16_t mask = static_cast<uint16_t>(1U << channelIdx);
        totalMask = static_cast<uint16_t>(totalMask | mask);
        CCU_RETURN_IF_ERROR(ccu::Write(ctx.arg->channels[channelIdx], dst, src, ctx.sliceSize, ctx.event, mask));
    }
    if (totalMask != 0U) {
        CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.event, totalMask));
    }
    return CCU_SUCCESS;
}

// Directed ready/done handshake for a push. Receiver publishes address/token,
// sender consumes READY, writes, waits for completion, and sends DONE. Receiver
// consumes DONE before publishing the next READY. The next transfer is thus
// also an implicit acknowledgement; no redundant reverse traffic is needed.
CcuResult PreSyncMemory(ScatterContext &ctx, const ccu::Variable &localAddr,
    const ccu::Variable &localToken, bool sender)
{
    if (!sender) {
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
                ctx.arg->channels[i], localAddr, OUTPUT_XN_ID, CKE_IDX_0, 1U << OUTPUT_XN_ID));
            CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(
                ctx.arg->channels[i], localToken, TOKEN_XN_ID, CKE_IDX_0, 1U << TOKEN_XN_ID));
        }
    } else {
        const uint32_t allBit = (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            CCU_RETURN_IF_ERROR(ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, allBit));
        }
    }
    return CCU_SUCCESS;
}

CcuResult PreSync(ScatterContext &ctx, bool sender)
{
    return PreSyncMemory(ctx, ctx.output[0], ctx.token[0], sender);
}

CcuResult PreSyncRelayScratch(ScatterContext &ctx)
{
    return PreSyncMemory(ctx, ctx.relayBuffer, ctx.relayBufferToken,
        ctx.arg->localRole == SCATTER_ROLE_ROOT);
}

CcuResult PostSync(ScatterContext &ctx, bool sender)
{
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        if (sender) {
            CCU_RETURN_IF_ERROR(ccu::NotifyRecord(ctx.arg->channels[i], CKE_IDX_0, 1U << POST_SYNC_ID));
        } else {
            CCU_RETURN_IF_ERROR(ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, 1U << POST_SYNC_ID));
        }
    }
    return CCU_SUCCESS;
}

CcuResult StartRootCopy(ScatterContext &ctx, const ccu::Event &event, uint16_t &mask)
{
    if (ctx.arg->copyRoot == 0U) { return CCU_SUCCESS; }
    ccu::LocalAddr src, dst;
    src.addr = ctx.input;
    AddRankOffset(src, ctx.sliceStride, ctx.arg->rootId);
    src.addr += ctx.windowOffset;
    src.token = ctx.inputToken;
    dst.addr = ctx.output[0];
    dst.addr += ctx.windowOffset;
    dst.token = ctx.token[0];
    CCU_IF(ctx.rootCopyEnabled != 0)
    {
        CCU_RETURN_IF_ERROR(ccu::LocalCopy(dst, src, ctx.sliceSize, event, ROOT_COPY_MASK));
    }
    CCU_ELSE
    {
        CCU_RETURN_IF_ERROR(ccu::EventRecord(event, ROOT_COPY_MASK));
    }
    mask = static_cast<uint16_t>(mask | ROOT_COPY_MASK);
    return CCU_SUCCESS;
}

CcuResult WaitEventIfNeeded(ScatterContext &ctx, uint16_t mask)
{
    if (mask != 0) {
        CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.event, mask));
    }
    return CCU_SUCCESS;
}

CcuResult DoDirectScatter(ScatterContext &ctx, const ccu::Event &event)
{
    if (ctx.arg->rankId != ctx.arg->rootId) {
        return CCU_SUCCESS;
    }

    uint16_t totalMask = 0;
    if (ctx.arg->copyRootLate == 0U) {
        CCU_RETURN_IF_ERROR(StartRootCopy(ctx, event, totalMask));
    }
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        const uint32_t peer = ctx.arg->peerRanks[channelIdx];

        ccu::LocalAddr src;
        src.addr = ctx.input;
        AddRankOffset(src, ctx.sliceStride, peer);
        src.addr += ctx.windowOffset;
        src.token = ctx.inputToken;

        ccu::RemoteAddr dst;
        dst.addr = ctx.output[static_cast<size_t>(channelIdx) + 1U];
        dst.addr += ctx.windowOffset;
        dst.token = ctx.token[static_cast<size_t>(channelIdx) + 1U];

        const uint16_t mask = static_cast<uint16_t>(1U << channelIdx);
        totalMask = static_cast<uint16_t>(totalMask | mask);
        CCU_RETURN_IF_ERROR(ccu::Write(ctx.arg->channels[channelIdx], dst, src, ctx.sliceSize, event, mask));
    }
    if (ctx.arg->copyRootLate != 0U) {
        CCU_RETURN_IF_ERROR(StartRootCopy(ctx, event, totalMask));
    }
    if (totalMask != 0) {
        CCU_RETURN_IF_ERROR(ccu::EventWait(event, totalMask));
    }
    return CCU_SUCCESS;
}

/* Phase A: root stages the relay tail in helper HcclBuffer scratch. */
CcuResult DoRelayLoad(ScatterContext &ctx)
{
    if (ctx.arg->localRole != SCATTER_ROLE_ROOT) {
        return CCU_SUCCESS;
    }
    if (ctx.arg->channelCount != 1U ||
        ctx.arg->pairedRanks[0] >= ctx.arg->rankSize ||
        ctx.arg->pairedRanks[1] >= ctx.arg->rankSize ||
        ctx.arg->pairedRanks[0] == ctx.arg->rankId ||
        ctx.arg->pairedRanks[1] == ctx.arg->rankId) {
        return CcuResult::CCU_E_PARA;
    }

    ccu::RemoteAddr dst;
    dst.addr = ctx.output[1];
    dst.addr += ctx.scratchOffset;
    dst.token = ctx.token[1];

    // relayBytes is a V30.6 mode only for this fused root-helper kernel:
    // mode 0/1 chooses which target slice is staged in the helper scratch.
    CCU_IF(ctx.relayBytes == 0)
    {
        ccu::LocalAddr src;
        src.addr = ctx.input;
        AddRankOffset(src, ctx.sliceStride, ctx.arg->pairedRanks[0]);
        src.addr += ctx.windowOffset;
        src.addr += ctx.directBytes;
        src.addr += ctx.relayOffset;
        src.token = ctx.inputToken;
        CCU_RETURN_IF_ERROR(ccu::Write(ctx.arg->channels[0], dst, src,
            ctx.relayChunkBytes, ctx.event, 1U));
    }
    CCU_ELSE
    {
        ccu::LocalAddr src;
        src.addr = ctx.input;
        AddRankOffset(src, ctx.sliceStride, ctx.arg->pairedRanks[1]);
        src.addr += ctx.windowOffset;
        src.addr += ctx.directBytes;
        src.addr += ctx.relayOffset;
        src.token = ctx.inputToken;
        CCU_RETURN_IF_ERROR(ccu::Write(ctx.arg->channels[0], dst, src,
            ctx.relayChunkBytes, ctx.event, 1U));
    }
    CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.event, 1U));
    return CCU_SUCCESS;
}

/* Phase B: forward one chunk into the target's CURRENT window tail. */
CcuResult DoRelayForward(ScatterContext &ctx)
{
    if (ctx.arg->localRole != SCATTER_ROLE_HELPER) {
        return CCU_SUCCESS;
    }
    if (ctx.arg->channelCount != 1) {
        return CcuResult::CCU_E_PARA;
    }


    ccu::LocalAddr src;
    src.addr = ctx.relayBuffer;
    src.addr += ctx.scratchOffset;
    src.token = ctx.relayBufferToken;

    ccu::RemoteAddr dst;
    dst.addr = ctx.output[1];
    dst.addr += ctx.windowOffset;
    dst.addr += ctx.directBytes;
    dst.addr += ctx.relayOffset;
    dst.token = ctx.token[1];

    CCU_RETURN_IF_ERROR(ccu::Write(ctx.arg->channels[0], dst, src, ctx.relayChunkBytes, ctx.event, 1));
    CCU_RETURN_IF_ERROR(ccu::EventWait(ctx.event, 1));
    return CCU_SUCCESS;
}

/*
 * Direct prefix/full-window delivery. Scratch and recvBuf are disjoint, so this
 * is NOT the completion barrier for Forward. The host explicitly joins the
 * Forward lane before reusing a window's resources or completing Scatter.
 */
CcuResult DoRelayFinal(ScatterContext &ctx)
{
    if (ctx.arg->rankId != ctx.arg->rootId) {
        return CCU_SUCCESS;
    }

    uint16_t totalMask = 0;
    CCU_RETURN_IF_ERROR(StartRootCopy(ctx, ctx.event, totalMask));
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        const uint32_t peer = ctx.arg->peerRanks[channelIdx];

        ccu::LocalAddr src;
        src.addr = ctx.input;
        AddRankOffset(src, ctx.sliceStride, peer);
        src.addr += ctx.windowOffset;
        src.token = ctx.inputToken;

        ccu::RemoteAddr dst;
        dst.addr = ctx.output[static_cast<size_t>(channelIdx) + 1U];
        dst.addr += ctx.windowOffset;
        dst.token = ctx.token[static_cast<size_t>(channelIdx) + 1U];

        const uint16_t mask = static_cast<uint16_t>(1U << channelIdx);
        totalMask = static_cast<uint16_t>(totalMask | mask);

        if (ctx.arg->peerRoles[channelIdx] == SCATTER_PEER_RELAY_TARGET) {
            CCU_IF(ctx.directBytes != 0)
            {
                CCU_RETURN_IF_ERROR(
                    ccu::Write(ctx.arg->channels[channelIdx], dst, src, ctx.directBytes, ctx.event, mask));
            }
            CCU_ELSE
            {
                CCU_RETURN_IF_ERROR(ccu::EventRecord(ctx.event, mask));
            }
        } else {
            CCU_RETURN_IF_ERROR(
                ccu::Write(ctx.arg->channels[channelIdx], dst, src, ctx.sliceSize, ctx.event, mask));
        }
    }
    return WaitEventIfNeeded(ctx, totalMask);
}

CcuResult ValidateKernelArg(const CcuKernelArgScatter *kernelArg)
{
    if (kernelArg == nullptr || kernelArg->rankSize <= 1 || kernelArg->rankSize > MAX_RANK_SIZE ||
        kernelArg->rankId >= kernelArg->rankSize || kernelArg->rootId >= kernelArg->rankSize ||
        kernelArg->channelCount == 0 || kernelArg->channelCount >= 16 ||
        kernelArg->copyRoot > 1U || kernelArg->copyRootLate > 1U ||
        (kernelArg->copyRootLate != 0U && kernelArg->copyRoot == 0U) ||
        (kernelArg->copyRoot != 0U && kernelArg->rankId != kernelArg->rootId)) {
        return CcuResult::CCU_E_PARA;
    }
    return CCU_SUCCESS;
}

} // namespace

CcuResult CcuScatterDirectLegacyKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgScatter *>(arg);
    CCU_RETURN_IF_ERROR(ValidateKernelArg(kernelArg));
    ScatterContext ctx{};
    ctx.arg = kernelArg;
    CCU_RETURN_IF_ERROR(InitRemoteResources(ctx));
    uint32_t argId = 0;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.input, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.output[0], argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.inputToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.token[0], argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.sliceStride, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.windowOffset, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(ctx.sliceSize, argId++));
    CCU_RETURN_IF_ERROR(LegacyPreSyncMemory(ctx, ctx.output[0], ctx.token[0]));
    CCU_RETURN_IF_ERROR(LegacyDoDirectScatter(ctx));
    CCU_RETURN_IF_ERROR(LegacyPostSync(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuScatterDirectKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgScatter *>(arg);
    CCU_RETURN_IF_ERROR(ValidateKernelArg(kernelArg));

    ScatterContext ctx{};
    ctx.arg = kernelArg;
    CCU_RETURN_IF_ERROR(InitRemoteResources(ctx));
    CCU_RETURN_IF_ERROR(LoadDirectArgs(ctx));
    CCU_RETURN_IF_ERROR(PreSync(ctx, ctx.arg->rankId == ctx.arg->rootId));
    CCU_RETURN_IF_ERROR(DoDirectScatter(ctx, ctx.event));
    CCU_RETURN_IF_ERROR(PostSync(ctx, ctx.arg->rankId == ctx.arg->rootId));
    return CCU_SUCCESS;
}

// One address exchange and completion barrier for up to two bounded writes.
// No dependence on channel registers surviving between separate launches.
// Direct now has eight runtime arguments; Batch has nine.
CcuResult CcuScatterDirectBatchKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgScatter *>(arg);
    CCU_RETURN_IF_ERROR(ValidateKernelArg(kernelArg));
    ScatterContext ctx{};
    ctx.arg = kernelArg;
    CCU_RETURN_IF_ERROR(InitRemoteResources(ctx));
    CCU_RETURN_IF_ERROR(LoadDirectArgs(ctx));
    ccu::Variable nextSliceSize;
    ccu::Event nextWindowEvent; // Independent completion bits for the second window.
    CCU_RETURN_IF_ERROR(ccu::LoadArg(nextSliceSize, 8U));
    CCU_RETURN_IF_ERROR(PreSync(ctx, ctx.arg->rankId == ctx.arg->rootId));
    CCU_RETURN_IF_ERROR(DoDirectScatter(ctx, ctx.event));
    CCU_IF(nextSliceSize != 0)
    {
        ctx.windowOffset += ctx.sliceSize;
        ctx.sliceSize = nextSliceSize;
        CCU_RETURN_IF_ERROR(DoDirectScatter(ctx, nextWindowEvent));
    }
    CCU_RETURN_IF_ERROR(PostSync(ctx, ctx.arg->rankId == ctx.arg->rootId));
    return CCU_SUCCESS;
}

CcuResult CcuScatterRelayLoadKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgScatter *>(arg);
    CCU_RETURN_IF_ERROR(ValidateKernelArg(kernelArg));

    ScatterContext ctx{};
    ctx.arg = kernelArg;
    CCU_RETURN_IF_ERROR(InitRemoteResources(ctx));
    CCU_RETURN_IF_ERROR(LoadRelayArgs(ctx));

    // V30.6 fuses three root-helper operations into one registered handle.
    // Runtime arg #8 is mode 0/1 for the two relay targets and mode 2 for the
    // helper's own final Scatter slice.  This removes duplicate CCU missions
    // on the same root-helper Channel while preserving the original directed
    // ready/done handshakes.
    CCU_IF(ctx.relayBytes == 2)
    {
        CCU_RETURN_IF_ERROR(PreSync(ctx, ctx.arg->rankId == ctx.arg->rootId));
        CCU_RETURN_IF_ERROR(DoRelayFinal(ctx));
        CCU_RETURN_IF_ERROR(PostSync(ctx, ctx.arg->rankId == ctx.arg->rootId));
    }
    CCU_ELSE
    {
        CCU_RETURN_IF_ERROR(PreSyncRelayScratch(ctx));
        CCU_RETURN_IF_ERROR(DoRelayLoad(ctx));
        CCU_RETURN_IF_ERROR(PostSync(ctx, ctx.arg->localRole == SCATTER_ROLE_ROOT));
    }
    return CCU_SUCCESS;
}

CcuResult CcuScatterRelayForwardKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgScatter *>(arg);
    CCU_RETURN_IF_ERROR(ValidateKernelArg(kernelArg));

    ScatterContext ctx{};
    ctx.arg = kernelArg;
    CCU_RETURN_IF_ERROR(InitRemoteResources(ctx));
    CCU_RETURN_IF_ERROR(LoadRelayArgs(ctx));

    // Every stage consumes fresh receiver READY before writing. The receiver
    // waits for DONE before its next READY, preserving safe notify reuse.
    CCU_RETURN_IF_ERROR(PreSync(ctx, ctx.arg->localRole == SCATTER_ROLE_HELPER));

    CCU_RETURN_IF_ERROR(DoRelayForward(ctx));
    CCU_RETURN_IF_ERROR(PostSync(ctx, ctx.arg->localRole == SCATTER_ROLE_HELPER));
    return CCU_SUCCESS;
}

CcuResult CcuScatterRelayFinalKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgScatter *>(arg);
    CCU_RETURN_IF_ERROR(ValidateKernelArg(kernelArg));

    ScatterContext ctx{};
    ctx.arg = kernelArg;
    CCU_RETURN_IF_ERROR(InitRemoteResources(ctx));
    CCU_RETURN_IF_ERROR(LoadRelayArgs(ctx));
    CCU_RETURN_IF_ERROR(PreSync(ctx, ctx.arg->rankId == ctx.arg->rootId));
    CCU_RETURN_IF_ERROR(DoRelayFinal(ctx));
    CCU_RETURN_IF_ERROR(PostSync(ctx, ctx.arg->rankId == ctx.arg->rootId));
    return CCU_SUCCESS;
}

// V20 uses the same directed handshake for Hub transfers. The receiver
// cannot publish the next READY until it has consumed this transfer's DONE.
namespace {
struct HubContext {
    const CcuKernelArgHub *arg = nullptr;
    ccu::Variable srcBase, srcToken, receiveBase, receiveToken;
    ccu::Variable stride, srcOffset, dstOffset, bytes, pitch, rootCopyEnabled;
    std::vector<ccu::Variable> remoteBase, remoteToken;
    ccu::Event complete;
};

CcuResult InitHub(HubContext &c, const CcuKernelArgHub *a, bool ingress)
{
    if (a == nullptr || a->rankSize != 16U || a->rankId >= 16U ||
        a->rootId >= 16U || a->isSender > 1U || a->copyRoot > 1U ||
        (a->copyRoot != 0U && (a->isSender == 0U || a->rankId != a->rootId || ingress)) || a->channelCount == 0U ||
        a->channelCount > 7U || (ingress && a->channelCount != 1U)) {
        return CCU_E_PARA;
    }
    c.arg = a;
    c.remoteBase.resize(a->channelCount);
    c.remoteToken.resize(a->channelCount);
    for (uint32_t i = 0; i < a->channelCount; ++i) {
        if (a->peerRanks[i] >= 16U || a->peerRanks[i] == a->rankId ||
            a->sourceIndices[i] >= 16U) { return CCU_E_PARA; }
        for (uint32_t j = 0; j < i; ++j) {
            if (a->peerRanks[i] == a->peerRanks[j]) { return CCU_E_PARA; }
        }
        c.remoteBase[i] = ccu::GetResByChannel<ccu::Variable>(a->channels[i], OUTPUT_XN_ID);
        c.remoteToken[i] = ccu::GetResByChannel<ccu::Variable>(a->channels[i], TOKEN_XN_ID);
    }
    if (ingress) {
        for (uint32_t i = 0; i < 7U; ++i) {
            if (a->targetRanks[i] >= 16U || a->targetRanks[i] == a->rootId) { return CCU_E_PARA; }
            for (uint32_t j = 0; j < i; ++j) {
                if (a->targetRanks[j] == a->targetRanks[i]) { return CCU_E_PARA; }
            }
        }
    }
    uint32_t idx = 0;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(c.srcBase, idx++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(c.srcToken, idx++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(c.receiveBase, idx++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(c.receiveToken, idx++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(c.stride, idx++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(c.srcOffset, idx++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(c.dstOffset, idx++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(c.bytes, idx++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(c.rootCopyEnabled, idx++));
    if (ingress) { CCU_RETURN_IF_ERROR(ccu::LoadArg(c.pitch, idx++)); }
    return CCU_SUCCESS;
}

CcuResult HubPre(HubContext &c)
{
    for (uint32_t i = 0; i < c.arg->channelCount; ++i) {
        if (c.arg->isSender == 0U) {
            CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(c.arg->channels[i], c.receiveBase,
                OUTPUT_XN_ID, CKE_IDX_0, 1U << OUTPUT_XN_ID));
            CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(c.arg->channels[i], c.receiveToken,
                TOKEN_XN_ID, CKE_IDX_0, 1U << TOKEN_XN_ID));
        } else {
            CCU_RETURN_IF_ERROR(ccu::NotifyWait(c.arg->channels[i], CKE_IDX_0,
                (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID)));
        }
    }
    return CCU_SUCCESS;
}

CcuResult HubPost(HubContext &c)
{
    for (uint32_t i = 0; i < c.arg->channelCount; ++i) {
        if (c.arg->isSender != 0U) {
            CCU_RETURN_IF_ERROR(ccu::NotifyRecord(c.arg->channels[i], CKE_IDX_0, 1U << POST_SYNC_ID));
        } else {
            CCU_RETURN_IF_ERROR(ccu::NotifyWait(c.arg->channels[i], CKE_IDX_0, 1U << POST_SYNC_ID));
        }
    }
    return CCU_SUCCESS;
}

CcuResult HubRootCopy(HubContext &c, uint16_t &mask)
{
    if (c.arg->copyRoot == 0U) { return CCU_SUCCESS; }
    ccu::LocalAddr src, dst;
    src.addr = c.srcBase;
    AddRankOffset(src, c.stride, c.arg->rootId);
    src.addr += c.srcOffset; src.token = c.srcToken;
    dst.addr = c.receiveBase; dst.addr += c.dstOffset; dst.token = c.receiveToken;
    CCU_IF(c.rootCopyEnabled != 0)
    {
        CCU_RETURN_IF_ERROR(ccu::LocalCopy(dst, src, c.bytes, c.complete, ROOT_COPY_MASK));
    }
    CCU_ELSE
    {
        CCU_RETURN_IF_ERROR(ccu::EventRecord(c.complete, ROOT_COPY_MASK));
    }
    mask = static_cast<uint16_t>(mask | ROOT_COPY_MASK);
    return CCU_SUCCESS;
}

} // namespace

CcuResult CcuScatterHubCopyKernel(CcuKernelArg arg)
{
    HubContext c;
    CCU_RETURN_IF_ERROR(InitHub(c, static_cast<CcuKernelArgHub *>(arg), false));
    CCU_RETURN_IF_ERROR(HubPre(c));
    if (c.arg->isSender != 0U) {
        uint16_t mask = 0;
        CCU_RETURN_IF_ERROR(HubRootCopy(c, mask));
        for (uint32_t i = 0; i < c.arg->channelCount; ++i) {
            ccu::LocalAddr src;
            src.addr = c.srcBase;
            AddRankOffset(src, c.stride, c.arg->sourceIndices[i]);
            src.addr += c.srcOffset;
            src.token = c.srcToken;
            ccu::RemoteAddr dst;
            dst.addr = c.remoteBase[i];
            dst.addr += c.dstOffset;
            dst.token = c.remoteToken[i];
            const uint16_t bit = static_cast<uint16_t>(1U << i);
            mask = static_cast<uint16_t>(mask | bit);
            CCU_RETURN_IF_ERROR(ccu::Write(c.arg->channels[i], dst, src, c.bytes, c.complete, bit));
        }
        CCU_RETURN_IF_ERROR(ccu::EventWait(c.complete, mask));
    }
    CCU_RETURN_IF_ERROR(HubPost(c));
    return CCU_SUCCESS;
}

CcuResult CcuScatterHubIngressKernel(CcuKernelArg arg)
{
    HubContext c;
    CCU_RETURN_IF_ERROR(InitHub(c, static_cast<CcuKernelArgHub *>(arg), true));
    CCU_RETURN_IF_ERROR(HubPre(c));
    if (c.arg->isSender != 0U) {
        // Seven disjoint input segments on ONE root->B0 Channel. Distinct event
        // bits are used even though the Channel is shared; one final barrier.
        for (uint32_t i = 0; i < 7U; ++i) {
            ccu::LocalAddr src;
            src.addr = c.srcBase;
            AddRankOffset(src, c.stride, c.arg->targetRanks[i]);
            src.addr += c.srcOffset;
            src.token = c.srcToken;
            ccu::RemoteAddr dst;
            dst.addr = c.remoteBase[0];
            dst.addr += c.dstOffset;
            for (uint32_t j = 0; j < i; ++j) { dst.addr += c.pitch; }
            dst.token = c.remoteToken[0];
            CCU_RETURN_IF_ERROR(ccu::Write(c.arg->channels[0], dst, src, c.bytes,
                c.complete, static_cast<uint16_t>(1U << i)));
        }
        CCU_RETURN_IF_ERROR(ccu::EventWait(c.complete, 0x7FU));
    }
    CCU_RETURN_IF_ERROR(HubPost(c));
    return CCU_SUCCESS;
}

#undef CCU_RETURN_IF_ERROR

} // namespace ops_hccl
