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

// Direct kernel keeps the original single-notify protocol.
// Relay uses the four CCU notify slots independently so address exchange,
// READY, pair completion and final root synchronization cannot alias.
constexpr int CKE_IDX_0 = 0;
constexpr int POST_SYNC_ID = 3;
// Separate logically different synchronization epochs across Notify slots.
// This avoids reusing the same CKE index + mask for consecutive address
// handshakes on the same Channel.
constexpr int RELAY_STAGE_ADDR_CKE_IDX = 0;
constexpr int RELAY_READY_CKE_IDX = 1;
constexpr int RELAY_RECV_ADDR_CKE_IDX = 2;
constexpr int RELAY_ROOT_DONE_CKE_IDX = 3;
constexpr int RELAY_FORWARD_ADDR_CKE_IDX = 4;
constexpr int RELAY_FORWARD_DONE_CKE_IDX = 5;
constexpr uint16_t RELAY_SYNC_MASK = 1U;
constexpr uint64_t RELAY_PHASE_ROOT_FANOUT = 1;
constexpr uint64_t RELAY_PHASE_FORWARD = 2;
constexpr uint64_t RELAY_PHASE_FINAL_SYNC = 3;

#define SCATTER_CCU_CHK_RET(call) \
    do { \
        CcuResult scatterCcuRet = (call); \
        if (scatterCcuRet != CCU_SUCCESS) { \
            return scatterCcuRet; \
        } \
    } while (0)

CcuResult InitRemoteResources(CcuScatterDirectContext &ctx)
{
    if (ctx.arg->channelCount == 0 || ctx.arg->channelCount > ctx.arg->rankSize - 1) {
        return CcuResult::CCU_E_INTERNAL;
    }

    ctx.output.resize(ctx.arg->rankSize);
    ctx.outputToken.resize(ctx.arg->rankSize);

    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        const uint32_t peerRank = ctx.arg->peerRanks[i];
        if (peerRank >= ctx.arg->rankSize || peerRank == ctx.arg->rankId) {
            return CcuResult::CCU_E_INTERNAL;
        }
        ctx.output[peerRank] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[i], OUTPUT_XN_ID);
        ctx.outputToken[peerRank] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[i], TOKEN_XN_ID);
    }
    return CCU_SUCCESS;
}

CcuResult LoadArgs(CcuScatterDirectContext &ctx)
{
    uint32_t argId = 0;
    SCATTER_CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    SCATTER_CCU_CHK_RET(ccu::LoadArg(ctx.inputToken, argId++));
    SCATTER_CCU_CHK_RET(ccu::LoadArg(ctx.output[ctx.arg->rankId], argId++));
    SCATTER_CCU_CHK_RET(ccu::LoadArg(ctx.outputToken[ctx.arg->rankId], argId++));
    SCATTER_CCU_CHK_RET(ccu::LoadArg(ctx.recvBytes, argId++));
    SCATTER_CCU_CHK_RET(ccu::LoadArg(ctx.chunkOffset, argId++));
    SCATTER_CCU_CHK_RET(ccu::LoadArg(ctx.chunkSize, argId++));
    SCATTER_CCU_CHK_RET(ccu::LoadArg(ctx.root, argId++));
    return CCU_SUCCESS;
}

CcuResult ExchangeAddressOnChannel(CcuScatterDirectContext &ctx, uint32_t channelIdx)
{
    SCATTER_CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx],
        ctx.output[ctx.arg->rankId], OUTPUT_XN_ID, CKE_IDX_0, 1U << OUTPUT_XN_ID));
    SCATTER_CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx],
        ctx.outputToken[ctx.arg->rankId], TOKEN_XN_ID, CKE_IDX_0, 1U << TOKEN_XN_ID));

    const uint32_t allBit = (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
    SCATTER_CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIdx], CKE_IDX_0, allBit));
    return CCU_SUCCESS;
}

CcuResult PreSync(CcuScatterDirectContext &ctx)
{
    const uint64_t myRankImm = static_cast<uint64_t>(ctx.arg->rankId);

    // Root exchanges its address/token with every peer represented by this
    // same-die kernel.
    CCU_IF(ctx.root == myRankImm)
    {
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            SCATTER_CCU_CHK_RET(ExchangeAddressOnChannel(ctx, i));
        }
    }

    // Non-root ranks launch all local die kernels so opposite-end die numbering
    // cannot deadlock. Only the kernel that actually contains runtime root does
    // any synchronization; all other kernels become no-ops.
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        const uint64_t peerRankImm = static_cast<uint64_t>(ctx.arg->peerRanks[i]);
        CCU_IF(ctx.root == peerRankImm)
        {
            SCATTER_CCU_CHK_RET(ExchangeAddressOnChannel(ctx, i));
        }
    }

    return CCU_SUCCESS;
}

CcuResult DoScatter(CcuScatterDirectContext &ctx)
{
    const uint64_t myRankImm = static_cast<uint64_t>(ctx.arg->rankId);

    CCU_IF(ctx.root == myRankImm)
    {
        uint16_t allEventMask = 0;
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            const uint32_t peerRank = ctx.arg->peerRanks[i];

            ccu::Variable srcOffset;
            srcOffset = ctx.chunkOffset;
            // peerRank is a registration-time immediate. Build
            // peerRank * recvBytes without relying on Variable*Variable.
            for (uint32_t rank = 0; rank < peerRank; ++rank) {
                srcOffset += ctx.recvBytes;
            }

            ccu::LocalAddr src;
            src.addr = ctx.input;
            src.addr += srcOffset;
            src.token = ctx.inputToken;

            ccu::RemoteAddr dst;
            dst.addr = ctx.output[peerRank];
            dst.addr += ctx.chunkOffset;
            dst.token = ctx.outputToken[peerRank];

            const uint16_t eventMask = static_cast<uint16_t>(1U << i);
            allEventMask = static_cast<uint16_t>(allEventMask | eventMask);
            SCATTER_CCU_CHK_RET(
                ccu::Write(ctx.arg->channels[i], dst, src, ctx.chunkSize, ctx.event, eventMask));
        }
        SCATTER_CCU_CHK_RET(ccu::EventWait(ctx.event, allEventMask));
    }

    return CCU_SUCCESS;
}

CcuResult PostSyncOne(CcuScatterDirectContext &ctx, uint32_t channelIdx)
{
    SCATTER_CCU_CHK_RET(
        ccu::NotifyRecord(ctx.arg->channels[channelIdx], CKE_IDX_0, 1U << POST_SYNC_ID));
    SCATTER_CCU_CHK_RET(
        ccu::NotifyWait(ctx.arg->channels[channelIdx], CKE_IDX_0, 1U << POST_SYNC_ID));
    return CCU_SUCCESS;
}

CcuResult PostSync(CcuScatterDirectContext &ctx)
{
    const uint64_t myRankImm = static_cast<uint64_t>(ctx.arg->rankId);

    CCU_IF(ctx.root == myRankImm)
    {
        // Record to all peers first, then wait for all peers. This avoids
        // serializing same-die channels behind pairwise post-sync handshakes.
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            SCATTER_CCU_CHK_RET(
                ccu::NotifyRecord(ctx.arg->channels[i], CKE_IDX_0, 1U << POST_SYNC_ID));
        }
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            SCATTER_CCU_CHK_RET(
                ccu::NotifyWait(ctx.arg->channels[i], CKE_IDX_0, 1U << POST_SYNC_ID));
        }
    }

    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        const uint64_t peerRankImm = static_cast<uint64_t>(ctx.arg->peerRanks[i]);
        CCU_IF(ctx.root == peerRankImm)
        {
            SCATTER_CCU_CHK_RET(PostSyncOne(ctx, i));
        }
    }

    return CCU_SUCCESS;
}

CcuResult ExchangeRelayValuesOnChannel(CcuScatterRelayContext &ctx, uint32_t channelIdx,
    ccu::Variable &value, ccu::Variable &token, uint32_t notifyIdx)
{
    SCATTER_CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx],
        value, OUTPUT_XN_ID, notifyIdx, 1U << OUTPUT_XN_ID));
    SCATTER_CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx],
        token, TOKEN_XN_ID, notifyIdx, 1U << TOKEN_XN_ID));

    const uint32_t allBit = (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
    SCATTER_CCU_CHK_RET(
        ccu::NotifyWait(ctx.arg->channels[channelIdx], notifyIdx, allBit));
    return CCU_SUCCESS;
}

CcuResult InitRelayResources(CcuScatterRelayContext &ctx)
{
    if (ctx.arg->channelCount == 0 || ctx.arg->channelCount > ctx.arg->rankSize - 1) {
        return CcuResult::CCU_E_INTERNAL;
    }

    ctx.output.resize(ctx.arg->rankSize);
    ctx.outputToken.resize(ctx.arg->rankSize);

    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        const uint32_t peerRank = ctx.arg->peerRanks[i];
        if (peerRank >= ctx.arg->rankSize || peerRank == ctx.arg->rankId) {
            return CcuResult::CCU_E_INTERNAL;
        }
        ctx.output[peerRank] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[i], OUTPUT_XN_ID);
        ctx.outputToken[peerRank] = ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[i], TOKEN_XN_ID);
    }
    return CCU_SUCCESS;
}

CcuResult LoadRelayArgs(CcuScatterRelayContext &ctx)
{
    uint32_t argId = 0;
    SCATTER_CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    SCATTER_CCU_CHK_RET(ccu::LoadArg(ctx.inputToken, argId++));
    SCATTER_CCU_CHK_RET(ccu::LoadArg(ctx.output[ctx.arg->rankId], argId++));
    SCATTER_CCU_CHK_RET(ccu::LoadArg(ctx.outputToken[ctx.arg->rankId], argId++));
    SCATTER_CCU_CHK_RET(ccu::LoadArg(ctx.relayBufferLocal, argId++));
    SCATTER_CCU_CHK_RET(ccu::LoadArg(ctx.relayBufferTokenLocal, argId++));
    SCATTER_CCU_CHK_RET(ccu::LoadArg(ctx.recvBytes, argId++));
    SCATTER_CCU_CHK_RET(ccu::LoadArg(ctx.chunkOffset, argId++));
    SCATTER_CCU_CHK_RET(ccu::LoadArg(ctx.chunkSize, argId++));
    SCATTER_CCU_CHK_RET(ccu::LoadArg(ctx.directBytes, argId++));
    SCATTER_CCU_CHK_RET(ccu::LoadArg(ctx.relayBytes, argId++));
    SCATTER_CCU_CHK_RET(ccu::LoadArg(ctx.root, argId++));
    SCATTER_CCU_CHK_RET(ccu::LoadArg(ctx.relayTargetRank, argId++));
    SCATTER_CCU_CHK_RET(ccu::LoadArg(ctx.relaySourceRank, argId++));
    SCATTER_CCU_CHK_RET(ccu::LoadArg(ctx.phase, argId++));
    return CCU_SUCCESS;
}

CcuResult PreSyncRelayRootFanout(CcuScatterRelayContext &ctx)
{
    const uint64_t myRankImm = static_cast<uint64_t>(ctx.arg->rankId);

    CCU_IF(ctx.root == myRankImm)
    {
        // Root<->local-relay channels first exchange the relay HCCL staging
        // address. Their normal recvBuf exchange is deliberately postponed
        // until the staged suffix has landed, so XN1/XN2 can be reused without
        // needing extra scratch Variables. All other channels exchange recvBuf
        // addresses immediately.
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            const uint32_t peerRank = ctx.arg->peerRanks[i];
            if (ctx.arg->rootRelayTargetByRank[peerRank] != INVALID_VALUE_RANKID) {
                SCATTER_CCU_CHK_RET(ExchangeRelayValuesOnChannel(
                    ctx, i, ctx.relayBufferLocal, ctx.relayBufferTokenLocal,
                    RELAY_STAGE_ADDR_CKE_IDX));
            } else {
                SCATTER_CCU_CHK_RET(ExchangeRelayValuesOnChannel(
                    ctx, i, ctx.output[ctx.arg->rankId], ctx.outputToken[ctx.arg->rankId],
                    RELAY_STAGE_ADDR_CKE_IDX));
            }
        }
    }

    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        const uint64_t peerRankImm = static_cast<uint64_t>(ctx.arg->peerRanks[i]);
        CCU_IF(ctx.root == peerRankImm)
        {
            if (ctx.arg->peerIsLocal[i] != 0) {
                // In both enabled relay topologies every non-root rank on the
                // root server is paired as a relay (7/7 for 2x8, 2/2 for 4x3).
                SCATTER_CCU_CHK_RET(ExchangeRelayValuesOnChannel(
                    ctx, i, ctx.relayBufferLocal, ctx.relayBufferTokenLocal,
                    RELAY_STAGE_ADDR_CKE_IDX));
            } else {
                SCATTER_CCU_CHK_RET(ExchangeRelayValuesOnChannel(
                    ctx, i, ctx.output[ctx.arg->rankId], ctx.outputToken[ctx.arg->rankId],
                    RELAY_STAGE_ADDR_CKE_IDX));
            }
        }
    }
    return CCU_SUCCESS;
}

CcuResult DoRelayRootFanout(CcuScatterRelayContext &ctx)
{
    const uint64_t myRankImm = static_cast<uint64_t>(ctx.arg->rankId);

    CCU_IF(ctx.root == myRankImm)
    {
        uint16_t localRelayMask = 0;
        uint16_t remoteDirectMask = 0;

        // Wave 1: root Clos direct traffic and root->local-relay Mesh staging
        // start together. Each channel issues exactly one write in this wave.
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            const uint32_t peerRank = ctx.arg->peerRanks[i];
            const uint16_t eventMask = static_cast<uint16_t>(1U << i);

            if (ctx.arg->rootRelayTargetByRank[peerRank] != INVALID_VALUE_RANKID) {
                localRelayMask = static_cast<uint16_t>(localRelayMask | eventMask);
                const uint32_t targetRank = ctx.arg->rootRelayTargetByRank[peerRank];

                ccu::Variable srcOffset;
                srcOffset = ctx.chunkOffset;
                for (uint32_t rank = 0; rank < targetRank; ++rank) {
                    srcOffset += ctx.recvBytes;
                }
                srcOffset += ctx.directBytes;

                ccu::LocalAddr src;
                src.addr = ctx.input;
                src.addr += srcOffset;
                src.token = ctx.inputToken;

                ccu::RemoteAddr dst;
                dst.addr = ctx.output[peerRank];
                dst.token = ctx.outputToken[peerRank];

                SCATTER_CCU_CHK_RET(ccu::Write(
                    ctx.arg->channels[i], dst, src, ctx.relayBytes, ctx.firstWaveEvent, eventMask));
            } else {
                remoteDirectMask = static_cast<uint16_t>(remoteDirectMask | eventMask);

                ccu::Variable srcOffset;
                srcOffset = ctx.chunkOffset;
                for (uint32_t rank = 0; rank < peerRank; ++rank) {
                    srcOffset += ctx.recvBytes;
                }

                ccu::LocalAddr src;
                src.addr = ctx.input;
                src.addr += srcOffset;
                src.token = ctx.inputToken;

                ccu::RemoteAddr dst;
                dst.addr = ctx.output[peerRank];
                dst.addr += ctx.chunkOffset;
                dst.token = ctx.outputToken[peerRank];

                if (ctx.arg->rootRelaySourceByRank[peerRank] != INVALID_VALUE_RANKID) {
                    SCATTER_CCU_CHK_RET(ccu::Write(
                        ctx.arg->channels[i], dst, src, ctx.directBytes, ctx.firstWaveEvent, eventMask));
                } else {
                    SCATTER_CCU_CHK_RET(ccu::Write(
                        ctx.arg->channels[i], dst, src, ctx.chunkSize, ctx.firstWaveEvent, eventMask));
                }
            }
        }

        // Once the staged suffixes on this die have landed, wake the relays.
        // Their next-phase Clos forwarding can begin while root finishes the
        // remaining direct traffic and wave-2 local Mesh writes.
        if (localRelayMask != 0) {
            SCATTER_CCU_CHK_RET(ccu::EventWait(ctx.firstWaveEvent, localRelayMask));
            for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
                const uint32_t peerRank = ctx.arg->peerRanks[i];
                if (ctx.arg->rootRelayTargetByRank[peerRank] != INVALID_VALUE_RANKID) {
                    SCATTER_CCU_CHK_RET(
                        ccu::NotifyRecord(ctx.arg->channels[i], RELAY_READY_CKE_IDX, RELAY_SYNC_MASK));
                }
            }

            // The staged suffix is now safe in relay HCCL memory. Reuse XN1/XN2
            // on root<->relay channels for the normal recvBuf exchange needed by
            // wave 2. Each relay performs the matching exchange after READY.
            for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
                const uint32_t peerRank = ctx.arg->peerRanks[i];
                if (ctx.arg->rootRelayTargetByRank[peerRank] != INVALID_VALUE_RANKID) {
                    SCATTER_CCU_CHK_RET(ExchangeRelayValuesOnChannel(
                        ctx, i, ctx.output[ctx.arg->rankId], ctx.outputToken[ctx.arg->rankId],
                        RELAY_RECV_ADDR_CKE_IDX));
                }
            }

            // Wave 2 on a local-relay channel is that relay rank's own complete
            // Scatter slice. It follows staging on the same Mesh channel.
            for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
                const uint32_t peerRank = ctx.arg->peerRanks[i];
                if (ctx.arg->rootRelayTargetByRank[peerRank] == INVALID_VALUE_RANKID) {
                    continue;
                }

                ccu::Variable srcOffset;
                srcOffset = ctx.chunkOffset;
                for (uint32_t rank = 0; rank < peerRank; ++rank) {
                    srcOffset += ctx.recvBytes;
                }

                ccu::LocalAddr src;
                src.addr = ctx.input;
                src.addr += srcOffset;
                src.token = ctx.inputToken;

                ccu::RemoteAddr dst;
                dst.addr = ctx.output[peerRank];
                dst.addr += ctx.chunkOffset;
                dst.token = ctx.outputToken[peerRank];

                const uint16_t eventMask = static_cast<uint16_t>(1U << i);
                SCATTER_CCU_CHK_RET(ccu::Write(
                    ctx.arg->channels[i], dst, src, ctx.chunkSize, ctx.secondWaveEvent, eventMask));
            }
        }

        if (remoteDirectMask != 0) {
            SCATTER_CCU_CHK_RET(ccu::EventWait(ctx.firstWaveEvent, remoteDirectMask));
        }
        if (localRelayMask != 0) {
            SCATTER_CCU_CHK_RET(ccu::EventWait(ctx.secondWaveEvent, localRelayMask));
        }

        // Directional phase boundary: after every root-originating transfer in
        // phase 1 is complete, root releases every peer. Root never waits back
        // here, so this boundary cannot participate in a cyclic dependency.
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            SCATTER_CCU_CHK_RET(
                ccu::NotifyRecord(ctx.arg->channels[i], RELAY_ROOT_DONE_CKE_IDX, RELAY_SYNC_MASK));
        }
    }

    // Only the kernel containing the runtime-root channel participates.
    // Local relays wait only until their staged suffix is ready, then exchange
    // their real recvBuf and are free to enter phase 2 immediately.
    //
    // For 2x8 (rankSize==16), ROOT_DONE is deliberately NOT consumed here:
    // relay sources and relay targets may start phase-2 forwarding while root
    // is still finishing its direct Clos traffic / local Mesh writes.  The
    // ROOT_DONE completion wait is deferred to phase 3.
    //
    // Keep the existing 12-rank behavior unchanged until 4x3 relay is enabled
    // and validated separately.
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        const uint64_t peerRankImm = static_cast<uint64_t>(ctx.arg->peerRanks[i]);
        CCU_IF(ctx.root == peerRankImm)
        {
            if (ctx.arg->peerIsLocal[i] != 0) {
                SCATTER_CCU_CHK_RET(
                    ccu::NotifyWait(ctx.arg->channels[i], RELAY_READY_CKE_IDX, RELAY_SYNC_MASK));
                SCATTER_CCU_CHK_RET(ExchangeRelayValuesOnChannel(
                    ctx, i, ctx.output[ctx.arg->rankId], ctx.outputToken[ctx.arg->rankId],
                        RELAY_RECV_ADDR_CKE_IDX));
            }


        }
    }
    return CCU_SUCCESS;
}

CcuResult PreSyncRelayForward(CcuScatterRelayContext &ctx)
{
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        const uint64_t peerRankImm = static_cast<uint64_t>(ctx.arg->peerRanks[i]);
        CCU_IF(ctx.relayTargetRank == peerRankImm)
        {
            SCATTER_CCU_CHK_RET(ExchangeRelayValuesOnChannel(
                ctx, i, ctx.output[ctx.arg->rankId], ctx.outputToken[ctx.arg->rankId],
                RELAY_FORWARD_ADDR_CKE_IDX));
        }
        CCU_IF(ctx.relaySourceRank == peerRankImm)
        {
            SCATTER_CCU_CHK_RET(ExchangeRelayValuesOnChannel(
                ctx, i, ctx.output[ctx.arg->rankId], ctx.outputToken[ctx.arg->rankId],
                RELAY_FORWARD_ADDR_CKE_IDX));
        }
    }
    return CCU_SUCCESS;
}

CcuResult DoRelayForward(CcuScatterRelayContext &ctx)
{
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        const uint64_t peerRankImm = static_cast<uint64_t>(ctx.arg->peerRanks[i]);
        CCU_IF(ctx.relayTargetRank == peerRankImm)
        {
            ccu::LocalAddr src;
            src.addr = ctx.relayBufferLocal;
            src.token = ctx.relayBufferTokenLocal;

            ccu::RemoteAddr dst;
            dst.addr = ctx.output[ctx.arg->peerRanks[i]];
            dst.addr += ctx.chunkOffset;
            dst.addr += ctx.directBytes;
            dst.token = ctx.outputToken[ctx.arg->peerRanks[i]];

            SCATTER_CCU_CHK_RET(
                ccu::Write(ctx.arg->channels[i], dst, src, ctx.relayBytes, ctx.secondWaveEvent, 1U));
            SCATTER_CCU_CHK_RET(ccu::EventWait(ctx.secondWaveEvent, 1U));
        }
    }
    return CCU_SUCCESS;
}

CcuResult PostSyncRelayForward(CcuScatterRelayContext &ctx)
{
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        const uint64_t peerRankImm = static_cast<uint64_t>(ctx.arg->peerRanks[i]);

        // Relay side: DoRelayForward already EventWait'ed the remote write, so
        // only produce a completion signal for the target.
        CCU_IF(ctx.relayTargetRank == peerRankImm)
        {
            SCATTER_CCU_CHK_RET(
                ccu::NotifyRecord(ctx.arg->channels[i], RELAY_FORWARD_DONE_CKE_IDX, RELAY_SYNC_MASK));
        }

        // Target side: consume the relay completion. No reciprocal Record is
        // needed; this keeps the dependency graph strictly one-way.
        CCU_IF(ctx.relaySourceRank == peerRankImm)
        {
            SCATTER_CCU_CHK_RET(
                ccu::NotifyWait(ctx.arg->channels[i], RELAY_FORWARD_DONE_CKE_IDX, RELAY_SYNC_MASK));
        }
    }
    return CCU_SUCCESS;
}

CcuResult PostSyncRelayBufferFree(CcuScatterRelayContext &ctx)
{
    const uint64_t myRankImm = static_cast<uint64_t>(ctx.arg->rankId);

    // Completion fence is deferred from phase 1 to phase 3 for every enabled
    // Relay topology (2x8 and 4x3). Every non-root rank consumes ROOT_DONE here,
    // after phase-2 forwarding has had a chance to overlap root's remaining
    // direct Clos / local Mesh traffic.
    CCU_IF(ctx.root != myRankImm)
    {
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            const uint64_t peerRankImm = static_cast<uint64_t>(ctx.arg->peerRanks[i]);
            CCU_IF(ctx.root == peerRankImm)
            {
                SCATTER_CCU_CHK_RET(
                    ccu::NotifyWait(ctx.arg->channels[i], RELAY_ROOT_DONE_CKE_IDX, RELAY_SYNC_MASK));
            }
        }
    }

    // Root waits only for ranks that actually own a relay buffer used by this
    // chunk. The relay signal is produced in phase 3, after the per-rank phase-2
    // kernel barrier, so root cannot overwrite a relay buffer too early.
    CCU_IF(ctx.root == myRankImm)
    {
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            const uint32_t peerRank = ctx.arg->peerRanks[i];
            if (ctx.arg->rootRelayTargetByRank[peerRank] != INVALID_VALUE_RANKID) {
                SCATTER_CCU_CHK_RET(
                    ccu::NotifyWait(ctx.arg->channels[i], RELAY_FORWARD_DONE_CKE_IDX, RELAY_SYNC_MASK));
            }
        }
    }

    // Relay side: tell runtime root that this relay's HCCL staging buffer is no
    // longer being read by the forwarding write. This is one-way; relay never
    // waits for root.
    CCU_IF(ctx.relayTargetRank != static_cast<uint64_t>(INVALID_VALUE_RANKID))
    {
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            const uint64_t peerRankImm = static_cast<uint64_t>(ctx.arg->peerRanks[i]);
            CCU_IF(ctx.root == peerRankImm)
            {
                SCATTER_CCU_CHK_RET(
                    ccu::NotifyRecord(ctx.arg->channels[i], RELAY_FORWARD_DONE_CKE_IDX, RELAY_SYNC_MASK));
            }
        }
    }
    return CCU_SUCCESS;
}

} // namespace

CcuResult CcuScatterDirectKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgScatterDirect *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount == 0 ||
        kernelArg->channelCount > kernelArg->rankSize - 1) {
        return CcuResult::CCU_E_INTERNAL;
    }

    CcuScatterDirectContext ctx{};
    ctx.arg = kernelArg;

    SCATTER_CCU_CHK_RET(InitRemoteResources(ctx));
    SCATTER_CCU_CHK_RET(LoadArgs(ctx));
    SCATTER_CCU_CHK_RET(PreSync(ctx));
    SCATTER_CCU_CHK_RET(DoScatter(ctx));
    SCATTER_CCU_CHK_RET(PostSync(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuScatterRelayKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgScatterRelay *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount == 0 ||
        kernelArg->channelCount > kernelArg->rankSize - 1) {
        return CcuResult::CCU_E_INTERNAL;
    }

    CcuScatterRelayContext ctx{};
    ctx.arg = kernelArg;

    SCATTER_CCU_CHK_RET(InitRelayResources(ctx));
    SCATTER_CCU_CHK_RET(LoadRelayArgs(ctx));

    CCU_IF(ctx.phase == RELAY_PHASE_ROOT_FANOUT)
    {
        SCATTER_CCU_CHK_RET(PreSyncRelayRootFanout(ctx));
        SCATTER_CCU_CHK_RET(DoRelayRootFanout(ctx));
    }

    CCU_IF(ctx.phase == RELAY_PHASE_FORWARD)
    {
        SCATTER_CCU_CHK_RET(PreSyncRelayForward(ctx));
        SCATTER_CCU_CHK_RET(DoRelayForward(ctx));
        SCATTER_CCU_CHK_RET(PostSyncRelayForward(ctx));
    }

    CCU_IF(ctx.phase == RELAY_PHASE_FINAL_SYNC)
    {
        SCATTER_CCU_CHK_RET(PostSyncRelayBufferFree(ctx));
    }

    return CCU_SUCCESS;
}

#undef SCATTER_CCU_CHK_RET

} // namespace ops_hccl
