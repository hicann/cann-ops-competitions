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
#include <vector>

#include <hcomm/hcomm_primitives.h>

#include "ccu_kernel.h"

namespace ccu = ::AscendC::ccu;

#define CCU_CALL_RETURN(call) \
    do { \
        const CcuResult ccuCallResult = (call); \
        if (ccuCallResult != CCU_SUCCESS) { \
            return ccuCallResult; \
        } \
    } while (0)

namespace ops_hccl {
namespace {

constexpr uint32_t OUTPUT_XN_ID = 0;
constexpr uint32_t TOKEN_XN_ID = 1;
constexpr uint32_t STAGE_XN_ID = 2;
constexpr uint32_t STAGE_TOKEN_XN_ID = 3;
constexpr uint32_t NOTIFY_INDEX = 0;
constexpr uint32_t POST_SYNC_BIT = 3;
constexpr uint32_t HELPER_SENDER_READY_BIT = 4;
constexpr uint32_t HELPER_ROOT_POST_BIT = 5;
constexpr uint32_t HELPER_STAGE_READY_BIT = 6;
constexpr uint32_t HELPER_DATA_POST_BIT = 7;
constexpr uint32_t HELPER_TARGET_ACK_BIT = 8;
constexpr uint32_t HELPER_OUTPUT_READY_BIT = 9;
constexpr uint32_t CLOS_CAPACITY = 4;
constexpr uint64_t RANK4_TARGET_BLOCK_BYTES = 128ULL * 1024ULL;
constexpr uint64_t RANK16_TARGET_BLOCK_BYTES = 32ULL * 1024ULL;

struct ScatterContext {
    const CcuKernelArgScatter *arg{nullptr};
    ccu::Variable sendBase;
    std::vector<ccu::Variable> recvBase;
    ccu::Variable sendToken;
    std::vector<ccu::Variable> recvToken;
    ccu::Variable blockBytes;
    ccu::Variable chunkOffset;
    ccu::Variable chunkBytes;
    ccu::Variable root;
    ccu::Variable rootSliceInPlace;
    ccu::Variable metadataReuse;
    std::vector<ccu::Variable> sourceAddress;
    ccu::LocalAddr sourceCursor;
    std::vector<ccu::LocalAddr> source;
    std::vector<ccu::RemoteAddr> remoteDestination;
    ccu::LocalAddr localDestination;
    ccu::Event completion;
};

struct SmallReceiverContext {
    const CcuKernelArgScatter *arg{nullptr};
    ccu::Variable recvBase;
    ccu::Variable recvToken;
    ccu::Variable root;
    ccu::Variable metadataReuse;
};

struct Rank12SmallRootContext {
    const CcuKernelArgScatter *arg{nullptr};
    ccu::Variable sendBase;
    std::vector<ccu::Variable> recvBase;
    ccu::Variable sendToken;
    std::vector<ccu::Variable> recvToken;
    ccu::Variable blockBytes;
    ccu::Variable rootSliceInPlace;
    ccu::LocalAddr sourceCursor;
    std::vector<ccu::LocalAddr> source;
    std::vector<ccu::RemoteAddr> remoteDestination;
    ccu::LocalAddr localDestination;
    ccu::Event completion;
};

// Exact 8+4 wide-root Large path.  This keeps the v18 Push direction, but is
// registered as its own mission so the comparison is not polluted by the
// generic Helper phase/route branches.
struct WidePushContext {
    const CcuKernelArgScatter *arg{nullptr};
    ccu::Variable sendChunkBase;
    std::vector<ccu::Variable> recvChunkBase;
    ccu::Variable sendToken;
    std::vector<ccu::Variable> recvToken;
    ccu::Variable blockBytes;
    ccu::Variable fullBytes;
    ccu::Variable root;
    ccu::Variable rootSliceInPlace;
    ccu::LocalAddr sourceCursor;
    std::vector<ccu::LocalAddr> source;
    std::vector<ccu::RemoteAddr> remoteDestination;
    ccu::LocalAddr localDestination;
    ccu::Event completion;
};

struct HelperContext {
    const CcuKernelArgScatter *arg{nullptr};
    ccu::Variable sendChunkBase;
    std::vector<ccu::Variable> recvChunkBase;
    ccu::Variable sendToken;
    std::vector<ccu::Variable> recvToken;
    std::vector<ccu::Variable> stageBase;
    std::vector<ccu::Variable> stageToken;
    ccu::Variable blockBytes;
    ccu::Variable directBytes;
    ccu::Variable root;
    ccu::Variable rootSliceInPlace;
    ccu::Variable delegatedBytes;
    ccu::Variable phase;
    ccu::Variable fullBytes;
    ccu::LocalAddr sourceCursor;
    std::vector<ccu::LocalAddr> source;
    std::vector<ccu::RemoteAddr> remoteDestination;
    ccu::LocalAddr localDestination;
    ccu::LocalAddr localStage;
    ccu::Event completion;
    ccu::Event stageCompletion;
    ccu::Event helperCompletion;
};

CcuResult InitResources(ScatterContext &ctx)
{
    ctx.recvBase.resize(ctx.arg->rankSize);
    ctx.recvToken.resize(ctx.arg->rankSize);
    if (ctx.arg->rankSize == 4U || ctx.arg->rankSize == 16U) {
        ctx.sourceAddress.resize(ctx.arg->rankSize);
    }
    ctx.source.resize(ctx.arg->rankSize);
    ctx.remoteDestination.resize(ctx.arg->rankSize);

    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->peerCount; ++channelIndex) {
        const uint32_t peer = ctx.arg->peerRanks[channelIndex];
        ctx.recvBase[peer] = ccu::GetResByChannel<ccu::Variable>(
            ctx.arg->channels[channelIndex], OUTPUT_XN_ID);
        ctx.recvToken[peer] = ccu::GetResByChannel<ccu::Variable>(
            ctx.arg->channels[channelIndex], TOKEN_XN_ID);
    }
    return CCU_SUCCESS;
}

CcuResult LoadTaskArgs(ScatterContext &ctx)
{
    uint32_t argId = 0;
    CCU_CALL_RETURN(ccu::LoadArg(ctx.sendBase, argId++));
    CCU_CALL_RETURN(ccu::LoadArg(ctx.recvBase[ctx.arg->rankId], argId++));
    CCU_CALL_RETURN(ccu::LoadArg(ctx.sendToken, argId++));
    CCU_CALL_RETURN(ccu::LoadArg(ctx.recvToken[ctx.arg->rankId], argId++));
    CCU_CALL_RETURN(ccu::LoadArg(ctx.blockBytes, argId++));
    CCU_CALL_RETURN(ccu::LoadArg(ctx.chunkOffset, argId++));
    CCU_CALL_RETURN(ccu::LoadArg(ctx.chunkBytes, argId++));
    CCU_CALL_RETURN(ccu::LoadArg(ctx.root, argId++));
    CCU_CALL_RETURN(ccu::LoadArg(ctx.rootSliceInPlace, argId++));
    CCU_CALL_RETURN(ccu::LoadArg(ctx.metadataReuse, argId++));
    return CCU_SUCCESS;
}

CcuResult RootScatter(ScatterContext &ctx)
{
    const uint32_t addressBit = 1U << OUTPUT_XN_ID;
    const uint32_t tokenBit = 1U << TOKEN_XN_ID;
    uint16_t completionMask = 0;
    if (ctx.arg->includeLocal) {
        const uint32_t peer = ctx.arg->rankId;
        ctx.source[peer].addr = ctx.sendBase;
        for (uint32_t offsetRank = 0; offsetRank < peer; ++offsetRank) {
            ctx.source[peer].addr += ctx.blockBytes;
        }
        ctx.source[peer].addr += ctx.chunkOffset;
        ctx.source[peer].token = ctx.sendToken;

        const uint16_t eventBit = static_cast<uint16_t>(1U << peer);
        completionMask = static_cast<uint16_t>(completionMask | eventBit);
        ctx.localDestination.addr = ctx.recvBase[peer];
        ctx.localDestination.addr += ctx.chunkOffset;
        ctx.localDestination.token = ctx.recvToken[peer];
        CCU_IF(ctx.rootSliceInPlace == 0)
        {
            CCU_CALL_RETURN(ccu::LocalCopy(ctx.localDestination, ctx.source[peer],
                ctx.chunkBytes, ctx.completion, eventBit));
        }
        CCU_IF(ctx.rootSliceInPlace != 0)
        {
            CCU_CALL_RETURN(ccu::EventRecord(ctx.completion, eventBit));
        }
    }

    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        CCU_CALL_RETURN(ccu::NotifyWait(ctx.arg->channels[i], NOTIFY_INDEX,
            addressBit | tokenBit));
    }

    ctx.sourceCursor.addr = ctx.sendBase;
    ctx.sourceCursor.token = ctx.sendToken;
    uint32_t cursorRank = 0;
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->peerCount; ++channelIndex) {
        const uint32_t peer = ctx.arg->peerRanks[channelIndex];
        for (; cursorRank < peer; ++cursorRank) {
            ctx.sourceCursor.addr += ctx.blockBytes;
        }
        ctx.source[peer].addr = ctx.sourceCursor.addr;
        ctx.source[peer].addr += ctx.chunkOffset;
        ctx.source[peer].token = ctx.sourceCursor.token;

        ctx.remoteDestination[peer].addr = ctx.recvBase[peer];
        ctx.remoteDestination[peer].addr += ctx.chunkOffset;
        ctx.remoteDestination[peer].token = ctx.recvToken[peer];
        const uint16_t eventBit = static_cast<uint16_t>(1U << peer);
        completionMask = static_cast<uint16_t>(completionMask | eventBit);
        CCU_CALL_RETURN(ccu::Write(ctx.arg->channels[channelIndex],
            ctx.remoteDestination[peer], ctx.source[peer], ctx.chunkBytes,
            ctx.completion, eventBit));
    }
    CCU_CALL_RETURN(ccu::EventWait(ctx.completion, completionMask));

    const uint32_t postBit = 1U << POST_SYNC_BIT;
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        CCU_CALL_RETURN(ccu::NotifyRecord(ctx.arg->channels[i], NOTIFY_INDEX, postBit));
    }
    return CCU_SUCCESS;
}

CcuResult Rank4EarlyWriteRoot(ScatterContext &ctx)
{
    const uint32_t readyBits =
        (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
    uint16_t completionMask = 0;
    if (ctx.arg->includeLocal) {
        const uint32_t peer = ctx.arg->rankId;
        ctx.source[peer].addr = ctx.sendBase;
        for (uint32_t offsetRank = 0; offsetRank < peer; ++offsetRank) {
            ctx.source[peer].addr += ctx.blockBytes;
        }
        ctx.source[peer].addr += ctx.chunkOffset;
        ctx.source[peer].token = ctx.sendToken;

        const uint16_t eventBit = static_cast<uint16_t>(1U << peer);
        completionMask = static_cast<uint16_t>(completionMask | eventBit);
        ctx.localDestination.addr = ctx.recvBase[peer];
        ctx.localDestination.addr += ctx.chunkOffset;
        ctx.localDestination.token = ctx.recvToken[peer];
        CCU_IF(ctx.rootSliceInPlace == 0)
        {
            CCU_CALL_RETURN(ccu::LocalCopy(ctx.localDestination, ctx.source[peer],
                ctx.chunkBytes, ctx.completion, eventBit));
        }
        CCU_IF(ctx.rootSliceInPlace != 0)
        {
            CCU_CALL_RETURN(ccu::EventRecord(ctx.completion, eventBit));
        }
    }

    // Rank4 has one mission and three peer channels.  Keep each peer's
    // address/token dependency, but start its payload as soon as that peer is
    // ready instead of holding all payloads behind the slowest READY.
    ctx.sourceCursor.addr = ctx.sendBase;
    ctx.sourceCursor.token = ctx.sendToken;
    uint32_t cursorRank = 0;
    for (uint32_t channelIndex = 0;
         channelIndex < ctx.arg->peerCount; ++channelIndex) {
        const uint32_t peer = ctx.arg->peerRanks[channelIndex];
        for (; cursorRank < peer; ++cursorRank) {
            ctx.sourceCursor.addr += ctx.blockBytes;
        }
        ctx.source[peer].addr = ctx.sourceCursor.addr;
        ctx.source[peer].addr += ctx.chunkOffset;
        ctx.source[peer].token = ctx.sourceCursor.token;
        CCU_CALL_RETURN(ccu::NotifyWait(ctx.arg->channels[channelIndex],
            NOTIFY_INDEX, readyBits));
        ctx.remoteDestination[peer].addr = ctx.recvBase[peer];
        ctx.remoteDestination[peer].addr += ctx.chunkOffset;
        ctx.remoteDestination[peer].token = ctx.recvToken[peer];
        const uint16_t eventBit = static_cast<uint16_t>(1U << peer);
        completionMask = static_cast<uint16_t>(completionMask | eventBit);
        CCU_CALL_RETURN(ccu::Write(ctx.arg->channels[channelIndex],
            ctx.remoteDestination[peer], ctx.source[peer], ctx.chunkBytes,
            ctx.completion, eventBit));
    }
    CCU_CALL_RETURN(ccu::EventWait(ctx.completion, completionMask));

    const uint32_t postBit = 1U << POST_SYNC_BIT;
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        CCU_CALL_RETURN(ccu::NotifyRecord(ctx.arg->channels[i],
            NOTIFY_INDEX, postBit));
    }
    return CCU_SUCCESS;
}

CcuResult ReceiverSync(ScatterContext &ctx)
{
    const uint32_t addressBit = 1U << OUTPUT_XN_ID;
    const uint32_t tokenBit = 1U << TOKEN_XN_ID;
    const uint32_t postBit = 1U << POST_SYNC_BIT;
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        const uint32_t peer = ctx.arg->peerRanks[i];
        CCU_IF(ctx.root == peer)
        {
            CCU_IF(ctx.metadataReuse == 0)
            {
                CCU_CALL_RETURN(ccu::WriteVariableWithNotify(ctx.arg->channels[i],
                    ctx.recvBase[ctx.arg->rankId], OUTPUT_XN_ID, NOTIFY_INDEX,
                    addressBit));
                CCU_CALL_RETURN(ccu::WriteVariableWithNotify(ctx.arg->channels[i],
                    ctx.recvToken[ctx.arg->rankId], TOKEN_XN_ID, NOTIFY_INDEX,
                    tokenBit));
            }
            CCU_IF(ctx.metadataReuse != 0)
            {
                // The host validated this exact address/token pair against
                // the descriptor last published to this root's channel.  The
                // channel XNs remain unchanged; only issue the per-round
                // consumable READY credit.
                CCU_CALL_RETURN(ccu::NotifyRecord(ctx.arg->channels[i],
                    NOTIFY_INDEX, addressBit | tokenBit));
            }
            CCU_CALL_RETURN(ccu::NotifyWait(ctx.arg->channels[i], NOTIFY_INDEX, postBit));
        }
    }
    return CCU_SUCCESS;
}

CcuResult SmallReceiverSync(SmallReceiverContext &ctx)
{
    const uint32_t addressBit = 1U << OUTPUT_XN_ID;
    const uint32_t tokenBit = 1U << TOKEN_XN_ID;
    const uint32_t postBit = 1U << POST_SYNC_BIT;
    for (uint32_t i = 0; i < ctx.arg->peerCount; ++i) {
        const uint32_t peer = ctx.arg->peerRanks[i];
        CCU_IF(ctx.root == peer)
        {
            CCU_IF(ctx.metadataReuse == 0)
            {
                // Address and token are written in-order on this channel.  A
                // specialized Small root consumes token as the single ready
                // edge after both XNs are current.
                const uint32_t addressNotifyMask =
                    ctx.arg->smallSingleReady != 0 ? 0U : addressBit;
                CCU_CALL_RETURN(ccu::WriteVariableWithNotify(ctx.arg->channels[i],
                    ctx.recvBase, OUTPUT_XN_ID, NOTIFY_INDEX,
                    addressNotifyMask));
                CCU_CALL_RETURN(ccu::WriteVariableWithNotify(ctx.arg->channels[i],
                    ctx.recvToken, TOKEN_XN_ID, NOTIFY_INDEX, tokenBit));
            }
            CCU_IF(ctx.metadataReuse != 0)
            {
                const uint32_t readyBits = ctx.arg->smallSingleReady != 0 ?
                    tokenBit : (addressBit | tokenBit);
                CCU_CALL_RETURN(ccu::NotifyRecord(ctx.arg->channels[i],
                    NOTIFY_INDEX, readyBits));
            }
            CCU_CALL_RETURN(ccu::NotifyWait(ctx.arg->channels[i],
                NOTIFY_INDEX, postBit));
        }
    }
    return CCU_SUCCESS;
}

CcuResult InitRank12SmallRootResources(Rank12SmallRootContext &ctx)
{
    ctx.recvBase.resize(ctx.arg->rankSize);
    ctx.recvToken.resize(ctx.arg->rankSize);
    ctx.source.resize(ctx.arg->rankSize);
    ctx.remoteDestination.resize(ctx.arg->rankSize);
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->peerCount;
        ++channelIndex) {
        const uint32_t peer = ctx.arg->peerRanks[channelIndex];
        ctx.recvBase[peer] = ccu::GetResByChannel<ccu::Variable>(
            ctx.arg->channels[channelIndex], OUTPUT_XN_ID);
        ctx.recvToken[peer] = ccu::GetResByChannel<ccu::Variable>(
            ctx.arg->channels[channelIndex], TOKEN_XN_ID);
    }
    return CCU_SUCCESS;
}

CcuResult LoadRank12SmallRootTaskArgs(Rank12SmallRootContext &ctx)
{
    uint32_t argId = 0;
    CCU_CALL_RETURN(ccu::LoadArg(ctx.sendBase, argId++));
    CCU_CALL_RETURN(ccu::LoadArg(ctx.recvBase[ctx.arg->rankId], argId++));
    CCU_CALL_RETURN(ccu::LoadArg(ctx.sendToken, argId++));
    CCU_CALL_RETURN(ccu::LoadArg(ctx.recvToken[ctx.arg->rankId], argId++));
    CCU_CALL_RETURN(ccu::LoadArg(ctx.blockBytes, argId++));
    CCU_CALL_RETURN(ccu::LoadArg(ctx.rootSliceInPlace, argId++));
    return CCU_SUCCESS;
}

bool IsEightPlusFourWideRoot(const Rank12SmallRootContext &ctx)
{
    // The host-side RankGraph gate sets smallSingleReady for the exact {8, 4}
    // or 4x3 twelve-rank Small paths.  The explicit 4x3 bit excludes the
    // latter.  On the root rank, counting the ranks that share its RankGraph
    // local group identifies the root's actual side without assuming a root
    // rank ID or a CCU die ordering.
    if (ctx.arg->rankSize != 12U || ctx.arg->smallSingleReady == 0 ||
        ctx.arg->rank12FourByThree != 0) {
        return false;
    }
    const uint32_t rootGroup =
        ctx.arg->localGroupByRank[ctx.arg->rankId];
    uint32_t rootInstanceSize = 0;
    for (uint32_t rank = 0; rank < ctx.arg->rankSize; ++rank) {
        rootInstanceSize += ctx.arg->localGroupByRank[rank] == rootGroup ? 1U : 0U;
    }
    return rootInstanceSize == 8U;
}

CcuResult Rank12SmallRoot(Rank12SmallRootContext &ctx)
{
    // The receiver writes address before token on the same channel, so token
    // is the single ready edge.  Rank4 pipelines each READY into its payload.
    // On exact 4x3, only cross-instance peers do the same.  RankGraph can put
    // one local peer on the same die as the nine remote peers, so classify by
    // topology membership rather than assuming a fixed die number.
    const uint32_t readyBits = 1U << TOKEN_XN_ID;
    const uint32_t postBit = 1U << POST_SYNC_BIT;
    uint16_t completionMask = 0;
    uint16_t localCompletionMask = 0;
    if (ctx.arg->includeLocal) {
        const uint32_t peer = ctx.arg->rankId;
        ctx.source[peer].addr = ctx.sendBase;
        for (uint32_t offsetRank = 0; offsetRank < peer; ++offsetRank) {
            ctx.source[peer].addr += ctx.blockBytes;
        }
        ctx.source[peer].token = ctx.sendToken;
        ctx.localDestination.addr = ctx.recvBase[peer];
        ctx.localDestination.token = ctx.recvToken[peer];
        const uint16_t eventBit = static_cast<uint16_t>(1U << peer);
        completionMask = static_cast<uint16_t>(completionMask | eventBit);
        localCompletionMask = eventBit;
        CCU_IF(ctx.rootSliceInPlace == 0)
        {
            CCU_CALL_RETURN(ccu::LocalCopy(ctx.localDestination, ctx.source[peer],
                ctx.blockBytes, ctx.completion, eventBit));
        }
        CCU_IF(ctx.rootSliceInPlace != 0)
        {
            CCU_CALL_RETURN(ccu::EventRecord(ctx.completion, eventBit));
        }
    }

    const bool rank4PerPeer = ctx.arg->rankSize == 4U;
    const bool fourByThreePerRemotePeer =
        ctx.arg->rank12FourByThree != 0;
    const bool eightPlusFourPerRemotePeer = IsEightPlusFourWideRoot(ctx);
    uint16_t localPeerCompletionMask = localCompletionMask;

    if (rank4PerPeer) {
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->peerCount;
            ++channelIndex) {
            const uint32_t peer = ctx.arg->peerRanks[channelIndex];
            CCU_CALL_RETURN(ccu::NotifyWait(ctx.arg->channels[channelIndex],
                NOTIFY_INDEX, readyBits));
            ctx.source[peer].addr = ctx.sendBase;
            for (uint32_t offsetRank = 0; offsetRank < peer; ++offsetRank) {
                ctx.source[peer].addr += ctx.blockBytes;
            }
            ctx.source[peer].token = ctx.sendToken;
            ctx.remoteDestination[peer].addr = ctx.recvBase[peer];
            ctx.remoteDestination[peer].token = ctx.recvToken[peer];
            const uint16_t eventBit = static_cast<uint16_t>(1U << peer);
            completionMask = static_cast<uint16_t>(completionMask | eventBit);
            CCU_CALL_RETURN(ccu::Write(ctx.arg->channels[channelIndex],
                ctx.remoteDestination[peer], ctx.source[peer], ctx.blockBytes,
                ctx.completion, eventBit));
        }
    } else if (fourByThreePerRemotePeer || eightPlusFourPerRemotePeer) {
        // Submit every cross-instance Write through READY_i -> Write_i first.
        // No completion wait is allowed until all cross-instance Writes on
        // the responsible mission have been submitted.  The same loop is
        // used for 4x3 and 8+4; only the host-provided RankGraph grouping
        // decides which actual peers are remote.
        ctx.sourceCursor.addr = ctx.sendBase;
        ctx.sourceCursor.token = ctx.sendToken;
        uint32_t remoteCursorRank = 0;
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->peerCount;
            ++channelIndex) {
            const uint32_t peer = ctx.arg->peerRanks[channelIndex];
            if (ctx.arg->localGroupByRank[peer] ==
                ctx.arg->localGroupByRank[ctx.arg->rankId]) {
                continue;
            }
            CCU_CALL_RETURN(ccu::NotifyWait(ctx.arg->channels[channelIndex],
                NOTIFY_INDEX, readyBits));
            for (; remoteCursorRank < peer; ++remoteCursorRank) {
                ctx.sourceCursor.addr += ctx.blockBytes;
            }
            ctx.source[peer].addr = ctx.sourceCursor.addr;
            ctx.source[peer].token = ctx.sourceCursor.token;
            ctx.remoteDestination[peer].addr = ctx.recvBase[peer];
            ctx.remoteDestination[peer].token = ctx.recvToken[peer];
            const uint16_t eventBit = static_cast<uint16_t>(1U << peer);
            completionMask = static_cast<uint16_t>(completionMask | eventBit);
            CCU_CALL_RETURN(ccu::Write(ctx.arg->channels[channelIndex],
                ctx.remoteDestination[peer], ctx.source[peer], ctx.blockBytes,
                ctx.completion, eventBit));
        }

        // 4x3 keeps v25's local all-ready barrier.  The exact 8+4
        // root-on-8-side path instead pipelines each of its seven local
        // peers independently, still after this mission's remote Writes.
        if (fourByThreePerRemotePeer) {
            for (uint32_t channelIndex = 0; channelIndex < ctx.arg->peerCount;
                ++channelIndex) {
                const uint32_t peer = ctx.arg->peerRanks[channelIndex];
                if (ctx.arg->localGroupByRank[peer] !=
                    ctx.arg->localGroupByRank[ctx.arg->rankId]) {
                    continue;
                }
                CCU_CALL_RETURN(ccu::NotifyWait(ctx.arg->channels[channelIndex],
                    NOTIFY_INDEX, readyBits));
            }
        }
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->peerCount;
            ++channelIndex) {
            const uint32_t peer = ctx.arg->peerRanks[channelIndex];
            if (ctx.arg->localGroupByRank[peer] !=
                ctx.arg->localGroupByRank[ctx.arg->rankId]) {
                continue;
            }
            if (eightPlusFourPerRemotePeer) {
                CCU_CALL_RETURN(ccu::NotifyWait(ctx.arg->channels[channelIndex],
                    NOTIFY_INDEX, readyBits));
            }
            ctx.source[peer].addr = ctx.sendBase;
            for (uint32_t offsetRank = 0; offsetRank < peer; ++offsetRank) {
                ctx.source[peer].addr += ctx.blockBytes;
            }
            ctx.source[peer].token = ctx.sendToken;
            ctx.remoteDestination[peer].addr = ctx.recvBase[peer];
            ctx.remoteDestination[peer].token = ctx.recvToken[peer];
            const uint16_t eventBit = static_cast<uint16_t>(1U << peer);
            completionMask = static_cast<uint16_t>(completionMask | eventBit);
            localPeerCompletionMask = static_cast<uint16_t>(
                localPeerCompletionMask | eventBit);
            CCU_CALL_RETURN(ccu::Write(ctx.arg->channels[channelIndex],
                ctx.remoteDestination[peer], ctx.source[peer], ctx.blockBytes,
                ctx.completion, eventBit));
        }
    } else {
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            CCU_CALL_RETURN(ccu::NotifyWait(ctx.arg->channels[i], NOTIFY_INDEX,
                readyBits));
        }

        ctx.sourceCursor.addr = ctx.sendBase;
        ctx.sourceCursor.token = ctx.sendToken;
        uint32_t cursorRank = 0;
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->peerCount;
            ++channelIndex) {
            const uint32_t peer = ctx.arg->peerRanks[channelIndex];
            for (; cursorRank < peer; ++cursorRank) {
                ctx.sourceCursor.addr += ctx.blockBytes;
            }
            ctx.source[peer].addr = ctx.sourceCursor.addr;
            ctx.source[peer].token = ctx.sourceCursor.token;
            ctx.remoteDestination[peer].addr = ctx.recvBase[peer];
            ctx.remoteDestination[peer].token = ctx.recvToken[peer];
            const uint16_t eventBit = static_cast<uint16_t>(1U << peer);
            completionMask = static_cast<uint16_t>(completionMask | eventBit);
            CCU_CALL_RETURN(ccu::Write(ctx.arg->channels[channelIndex],
                ctx.remoteDestination[peer], ctx.source[peer], ctx.blockBytes,
                ctx.completion, eventBit));
        }
    }

    if (rank4PerPeer) {
        // All peer Writes are submitted before this completion phase.  Pair
        // each POST with its own payload event, then protect the root's local
        // copy before returning from the mission.
        for (uint32_t i = 0; i < ctx.arg->peerCount; ++i) {
            const uint32_t peer = ctx.arg->peerRanks[i];
            const uint16_t eventBit = static_cast<uint16_t>(1U << peer);
            CCU_CALL_RETURN(ccu::EventWait(ctx.completion, eventBit));
            CCU_CALL_RETURN(ccu::NotifyRecord(ctx.arg->channels[i],
                NOTIFY_INDEX, postBit));
        }
        if (localCompletionMask != 0) {
            CCU_CALL_RETURN(ccu::EventWait(ctx.completion,
                localCompletionMask));
        }
    } else if (fourByThreePerRemotePeer) {
        for (uint32_t i = 0; i < ctx.arg->peerCount; ++i) {
            const uint32_t peer = ctx.arg->peerRanks[i];
            if (ctx.arg->localGroupByRank[peer] ==
                ctx.arg->localGroupByRank[ctx.arg->rankId]) {
                continue;
            }
            const uint16_t eventBit = static_cast<uint16_t>(1U << peer);
            CCU_CALL_RETURN(ccu::EventWait(ctx.completion, eventBit));
            CCU_CALL_RETURN(ccu::NotifyRecord(ctx.arg->channels[i],
                NOTIFY_INDEX, postBit));
        }
        if (localPeerCompletionMask != 0) {
            CCU_CALL_RETURN(ccu::EventWait(ctx.completion,
                localPeerCompletionMask));
        }
        for (uint32_t i = 0; i < ctx.arg->peerCount; ++i) {
            const uint32_t peer = ctx.arg->peerRanks[i];
            if (ctx.arg->localGroupByRank[peer] !=
                ctx.arg->localGroupByRank[ctx.arg->rankId]) {
                continue;
            }
            CCU_CALL_RETURN(ccu::NotifyRecord(ctx.arg->channels[i],
                NOTIFY_INDEX, postBit));
        }
    } else {
        CCU_CALL_RETURN(ccu::EventWait(ctx.completion, completionMask));
        for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
            CCU_CALL_RETURN(ccu::NotifyRecord(ctx.arg->channels[i],
                NOTIFY_INDEX, postBit));
        }
    }
    return CCU_SUCCESS;
}

CcuResult PullRoot(ScatterContext &ctx)
{
    const uint32_t addressBit = 1U << OUTPUT_XN_ID;
    const uint32_t tokenBit = 1U << TOKEN_XN_ID;
    const uint32_t doneBit = 1U << POST_SYNC_BIT;
    uint16_t localCompletionMask = 0;

    // Preserve v12's useful overlap: start the root slice before publishing
    // source metadata to receivers. The final event wait still protects it.
    if (ctx.arg->includeLocal) {
        const uint32_t peer = ctx.arg->rankId;
        ctx.source[peer].addr = ctx.sendBase;
        for (uint32_t offsetRank = 0; offsetRank < peer; ++offsetRank) {
            ctx.source[peer].addr += ctx.blockBytes;
        }
        ctx.source[peer].addr += ctx.chunkOffset;
        ctx.source[peer].token = ctx.sendToken;
        ctx.localDestination.addr = ctx.recvBase[peer];
        ctx.localDestination.addr += ctx.chunkOffset;
        ctx.localDestination.token = ctx.recvToken[peer];
        const uint16_t eventBit = static_cast<uint16_t>(1U << peer);
        localCompletionMask = eventBit;
        CCU_IF(ctx.rootSliceInPlace == 0)
        {
            CCU_CALL_RETURN(ccu::LocalCopy(ctx.localDestination, ctx.source[peer],
                ctx.chunkBytes, ctx.completion, eventBit));
        }
        CCU_IF(ctx.rootSliceInPlace != 0)
        {
            CCU_CALL_RETURN(ccu::EventRecord(ctx.completion, eventBit));
        }
    }

    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->peerCount; ++channelIndex) {
        const uint32_t peer = ctx.arg->peerRanks[channelIndex];
        ctx.sourceAddress[peer] = ctx.sendBase;
        for (uint32_t offsetRank = 0; offsetRank < peer; ++offsetRank) {
            ctx.sourceAddress[peer] += ctx.blockBytes;
        }
        ctx.sourceAddress[peer] += ctx.chunkOffset;
        CCU_IF(ctx.metadataReuse == 0)
        {
            CCU_CALL_RETURN(ccu::WriteVariableWithNotify(
                ctx.arg->channels[channelIndex], ctx.sourceAddress[peer],
                OUTPUT_XN_ID, NOTIFY_INDEX, addressBit));
            CCU_CALL_RETURN(ccu::WriteVariableWithNotify(
                ctx.arg->channels[channelIndex], ctx.sendToken,
                TOKEN_XN_ID, NOTIFY_INDEX, tokenBit));
        }
        CCU_IF(ctx.metadataReuse != 0)
        {
            CCU_CALL_RETURN(ccu::NotifyRecord(ctx.arg->channels[channelIndex],
                NOTIFY_INDEX, addressBit | tokenBit));
        }
    }

    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
        CCU_CALL_RETURN(ccu::NotifyWait(ctx.arg->channels[channelIndex],
            NOTIFY_INDEX, doneBit));
    }
    if (localCompletionMask != 0) {
        CCU_CALL_RETURN(ccu::EventWait(ctx.completion, localCompletionMask));
    }
    return CCU_SUCCESS;
}

CcuResult PullReceiver(ScatterContext &ctx)
{
    const uint32_t metadataBits = (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
    const uint32_t doneBit = 1U << POST_SYNC_BIT;
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->peerCount; ++channelIndex) {
        const uint32_t peer = ctx.arg->peerRanks[channelIndex];
        CCU_IF(ctx.root == peer)
        {
            CCU_CALL_RETURN(ccu::NotifyWait(ctx.arg->channels[channelIndex],
                NOTIFY_INDEX, metadataBits));
            ctx.remoteDestination[peer].addr = ctx.recvBase[peer];
            ctx.remoteDestination[peer].token = ctx.recvToken[peer];
            ctx.localDestination.addr = ctx.recvBase[ctx.arg->rankId];
            ctx.localDestination.addr += ctx.chunkOffset;
            ctx.localDestination.token = ctx.recvToken[ctx.arg->rankId];
            CCU_CALL_RETURN(ccu::Read(ctx.arg->channels[channelIndex],
                ctx.localDestination, ctx.remoteDestination[peer], ctx.chunkBytes,
                ctx.completion, 1));
            CCU_CALL_RETURN(ccu::EventWait(ctx.completion, 1));
            CCU_CALL_RETURN(ccu::NotifyRecord(ctx.arg->channels[channelIndex],
                NOTIFY_INDEX, doneBit));
        }
    }
    return CCU_SUCCESS;
}

struct StaticHelperRoute {
    uint32_t helperTarget{INVALID_VALUE_RANKID};
    uint32_t senderRank{INVALID_VALUE_RANKID};
    bool enabled{false};
};

StaticHelperRoute GetStaticHelperRoute(const CcuKernelArgScatter &arg,
    uint32_t rootRank, uint32_t rank)
{
    StaticHelperRoute route;
    route.senderRank = rootRank;
    const uint32_t rootGroup = arg.localGroupByRank[rootRank];
    std::vector<uint32_t> helpers;
    std::vector<uint32_t> remoteTargets;
    for (uint32_t candidate = 0; candidate < arg.rankSize; ++candidate) {
        if (candidate == rootRank) {
            continue;
        }
        if (arg.localGroupByRank[candidate] == rootGroup) {
            helpers.push_back(candidate);
        } else {
            remoteTargets.push_back(candidate);
        }
    }
    if (helpers.empty() || remoteTargets.size() <= CLOS_CAPACITY ||
        helpers.size() > remoteTargets.size()) {
        return route;
    }
    route.enabled = true;
    for (uint32_t i = 0; i < helpers.size(); ++i) {
        if (rank == helpers[i]) {
            route.helperTarget = remoteTargets[i];
        }
        if (rank == remoteTargets[i]) {
            route.senderRank = helpers[i];
        }
    }
    return route;
}

bool HasWideRootInstance(const CcuKernelArgScatter &arg)
{
    // Static registration-time filter. The host independently selects phase 4
    // only for the exact {8, 4} RankGraph topology.
    return arg.rankSize == 12U &&
        (arg.peerCount == 3U || arg.peerCount == 4U ||
         arg.peerCount == 7U || arg.peerCount == 8U);
}

bool IsWideRootRank(const CcuKernelArgScatter &arg)
{
    uint32_t instanceSize = 0;
    for (uint32_t rank = 0; rank < arg.rankSize; ++rank) {
        instanceSize += arg.localGroupByRank[rank] ==
            arg.localGroupByRank[arg.rankId] ? 1U : 0U;
    }
    const uint32_t localPeers = instanceSize == 0 ? 0 : instanceSize - 1U;
    const uint32_t remotePeers = arg.rankSize - instanceSize;
    return remotePeers != 0 && localPeers > remotePeers;
}

CcuResult InitHelperResources(HelperContext &ctx)
{
    ctx.recvChunkBase.resize(ctx.arg->rankSize);
    ctx.recvToken.resize(ctx.arg->rankSize);
    ctx.stageBase.resize(ctx.arg->rankSize);
    ctx.stageToken.resize(ctx.arg->rankSize);
    ctx.source.resize(ctx.arg->rankSize);
    ctx.remoteDestination.resize(ctx.arg->rankSize);
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->peerCount; ++channelIndex) {
        const uint32_t peer = ctx.arg->peerRanks[channelIndex];
        ctx.recvChunkBase[peer] = ccu::GetResByChannel<ccu::Variable>(
            ctx.arg->channels[channelIndex], OUTPUT_XN_ID);
        ctx.recvToken[peer] = ccu::GetResByChannel<ccu::Variable>(
            ctx.arg->channels[channelIndex], TOKEN_XN_ID);
        ctx.stageBase[peer] = ccu::GetResByChannel<ccu::Variable>(
            ctx.arg->channels[channelIndex], STAGE_XN_ID);
        ctx.stageToken[peer] = ccu::GetResByChannel<ccu::Variable>(
            ctx.arg->channels[channelIndex], STAGE_TOKEN_XN_ID);
    }
    return CCU_SUCCESS;
}

CcuResult LoadHelperTaskArgs(HelperContext &ctx)
{
    uint32_t argId = 0;
    CCU_CALL_RETURN(ccu::LoadArg(ctx.sendChunkBase, argId++));
    CCU_CALL_RETURN(ccu::LoadArg(ctx.recvChunkBase[ctx.arg->rankId], argId++));
    CCU_CALL_RETURN(ccu::LoadArg(ctx.sendToken, argId++));
    CCU_CALL_RETURN(ccu::LoadArg(ctx.recvToken[ctx.arg->rankId], argId++));
    CCU_CALL_RETURN(ccu::LoadArg(ctx.stageBase[ctx.arg->rankId], argId++));
    CCU_CALL_RETURN(ccu::LoadArg(ctx.stageToken[ctx.arg->rankId], argId++));
    CCU_CALL_RETURN(ccu::LoadArg(ctx.blockBytes, argId++));
    CCU_CALL_RETURN(ccu::LoadArg(ctx.directBytes, argId++));
    CCU_CALL_RETURN(ccu::LoadArg(ctx.root, argId++));
    CCU_CALL_RETURN(ccu::LoadArg(ctx.rootSliceInPlace, argId++));
    CCU_CALL_RETURN(ccu::LoadArg(ctx.delegatedBytes, argId++));
    CCU_CALL_RETURN(ccu::LoadArg(ctx.phase, argId++));
    ctx.fullBytes = ctx.directBytes + ctx.delegatedBytes;
    return CCU_SUCCESS;
}

bool FindChannel(const CcuKernelArgScatter &arg, uint32_t peer, uint32_t &channelIndex)
{
    for (uint32_t i = 0; i < arg.peerCount; ++i) {
        if (arg.peerRanks[i] == peer) {
            channelIndex = i;
            return true;
        }
    }
    return false;
}

CcuResult PublishOutput(HelperContext &ctx, uint32_t peer)
{
    uint32_t channelIndex = 0;
    if (!FindChannel(*ctx.arg, peer, channelIndex)) {
        return CCU_SUCCESS;
    }
    const uint32_t addressBit = 1U << OUTPUT_XN_ID;
    const uint32_t tokenBit = 1U << TOKEN_XN_ID;
    CCU_CALL_RETURN(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIndex],
        ctx.recvChunkBase[ctx.arg->rankId], OUTPUT_XN_ID, NOTIFY_INDEX, addressBit));
    CCU_CALL_RETURN(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIndex],
        ctx.recvToken[ctx.arg->rankId], TOKEN_XN_ID, NOTIFY_INDEX, tokenBit));
    return CCU_SUCCESS;
}

CcuResult PublishStage(HelperContext &ctx, uint32_t peer)
{
    uint32_t channelIndex = 0;
    if (!FindChannel(*ctx.arg, peer, channelIndex)) {
        return CCU_SUCCESS;
    }
    CCU_CALL_RETURN(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIndex],
        ctx.stageBase[ctx.arg->rankId], STAGE_XN_ID, NOTIFY_INDEX,
        1U << STAGE_XN_ID));
    CCU_CALL_RETURN(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIndex],
        ctx.stageToken[ctx.arg->rankId], STAGE_TOKEN_XN_ID, NOTIFY_INDEX,
        1U << STAGE_TOKEN_XN_ID));
    return CCU_SUCCESS;
}

CcuResult WaitPeerNotify(HelperContext &ctx, uint32_t peer, uint32_t bit)
{
    uint32_t channelIndex = 0;
    if (FindChannel(*ctx.arg, peer, channelIndex)) {
        CCU_CALL_RETURN(ccu::NotifyWait(ctx.arg->channels[channelIndex],
            NOTIFY_INDEX, 1U << bit));
    }
    return CCU_SUCCESS;
}

CcuResult RecordPeerNotify(HelperContext &ctx, uint32_t peer, uint32_t bit)
{
    uint32_t channelIndex = 0;
    if (FindChannel(*ctx.arg, peer, channelIndex)) {
        CCU_CALL_RETURN(ccu::NotifyRecord(ctx.arg->channels[channelIndex],
            NOTIFY_INDEX, 1U << bit));
    }
    return CCU_SUCCESS;
}

CcuResult RootHelperScatter(HelperContext &ctx)
{
    const uint32_t addressTokenBits = (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
    uint16_t completionMask = 0;
    uint16_t stageMask = 0;
    uint16_t helperMask = 0;

    // Start the root's own slice before the receiver metadata handshake.  Its
    // event remains part of completionMask, so user-stream completion semantics
    // are unchanged while the local DMA overlaps the ready round trip.
    if (ctx.arg->includeLocal) {
        const uint32_t peer = ctx.arg->rankId;
        ctx.source[peer].addr = ctx.sendChunkBase;
        for (uint32_t offsetRank = 0; offsetRank < peer; ++offsetRank) {
            ctx.source[peer].addr += ctx.blockBytes;
        }
        ctx.source[peer].token = ctx.sendToken;
        ctx.localDestination.addr = ctx.recvChunkBase[peer];
        ctx.localDestination.token = ctx.recvToken[peer];
        const uint16_t eventBit = static_cast<uint16_t>(1U << peer);
        completionMask = static_cast<uint16_t>(completionMask | eventBit);
        CCU_IF(ctx.rootSliceInPlace == 0)
        {
            CCU_CALL_RETURN(ccu::LocalCopy(ctx.localDestination, ctx.source[peer],
                ctx.fullBytes, ctx.completion, eventBit));
        }
        CCU_IF(ctx.rootSliceInPlace != 0)
        {
            CCU_CALL_RETURN(ccu::EventRecord(ctx.completion, eventBit));
        }
    }

    // Publish every remote target's selected sender before waiting for any
    // target READY.  A target needs this value before it can expose its output
    // address/token to the selected helper, so delaying a later publication
    // behind an earlier target would also delay Helper Forward.
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        const uint32_t peer = ctx.arg->peerRanks[i];
        const StaticHelperRoute peerRoute = GetStaticHelperRoute(*ctx.arg,
            ctx.arg->rankId, peer);
        if (ctx.arg->localGroupByRank[peer] !=
            ctx.arg->localGroupByRank[ctx.arg->rankId]) {
            // XN2 carries the staging address on a root-helper L0 channel and
            // the selected sender rank on a root-target L1 channel.  Those
            // channel roles are mutually exclusive, so this does not request
            // an additional per-peer resource.
            ctx.stageBase[peer] = peerRoute.senderRank;
            CCU_CALL_RETURN(ccu::WriteVariableWithNotify(ctx.arg->channels[i],
                ctx.stageBase[peer], STAGE_XN_ID, NOTIFY_INDEX,
                1U << HELPER_SENDER_READY_BIT));
        }
    }

    // Phase 1 on the local Helper die: consume one helper's complete READY
    // set and immediately submit its staging Write.  The Stage READY includes
    // both output metadata and the staging address/token required by this DMA.
    // The loop submits every staging Write before Phase 2 can issue an
    // ordinary output Write on this mission.
    for (uint32_t i = 0; i < ctx.arg->peerCount; ++i) {
        const uint32_t peer = ctx.arg->peerRanks[i];
        if (ctx.arg->localGroupByRank[peer] !=
            ctx.arg->localGroupByRank[ctx.arg->rankId]) {
            continue;
        }
        const StaticHelperRoute peerRoute = GetStaticHelperRoute(*ctx.arg,
            ctx.arg->rankId, peer);
        uint32_t readyBits = addressTokenBits;
        if (peerRoute.helperTarget != INVALID_VALUE_RANKID) {
            readyBits |= (1U << STAGE_XN_ID) | (1U << STAGE_TOKEN_XN_ID);
        }
        CCU_CALL_RETURN(ccu::NotifyWait(ctx.arg->channels[i], NOTIFY_INDEX,
            readyBits));
        if (peerRoute.helperTarget != INVALID_VALUE_RANKID) {
            const uint16_t eventBit = static_cast<uint16_t>(1U << peer);
            ctx.source[peer].addr = ctx.sendChunkBase;
            for (uint32_t targetOffset = 0;
                targetOffset < peerRoute.helperTarget; ++targetOffset) {
                ctx.source[peer].addr += ctx.blockBytes;
            }
            ctx.source[peer].token = ctx.sendToken;
            ctx.remoteDestination[peer].addr = ctx.stageBase[peer];
            ctx.remoteDestination[peer].token = ctx.stageToken[peer];
            stageMask = static_cast<uint16_t>(stageMask | eventBit);
            CCU_CALL_RETURN(ccu::Write(ctx.arg->channels[i], ctx.remoteDestination[peer],
                ctx.source[peer], ctx.delegatedBytes, ctx.stageCompletion, eventBit));
        }
    }

    // Phase 2: local ordinary output Writes remain behind submission of every
    // Stage Write.  On the remote Target die, all sender-rank publications
    // above happen first, then each READY immediately releases that target's
    // Direct Write.  The two dies execute independent mission queues.
    ctx.sourceCursor.addr = ctx.sendChunkBase;
    ctx.sourceCursor.token = ctx.sendToken;
    uint32_t cursorRank = 0;
    for (uint32_t i = 0; i < ctx.arg->peerCount; ++i) {
        const uint32_t peer = ctx.arg->peerRanks[i];
        for (; cursorRank < peer; ++cursorRank) {
            ctx.sourceCursor.addr += ctx.blockBytes;
        }
        ctx.source[peer].addr = ctx.sourceCursor.addr;
        ctx.source[peer].token = ctx.sourceCursor.token;
        const StaticHelperRoute peerRoute = GetStaticHelperRoute(*ctx.arg,
            ctx.arg->rankId, peer);
        const uint16_t eventBit = static_cast<uint16_t>(1U << peer);
        const bool isLocalPeer = ctx.arg->localGroupByRank[peer] ==
            ctx.arg->localGroupByRank[ctx.arg->rankId];
        if (!isLocalPeer) {
            CCU_CALL_RETURN(ccu::NotifyWait(ctx.arg->channels[i], NOTIFY_INDEX,
                addressTokenBits));
            // Materialize the remotely published address/token only after its
            // READY edge.  Loading these XNs before NotifyWait can consume the
            // previous value while the current target is still publishing it.
            ctx.remoteDestination[peer].addr = ctx.recvChunkBase[peer];
            ctx.remoteDestination[peer].token = ctx.recvToken[peer];
            completionMask = static_cast<uint16_t>(completionMask | eventBit);
            if (peerRoute.senderRank != ctx.arg->rankId) {
                ctx.source[peer].addr += ctx.delegatedBytes;
                ctx.remoteDestination[peer].addr += ctx.delegatedBytes;
                CCU_CALL_RETURN(ccu::Write(ctx.arg->channels[i],
                    ctx.remoteDestination[peer], ctx.source[peer],
                    ctx.directBytes, ctx.completion, eventBit));
            } else {
                CCU_CALL_RETURN(ccu::Write(ctx.arg->channels[i],
                    ctx.remoteDestination[peer], ctx.source[peer],
                    ctx.fullBytes, ctx.completion, eventBit));
            }
        } else if (peerRoute.helperTarget != INVALID_VALUE_RANKID) {
            ctx.remoteDestination[peer].addr = ctx.recvChunkBase[peer];
            ctx.remoteDestination[peer].token = ctx.recvToken[peer];
            // Deferred until all delegated prefixes are visible and their
            // STAGE_READY notifications have been posted below.
        } else {
            ctx.remoteDestination[peer].addr = ctx.recvChunkBase[peer];
            ctx.remoteDestination[peer].token = ctx.recvToken[peer];
            completionMask = static_cast<uint16_t>(completionMask | eventBit);
            CCU_CALL_RETURN(ccu::Write(ctx.arg->channels[i],
                ctx.remoteDestination[peer], ctx.source[peer], ctx.fullBytes,
                ctx.completion, eventBit));
        }

    }

    if (stageMask != 0) {
        // All delegated writes were issued above.  Consume their completion
        // bits independently so a ready helper can start forwarding without
        // waiting for every other L0 channel to complete its staged prefix.
        for (uint32_t i = 0; i < ctx.arg->peerCount; ++i) {
            const uint32_t peer = ctx.arg->peerRanks[i];
            const StaticHelperRoute peerRoute = GetStaticHelperRoute(*ctx.arg,
                ctx.arg->rankId, peer);
            if (peerRoute.helperTarget != INVALID_VALUE_RANKID) {
                const uint16_t eventBit = static_cast<uint16_t>(1U << peer);
                CCU_CALL_RETURN(ccu::EventWait(ctx.stageCompletion, eventBit));
                CCU_CALL_RETURN(ccu::NotifyRecord(ctx.arg->channels[i], NOTIFY_INDEX,
                    1U << HELPER_STAGE_READY_BIT));
            }
        }

        // The ready notification is now ahead of every helper-output Write in
        // this instruction queue.  Completion remains separately tracked: a
        // helper cannot finish its user stream until HELPER_OUTPUT_READY.
        for (uint32_t i = 0; i < ctx.arg->peerCount; ++i) {
            const uint32_t peer = ctx.arg->peerRanks[i];
            const StaticHelperRoute peerRoute = GetStaticHelperRoute(*ctx.arg,
                ctx.arg->rankId, peer);
            if (peerRoute.helperTarget != INVALID_VALUE_RANKID) {
                const uint16_t eventBit = static_cast<uint16_t>(1U << peer);
                helperMask = static_cast<uint16_t>(helperMask | eventBit);
                CCU_CALL_RETURN(ccu::Write(ctx.arg->channels[i],
                    ctx.remoteDestination[peer], ctx.source[peer], ctx.fullBytes,
                    ctx.helperCompletion, eventBit));
            }
        }
    }
    if (helperMask != 0) {
        CCU_CALL_RETURN(ccu::EventWait(ctx.helperCompletion, helperMask));
        for (uint32_t i = 0; i < ctx.arg->peerCount; ++i) {
            const uint32_t peer = ctx.arg->peerRanks[i];
            const StaticHelperRoute peerRoute = GetStaticHelperRoute(*ctx.arg,
                ctx.arg->rankId, peer);
            if (peerRoute.helperTarget != INVALID_VALUE_RANKID) {
                CCU_CALL_RETURN(ccu::NotifyRecord(ctx.arg->channels[i], NOTIFY_INDEX,
                    1U << HELPER_OUTPUT_READY_BIT));
            }
        }
    }
    if (completionMask != 0) {
        CCU_CALL_RETURN(ccu::EventWait(ctx.completion, completionMask));
    }
    for (uint32_t i = 0; i < ctx.arg->peerCount; ++i) {
        const uint32_t peer = ctx.arg->peerRanks[i];
        const StaticHelperRoute peerRoute = GetStaticHelperRoute(*ctx.arg,
            ctx.arg->rankId, peer);
        if (peerRoute.senderRank != ctx.arg->rankId) {
            CCU_CALL_RETURN(ccu::NotifyWait(ctx.arg->channels[i], NOTIFY_INDEX,
                1U << HELPER_TARGET_ACK_BIT));
        }
    }
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        const uint32_t peer = ctx.arg->peerRanks[i];
        const StaticHelperRoute peerRoute = GetStaticHelperRoute(*ctx.arg,
            ctx.arg->rankId, peer);
        if (peerRoute.helperTarget == INVALID_VALUE_RANKID) {
            CCU_CALL_RETURN(ccu::NotifyRecord(ctx.arg->channels[i], NOTIFY_INDEX,
                1U << HELPER_ROOT_POST_BIT));
        }
    }
    return CCU_SUCCESS;
}

CcuResult RootWideDirect(HelperContext &ctx)
{
    const uint32_t addressTokenBits =
        (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
    uint16_t completionMask = 0;

    // v18 Push protocol: consume every receiver's metadata before issuing any
    // payload Write.  The dedicated v24 mission below only removes the
    // generic Helper image; it deliberately does not change this dependency.
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        CCU_CALL_RETURN(ccu::NotifyWait(ctx.arg->channels[i], NOTIFY_INDEX,
            addressTokenBits));
    }

    ctx.sourceCursor.addr = ctx.sendChunkBase;
    ctx.sourceCursor.token = ctx.sendToken;
    uint32_t cursorRank = 0;
    for (uint32_t i = 0; i < ctx.arg->peerCount; ++i) {
        const uint32_t peer = ctx.arg->peerRanks[i];
        for (; cursorRank < peer; ++cursorRank) {
            ctx.sourceCursor.addr += ctx.blockBytes;
        }
        ctx.source[peer].addr = ctx.sourceCursor.addr;
        ctx.source[peer].token = ctx.sourceCursor.token;
        ctx.remoteDestination[peer].addr = ctx.recvChunkBase[peer];
        ctx.remoteDestination[peer].token = ctx.recvToken[peer];
        const uint16_t eventBit = static_cast<uint16_t>(1U << peer);
        completionMask = static_cast<uint16_t>(completionMask | eventBit);
        CCU_CALL_RETURN(ccu::Write(ctx.arg->channels[i],
            ctx.remoteDestination[peer], ctx.source[peer], ctx.fullBytes,
            ctx.completion, eventBit));
    }

    if (ctx.arg->includeLocal) {
        const uint32_t peer = ctx.arg->rankId;
        ctx.source[peer].addr = ctx.sendChunkBase;
        for (uint32_t offsetRank = 0; offsetRank < peer; ++offsetRank) {
            ctx.source[peer].addr += ctx.blockBytes;
        }
        ctx.source[peer].token = ctx.sendToken;
        ctx.localDestination.addr = ctx.recvChunkBase[peer];
        ctx.localDestination.token = ctx.recvToken[peer];
        const uint16_t eventBit = static_cast<uint16_t>(1U << peer);
        completionMask = static_cast<uint16_t>(completionMask | eventBit);
        CCU_IF(ctx.rootSliceInPlace == 0)
        {
            CCU_CALL_RETURN(ccu::LocalCopy(ctx.localDestination, ctx.source[peer],
                ctx.fullBytes, ctx.completion, eventBit));
        }
        CCU_IF(ctx.rootSliceInPlace != 0)
        {
            CCU_CALL_RETURN(ccu::EventRecord(ctx.completion, eventBit));
        }
    }

    if (completionMask != 0) {
        CCU_CALL_RETURN(ccu::EventWait(ctx.completion, completionMask));
    }
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        CCU_CALL_RETURN(ccu::NotifyRecord(ctx.arg->channels[i], NOTIFY_INDEX,
            1U << HELPER_ROOT_POST_BIT));
    }
    return CCU_SUCCESS;
}

CcuResult WideDirectReceiver(HelperContext &ctx, uint32_t rootRank)
{
    CCU_CALL_RETURN(PublishOutput(ctx, rootRank));
    CCU_CALL_RETURN(WaitPeerNotify(ctx, rootRank, HELPER_ROOT_POST_BIT));
    return CCU_SUCCESS;
}

CcuResult InitWidePushResources(WidePushContext &ctx)
{
    ctx.recvChunkBase.resize(ctx.arg->rankSize);
    ctx.recvToken.resize(ctx.arg->rankSize);
    ctx.source.resize(ctx.arg->rankSize);
    ctx.remoteDestination.resize(ctx.arg->rankSize);
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->peerCount;
        ++channelIndex) {
        const uint32_t peer = ctx.arg->peerRanks[channelIndex];
        ctx.recvChunkBase[peer] = ccu::GetResByChannel<ccu::Variable>(
            ctx.arg->channels[channelIndex], OUTPUT_XN_ID);
        ctx.recvToken[peer] = ccu::GetResByChannel<ccu::Variable>(
            ctx.arg->channels[channelIndex], TOKEN_XN_ID);
    }
    return CCU_SUCCESS;
}

CcuResult LoadWidePushTaskArgs(WidePushContext &ctx)
{
    uint32_t argId = 0;
    CCU_CALL_RETURN(ccu::LoadArg(ctx.sendChunkBase, argId++));
    CCU_CALL_RETURN(ccu::LoadArg(ctx.recvChunkBase[ctx.arg->rankId], argId++));
    CCU_CALL_RETURN(ccu::LoadArg(ctx.sendToken, argId++));
    CCU_CALL_RETURN(ccu::LoadArg(ctx.recvToken[ctx.arg->rankId], argId++));
    CCU_CALL_RETURN(ccu::LoadArg(ctx.blockBytes, argId++));
    CCU_CALL_RETURN(ccu::LoadArg(ctx.fullBytes, argId++));
    CCU_CALL_RETURN(ccu::LoadArg(ctx.root, argId++));
    CCU_CALL_RETURN(ccu::LoadArg(ctx.rootSliceInPlace, argId++));
    return CCU_SUCCESS;
}

CcuResult DedicatedWidePushRoot(WidePushContext &ctx)
{
    const uint32_t addressTokenBits =
        (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
    uint16_t completionMask = 0;

    // The four cross-instance channels are the bandwidth-critical leg for a
    // root in the 8-rank instance.  Start each remote payload as soon as that
    // receiver publishes its metadata instead of holding all four Writes
    // behind the slowest READY.  Completion and POST remain aggregate below.
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->peerCount;
        ++channelIndex) {
        const uint32_t peer = ctx.arg->peerRanks[channelIndex];
        if (ctx.arg->localGroupByRank[peer] ==
            ctx.arg->localGroupByRank[ctx.arg->rankId]) {
            continue;
        }
        CCU_CALL_RETURN(ccu::NotifyWait(ctx.arg->channels[channelIndex],
            NOTIFY_INDEX, addressTokenBits));
        ctx.source[peer].addr = ctx.sendChunkBase;
        for (uint32_t offsetRank = 0; offsetRank < peer; ++offsetRank) {
            ctx.source[peer].addr += ctx.blockBytes;
        }
        ctx.source[peer].token = ctx.sendToken;
        ctx.remoteDestination[peer].addr = ctx.recvChunkBase[peer];
        ctx.remoteDestination[peer].token = ctx.recvToken[peer];
        const uint16_t eventBit = static_cast<uint16_t>(1U << peer);
        completionMask = static_cast<uint16_t>(completionMask | eventBit);
        CCU_CALL_RETURN(ccu::Write(ctx.arg->channels[channelIndex],
            ctx.remoteDestination[peer], ctx.source[peer], ctx.fullBytes,
            ctx.completion, eventBit));
    }

    // Preserve the v24 all-ready schedule for peers in the root's local
    // instance; this experiment changes only the critical cross-instance leg.
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->peerCount;
        ++channelIndex) {
        const uint32_t peer = ctx.arg->peerRanks[channelIndex];
        if (ctx.arg->localGroupByRank[peer] !=
            ctx.arg->localGroupByRank[ctx.arg->rankId]) {
            continue;
        }
        CCU_CALL_RETURN(ccu::NotifyWait(ctx.arg->channels[channelIndex],
            NOTIFY_INDEX, addressTokenBits));
    }
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->peerCount;
        ++channelIndex) {
        const uint32_t peer = ctx.arg->peerRanks[channelIndex];
        if (ctx.arg->localGroupByRank[peer] !=
            ctx.arg->localGroupByRank[ctx.arg->rankId]) {
            continue;
        }
        ctx.source[peer].addr = ctx.sendChunkBase;
        for (uint32_t offsetRank = 0; offsetRank < peer; ++offsetRank) {
            ctx.source[peer].addr += ctx.blockBytes;
        }
        ctx.source[peer].token = ctx.sendToken;
        ctx.remoteDestination[peer].addr = ctx.recvChunkBase[peer];
        ctx.remoteDestination[peer].token = ctx.recvToken[peer];
        const uint16_t eventBit = static_cast<uint16_t>(1U << peer);
        completionMask = static_cast<uint16_t>(completionMask | eventBit);
        CCU_CALL_RETURN(ccu::Write(ctx.arg->channels[channelIndex],
            ctx.remoteDestination[peer], ctx.source[peer], ctx.fullBytes,
            ctx.completion, eventBit));
    }

    if (ctx.arg->includeLocal) {
        const uint32_t peer = ctx.arg->rankId;
        ctx.source[peer].addr = ctx.sendChunkBase;
        for (uint32_t offsetRank = 0; offsetRank < peer; ++offsetRank) {
            ctx.source[peer].addr += ctx.blockBytes;
        }
        ctx.source[peer].token = ctx.sendToken;
        ctx.localDestination.addr = ctx.recvChunkBase[peer];
        ctx.localDestination.token = ctx.recvToken[peer];
        const uint16_t eventBit = static_cast<uint16_t>(1U << peer);
        completionMask = static_cast<uint16_t>(completionMask | eventBit);
        CCU_IF(ctx.rootSliceInPlace == 0)
        {
            CCU_CALL_RETURN(ccu::LocalCopy(ctx.localDestination, ctx.source[peer],
                ctx.fullBytes, ctx.completion, eventBit));
        }
        CCU_IF(ctx.rootSliceInPlace != 0)
        {
            CCU_CALL_RETURN(ccu::EventRecord(ctx.completion, eventBit));
        }
    }

    if (completionMask != 0) {
        CCU_CALL_RETURN(ccu::EventWait(ctx.completion, completionMask));
    }
    for (uint32_t i = 0; i < ctx.arg->channelCount; ++i) {
        CCU_CALL_RETURN(ccu::NotifyRecord(ctx.arg->channels[i], NOTIFY_INDEX,
            1U << HELPER_ROOT_POST_BIT));
    }
    return CCU_SUCCESS;
}

CcuResult DedicatedWidePushReceiver(WidePushContext &ctx, uint32_t rootRank)
{
    uint32_t channelIndex = 0;
    if (!FindChannel(*ctx.arg, rootRank, channelIndex)) {
        return CCU_SUCCESS;
    }
    const uint32_t addressBit = 1U << OUTPUT_XN_ID;
    const uint32_t tokenBit = 1U << TOKEN_XN_ID;
    CCU_CALL_RETURN(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIndex],
        ctx.recvChunkBase[ctx.arg->rankId], OUTPUT_XN_ID, NOTIFY_INDEX, addressBit));
    CCU_CALL_RETURN(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIndex],
        ctx.recvToken[ctx.arg->rankId], TOKEN_XN_ID, NOTIFY_INDEX, tokenBit));
    CCU_CALL_RETURN(ccu::NotifyWait(ctx.arg->channels[channelIndex], NOTIFY_INDEX,
        1U << HELPER_ROOT_POST_BIT));
    return CCU_SUCCESS;
}

CcuResult HelperStage(HelperContext &ctx, uint32_t rootRank)
{
    CCU_CALL_RETURN(PublishOutput(ctx, rootRank));
    CCU_CALL_RETURN(PublishStage(ctx, rootRank));
    CCU_CALL_RETURN(WaitPeerNotify(ctx, rootRank, HELPER_STAGE_READY_BIT));
    return CCU_SUCCESS;
}

CcuResult HelperForward(HelperContext &ctx, uint32_t targetRank)
{
    uint32_t channelIndex = 0;
    if (!FindChannel(*ctx.arg, targetRank, channelIndex)) {
        return CCU_SUCCESS;
    }
    const uint32_t addressTokenBits = (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
    CCU_CALL_RETURN(ccu::NotifyWait(ctx.arg->channels[channelIndex], NOTIFY_INDEX,
        addressTokenBits));
    ctx.localStage.addr = ctx.stageBase[ctx.arg->rankId];
    ctx.localStage.token = ctx.stageToken[ctx.arg->rankId];
    ctx.remoteDestination[targetRank].addr = ctx.recvChunkBase[targetRank];
    ctx.remoteDestination[targetRank].token = ctx.recvToken[targetRank];
    CCU_CALL_RETURN(ccu::Write(ctx.arg->channels[channelIndex],
        ctx.remoteDestination[targetRank], ctx.localStage, ctx.delegatedBytes,
        ctx.completion, 1));
    CCU_CALL_RETURN(ccu::EventWait(ctx.completion, 1));
    CCU_CALL_RETURN(ccu::NotifyRecord(ctx.arg->channels[channelIndex], NOTIFY_INDEX,
        1U << HELPER_DATA_POST_BIT));
    return CCU_SUCCESS;
}

CcuResult HelperFinish(HelperContext &ctx, uint32_t rootRank)
{
    // This wait is deliberately a separate L0-die kernel.  It preserves the
    // helper rank's user-stream completion semantics without holding back the
    // L1 forwarding kernel on the other die.
    CCU_CALL_RETURN(WaitPeerNotify(ctx, rootRank, HELPER_OUTPUT_READY_BIT));
    return CCU_SUCCESS;
}

CcuResult HybridReceiver(HelperContext &ctx, uint32_t rootRank)
{
    CCU_CALL_RETURN(PublishOutput(ctx, rootRank));
    CCU_CALL_RETURN(WaitPeerNotify(ctx, rootRank, HELPER_SENDER_READY_BIT));
    for (uint32_t i = 0; i < ctx.arg->peerCount; ++i) {
        const uint32_t peer = ctx.arg->peerRanks[i];
        if (peer == rootRank) {
            continue;
        }
        CCU_IF(ctx.stageBase[rootRank] == peer)
        {
            CCU_CALL_RETURN(PublishOutput(ctx, peer));
            CCU_CALL_RETURN(WaitPeerNotify(ctx, peer, HELPER_DATA_POST_BIT));
            CCU_CALL_RETURN(RecordPeerNotify(ctx, rootRank,
                HELPER_TARGET_ACK_BIT));
        }
    }
    CCU_CALL_RETURN(WaitPeerNotify(ctx, rootRank, HELPER_ROOT_POST_BIT));
    return CCU_SUCCESS;
}

} // namespace

CcuResult CcuKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgScatter *>(arg);
    if (kernelArg == nullptr || kernelArg->rankSize == 0 ||
        kernelArg->rankSize > MAX_RANK_SIZE || kernelArg->rankId >= kernelArg->rankSize ||
        kernelArg->channelCount != kernelArg->peerCount ||
        (kernelArg->peerCount == 0 && !kernelArg->includeLocal)) {
        return CCU_E_PARA;
    }
    for (uint32_t i = 0; i < kernelArg->peerCount; ++i) {
        if (kernelArg->peerRanks[i] >= kernelArg->rankSize ||
            kernelArg->peerRanks[i] == kernelArg->rankId ||
            (i > 0 && kernelArg->peerRanks[i] <= kernelArg->peerRanks[i - 1])) {
            return CCU_E_PARA;
        }
    }

    ScatterContext ctx;
    ctx.arg = kernelArg;
    CCU_CALL_RETURN(InitResources(ctx));
    CCU_CALL_RETURN(LoadTaskArgs(ctx));
    if (ctx.arg->rankSize == 4U) {
        CCU_IF(ctx.root == ctx.arg->rankId)
        {
            if (ctx.arg->largeBlock != 0) {
                CCU_CALL_RETURN(Rank4EarlyWriteRoot(ctx));
            } else {
                CCU_IF(ctx.blockBytes == RANK4_TARGET_BLOCK_BYTES)
                {
                    CCU_CALL_RETURN(Rank4EarlyWriteRoot(ctx));
                }
                CCU_ELSE
                {
                    CCU_CALL_RETURN(RootScatter(ctx));
                }
            }
        } CCU_ELSE {
            CCU_CALL_RETURN(ReceiverSync(ctx));
        }
    } else if (ctx.arg->rankSize == 16U) {
        CCU_IF(ctx.root == ctx.arg->rankId)
        {
            CCU_IF(ctx.blockBytes == RANK16_TARGET_BLOCK_BYTES)
            {
                CCU_CALL_RETURN(PullRoot(ctx));
            }
            CCU_ELSE
            {
                CCU_CALL_RETURN(RootScatter(ctx));
            }
        } CCU_ELSE {
            CCU_IF(ctx.blockBytes == RANK16_TARGET_BLOCK_BYTES)
            {
                CCU_CALL_RETURN(PullReceiver(ctx));
            }
            CCU_ELSE
            {
                CCU_CALL_RETURN(ReceiverSync(ctx));
            }
        }
    } else {
        CCU_IF(ctx.root == ctx.arg->rankId)
        {
            CCU_CALL_RETURN(RootScatter(ctx));
        } CCU_ELSE {
            CCU_CALL_RETURN(ReceiverSync(ctx));
        }
    }
    return CCU_SUCCESS;
}

CcuResult CcuSmallReceiverKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgScatter *>(arg);
    if (kernelArg == nullptr ||
        (kernelArg->rankSize != 4U && kernelArg->rankSize != 12U) ||
        kernelArg->rankId >= kernelArg->rankSize ||
        kernelArg->channelCount != kernelArg->peerCount ||
        kernelArg->peerCount == 0) {
        return CCU_E_PARA;
    }
    for (uint32_t i = 0; i < kernelArg->peerCount; ++i) {
        if (kernelArg->peerRanks[i] >= kernelArg->rankSize ||
            kernelArg->peerRanks[i] == kernelArg->rankId ||
            (i > 0 && kernelArg->peerRanks[i] <= kernelArg->peerRanks[i - 1])) {
            return CCU_E_PARA;
        }
    }

    SmallReceiverContext ctx;
    ctx.arg = kernelArg;
    CCU_CALL_RETURN(ccu::LoadArg(ctx.recvBase, 0));
    CCU_CALL_RETURN(ccu::LoadArg(ctx.recvToken, 1));
    CCU_CALL_RETURN(ccu::LoadArg(ctx.root, 2));
    CCU_CALL_RETURN(ccu::LoadArg(ctx.metadataReuse, 3));
    CCU_CALL_RETURN(SmallReceiverSync(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuRank12SmallRootKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgScatter *>(arg);
    if (kernelArg == nullptr ||
        (kernelArg->rankSize != 4U && kernelArg->rankSize != 12U) ||
        kernelArg->rankId >= kernelArg->rankSize ||
        kernelArg->channelCount != kernelArg->peerCount ||
        (kernelArg->peerCount == 0 && !kernelArg->includeLocal)) {
        return CCU_E_PARA;
    }
    for (uint32_t i = 0; i < kernelArg->peerCount; ++i) {
        if (kernelArg->peerRanks[i] >= kernelArg->rankSize ||
            kernelArg->peerRanks[i] == kernelArg->rankId ||
            (i > 0 && kernelArg->peerRanks[i] <= kernelArg->peerRanks[i - 1])) {
            return CCU_E_PARA;
        }
    }

    Rank12SmallRootContext ctx;
    ctx.arg = kernelArg;
    CCU_CALL_RETURN(InitRank12SmallRootResources(ctx));
    CCU_CALL_RETURN(LoadRank12SmallRootTaskArgs(ctx));
    CCU_CALL_RETURN(Rank12SmallRoot(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuWidePushKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgScatter *>(arg);
    if (kernelArg == nullptr || kernelArg->rankSize != 12U ||
        kernelArg->rankId >= kernelArg->rankSize ||
        kernelArg->channelCount != kernelArg->peerCount ||
        (kernelArg->peerCount == 0 && !kernelArg->includeLocal)) {
        return CCU_E_PARA;
    }
    for (uint32_t i = 0; i < kernelArg->peerCount; ++i) {
        if (kernelArg->peerRanks[i] >= kernelArg->rankSize ||
            kernelArg->peerRanks[i] == kernelArg->rankId ||
            (i > 0 && kernelArg->peerRanks[i] <= kernelArg->peerRanks[i - 1])) {
            return CCU_E_PARA;
        }
    }

    WidePushContext ctx;
    ctx.arg = kernelArg;
    CCU_CALL_RETURN(InitWidePushResources(ctx));
    CCU_CALL_RETURN(LoadWidePushTaskArgs(ctx));
    CCU_IF(ctx.root == ctx.arg->rankId)
    {
        CCU_CALL_RETURN(DedicatedWidePushRoot(ctx));
    } CCU_ELSE {
        for (uint32_t rootRank = 0; rootRank < ctx.arg->rankSize; ++rootRank) {
            if (rootRank == ctx.arg->rankId) {
                continue;
            }
            CCU_IF(ctx.root == rootRank)
            {
                CCU_CALL_RETURN(DedicatedWidePushReceiver(ctx, rootRank));
            }
        }
    }
    return CCU_SUCCESS;
}

CcuResult CcuHelperKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgScatter *>(arg);
    if (kernelArg == nullptr || kernelArg->rankSize == 0 ||
        kernelArg->rankSize > MAX_RANK_SIZE || kernelArg->rankId >= kernelArg->rankSize ||
        kernelArg->channelCount != kernelArg->peerCount ||
        (kernelArg->peerCount == 0 && !kernelArg->includeLocal)) {
        return CCU_E_PARA;
    }
    HelperContext ctx;
    ctx.arg = kernelArg;
    CCU_CALL_RETURN(InitHelperResources(ctx));
    CCU_CALL_RETURN(LoadHelperTaskArgs(ctx));
    if (HasWideRootInstance(*ctx.arg)) {
        CCU_IF(ctx.root == ctx.arg->rankId)
        {
            if (IsWideRootRank(*ctx.arg)) {
                CCU_IF(ctx.phase == SCATTER_WIDE_DIRECT)
                {
                    CCU_CALL_RETURN(RootWideDirect(ctx));
                }
                CCU_IF(ctx.phase != SCATTER_WIDE_DIRECT)
                {
                    CCU_CALL_RETURN(RootHelperScatter(ctx));
                }
            } else {
                CCU_CALL_RETURN(RootHelperScatter(ctx));
            }
        } CCU_ELSE {
            for (uint32_t rootRank = 0; rootRank < ctx.arg->rankSize; ++rootRank) {
                if (rootRank == ctx.arg->rankId) {
                    continue;
                }
                const StaticHelperRoute route = GetStaticHelperRoute(*ctx.arg,
                    rootRank, ctx.arg->rankId);
                CCU_IF(ctx.root == rootRank)
                {
                    CCU_IF(ctx.phase == SCATTER_WIDE_DIRECT)
                    {
                        CCU_CALL_RETURN(WideDirectReceiver(ctx, rootRank));
                    }
                    CCU_IF(ctx.phase != SCATTER_WIDE_DIRECT)
                    {
                        if (route.helperTarget != INVALID_VALUE_RANKID) {
                            CCU_IF(ctx.phase == SCATTER_HELPER_STAGE)
                            {
                                CCU_CALL_RETURN(HelperStage(ctx, rootRank));
                            }
                            CCU_IF(ctx.phase == SCATTER_HELPER_FORWARD)
                            {
                                CCU_CALL_RETURN(HelperForward(ctx,
                                    route.helperTarget));
                            }
                            CCU_IF(ctx.phase == SCATTER_HELPER_FINISH)
                            {
                                CCU_CALL_RETURN(HelperFinish(ctx, rootRank));
                            }
                        } else {
                            CCU_IF(ctx.phase == SCATTER_HELPER_NORMAL)
                            {
                                CCU_CALL_RETURN(HybridReceiver(ctx, rootRank));
                            }
                        }
                    }
                }
            }
        }
    } else {
        CCU_IF(ctx.root == ctx.arg->rankId)
        {
            CCU_CALL_RETURN(RootHelperScatter(ctx));
        } CCU_ELSE {
            for (uint32_t rootRank = 0; rootRank < ctx.arg->rankSize; ++rootRank) {
                if (rootRank == ctx.arg->rankId) {
                    continue;
                }
                const StaticHelperRoute route = GetStaticHelperRoute(*ctx.arg,
                    rootRank, ctx.arg->rankId);
                CCU_IF(ctx.root == rootRank)
                {
                    if (route.helperTarget != INVALID_VALUE_RANKID) {
                        CCU_IF(ctx.phase == SCATTER_HELPER_STAGE)
                        {
                            CCU_CALL_RETURN(HelperStage(ctx, rootRank));
                        }
                        CCU_IF(ctx.phase == SCATTER_HELPER_FORWARD)
                        {
                            CCU_CALL_RETURN(HelperForward(ctx, route.helperTarget));
                        }
                        CCU_IF(ctx.phase == SCATTER_HELPER_FINISH)
                        {
                            CCU_CALL_RETURN(HelperFinish(ctx, rootRank));
                        }
                    } else {
                        CCU_IF(ctx.phase == SCATTER_HELPER_NORMAL)
                        {
                            CCU_CALL_RETURN(HybridReceiver(ctx, rootRank));
                        }
                    }
                }
            }
        }
    }
    return CCU_SUCCESS;
}

} // namespace ops_hccl

#undef CCU_CALL_RETURN
