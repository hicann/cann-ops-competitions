/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <map>
#include <memory>
#include <string>

#include <hcomm/hcomm_primitives.h>

#include "ccu_kernel.h"

#define CCU_CHECK(call) \
    do { \
        CcuResult ccuCheckResult = (call); \
        if (ccuCheckResult != CCU_SUCCESS) { \
            return ccuCheckResult; \
        } \
    } while (0)

namespace ops_hccl {
namespace {
namespace ccu = ::AscendC::ccu;

constexpr uint32_t OUTPUT_XN_ID = 1;
constexpr uint32_t OUTPUT_TOKEN_XN_ID = 2;
constexpr uint32_t CKE_INDEX = 0;
constexpr uint32_t POST_SYNC_ID = 3;
constexpr uint32_t RELAY_READY_SYNC_ID = 4;
constexpr uint32_t PHASE_ONE_OUTPUT_SYNC_ID = 5;
constexpr uint32_t PHASE_ONE_TOKEN_SYNC_ID = 6;
constexpr uint32_t RANK12_PHASE_ZERO_SYNC_ID = 7;
constexpr uint32_t NHR_STEP_SYNC_ID = 4;
constexpr uint8_t CHANNEL_KIND_MESH = 0;
constexpr uint8_t CHANNEL_KIND_CLOS = 1;
constexpr uint64_t CCU_MS_SIZE = 4096;
constexpr uint32_t LOCAL_COPY_MS_INTERLEAVE = 8;
constexpr uint32_t LOCAL_COPY_MS_PER_LOOP = 8;
constexpr uint32_t LOCAL_COPY_LOOP_COUNT = 8;
constexpr uint32_t FINE_LOCAL_COPY_MS_INTERLEAVE = 4;
constexpr uint32_t FINE_LOCAL_COPY_MS_PER_LOOP = 4;
constexpr uint32_t FINE_LOCAL_COPY_LOOP_COUNT = 16;
constexpr uint64_t GROUP_BROADCAST_MS_INTERLEAVE = 4;
constexpr uint32_t GROUP_BROADCAST_MS_PER_LOOP = 4;
constexpr uint32_t GROUP_BROADCAST_LOOP_COUNT = 16;
constexpr uint32_t GROUP_BROADCAST_SAFE_EVENT_LIMIT = 16;
static_assert(GROUP_BROADCAST_LOOP_COUNT <=
    GROUP_BROADCAST_SAFE_EVENT_LIMIT,
    "group broadcast Event count exceeds the verified CCU limit");
static_assert(FINE_LOCAL_COPY_LOOP_COUNT <=
    GROUP_BROADCAST_SAFE_EVENT_LIMIT,
    "local copy Event count exceeds the verified CCU limit");
static_assert(FINE_LOCAL_COPY_LOOP_COUNT *
        FINE_LOCAL_COPY_MS_INTERLEAVE ==
    LOCAL_COPY_LOOP_COUNT * LOCAL_COPY_MS_INTERLEAVE,
    "fine local copy must preserve the CCU buffer footprint");

struct GroupCopySize {
    ccu::Variable addrOffset;
    ccu::Variable loopParam;
    ccu::Variable parallelParam;
    ccu::Variable residual;
};

struct GroupCopyResource {
    ccu::Array<ccu::Event> completedEvents{0};
    ccu::Array<ccu::CcuBuffer> buffers{0};
};

struct GroupBroadcastVariables {
    ccu::LocalAddr sources[2];
    ccu::LocalAddr localDestinations[2];
    std::vector<ccu::RemoteAddr> remoteDestinations[2];
    ccu::Variable lengths[2];
};

struct LoopEntity {
    std::unique_ptr<ccu::Func> bodies[2];
    std::unique_ptr<ccu::Loop> loops[2];
    ccu::Variable loopParams[2];
};

struct AllGatherContext {
    const CcuKernelArgAllGather *arg = nullptr;
    ccu::Variable input;
    ccu::Variable localOutput;
    std::vector<ccu::Variable> remoteOutputs;
    ccu::Variable inputToken;
    ccu::Variable localOutputToken;
    std::vector<ccu::Variable> remoteOutputTokens;
    std::vector<uint32_t> broadcastChannelIndices;
    ccu::Variable outputOffset;
    ccu::Variable sliceBytes;
    ccu::Variable closPrimaryBytes;
    ccu::Variable closAlternateBytes;
    ccu::Variable secondSliceBytes;
    ccu::Variable directBytes;
    ccu::Variable relayBytes;
    ccu::Variable relayOutputOffset;
    ccu::Variable rank16Phase;
    ccu::Variable fanoutOutputOffsets[2];
    ccu::Variable rank12Phase;
    ccu::Variable localCopyOffset;
    ccu::Variable directLocalCopyBytes;
    ccu::LocalAddr source;
    std::vector<ccu::RemoteAddr> remoteDestinations;
    ccu::LocalAddr localSource;
    ccu::LocalAddr localDestination;
    ccu::Event completion;
    ccu::Event secondCompletion;
    ccu::Event directLocalCopyCompletion;
    GroupCopySize groupCopySize;
    GroupCopyResource groupCopyResource;
    std::map<std::string, LoopEntity> loopEntities;
    bool groupCopyResourceAllocated = false;
};

constexpr uint64_t LowBits(uint32_t highestBit)
{
    return (uint64_t{1} << (highestBit + 1)) - 1;
}

uint64_t GetLoopParam(
    uint64_t loopContextId, uint64_t addressOffset, uint64_t loopIterations)
{
    return ((loopContextId & LowBits(8)) << 45) |
        ((addressOffset & LowBits(32)) << 13) |
        (loopIterations & LowBits(13));
}

uint64_t GetParallelParam(
    uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum)
{
    return ((repeatNum & LowBits(7)) << 55) |
        ((repeatLoopIndex & LowBits(7)) << 48) |
        ((totalLoopNum & LowBits(7)) << 41);
}

uint64_t GetOffsetParam(
    uint64_t addressOffset, uint64_t bufferOffset, uint64_t eventOffset)
{
    return ((addressOffset & LowBits(32)) << 21) |
        ((bufferOffset & LowBits(11)) << 10) |
        (eventOffset & LowBits(10));
}

CcuResult InitResources(AllGatherContext &ctx)
{
    if (ctx.arg == nullptr || ctx.arg->rankSize <= 1 ||
        ctx.arg->rankSize > MAX_RANK_SIZE ||
        ctx.arg->rankId >= ctx.arg->rankSize ||
        ctx.arg->channelCount == 0 ||
        ctx.arg->channelCount > ctx.arg->rankSize - 1 ||
        (ctx.arg->directMeshFirst &&
            (ctx.arg->rankSize != 12 ||
                ctx.arg->useGroupBroadcast)) ||
        (ctx.arg->fuseTwoSlices &&
            ((ctx.arg->rankSize != 4 &&
                 ctx.arg->rankSize != 12) ||
                ctx.arg->useGroupBroadcast ||
                (ctx.arg->rankSize == 4 &&
                    ctx.arg->channelCount > 8))) ||
        (ctx.arg->useFineLocalCopy &&
            (ctx.arg->rankSize != 12 ||
                ctx.arg->useGroupBroadcast ||
                !ctx.arg->copyLocal))) {
        return CCU_E_PARA;
    }
    if (ctx.arg->useRank12DualRail) {
        if (ctx.arg->rankSize != 12 ||
            ctx.arg->useRank12SplitRelay ||
            ctx.arg->useRank16MixedRelay ||
            ctx.arg->useGroupBroadcast) {
            return CCU_E_PARA;
        }
        bool hasClosChannel = false;
        for (uint32_t channelIndex = 0;
             channelIndex < ctx.arg->channelCount;
             ++channelIndex) {
            hasClosChannel =
                hasClosChannel ||
                ctx.arg->channelKinds[channelIndex] ==
                    CHANNEL_KIND_CLOS;
            if (ctx.arg->alternateRailChannels[channelIndex] &&
                ctx.arg->channelKinds[channelIndex] !=
                    CHANNEL_KIND_CLOS) {
                return CCU_E_PARA;
            }
        }
        if (!hasClosChannel) {
            return CCU_E_PARA;
        }
    }
    if (ctx.arg->useDirectLocalCopy &&
        (ctx.arg->rankSize != 12 ||
            !ctx.arg->copyLocal ||
            ctx.arg->useFineLocalCopy ||
            ctx.arg->useGroupBroadcast ||
            ctx.arg->useRank12DualRail ||
            ctx.arg->useRank12SplitRelay ||
            ctx.arg->useRank16MixedRelay)) {
        return CCU_E_PARA;
    }
    if (ctx.arg->useRank4DirectReuse &&
        (ctx.arg->rankSize != 4 ||
            ctx.arg->useGroupBroadcast ||
            ctx.arg->directMeshFirst ||
            ctx.arg->fuseTwoSlices ||
            ctx.arg->useFineLocalCopy ||
            ctx.arg->useDirectLocalCopy ||
            ctx.arg->useRank12Nhr ||
            ctx.arg->useRank12DualRail ||
            ctx.arg->useRank12SplitRelay ||
            ctx.arg->useRank16MixedRelay)) {
        return CCU_E_PARA;
    }
    if (ctx.arg->useRank12DirectReuse &&
        (ctx.arg->rankSize != 12 ||
            ctx.arg->useGroupBroadcast ||
            ctx.arg->fuseTwoSlices ||
            ctx.arg->useDirectLocalCopy ||
            ctx.arg->useRank4DirectReuse ||
            ctx.arg->useRank12Nhr ||
            ctx.arg->useRank12DualRail ||
            ctx.arg->useRank12SplitRelay ||
            ctx.arg->useRank16MixedRelay)) {
        return CCU_E_PARA;
    }
    if (ctx.arg->useRank12Nhr) {
        if (ctx.arg->rankSize != 12 ||
            ctx.arg->nhrAxisId >= 2 ||
            ctx.arg->nhrStepCount !=
                NHR_MAX_STEP_COUNT ||
            ctx.arg->copyLocal ||
            ctx.arg->useGroupBroadcast ||
            ctx.arg->directMeshFirst ||
            ctx.arg->fuseTwoSlices ||
            ctx.arg->useFineLocalCopy ||
            ctx.arg->useDirectLocalCopy ||
            ctx.arg->useRank12DualRail ||
            ctx.arg->useRank12SplitRelay ||
            ctx.arg->useRank16MixedRelay) {
            return CCU_E_PARA;
        }
        for (uint32_t step = 0;
             step < ctx.arg->nhrStepCount; ++step) {
            uint32_t sliceCount =
                ctx.arg->nhrStepSliceCounts[step];
            if (ctx.arg->nhrToChannelIndices[step] >=
                    ctx.arg->channelCount ||
                ctx.arg->nhrFromChannelIndices[step] >=
                    ctx.arg->channelCount ||
                sliceCount == 0 ||
                sliceCount >
                    NHR_MAX_STEP_SLICE_COUNT) {
                return CCU_E_PARA;
            }
            for (uint32_t index = 0;
                 index < sliceCount; ++index) {
                if (ctx.arg->nhrStepSliceIndices[step]
                        [index] >= ctx.arg->rankSize) {
                    return CCU_E_PARA;
                }
            }
        }
    }
    if (ctx.arg->useRank12SplitRelay) {
        uint32_t expectedChannelCount =
            ctx.arg->rank12SmallServer ?
                (ctx.arg->isRank12MeshKernel ? 3 : 8) :
                (ctx.arg->isRank12MeshKernel ? 7 : 4);
        uint8_t expectedChannelKind =
            ctx.arg->isRank12MeshKernel ?
                CHANNEL_KIND_MESH : CHANNEL_KIND_CLOS;
        if (ctx.arg->rankSize != 12 ||
            ctx.arg->channelCount != expectedChannelCount ||
            ctx.arg->useRank16MixedRelay ||
            ctx.arg->useGroupBroadcast ||
            ctx.arg->directMeshFirst ||
            ctx.arg->fuseTwoSlices ||
            ctx.arg->copyLocal ==
                ctx.arg->isRank12MeshKernel ||
            ctx.arg->useFineLocalCopy ==
                ctx.arg->isRank12MeshKernel ||
            ctx.arg->rank12SmallServer !=
                (ctx.arg->rankId >= 8)) {
            return CCU_E_PARA;
        }
        for (uint32_t channelIndex = 0;
             channelIndex < ctx.arg->channelCount;
             ++channelIndex) {
            if (ctx.arg->channelKinds[channelIndex] !=
                expectedChannelKind) {
                return CCU_E_PARA;
            }
        }
        uint32_t expectedFanoutSourceCount =
            ctx.arg->rank12SmallServer ? 2 :
                (ctx.arg->rankId < 4 ? 1 : 0);
        if (ctx.arg->rank12FanoutSourceCount !=
                expectedFanoutSourceCount) {
            return CCU_E_PARA;
        }
        if (!ctx.arg->isRank12MeshKernel) {
            if (ctx.arg->rank12BridgeTargetChannelIndex >=
                    ctx.arg->channelCount) {
                return CCU_E_PARA;
            }
            for (uint32_t sourceIndex = 0;
                 sourceIndex <
                     ctx.arg->rank12FanoutSourceCount;
                 ++sourceIndex) {
                if (ctx.arg->
                        rank12FanoutSourceChannelIndices[
                            sourceIndex] >=
                    ctx.arg->channelCount) {
                    return CCU_E_PARA;
                }
            }
        }
    }
    if (ctx.arg->useRank16MixedRelay) {
        uint32_t expectedChannelCount =
            ctx.arg->isRank16MeshKernel ? 7 : 8;
        uint8_t expectedChannelKind =
            ctx.arg->isRank16MeshKernel ?
                CHANNEL_KIND_MESH : CHANNEL_KIND_CLOS;
        bool expectedGroupBroadcast =
            !ctx.arg->isRank16MeshKernel;
        if (ctx.arg->rankSize != 16 ||
            ctx.arg->channelCount != expectedChannelCount ||
            ctx.arg->useGroupBroadcast !=
                expectedGroupBroadcast ||
            ctx.arg->directMeshFirst ||
            ctx.arg->fuseTwoSlices ||
            ctx.arg->useFineLocalCopy ||
            ctx.arg->copyLocal !=
                ctx.arg->isRank16MeshKernel ||
            ctx.arg->relayPeerRank >= ctx.arg->rankSize ||
            ctx.arg->relayPeerRank == ctx.arg->rankId ||
            (!ctx.arg->isRank16MeshKernel &&
                ctx.arg->relayChannelIndex >=
                    ctx.arg->channelCount)) {
            return CCU_E_PARA;
        }
        for (uint32_t channelIndex = 0;
             channelIndex < ctx.arg->channelCount;
             ++channelIndex) {
            if (ctx.arg->channelKinds[channelIndex] !=
                expectedChannelKind) {
                return CCU_E_PARA;
            }
        }
        if (!ctx.arg->isRank16MeshKernel &&
            ctx.arg->remoteRanks[ctx.arg->relayChannelIndex] !=
                ctx.arg->relayPeerRank) {
            return CCU_E_PARA;
        }
    }
    ctx.remoteOutputs.resize(ctx.arg->channelCount);
    ctx.remoteOutputTokens.resize(ctx.arg->channelCount);
    for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount;
         ++channelIndex) {
        uint32_t remoteRank = ctx.arg->remoteRanks[channelIndex];
        if (remoteRank >= ctx.arg->rankSize ||
            remoteRank == ctx.arg->rankId ||
            ctx.arg->channelKinds[channelIndex] > CHANNEL_KIND_CLOS) {
            return CCU_E_PARA;
        }
        ctx.remoteOutputs[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIndex], OUTPUT_XN_ID);
        ctx.remoteOutputTokens[channelIndex] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIndex], OUTPUT_TOKEN_XN_ID);
        if (ctx.arg->useGroupBroadcast &&
            (ctx.arg->useRank16MixedRelay ||
                ctx.arg->rankSize == 4 ||
                ctx.arg->channelKinds[channelIndex] ==
                    CHANNEL_KIND_CLOS)) {
            ctx.broadcastChannelIndices.push_back(channelIndex);
        }
    }
    if (ctx.arg->useGroupBroadcast &&
        ctx.broadcastChannelIndices.empty()) {
        return CCU_E_PARA;
    }
    return CCU_SUCCESS;
}

CcuResult LoadTaskArgs(AllGatherContext &ctx)
{
    uint32_t argIndex = 0;
    CCU_CHECK(ccu::LoadArg(ctx.input, argIndex++));
    CCU_CHECK(ccu::LoadArg(ctx.localOutput, argIndex++));
    CCU_CHECK(ccu::LoadArg(ctx.inputToken, argIndex++));
    CCU_CHECK(ccu::LoadArg(ctx.localOutputToken, argIndex++));
    CCU_CHECK(ccu::LoadArg(ctx.outputOffset, argIndex++));
    if (ctx.arg->useRank12Nhr) {
        CCU_CHECK(ccu::LoadArg(
            ctx.localCopyOffset, argIndex++));
        CCU_CHECK(ccu::LoadArg(
            ctx.sliceBytes, argIndex++));
        CCU_CHECK(ccu::LoadArg(
            ctx.directLocalCopyBytes, argIndex++));
        return CCU_SUCCESS;
    }
    if (ctx.arg->useRank12SplitRelay) {
        if (ctx.arg->isRank12MeshKernel) {
            CCU_CHECK(ccu::LoadArg(
                ctx.sliceBytes, argIndex++));
            CCU_CHECK(ccu::LoadArg(
                ctx.fanoutOutputOffsets[0], argIndex++));
            CCU_CHECK(ccu::LoadArg(
                ctx.fanoutOutputOffsets[1], argIndex++));
        } else {
            CCU_CHECK(ccu::LoadArg(
                ctx.sliceBytes, argIndex++));
            CCU_CHECK(ccu::LoadArg(
                ctx.directBytes, argIndex++));
            CCU_CHECK(ccu::LoadArg(
                ctx.relayBytes, argIndex++));
            CCU_CHECK(ccu::LoadArg(
                ctx.groupCopySize.addrOffset, argIndex++));
            CCU_CHECK(ccu::LoadArg(
                ctx.groupCopySize.loopParam, argIndex++));
            CCU_CHECK(ccu::LoadArg(
                ctx.groupCopySize.parallelParam, argIndex++));
            CCU_CHECK(ccu::LoadArg(
                ctx.groupCopySize.residual, argIndex++));
        }
        CCU_CHECK(ccu::LoadArg(
            ctx.rank12Phase, argIndex++));
        return CCU_SUCCESS;
    }
    if (ctx.arg->useRank12DualRail) {
        CCU_CHECK(ccu::LoadArg(ctx.sliceBytes, argIndex++));
        CCU_CHECK(ccu::LoadArg(
            ctx.closPrimaryBytes, argIndex++));
        CCU_CHECK(ccu::LoadArg(
            ctx.closAlternateBytes, argIndex++));
        if (ctx.arg->copyLocal) {
            ctx.localCopyOffset = 0;
            CCU_CHECK(ccu::LoadArg(
                ctx.groupCopySize.addrOffset, argIndex++));
            CCU_CHECK(ccu::LoadArg(
                ctx.groupCopySize.loopParam, argIndex++));
            CCU_CHECK(ccu::LoadArg(
                ctx.groupCopySize.parallelParam, argIndex++));
            CCU_CHECK(ccu::LoadArg(
                ctx.groupCopySize.residual, argIndex++));
        }
        if (ctx.arg->fuseTwoSlices) {
            CCU_CHECK(ccu::LoadArg(
                ctx.secondSliceBytes, argIndex++));
        }
        return CCU_SUCCESS;
    }
    if (ctx.arg->useDirectLocalCopy) {
        CCU_CHECK(ccu::LoadArg(
            ctx.sliceBytes, argIndex++));
        CCU_CHECK(ccu::LoadArg(
            ctx.localCopyOffset, argIndex++));
        CCU_CHECK(ccu::LoadArg(
            ctx.directLocalCopyBytes, argIndex++));
        if (ctx.arg->fuseTwoSlices) {
            CCU_CHECK(ccu::LoadArg(
                ctx.secondSliceBytes, argIndex++));
        }
        return CCU_SUCCESS;
    }
    if (ctx.arg->useRank16MixedRelay) {
        CCU_CHECK(ccu::LoadArg(ctx.directBytes, argIndex++));
        CCU_CHECK(ccu::LoadArg(ctx.relayBytes, argIndex++));
        CCU_CHECK(ccu::LoadArg(
            ctx.relayOutputOffset, argIndex++));
        if (ctx.arg->isRank16MeshKernel) {
            CCU_CHECK(ccu::LoadArg(
                ctx.groupCopySize.addrOffset, argIndex++));
            CCU_CHECK(ccu::LoadArg(
                ctx.groupCopySize.loopParam, argIndex++));
            CCU_CHECK(ccu::LoadArg(
                ctx.groupCopySize.parallelParam, argIndex++));
            CCU_CHECK(ccu::LoadArg(
                ctx.groupCopySize.residual, argIndex++));
        } else {
            ctx.groupCopySize.addrOffset = ctx.directBytes;
            CCU_CHECK(ccu::LoadArg(
                ctx.groupCopySize.loopParam, argIndex++));
            ctx.groupCopySize.parallelParam = 0;
            ctx.groupCopySize.residual = 0;
        }
        CCU_CHECK(ccu::LoadArg(
            ctx.rank16Phase, argIndex++));
        return CCU_SUCCESS;
    }
    if (ctx.arg->useGroupBroadcast) {
        if (ctx.arg->rankSize != 4) {
            CCU_CHECK(ccu::LoadArg(ctx.sliceBytes, argIndex++));
        }
        CCU_CHECK(ccu::LoadArg(
            ctx.groupCopySize.addrOffset, argIndex++));
        CCU_CHECK(ccu::LoadArg(
            ctx.groupCopySize.loopParam, argIndex++));
        CCU_CHECK(ccu::LoadArg(
            ctx.groupCopySize.parallelParam, argIndex++));
        CCU_CHECK(ccu::LoadArg(
            ctx.groupCopySize.residual, argIndex++));
        return CCU_SUCCESS;
    }

    CCU_CHECK(ccu::LoadArg(ctx.sliceBytes, argIndex++));
    if (ctx.arg->copyLocal) {
        CCU_CHECK(ccu::LoadArg(ctx.localCopyOffset, argIndex++));
        CCU_CHECK(ccu::LoadArg(ctx.groupCopySize.addrOffset, argIndex++));
        CCU_CHECK(ccu::LoadArg(ctx.groupCopySize.loopParam, argIndex++));
        CCU_CHECK(ccu::LoadArg(ctx.groupCopySize.parallelParam, argIndex++));
        CCU_CHECK(ccu::LoadArg(ctx.groupCopySize.residual, argIndex++));
    }
    if (ctx.arg->fuseTwoSlices) {
        CCU_CHECK(ccu::LoadArg(ctx.secondSliceBytes, argIndex++));
    }
    return CCU_SUCCESS;
}

CcuResult PublishRemoteChannelResources(
    AllGatherContext &ctx, uint32_t channelIndex,
    uint32_t outputSyncId = OUTPUT_XN_ID,
    uint32_t tokenSyncId = OUTPUT_TOKEN_XN_ID)
{
    CCU_CHECK(ccu::WriteVariableWithNotify(
        ctx.arg->channels[channelIndex], ctx.localOutput,
        OUTPUT_XN_ID, CKE_INDEX, 1U << outputSyncId));
    CCU_CHECK(ccu::WriteVariableWithNotify(
        ctx.arg->channels[channelIndex], ctx.localOutputToken,
        OUTPUT_TOKEN_XN_ID, CKE_INDEX, 1U << tokenSyncId));
    return CCU_SUCCESS;
}

CcuResult PublishRemoteResources(AllGatherContext &ctx)
{
    if (ctx.arg->useGroupBroadcast && ctx.arg->rankSize != 4) {
        constexpr uint8_t publishOrder[] = {
            CHANNEL_KIND_MESH,
            CHANNEL_KIND_CLOS,
        };
        for (uint8_t channelKind : publishOrder) {
            for (uint32_t channelIndex = 0;
                 channelIndex < ctx.arg->channelCount; ++channelIndex) {
                if (ctx.arg->channelKinds[channelIndex] != channelKind) {
                    continue;
                }
                CCU_CHECK(PublishRemoteChannelResources(
                    ctx, channelIndex));
            }
        }
        return CCU_SUCCESS;
    }

    for (uint32_t channelIndex = 0;
         channelIndex < ctx.arg->channelCount; ++channelIndex) {
        CCU_CHECK(PublishRemoteChannelResources(ctx, channelIndex));
    }
    return CCU_SUCCESS;
}

CcuResult InitTransferAddresses(AllGatherContext &ctx)
{
    ctx.source.addr = ctx.input;
    ctx.source.token = ctx.inputToken;

    ctx.remoteDestinations.resize(ctx.arg->channelCount);

    if (ctx.arg->copyLocal) {
        ctx.localSource.addr = ctx.input;
        ctx.localSource.token = ctx.inputToken;
        ctx.localDestination.addr = ctx.localOutput;
        ctx.localDestination.addr += ctx.outputOffset;
        ctx.localDestination.token = ctx.localOutputToken;
        if (!ctx.arg->useGroupBroadcast) {
            ctx.localSource.addr += ctx.localCopyOffset;
            ctx.localDestination.addr += ctx.localCopyOffset;
        }
    }
    return CCU_SUCCESS;
}

CcuResult PrepareRemoteChannel(
    AllGatherContext &ctx, uint32_t channelIndex, bool issueDirect,
    uint32_t outputSyncId = OUTPUT_XN_ID,
    uint32_t tokenSyncId = OUTPUT_TOKEN_XN_ID)
{
    uint32_t readyMask =
        (1U << outputSyncId) | (1U << tokenSyncId);
    CCU_CHECK(ccu::NotifyWait(ctx.arg->channels[channelIndex],
        CKE_INDEX, readyMask));
    ctx.remoteDestinations[channelIndex].addr =
        ctx.remoteOutputs[channelIndex];
    ctx.remoteDestinations[channelIndex].addr += ctx.outputOffset;
    ctx.remoteDestinations[channelIndex].token =
        ctx.remoteOutputTokens[channelIndex];

    if (issueDirect) {
        ccu::LocalAddr transferSource = ctx.source;
        ccu::Variable transferBytes;
        transferBytes = ctx.sliceBytes;
        if (ctx.arg->useRank12DualRail &&
            ctx.arg->channelKinds[channelIndex] ==
                CHANNEL_KIND_CLOS) {
            if (ctx.arg->alternateRailChannels[channelIndex]) {
                transferSource.addr +=
                    ctx.closPrimaryBytes;
                ctx.remoteDestinations[channelIndex].addr +=
                    ctx.closPrimaryBytes;
                transferBytes =
                    ctx.closAlternateBytes;
            } else {
                transferBytes = ctx.closPrimaryBytes;
            }
        }
        uint16_t remoteMask =
            static_cast<uint16_t>(1U << channelIndex);
        CCU_CHECK(ccu::Write(ctx.arg->channels[channelIndex],
            ctx.remoteDestinations[channelIndex], transferSource,
            transferBytes, ctx.completion, remoteMask));
    }
    return CCU_SUCCESS;
}

CcuResult IssueAllGather(AllGatherContext &ctx)
{
    if ((ctx.arg->useGroupBroadcast && ctx.arg->rankSize != 4) ||
        ctx.arg->directMeshFirst) {
        constexpr uint8_t issueOrder[] = {
            CHANNEL_KIND_MESH,
            CHANNEL_KIND_CLOS,
        };
        for (uint8_t channelKind : issueOrder) {
            for (uint32_t channelIndex = 0;
                 channelIndex < ctx.arg->channelCount; ++channelIndex) {
                if (ctx.arg->channelKinds[channelIndex] != channelKind) {
                    continue;
                }
                CCU_CHECK(PrepareRemoteChannel(ctx, channelIndex,
                    ctx.arg->directMeshFirst ||
                        channelKind == CHANNEL_KIND_MESH));
            }
        }
        return CCU_SUCCESS;
    }

    for (uint32_t channelIndex = 0;
         channelIndex < ctx.arg->channelCount; ++channelIndex) {
        CCU_CHECK(PrepareRemoteChannel(ctx, channelIndex,
            !ctx.arg->useGroupBroadcast));
    }
    return CCU_SUCCESS;
}

CcuResult IssueSecondDirectChannel(AllGatherContext &ctx,
    uint32_t channelIndex, ccu::LocalAddr secondSource)
{
    ccu::RemoteAddr secondDestination =
        ctx.remoteDestinations[channelIndex];
    secondDestination.addr += ctx.sliceBytes;
    ccu::Variable transferBytes;
    transferBytes = ctx.secondSliceBytes;
    if (ctx.arg->useRank12DualRail &&
        ctx.arg->channelKinds[channelIndex] ==
            CHANNEL_KIND_CLOS) {
        if (ctx.arg->alternateRailChannels[channelIndex]) {
            secondSource.addr += ctx.closPrimaryBytes;
            transferBytes = ctx.closAlternateBytes;
        } else {
            transferBytes = ctx.closPrimaryBytes;
        }
    }
    if (ctx.arg->rankSize == 12) {
        uint16_t completionBit =
            static_cast<uint16_t>(1U << channelIndex);
        CCU_CHECK(ccu::Write(ctx.arg->channels[channelIndex],
            secondDestination, secondSource,
            transferBytes, ctx.secondCompletion, completionBit));
    } else {
        uint16_t completionBit = static_cast<uint16_t>(
            1U << (channelIndex + ctx.arg->channelCount));
        CCU_CHECK(ccu::Write(ctx.arg->channels[channelIndex],
            secondDestination, secondSource,
            ctx.secondSliceBytes, ctx.completion, completionBit));
    }
    return CCU_SUCCESS;
}

CcuResult IssueSecondDirectSlice(AllGatherContext &ctx)
{
    if (!ctx.arg->fuseTwoSlices) {
        return CCU_SUCCESS;
    }

    CCU_IF(ctx.secondSliceBytes != 0) {
        ccu::LocalAddr secondSource = ctx.source;
        secondSource.addr += ctx.sliceBytes;
        if (ctx.arg->directMeshFirst) {
            constexpr uint8_t issueOrder[] = {
                CHANNEL_KIND_MESH,
                CHANNEL_KIND_CLOS,
            };
            for (uint8_t channelKind : issueOrder) {
                for (uint32_t channelIndex = 0;
                     channelIndex < ctx.arg->channelCount;
                     ++channelIndex) {
                    if (ctx.arg->channelKinds[channelIndex] !=
                        channelKind) {
                        continue;
                    }
                    CCU_CHECK(IssueSecondDirectChannel(
                        ctx, channelIndex, secondSource));
                }
            }
        } else {
            for (uint32_t channelIndex = 0;
                 channelIndex < ctx.arg->channelCount;
                 ++channelIndex) {
                CCU_CHECK(IssueSecondDirectChannel(
                    ctx, channelIndex, secondSource));
            }
        }
    }
    return CCU_SUCCESS;
}

CcuResult CompleteTransfersAndNotify(AllGatherContext &ctx)
{
    constexpr uint8_t completionOrder[] = {
        CHANNEL_KIND_MESH,
        CHANNEL_KIND_CLOS,
    };
    for (uint8_t channelKind : completionOrder) {
        for (uint32_t channelIndex = 0;
             channelIndex < ctx.arg->channelCount; ++channelIndex) {
            if (ctx.arg->channelKinds[channelIndex] != channelKind) {
                continue;
            }
            bool directChannel =
                !ctx.arg->useGroupBroadcast ||
                (ctx.arg->rankSize != 4 &&
                    ctx.arg->channelKinds[channelIndex] ==
                        CHANNEL_KIND_MESH);
            if (directChannel) {
                uint16_t completionBit =
                    static_cast<uint16_t>(1U << channelIndex);
                if (ctx.arg->fuseTwoSlices &&
                    ctx.arg->rankSize == 4) {
                    uint16_t secondCompletionBit =
                        static_cast<uint16_t>(
                            1U << (channelIndex +
                                ctx.arg->channelCount));
                    uint16_t fusedCompletionMask =
                        static_cast<uint16_t>(
                            completionBit | secondCompletionBit);
                    CCU_IF(ctx.secondSliceBytes != 0) {
                        CCU_CHECK(ccu::EventWait(
                            ctx.completion, fusedCompletionMask));
                    } CCU_ELSE {
                        CCU_CHECK(ccu::EventWait(
                            ctx.completion, completionBit));
                    }
                } else {
                    CCU_CHECK(ccu::EventWait(
                        ctx.completion, completionBit));
                    if (ctx.arg->fuseTwoSlices) {
                        CCU_IF(ctx.secondSliceBytes != 0) {
                            uint16_t secondCompletionBit =
                                static_cast<uint16_t>(
                                    1U << channelIndex);
                            CCU_CHECK(ccu::EventWait(
                                ctx.secondCompletion,
                                secondCompletionBit));
                        }
                    }
                }
            }
            CCU_CHECK(ccu::NotifyRecord(
                ctx.arg->channels[channelIndex],
                CKE_INDEX, 1U << POST_SYNC_ID));
        }
    }
    return CCU_SUCCESS;
}

void InitGroupCopyResources(AllGatherContext &ctx,
    ccu::LocalAddr *loopSources, ccu::LocalAddr *loopDestinations,
    ccu::Variable *loopLengths)
{
    uint32_t eventCount = ctx.arg->useFineLocalCopy ?
        FINE_LOCAL_COPY_LOOP_COUNT : LOCAL_COPY_LOOP_COUNT;
    uint32_t msInterleave = ctx.arg->useFineLocalCopy ?
        FINE_LOCAL_COPY_MS_INTERLEAVE : LOCAL_COPY_MS_INTERLEAVE;
    uint32_t bufferCount = eventCount * msInterleave;
    constexpr uint32_t bodyCount = 2;

    if (!ctx.groupCopyResourceAllocated) {
        ctx.groupCopyResource.completedEvents =
            ccu::Array<ccu::Event>(eventCount);
        ctx.groupCopyResource.buffers =
            ccu::Array<ccu::CcuBuffer>(bufferCount);
        ctx.groupCopyResourceAllocated = true;
    }

    constexpr char loopName[] = "allgather_local_copy";
    if (ctx.loopEntities.count(loopName) == 0) {
        ctx.loopEntities.emplace(loopName, LoopEntity{});
        LoopEntity &entity = ctx.loopEntities[loopName];
        for (uint32_t index = 0; index < bodyCount; ++index) {
            uint32_t bufferBase = index * msInterleave;
            ccu::Event completion =
                ctx.groupCopyResource.completedEvents[index];
            entity.bodies[index].reset(new ccu::Func(
                [&ctx, index, bufferBase, completion, loopSources,
                    loopDestinations, loopLengths]() {
                    (void)ccu::LocalCopy(
                        ctx.groupCopyResource.buffers[bufferBase],
                        loopSources[index], loopLengths[index], completion, 1);
                    (void)ccu::EventWait(completion, 1);
                    (void)ccu::LocalCopy(
                        loopDestinations[index],
                        ctx.groupCopyResource.buffers[bufferBase],
                        loopLengths[index], completion, 1);
                    (void)ccu::EventWait(completion, 1);
                }));
            entity.loops[index].reset(new ccu::Loop(
                entity.loopParams[index], *entity.bodies[index]));
        }
    }
}

CcuResult GroupCopy(AllGatherContext &ctx)
{
    uint64_t msInterleave = ctx.arg->useFineLocalCopy ?
        FINE_LOCAL_COPY_MS_INTERLEAVE : LOCAL_COPY_MS_INTERLEAVE;
    uint32_t msPerLoop = ctx.arg->useFineLocalCopy ?
        FINE_LOCAL_COPY_MS_PER_LOOP : LOCAL_COPY_MS_PER_LOOP;
    uint32_t loopCount = ctx.arg->useFineLocalCopy ?
        FINE_LOCAL_COPY_LOOP_COUNT : LOCAL_COPY_LOOP_COUNT;
    uint64_t copySlice = CCU_MS_SIZE * msPerLoop;
    uint64_t loopStride = copySlice * loopCount;

    ccu::LocalAddr loopSources[2];
    ccu::LocalAddr loopDestinations[2];
    ccu::Variable loopLengths[2];
    InitGroupCopyResources(
        ctx, loopSources, loopDestinations, loopLengths);
    LoopEntity &entity = ctx.loopEntities["allgather_local_copy"];

    CCU_IF(ctx.groupCopySize.addrOffset != 0) {
        ccu::Variable loopParam;
        loopParam = GetLoopParam(0, loopStride, 0);
        loopParam += ctx.groupCopySize.loopParam;

        ccu::Variable sliceSize;
        sliceSize = copySlice;
        loopSources[0].addr = ctx.localSource.addr;
        loopSources[0].token = ctx.localSource.token;
        loopDestinations[0].addr = ctx.localDestination.addr;
        loopDestinations[0].token = ctx.localDestination.token;
        loopLengths[0] = sliceSize;
        entity.loopParams[0] = loopParam;

        ccu::Variable parallelConfig;
        parallelConfig =
            GetParallelParam(loopCount - 1, 0, 1);
        ccu::Variable offsetConfig;
        offsetConfig =
            GetOffsetParam(copySlice, msInterleave, 1);
        std::vector<ccu::Loop> loops{*entity.loops[0]};
        ccu::LoopGroup group(
            parallelConfig, offsetConfig, loopCount, loops);
    }

    CCU_IF(ctx.groupCopySize.parallelParam != 0) {
        ctx.localSource.addr += ctx.groupCopySize.addrOffset;
        ctx.localDestination.addr += ctx.groupCopySize.addrOffset;

        loopSources[0].addr = ctx.localSource.addr;
        loopSources[0].token = ctx.localSource.token;
        loopDestinations[0].addr = ctx.localDestination.addr;
        loopDestinations[0].token = ctx.localDestination.token;
        loopLengths[0] = ctx.groupCopySize.residual;

        ctx.localSource.addr += ctx.groupCopySize.residual;
        ctx.localDestination.addr += ctx.groupCopySize.residual;

        ccu::Variable sliceSize;
        sliceSize = copySlice;
        loopSources[1].addr = ctx.localSource.addr;
        loopSources[1].token = ctx.localSource.token;
        loopDestinations[1].addr = ctx.localDestination.addr;
        loopDestinations[1].token = ctx.localDestination.token;
        loopLengths[1] = sliceSize;

        ccu::Variable firstLoopConfig;
        firstLoopConfig = GetLoopParam(0, 0, 1);
        ccu::Variable secondLoopConfig;
        secondLoopConfig = GetLoopParam(0, 0, 1);
        entity.loopParams[0] = firstLoopConfig;
        entity.loopParams[1] = secondLoopConfig;

        ccu::Variable offsetConfig;
        offsetConfig =
            GetOffsetParam(copySlice, msInterleave, 1);
        std::vector<ccu::Loop> loops{
            *entity.loops[0], *entity.loops[1]};
        ccu::LoopGroup group(ctx.groupCopySize.parallelParam,
            offsetConfig, loopCount, loops);
    }
    return CCU_SUCCESS;
}

CcuResult StartDirectLocalCopy(AllGatherContext &ctx)
{
    CCU_IF(ctx.directLocalCopyBytes != 0) {
        CCU_CHECK(ccu::LocalCopy(
            ctx.localDestination, ctx.localSource,
            ctx.directLocalCopyBytes,
            ctx.directLocalCopyCompletion, 1));
    }
    return CCU_SUCCESS;
}

CcuResult WaitDirectLocalCopy(AllGatherContext &ctx)
{
    CCU_IF(ctx.directLocalCopyBytes != 0) {
        CCU_CHECK(ccu::EventWait(
            ctx.directLocalCopyCompletion, 1));
    }
    return CCU_SUCCESS;
}

CcuResult InitGroupBroadcastResources(AllGatherContext &ctx,
    GroupBroadcastVariables &variables)
{
    constexpr uint32_t eventCount = GROUP_BROADCAST_LOOP_COUNT;
    constexpr uint32_t bufferCount =
        GROUP_BROADCAST_LOOP_COUNT *
        GROUP_BROADCAST_MS_INTERLEAVE;
    constexpr uint32_t bodyCount = 2;

    if (!ctx.groupCopyResourceAllocated) {
        ctx.groupCopyResource.completedEvents =
            ccu::Array<ccu::Event>(eventCount);
        ctx.groupCopyResource.buffers =
            ccu::Array<ccu::CcuBuffer>(bufferCount);
        ctx.groupCopyResourceAllocated = true;
    }

    for (uint32_t index = 0; index < bodyCount; ++index) {
        variables.remoteDestinations[index].resize(
            ctx.broadcastChannelIndices.size());
    }

    constexpr char loopName[] = "allgather_group_broadcast";
    if (ctx.loopEntities.count(loopName) != 0) {
        return CCU_SUCCESS;
    }

    ctx.loopEntities.emplace(loopName, LoopEntity{});
    LoopEntity &entity = ctx.loopEntities[loopName];
    uint16_t completionMask = static_cast<uint16_t>(
        (1U << ctx.broadcastChannelIndices.size()) - 1);
    if (ctx.arg->copyLocal) {
        completionMask = static_cast<uint16_t>(
            completionMask |
            (1U << ctx.broadcastChannelIndices.size()));
    }

    for (uint32_t index = 0; index < bodyCount; ++index) {
        uint32_t bufferBase =
            index * GROUP_BROADCAST_MS_INTERLEAVE;
        ccu::Event completion =
            ctx.groupCopyResource.completedEvents[index];
        entity.bodies[index].reset(new ccu::Func(
            [&ctx, &variables, index, bufferBase, completion,
                completionMask]() {
                (void)ccu::LocalCopy(
                    ctx.groupCopyResource.buffers[bufferBase],
                    variables.sources[index], variables.lengths[index],
                    completion, 1);
                (void)ccu::EventWait(completion, 1);
                for (uint32_t broadcastIndex = 0;
                     broadcastIndex <
                        ctx.broadcastChannelIndices.size();
                     ++broadcastIndex) {
                    uint32_t channelIndex =
                        ctx.broadcastChannelIndices[broadcastIndex];
                    (void)ccu::Write(ctx.arg->channels[channelIndex],
                        variables.remoteDestinations[index][broadcastIndex],
                        ctx.groupCopyResource.buffers[bufferBase],
                        variables.lengths[index], completion,
                        static_cast<uint16_t>(1U << broadcastIndex));
                }
                if (ctx.arg->copyLocal) {
                    (void)ccu::LocalCopy(
                        variables.localDestinations[index],
                        ctx.groupCopyResource.buffers[bufferBase],
                        variables.lengths[index], completion,
                        static_cast<uint16_t>(
                            1U <<
                                ctx.broadcastChannelIndices.size()));
                }
                (void)ccu::EventWait(completion, completionMask);
            }));
        entity.loops[index].reset(new ccu::Loop(
            entity.loopParams[index], *entity.bodies[index]));
    }
    return CCU_SUCCESS;
}

CcuResult GroupBroadcast(AllGatherContext &ctx)
{
    constexpr uint64_t copySlice =
        CCU_MS_SIZE * GROUP_BROADCAST_MS_PER_LOOP;
    constexpr uint64_t loopStride =
        copySlice * GROUP_BROADCAST_LOOP_COUNT;

    GroupBroadcastVariables variables;
    CCU_CHECK(InitGroupBroadcastResources(ctx, variables));
    LoopEntity &entity =
        ctx.loopEntities["allgather_group_broadcast"];

    ccu::LocalAddr source = ctx.source;
    ccu::LocalAddr localDestination = ctx.localDestination;
    std::vector<ccu::RemoteAddr> remoteDestinations =
        ctx.remoteDestinations;

    CCU_IF(ctx.groupCopySize.addrOffset != 0) {
        ccu::Variable loopParam;
        loopParam = GetLoopParam(0, loopStride, 0);
        loopParam += ctx.groupCopySize.loopParam;

        ccu::Variable sliceSize;
        sliceSize = copySlice;
        variables.sources[0].addr = source.addr;
        variables.sources[0].token = source.token;
        if (ctx.arg->copyLocal) {
            variables.localDestinations[0].addr =
                localDestination.addr;
            variables.localDestinations[0].token =
                localDestination.token;
        }
        for (uint32_t broadcastIndex = 0;
             broadcastIndex < ctx.broadcastChannelIndices.size();
             ++broadcastIndex) {
            uint32_t channelIndex =
                ctx.broadcastChannelIndices[broadcastIndex];
            variables.remoteDestinations[0][broadcastIndex].addr =
                remoteDestinations[channelIndex].addr;
            variables.remoteDestinations[0][broadcastIndex].token =
                remoteDestinations[channelIndex].token;
        }
        variables.lengths[0] = sliceSize;
        entity.loopParams[0] = loopParam;

        ccu::Variable parallelConfig;
        parallelConfig =
            GetParallelParam(
                GROUP_BROADCAST_LOOP_COUNT - 1, 0, 1);
        ccu::Variable offsetConfig;
        offsetConfig =
            GetOffsetParam(
                copySlice, GROUP_BROADCAST_MS_INTERLEAVE, 1);
        std::vector<ccu::Loop> loops{*entity.loops[0]};
        ccu::LoopGroup group(
            parallelConfig, offsetConfig,
            GROUP_BROADCAST_LOOP_COUNT, loops);
    }

    CCU_IF(ctx.groupCopySize.parallelParam != 0) {
        source.addr += ctx.groupCopySize.addrOffset;
        if (ctx.arg->copyLocal) {
            localDestination.addr += ctx.groupCopySize.addrOffset;
        }
        for (uint32_t broadcastIndex = 0;
             broadcastIndex < ctx.broadcastChannelIndices.size();
             ++broadcastIndex) {
            uint32_t channelIndex =
                ctx.broadcastChannelIndices[broadcastIndex];
            remoteDestinations[channelIndex].addr +=
                ctx.groupCopySize.addrOffset;
        }

        variables.sources[0].addr = source.addr;
        variables.sources[0].token = source.token;
        if (ctx.arg->copyLocal) {
            variables.localDestinations[0].addr =
                localDestination.addr;
            variables.localDestinations[0].token =
                localDestination.token;
        }
        for (uint32_t broadcastIndex = 0;
             broadcastIndex < ctx.broadcastChannelIndices.size();
             ++broadcastIndex) {
            uint32_t channelIndex =
                ctx.broadcastChannelIndices[broadcastIndex];
            variables.remoteDestinations[0][broadcastIndex].addr =
                remoteDestinations[channelIndex].addr;
            variables.remoteDestinations[0][broadcastIndex].token =
                remoteDestinations[channelIndex].token;
        }
        variables.lengths[0] = ctx.groupCopySize.residual;

        source.addr += ctx.groupCopySize.residual;
        if (ctx.arg->copyLocal) {
            localDestination.addr += ctx.groupCopySize.residual;
        }
        for (uint32_t broadcastIndex = 0;
             broadcastIndex < ctx.broadcastChannelIndices.size();
             ++broadcastIndex) {
            uint32_t channelIndex =
                ctx.broadcastChannelIndices[broadcastIndex];
            remoteDestinations[channelIndex].addr +=
                ctx.groupCopySize.residual;
        }

        ccu::Variable sliceSize;
        sliceSize = copySlice;
        variables.sources[1].addr = source.addr;
        variables.sources[1].token = source.token;
        if (ctx.arg->copyLocal) {
            variables.localDestinations[1].addr =
                localDestination.addr;
            variables.localDestinations[1].token =
                localDestination.token;
        }
        for (uint32_t broadcastIndex = 0;
             broadcastIndex < ctx.broadcastChannelIndices.size();
             ++broadcastIndex) {
            uint32_t channelIndex =
                ctx.broadcastChannelIndices[broadcastIndex];
            variables.remoteDestinations[1][broadcastIndex].addr =
                remoteDestinations[channelIndex].addr;
            variables.remoteDestinations[1][broadcastIndex].token =
                remoteDestinations[channelIndex].token;
        }
        variables.lengths[1] = sliceSize;

        ccu::Variable firstLoopConfig;
        firstLoopConfig = GetLoopParam(0, 0, 1);
        ccu::Variable secondLoopConfig;
        secondLoopConfig = GetLoopParam(0, 0, 1);
        entity.loopParams[0] = firstLoopConfig;
        entity.loopParams[1] = secondLoopConfig;

        ccu::Variable offsetConfig;
        offsetConfig =
            GetOffsetParam(
                copySlice, GROUP_BROADCAST_MS_INTERLEAVE, 1);
        std::vector<ccu::Loop> loops{
            *entity.loops[0], *entity.loops[1]};
        ccu::LoopGroup group(ctx.groupCopySize.parallelParam,
            offsetConfig, GROUP_BROADCAST_LOOP_COUNT, loops);
    }
    return CCU_SUCCESS;
}

CcuResult PrepareRank16MixedChannels(AllGatherContext &ctx)
{
    for (uint32_t channelIndex = 0;
         channelIndex < ctx.arg->channelCount; ++channelIndex) {
        CCU_CHECK(PrepareRemoteChannel(
            ctx, channelIndex, false));
    }
    return CCU_SUCCESS;
}

CcuResult ReuseRemoteChannels(AllGatherContext &ctx)
{
    for (uint32_t channelIndex = 0;
         channelIndex < ctx.arg->channelCount; ++channelIndex) {
        ctx.remoteDestinations[channelIndex].addr =
            ctx.remoteOutputs[channelIndex];
        ctx.remoteDestinations[channelIndex].addr +=
            ctx.outputOffset;
        ctx.remoteDestinations[channelIndex].token =
            ctx.remoteOutputTokens[channelIndex];
    }
    return CCU_SUCCESS;
}

CcuResult PublishRank16PhaseOneResources(
    AllGatherContext &ctx)
{
    for (uint32_t channelIndex = 0;
         channelIndex < ctx.arg->channelCount; ++channelIndex) {
        CCU_CHECK(PublishRemoteChannelResources(
            ctx, channelIndex, PHASE_ONE_OUTPUT_SYNC_ID,
            PHASE_ONE_TOKEN_SYNC_ID));
    }
    return CCU_SUCCESS;
}

CcuResult PrepareRank16PhaseOneChannels(AllGatherContext &ctx)
{
    for (uint32_t channelIndex = 0;
         channelIndex < ctx.arg->channelCount; ++channelIndex) {
        CCU_CHECK(PrepareRemoteChannel(
            ctx, channelIndex, false,
            PHASE_ONE_OUTPUT_SYNC_ID,
            PHASE_ONE_TOKEN_SYNC_ID));
    }
    return CCU_SUCCESS;
}

CcuResult PostRank16MixedChannels(AllGatherContext &ctx)
{
    for (uint32_t channelIndex = 0;
         channelIndex < ctx.arg->channelCount; ++channelIndex) {
        CCU_CHECK(ccu::NotifyRecord(
            ctx.arg->channels[channelIndex], CKE_INDEX,
            1U << POST_SYNC_ID));
    }
    return CCU_SUCCESS;
}

CcuResult RunRank16MixedMeshPhaseZero(AllGatherContext &ctx)
{
    ctx.sliceBytes = ctx.directBytes;
    ctx.sliceBytes += ctx.relayBytes;

    uint16_t directCompletionMask = 0;
    for (uint32_t channelIndex = 0;
         channelIndex < ctx.arg->channelCount; ++channelIndex) {
        uint16_t completionBit = static_cast<uint16_t>(
            1U << channelIndex);
        directCompletionMask = static_cast<uint16_t>(
            directCompletionMask | completionBit);
        CCU_CHECK(ccu::Write(
            ctx.arg->channels[channelIndex],
            ctx.remoteDestinations[channelIndex],
            ctx.source, ctx.sliceBytes,
            ctx.completion, completionBit));
    }
    CCU_CHECK(GroupCopy(ctx));
    CCU_CHECK(ccu::EventWait(
        ctx.completion, directCompletionMask));
    return CCU_SUCCESS;
}

CcuResult RunRank16MixedMeshPhaseOne(AllGatherContext &ctx)
{
    ccu::LocalAddr relaySource;
    relaySource.addr = ctx.localOutput;
    relaySource.addr += ctx.relayOutputOffset;
    relaySource.token = ctx.localOutputToken;

    uint16_t completionMask = 0;
    for (uint32_t channelIndex = 0;
         channelIndex < ctx.arg->channelCount; ++channelIndex) {
        ccu::RemoteAddr relayDestination;
        relayDestination.addr =
            ctx.remoteOutputs[channelIndex];
        relayDestination.addr += ctx.relayOutputOffset;
        relayDestination.token =
            ctx.remoteOutputTokens[channelIndex];
        uint16_t completionBit = static_cast<uint16_t>(
            1U << channelIndex);
        completionMask = static_cast<uint16_t>(
            completionMask | completionBit);
        CCU_CHECK(ccu::Write(
            ctx.arg->channels[channelIndex],
            relayDestination, relaySource, ctx.relayBytes,
            ctx.secondCompletion, completionBit));
    }
    CCU_CHECK(ccu::EventWait(
        ctx.secondCompletion, completionMask));
    return PostRank16MixedChannels(ctx);
}

CcuResult RunRank16MixedClosPhaseZero(AllGatherContext &ctx)
{
    uint32_t relayChannelIndex =
        ctx.arg->relayChannelIndex;
    ccu::LocalAddr relaySource;
    relaySource.addr = ctx.input;
    relaySource.addr += ctx.directBytes;
    relaySource.token = ctx.inputToken;
    ccu::RemoteAddr relayDestination;
    relayDestination.addr =
        ctx.remoteOutputs[relayChannelIndex];
    relayDestination.addr += ctx.outputOffset;
    relayDestination.addr += ctx.directBytes;
    relayDestination.token =
        ctx.remoteOutputTokens[relayChannelIndex];

    CCU_CHECK(ccu::Write(
        ctx.arg->channels[relayChannelIndex],
        relayDestination, relaySource, ctx.relayBytes,
        ctx.completion, 1));
    CCU_CHECK(PublishRank16PhaseOneResources(ctx));
    CCU_CHECK(ccu::EventWait(ctx.completion, 1));
    CCU_CHECK(ccu::NotifyRecord(
        ctx.arg->channels[relayChannelIndex],
        CKE_INDEX, 1U << RELAY_READY_SYNC_ID));
    CCU_CHECK(ccu::NotifyWait(
        ctx.arg->channels[relayChannelIndex],
        CKE_INDEX, 1U << RELAY_READY_SYNC_ID));
    return CCU_SUCCESS;
}

CcuResult RunRank16MixedClosPhaseOne(AllGatherContext &ctx)
{
    CCU_CHECK(GroupBroadcast(ctx));
    return PostRank16MixedChannels(ctx);
}

CcuResult RunRank16Mixed(AllGatherContext &ctx)
{
    if (ctx.arg->isRank16MeshKernel) {
        ctx.localCopyOffset = 0;
    }
    CCU_CHECK(InitTransferAddresses(ctx));
    if (ctx.arg->isRank16MeshKernel) {
        CCU_IF(ctx.rank16Phase == 0) {
            CCU_CHECK(PublishRemoteResources(ctx));
            CCU_CHECK(PrepareRank16MixedChannels(ctx));
            CCU_CHECK(RunRank16MixedMeshPhaseZero(ctx));
        } CCU_ELSE {
            CCU_CHECK(RunRank16MixedMeshPhaseOne(ctx));
        }
        return CCU_SUCCESS;
    }
    CCU_IF(ctx.rank16Phase == 0) {
        CCU_CHECK(PublishRemoteChannelResources(
            ctx, ctx.arg->relayChannelIndex));
        CCU_CHECK(PrepareRemoteChannel(
            ctx, ctx.arg->relayChannelIndex, false));
        CCU_CHECK(RunRank16MixedClosPhaseZero(ctx));
    } CCU_ELSE {
        CCU_CHECK(PrepareRank16PhaseOneChannels(ctx));
        CCU_CHECK(RunRank16MixedClosPhaseOne(ctx));
    }
    return CCU_SUCCESS;
}

CcuResult PrepareRank12SplitChannels(AllGatherContext &ctx)
{
    for (uint32_t channelIndex = 0;
         channelIndex < ctx.arg->channelCount; ++channelIndex) {
        CCU_CHECK(PrepareRemoteChannel(
            ctx, channelIndex, false));
    }
    return CCU_SUCCESS;
}

CcuResult PostRank12SplitChannels(AllGatherContext &ctx)
{
    for (uint32_t channelIndex = 0;
         channelIndex < ctx.arg->channelCount; ++channelIndex) {
        CCU_CHECK(ccu::NotifyRecord(
            ctx.arg->channels[channelIndex], CKE_INDEX,
            1U << POST_SYNC_ID));
    }
    return CCU_SUCCESS;
}

CcuResult RunRank12MeshPhaseZero(AllGatherContext &ctx)
{
    uint16_t completionMask = static_cast<uint16_t>(
        (1U << ctx.arg->channelCount) - 1);
    CCU_IF(ctx.sliceBytes != 0) {
        for (uint32_t channelIndex = 0;
             channelIndex < ctx.arg->channelCount;
             ++channelIndex) {
            uint16_t completionBit =
                static_cast<uint16_t>(
                    1U << channelIndex);
            CCU_CHECK(ccu::Write(
                ctx.arg->channels[channelIndex],
                ctx.remoteDestinations[channelIndex],
                ctx.source, ctx.sliceBytes,
                ctx.completion, completionBit));
        }
        CCU_CHECK(ccu::EventWait(
            ctx.completion, completionMask));
    }
    return CCU_SUCCESS;
}

CcuResult RunRank12MeshPhaseOne(AllGatherContext &ctx)
{
    uint32_t transferCount =
        ctx.arg->rank12FanoutSourceCount *
        ctx.arg->channelCount;
    uint16_t completionMask =
        transferCount == 0 ? 0 :
            static_cast<uint16_t>(
                (1U << transferCount) - 1);

    CCU_IF(ctx.sliceBytes != 0) {
        for (uint32_t sourceIndex = 0;
             sourceIndex <
                 ctx.arg->rank12FanoutSourceCount;
             ++sourceIndex) {
            ccu::LocalAddr fanoutSource;
            fanoutSource.addr = ctx.localOutput;
            fanoutSource.addr +=
                ctx.fanoutOutputOffsets[sourceIndex];
            fanoutSource.token = ctx.localOutputToken;
            for (uint32_t channelIndex = 0;
                 channelIndex < ctx.arg->channelCount;
                 ++channelIndex) {
                ccu::RemoteAddr fanoutDestination;
                fanoutDestination.addr =
                    ctx.remoteOutputs[channelIndex];
                fanoutDestination.addr +=
                    ctx.fanoutOutputOffsets[sourceIndex];
                fanoutDestination.token =
                    ctx.remoteOutputTokens[channelIndex];
                uint16_t completionBit =
                    static_cast<uint16_t>(
                        1U << (sourceIndex *
                            ctx.arg->channelCount +
                            channelIndex));
                CCU_CHECK(ccu::Write(
                    ctx.arg->channels[channelIndex],
                    fanoutDestination, fanoutSource,
                    ctx.sliceBytes,
                    ctx.secondCompletion,
                    completionBit));
            }
        }
        if (transferCount != 0) {
            CCU_CHECK(ccu::EventWait(
                ctx.secondCompletion, completionMask));
        }
    }
    return PostRank12SplitChannels(ctx);
}

CcuResult RunRank12ClosPhaseZero(AllGatherContext &ctx)
{
    uint32_t targetChannel =
        ctx.arg->rank12BridgeTargetChannelIndex;
    uint16_t directCompletionMask = 0;
    for (uint32_t channelIndex = 0;
         channelIndex < ctx.arg->channelCount;
         ++channelIndex) {
        uint16_t completionBit =
            static_cast<uint16_t>(
                1U << channelIndex);
        directCompletionMask = static_cast<uint16_t>(
            directCompletionMask | completionBit);
        CCU_CHECK(ccu::Write(
            ctx.arg->channels[channelIndex],
            ctx.remoteDestinations[channelIndex],
            ctx.source, ctx.directBytes,
            ctx.completion, completionBit));
    }

    ccu::LocalAddr relaySource = ctx.source;
    relaySource.addr += ctx.directBytes;
    ccu::RemoteAddr relayDestination =
        ctx.remoteDestinations[targetChannel];
    relayDestination.addr += ctx.directBytes;
    constexpr uint16_t relayCompletionBit = 1;
    CCU_CHECK(ccu::Write(
        ctx.arg->channels[targetChannel],
        relayDestination, relaySource,
        ctx.relayBytes, ctx.secondCompletion,
        relayCompletionBit));

    CCU_CHECK(ccu::EventWait(
        ctx.secondCompletion, relayCompletionBit));
    CCU_CHECK(ccu::NotifyRecord(
        ctx.arg->channels[targetChannel],
        CKE_INDEX,
        1U << RANK12_PHASE_ZERO_SYNC_ID));
    CCU_CHECK(GroupCopy(ctx));
    CCU_CHECK(ccu::EventWait(
        ctx.completion, directCompletionMask));

    for (uint32_t sourceIndex = 0;
         sourceIndex < ctx.arg->rank12FanoutSourceCount;
         ++sourceIndex) {
        uint32_t sourceChannel =
            ctx.arg->rank12FanoutSourceChannelIndices[
                sourceIndex];
        CCU_CHECK(ccu::NotifyWait(
            ctx.arg->channels[sourceChannel],
            CKE_INDEX,
            1U << RANK12_PHASE_ZERO_SYNC_ID));
    }
    return CCU_SUCCESS;
}

CcuResult RunRank12Split(AllGatherContext &ctx)
{
    CCU_IF(ctx.rank12Phase == 0) {
        if (!ctx.arg->isRank12MeshKernel) {
            ctx.localCopyOffset = 0;
        }
        CCU_CHECK(InitTransferAddresses(ctx));
        CCU_CHECK(PublishRemoteResources(ctx));
        CCU_CHECK(PrepareRank12SplitChannels(ctx));
        if (ctx.arg->isRank12MeshKernel) {
            CCU_CHECK(RunRank12MeshPhaseZero(ctx));
        } else {
            CCU_CHECK(RunRank12ClosPhaseZero(ctx));
        }
    } CCU_ELSE {
        if (ctx.arg->isRank12MeshKernel) {
            CCU_CHECK(RunRank12MeshPhaseOne(ctx));
        } else {
            CCU_CHECK(PostRank12SplitChannels(ctx));
        }
    }
    return CCU_SUCCESS;
}

CcuResult PrepareRank12NhrChannels(AllGatherContext &ctx)
{
    uint32_t readyMask =
        (1U << OUTPUT_XN_ID) |
        (1U << OUTPUT_TOKEN_XN_ID);
    ctx.remoteDestinations.resize(ctx.arg->channelCount);
    for (uint32_t channelIndex = 0;
         channelIndex < ctx.arg->channelCount; ++channelIndex) {
        CCU_CHECK(ccu::NotifyWait(
            ctx.arg->channels[channelIndex],
            CKE_INDEX, readyMask));
        ctx.remoteDestinations[channelIndex].addr =
            ctx.remoteOutputs[channelIndex];
        ctx.remoteDestinations[channelIndex].token =
            ctx.remoteOutputTokens[channelIndex];
    }
    return CCU_SUCCESS;
}

CcuResult StartRank12NhrLocalCopy(AllGatherContext &ctx)
{
    ctx.localSource.addr = ctx.input;
    ctx.localSource.addr += ctx.localCopyOffset;
    ctx.localSource.token = ctx.inputToken;

    ctx.localDestination.addr = ctx.localOutput;
    for (uint32_t rank = 0; rank < ctx.arg->rankId; ++rank) {
        ctx.localDestination.addr += ctx.outputOffset;
    }
    ctx.localDestination.addr += ctx.localCopyOffset;
    ctx.localDestination.token = ctx.localOutputToken;

    return StartDirectLocalCopy(ctx);
}

CcuResult RunRank12NhrStep(
    AllGatherContext &ctx, uint32_t step)
{
    uint32_t toChannelIndex =
        ctx.arg->nhrToChannelIndices[step];
    uint32_t fromChannelIndex =
        ctx.arg->nhrFromChannelIndices[step];
    uint32_t sliceCount =
        ctx.arg->nhrStepSliceCounts[step];
    uint16_t completionMask = static_cast<uint16_t>(
        (1U << sliceCount) - 1);

    for (uint32_t index = 0; index < sliceCount; ++index) {
        uint32_t sliceIndex =
            ctx.arg->nhrStepSliceIndices[step][index];
        ccu::LocalAddr source;
        if (sliceIndex == ctx.arg->rankId) {
            source.addr = ctx.input;
            source.addr += ctx.localCopyOffset;
            source.token = ctx.inputToken;
        } else {
            source.addr = ctx.localOutput;
            for (uint32_t rank = 0; rank < sliceIndex; ++rank) {
                source.addr += ctx.outputOffset;
            }
            source.addr += ctx.localCopyOffset;
            source.token = ctx.localOutputToken;
        }

        ccu::RemoteAddr destination =
            ctx.remoteDestinations[toChannelIndex];
        for (uint32_t rank = 0; rank < sliceIndex; ++rank) {
            destination.addr += ctx.outputOffset;
        }
        destination.addr += ctx.localCopyOffset;

        uint16_t completionBit =
            static_cast<uint16_t>(1U << index);
        CCU_CHECK(ccu::Write(
            ctx.arg->channels[toChannelIndex],
            destination, source, ctx.sliceBytes,
            ctx.completion, completionBit));
    }
    CCU_CHECK(ccu::EventWait(
        ctx.completion, completionMask));

    if (step + 1 != ctx.arg->nhrStepCount) {
        CCU_CHECK(ccu::NotifyRecord(
            ctx.arg->channels[toChannelIndex],
            CKE_INDEX, 1U << NHR_STEP_SYNC_ID));
        CCU_CHECK(ccu::NotifyWait(
            ctx.arg->channels[fromChannelIndex],
            CKE_INDEX, 1U << NHR_STEP_SYNC_ID));
    }
    return CCU_SUCCESS;
}

CcuResult RunRank12Nhr(AllGatherContext &ctx)
{
    CCU_CHECK(PublishRemoteResources(ctx));
    CCU_CHECK(PrepareRank12NhrChannels(ctx));
    CCU_CHECK(StartRank12NhrLocalCopy(ctx));

    for (uint32_t step = 0;
         step < ctx.arg->nhrStepCount; ++step) {
        CCU_CHECK(RunRank12NhrStep(ctx, step));
    }
    CCU_CHECK(WaitDirectLocalCopy(ctx));

    for (uint32_t channelIndex = 0;
         channelIndex < ctx.arg->channelCount; ++channelIndex) {
        CCU_CHECK(ccu::NotifyRecord(
            ctx.arg->channels[channelIndex],
            CKE_INDEX, 1U << POST_SYNC_ID));
    }
    return CCU_SUCCESS;
}

CcuResult RunAllGather(AllGatherContext &ctx)
{
    CCU_CHECK(InitTransferAddresses(ctx));
    CCU_CHECK(IssueAllGather(ctx));
    CCU_CHECK(IssueSecondDirectSlice(ctx));

    if (ctx.arg->useGroupBroadcast) {
        CCU_CHECK(GroupBroadcast(ctx));
    } else if (ctx.arg->useDirectLocalCopy) {
        CCU_CHECK(StartDirectLocalCopy(ctx));
    } else if (ctx.arg->copyLocal) {
        CCU_CHECK(GroupCopy(ctx));
    }

    CCU_CHECK(CompleteTransfersAndNotify(ctx));
    if (ctx.arg->useDirectLocalCopy) {
        CCU_CHECK(WaitDirectLocalCopy(ctx));
    }
    return CCU_SUCCESS;
}

CcuResult RunReusedDirectAllGather(AllGatherContext &ctx)
{
    CCU_CHECK(InitTransferAddresses(ctx));
    CCU_CHECK(ReuseRemoteChannels(ctx));
    if (ctx.arg->directMeshFirst) {
        constexpr uint8_t issueOrder[] = {
            CHANNEL_KIND_MESH,
            CHANNEL_KIND_CLOS,
        };
        for (uint8_t channelKind : issueOrder) {
            for (uint32_t channelIndex = 0;
                 channelIndex < ctx.arg->channelCount;
                 ++channelIndex) {
                if (ctx.arg->channelKinds[channelIndex] !=
                    channelKind) {
                    continue;
                }
                uint16_t completionBit =
                    static_cast<uint16_t>(
                        1U << channelIndex);
                CCU_CHECK(ccu::Write(
                    ctx.arg->channels[channelIndex],
                    ctx.remoteDestinations[channelIndex],
                    ctx.source, ctx.sliceBytes,
                    ctx.completion, completionBit));
            }
        }
    } else {
        for (uint32_t channelIndex = 0;
             channelIndex < ctx.arg->channelCount;
             ++channelIndex) {
            uint16_t completionBit =
                static_cast<uint16_t>(1U << channelIndex);
            CCU_CHECK(ccu::Write(
                ctx.arg->channels[channelIndex],
                ctx.remoteDestinations[channelIndex],
                ctx.source, ctx.sliceBytes,
                ctx.completion, completionBit));
        }
    }
    if (ctx.arg->copyLocal) {
        CCU_CHECK(GroupCopy(ctx));
    }
    return CompleteTransfersAndNotify(ctx);
}

CcuResult WaitForPeers(AllGatherContext &ctx)
{
    constexpr uint8_t waitOrder[] = {
        CHANNEL_KIND_MESH,
        CHANNEL_KIND_CLOS,
    };
    for (uint8_t channelKind : waitOrder) {
        for (uint32_t channelIndex = 0;
             channelIndex < ctx.arg->channelCount; ++channelIndex) {
            if (ctx.arg->channelKinds[channelIndex] != channelKind) {
                continue;
            }
            CCU_CHECK(ccu::NotifyWait(
                ctx.arg->channels[channelIndex],
                CKE_INDEX, 1U << POST_SYNC_ID));
        }
    }
    return CCU_SUCCESS;
}
} // namespace

CcuResult CcuKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllGather *>(arg);
    AllGatherContext ctx;
    ctx.arg = kernelArg;

    CCU_CHECK(InitResources(ctx));
    CCU_CHECK(LoadTaskArgs(ctx));
    if (ctx.arg->useRank12Nhr) {
        CCU_CHECK(RunRank12Nhr(ctx));
        CCU_CHECK(WaitForPeers(ctx));
    } else if (ctx.arg->useRank12SplitRelay) {
        CCU_CHECK(RunRank12Split(ctx));
        CCU_IF(ctx.rank12Phase != 0) {
            CCU_CHECK(WaitForPeers(ctx));
        }
    } else if (ctx.arg->useRank16MixedRelay) {
        CCU_CHECK(RunRank16Mixed(ctx));
        CCU_IF(ctx.rank16Phase != 0) {
            CCU_CHECK(WaitForPeers(ctx));
        }
    } else if (ctx.arg->useRank12DirectReuse) {
        CCU_IF(ctx.outputOffset ==
            ctx.arg->rank12FirstOutputOffset) {
            CCU_CHECK(PublishRemoteResources(ctx));
            CCU_CHECK(RunAllGather(ctx));
        } CCU_ELSE {
            CCU_CHECK(RunReusedDirectAllGather(ctx));
        }
        CCU_CHECK(WaitForPeers(ctx));
    } else if (ctx.arg->useRank4DirectReuse) {
        CCU_IF(ctx.outputOffset ==
            ctx.arg->rank4FirstOutputOffset) {
            CCU_CHECK(PublishRemoteResources(ctx));
            CCU_CHECK(RunAllGather(ctx));
        } CCU_ELSE {
            CCU_CHECK(RunReusedDirectAllGather(ctx));
        }
        CCU_CHECK(WaitForPeers(ctx));
    } else {
        CCU_CHECK(PublishRemoteResources(ctx));
        CCU_CHECK(RunAllGather(ctx));
        CCU_CHECK(WaitForPeers(ctx));
    }
    return CCU_SUCCESS;
}

} // namespace ops_hccl

#undef CCU_CHECK
