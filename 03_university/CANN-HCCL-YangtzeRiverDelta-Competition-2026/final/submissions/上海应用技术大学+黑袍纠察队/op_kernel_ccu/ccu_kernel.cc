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

#include <ccu/ccu_event.hpp>
#include <ccu/ccu_primitives.hpp>
#include <ccu/ccu_variable.hpp>

#include "ccu_kernel.h"
#include "log.h"

#define CCU_CHK_RET(call) \
    do { \
        const CcuResult ccuResult = (call); \
        if (ccuResult != CCU_SUCCESS) { \
            HCCL_ERROR("CCU call failed, result[%d]", ccuResult); \
            return ccuResult; \
        } \
    } while (0)

namespace ccu = ::AscendC::ccu;

namespace ops_hccl {
namespace {
    constexpr uint32_t BUFFER_XN_ID = 0;
    constexpr uint32_t TOKEN_XN_ID = 1;
    constexpr uint32_t CKE_INDEX = 0;
    constexpr uint32_t DATA_READY_ID = 2;
    constexpr uint32_t FINAL_SYNC_ID = 3;
    constexpr uint32_t PIPELINE_BLOCK_COUNT = 8;
    constexpr uint32_t PIPELINE_READY_ID_BASE = 2;
    constexpr uint32_t PIPELINE_FINAL_SYNC_ID = PIPELINE_READY_ID_BASE + PIPELINE_BLOCK_COUNT;

    struct BroadcastContext {
        const CcuKernelArgBroadcast *arg = nullptr;
        ccu::Variable myBuffer;
        ccu::Variable myToken;
        ccu::Variable dataSize;
        ccu::Variable normalSliceSize;
        ccu::Variable lastSliceSize;
        ccu::Variable normalBlockSize;
        ccu::Variable normalLastBlockSize;
        ccu::Variable lastBlockSize;
        ccu::Variable lastLastBlockSize;
        std::vector<ccu::Variable> remoteBuffers;
        std::vector<ccu::Variable> remoteTokens;
        std::vector<ccu::RemoteAddr> remoteDestinations;
        ccu::Event event;
    };

    uint16_t ChannelMask(uint32_t channelIndex)
    {
        return static_cast<uint16_t>(uint32_t{1} << channelIndex);
    }

    uint16_t AllChannelMask(uint32_t channelCount)
    {
        return static_cast<uint16_t>((uint32_t{1} << channelCount) - 1U);
    }

    CcuResult InitResource(BroadcastContext &ctx, CcuKernelArg arg)
    {
        ctx.arg = static_cast<CcuKernelArgBroadcast *>(arg);
        if (ctx.arg == nullptr || ctx.arg->rankSize < 2 || ctx.arg->rankSize > MAX_RANK_SIZE
            || ctx.arg->rankId >= ctx.arg->rankSize || ctx.arg->root >= ctx.arg->rankSize || ctx.arg->channelCount == 0
            || ctx.arg->channelCount >= MAX_RANK_SIZE) {
            HCCL_ERROR("Invalid CCU broadcast resource");
            return CCU_E_INTERNAL;
        }

        ctx.remoteBuffers.reserve(ctx.arg->channelCount);
        ctx.remoteTokens.reserve(ctx.arg->channelCount);
        ctx.remoteDestinations.resize(ctx.arg->channelCount);
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            if (ctx.arg->peerRanks[channelIndex] >= ctx.arg->rankSize
                || ctx.arg->peerRanks[channelIndex] == ctx.arg->rankId
                || (channelIndex > 0 && ctx.arg->peerRanks[channelIndex] <= ctx.arg->peerRanks[channelIndex - 1])) {
                HCCL_ERROR("Invalid CCU broadcast peer");
                return CCU_E_INTERNAL;
            }
            ctx.remoteBuffers.push_back(
                ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIndex], BUFFER_XN_ID));
            ctx.remoteTokens.push_back(
                ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIndex], TOKEN_XN_ID));
        }
        return CCU_SUCCESS;
    }

    CcuResult LoadCommonArgs(BroadcastContext &ctx, uint32_t &argIndex)
    {
        CCU_CHK_RET(ccu::LoadArg(ctx.myBuffer, argIndex++));
        CCU_CHK_RET(ccu::LoadArg(ctx.myToken, argIndex++));
        return CCU_SUCCESS;
    }

    CcuResult LoadSliceArgs(BroadcastContext &ctx)
    {
        uint32_t argIndex = 0;
        CCU_CHK_RET(LoadCommonArgs(ctx, argIndex));
        CCU_CHK_RET(ccu::LoadArg(ctx.normalSliceSize, argIndex++));
        CCU_CHK_RET(ccu::LoadArg(ctx.lastSliceSize, argIndex++));
        return CCU_SUCCESS;
    }

    CcuResult LoadPipelineArgs(BroadcastContext &ctx)
    {
        uint32_t argIndex = 0;
        CCU_CHK_RET(LoadCommonArgs(ctx, argIndex));
        CCU_CHK_RET(ccu::LoadArg(ctx.normalSliceSize, argIndex++));
        CCU_CHK_RET(ccu::LoadArg(ctx.lastSliceSize, argIndex++));
        CCU_CHK_RET(ccu::LoadArg(ctx.normalBlockSize, argIndex++));
        CCU_CHK_RET(ccu::LoadArg(ctx.normalLastBlockSize, argIndex++));
        CCU_CHK_RET(ccu::LoadArg(ctx.lastBlockSize, argIndex++));
        CCU_CHK_RET(ccu::LoadArg(ctx.lastLastBlockSize, argIndex++));
        return CCU_SUCCESS;
    }

    int32_t FindRootChannel(const CcuKernelArgBroadcast &arg)
    {
        for (uint32_t channelIndex = 0; channelIndex < arg.channelCount; ++channelIndex) {
            if (arg.peerRanks[channelIndex] == arg.root) {
                return static_cast<int32_t>(channelIndex);
            }
        }
        return -1;
    }

    CcuResult WaitRemoteVariables(BroadcastContext &ctx)
    {
        const uint16_t variableMask = (1U << BUFFER_XN_ID) | (1U << TOKEN_XN_ID);
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIndex], CKE_INDEX, variableMask));
        }
        return CCU_SUCCESS;
    }

    CcuResult SendVariables(BroadcastContext &ctx, uint32_t channelIndex)
    {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            ctx.arg->channels[channelIndex], ctx.myBuffer, BUFFER_XN_ID, CKE_INDEX, 1U << BUFFER_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(
            ctx.arg->channels[channelIndex], ctx.myToken, TOKEN_XN_ID, CKE_INDEX, 1U << TOKEN_XN_ID));
        return CCU_SUCCESS;
    }

    CcuResult RunDirectRoot(BroadcastContext &ctx)
    {
        CCU_CHK_RET(WaitRemoteVariables(ctx));

        ccu::LocalAddr source;
        source.addr = ctx.myBuffer;
        source.token = ctx.myToken;
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            ctx.remoteDestinations[channelIndex].addr = ctx.remoteBuffers[channelIndex];
            ctx.remoteDestinations[channelIndex].token = ctx.remoteTokens[channelIndex];
            CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIndex], ctx.remoteDestinations[channelIndex], source,
                ctx.dataSize, ctx.event, ChannelMask(channelIndex)));
        }
        CCU_CHK_RET(ccu::EventWait(ctx.event, AllChannelMask(ctx.arg->channelCount)));

        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[channelIndex], CKE_INDEX, 1U << DATA_READY_ID));
        }
        return CCU_SUCCESS;
    }

    CcuResult RunDirectPeer(BroadcastContext &ctx)
    {
        const int32_t rootChannel = FindRootChannel(*ctx.arg);
        if (rootChannel < 0) {
            return CCU_SUCCESS;
        }

        const uint32_t channelIndex = static_cast<uint32_t>(rootChannel);
        CCU_CHK_RET(SendVariables(ctx, channelIndex));
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIndex], CKE_INDEX, 1U << DATA_READY_ID));
        return CCU_SUCCESS;
    }

    uint32_t PeerOwnerIndex(uint32_t rank, uint32_t root)
    {
        return rank < root ? rank : rank - 1;
    }

    bool IsLastOwner(const BroadcastContext &ctx, uint32_t rank)
    {
        return PeerOwnerIndex(rank, ctx.arg->root) + 1 == ctx.arg->rankSize - 1;
    }

    bool UsePeerOwnedSlices(const BroadcastContext &ctx)
    {
        return ctx.arg->rankSize == 12;
    }

    CcuResult ExecuteScatterRoot(BroadcastContext &ctx)
    {
        ccu::Variable sliceOffset;
        sliceOffset = 0;
        uint32_t offsetIndex = 0;
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            const uint32_t targetRank = ctx.arg->peerRanks[channelIndex];
            const uint32_t targetIndex
                = UsePeerOwnedSlices(ctx) ? PeerOwnerIndex(targetRank, ctx.arg->root) : targetRank;
            while (offsetIndex < targetIndex) {
                sliceOffset += ctx.normalSliceSize;
                ++offsetIndex;
            }

            ccu::LocalAddr source;
            source.addr = ctx.myBuffer;
            source.addr += sliceOffset;
            source.token = ctx.myToken;
            ctx.remoteDestinations[channelIndex].addr = ctx.remoteBuffers[channelIndex];
            ctx.remoteDestinations[channelIndex].addr += sliceOffset;
            ctx.remoteDestinations[channelIndex].token = ctx.remoteTokens[channelIndex];

            const bool isLastSlice
                = UsePeerOwnedSlices(ctx) ? IsLastOwner(ctx, targetRank) : targetRank + 1 == ctx.arg->rankSize;
            auto &sliceSize = isLastSlice ? ctx.lastSliceSize : ctx.normalSliceSize;
            CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIndex], ctx.remoteDestinations[channelIndex], source,
                sliceSize, ctx.event, ChannelMask(channelIndex)));
        }
        CCU_CHK_RET(ccu::EventWait(ctx.event, AllChannelMask(ctx.arg->channelCount)));

        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[channelIndex], CKE_INDEX, 1U << DATA_READY_ID));
        }
        return CCU_SUCCESS;
    }

    CcuResult RunScatterRoot(BroadcastContext &ctx)
    {
        CCU_CHK_RET(WaitRemoteVariables(ctx));
        CCU_CHK_RET(ExecuteScatterRoot(ctx));
        return CCU_SUCCESS;
    }

    CcuResult WaitScatterPeer(BroadcastContext &ctx)
    {
        const int32_t rootChannel = FindRootChannel(*ctx.arg);
        if (rootChannel < 0) {
            return CCU_SUCCESS;
        }

        const uint32_t channelIndex = static_cast<uint32_t>(rootChannel);
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIndex], CKE_INDEX, 1U << DATA_READY_ID));
        return CCU_SUCCESS;
    }

    CcuResult RunScatterPeer(BroadcastContext &ctx)
    {
        const int32_t rootChannel = FindRootChannel(*ctx.arg);
        if (rootChannel < 0) {
            return CCU_SUCCESS;
        }

        const uint32_t channelIndex = static_cast<uint32_t>(rootChannel);
        CCU_CHK_RET(SendVariables(ctx, channelIndex));
        CCU_CHK_RET(WaitScatterPeer(ctx));
        return CCU_SUCCESS;
    }

    CcuResult AllToAllPreSync(BroadcastContext &ctx)
    {
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            CCU_CHK_RET(SendVariables(ctx, channelIndex));
        }
        CCU_CHK_RET(WaitRemoteVariables(ctx));
        return CCU_SUCCESS;
    }

    CcuResult RunAllGather(BroadcastContext &ctx)
    {
        const bool usePeerOwnedSlices = UsePeerOwnedSlices(ctx);
        if (usePeerOwnedSlices && ctx.arg->rankId == ctx.arg->root) {
            for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
                CCU_CHK_RET(ccu::EventRecord(ctx.event, ChannelMask(channelIndex)));
            }
            CCU_CHK_RET(ccu::EventWait(ctx.event, AllChannelMask(ctx.arg->channelCount)));
            return CCU_SUCCESS;
        }

        const uint32_t ownerIndex
            = usePeerOwnedSlices ? PeerOwnerIndex(ctx.arg->rankId, ctx.arg->root) : ctx.arg->rankId;
        ccu::Variable sliceOffset;
        sliceOffset = 0;
        for (uint32_t rank = 0; rank < ownerIndex; ++rank) {
            sliceOffset += ctx.normalSliceSize;
        }

        ccu::LocalAddr source;
        source.addr = ctx.myBuffer;
        source.addr += sliceOffset;
        source.token = ctx.myToken;
        const bool isLastSlice = usePeerOwnedSlices ? IsLastOwner(ctx, ctx.arg->rankId)
                                                    : ctx.arg->rankId + 1 == ctx.arg->rankSize;
        auto &sliceSize = isLastSlice ? ctx.lastSliceSize : ctx.normalSliceSize;

        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            const uint32_t targetRank = ctx.arg->peerRanks[channelIndex];
            if (targetRank == ctx.arg->root && ctx.arg->rankId != ctx.arg->root) {
                CCU_CHK_RET(ccu::EventRecord(ctx.event, ChannelMask(channelIndex)));
                continue;
            }

            ctx.remoteDestinations[channelIndex].addr = ctx.remoteBuffers[channelIndex];
            ctx.remoteDestinations[channelIndex].addr += sliceOffset;
            ctx.remoteDestinations[channelIndex].token = ctx.remoteTokens[channelIndex];
            CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIndex], ctx.remoteDestinations[channelIndex], source,
                sliceSize, ctx.event, ChannelMask(channelIndex)));
        }
        CCU_CHK_RET(ccu::EventWait(ctx.event, AllChannelMask(ctx.arg->channelCount)));
        return CCU_SUCCESS;
    }

    CcuResult FinalSync(BroadcastContext &ctx)
    {
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[channelIndex], CKE_INDEX, 1U << FINAL_SYNC_ID));
        }
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIndex], CKE_INDEX, 1U << FINAL_SYNC_ID));
        }
        return CCU_SUCCESS;
    }

    CcuResult PipelinePreSync(BroadcastContext &ctx)
    {
        if (ctx.arg->rankId != ctx.arg->root) {
            for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
                CCU_CHK_RET(SendVariables(ctx, channelIndex));
            }
        }

        const uint16_t variableMask = (1U << BUFFER_XN_ID) | (1U << TOKEN_XN_ID);
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            if (ctx.arg->rankId != ctx.arg->root && ctx.arg->peerRanks[channelIndex] == ctx.arg->root) {
                continue;
            }
            CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIndex], CKE_INDEX, variableMask));
        }
        return CCU_SUCCESS;
    }

    CcuResult ScatterPipelineBlock(BroadcastContext &ctx, uint32_t readyId, ccu::Variable &normalBlockOffset,
        ccu::Variable &lastBlockOffset, ccu::Variable &normalTransferSize, ccu::Variable &lastTransferSize)
    {
        ccu::Variable sliceOffset;
        sliceOffset = 0;
        uint32_t offsetOwner = 0;
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            const uint32_t targetRank = ctx.arg->peerRanks[channelIndex];
            const uint32_t targetOwner = PeerOwnerIndex(targetRank, ctx.arg->root);
            while (offsetOwner < targetOwner) {
                sliceOffset += ctx.normalSliceSize;
                ++offsetOwner;
            }

            const bool isLastOwner = IsLastOwner(ctx, targetRank);
            ccu::LocalAddr source;
            source.addr = ctx.myBuffer;
            source.addr += sliceOffset;
            source.addr += isLastOwner ? lastBlockOffset : normalBlockOffset;
            source.token = ctx.myToken;
            ctx.remoteDestinations[channelIndex].addr = ctx.remoteBuffers[channelIndex];
            ctx.remoteDestinations[channelIndex].addr += sliceOffset;
            ctx.remoteDestinations[channelIndex].addr += isLastOwner ? lastBlockOffset : normalBlockOffset;
            ctx.remoteDestinations[channelIndex].token = ctx.remoteTokens[channelIndex];

            auto &transferSize = isLastOwner ? lastTransferSize : normalTransferSize;
            CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIndex], ctx.remoteDestinations[channelIndex], source,
                transferSize, ctx.event, ChannelMask(channelIndex)));
        }
        CCU_CHK_RET(ccu::EventWait(ctx.event, AllChannelMask(ctx.arg->channelCount)));
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[channelIndex], CKE_INDEX, 1U << readyId));
        }
        return CCU_SUCCESS;
    }

    CcuResult AllGatherPipelineBlock(BroadcastContext &ctx, ccu::Variable &normalBlockOffset,
        ccu::Variable &lastBlockOffset, ccu::Variable &normalTransferSize, ccu::Variable &lastTransferSize)
    {
        const uint32_t ownerIndex = PeerOwnerIndex(ctx.arg->rankId, ctx.arg->root);
        ccu::Variable sliceOffset;
        sliceOffset = 0;
        for (uint32_t index = 0; index < ownerIndex; ++index) {
            sliceOffset += ctx.normalSliceSize;
        }

        const bool isLastOwner = IsLastOwner(ctx, ctx.arg->rankId);
        ccu::LocalAddr source;
        source.addr = ctx.myBuffer;
        source.addr += sliceOffset;
        source.addr += isLastOwner ? lastBlockOffset : normalBlockOffset;
        source.token = ctx.myToken;
        auto &transferSize = isLastOwner ? lastTransferSize : normalTransferSize;

        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            const uint32_t targetRank = ctx.arg->peerRanks[channelIndex];
            if (targetRank == ctx.arg->root) {
                CCU_CHK_RET(ccu::EventRecord(ctx.event, ChannelMask(channelIndex)));
                continue;
            }
            ctx.remoteDestinations[channelIndex].addr = ctx.remoteBuffers[channelIndex];
            ctx.remoteDestinations[channelIndex].addr += sliceOffset;
            ctx.remoteDestinations[channelIndex].addr += isLastOwner ? lastBlockOffset : normalBlockOffset;
            ctx.remoteDestinations[channelIndex].token = ctx.remoteTokens[channelIndex];
            CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIndex], ctx.remoteDestinations[channelIndex], source,
                transferSize, ctx.event, ChannelMask(channelIndex)));
        }
        CCU_CHK_RET(ccu::EventWait(ctx.event, AllChannelMask(ctx.arg->channelCount)));
        return CCU_SUCCESS;
    }

    CcuResult RunPipelinedTwoShot(BroadcastContext &ctx)
    {
        CCU_CHK_RET(PipelinePreSync(ctx));
        const int32_t rootChannel = FindRootChannel(*ctx.arg);
        if (ctx.arg->rankId != ctx.arg->root && rootChannel < 0) {
            HCCL_ERROR("Pipelined two-shot kernel does not contain the root channel");
            return CCU_E_INTERNAL;
        }

        ccu::Variable normalBlockOffset;
        ccu::Variable lastBlockOffset;
        normalBlockOffset = 0;
        lastBlockOffset = 0;
        for (uint32_t blockIndex = 0; blockIndex < PIPELINE_BLOCK_COUNT; ++blockIndex) {
            auto &normalTransferSize
                = blockIndex + 1 == PIPELINE_BLOCK_COUNT ? ctx.normalLastBlockSize : ctx.normalBlockSize;
            auto &lastTransferSize = blockIndex + 1 == PIPELINE_BLOCK_COUNT ? ctx.lastLastBlockSize : ctx.lastBlockSize;
            const uint32_t readyId = PIPELINE_READY_ID_BASE + blockIndex;
            if (ctx.arg->rankId == ctx.arg->root) {
                CCU_CHK_RET(ScatterPipelineBlock(
                    ctx, readyId, normalBlockOffset, lastBlockOffset, normalTransferSize, lastTransferSize));
            } else {
                CCU_CHK_RET(
                    ccu::NotifyWait(ctx.arg->channels[static_cast<uint32_t>(rootChannel)], CKE_INDEX, 1U << readyId));
                CCU_CHK_RET(AllGatherPipelineBlock(
                    ctx, normalBlockOffset, lastBlockOffset, normalTransferSize, lastTransferSize));
            }
            normalBlockOffset += normalTransferSize;
            lastBlockOffset += lastTransferSize;
        }

        if (ctx.arg->rankId != ctx.arg->root) {
            for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
                CCU_CHK_RET(
                    ccu::NotifyRecord(ctx.arg->channels[channelIndex], CKE_INDEX, 1U << PIPELINE_FINAL_SYNC_ID));
            }
        }
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            if (ctx.arg->rankId != ctx.arg->root && ctx.arg->peerRanks[channelIndex] == ctx.arg->root) {
                continue;
            }
            CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIndex], CKE_INDEX, 1U << PIPELINE_FINAL_SYNC_ID));
        }
        return CCU_SUCCESS;
    }
} // namespace

CcuResult CcuBroadcastDirectKernel(CcuKernelArg arg)
{
    BroadcastContext ctx;
    CCU_CHK_RET(InitResource(ctx, arg));
    uint32_t argIndex = 0;
    CCU_CHK_RET(LoadCommonArgs(ctx, argIndex));
    CCU_CHK_RET(ccu::LoadArg(ctx.dataSize, argIndex++));
    return ctx.arg->rankId == ctx.arg->root ? RunDirectRoot(ctx) : RunDirectPeer(ctx);
}

CcuResult CcuBroadcastTwoShotKernel(CcuKernelArg arg)
{
    BroadcastContext ctx;
    CCU_CHK_RET(InitResource(ctx, arg));
    CCU_CHK_RET(LoadPipelineArgs(ctx));
    return RunPipelinedTwoShot(ctx);
}

CcuResult CcuBroadcastScatterKernel(CcuKernelArg arg)
{
    BroadcastContext ctx;
    CCU_CHK_RET(InitResource(ctx, arg));
    CCU_CHK_RET(LoadSliceArgs(ctx));
    return ctx.arg->rankId == ctx.arg->root ? RunScatterRoot(ctx) : RunScatterPeer(ctx);
}

CcuResult CcuBroadcastAllGatherKernel(CcuKernelArg arg)
{
    BroadcastContext ctx;
    CCU_CHK_RET(InitResource(ctx, arg));
    CCU_CHK_RET(LoadSliceArgs(ctx));
    CCU_CHK_RET(AllToAllPreSync(ctx));
    CCU_CHK_RET(RunAllGather(ctx));
    CCU_CHK_RET(FinalSync(ctx));
    return CCU_SUCCESS;
}
} // namespace ops_hccl
