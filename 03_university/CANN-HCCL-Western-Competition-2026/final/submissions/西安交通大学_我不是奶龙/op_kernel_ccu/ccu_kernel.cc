/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>

#include <hcomm/hcomm_primitives.h>

#include "ccu_kernel.h"
#include "log.h"

namespace ops_hccl {
namespace {
namespace ccu = ::AscendC::ccu;

#define F002_CCU_CHK_RET(call) \
    do { \
        CcuResult result = (call); \
        if (result != CCU_SUCCESS) { \
            return result; \
        } \
    } while (0)

constexpr uint32_t REMOTE_CCL_ADDR_VARIABLE = 0;
constexpr uint32_t REMOTE_CCL_TOKEN_VARIABLE = 1;
constexpr uint32_t REMOTE_INPUT_ADDR_VARIABLE = 0;
constexpr uint32_t REMOTE_INPUT_TOKEN_VARIABLE = 1;
constexpr uint32_t REMOTE_OUTPUT_ADDR_VARIABLE = 2;
constexpr uint32_t REMOTE_OUTPUT_TOKEN_VARIABLE = 3;

constexpr uint32_t INPUT_READY_NOTIFY = 0;
constexpr uint32_t DATA_COMPLETE_NOTIFY = 1;

constexpr uint16_t CCL_ADDR_MASK = 1U << 0;
constexpr uint16_t CCL_TOKEN_MASK = 1U << 1;
constexpr uint16_t INPUT_READY_MASK = CCL_ADDR_MASK | CCL_TOKEN_MASK;
constexpr uint16_t DATA_COMPLETE_MASK = 1U << 0;
constexpr uint16_t DIRECT_INPUT_ADDR_MASK = 1U << 0;
constexpr uint16_t DIRECT_INPUT_TOKEN_MASK = 1U << 1;
constexpr uint16_t DIRECT_OUTPUT_ADDR_MASK = 1U << 2;
constexpr uint16_t DIRECT_OUTPUT_TOKEN_MASK = 1U << 3;
constexpr uint16_t DIRECT_BUFFER_READY_MASK = DIRECT_INPUT_ADDR_MASK |
    DIRECT_INPUT_TOKEN_MASK | DIRECT_OUTPUT_ADDR_MASK | DIRECT_OUTPUT_TOKEN_MASK;
constexpr uint16_t SERIAL_EVENT_MASK = 1U << 0;
constexpr uint32_t RS_WAVEFRONT_STRIPE_NUM = 4;
constexpr uint32_t DIRECT_BUFFER_RS_STRIPE_NUM = 16;
constexpr uint32_t ASYMMETRIC_DIRECT_BUFFER_RS_STRIPE_NUM = 8;
constexpr uint32_t DIRECT_BUFFER_MAX_ACTIVE_NUM = 8;
constexpr uint32_t ASYMMETRIC_RANK_SIZE = 12;
constexpr uint32_t FOUR_RANK_NHR_SIZE = 4;
constexpr uint16_t FOUR_RANK_PAIR_EVENT_MASK = (1U << 2) - 1U;
constexpr uint16_t NHR_RS0_DONE_MASK = 1U << 2;
constexpr uint16_t NHR_RS1_READY_MASK = 1U << 3;
constexpr uint16_t NHR_RS1_DONE_MASK = 1U << 4;
constexpr uint16_t NHR_AG0_DONE_MASK = 1U << 5;
constexpr uint16_t NHR_AG1_DONE_MASK = 1U << 6;
constexpr uint32_t RANK16_CLOS_SIZE = 16;
constexpr uint32_t RANK16_SERVER_SIZE = 8;
constexpr uint32_t RANK16_CLOS_CHANNEL_NUM = 8;
constexpr uint32_t RANK16_RSAG_CCL_ADDR_VARIABLE = 0;
constexpr uint32_t RANK16_RSAG_CCL_TOKEN_VARIABLE = 1;
constexpr uint32_t RANK16_RSAG_OUTPUT_ADDR_VARIABLE = 2;
constexpr uint32_t RANK16_RSAG_OUTPUT_TOKEN_VARIABLE = 3;
constexpr uint16_t RANK16_RSAG_CCL_ADDR_MASK = 1U << 0;
constexpr uint16_t RANK16_RSAG_CCL_TOKEN_MASK = 1U << 1;
constexpr uint16_t RANK16_RSAG_BUFFER_READY_MASK =
    RANK16_RSAG_CCL_ADDR_MASK | RANK16_RSAG_CCL_TOKEN_MASK;
constexpr uint16_t RANK16_RSAG_PARTIAL_READY_MASK = 1U << 2;
constexpr uint16_t RANK16_RSAG_OUTPUT_ADDR_MASK = 1U << 3;
constexpr uint16_t RANK16_RSAG_OUTPUT_TOKEN_MASK = 1U << 4;
constexpr uint16_t RANK16_RSAG_REMOTE_OUTPUT_READY_MASK =
    RANK16_RSAG_BUFFER_READY_MASK | RANK16_RSAG_OUTPUT_ADDR_MASK |
    RANK16_RSAG_OUTPUT_TOKEN_MASK;
constexpr uint16_t RANK16_RSAG_FOUR_LANE_MASK = (1U << 4) - 1U;
constexpr uint16_t RANK16_RSAG_TWO_LANE_MASK = (1U << 2) - 1U;
constexpr uint16_t RANK16_RSAG_AG_EVENT_MASK = (1U << 7) - 1U;
constexpr uint64_t RANK16_RSAG_OWNER_SLICE_BYTES =
    (512ULL * 1024) / RANK16_SERVER_SIZE;
constexpr uint16_t RANK16_FINAL_FENCE_MASK = 1U << 0;
// F097 changes only the issue order of the seven exact-512KiB AllGather
// writes.  Each slot is a K8,8 matching; opposite directions use different
// slots so no endpoint receives a seven-way burst from the source loop order.
constexpr std::array<uint32_t, RANK16_SERVER_SIZE - 1>
    F097_RANK16_AG_LOW_OFFSETS = {{1, 2, 3, 4, 5, 6, 7}};
constexpr std::array<uint32_t, RANK16_SERVER_SIZE - 1>
    F097_RANK16_AG_HIGH_OFFSETS = {{1, 3, 2, 7, 6, 5, 4}};
constexpr uint32_t RANK12_CLOS_SIZE = 12;
constexpr uint32_t RANK12_LARGE_SERVER_SIZE = 8;
constexpr uint32_t RANK12_SMALL_SERVER_SIZE = 4;
constexpr uint32_t RANK12_PAIR_RELAY_NUM = 6;
constexpr uint64_t RANK12_SLICE_BYTES =
    (512ULL * 1024) / RANK12_SMALL_SERVER_SIZE;
constexpr uint32_t RANK12_SOURCE_ADDR_VARIABLE = 0;
constexpr uint32_t RANK12_SOURCE_TOKEN_VARIABLE = 1;
constexpr uint32_t RANK12_OUTPUT_ADDR_VARIABLE = 2;
constexpr uint32_t RANK12_OUTPUT_TOKEN_VARIABLE = 3;
constexpr uint16_t RANK12_SOURCE_ADDR_MASK = 1U << 0;
constexpr uint16_t RANK12_SOURCE_TOKEN_MASK = 1U << 1;
constexpr uint16_t RANK12_OUTPUT_ADDR_MASK = 1U << 2;
constexpr uint16_t RANK12_OUTPUT_TOKEN_MASK = 1U << 3;
constexpr uint16_t RANK12_SOURCE_READY_MASK = RANK12_SOURCE_ADDR_MASK |
    RANK12_SOURCE_TOKEN_MASK;
constexpr uint16_t RANK12_OUTPUT_READY_MASK = RANK12_OUTPUT_ADDR_MASK |
    RANK12_OUTPUT_TOKEN_MASK;
// F085 keeps the F080 full-push metadata/event masks private to the exact
// rank12/512KiB entry.  The F082 shared Clos helper and every other route keep
// their original constants and task graph byte-for-byte.
constexpr uint16_t F085_RANK12_ALL_BUFFER_READY_MASK =
    RANK12_SOURCE_READY_MASK | RANK12_OUTPUT_READY_MASK;
constexpr uint16_t RANK12_TWO_WAY_EVENT_MASK = (1U << 2) - 1U;
constexpr uint16_t RANK12_THREE_WAY_EVENT_MASK = (1U << 3) - 1U;
constexpr uint16_t RANK12_FOUR_WAY_EVENT_MASK = (1U << 4) - 1U;
constexpr uint16_t RANK12_FIVE_WAY_EVENT_MASK = (1U << 5) - 1U;
constexpr uint16_t F085_RANK12_EIGHT_WAY_EVENT_MASK = (1U << 8) - 1U;
constexpr std::array<std::array<uint32_t, 2>, RANK12_PAIR_RELAY_NUM>
    RANK12_PAIR_ROOTS = {{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}};

struct DirectKernelContext {
    const CcuKernelArgGroup *arg = nullptr;

    ccu::Variable localCclAddr;
    ccu::Variable localCclToken;
    ccu::Variable dataBytes;
    ccu::Variable slotStrideBytes;
    ccu::Variable mySliceOffset;
    ccu::Variable mySliceBytes;
    ccu::Variable resultOffsetBytes;
    ccu::Variable phase;
};

struct FourRankNhrContext {
    const CcuKernelArgGroup *arg = nullptr;

    ccu::Variable inputAddr;
    ccu::Variable inputToken;
    ccu::Variable outputAddr;
    ccu::Variable outputToken;
    ccu::Variable regularSliceBytes;
    ccu::Variable lastSliceBytes;
    ccu::Variable isInplace;
};

struct DirectBufferContext {
    const CcuKernelArgGroup *arg = nullptr;

    ccu::Variable inputAddr;
    ccu::Variable inputToken;
    ccu::Variable outputAddr;
    ccu::Variable outputToken;
    ccu::Variable scratchAddr;
    ccu::Variable scratchToken;
    ccu::Variable regularSliceBytes;
    ccu::Variable lastSliceBytes;
    ccu::Variable regularStripeBytes;
    ccu::Variable tailStripeBytes;
    ccu::Variable phase;
};

struct Rank12ClosTreeContext {
    const CcuKernelArgGroup *arg = nullptr;

    ccu::Variable sourceAddr;
    ccu::Variable sourceToken;
    ccu::Variable outputAddr;
    ccu::Variable outputToken;
    ccu::Variable scratchAddr;
    ccu::Variable scratchToken;
    ccu::Variable dataBytes;
};

CcuResult LoadTaskArgs(DirectKernelContext &ctx)
{
    uint32_t argId = 0;
    F002_CCU_CHK_RET(ccu::LoadArg(ctx.localCclAddr, argId++));
    F002_CCU_CHK_RET(ccu::LoadArg(ctx.localCclToken, argId++));
    F002_CCU_CHK_RET(ccu::LoadArg(ctx.dataBytes, argId++));
    F002_CCU_CHK_RET(ccu::LoadArg(ctx.slotStrideBytes, argId++));
    F002_CCU_CHK_RET(ccu::LoadArg(ctx.mySliceOffset, argId++));
    F002_CCU_CHK_RET(ccu::LoadArg(ctx.mySliceBytes, argId++));
    F002_CCU_CHK_RET(ccu::LoadArg(ctx.resultOffsetBytes, argId++));
    F002_CCU_CHK_RET(ccu::LoadArg(ctx.phase, argId++));
    return CCU_SUCCESS;
}

CcuResult InitChannelResources(DirectKernelContext &ctx)
{
    if (ctx.arg == nullptr || ctx.arg->channelCount == 0 ||
        ctx.arg->channelCount >= MAX_RANK_SIZE || ctx.arg->rankSize <= 1 ||
        ctx.arg->rankSize > MAX_RANK_SIZE || ctx.arg->myRank >= ctx.arg->rankSize ||
        ctx.arg->groupIndex >= ctx.arg->groupCount) {
        HCCL_ERROR("F002 grouped kernel has invalid static metadata");
        return CCU_E_PARA;
    }

    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        if (ctx.arg->channels[index] == 0 ||
            ctx.arg->remoteRanks[index] >= ctx.arg->rankSize ||
            ctx.arg->remoteRanks[index] == ctx.arg->myRank) {
            HCCL_ERROR("F002 grouped kernel contains invalid peer metadata");
            return CCU_E_PARA;
        }
    }
    return CCU_SUCCESS;
}

CcuResult LoadFourRankNhrArgs(FourRankNhrContext &ctx)
{
    uint32_t argId = 0;
    F002_CCU_CHK_RET(ccu::LoadArg(ctx.inputAddr, argId++));
    F002_CCU_CHK_RET(ccu::LoadArg(ctx.inputToken, argId++));
    F002_CCU_CHK_RET(ccu::LoadArg(ctx.outputAddr, argId++));
    F002_CCU_CHK_RET(ccu::LoadArg(ctx.outputToken, argId++));
    F002_CCU_CHK_RET(ccu::LoadArg(ctx.regularSliceBytes, argId++));
    F002_CCU_CHK_RET(ccu::LoadArg(ctx.lastSliceBytes, argId++));
    F002_CCU_CHK_RET(ccu::LoadArg(ctx.isInplace, argId++));
    return CCU_SUCCESS;
}

CcuResult LoadDirectBufferArgs(DirectBufferContext &ctx)
{
    uint32_t argId = 0;
    F002_CCU_CHK_RET(ccu::LoadArg(ctx.inputAddr, argId++));
    F002_CCU_CHK_RET(ccu::LoadArg(ctx.inputToken, argId++));
    F002_CCU_CHK_RET(ccu::LoadArg(ctx.outputAddr, argId++));
    F002_CCU_CHK_RET(ccu::LoadArg(ctx.outputToken, argId++));
    F002_CCU_CHK_RET(ccu::LoadArg(ctx.scratchAddr, argId++));
    F002_CCU_CHK_RET(ccu::LoadArg(ctx.scratchToken, argId++));
    F002_CCU_CHK_RET(ccu::LoadArg(ctx.regularSliceBytes, argId++));
    F002_CCU_CHK_RET(ccu::LoadArg(ctx.lastSliceBytes, argId++));
    F002_CCU_CHK_RET(ccu::LoadArg(ctx.regularStripeBytes, argId++));
    F002_CCU_CHK_RET(ccu::LoadArg(ctx.tailStripeBytes, argId++));
    F002_CCU_CHK_RET(ccu::LoadArg(ctx.phase, argId++));
    return CCU_SUCCESS;
}

CcuResult LoadRank12ClosTreeArgs(Rank12ClosTreeContext &ctx)
{
    uint32_t argId = 0;
    F002_CCU_CHK_RET(ccu::LoadArg(ctx.sourceAddr, argId++));
    F002_CCU_CHK_RET(ccu::LoadArg(ctx.sourceToken, argId++));
    F002_CCU_CHK_RET(ccu::LoadArg(ctx.outputAddr, argId++));
    F002_CCU_CHK_RET(ccu::LoadArg(ctx.outputToken, argId++));
    F002_CCU_CHK_RET(ccu::LoadArg(ctx.scratchAddr, argId++));
    F002_CCU_CHK_RET(ccu::LoadArg(ctx.scratchToken, argId++));
    F002_CCU_CHK_RET(ccu::LoadArg(ctx.dataBytes, argId++));
    return CCU_SUCCESS;
}

CcuResult InitRank12ClosTreeResources(Rank12ClosTreeContext &ctx)
{
    if (ctx.arg == nullptr || ctx.arg->rankSize != RANK12_CLOS_SIZE ||
        ctx.arg->myRank >= RANK12_CLOS_SIZE ||
        ctx.arg->groupCount != 2 ||
        ctx.arg->groupIndex >= ctx.arg->groupCount) {
        HCCL_ERROR("F031 rank-12 Clos tree has invalid static metadata");
        return CCU_E_PARA;
    }

    bool onLargeServer = ctx.arg->myRank < RANK12_LARGE_SERVER_SIZE;
    uint32_t expectedChannelCount = onLargeServer ?
        RANK12_SMALL_SERVER_SIZE : RANK12_LARGE_SERVER_SIZE;
    uint32_t firstRemoteRank = onLargeServer ?
        RANK12_LARGE_SERVER_SIZE : 0;
    if (ctx.arg->channelCount != expectedChannelCount) {
        HCCL_ERROR("F031 rank-12 Clos tree has invalid channel count");
        return CCU_E_PARA;
    }

    std::array<bool, RANK12_LARGE_SERVER_SIZE> seen{};
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        uint32_t remoteRank = ctx.arg->remoteRanks[index];
        if (ctx.arg->channels[index] == 0 ||
            remoteRank < firstRemoteRank ||
            remoteRank >= firstRemoteRank + expectedChannelCount ||
            seen[remoteRank - firstRemoteRank]) {
            HCCL_ERROR("F031 rank-12 Clos tree contains invalid peer metadata");
            return CCU_E_PARA;
        }
        seen[remoteRank - firstRemoteRank] = true;
    }
    return CCU_SUCCESS;
}

CcuResult InitFourRankNhrResources(FourRankNhrContext &ctx)
{
    if (ctx.arg == nullptr || ctx.arg->rankSize != FOUR_RANK_NHR_SIZE ||
        ctx.arg->myRank >= FOUR_RANK_NHR_SIZE || ctx.arg->channelCount != FOUR_RANK_NHR_SIZE - 1 ||
        ctx.arg->groupIndex != 0 || ctx.arg->groupCount != 1) {
        HCCL_ERROR("F015 4-rank NHR has invalid static metadata");
        return CCU_E_PARA;
    }

    std::array<bool, FOUR_RANK_NHR_SIZE> seen{};
    seen[ctx.arg->myRank] = true;
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        uint32_t remoteRank = ctx.arg->remoteRanks[index];
        if (ctx.arg->channels[index] == 0 || remoteRank >= FOUR_RANK_NHR_SIZE || seen[remoteRank]) {
            HCCL_ERROR("F015 4-rank NHR contains invalid peer metadata");
            return CCU_E_PARA;
        }
        seen[remoteRank] = true;
    }
    for (bool rankSeen : seen) {
        if (!rankSeen) {
            HCCL_ERROR("F015 4-rank NHR does not cover every peer");
            return CCU_E_PARA;
        }
    }
    return CCU_SUCCESS;
}

CcuResult InitDirectBufferResources(DirectBufferContext &ctx)
{
    if (ctx.arg == nullptr) {
        HCCL_ERROR("F015 direct-buffer kernel has null static metadata");
        return CCU_E_PARA;
    }

    bool supportedRankSize = ctx.arg->rankSize == ASYMMETRIC_RANK_SIZE ||
        ctx.arg->rankSize == MAX_RANK_SIZE;
    bool supportedChannelCount =
        (ctx.arg->rankSize == MAX_RANK_SIZE &&
            (ctx.arg->channelCount == 7 || ctx.arg->channelCount == 8)) ||
        (ctx.arg->rankSize == ASYMMETRIC_RANK_SIZE &&
            ctx.arg->channelCount >= 3 &&
            ctx.arg->channelCount <= DIRECT_BUFFER_MAX_ACTIVE_NUM);
    if (!supportedRankSize || !supportedChannelCount ||
        ctx.arg->myRank >= ctx.arg->rankSize || ctx.arg->groupCount != 2 ||
        ctx.arg->groupIndex >= ctx.arg->groupCount) {
        HCCL_ERROR("F015 direct-buffer kernel has invalid static metadata");
        return CCU_E_PARA;
    }

    uint32_t previousRank = 0;
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        uint32_t remoteRank = ctx.arg->remoteRanks[index];
        if (ctx.arg->channels[index] == 0 || remoteRank >= ctx.arg->rankSize ||
            remoteRank == ctx.arg->myRank || (index > 0 && remoteRank <= previousRank)) {
            HCCL_ERROR("F015 direct-buffer kernel contains invalid peer metadata");
            return CCU_E_PARA;
        }
        previousRank = remoteRank;
    }
    return CCU_SUCCESS;
}

void AddDirectBufferSliceOffset(
    ccu::Address &address, ccu::Variable &regularSliceBytes, uint32_t slice)
{
    for (uint32_t repeat = 0; repeat < slice; ++repeat) {
        address += regularSliceBytes;
    }
}

ccu::Variable &DirectBufferMySliceBytes(DirectBufferContext &ctx)
{
    return ctx.arg->myRank + 1 == ctx.arg->rankSize ?
        ctx.lastSliceBytes : ctx.regularSliceBytes;
}

ccu::Variable &DirectBufferStripeBytes(DirectBufferContext &ctx, uint32_t stripe)
{
    return stripe + 1 == DIRECT_BUFFER_RS_STRIPE_NUM ?
        ctx.tailStripeBytes : ctx.regularStripeBytes;
}

ccu::Variable &AsymmetricDirectBufferStripeBytes(
    DirectBufferContext &ctx, uint32_t stripe)
{
    return stripe + 1 == ASYMMETRIC_DIRECT_BUFFER_RS_STRIPE_NUM ?
        ctx.tailStripeBytes : ctx.regularStripeBytes;
}

CcuResult FindFourRankChannel(
    const FourRankNhrContext &ctx, uint32_t remoteRank, ChannelHandle &channel)
{
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        if (ctx.arg->remoteRanks[index] == remoteRank) {
            channel = ctx.arg->channels[index];
            return CCU_SUCCESS;
        }
    }
    HCCL_ERROR("F015 4-rank NHR cannot find its partner channel");
    return CCU_E_PARA;
}

void AddFourRankSliceOffset(
    ccu::Address &address, ccu::Variable &regularSliceBytes, uint32_t slice)
{
    for (uint32_t repeat = 0; repeat < slice; ++repeat) {
        address += regularSliceBytes;
    }
}

ccu::Variable &FourRankSliceBytes(FourRankNhrContext &ctx, uint32_t slice)
{
    return slice + 1 == FOUR_RANK_NHR_SIZE ? ctx.lastSliceBytes : ctx.regularSliceBytes;
}

CcuResult InitializeFourRankOutput(FourRankNhrContext &ctx)
{
    // RS0 sends the two slices of the opposite parity.  Only the complementary
    // pair can receive a remote reduction, so an out-of-place call copies just
    // that pair to output.  In-place calls are already initialized and the two
    // sets are disjoint, hence no input can be overwritten before it is sent.
    CCU_IF(ctx.isInplace == 0)
    {
        std::array<ccu::LocalAddr, 2> sources;
        std::array<ccu::LocalAddr, 2> destinations;
        ccu::Event copyEvent;
        uint32_t copyIndex = 0;
        for (uint32_t slice = 0; slice < FOUR_RANK_NHR_SIZE; ++slice) {
            if ((slice & 1U) != (ctx.arg->myRank & 1U)) {
                continue;
            }

            sources[copyIndex].addr = ctx.inputAddr;
            sources[copyIndex].token = ctx.inputToken;
            destinations[copyIndex].addr = ctx.outputAddr;
            destinations[copyIndex].token = ctx.outputToken;
            AddFourRankSliceOffset(
                sources[copyIndex].addr, ctx.regularSliceBytes, slice);
            AddFourRankSliceOffset(
                destinations[copyIndex].addr, ctx.regularSliceBytes, slice);
            uint16_t eventMask = static_cast<uint16_t>(1U << copyIndex);
            F002_CCU_CHK_RET(ccu::LocalCopy(destinations[copyIndex], sources[copyIndex],
                FourRankSliceBytes(ctx, slice), copyEvent, eventMask));
            ++copyIndex;
        }
        F002_CCU_CHK_RET(ccu::EventWait(copyEvent,
            static_cast<uint16_t>((1U << copyIndex) - 1U)));
    }
    return CCU_SUCCESS;
}

CcuResult ExchangeFourRankOutput(FourRankNhrContext &ctx)
{
    // Record every edge before waiting on any edge.  This remains deadlock-free
    // even if the rank graph enumerates peer channels in a different order.
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        ChannelHandle channel = ctx.arg->channels[index];
        F002_CCU_CHK_RET(ccu::WriteVariableWithNotify(channel, ctx.outputAddr,
            REMOTE_CCL_ADDR_VARIABLE, INPUT_READY_NOTIFY, CCL_ADDR_MASK));
        F002_CCU_CHK_RET(ccu::WriteVariableWithNotify(channel, ctx.outputToken,
            REMOTE_CCL_TOKEN_VARIABLE, INPUT_READY_NOTIFY, CCL_TOKEN_MASK));
    }
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        F002_CCU_CHK_RET(ccu::NotifyWait(
            ctx.arg->channels[index], INPUT_READY_NOTIFY, INPUT_READY_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult FourRankReduceStep(FourRankNhrContext &ctx, uint32_t sendRank,
    uint32_t receiveRank, const std::array<uint32_t, 2> &slices, uint32_t sliceCount,
    bool sourceIsInput, bool needsReady, uint16_t readyMask, uint16_t doneMask)
{
    ChannelHandle sendChannel = 0;
    ChannelHandle receiveChannel = 0;
    F002_CCU_CHK_RET(FindFourRankChannel(ctx, sendRank, sendChannel));
    F002_CCU_CHK_RET(FindFourRankChannel(ctx, receiveRank, receiveChannel));

    if (needsReady) {
        F002_CCU_CHK_RET(ccu::NotifyRecord(
            receiveChannel, INPUT_READY_NOTIFY, readyMask));
        F002_CCU_CHK_RET(ccu::NotifyWait(
            sendChannel, INPUT_READY_NOTIFY, readyMask));
    }

    ccu::Variable remoteOutputAddr =
        ccu::GetResByChannel<ccu::Variable>(sendChannel, REMOTE_CCL_ADDR_VARIABLE);
    ccu::Variable remoteOutputToken =
        ccu::GetResByChannel<ccu::Variable>(sendChannel, REMOTE_CCL_TOKEN_VARIABLE);
    std::array<ccu::LocalAddr, 2> sources;
    std::array<ccu::RemoteAddr, 2> destinations;
    ccu::Event reduceEvent;
    uint16_t eventMask = 0;
    for (uint32_t index = 0; index < sliceCount; ++index) {
        uint32_t slice = slices[index];
        sources[index].addr = sourceIsInput ? ctx.inputAddr : ctx.outputAddr;
        sources[index].token = sourceIsInput ? ctx.inputToken : ctx.outputToken;
        destinations[index].addr = remoteOutputAddr;
        destinations[index].token = remoteOutputToken;
        AddFourRankSliceOffset(sources[index].addr, ctx.regularSliceBytes, slice);
        AddFourRankSliceOffset(destinations[index].addr, ctx.regularSliceBytes, slice);

        uint16_t sliceMask = static_cast<uint16_t>(1U << index);
        eventMask = static_cast<uint16_t>(eventMask | sliceMask);
        F002_CCU_CHK_RET(ccu::WriteReduce(sendChannel, destinations[index], sources[index],
            FourRankSliceBytes(ctx, slice), HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM,
            reduceEvent, sliceMask));
    }
    F002_CCU_CHK_RET(ccu::EventWait(reduceEvent, eventMask));
    F002_CCU_CHK_RET(ccu::NotifyRecord(
        sendChannel, INPUT_READY_NOTIFY, doneMask));
    F002_CCU_CHK_RET(ccu::NotifyWait(
        receiveChannel, INPUT_READY_NOTIFY, doneMask));
    return CCU_SUCCESS;
}

CcuResult FourRankGatherStep(FourRankNhrContext &ctx, uint32_t sendRank,
    uint32_t receiveRank, const std::array<uint32_t, 2> &slices,
    uint32_t sliceCount, uint16_t doneMask)
{
    ChannelHandle sendChannel = 0;
    ChannelHandle receiveChannel = 0;
    F002_CCU_CHK_RET(FindFourRankChannel(ctx, sendRank, sendChannel));
    F002_CCU_CHK_RET(FindFourRankChannel(ctx, receiveRank, receiveChannel));

    ccu::Variable remoteOutputAddr =
        ccu::GetResByChannel<ccu::Variable>(sendChannel, REMOTE_CCL_ADDR_VARIABLE);
    ccu::Variable remoteOutputToken =
        ccu::GetResByChannel<ccu::Variable>(sendChannel, REMOTE_CCL_TOKEN_VARIABLE);
    std::array<ccu::LocalAddr, 2> sources;
    std::array<ccu::RemoteAddr, 2> destinations;
    ccu::Event writeEvent;
    uint16_t eventMask = 0;
    for (uint32_t index = 0; index < sliceCount; ++index) {
        uint32_t slice = slices[index];
        sources[index].addr = ctx.outputAddr;
        sources[index].token = ctx.outputToken;
        destinations[index].addr = remoteOutputAddr;
        destinations[index].token = remoteOutputToken;
        AddFourRankSliceOffset(sources[index].addr, ctx.regularSliceBytes, slice);
        AddFourRankSliceOffset(destinations[index].addr, ctx.regularSliceBytes, slice);

        uint16_t sliceMask = static_cast<uint16_t>(1U << index);
        eventMask = static_cast<uint16_t>(eventMask | sliceMask);
        F002_CCU_CHK_RET(ccu::Write(sendChannel, destinations[index], sources[index],
            FourRankSliceBytes(ctx, slice), writeEvent, sliceMask));
    }
    F002_CCU_CHK_RET(ccu::EventWait(writeEvent, eventMask));
    F002_CCU_CHK_RET(ccu::NotifyRecord(
        sendChannel, INPUT_READY_NOTIFY, doneMask));
    F002_CCU_CHK_RET(ccu::NotifyWait(
        receiveChannel, INPUT_READY_NOTIFY, doneMask));
    return CCU_SUCCESS;
}

CcuResult RunFourRankNhr(FourRankNhrContext &ctx)
{
    uint32_t rank = ctx.arg->myRank;
    uint32_t previous = (rank + FOUR_RANK_NHR_SIZE - 1) % FOUR_RANK_NHR_SIZE;
    uint32_t next = (rank + 1) % FOUR_RANK_NHR_SIZE;
    uint32_t opposite = (rank + 2) % FOUR_RANK_NHR_SIZE;

    F002_CCU_CHK_RET(InitializeFourRankOutput(ctx));
    F002_CCU_CHK_RET(ExchangeFourRankOutput(ctx));

    // NHR ReduceScatter.  RS0 folds adjacent rank pairs; RS1 combines the two
    // fixed partials.  Exactly one writer reaches an owner in each step, so
    // floating-point evaluation order cannot depend on transfer completion.
    std::array<uint32_t, 2> rs0Slices{previous, next};
    F002_CCU_CHK_RET(FourRankReduceStep(ctx, previous, next, rs0Slices, 2,
        true, false, 0, NHR_RS0_DONE_MASK));
    std::array<uint32_t, 2> rs1Slices{opposite, 0};
    F002_CCU_CHK_RET(FourRankReduceStep(ctx, opposite, opposite, rs1Slices, 1,
        false, true, NHR_RS1_READY_MASK, NHR_RS1_DONE_MASK));

    // Reverse the two steps for AllGather.  Each stage waits for the incoming
    // write before its newly-received slice can become a source in the next
    // stage; the final wait also makes every output safe for immediate reuse.
    std::array<uint32_t, 2> ag0Slices{rank, 0};
    F002_CCU_CHK_RET(FourRankGatherStep(ctx, opposite, opposite,
        ag0Slices, 1, NHR_AG0_DONE_MASK));
    std::array<uint32_t, 2> ag1Slices{rank, opposite};
    F002_CCU_CHK_RET(FourRankGatherStep(ctx, next, previous,
        ag1Slices, 2, NHR_AG1_DONE_MASK));
    return CCU_SUCCESS;
}

CcuResult RecordInputReady(DirectKernelContext &ctx)
{
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        ChannelHandle channel = ctx.arg->channels[index];
        F002_CCU_CHK_RET(ccu::WriteVariableWithNotify(channel, ctx.localCclAddr,
            REMOTE_CCL_ADDR_VARIABLE, INPUT_READY_NOTIFY, CCL_ADDR_MASK));
        F002_CCU_CHK_RET(ccu::WriteVariableWithNotify(channel, ctx.localCclToken,
            REMOTE_CCL_TOKEN_VARIABLE, INPUT_READY_NOTIFY, CCL_TOKEN_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult WaitInputReady(DirectKernelContext &ctx)
{
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        F002_CCU_CHK_RET(ccu::NotifyWait(
            ctx.arg->channels[index], INPUT_READY_NOTIFY, INPUT_READY_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult OneShotReadAndRecord(DirectKernelContext &ctx)
{
    F002_CCU_CHK_RET(WaitInputReady(ctx));

    std::array<ccu::LocalAddr, MAX_RANK_SIZE> localSlots;
    std::array<ccu::RemoteAddr, MAX_RANK_SIZE> remoteSlots;
    ccu::Event dataEvent;
    uint16_t allDataMask = 0;
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        ChannelHandle channel = ctx.arg->channels[index];
        uint32_t remoteRank = ctx.arg->remoteRanks[index];
        ccu::Variable remoteCclAddr =
            ccu::GetResByChannel<ccu::Variable>(channel, REMOTE_CCL_ADDR_VARIABLE);
        ccu::Variable remoteCclToken =
            ccu::GetResByChannel<ccu::Variable>(channel, REMOTE_CCL_TOKEN_VARIABLE);

        localSlots[index].addr = ctx.localCclAddr;
        for (uint32_t repeat = 0; repeat < remoteRank; ++repeat) {
            localSlots[index].addr += ctx.slotStrideBytes;
        }
        localSlots[index].token = ctx.localCclToken;
        remoteSlots[index].addr = remoteCclAddr;
        for (uint32_t repeat = 0; repeat < remoteRank; ++repeat) {
            remoteSlots[index].addr += ctx.slotStrideBytes;
        }
        remoteSlots[index].token = remoteCclToken;

        uint16_t eventMask = static_cast<uint16_t>(1U << index);
        allDataMask = static_cast<uint16_t>(allDataMask | eventMask);
        F002_CCU_CHK_RET(ccu::Read(channel, localSlots[index], remoteSlots[index],
            ctx.dataBytes, dataEvent, eventMask));
    }
    F002_CCU_CHK_RET(ccu::EventWait(dataEvent, allDataMask));

    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        F002_CCU_CHK_RET(ccu::NotifyRecord(
            ctx.arg->channels[index], DATA_COMPLETE_NOTIFY, DATA_COMPLETE_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult WaitDataComplete(DirectKernelContext &ctx)
{
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        F002_CCU_CHK_RET(ccu::NotifyWait(
            ctx.arg->channels[index], DATA_COMPLETE_NOTIFY, DATA_COMPLETE_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult FusedOneShotCommunication(DirectKernelContext &ctx)
{
    // Each die records every edge before waiting on any edge. The two kernels
    // run on independent CCU threads, so differing endpoint group order cannot
    // create a cross-die cycle. Every notify bit is produced and consumed once
    // before this mission returns.
    F002_CCU_CHK_RET(RecordInputReady(ctx));
    F002_CCU_CHK_RET(OneShotReadAndRecord(ctx));
    F002_CCU_CHK_RET(WaitDataComplete(ctx));
    return CCU_SUCCESS;
}

CcuResult OneShotLocalReduce(DirectKernelContext &ctx)
{
    ccu::LocalAddr accumulator;
    accumulator.addr = ctx.localCclAddr;
    accumulator.token = ctx.localCclToken;

    ccu::LocalAddr source;
    source.token = ctx.localCclToken;
    ccu::Event reduceEvent;
    for (uint32_t sender = 1; sender < ctx.arg->rankSize; ++sender) {
        source.addr = ctx.localCclAddr;
        for (uint32_t repeat = 0; repeat < sender; ++repeat) {
            source.addr += ctx.slotStrideBytes;
        }
        F002_CCU_CHK_RET(ccu::LocalReduce(accumulator, source, ctx.dataBytes,
            HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, reduceEvent, SERIAL_EVENT_MASK));
        F002_CCU_CHK_RET(ccu::EventWait(reduceEvent, SERIAL_EVENT_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult ValidateFourRankDirectSource(const DirectKernelContext &ctx)
{
    if (ctx.arg == nullptr || ctx.arg->rankSize != FOUR_RANK_NHR_SIZE ||
        ctx.arg->myRank >= FOUR_RANK_NHR_SIZE ||
        ctx.arg->channelCount != FOUR_RANK_NHR_SIZE - 1 ||
        ctx.arg->groupCount != 1 || ctx.arg->groupIndex != 0) {
        HCCL_ERROR("F031 4-rank direct-source kernel has invalid static metadata");
        return CCU_E_PARA;
    }

    std::array<bool, FOUR_RANK_NHR_SIZE> seen{};
    seen[ctx.arg->myRank] = true;
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        uint32_t remoteRank = ctx.arg->remoteRanks[index];
        if (ctx.arg->channels[index] == 0 ||
            remoteRank >= FOUR_RANK_NHR_SIZE || seen[remoteRank]) {
            HCCL_ERROR("F031 4-rank direct-source kernel has an invalid peer set");
            return CCU_E_PARA;
        }
        seen[remoteRank] = true;
    }
    for (bool present : seen) {
        if (!present) {
            HCCL_ERROR("F031 4-rank direct-source kernel does not cover every peer");
            return CCU_E_PARA;
        }
    }
    return CCU_SUCCESS;
}

CcuResult FindGroupedChannel(
    const DirectKernelContext &ctx, uint32_t remoteRank,
    ChannelHandle &channel)
{
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        if (ctx.arg->remoteRanks[index] == remoteRank) {
            channel = ctx.arg->channels[index];
            return CCU_SUCCESS;
        }
    }
    HCCL_ERROR("F031 4-rank direct-source kernel cannot find a peer channel");
    return CCU_E_PARA;
}

ccu::Variable &FourRankSourceAddr(DirectKernelContext &ctx)
{
    return ctx.localCclAddr;
}

ccu::Variable &FourRankSourceToken(DirectKernelContext &ctx)
{
    return ctx.localCclToken;
}

ccu::Variable &FourRankOutputAddr(DirectKernelContext &ctx)
{
    return ctx.dataBytes;
}

ccu::Variable &FourRankOutputToken(DirectKernelContext &ctx)
{
    return ctx.slotStrideBytes;
}

ccu::Variable &FourRankScratchAddr(DirectKernelContext &ctx)
{
    return ctx.mySliceOffset;
}

ccu::Variable &FourRankScratchToken(DirectKernelContext &ctx)
{
    return ctx.mySliceBytes;
}

ccu::Variable &FourRankDataBytes(DirectKernelContext &ctx)
{
    return ctx.resultOffsetBytes;
}

CcuResult RunFourRankPairStage(DirectKernelContext &ctx, bool reduce,
    ccu::Event &stageEvent)
{
    std::array<uint32_t, 2> sourceRanks{
        reduce ? 1U : 0U,
        reduce ? 3U : 2U,
    };
    std::array<ccu::LocalAddr, 2> destinations;
    std::array<ccu::LocalAddr, 2> localSources;
    std::array<ccu::RemoteAddr, 2> remoteSources;
    destinations[0].addr = FourRankOutputAddr(ctx);
    destinations[0].token = FourRankOutputToken(ctx);
    destinations[1].addr = FourRankScratchAddr(ctx);
    destinations[1].token = FourRankScratchToken(ctx);

    for (uint32_t pair = 0; pair < sourceRanks.size(); ++pair) {
        uint32_t sourceRank = sourceRanks[pair];
        uint16_t eventMask = static_cast<uint16_t>(1U << pair);
        if (sourceRank == ctx.arg->myRank) {
            localSources[pair].addr = FourRankSourceAddr(ctx);
            localSources[pair].token = FourRankSourceToken(ctx);
            if (reduce) {
                F002_CCU_CHK_RET(ccu::LocalReduce(destinations[pair],
                    localSources[pair], FourRankDataBytes(ctx),
                    HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM,
                    stageEvent, eventMask));
            } else {
                F002_CCU_CHK_RET(ccu::LocalCopy(destinations[pair],
                    localSources[pair], FourRankDataBytes(ctx),
                    stageEvent, eventMask));
            }
            continue;
        }

        ChannelHandle channel = 0;
        F002_CCU_CHK_RET(
            FindGroupedChannel(ctx, sourceRank, channel));
        remoteSources[pair].addr =
            ccu::GetResByChannel<ccu::Variable>(
                channel, REMOTE_CCL_ADDR_VARIABLE);
        remoteSources[pair].token =
            ccu::GetResByChannel<ccu::Variable>(
                channel, REMOTE_CCL_TOKEN_VARIABLE);
        if (reduce) {
            F002_CCU_CHK_RET(ccu::ReadReduce(channel,
                destinations[pair], remoteSources[pair],
                FourRankDataBytes(ctx), HCCL_DATA_TYPE_FP32,
                HCCL_REDUCE_SUM, stageEvent, eventMask));
        } else {
            F002_CCU_CHK_RET(ccu::Read(channel, destinations[pair],
                remoteSources[pair], FourRankDataBytes(ctx),
                stageEvent, eventMask));
        }
    }

    F002_CCU_CHK_RET(
        ccu::EventWait(stageEvent, FOUR_RANK_PAIR_EVENT_MASK));
    return CCU_SUCCESS;
}

CcuResult FusedOneShotAllReduce(DirectKernelContext &ctx)
{
    // Phase 13 reinterprets the eight grouped arguments as immutable source,
    // output and scratch address/token pairs, dataBytes and phase.
    F002_CCU_CHK_RET(ValidateFourRankDirectSource(ctx));
    F002_CCU_CHK_RET(RecordInputReady(ctx));
    F002_CCU_CHK_RET(WaitInputReady(ctx));

    // Form (x0+x1) and (x2+x3) on disjoint destinations.  Every rank uses
    // the same operand order regardless of whether a source is local/remote.
    ccu::Event stageEvent;
    F002_CCU_CHK_RET(
        RunFourRankPairStage(ctx, false, stageEvent));
    F002_CCU_CHK_RET(
        RunFourRankPairStage(ctx, true, stageEvent));

    ccu::LocalAddr output;
    output.addr = FourRankOutputAddr(ctx);
    output.token = FourRankOutputToken(ctx);
    ccu::LocalAddr scratch;
    scratch.addr = FourRankScratchAddr(ctx);
    scratch.token = FourRankScratchToken(ctx);
    F002_CCU_CHK_RET(ccu::LocalReduce(output, scratch,
        FourRankDataBytes(ctx), HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM,
        stageEvent, SERIAL_EVENT_MASK));

    // The read-done records can propagate while the final local merge runs.
    // Waiting all three peers before return closes consecutive invocations.
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        F002_CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[index],
            DATA_COMPLETE_NOTIFY, DATA_COMPLETE_MASK));
    }
    F002_CCU_CHK_RET(ccu::EventWait(stageEvent, SERIAL_EVENT_MASK));
    F002_CCU_CHK_RET(WaitDataComplete(ctx));
    return CCU_SUCCESS;
}

CcuResult SerialDirectReduceScatter(DirectKernelContext &ctx)
{
    ccu::LocalAddr accumulator;
    accumulator.addr = ctx.localCclAddr;
    accumulator.addr += ctx.resultOffsetBytes;
    accumulator.token = ctx.localCclToken;
    ccu::RemoteAddr remoteSource;
    ccu::Event reduceEvent;
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        ChannelHandle channel = ctx.arg->channels[index];
        ccu::Variable remoteCclAddr =
            ccu::GetResByChannel<ccu::Variable>(channel, REMOTE_CCL_ADDR_VARIABLE);
        ccu::Variable remoteCclToken =
            ccu::GetResByChannel<ccu::Variable>(channel, REMOTE_CCL_TOKEN_VARIABLE);
        remoteSource.addr = remoteCclAddr;
        remoteSource.addr += ctx.mySliceOffset;
        remoteSource.token = remoteCclToken;
        F002_CCU_CHK_RET(ccu::ReadReduce(channel, accumulator, remoteSource,
            ctx.mySliceBytes, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM,
            reduceEvent, SERIAL_EVENT_MASK));
        F002_CCU_CHK_RET(ccu::EventWait(reduceEvent, SERIAL_EVENT_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult WavefrontDirectReduceScatter(DirectKernelContext &ctx)
{
    // Four address pairs are sufficient because a diagonal contains at most
    // four active (channel, stripe) cells. A slot is never reassigned until
    // EventWait has consumed every completion bit from the current diagonal.
    std::array<ccu::LocalAddr, RS_WAVEFRONT_STRIPE_NUM> accumulators;
    std::array<ccu::RemoteAddr, RS_WAVEFRONT_STRIPE_NUM> remoteSources;
    ccu::Event reduceEvent;

    uint32_t diagonalCount = ctx.arg->channelCount + RS_WAVEFRONT_STRIPE_NUM - 1;
    for (uint32_t diagonal = 0; diagonal < diagonalCount; ++diagonal) {
        uint32_t activeCount = 0;
        uint16_t activeMask = 0;
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            if (diagonal < channelIndex) {
                continue;
            }
            uint32_t stripe = diagonal - channelIndex;
            if (stripe >= RS_WAVEFRONT_STRIPE_NUM) {
                continue;
            }

            ChannelHandle channel = ctx.arg->channels[channelIndex];
            ccu::Variable remoteCclAddr =
                ccu::GetResByChannel<ccu::Variable>(channel, REMOTE_CCL_ADDR_VARIABLE);
            ccu::Variable remoteCclToken =
                ccu::GetResByChannel<ccu::Variable>(channel, REMOTE_CCL_TOKEN_VARIABLE);

            accumulators[activeCount].addr = ctx.localCclAddr;
            accumulators[activeCount].addr += ctx.resultOffsetBytes;
            remoteSources[activeCount].addr = remoteCclAddr;
            remoteSources[activeCount].addr += ctx.mySliceOffset;
            for (uint32_t repeat = 0; repeat < stripe; ++repeat) {
                accumulators[activeCount].addr += ctx.dataBytes;
                remoteSources[activeCount].addr += ctx.dataBytes;
            }
            accumulators[activeCount].token = ctx.localCclToken;
            remoteSources[activeCount].token = remoteCclToken;

            uint16_t eventMask = static_cast<uint16_t>(1U << activeCount);
            activeMask = static_cast<uint16_t>(activeMask | eventMask);
            if (stripe + 1 == RS_WAVEFRONT_STRIPE_NUM) {
                F002_CCU_CHK_RET(ccu::ReadReduce(channel, accumulators[activeCount],
                    remoteSources[activeCount], ctx.slotStrideBytes, HCCL_DATA_TYPE_FP32,
                    HCCL_REDUCE_SUM, reduceEvent, eventMask));
            } else {
                F002_CCU_CHK_RET(ccu::ReadReduce(channel, accumulators[activeCount],
                    remoteSources[activeCount], ctx.dataBytes, HCCL_DATA_TYPE_FP32,
                    HCCL_REDUCE_SUM, reduceEvent, eventMask));
            }
            ++activeCount;
        }
        F002_CCU_CHK_RET(ccu::EventWait(reduceEvent, activeMask));
    }
    return CCU_SUCCESS;
}

CcuResult SerialParallelReduceScatter(
    DirectKernelContext &ctx, bool initializeFromFirstPeer)
{
    ccu::LocalAddr accumulator;
    accumulator.addr = ctx.localCclAddr;
    if (initializeFromFirstPeer) {
        accumulator.addr += ctx.mySliceOffset;
    } else {
        accumulator.addr += ctx.resultOffsetBytes;
    }
    accumulator.token = ctx.localCclToken;
    ccu::RemoteAddr remoteSource;
    ccu::Event reduceEvent;
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        ChannelHandle channel = ctx.arg->channels[index];
        ccu::Variable remoteCclAddr =
            ccu::GetResByChannel<ccu::Variable>(channel, REMOTE_CCL_ADDR_VARIABLE);
        ccu::Variable remoteCclToken =
            ccu::GetResByChannel<ccu::Variable>(channel, REMOTE_CCL_TOKEN_VARIABLE);
        remoteSource.addr = remoteCclAddr;
        remoteSource.addr += ctx.mySliceOffset;
        remoteSource.token = remoteCclToken;
        if (initializeFromFirstPeer && index == 0) {
            F002_CCU_CHK_RET(ccu::Read(channel, accumulator, remoteSource,
                ctx.mySliceBytes, reduceEvent, SERIAL_EVENT_MASK));
        } else {
            F002_CCU_CHK_RET(ccu::ReadReduce(channel, accumulator, remoteSource,
                ctx.mySliceBytes, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM,
                reduceEvent, SERIAL_EVENT_MASK));
        }
        F002_CCU_CHK_RET(ccu::EventWait(reduceEvent, SERIAL_EVENT_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult WavefrontParallelReduceScatter(
    DirectKernelContext &ctx, bool initializeFromFirstPeer)
{
    // Four address pairs are sufficient because a diagonal contains at most
    // four active (channel, stripe) cells. A slot is never reassigned until
    // EventWait has consumed every completion bit from the current diagonal.
    std::array<ccu::LocalAddr, RS_WAVEFRONT_STRIPE_NUM> accumulators;
    std::array<ccu::RemoteAddr, RS_WAVEFRONT_STRIPE_NUM> remoteSources;
    ccu::Event reduceEvent;

    uint32_t diagonalCount = ctx.arg->channelCount + RS_WAVEFRONT_STRIPE_NUM - 1;
    for (uint32_t diagonal = 0; diagonal < diagonalCount; ++diagonal) {
        uint32_t activeCount = 0;
        uint16_t activeMask = 0;
        for (uint32_t channelIndex = 0; channelIndex < ctx.arg->channelCount; ++channelIndex) {
            if (diagonal < channelIndex) {
                continue;
            }
            uint32_t stripe = diagonal - channelIndex;
            if (stripe >= RS_WAVEFRONT_STRIPE_NUM) {
                continue;
            }

            ChannelHandle channel = ctx.arg->channels[channelIndex];
            ccu::Variable remoteCclAddr =
                ccu::GetResByChannel<ccu::Variable>(channel, REMOTE_CCL_ADDR_VARIABLE);
            ccu::Variable remoteCclToken =
                ccu::GetResByChannel<ccu::Variable>(channel, REMOTE_CCL_TOKEN_VARIABLE);

            accumulators[activeCount].addr = ctx.localCclAddr;
            if (initializeFromFirstPeer) {
                accumulators[activeCount].addr += ctx.mySliceOffset;
            } else {
                accumulators[activeCount].addr += ctx.resultOffsetBytes;
            }
            remoteSources[activeCount].addr = remoteCclAddr;
            remoteSources[activeCount].addr += ctx.mySliceOffset;
            for (uint32_t repeat = 0; repeat < stripe; ++repeat) {
                accumulators[activeCount].addr += ctx.dataBytes;
                remoteSources[activeCount].addr += ctx.dataBytes;
            }
            accumulators[activeCount].token = ctx.localCclToken;
            remoteSources[activeCount].token = remoteCclToken;

            uint16_t eventMask = static_cast<uint16_t>(1U << activeCount);
            activeMask = static_cast<uint16_t>(activeMask | eventMask);
            if (initializeFromFirstPeer && channelIndex == 0) {
                if (stripe + 1 == RS_WAVEFRONT_STRIPE_NUM) {
                    F002_CCU_CHK_RET(ccu::Read(channel, accumulators[activeCount],
                        remoteSources[activeCount], ctx.slotStrideBytes, reduceEvent, eventMask));
                } else {
                    F002_CCU_CHK_RET(ccu::Read(channel, accumulators[activeCount],
                        remoteSources[activeCount], ctx.dataBytes, reduceEvent, eventMask));
                }
            } else {
                if (stripe + 1 == RS_WAVEFRONT_STRIPE_NUM) {
                    F002_CCU_CHK_RET(ccu::ReadReduce(channel, accumulators[activeCount],
                        remoteSources[activeCount], ctx.slotStrideBytes, HCCL_DATA_TYPE_FP32,
                        HCCL_REDUCE_SUM, reduceEvent, eventMask));
                } else {
                    F002_CCU_CHK_RET(ccu::ReadReduce(channel, accumulators[activeCount],
                        remoteSources[activeCount], ctx.dataBytes, HCCL_DATA_TYPE_FP32,
                        HCCL_REDUCE_SUM, reduceEvent, eventMask));
                }
            }
            ++activeCount;
        }
        F002_CCU_CHK_RET(ccu::EventWait(reduceEvent, activeMask));
    }
    return CCU_SUCCESS;
}

CcuResult ParallelReduceScatterChannels(
    DirectKernelContext &ctx, bool initializeFromFirstPeer)
{
    // Groups smaller than four cannot fill a four-way wavefront. Tiny slices
    // also produce a zero aligned regular stripe. Both cases retain F002's
    // whole-slice serial reduction exactly.
    if (ctx.arg->channelCount < RS_WAVEFRONT_STRIPE_NUM) {
        F002_CCU_CHK_RET(SerialParallelReduceScatter(ctx, initializeFromFirstPeer));
        return CCU_SUCCESS;
    }
    CCU_IF(ctx.dataBytes == 0)
    {
        F002_CCU_CHK_RET(SerialParallelReduceScatter(ctx, initializeFromFirstPeer));
    }
    CCU_IF(ctx.dataBytes != 0)
    {
        F002_CCU_CHK_RET(WavefrontParallelReduceScatter(ctx, initializeFromFirstPeer));
    }
    return CCU_SUCCESS;
}

CcuResult DirectReduceScatter(DirectKernelContext &ctx)
{
    F002_CCU_CHK_RET(WaitInputReady(ctx));

    // Groups smaller than four cannot fill a four-way wavefront. Tiny slices
    // also produce a zero aligned regular stripe. Both cases retain F002's
    // whole-slice serial reduction exactly.
    if (ctx.arg->channelCount < RS_WAVEFRONT_STRIPE_NUM) {
        F002_CCU_CHK_RET(SerialDirectReduceScatter(ctx));
        return CCU_SUCCESS;
    }
    CCU_IF(ctx.dataBytes == 0)
    {
        F002_CCU_CHK_RET(SerialDirectReduceScatter(ctx));
    }
    CCU_IF(ctx.dataBytes != 0)
    {
        F002_CCU_CHK_RET(WavefrontDirectReduceScatter(ctx));
    }
    return CCU_SUCCESS;
}

CcuResult ParallelReduceScatter(DirectKernelContext &ctx)
{
    // CCU_IF builds every phase while the kernel is registered. Non-two-group
    // kernels must therefore compile this phase as an empty graph instead of
    // returning an error; host dispatch never launches it for those kernels.
    if (ctx.arg->groupCount != 2 || ctx.arg->groupIndex >= 2) {
        return CCU_SUCCESS;
    }

    F002_CCU_CHK_RET(WaitInputReady(ctx));
    // Group 0 starts from the local contribution copied by the host. Group 1
    // reuses this rank's logical owner slice: no remote owner reads that range
    // on this rank, because every peer reads the slice matching its own rank.
    // Its first peer initializes each stripe with Read, and later peers fold
    // into it with ReadReduce. The diagonal wait preserves that fixed order
    // without an extra full-slice initialization stage.
    bool initializeFromFirstPeer = ctx.arg->groupIndex == 1;
    F002_CCU_CHK_RET(ParallelReduceScatterChannels(ctx, initializeFromFirstPeer));
    return CCU_SUCCESS;
}

CcuResult MergeParallelReduceScatter(DirectKernelContext &ctx)
{
    // The group-1 kernel also builds this phase at registration time, but only
    // group 0 is launched for the deterministic partial merge.
    if (ctx.arg->groupCount != 2 || ctx.arg->groupIndex != 0) {
        return CCU_SUCCESS;
    }

    ccu::LocalAddr accumulator;
    accumulator.addr = ctx.localCclAddr;
    accumulator.addr += ctx.resultOffsetBytes;
    accumulator.token = ctx.localCclToken;

    ccu::LocalAddr secondary;
    secondary.addr = ctx.localCclAddr;
    secondary.addr += ctx.mySliceOffset;
    secondary.token = ctx.localCclToken;

    ccu::Event reduceEvent;
    F002_CCU_CHK_RET(ccu::LocalReduce(accumulator, secondary, ctx.mySliceBytes,
        HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, reduceEvent, SERIAL_EVENT_MASK));
    F002_CCU_CHK_RET(ccu::EventWait(reduceEvent, SERIAL_EVENT_MASK));
    return CCU_SUCCESS;
}

CcuResult DirectAllGatherAndRecord(DirectKernelContext &ctx)
{
    ccu::LocalAddr localResult;
    localResult.addr = ctx.localCclAddr;
    localResult.addr += ctx.resultOffsetBytes;
    localResult.token = ctx.localCclToken;

    std::array<ccu::RemoteAddr, MAX_RANK_SIZE> remoteDestinations;
    ccu::Event dataEvent;
    uint16_t allDataMask = 0;
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        ChannelHandle channel = ctx.arg->channels[index];
        ccu::Variable remoteCclAddr =
            ccu::GetResByChannel<ccu::Variable>(channel, REMOTE_CCL_ADDR_VARIABLE);
        ccu::Variable remoteCclToken =
            ccu::GetResByChannel<ccu::Variable>(channel, REMOTE_CCL_TOKEN_VARIABLE);
        remoteDestinations[index].addr = remoteCclAddr;
        remoteDestinations[index].addr += ctx.mySliceOffset;
        remoteDestinations[index].token = remoteCclToken;

        uint16_t eventMask = static_cast<uint16_t>(1U << index);
        allDataMask = static_cast<uint16_t>(allDataMask | eventMask);
        F002_CCU_CHK_RET(ccu::Write(channel, remoteDestinations[index], localResult,
            ctx.mySliceBytes, dataEvent, eventMask));
    }
    F002_CCU_CHK_RET(ccu::EventWait(dataEvent, allDataMask));

    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        F002_CCU_CHK_RET(ccu::NotifyRecord(
            ctx.arg->channels[index], DATA_COMPLETE_NOTIFY, DATA_COMPLETE_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult RecordDirectBufferReady(DirectBufferContext &ctx)
{
    // All four values are published once per edge and consumed by one
    // four-bit wait in this window. Output values remain stable through the
    // later AllGather phase, so no second address exchange or notify reuse is
    // required before the completion barrier.
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        ChannelHandle channel = ctx.arg->channels[index];
        F002_CCU_CHK_RET(ccu::WriteVariableWithNotify(channel, ctx.inputAddr,
            REMOTE_INPUT_ADDR_VARIABLE, INPUT_READY_NOTIFY, DIRECT_INPUT_ADDR_MASK));
        F002_CCU_CHK_RET(ccu::WriteVariableWithNotify(channel, ctx.inputToken,
            REMOTE_INPUT_TOKEN_VARIABLE, INPUT_READY_NOTIFY, DIRECT_INPUT_TOKEN_MASK));
        F002_CCU_CHK_RET(ccu::WriteVariableWithNotify(channel, ctx.outputAddr,
            REMOTE_OUTPUT_ADDR_VARIABLE, INPUT_READY_NOTIFY, DIRECT_OUTPUT_ADDR_MASK));
        F002_CCU_CHK_RET(ccu::WriteVariableWithNotify(channel, ctx.outputToken,
            REMOTE_OUTPUT_TOKEN_VARIABLE, INPUT_READY_NOTIFY, DIRECT_OUTPUT_TOKEN_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult WaitDirectBufferReady(DirectBufferContext &ctx)
{
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        F002_CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[index],
            INPUT_READY_NOTIFY, DIRECT_BUFFER_READY_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult CyclicAsymmetricDirectBufferReduceScatter(DirectBufferContext &ctx)
{
    // Rank 12 has at most eight Channels on either Die. Assign
    // (round + channel) modulo eight to stripe so every round keeps every
    // Channel active on a distinct range. A fixed stripe sees one static peer
    // permutation, and each round's EventWait makes its FP32 order deterministic.
    std::array<ccu::LocalAddr, DIRECT_BUFFER_MAX_ACTIVE_NUM> channelAccumulators;
    std::array<ccu::RemoteAddr, DIRECT_BUFFER_MAX_ACTIVE_NUM> channelSources;
    ccu::Event reduceEvent;

    ccu::Variable ownerOffsetBytes;
    ownerOffsetBytes = 0;
    for (uint32_t repeat = 0; repeat < ctx.arg->myRank; ++repeat) {
        ownerOffsetBytes += ctx.regularSliceBytes;
    }

    for (uint32_t channelIndex = 0;
        channelIndex < ctx.arg->channelCount; ++channelIndex) {
        ChannelHandle channel = ctx.arg->channels[channelIndex];
        ccu::Variable remoteInputAddr =
            ccu::GetResByChannel<ccu::Variable>(channel, REMOTE_INPUT_ADDR_VARIABLE);
        ccu::Variable remoteInputToken =
            ccu::GetResByChannel<ccu::Variable>(channel, REMOTE_INPUT_TOKEN_VARIABLE);

        if (ctx.arg->groupIndex == 0) {
            channelAccumulators[channelIndex].addr = ctx.outputAddr;
            channelAccumulators[channelIndex].addr += ownerOffsetBytes;
        } else {
            channelAccumulators[channelIndex].addr = ctx.scratchAddr;
        }
        channelAccumulators[channelIndex].token =
            ctx.arg->groupIndex == 0 ? ctx.outputToken : ctx.scratchToken;

        channelSources[channelIndex].addr = remoteInputAddr;
        channelSources[channelIndex].addr += ownerOffsetBytes;
        channelSources[channelIndex].token = remoteInputToken;

        // Channel c starts at stripe c. Since channelCount is at most eight,
        // every initial cursor is inside this rank-12 schedule.
        for (uint32_t repeat = 0; repeat < channelIndex; ++repeat) {
            channelAccumulators[channelIndex].addr += ctx.regularStripeBytes;
            channelSources[channelIndex].addr += ctx.regularStripeBytes;
        }
    }

    uint16_t allChannelsMask = static_cast<uint16_t>(
        (1U << ctx.arg->channelCount) - 1U);
    for (uint32_t round = 0;
        round < ASYMMETRIC_DIRECT_BUFFER_RS_STRIPE_NUM; ++round) {
        for (uint32_t channelIndex = 0;
            channelIndex < ctx.arg->channelCount; ++channelIndex) {
            uint32_t stripe = (round + channelIndex) %
                ASYMMETRIC_DIRECT_BUFFER_RS_STRIPE_NUM;
            uint32_t firstChannel = stripe < ctx.arg->channelCount ?
                stripe : ctx.arg->channelCount - 1;
            ChannelHandle channel = ctx.arg->channels[channelIndex];
            uint16_t eventMask = static_cast<uint16_t>(1U << channelIndex);
            if (ctx.arg->groupIndex == 1 && channelIndex == firstChannel) {
                F002_CCU_CHK_RET(ccu::Read(channel,
                    channelAccumulators[channelIndex], channelSources[channelIndex],
                    AsymmetricDirectBufferStripeBytes(ctx, stripe),
                    reduceEvent, eventMask));
            } else {
                F002_CCU_CHK_RET(ccu::ReadReduce(channel,
                    channelAccumulators[channelIndex], channelSources[channelIndex],
                    AsymmetricDirectBufferStripeBytes(ctx, stripe),
                    HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM,
                    reduceEvent, eventMask));
            }
        }
        F002_CCU_CHK_RET(ccu::EventWait(reduceEvent, allChannelsMask));

        if (round + 1 == ASYMMETRIC_DIRECT_BUFFER_RS_STRIPE_NUM) {
            continue;
        }
        for (uint32_t channelIndex = 0;
            channelIndex < ctx.arg->channelCount; ++channelIndex) {
            uint32_t stripe = (round + channelIndex) %
                ASYMMETRIC_DIRECT_BUFFER_RS_STRIPE_NUM;
            if (stripe + 1 == ASYMMETRIC_DIRECT_BUFFER_RS_STRIPE_NUM) {
                // Reset only after all primitives in this round have completed.
                if (ctx.arg->groupIndex == 0) {
                    channelAccumulators[channelIndex].addr = ctx.outputAddr;
                    channelAccumulators[channelIndex].addr += ownerOffsetBytes;
                } else {
                    channelAccumulators[channelIndex].addr = ctx.scratchAddr;
                }
                ChannelHandle channel = ctx.arg->channels[channelIndex];
                ccu::Variable remoteInputAddr =
                    ccu::GetResByChannel<ccu::Variable>(
                        channel, REMOTE_INPUT_ADDR_VARIABLE);
                channelSources[channelIndex].addr = remoteInputAddr;
                channelSources[channelIndex].addr += ownerOffsetBytes;
            } else {
                channelAccumulators[channelIndex].addr += ctx.regularStripeBytes;
                channelSources[channelIndex].addr += ctx.regularStripeBytes;
            }
        }
    }
    return CCU_SUCCESS;
}

CcuResult CyclicDirectBufferReduceScatter(DirectBufferContext &ctx)
{
    // The rank-16 direct path has seven or eight Channels per Die.  Assign
    // (round + channel) modulo sixteen to stripe so every round keeps every
    // Channel active on a distinct range.  A fixed stripe consequently sees
    // a stripe-specific, but registration-time fixed, peer permutation.  The
    // round EventWait makes that permutation the deterministic FP32 order.
    std::array<ccu::LocalAddr, DIRECT_BUFFER_MAX_ACTIVE_NUM> channelAccumulators;
    std::array<ccu::RemoteAddr, DIRECT_BUFFER_MAX_ACTIVE_NUM> channelSources;
    ccu::Event reduceEvent;

    // Compute the owner-slice offset once in one XN.  Reusing it for the
    // initial cursors and the single wrap of each nonzero Channel avoids
    // rebuilding myRank address additions for every cursor reset.
    ccu::Variable ownerOffsetBytes;
    ownerOffsetBytes = 0;
    for (uint32_t repeat = 0; repeat < ctx.arg->myRank; ++repeat) {
        ownerOffsetBytes += ctx.regularSliceBytes;
    }

    for (uint32_t channelIndex = 0;
        channelIndex < ctx.arg->channelCount; ++channelIndex) {
        ChannelHandle channel = ctx.arg->channels[channelIndex];
        ccu::Variable remoteInputAddr =
            ccu::GetResByChannel<ccu::Variable>(channel, REMOTE_INPUT_ADDR_VARIABLE);
        ccu::Variable remoteInputToken =
            ccu::GetResByChannel<ccu::Variable>(channel, REMOTE_INPUT_TOKEN_VARIABLE);

        if (ctx.arg->groupIndex == 0) {
            channelAccumulators[channelIndex].addr = ctx.outputAddr;
            channelAccumulators[channelIndex].addr += ownerOffsetBytes;
        } else {
            channelAccumulators[channelIndex].addr = ctx.scratchAddr;
        }
        channelAccumulators[channelIndex].token =
            ctx.arg->groupIndex == 0 ? ctx.outputToken : ctx.scratchToken;

        channelSources[channelIndex].addr = remoteInputAddr;
        channelSources[channelIndex].addr += ownerOffsetBytes;
        channelSources[channelIndex].token = remoteInputToken;

        // Channel c starts at stripe c and then advances monotonically, with
        // exactly one wrap after its stripe-15 primitive has completed.
        for (uint32_t repeat = 0; repeat < channelIndex; ++repeat) {
            channelAccumulators[channelIndex].addr += ctx.regularStripeBytes;
            channelSources[channelIndex].addr += ctx.regularStripeBytes;
        }
    }

    uint16_t allChannelsMask = static_cast<uint16_t>(
        (1U << ctx.arg->channelCount) - 1U);
    for (uint32_t round = 0; round < DIRECT_BUFFER_RS_STRIPE_NUM; ++round) {
        for (uint32_t channelIndex = 0;
            channelIndex < ctx.arg->channelCount; ++channelIndex) {
            uint32_t stripe =
                (round + channelIndex) % DIRECT_BUFFER_RS_STRIPE_NUM;
            uint32_t firstChannel = stripe < ctx.arg->channelCount ?
                stripe : ctx.arg->channelCount - 1;
            ChannelHandle channel = ctx.arg->channels[channelIndex];
            uint16_t eventMask = static_cast<uint16_t>(1U << channelIndex);
            if (ctx.arg->groupIndex == 1 && channelIndex == firstChannel) {
                // Every group-1 stripe is initialized exactly once by the
                // first cell in its fixed cyclic peer order.
                F002_CCU_CHK_RET(ccu::Read(channel,
                    channelAccumulators[channelIndex], channelSources[channelIndex],
                    DirectBufferStripeBytes(ctx, stripe), reduceEvent, eventMask));
            } else {
                F002_CCU_CHK_RET(ccu::ReadReduce(channel,
                    channelAccumulators[channelIndex], channelSources[channelIndex],
                    DirectBufferStripeBytes(ctx, stripe), HCCL_DATA_TYPE_FP32,
                    HCCL_REDUCE_SUM, reduceEvent, eventMask));
            }
        }
        F002_CCU_CHK_RET(ccu::EventWait(reduceEvent, allChannelsMask));

        if (round + 1 == DIRECT_BUFFER_RS_STRIPE_NUM) {
            continue;
        }
        for (uint32_t channelIndex = 0;
            channelIndex < ctx.arg->channelCount; ++channelIndex) {
            uint32_t stripe =
                (round + channelIndex) % DIRECT_BUFFER_RS_STRIPE_NUM;
            if (stripe + 1 == DIRECT_BUFFER_RS_STRIPE_NUM) {
                // Address assignment is a device-side register reset.  It is
                // deliberately emitted only after this round's EventWait, so
                // no in-flight primitive can observe the wrapped cursor.
                if (ctx.arg->groupIndex == 0) {
                    channelAccumulators[channelIndex].addr = ctx.outputAddr;
                    channelAccumulators[channelIndex].addr += ownerOffsetBytes;
                } else {
                    channelAccumulators[channelIndex].addr = ctx.scratchAddr;
                }
                ChannelHandle channel = ctx.arg->channels[channelIndex];
                ccu::Variable remoteInputAddr =
                    ccu::GetResByChannel<ccu::Variable>(
                        channel, REMOTE_INPUT_ADDR_VARIABLE);
                channelSources[channelIndex].addr = remoteInputAddr;
                channelSources[channelIndex].addr += ownerOffsetBytes;
            } else {
                channelAccumulators[channelIndex].addr += ctx.regularStripeBytes;
                channelSources[channelIndex].addr += ctx.regularStripeBytes;
            }
        }
    }
    return CCU_SUCCESS;
}

CcuResult DirectBufferReduceScatter(DirectBufferContext &ctx)
{
    // Publish every edge before waiting. Group 0 overlaps its sole owner-slice
    // copy with the address handshake; group 1 starts each stripe from the
    // first peer in its selected static order and needs no scratch init copy.
    F002_CCU_CHK_RET(RecordDirectBufferReady(ctx));
    if (ctx.arg->groupIndex == 0) {
        ccu::LocalAddr source;
        source.addr = ctx.inputAddr;
        source.token = ctx.inputToken;
        AddDirectBufferSliceOffset(
            source.addr, ctx.regularSliceBytes, ctx.arg->myRank);

        ccu::LocalAddr destination;
        destination.addr = ctx.outputAddr;
        destination.token = ctx.outputToken;
        AddDirectBufferSliceOffset(
            destination.addr, ctx.regularSliceBytes, ctx.arg->myRank);

        ccu::Event copyEvent;
        F002_CCU_CHK_RET(ccu::LocalCopy(destination, source,
            DirectBufferMySliceBytes(ctx), copyEvent, SERIAL_EVENT_MASK));
        F002_CCU_CHK_RET(WaitDirectBufferReady(ctx));
        F002_CCU_CHK_RET(ccu::EventWait(copyEvent, SERIAL_EVENT_MASK));
    } else {
        F002_CCU_CHK_RET(WaitDirectBufferReady(ctx));
    }

    if (ctx.arg->rankSize == MAX_RANK_SIZE) {
        F002_CCU_CHK_RET(CyclicDirectBufferReduceScatter(ctx));
    } else {
        // Rank 12 uses the independent cyclic-8 graph; rank 16 above keeps
        // F014's proven cyclic-16 graph byte-for-byte isolated.
        F002_CCU_CHK_RET(CyclicAsymmetricDirectBufferReduceScatter(ctx));
    }
    return CCU_SUCCESS;
}

CcuResult MergeDirectBufferPartials(DirectBufferContext &ctx)
{
    // Host dispatch launches this phase only on group 0 after both Die
    // missions have completed. The fixed P0 + P1 merge preserves F006's
    // deterministic two-group reduction DAG.
    if (ctx.arg->groupIndex != 0) {
        return CCU_SUCCESS;
    }

    ccu::LocalAddr accumulator;
    accumulator.addr = ctx.outputAddr;
    accumulator.token = ctx.outputToken;
    AddDirectBufferSliceOffset(
        accumulator.addr, ctx.regularSliceBytes, ctx.arg->myRank);

    ccu::LocalAddr secondary;
    secondary.addr = ctx.scratchAddr;
    secondary.token = ctx.scratchToken;

    ccu::Event reduceEvent;
    F002_CCU_CHK_RET(ccu::LocalReduce(accumulator, secondary,
        DirectBufferMySliceBytes(ctx), HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM,
        reduceEvent, SERIAL_EVENT_MASK));
    F002_CCU_CHK_RET(ccu::EventWait(reduceEvent, SERIAL_EVENT_MASK));
    return CCU_SUCCESS;
}

CcuResult DirectBufferAllGather(DirectBufferContext &ctx)
{
    ccu::LocalAddr localResult;
    localResult.addr = ctx.outputAddr;
    localResult.token = ctx.outputToken;
    AddDirectBufferSliceOffset(
        localResult.addr, ctx.regularSliceBytes, ctx.arg->myRank);

    std::array<ccu::RemoteAddr, MAX_RANK_SIZE> remoteDestinations;
    ccu::Event writeEvent;
    uint16_t allWritesMask = 0;
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        ChannelHandle channel = ctx.arg->channels[index];
        ccu::Variable remoteOutputAddr =
            ccu::GetResByChannel<ccu::Variable>(channel, REMOTE_OUTPUT_ADDR_VARIABLE);
        ccu::Variable remoteOutputToken =
            ccu::GetResByChannel<ccu::Variable>(channel, REMOTE_OUTPUT_TOKEN_VARIABLE);
        remoteDestinations[index].addr = remoteOutputAddr;
        remoteDestinations[index].token = remoteOutputToken;
        AddDirectBufferSliceOffset(remoteDestinations[index].addr,
            ctx.regularSliceBytes, ctx.arg->myRank);

        uint16_t eventMask = static_cast<uint16_t>(1U << index);
        allWritesMask = static_cast<uint16_t>(allWritesMask | eventMask);
        F002_CCU_CHK_RET(ccu::Write(channel, remoteDestinations[index], localResult,
            DirectBufferMySliceBytes(ctx), writeEvent, eventMask));
    }
    F002_CCU_CHK_RET(ccu::EventWait(writeEvent, allWritesMask));

    // Record every edge before waiting on any edge, so peer enumeration order
    // cannot form a cycle. Each channel sees exactly one Record and one Wait
    // for notify 1 in every window.
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        F002_CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[index],
            DATA_COMPLETE_NOTIFY, DATA_COMPLETE_MASK));
    }
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        F002_CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[index],
            DATA_COMPLETE_NOTIFY, DATA_COMPLETE_MASK));
    }
    return CCU_SUCCESS;
}

bool IsRank16ClosRsagGroup(const DirectKernelContext &ctx)
{
    if (ctx.arg == nullptr || ctx.arg->rankSize != RANK16_CLOS_SIZE ||
        ctx.arg->groupCount != 2 || ctx.arg->groupIndex >= ctx.arg->groupCount ||
        ctx.arg->channelCount != RANK16_CLOS_CHANNEL_NUM ||
        ctx.arg->myRank >= RANK16_CLOS_SIZE) {
        return false;
    }

    uint32_t remoteBase = ctx.arg->myRank < RANK16_SERVER_SIZE ?
        RANK16_SERVER_SIZE : 0;
    std::array<bool, RANK16_SERVER_SIZE> seen{};
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        uint32_t remoteRank = ctx.arg->remoteRanks[index];
        if (remoteRank < remoteBase ||
            remoteRank >= remoteBase + RANK16_SERVER_SIZE) {
            return false;
        }
        uint32_t pairIndex = remoteRank % RANK16_SERVER_SIZE;
        if (seen[pairIndex]) {
            return false;
        }
        seen[pairIndex] = true;
    }
    for (bool pairSeen : seen) {
        if (!pairSeen) {
            return false;
        }
    }
    return true;
}

void BuildRank16PairChannels(const DirectKernelContext &ctx,
    std::array<ChannelHandle, RANK16_CLOS_CHANNEL_NUM> &pairChannels)
{
    // Map by the opposite-server local rank, independent of registration
    // order.  Both servers therefore use the same fixed 0..7 reduction tree.
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        uint32_t pairIndex =
            ctx.arg->remoteRanks[index] % RANK16_SERVER_SIZE;
        pairChannels[pairIndex] = ctx.arg->channels[index];
    }
}

struct Rank16ClosRsagAddresses {
    ccu::LocalAddr source;
    ccu::LocalAddr output;
    ccu::Variable ownerSliceBytes;
    std::array<ccu::LocalAddr, 4> lanes;
};

void BuildRank16ClosRsagAddresses(DirectKernelContext &ctx,
    Rank16ClosRsagAddresses &addresses)
{
    // Phase 14 reinterprets the existing eight-argument grouped ABI as:
    //   sourceAddr, sourceToken, outputAddr, outputToken,
    //   cclAddr, cclToken, dataBytes, phase.
    addresses.source.addr = ctx.localCclAddr;
    addresses.source.token = ctx.localCclToken;
    addresses.output.addr = ctx.dataBytes;
    addresses.output.token = ctx.slotStrideBytes;
    addresses.ownerSliceBytes = RANK16_RSAG_OWNER_SLICE_BYTES;

    // The overlap-safe Host path stores its source snapshot in [0, D); the
    // disjoint path reads user input directly and leaves that interval unused.
    // All four owner accumulators stay in the disjoint [D, 1.5D) scratch.
    for (uint32_t lane = 0; lane < addresses.lanes.size(); ++lane) {
        addresses.lanes[lane].addr = ctx.mySliceOffset;
        addresses.lanes[lane].addr += ctx.resultOffsetBytes;
        addresses.lanes[lane].token = ctx.mySliceBytes;
        for (uint32_t repeat = 0; repeat < lane; ++repeat) {
            addresses.lanes[lane].addr += addresses.ownerSliceBytes;
        }
    }
}

CcuResult PublishRank16RsagBuffer(DirectKernelContext &ctx,
    Rank16ClosRsagAddresses &addresses)
{
    (void)addresses;
    // Preserve F051's 30-write metadata budget.  On the matching edge,
    // variables 0/1 name lane 0's partial.  On seven nonmatching edges they
    // name the immutable direct source and variables 2/3 name user output.
    // PARTIAL_READY later proves the matching lane data is complete.
    uint32_t owner = ctx.arg->myRank % RANK16_SERVER_SIZE;
    ccu::Variable matchingPartialAddr;
    matchingPartialAddr = ctx.mySliceOffset;
    matchingPartialAddr += ctx.resultOffsetBytes;
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        ChannelHandle channel = ctx.arg->channels[index];
        uint32_t pair = ctx.arg->remoteRanks[index] % RANK16_SERVER_SIZE;
        if (pair == owner) {
            F002_CCU_CHK_RET(ccu::WriteVariableWithNotify(channel,
                matchingPartialAddr, RANK16_RSAG_CCL_ADDR_VARIABLE,
                INPUT_READY_NOTIFY, RANK16_RSAG_CCL_ADDR_MASK));
            F002_CCU_CHK_RET(ccu::WriteVariableWithNotify(channel,
                ctx.mySliceBytes, RANK16_RSAG_CCL_TOKEN_VARIABLE,
                INPUT_READY_NOTIFY, RANK16_RSAG_CCL_TOKEN_MASK));
            continue;
        }

        F002_CCU_CHK_RET(ccu::WriteVariableWithNotify(channel,
            ctx.localCclAddr, RANK16_RSAG_CCL_ADDR_VARIABLE,
            INPUT_READY_NOTIFY, RANK16_RSAG_CCL_ADDR_MASK));
        F002_CCU_CHK_RET(ccu::WriteVariableWithNotify(channel,
            ctx.localCclToken, RANK16_RSAG_CCL_TOKEN_VARIABLE,
            INPUT_READY_NOTIFY, RANK16_RSAG_CCL_TOKEN_MASK));
        F002_CCU_CHK_RET(ccu::WriteVariableWithNotify(channel,
            ctx.dataBytes, RANK16_RSAG_OUTPUT_ADDR_VARIABLE,
            INPUT_READY_NOTIFY, RANK16_RSAG_OUTPUT_ADDR_MASK));
        F002_CCU_CHK_RET(ccu::WriteVariableWithNotify(channel,
            ctx.slotStrideBytes, RANK16_RSAG_OUTPUT_TOKEN_VARIABLE,
            INPUT_READY_NOTIFY, RANK16_RSAG_OUTPUT_TOKEN_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult ReduceRank16RemoteOwnerSlice(DirectKernelContext &ctx,
    const std::array<ChannelHandle, RANK16_CLOS_CHANNEL_NUM> &pairChannels,
    Rank16ClosRsagAddresses &addresses)
{
    std::array<ccu::RemoteAddr, RANK16_SERVER_SIZE> remoteSlices;
    ccu::Variable ownerOffset;
    ownerOffset = 0;
    uint32_t owner = ctx.arg->myRank % RANK16_SERVER_SIZE;
    for (uint32_t repeat = 0; repeat < owner; ++repeat) {
        ownerOffset += addresses.ownerSliceBytes;
    }

    // Seed lane 0 from this rank's immutable owner leaf.  This 1/8-D local
    // copy starts before the metadata waits and shares the first-wave Event,
    // hiding it behind peer publication without modifying the user input.
    ccu::LocalAddr localOwner;
    localOwner.addr = addresses.source.addr;
    localOwner.token = addresses.source.token;
    localOwner.addr += ownerOffset;
    ccu::Event networkEvent;
    F002_CCU_CHK_RET(ccu::LocalCopy(addresses.lanes[0], localOwner,
        addresses.ownerSliceBytes, networkEvent, SERIAL_EVENT_MASK));

    for (uint32_t pair = 0; pair < RANK16_SERVER_SIZE; ++pair) {
        ChannelHandle channel = pairChannels[pair];
        uint16_t readyMask = pair == owner ?
            RANK16_RSAG_BUFFER_READY_MASK :
            RANK16_RSAG_REMOTE_OUTPUT_READY_MASK;
        F002_CCU_CHK_RET(ccu::NotifyWait(
            channel, INPUT_READY_NOTIFY, readyMask));
        if (pair == owner) {
            continue;
        }
        remoteSlices[pair].addr =
            ccu::GetResByChannel<ccu::Variable>(
                channel, RANK16_RSAG_CCL_ADDR_VARIABLE);
        remoteSlices[pair].addr += ownerOffset;
        remoteSlices[pair].token =
            ccu::GetResByChannel<ccu::Variable>(
                channel, RANK16_RSAG_CCL_TOKEN_VARIABLE);
    }

    // Read the seven nonmatching peers in cyclic order.  Wave 1 combines the
    // local seed with three remote Reads; wave 2 performs four ReadReduces, so
    // every source serves exactly three/four requests and the fixed tree is:
    // ((self+r1)+(r2+r3))+((r4+r5)+(r6+r7)).
    std::array<uint32_t, RANK16_SERVER_SIZE - 1> remotePairs;
    for (uint32_t index = 0; index < remotePairs.size(); ++index) {
        remotePairs[index] =
            (owner + index + 1) % RANK16_SERVER_SIZE;
    }

    for (uint32_t lane = 1; lane < addresses.lanes.size(); ++lane) {
        uint32_t pair = remotePairs[lane * 2 - 1];
        uint16_t eventMask = static_cast<uint16_t>(1U << lane);
        F002_CCU_CHK_RET(ccu::Read(pairChannels[pair], addresses.lanes[lane],
            remoteSlices[pair], addresses.ownerSliceBytes,
            networkEvent, eventMask));
    }
    F002_CCU_CHK_RET(
        ccu::EventWait(networkEvent, RANK16_RSAG_FOUR_LANE_MASK));
    for (uint32_t lane = 0; lane < addresses.lanes.size(); ++lane) {
        uint32_t pair = remotePairs[lane * 2];
        uint16_t eventMask = static_cast<uint16_t>(1U << lane);
        F002_CCU_CHK_RET(ccu::ReadReduce(pairChannels[pair],
            addresses.lanes[lane], remoteSlices[pair],
            addresses.ownerSliceBytes, HCCL_DATA_TYPE_FP32,
            HCCL_REDUCE_SUM, networkEvent, eventMask));
    }
    F002_CCU_CHK_RET(
        ccu::EventWait(networkEvent, RANK16_RSAG_FOUR_LANE_MASK));

    ccu::Event reduceEvent;
    F002_CCU_CHK_RET(ccu::LocalReduce(addresses.lanes[0], addresses.lanes[1],
        addresses.ownerSliceBytes, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM,
        reduceEvent, static_cast<uint16_t>(1U << 0)));
    F002_CCU_CHK_RET(ccu::LocalReduce(addresses.lanes[2], addresses.lanes[3],
        addresses.ownerSliceBytes, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM,
        reduceEvent, static_cast<uint16_t>(1U << 1)));
    F002_CCU_CHK_RET(
        ccu::EventWait(reduceEvent, RANK16_RSAG_TWO_LANE_MASK));
    F002_CCU_CHK_RET(ccu::LocalReduce(addresses.lanes[0], addresses.lanes[2],
        addresses.ownerSliceBytes, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM,
        reduceEvent, SERIAL_EVENT_MASK));
    F002_CCU_CHK_RET(ccu::EventWait(reduceEvent, SERIAL_EVENT_MASK));
    return CCU_SUCCESS;
}

CcuResult CombineRank16OwnerPartials(DirectKernelContext &ctx,
    const std::array<ChannelHandle, RANK16_CLOS_CHANNEL_NUM> &pairChannels,
    Rank16ClosRsagAddresses &addresses)
{
    uint32_t owner = ctx.arg->myRank % RANK16_SERVER_SIZE;
    ChannelHandle matchingChannel = pairChannels[owner];

    ccu::LocalAddr ownerResult;
    ownerResult.addr = addresses.output.addr;
    ownerResult.token = addresses.output.token;
    for (uint32_t repeat = 0; repeat < owner; ++repeat) {
        ownerResult.addr += addresses.ownerSliceBytes;
    }

    // Seed the final owner slice while the matching PARTIAL_READY handshake
    // is in flight.  The local seven-leaf partial is already complete and
    // lane 0 stays immutable until both matching ReadReduces have finished.
    // This turns the former low/high asymmetric Read+LocalReduce and
    // LocalCopy+ReadReduce tails into the same two-operation path while
    // hiding the local copy behind the unchanged bidirectional handshake.
    ccu::Event combineEvent;
    F002_CCU_CHK_RET(ccu::LocalCopy(ownerResult, addresses.lanes[0],
        addresses.ownerSliceBytes, combineEvent, SERIAL_EVENT_MASK));

    F002_CCU_CHK_RET(ccu::NotifyRecord(matchingChannel, INPUT_READY_NOTIFY,
        RANK16_RSAG_PARTIAL_READY_MASK));
    F002_CCU_CHK_RET(ccu::NotifyWait(matchingChannel, INPUT_READY_NOTIFY,
        RANK16_RSAG_PARTIAL_READY_MASK));

    ccu::RemoteAddr remotePartial;
    remotePartial.addr = ccu::GetResByChannel<ccu::Variable>(
        matchingChannel, RANK16_RSAG_CCL_ADDR_VARIABLE);
    remotePartial.token = ccu::GetResByChannel<ccu::Variable>(
        matchingChannel, RANK16_RSAG_CCL_TOKEN_VARIABLE);

    F002_CCU_CHK_RET(
        ccu::EventWait(combineEvent, SERIAL_EVENT_MASK));
    F002_CCU_CHK_RET(ccu::ReadReduce(matchingChannel, ownerResult,
        remotePartial, addresses.ownerSliceBytes, HCCL_DATA_TYPE_FP32,
        HCCL_REDUCE_SUM, combineEvent, SERIAL_EVENT_MASK));
    F002_CCU_CHK_RET(
        ccu::EventWait(combineEvent, SERIAL_EVENT_MASK));
    return CCU_SUCCESS;
}

CcuResult AllGatherRank16OwnerSlice(DirectKernelContext &ctx,
    const std::array<ChannelHandle, RANK16_CLOS_CHANNEL_NUM> &pairChannels,
    Rank16ClosRsagAddresses &addresses)
{
    uint32_t owner = ctx.arg->myRank % RANK16_SERVER_SIZE;
    ccu::LocalAddr localResult;
    localResult.addr = addresses.output.addr;
    localResult.token = addresses.output.token;
    for (uint32_t repeat = 0; repeat < owner; ++repeat) {
        localResult.addr += addresses.ownerSliceBytes;
    }

    ccu::Event writeEvent;
    uint32_t writeIndex = 0;
    for (uint32_t pair = 0; pair < RANK16_SERVER_SIZE; ++pair) {
        if (pair == owner) {
            continue;
        }
        ChannelHandle channel = pairChannels[pair];
        ccu::RemoteAddr remoteResult;
        remoteResult.addr = ccu::GetResByChannel<ccu::Variable>(
            channel, RANK16_RSAG_OUTPUT_ADDR_VARIABLE);
        for (uint32_t repeat = 0; repeat < owner; ++repeat) {
            remoteResult.addr += addresses.ownerSliceBytes;
        }
        remoteResult.token = ccu::GetResByChannel<ccu::Variable>(
            channel, RANK16_RSAG_OUTPUT_TOKEN_VARIABLE);
        uint16_t eventMask = static_cast<uint16_t>(1U << writeIndex++);
        F002_CCU_CHK_RET(ccu::Write(channel, remoteResult, localResult,
            addresses.ownerSliceBytes, writeEvent, eventMask));
    }
    F002_CCU_CHK_RET(
        ccu::EventWait(writeEvent, RANK16_RSAG_AG_EVENT_MASK));
    return CCU_SUCCESS;
}

CcuResult AllGatherRank16OwnerSliceF097Cyclic(
    DirectKernelContext &ctx,
    const std::array<ChannelHandle, RANK16_CLOS_CHANNEL_NUM> &pairChannels,
    Rank16ClosRsagAddresses &addresses)
{
    uint32_t owner = ctx.arg->myRank % RANK16_SERVER_SIZE;
    ccu::LocalAddr localResult;
    localResult.addr = addresses.output.addr;
    localResult.token = addresses.output.token;
    for (uint32_t repeat = 0; repeat < owner; ++repeat) {
        localResult.addr += addresses.ownerSliceBytes;
    }

    const auto &offsets = ctx.arg->myRank < RANK16_SERVER_SIZE ?
        F097_RANK16_AG_LOW_OFFSETS : F097_RANK16_AG_HIGH_OFFSETS;
    ccu::Event writeEvent;
    for (uint32_t writeIndex = 0;
        writeIndex < RANK16_SERVER_SIZE - 1; ++writeIndex) {
        uint32_t pair = (owner + offsets[writeIndex]) % RANK16_SERVER_SIZE;
        ChannelHandle channel = pairChannels[pair];
        ccu::RemoteAddr remoteResult;
        remoteResult.addr = ccu::GetResByChannel<ccu::Variable>(
            channel, RANK16_RSAG_OUTPUT_ADDR_VARIABLE);
        for (uint32_t repeat = 0; repeat < owner; ++repeat) {
            remoteResult.addr += addresses.ownerSliceBytes;
        }
        remoteResult.token = ccu::GetResByChannel<ccu::Variable>(
            channel, RANK16_RSAG_OUTPUT_TOKEN_VARIABLE);
        uint16_t eventMask = static_cast<uint16_t>(1U << writeIndex);
        F002_CCU_CHK_RET(ccu::Write(channel, remoteResult, localResult,
            addresses.ownerSliceBytes, writeEvent, eventMask));
    }
    F002_CCU_CHK_RET(
        ccu::EventWait(writeEvent, RANK16_RSAG_AG_EVENT_MASK));
    return CCU_SUCCESS;
}

CcuResult Rank16RsagFinalFence(DirectKernelContext &ctx)
{
    if (ctx.arg->myRank < RANK16_SERVER_SIZE) {
        // Every request is issued only after this rank's seven owner-slice
        // writes complete.  Eight acknowledgements therefore prove that all
        // seven remote writers have completed this rank's direct output.
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            F002_CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[index],
                DATA_COMPLETE_NOTIFY, RANK16_FINAL_FENCE_MASK));
        }
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            F002_CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[index],
                DATA_COMPLETE_NOTIFY, RANK16_FINAL_FENCE_MASK));
        }
    } else {
        // Consuming every low request proves that all low-to-high output
        // writes are complete; this rank's own writes completed before the
        // fence.  The directional ACK closes both completion and reuse.
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            F002_CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[index],
                DATA_COMPLETE_NOTIFY, RANK16_FINAL_FENCE_MASK));
        }
        for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
            F002_CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[index],
                DATA_COMPLETE_NOTIFY, RANK16_FINAL_FENCE_MASK));
        }
    }
    return CCU_SUCCESS;
}

void BuildRank12ClosChannels(const Rank12ClosTreeContext &ctx,
    std::array<ChannelHandle, RANK12_LARGE_SERVER_SIZE> &channels)
{
    uint32_t firstRemoteRank =
        ctx.arg->myRank < RANK12_LARGE_SERVER_SIZE ?
            RANK12_LARGE_SERVER_SIZE : 0;
    for (uint32_t index = 0; index < ctx.arg->channelCount; ++index) {
        channels[ctx.arg->remoteRanks[index] - firstRemoteRank] =
            ctx.arg->channels[index];
    }
}

bool IsRank12PairRelay(uint32_t rank)
{
    return rank < RANK12_PAIR_RELAY_NUM;
}

bool Rank12PairContains(uint32_t pairRank, uint32_t root)
{
    return IsRank12PairRelay(pairRank) &&
        (RANK12_PAIR_ROOTS[pairRank][0] == root ||
            RANK12_PAIR_ROOTS[pairRank][1] == root);
}

uint32_t FindRank12PairRelay(uint32_t firstRoot, uint32_t secondRoot)
{
    for (uint32_t pairRank = 0;
        pairRank < RANK12_PAIR_RELAY_NUM; ++pairRank) {
        if (Rank12PairContains(pairRank, firstRoot) &&
            Rank12PairContains(pairRank, secondRoot)) {
            return pairRank;
        }
    }
    return RANK12_PAIR_RELAY_NUM;
}

uint16_t Rank12RootReadyMask(uint32_t root)
{
    return static_cast<uint16_t>(1U << root);
}

uint16_t Rank12RootAckMask(uint32_t root)
{
    return static_cast<uint16_t>(1U << (root + 4));
}

uint16_t Rank12ChildReadyMask(uint32_t slice)
{
    return static_cast<uint16_t>(1U << (slice + 8));
}

uint16_t Rank12ChildAckMask(uint32_t slice)
{
    return static_cast<uint16_t>(1U << (slice + 12));
}

void AddRank12SliceOffset(
    ccu::Address &address, ccu::Variable &sliceBytes, uint32_t slice)
{
    for (uint32_t repeat = 0; repeat < slice; ++repeat) {
        address += sliceBytes;
    }
}

struct Rank12FourRootAddresses {
    ccu::LocalAddr source;
    ccu::LocalAddr output;
    ccu::Variable sliceBytes;
    // F039 root fusion needs only four quarter-slice temporaries. The Host
    // allocation remains deliberately unchanged; shrinking this translated
    // address table removes fifteen unused Variable-add instructions.
    std::array<ccu::LocalAddr, 4> scratch;
};

void BuildRank12FourRootAddresses(Rank12ClosTreeContext &ctx,
    Rank12FourRootAddresses &addresses)
{
    addresses.source.addr = ctx.sourceAddr;
    addresses.source.token = ctx.sourceToken;
    addresses.output.addr = ctx.outputAddr;
    addresses.output.token = ctx.outputToken;
    // The Host dispatch admits this Kernel only for the fixed 512 KiB case.
    // A constant quarter therefore avoids an unsupported CCU Variable divide
    // while remaining exactly q = dataBytes / 4.
    addresses.sliceBytes = RANK12_SLICE_BYTES;
    for (uint32_t lane = 0; lane < addresses.scratch.size(); ++lane) {
        addresses.scratch[lane].addr = ctx.scratchAddr;
        addresses.scratch[lane].token = ctx.scratchToken;
        AddRank12SliceOffset(
            addresses.scratch[lane].addr, addresses.sliceBytes, lane);
    }
}

CcuResult PublishRank12FourRootMetadata(Rank12ClosTreeContext &ctx,
    const std::array<ChannelHandle, RANK12_LARGE_SERVER_SIZE> &channels)
{
    if (ctx.arg->myRank < RANK12_LARGE_SERVER_SIZE) {
        bool pairRelay = IsRank12PairRelay(ctx.arg->myRank);
        for (uint32_t root = 0; root < RANK12_SMALL_SERVER_SIZE; ++root) {
            ChannelHandle channel = channels[root];
            if (pairRelay && Rank12PairContains(ctx.arg->myRank, root)) {
                // An incident pair exposes its output base as both the phase-2
                // partial source and the phase-4 child-result source.  The
                // token is deliberately withheld until both partials finish.
                F002_CCU_CHK_RET(ccu::WriteVariableWithNotify(channel,
                    ctx.outputAddr, RANK12_OUTPUT_ADDR_VARIABLE,
                    INPUT_READY_NOTIFY, RANK12_OUTPUT_ADDR_MASK));
            } else {
                // Exactly the five raw-A edges of each root publish source.
                F002_CCU_CHK_RET(ccu::WriteVariableWithNotify(channel,
                    ctx.sourceAddr, RANK12_SOURCE_ADDR_VARIABLE,
                    INPUT_READY_NOTIFY, RANK12_SOURCE_ADDR_MASK));
                F002_CCU_CHK_RET(ccu::WriteVariableWithNotify(channel,
                    ctx.sourceToken, RANK12_SOURCE_TOKEN_VARIABLE,
                    INPUT_READY_NOTIFY, RANK12_SOURCE_TOKEN_MASK));
            }
        }
        return CCU_SUCCESS;
    }

    uint32_t root = ctx.arg->myRank - RANK12_LARGE_SERVER_SIZE;
    for (uint32_t largeRank = 0;
        largeRank < RANK12_LARGE_SERVER_SIZE; ++largeRank) {
        ChannelHandle channel = channels[largeRank];
        // All eight A ranks pull this root's final slice from output.
        F002_CCU_CHK_RET(ccu::WriteVariableWithNotify(channel,
            ctx.outputAddr, RANK12_OUTPUT_ADDR_VARIABLE,
            INPUT_READY_NOTIFY, RANK12_OUTPUT_ADDR_MASK));
        F002_CCU_CHK_RET(ccu::WriteVariableWithNotify(channel,
            ctx.outputToken, RANK12_OUTPUT_TOKEN_VARIABLE,
            INPUT_READY_NOTIFY, RANK12_OUTPUT_TOKEN_MASK));

        if (IsRank12PairRelay(largeRank) &&
            Rank12PairContains(largeRank, root)) {
            // Only the three incident pair relays read a raw B source.
            F002_CCU_CHK_RET(ccu::WriteVariableWithNotify(channel,
                ctx.sourceAddr, RANK12_SOURCE_ADDR_VARIABLE,
                INPUT_READY_NOTIFY, RANK12_SOURCE_ADDR_MASK));
            F002_CCU_CHK_RET(ccu::WriteVariableWithNotify(channel,
                ctx.sourceToken, RANK12_SOURCE_TOKEN_VARIABLE,
                INPUT_READY_NOTIFY, RANK12_SOURCE_TOKEN_MASK));
        }
    }
    return CCU_SUCCESS;
}

CcuResult FormRank12PairPartials(Rank12ClosTreeContext &ctx,
    const std::array<ChannelHandle, RANK12_LARGE_SERVER_SIZE> &channels,
    Rank12FourRootAddresses &addresses)
{
    if (ctx.arg->myRank >= RANK12_PAIR_RELAY_NUM) {
        return CCU_SUCCESS;
    }

    uint32_t pairRank = ctx.arg->myRank;
    uint32_t firstRoot = RANK12_PAIR_ROOTS[pairRank][0];
    uint32_t lastRoot = RANK12_PAIR_ROOTS[pairRank][1];

    // Seed both incident A leaves with one contiguous local copy before
    // waiting for the two remote B source descriptors.  The six pair maps
    // are sorted, so copying [firstRoot, lastRoot] initializes both required
    // output slices; any interior slice is temporary and is overwritten by
    // its final root result before this rank returns.  One copy replaces the
    // former two post-network LocalReduce tasks and overlaps metadata waits.
    ccu::LocalAddr localSpanSource;
    localSpanSource.addr = addresses.source.addr;
    localSpanSource.token = addresses.source.token;
    AddRank12SliceOffset(
        localSpanSource.addr, addresses.sliceBytes, firstRoot);
    ccu::LocalAddr localSpanDestination;
    localSpanDestination.addr = addresses.output.addr;
    localSpanDestination.token = addresses.output.token;
    AddRank12SliceOffset(
        localSpanDestination.addr, addresses.sliceBytes, firstRoot);
    ccu::Variable localSpanBytes;
    localSpanBytes = 0;
    for (uint32_t root = firstRoot; root <= lastRoot; ++root) {
        localSpanBytes += addresses.sliceBytes;
    }
    ccu::Event partialEvent;
    F002_CCU_CHK_RET(ccu::LocalCopy(localSpanDestination, localSpanSource,
        localSpanBytes, partialEvent, SERIAL_EVENT_MASK));

    std::array<ccu::RemoteAddr, 2> remoteChildren;
    std::array<ccu::LocalAddr, 2> partials;
    for (uint32_t lane = 0; lane < 2; ++lane) {
        uint32_t root = RANK12_PAIR_ROOTS[pairRank][lane];
        uint32_t child = RANK12_PAIR_ROOTS[pairRank][1 - lane];
        ChannelHandle channel = channels[child];
        F002_CCU_CHK_RET(ccu::NotifyWait(channel,
            INPUT_READY_NOTIFY, RANK12_SOURCE_READY_MASK));
        remoteChildren[lane].addr = ccu::GetResByChannel<ccu::Variable>(
            channel, RANK12_SOURCE_ADDR_VARIABLE);
        remoteChildren[lane].token = ccu::GetResByChannel<ccu::Variable>(
            channel, RANK12_SOURCE_TOKEN_VARIABLE);
        AddRank12SliceOffset(
            remoteChildren[lane].addr, addresses.sliceBytes, root);

        // The pair partial lives in its final output slice.  Roots consume it
        // before root-ready allows this A rank to overwrite it with Y[root].
        partials[lane].addr = addresses.output.addr;
        partials[lane].token = addresses.output.token;
        AddRank12SliceOffset(partials[lane].addr, addresses.sliceBytes, root);
    }

    // Both incident output leaves must contain local A before the two remote
    // B values are folded into them.  Reusing the consumed Event bits keeps
    // the same two-way completion mask while changing Read+LocalReduce into
    // one preseed LocalCopy plus two ReadReduce operations.
    F002_CCU_CHK_RET(
        ccu::EventWait(partialEvent, SERIAL_EVENT_MASK));
    for (uint32_t lane = 0; lane < 2; ++lane) {
        uint32_t child = RANK12_PAIR_ROOTS[pairRank][1 - lane];
        F002_CCU_CHK_RET(ccu::ReadReduce(channels[child], partials[lane],
            remoteChildren[lane], addresses.sliceBytes,
            HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, partialEvent,
            static_cast<uint16_t>(1U << lane)));
    }
    F002_CCU_CHK_RET(
        ccu::EventWait(partialEvent, RANK12_TWO_WAY_EVENT_MASK));

    // Address bit 2 was published in phase 0.  Delaying bit 3 is both the
    // pair barrier and the proof that the two output-resident partials are
    // stable. Each incident direction gets exactly one token Record/Wait.
    for (uint32_t lane = 0; lane < 2; ++lane) {
        uint32_t root = RANK12_PAIR_ROOTS[pairRank][lane];
        F002_CCU_CHK_RET(ccu::WriteVariableWithNotify(channels[root],
            ctx.outputToken, RANK12_OUTPUT_TOKEN_VARIABLE,
            INPUT_READY_NOTIFY, RANK12_OUTPUT_TOKEN_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult FormRank12RootResult(Rank12ClosTreeContext &ctx,
    const std::array<ChannelHandle, RANK12_LARGE_SERVER_SIZE> &channels,
    Rank12FourRootAddresses &addresses)
{
    if (ctx.arg->myRank < RANK12_LARGE_SERVER_SIZE) {
        return CCU_SUCCESS;
    }

    uint32_t root = ctx.arg->myRank - RANK12_LARGE_SERVER_SIZE;
    std::array<ccu::LocalAddr, 5> leaves;
    leaves[0].addr = addresses.output.addr;
    leaves[0].token = addresses.output.token;
    AddRank12SliceOffset(leaves[0].addr, addresses.sliceBytes, root);
    for (uint32_t leaf = 1; leaf < leaves.size(); ++leaf) {
        leaves[leaf].addr = addresses.scratch[leaf - 1].addr;
        leaves[leaf].token = addresses.scratch[leaf - 1].token;
    }

    std::array<ccu::RemoteAddr, 5> remoteRawSources;
    std::array<ChannelHandle, 5> rawChannels{};
    uint32_t rawCount = 0;
    for (uint32_t largeRank = 0;
        largeRank < RANK12_LARGE_SERVER_SIZE; ++largeRank) {
        bool incidentPair = IsRank12PairRelay(largeRank) &&
            Rank12PairContains(largeRank, root);
        if (incidentPair) {
            continue;
        }
        ChannelHandle channel = channels[largeRank];
        F002_CCU_CHK_RET(ccu::NotifyWait(channel,
            INPUT_READY_NOTIFY, RANK12_SOURCE_READY_MASK));
        remoteRawSources[rawCount].addr =
            ccu::GetResByChannel<ccu::Variable>(
                channel, RANK12_SOURCE_ADDR_VARIABLE);
        remoteRawSources[rawCount].token =
            ccu::GetResByChannel<ccu::Variable>(
                channel, RANK12_SOURCE_TOKEN_VARIABLE);
        AddRank12SliceOffset(remoteRawSources[rawCount].addr,
            addresses.sliceBytes, root);
        rawChannels[rawCount] = channel;
        ++rawCount;
    }

    // Every root reads its five non-incident raw A leaves. Together with the
    // twelve B->pair Reads this covers all 32 Clos edges exactly once and in
    // one direction during phase 1.
    ccu::Event readEvent;
    for (uint32_t leaf = 0; leaf < remoteRawSources.size(); ++leaf) {
        F002_CCU_CHK_RET(ccu::Read(rawChannels[leaf], leaves[leaf],
            remoteRawSources[leaf], addresses.sliceBytes, readEvent,
            static_cast<uint16_t>(1U << leaf)));
    }
    F002_CCU_CHK_RET(
        ccu::EventWait(readEvent, RANK12_FIVE_WAY_EVENT_MASK));

    // Raw enumeration is rank ordered: the three non-incident pair A ranks,
    // A6, then A7. Fold local B_j into rawPair0 while independently combining
    // A6+A7. Both local destinations are disjoint, so one two-bit EventWait
    // can fence them after the incident-partial token waits below. Across the
    // resulting four fixed leaves every one of the 12 inputs appears once.
    ccu::LocalAddr localSource;
    localSource.addr = addresses.source.addr;
    localSource.token = addresses.source.token;
    AddRank12SliceOffset(localSource.addr, addresses.sliceBytes, root);
    ccu::Event reduceEvent;
    F002_CCU_CHK_RET(ccu::LocalReduce(leaves[0], localSource,
        addresses.sliceBytes, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM,
        reduceEvent, static_cast<uint16_t>(1U << 0)));
    F002_CCU_CHK_RET(ccu::LocalReduce(leaves[3], leaves[4],
        addresses.sliceBytes, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM,
        reduceEvent, static_cast<uint16_t>(1U << 1)));

    std::array<ccu::RemoteAddr, 3> remotePartials;
    std::array<ChannelHandle, 3> partialChannels{};
    uint32_t partialCount = 0;
    for (uint32_t pairRank = 0;
        pairRank < RANK12_PAIR_RELAY_NUM; ++pairRank) {
        if (!Rank12PairContains(pairRank, root)) {
            continue;
        }
        ChannelHandle channel = channels[pairRank];
        F002_CCU_CHK_RET(ccu::NotifyWait(channel,
            INPUT_READY_NOTIFY, RANK12_OUTPUT_READY_MASK));
        remotePartials[partialCount].addr =
            ccu::GetResByChannel<ccu::Variable>(
                channel, RANK12_OUTPUT_ADDR_VARIABLE);
        remotePartials[partialCount].token =
            ccu::GetResByChannel<ccu::Variable>(
                channel, RANK12_OUTPUT_TOKEN_VARIABLE);
        AddRank12SliceOffset(remotePartials[partialCount].addr,
            addresses.sliceBytes, root);
        partialChannels[partialCount] = channel;
        ++partialCount;
    }

    // The two local reductions are normally hidden by partial-token waits.
    // Fence both initial values before three disjoint ReadReduce operations.
    // Each ReadReduce consumes one phase-1 rawPair leaf and its matching
    // incident pair partial, directly producing the four-leaf tree input.
    F002_CCU_CHK_RET(
        ccu::EventWait(reduceEvent, RANK12_TWO_WAY_EVENT_MASK));
    for (uint32_t partial = 0;
        partial < remotePartials.size(); ++partial) {
        F002_CCU_CHK_RET(ccu::ReadReduce(partialChannels[partial],
            leaves[partial], remotePartials[partial],
            addresses.sliceBytes, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM,
            readEvent, static_cast<uint16_t>(1U << partial)));
    }
    F002_CCU_CHK_RET(
        ccu::EventWait(readEvent, RANK12_THREE_WAY_EVENT_MASK));

    // Fixed balanced tree over four aggregate leaves:
    //   [((rawPair0+B_j)+P0), (rawPair1+P1),
    //    (rawPair2+P2), (A6+A7)].
    // The three ReadReduce destinations are disjoint, so completion order and
    // channel registration order never affect the floating-point expression.
    for (uint32_t pair = 0; pair < 2; ++pair) {
        F002_CCU_CHK_RET(ccu::LocalReduce(leaves[pair * 2],
            leaves[pair * 2 + 1], addresses.sliceBytes,
            HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, reduceEvent,
            static_cast<uint16_t>(1U << pair)));
    }
    F002_CCU_CHK_RET(
        ccu::EventWait(reduceEvent, RANK12_TWO_WAY_EVENT_MASK));
    F002_CCU_CHK_RET(ccu::LocalReduce(leaves[0], leaves[2],
        addresses.sliceBytes, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM,
        reduceEvent, SERIAL_EVENT_MASK));
    F002_CCU_CHK_RET(ccu::EventWait(reduceEvent, SERIAL_EVENT_MASK));

    for (uint32_t largeRank = 0;
        largeRank < RANK12_LARGE_SERVER_SIZE; ++largeRank) {
        F002_CCU_CHK_RET(ccu::NotifyRecord(channels[largeRank],
            DATA_COMPLETE_NOTIFY, Rank12RootReadyMask(root)));
    }
    return CCU_SUCCESS;
}

CcuResult PullRank12RootResults(Rank12ClosTreeContext &ctx,
    const std::array<ChannelHandle, RANK12_LARGE_SERVER_SIZE> &channels,
    Rank12FourRootAddresses &addresses)
{
    if (ctx.arg->myRank >= RANK12_LARGE_SERVER_SIZE) {
        return CCU_SUCCESS;
    }

    std::array<ccu::RemoteAddr, RANK12_SMALL_SERVER_SIZE> remoteResults;
    std::array<ccu::LocalAddr, RANK12_SMALL_SERVER_SIZE> localResults;
    for (uint32_t root = 0; root < RANK12_SMALL_SERVER_SIZE; ++root) {
        ChannelHandle channel = channels[root];
        F002_CCU_CHK_RET(ccu::NotifyWait(channel,
            INPUT_READY_NOTIFY, RANK12_OUTPUT_READY_MASK));
        F002_CCU_CHK_RET(ccu::NotifyWait(channel,
            DATA_COMPLETE_NOTIFY, Rank12RootReadyMask(root)));
        remoteResults[root].addr = ccu::GetResByChannel<ccu::Variable>(
            channel, RANK12_OUTPUT_ADDR_VARIABLE);
        remoteResults[root].token = ccu::GetResByChannel<ccu::Variable>(
            channel, RANK12_OUTPUT_TOKEN_VARIABLE);
        AddRank12SliceOffset(
            remoteResults[root].addr, addresses.sliceBytes, root);
        localResults[root].addr = addresses.output.addr;
        localResults[root].token = addresses.output.token;
        AddRank12SliceOffset(
            localResults[root].addr, addresses.sliceBytes, root);
    }

    ccu::Event readEvent;
    for (uint32_t root = 0; root < RANK12_SMALL_SERVER_SIZE; ++root) {
        F002_CCU_CHK_RET(ccu::Read(channels[root], localResults[root],
            remoteResults[root], addresses.sliceBytes, readEvent,
            static_cast<uint16_t>(1U << root)));
    }
    F002_CCU_CHK_RET(
        ccu::EventWait(readEvent, RANK12_FOUR_WAY_EVENT_MASK));
    return CCU_SUCCESS;
}

CcuResult PublishRank12ChildReady(Rank12ClosTreeContext &ctx,
    const std::array<ChannelHandle, RANK12_LARGE_SERVER_SIZE> &channels)
{
    if (ctx.arg->myRank >= RANK12_PAIR_RELAY_NUM) {
        return CCU_SUCCESS;
    }

    uint32_t pairRank = ctx.arg->myRank;
    // For pair {j,k}, B_k pulls result slice j and B_j pulls result slice k.
    // The ready bits are emitted only after all four root Reads completed, so
    // the same output base that held partials now contains the final result.
    for (uint32_t lane = 0; lane < 2; ++lane) {
        uint32_t slice = RANK12_PAIR_ROOTS[pairRank][lane];
        uint32_t child = RANK12_PAIR_ROOTS[pairRank][1 - lane];
        F002_CCU_CHK_RET(ccu::NotifyRecord(channels[child],
            DATA_COMPLETE_NOTIFY, Rank12ChildReadyMask(slice)));
    }
    return CCU_SUCCESS;
}

CcuResult PullRank12ChildResultsAndAck(Rank12ClosTreeContext &ctx,
    const std::array<ChannelHandle, RANK12_LARGE_SERVER_SIZE> &channels,
    Rank12FourRootAddresses &addresses)
{
    if (ctx.arg->myRank < RANK12_LARGE_SERVER_SIZE) {
        return CCU_SUCCESS;
    }

    uint32_t root = ctx.arg->myRank - RANK12_LARGE_SERVER_SIZE;
    std::array<ccu::RemoteAddr, 3> remoteResults;
    std::array<ccu::LocalAddr, 3> localResults;
    std::array<ChannelHandle, 3> relayChannels{};
    std::array<uint32_t, 3> slices{};
    uint32_t result = 0;
    for (uint32_t slice = 0;
        slice < RANK12_SMALL_SERVER_SIZE; ++slice) {
        if (slice == root) {
            continue;
        }
        uint32_t pairRank = FindRank12PairRelay(root, slice);
        ChannelHandle channel = channels[pairRank];
        F002_CCU_CHK_RET(ccu::NotifyWait(channel,
            DATA_COMPLETE_NOTIFY, Rank12ChildReadyMask(slice)));
        // This edge's output address/token notification was consumed in
        // phase 2. Child-ready now proves the unchanged output base contains
        // the final slice rather than its earlier pair partial.
        remoteResults[result].addr = ccu::GetResByChannel<ccu::Variable>(
            channel, RANK12_OUTPUT_ADDR_VARIABLE);
        remoteResults[result].token = ccu::GetResByChannel<ccu::Variable>(
            channel, RANK12_OUTPUT_TOKEN_VARIABLE);
        AddRank12SliceOffset(
            remoteResults[result].addr, addresses.sliceBytes, slice);
        localResults[result].addr = addresses.output.addr;
        localResults[result].token = addresses.output.token;
        AddRank12SliceOffset(
            localResults[result].addr, addresses.sliceBytes, slice);
        relayChannels[result] = channel;
        slices[result] = slice;
        ++result;
    }

    ccu::Event readEvent;
    for (uint32_t index = 0; index < remoteResults.size(); ++index) {
        F002_CCU_CHK_RET(ccu::Read(relayChannels[index], localResults[index],
            remoteResults[index], addresses.sliceBytes, readEvent,
            static_cast<uint16_t>(1U << index)));
    }
    F002_CCU_CHK_RET(
        ccu::EventWait(readEvent, RANK12_THREE_WAY_EVENT_MASK));

    // ACK all three relay Reads before this B root waits for its eight A
    // readers. This ordering removes the root/child role cycle.
    for (uint32_t index = 0; index < relayChannels.size(); ++index) {
        F002_CCU_CHK_RET(ccu::NotifyRecord(relayChannels[index],
            DATA_COMPLETE_NOTIFY, Rank12ChildAckMask(slices[index])));
    }
    return CCU_SUCCESS;
}

CcuResult FinishRank12AAndAckRoots(Rank12ClosTreeContext &ctx,
    const std::array<ChannelHandle, RANK12_LARGE_SERVER_SIZE> &channels)
{
    if (ctx.arg->myRank >= RANK12_LARGE_SERVER_SIZE) {
        return CCU_SUCCESS;
    }

    if (IsRank12PairRelay(ctx.arg->myRank)) {
        uint32_t pairRank = ctx.arg->myRank;
        for (uint32_t lane = 0; lane < 2; ++lane) {
            uint32_t slice = RANK12_PAIR_ROOTS[pairRank][lane];
            uint32_t child = RANK12_PAIR_ROOTS[pairRank][1 - lane];
            F002_CCU_CHK_RET(ccu::NotifyWait(channels[child],
                DATA_COMPLETE_NOTIFY, Rank12ChildAckMask(slice)));
        }
    }

    // Pair relays delay all four root ACKs until both child Reads completed.
    // A6/A7 have no children and can ACK immediately after their four Reads.
    for (uint32_t root = 0; root < RANK12_SMALL_SERVER_SIZE; ++root) {
        F002_CCU_CHK_RET(ccu::NotifyRecord(channels[root],
            DATA_COMPLETE_NOTIFY, Rank12RootAckMask(root)));
    }
    return CCU_SUCCESS;
}

CcuResult WaitRank12RootAcks(Rank12ClosTreeContext &ctx,
    const std::array<ChannelHandle, RANK12_LARGE_SERVER_SIZE> &channels)
{
    if (ctx.arg->myRank < RANK12_LARGE_SERVER_SIZE) {
        return CCU_SUCCESS;
    }

    uint32_t root = ctx.arg->myRank - RANK12_LARGE_SERVER_SIZE;
    for (uint32_t largeRank = 0;
        largeRank < RANK12_LARGE_SERVER_SIZE; ++largeRank) {
        F002_CCU_CHK_RET(ccu::NotifyWait(channels[largeRank],
            DATA_COMPLETE_NOTIFY, Rank12RootAckMask(root)));
    }
    return CCU_SUCCESS;
}

CcuResult RunRank12ClosTree(Rank12ClosTreeContext &ctx)
{
    std::array<ChannelHandle, RANK12_LARGE_SERVER_SIZE> channels{};
    BuildRank12ClosChannels(ctx, channels);
    Rank12FourRootAddresses addresses;
    BuildRank12FourRootAddresses(ctx, addresses);

    F002_CCU_CHK_RET(PublishRank12FourRootMetadata(ctx, channels));
    F002_CCU_CHK_RET(FormRank12PairPartials(ctx, channels, addresses));
    F002_CCU_CHK_RET(FormRank12RootResult(ctx, channels, addresses));
    F002_CCU_CHK_RET(PullRank12RootResults(ctx, channels, addresses));
    F002_CCU_CHK_RET(PublishRank12ChildReady(ctx, channels));
    F002_CCU_CHK_RET(PullRank12ChildResultsAndAck(
        ctx, channels, addresses));
    F002_CCU_CHK_RET(FinishRank12AAndAckRoots(ctx, channels));
    F002_CCU_CHK_RET(WaitRank12RootAcks(ctx, channels));
    return CCU_SUCCESS;
}

CcuResult RunRank16ClosRsag(DirectKernelContext &ctx)
{
    std::array<ChannelHandle, RANK16_CLOS_CHANNEL_NUM> pairChannels{};
    BuildRank16PairChannels(ctx, pairChannels);

    Rank16ClosRsagAddresses addresses;
    BuildRank16ClosRsagAddresses(ctx, addresses);
    F002_CCU_CHK_RET(PublishRank16RsagBuffer(ctx, addresses));
    F002_CCU_CHK_RET(
        ReduceRank16RemoteOwnerSlice(ctx, pairChannels, addresses));
    F002_CCU_CHK_RET(
        CombineRank16OwnerPartials(ctx, pairChannels, addresses));
    F002_CCU_CHK_RET(
        AllGatherRank16OwnerSlice(ctx, pairChannels, addresses));
    F002_CCU_CHK_RET(Rank16RsagFinalFence(ctx));
    return CCU_SUCCESS;
}

CcuResult RunRank16ClosRsagF097CyclicAg(DirectKernelContext &ctx)
{
    std::array<ChannelHandle, RANK16_CLOS_CHANNEL_NUM> pairChannels{};
    BuildRank16PairChannels(ctx, pairChannels);

    Rank16ClosRsagAddresses addresses;
    BuildRank16ClosRsagAddresses(ctx, addresses);
    F002_CCU_CHK_RET(PublishRank16RsagBuffer(ctx, addresses));
    F002_CCU_CHK_RET(
        ReduceRank16RemoteOwnerSlice(ctx, pairChannels, addresses));
    F002_CCU_CHK_RET(
        CombineRank16OwnerPartials(ctx, pairChannels, addresses));
    F002_CCU_CHK_RET(
        AllGatherRank16OwnerSliceF097Cyclic(
            ctx, pairChannels, addresses));
    F002_CCU_CHK_RET(Rank16RsagFinalFence(ctx));
    return CCU_SUCCESS;
}
} // namespace

CcuResult CcuGroupedPairKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgGroup *>(arg);
    DirectKernelContext ctx;
    ctx.arg = kernelArg;

    F002_CCU_CHK_RET(LoadTaskArgs(ctx));
    F002_CCU_CHK_RET(InitChannelResources(ctx));

    CCU_IF(ctx.phase == static_cast<uint32_t>(DirectKernelPhase::INPUT_RECORD))
    {
        F002_CCU_CHK_RET(RecordInputReady(ctx));
    }
    CCU_IF(ctx.phase == static_cast<uint32_t>(DirectKernelPhase::ONESHOT_READ_RECORD))
    {
        F002_CCU_CHK_RET(OneShotReadAndRecord(ctx));
    }
    CCU_IF(ctx.phase == static_cast<uint32_t>(DirectKernelPhase::ONESHOT_DONE_WAIT))
    {
        F002_CCU_CHK_RET(WaitDataComplete(ctx));
    }
    CCU_IF(ctx.phase == static_cast<uint32_t>(DirectKernelPhase::ONESHOT_LOCAL_REDUCE))
    {
        F002_CCU_CHK_RET(OneShotLocalReduce(ctx));
    }
    CCU_IF(ctx.phase == static_cast<uint32_t>(DirectKernelPhase::DIRECT_REDUCE_SCATTER))
    {
        F002_CCU_CHK_RET(DirectReduceScatter(ctx));
    }
    CCU_IF(ctx.phase == static_cast<uint32_t>(DirectKernelPhase::DIRECT_ALL_GATHER_RECORD))
    {
        F002_CCU_CHK_RET(DirectAllGatherAndRecord(ctx));
    }
    CCU_IF(ctx.phase == static_cast<uint32_t>(DirectKernelPhase::DIRECT_ALL_GATHER_WAIT))
    {
        F002_CCU_CHK_RET(WaitDataComplete(ctx));
    }
    // This ordinary registration-time condition prevents rank-12/16 grouped
    // kernels from translating or reserving resources for the rank-4-only
    // fused graph. Their existing phase-12 path remains byte-for-byte isolated.
    if (ctx.arg->rankSize == FOUR_RANK_NHR_SIZE &&
        ctx.arg->groupCount == 1 && ctx.arg->groupIndex == 0 &&
        ctx.arg->channelCount == FOUR_RANK_NHR_SIZE - 1) {
        CCU_IF(ctx.phase == static_cast<uint32_t>(DirectKernelPhase::FUSED_ONESHOT_ALL_REDUCE))
        {
            F002_CCU_CHK_RET(FusedOneShotAllReduce(ctx));
        }
    }
    // The rank-16 phase-14 graph may use only the full eight-edge Clos group.
    // Keeping this an ordinary registration-time condition prevents the
    // seven-edge Mesh group (the other IO Die) from translating any phase-14
    // communication primitive or reserving its resources.
    if (IsRank16ClosRsagGroup(ctx)) {
        CCU_IF(ctx.phase ==
            static_cast<uint32_t>(DirectKernelPhase::RANK16_CLOS_PAIR_TREE))
        {
            F002_CCU_CHK_RET(RunRank16ClosRsag(ctx));
        }
    }
    return CCU_SUCCESS;
}

CcuResult CcuParallelRsKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgGroup *>(arg);
    DirectKernelContext ctx;
    ctx.arg = kernelArg;

    F002_CCU_CHK_RET(LoadTaskArgs(ctx));
    F002_CCU_CHK_RET(InitChannelResources(ctx));

    CCU_IF(ctx.phase == static_cast<uint32_t>(DirectKernelPhase::PARALLEL_REDUCE_SCATTER))
    {
        F002_CCU_CHK_RET(ParallelReduceScatter(ctx));
    }
    CCU_IF(ctx.phase == static_cast<uint32_t>(DirectKernelPhase::PARALLEL_REDUCE_MERGE))
    {
        F002_CCU_CHK_RET(MergeParallelReduceScatter(ctx));
    }
    CCU_IF(ctx.phase == static_cast<uint32_t>(DirectKernelPhase::FUSED_ONESHOT_COMMUNICATION))
    {
        F002_CCU_CHK_RET(FusedOneShotCommunication(ctx));
    }
    return CCU_SUCCESS;
}

CcuResult CcuRank12ClosTreeKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgGroup *>(arg);
    Rank12ClosTreeContext ctx;
    ctx.arg = kernelArg;

    F002_CCU_CHK_RET(LoadRank12ClosTreeArgs(ctx));
    F002_CCU_CHK_RET(InitRank12ClosTreeResources(ctx));
    F002_CCU_CHK_RET(RunRank12ClosTree(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuFourRankNhrKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgGroup *>(arg);
    FourRankNhrContext ctx;
    ctx.arg = kernelArg;

    F002_CCU_CHK_RET(LoadFourRankNhrArgs(ctx));
    F002_CCU_CHK_RET(InitFourRankNhrResources(ctx));
    F002_CCU_CHK_RET(RunFourRankNhr(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuDirectBufferKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgGroup *>(arg);
    DirectBufferContext ctx;
    ctx.arg = kernelArg;

    F002_CCU_CHK_RET(LoadDirectBufferArgs(ctx));
    F002_CCU_CHK_RET(InitDirectBufferResources(ctx));

    CCU_IF(ctx.phase ==
        static_cast<uint32_t>(DirectKernelPhase::DIRECT_BUFFER_REDUCE_SCATTER))
    {
        F002_CCU_CHK_RET(DirectBufferReduceScatter(ctx));
    }
    CCU_IF(ctx.phase ==
        static_cast<uint32_t>(DirectKernelPhase::DIRECT_BUFFER_MERGE))
    {
        F002_CCU_CHK_RET(MergeDirectBufferPartials(ctx));
    }
    CCU_IF(ctx.phase ==
        static_cast<uint32_t>(DirectKernelPhase::DIRECT_BUFFER_ALL_GATHER))
    {
        F002_CCU_CHK_RET(DirectBufferAllGather(ctx));
    }
    return CCU_SUCCESS;
}

CcuResult CcuRank4SmallProbeKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgGroup *>(arg);
    DirectKernelContext ctx;
    ctx.arg = kernelArg;
    F002_CCU_CHK_RET(LoadTaskArgs(ctx));
    F002_CCU_CHK_RET(InitChannelResources(ctx));
    if (ctx.arg->rankSize == FOUR_RANK_NHR_SIZE &&
        ctx.arg->groupCount == 1 && ctx.arg->groupIndex == 0 &&
        ctx.arg->channelCount == FOUR_RANK_NHR_SIZE - 1) {
        CCU_IF(ctx.phase == static_cast<uint32_t>(
            DirectKernelPhase::FUSED_ONESHOT_ALL_REDUCE))
        {
            F002_CCU_CHK_RET(FusedOneShotAllReduce(ctx));
        }
    }
    return CCU_SUCCESS;
}

CcuResult CcuRank4Large512ProbeKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgGroup *>(arg);
    FourRankNhrContext ctx;
    ctx.arg = kernelArg;
    F002_CCU_CHK_RET(LoadFourRankNhrArgs(ctx));
    F002_CCU_CHK_RET(InitFourRankNhrResources(ctx));
    F002_CCU_CHK_RET(RunFourRankNhr(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuRank4Large400MiB4BProbeKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgGroup *>(arg);
    FourRankNhrContext ctx;
    ctx.arg = kernelArg;
    F002_CCU_CHK_RET(LoadFourRankNhrArgs(ctx));
    F002_CCU_CHK_RET(InitFourRankNhrResources(ctx));
    F002_CCU_CHK_RET(RunFourRankNhr(ctx));
    return CCU_SUCCESS;
}

CcuResult RunF085Rank12ClosTree(Rank12ClosTreeContext &ctx);

CcuResult CcuRank12SmallProbeKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgGroup *>(arg);
    Rank12ClosTreeContext ctx;
    ctx.arg = kernelArg;
    F002_CCU_CHK_RET(LoadRank12ClosTreeArgs(ctx));
    F002_CCU_CHK_RET(InitRank12ClosTreeResources(ctx));
    F002_CCU_CHK_RET(RunF085Rank12ClosTree(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuRank12Large512ProbeKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgGroup *>(arg);
    DirectBufferContext ctx;
    ctx.arg = kernelArg;
    F002_CCU_CHK_RET(LoadDirectBufferArgs(ctx));
    F002_CCU_CHK_RET(InitDirectBufferResources(ctx));
    CCU_IF(ctx.phase == static_cast<uint32_t>(
        DirectKernelPhase::DIRECT_BUFFER_REDUCE_SCATTER))
    {
        F002_CCU_CHK_RET(DirectBufferReduceScatter(ctx));
    }
    CCU_IF(ctx.phase == static_cast<uint32_t>(
        DirectKernelPhase::DIRECT_BUFFER_MERGE))
    {
        F002_CCU_CHK_RET(MergeDirectBufferPartials(ctx));
    }
    CCU_IF(ctx.phase == static_cast<uint32_t>(
        DirectKernelPhase::DIRECT_BUFFER_ALL_GATHER))
    {
        F002_CCU_CHK_RET(DirectBufferAllGather(ctx));
    }
    return CCU_SUCCESS;
}

CcuResult CcuRank12Large400MiB4BProbeKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgGroup *>(arg);
    DirectBufferContext ctx;
    ctx.arg = kernelArg;
    F002_CCU_CHK_RET(LoadDirectBufferArgs(ctx));
    F002_CCU_CHK_RET(InitDirectBufferResources(ctx));
    CCU_IF(ctx.phase == static_cast<uint32_t>(
        DirectKernelPhase::DIRECT_BUFFER_REDUCE_SCATTER))
    {
        F002_CCU_CHK_RET(DirectBufferReduceScatter(ctx));
    }
    CCU_IF(ctx.phase == static_cast<uint32_t>(
        DirectKernelPhase::DIRECT_BUFFER_MERGE))
    {
        F002_CCU_CHK_RET(MergeDirectBufferPartials(ctx));
    }
    CCU_IF(ctx.phase == static_cast<uint32_t>(
        DirectKernelPhase::DIRECT_BUFFER_ALL_GATHER))
    {
        F002_CCU_CHK_RET(DirectBufferAllGather(ctx));
    }
    return CCU_SUCCESS;
}

CcuResult CcuRank16SmallProbeKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgGroup *>(arg);
    DirectKernelContext ctx;
    ctx.arg = kernelArg;
    F002_CCU_CHK_RET(LoadTaskArgs(ctx));
    F002_CCU_CHK_RET(InitChannelResources(ctx));
    if (IsRank16ClosRsagGroup(ctx)) {
        CCU_IF(ctx.phase == static_cast<uint32_t>(
            DirectKernelPhase::RANK16_CLOS_PAIR_TREE))
        {
            F002_CCU_CHK_RET(RunRank16ClosRsagF097CyclicAg(ctx));
        }
    }
    return CCU_SUCCESS;
}

CcuResult CcuRank16Large512ProbeKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgGroup *>(arg);
    DirectBufferContext ctx;
    ctx.arg = kernelArg;
    F002_CCU_CHK_RET(LoadDirectBufferArgs(ctx));
    F002_CCU_CHK_RET(InitDirectBufferResources(ctx));
    CCU_IF(ctx.phase == static_cast<uint32_t>(
        DirectKernelPhase::DIRECT_BUFFER_REDUCE_SCATTER))
    {
        F002_CCU_CHK_RET(DirectBufferReduceScatter(ctx));
    }
    CCU_IF(ctx.phase == static_cast<uint32_t>(
        DirectKernelPhase::DIRECT_BUFFER_MERGE))
    {
        F002_CCU_CHK_RET(MergeDirectBufferPartials(ctx));
    }
    CCU_IF(ctx.phase == static_cast<uint32_t>(
        DirectKernelPhase::DIRECT_BUFFER_ALL_GATHER))
    {
        F002_CCU_CHK_RET(DirectBufferAllGather(ctx));
    }
    return CCU_SUCCESS;
}

CcuResult CcuRank16Large400MiB4BProbeKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgGroup *>(arg);
    DirectBufferContext ctx;
    ctx.arg = kernelArg;
    F002_CCU_CHK_RET(LoadDirectBufferArgs(ctx));
    F002_CCU_CHK_RET(InitDirectBufferResources(ctx));
    CCU_IF(ctx.phase == static_cast<uint32_t>(
        DirectKernelPhase::DIRECT_BUFFER_REDUCE_SCATTER))
    {
        F002_CCU_CHK_RET(DirectBufferReduceScatter(ctx));
    }
    CCU_IF(ctx.phase == static_cast<uint32_t>(
        DirectKernelPhase::DIRECT_BUFFER_MERGE))
    {
        F002_CCU_CHK_RET(MergeDirectBufferPartials(ctx));
    }
    CCU_IF(ctx.phase == static_cast<uint32_t>(
        DirectKernelPhase::DIRECT_BUFFER_ALL_GATHER))
    {
        F002_CCU_CHK_RET(DirectBufferAllGather(ctx));
    }
    return CCU_SUCCESS;
}

// F085 point-16-only copy of the F080 rank12 full-push graph.  These helpers
// intentionally do not replace the shared F082 rank12 helpers: generic rank12
// and both exact large-message entries remain on their original graphs.
uint16_t F085Rank12RootDoneMask(uint32_t root)
{
    return static_cast<uint16_t>(1U << root);
}

uint16_t F085Rank12PushDoneMask(uint32_t slice)
{
    return static_cast<uint16_t>(1U << (slice + 8));
}

CcuResult F085PublishRank12FourRootMetadata(Rank12ClosTreeContext &ctx,
    const std::array<ChannelHandle, RANK12_LARGE_SERVER_SIZE> &channels)
{
    if (ctx.arg->myRank < RANK12_LARGE_SERVER_SIZE) {
        bool pairRelay = IsRank12PairRelay(ctx.arg->myRank);
        for (uint32_t root = 0; root < RANK12_SMALL_SERVER_SIZE; ++root) {
            ChannelHandle channel = channels[root];
            bool incident = pairRelay &&
                Rank12PairContains(ctx.arg->myRank, root);
            if (!incident) {
                // A non-incident edge is first read by B from source and later
                // written by B into output, so publish both disjoint buffers.
                F002_CCU_CHK_RET(ccu::WriteVariableWithNotify(channel,
                    ctx.sourceAddr, RANK12_SOURCE_ADDR_VARIABLE,
                    INPUT_READY_NOTIFY, RANK12_SOURCE_ADDR_MASK));
                F002_CCU_CHK_RET(ccu::WriteVariableWithNotify(channel,
                    ctx.sourceToken, RANK12_SOURCE_TOKEN_VARIABLE,
                    INPUT_READY_NOTIFY, RANK12_SOURCE_TOKEN_MASK));
            }
            F002_CCU_CHK_RET(ccu::WriteVariableWithNotify(channel,
                ctx.outputAddr, RANK12_OUTPUT_ADDR_VARIABLE,
                INPUT_READY_NOTIFY, RANK12_OUTPUT_ADDR_MASK));
            if (!incident) {
                F002_CCU_CHK_RET(ccu::WriteVariableWithNotify(channel,
                    ctx.outputToken, RANK12_OUTPUT_TOKEN_VARIABLE,
                    INPUT_READY_NOTIFY, RANK12_OUTPUT_TOKEN_MASK));
            }
        }
        return CCU_SUCCESS;
    }

    uint32_t root = ctx.arg->myRank - RANK12_LARGE_SERVER_SIZE;
    for (uint32_t largeRank = 0;
        largeRank < RANK12_LARGE_SERVER_SIZE; ++largeRank) {
        ChannelHandle channel = channels[largeRank];
        if (IsRank12PairRelay(largeRank) &&
            Rank12PairContains(largeRank, root)) {
            // An incident pair first reads this B source and later pushes one
            // child slice into this B output, so it needs all four variables.
            F002_CCU_CHK_RET(ccu::WriteVariableWithNotify(channel,
                ctx.sourceAddr, RANK12_SOURCE_ADDR_VARIABLE,
                INPUT_READY_NOTIFY, RANK12_SOURCE_ADDR_MASK));
            F002_CCU_CHK_RET(ccu::WriteVariableWithNotify(channel,
                ctx.sourceToken, RANK12_SOURCE_TOKEN_VARIABLE,
                INPUT_READY_NOTIFY, RANK12_SOURCE_TOKEN_MASK));
            F002_CCU_CHK_RET(ccu::WriteVariableWithNotify(channel,
                ctx.outputAddr, RANK12_OUTPUT_ADDR_VARIABLE,
                INPUT_READY_NOTIFY, RANK12_OUTPUT_ADDR_MASK));
            F002_CCU_CHK_RET(ccu::WriteVariableWithNotify(channel,
                ctx.outputToken, RANK12_OUTPUT_TOKEN_VARIABLE,
                INPUT_READY_NOTIFY, RANK12_OUTPUT_TOKEN_MASK));
        }
    }
    return CCU_SUCCESS;
}

CcuResult F085FormRank12PairPartials(Rank12ClosTreeContext &ctx,
    const std::array<ChannelHandle, RANK12_LARGE_SERVER_SIZE> &channels,
    Rank12FourRootAddresses &addresses)
{
    if (ctx.arg->myRank >= RANK12_PAIR_RELAY_NUM) {
        return CCU_SUCCESS;
    }

    uint32_t pairRank = ctx.arg->myRank;
    uint32_t firstRoot = RANK12_PAIR_ROOTS[pairRank][0];
    uint32_t lastRoot = RANK12_PAIR_ROOTS[pairRank][1];

    // Seed only the two incident A leaves. Adjacent pairs use one contiguous
    // 2q Copy; non-adjacent pairs use two disjoint q Copies in parallel. The
    // latter must never overwrite a concurrently pushed nonincident slice.
    ccu::Event partialEvent;
    uint16_t localCopyMask = SERIAL_EVENT_MASK;
    if (lastRoot == firstRoot + 1) {
        ccu::LocalAddr localSpanSource;
        localSpanSource.addr = addresses.source.addr;
        localSpanSource.token = addresses.source.token;
        AddRank12SliceOffset(
            localSpanSource.addr, addresses.sliceBytes, firstRoot);
        ccu::LocalAddr localSpanDestination;
        localSpanDestination.addr = addresses.output.addr;
        localSpanDestination.token = addresses.output.token;
        AddRank12SliceOffset(
            localSpanDestination.addr, addresses.sliceBytes, firstRoot);
        ccu::Variable localSpanBytes;
        localSpanBytes = addresses.sliceBytes;
        localSpanBytes += addresses.sliceBytes;
        F002_CCU_CHK_RET(ccu::LocalCopy(localSpanDestination,
            localSpanSource, localSpanBytes, partialEvent,
            SERIAL_EVENT_MASK));
    } else {
        localCopyMask = RANK12_TWO_WAY_EVENT_MASK;
        for (uint32_t lane = 0; lane < 2; ++lane) {
            uint32_t root = RANK12_PAIR_ROOTS[pairRank][lane];
            ccu::LocalAddr localSource;
            localSource.addr = addresses.source.addr;
            localSource.token = addresses.source.token;
            AddRank12SliceOffset(
                localSource.addr, addresses.sliceBytes, root);
            ccu::LocalAddr localDestination;
            localDestination.addr = addresses.output.addr;
            localDestination.token = addresses.output.token;
            AddRank12SliceOffset(
                localDestination.addr, addresses.sliceBytes, root);
            F002_CCU_CHK_RET(ccu::LocalCopy(localDestination, localSource,
                addresses.sliceBytes, partialEvent,
                static_cast<uint16_t>(1U << lane)));
        }
    }

    std::array<ccu::RemoteAddr, 2> remoteChildren;
    std::array<ccu::LocalAddr, 2> partials;
    for (uint32_t lane = 0; lane < 2; ++lane) {
        uint32_t root = RANK12_PAIR_ROOTS[pairRank][lane];
        uint32_t child = RANK12_PAIR_ROOTS[pairRank][1 - lane];
        ChannelHandle channel = channels[child];
        // One 0xF wait obtains the B source used now and caches the B output
        // destination used by this pair's later child push.
        F002_CCU_CHK_RET(ccu::NotifyWait(channel,
            INPUT_READY_NOTIFY, F085_RANK12_ALL_BUFFER_READY_MASK));
        remoteChildren[lane].addr = ccu::GetResByChannel<ccu::Variable>(
            channel, RANK12_SOURCE_ADDR_VARIABLE);
        remoteChildren[lane].token = ccu::GetResByChannel<ccu::Variable>(
            channel, RANK12_SOURCE_TOKEN_VARIABLE);
        AddRank12SliceOffset(
            remoteChildren[lane].addr, addresses.sliceBytes, root);

        partials[lane].addr = addresses.output.addr;
        partials[lane].token = addresses.output.token;
        AddRank12SliceOffset(partials[lane].addr, addresses.sliceBytes, root);
    }

    F002_CCU_CHK_RET(ccu::EventWait(partialEvent, localCopyMask));
    for (uint32_t lane = 0; lane < 2; ++lane) {
        uint32_t child = RANK12_PAIR_ROOTS[pairRank][1 - lane];
        F002_CCU_CHK_RET(ccu::ReadReduce(channels[child], partials[lane],
            remoteChildren[lane], addresses.sliceBytes,
            HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, partialEvent,
            static_cast<uint16_t>(1U << lane)));
    }
    F002_CCU_CHK_RET(
        ccu::EventWait(partialEvent, RANK12_TWO_WAY_EVENT_MASK));

    // Address bit 2 was published in phase 0. Delaying bit 3 is both the
    // pair barrier and proof that both output-resident partials are stable.
    for (uint32_t lane = 0; lane < 2; ++lane) {
        uint32_t root = RANK12_PAIR_ROOTS[pairRank][lane];
        F002_CCU_CHK_RET(ccu::WriteVariableWithNotify(channels[root],
            ctx.outputToken, RANK12_OUTPUT_TOKEN_VARIABLE,
            INPUT_READY_NOTIFY, RANK12_OUTPUT_TOKEN_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult F085FormRank12RootResult(Rank12ClosTreeContext &ctx,
    const std::array<ChannelHandle, RANK12_LARGE_SERVER_SIZE> &channels,
    Rank12FourRootAddresses &addresses)
{
    if (ctx.arg->myRank < RANK12_LARGE_SERVER_SIZE) {
        return CCU_SUCCESS;
    }

    uint32_t root = ctx.arg->myRank - RANK12_LARGE_SERVER_SIZE;
    std::array<ccu::LocalAddr, 5> leaves;
    leaves[0].addr = addresses.output.addr;
    leaves[0].token = addresses.output.token;
    AddRank12SliceOffset(leaves[0].addr, addresses.sliceBytes, root);
    for (uint32_t leaf = 1; leaf < leaves.size(); ++leaf) {
        leaves[leaf].addr = addresses.scratch[leaf - 1].addr;
        leaves[leaf].token = addresses.scratch[leaf - 1].token;
    }

    std::array<ccu::RemoteAddr, 5> remoteRawSources;
    std::array<ChannelHandle, 5> rawChannels{};
    uint32_t rawCount = 0;
    for (uint32_t largeRank = 0;
        largeRank < RANK12_LARGE_SERVER_SIZE; ++largeRank) {
        bool incidentPair = IsRank12PairRelay(largeRank) &&
            Rank12PairContains(largeRank, root);
        if (incidentPair) {
            continue;
        }
        ChannelHandle channel = channels[largeRank];
        // This wait obtains both the raw source consumed below and the output
        // destination used by the later root push on the same edge.
        F002_CCU_CHK_RET(ccu::NotifyWait(channel,
            INPUT_READY_NOTIFY, F085_RANK12_ALL_BUFFER_READY_MASK));
        remoteRawSources[rawCount].addr =
            ccu::GetResByChannel<ccu::Variable>(
                channel, RANK12_SOURCE_ADDR_VARIABLE);
        remoteRawSources[rawCount].token =
            ccu::GetResByChannel<ccu::Variable>(
                channel, RANK12_SOURCE_TOKEN_VARIABLE);
        AddRank12SliceOffset(remoteRawSources[rawCount].addr,
            addresses.sliceBytes, root);
        rawChannels[rawCount] = channel;
        ++rawCount;
    }

    // Five non-incident raw A leaves plus the three incident pair partials
    // cover all eight A ranks exactly once for this root.
    ccu::Event readEvent;
    for (uint32_t leaf = 0; leaf < remoteRawSources.size(); ++leaf) {
        F002_CCU_CHK_RET(ccu::Read(rawChannels[leaf], leaves[leaf],
            remoteRawSources[leaf], addresses.sliceBytes, readEvent,
            static_cast<uint16_t>(1U << leaf)));
    }
    F002_CCU_CHK_RET(
        ccu::EventWait(readEvent, RANK12_FIVE_WAY_EVENT_MASK));

    ccu::LocalAddr localSource;
    localSource.addr = addresses.source.addr;
    localSource.token = addresses.source.token;
    AddRank12SliceOffset(localSource.addr, addresses.sliceBytes, root);
    ccu::Event reduceEvent;
    F002_CCU_CHK_RET(ccu::LocalReduce(leaves[0], localSource,
        addresses.sliceBytes, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM,
        reduceEvent, static_cast<uint16_t>(1U << 0)));
    F002_CCU_CHK_RET(ccu::LocalReduce(leaves[3], leaves[4],
        addresses.sliceBytes, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM,
        reduceEvent, static_cast<uint16_t>(1U << 1)));

    std::array<ccu::RemoteAddr, 3> remotePartials;
    std::array<ChannelHandle, 3> partialChannels{};
    uint32_t partialCount = 0;
    for (uint32_t pairRank = 0;
        pairRank < RANK12_PAIR_RELAY_NUM; ++pairRank) {
        if (!Rank12PairContains(pairRank, root)) {
            continue;
        }
        ChannelHandle channel = channels[pairRank];
        F002_CCU_CHK_RET(ccu::NotifyWait(channel,
            INPUT_READY_NOTIFY, RANK12_OUTPUT_READY_MASK));
        remotePartials[partialCount].addr =
            ccu::GetResByChannel<ccu::Variable>(
                channel, RANK12_OUTPUT_ADDR_VARIABLE);
        remotePartials[partialCount].token =
            ccu::GetResByChannel<ccu::Variable>(
                channel, RANK12_OUTPUT_TOKEN_VARIABLE);
        AddRank12SliceOffset(remotePartials[partialCount].addr,
            addresses.sliceBytes, root);
        partialChannels[partialCount] = channel;
        ++partialCount;
    }

    F002_CCU_CHK_RET(
        ccu::EventWait(reduceEvent, RANK12_TWO_WAY_EVENT_MASK));
    for (uint32_t partial = 0;
        partial < remotePartials.size(); ++partial) {
        F002_CCU_CHK_RET(ccu::ReadReduce(partialChannels[partial],
            leaves[partial], remotePartials[partial],
            addresses.sliceBytes, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM,
            readEvent, static_cast<uint16_t>(1U << partial)));
    }
    F002_CCU_CHK_RET(
        ccu::EventWait(readEvent, RANK12_THREE_WAY_EVENT_MASK));

    // Preserve F080's fixed balanced FP32 expression tree.
    for (uint32_t pair = 0; pair < 2; ++pair) {
        F002_CCU_CHK_RET(ccu::LocalReduce(leaves[pair * 2],
            leaves[pair * 2 + 1], addresses.sliceBytes,
            HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, reduceEvent,
            static_cast<uint16_t>(1U << pair)));
    }
    F002_CCU_CHK_RET(
        ccu::EventWait(reduceEvent, RANK12_TWO_WAY_EVENT_MASK));
    F002_CCU_CHK_RET(ccu::LocalReduce(leaves[0], leaves[2],
        addresses.sliceBytes, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM,
        reduceEvent, SERIAL_EVENT_MASK));
    F002_CCU_CHK_RET(ccu::EventWait(reduceEvent, SERIAL_EVENT_MASK));

    // Producer-owned root broadcast: four B roots each push one final slice
    // to all eight A ranks, fence all Writes, then publish RootDone.
    std::array<ccu::RemoteAddr, RANK12_LARGE_SERVER_SIZE> remoteOutputs;
    for (uint32_t largeRank = 0;
        largeRank < RANK12_LARGE_SERVER_SIZE; ++largeRank) {
        ChannelHandle channel = channels[largeRank];
        remoteOutputs[largeRank].addr =
            ccu::GetResByChannel<ccu::Variable>(
                channel, RANK12_OUTPUT_ADDR_VARIABLE);
        remoteOutputs[largeRank].token =
            ccu::GetResByChannel<ccu::Variable>(
                channel, RANK12_OUTPUT_TOKEN_VARIABLE);
        AddRank12SliceOffset(remoteOutputs[largeRank].addr,
            addresses.sliceBytes, root);
        F002_CCU_CHK_RET(ccu::Write(channel, remoteOutputs[largeRank],
            leaves[0], addresses.sliceBytes, readEvent,
            static_cast<uint16_t>(1U << largeRank)));
    }
    F002_CCU_CHK_RET(
        ccu::EventWait(readEvent, F085_RANK12_EIGHT_WAY_EVENT_MASK));

    for (uint32_t largeRank = 0;
        largeRank < RANK12_LARGE_SERVER_SIZE; ++largeRank) {
        F002_CCU_CHK_RET(ccu::NotifyRecord(channels[largeRank],
            DATA_COMPLETE_NOTIFY, F085Rank12RootDoneMask(root)));
    }
    return CCU_SUCCESS;
}

CcuResult F085PushRank12ChildResults(Rank12ClosTreeContext &ctx,
    const std::array<ChannelHandle, RANK12_LARGE_SERVER_SIZE> &channels,
    Rank12FourRootAddresses &addresses)
{
    if (ctx.arg->myRank >= RANK12_LARGE_SERVER_SIZE) {
        return CCU_SUCCESS;
    }

    if (!IsRank12PairRelay(ctx.arg->myRank)) {
        // A6/A7 have no child role, but must consume all RootDone bits before
        // the next round reuses the one-bit generations.
        for (uint32_t root = 0;
            root < RANK12_SMALL_SERVER_SIZE; ++root) {
            F002_CCU_CHK_RET(ccu::NotifyWait(channels[root],
                DATA_COMPLETE_NOTIFY, F085Rank12RootDoneMask(root)));
        }
        return CCU_SUCCESS;
    }

    uint32_t pairRank = ctx.arg->myRank;
    for (uint32_t lane = 0; lane < 2; ++lane) {
        uint32_t root = RANK12_PAIR_ROOTS[pairRank][lane];
        F002_CCU_CHK_RET(ccu::NotifyWait(channels[root],
            DATA_COMPLETE_NOTIFY, F085Rank12RootDoneMask(root)));
    }

    std::array<ccu::LocalAddr, 2> localResults;
    std::array<ccu::RemoteAddr, 2> remoteResults;
    ccu::Event pushEvent;
    for (uint32_t lane = 0; lane < 2; ++lane) {
        uint32_t slice = RANK12_PAIR_ROOTS[pairRank][lane];
        uint32_t child = RANK12_PAIR_ROOTS[pairRank][1 - lane];
        ChannelHandle channel = channels[child];

        localResults[lane].addr = addresses.output.addr;
        localResults[lane].token = addresses.output.token;
        AddRank12SliceOffset(localResults[lane].addr,
            addresses.sliceBytes, slice);
        remoteResults[lane].addr =
            ccu::GetResByChannel<ccu::Variable>(
                channel, RANK12_OUTPUT_ADDR_VARIABLE);
        remoteResults[lane].token =
            ccu::GetResByChannel<ccu::Variable>(
                channel, RANK12_OUTPUT_TOKEN_VARIABLE);
        AddRank12SliceOffset(remoteResults[lane].addr,
            addresses.sliceBytes, slice);

        F002_CCU_CHK_RET(ccu::Write(channel, remoteResults[lane],
            localResults[lane], addresses.sliceBytes, pushEvent,
            static_cast<uint16_t>(1U << lane)));
    }
    F002_CCU_CHK_RET(
        ccu::EventWait(pushEvent, RANK12_TWO_WAY_EVENT_MASK));

    for (uint32_t lane = 0; lane < 2; ++lane) {
        uint32_t slice = RANK12_PAIR_ROOTS[pairRank][lane];
        uint32_t child = RANK12_PAIR_ROOTS[pairRank][1 - lane];
        F002_CCU_CHK_RET(ccu::NotifyRecord(channels[child],
            DATA_COMPLETE_NOTIFY, F085Rank12PushDoneMask(slice)));
    }

    // Nonincident waits are deliberately after PushDone and therefore off
    // the child critical path, while still closing cross-round lifetimes.
    for (uint32_t root = 0; root < RANK12_SMALL_SERVER_SIZE; ++root) {
        if (Rank12PairContains(pairRank, root)) {
            continue;
        }
        F002_CCU_CHK_RET(ccu::NotifyWait(channels[root],
            DATA_COMPLETE_NOTIFY, F085Rank12RootDoneMask(root)));
    }
    return CCU_SUCCESS;
}

CcuResult F085WaitRank12PushDone(Rank12ClosTreeContext &ctx,
    const std::array<ChannelHandle, RANK12_LARGE_SERVER_SIZE> &channels)
{
    if (ctx.arg->myRank < RANK12_LARGE_SERVER_SIZE) {
        return CCU_SUCCESS;
    }

    uint32_t root = ctx.arg->myRank - RANK12_LARGE_SERVER_SIZE;
    for (uint32_t pairRank = 0;
        pairRank < RANK12_PAIR_RELAY_NUM; ++pairRank) {
        if (!Rank12PairContains(pairRank, root)) {
            continue;
        }
        uint32_t slice = RANK12_PAIR_ROOTS[pairRank][0] == root ?
            RANK12_PAIR_ROOTS[pairRank][1] :
            RANK12_PAIR_ROOTS[pairRank][0];
        F002_CCU_CHK_RET(ccu::NotifyWait(channels[pairRank],
            DATA_COMPLETE_NOTIFY, F085Rank12PushDoneMask(slice)));
    }
    return CCU_SUCCESS;
}

CcuResult RunF085Rank12ClosTree(Rank12ClosTreeContext &ctx)
{
    std::array<ChannelHandle, RANK12_LARGE_SERVER_SIZE> channels{};
    BuildRank12ClosChannels(ctx, channels);
    Rank12FourRootAddresses addresses;
    BuildRank12FourRootAddresses(ctx, addresses);

    F002_CCU_CHK_RET(F085PublishRank12FourRootMetadata(ctx, channels));
    F002_CCU_CHK_RET(F085FormRank12PairPartials(ctx, channels, addresses));
    F002_CCU_CHK_RET(F085FormRank12RootResult(ctx, channels, addresses));
    F002_CCU_CHK_RET(F085PushRank12ChildResults(ctx, channels, addresses));
    F002_CCU_CHK_RET(F085WaitRank12PushDone(ctx, channels));
    return CCU_SUCCESS;
}

#undef F002_CCU_CHK_RET

} // namespace ops_hccl
