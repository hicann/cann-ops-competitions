/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include <hcomm/hcomm_primitives.h>

#include "ccu_kernel.h"
#include "custom.h"

#ifndef CCU_CHK_RET
#define CCU_CHK_RET(call) \
    do { \
        const CcuResult ccuCallRet = static_cast<CcuResult>(call); \
        if (ccuCallRet != CCU_SUCCESS) { \
            return ccuCallRet; \
        } \
    } while (0)
#endif

namespace ops_hccl {
namespace ccu = ::AscendC::ccu;
namespace {

constexpr uint32_t OUTPUT_ADDR_XN_ID = 0U;
constexpr uint32_t OUTPUT_TOKEN_XN_ID = 1U;
constexpr uint32_t CHANNEL_NOTIFY_INDEX = 0U;
// 每条 Channel 的 XN 0/1 保存本轮对端地址和 Token：Push 路径交换 Output，
// 一次性 Pull 路径交换 Source。Notify 位用于地址 READY、Token READY 和 DMA DONE。
constexpr uint16_t OUTPUT_ADDR_READY_MASK = 1U << 0U;
constexpr uint16_t OUTPUT_TOKEN_READY_MASK = 1U << 1U;
constexpr uint16_t PARAM_READY_MASK = OUTPUT_ADDR_READY_MASK | OUTPUT_TOKEN_READY_MASK;
constexpr uint16_t TRANSFER_DONE_MASK = 1U << 2U;
constexpr uint16_t HYBRID_PHASE_A_DONE_MASK = 1U << 3U;
constexpr uint16_t HYBRID_CKE_READY_MASK = 1U << 0U;
constexpr uint32_t P2X8_SERVER_RANKS = 8U;
constexpr uint32_t HYBRID_INVALID_SLOT = P2X8_SERVER_RANKS;
constexpr uint32_t HYBRID_INVALID_CHANNEL = MAX_RANK_SIZE;
constexpr uint8_t HYBRID_UNUSED = 0xffU;
constexpr const char *HYBRID_SEED_READY_TAG = "p2x8_hybrid_seed_ready";
constexpr uint32_t P8P4_LARGE_SERVER_RANKS = 8U;
constexpr uint32_t P8P4_SMALL_SERVER_RANKS = 4U;
constexpr uint32_t P8P4_INVALID_SLOT = P8P4_LARGE_SERVER_RANKS;
static_assert(MAX_RANK_SIZE <= 16U, "one Event mask must cover every channel in a Die group");
static_assert(ALLGATHER_P2X8_HYBRID_STRIPE_COUNT == 11U, "hybrid route table has eleven stripes");
static_assert(ALLGATHER_P8P4_HYBRID_STRIPE_COUNT == 9U, "8+4 hybrid route table has nine stripes");

struct HybridPhaseASeedRun {
    uint8_t beginStripe;
    uint8_t endStripe;
};

constexpr uint8_t HYBRID_PHASE_A_SEEDS[ALLGATHER_P2X8_HYBRID_STRIPE_COUNT][4] = {
    {0U, 1U, 2U, 3U},
    {1U, 2U, 3U, 4U},
    {2U, 3U, 4U, 5U},
    {3U, 4U, 5U, 6U},
    {4U, 5U, 6U, 7U},
    {0U, 5U, 6U, 7U},
    {0U, 1U, 6U, 7U},
    {0U, 1U, 2U, 7U},
    {0U, 1U, 2U, 3U},
    {1U, 2U, 3U, 4U},
    {2U, 3U, 4U, 5U},
};

// 对同一 Clos Peer 的相邻 Seed stripe 合并成一个 Write。每个 Event 槽对每条
// Channel 至多保留一个未完成 WQE，因此两个不相邻区间使用不同槽。
constexpr uint8_t HYBRID_PHASE_A_RUN_COUNT[P2X8_SERVER_RANKS] = {
    2U, 2U, 2U, 2U, 2U, 2U, 1U, 1U,
};
constexpr HybridPhaseASeedRun HYBRID_PHASE_A_RUNS[P2X8_SERVER_RANKS][2] = {
    {{0U, 1U}, {5U, 9U}},
    {{0U, 2U}, {6U, 10U}},
    {{0U, 3U}, {7U, 11U}},
    {{0U, 4U}, {8U, 11U}},
    {{1U, 5U}, {9U, 11U}},
    {{2U, 6U}, {10U, 11U}},
    {{3U, 7U}, {HYBRID_UNUSED, HYBRID_UNUSED}},
    {{4U, 8U}, {HYBRID_UNUSED, HYBRID_UNUSED}},
};

constexpr uint8_t HYBRID_PHASE_B_EXTRA_COUNT[ALLGATHER_P2X8_HYBRID_STRIPE_COUNT] = {
    1U, 1U, 1U, 1U, 1U, 1U, 2U, 2U, 2U, 2U, 2U,
};
constexpr uint8_t HYBRID_PHASE_B_EXTRAS[ALLGATHER_P2X8_HYBRID_STRIPE_COUNT][2] = {
    {6U, HYBRID_UNUSED},
    {7U, HYBRID_UNUSED},
    {7U, HYBRID_UNUSED},
    {1U, HYBRID_UNUSED},
    {0U, HYBRID_UNUSED},
    {4U, HYBRID_UNUSED},
    {2U, 5U},
    {3U, 6U},
    {6U, 7U},
    {0U, 5U},
    {6U, 7U},
};

constexpr uint8_t HYBRID_PHASE_B_RELAY_COUNT[ALLGATHER_P2X8_HYBRID_STRIPE_COUNT] = {
    3U, 3U, 3U, 3U, 3U, 3U, 2U, 2U, 2U, 2U, 2U,
};
constexpr uint8_t HYBRID_PHASE_B_RELAYS[ALLGATHER_P2X8_HYBRID_STRIPE_COUNT][3][2] = {
    {{3U, 4U}, {3U, 5U}, {0U, 7U}},
    {{1U, 0U}, {4U, 5U}, {3U, 6U}},
    {{3U, 0U}, {5U, 1U}, {5U, 6U}},
    {{4U, 0U}, {4U, 2U}, {6U, 7U}},
    {{4U, 1U}, {4U, 2U}, {4U, 3U}},
    {{7U, 1U}, {6U, 2U}, {0U, 3U}},
    {{7U, 3U}, {6U, 4U}, {HYBRID_UNUSED, HYBRID_UNUSED}},
    {{1U, 4U}, {7U, 5U}, {HYBRID_UNUSED, HYBRID_UNUSED}},
    {{2U, 4U}, {0U, 5U}, {HYBRID_UNUSED, HYBRID_UNUSED}},
    {{4U, 6U}, {4U, 7U}, {HYBRID_UNUSED, HYBRID_UNUSED}},
    {{3U, 0U}, {2U, 1U}, {HYBRID_UNUSED, HYBRID_UNUSED}},
};

// 4 卡侧每个源在前 8 个 stripe 中先发 4 个 Seed，Phase B 再补 1 个直达；
// 第 9 个 stripe 只发 4 个 Seed。连续 Seed 区间合并后，每个源仅有 11~12 个 Phase-A WQE。
constexpr uint8_t P8P4_PHASE_B_EXTRAS[P8P4_SMALL_SERVER_RANKS]
    [ALLGATHER_P8P4_HYBRID_STRIPE_COUNT - 1U] = {
    {4U, 4U, 1U, 2U, 3U, 3U, 3U, 3U},
    {5U, 5U, 5U, 6U, 3U, 3U, 3U, 6U},
    {3U, 0U, 0U, 0U, 0U, 0U, 1U, 2U},
    {3U, 0U, 0U, 0U, 0U, 0U, 1U, 3U},
};

constexpr uint8_t P8P4_PHASE_A_RUN_COUNT[P8P4_SMALL_SERVER_RANKS]
    [P8P4_LARGE_SERVER_RANKS] = {
    {1U, 1U, 1U, 1U, 1U, 2U, 2U, 2U},
    {2U, 2U, 1U, 2U, 2U, 1U, 1U, 1U},
    {0U, 2U, 1U, 1U, 2U, 2U, 2U, 2U},
    {1U, 1U, 1U, 1U, 2U, 2U, 2U, 1U},
};

constexpr HybridPhaseASeedRun P8P4_PHASE_A_RUNS[P8P4_SMALL_SERVER_RANKS]
    [P8P4_LARGE_SERVER_RANKS][2] = {
    {
        {{1U, 6U}, {HYBRID_UNUSED, HYBRID_UNUSED}},
        {{3U, 7U}, {HYBRID_UNUSED, HYBRID_UNUSED}},
        {{4U, 8U}, {HYBRID_UNUSED, HYBRID_UNUSED}},
        {{0U, 1U}, {HYBRID_UNUSED, HYBRID_UNUSED}},
        {{5U, 9U}, {HYBRID_UNUSED, HYBRID_UNUSED}},
        {{0U, 3U}, {6U, 9U}},
        {{0U, 4U}, {7U, 9U}},
        {{0U, 5U}, {8U, 9U}},
    },
    {
        {{1U, 6U}, {8U, 9U}},
        {{2U, 7U}, {8U, 9U}},
        {{3U, 9U}, {HYBRID_UNUSED, HYBRID_UNUSED}},
        {{0U, 1U}, {7U, 9U}},
        {{0U, 2U}, {5U, 8U}},
        {{6U, 8U}, {HYBRID_UNUSED, HYBRID_UNUSED}},
        {{0U, 3U}, {HYBRID_UNUSED, HYBRID_UNUSED}},
        {{0U, 5U}, {HYBRID_UNUSED, HYBRID_UNUSED}},
    },
    {
        {{HYBRID_UNUSED, HYBRID_UNUSED}, {HYBRID_UNUSED, HYBRID_UNUSED}},
        {{2U, 6U}, {8U, 9U}},
        {{3U, 7U}, {HYBRID_UNUSED, HYBRID_UNUSED}},
        {{4U, 9U}, {HYBRID_UNUSED, HYBRID_UNUSED}},
        {{0U, 2U}, {5U, 8U}},
        {{0U, 3U}, {6U, 9U}},
        {{0U, 4U}, {7U, 8U}},
        {{0U, 5U}, {8U, 9U}},
    },
    {
        {{8U, 9U}, {HYBRID_UNUSED, HYBRID_UNUSED}},
        {{2U, 6U}, {HYBRID_UNUSED, HYBRID_UNUSED}},
        {{3U, 9U}, {HYBRID_UNUSED, HYBRID_UNUSED}},
        {{4U, 7U}, {HYBRID_UNUSED, HYBRID_UNUSED}},
        {{0U, 2U}, {5U, 9U}},
        {{0U, 3U}, {6U, 8U}},
        {{0U, 4U}, {7U, 9U}},
        {{0U, 5U}, {HYBRID_UNUSED, HYBRID_UNUSED}},
    },
};

constexpr uint8_t P8P4_SPECIAL_MISSING[P8P4_SMALL_SERVER_RANKS][4] = {
    {0U, 1U, 2U, 3U},
    {4U, 5U, 6U, 7U},
    {0U, 2U, 4U, 6U},
    {1U, 3U, 5U, 7U},
};

// 每项按该 stripe 缺失目的槽位的升序给出 relay 源槽位。四个远端源合计
// 112 个 relay stripe，56 条有向 Mesh 链路恰好各承担 2 个。
constexpr uint8_t P8P4_RELAY_TAILS[P8P4_SMALL_SERVER_RANKS]
    [ALLGATHER_P8P4_HYBRID_STRIPE_COUNT][4] = {
    {
        {3U, 3U, 3U, HYBRID_UNUSED}, {7U, 0U, 0U, HYBRID_UNUSED},
        {0U, 0U, 0U, HYBRID_UNUSED}, {1U, 0U, 0U, HYBRID_UNUSED},
        {1U, 0U, 0U, HYBRID_UNUSED}, {1U, 0U, 0U, HYBRID_UNUSED},
        {1U, 1U, 1U, HYBRID_UNUSED}, {2U, 2U, 2U, HYBRID_UNUSED},
        {4U, 4U, 4U, 4U},
    },
    {
        {3U, 3U, 3U, HYBRID_UNUSED}, {0U, 4U, 4U, HYBRID_UNUSED},
        {1U, 1U, 1U, HYBRID_UNUSED}, {2U, 2U, 1U, HYBRID_UNUSED},
        {2U, 2U, 1U, HYBRID_UNUSED}, {2U, 2U, 0U, HYBRID_UNUSED},
        {1U, 2U, 1U, HYBRID_UNUSED}, {2U, 2U, 2U, HYBRID_UNUSED},
        {3U, 3U, 3U, 3U},
    },
    {
        {4U, 4U, 5U, HYBRID_UNUSED}, {5U, 5U, 5U, HYBRID_UNUSED},
        {1U, 5U, 5U, HYBRID_UNUSED}, {2U, 6U, 6U, HYBRID_UNUSED},
        {3U, 3U, 3U, HYBRID_UNUSED}, {4U, 4U, 3U, HYBRID_UNUSED},
        {5U, 5U, 4U, HYBRID_UNUSED}, {6U, 5U, 5U, HYBRID_UNUSED},
        {7U, 7U, 5U, 7U},
    },
    {
        {7U, 6U, 6U, HYBRID_UNUSED}, {7U, 6U, 6U, HYBRID_UNUSED},
        {7U, 7U, 6U, HYBRID_UNUSED}, {7U, 7U, 7U, HYBRID_UNUSED},
        {7U, 7U, 7U, HYBRID_UNUSED}, {4U, 4U, 4U, HYBRID_UNUSED},
        {5U, 5U, 5U, HYBRID_UNUSED}, {6U, 6U, 6U, HYBRID_UNUSED},
        {0U, 6U, 6U, 6U},
    },
};

struct TransferContext {
    const CcuKernelArgAllGather *arg{nullptr};
    ccu::Variable sourceAddr;
    ccu::Variable outputAddr;
    ccu::Variable sourceToken;
    ccu::Variable outputToken;
    ccu::Variable localOffset;
    std::vector<ccu::Variable> remoteOutputAddr;
    std::vector<ccu::Variable> remoteOutputToken;
    std::vector<ccu::Event> peerEvents;
    std::vector<ccu::Event> localEvents;
};

struct HybridRankMap {
    uint32_t selfSlot{HYBRID_INVALID_SLOT};
    std::array<uint32_t, P2X8_SERVER_RANKS> localRanks{};
    std::array<uint32_t, P2X8_SERVER_RANKS> remoteRanks{};
    std::array<uint32_t, P2X8_SERVER_RANKS> channelByPeerSlot{};
};

struct P8P4HybridRankMap {
    bool isLargeSide{false};
    uint32_t selfSlot{P8P4_INVALID_SLOT};
    std::array<uint32_t, P8P4_LARGE_SERVER_RANKS> largeRanks{};
    std::array<uint32_t, P8P4_SMALL_SERVER_RANKS> smallRanks{};
    std::array<uint32_t, P8P4_LARGE_SERVER_RANKS> channelByLargeSlot{};
    std::array<uint32_t, P8P4_SMALL_SERVER_RANKS> channelBySmallSlot{};
};

struct HybridStripeGeometry {
    uint64_t offset{0U};
    uint64_t bytes{0U};
};

struct DistributedRootGeometry {
    uint64_t relayBytes{0U};
    uint64_t cappedLastRelayBytes{0U};
};

uint32_t GetP2X8DistributedRootPeerSlot(const HybridRankMap &rankMap,
    uint32_t peerOffset, bool matchedSerial)
{
    if (rankMap.selfSlot >= P2X8_SERVER_RANKS
        || peerOffset >= P2X8_SERVER_RANKS) {
        return HYBRID_INVALID_SLOT;
    }
    // localRanks/remoteRanks are sorted by global Rank. Reversing offsets on
    // the higher-Rank Server makes logical offset k select A[i] <-> B[i+k].
    const bool reverseOffset = matchedSerial
        && rankMap.localRanks[0] > rankMap.remoteRanks[0];
    const uint32_t directedOffset = reverseOffset
        ? P2X8_SERVER_RANKS - peerOffset
        : peerOffset;
    return (rankMap.selfSlot + directedOffset) & 7U;
}

struct P8P4BidirectionalGeometry {
    uint64_t halfBytes{0U};
    uint64_t relayBytes{0U};
    uint64_t tailBytes{0U};
};

CcuResult RunP8P4LargeSingleDma(TransferContext &ctx, uint64_t sliceBytes);

bool IsP2X8LargeProfile(AllGatherProfile profile)
{
    return profile == AllGatherProfile::P2X8_512M
           || profile == AllGatherProfile::P2X8_400M4B;
}

bool IsP4X1LargeProfile(AllGatherProfile profile)
{
    return profile == AllGatherProfile::P4X1_512M
           || profile == AllGatherProfile::P4X1_400M4B;
}

bool IsP8P4LargeProfile(AllGatherProfile profile)
{
    return profile == AllGatherProfile::P8P4_512M
           || profile == AllGatherProfile::P8P4_400M4B;
}

CcuResult ValidateTransferArg(const CcuKernelArgAllGather *arg)
{
    if (arg == nullptr || arg->rankId >= arg->rankSize
        || (arg->rankSize != 4U && arg->rankSize != 12U && arg->rankSize != 16U)
        || arg->channelCount == 0U || arg->channelCount > arg->rankSize - 1U
        || arg->channelCount > MAX_RANK_SIZE - 1U || arg->copyLocalSlice > 1U
        || (arg->sourceMode != AllGatherSourceMode::DIRECT_INPUT
            && arg->sourceMode != AllGatherSourceMode::CANONICAL_OUTPUT
            && arg->sourceMode != AllGatherSourceMode::STAGED_OUTPUT)
        || (arg->dataTypeSize != 1U && arg->dataTypeSize != 2U && arg->dataTypeSize != 4U
            && arg->dataTypeSize != 8U && arg->dataTypeSize != 16U)) {
        return CcuResult::CCU_E_PARA;
    }
    return CCU_SUCCESS;
}

CcuResult ValidateProfileArg(const CcuKernelArgAllGather *arg, AllGatherProfile expectedProfile,
    AllGatherDataPath expectedDataPath)
{
    CCU_CHK_RET(ValidateTransferArg(arg));
    AllGatherProfileSpec spec;
    if (arg->profile != expectedProfile || arg->dataPath != expectedDataPath
        || expectedProfile == AllGatherProfile::GENERIC
        || !GetAllGatherProfileSpec(expectedProfile, spec)
        || arg->rankSize != spec.expectedRankSize || arg->fixedSliceBytes == 0U
        || arg->fixedSliceBytes % arg->dataTypeSize != 0U) {
        return CcuResult::CCU_E_PARA;
    }
    const uint64_t sendCount = arg->fixedSliceBytes / arg->dataTypeSize;
    if (!MatchesAllGatherOutputProfile(
            sendCount, spec.targetOutputBytes, arg->rankSize, arg->dataTypeSize)) {
        return CcuResult::CCU_E_PARA;
    }
    const bool mixedLayerPolicy
        = spec.layerPolicy == AllGatherLayerPolicy::LOCAL_MESH_REMOTE_CLOS;
    if ((spec.layerPolicy == AllGatherLayerPolicy::CLOS_ONLY
            && arg->layerRole != AllGatherLayerRole::CLOS)
        || (mixedLayerPolicy
            && arg->layerRole != AllGatherLayerRole::LOCAL_MESH
            && arg->layerRole != AllGatherLayerRole::CLOS)
        || (!mixedLayerPolicy && spec.layerPolicy != AllGatherLayerPolicy::CLOS_ONLY)) {
        return CcuResult::CCU_E_PARA;
    }
    const uint64_t expectedCellBytes
        = spec.algorithm == AllGatherAlgorithm::FIXED_DIRECT_SMALL
              ? arg->fixedSliceBytes
              : spec.cellBytes;
    if (arg->cellBytes != expectedCellBytes || arg->cellBytes == 0U
        || arg->cellBytes >= MAX_DATA_SIZE) {
        return CcuResult::CCU_E_PARA;
    }
    return CCU_SUCCESS;
}

CcuResult InitTransferResources(TransferContext &ctx)
{
    CCU_CHK_RET(ValidateTransferArg(ctx.arg));
    ctx.remoteOutputAddr.reserve(ctx.arg->channelCount);
    ctx.remoteOutputToken.reserve(ctx.arg->channelCount);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        ctx.remoteOutputAddr.push_back(
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], OUTPUT_ADDR_XN_ID));
        ctx.remoteOutputToken.push_back(
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], OUTPUT_TOKEN_XN_ID));
    }
    AllGatherProfileSpec spec;
    if (!GetAllGatherProfileSpec(ctx.arg->profile, spec)) {
        return CcuResult::CCU_E_PARA;
    }
    // Profile 显式决定 Event 槽数；本地 Copy 使用独立 Event，避免与 Channel 位掩码混用。
    const uint32_t eventCount = spec.eventSlots;
    if (eventCount == 0U || eventCount > 2U) {
        return CcuResult::CCU_E_PARA;
    }
    ctx.peerEvents.resize(eventCount);
    if (ctx.arg->copyLocalSlice != 0U) {
        ctx.localEvents.resize(eventCount);
    }
    return CCU_SUCCESS;
}

CcuResult InitP2X8SingleDmaResources(TransferContext &ctx)
{
    CCU_CHK_RET(ValidateTransferArg(ctx.arg));
    if (!IsP2X8LargeProfile(ctx.arg->profile)
        || ctx.arg->fixedSliceBytes == 0U || ctx.arg->fixedSliceBytes >= MAX_DATA_SIZE) {
        return CcuResult::CCU_E_PARA;
    }
    ctx.remoteOutputAddr.reserve(ctx.arg->channelCount);
    ctx.remoteOutputToken.reserve(ctx.arg->channelCount);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        ctx.remoteOutputAddr.push_back(
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], OUTPUT_ADDR_XN_ID));
        ctx.remoteOutputToken.push_back(
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], OUTPUT_TOKEN_XN_ID));
    }
    // 每个 Die Kernel 对自己持有的每个 Peer 只提交一次完整 Slice DMA，故只需一个 Event 槽。
    ctx.peerEvents.resize(1U);
    if (ctx.arg->copyLocalSlice != 0U) {
        ctx.localEvents.resize(1U);
    }
    return CCU_SUCCESS;
}

CcuResult InitP4X1SingleDmaResources(TransferContext &ctx)
{
    CCU_CHK_RET(ValidateTransferArg(ctx.arg));
    if (!IsP4X1LargeProfile(ctx.arg->profile)
        || ctx.arg->fixedSliceBytes == 0U || ctx.arg->fixedSliceBytes >= MAX_DATA_SIZE) {
        return CcuResult::CCU_E_PARA;
    }
    ctx.remoteOutputAddr.reserve(ctx.arg->channelCount);
    ctx.remoteOutputToken.reserve(ctx.arg->channelCount);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        ctx.remoteOutputAddr.push_back(
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], OUTPUT_ADDR_XN_ID));
        ctx.remoteOutputToken.push_back(
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], OUTPUT_TOKEN_XN_ID));
    }
    // 每个实际 Die group 注册一个 Clos Kernel；组内每个 Peer 只需一个完整 Slice Event。
    ctx.peerEvents.resize(1U);
    if (ctx.arg->copyLocalSlice != 0U) {
        ctx.localEvents.resize(1U);
    }
    return CCU_SUCCESS;
}

CcuResult InitP8P4SingleDmaResources(TransferContext &ctx)
{
    CCU_CHK_RET(ValidateTransferArg(ctx.arg));
    if (!IsP8P4LargeProfile(ctx.arg->profile)
        || ctx.arg->fixedSliceBytes == 0U || ctx.arg->fixedSliceBytes >= MAX_DATA_SIZE) {
        return CcuResult::CCU_E_PARA;
    }
    ctx.remoteOutputAddr.reserve(ctx.arg->channelCount);
    ctx.remoteOutputToken.reserve(ctx.arg->channelCount);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        ctx.remoteOutputAddr.push_back(
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], OUTPUT_ADDR_XN_ID));
        ctx.remoteOutputToken.push_back(
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], OUTPUT_TOKEN_XN_ID));
    }
    // 固定分层后每个 {通信层, 实际 Die} Kernel 内一 Peer 一次完整 Slice DMA；
    // 8+4 两侧最大 CLOS 组为 8 个 Peer，16-bit Event 掩码可完整覆盖。
    ctx.peerEvents.resize(1U);
    if (ctx.arg->copyLocalSlice != 0U) {
        ctx.localEvents.resize(1U);
    }
    return CCU_SUCCESS;
}

CcuResult PublishOutputIdentityChannel(TransferContext &ctx, uint32_t channelIdx)
{
    if (channelIdx >= ctx.arg->channelCount) {
        return CcuResult::CCU_E_PARA;
    }
    CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx], ctx.outputAddr,
        OUTPUT_ADDR_XN_ID, CHANNEL_NOTIFY_INDEX, OUTPUT_ADDR_READY_MASK));
    CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx], ctx.outputToken,
        OUTPUT_TOKEN_XN_ID, CHANNEL_NOTIFY_INDEX, OUTPUT_TOKEN_READY_MASK));
    return CCU_SUCCESS;
}

CcuResult PublishOutputIdentity(TransferContext &ctx)
{
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        CCU_CHK_RET(PublishOutputIdentityChannel(ctx, channelIdx));
    }
    return CCU_SUCCESS;
}

CcuResult PublishSourceIdentity(TransferContext &ctx)
{
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx], ctx.sourceAddr,
            OUTPUT_ADDR_XN_ID, CHANNEL_NOTIFY_INDEX, OUTPUT_ADDR_READY_MASK));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx], ctx.sourceToken,
            OUTPUT_TOKEN_XN_ID, CHANNEL_NOTIFY_INDEX, OUTPUT_TOKEN_READY_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult WaitOutputIdentityChannel(TransferContext &ctx, uint32_t channelIdx)
{
    if (channelIdx >= ctx.arg->channelCount) {
        return CcuResult::CCU_E_PARA;
    }
    CCU_CHK_RET(ccu::NotifyWait(
        ctx.arg->channels[channelIdx], CHANNEL_NOTIFY_INDEX, PARAM_READY_MASK));
    return CCU_SUCCESS;
}

CcuResult WaitOutputIdentity(TransferContext &ctx)
{
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        CCU_CHK_RET(WaitOutputIdentityChannel(ctx, channelIdx));
    }
    return CCU_SUCCESS;
}

CcuResult ExchangeOutputIdentity(TransferContext &ctx)
{
    // 只交换 Output 身份，Source 始终由源 Rank 本地持有并主动 Push。所有 Channel 的
    // 地址和 Token 均 READY 后才进入数据面，同时也为 staged 路径形成跨 Rank 屏障。
    CCU_CHK_RET(PublishOutputIdentity(ctx));
    CCU_CHK_RET(WaitOutputIdentity(ctx));
    return CCU_SUCCESS;
}

CcuResult ExchangeP8P4SharedClosOutputIdentity(
    TransferContext &ctx, const P8P4HybridRankMap &rankMap)
{
    if (ctx.arg->layerRole != AllGatherLayerRole::CLOS) {
        return CcuResult::CCU_E_PARA;
    }
    if (rankMap.isLargeSide) {
        // U_floor(A/2) is the only remote rank that writes a half-root into this A.
        const uint32_t publishChannel
            = rankMap.channelBySmallSlot[rankMap.selfSlot / 2U];
        CCU_CHK_RET(PublishOutputIdentityChannel(ctx, publishChannel));
        // Every A writes its own Slice into all four U ranks across Phase A/B.
        CCU_CHK_RET(WaitOutputIdentity(ctx));
        return CCU_SUCCESS;
    }

    // Every A writes into this U, so U publishes to all eight A ranks. U itself
    // writes only to its two half-roots and needs identities from those roots.
    CCU_CHK_RET(PublishOutputIdentity(ctx));
    for (uint32_t rootOffset = 0U; rootOffset < 2U; ++rootOffset) {
        const uint32_t rootSlot = 2U * rankMap.selfSlot + rootOffset;
        CCU_CHK_RET(WaitOutputIdentityChannel(
            ctx, rankMap.channelByLargeSlot[rootSlot]));
    }
    return CCU_SUCCESS;
}

CcuResult PairwiseCompletion(TransferContext &ctx, uint16_t doneMask = TRANSFER_DONE_MASK)
{
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[channelIdx], CHANNEL_NOTIFY_INDEX, doneMask));
    }
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIdx], CHANNEL_NOTIFY_INDEX, doneMask));
    }
    return CCU_SUCCESS;
}

CcuResult PairwiseCompletionChannel(
    TransferContext &ctx, uint32_t channelIdx, uint16_t doneMask)
{
    if (channelIdx >= ctx.arg->channelCount) {
        return CcuResult::CCU_E_PARA;
    }
    CCU_CHK_RET(ccu::NotifyRecord(
        ctx.arg->channels[channelIdx], CHANNEL_NOTIFY_INDEX, doneMask));
    CCU_CHK_RET(ccu::NotifyWait(
        ctx.arg->channels[channelIdx], CHANNEL_NOTIFY_INDEX, doneMask));
    return CCU_SUCCESS;
}

CcuResult PairwiseCompletionP2X8ClosOffsets(TransferContext &ctx,
    const HybridRankMap &rankMap, uint32_t recordBeginOffset, bool matchedSerial)
{
    if (ctx.arg->layerRole != AllGatherLayerRole::CLOS
        || ctx.arg->channelCount != P2X8_SERVER_RANKS
        || rankMap.selfSlot >= P2X8_SERVER_RANKS
        || recordBeginOffset == 0U || recordBeginOffset >= P2X8_SERVER_RANKS) {
        return CcuResult::CCU_E_PARA;
    }
    for (uint32_t peerOffset = recordBeginOffset;
         peerOffset < P2X8_SERVER_RANKS; ++peerOffset) {
        const uint32_t peerSlot = GetP2X8DistributedRootPeerSlot(
            rankMap, peerOffset, matchedSerial);
        const uint32_t channelIdx = rankMap.channelByPeerSlot[peerSlot];
        if (channelIdx >= ctx.arg->channelCount) {
            return CcuResult::CCU_E_PARA;
        }
        CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[channelIdx],
            CHANNEL_NOTIFY_INDEX, TRANSFER_DONE_MASK));
    }
    const uint32_t waitBeginOffset = matchedSerial ? recordBeginOffset : 1U;
    const uint32_t waitEndOffset = matchedSerial
        ? P2X8_SERVER_RANKS - 1U
        : P2X8_SERVER_RANKS - recordBeginOffset;
    for (uint32_t peerOffset = waitBeginOffset;
         peerOffset <= waitEndOffset; ++peerOffset) {
        const uint32_t peerSlot = GetP2X8DistributedRootPeerSlot(
            rankMap, peerOffset, matchedSerial);
        const uint32_t channelIdx = rankMap.channelByPeerSlot[peerSlot];
        if (channelIdx >= ctx.arg->channelCount) {
            return CcuResult::CCU_E_PARA;
        }
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIdx],
            CHANNEL_NOTIFY_INDEX, TRANSFER_DONE_MASK));
    }
    return CCU_SUCCESS;
}

CcuResult CompleteP8P4PhaseASeeds(
    TransferContext &ctx, const P8P4HybridRankMap &rankMap)
{
    if (ctx.arg->layerRole != AllGatherLayerRole::CLOS) {
        return CcuResult::CCU_E_PARA;
    }
    if (rankMap.isLargeSide) {
        // A0..A3 provide the relay prefix consumed by U0..U3.  Every A rank
        // consumes exactly one U half-slice as its local half-root.
        if (rankMap.selfSlot < P8P4_SMALL_SERVER_RANKS) {
            const uint32_t recordChannel
                = rankMap.channelBySmallSlot[rankMap.selfSlot];
            if (recordChannel >= ctx.arg->channelCount) {
                return CcuResult::CCU_E_PARA;
            }
            CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[recordChannel],
                CHANNEL_NOTIFY_INDEX, HYBRID_PHASE_A_DONE_MASK));
        }
        const uint32_t waitChannel
            = rankMap.channelBySmallSlot[rankMap.selfSlot / 2U];
        if (waitChannel >= ctx.arg->channelCount) {
            return CcuResult::CCU_E_PARA;
        }
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[waitChannel],
            CHANNEL_NOTIFY_INDEX, HYBRID_PHASE_A_DONE_MASK));
        return CCU_SUCCESS;
    }

    // Uj publishes its two half-slices only to their A(2j)/A(2j+1) roots,
    // then waits only for Aj, whose prefix is read by the small-side Mesh.
    for (uint32_t rootOffset = 0U; rootOffset < 2U; ++rootOffset) {
        const uint32_t rootSlot = 2U * rankMap.selfSlot + rootOffset;
        const uint32_t channelIdx = rankMap.channelByLargeSlot[rootSlot];
        if (channelIdx >= ctx.arg->channelCount) {
            return CcuResult::CCU_E_PARA;
        }
        CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[channelIdx],
            CHANNEL_NOTIFY_INDEX, HYBRID_PHASE_A_DONE_MASK));
    }
    const uint32_t waitChannel
        = rankMap.channelByLargeSlot[rankMap.selfSlot];
    if (waitChannel >= ctx.arg->channelCount) {
        return CcuResult::CCU_E_PARA;
    }
    CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[waitChannel],
        CHANNEL_NOTIFY_INDEX, HYBRID_PHASE_A_DONE_MASK));
    return CCU_SUCCESS;
}

CcuResult CompleteP8P4ClosReceives(
    TransferContext &ctx, const P8P4HybridRankMap &rankMap)
{
    if (ctx.arg->layerRole != AllGatherLayerRole::CLOS) {
        return CcuResult::CCU_E_PARA;
    }
    if (rankMap.isLargeSide) {
        // Only the four-card side consumes A data at the end of the Clos path.
        // Large-side U data is completed transitively by the half-root Mesh.
        for (uint32_t destinationSlot = 0U;
             destinationSlot < P8P4_SMALL_SERVER_RANKS; ++destinationSlot) {
            const uint32_t channelIdx
                = rankMap.channelBySmallSlot[destinationSlot];
            if (channelIdx >= ctx.arg->channelCount) {
                return CcuResult::CCU_E_PARA;
            }
            CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[channelIdx],
                CHANNEL_NOTIFY_INDEX, TRANSFER_DONE_MASK));
        }
        return CCU_SUCCESS;
    }

    for (uint32_t sourceSlot = 0U;
         sourceSlot < P8P4_LARGE_SERVER_RANKS; ++sourceSlot) {
        const uint32_t channelIdx = rankMap.channelByLargeSlot[sourceSlot];
        if (channelIdx >= ctx.arg->channelCount) {
            return CcuResult::CCU_E_PARA;
        }
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIdx],
            CHANNEL_NOTIFY_INDEX, TRANSFER_DONE_MASK));
    }
    return CCU_SUCCESS;
}

uint16_t ChannelEventMask(uint32_t channelCount)
{
    return static_cast<uint16_t>((1U << channelCount) - 1U);
}

CcuResult RunOneChunk(TransferContext &ctx, ccu::Variable chunkOffset, ccu::Variable chunkBytes)
{
    // 源 Rank 将自己的 Slice 写到每个 Peer 的 output + rank * sliceBytes + chunkOffset。
    // 仅 direct out-of-place 由主 Kernel 复制本地 Offset；全部已提交 DMA
    // 的 Event 完成后才能发布 DONE。
    ccu::LocalAddr source;
    source.addr = ctx.sourceAddr;
    source.addr += chunkOffset;
    source.token = ctx.sourceToken;

    std::vector<ccu::RemoteAddr> remoteDestination(ctx.arg->channelCount);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        remoteDestination[channelIdx].addr = ctx.remoteOutputAddr[channelIdx];
        remoteDestination[channelIdx].addr += ctx.localOffset;
        remoteDestination[channelIdx].addr += chunkOffset;
        remoteDestination[channelIdx].token = ctx.remoteOutputToken[channelIdx];
        CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIdx], remoteDestination[channelIdx], source,
            chunkBytes, ctx.peerEvents[0], static_cast<uint16_t>(1U << channelIdx)));
    }

    if (ctx.arg->copyLocalSlice != 0U) {
        ccu::LocalAddr localDestination;
        localDestination.addr = ctx.outputAddr;
        localDestination.addr += ctx.localOffset;
        localDestination.addr += chunkOffset;
        localDestination.token = ctx.outputToken;
        CCU_CHK_RET(ccu::LocalCopy(
            localDestination, source, chunkBytes, ctx.localEvents[0], static_cast<uint16_t>(1U)));
    }

    CCU_CHK_RET(ccu::EventWait(ctx.peerEvents[0], ChannelEventMask(ctx.arg->channelCount)));
    if (ctx.arg->copyLocalSlice != 0U) {
        CCU_CHK_RET(ccu::EventWait(ctx.localEvents[0], static_cast<uint16_t>(1U)));
    }
    return CCU_SUCCESS;
}

CcuResult RunFixedSmall(TransferContext &ctx, uint64_t sliceBytes)
{
    if (sliceBytes == 0U || sliceBytes >= MAX_DATA_SIZE) {
        return CcuResult::CCU_E_PARA;
    }

    ccu::LocalAddr source;
    source.addr = ctx.sourceAddr;
    source.token = ctx.sourceToken;

    ccu::Variable transferBytes;
    transferBytes = sliceBytes;
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        ccu::RemoteAddr remoteDestination;
        remoteDestination.addr = ctx.remoteOutputAddr[channelIdx];
        remoteDestination.addr += ctx.localOffset;
        remoteDestination.token = ctx.remoteOutputToken[channelIdx];
        CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIdx], remoteDestination, source,
            transferBytes, ctx.peerEvents[0], static_cast<uint16_t>(1U << channelIdx)));
    }

    if (ctx.arg->copyLocalSlice != 0U) {
        ccu::LocalAddr localDestination;
        localDestination.addr = ctx.outputAddr;
        localDestination.addr += ctx.localOffset;
        localDestination.token = ctx.outputToken;
        // 本地拷贝与远端 Write 同批提交，统一等待 Event，避免小消息额外串行化。
        CCU_CHK_RET(ccu::LocalCopy(
            localDestination, source, transferBytes, ctx.localEvents[0], static_cast<uint16_t>(1U)));
    }

    CCU_CHK_RET(ccu::EventWait(ctx.peerEvents[0], ChannelEventMask(ctx.arg->channelCount)));
    if (ctx.arg->copyLocalSlice != 0U) {
        CCU_CHK_RET(ccu::EventWait(ctx.localEvents[0], static_cast<uint16_t>(1U)));
    }
    return CCU_SUCCESS;
}

CcuResult SubmitDirectLocalCopy(TransferContext &ctx, uint64_t sliceBytes)
{
    if (ctx.arg->copyLocalSlice == 0U) {
        return CCU_SUCCESS;
    }
    if (sliceBytes == 0U || sliceBytes >= MAX_DATA_SIZE) {
        return CcuResult::CCU_E_PARA;
    }

    ccu::LocalAddr source;
    source.addr = ctx.sourceAddr;
    source.token = ctx.sourceToken;

    ccu::LocalAddr destination;
    destination.addr = ctx.outputAddr;
    destination.addr += ctx.localOffset;
    destination.token = ctx.outputToken;

    ccu::Variable transferBytes;
    transferBytes = sliceBytes;
    return ccu::LocalCopy(
        destination, source, transferBytes, ctx.localEvents[0], static_cast<uint16_t>(1U));
}

CcuResult SubmitDirectPull(TransferContext &ctx, uint64_t sliceBytes)
{
    if (sliceBytes == 0U || sliceBytes >= MAX_DATA_SIZE
        || ctx.arg->sourceMode != AllGatherSourceMode::DIRECT_INPUT) {
        return CcuResult::CCU_E_PARA;
    }

    ccu::Variable transferBytes;
    transferBytes = sliceBytes;
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        const uint32_t peerRank = ctx.arg->peerRanks[channelIdx];
        if (peerRank >= ctx.arg->rankSize || peerRank == ctx.arg->rankId) {
            return CcuResult::CCU_E_PARA;
        }

        ccu::RemoteAddr remoteSource;
        remoteSource.addr = ctx.remoteOutputAddr[channelIdx];
        remoteSource.token = ctx.remoteOutputToken[channelIdx];

        ccu::LocalAddr localDestination;
        localDestination.addr = ctx.outputAddr;
        ccu::Variable peerOffset;
        peerOffset = static_cast<uint64_t>(peerRank) * sliceBytes;
        localDestination.addr += peerOffset;
        localDestination.token = ctx.outputToken;
        CCU_CHK_RET(ccu::Read(ctx.arg->channels[channelIdx], localDestination, remoteSource,
            transferBytes, ctx.peerEvents[0], static_cast<uint16_t>(1U << channelIdx)));
    }

    return CCU_SUCCESS;
}

CcuResult WaitDirectPull(TransferContext &ctx)
{
    CCU_CHK_RET(ccu::EventWait(
        ctx.peerEvents[0], ChannelEventMask(ctx.arg->channelCount)));
    if (ctx.arg->copyLocalSlice != 0U) {
        CCU_CHK_RET(ccu::EventWait(ctx.localEvents[0], static_cast<uint16_t>(1U)));
    }
    return CCU_SUCCESS;
}

CcuResult RunP2X8LargeSingleDma(TransferContext &ctx, uint64_t sliceBytes)
{
    if (sliceBytes == 0U || sliceBytes >= MAX_DATA_SIZE) {
        return CcuResult::CCU_E_PARA;
    }

    ccu::LocalAddr source;
    source.addr = ctx.sourceAddr;
    source.token = ctx.sourceToken;

    ccu::Variable transferBytes;
    transferBytes = sliceBytes;
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        ccu::RemoteAddr remoteDestination;
        remoteDestination.addr = ctx.remoteOutputAddr[channelIdx];
        remoteDestination.addr += ctx.localOffset;
        remoteDestination.token = ctx.remoteOutputToken[channelIdx];
        CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIdx], remoteDestination, source,
            transferBytes, ctx.peerEvents[0], static_cast<uint16_t>(1U << channelIdx)));
    }

    if (ctx.arg->copyLocalSlice != 0U) {
        ccu::LocalAddr localDestination;
        localDestination.addr = ctx.outputAddr;
        localDestination.addr += ctx.localOffset;
        localDestination.token = ctx.outputToken;
        // 双 Die 中仅 primary Kernel 提交一次完整 Slice 本地 Copy，并与两层远端 Write 重叠。
        CCU_CHK_RET(ccu::LocalCopy(
            localDestination, source, transferBytes, ctx.localEvents[0], static_cast<uint16_t>(1U)));
    }

    CCU_CHK_RET(ccu::EventWait(ctx.peerEvents[0], ChannelEventMask(ctx.arg->channelCount)));
    if (ctx.arg->copyLocalSlice != 0U) {
        CCU_CHK_RET(ccu::EventWait(ctx.localEvents[0], static_cast<uint16_t>(1U)));
    }
    return CCU_SUCCESS;
}

uint32_t FindHybridRankSlot(
    const std::array<uint32_t, P2X8_SERVER_RANKS> &ranks, uint32_t rank)
{
    for (uint32_t slot = 0; slot < P2X8_SERVER_RANKS; ++slot) {
        if (ranks[slot] == rank) {
            return slot;
        }
    }
    return HYBRID_INVALID_SLOT;
}

bool ValidateHybridRouteTable()
{
    std::array<uint32_t, P2X8_SERVER_RANKS> relayDifferenceCounts{};
    uint32_t totalExtras = 0U;
    uint32_t totalRelays = 0U;
    for (uint32_t stripe = 0; stripe < ALLGATHER_P2X8_HYBRID_STRIPE_COUNT; ++stripe) {
        std::array<bool, P2X8_SERVER_RANKS> phaseASeed{};
        std::array<bool, P2X8_SERVER_RANKS> direct{};
        std::array<bool, P2X8_SERVER_RANKS> relayed{};
        uint32_t stripeRelayDifferenceMask = 0U;
        for (uint32_t seed = 0; seed < 4U; ++seed) {
            const uint32_t slot = HYBRID_PHASE_A_SEEDS[stripe][seed];
            if (slot >= P2X8_SERVER_RANKS || direct[slot]) {
                return false;
            }
            phaseASeed[slot] = true;
            direct[slot] = true;
        }
        const uint32_t extraCount = HYBRID_PHASE_B_EXTRA_COUNT[stripe];
        if (extraCount == 0U || extraCount > 2U) {
            return false;
        }
        totalExtras += extraCount;
        for (uint32_t extra = 0; extra < extraCount; ++extra) {
            const uint32_t slot = HYBRID_PHASE_B_EXTRAS[stripe][extra];
            if (slot >= P2X8_SERVER_RANKS || direct[slot]) {
                return false;
            }
            direct[slot] = true;
        }

        const uint32_t relayCount = HYBRID_PHASE_B_RELAY_COUNT[stripe];
        if (relayCount != P2X8_SERVER_RANKS - 4U - extraCount) {
            return false;
        }
        totalRelays += relayCount;
        for (uint32_t relay = 0; relay < relayCount; ++relay) {
            const uint32_t source = HYBRID_PHASE_B_RELAYS[stripe][relay][0];
            const uint32_t destination = HYBRID_PHASE_B_RELAYS[stripe][relay][1];
            if (source >= P2X8_SERVER_RANKS || destination >= P2X8_SERVER_RANKS
                || !phaseASeed[source] || direct[destination] || relayed[destination]) {
                return false;
            }
            relayed[destination] = true;
            const uint32_t difference
                = (destination + P2X8_SERVER_RANKS - source) % P2X8_SERVER_RANKS;
            const uint32_t differenceBit = 1U << difference;
            if (difference == 0U || (stripeRelayDifferenceMask & differenceBit) != 0U) {
                return false;
            }
            stripeRelayDifferenceMask |= differenceBit;
            ++relayDifferenceCounts[difference];
        }
        for (uint32_t slot = 0; slot < P2X8_SERVER_RANKS; ++slot) {
            if (!direct[slot] && !relayed[slot]) {
                return false;
            }
        }
    }

    uint32_t totalPhaseARuns = 0U;
    for (uint32_t peerOffset = 0; peerOffset < P2X8_SERVER_RANKS; ++peerOffset) {
        const uint32_t runCount = HYBRID_PHASE_A_RUN_COUNT[peerOffset];
        if (runCount == 0U || runCount > 2U) {
            return false;
        }
        totalPhaseARuns += runCount;
        std::array<bool, ALLGATHER_P2X8_HYBRID_STRIPE_COUNT> runCoverage{};
        uint32_t previousEnd = 0U;
        for (uint32_t runIndex = 0; runIndex < runCount; ++runIndex) {
            const HybridPhaseASeedRun run = HYBRID_PHASE_A_RUNS[peerOffset][runIndex];
            if (run.beginStripe >= run.endStripe
                || run.endStripe > ALLGATHER_P2X8_HYBRID_STRIPE_COUNT
                || (runIndex != 0U && previousEnd >= run.beginStripe)) {
                return false;
            }
            previousEnd = run.endStripe;
            for (uint32_t stripe = run.beginStripe; stripe < run.endStripe; ++stripe) {
                if (runCoverage[stripe]) {
                    return false;
                }
                runCoverage[stripe] = true;
            }
        }
        for (uint32_t runIndex = runCount; runIndex < 2U; ++runIndex) {
            const HybridPhaseASeedRun run = HYBRID_PHASE_A_RUNS[peerOffset][runIndex];
            if (run.beginStripe != HYBRID_UNUSED || run.endStripe != HYBRID_UNUSED) {
                return false;
            }
        }
        for (uint32_t stripe = 0; stripe < ALLGATHER_P2X8_HYBRID_STRIPE_COUNT; ++stripe) {
            bool seedCoverage = false;
            for (uint32_t seed = 0; seed < 4U; ++seed) {
                seedCoverage = seedCoverage || HYBRID_PHASE_A_SEEDS[stripe][seed] == peerOffset;
            }
            if (runCoverage[stripe] != seedCoverage) {
                return false;
            }
        }
    }
    if (totalPhaseARuns != 14U || totalExtras != 16U || totalRelays != 28U) {
        return false;
    }
    for (uint32_t difference = 1U; difference < P2X8_SERVER_RANKS; ++difference) {
        if (relayDifferenceCounts[difference] != 4U) {
            return false;
        }
    }
    return true;
}

uint32_t P8P4RelayCount(uint32_t stripe)
{
    return stripe + 1U == ALLGATHER_P8P4_HYBRID_STRIPE_COUNT ? 4U : 3U;
}

uint32_t P8P4RelayHead(uint32_t sourceSlot, uint32_t stripe, uint32_t relayIndex)
{
    if (sourceSlot >= P8P4_SMALL_SERVER_RANKS
        || stripe >= ALLGATHER_P8P4_HYBRID_STRIPE_COUNT
        || relayIndex >= P8P4RelayCount(stripe)) {
        return P8P4_INVALID_SLOT;
    }
    if (stripe < 6U) {
        return stripe + relayIndex;
    }
    if (stripe == 6U) {
        constexpr uint8_t heads[3] = {0U, 6U, 7U};
        return heads[relayIndex];
    }
    if (stripe == 7U) {
        constexpr uint8_t heads[3] = {0U, 1U, 7U};
        return heads[relayIndex];
    }
    return P8P4_SPECIAL_MISSING[sourceSlot][relayIndex];
}

bool IsP8P4PhaseASeed(uint32_t sourceSlot, uint32_t destinationSlot, uint32_t stripe)
{
    if (sourceSlot >= P8P4_SMALL_SERVER_RANKS
        || destinationSlot >= P8P4_LARGE_SERVER_RANKS
        || stripe >= ALLGATHER_P8P4_HYBRID_STRIPE_COUNT) {
        return false;
    }
    const uint32_t runCount = P8P4_PHASE_A_RUN_COUNT[sourceSlot][destinationSlot];
    for (uint32_t runIndex = 0U; runIndex < runCount; ++runIndex) {
        const HybridPhaseASeedRun run
            = P8P4_PHASE_A_RUNS[sourceSlot][destinationSlot][runIndex];
        if (run.beginStripe <= stripe && stripe < run.endStripe) {
            return true;
        }
    }
    return false;
}

bool ValidateP8P4HybridRouteTable()
{
    std::array<std::array<uint32_t, P8P4_LARGE_SERVER_RANKS>,
        P8P4_LARGE_SERVER_RANKS> relayEdgeCounts{};
    uint32_t totalPhaseARuns = 0U;
    uint32_t totalPhaseASeeds = 0U;
    uint32_t totalExtras = 0U;
    uint32_t totalRelays = 0U;
    for (uint32_t sourceSlot = 0U; sourceSlot < P8P4_SMALL_SERVER_RANKS; ++sourceSlot) {
        std::array<uint32_t, P8P4_LARGE_SERVER_RANKS> directCounts{};
        for (uint32_t destinationSlot = 0U;
             destinationSlot < P8P4_LARGE_SERVER_RANKS; ++destinationSlot) {
            const uint32_t runCount = P8P4_PHASE_A_RUN_COUNT[sourceSlot][destinationSlot];
            if (runCount > 2U) {
                return false;
            }
            totalPhaseARuns += runCount;
            uint32_t previousEnd = 0U;
            for (uint32_t runIndex = 0U; runIndex < 2U; ++runIndex) {
                const HybridPhaseASeedRun run
                    = P8P4_PHASE_A_RUNS[sourceSlot][destinationSlot][runIndex];
                if (runIndex < runCount) {
                    if (run.beginStripe >= run.endStripe
                        || run.endStripe > ALLGATHER_P8P4_HYBRID_STRIPE_COUNT
                        || (runIndex != 0U && previousEnd >= run.beginStripe)) {
                        return false;
                    }
                    previousEnd = run.endStripe;
                } else if (run.beginStripe != HYBRID_UNUSED || run.endStripe != HYBRID_UNUSED) {
                    return false;
                }
            }
        }

        for (uint32_t stripe = 0U; stripe < ALLGATHER_P8P4_HYBRID_STRIPE_COUNT; ++stripe) {
            std::array<bool, P8P4_LARGE_SERVER_RANKS> phaseASeed{};
            std::array<bool, P8P4_LARGE_SERVER_RANKS> direct{};
            std::array<bool, P8P4_LARGE_SERVER_RANKS> relayed{};
            uint32_t phaseASeedCount = 0U;
            for (uint32_t destinationSlot = 0U;
                 destinationSlot < P8P4_LARGE_SERVER_RANKS; ++destinationSlot) {
                phaseASeed[destinationSlot]
                    = IsP8P4PhaseASeed(sourceSlot, destinationSlot, stripe);
                direct[destinationSlot] = phaseASeed[destinationSlot];
                phaseASeedCount += phaseASeed[destinationSlot] ? 1U : 0U;
            }
            if (phaseASeedCount != 4U) {
                return false;
            }
            totalPhaseASeeds += phaseASeedCount;

            if (stripe + 1U < ALLGATHER_P8P4_HYBRID_STRIPE_COUNT) {
                const uint32_t extra = P8P4_PHASE_B_EXTRAS[sourceSlot][stripe];
                if (extra >= P8P4_LARGE_SERVER_RANKS || direct[extra]) {
                    return false;
                }
                direct[extra] = true;
                ++totalExtras;
            }

            const uint32_t relayCount = P8P4RelayCount(stripe);
            totalRelays += relayCount;
            for (uint32_t relayIndex = 0U; relayIndex < relayCount; ++relayIndex) {
                const uint32_t head = P8P4RelayHead(sourceSlot, stripe, relayIndex);
                const uint32_t tail = P8P4_RELAY_TAILS[sourceSlot][stripe][relayIndex];
                if (head >= P8P4_LARGE_SERVER_RANKS || tail >= P8P4_LARGE_SERVER_RANKS
                    || head == tail || direct[head] || relayed[head] || !phaseASeed[tail]) {
                    return false;
                }
                relayed[head] = true;
                ++relayEdgeCounts[tail][head];
            }
            for (uint32_t relayIndex = relayCount; relayIndex < 4U; ++relayIndex) {
                if (P8P4_RELAY_TAILS[sourceSlot][stripe][relayIndex] != HYBRID_UNUSED) {
                    return false;
                }
            }
            for (uint32_t destinationSlot = 0U;
                 destinationSlot < P8P4_LARGE_SERVER_RANKS; ++destinationSlot) {
                if (direct[destinationSlot] == relayed[destinationSlot]) {
                    return false;
                }
                directCounts[destinationSlot] += direct[destinationSlot] ? 1U : 0U;
            }
        }
        for (uint32_t destinationSlot = 0U;
             destinationSlot < P8P4_LARGE_SERVER_RANKS; ++destinationSlot) {
            if (directCounts[destinationSlot] != 5U && directCounts[destinationSlot] != 6U) {
                return false;
            }
        }
    }

    if (totalPhaseARuns != 46U || totalPhaseASeeds != 144U
        || totalExtras != 32U || totalRelays != 112U) {
        return false;
    }
    for (uint32_t tail = 0U; tail < P8P4_LARGE_SERVER_RANKS; ++tail) {
        for (uint32_t head = 0U; head < P8P4_LARGE_SERVER_RANKS; ++head) {
            const uint32_t expected = tail == head ? 0U : 2U;
            if (relayEdgeCounts[tail][head] != expected) {
                return false;
            }
        }
    }
    return true;
}

uint64_t GetP8P4CappedHighPhaseAEnd(const P8P4BidirectionalGeometry &geometry,
    uint64_t sliceBytes, uint32_t sourceGroup, uint32_t destinationSlot)
{
    if (sourceGroup >= P8P4_SMALL_SERVER_RANKS
        || destinationSlot >= P8P4_SMALL_SERVER_RANKS) {
        return 0U;
    }
    if (destinationSlot == sourceGroup) {
        return geometry.tailBytes;
    }
    const uint32_t splitEarlyDestination = (sourceGroup + 3U) & 3U;
    return destinationSlot == splitEarlyDestination ? geometry.relayBytes : sliceBytes;
}

bool GetP8P4CappedHighPhaseBRange(const P8P4BidirectionalGeometry &geometry,
    uint64_t sliceBytes, uint32_t sourceGroup, uint32_t destinationSlot,
    uint64_t &beginBytes, uint64_t &endBytes)
{
    beginBytes = 0U;
    endBytes = 0U;
    if (sourceGroup >= P8P4_SMALL_SERVER_RANKS
        || destinationSlot >= P8P4_SMALL_SERVER_RANKS) {
        return false;
    }
    if (destinationSlot == sourceGroup) {
        beginBytes = geometry.tailBytes;
    } else if (destinationSlot == ((sourceGroup + 3U) & 3U)) {
        beginBytes = geometry.relayBytes;
    } else {
        return false;
    }
    endBytes = sliceBytes;
    return beginBytes < endBytes;
}

bool ValidateP8P4BidirectionalRoute(const P8P4BidirectionalGeometry &geometry,
    uint64_t sliceBytes, bool capPeerBandwidth)
{
    if (geometry.halfBytes == 0U || geometry.halfBytes >= sliceBytes
        || geometry.relayBytes == 0U || geometry.relayBytes >= sliceBytes
        || geometry.tailBytes == 0U || geometry.tailBytes >= sliceBytes
        || geometry.relayBytes + geometry.tailBytes != sliceBytes
        || geometry.tailBytes > geometry.relayBytes) {
        return false;
    }
    std::array<uint32_t, P8P4_LARGE_SERVER_RANKS> halfRootCounts{};
    std::array<uint32_t, P8P4_SMALL_SERVER_RANKS> phaseAReceiveCounts{};
    std::array<uint32_t, P8P4_SMALL_SERVER_RANKS> phaseBReceiveCounts{};
    std::array<std::array<uint32_t, P8P4_SMALL_SERVER_RANKS>,
        P8P4_SMALL_SERVER_RANKS> relayEdgeCounts{};
    for (uint32_t smallSlot = 0U; smallSlot < P8P4_SMALL_SERVER_RANKS; ++smallSlot) {
        ++halfRootCounts[2U * smallSlot];
        ++halfRootCounts[2U * smallSlot + 1U];
        // A_smallSlot 在 Phase A 向 Root 完整直发，Phase B 向其他三个 U 发 tail。
        ++phaseAReceiveCounts[smallSlot];
        for (uint32_t destinationSlot = 0U;
             destinationSlot < P8P4_SMALL_SERVER_RANKS; ++destinationSlot) {
            if (destinationSlot == smallSlot) {
                // Root 已在 Phase A 收到低位 A Slice 的完整区间。
            } else {
                ++phaseBReceiveCounts[destinationSlot];
                ++relayEdgeCounts[smallSlot][destinationSlot];
            }

            // A_(4+smallSlot) 默认整片只出现在一个阶段；capped 路径按同一
            // Rank Pair 的互补前后缀覆盖，且禁止缺口或重叠。
            if (!capPeerBandwidth) {
                if (destinationSlot == smallSlot) {
                    ++phaseBReceiveCounts[destinationSlot];
                } else {
                    ++phaseAReceiveCounts[destinationSlot];
                }
                continue;
            }
            const uint64_t phaseAEnd = GetP8P4CappedHighPhaseAEnd(
                geometry, sliceBytes, smallSlot, destinationSlot);
            if (phaseAEnd == 0U || phaseAEnd > sliceBytes) {
                return false;
            }
            ++phaseAReceiveCounts[destinationSlot];
            uint64_t phaseBBegin = 0U;
            uint64_t phaseBEnd = 0U;
            const bool hasPhaseB = GetP8P4CappedHighPhaseBRange(geometry,
                sliceBytes, smallSlot, destinationSlot, phaseBBegin, phaseBEnd);
            if (hasPhaseB) {
                if (phaseBBegin != phaseAEnd || phaseBEnd != sliceBytes) {
                    return false;
                }
                ++phaseBReceiveCounts[destinationSlot];
            } else if (phaseAEnd != sliceBytes) {
                return false;
            }
        }
    }
    for (uint32_t largeSlot = 0U; largeSlot < P8P4_LARGE_SERVER_RANKS; ++largeSlot) {
        if (halfRootCounts[largeSlot] != 1U) {
            return false;
        }
    }
    for (uint32_t destinationSlot = 0U;
         destinationSlot < P8P4_SMALL_SERVER_RANKS; ++destinationSlot) {
        const uint32_t expectedReceiveCount = capPeerBandwidth ? 5U : 4U;
        if (phaseAReceiveCounts[destinationSlot] != expectedReceiveCount
            || phaseBReceiveCounts[destinationSlot] != expectedReceiveCount) {
            return false;
        }
        for (uint32_t sourceSlot = 0U;
             sourceSlot < P8P4_SMALL_SERVER_RANKS; ++sourceSlot) {
            const uint32_t expected = sourceSlot == destinationSlot ? 0U : 1U;
            if (relayEdgeCounts[sourceSlot][destinationSlot] != expected) {
                return false;
            }
        }
    }
    return true;
}

bool GetP8P4SharedRelayRatio(
    AllGatherDataPath dataPath, uint64_t &numerator, uint64_t &denominator)
{
    if (dataPath == AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_SHARED) {
        numerator = 5U;
        denominator = 7U;
        return true;
    }
    if (dataPath == AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_SHARED_5_8) {
        numerator = 5U;
        denominator = 8U;
        return true;
    }
    if (dataPath == AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_FULL_SEED_4_7) {
        numerator = 4U;
        denominator = 7U;
        return true;
    }
    return false;
}

bool ValidateP8P4SharedBidirectionalRoute(const P8P4BidirectionalGeometry &geometry,
    uint64_t sliceBytes, uint64_t dataTypeSize, AllGatherDataPath dataPath)
{
    if (dataTypeSize == 0U || sliceBytes % dataTypeSize != 0U
        || geometry.halfBytes % dataTypeSize != 0U
        || geometry.relayBytes % dataTypeSize != 0U
        || geometry.tailBytes % dataTypeSize != 0U
        || geometry.halfBytes == 0U || geometry.halfBytes >= sliceBytes
        || geometry.relayBytes == 0U || geometry.relayBytes >= sliceBytes
        || geometry.tailBytes == 0U || geometry.tailBytes >= sliceBytes
        || geometry.relayBytes + geometry.tailBytes != sliceBytes
        || geometry.tailBytes > geometry.relayBytes) {
        return false;
    }
    uint64_t relayNumerator = 0U;
    uint64_t relayDenominator = 0U;
    if (!GetP8P4SharedRelayRatio(dataPath, relayNumerator, relayDenominator)) {
        return false;
    }
    const uint64_t elementCount = sliceBytes / dataTypeSize;
    const uint64_t expectedHalfElements = elementCount / 2U + elementCount % 2U;
    const uint64_t expectedRelayElements
        = relayNumerator * (elementCount / relayDenominator)
          + (relayNumerator * (elementCount % relayDenominator)
                + relayDenominator - 1U)
                / relayDenominator;
    if (geometry.halfBytes != expectedHalfElements * dataTypeSize
        || geometry.relayBytes != expectedRelayElements * dataTypeSize) {
        return false;
    }
    std::array<uint32_t, P8P4_LARGE_SERVER_RANKS> halfRootCounts{};
    std::array<uint32_t, P8P4_SMALL_SERVER_RANKS> phaseAReceiveCounts{};
    std::array<uint32_t, P8P4_SMALL_SERVER_RANKS> phaseBReceiveCounts{};
    std::array<std::array<uint32_t, P8P4_SMALL_SERVER_RANKS>,
        P8P4_SMALL_SERVER_RANKS> relayEdgeCounts{};
    const bool fullSeed
        = dataPath == AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_FULL_SEED_4_7;
    for (uint32_t sourceGroup = 0U;
         sourceGroup < P8P4_SMALL_SERVER_RANKS; ++sourceGroup) {
        ++halfRootCounts[2U * sourceGroup];
        ++halfRootCounts[2U * sourceGroup + 1U];
        // Shared publishes the relay prefix; Full-Seed publishes the complete
        // low-A Slice. Both routes have one matching Phase-A receive per U.
        ++phaseAReceiveCounts[sourceGroup];
        for (uint32_t destinationSlot = 0U;
             destinationSlot < P8P4_SMALL_SERVER_RANKS; ++destinationSlot) {
            // Every destination receives the high-A full Slice. Full-Seed has
            // already completed the matching low-A Slice in Phase A.
            ++phaseBReceiveCounts[destinationSlot];
            if (!fullSeed || destinationSlot != sourceGroup) {
                ++phaseBReceiveCounts[destinationSlot];
            }
            if (destinationSlot != sourceGroup) {
                ++relayEdgeCounts[sourceGroup][destinationSlot];
            }
        }
    }
    for (uint32_t largeSlot = 0U;
         largeSlot < P8P4_LARGE_SERVER_RANKS; ++largeSlot) {
        if (halfRootCounts[largeSlot] != 1U) {
            return false;
        }
    }
    for (uint32_t destinationSlot = 0U;
         destinationSlot < P8P4_SMALL_SERVER_RANKS; ++destinationSlot) {
        const uint32_t expectedPhaseBReceives = fullSeed ? 7U : 8U;
        if (phaseAReceiveCounts[destinationSlot] != 1U
            || phaseBReceiveCounts[destinationSlot] != expectedPhaseBReceives) {
            return false;
        }
        for (uint32_t sourceGroup = 0U;
             sourceGroup < P8P4_SMALL_SERVER_RANKS; ++sourceGroup) {
            const uint32_t expected = sourceGroup == destinationSlot ? 0U : 1U;
            if (relayEdgeCounts[sourceGroup][destinationSlot] != expected) {
                return false;
            }
        }
    }
    return true;
}

uint32_t FindP8P4LargeRankSlot(
    const std::array<uint32_t, P8P4_LARGE_SERVER_RANKS> &ranks, uint32_t rank)
{
    for (uint32_t slot = 0U; slot < P8P4_LARGE_SERVER_RANKS; ++slot) {
        if (ranks[slot] == rank) {
            return slot;
        }
    }
    return P8P4_INVALID_SLOT;
}

uint32_t FindP8P4SmallRankSlot(
    const std::array<uint32_t, P8P4_SMALL_SERVER_RANKS> &ranks, uint32_t rank)
{
    for (uint32_t slot = 0U; slot < P8P4_SMALL_SERVER_RANKS; ++slot) {
        if (ranks[slot] == rank) {
            return slot;
        }
    }
    return P8P4_INVALID_SLOT;
}

CcuResult InitP8P4PhasedResources(TransferContext &ctx, P8P4HybridRankMap &rankMap)
{
    if (ctx.arg->rankSize != 12U || ctx.arg->localRankMask == 0U
        || (ctx.arg->localRankMask & (1U << ctx.arg->rankId)) == 0U) {
        return CcuResult::CCU_E_PARA;
    }

    uint32_t localCount = 0U;
    for (uint32_t rank = 0U; rank < ctx.arg->rankSize; ++rank) {
        localCount += (ctx.arg->localRankMask & (1U << rank)) != 0U ? 1U : 0U;
    }
    if (localCount != P8P4_LARGE_SERVER_RANKS && localCount != P8P4_SMALL_SERVER_RANKS) {
        return CcuResult::CCU_E_PARA;
    }
    rankMap.isLargeSide = localCount == P8P4_LARGE_SERVER_RANKS;
    uint32_t largeCount = 0U;
    uint32_t smallCount = 0U;
    for (uint32_t rank = 0U; rank < ctx.arg->rankSize; ++rank) {
        const bool local = (ctx.arg->localRankMask & (1U << rank)) != 0U;
        const bool large = rankMap.isLargeSide ? local : !local;
        if (large) {
            if (largeCount >= P8P4_LARGE_SERVER_RANKS) {
                return CcuResult::CCU_E_PARA;
            }
            rankMap.largeRanks[largeCount++] = rank;
        } else {
            if (smallCount >= P8P4_SMALL_SERVER_RANKS) {
                return CcuResult::CCU_E_PARA;
            }
            rankMap.smallRanks[smallCount++] = rank;
        }
    }
    if (largeCount != P8P4_LARGE_SERVER_RANKS || smallCount != P8P4_SMALL_SERVER_RANKS) {
        return CcuResult::CCU_E_PARA;
    }
    rankMap.selfSlot = rankMap.isLargeSide
                           ? FindP8P4LargeRankSlot(rankMap.largeRanks, ctx.arg->rankId)
                           : FindP8P4SmallRankSlot(rankMap.smallRanks, ctx.arg->rankId);
    if (rankMap.selfSlot >= (rankMap.isLargeSide
            ? P8P4_LARGE_SERVER_RANKS : P8P4_SMALL_SERVER_RANKS)) {
        return CcuResult::CCU_E_PARA;
    }

    rankMap.channelByLargeSlot.fill(HYBRID_INVALID_CHANNEL);
    rankMap.channelBySmallSlot.fill(HYBRID_INVALID_CHANNEL);
    const bool isMesh = ctx.arg->layerRole == AllGatherLayerRole::LOCAL_MESH;
    if (!isMesh && ctx.arg->layerRole != AllGatherLayerRole::CLOS) {
        return CcuResult::CCU_E_PARA;
    }
    const uint32_t expectedChannelCount = isMesh
        ? localCount - 1U
        : ctx.arg->rankSize - localCount;
    if (ctx.arg->channelCount != expectedChannelCount) {
        return CcuResult::CCU_E_PARA;
    }

    uint32_t peerMask = 0U;
    for (uint32_t channelIdx = 0U; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        const uint32_t peerRank = ctx.arg->peerRanks[channelIdx];
        if (peerRank >= ctx.arg->rankSize || peerRank == ctx.arg->rankId
            || (peerMask & (1U << peerRank)) != 0U) {
            return CcuResult::CCU_E_PARA;
        }
        peerMask |= 1U << peerRank;
        const bool peerIsLarge = FindP8P4LargeRankSlot(rankMap.largeRanks, peerRank)
                                 < P8P4_LARGE_SERVER_RANKS;
        if (peerIsLarge) {
            const uint32_t peerSlot = FindP8P4LargeRankSlot(rankMap.largeRanks, peerRank);
            if (rankMap.channelByLargeSlot[peerSlot] != HYBRID_INVALID_CHANNEL) {
                return CcuResult::CCU_E_PARA;
            }
            rankMap.channelByLargeSlot[peerSlot] = channelIdx;
        } else {
            const uint32_t peerSlot = FindP8P4SmallRankSlot(rankMap.smallRanks, peerRank);
            if (peerSlot >= P8P4_SMALL_SERVER_RANKS
                || rankMap.channelBySmallSlot[peerSlot] != HYBRID_INVALID_CHANNEL) {
                return CcuResult::CCU_E_PARA;
            }
            rankMap.channelBySmallSlot[peerSlot] = channelIdx;
        }
    }

    for (uint32_t slot = 0U; slot < P8P4_LARGE_SERVER_RANKS; ++slot) {
        const bool expected = isMesh
            ? (rankMap.isLargeSide && slot != rankMap.selfSlot)
            : !rankMap.isLargeSide;
        const bool present = rankMap.channelByLargeSlot[slot] != HYBRID_INVALID_CHANNEL;
        if (expected != present) {
            return CcuResult::CCU_E_PARA;
        }
    }
    for (uint32_t slot = 0U; slot < P8P4_SMALL_SERVER_RANKS; ++slot) {
        const bool expected = isMesh
            ? (!rankMap.isLargeSide && slot != rankMap.selfSlot)
            : rankMap.isLargeSide;
        const bool present = rankMap.channelBySmallSlot[slot] != HYBRID_INVALID_CHANNEL;
        if (expected != present) {
            return CcuResult::CCU_E_PARA;
        }
    }

    CCU_CHK_RET(InitTransferResources(ctx));
    if (ctx.peerEvents.size() != 2U) {
        return CcuResult::CCU_E_PARA;
    }
    return CCU_SUCCESS;
}

CcuResult InitP2X8HybridResources(TransferContext &ctx, HybridRankMap &rankMap)
{
    if (!ValidateHybridRouteTable() || ctx.arg->rankSize != 2U * P2X8_SERVER_RANKS
        || ctx.arg->localRankMask == 0U
        || (ctx.arg->localRankMask & (1U << ctx.arg->rankId)) == 0U) {
        return CcuResult::CCU_E_PARA;
    }

    uint32_t localCount = 0U;
    uint32_t remoteCount = 0U;
    for (uint32_t rank = 0; rank < ctx.arg->rankSize; ++rank) {
        if ((ctx.arg->localRankMask & (1U << rank)) != 0U) {
            if (localCount >= P2X8_SERVER_RANKS) {
                return CcuResult::CCU_E_PARA;
            }
            rankMap.localRanks[localCount++] = rank;
        } else {
            if (remoteCount >= P2X8_SERVER_RANKS) {
                return CcuResult::CCU_E_PARA;
            }
            rankMap.remoteRanks[remoteCount++] = rank;
        }
    }
    if (localCount != P2X8_SERVER_RANKS || remoteCount != P2X8_SERVER_RANKS) {
        return CcuResult::CCU_E_PARA;
    }
    rankMap.selfSlot = FindHybridRankSlot(rankMap.localRanks, ctx.arg->rankId);
    if (rankMap.selfSlot >= P2X8_SERVER_RANKS) {
        return CcuResult::CCU_E_PARA;
    }

    rankMap.channelByPeerSlot.fill(HYBRID_INVALID_CHANNEL);
    uint32_t peerMask = 0U;
    const bool isMesh = ctx.arg->layerRole == AllGatherLayerRole::LOCAL_MESH;
    const uint32_t expectedChannelCount = isMesh ? P2X8_SERVER_RANKS - 1U : P2X8_SERVER_RANKS;
    if (ctx.arg->channelCount != expectedChannelCount
        || (!isMesh && ctx.arg->layerRole != AllGatherLayerRole::CLOS)
        || (!isMesh && ctx.arg->copyLocalSlice != 0U)) {
        return CcuResult::CCU_E_PARA;
    }
    const auto &peerRanks = isMesh ? rankMap.localRanks : rankMap.remoteRanks;
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        const uint32_t peerRank = ctx.arg->peerRanks[channelIdx];
        if (peerRank >= ctx.arg->rankSize || peerRank == ctx.arg->rankId
            || (peerMask & (1U << peerRank)) != 0U) {
            return CcuResult::CCU_E_PARA;
        }
        const uint32_t peerSlot = FindHybridRankSlot(peerRanks, peerRank);
        if (peerSlot >= P2X8_SERVER_RANKS
            || rankMap.channelByPeerSlot[peerSlot] != HYBRID_INVALID_CHANNEL) {
            return CcuResult::CCU_E_PARA;
        }
        peerMask |= 1U << peerRank;
        rankMap.channelByPeerSlot[peerSlot] = channelIdx;
    }
    for (uint32_t peerSlot = 0; peerSlot < P2X8_SERVER_RANKS; ++peerSlot) {
        const bool channelExpected = !isMesh || peerSlot != rankMap.selfSlot;
        const bool channelPresent
            = rankMap.channelByPeerSlot[peerSlot] != HYBRID_INVALID_CHANNEL;
        if (channelExpected != channelPresent) {
            return CcuResult::CCU_E_PARA;
        }
    }

    CCU_CHK_RET(InitTransferResources(ctx));
    if (ctx.peerEvents.size() != 2U) {
        return CcuResult::CCU_E_PARA;
    }
    return CCU_SUCCESS;
}

HybridStripeGeometry GetHybridStripeRangeGeometry(uint64_t sliceBytes, uint64_t dataTypeSize,
    uint32_t beginStripe, uint32_t endStripe)
{
    const uint64_t elementCount = sliceBytes / dataTypeSize;
    const uint64_t beginElement
        = elementCount * beginStripe / ALLGATHER_P2X8_HYBRID_STRIPE_COUNT;
    const uint64_t endElement
        = elementCount * endStripe / ALLGATHER_P2X8_HYBRID_STRIPE_COUNT;
    return HybridStripeGeometry{
        beginElement * dataTypeSize,
        (endElement - beginElement) * dataTypeSize,
    };
}

HybridStripeGeometry GetHybridStripeGeometry(
    uint64_t sliceBytes, uint64_t dataTypeSize, uint32_t stripe)
{
    return GetHybridStripeRangeGeometry(
        sliceBytes, dataTypeSize, stripe, stripe + 1U);
}

bool GetP2X8DistributedRootGeometry(uint64_t sliceBytes, uint64_t dataTypeSize,
    DistributedRootGeometry &geometry)
{
    if (sliceBytes == 0U || sliceBytes >= MAX_DATA_SIZE || dataTypeSize == 0U
        || sliceBytes % dataTypeSize != 0U) {
        return false;
    }
    const uint64_t elementCount = sliceBytes / dataTypeSize;
    const uint64_t quotient = elementCount / ALLGATHER_P2X8_HYBRID_STRIPE_COUNT;
    const uint64_t remainder = elementCount % ALLGATHER_P2X8_HYBRID_STRIPE_COUNT;
    const uint64_t scaledRemainder = 4U * remainder;
    const uint64_t roundedRemainder
        = (scaledRemainder + ALLGATHER_P2X8_HYBRID_STRIPE_COUNT - 1U)
          / ALLGATHER_P2X8_HYBRID_STRIPE_COUNT;
    const uint64_t relayElements = 4U * quotient + roundedRemainder;
    // delta = 11 * ceil(4N/11) - 4N，范围为 [0, 10]。
    const uint64_t roundingDelta
        = ALLGATHER_P2X8_HYBRID_STRIPE_COUNT * roundedRemainder - scaledRemainder;
    if (relayElements <= roundingDelta) {
        return false;
    }
    const uint64_t cappedLastRelayElements = relayElements - roundingDelta;
    if (relayElements == 0U || 2U * relayElements >= elementCount
        || relayElements + cappedLastRelayElements >= elementCount) {
        return false;
    }
    geometry.relayBytes = relayElements * dataTypeSize;
    geometry.cappedLastRelayBytes = cappedLastRelayElements * dataTypeSize;
    return geometry.relayBytes != 0U && geometry.cappedLastRelayBytes != 0U
           && geometry.relayBytes < sliceBytes
           && geometry.cappedLastRelayBytes <= geometry.relayBytes;
}

bool GetP8P4BidirectionalGeometry(uint64_t sliceBytes, uint64_t dataTypeSize,
    P8P4BidirectionalGeometry &geometry)
{
    if (sliceBytes == 0U || sliceBytes >= MAX_DATA_SIZE || dataTypeSize == 0U
        || sliceBytes % dataTypeSize != 0U) {
        return false;
    }
    const uint64_t elementCount = sliceBytes / dataTypeSize;
    const uint64_t halfElements = elementCount / 2U + elementCount % 2U;
    const uint64_t quotient = elementCount / 7U;
    const uint64_t remainder = elementCount % 7U;
    const uint64_t relayElements
        = 4U * quotient + (4U * remainder + 6U) / 7U;
    if (halfElements == 0U || halfElements >= elementCount
        || relayElements == 0U || relayElements >= elementCount) {
        return false;
    }
    geometry.halfBytes = halfElements * dataTypeSize;
    geometry.relayBytes = relayElements * dataTypeSize;
    geometry.tailBytes = sliceBytes - geometry.relayBytes;
    return geometry.halfBytes != 0U && geometry.halfBytes < sliceBytes
           && geometry.relayBytes != 0U && geometry.relayBytes < sliceBytes
           && geometry.tailBytes != 0U && geometry.tailBytes < sliceBytes;
}

bool GetP8P4SharedBidirectionalGeometry(uint64_t sliceBytes, uint64_t dataTypeSize,
    AllGatherDataPath dataPath, P8P4BidirectionalGeometry &geometry)
{
    if (sliceBytes == 0U || sliceBytes >= MAX_DATA_SIZE || dataTypeSize == 0U
        || sliceBytes % dataTypeSize != 0U) {
        return false;
    }
    uint64_t relayNumerator = 0U;
    uint64_t relayDenominator = 0U;
    if (!GetP8P4SharedRelayRatio(dataPath, relayNumerator, relayDenominator)) {
        return false;
    }
    const uint64_t elementCount = sliceBytes / dataTypeSize;
    const uint64_t halfElements = elementCount / 2U + elementCount % 2U;
    const uint64_t quotient = elementCount / relayDenominator;
    const uint64_t remainder = elementCount % relayDenominator;
    const uint64_t relayElements
        = relayNumerator * quotient
          + (relayNumerator * remainder + relayDenominator - 1U)
                / relayDenominator;
    if (halfElements == 0U || halfElements >= elementCount
        || relayElements == 0U || relayElements >= elementCount) {
        return false;
    }
    geometry.halfBytes = halfElements * dataTypeSize;
    geometry.relayBytes = relayElements * dataTypeSize;
    geometry.tailBytes = sliceBytes - geometry.relayBytes;
    return geometry.halfBytes != 0U && geometry.halfBytes < sliceBytes
           && geometry.relayBytes != 0U && geometry.relayBytes < sliceBytes
           && geometry.tailBytes != 0U && geometry.tailBytes < sliceBytes;
}

HybridStripeGeometry GetP8P4HybridStripeRangeGeometry(uint64_t sliceBytes, uint64_t dataTypeSize,
    uint32_t beginStripe, uint32_t endStripe)
{
    const uint64_t elementCount = sliceBytes / dataTypeSize;
    const uint64_t beginElement
        = elementCount * beginStripe / ALLGATHER_P8P4_HYBRID_STRIPE_COUNT;
    const uint64_t endElement
        = elementCount * endStripe / ALLGATHER_P8P4_HYBRID_STRIPE_COUNT;
    return HybridStripeGeometry{
        beginElement * dataTypeSize,
        (endElement - beginElement) * dataTypeSize,
    };
}

HybridStripeGeometry GetP8P4HybridStripeGeometry(
    uint64_t sliceBytes, uint64_t dataTypeSize, uint32_t stripe)
{
    return GetP8P4HybridStripeRangeGeometry(
        sliceBytes, dataTypeSize, stripe, stripe + 1U);
}

CcuResult ReuseHybridEventSlot(TransferContext &ctx,
    std::array<uint16_t, 2> &activeMasks, uint32_t iteration)
{
    const uint32_t eventSlot = iteration & 1U;
    if (activeMasks[eventSlot] != 0U) {
        CCU_CHK_RET(ccu::EventWait(ctx.peerEvents[eventSlot], activeMasks[eventSlot]));
        activeMasks[eventSlot] = 0U;
    }
    return CCU_SUCCESS;
}

CcuResult DrainHybridEvents(TransferContext &ctx, const std::array<uint16_t, 2> &activeMasks)
{
    for (uint32_t eventSlot = 0; eventSlot < 2U; ++eventSlot) {
        if (activeMasks[eventSlot] != 0U) {
            CCU_CHK_RET(ccu::EventWait(ctx.peerEvents[eventSlot], activeMasks[eventSlot]));
        }
    }
    return CCU_SUCCESS;
}

CcuResult WriteP2X8DistributedRootClosRange(TransferContext &ctx,
    const HybridRankMap &rankMap, uint32_t peerOffset, uint64_t beginBytes,
    uint64_t endBytes, uint16_t &activeMask, bool matchedSerial)
{
    if (peerOffset >= P2X8_SERVER_RANKS || beginBytes >= endBytes
        || endBytes > ctx.arg->fixedSliceBytes || endBytes - beginBytes >= MAX_DATA_SIZE) {
        return CcuResult::CCU_E_PARA;
    }
    const uint32_t peerSlot = GetP2X8DistributedRootPeerSlot(
        rankMap, peerOffset, matchedSerial);
    const uint32_t channelIdx = rankMap.channelByPeerSlot[peerSlot];
    if (channelIdx >= ctx.arg->channelCount) {
        return CcuResult::CCU_E_PARA;
    }
    const uint16_t eventMask = static_cast<uint16_t>(1U << channelIdx);
    if ((activeMask & eventMask) != 0U) {
        return CcuResult::CCU_E_PARA;
    }

    ccu::Variable rangeOffset;
    rangeOffset = beginBytes;
    ccu::Variable transferBytes;
    transferBytes = endBytes - beginBytes;
    ccu::LocalAddr source;
    source.addr = ctx.sourceAddr;
    source.addr += rangeOffset;
    source.token = ctx.sourceToken;
    ccu::RemoteAddr destination;
    destination.addr = ctx.remoteOutputAddr[channelIdx];
    destination.addr += ctx.localOffset;
    destination.addr += rangeOffset;
    destination.token = ctx.remoteOutputToken[channelIdx];
    CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIdx], destination, source,
        transferBytes, ctx.peerEvents[0], eventMask));
    activeMask = static_cast<uint16_t>(activeMask | eventMask);
    if (matchedSerial) {
        CCU_CHK_RET(ccu::EventWait(ctx.peerEvents[0], activeMask));
        activeMask = 0U;
    }
    return CCU_SUCCESS;
}

CcuResult RunP2X8DistributedRootClosPhaseA(
    TransferContext &ctx, const HybridRankMap &rankMap, bool capPeerBandwidth,
    bool matchedSerial)
{
    DistributedRootGeometry geometry;
    if (!GetP2X8DistributedRootGeometry(
            ctx.arg->fixedSliceBytes, ctx.arg->dataTypeSize, geometry)) {
        return CcuResult::CCU_E_PARA;
    }

    uint16_t activeMask = 0U;
    CCU_CHK_RET(WriteP2X8DistributedRootClosRange(
        ctx, rankMap, 0U, 0U, ctx.arg->fixedSliceBytes, activeMask, matchedSerial));
    const uint32_t fullTailEndOffset = capPeerBandwidth ? 3U : 4U;
    for (uint32_t peerOffset = 1U; peerOffset <= fullTailEndOffset; ++peerOffset) {
        CCU_CHK_RET(WriteP2X8DistributedRootClosRange(ctx, rankMap, peerOffset,
            geometry.relayBytes, ctx.arg->fixedSliceBytes, activeMask, matchedSerial));
    }
    if (capPeerBandwidth) {
        for (uint32_t peerOffset = 4U; peerOffset <= 6U; ++peerOffset) {
            CCU_CHK_RET(WriteP2X8DistributedRootClosRange(ctx, rankMap, peerOffset,
                geometry.relayBytes,
                ctx.arg->fixedSliceBytes - geometry.relayBytes, activeMask, matchedSerial));
        }
        CCU_CHK_RET(WriteP2X8DistributedRootClosRange(ctx, rankMap, 7U,
            geometry.relayBytes,
            ctx.arg->fixedSliceBytes - geometry.cappedLastRelayBytes, activeMask, matchedSerial));
    }
    if (activeMask != 0U) {
        CCU_CHK_RET(ccu::EventWait(ctx.peerEvents[0], activeMask));
    }
    return CCU_SUCCESS;
}

CcuResult RunP2X8DistributedRootClosPhaseB(
    TransferContext &ctx, const HybridRankMap &rankMap, bool capPeerBandwidth,
    bool matchedSerial)
{
    DistributedRootGeometry geometry;
    if (!GetP2X8DistributedRootGeometry(
            ctx.arg->fixedSliceBytes, ctx.arg->dataTypeSize, geometry)) {
        return CcuResult::CCU_E_PARA;
    }

    uint16_t activeMask = 0U;
    if (capPeerBandwidth) {
        for (uint32_t peerOffset = 4U; peerOffset <= 6U; ++peerOffset) {
            CCU_CHK_RET(WriteP2X8DistributedRootClosRange(ctx, rankMap, peerOffset,
                ctx.arg->fixedSliceBytes - geometry.relayBytes,
                ctx.arg->fixedSliceBytes, activeMask, matchedSerial));
        }
        CCU_CHK_RET(WriteP2X8DistributedRootClosRange(ctx, rankMap, 7U,
            ctx.arg->fixedSliceBytes - geometry.cappedLastRelayBytes,
            ctx.arg->fixedSliceBytes, activeMask, matchedSerial));
    } else {
        for (uint32_t peerOffset = 5U; peerOffset <= 7U; ++peerOffset) {
            CCU_CHK_RET(WriteP2X8DistributedRootClosRange(ctx, rankMap, peerOffset,
                geometry.relayBytes, ctx.arg->fixedSliceBytes, activeMask, matchedSerial));
        }
    }
    if (activeMask != 0U) {
        CCU_CHK_RET(ccu::EventWait(ctx.peerEvents[0], activeMask));
    }
    return CCU_SUCCESS;
}

CcuResult RunP2X8DistributedRootMeshPhaseB(
    TransferContext &ctx, const HybridRankMap &rankMap)
{
    DistributedRootGeometry geometry;
    if (!GetP2X8DistributedRootGeometry(
            ctx.arg->fixedSliceBytes, ctx.arg->dataTypeSize, geometry)) {
        return CcuResult::CCU_E_PARA;
    }
    const uint32_t sourceRank = rankMap.remoteRanks[rankMap.selfSlot];
    if (sourceRank >= ctx.arg->rankSize) {
        return CcuResult::CCU_E_PARA;
    }

    const uint64_t sourceRankOffset
        = static_cast<uint64_t>(sourceRank) * ctx.arg->fixedSliceBytes;
    ccu::Variable sourceOffset;
    sourceOffset = sourceRankOffset;
    ccu::Variable transferBytes;
    transferBytes = geometry.relayBytes;
    ccu::LocalAddr source;
    source.addr = ctx.outputAddr;
    source.addr += sourceOffset;
    source.token = ctx.outputToken;

    uint16_t activeMask = 0U;
    for (uint32_t destinationOffset = 1U;
         destinationOffset < P2X8_SERVER_RANKS; ++destinationOffset) {
        const uint32_t destinationSlot = (rankMap.selfSlot + destinationOffset) & 7U;
        const uint32_t channelIdx = rankMap.channelByPeerSlot[destinationSlot];
        if (channelIdx >= ctx.arg->channelCount) {
            return CcuResult::CCU_E_PARA;
        }
        const uint16_t eventMask = static_cast<uint16_t>(1U << channelIdx);
        if ((activeMask & eventMask) != 0U) {
            return CcuResult::CCU_E_PARA;
        }
        ccu::RemoteAddr destination;
        destination.addr = ctx.remoteOutputAddr[channelIdx];
        destination.addr += sourceOffset;
        destination.token = ctx.remoteOutputToken[channelIdx];
        CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIdx], destination, source,
            transferBytes, ctx.peerEvents[0], eventMask));
        activeMask = static_cast<uint16_t>(activeMask | eventMask);
    }
    CCU_CHK_RET(ccu::EventWait(ctx.peerEvents[0], activeMask));
    return CCU_SUCCESS;
}

CcuResult RunP2X8HybridClosPhaseA(TransferContext &ctx, const HybridRankMap &rankMap)
{
    std::array<uint16_t, 2> activeMasks{};
    ccu::Variable transferBytes;
    ccu::Variable runOffset;
    for (uint32_t eventSlot = 0; eventSlot < 2U; ++eventSlot) {
        for (uint32_t peerOffset = 0; peerOffset < P2X8_SERVER_RANKS; ++peerOffset) {
            if (eventSlot >= HYBRID_PHASE_A_RUN_COUNT[peerOffset]) {
                continue;
            }
            const HybridPhaseASeedRun run = HYBRID_PHASE_A_RUNS[peerOffset][eventSlot];
            const HybridStripeGeometry geometry = GetHybridStripeRangeGeometry(
                ctx.arg->fixedSliceBytes, ctx.arg->dataTypeSize, run.beginStripe, run.endStripe);
            if (geometry.bytes == 0U || geometry.bytes >= MAX_DATA_SIZE) {
                return CcuResult::CCU_E_PARA;
            }
            transferBytes = geometry.bytes;
            runOffset = geometry.offset;
            ccu::LocalAddr source;
            source.addr = ctx.sourceAddr;
            source.addr += runOffset;
            source.token = ctx.sourceToken;
            const uint32_t peerSlot = (rankMap.selfSlot + peerOffset) & 7U;
            const uint32_t channelIdx = rankMap.channelByPeerSlot[peerSlot];
            if (channelIdx >= ctx.arg->channelCount) {
                return CcuResult::CCU_E_PARA;
            }
            ccu::RemoteAddr destination;
            destination.addr = ctx.remoteOutputAddr[channelIdx];
            destination.addr += ctx.localOffset;
            destination.addr += runOffset;
            destination.token = ctx.remoteOutputToken[channelIdx];
            const uint16_t eventMask = static_cast<uint16_t>(1U << channelIdx);
            if ((activeMasks[eventSlot] & eventMask) != 0U) {
                return CcuResult::CCU_E_PARA;
            }
            CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIdx], destination, source,
                transferBytes, ctx.peerEvents[eventSlot], eventMask));
            activeMasks[eventSlot] = static_cast<uint16_t>(activeMasks[eventSlot] | eventMask);
        }
    }
    return DrainHybridEvents(ctx, activeMasks);
}

CcuResult RunP2X8HybridClosPhaseB(TransferContext &ctx, const HybridRankMap &rankMap)
{
    std::array<uint16_t, 2> activeMasks{};
    ccu::Variable transferBytes;
    ccu::Variable stripeOffset;
    for (uint32_t stripe = 0; stripe < ALLGATHER_P2X8_HYBRID_STRIPE_COUNT; ++stripe) {
        CCU_CHK_RET(ReuseHybridEventSlot(ctx, activeMasks, stripe));
        const uint32_t eventSlot = stripe & 1U;
        const HybridStripeGeometry geometry
            = GetHybridStripeGeometry(ctx.arg->fixedSliceBytes, ctx.arg->dataTypeSize, stripe);
        if (geometry.bytes == 0U || geometry.bytes >= MAX_DATA_SIZE) {
            return CcuResult::CCU_E_PARA;
        }
        transferBytes = geometry.bytes;
        stripeOffset = geometry.offset;
        ccu::LocalAddr source;
        source.addr = ctx.sourceAddr;
        source.addr += stripeOffset;
        source.token = ctx.sourceToken;
        const uint32_t extraCount = HYBRID_PHASE_B_EXTRA_COUNT[stripe];
        for (uint32_t extra = 0; extra < extraCount; ++extra) {
            const uint32_t peerSlot
                = (rankMap.selfSlot + HYBRID_PHASE_B_EXTRAS[stripe][extra]) & 7U;
            const uint32_t channelIdx = rankMap.channelByPeerSlot[peerSlot];
            if (channelIdx >= ctx.arg->channelCount) {
                return CcuResult::CCU_E_PARA;
            }
            ccu::RemoteAddr destination;
            destination.addr = ctx.remoteOutputAddr[channelIdx];
            destination.addr += ctx.localOffset;
            destination.addr += stripeOffset;
            destination.token = ctx.remoteOutputToken[channelIdx];
            const uint16_t eventMask = static_cast<uint16_t>(1U << channelIdx);
            CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIdx], destination, source,
                transferBytes, ctx.peerEvents[eventSlot], eventMask));
            activeMasks[eventSlot] = static_cast<uint16_t>(activeMasks[eventSlot] | eventMask);
        }
    }
    return DrainHybridEvents(ctx, activeMasks);
}

CcuResult RunP2X8HybridMeshPhaseB(TransferContext &ctx, const HybridRankMap &rankMap)
{
    std::array<uint16_t, 2> activeMasks{};
    ccu::Variable transferBytes;
    ccu::Variable sourceOffset;
    for (uint32_t stripe = 0; stripe < ALLGATHER_P2X8_HYBRID_STRIPE_COUNT; ++stripe) {
        CCU_CHK_RET(ReuseHybridEventSlot(ctx, activeMasks, stripe));
        const uint32_t eventSlot = stripe & 1U;
        const HybridStripeGeometry geometry
            = GetHybridStripeGeometry(ctx.arg->fixedSliceBytes, ctx.arg->dataTypeSize, stripe);
        if (geometry.bytes == 0U || geometry.bytes >= MAX_DATA_SIZE) {
            return CcuResult::CCU_E_PARA;
        }
        transferBytes = geometry.bytes;
        const uint32_t relayCount = HYBRID_PHASE_B_RELAY_COUNT[stripe];
        for (uint32_t relay = 0; relay < relayCount; ++relay) {
            const uint32_t relayOffset = HYBRID_PHASE_B_RELAYS[stripe][relay][0];
            const uint32_t destinationOffset = HYBRID_PHASE_B_RELAYS[stripe][relay][1];
            const uint32_t remoteSourceSlot
                = (rankMap.selfSlot + P2X8_SERVER_RANKS - relayOffset) & 7U;
            const uint32_t destinationSlot = (rankMap.selfSlot + destinationOffset
                + P2X8_SERVER_RANKS - relayOffset) & 7U;
            const uint32_t sourceRank = rankMap.remoteRanks[remoteSourceSlot];
            const uint32_t channelIdx = rankMap.channelByPeerSlot[destinationSlot];
            if (sourceRank >= ctx.arg->rankSize || channelIdx >= ctx.arg->channelCount) {
                return CcuResult::CCU_E_PARA;
            }

            const uint64_t sourceRankOffset
                = static_cast<uint64_t>(sourceRank) * ctx.arg->fixedSliceBytes;
            sourceOffset = sourceRankOffset + geometry.offset;
            ccu::LocalAddr source;
            source.addr = ctx.outputAddr;
            source.addr += sourceOffset;
            source.token = ctx.outputToken;
            ccu::RemoteAddr destination;
            destination.addr = ctx.remoteOutputAddr[channelIdx];
            destination.addr += sourceOffset;
            destination.token = ctx.remoteOutputToken[channelIdx];
            const uint16_t eventMask = static_cast<uint16_t>(1U << channelIdx);
            CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIdx], destination, source,
                transferBytes, ctx.peerEvents[eventSlot], eventMask));
            activeMasks[eventSlot] = static_cast<uint16_t>(activeMasks[eventSlot] | eventMask);
        }
    }
    return DrainHybridEvents(ctx, activeMasks);
}

CcuResult RunP8P4HybridClosPhaseA(TransferContext &ctx, const P8P4HybridRankMap &rankMap)
{
    if (rankMap.isLargeSide) {
        // 8 卡侧向 4 个远端 Rank 各直发完整 Slice，正好占满 4B Clos 聚合带宽。
        return RunP8P4LargeSingleDma(ctx, ctx.arg->fixedSliceBytes);
    }
    if (ctx.arg->copyLocalSlice != 0U || rankMap.selfSlot >= P8P4_SMALL_SERVER_RANKS) {
        return CcuResult::CCU_E_PARA;
    }

    std::array<uint16_t, 2> activeMasks{};
    ccu::Variable transferBytes;
    ccu::Variable runOffset;
    for (uint32_t eventSlot = 0U; eventSlot < 2U; ++eventSlot) {
        for (uint32_t destinationSlot = 0U;
             destinationSlot < P8P4_LARGE_SERVER_RANKS; ++destinationSlot) {
            if (eventSlot >= P8P4_PHASE_A_RUN_COUNT[rankMap.selfSlot][destinationSlot]) {
                continue;
            }
            const HybridPhaseASeedRun run
                = P8P4_PHASE_A_RUNS[rankMap.selfSlot][destinationSlot][eventSlot];
            const HybridStripeGeometry geometry = GetP8P4HybridStripeRangeGeometry(
                ctx.arg->fixedSliceBytes, ctx.arg->dataTypeSize,
                run.beginStripe, run.endStripe);
            if (geometry.bytes == 0U || geometry.bytes >= MAX_DATA_SIZE) {
                return CcuResult::CCU_E_PARA;
            }
            const uint32_t channelIdx = rankMap.channelByLargeSlot[destinationSlot];
            if (channelIdx >= ctx.arg->channelCount) {
                return CcuResult::CCU_E_PARA;
            }
            transferBytes = geometry.bytes;
            runOffset = geometry.offset;
            ccu::LocalAddr source;
            source.addr = ctx.sourceAddr;
            source.addr += runOffset;
            source.token = ctx.sourceToken;
            ccu::RemoteAddr destination;
            destination.addr = ctx.remoteOutputAddr[channelIdx];
            destination.addr += ctx.localOffset;
            destination.addr += runOffset;
            destination.token = ctx.remoteOutputToken[channelIdx];
            const uint16_t eventMask = static_cast<uint16_t>(1U << channelIdx);
            if ((activeMasks[eventSlot] & eventMask) != 0U) {
                return CcuResult::CCU_E_PARA;
            }
            CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIdx], destination, source,
                transferBytes, ctx.peerEvents[eventSlot], eventMask));
            activeMasks[eventSlot] = static_cast<uint16_t>(activeMasks[eventSlot] | eventMask);
        }
    }
    return DrainHybridEvents(ctx, activeMasks);
}

CcuResult RunP8P4HybridClosPhaseB(TransferContext &ctx, const P8P4HybridRankMap &rankMap)
{
    if (rankMap.isLargeSide) {
        return CCU_SUCCESS;
    }
    if (rankMap.selfSlot >= P8P4_SMALL_SERVER_RANKS) {
        return CcuResult::CCU_E_PARA;
    }

    std::array<uint16_t, 2> activeMasks{};
    ccu::Variable transferBytes;
    ccu::Variable stripeOffset;
    for (uint32_t stripe = 0U;
         stripe + 1U < ALLGATHER_P8P4_HYBRID_STRIPE_COUNT; ++stripe) {
        CCU_CHK_RET(ReuseHybridEventSlot(ctx, activeMasks, stripe));
        const uint32_t eventSlot = stripe & 1U;
        const HybridStripeGeometry geometry = GetP8P4HybridStripeGeometry(
            ctx.arg->fixedSliceBytes, ctx.arg->dataTypeSize, stripe);
        if (geometry.bytes == 0U || geometry.bytes >= MAX_DATA_SIZE) {
            return CcuResult::CCU_E_PARA;
        }
        const uint32_t destinationSlot = P8P4_PHASE_B_EXTRAS[rankMap.selfSlot][stripe];
        const uint32_t channelIdx = rankMap.channelByLargeSlot[destinationSlot];
        if (channelIdx >= ctx.arg->channelCount) {
            return CcuResult::CCU_E_PARA;
        }
        transferBytes = geometry.bytes;
        stripeOffset = geometry.offset;
        ccu::LocalAddr source;
        source.addr = ctx.sourceAddr;
        source.addr += stripeOffset;
        source.token = ctx.sourceToken;
        ccu::RemoteAddr destination;
        destination.addr = ctx.remoteOutputAddr[channelIdx];
        destination.addr += ctx.localOffset;
        destination.addr += stripeOffset;
        destination.token = ctx.remoteOutputToken[channelIdx];
        const uint16_t eventMask = static_cast<uint16_t>(1U << channelIdx);
        CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIdx], destination, source,
            transferBytes, ctx.peerEvents[eventSlot], eventMask));
        activeMasks[eventSlot] = static_cast<uint16_t>(activeMasks[eventSlot] | eventMask);
    }
    return DrainHybridEvents(ctx, activeMasks);
}

CcuResult RunP8P4HybridMeshPhaseB(TransferContext &ctx, const P8P4HybridRankMap &rankMap)
{
    if (!rankMap.isLargeSide) {
        return CCU_SUCCESS;
    }
    if (rankMap.selfSlot >= P8P4_LARGE_SERVER_RANKS) {
        return CcuResult::CCU_E_PARA;
    }

    ccu::Variable transferBytes;
    ccu::Variable sourceOffset;
    std::array<uint16_t, 2> activeMasks{};
    std::array<uint32_t, P8P4_LARGE_SERVER_RANKS> relayCountByDestination{};
    // 路由表保证本 Rank 到每个本地 Peer 恰好有两个 relay stripe。分别放入两个
    // Event 槽后统一 drain，使 7 条独立 Mesh Channel 并行且不复用未完成的 Event bit。
    for (uint32_t sourceSlot = 0U; sourceSlot < P8P4_SMALL_SERVER_RANKS; ++sourceSlot) {
        for (uint32_t stripe = 0U; stripe < ALLGATHER_P8P4_HYBRID_STRIPE_COUNT; ++stripe) {
            const HybridStripeGeometry geometry = GetP8P4HybridStripeGeometry(
                ctx.arg->fixedSliceBytes, ctx.arg->dataTypeSize, stripe);
            if (geometry.bytes == 0U || geometry.bytes >= MAX_DATA_SIZE) {
                return CcuResult::CCU_E_PARA;
            }
            transferBytes = geometry.bytes;
            const uint32_t relayCount = P8P4RelayCount(stripe);
            for (uint32_t relayIndex = 0U; relayIndex < relayCount; ++relayIndex) {
                if (P8P4_RELAY_TAILS[sourceSlot][stripe][relayIndex]
                    != rankMap.selfSlot) {
                    continue;
                }
                const uint32_t destinationSlot
                    = P8P4RelayHead(sourceSlot, stripe, relayIndex);
                const uint32_t channelIdx = rankMap.channelByLargeSlot[destinationSlot];
                if (destinationSlot == rankMap.selfSlot || channelIdx >= ctx.arg->channelCount) {
                    return CcuResult::CCU_E_PARA;
                }
                const uint32_t eventSlot = relayCountByDestination[destinationSlot]++;
                if (eventSlot >= activeMasks.size()) {
                    return CcuResult::CCU_E_PARA;
                }
                const uint32_t sourceRank = rankMap.smallRanks[sourceSlot];
                const uint64_t sourceRankOffset
                    = static_cast<uint64_t>(sourceRank) * ctx.arg->fixedSliceBytes;
                sourceOffset = sourceRankOffset + geometry.offset;
                ccu::LocalAddr source;
                source.addr = ctx.outputAddr;
                source.addr += sourceOffset;
                source.token = ctx.outputToken;
                ccu::RemoteAddr destination;
                destination.addr = ctx.remoteOutputAddr[channelIdx];
                destination.addr += sourceOffset;
                destination.token = ctx.remoteOutputToken[channelIdx];
                const uint16_t eventMask = static_cast<uint16_t>(1U << channelIdx);
                if ((activeMasks[eventSlot] & eventMask) != 0U) {
                    return CcuResult::CCU_E_PARA;
                }
                CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIdx], destination, source,
                    transferBytes, ctx.peerEvents[eventSlot], eventMask));
                activeMasks[eventSlot]
                    = static_cast<uint16_t>(activeMasks[eventSlot] | eventMask);
            }
        }
    }
    for (uint32_t destinationSlot = 0U;
         destinationSlot < P8P4_LARGE_SERVER_RANKS; ++destinationSlot) {
        const uint32_t expected = destinationSlot == rankMap.selfSlot ? 0U : 2U;
        if (relayCountByDestination[destinationSlot] != expected) {
            return CcuResult::CCU_E_PARA;
        }
    }
    return DrainHybridEvents(ctx, activeMasks);
}

CcuResult WriteP8P4BidirectionalClosRange(TransferContext &ctx, uint32_t channelIdx,
    uint64_t beginBytes, uint64_t endBytes, uint16_t &activeMask)
{
    if (channelIdx >= ctx.arg->channelCount || beginBytes >= endBytes
        || endBytes > ctx.arg->fixedSliceBytes || endBytes - beginBytes >= MAX_DATA_SIZE) {
        return CcuResult::CCU_E_PARA;
    }
    const uint16_t eventMask = static_cast<uint16_t>(1U << channelIdx);
    if ((activeMask & eventMask) != 0U) {
        return CcuResult::CCU_E_PARA;
    }
    ccu::Variable rangeOffset;
    rangeOffset = beginBytes;
    ccu::Variable transferBytes;
    transferBytes = endBytes - beginBytes;
    ccu::LocalAddr source;
    source.addr = ctx.sourceAddr;
    source.addr += rangeOffset;
    source.token = ctx.sourceToken;
    ccu::RemoteAddr destination;
    destination.addr = ctx.remoteOutputAddr[channelIdx];
    destination.addr += ctx.localOffset;
    destination.addr += rangeOffset;
    destination.token = ctx.remoteOutputToken[channelIdx];
    CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIdx], destination, source,
        transferBytes, ctx.peerEvents[0], eventMask));
    activeMask = static_cast<uint16_t>(activeMask | eventMask);
    return CCU_SUCCESS;
}

CcuResult RunP8P4BidirectionalClosPhaseA(TransferContext &ctx,
    const P8P4HybridRankMap &rankMap, const P8P4BidirectionalGeometry &geometry,
    bool capPeerBandwidth, bool matchedOrdering)
{
    if (ctx.arg->layerRole != AllGatherLayerRole::CLOS) {
        return CcuResult::CCU_E_PARA;
    }
    uint16_t activeMask = 0U;
    if (!rankMap.isLargeSide) {
        const uint32_t firstRoot = 2U * rankMap.selfSlot;
        const uint32_t secondRoot = firstRoot + 1U;
        CCU_CHK_RET(WriteP8P4BidirectionalClosRange(ctx,
            rankMap.channelByLargeSlot[firstRoot], 0U, geometry.halfBytes, activeMask));
        CCU_CHK_RET(WriteP8P4BidirectionalClosRange(ctx,
            rankMap.channelByLargeSlot[secondRoot], geometry.halfBytes,
            ctx.arg->fixedSliceBytes, activeMask));
    } else {
        const uint32_t sourceGroup = rankMap.selfSlot & 3U;
        if (rankMap.selfSlot < P8P4_SMALL_SERVER_RANKS) {
            CCU_CHK_RET(WriteP8P4BidirectionalClosRange(ctx,
                rankMap.channelBySmallSlot[sourceGroup], 0U,
                ctx.arg->fixedSliceBytes, activeMask));
        } else if (!capPeerBandwidth && matchedOrdering) {
            // Each relative offset is one A4..A7 -> U0..U3 perfect matching.
            for (uint32_t destinationOffset = 1U;
                 destinationOffset < P8P4_SMALL_SERVER_RANKS;
                 ++destinationOffset) {
                const uint32_t destinationSlot
                    = (sourceGroup + destinationOffset) & 3U;
                CCU_CHK_RET(WriteP8P4BidirectionalClosRange(ctx,
                    rankMap.channelBySmallSlot[destinationSlot], 0U,
                    ctx.arg->fixedSliceBytes, activeMask));
            }
        } else if (!capPeerBandwidth) {
            for (uint32_t destinationSlot = 0U;
                 destinationSlot < P8P4_SMALL_SERVER_RANKS; ++destinationSlot) {
                if (destinationSlot == sourceGroup) {
                    continue;
                }
                CCU_CHK_RET(WriteP8P4BidirectionalClosRange(ctx,
                    rankMap.channelBySmallSlot[destinationSlot], 0U,
                    ctx.arg->fixedSliceBytes, activeMask));
            }
        } else {
            for (uint32_t destinationSlot = 0U;
                 destinationSlot < P8P4_SMALL_SERVER_RANKS; ++destinationSlot) {
                const uint64_t endBytes = GetP8P4CappedHighPhaseAEnd(
                    geometry, ctx.arg->fixedSliceBytes, sourceGroup, destinationSlot);
                CCU_CHK_RET(WriteP8P4BidirectionalClosRange(ctx,
                    rankMap.channelBySmallSlot[destinationSlot], 0U,
                    endBytes, activeMask));
            }
        }
    }
    if (ctx.arg->copyLocalSlice != 0U) {
        ccu::LocalAddr source;
        source.addr = ctx.sourceAddr;
        source.token = ctx.sourceToken;
        ccu::LocalAddr destination;
        destination.addr = ctx.outputAddr;
        destination.addr += ctx.localOffset;
        destination.token = ctx.outputToken;
        ccu::Variable transferBytes;
        transferBytes = ctx.arg->fixedSliceBytes;
        CCU_CHK_RET(ccu::LocalCopy(destination, source, transferBytes,
            ctx.localEvents[0], static_cast<uint16_t>(1U)));
    }
    CCU_CHK_RET(ccu::EventWait(ctx.peerEvents[0], activeMask));
    if (ctx.arg->copyLocalSlice != 0U) {
        CCU_CHK_RET(ccu::EventWait(ctx.localEvents[0], static_cast<uint16_t>(1U)));
    }
    return CCU_SUCCESS;
}

CcuResult RunP8P4BidirectionalClosPhaseB(TransferContext &ctx,
    const P8P4HybridRankMap &rankMap, const P8P4BidirectionalGeometry &geometry,
    bool capPeerBandwidth, bool matchedOrdering)
{
    if (ctx.arg->layerRole != AllGatherLayerRole::CLOS) {
        return CcuResult::CCU_E_PARA;
    }
    if (!rankMap.isLargeSide) {
        return CCU_SUCCESS;
    }
    uint16_t activeMask = 0U;
    const uint32_t sourceGroup = rankMap.selfSlot & 3U;
    if (rankMap.selfSlot < P8P4_SMALL_SERVER_RANKS && matchedOrdering) {
        for (uint32_t destinationOffset = 1U;
             destinationOffset < P8P4_SMALL_SERVER_RANKS;
             ++destinationOffset) {
            const uint32_t destinationSlot
                = (sourceGroup + destinationOffset) & 3U;
            CCU_CHK_RET(WriteP8P4BidirectionalClosRange(ctx,
                rankMap.channelBySmallSlot[destinationSlot], geometry.relayBytes,
                ctx.arg->fixedSliceBytes, activeMask));
        }
    } else if (rankMap.selfSlot < P8P4_SMALL_SERVER_RANKS) {
        for (uint32_t destinationSlot = 0U;
             destinationSlot < P8P4_SMALL_SERVER_RANKS; ++destinationSlot) {
            if (destinationSlot == sourceGroup) {
                continue;
            }
            CCU_CHK_RET(WriteP8P4BidirectionalClosRange(ctx,
                rankMap.channelBySmallSlot[destinationSlot], geometry.relayBytes,
                ctx.arg->fixedSliceBytes, activeMask));
        }
    } else if (!capPeerBandwidth) {
        CCU_CHK_RET(WriteP8P4BidirectionalClosRange(ctx,
            rankMap.channelBySmallSlot[sourceGroup], 0U,
            ctx.arg->fixedSliceBytes, activeMask));
    } else {
        for (uint32_t destinationSlot = 0U;
             destinationSlot < P8P4_SMALL_SERVER_RANKS; ++destinationSlot) {
            uint64_t beginBytes = 0U;
            uint64_t endBytes = 0U;
            if (!GetP8P4CappedHighPhaseBRange(geometry,
                    ctx.arg->fixedSliceBytes, sourceGroup, destinationSlot,
                    beginBytes, endBytes)) {
                continue;
            }
            CCU_CHK_RET(WriteP8P4BidirectionalClosRange(ctx,
                rankMap.channelBySmallSlot[destinationSlot], beginBytes,
                endBytes, activeMask));
        }
    }
    CCU_CHK_RET(ccu::EventWait(ctx.peerEvents[0], activeMask));
    return CCU_SUCCESS;
}

CcuResult RunP8P4SharedClosPhaseA(TransferContext &ctx,
    const P8P4HybridRankMap &rankMap, const P8P4BidirectionalGeometry &geometry,
    bool fullSeed)
{
    if (ctx.arg->layerRole != AllGatherLayerRole::CLOS) {
        return CcuResult::CCU_E_PARA;
    }
    uint16_t activeMask = 0U;
    if (!rankMap.isLargeSide) {
        const uint32_t firstRoot = 2U * rankMap.selfSlot;
        const uint32_t secondRoot = firstRoot + 1U;
        CCU_CHK_RET(WriteP8P4BidirectionalClosRange(ctx,
            rankMap.channelByLargeSlot[firstRoot], 0U, geometry.halfBytes, activeMask));
        CCU_CHK_RET(WriteP8P4BidirectionalClosRange(ctx,
            rankMap.channelByLargeSlot[secondRoot], geometry.halfBytes,
            ctx.arg->fixedSliceBytes, activeMask));
    } else if (rankMap.selfSlot < P8P4_SMALL_SERVER_RANKS) {
        const uint64_t phaseAEnd
            = fullSeed ? ctx.arg->fixedSliceBytes : geometry.relayBytes;
        CCU_CHK_RET(WriteP8P4BidirectionalClosRange(ctx,
            rankMap.channelBySmallSlot[rankMap.selfSlot], 0U,
            phaseAEnd, activeMask));
    }

    if (ctx.arg->copyLocalSlice != 0U) {
        ccu::LocalAddr source;
        source.addr = ctx.sourceAddr;
        source.token = ctx.sourceToken;
        ccu::LocalAddr destination;
        destination.addr = ctx.outputAddr;
        destination.addr += ctx.localOffset;
        destination.token = ctx.outputToken;
        ccu::Variable transferBytes;
        transferBytes = ctx.arg->fixedSliceBytes;
        CCU_CHK_RET(ccu::LocalCopy(destination, source, transferBytes,
            ctx.localEvents[0], static_cast<uint16_t>(1U)));
    }
    if (activeMask != 0U) {
        CCU_CHK_RET(ccu::EventWait(ctx.peerEvents[0], activeMask));
    }
    if (ctx.arg->copyLocalSlice != 0U) {
        CCU_CHK_RET(ccu::EventWait(ctx.localEvents[0], static_cast<uint16_t>(1U)));
    }
    return CCU_SUCCESS;
}

CcuResult RunP8P4SharedClosPhaseB(TransferContext &ctx,
    const P8P4HybridRankMap &rankMap, const P8P4BidirectionalGeometry &geometry,
    bool fullSeed)
{
    if (ctx.arg->layerRole != AllGatherLayerRole::CLOS) {
        return CcuResult::CCU_E_PARA;
    }
    if (!rankMap.isLargeSide) {
        return CCU_SUCCESS;
    }

    uint16_t activeMask = 0U;
    const uint32_t sourceGroup = rankMap.selfSlot & 3U;
    const uint64_t beginBytes
        = rankMap.selfSlot < P8P4_SMALL_SERVER_RANKS ? geometry.relayBytes : 0U;
    for (uint32_t destinationOffset = 0U;
         destinationOffset < P8P4_SMALL_SERVER_RANKS; ++destinationOffset) {
        if (fullSeed && rankMap.selfSlot < P8P4_SMALL_SERVER_RANKS
            && destinationOffset == 0U) {
            // The matching U already received this complete low-A Slice in Phase A.
            continue;
        }
        const uint32_t destinationSlot
            = (sourceGroup + destinationOffset) & 3U;
        CCU_CHK_RET(WriteP8P4BidirectionalClosRange(ctx,
            rankMap.channelBySmallSlot[destinationSlot], beginBytes,
            ctx.arg->fixedSliceBytes, activeMask));
    }
    CCU_CHK_RET(ccu::EventWait(ctx.peerEvents[0], activeMask));
    return CCU_SUCCESS;
}

CcuResult RunP8P4BidirectionalMeshPhaseB(TransferContext &ctx,
    const P8P4HybridRankMap &rankMap, const P8P4BidirectionalGeometry &geometry)
{
    if (ctx.arg->layerRole != AllGatherLayerRole::LOCAL_MESH) {
        return CcuResult::CCU_E_PARA;
    }
    uint32_t sourceRank = 0U;
    uint64_t beginBytes = 0U;
    uint64_t endBytes = 0U;
    if (rankMap.isLargeSide) {
        const uint32_t sourceSlot = rankMap.selfSlot / 2U;
        sourceRank = rankMap.smallRanks[sourceSlot];
        beginBytes = (rankMap.selfSlot & 1U) == 0U ? 0U : geometry.halfBytes;
        endBytes = (rankMap.selfSlot & 1U) == 0U
                       ? geometry.halfBytes
                       : ctx.arg->fixedSliceBytes;
    } else {
        sourceRank = rankMap.largeRanks[rankMap.selfSlot];
        endBytes = geometry.relayBytes;
    }
    if (sourceRank >= ctx.arg->rankSize || beginBytes >= endBytes
        || endBytes > ctx.arg->fixedSliceBytes || endBytes - beginBytes >= MAX_DATA_SIZE) {
        return CcuResult::CCU_E_PARA;
    }

    const uint64_t sourceRankOffset
        = static_cast<uint64_t>(sourceRank) * ctx.arg->fixedSliceBytes;
    ccu::Variable rangeOffset;
    rangeOffset = sourceRankOffset + beginBytes;
    ccu::Variable transferBytes;
    transferBytes = endBytes - beginBytes;
    ccu::LocalAddr source;
    source.addr = ctx.outputAddr;
    source.addr += rangeOffset;
    source.token = ctx.outputToken;

    uint16_t activeMask = 0U;
    const uint32_t destinationCount = rankMap.isLargeSide
                                          ? P8P4_LARGE_SERVER_RANKS
                                          : P8P4_SMALL_SERVER_RANKS;
    for (uint32_t destinationSlot = 0U;
         destinationSlot < destinationCount; ++destinationSlot) {
        if (destinationSlot == rankMap.selfSlot) {
            continue;
        }
        const uint32_t channelIdx = rankMap.isLargeSide
                                        ? rankMap.channelByLargeSlot[destinationSlot]
                                        : rankMap.channelBySmallSlot[destinationSlot];
        if (channelIdx >= ctx.arg->channelCount) {
            return CcuResult::CCU_E_PARA;
        }
        const uint16_t eventMask = static_cast<uint16_t>(1U << channelIdx);
        if ((activeMask & eventMask) != 0U) {
            return CcuResult::CCU_E_PARA;
        }
        ccu::RemoteAddr destination;
        destination.addr = ctx.remoteOutputAddr[channelIdx];
        destination.addr += rangeOffset;
        destination.token = ctx.remoteOutputToken[channelIdx];
        CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIdx], destination, source,
            transferBytes, ctx.peerEvents[0], eventMask));
        activeMask = static_cast<uint16_t>(activeMask | eventMask);
    }
    CCU_CHK_RET(ccu::EventWait(ctx.peerEvents[0], activeMask));
    return CCU_SUCCESS;
}

CcuResult RunP4X1LargeSingleDma(TransferContext &ctx, uint64_t sliceBytes)
{
    if (sliceBytes == 0U || sliceBytes >= MAX_DATA_SIZE) {
        return CcuResult::CCU_E_PARA;
    }

    ccu::LocalAddr source;
    source.addr = ctx.sourceAddr;
    source.token = ctx.sourceToken;

    ccu::Variable transferBytes;
    transferBytes = sliceBytes;
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        ccu::RemoteAddr remoteDestination;
        remoteDestination.addr = ctx.remoteOutputAddr[channelIdx];
        remoteDestination.addr += ctx.localOffset;
        remoteDestination.token = ctx.remoteOutputToken[channelIdx];
        CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIdx], remoteDestination, source,
            transferBytes, ctx.peerEvents[0], static_cast<uint16_t>(1U << channelIdx)));
    }

    if (ctx.arg->copyLocalSlice != 0U) {
        ccu::LocalAddr localDestination;
        localDestination.addr = ctx.outputAddr;
        localDestination.addr += ctx.localOffset;
        localDestination.token = ctx.outputToken;
        // 仅 primary Clos Kernel 提交一次完整 Slice 本地 Copy，并与本组远端 Write 重叠。
        CCU_CHK_RET(ccu::LocalCopy(
            localDestination, source, transferBytes, ctx.localEvents[0], static_cast<uint16_t>(1U)));
    }

    CCU_CHK_RET(ccu::EventWait(ctx.peerEvents[0], ChannelEventMask(ctx.arg->channelCount)));
    if (ctx.arg->copyLocalSlice != 0U) {
        CCU_CHK_RET(ccu::EventWait(ctx.localEvents[0], static_cast<uint16_t>(1U)));
    }
    return CCU_SUCCESS;
}

bool HasP4X1MatchingChannelOrder(const CcuKernelArgAllGather *arg)
{
    if (arg == nullptr || arg->rankId >= ALLGATHER_P4X1_RANK_COUNT
        || arg->rankSize != ALLGATHER_P4X1_RANK_COUNT
        || arg->channelCount != ALLGATHER_P4X1_PEER_COUNT) {
        return false;
    }
    for (uint32_t channelIdx = 0U; channelIdx < arg->channelCount; ++channelIdx) {
        if (arg->peerRanks[channelIdx]
            != ALLGATHER_P4X1_MATCHING_ORDER[arg->rankId][channelIdx]) {
            return false;
        }
    }
    return true;
}

CcuResult RunP4X1MatchedReadySingleDma(TransferContext &ctx, uint64_t sliceBytes)
{
    if (sliceBytes == 0U || sliceBytes >= MAX_DATA_SIZE
        || !HasP4X1MatchingChannelOrder(ctx.arg)) {
        return CcuResult::CCU_E_PARA;
    }

    ccu::LocalAddr source;
    source.addr = ctx.sourceAddr;
    source.token = ctx.sourceToken;
    ccu::Variable transferBytes;
    transferBytes = sliceBytes;

    if (ctx.arg->copyLocalSlice != 0U) {
        ccu::LocalAddr localDestination;
        localDestination.addr = ctx.outputAddr;
        localDestination.addr += ctx.localOffset;
        localDestination.token = ctx.outputToken;
        // 身份发布后立即启动本地 Copy，使其与逐 Peer READY 等待和网络 Write 重叠。
        CCU_CHK_RET(ccu::LocalCopy(
            localDestination, source, transferBytes, ctx.localEvents[0], static_cast<uint16_t>(1U)));
    }

    for (uint32_t channelIdx = 0U; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        CCU_CHK_RET(ccu::NotifyWait(
            ctx.arg->channels[channelIdx], CHANNEL_NOTIFY_INDEX, PARAM_READY_MASK));
        ccu::RemoteAddr remoteDestination;
        remoteDestination.addr = ctx.remoteOutputAddr[channelIdx];
        remoteDestination.addr += ctx.localOffset;
        remoteDestination.token = ctx.remoteOutputToken[channelIdx];
        CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIdx], remoteDestination, source,
            transferBytes, ctx.peerEvents[0], static_cast<uint16_t>(1U << channelIdx)));
    }

    CCU_CHK_RET(ccu::EventWait(ctx.peerEvents[0], ChannelEventMask(ctx.arg->channelCount)));
    if (ctx.arg->copyLocalSlice != 0U) {
        CCU_CHK_RET(ccu::EventWait(ctx.localEvents[0], static_cast<uint16_t>(1U)));
    }
    return CCU_SUCCESS;
}

CcuResult RunP4X1MatchedSerialRolling2(TransferContext &ctx, uint64_t sliceBytes)
{
    constexpr uint64_t cellBytesLimit = ALLGATHER_P4X1_MATCHED_ROLLING2_CELL_BYTES;
    if (sliceBytes == 0U || sliceBytes >= MAX_DATA_SIZE
        || cellBytesLimit == 0U || cellBytesLimit >= MAX_DATA_SIZE
        || !HasP4X1MatchingChannelOrder(ctx.arg) || ctx.peerEvents.size() < 2U) {
        return CcuResult::CCU_E_PARA;
    }

    uint64_t cellCount = (sliceBytes + cellBytesLimit - 1U) / cellBytesLimit;
    if (cellCount > 1U && sliceBytes % cellBytesLimit != 0U
        && sliceBytes % cellBytesLimit <= 4096U) {
        --cellCount;
    }
    if (cellCount == 0U) {
        return CcuResult::CCU_E_PARA;
    }

    ccu::Variable fullSliceBytes;
    fullSliceBytes = sliceBytes;
    if (ctx.arg->copyLocalSlice != 0U) {
        ccu::LocalAddr localSource;
        localSource.addr = ctx.sourceAddr;
        localSource.token = ctx.sourceToken;
        ccu::LocalAddr localDestination;
        localDestination.addr = ctx.outputAddr;
        localDestination.addr += ctx.localOffset;
        localDestination.token = ctx.outputToken;
        CCU_CHK_RET(ccu::LocalCopy(localDestination, localSource, fullSliceBytes,
            ctx.localEvents[0], static_cast<uint16_t>(1U)));
    }

    ccu::Variable transferBytes;
    for (uint32_t channelIdx = 0U; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        CCU_CHK_RET(WaitOutputIdentityChannel(ctx, channelIdx));
        const uint16_t eventMask = static_cast<uint16_t>(1U << channelIdx);
        for (uint64_t cell = 0U; cell < cellCount; ++cell) {
            const uint32_t eventSlot = static_cast<uint32_t>(cell & 1U);
            if (cell >= 2U) {
                CCU_CHK_RET(ccu::EventWait(ctx.peerEvents[eventSlot], eventMask));
            }

            const uint64_t cellOffset = cell * cellBytesLimit;
            const uint64_t cellBytes = cell + 1U == cellCount
                                           ? sliceBytes - cellOffset
                                           : std::min(cellBytesLimit, sliceBytes - cellOffset);
            ccu::Variable cellOffsetVariable;
            cellOffsetVariable = cellOffset;
            // Address copy shares the underlying CCU address handle. Rebuild from the immutable
            // source Variable so offsets cannot accumulate across cells or matching peers.
            ccu::LocalAddr source;
            source.addr = ctx.sourceAddr;
            source.addr += cellOffsetVariable;
            source.token = ctx.sourceToken;
            ccu::RemoteAddr remoteDestination;
            remoteDestination.addr = ctx.remoteOutputAddr[channelIdx];
            remoteDestination.addr += ctx.localOffset;
            remoteDestination.addr += cellOffsetVariable;
            remoteDestination.token = ctx.remoteOutputToken[channelIdx];
            transferBytes = cellBytes;
            CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIdx], remoteDestination, source,
                transferBytes, ctx.peerEvents[eventSlot], eventMask));
        }

        const uint32_t firstDrainSlot = static_cast<uint32_t>(
            (cellCount - std::min<uint64_t>(2U, cellCount)) & 1U);
        CCU_CHK_RET(ccu::EventWait(ctx.peerEvents[firstDrainSlot], eventMask));
        if (cellCount > 1U) {
            const uint32_t secondDrainSlot = static_cast<uint32_t>((cellCount - 1U) & 1U);
            CCU_CHK_RET(ccu::EventWait(ctx.peerEvents[secondDrainSlot], eventMask));
        }
    }

    if (ctx.arg->copyLocalSlice != 0U) {
        CCU_CHK_RET(ccu::EventWait(ctx.localEvents[0], static_cast<uint16_t>(1U)));
    }
    return CCU_SUCCESS;
}

CcuResult RunP8P4LargeSingleDma(TransferContext &ctx, uint64_t sliceBytes)
{
    if (sliceBytes == 0U || sliceBytes >= MAX_DATA_SIZE) {
        return CcuResult::CCU_E_PARA;
    }

    ccu::LocalAddr source;
    source.addr = ctx.sourceAddr;
    source.token = ctx.sourceToken;

    ccu::Variable transferBytes;
    transferBytes = sliceBytes;
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        ccu::RemoteAddr remoteDestination;
        remoteDestination.addr = ctx.remoteOutputAddr[channelIdx];
        remoteDestination.addr += ctx.localOffset;
        remoteDestination.token = ctx.remoteOutputToken[channelIdx];
        CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIdx], remoteDestination, source,
            transferBytes, ctx.peerEvents[0], static_cast<uint16_t>(1U << channelIdx)));
    }

    if (ctx.arg->copyLocalSlice != 0U) {
        ccu::LocalAddr localDestination;
        localDestination.addr = ctx.outputAddr;
        localDestination.addr += ctx.localOffset;
        localDestination.token = ctx.outputToken;
        // 仅轻负载 Die 上的 primary 提交本地 Copy，与另一个 Die 的完整 Slice Write 重叠。
        CCU_CHK_RET(ccu::LocalCopy(
            localDestination, source, transferBytes, ctx.localEvents[0], static_cast<uint16_t>(1U)));
    }

    CCU_CHK_RET(ccu::EventWait(ctx.peerEvents[0], ChannelEventMask(ctx.arg->channelCount)));
    if (ctx.arg->copyLocalSlice != 0U) {
        CCU_CHK_RET(ccu::EventWait(ctx.localEvents[0], static_cast<uint16_t>(1U)));
    }
    return CCU_SUCCESS;
}

CcuResult RunFixedRolling2(TransferContext &ctx, uint64_t sliceBytes)
{
    const uint64_t cellBytesLimit = ctx.arg->cellBytes;
    if (cellBytesLimit == 0U || cellBytesLimit >= MAX_DATA_SIZE) {
        return CcuResult::CCU_E_PARA;
    }
    // 小消息 cell 等于完整 Slice，只产生一次 DMA；大消息按 Profile cell 使用 Rolling-2。
    // cell 数和最后一块都依据实际的每 Rank Slice 计算；若非整齐余数不超过 4 KiB，
    // 将其并入前一个 cell，避免额外的微小 DMA。退出前显式 drain 所有活跃 slot。
    uint64_t cellCount = (sliceBytes + cellBytesLimit - 1U) / cellBytesLimit;
    if (cellCount > 1U && sliceBytes % cellBytesLimit != 0U
        && sliceBytes % cellBytesLimit <= 4096U) {
        --cellCount;
    }
    if (cellCount == 0U) {
        return CcuResult::CCU_E_PARA;
    }

    ccu::LocalAddr source;
    source.addr = ctx.sourceAddr;
    source.token = ctx.sourceToken;

    ccu::LocalAddr localDestination;
    if (ctx.arg->copyLocalSlice != 0U) {
        localDestination.addr = ctx.outputAddr;
        localDestination.addr += ctx.localOffset;
        localDestination.token = ctx.outputToken;
    }

    std::vector<ccu::RemoteAddr> remoteDestination(ctx.arg->channelCount);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        remoteDestination[channelIdx].addr = ctx.remoteOutputAddr[channelIdx];
        remoteDestination[channelIdx].addr += ctx.localOffset;
        remoteDestination[channelIdx].token = ctx.remoteOutputToken[channelIdx];
    }

    const uint16_t allPeerMask = ChannelEventMask(ctx.arg->channelCount);
    ccu::Variable transferBytes;
    ccu::Variable advance;
    if (cellCount > 1U) {
        advance = cellBytesLimit;
    }
    for (uint64_t cell = 0; cell < cellCount; ++cell) {
        const uint32_t eventSlot = static_cast<uint32_t>(cell & 1U);
        if (cell >= 2U) {
            CCU_CHK_RET(ccu::EventWait(ctx.peerEvents[eventSlot], allPeerMask));
            if (ctx.arg->copyLocalSlice != 0U) {
                CCU_CHK_RET(ccu::EventWait(ctx.localEvents[eventSlot], static_cast<uint16_t>(1U)));
            }
        }

        const uint64_t cellOffset = cell * cellBytesLimit;
        const uint64_t cellBytes = cell + 1U == cellCount
                                       ? sliceBytes - cellOffset
                                       : std::min(cellBytesLimit, sliceBytes - cellOffset);
        transferBytes = cellBytes;
        for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
            CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIdx], remoteDestination[channelIdx], source,
                transferBytes, ctx.peerEvents[eventSlot], static_cast<uint16_t>(1U << channelIdx)));
        }
        if (ctx.arg->copyLocalSlice != 0U) {
            CCU_CHK_RET(ccu::LocalCopy(localDestination, source, transferBytes,
                ctx.localEvents[eventSlot], static_cast<uint16_t>(1U)));
        }

        if (cell + 1U < cellCount) {
            source.addr += advance;
            for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
                remoteDestination[channelIdx].addr += advance;
            }
            if (ctx.arg->copyLocalSlice != 0U) {
                localDestination.addr += advance;
            }
        }
    }

    const uint32_t firstDrainSlot = static_cast<uint32_t>((cellCount - std::min<uint64_t>(2U, cellCount)) & 1U);
    CCU_CHK_RET(ccu::EventWait(ctx.peerEvents[firstDrainSlot], allPeerMask));
    if (ctx.arg->copyLocalSlice != 0U) {
        CCU_CHK_RET(ccu::EventWait(ctx.localEvents[firstDrainSlot], static_cast<uint16_t>(1U)));
    }
    if (cellCount > 1U) {
        const uint32_t secondDrainSlot = static_cast<uint32_t>((cellCount - 1U) & 1U);
        CCU_CHK_RET(ccu::EventWait(ctx.peerEvents[secondDrainSlot], allPeerMask));
        if (ctx.arg->copyLocalSlice != 0U) {
            CCU_CHK_RET(ccu::EventWait(ctx.localEvents[secondDrainSlot], static_cast<uint16_t>(1U)));
        }
    }
    return CCU_SUCCESS;
}

CcuResult LoadFixedArgs(TransferContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.sourceAddr, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputAddr, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sourceToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputToken, argId++));
    ctx.localOffset = static_cast<uint64_t>(ctx.arg->rankId) * ctx.arg->fixedSliceBytes;
    return CCU_SUCCESS;
}

CcuResult RunProfileKernel(CcuKernelArg kernelArg, AllGatherProfile expectedProfile)
{
    auto *arg = static_cast<CcuKernelArgAllGather *>(kernelArg);
    AllGatherProfileSpec spec;
    if (!GetAllGatherProfileSpec(expectedProfile, spec)) {
        return CcuResult::CCU_E_PARA;
    }
    const AllGatherDataPath expectedDataPath
        = spec.algorithm == AllGatherAlgorithm::FIXED_DIRECT_SMALL
              ? AllGatherDataPath::FIXED_SMALL
              : AllGatherDataPath::ROLLING2;
    CCU_CHK_RET(ValidateProfileArg(arg, expectedProfile, expectedDataPath));
    TransferContext ctx;
    ctx.arg = arg;
    CCU_CHK_RET(InitTransferResources(ctx));
    CCU_CHK_RET(LoadFixedArgs(ctx));
    if (spec.algorithm == AllGatherAlgorithm::FIXED_DIRECT_SMALL
        && arg->sourceMode == AllGatherSourceMode::DIRECT_INPUT) {
        // 一次性 out-of-place Pull：每个 Rank 主动完成自己的全部 Output，省去
        // Push 协议为确认入站完成而执行的逐 Peer DONE record/wait。
        CCU_CHK_RET(PublishSourceIdentity(ctx));
        CCU_CHK_RET(SubmitDirectLocalCopy(ctx, arg->fixedSliceBytes));
        CCU_CHK_RET(WaitOutputIdentity(ctx));
        CCU_CHK_RET(SubmitDirectPull(ctx, arg->fixedSliceBytes));
        CCU_CHK_RET(WaitDirectPull(ctx));
        return CCU_SUCCESS;
    }
    CCU_CHK_RET(ExchangeOutputIdentity(ctx));
    if (spec.algorithm == AllGatherAlgorithm::FIXED_DIRECT_SMALL) {
        CCU_CHK_RET(RunFixedSmall(ctx, arg->fixedSliceBytes));
    } else {
        CCU_CHK_RET(RunFixedRolling2(ctx, arg->fixedSliceBytes));
    }
    CCU_CHK_RET(PairwiseCompletion(ctx));
    return CCU_SUCCESS;
}

CcuResult RunP2X8SingleDmaKernel(CcuKernelArg kernelArg, AllGatherProfile expectedProfile)
{
    auto *arg = static_cast<CcuKernelArgAllGather *>(kernelArg);
    if (!IsP2X8LargeProfile(expectedProfile)) {
        return CcuResult::CCU_E_PARA;
    }
    CCU_CHK_RET(ValidateProfileArg(arg, expectedProfile, AllGatherDataPath::SINGLE_DMA));
    TransferContext ctx;
    ctx.arg = arg;
    CCU_CHK_RET(InitP2X8SingleDmaResources(ctx));
    CCU_CHK_RET(LoadFixedArgs(ctx));
    CCU_CHK_RET(ExchangeOutputIdentity(ctx));
    CCU_CHK_RET(RunP2X8LargeSingleDma(ctx, arg->fixedSliceBytes));
    CCU_CHK_RET(PairwiseCompletion(ctx));
    return CCU_SUCCESS;
}

CcuResult RunP2X8HybridRelayKernel(CcuKernelArg kernelArg, AllGatherProfile expectedProfile)
{
    auto *arg = static_cast<CcuKernelArgAllGather *>(kernelArg);
    if (!IsP2X8LargeProfile(expectedProfile)) {
        return CcuResult::CCU_E_PARA;
    }
    CCU_CHK_RET(ValidateProfileArg(arg, expectedProfile, AllGatherDataPath::HYBRID_RELAY_CKE));
    TransferContext ctx;
    ctx.arg = arg;
    HybridRankMap rankMap;
    CCU_CHK_RET(InitP2X8HybridResources(ctx, rankMap));
    CCU_CHK_RET(LoadFixedArgs(ctx));
    CCU_CHK_RET(ExchangeOutputIdentity(ctx));

    if (arg->layerRole == AllGatherLayerRole::CLOS) {
        CCU_CHK_RET(RunP2X8HybridClosPhaseA(ctx, rankMap));
        // 全部 Phase-A 入站 Seed 可见后再唤醒本卡 Mesh CCU；Clos 无需反向等待。
        CCU_CHK_RET(PairwiseCompletion(ctx, HYBRID_PHASE_A_DONE_MASK));
        CCU_CHK_RET(ccu::EventRecord(HYBRID_SEED_READY_TAG, HYBRID_CKE_READY_MASK));
        CCU_CHK_RET(RunP2X8HybridClosPhaseB(ctx, rankMap));
    } else {
        // 本地原始 Slice 广播与 Clos Seed 传输并行；仅 relay 读取依赖 Clos Phase A。
        CCU_CHK_RET(RunP2X8LargeSingleDma(ctx, arg->fixedSliceBytes));
        CCU_CHK_RET(ccu::EventWait(HYBRID_SEED_READY_TAG, HYBRID_CKE_READY_MASK));
        CCU_CHK_RET(RunP2X8HybridMeshPhaseB(ctx, rankMap));
    }
    CCU_CHK_RET(PairwiseCompletion(ctx));
    return CCU_SUCCESS;
}

CcuResult RunP2X8HybridRelayPhaseKernel(
    CcuKernelArg kernelArg, AllGatherProfile expectedProfile, bool phaseB)
{
    auto *arg = static_cast<CcuKernelArgAllGather *>(kernelArg);
    if (!IsP2X8LargeProfile(expectedProfile) || (phaseB && arg != nullptr && arg->copyLocalSlice != 0U)) {
        return CcuResult::CCU_E_PARA;
    }
    CCU_CHK_RET(ValidateProfileArg(arg, expectedProfile, AllGatherDataPath::HYBRID_RELAY));
    TransferContext ctx;
    ctx.arg = arg;
    HybridRankMap rankMap;
    CCU_CHK_RET(InitP2X8HybridResources(ctx, rankMap));
    CCU_CHK_RET(LoadFixedArgs(ctx));

    if (!phaseB) {
        // Phase-B Kernel 复用同一 Channel 的地址/Token XN；只在 Phase A 交换一次。
        CCU_CHK_RET(ExchangeOutputIdentity(ctx));
        if (arg->layerRole == AllGatherLayerRole::CLOS) {
            CCU_CHK_RET(RunP2X8HybridClosPhaseA(ctx, rankMap));
            // 远端 Seed 全部可见后，Host 在该 Kernel 后插入 Clos Thread -> Mesh Thread Notify。
            CCU_CHK_RET(PairwiseCompletion(ctx, HYBRID_PHASE_A_DONE_MASK));
        } else {
            CCU_CHK_RET(RunP2X8LargeSingleDma(ctx, arg->fixedSliceBytes));
        }
        return CCU_SUCCESS;
    }

    if (arg->layerRole == AllGatherLayerRole::CLOS) {
        CCU_CHK_RET(RunP2X8HybridClosPhaseB(ctx, rankMap));
    } else {
        CCU_CHK_RET(RunP2X8HybridMeshPhaseB(ctx, rankMap));
    }
    CCU_CHK_RET(PairwiseCompletion(ctx));
    return CCU_SUCCESS;
}

CcuResult RunP2X8DistributedRootPhaseKernel(
    CcuKernelArg kernelArg, AllGatherProfile expectedProfile,
    AllGatherDataPath expectedDataPath, bool phaseB)
{
    auto *arg = static_cast<CcuKernelArgAllGather *>(kernelArg);
    const bool capPeerBandwidth
        = expectedDataPath == AllGatherDataPath::DISTRIBUTED_ROOT_CAPPED;
    const bool matchedSerial
        = expectedDataPath == AllGatherDataPath::DISTRIBUTED_ROOT_MATCHED_SERIAL;
    if (!IsP2X8LargeProfile(expectedProfile)
        || (expectedDataPath != AllGatherDataPath::DISTRIBUTED_ROOT
            && !capPeerBandwidth && !matchedSerial)
        || (phaseB && arg != nullptr && arg->copyLocalSlice != 0U)) {
        return CcuResult::CCU_E_PARA;
    }
    CCU_CHK_RET(ValidateProfileArg(arg, expectedProfile, expectedDataPath));
    TransferContext ctx;
    ctx.arg = arg;
    HybridRankMap rankMap;
    CCU_CHK_RET(InitP2X8HybridResources(ctx, rankMap));
    CCU_CHK_RET(LoadFixedArgs(ctx));

    if (!phaseB) {
        // Phase B 复用本阶段写入的地址/Token XN；四个 Kernel 仍由 Host Thread Notify 编排。
        CCU_CHK_RET(ExchangeOutputIdentity(ctx));
        if (arg->layerRole == AllGatherLayerRole::CLOS) {
            CCU_CHK_RET(RunP2X8DistributedRootClosPhaseA(
                ctx, rankMap, capPeerBandwidth, matchedSerial));
            // Mesh-B 只读取同 slot Root；其余 Clos-A 直达数据由后续 Mesh completion
            // 经各自 Root 形成传递依赖。
            CCU_CHK_RET(PairwiseCompletionChannel(ctx,
                rankMap.channelByPeerSlot[rankMap.selfSlot], HYBRID_PHASE_A_DONE_MASK));
        } else {
            CCU_CHK_RET(RunP2X8LargeSingleDma(ctx, arg->fixedSliceBytes));
        }
        return CCU_SUCCESS;
    }

    if (arg->layerRole == AllGatherLayerRole::CLOS) {
        CCU_CHK_RET(RunP2X8DistributedRootClosPhaseB(
            ctx, rankMap, capPeerBandwidth, matchedSerial));
        // 只等待本阶段实际写入本 Rank 的反向 Peer。Phase-A 入站由 Root-ready
        // 和 Mesh-B 全 Peer completion 形成传递依赖，无需在 Clos 上重复屏障。
        CCU_CHK_RET(PairwiseCompletionP2X8ClosOffsets(
            ctx, rankMap, capPeerBandwidth ? 4U : 5U, matchedSerial));
    } else {
        CCU_CHK_RET(RunP2X8DistributedRootMeshPhaseB(ctx, rankMap));
        CCU_CHK_RET(PairwiseCompletion(ctx));
    }
    return CCU_SUCCESS;
}

CcuResult RunP8P4HybridRelayPhaseKernel(
    CcuKernelArg kernelArg, AllGatherProfile expectedProfile, bool phaseB)
{
    auto *arg = static_cast<CcuKernelArgAllGather *>(kernelArg);
    if (!IsP8P4LargeProfile(expectedProfile) || (phaseB && arg != nullptr
        && arg->copyLocalSlice != 0U) || !ValidateP8P4HybridRouteTable()) {
        return CcuResult::CCU_E_PARA;
    }
    CCU_CHK_RET(ValidateProfileArg(arg, expectedProfile, AllGatherDataPath::HYBRID_RELAY));
    TransferContext ctx;
    ctx.arg = arg;
    P8P4HybridRankMap rankMap;
    CCU_CHK_RET(InitP8P4PhasedResources(ctx, rankMap));
    CCU_CHK_RET(LoadFixedArgs(ctx));

    if (!phaseB) {
        CCU_CHK_RET(ExchangeOutputIdentity(ctx));
        if (arg->layerRole == AllGatherLayerRole::CLOS) {
            CCU_CHK_RET(RunP8P4HybridClosPhaseA(ctx, rankMap));
            // 所有远端 Seed/Direct DMA 完成后建立跨 Rank 可见性，再由 Host 唤醒 Mesh-B。
            CCU_CHK_RET(PairwiseCompletion(ctx, HYBRID_PHASE_A_DONE_MASK));
        } else {
            CCU_CHK_RET(RunP8P4LargeSingleDma(ctx, arg->fixedSliceBytes));
        }
        return CCU_SUCCESS;
    }

    if (arg->layerRole == AllGatherLayerRole::CLOS) {
        CCU_CHK_RET(RunP8P4HybridClosPhaseB(ctx, rankMap));
    } else {
        CCU_CHK_RET(RunP8P4HybridMeshPhaseB(ctx, rankMap));
    }
    CCU_CHK_RET(PairwiseCompletion(ctx));
    return CCU_SUCCESS;
}

CcuResult RunP8P4BidirectionalHalfRootPhaseKernel(CcuKernelArg kernelArg,
    AllGatherProfile expectedProfile, AllGatherDataPath expectedDataPath, bool phaseB)
{
    auto *arg = static_cast<CcuKernelArgAllGather *>(kernelArg);
    const bool capPeerBandwidth
        = expectedDataPath == AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_CAPPED;
    const bool matchedOrdering
        = expectedDataPath == AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_MATCHED;
    if (!IsP8P4LargeProfile(expectedProfile)
        || (expectedDataPath != AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT
            && !capPeerBandwidth && !matchedOrdering)
        || (phaseB && arg != nullptr && arg->copyLocalSlice != 0U)) {
        return CcuResult::CCU_E_PARA;
    }
    CCU_CHK_RET(ValidateProfileArg(arg, expectedProfile, expectedDataPath));
    TransferContext ctx;
    ctx.arg = arg;
    P8P4HybridRankMap rankMap;
    CCU_CHK_RET(InitP8P4PhasedResources(ctx, rankMap));
    CCU_CHK_RET(LoadFixedArgs(ctx));
    P8P4BidirectionalGeometry geometry;
    if (!GetP8P4BidirectionalGeometry(
            arg->fixedSliceBytes, arg->dataTypeSize, geometry)) {
        return CcuResult::CCU_E_PARA;
    }
    if (!ValidateP8P4BidirectionalRoute(
            geometry, arg->fixedSliceBytes, capPeerBandwidth)) {
        return CcuResult::CCU_E_PARA;
    }

    if (!phaseB) {
        CCU_CHK_RET(ExchangeOutputIdentity(ctx));
        if (arg->layerRole == AllGatherLayerRole::CLOS) {
            CCU_CHK_RET(RunP8P4BidirectionalClosPhaseA(
                ctx, rankMap, geometry, capPeerBandwidth, matchedOrdering));
            if (matchedOrdering) {
                // Only U half-roots and Aj relay prefixes feed Phase-B Mesh.
                CCU_CHK_RET(CompleteP8P4PhaseASeeds(ctx, rankMap));
            } else {
                CCU_CHK_RET(PairwiseCompletion(ctx, HYBRID_PHASE_A_DONE_MASK));
            }
        } else {
            CCU_CHK_RET(RunP8P4LargeSingleDma(ctx, arg->fixedSliceBytes));
        }
        return CCU_SUCCESS;
    }

    if (arg->layerRole == AllGatherLayerRole::CLOS) {
        CCU_CHK_RET(RunP8P4BidirectionalClosPhaseB(
            ctx, rankMap, geometry, capPeerBandwidth, matchedOrdering));
        if (matchedOrdering) {
            CCU_CHK_RET(CompleteP8P4ClosReceives(ctx, rankMap));
        } else {
            CCU_CHK_RET(PairwiseCompletion(ctx));
        }
    } else {
        CCU_CHK_RET(RunP8P4BidirectionalMeshPhaseB(ctx, rankMap, geometry));
        CCU_CHK_RET(PairwiseCompletion(ctx));
    }
    return CCU_SUCCESS;
}

CcuResult RunP8P4SharedHalfRootPhaseKernel(CcuKernelArg kernelArg,
    AllGatherProfile expectedProfile, AllGatherDataPath expectedDataPath, bool phaseB)
{
    auto *arg = static_cast<CcuKernelArgAllGather *>(kernelArg);
    if (!IsP8P4LargeProfile(expectedProfile)
        || (phaseB && arg != nullptr && arg->copyLocalSlice != 0U)) {
        return CcuResult::CCU_E_PARA;
    }
    CCU_CHK_RET(ValidateProfileArg(arg, expectedProfile, expectedDataPath));
    TransferContext ctx;
    ctx.arg = arg;
    P8P4HybridRankMap rankMap;
    CCU_CHK_RET(InitP8P4PhasedResources(ctx, rankMap));
    CCU_CHK_RET(LoadFixedArgs(ctx));
    P8P4BidirectionalGeometry geometry;
    if (!GetP8P4SharedBidirectionalGeometry(
            arg->fixedSliceBytes, arg->dataTypeSize, expectedDataPath, geometry)
        || !ValidateP8P4SharedBidirectionalRoute(
            geometry, arg->fixedSliceBytes, arg->dataTypeSize, expectedDataPath)) {
        return CcuResult::CCU_E_PARA;
    }
    const bool fullSeed
        = expectedDataPath == AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_FULL_SEED_4_7;

    if (!phaseB) {
        if (arg->layerRole == AllGatherLayerRole::CLOS) {
            CCU_CHK_RET(ExchangeP8P4SharedClosOutputIdentity(ctx, rankMap));
            CCU_CHK_RET(RunP8P4SharedClosPhaseA(ctx, rankMap, geometry, fullSeed));
            CCU_CHK_RET(CompleteP8P4PhaseASeeds(ctx, rankMap));
        } else {
            CCU_CHK_RET(ExchangeOutputIdentity(ctx));
            CCU_CHK_RET(RunP8P4LargeSingleDma(ctx, arg->fixedSliceBytes));
        }
        return CCU_SUCCESS;
    }

    if (arg->layerRole == AllGatherLayerRole::CLOS) {
        CCU_CHK_RET(RunP8P4SharedClosPhaseB(ctx, rankMap, geometry, fullSeed));
        CCU_CHK_RET(CompleteP8P4ClosReceives(ctx, rankMap));
    } else {
        CCU_CHK_RET(RunP8P4BidirectionalMeshPhaseB(ctx, rankMap, geometry));
        CCU_CHK_RET(PairwiseCompletion(ctx));
    }
    return CCU_SUCCESS;
}

CcuResult RunP8P4SingleDmaKernel(CcuKernelArg kernelArg, AllGatherProfile expectedProfile)
{
    auto *arg = static_cast<CcuKernelArgAllGather *>(kernelArg);
    if (!IsP8P4LargeProfile(expectedProfile)) {
        return CcuResult::CCU_E_PARA;
    }
    CCU_CHK_RET(ValidateProfileArg(arg, expectedProfile, AllGatherDataPath::SINGLE_DMA));
    TransferContext ctx;
    ctx.arg = arg;
    CCU_CHK_RET(InitP8P4SingleDmaResources(ctx));
    CCU_CHK_RET(LoadFixedArgs(ctx));
    CCU_CHK_RET(ExchangeOutputIdentity(ctx));
    CCU_CHK_RET(RunP8P4LargeSingleDma(ctx, arg->fixedSliceBytes));
    CCU_CHK_RET(PairwiseCompletion(ctx));
    return CCU_SUCCESS;
}

} // namespace

CcuResult CcuAllGatherDirectKernel(CcuKernelArg kernelArg)
{
    auto *arg = static_cast<CcuKernelArgAllGather *>(kernelArg);
    if (arg == nullptr || arg->profile != AllGatherProfile::GENERIC
        || arg->dataPath != AllGatherDataPath::GENERIC_DIRECT
        || (arg->layerRole != AllGatherLayerRole::CLOS
            && arg->layerRole != AllGatherLayerRole::LOCAL_MESH)
        || arg->fixedSliceBytes != 0U || arg->cellBytes != 0U) {
        return CcuResult::CCU_E_PARA;
    }
    TransferContext ctx;
    ctx.arg = arg;
    CCU_CHK_RET(InitTransferResources(ctx));

    // 通用图的七个动态参数依次为源/目标地址、源/目标 Token、本 Rank Offset、块 Offset 和块长度。
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.sourceAddr, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputAddr, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.sourceToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.outputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localOffset, argId++));
    ccu::Variable chunkOffset;
    ccu::Variable chunkBytes;
    CCU_CHK_RET(ccu::LoadArg(chunkOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(chunkBytes, argId++));

    CCU_CHK_RET(ExchangeOutputIdentity(ctx));
    CCU_CHK_RET(RunOneChunk(ctx, chunkOffset, chunkBytes));
    CCU_CHK_RET(PairwiseCompletion(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuAllGatherP2X8_512KKernel(CcuKernelArg arg)
{
    return RunProfileKernel(arg, AllGatherProfile::P2X8_512K);
}

CcuResult CcuAllGatherP2X8_512MKernel(CcuKernelArg arg)
{
    return RunProfileKernel(arg, AllGatherProfile::P2X8_512M);
}

CcuResult CcuAllGatherP2X8_400M4BKernel(CcuKernelArg arg)
{
    return RunProfileKernel(arg, AllGatherProfile::P2X8_400M4B);
}

CcuResult CcuAllGatherP2X8_512MSingleDmaKernel(CcuKernelArg kernelArg)
{
    return RunP2X8SingleDmaKernel(kernelArg, AllGatherProfile::P2X8_512M);
}

CcuResult CcuAllGatherP2X8_400M4BSingleDmaKernel(CcuKernelArg kernelArg)
{
    return RunP2X8SingleDmaKernel(kernelArg, AllGatherProfile::P2X8_400M4B);
}

CcuResult CcuAllGatherP2X8_512MHybridRelayKernel(CcuKernelArg kernelArg)
{
    return RunP2X8HybridRelayKernel(kernelArg, AllGatherProfile::P2X8_512M);
}

CcuResult CcuAllGatherP2X8_400M4BHybridRelayKernel(CcuKernelArg kernelArg)
{
    return RunP2X8HybridRelayKernel(kernelArg, AllGatherProfile::P2X8_400M4B);
}

CcuResult CcuAllGatherP2X8_512MHybridPhaseAKernel(CcuKernelArg kernelArg)
{
    return RunP2X8HybridRelayPhaseKernel(kernelArg, AllGatherProfile::P2X8_512M, false);
}

CcuResult CcuAllGatherP2X8_512MHybridPhaseBKernel(CcuKernelArg kernelArg)
{
    return RunP2X8HybridRelayPhaseKernel(kernelArg, AllGatherProfile::P2X8_512M, true);
}

CcuResult CcuAllGatherP2X8_400M4BHybridPhaseAKernel(CcuKernelArg kernelArg)
{
    return RunP2X8HybridRelayPhaseKernel(kernelArg, AllGatherProfile::P2X8_400M4B, false);
}

CcuResult CcuAllGatherP2X8_400M4BHybridPhaseBKernel(CcuKernelArg kernelArg)
{
    return RunP2X8HybridRelayPhaseKernel(kernelArg, AllGatherProfile::P2X8_400M4B, true);
}

CcuResult CcuAllGatherP2X8_512MDistributedRootPhaseAKernel(CcuKernelArg kernelArg)
{
    return RunP2X8DistributedRootPhaseKernel(
        kernelArg, AllGatherProfile::P2X8_512M,
        AllGatherDataPath::DISTRIBUTED_ROOT, false);
}

CcuResult CcuAllGatherP2X8_512MDistributedRootPhaseBKernel(CcuKernelArg kernelArg)
{
    return RunP2X8DistributedRootPhaseKernel(
        kernelArg, AllGatherProfile::P2X8_512M,
        AllGatherDataPath::DISTRIBUTED_ROOT, true);
}

CcuResult CcuAllGatherP2X8_512MDistributedRootCappedPhaseAKernel(CcuKernelArg kernelArg)
{
    return RunP2X8DistributedRootPhaseKernel(
        kernelArg, AllGatherProfile::P2X8_512M,
        AllGatherDataPath::DISTRIBUTED_ROOT_CAPPED, false);
}

CcuResult CcuAllGatherP2X8_512MDistributedRootCappedPhaseBKernel(CcuKernelArg kernelArg)
{
    return RunP2X8DistributedRootPhaseKernel(
        kernelArg, AllGatherProfile::P2X8_512M,
        AllGatherDataPath::DISTRIBUTED_ROOT_CAPPED, true);
}

CcuResult CcuAllGatherP2X8_512MDistributedRootMatchedSerialPhaseAKernel(
    CcuKernelArg kernelArg)
{
    return RunP2X8DistributedRootPhaseKernel(
        kernelArg, AllGatherProfile::P2X8_512M,
        AllGatherDataPath::DISTRIBUTED_ROOT_MATCHED_SERIAL, false);
}

CcuResult CcuAllGatherP2X8_512MDistributedRootMatchedSerialPhaseBKernel(
    CcuKernelArg kernelArg)
{
    return RunP2X8DistributedRootPhaseKernel(
        kernelArg, AllGatherProfile::P2X8_512M,
        AllGatherDataPath::DISTRIBUTED_ROOT_MATCHED_SERIAL, true);
}

CcuResult CcuAllGatherP2X8_400M4BDistributedRootPhaseAKernel(CcuKernelArg kernelArg)
{
    return RunP2X8DistributedRootPhaseKernel(
        kernelArg, AllGatherProfile::P2X8_400M4B,
        AllGatherDataPath::DISTRIBUTED_ROOT, false);
}

CcuResult CcuAllGatherP2X8_400M4BDistributedRootPhaseBKernel(CcuKernelArg kernelArg)
{
    return RunP2X8DistributedRootPhaseKernel(
        kernelArg, AllGatherProfile::P2X8_400M4B,
        AllGatherDataPath::DISTRIBUTED_ROOT, true);
}

CcuResult CcuAllGatherP2X8_400M4BDistributedRootCappedPhaseAKernel(CcuKernelArg kernelArg)
{
    return RunP2X8DistributedRootPhaseKernel(
        kernelArg, AllGatherProfile::P2X8_400M4B,
        AllGatherDataPath::DISTRIBUTED_ROOT_CAPPED, false);
}

CcuResult CcuAllGatherP2X8_400M4BDistributedRootCappedPhaseBKernel(CcuKernelArg kernelArg)
{
    return RunP2X8DistributedRootPhaseKernel(
        kernelArg, AllGatherProfile::P2X8_400M4B,
        AllGatherDataPath::DISTRIBUTED_ROOT_CAPPED, true);
}

CcuResult CcuAllGatherP2X8_400M4BDistributedRootMatchedSerialPhaseAKernel(
    CcuKernelArg kernelArg)
{
    return RunP2X8DistributedRootPhaseKernel(
        kernelArg, AllGatherProfile::P2X8_400M4B,
        AllGatherDataPath::DISTRIBUTED_ROOT_MATCHED_SERIAL, false);
}

CcuResult CcuAllGatherP2X8_400M4BDistributedRootMatchedSerialPhaseBKernel(
    CcuKernelArg kernelArg)
{
    return RunP2X8DistributedRootPhaseKernel(
        kernelArg, AllGatherProfile::P2X8_400M4B,
        AllGatherDataPath::DISTRIBUTED_ROOT_MATCHED_SERIAL, true);
}

CcuResult CcuAllGatherP4X1_512KKernel(CcuKernelArg arg)
{
    return RunProfileKernel(arg, AllGatherProfile::P4X1_512K);
}

CcuResult CcuAllGatherP4X1_512MKernel(CcuKernelArg arg)
{
    return RunProfileKernel(arg, AllGatherProfile::P4X1_512M);
}

CcuResult CcuAllGatherP4X1_400M4BKernel(CcuKernelArg arg)
{
    return RunProfileKernel(arg, AllGatherProfile::P4X1_400M4B);
}

CcuResult CcuAllGatherP4X1LargeSingleDmaKernel(CcuKernelArg kernelArg)
{
    auto *arg = static_cast<CcuKernelArgAllGather *>(kernelArg);
    if (arg == nullptr || !IsP4X1LargeProfile(arg->profile)) {
        return CcuResult::CCU_E_PARA;
    }
    // 复用固定 Profile 对 4 Rank、Clos 层、dtype 和实际 Slice 的校验，不进入 Rolling-2 分派。
    CCU_CHK_RET(ValidateProfileArg(arg, arg->profile, AllGatherDataPath::SINGLE_DMA));
    TransferContext ctx;
    ctx.arg = arg;
    CCU_CHK_RET(InitP4X1SingleDmaResources(ctx));
    CCU_CHK_RET(LoadFixedArgs(ctx));
    CCU_CHK_RET(ExchangeOutputIdentity(ctx));
    CCU_CHK_RET(RunP4X1LargeSingleDma(ctx, arg->fixedSliceBytes));
    CCU_CHK_RET(PairwiseCompletion(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuAllGatherP4X1LargeMatchedReadyKernel(CcuKernelArg kernelArg)
{
    auto *arg = static_cast<CcuKernelArgAllGather *>(kernelArg);
    if (arg == nullptr || !IsP4X1LargeProfile(arg->profile)) {
        return CcuResult::CCU_E_PARA;
    }
    CCU_CHK_RET(ValidateProfileArg(arg, arg->profile, AllGatherDataPath::MATCHED_READY));
    TransferContext ctx;
    ctx.arg = arg;
    CCU_CHK_RET(InitP4X1SingleDmaResources(ctx));
    CCU_CHK_RET(LoadFixedArgs(ctx));
    if (arg->sourceMode == AllGatherSourceMode::DIRECT_INPUT) {
        // 一次性大消息 Pull：本 Rank 等待自己的三个 Read 完成即可确认 Output 完整，
        // 不再执行 Push 路径的逐 Peer DONE record/wait。网络 Read 先于 LocalCopy 提交。
        CCU_CHK_RET(PublishSourceIdentity(ctx));
        CCU_CHK_RET(WaitOutputIdentity(ctx));
        CCU_CHK_RET(SubmitDirectPull(ctx, arg->fixedSliceBytes));
        CCU_CHK_RET(SubmitDirectLocalCopy(ctx, arg->fixedSliceBytes));
        CCU_CHK_RET(WaitDirectPull(ctx));
        return CCU_SUCCESS;
    }
    CCU_CHK_RET(PublishOutputIdentity(ctx));
    CCU_CHK_RET(RunP4X1MatchedReadySingleDma(ctx, arg->fixedSliceBytes));
    CCU_CHK_RET(PairwiseCompletion(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuAllGatherP4X1LargeMatchedSerialRolling2Kernel(CcuKernelArg kernelArg)
{
    auto *arg = static_cast<CcuKernelArgAllGather *>(kernelArg);
    if (arg == nullptr || !IsP4X1LargeProfile(arg->profile)) {
        return CcuResult::CCU_E_PARA;
    }
    CCU_CHK_RET(ValidateProfileArg(
        arg, arg->profile, AllGatherDataPath::MATCHED_SERIAL_ROLLING2));
    TransferContext ctx;
    ctx.arg = arg;
    CCU_CHK_RET(InitP4X1SingleDmaResources(ctx));
    ctx.peerEvents.resize(2U);
    CCU_CHK_RET(LoadFixedArgs(ctx));
    CCU_CHK_RET(PublishOutputIdentity(ctx));
    CCU_CHK_RET(RunP4X1MatchedSerialRolling2(ctx, arg->fixedSliceBytes));
    CCU_CHK_RET(PairwiseCompletion(ctx));
    return CCU_SUCCESS;
}

CcuResult CcuAllGatherP8P4_512KKernel(CcuKernelArg arg)
{
    return RunProfileKernel(arg, AllGatherProfile::P8P4_512K);
}

CcuResult CcuAllGatherP8P4_512MKernel(CcuKernelArg arg)
{
    return RunProfileKernel(arg, AllGatherProfile::P8P4_512M);
}

CcuResult CcuAllGatherP8P4_400M4BKernel(CcuKernelArg arg)
{
    return RunProfileKernel(arg, AllGatherProfile::P8P4_400M4B);
}

CcuResult CcuAllGatherP8P4_512MSingleDmaKernel(CcuKernelArg kernelArg)
{
    return RunP8P4SingleDmaKernel(kernelArg, AllGatherProfile::P8P4_512M);
}

CcuResult CcuAllGatherP8P4_400M4BSingleDmaKernel(CcuKernelArg kernelArg)
{
    return RunP8P4SingleDmaKernel(kernelArg, AllGatherProfile::P8P4_400M4B);
}

CcuResult CcuAllGatherP8P4_512MHybridPhaseAKernel(CcuKernelArg kernelArg)
{
    return RunP8P4HybridRelayPhaseKernel(kernelArg, AllGatherProfile::P8P4_512M, false);
}

CcuResult CcuAllGatherP8P4_512MHybridPhaseBKernel(CcuKernelArg kernelArg)
{
    return RunP8P4HybridRelayPhaseKernel(kernelArg, AllGatherProfile::P8P4_512M, true);
}

CcuResult CcuAllGatherP8P4_400M4BHybridPhaseAKernel(CcuKernelArg kernelArg)
{
    return RunP8P4HybridRelayPhaseKernel(kernelArg, AllGatherProfile::P8P4_400M4B, false);
}

CcuResult CcuAllGatherP8P4_400M4BHybridPhaseBKernel(CcuKernelArg kernelArg)
{
    return RunP8P4HybridRelayPhaseKernel(kernelArg, AllGatherProfile::P8P4_400M4B, true);
}

CcuResult CcuAllGatherP8P4_512MBidirectionalHalfRootPhaseAKernel(CcuKernelArg kernelArg)
{
    return RunP8P4BidirectionalHalfRootPhaseKernel(kernelArg,
        AllGatherProfile::P8P4_512M, AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT, false);
}

CcuResult CcuAllGatherP8P4_512MBidirectionalHalfRootPhaseBKernel(CcuKernelArg kernelArg)
{
    return RunP8P4BidirectionalHalfRootPhaseKernel(kernelArg,
        AllGatherProfile::P8P4_512M, AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT, true);
}

CcuResult CcuAllGatherP8P4_512MBidirectionalHalfRootCappedPhaseAKernel(CcuKernelArg kernelArg)
{
    return RunP8P4BidirectionalHalfRootPhaseKernel(kernelArg,
        AllGatherProfile::P8P4_512M,
        AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_CAPPED, false);
}

CcuResult CcuAllGatherP8P4_512MBidirectionalHalfRootCappedPhaseBKernel(CcuKernelArg kernelArg)
{
    return RunP8P4BidirectionalHalfRootPhaseKernel(kernelArg,
        AllGatherProfile::P8P4_512M,
        AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_CAPPED, true);
}

CcuResult CcuAllGatherP8P4_512MBidirectionalHalfRootMatchedPhaseAKernel(CcuKernelArg kernelArg)
{
    return RunP8P4BidirectionalHalfRootPhaseKernel(kernelArg,
        AllGatherProfile::P8P4_512M,
        AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_MATCHED, false);
}

CcuResult CcuAllGatherP8P4_512MBidirectionalHalfRootMatchedPhaseBKernel(CcuKernelArg kernelArg)
{
    return RunP8P4BidirectionalHalfRootPhaseKernel(kernelArg,
        AllGatherProfile::P8P4_512M,
        AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_MATCHED, true);
}

CcuResult CcuAllGatherP8P4_512MBidirectionalHalfRootSharedPhaseAKernel(CcuKernelArg kernelArg)
{
    return RunP8P4SharedHalfRootPhaseKernel(
        kernelArg, AllGatherProfile::P8P4_512M,
        AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_SHARED, false);
}

CcuResult CcuAllGatherP8P4_512MBidirectionalHalfRootSharedPhaseBKernel(CcuKernelArg kernelArg)
{
    return RunP8P4SharedHalfRootPhaseKernel(
        kernelArg, AllGatherProfile::P8P4_512M,
        AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_SHARED, true);
}

CcuResult CcuAllGatherP8P4_512MBidirectionalHalfRootShared5Over8PhaseAKernel(
    CcuKernelArg kernelArg)
{
    return RunP8P4SharedHalfRootPhaseKernel(
        kernelArg, AllGatherProfile::P8P4_512M,
        AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_SHARED_5_8, false);
}

CcuResult CcuAllGatherP8P4_512MBidirectionalHalfRootShared5Over8PhaseBKernel(
    CcuKernelArg kernelArg)
{
    return RunP8P4SharedHalfRootPhaseKernel(
        kernelArg, AllGatherProfile::P8P4_512M,
        AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_SHARED_5_8, true);
}

CcuResult CcuAllGatherP8P4_512MBidirectionalHalfRootFullSeed4Over7PhaseAKernel(
    CcuKernelArg kernelArg)
{
    return RunP8P4SharedHalfRootPhaseKernel(
        kernelArg, AllGatherProfile::P8P4_512M,
        AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_FULL_SEED_4_7, false);
}

CcuResult CcuAllGatherP8P4_512MBidirectionalHalfRootFullSeed4Over7PhaseBKernel(
    CcuKernelArg kernelArg)
{
    return RunP8P4SharedHalfRootPhaseKernel(
        kernelArg, AllGatherProfile::P8P4_512M,
        AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_FULL_SEED_4_7, true);
}

CcuResult CcuAllGatherP8P4_400M4BBidirectionalHalfRootPhaseAKernel(CcuKernelArg kernelArg)
{
    return RunP8P4BidirectionalHalfRootPhaseKernel(kernelArg,
        AllGatherProfile::P8P4_400M4B, AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT, false);
}

CcuResult CcuAllGatherP8P4_400M4BBidirectionalHalfRootPhaseBKernel(CcuKernelArg kernelArg)
{
    return RunP8P4BidirectionalHalfRootPhaseKernel(kernelArg,
        AllGatherProfile::P8P4_400M4B, AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT, true);
}

CcuResult CcuAllGatherP8P4_400M4BBidirectionalHalfRootCappedPhaseAKernel(CcuKernelArg kernelArg)
{
    return RunP8P4BidirectionalHalfRootPhaseKernel(kernelArg,
        AllGatherProfile::P8P4_400M4B,
        AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_CAPPED, false);
}

CcuResult CcuAllGatherP8P4_400M4BBidirectionalHalfRootCappedPhaseBKernel(CcuKernelArg kernelArg)
{
    return RunP8P4BidirectionalHalfRootPhaseKernel(kernelArg,
        AllGatherProfile::P8P4_400M4B,
        AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_CAPPED, true);
}

CcuResult CcuAllGatherP8P4_400M4BBidirectionalHalfRootSharedPhaseAKernel(CcuKernelArg kernelArg)
{
    return RunP8P4SharedHalfRootPhaseKernel(
        kernelArg, AllGatherProfile::P8P4_400M4B,
        AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_SHARED, false);
}

CcuResult CcuAllGatherP8P4_400M4BBidirectionalHalfRootSharedPhaseBKernel(CcuKernelArg kernelArg)
{
    return RunP8P4SharedHalfRootPhaseKernel(
        kernelArg, AllGatherProfile::P8P4_400M4B,
        AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_SHARED, true);
}

CcuResult CcuAllGatherP8P4_400M4BBidirectionalHalfRootShared5Over8PhaseAKernel(
    CcuKernelArg kernelArg)
{
    return RunP8P4SharedHalfRootPhaseKernel(
        kernelArg, AllGatherProfile::P8P4_400M4B,
        AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_SHARED_5_8, false);
}

CcuResult CcuAllGatherP8P4_400M4BBidirectionalHalfRootShared5Over8PhaseBKernel(
    CcuKernelArg kernelArg)
{
    return RunP8P4SharedHalfRootPhaseKernel(
        kernelArg, AllGatherProfile::P8P4_400M4B,
        AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_SHARED_5_8, true);
}

CcuResult CcuAllGatherP8P4_400M4BBidirectionalHalfRootFullSeed4Over7PhaseAKernel(
    CcuKernelArg kernelArg)
{
    return RunP8P4SharedHalfRootPhaseKernel(
        kernelArg, AllGatherProfile::P8P4_400M4B,
        AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_FULL_SEED_4_7, false);
}

CcuResult CcuAllGatherP8P4_400M4BBidirectionalHalfRootFullSeed4Over7PhaseBKernel(
    CcuKernelArg kernelArg)
{
    return RunP8P4SharedHalfRootPhaseKernel(
        kernelArg, AllGatherProfile::P8P4_400M4B,
        AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_FULL_SEED_4_7, true);
}

CcuResult CcuAllGatherPrepareKernel(CcuKernelArg kernelArg)
{
    const auto *arg = static_cast<CcuKernelArgPrepare *>(kernelArg);
    if (arg == nullptr || arg->channelCount != 0U) {
        return CcuResult::CCU_E_PARA;
    }

    ccu::Variable inputAddr;
    ccu::Variable canonicalAddr;
    ccu::Variable scratchAddr;
    ccu::Variable inputToken;
    ccu::Variable outputToken;
    ccu::Variable scratchToken;
    ccu::Variable chunkOffset;
    ccu::Variable chunkBytes;
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(inputAddr, argId++));
    CCU_CHK_RET(ccu::LoadArg(canonicalAddr, argId++));
    CCU_CHK_RET(ccu::LoadArg(scratchAddr, argId++));
    CCU_CHK_RET(ccu::LoadArg(inputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(outputToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(scratchToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(chunkOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(chunkBytes, argId++));

    ccu::LocalAddr input;
    input.addr = inputAddr;
    input.addr += chunkOffset;
    input.token = inputToken;
    ccu::LocalAddr scratch;
    scratch.addr = scratchAddr;
    scratch.token = scratchToken;
    ccu::LocalAddr canonical;
    canonical.addr = canonicalAddr;
    canonical.addr += chunkOffset;
    canonical.token = outputToken;
    ccu::Event copyEvent;

    // 每块严格等待 input -> scratch 完成后再执行 scratch -> canonical，第二次等待结束前
    // Host 不会下发下一块，因此 bounce buffer 不会被过早覆盖。
    CCU_CHK_RET(ccu::LocalCopy(scratch, input, chunkBytes, copyEvent, static_cast<uint16_t>(1U)));
    CCU_CHK_RET(ccu::EventWait(copyEvent, static_cast<uint16_t>(1U)));
    CCU_CHK_RET(ccu::LocalCopy(canonical, scratch, chunkBytes, copyEvent, static_cast<uint16_t>(1U)));
    CCU_CHK_RET(ccu::EventWait(copyEvent, static_cast<uint16_t>(1U)));
    return CCU_SUCCESS;
}

} // namespace ops_hccl
