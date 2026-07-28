/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CCU_KERNEL_H
#define OPS_HCCL_CCU_KERNEL_H

#include <ccu/ccu_types.h>

#include <array>
#include <type_traits>

#include "custom.h"

namespace ops_hccl {

struct Direct12DieStaticArg {
    uint32_t canonicalRank = INVALID_VALUE_RANKID;
    uint32_t canonicalGroup = INVALID_VALUE_RANKID;
    uint32_t canonicalGroupRank = INVALID_VALUE_RANKID;
    uint32_t copyLocalInput = 0;
    uint32_t mode = DIRECT12_MODE_STANDARD;
    uint64_t scratchStrideBytes = 0;
    std::array<ChannelHandle, MAX_PEER_CHANNELS> l0Channels{};
    std::array<uint32_t, MAX_PEER_CHANNELS> l0PeerCanonicalRanks{};
    uint32_t l0ChannelCount = 0;
    std::array<ChannelHandle, MAX_PEER_CHANNELS> l1Channels{};
    std::array<uint32_t, MAX_PEER_CHANNELS> l1PeerCanonicalRanks{};
    uint32_t l1ChannelCount = 0;
};

static_assert(std::is_trivially_copyable<Direct12DieStaticArg>::value);

CcuResult Direct12DieKernel(CcuKernelArg arg);
CcuResult Direct12PrefixScalarDieKernel(CcuKernelArg arg);

struct FullExchangeDieStaticArg {
    uint32_t canonicalRank = INVALID_VALUE_RANKID;
    uint32_t canonicalGroup = INVALID_VALUE_RANKID;
    FabricLayer layer = FabricLayer::INVALID;
    uint32_t copyLocalInput = 0;
    std::array<ChannelHandle, MAX_PEER_CHANNELS> channels{};
    std::array<uint32_t, MAX_PEER_CHANNELS> peerCanonicalRanks{};
    uint32_t channelCount = 0;
};

static_assert(std::is_trivially_copyable<FullExchangeDieStaticArg>::value);

CcuResult FullExchangeDieKernel(CcuKernelArg arg);

struct FourXOneRingStaticArg {
    uint32_t canonicalRank = INVALID_VALUE_RANKID;
    std::array<ChannelHandle, 2> channels{};
    std::array<uint32_t, 2> peerCanonicalRanks{};
    uint32_t channelCount = 0;
};

static_assert(std::is_trivially_copyable<FourXOneRingStaticArg>::value);

CcuResult FourXOneRingKernel(CcuKernelArg arg);

struct FourXOnePullStaticArg {
    uint32_t canonicalRank = INVALID_VALUE_RANKID;
    uint32_t worker = 0;
    uint32_t hasScalarTail = 0;
    std::array<ChannelHandle, FOUR_X_ONE_PEER_COUNT> channels{};
    std::array<uint32_t, FOUR_X_ONE_PEER_COUNT> peerCanonicalRanks{};
    uint32_t channelCount = 0;
};

static_assert(std::is_trivially_copyable<FourXOnePullStaticArg>::value);

CcuResult FourXOnePullFullExchangeKernel(CcuKernelArg arg);

struct TwoXEightMultiRootStaticArg {
    uint32_t canonicalRank = INVALID_VALUE_RANKID;
    FabricLayer layer = FabricLayer::INVALID;
    std::array<ChannelHandle, 8> channels{};
    std::array<uint32_t, 8> peerCanonicalRanks{};
    uint32_t channelCount = 0;
};

static_assert(std::is_trivially_copyable<TwoXEightMultiRootStaticArg>::value);

CcuResult TwoXEightMultiRootKernel(CcuKernelArg arg);

using TwoXEightBulkStaticArg = TwoXEightMultiRootStaticArg;
static_assert(std::is_trivially_copyable<TwoXEightBulkStaticArg>::value);

CcuResult TwoXEightBulkKernel(CcuKernelArg arg);
} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
