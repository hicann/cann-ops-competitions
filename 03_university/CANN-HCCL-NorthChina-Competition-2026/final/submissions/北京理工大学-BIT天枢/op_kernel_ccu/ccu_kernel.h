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

#include <vector>

#include <ccu/ccu_event.hpp>
#include <ccu/ccu_primitives.hpp>
#include <ccu/ccu_types.h>
#include <ccu/ccu_variable.hpp>

#include "custom.h"

namespace ops_hccl {

namespace ccu = ::AscendC::ccu;

struct CcuPeerLanePullContext {
    const CcuDirectKernelArg *arg = nullptr;
    std::vector<ccu::Variable> inputAddrs;
    std::vector<ccu::Variable> inputTokens;
    ccu::Variable outputToken;
    std::vector<ccu::Variable> destinationAddrs;
    ccu::Variable firstSliceSize;
    ccu::Variable secondSliceSize;
    ccu::Event firstEvent;
    ccu::Event secondEvent;
};

// 每个Die仅注册一个Receiver-Owned Pull Kernel，动态参数不超过13项。
CcuResult CcuPeerLanePullKernel(CcuKernelArg arg);

// 4x1专用以及8+4机内阶段：一个Kernel内完成轮转Direct Push和OwnCopy。
// 动态参数固定为7项；4x1双Slice最多使用6个网络Event位，8+4机内单Slice最多7位。
CcuResult CcuRotatingDirectPushKernel(CcuKernelArg arg);
// Bandwidth-only variant imported for the teammate 8+4 large-message path.
CcuResult CcuWideRotatingDirectPushKernel(CcuKernelArg arg);

// 512KB专用微内核：固定单Slice，网络Write全部提交后再下发OwnCopy，
// 地址到达、数据完成和PostSync均按Channel流水；动态参数仅6项。
CcuResult CcuLatencyDirectPushKernel(CcuKernelArg arg);
CcuResult CcuRegistered4x1DirectPushKernel(CcuKernelArg arg);
CcuResult CcuOutput512StaticDirectPushKernel(CcuKernelArg arg);

// 4x1公共CLOS Die专用：固定三Channel；地址就绪后立即按Peer流水下发，
// latency静态实例为1 Chunk并融合OwnCopy，bandwidth实例为4 Chunk。
CcuResult CcuClos4x1DirectPushKernel(CcuKernelArg arg);

// 8+4跨Server阶段：小Server源执行5/9全扇出和两个2/9 Seed条带；
// 大Server源执行5/7全扇出和一个2/7 Seed条带。同一静态Kernel由动态phaseMode
// 分别执行Seed-only与Direct-only，固定11项参数；单Phase最多8个Event位。
CcuResult CcuDualSeedCrossKernel(CcuKernelArg arg);

// V30D bandwidth-only 8+4 Seed/Direct split kernels.
CcuResult CcuWideDualSeedCrossKernel(CcuKernelArg arg);

// 8+4双向Relay：大Server Rank广播一个2/9条带；小Server Rank广播
// 两个大源的2/7尾条带。动态参数固定为9项，Event最多7位。
CcuResult CcuDualSeedRelayKernel(CcuKernelArg arg);
// Bandwidth-only relay variant; V27 latency and 4x1 code keep the original.
CcuResult CcuWideDualSeedRelayKernel(CcuKernelArg arg);

// V22 rank-16 (2x8) bandwidth-balanced path. These static arguments are
// deliberately isolated from the V10A 4x1 and 8+4 kernel layouts.
constexpr uint32_t V22_MAX_BATCH_PLAN_SIZE = 7;

struct V22CcuKernelArgAllGatherGroup : public CcuKernelArgBase {
    bool handleLocalCopy;
    bool exchangeResources;
    bool finalSync;
    // V27 exact-512 path.  When true, the data Event is the completion
    // boundary and the per-peer FullSync tail is omitted.
    bool latencyPipeline;
    // Bandwidth-only controls imported from the teammate implementation.
    bool progressiveAddressReady;
    bool rotateSecondSlice;
    bool combineSliceWaits;
};

enum class V22CcuPhaseKind : uint32_t {
    INTRA_RELAY_BATCH = 0,
    CROSS_DIRECT_BATCH = 1,
};

struct V22CcuTransferPlan {
    uint32_t chunkIndex;
    uint32_t patternOffset;
    uint32_t missingIndex;
    uint32_t relayOwnerOffset;
    uint32_t sourceIndex;
};

struct V22CcuKernelArgAllGatherPhase : public CcuKernelArgBase {
    uint32_t rankId;
    uint32_t peerRanks[MAX_RANK_SIZE];
    V22CcuPhaseKind phaseKind;
    uint32_t patternSize;
    uint32_t incomingPatternSize;
    uint32_t seedCount;
    uint32_t incomingSeedCount;
    uint32_t chunkCount;
    uint32_t operationCount;
    V22CcuTransferPlan plans[V22_MAX_BATCH_PLAN_SIZE];
    bool exchangeResources;
    bool seedRecord;
    bool seedWait;
    bool finalSync;
    bool progressiveAddressReady;
    bool rotateSecondChunk;
};

CcuResult CcuV22AllGatherGroupKernel(CcuKernelArg arg);
CcuResult CcuV22AllGatherPhaseKernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
