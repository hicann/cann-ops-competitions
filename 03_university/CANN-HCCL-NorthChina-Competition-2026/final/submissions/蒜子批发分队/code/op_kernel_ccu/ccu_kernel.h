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

#include "custom.h"

namespace ops_hccl {

constexpr uint32_t NHR_MAX_STEP_COUNT = 4;
constexpr uint32_t NHR_MAX_STEP_SLICE_COUNT = 6;

struct CcuKernelArgAllGather : public CcuKernelArgBase {
    uint32_t rankSize = 0;
    uint32_t rankId = 0;
    uint32_t remoteRanks[MAX_RANK_SIZE] = {};
    uint8_t channelKinds[MAX_RANK_SIZE] = {};
    bool copyLocal = false;
    bool useGroupBroadcast = false;
    bool directMeshFirst = false;
    bool fuseTwoSlices = false;
    bool useFineLocalCopy = false;
    bool useDirectLocalCopy = false;
    bool useRank4DirectReuse = false;
    bool useRank12DirectReuse = false;
    bool useRank12Nhr = false;
    uint32_t nhrAxisId = 0;
    uint32_t nhrStepCount = 0;
    uint32_t nhrToChannelIndices[NHR_MAX_STEP_COUNT] = {};
    uint32_t nhrFromChannelIndices[NHR_MAX_STEP_COUNT] = {};
    uint32_t nhrStepSliceCounts[NHR_MAX_STEP_COUNT] = {};
    uint32_t nhrStepSliceIndices[NHR_MAX_STEP_COUNT]
        [NHR_MAX_STEP_SLICE_COUNT] = {};
    bool useRank12DualRail = false;
    bool alternateRailChannels[MAX_RANK_SIZE] = {};
    bool useRank12SplitRelay = false;
    bool isRank12MeshKernel = false;
    bool rank12SmallServer = false;
    uint32_t rank12BridgeTargetChannelIndex = MAX_RANK_SIZE;
    uint32_t rank12FanoutSourceChannelIndices[2] = {
        MAX_RANK_SIZE,
        MAX_RANK_SIZE,
    };
    uint32_t rank12FanoutSourceCount = 0;
    bool useRank16MixedRelay = false;
    bool isRank16MeshKernel = false;
    uint32_t relayChannelIndex = MAX_RANK_SIZE;
    uint32_t relayPeerRank = MAX_RANK_SIZE;
    uint64_t rank4FirstOutputOffset = 0;
    uint64_t rank12FirstOutputOffset = 0;
};

// CCU Kernel 函数
CcuResult CcuKernel(CcuKernelArg arg);
} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
