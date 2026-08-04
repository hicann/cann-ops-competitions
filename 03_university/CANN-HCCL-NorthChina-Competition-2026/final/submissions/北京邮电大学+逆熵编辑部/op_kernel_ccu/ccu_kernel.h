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

struct CcuKernelArgDirect : public CcuKernelArgBase {
    uint32_t layerId = 0;
    uint32_t localRank = 0;
    uint32_t pairChannelCount = 0;
    uint32_t relayChannelCount = 0;
    uint32_t relaySourceCount = 0;
    uint32_t relaySourceRanks0[MAX_RANK_SIZE]{};
    uint32_t relaySourceRanks1[MAX_RANK_SIZE]{};
    uint32_t asymmetricPairRound0Channel = MAX_RANK_SIZE;
    uint32_t asymmetricPairRound1Channel = MAX_RANK_SIZE;
    bool asymmetricPairLargeSide = false;
    bool asymmetricLayer0 = false;
};

// Each IO Die owns one kernel. Translation-time subset counts specialize
// layer-0 relay and layer-1 pair stages without sharing a network device.
CcuResult CcuStagedLayer0(CcuKernelArg arg);
CcuResult CcuStagedLayer1(CcuKernelArg arg);

// Pure layer-1, four-rank kernel for the formal 512 KiB output case. Keeping
// it separate prevents its instruction footprint from affecting the staged
// kernels used by every large-message and mixed-topology path.
CcuResult CcuSmall4x1DirectPull(CcuKernelArg arg);

// Dedicated, stage-free 2x8 small-message kernels. Layer-1 owns the only
// LocalCopy; every layer/Die registration receives only its own Channels.
CcuResult CcuSmall2x8Layer0DirectPull(CcuKernelArg arg);
CcuResult CcuSmall2x8Layer1DirectPull(CcuKernelArg arg);

// Dedicated two-Die direct-Pull kernels for the formal 8+4 large cases.
CcuResult CcuLarge8Plus4Layer0DirectPull(CcuKernelArg arg);
CcuResult CcuLarge8Plus4Layer1DirectPull(CcuKernelArg arg);

} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
