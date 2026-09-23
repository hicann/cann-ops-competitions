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

enum ScatterHelperPhase : uint64_t {
    SCATTER_HELPER_NORMAL = 0,
    SCATTER_HELPER_STAGE = 1,
    SCATTER_HELPER_FORWARD = 2,
    SCATTER_HELPER_FINISH = 3,
    SCATTER_WIDE_DIRECT = 4,
};

struct CcuKernelArgScatter : public CcuKernelArgBase {
    uint32_t rankSize{0};
    uint32_t rankId{INVALID_VALUE_RANKID};
    uint32_t peerRanks[MAX_RANK_SIZE]{};
    uint32_t peerCount{0};
    bool includeLocal{false};
    uint32_t localGroupByRank[MAX_RANK_SIZE]{};
    uint32_t smallSingleReady{0};
    uint32_t rank12FourByThree{0};
    uint32_t largeBlock{0};
};

// CCU Kernel 函数
CcuResult CcuKernel(CcuKernelArg arg);
CcuResult CcuSmallReceiverKernel(CcuKernelArg arg);
CcuResult CcuRank12SmallRootKernel(CcuKernelArg arg);
CcuResult CcuHelperKernel(CcuKernelArg arg);
CcuResult CcuWidePushKernel(CcuKernelArg arg);
} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
