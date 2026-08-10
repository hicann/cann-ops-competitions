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

enum CcuTaskArgId : uint32_t {
    TASK_INPUT_ADDR = 0,
    TASK_OUTPUT_ADDR,
    TASK_INPUT_TOKEN,
    TASK_OUTPUT_TOKEN,
    TASK_SELF_OFFSET,
    TASK_CHUNK_SIZE,
    TASK_LOCAL_COPY,
    TASK_DIRECT_SIZE,
    TASK_RELAY_SOURCE_ADDR,
    TASK_RELAY_SOURCE_OFFSET,
    TASK_RELAY_SIZE,
    TASK_ARG_COUNT,
};

enum class AllGatherKernelPhase : uint32_t {
    DISTRIBUTE = 0,
    RELAY = 1,
};

enum class AllGatherTransferMode : uint32_t {
    FULL_CHUNK = 0,
    DIRECT_PREFIX = 1,
};

struct CcuAllGatherKernelArg : public CcuKernelArgBase {
    uint32_t rankSize = 0;
    uint32_t rankId = 0;
    uint32_t algorithm = static_cast<uint32_t>(AllGatherAlgorithm::DIRECT);
    uint32_t phase = static_cast<uint32_t>(AllGatherKernelPhase::DISTRIBUTE);
    uint32_t handleLocalCopy = 0;
    uint32_t peerRanks[MAX_RANK_SIZE]{};
    uint32_t transferModes[MAX_RANK_SIZE]{};
};

// CCU Kernel 函数
CcuResult CcuKernel(CcuKernelArg arg);

// TODO: 可编写多个 CCU Kernel 函数，以最大化性能
// CcuResult CcuKernel2(CcuKernelArg arg);
} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
