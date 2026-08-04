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

#include <cstdint>
#include <ccu/ccu_types.h>

namespace ops_hccl {

constexpr uint32_t MAX_LAYER_SOURCES = 8;
constexpr uint64_t CCU_MODE_GENERIC = 0;
constexpr uint64_t CCU_MODE_TWO_BY_EIGHT_CLOS = 1;
constexpr uint64_t CCU_MODE_TWO_BY_EIGHT_MESH = 2;
constexpr uint64_t CCU_MODE_MIXED_CLOS = 3;
constexpr uint64_t CCU_MODE_MIXED_MESH = 4;
// One-round fan-in used by the 512 KiB performance paths.  Each source is
// landed in a private CCL slot before the deterministic local reduction tree.
constexpr uint64_t CCU_MODE_SMALL_FANIN = 5;
// Large-message partial fan-in. Sources are paired inside one CCU kernel:
// two striped network rounds produce independent group partials, followed by
// a deterministic local binary tree. This reduces network waves without
// adding host phases or cross-die dependencies.
constexpr uint64_t CCU_MODE_LARGE_GROUPED = 6;
constexpr uint64_t CCU_MODE_LARGE_GROUPED4_INIT = 7;
constexpr uint64_t CCU_MODE_LARGE_GROUPED4_ACCUM = 8;
constexpr uint64_t CCU_MODE_V23_MESH_HELPER = 9;
constexpr uint64_t CCU_MODE_V23_CLOS_DIRECT = 10;
constexpr uint64_t CCU_MODE_V23_CLOS_FINAL = 11;
constexpr uint32_t CCU_TASK_ARG_COUNT = 12;
constexpr uint32_t CCU_V23_TASK_ARG_COUNT = 15;

// layer-0 and layer-1 must be registered as different kernels.
CcuResult CcuReduceLayer0Kernel(CcuKernelArg arg);
CcuResult CcuReduceLayer1Kernel(CcuKernelArg arg);

// Dedicated large-message entry points keep the grouped instruction graph out
// of the already feature-rich generic kernels and below translator limits.
CcuResult CcuReduceLayer0GroupedKernel(CcuKernelArg arg);
CcuResult CcuReduceLayer1GroupedKernel(CcuKernelArg arg);
CcuResult CcuReduceLayer0GroupedEarlyKernel(CcuKernelArg arg);
CcuResult CcuReduceLayer1GroupedEarlyKernel(CcuKernelArg arg);
CcuResult CcuReduceLayer0GroupedV13Kernel(CcuKernelArg arg);
CcuResult CcuReduceLayer1GroupedV13Kernel(CcuKernelArg arg);
CcuResult CcuReduceLayer0GroupedV23Kernel(CcuKernelArg arg);
CcuResult CcuReduceLayer1GroupedV23Kernel(CcuKernelArg arg);

// Combines layer partials locally in the deterministic order local + remote.
CcuResult CcuReduceFinalizeKernel(CcuKernelArg arg);
} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
