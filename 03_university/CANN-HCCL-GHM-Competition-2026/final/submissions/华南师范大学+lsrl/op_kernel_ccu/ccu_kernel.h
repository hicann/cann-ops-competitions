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

namespace ops_hccl {

// Deterministic HBM-tree ReduceScatter kernel registered by the host.
CcuResult CcuKernel(CcuKernelArg arg);

// Main/worker pair used when the communication channels span two IO Dies.
CcuResult CcuReduceScatterMainKernel(CcuKernelArg arg);
CcuResult CcuReduceScatterWorkerKernel(CcuKernelArg arg);
CcuResult CcuReduceScatterReduceKernel(CcuKernelArg arg);

// Two-dimensional 2x8 ReduceScatter pipeline.  The first pair of kernels
// processes disjoint A/B partitions concurrently; the second pair swaps the
// topology dimension and finishes both partitions in the user output.
CcuResult CcuReduceScatter2DMeshAStage1Kernel(CcuKernelArg arg);
CcuResult CcuReduceScatter2DNhrBStage1Kernel(CcuKernelArg arg);
CcuResult CcuReduceScatter2DMeshBStage2Kernel(CcuKernelArg arg);
CcuResult CcuReduceScatter2DNhrAStage2Kernel(CcuKernelArg arg);

// Asymmetric 8+4 two-dimensional pipeline.  The topology is supplied in
// rank-graph order (the eight-rank server followed by the four-rank server);
// no global-rank numbering convention is assumed.
CcuResult CcuReduceScatterAsymMeshAStage1Kernel(CcuKernelArg arg);
CcuResult CcuReduceScatterAsymNhrBStage1Kernel(CcuKernelArg arg);
CcuResult CcuReduceScatterAsymMeshBStage2Kernel(CcuKernelArg arg);
CcuResult CcuReduceScatterAsymNhrAStage2Kernel(CcuKernelArg arg);

// Three-lane, three-step deterministic ReadReduce for large 4x1 inputs.
CcuResult CcuReduceScatter4x1LatinKernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
