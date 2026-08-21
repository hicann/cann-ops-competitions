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

CcuResult CcuReduceScatterLayerDiePairwiseKernel(CcuKernelArg arg);
CcuResult CcuReduceScatterSmallReadyKernel(CcuKernelArg arg);
CcuResult CcuReduceScatterFusedReadyKernel(CcuKernelArg arg);
CcuResult CcuReduceScatterFlatTreeKernel(CcuKernelArg arg);
CcuResult CcuReduceScatterTeammateTreeKernel(CcuKernelArg arg);
CcuResult CcuReduceScatterRank4SmallRhKernel(CcuKernelArg arg);
CcuResult CcuReduceScatterRank16SmallThinKernel(CcuKernelArg arg);
namespace teammate_exact_kernel {
CcuResult CcuReduceScatterLayerDiePairwiseKernel(CcuKernelArg arg);
CcuResult CcuReduceScatterSmallKernel(CcuKernelArg arg);
CcuResult CcuReduceScatterReadyPipelineKernel(CcuKernelArg arg);
} // namespace teammate_exact_kernel

#endif // OPS_HCCL_CCU_KERNEL_H
