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
#include "../include/common.h"
#include "../include/custom.h"

namespace ccu = ::AscendC::ccu;

namespace ops_hccl {

// CCU Kernel 函数
CcuResult CcuKernel(CcuKernelArg arg);
CcuResult CcuCombineKernel(CcuKernelArg arg);
CcuResult CcuSmall4LaneKernel(CcuKernelArg arg);
CcuResult CcuSmall16HierKernel(CcuKernelArg arg);
CcuResult CcuSmall4Kernel(CcuKernelArg arg);
CcuResult CcuSmall12Kernel(CcuKernelArg arg);
CcuResult CcuSmall16Kernel(CcuKernelArg arg);
CcuResult CcuSmall4TiledFusedKernel(CcuKernelArg arg);
CcuResult CcuSmall12TiledKernel(CcuKernelArg arg);
CcuResult CcuSmall16TiledKernel(CcuKernelArg arg);
CcuResult CcuLarge4Kernel(CcuKernelArg arg);
CcuResult CcuLarge12Kernel(CcuKernelArg arg);
CcuResult CcuLarge16Kernel(CcuKernelArg arg);
CcuResult CcuLargePipeKernel(CcuKernelArg arg);
CcuResult CcuTopo1LargeReduce4Kernel(CcuKernelArg arg);
CcuResult CcuTopo1LargeGather4Kernel(CcuKernelArg arg);
CcuResult CcuTopo2LargeReduce12Kernel(CcuKernelArg arg);
CcuResult CcuTopo2LargeGather12Kernel(CcuKernelArg arg);
CcuResult CcuTopo3SmallReduce8x8Kernel(CcuKernelArg arg);
CcuResult CcuTopo3LargeReduce8x8Kernel(CcuKernelArg arg);
CcuResult CcuTopo3LargeGather8x8Kernel(CcuKernelArg arg);
CcuResult CcuHierarchicalLarge12Kernel(CcuKernelArg arg);
CcuResult CcuHierarchicalLarge16Kernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
