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

#include "custom.h"

namespace ops_hccl {

// CCU Kernel 函数
CcuResult CcuDirectBroadcastKernel(CcuKernelArg arg);
CcuResult CcuAsymmetricDirectBroadcastKernel(CcuKernelArg arg);
CcuResult CcuAsymmetricDirect512K2ArgBroadcastKernel(CcuKernelArg arg);
CcuResult CcuAsymmetricDirect512KRegisteredBroadcastKernel(CcuKernelArg arg);
CcuResult CcuAsymmetricDirect512KRegisteredNonRootBroadcastKernel(CcuKernelArg arg);
CcuResult CcuAsymmetricPull512K2ArgBroadcastKernel(CcuKernelArg arg);
CcuResult CcuScatterBroadcastKernel(CcuKernelArg arg);
CcuResult CcuAllGatherBroadcastKernel(CcuKernelArg arg);
CcuResult CcuLeanScatterBroadcastKernel(CcuKernelArg arg);
CcuResult CcuLeanAllGatherBroadcastKernel(CcuKernelArg arg);
CcuResult CcuPipelinedChainBroadcastKernel(CcuKernelArg arg);
CcuResult CcuPipelinedChainPairFirstBroadcastKernel(CcuKernelArg arg);
CcuResult CcuPipelinedChainPairMiddleBroadcastKernel(CcuKernelArg arg);
CcuResult CcuPipelinedChainPairLastBroadcastKernel(CcuKernelArg arg);
CcuResult CcuPipelinedChainPairTailBroadcastKernel(CcuKernelArg arg);
CcuResult CcuUniquePairBroadcastKernel(CcuKernelArg arg);
CcuResult CcuUniqueTailBroadcastKernel(CcuKernelArg arg);
CcuResult CcuUniqueDrainBroadcastKernel(CcuKernelArg arg);
CcuResult CcuCutThroughPairBroadcastKernel(CcuKernelArg arg);
CcuResult CcuQuad4Rolling2BroadcastKernel(CcuKernelArg arg);
CcuResult CcuFinalRolling5DrainBroadcastKernel(CcuKernelArg arg);
CcuResult CcuPersistentRolling2_400BroadcastKernel(CcuKernelArg arg);
CcuResult CcuPersistentRolling2_512BroadcastKernel(CcuKernelArg arg);
CcuResult CcuCutThroughTailBroadcastKernel(CcuKernelArg arg);
} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
