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

#include <ccu/ccu_types.h>

#include "common.h"
#include "custom.h"
#include "log.h"

namespace ccu = ::AscendC::ccu;

#define CCU_CHECK_RET(call) \
    do { \
        CcuResult ccuRet = (call); \
        if (ccuRet != CCU_SUCCESS) { \
            HCCL_ERROR("[%s] CCU call failed: %d", __func__, ccuRet); \
            return ccuRet; \
        } \
    } while (0)

namespace ops_hccl {

// CCU Kernel 函数
CcuResult CcuKernel(CcuKernelArg arg);

// Dedicated 16-rank large-message kernel. Keeping the Scratch FullMesh
// instruction graph separate avoids inflating the proven baseline kernel.
CcuResult CcuMesh16ScratchKernel(CcuKernelArg arg);

// Dedicated 12-rank large-message kernel. It combines the official Mesh
// Mem2Mem Scratch ReduceScatter structure with four-step NHR AllGather.
CcuResult CcuMesh12ScratchKernel(CcuKernelArg arg);

// Dedicated 4-rank large-message kernel. Reuses the rank-generic Scratch RS
// and Doubling AG path already validated on 16-rank topology.
CcuResult CcuMesh4ScratchKernel(CcuKernelArg arg);

// Fixed 2x8 large-message parallel executor.  The Mesh and Clos kernels are
// registered on their matching IO dies and are ordered only by coarse host
// thread barriers between the four complementary stages.
CcuResult CcuParallel2x8MeshKernel(CcuKernelArg arg);
CcuResult CcuParallel2x8ClosKernel(CcuKernelArg arg);

// TODO: 可编写多个 CCU Kernel 函数，以最大化性能
// CcuResult CcuKernel2(CcuKernelArg arg);
} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H