/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_BROADCAST_CCU_PROBLEM_247_CCU_KERNEL_H
#define HCCL_BROADCAST_CCU_PROBLEM_247_CCU_KERNEL_H

#include <vector>
#include <ccu/ccu_types.h>

#include "custom.h"

namespace ccu = ::AscendC::ccu;

namespace ops_hccl {

struct BroadcastMesh1DMem2MemContext {
    const CcuBroadcastMesh1DMem2MemKernelArg *arg = nullptr;

    ccu::Variable myInput;
    ccu::Variable myOutput;
    ccu::Variable myToken;
    std::vector<ccu::Variable> remoteOutput;
    std::vector<ccu::Variable> remoteToken;

    ccu::Variable dataSize;
    ccu::Variable normalSliceSize;
    ccu::Variable lastSliceSize;
    ccu::Variable allgatherOffset;
    ccu::Variable secondNormalSliceSize;
    ccu::Variable secondLastSliceSize;
    ccu::Variable secondAllgatherOffset;
    ccu::Variable secondOffset;
    ccu::Variable secondDataSize;
    ccu::Variable zeroOffset;
    std::vector<ccu::Variable> nhrSliceOffset;
    ccu::Event event;
    ccu::LocalAddr localSrc;
    std::vector<ccu::LocalAddr> scatterSrc;
    std::vector<ccu::RemoteAddr> remoteDst;
};

CcuResult CcuBroadcastMesh1DMem2MemKernel(CcuKernelArg arg);
CcuResult CcuBroadcastDirectKernel(CcuKernelArg arg);
CcuResult CcuBroadcastParallelDirectLayer0Kernel(CcuKernelArg arg);
CcuResult CcuBroadcastParallelDirectLayer1Kernel(CcuKernelArg arg);
CcuResult CcuBroadcastNhr1DMem2MemKernel(CcuKernelArg arg);
CcuResult CcuBroadcastHierarchicalLayer0Kernel(CcuKernelArg arg);
CcuResult CcuBroadcastHierarchicalLayer1Kernel(CcuKernelArg arg);
} // namespace ops_hccl

#endif // HCCL_BROADCAST_CCU_PROBLEM_247_CCU_KERNEL_H