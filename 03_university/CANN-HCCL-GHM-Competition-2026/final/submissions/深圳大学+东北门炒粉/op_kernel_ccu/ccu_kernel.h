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

namespace ccu = ::AscendC::ccu;

namespace ops_hccl {

struct CcuKernelArgReduceScatterPull : public CcuKernelArgBase {
    uint32_t rankId = 0;
    uint32_t rankSize = 0;
    uint32_t remoteRanks[MAX_RANK_SIZE]{};
    uint16_t addrMask = 0;
    uint16_t tokenMask = 0;
    bool prepareOwnContribution = false;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    HcclReduceOp reduceOp = HCCL_REDUCE_SUM;
};

struct CcuKernelArgReduceScatterReduce {
    uint32_t rankId = 0;
    uint32_t rankSize = 0;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    HcclReduceOp reduceOp = HCCL_REDUCE_SUM;
};

struct CcuKernelArgReduceScatterStripe : public CcuKernelArgReduceScatterPull {
    uint32_t groupIndex = 0;
    bool includeOwnContribution = false;
    bool writeOutput = false;
    bool finalBarrier = false;
    bool directStripe = false;
};

CcuResult CcuReduceScatterPullKernel(CcuKernelArg arg);
CcuResult CcuReduceScatterReduceKernel(CcuKernelArg arg);
CcuResult CcuReduceScatterReleaseKernel(CcuKernelArg arg);
CcuResult CcuReduceScatterFusedKernel(CcuKernelArg arg);
CcuResult CcuReduceScatterStripeKernel(CcuKernelArg arg);
CcuResult CcuReduceScatterStripeReduceKernel(CcuKernelArg arg);
CcuResult CcuReduceScatterDirectStripeKernel(CcuKernelArg arg);
CcuResult CcuReduceScatterDirectMergeKernel(CcuKernelArg arg);
} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
