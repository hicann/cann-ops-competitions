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

#include "custom.h"

namespace ccu = ::AscendC::ccu;

namespace ops_hccl {

struct CcuKernelArgAllGather : public CcuKernelArgBase {
    uint32_t rankSize;
    uint32_t rankId;
    uint32_t handleSelf;
};

struct CcuKernelArgPipelineIntra : public CcuKernelArgBase {
    uint32_t rankSize;
    uint32_t rankId;
    uint32_t pairRank;
};

struct CcuKernelArgPipelineInter : public CcuKernelArgBase {
    uint32_t rankSize;
    uint32_t rankId;
    uint32_t localRankCount;
    uint32_t localRanks[MAX_RANK_SIZE];
};

struct CcuKernelArgAsymmetricIntra : public CcuKernelArgBase {
    uint32_t rankSize;
    uint32_t rankId;
    uint32_t relayRankCount;
    uint32_t relayRanks[MAX_RANK_SIZE];
};

struct CcuKernelArgAsymmetricInter : public CcuKernelArgBase {
    uint32_t rankSize;
    uint32_t rankId;
    uint32_t localRankCount;
    uint32_t localRanks[MAX_RANK_SIZE];
    uint32_t closSendCount;
    uint32_t meshSendCount;
};

struct CcuKernelContext {
    const CcuKernelArgAllGather *arg;
    ccu::Variable input;
    ccu::Variable localOutput;
    ccu::Variable localToken;
    std::vector<ccu::Variable> output;
    std::vector<ccu::Variable> token;
    ccu::Variable currentRankOutputOffset;
    ccu::Variable normalSliceSize;
    ccu::Variable lastSliceSize;
    ccu::Variable repeatNum;
    ccu::Variable currentSliceSize;
    ccu::Variable currentOffset;
    ccu::Variable constOne;
    ccu::Event event;
};

struct CcuPipelineContext {
    const CcuKernelArgBase *arg;
    ccu::Variable input;
    ccu::Variable localOutput;
    ccu::Variable localToken;
    std::vector<ccu::Variable> output;
    std::vector<ccu::Variable> token;
    ccu::Variable dataSize;
    ccu::Variable meshBytes;
    ccu::Variable nhrOffset;
    ccu::Variable nhrBytes;
    ccu::Variable stage;
    ccu::Event event;
};

// CCU Kernel 函数
CcuResult CcuKernel(CcuKernelArg arg);
CcuResult CcuPipelineIntraKernel(CcuKernelArg arg);
CcuResult CcuPipelineInterKernel(CcuKernelArg arg);
CcuResult CcuAsymmetricIntraKernel(CcuKernelArg arg);
CcuResult CcuAsymmetricInterKernel(CcuKernelArg arg);
} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
