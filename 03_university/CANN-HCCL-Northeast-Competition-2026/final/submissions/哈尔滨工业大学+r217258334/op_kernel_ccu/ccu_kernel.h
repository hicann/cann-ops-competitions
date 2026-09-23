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

// Load the competition project's own definitions first and by an explicit
// relative path.  CcuKernelArgBase and MAX_RANK_SIZE are defined through
// custom.h/common.h and must be visible before the CCU kernel argument structs
// below are parsed.
#include "../include/custom.h"

#include <vector>

#include <ccu/ccu_types.h>
#include <ccu/ccu_variable.hpp>
#include <ccu/ccu_event.hpp>
#include <ccu/ccu_primitives.hpp>

namespace ccu = ::AscendC::ccu;

namespace ops_hccl {

struct CcuKernelArgScatterDirect : public CcuKernelArgBase {
    uint32_t rankSize;
    uint32_t rankId;
    uint32_t peerRanks[MAX_RANK_SIZE];
};

struct CcuScatterDirectContext {
    const CcuKernelArgScatterDirect *arg;

    ccu::Variable input;
    ccu::Variable inputToken;
    std::vector<ccu::Variable> output;
    std::vector<ccu::Variable> outputToken;
    ccu::Variable recvBytes;
    ccu::Variable chunkOffset;
    ccu::Variable chunkSize;
    ccu::Variable root;
    ccu::Event event;
};

struct CcuKernelArgScatterRelay : public CcuKernelArgBase {
    uint32_t rankSize;
    uint32_t rankId;
    uint32_t peerRanks[MAX_RANK_SIZE];
    uint32_t peerIsLocal[MAX_RANK_SIZE];
    uint32_t rootRelayTargetByRank[MAX_RANK_SIZE];
    uint32_t rootRelaySourceByRank[MAX_RANK_SIZE];
};

struct CcuScatterRelayContext {
    const CcuKernelArgScatterRelay *arg;

    ccu::Variable input;
    ccu::Variable inputToken;
    std::vector<ccu::Variable> output;
    std::vector<ccu::Variable> outputToken;
    ccu::Variable relayBufferLocal;
    ccu::Variable relayBufferTokenLocal;
    ccu::Variable recvBytes;
    ccu::Variable chunkOffset;
    ccu::Variable chunkSize;
    ccu::Variable directBytes;
    ccu::Variable relayBytes;
    ccu::Variable root;
    ccu::Variable relayTargetRank;
    ccu::Variable relaySourceRank;
    ccu::Variable phase;
    ccu::Event firstWaveEvent;
    ccu::Event secondWaveEvent;
};

CcuResult CcuScatterDirectKernel(CcuKernelArg arg);
CcuResult CcuScatterRelayKernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
