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

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "custom.h"

namespace ccu = ::AscendC::ccu;

namespace ops_hccl {

constexpr uint64_t CCU_MEMORY_SLICE_BYTES = 4096;
constexpr uint32_t CCU_MEMORY_INTERLEAVE = 8;
constexpr uint32_t CCU_LOCAL_COPY_SLICES_PER_LOOP = 8;
constexpr uint32_t CCU_LOCAL_COPY_LOOP_COUNT = 8;

struct LoopGroupConfig {
    uint32_t memoryInterleave = 0;
    uint32_t loopCount = 0;
    uint64_t memorySliceBytes = 0;
};

struct LoopGroupResource {
    ccu::Array<ccu::Event> completedEvents{0};
    ccu::Array<ccu::CcuBuffer> buffers{0};
    uint32_t eventCount = 0;
    uint32_t bufferCount = 0;
};

struct GroupCopySize {
    ccu::Variable addressOffset;
    ccu::Variable loopParameter;
    ccu::Variable parallelParameter;
    ccu::Variable residualBytes;
};

struct LoopEntity {
    std::unique_ptr<ccu::Func> body[2];
    std::unique_ptr<ccu::Loop> loops[2];
    ccu::Variable loopParameter[2];
};

struct AllGatherKernelContext {
    const AllGatherKernelArg *arg = nullptr;
    ccu::Variable input;
    std::vector<ccu::Variable> outputs;
    std::vector<ccu::Variable> tokens;
    ccu::Variable inputOffset;
    ccu::Variable outputOffset;
    ccu::Variable sliceBytes;
    ccu::Variable relayOutputOffset;
    ccu::Variable directBytes;
    ccu::Variable relayBytes;
    ccu::Variable phase;
    ccu::Variable firstSlice;
    ccu::Variable lastSlice;
    ccu::Event completionEvent;
    GroupCopySize groupCopySize;
    LoopGroupConfig copyConfig;
    LoopGroupResource copyResource;
    bool copyResourceAllocated = false;
    std::map<std::string, LoopEntity> loopEntities;
};

// CCU Kernel 函数
CcuResult CcuKernel(CcuKernelArg arg);
CcuResult CcuContinuationKernel(CcuKernelArg arg);
CcuResult CcuRelayKernel(CcuKernelArg arg);
} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
