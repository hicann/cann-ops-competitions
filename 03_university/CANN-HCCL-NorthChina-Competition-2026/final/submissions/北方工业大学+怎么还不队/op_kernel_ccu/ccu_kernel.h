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

#include <ccu/ccu_types.h>

#include "common.h"
#include "custom.h"

namespace ccu = ::AscendC::ccu;

namespace ops_hccl {

// ============================================================
// CCU LocalCopy 参数（保持硬件安全值，不改 SLICE_SIZE 避免 MS_CCU 内存冲突）
// ============================================================
constexpr uint64_t CCU_MEMORY_SLICE_SIZE = 4096;       // 4KB，硬件对齐安全值
constexpr uint32_t CCU_MEMORY_SLICE_INTERLEAVE = 8;
constexpr uint32_t LOCAL_COPY_SLICES_PER_LOOP = 8;
constexpr uint32_t LOCAL_COPY_LOOP_COUNT = 8;
// 每次 LoopGroup 最小搬移粒度 = 4KB * 8 = 32KB
constexpr uint64_t LOCAL_COPY_MEMORY_SLICE = CCU_MEMORY_SLICE_SIZE * LOCAL_COPY_SLICES_PER_LOOP;

struct AllGatherKernelArg : public CcuKernelArgBase {
    uint32_t rankSize = 0;
    uint32_t rankId = 0;
    std::vector<uint32_t> peerRanks;
    bool copyLocal = false;
};

struct GroupCopySize {
    ccu::Variable addressOffset;
    ccu::Variable loopParam;
    ccu::Variable parallelParam;
    ccu::Variable residual;
};

struct LoopEntity {
    std::unique_ptr<ccu::Func> bodies[2];
    std::unique_ptr<ccu::Loop> loops[2];
    ccu::Variable parameters[2];
};

struct AllGatherContext {
    const AllGatherKernelArg *arg = nullptr;
    ccu::Variable input;
    std::vector<ccu::Variable> outputs;
    std::vector<ccu::Variable> tokens;
    ccu::Variable inputOffset;
    ccu::Variable outputOffset;
    ccu::Variable sliceSize;
    ccu::Event transferEvent;
    GroupCopySize groupCopySize;

    ccu::Array<ccu::Event> copyEvents{0};
    ccu::Array<ccu::CcuBuffer> copyBuffers{0};
    bool copyResourcesAllocated = false;
    std::map<std::string, LoopEntity> loopEntities;
};

namespace large {
constexpr uint32_t ALLGATHER_SOURCE_VERSION = 0x090000;
constexpr uint64_t CCU_MEMORY_SLICE_SIZE = 4096;
constexpr uint32_t CCU_MEMORY_SLICE_INTERLEAVE = 8;
constexpr uint32_t LOCAL_COPY_SLICES_PER_LOOP = 8;
constexpr uint32_t LOCAL_COPY_LOOP_COUNT = 8;
constexpr uint64_t LOCAL_COPY_MEMORY_SLICE = CCU_MEMORY_SLICE_SIZE * LOCAL_COPY_SLICES_PER_LOOP;
constexpr uint32_t BALANCED_SERVER_RANKS = 8;
constexpr uint32_t BALANCED_SMALL_SERVER_RANKS = 4;
constexpr uint32_t BALANCED_STRIPE_COUNT = 4;

struct PipelineRelayTask {
    uint32_t sourceRank = 0;
    std::vector<uint32_t> destinationRanks;
};

struct AllGatherKernelArg : public CcuKernelArgBase {
    uint32_t rankSize = 0;
    uint32_t rankId = 0;
    std::vector<uint32_t> peerRanks;
    bool copyLocal = false;
    uint32_t networkClass = 0;
    bool test12Special = false;
    std::vector<uint32_t> pipelineDirectRanks;
    std::vector<uint32_t> pipelineRemoteRelayRanks;
    std::vector<PipelineRelayTask> pipelineRelayTasks;
};

struct GroupCopySize {
    ccu::Variable addressOffset;
    ccu::Variable loopParam;
    ccu::Variable parallelParam;
    ccu::Variable residual;
};

struct LoopEntity {
    std::unique_ptr<ccu::Func> bodies[2];
    std::unique_ptr<ccu::Loop> loops[2];
    ccu::Variable parameters[2];
};

struct AllGatherContext {
    const AllGatherKernelArg *arg = nullptr;
    ccu::Variable input;
    std::vector<ccu::Variable> outputs;
    std::vector<ccu::Variable> tokens;
    ccu::Variable inputOffset;
    ccu::Variable outputOffset;
    ccu::Variable sliceSize;
    ccu::Array<ccu::Event> transferEvents{0};
    GroupCopySize groupCopySize;

    ccu::Variable balancedMode;
    ccu::Variable rankDataSize;
    ccu::Variable stripeOffsets[BALANCED_STRIPE_COUNT];
    ccu::Variable stripeSizes[BALANCED_STRIPE_COUNT];
    ccu::Array<ccu::Event> balancedEvents{0};

    ccu::Array<ccu::Event> copyEvents{0};
    ccu::Array<ccu::CcuBuffer> copyBuffers{0};
    bool copyResourcesAllocated = false;
    std::map<std::string, LoopEntity> loopEntities;
};

CcuResult CcuKernel(CcuKernelArg arg);
CcuResult CcuKernelRank16(CcuKernelArg arg);
CcuResult CcuKernelRank12(CcuKernelArg arg);
} // namespace large
CcuResult CcuKernel(CcuKernelArg arg);
} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
