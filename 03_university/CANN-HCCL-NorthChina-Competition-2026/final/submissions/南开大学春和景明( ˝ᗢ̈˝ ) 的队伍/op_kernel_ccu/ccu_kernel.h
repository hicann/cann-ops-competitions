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

#include "custom.h"

namespace ccu = ::AscendC::ccu;

namespace ops_hccl {

constexpr uint64_t CCU_MS_INTERLEAVE = 8;
constexpr uint64_t CCU_MS_SIZE = 4096;
constexpr uint32_t CCU_LOCAL_COPY_MS_PER_LOOP = 8;
constexpr uint32_t CCU_MS_LOCAL_COPY_LOOP_COUNT = 8;

enum class AllGatherKernelRole : uint32_t {
    DIRECT = 0,
    INTRA_SERVER = 1,
    INTER_SERVER = 2,
};

enum class AllGatherExecutionMode : uint64_t {
    DIRECT = 0,
    HIERARCHICAL_PHASE_ONE = 1,
    HIERARCHICAL_PHASE_TWO = 2,
};

struct LoopGroupConfig {
    uint32_t msInterleave;
    uint32_t loopCount;
    uint64_t memSlice;
};

struct LoopGroupResource {
    ccu::Array<ccu::Event> completedEvent{0};
    ccu::Array<ccu::CcuBuffer> ccuBuf{0};
    uint32_t eventCount = 0;
    uint32_t bufCount = 0;
};

struct GroupOpSizeVars {
    ccu::Variable addrOffset;
    ccu::Variable loopParam;
    ccu::Variable parallelParam;
    ccu::Variable residual;
};

struct CcuLoopEntity {
    std::unique_ptr<ccu::Func> body[2];
    std::unique_ptr<ccu::Loop> loops[2];
    ccu::Variable loopParam[2];
};

struct AllGatherKernelArg : public CcuKernelArgBase {
    uint32_t rankSize = 0;
    uint32_t rankId = 0;
    uint32_t peerRanks[MAX_RANK_SIZE]{};
    uint32_t localRanks[MAX_RANK_SIZE]{};
    uint32_t interRanks[MAX_RANK_SIZE]{};
    uint32_t localRankCount = 0;
    uint32_t interRankCount = 0;
    uint32_t localPeerCount = 0;
    uint32_t pairedRank = 0;
    uint32_t pairedChannelIndex = 0;
    AllGatherKernelRole role = AllGatherKernelRole::DIRECT;
    bool hierarchical2x8 = false;
    bool hierarchicalGrid = false;
    bool purePhysicalGrid = false;
    bool physical8Plus4 = false;
    bool latencyOptimizedBarrier = false;
    bool copyLocal = false;
    bool useNhr = false;
    bool hasGatewayTarget = false;
    uint32_t gatewayTargetChannelIndex = 0;
    uint32_t gatewaySourceCount = 0;
    uint32_t gatewaySourceRanks[MAX_RANK_SIZE]{};
    uint32_t nhrStepCount = 0;
    uint32_t nhrToRanks[4]{};
    uint32_t nhrFromRanks[4]{};
    uint32_t nhrToChannelIndices[4]{};
    uint32_t nhrFromChannelIndices[4]{};
    uint32_t nhrTxOffsets[5]{};
    uint32_t nhrTxRanks[MAX_RANK_SIZE]{};
};

struct AllGatherContext {
    const AllGatherKernelArg *arg = nullptr;
    ccu::Variable input;
    ccu::Variable inputToken;
    std::vector<ccu::Variable> output;
    std::vector<ccu::Variable> outputToken;
    ccu::Variable currentRankOutputOffset;
    ccu::Variable sliceSize;
    ccu::Variable localCopyNeeded;
    ccu::Variable dataSize;
    ccu::Variable partOffset;
    ccu::Variable otherPartOffset;
    ccu::Variable otherPartSize;
    ccu::Variable executionMode;
    std::vector<ccu::Variable> rankOutputOffsets;
    ccu::Event event;
    GroupOpSizeVars goSize;
    LoopGroupConfig copyConfig{};
    LoopGroupResource copyResource;
    ccu::LocalAddr copyLoopSrc[2];
    ccu::LocalAddr copyLoopDst[2];
    ccu::Variable copyLoopLength[2];
    bool resourceAllocated = false;
    std::map<std::string, CcuLoopEntity> loopMap;

    void CreateLoopEntity(const std::string &name)
    {
        loopMap.emplace(name, CcuLoopEntity());
    }

    bool IsLoopEntityRegistered(const std::string &name) const
    {
        return loopMap.count(name) != 0;
    }
};

// CCU Kernel 函数
CcuResult CcuKernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
