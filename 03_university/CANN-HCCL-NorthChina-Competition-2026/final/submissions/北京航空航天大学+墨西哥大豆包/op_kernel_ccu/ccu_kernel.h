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
#include "log.h"

namespace ccu = ::AscendC::ccu;

namespace ops_hccl {

constexpr uint64_t CCU_MS_INTERLEAVE = 8;
constexpr uint64_t CCU_MS_SIZE = 4096;
constexpr uint32_t CCU_LOCAL_COPY_MS_PER_LOOP = 8;
constexpr uint32_t CCU_MS_LOCAL_COPY_LOOP_COUNT = 8;
constexpr uint64_t EIGHT_PLUS_FOUR_CCU_MS_INTERLEAVE = 16;
constexpr uint32_t EIGHT_PLUS_FOUR_CCU_LOCAL_COPY_MS_PER_LOOP = 16;
constexpr uint32_t EIGHT_PLUS_FOUR_CCU_LOOP_COUNT = 4;
constexpr uint32_t CCU_ALLGATHER_REPEAT_NUM = 2;
constexpr uint64_t NHR_2X8_MS_INTERLEAVE = 4;
constexpr uint32_t NHR_2X8_LOCAL_COPY_MS_PER_LOOP = 4;
constexpr uint32_t NHR_2X8_LOCAL_COPY_LOOP_COUNT = 8;

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

struct GroupCopyVars {
    ccu::LocalAddr loopSrc[2];
    ccu::LocalAddr loopDst[2];
    ccu::Variable loopLen[2];
};

struct GroupBroadcastVars {
    ccu::LocalAddr loopSrc[2];
    ccu::LocalAddr loopLocalDst[2];
    std::vector<ccu::RemoteAddr> loopRemoteDst[2];
    ccu::Variable loopLen[2];
};

struct CcuLoopEntity {
    std::unique_ptr<ccu::Func> body[2];
    std::unique_ptr<ccu::Loop> loops[2];
    ccu::Variable loopParam[2];
};

struct CcuKernelArgAllGather : public CcuKernelArgBase {
    uint64_t rankSize;
    uint32_t rankId;
    uint32_t netLayer;
    uint32_t handleSelfRank;
    uint32_t small512FastMode;
    AllGatherTopology topology;
    AllGatherSizeClass sizeClass;
};

struct AllGatherBaseContext {
    const CcuKernelArgAllGather *arg = nullptr;

    ccu::Variable input;
    std::vector<ccu::Variable> output;
    std::vector<ccu::Variable> token;
    std::vector<ccu::Event> events;
};

struct SmallAllGatherContext : public AllGatherBaseContext {
    ccu::Variable currentRankSliceInputOffset;
    ccu::Variable currentRankSliceOutputOffset;
    ccu::Variable sliceSize;
};

struct Small512FastContext : public SmallAllGatherContext {
    ccu::Variable refreshResources;
};

struct AllGatherContext : public AllGatherBaseContext {
    ccu::Variable repeatCount;
    ccu::Variable currentRankSliceInputOffset[CCU_ALLGATHER_REPEAT_NUM];
    ccu::Variable currentRankSliceOutputOffset[CCU_ALLGATHER_REPEAT_NUM];
    ccu::Variable sliceSize[CCU_ALLGATHER_REPEAT_NUM];

    GroupOpSizeVars goSize[CCU_ALLGATHER_REPEAT_NUM];
    GroupCopyVars groupCopyVars;
    GroupBroadcastVars groupBroadcastVars;
    LoopGroupConfig loopConfig{};
    LoopGroupResource loopResource;
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

struct Nhr4x1LargeContext : public AllGatherContext {
    ccu::Variable rankDataSize;
};

struct MeshNhr2x8LargeContext : public AllGatherContext {
    ccu::Variable phase;
    ccu::Variable rankDataSize;
};

struct EightPlusFourLargeContext : public AllGatherContext {
    ccu::Variable rankDataSize;
    ccu::Variable batchOffset;
    ccu::Variable half0Size;
    ccu::Variable half1Size;
    ccu::Variable lSize;
    ccu::Variable sSize;
    ccu::Variable qSize;
    GroupOpSizeVars half0GoSize;
    GroupOpSizeVars half1GoSize;
    ccu::Variable phase;
};

namespace topo_2x8 {
    CcuResult CcuAllGatherSmallKernel(CcuKernelArg arg);
    CcuResult CcuAllGatherLargeKernel(CcuKernelArg arg);
} // namespace topo_2x8

namespace topo_4x1 {
    CcuResult CcuAllGatherSmallKernel(CcuKernelArg arg);
    CcuResult CcuAllGatherLargeKernel(CcuKernelArg arg);
} // namespace topo_4x1

namespace topo_8plus4 {
    CcuResult CcuAllGatherSmallKernel(CcuKernelArg arg);
    CcuResult CcuAllGatherLargeKernel(CcuKernelArg arg);
} // namespace topo_8plus4

} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
