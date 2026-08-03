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

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <ccu/ccu_types.h>

#include "common.h"
#include "custom.h"

namespace ccu = ::AscendC::ccu;

namespace ops_hccl {

constexpr uint32_t RELAY_CHUNK_COUNT = 8;
constexpr uint32_t RELAY_SLOT_COUNT = 12;
constexpr uint32_t MAX_MESH_OPS_PER_SLOT = 7;
constexpr uint32_t MAX_CLOS_OPS_PER_SLOT = 4;
constexpr uint64_t DIRECT_FAST_PATH_MAX_BYTES = 16ULL * 1024ULL * 1024ULL;

constexpr uint64_t CCU_MS_INTERLEAVE = 8;
constexpr uint64_t CCU_MS_SIZE = 4096;
constexpr uint32_t CCU_LOCAL_COPY_MS_PER_LOOP = 8;
constexpr uint32_t CCU_MS_LOCAL_COPY_LOOP_COUNT = 8;

enum class TransferKind : uint8_t {
    NATIVE = 0,
    CLOS = 1,
    RELAY = 2,
};

struct TransferDesc {
    TransferKind kind = TransferKind::NATIVE;
    uint8_t chunkId = 0;
    uint8_t ownerRank = 0;
    uint8_t peerRank = 0;
    uint8_t signalReady = 0;
    uint8_t waitReady = 0;
};

struct RelaySlotSchedule {
    uint8_t meshCount = 0;
    uint8_t closCount = 0;
    uint8_t localCopyMask = 0;
    std::array<TransferDesc, MAX_MESH_OPS_PER_SLOT> meshOps{};
    std::array<TransferDesc, MAX_CLOS_OPS_PER_SLOT> closOps{};
};

struct CcuKernelArgDirect : public CcuKernelArgBase {
    uint32_t rankSize = 0;
    uint32_t rankId = 0;
    uint32_t ownsLocalCopy = 0;
    uint32_t useSmallMessageFastPath = 0;
};

struct CcuKernelArgRelay2x8 : public CcuKernelArgBase {
    uint32_t rankSize = 0;
    uint32_t rankId = 0;
    std::array<RelaySlotSchedule, RELAY_SLOT_COUNT> slots{};
};

struct LoopGroupConfig {
    uint32_t msInterleave = 0;
    uint32_t loopCount = 0;
    uint64_t memSlice = 0;
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

struct CopyContext {
    LoopGroupConfig config;
    LoopGroupResource resources;
    bool resourceAllocated = false;
    std::map<std::string, CcuLoopEntity> loopMap;
    std::array<ccu::LocalAddr, 2> loopSource;
    std::array<ccu::LocalAddr, 2> loopDestination;
    std::array<ccu::Variable, 2> loopLength;
    ccu::Variable scratchLoopParam;
    ccu::Variable scratchMemSlice;
    ccu::Variable scratchParallelConfig;
    ccu::Variable scratchOffsetConfig;
    std::array<ccu::Variable, 2> scratchLoopConfig;

    void CreateLoopEntity(const std::string &name)
    {
        loopMap.emplace(name, CcuLoopEntity());
    }

    bool IsLoopEntityRegistered(const std::string &name) const
    {
        return loopMap.count(name) != 0;
    }
};

CcuResult DirectAllGatherKernel(CcuKernelArg arg);
CcuResult RelayAllGather2x8Kernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
