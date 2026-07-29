/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CUSTOM_H
#define OPS_HCCL_CUSTOM_H

#include <vector>
#include <hccl/hccl_res.h>
#include <hccl/hccl_types.h>

#include "binary_stream.h"
#include "common.h"

enum class TopologyKind : uint32_t {
    TWO_BY_EIGHT = 0,
    FOUR_BY_ONE = 1,
    EIGHT_PLUS_FOUR = 2,
};

// KernelMode 会随执行资源序列化，每种模式只匹配对应的注册参数和任务参数 ABI。
enum class KernelMode : uint32_t {
    SMALL_OWNER_PULL = 0,
    SMALL_4X1_ONESHOT = 1,
    LARGE_LATIN = 2,
    LARGE_2X8_PACKED_PULL = 3,
    LARGE_4X1_FUSED_RSAG = 4,
};

namespace latin {
constexpr uint64_t INIT_PHASE = 0;
constexpr uint64_t REDUCE_PHASE = 1;
constexpr uint64_t GATHER_PHASE = 2;
constexpr uint64_t BARRIER_PHASE = 3;
constexpr uint32_t OFFSET_ARG = 9;
constexpr uint32_t MAX_ARG_COUNT = OFFSET_ARG + MAX_RANK_SIZE;
} // namespace latin

namespace small_data {
constexpr uint64_t TOTAL_BYTES = 512ULL * 1024ULL;
constexpr uint32_t TASK_ARG_COUNT = 5;
} // namespace small_data

namespace packed_pull {
constexpr uint64_t INIT_PHASE = 0;
constexpr uint64_t MERGE_PHASE = 1;
constexpr uint64_t GATHER_PHASE = 2;
constexpr uint32_t TASK_ARG_COUNT = 13;
} // namespace packed_pull

namespace fused_4x1 {
constexpr uint32_t RANK_SIZE = 4;
constexpr uint32_t TASK_ARG_COUNT = 12;
} // namespace fused_4x1

struct SmallKernelArg {
    // 小消息 CCU 图在注册阶段固化的常量参数。
    ChannelHandle channels[MAX_RANK_SIZE];
    uint32_t peers[MAX_RANK_SIZE];
    uint32_t channelCount;
    uint32_t myRank;
    uint32_t rankSize;
    KernelMode mode;
    uint64_t localScratch[MAX_RANK_SIZE];
    uint64_t remoteScratch[MAX_RANK_SIZE];
    uint64_t ownerOffset;
    uint64_t ownerBytes;
    uint64_t smallBytes;
};

struct LargeKernelArg {
    // 每个大消息 Kernel 只持有终结于其注册 Die 的 Channel。
    ChannelHandle channels[MAX_RANK_SIZE];
    uint32_t peers[MAX_RANK_SIZE];
    uint32_t channelCount;
    uint32_t dieId;
    uint32_t myRank;
    uint32_t rankSize;
    uint32_t primaryDie;
    KernelMode mode;
};

struct AlgResourceCtx {
    // 由 HCCL 缓存、每次任务下发时反序列化的 Host 资源。
    std::vector<ThreadHandle> ccuThreads;
    std::vector<CcuKernelHandle> kernels;
    KernelMode mode;
    uint64_t scratchAddr;
    uint64_t scratchSize;
    uint64_t scratchToken;
    uint32_t primaryDie;
    uint32_t launchFirstDie;
    uint32_t channelCounts[2];

    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << ccuThreads;
        binaryStream << kernels;
        binaryStream << mode;
        binaryStream << scratchAddr;
        binaryStream << scratchSize;
        binaryStream << scratchToken;
        binaryStream << primaryDie;
        binaryStream << launchFirstDie;
        binaryStream << channelCounts[0];
        binaryStream << channelCounts[1];
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> ccuThreads;
        binaryStream >> kernels;
        binaryStream >> mode;
        binaryStream >> scratchAddr;
        binaryStream >> scratchSize;
        binaryStream >> scratchToken;
        binaryStream >> primaryDie;
        binaryStream >> launchFirstDie;
        binaryStream >> channelCounts[0];
        binaryStream >> channelCounts[1];
    }
};

#endif // OPS_HCCL_CUSTOM_H
