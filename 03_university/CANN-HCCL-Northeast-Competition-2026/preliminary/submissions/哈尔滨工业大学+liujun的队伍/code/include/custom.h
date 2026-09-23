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

#include <algorithm>
#include <array>
#include <limits>
#include <type_traits>
#include "common.h"

constexpr uint32_t SCATTER_WIDTH = 8;
constexpr uint32_t SCATTER_RANKS = 16;
constexpr uint32_t SCATTER_CHANNELS = SCATTER_RANKS - 1;
constexpr uint32_t SCATTER_THREADS = SCATTER_CHANNELS + 1;
constexpr uint32_t SCATTER_CONTEXT_MAGIC = 0x53435435;
constexpr uint64_t SCATTER_ALIGN = 128;
constexpr uint64_t SCATTER_MAX_TRANSFER = 256ULL * 1024 * 1024;
constexpr uint64_t SCATTER_RELAY_WEIGHT = 4;
constexpr uint64_t SCATTER_SPLIT_WEIGHT = 11;

struct CommBuffer {
    void *addr = nullptr;
    uint64_t size = 0;
};

struct ChannelInfo {
    uint32_t remoteRank = INVALID_VALUE_RANKID;
    uint32_t notifyNum = 2;
    ChannelHandle handle = 0;
    CommBuffer remoteCclMem;
};

struct AlgResourceCtx {
    uint32_t magic = SCATTER_CONTEXT_MAGIC;
    uint32_t rankSize = 0;
    uint32_t myRank = INVALID_VALUE_RANKID;
    uint32_t localIndex = 0;
    ThreadHandle aicpuThread = 0;
    CommBuffer localBuffer;
    std::array<ThreadHandle, SCATTER_THREADS> threads{};
    std::array<ChannelInfo, SCATTER_CHANNELS> channels{};
    std::array<uint32_t, SCATTER_WIDTH> localRanks{};
    std::array<uint32_t, SCATTER_WIDTH> remoteRanks{};
};
static_assert(std::is_trivially_copyable<AlgResourceCtx>::value, "Context must be a wire value");

struct ScatterSlice {
    uint64_t offset = 0;
    uint64_t aBytes = 0;
    uint64_t bBytes = 0;
    uint64_t aStride = 0;
    uint64_t bStride = 0;
    uint64_t BOffset() const
    {
        return 2 * aStride;
    }
};

inline uint64_t ScatterAlignUp(uint64_t bytes)
{
    return (bytes + SCATTER_ALIGN - 1) / SCATTER_ALIGN * SCATTER_ALIGN;
}

inline uint64_t ScatterRelayCount(uint64_t count)
{
    return (count * SCATTER_RELAY_WEIGHT + SCATTER_SPLIT_WEIGHT - 1) / SCATTER_SPLIT_WEIGHT;
}

inline uint64_t ScatterChunkCount(uint64_t bufferBytes)
{
    uint64_t low = 0;
    uint64_t high = std::min(bufferBytes, SCATTER_MAX_TRANSFER) / sizeof(float);
    while (low < high) {
        const uint64_t count = low + (high - low + 1) / 2;
        const uint64_t aCount = ScatterRelayCount(count);
        const uint64_t required
            = 2 * ScatterAlignUp(aCount * sizeof(float)) + ScatterAlignUp((count - aCount) * sizeof(float));
        if (required <= bufferBytes) {
            low = count;
        } else {
            high = count - 1;
        }
    }
    return low;
}

#endif
