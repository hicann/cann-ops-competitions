/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the LICENSE.
 */

#ifndef OPS_HCCL_CUSTOM_H
#define OPS_HCCL_CUSTOM_H

#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

#include <hccl/hccl_res.h>
#include <hccl/hccl_types.h>

#include "common.h"

constexpr uint32_t SCATTER_MAX_RANK_SIZE = 16;
constexpr uint32_t SCATTER_MAX_CHANNEL_NUM = 15;
constexpr uint32_t SCATTER_MAX_THREAD_NUM = 10;
constexpr uint64_t SCATTER_RESOURCE_MAGIC = 0x5343324456313641ULL; 

enum ScatterAlgo : uint32_t {
    SCATTER_ALGO_DIRECT = 0,
    SCATTER_ALGO_SMALL = 1,
    SCATTER_ALGO_2D = 2,
};

enum ScatterRole : uint32_t {
    SCATTER_ROLE_DIRECT = 0,
    SCATTER_ROLE_ROOT = 1,
    SCATTER_ROLE_LOCAL_RELAY = 2,
    SCATTER_ROLE_REMOTE_LEADER = 3,
    SCATTER_ROLE_REMOTE_LEAF = 4,
};

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

struct ChannelInfo {
    uint32_t remoteRank = INVALID_VALUE_RANKID;
    uint32_t notifyNum = 0;
    ChannelHandle handle = 0;
    CommBuffer remoteCclMem{};
};

// Fixed-size POD resource context: the immutable template AICPU kernel can
// deserialize it with one memcpy. The algorithm field separates the latency
// optimized 512 KiB path from the throughput optimized 2-D large-message path.
struct AlgResourceCtx {
    uint64_t magic = SCATTER_RESOURCE_MAGIC;
    uint32_t algo = SCATTER_ALGO_DIRECT;
    uint32_t role = SCATTER_ROLE_DIRECT;

    ThreadHandle aicpuThread{};
    CommBuffer localBuffer{};

    uint32_t threadCount = 0;
    uint32_t channelCount = 0;
    ThreadHandle threads[SCATTER_MAX_THREAD_NUM]{};
    ChannelInfo channels[SCATTER_MAX_CHANNEL_NUM]{};
    uint8_t channelIndex[SCATTER_MAX_RANK_SIZE]{};

    std::vector<char> Serialize() const
    {
        std::vector<char> result(sizeof(AlgResourceCtx));
        std::memcpy(result.data(), this, sizeof(AlgResourceCtx));
        return result;
    }

    void DeSerialize(std::vector<char> &data)
    {
        if (data.size() != sizeof(AlgResourceCtx)) {
            *this = AlgResourceCtx{};
            magic = 0;
            return;
        }
        std::memcpy(this, data.data(), sizeof(AlgResourceCtx));
    }

    bool IsValid() const
    {
        return magic == SCATTER_RESOURCE_MAGIC && threadCount > 0 && threadCount <= SCATTER_MAX_THREAD_NUM &&
            channelCount <= SCATTER_MAX_CHANNEL_NUM && aicpuThread == threads[0];
    }
};

static_assert(std::is_trivially_copyable<CommBuffer>::value, "CommBuffer must be trivially copyable");
static_assert(std::is_trivially_copyable<ChannelInfo>::value, "ChannelInfo must be trivially copyable");
static_assert(std::is_trivially_copyable<AlgResourceCtx>::value, "AlgResourceCtx must be trivially copyable");

#endif // OPS_HCCL_CUSTOM_H
