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

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include <hccl/hccl_res.h>
#include <hccl/hccl_types.h>

#include "common.h"

template <typename T, std::size_t Capacity>
struct FixedVector {
    std::array<T, Capacity> values = {};
    uint32_t count = 0;

    void resize(std::size_t size)
    {
        count = static_cast<uint32_t>(size);
    }

    void push_back(const T &value)
    {
        values[count++] = value;
    }

    std::size_t size() const
    {
        return count;
    }

    T *data()
    {
        return values.data();
    }

    const T *data() const
    {
        return values.data();
    }

    T &operator[](std::size_t index)
    {
        return values[index];
    }

    const T &operator[](std::size_t index) const
    {
        return values[index];
    }

    auto begin()
    {
        return values.begin();
    }

    auto end()
    {
        return values.begin() + count;
    }

    auto begin() const
    {
        return values.begin();
    }

    auto end() const
    {
        return values.begin() + count;
    }
};

struct CommBuffer {
    void *addr = nullptr;
    uint64_t size = 0;
};

struct ChannelInfo {
    uint32_t remoteRank = INVALID_VALUE_RANKID;
    uint32_t notifyNum = 0;
    ChannelHandle handle = 0;
    CommBuffer remoteCclMem;
};

constexpr std::size_t MAX_ALG_THREADS = 8;
constexpr std::size_t MAX_ALG_CHANNELS = 8;
using ThreadCollection = FixedVector<ThreadHandle, MAX_ALG_THREADS>;
using ChannelCollection = FixedVector<ChannelInfo, MAX_ALG_CHANNELS>;

struct AlgResourceCtx {
    ThreadHandle aicpuThread = 0;
    CommBuffer localBuffer;
    uint64_t minCclBufferBytes = 0;
    ThreadCollection threads;
    ChannelCollection channels;

    bool DeSerialize(const void *data, uint64_t size)
    {
        if (data == nullptr || size != sizeof(*this)) {
            return false;
        }
        std::memcpy(this, data, sizeof(*this));
        return threads.size() <= MAX_ALG_THREADS && channels.size() <= MAX_ALG_CHANNELS;
    }
};

struct HostBaseResourceCtx {
    ThreadHandle aicpuThread = 0;
    uint32_t myRank = INVALID_VALUE_RANKID;
    uint32_t rankSize = 0;
};

struct HostStreamResourceCtx {
    ThreadHandle cpuThread = 0;
    ThreadHandle cpuThreadOnAicpu = 0;
    ThreadHandle aicpuThreadOnCpu = 0;
};

static_assert(std::is_trivially_copyable<AlgResourceCtx>::value,
    "AlgResourceCtx must remain trivially copyable");
static_assert(std::is_trivially_copyable<HostBaseResourceCtx>::value,
    "HostBaseResourceCtx must remain trivially copyable");
static_assert(std::is_trivially_copyable<HostStreamResourceCtx>::value,
    "HostStreamResourceCtx must remain trivially copyable");

#endif // OPS_HCCL_CUSTOM_H
