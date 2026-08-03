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

#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>

#include <array>
#include <cstring>

#include "binary_stream.h"
#include "common.h"

// Legacy generic-path constants remain available for non-competition sizes.
constexpr uint32_t CUSTOM_PARALLEL_REDUCE_EXTRA_THREAD_NUM = 3;
constexpr uint64_t CUSTOM_LARGE_MESSAGE_THRESHOLD_BYTES = 16ULL * 1024ULL * 1024ULL;
constexpr uint32_t CUSTOM_COMPETITION_GROUP_RANK_NUM = 8;
// The guarded 16-rank large-message path separates transport from reduction:
// 15 peer queues keep every physical link active, while eight additional
// queues execute the 8 -> 4 -> 2 -> 1 reduction tree over two receive banks.
constexpr uint32_t CUSTOM_LARGE_TREE_WORKER_NUM = 8;
constexpr uint32_t CUSTOM_LARGE_TREE_THREAD_NUM =
    1 + (2 * CUSTOM_COMPETITION_GROUP_RANK_NUM - 1) + CUSTOM_LARGE_TREE_WORKER_NUM;
constexpr uint32_t CUSTOM_LARGE_TREE_CHANNEL_NOTIFY_NUM = 4;
// The exact 512 KB competition case uses a four-round recursive-halving path.
// One AICPU queue visits the four XOR partners in topology-weighted order:
// the high-bandwidth cross-server dimension first, followed by three local
// full-mesh dimensions.
constexpr uint32_t CUSTOM_SMALL_RH_STEP_NUM = 4;
constexpr uint32_t CUSTOM_SMALL_RH_THREAD_NUM = 1;
constexpr uint32_t CUSTOM_SMALL_RH_CHANNEL_NOTIFY_NUM = 2;
constexpr uint32_t CUSTOM_SMALL_RH_PARTNER_DELTAS[CUSTOM_SMALL_RH_STEP_NUM] = {8, 4, 2, 1};
constexpr uint32_t CUSTOM_SMALL_COMM_WORKER_NUM = 8;
// The competition rank table places ranks 0..7 and 8..15 in two active
// eight-device network groups. This is used only by the guarded 16-rank small
// path; other rank layouts retain generic round-robin channel assignment.
constexpr uint64_t CUSTOM_COMPETITION_512KB_RECV_COUNT = 8192;
constexpr uint64_t CUSTOM_COMPETITION_512MB_RECV_COUNT = 8388608;
constexpr uint64_t CUSTOM_COMPETITION_400MB_4B_RECV_COUNT = 6553601;
constexpr uint32_t CUSTOM_ROLLING_STRIPE_NUM = 8;
constexpr uint32_t CUSTOM_ROLLING_CHANNEL_NOTIFY_NUM = 17;
constexpr uint32_t CUSTOM_MAX_RANK_NUM = 2 * CUSTOM_COMPETITION_GROUP_RANK_NUM;
constexpr uint32_t CUSTOM_MAX_THREAD_NUM = CUSTOM_LARGE_TREE_THREAD_NUM;
constexpr uint32_t CUSTOM_MAX_CHANNEL_NUM = CUSTOM_MAX_RANK_NUM - 1;

// Resource counts are bounded by the 16-rank competition contract. A fixed
// container keeps the host-side acquisition API convenient while avoiding
// heap allocation when the AICPU kernel decodes the cached resource context.
template <typename T, size_t Capacity>
class FixedResourceList {
public:
    size_t size() const
    {
        return size_;
    }

    void reserve(size_t)
    {
    }

    void resize(size_t size)
    {
        size_ = size <= Capacity ? size : Capacity;
    }

    void clear()
    {
        size_ = 0;
    }

    void push_back(const T &value)
    {
        if (size_ < Capacity) {
            values_[size_++] = value;
        }
    }

    T *data()
    {
        return values_.data();
    }

    const T *data() const
    {
        return values_.data();
    }

    T &operator[](size_t index)
    {
        return values_[index];
    }

    const T &operator[](size_t index) const
    {
        return values_[index];
    }

    T *begin()
    {
        return values_.data();
    }

    const T *begin() const
    {
        return values_.data();
    }

    T *end()
    {
        return values_.data() + size_;
    }

    const T *end() const
    {
        return values_.data() + size_;
    }

private:
    std::array<T, Capacity> values_{};
    size_t size_ = 0;
};

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

struct ChannelInfo {
    uint32_t remoteRank = INVALID_VALUE_RANKID;
    uint32_t notifyNum = 0;
    ChannelHandle handle = 0;
    CommBuffer remoteCclMem;
};

struct AlgResourceCtx {
    ThreadHandle aicpuThread;          ///< AICPU_TS通信引擎上的thread资源
    CommBuffer localBuffer;            ///< 本端HCCL通信内存
    FixedResourceList<ThreadHandle, CUSTOM_MAX_THREAD_NUM> threads;
    FixedResourceList<ChannelInfo, CUSTOM_MAX_CHANNEL_NUM> channels;

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << aicpuThread;
        binaryStream << localBuffer;
        const size_t threadNum = threads.size();
        binaryStream << threadNum;
        for (const ThreadHandle &thread : threads) {
            binaryStream << thread;
        }
        const size_t channelNum = channels.size();
        binaryStream << channelNum;
        for (const ChannelInfo &channel : channels) {
            binaryStream << channel;
        }
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    // Directly decode the existing BinaryStream wire format from EngineCtx.
    // This removes a vector copy, a stringstream buffer, and two device-side
    // vector allocations from every collective invocation.
    bool DeSerialize(const void *data, uint64_t dataSize)
    {
        const uint8_t *cursor = static_cast<const uint8_t *>(data);
        uint64_t remaining = dataSize;
        size_t threadNum = 0;
        size_t channelNum = 0;
        if (!ReadValue(cursor, remaining, aicpuThread)
            || !ReadValue(cursor, remaining, localBuffer)
            || !ReadValue(cursor, remaining, threadNum)
            || threadNum > CUSTOM_MAX_THREAD_NUM) {
            return false;
        }
        threads.resize(threadNum);
        for (size_t idx = 0; idx < threadNum; ++idx) {
            if (!ReadValue(cursor, remaining, threads[idx])) {
                threads.clear();
                return false;
            }
        }
        if (!ReadValue(cursor, remaining, channelNum)
            || channelNum > CUSTOM_MAX_CHANNEL_NUM) {
            threads.clear();
            return false;
        }
        channels.resize(channelNum);
        for (size_t idx = 0; idx < channelNum; ++idx) {
            if (!ReadValue(cursor, remaining, channels[idx])) {
                threads.clear();
                channels.clear();
                return false;
            }
        }
        return remaining == 0;
    }

private:
    template <typename T>
    static bool ReadValue(const uint8_t *&cursor, uint64_t &remaining, T &value)
    {
        if (remaining < sizeof(T)) {
            return false;
        }
        std::memcpy(&value, cursor, sizeof(T));
        cursor += sizeof(T);
        remaining -= sizeof(T);
        return true;
    }
};

#endif // OPS_HCCL_CUSTOM_H
