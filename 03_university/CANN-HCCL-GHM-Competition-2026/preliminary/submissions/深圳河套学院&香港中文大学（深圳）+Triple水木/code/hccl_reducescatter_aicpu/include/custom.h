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

#include <cstdint>
#include <vector>

#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>

#include "binary_stream.h"
#include "common.h"

namespace custom_rs {

constexpr uint32_t SERVER_NUM = 2;
constexpr uint32_t RANK_NUM_PER_SERVER = 8;
constexpr uint32_t RANK_SIZE = SERVER_NUM * RANK_NUM_PER_SERVER;
constexpr uint32_t PIPELINE_BUFFER_NUM = 2;
constexpr uint32_t SCRATCH_SLOT_NUM_PER_BUFFER = RANK_SIZE;
constexpr uint32_t SCRATCH_SLOT_NUM = PIPELINE_BUFFER_NUM * SCRATCH_SLOT_NUM_PER_BUFFER;
constexpr uint32_t COORDINATOR_THREAD_INDEX = 0;
constexpr uint32_t COMM_THREAD_BEGIN = COORDINATOR_THREAD_INDEX + 1;
constexpr uint32_t COMM_THREAD_NUM = RANK_SIZE - 1;
constexpr uint32_t REDUCE_THREAD_BEGIN = COMM_THREAD_BEGIN + COMM_THREAD_NUM;
constexpr uint32_t REDUCE_GROUP_NUM = 2;
constexpr uint32_t REDUCE_MAIN_THREAD_INDEX = REDUCE_THREAD_BEGIN;
constexpr uint32_t REDUCE_GROUP_THREAD_BEGIN = REDUCE_MAIN_THREAD_INDEX + 1;
constexpr uint32_t REDUCE_THREAD_NUM = 1 + REDUCE_GROUP_NUM;
constexpr uint32_t DIRECT_THREAD_NUM = REDUCE_THREAD_BEGIN + REDUCE_THREAD_NUM;
constexpr uint32_t DIRECT_THREAD_NOTIFY_NUM = 21;
constexpr uint32_t DIRECT_CHANNEL_NUM = RANK_SIZE - 1;
constexpr uint32_t RECURSIVE_HALVING_ROUND_NUM = 4;
constexpr uint32_t RECURSIVE_HALVING_BUFFER_SLOT_NUM = RANK_SIZE;
constexpr uint32_t DIRECT_CHANNEL_NOTIFY_NUM = 3;

constexpr uint32_t NOTIFY_READY = 0;
constexpr uint32_t NOTIFY_DATA_BEGIN = 1;
// Competition case sizes are the aggregate input bytes of all sixteen ranks.
// 512 KiB therefore corresponds to a 32 KiB output slice on each rank.
constexpr uint64_t SMALL_INPUT_BYTES = 512ULL * 1024;
constexpr uint64_t SMALL_DATA_BYTES = SMALL_INPUT_BYTES / RANK_SIZE;
constexpr uint64_t MIN_SLICE_ALIGN = 4ULL * 1024;
constexpr uint64_t MAX_TRANSFER_BYTES = 256ULL * 1024 * 1024;

static_assert(RANK_SIZE == 16, "The competition topology must contain 16 ranks");
static_assert(SCRATCH_SLOT_NUM == 32, "The direct A2A pipeline needs two sixteen-slot buffers");
static_assert(COMM_THREAD_NUM == DIRECT_CHANNEL_NUM,
    "Each peer channel needs an independent communication thread");
static_assert(DIRECT_THREAD_NUM == 19,
    "The direct A2A pipeline needs one coordinator, fifteen communication threads and three reduce threads");
static_assert(NOTIFY_DATA_BEGIN + PIPELINE_BUFFER_NUM <= DIRECT_CHANNEL_NOTIFY_NUM,
    "Each direct-A2A pipeline buffer needs an independent channel data notify");
static_assert((1U << RECURSIVE_HALVING_ROUND_NUM) == RANK_SIZE,
    "Recursive halving requires a power-of-two rank size");
static_assert(RECURSIVE_HALVING_BUFFER_SLOT_NUM == RANK_SIZE,
    "Read-reduce recursive halving only needs the sixteen input slices");
static_assert(SMALL_INPUT_BYTES % RANK_SIZE == 0,
    "The aggregate small-data threshold must be divisible by rank size");

inline bool UseRecursiveHalving(uint64_t bytes)
{
    return bytes <= SMALL_DATA_BYTES;
}

inline bool UseCompactTileStride(uint64_t bytes)
{
    return !UseRecursiveHalving(bytes) && bytes % MIN_SLICE_ALIGN == 0;
}

inline uint32_t GetServerIndex(uint32_t rank)
{
    return rank / RANK_NUM_PER_SERVER;
}

inline uint32_t GetNetLayer(uint32_t srcRank, uint32_t dstRank)
{
    return GetServerIndex(srcRank) == GetServerIndex(dstRank) ? 0 : 1;
}

inline uint32_t GetChannelIndex(uint32_t myRank, uint32_t remoteRank)
{
    return remoteRank < myRank ? remoteRank : remoteRank - 1;
}

inline uint32_t GetCommThreadIndex(uint32_t myRank, uint32_t remoteRank)
{
    return COMM_THREAD_BEGIN + GetChannelIndex(myRank, remoteRank);
}

inline uint32_t GetRecursiveHalvingMask(uint32_t round)
{
    return RANK_SIZE >> (round + 1);
}

inline uint32_t GetRecursiveHalvingPeer(uint32_t rank, uint32_t round)
{
    return rank ^ GetRecursiveHalvingMask(round);
}

inline uint64_t AlignDown(uint64_t value, uint64_t alignment)
{
    return value / alignment * alignment;
}

inline uint64_t ScratchOffset(uint32_t bufferIndex, uint32_t sourceRank,
    uint64_t bankSlotStride, uint64_t tileSlotStride)
{
    const uint64_t bankOffset = static_cast<uint64_t>(bufferIndex) *
        SCRATCH_SLOT_NUM_PER_BUFFER * bankSlotStride;
    return bankOffset + static_cast<uint64_t>(sourceRank) * tileSlotStride;
}

} // namespace custom_rs

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
    std::vector<ThreadHandle> threads; ///< AICPU_TS通信引擎上的thread资源
    std::vector<ChannelInfo> channels; ///< AICPU_TS通信引擎上的channel资源

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << aicpuThread;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << channels;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    // 反序列化
    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> aicpuThread;
        binaryStream >> localBuffer;
        binaryStream >> threads;
        binaryStream >> channels;
    }
};

#endif // OPS_HCCL_CUSTOM_H
