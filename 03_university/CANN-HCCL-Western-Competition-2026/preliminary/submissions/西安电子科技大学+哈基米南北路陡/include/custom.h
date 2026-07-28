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

#include <hccl/hccl_res.h>
#include <hccl/hccl_types.h>

#include <algorithm>

#include "binary_stream.h"
#include "common.h"

namespace custom_allreduce {
constexpr uint32_t RANK_SIZE = 16;
constexpr uint32_t LOCAL_RANK_SIZE = 8;
constexpr uint32_t PEER_SIZE = RANK_SIZE - 1;
constexpr uint64_t DATA_TYPE_SIZE = sizeof(float);
constexpr uint64_t CORRECTNESS_ALIGN = DATA_TYPE_SIZE;
constexpr uint64_t PREFERRED_ALIGN = 512;
constexpr uint64_t BULK_GRANULARITY = PREFERRED_ALIGN;
constexpr uint64_t SMALL_THRESHOLD_BYTES = 8ULL * 1024 * 1024;
constexpr uint64_t TILE_16_MIB = 16ULL * 1024 * 1024;
constexpr uint64_t TILE_8_MIB = 8ULL * 1024 * 1024;
constexpr uint64_t TILE_5_MIB = 5ULL * 1024 * 1024;
constexpr uint64_t TILE_4_MIB = 4ULL * 1024 * 1024;
static_assert(BULK_GRANULARITY % DATA_TYPE_SIZE == 0, "Rank slices must preserve FP32 element boundaries");
static_assert(TILE_16_MIB % PREFERRED_ALIGN == 0, "Tile size must preserve the preferred alignment");

enum class AlgorithmType : uint32_t {
    OneShot16 = 0,
    DirectAuto = 1,
    DirectWhole = 2,
    Direct16MiB = 3,
    Direct8MiB = 4,
    Direct5MiB = 5,
    Direct4MiB = 6,
};

constexpr uint32_t CONFIG_ALGORITHM_MASK = 0xFFU;
constexpr uint32_t CONFIG_WORKER_SHIFT = 8U;
constexpr uint32_t CONFIG_WORKER_MASK = 0xFFU;

inline uint32_t PackKernelConfig(AlgorithmType algorithm, uint32_t workerCount)
{
    return (workerCount & CONFIG_WORKER_MASK) << CONFIG_WORKER_SHIFT
        | (static_cast<uint32_t>(algorithm) & CONFIG_ALGORITHM_MASK);
}

inline AlgorithmType DecodeAlgorithm(uint32_t config)
{
    return static_cast<AlgorithmType>(config & CONFIG_ALGORITHM_MASK);
}

inline uint32_t DecodeWorkerCount(uint32_t config)
{
    return config >> CONFIG_WORKER_SHIFT & CONFIG_WORKER_MASK;
}

struct RankSlice {
    uint64_t offsetBytes = 0;
    uint64_t validBytes = 0;
};

struct SegmentDesc {
    uint64_t offsetBytes = 0;
    uint64_t validBytes = 0;
};

inline uint64_t AlignDown(uint64_t value, uint64_t alignment)
{
    return alignment == 0 ? value : value / alignment * alignment;
}

inline uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return alignment == 0 || value == 0 ? value : (value - 1) / alignment * alignment + alignment;
}

inline uint64_t DivUp(uint64_t value, uint64_t divisor)
{
    return divisor == 0 || value == 0 ? 0 : (value - 1) / divisor + 1;
}

inline RankSlice BuildRankSlice(uint64_t totalBytes, uint32_t rankIndex)
{
    RankSlice slice;
    if (rankIndex >= RANK_SIZE) {
        return slice;
    }

    const uint64_t commonBytes = AlignDown(totalBytes / RANK_SIZE, BULK_GRANULARITY);
    slice.offsetBytes = static_cast<uint64_t>(rankIndex) * commonBytes;
    slice.validBytes = commonBytes;
    if (rankIndex == RANK_SIZE - 1) {
        slice.validBytes = totalBytes - slice.offsetBytes;
    }
    return slice;
}

inline SegmentDesc BuildSegment(const RankSlice &slice, uint64_t round, uint64_t roundCount, uint64_t tileBytes)
{
    SegmentDesc segment;
    const uint64_t tileOffset = round * tileBytes;
    if (round >= roundCount || tileOffset >= slice.validBytes) {
        return segment;
    }
    segment.offsetBytes = slice.offsetBytes + tileOffset;
    segment.validBytes
        = round + 1 == roundCount ? slice.validBytes - tileOffset : std::min(tileBytes, slice.validBytes - tileOffset);
    return segment;
}
} // namespace custom_allreduce

struct CommBuffer {
    void *addr = nullptr;
    uint64_t size = 0;
};

struct ChannelInfo {
    uint32_t remoteRank = INVALID_VALUE_RANKID;
    uint32_t remoteRankIndex = INVALID_VALUE_RANKID;
    uint32_t netLayer = INVALID_VALUE_RANKID;
    uint32_t notifyNum = 0;
    ChannelHandle handle = 0;
    CommBuffer remoteCclMem;
};

struct AlgResourceCtx {
    ThreadHandle aicpuThread;
    CommBuffer localBuffer;
    uint32_t rankIndex = INVALID_VALUE_RANKID;
    std::vector<uint32_t> ranks;
    std::vector<ThreadHandle> threads;
    std::vector<ThreadHandle> workerThreads;
    std::vector<ChannelInfo> channels;

    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << aicpuThread;
        binaryStream << localBuffer;
        binaryStream << rankIndex;
        binaryStream << ranks;
        binaryStream << threads;
        binaryStream << workerThreads;
        binaryStream << channels;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> aicpuThread;
        binaryStream >> localBuffer;
        binaryStream >> rankIndex;
        binaryStream >> ranks;
        binaryStream >> threads;
        binaryStream >> workerThreads;
        binaryStream >> channels;
    }
};

#endif // OPS_HCCL_CUSTOM_H
