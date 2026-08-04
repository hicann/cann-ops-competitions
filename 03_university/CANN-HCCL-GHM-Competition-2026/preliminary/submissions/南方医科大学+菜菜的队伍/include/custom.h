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

#include "binary_stream.h"
#include "common.h"

namespace yga_reduce_scatter {
constexpr uint32_t RANK_COUNT = 16;
constexpr uint32_t RANKS_PER_SERVER = 8;
constexpr uint32_t PEER_COUNT = RANK_COUNT - 1;
constexpr uint64_t FP32_BYTES = sizeof(float);
constexpr uint64_t DMA_ALIGNMENT = 512;
constexpr uint64_t HALF_MEG_SLICE_BYTES = 512ULL * 1024;
constexpr uint64_t TARGET_TILE_BYTES = 16ULL * 1024 * 1024;
constexpr uint64_t SMALL_TAIL_BYTES = 1024ULL * 1024;

static_assert(TARGET_TILE_BYTES % DMA_ALIGNMENT == 0, "Tile size must satisfy the DMA alignment");
static_assert(RANK_COUNT == 2U * RANKS_PER_SERVER, "The optimized schedule requires two equal servers");

inline uint64_t AlignDown(uint64_t value, uint64_t alignment)
{
    return alignment == 0 ? value : value / alignment * alignment;
}

inline uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return alignment == 0 || value == 0 ? value : (value - 1) / alignment * alignment + alignment;
}

inline uint64_t DivideRoundUp(uint64_t value, uint64_t divisor)
{
    return divisor == 0 || value == 0 ? 0 : (value - 1) / divisor + 1;
}
} // namespace yga_reduce_scatter

struct BufferSpan {
    void *address = nullptr;
    uint64_t bytes = 0;
};

struct PeerPath {
    uint32_t rankId = INVALID_VALUE_RANKID;
    uint32_t rankIndex = INVALID_VALUE_RANKID;
    uint32_t fabricLayer = INVALID_VALUE_RANKID;
    uint32_t notifyCount = 0;
    ChannelHandle channel = 0;
    BufferSpan remoteWorkspace;
};

struct AlgResourceCtx {
    ThreadHandle aicpuThread;
    BufferSpan workspace;
    uint32_t selfIndex = INVALID_VALUE_RANKID;
    std::vector<uint32_t> orderedRanks;
    std::vector<ThreadHandle> workerThreads;
    std::vector<PeerPath> peerPaths;

    std::vector<char> Serialize()
    {
        BinaryStream stream;
        stream << aicpuThread;
        stream << workspace;
        stream << selfIndex;
        stream << orderedRanks;
        stream << workerThreads;
        stream << peerPaths;
        std::vector<char> encoded;
        stream.Dump(encoded);
        return encoded;
    }

    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream stream(data);
        stream >> aicpuThread;
        stream >> workspace;
        stream >> selfIndex;
        stream >> orderedRanks;
        stream >> workerThreads;
        stream >> peerPaths;
    }
};

#endif // OPS_HCCL_CUSTOM_H
