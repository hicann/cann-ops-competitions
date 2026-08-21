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
#include <vector>

#include "binary_stream.h"
#include "common.h"

namespace custom_broadcast {
constexpr uint32_t RANK_SIZE = 16;
constexpr uint32_t LOCAL_RANK_SIZE = 8;
constexpr uint32_t PEER_SIZE = RANK_SIZE - 1;
constexpr uint32_t INTRA_NET_LAYER = 0;
constexpr uint32_t INTER_NET_LAYER = 1;
constexpr uint64_t DATA_TYPE_SIZE = sizeof(float);
constexpr uint64_t PREFERRED_ALIGN = 512;
constexpr uint64_t BUTTERFLY_BYTES = 512ULL * 1024;

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

    const uint64_t commonBytes = AlignDown(totalBytes / RANK_SIZE, PREFERRED_ALIGN);
    slice.offsetBytes = static_cast<uint64_t>(rankIndex) * commonBytes;
    slice.validBytes = commonBytes;
    if (rankIndex == RANK_SIZE - 1) {
        slice.validBytes = totalBytes - slice.offsetBytes;
    }
    return slice;
}

inline SegmentDesc BuildSegment(const RankSlice &slice, uint64_t round, uint64_t tileBytes)
{
    SegmentDesc segment;
    if (tileBytes == 0 || round > slice.validBytes / tileBytes) {
        return segment;
    }
    const uint64_t tileOffset = round * tileBytes;
    if (tileOffset >= slice.validBytes) {
        return segment;
    }
    segment.offsetBytes = slice.offsetBytes + tileOffset;
    segment.validBytes = std::min(tileBytes, slice.validBytes - tileOffset);
    return segment;
}
} // namespace custom_broadcast

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
    std::vector<ChannelInfo> channels;

    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << aicpuThread;
        binaryStream << localBuffer;
        binaryStream << rankIndex;
        binaryStream << ranks;
        binaryStream << threads;
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
        binaryStream >> channels;
    }
};

#endif // OPS_HCCL_CUSTOM_H
