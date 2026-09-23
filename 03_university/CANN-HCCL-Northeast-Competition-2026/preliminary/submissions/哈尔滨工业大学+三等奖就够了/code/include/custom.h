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

#include "binary_stream.h"
#include "common.h"

constexpr uint32_t SCATTER_CHANNEL_NOTIFY_NUM = 7;
constexpr uint32_t SCATTER_NOTIFY_CREDIT = 0;
constexpr uint32_t SCATTER_NOTIFY_DATA = 2;
constexpr uint32_t SCATTER_NOTIFY_FINISHED = 4;
constexpr uint32_t SCATTER_NOTIFY_SMALL_DATA = 5;
constexpr uint32_t SCATTER_NOTIFY_SMALL_CONSUMED = 6;
constexpr uint64_t SCATTER_SMALL_INPUT_BYTES = 16ULL * 1024 * 1024;
constexpr uint64_t SCATTER_PIPELINE_BYTES = 16ULL * 1024 * 1024;
constexpr uint64_t SCATTER_MAX_COPY_BYTES = 256ULL * 1024 * 1024;

inline uint32_t ScatterReceiveLanes(uint64_t receiverBufferBytes)
{
    return receiverBufferBytes >= 2 * sizeof(float) ? 2 : 1;
}

// Each outgoing channel owns a disjoint staging slot on the root. Compute the
// same capacity from the root's advertised buffer on both ends of the channel.
inline uint64_t ScatterSlotBytes(uint64_t rootBufferBytes, uint32_t rankSize)
{
    if (rankSize <= 1) {
        return 0;
    }
    const uint64_t bytes = std::min(SCATTER_PIPELINE_BYTES, rootBufferBytes / (rankSize - 1));
    return bytes / sizeof(float) * sizeof(float);
}

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
    std::vector<uint32_t> serverRanks; // Sorted members of this rank's lowest topology layer.

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << aicpuThread;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << channels;
        binaryStream << serverRanks;
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
        binaryStream >> serverRanks;
    }
};

#endif // OPS_HCCL_CUSTOM_H
