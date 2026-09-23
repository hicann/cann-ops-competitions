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

#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>

#include "binary_stream.h"
#include "common.h"

// Queue 0 coordinates Host/Device completion. For large cross-server slices,
// lane 0 carries the direct Clos region and lane 1 carries the Mesh-relay region.
constexpr uint32_t SCATTER_LANES = 2;
constexpr uint32_t SCATTER_WINDOW_DEPTH = 2;
constexpr uint32_t SCATTER_CHANNEL_NOTIFY_NUM = 4 * SCATTER_LANES * SCATTER_WINDOW_DEPTH;
constexpr uint64_t SCATTER_CHUNK_BYTES = 12ULL * 1024 * 1024;
constexpr uint64_t SCATTER_ONESHOT_BYTES = 16ULL * 1024 * 1024;
constexpr uint64_t SCATTER_HALVING_INPUT_BYTES = 512ULL * 1024;
constexpr uint64_t SCATTER_HYBRID_MIN_BYTES = 8ULL * 1024 * 1024;
constexpr uint32_t SCATTER_SERVER_RANKS = 8;
constexpr uint32_t SCATTER_RELAY_NUMERATOR = 3;
constexpr uint32_t SCATTER_ROUTE_DENOMINATOR = 8;

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
