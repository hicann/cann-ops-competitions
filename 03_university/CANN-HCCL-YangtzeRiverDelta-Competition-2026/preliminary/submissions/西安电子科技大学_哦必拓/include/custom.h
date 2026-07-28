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

constexpr uint32_t BROADCAST_RANK_SIZE = 16;
constexpr uint32_t BROADCAST_SERVER_RANKS = 8;
constexpr uint32_t BROADCAST_CHANNEL_NOTIFY_COUNT = 6;
constexpr uint32_t BROADCAST_THREAD_COUNT = 15;
constexpr uint32_t BROADCAST_THREAD_NOTIFY_COUNT = 15;
constexpr uint32_t BROADCAST_WORKER_START_NOTIFY = 0;
constexpr uint32_t BROADCAST_PIECE0_READY_NOTIFY = 1;
constexpr uint32_t BROADCAST_PIECE1_READY_NOTIFY = 2;
constexpr uint32_t BROADCAST_SIGNALS_PER_DIRECTION = 3;

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
    ThreadHandle aicpuThread;          ///< Coordinator exported to the CPU engine.
    CommBuffer localBuffer;            ///< Local CCL staging memory.
    std::vector<ThreadHandle> threads; ///< Coordinator followed by transfer workers.
    std::vector<ChannelInfo> channels; ///< Channels kept for context reuse.

    // Serialize and deserialize in exactly the same field order.
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