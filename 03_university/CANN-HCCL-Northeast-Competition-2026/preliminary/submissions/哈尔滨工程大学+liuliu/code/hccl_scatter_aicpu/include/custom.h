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

// Separate notifications prevent READY and CONSUMED from overwriting each other.
constexpr uint32_t SCATTER_NOTIFY_READY = 0;
constexpr uint32_t SCATTER_NOTIFY_DATA = 1;
constexpr uint32_t SCATTER_NOTIFY_CONSUMED = 2;
constexpr uint32_t SCATTER_CHANNEL_NOTIFY_NUM = 3;
constexpr uint64_t SCATTER_CHUNK_BYTES = 64ULL * 1024 * 1024;

// Pipelined path: credits [0, 1] at sender, data [0, 1] at receiver, release [2] at receiver.
// Serial and pipelined invocations drain all notifications before returning.
constexpr uint32_t SCATTER_PIPE_DATA_BASE = 0;
constexpr uint32_t SCATTER_PIPE_RELEASE = 2;
constexpr uint64_t SCATTER_PIPE_CHUNK_BYTES = 8ULL * 1024 * 1024;
constexpr uint64_t SCATTER_PARALLEL_THRESHOLD = 4096;
constexpr uint64_t SCATTER_BUFFER_ALIGNMENT = 512;

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

    // Canonical server IDs for the competition's two groups of eight ranks.
    // Empty outside the supported topology; group 0 contains communicator rank 0.
    std::vector<uint32_t> serverGroup;

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << aicpuThread;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << channels;
        binaryStream << serverGroup;
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
        binaryStream >> serverGroup;
    }
};

#endif // OPS_HCCL_CUSTOM_H
