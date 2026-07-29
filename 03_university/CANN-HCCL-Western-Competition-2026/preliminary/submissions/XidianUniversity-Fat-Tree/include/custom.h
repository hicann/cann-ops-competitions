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

#include "common.h"

#include <cstring>
#include <vector>

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

struct ChannelInfo {
    uint32_t remoteRank;
    uint32_t netLayer;
    uint32_t notifyNum;
    ChannelHandle handle;
    CommBuffer remoteCclMem;
};

constexpr uint32_t MAX_ALG_THREAD_NUM = 16;
constexpr uint32_t MAX_ALG_CHANNEL_NUM = 15;

struct AlgResourceCtx {
    ThreadHandle aicpuThread;
    CommBuffer localBuffer;
    uint64_t minCclBufferSize = 0;
    uint32_t threadNum = 0;
    uint32_t channelNum = 0;
    ThreadHandle threads[MAX_ALG_THREAD_NUM];
    ChannelInfo channels[MAX_ALG_CHANNEL_NUM];

    // Serialize host-created communication resources into an AICPU engine context.
    std::vector<char> Serialize() const
    {
        size_t resultSize = sizeof(aicpuThread) + sizeof(localBuffer) + sizeof(minCclBufferSize) +
            sizeof(threadNum) + static_cast<size_t>(threadNum) * sizeof(threads[0]) + sizeof(channelNum) +
            static_cast<size_t>(channelNum) * sizeof(channels[0]);
        std::vector<char> result(resultSize);
        char *cursor = result.data();
        auto write = [&cursor](const void *source, size_t size) {
            std::memcpy(cursor, source, size);
            cursor += size;
        };
        write(&aicpuThread, sizeof(aicpuThread));
        write(&localBuffer, sizeof(localBuffer));
        write(&minCclBufferSize, sizeof(minCclBufferSize));
        write(&threadNum, sizeof(threadNum));
        write(threads, static_cast<size_t>(threadNum) * sizeof(threads[0]));
        write(&channelNum, sizeof(channelNum));
        write(channels, static_cast<size_t>(channelNum) * sizeof(channels[0]));
        return result;
    }

    // Restore communication resources from an AICPU engine context.
    void DeSerialize(const std::vector<char> &data)
    {
        const char *cursor = data.data();
        auto read = [&cursor](void *destination, size_t size) {
            std::memcpy(destination, cursor, size);
            cursor += size;
        };
        read(&aicpuThread, sizeof(aicpuThread));
        read(&localBuffer, sizeof(localBuffer));
        read(&minCclBufferSize, sizeof(minCclBufferSize));
        read(&threadNum, sizeof(threadNum));
        read(threads, static_cast<size_t>(threadNum) * sizeof(threads[0]));
        read(&channelNum, sizeof(channelNum));
        read(channels, static_cast<size_t>(channelNum) * sizeof(channels[0]));
    }
};

#endif // OPS_HCCL_CUSTOM_H
