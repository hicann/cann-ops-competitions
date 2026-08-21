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

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

struct ChannelInfo {
    uint32_t remoteRank = INVALID_VALUE_RANKID;
    uint32_t notifyNum = 0;
    ChannelHandle handle = 0;
    CommBuffer remoteCclMem{};
};

constexpr uint32_t REDUCE_SCATTER_RANK_SIZE = 16;
constexpr uint32_t REDUCE_SCATTER_CHANNEL_NUM = REDUCE_SCATTER_RANK_SIZE - 1;
constexpr uint32_t REDUCE_SCATTER_CHANNEL_NOTIFY_NUM = 3;
// Thread 0 schedules the operation; threads 1..7 drive the seven intra-server
// Mesh peers; thread 8 pipelines the next cross-server slice; threads 9..16
// pack its eight input blocks in parallel.
constexpr uint32_t REDUCE_SCATTER_AICPU_THREAD_NUM = 17;
constexpr uint32_t REDUCE_SCATTER_THREAD_NOTIFY_NUM = 9;

// This context is copied verbatim into AICPU engine memory. Do not put dynamic containers here.
struct AlgResourceCtx {
    // Thread 0 is the control/reduction thread. Threads 1..7 execute the seven
    // Mesh transfers. Thread 8 drives pipelined cross-server work, and threads
    // 9..16 pack the eight output-rank blocks.
    ThreadHandle aicpuThreads[REDUCE_SCATTER_AICPU_THREAD_NUM] = {};
    CommBuffer localBuffer{};
    ChannelInfo channels[REDUCE_SCATTER_CHANNEL_NUM];
};

#endif // OPS_HCCL_CUSTOM_H
