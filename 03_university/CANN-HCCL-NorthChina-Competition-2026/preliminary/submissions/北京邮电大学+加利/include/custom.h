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

#include <type_traits>

#include "common.h"

constexpr uint32_t ALG_MAX_THREAD_NUM = 16;
constexpr uint32_t ALG_MAX_CHANNEL_NUM = 15;
constexpr uint32_t ALG_HOST_TO_AICPU_NOTIFY_IDX = 31;

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

struct ChannelInfo {
    uint32_t remoteRank = INVALID_VALUE_RANKID;
    uint32_t notifyNum = 0;
    ChannelHandle handle = 0;
    CommBuffer remoteCclMem;
    CommBuffer remoteInputMem;
    CommBuffer remoteOutputMem;
};

struct AlgResourceCtx {
    ThreadHandle aicpuThread = 0;       ///< AICPU_TS通信引擎上的thread资源
    CommBuffer localBuffer = {};        ///< 本端HCCL通信内存
    CommBuffer registeredInput = {};    ///< 建链时交换的本端用户输入
    CommBuffer registeredOutput = {};   ///< 建链时交换的本端用户输出
    uint32_t localRankCount = 0;       ///< 每台Server中的rank数
    uint32_t threadNum = 0;
    uint32_t channelNum = 0;
    uint32_t reserved = 0;
    ThreadHandle threads[ALG_MAX_THREAD_NUM] = {}; ///< AICPU_TS通信引擎上的thread资源
    ChannelInfo channels[ALG_MAX_CHANNEL_NUM] = {}; ///< AICPU_TS通信引擎上的channel资源

};

static_assert(std::is_trivially_copyable<AlgResourceCtx>::value,
              "AlgResourceCtx must be copied byte-for-byte into the AICPU engine context");
static_assert(sizeof(AlgResourceCtx) == 1160,
              "Host and AICPU must agree on the fixed resource-context ABI");

#endif // OPS_HCCL_CUSTOM_H
