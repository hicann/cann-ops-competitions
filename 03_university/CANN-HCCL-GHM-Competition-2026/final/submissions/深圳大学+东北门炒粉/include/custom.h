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

#include <type_traits>

#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>

#include "common.h"

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE]{};
    uint32_t channelCount = 0;
};

constexpr uint32_t MAX_CHANNEL_GROUP_COUNT = 2;
constexpr uint32_t MAX_KERNEL_PHASE_COUNT = 2;
constexpr uint32_t MAX_GROUP_KERNEL_COUNT = MAX_CHANNEL_GROUP_COUNT * MAX_KERNEL_PHASE_COUNT;

struct AlgResourceCtx {
    CommBuffer localBuffer; ///< 本端HCCL通信内存
    uint64_t minBufferSize = 0;
    uint64_t scratchToken = 0;
    uint32_t rankId = 0;
    uint32_t rankSize = 0;
    ThreadHandle subThreads[MAX_CHANNEL_GROUP_COUNT - 1]{};
    uint32_t subThreadCount = 0;
    uint32_t channelGroupCount = 0;
    uint32_t channelGroupSizes[MAX_CHANNEL_GROUP_COUNT]{};
    CcuKernelHandle pullKernels[MAX_GROUP_KERNEL_COUNT]{};
    uint32_t pullKernelCount = 0;
    CcuKernelHandle stripeKernels[MAX_GROUP_KERNEL_COUNT]{};
    uint32_t stripeKernelCount = 0;
    CcuKernelHandle releaseKernels[MAX_CHANNEL_GROUP_COUNT]{};
    uint32_t releaseKernelCount = 0;
    CcuKernelHandle fusedKernel = 0;
    CcuKernelHandle reduceKernel = 0;
    CcuKernelHandle stripeReduceKernel = 0;
};

static_assert(std::is_trivially_copyable<AlgResourceCtx>::value,
    "AlgResourceCtx must remain safe for engine-context copies");

#endif // OPS_HCCL_CUSTOM_H
