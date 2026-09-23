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

#include <cstdint>
#include <type_traits>
#include "common.h"

constexpr uint32_t SCATTER_MAX_RANKS = 32;
constexpr uint64_t SCATTER_TRANSFER_BYTES = 256ULL * 1024 * 1024;
constexpr uint32_t SCATTER_CHANNEL_NOTIFY_NUM = 2;
constexpr uint64_t SCATTER_OVERLAP_BYTES = 1024ULL * 1024;
constexpr uint64_t SCATTER_STAGING_BYTES = 64ULL * 1024 * 1024;

enum class ScatterCopyMode : uint64_t { NONE = 0, AFTER_READS = 1, WITH_READS = 2 };

enum class ScatterPhase : uint32_t {
    SMALL_FULL, SMALL_PUBLISH, SMALL_FINISH, PUBLISH, STAGE, READ_OWN,
    READ_PROVIDERS, DRAIN, FINISH, ROOT_WAIT, ROOT_RELEASE
};
inline bool ScatterIsLocal(uint32_t mask, uint32_t rank) { return (mask & (uint32_t(1) << rank)) != 0; }
inline uint32_t ScatterLocalCount(uint32_t mask, uint32_t size)
{
    uint32_t count = 0;
    for (uint32_t rank = 0; rank < size; ++rank) {
        count += ScatterIsLocal(mask, rank);
    }
    return count;
}

struct ScatterKernelArg {
    ChannelHandle channels[SCATTER_MAX_RANKS]{};
    uint32_t rankId = 0;
    uint32_t rankSize = 0;
    uint32_t root = 0;
    uint32_t localMask = 0;
};

struct AlgResourceCtx {
    ChannelHandle channels[SCATTER_MAX_RANKS]{};
    uint32_t channelDie[SCATTER_MAX_RANKS]{};
    CcuKernelHandle kernels[SCATTER_MAX_RANKS][SCATTER_MAX_RANKS]{};
    uint32_t kernelCount[SCATTER_MAX_RANKS]{};
    uint32_t rootChannelGroup[SCATTER_MAX_RANKS]{};
    uint32_t smallGroupOrder[SCATTER_MAX_RANKS]{};
    uint32_t localMask = 0;
    bool failed = false;
    uint64_t stagingBase = 0;
    uint64_t stagingSize = 0;
    uint32_t rankId = 0;
    uint32_t rankSize = 0;
};
static_assert(std::is_trivially_copyable<AlgResourceCtx>::value, "Resource context must be plain data");

#endif // OPS_HCCL_CUSTOM_H
