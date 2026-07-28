/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

// ===== 提交版本 v63-final | v60安全基线 + peer就绪流水规约 | 2026-07-24 =====
// v63-final: notify数量和槽位完全保持v60，仅缩短peer就绪后的等待路径
// custom.h: 无代码改动 — 与 v44 结构一致

#ifndef OPS_HCCL_CUSTOM_H
#define OPS_HCCL_CUSTOM_H

#include <memory>
#include <vector>
#include <cstdint>
#include <cstring>

#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>
#include <hccl/hccl_ccu_res.h>

#include "binary_stream.h"
#include "common.h"

// 评测空工程的 MAX_RANK_SIZE 仍为 16，但决赛拓扑包含 32 个 rank。
// 使用独立容量，避免单个 die 分组或回退分组写越界。
constexpr uint32_t ALLREDUCE_MAX_CHANNELS = 32;

#ifndef HCCL_CCU_KERNEL_ARG_BASE_DEFINED
#define HCCL_CCU_KERNEL_ARG_BASE_DEFINED
struct CcuKernelArgBase {
    ChannelHandle channels[ALLREDUCE_MAX_CHANNELS];
    uint32_t channelCount;
};
#endif // HCCL_CCU_KERNEL_ARG_BASE_DEFINED

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

inline BinaryStream &operator<<(BinaryStream &bs, const CommBuffer &buf)
{
    bs << reinterpret_cast<uint64_t>(buf.addr);
    bs << buf.size;
    return bs;
}

inline BinaryStream &operator>>(BinaryStream &bs, CommBuffer &buf)
{
    uint64_t addr = 0;
    bs >> addr;
    bs >> buf.size;
    buf.addr = reinterpret_cast<void *>(addr);
    return bs;
}

struct AlgResourceCtx {
    CommBuffer cclMem;
    uint32_t notifyNumOnMainThread = 0;
    std::vector<ThreadHandle> threads;
    std::vector<CcuKernelHandle> ccuKernels;

    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << cclMem;
        binaryStream << notifyNumOnMainThread;
        binaryStream << threads;
        binaryStream << ccuKernels;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> cclMem;
        binaryStream >> notifyNumOnMainThread;
        binaryStream >> threads;
        binaryStream >> ccuKernels;
    }
};

#endif // OPS_HCCL_CUSTOM_H
