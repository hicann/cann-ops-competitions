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

#include <hccl/hccl_res.h>
#include <hccl/hccl_types.h>

#include "binary_stream.h"
#include "common.h"

// 综合候选共享同一套经过验证的资源与 notify 框架，按输入大小选择执行路径。
// 数值必须是预处理器常量，因为 Host 与 AICPU Kernel 分别编译。
#define RS_CANDIDATE_FLATTEN 0
#define RS_CANDIDATE_PAIR_MESH 1
#define RS_CANDIDATE_DUAL_PATH 2
#define HCCL_RS_CANDIDATE_MODE RS_CANDIDATE_FLATTEN
#define HCCL_RS_VARIANT_TAG "v148g2a"
#define HCCL_RS_PERSISTENT_PACKED_HANDOFF 0
#define HCCL_RS_PERSISTENT_REFERENCE_HANDOFF 0
#define RS_TINY_MODE_XOR_HALF_COPY 0
#define RS_TINY_MODE_DIRECT_15_WORKER 1
#define RS_TINY_MODE_HIERARCHY_SEQUENTIAL 2
#define RS_TINY_MODE_HIERARCHY_TREE 3
#define HCCL_RS_TINY_MODE RS_TINY_MODE_XOR_HALF_COPY
#define HCCL_RS_DEDICATED_PACKED_REDUCE_PIPELINE 0
#define HCCL_RS_DEDICATED_REFERENCE_REDUCE_PIPELINE 0

// Host 资源选择与 AICPU 算法选择必须共享同一个题面总输入边界。
// 8 MiB 以下使用拓扑对齐 XOR 递归折半；8 MiB 起沿用 V12 大包路径。
constexpr uint64_t RS_WAVEFRONT_INPUT_MIN_BYTES = 512ULL * 1024;
constexpr uint64_t RS_NHR_INPUT_LIMIT_BYTES = 8ULL * 1024 * 1024;
// 已有参考真值证明四条带+配对规约对单outer的400 MiB坐标有效；只在
// packed阈值以下启用，避免覆盖512 MiB的单outer packed搜索分支。
constexpr uint64_t RS_REFERENCE_V2_INPUT_MIN_BYTES = 384ULL * 1024 * 1024;
// 只在 512 MiB 性能坐标启用 packed-stripe；400 MiB+4 B 及以下逐字复现 V12。
constexpr uint64_t RS_PACKED_STRIPE_INPUT_MIN_BYTES = 448ULL * 1024 * 1024;
// 点6规约使用独立worker，避免继承通信worker的链路前序；点7只在最后
// 条带复用已完成通信的三个worker。V87 把
// 15 个远端贡献固定分成 3/4/4/4：主线程把第一组直接规约进已含本地贡献
// 的输出，3 个独立 worker 同时构造其余组根。完成树让 worker1 先合并
// worker2 的根，主线程同时合并 worker0 的根，最后只再合并一个根。
// worker数、通知数和总规约不变，关键规约深度从 V86 的 6 降到 5。
constexpr uint32_t RS_DEDICATED_REDUCE_GROUPS = 3;
constexpr uint32_t RS_TOTAL_REDUCE_GROUPS = 4;
constexpr uint32_t RS_OUTPUT_ROOT_REMOTE_COUNT = 3;
constexpr uint32_t RS_DEDICATED_REDUCE_MAX_STRIPES = 4;

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
    ThreadHandle aicpuThread;
    CommBuffer localBuffer;
    std::vector<ThreadHandle> threads;
    std::vector<ChannelInfo> channels;

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
