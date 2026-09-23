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

namespace ops_hccl {
// 每个 peer 始终只使用一个 Channel，在该 Channel 上复用两个有界流水槽。
// READY[s] 发布数据；ACK[s] 归还对应槽。整段常驻转发前缀另用一个 READY。
constexpr uint32_t SCATTER_PIPELINE_DEPTH = 2;
constexpr uint32_t SCATTER_READY_NOTIFY_BASE = 0;
constexpr uint32_t SCATTER_ACK_NOTIFY_BASE = SCATTER_PIPELINE_DEPTH;
constexpr uint32_t SCATTER_RELAY_READY_NOTIFY = 2 * SCATTER_PIPELINE_DEPTH;
constexpr uint32_t SCATTER_CHANNEL_NOTIFY_NUM = SCATTER_RELAY_READY_NOTIFY + 1;
constexpr uint32_t SCATTER_WORKER_START_NOTIFY = 0;

// 按实际 recvCount 选择路径；这些值是首版调度参数，并非实测最优值。
constexpr uint64_t SCATTER_SMALL_RECV_BYTES = 512ULL * 1024;
constexpr uint64_t SCATTER_MAX_TILE_BYTES = 8ULL * 1024 * 1024;
constexpr uint64_t SCATTER_TILE_ALIGNMENT = 64;

// V16 的 own/suffix 粒度；root Copy 和旧双槽仍用 8MiB。
constexpr uint64_t SCATTER_RESIDENT_MAX_TILE_BYTES = 16ULL * 1024 * 1024;
static_assert(SCATTER_RESIDENT_MAX_TILE_BYTES >= SCATTER_MAX_TILE_BYTES &&
    SCATTER_RESIDENT_MAX_TILE_BYTES % SCATTER_TILE_ALIGNMENT == 0,
    "Invalid Scatter resident tile limit");

// V17 仅在主段已常驻时，将不超过此上限的完整前缀合为一次 Write/Read。
constexpr uint64_t SCATTER_PREFIX_SINGLE_TRANSFER_BYTES = 16ULL * 1024 * 1024;

// V18 只将原常驻计划中恰为两块的直传后缀合一，不改变 helper 自身段粒度。
constexpr uint64_t SCATTER_SUFFIX_SINGLE_TRANSFER_BYTES = 24ULL * 1024 * 1024;

// 16 rank 固定按 [0,8) / [8,16) 分工；沿用 V1 全 peer Channel，不依赖拓扑类型枚举。
// 原 MeshClos 回退仍转发 3/8；其分块对齐和容量因子不能改用 Push 的比例。
constexpr uint32_t SCATTER_MESH_RANKS = 8;
constexpr uint32_t SCATTER_TOPO_RANKS = 2 * SCATTER_MESH_RANKS;
constexpr uint64_t SCATTER_RELAY_PARTS = 3;
constexpr uint64_t SCATTER_SPLIT_PARTS = 8;
constexpr uint64_t SCATTER_TOPO_ALIGNMENT = SCATTER_TILE_ALIGNMENT * SCATTER_SPLIT_PARTS;

// V15 仅用于 TopologyPushPlan 的连续前缀。单 Mesh 边 B、root Clos 聚合 4B 时，
// 1+x = (8-7x)/4 的平衡点为 x=4/11；不是平台实测最优值。
constexpr uint64_t SCATTER_PUSH_RELAY_PARTS = 4;
constexpr uint64_t SCATTER_PUSH_SPLIT_PARTS = 11;
static_assert(SCATTER_PUSH_SPLIT_PARTS != 0 && SCATTER_PUSH_RELAY_PARTS < SCATTER_PUSH_SPLIT_PARTS,
    "Invalid Scatter Push relay fraction");
} // namespace ops_hccl

typedef struct {
    void *addr = nullptr;
    uint64_t size = 0;
} CommBuffer;

struct ChannelInfo {
    uint32_t remoteRank = INVALID_VALUE_RANKID;
    uint32_t notifyNum = 0;
    ChannelHandle handle = 0;
    CommBuffer remoteCclMem{};
};

struct AlgResourceCtx {
    ThreadHandle aicpuThread = 0;     ///< threads[0]，保留 notify[0] 供 Host/Device 同步
    CommBuffer localBuffer{};        ///< 本端 HCCL 通信内存，使用查询得到的实际容量
    std::vector<ThreadHandle> threads; ///< 主线程 + rankSize-1 个 worker；worker 的首任务必须是 WAIT
    std::vector<ChannelInfo> channels; ///< 按远端真实 rank 索引；本 rank 的位置不申请 Channel

    // 仅缓存通信资源，不缓存 root、count 或用户地址，允许同一通信域连续调用时切换 root。
    // 大消息优先使用 helper 常驻前缀和接收双槽；容量不足时回退到原 root 暂存布局。

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
