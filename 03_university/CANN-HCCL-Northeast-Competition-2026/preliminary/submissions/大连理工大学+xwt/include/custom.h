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

// 每条 channel 上申请的 notify 数量。Scatter 单向节拍用到 3 个下标：
//   NOTIFY_IDX_ACK(0)         非 root -> root，"本段已从我的 ccl buffer 取走，槽位可复用"
//   NOTIFY_IDX_DATA_SIGNAL(1) root -> 非 root，"本段已写入你的 ccl buffer"
//   NOTIFY_IDX_BUF_READY(2)   非 root -> root，"我已进入本次算子，ccl buffer 槽位初始可写"
constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;

// 起始就绪屏障使用的 notify 下标（NOTIFY_IDX_ACK / NOTIFY_IDX_DATA_SIGNAL 由 common.h 给出）
constexpr uint32_t NOTIFY_IDX_BUF_READY = 2;

// 单段通信允许的最大字节数：256 MB
constexpr uint64_t MAX_DATA_SIZE = 256ULL * 1024 * 1024;

// Thread（= 硬件流）资源规格。
//   threads[0] 兼作 aicpuThread，承担 aicpu_kernel.cc（不可改）的 Host/Device 握手，
//              因此它的 thread notify 下标 0 被框架占用，算法只能用 1 及以上。
//   threads[1..] 为 root 并行发送用的从流：单条流内任务顺序执行，要让 15 条链路同时跑
//              就必须有 15 条流，所以 1 + (rankSize-1) = rankSize。
// notifyNumPerThread 必须 >= 1(框架占用) + 从流数，故与 threadNum 取同值。
constexpr uint32_t MAX_THREAD_NUM = 16;

// 大消息路径（同 Server helper 中继）的启用门槛，按 recvCount*dtype*rankSize 的总量判定。
// 小消息（点 9 规模）里 97% 是框架固定开销，中继多一跳 notify 往返净亏，所以只在大消息启用。
// ★ 一键回退：把这里改成 UINT64_MAX，大消息路径永不触发，行为逐字节退回"纯直发 + 多流并行"。
constexpr uint64_t LARGE_MSG_THRESHOLD = 8ULL * 1024 * 1024;

// 远端写入起始偏移的对齐粒度
constexpr uint64_t SLICE_ALIGN = 512;

// BW_COEFF 的合理取值区间，超出即视为查询结果不可信 -> 退回纯直发
constexpr uint32_t BW_COEFF_MIN = 1;
constexpr uint32_t BW_COEFF_MAX = 1024;

// 网络层次：layer-0 为框内直连（1DMesh 全互联），layer-1 为跨框（Clos/RDMA 全互联）
constexpr uint32_t NET_LAYER_INTRA = 0;
constexpr uint32_t NET_LAYER_INTER = 1;

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

    // ===== v2 新增字段：只追加在末尾，不打乱既有字段顺序 =====
    // bit j = 1 表示 rank j 与本 rank 同 Server（layer-0 上有直连）。device 侧据此还原
    // "本 Server 的 rank 列表 / 列号 / 同列伙伴"，不依赖 rank 编号一定是 0-7 / 8-15。
    uint32_t intraPeerMask = 0;
    // 框内链路与跨框 Clos 的带宽系数（ENDPOINT_ATTR_BW_COEFF）。
    // 任一为 0 表示查询失败或取值不可信，device 侧据此把中继比例 q 钳到 0（退回纯直发）。
    uint32_t bwCoeffIntra = 0;
    uint32_t bwCoeffInter = 0;

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << aicpuThread;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << channels;
        binaryStream << intraPeerMask;
        binaryStream << bwCoeffIntra;
        binaryStream << bwCoeffInter;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    // 反序列化（顺序必须与 Serialize 严格一致）
    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> aicpuThread;
        binaryStream >> localBuffer;
        binaryStream >> threads;
        binaryStream >> channels;
        binaryStream >> intraPeerMask;
        binaryStream >> bwCoeffIntra;
        binaryStream >> bwCoeffInter;
    }
};

#endif // OPS_HCCL_CUSTOM_H
