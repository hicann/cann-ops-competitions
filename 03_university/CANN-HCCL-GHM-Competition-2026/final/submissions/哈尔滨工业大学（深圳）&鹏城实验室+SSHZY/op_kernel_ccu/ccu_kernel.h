/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CCU_KERNEL_H
#define OPS_HCCL_CCU_KERNEL_H

#include <cstdint>
#include <hccl/hccl_types.h>

// ChannelHandle 前向定义（对齐 hcomm_primitives.h 中的 typedef）
#ifndef CHANNEL_HANDLE_DEFINED
#define CHANNEL_HANDLE_DEFINED
typedef uint64_t ChannelHandle;
#endif

namespace ops_hccl {

// ===========================================================================
// Channel Variable 索引常量（Host/Device 共用）
// ===========================================================================
constexpr uint32_t RS_INPUT_XN_ID = 0;  // 远端 rank 的 INPUT VA (Variable slot 0)
constexpr uint32_t RS_TOKEN_XN_ID = 2;  // 远端 rank 的 token (Variable slot 2)
constexpr uint32_t RS_POST_SYNC_ID = 3; // PostSync notify bit

constexpr uint32_t RS_CKE_IDX_0 = 0; // notify 寄存器索引
// RH 步同步：STEP_PRE_SYNC(bit4) → WriteReduce → STEP_POST_SYNC(bit5)
// PostSync 用 bit3
constexpr uint32_t RS_STEP_PRE_SYNC_BIT = 1 << 4;
constexpr uint32_t RS_STEP_POST_SYNC_BIT = 1 << 5;
constexpr uint32_t RS_MAX_RANK_SIZE = 16;

// ===========================================================================
// ReduceScatterKernelArg —— CCU Kernel 静态参数
//
// **布局必须与 Host 侧 ReduceScatterKernelArgHost 严格一致**
// Host 侧通过 CcuKernelArgBase 继承，channels 在最前面：
//   channels[16] + channelCount + padding + rank/reduce flags
// ===========================================================================
struct ReduceScatterKernelArg {
    ChannelHandle channels[RS_MAX_RANK_SIZE]; // offset 0
    uint32_t channelCount;
    uint32_t pad_;         // ← CcuKernelArgBase 继承带来的 4B padding
    uint32_t rankSize;     // offset 136
    uint32_t rankId;       // offset 140
    HcclDataType dataType; // offset 144
    HcclReduceOp reduceOp; // offset 148
    bool initOutput;       // offset 152: 是否包含本 rank 输入
    bool reduceToOutput;   // offset 153: 是否累加到 output
    bool skipOutput;       // offset 154: 是否跳过 output 写回
    uint32_t stepOffset;   // offset 156: RH 全局步号偏移（die1=0, die0=1）
};

// ===========================================================================
// CCU Kernel 函数声明
// ===========================================================================

// 算法 0: Read + LocalReduce（Mesh 全连接，O(n) Read，适合大数据量）
CcuResult CcuKernel(CcuKernelArg arg);

// 算法 0 pipeline 变体：首 slice 执行 PreSync，最终 PostSync 由 drain merge 负责
CcuResult CcuKernelPipeline(CcuKernelArg arg);

// 4x1 512KB 小数据：单 die 单 kernel，peer 并发 Read 到 3 槽 scratch，self 直写 output。
CcuResult CcuKernelSmall4x1(CcuKernelArg arg);

// 双 die mesh 收尾：归并两个 die 的 CCL 局部和并写回 output
CcuResult CcuKernelMergeScratch(CcuKernelArg arg);

// 大数据 grouped-Drr：按 kernelArg.stepOffset 指定的 maxGroupSize 拆组，组间同轮并发，组内轮转
// ReadReduce。第一组写主目标，其余组写 CCL scratch 后本地合并。
CcuResult CcuKernelGroupDrr(CcuKernelArg arg);

// 算法 1: Recursive Halving WriteReduce（O(log n) 步，适合小数据量 + 2的幂）
// 每步用 1 个 channel，一个 kernel 处理一个 die 上的全部 RH 步
CcuResult CcuKernelRH(CcuKernelArg arg);

// 算法 2: Recursive Halving WriteReduce sliced（4-rank 大数据）
// 每个 block 分别传当前 slice，避免跨 block 切片时的非连续内存问题
CcuResult CcuKernelRHSliced(CcuKernelArg arg);

} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
