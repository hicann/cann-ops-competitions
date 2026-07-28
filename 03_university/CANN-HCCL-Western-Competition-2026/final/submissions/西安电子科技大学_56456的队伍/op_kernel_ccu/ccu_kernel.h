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
// 关键修复: 回到v44单槽位模式 — 与官方demo一致, v52-v54多槽位模式导致不出分
//   WriteVariableWithNotify: 槽位=CKE_IDX_0, mask=1<<XN_ID (不同变量用不同bit)
//   NotifyWait: 槽位=CKE_IDX_0, mask=组合bit
//   PostSync: 槽位=CKE_IDX_0, mask=1<<POST_SYNC_ID

#ifndef OPS_HCCL_CCU_KERNEL_H
#define OPS_HCCL_CCU_KERNEL_H

#include <cstdint>
#include <vector>
#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>
#include <hccl/hcomm_primitives.h>
#include <ccu/ccu_types.h>
#include <ccu/ccu_variable.hpp>
#include <ccu/ccu_event.hpp>
#include <ccu/ccu_primitives.hpp>

#include "common.h"
#include "custom.h"   // 包含 CcuKernelArgBase

namespace ops_hccl {

// CCU kernel 名称 (注册时使用)
constexpr const char *CCU_KERNEL_NAME = "CcuAllReduceMesh1DKernel";

// Channel 上同步寄存器索引 (单槽位模式, 与v44/官方demo一致)
constexpr uint32_t CKE_IDX_0 = 0;

// 在 channel 上交换的变量 ID (与 allreduce.cc 中保持一致)
constexpr uint32_t OUTPUT_XN_ID = 1;
constexpr uint32_t TOKEN_XN_ID = 2;
constexpr uint32_t POST_SYNC_ID = 3;

// CCU Kernel 入参: 继承 CcuKernelArgBase (含 channels[]/channelCount)
struct CcuKernelArg : public CcuKernelArgBase {
    uint32_t rankSize = 0;
    uint32_t rankId = 0xFFFFFFFF;
    HcclReduceOp reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    uint32_t phase = 0;
};

// CCU Kernel 函数声明
CcuResult CcuKernel(::CcuKernelArg arg);

} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
