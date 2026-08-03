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

#include <ccu/ccu_types.h>

namespace ops_hccl {

// 按本地Die并发pull分片贡献，并使用固定归约树生成确定性的局部结果。
CcuResult CcuReduceScatterDeterministicKernel(CcuKernelArg arg);

// 在CCU内合并两个Die的局部结果，避免Host Runtime Reduce依赖当前Context。
CcuResult CcuReduceScatterMergeKernel(CcuKernelArg arg);

// 小消息分层路径：先在Server内生成多个输出分片的局部规约结果。
CcuResult CcuReduceScatterHierLocalKernel(CcuKernelArg arg);

// 小消息分层路径：跨Server仅交换每个输出分片的一份局部规约结果。
CcuResult CcuReduceScatterHierCrossKernel(CcuKernelArg arg);
} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
