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

// 小包及 4x1 拓扑使用直达 push；2x8 / 8+4 大包把数据沿两个维度拆分，
// 由同一函数注册出的 layer-0 / layer-1 kernel 并行执行两个阶段。
CcuResult CcuAllGatherDirectPush(CcuKernelArg arg);
// 极简扁平 kernel：小消息与 4*1 拓扑（独立注册，指令数最小化）
CcuResult CcuAllGatherFlat(CcuKernelArg arg);
} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
