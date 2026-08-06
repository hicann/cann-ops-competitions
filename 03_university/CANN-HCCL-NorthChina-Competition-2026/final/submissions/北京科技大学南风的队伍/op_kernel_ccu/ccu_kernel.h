/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

// 北京科技大学南风的队伍：决赛 CCU Kernel 接口声明。

#ifndef OPS_HCCL_CCU_KERNEL_H
#define OPS_HCCL_CCU_KERNEL_H

#include <ccu/ccu_types.h>

namespace ops_hccl {

// 在 CCU 上执行一个直写或小消息拉取的 AllGather 分片。
CcuResult CcuKernel(CcuKernelArg arg);
// 将已收集的 Gateway 数据转发给其余本地 Peer。
CcuResult G2ForwardKernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
