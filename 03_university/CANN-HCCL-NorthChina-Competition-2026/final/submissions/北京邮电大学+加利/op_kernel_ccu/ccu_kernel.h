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

// 通用单块 Direct Kernel：既是正确性基线，也是非固定消息大小的回退路径。
CcuResult CcuAllGatherDirectKernel(CcuKernelArg arg);

// 九类 Profile 保留独立入口；入口只接受匹配的 Rank、大小和层角色，数据面复用安全核心。
CcuResult CcuAllGatherP2X8_512KKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP2X8_512MKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP2X8_400M4BKernel(CcuKernelArg arg);
// 大消息 A/B 使用精确到拓扑和数据量的入口，错误 Profile 不允许进入数据面。
CcuResult CcuAllGatherP2X8_512MSingleDmaKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP2X8_400M4BSingleDmaKernel(CcuKernelArg arg);
// 原 shared-CKE 两阶段入口保留用于真实硬件 A/B；默认路径由四个阶段入口配合
// Host 的跨 Thread Notify 编排，兼容当前 HCCL-VM checker。
CcuResult CcuAllGatherP2X8_512MHybridRelayKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP2X8_400M4BHybridRelayKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP2X8_512MHybridPhaseAKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP2X8_512MHybridPhaseBKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP2X8_400M4BHybridPhaseAKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP2X8_400M4BHybridPhaseBKernel(CcuKernelArg arg);
// 2x8 大消息独立 A/B 路径：同 slot 远端 Rank 为分布式 Root，每层仍由独立 Kernel 执行。
CcuResult CcuAllGatherP2X8_512MDistributedRootPhaseAKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP2X8_512MDistributedRootPhaseBKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP2X8_512MDistributedRootCappedPhaseAKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP2X8_512MDistributedRootCappedPhaseBKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP2X8_512MDistributedRootMatchedSerialPhaseAKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP2X8_512MDistributedRootMatchedSerialPhaseBKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP2X8_400M4BDistributedRootPhaseAKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP2X8_400M4BDistributedRootPhaseBKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP2X8_400M4BDistributedRootCappedPhaseAKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP2X8_400M4BDistributedRootCappedPhaseBKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP2X8_400M4BDistributedRootMatchedSerialPhaseAKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP2X8_400M4BDistributedRootMatchedSerialPhaseBKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP4X1_512KKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP4X1_512MKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP4X1_400M4BKernel(CcuKernelArg arg);
// 仅供 4x1 两个大消息 Profile 的 Host 注册实验入口；不参与通用 Profile 算法选择。
CcuResult CcuAllGatherP4X1LargeSingleDmaKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP4X1LargeMatchedReadyKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP4X1LargeMatchedSerialRolling2Kernel(CcuKernelArg arg);
CcuResult CcuAllGatherP8P4_512KKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP8P4_512MKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP8P4_400M4BKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP8P4_512MSingleDmaKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP8P4_400M4BSingleDmaKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP8P4_512MHybridPhaseAKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP8P4_512MHybridPhaseBKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP8P4_400M4BHybridPhaseAKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP8P4_400M4BHybridPhaseBKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP8P4_512MBidirectionalHalfRootPhaseAKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP8P4_512MBidirectionalHalfRootPhaseBKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP8P4_512MBidirectionalHalfRootCappedPhaseAKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP8P4_512MBidirectionalHalfRootCappedPhaseBKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP8P4_512MBidirectionalHalfRootMatchedPhaseAKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP8P4_512MBidirectionalHalfRootMatchedPhaseBKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP8P4_512MBidirectionalHalfRootSharedPhaseAKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP8P4_512MBidirectionalHalfRootSharedPhaseBKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP8P4_512MBidirectionalHalfRootShared5Over8PhaseAKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP8P4_512MBidirectionalHalfRootShared5Over8PhaseBKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP8P4_512MBidirectionalHalfRootFullSeed4Over7PhaseAKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP8P4_512MBidirectionalHalfRootFullSeed4Over7PhaseBKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP8P4_400M4BBidirectionalHalfRootPhaseAKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP8P4_400M4BBidirectionalHalfRootPhaseBKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP8P4_400M4BBidirectionalHalfRootCappedPhaseAKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP8P4_400M4BBidirectionalHalfRootCappedPhaseBKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP8P4_400M4BBidirectionalHalfRootSharedPhaseAKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP8P4_400M4BBidirectionalHalfRootSharedPhaseBKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP8P4_400M4BBidirectionalHalfRootShared5Over8PhaseAKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP8P4_400M4BBidirectionalHalfRootShared5Over8PhaseBKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP8P4_400M4BBidirectionalHalfRootFullSeed4Over7PhaseAKernel(CcuKernelArg arg);
CcuResult CcuAllGatherP8P4_400M4BBidirectionalHalfRootFullSeed4Over7PhaseBKernel(CcuKernelArg arg);

// 经 HCCL Buffer 搬移一个重叠块；Host 通过正向或反向下发使连续调用满足 memmove 语义。
CcuResult CcuAllGatherPrepareKernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
