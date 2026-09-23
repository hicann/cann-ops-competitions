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

// Scatter 直发星形 Kernel（按角色拆分）。
// root：等待各对端 recvBuf 地址/Token，RDMA 写每个对端的 chunk 并本地拷贝自留 chunk，最后通知各对端完成。
// 非 root：发布自身 recvBuf 地址/Token，等待 root 的完成通知。
CcuResult CcuScatterRootKernel(CcuKernelArg arg);
CcuResult CcuScatterNonRootKernel(CcuKernelArg arg);

// Scatter 4×1 小消息专用 Kernel（PUSH 直发星形）：
// root 自留块本地拷贝先提交 + 逐 channel NotifyWait→Write 交错；非 root 只加载 recvBuf 地址/Token 尽早发布。
CcuResult CcuScatter4x1RootKernel(CcuKernelArg arg);
CcuResult CcuScatter4x1NonRootKernel(CcuKernelArg arg);

// Scatter 拉模型 Kernel（小消息）：root 只发布 sendBuf 地址/Token，非 root 主动从 root 读自己的 chunk。
CcuResult CcuScatterPullRootKernel(CcuKernelArg arg);
CcuResult CcuScatterPullNonRootKernel(CcuKernelArg arg);

// Scatter 大消息 Kernel（>256MB）：地址交换/DONE 只做一次，分片下放到 kernel 内，单次 launch 背靠背发 2 片。
CcuResult CcuScatterLargeRootKernel(CcuKernelArg arg);
CcuResult CcuScatterLargeNonRootKernel(CcuKernelArg arg);

// Scatter 中继 Kernel（2×8 / 4×3 大消息）：root 把跨机块的一部分交给本地 relay 转发，摊薄 Clos 上行。
// relayRole：1=relay（mesh(die1) 拆 gather(读 payload)+own(读自己块) 两 kernel，Clos(die0) 转发，
//             forward 在 gather 完成后启动、与 own 读并行，跨 die 先后由 host 线程 notify 串），
//            2=target（跨机，收 root 尾块+relay 前块）。
CcuResult CcuScatterRelayRootKernel(CcuKernelArg arg);
CcuResult CcuScatterRelayRelayMeshKernel(CcuKernelArg arg);
CcuResult CcuScatterRelayRelayOwnKernel(CcuKernelArg arg);
CcuResult CcuScatterRelayRelayClosKernel(CcuKernelArg arg);
CcuResult CcuScatterRelayTargetKernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
