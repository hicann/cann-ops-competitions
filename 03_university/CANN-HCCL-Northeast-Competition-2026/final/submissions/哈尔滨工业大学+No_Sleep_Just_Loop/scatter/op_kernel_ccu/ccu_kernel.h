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

#include "custom.h"

namespace ops_hccl {

// 各执行路径各有独立的 CCU Kernel, 由 host 侧按拓扑能力选定并注册, 互不共享代码路径。

// 路径 1 Push fallback (512KB): Root-Star。Root 写块后单向发 DONE, Peer 只连 Root。
CcuResult CcuScatterRootStarKernel(CcuKernelArg arg);

// 路径 1 Push SMALL 副本 (sendBytes < 1MiB): 协议与 CcuScatterRootStarKernel 相同, 独立注册。
CcuResult CcuScatterRootStarSmallKernel(CcuKernelArg arg);

// 4x3 SMALL Root Clos only: first-channel READY latch + 9-way Write burst.
// Mesh / Peer / LARGE / 其它拓扑仍走 CcuScatterRootStarSmallKernel, 本函数体不得回写旧核。
CcuResult CcuScatterRootStarSmallClosBurstKernel(CcuKernelArg arg);

// 路径 2: Wave。把一个 Slice 拆成两段连续 wave 复用同一批 channel。4x1 等非 mixed 大包走这里。
CcuResult CcuScatterWaveKernel(CcuKernelArg arg);

// 路径 3: Proxy。本 Server peer 用各自的 Clos 口分担 root 的跨服负载。
CcuResult CcuScatterProxyKernel(CcuKernelArg arg);

// 路径 4: Wave-Star。8+4 等 Mesh-dominant mixed 大包:
//   Root: 双 Thread 直发, Ready->Write, EventWait 后单向 DONE
//         sliceBytes <= MAX_DATA_SIZE 走 Single-Wave fast path, 否则 two-wave
//   Peer: 单 Kernel, 只 publish recvBuf 到 Root 并等 DONE
CcuResult CcuScatterWaveStarRootKernel(CcuKernelArg arg);
CcuResult CcuScatterWaveStarPeerKernel(CcuKernelArg arg);

} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
