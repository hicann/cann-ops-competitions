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

namespace ccu = ::AscendC::ccu;

namespace ops_hccl {

// Compact phased multi-die or fused single-die owner-shard AllReduce.
CcuResult CcuDirectAllReduceKernel(CcuKernelArg arg);

// Isolated 4x1 small-payload owner-shard GroupReduce followed by direct
// producer-push.  The separate entry freezes all non-point-13 kernels.
CcuResult CcuSingleGroupDirectPushAllReduceKernel(CcuKernelArg arg);

// Online-proven isolated 2x8 small-payload GroupReduce plus producer-push.
CcuResult CcuDualGroupPushAllReduceKernel(CcuKernelArg arg);

// Isolated 8+4 small-payload Direct owner reduction plus producer-push.  A
// separate entry keeps the point-10 kernel body unchanged.
CcuResult CcuEightPlusFourDirectPushAllReduceKernel(CcuKernelArg arg);

// Layout-safe exact point-16 entry.  This declaration is appended after every
// V030 kernel declaration so the established entries keep their source order.
CcuResult CcuEightPlusFourTreeDirectPushAllReduceKernelV041(CcuKernelArg arg);

// Exact 4x1/512MiB NHR entry appended after the layout-safe P16 declaration.
// Its dedicated tag and resource flag keep every fallback ABI isolated.
CcuResult CcuExactP14NhrAllReduceKernelV042(CcuKernelArg arg);
} // namespace ops_hccl

namespace ops_hccl {
// Exact 4x1/512KiB P13 entry. This is deliberately distinct from the
// broad small-payload entry so no adjacent payload uses its compact handshake.
CcuResult CcuExactP13CompactHandshakeAllReduceKernelV051(CcuKernelArg arg);

// Exact P16 entry: V041 tree data path with one combined input/output
// address-token handshake before phase zero.
CcuResult CcuEightPlusFourTreeDirectPushCompactPreSyncAllReduceKernelV065(CcuKernelArg arg);

// Exact P11 entry: V042 two-window envelope with per-window local tree.
CcuResult CcuTwoByEightTreeDirectPushAllReduceKernelV067(CcuKernelArg arg);
// Exact 2x8/400MiB+4B P12 tree-reduce / owner-push candidate.
CcuResult CcuTwoByEightTreeDirectPushAllReduceKernelV068(CcuKernelArg arg);
} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
