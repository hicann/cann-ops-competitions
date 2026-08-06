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

namespace final_small512 {
namespace ops_hccl {

CcuResult CcuReduceScatterTreeKernel(CcuKernelArg arg);
CcuResult CcuReduceScatterHalfRingKernel(CcuKernelArg arg);
CcuResult CcuReduceScatterSmallKernel(CcuKernelArg arg);
CcuResult CcuReduceScatterSmallFinalizeKernel(CcuKernelArg arg);

} // namespace ops_hccl
} // namespace final_small512

namespace final_scratch_lite {
namespace ops_hccl {

CcuResult CcuReduceScatterTreeKernel(CcuKernelArg arg);

} // namespace ops_hccl
} // namespace final_scratch_lite

namespace final_epoch_cache {
namespace ops_hccl {

CcuResult CcuReduceScatterTreeKernel(CcuKernelArg arg);

} // namespace ops_hccl
} // namespace final_epoch_cache

namespace final_direct_pipeline {
namespace ops_hccl {

CcuResult CcuReduceScatterTreeKernel(CcuKernelArg arg);

} // namespace ops_hccl
} // namespace final_direct_pipeline

namespace final_adaptive41 {
namespace ops_hccl {

CcuResult CcuReduceScatterTreeKernel(CcuKernelArg arg);
CcuResult CcuReduceScatterRingKernel(CcuKernelArg arg);
CcuResult CcuReduceScatterEightPlusFourPreReduceKernel(CcuKernelArg arg);
CcuResult CcuReduceScatterEightPlusFourExchangeKernel(CcuKernelArg arg);
CcuResult CcuReduceScatterEightPlusFourEdgeInitKernel(CcuKernelArg arg);
CcuResult CcuReduceScatterEightPlusFourEdgeSendStageZeroKernel(CcuKernelArg arg);
CcuResult CcuReduceScatterEightPlusFourEdgeWaitStageZeroKernel(CcuKernelArg arg);
CcuResult CcuReduceScatterEightPlusFourEdgeSendStageOneKernel(CcuKernelArg arg);
CcuResult CcuReduceScatterEightPlusFourEdgeWaitStageOneKernel(CcuKernelArg arg);
CcuResult CcuReduceScatterEightPlusFourEdgeCopyOutputKernel(CcuKernelArg arg);

} // namespace ops_hccl
} // namespace final_adaptive41

#endif // OPS_HCCL_CCU_KERNEL_H
