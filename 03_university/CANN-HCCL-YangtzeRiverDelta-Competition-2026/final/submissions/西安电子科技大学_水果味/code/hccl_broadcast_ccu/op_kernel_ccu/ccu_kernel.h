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

CcuResult CcuFlatBroadcastLayer0Kernel(CcuKernelArg arg);
CcuResult CcuFlatBroadcastLayer1Kernel(CcuKernelArg arg);
CcuResult CcuScatterLayer0Kernel(CcuKernelArg arg);
CcuResult CcuScatterLayer1Kernel(CcuKernelArg arg);
CcuResult CcuAllGatherLayer0Kernel(CcuKernelArg arg);
CcuResult CcuAllGatherLayer1Kernel(CcuKernelArg arg);
CcuResult CcuBroadcastNhrLayer1Kernel(CcuKernelArg arg);
} // namespace ops_hccl

#endif // OPS_HCCL_CCU_KERNEL_H
