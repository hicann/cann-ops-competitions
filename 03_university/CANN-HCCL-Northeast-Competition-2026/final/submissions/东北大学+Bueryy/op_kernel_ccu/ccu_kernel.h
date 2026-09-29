/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SCATTER_FINAL_CCU_KERNEL_H
#define SCATTER_FINAL_CCU_KERNEL_H

#include "custom.h"
#include "log.h"

namespace ccu = ::AscendC::ccu;

#define CCU_CHK_RET(call) do { \
    CcuResult result = (call); \
    if (result != CCU_SUCCESS) { \
        HCCL_ERROR("[%s] CCU 调用失败：%d", __func__, result); \
        return result; \
    } \
} while (0)

namespace ops_hccl {
CcuResult ScatterWriteKernel(CcuKernelArg arg);
CcuResult ScatterReadKernel(CcuKernelArg arg);
CcuResult ScatterRelayKernel(CcuKernelArg arg);
}
#endif
