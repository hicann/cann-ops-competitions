/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <hcomm/hcomm_primitives.h>
#include "custom.h"
#include "log.h"
#include "exec_op.h"

extern "C" unsigned int HcclAICPUKernel(OpParam *param)
{
    if (param == nullptr || param->resCtx == nullptr || param->ctxSize != sizeof(AlgResourceCtx)) {
        return static_cast<unsigned int>(HCCL_E_PARA);
    }
    AlgResourceCtx ctx{};
    if (memcpy_s(&ctx, sizeof(ctx), param->resCtx, sizeof(ctx)) != EOK) {
        return static_cast<unsigned int>(HCCL_E_INTERNAL);
    }
    HcclResult ret = ops_hccl::ValidateScatter(*param, ctx);
    if (ret != HCCL_SUCCESS) {
        return static_cast<unsigned int>(ret);
    }
    ret = static_cast<HcclResult>(HcommBatchModeStart(param->tag));
    if (ret != HCCL_SUCCESS) {
        return static_cast<unsigned int>(ret);
    }
    ret = static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(ctx.aicpuThread, 0, CUSTOM_TIMEOUT));
    if (ret == HCCL_SUCCESS) {
        ret = ops_hccl::ExecOp(*param, ctx);
    }
    if (ret == HCCL_SUCCESS) {
        ret = static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(ctx.aicpuThread, param->cpuThreadOnAicpu, 0));
    }
    const int32_t endRet = HcommBatchModeEnd(param->tag);
    if (ret == HCCL_SUCCESS) {
        ret = static_cast<HcclResult>(endRet);
    }
    return static_cast<unsigned int>(ret);
}
