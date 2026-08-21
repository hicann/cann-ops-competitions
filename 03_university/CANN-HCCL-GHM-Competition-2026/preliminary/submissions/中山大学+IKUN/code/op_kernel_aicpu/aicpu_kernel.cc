/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <hcomm/hcomm_primitives.h>

#include "common.h"
#include "custom.h"
#include "exec_op.h"
#include "log.h"

extern "C" unsigned int HcclAICPUKernel(OpParam *param)
{
    CHK_PTR_NULL(param);

    char *ctx = static_cast<char *>(param->resCtx);
    std::vector<char> seq(ctx, ctx + param->ctxSize);
    AlgResourceCtx resCtx;
    resCtx.DeSerialize(seq);

    if (HcommBatchModeStart(param->tag) != HCCL_SUCCESS) {
        HCCL_ERROR("Failed to start batch mode");
        return 1;
    }
    if (HcommThreadNotifyWaitOnThread(resCtx.aicpuThread, 0, CUSTOM_TIMEOUT) != HCCL_SUCCESS) {
        HCCL_ERROR("Failed to wait notify from host main stream");
        return 1;
    }
    if (ops_hccl::ExecOp(*param, resCtx) != HCCL_SUCCESS) {
        HCCL_ERROR("Failed to execute op");
        return 1;
    }
    if (HcommThreadNotifyRecordOnThread(
            resCtx.aicpuThread, param->cpuThreadOnAicpu, 0) != HCCL_SUCCESS) {
        HCCL_ERROR("Failed to record host main stream");
        return 1;
    }
    if (HcommBatchModeEnd(param->tag) != HCCL_SUCCESS) {
        HCCL_ERROR("Failed to end batch mode");
        return 1;
    }
    return 0;
}
