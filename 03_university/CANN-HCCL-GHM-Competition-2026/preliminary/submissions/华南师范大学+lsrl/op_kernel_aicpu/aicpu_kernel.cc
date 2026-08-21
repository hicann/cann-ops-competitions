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

#include "common.h"
#include "custom.h"
#include "exec_op.h"
#include "log.h"

extern "C" unsigned int HcclAICPUKernel(OpParam *param)
{
    CHK_PTR_NULL(param);

    const AlgResourceCtx *resCtx = static_cast<const AlgResourceCtx *>(param->resCtx);
    CHK_PTR_NULL(resCtx);

    if (HcommBatchModeStart(param->tag) != HCCL_SUCCESS) {
        HCCL_ERROR("Failed to start batch mode");
        return 1;
    }

    bool opSucceeded = true;
    if (HcommThreadNotifyWaitOnThread(resCtx->aicpuThreads[0], 0, CUSTOM_TIMEOUT) != HCCL_SUCCESS) {
        HCCL_ERROR("Failed to wait notify from host main stream");
        opSucceeded = false;
    }

    if (opSucceeded && ops_hccl::ExecOp(*param, *resCtx) != HCCL_SUCCESS) {
        HCCL_ERROR("Failed to execute op");
        opSucceeded = false;
    }

    if (HcommThreadNotifyRecordOnThread(resCtx->aicpuThreads[0], param->cpuThreadOnAicpu, 0) != HCCL_SUCCESS) {
        HCCL_ERROR("Failed to record host main stream");
        opSucceeded = false;
    }

    if (HcommBatchModeEnd(param->tag) != HCCL_SUCCESS) {
        HCCL_ERROR("Failed to end batch mode");
        opSucceeded = false;
    }

    return opSucceeded ? 0 : 1;
}
