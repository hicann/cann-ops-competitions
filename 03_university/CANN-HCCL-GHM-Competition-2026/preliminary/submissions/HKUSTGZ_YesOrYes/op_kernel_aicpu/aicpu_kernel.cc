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
#include "log.h"
#include "exec_op.h"

extern "C" unsigned int HcclAICPUKernel(OpParam *param)
{
    CHK_PTR_NULL(param);

    // Decode the cached EngineCtx directly into fixed-capacity resource lists.
    // No heap allocation or temporary context copy is needed on the AICPU hot
    // path, and every invocation still owns an independent stack context.
    AlgResourceCtx resCtx;
    if (!resCtx.DeSerialize(param->resCtx, param->ctxSize)) {
        HCCL_ERROR("Failed to deserialize resource context");
        return 1;
    }

    // 开启批量模式，Start-End 中间的任务会在同一个线程上执行
    if (HcommBatchModeStart(param->tag) != HCCL_SUCCESS) {
        HCCL_ERROR("Failed to start batch mode");
        return 1;
    }

    // Device 等待 Host 通知
    if (HcommThreadNotifyWaitOnThread(resCtx.aicpuThread, 0, CUSTOM_TIMEOUT) != HCCL_SUCCESS) {
        HCCL_ERROR("Failed to wait notify from host main stream");
        return 1;
    }

    // 执行通信算法任务编排
    if (ops_hccl::ExecOp(*param, resCtx) != HCCL_SUCCESS) {
        HCCL_ERROR("Failed to execute op");
        return 1;
    }

    // Device 通知 Host 任务下发完成
    if (HcommThreadNotifyRecordOnThread(resCtx.aicpuThread, param->cpuThreadOnAicpu, 0) != HCCL_SUCCESS) {
        HCCL_ERROR("Failed to record host main stream");
        return 1;
    }

    // 停止批量模式
    if (HcommBatchModeEnd(param->tag) != HCCL_SUCCESS) {
        HCCL_ERROR("Failed to end batch mode");
        return 1;
    }

    return 0;
}
