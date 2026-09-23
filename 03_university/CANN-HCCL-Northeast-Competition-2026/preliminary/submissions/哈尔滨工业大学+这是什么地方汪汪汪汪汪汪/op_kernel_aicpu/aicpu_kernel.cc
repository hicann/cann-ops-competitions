/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <hcomm/hcomm_primitives.h>

#include "common.h"
#include "custom.h"
#include "log.h"
#include "exec_op.h"

extern "C" unsigned int HcclAICPUKernel(OpParam *param)
{
    CHK_PTR_NULL(param);

    // The EngineCtx contains one fixed-layout, trivially-copyable resource
    // object. Restore it with one bounded copy; the previous vector path
    // allocated and copied once before DeSerialize copied the same bytes again.
    CHK_PTR_NULL(param->resCtx);
    CHK_PRT_RET(param->ctxSize != sizeof(AlgResourceCtx),
        HCCL_ERROR("Invalid resource context size[%llu], expected[%zu]",
            static_cast<unsigned long long>(param->ctxSize), sizeof(AlgResourceCtx)),
        1);
    AlgResourceCtx resCtx;
    std::memcpy(&resCtx, param->resCtx, sizeof(resCtx));

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
