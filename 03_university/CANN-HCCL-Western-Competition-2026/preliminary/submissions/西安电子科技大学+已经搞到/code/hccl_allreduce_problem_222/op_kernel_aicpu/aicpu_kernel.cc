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
#include "log.h"
#include "exec_op.h"

extern "C" unsigned int HcclAICPUKernel(OpParam *param)
{
    CHK_PTR_NULL(param);
    if (param->resCtx == nullptr || param->ctxSize == 0) {
        HCCL_ERROR("Invalid resource context, addr[%p], size[%llu]", param->resCtx,
            static_cast<unsigned long long>(param->ctxSize));
        return 1;
    }

    // 反序列化
    char *ctx = static_cast<char *>(param->resCtx);
    std::vector<char> seq(ctx, ctx + param->ctxSize);
    AlgResourceCtx resCtx;
    resCtx.DeSerialize(seq);
    if (resCtx.threads.size() != 8 || resCtx.channels.size() != 8 || resCtx.aicpuThread != resCtx.threads[0]) {
        HCCL_ERROR("Invalid deserialized resources, threads[%zu], channels[%zu]", resCtx.threads.size(),
            resCtx.channels.size());
        return 1;
    }

    // 开启批量模式，Start-End 中间的任务会在同一个线程上执行
    if (HcommBatchModeStart(param->tag) != HCCL_SUCCESS) {
        HCCL_ERROR("Failed to start batch mode");
        return 1;
    }

    unsigned int result = 0;
    if (HcommThreadNotifyWaitOnThread(resCtx.aicpuThread, 0, CUSTOM_TIMEOUT) != HCCL_SUCCESS) {
        HCCL_ERROR("Failed to wait notify from host main stream");
        result = 1;
    } else if (ops_hccl::ExecOp(*param, resCtx) != HCCL_SUCCESS) {
        HCCL_ERROR("Failed to execute op");
        result = 1;
    } else if (HcommThreadNotifyRecordOnThread(resCtx.aicpuThread, param->cpuThreadOnAicpu, 0) != HCCL_SUCCESS) {
        HCCL_ERROR("Failed to record host main stream");
        result = 1;
    }

    if (HcommBatchModeEnd(param->tag) != HCCL_SUCCESS) {
        HCCL_ERROR("Failed to end batch mode");
        result = 1;
    }

    return result;
}
