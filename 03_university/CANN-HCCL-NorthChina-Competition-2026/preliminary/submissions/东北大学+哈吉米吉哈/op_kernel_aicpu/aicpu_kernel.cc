/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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

    // 反序列化
    char *ctx = static_cast<char *>(param->resCtx);
    std::vector<char> seq(ctx, ctx + param->ctxSize);
    AlgResourceCtx resCtx;
    resCtx.DeSerialize(seq);

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
