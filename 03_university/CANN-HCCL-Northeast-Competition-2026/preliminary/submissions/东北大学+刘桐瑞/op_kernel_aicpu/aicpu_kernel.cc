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
#include <cstring>
#include <hcomm/hcomm_primitives.h>

#include "common.h"
#include "custom.h"
#include "exec_op.h"

namespace {
struct ResCtxCache {
    void *key = nullptr;
    uint64_t ctxSize = 0;
    uint32_t root = INVALID_VALUE_RANKID;
    bool valid = false;
    char head[SCATTER_RES_CTX_KEY_BYTES] = {0};
    AlgResourceCtx ctx;
};
ResCtxCache g_resCtxCache;

const AlgResourceCtx &GetResCtx(OpParam *param, AlgResourceCtx &fallback)
{
    char *ctx = static_cast<char *>(param->resCtx);
#if SCATTER_CACHE_RES_CTX
    const uint32_t headLen = (param->ctxSize < SCATTER_RES_CTX_KEY_BYTES)
        ? static_cast<uint32_t>(param->ctxSize)
        : SCATTER_RES_CTX_KEY_BYTES;
    if (g_resCtxCache.valid && g_resCtxCache.key == param->resCtx && g_resCtxCache.ctxSize == param->ctxSize
        && g_resCtxCache.root == param->root && memcmp(g_resCtxCache.head, ctx, headLen) == 0) {
        (void)fallback;
        return g_resCtxCache.ctx;
    }
    std::vector<char> seq(ctx, ctx + param->ctxSize);
    g_resCtxCache.valid = false;
    g_resCtxCache.ctx.DeSerialize(seq);
    g_resCtxCache.key = param->resCtx;
    g_resCtxCache.ctxSize = param->ctxSize;
    g_resCtxCache.root = param->root;
    (void)memcpy(g_resCtxCache.head, ctx, headLen);
    g_resCtxCache.valid = true;
    (void)fallback;
    return g_resCtxCache.ctx;
#else
    std::vector<char> seq(ctx, ctx + param->ctxSize);
    fallback.DeSerialize(seq);
    return fallback;
#endif
}
} // namespace

extern "C" unsigned int HcclAICPUKernel(OpParam *param)
{
    if (param == nullptr) {
        return 1;
    }

    AlgResourceCtx localCtx;
    const AlgResourceCtx &resCtx = GetResCtx(param, localCtx);

#if SCATTER_USE_BATCH_MODE
    if (HcommBatchModeStart(param->tag) != HCCL_SUCCESS) {
        return 1;
    }
#endif

    if (HcommThreadNotifyWaitOnThread(resCtx.aicpuThread, 0, CUSTOM_TIMEOUT) != HCCL_SUCCESS) {
        return 1;
    }

    if (ops_hccl::ExecOp(*param, resCtx) != HCCL_SUCCESS) {
        return 1;
    }

    if (HcommThreadNotifyRecordOnThread(resCtx.aicpuThread, param->cpuThreadOnAicpu, 0) != HCCL_SUCCESS) {
        return 1;
    }

#if SCATTER_USE_BATCH_MODE
    if (HcommBatchModeEnd(param->tag) != HCCL_SUCCESS) {
        return 1;
    }
#endif

    return 0;
}
