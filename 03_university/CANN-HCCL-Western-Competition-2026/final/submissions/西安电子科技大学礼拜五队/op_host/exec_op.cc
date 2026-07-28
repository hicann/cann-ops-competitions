/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
HcclResult ExecOp(const OpParam &param)
{
    CHK_PRT_RET(param.dataType != HCCL_DATA_TYPE_FP32 || param.reduceType != HCCL_REDUCE_SUM,
        HCCL_ERROR("Only FP32 SUM is supported"), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || param.myRank >= param.rankSize,
        HCCL_ERROR("Invalid rank information"), HCCL_E_PARA);
    CHK_PTR_NULL(param.resCtx);
    CHK_PRT_RET(param.ctxSize == 0, HCCL_ERROR("Empty CCU resource context"), HCCL_E_INTERNAL);

    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> seq(ctx, ctx + param.ctxSize);
    AlgResourceCtx resCtx;
    resCtx.DeSerialize(seq);

    constexpr uint64_t dataTypeSize = sizeof(float);
    CHK_PRT_RET(param.count > UINT64_MAX / dataTypeSize, HCCL_ERROR("AllReduce byte size overflow"), HCCL_E_PARA);

    const uint64_t totalBytes = param.count * dataTypeSize;
    if (totalBytes == 0) {
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(resCtx.threads.empty(), HCCL_ERROR("No CCU thread"), HCCL_E_INTERNAL);
    CHK_PTR_NULL(resCtx.localBuffer.addr);

    if (param.rankSize == 1) {
        return static_cast<HcclResult>(
            HcommLocalCopyOnThread(param.cpuThread, param.outputPtr, param.inputPtr, totalBytes));
    }

    CHK_PRT_RET(resCtx.ccuKernels.size() != 1, HCCL_ERROR("CCU kernel resource is missing"), HCCL_E_INTERNAL);

    // HCCL Buffer 一半存放本地稳定输入，一半存放远端临时输入
    uint64_t maxTileBytes = resCtx.localBuffer.size / 2;
    if (maxTileBytes > MAX_DATA_SIZE) {
        maxTileBytes = MAX_DATA_SIZE;
    }
    maxTileBytes = maxTileBytes / dataTypeSize * dataTypeSize;
    CHK_PRT_RET(maxTileBytes == 0, HCCL_ERROR("HCCL Buffer is too small"), HCCL_E_INTERNAL);

    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    uint64_t cclBufferToken = 0;

    CcuResult ccuRet = HcommCcuGetMemToken(reinterpret_cast<uint64_t>(param.inputPtr), totalBytes, &inputToken);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS, HCCL_ERROR("Failed to get input memory token: %d", ccuRet), HCCL_E_INTERNAL);

    ccuRet = HcommCcuGetMemToken(reinterpret_cast<uint64_t>(param.outputPtr), totalBytes, &outputToken);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS, HCCL_ERROR("Failed to get output memory token: %d", ccuRet), HCCL_E_INTERNAL);

    ccuRet = HcommCcuGetMemToken(
        reinterpret_cast<uint64_t>(resCtx.localBuffer.addr), resCtx.localBuffer.size, &cclBufferToken);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS, HCCL_ERROR("Failed to get HCCL Buffer token: %d", ccuRet), HCCL_E_INTERNAL);

    const uint64_t baseInputAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t baseOutputAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    const uint64_t cclBufferAddr = reinterpret_cast<uint64_t>(resCtx.localBuffer.addr);

    for (uint64_t offset = 0; offset < totalBytes;) {
        const uint64_t remaining = totalBytes - offset;
        const uint64_t tileBytes = remaining < maxTileBytes ? remaining : maxTileBytes;

        const uint64_t taskArgs[] = {
            baseInputAddr + offset,
            baseOutputAddr + offset,
            inputToken,
            outputToken,
            cclBufferAddr,
            cclBufferToken,
            tileBytes,
        };

        ccuRet = HcommCcuKernelLaunch(
            param.cpuThread, resCtx.ccuKernels[0], taskArgs, sizeof(taskArgs) / sizeof(taskArgs[0]));
        CHK_PRT_RET(ccuRet != CCU_SUCCESS, HCCL_ERROR("CCU kernel launch failed: %d", ccuRet), HCCL_E_INTERNAL);

        offset += tileBytes;
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl