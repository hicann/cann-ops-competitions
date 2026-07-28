/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

// ===== 提交版本 v63-final | v60安全基线 + peer就绪流水规约 | 2026-07-24 =====
// v63-final: notify数量和槽位完全保持v60，仅缩短peer就绪后的等待路径
// exec_op.cc: 与v44一致, 5 taskArgs

#include <algorithm>
#include <limits>
#include <vector>

#include <hcomm/hcomm_primitives.h>
#include <ccu/ccu_res.h>
#include <ccu/ccu_launch.h>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "exec_op.h"
#include "ccu_kernel.h"

namespace ops_hccl {
namespace {
constexpr uint64_t MAX_DATA_SIZE_PER_KERNEL = MAX_DATA_SIZE;

HcclResult LaunchAllReduceKernels(const OpParam &param, const AlgResourceCtx &resCtx)
{
    auto sizeIt = SIZE_TABLE.find(param.dataType);
    if (sizeIt == SIZE_TABLE.end()) {
        HCCL_ERROR("[ExecOp] unsupported data type: %d", static_cast<int>(param.dataType));
        return HCCL_E_NOT_SUPPORT;
    }
    const uint32_t dataTypeSize = sizeIt->second;
    if (param.count > std::numeric_limits<uint64_t>::max() / dataTypeSize) {
        HCCL_ERROR("[ExecOp] data size overflow, count=%lu", param.count);
        return HCCL_E_PARA;
    }
    const uint64_t totalBytes = param.count * dataTypeSize;
    if (totalBytes == 0) {
        HCCL_INFO("[ExecOp] count == 0, nothing to do.");
        return HCCL_SUCCESS;
    }

    const uint64_t maxCountPerKernel = MAX_DATA_SIZE_PER_KERNEL / dataTypeSize;
    const uint64_t kernelLoopCount =
        param.count / maxCountPerKernel + static_cast<uint64_t>(param.count % maxCountPerKernel != 0);

    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    const uint64_t baseInputAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t baseOutputAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    if (param.inputPtr != nullptr) {
        CcuResult tokenRet = HcommCcuGetMemToken(baseInputAddr, totalBytes, &inputToken);
        if (tokenRet != CCU_SUCCESS) {
            HCCL_ERROR("[ExecOp] get input token failed, ccuRet=%d", tokenRet);
            return ConvertCcuToHccl(tokenRet);
        }
    }
    if (param.outputPtr != nullptr) {
        CcuResult tokenRet = HcommCcuGetMemToken(baseOutputAddr, totalBytes, &outputToken);
        if (tokenRet != CCU_SUCCESS) {
            HCCL_ERROR("[ExecOp] get output token failed, ccuRet=%d", tokenRet);
            return ConvertCcuToHccl(tokenRet);
        }
    }

    if (resCtx.threads.empty()) {
        HCCL_ERROR("[ExecOp] thread not allocated");
        return HCCL_E_INTERNAL;
    }

    if (resCtx.ccuKernels.empty()) {
        HCCL_ERROR("[ExecOp] no ccu kernel for multi-rank AllReduce");
        return HCCL_E_INTERNAL;
    }

    HCCL_INFO("[ExecOp] kernels=%zu, rankSize=%u",
        resCtx.ccuKernels.size(), param.rankSize);

    uint64_t processedCount = 0;
    for (uint64_t loop = 0; loop < kernelLoopCount; loop++) {
        const uint64_t sliceCount = std::min(maxCountPerKernel, param.count - loop * maxCountPerKernel);
        const uint64_t sliceBytes = sliceCount * dataTypeSize;

        const uint64_t inputSliceAddr = baseInputAddr + processedCount * dataTypeSize;
        const uint64_t outputSliceAddr = baseOutputAddr + processedCount * dataTypeSize;

        std::vector<uint64_t> taskArgs = {
            outputSliceAddr,  // 0: localDstAddr = OUTPUT
            inputSliceAddr,   // 1: exchangeAddr = INPUT
            outputToken,      // 2: localDstToken
            inputToken,       // 3: exchangeToken
            sliceBytes,       // 4: sliceSize
        };

        for (size_t k = 0; k < resCtx.ccuKernels.size(); k++) {
            CcuResult launchRet = HcommCcuKernelLaunch(resCtx.threads[0],
                resCtx.ccuKernels[k],
                taskArgs.data(), taskArgs.size());
            if (launchRet != CCU_SUCCESS) {
                HCCL_ERROR("[ExecOp] kernel[%zu] launch failed, ccuRet -> %d", k, launchRet);
                return ConvertCcuToHccl(launchRet);
            }
        }

        processedCount += sliceCount;
    }

    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param)
{
    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> seq(ctx, ctx + param.ctxSize);
    AlgResourceCtx resCtx;
    resCtx.DeSerialize(seq);
    if (resCtx.threads.empty()) {
        resCtx.threads.push_back(param.cpuThread);
    } else {
        // ThreadHandle 绑定调用方 stream，不能复用缓存 Context 中的旧 stream。
        resCtx.threads[0] = param.cpuThread;
    }

    if (param.rankSize == 1) {
        auto sizeIt = SIZE_TABLE.find(param.dataType);
        if (sizeIt == SIZE_TABLE.end()) {
            HCCL_ERROR("[ExecOp] unsupported data type: %d", static_cast<int>(param.dataType));
            return HCCL_E_NOT_SUPPORT;
        }
        const uint32_t dataTypeSize = sizeIt->second;
        if (param.count > std::numeric_limits<uint64_t>::max() / dataTypeSize) {
            HCCL_ERROR("[ExecOp] data size overflow, count=%lu", param.count);
            return HCCL_E_PARA;
        }
        const uint64_t totalBytes = param.count * dataTypeSize;
        if (totalBytes > 0 && !resCtx.threads.empty()) {
            HcclResult ret = static_cast<HcclResult>(
                HcommLocalCopyOnThread(resCtx.threads[0], param.outputPtr, param.inputPtr, totalBytes));
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR("[ExecOp] local copy failed, ret -> %d", ret);
                return ret;
            }
        }
        return HCCL_SUCCESS;
    }

    return LaunchAllReduceKernels(param, resCtx);
}
} // namespace ops_hccl
