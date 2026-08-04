/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <vector>

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>

#include "common.h"
#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {
constexpr uint64_t MAX_DATA_SIZE = 256ULL * 1024 * 1024;
constexpr uint64_t CHUNK_ALIGN = 128;
constexpr uint32_t TASK_ARG_COUNT = 9;

HcclResult SafeMultiply(uint64_t left, uint64_t right, uint64_t &result)
{
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
        return HCCL_E_PARA;
    }
    result = left * right;
    return HCCL_SUCCESS;
}

HcclResult CopyBytesOnThread(ThreadHandle thread, void *dst, const void *src, uint64_t bytes)
{
    auto *dstBytes = static_cast<uint8_t *>(dst);
    const auto *srcBytes = static_cast<const uint8_t *>(src);
    for (uint64_t offset = 0; offset < bytes;) {
        const uint64_t currentBytes = std::min(MAX_DATA_SIZE, bytes - offset);
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(thread, dstBytes + offset, srcBytes + offset, currentBytes)));
        offset += currentBytes;
    }
    return HCCL_SUCCESS;
}

uint64_t AlignDown(uint64_t value, uint64_t alignment)
{
    return value / alignment * alignment;
}
} // namespace

HcclResult ExecOp(const OpParam &param)
{
    const auto sizeIt = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(sizeIt == SIZE_TABLE.end(),
        HCCL_ERROR("ExecOp: data type[%d] has no size mapping", param.dataType), HCCL_E_PARA);
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE,
        HCCL_ERROR("ExecOp: invalid rank size[%u]", param.rankSize), HCCL_E_PARA);
    CHK_PRT_RET(param.myRank >= param.rankSize,
        HCCL_ERROR("ExecOp: rank id[%u] is outside rank size[%u]", param.myRank, param.rankSize), HCCL_E_PARA);

    uint64_t recvBytes = 0;
    CHK_RET(SafeMultiply(param.count, sizeIt->second, recvBytes));
    uint64_t sendBytes = 0;
    CHK_RET(SafeMultiply(recvBytes, param.rankSize, sendBytes));
    if (recvBytes == 0) {
        return HCCL_SUCCESS;
    }

    CHK_PTR_NULL(param.inputPtr);
    CHK_PTR_NULL(param.outputPtr);
    CHK_PTR_NULL(param.resCtx);
    CHK_PRT_RET(param.ctxSize == 0,
        HCCL_ERROR("ExecOp: resource context is empty"), HCCL_E_PARA);
    std::vector<char> serializedContext(static_cast<const char *>(param.resCtx),
        static_cast<const char *>(param.resCtx) + param.ctxSize);
    AlgResourceCtx resCtx;
    resCtx.DeSerialize(serializedContext);
    CHK_PRT_RET(resCtx.threads.empty(),
        HCCL_ERROR("ExecOp: resource context has no execution thread"), HCCL_E_PARA);

    if (param.rankSize == 1) {
        return CopyBytesOnThread(resCtx.threads[0], param.outputPtr, param.inputPtr, recvBytes);
    }

    CHK_PTR_NULL(resCtx.localBuffer.addr);
    CHK_PRT_RET(resCtx.ccuKernels.size() != 1,
        HCCL_ERROR("ExecOp: expected one CCU kernel, got[%zu]", resCtx.ccuKernels.size()), HCCL_E_PARA);

    uint64_t inputToken = 0;
    CcuResult ccuRet = HcommCcuGetMemToken(
        reinterpret_cast<uint64_t>(param.inputPtr), sendBytes, &inputToken);
    if (ccuRet != CCU_SUCCESS) {
        return ConvertCcuToHccl(ccuRet);
    }
    uint64_t outputToken = 0;
    ccuRet = HcommCcuGetMemToken(
        reinterpret_cast<uint64_t>(param.outputPtr), recvBytes, &outputToken);
    if (ccuRet != CCU_SUCCESS) {
        return ConvertCcuToHccl(ccuRet);
    }
    uint64_t scratchToken = 0;
    ccuRet = HcommCcuGetMemToken(
        reinterpret_cast<uint64_t>(resCtx.localBuffer.addr), resCtx.localBuffer.size, &scratchToken);
    if (ccuRet != CCU_SUCCESS) {
        return ConvertCcuToHccl(ccuRet);
    }

    const uint64_t maxReducePieces = std::max<uint64_t>(1, param.rankSize / 2);
    const uint64_t reductionChunkLimit = MAX_DATA_SIZE / maxReducePieces;
    uint64_t mainChunk = std::min<uint64_t>(
        {MAX_DATA_SIZE, reductionChunkLimit, resCtx.localBuffer.size / param.rankSize});
    mainChunk = AlignDown(mainChunk, CHUNK_ALIGN);
    CHK_PRT_RET(mainChunk < sizeof(float),
        HCCL_ERROR("ExecOp: HCCL Buffer cannot hold one FP32 value per rank"), HCCL_E_PARA);

    for (uint64_t processedBytes = 0; processedBytes < recvBytes;) {
        const uint64_t currentBytes = std::min(mainChunk, recvBytes - processedBytes);
        const uint64_t inputSliceOffset = param.myRank * recvBytes + processedBytes;
        const uint64_t outputOffset = processedBytes;
        const std::array<uint64_t, TASK_ARG_COUNT> taskArgs = {
            reinterpret_cast<uint64_t>(param.inputPtr), inputToken,
            reinterpret_cast<uint64_t>(param.outputPtr), outputToken,
            reinterpret_cast<uint64_t>(resCtx.localBuffer.addr), scratchToken,
            inputSliceOffset, outputOffset, currentBytes,
        };
        const CcuResult launchRet = HcommCcuKernelLaunch(
            resCtx.threads[0], resCtx.ccuKernels[0], taskArgs.data(), TASK_ARG_COUNT);
        if (launchRet != CCU_SUCCESS) {
            return ConvertCcuToHccl(launchRet);
        }
        processedBytes += currentBytes;
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
