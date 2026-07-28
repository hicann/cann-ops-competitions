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
#include <hccl/hcomm_primitives.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {
constexpr uint64_t FP32_SIZE = sizeof(float);
constexpr uint64_t CCU_MS_SIZE = 4096;
constexpr uint64_t REDUCE_LOOP_COUNT = 16;
constexpr uint64_t MS_REDUCE_SLICE_LIMIT = 1024 * 1024;
constexpr size_t PHASE_ARG_INDEX = 6;

constexpr uint64_t SetBits(uint16_t end)
{
    return (uint64_t{1} << (end + 1)) - uint64_t{1};
}

uint64_t GetMaxLoopIterNum()
{
    constexpr uint16_t LOOP_NUM_BIT_NUM = 12;
    return SetBits(LOOP_NUM_BIT_NUM);
}

uint64_t GetParallelParam(uint64_t repeatNum, uint64_t repeatLoopIndex,
    uint64_t totalLoopNum)
{
    constexpr uint16_t REPEAT_BIT_NUM = 7;
    constexpr uint16_t REPEAT_NUM_SHIFT_BIT = 55;
    constexpr uint16_t REPEAT_LOOP_BIT_NUM = 7;
    constexpr uint16_t REPEAT_LOOP_SHIFT_BIT = 48;
    constexpr uint16_t TOTAL_LOOP_BIT_NUM = 7;
    constexpr uint16_t TOTAL_LOOP_SHIFT_BIT = 41;
    return ((repeatNum & SetBits(REPEAT_BIT_NUM)) << REPEAT_NUM_SHIFT_BIT) |
        ((repeatLoopIndex & SetBits(REPEAT_LOOP_BIT_NUM)) << REPEAT_LOOP_SHIFT_BIT) |
        ((totalLoopNum & SetBits(TOTAL_LOOP_BIT_NUM)) << TOTAL_LOOP_SHIFT_BIT);
}

void AppendReduceLoopArgs(uint64_t size, std::vector<uint64_t> &args)
{
    const uint64_t loopSize = REDUCE_LOOP_COUNT * CCU_MS_SIZE;
    const uint64_t maxSize = loopSize * (GetMaxLoopIterNum() + 1);
    uint64_t fullLoopCount = size / loopSize;
    uint64_t msCount = (size - fullLoopCount * loopSize) / CCU_MS_SIZE;
    uint64_t residual = size - fullLoopCount * loopSize - msCount * CCU_MS_SIZE;

    if (size == maxSize) {
        fullLoopCount = GetMaxLoopIterNum();
        msCount = REDUCE_LOOP_COUNT - 1;
        residual = CCU_MS_SIZE;
    }

    uint64_t parallelParam = 0;
    uint64_t tailSize = 0;
    if (msCount != 0 && residual == 0) {
        parallelParam = GetParallelParam(msCount - 1, 0, 1);
        tailSize = CCU_MS_SIZE;
    } else if (msCount == 0 && residual != 0) {
        parallelParam = GetParallelParam(0, 0, 1);
        tailSize = residual;
    } else if (msCount != 0) {
        parallelParam = GetParallelParam(msCount - 1, 1, 2);
        tailSize = residual;
    }

    args.push_back(loopSize * fullLoopCount);
    args.push_back(fullLoopCount);
    args.push_back(parallelParam);
    args.push_back(tailSize);
    args.push_back(size <= MS_REDUCE_SLICE_LIMIT ? 1 : 0);
}

HcclResult LaunchSegment(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t inputAddr,
    uint64_t outputAddr, uint64_t token, uint64_t segmentBytes)
{
    const uint64_t elementCount = segmentBytes / FP32_SIZE;
    const uint64_t normalSliceElements =
        (elementCount + param.rankSize - 1) / param.rankSize;
    const uint64_t sliceBegin = std::min<uint64_t>(
        normalSliceElements * param.myRank, elementCount);
    const uint64_t sliceEnd = std::min<uint64_t>(
        sliceBegin + normalSliceElements, elementCount);
    const uint64_t sliceOffset = sliceBegin * FP32_SIZE;
    const uint64_t sliceBytes = (sliceEnd - sliceBegin) * FP32_SIZE;
    const uint64_t scratchBytes = normalSliceElements * FP32_SIZE * param.rankSize;
    if (scratchBytes > resCtx.localBuffer.size) {
        HCCL_ERROR("[LaunchSegment] scratch size %llu exceeds HCCL buffer %llu",
            static_cast<unsigned long long>(scratchBytes),
            static_cast<unsigned long long>(resCtx.localBuffer.size));
        return HCCL_E_MEMORY;
    }

    std::vector<uint64_t> taskArgs = {
        inputAddr,
        outputAddr,
        token,
        reinterpret_cast<uint64_t>(resCtx.localBuffer.addr),
        sliceOffset,
        sliceBytes,
        0,
    };
    AppendReduceLoopArgs(sliceBytes, taskArgs);

    if (resCtx.ccuKernels.size() == 1 && resCtx.threads.size() == 1) {
        CcuResult ret = HcommCcuKernelLaunch(resCtx.threads[0], resCtx.ccuKernels[0],
            taskArgs.data(), static_cast<uint32_t>(taskArgs.size()));
        return ConvertCcuToHccl(ret);
    }

    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        resCtx.threads[0], resCtx.threads[1], 0)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        resCtx.threads[1], 0, CUSTOM_TIMEOUT)));
    for (size_t i = 0; i < 2; ++i) {
        CcuResult ret = HcommCcuKernelLaunch(resCtx.threads[i], resCtx.ccuKernels[i],
            taskArgs.data(), static_cast<uint32_t>(taskArgs.size()));
        if (ret != CCU_SUCCESS) {
            HCCL_ERROR("[LaunchSegment] CCU kernel %zu launch failed: %d", i, ret);
            return ConvertCcuToHccl(ret);
        }
    }
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        resCtx.threads[0], 0, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        resCtx.threads[1], resCtx.threads[0], 0)));
    CcuResult combineRet = HcommCcuKernelLaunch(resCtx.threads[0], resCtx.ccuKernels[2],
        taskArgs.data(), static_cast<uint32_t>(taskArgs.size()));
    if (combineRet != CCU_SUCCESS) {
        return ConvertCcuToHccl(combineRet);
    }

    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        resCtx.threads[0], resCtx.threads[1], 0)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        resCtx.threads[1], 0, CUSTOM_TIMEOUT)));
    taskArgs[PHASE_ARG_INDEX] = 1;
    for (size_t i = 0; i < 2; ++i) {
        CcuResult ret = HcommCcuKernelLaunch(resCtx.threads[i], resCtx.ccuKernels[i],
            taskArgs.data(), static_cast<uint32_t>(taskArgs.size()));
        if (ret != CCU_SUCCESS) {
            HCCL_ERROR("[LaunchSegment] CCU broadcast kernel %zu launch failed: %d", i, ret);
            return ConvertCcuToHccl(ret);
        }
    }
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        resCtx.threads[0], 0, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        resCtx.threads[1], resCtx.threads[0], 0)));
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param)
{
    char *ctx = static_cast<char *>(param.resCtx);
    CHK_PTR_NULL(ctx);
    std::vector<char> sequence(ctx, ctx + param.ctxSize);
    AlgResourceCtx resCtx;
    resCtx.DeSerialize(sequence);

    const uint64_t dataBytes = param.count * FP32_SIZE;
    if (dataBytes == 0) {
        return HCCL_SUCCESS;
    }
    if (param.rankSize == 1) {
        return static_cast<HcclResult>(HcommLocalCopyOnThread(
            param.cpuThread, param.outputPtr, param.inputPtr, dataBytes));
    }
    const bool singleGroup = resCtx.ccuKernels.size() == 1 && resCtx.threads.size() == 1;
    const bool dualGroup = resCtx.ccuKernels.size() == 3 && resCtx.threads.size() == 2;
    if ((!singleGroup && !dualGroup) ||
        resCtx.localBuffer.addr == nullptr) {
        HCCL_ERROR("[ExecOp] incomplete CCU resource context");
        return HCCL_E_INTERNAL;
    }

    uint64_t token = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(param.inputPtr), dataBytes, &token));

    const uint64_t segmentAlignment = FP32_SIZE * param.rankSize;
    uint64_t maxSegmentBytes = std::min<uint64_t>(MAX_DATA_SIZE, resCtx.localBuffer.size);
    maxSegmentBytes = (maxSegmentBytes / segmentAlignment) * segmentAlignment;
    if (maxSegmentBytes == 0) {
        return HCCL_E_MEMORY;
    }

    const uint64_t baseInput = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t baseOutput = reinterpret_cast<uint64_t>(param.outputPtr);
    uint64_t processedBytes = 0;
    while (processedBytes < dataBytes) {
        const uint64_t segmentBytes = std::min(maxSegmentBytes, dataBytes - processedBytes);
        CHK_RET(LaunchSegment(param, resCtx, baseInput + processedBytes,
            baseOutput + processedBytes, token, segmentBytes));
        processedBytes += segmentBytes;
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
