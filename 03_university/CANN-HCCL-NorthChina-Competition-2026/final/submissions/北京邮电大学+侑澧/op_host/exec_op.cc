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
#include <vector>

#include <ccu/ccu_res.h>

#include "ccu_launch.h"
#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {

static HcclResult PreSyncThreads(const AlgResourceCtx &resCtx)
{
    uint32_t threadCount = static_cast<uint32_t>(resCtx.threads.size());
    if (threadCount <= 1) {
        return HCCL_SUCCESS;
    }

    for (uint32_t threadIdx = 1; threadIdx < threadCount; ++threadIdx) {
        uint32_t notifyIdx = threadIdx - 1;
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(resCtx.threads[0], resCtx.threads[threadIdx], notifyIdx)));
    }
    for (uint32_t threadIdx = 1; threadIdx < threadCount; ++threadIdx) {
        uint32_t notifyIdx = threadIdx - 1;
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(resCtx.threads[threadIdx], notifyIdx, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

static HcclResult PostSyncThreads(const AlgResourceCtx &resCtx)
{
    uint32_t threadCount = static_cast<uint32_t>(resCtx.threads.size());
    if (threadCount <= 1) {
        return HCCL_SUCCESS;
    }

    for (uint32_t threadIdx = 1; threadIdx < threadCount; ++threadIdx) {
        uint32_t notifyIdx = threadIdx - 1;
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(resCtx.threads[0], notifyIdx, CUSTOM_TIMEOUT)));
    }
    for (uint32_t threadIdx = 1; threadIdx < threadCount; ++threadIdx) {
        uint32_t notifyIdx = threadIdx - 1;
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(resCtx.threads[threadIdx], resCtx.threads[0], notifyIdx)));
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchCcuKernels(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t inputAddr,
    uint64_t outputAddr, uint64_t token, uint64_t dataSize, uint64_t normalSliceSize,
    uint64_t lastSliceSize, uint64_t repeatNum)
{
    uint32_t kernelCount = static_cast<uint32_t>(resCtx.ccuKernels.size());
    uint32_t threadCount = static_cast<uint32_t>(resCtx.threads.size());
    CHK_PRT_RET(kernelCount == 0 || kernelCount != threadCount,
        HCCL_ERROR("[LaunchCcuKernels] Kernel count[%u] does not match thread count[%u].",
            kernelCount, threadCount),
        HCCL_E_INTERNAL);

    uint64_t currentRankOutputOffset = dataSize * param.myRank;
    std::array<uint64_t, 7> taskArgs = {
        inputAddr,
        outputAddr,
        token,
        currentRankOutputOffset,
        normalSliceSize,
        lastSliceSize,
        repeatNum,
    };

    CHK_RET(PreSyncThreads(resCtx));
    for (uint32_t kernelIdx = 0; kernelIdx < kernelCount; ++kernelIdx) {
        CcuResult ccuRet = HcommCcuKernelLaunch(
            resCtx.threads[kernelIdx], resCtx.ccuKernels[kernelIdx], taskArgs.data(), taskArgs.size());
        CHK_PRT_RET(ccuRet != CCU_SUCCESS,
            HCCL_ERROR("[LaunchCcuKernels] Kernel[%u] launch failed, ret[%d].", kernelIdx, ccuRet),
            ConvertCcuToHccl(ccuRet));
    }
    CHK_RET(PostSyncThreads(resCtx));
    return HCCL_SUCCESS;
}

static HcclResult LaunchPipelineStage(const AlgResourceCtx &resCtx,
    std::array<uint64_t, 8> &taskArgs)
{
    CHK_RET(PreSyncThreads(resCtx));
    for (uint32_t kernelIdx = 0; kernelIdx < 2; ++kernelIdx) {
        CcuResult ccuRet = HcommCcuKernelLaunch(
            resCtx.threads[kernelIdx], resCtx.ccuKernels[kernelIdx],
            taskArgs.data(), taskArgs.size());
        CHK_PRT_RET(ccuRet != CCU_SUCCESS,
            HCCL_ERROR("[LaunchPipelineStage] Kernel[%u] launch failed, ret[%d].", kernelIdx, ccuRet),
            ConvertCcuToHccl(ccuRet));
    }
    CHK_RET(PostSyncThreads(resCtx));
    return HCCL_SUCCESS;
}

static HcclResult LaunchPipeline(const AlgResourceCtx &resCtx, uint64_t inputAddr,
    uint64_t outputAddr, uint64_t token, uint64_t dataSize)
{
    CHK_PRT_RET(resCtx.threads.size() != 2 || resCtx.ccuKernels.size() != 2,
        HCCL_ERROR("[LaunchPipeline] Expected two threads and two kernels."),
        HCCL_E_INTERNAL);

    constexpr uint64_t splitAlign = 128;
    uint64_t meshBytes = dataSize / 2;
    if (resCtx.pipelineMode == 2) {
        meshBytes = dataSize * 3 / 5;
    }
    meshBytes = (meshBytes / splitAlign) * splitAlign;
    uint64_t nhrOffset = meshBytes;
    uint64_t nhrBytes = dataSize - meshBytes;
    CHK_PRT_RET(meshBytes == 0 || nhrBytes == 0 ||
        meshBytes > MAX_DATA_SIZE || nhrBytes > MAX_DATA_SIZE,
        HCCL_ERROR("[LaunchPipeline] Invalid split, data[%llu], mesh[%llu], nhr[%llu].",
            static_cast<unsigned long long>(dataSize),
            static_cast<unsigned long long>(meshBytes),
            static_cast<unsigned long long>(nhrBytes)),
        HCCL_E_PARA);

    std::array<uint64_t, 8> taskArgs = {
        inputAddr,
        outputAddr,
        token,
        dataSize,
        meshBytes,
        nhrOffset,
        nhrBytes,
        0,
    };
    CHK_RET(LaunchPipelineStage(resCtx, taskArgs));
    taskArgs[7] = 1;
    CHK_RET(LaunchPipelineStage(resCtx, taskArgs));
    return HCCL_SUCCESS;
}

HcclResult ExecOp(const OpParam &param)
{
    // 反序列化
    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> seq(ctx, ctx + param.ctxSize);
    AlgResourceCtx resCtx;
    resCtx.DeSerialize(seq);

    auto typeSizeIt = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(typeSizeIt == SIZE_TABLE.end(),
        HCCL_ERROR("[ExecOp] Unsupported dataType[%d].", static_cast<int32_t>(param.dataType)),
        HCCL_E_NOT_SUPPORT);

    uint64_t dataTypeSize = typeSizeIt->second;
    uint64_t dataSize = param.count * dataTypeSize;
    if (dataSize == 0) {
        return HCCL_SUCCESS;
    }

    if (param.rankSize == 1) {
        return static_cast<HcclResult>(
            HcommLocalCopyOnThread(param.cpuThread, param.outputPtr, param.inputPtr, dataSize));
    }

    CHK_PRT_RET(resCtx.ccuKernels.empty(),
        HCCL_ERROR("[ExecOp] CCU kernel is missing."),
        HCCL_E_INTERNAL);

    uint64_t baseInputAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    uint64_t baseOutputAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    uint64_t token = 0;
    CcuResult ccuRet = HcommCcuGetMemToken(baseInputAddr, dataSize, &token);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS,
        HCCL_ERROR("[ExecOp] Get memory token failed, ret[%d].", ccuRet),
        ConvertCcuToHccl(ccuRet));

    if (resCtx.pipelineMode != 0) {
        return LaunchPipeline(resCtx, baseInputAddr, baseOutputAddr, token, dataSize);
    }

    uint64_t normalSliceSize = std::min<uint64_t>(MAX_DATA_SIZE, dataSize);
    uint64_t sliceCount = (dataSize + MAX_DATA_SIZE - 1) / MAX_DATA_SIZE;
    uint64_t lastSliceSize = dataSize - normalSliceSize * (sliceCount - 1);
    uint64_t repeatNum = UINT64_MAX - sliceCount;
    return LaunchCcuKernels(param, resCtx, baseInputAddr, baseOutputAddr, token, dataSize,
        normalSliceSize, lastSliceSize, repeatNum);
}
} // namespace ops_hccl
