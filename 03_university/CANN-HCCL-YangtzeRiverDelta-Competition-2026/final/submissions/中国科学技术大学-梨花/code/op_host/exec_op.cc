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
#include <cstdint>
#include <vector>

#include "ccu_launch.h"
#include "ccu_res.h"

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {
constexpr uint64_t M2M_TASK_ARG_NUM = 11;
constexpr uint64_t DIRECT_TASK_ARG_NUM = 4;
constexpr uint64_t HIERARCHY_TASK_ARG_NUM = 17;
constexpr uint32_t HIERARCHY_MAIN_THREAD_IDX = 0;
constexpr uint32_t HIERARCHY_SLAVE_THREAD_IDX = 1;
constexpr uint32_t HIERARCHY_THREAD_NOTIFY_IDX = 0;

HcclResult LaunchBroadcastSlicePair(const AlgResourceCtx &resCtx, uint64_t inputAddr, uint64_t outputAddr,
    uint64_t token, uint64_t firstDataSize, uint64_t secondDataSize, uint32_t myRank,
    uint32_t sliceCount)
{
    const uint64_t firstNormalSliceSize = firstDataSize / sliceCount;
    const uint64_t firstLastSliceSize = firstNormalSliceSize + firstDataSize % sliceCount;
    const uint64_t firstAllgatherOffset = myRank < sliceCount ? firstNormalSliceSize * myRank : 0;
    const uint64_t secondNormalSliceSize = secondDataSize / sliceCount;
    const uint64_t secondLastSliceSize = secondNormalSliceSize + secondDataSize % sliceCount;
    const uint64_t secondAllgatherOffset = myRank < sliceCount ? secondNormalSliceSize * myRank : 0;
    std::vector<uint64_t> taskArgs = {
        inputAddr,
        outputAddr,
        token,
        firstNormalSliceSize,
        firstLastSliceSize,
        firstAllgatherOffset,
        secondNormalSliceSize,
        secondLastSliceSize,
        secondAllgatherOffset,
        firstDataSize,
        secondDataSize,
    };

    CcuResult launchRet = HcommCcuKernelLaunch(resCtx.threads[0],
        resCtx.ccuKernels[BROADCAST_KERNEL_IDX], taskArgs.data(), M2M_TASK_ARG_NUM);
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[LaunchBroadcastSlicePair] kernel launch failed, ccuRet[%d]", launchRet);
        return ConvertCcuToHccl(launchRet);
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchDirectKernel(const AlgResourceCtx &resCtx, uint32_t threadIdx, uint32_t kernelIdx,
    uint64_t inputAddr, uint64_t outputAddr, uint64_t token, uint64_t dataSize)
{
    CHK_PRT_RET(threadIdx >= resCtx.threads.size() || kernelIdx >= resCtx.ccuKernels.size(),
        HCCL_ERROR("[LaunchDirectKernel] invalid thread/kernel index[%u,%u]", threadIdx, kernelIdx),
        HCCL_E_INTERNAL);
    std::vector<uint64_t> taskArgs = {
        inputAddr,
        outputAddr,
        token,
        dataSize,
    };

    CcuResult launchRet = HcommCcuKernelLaunch(
        resCtx.threads[threadIdx], resCtx.ccuKernels[kernelIdx], taskArgs.data(), DIRECT_TASK_ARG_NUM);
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[LaunchDirectKernel] kernel[%u] launch failed, ccuRet[%d]", kernelIdx, launchRet);
        return ConvertCcuToHccl(launchRet);
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchHierarchicalPhase(const AlgResourceCtx &resCtx, uint32_t threadIdx, uint32_t kernelIdx,
    uint64_t inputAddr, uint64_t outputAddr, uint64_t token, uint64_t firstDataSize,
    uint64_t secondDataSize, uint32_t phase)
{
    CHK_PRT_RET(resCtx.hierarchyLaneCount == 0 || threadIdx >= resCtx.threads.size(),
        HCCL_ERROR("[LaunchHierarchicalPhase] invalid lane count"), HCCL_E_INTERNAL);
    const uint64_t firstNormalLaneSize = firstDataSize / resCtx.hierarchyLaneCount;
    const uint64_t firstLastLaneSize = firstNormalLaneSize + firstDataSize % resCtx.hierarchyLaneCount;
    const uint64_t firstNormalSourceSize = firstNormalLaneSize / 2;
    const uint64_t firstNormalDestinationSize = firstNormalLaneSize - firstNormalSourceSize;
    const uint64_t firstLastSourceSize = firstLastLaneSize / 2;
    const uint64_t firstLastDestinationSize = firstLastLaneSize - firstLastSourceSize;
    const uint64_t secondSourceHalfSize = secondDataSize / 2;
    const uint64_t secondDestinationHalfSize = secondDataSize - secondSourceHalfSize;
    const uint64_t secondSourceNormalLaneSize = secondSourceHalfSize / resCtx.hierarchyLaneCount;
    const uint64_t secondSourceLastLaneSize =
        secondSourceNormalLaneSize + secondSourceHalfSize % resCtx.hierarchyLaneCount;
    const uint64_t secondDestinationNormalLaneSize = secondDestinationHalfSize / resCtx.hierarchyLaneCount;
    const uint64_t secondDestinationLastLaneSize =
        secondDestinationNormalLaneSize + secondDestinationHalfSize % resCtx.hierarchyLaneCount;
    std::vector<uint64_t> taskArgs = {
        inputAddr,
        outputAddr,
        token,
        firstNormalLaneSize,
        firstLastLaneSize,
        firstNormalSourceSize,
        firstNormalDestinationSize,
        firstLastSourceSize,
        firstLastDestinationSize,
        secondSourceHalfSize,
        secondDestinationHalfSize,
        secondSourceNormalLaneSize,
        secondSourceLastLaneSize,
        secondDestinationNormalLaneSize,
        secondDestinationLastLaneSize,
        firstDataSize,
        phase,
    };

    CcuResult launchRet = HcommCcuKernelLaunch(
        resCtx.threads[threadIdx], resCtx.ccuKernels[kernelIdx], taskArgs.data(), HIERARCHY_TASK_ARG_NUM);
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[LaunchHierarchicalPhase] kernel[%u] phase[%u] launch failed, ccuRet[%d]",
            kernelIdx, phase, launchRet);
        return ConvertCcuToHccl(launchRet);
    }
    return HCCL_SUCCESS;
}

HcclResult PreSyncHierarchyThreads(const AlgResourceCtx &resCtx)
{
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        resCtx.threads[HIERARCHY_MAIN_THREAD_IDX], resCtx.threads[HIERARCHY_SLAVE_THREAD_IDX],
        HIERARCHY_THREAD_NOTIFY_IDX)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        resCtx.threads[HIERARCHY_SLAVE_THREAD_IDX], HIERARCHY_THREAD_NOTIFY_IDX, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult PostSyncHierarchyThreads(const AlgResourceCtx &resCtx)
{
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        resCtx.threads[HIERARCHY_MAIN_THREAD_IDX], HIERARCHY_THREAD_NOTIFY_IDX, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        resCtx.threads[HIERARCHY_SLAVE_THREAD_IDX], resCtx.threads[HIERARCHY_MAIN_THREAD_IDX],
        HIERARCHY_THREAD_NOTIFY_IDX)));
    return HCCL_SUCCESS;
}

HcclResult LaunchParallelDirectBroadcast(const AlgResourceCtx &resCtx, uint64_t inputAddr,
    uint64_t outputAddr, uint64_t token, uint64_t dataSize)
{
    if (resCtx.ccuKernels.size() == 1) {
        return LaunchDirectKernel(resCtx, 0, 0, inputAddr, outputAddr, token, dataSize);
    }
    CHK_PRT_RET(resCtx.ccuKernels.size() != 2 || resCtx.threads.size() < 2,
        HCCL_ERROR("[LaunchParallelDirectBroadcast] invalid kernel/thread count[%zu,%zu]",
            resCtx.ccuKernels.size(), resCtx.threads.size()), HCCL_E_INTERNAL);

    CHK_RET(PreSyncHierarchyThreads(resCtx));
    CHK_RET(LaunchDirectKernel(resCtx, HIERARCHY_MAIN_THREAD_IDX, 0,
        inputAddr, outputAddr, token, dataSize));
    CHK_RET(LaunchDirectKernel(resCtx, HIERARCHY_SLAVE_THREAD_IDX, 1,
        inputAddr, outputAddr, token, dataSize));
    CHK_RET(PostSyncHierarchyThreads(resCtx));
    return HCCL_SUCCESS;
}

HcclResult LaunchHierarchicalPhasePair(const AlgResourceCtx &resCtx, uint64_t inputAddr,
    uint64_t outputAddr, uint64_t token, uint64_t firstDataSize, uint64_t secondDataSize,
    uint32_t phase)
{
    CHK_RET(PreSyncHierarchyThreads(resCtx));
    CHK_RET(LaunchHierarchicalPhase(resCtx, HIERARCHY_MAIN_THREAD_IDX,
        BROADCAST_HIERARCHY_LAYER0_KERNEL_IDX, inputAddr, outputAddr, token,
        firstDataSize, secondDataSize, phase));
    CHK_RET(LaunchHierarchicalPhase(resCtx, HIERARCHY_SLAVE_THREAD_IDX,
        BROADCAST_HIERARCHY_LAYER1_KERNEL_IDX, inputAddr, outputAddr, token,
        firstDataSize, secondDataSize, phase));
    CHK_RET(PostSyncHierarchyThreads(resCtx));
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param)
{
    CHK_PTR_NULL(param.resCtx);
    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> seq(ctx, ctx + param.ctxSize);
    AlgResourceCtx resCtx;
    resCtx.DeSerialize(seq);
    if (resCtx.threads.empty()) {
        resCtx.threads.push_back(param.cpuThread);
    } else {
        resCtx.threads[0] = param.cpuThread;
    }
    resCtx.ccuThread = param.cpuThread;

    auto dataTypeIt = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(dataTypeIt == SIZE_TABLE.end(),
        HCCL_ERROR("[ExecOp] unsupported dataType[%d]", static_cast<int>(param.dataType)), HCCL_E_NOT_SUPPORT);
    const uint64_t dataTypeSize = dataTypeIt->second;
    const uint64_t dataSize = param.count * dataTypeSize;

    if (param.count == 0 || param.rankSize == 1) {
        return HCCL_SUCCESS;
    }
    const bool supportedAlgorithm = resCtx.algorithmMode == BROADCAST_ALG_DIRECT ||
        resCtx.algorithmMode == BROADCAST_ALG_MESH1D_MEM2MEM ||
        resCtx.algorithmMode == BROADCAST_ALG_NHR1D_MEM2MEM ||
        resCtx.algorithmMode == BROADCAST_ALG_PARALLEL_HIERARCHICAL ||
        resCtx.algorithmMode == BROADCAST_ALG_PARALLEL_DIRECT;
    CHK_PRT_RET(!supportedAlgorithm,
        HCCL_ERROR("[ExecOp] unsupported algorithmMode[%u]", resCtx.algorithmMode), HCCL_E_NOT_SUPPORT);
    uint32_t requiredKernelNum = BROADCAST_KERNEL_NUM;
    uint32_t requiredThreadNum = 1;
    if (resCtx.algorithmMode == BROADCAST_ALG_PARALLEL_HIERARCHICAL) {
        requiredKernelNum += resCtx.hierarchyHasLayer1Kernel;
        requiredThreadNum = 2;
    } else if (resCtx.algorithmMode == BROADCAST_ALG_PARALLEL_DIRECT && param.myRank == param.root) {
        requiredKernelNum = 2;
        requiredThreadNum = 2;
    }
    CHK_PRT_RET(resCtx.threads.size() < requiredThreadNum || resCtx.ccuKernels.size() < requiredKernelNum,
        HCCL_ERROR("[ExecOp] missing CCU thread or kernel"), HCCL_E_INTERNAL);

    uint64_t token = 0;
    const uint64_t baseInputAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t baseOutputAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    CHK_RET_CCU(HcommCcuGetMemToken(baseOutputAddr, dataSize, &token));

    if (resCtx.algorithmMode == BROADCAST_ALG_DIRECT) {
        CHK_RET(LaunchDirectKernel(
            resCtx, 0, BROADCAST_KERNEL_IDX, baseInputAddr, baseOutputAddr, token, dataSize));
        return HCCL_SUCCESS;
    }
    if (resCtx.algorithmMode == BROADCAST_ALG_PARALLEL_DIRECT) {
        HCCL_INFO("[ExecOp] small packet kernel[ParallelDirect], rankSize[%u], dataSize[%llu], kernels[%zu]",
            param.rankSize, static_cast<unsigned long long>(dataSize), resCtx.ccuKernels.size());
        CHK_RET(LaunchParallelDirectBroadcast(
            resCtx, baseInputAddr, baseOutputAddr, token, dataSize));
        return HCCL_SUCCESS;
    }

    const uint64_t maxCountPerLoop = MAX_DATA_SIZE / dataTypeSize;
    if (resCtx.algorithmMode == BROADCAST_ALG_PARALLEL_HIERARCHICAL) {
        CHK_PRT_RET(resCtx.hierarchyHasLayer1Kernel == 0,
            HCCL_ERROR("[ExecOp] 2D hierarchy requires a layer-1 kernel"), HCCL_E_INTERNAL);
        HCCL_INFO("[ExecOp] large packet kernel[Parallel2D], rankSize[%u], dataSize[%llu], lanes[%u]",
            param.rankSize, static_cast<unsigned long long>(dataSize), resCtx.hierarchyLaneCount);
        const uint64_t maxCountPerPair = maxCountPerLoop * 2;
        const bool useWeightedRank16Split = BROADCAST_ENABLE_RANK16_WEIGHTED_SPLIT &&
            param.rankSize == BROADCAST_PARALLEL_2D_RANK_SIZE;
        const uint32_t splitNumerator = useWeightedRank16Split ?
            BROADCAST_RANK16_WEIGHTED_SPLIT_NUMERATOR : 1U;
        const uint32_t splitDenominator = useWeightedRank16Split ?
            BROADCAST_RANK16_WEIGHTED_SPLIT_DENOMINATOR : 2U;
        HCCL_INFO("[ExecOp] Parallel2D split meshFirst[%u/%u], closFirst[%u/%u]",
            splitNumerator, splitDenominator, splitDenominator - splitNumerator, splitDenominator);
        uint64_t processedCount = 0;
        while (processedCount < param.count) {
            const uint64_t pairCount = std::min(maxCountPerPair, param.count - processedCount);
            // 加权模式按官方 SCATTER 比例向下取整；关闭时精确复现原有 50/50 行为。
            const uint64_t firstCount = BroadcastSplitCountByRatio(
                pairCount, splitNumerator, splitDenominator, !useWeightedRank16Split);
            const uint64_t secondCount = pairCount - firstCount;
            const uint64_t firstSize = firstCount * dataTypeSize;
            const uint64_t secondSize = secondCount * dataTypeSize;
            const uint64_t offset = processedCount * dataTypeSize;

            HCCL_DEBUG("[ExecOp] Parallel2D window offset[%llu], firstCount[%llu], secondCount[%llu], "
                "firstSize[%llu], secondSize[%llu]",
                static_cast<unsigned long long>(offset),
                static_cast<unsigned long long>(firstCount),
                static_cast<unsigned long long>(secondCount),
                static_cast<unsigned long long>(firstSize),
                static_cast<unsigned long long>(secondSize));

            CHK_RET(LaunchHierarchicalPhasePair(resCtx, baseInputAddr + offset,
                baseOutputAddr + offset, token, firstSize, secondSize,
                BROADCAST_HIERARCHY_PHASE_SCATTER_0));
            CHK_RET(LaunchHierarchicalPhasePair(resCtx, baseInputAddr + offset,
                baseOutputAddr + offset, token, firstSize, secondSize,
                BROADCAST_HIERARCHY_PHASE_SCATTER_1));
            CHK_RET(LaunchHierarchicalPhasePair(resCtx, baseInputAddr + offset,
                baseOutputAddr + offset, token, firstSize, secondSize,
                BROADCAST_HIERARCHY_PHASE_ALLGATHER_0));
            CHK_RET(LaunchHierarchicalPhasePair(resCtx, baseInputAddr + offset,
                baseOutputAddr + offset, token, firstSize, secondSize,
                BROADCAST_HIERARCHY_PHASE_ALLGATHER_1));
            processedCount += pairCount;
        }
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[ExecOp] large packet kernel[%s], rankSize[%u], dataSize[%llu]",
        resCtx.algorithmMode == BROADCAST_ALG_NHR1D_MEM2MEM ? "NHR1D" : "Mesh1D",
        param.rankSize, static_cast<unsigned long long>(dataSize));
    uint64_t processedCount = 0;
    while (processedCount < param.count) {
        const uint64_t firstCount = std::min(maxCountPerLoop, param.count - processedCount);
        const uint64_t remainingCount = param.count - processedCount - firstCount;
        const uint64_t secondCount = std::min(maxCountPerLoop, remainingCount);
        const uint64_t firstSize = firstCount * dataTypeSize;
        const uint64_t secondSize = secondCount * dataTypeSize;
        const uint64_t offset = processedCount * dataTypeSize;

        CHK_RET(LaunchBroadcastSlicePair(resCtx, baseInputAddr + offset, baseOutputAddr + offset,
            token, firstSize, secondSize, param.myRank, param.rankSize));
        processedCount += firstCount + secondCount;
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl