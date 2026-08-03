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

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>

#include "log.h"
#include "custom.h"
#include "exec_op.h"
#include "ccu_kernel.h"

namespace ops_hccl {
namespace {
// The physical 2x8 path still has a dependency barrier between its two
// phases. Balance the slowest edge inside each phase: 2(1-p)/47 = 8p/180,
// giving p=45/92. Ratios near 5/8 become candidates only after chunk-level
// pipelining removes that coarse phase boundary.
constexpr uint64_t LOCAL_FIRST_NUMERATOR_2X8 = 45;
constexpr uint64_t LOCAL_FIRST_NUMERATOR_8PLUS4 = 2;
constexpr uint64_t LOCAL_FIRST_NUMERATOR_UNIFORM = 1;
constexpr uint64_t LOCAL_FIRST_DENOMINATOR_2X8 = 92;
constexpr uint64_t LOCAL_FIRST_DENOMINATOR_8PLUS4 = 5;
constexpr uint64_t LOCAL_FIRST_DENOMINATOR_UNIFORM = 2;

constexpr uint64_t SetBits(uint16_t end)
{
    return (uint64_t{1} << (end + 1)) - uint64_t{1};
}

uint64_t GetMaxLoopIterNum()
{
    constexpr uint16_t LOOP_NUM_END_BIT = 12;
    return SetBits(LOOP_NUM_END_BIT);
}

uint64_t GetParallelParam(uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum)
{
    constexpr uint16_t REPEAT_END_BIT = 7;
    constexpr uint16_t REPEAT_SHIFT = 55;
    constexpr uint16_t REPEAT_LOOP_END_BIT = 7;
    constexpr uint16_t REPEAT_LOOP_SHIFT = 48;
    constexpr uint16_t TOTAL_LOOP_END_BIT = 7;
    constexpr uint16_t TOTAL_LOOP_SHIFT = 41;
    return ((repeatNum & SetBits(REPEAT_END_BIT)) << REPEAT_SHIFT) |
        ((repeatLoopIndex & SetBits(REPEAT_LOOP_END_BIT)) << REPEAT_LOOP_SHIFT) |
        ((totalLoopNum & SetBits(TOTAL_LOOP_END_BIT)) << TOTAL_LOOP_SHIFT);
}

std::array<uint64_t, 4> CalGoSize(uint64_t size, const LoopGroupConfig &config)
{
    const uint64_t loopSize = config.loopCount * config.memSlice;
    const uint64_t maxSize = loopSize * (GetMaxLoopIterNum() + 1);
    uint64_t fullLoopCount = size / loopSize;
    uint64_t parallelCount = (size - fullLoopCount * loopSize) / config.memSlice;
    uint64_t tailSize = size - fullLoopCount * loopSize - parallelCount * config.memSlice;

    if (size == maxSize) {
        fullLoopCount = GetMaxLoopIterNum();
        parallelCount = config.loopCount - 1;
        tailSize = config.memSlice;
    }

    const uint64_t addressOffset = config.memSlice * config.loopCount * fullLoopCount;
    uint64_t parallelParam = 0;
    uint64_t residual = 0;
    if (parallelCount == 0 && tailSize == 0) {
        parallelParam = 0;
    } else if (parallelCount != 0 && tailSize == 0) {
        parallelParam = GetParallelParam(parallelCount - 1, 0, 1);
        residual = config.memSlice;
    } else if (parallelCount == 0) {
        parallelParam = GetParallelParam(0, 0, 1);
        residual = tailSize;
    } else {
        parallelParam = GetParallelParam(parallelCount - 1, 1, 2);
        residual = tailSize;
    }
    return {addressOffset, fullLoopCount, parallelParam, residual};
}

bool ShouldRunDieKernelsInParallel(const AlgResourceCtx &resource)
{
    return resource.kernelCount > 1 && resource.threadCount > 1;
}

HcclResult PreSyncKernelThreads(const AlgResourceCtx &resource)
{
    for (uint32_t threadIndex = 1; threadIndex < resource.threadCount; ++threadIndex) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(resource.threads[0], resource.threads[threadIndex], 0)));
    }
    for (uint32_t threadIndex = 1; threadIndex < resource.threadCount; ++threadIndex) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThreadWithDefaultTimeout(resource.threads[threadIndex], 0)));
    }
    return HCCL_SUCCESS;
}

HcclResult PostSyncKernelThreads(const AlgResourceCtx &resource)
{
    for (uint32_t threadIndex = 1; threadIndex < resource.threadCount; ++threadIndex) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThreadWithDefaultTimeout(
            resource.threads[0], threadIndex - 1)));
    }
    for (uint32_t threadIndex = 1; threadIndex < resource.threadCount; ++threadIndex) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resource.threads[threadIndex], resource.threads[0], threadIndex - 1)));
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchSlice(ThreadHandle thread, CcuKernelHandle kernel, uint64_t inputAddr, uint64_t outputAddr,
    uint64_t inputToken, uint64_t outputToken, uint64_t currentRankOutputOffset, uint64_t sliceSize,
    bool localCopyNeeded, uint64_t dataSize, uint64_t partOffset, AllGatherExecutionMode executionMode,
    bool compactDirectArgs = false, bool directNhr = false,
    uint64_t otherPartOffset = 0, uint64_t otherPartSize = 0)
{
    if (compactDirectArgs) {
        const std::array<uint64_t, 7> taskArgs = {
            inputAddr,
            outputAddr,
            inputToken,
            outputToken,
            currentRankOutputOffset,
            sliceSize,
            static_cast<uint64_t>(localCopyNeeded),
        };
        const CcuResult launchRet = HcommCcuKernelLaunch(
            thread, kernel, taskArgs.data(), static_cast<uint32_t>(taskArgs.size()));
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("CCU AllGather kernel launch failed: %d", launchRet);
            return ConvertCcuToHccl(launchRet);
        }
        return HCCL_SUCCESS;
    }

    LoopGroupConfig config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount = CCU_MS_LOCAL_COPY_LOOP_COUNT;
    config.memSlice = CCU_MS_SIZE * CCU_LOCAL_COPY_MS_PER_LOOP;
    const auto goSize = CalGoSize(sliceSize, config);

    const std::array<uint64_t, 16> taskArgs = {
        inputAddr,
        outputAddr,
        inputToken,
        outputToken,
        currentRankOutputOffset,
        sliceSize,
        static_cast<uint64_t>(localCopyNeeded),
        goSize[0],
        goSize[1],
        goSize[2],
        goSize[3],
        dataSize,
        partOffset,
        otherPartOffset,
        otherPartSize,
        static_cast<uint64_t>(executionMode),
    };
    constexpr uint32_t DIRECT_TASK_ARG_COUNT = 11;
    constexpr uint32_t DIRECT_NHR_TASK_ARG_COUNT = 12;
    const uint32_t taskArgCount = executionMode != AllGatherExecutionMode::DIRECT ?
        static_cast<uint32_t>(taskArgs.size()) :
        (directNhr ? DIRECT_NHR_TASK_ARG_COUNT : DIRECT_TASK_ARG_COUNT);
    CcuResult launchRet = HcommCcuKernelLaunch(
        thread, kernel, taskArgs.data(), taskArgCount);
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("CCU AllGather kernel launch failed: %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchHierarchicalPhase(const AlgResourceCtx &resource, uint64_t inputAddr,
    uint64_t outputAddr, uint64_t inputToken, uint64_t outputToken, uint64_t currentRankOutputOffset,
    uint64_t dataSize, uint64_t firstPartOffset, uint64_t firstPartSize, uint64_t secondPartOffset,
    uint64_t secondPartSize, AllGatherExecutionMode executionMode)
{
    CHK_RET(PreSyncKernelThreads(resource));
    for (uint32_t kernelIndex = 0; kernelIndex < resource.kernelCount; ++kernelIndex) {
        const bool useFirstPart = resource.purePhysicalGrid ?
            resource.kernelTrafficClass[kernelIndex] == KernelTrafficClass::INTRA_SERVER :
            kernelIndex == 0;
        const uint64_t partOffset = useFirstPart ? firstPartOffset : secondPartOffset;
        const uint64_t partSize = useFirstPart ? firstPartSize : secondPartSize;
        const uint64_t otherPartOffset = useFirstPart ? secondPartOffset : firstPartOffset;
        const uint64_t otherPartSize = useFirstPart ? secondPartSize : firstPartSize;
        const bool localCopyNeeded =
            inputAddr + partOffset != outputAddr + currentRankOutputOffset + partOffset;
        const uint32_t threadIndex = resource.kernelThreadIndex[kernelIndex];
        if (threadIndex >= resource.threadCount) {
            return HCCL_E_INTERNAL;
        }
        CHK_RET(LaunchSlice(resource.threads[threadIndex], resource.ccuKernels[kernelIndex],
            inputAddr, outputAddr, inputToken, outputToken, currentRankOutputOffset,
            partSize, localCopyNeeded, dataSize, partOffset, executionMode,
            false, false, otherPartOffset, otherPartSize));
    }
    CHK_RET(PostSyncKernelThreads(resource));
    return HCCL_SUCCESS;
}

uint64_t CalculateBalancedPart0Size(
    const AlgResourceCtx &resource, uint64_t dataSize, uint64_t dataTypeSize)
{
    const uint64_t elementCount = dataSize / dataTypeSize;
    const uint64_t localFirstNumerator = resource.hierarchical2x8 ?
        LOCAL_FIRST_NUMERATOR_2X8 :
        (resource.purePhysicalGrid ? LOCAL_FIRST_NUMERATOR_UNIFORM :
                                     LOCAL_FIRST_NUMERATOR_8PLUS4);
    const uint64_t localFirstDenominator = resource.hierarchical2x8 ?
        LOCAL_FIRST_DENOMINATOR_2X8 :
        (resource.purePhysicalGrid ? LOCAL_FIRST_DENOMINATOR_UNIFORM :
                                     LOCAL_FIRST_DENOMINATOR_8PLUS4);
    const uint64_t desiredPart0ElementCount =
        (elementCount / localFirstDenominator) * localFirstNumerator +
        ((elementCount % localFirstDenominator) * localFirstNumerator) / localFirstDenominator;
    return desiredPart0ElementCount * dataTypeSize;
}

HcclResult RunHierarchicalWindow(const AlgResourceCtx &resource, uint64_t inputAddr,
    uint64_t outputAddr, uint64_t inputToken, uint64_t outputToken, uint64_t fullDataSize,
    uint64_t dataTypeSize, uint64_t currentRankOutputOffset, uint64_t windowOffset,
    uint64_t windowSize)
{
    const uint64_t balancedPart0Size =
        CalculateBalancedPart0Size(resource, windowSize, dataTypeSize);
    uint64_t part0Size = std::min<uint64_t>(balancedPart0Size, MAX_DATA_SIZE);
    if (windowSize > MAX_DATA_SIZE) {
        part0Size = std::max<uint64_t>(part0Size, windowSize - MAX_DATA_SIZE);
    }
    const uint64_t part1Size = windowSize - part0Size;
    if (part0Size == 0 || part1Size == 0) {
        return HCCL_E_INTERNAL;
    }
    const uint64_t part0Offset = windowOffset;
    const uint64_t part1Offset = windowOffset + part0Size;

    CHK_RET(LaunchHierarchicalPhase(resource, inputAddr, outputAddr, inputToken, outputToken,
        currentRankOutputOffset, fullDataSize, part0Offset, part0Size, part1Offset, part1Size,
        AllGatherExecutionMode::HIERARCHICAL_PHASE_ONE));
    CHK_RET(LaunchHierarchicalPhase(resource, inputAddr, outputAddr, inputToken, outputToken,
        currentRankOutputOffset, fullDataSize, part1Offset, part1Size, part0Offset, part0Size,
        AllGatherExecutionMode::HIERARCHICAL_PHASE_TWO));
    return HCCL_SUCCESS;
}

HcclResult LaunchPhysical8Plus4Phase(const AlgResourceCtx &resource, uint64_t inputAddr,
    uint64_t outputAddr, uint64_t inputToken, uint64_t outputToken,
    uint64_t currentRankOutputOffset, uint64_t fullDataSize, uint64_t sliceSize,
    AllGatherExecutionMode executionMode)
{
    CHK_RET(PreSyncKernelThreads(resource));
    for (uint32_t kernelIndex = 0; kernelIndex < resource.kernelCount; ++kernelIndex) {
        if (executionMode == AllGatherExecutionMode::HIERARCHICAL_PHASE_TWO &&
            resource.kernelTrafficClass[kernelIndex] == KernelTrafficClass::INTER_SERVER) {
            continue;
        }
        const uint32_t threadIndex = resource.kernelThreadIndex[kernelIndex];
        if (threadIndex >= resource.threadCount) {
            return HCCL_E_INTERNAL;
        }
        const bool localCopyNeeded =
            executionMode == AllGatherExecutionMode::HIERARCHICAL_PHASE_ONE &&
            inputAddr != outputAddr + currentRankOutputOffset;
        CHK_RET(LaunchSlice(resource.threads[threadIndex], resource.ccuKernels[kernelIndex],
            inputAddr, outputAddr, inputToken, outputToken, currentRankOutputOffset,
            sliceSize, localCopyNeeded, fullDataSize, 0, executionMode));
    }
    CHK_RET(PostSyncKernelThreads(resource));
    return HCCL_SUCCESS;
}

HcclResult RunPhysical8Plus4(const AlgResourceCtx &resource, uint64_t inputAddr,
    uint64_t outputAddr, uint64_t inputToken, uint64_t outputToken, uint64_t dataSize,
    uint64_t currentRankOutputOffset)
{
    uint64_t windowOffset = 0;
    while (windowOffset < dataSize) {
        const uint64_t windowSize =
            std::min<uint64_t>(MAX_DATA_SIZE, dataSize - windowOffset);
        CHK_RET(LaunchPhysical8Plus4Phase(resource, inputAddr + windowOffset,
            outputAddr + windowOffset, inputToken, outputToken, currentRankOutputOffset,
            dataSize, windowSize, AllGatherExecutionMode::HIERARCHICAL_PHASE_ONE));
        CHK_RET(LaunchPhysical8Plus4Phase(resource, inputAddr + windowOffset,
            outputAddr + windowOffset, inputToken, outputToken, currentRankOutputOffset,
            dataSize, windowSize, AllGatherExecutionMode::HIERARCHICAL_PHASE_TWO));
        windowOffset += windowSize;
    }
    return HCCL_SUCCESS;
}

HcclResult RunHierarchicalGrid(const AlgResourceCtx &resource, uint64_t inputAddr, uint64_t outputAddr,
    uint64_t inputToken, uint64_t outputToken, uint64_t dataSize, uint64_t dataTypeSize,
    uint64_t currentRankOutputOffset)
{
    const uint64_t balancedPart0Size =
        CalculateBalancedPart0Size(resource, dataSize, dataTypeSize);
    if (!resource.hierarchical2x8 &&
        (balancedPart0Size > MAX_DATA_SIZE ||
         dataSize - balancedPart0Size > MAX_DATA_SIZE)) {
        uint64_t windowOffset = 0;
        while (windowOffset < dataSize) {
            const uint64_t windowSize =
                std::min<uint64_t>(MAX_DATA_SIZE, dataSize - windowOffset);
            CHK_RET(RunHierarchicalWindow(resource, inputAddr, outputAddr, inputToken, outputToken,
                dataSize, dataTypeSize, currentRankOutputOffset, windowOffset, windowSize));
            windowOffset += windowSize;
        }
        return HCCL_SUCCESS;
    }
    return RunHierarchicalWindow(resource, inputAddr, outputAddr, inputToken, outputToken,
        dataSize, dataTypeSize, currentRankOutputOffset, 0, dataSize);
}

HcclResult RunHierarchicalSequence(const AlgResourceCtx &resource, uint64_t inputAddr,
    uint64_t outputAddr, uint64_t inputToken, uint64_t outputToken, uint64_t dataSize,
    uint64_t currentRankOutputOffset)
{
    if (resource.kernelCount != 2 || resource.threadCount == 0) {
        return HCCL_E_INTERNAL;
    }
    uint32_t firstKernelIndex = resource.hierarchical2x8 ? 1U : 0U;
    if (resource.purePhysicalGrid) {
        firstKernelIndex = resource.kernelCount;
        for (uint32_t kernelIndex = 0; kernelIndex < resource.kernelCount; ++kernelIndex) {
            if (resource.kernelTrafficClass[kernelIndex] == KernelTrafficClass::INTER_SERVER) {
                firstKernelIndex = kernelIndex;
                break;
            }
        }
        if (firstKernelIndex == resource.kernelCount) {
            return HCCL_E_INTERNAL;
        }
    }
    const uint32_t secondKernelIndex = 1U - firstKernelIndex;
    const bool localCopyNeeded = inputAddr != outputAddr + currentRankOutputOffset;
    CHK_RET(LaunchSlice(resource.threads[0], resource.ccuKernels[firstKernelIndex], inputAddr, outputAddr,
        inputToken, outputToken, currentRankOutputOffset, dataSize, localCopyNeeded,
        dataSize, 0, AllGatherExecutionMode::HIERARCHICAL_PHASE_ONE));
    CHK_RET(LaunchSlice(resource.threads[0], resource.ccuKernels[secondKernelIndex], inputAddr, outputAddr,
        inputToken, outputToken, currentRankOutputOffset, dataSize, false,
        dataSize, 0, AllGatherExecutionMode::HIERARCHICAL_PHASE_TWO));
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resource)
{
    if (resource.threadCount == 0 || resource.threadCount > AlgResourceCtx::MAX_THREAD_COUNT ||
        resource.kernelCount > AlgResourceCtx::MAX_KERNEL_COUNT || resource.rankSize == 0 ||
        resource.rankSize > MAX_RANK_SIZE ||
        resource.rankId >= resource.rankSize) {
        HCCL_ERROR("CCU AllGather resource context is inconsistent");
        return HCCL_E_INTERNAL;
    }

    if (param.dataType != HCCL_DATA_TYPE_FP32) {
        HCCL_ERROR("Unsupported AllGather data type: %d", static_cast<int>(param.dataType));
        return HCCL_E_NOT_SUPPORT;
    }
    constexpr uint64_t dataTypeSize = sizeof(float);
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    if (param.count > std::numeric_limits<uint64_t>::max() / dataTypeSize) {
        return HCCL_E_PARA;
    }
    const uint64_t dataSize = param.count * dataTypeSize;
    if (dataSize > std::numeric_limits<uint64_t>::max() / resource.rankSize) {
        return HCCL_E_PARA;
    }
    const uint64_t outputSize = dataSize * resource.rankSize;

    if (resource.rankSize == 1) {
        return static_cast<HcclResult>(
            HcommLocalCopyOnThread(resource.threads[0], param.outputPtr, param.inputPtr, dataSize));
    }
    if (resource.kernelCount == 0) {
        HCCL_ERROR("CCU AllGather resource has no kernel");
        return HCCL_E_INTERNAL;
    }
    const uint64_t baseInputAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t baseOutputAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    CcuResult tokenRet = HcommCcuGetMemToken(baseInputAddr, dataSize, &inputToken);
    if (tokenRet != CCU_SUCCESS) {
        HCCL_ERROR("Failed to get input memory token: %d", tokenRet);
        return ConvertCcuToHccl(tokenRet);
    }
    tokenRet = HcommCcuGetMemToken(baseOutputAddr, outputSize, &outputToken);
    if (tokenRet != CCU_SUCCESS) {
        HCCL_ERROR("Failed to get output memory token: %d", tokenRet);
        return ConvertCcuToHccl(tokenRet);
    }

    const uint64_t currentRankOutputOffset = dataSize * resource.rankId;
    const bool runInParallel = ShouldRunDieKernelsInParallel(resource);
    if (runInParallel) {
        for (uint32_t threadIndex = 0; threadIndex < resource.threadCount; ++threadIndex) {
            for (uint32_t otherIndex = threadIndex + 1; otherIndex < resource.threadCount; ++otherIndex) {
                if (resource.threads[threadIndex] == resource.threads[otherIndex]) {
                    HCCL_ERROR("CCU AllGather parallel resource has duplicate thread handles");
                    return HCCL_E_INTERNAL;
                }
            }
        }
    }
    if (resource.hierarchicalGrid) {
        if (resource.physical8Plus4) {
            return RunPhysical8Plus4(resource, baseInputAddr, baseOutputAddr, inputToken, outputToken,
                dataSize, currentRankOutputOffset);
        }
        if (runInParallel) {
            return RunHierarchicalGrid(resource, baseInputAddr, baseOutputAddr, inputToken, outputToken,
                dataSize, dataTypeSize, currentRankOutputOffset);
        }
        return RunHierarchicalSequence(resource, baseInputAddr, baseOutputAddr, inputToken, outputToken,
            dataSize, currentRankOutputOffset);
    }
    uint64_t processedBytes = 0;
    while (processedBytes < dataSize) {
        const uint64_t sliceSize = std::min<uint64_t>(MAX_DATA_SIZE, dataSize - processedBytes);
        const uint64_t inputAddr = baseInputAddr + processedBytes;
        const uint64_t outputAddr = baseOutputAddr + processedBytes;
        const bool localCopyNeeded = inputAddr != outputAddr + currentRankOutputOffset;
        if (runInParallel) {
            CHK_RET(PreSyncKernelThreads(resource));
        }
        for (uint32_t kernelIndex = 0; kernelIndex < resource.kernelCount; ++kernelIndex) {
            const uint32_t threadIndex = resource.kernelThreadIndex[kernelIndex];
            if (threadIndex >= resource.threadCount) {
                return HCCL_E_INTERNAL;
            }
            const ThreadHandle thread = resource.threads[threadIndex];
            CHK_RET(LaunchSlice(thread, resource.ccuKernels[kernelIndex], inputAddr, outputAddr,
                inputToken, outputToken, currentRankOutputOffset, sliceSize, localCopyNeeded,
                dataSize, 0, AllGatherExecutionMode::DIRECT,
                resource.directSmallFastPath, resource.directNhr));
        }
        if (runInParallel) {
            CHK_RET(PostSyncKernelThreads(resource));
        }
        processedBytes += sliceSize;
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
