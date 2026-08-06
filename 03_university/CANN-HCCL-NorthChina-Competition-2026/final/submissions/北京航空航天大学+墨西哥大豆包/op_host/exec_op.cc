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
#include "ccu_kernel.h"
#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {

    constexpr uint64_t SMALL_DATA_THRESHOLD = 4ULL * 1024 * 1024;
    constexpr uint64_t FOUR_BY_ONE_SMALL_DATA_THRESHOLD = 512ULL * 1024;
    constexpr uint64_t EIGHT_PLUS_FOUR_SMALL_DATA_THRESHOLD = 512ULL * 1024;
    constexpr uint64_t EIGHT_PLUS_FOUR_PHASE0 = 0;
    constexpr uint64_t EIGHT_PLUS_FOUR_PHASE1 = 1;
    constexpr uint64_t EIGHT_PLUS_FOUR_ALIGNMENT = sizeof(float);
    constexpr uint64_t EIGHT_PLUS_FOUR_MAX_BATCH_SIZE = static_cast<uint64_t>(MAX_DATA_SIZE);
    constexpr uint64_t NHR_2X8_PHASE_0 = 0;
    constexpr uint64_t NHR_2X8_PHASE_1 = 1;
    constexpr uint64_t NHR_2X8_PART0_NUMERATOR = 8;
    constexpr uint64_t NHR_2X8_PART_DENOMINATOR = 13;
    using Small512FastTaskArgs = std::array<uint64_t, 4>;

    constexpr uint64_t SetBits(uint16_t end)
    {
        return (uint64_t(1) << (end + 1)) - uint64_t(1);
    }

    uint64_t GetMaxLoopIterNum()
    {
        constexpr uint16_t loopNumBitNum = 12;
        return SetBits(loopNumBitNum);
    }

    uint64_t GetParallelParam(uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum)
    {
        constexpr uint16_t repeatBitNum = 7;
        constexpr uint16_t repeatNumShiftBit = 55;
        constexpr uint16_t repeatLoopBitNum = 7;
        constexpr uint16_t repeatLoopShiftBit = 48;
        constexpr uint16_t totalLoopBitNum = 7;
        constexpr uint16_t totalLoopShiftBit = 41;
        return ((repeatNum & SetBits(repeatBitNum)) << repeatNumShiftBit)
               | ((repeatLoopIndex & SetBits(repeatLoopBitNum)) << repeatLoopShiftBit)
               | ((totalLoopNum & SetBits(totalLoopBitNum)) << totalLoopShiftBit);
    }

    std::vector<uint64_t> CalGoSize(uint64_t size, const LoopGroupConfig &config)
    {
        const uint64_t loopSize = config.loopCount * config.memSlice;
        const uint64_t maxSize = loopSize * (GetMaxLoopIterNum() + 1);
        uint64_t completeLoops = size / loopSize;
        uint64_t completeSlices = (size - completeLoops * loopSize) / config.memSlice;
        uint64_t residual = size - completeLoops * loopSize - completeSlices * config.memSlice;

        if (size == maxSize) {
            completeLoops = GetMaxLoopIterNum();
            completeSlices = config.loopCount - 1;
            residual = config.memSlice;
        }

        const uint64_t addressOffset = config.memSlice * config.loopCount * completeLoops;
        uint64_t parallelParam = 0;
        uint64_t tailSize = 0;
        if (completeSlices != 0 && residual == 0) {
            parallelParam = GetParallelParam(completeSlices - 1, 0, 1);
            tailSize = config.memSlice;
        } else if (completeSlices == 0 && residual != 0) {
            parallelParam = GetParallelParam(0, 0, 1);
            tailSize = residual;
        } else if (completeSlices != 0 && residual != 0) {
            parallelParam = GetParallelParam(completeSlices - 1, 1, 2);
            tailSize = residual;
        }

        return {addressOffset, completeLoops, parallelParam, tailSize};
    }

    uint64_t AlignDown8Plus4(uint64_t value)
    {
        return value / EIGHT_PLUS_FOUR_ALIGNMENT * EIGHT_PLUS_FOUR_ALIGNMENT;
    }

    uint64_t Get8Plus4BalancedBatchSize(uint64_t rankDataSize, uint64_t processedBytes)
    {
        const uint64_t remainingBytes = rankDataSize - processedBytes;
        const uint64_t remainingBatchCount =
            (remainingBytes + EIGHT_PLUS_FOUR_MAX_BATCH_SIZE - 1)
            / EIGHT_PLUS_FOUR_MAX_BATCH_SIZE;
        if (remainingBatchCount == 1) {
            return remainingBytes;
        }
        return AlignDown8Plus4(remainingBytes / remainingBatchCount);
    }

    HcclResult ThreadSyncBefore(const std::vector<ThreadHandle> &threads)
    {
        for (uint32_t idx = 1; idx < threads.size(); ++idx) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[0], threads[idx], 0)));
        }
        for (uint32_t idx = 1; idx < threads.size(); ++idx) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[idx], 0, CUSTOM_TIMEOUT)));
        }
        return HCCL_SUCCESS;
    }

    HcclResult ThreadSyncAfter(const std::vector<ThreadHandle> &threads)
    {
        for (uint32_t idx = 1; idx < threads.size(); ++idx) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[0], idx - 1, CUSTOM_TIMEOUT)));
        }
        for (uint32_t idx = 1; idx < threads.size(); ++idx) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[idx], threads[0], idx - 1)));
        }
        return HCCL_SUCCESS;
    }

    HcclResult LaunchSelectedKernels(
        const AlgResourceCtx &resCtx, const AllGatherRoute &route, const std::vector<uint64_t> &taskArgs)
    {
        if (resCtx.ccuKernels.size() != resCtx.ccuKernelMetas.size() || resCtx.threads.empty()) {
            HCCL_ERROR("[LaunchSelectedKernels] invalid kernel metadata or empty thread list");
            return HCCL_E_INTERNAL;
        }

        std::vector<uint32_t> selectedKernelIndices;
        std::vector<bool> threadSeen(resCtx.threads.size(), false);
        for (uint32_t kernelIdx = 0; kernelIdx < resCtx.ccuKernelMetas.size(); ++kernelIdx) {
            const AllGatherKernelMeta &meta = resCtx.ccuKernelMetas[kernelIdx];
            if (meta.sizeClass != route.sizeClass) {
                continue;
            }
            if (meta.threadIndex >= resCtx.threads.size() || threadSeen[meta.threadIndex]) {
                HCCL_ERROR("[LaunchSelectedKernels] invalid or duplicate thread index %u for net layer %u",
                    meta.threadIndex, meta.netLayer);
                return HCCL_E_INTERNAL;
            }
            threadSeen[meta.threadIndex] = true;
            selectedKernelIndices.push_back(kernelIdx);
        }
        if (selectedKernelIndices.size() != resCtx.threads.size()) {
            HCCL_ERROR("[LaunchSelectedKernels] selected %zu kernels for %zu layer threads",
                selectedKernelIndices.size(), resCtx.threads.size());
            return HCCL_E_INTERNAL;
        }

        std::stable_sort(selectedKernelIndices.begin(), selectedKernelIndices.end(),
            [&resCtx](uint32_t left, uint32_t right) {
                const AllGatherKernelMeta &leftMeta = resCtx.ccuKernelMetas[left];
                const AllGatherKernelMeta &rightMeta = resCtx.ccuKernelMetas[right];
                return leftMeta.netLayer > rightMeta.netLayer;
            });

        CHK_RET(ThreadSyncBefore(resCtx.threads));
        for (uint32_t kernelIdx : selectedKernelIndices) {
            const AllGatherKernelMeta &meta = resCtx.ccuKernelMetas[kernelIdx];
            const CcuResult launchRet = HcommCcuKernelLaunch(
                resCtx.threads[meta.threadIndex], resCtx.ccuKernels[kernelIdx], taskArgs.data(), taskArgs.size());
            if (launchRet != CCU_SUCCESS) {
                HCCL_ERROR("[LaunchSelectedKernels] kernel launch failed on net layer %u, ccuRet -> %d",
                    meta.netLayer, launchRet);
                return ConvertCcuToHccl(launchRet);
            }
        }
        CHK_RET(ThreadSyncAfter(resCtx.threads));
        return HCCL_SUCCESS;
    }

    HcclResult LaunchSmallKernel(const AlgResourceCtx &resCtx, const AllGatherRoute &route, uint64_t inputAddr,
        uint64_t outputAddr, uint64_t token, uint64_t rankDataSize, uint32_t myRank)
    {
        const uint64_t currentRankSliceInputOffset = 0;
        const uint64_t currentRankSliceOutputOffset = rankDataSize * myRank;
        const std::vector<uint64_t> taskArgs = {inputAddr, outputAddr, token, currentRankSliceInputOffset,
            currentRankSliceOutputOffset, rankDataSize};
        return LaunchSelectedKernels(resCtx, route, taskArgs);
    }

    std::vector<uint64_t> BuildNhr4x1LargeTaskArgs(uint64_t inputAddr, uint64_t outputAddr, uint64_t token,
        uint64_t rankDataSize, uint32_t myRank, uint64_t processedBytes)
    {
        LoopGroupConfig config{};
        config.msInterleave = CCU_MS_INTERLEAVE;
        config.loopCount = CCU_MS_LOCAL_COPY_LOOP_COUNT;
        config.memSlice = CCU_MS_SIZE * CCU_LOCAL_COPY_MS_PER_LOOP;

        const uint64_t remainingBytes = rankDataSize - processedBytes;
        const uint64_t batchSize = std::min(remainingBytes, 2 * static_cast<uint64_t>(MAX_DATA_SIZE));
        const uint32_t repeatCount = batchSize > MAX_DATA_SIZE ? 2 : 1;
        uint64_t currentRankSliceInputOffset[CCU_ALLGATHER_REPEAT_NUM] = {processedBytes, 0};
        uint64_t currentRankSliceOutputOffset[CCU_ALLGATHER_REPEAT_NUM] = {
            rankDataSize * myRank + processedBytes, 0};
        uint64_t sliceSize[CCU_ALLGATHER_REPEAT_NUM] = {batchSize, 0};
        if (batchSize > MAX_DATA_SIZE) {
            const uint64_t firstSliceSize = (batchSize / 2) & ~uint64_t(sizeof(float) - 1);
            const uint64_t secondSliceSize = batchSize - firstSliceSize;
            sliceSize[0] = firstSliceSize;
            sliceSize[1] = secondSliceSize;
            currentRankSliceInputOffset[1] = currentRankSliceInputOffset[0] + firstSliceSize;
            currentRankSliceOutputOffset[1] = currentRankSliceOutputOffset[0] + firstSliceSize;
        }

        std::vector<uint64_t> taskArgs = {inputAddr, outputAddr, token};
        taskArgs.push_back(repeatCount);
        for (uint32_t repeatIdx = 0; repeatIdx < CCU_ALLGATHER_REPEAT_NUM; ++repeatIdx) {
            const std::vector<uint64_t> goSize = CalGoSize(sliceSize[repeatIdx], config);
            taskArgs.push_back(currentRankSliceInputOffset[repeatIdx]);
            taskArgs.push_back(currentRankSliceOutputOffset[repeatIdx]);
            taskArgs.push_back(sliceSize[repeatIdx]);
            taskArgs.insert(taskArgs.end(), goSize.begin(), goSize.end());
        }
        taskArgs.push_back(rankDataSize);
        return taskArgs;
    }

    std::vector<uint64_t> Build2x8ParallelTaskArgs(uint64_t inputAddr, uint64_t outputAddr,
        uint64_t token, uint64_t rankDataSize, uint32_t myRank, uint64_t processedBytes, uint64_t phase)
    {
        LoopGroupConfig config{};
        config.msInterleave = NHR_2X8_MS_INTERLEAVE;
        config.loopCount = NHR_2X8_LOCAL_COPY_LOOP_COUNT;
        config.memSlice = CCU_MS_SIZE * NHR_2X8_LOCAL_COPY_MS_PER_LOOP;

        const uint64_t maxBatchSize = static_cast<uint64_t>(MAX_DATA_SIZE) * CCU_ALLGATHER_REPEAT_NUM;
        const uint64_t batchSize = std::min(maxBatchSize, rankDataSize - processedBytes);
        const uint64_t alignment = sizeof(float);
        const uint64_t part0Size =
            (batchSize * NHR_2X8_PART0_NUMERATOR / NHR_2X8_PART_DENOMINATOR / alignment) * alignment;
        const uint64_t part1Size = batchSize - part0Size;
        const uint64_t partOffsets[CCU_ALLGATHER_REPEAT_NUM] = {
            processedBytes, processedBytes + part0Size};
        const uint64_t partSizes[CCU_ALLGATHER_REPEAT_NUM] = {part0Size, part1Size};
        const uint32_t partCount = static_cast<uint32_t>((part0Size != 0) + (part1Size != 0));

        std::vector<uint64_t> taskArgs = {inputAddr, outputAddr, token};
        taskArgs.push_back(partCount);
        for (uint32_t partIdx = 0; partIdx < CCU_ALLGATHER_REPEAT_NUM; ++partIdx) {
            const uint64_t currentRankSliceInputOffset = partOffsets[partIdx];
            const uint64_t currentRankSliceOutputOffset =
                rankDataSize * myRank + currentRankSliceInputOffset;
            const uint64_t sliceSize = partSizes[partIdx];
            const std::vector<uint64_t> goSize =
                sliceSize == 0 ? std::vector<uint64_t>(4, 0) : CalGoSize(sliceSize, config);
            taskArgs.push_back(currentRankSliceInputOffset);
            taskArgs.push_back(currentRankSliceOutputOffset);
            taskArgs.push_back(sliceSize);
            taskArgs.insert(taskArgs.end(), goSize.begin(), goSize.end());
        }
        taskArgs.push_back(phase);
        taskArgs.push_back(rankDataSize);
        return taskArgs;
    }

    std::vector<uint64_t> Build8Plus4LargeTaskArgs(uint64_t inputAddr, uint64_t outputAddr, uint64_t token,
        uint64_t rankDataSize, uint64_t processedBytes)
    {
        LoopGroupConfig config{};
        config.msInterleave = EIGHT_PLUS_FOUR_CCU_MS_INTERLEAVE;
        config.loopCount = EIGHT_PLUS_FOUR_CCU_LOOP_COUNT;
        config.memSlice = CCU_MS_SIZE * EIGHT_PLUS_FOUR_CCU_LOCAL_COPY_MS_PER_LOOP;

        const uint64_t batchSize = Get8Plus4BalancedBatchSize(rankDataSize, processedBytes);
        const uint64_t half0Size = AlignDown8Plus4(batchSize / 2);
        const uint64_t half1Size = batchSize - half0Size;
        const uint64_t lEnd = AlignDown8Plus4(batchSize * 4 / 7);
        const uint64_t sEnd = AlignDown8Plus4(batchSize * 6 / 7);
        const uint64_t lSize = lEnd;
        const uint64_t sSize = sEnd - lEnd;
        const uint64_t qSize = batchSize - sEnd;
        std::vector<uint64_t> half0GoSize(4, 0);
        std::vector<uint64_t> half1GoSize(4, 0);
        if (half0Size != 0) {
            half0GoSize = CalGoSize(half0Size, config);
        }
        if (half1Size != 0) {
            half1GoSize = CalGoSize(half1Size, config);
        }

        std::vector<uint64_t> taskArgs = {inputAddr, outputAddr, token, rankDataSize, processedBytes,
            half0Size, half1Size, lSize, sSize, qSize};
        taskArgs.insert(taskArgs.end(), half0GoSize.begin(), half0GoSize.end());
        taskArgs.insert(taskArgs.end(), half1GoSize.begin(), half1GoSize.end());
        return taskArgs;
    }

    HcclResult Launch2x8ParallelKernelBatch(const AlgResourceCtx &resCtx, const AllGatherRoute &route,
        uint64_t inputAddr, uint64_t outputAddr, uint64_t token, uint64_t rankDataSize, uint32_t myRank,
        uint64_t processedBytes)
    {
        const std::vector<uint64_t> phase0Args = Build2x8ParallelTaskArgs(
            inputAddr, outputAddr, token, rankDataSize, myRank, processedBytes, NHR_2X8_PHASE_0);
        CHK_RET(LaunchSelectedKernels(resCtx, route, phase0Args));
        const std::vector<uint64_t> phase1Args = Build2x8ParallelTaskArgs(
            inputAddr, outputAddr, token, rankDataSize, myRank, processedBytes, NHR_2X8_PHASE_1);
        return LaunchSelectedKernels(resCtx, route, phase1Args);
    }

    HcclResult LaunchNhr4x1KernelBatch(const AlgResourceCtx &resCtx, const AllGatherRoute &route,
        uint64_t inputAddr, uint64_t outputAddr, uint64_t token, uint64_t rankDataSize, uint32_t myRank,
        uint64_t processedBytes)
    {
        std::vector<uint64_t> taskArgs =
            BuildNhr4x1LargeTaskArgs(inputAddr, outputAddr, token, rankDataSize, myRank, processedBytes);
        return LaunchSelectedKernels(resCtx, route, taskArgs);
    }

    HcclResult Launch8Plus4KernelBatch(const AlgResourceCtx &resCtx, const AllGatherRoute &route,
        uint64_t inputAddr, uint64_t outputAddr, uint64_t token, uint64_t rankDataSize, uint64_t processedBytes)
    {
        std::vector<uint64_t> taskArgs =
            Build8Plus4LargeTaskArgs(inputAddr, outputAddr, token, rankDataSize, processedBytes);
        taskArgs.push_back(EIGHT_PLUS_FOUR_PHASE0);
        CHK_RET(LaunchSelectedKernels(resCtx, route, taskArgs));
        taskArgs.back() = EIGHT_PLUS_FOUR_PHASE1;
        return LaunchSelectedKernels(resCtx, route, taskArgs);
    }

} // namespace

HcclResult ExecSmall512Fast(
    void *inputPtr, void *outputPtr, void *resCtx, uint64_t ctxSize)
{
    CHK_PTR_NULL(inputPtr);
    CHK_PTR_NULL(outputPtr);
    CHK_PTR_NULL(resCtx);
    CHK_PRT_RET(ctxSize != sizeof(Small512FastResourceCtx),
        HCCL_ERROR("[ExecSmall512Fast] invalid context size %llu",
            static_cast<unsigned long long>(ctxSize)),
        HCCL_E_INTERNAL);

    auto *fastCtx = static_cast<Small512FastResourceCtx *>(resCtx);
    CHK_PRT_RET(fastCtx->magic != SMALL_512_FAST_CTX_MAGIC
            || (fastCtx->rankSize != 4 && fastCtx->rankSize != 12),
        HCCL_ERROR("[ExecSmall512Fast] invalid fast context"),
        HCCL_E_INTERNAL);

    constexpr uint64_t small512DataSize = 512ULL * 1024;
    const uint64_t inputAddr = reinterpret_cast<uint64_t>(inputPtr);
    const uint64_t outputAddr = reinterpret_cast<uint64_t>(outputPtr);
    const bool refreshToken =
        fastCtx->cacheValid == 0 || fastCtx->cachedInputAddr != inputAddr;
    const bool refreshResources =
        refreshToken || fastCtx->cachedOutputAddr != outputAddr;

    uint64_t token = fastCtx->cachedToken;
    if (refreshToken) {
        CHK_RET_CCU(HcommCcuGetMemToken(inputAddr, small512DataSize, &token));
    }

    const Small512FastTaskArgs taskArgs = {
        inputAddr, outputAddr, token,
        static_cast<uint64_t>(refreshResources)};
    const CcuResult launchRet = HcommCcuKernelLaunch(
        fastCtx->thread, fastCtx->kernels[0], taskArgs.data(), taskArgs.size());
    if (launchRet != CCU_SUCCESS) {
        HCCL_ERROR("[ExecSmall512Fast] kernel launch failed, ccuRet -> %d", launchRet);
        return ConvertCcuToHccl(launchRet);
    }

    if (refreshResources) {
        fastCtx->cachedInputAddr = inputAddr;
        fastCtx->cachedOutputAddr = outputAddr;
        fastCtx->cachedToken = token;
        fastCtx->cacheValid = 1;
    }
    return HCCL_SUCCESS;
}

HcclResult SelectAllGatherRoute(uint64_t dataSize, uint32_t rankSize, AllGatherRoute &route)
{
    route.sizeClass = dataSize <= SMALL_DATA_THRESHOLD ? AllGatherSizeClass::SMALL : AllGatherSizeClass::LARGE;

    switch (rankSize) {
        case 16:
            route.topology = AllGatherTopology::TWO_SERVER_EIGHT_NPU;
            break;
        case 4:
            route.topology = AllGatherTopology::FOUR_SERVER_ONE_NPU;
            route.sizeClass = dataSize <= FOUR_BY_ONE_SMALL_DATA_THRESHOLD
                ? AllGatherSizeClass::SMALL
                : AllGatherSizeClass::LARGE;
            break;
        case 12:
            route.topology = AllGatherTopology::EIGHT_PLUS_FOUR;
            route.sizeClass = dataSize <= EIGHT_PLUS_FOUR_SMALL_DATA_THRESHOLD
                ? AllGatherSizeClass::SMALL
                : AllGatherSizeClass::LARGE;
            break;
        default:
            HCCL_ERROR("[SelectAllGatherRoute] unsupported rank size %u", rankSize);
            return HCCL_E_NOT_SUPPORT;
    }

    HCCL_INFO("[SelectAllGatherRoute] dataSize=%llu rankSize=%u topology=%u sizeClass=%u",
        static_cast<unsigned long long>(dataSize), rankSize, static_cast<uint32_t>(route.topology),
        static_cast<uint32_t>(route.sizeClass));
    return HCCL_SUCCESS;
}

HcclResult ExecOp(const OpParam &param)
{
    char *ctx = static_cast<char *>(param.resCtx);
    CHK_PTR_NULL(ctx);
    std::vector<char> seq(ctx, ctx + param.ctxSize);
    AlgResourceCtx resCtx;
    resCtx.DeSerialize(seq);

    const auto sizeIt = SIZE_TABLE.find(param.dataType);
    if (sizeIt == SIZE_TABLE.end()) {
        HCCL_ERROR("[ExecOp] unsupported data type %d", static_cast<int32_t>(param.dataType));
        return HCCL_E_NOT_SUPPORT;
    }

    const uint64_t dataTypeSize = sizeIt->second;
    const uint64_t rankDataSize = param.count * dataTypeSize;
    if (rankDataSize == 0) {
        return HCCL_SUCCESS;
    }

    AllGatherRoute route;
    CHK_RET(SelectAllGatherRoute(rankDataSize, param.rankSize, route));

    uint64_t token = 0;
    const uint64_t baseInputAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t baseOutputAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    CHK_RET_CCU(HcommCcuGetMemToken(baseInputAddr, rankDataSize, &token));

    if (route.sizeClass == AllGatherSizeClass::SMALL) {
        return LaunchSmallKernel(
            resCtx, route, baseInputAddr, baseOutputAddr, token, rankDataSize, param.myRank);
    }

    const uint64_t defaultMaxBatchSize =
        static_cast<uint64_t>(MAX_DATA_SIZE) * CCU_ALLGATHER_REPEAT_NUM;
    uint64_t processedBytes = 0;
    while (processedBytes < rankDataSize) {
        uint64_t batchSize = std::min(defaultMaxBatchSize, rankDataSize - processedBytes);
        if (route.topology == AllGatherTopology::EIGHT_PLUS_FOUR) {
            batchSize = Get8Plus4BalancedBatchSize(rankDataSize, processedBytes);
        }

        if (route.topology == AllGatherTopology::TWO_SERVER_EIGHT_NPU) {
            CHK_RET(Launch2x8ParallelKernelBatch(
                resCtx, route, baseInputAddr, baseOutputAddr, token, rankDataSize, param.myRank, processedBytes));
        } else if (route.topology == AllGatherTopology::FOUR_SERVER_ONE_NPU) {
            CHK_RET(LaunchNhr4x1KernelBatch(
                resCtx, route, baseInputAddr, baseOutputAddr, token, rankDataSize, param.myRank, processedBytes));
        } else if (route.topology == AllGatherTopology::EIGHT_PLUS_FOUR) {
            CHK_RET(Launch8Plus4KernelBatch(
                resCtx, route, baseInputAddr, baseOutputAddr, token, rankDataSize, processedBytes));
        }
        processedBytes += batchSize;
    }
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
