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
#include <cmath>
#include <cstdint>
#include <limits>

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>

#include "custom.h"
#include "ccu_kernel.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {

    constexpr uint64_t SCRATCH_FANIN_THRESHOLD_BYTES = 1UL * 1024 * 1024;
    constexpr uint64_t PARALLEL_2D_OMNIPIPE_MIN_BYTES = 1UL * 1024 * 1024;
    constexpr uint64_t OMNIPIPE_SLICE_ALIGNMENT = 128;
    constexpr uint32_t OMNIPIPE_MAX_STEP_COUNT = 6;
    constexpr uint32_t OMNIPIPE_MAX_NHR_RANKS = 4;
    constexpr double OMNIPIPE_MESH_BANDWIDTH = 25;
    constexpr double OMNIPIPE_CLOS_BANDWIDTH = 200;
    constexpr uint64_t MS_PIPELINE_MIN_BYTES = 16UL * 1024;
    constexpr uint64_t MS_PIPELINE_MAX_BYTES = 128UL * 1024;
    constexpr uint32_t MAX_MS_PIPELINE_INPUTS = 8;
    constexpr uint64_t MS_PIPELINE_TILE_BYTES = 4096;
    constexpr uint64_t MS_PIPELINE_PARALLEL_TILES = 16;
    constexpr uint64_t MAX_MS_PIPELINE_EXPANDED_REMOTE_READS = 512UL * 1024;
    constexpr std::array<uint64_t, 4> EMPTY_MS_PIPELINE_GROUP_OP_SIZE = {0, 0, 0, 0};

    struct OmnipipePlan {
        std::array<uint64_t, OMNIPIPE_MAX_STEP_COUNT> meshSizes{};
        std::array<uint64_t, OMNIPIPE_MAX_STEP_COUNT> closSizes{};
        std::array<uint64_t, OMNIPIPE_MAX_STEP_COUNT> meshOffsets{};
        std::array<uint64_t, OMNIPIPE_MAX_STEP_COUNT> closOffsets{};
        uint32_t stepCount = 0;
        uint64_t scratchBytes = 0;
    };

    constexpr uint64_t SetLowBits(uint16_t end)
    {
        return (uint64_t{1} << (end + 1)) - uint64_t{1};
    }

    constexpr uint64_t GetParallelParam(uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum)
    {
        constexpr uint16_t REPEAT_BIT_END = 7;
        constexpr uint16_t REPEAT_NUM_SHIFT = 55;
        constexpr uint16_t REPEAT_LOOP_BIT_END = 7;
        constexpr uint16_t REPEAT_LOOP_SHIFT = 48;
        constexpr uint16_t TOTAL_LOOP_BIT_END = 7;
        constexpr uint16_t TOTAL_LOOP_SHIFT = 41;
        return ((repeatNum & SetLowBits(REPEAT_BIT_END)) << REPEAT_NUM_SHIFT)
               | ((repeatLoopIndex & SetLowBits(REPEAT_LOOP_BIT_END)) << REPEAT_LOOP_SHIFT)
               | ((totalLoopNum & SetLowBits(TOTAL_LOOP_BIT_END)) << TOTAL_LOOP_SHIFT);
    }

    constexpr uint64_t GetLoopParam(uint64_t loopContextId, uint64_t addressOffset, uint64_t loopIterNum)
    {
        constexpr uint16_t CONTEXT_ID_BIT_END = 8;
        constexpr uint16_t CONTEXT_ID_SHIFT = 45;
        constexpr uint16_t ADDRESS_OFFSET_BIT_END = 32;
        constexpr uint16_t ADDRESS_OFFSET_SHIFT = 13;
        constexpr uint16_t LOOP_NUM_BIT_END = 13;
        return ((loopContextId & SetLowBits(CONTEXT_ID_BIT_END)) << CONTEXT_ID_SHIFT)
               | ((addressOffset & SetLowBits(ADDRESS_OFFSET_BIT_END)) << ADDRESS_OFFSET_SHIFT)
               | (loopIterNum & SetLowBits(LOOP_NUM_BIT_END));
    }

    std::array<uint64_t, 4> CalculateMsPipelineGroupOpSize(uint64_t size)
    {
        const uint64_t loopGroupSize = MS_PIPELINE_PARALLEL_TILES * MS_PIPELINE_TILE_BYTES;
        const uint64_t loopIterNum = size / loopGroupSize;
        const uint64_t addressOffset = loopIterNum * loopGroupSize;
        const uint64_t remainingBytes = size - addressOffset;
        const uint64_t parallelCount = remainingBytes / MS_PIPELINE_TILE_BYTES;
        const uint64_t residual = remainingBytes - parallelCount * MS_PIPELINE_TILE_BYTES;
        const uint64_t loopParam = loopIterNum == 0 ? 0 : GetLoopParam(0, loopGroupSize, loopIterNum);
        uint64_t parallelParam = 0;
        uint64_t tailSize = 0;
        if (parallelCount != 0 && residual == 0) {
            parallelParam = GetParallelParam(parallelCount - 1, 0, 1);
            tailSize = MS_PIPELINE_TILE_BYTES;
        } else if (parallelCount == 0 && residual != 0) {
            parallelParam = GetParallelParam(0, 0, 1);
            tailSize = residual;
        } else if (parallelCount != 0) {
            parallelParam = GetParallelParam(parallelCount - 1, 1, 2);
            tailSize = residual;
        }
        return {addressOffset, loopParam, parallelParam, tailSize};
    }

    bool CanUseMsPipeline(const std::vector<uint32_t> &channelCounts)
    {
        if (channelCounts.empty()) {
            return false;
        }
        for (uint32_t channelCount : channelCounts) {
            if (channelCount == 0 || channelCount > MAX_MS_PIPELINE_INPUTS) {
                return false;
            }
        }
        return true;
    }

    bool CheckedMultiply(uint64_t left, uint64_t right, uint64_t &result)
    {
        if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
            return false;
        }
        result = left * right;
        return true;
    }

    bool IsMsPipelineGraphWithinBudget(uint64_t size, uint32_t rankSize, uint64_t totalChannelCount)
    {
        if (rankSize <= 1 || totalChannelCount == 0) {
            return false;
        }

        const uint64_t tileCount
            = size / MS_PIPELINE_TILE_BYTES + static_cast<uint64_t>(size % MS_PIPELINE_TILE_BYTES != 0);
        uint64_t remoteReadsPerRank = 0;
        uint64_t expandedRemoteReads = 0;
        return tileCount != 0 && CheckedMultiply(tileCount, totalChannelCount, remoteReadsPerRank)
               && CheckedMultiply(remoteReadsPerRank, rankSize, expandedRemoteReads)
               && expandedRemoteReads <= MAX_MS_PIPELINE_EXPANDED_REMOTE_READS;
    }

    bool IsMsPipelineSizeProfitable(uint64_t size)
    {
        return size >= MS_PIPELINE_MIN_BYTES && size <= MS_PIPELINE_MAX_BYTES;
    }

    uint64_t AlignDown(uint64_t value, uint64_t alignment)
    {
        return value / alignment * alignment;
    }

    bool CalculateOmnipipePlan(uint64_t sliceBytes, uint64_t dataTypeSize, uint32_t localGroupSize,
        uint32_t serverGroupCount, OmnipipePlan &plan)
    {
        if (sliceBytes == 0 || dataTypeSize == 0 || localGroupSize <= 1 || serverGroupCount <= 1
            || sliceBytes % dataTypeSize != 0 || OMNIPIPE_SLICE_ALIGNMENT % dataTypeSize != 0) {
            return false;
        }

        const double bandwidthRatio
            = OMNIPIPE_CLOS_BANDWIDTH / static_cast<double>(serverGroupCount - 1) / OMNIPIPE_MESH_BANDWIDTH;
        const double omnipipeRatio = static_cast<double>(localGroupSize - 1) / bandwidthRatio;
        double scaleDenominator = 0.0;
        for (uint32_t power = 0; power < OMNIPIPE_MAX_STEP_COUNT - 2; ++power) {
            scaleDenominator += std::pow(omnipipeRatio, power);
        }
        if (scaleDenominator <= 0.0) {
            return false;
        }
        double scale = bandwidthRatio / scaleDenominator;
        uint32_t stepCount = OMNIPIPE_MAX_STEP_COUNT;
        if (static_cast<double>(localGroupSize) - bandwidthRatio > 0.0) {
            if (std::abs(omnipipeRatio - 1.0) < 1e-12) {
                stepCount = static_cast<uint32_t>(bandwidthRatio) + 2;
            } else {
                const double calculatedSteps = std::ceil(std::log(static_cast<double>(localGroupSize) - bandwidthRatio)
                                                         / std::log(omnipipeRatio))
                                               + 2.0;
                if (calculatedSteps < 3.0 || calculatedSteps > std::numeric_limits<uint32_t>::max()) {
                    return false;
                }
                stepCount = static_cast<uint32_t>(calculatedSteps);
            }
            if (stepCount <= OMNIPIPE_MAX_STEP_COUNT) {
                scale = 1.0;
            } else {
                stepCount = OMNIPIPE_MAX_STEP_COUNT;
            }
        }
        if (stepCount < 3 || stepCount > OMNIPIPE_MAX_STEP_COUNT) {
            return false;
        }

        if (scale > 1.0) {
            const double numerator = static_cast<double>(sliceBytes) * scale
                                     * std::pow(static_cast<double>(localGroupSize - 1), stepCount - 2);
            const double denominator = (static_cast<double>(serverGroupCount - 1) * bandwidthRatio + localGroupSize - 1)
                                       * std::pow(bandwidthRatio, stepCount - 2);
            if (denominator <= 0.0) {
                return false;
            }
            plan.meshSizes[0] = AlignDown(static_cast<uint64_t>(numerator / denominator), OMNIPIPE_SLICE_ALIGNMENT);
        } else {
            const double numerator = (static_cast<double>(localGroupSize) - bandwidthRatio) * sliceBytes;
            const double denominator = static_cast<double>(serverGroupCount - 1) * bandwidthRatio + localGroupSize - 1;
            if (numerator <= 0.0 || denominator <= 0.0) {
                return false;
            }
            plan.meshSizes[0] = AlignDown(static_cast<uint64_t>(numerator / denominator), OMNIPIPE_SLICE_ALIGNMENT);
        }

        if (stepCount == 3) {
            plan.closSizes[0] = sliceBytes - plan.meshSizes[0];
        } else {
            plan.closSizes[0] = AlignDown(static_cast<uint64_t>(static_cast<double>(plan.meshSizes[0]) * bandwidthRatio
                                                                * (serverGroupCount - 1) / (localGroupSize - 1)),
                OMNIPIPE_SLICE_ALIGNMENT);
        }

        uint64_t sumMeshSizes = 0;
        uint64_t sumCombinedSizes = plan.meshSizes[0] + plan.closSizes[0];
        for (uint32_t index = 1; index < stepCount - 2; ++index) {
            if (index == stepCount - 3) {
                if (sumCombinedSizes > sliceBytes) {
                    return false;
                }
                plan.closSizes[index] = sliceBytes - sumCombinedSizes;
                plan.meshSizes[index] = static_cast<uint64_t>(
                    static_cast<double>(plan.closSizes[index]) * (localGroupSize - 1) / bandwidthRatio);
                if (index == 1 && plan.meshSizes[index] > sumCombinedSizes) {
                    plan.meshSizes[index] = sumCombinedSizes;
                } else if (plan.meshSizes[index] > plan.closSizes[index - 1]) {
                    plan.meshSizes[index] = plan.closSizes[index - 1];
                }
                plan.meshSizes[index] = AlignDown(plan.meshSizes[index], OMNIPIPE_SLICE_ALIGNMENT);
                sumMeshSizes += plan.meshSizes[index];
                sumCombinedSizes += plan.closSizes[index];
                continue;
            }
            if (index == 1) {
                plan.meshSizes[index] = sumCombinedSizes;
            } else {
                plan.meshSizes[index] = plan.closSizes[index - 1];
            }
            plan.closSizes[index] = AlignDown(static_cast<uint64_t>(static_cast<double>(plan.meshSizes[index])
                                                                    * bandwidthRatio / (localGroupSize - 1)),
                OMNIPIPE_SLICE_ALIGNMENT);
            sumMeshSizes += plan.meshSizes[index];
            sumCombinedSizes += plan.closSizes[index];
        }

        if (sumMeshSizes > sliceBytes) {
            return false;
        }
        plan.meshSizes[stepCount - 2] = AlignDown(
            std::min<uint64_t>(static_cast<uint64_t>(static_cast<double>(sliceBytes) / (1.0 + bandwidthRatio)),
                sliceBytes - sumMeshSizes),
            OMNIPIPE_SLICE_ALIGNMENT);
        if (sumMeshSizes + plan.meshSizes[stepCount - 2] > sliceBytes) {
            return false;
        }
        plan.meshSizes[stepCount - 1] = sliceBytes - sumMeshSizes - plan.meshSizes[stepCount - 2];
        plan.closSizes[stepCount - 2] = sliceBytes - plan.meshSizes[stepCount - 2];
        plan.closSizes[stepCount - 1] = sliceBytes - plan.closSizes[stepCount - 2];

        plan.meshOffsets[0] = 0;
        plan.closOffsets[0] = plan.meshSizes[0];
        plan.meshOffsets[1] = 0;
        for (uint32_t index = 1; index < stepCount - 2; ++index) {
            plan.closOffsets[index] = plan.closOffsets[index - 1] + plan.closSizes[index - 1];
        }
        for (uint32_t index = 2; index < stepCount - 2; ++index) {
            plan.meshOffsets[index] = plan.meshOffsets[index - 1] + plan.meshSizes[index - 1];
        }
        plan.closOffsets[stepCount - 2] = 0;
        plan.closOffsets[stepCount - 1] = plan.closSizes[stepCount - 2];
        plan.meshOffsets[stepCount - 1] = plan.meshOffsets[stepCount - 3] + plan.meshSizes[stepCount - 3];
        plan.meshOffsets[stepCount - 2] = plan.meshOffsets[stepCount - 1] + plan.meshSizes[stepCount - 1];

        const uint64_t scratchSlotCount = static_cast<uint64_t>(localGroupSize - 1);
        if (!CheckedMultiply(scratchSlotCount, sliceBytes, plan.scratchBytes)) {
            return false;
        }
        plan.stepCount = stepCount;
        for (uint32_t index = 0; index < stepCount; ++index) {
            if (plan.meshSizes[index] == 0 || plan.closSizes[index] == 0
                || plan.meshOffsets[index] > sliceBytes - plan.meshSizes[index]
                || plan.closOffsets[index] > sliceBytes - plan.closSizes[index]
                || plan.meshSizes[index] % dataTypeSize != 0 || plan.closSizes[index] % dataTypeSize != 0
                || plan.meshOffsets[index] % dataTypeSize != 0 || plan.closOffsets[index] % dataTypeSize != 0) {
                return false;
            }
        }
        return true;
    }

    bool RangesOverlap(uint64_t firstAddress, uint64_t firstSize, uint64_t secondAddress, uint64_t secondSize)
    {
        if (firstSize == 0 || secondSize == 0) {
            return false;
        }
        if (firstAddress > std::numeric_limits<uint64_t>::max() - firstSize
            || secondAddress > std::numeric_limits<uint64_t>::max() - secondSize) {
            return true;
        }
        return firstAddress < secondAddress + secondSize && secondAddress < firstAddress + firstSize;
    }

    HcclResult SyncMainToSub(const AlgResourceCtx &resourceCtx)
    {
        if (resourceCtx.threads.size() < 2) {
            return HCCL_SUCCESS;
        }
        CHK_RET(HcommThreadNotifyRecordOnThread(resourceCtx.threads[0], resourceCtx.threads[1], 0));
        CHK_RET(HcommThreadNotifyWaitOnThread(resourceCtx.threads[1], 0, CUSTOM_TIMEOUT));
        return HCCL_SUCCESS;
    }

    HcclResult SyncSubToMain(const AlgResourceCtx &resourceCtx)
    {
        if (resourceCtx.threads.size() < 2) {
            return HCCL_SUCCESS;
        }
        CHK_RET(HcommThreadNotifyRecordOnThread(resourceCtx.threads[1], resourceCtx.threads[0], 0));
        CHK_RET(HcommThreadNotifyWaitOnThread(resourceCtx.threads[0], 0, CUSTOM_TIMEOUT));
        return HCCL_SUCCESS;
    }

    HcclResult SyncControlToAxes(const AlgResourceCtx &resourceCtx)
    {
        CHK_PRT_RET(resourceCtx.threads.size() != 3,
            HCCL_ERROR("[SyncControlToAxes] expected three CCU threads, got[%zu]", resourceCtx.threads.size()),
            HCCL_E_INTERNAL);
        for (uint32_t axis = 1; axis < resourceCtx.threads.size(); ++axis) {
            CHK_RET(HcommThreadNotifyRecordOnThread(resourceCtx.threads[0], resourceCtx.threads[axis], 0));
            CHK_RET(HcommThreadNotifyWaitOnThread(resourceCtx.threads[axis], 0, CUSTOM_TIMEOUT));
        }
        return HCCL_SUCCESS;
    }

    HcclResult SyncAxesToControl(const AlgResourceCtx &resourceCtx)
    {
        CHK_PRT_RET(resourceCtx.threads.size() != 3,
            HCCL_ERROR("[SyncAxesToControl] expected three CCU threads, got[%zu]", resourceCtx.threads.size()),
            HCCL_E_INTERNAL);
        for (uint32_t axis = 1; axis < resourceCtx.threads.size(); ++axis) {
            const uint32_t controlNotifyIndex = axis - 1;
            CHK_RET(HcommThreadNotifyRecordOnThread(
                resourceCtx.threads[axis], resourceCtx.threads[0], controlNotifyIndex));
            CHK_RET(HcommThreadNotifyWaitOnThread(resourceCtx.threads[0], controlNotifyIndex, CUSTOM_TIMEOUT));
        }
        return HCCL_SUCCESS;
    }

    HcclResult LaunchKernel(ThreadHandle thread, CcuKernelHandle kernel, uint64_t inputAddress,
        uint64_t destinationAddress, uint64_t inputToken, uint64_t destinationToken, uint64_t remoteSliceOffset,
        uint64_t chunkBytes, uint64_t executionMode, uint64_t scratchBaseAddress, uint64_t scratchToken,
        uint64_t scratchStride, const std::array<uint64_t, 4> &msPipelineGroupOpSize)
    {
        const std::array<uint64_t, 14> taskArgs = {
            inputAddress,
            destinationAddress,
            inputToken,
            destinationToken,
            remoteSliceOffset,
            chunkBytes,
            executionMode,
            scratchBaseAddress,
            scratchToken,
            scratchStride,
            msPipelineGroupOpSize[0],
            msPipelineGroupOpSize[1],
            msPipelineGroupOpSize[2],
            msPipelineGroupOpSize[3],
        };
        CcuResult result
            = HcommCcuKernelLaunch(thread, kernel, taskArgs.data(), static_cast<uint32_t>(taskArgs.size()));
        if (result != CCU_SUCCESS) {
            HCCL_ERROR("[LaunchKernel] CCU kernel launch failed[%d]", result);
            return ConvertCcuToHccl(result);
        }
        return HCCL_SUCCESS;
    }

    HcclResult LaunchNhrKernel(ThreadHandle thread, CcuKernelHandle kernel, uint64_t inputAddress,
        uint64_t inputToken, const std::vector<uint64_t> &sliceOffsets, const std::vector<uint64_t> &sliceSizes)
    {
        CHK_PRT_RET(sliceOffsets.size() <= 1 || sliceOffsets.size() > OMNIPIPE_MAX_NHR_RANKS
                        || sliceSizes.size() != sliceOffsets.size(),
            HCCL_ERROR("[LaunchNhrKernel] invalid NHR slice vectors: offsets[%zu], sizes[%zu]", sliceOffsets.size(),
                sliceSizes.size()),
            HCCL_E_PARA);

        std::array<uint64_t, 14> taskArgs{};
        taskArgs[0] = inputAddress;
        taskArgs[1] = inputAddress;
        taskArgs[2] = inputToken;
        taskArgs[3] = 0;
        for (uint32_t index = 0; index < sliceOffsets.size(); ++index) {
            taskArgs[4 + index] = sliceOffsets[index];
            taskArgs[4 + sliceOffsets.size() + index] = sliceSizes[index];
        }
        CcuResult result
            = HcommCcuKernelLaunch(thread, kernel, taskArgs.data(), static_cast<uint32_t>(taskArgs.size()));
        if (result != CCU_SUCCESS) {
            HCCL_ERROR("[LaunchNhrKernel] CCU NHR kernel launch failed[%d]", result);
            return ConvertCcuToHccl(result);
        }
        return HCCL_SUCCESS;
    }

    HcclResult ExecParallel2dOmnipipe(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t sliceBytes,
        uint64_t dataTypeSize, uint64_t inputAddress, uint64_t outputAddress, uint64_t inputToken)
    {
        const uint64_t localGroupSize = resCtx.localRanks.size();
        const uint64_t serverGroupCount = resCtx.columnRanks.size();
        CHK_PRT_RET((param.rankSize != 12 && param.rankSize != 16) || localGroupSize <= 1 || serverGroupCount <= 1
                        || localGroupSize * serverGroupCount != param.rankSize
                        || serverGroupCount > OMNIPIPE_MAX_NHR_RANKS
                        || resCtx.globalRanks.size() != param.rankSize
                        || resCtx.localRankIndex >= localGroupSize
                        || resCtx.localRanks[resCtx.localRankIndex] != param.myRank
                        || resCtx.columnRankIndex >= serverGroupCount
                        || resCtx.columnRanks[resCtx.columnRankIndex] != param.myRank,
            HCCL_ERROR("[ExecParallel2dOmnipipe] invalid 2D rank metadata"), HCCL_E_INTERNAL);
        CHK_PRT_RET(resCtx.threads.size() != 3 || resCtx.ccuKernels.size() != 2 || resCtx.channelCounts.size() != 2
                        || resCtx.channelCounts[0] != localGroupSize - 1
                        || resCtx.channelCounts[1] != serverGroupCount - 1,
            HCCL_ERROR("[ExecParallel2dOmnipipe] invalid 2D resource layout"), HCCL_E_INTERNAL);
        CHK_PRT_RET(sliceBytes < PARALLEL_2D_OMNIPIPE_MIN_BYTES || sliceBytes > MAX_DATA_SIZE
                        || sliceBytes % OMNIPIPE_SLICE_ALIGNMENT != 0,
            HCCL_ERROR("[ExecParallel2dOmnipipe] slice size[%lu] is outside the OmniPipe envelope", sliceBytes),
            HCCL_E_INTERNAL);

        OmnipipePlan plan;
        CHK_PRT_RET(!CalculateOmnipipePlan(sliceBytes, dataTypeSize, static_cast<uint32_t>(localGroupSize),
                        static_cast<uint32_t>(serverGroupCount), plan),
            HCCL_ERROR("[ExecParallel2dOmnipipe] failed to calculate OmniPipe slices"), HCCL_E_INTERNAL);
        CHK_PRT_RET(resCtx.localBuffer.addr == nullptr || resCtx.localBuffer.size < plan.scratchBytes,
            HCCL_ERROR("[ExecParallel2dOmnipipe] HCCL buffer[%lu] is smaller than required scratch[%lu]",
                resCtx.localBuffer.size, plan.scratchBytes),
            HCCL_E_INTERNAL);

        const uint64_t scratchAddress = reinterpret_cast<uint64_t>(resCtx.localBuffer.addr);
        CHK_PRT_RET(scratchAddress > std::numeric_limits<uint64_t>::max() - plan.scratchBytes,
            HCCL_ERROR("[ExecParallel2dOmnipipe] scratch address overflow"), HCCL_E_PARA);
        uint64_t scratchToken = 0;
        CHK_RET_CCU(HcommCcuGetMemToken(scratchAddress, plan.scratchBytes, &scratchToken));

        const uint64_t meshScratchAddress = scratchAddress;
        const uint64_t ownSliceOffset = static_cast<uint64_t>(param.myRank) * sliceBytes;

        for (uint32_t step = 0; step < plan.stepCount; ++step) {
            CHK_RET(SyncControlToAxes(resCtx));
            const std::array<uint64_t, 4> meshGroupOpSize
                = CalculateMsPipelineGroupOpSize(plan.meshSizes[step]);
            if (step == 0) {
                for (uint32_t destinationRank : resCtx.columnRanks) {
                    if (destinationRank == param.myRank) {
                        continue;
                    }
                    const uint64_t sliceOffset
                        = static_cast<uint64_t>(destinationRank) * sliceBytes + plan.meshOffsets[step];
                    CHK_RET(LaunchKernel(resCtx.threads[1], resCtx.ccuKernels[0], inputAddress,
                        inputAddress + sliceOffset, inputToken, inputToken, sliceOffset, plan.meshSizes[step],
                        CCU_REDUCE_SCATTER_MODE_OMNIPIPE_MESH, meshScratchAddress, scratchToken, sliceBytes,
                        meshGroupOpSize));
                }
            } else {
                const uint64_t sliceOffset = ownSliceOffset + plan.meshOffsets[step];
                CHK_RET(LaunchKernel(resCtx.threads[1], resCtx.ccuKernels[0], inputAddress, inputAddress + sliceOffset,
                    inputToken, inputToken, sliceOffset, plan.meshSizes[step],
                    CCU_REDUCE_SCATTER_MODE_OMNIPIPE_MESH, meshScratchAddress, scratchToken, sliceBytes,
                    meshGroupOpSize));
            }

            const uint32_t firstTargetLocalIndex
                = step < plan.stepCount - 2 ? 0 : resCtx.localRankIndex;
            const uint32_t targetLocalIndexLimit
                = step < plan.stepCount - 2 ? static_cast<uint32_t>(localGroupSize) : resCtx.localRankIndex + 1;
            std::vector<uint64_t> nhrSliceOffsets(serverGroupCount, 0);
            std::vector<uint64_t> nhrSliceSizes(serverGroupCount, plan.closSizes[step]);
            for (uint32_t targetLocalIndex = firstTargetLocalIndex;
                 targetLocalIndex < targetLocalIndexLimit; ++targetLocalIndex) {
                if (step < plan.stepCount - 2 && targetLocalIndex == resCtx.localRankIndex) {
                    continue;
                }
                for (uint32_t serverIndex = 0; serverIndex < serverGroupCount; ++serverIndex) {
                    const uint64_t gridIndex = serverIndex * localGroupSize + targetLocalIndex;
                    const uint64_t destinationRank = resCtx.globalRanks[gridIndex];
                    uint64_t destinationOffset = 0;
                    CHK_PRT_RET(!CheckedMultiply(destinationRank, sliceBytes, destinationOffset)
                                    || destinationOffset
                                           > std::numeric_limits<uint64_t>::max() - plan.closOffsets[step],
                        HCCL_ERROR("[ExecParallel2dOmnipipe] NHR slice address overflow"), HCCL_E_PARA);
                    nhrSliceOffsets[serverIndex] = destinationOffset + plan.closOffsets[step];
                }
                CHK_RET(LaunchNhrKernel(resCtx.threads[2], resCtx.ccuKernels[1], inputAddress, inputToken,
                    nhrSliceOffsets, nhrSliceSizes));
            }
            CHK_RET(SyncAxesToControl(resCtx));
        }

        CHK_RET(HcommLocalCopyOnThread(resCtx.threads[0], reinterpret_cast<void *>(outputAddress),
            reinterpret_cast<void *>(inputAddress + ownSliceOffset), sliceBytes));
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

    CHK_PRT_RET(resCtx.threads.empty(), HCCL_ERROR("[ExecOp] no CCU thread in resource context"), HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.ccuKernels.size() > 2 || resCtx.threads.size() < std::max<size_t>(1, resCtx.ccuKernels.size()),
        HCCL_ERROR("[ExecOp] invalid resource counts: threads[%zu], kernels[%zu]", resCtx.threads.size(),
            resCtx.ccuKernels.size()),
        HCCL_E_INTERNAL);

    auto dataTypeIt = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(dataTypeIt == SIZE_TABLE.end(), HCCL_ERROR("[ExecOp] unsupported dataType[%d]", param.dataType),
        HCCL_E_NOT_SUPPORT);
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }

    uint64_t sliceBytes = 0;
    uint64_t totalInputBytes = 0;
    CHK_PRT_RET(!CheckedMultiply(param.count, dataTypeIt->second, sliceBytes)
                    || !CheckedMultiply(sliceBytes, param.rankSize, totalInputBytes),
        HCCL_ERROR("[ExecOp] buffer size overflow"), HCCL_E_PARA);

    const uint64_t inputAddress = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outputAddress = reinterpret_cast<uint64_t>(param.outputPtr);
    const bool hasRemoteRanks = param.rankSize > 1;
    if (!hasRemoteRanks) {
        CHK_RET(HcommLocalCopyOnThread(resCtx.threads[0], param.outputPtr, param.inputPtr, sliceBytes));
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(RangesOverlap(inputAddress, totalInputBytes, outputAddress, sliceBytes),
        HCCL_ERROR("[ExecOp] in-place or overlapping buffers are not supported"), HCCL_E_NOT_SUPPORT);

    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(inputAddress, totalInputBytes, &inputToken));
    CHK_RET_CCU(HcommCcuGetMemToken(outputAddress, sliceBytes, &outputToken));

    if (resCtx.algorithm == REDUCE_SCATTER_ALGORITHM_PARALLEL_2D_OMNIPIPE) {
        return ExecParallel2dOmnipipe(
            param, resCtx, sliceBytes, dataTypeIt->second, inputAddress, outputAddress, inputToken);
    }
    CHK_PRT_RET(resCtx.algorithm != REDUCE_SCATTER_ALGORITHM_FULL_MESH,
        HCCL_ERROR("[ExecOp] unsupported algorithm[%u]", resCtx.algorithm), HCCL_E_INTERNAL);

    CHK_PRT_RET(hasRemoteRanks && resCtx.ccuKernels.empty(),
        HCCL_ERROR("[ExecOp] remote ranks exist but no CCU kernel was registered"), HCCL_E_INTERNAL);

    CHK_PRT_RET(resCtx.channelCounts.size() != resCtx.ccuKernels.size(),
        HCCL_ERROR("[ExecOp] channel count metadata[%zu] does not match kernels[%zu]", resCtx.channelCounts.size(),
            resCtx.ccuKernels.size()),
        HCCL_E_INTERNAL);

    uint64_t totalChannelCount = 0;
    for (uint32_t channelCount : resCtx.channelCounts) {
        totalChannelCount += channelCount;
    }
    CHK_PRT_RET(totalChannelCount != param.rankSize - 1,
        HCCL_ERROR(
            "[ExecOp] channel count[%lu] does not cover remote ranks[%u]", totalChannelCount, param.rankSize - 1),
        HCCL_E_INTERNAL);
    const bool msPipelineGraphWithinBudget
        = IsMsPipelineGraphWithinBudget(sliceBytes, param.rankSize, totalChannelCount);
    const bool useMsPipeline = IsMsPipelineSizeProfitable(sliceBytes) && CanUseMsPipeline(resCtx.channelCounts)
                               && msPipelineGraphWithinBudget;
    const bool useScratchFanIn = !useMsPipeline && sliceBytes >= SCRATCH_FANIN_THRESHOLD_BYTES && totalChannelCount > 1;
    const bool scratchBufferRequired = useScratchFanIn || resCtx.ccuKernels.size() == 2;
    uint64_t scratchAddress = 0;
    uint64_t scratchToken = 0;
    uint64_t scratchSlotBytes = MAX_DATA_SIZE;
    std::vector<uint64_t> scratchGroupOffsets(resCtx.channelCounts.size(), 0);
    if (scratchBufferRequired) {
        CHK_PRT_RET(resCtx.localBuffer.addr == nullptr || resCtx.localBuffer.size < dataTypeIt->second,
            HCCL_ERROR("[ExecOp] HCCL scratch buffer is unavailable"), HCCL_E_INTERNAL);
        scratchAddress = reinterpret_cast<uint64_t>(resCtx.localBuffer.addr);
        CHK_RET_CCU(HcommCcuGetMemToken(scratchAddress, resCtx.localBuffer.size, &scratchToken));

        if (useScratchFanIn) {
            CHK_PRT_RET(resCtx.localBuffer.size < totalChannelCount * dataTypeIt->second,
                HCCL_ERROR("[ExecOp] HCCL scratch buffer is too small for peer slots"), HCCL_E_INTERNAL);
            scratchSlotBytes = std::min<uint64_t>(MAX_DATA_SIZE, resCtx.localBuffer.size / totalChannelCount);
            scratchSlotBytes -= scratchSlotBytes % dataTypeIt->second;
            CHK_PRT_RET(scratchSlotBytes == 0, HCCL_ERROR("[ExecOp] scratch slot size is zero"), HCCL_E_INTERNAL);
            uint64_t scratchGroupOffset = 0;
            for (uint32_t index = 0; index < resCtx.channelCounts.size(); ++index) {
                scratchGroupOffsets[index] = scratchGroupOffset;
                scratchGroupOffset += static_cast<uint64_t>(resCtx.channelCounts[index]) * scratchSlotBytes;
            }
            CHK_PRT_RET(scratchGroupOffset > resCtx.localBuffer.size,
                HCCL_ERROR(
                    "[ExecOp] scratch slots[%lu] exceed HCCL buffer[%lu]", scratchGroupOffset, resCtx.localBuffer.size),
                HCCL_E_INTERNAL);
        } else {
            scratchSlotBytes = std::min<uint64_t>(MAX_DATA_SIZE, resCtx.localBuffer.size);
        }
    }
    scratchSlotBytes -= scratchSlotBytes % dataTypeIt->second;
    CHK_PRT_RET(scratchSlotBytes == 0, HCCL_ERROR("[ExecOp] scratch slot size is zero"), HCCL_E_INTERNAL);

    const uint64_t maxChunkCount = scratchSlotBytes / dataTypeIt->second;
    CHK_PRT_RET(maxChunkCount == 0, HCCL_ERROR("[ExecOp] maximum chunk size is zero"), HCCL_E_INTERNAL);

    const uint64_t rankSliceOffset = static_cast<uint64_t>(param.myRank) * sliceBytes;
    uint64_t processedCount = 0;
    while (processedCount < param.count) {
        uint64_t chunkCount = std::min(maxChunkCount, param.count - processedCount);
        const uint64_t processedBytes = processedCount * dataTypeIt->second;
        uint64_t chunkBytes = chunkCount * dataTypeIt->second;
        const uint64_t remoteSliceOffset = rankSliceOffset + processedBytes;

        if (hasRemoteRanks) {
            uint64_t reducePeersMode = CCU_REDUCE_SCATTER_MODE_REDUCE_PEERS;
            if (useMsPipeline && chunkBytes >= MS_PIPELINE_MIN_BYTES) {
                reducePeersMode = CCU_REDUCE_SCATTER_MODE_MS_PIPELINE;
            } else if (useScratchFanIn && chunkBytes >= SCRATCH_FANIN_THRESHOLD_BYTES) {
                reducePeersMode = CCU_REDUCE_SCATTER_MODE_REDUCE_PEERS_SCRATCH;
            }
            const std::array<uint64_t, 4> msPipelineGroupOpSize = reducePeersMode == CCU_REDUCE_SCATTER_MODE_MS_PIPELINE
                                                                      ? CalculateMsPipelineGroupOpSize(chunkBytes)
                                                                      : EMPTY_MS_PIPELINE_GROUP_OP_SIZE;
            CHK_RET(SyncMainToSub(resCtx));
            uint64_t scratchBaseAddress = scratchAddress + scratchGroupOffsets[0];
            CHK_RET(LaunchKernel(resCtx.threads[0], resCtx.ccuKernels[0], inputAddress, outputAddress + processedBytes,
                inputToken, outputToken, remoteSliceOffset, chunkBytes, reducePeersMode, scratchBaseAddress,
                scratchToken, scratchSlotBytes, msPipelineGroupOpSize));

            if (resCtx.ccuKernels.size() == 2) {
                scratchBaseAddress = scratchAddress + scratchGroupOffsets[1];
                CHK_RET(LaunchKernel(resCtx.threads[1], resCtx.ccuKernels[1], inputAddress, scratchBaseAddress,
                    inputToken, scratchToken, remoteSliceOffset, chunkBytes, reducePeersMode, scratchBaseAddress,
                    scratchToken, scratchSlotBytes, msPipelineGroupOpSize));
                CHK_RET(SyncSubToMain(resCtx));
                CHK_RET(LaunchKernel(resCtx.threads[0], resCtx.ccuKernels[0], scratchBaseAddress,
                    outputAddress + processedBytes, scratchToken, outputToken, 0, chunkBytes,
                    CCU_REDUCE_SCATTER_MODE_MERGE_PARTIALS, scratchAddress, scratchToken, scratchSlotBytes,
                    EMPTY_MS_PIPELINE_GROUP_OP_SIZE));
            }
        }
        processedCount += chunkCount;
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
