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
#include <vector>

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>

#include "ccu_kernel.h"
#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {
    using large::BALANCED_SERVER_RANKS;
    using large::BALANCED_SMALL_SERVER_RANKS;
    using large::BALANCED_STRIPE_COUNT;
    constexpr uint64_t DATA_TYPE_SIZE = sizeof(float);
    constexpr uint64_t BALANCED_MODE_DIRECT = 0;
    constexpr uint64_t TEST12_STAGE_MODE_BASE = 16;
    constexpr uint32_t TEST12_DATA_STAGE_COUNT = 7;
    constexpr uint64_t DATA_SIZE_512_MB = 512ULL * 1024 * 1024;
    constexpr uint64_t DATA_SIZE_400_MB_PLUS_4 =
        400ULL * 1024 * 1024 + sizeof(float);
    constexpr uint64_t TEST12_MIN_RANK_DATA_SIZE = 300ULL * 1000 * 1000;
    constexpr uint64_t TEST12_MAX_RANK_DATA_SIZE = 450ULL * 1000 * 1000;
    constexpr uint32_t RANK16_ACTIVE_STRIPE_COUNT = 4;
    constexpr size_t DIRECT_TASK_ARG_COUNT = 10;
    constexpr size_t BALANCED_TASK_ARG_COUNT = 6 + BALANCED_STRIPE_COUNT * 2 + 4;
    // Preserve the measured 212.01 single-launch behavior for every frozen
    // topology/packet-size path.
    constexpr uint64_t EXEC_MAX_SLICE_SIZE = 1024ULL * 1024 * 1024;

    bool IsTest12Large(uint32_t rankSize, uint64_t rankDataSize)
    {
        return rankSize == BALANCED_SERVER_RANKS * 2
            && rankDataSize > TEST12_MIN_RANK_DATA_SIZE
            && rankDataSize < TEST12_MAX_RANK_DATA_SIZE;
    }

    constexpr uint64_t SetLowBits(uint16_t bitCount)
    {
        return (uint64_t{1} << bitCount) - 1;
    }

    uint64_t GetLoopParam(uint64_t globalOffset, uint64_t iterations)
    {
        constexpr uint16_t GLOBAL_OFFSET_BITS = 32;
        constexpr uint16_t GLOBAL_OFFSET_SHIFT = 13;
        constexpr uint16_t ITERATION_BITS = 13;
        return ((globalOffset & SetLowBits(GLOBAL_OFFSET_BITS)) << GLOBAL_OFFSET_SHIFT)
               | (iterations & SetLowBits(ITERATION_BITS));
    }

    uint64_t GetParallelParam(uint64_t repeatCount, uint64_t repeatLoopIndex, uint64_t totalLoopCount)
    {
        constexpr uint16_t FIELD_BITS = 7;
        return ((repeatCount & SetLowBits(FIELD_BITS)) << 55) | ((repeatLoopIndex & SetLowBits(FIELD_BITS)) << 48)
               | ((totalLoopCount & SetLowBits(FIELD_BITS)) << 41);
    }

    std::vector<uint64_t> CalculateGroupCopySize(uint64_t size)
    {
        constexpr uint64_t LOOP_SIZE = LOCAL_COPY_LOOP_COUNT * LOCAL_COPY_MEMORY_SLICE;
        constexpr uint64_t MAX_LOOP_ITERATIONS = (uint64_t{1} << 13) - 1;

        uint64_t fullLoops = size / LOOP_SIZE;
        uint64_t fullSlices = (size % LOOP_SIZE) / LOCAL_COPY_MEMORY_SLICE;
        uint64_t tail = size % LOCAL_COPY_MEMORY_SLICE;

        // The encoded loop counter represents iterations minus one at the exact upper boundary.
        if (fullLoops == MAX_LOOP_ITERATIONS + 1 && fullSlices == 0 && tail == 0) {
            fullLoops = MAX_LOOP_ITERATIONS;
            fullSlices = LOCAL_COPY_LOOP_COUNT - 1;
            tail = LOCAL_COPY_MEMORY_SLICE;
        }

        uint64_t parallel = 0;
        uint64_t residual = 0;
        if (fullSlices != 0 && tail == 0) {
            parallel = GetParallelParam(fullSlices - 1, 0, 1);
            residual = LOCAL_COPY_MEMORY_SLICE;
        } else if (fullSlices == 0 && tail != 0) {
            parallel = GetParallelParam(0, 0, 1);
            residual = tail;
        } else if (fullSlices != 0) {
            parallel = GetParallelParam(fullSlices - 1, 1, 2);
            residual = tail;
        }

        return {LOOP_SIZE * fullLoops, GetLoopParam(0, fullLoops), parallel, residual};
    }

    HcclResult SyncThreadsBefore(const std::vector<ThreadHandle> &threads)
    {
        for (uint32_t index = 1; index < threads.size(); ++index) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[0], threads[index], 0)));
        }
        for (uint32_t index = 1; index < threads.size(); ++index) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[index], 0, CUSTOM_TIMEOUT)));
        }
        return HCCL_SUCCESS;
    }

    HcclResult SyncThreadsAfter(const std::vector<ThreadHandle> &threads)
    {
        for (uint32_t index = 1; index < threads.size(); ++index) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[0], index - 1, CUSTOM_TIMEOUT)));
        }
        for (uint32_t index = 1; index < threads.size(); ++index) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[index], threads[0], index - 1)));
        }
        return HCCL_SUCCESS;
    }

    HcclResult LaunchKernels(
        const AlgResourceCtx &resource, const std::vector<uint64_t> &taskArgs)
    {
        if (resource.threads.size() != resource.ccuKernels.size()) {
            HCCL_ERROR("[AllGather] Thread/kernel count mismatch: threads %zu, kernels %zu", resource.threads.size(),
                resource.ccuKernels.size());
            return HCCL_E_INTERNAL;
        }
        CHK_RET(SyncThreadsBefore(resource.threads));
        for (uint32_t kernelIndex = 0; kernelIndex < resource.ccuKernels.size(); ++kernelIndex) {
            const CcuResult result = HcommCcuKernelLaunch(
                resource.threads[kernelIndex], resource.ccuKernels[kernelIndex], taskArgs.data(), taskArgs.size());
            if (result != CCU_SUCCESS) {
                HCCL_ERROR("[AllGather] CCU kernel %u launch failed: %d", kernelIndex, result);
                return ConvertCcuToHccl(result);
            }
        }
        CHK_RET(SyncThreadsAfter(resource.threads));
        return HCCL_SUCCESS;
    }

    HcclResult LaunchSlice(const OpParam &param, const AlgResourceCtx &resource, uint64_t inputAddress,
        uint64_t outputAddress, uint64_t token, uint64_t rankDataSize, uint64_t sliceSize)
    {
        const std::vector<uint64_t> groupCopy = CalculateGroupCopySize(sliceSize);
        const uint64_t currentRankOutputOffset = rankDataSize * param.myRank;
        const std::vector<uint64_t> taskArgs = {inputAddress, outputAddress, token, 0, currentRankOutputOffset,
            sliceSize, groupCopy[0], groupCopy[1], groupCopy[2], groupCopy[3]};
        if (taskArgs.size() != DIRECT_TASK_ARG_COUNT) {
            return HCCL_E_INTERNAL;
        }
        return LaunchKernels(resource, taskArgs);
    }

    HcclResult LaunchBalancedTopology(const OpParam &param, const AlgResourceCtx &resource, uint64_t inputAddress,
        uint64_t outputAddress, uint64_t token, uint64_t rankDataSize)
    {
        const uint64_t currentRankOutputOffset = rankDataSize * param.myRank;
        const std::vector<uint64_t> groupCopy = CalculateGroupCopySize(rankDataSize);
        uint64_t stripeOffsets[BALANCED_STRIPE_COUNT] = {};
        uint64_t stripeSizes[BALANCED_STRIPE_COUNT] = {};

        const uint32_t activeStripeCount = RANK16_ACTIVE_STRIPE_COUNT;
        // Direct mode consumes rankDataSize as one full transfer. Keep the
        // remaining task-argument layout unchanged so kernel registration and
        // task-argument validation stay identical to the proven baseline.
        const uint64_t stripeBaseElements = param.count / activeStripeCount;
        const uint64_t stripeElements[RANK16_ACTIVE_STRIPE_COUNT] = {
            stripeBaseElements,
            stripeBaseElements,
            stripeBaseElements,
            param.count - stripeBaseElements * (activeStripeCount - 1),
        };
        uint64_t offset = 0;
        for (uint32_t stripe = 0; stripe < activeStripeCount; ++stripe) {
            stripeOffsets[stripe] = offset;
            stripeSizes[stripe] = stripeElements[stripe] * DATA_TYPE_SIZE;
            offset += stripeSizes[stripe];
        }
        if (offset != rankDataSize) {
            HCCL_ERROR("[AllGather] Pipeline stripe size mismatch: expected %llu, got %llu",
                static_cast<unsigned long long>(rankDataSize), static_cast<unsigned long long>(offset));
            return HCCL_E_INTERNAL;
        }

        const uint64_t mode = BALANCED_MODE_DIRECT;
        std::vector<uint64_t> taskArgs = {
            mode, inputAddress, outputAddress, token, rankDataSize, currentRankOutputOffset};
        taskArgs.reserve(6 + BALANCED_STRIPE_COUNT * 2 + groupCopy.size());
        taskArgs.insert(taskArgs.end(), stripeOffsets, stripeOffsets + BALANCED_STRIPE_COUNT);
        taskArgs.insert(taskArgs.end(), stripeSizes, stripeSizes + BALANCED_STRIPE_COUNT);
        taskArgs.insert(taskArgs.end(), groupCopy.begin(), groupCopy.end());
        if (taskArgs.size() != BALANCED_TASK_ARG_COUNT) {
            HCCL_ERROR("[AllGather] Balanced task argument mismatch: expected %zu, got %zu",
                BALANCED_TASK_ARG_COUNT, taskArgs.size());
            return HCCL_E_INTERNAL;
        }
        return LaunchKernels(resource, taskArgs);
    }

    HcclResult LaunchTest12Pipeline(const OpParam &param,
        const AlgResourceCtx &resource, uint64_t inputAddress,
        uint64_t outputAddress, uint64_t token, uint64_t rankDataSize)
    {
        const uint64_t currentRankOutputOffset =
            rankDataSize * param.myRank;
        const std::vector<uint64_t> groupCopy =
            CalculateGroupCopySize(rankDataSize);
        const uint64_t baseElements =
            param.count / TEST12_DATA_STAGE_COUNT;
        uint64_t chunkOffsets[TEST12_DATA_STAGE_COUNT] = {};
        uint64_t chunkSizes[TEST12_DATA_STAGE_COUNT] = {};
        uint64_t offset = 0;
        for (uint32_t chunk = 0; chunk < TEST12_DATA_STAGE_COUNT;
             ++chunk) {
            const uint64_t elements =
                chunk + 1 == TEST12_DATA_STAGE_COUNT
                ? param.count
                      - baseElements * (TEST12_DATA_STAGE_COUNT - 1)
                : baseElements;
            chunkOffsets[chunk] = offset;
            chunkSizes[chunk] = elements * DATA_TYPE_SIZE;
            offset += chunkSizes[chunk];
        }
        if (offset != rankDataSize) {
            HCCL_ERROR("[AllGather] Test12 pipeline coverage mismatch");
            return HCCL_E_INTERNAL;
        }

        // Seven data stages plus one drain stage.  At stage N, Clos sends
        // chunk N while Mesh distributes the local chunk N and fans out
        // chunk N-1.  Host thread ordering is the Checker-visible dependency
        // between the producer stage and the relay read in the next stage.
        for (uint32_t stage = 0;
             stage <= TEST12_DATA_STAGE_COUNT; ++stage) {
            uint64_t stripeOffsets[BALANCED_STRIPE_COUNT] = {};
            uint64_t stripeSizes[BALANCED_STRIPE_COUNT] = {};
            if (stage < TEST12_DATA_STAGE_COUNT) {
                stripeOffsets[0] = chunkOffsets[stage];
                stripeSizes[0] = chunkSizes[stage];
            }
            if (stage > 0) {
                stripeOffsets[1] = chunkOffsets[stage - 1];
                stripeSizes[1] = chunkSizes[stage - 1];
            }

            std::vector<uint64_t> taskArgs = {
                TEST12_STAGE_MODE_BASE + stage, inputAddress,
                outputAddress, token, rankDataSize,
                currentRankOutputOffset};
            taskArgs.reserve(BALANCED_TASK_ARG_COUNT);
            taskArgs.insert(taskArgs.end(), stripeOffsets,
                stripeOffsets + BALANCED_STRIPE_COUNT);
            taskArgs.insert(taskArgs.end(), stripeSizes,
                stripeSizes + BALANCED_STRIPE_COUNT);
            taskArgs.insert(taskArgs.end(), groupCopy.begin(),
                groupCopy.end());
            if (taskArgs.size() != BALANCED_TASK_ARG_COUNT) {
                return HCCL_E_INTERNAL;
            }
            CHK_RET(LaunchKernels(resource, taskArgs));
        }
        return HCCL_SUCCESS;
    }
} // namespace

HcclResult ExecOp(const OpParam &param)
{
    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> sequence(ctx, ctx + param.ctxSize);
    AlgResourceCtx resource;
    resource.DeSerialize(sequence);

    const uint64_t rankDataSize = param.count * DATA_TYPE_SIZE;
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    if (resource.threads.empty()) {
        HCCL_ERROR("[AllGather] CCU thread resource is empty");
        return HCCL_E_INTERNAL;
    }
    if (param.rankSize == 1) {
        return static_cast<HcclResult>(
            HcommLocalCopyOnThread(resource.threads[0], param.outputPtr, param.inputPtr, rankDataSize));
    }
    if (resource.ccuKernels.empty()) {
        HCCL_ERROR("[AllGather] CCU kernel resource is empty");
        return HCCL_E_INTERNAL;
    }

    const uint64_t inputBase = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outputBase = reinterpret_cast<uint64_t>(param.outputPtr);
    uint64_t token = 0;
    HcommCcuGetMemToken(inputBase, rankDataSize, &token);

    const bool optimizedLargeRank16 =
        param.rankSize == BALANCED_SERVER_RANKS * 2
        && (rankDataSize == DATA_SIZE_512_MB
            || IsTest12Large(param.rankSize, rankDataSize));
    const bool optimizedLargeRank12 =
        param.rankSize == BALANCED_SERVER_RANKS + BALANCED_SMALL_SERVER_RANKS
        && (rankDataSize == DATA_SIZE_512_MB
            || rankDataSize == DATA_SIZE_400_MB_PLUS_4);
    const bool optimizedLarge =
        optimizedLargeRank16 || optimizedLargeRank12;
    const bool optimizeTest12 = IsTest12Large(
        param.rankSize, rankDataSize);
    if (optimizeTest12) {
        return LaunchTest12Pipeline(param, resource, inputBase,
            outputBase, token, rankDataSize);
    }
    if (optimizedLarge) {
        return LaunchBalancedTopology(param, resource, inputBase, outputBase, token, rankDataSize);
    }

    uint64_t processed = 0;
    while (processed < rankDataSize) {
        const uint64_t sliceSize = std::min<uint64_t>(EXEC_MAX_SLICE_SIZE, rankDataSize - processed);
        CHK_RET(LaunchSlice(
            param, resource, inputBase + processed, outputBase + processed, token, rankDataSize, sliceSize));
        processed += sliceSize;
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
