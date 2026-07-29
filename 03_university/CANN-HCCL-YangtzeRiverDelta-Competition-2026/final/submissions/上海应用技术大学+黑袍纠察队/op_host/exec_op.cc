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

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {
    constexpr uint32_t THREAD_NOTIFY_INDEX = 0;
    constexpr uint64_t DIRECT_THRESHOLD_BYTES = 1024ULL * 1024ULL;
    constexpr uint32_t COMBINED_TWO_SHOT_RANK_SIZE = 4;
    constexpr uint32_t PIPELINE_BLOCK_COUNT = 8;

    template <size_t N>
    HcclResult LaunchKernel(ThreadHandle thread, CcuKernelHandle kernel, const std::array<uint64_t, N> &taskArgs)
    {
        const CcuResult result = HcommCcuKernelLaunch(thread, kernel, taskArgs.data(), static_cast<uint32_t>(N));
        if (result != CCU_SUCCESS) {
            HCCL_ERROR("Launch CCU broadcast kernel failed, result[%d]", result);
            return ConvertCcuToHccl(result);
        }
        return HCCL_SUCCESS;
    }

    HcclResult PreSyncThreads(const std::vector<ThreadHandle> &threads)
    {
        if (threads.size() < 2) {
            return HCCL_SUCCESS;
        }
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[0], threads[1], THREAD_NOTIFY_INDEX)));
        CHK_RET(
            static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[1], THREAD_NOTIFY_INDEX, CUSTOM_TIMEOUT)));
        return HCCL_SUCCESS;
    }

    HcclResult PostSyncThreads(const std::vector<ThreadHandle> &threads)
    {
        if (threads.size() < 2) {
            return HCCL_SUCCESS;
        }
        CHK_RET(
            static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(threads[0], THREAD_NOTIFY_INDEX, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[1], threads[0], THREAD_NOTIFY_INDEX)));
        return HCCL_SUCCESS;
    }

    template <size_t N>
    HcclResult RunKernelGroup(const std::vector<ThreadHandle> &threads, const std::vector<CcuKernelHandle> &kernels,
        const std::array<uint64_t, N> &taskArgs)
    {
        CHK_PRT_RET(
            threads.size() != kernels.size(), HCCL_ERROR("CCU thread and kernel count mismatch"), HCCL_E_INTERNAL);
        CHK_RET(PreSyncThreads(threads));
        for (uint32_t kernelIndex = 0; kernelIndex < kernels.size(); ++kernelIndex) {
            CHK_RET(LaunchKernel(threads[kernelIndex], kernels[kernelIndex], taskArgs));
        }
        CHK_RET(PostSyncThreads(threads));
        return HCCL_SUCCESS;
    }
} // namespace

HcclResult ExecOp(const OpParam &param)
{
    CHK_PTR_NULL(param.resCtx);
    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> sequence(ctx, ctx + param.ctxSize);
    AlgResourceCtx resCtx;
    resCtx.DeSerialize(sequence);
    const size_t kernelGroupCount = resCtx.directKernels.size();
    const bool useCombinedTwoShot = !resCtx.twoShotKernels.empty();
    const bool validCombinedResource = useCombinedTwoShot && param.rankSize == COMBINED_TWO_SHOT_RANK_SIZE
                                       && kernelGroupCount == 1 && resCtx.twoShotKernels.size() == 1
                                       && resCtx.scatterKernels.empty() && resCtx.allGatherKernels.empty();
    const bool validSplitResource = !useCombinedTwoShot && resCtx.scatterKernels.size() == kernelGroupCount
                                    && resCtx.allGatherKernels.size() == kernelGroupCount;
    CHK_PRT_RET(kernelGroupCount == 0 || kernelGroupCount > 2 || resCtx.threads.size() != kernelGroupCount
                    || (!validCombinedResource && !validSplitResource),
        HCCL_ERROR(
            "Invalid CCU broadcast resource: threads[%zu], direct[%zu], twoshot[%zu], scatter[%zu], allgather[%zu]",
            resCtx.threads.size(), resCtx.directKernels.size(), resCtx.twoShotKernels.size(),
            resCtx.scatterKernels.size(), resCtx.allGatherKernels.size()),
        HCCL_E_INTERNAL);
    resCtx.threads[0] = param.cpuThread;

    const auto sizeIt = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(sizeIt == SIZE_TABLE.end(), HCCL_ERROR("Unsupported data type[%d]", param.dataType), HCCL_E_PARA);
    const uint64_t typeSize = sizeIt->second;
    const uint64_t totalBytes = param.count * typeSize;
    if (totalBytes == 0 || param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    const uint64_t baseAddress = reinterpret_cast<uint64_t>(param.inputPtr);
    uint64_t token = 0;
    const CcuResult tokenResult = HcommCcuGetMemToken(baseAddress, totalBytes, &token);
    if (tokenResult != CCU_SUCCESS) {
        HCCL_ERROR("Get CCU memory token failed, result[%d]", tokenResult);
        return ConvertCcuToHccl(tokenResult);
    }

    const uint64_t maxElementsPerTransfer = static_cast<uint64_t>(MAX_DATA_SIZE) / typeSize;
    const uint64_t rankRemainder = param.rankSize - 1;
    // MAX_DATA_SIZE limits each generated Write, not the whole collective. Account for the implementation's
    // remainder-in-last-slice layout so every slice remains within the transfer limit.
    const uint64_t maxElementsPerChunk
        = maxElementsPerTransfer * param.rankSize - rankRemainder * rankRemainder;
    uint64_t processedElements = 0;
    while (processedElements < param.count) {
        const uint64_t currentElements = std::min(maxElementsPerChunk, param.count - processedElements);
        const uint64_t chunkAddress = baseAddress + processedElements * typeSize;
        const uint64_t currentBytes = currentElements * typeSize;
        if (currentBytes <= DIRECT_THRESHOLD_BYTES) {
            const std::array<uint64_t, 3> taskArgs = {chunkAddress, token, currentBytes};
            CHK_RET(RunKernelGroup(resCtx.threads, resCtx.directKernels, taskArgs));
        } else {
            // On the asymmetric 12-rank topology, keep the root out of the all-gather ownership set.
            // The root already owns the complete input, so assigning it another slice only adds traffic.
            const uint64_t sliceCount = param.rankSize == 12 ? param.rankSize - 1 : param.rankSize;
            const uint64_t normalElements = currentElements / sliceCount;
            const uint64_t lastElements = normalElements + currentElements % sliceCount;
            const std::array<uint64_t, 4> taskArgs = {
                chunkAddress,
                token,
                normalElements * typeSize,
                lastElements * typeSize,
            };
            if (useCombinedTwoShot) {
                const uint64_t ownerCount = param.rankSize - 1;
                const uint64_t normalOwnerElements = currentElements / ownerCount;
                const uint64_t lastOwnerElements = normalOwnerElements + currentElements % ownerCount;
                const uint64_t normalBlockElements = normalOwnerElements / PIPELINE_BLOCK_COUNT;
                const uint64_t lastBlockElements = lastOwnerElements / PIPELINE_BLOCK_COUNT;
                const std::array<uint64_t, 8> pipelineTaskArgs = {
                    chunkAddress,
                    token,
                    normalOwnerElements * typeSize,
                    lastOwnerElements * typeSize,
                    normalBlockElements * typeSize,
                    (normalBlockElements + normalOwnerElements % PIPELINE_BLOCK_COUNT) * typeSize,
                    lastBlockElements * typeSize,
                    (lastBlockElements + lastOwnerElements % PIPELINE_BLOCK_COUNT) * typeSize,
                };
                CHK_RET(RunKernelGroup(resCtx.threads, resCtx.twoShotKernels, pipelineTaskArgs));
            } else {
                CHK_RET(RunKernelGroup(resCtx.threads, resCtx.scatterKernels, taskArgs));
                CHK_RET(RunKernelGroup(resCtx.threads, resCtx.allGatherKernels, taskArgs));
            }
        }
        processedElements += currentElements;
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
