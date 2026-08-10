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
#include <limits>
#include <vector>

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>

#include "ccu_kernel.h"
#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {

namespace {

HcclResult PreSyncInterThreads(
    const ThreadHandle &mainThread, const std::vector<ThreadHandle> &subThreads)
{
    for (const ThreadHandle &subThread : subThreads) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(mainThread, subThread, NOTIFY_IDX_ACK)));
    }
    for (const ThreadHandle &subThread : subThreads) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(subThread, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult PostSyncInterThreads(
    const ThreadHandle &mainThread, const std::vector<ThreadHandle> &subThreads)
{
    for (size_t index = 0; index < subThreads.size(); ++index) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(mainThread, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
    }
    for (const ThreadHandle &subThread : subThreads) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(subThread, mainThread, NOTIFY_IDX_ACK)));
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchKernelRange(const OpParam &param, const AlgResourceCtx &resource,
    size_t firstKernel, size_t kernelCount, const std::vector<uint64_t> &taskArgs)
{
    CHK_PRT_RET(kernelCount == 0 ||
            firstKernel > resource.ccuKernels.size() ||
            kernelCount > resource.ccuKernels.size() - firstKernel ||
            resource.threads.size() + 1 < kernelCount,
        HCCL_ERROR("[AllGather] invalid kernel launch range"), HCCL_E_INTERNAL);

    std::vector<ThreadHandle> activeSubThreads;
    if (kernelCount > 1) {
        activeSubThreads.assign(
            resource.threads.begin(), resource.threads.begin() + kernelCount - 1);
        CHK_RET(PreSyncInterThreads(param.cpuThread, activeSubThreads));
    }

    for (size_t relativeIndex = 0; relativeIndex < kernelCount; ++relativeIndex) {
        const ThreadHandle launchThread =
            relativeIndex == 0 ? param.cpuThread : activeSubThreads[relativeIndex - 1];
        const size_t kernelIndex = firstKernel + relativeIndex;
        CcuResult launchRet = HcommCcuKernelLaunch(launchThread,
            resource.ccuKernels[kernelIndex], taskArgs.data(),
            static_cast<uint32_t>(taskArgs.size()));
        if (launchRet != CCU_SUCCESS) {
            HCCL_ERROR("[AllGather] CCU kernel launch %zu failed, ret %d",
                kernelIndex, launchRet);
            return ConvertCcuToHccl(launchRet);
        }
    }

    if (!activeSubThreads.empty()) {
        CHK_RET(PostSyncInterThreads(param.cpuThread, activeSubThreads));
    }
    return HCCL_SUCCESS;
}

} // namespace

HcclResult ExecOp(const OpParam &param)
{
    CHK_PRT_RET(param.resCtx == nullptr || param.ctxSize == 0,
        HCCL_ERROR("[AllGather] resource context is invalid"), HCCL_E_INTERNAL);
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || param.myRank >= param.rankSize,
        HCCL_ERROR("[AllGather] invalid rank %u of %u", param.myRank, param.rankSize), HCCL_E_PARA);

    const auto typeSizeIter = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(typeSizeIter == SIZE_TABLE.end(),
        HCCL_ERROR("[AllGather] unsupported data type %d", param.dataType), HCCL_E_NOT_SUPPORT);
    const uint64_t typeSize = typeSizeIter->second;
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / typeSize,
        HCCL_ERROR("[AllGather] input byte size overflows"), HCCL_E_PARA);
    const uint64_t rankBytes = param.count * typeSize;
    if (rankBytes == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(rankBytes > std::numeric_limits<uint64_t>::max() / param.rankSize,
        HCCL_ERROR("[AllGather] output byte size overflows"), HCCL_E_PARA);
    const uint64_t outputBytes = rankBytes * param.rankSize;

    const char *contextBytes = static_cast<const char *>(param.resCtx);
    std::vector<char> sequence(contextBytes, contextBytes + param.ctxSize);
    AlgResourceCtx resource{};
    resource.DeSerialize(sequence);
    CHK_PRT_RET(resource.ccuKernels.empty() || resource.ccuKernels.size() > 4,
        HCCL_ERROR("[AllGather] invalid registered kernel count %zu", resource.ccuKernels.size()),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(resource.phaseOneKernelCount == 0 ||
            resource.phaseOneKernelCount > 2 ||
            resource.phaseTwoKernelCount > 2 ||
            static_cast<size_t>(resource.phaseOneKernelCount + resource.phaseTwoKernelCount) !=
                resource.ccuKernels.size(),
        HCCL_ERROR("[AllGather] invalid phase kernel counts %u/%u",
            resource.phaseOneKernelCount, resource.phaseTwoKernelCount),
        HCCL_E_INTERNAL);
    const size_t maxPhaseKernelCount =
        std::max(resource.phaseOneKernelCount, resource.phaseTwoKernelCount);
    CHK_PRT_RET(resource.threads.size() + 1 != maxPhaseKernelCount,
        HCCL_ERROR("[AllGather] phase kernel/thread count mismatch"), HCCL_E_INTERNAL);
    const bool directAlgorithm =
        resource.algorithm == static_cast<uint32_t>(AllGatherAlgorithm::DIRECT);
    const bool mixedAlgorithm =
        resource.algorithm == static_cast<uint32_t>(AllGatherAlgorithm::TWO_SERVER_EIGHT);
    CHK_PRT_RET(!directAlgorithm && !mixedAlgorithm,
        HCCL_ERROR("[AllGather] unsupported cached algorithm %u", resource.algorithm), HCCL_E_INTERNAL);
    CHK_PRT_RET((directAlgorithm &&
                    (resource.phaseTwoKernelCount != 0 ||
                        resource.partnerRank != INVALID_VALUE_RANKID)) ||
            (mixedAlgorithm &&
                (param.rankSize != 16 ||
                    resource.phaseTwoKernelCount == 0 ||
                    resource.partnerRank == param.myRank ||
                    resource.partnerRank >= param.rankSize)),
        HCCL_ERROR("[AllGather] inconsistent cached algorithm resources"), HCCL_E_INTERNAL);

    const uint64_t inputBase = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outputBase = reinterpret_cast<uint64_t>(param.outputPtr);
    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(inputBase, rankBytes, &inputToken));
    CHK_RET_CCU(HcommCcuGetMemToken(outputBase, outputBytes, &outputToken));

    const uint64_t selfRankOffset = static_cast<uint64_t>(param.myRank) * rankBytes;
    const bool inputAlreadyInPlace = inputBase == outputBase + selfRankOffset;

    uint64_t processedBytes = 0;
    while (processedBytes < rankBytes) {
        const uint64_t chunkBytes =
            std::min<uint64_t>(MAX_DATA_SIZE, rankBytes - processedBytes);
        constexpr uint64_t MIXED_RELAY_MIN_BYTES = 4ULL * 1024ULL * 1024ULL;
        uint64_t directBytes = chunkBytes;
        if (mixedAlgorithm && chunkBytes >= MIXED_RELAY_MIN_BYTES) {
            directBytes = (chunkBytes * 3ULL / 7ULL);
            directBytes -= directBytes % typeSize;
        }
        const uint64_t relayBytes = chunkBytes - directBytes;
        uint64_t relaySourceOffset = 0;
        uint64_t relaySourceAddress = outputBase;
        if (relayBytes != 0) {
            relaySourceOffset =
                static_cast<uint64_t>(resource.partnerRank) * rankBytes +
                processedBytes + directBytes;
            relaySourceAddress = outputBase + relaySourceOffset;
        }
        const std::vector<uint64_t> taskArgs = {
            inputBase + processedBytes,
            outputBase,
            inputToken,
            outputToken,
            selfRankOffset + processedBytes,
            chunkBytes,
            inputAlreadyInPlace ? 0ULL : 1ULL,
            directBytes,
            relaySourceAddress,
            relaySourceOffset,
            relayBytes,
        };
        static_assert(TASK_ARG_COUNT == 11, "task argument contract changed");
        CHK_PRT_RET(taskArgs.size() != TASK_ARG_COUNT,
            HCCL_ERROR("[AllGather] invalid task argument count"), HCCL_E_INTERNAL);

        CHK_RET(LaunchKernelRange(
            param, resource, 0, resource.phaseOneKernelCount, taskArgs));
        if (relayBytes != 0) {
            CHK_RET(LaunchKernelRange(param, resource, resource.phaseOneKernelCount,
                resource.phaseTwoKernelCount, taskArgs));
        }
        processedBytes += chunkBytes;
    }

    return HCCL_SUCCESS;
}

} // namespace ops_hccl
