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
#include <cstddef>
#include <cstring>
#include <limits>

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {
constexpr uint64_t SLICE_ALIGNMENT = 128;
constexpr uint64_t DATA_TYPE_SIZE = sizeof(float);
constexpr uint32_t NOTIFY_PHASE_COUNT = MAX_KERNEL_PHASE_COUNT;
constexpr uint32_t GROUP_FLAT_PREFERRED_BATCH_TILES = 4;
constexpr uint32_t GROUP_FLAT_MAX_BATCH_TILES = 16;
constexpr uint64_t STRIPE_MESSAGE_THRESHOLD = 1ULL << 20;

struct GroupFlatBatchPlan {
    uint64_t tileBytes = 0;
    uint64_t batchBytes = 0;
    uint64_t tileCount = 0;
};

bool IsAddressRangeValid(uint64_t address, uint64_t size)
{
    return size == 0 || address <= std::numeric_limits<uint64_t>::max() - (size - 1);
}

HcclResult GetToken(uint64_t address, uint64_t size, uint64_t &token)
{
    CHK_PRT_RET(address == 0 || size == 0 || !IsAddressRangeValid(address, size),
        HCCL_ERROR("[GetToken] invalid memory range"), HCCL_E_PARA);
    CHK_RET_CCU(HcommCcuGetMemToken(address, size, &token));
    return HCCL_SUCCESS;
}

template <size_t N>
HcclResult LaunchKernel(ThreadHandle thread, CcuKernelHandle kernel,
    const std::array<uint64_t, N> &taskArgs)
{
    const CcuResult launchRet = HcommCcuKernelLaunch(
        thread, kernel, taskArgs.data(), static_cast<uint32_t>(N));
    CHK_PRT_RET(launchRet != CCU_SUCCESS,
        HCCL_ERROR("[LaunchKernel] CCU kernel launch failed, ccuRet[%d]", launchRet),
        ConvertCcuToHccl(launchRet));
    return HCCL_SUCCESS;
}

HcclResult LaunchKernelWithoutArgs(ThreadHandle thread, CcuKernelHandle kernel)
{
    const CcuResult launchRet = HcommCcuKernelLaunch(thread, kernel, nullptr, 0);
    CHK_PRT_RET(launchRet != CCU_SUCCESS,
        HCCL_ERROR("[LaunchKernelWithoutArgs] CCU kernel launch failed, ccuRet[%d]", launchRet),
        ConvertCcuToHccl(launchRet));
    return HCCL_SUCCESS;
}

uint64_t AlignSliceBytes(uint64_t sliceBytes)
{
    if (sliceBytes >= SLICE_ALIGNMENT) {
        return sliceBytes - sliceBytes % SLICE_ALIGNMENT;
    }
    return sliceBytes - sliceBytes % DATA_TYPE_SIZE;
}

uint64_t AlignUpSliceBytes(uint64_t sliceBytes)
{
    if (sliceBytes <= DATA_TYPE_SIZE) {
        return DATA_TYPE_SIZE;
    }
    if (sliceBytes < SLICE_ALIGNMENT) {
        return (sliceBytes + DATA_TYPE_SIZE - 1) / DATA_TYPE_SIZE * DATA_TYPE_SIZE;
    }
    return (sliceBytes + SLICE_ALIGNMENT - 1) / SLICE_ALIGNMENT * SLICE_ALIGNMENT;
}

template <size_t N>
HcclResult LaunchTwoDieKernels(const OpParam &param, const AlgResourceCtx &resCtx,
    const CcuKernelHandle *kernels, uint32_t kernelCount, uint32_t phase,
    const std::array<uint64_t, N> &taskArgs)
{
    CHK_PRT_RET(phase >= NOTIFY_PHASE_COUNT || resCtx.channelGroupCount != 2 ||
            resCtx.subThreadCount != 1 ||
            kernelCount != resCtx.channelGroupCount * NOTIFY_PHASE_COUNT,
        HCCL_ERROR("[LaunchTwoDieKernels] invalid multi-die resource"), HCCL_E_INTERNAL);

    const ThreadHandle subThread = resCtx.subThreads[0];
    CHK_RET(HcommThreadNotifyRecordOnThread(param.cpuThread, subThread, 0));
    CHK_RET(HcommThreadNotifyWaitOnThread(subThread, 0, CUSTOM_TIMEOUT));
    const uint32_t phaseOffset = phase * resCtx.channelGroupCount;
    CHK_RET(LaunchKernel(subThread, kernels[phaseOffset + 1], taskArgs));
    CHK_RET(HcommThreadNotifyRecordOnThread(subThread, param.cpuThread, 0));

    CHK_RET(LaunchKernel(param.cpuThread, kernels[phaseOffset], taskArgs));
    CHK_RET(HcommThreadNotifyWaitOnThread(param.cpuThread, 0, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult LaunchReleases(const OpParam &param, const AlgResourceCtx &resCtx)
{
    CHK_PRT_RET(resCtx.channelGroupCount != 2 || resCtx.subThreadCount != 1 ||
            resCtx.releaseKernelCount != resCtx.channelGroupCount,
        HCCL_ERROR("[LaunchReleases] invalid multi-die resource"), HCCL_E_INTERNAL);

    const ThreadHandle subThread = resCtx.subThreads[0];
    CHK_RET(HcommThreadNotifyRecordOnThread(param.cpuThread, subThread, 0));
    CHK_RET(HcommThreadNotifyWaitOnThread(subThread, 0, CUSTOM_TIMEOUT));
    CHK_RET(LaunchKernelWithoutArgs(subThread, resCtx.releaseKernels[1]));
    CHK_RET(HcommThreadNotifyRecordOnThread(subThread, param.cpuThread, 0));

    CHK_RET(LaunchKernelWithoutArgs(param.cpuThread, resCtx.releaseKernels[0]));
    CHK_RET(HcommThreadNotifyWaitOnThread(param.cpuThread, 0, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult ValidateAlgorithmResource(const OpParam &param, const AlgResourceCtx &resCtx,
    bool &useFused)
{
    CHK_PRT_RET(resCtx.rankId != param.myRank || resCtx.rankSize != param.rankSize,
        HCCL_ERROR("[ValidateAlgorithmResource] cached rank information does not match"),
        HCCL_E_INTERNAL);
    useFused = param.rankSize == 4 && resCtx.channelGroupCount == 1;
    if (useFused) {
        CHK_PRT_RET(resCtx.channelGroupCount != 1 || resCtx.subThreadCount != 0 ||
                resCtx.pullKernelCount != 0 || resCtx.releaseKernelCount != 0 ||
                resCtx.channelGroupSizes[0] != 3 ||
                resCtx.stripeKernelCount != NOTIFY_PHASE_COUNT ||
                resCtx.fusedKernel == 0 || resCtx.reduceKernel != 0 ||
                resCtx.stripeReduceKernel != 0,
            HCCL_ERROR("[ValidateAlgorithmResource] invalid fused resource"), HCCL_E_INTERNAL);
        return HCCL_SUCCESS;
    }

    const bool hasTwoGroups = resCtx.channelGroupCount == 2;
    const uint32_t smallerGroup = hasTwoGroups ?
        std::min(resCtx.channelGroupSizes[0], resCtx.channelGroupSizes[1]) : 0;
    const uint32_t largerGroup = hasTwoGroups ?
        std::max(resCtx.channelGroupSizes[0], resCtx.channelGroupSizes[1]) : 0;
    const bool validRank16Groups = param.rankSize == MAX_RANK_SIZE &&
        smallerGroup == 7 && largerGroup == 8;
    const bool validRank4Groups = param.rankSize == 4 &&
        smallerGroup == 3 && largerGroup == 3;
    const bool validRank12Groups = param.rankSize == 12 &&
        ((smallerGroup == 3 && largerGroup == 8) ||
            (smallerGroup == 4 && largerGroup == 7));
    const bool validSplitKernels = validRank4Groups ?
        (resCtx.pullKernelCount == 0 && resCtx.reduceKernel == 0 &&
            resCtx.stripeReduceKernel == 0) :
        (resCtx.pullKernelCount == resCtx.channelGroupCount * NOTIFY_PHASE_COUNT &&
            resCtx.reduceKernel != 0 && resCtx.stripeReduceKernel != 0);
    CHK_PRT_RET((param.rankSize != 4 && param.rankSize != 12 &&
            param.rankSize != MAX_RANK_SIZE) ||
            resCtx.channelGroupCount != 2 || resCtx.subThreadCount != 1 ||
            (!validRank4Groups && !validRank12Groups && !validRank16Groups) ||
            !validSplitKernels ||
            resCtx.stripeKernelCount != resCtx.channelGroupCount * NOTIFY_PHASE_COUNT ||
            resCtx.releaseKernelCount != resCtx.channelGroupCount || resCtx.fusedKernel != 0 ||
            (validRank4Groups && (resCtx.reduceKernel != 0 || resCtx.stripeReduceKernel != 0)),
        HCCL_ERROR("[ValidateAlgorithmResource] invalid split resource"), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

uint64_t GetStripeBytes(uint64_t sliceBytes, uint32_t channelCount)
{
    if (channelCount == 0) {
        return 0;
    }
    return AlignSliceBytes(sliceBytes / channelCount);
}

uint64_t GetTailStripeBytes(uint64_t sliceBytes, uint64_t stripeBytes, uint32_t channelCount)
{
    if (channelCount == 0) {
        return 0;
    }
    return sliceBytes - stripeBytes * (channelCount - 1);
}

uint64_t GetStripeSliceBytes(uint64_t remainingBytes, uint64_t maxSliceBytes,
    uint64_t minStripeSliceBytes)
{
    if (remainingBytes <= maxSliceBytes) {
        return remainingBytes;
    }
    const uint64_t tailBytes = remainingBytes - maxSliceBytes;
    if (tailBytes >= minStripeSliceBytes) {
        return maxSliceBytes;
    }
    return maxSliceBytes - (minStripeSliceBytes - tailBytes);
}

GroupFlatBatchPlan GetGroupFlatBatchPlan(uint64_t remainingBytes, uint64_t bufferBytes,
    uint32_t rankSize, uint32_t maxReducePieces)
{
    GroupFlatBatchPlan plan;
    if (remainingBytes == 0 || maxReducePieces == 0) {
        return plan;
    }

    const uint64_t reduceLimitedTileBytes =
        AlignSliceBytes(static_cast<uint64_t>(MAX_DATA_SIZE) / maxReducePieces);
    const uint64_t preferredScratchPieces =
        static_cast<uint64_t>(rankSize) + GROUP_FLAT_PREFERRED_BATCH_TILES;
    const uint64_t preferredTileBytes = AlignSliceBytes(
        std::min(reduceLimitedTileBytes, bufferBytes / preferredScratchPieces));
    if (preferredTileBytes < DATA_TYPE_SIZE) {
        return plan;
    }

    // Widen a batch only when the extra retention slots do not shrink the transfer tile.
    const uint64_t residentPieces = bufferBytes / preferredTileBytes;
    const uint64_t residentTiles = residentPieces > rankSize ? residentPieces - rankSize : 0;
    const uint32_t batchTileLimit = static_cast<uint32_t>(std::min<uint64_t>(
        GROUP_FLAT_MAX_BATCH_TILES,
        std::max<uint64_t>(GROUP_FLAT_PREFERRED_BATCH_TILES, residentTiles)));
    const uint64_t batchCapacityBytes = preferredTileBytes * batchTileLimit;
    if (remainingBytes > batchCapacityBytes) {
        plan.tileBytes = preferredTileBytes;
        plan.batchBytes = batchCapacityBytes;
        plan.tileCount = batchTileLimit;
        return plan;
    }

    for (uint32_t reservedTiles = 1; reservedTiles <= batchTileLimit;
         ++reservedTiles) {
        const uint64_t scratchPieces = static_cast<uint64_t>(rankSize) + reservedTiles;
        const uint64_t capacityTileBytes = AlignSliceBytes(
            std::min(reduceLimitedTileBytes, bufferBytes / scratchPieces));
        if (capacityTileBytes < DATA_TYPE_SIZE) {
            continue;
        }

        const uint64_t requiredTiles =
            1 + (remainingBytes - 1) / capacityTileBytes;
        if (requiredTiles > reservedTiles) {
            continue;
        }

        uint64_t balancedTileBytes =
            AlignUpSliceBytes(1 + (remainingBytes - 1) / requiredTiles);
        if (balancedTileBytes > capacityTileBytes) {
            balancedTileBytes = capacityTileBytes;
        }
        plan.tileBytes = balancedTileBytes;
        plan.batchBytes = remainingBytes;
        plan.tileCount = requiredTiles;
        return plan;
    }
    return plan;
}
} // namespace

HcclResult ExecOp(const OpParam &param)
{
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE ||
            param.myRank >= param.rankSize,
        HCCL_ERROR("[ExecOp] invalid rank information"), HCCL_E_PARA);
    CHK_PRT_RET(param.resCtx == nullptr || param.ctxSize != sizeof(AlgResourceCtx),
        HCCL_ERROR("[ExecOp] invalid engine context"), HCCL_E_INTERNAL);

    AlgResourceCtx resCtx;
    std::memcpy(&resCtx, param.resCtx, sizeof(resCtx));

    const uint64_t rankSliceBytes = param.count * DATA_TYPE_SIZE;
    if (rankSliceBytes == 0) {
        return HCCL_SUCCESS;
    }
    const uint64_t inputBytes = rankSliceBytes * param.rankSize;
    const uint64_t inputAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outputAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    CHK_PRT_RET(!IsAddressRangeValid(inputAddr, inputBytes) ||
            !IsAddressRangeValid(outputAddr, rankSliceBytes),
        HCCL_ERROR("[ExecOp] input or output address range overflows"), HCCL_E_PARA);

    if (param.rankSize == 1) {
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(param.cpuThread, param.outputPtr, param.inputPtr, rankSliceBytes)));
        return HCCL_SUCCESS;
    }

    bool useFused = false;
    CHK_RET(ValidateAlgorithmResource(param, resCtx, useFused));
    CHK_PRT_RET(resCtx.localBuffer.addr == nullptr || resCtx.localBuffer.size == 0 ||
            resCtx.minBufferSize == 0,
        HCCL_ERROR("[ExecOp] invalid HCCL buffer resource"), HCCL_E_INTERNAL);

    const uint64_t scratchAddr = reinterpret_cast<uint64_t>(resCtx.localBuffer.addr);
    CHK_PRT_RET(!IsAddressRangeValid(scratchAddr, resCtx.localBuffer.size),
        HCCL_ERROR("[ExecOp] scratch address range overflows"), HCCL_E_PARA);
    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    const uint64_t scratchToken = resCtx.scratchToken;
    CHK_RET(GetToken(inputAddr, inputBytes, inputToken));
    CHK_RET(GetToken(outputAddr, rankSliceBytes, outputToken));

    const bool isLargeMessage = rankSliceBytes > STRIPE_MESSAGE_THRESHOLD;
    const bool useStripe = isLargeMessage && param.rankSize == 4;
    const bool useDirectStripe = isLargeMessage &&
        (param.rankSize == 12 || param.rankSize == MAX_RANK_SIZE);
    const bool useGroupFlat = isLargeMessage && !useFused && !useStripe &&
        !useDirectStripe;
    uint64_t maxSliceBytes = 0;
    uint32_t groupFlatMaxReducePieces = 0;
    if (useDirectStripe) {
        maxSliceBytes = std::min(2 * static_cast<uint64_t>(MAX_DATA_SIZE),
            resCtx.minBufferSize);
    } else if (useStripe) {
        maxSliceBytes = std::min(static_cast<uint64_t>(MAX_DATA_SIZE), resCtx.minBufferSize);
    } else if (useGroupFlat) {
        // Each die reduces only its own group; the first tree level is the largest LocalReduce.
        const uint32_t primaryPieces = resCtx.channelGroupSizes[0] + 1;
        const uint32_t secondaryPieces = resCtx.channelGroupSizes[1];
        groupFlatMaxReducePieces = std::max(primaryPieces / 2, secondaryPieces / 2);
    } else {
        const uint64_t maxReducePieces = param.rankSize / 2;
        maxSliceBytes = std::min({static_cast<uint64_t>(MAX_DATA_SIZE),
            resCtx.minBufferSize / param.rankSize,
            static_cast<uint64_t>(MAX_DATA_SIZE) / maxReducePieces});
    }
    if (!useGroupFlat) {
        maxSliceBytes = AlignSliceBytes(maxSliceBytes);
        CHK_PRT_RET(maxSliceBytes < DATA_TYPE_SIZE,
            HCCL_ERROR("[ExecOp] HCCL buffer is too small for rankSize[%u]", param.rankSize),
            HCCL_E_PARA);
    }

    uint64_t minStripeSliceBytes = 0;
    if (useStripe || useDirectStripe) {
        const uint32_t maxChannelCount = useFused ? resCtx.channelGroupSizes[0] :
            std::max(resCtx.channelGroupSizes[0], resCtx.channelGroupSizes[1]);
        const uint32_t stripeGroupCount =
            (useFused || useDirectStripe) ? 1 : resCtx.channelGroupCount;
        minStripeSliceBytes = static_cast<uint64_t>(maxChannelCount) * stripeGroupCount *
            SLICE_ALIGNMENT;
        CHK_PRT_RET(maxSliceBytes < minStripeSliceBytes * 2,
            HCCL_ERROR("[ExecOp] HCCL buffer is too small for striped transfer"),
            HCCL_E_PARA);
    }

    uint64_t processedBytes = 0;
    uint32_t phase = 0;
    while (processedBytes < rankSliceBytes) {
        const uint64_t remainingBytes = rankSliceBytes - processedBytes;
        const GroupFlatBatchPlan groupFlatPlan = useGroupFlat ?
            GetGroupFlatBatchPlan(remainingBytes, resCtx.minBufferSize, param.rankSize,
                groupFlatMaxReducePieces) :
            GroupFlatBatchPlan{};
        CHK_PRT_RET(useGroupFlat && groupFlatPlan.tileBytes == 0,
            HCCL_ERROR("[ExecOp] HCCL buffer is too small for grouped flat transfer"),
            HCCL_E_PARA);
        const uint64_t sliceBytes = useGroupFlat ? groupFlatPlan.tileBytes :
            ((useStripe || useDirectStripe) ? GetStripeSliceBytes(
                remainingBytes, maxSliceBytes, minStripeSliceBytes) :
                std::min(maxSliceBytes, remainingBytes));
        uint64_t consumedBytes = sliceBytes;
        if (useStripe || useDirectStripe) {
            const uint32_t primaryChannelCount = resCtx.channelGroupSizes[0];
            const uint64_t primaryGroupBytes = (useFused || useDirectStripe) ? sliceBytes :
                AlignSliceBytes(sliceBytes / resCtx.channelGroupCount);
            const uint64_t primaryStripeBytes =
                GetStripeBytes(primaryGroupBytes, primaryChannelCount);
            const uint64_t primaryTailStripeBytes =
                GetTailStripeBytes(primaryGroupBytes, primaryStripeBytes, primaryChannelCount);
            const uint32_t secondaryChannelCount = useFused ? 0 : resCtx.channelGroupSizes[1];
            const uint64_t secondaryGroupBytes = useFused ? 0 :
                (useDirectStripe ? sliceBytes : sliceBytes - primaryGroupBytes);
            const uint64_t secondaryStripeBytes =
                GetStripeBytes(secondaryGroupBytes, secondaryChannelCount);
            const uint64_t secondaryTailStripeBytes =
                GetTailStripeBytes(secondaryGroupBytes, secondaryStripeBytes, secondaryChannelCount);
            CHK_PRT_RET(primaryStripeBytes == 0 || primaryTailStripeBytes == 0 ||
                    primaryStripeBytes > MAX_DATA_SIZE || primaryTailStripeBytes > MAX_DATA_SIZE ||
                    (!useFused && (secondaryStripeBytes == 0 || secondaryTailStripeBytes == 0 ||
                        secondaryStripeBytes > MAX_DATA_SIZE ||
                        secondaryTailStripeBytes > MAX_DATA_SIZE)),
                HCCL_ERROR("[ExecOp] invalid zero-sized stripe"), HCCL_E_INTERNAL);
            const std::array<uint64_t, 13> stripeArgs = {inputAddr, outputAddr, scratchAddr,
                inputToken, outputToken, scratchToken, rankSliceBytes, processedBytes, sliceBytes,
                primaryStripeBytes, secondaryStripeBytes, primaryTailStripeBytes,
                secondaryTailStripeBytes};
            if (useFused) {
                CHK_RET(LaunchKernel(param.cpuThread, resCtx.stripeKernels[phase], stripeArgs));
            } else {
                CHK_RET(LaunchTwoDieKernels(param, resCtx, resCtx.stripeKernels,
                    resCtx.stripeKernelCount, phase, stripeArgs));
            }
            if (useDirectStripe) {
                const uint64_t firstMergeBytes =
                    std::min(sliceBytes, static_cast<uint64_t>(MAX_DATA_SIZE));
                const uint64_t secondMergeBytes = sliceBytes - firstMergeBytes;
                const std::array<uint64_t, 11> mergeArgs = {inputAddr, outputAddr, scratchAddr,
                    inputToken, outputToken, scratchToken, rankSliceBytes, processedBytes,
                    sliceBytes, firstMergeBytes, secondMergeBytes};
                CHK_RET(LaunchKernel(param.cpuThread, resCtx.stripeReduceKernel, mergeArgs));
            }
            phase ^= 1U;
        } else if (useGroupFlat) {
            const uint64_t lastTileBytes = groupFlatPlan.batchBytes -
                groupFlatPlan.tileBytes * (groupFlatPlan.tileCount - 1);
            const uint64_t loopCounter =
                std::numeric_limits<uint64_t>::max() - groupFlatPlan.tileCount;
            const std::array<uint64_t, 11> groupArgs = {inputAddr, outputAddr, scratchAddr,
                inputToken, outputToken, scratchToken, rankSliceBytes, processedBytes,
                groupFlatPlan.tileBytes, loopCounter, lastTileBytes};
            CHK_RET(LaunchTwoDieKernels(
                param, resCtx, resCtx.stripeKernels, resCtx.stripeKernelCount, phase, groupArgs));
            CHK_RET(LaunchKernel(param.cpuThread, resCtx.stripeReduceKernel, groupArgs));
            phase ^= 1U;
            consumedBytes = groupFlatPlan.batchBytes;
        } else if (useFused) {
            const std::array<uint64_t, 9> fusedArgs = {inputAddr, outputAddr, scratchAddr,
                inputToken, outputToken, scratchToken, rankSliceBytes, processedBytes, sliceBytes};
            CHK_RET(LaunchKernel(param.cpuThread, resCtx.fusedKernel, fusedArgs));
        } else {
            const std::array<uint64_t, 7> pullArgs = {inputAddr, scratchAddr, inputToken,
                scratchToken, rankSliceBytes, processedBytes, sliceBytes};
            const std::array<uint64_t, 9> reduceArgs = {inputAddr, outputAddr, scratchAddr,
                inputToken, outputToken, scratchToken, rankSliceBytes, processedBytes, sliceBytes};
            CHK_RET(LaunchTwoDieKernels(
                param, resCtx, resCtx.pullKernels, resCtx.pullKernelCount, phase, pullArgs));
            CHK_RET(LaunchKernel(param.cpuThread, resCtx.reduceKernel, reduceArgs));
            phase ^= 1U;
        }
        processedBytes += consumedBytes;
    }

    const bool barrierInDataKernel = param.rankSize == 12 || param.rankSize == MAX_RANK_SIZE;
    if (!useFused && !barrierInDataKernel) {
        CHK_RET(LaunchReleases(param, resCtx));
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
