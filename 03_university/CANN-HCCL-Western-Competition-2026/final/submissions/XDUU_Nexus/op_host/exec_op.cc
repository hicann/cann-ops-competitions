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
#include <limits>
#include <vector>

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>
#include <hcomm/hcomm_primitives.h>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {

constexpr uint64_t ROOT_ALGORITHM_MAX_BYTES = 512 * 1024;
constexpr uint64_t RECURSIVE_DOUBLING_BYTES = 512 * 1024;
constexpr uint64_t LARGE_512MB_BYTES = 512ULL * 1024 * 1024;
constexpr uint64_t LARGE_400MB_PLUS_4_BYTES = 400ULL * 1024 * 1024 + 4;
constexpr uint64_t RECURSIVE_DOUBLING_SCRATCH_SLOTS = 5;
constexpr uint64_t ALGORITHM_ROOT = 0;
constexpr uint64_t ALGORITHM_FULL_MESH = 1;
constexpr uint64_t ALGORITHM_RECURSIVE_DOUBLING = 2;
constexpr uint64_t ALGORITHM_SMALL_2D_BEST = 3;
constexpr uint64_t ALGORITHM_NHR_WRITE_REDUCE = 4;
constexpr uint64_t ALGORITHM_PAR_MESH_RS_INPUT = 6;
constexpr uint64_t ALGORITHM_PAR_MESH_RS_OUTPUT = 7;
constexpr uint64_t ALGORITHM_PAR_MESH_AG = 8;
constexpr uint64_t ALGORITHM_PAR_CLOS_RS = 9;
constexpr uint64_t ALGORITHM_PAR_CLOS_AG = 10;
constexpr uint64_t ALGORITHM_PAR_CLOS_RS_OUTPUT = 11;
constexpr uint64_t ALGORITHM_PAR_MESH_RS_AG = 12;
constexpr uint64_t ALGORITHM_PAR_CLOS_RS_AG = 13;
constexpr uint64_t ALGORITHM_MIXED_RADIX_12 = 20;
constexpr uint64_t BASELINE_KERNEL_INDEX = 0;
constexpr uint64_t MESH_SCRATCH_KERNEL_INDEX = 1;
constexpr uint64_t PARALLEL_MESH_KERNEL_INDEX = 1;
constexpr uint64_t PARALLEL_CLOS_KERNEL_INDEX = 2;
constexpr uint32_t PARALLEL_SYNC_NOTIFY = 0;

enum class DispatchPath : uint32_t {
    GENERIC,
    RANK16_512KB,
    RANK16_512MB,
    RANK16_400MB_PLUS_4,
    RANK4_512KB,
    RANK4_512MB,
    RANK4_400MB_PLUS_4,
    RANK12_512KB,
    RANK12_512MB,
    RANK12_400MB_PLUS_4,
};

DispatchPath SelectDispatchPath(uint32_t rankSize, uint64_t dataSize)
{
    if (rankSize == 16) {
        if (dataSize == RECURSIVE_DOUBLING_BYTES) {
            return DispatchPath::RANK16_512KB;
        }
        if (dataSize == LARGE_512MB_BYTES) {
            return DispatchPath::RANK16_512MB;
        }
        if (dataSize == LARGE_400MB_PLUS_4_BYTES) {
            return DispatchPath::RANK16_400MB_PLUS_4;
        }
    } else if (rankSize == 4) {
        if (dataSize == RECURSIVE_DOUBLING_BYTES) {
            return DispatchPath::RANK4_512KB;
        }
        if (dataSize == LARGE_512MB_BYTES) {
            return DispatchPath::RANK4_512MB;
        }
        if (dataSize == LARGE_400MB_PLUS_4_BYTES) {
            return DispatchPath::RANK4_400MB_PLUS_4;
        }
    } else if (rankSize == 12) {
        if (dataSize == RECURSIVE_DOUBLING_BYTES) {
            return DispatchPath::RANK12_512KB;
        }
        if (dataSize == LARGE_512MB_BYTES) {
            return DispatchPath::RANK12_512MB;
        }
        if (dataSize == LARGE_400MB_PLUS_4_BYTES) {
            return DispatchPath::RANK12_400MB_PLUS_4;
        }
    }
    return DispatchPath::GENERIC;
}

bool IsRecursiveDoublingPath(DispatchPath path)
{
    return path == DispatchPath::RANK16_512KB ||
        path == DispatchPath::RANK4_512KB ||
        path == DispatchPath::RANK12_512KB;
}

bool IsRank16ScratchPath(DispatchPath path)
{
    return path == DispatchPath::RANK16_512MB ||
        path == DispatchPath::RANK16_400MB_PLUS_4;
}

bool IsRank12ScratchPath(DispatchPath path)
{
    return path == DispatchPath::RANK12_512MB ||
        path == DispatchPath::RANK12_400MB_PLUS_4;
}

bool IsRank4ScratchPath(DispatchPath path)
{
    return path == DispatchPath::RANK4_512MB ||
        path == DispatchPath::RANK4_400MB_PLUS_4;
}

HcclResult ConvertCcuResult(CcuResult result)
{
    switch (result) {
        case CCU_SUCCESS:
            return HCCL_SUCCESS;
        case CCU_E_PARA:
            return HCCL_E_PARA;
        case CCU_E_PTR:
            return HCCL_E_PTR;
        case CCU_E_NOT_SUPPORT:
            return HCCL_E_NOT_SUPPORT;
        case CCU_E_NOT_FOUND:
            return HCCL_E_NOT_FOUND;
        case CCU_E_UNAVAIL:
            return HCCL_E_UNAVAIL;
        default:
            return HCCL_E_INTERNAL;
    }
}

HcclResult LaunchSliceOnThread(const AlgResourceCtx &resCtx, uint64_t inputAddr,
    uint64_t outputAddr, uint64_t token, uint64_t sliceSize, bool isInPlace,
    uint64_t ownerOffset, uint64_t ownerSize, uint64_t algorithmType,
    uint64_t scratchAddr, uint64_t scratchToken, uint64_t kernelIndex,
    uint64_t threadIndex)
{
    CHK_PRT_RET(kernelIndex >= resCtx.ccuKernels.size(),
        HCCL_ERROR("[AllReduce] CCU kernel index %llu is unavailable, kernel count %zu",
            static_cast<unsigned long long>(kernelIndex), resCtx.ccuKernels.size()),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(threadIndex >= resCtx.threads.size(),
        HCCL_ERROR("[AllReduce] CCU thread index %llu is unavailable, thread count %zu",
            static_cast<unsigned long long>(threadIndex), resCtx.threads.size()),
        HCCL_E_INTERNAL);
    std::vector<uint64_t> taskArgs = {
        inputAddr,
        outputAddr,
        token,
        sliceSize,
        static_cast<uint64_t>(isInPlace),
        ownerOffset,
        ownerSize,
        algorithmType,
        scratchAddr,
        scratchToken,
    };
    // NHR XOR recursive-halving sends 8/4/2 owner blocks in its first three
    // rounds. Pass those transfer sizes directly so the VM checker does not
    // need to infer a multiplied ccu::Variable through the kernel graph.
    if (algorithmType == ALGORITHM_NHR_WRITE_REDUCE) {
        taskArgs.push_back(ownerSize * 8);
        taskArgs.push_back(ownerSize * 4);
        taskArgs.push_back(ownerSize * 2);
    }
    CcuResult result = HcommCcuKernelLaunch(resCtx.threads[threadIndex], resCtx.ccuKernels[kernelIndex],
        taskArgs.data(), taskArgs.size());
    if (result != CCU_SUCCESS) {
        HCCL_ERROR("[AllReduce] CCU kernel launch failed: %d", result);
        return ConvertCcuResult(result);
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchSlice(const AlgResourceCtx &resCtx, uint64_t inputAddr,
    uint64_t outputAddr, uint64_t token, uint64_t sliceSize, bool isInPlace,
    uint64_t ownerOffset, uint64_t ownerSize, uint64_t algorithmType,
    uint64_t scratchAddr, uint64_t scratchToken, uint64_t kernelIndex)
{
    return LaunchSliceOnThread(resCtx, inputAddr, outputAddr, token, sliceSize,
        isInPlace, ownerOffset, ownerSize, algorithmType, scratchAddr,
        scratchToken, kernelIndex, 0);
}

HcclResult QueueParallelStageBarrier(const AlgResourceCtx &resCtx)
{
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        resCtx.threads[0], resCtx.threads[1], PARALLEL_SYNC_NOTIFY)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        resCtx.threads[1], resCtx.threads[0], PARALLEL_SYNC_NOTIFY)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        resCtx.threads[0], PARALLEL_SYNC_NOTIFY, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        resCtx.threads[1], PARALLEL_SYNC_NOTIFY, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult ExecuteParallel2x8(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t inputBase, uint64_t outputBase, uint64_t token,
    uint64_t dataSize, uint64_t dataTypeSize)
{
    CHK_PRT_RET(resCtx.threads.size() < 2 ||
        resCtx.ccuKernels.size() <= PARALLEL_CLOS_KERNEL_INDEX ||
        resCtx.hierLocalSize != 8 || resCtx.hierLaneCount != 8 ||
        resCtx.hierLocalRank >= 8 || resCtx.hierServerId >= 2,
        HCCL_ERROR("[AllReduce] invalid 2x8 parallel resources"),
        HCCL_E_INTERNAL);

    const bool isInPlace = param.inputPtr == param.outputPtr;
    const uint64_t scratchBase =
        reinterpret_cast<uint64_t>(resCtx.localBuffer.addr);

    // CheckerV3 requires every slave stream to begin with a local WAIT.
    // Establish the official PreSyncInterThreads shape before queuing the
    // first Clos CCU graph.  The main-thread record is ordered before its
    // first Mesh graph, and the slave cannot enter Clos until it consumes it.
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        resCtx.threads[0], resCtx.threads[1], PARALLEL_SYNC_NOTIFY)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        resCtx.threads[1], PARALLEL_SYNC_NOTIFY, CUSTOM_TIMEOUT)));

    uint64_t processed = 0;
    while (processed < dataSize) {
        uint64_t roundSize =
            std::min<uint64_t>(MAX_DATA_SIZE, dataSize - processed);
        const uint64_t roundCount = roundSize / dataTypeSize;
        const uint64_t alignedCount = roundCount - roundCount % 16U;
        const uint64_t alignedBytes = alignedCount * dataTypeSize;
        if (alignedBytes <= ROOT_ALGORITHM_MAX_BYTES) {
            CHK_RET(LaunchSlice(resCtx, inputBase + processed,
                outputBase + processed, token, roundSize, isInPlace,
                0, roundSize, ALGORITHM_ROOT, 0, 0,
                BASELINE_KERNEL_INDEX));
            processed += roundSize;
            continue;
        }
        roundSize = alignedBytes;

        // Initial AICPU measurements and the official parallel split model
        // both favor a larger Clos-first partition on this fixed 2x8 shape.
        const uint64_t unit = roundSize / 8U;
        const uint64_t meshFirstBytes = unit * 3U;
        const uint64_t closFirstBytes = roundSize - meshFirstBytes;
        const uint64_t aMeshOwner = meshFirstBytes / 8U;
        const uint64_t aClosOwner = aMeshOwner / 2U;
        const uint64_t bClosOwner = closFirstBytes / 2U;
        const uint64_t bMeshOwner = bClosOwner / 8U;
        const uint64_t scratchSize =
            std::max<uint64_t>(meshFirstBytes, bClosOwner);
        CHK_PRT_RET(resCtx.localBuffer.addr == nullptr ||
            resCtx.localBuffer.size < scratchSize,
            HCCL_ERROR("[AllReduce] 2x8 parallel scratch is too small: need %llu, have %llu",
                static_cast<unsigned long long>(scratchSize),
                static_cast<unsigned long long>(resCtx.localBuffer.size)),
            HCCL_E_MEMORY);
        uint64_t scratchToken = 0;
        CcuResult tokenRet =
            HcommCcuGetMemToken(scratchBase, scratchSize, &scratchToken);
        if (tokenRet != CCU_SUCCESS) {
            return ConvertCcuResult(tokenRet);
        }

        const uint64_t aInput = inputBase + processed;
        const uint64_t aOutput = outputBase + processed;
        const uint64_t bInput = aInput + meshFirstBytes;
        const uint64_t bOutput = aOutput + meshFirstBytes;
        const uint64_t aLaneOffset =
            resCtx.hierLocalRank * aMeshOwner;
        const uint64_t bServerOffset =
            resCtx.hierServerId * bClosOwner;
        const uint64_t bLaneOffset =
            resCtx.hierLocalRank * bMeshOwner;

        // Stage 1: Mesh-first A and Clos-first B occupy different IO dies.
        CHK_RET(LaunchSliceOnThread(resCtx, aInput, aOutput, token,
            meshFirstBytes, isInPlace, aLaneOffset, aMeshOwner,
            ALGORITHM_PAR_MESH_RS_INPUT, scratchBase, scratchToken,
            PARALLEL_MESH_KERNEL_INDEX, 0));
        CHK_RET(LaunchSliceOnThread(resCtx, bInput, bOutput, token,
            closFirstBytes, isInPlace, bServerOffset, bClosOwner,
            ALGORITHM_PAR_CLOS_RS, 0, 0,
            PARALLEL_CLOS_KERNEL_INDEX, 1));
        CHK_RET(QueueParallelStageBarrier(resCtx));

        // Stages 2+3: after the first cross-domain barrier, each partition
        // remains on the same execution domain for RS followed by AG. Fuse
        // both operations into one CCU graph per thread, avoiding an extra
        // launch plus a redundant kernel PreSync/PostSync pair.
        CHK_RET(LaunchSliceOnThread(resCtx,
            aInput + aLaneOffset, aOutput + aLaneOffset, token,
            aMeshOwner, isInPlace,
            resCtx.hierServerId * aClosOwner, aClosOwner,
            ALGORITHM_PAR_CLOS_RS_AG, 0, 0,
            PARALLEL_CLOS_KERNEL_INDEX, 1));
        CHK_RET(LaunchSliceOnThread(resCtx,
            bInput + bServerOffset, bOutput + bServerOffset, token,
            bClosOwner, isInPlace, bLaneOffset, bMeshOwner,
            ALGORITHM_PAR_MESH_RS_AG, scratchBase, scratchToken,
            PARALLEL_MESH_KERNEL_INDEX, 0));
        CHK_RET(QueueParallelStageBarrier(resCtx));

        // Stage 4: both partitions become complete on every rank.
        CHK_RET(LaunchSliceOnThread(resCtx, aInput, aOutput, token,
            meshFirstBytes, isInPlace, aLaneOffset, aMeshOwner,
            ALGORITHM_PAR_MESH_AG, scratchBase, scratchToken,
            PARALLEL_MESH_KERNEL_INDEX, 0));
        CHK_RET(LaunchSliceOnThread(resCtx, bInput, bOutput, token,
            closFirstBytes, isInPlace, bServerOffset, bClosOwner,
            ALGORITHM_PAR_CLOS_AG, 0, 0,
            PARALLEL_CLOS_KERNEL_INDEX, 1));
        // Stage 4 is the final operation for each partition on its respective
        // thread. The function-tail slave RECORD/main WAIT below joins both
        // queues before return, so a second bidirectional barrier is redundant.
        processed += roundSize;
    }

    // CheckerV3 also requires the slave stream to end with a local RECORD.
    // Close the auxiliary queue with the official PostSyncInterThreads shape:
    // slave records completion and the main stream consumes it.
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        resCtx.threads[1], resCtx.threads[0], PARALLEL_SYNC_NOTIFY)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        resCtx.threads[0], PARALLEL_SYNC_NOTIFY, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

} // namespace

HcclResult ExecOp(const OpParam &param)
{
    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> serialized(ctx, ctx + param.ctxSize);
    AlgResourceCtx resCtx;
    resCtx.DeSerialize(serialized);

    CHK_PRT_RET(resCtx.threads.empty(),
        HCCL_ERROR("[AllReduce] no CCU thread in resource context"), HCCL_E_INTERNAL);

    if (param.count == 0) {
        return HCCL_SUCCESS;
    }

    auto sizeIt = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(sizeIt == SIZE_TABLE.end(),
        HCCL_ERROR("[AllReduce] unsupported data type %d", param.dataType), HCCL_E_NOT_SUPPORT);
    const uint64_t dataTypeSize = sizeIt->second;
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / dataTypeSize,
        HCCL_ERROR("[AllReduce] data size overflow"), HCCL_E_PARA);
    const uint64_t dataSize = param.count * dataTypeSize;

    if (param.rankSize == 1) {
        return static_cast<HcclResult>(HcommLocalCopyOnThread(
            resCtx.threads[0], param.outputPtr, param.inputPtr, dataSize));
    }
    CHK_PRT_RET(resCtx.ccuKernels.empty(),
        HCCL_ERROR("[AllReduce] no registered CCU kernel"), HCCL_E_INTERNAL);

    uint64_t token = 0;
    CcuResult tokenRet = HcommCcuGetMemToken(
        reinterpret_cast<uint64_t>(param.inputPtr), dataSize, &token);
    if (tokenRet != CCU_SUCCESS) {
        HCCL_ERROR("[AllReduce] failed to get memory token: %d", tokenRet);
        return ConvertCcuResult(tokenRet);
    }

    const uint64_t inputBase = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outputBase = reinterpret_cast<uint64_t>(param.outputPtr);
    const DispatchPath dispatchPath = SelectDispatchPath(param.rankSize, dataSize);
    const bool useRecursiveDoubling = IsRecursiveDoublingPath(dispatchPath) ||
        (dispatchPath == DispatchPath::GENERIC &&
            dataSize == RECURSIVE_DOUBLING_BYTES &&
            (param.rankSize == 4 || param.rankSize == 12 || param.rankSize == 16));
    const bool useSmall2DBest =
        dataSize == RECURSIVE_DOUBLING_BYTES &&
        param.rankSize == 12;
    const bool useRank16ScratchPath = IsRank16ScratchPath(dispatchPath) ||
        (dispatchPath == DispatchPath::GENERIC && param.rankSize == 16 &&
            dataSize > ROOT_ALGORITHM_MAX_BYTES);
    const bool useRank12ScratchPath = IsRank12ScratchPath(dispatchPath);
    const bool useRank4ScratchPath = IsRank4ScratchPath(dispatchPath);
    const bool useLargeScratchPath = useRank16ScratchPath || useRank12ScratchPath || useRank4ScratchPath;
    if (resCtx.useParallel2x8 != 0 && useRank16ScratchPath) {
        return ExecuteParallel2x8(param, resCtx, inputBase, outputBase,
            token, dataSize, dataTypeSize);
    }
    uint64_t scratchAddr = 0;
    uint64_t scratchToken = 0;
    bool scratchTokenReady = false;
    if (useRecursiveDoubling || useSmall2DBest) {
        const uint64_t scratchSize = useSmall2DBest ?
            4U * dataSize :
            RECURSIVE_DOUBLING_SCRATCH_SLOTS * dataSize;
        CHK_PRT_RET(resCtx.localBuffer.addr == nullptr || resCtx.localBuffer.size < scratchSize,
            HCCL_ERROR("[AllReduce] small-message scratch is too small: need %llu, have %llu",
                static_cast<unsigned long long>(scratchSize),
                static_cast<unsigned long long>(resCtx.localBuffer.size)),
            HCCL_E_MEMORY);
        scratchAddr = reinterpret_cast<uint64_t>(resCtx.localBuffer.addr);
        CcuResult scratchTokenRet = HcommCcuGetMemToken(scratchAddr, scratchSize, &scratchToken);
        if (scratchTokenRet != CCU_SUCCESS) {
            HCCL_ERROR("[AllReduce] failed to get small-message scratch token: %d", scratchTokenRet);
            return ConvertCcuResult(scratchTokenRet);
        }
        scratchTokenReady = true;
    }
    uint64_t processed = 0;
    while (processed < dataSize) {
        uint64_t sliceSize = std::min<uint64_t>(MAX_DATA_SIZE, dataSize - processed);

        // Scratch slots must have equal owner sizes. Keep each specialized
        // large-message slice aligned to rankSize elements; any tiny remainder
        // (including the +4 B case) is launched through the proven Root path.
        if (useLargeScratchPath &&
            sliceSize > ROOT_ALGORITHM_MAX_BYTES) {
            const uint64_t candidateCount = sliceSize / dataTypeSize;
            const uint64_t alignedCount = candidateCount - candidateCount % param.rankSize;
            if (alignedCount > 0) {
                sliceSize = alignedCount * dataTypeSize;
            }
        }
        // 16-rank large-message body slices use NHR WriteReduce (B-21 path).
        // 4 B tail falls through to ALGORITHM_ROOT for the baseline kernel.
        const bool useNhrSlice = useRank16ScratchPath &&
            sliceSize > ROOT_ALGORITHM_MAX_BYTES;
        const bool useMixedRadix12Slice = useRank12ScratchPath &&
            sliceSize > ROOT_ALGORITHM_MAX_BYTES;
        const uint64_t algorithmType = useSmall2DBest ? ALGORITHM_SMALL_2D_BEST :
            (useRecursiveDoubling ? ALGORITHM_RECURSIVE_DOUBLING :
            (useNhrSlice ? ALGORITHM_NHR_WRITE_REDUCE :
            (useMixedRadix12Slice ? ALGORITHM_MIXED_RADIX_12 :
            (sliceSize > ROOT_ALGORITHM_MAX_BYTES ? ALGORITHM_FULL_MESH : ALGORITHM_ROOT))));

        // Split each host slice by element rather than by byte.  The balanced
        // prefix distribution also covers the 400 MiB + 4 B test without
        // dropping or overlapping the final float.
        const uint64_t sliceCount = sliceSize / dataTypeSize;
        const uint64_t baseCount = sliceCount / param.rankSize;
        const uint64_t extraCount = sliceCount % param.rankSize;
        const uint64_t ownerCount = baseCount + (param.myRank < extraCount ? 1 : 0);
        const uint64_t ownerOffsetCount = param.myRank * baseCount +
            std::min<uint64_t>(param.myRank, extraCount);
        const uint64_t ownerOffset = ownerOffsetCount * dataTypeSize;
        const uint64_t ownerSize = ownerCount * dataTypeSize;

        const bool useMeshScratchKernel = useLargeScratchPath &&
            (algorithmType == ALGORITHM_FULL_MESH ||
                algorithmType == ALGORITHM_MIXED_RADIX_12);
        if (useMeshScratchKernel && !scratchTokenReady) {
            const uint64_t scratchSize =
                algorithmType == ALGORITHM_MIXED_RADIX_12 ?
                ownerSize * 2U : ownerSize * param.rankSize;
            CHK_PRT_RET(resCtx.localBuffer.addr == nullptr || resCtx.localBuffer.size < scratchSize,
                HCCL_ERROR("[AllReduce] mesh scratch is too small: need %llu, have %llu",
                    static_cast<unsigned long long>(scratchSize),
                    static_cast<unsigned long long>(resCtx.localBuffer.size)),
                HCCL_E_MEMORY);
            scratchAddr = reinterpret_cast<uint64_t>(resCtx.localBuffer.addr);
            CcuResult scratchTokenRet = HcommCcuGetMemToken(scratchAddr, scratchSize, &scratchToken);
            if (scratchTokenRet != CCU_SUCCESS) {
                HCCL_ERROR("[AllReduce] failed to get mesh scratch token: %d", scratchTokenRet);
                return ConvertCcuResult(scratchTokenRet);
            }
            scratchTokenReady = true;
        }
        const bool useNhrWriteReduce = useNhrSlice &&
            algorithmType == ALGORITHM_NHR_WRITE_REDUCE;
        const uint64_t kernelIndex = (useMeshScratchKernel || useNhrWriteReduce) ?
            MESH_SCRATCH_KERNEL_INDEX : BASELINE_KERNEL_INDEX;

        CHK_RET(LaunchSlice(resCtx, inputBase + processed, outputBase + processed,
            token, sliceSize, param.inputPtr == param.outputPtr,
            ownerOffset, ownerSize, algorithmType, scratchAddr, scratchToken, kernelIndex));
        processed += sliceSize;
    }
    return HCCL_SUCCESS;
}

} // namespace ops_hccl