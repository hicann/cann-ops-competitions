/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <ccu/ccu_res.h>
#include <ccu/ccu_launch.h>

#include <algorithm>
#include <array>
#include <limits>

#include "custom.h"
#include "exec_op.h"

namespace ops_hccl {
namespace {
struct CollectiveLayout {
    uint64_t baseSliceSize = 0;
    uint64_t tailSize = 0;
    bool rootParticipates = true;
};

struct CollectiveSlice {
    uint64_t addr = 0;
    uint64_t size = 0;
};

struct CollectiveLane {
    uint64_t baseAddr = 0;
    CollectiveLayout layout{};
    CollectiveSlice localSlice{};
};

bool RootParticipates(uint32_t rankSize)
{
    return rankSize == 4 || rankSize == 12 || rankSize == 16;
}

CollectiveLayout GetCollectiveLayout(const OpParam &param, uint64_t elementCount,
    uint64_t dataTypeSize, bool rootParticipates)
{
    if (param.rankSize <= 1) {
        return CollectiveLayout{};
    }
    const uint32_t workerCount = rootParticipates ? param.rankSize : param.rankSize - 1;
    const uint64_t baseCount = elementCount / workerCount;
    const uint64_t remainder = elementCount % workerCount;
    return CollectiveLayout{baseCount * dataTypeSize, remainder * dataTypeSize, rootParticipates};
}

CollectiveSlice GetCollectiveSlice(const OpParam &param, const CollectiveLayout &layout,
    uint64_t baseAddr, uint32_t rank);

CollectiveLane GetCollectiveLane(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t baseAddr, uint64_t dataTypeSize, uint32_t kernelIdx)
{
    uint64_t laneBase = baseAddr;
    uint64_t laneCount = param.count;
    if (resCtx.rank4DualLane) {
        const uint64_t firstLaneCount = param.count / 2;
        if (kernelIdx == 0) {
            laneCount = firstLaneCount;
        } else {
            laneBase += firstLaneCount * dataTypeSize;
            laneCount -= firstLaneCount;
        }
    }
    const bool rootParticipates = RootParticipates(param.rankSize);
    const CollectiveLayout layout = GetCollectiveLayout(
        param, laneCount, dataTypeSize, rootParticipates);
    return CollectiveLane{laneBase, layout, GetCollectiveSlice(param, layout, laneBase, param.myRank)};
}

CollectiveSlice GetCollectiveSlice(const OpParam &param, const CollectiveLayout &layout,
    uint64_t baseAddr, uint32_t rank)
{
    const bool rootParticipates = layout.rootParticipates;
    if (!rootParticipates && rank == param.root) {
        return CollectiveSlice{};
    }
    const uint32_t workerCount = rootParticipates ? param.rankSize : param.rankSize - 1;
    const uint32_t workerIndex = rootParticipates ? rank : (rank < param.root ? rank : rank - 1);
    return CollectiveSlice{baseAddr + workerIndex * layout.baseSliceSize,
        layout.baseSliceSize + (workerIndex + 1 == workerCount ? layout.tailSize : 0)};
}

HcclResult LaunchKernel(ThreadHandle thread, CcuKernelHandle kernel,
    const uint64_t *taskArgs, uint32_t taskArgCount)
{
    const CcuResult ccuRet = HcommCcuKernelLaunch(thread, kernel, taskArgs, taskArgCount);
    if (ccuRet != CCU_SUCCESS) {
        HCCL_ERROR("HcommCcuKernelLaunch failed, ret[%d]", ccuRet);
        return CcuResultToHccl(ccuRet);
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchParallelPair(const OpParam &param, const AlgResourceCtx &resCtx,
    const CcuKernelHandle kernels[AlgResourceCtx::MAX_KERNEL_GROUPS],
    const uint64_t *taskArgs, uint32_t taskArgCount)
{
    int32_t notifyRet = HcommThreadNotifyRecordOnThread(param.cpuThread, resCtx.parallelThread, 0);
    CHK_PRT_RET(notifyRet != HCCL_SUCCESS,
        HCCL_ERROR("Failed to record two-die start notify, ret[%d]", notifyRet),
        static_cast<HcclResult>(notifyRet));
    notifyRet = HcommThreadNotifyWaitOnThreadWithDefaultTimeout(resCtx.parallelThread, 0);
    CHK_PRT_RET(notifyRet != HCCL_SUCCESS,
        HCCL_ERROR("Failed to enqueue two-die start wait, ret[%d]", notifyRet),
        static_cast<HcclResult>(notifyRet));
    CHK_RET(LaunchKernel(resCtx.parallelThread, kernels[1], taskArgs, taskArgCount));
    notifyRet = HcommThreadNotifyRecordOnThread(resCtx.parallelThread, param.cpuThread, 0);
    CHK_PRT_RET(notifyRet != HCCL_SUCCESS,
        HCCL_ERROR("Failed to record two-die completion notify, ret[%d]", notifyRet),
        static_cast<HcclResult>(notifyRet));
    CHK_RET(LaunchKernel(param.cpuThread, kernels[0], taskArgs, taskArgCount));
    notifyRet = HcommThreadNotifyWaitOnThreadWithDefaultTimeout(param.cpuThread, 0);
    CHK_PRT_RET(notifyRet != HCCL_SUCCESS,
        HCCL_ERROR("Failed to enqueue two-die completion wait, ret[%d]", notifyRet),
        static_cast<HcclResult>(notifyRet));
    return HCCL_SUCCESS;
}

HcclResult BeginParallelPair(const OpParam &param, const AlgResourceCtx &resCtx)
{
    int32_t notifyRet = HcommThreadNotifyRecordOnThread(param.cpuThread, resCtx.parallelThread, 0);
    CHK_PRT_RET(notifyRet != HCCL_SUCCESS,
        HCCL_ERROR("Failed to record two-die start notify, ret[%d]", notifyRet),
        static_cast<HcclResult>(notifyRet));
    notifyRet = HcommThreadNotifyWaitOnThreadWithDefaultTimeout(resCtx.parallelThread, 0);
    CHK_PRT_RET(notifyRet != HCCL_SUCCESS,
        HCCL_ERROR("Failed to enqueue two-die start wait, ret[%d]", notifyRet),
        static_cast<HcclResult>(notifyRet));
    return HCCL_SUCCESS;
}

HcclResult RecordParallelCompletion(const OpParam &param, const AlgResourceCtx &resCtx)
{
    const int32_t notifyRet = HcommThreadNotifyRecordOnThread(resCtx.parallelThread, param.cpuThread, 0);
    CHK_PRT_RET(notifyRet != HCCL_SUCCESS,
        HCCL_ERROR("Failed to record two-die completion notify, ret[%d]", notifyRet),
        static_cast<HcclResult>(notifyRet));
    return HCCL_SUCCESS;
}

HcclResult JoinParallelPair(const OpParam &param)
{
    const int32_t notifyRet = HcommThreadNotifyWaitOnThreadWithDefaultTimeout(param.cpuThread, 0);
    CHK_PRT_RET(notifyRet != HCCL_SUCCESS,
        HCCL_ERROR("Failed to enqueue two-die completion wait, ret[%d]", notifyRet),
        static_cast<HcclResult>(notifyRet));
    return HCCL_SUCCESS;
}

HcclResult ExecDirect(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t baseAddr, uint64_t dataSize, uint64_t token)
{
    CHK_PRT_RET(resCtx.stageOneKernelCount == 0, HCCL_ERROR("Direct CCU kernels are not registered"),
        HCCL_E_INTERNAL);

    uint64_t processedSize = 0;
    while (processedSize < dataSize) {
        const uint64_t chunkSize = std::min<uint64_t>(MAX_DATA_SIZE, dataSize - processedSize);
        const std::array<uint64_t, 3> taskArgs = {baseAddr + processedSize, token, chunkSize};
        const bool compactArgs =
            (resCtx.algorithm == BroadcastAlgorithm::DIRECT && param.myRank != param.root) ||
            (resCtx.algorithm == BroadcastAlgorithm::DIRECT_PULL && param.myRank == param.root);
        const uint32_t taskArgCount = compactArgs ? 2 : static_cast<uint32_t>(taskArgs.size());
        if (resCtx.stageOneParallel) {
            CHK_PRT_RET(resCtx.stageOneKernelCount != 2,
                HCCL_ERROR("Parallel direct phase must contain two kernels"), HCCL_E_INTERNAL);
            CHK_RET(LaunchParallelPair(param, resCtx, resCtx.stageOneKernels,
                taskArgs.data(), taskArgCount));
        } else {
            for (uint32_t kernelIdx = 0; kernelIdx < resCtx.stageOneKernelCount; ++kernelIdx) {
                CHK_RET(LaunchKernel(param.cpuThread, resCtx.stageOneKernels[kernelIdx],
                    taskArgs.data(), taskArgCount));
            }
        }
        processedSize += chunkSize;
    }
    return HCCL_SUCCESS;
}

HcclResult ExecRank4PipelineSplit(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t baseAddr, uint64_t dataTypeSize, uint64_t token)
{
    constexpr uint64_t PIPELINE_CHUNK_COUNT = 26;
    constexpr uint64_t PIPELINE_ALIGNMENT = 4 * 1024;
    CHK_PRT_RET((param.rankSize != 4 && param.rankSize != 12 && param.rankSize != 16) ||
        resCtx.stageOneKernelCount == 0 ||
        resCtx.stageOneKernelCount > 2,
        HCCL_ERROR("Invalid topology pipeline-split resources, rankSize[%u], kernelCount[%u]",
            param.rankSize, resCtx.stageOneKernelCount), HCCL_E_INTERNAL);
    const uint64_t firstSize = (param.count / 2) * dataTypeSize;
    const uint64_t dataSize = param.count * dataTypeSize;
    const uint64_t secondSize = dataSize - firstSize;
    const auto getChunkSize = [](uint64_t halfSize) {
        const uint64_t averageSize =
            (halfSize + PIPELINE_CHUNK_COUNT - 1) / PIPELINE_CHUNK_COUNT;
        return (averageSize + PIPELINE_ALIGNMENT - 1) & ~(PIPELINE_ALIGNMENT - 1);
    };
    const uint64_t firstChunkSize = getChunkSize(firstSize);
    const uint64_t secondChunkSize = getChunkSize(secondSize);
    CHK_PRT_RET(firstChunkSize * (PIPELINE_CHUNK_COUNT - 1) >= firstSize ||
        secondChunkSize * (PIPELINE_CHUNK_COUNT - 1) >= secondSize,
        HCCL_ERROR("Balanced pipeline chunk size exceeds half payload"),
        HCCL_E_INTERNAL);
    const uint64_t firstLastSize =
        firstSize - firstChunkSize * (PIPELINE_CHUNK_COUNT - 1);
    const uint64_t secondLastSize =
        secondSize - secondChunkSize * (PIPELINE_CHUNK_COUNT - 1);
    CHK_PRT_RET(firstChunkSize == 0 || secondChunkSize == 0 ||
        firstLastSize > MAX_DATA_SIZE || secondLastSize > MAX_DATA_SIZE,
        HCCL_ERROR("Invalid pipeline-split chunk sizes[%llu, %llu, %llu, %llu]",
            static_cast<unsigned long long>(firstChunkSize),
            static_cast<unsigned long long>(firstLastSize),
            static_cast<unsigned long long>(secondChunkSize),
            static_cast<unsigned long long>(secondLastSize)), HCCL_E_INTERNAL);
    const std::array<uint64_t, 7> taskArgs = {token, baseAddr, baseAddr + firstSize,
        firstChunkSize, firstLastSize, secondChunkSize, secondLastSize};
    if (resCtx.stageOneParallel) {
        CHK_PRT_RET(resCtx.stageOneKernelCount != 2,
            HCCL_ERROR("Parallel pipeline-split phase must contain two kernels"),
            HCCL_E_INTERNAL);
        return LaunchParallelPair(param, resCtx, resCtx.stageOneKernels,
            taskArgs.data(), static_cast<uint32_t>(taskArgs.size()));
    }
    CHK_PRT_RET(resCtx.stageOneKernelCount != 1,
        HCCL_ERROR("Serial pipeline-split phase must contain one kernel"), HCCL_E_INTERNAL);
    return LaunchKernel(param.cpuThread, resCtx.stageOneKernels[0],
        taskArgs.data(), static_cast<uint32_t>(taskArgs.size()));
}

HcclResult ExecScatterAllgather(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t baseAddr, uint64_t dataTypeSize, uint64_t token)
{
    const bool fuseRootSlice = resCtx.algorithm == BroadcastAlgorithm::FUSED_SCATTER_ALLGATHER;
    const bool participatesInAllgather = param.myRank != param.root ||
        (!fuseRootSlice && RootParticipates(param.rankSize));
    const bool invalidStageTwo = participatesInAllgather ? resCtx.stageTwoKernelCount == 0 :
        resCtx.stageTwoKernelCount != 0;
    CHK_PRT_RET(resCtx.stageOneKernelCount == 0 || invalidStageTwo,
        HCCL_ERROR("Scatter-AllGather CCU context is invalid"), HCCL_E_INTERNAL);

    const auto launchScatterKernel = [&](uint32_t kernelIdx, ThreadHandle thread) -> HcclResult {
        const CollectiveLane lane = GetCollectiveLane(param, resCtx, baseAddr, dataTypeSize, kernelIdx);
        std::array<uint64_t, MAX_RANK_SIZE + 7> scatterArgs;
        uint32_t scatterArgCount = 0;
        if (param.myRank == param.root) {
            scatterArgs[scatterArgCount++] = token;
            if (fuseRootSlice) {
                const CollectiveSlice rootSlice =
                    GetCollectiveSlice(param, lane.layout, lane.baseAddr, param.root);
                scatterArgs[scatterArgCount++] = lane.baseAddr;
                scatterArgs[scatterArgCount++] = lane.layout.baseSliceSize;
                scatterArgs[scatterArgCount++] = lane.layout.tailSize;
                scatterArgs[scatterArgCount++] = rootSlice.addr - lane.baseAddr;
                scatterArgs[scatterArgCount++] = rootSlice.size;
            } else {
                scatterArgs[scatterArgCount++] = lane.layout.baseSliceSize;
                scatterArgs[scatterArgCount++] = lane.layout.tailSize;
            }
            for (uint32_t peerIdx = 0; peerIdx < resCtx.stageOnePeerCounts[kernelIdx]; ++peerIdx) {
                const uint32_t peerRank = resCtx.stageOnePeerRanks[kernelIdx][peerIdx];
                const CollectiveSlice peerSlice =
                    GetCollectiveSlice(param, lane.layout, lane.baseAddr, peerRank);
                scatterArgs[scatterArgCount++] =
                    fuseRootSlice ? peerSlice.addr - lane.baseAddr : peerSlice.addr;
            }
        } else {
            if (fuseRootSlice) {
                scatterArgs[scatterArgCount++] = lane.baseAddr;
            } else {
                scatterArgs[scatterArgCount++] = lane.localSlice.addr;
            }
            scatterArgs[scatterArgCount++] = token;
        }
        return LaunchKernel(thread, resCtx.stageOneKernels[kernelIdx],
            scatterArgs.data(), scatterArgCount);
    };
    if (resCtx.stageOneParallel) {
        CHK_PRT_RET(resCtx.stageOneKernelCount != 2,
            HCCL_ERROR("Parallel scatter phase must contain two kernels"), HCCL_E_INTERNAL);
        CHK_RET(BeginParallelPair(param, resCtx));
        CHK_RET(launchScatterKernel(1, resCtx.parallelThread));
        CHK_RET(RecordParallelCompletion(param, resCtx));
        CHK_RET(launchScatterKernel(0, param.cpuThread));
        CHK_RET(JoinParallelPair(param));
    } else {
        for (uint32_t kernelIdx = 0; kernelIdx < resCtx.stageOneKernelCount; ++kernelIdx) {
            CHK_RET(launchScatterKernel(kernelIdx, param.cpuThread));
        }
    }

    if (participatesInAllgather) {
        const auto launchAllgatherKernel = [&](uint32_t kernelIdx, ThreadHandle thread) -> HcclResult {
            const CollectiveLane lane = GetCollectiveLane(param, resCtx, baseAddr, dataTypeSize, kernelIdx);
            if (fuseRootSlice) {
                const std::array<uint64_t, 5> fusedPushArgs = {
                    lane.baseAddr, lane.localSlice.addr, token,
                    lane.localSlice.addr - lane.baseAddr, lane.localSlice.size};
                return LaunchKernel(thread, resCtx.stageTwoKernels[kernelIdx],
                    fusedPushArgs.data(), static_cast<uint32_t>(fusedPushArgs.size()));
            }
            if (param.rankSize != 4) {
                const std::array<uint64_t, 5> pushArgs = {
                    lane.baseAddr, lane.localSlice.addr, token,
                    lane.localSlice.addr - lane.baseAddr, lane.localSlice.size};
                return LaunchKernel(thread, resCtx.stageTwoKernels[kernelIdx],
                    pushArgs.data(), static_cast<uint32_t>(pushArgs.size()));
            }
            std::array<uint64_t, MAX_RANK_SIZE + 4> allgatherArgs;
            uint32_t allgatherArgCount = 0;
            allgatherArgs[allgatherArgCount++] = lane.localSlice.addr;
            allgatherArgs[allgatherArgCount++] = token;
            allgatherArgs[allgatherArgCount++] = lane.layout.baseSliceSize;
            allgatherArgs[allgatherArgCount++] = lane.layout.tailSize;
            for (uint32_t peerIdx = 0; peerIdx < resCtx.stageTwoPeerCounts[kernelIdx]; ++peerIdx) {
                const uint32_t peerRank = resCtx.stageTwoPeerRanks[kernelIdx][peerIdx];
                allgatherArgs[allgatherArgCount++] =
                    GetCollectiveSlice(param, lane.layout, lane.baseAddr, peerRank).addr;
            }
            return LaunchKernel(thread, resCtx.stageTwoKernels[kernelIdx],
                allgatherArgs.data(), allgatherArgCount);
        };
        if (resCtx.stageTwoParallel) {
            CHK_PRT_RET(resCtx.stageTwoKernelCount != 2,
                HCCL_ERROR("Parallel AllGather phase must contain two kernels"), HCCL_E_INTERNAL);
            CHK_RET(BeginParallelPair(param, resCtx));
            CHK_RET(launchAllgatherKernel(1, resCtx.parallelThread));
            CHK_RET(RecordParallelCompletion(param, resCtx));
            CHK_RET(launchAllgatherKernel(0, param.cpuThread));
            CHK_RET(JoinParallelPair(param));
        } else {
            for (uint32_t kernelIdx = 0; kernelIdx < resCtx.stageTwoKernelCount; ++kernelIdx) {
                CHK_RET(launchAllgatherKernel(kernelIdx, param.cpuThread));
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ExecRank4Recursive(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t baseAddr, uint64_t dataTypeSize, uint64_t token)
{
    CHK_PRT_RET(param.rankSize != 4, HCCL_ERROR("Rank4 recursive algorithm requires four ranks"),
        HCCL_E_INTERNAL);
    const uint32_t relativeRank = (param.myRank + param.rankSize - param.root) % param.rankSize;
    const uint64_t quarterCount = param.count / 4;
    const uint64_t quarterSize = quarterCount * dataTypeSize;
    const uint64_t tailSize = (param.count % 4) * dataTypeSize;

    for (uint32_t phase = 0; phase < 3; ++phase) {
        if ((resCtx.recursiveKernelMask & (1U << phase)) == 0) {
            continue;
        }

        uint64_t localSource = baseAddr;
        uint64_t localDestination = baseAddr;
        uint64_t transferSize = 0;
        uint64_t operation = 2;
        if (phase == 0) {
            localSource += 2 * quarterSize;
            localDestination += 2 * quarterSize;
            transferSize = 2 * quarterSize + tailSize;
            operation = relativeRank == 0 ? 0 : 1;
        } else if (phase == 1) {
            const bool upperPair = relativeRank >= 2;
            const uint64_t halfOffset = upperPair ? 2 * quarterSize : 0;
            localSource += halfOffset;
            localDestination += halfOffset;
            transferSize = 2 * quarterSize + (upperPair ? tailSize : 0);
            operation = (relativeRank & 1U) == 0 ? 0 : 1;
        } else {
            const bool upperHalf = relativeRank >= 2;
            localSource += upperHalf ? 2 * quarterSize : 0;
            localDestination += upperHalf ? 0 : 2 * quarterSize;
            transferSize = 2 * quarterSize + (upperHalf ? tailSize : 0);
            if (relativeRank == 0) {
                operation = 0;
            } else if (relativeRank == 2) {
                operation = 1;
            }
        }

        const std::array<uint64_t, 5> taskArgs = {
            token, localSource, localDestination, transferSize, operation};
        CHK_RET(LaunchKernel(param.cpuThread, resCtx.recursiveKernels[phase],
            taskArgs.data(), static_cast<uint32_t>(taskArgs.size())));
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param)
{
    AlgResourceCtx resCtx{};
    CHK_RET(resCtx.DeSerialize(param.resCtx, param.ctxSize));

    const auto sizeIter = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(sizeIter == SIZE_TABLE.end(), HCCL_ERROR("Unsupported dataType[%d]", param.dataType), HCCL_E_PARA);
    const uint64_t dataTypeSize = sizeIter->second;
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / dataTypeSize,
        HCCL_ERROR("Broadcast data size overflow, count[%llu]", static_cast<unsigned long long>(param.count)),
        HCCL_E_PARA);
    const uint64_t dataSize = param.count * dataTypeSize;
    if (dataSize == 0 || param.rankSize <= 1) {
        return HCCL_SUCCESS;
    }
    const uint64_t baseAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    uint64_t token = 0;
    const CcuResult ccuRet = HcommCcuGetMemToken(baseAddr, dataSize, &token);
    if (ccuRet != CCU_SUCCESS) {
        HCCL_ERROR("HcommCcuGetMemToken failed, ret[%d]", ccuRet);
        return CcuResultToHccl(ccuRet);
    }

    switch (resCtx.algorithm) {
        case BroadcastAlgorithm::DIRECT:
        case BroadcastAlgorithm::DIRECT_PULL:
            return ExecDirect(param, resCtx, baseAddr, dataSize, token);
        case BroadcastAlgorithm::SCATTER_ALLGATHER:
        case BroadcastAlgorithm::FUSED_SCATTER_ALLGATHER:
            return ExecScatterAllgather(param, resCtx, baseAddr, dataTypeSize, token);
        case BroadcastAlgorithm::RANK4_RECURSIVE:
            return ExecRank4Recursive(param, resCtx, baseAddr, dataTypeSize, token);
        case BroadcastAlgorithm::RANK4_PIPELINE_SPLIT:
            return ExecRank4PipelineSplit(param, resCtx, baseAddr, dataTypeSize, token);
        default:
            HCCL_ERROR("Unsupported broadcast algorithm[%u]", static_cast<uint32_t>(resCtx.algorithm));
            return HCCL_E_INTERNAL;
    }
}
} // namespace ops_hccl
