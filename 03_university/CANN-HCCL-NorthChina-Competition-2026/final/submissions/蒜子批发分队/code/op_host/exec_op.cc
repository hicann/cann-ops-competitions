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

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>
#include <hccl/hcomm_primitives.h>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {
constexpr uint64_t CCU_MS_SIZE = 4096;
constexpr uint64_t LOCAL_COPY_MS_PER_LOOP = 8;
constexpr uint64_t LOCAL_COPY_LOOP_COUNT = 8;
constexpr uint64_t LOCAL_COPY_SLICE =
    CCU_MS_SIZE * LOCAL_COPY_MS_PER_LOOP;
constexpr uint64_t FINE_LOCAL_COPY_MS_PER_LOOP = 4;
constexpr uint64_t FINE_LOCAL_COPY_LOOP_COUNT = 16;
constexpr uint64_t FINE_LOCAL_COPY_SLICE =
    CCU_MS_SIZE * FINE_LOCAL_COPY_MS_PER_LOOP;
constexpr uint64_t FINE_LOCAL_COPY_MIN_BYTES = 1ULL << 20;
constexpr uint64_t RANK12_MESH_FIRST_BYTES =
    (400ULL << 20) + 4;
constexpr uint64_t FUSED_512_BYTES = 512ULL << 20;
constexpr uint64_t FUSED_DIRECT_SLICE_BYTES =
    static_cast<uint64_t>(MAX_DATA_SIZE);
constexpr uint64_t FUSED_SECOND_SLICE_BYTES =
    FUSED_512_BYTES - FUSED_DIRECT_SLICE_BYTES;
constexpr uint64_t GROUP_BROADCAST_MS_PER_LOOP = 4;
constexpr uint64_t GROUP_BROADCAST_LOOP_COUNT = 16;
constexpr uint64_t GROUP_BROADCAST_SLICE =
    CCU_MS_SIZE * GROUP_BROADCAST_MS_PER_LOOP;
constexpr uint64_t RANK16_DIRECT_NUMERATOR = 1;
constexpr uint64_t RANK16_DIRECT_DENOMINATOR = 3;
constexpr uint64_t RANK16_SPLIT_ALIGNMENT =
    GROUP_BROADCAST_SLICE * GROUP_BROADCAST_LOOP_COUNT;
constexpr uint64_t RANK12_DUAL_RAIL_ALIGNMENT =
    FINE_LOCAL_COPY_SLICE * FINE_LOCAL_COPY_LOOP_COUNT;
constexpr uint64_t RANK12_BIG_DIRECT_NUMERATOR = 5;
constexpr uint64_t RANK12_BIG_DIRECT_DENOMINATOR = 7;
constexpr uint64_t RANK12_SMALL_DIRECT_NUMERATOR = 7;
constexpr uint64_t RANK12_SMALL_DIRECT_DENOMINATOR = 11;
constexpr uint64_t RANK12_PHASE_ZERO = 0;
constexpr uint64_t RANK12_PHASE_ONE = 1;
constexpr uint64_t RANK16_PHASE_ZERO = 0;
constexpr uint64_t RANK16_PHASE_ONE = 1;
constexpr uint64_t RANK12_PIPE_CHUNK_BYTES = 128ULL << 20;
constexpr uint64_t RANK12_MIN_PIPE_CHUNK_BYTES =
    RANK12_DUAL_RAIL_ALIGNMENT * 2;
constexpr uint32_t PHASE_START_NOTIFY_INDEX = 0;
constexpr uint32_t PHASE_BARRIER_NOTIFY_INDEX = 1;
using KernelTaskArgs =
    std::array<uint64_t, AlgResourceCtx::MAX_TASK_ARG_COUNT>;
using KernelTaskArgSet =
    std::array<KernelTaskArgs, AlgResourceCtx::MAX_KERNEL_COUNT>;
static_assert(FUSED_DIRECT_SLICE_BYTES > 0 &&
        FUSED_DIRECT_SLICE_BYTES <= MAX_DATA_SIZE,
    "first fused Direct slice exceeds the transfer limit");
static_assert(FUSED_SECOND_SLICE_BYTES > 0 &&
        FUSED_SECOND_SLICE_BYTES <= MAX_DATA_SIZE,
    "second fused Direct slice exceeds the transfer limit");
static_assert(FUSED_DIRECT_SLICE_BYTES == (256ULL << 20) &&
        FUSED_SECOND_SLICE_BYTES == (256ULL << 20) &&
        FUSED_DIRECT_SLICE_BYTES + FUSED_SECOND_SLICE_BYTES ==
            FUSED_512_BYTES,
    "512 MiB fusion requires two exact 256 MiB slices");
static_assert(RANK12_PIPE_CHUNK_BYTES > 0 &&
        RANK12_PIPE_CHUNK_BYTES <= MAX_DATA_SIZE,
    "rank12 pipeline chunk exceeds the transfer limit");

constexpr uint64_t LowBits(uint32_t highestBit)
{
    return (uint64_t{1} << (highestBit + 1)) - 1;
}

uint64_t GetRank12PipelineChunkBytes(uint64_t remainingBytes)
{
    if (remainingBytes > RANK12_PIPE_CHUNK_BYTES &&
        remainingBytes - RANK12_PIPE_CHUNK_BYTES <
            RANK12_MIN_PIPE_CHUNK_BYTES) {
        return remainingBytes;
    }
    return std::min<uint64_t>(
        RANK12_PIPE_CHUNK_BYTES, remainingBytes);
}

uint64_t GetRank12DirectBytes(
    uint64_t operationBytes, bool smallServerSource)
{
    uint64_t directNumerator =
        smallServerSource ?
            RANK12_SMALL_DIRECT_NUMERATOR :
            RANK12_BIG_DIRECT_NUMERATOR;
    uint64_t directDenominator =
        smallServerSource ?
            RANK12_SMALL_DIRECT_DENOMINATOR :
            RANK12_BIG_DIRECT_DENOMINATOR;
    uint64_t directBytes =
        operationBytes * directNumerator /
        directDenominator;
    return directBytes /
        RANK12_DUAL_RAIL_ALIGNMENT *
        RANK12_DUAL_RAIL_ALIGNMENT;
}

uint64_t GetParallelParam(
    uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum)
{
    return ((repeatNum & LowBits(7)) << 55) |
        ((repeatLoopIndex & LowBits(7)) << 48) |
        ((totalLoopNum & LowBits(7)) << 41);
}

std::array<uint64_t, 4> GetGroupCopySize(
    uint64_t size, uint64_t copySlice, uint64_t loopCount)
{
    uint64_t loopSize = copySlice * loopCount;
    uint64_t fullLoops = size / loopSize;
    uint64_t remainder = size - fullLoops * loopSize;
    uint64_t fullSlices = remainder / copySlice;
    uint64_t tailBytes = remainder - fullSlices * copySlice;

    uint64_t parallelParam = 0;
    uint64_t residual = 0;
    if (fullSlices != 0 && tailBytes == 0) {
        parallelParam = GetParallelParam(fullSlices - 1, 0, 1);
        residual = copySlice;
    } else if (fullSlices == 0 && tailBytes != 0) {
        parallelParam = GetParallelParam(0, 0, 1);
        residual = tailBytes;
    } else if (fullSlices != 0 && tailBytes != 0) {
        parallelParam = GetParallelParam(fullSlices - 1, 1, 2);
        residual = tailBytes;
    }

    return {
        fullLoops * loopSize,
        fullLoops,
        parallelParam,
        residual,
    };
}

HcclResult LaunchKernelGroup(const OpParam &param, const AlgResourceCtx &resource,
    const KernelTaskArgSet &taskArgs)
{
    ThreadHandle mainThread = param.cpuThread;
    ThreadHandle slaveThread = resource.threads[1];

    if (resource.kernelCount == 1) {
        CcuResult ccuRet = HcommCcuKernelLaunch(mainThread,
            resource.ccuKernels[0], taskArgs[0].data(),
            resource.taskArgCounts[0]);
        if (ccuRet != CCU_SUCCESS) {
            HCCL_ERROR("CCU kernel launch failed, index=0 ret=%d", ccuRet);
            return ConvertCcuResult(ccuRet);
        }
        return HCCL_SUCCESS;
    }

    CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, slaveThread, 0));
    CHK_RET(HcommThreadNotifyWaitOnThread(slaveThread, 0, CUSTOM_TIMEOUT));

    CcuResult ccuRet = HcommCcuKernelLaunch(slaveThread,
        resource.ccuKernels[1], taskArgs[1].data(),
        resource.taskArgCounts[1]);
    if (ccuRet != CCU_SUCCESS) {
        HCCL_ERROR("CCU kernel launch failed, index=1 ret=%d", ccuRet);
        return ConvertCcuResult(ccuRet);
    }
    CHK_RET(HcommThreadNotifyRecordOnThread(slaveThread, mainThread, 0));

    ccuRet = HcommCcuKernelLaunch(mainThread,
        resource.ccuKernels[0], taskArgs[0].data(),
        resource.taskArgCounts[0]);
    if (ccuRet != CCU_SUCCESS) {
        HCCL_ERROR("CCU kernel launch failed, index=0 ret=%d", ccuRet);
        return ConvertCcuResult(ccuRet);
    }
    CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, 0, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult LaunchRank16MixedPhases(const OpParam &param,
    const AlgResourceCtx &resource,
    const KernelTaskArgSet &phaseZeroTaskArgs,
    const KernelTaskArgSet &phaseOneTaskArgs)
{
    ThreadHandle mainThread = param.cpuThread;
    ThreadHandle slaveThread = resource.threads[1];
    uint32_t meshIndex = resource.rank16MeshKernelIndex;
    uint32_t closIndex = resource.rank16ClosKernelIndex;

    CHK_RET(HcommThreadNotifyRecordOnThread(
        mainThread, slaveThread,
        PHASE_START_NOTIFY_INDEX));
    CHK_RET(HcommThreadNotifyWaitOnThread(
        slaveThread, PHASE_START_NOTIFY_INDEX,
        CUSTOM_TIMEOUT));

    CcuResult ccuRet = HcommCcuKernelLaunch(slaveThread,
        resource.ccuKernels[meshIndex],
        phaseZeroTaskArgs[meshIndex].data(),
        resource.taskArgCounts[meshIndex]);
    if (ccuRet != CCU_SUCCESS) {
        HCCL_ERROR("rank16 Mesh phase-0 launch failed, ret=%d",
            ccuRet);
        return ConvertCcuResult(ccuRet);
    }
    CHK_RET(HcommThreadNotifyWaitOnThread(
        slaveThread, PHASE_BARRIER_NOTIFY_INDEX,
        CUSTOM_TIMEOUT));

    ccuRet = HcommCcuKernelLaunch(slaveThread,
        resource.ccuKernels[meshIndex],
        phaseOneTaskArgs[meshIndex].data(),
        resource.taskArgCounts[meshIndex]);
    if (ccuRet != CCU_SUCCESS) {
        HCCL_ERROR("rank16 Mesh phase-1 launch failed, ret=%d",
            ccuRet);
        return ConvertCcuResult(ccuRet);
    }
    CHK_RET(HcommThreadNotifyRecordOnThread(
        slaveThread, mainThread, PHASE_START_NOTIFY_INDEX));

    ccuRet = HcommCcuKernelLaunch(mainThread,
        resource.ccuKernels[closIndex],
        phaseZeroTaskArgs[closIndex].data(),
        resource.taskArgCounts[closIndex]);
    if (ccuRet != CCU_SUCCESS) {
        HCCL_ERROR("rank16 Clos phase-0 launch failed, ret=%d",
            ccuRet);
        return ConvertCcuResult(ccuRet);
    }
    CHK_RET(HcommThreadNotifyRecordOnThread(
        mainThread, slaveThread,
        PHASE_BARRIER_NOTIFY_INDEX));

    ccuRet = HcommCcuKernelLaunch(mainThread,
        resource.ccuKernels[closIndex],
        phaseOneTaskArgs[closIndex].data(),
        resource.taskArgCounts[closIndex]);
    if (ccuRet != CCU_SUCCESS) {
        HCCL_ERROR("rank16 Clos phase-1 launch failed, ret=%d",
            ccuRet);
        return ConvertCcuResult(ccuRet);
    }
    CHK_RET(HcommThreadNotifyWaitOnThread(
        mainThread, PHASE_START_NOTIFY_INDEX, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

KernelTaskArgSet BuildRank12TaskArgs(
    const AlgResourceCtx &resource,
    uint64_t inputAddress, uint64_t outputAddress,
    uint64_t inputToken, uint64_t outputToken,
    uint64_t inputBytes, uint64_t processedBytes,
    uint64_t operationBytes, bool inputAlreadyPlaced,
    uint64_t phase)
{
    uint64_t closDirectBytes =
        GetRank12DirectBytes(
            operationBytes, resource.rank12SmallServer);
    uint64_t closRelayBytes =
        operationBytes - closDirectBytes;
    uint64_t fanoutDirectBytes =
        GetRank12DirectBytes(
            operationBytes, !resource.rank12SmallServer);
    uint64_t fanoutRelayBytes =
        operationBytes - fanoutDirectBytes;
    uint64_t meshTransferBytes =
        phase == RANK12_PHASE_ZERO ?
            operationBytes : fanoutRelayBytes;
    uint64_t rankOutputOffset =
        static_cast<uint64_t>(resource.rankId) *
        inputBytes;
    uint64_t outputOffset =
        rankOutputOffset + processedBytes;
    uint64_t fanoutOutputOffsets[2] = {};
    for (uint32_t sourceIndex = 0;
         sourceIndex < resource.rank12FanoutSourceCount;
         ++sourceIndex) {
        fanoutOutputOffsets[sourceIndex] =
            static_cast<uint64_t>(
                resource.rank12FanoutSourceRanks[
                    sourceIndex]) *
                inputBytes +
            processedBytes +
            fanoutDirectBytes;
    }
    std::array<uint64_t, 4> closGroupSize =
        GetGroupCopySize(
            inputAlreadyPlaced ? 0 : operationBytes,
            FINE_LOCAL_COPY_SLICE,
            FINE_LOCAL_COPY_LOOP_COUNT);

    KernelTaskArgSet taskArgs{};
    taskArgs[resource.rank12MeshKernelIndex] = {
        inputAddress + processedBytes,
        outputAddress,
        inputToken,
        outputToken,
        outputOffset,
        meshTransferBytes,
        fanoutOutputOffsets[0],
        fanoutOutputOffsets[1],
        phase,
    };
    taskArgs[resource.rank12ClosKernelIndex] = {
        inputAddress + processedBytes,
        outputAddress,
        inputToken,
        outputToken,
        outputOffset,
        operationBytes,
        closDirectBytes,
        closRelayBytes,
        closGroupSize[0],
        closGroupSize[1],
        closGroupSize[2],
        closGroupSize[3],
        phase,
    };
    return taskArgs;
}

HcclResult LaunchRank12Pipeline(const OpParam &param,
    const AlgResourceCtx &resource, uint64_t inputAddress,
    uint64_t outputAddress, uint64_t inputToken,
    uint64_t outputToken, uint64_t inputBytes,
    bool inputAlreadyPlaced)
{
    ThreadHandle mainThread = param.cpuThread;
    ThreadHandle slaveThread = resource.threads[1];
    uint32_t meshIndex = resource.rank12MeshKernelIndex;
    uint32_t closIndex = resource.rank12ClosKernelIndex;

    auto launchKernel = [&resource](
        ThreadHandle thread, uint32_t kernelIndex,
        const KernelTaskArgSet &taskArgs,
        const char *stage) -> HcclResult {
        CcuResult ccuRet = HcommCcuKernelLaunch(
            thread, resource.ccuKernels[kernelIndex],
            taskArgs[kernelIndex].data(),
            resource.taskArgCounts[kernelIndex]);
        if (ccuRet != CCU_SUCCESS) {
            HCCL_ERROR("rank12 %s launch failed, ret=%d",
                stage, ccuRet);
            return ConvertCcuResult(ccuRet);
        }
        return HCCL_SUCCESS;
    };

    uint64_t processedBytes = 0;
    uint64_t currentBytes =
        GetRank12PipelineChunkBytes(inputBytes);
    KernelTaskArgSet currentPhaseZero =
        BuildRank12TaskArgs(resource, inputAddress,
            outputAddress, inputToken, outputToken,
            inputBytes, processedBytes, currentBytes,
            inputAlreadyPlaced, RANK12_PHASE_ZERO);
    KernelTaskArgSet currentPhaseOne =
        BuildRank12TaskArgs(resource, inputAddress,
            outputAddress, inputToken, outputToken,
            inputBytes, processedBytes, currentBytes,
            inputAlreadyPlaced, RANK12_PHASE_ONE);

    CHK_RET(HcommThreadNotifyRecordOnThread(
        mainThread, slaveThread, PHASE_START_NOTIFY_INDEX));
    CHK_RET(HcommThreadNotifyWaitOnThread(
        slaveThread, PHASE_START_NOTIFY_INDEX,
        CUSTOM_TIMEOUT));
    CHK_RET(launchKernel(
        slaveThread, meshIndex, currentPhaseZero,
        "Mesh phase-0"));
    CHK_RET(HcommThreadNotifyRecordOnThread(
        slaveThread, mainThread,
        PHASE_BARRIER_NOTIFY_INDEX));
    CHK_RET(launchKernel(
        mainThread, closIndex, currentPhaseZero,
        "Clos phase-0"));
    CHK_RET(HcommThreadNotifyWaitOnThread(
        mainThread, PHASE_BARRIER_NOTIFY_INDEX,
        CUSTOM_TIMEOUT));
    CHK_RET(HcommThreadNotifyRecordOnThread(
        mainThread, slaveThread,
        PHASE_START_NOTIFY_INDEX));
    processedBytes += currentBytes;

    while (processedBytes < inputBytes) {
        uint64_t nextBytes =
            GetRank12PipelineChunkBytes(
                inputBytes - processedBytes);
        KernelTaskArgSet nextPhaseZero =
            BuildRank12TaskArgs(resource, inputAddress,
                outputAddress, inputToken, outputToken,
                inputBytes, processedBytes, nextBytes,
                inputAlreadyPlaced, RANK12_PHASE_ZERO);
        KernelTaskArgSet nextPhaseOne =
            BuildRank12TaskArgs(resource, inputAddress,
                outputAddress, inputToken, outputToken,
                inputBytes, processedBytes, nextBytes,
                inputAlreadyPlaced, RANK12_PHASE_ONE);

        CHK_RET(HcommThreadNotifyWaitOnThread(
            slaveThread, PHASE_START_NOTIFY_INDEX,
            CUSTOM_TIMEOUT));
        CHK_RET(launchKernel(
            slaveThread, meshIndex, currentPhaseOne,
            "Mesh phase-1"));
        CHK_RET(launchKernel(
            slaveThread, meshIndex, nextPhaseZero,
            "Mesh phase-0"));
        CHK_RET(HcommThreadNotifyRecordOnThread(
            slaveThread, mainThread,
            PHASE_BARRIER_NOTIFY_INDEX));

        CHK_RET(launchKernel(
            mainThread, closIndex, currentPhaseOne,
            "Clos phase-1"));
        CHK_RET(launchKernel(
            mainThread, closIndex, nextPhaseZero,
            "Clos phase-0"));
        CHK_RET(HcommThreadNotifyWaitOnThread(
            mainThread, PHASE_BARRIER_NOTIFY_INDEX,
            CUSTOM_TIMEOUT));
        CHK_RET(HcommThreadNotifyRecordOnThread(
            mainThread, slaveThread,
            PHASE_START_NOTIFY_INDEX));

        processedBytes += nextBytes;
        currentPhaseOne = nextPhaseOne;
    }

    CHK_RET(HcommThreadNotifyWaitOnThread(
        slaveThread, PHASE_START_NOTIFY_INDEX,
        CUSTOM_TIMEOUT));
    CHK_RET(launchKernel(
        slaveThread, meshIndex, currentPhaseOne,
        "Mesh phase-1"));
    CHK_RET(HcommThreadNotifyRecordOnThread(
        slaveThread, mainThread,
        PHASE_START_NOTIFY_INDEX));
    CHK_RET(launchKernel(
        mainThread, closIndex, currentPhaseOne,
        "Clos phase-1"));
    CHK_RET(HcommThreadNotifyWaitOnThread(
        mainThread, PHASE_START_NOTIFY_INDEX,
        CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param)
{
    CHK_PTR_NULL(param.resCtx);
    if (param.ctxSize == 0) {
        HCCL_ERROR("empty CCU engine context");
        return HCCL_E_INTERNAL;
    }

    AlgResourceCtx resCtx;
    CHK_RET(resCtx.DeSerialize(param.resCtx, param.ctxSize));

    if (resCtx.layoutVersion != 143 || resCtx.rankSize != param.rankSize ||
        resCtx.rankId != param.myRank ||
        resCtx.kernelCount == 0 ||
        resCtx.kernelCount > AlgResourceCtx::MAX_KERNEL_COUNT ||
        param.cpuThread == 0) {
        HCCL_ERROR("invalid CCU resource context, version=%u resourceRankSize=%u "
                   "rankSize=%u kernelCount=%u",
            resCtx.layoutVersion, resCtx.rankSize, param.rankSize,
            resCtx.kernelCount);
        return HCCL_E_INTERNAL;
    }
    if (resCtx.rank12DirectLocalCopy &&
        (resCtx.rankSize != 12 ||
            resCtx.cachedInputBytes <
                FINE_LOCAL_COPY_MIN_BYTES ||
            resCtx.rank12DualRail ||
            resCtx.rank12SplitRelay ||
            resCtx.rank16MixedRelay)) {
        HCCL_ERROR("invalid rank12 direct-local-copy metadata");
        return HCCL_E_INTERNAL;
    }
    if (resCtx.rank12DirectReuse &&
        (resCtx.rankSize != 12 ||
            resCtx.cachedInputBytes !=
                RANK12_MESH_FIRST_BYTES ||
            resCtx.rank12DirectLocalCopy ||
            resCtx.rank12Nhr ||
            resCtx.rank12DualRail ||
            resCtx.rank12SplitRelay ||
            resCtx.rank16MixedRelay)) {
        HCCL_ERROR("invalid rank12 direct-reuse metadata");
        return HCCL_E_INTERNAL;
    }
    if (resCtx.rank12Nhr &&
        (resCtx.rankSize != 12 ||
            resCtx.kernelCount != 2 ||
            resCtx.cachedInputBytes <
                FINE_LOCAL_COPY_MIN_BYTES ||
            resCtx.cachedInputBytes > FUSED_512_BYTES ||
            resCtx.rank12DirectLocalCopy ||
            resCtx.rank12DualRail ||
            resCtx.rank12SplitRelay ||
            resCtx.rank16MixedRelay)) {
        HCCL_ERROR("invalid rank12 NHR metadata");
        return HCCL_E_INTERNAL;
    }
    if (resCtx.rank12DualRail &&
        (resCtx.rankSize != 12 ||
            resCtx.kernelCount != 2 ||
            resCtx.rank12SplitRelay ||
            resCtx.rank16MixedRelay ||
            resCtx.rank12PrimaryRailWeight == 0 ||
            resCtx.rank12AlternateRailWeight == 0)) {
        HCCL_ERROR("invalid rank12 dual-rail metadata");
        return HCCL_E_INTERNAL;
    }
    if (resCtx.rank12SplitRelay &&
        (resCtx.rankSize != 12 ||
            resCtx.kernelCount != 2 ||
            resCtx.rank16MixedRelay ||
            resCtx.rank12MeshKernelIndex >= resCtx.kernelCount ||
            resCtx.rank12ClosKernelIndex >= resCtx.kernelCount ||
            resCtx.rank12MeshKernelIndex ==
                resCtx.rank12ClosKernelIndex ||
            resCtx.cachedInputBytes <
                FINE_LOCAL_COPY_MIN_BYTES ||
            resCtx.rank12FanoutSourceCount > 2 ||
            (resCtx.rank12SmallServer ?
                resCtx.rank12FanoutSourceCount != 2 :
                resCtx.rank12FanoutSourceCount !=
                    (resCtx.rankId < 4 ? 1U : 0U)))) {
        HCCL_ERROR("invalid rank12 split relay metadata");
        return HCCL_E_INTERNAL;
    }
    if (resCtx.rank12SplitRelay) {
        for (uint32_t sourceIndex = 0;
             sourceIndex < resCtx.rank12FanoutSourceCount;
             ++sourceIndex) {
            uint32_t sourceRank =
                resCtx.rank12FanoutSourceRanks[sourceIndex];
            if (sourceRank >= resCtx.rankSize ||
                sourceRank == resCtx.rankId ||
                (resCtx.rank12SmallServer ?
                    sourceRank >= 8 : sourceRank < 8)) {
                HCCL_ERROR(
                    "invalid rank12 hierarchical source metadata");
                return HCCL_E_INTERNAL;
            }
        }
    }
    if (resCtx.rank16MixedRelay &&
        (resCtx.rankSize != 16 ||
            resCtx.kernelCount != 2 ||
            resCtx.rank16MeshKernelIndex >= resCtx.kernelCount ||
            resCtx.rank16ClosKernelIndex >= resCtx.kernelCount ||
            resCtx.rank16MeshKernelIndex ==
                resCtx.rank16ClosKernelIndex ||
            resCtx.rank16RelayPeerRank >= resCtx.rankSize ||
            resCtx.rank16RelayPeerRank == resCtx.rankId)) {
        HCCL_ERROR("invalid rank16 mixed relay metadata");
        return HCCL_E_INTERNAL;
    }
    uint32_t fusedKernelCount = 0;
    for (uint32_t index = 0; index < resCtx.kernelCount; ++index) {
        bool rank12SplitKernel =
            resCtx.rank12SplitRelay &&
            (index == resCtx.rank12MeshKernelIndex ||
                index == resCtx.rank12ClosKernelIndex);
        uint32_t rank12TaskArgCount =
            index == resCtx.rank12MeshKernelIndex ?
                AlgResourceCtx::RANK12_MESH_TASK_ARG_COUNT :
                AlgResourceCtx::RANK12_CLOS_TASK_ARG_COUNT;
        bool rank16MixedKernel =
            resCtx.rank16MixedRelay &&
            (index == resCtx.rank16MeshKernelIndex ||
                index == resCtx.rank16ClosKernelIndex);
        uint32_t rank16TaskArgCount =
            index == resCtx.rank16MeshKernelIndex ?
                AlgResourceCtx::RANK16_MESH_TASK_ARG_COUNT :
                AlgResourceCtx::RANK16_CLOS_TASK_ARG_COUNT;
        bool rank12DualRailKernel =
            resCtx.rank12DualRail;
        bool rank12DualRailTaskLayout =
            resCtx.taskArgCounts[index] ==
                AlgResourceCtx::DUAL_RAIL_BASE_TASK_ARG_COUNT ||
            resCtx.taskArgCounts[index] ==
                AlgResourceCtx::
                    DUAL_RAIL_FUSED_BASE_TASK_ARG_COUNT ||
            resCtx.taskArgCounts[index] ==
                AlgResourceCtx::DUAL_RAIL_FULL_TASK_ARG_COUNT ||
            resCtx.taskArgCounts[index] ==
                AlgResourceCtx::
                    DUAL_RAIL_FUSED_FULL_TASK_ARG_COUNT;
        bool rank12DirectLocalCopyKernel =
            resCtx.rank12DirectLocalCopy &&
            (resCtx.taskArgCounts[index] ==
                    AlgResourceCtx::
                        DIRECT_LOCAL_COPY_TASK_ARG_COUNT ||
                resCtx.taskArgCounts[index] ==
                    AlgResourceCtx::
                        FUSED_DIRECT_LOCAL_COPY_TASK_ARG_COUNT);
        bool rank12NhrKernel =
            resCtx.rank12Nhr &&
            resCtx.taskArgCounts[index] ==
                AlgResourceCtx::RANK12_NHR_TASK_ARG_COUNT;
        if (resCtx.threads[index] == 0 || resCtx.ccuKernels[index] == 0 ||
            resCtx.kernelLinkLoads[index] == 0 ||
            (resCtx.rank12Nhr ?
                !rank12NhrKernel :
             resCtx.rank12DirectLocalCopy ?
                !rank12DirectLocalCopyKernel :
             rank12DualRailKernel ?
                !rank12DualRailTaskLayout :
             rank12SplitKernel ?
                resCtx.taskArgCounts[index] !=
                    rank12TaskArgCount :
             rank16MixedKernel ?
                resCtx.taskArgCounts[index] !=
                    rank16TaskArgCount :
                (resCtx.taskArgCounts[index] !=
                     AlgResourceCtx::BASE_TASK_ARG_COUNT &&
                 resCtx.taskArgCounts[index] !=
                     AlgResourceCtx::FUSED_BASE_TASK_ARG_COUNT &&
                 resCtx.taskArgCounts[index] !=
                     AlgResourceCtx::GROUP_BROADCAST_TASK_ARG_COUNT &&
                 resCtx.taskArgCounts[index] !=
                     AlgResourceCtx::HYBRID_TASK_ARG_COUNT &&
                 resCtx.taskArgCounts[index] !=
                     AlgResourceCtx::FULL_TASK_ARG_COUNT &&
                 resCtx.taskArgCounts[index] !=
                     AlgResourceCtx::FUSED_FULL_TASK_ARG_COUNT))) {
            HCCL_ERROR("invalid CCU resource handle, index=%u", index);
            return HCCL_E_INTERNAL;
        }
        bool fusedLayout =
            !rank12NhrKernel &&
            !rank12SplitKernel &&
            !rank16MixedKernel &&
            ((rank12DirectLocalCopyKernel &&
                 resCtx.taskArgCounts[index] ==
                    AlgResourceCtx::
                        FUSED_DIRECT_LOCAL_COPY_TASK_ARG_COUNT) ||
                (!rank12DirectLocalCopyKernel &&
                 !rank12DualRailKernel &&
                 (resCtx.taskArgCounts[index] ==
                      AlgResourceCtx::FUSED_BASE_TASK_ARG_COUNT ||
                     resCtx.taskArgCounts[index] ==
                      AlgResourceCtx::FUSED_FULL_TASK_ARG_COUNT)) ||
                (rank12DualRailKernel &&
                 (resCtx.taskArgCounts[index] ==
                      AlgResourceCtx::
                          DUAL_RAIL_FUSED_BASE_TASK_ARG_COUNT ||
                     resCtx.taskArgCounts[index] ==
                      AlgResourceCtx::
                          DUAL_RAIL_FUSED_FULL_TASK_ARG_COUNT)));
        if (fusedLayout) {
            if (param.rankSize != 4 && param.rankSize != 12) {
                HCCL_ERROR("invalid fused CCU task layout, index=%u",
                    index);
                return HCCL_E_INTERNAL;
            }
            ++fusedKernelCount;
        }
        if (!rank12NhrKernel &&
            !rank12SplitKernel &&
            !rank12DualRailKernel &&
            !rank12DirectLocalCopyKernel &&
            (resCtx.taskArgCounts[index] ==
                    AlgResourceCtx::GROUP_BROADCAST_TASK_ARG_COUNT ||
                (resCtx.taskArgCounts[index] ==
                     AlgResourceCtx::HYBRID_TASK_ARG_COUNT &&
                 param.rankSize != 16))) {
            HCCL_ERROR("invalid CCU task layout, index=%u", index);
            return HCCL_E_INTERNAL;
        }
    }
    if (fusedKernelCount != 0 &&
        (fusedKernelCount != resCtx.kernelCount ||
            resCtx.cachedInputBytes != FUSED_512_BYTES)) {
        HCCL_ERROR("inconsistent fused CCU task metadata");
        return HCCL_E_INTERNAL;
    }
    if (resCtx.kernelCount == 2 &&
        resCtx.kernelLinkLoads[0] < resCtx.kernelLinkLoads[1]) {
        HCCL_ERROR("invalid CCU kernel load order");
        return HCCL_E_INTERNAL;
    }

    auto dataTypeIter = SIZE_TABLE.find(param.dataType);
    if (dataTypeIter == SIZE_TABLE.end() || dataTypeIter->second == 0 ||
        param.count > UINT64_MAX / dataTypeIter->second) {
        return HCCL_E_PARA;
    }
    uint64_t inputBytes = param.count * dataTypeIter->second;
    if (inputBytes == 0) {
        return HCCL_SUCCESS;
    }
    if (param.rankSize == 0 || inputBytes > UINT64_MAX / param.rankSize) {
        return HCCL_E_PARA;
    }
    if (resCtx.rank12DirectReuse &&
        inputBytes != resCtx.cachedInputBytes) {
        HCCL_ERROR("rank12 direct-reuse input size changed, cached=%llu "
                   "current=%llu",
            static_cast<unsigned long long>(
                resCtx.cachedInputBytes),
            static_cast<unsigned long long>(inputBytes));
        return HCCL_E_INTERNAL;
    }
    uint64_t outputBytes = inputBytes * param.rankSize;

    uint64_t inputAddress = reinterpret_cast<uint64_t>(param.inputPtr);
    uint64_t outputAddress = reinterpret_cast<uint64_t>(param.outputPtr);
    uint64_t rankOutputOffset =
        static_cast<uint64_t>(param.myRank) * inputBytes;
    bool inputAlreadyPlaced =
        outputAddress <= UINT64_MAX - rankOutputOffset &&
        inputAddress == outputAddress + rankOutputOffset;
    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    CcuResult ccuRet = CCU_SUCCESS;
    if (inputAddress == resCtx.cachedInputAddress &&
        inputBytes == resCtx.cachedInputBytes) {
        inputToken = resCtx.cachedInputToken;
    } else {
        ccuRet = HcommCcuGetMemToken(
            inputAddress, inputBytes, &inputToken);
        if (ccuRet != CCU_SUCCESS) {
            HCCL_ERROR("failed to get input token, ret=%d", ccuRet);
            return ConvertCcuResult(ccuRet);
        }
    }
    if (outputAddress == resCtx.cachedOutputAddress &&
        outputBytes == resCtx.cachedOutputBytes) {
        outputToken = resCtx.cachedOutputToken;
    } else {
        ccuRet = HcommCcuGetMemToken(
            outputAddress, outputBytes, &outputToken);
        if (ccuRet != CCU_SUCCESS) {
            HCCL_ERROR("failed to get output token, ret=%d", ccuRet);
            return ConvertCcuResult(ccuRet);
        }
    }

    if (resCtx.rank12Nhr) {
        uint64_t totalLinkLoad =
            static_cast<uint64_t>(
                resCtx.kernelLinkLoads[0]) +
            resCtx.kernelLinkLoads[1];
        if (totalLinkLoad == 0) {
            HCCL_ERROR("invalid rank12 NHR link load");
            return HCCL_E_INTERNAL;
        }
        uint64_t firstAxisBytes =
            inputBytes *
            resCtx.kernelLinkLoads[0] /
            totalLinkLoad /
            RANK12_DUAL_RAIL_ALIGNMENT *
            RANK12_DUAL_RAIL_ALIGNMENT;
        uint64_t secondAxisBytes =
            inputBytes - firstAxisBytes;
        if (firstAxisBytes == 0 ||
            secondAxisBytes == 0 ||
            firstAxisBytes > MAX_DATA_SIZE ||
            secondAxisBytes > MAX_DATA_SIZE) {
            HCCL_ERROR("invalid rank12 NHR split, input=%llu "
                       "axis0=%llu axis1=%llu",
                static_cast<unsigned long long>(inputBytes),
                static_cast<unsigned long long>(firstAxisBytes),
                static_cast<unsigned long long>(secondAxisBytes));
            return HCCL_E_INTERNAL;
        }

        KernelTaskArgSet taskArgs{};
        std::array<uint64_t, AlgResourceCtx::MAX_KERNEL_COUNT>
            axisOffsets{0, firstAxisBytes};
        std::array<uint64_t, AlgResourceCtx::MAX_KERNEL_COUNT>
            axisSizes{firstAxisBytes, secondAxisBytes};
        for (uint32_t index = 0;
             index < resCtx.kernelCount; ++index) {
            taskArgs[index] = {
                inputAddress,
                outputAddress,
                inputToken,
                outputToken,
                inputBytes,
                axisOffsets[index],
                axisSizes[index],
                inputAlreadyPlaced ? 0 : axisSizes[index],
            };
        }
        return LaunchKernelGroup(param, resCtx, taskArgs);
    }

    if (resCtx.rank12SplitRelay) {
        return LaunchRank12Pipeline(param, resCtx,
            inputAddress, outputAddress, inputToken,
            outputToken, inputBytes, inputAlreadyPlaced);
    }

    uint64_t processedBytes = 0;
    if (resCtx.rank16MixedRelay) {
        while (processedBytes < inputBytes) {
            uint64_t sliceBytes = std::min<uint64_t>(
                MAX_DATA_SIZE, inputBytes - processedBytes);
            uint64_t directBytes =
                sliceBytes * RANK16_DIRECT_NUMERATOR /
                RANK16_DIRECT_DENOMINATOR;
            directBytes =
                directBytes / RANK16_SPLIT_ALIGNMENT *
                RANK16_SPLIT_ALIGNMENT;
            uint64_t relayBytes = sliceBytes - directBytes;
            if (sliceBytes == 0 || relayBytes == 0) {
                HCCL_ERROR("invalid rank16 mixed split, slice=%llu "
                           "direct=%llu relay=%llu",
                    static_cast<unsigned long long>(sliceBytes),
                    static_cast<unsigned long long>(directBytes),
                    static_cast<unsigned long long>(relayBytes));
                return HCCL_E_INTERNAL;
            }

            uint64_t outputOffset =
                rankOutputOffset + processedBytes;
            uint64_t relayOutputOffset =
                static_cast<uint64_t>(
                    resCtx.rank16RelayPeerRank) * inputBytes +
                processedBytes + directBytes;
            std::array<uint64_t, 4> meshGroupSize =
                GetGroupCopySize(inputAlreadyPlaced ? 0 : sliceBytes,
                    LOCAL_COPY_SLICE,
                    LOCAL_COPY_LOOP_COUNT);
            std::array<uint64_t, 4> closGroupSize =
                GetGroupCopySize(directBytes,
                    GROUP_BROADCAST_SLICE,
                    GROUP_BROADCAST_LOOP_COUNT);

            KernelTaskArgSet phaseZeroTaskArgs{};
            phaseZeroTaskArgs[resCtx.rank16MeshKernelIndex] = {
                inputAddress + processedBytes,
                outputAddress,
                inputToken,
                outputToken,
                outputOffset,
                directBytes,
                relayBytes,
                relayOutputOffset,
                meshGroupSize[0],
                meshGroupSize[1],
                meshGroupSize[2],
                meshGroupSize[3],
                RANK16_PHASE_ZERO,
            };
            phaseZeroTaskArgs[resCtx.rank16ClosKernelIndex] = {
                inputAddress + processedBytes,
                outputAddress,
                inputToken,
                outputToken,
                outputOffset,
                directBytes,
                relayBytes,
                relayOutputOffset,
                closGroupSize[1],
                RANK16_PHASE_ZERO,
            };
            KernelTaskArgSet phaseOneTaskArgs =
                phaseZeroTaskArgs;
            phaseOneTaskArgs[resCtx.rank16MeshKernelIndex]
                [AlgResourceCtx::RANK16_MESH_TASK_ARG_COUNT - 1] =
                    RANK16_PHASE_ONE;
            phaseOneTaskArgs[resCtx.rank16ClosKernelIndex]
                [AlgResourceCtx::RANK16_CLOS_TASK_ARG_COUNT - 1] =
                    RANK16_PHASE_ONE;
            CHK_RET(LaunchRank16MixedPhases(param, resCtx,
                phaseZeroTaskArgs, phaseOneTaskArgs));
            processedBytes += sliceBytes;
        }
        return HCCL_SUCCESS;
    }

    bool useFineLocalCopy =
        resCtx.rankSize == 12 &&
        resCtx.cachedInputBytes >= FINE_LOCAL_COPY_MIN_BYTES;
    uint64_t localCopySlice =
        useFineLocalCopy ? FINE_LOCAL_COPY_SLICE : LOCAL_COPY_SLICE;
    uint64_t localCopyLoopCount =
        useFineLocalCopy ?
            FINE_LOCAL_COPY_LOOP_COUNT : LOCAL_COPY_LOOP_COUNT;
    bool alignLocalCopyToLoopStride =
        useFineLocalCopy;
    uint64_t localCopyAlignment =
        alignLocalCopyToLoopStride ?
            localCopySlice * localCopyLoopCount : localCopySlice;
    bool usesGroupFanout = false;
    bool supportsFusedSlices =
        fusedKernelCount == resCtx.kernelCount;
    uint32_t localCopyKernelCount = 0;
    uint32_t localCopyKernelIndex = 0;
    for (uint32_t index = 0; index < resCtx.kernelCount; ++index) {
        usesGroupFanout =
            usesGroupFanout ||
            (!resCtx.rank12DirectLocalCopy &&
                resCtx.taskArgCounts[index] ==
                    AlgResourceCtx::HYBRID_TASK_ARG_COUNT);
        if (resCtx.taskArgCounts[index] ==
                AlgResourceCtx::FULL_TASK_ARG_COUNT ||
            resCtx.taskArgCounts[index] ==
                AlgResourceCtx::FUSED_FULL_TASK_ARG_COUNT ||
            resCtx.taskArgCounts[index] ==
                AlgResourceCtx::DUAL_RAIL_FULL_TASK_ARG_COUNT ||
            resCtx.taskArgCounts[index] ==
                AlgResourceCtx::
                    DUAL_RAIL_FUSED_FULL_TASK_ARG_COUNT ||
            resCtx.taskArgCounts[index] ==
                AlgResourceCtx::
                    DIRECT_LOCAL_COPY_TASK_ARG_COUNT ||
            resCtx.taskArgCounts[index] ==
                AlgResourceCtx::
                    FUSED_DIRECT_LOCAL_COPY_TASK_ARG_COUNT) {
            ++localCopyKernelCount;
            localCopyKernelIndex = index;
        }
    }
    while (processedBytes < inputBytes) {
        bool fuseCurrentOperation =
            supportsFusedSlices &&
            (resCtx.rankSize == 4 || resCtx.rankSize == 12) &&
            inputBytes == FUSED_512_BYTES &&
            processedBytes == 0;
        uint64_t sliceBytes = fuseCurrentOperation ?
            FUSED_DIRECT_SLICE_BYTES :
            std::min<uint64_t>(
                MAX_DATA_SIZE, inputBytes - processedBytes);
        uint64_t secondSliceBytes = fuseCurrentOperation ?
            FUSED_SECOND_SLICE_BYTES : 0;
        uint64_t operationBytes = sliceBytes + secondSliceBytes;
        if (operationBytes == 0 ||
            operationBytes > inputBytes - processedBytes) {
            HCCL_ERROR("invalid fused operation size");
            return HCCL_E_INTERNAL;
        }
        uint64_t outputOffset = rankOutputOffset + processedBytes;
        uint64_t closPrimaryBytes = 0;
        uint64_t closAlternateBytes = 0;
        if (resCtx.rank12DualRail) {
            uint64_t totalRailWeight =
                static_cast<uint64_t>(
                    resCtx.rank12PrimaryRailWeight) +
                resCtx.rank12AlternateRailWeight;
            closPrimaryBytes =
                sliceBytes *
                resCtx.rank12PrimaryRailWeight /
                totalRailWeight;
            closPrimaryBytes =
                closPrimaryBytes /
                RANK12_DUAL_RAIL_ALIGNMENT *
                RANK12_DUAL_RAIL_ALIGNMENT;
            if (closPrimaryBytes == 0 ||
                closPrimaryBytes >= sliceBytes) {
                HCCL_ERROR("invalid rank12 dual-rail split, slice=%llu "
                           "primary=%llu weights=%u/%u",
                    static_cast<unsigned long long>(sliceBytes),
                    static_cast<unsigned long long>(
                        closPrimaryBytes),
                    resCtx.rank12PrimaryRailWeight,
                    resCtx.rank12AlternateRailWeight);
                return HCCL_E_INTERNAL;
            }
            closAlternateBytes =
                sliceBytes - closPrimaryBytes;
        }
        uint64_t localCopyBytes =
            inputAlreadyPlaced ? 0 : operationBytes;
        std::array<uint64_t, AlgResourceCtx::MAX_KERNEL_COUNT>
            localCopyOffsets{};
        std::array<uint64_t, AlgResourceCtx::MAX_KERNEL_COUNT>
            localCopySizes{};
        if (usesGroupFanout) {
            for (uint32_t index = 0; index < resCtx.kernelCount; ++index) {
                if (resCtx.taskArgCounts[index] ==
                        AlgResourceCtx::GROUP_BROADCAST_TASK_ARG_COUNT ||
                    resCtx.taskArgCounts[index] ==
                        AlgResourceCtx::HYBRID_TASK_ARG_COUNT) {
                    localCopySizes[index] = operationBytes;
                }
            }
        } else if (resCtx.rank12DirectLocalCopy) {
            uint64_t firstCopyBytes =
                localCopyBytes / 2;
            firstCopyBytes =
                firstCopyBytes /
                localCopyAlignment *
                localCopyAlignment;
            localCopySizes[0] = firstCopyBytes;
            localCopyOffsets[1] = firstCopyBytes;
            localCopySizes[1] =
                localCopyBytes - firstCopyBytes;
        } else if (localCopyKernelCount == 1) {
            localCopySizes[localCopyKernelIndex] = localCopyBytes;
        } else {
            uint64_t totalLinkLoad =
                static_cast<uint64_t>(resCtx.kernelLinkLoads[0]) +
                resCtx.kernelLinkLoads[1];
            uint64_t firstShare =
                localCopyBytes * resCtx.kernelLinkLoads[1] /
                totalLinkLoad;
            uint64_t firstCopyBytes = 0;
            if (localCopyBytes >= localCopyAlignment) {
                uint64_t alignedLimit =
                    localCopyBytes -
                    localCopyBytes % localCopyAlignment;
                firstCopyBytes = std::min<uint64_t>(
                    (firstShare + localCopyAlignment / 2) /
                        localCopyAlignment * localCopyAlignment,
                    alignedLimit);
            }
            localCopySizes[0] = firstCopyBytes;
            localCopyOffsets[1] = firstCopyBytes;
            localCopySizes[1] = localCopyBytes - firstCopyBytes;
        }

        KernelTaskArgSet taskArgs{};
        for (uint32_t index = 0; index < resCtx.kernelCount; ++index) {
            bool groupBroadcast =
                !resCtx.rank12DirectLocalCopy &&
                (resCtx.taskArgCounts[index] ==
                        AlgResourceCtx::
                            GROUP_BROADCAST_TASK_ARG_COUNT ||
                    resCtx.taskArgCounts[index] ==
                        AlgResourceCtx::HYBRID_TASK_ARG_COUNT);
            std::array<uint64_t, 4> groupCopySize =
                GetGroupCopySize(localCopySizes[index],
                    groupBroadcast ? GROUP_BROADCAST_SLICE :
                        localCopySlice,
                    groupBroadcast ? GROUP_BROADCAST_LOOP_COUNT :
                        localCopyLoopCount);
            if (!resCtx.rank12DirectLocalCopy &&
                resCtx.taskArgCounts[index] ==
                    AlgResourceCtx::
                        GROUP_BROADCAST_TASK_ARG_COUNT) {
                taskArgs[index] = {
                    inputAddress + processedBytes,
                    outputAddress,
                    inputToken,
                    outputToken,
                    outputOffset,
                    groupCopySize[0],
                    groupCopySize[1],
                    groupCopySize[2],
                    groupCopySize[3],
                };
            } else if (resCtx.taskArgCounts[index] ==
                AlgResourceCtx::HYBRID_TASK_ARG_COUNT) {
                taskArgs[index] = {
                    inputAddress + processedBytes,
                    outputAddress,
                    inputToken,
                    outputToken,
                    outputOffset,
                    sliceBytes,
                    groupCopySize[0],
                    groupCopySize[1],
                    groupCopySize[2],
                    groupCopySize[3],
                };
            } else if (resCtx.taskArgCounts[index] ==
                AlgResourceCtx::BASE_TASK_ARG_COUNT) {
                taskArgs[index] = {
                    inputAddress + processedBytes,
                    outputAddress,
                    inputToken,
                    outputToken,
                    outputOffset,
                    sliceBytes,
                };
            } else if (resCtx.rank12DualRail &&
                resCtx.taskArgCounts[index] ==
                    AlgResourceCtx::DUAL_RAIL_BASE_TASK_ARG_COUNT) {
                taskArgs[index] = {
                    inputAddress + processedBytes,
                    outputAddress,
                    inputToken,
                    outputToken,
                    outputOffset,
                    sliceBytes,
                    closPrimaryBytes,
                    closAlternateBytes,
                };
            } else if (resCtx.rank12DirectLocalCopy &&
                resCtx.taskArgCounts[index] ==
                    AlgResourceCtx::
                        DIRECT_LOCAL_COPY_TASK_ARG_COUNT) {
                taskArgs[index] = {
                    inputAddress + processedBytes,
                    outputAddress,
                    inputToken,
                    outputToken,
                    outputOffset,
                    sliceBytes,
                    localCopyOffsets[index],
                    localCopySizes[index],
                };
            } else if (!resCtx.rank12DualRail &&
                resCtx.taskArgCounts[index] ==
                    AlgResourceCtx::FUSED_BASE_TASK_ARG_COUNT) {
                taskArgs[index] = {
                    inputAddress + processedBytes,
                    outputAddress,
                    inputToken,
                    outputToken,
                    outputOffset,
                    sliceBytes,
                    secondSliceBytes,
                };
            } else if (resCtx.rank12DualRail &&
                resCtx.taskArgCounts[index] ==
                    AlgResourceCtx::
                        DUAL_RAIL_FUSED_BASE_TASK_ARG_COUNT) {
                taskArgs[index] = {
                    inputAddress + processedBytes,
                    outputAddress,
                    inputToken,
                    outputToken,
                    outputOffset,
                    sliceBytes,
                    closPrimaryBytes,
                    closAlternateBytes,
                    secondSliceBytes,
                };
            } else if (resCtx.rank12DirectLocalCopy &&
                resCtx.taskArgCounts[index] ==
                    AlgResourceCtx::
                        FUSED_DIRECT_LOCAL_COPY_TASK_ARG_COUNT) {
                taskArgs[index] = {
                    inputAddress + processedBytes,
                    outputAddress,
                    inputToken,
                    outputToken,
                    outputOffset,
                    sliceBytes,
                    localCopyOffsets[index],
                    localCopySizes[index],
                    secondSliceBytes,
                };
            } else if (!resCtx.rank12DualRail &&
                resCtx.taskArgCounts[index] ==
                    AlgResourceCtx::FUSED_FULL_TASK_ARG_COUNT) {
                taskArgs[index] = {
                    inputAddress + processedBytes,
                    outputAddress,
                    inputToken,
                    outputToken,
                    outputOffset,
                    sliceBytes,
                    localCopyOffsets[index],
                    groupCopySize[0],
                    groupCopySize[1],
                    groupCopySize[2],
                    groupCopySize[3],
                    secondSliceBytes,
                };
            } else if (resCtx.rank12DualRail &&
                resCtx.taskArgCounts[index] ==
                    AlgResourceCtx::
                        DUAL_RAIL_FUSED_FULL_TASK_ARG_COUNT) {
                taskArgs[index] = {
                    inputAddress + processedBytes,
                    outputAddress,
                    inputToken,
                    outputToken,
                    outputOffset,
                    sliceBytes,
                    closPrimaryBytes,
                    closAlternateBytes,
                    groupCopySize[0],
                    groupCopySize[1],
                    groupCopySize[2],
                    groupCopySize[3],
                    secondSliceBytes,
                };
            } else if (resCtx.rank12DualRail &&
                resCtx.taskArgCounts[index] ==
                    AlgResourceCtx::DUAL_RAIL_FULL_TASK_ARG_COUNT) {
                taskArgs[index] = {
                    inputAddress + processedBytes,
                    outputAddress,
                    inputToken,
                    outputToken,
                    outputOffset,
                    sliceBytes,
                    closPrimaryBytes,
                    closAlternateBytes,
                    groupCopySize[0],
                    groupCopySize[1],
                    groupCopySize[2],
                    groupCopySize[3],
                };
            } else {
                taskArgs[index] = {
                    inputAddress + processedBytes,
                    outputAddress,
                    inputToken,
                    outputToken,
                    outputOffset,
                    sliceBytes,
                    localCopyOffsets[index],
                    groupCopySize[0],
                    groupCopySize[1],
                    groupCopySize[2],
                    groupCopySize[3],
                };
            }
        }
        CHK_RET(LaunchKernelGroup(param, resCtx, taskArgs));
        processedBytes += operationBytes;
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
