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
#include <iterator>
#include <limits>
#include <vector>

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>
#include <hccl/hcomm_primitives.h>

#include "ccu_kernel.h"
#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {
constexpr uint64_t TRANSFER_ALIGNMENT_BYTES = 128;
constexpr uint64_t SAFE_TRANSFER_BYTES = static_cast<uint64_t>(MAX_DATA_SIZE) - TRANSFER_ALIGNMENT_BYTES;
constexpr uint64_t SMALL_ONE_SHOT_BYTES = 512ULL * 1024;
constexpr uint64_t LARGE_512_MIB_BYTES = 512ULL * 1024 * 1024;
constexpr uint64_t LARGE_400_MIB_PLUS_4_BYTES = 400ULL * 1024 * 1024 + 4;
constexpr uint32_t FOUR_RANK_NHR_SIZE = 4;
constexpr uint32_t ASYMMETRIC_RANK_SIZE = 12;
constexpr uint32_t RANK12_CLOS_TREE_SLOT_NUM = 4;
constexpr uint32_t RANK12_CLOS_TREE_TASK_ARG_NUM = 7;
constexpr uint32_t RANK16_CLOS_RSAG_SIZE = 16;
constexpr uint32_t MESH_NET_LAYER = 0;
constexpr uint32_t CLOS_NET_LAYER = 1;
constexpr uint32_t FOUR_RANK_NHR_TASK_ARG_NUM = 7;
constexpr uint32_t RS_WAVEFRONT_STRIPE_NUM = 4;
constexpr uint32_t DIRECT_BUFFER_RS_STRIPE_NUM = 16;
constexpr uint32_t ASYMMETRIC_DIRECT_BUFFER_RS_STRIPE_NUM = 8;
constexpr uint32_t DIRECT_TASK_ARG_NUM = 8;
constexpr uint32_t DIRECT_BUFFER_TASK_ARG_NUM = 11;
constexpr uint32_t LAYER_THREAD_NOTIFY = 0;
constexpr size_t ASCEND950_DIE_NUM = 2;
constexpr uint32_t MIXED_NET_LAYER = std::numeric_limits<uint32_t>::max();

bool IsSupportedDirectBufferRankSize(uint32_t rankSize)
{
    return rankSize == ASYMMETRIC_RANK_SIZE || rankSize == MAX_RANK_SIZE;
}

uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return value / divisor + (value % divisor != 0 ? 1 : 0);
}

uint64_t AlignDown(uint64_t value, uint64_t alignment)
{
    return value - value % alignment;
}

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return CeilDiv(value, alignment) * alignment;
}

HcclResult LaunchGroupedKernel(ThreadHandle mainThread, const AlgResourceCtx &resource,
    size_t kernelIndex, const uint64_t *taskArgs)
{
    CHK_PRT_RET(kernelIndex >= resource.ccuKernels.size(),
        HCCL_ERROR("Invalid pair kernel index[%llu]", static_cast<unsigned long long>(kernelIndex)),
        HCCL_E_INTERNAL);
    CHK_RET_CCU(HcommCcuKernelLaunch(mainThread, resource.ccuKernels[kernelIndex],
        taskArgs, DIRECT_TASK_ARG_NUM));
    return HCCL_SUCCESS;
}

HcclResult LaunchParallelRsKernel(ThreadHandle thread, const AlgResourceCtx &resource,
    size_t kernelIndex, const uint64_t *taskArgs)
{
    CHK_PRT_RET(kernelIndex >= resource.parallelRsKernels.size(),
        HCCL_ERROR("Invalid parallel RS kernel index[%llu]",
            static_cast<unsigned long long>(kernelIndex)), HCCL_E_INTERNAL);
    CHK_RET_CCU(HcommCcuKernelLaunch(thread, resource.parallelRsKernels[kernelIndex],
        taskArgs, DIRECT_TASK_ARG_NUM));
    return HCCL_SUCCESS;
}

HcclResult LaunchDirectBufferKernel(ThreadHandle thread,
    const std::vector<CcuKernelHandle> &kernels, size_t kernelIndex,
    const uint64_t *taskArgs)
{
    CHK_PRT_RET(kernelIndex >= kernels.size() || kernels[kernelIndex] == 0,
        HCCL_ERROR("Invalid direct-buffer kernel index[%llu]",
            static_cast<unsigned long long>(kernelIndex)), HCCL_E_INTERNAL);
    CHK_RET_CCU(HcommCcuKernelLaunch(thread, kernels[kernelIndex],
        taskArgs, DIRECT_BUFFER_TASK_ARG_NUM));
    return HCCL_SUCCESS;
}

HcclResult LaunchRank12ClosTreeKernel(ThreadHandle thread,
    CcuKernelHandle kernel, const uint64_t *taskArgs)
{
    CHK_PRT_RET(kernel == 0,
        HCCL_ERROR("Invalid rank-12 Clos tree kernel handle"),
        HCCL_E_INTERNAL);
    CHK_RET_CCU(HcommCcuKernelLaunch(thread,
        kernel, taskArgs,
        RANK12_CLOS_TREE_TASK_ARG_NUM));
    return HCCL_SUCCESS;
}

void BuildTaskArgs(uint64_t cclAddr, uint64_t cclToken, uint64_t dataBytes,
    uint64_t slotStrideBytes, uint64_t mySliceOffset, uint64_t mySliceBytes,
    uint64_t resultOffsetBytes, DirectKernelPhase phase,
    uint64_t (&taskArgs)[DIRECT_TASK_ARG_NUM])
{
    uint64_t values[DIRECT_TASK_ARG_NUM] = {
        cclAddr,
        cclToken,
        dataBytes,
        slotStrideBytes,
        mySliceOffset,
        mySliceBytes,
        resultOffsetBytes,
        static_cast<uint64_t>(phase),
    };
    std::copy(std::begin(values), std::end(values), std::begin(taskArgs));
}

void BuildDirectBufferTaskArgs(uint64_t inputAddr, uint64_t inputToken,
    uint64_t outputAddr, uint64_t outputToken, uint64_t scratchAddr,
    uint64_t scratchToken, uint64_t regularSliceBytes, uint64_t lastSliceBytes,
    uint64_t regularStripeBytes, uint64_t tailStripeBytes, DirectKernelPhase phase,
    uint64_t (&taskArgs)[DIRECT_BUFFER_TASK_ARG_NUM])
{
    uint64_t values[DIRECT_BUFFER_TASK_ARG_NUM] = {
        inputAddr,
        inputToken,
        outputAddr,
        outputToken,
        scratchAddr,
        scratchToken,
        regularSliceBytes,
        lastSliceBytes,
        regularStripeBytes,
        tailStripeBytes,
        static_cast<uint64_t>(phase),
    };
    std::copy(std::begin(values), std::end(values), std::begin(taskArgs));
}

HcclResult LaunchAllGroups(const OpParam &param, const AlgResourceCtx &resource,
    uint64_t cclAddr, uint64_t cclToken, uint64_t dataBytes,
    uint64_t slotStrideBytes, uint64_t mySliceOffset, uint64_t mySliceBytes,
    uint64_t resultOffsetBytes, DirectKernelPhase phase)
{
    uint64_t taskArgs[DIRECT_TASK_ARG_NUM]{};
    BuildTaskArgs(cclAddr, cclToken, dataBytes, slotStrideBytes,
        mySliceOffset, mySliceBytes, resultOffsetBytes, phase, taskArgs);
    for (size_t kernelIndex = 0; kernelIndex < resource.ccuKernels.size(); ++kernelIndex) {
        CHK_RET(LaunchGroupedKernel(param.cpuThread, resource, kernelIndex, taskArgs));
    }
    return HCCL_SUCCESS;
}

bool CanLaunchParallelLayerAllGather(const AlgResourceCtx &resource)
{
    return resource.ccuKernels.size() == ASCEND950_DIE_NUM &&
        resource.kernelNetLayers.size() == resource.ccuKernels.size() &&
        resource.threads.size() >= ASCEND950_DIE_NUM &&
        resource.kernelNetLayers[0] != MIXED_NET_LAYER &&
        resource.kernelNetLayers[1] != MIXED_NET_LAYER &&
        resource.kernelNetLayers[0] != resource.kernelNetLayers[1];
}

bool CanLaunchParallelLayerReduceScatter(const AlgResourceCtx &resource)
{
    return CanLaunchParallelLayerAllGather(resource) &&
        resource.parallelRsKernels.size() == ASCEND950_DIE_NUM;
}

bool CanLaunchFusedOneShot(const OpParam &param, const AlgResourceCtx &resource)
{
    return IsSupportedDirectBufferRankSize(param.rankSize) &&
        CanLaunchParallelLayerReduceScatter(resource);
}

bool CanLaunchFourRankDirectSourceOneShot(
    const OpParam &param, const AlgResourceCtx &resource)
{
    return param.rankSize == FOUR_RANK_NHR_SIZE &&
        resource.ccuKernels.size() == 1 &&
        resource.kernelNetLayers.size() == 1 &&
        resource.threads.size() == 1 &&
        resource.perf512KiBKernels.size() == 1 &&
        resource.perf512KiBKernels[0] != 0;
}

HcclResult LaunchFusedOneShotCommunication(const OpParam &param,
    const AlgResourceCtx &resource, uint64_t cclAddr, uint64_t cclToken,
    uint64_t dataBytes, uint64_t slotStrideBytes)
{
    uint64_t taskArgs[DIRECT_TASK_ARG_NUM]{};
    BuildTaskArgs(cclAddr, cclToken, dataBytes, slotStrideBytes,
        0, 0, 0, DirectKernelPhase::FUSED_ONESHOT_COMMUNICATION, taskArgs);

    ThreadHandle workerThread = resource.threads[1];
    // Both die-local missions first publish every edge, then consume every
    // edge, so endpoint-specific die enumeration cannot form a wait cycle.
    // START and DONE are fully consumed before the thread notify is reused.
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        param.cpuThread, workerThread, LAYER_THREAD_NOTIFY)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        workerThread, LAYER_THREAD_NOTIFY, CUSTOM_TIMEOUT)));

    CHK_RET(LaunchParallelRsKernel(param.cpuThread, resource, 0, taskArgs));
    CHK_RET(LaunchParallelRsKernel(workerThread, resource, 1, taskArgs));

    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        workerThread, param.cpuThread, LAYER_THREAD_NOTIFY)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        param.cpuThread, LAYER_THREAD_NOTIFY, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult LaunchParallelDirectBufferPhase(const OpParam &param,
    const AlgResourceCtx &resource,
    const std::vector<CcuKernelHandle> &kernels,
    const uint64_t *taskArgs)
{
    CHK_PRT_RET(kernels.size() != ASCEND950_DIE_NUM ||
            resource.threads.size() < ASCEND950_DIE_NUM,
        HCCL_ERROR("F015 direct-buffer path requires two kernels and two threads"),
        HCCL_E_INTERNAL);
    ThreadHandle workerThread = resource.threads[1];

    // Preserve the validated registration/submission pairing on both dies.
    // START and DONE are fully consumed before this helper can be reused for
    // the next phase, so the single thread-notify resource cannot be hit twice.
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        param.cpuThread, workerThread, LAYER_THREAD_NOTIFY)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        workerThread, LAYER_THREAD_NOTIFY, CUSTOM_TIMEOUT)));

    CHK_RET(LaunchDirectBufferKernel(param.cpuThread, kernels, 0, taskArgs));
    CHK_RET(LaunchDirectBufferKernel(workerThread, kernels, 1, taskArgs));

    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        workerThread, param.cpuThread, LAYER_THREAD_NOTIFY)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        param.cpuThread, LAYER_THREAD_NOTIFY, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult LaunchParallelLayerReduceScatter(const OpParam &param,
    const AlgResourceCtx &resource, uint64_t cclAddr, uint64_t cclToken,
    uint64_t regularStripeBytes, uint64_t tailStripeBytes,
    uint64_t mySliceOffset, uint64_t mySliceBytes, uint64_t resultOffsetBytes)
{
    uint64_t partialArgs[DIRECT_TASK_ARG_NUM]{};
    uint64_t mergeArgs[DIRECT_TASK_ARG_NUM]{};
    BuildTaskArgs(cclAddr, cclToken, regularStripeBytes, tailStripeBytes,
        mySliceOffset, mySliceBytes, resultOffsetBytes,
        DirectKernelPhase::PARALLEL_REDUCE_SCATTER, partialArgs);
    BuildTaskArgs(cclAddr, cclToken, 0, 0,
        mySliceOffset, mySliceBytes, resultOffsetBytes,
        DirectKernelPhase::PARALLEL_REDUCE_MERGE, mergeArgs);

    ThreadHandle workerThread = resource.threads[1];

    // Keep the registration/submission pairing used by the validated F004
    // two-layer AllGather: kernel 0 must be enqueued before kernel 1. The main
    // stream reaches DONE_WAIT only after its own partial mission; the worker
    // records DONE only after its partial mission, so the merge sees both
    // disjoint partials complete.
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        param.cpuThread, workerThread, LAYER_THREAD_NOTIFY)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        workerThread, LAYER_THREAD_NOTIFY, CUSTOM_TIMEOUT)));

    CHK_RET(LaunchParallelRsKernel(param.cpuThread, resource, 0, partialArgs));
    CHK_RET(LaunchParallelRsKernel(workerThread, resource, 1, partialArgs));

    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        workerThread, param.cpuThread, LAYER_THREAD_NOTIFY)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        param.cpuThread, LAYER_THREAD_NOTIFY, CUSTOM_TIMEOUT)));

    CHK_RET(LaunchParallelRsKernel(param.cpuThread, resource, 0, mergeArgs));
    return HCCL_SUCCESS;
}

HcclResult LaunchDirectAllGather(const OpParam &param, const AlgResourceCtx &resource,
    uint64_t cclAddr, uint64_t cclToken, uint64_t windowBytes,
    uint64_t mySliceOffset, uint64_t mySliceBytes, uint64_t resultOffsetBytes)
{
    if (!CanLaunchParallelLayerAllGather(resource)) {
        CHK_RET(LaunchAllGroups(param, resource, cclAddr, cclToken, windowBytes,
            0, mySliceOffset, mySliceBytes, resultOffsetBytes,
            DirectKernelPhase::DIRECT_ALL_GATHER_RECORD));
        CHK_RET(LaunchAllGroups(param, resource, cclAddr, cclToken, windowBytes,
            0, mySliceOffset, mySliceBytes, resultOffsetBytes,
            DirectKernelPhase::DIRECT_ALL_GATHER_WAIT));
        return HCCL_SUCCESS;
    }

    uint64_t recordArgs[DIRECT_TASK_ARG_NUM]{};
    uint64_t waitArgs[DIRECT_TASK_ARG_NUM]{};
    BuildTaskArgs(cclAddr, cclToken, windowBytes, 0,
        mySliceOffset, mySliceBytes, resultOffsetBytes,
        DirectKernelPhase::DIRECT_ALL_GATHER_RECORD, recordArgs);
    BuildTaskArgs(cclAddr, cclToken, windowBytes, 0,
        mySliceOffset, mySliceBytes, resultOffsetBytes,
        DirectKernelPhase::DIRECT_ALL_GATHER_WAIT, waitArgs);

    ThreadHandle workerThread = resource.threads[1];

    // Follow the native multi-thread template's submission order exactly:
    // main records START before the worker waits, then kernel 0 is submitted
    // before kernel 1.  CCU missions are paired in registration/submission
    // order, so submitting the complete worker program ahead of kernel 0 can
    // produce a valid TS graph while pairing the two CCU missions incorrectly.
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        param.cpuThread, workerThread, LAYER_THREAD_NOTIFY)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        workerThread, LAYER_THREAD_NOTIFY, CUSTOM_TIMEOUT)));

    CHK_RET(LaunchGroupedKernel(param.cpuThread, resource, 0, recordArgs));
    CHK_RET(LaunchGroupedKernel(workerThread, resource, 1, recordArgs));
    CHK_RET(LaunchGroupedKernel(param.cpuThread, resource, 0, waitArgs));
    CHK_RET(LaunchGroupedKernel(workerThread, resource, 1, waitArgs));

    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        workerThread, param.cpuThread, LAYER_THREAD_NOTIFY)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        param.cpuThread, LAYER_THREAD_NOTIFY, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult LaunchFirstGroup(const OpParam &param, const AlgResourceCtx &resource,
    uint64_t cclAddr, uint64_t cclToken, uint64_t dataBytes,
    uint64_t slotStrideBytes, DirectKernelPhase phase)
{
    uint64_t taskArgs[DIRECT_TASK_ARG_NUM]{};
    BuildTaskArgs(cclAddr, cclToken, dataBytes, slotStrideBytes,
        0, 0, 0, phase, taskArgs);
    return LaunchGroupedKernel(param.cpuThread, resource, 0, taskArgs);
}

uint64_t DirectWindowCapacity(uint64_t bufferBytes, uint32_t rankSize)
{
    uint64_t divisor = static_cast<uint64_t>(rankSize) + 1;
    uint64_t quotient = bufferBytes / divisor;
    uint64_t remainder = bufferBytes % divisor;
    uint64_t bufferCapacity = quotient * rankSize + remainder * rankSize / divisor;
    uint64_t alignment = static_cast<uint64_t>(rankSize) * TRANSFER_ALIGNMENT_BYTES;
    return AlignDown(std::min(SAFE_TRANSFER_BYTES, bufferCapacity), alignment);
}

uint64_t DirectBufferWindowCapacity(uint64_t bufferBytes, uint32_t rankSize)
{
    if (rankSize == 0) {
        return 0;
    }

    // Only the group-1 partial occupies CCL memory. Reserve the maximum
    // possible aligned tail skew so every rank's owner slice remains within
    // both the scratch buffer and the 256 MiB primitive limit.
    uint64_t sliceLimit = AlignDown(
        std::min(bufferBytes, SAFE_TRANSFER_BYTES), TRANSFER_ALIGNMENT_BYTES);
    uint64_t tailReserve = static_cast<uint64_t>(rankSize - 1) * TRANSFER_ALIGNMENT_BYTES;
    if (sliceLimit <= tailReserve) {
        return 0;
    }
    uint64_t regularCapacity = AlignDown(
        sliceLimit - tailReserve, TRANSFER_ALIGNMENT_BYTES);
    if (regularCapacity > std::numeric_limits<uint64_t>::max() / rankSize) {
        return 0;
    }
    return regularCapacity * rankSize;
}

void CalculateSlices(uint64_t windowBytes, uint32_t rankSize, uint32_t myRank,
    uint64_t &regularSliceBytes, uint64_t &lastSliceBytes,
    uint64_t &mySliceOffset, uint64_t &mySliceBytes,
    uint64_t &resultOffsetBytes, uint64_t &maxSliceBytes)
{
    regularSliceBytes = AlignDown(windowBytes / rankSize, TRANSFER_ALIGNMENT_BYTES);
    lastSliceBytes = windowBytes - regularSliceBytes * (rankSize - 1);
    mySliceOffset = regularSliceBytes * myRank;
    mySliceBytes = myRank + 1 == rankSize ? lastSliceBytes : regularSliceBytes;
    resultOffsetBytes = AlignUp(windowBytes, TRANSFER_ALIGNMENT_BYTES);
    maxSliceBytes = std::max(regularSliceBytes, lastSliceBytes);
}

bool HasPartialBufferOverlap(const OpParam &param, uint64_t totalBytes)
{
    if (param.inputPtr == param.outputPtr) {
        return false;
    }

    uintptr_t inputAddr = reinterpret_cast<uintptr_t>(param.inputPtr);
    uintptr_t outputAddr = reinterpret_cast<uintptr_t>(param.outputPtr);
    uintptr_t maxAddr = std::numeric_limits<uintptr_t>::max();
    if (totalBytes > maxAddr - inputAddr || totalBytes > maxAddr - outputAddr) {
        return true;
    }
    return inputAddr < outputAddr + totalBytes && outputAddr < inputAddr + totalBytes;
}

bool HasFourRankDirectSourceCapacity(
    const AlgResourceCtx &resource, uint64_t dataBytes, bool needsSnapshot)
{
    return needsSnapshot ? dataBytes <= resource.localBuffer.size / 2 :
        dataBytes <= resource.localBuffer.size;
}

HcclResult ExecFourRankDirectSourceOneShot(const OpParam &param,
    const AlgResourceCtx &resource, uint64_t cclAddr, uint64_t cclToken,
    uint64_t dataBytes, bool needsSnapshot)
{
    CHK_PRT_RET(!CanLaunchFourRankDirectSourceOneShot(param, resource) ||
            !HasFourRankDirectSourceCapacity(
                resource, dataBytes, needsSnapshot),
        HCCL_ERROR("Invalid F031 4-rank direct-source resources"),
        HCCL_E_INTERNAL);

    uint64_t inputAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    uint64_t outputAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    uint64_t sourceAddr = needsSnapshot ? cclAddr : inputAddr;
    uint64_t scratchAddr = needsSnapshot ? cclAddr + dataBytes : cclAddr;
    uint64_t sourceToken = cclToken;
    uint64_t outputToken = 0;

    // Resolve all tokens before the optional snapshot so an error cannot
    // leave a partial task sequence followed by a fallback algorithm.
    CHK_RET_CCU(HcommCcuGetMemToken(outputAddr, dataBytes, &outputToken));
    if (!needsSnapshot) {
        CHK_RET_CCU(HcommCcuGetMemToken(inputAddr, dataBytes, &sourceToken));
    }

    if (needsSnapshot) {
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
            param.cpuThread, resource.localBuffer.addr,
            param.inputPtr, dataBytes)));
    }

    // Phase 13 has a rank-4-only ABI and is statically excluded from the
    // rank-12 dedicated Kernel and rank-16 phase 14 graph.
    uint64_t taskArgs[DIRECT_TASK_ARG_NUM] = {
        sourceAddr,
        sourceToken,
        outputAddr,
        outputToken,
        scratchAddr,
        cclToken,
        dataBytes,
        static_cast<uint64_t>(
            DirectKernelPhase::FUSED_ONESHOT_ALL_REDUCE),
    };
    CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread,
        resource.perf512KiBKernels[0], taskArgs, DIRECT_TASK_ARG_NUM));
    return HCCL_SUCCESS;
}

bool CanLaunchRank12ClosTree(const OpParam &param,
    const AlgResourceCtx &resource, uint64_t dataBytes)
{
    if (param.rankSize != ASYMMETRIC_RANK_SIZE ||
        dataBytes != SMALL_ONE_SHOT_BYTES ||
        resource.perf512KiBKernels.size() != 1 ||
        resource.perf512KiBKernels[0] == 0 ||
        resource.ccuKernels.size() != ASCEND950_DIE_NUM ||
        resource.kernelNetLayers.size() != ASCEND950_DIE_NUM ||
        resource.threads.size() != ASCEND950_DIE_NUM) {
        return false;
    }

    size_t closCount = 0;
    size_t meshCount = 0;
    for (uint32_t layer : resource.kernelNetLayers) {
        if (layer == CLOS_NET_LAYER) {
            ++closCount;
        } else if (layer == MESH_NET_LAYER) {
            ++meshCount;
        }
    }
    return closCount == 1 && meshCount == 1;
}

bool HasRank12ClosTreeCapacity(const AlgResourceCtx &resource,
    uint64_t dataBytes, bool needsSnapshot)
{
    uint32_t slotNum = needsSnapshot ?
        RANK12_CLOS_TREE_SLOT_NUM : RANK12_CLOS_TREE_SLOT_NUM - 1;
    uint64_t slotStrideBytes = AlignUp(dataBytes, TRANSFER_ALIGNMENT_BYTES);
    return slotStrideBytes <= resource.localBuffer.size / slotNum;
}

HcclResult ExecRank12ClosTree(const OpParam &param,
    const AlgResourceCtx &resource, uint64_t cclAddr, uint64_t cclToken,
    uint64_t dataBytes, bool needsSnapshot)
{
    uint64_t inputAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    uint64_t outputAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    uint64_t sourceAddr = needsSnapshot ? cclAddr : inputAddr;
    uint64_t sourceToken = cclToken;
    uint64_t outputToken = 0;
    uint64_t slotStrideBytes = AlignUp(dataBytes, TRANSFER_ALIGNMENT_BYTES);
    uint64_t scratchAddr = cclAddr +
        (needsSnapshot ? slotStrideBytes : 0);

    // Resolve every token before submitting the optional snapshot.  An error
    // therefore cannot leave half of an algorithm followed by the fallback.
    CHK_RET_CCU(HcommCcuGetMemToken(outputAddr, dataBytes, &outputToken));
    if (!needsSnapshot) {
        CHK_RET_CCU(HcommCcuGetMemToken(inputAddr, dataBytes, &sourceToken));
    }

    if (needsSnapshot) {
        // Keep slot 0 immutable for the full mission.  The dedicated Kernel
        // starts its three temporary slots at slot 1, so in-place and either
        // direction of partial overlap have the same deterministic tree as
        // disjoint user buffers.
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
            param.cpuThread, resource.localBuffer.addr,
            param.inputPtr, dataBytes)));
    }

    uint64_t taskArgs[RANK12_CLOS_TREE_TASK_ARG_NUM] = {
        sourceAddr,
        sourceToken,
        outputAddr,
        outputToken,
        scratchAddr,
        cclToken,
        dataBytes,
    };
    return LaunchRank12ClosTreeKernel(
        param.cpuThread, resource.perf512KiBKernels[0], taskArgs);
}

bool CanLaunchRank16ClosRsagKernel(const OpParam &param,
    const AlgResourceCtx &resource, uint64_t dataBytes)
{
    if (param.rankSize != RANK16_CLOS_RSAG_SIZE ||
        dataBytes != SMALL_ONE_SHOT_BYTES ||
        resource.ccuKernels.size() != ASCEND950_DIE_NUM ||
        resource.kernelNetLayers.size() != ASCEND950_DIE_NUM ||
        resource.threads.size() != ASCEND950_DIE_NUM ||
        resource.perf512KiBKernels.size() != 1 ||
        resource.perf512KiBKernels[0] == 0) {
        return false;
    }

    size_t closCount = 0;
    size_t meshCount = 0;
    for (uint32_t layer : resource.kernelNetLayers) {
        if (layer == CLOS_NET_LAYER) {
            ++closCount;
        } else if (layer == MESH_NET_LAYER) {
            ++meshCount;
        }
    }
    return closCount == 1 && meshCount == 1;
}

bool HasRank16ClosRsagCapacity(
    const AlgResourceCtx &resource, uint64_t dataBytes)
{
    // The overlap-safe path uses D bytes for an immutable snapshot and four
    // D/8 lanes.  The disjoint fast path leaves [0, D) unused but preserves
    // the same scratch layout.  Keep the conservative 2D capacity gate.
    return dataBytes <= resource.localBuffer.size / 2;
}

HcclResult ExecRank16ClosRsag(const OpParam &param,
    const AlgResourceCtx &resource,
    uint64_t cclAddr, uint64_t cclToken, uint64_t dataBytes,
    bool needsSnapshot)
{
    uint64_t inputAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    uint64_t outputAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    uint64_t sourceAddr = needsSnapshot ? cclAddr : inputAddr;
    uint64_t sourceToken = cclToken;
    uint64_t outputToken = 0;

    // Match F042's online-proven direct-source boundary.  Resolve all tokens
    // before enqueueing the optional snapshot, then keep alias/partial-overlap
    // calls on the immutable CCL source while disjoint calls read user input.
    CHK_RET_CCU(HcommCcuGetMemToken(outputAddr, dataBytes, &outputToken));
    if (!needsSnapshot) {
        CHK_RET_CCU(HcommCcuGetMemToken(inputAddr, dataBytes, &sourceToken));
    }
    if (needsSnapshot) {
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
            param.cpuThread, resource.localBuffer.addr,
            param.inputPtr, dataBytes)));
    }

    uint64_t taskArgs[DIRECT_TASK_ARG_NUM] = {
        sourceAddr,
        sourceToken,
        outputAddr,
        outputToken,
        cclAddr,
        cclToken,
        dataBytes,
        static_cast<uint64_t>(DirectKernelPhase::RANK16_CLOS_PAIR_TREE),
    };
    CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread,
        resource.perf512KiBKernels[0], taskArgs, DIRECT_TASK_ARG_NUM));
    return HCCL_SUCCESS;
}

HcclResult ExecOneShot(const OpParam &param, const AlgResourceCtx &resource,
    uint64_t cclAddr, uint64_t cclToken, uint64_t dataBytes)
{
    bool needsSnapshot = param.inputPtr == param.outputPtr ||
        HasPartialBufferOverlap(param, dataBytes);
    if (CanLaunchFourRankDirectSourceOneShot(param, resource) &&
        HasFourRankDirectSourceCapacity(
            resource, dataBytes, needsSnapshot)) {
        return ExecFourRankDirectSourceOneShot(param, resource,
            cclAddr, cclToken, dataBytes, needsSnapshot);
    }

    if (CanLaunchRank12ClosTree(param, resource, dataBytes) &&
        HasRank12ClosTreeCapacity(resource, dataBytes, needsSnapshot)) {
        return ExecRank12ClosTree(param, resource, cclAddr, cclToken,
            dataBytes, needsSnapshot);
    }

    if (CanLaunchRank16ClosRsagKernel(param, resource, dataBytes) &&
        HasRank16ClosRsagCapacity(resource, dataBytes)) {
        return ExecRank16ClosRsag(param, resource,
            cclAddr, cclToken, dataBytes, needsSnapshot);
    }

    uint64_t slotStrideBytes = AlignUp(dataBytes, TRANSFER_ALIGNMENT_BYTES);
    CHK_PRT_RET(slotStrideBytes > resource.localBuffer.size / param.rankSize,
        HCCL_ERROR("F015 OneShot scratch is too small, stride[%llu] ranks[%u] buffer[%llu]",
            static_cast<unsigned long long>(slotStrideBytes), param.rankSize,
            static_cast<unsigned long long>(resource.localBuffer.size)), HCCL_E_INTERNAL);

    uint8_t *localBuffer = static_cast<uint8_t *>(resource.localBuffer.addr);
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(param.cpuThread,
        localBuffer + param.myRank * slotStrideBytes, param.inputPtr, dataBytes)));

    if (CanLaunchFusedOneShot(param, resource)) {
        CHK_RET(LaunchFusedOneShotCommunication(
            param, resource, cclAddr, cclToken, dataBytes, slotStrideBytes));
    } else {
        // Record every group before any group waits. This fallback is the
        // unchanged path for unsupported and non-two-die resource layouts.
        CHK_RET(LaunchAllGroups(param, resource, cclAddr, cclToken, dataBytes,
            slotStrideBytes, 0, 0, 0, DirectKernelPhase::INPUT_RECORD));
        CHK_RET(LaunchAllGroups(param, resource, cclAddr, cclToken, dataBytes,
            slotStrideBytes, 0, 0, 0, DirectKernelPhase::ONESHOT_READ_RECORD));
        CHK_RET(LaunchAllGroups(param, resource, cclAddr, cclToken, dataBytes,
            slotStrideBytes, 0, 0, 0, DirectKernelPhase::ONESHOT_DONE_WAIT));
    }

    // All peer inputs now occupy sender-indexed slots. One mission folds them
    // strictly in global rank order, independent of transfer completion order.
    CHK_RET(LaunchFirstGroup(param, resource, cclAddr, cclToken, dataBytes,
        slotStrideBytes, DirectKernelPhase::ONESHOT_LOCAL_REDUCE));
    CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
        param.cpuThread, param.outputPtr, localBuffer, dataBytes)));
    return HCCL_SUCCESS;
}

HcclResult ExecDirectRsAg(const OpParam &param, const AlgResourceCtx &resource,
    uint64_t cclAddr, uint64_t cclToken, uint64_t totalBytes)
{
    bool useParallelLayerRs = totalBytes > SMALL_ONE_SHOT_BYTES &&
        CanLaunchParallelLayerReduceScatter(resource);
    uint64_t windowCapacity = DirectWindowCapacity(resource.localBuffer.size, param.rankSize);
    CHK_PRT_RET(windowCapacity < static_cast<uint64_t>(param.rankSize) * sizeof(float),
        HCCL_ERROR("F015 Direct RSAG has no usable window capacity"), HCCL_E_INTERNAL);

    uint64_t windowAlignment =
        static_cast<uint64_t>(param.rankSize) * TRANSFER_ALIGNMENT_BYTES;
    uint64_t windowsLeft = CeilDiv(totalBytes, windowCapacity);
    uint64_t remainingBytes = totalBytes;
    uint64_t offset = 0;
    uint8_t *localBuffer = static_cast<uint8_t *>(resource.localBuffer.addr);
    uint8_t *inputBase = static_cast<uint8_t *>(param.inputPtr);
    uint8_t *outputBase = static_cast<uint8_t *>(param.outputPtr);

    while (windowsLeft > 0) {
        uint64_t targetBytes = CeilDiv(remainingBytes, windowsLeft);
        uint64_t windowBytes = windowsLeft > 1 ?
            AlignUp(targetBytes, windowAlignment) : targetBytes;
        CHK_PRT_RET(windowBytes == 0 || windowBytes > windowCapacity || windowBytes > remainingBytes,
            HCCL_ERROR("Invalid F015 window[%llu], capacity[%llu], remaining[%llu]",
                static_cast<unsigned long long>(windowBytes),
                static_cast<unsigned long long>(windowCapacity),
                static_cast<unsigned long long>(remainingBytes)), HCCL_E_INTERNAL);

        uint64_t regularSliceBytes = 0;
        uint64_t lastSliceBytes = 0;
        uint64_t mySliceOffset = 0;
        uint64_t mySliceBytes = 0;
        uint64_t resultOffsetBytes = 0;
        uint64_t maxSliceBytes = 0;
        CalculateSlices(windowBytes, param.rankSize, param.myRank,
            regularSliceBytes, lastSliceBytes, mySliceOffset, mySliceBytes,
            resultOffsetBytes, maxSliceBytes);
        CHK_PRT_RET(regularSliceBytes == 0 || mySliceBytes == 0 ||
                resultOffsetBytes > resource.localBuffer.size ||
                maxSliceBytes > resource.localBuffer.size - resultOffsetBytes,
            HCCL_ERROR("F015 slice layout exceeds CCL buffer, window[%llu] regular[%llu] "
                "last[%llu] resultOffset[%llu] buffer[%llu]",
                static_cast<unsigned long long>(windowBytes),
                static_cast<unsigned long long>(regularSliceBytes),
                static_cast<unsigned long long>(lastSliceBytes),
                static_cast<unsigned long long>(resultOffsetBytes),
                static_cast<unsigned long long>(resource.localBuffer.size)), HCCL_E_INTERNAL);

        // For ReduceScatter only, reuse the two generic task fields as the
        // regular and tail stripe lengths. The first three stripes are 128B
        // aligned; stripe 3 absorbs the FP32-aligned remainder. A zero regular
        // length selects the unchanged F002 serial path for tiny messages.
        uint64_t rsStripeBytes = AlignDown(
            mySliceBytes / RS_WAVEFRONT_STRIPE_NUM, TRANSFER_ALIGNMENT_BYTES);
        uint64_t rsTailStripeBytes =
            mySliceBytes - rsStripeBytes * (RS_WAVEFRONT_STRIPE_NUM - 1);
        CHK_PRT_RET(rsTailStripeBytes == 0 || rsTailStripeBytes % sizeof(float) != 0,
            HCCL_ERROR("Invalid F015 RS stripes, slice[%llu] regular[%llu] tail[%llu]",
                static_cast<unsigned long long>(mySliceBytes),
                static_cast<unsigned long long>(rsStripeBytes),
                static_cast<unsigned long long>(rsTailStripeBytes)), HCCL_E_INTERNAL);

        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
            param.cpuThread, localBuffer, inputBase + offset, windowBytes)));
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(param.cpuThread,
            localBuffer + resultOffsetBytes, localBuffer + mySliceOffset, mySliceBytes)));

        CHK_RET(LaunchAllGroups(param, resource, cclAddr, cclToken, windowBytes,
            0, mySliceOffset, mySliceBytes, resultOffsetBytes,
            DirectKernelPhase::INPUT_RECORD));
        if (useParallelLayerRs) {
            CHK_RET(LaunchParallelLayerReduceScatter(param, resource, cclAddr, cclToken,
                rsStripeBytes, rsTailStripeBytes, mySliceOffset, mySliceBytes,
                resultOffsetBytes));
        } else {
            CHK_RET(LaunchAllGroups(param, resource, cclAddr, cclToken, rsStripeBytes,
                rsTailStripeBytes, mySliceOffset, mySliceBytes, resultOffsetBytes,
                DirectKernelPhase::DIRECT_REDUCE_SCATTER));
        }

        // Put the locally-owned final slice in its logical position before
        // peers concurrently write the other owner slices into this window.
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(param.cpuThread,
            localBuffer + mySliceOffset, localBuffer + resultOffsetBytes, mySliceBytes)));
        CHK_RET(LaunchDirectAllGather(param, resource, cclAddr, cclToken,
            windowBytes, mySliceOffset, mySliceBytes, resultOffsetBytes));

        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(
            param.cpuThread, outputBase + offset, localBuffer, windowBytes)));

        offset += windowBytes;
        remainingBytes -= windowBytes;
        --windowsLeft;
    }
    return HCCL_SUCCESS;
}

HcclResult ExecDirectBufferRsAg(const OpParam &param, const AlgResourceCtx &resource,
    const std::vector<CcuKernelHandle> &kernels, uint64_t totalBytes)
{
    CHK_PRT_RET(!IsSupportedDirectBufferRankSize(param.rankSize) ||
            kernels.size() != ASCEND950_DIE_NUM ||
            resource.threads.size() < ASCEND950_DIE_NUM,
        HCCL_ERROR("Invalid F015 direct-buffer resources"), HCCL_E_INTERNAL);

    uint64_t inputAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    uint64_t outputAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    uint64_t scratchAddr = reinterpret_cast<uint64_t>(resource.localBuffer.addr);
    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    uint64_t scratchToken = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(inputAddr, totalBytes, &inputToken));
    CHK_RET_CCU(HcommCcuGetMemToken(outputAddr, totalBytes, &outputToken));
    CHK_RET_CCU(HcommCcuGetMemToken(
        scratchAddr, resource.localBuffer.size, &scratchToken));

    uint64_t windowCapacity =
        DirectBufferWindowCapacity(resource.localBuffer.size, param.rankSize);
    CHK_PRT_RET(windowCapacity < static_cast<uint64_t>(param.rankSize) * sizeof(float),
        HCCL_ERROR("F015 direct-buffer path has no usable window capacity"), HCCL_E_INTERNAL);
    uint64_t windowAlignment =
        static_cast<uint64_t>(param.rankSize) * TRANSFER_ALIGNMENT_BYTES;
    uint64_t windowsLeft = CeilDiv(totalBytes, windowCapacity);
    uint64_t remainingBytes = totalBytes;
    uint64_t offset = 0;

    while (windowsLeft > 0) {
        uint64_t targetBytes = CeilDiv(remainingBytes, windowsLeft);
        uint64_t windowBytes = windowsLeft > 1 ?
            AlignUp(targetBytes, windowAlignment) : targetBytes;
        CHK_PRT_RET(windowBytes == 0 || windowBytes > windowCapacity ||
                windowBytes > remainingBytes,
            HCCL_ERROR("Invalid F015 direct-buffer window[%llu], capacity[%llu], remaining[%llu]",
                static_cast<unsigned long long>(windowBytes),
                static_cast<unsigned long long>(windowCapacity),
                static_cast<unsigned long long>(remainingBytes)), HCCL_E_INTERNAL);

        uint64_t regularSliceBytes =
            AlignDown(windowBytes / param.rankSize, TRANSFER_ALIGNMENT_BYTES);
        uint64_t lastSliceBytes = windowBytes -
            regularSliceBytes * (param.rankSize - 1);
        uint64_t mySliceBytes = param.myRank + 1 == param.rankSize ?
            lastSliceBytes : regularSliceBytes;
        uint64_t maxSliceBytes = std::max(regularSliceBytes, lastSliceBytes);
        CHK_PRT_RET(regularSliceBytes == 0 || mySliceBytes == 0 ||
                maxSliceBytes > resource.localBuffer.size ||
                maxSliceBytes > SAFE_TRANSFER_BYTES ||
                regularSliceBytes % sizeof(float) != 0 ||
                lastSliceBytes % sizeof(float) != 0,
            HCCL_ERROR("Invalid F015 direct-buffer slices, window[%llu] regular[%llu] "
                "last[%llu] scratch[%llu]",
                static_cast<unsigned long long>(windowBytes),
                static_cast<unsigned long long>(regularSliceBytes),
                static_cast<unsigned long long>(lastSliceBytes),
                static_cast<unsigned long long>(resource.localBuffer.size)), HCCL_E_INTERNAL);

        // Rank 16 keeps the proven cyclic-16 schedule. Rank 12 has at most
        // eight Channels on either Die, so cyclic-8 is the smallest schedule
        // that keeps every Channel active in every round. Host and Kernel
        // must use the same stripe count to preserve complete slice coverage.
        uint32_t stripeNum = param.rankSize == ASYMMETRIC_RANK_SIZE ?
            ASYMMETRIC_DIRECT_BUFFER_RS_STRIPE_NUM : DIRECT_BUFFER_RS_STRIPE_NUM;
        uint64_t regularStripeBytes = AlignDown(
            mySliceBytes / stripeNum, TRANSFER_ALIGNMENT_BYTES);
        uint64_t tailStripeBytes = mySliceBytes -
            regularStripeBytes * (stripeNum - 1);
        CHK_PRT_RET(regularStripeBytes == 0 || tailStripeBytes == 0 ||
                tailStripeBytes % sizeof(float) != 0,
            HCCL_ERROR("Invalid F015 direct-buffer stripes, slice[%llu] regular[%llu] tail[%llu]",
                static_cast<unsigned long long>(mySliceBytes),
                static_cast<unsigned long long>(regularStripeBytes),
                static_cast<unsigned long long>(tailStripeBytes)), HCCL_E_INTERNAL);

        uint64_t inputWindowAddr = inputAddr + offset;
        uint64_t outputWindowAddr = outputAddr + offset;
        uint64_t rsArgs[DIRECT_BUFFER_TASK_ARG_NUM]{};
        uint64_t mergeArgs[DIRECT_BUFFER_TASK_ARG_NUM]{};
        uint64_t agArgs[DIRECT_BUFFER_TASK_ARG_NUM]{};
        BuildDirectBufferTaskArgs(inputWindowAddr, inputToken,
            outputWindowAddr, outputToken, scratchAddr, scratchToken,
            regularSliceBytes, lastSliceBytes, regularStripeBytes, tailStripeBytes,
            DirectKernelPhase::DIRECT_BUFFER_REDUCE_SCATTER, rsArgs);
        BuildDirectBufferTaskArgs(inputWindowAddr, inputToken,
            outputWindowAddr, outputToken, scratchAddr, scratchToken,
            regularSliceBytes, lastSliceBytes, 0, 0,
            DirectKernelPhase::DIRECT_BUFFER_MERGE, mergeArgs);
        BuildDirectBufferTaskArgs(inputWindowAddr, inputToken,
            outputWindowAddr, outputToken, scratchAddr, scratchToken,
            regularSliceBytes, lastSliceBytes, 0, 0,
            DirectKernelPhase::DIRECT_BUFFER_ALL_GATHER, agArgs);

        CHK_RET(LaunchParallelDirectBufferPhase(
            param, resource, kernels, rsArgs));
        CHK_RET(LaunchDirectBufferKernel(
            param.cpuThread, kernels, 0, mergeArgs));
        CHK_RET(LaunchParallelDirectBufferPhase(
            param, resource, kernels, agArgs));

        offset += windowBytes;
        remainingBytes -= windowBytes;
        --windowsLeft;
    }
    return HCCL_SUCCESS;
}

HcclResult ExecFourRankDirectNhr(const OpParam &param, const AlgResourceCtx &resource,
    CcuKernelHandle kernel, uint64_t totalBytes)
{
    CHK_PRT_RET(param.rankSize != FOUR_RANK_NHR_SIZE || kernel == 0,
        HCCL_ERROR("Invalid F015 4-rank NHR resources"), HCCL_E_INTERNAL);

    // Split by FP32 element count.  For 400 MiB + 4 B this produces three
    // exact 100 MiB slices and a 100 MiB + 4 B tail instead of truncating the
    // final element through a byte-based division.
    uint64_t regularSliceCount = param.count / FOUR_RANK_NHR_SIZE;
    uint64_t regularSliceBytes = regularSliceCount * sizeof(float);
    uint64_t lastSliceBytes = totalBytes -
        regularSliceBytes * (FOUR_RANK_NHR_SIZE - 1);
    uint64_t maxSliceBytes = std::max(regularSliceBytes, lastSliceBytes);
    CHK_PRT_RET(regularSliceBytes == 0 || lastSliceBytes == 0 ||
            maxSliceBytes > SAFE_TRANSFER_BYTES ||
            regularSliceBytes % sizeof(float) != 0 || lastSliceBytes % sizeof(float) != 0,
        HCCL_ERROR("Invalid F015 NHR slices, regular[%llu] last[%llu] limit[%llu]",
            static_cast<unsigned long long>(regularSliceBytes),
            static_cast<unsigned long long>(lastSliceBytes),
            static_cast<unsigned long long>(SAFE_TRANSFER_BYTES)), HCCL_E_INTERNAL);

    uint64_t inputAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    uint64_t outputAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(inputAddr, totalBytes, &inputToken));
    if (inputAddr == outputAddr) {
        outputToken = inputToken;
    } else {
        CHK_RET_CCU(HcommCcuGetMemToken(outputAddr, totalBytes, &outputToken));
    }

    uint64_t taskArgs[FOUR_RANK_NHR_TASK_ARG_NUM] = {
        inputAddr,
        inputToken,
        outputAddr,
        outputToken,
        regularSliceBytes,
        lastSliceBytes,
        inputAddr == outputAddr ? 1ULL : 0ULL,
    };
    CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread, kernel,
        taskArgs, FOUR_RANK_NHR_TASK_ARG_NUM));
    return HCCL_SUCCESS;
}

HcclResult ExecSingleRank(const OpParam &param, uint64_t totalBytes)
{
    if (param.inputPtr == param.outputPtr) {
        return HCCL_SUCCESS;
    }

    uint64_t offset = 0;
    while (offset < totalBytes) {
        uint64_t bytes = std::min(SAFE_TRANSFER_BYTES, totalBytes - offset);
        CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(param.cpuThread,
            static_cast<uint8_t *>(param.outputPtr) + offset,
            static_cast<uint8_t *>(param.inputPtr) + offset, bytes)));
        offset += bytes;
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param)
{
    CHK_PRT_RET(param.resCtx == nullptr || param.ctxSize == 0,
        HCCL_ERROR("Invalid serialized resource context"), HCCL_E_INTERNAL);

    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> sequence(ctx, ctx + param.ctxSize);
    AlgResourceCtx resource;
    resource.DeSerialize(sequence);

    uint64_t totalBytes = param.count * sizeof(float);
    if (param.rankSize == 1) {
        return ExecSingleRank(param, totalBytes);
    }

    bool invalidKernelCount = resource.ccuKernels.empty() ||
        resource.ccuKernels.size() > ASCEND950_DIE_NUM;
    bool invalidKernelMetadata =
        resource.kernelNetLayers.size() != resource.ccuKernels.size();
    size_t expectedSmallCount = 0;
    size_t expectedLargeCount = 0;
    if (param.rankSize == FOUR_RANK_NHR_SIZE) {
        expectedSmallCount = 1;
        expectedLargeCount = 1;
    } else if (IsSupportedDirectBufferRankSize(param.rankSize)) {
        expectedSmallCount = 1;
        expectedLargeCount = ASCEND950_DIE_NUM;
    }
    bool invalidProbeKernelCount =
        resource.perf512KiBKernels.size() != expectedSmallCount ||
        resource.perf512MiBKernels.size() != expectedLargeCount ||
        resource.perf400MiB4BKernels.size() != expectedLargeCount;
    auto containsZeroHandle = [](const std::vector<CcuKernelHandle> &kernels) {
        return std::any_of(kernels.begin(), kernels.end(),
            [](CcuKernelHandle kernel) { return kernel == 0; });
    };
    bool invalidProbeHandle = containsZeroHandle(resource.perf512KiBKernels) ||
        containsZeroHandle(resource.perf512MiBKernels) ||
        containsZeroHandle(resource.perf400MiB4BKernels);
    CHK_PRT_RET(resource.localBuffer.addr == nullptr || resource.localBuffer.size < sizeof(float),
        HCCL_ERROR("Invalid local HCCL buffer"), HCCL_E_INTERNAL);
    CHK_PRT_RET(resource.threads.empty() || resource.threads.size() > ASCEND950_DIE_NUM ||
            invalidKernelCount || invalidKernelMetadata ||
            !resource.parallelRsKernels.empty() || invalidProbeKernelCount ||
            invalidProbeHandle,
        HCCL_ERROR("Invalid F081 probe resources: threads[%llu] generic[%llu] "
            "small[%llu] large512[%llu] large400p4[%llu] layers[%llu]",
            static_cast<unsigned long long>(resource.threads.size()),
            static_cast<unsigned long long>(resource.ccuKernels.size()),
            static_cast<unsigned long long>(resource.perf512KiBKernels.size()),
            static_cast<unsigned long long>(resource.perf512MiBKernels.size()),
            static_cast<unsigned long long>(resource.perf400MiB4BKernels.size()),
            static_cast<unsigned long long>(resource.kernelNetLayers.size())), HCCL_E_INTERNAL);

    if (totalBytes == SMALL_ONE_SHOT_BYTES) {
        uint64_t cclAddr = reinterpret_cast<uint64_t>(resource.localBuffer.addr);
        uint64_t cclToken = 0;
        CHK_RET_CCU(HcommCcuGetMemToken(cclAddr, resource.localBuffer.size, &cclToken));
        return ExecOneShot(param, resource, cclAddr, cclToken, totalBytes);
    }

    const std::vector<CcuKernelHandle> *largeKernels = nullptr;
    if (totalBytes == LARGE_512_MIB_BYTES) {
        largeKernels = &resource.perf512MiBKernels;
    } else if (totalBytes == LARGE_400_MIB_PLUS_4_BYTES) {
        largeKernels = &resource.perf400MiB4BKernels;
    }

    bool canUseFourRankNhr = largeKernels != nullptr &&
        param.rankSize == FOUR_RANK_NHR_SIZE &&
        resource.ccuKernels.size() == 1 && largeKernels->size() == 1 &&
        !HasPartialBufferOverlap(param, totalBytes);
    if (canUseFourRankNhr) {
        return ExecFourRankDirectNhr(
            param, resource, (*largeKernels)[0], totalBytes);
    }

    bool canUseDirectBuffer = largeKernels != nullptr &&
        IsSupportedDirectBufferRankSize(param.rankSize) &&
        largeKernels->size() == ASCEND950_DIE_NUM &&
        param.inputPtr != param.outputPtr &&
        !HasPartialBufferOverlap(param, totalBytes);
    if (canUseDirectBuffer) {
        return ExecDirectBufferRsAg(
            param, resource, *largeKernels, totalBytes);
    }

    uint64_t cclAddr = reinterpret_cast<uint64_t>(resource.localBuffer.addr);
    uint64_t cclToken = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(cclAddr, resource.localBuffer.size, &cclToken));
    return ExecDirectRsAg(param, resource, cclAddr, cclToken, totalBytes);
}

} // namespace ops_hccl
