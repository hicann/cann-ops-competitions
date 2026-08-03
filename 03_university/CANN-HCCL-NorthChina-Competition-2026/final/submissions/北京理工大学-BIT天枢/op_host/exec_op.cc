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

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {

namespace {

constexpr uint32_t DIE_WORKER_NOTIFY_ID = 0;
constexpr uint32_t OWN_COPY_START_NOTIFY_ID = 0;
constexpr uint32_t OWN_COPY_DONE_NOTIFY_ID = 1;
constexpr uint32_t DEFERRED_OWN_COPY_DONE_NOTIFY_ID = 2;
constexpr uint32_t CROSS_DONE_NOTIFY_ID = 1;
constexpr uint32_t FIXED_PULL_TASK_ARGS = 5;
constexpr uint32_t MAX_DIE_GROUPS = 2;
constexpr uint32_t ALGORITHM_V10_PULL = 0;
constexpr uint32_t ALGORITHM_4X1_COMMON_DIE_PUSH = 1;
constexpr uint32_t ALGORITHM_8P4_DUAL_SEED = 2;
constexpr uint32_t ALGORITHM_4X1_DUAL_PLANE_PUSH = 3;
constexpr uint32_t ALGORITHM_2X8_V22_HYBRID = 4;
constexpr uint32_t ALGORITHM_8P4_LATENCY_DIRECT = 5;
constexpr uint32_t ALGORITHM_8P4_WIDE_PIPELINE = 6;
constexpr uint32_t ALGORITHM_OUTPUT_512_STATIC = 7;
constexpr uint32_t SPECIALIZED_PUSH_TASK_ARGS = 7;
constexpr uint32_t CLOS_4X1_FIXED_TASK_ARGS = 5;
constexpr uint32_t CLOS_4X1_MAX_CHUNKS = 4;
constexpr uint32_t DUAL_SEED_CROSS_TASK_ARGS = 11;
constexpr uint32_t WIDE_DUAL_SEED_CROSS_TASK_ARGS = 12;
constexpr uint32_t DUAL_SEED_RELAY_TASK_ARGS = 9;
constexpr uint32_t LATENCY_PUSH_TASK_ARGS = 6;
constexpr uint32_t CROSS_PHASE_SEED_ONLY = 0;
constexpr uint32_t CROSS_PHASE_DIRECT_ONLY = 1;
thread_local bool g_4x1Exact512WarmDispatch = false;

// V22b's 2x8 path is deliberately kept in a separate namespace-by-prefix.
// Its seven registered kernels and two thread notifies form one protocol;
// reusing any of the V10/V10A indices or synchronization constants would
// silently couple the 2x8 schedule to the 4x1 and 8+4 implementations.
constexpr uint64_t FUSION_V22_MAX_SLICE_BYTES = 256ULL * 1024ULL * 1024ULL;
constexpr uint64_t FUSION_V22_MAX_KERNEL_BYTES = 2ULL * FUSION_V22_MAX_SLICE_BYTES;
constexpr uint64_t FUSION_V22_PIPELINE_THRESHOLD = 1ULL * 1024ULL * 1024ULL;
constexpr uint32_t FUSION_V22_THREAD_NOTIFY_INDEX = 0;
constexpr uint32_t FUSION_V22_CROSS_DONE_NOTIFY_INDEX = 1;
constexpr uint32_t FUSION_V22_RANK_SIZE = 16;
constexpr uint32_t FUSION_V22_RANKS_PER_SERVER = 8;
constexpr uint32_t FUSION_V22_GENERIC_INTRA_KERNEL = 0;
constexpr uint32_t FUSION_V22_GENERIC_CROSS_KERNEL = 1;
constexpr uint32_t FUSION_V22_PIPELINE_INTRA_ORIGINAL_KERNEL = 2;
constexpr uint32_t FUSION_V22_INTRA_RELAY_FINAL_KERNEL = 3;
constexpr uint32_t FUSION_V22_CROSS_HIERARCHICAL_KERNEL = 4;
constexpr uint32_t FUSION_V22_CROSS_DIRECT_SINGLE_FINAL_KERNEL = 5;
constexpr uint32_t FUSION_V22_CROSS_DIRECT_DOUBLE_FINAL_KERNEL = 6;
constexpr uint32_t FUSION_V22_PIPELINE_KERNEL_NUM = 7;
constexpr uint32_t FUSION_V22_HIERARCHICAL_NUMERATOR = 4;
constexpr uint32_t FUSION_V22_HIERARCHICAL_DENOMINATOR = 11;

struct AsymmetricPartitions {
    uint64_t smallDirectBytes = 0;
    SlicePair smallDirectSlices{};
    uint64_t smallStripe0Bytes = 0;
    uint64_t smallStripe1Bytes = 0;
    uint64_t largeDirectBytes = 0;
    SlicePair largeDirectSlices{};
    uint64_t largeTailBytes = 0;
    bool useRelay = false;
};

struct Clos4x1Chunks {
    uint64_t bytes[CLOS_4X1_MAX_CHUNKS]{};
    uint32_t count = 0U;

    uint64_t Total() const
    {
        uint64_t total = 0U;
        for (uint32_t index = 0; index < count; ++index) {
            total += bytes[index];
        }
        return total;
    }
};

HcclResult StartOwnCopyAfterThread(const OpParam &param,
    const AlgResourceCtx &resource, uint64_t rankBytes,
    ThreadHandle startThread, ThreadHandle completionThread,
    uint32_t doneNotifyId, bool &started);
HcclResult WaitOwnCopyOnThread(
    ThreadHandle completionThread, uint32_t doneNotifyId, bool started);

HcclResult FusionV22LaunchKernel(ThreadHandle thread, CcuKernelHandle kernel,
    const std::vector<uint64_t> &taskArgs)
{
    const CcuResult result = HcommCcuKernelLaunch(
        thread, kernel, taskArgs.data(), static_cast<uint32_t>(taskArgs.size()));
    if (result != CCU_SUCCESS) {
        HCCL_ERROR("[FusionV22LaunchKernel] Kernel launch failed, ccuRet=%d", result);
        return ConvertCcuToHccl(result);
    }
    return HCCL_SUCCESS;
}

HcclResult FusionV22PreSyncThreads(
    ThreadHandle intraThread, ThreadHandle crossThread)
{
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        intraThread, crossThread, FUSION_V22_THREAD_NOTIFY_INDEX)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        crossThread, FUSION_V22_THREAD_NOTIFY_INDEX, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult FusionV22PostSyncThreads(
    ThreadHandle intraThread, ThreadHandle crossThread)
{
    // Preserve V22b's queue ordering: enqueue the main-thread wait first and
    // then enqueue the matching record after the cross kernel on its worker.
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        intraThread, FUSION_V22_THREAD_NOTIFY_INDEX, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        crossThread, intraThread, FUSION_V22_THREAD_NOTIFY_INDEX)));
    return HCCL_SUCCESS;
}

HcclResult FusionV22LaunchGeneric(
    const AlgResourceCtx &resource, ThreadHandle intraThread,
    const std::vector<uint64_t> &taskArgs)
{
    CHK_PRT_RET(resource.threads.size() != 2U
            || resource.ccuKernels.size() < FUSION_V22_PIPELINE_KERNEL_NUM,
        HCCL_ERROR("[FusionV22LaunchGeneric] Invalid 2x8 resources"),
        HCCL_E_INTERNAL);

    const ThreadHandle crossThread = resource.threads[1];
    CHK_RET(FusionV22PreSyncThreads(intraThread, crossThread));
    CHK_RET(FusionV22LaunchKernel(intraThread,
        resource.ccuKernels[FUSION_V22_GENERIC_INTRA_KERNEL], taskArgs));
    CHK_RET(FusionV22LaunchKernel(crossThread,
        resource.ccuKernels[FUSION_V22_GENERIC_CROSS_KERNEL], taskArgs));
    return FusionV22PostSyncThreads(intraThread, crossThread);
}

std::vector<uint64_t> FusionV22BuildHybridArgs(
    uint64_t inputAddr, uint64_t outputAddr, uint64_t token,
    uint64_t rankOutputOffset, uint64_t batchOffset, uint64_t baseChunkSize,
    uint64_t lastChunkSize, uint64_t dataSize, uint32_t remoteServerBase)
{
    // CcuV22AllGatherPhaseKernel consumes exactly fifteen dynamic arguments:
    // seven common fields followed by eight remote-rank output block offsets.
    std::vector<uint64_t> args = {
        inputAddr,
        outputAddr,
        token,
        rankOutputOffset,
        batchOffset,
        baseChunkSize,
        lastChunkSize,
    };
    args.reserve(7U + FUSION_V22_RANKS_PER_SERVER);
    for (uint32_t localRank = 0; localRank < FUSION_V22_RANKS_PER_SERVER; ++localRank) {
        args.push_back(
            static_cast<uint64_t>(remoteServerBase + localRank) * dataSize);
    }
    return args;
}

HcclResult FusionV22LaunchBandwidthBalancedHybrid(
    const OpParam &param, const AlgResourceCtx &resource, ThreadHandle intraThread,
    uint32_t myRank, uint64_t inputAddr, uint64_t outputAddr,
    uint64_t token, uint64_t dataSize)
{
    CHK_PRT_RET(resource.threads.size() != 2U
            || resource.ccuKernels.size() < FUSION_V22_PIPELINE_KERNEL_NUM,
        HCCL_ERROR("[FusionV22LaunchBandwidthBalancedHybrid] Invalid 2x8 resources"),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(dataSize
            > std::numeric_limits<uint64_t>::max()
                / FUSION_V22_HIERARCHICAL_NUMERATOR,
        HCCL_ERROR("[FusionV22LaunchBandwidthBalancedHybrid] Data size overflows"),
        HCCL_E_PARA);

    // 借8+4的尾块策略：层级前缀向下64B对齐，所有奇尾都由Direct
    // 连续后缀承接，既不丢4B尾部，也不改变[0,dataSize)覆盖。
    const uint64_t hierarchicalSize =
        (dataSize * FUSION_V22_HIERARCHICAL_NUMERATOR
            / FUSION_V22_HIERARCHICAL_DENOMINATOR
            / DMA_ALIGNMENT_BYTES)
        * DMA_ALIGNMENT_BYTES;
    const uint64_t directSize = dataSize - hierarchicalSize;
    const uint64_t directFirstSize =
        directSize <= FUSION_V22_MAX_SLICE_BYTES
        ? directSize
        : (directSize / 2U / DMA_ALIGNMENT_BYTES) * DMA_ALIGNMENT_BYTES;
    const uint64_t directSecondSize = directSize - directFirstSize;
    CHK_PRT_RET(hierarchicalSize == 0U || directFirstSize == 0U
            || hierarchicalSize > FUSION_V22_MAX_SLICE_BYTES
            || directFirstSize > FUSION_V22_MAX_SLICE_BYTES
            || directSecondSize > FUSION_V22_MAX_SLICE_BYTES,
        HCCL_ERROR("[FusionV22LaunchBandwidthBalancedHybrid] Invalid 4/11-7/11 partition"),
        HCCL_E_PARA);

    const uint64_t rankOutputOffset = static_cast<uint64_t>(myRank) * dataSize;
    const uint32_t remoteServerBase =
        myRank < FUSION_V22_RANKS_PER_SERVER ? FUSION_V22_RANKS_PER_SERVER : 0U;
    const std::vector<uint64_t> hierarchicalArgs = FusionV22BuildHybridArgs(
        inputAddr, outputAddr, token, rankOutputOffset, 0U,
        hierarchicalSize, hierarchicalSize, dataSize, remoteServerBase);
    const std::vector<uint64_t> directArgs = FusionV22BuildHybridArgs(
        inputAddr, outputAddr, token, rankOutputOffset, hierarchicalSize,
        directFirstSize,
        directSecondSize == 0U ? directFirstSize : directSecondSize,
        dataSize, remoteServerBase);
    const uint64_t firstSliceSize =
        std::min<uint64_t>(dataSize, FUSION_V22_MAX_SLICE_BYTES);
    const uint64_t secondSliceSize = dataSize - firstSliceSize;
    const std::vector<uint64_t> intraOriginalArgs = {
        inputAddr,
        outputAddr,
        token,
        rankOutputOffset,
        firstSliceSize,
        secondSliceSize,
    };

    const ThreadHandle crossThread = resource.threads[1];
    const bool inlineTailCopy = dataSize % DMA_ALIGNMENT_BYTES != 0U;

    // Preserve V22b's launch allocation.  Both 4/11 critical paths are
    // queued before the remaining 7/11 direct traffic: CLOS seeds one
    // matching peer while the main thread starts the full intra-server push.
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        intraThread, crossThread, FUSION_V22_THREAD_NOTIFY_INDEX)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        crossThread, FUSION_V22_THREAD_NOTIFY_INDEX, CUSTOM_TIMEOUT)));

    CHK_RET(FusionV22LaunchKernel(crossThread,
        resource.ccuKernels[FUSION_V22_CROSS_HIERARCHICAL_KERNEL],
        hierarchicalArgs));

    // 不让OwnCopy抢占Seed的初始带宽。Cross线程完成Seed后才唤醒第三线程，
    // 随后本地拷贝与Direct/Relay阶段重叠。4B尾型使用独立Context恢复
    // 已验证的Kernel内网络优先OwnCopy，从而删除四个Host通知节点。
    bool ownCopyStarted = false;
    if (!inlineTailCopy) {
        CHK_RET(StartOwnCopyAfterThread(param, resource, dataSize,
            crossThread, intraThread, DEFERRED_OWN_COPY_DONE_NOTIFY_ID,
            ownCopyStarted));
    }

    const HcclResult networkRet = [&]() -> HcclResult {
        CHK_RET(FusionV22LaunchKernel(intraThread,
            resource.ccuKernels[FUSION_V22_PIPELINE_INTRA_ORIGINAL_KERNEL],
            intraOriginalArgs));

        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            crossThread, intraThread, FUSION_V22_THREAD_NOTIFY_INDEX)));
        CHK_RET(FusionV22LaunchKernel(crossThread,
            resource.ccuKernels[
                directSecondSize == 0U
                    ? FUSION_V22_CROSS_DIRECT_SINGLE_FINAL_KERNEL
                    : FUSION_V22_CROSS_DIRECT_DOUBLE_FINAL_KERNEL],
            directArgs));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            crossThread, intraThread, FUSION_V22_CROSS_DONE_NOTIFY_INDEX)));

        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            intraThread, FUSION_V22_THREAD_NOTIFY_INDEX, CUSTOM_TIMEOUT)));
        CHK_RET(FusionV22LaunchKernel(intraThread,
            resource.ccuKernels[FUSION_V22_INTRA_RELAY_FINAL_KERNEL],
            hierarchicalArgs));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            intraThread, FUSION_V22_CROSS_DONE_NOTIFY_INDEX, CUSTOM_TIMEOUT)));
        return HCCL_SUCCESS;
    }();
    const HcclResult copyRet = inlineTailCopy
        ? HCCL_SUCCESS
        : WaitOwnCopyOnThread(
            intraThread, DEFERRED_OWN_COPY_DONE_NOTIFY_ID, ownCopyStarted);
    return networkRet != HCCL_SUCCESS ? networkRet : copyRet;
}

HcclResult Exec2x8V22(
    const OpParam &param, const AlgResourceCtx &resource, const ExecPlan &plan)
{
    CHK_PRT_RET(param.rankSize != FUSION_V22_RANK_SIZE
            || resource.threads.size() != 2U
            || resource.ccuKernels.size() < FUSION_V22_PIPELINE_KERNEL_NUM,
        HCCL_ERROR("[Exec2x8V22] Invalid specialized 2x8 resources"),
        HCCL_E_INTERNAL);

    const uint64_t inputAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outputAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    uint64_t token = 0U;
    CHK_RET_CCU(HcommCcuGetMemToken(inputAddr, plan.rankBytes, &token));

    if (plan.rankBytes >= FUSION_V22_PIPELINE_THRESHOLD
        && plan.rankBytes <= FUSION_V22_MAX_KERNEL_BYTES) {
        return FusionV22LaunchBandwidthBalancedHybrid(
            param, resource, param.cpuThread, param.myRank,
            inputAddr, outputAddr, token, plan.rankBytes);
    }

    // The 512KB latency point, and any out-of-range fallback, keep V22b's
    // proven dual-group kernel.  Group 0 performs the one local copy while
    // groups 0 and 1 concurrently push to the 7 intra and 8 cross peers.
    const uint64_t rankOutputOffset =
        static_cast<uint64_t>(param.myRank) * plan.rankBytes;
    uint64_t processedBytes = 0U;
    while (processedBytes < plan.rankBytes) {
        const uint64_t kernelBytes = std::min<uint64_t>(
            plan.rankBytes - processedBytes, FUSION_V22_MAX_KERNEL_BYTES);
        uint64_t firstSliceSize =
            std::min<uint64_t>(kernelBytes, FUSION_V22_MAX_SLICE_BYTES);
        uint64_t secondSliceSize = kernelBytes - firstSliceSize;

        if (kernelBytes >= FUSION_V22_PIPELINE_THRESHOLD) {
            const uint64_t alignedHalf =
                (kernelBytes / 2U / sizeof(float)) * sizeof(float);
            if (alignedHalf != 0U && alignedHalf < kernelBytes) {
                firstSliceSize = alignedHalf;
                secondSliceSize = kernelBytes - alignedHalf;
            }
        }

        const std::vector<uint64_t> taskArgs = {
            inputAddr + processedBytes,
            outputAddr + processedBytes,
            token,
            rankOutputOffset,
            firstSliceSize,
            secondSliceSize,
        };
        CHK_RET(FusionV22LaunchGeneric(
            resource, param.cpuThread, taskArgs));
        processedBytes += kernelBytes;
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchPeerLanePull(ThreadHandle thread, CcuKernelHandle kernel,
    const std::vector<uint32_t> &peerRanks, uint64_t inputAddr, uint64_t inputToken,
    uint64_t outputAddr, uint64_t outputToken, uint64_t rankBytes, uint64_t batchOffset,
    const SlicePair &slices)
{
    CHK_PRT_RET(peerRanks.empty() || peerRanks.size() > MAX_CHANNELS_PER_DIE
            || FIXED_PULL_TASK_ARGS + peerRanks.size() > MAX_CCU_TASK_ARGS,
        HCCL_ERROR("[LaunchPeerLanePull] Invalid peer count %zu", peerRanks.size()), HCCL_E_INTERNAL);

    std::vector<uint64_t> taskArgs;
    taskArgs.reserve(FIXED_PULL_TASK_ARGS + peerRanks.size());
    taskArgs.push_back(inputAddr + batchOffset);
    taskArgs.push_back(inputToken);
    taskArgs.push_back(outputToken);
    taskArgs.push_back(slices.first);
    taskArgs.push_back(slices.second);
    for (uint32_t peerRank : peerRanks) {
        taskArgs.push_back(outputAddr + static_cast<uint64_t>(peerRank) * rankBytes + batchOffset);
    }

    const CcuResult ret = HcommCcuKernelLaunch(
        thread, kernel, taskArgs.data(), static_cast<uint32_t>(taskArgs.size()));
    if (ret != CCU_SUCCESS) {
        HCCL_ERROR("[LaunchPeerLanePull] Kernel launch failed, ccuRet=%d", ret);
        return ConvertCcuToHccl(ret);
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchDieGroups(const OpParam &param, const AlgResourceCtx &resource, const ExecPlan &plan,
    uint64_t inputAddr, uint64_t inputToken, uint64_t outputAddr, uint64_t outputToken,
    uint64_t batchOffset, const SlicePair &slices, void *inlineCopyDestination,
    const void *inlineCopySource, uint64_t inlineCopyBytes)
{
    const size_t groupCount = resource.ccuKernels.size();
    CHK_PRT_RET(groupCount == 0 || groupCount > MAX_DIE_GROUPS
            || resource.threads.size() != groupCount
            || resource.peerRanksByGroup.size() != groupCount,
        HCCL_ERROR("[LaunchDieGroups] Inconsistent direct-matrix resources"), HCCL_E_INTERNAL);

    if (groupCount == 2) {
        const ThreadHandle worker = resource.threads[1];
        CHK_RET(HcommThreadNotifyRecordOnThread(param.cpuThread, worker, DIE_WORKER_NOTIFY_ID));
        CHK_RET(HcommThreadNotifyWaitOnThread(worker, DIE_WORKER_NOTIFY_ID, CUSTOM_TIMEOUT));

        CHK_RET(LaunchPeerLanePull(param.cpuThread, resource.ccuKernels[0],
            resource.peerRanksByGroup[0], inputAddr, inputToken, outputAddr, outputToken,
            plan.rankBytes, batchOffset, slices));
        CHK_RET(LaunchPeerLanePull(worker, resource.ccuKernels[1],
            resource.peerRanksByGroup[1], inputAddr, inputToken, outputAddr, outputToken,
            plan.rankBytes, batchOffset, slices));
        CHK_RET(HcommThreadNotifyRecordOnThread(worker, param.cpuThread, DIE_WORKER_NOTIFY_ID));

        if (inlineCopyBytes != 0) {
            CHK_RET(HcommLocalCopyOnThread(
                param.cpuThread, inlineCopyDestination, inlineCopySource, inlineCopyBytes));
        }
        CHK_RET(HcommThreadNotifyWaitOnThread(param.cpuThread, DIE_WORKER_NOTIFY_ID, CUSTOM_TIMEOUT));
        return HCCL_SUCCESS;
    }

    CHK_RET(LaunchPeerLanePull(param.cpuThread, resource.ccuKernels[0],
        resource.peerRanksByGroup[0], inputAddr, inputToken, outputAddr, outputToken,
        plan.rankBytes, batchOffset, slices));
    if (inlineCopyBytes != 0) {
        CHK_RET(HcommLocalCopyOnThread(
            param.cpuThread, inlineCopyDestination, inlineCopySource, inlineCopyBytes));
    }
    return HCCL_SUCCESS;
}

HcclResult StartOwnCopyAfterThread(const OpParam &param,
    const AlgResourceCtx &resource, uint64_t rankBytes,
    ThreadHandle startThread, ThreadHandle completionThread,
    uint32_t doneNotifyId, bool &started)
{
    started = false;
    char *ownOutput =
        static_cast<char *>(param.outputPtr) + static_cast<uint64_t>(param.myRank) * rankBytes;
    if (param.inputPtr == ownOutput) {
        return HCCL_SUCCESS;
    }

    CHK_RET(HcommThreadNotifyRecordOnThread(
        startThread, resource.ownCopyThread, OWN_COPY_START_NOTIFY_ID));
    CHK_RET(HcommThreadNotifyWaitOnThread(
        resource.ownCopyThread, OWN_COPY_START_NOTIFY_ID, CUSTOM_TIMEOUT));
    uint64_t copiedBytes = 0;
    while (copiedBytes < rankBytes) {
        const uint64_t sliceBytes =
            std::min<uint64_t>(static_cast<uint64_t>(MAX_DATA_SIZE), rankBytes - copiedBytes);
        CHK_RET(HcommLocalCopyOnThread(resource.ownCopyThread, ownOutput + copiedBytes,
            static_cast<char *>(param.inputPtr) + copiedBytes, sliceBytes));
        copiedBytes += sliceBytes;
    }
    CHK_RET(HcommThreadNotifyRecordOnThread(
        resource.ownCopyThread, completionThread, doneNotifyId));
    started = true;
    return HCCL_SUCCESS;
}

HcclResult WaitOwnCopyOnThread(
    ThreadHandle completionThread, uint32_t doneNotifyId, bool started)
{
    if (!started) {
        return HCCL_SUCCESS;
    }
    return static_cast<HcclResult>(
        HcommThreadNotifyWaitOnThread(completionThread, doneNotifyId, CUSTOM_TIMEOUT));
}

HcclResult StartOwnCopy(const OpParam &param, const AlgResourceCtx &resource,
    uint64_t rankBytes, bool &started)
{
    return StartOwnCopyAfterThread(param, resource, rankBytes,
        param.cpuThread, param.cpuThread, OWN_COPY_DONE_NOTIFY_ID, started);
}

HcclResult WaitOwnCopy(const OpParam &param, bool started)
{
    return WaitOwnCopyOnThread(
        param.cpuThread, OWN_COPY_DONE_NOTIFY_ID, started);
}

uint64_t AlignDown(uint64_t value, uint64_t alignment)
{
    return alignment == 0U ? value : value / alignment * alignment;
}

uint64_t ScaleFloor(uint64_t value, uint64_t numerator, uint64_t denominator)
{
    return value / denominator * numerator + (value % denominator) * numerator / denominator;
}

SlicePair SplitDirectBytes(uint64_t totalBytes)
{
    const uint64_t maxSliceBytes = static_cast<uint64_t>(MAX_DATA_SIZE);
    if (totalBytes <= maxSliceBytes) {
        return SlicePair{totalBytes, 0U};
    }
    const uint64_t alignedHalf =
        AlignDown(totalBytes / 2U, DMA_ALIGNMENT_BYTES);
    if (alignedHalf != 0U && alignedHalf <= maxSliceBytes
        && totalBytes - alignedHalf <= maxSliceBytes) {
        return SlicePair{alignedHalf, totalBytes - alignedHalf};
    }
    return SlicePair{maxSliceBytes, totalBytes - maxSliceBytes};
}

AsymmetricPartitions BuildAsymmetricPartitions(uint64_t batchBytes)
{
    AsymmetricPartitions partitions{};
    if (batchBytes <= INLINE_COPY_THRESHOLD_BYTES) {
        // 512KB输出用例走一次全量直达，不引入Relay Kernel和额外阶段。
        partitions.smallDirectBytes = batchBytes;
        partitions.smallDirectSlices = SplitDirectBytes(batchBytes);
        partitions.largeDirectBytes = batchBytes;
        partitions.largeDirectSlices = SplitDirectBytes(batchBytes);
        return partitions;
    }

    // Relay使用对齐前缀，所有非64B奇尾都留给一次覆盖的Direct后缀。
    const uint64_t smallRelayBytes =
        AlignDown(ScaleFloor(batchBytes, 4U, 9U), DMA_ALIGNMENT_BYTES);
    partitions.smallDirectBytes = batchBytes - smallRelayBytes;
    partitions.smallStripe0Bytes =
        AlignDown(smallRelayBytes / 2U, DMA_ALIGNMENT_BYTES);
    partitions.smallStripe1Bytes = smallRelayBytes - partitions.smallStripe0Bytes;

    partitions.largeTailBytes =
        AlignDown(ScaleFloor(batchBytes, 2U, 7U), DMA_ALIGNMENT_BYTES);
    partitions.largeDirectBytes = batchBytes - partitions.largeTailBytes;
    partitions.smallDirectSlices =
        SplitDirectBytes(partitions.smallDirectBytes);
    partitions.largeDirectSlices =
        SplitDirectBytes(partitions.largeDirectBytes);
    partitions.useRelay = partitions.smallStripe0Bytes != 0U
        && partitions.smallStripe1Bytes != 0U && partitions.largeTailBytes != 0U;
    return partitions;
}

HcclResult GetInputOutputTokens(const OpParam &param, const ExecPlan &plan,
    uint64_t &inputToken, uint64_t &outputToken)
{
    CHK_RET_CCU(HcommCcuGetMemToken(
        reinterpret_cast<uint64_t>(param.inputPtr), plan.rankBytes, &inputToken));
    CHK_RET_CCU(HcommCcuGetMemToken(
        reinterpret_cast<uint64_t>(param.outputPtr), plan.outputBytes, &outputToken));
    return HCCL_SUCCESS;
}

HcclResult LaunchZeroArgKernel(ThreadHandle thread, CcuKernelHandle kernel)
{
    const CcuResult ret = HcommCcuKernelLaunch(thread, kernel, nullptr, 0U);
    if (ret != CCU_SUCCESS) {
        HCCL_ERROR("[LaunchZeroArgKernel] Kernel launch failed, ccuRet=%d", ret);
        return ConvertCcuToHccl(ret);
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchRotatingPush(ThreadHandle thread, CcuKernelHandle kernel,
    uint64_t inputAddr, uint64_t outputAddr, uint64_t inputToken, uint64_t outputToken,
    uint64_t rankOutputOffset, const SlicePair &slices)
{
    const uint64_t taskArgs[SPECIALIZED_PUSH_TASK_ARGS] = {
        inputAddr,
        outputAddr,
        inputToken,
        outputToken,
        rankOutputOffset,
        slices.first,
        slices.second,
    };
    const CcuResult ret = HcommCcuKernelLaunch(
        thread, kernel, taskArgs, SPECIALIZED_PUSH_TASK_ARGS);
    if (ret != CCU_SUCCESS) {
        HCCL_ERROR("[LaunchRotatingPush] Kernel launch failed, ccuRet=%d", ret);
        return ConvertCcuToHccl(ret);
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchLatencyPush(ThreadHandle thread, CcuKernelHandle kernel,
    uint64_t inputAddr, uint64_t outputAddr, uint64_t inputToken,
    uint64_t outputToken, uint64_t rankOutputOffset, uint64_t transferBytes)
{
    const uint64_t taskArgs[LATENCY_PUSH_TASK_ARGS] = {
        inputAddr,
        outputAddr,
        inputToken,
        outputToken,
        rankOutputOffset,
        transferBytes,
    };
    const CcuResult ret = HcommCcuKernelLaunch(
        thread, kernel, taskArgs, LATENCY_PUSH_TASK_ARGS);
    if (ret != CCU_SUCCESS) {
        HCCL_ERROR("[LaunchLatencyPush] Kernel launch failed, ccuRet=%d", ret);
        return ConvertCcuToHccl(ret);
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchOutput512KernelSet(const OpParam &param,
    const AlgResourceCtx &resource, uint32_t kernelBase, bool zeroArgument,
    uint64_t inputAddr, uint64_t outputAddr, uint64_t inputToken,
    uint64_t outputToken, uint64_t rankOutputOffset, uint64_t transferBytes)
{
    const uint32_t groupCount = static_cast<uint32_t>(resource.threads.size());
    CHK_PRT_RET(groupCount == 0U || groupCount > 2U
            || kernelBase + groupCount > resource.ccuKernels.size(),
        HCCL_ERROR("[LaunchOutput512KernelSet] Invalid kernel set"),
        HCCL_E_INTERNAL);
    const auto launch = [&](uint32_t groupIndex) -> HcclResult {
        const ThreadHandle thread =
            groupIndex == 0U ? param.cpuThread : resource.threads[groupIndex];
        if (zeroArgument) {
            return LaunchZeroArgKernel(thread,
                resource.ccuKernels[kernelBase + groupIndex]);
        }
        return LaunchLatencyPush(thread,
            resource.ccuKernels[kernelBase + groupIndex], inputAddr,
            outputAddr, inputToken, outputToken, rankOutputOffset,
            transferBytes);
    };
    if (groupCount == 1U) {
        return launch(0U);
    }

    const ThreadHandle worker = resource.threads[1];
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        param.cpuThread, worker, DIE_WORKER_NOTIFY_ID)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        worker, DIE_WORKER_NOTIFY_ID, CUSTOM_TIMEOUT)));
    // Keep the main/intra then worker/cross mission order used by the proven
    // V35 latency path.  The two kernels remain on independent IO-Die threads.
    CHK_RET(launch(0U));
    CHK_RET(launch(1U));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        param.cpuThread, DIE_WORKER_NOTIFY_ID, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        worker, param.cpuThread, DIE_WORKER_NOTIFY_ID)));
    return HCCL_SUCCESS;
}

HcclResult LaunchClos4x1Push(ThreadHandle thread, CcuKernelHandle kernel,
    uint64_t inputAddr, uint64_t outputAddr, uint64_t inputToken, uint64_t outputToken,
    uint64_t rankOutputOffset, const Clos4x1Chunks &chunks)
{
    CHK_PRT_RET(chunks.count != 1U && chunks.count != CLOS_4X1_MAX_CHUNKS,
        HCCL_ERROR("[LaunchClos4x1Push] Invalid chunk count %u", chunks.count),
        HCCL_E_INTERNAL);
    uint64_t taskArgs[CLOS_4X1_FIXED_TASK_ARGS + CLOS_4X1_MAX_CHUNKS] = {
        inputAddr,
        outputAddr,
        inputToken,
        outputToken,
        rankOutputOffset,
        0U,
        0U,
        0U,
        0U,
    };
    for (uint32_t chunkIndex = 0; chunkIndex < chunks.count; ++chunkIndex) {
        taskArgs[CLOS_4X1_FIXED_TASK_ARGS + chunkIndex] = chunks.bytes[chunkIndex];
    }
    const uint32_t taskArgNum = CLOS_4X1_FIXED_TASK_ARGS + chunks.count;
    const CcuResult ret = HcommCcuKernelLaunch(thread, kernel, taskArgs, taskArgNum);
    if (ret != CCU_SUCCESS) {
        HCCL_ERROR("[LaunchClos4x1Push] Kernel launch failed, ccuRet=%d", ret);
        return ConvertCcuToHccl(ret);
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchDualSeedCross(ThreadHandle thread, CcuKernelHandle kernel,
    uint64_t inputAddr, uint64_t outputAddr, uint64_t inputToken, uint64_t outputToken,
    uint64_t rankOutputOffset, const AsymmetricPartitions &partitions, uint32_t phaseMode)
{
    const uint64_t taskArgs[DUAL_SEED_CROSS_TASK_ARGS] = {
        inputAddr,
        outputAddr,
        inputToken,
        outputToken,
        rankOutputOffset,
        partitions.smallDirectBytes,
        partitions.smallStripe0Bytes,
        partitions.smallStripe1Bytes,
        partitions.largeDirectBytes,
        partitions.largeTailBytes,
        phaseMode,
    };
    const CcuResult ret = HcommCcuKernelLaunch(
        thread, kernel, taskArgs, DUAL_SEED_CROSS_TASK_ARGS);
    if (ret != CCU_SUCCESS) {
        HCCL_ERROR("[LaunchDualSeedCross] Kernel launch failed, ccuRet=%d", ret);
        return ConvertCcuToHccl(ret);
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchWideDualSeedCross(ThreadHandle thread, CcuKernelHandle kernel,
    uint64_t inputAddr, uint64_t outputAddr, uint64_t inputToken,
    uint64_t outputToken, uint64_t rankOutputOffset,
    const AsymmetricPartitions &partitions)
{
    const uint64_t taskArgs[WIDE_DUAL_SEED_CROSS_TASK_ARGS] = {
        inputAddr,
        outputAddr,
        inputToken,
        outputToken,
        rankOutputOffset,
        partitions.smallDirectSlices.first,
        partitions.smallDirectSlices.second,
        partitions.smallStripe0Bytes,
        partitions.smallStripe1Bytes,
        partitions.largeDirectSlices.first,
        partitions.largeDirectSlices.second,
        partitions.largeTailBytes,
    };
    const CcuResult ret = HcommCcuKernelLaunch(
        thread, kernel, taskArgs, WIDE_DUAL_SEED_CROSS_TASK_ARGS);
    if (ret != CCU_SUCCESS) {
        HCCL_ERROR("[LaunchWideDualSeedCross] Kernel launch failed, ccuRet=%d", ret);
        return ConvertCcuToHccl(ret);
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchDualSeedRelay(const OpParam &param, const AlgResourceCtx &resource,
    const ExecPlan &plan, uint64_t batchOffset, uint64_t outputToken,
    const AsymmetricPartitions &partitions, uint32_t kernelIndex = 2U)
{
    uint32_t sourceRank0 = 0U;
    uint32_t sourceRank1 = 0U;
    if (param.myRank < 8U) {
        sourceRank0 = 8U + param.myRank / 2U;
        sourceRank1 = sourceRank0;
    } else {
        const uint32_t smallIndex = param.myRank - 8U;
        sourceRank0 = smallIndex;
        sourceRank1 = smallIndex + 4U;
    }
    const uint64_t taskArgs[DUAL_SEED_RELAY_TASK_ARGS] = {
        reinterpret_cast<uint64_t>(param.outputPtr),
        outputToken,
        static_cast<uint64_t>(sourceRank0) * plan.rankBytes + batchOffset,
        static_cast<uint64_t>(sourceRank1) * plan.rankBytes + batchOffset,
        partitions.smallDirectBytes,
        partitions.smallStripe0Bytes,
        partitions.smallStripe1Bytes,
        partitions.largeDirectBytes,
        partitions.largeTailBytes,
    };
    CHK_PRT_RET(kernelIndex >= resource.ccuKernels.size(),
        HCCL_ERROR("[LaunchDualSeedRelay] Invalid kernel index %u", kernelIndex),
        HCCL_E_INTERNAL);
    const CcuResult ret = HcommCcuKernelLaunch(param.cpuThread,
        resource.ccuKernels[kernelIndex], taskArgs, DUAL_SEED_RELAY_TASK_ARGS);
    if (ret != CCU_SUCCESS) {
        HCCL_ERROR("[LaunchDualSeedRelay] Kernel launch failed, ccuRet=%d", ret);
        return ConvertCcuToHccl(ret);
    }
    return HCCL_SUCCESS;
}

Clos4x1Chunks BuildClos4x1Chunks(const ExecPlan &plan, uint64_t remainingBytes)
{
    Clos4x1Chunks chunks{};
    if (plan.payload == PayloadType::LATENCY) {
        chunks.count = 1U;
        chunks.bytes[0] =
            std::min<uint64_t>(remainingBytes, static_cast<uint64_t>(MAX_DATA_SIZE));
        return chunks;
    }

    const uint64_t batchBytes = std::min<uint64_t>(remainingBytes,
        static_cast<uint64_t>(MAX_DATA_SIZE) * CLOS_4X1_MAX_CHUNKS);
    const uint64_t baseChunk =
        AlignDown(batchBytes / CLOS_4X1_MAX_CHUNKS, DMA_ALIGNMENT_BYTES);
    if (baseChunk == 0U
        || batchBytes - baseChunk * (CLOS_4X1_MAX_CHUNKS - 1U)
            > static_cast<uint64_t>(MAX_DATA_SIZE)) {
        chunks.count = 1U;
        chunks.bytes[0] =
            std::min<uint64_t>(remainingBytes, static_cast<uint64_t>(MAX_DATA_SIZE));
        return chunks;
    }
    chunks.count = CLOS_4X1_MAX_CHUNKS;
    for (uint32_t chunkIndex = 0; chunkIndex + 1U < chunks.count; ++chunkIndex) {
        chunks.bytes[chunkIndex] = baseChunk;
    }
    // 400MB+4B的4B奇尾只进入最后一个Chunk，不产生额外Launch或同步阶段。
    chunks.bytes[chunks.count - 1U] =
        batchBytes - baseChunk * (chunks.count - 1U);
    return chunks;
}

HcclResult Exec4x1CommonDiePush(
    const OpParam &param, const AlgResourceCtx &resource, const ExecPlan &plan)
{
    const bool exact512Registered =
        plan.payload == PayloadType::LATENCY
        && plan.rankBytes == 512ULL * 1024ULL
        && resource.hasRegistered4x1Exact512 != 0U;
    const uint32_t expectedKernelCount = exact512Registered ? 3U : 1U;
    CHK_PRT_RET(param.rankSize != 4U
            || resource.ccuKernels.size() != expectedKernelCount
            || resource.threads.size() != 1U
            || resource.peerRanksByGroup.size() != 1U,
        HCCL_ERROR("[Exec4x1CommonDiePush] Invalid specialized resources"),
        HCCL_E_INTERNAL);

    if (exact512Registered && g_4x1Exact512WarmDispatch) {
        // The public entry already matched communicator, stream and both
        // buffer pointers against the cache.  Kernel[2] is a zero-argument
        // warm graph and reuses the XN identity established by kernel[1].
        return LaunchZeroArgKernel(param.cpuThread, resource.ccuKernels[2]);
    }

    uint64_t inputToken = 0U;
    uint64_t outputToken = 0U;
    CHK_RET(GetInputOutputTokens(param, plan, inputToken, outputToken));

    if (exact512Registered
        && reinterpret_cast<uint64_t>(param.inputPtr)
            == resource.registered4x1InputAddr
        && reinterpret_cast<uint64_t>(param.outputPtr)
            == resource.registered4x1OutputAddr
        && inputToken == resource.registered4x1InputToken
        && outputToken == resource.registered4x1OutputToken) {
        // Cold registered launch: zero runtime arguments, but it republishes
        // the output address/token once so all channel XN variables are valid
        // before the warm cache becomes eligible.
        return LaunchZeroArgKernel(param.cpuThread, resource.ccuKernels[1]);
    }

    char *ownOutput =
        static_cast<char *>(param.outputPtr)
        + static_cast<uint64_t>(param.myRank) * plan.rankBytes;
    const bool needsOwnCopy = param.inputPtr != ownOutput;
    const bool useBandwidthKernel = plan.payload != PayloadType::LATENCY;
    bool ownCopyStarted = false;
    if (useBandwidthKernel && needsOwnCopy) {
        CHK_RET(StartOwnCopy(param, resource, plan.rankBytes, ownCopyStarted));
    }

    uint64_t processedBytes = 0U;
    while (processedBytes < plan.rankBytes) {
        const Clos4x1Chunks chunks =
            BuildClos4x1Chunks(plan, plan.rankBytes - processedBytes);
        const uint64_t rankOutputOffset =
            static_cast<uint64_t>(param.myRank) * plan.rankBytes + processedBytes;
        const HcclResult launchRet = LaunchClos4x1Push(
            param.cpuThread, resource.ccuKernels[0],
            reinterpret_cast<uint64_t>(param.inputPtr) + processedBytes,
            reinterpret_cast<uint64_t>(param.outputPtr),
            inputToken, outputToken, rankOutputOffset, chunks);
        if (launchRet != HCCL_SUCCESS) {
            (void)WaitOwnCopy(param, ownCopyStarted);
            return launchRet;
        }
        processedBytes += chunks.Total();
    }
    return WaitOwnCopy(param, ownCopyStarted);
}

HcclResult Exec4x1DualPlanePush(
    const OpParam &param, const AlgResourceCtx &resource, const ExecPlan &plan)
{
    CHK_PRT_RET(param.rankSize != 4U || plan.payload == PayloadType::LATENCY
            || resource.ccuKernels.size() != 2U || resource.threads.size() != 2U
            || resource.peerRanksByGroup.size() != 2U
            || resource.peerRanksByGroup[0].size() != 2U
            || resource.peerRanksByGroup[1].size() != 1U,
        HCCL_ERROR("[Exec4x1DualPlanePush] Invalid dual-plane resources"),
        HCCL_E_INTERNAL);

    uint64_t inputToken = 0U;
    uint64_t outputToken = 0U;
    CHK_RET(GetInputOutputTokens(param, plan, inputToken, outputToken));
    bool ownCopyStarted = false;
    CHK_RET(StartOwnCopy(param, resource, plan.rankBytes, ownCopyStarted));

    const ThreadHandle networkWorker = resource.threads[1];
    uint64_t processedBytes = 0U;
    while (processedBytes < plan.rankBytes) {
        const SlicePair slices = BuildSlicePair(plan, plan.rankBytes - processedBytes);
        const uint64_t inputAddr =
            reinterpret_cast<uint64_t>(param.inputPtr) + processedBytes;
        const uint64_t rankOutputOffset =
            static_cast<uint64_t>(param.myRank) * plan.rankBytes + processedBytes;

        // 主Thread服务2-Peer平面，worker服务1-Peer matching平面。Kernel注册时
        // 每个静态参数只含同一localDie的Channel；worker并不是用来绕过跨Die约束，
        // 而是让两个物理CCU Kernel的任务图能够并发下发。
        CHK_RET(HcommThreadNotifyRecordOnThread(
            param.cpuThread, networkWorker, DIE_WORKER_NOTIFY_ID));
        CHK_RET(HcommThreadNotifyWaitOnThread(
            networkWorker, DIE_WORKER_NOTIFY_ID, CUSTOM_TIMEOUT));

        HcclResult launchRet = LaunchRotatingPush(param.cpuThread,
            resource.ccuKernels[0], inputAddr,
            reinterpret_cast<uint64_t>(param.outputPtr),
            inputToken, outputToken, rankOutputOffset, slices);
        if (launchRet == HCCL_SUCCESS) {
            launchRet = LaunchRotatingPush(networkWorker,
                resource.ccuKernels[1], inputAddr,
                reinterpret_cast<uint64_t>(param.outputPtr),
                inputToken, outputToken, rankOutputOffset, slices);
        }
        if (launchRet != HCCL_SUCCESS) {
            (void)WaitOwnCopy(param, ownCopyStarted);
            return launchRet;
        }

        CHK_RET(HcommThreadNotifyRecordOnThread(
            networkWorker, param.cpuThread, DIE_WORKER_NOTIFY_ID));
        CHK_RET(HcommThreadNotifyWaitOnThread(
            param.cpuThread, DIE_WORKER_NOTIFY_ID, CUSTOM_TIMEOUT));
        processedBytes += slices.Total();
    }
    return WaitOwnCopy(param, ownCopyStarted);
}

HcclResult Exec8p4DualSeed(
    const OpParam &param, const AlgResourceCtx &resource, const ExecPlan &plan)
{
    CHK_PRT_RET(param.rankSize != 12U || resource.ccuKernels.size() != 3U
            || resource.threads.size() != 2U || resource.peerRanksByGroup.size() != 2U,
        HCCL_ERROR("[Exec8p4DualSeed] Invalid specialized resources"), HCCL_E_INTERNAL);
    uint64_t inputToken = 0U;
    uint64_t outputToken = 0U;
    CHK_RET(GetInputOutputTokens(param, plan, inputToken, outputToken));
    const uint64_t outputAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    const ThreadHandle crossThread = resource.threads[1];

    uint64_t processedBytes = 0U;
    while (processedBytes < plan.rankBytes) {
        const uint64_t batchBytes = std::min<uint64_t>(
            static_cast<uint64_t>(MAX_DATA_SIZE), plan.rankBytes - processedBytes);
        const SlicePair intraSlices{batchBytes, 0U};
        const AsymmetricPartitions partitions = BuildAsymmetricPartitions(batchBytes);
        const uint64_t inputAddr =
            reinterpret_cast<uint64_t>(param.inputPtr) + processedBytes;
        const uint64_t rankOutputOffset =
            static_cast<uint64_t>(param.myRank) * plan.rankBytes + processedBytes;

        // 机内全量传播与Cross Seed前缀并行。
        CHK_RET(HcommThreadNotifyRecordOnThread(
            param.cpuThread, crossThread, DIE_WORKER_NOTIFY_ID));
        CHK_RET(HcommThreadNotifyWaitOnThread(
            crossThread, DIE_WORKER_NOTIFY_ID, CUSTOM_TIMEOUT));
        CHK_RET(LaunchRotatingPush(param.cpuThread, resource.ccuKernels[0],
            inputAddr, outputAddr, inputToken, outputToken, rankOutputOffset, intraSlices));

        if (partitions.useRelay) {
            CHK_RET(LaunchDualSeedCross(crossThread, resource.ccuKernels[1],
                inputAddr, outputAddr, inputToken, outputToken,
                rankOutputOffset, partitions, CROSS_PHASE_SEED_ONLY));
            // READY只依赖Seed的组合EventWait和PostSync。
            CHK_RET(HcommThreadNotifyRecordOnThread(
                crossThread, param.cpuThread, DIE_WORKER_NOTIFY_ID));
            // Cross线程立即提交Direct后缀，不等待Relay。
            CHK_RET(LaunchDualSeedCross(crossThread, resource.ccuKernels[1],
                inputAddr, outputAddr, inputToken, outputToken,
                rankOutputOffset, partitions, CROSS_PHASE_DIRECT_ONLY));
            CHK_RET(HcommThreadNotifyRecordOnThread(
                crossThread, param.cpuThread, CROSS_DONE_NOTIFY_ID));

            // 主线程先完成机内Push，再等待Seed READY并启动Relay；
            // Relay与Cross Direct后缀重叠，最后等待Direct DONE。
            CHK_RET(HcommThreadNotifyWaitOnThread(
                param.cpuThread, DIE_WORKER_NOTIFY_ID, CUSTOM_TIMEOUT));
            CHK_RET(LaunchDualSeedRelay(
                param, resource, plan, processedBytes, outputToken, partitions));
            CHK_RET(HcommThreadNotifyWaitOnThread(
                param.cpuThread, CROSS_DONE_NOTIFY_ID, CUSTOM_TIMEOUT));
        } else {
            // 小数据一次Direct直达，跳过Seed和Relay两个阶段。
            CHK_RET(LaunchDualSeedCross(crossThread, resource.ccuKernels[1],
                inputAddr, outputAddr, inputToken, outputToken,
                rankOutputOffset, partitions, CROSS_PHASE_DIRECT_ONLY));
            CHK_RET(HcommThreadNotifyRecordOnThread(
                crossThread, param.cpuThread, DIE_WORKER_NOTIFY_ID));
            CHK_RET(HcommThreadNotifyWaitOnThread(
                param.cpuThread, DIE_WORKER_NOTIFY_ID, CUSTOM_TIMEOUT));
        }
        processedBytes += batchBytes;
    }
    return HCCL_SUCCESS;
}

HcclResult Exec8p4WidePipeline(
    const OpParam &param, const AlgResourceCtx &resource, const ExecPlan &plan)
{
    CHK_PRT_RET(param.rankSize != 12U || resource.ccuKernels.size() != 4U
            || resource.threads.size() != 2U
            || resource.peerRanksByGroup.size() != 2U,
        HCCL_ERROR("[Exec8p4WidePipeline] Invalid specialized resources"),
        HCCL_E_INTERNAL);

    uint64_t inputToken = 0U;
    uint64_t outputToken = 0U;
    CHK_RET(GetInputOutputTokens(param, plan, inputToken, outputToken));
    const uint64_t outputAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    const ThreadHandle crossThread = resource.threads[1];
    bool ownCopyStarted = false;

    const HcclResult networkRet = [&]() -> HcclResult {
        uint64_t processedBytes = 0U;
        while (processedBytes < plan.rankBytes) {
            // 4×1的“单Launch多Chunk”迁移为受16位Event约束的双切片：
            // 7条机内Channel用14位，8条跨机Channel恰好用满16位。
            const SlicePair intraSlices =
                BuildSlicePair(plan, plan.rankBytes - processedBytes);
            const uint64_t batchBytes = intraSlices.Total();
            const AsymmetricPartitions partitions =
                BuildAsymmetricPartitions(batchBytes);
            CHK_PRT_RET(!partitions.useRelay,
                HCCL_ERROR("[Exec8p4WidePipeline] Wide batch has no Relay partition"),
                HCCL_E_INTERNAL);

            const uint64_t inputAddr =
                reinterpret_cast<uint64_t>(param.inputPtr) + processedBytes;
            const uint64_t rankOutputOffset =
                static_cast<uint64_t>(param.myRank) * plan.rankBytes + processedBytes;

            // Cross Seed排在机内原始Push之前；二者通过独立Thread并行。
            CHK_RET(HcommThreadNotifyRecordOnThread(
                param.cpuThread, crossThread, DIE_WORKER_NOTIFY_ID));
            CHK_RET(HcommThreadNotifyWaitOnThread(
                crossThread, DIE_WORKER_NOTIFY_ID, CUSTOM_TIMEOUT));
            CHK_RET(LaunchWideDualSeedCross(crossThread, resource.ccuKernels[1],
                inputAddr, outputAddr, inputToken, outputToken,
                rankOutputOffset, partitions));

            if (!ownCopyStarted) {
                CHK_RET(StartOwnCopyAfterThread(param, resource, plan.rankBytes,
                    crossThread, param.cpuThread,
                    DEFERRED_OWN_COPY_DONE_NOTIFY_ID, ownCopyStarted));
            }

            // finalSync=false：机内原始数据完成后直接进入Relay，最终生命周期
            // 屏障由kernel[3]统一承担。
            CHK_RET(LaunchRotatingPush(param.cpuThread, resource.ccuKernels[0],
                inputAddr, outputAddr, inputToken, outputToken,
                rankOutputOffset, intraSlices));

            // Seed静态Kernel已完成定向ID4同步，Host READY后Relay读取一定可见。
            CHK_RET(HcommThreadNotifyRecordOnThread(
                crossThread, param.cpuThread, DIE_WORKER_NOTIFY_ID));
            CHK_RET(LaunchWideDualSeedCross(crossThread, resource.ccuKernels[2],
                inputAddr, outputAddr, inputToken, outputToken,
                rankOutputOffset, partitions));
            CHK_RET(HcommThreadNotifyRecordOnThread(
                crossThread, param.cpuThread, CROSS_DONE_NOTIFY_ID));

            CHK_RET(HcommThreadNotifyWaitOnThread(
                param.cpuThread, DIE_WORKER_NOTIFY_ID, CUSTOM_TIMEOUT));
            CHK_RET(LaunchDualSeedRelay(
                param, resource, plan, processedBytes, outputToken, partitions, 3U));
            CHK_RET(HcommThreadNotifyWaitOnThread(
                param.cpuThread, CROSS_DONE_NOTIFY_ID, CUSTOM_TIMEOUT));
            processedBytes += batchBytes;
        }
        return HCCL_SUCCESS;
    }();

    // 即便网络提交中途失败，也排空已经启动的OwnCopy，避免遗留跨调用通知。
    const HcclResult copyRet = WaitOwnCopyOnThread(
        param.cpuThread, DEFERRED_OWN_COPY_DONE_NOTIFY_ID, ownCopyStarted);
    return networkRet != HCCL_SUCCESS ? networkRet : copyRet;
}

HcclResult Exec8p4LatencyDirect(
    const OpParam &param, const AlgResourceCtx &resource, const ExecPlan &plan)
{
    CHK_PRT_RET(param.rankSize != 12U || plan.payload != PayloadType::LATENCY
            || resource.ccuKernels.size() != 2U
            || resource.threads.size() != 2U
            || resource.peerRanksByGroup.size() != 2U,
        HCCL_ERROR("[Exec8p4LatencyDirect] Invalid latency resources"),
        HCCL_E_INTERNAL);

    uint64_t inputToken = 0U;
    uint64_t outputToken = 0U;
    CHK_RET(GetInputOutputTokens(param, plan, inputToken, outputToken));
    const uint64_t inputAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outputAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    const uint64_t rankOutputOffset =
        static_cast<uint64_t>(param.myRank) * plan.rankBytes;
    const ThreadHandle crossThread = resource.threads[1];

    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        param.cpuThread, crossThread, DIE_WORKER_NOTIFY_ID)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        crossThread, DIE_WORKER_NOTIFY_ID, CUSTOM_TIMEOUT)));
    CHK_RET(LaunchLatencyPush(
        param.cpuThread, resource.ccuKernels[0],
        inputAddr, outputAddr, inputToken, outputToken,
        rankOutputOffset, plan.rankBytes));
    CHK_RET(LaunchLatencyPush(
        crossThread, resource.ccuKernels[1],
        inputAddr, outputAddr, inputToken, outputToken,
        rankOutputOffset, plan.rankBytes));

    // 与测试点10相同：先把主线程Wait压入队列，再让Cross Worker在
    // 自己的微内核之后Record，确保主Stream只在两个IO Die均完成后返回。
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        param.cpuThread, DIE_WORKER_NOTIFY_ID, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        crossThread, param.cpuThread, DIE_WORKER_NOTIFY_ID)));
    return HCCL_SUCCESS;
}

HcclResult ExecOutput512Static(const OpParam &param,
    const AlgResourceCtx &resource, const ExecPlan &plan)
{
    const uint32_t groupCount = static_cast<uint32_t>(resource.threads.size());
    const uint64_t targetBytes = 512ULL * 1024ULL;
    const uint64_t delta = plan.outputBytes >= targetBytes
        ? plan.outputBytes - targetBytes : targetBytes - plan.outputBytes;
    CHK_PRT_RET(resource.hasOutput512StaticSession == 0U
            || (param.rankSize != 4U && param.rankSize != 12U
                && param.rankSize != 16U)
            || delta >= static_cast<uint64_t>(param.rankSize) * sizeof(float)
            || groupCount == 0U || groupCount > 2U
            || resource.peerRanksByGroup.size() != groupCount
            || resource.ccuKernels.size() != groupCount,
        HCCL_ERROR("[ExecOutput512Static] Invalid resident resources"),
        HCCL_E_INTERNAL);

    const uint64_t inputAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outputAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    uint64_t inputToken = 0U;
    uint64_t outputToken = 0U;
    CHK_RET(GetInputOutputTokens(param, plan, inputToken, outputToken));
    const uint64_t rankOutputOffset =
        static_cast<uint64_t>(param.myRank) * plan.rankBytes;
    // Keep B's group order, but execute the same loader-proven dynamic graph
    // as V35 with live addresses and tokens.
    return LaunchOutput512KernelSet(param, resource, 0U, false,
        inputAddr, outputAddr, inputToken, outputToken,
        rankOutputOffset, plan.rankBytes);
}

} // namespace

ExecPlan SelectExecPlan(const OpParam &param, uint32_t topologyType)
{
    ExecPlan plan{};
    plan.topology = static_cast<TopologyType>(topologyType);
    plan.rankBytes = param.count * sizeof(float);
    plan.outputBytes = plan.rankBytes * param.rankSize;
    plan.payload = plan.rankBytes <= INLINE_COPY_THRESHOLD_BYTES
        ? PayloadType::LATENCY
        : (plan.rankBytes % DMA_ALIGNMENT_BYTES == 0 ? PayloadType::BANDWIDTH_ALIGNED
                                                     : PayloadType::BANDWIDTH_TAIL);
    plan.inlineOwnCopy = plan.payload == PayloadType::LATENCY;
    return plan;
}

SlicePair BuildSlicePair(const ExecPlan &plan, uint64_t remainingBytes)
{
    const uint64_t maxSliceBytes = static_cast<uint64_t>(MAX_DATA_SIZE);
    const uint64_t batchBytes = std::min<uint64_t>(remainingBytes, maxSliceBytes * 2);
    if (batchBytes <= maxSliceBytes) {
        return SlicePair{batchBytes, 0};
    }

    // 仅在单次批次确实超过256MB时生成第二事件；Host直接传绝对目标地址，
    // Kernel不再包含peerRank次地址加法。
    if (plan.payload == PayloadType::BANDWIDTH_TAIL) {
        const uint64_t first =
            (batchBytes / 2 / DMA_ALIGNMENT_BYTES) * DMA_ALIGNMENT_BYTES;
        const uint64_t second = batchBytes - first;
        if (first != 0 && first <= maxSliceBytes && second <= maxSliceBytes) {
            return SlicePair{first, second};
        }
    }
    return SlicePair{maxSliceBytes, batchBytes - maxSliceBytes};
}

HcclResult ExecOp(const OpParam &param)
{
    CHK_PTR_NULL(param.resCtx);
    CHK_PRT_RET(param.ctxSize == 0,
        HCCL_ERROR("[ExecOp] Empty resource context"), HCCL_E_INTERNAL);

    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> seq(ctx, ctx + param.ctxSize);
    AlgResourceCtx resource{};
    resource.DeSerialize(seq);

    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(param.dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("[ExecOp] Only HCCL_DATA_TYPE_FP32 is supported"), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE
            || param.myRank >= param.rankSize,
        HCCL_ERROR("[ExecOp] Invalid rank information"), HCCL_E_PARA);
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / sizeof(float),
        HCCL_ERROR("[ExecOp] Input byte size overflows uint64_t"), HCCL_E_PARA);

    const ExecPlan plan = SelectExecPlan(param, resource.topologyType);
    CHK_PRT_RET(plan.rankBytes > std::numeric_limits<uint64_t>::max() / param.rankSize,
        HCCL_ERROR("[ExecOp] Output byte size overflows uint64_t"), HCCL_E_PARA);
    if (param.rankSize == 1) {
        if (param.inputPtr != param.outputPtr) {
            CHK_RET(HcommLocalCopyOnThread(
                param.cpuThread, param.outputPtr, param.inputPtr, plan.rankBytes));
        }
        return HCCL_SUCCESS;
    }

    if (resource.algorithmMode == ALGORITHM_OUTPUT_512_STATIC) {
        return ExecOutput512Static(param, resource, plan);
    }

    if (resource.algorithmMode == ALGORITHM_4X1_COMMON_DIE_PUSH) {
        return Exec4x1CommonDiePush(param, resource, plan);
    }
    if (resource.algorithmMode == ALGORITHM_4X1_DUAL_PLANE_PUSH) {
        return Exec4x1DualPlanePush(param, resource, plan);
    }
    if (resource.algorithmMode == ALGORITHM_8P4_DUAL_SEED) {
        return Exec8p4DualSeed(param, resource, plan);
    }
    if (resource.algorithmMode == ALGORITHM_2X8_V22_HYBRID) {
        return Exec2x8V22(param, resource, plan);
    }
    if (resource.algorithmMode == ALGORITHM_8P4_WIDE_PIPELINE) {
        return Exec8p4WidePipeline(param, resource, plan);
    }
    if (resource.algorithmMode == ALGORITHM_8P4_LATENCY_DIRECT) {
        return Exec8p4LatencyDirect(param, resource, plan);
    }
    CHK_PRT_RET(resource.algorithmMode != ALGORITHM_V10_PULL,
        HCCL_ERROR("[ExecOp] Unknown algorithm mode %u", resource.algorithmMode), HCCL_E_INTERNAL);

    // Mode 0 continues to use the original V10 PeerLanePullMatrix fallback.
    // The rank-16 V22 path has already returned through the explicit mode-4 branch.
    CHK_PRT_RET(resource.ccuKernels.empty()
            || resource.ccuKernels.size() > MAX_DIE_GROUPS
            || resource.threads.size() != resource.ccuKernels.size()
            || resource.peerRanksByGroup.size() != resource.ccuKernels.size(),
        HCCL_ERROR("[ExecOp] Invalid peer-lane resources"), HCCL_E_INTERNAL);

    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(
        reinterpret_cast<uint64_t>(param.inputPtr), plan.rankBytes, &inputToken));
    CHK_RET_CCU(HcommCcuGetMemToken(
        reinterpret_cast<uint64_t>(param.outputPtr), plan.outputBytes, &outputToken));

    char *ownOutput =
        static_cast<char *>(param.outputPtr) + static_cast<uint64_t>(param.myRank) * plan.rankBytes;
    const bool needsOwnCopy = param.inputPtr != ownOutput;
    bool ownCopyStarted = false;
    if (needsOwnCopy && !plan.inlineOwnCopy) {
        CHK_RET(StartOwnCopy(param, resource, plan.rankBytes, ownCopyStarted));
    }

    uint64_t processedBytes = 0;
    while (processedBytes < plan.rankBytes) {
        const SlicePair slices = BuildSlicePair(plan, plan.rankBytes - processedBytes);
        const uint64_t inlineCopyBytes =
            needsOwnCopy && plan.inlineOwnCopy && processedBytes == 0 ? plan.rankBytes : 0;
        const HcclResult ret = LaunchDieGroups(param, resource, plan,
            reinterpret_cast<uint64_t>(param.inputPtr), inputToken,
            reinterpret_cast<uint64_t>(param.outputPtr), outputToken,
            processedBytes, slices, ownOutput, param.inputPtr, inlineCopyBytes);
        if (ret != HCCL_SUCCESS) {
            (void)WaitOwnCopy(param, ownCopyStarted);
            return ret;
        }
        processedBytes += slices.Total();
    }
    return WaitOwnCopy(param, ownCopyStarted);
}

HcclResult Exec4x1Exact512WarmOp(const OpParam &param)
{
    const bool previous = g_4x1Exact512WarmDispatch;
    g_4x1Exact512WarmDispatch = true;
    const HcclResult ret = ExecOp(param);
    g_4x1Exact512WarmDispatch = previous;
    return ret;
}

} // namespace ops_hccl
