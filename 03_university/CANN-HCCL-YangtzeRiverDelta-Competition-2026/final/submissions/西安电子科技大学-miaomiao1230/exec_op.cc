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
#include <limits>
#include <vector>

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace {

constexpr uint32_t FIRST_SERVER_SIZE = 8;
constexpr uint64_t SMALL_MESSAGE_BYTES = 512ULL * 1024ULL;
constexpr uint32_t HIERARCHICAL_JOIN_NOTIFY_INDEX = 0;
constexpr size_t HIERARCHICAL_WINDOW_SIZE = 2;
constexpr size_t BALANCED_FUSED_WINDOW_SIZE = 2;
// Variant C: 8+4 split tuned UP to 2/3 mesh (from 3/5=0.60 -> 0.667).
// Gives the intra-server mesh a larger share and the Clos cross path a smaller
// one. Wins if the cross return path / server1 side is the current bottleneck.
constexpr uint64_t BALANCED_MESH_NUMERATOR = 2;
constexpr uint64_t BALANCED_SPLIT_DENOMINATOR = 3;
constexpr uint32_t PARALLEL_START_NOTIFY_INDEX = 0;
constexpr uint32_t PARALLEL_PHASE_NOTIFY_INDEX = 1;
constexpr uint32_t SMALL_DIRECT_NOTIFY_INDEX = 0;
constexpr uint32_t DUAL_STRIPE_NOTIFY_INDEX = 0;
constexpr bool SMALL_DUAL_DIE_PARALLEL = true;
constexpr size_t HIERARCHICAL_INTRA_SCATTER_KERNEL = 0;
constexpr size_t HIERARCHICAL_INTER_TRANSFER_KERNEL = 1;
constexpr size_t HIERARCHICAL_INTRA_ALLGATHER_KERNEL = 2;
constexpr size_t PARALLEL_INTRA_KERNEL = 0;
constexpr size_t PARALLEL_INTER_KERNEL = 1;
constexpr size_t RELAY_SOURCE_INTRA_SCATTER_KERNEL = 0;
constexpr size_t RELAY_SOURCE_INTER_TRANSFER_KERNEL = 1;
constexpr size_t RELAY_SOURCE_INTRA_ALLGATHER_KERNEL = 2;
constexpr size_t RELAY_REMOTE_INTER_TRANSFER_KERNEL = 0;
constexpr size_t RELAY_REMOTE_INTRA_ALLGATHER_KERNEL = 1;
constexpr size_t RELAY_WINDOW_SIZE = 2;
using HierarchicalTaskArgs = std::array<uint64_t, 4>;
struct BalancedFusedTaskArgs {
    std::array<uint64_t, 4> intra;
    std::array<uint64_t, 7> fused;
    std::array<uint64_t, 4> cross;
};
using RelayTaskArgs = std::array<uint64_t, 4>;

HcclResult LaunchSmallDualDieDirect(const OpParam &param,
    const AlgResourceCtx &resource, const uint64_t *taskArgs)
{
    const bool isRoot = param.myRank == param.root;
    if (!isRoot) {
        if (resource.ccuKernels.size() != 1 || !resource.threads.empty()) {
            HCCL_ERROR("Small direct receiver resource mismatch: rank=%u kernels=%zu workers=%zu",
                param.myRank, resource.ccuKernels.size(), resource.threads.size());
            return HCCL_E_INTERNAL;
        }
        CHK_RET_CCU(HcommCcuKernelLaunch(
            param.cpuThread, resource.ccuKernels[0], taskArgs, 3));
        return HCCL_SUCCESS;
    }

    const size_t expectedWorkerCount = SMALL_DUAL_DIE_PARALLEL ? 1 : 0;
    if (resource.ccuKernels.size() != 2 ||
        resource.threads.size() != expectedWorkerCount) {
        HCCL_ERROR("Small direct root resource mismatch: rank=%u kernels=%zu workers=%zu expectedWorkers=%zu",
            param.myRank, resource.ccuKernels.size(), resource.threads.size(),
            expectedWorkerCount);
        return HCCL_E_INTERNAL;
    }

    if (!SMALL_DUAL_DIE_PARALLEL) {
        CHK_RET_CCU(HcommCcuKernelLaunch(
            param.cpuThread, resource.ccuKernels[0], taskArgs, 3));
        CHK_RET_CCU(HcommCcuKernelLaunch(
            param.cpuThread, resource.ccuKernels[1], taskArgs, 3));
        return HCCL_SUCCESS;
    }

    const ThreadHandle mainThread = param.cpuThread;
    const ThreadHandle workerThread = resource.threads[0];
    // The worker stream begins with exactly one wait.  The two one-die tree
    // kernels then transfer the same buffer directly to disjoint peer sets.
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        mainThread, workerThread, SMALL_DIRECT_NOTIFY_INDEX)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        workerThread, SMALL_DIRECT_NOTIFY_INDEX, CUSTOM_TIMEOUT)));

    // Queue the inter-server die first: it has the longer physical path and,
    // on 2x8, one more child.  The main die proceeds independently.
    CHK_RET_CCU(HcommCcuKernelLaunch(
        workerThread, resource.ccuKernels[1], taskArgs, 3));
    CHK_RET_CCU(HcommCcuKernelLaunch(
        mainThread, resource.ccuKernels[0], taskArgs, 3));

    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        workerThread, mainThread, SMALL_DIRECT_NOTIFY_INDEX)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        mainThread, SMALL_DIRECT_NOTIFY_INDEX, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult LaunchDualStripePipeline(const OpParam &param,
    const AlgResourceCtx &resource, uint64_t baseAddress, uint64_t memoryToken,
    uint64_t firstStripeBytes)
{
    if (resource.ccuKernels.size() != 2 || resource.threads.size() != 1) {
        HCCL_ERROR("Dual-stripe resource mismatch: rank=%u kernels=%zu workers=%zu",
            param.myRank, resource.ccuKernels.size(), resource.threads.size());
        return HCCL_E_INTERNAL;
    }

    const ThreadHandle mainThread = param.cpuThread;
    const ThreadHandle workerThread = resource.threads[0];
    const uint64_t firstArgs[] = {baseAddress, memoryToken};
    const uint64_t secondArgs[] = {baseAddress + firstStripeBytes, memoryToken};

    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        mainThread, workerThread, DUAL_STRIPE_NOTIFY_INDEX)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        workerThread, DUAL_STRIPE_NOTIFY_INDEX, CUSTOM_TIMEOUT)));
    CHK_RET_CCU(HcommCcuKernelLaunch(
        workerThread, resource.ccuKernels[1], secondArgs, 2));
    CHK_RET_CCU(HcommCcuKernelLaunch(
        mainThread, resource.ccuKernels[0], firstArgs, 2));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        workerThread, mainThread, DUAL_STRIPE_NOTIFY_INDEX)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        mainThread, DUAL_STRIPE_NOTIFY_INDEX, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult LaunchHierarchicalWindow(const OpParam &param,
    const AlgResourceCtx &resource, const std::vector<HierarchicalTaskArgs> &taskArgs)
{
    if (taskArgs.empty() || taskArgs.size() > HIERARCHICAL_WINDOW_SIZE) {
        HCCL_ERROR("Invalid hierarchical window size %zu", taskArgs.size());
        return HCCL_E_INTERNAL;
    }

    const bool rankOnRootServer = param.myRank < FIRST_SERVER_SIZE;
    if (!rankOnRootServer) {
        if (resource.ccuKernels.size() != 1 || !resource.threads.empty()) {
            HCCL_ERROR("Hierarchical receiver resource mismatch: kernels=%zu workers=%zu",
                resource.ccuKernels.size(), resource.threads.size());
            return HCCL_E_INTERNAL;
        }
        for (const HierarchicalTaskArgs &args : taskArgs) {
            CHK_RET_CCU(HcommCcuKernelLaunch(
                param.cpuThread, resource.ccuKernels[0], args.data(), 4));
        }
        return HCCL_SUCCESS;
    }

    if (resource.ccuKernels.size() != 3 || resource.threads.size() != 1) {
        HCCL_ERROR("Hierarchical root-server resource mismatch: kernels=%zu workers=%zu",
            resource.ccuKernels.size(), resource.threads.size());
        return HCCL_E_INTERNAL;
    }

    const ThreadHandle mainThread = param.cpuThread;
    const ThreadHandle workerThread = resource.threads[0];
    for (size_t chunkIdx = 0; chunkIdx < taskArgs.size(); ++chunkIdx) {
        const HierarchicalTaskArgs &args = taskArgs[chunkIdx];
        CHK_RET_CCU(HcommCcuKernelLaunch(mainThread,
            resource.ccuKernels[HIERARCHICAL_INTRA_SCATTER_KERNEL], args.data(), 4));

        // Each chunk owns a distinct main-to-worker notify.  The worker waits
        // after the corresponding Scatter and processes inter transfers in
        // order, while the main die advances local AllGather and the next
        // chunk on disjoint buffer ranges.
        const uint32_t notifyIndex = static_cast<uint32_t>(chunkIdx);
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            mainThread, workerThread, notifyIndex)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            workerThread, notifyIndex, CUSTOM_TIMEOUT)));
        CHK_RET_CCU(HcommCcuKernelLaunch(workerThread,
            resource.ccuKernels[HIERARCHICAL_INTER_TRANSFER_KERNEL], args.data(), 4));
        CHK_RET_CCU(HcommCcuKernelLaunch(mainThread,
            resource.ccuKernels[HIERARCHICAL_INTRA_ALLGATHER_KERNEL], args.data(), 4));
    }

    // One opposite-direction join closes the complete window.  Index 0 is
    // safe here because it targets the main Thread's independent notify set.
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        workerThread, mainThread, HIERARCHICAL_JOIN_NOTIFY_INDEX)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        mainThread, HIERARCHICAL_JOIN_NOTIFY_INDEX, CUSTOM_TIMEOUT)));

    return HCCL_SUCCESS;
}

HcclResult LaunchBalancedFusedWindow(const OpParam &param,
    const AlgResourceCtx &resource,
    const std::vector<BalancedFusedTaskArgs> &taskArgs)
{
    if (taskArgs.empty() ||
        taskArgs.size() > BALANCED_FUSED_WINDOW_SIZE ||
        resource.ccuKernels.size() != 3 ||
        resource.threads.size() != 1) {
        HCCL_ERROR("Invalid balanced-fused window: rank=%u tasks=%zu kernels=%zu workers=%zu",
            param.myRank, taskArgs.size(), resource.ccuKernels.size(),
            resource.threads.size());
        return HCCL_E_INTERNAL;
    }

    constexpr size_t FIRST_KERNEL = 0;
    constexpr size_t FUSED_INTER_KERNEL = 1;
    constexpr size_t FINAL_INTRA_KERNEL = 2;
    const ThreadHandle mainThread = param.cpuThread;
    const ThreadHandle workerThread = resource.threads[0];
    const bool rankOnSourceServer = param.myRank < FIRST_SERVER_SIZE;

    for (size_t chunkIdx = 0; chunkIdx < taskArgs.size(); ++chunkIdx) {
        const BalancedFusedTaskArgs &args = taskArgs[chunkIdx];
        const uint32_t notifyIndex = static_cast<uint32_t>(chunkIdx);
        if (rankOnSourceServer) {
            if (param.myRank == param.root) {
                // The root's cross range is valid at entry, so its Clos plane
                // can start before the source-server Scatter.
                CHK_RET(static_cast<HcclResult>(
                    HcommThreadNotifyRecordOnThread(
                        mainThread, workerThread, notifyIndex)));
                CHK_RET(static_cast<HcclResult>(
                    HcommThreadNotifyWaitOnThread(
                        workerThread, notifyIndex, CUSTOM_TIMEOUT)));
                CHK_RET_CCU(HcommCcuKernelLaunch(workerThread,
                    resource.ccuKernels[FUSED_INTER_KERNEL],
                    args.fused.data(), args.fused.size()));
                CHK_RET_CCU(HcommCcuKernelLaunch(mainThread,
                    resource.ccuKernels[FIRST_KERNEL],
                    args.intra.data(), args.intra.size()));
            } else {
                // A non-root source slice becomes valid only after Scatter.
                CHK_RET_CCU(HcommCcuKernelLaunch(mainThread,
                    resource.ccuKernels[FIRST_KERNEL],
                    args.intra.data(), args.intra.size()));
                CHK_RET(static_cast<HcclResult>(
                    HcommThreadNotifyRecordOnThread(
                        mainThread, workerThread, notifyIndex)));
                CHK_RET(static_cast<HcclResult>(
                    HcommThreadNotifyWaitOnThread(
                        workerThread, notifyIndex, CUSTOM_TIMEOUT)));
                CHK_RET_CCU(HcommCcuKernelLaunch(workerThread,
                    resource.ccuKernels[FUSED_INTER_KERNEL],
                    args.fused.data(), args.fused.size()));
            }
            CHK_RET_CCU(HcommCcuKernelLaunch(mainThread,
                resource.ccuKernels[FINAL_INTRA_KERNEL],
                args.intra.data(), args.intra.size()));
            continue;
        }

        // Stage 0 receives this remote rank's C block.  Afterwards the local
        // Mesh AllGather and the fused Clos A-receive/C-return are independent.
        CHK_RET_CCU(HcommCcuKernelLaunch(mainThread,
            resource.ccuKernels[FIRST_KERNEL],
            args.fused.data(), args.fused.size()));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            mainThread, workerThread, notifyIndex)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            workerThread, notifyIndex, CUSTOM_TIMEOUT)));
        CHK_RET_CCU(HcommCcuKernelLaunch(workerThread,
            resource.ccuKernels[FINAL_INTRA_KERNEL],
            args.cross.data(), args.cross.size()));
        CHK_RET_CCU(HcommCcuKernelLaunch(mainThread,
            resource.ccuKernels[FUSED_INTER_KERNEL],
            args.fused.data(), args.fused.size()));
    }

    // The target Thread owns an independent notify set, so index 0 can be
    // reused for this opposite-direction join after both window chunks.
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        workerThread, mainThread, HIERARCHICAL_JOIN_NOTIFY_INDEX)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        mainThread, HIERARCHICAL_JOIN_NOTIFY_INDEX, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult LaunchParallelMeshNhrChunk(const OpParam &param,
    const AlgResourceCtx &resource, const uint64_t *intraFirstArgs,
    const uint64_t *interFirstArgs, const uint64_t *intraSecondArgs,
    const uint64_t *interSecondArgs)
{
    if (resource.ccuKernels.size() != 2 || resource.threads.size() != 1) {
        HCCL_ERROR("Parallel Mesh-NHR resource mismatch: rank=%u kernels=%zu workers=%zu",
            param.myRank, resource.ccuKernels.size(), resource.threads.size());
        return HCCL_E_INTERNAL;
    }

    const ThreadHandle mainThread = param.cpuThread;
    const ThreadHandle workerThread = resource.threads[0];

    // Every slave stream must start with a local WAIT.  Use a dedicated
    // start notification so the worker can be released before either phase-1
    // kernel is queued, preserving the intended cross-die concurrency.
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        mainThread, workerThread, PARALLEL_START_NOTIFY_INDEX)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        workerThread, PARALLEL_START_NOTIFY_INDEX, CUSTOM_TIMEOUT)));

    // Phase 1 follows the official multi-level Broadcast schedule.  The first
    // half is distributed by a complete local Mesh1D Scatter+AllGather on the
    // root server while the second half crosses only the root pair.
    if (param.myRank < FIRST_SERVER_SIZE) {
        CHK_RET_CCU(HcommCcuKernelLaunch(mainThread,
            resource.ccuKernels[PARALLEL_INTRA_KERNEL], intraFirstArgs, 4));
    }
    if (param.myRank == 0 || param.myRank == FIRST_SERVER_SIZE) {
        CHK_RET_CCU(HcommCcuKernelLaunch(workerThread,
            resource.ccuKernels[PARALLEL_INTER_KERNEL], interFirstArgs, 3));
    }

    // Bidirectional one-record/one-wait barrier.  The worker-side phase
    // notification is distinct from the start gate, so even ranks without a
    // phase-1 kernel cannot issue two Records to one unconsumed Wait.
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        mainThread, workerThread, PARALLEL_PHASE_NOTIFY_INDEX)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        workerThread, PARALLEL_PHASE_NOTIFY_INDEX, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        workerThread, mainThread, PARALLEL_START_NOTIFY_INDEX)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        mainThread, PARALLEL_START_NOTIFY_INDEX, CUSTOM_TIMEOUT)));

    // Phase 2 crosses the first half on all eight paired NHR planes while both
    // servers distribute the second half locally.  Layer-0 and layer-1 work
    // remains on different kernels and different IO dies.
    CHK_RET_CCU(HcommCcuKernelLaunch(mainThread,
        resource.ccuKernels[PARALLEL_INTRA_KERNEL], intraSecondArgs, 4));
    CHK_RET_CCU(HcommCcuKernelLaunch(workerThread,
        resource.ccuKernels[PARALLEL_INTER_KERNEL], interSecondArgs, 3));

    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        workerThread, mainThread, PARALLEL_PHASE_NOTIFY_INDEX)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        mainThread, PARALLEL_PHASE_NOTIFY_INDEX, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult LaunchRelayWindow(const OpParam &param,
    const AlgResourceCtx &resource, const std::vector<RelayTaskArgs> &taskArgs)
{
    if (taskArgs.empty() || taskArgs.size() > RELAY_WINDOW_SIZE ||
        resource.threads.size() != 1) {
        HCCL_ERROR("Relay window resource mismatch: rank=%u chunks=%zu workers=%zu",
            param.myRank, taskArgs.size(), resource.threads.size());
        return HCCL_E_INTERNAL;
    }

    const bool rankOnSourceServer = param.myRank < FIRST_SERVER_SIZE;
    const size_t expectedKernelCount = rankOnSourceServer ? 3 : 2;
    if (resource.ccuKernels.size() != expectedKernelCount) {
        HCCL_ERROR("Relay kernel mismatch: rank=%u kernels=%zu expected=%zu",
            param.myRank, resource.ccuKernels.size(), expectedKernelCount);
        return HCCL_E_INTERNAL;
    }

    const ThreadHandle mainThread = param.cpuThread;
    const ThreadHandle workerThread = resource.threads[0];
    for (size_t chunkIdx = 0; chunkIdx < taskArgs.size(); ++chunkIdx) {
        const size_t mainKernel = rankOnSourceServer ?
            RELAY_SOURCE_INTRA_SCATTER_KERNEL : RELAY_REMOTE_INTER_TRANSFER_KERNEL;
        CHK_RET_CCU(HcommCcuKernelLaunch(mainThread,
            resource.ccuKernels[mainKernel], taskArgs[chunkIdx].data(), 4));

        // Each chunk uses a distinct worker notify.  This permits both chunks
        // to be queued without two Records ever targeting one unconsumed Wait.
        const uint32_t notifyIndex = static_cast<uint32_t>(chunkIdx);
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            mainThread, workerThread, notifyIndex)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            workerThread, notifyIndex, CUSTOM_TIMEOUT)));

        const size_t workerKernel = rankOnSourceServer ?
            RELAY_SOURCE_INTER_TRANSFER_KERNEL : RELAY_REMOTE_INTRA_ALLGATHER_KERNEL;
        CHK_RET_CCU(HcommCcuKernelLaunch(workerThread,
            resource.ccuKernels[workerKernel], taskArgs[chunkIdx].data(), 4));
    }

    // The source-server main die performs both local Scatters first.  This
    // lets the layer-1 worker transfer chunk 0 while chunk 1 is distributed.
    if (rankOnSourceServer) {
        for (const RelayTaskArgs &args : taskArgs) {
            CHK_RET_CCU(HcommCcuKernelLaunch(mainThread,
                resource.ccuKernels[RELAY_SOURCE_INTRA_ALLGATHER_KERNEL],
                args.data(), 4));
        }
    }

    // One worker-to-main join closes the complete two-chunk window.  Reusing
    // notify index 0 is safe because it belongs to the opposite Thread.
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        workerThread, mainThread, HIERARCHICAL_JOIN_NOTIFY_INDEX)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        mainThread, HIERARCHICAL_JOIN_NOTIFY_INDEX, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

} // namespace

namespace ops_hccl {
HcclResult ExecOp(const OpParam &param)
{
    char *ctx = static_cast<char *>(param.resCtx);
    CHK_PTR_NULL(ctx);
    std::vector<char> sequence(ctx, ctx + param.ctxSize);
    AlgResourceCtx resource;
    resource.DeSerialize(sequence);

    const auto sizeIter = SIZE_TABLE.find(param.dataType);
    if (sizeIter == SIZE_TABLE.end()) {
        HCCL_ERROR("Unsupported data type %d", static_cast<int32_t>(param.dataType));
        return HCCL_E_NOT_SUPPORT;
    }

    if (param.count > std::numeric_limits<uint64_t>::max() / sizeIter->second) {
        HCCL_ERROR("Broadcast byte count overflows uint64_t");
        return HCCL_E_PARA;
    }
    const uint64_t totalBytes = param.count * sizeIter->second;
    if (totalBytes == 0) {
        return HCCL_SUCCESS;
    }

    const uint64_t baseAddress = reinterpret_cast<uint64_t>(param.outputPtr);
    uint64_t memoryToken = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(baseAddress, totalBytes, &memoryToken));

    const bool useSmallDualDieDirect = totalBytes <= SMALL_MESSAGE_BYTES &&
        (param.rankSize == 12 || param.rankSize == 16) && param.root < FIRST_SERVER_SIZE;
    const bool useDualStripePipeline = totalBytes > SMALL_MESSAGE_BYTES &&
        param.rankSize == 2 * FIRST_SERVER_SIZE && param.root == 0;
    // v33 merge: 4x1 large messages run v27's chain4 pipeline (see broadcast.cc).
    const bool useChain4 = totalBytes > SMALL_MESSAGE_BYTES && param.rankSize == 4;
    const bool useStaticPipeline = totalBytes > SMALL_MESSAGE_BYTES &&
        param.rankSize == 2 * FIRST_SERVER_SIZE &&
        !useDualStripePipeline;
    const bool useParallelMeshNhr = false;
    const bool useRelay = false;
    const bool useBalancedFused =
        totalBytes >= BROADCAST_LARGE_MESSAGE_THRESHOLD &&
        param.rankSize == 12 && param.root == 0;
    const bool useHierarchical = totalBytes >= BROADCAST_LARGE_MESSAGE_THRESHOLD &&
        param.rankSize == 12 && param.root < FIRST_SERVER_SIZE;
    const bool useScatterAllgather = totalBytes >= BROADCAST_LARGE_MESSAGE_THRESHOLD &&
        param.rankSize == 12;
    if (useSmallDualDieDirect) {
        const size_t expectedKernelCount = param.myRank == param.root ? 2 : 1;
        const size_t expectedWorkerCount =
            (SMALL_DUAL_DIE_PARALLEL && param.myRank == param.root) ? 1 : 0;
        if (resource.ccuKernels.size() != expectedKernelCount ||
            resource.threads.size() != expectedWorkerCount) {
            HCCL_ERROR("Small direct resource mismatch: rank=%u kernels=%zu expected=%zu workers=%zu expectedWorkers=%zu",
                param.myRank, resource.ccuKernels.size(), expectedKernelCount,
                resource.threads.size(), expectedWorkerCount);
            return HCCL_E_INTERNAL;
        }
    } else if (useDualStripePipeline) {
        if (resource.ccuKernels.size() != 2 || resource.threads.size() != 1) {
            HCCL_ERROR("Dual-stripe pipeline resource mismatch: kernels=%zu workers=%zu",
                resource.ccuKernels.size(), resource.threads.size());
            return HCCL_E_INTERNAL;
        }
    } else if (useStaticPipeline) {
        if (resource.ccuKernels.size() != 1 || !resource.threads.empty()) {
            HCCL_ERROR("Static pipeline resource mismatch: kernels=%zu workers=%zu",
                resource.ccuKernels.size(), resource.threads.size());
            return HCCL_E_INTERNAL;
        }
    } else if (useParallelMeshNhr) {
        if (resource.ccuKernels.size() != 2 || resource.threads.size() != 1) {
            HCCL_ERROR("Parallel Mesh-NHR resource mismatch: rank=%u kernels=%zu workers=%zu",
                param.myRank, resource.ccuKernels.size(), resource.threads.size());
            return HCCL_E_INTERNAL;
        }
    } else if (useRelay) {
        const size_t expectedKernelCount = param.myRank < FIRST_SERVER_SIZE ? 3 : 2;
        if (resource.ccuKernels.size() != expectedKernelCount || resource.threads.size() != 1) {
            HCCL_ERROR("Relay resource mismatch: rank=%u kernels=%zu expected=%zu workers=%zu",
                param.myRank, resource.ccuKernels.size(), expectedKernelCount,
                resource.threads.size());
            return HCCL_E_INTERNAL;
        }
    } else if (useBalancedFused) {
        if (resource.ccuKernels.size() != 3 ||
            resource.threads.size() != 1) {
            HCCL_ERROR("Balanced-fused resource mismatch: rank=%u kernels=%zu workers=%zu",
                param.myRank, resource.ccuKernels.size(),
                resource.threads.size());
            return HCCL_E_INTERNAL;
        }
    } else if (useHierarchical) {
        if (param.myRank < FIRST_SERVER_SIZE) {
            if (resource.ccuKernels.size() != 3 || resource.threads.size() != 1) {
                HCCL_ERROR("Hierarchical root server requires three kernels and one worker: kernels=%zu workers=%zu",
                    resource.ccuKernels.size(), resource.threads.size());
                return HCCL_E_INTERNAL;
            }
        } else if (resource.ccuKernels.size() != 1 || !resource.threads.empty()) {
            HCCL_ERROR("Hierarchical receiver requires one kernel and no worker: kernels=%zu workers=%zu",
                resource.ccuKernels.size(), resource.threads.size());
            return HCCL_E_INTERNAL;
        }
    } else if (resource.ccuKernels.size() != 1) {
        HCCL_ERROR("Broadcast requires exactly one same-die CCU kernel, kernels=%zu",
            resource.ccuKernels.size());
        return HCCL_E_INTERNAL;
    }

    if (useSmallDualDieDirect) {
        const uint64_t taskArgs[] = {baseAddress, memoryToken, totalBytes};
        CHK_RET(LaunchSmallDualDieDirect(param, resource, taskArgs));
        return HCCL_SUCCESS;
    }

    if (useDualStripePipeline) {
        const uint64_t firstStripeElements = param.count / 2;
        const uint64_t firstStripeBytes = firstStripeElements * sizeIter->second;
        CHK_RET(LaunchDualStripePipeline(
            param, resource, baseAddress, memoryToken, firstStripeBytes));
        return HCCL_SUCCESS;
    }

    if (useStaticPipeline) {
        // Chunk lengths and offsets are registration-time constants.  One
        // launch covers the complete 400 MiB+4 B or 512 MiB message.
        const uint64_t taskArgs[] = {baseAddress, memoryToken};
        CHK_RET_CCU(HcommCcuKernelLaunch(
            param.cpuThread, resource.ccuKernels[0], taskArgs, 2));
        return HCCL_SUCCESS;
    }

    uint32_t sliceCount = 0;
    if ((useScatterAllgather || useChain4) && !useParallelMeshNhr && !useRelay &&
        !useBalancedFused) {
        if (useHierarchical) {
            sliceCount = FIRST_SERVER_SIZE;
        } else if (useChain4) {
            // v33 merge: v27 splits each 256 MiB chunk into 16 chain segments.
            sliceCount = CHAIN4_PIPELINE_SEGMENT_COUNT;
        } else {
            sliceCount = param.root < 8 ? param.rankSize - 8 : 8;
        }
        if (sliceCount == 0) {
            HCCL_ERROR("Broadcast scatter-allgather has zero slices");
            return HCCL_E_INTERNAL;
        }
    }

    uint64_t offset = 0;
    if (useParallelMeshNhr) {
        while (offset < totalBytes) {
            const uint64_t transferBytes =
                std::min<uint64_t>(MAX_DATA_SIZE, totalBytes - offset);
            const uint64_t transferElements = transferBytes / sizeIter->second;
            const uint64_t firstPartElements = transferElements / 2;
            const uint64_t secondPartElements = transferElements - firstPartElements;
            const uint64_t firstPartBytes = firstPartElements * sizeIter->second;
            const uint64_t secondPartBytes = transferBytes - firstPartBytes;
            if (firstPartBytes == 0 || secondPartBytes == 0) {
                HCCL_ERROR("Invalid parallel Mesh-NHR split: transfer=%llu first=%llu second=%llu",
                    static_cast<unsigned long long>(transferBytes),
                    static_cast<unsigned long long>(firstPartBytes),
                    static_cast<unsigned long long>(secondPartBytes));
                return HCCL_E_INTERNAL;
            }

            const uint64_t firstNormalSliceBytes =
                (firstPartElements / FIRST_SERVER_SIZE) * sizeIter->second;
            const uint64_t firstLastSliceBytes = firstPartBytes -
                firstNormalSliceBytes * (FIRST_SERVER_SIZE - 1);
            const uint64_t secondNormalSliceBytes =
                (secondPartElements / FIRST_SERVER_SIZE) * sizeIter->second;
            const uint64_t secondLastSliceBytes = secondPartBytes -
                secondNormalSliceBytes * (FIRST_SERVER_SIZE - 1);
            const uint64_t intraFirstArgs[] = {
                baseAddress + offset, memoryToken,
                firstNormalSliceBytes, firstLastSliceBytes};
            const uint64_t interFirstArgs[] = {
                baseAddress + offset + firstPartBytes, memoryToken, secondPartBytes};
            const uint64_t intraSecondArgs[] = {
                baseAddress + offset + firstPartBytes, memoryToken,
                secondNormalSliceBytes, secondLastSliceBytes};
            const uint64_t interSecondArgs[] = {
                baseAddress + offset, memoryToken, firstPartBytes};
            CHK_RET(LaunchParallelMeshNhrChunk(param, resource,
                intraFirstArgs, interFirstArgs, intraSecondArgs, interSecondArgs));
            offset += transferBytes;
        }
        return HCCL_SUCCESS;
    }

    if (useRelay) {
        while (offset < totalBytes) {
            std::vector<RelayTaskArgs> relayWindow;
            relayWindow.reserve(RELAY_WINDOW_SIZE);
            for (size_t chunkIdx = 0;
                chunkIdx < RELAY_WINDOW_SIZE && offset < totalBytes; ++chunkIdx) {
                const uint64_t transferBytes =
                    std::min<uint64_t>(MAX_DATA_SIZE, totalBytes - offset);
                const uint64_t transferElements = transferBytes / sizeIter->second;
                const uint64_t normalSliceBytes =
                    (transferElements / FIRST_SERVER_SIZE) * sizeIter->second;
                const uint64_t lastSliceBytes = transferBytes -
                    normalSliceBytes * (FIRST_SERVER_SIZE - 1);
                relayWindow.push_back(
                    {baseAddress + offset, memoryToken, normalSliceBytes, lastSliceBytes});
                offset += transferBytes;
            }
            CHK_RET(LaunchRelayWindow(param, resource, relayWindow));
        }
        return HCCL_SUCCESS;
    }

    if (useBalancedFused) {
        std::vector<BalancedFusedTaskArgs> balancedWindow;
        balancedWindow.reserve(BALANCED_FUSED_WINDOW_SIZE);
        while (offset < totalBytes) {
            const uint64_t transferBytes =
                std::min<uint64_t>(MAX_DATA_SIZE, totalBytes - offset);
            const uint64_t transferElements =
                transferBytes / sizeIter->second;
            uint64_t meshElements =
                transferElements * BALANCED_MESH_NUMERATOR /
                BALANCED_SPLIT_DENOMINATOR;
            meshElements =
                meshElements / FIRST_SERVER_SIZE * FIRST_SERVER_SIZE;
            const uint64_t crossElements =
                transferElements - meshElements;
            if (meshElements < FIRST_SERVER_SIZE ||
                crossElements < FIRST_SERVER_SIZE) {
                HCCL_ERROR("Invalid balanced-fused split: transfer=%llu mesh=%llu cross=%llu",
                    static_cast<unsigned long long>(transferElements),
                    static_cast<unsigned long long>(meshElements),
                    static_cast<unsigned long long>(crossElements));
                return HCCL_E_INTERNAL;
            }

            const uint64_t meshBytes =
                meshElements * sizeIter->second;
            const uint64_t crossBytes =
                transferBytes - meshBytes;
            const uint64_t meshNormalSliceBytes =
                (meshElements / FIRST_SERVER_SIZE) * sizeIter->second;
            const uint64_t meshLastSliceBytes =
                meshBytes - meshNormalSliceBytes *
                (FIRST_SERVER_SIZE - 1);
            const uint64_t crossNormalSliceBytes =
                (crossElements / FIRST_SERVER_SIZE) * sizeIter->second;
            const uint64_t crossLastSliceBytes =
                crossBytes - crossNormalSliceBytes *
                (FIRST_SERVER_SIZE - 1);
            const uint64_t meshAddress = baseAddress + offset;
            const uint64_t crossAddress = meshAddress + meshBytes;

            BalancedFusedTaskArgs args{};
            args.intra = {meshAddress, memoryToken,
                meshNormalSliceBytes, meshLastSliceBytes};
            args.fused = {meshAddress, memoryToken,
                meshNormalSliceBytes, meshLastSliceBytes,
                crossAddress, crossNormalSliceBytes,
                crossLastSliceBytes};
            args.cross = {crossAddress, memoryToken,
                crossNormalSliceBytes, crossLastSliceBytes};
            balancedWindow.push_back(args);
            offset += transferBytes;
            if (balancedWindow.size() == BALANCED_FUSED_WINDOW_SIZE ||
                offset == totalBytes) {
                CHK_RET(LaunchBalancedFusedWindow(
                    param, resource, balancedWindow));
                balancedWindow.clear();
            }
        }
        return HCCL_SUCCESS;
    }

    if (useHierarchical) {
        while (offset < totalBytes) {
            std::vector<HierarchicalTaskArgs> hierarchicalWindow;
            hierarchicalWindow.reserve(HIERARCHICAL_WINDOW_SIZE);
            for (size_t chunkIdx = 0;
                chunkIdx < HIERARCHICAL_WINDOW_SIZE && offset < totalBytes; ++chunkIdx) {
                const uint64_t transferBytes =
                    std::min<uint64_t>(MAX_DATA_SIZE, totalBytes - offset);
                const uint64_t transferElements = transferBytes / sizeIter->second;
                const uint64_t normalSliceBytes =
                    (transferElements / sliceCount) * sizeIter->second;
                const uint64_t lastSliceBytes =
                    transferBytes - normalSliceBytes * (sliceCount - 1);
                hierarchicalWindow.push_back(
                    {baseAddress + offset, memoryToken, normalSliceBytes, lastSliceBytes});
                offset += transferBytes;
            }
            CHK_RET(LaunchHierarchicalWindow(param, resource, hierarchicalWindow));
        }
        return HCCL_SUCCESS;
    }

    while (offset < totalBytes) {
        const uint64_t transferBytes = std::min<uint64_t>(MAX_DATA_SIZE, totalBytes - offset);
        if (useScatterAllgather || useChain4) {
            const uint64_t transferElements = transferBytes / sizeIter->second;
            const uint64_t normalSliceBytes =
                (transferElements / sliceCount) * sizeIter->second;
            const uint64_t lastSliceBytes =
                transferBytes - normalSliceBytes * (sliceCount - 1);
            const uint64_t taskArgs[] = {
                baseAddress + offset, memoryToken, normalSliceBytes, lastSliceBytes};
            CHK_RET_CCU(HcommCcuKernelLaunch(
                param.cpuThread, resource.ccuKernels[0], taskArgs, 4));
        } else {
            const uint64_t taskArgs[] = {baseAddress + offset, memoryToken, transferBytes};
            CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread, resource.ccuKernels[0], taskArgs, 3));
        }
        offset += transferBytes;
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
