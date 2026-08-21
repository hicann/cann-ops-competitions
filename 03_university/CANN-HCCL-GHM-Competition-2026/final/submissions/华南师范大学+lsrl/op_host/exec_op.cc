/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <vector>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace {
constexpr uint64_t SLICE_ALIGNMENT = 4 * 1024;
constexpr uint32_t REDUCE_SCATTER_KERNEL_INDEX = 0;
constexpr uint32_t REDUCE_SCATTER_TASK_ARG_NUM = 12;
constexpr uint32_t DUAL_MAIN_STAGE_KERNEL_INDEX = 0;
constexpr uint32_t DUAL_WORKER_STAGE_KERNEL_INDEX = 1;
constexpr uint32_t DUAL_MAIN_REDUCE_KERNEL_INDEX = 2;
constexpr uint32_t DUAL_TASK_ARG_NUM = 8;
constexpr uint32_t DUAL_STREAM_NOTIFY_INDEX = 0;
constexpr uint32_t PARALLEL_2X8_MESH_A_STAGE1_KERNEL_INDEX = 0;
constexpr uint32_t PARALLEL_2X8_NHR_B_STAGE1_KERNEL_INDEX = 1;
constexpr uint32_t PARALLEL_2X8_MESH_B_STAGE2_KERNEL_INDEX = 2;
constexpr uint32_t PARALLEL_2X8_NHR_A_STAGE2_KERNEL_INDEX = 3;
constexpr uint32_t PARALLEL_2X8_TASK_ARG_NUM = 12;
constexpr uint32_t PARALLEL_2X8_LOCAL_RANK_NUM = 8;
constexpr uint32_t PARALLEL_2X8_NOTIFY_INDEX = 0;
constexpr uint32_t ASYMMETRIC_MESH_A_STAGE1_KERNEL_INDEX = 0;
constexpr uint32_t ASYMMETRIC_NHR_B_STAGE1_KERNEL_INDEX = 1;
constexpr uint32_t ASYMMETRIC_MESH_B_STAGE2_KERNEL_INDEX = 2;
constexpr uint32_t ASYMMETRIC_NHR_A_STAGE2_KERNEL_INDEX = 3;
constexpr uint32_t ASYMMETRIC_TASK_ARG_NUM = 14;
constexpr uint32_t ASYMMETRIC_RANK_NUM = 12;
constexpr uint32_t ASYMMETRIC_NOTIFY_INDEX = 0;
constexpr uint32_t LATIN_4X1_KERNEL_INDEX = 0;
constexpr uint32_t LATIN_4X1_TASK_ARG_NUM = 8;
constexpr uint32_t LATIN_4X1_PEER_NUM = 3;

uint64_t AlignDown(uint64_t value, uint64_t alignment)
{
    return value / alignment * alignment;
}

HcclResult LaunchSlices(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t inputToken,
    uint64_t outputToken, uint64_t scratchToken, uint64_t recvBytes, uint64_t chunkOffset,
    uint64_t sliceBytes, uint64_t repeatCount)
{
    if (sliceBytes == 0 || repeatCount == 0) {
        return HCCL_SUCCESS;
    }

    const uint64_t inputAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outputAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    const uint64_t scratchAddr = reinterpret_cast<uint64_t>(resCtx.localBuffer.addr);
    const uint64_t currentInputOffset = static_cast<uint64_t>(param.myRank) * recvBytes + chunkOffset;
    const uint64_t repeatCounter = std::numeric_limits<uint64_t>::max() - repeatCount;

    const std::array<uint64_t, REDUCE_SCATTER_TASK_ARG_NUM> taskArgs = {
        inputAddr,
        outputAddr,
        inputToken,
        outputToken,
        scratchAddr,
        scratchToken,
        currentInputOffset,
        chunkOffset,
        sliceBytes,
        sliceBytes,
        sliceBytes,
        repeatCounter,
    };
    if (taskArgs.size() != REDUCE_SCATTER_TASK_ARG_NUM) {
        HCCL_ERROR("Unexpected CCU task argument number");
        return HCCL_E_INTERNAL;
    }

    CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread,
        resCtx.ccuKernels[REDUCE_SCATTER_KERNEL_INDEX], taskArgs.data(), taskArgs.size()));
    return HCCL_SUCCESS;
}

HcclResult LaunchDualSlice(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t inputToken, uint64_t outputToken, uint64_t scratchToken, uint64_t recvBytes,
    uint64_t chunkOffset, uint64_t sliceBytes)
{
    if (sliceBytes == 0) {
        return HCCL_SUCCESS;
    }

    const uint64_t inputAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outputAddr =
        reinterpret_cast<uint64_t>(param.outputPtr) + chunkOffset;
    const uint64_t scratchAddr = reinterpret_cast<uint64_t>(resCtx.localBuffer.addr);
    const uint64_t currentInputOffset =
        static_cast<uint64_t>(param.myRank) * recvBytes + chunkOffset;

    const std::array<uint64_t, DUAL_TASK_ARG_NUM> taskArgs = {
        inputAddr,
        outputAddr,
        inputToken,
        outputToken,
        scratchAddr,
        scratchToken,
        currentInputOffset,
        sliceBytes,
    };

    // CheckerV3 requires every slave stream to start with a local WAIT and end
    // with a local RECORD. The main stream releases the worker, both IO Dies
    // stage their disjoint rank slots concurrently, and the main stream joins
    // the worker before launching the deterministic local reduction.
    // Emit the main-stream RECORD first so task metadata assigns the user
    // stream as stream 0. The worker then appears as a slave stream whose
    // first and last tasks are WAIT and RECORD respectively.
    CHK_RET(HcommThreadNotifyRecordOnThread(param.cpuThread,
        resCtx.workerThread, DUAL_STREAM_NOTIFY_INDEX));
    CHK_RET(HcommThreadNotifyWaitOnThread(resCtx.workerThread,
        DUAL_STREAM_NOTIFY_INDEX, CUSTOM_TIMEOUT));
    CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.workerThread,
        resCtx.ccuKernels[DUAL_WORKER_STAGE_KERNEL_INDEX], taskArgs.data(), DUAL_TASK_ARG_NUM));
    CHK_RET(HcommThreadNotifyRecordOnThread(resCtx.workerThread,
        param.cpuThread, DUAL_STREAM_NOTIFY_INDEX));
    CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread,
        resCtx.ccuKernels[DUAL_MAIN_STAGE_KERNEL_INDEX], taskArgs.data(), DUAL_TASK_ARG_NUM));
    CHK_RET(HcommThreadNotifyWaitOnThread(param.cpuThread,
        DUAL_STREAM_NOTIFY_INDEX, CUSTOM_TIMEOUT));
    CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread,
        resCtx.ccuKernels[DUAL_MAIN_REDUCE_KERNEL_INDEX], taskArgs.data(), DUAL_TASK_ARG_NUM));
    return HCCL_SUCCESS;
}

HcclResult LaunchParallel2x8(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t inputToken, uint64_t outputToken, uint64_t scratchToken, uint64_t recvBytes)
{
    const uint64_t partA = AlignDown(recvBytes / 2, SLICE_ALIGNMENT);
    const uint64_t partB = recvBytes - partA;
    const uint64_t partALane =
        AlignDown(partA / (PARALLEL_2X8_LOCAL_RANK_NUM - 1), SLICE_ALIGNMENT);
    const uint64_t partBLane =
        AlignDown(partB / (PARALLEL_2X8_LOCAL_RANK_NUM - 1), SLICE_ALIGNMENT);
    if (partA == 0 || partB == 0 || partALane == 0 || partBLane == 0) {
        HCCL_ERROR("Invalid parallel 2x8 split, C[%llu], A[%llu], B[%llu]",
            static_cast<unsigned long long>(recvBytes),
            static_cast<unsigned long long>(partA),
            static_cast<unsigned long long>(partB));
        return HCCL_E_INTERNAL;
    }

    const uint64_t partALast =
        partA - (PARALLEL_2X8_LOCAL_RANK_NUM - 2) * partALane;
    const uint64_t partBLast =
        partB - (PARALLEL_2X8_LOCAL_RANK_NUM - 2) * partBLane;
    const std::array<uint64_t, PARALLEL_2X8_TASK_ARG_NUM> taskArgs = {
        reinterpret_cast<uint64_t>(param.inputPtr),
        reinterpret_cast<uint64_t>(param.outputPtr),
        inputToken,
        outputToken,
        reinterpret_cast<uint64_t>(resCtx.localBuffer.addr),
        scratchToken,
        recvBytes,
        partA,
        partALane,
        partALast,
        partBLane,
        partBLast,
    };

    // Keep the user stream as stream 0. The slave stream starts with a local
    // WAIT, ends with a local RECORD, and the crossed dependencies allow the
    // second Mesh/NHR stages to start as soon as their own first stage is ready.
    CHK_RET(HcommThreadNotifyRecordOnThread(param.cpuThread,
        resCtx.workerThread, PARALLEL_2X8_NOTIFY_INDEX));
    CHK_RET(HcommThreadNotifyWaitOnThread(resCtx.workerThread,
        PARALLEL_2X8_NOTIFY_INDEX, CUSTOM_TIMEOUT));
    CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.workerThread,
        resCtx.ccuKernels[PARALLEL_2X8_NHR_B_STAGE1_KERNEL_INDEX],
        taskArgs.data(), taskArgs.size()));
    CHK_RET(HcommThreadNotifyRecordOnThread(resCtx.workerThread,
        param.cpuThread, PARALLEL_2X8_NOTIFY_INDEX));
    CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread,
        resCtx.ccuKernels[PARALLEL_2X8_MESH_A_STAGE1_KERNEL_INDEX],
        taskArgs.data(), taskArgs.size()));
    CHK_RET(HcommThreadNotifyRecordOnThread(param.cpuThread,
        resCtx.workerThread, PARALLEL_2X8_NOTIFY_INDEX));
    CHK_RET(HcommThreadNotifyWaitOnThread(resCtx.workerThread,
        PARALLEL_2X8_NOTIFY_INDEX, CUSTOM_TIMEOUT));
    CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.workerThread,
        resCtx.ccuKernels[PARALLEL_2X8_NHR_A_STAGE2_KERNEL_INDEX],
        taskArgs.data(), taskArgs.size()));
    CHK_RET(HcommThreadNotifyRecordOnThread(resCtx.workerThread,
        param.cpuThread, PARALLEL_2X8_NOTIFY_INDEX));
    CHK_RET(HcommThreadNotifyWaitOnThread(param.cpuThread,
        PARALLEL_2X8_NOTIFY_INDEX, CUSTOM_TIMEOUT));
    CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread,
        resCtx.ccuKernels[PARALLEL_2X8_MESH_B_STAGE2_KERNEL_INDEX],
        taskArgs.data(), taskArgs.size()));
    CHK_RET(HcommThreadNotifyWaitOnThread(param.cpuThread,
        PARALLEL_2X8_NOTIFY_INDEX, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult LaunchLatin4x1(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t inputToken, uint64_t outputToken, uint64_t recvBytes)
{
    const uint64_t laneBytes =
        AlignDown(recvBytes / LATIN_4X1_PEER_NUM, SLICE_ALIGNMENT);
    const uint64_t lastLaneBytes =
        recvBytes - (LATIN_4X1_PEER_NUM - 1) * laneBytes;
    if (laneBytes == 0 || lastLaneBytes == 0) {
        HCCL_ERROR("Invalid 4x1 Latin split, C[%llu], lane[%llu], last[%llu]",
            static_cast<unsigned long long>(recvBytes),
            static_cast<unsigned long long>(laneBytes),
            static_cast<unsigned long long>(lastLaneBytes));
        return HCCL_E_INTERNAL;
    }

    const uint64_t currentInputOffset =
        static_cast<uint64_t>(param.myRank) * recvBytes;
    const std::array<uint64_t, LATIN_4X1_TASK_ARG_NUM> taskArgs = {
        reinterpret_cast<uint64_t>(param.inputPtr),
        reinterpret_cast<uint64_t>(param.outputPtr),
        inputToken,
        outputToken,
        currentInputOffset,
        recvBytes,
        laneBytes,
        lastLaneBytes,
    };
    CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread,
        resCtx.ccuKernels[LATIN_4X1_KERNEL_INDEX],
        taskArgs.data(), taskArgs.size()));
    return HCCL_SUCCESS;
}

HcclResult LaunchAsymmetric8x4(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t inputToken, uint64_t outputToken, uint64_t scratchToken, uint64_t recvBytes)
{
    if (param.rankSize != ASYMMETRIC_RANK_NUM ||
        resCtx.localPeerCount == 0 || resCtx.crossPeerCount == 0) {
        return HCCL_E_PARA;
    }

    // The four-rank side creates three Mesh-A partials.  Its eight logical
    // cross peers share a rank-side Clos bandwidth of four Mesh links, so the
    // aggregate NHR-B load is 2*B.  Balancing 3*A and 2*B gives A:B = 2:3.
    const uint64_t partA = AlignDown(recvBytes * 2U / 5U, SLICE_ALIGNMENT);
    const uint64_t partB = recvBytes - partA;
    const uint64_t aLane =
        AlignDown(partA / resCtx.localPeerCount, SLICE_ALIGNMENT);
    const uint64_t bCrossLane =
        AlignDown(partB / resCtx.crossPeerCount, SLICE_ALIGNMENT);
    const uint64_t bMeshLane =
        AlignDown(partB / resCtx.localPeerCount, SLICE_ALIGNMENT);
    if (partA == 0 || partB == 0 || aLane == 0 ||
        bCrossLane == 0 || bMeshLane == 0) {
        HCCL_ERROR("Invalid asymmetric 8x4 split, C[%llu], A[%llu], B[%llu], "
            "local peers[%u], cross peers[%u]",
            static_cast<unsigned long long>(recvBytes),
            static_cast<unsigned long long>(partA),
            static_cast<unsigned long long>(partB),
            resCtx.localPeerCount, resCtx.crossPeerCount);
        return HCCL_E_INTERNAL;
    }

    const uint64_t aLast =
        partA - (resCtx.localPeerCount - 1U) * aLane;
    const uint64_t bCrossLast =
        partB - (resCtx.crossPeerCount - 1U) * bCrossLane;
    const uint64_t bMeshLast =
        partB - (resCtx.localPeerCount - 1U) * bMeshLane;
    const std::array<uint64_t, ASYMMETRIC_TASK_ARG_NUM> taskArgs = {
        reinterpret_cast<uint64_t>(param.inputPtr),
        reinterpret_cast<uint64_t>(param.outputPtr),
        inputToken,
        outputToken,
        reinterpret_cast<uint64_t>(resCtx.localBuffer.addr),
        scratchToken,
        recvBytes,
        partA,
        aLane,
        aLast,
        bCrossLane,
        bCrossLast,
        bMeshLane,
        bMeshLast,
    };

    // The worker stream starts with WAIT and ends with RECORD for CheckerV3.
    // Crossed dependencies let A's cross stage and B's Mesh stage consume
    // their own completed phase-1 partials without a global four-kernel fence.
    CHK_RET(HcommThreadNotifyRecordOnThread(param.cpuThread,
        resCtx.workerThread, ASYMMETRIC_NOTIFY_INDEX));
    CHK_RET(HcommThreadNotifyWaitOnThread(resCtx.workerThread,
        ASYMMETRIC_NOTIFY_INDEX, CUSTOM_TIMEOUT));
    CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.workerThread,
        resCtx.ccuKernels[ASYMMETRIC_NHR_B_STAGE1_KERNEL_INDEX],
        taskArgs.data(), taskArgs.size()));
    CHK_RET(HcommThreadNotifyRecordOnThread(resCtx.workerThread,
        param.cpuThread, ASYMMETRIC_NOTIFY_INDEX));
    CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread,
        resCtx.ccuKernels[ASYMMETRIC_MESH_A_STAGE1_KERNEL_INDEX],
        taskArgs.data(), taskArgs.size()));
    CHK_RET(HcommThreadNotifyRecordOnThread(param.cpuThread,
        resCtx.workerThread, ASYMMETRIC_NOTIFY_INDEX));
    CHK_RET(HcommThreadNotifyWaitOnThread(resCtx.workerThread,
        ASYMMETRIC_NOTIFY_INDEX, CUSTOM_TIMEOUT));
    CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.workerThread,
        resCtx.ccuKernels[ASYMMETRIC_NHR_A_STAGE2_KERNEL_INDEX],
        taskArgs.data(), taskArgs.size()));
    CHK_RET(HcommThreadNotifyRecordOnThread(resCtx.workerThread,
        param.cpuThread, ASYMMETRIC_NOTIFY_INDEX));
    CHK_RET(HcommThreadNotifyWaitOnThread(param.cpuThread,
        ASYMMETRIC_NOTIFY_INDEX, CUSTOM_TIMEOUT));
    CHK_RET_CCU(HcommCcuKernelLaunch(param.cpuThread,
        resCtx.ccuKernels[ASYMMETRIC_MESH_B_STAGE2_KERNEL_INDEX],
        taskArgs.data(), taskArgs.size()));
    CHK_RET(HcommThreadNotifyWaitOnThread(param.cpuThread,
        ASYMMETRIC_NOTIFY_INDEX, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}
} // namespace

namespace ops_hccl {
HcclResult ExecOp(const OpParam &param)
{
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    if (param.dataType != HCCL_DATA_TYPE_FP32 || param.reduceType != HCCL_REDUCE_SUM ||
        param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || param.myRank >= param.rankSize) {
        HCCL_ERROR("Invalid ReduceScatter parameters");
        return HCCL_E_PARA;
    }
    if (param.count > std::numeric_limits<uint64_t>::max() / sizeof(float)) {
        HCCL_ERROR("ReduceScatter receive byte size overflows uint64");
        return HCCL_E_PARA;
    }

    const uint64_t recvBytes = param.count * sizeof(float);
    if (param.rankSize > std::numeric_limits<uint64_t>::max() / recvBytes) {
        HCCL_ERROR("ReduceScatter input byte size overflows uint64");
        return HCCL_E_PARA;
    }
    const uint64_t inputBytes = recvBytes * param.rankSize;

    if (param.rankSize == 1) {
        CHK_RET(HcommLocalCopyOnThread(param.cpuThread, param.outputPtr, param.inputPtr, recvBytes));
        return HCCL_SUCCESS;
    }

    CHK_PTR_NULL(param.resCtx);
    if (param.ctxSize == 0) {
        HCCL_ERROR("ReduceScatter resource context is empty");
        return HCCL_E_INTERNAL;
    }
    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> serializedCtx(ctx, ctx + param.ctxSize);
    AlgResourceCtx resCtx{};
    resCtx.DeSerialize(serializedCtx);

    if (resCtx.localBuffer.addr == nullptr || resCtx.localBuffer.size == 0) {
        HCCL_ERROR("ReduceScatter CCU resources are incomplete");
        return HCCL_E_INTERNAL;
    }
    if (resCtx.algorithm == ReduceScatterAlgorithm::SINGLE_LAYER &&
        resCtx.ccuKernels.size() <= REDUCE_SCATTER_KERNEL_INDEX) {
        HCCL_ERROR("Single-layer ReduceScatter kernel is unavailable");
        return HCCL_E_INTERNAL;
    }
    if (resCtx.algorithm == ReduceScatterAlgorithm::DUAL_LAYER &&
        (resCtx.ccuKernels.size() <= DUAL_MAIN_REDUCE_KERNEL_INDEX ||
            resCtx.workerThread == 0)) {
        HCCL_ERROR("Dual-layer ReduceScatter resources are incomplete");
        return HCCL_E_INTERNAL;
    }
    if (resCtx.algorithm == ReduceScatterAlgorithm::PARALLEL_2X8 &&
        (resCtx.ccuKernels.size() <= PARALLEL_2X8_NHR_A_STAGE2_KERNEL_INDEX ||
            resCtx.workerThread == 0)) {
        HCCL_ERROR("Parallel 2x8 ReduceScatter resources are incomplete");
        return HCCL_E_INTERNAL;
    }
    if (resCtx.algorithm == ReduceScatterAlgorithm::ASYMMETRIC_8X4 &&
        (resCtx.ccuKernels.size() <= ASYMMETRIC_NHR_A_STAGE2_KERNEL_INDEX ||
            resCtx.workerThread == 0 || resCtx.localPeerCount == 0 ||
            resCtx.crossPeerCount == 0)) {
        HCCL_ERROR("Asymmetric 8x4 ReduceScatter resources are incomplete");
        return HCCL_E_INTERNAL;
    }
    if (resCtx.algorithm == ReduceScatterAlgorithm::LATIN_4X1 &&
        resCtx.ccuKernels.size() <= LATIN_4X1_KERNEL_INDEX) {
        HCCL_ERROR("4x1 Latin ReduceScatter kernel is unavailable");
        return HCCL_E_INTERNAL;
    }

    uint64_t sliceBytes = std::min<uint64_t>(
        MAX_DATA_SIZE, resCtx.localBuffer.size / static_cast<uint64_t>(param.rankSize));
    sliceBytes = AlignDown(sliceBytes, SLICE_ALIGNMENT);
    if (sliceBytes == 0) {
        sliceBytes = AlignDown(
            resCtx.localBuffer.size / static_cast<uint64_t>(param.rankSize), sizeof(float));
    }
    if (sliceBytes == 0) {
        HCCL_ERROR("HCCL buffer is too small for ReduceScatter scratch slots");
        return HCCL_E_INTERNAL;
    }

    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    uint64_t scratchToken = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(
        reinterpret_cast<uint64_t>(param.inputPtr), inputBytes, &inputToken));
    CHK_RET_CCU(HcommCcuGetMemToken(
        reinterpret_cast<uint64_t>(param.outputPtr), recvBytes, &outputToken));
    CHK_RET_CCU(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(resCtx.localBuffer.addr),
        resCtx.localBuffer.size, &scratchToken));

    if (resCtx.algorithm == ReduceScatterAlgorithm::PARALLEL_2X8) {
        return LaunchParallel2x8(
            param, resCtx, inputToken, outputToken, scratchToken, recvBytes);
    }
    if (resCtx.algorithm == ReduceScatterAlgorithm::ASYMMETRIC_8X4) {
        return LaunchAsymmetric8x4(
            param, resCtx, inputToken, outputToken, scratchToken, recvBytes);
    }
    if (resCtx.algorithm == ReduceScatterAlgorithm::LATIN_4X1) {
        return LaunchLatin4x1(
            param, resCtx, inputToken, outputToken, recvBytes);
    }
    if (resCtx.algorithm == ReduceScatterAlgorithm::DUAL_LAYER) {
        uint64_t chunkOffset = 0;
        while (chunkOffset < recvBytes) {
            const uint64_t currentSliceBytes =
                std::min<uint64_t>(sliceBytes, recvBytes - chunkOffset);
            CHK_RET(LaunchDualSlice(param, resCtx, inputToken, outputToken, scratchToken,
                recvBytes, chunkOffset, currentSliceBytes));
            chunkOffset += currentSliceBytes;
        }
        return HCCL_SUCCESS;
    }
    if (resCtx.algorithm != ReduceScatterAlgorithm::SINGLE_LAYER) {
        HCCL_ERROR("Unknown ReduceScatter algorithm mode");
        return HCCL_E_INTERNAL;
    }

    const uint64_t fullSliceNum = recvBytes / sliceBytes;
    const uint64_t tailBytes = recvBytes % sliceBytes;
    CHK_RET(LaunchSlices(param, resCtx, inputToken, outputToken, scratchToken,
        recvBytes, 0, sliceBytes, fullSliceNum));
    if (tailBytes != 0) {
        CHK_RET(LaunchSlices(param, resCtx, inputToken, outputToken, scratchToken,
            recvBytes, fullSliceNum * sliceBytes, tailBytes, 1));
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
