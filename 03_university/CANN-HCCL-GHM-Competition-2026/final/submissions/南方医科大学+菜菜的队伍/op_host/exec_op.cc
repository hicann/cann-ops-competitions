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
#include <acl/acl_rt.h>

#include "custom.h"
#include "exec_op.h"

namespace ops_hccl {
HcclResult DispatchReduceScatter(const OpParam &param, aclrtStream)
{
    // 反序列化
    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> seq(ctx, ctx + param.ctxSize);
    DispatchContext executionState;
    executionState.DeSerialize(seq);

    const WorkloadCell expectedCell = ClassifyWorkload(param.rankSize, param.count);
    if (expectedCell == WorkloadCell::INVALID || executionState.magic != kReduceScatterSignature ||
        executionState.cell != expectedCell || executionState.version != WorkloadEpoch(expectedCell)) {
        HCCL_ERROR("[DispatchReduceScatter] resource context does not match worldSize=%u recvCount=%lu", param.rankSize, param.count);
        return HCCL_E_INTERNAL;
    }

    if (executionState.groupCuts.size() < 2) {
        HCCL_ERROR("[DispatchReduceScatter] missing Direct-Owner group metadata");
        return HCCL_E_INTERNAL;
    }
    const uint32_t groupCount = static_cast<uint32_t>(executionState.groupCuts.size() - 1);
    const bool p4Output = executionState.algorithm == ScheduleKind::P4_OUTPUT_ACCUMULATE &&
        (expectedCell == WorkloadCell::P4_512M || expectedCell == WorkloadCell::P4_400M4B);
    const bool directMeshSmall = executionState.algorithm == ScheduleKind::DIRECT_MESH_SMALL &&
        (expectedCell == WorkloadCell::P4_512K || expectedCell == WorkloadCell::P16_512K) &&
        groupCount == (param.rankSize == 4U ? 1U : 2U) &&
        executionState.workers.size() == groupCount && executionState.programs.size() == groupCount &&
        (groupCount == 1U || executionState.stagingBase != 0U);
    const bool capacityAwarePull = executionState.algorithm == ScheduleKind::CAPACITY_AWARE_PULL &&
        expectedCell == WorkloadCell::P12_512K &&
        groupCount == 2U && executionState.workers.size() == 2U && executionState.programs.size() == 3U &&
        executionState.stagingBase != 0U;
    const bool singleGroup = p4Output && groupCount == 1 &&
        executionState.workers.size() == 1 && executionState.programs.size() == 1;
    const bool slotGather = executionState.algorithm == ScheduleKind::SLOT_GATHER && groupCount == 2 &&
        executionState.workers.size() == 2 && executionState.programs.size() == 3 && executionState.stagingBase != 0;
    const bool dualGroup = slotGather;
    if (executionState.groupCuts.front() != 0 || executionState.groupCuts.back() != executionState.groupRanks.size() ||
        (!singleGroup && !dualGroup && !directMeshSmall && !capacityAwarePull) ||
        param.dataType != HCCL_DATA_TYPE_FP32 || param.reduceType != HCCL_REDUCE_SUM) {
        HCCL_ERROR("[DispatchReduceScatter] invalid Direct-Owner resource context");
        return HCCL_E_INTERNAL;
    }

    constexpr uint64_t kDataTypeBytes = sizeof(float);
    const uint64_t resultBytes = param.count * kDataTypeBytes;
    if (param.count == 0 || resultBytes / kDataTypeBytes != param.count ||
        resultBytes > UINT64_MAX / param.rankSize) {
        HCCL_ERROR("[DispatchReduceScatter] invalid recvCount");
        return HCCL_E_PARA;
    }
    const uint64_t inputAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t ownerInputAddr = inputAddr + param.myRank * resultBytes;
    const uint64_t inputBytes = resultBytes * param.rankSize;
    const uint64_t outputAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    uint64_t scratchToken = 0;
    if (HcommCcuGetMemToken(inputAddr, inputBytes, &inputToken) != CCU_SUCCESS ||
        HcommCcuGetMemToken(outputAddr, resultBytes, &outputToken) != CCU_SUCCESS ||
        ((dualGroup || (directMeshSmall && groupCount == 2U) || capacityAwarePull) &&
            HcommCcuGetMemToken(executionState.stagingBase, executionState.stagingBytes, &scratchToken) != CCU_SUCCESS)) {
        HCCL_ERROR("[DispatchReduceScatter] failed to acquire CCU memory token");
        return HCCL_E_INTERNAL;
    }

    if (capacityAwarePull) {
        constexpr uint64_t kScratchAlignment = 512U;
        const uint64_t remoteCount = param.rankSize - 1U;
        const uint64_t alignedOwner = (resultBytes + kScratchAlignment - 1U) & ~(kScratchAlignment - 1U);
        const uint64_t slotTotal = std::min<uint64_t>(remoteCount, executionState.stagingBytes / alignedOwner);
        const uint64_t spillCount = remoteCount - slotTotal;
        if (alignedOwner < resultBytes || slotTotal == 0U || slotTotal > UINT64_MAX / alignedOwner ||
            executionState.stagingBytes < slotTotal * alignedOwner ||
            executionState.groupRanks.size() != remoteCount || spillCount > 3U) {
            HCCL_ERROR("[DispatchReduceScatter] invalid capacity-aware scratch layout");
            return HCCL_E_PARA;
        }
        if (HcommLocalCopyOnThread(executionState.workers[0], reinterpret_cast<void *>(outputAddr),
                reinterpret_cast<const void *>(ownerInputAddr), resultBytes) != 0) {
            HCCL_ERROR("[DispatchReduceScatter] failed to initialize capacity-aware output");
            return HCCL_E_INTERNAL;
        }
        std::vector<uint64_t> slotArgs = {
            inputAddr, executionState.stagingBase, outputAddr, inputToken, scratchToken, outputToken};
        auto launchDiePair = [&](CcuKernelHandle primaryKernel, CcuKernelHandle secondaryKernel,
                                 std::vector<uint64_t> &args) -> HcclResult {
            if (HcommThreadNotifyRecordOnThread(executionState.workers[0], executionState.workers[1], 0U) != 0 ||
                HcommThreadNotifyWaitOnThread(executionState.workers[1], 0U, 0U) != 0 ||
                HcommCcuKernelLaunch(executionState.workers[1], secondaryKernel,
                    args.data(), args.size()) != CCU_SUCCESS ||
                HcommThreadNotifyRecordOnThread(executionState.workers[1], executionState.workers[0], 0U) != 0 ||
                HcommCcuKernelLaunch(executionState.workers[0], primaryKernel,
                    args.data(), args.size()) != CCU_SUCCESS ||
                HcommThreadNotifyWaitOnThread(executionState.workers[0], 0U, 0U) != 0) {
                return HCCL_E_INTERNAL;
            }
            return HCCL_SUCCESS;
        };
        if (launchDiePair(executionState.programs[0], executionState.programs[1], slotArgs) != HCCL_SUCCESS) {
            HCCL_ERROR("[DispatchReduceScatter] failed to schedule capacity-aware pulls");
            return HCCL_E_INTERNAL;
        }
        std::vector<uint64_t> foldArgs = {outputAddr, executionState.stagingBase, outputToken, scratchToken};
        if (HcommCcuKernelLaunch(executionState.workers[0], executionState.programs[2],
                foldArgs.data(), foldArgs.size()) != CCU_SUCCESS) {
            HCCL_ERROR("[DispatchReduceScatter] failed to launch capacity fold");
            return HCCL_E_INTERNAL;
        }
        return HCCL_SUCCESS;
    }

    if (directMeshSmall) {
        if (groupCount == 2U && executionState.stagingBytes < resultBytes) {
            HCCL_ERROR("[DispatchReduceScatter] CCU owner-pull scratch is too small");
            return HCCL_E_PARA;
        }
        std::vector<uint64_t> taskArgs = {inputAddr, outputAddr, executionState.stagingBase, inputToken,
            outputToken, scratchToken, static_cast<uint64_t>(MeshStage::PARTIAL)};
        if (groupCount == 1U) {
            if (HcommCcuKernelLaunch(executionState.workers[0], executionState.programs[0],
                    taskArgs.data(), taskArgs.size()) != CCU_SUCCESS) {
                HCCL_ERROR("[DispatchReduceScatter] failed to launch single-Die direct-mesh kernel");
                return HCCL_E_INTERNAL;
            }
            return HCCL_SUCCESS;
        }
        if (HcommThreadNotifyRecordOnThread(executionState.workers[0], executionState.workers[1], 0U) != 0 ||
            HcommThreadNotifyWaitOnThread(executionState.workers[1], 0U, 0U) != 0 ||
            HcommCcuKernelLaunch(executionState.workers[1], executionState.programs[1],
                taskArgs.data(), taskArgs.size()) != CCU_SUCCESS ||
            HcommThreadNotifyRecordOnThread(executionState.workers[1], executionState.workers[0], 0U) != 0 ||
            HcommCcuKernelLaunch(executionState.workers[0], executionState.programs[0],
                taskArgs.data(), taskArgs.size()) != CCU_SUCCESS ||
            HcommThreadNotifyWaitOnThread(executionState.workers[0], 0U, 0U) != 0) {
            HCCL_ERROR("[DispatchReduceScatter] failed to schedule dual-Die owner-pull partials");
            return HCCL_E_INTERNAL;
        }
        taskArgs.back() = static_cast<uint64_t>(MeshStage::MERGE);
        if (HcommCcuKernelLaunch(executionState.workers[0], executionState.programs[0],
                taskArgs.data(), taskArgs.size()) != CCU_SUCCESS) {
            HCCL_ERROR("[DispatchReduceScatter] failed to merge dual-Die owner-pull partials");
            return HCCL_E_INTERNAL;
        }
        return HCCL_SUCCESS;
    }

    const uint64_t remoteCount = param.rankSize - 1;
    uint64_t chunkCapacity = MAX_DATA_SIZE;
    if (slotGather) {
        chunkCapacity = std::min<uint64_t>(MAX_DATA_SIZE, executionState.stagingBytes / remoteCount);
        chunkCapacity -= chunkCapacity % kDataTypeBytes;
        if (chunkCapacity == 0 || chunkCapacity > executionState.stagingBytes / remoteCount) {
            HCCL_ERROR("[DispatchReduceScatter] HCCL scratch is too small for peer slots");
            return HCCL_E_PARA;
        }
    }

    for (uint64_t offset = 0; offset < resultBytes; offset += chunkCapacity) {
        const uint64_t chunkBytes = std::min<uint64_t>(chunkCapacity, resultBytes - offset);
        std::vector<std::vector<uint64_t>> groupTaskArgs(groupCount);
        for (uint32_t groupIdx = 0; groupIdx < groupCount; ++groupIdx) {
            auto &taskArgs = groupTaskArgs[groupIdx];
            const uint32_t begin = executionState.groupCuts[groupIdx];
            const uint32_t end = executionState.groupCuts[groupIdx + 1];
            taskArgs.reserve(dualGroup ? 2 * (end - begin) + 3 : end - begin + 4);
            for (uint32_t peerIdx = begin; peerIdx < end; ++peerIdx) {
                const uint32_t peerRank = executionState.groupRanks[peerIdx];
                taskArgs.push_back(inputAddr + peerRank * resultBytes + offset);
            }
            if (dualGroup) {
                for (uint32_t peerIdx = begin; peerIdx < end; ++peerIdx) {
                    const uint32_t sourceRank = executionState.groupRanks[peerIdx];
                    const uint32_t slotIdx = sourceRank < param.myRank ? sourceRank : sourceRank - 1;
                    taskArgs.push_back(executionState.stagingBase + slotIdx * chunkCapacity);
                }
                taskArgs.push_back(inputToken);
                taskArgs.push_back(scratchToken);
                taskArgs.push_back(chunkBytes);
                continue;
            }
            taskArgs.push_back(outputAddr + offset);
            taskArgs.push_back(inputToken);
            taskArgs.push_back(outputToken);
            taskArgs.push_back(chunkBytes);
        }

        if (HcommLocalCopyOnThread(executionState.workers[0], reinterpret_cast<void *>(outputAddr + offset),
                reinterpret_cast<const void *>(ownerInputAddr + offset), chunkBytes) != 0) {
            HCCL_ERROR("[DispatchReduceScatter] failed to execute a Direct-Owner chunk");
            return HCCL_E_INTERNAL;
        }
        if (singleGroup) {
            auto &taskArgs = groupTaskArgs[0];
            if (HcommCcuKernelLaunch(executionState.workers[0], executionState.programs[0], taskArgs.data(), taskArgs.size()) !=
                CCU_SUCCESS) {
                HCCL_ERROR("[DispatchReduceScatter] failed to launch the single-Die Direct-Owner kernel");
                return HCCL_E_INTERNAL;
            }
            continue;
        }

        if (HcommThreadNotifyRecordOnThread(executionState.workers[0], executionState.workers[1], 0) != 0 ||
            HcommThreadNotifyWaitOnThread(executionState.workers[1], 0, 0) != 0 ||
            HcommCcuKernelLaunch(executionState.workers[1], executionState.programs[1], groupTaskArgs[1].data(),
                groupTaskArgs[1].size()) != CCU_SUCCESS ||
            HcommThreadNotifyRecordOnThread(executionState.workers[1], executionState.workers[0], 0) != 0 ||
            HcommCcuKernelLaunch(executionState.workers[0], executionState.programs[0], groupTaskArgs[0].data(),
                groupTaskArgs[0].size()) != CCU_SUCCESS ||
            HcommThreadNotifyWaitOnThread(executionState.workers[0], 0, 0) != 0) {
            HCCL_ERROR("[DispatchReduceScatter] failed to schedule dual-Die slot-gather kernels");
            return HCCL_E_INTERNAL;
        }

        std::vector<uint64_t> mergeArgs;
        mergeArgs.reserve(remoteCount + 4);
        for (uint32_t slotIdx = 0; slotIdx < remoteCount; ++slotIdx) {
            mergeArgs.push_back(executionState.stagingBase + slotIdx * chunkCapacity);
        }
        mergeArgs.push_back(outputAddr + offset);
        mergeArgs.push_back(outputToken);
        mergeArgs.push_back(scratchToken);
        mergeArgs.push_back(chunkBytes);
        if (HcommCcuKernelLaunch(executionState.workers[0], executionState.programs[2], mergeArgs.data(), mergeArgs.size()) !=
            CCU_SUCCESS) {
            HCCL_ERROR("[DispatchReduceScatter] failed to launch the ordered slot merge kernel");
            return HCCL_E_INTERNAL;
        }
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
