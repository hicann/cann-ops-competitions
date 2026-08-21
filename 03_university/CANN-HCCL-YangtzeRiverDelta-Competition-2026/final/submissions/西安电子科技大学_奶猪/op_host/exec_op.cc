/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {
ThreadHandle ThreadFor(const OpParam &param, const ForestResourceContext &resource, uint32_t kernelIndex)
{
    return kernelIndex == 0 ? param.cpuThread : resource.auxiliaryThreads[kernelIndex - 1];
}

HcclResult OrderThreads(ThreadHandle producer, ThreadHandle consumer, uint32_t notifyIndex)
{
    CHK_RET(HcommThreadNotifyRecordOnThread(producer, consumer, notifyIndex));
    CHK_RET(HcommThreadNotifyWaitOnThread(consumer, notifyIndex, CUSTOM_TIMEOUT));
    return HCCL_SUCCESS;
}

HcclResult LaunchPhase(const OpParam &param, const ForestResourceContext &resource,
    const std::vector<uint32_t> &activeKernels, ForestPhase phase, uint64_t address, uint64_t token)
{
    for (uint32_t kernelIndex : activeKernels) {
        CHK_PRT_RET(kernelIndex >= resource.kernels.size(),
            HCCL_ERROR("[ForestV4] invalid active kernel index %u", kernelIndex), HCCL_E_INTERNAL);
        if (kernelIndex != 0) {
            CHK_RET(OrderThreads(param.cpuThread, ThreadFor(param, resource, kernelIndex), 0));
        }
    }

    uint64_t arguments[3 + 2 * FOREST_RING_COUNT] = {
        address,
        token,
        static_cast<uint32_t>(phase),
    };
    for (uint32_t kernelIndex : activeKernels) {
        CHK_RET_CCU(HcommCcuKernelLaunch(ThreadFor(param, resource, kernelIndex),
            resource.kernels[kernelIndex], arguments, 3 + 2 * FOREST_RING_COUNT));
    }

    for (uint32_t kernelIndex : activeKernels) {
        if (kernelIndex != 0) {
            CHK_RET(OrderThreads(ThreadFor(param, resource, kernelIndex), param.cpuThread, 1));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchRingStep(const OpParam &param, const ForestResourceContext &resource,
    const std::vector<uint32_t> &activeKernels, uint64_t address, uint64_t token, uint32_t step)
{
    for (uint32_t kernelIndex : activeKernels) {
        if (kernelIndex != 0) {
            CHK_RET(OrderThreads(param.cpuThread, ThreadFor(param, resource, kernelIndex), 0));
        }
    }

    uint64_t arguments[3 + 2 * FOREST_RING_COUNT] = {
        address, token, static_cast<uint32_t>(ForestPhase::RING),
    };
    const uint64_t laneQuotient = param.count / param.rankSize;
    const uint64_t laneRemainder = param.count % param.rankSize;
    for (uint32_t ring = 0; ring < FOREST_RING_COUNT; ++ring) {
        const auto begin = resource.ringOrders.begin() + ring * param.rankSize;
        const auto end = begin + param.rankSize;
        const auto position = std::find(begin, end, param.myRank);
        CHK_PRT_RET(position == end,
            HCCL_ERROR("[ForestV4] rank %u is absent from ring %u", param.myRank, ring),
            HCCL_E_INTERNAL);
        const uint32_t rankPosition = static_cast<uint32_t>(position - begin);
        const uint32_t ownerPosition = (rankPosition + param.rankSize - step) % param.rankSize;
        const uint32_t owner = *(begin + ownerPosition);
        const uint64_t laneElements = laneQuotient + (owner < laneRemainder ? 1ULL : 0ULL);
        const uint64_t laneOffset = owner * laneQuotient + std::min<uint64_t>(owner, laneRemainder);
        const uint64_t partQuotient = laneElements / FOREST_RING_COUNT;
        const uint64_t partRemainder = laneElements % FOREST_RING_COUNT;
        const uint64_t partElements = partQuotient + (ring < partRemainder ? 1ULL : 0ULL);
        const uint64_t partOffset = ring * partQuotient + std::min<uint64_t>(ring, partRemainder);
        arguments[3 + ring] = (laneOffset + partOffset) * sizeof(float);
        arguments[3 + FOREST_RING_COUNT + ring] = partElements * sizeof(float);
    }

    for (uint32_t kernelIndex : activeKernels) {
        CHK_RET_CCU(HcommCcuKernelLaunch(ThreadFor(param, resource, kernelIndex),
            resource.kernels[kernelIndex], arguments, 3 + 2 * FOREST_RING_COUNT));
    }
    for (uint32_t kernelIndex : activeKernels) {
        if (kernelIndex != 0) {
            CHK_RET(OrderThreads(ThreadFor(param, resource, kernelIndex), param.cpuThread, 1));
        }
    }
    return HCCL_SUCCESS;
}

uint32_t RequiredTreeStages(uint32_t rankSize)
{
    return rankSize <= FOREST_RADIX ? 1U : FOREST_STAGE_COUNT;
}
} // namespace

HcclResult ExecOp(const OpParam &param)
{
    CHK_PRT_RET(param.resCtx == nullptr || param.ctxSize == 0,
        HCCL_ERROR("[ForestV4] missing resource context"), HCCL_E_INTERNAL);
    char *rawContext = static_cast<char *>(param.resCtx);
    std::vector<char> encoded(rawContext, rawContext + param.ctxSize);
    ForestResourceContext resource;
    resource.Decode(encoded);
    CHK_PRT_RET(resource.kernels.empty() || resource.kernels.size() > 2
            || resource.auxiliaryThreads.size() + 1 != resource.kernels.size()
            || resource.directActive.size() != resource.kernels.size()
            || resource.ringOrders.size() != FOREST_RING_COUNT * param.rankSize,
        HCCL_ERROR("[ForestV4] malformed resource context"), HCCL_E_INTERNAL);

    const uint64_t totalBytes = param.count * sizeof(float);
    const uint64_t baseAddress = reinterpret_cast<uint64_t>(param.outputPtr);
    uint64_t memoryToken = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(baseAddress, totalBytes, &memoryToken));

    std::vector<uint32_t> activeKernels;
    if (totalBytes <= DIRECT_MESSAGE_LIMIT) {
        for (uint32_t index = 0; index < resource.kernels.size(); ++index) {
            if (resource.directActive[index] != 0) {
                activeKernels.push_back(index);
            }
        }
        CHK_PRT_RET(activeKernels.empty(),
            HCCL_ERROR("[ForestV4] no direct kernel reaches root %u", param.root), HCCL_E_INTERNAL);
        return LaunchPhase(param, resource, activeKernels,
            ForestPhase::DIRECT, baseAddress, memoryToken);
    }

    for (uint32_t index = 0; index < resource.kernels.size(); ++index) {
        activeKernels.push_back(index);
    }
    CHK_RET(LaunchPhase(param, resource, activeKernels,
        ForestPhase::SEED, baseAddress, memoryToken));

    if (param.rankSize > FOREST_RADIX) {
        for (uint32_t step = 0; step + 1 < param.rankSize; ++step) {
            CHK_RET(LaunchRingStep(param, resource, activeKernels,
                baseAddress, memoryToken, step));
        }
        return HCCL_SUCCESS;
    }

    const uint32_t stageCount = RequiredTreeStages(param.rankSize);
    CHK_PRT_RET(stageCount == 0 || stageCount > FOREST_STAGE_COUNT,
        HCCL_ERROR("[ForestV4] rank size %u needs unsupported tree depth %u", param.rankSize, stageCount),
        HCCL_E_NOT_SUPPORT);
    for (uint32_t stage = 0; stage < stageCount; ++stage) {
        const ForestPhase phase = static_cast<ForestPhase>(
            static_cast<uint32_t>(ForestPhase::LEVEL_ZERO) + stage);
        CHK_RET(LaunchPhase(param, resource, activeKernels, phase, baseAddress, memoryToken));
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
