/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include <ccu_launch.h>
#include <ccu_res.h>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {

HcclResult CheckCcuResult(CcuResult result, const char *operation)
{
    if (result == CCU_SUCCESS) {
        return HCCL_SUCCESS;
    }
    HCCL_ERROR("[F2ExecOp] %s failed, ret %d", operation, result);
    return ConvertCcuToHccl(result);
}

HcclResult CheckThreadResult(int32_t result, const char *operation)
{
    if (result == static_cast<int32_t>(CCU_SUCCESS)) {
        return HCCL_SUCCESS;
    }
    HCCL_ERROR("[F2ExecOp] %s failed, ret %d", operation, result);
    return ConvertCcuToHccl(static_cast<CcuResult>(result));
}

HcclResult StartWorker(const AlgResourceCtx &resource)
{
    CHK_RET(CheckThreadResult(HcommThreadNotifyRecordOnThread(
        resource.threads[0], resource.threads[1], 0), "worker start record"));
    return CheckThreadResult(HcommThreadNotifyWaitOnThread(
        resource.threads[1], 0, CUSTOM_TIMEOUT), "worker start wait");
}

HcclResult FinishAndWaitWorker(const AlgResourceCtx &resource)
{
    CHK_RET(CheckThreadResult(HcommThreadNotifyRecordOnThread(
        resource.threads[1], resource.threads[0], 0), "worker finish record"));
    return CheckThreadResult(HcommThreadNotifyWaitOnThread(
        resource.threads[0], 0, CUSTOM_TIMEOUT), "worker finish wait");
}

HcclResult LaunchKernel(ThreadHandle thread, CcuKernelHandle kernel, const std::vector<uint64_t> &arguments)
{
    return CheckCcuResult(
        HcommCcuKernelLaunch(thread, kernel, arguments.data(), arguments.size()), "kernel launch");
}

HcclResult RunPhase(const AlgResourceCtx &resource, const std::array<bool, F2_DIE_COUNT> &active,
    uint32_t kernelIndex, const std::vector<uint64_t> &arguments)
{
    uint32_t activeCount = 0;
    uint32_t firstDie = F2_DIE_COUNT;
    for (uint32_t die = 0; die < F2_DIE_COUNT; ++die) {
        if (active[die]) {
            ++activeCount;
            if (firstDie == F2_DIE_COUNT) {
                firstDie = die;
            }
        }
    }
    CHK_PRT_RET(activeCount == 0, HCCL_ERROR("[F2ExecOp] phase has no active die"), HCCL_E_INTERNAL);

    const uint32_t firstBase = firstDie * F2_KERNEL_PER_DIE;
    if (activeCount == 1) {
        return LaunchKernel(resource.threads[0],
            resource.ccuKernels[firstBase + kernelIndex], arguments);
    }

    const uint32_t workerDie = firstDie ^ 1U;
    const uint32_t workerBase = workerDie * F2_KERNEL_PER_DIE;
    CHK_RET(StartWorker(resource));
    CHK_RET(LaunchKernel(resource.threads[0],
        resource.ccuKernels[firstBase + kernelIndex], arguments));
    CHK_RET(LaunchKernel(resource.threads[1],
        resource.ccuKernels[workerBase + kernelIndex], arguments));
    return FinishAndWaitWorker(resource);
}

HcclResult LaunchDistributionOnThread(const AlgResourceCtx &resource, uint32_t die,
    ThreadHandle thread, const std::vector<uint64_t> &arguments)
{
    const uint32_t base = die * F2_KERNEL_PER_DIE;
    return LaunchKernel(thread, resource.ccuKernels[base + F2_KERNEL_SCATTER], arguments);
}

HcclResult RunDistribution(const AlgResourceCtx &resource,
    const std::array<bool, F2_DIE_COUNT> &active, const std::vector<uint64_t> &arguments)
{
    uint32_t activeCount = 0;
    uint32_t firstDie = F2_DIE_COUNT;
    for (uint32_t die = 0; die < F2_DIE_COUNT; ++die) {
        if (active[die]) {
            ++activeCount;
            if (firstDie == F2_DIE_COUNT) {
                firstDie = die;
            }
        }
    }
    CHK_PRT_RET(activeCount == 0,
        HCCL_ERROR("[F2ExecOp] distribution has no active die"), HCCL_E_INTERNAL);

    if (activeCount == 1) {
        return LaunchDistributionOnThread(resource, firstDie, resource.threads[0], arguments);
    }

    const uint32_t workerDie = firstDie ^ 1U;
    CHK_RET(StartWorker(resource));
    CHK_RET(LaunchDistributionOnThread(resource, firstDie, resource.threads[0], arguments));
    CHK_RET(LaunchDistributionOnThread(resource, workerDie, resource.threads[1], arguments));
    return FinishAndWaitWorker(resource);
}

HcclResult RunPacked(const AlgResourceCtx &resource,
    const std::array<bool, F2_DIE_COUNT> &active, const std::vector<uint64_t> &arguments)
{
    uint32_t packedDie = F2_DIE_COUNT;
    for (uint32_t die = 0; die < F2_DIE_COUNT; ++die) {
        if (active[die]) {
            CHK_PRT_RET(packedDie != F2_DIE_COUNT,
                HCCL_ERROR("[F2ExecOp] packed phase has multiple active dies"), HCCL_E_INTERNAL);
            packedDie = die;
        }
    }
    CHK_PRT_RET(packedDie == F2_DIE_COUNT,
        HCCL_ERROR("[F2ExecOp] packed phase has no active die"), HCCL_E_INTERNAL);
    const uint32_t base = packedDie * F2_KERNEL_PER_DIE;
    return LaunchKernel(resource.threads[0],
        resource.ccuKernels[base + F2_KERNEL_PACKED], arguments);
}

HcclResult RunMixedForest(const AlgResourceCtx &resource,
    const std::array<bool, F2_DIE_COUNT> &active, const std::vector<uint64_t> &arguments)
{
    return RunPhase(resource, active, F4_KERNEL_PIPELINED_FOREST, arguments);
}
HcclResult RunClosPipeline(const AlgResourceCtx &resource,
    const std::array<bool, F2_DIE_COUNT> &active, const std::vector<uint64_t> &arguments)
{
    uint32_t activeDie = F2_DIE_COUNT;
    for (uint32_t die = 0; die < F2_DIE_COUNT; ++die) {
        if (!active[die]) {
            continue;
        }
        CHK_PRT_RET(activeDie != F2_DIE_COUNT,
            HCCL_ERROR("[F2ExecOp] Clos pipeline spans multiple dies"), HCCL_E_INTERNAL);
        activeDie = die;
    }
    CHK_PRT_RET(activeDie == F2_DIE_COUNT,
        HCCL_ERROR("[F2ExecOp] Clos pipeline has no active die"), HCCL_E_INTERNAL);
    const uint32_t base = activeDie * F2_KERNEL_PER_DIE;
    return LaunchKernel(resource.threads[0],
        resource.ccuKernels[base + F2_KERNEL_CLOS_PIPELINE], arguments);
}

HcclResult RunMeshScatterAllgather(const OpParam &param, const AlgResourceCtx &resource,
    const std::array<bool, F2_DIE_COUNT> &active, uint64_t baseAddress,
    uint64_t token, uint64_t typeSize)
{
    constexpr uint64_t meshPhaseScatter = 0;
    constexpr uint64_t meshPhaseAllgather = 1;
    const uint64_t maxCount = F6_MESH_SLICE_BYTES / typeSize;
    uint64_t processed = 0;
    while (processed < param.count) {
        const uint64_t sliceCount = std::min(maxCount, param.count - processed);
        const uint64_t sliceBytes = sliceCount * typeSize;
        const uint64_t sliceAddress = baseAddress + processed * typeSize;
        const uint64_t normalSlice = sliceBytes / param.rankSize;
        const uint64_t lastSlice = normalSlice + sliceBytes % param.rankSize;
        const std::vector<uint64_t> scatterArguments = {
            sliceAddress, token, normalSlice, lastSlice, meshPhaseScatter};
        CHK_RET(RunPhase(resource, active, F6_KERNEL_MESH, scatterArguments));
        const std::vector<uint64_t> allgatherArguments = {
            sliceAddress, token, normalSlice, lastSlice, meshPhaseAllgather};
        CHK_RET(RunPhase(resource, active, F6_KERNEL_MESH, allgatherArguments));
        processed += sliceCount;
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param)
{
    if (param.count == 0 || param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    auto typeSize = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(typeSize == SIZE_TABLE.end(), HCCL_ERROR("[F2ExecOp] unsupported data type"), HCCL_E_NOT_SUPPORT);
    const uint64_t dataSize = param.count * typeSize->second;

    char *context = static_cast<char *>(param.resCtx);
    std::vector<char> serialized(context, context + param.ctxSize);
    AlgResourceCtx resource;
    resource.DeSerialize(serialized);
    CHK_PRT_RET(resource.threads.size() != F2_WORKER_COUNT + 1 ||
        resource.ccuKernels.size() != F2_KERNEL_COUNT ||
        resource.dieChannelCounts.size() != F2_DIE_COUNT ||
        resource.algorithmMode > F6_ALG_MESH_SCATTER_ALLGATHER,
        HCCL_ERROR("[F2ExecOp] invalid resource context"), HCCL_E_INTERNAL);

    const uint64_t baseAddress = reinterpret_cast<uint64_t>(param.outputPtr);
    uint64_t token = 0;
    HcommCcuGetMemToken(baseAddress, dataSize, &token);
    const std::vector<uint64_t> arguments = {baseAddress, token};

    std::array<bool, F2_DIE_COUNT> active = {};
    uint32_t activeCount = 0;
    for (uint32_t die = 0; die < F2_DIE_COUNT; ++die) {
        active[die] = resource.dieChannelCounts[die] != 0;
        activeCount += static_cast<uint32_t>(active[die]);
    }
    CHK_PRT_RET(activeCount == 0, HCCL_ERROR("[F2ExecOp] no active CCU die"), HCCL_E_INTERNAL);

    if (resource.algorithmMode == F6_ALG_MESH_SCATTER_ALLGATHER) {
        return RunMeshScatterAllgather(
            param, resource, active, baseAddress, token, typeSize->second);
    }

    if (dataSize <= F2_SMALL_DIRECT_LIMIT) {
        CHK_RET(RunPhase(resource, active, F2_KERNEL_DIRECT_0, arguments));
        return HCCL_SUCCESS;
    }

    if (resource.algorithmMode == F4_ALG_PIPELINED_FOREST) {
        return RunMixedForest(resource, active, arguments);
    }
    if (resource.algorithmMode == F4_ALG_CLOS_PIPELINE) {
        return RunClosPipeline(resource, active, arguments);
    }

    if (param.rankSize == 4) {
        uint32_t packedDie = F2_DIE_COUNT;
        for (uint32_t die = 0; die < F2_DIE_COUNT; ++die) {
            if (resource.dieChannelCounts[die] == F2_PATH_COUNT) {
                packedDie = die;
            }
        }
        CHK_PRT_RET(activeCount != 1 || packedDie >= F2_DIE_COUNT,
            HCCL_ERROR("[F2ExecOp] invalid packed topology, active=%u packedDie=%u",
                activeCount, packedDie), HCCL_E_INTERNAL);
        return RunPacked(resource, active, arguments);
    }

    CHK_RET(RunDistribution(resource, active, arguments));
    CHK_RET(RunPhase(resource, active, F2_KERNEL_LOCAL, arguments));
    return HCCL_SUCCESS;
}

} // namespace ops_hccl
