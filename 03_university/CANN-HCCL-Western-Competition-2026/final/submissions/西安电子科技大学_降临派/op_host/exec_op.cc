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
#include <hcomm/ccu/ccu_launch.h>
#include <hcomm/hcomm_primitives.h>

#include <array>
#include <vector>

#include "custom.h"
#include "exec_op.h"

namespace ops_hccl {
namespace {
    constexpr uint64_t FP32_BYTES = sizeof(float);
    constexpr uint64_t MAX_BYTES = 512ULL * 1024ULL * 1024ULL;
    constexpr uint32_t DIE_NUM = 2;
    constexpr uint64_t LARGE_TRANSFER_ALIGNMENT = 256ULL;
    constexpr uint64_t LARGE_TRANSFER_ALIGN_ELEMENTS = LARGE_TRANSFER_ALIGNMENT / FP32_BYTES;

    struct SplitLayout {
        uint64_t normalBytes;
        uint64_t lastBytes;
    };

    HcclResult GetToken(uint64_t address, uint64_t size, uint64_t &token)
    {
        CcuResult result = HcommCcuGetMemToken(address, size, &token);
        if (result != CCU_SUCCESS) {
            HCCL_ERROR("HcommCcuGetMemToken failed: %d", result);
            return ConvertCcuToHccl(result);
        }
        return HCCL_SUCCESS;
    }

    SplitLayout SplitElements(uint64_t count, uint32_t parts)
    {
        uint64_t normal = count / parts;
        uint64_t last = count - normal * (parts - 1U);
        uint64_t aligned = (normal + LARGE_TRANSFER_ALIGN_ELEMENTS - 1U) / LARGE_TRANSFER_ALIGN_ELEMENTS
                           * LARGE_TRANSFER_ALIGN_ELEMENTS;
        // 对齐普通 Cell 的起始地址和长度，同时保证尾 Cell 不超过原始最大值。
        if (aligned <= last) {
            normal = aligned;
            last = count - normal * (parts - 1U);
        }
        return SplitLayout{normal * FP32_BYTES, last * FP32_BYTES};
    }

    HcclResult PreSyncInterThreads(const std::vector<ThreadHandle> &threads, uint32_t primaryDie)
    {
        // 主 Thread 到达当前阶段后再释放从 Thread。
        const uint32_t otherDie = primaryDie ^ 1U;
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[primaryDie], threads[otherDie], 0)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThreadWithDefaultTimeout(threads[otherDie], 0)));
        return HCCL_SUCCESS;
    }

    HcclResult PostSyncInterThreads(const std::vector<ThreadHandle> &threads, uint32_t primaryDie)
    {
        // 后续阶段下发前，主 Thread 必须等待从 Thread 完成。
        const uint32_t otherDie = primaryDie ^ 1U;
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThreadWithDefaultTimeout(threads[primaryDie], 0)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(threads[otherDie], threads[primaryDie], 0)));
        return HCCL_SUCCESS;
    }

    template <size_t N>
    HcclResult LaunchOnce(const AlgResourceCtx &resources, const std::vector<CcuKernelHandle> &kernels,
        const std::array<uint64_t, N> &args, const char *name, uint32_t argCount = static_cast<uint32_t>(N))
    {
        CHK_PRT_RET(resources.ccuThreads.empty() || resources.ccuThreads.size() != kernels.size(),
            HCCL_ERROR("invalid launch resources for %s", name), HCCL_E_INTERNAL);
        CHK_PRT_RET(argCount == 0 || argCount > N, HCCL_ERROR("invalid task argument count %u for %s", argCount, name),
            HCCL_E_INTERNAL);
        uint32_t activeCount = 0;
        for (CcuKernelHandle kernel : kernels) {
            activeCount += kernel != 0 ? 1U : 0U;
        }
        CHK_PRT_RET(activeCount == 0, HCCL_ERROR("no kernel is available for %s", name), HCCL_E_INTERNAL);
        CHK_PRT_RET(activeCount > 1 && kernels.size() != 2, HCCL_ERROR("unsupported multi-thread launch for %s", name),
            HCCL_E_INTERNAL);
        if (activeCount == 2) {
            CHK_PRT_RET(resources.primaryDie >= 2U || resources.launchFirstDie >= 2U,
                HCCL_ERROR("invalid dual-die scheduling metadata"), HCCL_E_INTERNAL);
            CHK_RET(PreSyncInterThreads(resources.ccuThreads, resources.primaryDie));
        }
        for (size_t step = 0; step < kernels.size(); ++step) {
            const size_t index = activeCount == 2 ? (resources.launchFirstDie + step) % 2U : step;
            if (kernels[index] == 0) {
                continue;
            }
            CcuResult result = HcommCcuKernelLaunch(resources.ccuThreads[index], kernels[index], args.data(), argCount);
            if (result != CCU_SUCCESS) {
                HCCL_ERROR("%s kernel launch failed at index %zu: %d", name, index, result);
                return ConvertCcuToHccl(result);
            }
        }
        if (activeCount == 2) {
            CHK_RET(PostSyncInterThreads(resources.ccuThreads, resources.primaryDie));
        }
        return HCCL_SUCCESS;
    }

    HcclResult LaunchLatin(
        const AlgResourceCtx &resources, const OpParam &param, uint64_t inputToken, uint64_t outputToken)
    {
        uint32_t cellCount = param.rankSize * (param.rankSize - 1U);
        SplitLayout cells = SplitElements(param.count, cellCount);
        std::array<uint64_t, latin::MAX_ARG_COUNT> args = {reinterpret_cast<uint64_t>(param.inputPtr), inputToken,
            reinterpret_cast<uint64_t>(param.outputPtr), outputToken, param.inputPtr == param.outputPtr ? 1ULL : 0ULL,
            cells.normalBytes, cells.lastBytes, latin::INIT_PHASE, 0};
        uint32_t argCount = latin::OFFSET_ARG + param.rankSize;
        CHK_RET(LaunchOnce(resources, resources.kernels, args, "Latin v2 large", argCount));
        args[7] = latin::BARRIER_PHASE;
        CHK_RET(LaunchOnce(resources, resources.kernels, args, "Latin v2 barrier", argCount));
        uint32_t stripsPerOwner = param.rankSize - 1U;
        for (uint32_t round = 0; round < stripsPerOwner; ++round) {
            args[7] = latin::REDUCE_PHASE;
            args[8] = param.myRank == stripsPerOwner - 1U - round ? 1ULL : 0ULL;
            for (uint32_t owner = 0; owner < param.rankSize; ++owner) {
                uint32_t difference = (param.myRank + param.rankSize - owner) % param.rankSize;
                if (difference == 0U) {
                    continue;
                }
                uint32_t strip = (difference - 1U + round) % stripsPerOwner;
                uint64_t cell = static_cast<uint64_t>(owner) * stripsPerOwner + strip;
                args[latin::OFFSET_ARG + owner] = cell * cells.normalBytes;
            }
            // 每轮向各远端 Owner 写入一个 Strip，随后用 Barrier 保证轮次顺序。
            CHK_RET(LaunchOnce(resources, resources.kernels, args, "Latin v2 large", argCount));
            args[7] = latin::BARRIER_PHASE;
            CHK_RET(LaunchOnce(resources, resources.kernels, args, "Latin v2 barrier", argCount));
        }
        args[7] = latin::GATHER_PHASE;
        CHK_RET(LaunchOnce(resources, resources.kernels, args, "Latin v2 large", argCount));
        args[7] = latin::BARRIER_PHASE;
        return LaunchOnce(resources, resources.kernels, args, "Latin v2 release", argCount);
    }

    HcclResult LaunchSmall(
        const AlgResourceCtx &resources, const OpParam &param, uint64_t inputToken, uint64_t outputToken)
    {
        std::array<uint64_t, small_data::TASK_ARG_COUNT> args = {reinterpret_cast<uint64_t>(param.inputPtr), inputToken,
            reinterpret_cast<uint64_t>(param.outputPtr), outputToken, resources.scratchToken};
        return LaunchOnce(resources, resources.kernels, args, "final small allreduce");
    }

    HcclResult SplitPackedCells(uint64_t ownerBytes, uint32_t channelCount, uint64_t &cellBytes, uint64_t &tailBytes)
    {
        CHK_PRT_RET(channelCount == 0 || ownerBytes % FP32_BYTES != 0, HCCL_ERROR("invalid packed-pull owner split"),
            HCCL_E_PARA);
        const uint64_t cellElements = ownerBytes / FP32_BYTES / channelCount;
        CHK_PRT_RET(cellElements == 0, HCCL_ERROR("packed-pull cell is empty"), HCCL_E_PARA);
        cellBytes = cellElements * FP32_BYTES;
        tailBytes = ownerBytes - cellBytes * channelCount;
        CHK_PRT_RET(cellBytes > MAX_DATA_SIZE || tailBytes >= FP32_BYTES * channelCount,
            HCCL_ERROR("invalid packed-pull cell layout"), HCCL_E_PARA);
        return HCCL_SUCCESS;
    }

    HcclResult LaunchPacked2x8(
        const AlgResourceCtx &resources, const OpParam &param, uint64_t inputToken, uint64_t outputToken)
    {
        CHK_PRT_RET(param.rankSize != 16U || param.inputPtr == param.outputPtr || resources.ccuThreads.size() != DIE_NUM
                        || resources.kernels.size() != DIE_NUM || resources.channelCounts[0] == 0
                        || resources.channelCounts[1] == 0,
            HCCL_ERROR("invalid packed 2x8 resources"), HCCL_E_INTERNAL);
        const uint64_t normalElements = param.count / param.rankSize;
        const uint64_t lastElements = param.count - normalElements * (param.rankSize - 1U);
        const uint64_t normalBytes = normalElements * FP32_BYTES;
        const uint64_t lastBytes = lastElements * FP32_BYTES;
        const uint64_t localOwnerBytes = param.myRank + 1U == param.rankSize ? lastBytes : normalBytes;
        uint64_t cellBytes[DIE_NUM]{};
        uint64_t tailBytes[DIE_NUM]{};
        for (uint32_t die = 0; die < DIE_NUM; ++die) {
            CHK_RET(SplitPackedCells(localOwnerBytes, resources.channelCounts[die], cellBytes[die], tailBytes[die]));
        }
        CHK_PRT_RET(resources.scratchAddr == 0 || resources.scratchSize < lastBytes,
            HCCL_ERROR("packed-pull scratch is too small"), HCCL_E_UNAVAIL);
        uint64_t scratchToken = 0;
        CHK_RET(GetToken(resources.scratchAddr, lastBytes, scratchToken));
        std::array<uint64_t, packed_pull::TASK_ARG_COUNT> args = {reinterpret_cast<uint64_t>(param.inputPtr),
            inputToken, reinterpret_cast<uint64_t>(param.outputPtr), outputToken, resources.scratchAddr, scratchToken,
            normalBytes, lastBytes, cellBytes[0], tailBytes[0], cellBytes[1], tailBytes[1], packed_pull::INIT_PHASE};
        // 两个 Die 分别拉取互斥分片，在主 Die 合并后收集所有完整 Owner 分片。
        for (uint64_t phase : {packed_pull::INIT_PHASE, packed_pull::MERGE_PHASE, packed_pull::GATHER_PHASE}) {
            args[12] = phase;
            CHK_RET(LaunchOnce(resources, resources.kernels, args, "packed 2x8 endpoint-pull"));
        }
        return HCCL_SUCCESS;
    }

    HcclResult LaunchFused4x1(
        const AlgResourceCtx &resources, const OpParam &param, uint64_t inputToken, uint64_t outputToken)
    {
        CHK_PRT_RET(param.rankSize != fused_4x1::RANK_SIZE || resources.kernels.size() != 1U
                        || resources.ccuThreads.size() != 1U,
            HCCL_ERROR("invalid fused 4x1 resources"), HCCL_E_INTERNAL);
        const uint64_t sliceBytes = param.count / fused_4x1::RANK_SIZE * FP32_BYTES;
        const uint64_t lastBytes = param.count * FP32_BYTES - sliceBytes * (fused_4x1::RANK_SIZE - 1U);
        std::array<uint64_t, fused_4x1::TASK_ARG_COUNT> args = {reinterpret_cast<uint64_t>(param.inputPtr),
            reinterpret_cast<uint64_t>(param.outputPtr), inputToken, outputToken, 0, sliceBytes, sliceBytes, sliceBytes,
            sliceBytes * 2U, sliceBytes, sliceBytes * 3U, lastBytes};
        return LaunchOnce(resources, resources.kernels, args, "fused 4x1 RSAG");
    }
} // namespace

HcclResult ExecOp(const OpParam &param)
{
    char *context = static_cast<char *>(param.resCtx);
    std::vector<char> sequence(context, context + param.ctxSize);
    AlgResourceCtx resources{};
    resources.DeSerialize(sequence);

    uint64_t dataBytes = param.count * FP32_BYTES;
    CHK_PRT_RET(dataBytes > MAX_BYTES, HCCL_ERROR("message exceeds 512 MiB"), HCCL_E_PARA);
    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    CHK_RET(GetToken(reinterpret_cast<uint64_t>(param.inputPtr), dataBytes, inputToken));
    if (param.inputPtr == param.outputPtr) {
        outputToken = inputToken;
    } else {
        CHK_RET(GetToken(reinterpret_cast<uint64_t>(param.outputPtr), dataBytes, outputToken));
    }
    switch (resources.mode) {
        case KernelMode::SMALL_OWNER_PULL:
        case KernelMode::SMALL_4X1_ONESHOT:
            CHK_PRT_RET(dataBytes != small_data::TOTAL_BYTES,
                HCCL_ERROR("small resource reused for a non-small message"), HCCL_E_INTERNAL);
            return LaunchSmall(resources, param, inputToken, outputToken);
        case KernelMode::LARGE_2X8_PACKED_PULL:
            return LaunchPacked2x8(resources, param, inputToken, outputToken);
        case KernelMode::LARGE_4X1_FUSED_RSAG:
            return LaunchFused4x1(resources, param, inputToken, outputToken);
        default:
            return LaunchLatin(resources, param, inputToken, outputToken);
    }
}
} // namespace ops_hccl
