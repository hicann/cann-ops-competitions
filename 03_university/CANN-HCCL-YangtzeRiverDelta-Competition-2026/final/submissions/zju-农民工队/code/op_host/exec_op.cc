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
#include <cstring>
#include <ccu/ccu_res.h>
#include "ccu_launch.h"

#include "custom.h"
#include "log.h"
#include "exec_op.h"
#include "ccu_kernel.h"

namespace ops_hccl {
namespace {
    constexpr uint32_t WORKER_NOTIFY_ID = 0;
    constexpr uint32_t WORKER_PHASE0_NOTIFY_ID = 1;
    constexpr uint32_t WORKER_PHASE1_NOTIFY_ID = 2;
    constexpr uint32_t TWELVE_SLICE_NUM = 24;
    constexpr uint32_t TWELVE_MODE_ARG_INDEX = 2 + 2 * TWELVE_SLICE_NUM;
    constexpr uint32_t TWELVE_TASK_ARG_NUM = TWELVE_MODE_ARG_INDEX + 1;
    constexpr uint64_t TWELVE_LOCAL_SCATTER_A_MODE = 0;
    constexpr uint64_t TWELVE_REMOTE_SCATTER_D_MODE = 1;
    constexpr uint64_t TWELVE_LOCAL_SCATTER_B_GATHER_A_MODE = 2;
    constexpr uint64_t TWELVE_REMOTE_ALLGATHER_D_MODE = 3;
    constexpr uint64_t TWELVE_CROSS_WAVE_AD_C_MODE = 4;
    constexpr uint64_t TWELVE_LOCAL_ALLGATHER_B_MODE = 5;
    constexpr uint64_t TWELVE_REMOTE_ALLGATHER_C_MODE = 6;
    constexpr uint64_t TWELVE_CROSS_WAVE_BC_MODE = 7;
    constexpr uint32_t SIXTEEN_SLICE_NUM = 16;
    constexpr uint32_t SIXTEEN_TASK_ARG_NUM = 2 + 2 * SIXTEEN_SLICE_NUM;
    constexpr uint32_t SIXTEEN_AG_TASK_ARG_NUM = 4;
    constexpr uint32_t SIXTEEN_LOCAL_ALLGATHER_MARKER = 7;
    constexpr uint32_t SIXTEEN_CROSS_ALLGATHER_MARKER = 8;

    HcclResult LaunchKernel(ThreadHandle thread, const AlgResourceCtx &resCtx, uint32_t kernelIndex,
        const uint64_t *taskArgs, uint32_t argNum)
    {
        CcuResult ccuRet = HcommCcuKernelLaunch(thread, resCtx.ccuKernels[kernelIndex], taskArgs, argNum);
        CHK_PRT_RET(ccuRet != CCU_SUCCESS,
            HCCL_ERROR("CCU layer[%u] kernel launch failed[%d]", resCtx.kernelLayers[kernelIndex], ccuRet),
            ConvertCcuToHccl(ccuRet));
        return HCCL_SUCCESS;
    }

    template <std::size_t N>
    HcclResult LaunchKernel(ThreadHandle thread, const AlgResourceCtx &resCtx, uint32_t kernelIndex,
        const std::array<uint64_t, N> &taskArgs)
    {
        return LaunchKernel(thread, resCtx, kernelIndex, taskArgs.data(), static_cast<uint32_t>(N));
    }

    template <std::size_t N>
    HcclResult LaunchModeKernel(ThreadHandle thread, const AlgResourceCtx &resCtx, uint32_t kernelIndex,
        const std::array<uint64_t, N> &baseArgs, uint32_t modeIndex, uint64_t mode)
    {
        auto taskArgs = baseArgs;
        taskArgs[modeIndex] = mode;
        return LaunchKernel(thread, resCtx, kernelIndex, taskArgs);
    }

    HcclResult FindLayerKernel(const AlgResourceCtx &resCtx, uint32_t layer, uint32_t &kernelIndex)
    {
        auto layerIt = std::find(resCtx.kernelLayers.begin(), resCtx.kernelLayers.end(), layer);
        CHK_PRT_RET(layerIt == resCtx.kernelLayers.end(),
            HCCL_ERROR("CCU layer[%u] kernel not found", layer), HCCL_E_INTERNAL);
        kernelIndex = static_cast<uint32_t>(layerIt - resCtx.kernelLayers.begin());
        return HCCL_SUCCESS;
    }

    HcclResult LaunchChunk(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t baseAddr, uint64_t token,
        uint64_t offset, uint64_t size)
    {
        uint64_t sendEnabled = static_cast<uint64_t>(param.myRank == param.root);
        const std::array<uint64_t, 5> taskArgs = {baseAddr, token, offset, size, sendEnabled};

        const uint32_t invalidKernel = static_cast<uint32_t>(resCtx.ccuKernels.size());
        uint32_t layer0Kernel = invalidKernel;
        uint32_t layer1Kernel = invalidKernel;
        for (uint32_t i = 0; i < resCtx.ccuKernels.size(); ++i) {
            if (resCtx.kernelLayers[i] == 0) {
                layer0Kernel = i;
            } else if (resCtx.kernelLayers[i] == 1) {
                layer1Kernel = i;
            }
        }

        const bool hasLayer0 = layer0Kernel < invalidKernel;
        const bool hasLayer1 = layer1Kernel < invalidKernel;
        CHK_PRT_RET(!hasLayer0 && !hasLayer1, HCCL_ERROR("No layer 0 or layer 1 CCU kernel found"), HCCL_E_INTERNAL);
        if (!hasLayer0 || !hasLayer1) {
            // 只有一个 layer 时仍使用用户 stream 绑定的 thread，避免额外同步开销。
            uint32_t kernelIndex = hasLayer0 ? layer0Kernel : layer1Kernel;
            return LaunchKernel(param.cpuThread, resCtx, kernelIndex, taskArgs);
        }

        CHK_PRT_RET(resCtx.threads.size() < 2,
            HCCL_ERROR("Layer-parallel launch requires two CCU threads, got[%zu]", resCtx.threads.size()),
            HCCL_E_INTERNAL);
        ThreadHandle layer1Worker = resCtx.threads[1];

        // cpuThread 上的 start notify 保证 worker 在用户 stream 中的前置任务完成后启动。
        CHK_RET(
            static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(param.cpuThread, layer1Worker, WORKER_NOTIFY_ID)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(layer1Worker, WORKER_NOTIFY_ID, CUSTOM_TIMEOUT)));

        // Layer 1 进入 worker 队列，Layer 0 进入 cpuThread 队列，两者可并行执行。
        CHK_RET(LaunchKernel(layer1Worker, resCtx, layer1Kernel, taskArgs));
        CHK_RET(
            static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(layer1Worker, param.cpuThread, WORKER_NOTIFY_ID)));
        CHK_RET(LaunchKernel(param.cpuThread, resCtx, layer0Kernel, taskArgs));

        // 把 worker 的完成关系回收到用户 stream，避免算子提前返回。
        CHK_RET(
            static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(param.cpuThread, WORKER_NOTIFY_ID, CUSTOM_TIMEOUT)));
        return HCCL_SUCCESS;
    }

    HcclResult LaunchPeerKernel(ThreadHandle thread, const AlgResourceCtx &resCtx, uint32_t peer, uint64_t baseAddr,
        uint64_t token, uint64_t offset, uint64_t size, bool sendEnabled)
    {
        auto peerIt = std::find(resCtx.kernelPeers.begin(), resCtx.kernelPeers.end(), peer);
        CHK_PRT_RET(
            peerIt == resCtx.kernelPeers.end(), HCCL_ERROR("CCU kernel for peer[%u] not found", peer), HCCL_E_INTERNAL);
        uint32_t kernelIndex = static_cast<uint32_t>(peerIt - resCtx.kernelPeers.begin());
        const std::array<uint64_t, 5> taskArgs
            = {baseAddr, token, offset, size, static_cast<uint64_t>(sendEnabled)};
        return LaunchKernel(thread, resCtx, kernelIndex, taskArgs);
    }

    HcclResult ExecuteFourRankTreePlan(const OpParam &param, const BroadcastPlan &plan, const AlgResourceCtx &resCtx,
        uint64_t baseAddr, uint64_t token, uint64_t totalSize)
    {
        CHK_PRT_RET(param.rankSize != 4, HCCL_ERROR("Four-rank tree requires rankSize[4]"), HCCL_E_PARA);
        CHK_PRT_RET(plan.chunkSize == 0, HCCL_ERROR("Invalid chunk size[0]"), HCCL_E_PARA);

        const uint32_t firstPeer = FourRankFirstPeer(param.root);
        const uint32_t secondPeer = FourRankSecondPeer(param.root);
        const uint32_t forwardPeer = FourRankForwardPeer(param.root);
        uint64_t offset = 0;
        while (offset < totalSize) {
            uint64_t size = std::min<uint64_t>(plan.chunkSize, totalSize - offset);
            if (param.myRank == param.root) {
                // Round 1: root -> firstPeer. Round 2: root -> secondPeer.
                CHK_RET(LaunchPeerKernel(param.cpuThread, resCtx, firstPeer, baseAddr, token, offset, size, true));
                CHK_RET(LaunchPeerKernel(param.cpuThread, resCtx, secondPeer, baseAddr, token, offset, size, true));
            } else if (param.myRank == firstPeer) {
                // Round 1 接收完成后，在 Round 2 与 root 并行转发。
                CHK_RET(LaunchPeerKernel(param.cpuThread, resCtx, param.root, baseAddr, token, offset, size, false));
                CHK_RET(LaunchPeerKernel(param.cpuThread, resCtx, forwardPeer, baseAddr, token, offset, size, true));
            } else if (param.myRank == secondPeer) {
                CHK_RET(LaunchPeerKernel(param.cpuThread, resCtx, param.root, baseAddr, token, offset, size, false));
            } else {
                CHK_RET(LaunchPeerKernel(param.cpuThread, resCtx, firstPeer, baseAddr, token, offset, size, false));
            }
            offset += size;
        }
        return HCCL_SUCCESS;
    }

    HcclResult ExecuteFourRankPipelinePlan(
        const OpParam &param, const AlgResourceCtx &resCtx, uint64_t baseAddr, uint64_t token, uint64_t totalSize)
    {
        CHK_PRT_RET(param.rankSize != 4, HCCL_ERROR("Four-rank pipeline requires rankSize[4]"), HCCL_E_PARA);
        const uint64_t dataTypeSize = totalSize / param.count;
        const uint64_t regularSliceSize = (param.count / FOUR_RANK_PIPELINE_SLICE_NUM) * dataTypeSize;
        const uint64_t tailSliceSize = totalSize - regularSliceSize * (FOUR_RANK_PIPELINE_SLICE_NUM - 1U);
        CHK_PRT_RET(regularSliceSize == 0 || regularSliceSize > MAX_DATA_SIZE || tailSliceSize > MAX_DATA_SIZE
                        || tailSliceSize % dataTypeSize != 0,
            HCCL_ERROR("Invalid four-rank pipeline slice sizes[%lu, %lu]", regularSliceSize, tailSliceSize),
            HCCL_E_NOT_SUPPORT);

        CHK_PRT_RET(resCtx.threads.size() != 1 || resCtx.ccuKernels.size() != 1,
            HCCL_ERROR("Four-rank persistent pipeline requires one thread and one kernel"), HCCL_E_INTERNAL);
        // 前 15 片等长，最后一片承载余数；CCU kernel 内部维持偏移，只需 4 个动态参数。
        const std::array<uint64_t, 4> taskArgs = {baseAddr, token, regularSliceSize, tailSliceSize};
        const CcuResult ccuRet = HcommCcuKernelLaunch(
            param.cpuThread, resCtx.ccuKernels[0], taskArgs.data(), static_cast<uint32_t>(taskArgs.size()));
        CHK_PRT_RET(ccuRet != CCU_SUCCESS, HCCL_ERROR("CCU four-rank pipeline launch failed[%d]", ccuRet),
            ConvertCcuToHccl(ccuRet));
        return HCCL_SUCCESS;
    }

    template <uint32_t N> struct TwelveSlices {
        std::array<uint64_t, N> offsets{};
        std::array<uint64_t, N> sizes{};
    };

    TwelveSlices<TWELVE_SLICE_NUM> MakeTwelveSlices(uint64_t count, uint64_t dataTypeSize)
    {
        TwelveSlices<TWELVE_SLICE_NUM> slices;
        constexpr uint32_t baseUnitNum = 32;
        const uint64_t baseCount = count / baseUnitNum;
        const uint64_t remainder = count % baseUnitNum;
        auto hasRemainder = [remainder](uint32_t unit) {
            const uint32_t quarter = unit / 8U;
            const uint32_t position = unit % 8U;
            return 4U * position + quarter < remainder;
        };
        uint64_t offset = 0;
        for (uint32_t index = 0; index < TWELVE_SLICE_NUM; ++index) {
            // A/B 各为8个1/32分片，C/D各为4个1/16分片。
            uint64_t sliceCount = baseCount + static_cast<uint64_t>(hasRemainder(index));
            if (index >= 16U) {
                const uint32_t firstUnit = 16U + 2U * (index - 16U);
                sliceCount = 2U * baseCount + static_cast<uint64_t>(hasRemainder(firstUnit))
                             + static_cast<uint64_t>(hasRemainder(firstUnit + 1U));
            }
            slices.offsets[index] = offset;
            slices.sizes[index] = sliceCount * dataTypeSize;
            offset += slices.sizes[index];
        }
        return slices;
    }

    template <uint32_t N>
    HcclResult ValidateTwelveSlices(const TwelveSlices<N> &slices, uint64_t totalSize, uint64_t dataTypeSize)
    {
        uint64_t coveredSize = 0;
        for (uint32_t index = 0; index < N; ++index) {
            CHK_PRT_RET(slices.offsets[index] != coveredSize || slices.sizes[index] == 0
                            || slices.sizes[index] > MAX_DATA_SIZE || slices.sizes[index] % dataTypeSize != 0,
                HCCL_ERROR("Invalid twelve-rank slice[%u/%u] offset[%lu] size[%lu]", index, N, slices.offsets[index],
                    slices.sizes[index]),
                HCCL_E_NOT_SUPPORT);
            coveredSize += slices.sizes[index];
        }
        CHK_PRT_RET(coveredSize != totalSize,
            HCCL_ERROR("Twelve-rank slices cover[%lu] bytes, expected[%lu]", coveredSize, totalSize), HCCL_E_INTERNAL);
        return HCCL_SUCCESS;
    }

    HcclResult LaunchTwelveLocalPipeline(const OpParam &param, const AlgResourceCtx &resCtx,
        uint32_t localKernel, uint32_t crossKernel,
        const std::array<uint64_t, TWELVE_TASK_ARG_NUM> &taskArgs)
    {
        CHK_PRT_RET(resCtx.threads.size() != 2,
            HCCL_ERROR("Twelve-rank local pipeline requires two threads, got[%zu]", resCtx.threads.size()),
            HCCL_E_INTERNAL);
        const ThreadHandle worker = resCtx.threads[1];
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(param.cpuThread, worker, WORKER_NOTIFY_ID)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(worker, WORKER_NOTIFY_ID, CUSTOM_TIMEOUT)));
        if (param.myRank == param.root) {
            CHK_RET(LaunchModeKernel(worker, resCtx, crossKernel, taskArgs, TWELVE_MODE_ARG_INDEX,
                TWELVE_REMOTE_SCATTER_D_MODE));
        }
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(worker, WORKER_PHASE0_NOTIFY_ID, CUSTOM_TIMEOUT)));
        CHK_RET(LaunchModeKernel(
            worker, resCtx, crossKernel, taskArgs, TWELVE_MODE_ARG_INDEX, TWELVE_CROSS_WAVE_AD_C_MODE));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(worker, WORKER_PHASE1_NOTIFY_ID, CUSTOM_TIMEOUT)));
        CHK_RET(LaunchModeKernel(
            worker, resCtx, crossKernel, taskArgs, TWELVE_MODE_ARG_INDEX, TWELVE_CROSS_WAVE_BC_MODE));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(worker, param.cpuThread, WORKER_NOTIFY_ID)));

        CHK_RET(LaunchModeKernel(
            param.cpuThread, resCtx, localKernel, taskArgs, TWELVE_MODE_ARG_INDEX, TWELVE_LOCAL_SCATTER_A_MODE));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(param.cpuThread, worker, WORKER_PHASE0_NOTIFY_ID)));
        CHK_RET(LaunchModeKernel(param.cpuThread, resCtx, localKernel, taskArgs, TWELVE_MODE_ARG_INDEX,
            TWELVE_LOCAL_SCATTER_B_GATHER_A_MODE));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(param.cpuThread, worker, WORKER_PHASE1_NOTIFY_ID)));
        CHK_RET(LaunchModeKernel(
            param.cpuThread, resCtx, localKernel, taskArgs, TWELVE_MODE_ARG_INDEX, TWELVE_LOCAL_ALLGATHER_B_MODE));
        return static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(param.cpuThread, WORKER_NOTIFY_ID, CUSTOM_TIMEOUT));
    }

    HcclResult LaunchTwelveRemotePipeline(const OpParam &param, const AlgResourceCtx &resCtx,
        uint32_t localKernel, uint32_t crossKernel,
        const std::array<uint64_t, TWELVE_TASK_ARG_NUM> &taskArgs)
    {
        CHK_PRT_RET(resCtx.threads.size() != 2,
            HCCL_ERROR("Twelve-rank remote pipeline requires two threads, got[%zu]", resCtx.threads.size()),
            HCCL_E_INTERNAL);
        const ThreadHandle worker = resCtx.threads[1];

        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(param.cpuThread, worker, WORKER_NOTIFY_ID)));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(worker, WORKER_NOTIFY_ID, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(worker, WORKER_PHASE0_NOTIFY_ID, CUSTOM_TIMEOUT)));
        CHK_RET(LaunchModeKernel(
            worker, resCtx, crossKernel, taskArgs, TWELVE_MODE_ARG_INDEX, TWELVE_CROSS_WAVE_AD_C_MODE));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(worker, param.cpuThread, WORKER_PHASE1_NOTIFY_ID)));
        CHK_RET(LaunchModeKernel(
            worker, resCtx, crossKernel, taskArgs, TWELVE_MODE_ARG_INDEX, TWELVE_CROSS_WAVE_BC_MODE));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(worker, param.cpuThread, WORKER_NOTIFY_ID)));

        CHK_RET(LaunchModeKernel(
            param.cpuThread, resCtx, crossKernel, taskArgs, TWELVE_MODE_ARG_INDEX, TWELVE_REMOTE_SCATTER_D_MODE));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(param.cpuThread, worker, WORKER_PHASE0_NOTIFY_ID)));
        CHK_RET(LaunchModeKernel(
            param.cpuThread, resCtx, localKernel, taskArgs, TWELVE_MODE_ARG_INDEX, TWELVE_REMOTE_ALLGATHER_D_MODE));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(param.cpuThread, WORKER_PHASE1_NOTIFY_ID, CUSTOM_TIMEOUT)));
        CHK_RET(LaunchModeKernel(
            param.cpuThread, resCtx, localKernel, taskArgs, TWELVE_MODE_ARG_INDEX, TWELVE_REMOTE_ALLGATHER_C_MODE));
        return static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThread(param.cpuThread, WORKER_NOTIFY_ID, CUSTOM_TIMEOUT));
    }

    HcclResult ExecuteTwelvePipelinePlan(
        const OpParam &param, const AlgResourceCtx &resCtx, uint64_t baseAddr, uint64_t token, uint64_t totalSize)
    {
        CHK_PRT_RET(param.rankSize != 12 || param.root >= 8,
            HCCL_ERROR("Twelve-rank pipeline requires rankSize[12] and root on the 8-rank server"), HCCL_E_PARA);
        CHK_PRT_RET(resCtx.ccuKernels.size() != 2 || resCtx.kernelLayers.size() != 2
                        || resCtx.kernelLayers[0] == resCtx.kernelLayers[1]
                        || resCtx.kernelLayers[0] > 1 || resCtx.kernelLayers[1] > 1,
            HCCL_ERROR("Twelve-rank pipeline requires exactly two layer kernels"), HCCL_E_INTERNAL);
        const uint64_t dataTypeSize = totalSize / param.count;
        const auto slices = MakeTwelveSlices(param.count, dataTypeSize);
        CHK_RET(ValidateTwelveSlices(slices, totalSize, dataTypeSize));

        std::array<uint64_t, TWELVE_TASK_ARG_NUM> taskArgs{};
        taskArgs[0] = baseAddr;
        taskArgs[1] = token;
        std::copy(slices.offsets.begin(), slices.offsets.end(), taskArgs.begin() + 2);
        std::copy(slices.sizes.begin(), slices.sizes.end(), taskArgs.begin() + 2 + TWELVE_SLICE_NUM);

        uint32_t localKernel = 0;
        uint32_t crossKernel = 0;
        CHK_RET(FindLayerKernel(resCtx, 0, localKernel));
        CHK_RET(FindLayerKernel(resCtx, 1, crossKernel));

        if (param.myRank < 8) {
            return LaunchTwelveLocalPipeline(param, resCtx, localKernel, crossKernel, taskArgs);
        }
        return LaunchTwelveRemotePipeline(param, resCtx, localKernel, crossKernel, taskArgs);
    }

    HcclResult ExecuteLayerParallelPlan(const OpParam &param, const BroadcastPlan &plan, const AlgResourceCtx &resCtx,
        uint64_t baseAddr, uint64_t token, uint64_t totalSize)
    {
        CHK_PRT_RET(plan.chunkSize == 0, HCCL_ERROR("Invalid chunk size[0]"), HCCL_E_PARA);
        uint64_t offset = 0;
        while (offset < totalSize) {
            uint64_t size = std::min<uint64_t>(plan.chunkSize, totalSize - offset);
            CHK_RET(LaunchChunk(param, resCtx, baseAddr, token, offset, size));
            offset += size;
        }
        return HCCL_SUCCESS;
    }

    HcclResult ExecuteSmallFlatPlan(
        const OpParam &param, const SmallResourceCtx &resCtx, uint64_t baseAddr, uint64_t token, uint64_t totalSize)
    {
        CHK_PRT_RET(resCtx.magic != SMALL_RESOURCE_MAGIC || resCtx.version != 4 || resCtx.threadCount == 0
                        || resCtx.threadCount > resCtx.threads.size() || resCtx.kernelCount == 0
                        || resCtx.kernelCount > resCtx.kernels.size(),
            HCCL_ERROR("Invalid Small fixed resource context"), HCCL_E_INTERNAL);

        // 接收端只加载地址和 token；root 额外加载数据长度。
        const std::array<uint64_t, 3> taskArgs = {baseAddr, token, totalSize};
        const uint32_t argNum = param.myRank == param.root ? 3 : 2;

        const uint32_t invalidKernel = resCtx.kernelCount;
        const uint32_t layer0Kernel = resCtx.layerKernels[0];
        const uint32_t layer1Kernel = resCtx.layerKernels[1];

        const bool hasLayer0 = layer0Kernel < invalidKernel;
        const bool hasLayer1 = layer1Kernel < invalidKernel;
        CHK_PRT_RET(
            !hasLayer0 && !hasLayer1, HCCL_ERROR("No layer 0 or layer 1 small-flat kernel found"), HCCL_E_INTERNAL);
        auto launchSmallKernel = [&](ThreadHandle thread, uint32_t kernelIndex) -> HcclResult {
            CcuResult ccuRet = HcommCcuKernelLaunch(thread, resCtx.kernels[kernelIndex], taskArgs.data(), argNum);
            CHK_PRT_RET(ccuRet != CCU_SUCCESS,
                HCCL_ERROR("Small flat layer[%u] kernel launch failed[%d]", resCtx.kernelLayers[kernelIndex], ccuRet),
                ConvertCcuToHccl(ccuRet));
            return HCCL_SUCCESS;
        };
        if (!hasLayer0 || !hasLayer1) {
            return launchSmallKernel(param.cpuThread, hasLayer0 ? layer0Kernel : layer1Kernel);
        }

        CHK_PRT_RET(resCtx.threadCount < 2,
            HCCL_ERROR("Small flat layer-parallel launch requires two CCU threads"), HCCL_E_INTERNAL);
        ThreadHandle layer1Worker = resCtx.threads[1];
        CHK_RET(
            static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(param.cpuThread, layer1Worker, WORKER_NOTIFY_ID)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(layer1Worker, WORKER_NOTIFY_ID, CUSTOM_TIMEOUT)));
        CHK_RET(launchSmallKernel(layer1Worker, layer1Kernel));
        CHK_RET(
            static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(layer1Worker, param.cpuThread, WORKER_NOTIFY_ID)));
        CHK_RET(launchSmallKernel(param.cpuThread, layer0Kernel));
        CHK_RET(
            static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(param.cpuThread, WORKER_NOTIFY_ID, CUSTOM_TIMEOUT)));
        return HCCL_SUCCESS;
    }

    struct SixteenSlices {
        std::array<uint64_t, SIXTEEN_SLICE_NUM> offsets{};
        std::array<uint64_t, SIXTEEN_SLICE_NUM> sizes{};
    };

    SixteenSlices MakeSixteenSlices(uint64_t count, uint64_t dataTypeSize)
    {
        SixteenSlices slices;
        const uint64_t baseCount = count / SIXTEEN_SLICE_NUM;
        const uint64_t remainder = count % SIXTEEN_SLICE_NUM;
        uint64_t offset = 0;
        for (uint32_t index = 0; index < SIXTEEN_SLICE_NUM; ++index) {
            const uint64_t sliceCount = baseCount + static_cast<uint64_t>(index < remainder);
            slices.offsets[index] = offset;
            slices.sizes[index] = sliceCount * dataTypeSize;
            offset += slices.sizes[index];
        }
        return slices;
    }

    HcclResult FindSixteenKernel(const AlgResourceCtx &resCtx, uint32_t marker, uint32_t &kernelIndex)
    {
        auto markerIt = std::find(resCtx.kernelLayers.begin(), resCtx.kernelLayers.end(), marker);
        CHK_PRT_RET(markerIt == resCtx.kernelLayers.end(),
            HCCL_ERROR("Sixteen-rank kernel marker[%u] not found", marker), HCCL_E_INTERNAL);
        kernelIndex = static_cast<uint32_t>(markerIt - resCtx.kernelLayers.begin());
        return HCCL_SUCCESS;
    }

    std::array<uint64_t, SIXTEEN_TASK_ARG_NUM> MakeSixteenTaskArgs(
        uint64_t baseAddr, uint64_t token, const SixteenSlices &slices)
    {
        std::array<uint64_t, SIXTEEN_TASK_ARG_NUM> taskArgs{};
        taskArgs[0] = baseAddr;
        taskArgs[1] = token;
        std::copy(slices.offsets.begin(), slices.offsets.end(), taskArgs.begin() + 2);
        std::copy(slices.sizes.begin(), slices.sizes.end(), taskArgs.begin() + 2 + SIXTEEN_SLICE_NUM);
        return taskArgs;
    }

    HcclResult ExecuteSixteenFusedScatter(const OpParam &param, const AlgResourceCtx &resCtx,
        const std::array<uint64_t, SIXTEEN_TASK_ARG_NUM> &taskArgs)
    {
        CHK_PRT_RET(resCtx.threads.size() != 2,
            HCCL_ERROR("Sixteen-rank fused scatter requires two CCU threads, got[%zu]", resCtx.threads.size()),
            HCCL_E_INTERNAL);
        const uint32_t argNum = param.myRank == param.root ? SIXTEEN_TASK_ARG_NUM : 2U;
        const uint32_t invalidKernel = static_cast<uint32_t>(resCtx.ccuKernels.size());
        uint32_t layer0Kernel = invalidKernel;
        uint32_t layer1Kernel = invalidKernel;
        for (uint32_t index = 0; index < resCtx.ccuKernels.size(); ++index) {
            if (resCtx.kernelLayers[index] == 0) {
                layer0Kernel = index;
            } else if (resCtx.kernelLayers[index] == 1) {
                layer1Kernel = index;
            }
        }
        const bool hasLayer0 = layer0Kernel < invalidKernel;
        const bool hasLayer1 = layer1Kernel < invalidKernel;
        CHK_PRT_RET(!hasLayer0 && !hasLayer1, HCCL_ERROR("No sixteen-rank Scatter kernel found"), HCCL_E_INTERNAL);
        if (!hasLayer0) {
            return LaunchKernel(param.cpuThread, resCtx, layer1Kernel, taskArgs.data(), argNum);
        }
        if (!hasLayer1) {
            return LaunchKernel(param.cpuThread, resCtx, layer0Kernel, taskArgs.data(), argNum);
        }

        ThreadHandle layer1Worker = resCtx.threads[1];
        CHK_RET(
            static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(param.cpuThread, layer1Worker, WORKER_NOTIFY_ID)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(layer1Worker, WORKER_NOTIFY_ID, CUSTOM_TIMEOUT)));
        CHK_RET(LaunchKernel(layer1Worker, resCtx, layer1Kernel, taskArgs.data(), argNum));
        CHK_RET(
            static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(layer1Worker, param.cpuThread, WORKER_NOTIFY_ID)));
        CHK_RET(LaunchKernel(param.cpuThread, resCtx, layer0Kernel, taskArgs.data(), argNum));
        CHK_RET(
            static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(param.cpuThread, WORKER_NOTIFY_ID, CUSTOM_TIMEOUT)));
        return HCCL_SUCCESS;
    }

    HcclResult ExecuteSixteenFusedAllGather(const OpParam &param, const AlgResourceCtx &resCtx,
        uint64_t baseAddr, uint64_t token, const SixteenSlices &slices)
    {
        CHK_PRT_RET(resCtx.threads.size() != 2,
            HCCL_ERROR("Sixteen-rank flat AllGather requires two CCU threads, got[%zu]", resCtx.threads.size()),
            HCCL_E_INTERNAL);
        uint32_t localKernel = 0;
        uint32_t crossKernel = 0;
        CHK_RET(FindSixteenKernel(resCtx, SIXTEEN_LOCAL_ALLGATHER_MARKER, localKernel));
        CHK_RET(FindSixteenKernel(resCtx, SIXTEEN_CROSS_ALLGATHER_MARKER, crossKernel));
        const std::array<uint64_t, SIXTEEN_AG_TASK_ARG_NUM> taskArgs
            = {baseAddr, token, slices.offsets[param.myRank], slices.sizes[param.myRank]};
        ThreadHandle crossWorker = resCtx.threads[1];
        CHK_RET(
            static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(param.cpuThread, crossWorker, WORKER_NOTIFY_ID)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(crossWorker, WORKER_NOTIFY_ID, CUSTOM_TIMEOUT)));
        CHK_RET(LaunchKernel(crossWorker, resCtx, crossKernel, taskArgs));
        CHK_RET(
            static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(crossWorker, param.cpuThread, WORKER_NOTIFY_ID)));
        CHK_RET(LaunchKernel(param.cpuThread, resCtx, localKernel, taskArgs));
        CHK_RET(
            static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(param.cpuThread, WORKER_NOTIFY_ID, CUSTOM_TIMEOUT)));
        return HCCL_SUCCESS;
    }

    HcclResult ExecuteSixteenScatterDoublingPlan(
        const OpParam &param, const AlgResourceCtx &resCtx, uint64_t baseAddr, uint64_t token, uint64_t totalSize)
    {
        CHK_PRT_RET(
            param.rankSize != 16, HCCL_ERROR("Sixteen-rank scatter-doubling requires rankSize[16]"), HCCL_E_PARA);
        const uint64_t dataTypeSize = totalSize / param.count;
        const SixteenSlices slices = MakeSixteenSlices(param.count, dataTypeSize);
        for (uint32_t index = 0; index < SIXTEEN_SLICE_NUM; ++index) {
            CHK_PRT_RET(slices.sizes[index] == 0 || slices.sizes[index] > MAX_DATA_SIZE,
                HCCL_ERROR("Invalid sixteen-rank slice[%u] size[%lu]", index, slices.sizes[index]), HCCL_E_NOT_SUPPORT);
        }
        const auto taskArgs = MakeSixteenTaskArgs(baseAddr, token, slices);
        CHK_RET(ExecuteSixteenFusedScatter(param, resCtx, taskArgs));
        return ExecuteSixteenFusedAllGather(param, resCtx, baseAddr, token, slices);
    }

    HcclResult ExecutePlan(const OpParam &param, const BroadcastPlan &plan, const AlgResourceCtx &resCtx,
        uint64_t baseAddr, uint64_t token, uint64_t totalSize)
    {
        switch (plan.kernelPlan) {
            case KernelPlanType::LAYER_PARALLEL:
                return ExecuteLayerParallelPlan(param, plan, resCtx, baseAddr, token, totalSize);
            case KernelPlanType::FOUR_RANK_TREE:
                return ExecuteFourRankTreePlan(param, plan, resCtx, baseAddr, token, totalSize);
            case KernelPlanType::FOUR_RANK_PIPELINE:
                return ExecuteFourRankPipelinePlan(param, resCtx, baseAddr, token, totalSize);
            case KernelPlanType::SMALL_FLAT:
                HCCL_ERROR("Small plan must use the fixed-context fast path");
                return HCCL_E_INTERNAL;
            case KernelPlanType::TWELVE_PIPELINE:
                return ExecuteTwelvePipelinePlan(param, resCtx, baseAddr, token, totalSize);
            case KernelPlanType::SIXTEEN_SCATTER_DOUBLING:
                return ExecuteSixteenScatterDoublingPlan(param, resCtx, baseAddr, token, totalSize);
            default:
                HCCL_ERROR("Unsupported kernel plan[%u]", static_cast<uint32_t>(plan.kernelPlan));
                return HCCL_E_NOT_SUPPORT;
        }
    }
} // namespace

HcclResult ExecOp(const OpParam &param)
{
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(param.root >= param.rankSize,
        HCCL_ERROR("Invalid root[%u] for rankSize[%u]", param.root, param.rankSize), HCCL_E_PARA);
    CHK_PRT_RET(param.dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("Unsupported data type[%d]", static_cast<int>(param.dataType)), HCCL_E_PARA);
    constexpr uint64_t dataTypeSize = sizeof(float);
    CHK_PRT_RET(param.count > ~static_cast<uint64_t>(0) / dataTypeSize, HCCL_ERROR("Broadcast data size overflows"),
        HCCL_E_PARA);
    const uint64_t totalSize = param.count * dataTypeSize;
    BroadcastPlan plan = SelectBroadcastPlan(param.rankSize, totalSize, param.root);
    if (param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    uint64_t token = 0;
    const uint64_t baseAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    HcommCcuGetMemToken(baseAddr, totalSize, &token);

    if (plan.messageSize == MessageSizeClass::SMALL) {
        CHK_PRT_RET(param.resCtx == nullptr || param.ctxSize != sizeof(SmallResourceCtx),
            HCCL_ERROR("Invalid Small context size[%lu]", param.ctxSize), HCCL_E_INTERNAL);
        const auto &smallCtx = *static_cast<const SmallResourceCtx *>(param.resCtx);
        return ExecuteSmallFlatPlan(param, smallCtx, baseAddr, token, totalSize);
    }

    CHK_PRT_RET(param.resCtx == nullptr || param.ctxSize != sizeof(AlgResourceCtx),
        HCCL_ERROR("Invalid Large context size[%lu]", param.ctxSize), HCCL_E_INTERNAL);
    AlgResourceCtx resCtx;
    std::memcpy(&resCtx, param.resCtx, sizeof(resCtx));
    CHK_PRT_RET(resCtx.magic != LARGE_RESOURCE_MAGIC || resCtx.version != LARGE_RESOURCE_VERSION
                    || resCtx.threads.empty() || resCtx.threads.size() > resCtx.threads.capacity()
                    || resCtx.ccuKernels.empty() || resCtx.ccuKernels.size() > resCtx.ccuKernels.capacity()
                    || resCtx.kernelLayers.size() != resCtx.ccuKernels.size()
                    || resCtx.kernelPeers.size() > resCtx.kernelPeers.capacity(),
        HCCL_ERROR("Invalid CCU kernel resources"), HCCL_E_INTERNAL);
    CHK_PRT_RET(
        plan.kernelPlan == KernelPlanType::FOUR_RANK_TREE && resCtx.ccuKernels.size() != resCtx.kernelPeers.size(),
        HCCL_ERROR("Invalid pair kernel resources"), HCCL_E_INTERNAL);

    return ExecutePlan(param, plan, resCtx, baseAddr, token, totalSize);
}
} // namespace ops_hccl
