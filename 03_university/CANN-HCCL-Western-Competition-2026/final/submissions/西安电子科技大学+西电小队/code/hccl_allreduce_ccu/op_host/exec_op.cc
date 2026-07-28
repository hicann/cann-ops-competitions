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
#include <hcomm/hcomm_primitives.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {

    constexpr uint64_t FP32_BYTES = sizeof(float);
    constexpr uint32_t BASE_TASK_ARG_NUM = 10;
    constexpr uint32_t GROUP_TASK_ARG_NUM = 14;
    constexpr uint32_t RUNTIME_PHASE_ARG_INDEX = 9;
    constexpr uint64_t GROUP_REDUCE_INTERLEAVE = 8;
    constexpr uint64_t GROUP_REDUCE_LOOP_COUNT = 16;
    constexpr uint64_t GROUP_REDUCE_SLICE_BYTES = 4096;
    constexpr uint64_t GROUP_REDUCE_MAX_TOTAL_BYTES = 512U * 1024U;
    constexpr uint32_t DUAL_GROUP_PUSH_MIN_ELEMENTS_PER_RANK = 2U;
    constexpr uint32_t ISOLATED_DUAL_GROUP_RESOURCE_FLAG = 1U << 31U;
    constexpr uint32_t ISOLATED_EXACT_P14_NHR_RESOURCE_FLAG = 1U << 30U;
    constexpr uint32_t ISOLATED_EXACT_P13_COMPACT_RESOURCE_FLAG = 1U << 29U;
    constexpr uint32_t ISOLATED_EXACT_P15_NHR_RESOURCE_FLAG = 1U << 28U;
    constexpr uint32_t ISOLATED_EXACT_P17_TREE_PUSH_RESOURCE_FLAG = 1U << 27U;
    constexpr uint32_t ISOLATED_EXACT_P16_COMPACT_PRESYNC_RESOURCE_FLAG = 1U << 26U;
    constexpr uint32_t ISOLATED_EXACT_P18_TREE_RESOURCE_FLAG = 1U << 25U;
    constexpr uint32_t ISOLATED_EXACT_P11_TREE_PUSH_RESOURCE_FLAG = 1U << 24U;
    constexpr uint32_t ISOLATED_EXACT_P12_TREE_PUSH_RESOURCE_FLAG = 1U << 23U;
    constexpr uint64_t TWO_BY_EIGHT_EXACT_P11_TREE_PUSH_BYTES = 512ULL * 1024ULL * 1024ULL;
    constexpr uint64_t TWO_BY_EIGHT_EXACT_P12_TREE_PUSH_BYTES = 400ULL * 1024ULL * 1024ULL + FP32_BYTES;
    constexpr uint64_t EIGHT_PLUS_FOUR_EXACT_P16_COMPACT_PRESYNC_BYTES = 512U * 1024U;
    constexpr uint64_t EIGHT_PLUS_FOUR_EXACT_P17_TREE_PUSH_BYTES = 512ULL * 1024ULL * 1024ULL;
    constexpr uint64_t EIGHT_PLUS_FOUR_EXACT_P18_TREE_BYTES = 400ULL * 1024ULL * 1024ULL + FP32_BYTES;
    constexpr uint64_t FOUR_BY_ONE_EXACT_P14_NHR_BYTES = 512ULL * 1024ULL * 1024ULL;
    constexpr uint64_t FOUR_BY_ONE_EXACT_P13_COMPACT_BYTES = 512ULL * 1024ULL;
    constexpr uint64_t FOUR_BY_ONE_EXACT_P15_NHR_BYTES = 400ULL * 1024ULL * 1024ULL + FP32_BYTES;

    constexpr uint64_t SetBits(uint16_t end)
    {
        return (uint64_t{1} << (end + 1U)) - uint64_t{1};
    }

    constexpr uint64_t GetMaxLoopIterNum()
    {
        return SetBits(12);
    }

    constexpr uint64_t GetParallelParam(uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum)
    {
        return ((repeatNum & SetBits(7)) << 55U) | ((repeatLoopIndex & SetBits(7)) << 48U)
               | ((totalLoopNum & SetBits(7)) << 41U);
    }

    std::vector<uint64_t> CalGroupReduceSize(uint64_t size)
    {
        const uint64_t loopSize = GROUP_REDUCE_LOOP_COUNT * GROUP_REDUCE_SLICE_BYTES;
        const uint64_t maxSize = loopSize * (GetMaxLoopIterNum() + 1U);
        uint64_t loopIterNum = size / loopSize;
        uint64_t parallelSliceCount = (size - loopIterNum * loopSize) / GROUP_REDUCE_SLICE_BYTES;
        uint64_t residual = size - loopIterNum * loopSize - parallelSliceCount * GROUP_REDUCE_SLICE_BYTES;

        if (size == maxSize) {
            loopIterNum = GetMaxLoopIterNum();
            parallelSliceCount = GROUP_REDUCE_LOOP_COUNT - 1U;
            residual = GROUP_REDUCE_SLICE_BYTES;
        }

        const uint64_t addrOffset = loopSize * loopIterNum;
        uint64_t parallelParam = 0;
        uint64_t tailSize = 0;
        if (parallelSliceCount != 0 && residual == 0) {
            parallelParam = GetParallelParam(parallelSliceCount - 1U, 0U, 1U);
            tailSize = GROUP_REDUCE_SLICE_BYTES;
        } else if (parallelSliceCount == 0 && residual != 0) {
            parallelParam = GetParallelParam(0U, 0U, 1U);
            tailSize = residual;
        } else if (parallelSliceCount != 0 && residual != 0) {
            parallelParam = GetParallelParam(parallelSliceCount - 1U, 1U, 2U);
            tailSize = residual;
        }
        return {addrOffset, loopIterNum, parallelParam, tailSize};
    }

    HcclResult GetMemToken(uint64_t address, uint64_t size, uint64_t &token)
    {
        CcuResult ret = HcommCcuGetMemToken(address, size, &token);
        if (ret != CCU_SUCCESS) {
            HCCL_ERROR("[CCU_V001] HcommCcuGetMemToken failed, ret[%d]", ret);
            return ConvertCcuResult(ret);
        }
        return HCCL_SUCCESS;
    }

    bool ScratchFits(uint64_t windowElements, uint32_t rankSize, uint64_t cclBufferSize, uint64_t &normalElements,
        uint64_t &lastElements)
    {
        normalElements = windowElements / rankSize;
        lastElements = windowElements - normalElements * (rankSize - 1U);
        if (lastElements > MAX_DATA_SIZE / FP32_BYTES) {
            return false;
        }
        if (lastElements != 0 && rankSize - 1U > cclBufferSize / (lastElements * FP32_BYTES)) {
            return false;
        }
        return (rankSize - 1U) * lastElements * FP32_BYTES <= cclBufferSize;
    }

    HcclResult SelectWindow(uint64_t remainingElements, uint32_t rankSize, uint64_t cclBufferSize,
        uint64_t &windowElements, uint64_t &normalElements, uint64_t &lastElements)
    {
        CHK_PRT_RET(rankSize <= 1U, HCCL_ERROR("[CCU_V001] invalid rank size[%u]", rankSize), HCCL_E_PARA);
        const uint64_t maxSliceElementsByScratch = cclBufferSize / (rankSize - 1U) / FP32_BYTES;
        const uint64_t maxSliceElements = std::min<uint64_t>(maxSliceElementsByScratch, MAX_DATA_SIZE / FP32_BYTES);
        CHK_PRT_RET(maxSliceElements == 0, HCCL_ERROR("[CCU_V001] HCCL buffer is too small"), HCCL_E_MEMORY);

        if (maxSliceElements > std::numeric_limits<uint64_t>::max() / rankSize) {
            windowElements = remainingElements;
        } else {
            windowElements = std::min<uint64_t>(remainingElements, maxSliceElements * rankSize);
        }

        while (
            windowElements > 0 && !ScratchFits(windowElements, rankSize, cclBufferSize, normalElements, lastElements)) {
            --windowElements;
        }
        CHK_PRT_RET(windowElements == 0,
            HCCL_ERROR("[CCU_V001] no valid window for remaining elements[%llu]",
                static_cast<unsigned long long>(remainingElements)),
            HCCL_E_MEMORY);
        return HCCL_SUCCESS;
    }

    HcclResult LaunchWindow(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t inputAddress,
        uint64_t outputAddress, uint64_t inputToken, uint64_t outputToken, uint64_t scratchToken,
        uint64_t normalSliceBytes, uint64_t lastSliceBytes, bool useSingleDieGroupReduce)
    {
        const uint64_t inPlace = (param.inputPtr == param.outputPtr) ? 1U : 0U;
        const uint64_t localSliceBytes = (param.myRank == param.rankSize - 1U) ? lastSliceBytes : normalSliceBytes;

        std::vector<uint64_t> taskArgs = {
            inputAddress,
            outputAddress,
            inputToken,
            outputToken,
            reinterpret_cast<uint64_t>(resCtx.localBuffer.addr),
            scratchToken,
            normalSliceBytes,
            lastSliceBytes,
            inPlace,
            0U,
        };
        if (resCtx.activeKernelCount == 1U && useSingleDieGroupReduce) {
            const std::vector<uint64_t> groupReduceSize = CalGroupReduceSize(localSliceBytes);
            taskArgs.insert(taskArgs.end(), groupReduceSize.begin(), groupReduceSize.end());
        }

        const uint32_t expectedTaskArgNum
            = (resCtx.activeKernelCount == 1U && useSingleDieGroupReduce) ? GROUP_TASK_ARG_NUM : BASE_TASK_ARG_NUM;
        CHK_PRT_RET(taskArgs.size() != expectedTaskArgNum,
            HCCL_ERROR("[CCU_V003] invalid task arg count[%zu]", taskArgs.size()), HCCL_E_INTERNAL);

        const uint32_t expectedKernelCount = (resCtx.activeKernelCount == 1U) ? 2U : 3U;
        CHK_PRT_RET(resCtx.ccuKernels.size() < expectedKernelCount,
            HCCL_ERROR("[CCU_V001] incomplete phase kernels, expected[%u], actual[%zu]", expectedKernelCount,
                resCtx.ccuKernels.size()),
            HCCL_E_INTERNAL);

        auto launchKernel = [&](ThreadHandle thread, uint32_t handleIndex) -> HcclResult {
            const CcuResult ret = HcommCcuKernelLaunch(
                thread, resCtx.ccuKernels[handleIndex], taskArgs.data(), static_cast<uint32_t>(taskArgs.size()));
            if (ret != CCU_SUCCESS) {
                HCCL_ERROR("[CCU_V001] phase kernel[%u] launch failed, ret[%d]", handleIndex, ret);
                return ConvertCcuResult(ret);
            }
            return HCCL_SUCCESS;
        };

        auto launchDiePair = [&](uint64_t runtimePhase) -> HcclResult {
            taskArgs[RUNTIME_PHASE_ARG_INDEX] = runtimePhase;
            if (resCtx.activeKernelCount == 2U) {
                CHK_RET(
                    static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(resCtx.threads[0], resCtx.threads[1], 0U)));
                CHK_RET(
                    static_cast<HcclResult>(HcommThreadNotifyWaitOnThreadWithDefaultTimeout(resCtx.threads[1], 0U)));
            }

            for (uint32_t ordinal = 0; ordinal < resCtx.activeKernelCount; ++ordinal) {
                CHK_RET(launchKernel(resCtx.threads[ordinal], ordinal));
            }

            if (resCtx.activeKernelCount == 2U) {
                CHK_RET(
                    static_cast<HcclResult>(HcommThreadNotifyWaitOnThreadWithDefaultTimeout(resCtx.threads[0], 0U)));
                CHK_RET(
                    static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(resCtx.threads[1], resCtx.threads[0], 0U)));
            }
            return HCCL_SUCCESS;
        };

        if (resCtx.activeKernelCount == 1U) {
            const uint32_t handleIndex = useSingleDieGroupReduce ? 0U : 1U;
            return launchKernel(resCtx.threads[0], handleIndex);
        }

        CHK_RET(launchDiePair(0U));
        CHK_RET(launchKernel(resCtx.threads[0], resCtx.activeKernelCount));
        return launchDiePair(1U);
    }

    HcclResult LaunchDualGroupPushWindow(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t inputAddress,
        uint64_t outputAddress, uint64_t inputToken, uint64_t outputToken, uint64_t scratchToken,
        uint64_t normalSliceBytes, uint64_t lastSliceBytes)
    {
        const bool supportedTopology = resCtx.topologyKind == TopologyKind::TOPO_2X8
                                       || resCtx.topologyKind == TopologyKind::TOPO_8_PLUS_4;
        CHK_PRT_RET(param.inputPtr == param.outputPtr || !supportedTopology || resCtx.activeKernelCount != 2U,
            HCCL_ERROR("[CCU_V028] invalid isolated dual-group launch"), HCCL_E_INTERNAL);
        CHK_PRT_RET(resCtx.threads.size() < 2U || resCtx.ccuKernels.size() < 3U,
            HCCL_ERROR("[CCU_V028] incomplete isolated dual-group resources"), HCCL_E_INTERNAL);

        const uint64_t localSliceBytes
            = (param.myRank == param.rankSize - 1U) ? lastSliceBytes : normalSliceBytes;
        std::vector<uint64_t> taskArgs = {
            inputAddress,
            outputAddress,
            inputToken,
            outputToken,
            reinterpret_cast<uint64_t>(resCtx.localBuffer.addr),
            scratchToken,
            normalSliceBytes,
            lastSliceBytes,
            0U,
            0U,
        };
        const std::vector<uint64_t> groupReduceSize = CalGroupReduceSize(localSliceBytes);
        taskArgs.insert(taskArgs.end(), groupReduceSize.begin(), groupReduceSize.end());
        CHK_PRT_RET(taskArgs.size() != GROUP_TASK_ARG_NUM,
            HCCL_ERROR("[CCU_V028] invalid isolated task arg count[%zu]", taskArgs.size()), HCCL_E_INTERNAL);

        auto launchKernel = [&](ThreadHandle thread, uint32_t handleIndex, uint32_t argCount) -> HcclResult {
            const CcuResult ret
                = HcommCcuKernelLaunch(thread, resCtx.ccuKernels[handleIndex], taskArgs.data(), argCount);
            if (ret != CCU_SUCCESS) {
                HCCL_ERROR("[CCU_V028] isolated kernel[%u] launch failed, ret[%d]", handleIndex, ret);
                return ConvertCcuResult(ret);
            }
            return HCCL_SUCCESS;
        };

        auto launchDiePair = [&](uint64_t runtimePhase) -> HcclResult {
            taskArgs[RUNTIME_PHASE_ARG_INDEX] = runtimePhase;
            CHK_RET(static_cast<HcclResult>(
                HcommThreadNotifyRecordOnThread(resCtx.threads[0], resCtx.threads[1], 0U)));
            CHK_RET(static_cast<HcclResult>(
                HcommThreadNotifyWaitOnThreadWithDefaultTimeout(resCtx.threads[1], 0U)));

            CHK_RET(launchKernel(resCtx.threads[0], 0U, GROUP_TASK_ARG_NUM));
            CHK_RET(launchKernel(resCtx.threads[1], 1U, GROUP_TASK_ARG_NUM));

            CHK_RET(static_cast<HcclResult>(
                HcommThreadNotifyWaitOnThreadWithDefaultTimeout(resCtx.threads[0], 0U)));
            CHK_RET(static_cast<HcclResult>(
                HcommThreadNotifyRecordOnThread(resCtx.threads[1], resCtx.threads[0], 0U)));
            return HCCL_SUCCESS;
        };

        CHK_RET(launchDiePair(0U));
        // The merge entry calls LoadBaseArgs only, so its launch ABI is always
        // the ten base arguments.  This matches the online-proven V026 path;
        // passing the four GroupReduce-only arguments can be rejected by a
        // strict CCU runtime before phase 1 is released.
        CHK_RET(launchKernel(resCtx.threads[0], 2U, BASE_TASK_ARG_NUM));
        return launchDiePair(1U);
    }

    HcclResult LaunchExactP14NhrWindow(const OpParam &param, const AlgResourceCtx &resCtx, uint64_t inputAddress,
        uint64_t outputAddress, uint64_t inputToken, uint64_t outputToken, uint64_t scratchToken,
        uint64_t normalSliceBytes, uint64_t lastSliceBytes)
    {
        CHK_PRT_RET(param.rankSize != 4U || param.inputPtr == param.outputPtr || resCtx.activeKernelCount != 1U,
            HCCL_ERROR("[CCU_V042] invalid exact-P14 NHR launch"), HCCL_E_INTERNAL);
        CHK_PRT_RET(resCtx.threads.empty() || resCtx.ccuKernels.size() != 1U,
            HCCL_ERROR("[CCU_V042] incomplete exact-P14 NHR resources"), HCCL_E_INTERNAL);

        std::vector<uint64_t> taskArgs = {
            inputAddress,
            outputAddress,
            inputToken,
            outputToken,
            reinterpret_cast<uint64_t>(resCtx.localBuffer.addr),
            scratchToken,
            normalSliceBytes,
            lastSliceBytes,
            0U,
            0U,
        };
        const CcuResult ret = HcommCcuKernelLaunch(
            resCtx.threads[0], resCtx.ccuKernels[0], taskArgs.data(), BASE_TASK_ARG_NUM);
        if (ret != CCU_SUCCESS) {
            HCCL_ERROR("[CCU_V042] exact-P14 NHR kernel launch failed, ret[%d]", ret);
            return ConvertCcuResult(ret);
        }
        return HCCL_SUCCESS;
    }

    HcclResult LaunchExactP13CompactWindowV051(const OpParam &param, const AlgResourceCtx &resCtx,
        uint64_t inputAddress, uint64_t outputAddress, uint64_t inputToken, uint64_t outputToken,
        uint64_t scratchToken, uint64_t normalSliceBytes, uint64_t lastSliceBytes)
    {
        CHK_PRT_RET(param.rankSize != 4U || param.inputPtr == param.outputPtr || resCtx.activeKernelCount != 1U,
            HCCL_ERROR("[CCU_V051] invalid exact-P13 compact launch"), HCCL_E_INTERNAL);
        CHK_PRT_RET(resCtx.threads.empty() || resCtx.ccuKernels.size() != 1U,
            HCCL_ERROR("[CCU_V051] incomplete exact-P13 compact resources"), HCCL_E_INTERNAL);

        const uint64_t localSliceBytes
            = (param.myRank == param.rankSize - 1U) ? lastSliceBytes : normalSliceBytes;
        std::vector<uint64_t> taskArgs = {
            inputAddress,
            outputAddress,
            inputToken,
            outputToken,
            reinterpret_cast<uint64_t>(resCtx.localBuffer.addr),
            scratchToken,
            normalSliceBytes,
            lastSliceBytes,
            0U,
            0U,
        };
        const std::vector<uint64_t> groupReduceSize = CalGroupReduceSize(localSliceBytes);
        taskArgs.insert(taskArgs.end(), groupReduceSize.begin(), groupReduceSize.end());
        CHK_PRT_RET(taskArgs.size() != GROUP_TASK_ARG_NUM,
            HCCL_ERROR("[CCU_V051] invalid exact-P13 compact task arg count[%zu]", taskArgs.size()), HCCL_E_INTERNAL);

        const CcuResult ret
            = HcommCcuKernelLaunch(resCtx.threads[0], resCtx.ccuKernels[0], taskArgs.data(), GROUP_TASK_ARG_NUM);
        if (ret != CCU_SUCCESS) {
            HCCL_ERROR("[CCU_V051] exact-P13 compact kernel launch failed, ret[%d]", ret);
            return ConvertCcuResult(ret);
        }
        return HCCL_SUCCESS;
    }

} // namespace

HcclResult ExecOp(const OpParam &param)
{
    CHK_PTR_NULL(param.resCtx);
    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> seq(ctx, ctx + param.ctxSize);
    AlgResourceCtx resCtx;
    resCtx.DeSerialize(seq);

    const bool isolatedDualGroupRegistered
        = (resCtx.activeKernelCount & ISOLATED_DUAL_GROUP_RESOURCE_FLAG) != 0U;
    const bool isolatedExactP14NhrRegistered
        = (resCtx.activeKernelCount & ISOLATED_EXACT_P14_NHR_RESOURCE_FLAG) != 0U;
    const bool isolatedExactP13CompactRegistered
        = (resCtx.activeKernelCount & ISOLATED_EXACT_P13_COMPACT_RESOURCE_FLAG) != 0U;
    const bool isolatedExactP15NhrRegistered
        = (resCtx.activeKernelCount & ISOLATED_EXACT_P15_NHR_RESOURCE_FLAG) != 0U;
    const bool isolatedExactP17TreePushRegistered
        = (resCtx.activeKernelCount & ISOLATED_EXACT_P17_TREE_PUSH_RESOURCE_FLAG) != 0U;
    const bool isolatedExactP16CompactPreSyncRegistered
        = (resCtx.activeKernelCount & ISOLATED_EXACT_P16_COMPACT_PRESYNC_RESOURCE_FLAG) != 0U;
    const bool isolatedExactP18TreeRegistered
        = (resCtx.activeKernelCount & ISOLATED_EXACT_P18_TREE_RESOURCE_FLAG) != 0U;
    const bool isolatedExactP11TreePushRegistered
        = (resCtx.activeKernelCount & ISOLATED_EXACT_P11_TREE_PUSH_RESOURCE_FLAG) != 0U;
    const bool isolatedExactP12TreePushRegistered
        = (resCtx.activeKernelCount & ISOLATED_EXACT_P12_TREE_PUSH_RESOURCE_FLAG) != 0U;
    resCtx.activeKernelCount
        &= ~(ISOLATED_DUAL_GROUP_RESOURCE_FLAG | ISOLATED_EXACT_P14_NHR_RESOURCE_FLAG
            | ISOLATED_EXACT_P13_COMPACT_RESOURCE_FLAG | ISOLATED_EXACT_P15_NHR_RESOURCE_FLAG
            | ISOLATED_EXACT_P17_TREE_PUSH_RESOURCE_FLAG | ISOLATED_EXACT_P16_COMPACT_PRESYNC_RESOURCE_FLAG
            | ISOLATED_EXACT_P18_TREE_RESOURCE_FLAG | ISOLATED_EXACT_P11_TREE_PUSH_RESOURCE_FLAG
            | ISOLATED_EXACT_P12_TREE_PUSH_RESOURCE_FLAG);

    CHK_PRT_RET(resCtx.threads.empty(), HCCL_ERROR("[CCU_V001] no CCU thread"), HCCL_E_INTERNAL);
    resCtx.threads[0] = param.cpuThread;

    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / FP32_BYTES,
        HCCL_ERROR("[CCU_V001] count overflow"), HCCL_E_PARA);
    const uint64_t totalBytes = param.count * FP32_BYTES;
    const bool useSingleDieGroupReduce = totalBytes <= GROUP_REDUCE_MAX_TOTAL_BYTES;
    const bool supportedGroupPushTopology
        = (resCtx.topologyKind == TopologyKind::TOPO_2X8 && param.rankSize == 16U)
          || (resCtx.topologyKind == TopologyKind::TOPO_8_PLUS_4 && param.rankSize == 12U);
    const bool useDualGroupPushSmall = isolatedDualGroupRegistered && supportedGroupPushTopology
                                       && param.inputPtr != param.outputPtr
                                       && param.count >= param.rankSize * DUAL_GROUP_PUSH_MIN_ELEMENTS_PER_RANK
                                       && totalBytes <= GROUP_REDUCE_MAX_TOTAL_BYTES;
    const bool useExactP17TreePush
        = isolatedExactP17TreePushRegistered && resCtx.topologyKind == TopologyKind::TOPO_8_PLUS_4
          && param.rankSize == 12U && param.inputPtr != param.outputPtr
          && totalBytes == EIGHT_PLUS_FOUR_EXACT_P17_TREE_PUSH_BYTES;
    const bool useExactP16CompactPreSync
        = isolatedExactP16CompactPreSyncRegistered && resCtx.topologyKind == TopologyKind::TOPO_8_PLUS_4
          && param.rankSize == 12U && param.inputPtr != param.outputPtr
          && totalBytes == EIGHT_PLUS_FOUR_EXACT_P16_COMPACT_PRESYNC_BYTES;
    const bool useExactP18Tree
        = isolatedExactP18TreeRegistered && resCtx.topologyKind == TopologyKind::TOPO_8_PLUS_4
          && param.rankSize == 12U && param.inputPtr != param.outputPtr
          && totalBytes == EIGHT_PLUS_FOUR_EXACT_P18_TREE_BYTES;
    const bool useExactP11TreePush
        = isolatedExactP11TreePushRegistered && resCtx.topologyKind == TopologyKind::TOPO_2X8
          && param.rankSize == 16U && param.inputPtr != param.outputPtr
          && totalBytes == TWO_BY_EIGHT_EXACT_P11_TREE_PUSH_BYTES;
    const bool useExactP12TreePush
        = isolatedExactP12TreePushRegistered && resCtx.topologyKind == TopologyKind::TOPO_2X8
          && param.rankSize == 16U && param.inputPtr != param.outputPtr
          && totalBytes == TWO_BY_EIGHT_EXACT_P12_TREE_PUSH_BYTES;

    if (param.rankSize == 1U) {
        if (param.inputPtr != param.outputPtr) {
            return static_cast<HcclResult>(
                HcommLocalCopyOnThread(resCtx.threads[0], param.outputPtr, param.inputPtr, totalBytes));
        }
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(resCtx.activeKernelCount == 0 || resCtx.activeKernelCount > 2U,
        HCCL_ERROR("[CCU_V001] invalid active kernel count[%u]", resCtx.activeKernelCount), HCCL_E_INTERNAL);
    CHK_PRT_RET(resCtx.threads.size() < resCtx.activeKernelCount || resCtx.ccuKernels.size() < resCtx.activeKernelCount,
        HCCL_ERROR("[CCU_V001] incomplete thread/kernel resources"), HCCL_E_INTERNAL);
    CHK_PTR_NULL(resCtx.localBuffer.addr);

    const uint64_t inputBase = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outputBase = reinterpret_cast<uint64_t>(param.outputPtr);
    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    uint64_t scratchToken = 0;
    CHK_RET(GetMemToken(inputBase, totalBytes, inputToken));
    if (inputBase == outputBase) {
        outputToken = inputToken;
    } else {
        CHK_RET(GetMemToken(outputBase, totalBytes, outputToken));
    }
    CHK_RET(GetMemToken(reinterpret_cast<uint64_t>(resCtx.localBuffer.addr), resCtx.localBuffer.size, scratchToken));

    if (isolatedExactP14NhrRegistered) {
        CHK_PRT_RET(param.rankSize != 4U || param.inputPtr == param.outputPtr
                        || totalBytes != FOUR_BY_ONE_EXACT_P14_NHR_BYTES,
            HCCL_ERROR("[CCU_V042] invalid exact-P14 NHR invocation"), HCCL_E_PARA);
        const uint64_t normalElements = param.count / param.rankSize;
        const uint64_t lastElements = param.count - normalElements * (param.rankSize - 1U);
        return LaunchExactP14NhrWindow(param, resCtx, inputBase, outputBase, inputToken, outputToken, scratchToken,
            normalElements * FP32_BYTES, lastElements * FP32_BYTES);
    }

    if (isolatedExactP13CompactRegistered) {
        CHK_PRT_RET(param.rankSize != 4U || param.inputPtr == param.outputPtr
                        || totalBytes != FOUR_BY_ONE_EXACT_P13_COMPACT_BYTES,
            HCCL_ERROR("[CCU_V051] invalid exact-P13 compact invocation"), HCCL_E_PARA);
        const uint64_t normalElements = param.count / param.rankSize;
        const uint64_t lastElements = param.count - normalElements * (param.rankSize - 1U);
        return LaunchExactP13CompactWindowV051(param, resCtx, inputBase, outputBase, inputToken, outputToken,
            scratchToken, normalElements * FP32_BYTES, lastElements * FP32_BYTES);
    }

    if (isolatedExactP15NhrRegistered) {
        CHK_PRT_RET(param.rankSize != 4U || param.inputPtr == param.outputPtr
                        || totalBytes != FOUR_BY_ONE_EXACT_P15_NHR_BYTES,
            HCCL_ERROR("[CCU_V063] invalid exact-P15 NHR invocation"), HCCL_E_PARA);
        const uint64_t normalElements = param.count / param.rankSize;
        const uint64_t lastElements = param.count - normalElements * (param.rankSize - 1U);
        return LaunchExactP14NhrWindow(param, resCtx, inputBase, outputBase, inputToken, outputToken, scratchToken,
            normalElements * FP32_BYTES, lastElements * FP32_BYTES);
    }

    CHK_PRT_RET(isolatedExactP17TreePushRegistered && !useExactP17TreePush,
        HCCL_ERROR("[CCU_V064] invalid exact-P17 tree-push invocation"), HCCL_E_PARA);
    CHK_PRT_RET(isolatedExactP16CompactPreSyncRegistered && !useExactP16CompactPreSync,
        HCCL_ERROR("[CCU_V065] invalid exact-P16 compact-pre-sync invocation"), HCCL_E_PARA);
    CHK_PRT_RET(isolatedExactP18TreeRegistered && !useExactP18Tree,
        HCCL_ERROR("[CCU_V066] invalid exact-P18 tree invocation"), HCCL_E_PARA);
    CHK_PRT_RET(isolatedExactP11TreePushRegistered && !useExactP11TreePush,
        HCCL_ERROR("[CCU_V067] invalid exact-P11 tree-push invocation"), HCCL_E_PARA);
    CHK_PRT_RET(isolatedExactP12TreePushRegistered && !useExactP12TreePush,
        HCCL_ERROR("[CCU_V068] invalid exact-P12 tree-push invocation"), HCCL_E_PARA);

    uint64_t processedElements = 0;
    while (processedElements < param.count) {
        const uint64_t remainingElements = param.count - processedElements;
        uint64_t windowElements = 0;
        uint64_t normalElements = 0;
        uint64_t lastElements = 0;
        CHK_RET(SelectWindow(
            remainingElements, param.rankSize, resCtx.localBuffer.size, windowElements, normalElements, lastElements));

        const uint64_t byteOffset = processedElements * FP32_BYTES;
        if (useDualGroupPushSmall || useExactP17TreePush || useExactP16CompactPreSync || useExactP18Tree
            || useExactP11TreePush || useExactP12TreePush) {
            CHK_RET(LaunchDualGroupPushWindow(param, resCtx, inputBase + byteOffset, outputBase + byteOffset,
                inputToken, outputToken, scratchToken, normalElements * FP32_BYTES, lastElements * FP32_BYTES));
        } else {
            CHK_RET(LaunchWindow(param, resCtx, inputBase + byteOffset, outputBase + byteOffset, inputToken,
                outputToken, scratchToken, normalElements * FP32_BYTES, lastElements * FP32_BYTES,
                useSingleDieGroupReduce));
        }
        processedElements += windowElements;
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
