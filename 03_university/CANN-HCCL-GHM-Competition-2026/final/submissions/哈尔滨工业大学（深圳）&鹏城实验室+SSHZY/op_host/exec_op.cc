/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ===========================================================================
// ReduceScatter CCU 任务编排（Host 侧）
// 支持两种算法路径：Recursive Halving（小数据）和 Read+LocalReduce（大数据）
// ===========================================================================

#include <ccu/ccu_res.h>
#include <ccu/ccu_launch.h>

#include <array>

#include "custom.h"
#include "exec_op.h"
#include "ccu_kernel.h"

namespace ops_hccl {

// RH: WriteReduce(my_INPUT → peer_INPUT), 4*1 专用
// Mesh: Read(peer_INPUT → CCL) + tree LocalReduce, 其余场景
static constexpr uint64_t ALIGN = 128; // 128B 对齐
static constexpr uint64_t MAX_SINGLE_COMM = 256ULL * 1024 * 1024;

static bool CalDrrChunkGeometry(uint64_t blockBytes, uint32_t n, uint64_t &chunkBytes, uint64_t &lastChunkBytes)
{
    if (n == 0) {
        return false;
    }
    chunkBytes = (blockBytes / n) & ~(ALIGN - 1);
    if (chunkBytes == 0) {
        return false;
    }
    lastChunkBytes = blockBytes - chunkBytes * (n - 1U);
    return lastChunkBytes != 0 && lastChunkBytes <= MAX_SINGLE_COMM;
}

static constexpr uint32_t GDRR_MAX_GROUPS = 4;

static HcclResult CalGroupDrrGeometry(uint64_t blockBytes, uint32_t channelCount, uint32_t maxGroupSize,
    std::array<uint64_t, GDRR_MAX_GROUPS> &chunkBytes, std::array<uint64_t, GDRR_MAX_GROUPS> &lastChunkBytes)
{
    const uint32_t groupCount = GetGroupDrrGroupCount(channelCount, maxGroupSize);
    CHK_PRT_RET(groupCount == 0 || groupCount > GDRR_MAX_GROUPS,
        HCCL_ERROR("CalGroupDrrGeometry: invalid group count[%u], channelCount[%u], maxGroupSize[%u]", groupCount,
            channelCount, maxGroupSize),
        HCCL_E_INTERNAL);
    for (uint32_t g = 0; g < GDRR_MAX_GROUPS; g++) {
        chunkBytes[g] = 0;
        lastChunkBytes[g] = 0;
    }
    for (uint32_t g = 0; g < groupCount; g++) {
        const uint32_t groupSize = GetGroupDrrGroupSize(channelCount, maxGroupSize, g);
        CHK_PRT_RET(!CalDrrChunkGeometry(blockBytes, groupSize, chunkBytes[g], lastChunkBytes[g]),
            HCCL_ERROR("CalGroupDrrGeometry: chunk geometry failed, group[%u], size[%u]", g, groupSize),
            HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

static HcclResult SyncMainToSlave(const AlgResourceCtx &resCtx)
{
    CHK_PRT_RET(resCtx.threads.size() < 2,
        HCCL_ERROR("SyncMainToSlave: need at least 2 threads, got[%llu]",
            static_cast<unsigned long long>(resCtx.threads.size())),
        HCCL_E_INTERNAL);
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(resCtx.threads[0], resCtx.threads[1], 0)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resCtx.threads[1], 0, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

static HcclResult SyncSlaveToMain(const AlgResourceCtx &resCtx)
{
    CHK_PRT_RET(resCtx.threads.size() < 2,
        HCCL_ERROR("SyncSlaveToMain: need at least 2 threads, got[%llu]",
            static_cast<unsigned long long>(resCtx.threads.size())),
        HCCL_E_INTERNAL);
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(resCtx.threads[0], 0, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(resCtx.threads[1], resCtx.threads[0], 0)));
    return HCCL_SUCCESS;
}

static HcclResult LaunchKernel(
    ThreadHandle thread, CcuKernelHandle kernel, const uint64_t *taskArgs, size_t taskArgCount, const char *tag)
{
    CcuResult launchRet = HcommCcuKernelLaunch(thread, kernel, taskArgs, taskArgCount);
    CHK_PRT_RET(launchRet != CCU_SUCCESS,
        HCCL_ERROR("%s: kernel launch failed, ccuRet[%d]", tag, static_cast<int>(launchRet)), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

template <size_t N>
static HcclResult LaunchKernel(
    ThreadHandle thread, CcuKernelHandle kernel, const std::array<uint64_t, N> &taskArgs, const char *tag)
{
    return LaunchKernel(thread, kernel, taskArgs.data(), taskArgs.size(), tag);
}

static uint64_t GetWeightedMergeLowBytes(uint64_t mergeBytes, uint32_t elemSize, uint64_t die0Cost, uint64_t die1Cost)
{
    const uint64_t mergeElems = mergeBytes / elemSize;
    if (mergeElems < 2) {
        return (mergeElems / 2U) * elemSize;
    }

    const uint64_t totalCost = die0Cost + die1Cost;
    if (die0Cost == 0 || die1Cost == 0 || totalCost == 0) {
        return (mergeElems / 2U) * elemSize;
    }

    uint64_t die0Elems = (mergeElems * die1Cost + totalCost / 2U) / totalCost;
    const uint64_t minElems = mergeElems / 4U;
    const uint64_t maxElems = mergeElems - minElems;
    if (die0Elems < minElems) {
        die0Elems = minElems;
    } else if (die0Elems > maxElems) {
        die0Elems = maxElems;
    }
    return die0Elems * elemSize;
}

// RH: WriteReduce(my_INPUT → peer_INPUT), 数据原地累积, 无需 CCL
static HcclResult ExecOpRH(const OpParam &param, AlgResourceCtx &resCtx)
{
    const uint32_t ts = GetElemSize(param.dataType);
    const uint64_t totalCount = param.count;
    const uint64_t blockBytes = totalCount * ts;

    uint8_t *in = static_cast<uint8_t *>(param.inputPtr);
    uint8_t *out = static_cast<uint8_t *>(param.outputPtr);

    uint64_t inputVa = reinterpret_cast<uint64_t>(in);
    uint64_t outputVa = reinterpret_cast<uint64_t>(out);
    uint64_t inputToken = 0, outputToken = 0;
    CHK_RET(static_cast<HcclResult>(
        HcommCcuGetMemToken(inputVa, static_cast<uint64_t>(param.rankSize) * blockBytes, &inputToken)));
    CHK_RET(static_cast<HcclResult>(HcommCcuGetMemToken(outputVa, blockBytes, &outputToken)));

    const uint64_t maxSliceBytes = MAX_SINGLE_COMM & ~(ALIGN - 1);
    const uint64_t maxSliceCount = maxSliceBytes / ts;

    const uint64_t numSlices = (totalCount + maxSliceCount - 1) / maxSliceCount;
    const uint64_t evenSliceCount = (totalCount + numSlices - 1) / numSlices;

    uint64_t sliceOff = 0;
    while (sliceOff < totalCount) {
        const uint64_t sliceCount
            = ((evenSliceCount < (totalCount - sliceOff)) ? evenSliceCount : (totalCount - sliceOff));
        const uint64_t curSliceBytes = sliceCount * ts;
        const uint64_t sliceByteOff = sliceOff * ts;

        // task args: myInputVa, outputVa, myInputToken, outputToken,
        //            sliceBytes, sliceByteOff, blockBytes
        std::array<uint64_t, 7> taskArgs
            = {inputVa, outputVa, inputToken, outputToken, curSliceBytes, sliceByteOff, blockBytes};

        for (auto &kHandle : resCtx.ccuKernels) {
            CHK_RET(LaunchKernel(resCtx.ccuThread, kHandle, taskArgs, "ExecOpRH"));
        }
        sliceOff += sliceCount;
    }
    return HCCL_SUCCESS;
}

// 当前算法（Read + LocalReduce）：mesh 路径
static HcclResult ExecOpMesh(const OpParam &param, AlgResourceCtx &resCtx)
{
    const uint32_t ts = GetElemSize(param.dataType);
    const uint64_t totalCount = param.count;
    const uint64_t blockBytes = totalCount * ts;
    const uint64_t cclSize = resCtx.localBuffer.size;

    const bool twoDieMesh = !resCtx.dieChannels[0].empty() && !resCtx.dieChannels[1].empty();
    const bool pipelineMerge = twoDieMesh && resCtx.ccuKernels.size() >= 4;
    const uint32_t outputDie = SelectMeshOutputDie(resCtx);
    const uint64_t dieWorkCost[2]
        = {pipelineMerge ? EstimateDiePipelineCost(resCtx.dieChannels[0]) : 0U,
            pipelineMerge ? EstimateDiePipelineCost(resCtx.dieChannels[1]) : 0U};
    const uint64_t groupSlots[2] = {twoDieMesh ? static_cast<uint64_t>(resCtx.dieChannels[0].size()) : 0U,
        twoDieMesh ? static_cast<uint64_t>(resCtx.dieChannels[1].size()) : 0U};
    uint64_t pipelineScratchSlots[2] = {groupSlots[0], groupSlots[1]};
    if (pipelineMerge) {
        const uint32_t nonOutputDie = 1U - outputDie;
        if (pipelineScratchSlots[nonOutputDie] > 0) {
            pipelineScratchSlots[nonOutputDie]--;
        }
    }
    const uint64_t pipelineScratchSlotCount = pipelineScratchSlots[0] + pipelineScratchSlots[1];
    uint64_t maxSliceBytes = MAX_SINGLE_COMM;
    const uint64_t scratchSlots = pipelineMerge ? (pipelineScratchSlotCount + 2U)
                                                : (twoDieMesh ? (static_cast<uint64_t>(param.rankSize) - 1U)
                                                              : static_cast<uint64_t>(param.rankSize));
    uint64_t cclPerScratchSlot = cclSize / scratchSlots;
    if (cclPerScratchSlot < maxSliceBytes) {
        maxSliceBytes = cclPerScratchSlot;
    }
    maxSliceBytes = (maxSliceBytes >= ALIGN) ? (maxSliceBytes & ~(ALIGN - 1)) : ALIGN;
    CHK_PRT_RET(maxSliceBytes < ts,
        HCCL_ERROR("ExecOpMesh: CCL too small, size[%llu]", static_cast<unsigned long long>(cclSize)), HCCL_E_INTERNAL);

    const uint64_t maxSliceCount = maxSliceBytes / ts;
    const uint64_t numSlices = (totalCount + maxSliceCount - 1) / maxSliceCount;
    const uint64_t evenSliceCount = (totalCount + numSlices - 1) / numSlices;

    uint64_t inputVa = reinterpret_cast<uint64_t>(param.inputPtr);
    uint64_t outputVa = reinterpret_cast<uint64_t>(param.outputPtr);
    uint64_t cclVa = reinterpret_cast<uint64_t>(resCtx.localBuffer.addr);
    const uint64_t scratchVa = cclVa;

    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    CHK_RET(static_cast<HcclResult>(
        HcommCcuGetMemToken(inputVa, static_cast<uint64_t>(param.rankSize) * blockBytes, &inputToken)));
    CHK_RET(static_cast<HcclResult>(HcommCcuGetMemToken(outputVa, blockBytes, &outputToken)));

    CHK_PRT_RET(twoDieMesh
            && (resCtx.threads.size() < 2 || resCtx.ccuKernels.size() < (pipelineMerge ? 4U : 3U)),
        HCCL_ERROR("ExecOpMesh: 2-die mesh has invalid resources, threads[%llu], kernels[%llu], pipeline[%u]",
            static_cast<unsigned long long>(resCtx.threads.size()),
            static_cast<unsigned long long>(resCtx.ccuKernels.size()), static_cast<uint32_t>(pipelineMerge)),
        HCCL_E_INTERNAL);
    const uint64_t firstGroupSlots = twoDieMesh ? (pipelineMerge ? pipelineScratchSlots[0] : groupSlots[0]) : 0;
    const uint64_t partialBaseVa = pipelineMerge ? (scratchVa + pipelineScratchSlotCount * maxSliceBytes) : 0;

    uint64_t sliceOff = 0;
    uint64_t previousPartialVa = 0;
    uint64_t previousSliceBytes = 0;
    uint64_t previousSliceByteOff = 0;
    uint64_t sliceIndex = 0;
    while (sliceOff < totalCount) {
        const uint64_t sliceCount
            = ((evenSliceCount < (totalCount - sliceOff)) ? evenSliceCount : (totalCount - sliceOff));
        const uint64_t sliceBytes = sliceCount * ts;
        const uint64_t sliceByteOff = sliceOff * ts;
        // 与 ccu_kernel.cc LoadArgsBase 顺序一致
        std::array<uint64_t, 13> taskArgs = {
            inputVa,              // 0: myInputVa
            outputVa,             // 1: myOutputVa
            inputToken,           // 2: myInputToken
            outputToken,          // 3: myOutputToken
            scratchVa,            // 4: myCclVa
            resCtx.localCclToken, // 5: myCclToken
            sliceBytes,           // 6: sliceBytes
            sliceByteOff,         // 7: sliceByteOff
            blockBytes,           // 8: blockBytes
            0,                    // 9: currentPartialVa
            0,                    // 10: previousPartialVa
            0,                    // 11: previousMergeBytes
            0                     // 12: previousOutputOff
        };

        if (twoDieMesh) {
            const uint64_t secondScratchOff = firstGroupSlots * sliceBytes;
            const uint64_t partialVa = (outputDie == 0) ? (scratchVa + secondScratchOff) : scratchVa;
            std::array<uint64_t, 13> taskArgs0 = taskArgs;
            std::array<uint64_t, 13> taskArgs1 = taskArgs;
            taskArgs1[4] = scratchVa + secondScratchOff;

            if (pipelineMerge) {
                const uint64_t currentPartialVa = partialBaseVa + (sliceIndex % 2U) * maxSliceBytes;
                const uint64_t previousLowBytes
                    = GetWeightedMergeLowBytes(previousSliceBytes, ts, dieWorkCost[0], dieWorkCost[1]);
                const uint64_t previousMergeBytes[2] = {previousLowBytes, previousSliceBytes - previousLowBytes};
                const uint64_t previousMergeOffsets[2] = {0, previousLowBytes};
                std::array<uint64_t, 13> *dieTaskArgs[2] = {&taskArgs0, &taskArgs1};
                for (uint32_t d = 0; d < 2; d++) {
                    (*dieTaskArgs[d])[9] = currentPartialVa;
                    (*dieTaskArgs[d])[10] = previousPartialVa + previousMergeOffsets[d];
                    (*dieTaskArgs[d])[11] = previousMergeBytes[d];
                    (*dieTaskArgs[d])[12] = previousSliceByteOff + previousMergeOffsets[d];
                }
            }

            CHK_RET(SyncMainToSlave(resCtx));
            if (pipelineMerge) {
                std::array<uint64_t, 14> taskArgsPipe0 = {};
                std::array<uint64_t, 14> taskArgsPipe1 = {};
                for (uint32_t i = 0; i < taskArgs0.size(); i++) {
                    taskArgsPipe0[i] = taskArgs0[i];
                    taskArgsPipe1[i] = taskArgs1[i];
                }
                taskArgsPipe0[13] = (sliceIndex == 0) ? 1U : 0U;
                taskArgsPipe1[13] = taskArgsPipe0[13];
                CHK_RET(LaunchKernel(resCtx.threads[0], resCtx.ccuKernels[0], taskArgsPipe0, "ExecOpMeshPipeline"));
                CHK_RET(LaunchKernel(resCtx.threads[1], resCtx.ccuKernels[1], taskArgsPipe1, "ExecOpMeshPipeline"));
            } else {
                CHK_RET(LaunchKernel(resCtx.threads[0], resCtx.ccuKernels[0], taskArgs0, "ExecOpMesh"));
                CHK_RET(LaunchKernel(resCtx.threads[1], resCtx.ccuKernels[1], taskArgs1, "ExecOpMesh"));
            }
            CHK_RET(SyncSlaveToMain(resCtx));

            if (pipelineMerge) {
                previousPartialVa = partialBaseVa + (sliceIndex % 2U) * maxSliceBytes;
                previousSliceBytes = sliceBytes;
                previousSliceByteOff = sliceByteOff;
            } else {
                std::array<uint64_t, 8> mergeArgs
                    = {partialVa, outputVa, resCtx.localCclToken, outputToken, sliceBytes, 0, sliceByteOff, 0};
                CHK_RET(LaunchKernel(resCtx.threads[0], resCtx.ccuKernels[2], mergeArgs, "ExecOpMeshMerge"));
            }
        } else {
            for (auto &kHandle : resCtx.ccuKernels) {
                CHK_RET(LaunchKernel(resCtx.ccuThread, kHandle, taskArgs, "ExecOpMesh"));
            }
        }

        sliceOff += sliceCount;
        sliceIndex++;
    }

    if (pipelineMerge && previousSliceBytes != 0) {
        const uint64_t lowBytes = (previousSliceBytes / ts / 2U) * ts;
        const uint64_t mergeBytes[2] = {lowBytes, previousSliceBytes - lowBytes};
        const uint64_t mergeOffsets[2] = {0, lowBytes};
        std::array<uint64_t, 8> mergeArgs[2];
        for (uint32_t d = 0; d < 2; d++) {
            mergeArgs[d] = {previousPartialVa + mergeOffsets[d], outputVa, resCtx.localCclToken, outputToken,
                mergeBytes[d], 0, previousSliceByteOff + mergeOffsets[d], 1};
        }
        CHK_RET(SyncMainToSlave(resCtx));
        CHK_RET(LaunchKernel(resCtx.threads[0], resCtx.ccuKernels[2], mergeArgs[0], "ExecOpMeshMergeDrain"));
        CHK_RET(LaunchKernel(resCtx.threads[1], resCtx.ccuKernels[3], mergeArgs[1], "ExecOpMeshMergeDrain"));
        CHK_RET(SyncSlaveToMain(resCtx));
    }
    return HCCL_SUCCESS;
}

static HcclResult ExecOpSmall4x1(const OpParam &param, AlgResourceCtx &resCtx)
{
    CHK_PRT_RET(resCtx.ccuKernels.empty(), HCCL_ERROR("ExecOpSmall4x1: missing kernel"), HCCL_E_INTERNAL);

    const uint32_t ts = GetElemSize(param.dataType);
    const uint64_t totalCount = param.count;
    const uint64_t blockBytes = totalCount * ts;
    const uint64_t cclSize = resCtx.localBuffer.size;
    const uint64_t scratchSlots = 3U;
    uint64_t maxSliceBytes = cclSize / scratchSlots;
    if (maxSliceBytes > MAX_SINGLE_COMM) {
        maxSliceBytes = MAX_SINGLE_COMM;
    }
    maxSliceBytes = (maxSliceBytes >= ALIGN) ? (maxSliceBytes & ~(ALIGN - 1)) : ALIGN;
    CHK_PRT_RET(maxSliceBytes < ts,
        HCCL_ERROR("ExecOpSmall4x1: CCL too small, size[%llu]", static_cast<unsigned long long>(cclSize)),
        HCCL_E_INTERNAL);

    const uint64_t maxSliceCount = maxSliceBytes / ts;
    const uint64_t numSlices = (totalCount + maxSliceCount - 1) / maxSliceCount;
    const uint64_t evenSliceCount = (totalCount + numSlices - 1) / numSlices;

    const uint64_t inputVa = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outputVa = reinterpret_cast<uint64_t>(param.outputPtr);
    const uint64_t cclVa = reinterpret_cast<uint64_t>(resCtx.localBuffer.addr);

    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    CHK_RET(static_cast<HcclResult>(
        HcommCcuGetMemToken(inputVa, static_cast<uint64_t>(param.rankSize) * blockBytes, &inputToken)));
    CHK_RET(static_cast<HcclResult>(HcommCcuGetMemToken(outputVa, blockBytes, &outputToken)));

    uint64_t sliceOff = 0;
    while (sliceOff < totalCount) {
        const uint64_t sliceCount
            = ((evenSliceCount < (totalCount - sliceOff)) ? evenSliceCount : (totalCount - sliceOff));
        const uint64_t sliceBytes = sliceCount * ts;
        const uint64_t sliceByteOff = sliceOff * ts;
        const std::array<uint64_t, 9> taskArgs = {
            inputVa,              // 0: input VA
            outputVa,             // 1: output VA
            inputToken,           // 2: input token
            outputToken,          // 3: output token
            cclVa,                // 4: scratch VA
            resCtx.localCclToken, // 5: scratch token
            sliceBytes,           // 6: current slice bytes
            sliceByteOff,         // 7: current slice offset
            blockBytes            // 8: per-rank output block bytes
        };
        CHK_RET(LaunchKernel(resCtx.ccuThread, resCtx.ccuKernels[0], taskArgs, "ExecOpSmall4x1"));
        sliceOff += sliceCount;
    }
    return HCCL_SUCCESS;
}

static HcclResult ExecOpGroupDrr(const OpParam &param, AlgResourceCtx &resCtx)
{
    const uint32_t ts = GetElemSize(param.dataType);
    const uint64_t blockBytes = param.count * ts;
    const uint32_t dieCount[2] = {static_cast<uint32_t>(resCtx.dieChannels[0].size()),
        static_cast<uint32_t>(resCtx.dieChannels[1].size())};
    const uint32_t maxGroupSize[2] = {GetGroupDrrMaxGroupSize(dieCount[0]), GetGroupDrrMaxGroupSize(dieCount[1])};
    const uint32_t groupCount[2] = {GetGroupDrrGroupCount(dieCount[0], maxGroupSize[0]),
        GetGroupDrrGroupCount(dieCount[1], maxGroupSize[1])};
    const uint32_t requiredBlocks
        = (groupCount[0] == 0 || groupCount[1] == 0) ? 0U : (groupCount[0] + groupCount[1] - 1U);
    CHK_PRT_RET((param.rankSize != 16 && param.rankSize != 12) || resCtx.threads.size() < 2
            || resCtx.ccuKernels.size() != 3
            || dieCount[0] == 0 || dieCount[1] == 0 || dieCount[0] + dieCount[1] != param.rankSize - 1U
            || groupCount[0] == 0 || groupCount[1] == 0 || requiredBlocks == 0
            || blockBytes > resCtx.localBuffer.size / requiredBlocks,
        HCCL_ERROR("ExecOpGroupDrr: resource or layout mismatch"), HCCL_E_INTERNAL);

    const uint64_t inputVa = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outputVa = reinterpret_cast<uint64_t>(param.outputPtr);
    const uint64_t cclVa = reinterpret_cast<uint64_t>(resCtx.localBuffer.addr);
    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    CHK_RET(static_cast<HcclResult>(
        HcommCcuGetMemToken(inputVa, static_cast<uint64_t>(param.rankSize) * blockBytes, &inputToken)));
    CHK_RET(static_cast<HcclResult>(HcommCcuGetMemToken(outputVa, blockBytes, &outputToken)));

    std::array<uint64_t, GDRR_MAX_GROUPS> chunks[2];
    std::array<uint64_t, GDRR_MAX_GROUPS> lastChunks[2];
    CHK_RET(CalGroupDrrGeometry(blockBytes, dieCount[0], maxGroupSize[0], chunks[0], lastChunks[0]));
    CHK_RET(CalGroupDrrGeometry(blockBytes, dieCount[1], maxGroupSize[1], chunks[1], lastChunks[1]));

    const uint32_t outputDie = SelectGroupDrrOutputDie(resCtx);
    const uint32_t partialDie = 1U - outputDie;
    constexpr uint64_t partialOff = 0;
    uint64_t scratchOff[2] = {0, 0};
    uint64_t cursor = blockBytes;
    scratchOff[partialDie] = cursor;
    cursor += static_cast<uint64_t>(groupCount[partialDie] - 1U) * blockBytes;
    scratchOff[outputDie] = cursor;

    auto buildArgs = [&](uint32_t d) -> std::array<uint64_t, 17> {
        return {inputVa, inputToken, outputVa, outputToken, cclVa, resCtx.localCclToken, blockBytes, partialOff,
            scratchOff[d], chunks[d][0], chunks[d][1], chunks[d][2], chunks[d][3], lastChunks[d][0],
            lastChunks[d][1], lastChunks[d][2], lastChunks[d][3]};
    };
    const std::array<uint64_t, 17> die0Args = buildArgs(0);
    const std::array<uint64_t, 17> die1Args = buildArgs(1);

    CHK_RET(SyncMainToSlave(resCtx));
    CHK_RET(LaunchKernel(resCtx.threads[0], resCtx.ccuKernels[0], die0Args, "ExecOpGroupDrrDie0"));
    CHK_RET(LaunchKernel(resCtx.threads[1], resCtx.ccuKernels[1], die1Args, "ExecOpGroupDrrDie1"));
    CHK_RET(SyncSlaveToMain(resCtx));

    const std::array<uint64_t, 8> mergeArgs
        = {cclVa + partialOff, outputVa, resCtx.localCclToken, outputToken, blockBytes, 0, 0, 0};
    CHK_RET(LaunchKernel(resCtx.threads[0], resCtx.ccuKernels[2], mergeArgs, "ExecOpGroupDrrMerge"));
    return HCCL_SUCCESS;
}

HcclResult ExecOp(const OpParam &param)
{
    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> seq(ctx, ctx + param.ctxSize);
    AlgResourceCtx resCtx;
    resCtx.DeSerialize(seq);

    if (param.count == 0 || param.rankSize == 0) {
        return HCCL_SUCCESS;
    }

    const uint32_t ts = GetElemSize(param.dataType);
    if (ts == 0) {
        HCCL_ERROR("ExecOp: unsupported dataType[%d]", static_cast<int>(param.dataType));
        return HCCL_E_PARA;
    }

    if (param.rankSize == 1) {
        const uint64_t blockBytes = param.count * ts;
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(param.cpuThread, param.outputPtr, param.inputPtr, blockBytes)));
        return HCCL_SUCCESS;
    }

    if (resCtx.algoType == RS_ALGO_RH) {
        return ExecOpRH(param, resCtx);
    }
    if (resCtx.algoType == RS_ALGO_GROUP_DRR) {
        return ExecOpGroupDrr(param, resCtx);
    }
    if (resCtx.algoType == RS_ALGO_SMALL_4X1) {
        return ExecOpSmall4x1(param, resCtx);
    }
    CHK_PRT_RET(resCtx.algoType != RS_ALGO_MESH,
        HCCL_ERROR("ExecOp: unknown reduce scatter algo type[%u]", resCtx.algoType), HCCL_E_INTERNAL);
    return ExecOpMesh(param, resCtx);
}

} // namespace ops_hccl
