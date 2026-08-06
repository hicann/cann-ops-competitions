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
#include <limits>
#include <vector>

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>
#include <hcomm/hcomm_primitives.h>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace final_small512 {
namespace ops_hccl {
namespace {
constexpr uint64_t SCRATCH_ALIGNMENT = 256;
constexpr uint64_t KERNEL_MODE_PARTIAL = 0;
constexpr uint64_t KERNEL_MODE_FINALIZE = 1;
constexpr uint64_t KERNEL_MODE_DIRECT_OUTPUT = 2;
constexpr uint32_t KERNEL_MODE_ARG_INDEX = 9;

struct KernelSequence {
    std::vector<const std::vector<CcuKernelHandle> *> kernelSets;
    std::vector<const std::vector<uint32_t> *> pieceSets;
    bool reduceScratchResults = false;
    bool parallelGlobal = false;
    uint32_t requiredScratchSlots = 0;
};

HcclResult ConvertCcuResult(CcuResult result)
{
    switch (result) {
        case CCU_SUCCESS:
            return HCCL_SUCCESS;
        case CCU_E_PARA:
            return HCCL_E_PARA;
        case CCU_E_PTR:
            return HCCL_E_PTR;
        case CCU_E_NOT_SUPPORT:
            return HCCL_E_NOT_SUPPORT;
        case CCU_E_NOT_FOUND:
            return HCCL_E_NOT_FOUND;
        case CCU_E_UNAVAIL:
            return HCCL_E_UNAVAIL;
        default:
            return HCCL_E_INTERNAL;
    }
}

void SelectGlobal(const AlgResourceCtx &resourceCtx, KernelSequence &sequence, bool parallel)
{
    sequence.kernelSets.push_back(&resourceCtx.globalKernels);
    sequence.pieceSets.push_back(&resourceCtx.globalPieceCounts);
    sequence.reduceScratchResults = true;
    sequence.parallelGlobal = parallel && resourceCtx.globalKernels.size() > 1 && resourceCtx.threads.size() > 1;
    sequence.requiredScratchSlots = resourceCtx.globalScratchSlotCount;
}

void SelectHierarchical(const AlgResourceCtx &resourceCtx, KernelSequence &sequence)
{
    if (!resourceCtx.localKernels.empty()) {
        sequence.kernelSets.push_back(&resourceCtx.localKernels);
        sequence.pieceSets.push_back(&resourceCtx.localPieceCounts);
    }
    if (!resourceCtx.remoteKernels.empty()) {
        sequence.kernelSets.push_back(&resourceCtx.remoteKernels);
        sequence.pieceSets.push_back(&resourceCtx.remotePieceCounts);
    }
}

KernelSequence SelectKernelSequence(const AlgResourceCtx &resourceCtx, uint64_t inputBytes)
{
    KernelSequence sequence;
    const auto topologyKind = static_cast<Rs2TopologyKind>(resourceCtx.topologyKind);
    switch (RS2_BUILD_FLAVOR) {
        case Rs2BuildFlavor::HIERARCHICAL:
            SelectHierarchical(resourceCtx, sequence);
            break;
        case Rs2BuildFlavor::HCCL_TOPOLOGY:
            SelectGlobal(resourceCtx, sequence, true);
            break;
        case Rs2BuildFlavor::GLOBAL_MESH_TREE:
            SelectGlobal(resourceCtx, sequence, true);
            break;
        case Rs2BuildFlavor::HCCL_TOPOLOGY_HALF_RING:
        case Rs2BuildFlavor::HCCL_TOPOLOGY_512K_OPT:
            SelectGlobal(resourceCtx, sequence, true);
            break;
        case Rs2BuildFlavor::ADAPTIVE:
            if (inputBytes <= SMALL_INPUT_BYTES || topologyKind == Rs2TopologyKind::FOUR_BY_ONE ||
                topologyKind == Rs2TopologyKind::UNKNOWN) {
                SelectGlobal(resourceCtx, sequence, false);
            } else {
                SelectGlobal(resourceCtx, sequence, true);
            }
            break;
        default:
            break;
    }
    return sequence;
}

uint32_t GetMaxPieceCount(const KernelSequence &sequence)
{
    if (sequence.requiredScratchSlots != 0) {
        return sequence.requiredScratchSlots;
    }
    uint32_t maxPieceCount = 0;
    for (const std::vector<uint32_t> *pieceSet : sequence.pieceSets) {
        for (uint32_t pieceCount : *pieceSet) {
            maxPieceCount = std::max(maxPieceCount, pieceCount);
        }
    }
    return maxPieceCount;
}

HcclResult GetToken(uint64_t address, uint64_t bytes, uint64_t &token)
{
    const CcuResult result = HcommCcuGetMemToken(address, bytes, &token);
    if (result != CCU_SUCCESS) {
        HCCL_ERROR("HcommCcuGetMemToken failed, result[%d], bytes[%llu]", result,
            static_cast<unsigned long long>(bytes));
        return ConvertCcuResult(result);
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchSequence(const OpParam &param, const AlgResourceCtx &resourceCtx,
    const KernelSequence &sequence, uint64_t recvBytes, uint64_t chunkOffset, uint64_t chunkBytes,
    uint64_t inputToken, uint64_t outputToken, uint64_t scratchToken)
{
    const uint64_t inputAddress = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outputAddress = reinterpret_cast<uint64_t>(param.outputPtr);
    const uint64_t scratchAddress = reinterpret_cast<uint64_t>(resourceCtx.localBuffer.addr);
    const uint64_t targetInputOffset = static_cast<uint64_t>(param.myRank) * recvBytes + chunkOffset;
    if (resourceCtx.globalResultSlots.size() > MAX_GLOBAL_RESULT_COUNT) {
        return HCCL_E_INTERNAL;
    }
    std::array<uint64_t, 11 + MAX_GLOBAL_RESULT_COUNT> taskArgs = {inputAddress, outputAddress, inputToken,
        outputToken, scratchAddress, scratchToken, targetInputOffset, chunkOffset, chunkBytes,
        KERNEL_MODE_PARTIAL, resourceCtx.globalResultSlots.size()};
    for (uint32_t resultIdx = 0; resultIdx < MAX_GLOBAL_RESULT_COUNT; ++resultIdx) {
        const uint64_t resultOffset = resultIdx < resourceCtx.globalResultSlots.size() ?
            static_cast<uint64_t>(resourceCtx.globalResultSlots[resultIdx]) * chunkBytes : 0;
        taskArgs[11 + resultIdx] = resultOffset;
    }

    for (const std::vector<CcuKernelHandle> *kernelSet : sequence.kernelSets) {
        const bool launchParallel = sequence.parallelGlobal && kernelSet == &resourceCtx.globalKernels;
        taskArgs[KERNEL_MODE_ARG_INDEX] = sequence.reduceScratchResults && !launchParallel ?
            KERNEL_MODE_DIRECT_OUTPUT : KERNEL_MODE_PARTIAL;
        bool usesSlave = false;
        if (launchParallel) {
            for (uint32_t dieId : resourceCtx.globalKernelDieIds) {
                usesSlave = usesSlave || dieId != 0;
            }
            if (usesSlave) {
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                    resourceCtx.threads[0], resourceCtx.threads[1], 0)));
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                    resourceCtx.threads[1], 0, CUSTOM_TIMEOUT)));
            }
        }

        for (uint32_t kernelIdx = 0; kernelIdx < kernelSet->size(); ++kernelIdx) {
            uint32_t threadIdx = 0;
            if (launchParallel) {
                if (kernelIdx >= resourceCtx.globalKernelDieIds.size()) {
                    return HCCL_E_INTERNAL;
                }
                threadIdx = resourceCtx.globalKernelDieIds[kernelIdx] == 0 ? 0U : 1U;
            }
            const CcuResult launchResult = HcommCcuKernelLaunch(resourceCtx.threads[threadIdx],
                (*kernelSet)[kernelIdx], taskArgs.data(), taskArgs.size());
            if (launchResult != CCU_SUCCESS) {
                HCCL_ERROR("CCU kernel launch failed, result[%d]", launchResult);
                return ConvertCcuResult(launchResult);
            }
        }
        if (launchParallel && usesSlave) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                resourceCtx.threads[1], resourceCtx.threads[0], 0)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                resourceCtx.threads[0], 0, CUSTOM_TIMEOUT)));
        }
    }

    if (sequence.reduceScratchResults && sequence.parallelGlobal) {
        if (resourceCtx.globalResultSlots.empty() || resourceCtx.globalKernels.empty()) {
            return HCCL_E_INTERNAL;
        }
        taskArgs[KERNEL_MODE_ARG_INDEX] = KERNEL_MODE_FINALIZE;
        const CcuResult finalizeResult = HcommCcuKernelLaunch(resourceCtx.threads[0],
            resourceCtx.globalKernels[0], taskArgs.data(), taskArgs.size());
        if (finalizeResult != CCU_SUCCESS) {
            HCCL_ERROR("CCU finalize kernel launch failed, result[%d]", finalizeResult);
            return ConvertCcuResult(finalizeResult);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchSmall(const OpParam &param, const AlgResourceCtx &resourceCtx,
    uint64_t recvBytes, uint64_t inputToken, uint64_t outputToken, uint64_t scratchToken)
{
    const uint32_t kernelCount = static_cast<uint32_t>(resourceCtx.smallKernels.size());
    if (kernelCount == 0 || kernelCount > 2 || resourceCtx.smallKernelDieIds.size() != kernelCount ||
        resourceCtx.smallResultSlots.size() != kernelCount || resourceCtx.threads.empty() ||
        resourceCtx.smallDirectKernelIndex >= kernelCount || resourceCtx.localBuffer.addr == nullptr) {
        return HCCL_E_INTERNAL;
    }
    if (recvBytes > std::numeric_limits<uint64_t>::max() / resourceCtx.smallScratchSlotCount ||
        recvBytes * resourceCtx.smallScratchSlotCount > resourceCtx.localBuffer.size) {
        return HCCL_E_MEMORY;
    }

    std::array<uint64_t, 9> taskArgs = {
        reinterpret_cast<uint64_t>(param.inputPtr), reinterpret_cast<uint64_t>(param.outputPtr), inputToken,
        outputToken, reinterpret_cast<uint64_t>(resourceCtx.localBuffer.addr), scratchToken,
        static_cast<uint64_t>(param.myRank) * recvBytes, 0, recvBytes,
    };

    bool usesSlave = false;
    for (uint32_t dieId : resourceCtx.smallKernelDieIds) {
        usesSlave = usesSlave || dieId != 0;
    }
    if ((usesSlave || kernelCount == 2) && resourceCtx.threads.size() < 2) {
        return HCCL_E_INTERNAL;
    }
    if (usesSlave) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resourceCtx.threads[0], resourceCtx.threads[1], 0)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            resourceCtx.threads[1], 0, CUSTOM_TIMEOUT)));
    }

    for (uint32_t kernelIdx = 0; kernelIdx < kernelCount; ++kernelIdx) {
        const uint32_t threadIdx = usesSlave && resourceCtx.smallKernelDieIds[kernelIdx] != 0 ? 1U : 0U;
        const CcuResult launchResult = HcommCcuKernelLaunch(resourceCtx.threads[threadIdx],
            resourceCtx.smallKernels[kernelIdx], taskArgs.data(), taskArgs.size());
        if (launchResult != CCU_SUCCESS) {
            return ConvertCcuResult(launchResult);
        }
    }

    if (kernelCount == 2) {
        if (resourceCtx.smallFinalizeKernel == 0) {
            return HCCL_E_INTERNAL;
        }
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resourceCtx.threads[1], resourceCtx.threads[0], 0)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            resourceCtx.threads[0], 0, CUSTOM_TIMEOUT)));
        const uint32_t partialKernelIdx = resourceCtx.smallDirectKernelIndex == 0 ? 1U : 0U;
        const uint64_t partialOffset = static_cast<uint64_t>(resourceCtx.smallResultSlots[partialKernelIdx]) * recvBytes;
        const std::array<uint64_t, 6> finalizeArgs = {
            reinterpret_cast<uint64_t>(param.outputPtr), outputToken,
            reinterpret_cast<uint64_t>(resourceCtx.localBuffer.addr), scratchToken, partialOffset, recvBytes,
        };
        const CcuResult finalizeResult = HcommCcuKernelLaunch(resourceCtx.threads[0],
            resourceCtx.smallFinalizeKernel, finalizeArgs.data(), finalizeArgs.size());
        if (finalizeResult != CCU_SUCCESS) {
            return ConvertCcuResult(finalizeResult);
        }
    } else if (usesSlave) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resourceCtx.threads[1], resourceCtx.threads[0], 0)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            resourceCtx.threads[0], 0, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchHalfRingPair(const OpParam &param, const AlgResourceCtx &resourceCtx,
    uint64_t recvBytes, uint64_t inputToken, uint64_t outputToken)
{
    if (resourceCtx.threads.empty() || resourceCtx.halfRingKernels.size() != 1 ||
        resourceCtx.halfRingKernelDieIds.size() != 1) {
        return HCCL_E_INTERNAL;
    }
    const uint64_t recvElements = recvBytes / sizeof(float);
    const uint64_t firstElements = (recvElements + 1) / 2;
    const uint64_t firstBytes = firstElements * sizeof(float);
    const uint64_t secondBytes = recvBytes - firstBytes;
    if (firstBytes == 0 || secondBytes == 0 || firstBytes > MAX_DATA_SIZE || secondBytes > MAX_DATA_SIZE) {
        return HCCL_E_NOT_SUPPORT;
    }

    const std::array<uint64_t, 7> taskArgs = {
        reinterpret_cast<uint64_t>(param.inputPtr), reinterpret_cast<uint64_t>(param.outputPtr),
        inputToken, outputToken, recvBytes, firstBytes, secondBytes,
    };

    const CcuResult launchResult = HcommCcuKernelLaunch(resourceCtx.threads[0], resourceCtx.halfRingKernels[0],
        taskArgs.data(), taskArgs.size());
    if (launchResult != CCU_SUCCESS) {
        return ConvertCcuResult(launchResult);
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchBalancedChunks(const OpParam &param, const AlgResourceCtx &resourceCtx,
    const KernelSequence &kernelSequence, uint64_t recvBytes, uint64_t chunkLimit,
    uint64_t inputToken, uint64_t outputToken, uint64_t scratchToken)
{
    const uint64_t recvElements = recvBytes / sizeof(float);
    const uint64_t elementLimit = chunkLimit / sizeof(float);
    if (recvElements == 0 || elementLimit == 0) {
        return HCCL_E_PARA;
    }
    const uint64_t chunkCount = (recvElements + elementLimit - 1) / elementLimit;
    const uint64_t baseElements = recvElements / chunkCount;
    const uint64_t extraElements = recvElements % chunkCount;
    uint64_t chunkOffset = 0;
    for (uint64_t chunkIdx = 0; chunkIdx < chunkCount; ++chunkIdx) {
        const uint64_t chunkElements = baseElements + (chunkIdx < extraElements ? 1 : 0);
        const uint64_t chunkBytes = chunkElements * sizeof(float);
        CHK_RET(LaunchSequence(param, resourceCtx, kernelSequence, recvBytes, chunkOffset, chunkBytes,
            inputToken, outputToken, scratchToken));
        chunkOffset += chunkBytes;
    }
    return chunkOffset == recvBytes ? HCCL_SUCCESS : HCCL_E_INTERNAL;
}
} // namespace

HcclResult ExecOp(const OpParam &param)
{
    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> sequenceData(ctx, ctx + param.ctxSize);
    AlgResourceCtx resourceCtx;
    resourceCtx.DeSerialize(sequenceData);

    if (param.dataType != HCCL_DATA_TYPE_FP32 || param.reduceType != HCCL_REDUCE_SUM) {
        return HCCL_E_NOT_SUPPORT;
    }
    if (param.rankSize == 0 || param.myRank >= param.rankSize || resourceCtx.threads.empty()) {
        return HCCL_E_PARA;
    }
    if (param.count > std::numeric_limits<uint64_t>::max() / sizeof(float)) {
        return HCCL_E_PARA;
    }

    const uint64_t recvBytes = param.count * sizeof(float);
    if (recvBytes == 0) {
        return HCCL_SUCCESS;
    }
    if (param.rankSize == 1) {
        return static_cast<HcclResult>(HcommLocalCopyOnThread(resourceCtx.threads[0], param.outputPtr,
            param.inputPtr, recvBytes));
    }
    if (recvBytes > std::numeric_limits<uint64_t>::max() / param.rankSize) {
        return HCCL_E_PARA;
    }
    const uint64_t inputBytes = recvBytes * param.rankSize;

    const auto topologyKind = static_cast<Rs2TopologyKind>(resourceCtx.topologyKind);
    if (RS2_BUILD_FLAVOR == Rs2BuildFlavor::HCCL_TOPOLOGY_512K_OPT &&
        inputBytes <= SMALL_INPUT_BYTES && !resourceCtx.smallKernels.empty()) {
        uint64_t inputToken = 0;
        uint64_t outputToken = 0;
        uint64_t scratchToken = 0;
        CHK_RET(GetToken(reinterpret_cast<uint64_t>(param.inputPtr), inputBytes, inputToken));
        CHK_RET(GetToken(reinterpret_cast<uint64_t>(param.outputPtr), recvBytes, outputToken));
        CHK_RET(GetToken(reinterpret_cast<uint64_t>(resourceCtx.localBuffer.addr), resourceCtx.localBuffer.size,
            scratchToken));
        return LaunchSmall(param, resourceCtx, recvBytes, inputToken, outputToken, scratchToken);
    }

    if ((RS2_BUILD_FLAVOR == Rs2BuildFlavor::HCCL_TOPOLOGY_HALF_RING ||
        RS2_BUILD_FLAVOR == Rs2BuildFlavor::HCCL_TOPOLOGY_512K_OPT) &&
        topologyKind == Rs2TopologyKind::FOUR_BY_ONE && inputBytes > SMALL_INPUT_BYTES) {
        uint64_t inputToken = 0;
        uint64_t outputToken = 0;
        CHK_RET(GetToken(reinterpret_cast<uint64_t>(param.inputPtr), inputBytes, inputToken));
        CHK_RET(GetToken(reinterpret_cast<uint64_t>(param.outputPtr), recvBytes, outputToken));
        return LaunchHalfRingPair(param, resourceCtx, recvBytes, inputToken, outputToken);
    }

    const KernelSequence kernelSequence = SelectKernelSequence(resourceCtx, inputBytes);
    if (kernelSequence.kernelSets.empty()) {
        HCCL_ERROR("No CCU kernel sequence selected");
        return HCCL_E_INTERNAL;
    }
    const uint32_t maxPieceCount = GetMaxPieceCount(kernelSequence);
    if (maxPieceCount == 0 || resourceCtx.localBuffer.addr == nullptr || resourceCtx.localBuffer.size == 0) {
        return HCCL_E_MEMORY;
    }

    uint64_t chunkLimit = std::min<uint64_t>(MAX_DATA_SIZE, resourceCtx.localBuffer.size / maxPieceCount);
    chunkLimit = chunkLimit / SCRATCH_ALIGNMENT * SCRATCH_ALIGNMENT;
    if (chunkLimit == 0) {
        chunkLimit = resourceCtx.localBuffer.size / maxPieceCount;
        chunkLimit = chunkLimit / sizeof(float) * sizeof(float);
    }
    if (chunkLimit == 0) {
        return HCCL_E_MEMORY;
    }

    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    uint64_t scratchToken = 0;
    CHK_RET(GetToken(reinterpret_cast<uint64_t>(param.inputPtr), inputBytes, inputToken));
    CHK_RET(GetToken(reinterpret_cast<uint64_t>(param.outputPtr), recvBytes, outputToken));
    CHK_RET(GetToken(reinterpret_cast<uint64_t>(resourceCtx.localBuffer.addr), resourceCtx.localBuffer.size,
        scratchToken));
    return LaunchBalancedChunks(param, resourceCtx, kernelSequence, recvBytes, chunkLimit,
        inputToken, outputToken, scratchToken);
}
} // namespace ops_hccl
} // namespace final_small512

namespace final_scratch_lite {
namespace ops_hccl {
namespace {
constexpr uint64_t SMALL_INPUT_BYTES = 1ULL * 1024 * 1024;
constexpr uint64_t SCRATCH_ALIGNMENT = 256;
constexpr uint64_t KERNEL_MODE_PARTIAL = 0;
constexpr uint64_t KERNEL_MODE_FINALIZE = 1;
constexpr uint64_t KERNEL_MODE_DIRECT_OUTPUT = 2;
constexpr uint32_t KERNEL_MODE_ARG_INDEX = 9;

struct KernelSequence {
    std::vector<const std::vector<CcuKernelHandle> *> kernelSets;
    std::vector<const std::vector<uint32_t> *> pieceSets;
    bool reduceScratchResults = false;
    bool parallelGlobal = false;
    uint32_t requiredScratchSlots = 0;
};

HcclResult ConvertCcuResult(CcuResult result)
{
    switch (result) {
        case CCU_SUCCESS:
            return HCCL_SUCCESS;
        case CCU_E_PARA:
            return HCCL_E_PARA;
        case CCU_E_PTR:
            return HCCL_E_PTR;
        case CCU_E_NOT_SUPPORT:
            return HCCL_E_NOT_SUPPORT;
        case CCU_E_NOT_FOUND:
            return HCCL_E_NOT_FOUND;
        case CCU_E_UNAVAIL:
            return HCCL_E_UNAVAIL;
        default:
            return HCCL_E_INTERNAL;
    }
}

void SelectGlobal(const AlgResourceCtx &resourceCtx, KernelSequence &sequence, bool parallel)
{
    sequence.kernelSets.push_back(&resourceCtx.globalKernels);
    sequence.pieceSets.push_back(&resourceCtx.globalPieceCounts);
    sequence.reduceScratchResults = true;
    sequence.parallelGlobal = parallel && resourceCtx.globalKernels.size() > 1 && resourceCtx.threads.size() > 1;
    sequence.requiredScratchSlots = resourceCtx.globalScratchSlotCount;
}

void SelectHierarchical(const AlgResourceCtx &resourceCtx, KernelSequence &sequence)
{
    if (!resourceCtx.localKernels.empty()) {
        sequence.kernelSets.push_back(&resourceCtx.localKernels);
        sequence.pieceSets.push_back(&resourceCtx.localPieceCounts);
    }
    if (!resourceCtx.remoteKernels.empty()) {
        sequence.kernelSets.push_back(&resourceCtx.remoteKernels);
        sequence.pieceSets.push_back(&resourceCtx.remotePieceCounts);
    }
}

KernelSequence SelectKernelSequence(const AlgResourceCtx &resourceCtx, uint64_t inputBytes)
{
    KernelSequence sequence;
    const auto topologyKind = static_cast<Rs2TopologyKind>(resourceCtx.topologyKind);
    switch (RS2_BUILD_FLAVOR) {
        case Rs2BuildFlavor::HIERARCHICAL:
            SelectHierarchical(resourceCtx, sequence);
            break;
        case Rs2BuildFlavor::HCCL_TOPOLOGY:
            SelectGlobal(resourceCtx, sequence, true);
            break;
        case Rs2BuildFlavor::GLOBAL_MESH_TREE:
            SelectGlobal(resourceCtx, sequence, true);
            break;
        case Rs2BuildFlavor::SCRATCH_LITE:
            SelectGlobal(resourceCtx, sequence, true);
            break;
        case Rs2BuildFlavor::ADAPTIVE:
            if (inputBytes <= SMALL_INPUT_BYTES || topologyKind == Rs2TopologyKind::FOUR_BY_ONE ||
                topologyKind == Rs2TopologyKind::UNKNOWN) {
                SelectGlobal(resourceCtx, sequence, false);
            } else {
                SelectGlobal(resourceCtx, sequence, true);
            }
            break;
        default:
            break;
    }
    return sequence;
}

uint32_t GetMaxPieceCount(const KernelSequence &sequence)
{
    if (sequence.requiredScratchSlots != 0) {
        return sequence.requiredScratchSlots;
    }
    uint32_t maxPieceCount = 0;
    for (const std::vector<uint32_t> *pieceSet : sequence.pieceSets) {
        for (uint32_t pieceCount : *pieceSet) {
            maxPieceCount = std::max(maxPieceCount, pieceCount);
        }
    }
    return maxPieceCount;
}

HcclResult GetToken(uint64_t address, uint64_t bytes, uint64_t &token)
{
    const CcuResult result = HcommCcuGetMemToken(address, bytes, &token);
    if (result != CCU_SUCCESS) {
        HCCL_ERROR("HcommCcuGetMemToken failed, result[%d], bytes[%llu]", result,
            static_cast<unsigned long long>(bytes));
        return ConvertCcuResult(result);
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchSequence(const OpParam &param, const AlgResourceCtx &resourceCtx,
    const KernelSequence &sequence, uint64_t recvBytes, uint64_t chunkOffset, uint64_t chunkBytes,
    uint64_t inputToken, uint64_t outputToken, uint64_t scratchToken)
{
    const uint64_t inputAddress = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outputAddress = reinterpret_cast<uint64_t>(param.outputPtr);
    const uint64_t scratchAddress = reinterpret_cast<uint64_t>(resourceCtx.localBuffer.addr);
    const uint64_t targetInputOffset = static_cast<uint64_t>(param.myRank) * recvBytes + chunkOffset;
    if (resourceCtx.globalResultSlots.size() > MAX_GLOBAL_RESULT_COUNT) {
        return HCCL_E_INTERNAL;
    }
    std::vector<uint64_t> taskArgs = {inputAddress, outputAddress, inputToken, outputToken, scratchAddress,
        scratchToken, targetInputOffset, chunkOffset, chunkBytes, KERNEL_MODE_PARTIAL,
        resourceCtx.globalResultSlots.size()};
    for (uint32_t resultIdx = 0; resultIdx < MAX_GLOBAL_RESULT_COUNT; ++resultIdx) {
        const uint64_t resultOffset = resultIdx < resourceCtx.globalResultSlots.size() ?
            static_cast<uint64_t>(resourceCtx.globalResultSlots[resultIdx]) * chunkBytes : 0;
        taskArgs.push_back(resultOffset);
    }

    for (const std::vector<CcuKernelHandle> *kernelSet : sequence.kernelSets) {
        const bool launchParallel = sequence.parallelGlobal && kernelSet == &resourceCtx.globalKernels;
        taskArgs[KERNEL_MODE_ARG_INDEX] = sequence.reduceScratchResults && !launchParallel ?
            KERNEL_MODE_DIRECT_OUTPUT : KERNEL_MODE_PARTIAL;
        bool usesSlave = false;
        if (launchParallel) {
            for (uint32_t dieId : resourceCtx.globalKernelDieIds) {
                usesSlave = usesSlave || dieId != 0;
            }
            if (usesSlave) {
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                    resourceCtx.threads[0], resourceCtx.threads[1], 0)));
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                    resourceCtx.threads[1], 0, CUSTOM_TIMEOUT)));
            }
        }

        std::vector<uint32_t> launchOrder(kernelSet->size());
        for (uint32_t kernelIdx = 0; kernelIdx < kernelSet->size(); ++kernelIdx) {
            launchOrder[kernelIdx] = kernelIdx;
        }
        const bool smallInput = recvBytes <= SMALL_INPUT_BYTES / param.rankSize;
        if (smallInput && kernelSet == &resourceCtx.globalKernels &&
            resourceCtx.globalPieceCounts.size() == launchOrder.size()) {
            std::stable_sort(launchOrder.begin(), launchOrder.end(), [&](uint32_t lhs, uint32_t rhs) {
                return resourceCtx.globalPieceCounts[lhs] > resourceCtx.globalPieceCounts[rhs];
            });
        }

        for (uint32_t kernelIdx : launchOrder) {
            uint32_t threadIdx = 0;
            if (launchParallel) {
                if (kernelIdx >= resourceCtx.globalKernelDieIds.size()) {
                    return HCCL_E_INTERNAL;
                }
                threadIdx = resourceCtx.globalKernelDieIds[kernelIdx] == 0 ? 0U : 1U;
            }
            const CcuResult launchResult = HcommCcuKernelLaunch(resourceCtx.threads[threadIdx],
                (*kernelSet)[kernelIdx], taskArgs.data(), taskArgs.size());
            if (launchResult != CCU_SUCCESS) {
                HCCL_ERROR("CCU kernel launch failed, result[%d]", launchResult);
                return ConvertCcuResult(launchResult);
            }
        }
        if (launchParallel && usesSlave) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                resourceCtx.threads[1], resourceCtx.threads[0], 0)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                resourceCtx.threads[0], 0, CUSTOM_TIMEOUT)));
        }
    }

    if (sequence.reduceScratchResults && sequence.parallelGlobal) {
        if (resourceCtx.globalResultSlots.empty() || resourceCtx.globalKernels.empty()) {
            return HCCL_E_INTERNAL;
        }
        taskArgs[KERNEL_MODE_ARG_INDEX] = KERNEL_MODE_FINALIZE;
        const CcuResult finalizeResult = HcommCcuKernelLaunch(resourceCtx.threads[0],
            resourceCtx.globalKernels[0], taskArgs.data(), taskArgs.size());
        if (finalizeResult != CCU_SUCCESS) {
            HCCL_ERROR("CCU finalize kernel launch failed, result[%d]", finalizeResult);
            return ConvertCcuResult(finalizeResult);
        }
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param)
{
    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> sequenceData(ctx, ctx + param.ctxSize);
    AlgResourceCtx resourceCtx;
    resourceCtx.DeSerialize(sequenceData);

    if (param.dataType != HCCL_DATA_TYPE_FP32 || param.reduceType != HCCL_REDUCE_SUM) {
        return HCCL_E_NOT_SUPPORT;
    }
    if (param.rankSize == 0 || param.myRank >= param.rankSize || resourceCtx.threads.empty()) {
        return HCCL_E_PARA;
    }
    if (param.count > std::numeric_limits<uint64_t>::max() / sizeof(float)) {
        return HCCL_E_PARA;
    }

    const uint64_t recvBytes = param.count * sizeof(float);
    if (recvBytes == 0) {
        return HCCL_SUCCESS;
    }
    if (param.rankSize == 1) {
        return static_cast<HcclResult>(HcommLocalCopyOnThread(resourceCtx.threads[0], param.outputPtr,
            param.inputPtr, recvBytes));
    }
    if (recvBytes > std::numeric_limits<uint64_t>::max() / param.rankSize) {
        return HCCL_E_PARA;
    }
    const uint64_t inputBytes = recvBytes * param.rankSize;

    const KernelSequence kernelSequence = SelectKernelSequence(resourceCtx, inputBytes);
    if (kernelSequence.kernelSets.empty()) {
        HCCL_ERROR("No CCU kernel sequence selected");
        return HCCL_E_INTERNAL;
    }
    const uint32_t maxPieceCount = GetMaxPieceCount(kernelSequence);
    if (maxPieceCount == 0 || resourceCtx.localBuffer.addr == nullptr || resourceCtx.localBuffer.size == 0) {
        return HCCL_E_MEMORY;
    }

    uint64_t chunkLimit = std::min<uint64_t>(MAX_DATA_SIZE, resourceCtx.localBuffer.size / maxPieceCount);
    chunkLimit = chunkLimit / SCRATCH_ALIGNMENT * SCRATCH_ALIGNMENT;
    if (chunkLimit == 0) {
        chunkLimit = resourceCtx.localBuffer.size / maxPieceCount;
        chunkLimit = chunkLimit / sizeof(float) * sizeof(float);
    }
    if (chunkLimit == 0) {
        return HCCL_E_MEMORY;
    }

    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    uint64_t scratchToken = resourceCtx.localBufferToken;
    CHK_RET(GetToken(reinterpret_cast<uint64_t>(param.inputPtr), inputBytes, inputToken));
    CHK_RET(GetToken(reinterpret_cast<uint64_t>(param.outputPtr), recvBytes, outputToken));
    if (inputBytes > SMALL_INPUT_BYTES || scratchToken == 0) {
        CHK_RET(GetToken(reinterpret_cast<uint64_t>(resourceCtx.localBuffer.addr), resourceCtx.localBuffer.size,
            scratchToken));
    }

    for (uint64_t chunkOffset = 0; chunkOffset < recvBytes; chunkOffset += chunkLimit) {
        const uint64_t chunkBytes = std::min(chunkLimit, recvBytes - chunkOffset);
        CHK_RET(LaunchSequence(param, resourceCtx, kernelSequence, recvBytes, chunkOffset, chunkBytes,
            inputToken, outputToken, scratchToken));
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
} // namespace final_scratch_lite

namespace final_epoch_cache {
namespace ops_hccl {
namespace {
constexpr uint64_t SMALL_INPUT_BYTES = 1ULL * 1024 * 1024;
constexpr uint64_t SCRATCH_ALIGNMENT = 256;
constexpr uint64_t KERNEL_MODE_PARTIAL = 0;
constexpr uint64_t KERNEL_MODE_FINALIZE = 1;
constexpr uint64_t KERNEL_MODE_DIRECT_OUTPUT = 2;
constexpr uint32_t KERNEL_MODE_ARG_INDEX = 9;

struct KernelSequence {
    std::vector<const std::vector<CcuKernelHandle> *> kernelSets;
    std::vector<const std::vector<uint32_t> *> pieceSets;
    bool reduceScratchResults = false;
    bool parallelGlobal = false;
    uint32_t requiredScratchSlots = 0;
};

HcclResult ConvertCcuResult(CcuResult result)
{
    switch (result) {
        case CCU_SUCCESS:
            return HCCL_SUCCESS;
        case CCU_E_PARA:
            return HCCL_E_PARA;
        case CCU_E_PTR:
            return HCCL_E_PTR;
        case CCU_E_NOT_SUPPORT:
            return HCCL_E_NOT_SUPPORT;
        case CCU_E_NOT_FOUND:
            return HCCL_E_NOT_FOUND;
        case CCU_E_UNAVAIL:
            return HCCL_E_UNAVAIL;
        default:
            return HCCL_E_INTERNAL;
    }
}

void SelectGlobal(const AlgResourceCtx &resourceCtx, KernelSequence &sequence, bool parallel)
{
    sequence.kernelSets.push_back(&resourceCtx.globalKernels);
    sequence.pieceSets.push_back(&resourceCtx.globalPieceCounts);
    sequence.reduceScratchResults = true;
    sequence.parallelGlobal = parallel && resourceCtx.globalKernels.size() > 1 && resourceCtx.threads.size() > 1;
    sequence.requiredScratchSlots = resourceCtx.globalScratchSlotCount;
}

void SelectHierarchical(const AlgResourceCtx &resourceCtx, KernelSequence &sequence)
{
    if (!resourceCtx.localKernels.empty()) {
        sequence.kernelSets.push_back(&resourceCtx.localKernels);
        sequence.pieceSets.push_back(&resourceCtx.localPieceCounts);
    }
    if (!resourceCtx.remoteKernels.empty()) {
        sequence.kernelSets.push_back(&resourceCtx.remoteKernels);
        sequence.pieceSets.push_back(&resourceCtx.remotePieceCounts);
    }
}

KernelSequence SelectKernelSequence(const AlgResourceCtx &resourceCtx, uint64_t inputBytes)
{
    KernelSequence sequence;
    const auto topologyKind = static_cast<Rs2TopologyKind>(resourceCtx.topologyKind);
    switch (RS2_BUILD_FLAVOR) {
        case Rs2BuildFlavor::HIERARCHICAL:
            SelectHierarchical(resourceCtx, sequence);
            break;
        case Rs2BuildFlavor::HCCL_TOPOLOGY:
            SelectGlobal(resourceCtx, sequence, true);
            break;
        case Rs2BuildFlavor::GLOBAL_MESH_TREE:
            SelectGlobal(resourceCtx, sequence, true);
            break;
        case Rs2BuildFlavor::SCRATCH_LITE:
        case Rs2BuildFlavor::EPOCH_CACHE:
            SelectGlobal(resourceCtx, sequence, true);
            break;
        case Rs2BuildFlavor::ADAPTIVE:
            if (inputBytes <= SMALL_INPUT_BYTES || topologyKind == Rs2TopologyKind::FOUR_BY_ONE ||
                topologyKind == Rs2TopologyKind::UNKNOWN) {
                SelectGlobal(resourceCtx, sequence, false);
            } else {
                SelectGlobal(resourceCtx, sequence, true);
            }
            break;
        default:
            break;
    }
    return sequence;
}

uint32_t GetMaxPieceCount(const KernelSequence &sequence)
{
    if (sequence.requiredScratchSlots != 0) {
        return sequence.requiredScratchSlots;
    }
    uint32_t maxPieceCount = 0;
    for (const std::vector<uint32_t> *pieceSet : sequence.pieceSets) {
        for (uint32_t pieceCount : *pieceSet) {
            maxPieceCount = std::max(maxPieceCount, pieceCount);
        }
    }
    return maxPieceCount;
}

HcclResult GetToken(uint64_t address, uint64_t bytes, uint64_t &token)
{
    const CcuResult result = HcommCcuGetMemToken(address, bytes, &token);
    if (result != CCU_SUCCESS) {
        HCCL_ERROR("HcommCcuGetMemToken failed, result[%d], bytes[%llu]", result,
            static_cast<unsigned long long>(bytes));
        return ConvertCcuResult(result);
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchSequence(const OpParam &param, const AlgResourceCtx &resourceCtx,
    const KernelSequence &sequence, uint64_t recvBytes, uint64_t chunkOffset, uint64_t chunkBytes,
    uint64_t inputToken, uint64_t outputToken, uint64_t scratchToken)
{
    const uint64_t inputAddress = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outputAddress = reinterpret_cast<uint64_t>(param.outputPtr);
    const uint64_t scratchAddress = reinterpret_cast<uint64_t>(resourceCtx.localBuffer.addr);
    const uint64_t targetInputOffset = static_cast<uint64_t>(param.myRank) * recvBytes + chunkOffset;
    if (resourceCtx.globalResultSlots.size() > MAX_GLOBAL_RESULT_COUNT) {
        return HCCL_E_INTERNAL;
    }
    std::vector<uint64_t> taskArgs = {inputAddress, outputAddress, inputToken, outputToken, scratchAddress,
        scratchToken, targetInputOffset, chunkOffset, chunkBytes, KERNEL_MODE_PARTIAL,
        resourceCtx.globalResultSlots.size()};
    for (uint32_t resultIdx = 0; resultIdx < MAX_GLOBAL_RESULT_COUNT; ++resultIdx) {
        const uint64_t resultOffset = resultIdx < resourceCtx.globalResultSlots.size() ?
            static_cast<uint64_t>(resourceCtx.globalResultSlots[resultIdx]) * chunkBytes : 0;
        taskArgs.push_back(resultOffset);
    }
    for (const std::vector<CcuKernelHandle> *kernelSet : sequence.kernelSets) {
        const bool launchParallel = sequence.parallelGlobal && kernelSet == &resourceCtx.globalKernels;
        taskArgs[KERNEL_MODE_ARG_INDEX] = sequence.reduceScratchResults && !launchParallel ?
            KERNEL_MODE_DIRECT_OUTPUT : KERNEL_MODE_PARTIAL;
        bool usesSlave = false;
        if (launchParallel) {
            for (uint32_t dieId : resourceCtx.globalKernelDieIds) {
                usesSlave = usesSlave || dieId != 0;
            }
            if (usesSlave) {
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                    resourceCtx.threads[0], resourceCtx.threads[1], 0)));
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                    resourceCtx.threads[1], 0, CUSTOM_TIMEOUT)));
            }
        }

        std::vector<uint32_t> launchOrder(kernelSet->size());
        for (uint32_t kernelIdx = 0; kernelIdx < kernelSet->size(); ++kernelIdx) {
            launchOrder[kernelIdx] = kernelIdx;
        }
        const bool smallInput = recvBytes <= SMALL_INPUT_BYTES / param.rankSize;
        if (smallInput && kernelSet == &resourceCtx.globalKernels &&
            resourceCtx.globalPieceCounts.size() == launchOrder.size()) {
            std::stable_sort(launchOrder.begin(), launchOrder.end(), [&](uint32_t lhs, uint32_t rhs) {
                return resourceCtx.globalPieceCounts[lhs] > resourceCtx.globalPieceCounts[rhs];
            });
        }

        for (uint32_t kernelIdx : launchOrder) {
            uint32_t threadIdx = 0;
            if (launchParallel) {
                if (kernelIdx >= resourceCtx.globalKernelDieIds.size()) {
                    return HCCL_E_INTERNAL;
                }
                threadIdx = resourceCtx.globalKernelDieIds[kernelIdx] == 0 ? 0U : 1U;
            }
            const CcuResult launchResult = HcommCcuKernelLaunch(resourceCtx.threads[threadIdx],
                (*kernelSet)[kernelIdx], taskArgs.data(), taskArgs.size());
            if (launchResult != CCU_SUCCESS) {
                HCCL_ERROR("CCU kernel launch failed, result[%d]", launchResult);
                return ConvertCcuResult(launchResult);
            }
        }
        if (launchParallel && usesSlave) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                resourceCtx.threads[1], resourceCtx.threads[0], 0)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                resourceCtx.threads[0], 0, CUSTOM_TIMEOUT)));
        }
    }

    if (sequence.reduceScratchResults && sequence.parallelGlobal) {
        if (resourceCtx.globalResultSlots.empty() || resourceCtx.globalKernels.empty()) {
            return HCCL_E_INTERNAL;
        }
        taskArgs[KERNEL_MODE_ARG_INDEX] = KERNEL_MODE_FINALIZE;
        const CcuResult finalizeResult = HcommCcuKernelLaunch(resourceCtx.threads[0],
            resourceCtx.globalKernels[0], taskArgs.data(), taskArgs.size());
        if (finalizeResult != CCU_SUCCESS) {
            HCCL_ERROR("CCU finalize kernel launch failed, result[%d]", finalizeResult);
            return ConvertCcuResult(finalizeResult);
        }
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param)
{
    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> sequenceData(ctx, ctx + param.ctxSize);
    AlgResourceCtx resourceCtx;
    resourceCtx.DeSerialize(sequenceData);

    if (param.dataType != HCCL_DATA_TYPE_FP32 || param.reduceType != HCCL_REDUCE_SUM) {
        return HCCL_E_NOT_SUPPORT;
    }
    if (param.rankSize == 0 || param.myRank >= param.rankSize || resourceCtx.threads.empty()) {
        return HCCL_E_PARA;
    }
    if (param.count > std::numeric_limits<uint64_t>::max() / sizeof(float)) {
        return HCCL_E_PARA;
    }

    const uint64_t recvBytes = param.count * sizeof(float);
    if (recvBytes == 0) {
        return HCCL_SUCCESS;
    }
    if (param.rankSize == 1) {
        return static_cast<HcclResult>(HcommLocalCopyOnThread(resourceCtx.threads[0], param.outputPtr,
            param.inputPtr, recvBytes));
    }
    if (recvBytes > std::numeric_limits<uint64_t>::max() / param.rankSize) {
        return HCCL_E_PARA;
    }
    const uint64_t inputBytes = recvBytes * param.rankSize;

    const KernelSequence kernelSequence = SelectKernelSequence(resourceCtx, inputBytes);
    if (kernelSequence.kernelSets.empty()) {
        HCCL_ERROR("No CCU kernel sequence selected");
        return HCCL_E_INTERNAL;
    }
    const uint32_t maxPieceCount = GetMaxPieceCount(kernelSequence);
    if (maxPieceCount == 0 || resourceCtx.localBuffer.addr == nullptr || resourceCtx.localBuffer.size == 0) {
        return HCCL_E_MEMORY;
    }

    uint64_t chunkLimit = std::min<uint64_t>(MAX_DATA_SIZE, resourceCtx.localBuffer.size / maxPieceCount);
    chunkLimit = chunkLimit / SCRATCH_ALIGNMENT * SCRATCH_ALIGNMENT;
    if (chunkLimit == 0) {
        chunkLimit = resourceCtx.localBuffer.size / maxPieceCount;
        chunkLimit = chunkLimit / sizeof(float) * sizeof(float);
    }
    if (chunkLimit == 0) {
        return HCCL_E_MEMORY;
    }

    uint64_t inputToken = inputBytes <= SMALL_INPUT_BYTES ? resourceCtx.inputToken : 0;
    uint64_t outputToken = inputBytes <= SMALL_INPUT_BYTES ? resourceCtx.outputToken : 0;
    uint64_t scratchToken = resourceCtx.localBufferToken;
    if (inputToken == 0) {
        CHK_RET(GetToken(reinterpret_cast<uint64_t>(param.inputPtr), inputBytes, inputToken));
    }
    if (outputToken == 0) {
        CHK_RET(GetToken(reinterpret_cast<uint64_t>(param.outputPtr), recvBytes, outputToken));
    }
    if (inputBytes > SMALL_INPUT_BYTES || scratchToken == 0) {
        CHK_RET(GetToken(reinterpret_cast<uint64_t>(resourceCtx.localBuffer.addr), resourceCtx.localBuffer.size,
            scratchToken));
    }

    for (uint64_t chunkOffset = 0; chunkOffset < recvBytes; chunkOffset += chunkLimit) {
        const uint64_t chunkBytes = std::min(chunkLimit, recvBytes - chunkOffset);
        CHK_RET(LaunchSequence(param, resourceCtx, kernelSequence, recvBytes, chunkOffset, chunkBytes,
            inputToken, outputToken, scratchToken));
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
} // namespace final_epoch_cache

namespace final_direct_pipeline {
namespace ops_hccl {
namespace {
constexpr uint64_t SMALL_INPUT_BYTES = 1ULL * 1024 * 1024;
constexpr uint64_t DIRECT_PIPELINE_MIN_INPUT_BYTES = 480ULL * 1024 * 1024;
constexpr uint64_t SCRATCH_ALIGNMENT = 256;
constexpr uint64_t KERNEL_MODE_PARTIAL = 0;
constexpr uint64_t KERNEL_MODE_FINALIZE = 1;
constexpr uint64_t KERNEL_MODE_DIRECT_OUTPUT = 2;
constexpr uint32_t KERNEL_MODE_ARG_INDEX = 9;
constexpr uint32_t DIRECT_PIPELINE_CHUNK_COUNT = 2;
constexpr uint32_t PIPELINE_START_NOTIFY_ID = 0;
constexpr uint32_t PIPELINE_DIRECT_DONE_NOTIFY_BASE = 1;
constexpr uint32_t PIPELINE_FINAL_DONE_NOTIFY_ID = 3;

struct KernelSequence {
    std::vector<const std::vector<CcuKernelHandle> *> kernelSets;
    std::vector<const std::vector<uint32_t> *> pieceSets;
    bool reduceScratchResults = false;
    bool parallelGlobal = false;
    uint32_t requiredScratchSlots = 0;
};

HcclResult ConvertCcuResult(CcuResult result)
{
    switch (result) {
        case CCU_SUCCESS:
            return HCCL_SUCCESS;
        case CCU_E_PARA:
            return HCCL_E_PARA;
        case CCU_E_PTR:
            return HCCL_E_PTR;
        case CCU_E_NOT_SUPPORT:
            return HCCL_E_NOT_SUPPORT;
        case CCU_E_NOT_FOUND:
            return HCCL_E_NOT_FOUND;
        case CCU_E_UNAVAIL:
            return HCCL_E_UNAVAIL;
        default:
            return HCCL_E_INTERNAL;
    }
}

void SelectGlobal(const AlgResourceCtx &resourceCtx, KernelSequence &sequence, bool parallel)
{
    sequence.kernelSets.push_back(&resourceCtx.globalKernels);
    sequence.pieceSets.push_back(&resourceCtx.globalPieceCounts);
    sequence.reduceScratchResults = true;
    sequence.parallelGlobal = parallel && resourceCtx.globalKernels.size() > 1 && resourceCtx.threads.size() > 1;
    sequence.requiredScratchSlots = resourceCtx.globalScratchSlotCount;
}

void SelectHierarchical(const AlgResourceCtx &resourceCtx, KernelSequence &sequence)
{
    if (!resourceCtx.localKernels.empty()) {
        sequence.kernelSets.push_back(&resourceCtx.localKernels);
        sequence.pieceSets.push_back(&resourceCtx.localPieceCounts);
    }
    if (!resourceCtx.remoteKernels.empty()) {
        sequence.kernelSets.push_back(&resourceCtx.remoteKernels);
        sequence.pieceSets.push_back(&resourceCtx.remotePieceCounts);
    }
}

KernelSequence SelectKernelSequence(const AlgResourceCtx &resourceCtx, uint64_t inputBytes)
{
    KernelSequence sequence;
    const auto topologyKind = static_cast<Rs2TopologyKind>(resourceCtx.topologyKind);
    switch (RS2_BUILD_FLAVOR) {
        case Rs2BuildFlavor::HIERARCHICAL:
            SelectHierarchical(resourceCtx, sequence);
            break;
        case Rs2BuildFlavor::HCCL_TOPOLOGY:
            SelectGlobal(resourceCtx, sequence, true);
            break;
        case Rs2BuildFlavor::GLOBAL_MESH_TREE:
            SelectGlobal(resourceCtx, sequence, true);
            break;
        case Rs2BuildFlavor::SCRATCH_LITE:
        case Rs2BuildFlavor::EPOCH_CACHE:
        case Rs2BuildFlavor::DIRECT_PIPELINE:
            SelectGlobal(resourceCtx, sequence, true);
            break;
        case Rs2BuildFlavor::ADAPTIVE:
            if (inputBytes <= SMALL_INPUT_BYTES || topologyKind == Rs2TopologyKind::FOUR_BY_ONE ||
                topologyKind == Rs2TopologyKind::UNKNOWN) {
                SelectGlobal(resourceCtx, sequence, false);
            } else {
                SelectGlobal(resourceCtx, sequence, true);
            }
            break;
        default:
            break;
    }
    return sequence;
}

uint32_t GetMaxPieceCount(const KernelSequence &sequence)
{
    if (sequence.requiredScratchSlots != 0) {
        return sequence.requiredScratchSlots;
    }
    uint32_t maxPieceCount = 0;
    for (const std::vector<uint32_t> *pieceSet : sequence.pieceSets) {
        for (uint32_t pieceCount : *pieceSet) {
            maxPieceCount = std::max(maxPieceCount, pieceCount);
        }
    }
    return maxPieceCount;
}

HcclResult GetToken(uint64_t address, uint64_t bytes, uint64_t &token)
{
    const CcuResult result = HcommCcuGetMemToken(address, bytes, &token);
    if (result != CCU_SUCCESS) {
        HCCL_ERROR("HcommCcuGetMemToken failed, result[%d], bytes[%llu]", result,
            static_cast<unsigned long long>(bytes));
        return ConvertCcuResult(result);
    }
    return HCCL_SUCCESS;
}

HcclResult BuildTaskArgs(const OpParam &param, const AlgResourceCtx &resourceCtx,
    uint64_t recvBytes, uint64_t chunkOffset, uint64_t chunkBytes, uint64_t inputToken,
    uint64_t outputToken, uint64_t scratchToken, std::vector<uint64_t> &taskArgs)
{
    if (resourceCtx.globalResultSlots.size() > MAX_GLOBAL_RESULT_COUNT) {
        return HCCL_E_INTERNAL;
    }
    const uint64_t targetInputOffset = static_cast<uint64_t>(param.myRank) * recvBytes + chunkOffset;
    taskArgs = {reinterpret_cast<uint64_t>(param.inputPtr), reinterpret_cast<uint64_t>(param.outputPtr),
        inputToken, outputToken, reinterpret_cast<uint64_t>(resourceCtx.localBuffer.addr), scratchToken,
        targetInputOffset, chunkOffset, chunkBytes, KERNEL_MODE_PARTIAL, resourceCtx.globalResultSlots.size()};
    for (uint32_t resultIdx = 0; resultIdx < MAX_GLOBAL_RESULT_COUNT; ++resultIdx) {
        const uint64_t resultOffset = resultIdx < resourceCtx.globalResultSlots.size() ?
            static_cast<uint64_t>(resourceCtx.globalResultSlots[resultIdx]) * chunkBytes : 0;
        taskArgs.push_back(resultOffset);
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchKernel(ThreadHandle thread, CcuKernelHandle kernel, const std::vector<uint64_t> &taskArgs,
    const char *description)
{
    const CcuResult result = HcommCcuKernelLaunch(thread, kernel, taskArgs.data(), taskArgs.size());
    if (result != CCU_SUCCESS) {
        HCCL_ERROR("%s failed, result[%d]", description, result);
        return ConvertCcuResult(result);
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchSequence(const OpParam &param, const AlgResourceCtx &resourceCtx,
    const KernelSequence &sequence, uint64_t recvBytes, uint64_t chunkOffset, uint64_t chunkBytes,
    uint64_t inputToken, uint64_t outputToken, uint64_t scratchToken)
{
    std::vector<uint64_t> taskArgs;
    CHK_RET(BuildTaskArgs(param, resourceCtx, recvBytes, chunkOffset, chunkBytes,
        inputToken, outputToken, scratchToken, taskArgs));
    for (const std::vector<CcuKernelHandle> *kernelSet : sequence.kernelSets) {
        const bool launchParallel = sequence.parallelGlobal && kernelSet == &resourceCtx.globalKernels;
        taskArgs[KERNEL_MODE_ARG_INDEX] = sequence.reduceScratchResults && !launchParallel ?
            KERNEL_MODE_DIRECT_OUTPUT : KERNEL_MODE_PARTIAL;
        bool usesSlave = false;
        if (launchParallel) {
            for (uint32_t dieId : resourceCtx.globalKernelDieIds) {
                usesSlave = usesSlave || dieId != 0;
            }
            if (usesSlave) {
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                    resourceCtx.threads[0], resourceCtx.threads[1], 0)));
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                    resourceCtx.threads[1], 0, CUSTOM_TIMEOUT)));
            }
        }

        std::vector<uint32_t> launchOrder(kernelSet->size());
        for (uint32_t kernelIdx = 0; kernelIdx < kernelSet->size(); ++kernelIdx) {
            launchOrder[kernelIdx] = kernelIdx;
        }
        const bool smallInput = recvBytes <= SMALL_INPUT_BYTES / param.rankSize;
        if (smallInput && kernelSet == &resourceCtx.globalKernels &&
            resourceCtx.globalPieceCounts.size() == launchOrder.size()) {
            std::stable_sort(launchOrder.begin(), launchOrder.end(), [&](uint32_t lhs, uint32_t rhs) {
                return resourceCtx.globalPieceCounts[lhs] > resourceCtx.globalPieceCounts[rhs];
            });
        }

        for (uint32_t kernelIdx : launchOrder) {
            uint32_t threadIdx = 0;
            if (launchParallel) {
                if (kernelIdx >= resourceCtx.globalKernelDieIds.size()) {
                    return HCCL_E_INTERNAL;
                }
                threadIdx = resourceCtx.globalKernelDieIds[kernelIdx] == 0 ? 0U : 1U;
            }
            CHK_RET(LaunchKernel(resourceCtx.threads[threadIdx], (*kernelSet)[kernelIdx],
                taskArgs, "CCU kernel launch"));
        }
        if (launchParallel && usesSlave) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                resourceCtx.threads[1], resourceCtx.threads[0], 0)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                resourceCtx.threads[0], 0, CUSTOM_TIMEOUT)));
        }
    }

    if (sequence.reduceScratchResults && sequence.parallelGlobal) {
        if (resourceCtx.globalResultSlots.empty() || resourceCtx.globalKernels.empty()) {
            return HCCL_E_INTERNAL;
        }
        taskArgs[KERNEL_MODE_ARG_INDEX] = KERNEL_MODE_FINALIZE;
        CHK_RET(LaunchKernel(resourceCtx.threads[0], resourceCtx.globalKernels[0],
            taskArgs, "CCU finalize kernel launch"));
    }
    return HCCL_SUCCESS;
}

bool CanLaunchDirectPipeline(const AlgResourceCtx &resourceCtx, uint64_t inputBytes,
    uint64_t recvBytes, uint64_t chunkLimit)
{
    const auto topologyKind = static_cast<Rs2TopologyKind>(resourceCtx.topologyKind);
    const bool supportedTopology = topologyKind == Rs2TopologyKind::TWO_BY_EIGHT ||
        topologyKind == Rs2TopologyKind::EIGHT_PLUS_FOUR;
    if (RS2_BUILD_FLAVOR != Rs2BuildFlavor::DIRECT_PIPELINE || !supportedTopology ||
        inputBytes < DIRECT_PIPELINE_MIN_INPUT_BYTES || recvBytes <= chunkLimit ||
        recvBytes > DIRECT_PIPELINE_CHUNK_COUNT * chunkLimit) {
        return false;
    }
    return resourceCtx.threads.size() >= 2 && resourceCtx.globalKernels.size() == 2 &&
        resourceCtx.globalKernelDieIds.size() == 2 &&
        (resourceCtx.globalKernelDieIds[0] == 0) != (resourceCtx.globalKernelDieIds[1] == 0) &&
        resourceCtx.globalResultSlots.size() == 1;
}

HcclResult LaunchDirectPipeline(const OpParam &param, const AlgResourceCtx &resourceCtx,
    uint64_t recvBytes, uint64_t inputToken, uint64_t outputToken, uint64_t scratchToken)
{
    const uint64_t recvElements = recvBytes / sizeof(float);
    const uint64_t firstChunkElements = (recvElements + 1) / 2;
    const uint64_t chunkOffsets[DIRECT_PIPELINE_CHUNK_COUNT] = {
        0,
        firstChunkElements * sizeof(float),
    };
    const uint64_t chunkBytes[DIRECT_PIPELINE_CHUNK_COUNT] = {
        chunkOffsets[1],
        recvBytes - chunkOffsets[1],
    };
    if (chunkBytes[0] == 0 || chunkBytes[1] == 0) {
        return HCCL_E_PARA;
    }

    const uint32_t directThreadIdx = resourceCtx.globalKernelDieIds[0] == 0 ? 0U : 1U;
    const uint32_t partialThreadIdx = resourceCtx.globalKernelDieIds[1] == 0 ? 0U : 1U;

    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        resourceCtx.threads[0], resourceCtx.threads[1], PIPELINE_START_NOTIFY_ID)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        resourceCtx.threads[1], PIPELINE_START_NOTIFY_ID, CUSTOM_TIMEOUT)));

    for (uint32_t chunkIdx = 0; chunkIdx < DIRECT_PIPELINE_CHUNK_COUNT; ++chunkIdx) {
        std::vector<uint64_t> taskArgs;
        CHK_RET(BuildTaskArgs(param, resourceCtx, recvBytes, chunkOffsets[chunkIdx], chunkBytes[chunkIdx],
            inputToken, outputToken, scratchToken, taskArgs));
        CHK_RET(LaunchKernel(resourceCtx.threads[directThreadIdx], resourceCtx.globalKernels[0],
            taskArgs, "CCU direct pipeline kernel launch"));
        CHK_RET(LaunchKernel(resourceCtx.threads[partialThreadIdx], resourceCtx.globalKernels[1],
            taskArgs, "CCU partial pipeline kernel launch"));

        const uint32_t directDoneNotifyId = PIPELINE_DIRECT_DONE_NOTIFY_BASE + chunkIdx;
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resourceCtx.threads[directThreadIdx], resourceCtx.threads[partialThreadIdx], directDoneNotifyId)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            resourceCtx.threads[partialThreadIdx], directDoneNotifyId, CUSTOM_TIMEOUT)));

        taskArgs[KERNEL_MODE_ARG_INDEX] = KERNEL_MODE_FINALIZE;
        CHK_RET(LaunchKernel(resourceCtx.threads[partialThreadIdx], resourceCtx.globalKernels[1],
            taskArgs, "CCU pipeline finalize kernel launch"));
    }

    if (partialThreadIdx != 0) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resourceCtx.threads[partialThreadIdx], resourceCtx.threads[0], PIPELINE_FINAL_DONE_NOTIFY_ID)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            resourceCtx.threads[0], PIPELINE_FINAL_DONE_NOTIFY_ID, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult ExecOp(const OpParam &param)
{
    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> sequenceData(ctx, ctx + param.ctxSize);
    AlgResourceCtx resourceCtx;
    resourceCtx.DeSerialize(sequenceData);

    if (param.dataType != HCCL_DATA_TYPE_FP32 || param.reduceType != HCCL_REDUCE_SUM) {
        return HCCL_E_NOT_SUPPORT;
    }
    if (param.rankSize == 0 || param.myRank >= param.rankSize || resourceCtx.threads.empty()) {
        return HCCL_E_PARA;
    }
    if (param.count > std::numeric_limits<uint64_t>::max() / sizeof(float)) {
        return HCCL_E_PARA;
    }

    const uint64_t recvBytes = param.count * sizeof(float);
    if (recvBytes == 0) {
        return HCCL_SUCCESS;
    }
    if (param.rankSize == 1) {
        return static_cast<HcclResult>(HcommLocalCopyOnThread(resourceCtx.threads[0], param.outputPtr,
            param.inputPtr, recvBytes));
    }
    if (recvBytes > std::numeric_limits<uint64_t>::max() / param.rankSize) {
        return HCCL_E_PARA;
    }
    const uint64_t inputBytes = recvBytes * param.rankSize;

    const KernelSequence kernelSequence = SelectKernelSequence(resourceCtx, inputBytes);
    if (kernelSequence.kernelSets.empty()) {
        HCCL_ERROR("No CCU kernel sequence selected");
        return HCCL_E_INTERNAL;
    }
    const uint32_t maxPieceCount = GetMaxPieceCount(kernelSequence);
    if (maxPieceCount == 0 || resourceCtx.localBuffer.addr == nullptr || resourceCtx.localBuffer.size == 0) {
        return HCCL_E_MEMORY;
    }

    uint64_t chunkLimit = std::min<uint64_t>(MAX_DATA_SIZE, resourceCtx.localBuffer.size / maxPieceCount);
    chunkLimit = chunkLimit / SCRATCH_ALIGNMENT * SCRATCH_ALIGNMENT;
    if (chunkLimit == 0) {
        chunkLimit = resourceCtx.localBuffer.size / maxPieceCount;
        chunkLimit = chunkLimit / sizeof(float) * sizeof(float);
    }
    if (chunkLimit == 0) {
        return HCCL_E_MEMORY;
    }

    uint64_t inputToken = inputBytes <= SMALL_INPUT_BYTES ? resourceCtx.inputToken : 0;
    uint64_t outputToken = inputBytes <= SMALL_INPUT_BYTES ? resourceCtx.outputToken : 0;
    uint64_t scratchToken = resourceCtx.localBufferToken;
    if (inputToken == 0) {
        CHK_RET(GetToken(reinterpret_cast<uint64_t>(param.inputPtr), inputBytes, inputToken));
    }
    if (outputToken == 0) {
        CHK_RET(GetToken(reinterpret_cast<uint64_t>(param.outputPtr), recvBytes, outputToken));
    }
    if (inputBytes > SMALL_INPUT_BYTES || scratchToken == 0) {
        CHK_RET(GetToken(reinterpret_cast<uint64_t>(resourceCtx.localBuffer.addr), resourceCtx.localBuffer.size,
            scratchToken));
    }

    if (CanLaunchDirectPipeline(resourceCtx, inputBytes, recvBytes, chunkLimit)) {
        return LaunchDirectPipeline(param, resourceCtx, recvBytes, inputToken, outputToken, scratchToken);
    }

    for (uint64_t chunkOffset = 0; chunkOffset < recvBytes; chunkOffset += chunkLimit) {
        const uint64_t chunkBytes = std::min(chunkLimit, recvBytes - chunkOffset);
        CHK_RET(LaunchSequence(param, resourceCtx, kernelSequence, recvBytes, chunkOffset, chunkBytes,
            inputToken, outputToken, scratchToken));
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
} // namespace final_direct_pipeline

namespace final_adaptive41 {
namespace ops_hccl {
namespace {
constexpr uint64_t SMALL_INPUT_BYTES = 1ULL * 1024 * 1024;
constexpr uint64_t SCRATCH_ALIGNMENT = 256;
constexpr uint64_t KERNEL_MODE_PARTIAL = 0;
constexpr uint64_t KERNEL_MODE_FINALIZE = 1;
constexpr uint64_t KERNEL_MODE_DIRECT_OUTPUT = 2;
constexpr uint32_t KERNEL_MODE_ARG_INDEX = 9;

struct KernelSequence {
    std::vector<const std::vector<CcuKernelHandle> *> kernelSets;
    std::vector<const std::vector<uint32_t> *> pieceSets;
    bool reduceScratchResults = false;
    bool parallelGlobal = false;
    uint32_t requiredScratchSlots = 0;
};

HcclResult ConvertCcuResult(CcuResult result)
{
    switch (result) {
        case CCU_SUCCESS:
            return HCCL_SUCCESS;
        case CCU_E_PARA:
            return HCCL_E_PARA;
        case CCU_E_PTR:
            return HCCL_E_PTR;
        case CCU_E_NOT_SUPPORT:
            return HCCL_E_NOT_SUPPORT;
        case CCU_E_NOT_FOUND:
            return HCCL_E_NOT_FOUND;
        case CCU_E_UNAVAIL:
            return HCCL_E_UNAVAIL;
        default:
            return HCCL_E_INTERNAL;
    }
}

void SelectGlobal(const AlgResourceCtx &resourceCtx, KernelSequence &sequence, bool parallel)
{
    sequence.kernelSets.push_back(&resourceCtx.globalKernels);
    sequence.pieceSets.push_back(&resourceCtx.globalPieceCounts);
    sequence.reduceScratchResults = true;
    sequence.parallelGlobal = parallel && resourceCtx.globalKernels.size() > 1 && resourceCtx.threads.size() > 1;
    sequence.requiredScratchSlots = resourceCtx.globalScratchSlotCount;
}

void SelectHierarchical(const AlgResourceCtx &resourceCtx, KernelSequence &sequence)
{
    if (!resourceCtx.localKernels.empty()) {
        sequence.kernelSets.push_back(&resourceCtx.localKernels);
        sequence.pieceSets.push_back(&resourceCtx.localPieceCounts);
    }
    if (!resourceCtx.remoteKernels.empty()) {
        sequence.kernelSets.push_back(&resourceCtx.remoteKernels);
        sequence.pieceSets.push_back(&resourceCtx.remotePieceCounts);
    }
}

KernelSequence SelectKernelSequence(const AlgResourceCtx &resourceCtx, uint64_t inputBytes)
{
    KernelSequence sequence;
    const auto topologyKind = static_cast<Rs2TopologyKind>(resourceCtx.topologyKind);
    switch (RS2_BUILD_FLAVOR) {
        case Rs2BuildFlavor::HIERARCHICAL:
            SelectHierarchical(resourceCtx, sequence);
            break;
        case Rs2BuildFlavor::HCCL_TOPOLOGY:
            SelectGlobal(resourceCtx, sequence, true);
            break;
        case Rs2BuildFlavor::GLOBAL_MESH_TREE:
            SelectGlobal(resourceCtx, sequence, true);
            break;
        case Rs2BuildFlavor::ADAPTIVE:
        case Rs2BuildFlavor::FOUR_BY_ONE_RING:
            if (inputBytes <= SMALL_INPUT_BYTES || topologyKind == Rs2TopologyKind::FOUR_BY_ONE ||
                topologyKind == Rs2TopologyKind::UNKNOWN) {
                SelectGlobal(resourceCtx, sequence, false);
            } else {
                SelectGlobal(resourceCtx, sequence, true);
            }
            break;
        default:
            break;
    }
    return sequence;
}

uint32_t GetMaxPieceCount(const KernelSequence &sequence)
{
    if (sequence.requiredScratchSlots != 0) {
        return sequence.requiredScratchSlots;
    }
    uint32_t maxPieceCount = 0;
    for (const std::vector<uint32_t> *pieceSet : sequence.pieceSets) {
        for (uint32_t pieceCount : *pieceSet) {
            maxPieceCount = std::max(maxPieceCount, pieceCount);
        }
    }
    return maxPieceCount;
}

HcclResult GetToken(uint64_t address, uint64_t bytes, uint64_t &token)
{
    const CcuResult result = HcommCcuGetMemToken(address, bytes, &token);
    if (result != CCU_SUCCESS) {
        HCCL_ERROR("HcommCcuGetMemToken failed, result[%d], bytes[%llu]", result,
            static_cast<unsigned long long>(bytes));
        return ConvertCcuResult(result);
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchSequence(const OpParam &param, const AlgResourceCtx &resourceCtx,
    const KernelSequence &sequence, uint64_t recvBytes, uint64_t chunkOffset, uint64_t chunkBytes,
    uint64_t inputToken, uint64_t outputToken, uint64_t scratchToken)
{
    const uint64_t inputAddress = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outputAddress = reinterpret_cast<uint64_t>(param.outputPtr);
    const uint64_t scratchAddress = reinterpret_cast<uint64_t>(resourceCtx.localBuffer.addr);
    const uint64_t targetInputOffset = static_cast<uint64_t>(param.myRank) * recvBytes + chunkOffset;
    if (resourceCtx.globalResultSlots.size() > MAX_GLOBAL_RESULT_COUNT) {
        return HCCL_E_INTERNAL;
    }
    std::vector<uint64_t> taskArgs = {inputAddress, outputAddress, inputToken, outputToken, scratchAddress,
        scratchToken, targetInputOffset, chunkOffset, chunkBytes, KERNEL_MODE_PARTIAL,
        resourceCtx.globalResultSlots.size()};
    for (uint32_t resultIdx = 0; resultIdx < MAX_GLOBAL_RESULT_COUNT; ++resultIdx) {
        const uint64_t resultOffset = resultIdx < resourceCtx.globalResultSlots.size() ?
            static_cast<uint64_t>(resourceCtx.globalResultSlots[resultIdx]) * chunkBytes : 0;
        taskArgs.push_back(resultOffset);
    }

    for (const std::vector<CcuKernelHandle> *kernelSet : sequence.kernelSets) {
        const bool launchParallel = sequence.parallelGlobal && kernelSet == &resourceCtx.globalKernels;
        taskArgs[KERNEL_MODE_ARG_INDEX] = sequence.reduceScratchResults && !launchParallel ?
            KERNEL_MODE_DIRECT_OUTPUT : KERNEL_MODE_PARTIAL;
        bool usesSlave = false;
        if (launchParallel) {
            for (uint32_t dieId : resourceCtx.globalKernelDieIds) {
                usesSlave = usesSlave || dieId != 0;
            }
            if (usesSlave) {
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                    resourceCtx.threads[0], resourceCtx.threads[1], 0)));
                CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                    resourceCtx.threads[1], 0, CUSTOM_TIMEOUT)));
            }
        }

        for (uint32_t kernelIdx = 0; kernelIdx < kernelSet->size(); ++kernelIdx) {
            uint32_t threadIdx = 0;
            if (launchParallel) {
                if (kernelIdx >= resourceCtx.globalKernelDieIds.size()) {
                    return HCCL_E_INTERNAL;
                }
                threadIdx = resourceCtx.globalKernelDieIds[kernelIdx] == 0 ? 0U : 1U;
            }
            const CcuResult launchResult = HcommCcuKernelLaunch(resourceCtx.threads[threadIdx],
                (*kernelSet)[kernelIdx], taskArgs.data(), taskArgs.size());
            if (launchResult != CCU_SUCCESS) {
                HCCL_ERROR("CCU kernel launch failed, result[%d]", launchResult);
                return ConvertCcuResult(launchResult);
            }
        }
        if (launchParallel && usesSlave) {
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
                resourceCtx.threads[1], resourceCtx.threads[0], 0)));
            CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
                resourceCtx.threads[0], 0, CUSTOM_TIMEOUT)));
        }
    }

    if (sequence.reduceScratchResults && sequence.parallelGlobal) {
        if (resourceCtx.globalResultSlots.empty() || resourceCtx.globalKernels.empty()) {
            return HCCL_E_INTERNAL;
        }
        taskArgs[KERNEL_MODE_ARG_INDEX] = KERNEL_MODE_FINALIZE;
        const CcuResult finalizeResult = HcommCcuKernelLaunch(resourceCtx.threads[0],
            resourceCtx.globalKernels[0], taskArgs.data(), taskArgs.size());
        if (finalizeResult != CCU_SUCCESS) {
            HCCL_ERROR("CCU finalize kernel launch failed, result[%d]", finalizeResult);
            return ConvertCcuResult(finalizeResult);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchKernelOnDie(const AlgResourceCtx &resourceCtx, CcuKernelHandle kernelHandle, uint32_t dieId,
    std::vector<uint64_t> &taskArgs)
{
    if (kernelHandle == 0 || resourceCtx.threads.empty()) {
        return HCCL_E_INTERNAL;
    }
    const uint32_t threadIdx = dieId == 0 ? 0U : 1U;
    if (threadIdx >= resourceCtx.threads.size()) {
        return HCCL_E_INTERNAL;
    }
    if (threadIdx != 0) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resourceCtx.threads[0], resourceCtx.threads[threadIdx], 0)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            resourceCtx.threads[threadIdx], 0, CUSTOM_TIMEOUT)));
    }
    const CcuResult launchResult = HcommCcuKernelLaunch(resourceCtx.threads[threadIdx], kernelHandle,
        taskArgs.data(), taskArgs.size());
    if (launchResult != CCU_SUCCESS) {
        return ConvertCcuResult(launchResult);
    }
    if (threadIdx != 0) {
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
            resourceCtx.threads[threadIdx], resourceCtx.threads[0], 0)));
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
            resourceCtx.threads[0], 0, CUSTOM_TIMEOUT)));
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchRing(const OpParam &param, const AlgResourceCtx &resourceCtx, uint64_t recvBytes,
    uint64_t chunkOffset, uint64_t chunkBytes, uint64_t inputToken, uint64_t outputToken)
{
    uint64_t firstSliceBytes = chunkBytes / 2;
    firstSliceBytes = firstSliceBytes / sizeof(float) * sizeof(float);
    if (firstSliceBytes == 0) {
        firstSliceBytes = chunkBytes;
    }
    const uint64_t secondSliceBytes = chunkBytes - firstSliceBytes;
    std::vector<uint64_t> taskArgs = {
        reinterpret_cast<uint64_t>(param.inputPtr), reinterpret_cast<uint64_t>(param.outputPtr),
        inputToken, outputToken, recvBytes, chunkOffset, chunkBytes, firstSliceBytes, secondSliceBytes,
    };
    return LaunchKernelOnDie(resourceCtx, resourceCtx.ringKernel, resourceCtx.ringKernelDieId, taskArgs);
}
constexpr uint32_t EIGHT_PLUS_FOUR_SCRATCH_SLOTS = 4;
constexpr uint32_t EIGHT_PLUS_FOUR_GROUP_SIZE_EXEC = 4;

HcclResult LaunchEightPlusFourPreReduce(const OpParam &param, const AlgResourceCtx &resourceCtx,
    uint64_t recvBytes, uint64_t chunkOffset, uint64_t chunkBytes, uint64_t inputToken,
    uint64_t scratchToken)
{
    std::vector<uint64_t> taskArgs = {
        reinterpret_cast<uint64_t>(param.inputPtr), inputToken,
        reinterpret_cast<uint64_t>(resourceCtx.localBuffer.addr), scratchToken,
        recvBytes, chunkOffset, chunkBytes,
    };
    return LaunchKernelOnDie(resourceCtx, resourceCtx.eightPlusFourPreReduceKernel,
        resourceCtx.eightPlusFourPreReduceKernelDieId, taskArgs);
}

constexpr uint32_t EIGHT_PLUS_FOUR_RING_GROUP_COUNT = 3;
constexpr uint32_t EIGHT_PLUS_FOUR_EDGE_KERNEL_COUNT = 6;
constexpr uint32_t EIGHT_PLUS_FOUR_EDGE_INIT = 0;
constexpr uint32_t EIGHT_PLUS_FOUR_EDGE_SEND_STAGE_ZERO = 1;
constexpr uint32_t EIGHT_PLUS_FOUR_EDGE_WAIT_STAGE_ZERO = 2;
constexpr uint32_t EIGHT_PLUS_FOUR_EDGE_SEND_STAGE_ONE = 3;
constexpr uint32_t EIGHT_PLUS_FOUR_EDGE_WAIT_STAGE_ONE = 4;
constexpr uint32_t EIGHT_PLUS_FOUR_EDGE_COPY_OUTPUT = 5;
constexpr uint32_t EIGHT_PLUS_FOUR_FIRST_SLICE_ARG_INDEX = 7;
constexpr uint32_t EIGHT_PLUS_FOUR_SECOND_SLICE_ARG_INDEX = 8;
constexpr uint32_t EIGHT_PLUS_FOUR_BLOCK_OFFSET_ARG_INDEX = 9;

uint32_t GetEightPlusFourEdgeResourceIndex(uint32_t edgeIdx, uint32_t operationIdx)
{
    return edgeIdx * EIGHT_PLUS_FOUR_EDGE_KERNEL_COUNT + operationIdx;
}

bool HasEightPlusFourRingEdge(const AlgResourceCtx &resourceCtx, uint32_t edgeIdx)
{
    const uint32_t resourceIdx = GetEightPlusFourEdgeResourceIndex(edgeIdx, EIGHT_PLUS_FOUR_EDGE_INIT);
    return resourceIdx < resourceCtx.eightPlusFourRingEdgeKernels.size() &&
        resourceIdx < resourceCtx.eightPlusFourRingEdgeKernelDieIds.size() &&
        resourceCtx.eightPlusFourRingEdgeKernels[resourceIdx] != 0;
}

HcclResult LaunchEightPlusFourRingEdge(const AlgResourceCtx &resourceCtx, uint32_t edgeIdx,
    uint32_t operationIdx, std::vector<uint64_t> &taskArgs)
{
    if (edgeIdx >= EIGHT_PLUS_FOUR_RING_GROUP_COUNT ||
        operationIdx >= EIGHT_PLUS_FOUR_EDGE_KERNEL_COUNT) {
        return HCCL_E_PARA;
    }
    const uint32_t resourceIdx = GetEightPlusFourEdgeResourceIndex(edgeIdx, operationIdx);
    if (resourceIdx >= resourceCtx.eightPlusFourRingEdgeKernels.size() ||
        resourceIdx >= resourceCtx.eightPlusFourRingEdgeKernelDieIds.size() ||
        resourceCtx.eightPlusFourRingEdgeKernels[resourceIdx] == 0) {
        return HCCL_E_INTERNAL;
    }
    return LaunchKernelOnDie(resourceCtx, resourceCtx.eightPlusFourRingEdgeKernels[resourceIdx],
        resourceCtx.eightPlusFourRingEdgeKernelDieIds[resourceIdx], taskArgs);
}

std::vector<uint64_t> BuildEightPlusFourEdgeTaskArgs(const OpParam &param, uint64_t recvBytes,
    uint64_t chunkOffset, uint64_t chunkBytes, uint64_t inputToken, uint64_t outputToken)
{
    uint64_t firstSliceBytes = chunkBytes / 2;
    firstSliceBytes = firstSliceBytes / sizeof(float) * sizeof(float);
    if (firstSliceBytes == 0) {
        firstSliceBytes = chunkBytes;
    }
    const uint64_t secondSliceBytes = chunkBytes - firstSliceBytes;
    return {
        reinterpret_cast<uint64_t>(param.inputPtr), reinterpret_cast<uint64_t>(param.outputPtr),
        inputToken, outputToken, recvBytes, chunkOffset, chunkBytes, firstSliceBytes, secondSliceBytes, 0,
    };
}

void SetEightPlusFourBlockOffset(std::vector<uint64_t> &taskArgs, uint64_t recvBytes,
    uint64_t chunkOffset, uint32_t blockIdx)
{
    taskArgs[EIGHT_PLUS_FOUR_BLOCK_OFFSET_ARG_INDEX] =
        chunkOffset + static_cast<uint64_t>(blockIdx) * recvBytes;
}

HcclResult InitEightPlusFourRingEdges(const AlgResourceCtx &resourceCtx, std::vector<uint64_t> &taskArgs)
{
    for (uint32_t edgeIdx = 0; edgeIdx < EIGHT_PLUS_FOUR_RING_GROUP_COUNT; ++edgeIdx) {
        if (HasEightPlusFourRingEdge(resourceCtx, edgeIdx)) {
            CHK_RET(LaunchEightPlusFourRingEdge(resourceCtx, edgeIdx,
                EIGHT_PLUS_FOUR_EDGE_INIT, taskArgs));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult SignalEightPlusFourRingEdge(const AlgResourceCtx &resourceCtx, uint32_t edgeIdx,
    std::vector<uint64_t> &taskArgs)
{
    const uint64_t firstSliceBytes = taskArgs[EIGHT_PLUS_FOUR_FIRST_SLICE_ARG_INDEX];
    const uint64_t secondSliceBytes = taskArgs[EIGHT_PLUS_FOUR_SECOND_SLICE_ARG_INDEX];
    taskArgs[EIGHT_PLUS_FOUR_FIRST_SLICE_ARG_INDEX] = 0;
    taskArgs[EIGHT_PLUS_FOUR_SECOND_SLICE_ARG_INDEX] = 0;
    const HcclResult result = LaunchEightPlusFourRingEdge(resourceCtx, edgeIdx,
        EIGHT_PLUS_FOUR_EDGE_SEND_STAGE_ONE, taskArgs);
    taskArgs[EIGHT_PLUS_FOUR_FIRST_SLICE_ARG_INDEX] = firstSliceBytes;
    taskArgs[EIGHT_PLUS_FOUR_SECOND_SLICE_ARG_INDEX] = secondSliceBytes;
    return result;
}

HcclResult LaunchEightPlusFourTreeExchange(const OpParam &param, const AlgResourceCtx &resourceCtx,
    uint64_t recvBytes, uint64_t chunkOffset, uint64_t chunkBytes, uint64_t inputToken,
    uint64_t outputToken)
{
    std::vector<uint64_t> taskArgs = BuildEightPlusFourEdgeTaskArgs(param, recvBytes,
        chunkOffset, chunkBytes, inputToken, outputToken);
    CHK_RET(InitEightPlusFourRingEdges(resourceCtx, taskArgs));

    const uint32_t groupIndex = param.myRank / EIGHT_PLUS_FOUR_GROUP_SIZE_EXEC;
    const uint32_t groupRank = param.myRank % EIGHT_PLUS_FOUR_GROUP_SIZE_EXEC;
    if (groupIndex == 0) {
        SetEightPlusFourBlockOffset(taskArgs, recvBytes, chunkOffset,
            groupRank + EIGHT_PLUS_FOUR_GROUP_SIZE_EXEC);
        CHK_RET(LaunchEightPlusFourRingEdge(resourceCtx, 0,
            EIGHT_PLUS_FOUR_EDGE_SEND_STAGE_ZERO, taskArgs));
        CHK_RET(LaunchEightPlusFourRingEdge(resourceCtx, 0,
            EIGHT_PLUS_FOUR_EDGE_WAIT_STAGE_ZERO, taskArgs));
        SetEightPlusFourBlockOffset(taskArgs, recvBytes, chunkOffset,
            groupRank + 2 * EIGHT_PLUS_FOUR_GROUP_SIZE_EXEC);
        CHK_RET(LaunchEightPlusFourRingEdge(resourceCtx, 0,
            EIGHT_PLUS_FOUR_EDGE_SEND_STAGE_ONE, taskArgs));
        CHK_RET(SignalEightPlusFourRingEdge(resourceCtx, 2, taskArgs));
        CHK_RET(LaunchEightPlusFourRingEdge(resourceCtx, 2,
            EIGHT_PLUS_FOUR_EDGE_WAIT_STAGE_ZERO, taskArgs));
        return LaunchEightPlusFourRingEdge(resourceCtx, 2,
            EIGHT_PLUS_FOUR_EDGE_COPY_OUTPUT, taskArgs);
    }
    if (groupIndex == 1) {
        SetEightPlusFourBlockOffset(taskArgs, recvBytes, chunkOffset, groupRank);
        CHK_RET(LaunchEightPlusFourRingEdge(resourceCtx, 0,
            EIGHT_PLUS_FOUR_EDGE_SEND_STAGE_ZERO, taskArgs));
        CHK_RET(LaunchEightPlusFourRingEdge(resourceCtx, 0,
            EIGHT_PLUS_FOUR_EDGE_WAIT_STAGE_ZERO, taskArgs));
        CHK_RET(LaunchEightPlusFourRingEdge(resourceCtx, 0,
            EIGHT_PLUS_FOUR_EDGE_WAIT_STAGE_ONE, taskArgs));
        CHK_RET(SignalEightPlusFourRingEdge(resourceCtx, 1, taskArgs));
        SetEightPlusFourBlockOffset(taskArgs, recvBytes, chunkOffset,
            groupRank + 2 * EIGHT_PLUS_FOUR_GROUP_SIZE_EXEC);
        CHK_RET(LaunchEightPlusFourRingEdge(resourceCtx, 1,
            EIGHT_PLUS_FOUR_EDGE_SEND_STAGE_ZERO, taskArgs));
        CHK_RET(LaunchEightPlusFourRingEdge(resourceCtx, 1,
            EIGHT_PLUS_FOUR_EDGE_WAIT_STAGE_ZERO, taskArgs));
        return LaunchEightPlusFourRingEdge(resourceCtx, 1,
            EIGHT_PLUS_FOUR_EDGE_COPY_OUTPUT, taskArgs);
    }

    CHK_RET(LaunchEightPlusFourRingEdge(resourceCtx, 2,
        EIGHT_PLUS_FOUR_EDGE_WAIT_STAGE_ONE, taskArgs));
    CHK_RET(LaunchEightPlusFourRingEdge(resourceCtx, 1,
        EIGHT_PLUS_FOUR_EDGE_WAIT_STAGE_ONE, taskArgs));
    SetEightPlusFourBlockOffset(taskArgs, recvBytes, chunkOffset, groupRank);
    CHK_RET(LaunchEightPlusFourRingEdge(resourceCtx, 2,
        EIGHT_PLUS_FOUR_EDGE_SEND_STAGE_ZERO, taskArgs));
    SetEightPlusFourBlockOffset(taskArgs, recvBytes, chunkOffset,
        groupRank + EIGHT_PLUS_FOUR_GROUP_SIZE_EXEC);
    CHK_RET(LaunchEightPlusFourRingEdge(resourceCtx, 1,
        EIGHT_PLUS_FOUR_EDGE_SEND_STAGE_ZERO, taskArgs));
    CHK_RET(LaunchEightPlusFourRingEdge(resourceCtx, 1,
        EIGHT_PLUS_FOUR_EDGE_WAIT_STAGE_ZERO, taskArgs));
    return LaunchEightPlusFourRingEdge(resourceCtx, 1,
        EIGHT_PLUS_FOUR_EDGE_COPY_OUTPUT, taskArgs);
}

HcclResult LaunchEightPlusFourRing(const OpParam &param, const AlgResourceCtx &resourceCtx,
    uint64_t recvBytes, uint64_t chunkOffset, uint64_t chunkBytes, uint64_t inputToken,
    uint64_t outputToken)
{
    const uint32_t expectedResourceCount =
        EIGHT_PLUS_FOUR_RING_GROUP_COUNT * EIGHT_PLUS_FOUR_EDGE_KERNEL_COUNT;
    if (resourceCtx.eightPlusFourRingEdgeKernels.size() != expectedResourceCount ||
        resourceCtx.eightPlusFourRingEdgeKernelDieIds.size() != expectedResourceCount) {
        return HCCL_E_INTERNAL;
    }
    std::vector<uint64_t> taskArgs = BuildEightPlusFourEdgeTaskArgs(param, recvBytes,
        chunkOffset, chunkBytes, inputToken, outputToken);
    CHK_RET(InitEightPlusFourRingEdges(resourceCtx, taskArgs));

    const uint32_t groupIndex = param.myRank / EIGHT_PLUS_FOUR_GROUP_SIZE_EXEC;
    const uint32_t groupRank = param.myRank % EIGHT_PLUS_FOUR_GROUP_SIZE_EXEC;
    const uint32_t outgoingEdge = groupIndex;
    const uint32_t incomingEdge =
        (groupIndex + EIGHT_PLUS_FOUR_RING_GROUP_COUNT - 1) % EIGHT_PLUS_FOUR_RING_GROUP_COUNT;
    const uint32_t stageZeroBlockGroup =
        (groupIndex + EIGHT_PLUS_FOUR_RING_GROUP_COUNT - 1) % EIGHT_PLUS_FOUR_RING_GROUP_COUNT;
    const uint32_t stageOneBlockGroup =
        (groupIndex + 1) % EIGHT_PLUS_FOUR_RING_GROUP_COUNT;
    SetEightPlusFourBlockOffset(taskArgs, recvBytes, chunkOffset,
        groupRank + stageZeroBlockGroup * EIGHT_PLUS_FOUR_GROUP_SIZE_EXEC);
    CHK_RET(LaunchEightPlusFourRingEdge(resourceCtx, outgoingEdge,
        EIGHT_PLUS_FOUR_EDGE_SEND_STAGE_ZERO, taskArgs));
    CHK_RET(LaunchEightPlusFourRingEdge(resourceCtx, incomingEdge,
        EIGHT_PLUS_FOUR_EDGE_WAIT_STAGE_ZERO, taskArgs));
    SetEightPlusFourBlockOffset(taskArgs, recvBytes, chunkOffset,
        groupRank + stageOneBlockGroup * EIGHT_PLUS_FOUR_GROUP_SIZE_EXEC);
    CHK_RET(LaunchEightPlusFourRingEdge(resourceCtx, outgoingEdge,
        EIGHT_PLUS_FOUR_EDGE_SEND_STAGE_ONE, taskArgs));
    CHK_RET(LaunchEightPlusFourRingEdge(resourceCtx, incomingEdge,
        EIGHT_PLUS_FOUR_EDGE_WAIT_STAGE_ONE, taskArgs));
    return LaunchEightPlusFourRingEdge(resourceCtx, incomingEdge,
        EIGHT_PLUS_FOUR_EDGE_COPY_OUTPUT, taskArgs);
}
} // namespace

HcclResult ExecOp(const OpParam &param)
{
    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> sequenceData(ctx, ctx + param.ctxSize);
    AlgResourceCtx resourceCtx;
    resourceCtx.DeSerialize(sequenceData);

    if (param.dataType != HCCL_DATA_TYPE_FP32 || param.reduceType != HCCL_REDUCE_SUM) {
        return HCCL_E_NOT_SUPPORT;
    }
    if (param.rankSize == 0 || param.myRank >= param.rankSize || resourceCtx.threads.empty()) {
        return HCCL_E_PARA;
    }
    if (param.count > std::numeric_limits<uint64_t>::max() / sizeof(float)) {
        return HCCL_E_PARA;
    }

    const uint64_t recvBytes = param.count * sizeof(float);
    if (recvBytes == 0) {
        return HCCL_SUCCESS;
    }
    if (param.rankSize == 1) {
        return static_cast<HcclResult>(HcommLocalCopyOnThread(resourceCtx.threads[0], param.outputPtr,
            param.inputPtr, recvBytes));
    }
    if (recvBytes > std::numeric_limits<uint64_t>::max() / param.rankSize) {
        return HCCL_E_PARA;
    }
    const uint64_t inputBytes = recvBytes * param.rankSize;

    const auto topologyKind = static_cast<Rs2TopologyKind>(resourceCtx.topologyKind);
    if (RS2_BUILD_FLAVOR == Rs2BuildFlavor::FOUR_BY_ONE_RING &&
        topologyKind == Rs2TopologyKind::EIGHT_PLUS_FOUR) {
        if (resourceCtx.localBuffer.addr == nullptr ||
            resourceCtx.localBuffer.size < EIGHT_PLUS_FOUR_SCRATCH_SLOTS * sizeof(float)) {
            return HCCL_E_MEMORY;
        }
        uint64_t chunkLimit = std::min<uint64_t>(MAX_DATA_SIZE,
            resourceCtx.localBuffer.size / EIGHT_PLUS_FOUR_SCRATCH_SLOTS);
        chunkLimit = chunkLimit / SCRATCH_ALIGNMENT * SCRATCH_ALIGNMENT;
        if (chunkLimit == 0) {
            chunkLimit = resourceCtx.localBuffer.size / EIGHT_PLUS_FOUR_SCRATCH_SLOTS;
            chunkLimit = chunkLimit / sizeof(float) * sizeof(float);
        }
        if (chunkLimit == 0) {
            return HCCL_E_MEMORY;
        }

        uint64_t inputToken = 0;
        uint64_t outputToken = 0;
        uint64_t scratchToken = 0;
        CHK_RET(GetToken(reinterpret_cast<uint64_t>(param.inputPtr), inputBytes, inputToken));
        CHK_RET(GetToken(reinterpret_cast<uint64_t>(param.outputPtr), recvBytes, outputToken));
        CHK_RET(GetToken(reinterpret_cast<uint64_t>(resourceCtx.localBuffer.addr),
            resourceCtx.localBuffer.size, scratchToken));
        for (uint64_t chunkOffset = 0; chunkOffset < recvBytes; chunkOffset += chunkLimit) {
            const uint64_t chunkBytes = std::min(chunkLimit, recvBytes - chunkOffset);
            CHK_RET(LaunchEightPlusFourPreReduce(param, resourceCtx, recvBytes, chunkOffset,
                chunkBytes, inputToken, scratchToken));
            if (inputBytes <= SMALL_INPUT_BYTES) {
                CHK_RET(LaunchEightPlusFourTreeExchange(param, resourceCtx, recvBytes, chunkOffset,
                    chunkBytes, inputToken, outputToken));
            } else {
                CHK_RET(LaunchEightPlusFourRing(param, resourceCtx, recvBytes, chunkOffset,
                    chunkBytes, inputToken, outputToken));
            }
        }
        return HCCL_SUCCESS;
    }

    const bool useRing = RS2_BUILD_FLAVOR == Rs2BuildFlavor::FOUR_BY_ONE_RING &&
        topologyKind == Rs2TopologyKind::FOUR_BY_ONE &&
        inputBytes > SMALL_INPUT_BYTES;
    if (useRing) {
        uint64_t inputToken = 0;
        uint64_t outputToken = 0;
        CHK_RET(GetToken(reinterpret_cast<uint64_t>(param.inputPtr), inputBytes, inputToken));
        CHK_RET(GetToken(reinterpret_cast<uint64_t>(param.outputPtr), recvBytes, outputToken));
        const uint64_t chunkLimit = std::min<uint64_t>(MAX_DATA_SIZE, recvBytes);
        for (uint64_t chunkOffset = 0; chunkOffset < recvBytes; chunkOffset += chunkLimit) {
            const uint64_t chunkBytes = std::min(chunkLimit, recvBytes - chunkOffset);
            CHK_RET(LaunchRing(param, resourceCtx, recvBytes, chunkOffset, chunkBytes, inputToken, outputToken));
        }
        return HCCL_SUCCESS;
    }

    const KernelSequence kernelSequence = SelectKernelSequence(resourceCtx, inputBytes);
    if (kernelSequence.kernelSets.empty()) {
        HCCL_ERROR("No CCU kernel sequence selected");
        return HCCL_E_INTERNAL;
    }
    const uint32_t maxPieceCount = GetMaxPieceCount(kernelSequence);
    if (maxPieceCount == 0 || resourceCtx.localBuffer.addr == nullptr || resourceCtx.localBuffer.size == 0) {
        return HCCL_E_MEMORY;
    }

    uint64_t chunkLimit = std::min<uint64_t>(MAX_DATA_SIZE, resourceCtx.localBuffer.size / maxPieceCount);
    chunkLimit = chunkLimit / SCRATCH_ALIGNMENT * SCRATCH_ALIGNMENT;
    if (chunkLimit == 0) {
        chunkLimit = resourceCtx.localBuffer.size / maxPieceCount;
        chunkLimit = chunkLimit / sizeof(float) * sizeof(float);
    }
    if (chunkLimit == 0) {
        return HCCL_E_MEMORY;
    }

    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    uint64_t scratchToken = 0;
    CHK_RET(GetToken(reinterpret_cast<uint64_t>(param.inputPtr), inputBytes, inputToken));
    CHK_RET(GetToken(reinterpret_cast<uint64_t>(param.outputPtr), recvBytes, outputToken));
    CHK_RET(GetToken(reinterpret_cast<uint64_t>(resourceCtx.localBuffer.addr), resourceCtx.localBuffer.size,
        scratchToken));

    for (uint64_t chunkOffset = 0; chunkOffset < recvBytes; chunkOffset += chunkLimit) {
        const uint64_t chunkBytes = std::min(chunkLimit, recvBytes - chunkOffset);
        CHK_RET(LaunchSequence(param, resourceCtx, kernelSequence, recvBytes, chunkOffset, chunkBytes,
            inputToken, outputToken, scratchToken));
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
} // namespace final_adaptive41
