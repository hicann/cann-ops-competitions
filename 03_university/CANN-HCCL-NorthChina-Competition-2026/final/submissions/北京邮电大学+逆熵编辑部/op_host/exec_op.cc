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
#include <cstdint>
#include <limits>
#include <vector>

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {

namespace {

// CCU 的单次 UBC 传输要求严格小于 256MB。减去 128B 后仍满足
// float32 和 HCCL 最小切片对齐，并能覆盖 256MB 边界用例。
constexpr uint64_t MAX_TRANSFER_BYTES = static_cast<uint64_t>(MAX_DATA_SIZE) - 128U;
constexpr uint64_t PUBLISH_STAGE = 0;
constexpr uint64_t WRITE_RECORD_STAGE = 1;
constexpr uint64_t FINAL_WAIT_STAGE = 2;
constexpr uint64_t PAIR_PUBLISH_STAGE = 3;
constexpr uint64_t PAIR_WRITE_RECORD_STAGE = 4;
constexpr uint64_t PAIR_FINAL_WAIT_STAGE = 5;
constexpr uint64_t ACK_FINAL_WAIT_STAGE = 6;
constexpr uint64_t FUSED_WRITE_WAIT_STAGE = 7;
constexpr uint64_t MONOLITHIC_STAGE = 8;
constexpr uint64_t RELAY_PUBLISH_STAGE = 9;
constexpr uint64_t RELAY_WRITE_RECORD_STAGE = 10;
constexpr uint64_t RELAY_FINAL_WAIT_STAGE = 11;
constexpr uint64_t RELAY_ACK_FINAL_WAIT_STAGE = 12;
constexpr uint64_t ASYMMETRIC_PAIR_ROUND0_STAGE = 13;
constexpr uint64_t ASYMMETRIC_NONPAIR_STAGE = 15;
constexpr uint64_t ASYMMETRIC_LOCAL_WRITE_STAGE = 16;
constexpr uint64_t ASYMMETRIC_RELAY_WRITE_STAGE = 17;
constexpr uint64_t ASYMMETRIC_RELAY_WRITE_RECORD_STAGE = 18;
constexpr uint64_t ASYMMETRIC_RELAY_READ_ROUND0_STAGE = 19;
constexpr uint64_t ASYMMETRIC_RELAY_READ_ROUND1_RECORD_STAGE = 20;
constexpr uint64_t ASYMMETRIC_FUSED_PULL_RELAY_STAGE = 21;
constexpr uint64_t ASYMMETRIC_FUSED_THIRD_WRITE_PULL_RELAY_STAGE = 22;
constexpr uint64_t ASYMMETRIC_SEGMENTED_DIRECT_WRITE_RECORD_STAGE = 23;
constexpr uint64_t ASYMMETRIC_FUSED_QUARTER_WRITE_PULL_RELAY_STAGE = 24;
constexpr uint32_t LAYER_PARALLEL_NOTIFY_INDEX = 0;
constexpr uint32_t COPY_COMPLETION_NOTIFY_INDEX = 1;
constexpr uint32_t RANK_SIZE_8_PLUS_4 = 12;
constexpr uint32_t RANK_SIZE_2_X_8 = 16;
constexpr uint32_t RANK_SIZE_4_X_1 = 4;
constexpr size_t COPY_THREAD_INDEX = 1;
constexpr uint64_t HYBRID_DIRECT_NUMERATOR = 5;
constexpr uint64_t HYBRID_RATIO_DENOMINATOR = 11;
constexpr uint64_t HYBRID_SPLIT_ALIGNMENT = 128;
constexpr uint64_t HYBRID_MESSAGE_THRESHOLD = 512U * 1024U;
constexpr uint64_t FORMAL_SMALL_OUTPUT_SIZE = 512U * 1024U;
constexpr uint64_t FORMAL_SMALL_2X8_SLICE_SIZE =
    FORMAL_SMALL_OUTPUT_SIZE / RANK_SIZE_2_X_8;
constexpr uint64_t FORMAL_SMALL_8_PLUS_4_SLICE_SIZE = 43692U;
constexpr uint64_t FORMAL_SMALL_8_PLUS_4_OUTPUT_SIZE = 524304U;
constexpr uint64_t TWO_PHASE_SEGMENT_NUMERATOR = 7;
constexpr uint64_t TWO_PHASE_SEGMENT_DENOMINATOR = 22;
constexpr uint32_t ASYMMETRIC_CYCLE_CHUNK_COUNT = 7;
constexpr uint32_t ASYMMETRIC_DIRECT_CHUNK_COUNT = 6;
constexpr uint64_t ASYMMETRIC_COARSE_QUARTER_DENOMINATOR = 4;

HcclResult PreSyncLayerThreads(ThreadHandle mainThread, ThreadHandle slaveThread);
HcclResult PostSyncLayerThreads(ThreadHandle mainThread, ThreadHandle slaveThread);
HcclResult PostSyncCopyThread(ThreadHandle mainThread, ThreadHandle copyThread);

enum class KernelSelection {
    ALL,
    LAYER0,
    LAYER1,
    PAIR,
    RELAY,
};

struct EqualCycleChunk {
    uint64_t offset = 0;
    uint64_t size = 0;
    bool relay = false;
};

HcclResult GetDataTypeSize(HcclDataType dataType, uint64_t &dataTypeSize)
{
    const auto iter = SIZE_TABLE.find(dataType);
    if (iter == SIZE_TABLE.end()) {
        HCCL_ERROR("[ExecOp] Unsupported data type: %d", static_cast<int32_t>(dataType));
        return HCCL_E_NOT_SUPPORT;
    }
    dataTypeSize = iter->second;
    return HCCL_SUCCESS;
}

HcclResult GetMemToken(uint64_t address, uint64_t size, uint64_t &token)
{
    const CcuResult result = HcommCcuGetMemToken(address, size, &token);
    if (result != CCU_SUCCESS) {
        HCCL_ERROR("[ExecOp] HcommCcuGetMemToken failed, ccuRet=%d", static_cast<int32_t>(result));
        return ConvertCcuToHccl(result);
    }
    return HCCL_SUCCESS;
}

bool IsKernelSelected(
    const AlgResourceCtx &resCtx, uint32_t kernelIndex, KernelSelection selection)
{
    if (selection == KernelSelection::ALL) {
        return true;
    }
    if (selection == KernelSelection::LAYER0) {
        return resCtx.kernelLayers[kernelIndex] == 0;
    }
    if (selection == KernelSelection::LAYER1) {
        return resCtx.kernelLayers[kernelIndex] == 1;
    }
    if (selection == KernelSelection::PAIR) {
        return kernelIndex == resCtx.pairKernelIndex;
    }
    return std::find(resCtx.relayKernelIndices.begin(),
        resCtx.relayKernelIndices.end(), kernelIndex) !=
        resCtx.relayKernelIndices.end();
}

ThreadHandle GetKernelThread(
    const OpParam &param, const AlgResourceCtx &resCtx, uint32_t kernelIndex, bool parallelLayers)
{
    return parallelLayers && resCtx.kernelLayers[kernelIndex] == 1 ?
        resCtx.slaveThreads[0] : param.cpuThread;
}

HcclResult LaunchSelectedStage(const OpParam &param, const AlgResourceCtx &resCtx,
    std::vector<uint64_t> &taskArgs, KernelSelection selection, bool parallelLayers)
{
    uint32_t selectedKernelCount = 0;
    for (uint32_t kernelIndex = 0; kernelIndex < resCtx.ccuKernels.size(); ++kernelIndex) {
        if (!IsKernelSelected(resCtx, kernelIndex, selection)) {
            continue;
        }
        ++selectedKernelCount;
        const CcuResult result = HcommCcuKernelLaunch(
            GetKernelThread(param, resCtx, kernelIndex, parallelLayers),
            resCtx.ccuKernels[kernelIndex], taskArgs.data(),
            static_cast<uint32_t>(taskArgs.size()));
        if (result != CCU_SUCCESS) {
            HCCL_ERROR("[ExecOp] HcommCcuKernelLaunch failed, ccuRet=%d",
                static_cast<int32_t>(result));
            return ConvertCcuToHccl(result);
        }
    }
    if (selectedKernelCount == 0) {
        HCCL_ERROR("[ExecOp] No CCU kernel matched selection %u",
            static_cast<uint32_t>(selection));
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchSmall4x1DirectPull(
    const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t inputBase, uint64_t outputBase,
    uint64_t inputToken, uint64_t outputToken,
    uint64_t dataSize)
{
    if (resCtx.small4x1PullKernel == 0) {
        HCCL_ERROR("[ExecOp] Dedicated 4x1 Pull kernel was not registered");
        return HCCL_E_INTERNAL;
    }
    std::vector<uint64_t> taskArgs = {
        inputBase,
        outputBase,
        inputToken,
        outputToken,
        dataSize,
    };
    const CcuResult result = HcommCcuKernelLaunch(
        param.cpuThread, resCtx.small4x1PullKernel,
        taskArgs.data(), static_cast<uint32_t>(taskArgs.size()));
    if (result != CCU_SUCCESS) {
        HCCL_ERROR(
            "[ExecOp] Dedicated 4x1 Pull kernel launch failed, ccuRet=%d",
            static_cast<int32_t>(result));
        return ConvertCcuToHccl(result);
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchSmall2x8DirectPull(
    const OpParam &param,
    const Small2x8PullResourceCtx &resCtx,
    uint64_t inputBase, uint64_t outputBase,
    uint64_t inputToken, uint64_t outputToken,
    uint64_t dataSize)
{
    if (resCtx.kernels.size() != 2U ||
        resCtx.layers.size() != resCtx.kernels.size() ||
        resCtx.slaveThreads.size() != 1U) {
        HCCL_ERROR(
            "[ExecOp] Dedicated 2x8 Pull resources are incomplete");
        return HCCL_E_INTERNAL;
    }

    uint32_t layer0Index = MAX_RANK_SIZE;
    uint32_t layer1Index = MAX_RANK_SIZE;
    for (uint32_t kernelIndex = 0;
        kernelIndex < resCtx.layers.size();
        ++kernelIndex) {
        const uint32_t layerId =
            resCtx.layers[kernelIndex];
        if (layerId == 0U) {
            if (layer0Index != MAX_RANK_SIZE) {
                HCCL_ERROR(
                    "[ExecOp] Dedicated 2x8 Pull has multiple layer-0 groups");
                return HCCL_E_INTERNAL;
            }
            layer0Index = kernelIndex;
        } else if (layerId == 1U) {
            if (layer1Index != MAX_RANK_SIZE) {
                HCCL_ERROR(
                    "[ExecOp] Dedicated 2x8 Pull has multiple layer-1 groups");
                return HCCL_E_INTERNAL;
            }
            layer1Index = kernelIndex;
        } else {
            HCCL_ERROR(
                "[ExecOp] Dedicated 2x8 Pull has invalid layer %u",
                layerId);
            return HCCL_E_INTERNAL;
        }
    }
    if (layer0Index == MAX_RANK_SIZE ||
        layer1Index == MAX_RANK_SIZE) {
        HCCL_ERROR(
            "[ExecOp] Dedicated 2x8 Pull expected one group per layer");
        return HCCL_E_INTERNAL;
    }

    std::vector<uint64_t> taskArgs = {
        inputBase,
        outputBase,
        inputToken,
        outputToken,
        dataSize,
    };

    // The two IO Dies run concurrently: main owns Mesh and slave[0] owns Clos.
    CHK_RET(PreSyncLayerThreads(
        param.cpuThread, resCtx.slaveThreads[0]));

    auto launchKernel = [&](ThreadHandle thread,
                            uint32_t kernelIndex) -> HcclResult {
        const CcuResult result = HcommCcuKernelLaunch(
            thread, resCtx.kernels[kernelIndex],
            taskArgs.data(),
            static_cast<uint32_t>(taskArgs.size()));
        if (result != CCU_SUCCESS) {
            HCCL_ERROR(
                "[ExecOp] Dedicated 2x8 Pull kernel-%u launch failed, ccuRet=%d",
                kernelIndex, static_cast<int32_t>(result));
            return ConvertCcuToHccl(result);
        }
        return HCCL_SUCCESS;
    };

    const HcclResult layer1Launch = launchKernel(
        resCtx.slaveThreads[0], layer1Index);
    const HcclResult layer0Launch = launchKernel(
        param.cpuThread, layer0Index);

    const HcclResult joinLayer1 = PostSyncLayerThreads(
        param.cpuThread, resCtx.slaveThreads[0]);
    if (layer1Launch != HCCL_SUCCESS) {
        return layer1Launch;
    }
    if (layer0Launch != HCCL_SUCCESS) {
        return layer0Launch;
    }
    if (joinLayer1 != HCCL_SUCCESS) {
        return joinLayer1;
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchSmall8Plus4DirectPull(
    const OpParam &param,
    const Small8Plus4PullResourceCtx &resCtx,
    uint64_t inputBase, uint64_t outputBase,
    uint64_t inputToken, uint64_t outputToken,
    uint64_t dataSize)
{
    if (resCtx.kernels.size() != 2U ||
        resCtx.layers.size() != 2U ||
        resCtx.slaveThreads.size() != 1U) {
        HCCL_ERROR(
            "[ExecOp] Dedicated 8+4 Pull resources are incomplete");
        return HCCL_E_INTERNAL;
    }
    std::vector<uint64_t> taskArgs = {
        inputBase, outputBase, inputToken, outputToken, dataSize,
    };
    CHK_RET(PreSyncLayerThreads(
        param.cpuThread, resCtx.slaveThreads[0]));

    auto launchKernel = [&](ThreadHandle thread,
                            uint32_t kernelIndex) -> HcclResult {
        const CcuResult result = HcommCcuKernelLaunch(
            thread, resCtx.kernels[kernelIndex],
            taskArgs.data(),
            static_cast<uint32_t>(taskArgs.size()));
        return result == CCU_SUCCESS ?
            HCCL_SUCCESS : ConvertCcuToHccl(result);
    };
    const HcclResult slave0Launch =
        launchKernel(resCtx.slaveThreads[0], 1U);
    const HcclResult mainLaunch =
        launchKernel(param.cpuThread, 0U);
    const HcclResult join0 = PostSyncLayerThreads(
        param.cpuThread, resCtx.slaveThreads[0]);
    if (slave0Launch != HCCL_SUCCESS) {
        return slave0Launch;
    }
    if (mainLaunch != HCCL_SUCCESS) {
        return mainLaunch;
    }
    return join0;
}

HcclResult LaunchLarge8Plus4DirectPullGeneration(
    const OpParam &param,
    const Small8Plus4PullResourceCtx &resCtx,
    uint64_t inputAddress, uint64_t outputAddress,
    uint64_t inputToken, uint64_t outputToken,
    uint64_t rankStride, uint64_t transferSize)
{
    if (resCtx.kernels.size() != 2U || resCtx.layers.size() != 2U ||
        resCtx.slaveThreads.size() != 1U) {
        HCCL_ERROR("[ExecOp] Dedicated large 8+4 Pull resources are incomplete");
        return HCCL_E_INTERNAL;
    }
    uint32_t layer0Index = 2U;
    uint32_t layer1Index = 2U;
    for (uint32_t index = 0; index < 2U; ++index) {
        if (resCtx.layers[index] == 0U) {
            layer0Index = index;
        } else if (resCtx.layers[index] == 1U) {
            layer1Index = index;
        }
    }
    if (layer0Index >= 2U || layer1Index >= 2U) {
        HCCL_ERROR("[ExecOp] Dedicated large 8+4 Pull layer mapping is invalid");
        return HCCL_E_INTERNAL;
    }
    std::vector<uint64_t> taskArgs = {
        inputAddress, outputAddress, inputToken, outputToken,
        rankStride, transferSize,
    };
    CHK_RET(PreSyncLayerThreads(param.cpuThread, resCtx.slaveThreads[0]));
    const CcuResult layer1Result = HcommCcuKernelLaunch(
        resCtx.slaveThreads[0], resCtx.kernels[layer1Index],
        taskArgs.data(), static_cast<uint32_t>(taskArgs.size()));
    const CcuResult layer0Result = HcommCcuKernelLaunch(
        param.cpuThread, resCtx.kernels[layer0Index],
        taskArgs.data(), static_cast<uint32_t>(taskArgs.size()));
    const HcclResult joinResult = PostSyncLayerThreads(
        param.cpuThread, resCtx.slaveThreads[0]);
    if (layer1Result != CCU_SUCCESS) {
        return ConvertCcuToHccl(layer1Result);
    }
    if (layer0Result != CCU_SUCCESS) {
        return ConvertCcuToHccl(layer0Result);
    }
    return joinResult;
}

HcclResult LaunchStagedOperation(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t inputAddress, uint64_t outputAddress, uint64_t inputToken, uint64_t outputToken,
    uint64_t remoteOutputOffset, uint64_t sliceSize, KernelSelection selection,
    bool parallelLayers, bool globalLayerBarriers, bool acknowledgeCompletion, bool fuseWriteWait)
{
    const bool pairOnly = selection == KernelSelection::PAIR;
    const bool relayOnly = selection == KernelSelection::RELAY;
    std::vector<uint64_t> taskArgs = {
        inputAddress,
        outputAddress,
        inputToken,
        outputToken,
        remoteOutputOffset,
        sliceSize,
        pairOnly ? PAIR_PUBLISH_STAGE :
            (relayOnly ? RELAY_PUBLISH_STAGE : PUBLISH_STAGE),
    };

    CHK_RET(LaunchSelectedStage(param, resCtx, taskArgs, selection, parallelLayers));
    if (globalLayerBarriers) {
        CHK_RET(PostSyncLayerThreads(param.cpuThread, resCtx.slaveThreads[0]));
        CHK_RET(PreSyncLayerThreads(param.cpuThread, resCtx.slaveThreads[0]));
    }

    taskArgs.back() = fuseWriteWait ? FUSED_WRITE_WAIT_STAGE :
        (pairOnly ? PAIR_WRITE_RECORD_STAGE :
            (relayOnly ? RELAY_WRITE_RECORD_STAGE : WRITE_RECORD_STAGE));
    CHK_RET(LaunchSelectedStage(param, resCtx, taskArgs, selection, parallelLayers));
    if (fuseWriteWait) {
        return HCCL_SUCCESS;
    }
    if (globalLayerBarriers) {
        CHK_RET(PostSyncLayerThreads(param.cpuThread, resCtx.slaveThreads[0]));
        CHK_RET(PreSyncLayerThreads(param.cpuThread, resCtx.slaveThreads[0]));
    }

    taskArgs.back() = pairOnly ? PAIR_FINAL_WAIT_STAGE :
        (relayOnly ?
            (acknowledgeCompletion ? RELAY_ACK_FINAL_WAIT_STAGE :
                RELAY_FINAL_WAIT_STAGE) :
            (acknowledgeCompletion ? ACK_FINAL_WAIT_STAGE : FINAL_WAIT_STAGE));
    CHK_RET(LaunchSelectedStage(param, resCtx, taskArgs, selection, parallelLayers));
    return HCCL_SUCCESS;
}

uint64_t GetStageId(KernelSelection selection, uint64_t commonStage, uint64_t pairStage,
    bool acknowledgeCompletion)
{
    if (selection == KernelSelection::PAIR) {
        return pairStage;
    }
    if (selection == KernelSelection::RELAY) {
        if (commonStage == PUBLISH_STAGE) {
            return RELAY_PUBLISH_STAGE;
        }
        if (commonStage == WRITE_RECORD_STAGE) {
            return RELAY_WRITE_RECORD_STAGE;
        }
        return acknowledgeCompletion ?
            RELAY_ACK_FINAL_WAIT_STAGE : RELAY_FINAL_WAIT_STAGE;
    }
    if (commonStage == FINAL_WAIT_STAGE && acknowledgeCompletion) {
        return ACK_FINAL_WAIT_STAGE;
    }
    return commonStage;
}

HcclResult LaunchInterleavedStagedOperations(const OpParam &param,
    const AlgResourceCtx &resCtx, std::vector<uint64_t> firstTaskArgs,
    KernelSelection firstSelection, bool firstAcknowledgeCompletion,
    std::vector<uint64_t> secondTaskArgs, KernelSelection secondSelection,
    bool secondAcknowledgeCompletion, bool synchronizeAfterPublish = true)
{
    firstTaskArgs.back() = GetStageId(
        firstSelection, PUBLISH_STAGE, PAIR_PUBLISH_STAGE, false);
    secondTaskArgs.back() = GetStageId(
        secondSelection, PUBLISH_STAGE, PAIR_PUBLISH_STAGE, false);
    CHK_RET(LaunchSelectedStage(param, resCtx, firstTaskArgs, firstSelection, true));
    CHK_RET(LaunchSelectedStage(param, resCtx, secondTaskArgs, secondSelection, true));
    if (synchronizeAfterPublish) {
        CHK_RET(PostSyncLayerThreads(param.cpuThread, resCtx.slaveThreads[0]));
        CHK_RET(PreSyncLayerThreads(param.cpuThread, resCtx.slaveThreads[0]));
    }

    firstTaskArgs.back() = GetStageId(
        firstSelection, WRITE_RECORD_STAGE, PAIR_WRITE_RECORD_STAGE, false);
    secondTaskArgs.back() = GetStageId(
        secondSelection, WRITE_RECORD_STAGE, PAIR_WRITE_RECORD_STAGE, false);
    CHK_RET(LaunchSelectedStage(param, resCtx, firstTaskArgs, firstSelection, true));
    CHK_RET(LaunchSelectedStage(param, resCtx, secondTaskArgs, secondSelection, true));

    // Each selected kernel remains on one fixed Thread, so its FinalWait mission is
    // ordered after its own WriteRecord mission. The two layers use disjoint channels;
    // a second cross-Thread barrier is not required before queueing FinalWait.
    firstTaskArgs.back() = GetStageId(firstSelection, FINAL_WAIT_STAGE,
        PAIR_FINAL_WAIT_STAGE, firstAcknowledgeCompletion);
    secondTaskArgs.back() = GetStageId(secondSelection, FINAL_WAIT_STAGE,
        PAIR_FINAL_WAIT_STAGE, secondAcknowledgeCompletion);
    CHK_RET(LaunchSelectedStage(param, resCtx, firstTaskArgs, firstSelection, true));
    CHK_RET(LaunchSelectedStage(param, resCtx, secondTaskArgs, secondSelection, true));
    return HCCL_SUCCESS;
}

HcclResult LaunchDirectSlice(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t inputAddress, uint64_t outputAddress, uint64_t inputToken, uint64_t outputToken,
    uint64_t remoteOutputOffset, uint64_t sliceSize, bool parallelLayers,
    bool fuseWriteWait, bool monolithic, bool overlapPureClosCopy)
{
    if (overlapPureClosCopy &&
        (!monolithic || parallelLayers || resCtx.slaveThreads.size() != 1)) {
        HCCL_ERROR("[ExecOp] Invalid pure-Clos local-copy overlap state");
        return HCCL_E_INTERNAL;
    }
    if (monolithic) {
        if (overlapPureClosCopy) {
            CHK_RET(PreSyncLayerThreads(param.cpuThread, resCtx.slaveThreads[0]));
            const uint64_t localOutputAddress = outputAddress + remoteOutputOffset;
            const HcclResult copyResult = static_cast<HcclResult>(HcommLocalCopyOnThread(
                resCtx.slaveThreads[0], reinterpret_cast<void *>(localOutputAddress),
                reinterpret_cast<void *>(inputAddress), sliceSize));
            if (copyResult != HCCL_SUCCESS) {
                HCCL_ERROR("[ExecOp] Pure-Clos worker local copy failed, ret=%d",
                    static_cast<int32_t>(copyResult));
                const HcclResult joinResult =
                    PostSyncLayerThreads(param.cpuThread, resCtx.slaveThreads[0]);
                return joinResult == HCCL_SUCCESS ? copyResult : joinResult;
            }
        }
        std::vector<uint64_t> taskArgs = {
            inputAddress,
            outputAddress,
            inputToken,
            outputToken,
            remoteOutputOffset,
            sliceSize,
            MONOLITHIC_STAGE,
        };
        const HcclResult launchResult = LaunchSelectedStage(
            param, resCtx, taskArgs, KernelSelection::ALL, false);
        if (overlapPureClosCopy) {
            const HcclResult joinResult =
                PostSyncLayerThreads(param.cpuThread, resCtx.slaveThreads[0]);
            return launchResult == HCCL_SUCCESS ? joinResult : launchResult;
        }
        return launchResult;
    }
    if (parallelLayers) {
        CHK_RET(PreSyncLayerThreads(param.cpuThread, resCtx.slaveThreads[0]));
    }
    CHK_RET(LaunchStagedOperation(param, resCtx, inputAddress, outputAddress, inputToken,
        outputToken, remoteOutputOffset, sliceSize, KernelSelection::ALL, parallelLayers,
        parallelLayers, false, fuseWriteWait));
    if (parallelLayers) {
        CHK_RET(PostSyncLayerThreads(param.cpuThread, resCtx.slaveThreads[0]));
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchHierarchicalSmall2x8(const OpParam &param,
    const AlgResourceCtx &resCtx, uint64_t inputBase, uint64_t outputBase,
    uint64_t inputToken, uint64_t outputToken, uint64_t dataSize,
    uint64_t localOutputOffset, uint32_t pairedRemoteRank)
{
    constexpr uint32_t groupRankCount = RANK_SIZE_2_X_8 / 2U;
    if (resCtx.asymmetricPairedRanks.size() != 1U ||
        pairedRemoteRank != (param.myRank ^ groupRankCount) ||
        dataSize > std::numeric_limits<uint64_t>::max() / groupRankCount) {
        HCCL_ERROR("[ExecOp] Invalid 2x8 small hierarchical mapping");
        return HCCL_E_INTERNAL;
    }

    const uint64_t groupBytes = dataSize * groupRankCount;
    if (groupBytes > MAX_TRANSFER_BYTES) {
        HCCL_ERROR("[ExecOp] 2x8 small hierarchical group exceeds CCU limit");
        return HCCL_E_INTERNAL;
    }
    const uint32_t localGroupFirstRank =
        param.myRank / groupRankCount * groupRankCount;
    const uint32_t remoteGroupFirstRank =
        pairedRemoteRank / groupRankCount * groupRankCount;
    const uint64_t localGroupOffset = dataSize * localGroupFirstRank;
    const uint64_t remoteGroupOffset = dataSize * remoteGroupFirstRank;

    // Phase 1 is one self-contained Mesh AllGather inside the local 8-rank
    // server. The host-side local copy is already ordered before this launch.
    std::vector<uint64_t> localTaskArgs = {
        inputBase,
        outputBase,
        inputToken,
        outputToken,
        localOutputOffset,
        dataSize,
        MONOLITHIC_STAGE,
    };
    CHK_RET(LaunchSelectedStage(
        param, resCtx, localTaskArgs, KernelSelection::LAYER0, true));

    // Phase 2 exchanges one contiguous 8-rank block with the same-position
    // rank in the remote server. Stage 13 is the checker-proven one-channel
    // bidirectional monolithic path already used by the v043 relay protocol.
    // Keep this dependent small-message kernel on the main Thread as well:
    // main-Thread queue order replaces the cross-Thread record/wait pair.
    std::vector<uint64_t> pairTaskArgs = {
        outputBase + localGroupOffset,
        outputBase,
        outputToken,
        outputToken,
        remoteGroupOffset,
        groupBytes,
        ASYMMETRIC_PAIR_ROUND0_STAGE,
    };
    CHK_RET(LaunchSelectedStage(
        param, resCtx, pairTaskArgs, KernelSelection::PAIR, false));
    return HCCL_SUCCESS;
}

uint64_t CalcHybridDirectSize(uint64_t sliceSize)
{
    const uint64_t quotient = sliceSize / HYBRID_RATIO_DENOMINATOR;
    const uint64_t remainder = sliceSize % HYBRID_RATIO_DENOMINATOR;
    const uint64_t unalignedDirect =
        quotient * HYBRID_DIRECT_NUMERATOR +
        remainder * HYBRID_DIRECT_NUMERATOR / HYBRID_RATIO_DENOMINATOR;
    return unalignedDirect / HYBRID_SPLIT_ALIGNMENT * HYBRID_SPLIT_ALIGNMENT;
}

uint64_t CalcTwoPhaseSegmentSize(uint64_t dataSize)
{
    const uint64_t unalignedSegment =
        dataSize / TWO_PHASE_SEGMENT_DENOMINATOR *
            TWO_PHASE_SEGMENT_NUMERATOR +
        dataSize % TWO_PHASE_SEGMENT_DENOMINATOR *
            TWO_PHASE_SEGMENT_NUMERATOR /
            TWO_PHASE_SEGMENT_DENOMINATOR;
    return unalignedSegment /
        HYBRID_SPLIT_ALIGNMENT * HYBRID_SPLIT_ALIGNMENT;
}

HcclResult BuildEqualCycleChunks(
    uint64_t dataSize, std::vector<EqualCycleChunk> &chunks)
{
    const uint64_t alignedChunkSize =
        dataSize / ASYMMETRIC_CYCLE_CHUNK_COUNT /
        HYBRID_SPLIT_ALIGNMENT * HYBRID_SPLIT_ALIGNMENT;
    if (alignedChunkSize == 0U) {
        HCCL_ERROR("[ExecOp] 8+4 equal-cycle chunk size is zero, dataSize=%llu",
            static_cast<unsigned long long>(dataSize));
        return HCCL_E_INTERNAL;
    }

    chunks.clear();
    chunks.reserve(ASYMMETRIC_CYCLE_CHUNK_COUNT);
    uint64_t offset = 0;
    for (uint32_t index = 0; index < ASYMMETRIC_CYCLE_CHUNK_COUNT; ++index) {
        const uint64_t chunkSize =
            index + 1U == ASYMMETRIC_CYCLE_CHUNK_COUNT ?
            dataSize - offset : alignedChunkSize;
        if (chunkSize == 0U || chunkSize >= MAX_DATA_SIZE ||
            offset > dataSize - chunkSize) {
            HCCL_ERROR("[ExecOp] Invalid 8+4 equal-cycle chunk[%u]: "
                "offset=%llu size=%llu dataSize=%llu",
                index, static_cast<unsigned long long>(offset),
                static_cast<unsigned long long>(chunkSize),
                static_cast<unsigned long long>(dataSize));
            return HCCL_E_INTERNAL;
        }
        chunks.push_back(EqualCycleChunk{
            offset, chunkSize, index >= ASYMMETRIC_DIRECT_CHUNK_COUNT});
        offset += chunkSize;
    }
    if (offset != dataSize) {
        HCCL_ERROR("[ExecOp] 8+4 equal-cycle coverage mismatch: %llu != %llu",
            static_cast<unsigned long long>(offset),
            static_cast<unsigned long long>(dataSize));
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchCoarseTwoPhase2x8(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t inputBase, uint64_t outputBase, uint64_t inputToken, uint64_t outputToken,
    uint64_t dataSize, uint64_t localOutputOffset)
{
    if (resCtx.asymmetricPairedRanks.size() != 1U ||
        resCtx.relayKernelIndices.empty()) {
        HCCL_ERROR("[ExecOp] Invalid 2x8 group-safe relay metadata");
        return HCCL_E_INTERNAL;
    }

    const uint64_t segmentSize = CalcTwoPhaseSegmentSize(dataSize);
    const uint64_t directSize = segmentSize * 2U;
    const uint64_t relaySize = dataSize - directSize;
    if (directSize == 0U || relaySize == 0U ||
        segmentSize > MAX_TRANSFER_BYTES || relaySize > MAX_TRANSFER_BYTES) {
        HCCL_ERROR("[ExecOp] Invalid 2x8 coarse split: "
            "dataSize=%llu segment=%llu direct=%llu relay=%llu",
            static_cast<unsigned long long>(dataSize),
            static_cast<unsigned long long>(segmentSize),
            static_cast<unsigned long long>(directSize),
            static_cast<unsigned long long>(relaySize));
        return HCCL_E_INTERNAL;
    }

    // Phase 1 uses only the paired Clos Channel to place the 4/11 suffix at its
    // paired remote rank. Queueing the complete Publish/Write/ACK generation
    // on the main Thread makes the phase-2 pull dependency explicit.
    std::vector<uint64_t> pairSuffixArgs = {
        inputBase + directSize, outputBase + directSize,
        inputToken, outputToken, localOutputOffset, relaySize,
        PAIR_PUBLISH_STAGE,
    };
    CHK_RET(LaunchStagedOperation(param, resCtx,
        pairSuffixArgs[0], pairSuffixArgs[1], pairSuffixArgs[2],
        pairSuffixArgs[3], pairSuffixArgs[4], pairSuffixArgs[5],
        KernelSelection::PAIR, false, false, true, false));

    // Phase 2 keeps both IO Dies busy in one Channel generation. Layer-1 sends
    // the 7/11 prefix as two segment-size Writes. Layer-0 sends those two prefix
    // segments plus the relay suffix to every local peer, then pulls the paired
    // suffix in the same mission. For these two custom stages,
    // remoteOutputOffset carries segmentSize and sliceSize carries relaySize.
    std::vector<uint64_t> directPrefixArgs = {
        inputBase, outputBase, inputToken, outputToken,
        segmentSize, relaySize, PUBLISH_STAGE,
    };
    std::vector<uint64_t> layer0FullArgs = {
        inputBase, outputBase, inputToken, outputToken,
        segmentSize, relaySize, PUBLISH_STAGE,
    };
    CHK_RET(PreSyncLayerThreads(param.cpuThread, resCtx.slaveThreads[0]));
    CHK_RET(LaunchSelectedStage(param, resCtx, layer0FullArgs,
        KernelSelection::LAYER0, true));
    CHK_RET(LaunchSelectedStage(param, resCtx, directPrefixArgs,
        KernelSelection::LAYER1, true));
    CHK_RET(PostSyncLayerThreads(param.cpuThread, resCtx.slaveThreads[0]));
    CHK_RET(PreSyncLayerThreads(param.cpuThread, resCtx.slaveThreads[0]));

    layer0FullArgs.back() = ASYMMETRIC_FUSED_THIRD_WRITE_PULL_RELAY_STAGE;
    directPrefixArgs.back() = ASYMMETRIC_SEGMENTED_DIRECT_WRITE_RECORD_STAGE;
    CHK_RET(LaunchSelectedStage(param, resCtx, layer0FullArgs,
        KernelSelection::LAYER0, true));
    CHK_RET(LaunchSelectedStage(param, resCtx, directPrefixArgs,
        KernelSelection::LAYER1, true));

    layer0FullArgs.back() = ACK_FINAL_WAIT_STAGE;
    directPrefixArgs.back() = ACK_FINAL_WAIT_STAGE;
    CHK_RET(LaunchSelectedStage(param, resCtx, layer0FullArgs,
        KernelSelection::LAYER0, true));
    CHK_RET(LaunchSelectedStage(param, resCtx, directPrefixArgs,
        KernelSelection::LAYER1, true));
    CHK_RET(PostSyncLayerThreads(param.cpuThread, resCtx.slaveThreads[0]));
    return HCCL_SUCCESS;
}

HcclResult Launch8Plus4PairChunk(const OpParam &param,
    const AlgResourceCtx &resCtx, uint64_t inputBase, uint64_t outputBase,
    uint64_t inputToken, uint64_t outputToken, uint64_t localOutputOffset,
    const EqualCycleChunk &chunk)
{
    std::vector<uint64_t> pairTaskArgs = {
        inputBase + chunk.offset,
        outputBase + chunk.offset,
        inputToken,
        outputToken,
        localOutputOffset,
        chunk.size,
        PAIR_PUBLISH_STAGE,
    };
    CHK_RET(LaunchSelectedStage(
        param, resCtx, pairTaskArgs, KernelSelection::PAIR, true));
    pairTaskArgs.back() = PAIR_WRITE_RECORD_STAGE;
    CHK_RET(LaunchSelectedStage(
        param, resCtx, pairTaskArgs, KernelSelection::PAIR, true));
    pairTaskArgs.back() = PAIR_FINAL_WAIT_STAGE;
    CHK_RET(LaunchSelectedStage(
        param, resCtx, pairTaskArgs, KernelSelection::PAIR, true));
    return HCCL_SUCCESS;
}

HcclResult Launch8Plus4Layer0Body(const OpParam &param,
    const AlgResourceCtx &resCtx, uint64_t inputBase, uint64_t outputBase,
    uint64_t inputToken, uint64_t outputToken, uint64_t dataSize,
    uint64_t localOutputOffset, const EqualCycleChunk &chunk)
{
    std::vector<uint64_t> localTaskArgs = {
        inputBase + chunk.offset,
        outputBase,
        inputToken,
        outputToken,
        localOutputOffset + chunk.offset,
        chunk.size,
        chunk.relay ? ASYMMETRIC_LOCAL_WRITE_STAGE : WRITE_RECORD_STAGE,
    };
    CHK_RET(LaunchSelectedStage(
        param, resCtx, localTaskArgs, KernelSelection::LAYER0, true));

    if (chunk.relay) {
        for (uint32_t pairIndex = 0;
            pairIndex < resCtx.asymmetricPairedRanks.size(); ++pairIndex) {
            const uint32_t pairedRemoteRank =
                resCtx.asymmetricPairedRanks[pairIndex];
            const uint64_t pairedOutputOffset = dataSize * pairedRemoteRank;
            std::vector<uint64_t> relayTaskArgs = {
                outputBase + pairedOutputOffset + chunk.offset,
                outputBase,
                outputToken,
                outputToken,
                pairedOutputOffset + chunk.offset,
                chunk.size,
                pairIndex + 1U == resCtx.asymmetricPairedRanks.size() ?
                    ASYMMETRIC_RELAY_WRITE_RECORD_STAGE :
                    ASYMMETRIC_RELAY_WRITE_STAGE,
            };
            CHK_RET(LaunchSelectedStage(
                param, resCtx, relayTaskArgs, KernelSelection::RELAY, true));
        }
    }

    localTaskArgs.back() = ACK_FINAL_WAIT_STAGE;
    CHK_RET(LaunchSelectedStage(
        param, resCtx, localTaskArgs, KernelSelection::LAYER0, true));
    return HCCL_SUCCESS;
}

HcclResult Launch8Plus4Layer0Chunk(const OpParam &param,
    const AlgResourceCtx &resCtx, uint64_t inputBase, uint64_t outputBase,
    uint64_t inputToken, uint64_t outputToken, uint64_t dataSize,
    uint64_t localOutputOffset, const EqualCycleChunk &chunk)
{
    std::vector<uint64_t> localTaskArgs = {
        inputBase + chunk.offset,
        outputBase,
        inputToken,
        outputToken,
        localOutputOffset + chunk.offset,
        chunk.size,
        PUBLISH_STAGE,
    };
    CHK_RET(LaunchSelectedStage(
        param, resCtx, localTaskArgs, KernelSelection::LAYER0, true));
    return Launch8Plus4Layer0Body(param, resCtx, inputBase, outputBase,
        inputToken, outputToken, dataSize, localOutputOffset, chunk);
}

HcclResult Launch8Plus4Layer1DirectChunk(const OpParam &param,
    const AlgResourceCtx &resCtx, uint64_t inputBase, uint64_t outputBase,
    uint64_t inputToken, uint64_t outputToken, uint64_t localOutputOffset,
    const EqualCycleChunk &chunk)
{
    std::vector<uint64_t> directTaskArgs = {
        inputBase + chunk.offset,
        outputBase + chunk.offset,
        inputToken,
        outputToken,
        localOutputOffset,
        chunk.size,
        PUBLISH_STAGE,
    };
    CHK_RET(LaunchSelectedStage(
        param, resCtx, directTaskArgs, KernelSelection::LAYER1, true));
    directTaskArgs.back() = WRITE_RECORD_STAGE;
    CHK_RET(LaunchSelectedStage(
        param, resCtx, directTaskArgs, KernelSelection::LAYER1, true));
    directTaskArgs.back() = ACK_FINAL_WAIT_STAGE;
    CHK_RET(LaunchSelectedStage(
        param, resCtx, directTaskArgs, KernelSelection::LAYER1, true));
    return HCCL_SUCCESS;
}

HcclResult Launch8Plus4InterleavedDirect(const OpParam &param,
    const AlgResourceCtx &resCtx, uint64_t inputBase, uint64_t outputBase,
    uint64_t inputToken, uint64_t outputToken, uint64_t dataSize,
    uint64_t localOutputOffset, const EqualCycleChunk &localChunk,
    const EqualCycleChunk &directChunk)
{
    std::vector<uint64_t> localTaskArgs = {
        inputBase + localChunk.offset,
        outputBase,
        inputToken,
        outputToken,
        localOutputOffset + localChunk.offset,
        localChunk.size,
        PUBLISH_STAGE,
    };
    std::vector<uint64_t> directTaskArgs = {
        inputBase + directChunk.offset,
        outputBase + directChunk.offset,
        inputToken,
        outputToken,
        localOutputOffset,
        directChunk.size,
        PUBLISH_STAGE,
    };
    CHK_RET(LaunchSelectedStage(
        param, resCtx, localTaskArgs, KernelSelection::LAYER0, true));
    CHK_RET(LaunchSelectedStage(
        param, resCtx, directTaskArgs, KernelSelection::LAYER1, true));

    // Both remote-address publications are complete before either Die consumes
    // its next one-record/one-wait generation.
    CHK_RET(PostSyncLayerThreads(param.cpuThread, resCtx.slaveThreads[0]));
    CHK_RET(PreSyncLayerThreads(param.cpuThread, resCtx.slaveThreads[0]));

    directTaskArgs.back() = WRITE_RECORD_STAGE;
    CHK_RET(LaunchSelectedStage(
        param, resCtx, directTaskArgs, KernelSelection::LAYER1, true));
    directTaskArgs.back() = ACK_FINAL_WAIT_STAGE;
    CHK_RET(LaunchSelectedStage(
        param, resCtx, directTaskArgs, KernelSelection::LAYER1, true));
    return Launch8Plus4Layer0Body(param, resCtx, inputBase, outputBase,
        inputToken, outputToken, dataSize, localOutputOffset, localChunk);
}

[[maybe_unused]] HcclResult LaunchStagedPipeline8Plus4(
    const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t inputBase, uint64_t outputBase, uint64_t inputToken, uint64_t outputToken,
    uint64_t dataSize, uint64_t localOutputOffset)
{
    if (resCtx.asymmetricPairedRanks.empty() ||
        resCtx.asymmetricPairedRanks.size() > 2U) {
        HCCL_ERROR("[ExecOp] Invalid 8+4 asymmetric pair count %zu",
            resCtx.asymmetricPairedRanks.size());
        return HCCL_E_INTERNAL;
    }

    std::vector<EqualCycleChunk> chunks;
    CHK_RET(BuildEqualCycleChunks(dataSize, chunks));
    if (chunks.size() != ASYMMETRIC_CYCLE_CHUNK_COUNT ||
        chunks[0].relay) {
        HCCL_ERROR("[ExecOp] Invalid 8+4 equal-cycle chunk plan");
        return HCCL_E_INTERNAL;
    }

    CHK_RET(PreSyncLayerThreads(param.cpuThread, resCtx.slaveThreads[0]));

    // Seed layer-1 one chunk ahead. Every later layer-1 mission overlaps the
    // preceding ready layer-0 mission. A relay mission becomes ready only after
    // its paired Clos mission is closed by the explicit cross-Thread join.
    CHK_RET(Launch8Plus4Layer1DirectChunk(param, resCtx, inputBase,
        outputBase, inputToken, outputToken, localOutputOffset, chunks[0]));
    bool layerThreadsActive = true;
    for (uint32_t index = 1; index < chunks.size(); ++index) {
        const EqualCycleChunk &localChunk = chunks[index - 1U];
        const EqualCycleChunk &layer1Chunk = chunks[index];
        if (!layer1Chunk.relay) {
            CHK_RET(Launch8Plus4InterleavedDirect(param, resCtx, inputBase,
                outputBase, inputToken, outputToken, dataSize,
                localOutputOffset, localChunk, layer1Chunk));
            continue;
        }

        std::vector<uint64_t> localPublishArgs = {
            inputBase + localChunk.offset,
            outputBase,
            inputToken,
            outputToken,
            localOutputOffset + localChunk.offset,
            localChunk.size,
            PUBLISH_STAGE,
        };
        CHK_RET(LaunchSelectedStage(
            param, resCtx, localPublishArgs, KernelSelection::LAYER0, true));
        CHK_RET(Launch8Plus4PairChunk(param, resCtx, inputBase, outputBase,
            inputToken, outputToken, localOutputOffset, layer1Chunk));
        CHK_RET(Launch8Plus4Layer0Body(param, resCtx, inputBase, outputBase,
            inputToken, outputToken, dataSize, localOutputOffset, localChunk));

        CHK_RET(PostSyncLayerThreads(param.cpuThread, resCtx.slaveThreads[0]));
        layerThreadsActive = false;
        if (index + 1U < chunks.size()) {
            CHK_RET(PreSyncLayerThreads(param.cpuThread, resCtx.slaveThreads[0]));
            layerThreadsActive = true;
        }
    }
    if (layerThreadsActive) {
        CHK_RET(PostSyncLayerThreads(param.cpuThread, resCtx.slaveThreads[0]));
    }

    // The last paired block was proven resident by the final join. Drain its
    // local Mesh relay without touching the already-idle layer-1 Thread.
    CHK_RET(Launch8Plus4Layer0Chunk(param, resCtx, inputBase, outputBase,
        inputToken, outputToken, dataSize, localOutputOffset, chunks.back()));
    return HCCL_SUCCESS;
}

HcclResult LaunchCoarseTwoPhase8Plus4(const OpParam &param,
    const AlgResourceCtx &resCtx, uint64_t inputBase, uint64_t outputBase,
    uint64_t inputToken, uint64_t outputToken, uint64_t dataSize,
    uint64_t localOutputOffset)
{
    if (resCtx.asymmetricPairedRanks.empty() ||
        resCtx.asymmetricPairedRanks.size() > 2U) {
        HCCL_ERROR("[ExecOp] Invalid 8+4 coarse pair count %zu",
            resCtx.asymmetricPairedRanks.size());
        return HCCL_E_INTERNAL;
    }

    const uint64_t unalignedQuarter =
        dataSize / ASYMMETRIC_COARSE_QUARTER_DENOMINATOR;
    const uint64_t quarterSize =
        unalignedQuarter / HYBRID_SPLIT_ALIGNMENT * HYBRID_SPLIT_ALIGNMENT;
    const uint64_t directSize = quarterSize * 3U;
    const uint64_t relaySize = dataSize - directSize;
    if (directSize == 0U || directSize >= dataSize ||
        quarterSize > MAX_TRANSFER_BYTES || relaySize > MAX_TRANSFER_BYTES) {
        HCCL_ERROR("[ExecOp] Invalid 8+4 coarse split: "
            "dataSize=%llu quarter=%llu direct=%llu relay=%llu",
            static_cast<unsigned long long>(dataSize),
            static_cast<unsigned long long>(quarterSize),
            static_cast<unsigned long long>(directSize),
            static_cast<unsigned long long>(relaySize));
        return HCCL_E_INTERNAL;
    }
    const EqualCycleChunk directChunk{0U, directSize, false};
    const EqualCycleChunk relayChunk{directSize, relaySize, true};

    // Phase 1 uses only the paired Clos Channels to make every relay suffix
    // resident. The complete Publish/Write/ACK generation stays on the main
    // Thread, so phase 2 never reuses a Channel with an unfinished operation.
    std::vector<uint64_t> pairTaskArgs = {
        inputBase + relayChunk.offset,
        outputBase + relayChunk.offset,
        inputToken,
        outputToken,
        localOutputOffset,
        relayChunk.size,
        PAIR_PUBLISH_STAGE,
    };
    CHK_RET(LaunchStagedOperation(param, resCtx,
        pairTaskArgs[0], pairTaskArgs[1], pairTaskArgs[2],
        pairTaskArgs[3], pairTaskArgs[4], pairTaskArgs[5],
        KernelSelection::PAIR, false, false, true, false));

    // Phase 2 keeps both IO Dies busy in one Channel generation. Layer-1 sends
    // the 3/4 prefix directly to every remote rank. Layer-0 emits three
    // quarter-size prefix writes plus one suffix write, then pulls the paired
    // suffixes in the same mission. Every individual transfer stays <256MiB.
    std::vector<uint64_t> directTaskArgs = {
        inputBase,
        outputBase,
        inputToken,
        outputToken,
        localOutputOffset,
        directChunk.size,
        PUBLISH_STAGE,
    };
    std::vector<uint64_t> layer0FullArgs = {
        inputBase,
        outputBase,
        inputToken,
        outputToken,
        quarterSize,
        relayChunk.size,
        PUBLISH_STAGE,
    };
    CHK_RET(PreSyncLayerThreads(param.cpuThread, resCtx.slaveThreads[0]));
    CHK_RET(LaunchSelectedStage(param, resCtx, layer0FullArgs,
        KernelSelection::LAYER0, true));
    CHK_RET(LaunchSelectedStage(param, resCtx, directTaskArgs,
        KernelSelection::LAYER1, true));
    CHK_RET(PostSyncLayerThreads(param.cpuThread, resCtx.slaveThreads[0]));
    CHK_RET(PreSyncLayerThreads(param.cpuThread, resCtx.slaveThreads[0]));

    layer0FullArgs.back() = ASYMMETRIC_FUSED_QUARTER_WRITE_PULL_RELAY_STAGE;
    directTaskArgs.back() = WRITE_RECORD_STAGE;
    CHK_RET(LaunchSelectedStage(param, resCtx, layer0FullArgs,
        KernelSelection::LAYER0, true));
    CHK_RET(LaunchSelectedStage(param, resCtx, directTaskArgs,
        KernelSelection::LAYER1, true));

    layer0FullArgs.back() = ACK_FINAL_WAIT_STAGE;
    directTaskArgs.back() = ACK_FINAL_WAIT_STAGE;
    CHK_RET(LaunchSelectedStage(param, resCtx, layer0FullArgs,
        KernelSelection::LAYER0, true));
    CHK_RET(LaunchSelectedStage(param, resCtx, directTaskArgs,
        KernelSelection::LAYER1, true));
    CHK_RET(PostSyncLayerThreads(param.cpuThread, resCtx.slaveThreads[0]));
    return HCCL_SUCCESS;
}

HcclResult LaunchHybrid2x8Slice(const OpParam &param, const AlgResourceCtx &resCtx,
    uint64_t inputBase, uint64_t outputBase, uint64_t inputToken, uint64_t outputToken,
    uint64_t dataSize, uint64_t localOutputOffset, uint64_t processedBytes, uint64_t sliceSize)
{
    const uint64_t directSize = CalcHybridDirectSize(sliceSize);
    const uint64_t relaySize = sliceSize - directSize;
    if (directSize == 0 || relaySize == 0) {
        return LaunchDirectSlice(param, resCtx, inputBase + processedBytes,
            outputBase + processedBytes, inputToken, outputToken, localOutputOffset, sliceSize,
            true, false, false, false);
    }

    const uint64_t pairedOutputOffset = dataSize * resCtx.pairedRemoteRank;
    const uint64_t inputAddress = inputBase + processedBytes;
    const uint64_t outputAddress = outputBase + processedBytes;
    const uint64_t relayInputAddress = inputAddress + directSize;
    const uint64_t relayOutputAddress = outputAddress + directSize;

    CHK_RET(PreSyncLayerThreads(param.cpuThread, resCtx.slaveThreads[0]));

    // Phase 1: preserve the proven global Publish -> WriteRecord -> FinalWait order
    // even though layer-0 and pair-only layer-1 carry different address/size tuples.
    std::vector<uint64_t> localTaskArgs = {
        inputAddress, outputAddress, inputToken, outputToken,
        localOutputOffset, sliceSize, PUBLISH_STAGE,
    };
    std::vector<uint64_t> pairTaskArgs = {
        relayInputAddress, relayOutputAddress, inputToken, outputToken,
        localOutputOffset, relaySize, PAIR_PUBLISH_STAGE,
    };
    CHK_RET(LaunchInterleavedStagedOperations(param, resCtx, localTaskArgs,
        KernelSelection::LAYER0, true, pairTaskArgs, KernelSelection::PAIR, true));

    // The slave record proves that the paired suffix has landed in this rank's output buffer.
    CHK_RET(PostSyncLayerThreads(param.cpuThread, resCtx.slaveThreads[0]));
    // Complete the reverse handshake before either Thread can issue the next Post.
    // This preserves strict one-record/one-wait pairing across the phase boundary.
    CHK_RET(PreSyncLayerThreads(param.cpuThread, resCtx.slaveThreads[0]));

    // Phase 2: keep Clos busy with the direct prefix while Mesh relays the received suffix.
    const uint64_t receivedRelayInput =
        outputBase + pairedOutputOffset + processedBytes + directSize;
    std::vector<uint64_t> directTaskArgs = {
        inputAddress, outputAddress, inputToken, outputToken,
        localOutputOffset, directSize, PUBLISH_STAGE,
    };
    std::vector<uint64_t> relayTaskArgs = {
        receivedRelayInput, relayOutputAddress, outputToken, outputToken,
        pairedOutputOffset, relaySize, PUBLISH_STAGE,
    };
    CHK_RET(LaunchInterleavedStagedOperations(param, resCtx, directTaskArgs,
        KernelSelection::LAYER1, true, relayTaskArgs, KernelSelection::LAYER0, true));

    CHK_RET(PostSyncLayerThreads(param.cpuThread, resCtx.slaveThreads[0]));
    return HCCL_SUCCESS;
}

HcclResult PreSyncLayerThreads(ThreadHandle mainThread, ThreadHandle slaveThread)
{
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        mainThread, slaveThread, LAYER_PARALLEL_NOTIFY_INDEX)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        slaveThread, LAYER_PARALLEL_NOTIFY_INDEX, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult PostSyncLayerThreads(ThreadHandle mainThread, ThreadHandle slaveThread)
{
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        mainThread, LAYER_PARALLEL_NOTIFY_INDEX, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        slaveThread, mainThread, LAYER_PARALLEL_NOTIFY_INDEX)));
    return HCCL_SUCCESS;
}

HcclResult PostSyncCopyThread(ThreadHandle mainThread, ThreadHandle copyThread)
{
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(
        mainThread, COPY_COMPLETION_NOTIFY_INDEX, CUSTOM_TIMEOUT)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(
        copyThread, mainThread, COPY_COMPLETION_NOTIFY_INDEX)));
    return HCCL_SUCCESS;
}

} // namespace

HcclResult ExecOp(const OpParam &param)
{
    // 反序列化
    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> seq(ctx, ctx + param.ctxSize);

    if (param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || param.myRank >= param.rankSize) {
        HCCL_ERROR("[ExecOp] Invalid rank information: rank=%u rankSize=%u", param.myRank, param.rankSize);
        return HCCL_E_PARA;
    }

    uint64_t dataTypeSize = 0;
    CHK_RET(GetDataTypeSize(param.dataType, dataTypeSize));

    if (param.count == 0) {
        return HCCL_SUCCESS;
    }
    if (param.count > std::numeric_limits<uint64_t>::max() / dataTypeSize) {
        HCCL_ERROR("[ExecOp] Input byte size overflows uint64_t");
        return HCCL_E_PARA;
    }

    const uint64_t dataSize = param.count * dataTypeSize;
    if (dataSize > std::numeric_limits<uint64_t>::max() / param.rankSize) {
        HCCL_ERROR("[ExecOp] Output byte size overflows uint64_t");
        return HCCL_E_PARA;
    }
    const uint64_t outputSize = dataSize * param.rankSize;
    const uint64_t localOutputOffset = dataSize * param.myRank;

    AlgResourceCtx resCtx;
    resCtx.DeSerialize(seq);
    if (resCtx.small2x8DirectPull > 1U ||
        resCtx.small8Plus4DirectPull > 1U ||
        resCtx.large8Plus4DirectPull > 1U ||
        resCtx.small2x8DirectPull + resCtx.small8Plus4DirectPull +
            resCtx.large8Plus4DirectPull > 1U) {
        HCCL_ERROR("[ExecOp] Invalid adaptive direct-Pull flags");
        return HCCL_E_INTERNAL;
    }

    const bool useSmall2x8DirectPull =
        param.rankSize == RANK_SIZE_2_X_8 &&
        outputSize == FORMAL_SMALL_OUTPUT_SIZE &&
        dataSize == FORMAL_SMALL_2X8_SLICE_SIZE &&
        resCtx.small2x8DirectPull == 1U;
    if (useSmall2x8DirectPull) {
        const uint64_t inputBase =
            reinterpret_cast<uint64_t>(param.inputPtr);
        const uint64_t outputBase =
            reinterpret_cast<uint64_t>(param.outputPtr);
        const uint64_t maxAddress =
            std::numeric_limits<uint64_t>::max();
        if (inputBase > maxAddress - (dataSize - 1U) ||
            outputBase > maxAddress - (outputSize - 1U)) {
            HCCL_ERROR(
                "[ExecOp] Small 2x8 address range overflows uint64_t");
            return HCCL_E_PARA;
        }
        uint64_t inputToken = 0;
        uint64_t outputToken = 0;
        CHK_RET(GetMemToken(inputBase, dataSize, inputToken));
        CHK_RET(GetMemToken(
            outputBase, outputSize, outputToken));
        Small2x8PullResourceCtx smallResCtx;
        smallResCtx.kernels = resCtx.ccuKernels;
        smallResCtx.layers = resCtx.kernelLayers;
        smallResCtx.slaveThreads = resCtx.slaveThreads;
        return LaunchSmall2x8DirectPull(
            param, smallResCtx, inputBase, outputBase,
            inputToken, outputToken, dataSize);
    }

    const bool useSmall8Plus4DirectPull =
        param.rankSize == RANK_SIZE_8_PLUS_4 &&
        dataSize == FORMAL_SMALL_8_PLUS_4_SLICE_SIZE &&
        outputSize == FORMAL_SMALL_8_PLUS_4_OUTPUT_SIZE &&
        resCtx.small8Plus4DirectPull == 1U;
    if (useSmall8Plus4DirectPull) {
        const uint64_t inputBase =
            reinterpret_cast<uint64_t>(param.inputPtr);
        const uint64_t outputBase =
            reinterpret_cast<uint64_t>(param.outputPtr);
        const uint64_t maxAddress =
            std::numeric_limits<uint64_t>::max();
        if (inputBase > maxAddress - (dataSize - 1U) ||
            outputBase > maxAddress - (outputSize - 1U)) {
            HCCL_ERROR(
                "[ExecOp] Small 8+4 address range overflows uint64_t");
            return HCCL_E_PARA;
        }
        uint64_t inputToken = 0;
        uint64_t outputToken = 0;
        CHK_RET(GetMemToken(inputBase, dataSize, inputToken));
        CHK_RET(GetMemToken(
            outputBase, outputSize, outputToken));
        Small8Plus4PullResourceCtx smallResCtx;
        smallResCtx.kernels = resCtx.ccuKernels;
        smallResCtx.layers = resCtx.kernelLayers;
        smallResCtx.slaveThreads = resCtx.slaveThreads;
        return LaunchSmall8Plus4DirectPull(
            param, smallResCtx, inputBase, outputBase,
            inputToken, outputToken, dataSize);
    }

    const bool useLarge8Plus4DirectPull =
        param.rankSize == RANK_SIZE_8_PLUS_4 &&
        dataSize > FORMAL_SMALL_8_PLUS_4_SLICE_SIZE &&
        resCtx.large8Plus4DirectPull == 1U;
    if (useLarge8Plus4DirectPull) {
        const uint64_t inputBase =
            reinterpret_cast<uint64_t>(param.inputPtr);
        const uint64_t outputBase =
            reinterpret_cast<uint64_t>(param.outputPtr);
        const uint64_t maxAddress =
            std::numeric_limits<uint64_t>::max();
        if (inputBase > maxAddress - (dataSize - 1U) ||
            outputBase > maxAddress - (outputSize - 1U)) {
            HCCL_ERROR("[ExecOp] Large 8+4 address range overflows uint64_t");
            return HCCL_E_PARA;
        }
        uint64_t inputToken = 0U;
        uint64_t outputToken = 0U;
        CHK_RET(GetMemToken(inputBase, dataSize, inputToken));
        CHK_RET(GetMemToken(outputBase, outputSize, outputToken));
        Small8Plus4PullResourceCtx largeResCtx;
        largeResCtx.kernels = resCtx.ccuKernels;
        largeResCtx.layers = resCtx.kernelLayers;
        largeResCtx.slaveThreads = resCtx.slaveThreads;
        uint64_t processedBytes = 0U;
        while (processedBytes < dataSize) {
            const uint64_t transferSize = std::min(
                MAX_TRANSFER_BYTES, dataSize - processedBytes);
            CHK_RET(LaunchLarge8Plus4DirectPullGeneration(
                param, largeResCtx, inputBase + processedBytes,
                outputBase + processedBytes, inputToken, outputToken,
                dataSize, transferSize));
            processedBytes += transferSize;
        }
        return HCCL_SUCCESS;
    }

    if (param.rankSize > 1 && resCtx.ccuKernels.empty()) {
        HCCL_ERROR("[ExecOp] No CCU kernels were registered for a multi-rank communicator");
        return HCCL_E_INTERNAL;
    }
    if (resCtx.ccuKernels.size() != resCtx.kernelLayers.size()) {
        HCCL_ERROR("[ExecOp] CCU kernel/layer mapping size mismatch");
        return HCCL_E_INTERNAL;
    }

    bool hasLayer0 = false;
    bool hasLayer1 = false;
    for (const uint32_t layerId : resCtx.kernelLayers) {
        if (layerId >= 2) {
            HCCL_ERROR("[ExecOp] Invalid kernel layer: %u", layerId);
            return HCCL_E_INTERNAL;
        }
        hasLayer0 = hasLayer0 || layerId == 0;
        hasLayer1 = hasLayer1 || layerId == 1;
    }
    const bool parallelLayers = hasLayer0 && hasLayer1;
    const bool pureClosMonolithic =
        !hasLayer0 && hasLayer1 && resCtx.ccuKernels.size() == 1;
    const bool mixedCopyWorker =
        parallelLayers &&
        (param.rankSize == RANK_SIZE_8_PLUS_4 ||
         param.rankSize == RANK_SIZE_2_X_8);
    const size_t expectedSlaveThreadCount =
        mixedCopyWorker ? 2U :
        ((parallelLayers || pureClosMonolithic) ? 1U : 0U);
    if (resCtx.slaveThreads.size() != expectedSlaveThreadCount) {
        HCCL_ERROR("[ExecOp] Unexpected auxiliary Thread count: actual=%zu expected=%zu",
            resCtx.slaveThreads.size(), expectedSlaveThreadCount);
        return HCCL_E_INTERNAL;
    }
    const bool hasPairMetadata =
        resCtx.pairKernelIndex < resCtx.ccuKernels.size() &&
        resCtx.pairedRemoteRank < param.rankSize &&
        resCtx.pairedRemoteRank != param.myRank &&
        resCtx.relayKernelIndices.empty() &&
        resCtx.asymmetricPairedRanks.empty();
    bool asymmetricRanksValid =
        resCtx.asymmetricPairedRanks.size() == 1U ||
        resCtx.asymmetricPairedRanks.size() == 2U;
    for (const uint32_t pairedRank : resCtx.asymmetricPairedRanks) {
        asymmetricRanksValid = asymmetricRanksValid &&
            pairedRank < param.rankSize && pairedRank != param.myRank;
    }
    bool relayKernelsValid = !resCtx.relayKernelIndices.empty();
    for (const uint32_t relayKernelIndex :
        resCtx.relayKernelIndices) {
        relayKernelsValid = relayKernelsValid &&
            relayKernelIndex < resCtx.ccuKernels.size() &&
            resCtx.kernelLayers[relayKernelIndex] == 0U;
    }
    const bool hasGroupSafe2x8Metadata =
        param.rankSize == RANK_SIZE_2_X_8 && parallelLayers &&
        resCtx.pairKernelIndex < resCtx.ccuKernels.size() &&
        resCtx.pairedRemoteRank == MAX_RANK_SIZE &&
        asymmetricRanksValid &&
        resCtx.asymmetricPairedRanks.size() == 1U &&
        resCtx.asymmetricPairRoundMask == 1U &&
        relayKernelsValid &&
        resCtx.kernelLayers[resCtx.pairKernelIndex] == 1U &&
        resCtx.relayKernelIndices.size() <= 2U;
    const bool hasAsymmetric8Plus4Metadata =
        param.rankSize == RANK_SIZE_8_PLUS_4 && parallelLayers &&
        resCtx.pairKernelIndex < resCtx.ccuKernels.size() &&
        resCtx.pairedRemoteRank == MAX_RANK_SIZE &&
        asymmetricRanksValid &&
        resCtx.asymmetricPairRoundMask >= 1U &&
        resCtx.asymmetricPairRoundMask <= 3U &&
        ((resCtx.asymmetricPairedRanks.size() == 2U &&
          resCtx.asymmetricPairRoundMask == 3U) ||
         (resCtx.asymmetricPairedRanks.size() == 1U &&
          resCtx.asymmetricPairRoundMask != 3U)) &&
        relayKernelsValid &&
        resCtx.kernelLayers[resCtx.pairKernelIndex] == 1U &&
        resCtx.relayKernelIndices.size() <= 2U;
    const bool hasNoPairMetadata =
        resCtx.pairKernelIndex == MAX_RANK_SIZE &&
        resCtx.pairedRemoteRank == MAX_RANK_SIZE &&
        resCtx.relayKernelIndices.empty() &&
        resCtx.asymmetricPairedRanks.empty() &&
        resCtx.asymmetricPairRoundMask == 0U;
    if (!hasPairMetadata && !hasGroupSafe2x8Metadata &&
        !hasAsymmetric8Plus4Metadata &&
        !hasNoPairMetadata) {
        HCCL_ERROR("[ExecOp] Inconsistent paired/relay resource metadata");
        return HCCL_E_INTERNAL;
    }
    if (hasPairMetadata &&
        (param.rankSize != RANK_SIZE_2_X_8 || !parallelLayers ||
         resCtx.kernelLayers[resCtx.pairKernelIndex] != 1)) {
        HCCL_ERROR("[ExecOp] Invalid 2x8 paired kernel mapping");
        return HCCL_E_INTERNAL;
    }

    const uint64_t inputBase = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outputBase = reinterpret_cast<uint64_t>(param.outputPtr);
    const uint64_t maxAddress = std::numeric_limits<uint64_t>::max();
    if (inputBase > maxAddress - (dataSize - 1U) || outputBase > maxAddress - (outputSize - 1U)) {
        HCCL_ERROR("[ExecOp] Input or output address range overflows uint64_t");
        return HCCL_E_PARA;
    }

    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    CHK_RET(GetMemToken(inputBase, dataSize, inputToken));
    CHK_RET(GetMemToken(outputBase, outputSize, outputToken));

    const bool useSmall4x1DirectPull =
        param.rankSize == RANK_SIZE_4_X_1 &&
        outputSize == FORMAL_SMALL_OUTPUT_SIZE;
    if (useSmall4x1DirectPull) {
        return LaunchSmall4x1DirectPull(
            param, resCtx, inputBase, outputBase,
            inputToken, outputToken, dataSize);
    }

    const bool useHierarchicalSmall2x8 =
        hasGroupSafe2x8Metadata &&
        dataSize == HYBRID_MESSAGE_THRESHOLD &&
        resCtx.asymmetricPairedRanks[0] ==
            (param.myRank ^ (RANK_SIZE_2_X_8 / 2U));
    if (useHierarchicalSmall2x8) {
        const HcclResult copyResult = static_cast<HcclResult>(
            HcommLocalCopyOnThread(
                param.cpuThread,
                reinterpret_cast<void *>(outputBase + localOutputOffset),
                reinterpret_cast<void *>(inputBase), dataSize));
        if (copyResult != HCCL_SUCCESS) {
            HCCL_ERROR("[ExecOp] 2x8 small local copy failed, ret=%d",
                static_cast<int32_t>(copyResult));
            return copyResult;
        }
        return LaunchHierarchicalSmall2x8(
            param, resCtx, inputBase, outputBase, inputToken, outputToken,
            dataSize, localOutputOffset, resCtx.asymmetricPairedRanks[0]);
    }

    const bool useCoarseTwoPhase2x8 =
        hasGroupSafe2x8Metadata && dataSize > HYBRID_MESSAGE_THRESHOLD;
    if (useCoarseTwoPhase2x8) {
        const ThreadHandle copyThread = resCtx.slaveThreads[COPY_THREAD_INDEX];
        CHK_RET(PreSyncLayerThreads(param.cpuThread, copyThread));

        uint64_t copyOffset = 0;
        while (copyOffset < dataSize) {
            const uint64_t copySize =
                std::min(MAX_TRANSFER_BYTES, dataSize - copyOffset);
            const HcclResult copyResult = static_cast<HcclResult>(
                HcommLocalCopyOnThread(copyThread,
                    reinterpret_cast<void *>(
                        outputBase + localOutputOffset + copyOffset),
                    reinterpret_cast<void *>(inputBase + copyOffset), copySize));
            if (copyResult != HCCL_SUCCESS) {
                HCCL_ERROR("[ExecOp] 2x8 worker local copy failed, ret=%d",
                    static_cast<int32_t>(copyResult));
                const HcclResult joinResult =
                    PostSyncCopyThread(param.cpuThread, copyThread);
                return joinResult == HCCL_SUCCESS ? copyResult : joinResult;
            }
            copyOffset += copySize;
        }

        const HcclResult networkResult =
            LaunchCoarseTwoPhase2x8(param, resCtx, inputBase, outputBase,
            inputToken, outputToken, dataSize, localOutputOffset);
        const HcclResult joinResult =
            PostSyncCopyThread(param.cpuThread, copyThread);
        return networkResult == HCCL_SUCCESS ? joinResult : networkResult;
    }

    const bool useStagedPipeline8Plus4 =
        hasAsymmetric8Plus4Metadata &&
        dataSize > HYBRID_MESSAGE_THRESHOLD;
    if (useStagedPipeline8Plus4) {
        const ThreadHandle copyThread = resCtx.slaveThreads[COPY_THREAD_INDEX];
        CHK_RET(PreSyncLayerThreads(param.cpuThread, copyThread));
        const HcclResult copyResult = static_cast<HcclResult>(
            HcommLocalCopyOnThread(copyThread,
                reinterpret_cast<void *>(outputBase + localOutputOffset),
                reinterpret_cast<void *>(inputBase), dataSize));
        if (copyResult != HCCL_SUCCESS) {
            HCCL_ERROR("[ExecOp] 8+4 asymmetric full local copy failed, ret=%d",
                static_cast<int32_t>(copyResult));
            const HcclResult joinResult =
                PostSyncCopyThread(param.cpuThread, copyThread);
            return joinResult == HCCL_SUCCESS ? copyResult : joinResult;
        }

        const HcclResult networkResult =
            LaunchCoarseTwoPhase8Plus4(param, resCtx, inputBase, outputBase,
                inputToken, outputToken, dataSize, localOutputOffset);
        const HcclResult joinResult =
            PostSyncCopyThread(param.cpuThread, copyThread);
        return networkResult == HCCL_SUCCESS ? joinResult : networkResult;
    }

    const bool useHybrid2x8 =
        hasPairMetadata && dataSize > HYBRID_MESSAGE_THRESHOLD;
    const bool overlap8Plus4LocalCopy =
        param.rankSize == RANK_SIZE_8_PLUS_4 && parallelLayers &&
        !hasPairMetadata && dataSize > HYBRID_MESSAGE_THRESHOLD;

    // The copy worker writes only this rank's output slice, while every network
    // kernel writes slices owned by remote ranks. Start one full-size copy before
    // the network loop and join it once after all slices, instead of repeating a
    // copy mission and two Thread handshakes for every network slice.
    if (overlap8Plus4LocalCopy) {
        const ThreadHandle copyThread = resCtx.slaveThreads[COPY_THREAD_INDEX];
        CHK_RET(PreSyncLayerThreads(param.cpuThread, copyThread));
        const HcclResult copyResult = static_cast<HcclResult>(
            HcommLocalCopyOnThread(copyThread,
                reinterpret_cast<void *>(outputBase + localOutputOffset),
                reinterpret_cast<void *>(inputBase), dataSize));
        if (copyResult != HCCL_SUCCESS) {
            HCCL_ERROR("[ExecOp] 8+4 full local copy failed, ret=%d",
                static_cast<int32_t>(copyResult));
            const HcclResult joinResult =
                PostSyncCopyThread(param.cpuThread, copyThread);
            return joinResult == HCCL_SUCCESS ? copyResult : joinResult;
        }
    }

    uint64_t processedBytes = 0;
    while (processedBytes < dataSize) {
        const uint64_t sliceSize = std::min(MAX_TRANSFER_BYTES, dataSize - processedBytes);
        const uint64_t inputAddress = inputBase + processedBytes;
        const uint64_t outputAddress = outputBase + processedBytes;
        const bool overlapPureClosCopy = param.rankSize > 1 && pureClosMonolithic;
        const uint64_t localOutputAddress =
            outputBase + localOutputOffset + processedBytes;
        if (!overlapPureClosCopy && !overlap8Plus4LocalCopy) {
            const HcclResult copyResult = static_cast<HcclResult>(
                HcommLocalCopyOnThread(param.cpuThread,
                    reinterpret_cast<void *>(localOutputAddress),
                    reinterpret_cast<void *>(inputAddress), sliceSize));
            if (copyResult != HCCL_SUCCESS) {
                HCCL_ERROR("[ExecOp] HcommLocalCopyOnThread failed, ret=%d",
                    static_cast<int32_t>(copyResult));
                return copyResult;
            }
        }

        HcclResult networkResult = HCCL_SUCCESS;
        if (param.rankSize > 1) {
            if (useHybrid2x8 && sliceSize > HYBRID_MESSAGE_THRESHOLD) {
                networkResult = LaunchHybrid2x8Slice(param, resCtx, inputBase, outputBase,
                    inputToken, outputToken, dataSize, localOutputOffset, processedBytes, sliceSize);
            } else {
                // The 8+4 large-message path keeps Publish as a separate, globally
                // synchronized mission, but merges WriteRecord and FinalWait on
                // each fixed layer Thread. This is the same ordering already used
                // by the checker-proven small-message path.
                const bool fuseWriteWait =
                    dataSize <= HYBRID_MESSAGE_THRESHOLD ||
                    overlap8Plus4LocalCopy;
                const bool singleThreadSmall8Plus4 =
                    hasAsymmetric8Plus4Metadata &&
                    param.rankSize == RANK_SIZE_8_PLUS_4 &&
                    dataSize <= HYBRID_MESSAGE_THRESHOLD;
                networkResult = LaunchDirectSlice(param, resCtx, inputAddress, outputAddress, inputToken,
                    outputToken, localOutputOffset, sliceSize,
                    parallelLayers && !singleThreadSmall8Plus4,
                    fuseWriteWait, pureClosMonolithic,
                    overlapPureClosCopy);
            }
        }
        if (networkResult != HCCL_SUCCESS) {
            if (overlap8Plus4LocalCopy) {
                const HcclResult joinResult = PostSyncCopyThread(
                    param.cpuThread, resCtx.slaveThreads[COPY_THREAD_INDEX]);
                if (joinResult != HCCL_SUCCESS) {
                    HCCL_ERROR("[ExecOp] 8+4 copy join failed after network error, ret=%d",
                        static_cast<int32_t>(joinResult));
                }
            }
            return networkResult;
        }
        processedBytes += sliceSize;
    }

    if (overlap8Plus4LocalCopy) {
        CHK_RET(PostSyncCopyThread(
            param.cpuThread, resCtx.slaveThreads[COPY_THREAD_INDEX]));
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
