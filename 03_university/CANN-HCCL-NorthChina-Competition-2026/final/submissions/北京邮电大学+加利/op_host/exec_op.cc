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
#include <cstdint>
#include <limits>
#include <vector>

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>
#include <hcomm/hcomm_primitives.h>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {

constexpr uint64_t HOST_CHUNK_BYTES = 128ULL * 1024ULL * 1024ULL;
constexpr uint32_t MESH_LOAD_UNITS = 4U;
constexpr uint32_t THREAD_FORK_NOTIFY_INDEX = 0U;
constexpr uint32_t THREAD_JOIN_NOTIFY_INDEX = 1U;
constexpr uint32_t HYBRID_PHASE_READY_NOTIFY_INDEX = 2U;

bool GetDataTypeSize(HcclDataType dataType, uint64_t &size)
{
    switch (dataType) {
        case HCCL_DATA_TYPE_INT8:
        case HCCL_DATA_TYPE_UINT8:
        case HCCL_DATA_TYPE_HIF8:
        case HCCL_DATA_TYPE_FP8E4M3:
        case HCCL_DATA_TYPE_FP8E5M2:
        case HCCL_DATA_TYPE_FP8E8M0:
            size = 1U;
            return true;
        case HCCL_DATA_TYPE_INT16:
        case HCCL_DATA_TYPE_UINT16:
        case HCCL_DATA_TYPE_FP16:
        case HCCL_DATA_TYPE_BFP16:
            size = 2U;
            return true;
        case HCCL_DATA_TYPE_INT32:
        case HCCL_DATA_TYPE_UINT32:
        case HCCL_DATA_TYPE_FP32:
            size = 4U;
            return true;
        case HCCL_DATA_TYPE_INT64:
        case HCCL_DATA_TYPE_UINT64:
        case HCCL_DATA_TYPE_FP64:
            size = 8U;
            return true;
        case HCCL_DATA_TYPE_INT128:
            size = 16U;
            return true;
        default:
            return false;
    }
}

bool ExpectedTopology(uint32_t rankSize, FinalTopology &topology)
{
    switch (rankSize) {
        case 4U:
            topology = FinalTopology::TOPOLOGY_4X1;
            return true;
        case 12U:
            topology = FinalTopology::TOPOLOGY_8X4;
            return true;
        case 16U:
            topology = FinalTopology::TOPOLOGY_2X8;
            return true;
        default:
            return false;
    }
}

bool CheckedRange(uintptr_t begin, uint64_t bytes, uintptr_t &end)
{
    if (bytes > static_cast<uint64_t>(std::numeric_limits<uintptr_t>::max() - begin)) {
        return false;
    }
    end = begin + static_cast<uintptr_t>(bytes);
    return true;
}

bool RangesOverlap(uintptr_t leftBegin, uint64_t leftBytes, uintptr_t rightBegin, uint64_t rightBytes)
{
    uintptr_t leftEnd = 0;
    uintptr_t rightEnd = 0;
    if (!CheckedRange(leftBegin, leftBytes, leftEnd) || !CheckedRange(rightBegin, rightBytes, rightEnd)) {
        return true;
    }
    return leftBegin < rightEnd && rightBegin < leftEnd;
}

HcclResult ClassifySourceMode(const OpParam &param, uint64_t sliceBytes, uint64_t outputBytes,
    uint64_t localOffset, AllGatherSourceMode &mode)
{
    const uintptr_t inputBegin = reinterpret_cast<uintptr_t>(param.inputPtr);
    const uintptr_t outputBegin = reinterpret_cast<uintptr_t>(param.outputPtr);
    uintptr_t inputEnd = 0;
    uintptr_t outputEnd = 0;
    CHK_PRT_RET(!CheckedRange(inputBegin, sliceBytes, inputEnd)
                    || !CheckedRange(outputBegin, outputBytes, outputEnd)
                    || localOffset > std::numeric_limits<uintptr_t>::max() - outputBegin,
        HCCL_ERROR("[ClassifySourceMode] address range overflows uintptr_t"), HCCL_E_PARA);
    const uintptr_t canonical = outputBegin + static_cast<uintptr_t>(localOffset);
    if (inputBegin == canonical) {
        mode = AllGatherSourceMode::CANONICAL_OUTPUT;
    } else if (inputBegin < outputEnd && outputBegin < inputEnd) {
        mode = AllGatherSourceMode::STAGED_OUTPUT;
    } else {
        mode = AllGatherSourceMode::DIRECT_INPUT;
    }
    return HCCL_SUCCESS;
}

uint32_t CountBits(uint32_t value)
{
    uint32_t count = 0;
    while (value != 0U) {
        count += value & 1U;
        value >>= 1U;
    }
    return count;
}

uint32_t KernelLoadUnits(const KernelLaunchMeta &meta)
{
    return meta.layerRole == AllGatherLayerRole::LOCAL_MESH
               ? MESH_LOAD_UNITS
               : meta.channelCount;
}

bool IsThreadPhasedDataPath(AllGatherDataPath dataPath)
{
    return dataPath == AllGatherDataPath::HYBRID_RELAY
           || dataPath == AllGatherDataPath::DISTRIBUTED_ROOT
           || dataPath == AllGatherDataPath::DISTRIBUTED_ROOT_CAPPED
           || dataPath == AllGatherDataPath::DISTRIBUTED_ROOT_MATCHED_SERIAL
           || dataPath == AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT
           || dataPath == AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_CAPPED
           || dataPath == AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_MATCHED
           || dataPath == AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_SHARED
           || dataPath == AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_SHARED_5_8
           || dataPath == AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_FULL_SEED_4_7;
}

HcclResult ValidateResource(const OpParam &param, const AlgResourceCtx &resource,
    uint64_t dataTypeSize, uint64_t sliceBytes, uint64_t outputBytes, uint64_t localOffset)
{
    // Engine Context 来自反序列化数据，Profile、Peer 分区和 layer/die 归属均需重新核对。
    FinalTopology expectedTopology = FinalTopology::TOPOLOGY_4X1;
    AllGatherSourceMode expectedSource = AllGatherSourceMode::DIRECT_INPUT;
    CHK_PRT_RET(!ExpectedTopology(param.rankSize, expectedTopology),
        HCCL_ERROR("[ValidateResource] unsupported rank size %u", param.rankSize), HCCL_E_NOT_SUPPORT);
    CHK_RET(ClassifySourceMode(param, sliceBytes, outputBytes, localOffset, expectedSource));
    const AllGatherProfile expectedProfile
        = SelectAllGatherProfile(param.count, param.rankSize, dataTypeSize);
    AllGatherProfileSpec spec;
    CHK_PRT_RET(!GetAllGatherProfileSpec(expectedProfile, spec),
        HCCL_ERROR("[ValidateResource] invalid expected Profile"), HCCL_E_INTERNAL);
    ApplyAllGatherTopologyPolicy(expectedTopology, param.rankSize, spec);
    const AllGatherDataPath expectedDataPath = GetConfiguredAllGatherDataPath(expectedProfile);
    CHK_PRT_RET(expectedDataPath == AllGatherDataPath::INVALID,
        HCCL_ERROR("[ValidateResource] expected Profile has no configured data path"), HCCL_E_INTERNAL);
    const uint64_t expectedFixedSliceBytes
        = expectedProfile == AllGatherProfile::GENERIC ? 0U : sliceBytes;
    const uint64_t expectedCellBytes
        = expectedProfile == AllGatherProfile::GENERIC
              ? 0U
              : (spec.algorithm == AllGatherAlgorithm::FIXED_DIRECT_SMALL
                        ? sliceBytes
                        : spec.cellBytes);
    CHK_PRT_RET(resource.magic != AlgResourceCtx::MAGIC || resource.version != AlgResourceCtx::VERSION,
        HCCL_ERROR("[ValidateResource] invalid EngineCtx magic/version"), HCCL_E_INTERNAL);
    CHK_PRT_RET(resource.rankSize != param.rankSize || resource.topology != expectedTopology
                    || resource.profile != expectedProfile || resource.algorithm != spec.algorithm
                    || resource.dataPath != expectedDataPath
                    || resource.layerPolicy != spec.layerPolicy || resource.sourceMode != expectedSource
                    || resource.fixedSliceBytes != expectedFixedSliceBytes
                    || resource.dataTypeSize != dataTypeSize || resource.cellBytes != expectedCellBytes,
        HCCL_ERROR("[ValidateResource] EngineCtx dispatch does not match invocation"), HCCL_E_INTERNAL);

    bool duplicateKernel = false;
    for (size_t left = 0; left < resource.transferKernels.size(); ++left) {
        for (size_t right = left + 1U; right < resource.transferKernels.size(); ++right) {
            duplicateKernel = duplicateKernel
                              || resource.transferKernels[left] == resource.transferKernels[right];
        }
    }
    for (size_t left = 0; left < resource.hybridPhaseBKernels.size(); ++left) {
        for (size_t right = left + 1U; right < resource.hybridPhaseBKernels.size(); ++right) {
            duplicateKernel = duplicateKernel
                              || resource.hybridPhaseBKernels[left]
                                     == resource.hybridPhaseBKernels[right];
        }
        for (CcuKernelHandle phaseAKernel : resource.transferKernels) {
            duplicateKernel = duplicateKernel
                              || resource.hybridPhaseBKernels[left] == phaseAKernel;
        }
    }
    const bool threadPhasedHybrid = IsThreadPhasedDataPath(expectedDataPath);
    const bool invalidHybridPhaseShape
        = threadPhasedHybrid
              ? resource.hybridPhaseBKernels.size() != resource.transferKernels.size()
              : !resource.hybridPhaseBKernels.empty();
    CHK_PRT_RET(resource.transferKernels.empty()
                    || resource.transferKernels.size() < spec.minKernelCount
                    || resource.transferKernels.size() > spec.maxKernelCount
                    || resource.kernelMeta.size() != resource.transferKernels.size()
                    || resource.extraThreads.size() > AlgResourceCtx::MAX_EXTRA_THREAD_COUNT
                    || param.cpuThread == 0U
                    || std::any_of(resource.transferKernels.begin(), resource.transferKernels.end(),
                        [](CcuKernelHandle kernel) { return kernel == 0U; })
                    || std::any_of(resource.hybridPhaseBKernels.begin(),
                        resource.hybridPhaseBKernels.end(),
                        [](CcuKernelHandle kernel) { return kernel == 0U; })
                    || std::any_of(resource.extraThreads.begin(), resource.extraThreads.end(),
                        [&param](ThreadHandle thread) { return thread == 0U || thread == param.cpuThread; })
                    || duplicateKernel || invalidHybridPhaseShape,
        HCCL_ERROR("[ValidateResource] invalid thread/kernel shape"), HCCL_E_INTERNAL);

    const uint32_t allRankMask = (1U << param.rankSize) - 1U;
    const uint32_t selfMask = 1U << param.myRank;
    CHK_PRT_RET(resource.closLayer == ALLGATHER_INVALID_LAYER
                    || resource.localRankCount == 0U || resource.localRankCount > param.rankSize
                    || (resource.localRankMask & ~allRankMask) != 0U
                    || (resource.localRankMask & selfMask) == 0U
                    || CountBits(resource.localRankMask) != resource.localRankCount,
        HCCL_ERROR("[ValidateResource] invalid RankGraph layer/local-rank metadata"), HCCL_E_INTERNAL);
    const bool validMeshTopo = resource.meshTopoType == COMM_TOPO_1DMESH
                               || resource.meshTopoType == COMM_TOPO_CUSTOM;
    if (expectedTopology == FinalTopology::TOPOLOGY_4X1) {
        CHK_PRT_RET(resource.localRankCount != 1U
                        || (resource.meshLayer == ALLGATHER_INVALID_LAYER
                            && resource.meshTopoType != COMM_TOPO_RESERVED)
                        || (resource.meshLayer != ALLGATHER_INVALID_LAYER && !validMeshTopo)
                        || (resource.meshLayer != ALLGATHER_INVALID_LAYER
                            && resource.meshLayer == resource.closLayer),
            HCCL_ERROR("[ValidateResource] 4x1 must have one local rank"), HCCL_E_INTERNAL);
    } else if (expectedTopology == FinalTopology::TOPOLOGY_2X8) {
        CHK_PRT_RET(resource.localRankCount != 8U || resource.meshLayer == ALLGATHER_INVALID_LAYER
                        || !validMeshTopo
                        || resource.meshLayer == resource.closLayer,
            HCCL_ERROR("[ValidateResource] 2x8 must have eight local ranks"), HCCL_E_INTERNAL);
    } else {
        CHK_PRT_RET((resource.localRankCount != 4U && resource.localRankCount != 8U)
                        || resource.meshLayer == ALLGATHER_INVALID_LAYER
                        || !validMeshTopo
                        || resource.meshLayer == resource.closLayer,
            HCCL_ERROR("[ValidateResource] 8+4 local instance must contain four or eight ranks"),
            HCCL_E_INTERNAL);
    }

    uint32_t channelCount = 0;
    uint32_t primaryCount = 0;
    uint32_t dieMask = 0;
    uint32_t meshDieMask = 0;
    uint32_t closDieMask = 0;
    uint32_t peerUnion = 0;
    uint32_t meshPeerUnion = 0;
    uint32_t closPeerUnion = 0;
    uint32_t copyCount = 0;
    uint32_t meshGroupCount = 0;
    uint32_t closGroupCount = 0;
    std::array<uint32_t, 2> dieLoads{};
    for (size_t index = 0; index < resource.kernelMeta.size(); ++index) {
        const KernelLaunchMeta &meta = resource.kernelMeta[index];
        CHK_PRT_RET(meta.actualDieId >= 2U || meta.channelCount == 0U || meta.isPrimary > 1U
                        || meta.copyLocalSlice > 1U || meta.channelCount > MAX_RANK_SIZE - 1U
                        || meta.peerMask == 0U || (meta.peerMask & ~allRankMask) != 0U
                        || (meta.peerMask & selfMask) != 0U
                        || CountBits(meta.peerMask) != meta.channelCount
                        || (peerUnion & meta.peerMask) != 0U,
            HCCL_ERROR("[ValidateResource] invalid peer/die metadata"), HCCL_E_INTERNAL);
        dieMask |= 1U << meta.actualDieId;
        dieLoads[meta.actualDieId] += KernelLoadUnits(meta);
        peerUnion |= meta.peerMask;
        channelCount += meta.channelCount;
        primaryCount += meta.isPrimary != 0U ? 1U : 0U;
        copyCount += meta.copyLocalSlice != 0U ? 1U : 0U;
        if (meta.layerRole == AllGatherLayerRole::LOCAL_MESH) {
            const uint32_t dieBit = 1U << meta.actualDieId;
            CHK_PRT_RET((meshDieMask & dieBit) != 0U
                            || meta.topoType != resource.meshTopoType
                            || meta.layer != resource.meshLayer
                            || (meta.peerMask & ~(resource.localRankMask & ~selfMask)) != 0U,
                HCCL_ERROR("[ValidateResource] duplicate or invalid Mesh metadata"), HCCL_E_INTERNAL);
            meshDieMask |= dieBit;
            meshPeerUnion |= meta.peerMask;
            ++meshGroupCount;
        } else if (meta.layerRole == AllGatherLayerRole::CLOS) {
            const uint32_t dieBit = 1U << meta.actualDieId;
            CHK_PRT_RET((closDieMask & dieBit) != 0U || meta.topoType != COMM_TOPO_CLOS
                            || meta.layer != resource.closLayer,
                HCCL_ERROR("[ValidateResource] duplicate or invalid Clos metadata"), HCCL_E_INTERNAL);
            closDieMask |= dieBit;
            closPeerUnion |= meta.peerMask;
            ++closGroupCount;
        } else {
            HCCL_ERROR("[ValidateResource] invalid layer role");
            return HCCL_E_INTERNAL;
        }
        CHK_PRT_RET(meta.isPrimary == 0U && meta.copyLocalSlice != 0U,
            HCCL_ERROR("[ValidateResource] non-primary Kernel owns LocalCopy"), HCCL_E_INTERNAL);
    }
    const uint32_t dieCount = CountBits(dieMask);
    const uint32_t primaryDie = resource.kernelMeta[0].actualDieId;
    const bool primaryDieHasMesh
        = std::any_of(resource.kernelMeta.begin(), resource.kernelMeta.end(), [primaryDie](const KernelLaunchMeta &meta) {
              return meta.actualDieId == primaryDie
                     && meta.layerRole == AllGatherLayerRole::LOCAL_MESH;
          });
    bool invalidPrimary
        = primaryDieHasMesh && resource.kernelMeta[0].layerRole != AllGatherLayerRole::LOCAL_MESH;
    for (uint32_t dieId = 0; dieId < 2U; ++dieId) {
        invalidPrimary = invalidPrimary || ((dieMask & (1U << dieId)) != 0U
            && (dieLoads[dieId] < dieLoads[primaryDie]
                || (dieLoads[dieId] == dieLoads[primaryDie] && dieId < primaryDie)));
    }
    const uint32_t expectedLoad
        = MESH_LOAD_UNITS * meshGroupCount + CountBits(closPeerUnion);
    CHK_PRT_RET(channelCount != param.rankSize - 1U || primaryCount != 1U
                    || resource.kernelMeta[0].isPrimary == 0U
                    || peerUnion != (allRankMask & ~selfMask)
                    || dieCount == 0U || resource.extraThreads.size() + 1U != dieCount
                    || dieLoads[0] + dieLoads[1] != expectedLoad
                    || invalidPrimary
                    || (expectedSource == AllGatherSourceMode::DIRECT_INPUT && copyCount != 1U)
                    || (expectedSource != AllGatherSourceMode::DIRECT_INPUT && copyCount != 0U),
        HCCL_ERROR("[ValidateResource] peer coverage or primary metadata is invalid"), HCCL_E_INTERNAL);

    if (spec.layerPolicy == AllGatherLayerPolicy::CLOS_ONLY) {
        CHK_PRT_RET(meshGroupCount != 0U || closGroupCount != 1U
                        || closPeerUnion != (allRankMask & ~selfMask),
            HCCL_ERROR("[ValidateResource] Clos-only Profile has invalid groups"), HCCL_E_INTERNAL);
    } else if (spec.layerPolicy == AllGatherLayerPolicy::LOCAL_MESH_REMOTE_CLOS) {
        const bool invalidMeshGroupCount = meshGroupCount != 1U;
        const bool invalidClosGroupCount = closGroupCount != 1U;
        const bool invalidDieLayout
            = dieCount != 2U || dieMask != 0x3U || resource.extraThreads.size() != 1U
              || resource.transferKernels.size() != 2U
              || meshDieMask == 0U || closDieMask == 0U
              || (meshDieMask & closDieMask) != 0U;
        CHK_PRT_RET(invalidMeshGroupCount
                        || invalidClosGroupCount
                        || invalidDieLayout
                        || meshPeerUnion != (resource.localRankMask & ~selfMask)
                        || closPeerUnion != (allRankMask & ~resource.localRankMask)
                        || resource.meshLayer == resource.closLayer,
            HCCL_ERROR("[ValidateResource] layered Profile violates Mesh/Clos partition"), HCCL_E_INTERNAL);
    } else {
        HCCL_ERROR("[ValidateResource] unsupported layer policy");
        return HCCL_E_INTERNAL;
    }
    const bool needsPrepare = resource.sourceMode == AllGatherSourceMode::STAGED_OUTPUT;
    CHK_PRT_RET((needsPrepare
                    && (resource.localBuffer.addr == nullptr || resource.localBuffer.size == 0U
                        || resource.prepareKernel == 0U))
                    || (!needsPrepare && resource.prepareKernel != 0U),
        HCCL_ERROR("[ValidateResource] prepare-kernel shape does not match source mode"), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult GetMemToken(uint64_t address, uint64_t size, uint64_t &token)
{
    const CcuResult ret = HcommCcuGetMemToken(address, size, &token);
    if (ret != CCU_SUCCESS) {
        HCCL_ERROR("[GetMemToken] failed for address range of %lu bytes, ret %d", size, ret);
        return ConvertCcuToHccl(ret);
    }
    return HCCL_SUCCESS;
}

HcclResult SyncThreads(ThreadHandle producer, ThreadHandle consumer, uint32_t notifyIndex)
{
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(producer, consumer, notifyIndex)));
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(consumer, notifyIndex, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

HcclResult LaunchKernel(ThreadHandle thread, CcuKernelHandle kernel, const uint64_t *args, uint32_t argCount)
{
    const CcuResult ret = HcommCcuKernelLaunch(thread, kernel, args, argCount);
    if (ret != CCU_SUCCESS) {
        HCCL_ERROR("[LaunchKernel] CCU launch failed, ret %d", ret);
        return ConvertCcuToHccl(ret);
    }
    return HCCL_SUCCESS;
}

HcclResult LaunchTransferGroupsOnDie(ThreadHandle thread, const AlgResourceCtx &resource,
    uint32_t dieId, const uint64_t *args, uint32_t argCount)
{
    HcclResult firstError = HCCL_SUCCESS;
    constexpr std::array<AllGatherLayerRole, 2> launchOrder = {
        AllGatherLayerRole::LOCAL_MESH,
        AllGatherLayerRole::CLOS,
    };
    // 每个 Die 内固定 Mesh -> Clos。主组只决定用户 Thread 绑定和 LocalCopy 归属，
    // 不改变通信顺序，避免不同 Rank 因组大小不同形成 Mesh/Clos 交叉等待环。
    for (AllGatherLayerRole role : launchOrder) {
        for (size_t index = 0; index < resource.transferKernels.size(); ++index) {
            if (resource.kernelMeta[index].actualDieId != dieId
                || resource.kernelMeta[index].layerRole != role) {
                continue;
            }
            const HcclResult launchRet
                = LaunchKernel(thread, resource.transferKernels[index], args, argCount);
            if (firstError == HCCL_SUCCESS && launchRet != HCCL_SUCCESS) {
                firstError = launchRet;
            }
        }
    }
    return firstError;
}

HcclResult LaunchTransferGroups(
    const OpParam &param, const AlgResourceCtx &resource, const uint64_t *args, uint32_t argCount)
{
    const uint32_t primaryDie = resource.kernelMeta[0].actualDieId;
    if (resource.extraThreads.empty()) {
        CHK_PRT_RET(resource.layerPolicy == AllGatherLayerPolicy::LOCAL_MESH_REMOTE_CLOS,
            HCCL_ERROR("[LaunchTransferGroups] layered Profile reached single-Die path"),
            HCCL_E_INTERNAL);
        return LaunchTransferGroupsOnDie(param.cpuThread, resource, primaryDie, args, argCount);
    }

    uint32_t secondaryDie = 2U;
    for (const KernelLaunchMeta &meta : resource.kernelMeta) {
        if (meta.actualDieId != primaryDie) {
            secondaryDie = meta.actualDieId;
            break;
        }
    }
    CHK_PRT_RET(secondaryDie >= 2U,
        HCCL_ERROR("[LaunchTransferGroups] secondary Die is missing"), HCCL_E_INTERNAL);

    // Notify 0 将用户流 fork 到副 Thread，两个 Die 并行；同 Die 的 1~2 个 Kernel
    // 由上面的固定层次顺序串行入队。Notify 1 在返回前 join，保证连续调用可安全复用资源。
    const ThreadHandle secondary = resource.extraThreads[0];
    CHK_PRT_RET(secondary == 0U || secondary == param.cpuThread,
        HCCL_ERROR("[LaunchTransferGroups] secondary Thread is not independent"), HCCL_E_INTERNAL);
    CHK_RET(SyncThreads(param.cpuThread, secondary, THREAD_FORK_NOTIFY_INDEX));

    if (IsThreadPhasedDataPath(resource.dataPath)) {
        size_t meshIndex = resource.kernelMeta.size();
        size_t closIndex = resource.kernelMeta.size();
        for (size_t index = 0; index < resource.kernelMeta.size(); ++index) {
            if (resource.kernelMeta[index].layerRole == AllGatherLayerRole::LOCAL_MESH) {
                meshIndex = index;
            } else if (resource.kernelMeta[index].layerRole == AllGatherLayerRole::CLOS) {
                closIndex = index;
            }
        }
        HcclResult firstError = HCCL_SUCCESS;
        if (meshIndex >= resource.kernelMeta.size() || closIndex >= resource.kernelMeta.size()
            || resource.hybridPhaseBKernels.size() != resource.transferKernels.size()) {
            firstError = HCCL_E_INTERNAL;
        } else {
            const ThreadHandle meshThread
                = resource.kernelMeta[meshIndex].actualDieId == primaryDie
                      ? param.cpuThread
                      : secondary;
            const ThreadHandle closThread
                = resource.kernelMeta[closIndex].actualDieId == primaryDie
                      ? param.cpuThread
                      : secondary;
            const HcclResult meshPhaseARet
                = LaunchKernel(meshThread, resource.transferKernels[meshIndex], args, argCount);
            const HcclResult closPhaseARet
                = LaunchKernel(closThread, resource.transferKernels[closIndex], args, argCount);
            if (meshPhaseARet != HCCL_SUCCESS) {
                firstError = meshPhaseARet;
            } else if (closPhaseARet != HCCL_SUCCESS) {
                firstError = closPhaseARet;
            } else {
                // Clos Phase A 的远端屏障完成后唤醒 Mesh Phase B；各自 Phase B 随后并行。
                const HcclResult phaseReadyRet
                    = SyncThreads(closThread, meshThread, HYBRID_PHASE_READY_NOTIFY_INDEX);
                if (phaseReadyRet != HCCL_SUCCESS) {
                    firstError = phaseReadyRet;
                } else {
                    const HcclResult closPhaseBRet = LaunchKernel(
                        closThread, resource.hybridPhaseBKernels[closIndex], args, argCount);
                    const HcclResult meshPhaseBRet = LaunchKernel(
                        meshThread, resource.hybridPhaseBKernels[meshIndex], args, argCount);
                    if (closPhaseBRet != HCCL_SUCCESS) {
                        firstError = closPhaseBRet;
                    } else if (meshPhaseBRet != HCCL_SUCCESS) {
                        firstError = meshPhaseBRet;
                    }
                }
            }
        }
        const HcclResult joinRet
            = SyncThreads(secondary, param.cpuThread, THREAD_JOIN_NOTIFY_INDEX);
        return firstError != HCCL_SUCCESS ? firstError : joinRet;
    }

    const HcclResult primaryRet
        = LaunchTransferGroupsOnDie(param.cpuThread, resource, primaryDie, args, argCount);
    const HcclResult secondaryRet
        = LaunchTransferGroupsOnDie(secondary, resource, secondaryDie, args, argCount);
    // fork 一旦成功，即使任一 Kernel 下发失败也必须建立反向边，不能遗留副 Thread Notify。
    const HcclResult joinRet
        = SyncThreads(secondary, param.cpuThread, THREAD_JOIN_NOTIFY_INDEX);
    if (primaryRet != HCCL_SUCCESS) {
        return primaryRet;
    }
    if (secondaryRet != HCCL_SUCCESS) {
        return secondaryRet;
    }
    return joinRet;
}

HcclResult PrepareOverlappedInput(const OpParam &param, const AlgResourceCtx &resource,
    uint64_t sliceBytes, uint64_t localOffset, uint64_t inputToken, uint64_t outputToken)
{
    // HCCL Buffer 充当有界 bounce buffer，每块至多 128 MiB。目标起点落在源区间内部时
    // 从后向前搬，否则从前向后搬；每次 Prepare Kernel 都完成 input -> scratch -> canonical，
    // 从而形成 memmove 语义。全部 staging 均发生在 Transfer Kernel 的 READY 交换之前。
    const uint64_t scratchBytes = std::min(HOST_CHUNK_BYTES, resource.localBuffer.size);
    CHK_PRT_RET(scratchBytes == 0U || scratchBytes >= MAX_DATA_SIZE,
        HCCL_ERROR("[PrepareOverlappedInput] invalid scratch chunk size %lu", scratchBytes), HCCL_E_MEMORY);
    const uint64_t inputAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outputAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    CHK_PRT_RET(localOffset > std::numeric_limits<uint64_t>::max() - outputAddr,
        HCCL_ERROR("[PrepareOverlappedInput] canonical address overflows"), HCCL_E_PARA);
    const uint64_t canonicalAddr = outputAddr + localOffset;
    const uint64_t scratchAddr = reinterpret_cast<uint64_t>(resource.localBuffer.addr);
    CHK_PRT_RET(RangesOverlap(scratchAddr, scratchBytes, inputAddr, sliceBytes)
                    || RangesOverlap(scratchAddr, scratchBytes, outputAddr, sliceBytes * param.rankSize),
        HCCL_ERROR("[PrepareOverlappedInput] HCCL bounce buffer overlaps user buffers"), HCCL_E_NOT_SUPPORT);

    uint64_t scratchToken = 0;
    CHK_RET(GetMemToken(scratchAddr, scratchBytes, scratchToken));
    uintptr_t inputEnd = 0;
    CHK_PRT_RET(!CheckedRange(static_cast<uintptr_t>(inputAddr), sliceBytes, inputEnd),
        HCCL_ERROR("[PrepareOverlappedInput] input address range overflows"), HCCL_E_PARA);
    const bool copyBackward = canonicalAddr > inputAddr && canonicalAddr < inputEnd;
    uint64_t processed = 0;
    while (processed < sliceBytes) {
        const uint64_t bytes = std::min(scratchBytes, sliceBytes - processed);
        const uint64_t offset = copyBackward ? sliceBytes - processed - bytes : processed;
        const std::array<uint64_t, 8> args = {
            inputAddr,
            canonicalAddr,
            scratchAddr,
            inputToken,
            outputToken,
            scratchToken,
            offset,
            bytes,
        };
        CHK_RET(LaunchKernel(param.cpuThread, resource.prepareKernel, args.data(),
            static_cast<uint32_t>(args.size())));
        processed += bytes;
    }
    return HCCL_SUCCESS;
}

} // namespace

static HcclResult ExecuteResource(
    const OpParam &param, const AlgResourceCtx &resource, bool validateResource)
{
    uint64_t dataTypeSize = 0;
    CHK_PRT_RET(!GetDataTypeSize(param.dataType, dataTypeSize),
        HCCL_ERROR("[ExecOp] unsupported data type %d", param.dataType), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / dataTypeSize,
        HCCL_ERROR("[ExecOp] slice byte count overflows"), HCCL_E_PARA);
    const uint64_t sliceBytes = param.count * dataTypeSize;
    CHK_PRT_RET(param.rankSize == 0U || param.myRank >= param.rankSize
                    || param.opType != HcclCMDType::HCCL_CMD_ALLGATHER
                    || sliceBytes > std::numeric_limits<uint64_t>::max() / param.rankSize,
        HCCL_ERROR("[ExecOp] invalid rank/op or output byte count"), HCCL_E_PARA);
    const uint64_t outputBytes = sliceBytes * param.rankSize;
    const uint64_t localOffset = sliceBytes * param.myRank;
    if (validateResource) {
        CHK_RET(ValidateResource(
            param, resource, dataTypeSize, sliceBytes, outputBytes, localOffset));
    }

    const uint64_t inputAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outputAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    // Output Token 覆盖按 Rank 顺序排列的完整输出区，远端只需交换这一组地址与 Token。
    uint64_t outputToken = 0;
    CHK_RET(GetMemToken(outputAddr, outputBytes, outputToken));

    uint64_t sourceAddr = 0;
    uint64_t sourceToken = 0;
    if (resource.sourceMode == AllGatherSourceMode::CANONICAL_OUTPUT) {
        // 标准 in-place 直接复用 Output Token，不再为同一地址范围申请独立 Source Token。
        CHK_PRT_RET(localOffset > std::numeric_limits<uint64_t>::max() - outputAddr,
            HCCL_ERROR("[ExecOp] canonical source address overflows"), HCCL_E_PARA);
        sourceAddr = outputAddr + localOffset;
        sourceToken = outputToken;
    } else {
        uint64_t inputToken = 0;
        CHK_RET(GetMemToken(inputAddr, sliceBytes, inputToken));
        if (resource.sourceMode == AllGatherSourceMode::STAGED_OUTPUT) {
            CHK_RET(PrepareOverlappedInput(
                param, resource, sliceBytes, localOffset, inputToken, outputToken));
            sourceAddr = outputAddr + localOffset;
            sourceToken = outputToken;
        } else {
            sourceAddr = inputAddr;
            sourceToken = inputToken;
        }
    }

    if (resource.profile != AllGatherProfile::GENERIC) {
        // 固定图在注册时已捕获 Slice 几何，只需动态传入源/目标地址及其 Token。
        const std::array<uint64_t, 4> args = {sourceAddr, outputAddr, sourceToken, outputToken};
        return LaunchTransferGroups(param, resource, args.data(), static_cast<uint32_t>(args.size()));
    }

    // 通用 Direct 按不超过 128 MiB 的块执行，覆盖任意合法字节数及不整齐尾块。
    uint64_t chunkOffset = 0;
    while (chunkOffset < sliceBytes) {
        const uint64_t chunkBytes = std::min(HOST_CHUNK_BYTES, sliceBytes - chunkOffset);
        CHK_PRT_RET(chunkBytes == 0U || chunkBytes >= MAX_DATA_SIZE,
            HCCL_ERROR("[ExecOp] invalid direct chunk size %lu", chunkBytes), HCCL_E_INTERNAL);
        const std::array<uint64_t, 7> args = {
            sourceAddr,
            outputAddr,
            sourceToken,
            outputToken,
            localOffset,
            chunkOffset,
            chunkBytes,
        };
        CHK_RET(LaunchTransferGroups(param, resource, args.data(), static_cast<uint32_t>(args.size())));
        chunkOffset += chunkBytes;
    }
    return HCCL_SUCCESS;
}

HcclResult ExecPreparedOp(const OpParam &param, const AlgResourceCtx &resource)
{
    return ExecuteResource(param, resource, false);
}

HcclResult ExecOp(const OpParam &param)
{
    CHK_PRT_RET(param.resCtx == nullptr || param.ctxSize == 0U,
        HCCL_ERROR("[ExecOp] EngineCtx is empty"), HCCL_E_PTR);
    CHK_PRT_RET(param.ctxSize > AlgResourceCtx::MAX_SERIALIZED_SIZE,
        HCCL_ERROR("[ExecOp] EngineCtx is too large: %lu", param.ctxSize), HCCL_E_INTERNAL);

    const char *ctx = static_cast<const char *>(param.resCtx);
    std::vector<char> serialized(ctx, ctx + param.ctxSize);
    AlgResourceCtx resource;
    CHK_PRT_RET(!resource.DeSerialize(serialized),
        HCCL_ERROR("[ExecOp] invalid serialized EngineCtx"), HCCL_E_INTERNAL);
    return ExecuteResource(param, resource, true);
}

} // namespace ops_hccl
