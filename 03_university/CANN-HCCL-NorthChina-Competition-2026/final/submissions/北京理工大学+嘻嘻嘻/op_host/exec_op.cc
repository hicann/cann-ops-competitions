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

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace ops_hccl {
namespace {
constexpr uint64_t LOCAL_COPY_LOOP_COUNT = 8;
constexpr uint64_t LOCAL_COPY_MEM_SLICE = 8ULL * 4096ULL;
constexpr uint16_t LOOP_ITER_HIGHEST_BIT = 12;
constexpr uint16_t PARALLEL_FIELD_HIGHEST_BIT = 7;
constexpr uint16_t REPEAT_NUM_SHIFT = 55;
constexpr uint16_t REPEAT_LOOP_SHIFT = 48;
constexpr uint16_t TOTAL_LOOP_SHIFT = 41;
constexpr uint32_t MAIN_THREAD_SLOT = 0;
constexpr uint32_t SLAVE_THREAD_SLOT = 1;
constexpr uint32_t NET_LAYER_ZERO = 0;
constexpr uint32_t NET_LAYER_ONE = 1;

using TaskArgs = std::array<uint64_t, TASK_ARG_COUNT>;
using LocalGoArgs = std::array<uint64_t, 4>;

static_assert(TASK_ARG_COUNT == 11, "AllGather CCU task argument ABI must contain eleven entries");

/**
 * @brief 生成从最低位到指定最高位均为一的 64 位掩码
 * @param highestBit 掩码中最高有效位的下标
 * @return 返回生成的低位连续掩码
 */
constexpr uint64_t MakeLowBitMask(uint16_t highestBit)
{
    return highestBit >= 63 ? std::numeric_limits<uint64_t>::max()
                            : ((uint64_t {1} << (highestBit + 1)) - uint64_t {1});
}

/**
 * @brief 按官方 CCU LoopGroup 位布局编码并行尾部参数
 * @param repeatNum 第一段循环的重复次数减一
 * @param repeatLoopIndex 需要重复执行的循环下标
 * @param totalLoopNum 本次并行组中的循环总数
 * @return 返回编码后的并行尾部参数
 */
uint64_t PackParallelParam(uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum)
{
    const uint64_t fieldMask = MakeLowBitMask(PARALLEL_FIELD_HIGHEST_BIT);
    return ((repeatNum & fieldMask) << REPEAT_NUM_SHIFT)
        | ((repeatLoopIndex & fieldMask) << REPEAT_LOOP_SHIFT)
        | ((totalLoopNum & fieldMask) << TOTAL_LOOP_SHIFT);
}

/**
 * @brief 按官方本地 GroupCopy 配置计算四个动态 GoSize 参数
 * @param size 本次本地拷贝的字节数
 * @param localGoArgs 接收地址偏移、循环次数、并行参数和尾部字节数
 * @return 计算成功返回 HCCL_SUCCESS，数据规模超出官方编码范围返回 HCCL_E_PARA
 */
HcclResult CalculateLocalGoArgs(uint64_t size, LocalGoArgs &localGoArgs)
{
    const uint64_t loopSize = LOCAL_COPY_LOOP_COUNT * LOCAL_COPY_MEM_SLICE;
    const uint64_t maxLoopIterNum = MakeLowBitMask(LOOP_ITER_HIGHEST_BIT);
    const uint64_t maxSize = loopSize * (maxLoopIterNum + 1);
    if (size > maxSize) {
        HCCL_ERROR("[ExecOp] local copy size[%llu] exceeds GoSize limit[%llu]",
            static_cast<unsigned long long>(size), static_cast<unsigned long long>(maxSize));
        return HCCL_E_PARA;
    }

    uint64_t loopIterNum = size / loopSize;
    uint64_t residualLoopNum = (size - loopIterNum * loopSize) / LOCAL_COPY_MEM_SLICE;
    uint64_t residualBytes = size - loopIterNum * loopSize - residualLoopNum * LOCAL_COPY_MEM_SLICE;
    if (size == maxSize) {
        loopIterNum = maxLoopIterNum;
        residualLoopNum = LOCAL_COPY_LOOP_COUNT - 1;
        residualBytes = LOCAL_COPY_MEM_SLICE;
    }

    uint64_t parallelParam = 0;
    uint64_t tailBytes = 0;
    if (residualLoopNum != 0 && residualBytes == 0) {
        parallelParam = PackParallelParam(residualLoopNum - 1, 0, 1);
        tailBytes = LOCAL_COPY_MEM_SLICE;
    } else if (residualLoopNum == 0 && residualBytes != 0) {
        parallelParam = PackParallelParam(0, 0, 1);
        tailBytes = residualBytes;
    } else if (residualLoopNum != 0 && residualBytes != 0) {
        parallelParam = PackParallelParam(residualLoopNum - 1, 1, 2);
        tailBytes = residualBytes;
    }

    localGoArgs = {LOCAL_COPY_MEM_SLICE * LOCAL_COPY_LOOP_COUNT * loopIterNum, loopIterNum, parallelParam,
        tailBytes};
    return HCCL_SUCCESS;
}

/**
 * @brief 校验内存区域的闭区间末地址计算不会发生 64 位回绕
 * @param baseAddress 内存区域起始地址
 * @param size 内存区域字节数
 * @return 空区域或闭区间地址可安全表示时返回 true，否则返回 false
 */
bool IsAddressRangeValid(uint64_t baseAddress, uint64_t size)
{
    return size == 0 || baseAddress <= std::numeric_limits<uint64_t>::max() - (size - 1);
}

/**
 * @brief 校验 variant 的 phase 标识与严格单比特掩码完全一致
 * @param variant 待校验的静态 Kernel variant 描述
 * @return phase 标识和掩码合法时返回 true，否则返回 false
 */
bool IsVariantPhaseMetadataValid(const KernelVariantDesc &variant)
{
    if (variant.phaseId > static_cast<uint32_t>(KernelPhaseId::FAST_LOCAL_FORWARD)) {
        return false;
    }
    const uint32_t expectedPhaseMask = uint32_t {1} << variant.phaseId;
    return variant.phaseMask == expectedPhaseMask && (expectedPhaseMask & ALL_PHASE_MASK) != 0;
}

/**
 * @brief 按长链路优先和稳定物理组顺序比较两个待下发 variant
 * @param left 左侧待比较的静态 Kernel variant 描述
 * @param right 右侧待比较的静态 Kernel variant 描述
 * @return left 应先于 right 下发时返回 true，否则返回 false
 */
bool IsVariantLaunchBefore(const KernelVariantDesc &left, const KernelVariantDesc &right)
{
    if (left.layer != right.layer) {
        return left.layer > right.layer;
    }
    if (left.localIoDie != right.localIoDie) {
        return left.localIoDie < right.localIoDie;
    }
    if (left.groupId != right.groupId) {
        return left.groupId < right.groupId;
    }
    return left.variantId < right.variantId;
}

/**
 * @brief 根据调用参数计算单 rank 字节数和完整接收区字节数
 * @param param 本次 AllGather 调用参数
 * @param rankDataBytes 接收单个 rank 的数据字节数
 * @param totalRecvBytes 接收全部 rank 的结果字节数
 * @return 计算成功返回 HCCL_SUCCESS，数据类型或乘法溢出时返回 HCCL_E_PARA
 */
HcclResult CalculateDataBytes(const OpParam &param, uint64_t &rankDataBytes, uint64_t &totalRecvBytes)
{
    const auto dataTypeIter = SIZE_TABLE.find(param.dataType);
    if (dataTypeIter == SIZE_TABLE.end() || dataTypeIter->second == 0) {
        HCCL_ERROR("[ExecOp] unsupported data type[%d]", static_cast<int>(param.dataType));
        return HCCL_E_PARA;
    }
    if (param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || param.myRank >= param.rankSize) {
        HCCL_ERROR("[ExecOp] invalid rank metadata, myRank[%u], rankSize[%u]", param.myRank, param.rankSize);
        return HCCL_E_PARA;
    }

    const uint64_t dataTypeSize = dataTypeIter->second;
    if (param.count > std::numeric_limits<uint64_t>::max() / dataTypeSize) {
        HCCL_ERROR("[ExecOp] rank data size multiplication overflow, count[%llu], dataTypeSize[%llu]",
            static_cast<unsigned long long>(param.count), static_cast<unsigned long long>(dataTypeSize));
        return HCCL_E_PARA;
    }
    rankDataBytes = param.count * dataTypeSize;
    if (rankDataBytes > std::numeric_limits<uint64_t>::max() / param.rankSize) {
        HCCL_ERROR("[ExecOp] receive data size multiplication overflow, rankDataBytes[%llu], rankSize[%u]",
            static_cast<unsigned long long>(rankDataBytes), param.rankSize);
        return HCCL_E_PARA;
    }
    totalRecvBytes = rankDataBytes * param.rankSize;
    return HCCL_SUCCESS;
}

/**
 * @brief 校验当前调用和静态资源上下文能够安全执行 CCU 下发
 * @param param 本次 AllGather 调用参数
 * @param resCtx 已反序列化的静态资源上下文
 * @return 校验通过返回 HCCL_SUCCESS，指针、上下文或线程非法时返回对应错误码
 */
HcclResult ValidateExecutionState(const OpParam &param, const AlgResourceCtx &resCtx)
{
    CHK_RET(resCtx.Validate());
    if (resCtx.rankSize != param.rankSize || resCtx.myRank != param.myRank) {
        HCCL_ERROR("[ExecOp] context rank metadata mismatch, paramRank[%u/%u], contextRank[%u/%u]", param.myRank,
            param.rankSize, resCtx.myRank, resCtx.rankSize);
        return HCCL_E_INTERNAL;
    }
    if (resCtx.mainToSlaveNotifyIdx != 0 || resCtx.slaveToMainNotifyIdx != 0) {
        HCCL_ERROR("[ExecOp] invalid inter-thread notify index");
        return HCCL_E_INTERNAL;
    }
    if (resCtx.hasSlaveThread != 0) {
        if (resCtx.slaveThread == 0 || resCtx.slaveThread == param.cpuThread) {
            HCCL_ERROR("[ExecOp] invalid slave thread or inter-thread notify metadata");
            return HCCL_E_INTERNAL;
        }
    }
    for (uint32_t variantIndex = 0; variantIndex < resCtx.variantCount; ++variantIndex) {
        const KernelVariantDesc &variant = resCtx.variants[variantIndex];
        if (!IsVariantPhaseMetadataValid(variant) || variant.threadSlot > SLAVE_THREAD_SLOT
            || (variant.threadSlot == SLAVE_THREAD_SLOT && resCtx.hasSlaveThread == 0)) {
            HCCL_ERROR("[ExecOp] invalid variant phase or thread slot, variantId[%u], phaseId[%u], "
                       "phaseMask[0x%x], slot[%u]",
                variant.variantId, variant.phaseId, variant.phaseMask, variant.threadSlot);
            return HCCL_E_INTERNAL;
        }
    }
    return HCCL_SUCCESS;
}

/**
 * @brief 校验完整静态 Context 中 peer 与 ChannelHandle 严格一一映射
 * @param resCtx 已通过基础结构校验的静态资源上下文
 * @return 映射一一对应返回 HCCL_SUCCESS，否则返回 HCCL_E_INTERNAL
 */
HcclResult ValidateUniquePeerChannels(const AlgResourceCtx &resCtx)
{
    std::array<ChannelHandle, MAX_RANK_SIZE> peerChannels {};
    std::array<bool, MAX_RANK_SIZE> peerSeen {};
    for (uint32_t variantIndex = 0; variantIndex < resCtx.variantCount; ++variantIndex) {
        const KernelVariantDesc &variant = resCtx.variants[variantIndex];
        for (uint32_t peerIndex = 0; peerIndex < variant.channelCount; ++peerIndex) {
            const uint32_t peerRank = variant.peerRanks[peerIndex];
            const ChannelHandle channel = variant.channels[peerIndex];
            if (peerRank >= resCtx.rankSize || peerRank == resCtx.myRank || channel == 0) {
                HCCL_ERROR("[ExecOp] invalid peer Channel mapping, variantId[%u], peerRank[%u]",
                    variant.variantId, peerRank);
                return HCCL_E_INTERNAL;
            }
            if (peerSeen[peerRank] && peerChannels[peerRank] != channel) {
                HCCL_ERROR("[ExecOp] peer uses multiple Channel handles, peerRank[%u]", peerRank);
                return HCCL_E_INTERNAL;
            }
            for (uint32_t recordedRank = 0; recordedRank < resCtx.rankSize; ++recordedRank) {
                if (recordedRank != peerRank && peerSeen[recordedRank] && peerChannels[recordedRank] == channel) {
                    HCCL_ERROR("[ExecOp] Channel handle maps to multiple peers, peerRank[%u], recordedRank[%u]",
                        peerRank, recordedRank);
                    return HCCL_E_INTERNAL;
                }
            }
            peerSeen[peerRank] = true;
            peerChannels[peerRank] = channel;
        }
    }
    return HCCL_SUCCESS;
}

/**
 * @brief 校验 4×1 recursive-doubling 两轮静态 peer 与 segment 契约
 * @param resCtx 已校验通用结构的静态资源上下文
 * @return 全局 rank 异或配对、源类型和资源组完整时返回 HCCL_SUCCESS，否则返回 HCCL_E_INTERNAL
 */
HcclResult ValidateFourByOneRecursiveDoubling(const AlgResourceCtx &resCtx)
{
    constexpr uint32_t FOUR_BY_ONE_RANK_SIZE = 4;
    constexpr uint32_t FIRST_ROUND_RES_GROUP = 1;
    constexpr uint32_t SECOND_ROUND_RES_GROUP = 2;
    if (resCtx.rankSize != FOUR_BY_ONE_RANK_SIZE || resCtx.myRank >= resCtx.rankSize) {
        HCCL_ERROR("[ExecOp] invalid 4x1 rank metadata, myRank[%u], rankSize[%u]", resCtx.myRank, resCtx.rankSize);
        return HCCL_E_INTERNAL;
    }

    const KernelVariantDesc *firstRound = nullptr;
    const KernelVariantDesc *secondRound = nullptr;
    for (uint32_t variantIndex = 0; variantIndex < resCtx.variantCount; ++variantIndex) {
        const KernelVariantDesc &variant = resCtx.variants[variantIndex];
        if (variant.phaseMask == PHASE_FAST_LOCAL_GATHER) {
            if (firstRound != nullptr) {
                HCCL_ERROR("[ExecOp] duplicate 4x1 first-round variant");
                return HCCL_E_INTERNAL;
            }
            firstRound = &variant;
        } else if (variant.phaseMask == PHASE_FAST_LOCAL_FORWARD) {
            if (secondRound != nullptr) {
                HCCL_ERROR("[ExecOp] duplicate 4x1 second-round variant");
                return HCCL_E_INTERNAL;
            }
            secondRound = &variant;
        }
    }
    if (firstRound == nullptr || secondRound == nullptr) {
        HCCL_ERROR("[ExecOp] incomplete 4x1 recursive-doubling variants");
        return HCCL_E_INTERNAL;
    }

    const uint32_t expectedFirstRoundPeer = resCtx.myRank ^ 1U;
    const uint32_t expectedSecondRoundPeer = resCtx.myRank ^ 2U;
    const uint32_t expectedPairBaseRank = resCtx.myRank & ~1U;
    if (firstRound->variantType != static_cast<uint32_t>(KernelVariantType::FAST_LANE)
        || firstRound->resGroupId != FIRST_ROUND_RES_GROUP || firstRound->layer != NET_LAYER_ONE
        || firstRound->groupId != NET_LAYER_ONE * MAX_IO_DIE_COUNT + firstRound->localIoDie
        || firstRound->threadSlot != MAIN_THREAD_SLOT || firstRound->notifySlot != 0
        || firstRound->sourceKind != static_cast<uint32_t>(SourceKind::INPUT) || firstRound->handleSelf != 1
        || firstRound->channelCount != 1 || firstRound->segmentCount != 1
        || firstRound->peerRanks[0] != expectedFirstRoundPeer || firstRound->segmentRanks[0] != resCtx.myRank
        || firstRound->operationCount != 1 || firstRound->eventBits[0] != 0 || firstRound->selfEventBit != 1
        || firstRound->completionMask != 0x0003) {
        HCCL_ERROR("[ExecOp] invalid 4x1 first-round metadata, variantId[%u]", firstRound->variantId);
        return HCCL_E_INTERNAL;
    }
    if (secondRound->variantType != static_cast<uint32_t>(KernelVariantType::FAST_FORWARD)
        || secondRound->resGroupId != SECOND_ROUND_RES_GROUP || secondRound->layer != NET_LAYER_ONE
        || secondRound->groupId != NET_LAYER_ONE * MAX_IO_DIE_COUNT + secondRound->localIoDie
        || secondRound->threadSlot != MAIN_THREAD_SLOT || secondRound->notifySlot != 0
        || secondRound->sourceKind != static_cast<uint32_t>(SourceKind::OUTPUT) || secondRound->handleSelf != 0
        || secondRound->channelCount != 1 || secondRound->segmentCount != 2
        || secondRound->peerRanks[0] != expectedSecondRoundPeer
        || secondRound->segmentRanks[0] != expectedPairBaseRank
        || secondRound->segmentRanks[1] != expectedPairBaseRank + 1U || secondRound->operationCount != 2
        || secondRound->eventBits[0] != 0 || secondRound->eventBits[1] != 1
        || secondRound->selfEventBit != INVALID_EVENT_BIT || secondRound->completionMask != 0x0003
        || secondRound->channels[0] == firstRound->channels[0]) {
        HCCL_ERROR("[ExecOp] invalid 4x1 second-round metadata, variantId[%u]", secondRound->variantId);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

/**
 * @brief 聚合校验一个快路径 phase 内指定类型在按 IO Die 拆分后的精确 peer 与 segment 契约
 * @param resCtx 已校验通用结构的静态资源上下文
 * @param phaseMask 待校验的唯一 phase 位
 * @param variantType 当前 phase 预期的 variant 类型
 * @param resGroupId 当前 phase 预期的资源组编号
 * @param layer 当前 phase 预期使用的网络层
 * @param sourceKind 当前 phase 预期读取的源内存类别
 * @param expectedPeers 当前 rank 在本 phase 预期使用的完整 peer 集
 * @param expectedPeerCount expectedPeers 中有效 peer 的数量
 * @param expectedSegments 当前 phase 每个物理 variant 预期携带的完整 segment 列表
 * @param expectedSegmentCount expectedSegments 中有效 segment 的数量
 * @param expectedHandleSelfCount 当前 variant 集预期的 handleSelf 总数
 * @return 聚合 peer、逐 variant 元数据和线程放置均匹配时返回 HCCL_SUCCESS，否则返回 HCCL_E_INTERNAL
 */
HcclResult ValidateFastVariantSet(const AlgResourceCtx &resCtx, uint32_t phaseMask,
    KernelVariantType variantType, uint32_t resGroupId, uint32_t layer, SourceKind sourceKind,
    std::array<uint32_t, MAX_PEER_COUNT> expectedPeers, uint32_t expectedPeerCount,
    const std::array<uint32_t, MAX_SEGMENT_COUNT> &expectedSegments, uint32_t expectedSegmentCount,
    uint32_t expectedHandleSelfCount)
{
    if (expectedPeerCount == 0 || expectedPeerCount > MAX_PEER_COUNT || expectedSegmentCount == 0
        || expectedSegmentCount > MAX_SEGMENT_COUNT || expectedHandleSelfCount > 1 || layer > NET_LAYER_ONE) {
        HCCL_ERROR("[ExecOp] invalid expected fast variant set, phase[0x%x], peers[%u], segments[%u], layer[%u]",
            phaseMask, expectedPeerCount, expectedSegmentCount, layer);
        return HCCL_E_INTERNAL;
    }

    std::array<uint32_t, MAX_PEER_COUNT> actualPeers {};
    std::array<bool, MAX_IO_DIE_COUNT> ioDieSeen {};
    std::array<bool, MAX_IO_DIE_COUNT> phaseIoDieSeen {};
    uint32_t actualPeerCount = 0;
    uint32_t variantCount = 0;
    uint32_t ioDieCount = 0;
    uint32_t phaseIoDieCount = 0;
    uint32_t handleSelfCount = 0;
    for (uint32_t variantIndex = 0; variantIndex < resCtx.variantCount; ++variantIndex) {
        const KernelVariantDesc &variant = resCtx.variants[variantIndex];
        if (variant.phaseMask != phaseMask) {
            continue;
        }
        if (variant.localIoDie >= MAX_IO_DIE_COUNT) {
            HCCL_ERROR("[ExecOp] invalid phase IO Die, variantId[%u], die[%u]",
                variant.variantId, variant.localIoDie);
            return HCCL_E_INTERNAL;
        }
        if (!phaseIoDieSeen[variant.localIoDie]) {
            phaseIoDieSeen[variant.localIoDie] = true;
            ++phaseIoDieCount;
        }
    }
    for (uint32_t variantIndex = 0; variantIndex < resCtx.variantCount; ++variantIndex) {
        const KernelVariantDesc &variant = resCtx.variants[variantIndex];
        if (variant.phaseMask != phaseMask
            || variant.variantType != static_cast<uint32_t>(variantType)) {
            continue;
        }
        ++variantCount;
        if (variant.resGroupId != resGroupId || variant.layer != layer
            || variant.groupId != layer * MAX_IO_DIE_COUNT + variant.localIoDie
            || variant.sourceKind != static_cast<uint32_t>(sourceKind) || variant.notifySlot != 0
            || variant.segmentCount != expectedSegmentCount) {
            HCCL_ERROR("[ExecOp] fast variant metadata mismatch, variantId[%u], phase[0x%x]",
                variant.variantId, phaseMask);
            return HCCL_E_INTERNAL;
        }
        if (variant.localIoDie >= MAX_IO_DIE_COUNT || ioDieSeen[variant.localIoDie]) {
            HCCL_ERROR("[ExecOp] duplicate or invalid fast variant IO Die, variantId[%u], die[%u]",
                variant.variantId, variant.localIoDie);
            return HCCL_E_INTERNAL;
        }
        ioDieSeen[variant.localIoDie] = true;
        ++ioDieCount;

        for (uint32_t segmentIndex = 0; segmentIndex < expectedSegmentCount; ++segmentIndex) {
            if (variant.segmentRanks[segmentIndex] != expectedSegments[segmentIndex]) {
                HCCL_ERROR("[ExecOp] fast variant segment mismatch, variantId[%u], segmentIndex[%u], "
                           "actual[%u], expected[%u]",
                    variant.variantId, segmentIndex, variant.segmentRanks[segmentIndex],
                    expectedSegments[segmentIndex]);
                return HCCL_E_INTERNAL;
            }
        }

        if (variant.handleSelf > 1) {
            HCCL_ERROR("[ExecOp] invalid fast variant handleSelf, variantId[%u], handleSelf[%u]",
                variant.variantId, variant.handleSelf);
            return HCCL_E_INTERNAL;
        }
        handleSelfCount += variant.handleSelf;
        const uint32_t expectedOperationCount = variant.channelCount * expectedSegmentCount;
        const uint32_t expectedSelfEventBit
            = variant.handleSelf != 0 ? expectedOperationCount : INVALID_EVENT_BIT;
        const uint32_t expectedEventCount = expectedOperationCount + variant.handleSelf;
        const uint16_t expectedCompletionMask
            = static_cast<uint16_t>((uint32_t {1} << expectedEventCount) - 1U);
        if (variant.operationCount != expectedOperationCount || variant.selfEventBit != expectedSelfEventBit
            || variant.completionMask != expectedCompletionMask) {
            HCCL_ERROR("[ExecOp] fast variant operation mismatch, variantId[%u], operations[%u/%u], "
                       "selfEventBit[%u/%u], completionMask[0x%x/0x%x]",
                variant.variantId, variant.operationCount, expectedOperationCount, variant.selfEventBit,
                expectedSelfEventBit, variant.completionMask, expectedCompletionMask);
            return HCCL_E_INTERNAL;
        }
        for (uint32_t operationIndex = 0; operationIndex < expectedOperationCount; ++operationIndex) {
            if (variant.eventBits[operationIndex] != operationIndex) {
                HCCL_ERROR("[ExecOp] fast variant event bit mismatch, variantId[%u], operationIndex[%u], bit[%u]",
                    variant.variantId, operationIndex, variant.eventBits[operationIndex]);
                return HCCL_E_INTERNAL;
            }
        }

        if (actualPeerCount > MAX_PEER_COUNT - variant.channelCount) {
            HCCL_ERROR("[ExecOp] aggregated fast peer count exceeds limit, phase[0x%x]", phaseMask);
            return HCCL_E_INTERNAL;
        }
        for (uint32_t peerIndex = 0; peerIndex < variant.channelCount; ++peerIndex) {
            const uint32_t peerRank = variant.peerRanks[peerIndex];
            if (std::find(actualPeers.begin(), actualPeers.begin() + actualPeerCount, peerRank)
                != actualPeers.begin() + actualPeerCount) {
                HCCL_ERROR("[ExecOp] duplicate aggregated fast peer, phase[0x%x], peerRank[%u]",
                    phaseMask, peerRank);
                return HCCL_E_INTERNAL;
            }
            actualPeers[actualPeerCount++] = peerRank;
        }
    }

    if (variantCount == 0 || variantCount != ioDieCount || variantCount > MAX_IO_DIE_COUNT
        || actualPeerCount != expectedPeerCount || handleSelfCount != expectedHandleSelfCount) {
        HCCL_ERROR("[ExecOp] incomplete fast variant set, phase[0x%x], variants[%u], dies[%u], "
                   "peers[%u/%u], handleSelf[%u/%u]",
            phaseMask, variantCount, ioDieCount, actualPeerCount, expectedPeerCount, handleSelfCount,
            expectedHandleSelfCount);
        return HCCL_E_INTERNAL;
    }

    std::sort(actualPeers.begin(), actualPeers.begin() + actualPeerCount);
    std::sort(expectedPeers.begin(), expectedPeers.begin() + expectedPeerCount);
    for (uint32_t peerIndex = 0; peerIndex < expectedPeerCount; ++peerIndex) {
        if (actualPeers[peerIndex] != expectedPeers[peerIndex]) {
            HCCL_ERROR("[ExecOp] aggregated fast peer mismatch, phase[0x%x], index[%u], actual[%u], expected[%u]",
                phaseMask, peerIndex, actualPeers[peerIndex], expectedPeers[peerIndex]);
            return HCCL_E_INTERNAL;
        }
    }

    for (uint32_t variantIndex = 0; variantIndex < resCtx.variantCount; ++variantIndex) {
        const KernelVariantDesc &variant = resCtx.variants[variantIndex];
        if (variant.phaseMask != phaseMask) {
            continue;
        }
        const uint32_t expectedThreadSlot
            = phaseIoDieCount == MAX_IO_DIE_COUNT && variant.localIoDie == 1U
                ? SLAVE_THREAD_SLOT
                : MAIN_THREAD_SLOT;
        if (variant.threadSlot != expectedThreadSlot) {
            HCCL_ERROR("[ExecOp] phase variant thread slot mismatch, variantId[%u], slot[%u/%u]",
                variant.variantId, variant.threadSlot, expectedThreadSlot);
            return HCCL_E_INTERNAL;
        }
    }
    return HCCL_SUCCESS;
}

/**
 * @brief 从 direct fallback 重建 8+4 分组并校验对顶双播种与 Q3 三邻居分发契约
 * @param resCtx 已校验通用结构的静态资源上下文
 * @return 分组、lane 和 forward 精确匹配时返回 HCCL_SUCCESS，否则返回 HCCL_E_INTERNAL
 */
HcclResult ValidateEightPlusFourOppositeSeeding(const AlgResourceCtx &resCtx)
{
    constexpr uint32_t EIGHT_PLUS_FOUR_RANK_SIZE = 12;
    constexpr uint32_t LARGE_SERVER_RANK_SIZE = 8;
    constexpr uint32_t SMALL_SERVER_RANK_SIZE = 4;
    constexpr uint32_t OPPOSITE_MASK = 7;
    constexpr uint32_t DIRECT_RES_GROUP = 0;
    constexpr uint32_t GATHER_RES_GROUP = 1;
    constexpr uint32_t FORWARD_RES_GROUP = 3;
    if (resCtx.rankSize != EIGHT_PLUS_FOUR_RANK_SIZE || resCtx.myRank >= resCtx.rankSize) {
        HCCL_ERROR("[ExecOp] invalid 8+4 rank metadata, myRank[%u], rankSize[%u]",
            resCtx.myRank, resCtx.rankSize);
        return HCCL_E_INTERNAL;
    }

    std::array<bool, MAX_RANK_SIZE> localMembers {};
    std::array<bool, MAX_RANK_SIZE> remoteMembers {};
    localMembers[resCtx.myRank] = true;
    uint32_t localRankCount = 1;
    uint32_t remoteRankCount = 0;
    for (uint32_t variantIndex = 0; variantIndex < resCtx.variantCount; ++variantIndex) {
        const KernelVariantDesc &variant = resCtx.variants[variantIndex];
        if (variant.phaseMask != PHASE_DIRECT_L0) {
            continue;
        }
        if (variant.variantType != static_cast<uint32_t>(KernelVariantType::DIRECT)
            || variant.resGroupId != DIRECT_RES_GROUP
            || variant.sourceKind != static_cast<uint32_t>(SourceKind::INPUT) || variant.segmentCount != 1
            || variant.segmentRanks[0] != resCtx.myRank || variant.layer > NET_LAYER_ONE) {
            HCCL_ERROR("[ExecOp] invalid 8+4 direct reconstruction variant, variantId[%u]", variant.variantId);
            return HCCL_E_INTERNAL;
        }
        for (uint32_t peerIndex = 0; peerIndex < variant.channelCount; ++peerIndex) {
            const uint32_t peerRank = variant.peerRanks[peerIndex];
            bool *memberSet = variant.layer == NET_LAYER_ZERO ? localMembers.data() : remoteMembers.data();
            uint32_t *memberCount = variant.layer == NET_LAYER_ZERO ? &localRankCount : &remoteRankCount;
            if (peerRank >= resCtx.rankSize || peerRank == resCtx.myRank || memberSet[peerRank]) {
                HCCL_ERROR("[ExecOp] duplicate 8+4 reconstructed peer, variantId[%u], peerRank[%u]",
                    variant.variantId, peerRank);
                return HCCL_E_INTERNAL;
            }
            memberSet[peerRank] = true;
            ++(*memberCount);
        }
    }

    if (!((localRankCount == LARGE_SERVER_RANK_SIZE && remoteRankCount == SMALL_SERVER_RANK_SIZE)
            || (localRankCount == SMALL_SERVER_RANK_SIZE && remoteRankCount == LARGE_SERVER_RANK_SIZE))) {
        HCCL_ERROR("[ExecOp] invalid reconstructed 8+4 group sizes, local[%u], remote[%u]",
            localRankCount, remoteRankCount);
        return HCCL_E_INTERNAL;
    }

    std::array<uint32_t, LARGE_SERVER_RANK_SIZE> largeRanks {};
    std::array<uint32_t, SMALL_SERVER_RANK_SIZE> smallRanks {};
    uint32_t largeRankCount = 0;
    uint32_t smallRankCount = 0;
    const bool localServerIsLarge = localRankCount == LARGE_SERVER_RANK_SIZE;
    for (uint32_t rank = 0; rank < resCtx.rankSize; ++rank) {
        if (localMembers[rank] == remoteMembers[rank]) {
            HCCL_ERROR("[ExecOp] incomplete reconstructed 8+4 membership, rank[%u], local[%u], remote[%u]",
                rank, static_cast<uint32_t>(localMembers[rank]), static_cast<uint32_t>(remoteMembers[rank]));
            return HCCL_E_INTERNAL;
        }
        const bool belongsToLarge = localServerIsLarge ? localMembers[rank] : remoteMembers[rank];
        if (belongsToLarge) {
            if (largeRankCount >= LARGE_SERVER_RANK_SIZE) {
                return HCCL_E_INTERNAL;
            }
            largeRanks[largeRankCount++] = rank;
        } else {
            if (smallRankCount >= SMALL_SERVER_RANK_SIZE) {
                return HCCL_E_INTERNAL;
            }
            smallRanks[smallRankCount++] = rank;
        }
    }
    if (largeRankCount != LARGE_SERVER_RANK_SIZE || smallRankCount != SMALL_SERVER_RANK_SIZE) {
        HCCL_ERROR("[ExecOp] reconstructed 8+4 rank count mismatch, large[%u], small[%u]",
            largeRankCount, smallRankCount);
        return HCCL_E_INTERNAL;
    }

    std::array<uint32_t, MAX_PEER_COUNT> gatherPeers {};
    uint32_t gatherPeerCount = 0;
    for (uint32_t rank = 0; rank < resCtx.rankSize; ++rank) {
        if (rank != resCtx.myRank && localMembers[rank]) {
            gatherPeers[gatherPeerCount++] = rank;
        }
    }
    std::array<uint32_t, MAX_SEGMENT_COUNT> gatherSegments {};
    gatherSegments[0] = resCtx.myRank;
    CHK_RET(ValidateFastVariantSet(resCtx, PHASE_FAST_LOCAL_GATHER, KernelVariantType::DIRECT,
        GATHER_RES_GROUP, NET_LAYER_ZERO, SourceKind::INPUT, gatherPeers, gatherPeerCount, gatherSegments, 1, 1));

    std::array<uint32_t, MAX_PEER_COUNT> lanePeers {};
    std::array<uint32_t, MAX_PEER_COUNT> forwardPeers {};
    std::array<uint32_t, MAX_SEGMENT_COUNT> laneSegments {};
    std::array<uint32_t, MAX_SEGMENT_COUNT> forwardSegments {};
    if (localServerIsLarge) {
        uint32_t sideIndex = LARGE_SERVER_RANK_SIZE;
        for (uint32_t index = 0; index < LARGE_SERVER_RANK_SIZE; ++index) {
            if (largeRanks[index] == resCtx.myRank) {
                sideIndex = index;
                break;
            }
        }
        if (sideIndex >= LARGE_SERVER_RANK_SIZE) {
            HCCL_ERROR("[ExecOp] current rank is absent from reconstructed large server, myRank[%u]",
                resCtx.myRank);
            return HCCL_E_INTERNAL;
        }
        const uint32_t oppositeIndex = sideIndex ^ OPPOSITE_MASK;
        const uint32_t laneIndex = std::min(sideIndex, oppositeIndex);
        lanePeers[0] = smallRanks[laneIndex];
        laneSegments[0] = resCtx.myRank;
        CHK_RET(ValidateFastVariantSet(resCtx, PHASE_FAST_LOCAL_GATHER, KernelVariantType::FAST_LANE,
            GATHER_RES_GROUP, NET_LAYER_ONE, SourceKind::INPUT, lanePeers, 1, laneSegments, 1, 0));

        forwardPeers[0] = largeRanks[sideIndex ^ 1U];
        forwardPeers[1] = largeRanks[sideIndex ^ 2U];
        forwardPeers[2] = largeRanks[sideIndex ^ 4U];
        forwardSegments[0] = smallRanks[laneIndex];
        CHK_RET(ValidateFastVariantSet(resCtx, PHASE_FAST_LOCAL_FORWARD, KernelVariantType::FAST_FORWARD,
            FORWARD_RES_GROUP, NET_LAYER_ZERO, SourceKind::OUTPUT, forwardPeers, 3, forwardSegments, 1, 0));
        return HCCL_SUCCESS;
    }

    uint32_t sideIndex = SMALL_SERVER_RANK_SIZE;
    for (uint32_t index = 0; index < SMALL_SERVER_RANK_SIZE; ++index) {
        if (smallRanks[index] == resCtx.myRank) {
            sideIndex = index;
            break;
        }
    }
    if (sideIndex >= SMALL_SERVER_RANK_SIZE) {
        HCCL_ERROR("[ExecOp] current rank is absent from reconstructed small server, myRank[%u]", resCtx.myRank);
        return HCCL_E_INTERNAL;
    }
    const uint32_t oppositeIndex = sideIndex ^ OPPOSITE_MASK;
    lanePeers[0] = largeRanks[sideIndex];
    lanePeers[1] = largeRanks[oppositeIndex];
    laneSegments[0] = resCtx.myRank;
    CHK_RET(ValidateFastVariantSet(resCtx, PHASE_FAST_LOCAL_GATHER, KernelVariantType::FAST_LANE,
        GATHER_RES_GROUP, NET_LAYER_ONE, SourceKind::INPUT, lanePeers, 2, laneSegments, 1, 0));

    uint32_t forwardPeerCount = 0;
    for (uint32_t rank : smallRanks) {
        if (rank != resCtx.myRank) {
            forwardPeers[forwardPeerCount++] = rank;
        }
    }
    forwardSegments[0] = largeRanks[sideIndex];
    forwardSegments[1] = largeRanks[oppositeIndex];
    std::sort(forwardSegments.begin(), forwardSegments.begin() + 2);
    CHK_RET(ValidateFastVariantSet(resCtx, PHASE_FAST_LOCAL_FORWARD, KernelVariantType::FAST_FORWARD,
        FORWARD_RES_GROUP, NET_LAYER_ZERO, SourceKind::OUTPUT, forwardPeers, forwardPeerCount,
        forwardSegments, 2, 0));
    return HCCL_SUCCESS;
}

/**
 * @brief 逐 phase 校验本次选中路径的 variant、线程放置和本地自拷贝归属
 * @param resCtx 已校验的静态资源上下文
 * @param useFastPath 是否选择大消息拓扑快路径
 * @return 路径完整且资源匹配时返回 HCCL_SUCCESS，否则返回 HCCL_E_INTERNAL
 */
HcclResult ValidateSelectedPath(const AlgResourceCtx &resCtx, bool useFastPath)
{
    CHK_RET(ValidateUniquePeerChannels(resCtx));
    constexpr uint32_t FAST_GATHER_PHASE_INDEX = 0;
    constexpr uint32_t FAST_INTER_PHASE_INDEX = 1;
    constexpr uint32_t FAST_FORWARD_PHASE_INDEX = 2;
    std::array<uint32_t, 3> selectedPhases {};
    uint32_t selectedPhaseCount = 1;
    TopologyKind topologyKind = static_cast<TopologyKind>(resCtx.topologyKind);
    if (!useFastPath) {
        selectedPhases[0] = PHASE_DIRECT_L0;
    } else {
        if (topologyKind != TopologyKind::FOUR_BY_ONE && topologyKind != TopologyKind::TWO_BY_EIGHT
            && topologyKind != TopologyKind::EIGHT_PLUS_FOUR) {
            HCCL_ERROR("[ExecOp] unsupported fast path topology kind[%u]", resCtx.topologyKind);
            return HCCL_E_INTERNAL;
        }
        selectedPhases = {PHASE_FAST_LOCAL_GATHER, PHASE_FAST_INTER_SERVER, PHASE_FAST_LOCAL_FORWARD};
        selectedPhaseCount = 3;
    }
    const uint32_t selfCopyPhase = useFastPath ? PHASE_FAST_LOCAL_GATHER : PHASE_DIRECT_L0;
    std::array<uint32_t, 3> phaseVariantCounts {};
    std::array<uint32_t, 3> phaseMainCounts {};
    std::array<uint32_t, 3> phaseSlaveCounts {};
    std::array<uint32_t, 3> phaseHandleSelfCounts {};
    std::array<uint32_t, 3> phaseDirectCounts {};
    std::array<uint32_t, 3> phaseFastLaneCounts {};
    std::array<uint32_t, 3> phaseFastForwardCounts {};
    std::array<uint32_t, 3> phaseReceiveOnlyCounts {};
    std::array<uint32_t, 3> phaseResGroupIds {};
    phaseResGroupIds.fill(std::numeric_limits<uint32_t>::max());

    for (uint32_t variantIndex = 0; variantIndex < resCtx.variantCount; ++variantIndex) {
        const KernelVariantDesc &variant = resCtx.variants[variantIndex];
        if (!IsVariantPhaseMetadataValid(variant) || variant.phaseMask == PHASE_DIRECT_L1) {
            HCCL_ERROR("[ExecOp] invalid variant phase metadata in selected path, variantId[%u], phaseId[%u], "
                       "phaseMask[0x%x]",
                variant.variantId, variant.phaseId, variant.phaseMask);
            return HCCL_E_INTERNAL;
        }

        uint32_t phaseIndex = selectedPhaseCount;
        for (uint32_t selectedIndex = 0; selectedIndex < selectedPhaseCount; ++selectedIndex) {
            if (variant.phaseMask == selectedPhases[selectedIndex]) {
                phaseIndex = selectedIndex;
                break;
            }
        }
        if (phaseIndex == selectedPhaseCount) {
            continue;
        }

        if (variant.threadSlot > SLAVE_THREAD_SLOT || variant.handleSelf > 1
            || variant.variantType > static_cast<uint32_t>(KernelVariantType::RECEIVE_ONLY)) {
            HCCL_ERROR("[ExecOp] invalid selected variant placement, variantId[%u], type[%u], slot[%u], "
                       "handleSelf[%u]",
                variant.variantId, variant.variantType, variant.threadSlot, variant.handleSelf);
            return HCCL_E_INTERNAL;
        }
        if (phaseVariantCounts[phaseIndex] == 0) {
            phaseResGroupIds[phaseIndex] = variant.resGroupId;
        } else if (phaseResGroupIds[phaseIndex] != variant.resGroupId) {
            HCCL_ERROR("[ExecOp] selected phase spans multiple resource groups, phase[0x%x], expectedGroup[%u], "
                       "actualGroup[%u], variantId[%u]",
                selectedPhases[phaseIndex], phaseResGroupIds[phaseIndex], variant.resGroupId, variant.variantId);
            return HCCL_E_INTERNAL;
        }
        ++phaseVariantCounts[phaseIndex];
        phaseHandleSelfCounts[phaseIndex] += variant.handleSelf;
        if (variant.threadSlot == MAIN_THREAD_SLOT) {
            ++phaseMainCounts[phaseIndex];
        } else {
            ++phaseSlaveCounts[phaseIndex];
        }
        if (variant.variantType == static_cast<uint32_t>(KernelVariantType::DIRECT)) {
            ++phaseDirectCounts[phaseIndex];
        } else if (variant.variantType == static_cast<uint32_t>(KernelVariantType::FAST_LANE)) {
            ++phaseFastLaneCounts[phaseIndex];
        } else if (variant.variantType == static_cast<uint32_t>(KernelVariantType::FAST_FORWARD)) {
            ++phaseFastForwardCounts[phaseIndex];
        } else {
            ++phaseReceiveOnlyCounts[phaseIndex];
        }
    }

    for (uint32_t phaseIndex = 0; phaseIndex < selectedPhaseCount; ++phaseIndex) {
        const uint32_t phaseMask = selectedPhases[phaseIndex];
        const bool phaseRequired = !useFastPath || phaseIndex != FAST_INTER_PHASE_INDEX;
        if (phaseVariantCounts[phaseIndex] == 0) {
            if (phaseRequired) {
                HCCL_ERROR("[ExecOp] selected path is missing required phase, fastPath[%u], phase[0x%x]",
                    static_cast<uint32_t>(useFastPath), phaseMask);
                return HCCL_E_INTERNAL;
            }
            continue;
        }

        const uint32_t expectedHandleSelfCount = phaseMask == selfCopyPhase ? 1U : 0U;
        if (phaseMainCounts[phaseIndex] == 0 || phaseHandleSelfCounts[phaseIndex] != expectedHandleSelfCount) {
            HCCL_ERROR("[ExecOp] incomplete selected phase, fastPath[%u], phase[0x%x], variants[%u], main[%u], "
                       "slave[%u], handleSelf[%u], expectedSelf[%u]",
                static_cast<uint32_t>(useFastPath), phaseMask, phaseVariantCounts[phaseIndex],
                phaseMainCounts[phaseIndex], phaseSlaveCounts[phaseIndex], phaseHandleSelfCounts[phaseIndex],
                expectedHandleSelfCount);
            return HCCL_E_INTERNAL;
        }
        if (phaseSlaveCounts[phaseIndex] != 0
            && (resCtx.hasSlaveThread == 0 || resCtx.slaveThread == 0)) {
            HCCL_ERROR("[ExecOp] selected phase requires unavailable slave thread, phase[0x%x], slaveVariants[%u]",
                phaseMask, phaseSlaveCounts[phaseIndex]);
            return HCCL_E_INTERNAL;
        }
    }

    if (!useFastPath) {
        if (phaseDirectCounts[0] != phaseVariantCounts[0]) {
            HCCL_ERROR("[ExecOp] unified direct phase contains non-direct variant, variants[%u], direct[%u]",
                phaseVariantCounts[0], phaseDirectCounts[0]);
            return HCCL_E_INTERNAL;
        }
        return HCCL_SUCCESS;
    }

    if (topologyKind == TopologyKind::FOUR_BY_ONE) {
        if (phaseVariantCounts[FAST_GATHER_PHASE_INDEX] != 1
            || phaseFastLaneCounts[FAST_GATHER_PHASE_INDEX] != 1
            || phaseVariantCounts[FAST_INTER_PHASE_INDEX] != 0
            || phaseVariantCounts[FAST_FORWARD_PHASE_INDEX] != 1
            || phaseFastForwardCounts[FAST_FORWARD_PHASE_INDEX] != 1) {
            HCCL_ERROR("[ExecOp] invalid 4x1 recursive-doubling roles, gatherLane[%u/%u], interVariants[%u], "
                       "forwardVariants[%u/%u]",
                phaseFastLaneCounts[FAST_GATHER_PHASE_INDEX], phaseVariantCounts[FAST_GATHER_PHASE_INDEX],
                phaseVariantCounts[FAST_INTER_PHASE_INDEX], phaseFastForwardCounts[FAST_FORWARD_PHASE_INDEX],
                phaseVariantCounts[FAST_FORWARD_PHASE_INDEX]);
            return HCCL_E_INTERNAL;
        }
        return ValidateFourByOneRecursiveDoubling(resCtx);
    }

    if (topologyKind == TopologyKind::TWO_BY_EIGHT) {
        if (phaseVariantCounts[FAST_INTER_PHASE_INDEX] != 0
            || phaseDirectCounts[FAST_GATHER_PHASE_INDEX] == 0
            || phaseFastLaneCounts[FAST_GATHER_PHASE_INDEX] == 0
            || phaseDirectCounts[FAST_GATHER_PHASE_INDEX] + phaseFastLaneCounts[FAST_GATHER_PHASE_INDEX]
                != phaseVariantCounts[FAST_GATHER_PHASE_INDEX]
            || phaseFastForwardCounts[FAST_FORWARD_PHASE_INDEX]
                != phaseVariantCounts[FAST_FORWARD_PHASE_INDEX]) {
            HCCL_ERROR("[ExecOp] invalid 2x8 fast path roles, gatherDirect[%u], gatherLane[%u], "
                       "gatherVariants[%u], interVariants[%u], forwardVariants[%u/%u]",
                phaseDirectCounts[FAST_GATHER_PHASE_INDEX], phaseFastLaneCounts[FAST_GATHER_PHASE_INDEX],
                phaseVariantCounts[FAST_GATHER_PHASE_INDEX], phaseVariantCounts[FAST_INTER_PHASE_INDEX],
                phaseFastForwardCounts[FAST_FORWARD_PHASE_INDEX], phaseVariantCounts[FAST_FORWARD_PHASE_INDEX]);
            return HCCL_E_INTERNAL;
        }
        return HCCL_SUCCESS;
    }

    if (phaseVariantCounts[FAST_INTER_PHASE_INDEX] != 0
        || phaseDirectCounts[FAST_GATHER_PHASE_INDEX] == 0
        || phaseFastLaneCounts[FAST_GATHER_PHASE_INDEX] == 0
        || phaseDirectCounts[FAST_GATHER_PHASE_INDEX] + phaseFastLaneCounts[FAST_GATHER_PHASE_INDEX]
            != phaseVariantCounts[FAST_GATHER_PHASE_INDEX]
        || phaseFastForwardCounts[FAST_FORWARD_PHASE_INDEX]
            != phaseVariantCounts[FAST_FORWARD_PHASE_INDEX]
        || phaseReceiveOnlyCounts[FAST_GATHER_PHASE_INDEX] != 0
        || phaseReceiveOnlyCounts[FAST_INTER_PHASE_INDEX] != 0
        || phaseReceiveOnlyCounts[FAST_FORWARD_PHASE_INDEX] != 0) {
        HCCL_ERROR("[ExecOp] invalid 8+4 overlap roles, gatherDirect[%u], gatherLane[%u], "
                   "gatherVariants[%u], interVariants[%u], forwardVariants[%u/%u], receiveOnly[%u/%u/%u]",
            phaseDirectCounts[FAST_GATHER_PHASE_INDEX], phaseFastLaneCounts[FAST_GATHER_PHASE_INDEX],
            phaseVariantCounts[FAST_GATHER_PHASE_INDEX], phaseVariantCounts[FAST_INTER_PHASE_INDEX],
            phaseFastForwardCounts[FAST_FORWARD_PHASE_INDEX], phaseVariantCounts[FAST_FORWARD_PHASE_INDEX],
            phaseReceiveOnlyCounts[FAST_GATHER_PHASE_INDEX], phaseReceiveOnlyCounts[FAST_INTER_PHASE_INDEX],
            phaseReceiveOnlyCounts[FAST_FORWARD_PHASE_INDEX]);
        return HCCL_E_INTERNAL;
    }
    return ValidateEightPlusFourOppositeSeeding(resCtx);
}

/**
 * @brief 为指定 variant 装配动态参数并下发一个 CCU Kernel
 * @param thread 本次 Kernel 使用的通信引擎线程
 * @param variant 待下发的静态 Kernel variant 描述
 * @param baseTaskArgs 已填充公共字段且本地 GoSize 为零的任务参数
 * @param localGoArgs handleSelf variant 使用的四个本地拷贝参数
 * @return 下发成功返回 HCCL_SUCCESS，线程、参数或 CCU 下发失败返回对应错误码
 */
HcclResult LaunchVariant(ThreadHandle thread, const KernelVariantDesc &variant, const TaskArgs &baseTaskArgs,
    const LocalGoArgs &localGoArgs)
{
    if (thread == 0 || variant.kernelHandle == 0 || baseTaskArgs.size() != TASK_ARG_COUNT) {
        HCCL_ERROR("[ExecOp] invalid launch metadata, variantId[%u]", variant.variantId);
        return HCCL_E_INTERNAL;
    }

    TaskArgs taskArgs = baseTaskArgs;
    if (variant.handleSelf != 0) {
        taskArgs[TASK_ARG_LOCAL_GO_ADDR_OFFSET] = localGoArgs[0];
        taskArgs[TASK_ARG_LOCAL_GO_LOOP_PARAM] = localGoArgs[1];
        taskArgs[TASK_ARG_LOCAL_GO_PARALLEL_PARAM] = localGoArgs[2];
        taskArgs[TASK_ARG_LOCAL_GO_RESIDUAL] = localGoArgs[3];
    }
    CHK_RET_CCU(HcommCcuKernelLaunch(
        thread, variant.kernelHandle, taskArgs.data(), static_cast<uint32_t>(taskArgs.size())));
    return HCCL_SUCCESS;
}

/**
 * @brief 在最多两个线程上按稳定顺序下发一个 phase 的全部 Kernel variant
 * @param mainThread 本次调用绑定用户流的主线程
 * @param resCtx 已校验的静态资源上下文
 * @param phaseMask 当前需要执行的唯一 phase 位
 * @param baseTaskArgs 当前分片的公共任务参数
 * @param localGoArgs 当前分片的本地拷贝参数
 * @return phase 下发成功返回 HCCL_SUCCESS，同步、资源或 Kernel 下发失败返回对应错误码
 */
HcclResult ExecutePhase(ThreadHandle mainThread, const AlgResourceCtx &resCtx, uint32_t phaseMask,
    const TaskArgs &baseTaskArgs, const LocalGoArgs &localGoArgs)
{
    if (mainThread == 0 || phaseMask == 0 || (phaseMask & (phaseMask - 1)) != 0
        || (phaseMask & ~ALL_PHASE_MASK) != 0) {
        HCCL_ERROR("[ExecOp] invalid phase execution metadata, mainThread[%llu], phaseMask[0x%x]",
            static_cast<unsigned long long>(mainThread), phaseMask);
        return HCCL_E_INTERNAL;
    }

    std::array<const KernelVariantDesc *, MAX_KERNEL_VARIANT_COUNT> mainVariants {};
    std::array<const KernelVariantDesc *, MAX_KERNEL_VARIANT_COUNT> slaveVariants {};
    uint32_t mainVariantCount = 0;
    uint32_t slaveVariantCount = 0;
    for (uint32_t variantIndex = 0; variantIndex < resCtx.variantCount; ++variantIndex) {
        const KernelVariantDesc &variant = resCtx.variants[variantIndex];
        if (variant.phaseMask != phaseMask) {
            continue;
        }

        if (mainVariantCount + slaveVariantCount >= MAX_KERNEL_VARIANT_COUNT) {
            HCCL_ERROR("[ExecOp] phase[0x%x] variant count exceeds limit[%u]", phaseMask,
                MAX_KERNEL_VARIANT_COUNT);
            return HCCL_E_INTERNAL;
        }
        if (variant.threadSlot == MAIN_THREAD_SLOT) {
            mainVariants[mainVariantCount++] = &variant;
        } else if (variant.threadSlot == SLAVE_THREAD_SLOT) {
            slaveVariants[slaveVariantCount++] = &variant;
        } else {
            HCCL_ERROR("[ExecOp] phase[0x%x] contains invalid thread slot[%u], variantId[%u]", phaseMask,
                variant.threadSlot, variant.variantId);
            return HCCL_E_INTERNAL;
        }
    }

    if (mainVariantCount == 0 && slaveVariantCount == 0) {
        HCCL_ERROR("[ExecOp] phase[0x%x] contains no variant", phaseMask);
        return HCCL_E_INTERNAL;
    }

    const auto launchOrder = [](const KernelVariantDesc *left, const KernelVariantDesc *right) {
        return IsVariantLaunchBefore(*left, *right);
    };
    std::sort(mainVariants.begin(), mainVariants.begin() + mainVariantCount, launchOrder);
    std::sort(slaveVariants.begin(), slaveVariants.begin() + slaveVariantCount, launchOrder);

    if (slaveVariantCount == 0) {
        for (uint32_t variantIndex = 0; variantIndex < mainVariantCount; ++variantIndex) {
            CHK_RET(LaunchVariant(mainThread, *mainVariants[variantIndex], baseTaskArgs, localGoArgs));
        }
        return HCCL_SUCCESS;
    }
    if (mainVariantCount == 0 || resCtx.hasSlaveThread == 0 || resCtx.slaveThread == 0
        || resCtx.slaveThread == mainThread) {
        HCCL_ERROR("[ExecOp] phase[0x%x] has invalid slave execution resources, mainVariants[%u], "
                   "slaveVariants[%u]",
            phaseMask, mainVariantCount, slaveVariantCount);
        return HCCL_E_INTERNAL;
    }

    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyRecordOnThread(mainThread, resCtx.slaveThread, resCtx.mainToSlaveNotifyIdx)));
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyWaitOnThread(resCtx.slaveThread, resCtx.mainToSlaveNotifyIdx, CUSTOM_TIMEOUT)));

    for (uint32_t variantIndex = 0; variantIndex < mainVariantCount; ++variantIndex) {
        CHK_RET(LaunchVariant(mainThread, *mainVariants[variantIndex], baseTaskArgs, localGoArgs));
    }
    for (uint32_t variantIndex = 0; variantIndex < slaveVariantCount; ++variantIndex) {
        CHK_RET(LaunchVariant(resCtx.slaveThread, *slaveVariants[variantIndex], baseTaskArgs, localGoArgs));
    }

    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyRecordOnThread(resCtx.slaveThread, mainThread, resCtx.slaveToMainNotifyIdx)));
    CHK_RET(static_cast<HcclResult>(
        HcommThreadNotifyWaitOnThread(mainThread, resCtx.slaveToMainNotifyIdx, CUSTOM_TIMEOUT)));
    return HCCL_SUCCESS;
}

/**
 * @brief 判断当前资源上下文和数据规模是否应执行拓扑快路径
 * @param resCtx 已校验的静态资源上下文
 * @param rankDataBytes 单个 rank 的数据字节数
 * @return 满足大消息、快路径开关和正式拓扑条件时返回 true，否则返回 false
 */
bool ShouldUseFastPath(const AlgResourceCtx &resCtx, uint64_t rankDataBytes)
{
    const TopologyKind topologyKind = static_cast<TopologyKind>(resCtx.topologyKind);
    const bool supportedTopology = topologyKind == TopologyKind::FOUR_BY_ONE
        || topologyKind == TopologyKind::TWO_BY_EIGHT || topologyKind == TopologyKind::EIGHT_PLUS_FOUR;
    return resCtx.fastPathEnabled != 0 && rankDataBytes > SMALL_MESSAGE_THRESHOLD && supportedTopology;
}
}

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }

    uint64_t rankDataBytes = 0;
    uint64_t totalRecvBytes = 0;
    CHK_RET(CalculateDataBytes(param, rankDataBytes, totalRecvBytes));

    if (param.inputPtr == nullptr || param.outputPtr == nullptr) {
        HCCL_ERROR("[ExecOp] input or output pointer is null");
        return HCCL_E_PTR;
    }
    if (param.cpuThread == 0) {
        HCCL_ERROR("[ExecOp] current main thread is invalid");
        return HCCL_E_INTERNAL;
    }
    const uint64_t sendBase = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t recvBase = reinterpret_cast<uint64_t>(param.outputPtr);
    if (!IsAddressRangeValid(sendBase, rankDataBytes) || !IsAddressRangeValid(recvBase, totalRecvBytes)) {
        HCCL_ERROR("[ExecOp] input or output address range overflows");
        return HCCL_E_PARA;
    }

    if (param.rankSize == 1) {
        CHK_RET(static_cast<HcclResult>(
            HcommLocalCopyOnThread(param.cpuThread, param.outputPtr, param.inputPtr, rankDataBytes)));
        return HCCL_SUCCESS;
    }

    CHK_RET(ValidateExecutionState(param, resCtx));

    const bool useFastPath = ShouldUseFastPath(resCtx, rankDataBytes);
    CHK_RET(ValidateSelectedPath(resCtx, useFastPath));

    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(sendBase, rankDataBytes, &inputToken));
    CHK_RET_CCU(HcommCcuGetMemToken(recvBase, totalRecvBytes, &outputToken));
    uint64_t processedBytes = 0;
    while (processedBytes < rankDataBytes) {
        const uint64_t remainingBytes = rankDataBytes - processedBytes;
        const uint64_t sliceBytes = std::min<uint64_t>(remainingBytes, MAX_DATA_SIZE);
        if (sliceBytes == 0 || sliceBytes > remainingBytes
            || processedBytes > std::numeric_limits<uint64_t>::max() - sliceBytes) {
            HCCL_ERROR("[ExecOp] invalid slice progress, processedBytes[%llu], sliceBytes[%llu]",
                static_cast<unsigned long long>(processedBytes), static_cast<unsigned long long>(sliceBytes));
            return HCCL_E_INTERNAL;
        }

        TaskArgs baseTaskArgs {};
        baseTaskArgs[TASK_ARG_SEND_BASE] = sendBase;
        baseTaskArgs[TASK_ARG_INPUT_TOKEN] = inputToken;
        baseTaskArgs[TASK_ARG_RECV_BASE] = recvBase;
        baseTaskArgs[TASK_ARG_OUTPUT_TOKEN] = outputToken;
        baseTaskArgs[TASK_ARG_RANK_DATA_BYTES] = rankDataBytes;
        baseTaskArgs[TASK_ARG_SLICE_OFFSET] = processedBytes;
        baseTaskArgs[TASK_ARG_SLICE_BYTES] = sliceBytes;

        LocalGoArgs localGoArgs {};
        CHK_RET(CalculateLocalGoArgs(sliceBytes, localGoArgs));

        if (useFastPath) {
            CHK_RET(ExecutePhase(
                param.cpuThread, resCtx, PHASE_FAST_LOCAL_GATHER, baseTaskArgs, localGoArgs));
            CHK_RET(ExecutePhase(
                param.cpuThread, resCtx, PHASE_FAST_LOCAL_FORWARD, baseTaskArgs, localGoArgs));
        } else {
            CHK_RET(ExecutePhase(param.cpuThread, resCtx, PHASE_DIRECT_L0, baseTaskArgs, localGoArgs));
        }

        processedBytes += sliceBytes;
    }

    if (processedBytes != rankDataBytes) {
        HCCL_ERROR("[ExecOp] processed byte count mismatch, processed[%llu], expected[%llu]",
            static_cast<unsigned long long>(processedBytes), static_cast<unsigned long long>(rankDataBytes));
        return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}
}
