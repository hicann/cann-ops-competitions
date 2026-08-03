/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CUSTOM_H
#define OPS_HCCL_CUSTOM_H

#include <cstdint>
#include <type_traits>
#include <vector>

#include <hccl/hccl_res.h>
#include <hccl/hccl_types.h>
#include <securec.h>

#include "common.h"
#include "log.h"

constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;
constexpr uint32_t CKE_IDX = 0;
constexpr uint32_t OUTPUT_XN_ID = 1;
constexpr uint32_t TOKEN_XN_ID = 2;
constexpr uint32_t POST_SYNC_ID = 3;

constexpr uint32_t TASK_ARG_COUNT = 11;
constexpr uint32_t MAX_PEER_COUNT = static_cast<uint32_t>(MAX_RANK_SIZE - 1);
constexpr uint32_t MAX_SEGMENT_COUNT = 2;
constexpr uint32_t MAX_OPERATION_COUNT = 16;
constexpr uint32_t MAX_KERNEL_VARIANT_COUNT = 16;
constexpr uint32_t MAX_IO_DIE_COUNT = 2;
constexpr uint32_t MAX_PHASE_VARIANT_COUNT = MAX_IO_DIE_COUNT * 2;
constexpr uint32_t INVALID_EVENT_BIT = 0xFFFFFFFFU;
constexpr uint32_t INVALID_PHASE_ID = 0xFFFFFFFFU;
constexpr uint32_t CONTEXT_MAGIC = 0x41474343U;
constexpr uint32_t CONTEXT_VERSION = 2;
constexpr uint64_t SMALL_MESSAGE_THRESHOLD = 1024ULL * 1024ULL;

enum TaskArgIndex : uint32_t {
    TASK_ARG_SEND_BASE = 0,
    TASK_ARG_INPUT_TOKEN = 1,
    TASK_ARG_RECV_BASE = 2,
    TASK_ARG_OUTPUT_TOKEN = 3,
    TASK_ARG_RANK_DATA_BYTES = 4,
    TASK_ARG_SLICE_OFFSET = 5,
    TASK_ARG_SLICE_BYTES = 6,
    TASK_ARG_LOCAL_GO_ADDR_OFFSET = 7,
    TASK_ARG_LOCAL_GO_LOOP_PARAM = 8,
    TASK_ARG_LOCAL_GO_PARALLEL_PARAM = 9,
    TASK_ARG_LOCAL_GO_RESIDUAL = 10,
};

enum class SourceKind : uint32_t {
    INPUT = 0,
    OUTPUT = 1,
};

enum class TopologyKind : uint32_t {
    GENERIC = 0,
    FOUR_BY_ONE = 1,
    TWO_BY_EIGHT = 2,
    EIGHT_PLUS_FOUR = 3,
};

enum class KernelVariantType : uint32_t {
    DIRECT = 0,
    FAST_LANE = 1,
    FAST_FORWARD = 2,
    RECEIVE_ONLY = 3,
};

enum class KernelPhaseId : uint32_t {
    DIRECT_L0 = 0,
    DIRECT_L1 = 1,
    FAST_LOCAL_GATHER = 2,
    FAST_INTER_SERVER = 3,
    FAST_LOCAL_FORWARD = 4,
};

enum KernelPhaseMask : uint32_t {
    PHASE_DIRECT_L0 = 1U << 0,
    PHASE_DIRECT_L1 = 1U << 1,
    PHASE_FAST_LOCAL_GATHER = 1U << 2,
    PHASE_FAST_INTER_SERVER = 1U << 3,
    PHASE_FAST_LOCAL_FORWARD = 1U << 4,
};

constexpr uint32_t ALL_PHASE_MASK = PHASE_DIRECT_L0 | PHASE_DIRECT_L1 | PHASE_FAST_LOCAL_GATHER
    | PHASE_FAST_INTER_SERVER | PHASE_FAST_LOCAL_FORWARD;

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_PEER_COUNT];
    uint32_t channelCount;
};

struct KernelVariantDesc {
    CcuKernelHandle kernelHandle;
    uint32_t resGroupId;
    uint32_t groupId;
    uint32_t variantId;
    uint32_t variantType;
    uint32_t phaseId;
    uint32_t phaseMask;
    uint32_t layer;
    uint32_t localIoDie;
    uint32_t threadSlot;
    uint32_t notifySlot;
    uint32_t sourceKind;
    uint32_t handleSelf;
    uint32_t channelCount;
    uint32_t segmentCount;
    uint32_t operationCount;
    uint32_t selfEventBit;
    uint16_t completionMask;
    uint16_t reserved;
    ChannelHandle channels[MAX_PEER_COUNT];
    uint32_t peerRanks[MAX_PEER_COUNT];
    uint32_t segmentRanks[MAX_SEGMENT_COUNT];
    uint8_t eventBits[MAX_OPERATION_COUNT];
};

struct AlgResourceCtx {
    uint32_t magic;
    uint32_t version;
    uint32_t totalSize;
    uint32_t variantCount;
    uint32_t topologyKind;
    uint32_t rankSize;
    uint32_t myRank;
    uint32_t fastPathEnabled;
    uint64_t topologySignature;
    uint32_t hasSlaveThread;
    uint32_t mainToSlaveNotifyIdx;
    uint32_t slaveToMainNotifyIdx;
    uint32_t reserved;
    ThreadHandle slaveThread;
    KernelVariantDesc variants[MAX_KERNEL_VARIANT_COUNT];

    /**
     * @brief 将资源上下文初始化为确定性的固定布局空状态
     * @param currentRankSize 当前通信域中的 rank 数量
     * @param currentRank 当前进程对应的全局 rank 编号
     * @return 成功返回 HCCL_SUCCESS，初始化失败返回对应错误码
     */
    HcclResult Initialize(uint32_t currentRankSize, uint32_t currentRank)
    {
        errno_t ret = memset_s(this, sizeof(*this), 0, sizeof(*this));
        if (ret != EOK) {
            HCCL_ERROR("[AlgResourceCtx] initialize context failed, ret[%d]", ret);
            return HCCL_E_INTERNAL;
        }
        magic = CONTEXT_MAGIC;
        version = CONTEXT_VERSION;
        totalSize = sizeof(*this);
        rankSize = currentRankSize;
        myRank = currentRank;
        return HCCL_SUCCESS;
    }

    /**
     * @brief 校验固定布局资源上下文中的版本、边界和映射关系
     * @param 无参数
     * @return 校验通过返回 HCCL_SUCCESS，校验失败返回对应错误码
     */
    HcclResult Validate() const
    {
        if (magic != CONTEXT_MAGIC || version != CONTEXT_VERSION || totalSize != sizeof(*this)) {
            HCCL_ERROR("[AlgResourceCtx] invalid header, magic[0x%x], version[%u], totalSize[%u]", magic, version,
                totalSize);
            return HCCL_E_INTERNAL;
        }
        if (rankSize == 0 || rankSize > MAX_RANK_SIZE || myRank >= rankSize) {
            HCCL_ERROR("[AlgResourceCtx] invalid rank metadata, myRank[%u], rankSize[%u]", myRank, rankSize);
            return HCCL_E_INTERNAL;
        }
        if (variantCount > MAX_KERNEL_VARIANT_COUNT || fastPathEnabled > 1 || hasSlaveThread > 1) {
            HCCL_ERROR("[AlgResourceCtx] invalid count or flag, variantCount[%u], fastPathEnabled[%u], hasSlave[%u]",
                variantCount, fastPathEnabled, hasSlaveThread);
            return HCCL_E_INTERNAL;
        }
        if (topologyKind > static_cast<uint32_t>(TopologyKind::EIGHT_PLUS_FOUR)) {
            HCCL_ERROR("[AlgResourceCtx] invalid topology kind[%u]", topologyKind);
            return HCCL_E_INTERNAL;
        }
        if (rankSize > 1 && variantCount == 0) {
            HCCL_ERROR("[AlgResourceCtx] missing reusable kernel variant");
            return HCCL_E_INTERNAL;
        }
        if ((hasSlaveThread == 0 && slaveThread != 0) || (hasSlaveThread != 0 && slaveThread == 0)) {
            HCCL_ERROR("[AlgResourceCtx] invalid slave thread state, hasSlave[%u], slaveThread[%llu]", hasSlaveThread,
                static_cast<unsigned long long>(slaveThread));
            return HCCL_E_INTERNAL;
        }

        bool variantIds[MAX_KERNEL_VARIANT_COUNT] = {false};
        bool peerChannelValid[MAX_RANK_SIZE] = {false};
        ChannelHandle peerChannels[MAX_RANK_SIZE] = {0};
        uint32_t directPeerCoverage[MAX_RANK_SIZE] = {0};
        const uint32_t phaseBits[] = {PHASE_DIRECT_L0, PHASE_DIRECT_L1, PHASE_FAST_LOCAL_GATHER,
            PHASE_FAST_INTER_SERVER, PHASE_FAST_LOCAL_FORWARD};
        uint32_t phaseVariantCount[sizeof(phaseBits) / sizeof(phaseBits[0])] = {0};
        uint32_t phaseSelfCount[sizeof(phaseBits) / sizeof(phaseBits[0])] = {0};

        for (uint32_t variantIndex = 0; variantIndex < variantCount; ++variantIndex) {
            const KernelVariantDesc &variant = variants[variantIndex];
            if (variant.variantId >= MAX_KERNEL_VARIANT_COUNT || variantIds[variant.variantId]
                || variant.kernelHandle == 0) {
                HCCL_ERROR("[AlgResourceCtx] invalid variant identity, index[%u], variantId[%u], kernel[%llu]",
                    variantIndex, variant.variantId, static_cast<unsigned long long>(variant.kernelHandle));
                return HCCL_E_INTERNAL;
            }
            variantIds[variant.variantId] = true;
            if (variant.phaseId > static_cast<uint32_t>(KernelPhaseId::FAST_LOCAL_FORWARD)
                || variant.phaseMask != (uint32_t {1} << variant.phaseId)
                || (variant.phaseMask & ~ALL_PHASE_MASK) != 0) {
                HCCL_ERROR("[AlgResourceCtx] invalid phase metadata, variantId[%u], phaseId[%u], phaseMask[0x%x]",
                    variant.variantId, variant.phaseId, variant.phaseMask);
                return HCCL_E_INTERNAL;
            }
            if (variant.layer > 1 || variant.localIoDie >= MAX_IO_DIE_COUNT || variant.threadSlot > 1
                || variant.notifySlot != 0 || (variant.threadSlot == 1 && hasSlaveThread == 0)) {
                HCCL_ERROR("[AlgResourceCtx] invalid placement, variantId[%u], layer[%u], die[%u], threadSlot[%u]",
                    variant.variantId, variant.layer, variant.localIoDie, variant.threadSlot);
                return HCCL_E_INTERNAL;
            }
            if (variant.variantType > static_cast<uint32_t>(KernelVariantType::RECEIVE_ONLY)
                || variant.sourceKind > static_cast<uint32_t>(SourceKind::OUTPUT) || variant.handleSelf > 1
                || variant.channelCount == 0 || variant.channelCount > MAX_PEER_COUNT
                || variant.segmentCount > MAX_SEGMENT_COUNT
                || variant.operationCount != variant.channelCount * variant.segmentCount
                || variant.operationCount > MAX_OPERATION_COUNT) {
                HCCL_ERROR("[AlgResourceCtx] invalid operation metadata, variantId[%u], channels[%u], segments[%u], "
                           "operations[%u]",
                    variant.variantId, variant.channelCount, variant.segmentCount, variant.operationCount);
                return HCCL_E_INTERNAL;
            }
            if ((variant.variantType == static_cast<uint32_t>(KernelVariantType::RECEIVE_ONLY))
                != (variant.segmentCount == 0)) {
                HCCL_ERROR("[AlgResourceCtx] receive-only metadata mismatch, variantId[%u], segmentCount[%u]",
                    variant.variantId, variant.segmentCount);
                return HCCL_E_INTERNAL;
            }
            if (variant.sourceKind == static_cast<uint32_t>(SourceKind::INPUT)
                && (variant.segmentCount != 1 || variant.segmentRanks[0] != myRank)) {
                HCCL_ERROR("[AlgResourceCtx] invalid input source mapping, variantId[%u], segmentCount[%u]",
                    variant.variantId, variant.segmentCount);
                return HCCL_E_INTERNAL;
            }
            if (variant.handleSelf != 0 && variant.sourceKind != static_cast<uint32_t>(SourceKind::INPUT)) {
                HCCL_ERROR("[AlgResourceCtx] self copy must use input source, variantId[%u]", variant.variantId);
                return HCCL_E_INTERNAL;
            }

            bool peerSeen[MAX_RANK_SIZE] = {false};
            for (uint32_t peerIndex = 0; peerIndex < variant.channelCount; ++peerIndex) {
                uint32_t peerRank = variant.peerRanks[peerIndex];
                ChannelHandle channel = variant.channels[peerIndex];
                if (peerRank >= rankSize || peerRank == myRank || peerSeen[peerRank] || channel == 0) {
                    HCCL_ERROR("[AlgResourceCtx] invalid peer mapping, variantId[%u], peerIndex[%u], peerRank[%u]",
                        variant.variantId, peerIndex, peerRank);
                    return HCCL_E_INTERNAL;
                }
                peerSeen[peerRank] = true;
                if (peerChannelValid[peerRank] && peerChannels[peerRank] != channel) {
                    HCCL_ERROR("[AlgResourceCtx] multiple channels found for peerRank[%u]", peerRank);
                    return HCCL_E_INTERNAL;
                }
                peerChannelValid[peerRank] = true;
                peerChannels[peerRank] = channel;
                if ((variant.phaseMask & (PHASE_DIRECT_L0 | PHASE_DIRECT_L1)) != 0) {
                    ++directPeerCoverage[peerRank];
                }
            }
            for (uint32_t segmentIndex = 0; segmentIndex < variant.segmentCount; ++segmentIndex) {
                if (variant.segmentRanks[segmentIndex] >= rankSize) {
                    HCCL_ERROR("[AlgResourceCtx] invalid segment rank, variantId[%u], segmentRank[%u]",
                        variant.variantId, variant.segmentRanks[segmentIndex]);
                    return HCCL_E_INTERNAL;
                }
            }

            bool eventBitsUsed[MAX_OPERATION_COUNT] = {false};
            uint16_t expectedCompletionMask = 0;
            for (uint32_t operationIndex = 0; operationIndex < variant.operationCount; ++operationIndex) {
                uint32_t eventBit = variant.eventBits[operationIndex];
                if (eventBit >= MAX_OPERATION_COUNT || eventBitsUsed[eventBit]) {
                    HCCL_ERROR("[AlgResourceCtx] invalid event bit, variantId[%u], operationIndex[%u], eventBit[%u]",
                        variant.variantId, operationIndex, eventBit);
                    return HCCL_E_INTERNAL;
                }
                eventBitsUsed[eventBit] = true;
                expectedCompletionMask |= static_cast<uint16_t>(1U << eventBit);
            }
            if (variant.handleSelf != 0) {
                if (variant.selfEventBit >= MAX_OPERATION_COUNT || eventBitsUsed[variant.selfEventBit]) {
                    HCCL_ERROR("[AlgResourceCtx] invalid self event bit, variantId[%u], selfEventBit[%u]",
                        variant.variantId, variant.selfEventBit);
                    return HCCL_E_INTERNAL;
                }
                expectedCompletionMask |= static_cast<uint16_t>(1U << variant.selfEventBit);
            } else if (variant.selfEventBit != INVALID_EVENT_BIT) {
                HCCL_ERROR("[AlgResourceCtx] unexpected self event bit, variantId[%u], selfEventBit[%u]",
                    variant.variantId, variant.selfEventBit);
                return HCCL_E_INTERNAL;
            }
            if (variant.completionMask != expectedCompletionMask) {
                HCCL_ERROR("[AlgResourceCtx] invalid completion mask, variantId[%u], actual[0x%x], expected[0x%x]",
                    variant.variantId, variant.completionMask, expectedCompletionMask);
                return HCCL_E_INTERNAL;
            }

            for (uint32_t phaseIndex = 0; phaseIndex < sizeof(phaseBits) / sizeof(phaseBits[0]); ++phaseIndex) {
                if ((variant.phaseMask & phaseBits[phaseIndex]) != 0) {
                    ++phaseVariantCount[phaseIndex];
                    phaseSelfCount[phaseIndex] += variant.handleSelf;
                }
            }
        }

        for (uint32_t peerRank = 0; peerRank < rankSize; ++peerRank) {
            if (peerRank == myRank) {
                continue;
            }
            if (!peerChannelValid[peerRank] || directPeerCoverage[peerRank] != 1) {
                HCCL_ERROR("[AlgResourceCtx] incomplete direct mapping, peerRank[%u], channelValid[%u], coverage[%u]",
                    peerRank, peerChannelValid[peerRank], directPeerCoverage[peerRank]);
                return HCCL_E_INTERNAL;
            }
        }
        for (uint32_t phaseIndex = 0; phaseIndex < sizeof(phaseBits) / sizeof(phaseBits[0]); ++phaseIndex) {
            if (phaseVariantCount[phaseIndex] > MAX_PHASE_VARIANT_COUNT || phaseSelfCount[phaseIndex] > 1) {
                HCCL_ERROR("[AlgResourceCtx] invalid phase concurrency, phaseIndex[%u], variants[%u], selfCount[%u]",
                    phaseIndex, phaseVariantCount[phaseIndex], phaseSelfCount[phaseIndex]);
                return HCCL_E_INTERNAL;
            }
        }
        return HCCL_SUCCESS;
    }

    /**
     * @brief 将固定布局资源上下文序列化为字节数组
     * @param data 接收序列化结果的字节数组
     * @return 成功返回 HCCL_SUCCESS，序列化失败返回对应错误码
     */
    HcclResult Serialize(std::vector<char> &data) const
    {
        HcclResult validateRet = Validate();
        if (validateRet != HCCL_SUCCESS) {
            return validateRet;
        }
        data.resize(sizeof(*this));
        errno_t ret = memcpy_s(data.data(), data.size(), this, sizeof(*this));
        if (ret != EOK) {
            HCCL_ERROR("[AlgResourceCtx] serialize context failed, ret[%d]", ret);
            return HCCL_E_INTERNAL;
        }
        return HCCL_SUCCESS;
    }

    /**
     * @brief 从固定长度字节区域恢复并校验资源上下文
     * @param data 待反序列化的字节区域起始地址
     * @param dataSize 待反序列化的字节区域长度
     * @return 成功返回 HCCL_SUCCESS，长度或内容非法时返回对应错误码
     */
    HcclResult DeSerialize(const void *data, uint64_t dataSize)
    {
        if (data == nullptr || dataSize != sizeof(*this)) {
            HCCL_ERROR("[AlgResourceCtx] invalid serialized context, data[%p], dataSize[%llu], expected[%zu]", data,
                static_cast<unsigned long long>(dataSize), sizeof(*this));
            return data == nullptr ? HCCL_E_PTR : HCCL_E_INTERNAL;
        }
        AlgResourceCtx restored;
        errno_t ret = memcpy_s(&restored, sizeof(restored), data, dataSize);
        if (ret != EOK) {
            HCCL_ERROR("[AlgResourceCtx] deserialize context failed, ret[%d]", ret);
            return HCCL_E_INTERNAL;
        }
        HcclResult validateRet = restored.Validate();
        if (validateRet != HCCL_SUCCESS) {
            return validateRet;
        }
        *this = restored;
        return HCCL_SUCCESS;
    }
};

static_assert(std::is_trivially_copyable<KernelVariantDesc>::value, "KernelVariantDesc must be flat POD");
static_assert(std::is_trivially_copyable<AlgResourceCtx>::value, "AlgResourceCtx must be flat POD");
static_assert(TASK_ARG_LOCAL_GO_RESIDUAL + 1 == TASK_ARG_COUNT, "task argument count mismatch");

#endif
