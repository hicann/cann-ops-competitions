/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>

#include <hcomm/hcomm_primitives.h>

#include "ccu_kernel.h"

#define BROADCAST_CCU_CHECK(call) \
    do { \
        CcuResult result = (call); \
        if (result != CCU_SUCCESS) { \
            return result; \
        } \
    } while (0)

namespace ops_hccl {
CcuResult CcuKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgBase *>(arg);
    constexpr uint32_t outputResourceId = 1;
    constexpr uint32_t tokenResourceId = 2;
    constexpr uint32_t outputNotifyId = 0;
    constexpr uint32_t tokenNotifyId = 1;
    constexpr uint32_t postSyncId = 2;
    constexpr uint32_t ckeIndex = 0;

    ccu::Variable localBase;
    ccu::Variable localToken;
    ccu::Variable chunkOffset;
    ccu::Variable chunkSize;
    ccu::Variable sendEnabled;
    std::vector<ccu::Variable> remoteBases(kernelArg->channelCount);
    std::vector<ccu::Variable> remoteTokens(kernelArg->channelCount);
    ccu::Event transferDone;

    uint32_t argId = 0;
    BROADCAST_CCU_CHECK(ccu::LoadArg(localBase, argId++));
    BROADCAST_CCU_CHECK(ccu::LoadArg(localToken, argId++));
    BROADCAST_CCU_CHECK(ccu::LoadArg(chunkOffset, argId++));
    BROADCAST_CCU_CHECK(ccu::LoadArg(chunkSize, argId++));
    BROADCAST_CCU_CHECK(ccu::LoadArg(sendEnabled, argId++));

    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        ChannelHandle channel = kernelArg->channels[i];
        remoteBases[i] = ccu::GetResByChannel<ccu::Variable>(channel, outputResourceId);
        remoteTokens[i] = ccu::GetResByChannel<ccu::Variable>(channel, tokenResourceId);
        BROADCAST_CCU_CHECK(
            ccu::WriteVariableWithNotify(channel, localBase, outputResourceId, ckeIndex, 1U << outputNotifyId));
        BROADCAST_CCU_CHECK(
            ccu::WriteVariableWithNotify(channel, localToken, tokenResourceId, ckeIndex, 1U << tokenNotifyId));
    }
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        BROADCAST_CCU_CHECK(
            ccu::NotifyWait(kernelArg->channels[i], ckeIndex, (1U << outputNotifyId) | (1U << tokenNotifyId)));
    }

    const uint16_t transferMask = (1U << kernelArg->channelCount) - 1;
    CCU_IF(sendEnabled != 0)
    {
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            ccu::LocalAddr source;
            source.addr = localBase;
            source.addr += chunkOffset;
            source.token = localToken;
            ccu::RemoteAddr destination;
            destination.addr = remoteBases[i];
            destination.addr += chunkOffset;
            destination.token = remoteTokens[i];
            BROADCAST_CCU_CHECK(
                ccu::Write(kernelArg->channels[i], destination, source, chunkSize, transferDone, 1U << i));
        }
        BROADCAST_CCU_CHECK(ccu::EventWait(transferDone, transferMask));
    }

    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        BROADCAST_CCU_CHECK(ccu::NotifyRecord(kernelArg->channels[i], ckeIndex, 1U << postSyncId));
    }
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        BROADCAST_CCU_CHECK(ccu::NotifyWait(kernelArg->channels[i], ckeIndex, 1U << postSyncId));
    }

    return CCU_SUCCESS;
}

CcuResult CcuSmallFlatKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgBase *>(arg);
    constexpr uint32_t outputResourceId = 1;
    constexpr uint32_t tokenResourceId = 2;
    constexpr uint32_t outputNotifyId = 0;
    constexpr uint32_t tokenNotifyId = 1;
    constexpr uint32_t postSyncId = 2;
    constexpr uint32_t ckeIndex = 0;

    ccu::Variable localBase;
    ccu::Variable localToken;
    uint32_t argId = 0;
    BROADCAST_CCU_CHECK(ccu::LoadArg(localBase, argId++));
    BROADCAST_CCU_CHECK(ccu::LoadArg(localToken, argId++));

    if (kernelArg->isSender != 0) {
        ccu::Variable dataSize;
        std::vector<ccu::Variable> remoteBases(kernelArg->channelCount);
        std::vector<ccu::Variable> remoteTokens(kernelArg->channelCount);
        ccu::Event transferDone;
        BROADCAST_CCU_CHECK(ccu::LoadArg(dataSize, argId++));

        // 对端资源就绪后立即提交 Write，避免等待全部 channel 就绪形成阶段屏障。
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            const ChannelHandle channel = kernelArg->channels[i];
            remoteBases[i] = ccu::GetResByChannel<ccu::Variable>(channel, outputResourceId);
            remoteTokens[i] = ccu::GetResByChannel<ccu::Variable>(channel, tokenResourceId);
            BROADCAST_CCU_CHECK(ccu::NotifyWait(channel, ckeIndex, (1U << outputNotifyId) | (1U << tokenNotifyId)));
            ccu::LocalAddr source;
            source.addr = localBase;
            source.token = localToken;
            ccu::RemoteAddr destination;
            destination.addr = remoteBases[i];
            destination.token = remoteTokens[i];
            BROADCAST_CCU_CHECK(ccu::Write(channel, destination, source, dataSize, transferDone, 1U << i));
        }
        const uint16_t transferMask = (1U << kernelArg->channelCount) - 1;
        BROADCAST_CCU_CHECK(ccu::EventWait(transferDone, transferMask));
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            BROADCAST_CCU_CHECK(ccu::NotifyRecord(kernelArg->channels[i], ckeIndex, 1U << postSyncId));
        }
    } else {
        const ChannelHandle channel = kernelArg->channels[0];
        BROADCAST_CCU_CHECK(
            ccu::WriteVariableWithNotify(channel, localBase, outputResourceId, ckeIndex, 1U << outputNotifyId));
        BROADCAST_CCU_CHECK(
            ccu::WriteVariableWithNotify(channel, localToken, tokenResourceId, ckeIndex, 1U << tokenNotifyId));
        BROADCAST_CCU_CHECK(ccu::NotifyWait(channel, ckeIndex, 1U << postSyncId));
    }

    return CCU_SUCCESS;
}

CcuResult CcuFourRankPipelineKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuFourRankPipelineKernelArg *>(arg);
    constexpr uint32_t outputResourceId = 1;
    constexpr uint32_t tokenResourceId = 2;
    constexpr uint32_t outputNotifyId = 0;
    constexpr uint32_t tokenNotifyId = 1;
    constexpr uint32_t doneNotifyId = 2;
    constexpr uint32_t ckeIndex = 0;

    ccu::Variable localBase;
    ccu::Variable localToken;
    ccu::Variable regularSliceSize;
    ccu::Variable tailSliceSize;
    BROADCAST_CCU_CHECK(ccu::LoadArg(localBase, 0));
    BROADCAST_CCU_CHECK(ccu::LoadArg(localToken, 1));
    BROADCAST_CCU_CHECK(ccu::LoadArg(regularSliceSize, 2));
    BROADCAST_CCU_CHECK(ccu::LoadArg(tailSliceSize, 3));

    std::vector<ccu::Variable> remoteBases(2);
    std::vector<ccu::Variable> remoteTokens(2);
    for (uint32_t index = 0; index < kernelArg->channelCount; ++index) {
        const ChannelHandle channel = kernelArg->channels[index];
        remoteBases[index] = ccu::GetResByChannel<ccu::Variable>(channel, outputResourceId);
        remoteTokens[index] = ccu::GetResByChannel<ccu::Variable>(channel, tokenResourceId);
        BROADCAST_CCU_CHECK(
            ccu::WriteVariableWithNotify(channel, localBase, outputResourceId, ckeIndex, 1U << outputNotifyId));
        BROADCAST_CCU_CHECK(
            ccu::WriteVariableWithNotify(channel, localToken, tokenResourceId, ckeIndex, 1U << tokenNotifyId));
    }
    for (uint32_t index = 0; index < kernelArg->channelCount; ++index) {
        BROADCAST_CCU_CHECK(
            ccu::NotifyWait(kernelArg->channels[index], ckeIndex, (1U << outputNotifyId) | (1U << tokenNotifyId)));
    }
    // 双向屏障确保地址通知已被对端消费，之后才安全复用 bit 0/1 作为双 lane 信号。
    for (uint32_t index = 0; index < kernelArg->channelCount; ++index) {
        BROADCAST_CCU_CHECK(ccu::NotifyRecord(kernelArg->channels[index], ckeIndex, 1U << doneNotifyId));
    }
    for (uint32_t index = 0; index < kernelArg->channelCount; ++index) {
        BROADCAST_CCU_CHECK(ccu::NotifyWait(kernelArg->channels[index], ckeIndex, 1U << doneNotifyId));
    }

    const bool hasUpstream = kernelArg->position != 0U;
    const bool hasDownstream = kernelArg->position != 3U;
    const uint32_t upstreamIndex = 0;
    const uint32_t downstreamIndex = kernelArg->position == 0U ? 0U : 1U;
    const ChannelHandle upstream = kernelArg->channels[upstreamIndex];
    const ChannelHandle downstream = kernelArg->channels[hasDownstream ? downstreamIndex : 0U];

    ccu::LocalAddr source;
    source.addr = localBase;
    source.token = localToken;
    ccu::RemoteAddr destination;
    if (hasDownstream) {
        destination.addr = remoteBases[downstreamIndex];
        destination.token = remoteTokens[downstreamIndex];
    }

    for (uint32_t slice = 0; slice < FOUR_RANK_PIPELINE_SLICE_NUM; ++slice) {
        const uint32_t laneNotifyId = slice & 1U;
        // 两条 lane 在复用通知位前等待下游消费，保留原来的双缓冲流水语义。
        if (hasDownstream && slice >= 2U) {
            BROADCAST_CCU_CHECK(ccu::NotifyWait(downstream, ckeIndex, 1U << laneNotifyId));
        }
        if (hasUpstream) {
            BROADCAST_CCU_CHECK(ccu::NotifyWait(upstream, ckeIndex, 1U << laneNotifyId));
            BROADCAST_CCU_CHECK(ccu::NotifyRecord(upstream, ckeIndex, 1U << laneNotifyId));
        }
        if (hasDownstream) {
            ccu::Event transferDone;
            if (slice + 1U == FOUR_RANK_PIPELINE_SLICE_NUM) {
                BROADCAST_CCU_CHECK(ccu::Write(downstream, destination, source, tailSliceSize, transferDone, 1));
            } else {
                BROADCAST_CCU_CHECK(ccu::Write(downstream, destination, source, regularSliceSize, transferDone, 1));
            }
            BROADCAST_CCU_CHECK(ccu::EventWait(transferDone, 1));
            BROADCAST_CCU_CHECK(ccu::NotifyRecord(downstream, ckeIndex, 1U << laneNotifyId));
        }
        if (slice + 1U != FOUR_RANK_PIPELINE_SLICE_NUM) {
            source.addr += regularSliceSize;
            if (hasDownstream) {
                destination.addr += regularSliceSize;
            }
        }
    }

    if (hasDownstream) {
        BROADCAST_CCU_CHECK(ccu::NotifyWait(downstream, ckeIndex, 1U << 0));
        BROADCAST_CCU_CHECK(ccu::NotifyWait(downstream, ckeIndex, 1U << 1));
        BROADCAST_CCU_CHECK(ccu::NotifyWait(downstream, ckeIndex, 1U << doneNotifyId));
    }
    if (hasUpstream) {
        BROADCAST_CCU_CHECK(ccu::NotifyRecord(upstream, ckeIndex, 1U << doneNotifyId));
    }
    return CCU_SUCCESS;
}

namespace {
    constexpr uint32_t TWELVE_OUTPUT_RESOURCE_ID = 1;
    constexpr uint32_t TWELVE_TOKEN_RESOURCE_ID = 2;
    constexpr uint32_t TWELVE_OUTPUT_NOTIFY_ID = 0;
    constexpr uint32_t TWELVE_TOKEN_NOTIFY_ID = 1;
    constexpr uint32_t TWELVE_DONE_NOTIFY_ID = 2;
    constexpr uint32_t TWELVE_CKE_INDEX = 0;
    constexpr uint32_t TWELVE_SLICE_NUM = 24;
    constexpr uint32_t TWELVE_REMOTE_RANK_BASE = 8;

    enum class TwelveStage : uint32_t {
        LOCAL_SCATTER_A,
        REMOTE_SCATTER_D,
        LOCAL_SCATTER_B_GATHER_A,
        REMOTE_ALLGATHER_D,
        CROSS_WAVE_AD_C,
        LOCAL_ALLGATHER_B,
        REMOTE_ALLGATHER_C,
        CROSS_WAVE_BC,
    };

    CcuResult LoadTwelveStageArgs(ccu::Variable &localBase, ccu::Variable &localToken,
        std::vector<ccu::Variable> &offsets, std::vector<ccu::Variable> &sizes)
    {
        BROADCAST_CCU_CHECK(ccu::LoadArg(localBase, 0));
        BROADCAST_CCU_CHECK(ccu::LoadArg(localToken, 1));
        for (uint32_t i = 0; i < TWELVE_SLICE_NUM; ++i) {
            BROADCAST_CCU_CHECK(ccu::LoadArg(offsets[i], 2 + i));
            BROADCAST_CCU_CHECK(ccu::LoadArg(sizes[i], 2 + TWELVE_SLICE_NUM + i));
        }
        return CCU_SUCCESS;
    }

    CcuResult TwelveScatter(CcuTwelveKernelArg *kernelArg, TwelveStage stage,
        ccu::Variable &localBase, ccu::Variable &localToken, std::vector<ccu::Variable> &offsets,
        std::vector<ccu::Variable> &sizes)
    {
        if (kernelArg->myRank != kernelArg->root) {
            uint32_t rootIndex = kernelArg->channelCount;
            for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                if (kernelArg->peerRanks[i] == kernelArg->root) {
                    rootIndex = i;
                }
            }
            if (rootIndex == kernelArg->channelCount) {
                return CCU_E_PARA;
            }
            const ChannelHandle channel = kernelArg->channels[rootIndex];
            BROADCAST_CCU_CHECK(ccu::WriteVariableWithNotify(
                channel, localBase, TWELVE_OUTPUT_RESOURCE_ID, TWELVE_CKE_INDEX, 1U << TWELVE_OUTPUT_NOTIFY_ID));
            BROADCAST_CCU_CHECK(ccu::WriteVariableWithNotify(
                channel, localToken, TWELVE_TOKEN_RESOURCE_ID, TWELVE_CKE_INDEX, 1U << TWELVE_TOKEN_NOTIFY_ID));
            BROADCAST_CCU_CHECK(
                ccu::NotifyWait(channel, TWELVE_CKE_INDEX, 1U << TWELVE_DONE_NOTIFY_ID));
            return CCU_SUCCESS;
        }

        std::vector<ccu::Variable> remoteBases(kernelArg->channelCount);
        std::vector<ccu::Variable> remoteTokens(kernelArg->channelCount);
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            const ChannelHandle channel = kernelArg->channels[i];
            remoteBases[i] = ccu::GetResByChannel<ccu::Variable>(channel, TWELVE_OUTPUT_RESOURCE_ID);
            remoteTokens[i] = ccu::GetResByChannel<ccu::Variable>(channel, TWELVE_TOKEN_RESOURCE_ID);
            BROADCAST_CCU_CHECK(ccu::NotifyWait(channel, TWELVE_CKE_INDEX,
                (1U << TWELVE_OUTPUT_NOTIFY_ID) | (1U << TWELVE_TOKEN_NOTIFY_ID)));
        }

        ccu::Event transferDone;
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            const uint32_t peer = kernelArg->peerRanks[i];
            uint32_t slice = peer;
            if (stage == TwelveStage::LOCAL_SCATTER_B_GATHER_A) {
                slice = 8U + peer;
            } else if (stage == TwelveStage::REMOTE_SCATTER_D) {
                slice = 20U + peer - TWELVE_REMOTE_RANK_BASE;
            }
            ccu::LocalAddr source;
            source.addr = localBase;
            source.addr += offsets[slice];
            source.token = localToken;
            ccu::RemoteAddr destination;
            destination.addr = remoteBases[i];
            destination.addr += offsets[slice];
            destination.token = remoteTokens[i];
            BROADCAST_CCU_CHECK(
                ccu::Write(kernelArg->channels[i], destination, source, sizes[slice], transferDone, 1U << i));
        }
        BROADCAST_CCU_CHECK(
            ccu::EventWait(transferDone, static_cast<uint16_t>((1U << kernelArg->channelCount) - 1U)));
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            BROADCAST_CCU_CHECK(
                ccu::NotifyRecord(kernelArg->channels[i], TWELVE_CKE_INDEX, 1U << TWELVE_DONE_NOTIFY_ID));
        }
        return CCU_SUCCESS;
    }

    CcuResult TwelveAllGather(CcuTwelveKernelArg *kernelArg, TwelveStage stage,
        ccu::Variable &localBase, ccu::Variable &localToken, std::vector<ccu::Variable> &offsets,
        std::vector<ccu::Variable> &sizes)
    {
        std::vector<ccu::Variable> remoteBases(kernelArg->channelCount);
        std::vector<ccu::Variable> remoteTokens(kernelArg->channelCount);
        const bool rootFastPath = kernelArg->myRank < TWELVE_REMOTE_RANK_BASE;
        const bool receiveEnabled = !(rootFastPath && kernelArg->myRank == kernelArg->root);
        if (receiveEnabled) {
            for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                const ChannelHandle channel = kernelArg->channels[i];
                BROADCAST_CCU_CHECK(ccu::WriteVariableWithNotify(
                    channel, localBase, TWELVE_OUTPUT_RESOURCE_ID, TWELVE_CKE_INDEX,
                    1U << TWELVE_OUTPUT_NOTIFY_ID));
                BROADCAST_CCU_CHECK(ccu::WriteVariableWithNotify(
                    channel, localToken, TWELVE_TOKEN_RESOURCE_ID, TWELVE_CKE_INDEX,
                    1U << TWELVE_TOKEN_NOTIFY_ID));
            }
        }

        uint32_t slice = kernelArg->myRank;
        if (stage == TwelveStage::LOCAL_ALLGATHER_B) {
            slice = 8U + kernelArg->myRank;
        } else if (stage == TwelveStage::REMOTE_ALLGATHER_C) {
            slice = 16U + kernelArg->myRank - TWELVE_REMOTE_RANK_BASE;
        } else if (stage == TwelveStage::REMOTE_ALLGATHER_D) {
            slice = 20U + kernelArg->myRank - TWELVE_REMOTE_RANK_BASE;
        }

        ccu::Event transferDone;
        uint32_t eventId = 0;
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            const bool sendEnabled = !(rootFastPath && kernelArg->peerRanks[i] == kernelArg->root);
            if (!sendEnabled) {
                continue;
            }
            const ChannelHandle channel = kernelArg->channels[i];
            remoteBases[i] = ccu::GetResByChannel<ccu::Variable>(channel, TWELVE_OUTPUT_RESOURCE_ID);
            remoteTokens[i] = ccu::GetResByChannel<ccu::Variable>(channel, TWELVE_TOKEN_RESOURCE_ID);
            BROADCAST_CCU_CHECK(ccu::NotifyWait(channel, TWELVE_CKE_INDEX,
                (1U << TWELVE_OUTPUT_NOTIFY_ID) | (1U << TWELVE_TOKEN_NOTIFY_ID)));
            ccu::LocalAddr source;
            source.addr = localBase;
            source.addr += offsets[slice];
            source.token = localToken;
            ccu::RemoteAddr destination;
            destination.addr = remoteBases[i];
            destination.addr += offsets[slice];
            destination.token = remoteTokens[i];
            BROADCAST_CCU_CHECK(ccu::Write(
                channel, destination, source, sizes[slice], transferDone, 1U << eventId++));
        }
        if (eventId != 0U) {
            BROADCAST_CCU_CHECK(
                ccu::EventWait(transferDone, static_cast<uint16_t>((1U << eventId) - 1U)));
        }
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            if (!(rootFastPath && kernelArg->peerRanks[i] == kernelArg->root)) {
                BROADCAST_CCU_CHECK(
                    ccu::NotifyRecord(kernelArg->channels[i], TWELVE_CKE_INDEX, 1U << TWELVE_DONE_NOTIFY_ID));
            }
        }
        if (receiveEnabled) {
            for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                BROADCAST_CCU_CHECK(
                    ccu::NotifyWait(kernelArg->channels[i], TWELVE_CKE_INDEX, 1U << TWELVE_DONE_NOTIFY_ID));
            }
        }
        return CCU_SUCCESS;
    }

    CcuResult TwelveCrossWave(CcuTwelveKernelArg *kernelArg, TwelveStage stage,
        ccu::Variable &localBase, ccu::Variable &localToken, std::vector<ccu::Variable> &offsets,
        std::vector<ccu::Variable> &sizes)
    {
        std::vector<ccu::Variable> remoteBases(kernelArg->channelCount);
        std::vector<ccu::Variable> remoteTokens(kernelArg->channelCount);
        const bool isLocal = kernelArg->myRank < TWELVE_REMOTE_RANK_BASE;
        const bool receiveEnabled = kernelArg->myRank != kernelArg->root;
        if (receiveEnabled) {
            for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                const ChannelHandle channel = kernelArg->channels[i];
                BROADCAST_CCU_CHECK(ccu::WriteVariableWithNotify(
                    channel, localBase, TWELVE_OUTPUT_RESOURCE_ID, TWELVE_CKE_INDEX,
                    1U << TWELVE_OUTPUT_NOTIFY_ID));
                BROADCAST_CCU_CHECK(ccu::WriteVariableWithNotify(
                    channel, localToken, TWELVE_TOKEN_RESOURCE_ID, TWELVE_CKE_INDEX,
                    1U << TWELVE_TOKEN_NOTIFY_ID));
            }
        }
        uint32_t sourceSlice = 0;
        if (isLocal) {
            sourceSlice = (stage == TwelveStage::CROSS_WAVE_AD_C ? 0U : 8U) + kernelArg->myRank;
        } else {
            sourceSlice = (stage == TwelveStage::CROSS_WAVE_AD_C ? 20U : 16U)
                          + kernelArg->myRank - TWELVE_REMOTE_RANK_BASE;
        }

        ccu::Event transferDone;
        uint32_t eventId = 0;
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            const bool sendEnabled = !(kernelArg->myRank >= TWELVE_REMOTE_RANK_BASE
                                       && kernelArg->peerRanks[i] == kernelArg->root);
            if (!sendEnabled) {
                continue;
            }
            const ChannelHandle channel = kernelArg->channels[i];
            remoteBases[i] = ccu::GetResByChannel<ccu::Variable>(channel, TWELVE_OUTPUT_RESOURCE_ID);
            remoteTokens[i] = ccu::GetResByChannel<ccu::Variable>(channel, TWELVE_TOKEN_RESOURCE_ID);
            BROADCAST_CCU_CHECK(ccu::NotifyWait(channel, TWELVE_CKE_INDEX,
                (1U << TWELVE_OUTPUT_NOTIFY_ID) | (1U << TWELVE_TOKEN_NOTIFY_ID)));
            ccu::LocalAddr source;
            source.addr = localBase;
            source.addr += offsets[sourceSlice];
            source.token = localToken;
            ccu::RemoteAddr destination;
            destination.addr = remoteBases[i];
            destination.addr += offsets[sourceSlice];
            destination.token = remoteTokens[i];
            BROADCAST_CCU_CHECK(ccu::Write(
                channel, destination, source, sizes[sourceSlice], transferDone, 1U << eventId++));
        }
        if (eventId != 0U) {
            BROADCAST_CCU_CHECK(
                ccu::EventWait(transferDone, static_cast<uint16_t>((1U << eventId) - 1U)));
        }

        // wave 1 的 local->remote 方向在 A 后继续承载 C Scatter，不增加独立 Kernel。
        if (stage == TwelveStage::CROSS_WAVE_AD_C && kernelArg->myRank == kernelArg->root) {
            for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                const uint32_t slice = 16U + kernelArg->peerRanks[i] - TWELVE_REMOTE_RANK_BASE;
                ccu::LocalAddr source;
                source.addr = localBase;
                source.addr += offsets[slice];
                source.token = localToken;
                ccu::RemoteAddr destination;
                destination.addr = remoteBases[i];
                destination.addr += offsets[slice];
                destination.token = remoteTokens[i];
                BROADCAST_CCU_CHECK(
                    ccu::Write(kernelArg->channels[i], destination, source, sizes[slice], transferDone, 1U << i));
            }
            BROADCAST_CCU_CHECK(
                ccu::EventWait(transferDone, static_cast<uint16_t>((1U << kernelArg->channelCount) - 1U)));
        }

        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            const bool sendEnabled = !(kernelArg->myRank >= TWELVE_REMOTE_RANK_BASE
                                       && kernelArg->peerRanks[i] == kernelArg->root);
            if (sendEnabled) {
                BROADCAST_CCU_CHECK(
                    ccu::NotifyRecord(kernelArg->channels[i], TWELVE_CKE_INDEX, 1U << TWELVE_DONE_NOTIFY_ID));
            }
        }
        if (receiveEnabled) {
            for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                BROADCAST_CCU_CHECK(
                    ccu::NotifyWait(kernelArg->channels[i], TWELVE_CKE_INDEX, 1U << TWELVE_DONE_NOTIFY_ID));
            }
        }
        return CCU_SUCCESS;
    }

    CcuResult RunTwelveStage(CcuTwelveKernelArg *kernelArg, TwelveStage stage, ccu::Variable &localBase,
        ccu::Variable &localToken, std::vector<ccu::Variable> &offsets, std::vector<ccu::Variable> &sizes)
    {
        if (stage == TwelveStage::LOCAL_SCATTER_B_GATHER_A) {
            BROADCAST_CCU_CHECK(TwelveScatter(kernelArg, stage, localBase, localToken, offsets, sizes));
            return TwelveAllGather(
                kernelArg, TwelveStage::LOCAL_SCATTER_A, localBase, localToken, offsets, sizes);
        }
        if (stage == TwelveStage::LOCAL_SCATTER_A || stage == TwelveStage::REMOTE_SCATTER_D) {
            return TwelveScatter(kernelArg, stage, localBase, localToken, offsets, sizes);
        }
        if (stage == TwelveStage::CROSS_WAVE_AD_C || stage == TwelveStage::CROSS_WAVE_BC) {
            return TwelveCrossWave(kernelArg, stage, localBase, localToken, offsets, sizes);
        }
        return TwelveAllGather(kernelArg, stage, localBase, localToken, offsets, sizes);
    }
} // namespace

CcuResult CcuTwelveLayerKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuTwelveKernelArg *>(arg);
    ccu::Variable localBase;
    ccu::Variable localToken;
    ccu::Variable mode;
    std::vector<ccu::Variable> offsets(TWELVE_SLICE_NUM);
    std::vector<ccu::Variable> sizes(TWELVE_SLICE_NUM);
    BROADCAST_CCU_CHECK(LoadTwelveStageArgs(localBase, localToken, offsets, sizes));
    BROADCAST_CCU_CHECK(ccu::LoadArg(mode, 2 + 2 * TWELVE_SLICE_NUM));

    for (uint32_t stageIndex = 0; stageIndex < TWELVE_STAGE_NUM; ++stageIndex) {
        if (kernelArg->stageChannelCount[stageIndex] == 0) {
            continue;
        }
        CcuTwelveKernelArg stageArg{};
        stageArg.myRank = kernelArg->myRank;
        stageArg.root = kernelArg->root;
        stageArg.channelCount = kernelArg->stageChannelCount[stageIndex];
        const TwelveStage stage = static_cast<TwelveStage>(stageIndex);
        for (uint32_t i = 0; i < stageArg.channelCount; ++i) {
            const uint32_t channelIndex = kernelArg->stageChannelIndices[stageIndex][i];
            stageArg.channels[i] = kernelArg->channels[channelIndex];
            stageArg.peerRanks[i] = kernelArg->peerRanks[channelIndex];
        }
        CCU_IF(mode == static_cast<uint64_t>(stageIndex))
        {
            BROADCAST_CCU_CHECK(RunTwelveStage(&stageArg, stage, localBase, localToken, offsets, sizes));
        }
    }
    return CCU_SUCCESS;
}
CcuResult CcuSixteenScatterKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuSixteenScatterKernelArg *>(arg);
    constexpr uint32_t outputResourceId = 1;
    constexpr uint32_t tokenResourceId = 2;
    constexpr uint32_t outputNotifyId = 0;
    constexpr uint32_t tokenNotifyId = 1;
    constexpr uint32_t postSyncId = 2;
    constexpr uint32_t ckeIndex = 0;
    constexpr uint32_t sliceNum = 16;

    ccu::Variable localBase;
    ccu::Variable localToken;
    uint32_t argId = 0;
    BROADCAST_CCU_CHECK(ccu::LoadArg(localBase, argId++));
    BROADCAST_CCU_CHECK(ccu::LoadArg(localToken, argId++));

    if (kernelArg->isSender != 0) {
        std::vector<ccu::Variable> sliceOffsets(sliceNum);
        std::vector<ccu::Variable> sliceSizes(sliceNum);
        std::vector<ccu::Variable> remoteBases(kernelArg->channelCount);
        std::vector<ccu::Variable> remoteTokens(kernelArg->channelCount);
        ccu::Event transferDone;
        for (uint32_t i = 0; i < sliceNum; ++i) {
            BROADCAST_CCU_CHECK(ccu::LoadArg(sliceOffsets[i], argId++));
        }
        for (uint32_t i = 0; i < sliceNum; ++i) {
            BROADCAST_CCU_CHECK(ccu::LoadArg(sliceSizes[i], argId++));
        }

        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            const ChannelHandle channel = kernelArg->channels[i];
            remoteBases[i] = ccu::GetResByChannel<ccu::Variable>(channel, outputResourceId);
            remoteTokens[i] = ccu::GetResByChannel<ccu::Variable>(channel, tokenResourceId);
            BROADCAST_CCU_CHECK(ccu::NotifyWait(channel, ckeIndex, (1U << outputNotifyId) | (1U << tokenNotifyId)));
        }
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            const uint32_t slice = kernelArg->peerSlices[i];
            ccu::LocalAddr source;
            source.addr = localBase;
            source.addr += sliceOffsets[slice];
            source.token = localToken;
            ccu::RemoteAddr destination;
            destination.addr = remoteBases[i];
            destination.addr += sliceOffsets[slice];
            destination.token = remoteTokens[i];
            BROADCAST_CCU_CHECK(
                ccu::Write(kernelArg->channels[i], destination, source, sliceSizes[slice], transferDone, 1U << i));
        }
        const uint16_t transferMask = (1U << kernelArg->channelCount) - 1;
        BROADCAST_CCU_CHECK(ccu::EventWait(transferDone, transferMask));
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            BROADCAST_CCU_CHECK(ccu::NotifyRecord(kernelArg->channels[i], ckeIndex, 1U << postSyncId));
        }
    } else {
        const ChannelHandle channel = kernelArg->channels[0];
        BROADCAST_CCU_CHECK(
            ccu::WriteVariableWithNotify(channel, localBase, outputResourceId, ckeIndex, 1U << outputNotifyId));
        BROADCAST_CCU_CHECK(
            ccu::WriteVariableWithNotify(channel, localToken, tokenResourceId, ckeIndex, 1U << tokenNotifyId));
        BROADCAST_CCU_CHECK(ccu::NotifyWait(channel, ckeIndex, 1U << postSyncId));
    }
    return CCU_SUCCESS;
}

CcuResult CcuSixteenFlatAllGatherKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuSixteenFlatKernelArg *>(arg);
    constexpr uint32_t outputResourceId = 1;
    constexpr uint32_t tokenResourceId = 2;
    constexpr uint32_t outputNotifyId = 0;
    constexpr uint32_t tokenNotifyId = 1;
    constexpr uint32_t postSyncId = 2;
    constexpr uint32_t ckeIndex = 0;
    ccu::Variable localBase;
    ccu::Variable localToken;
    ccu::Variable sourceOffset;
    ccu::Variable sourceSize;
    uint32_t argId = 0;
    BROADCAST_CCU_CHECK(ccu::LoadArg(localBase, argId++));
    BROADCAST_CCU_CHECK(ccu::LoadArg(localToken, argId++));
    BROADCAST_CCU_CHECK(ccu::LoadArg(sourceOffset, argId++));
    BROADCAST_CCU_CHECK(ccu::LoadArg(sourceSize, argId++));

    std::vector<ccu::Variable> remoteBases(kernelArg->channelCount);
    std::vector<ccu::Variable> remoteTokens(kernelArg->channelCount);
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        const ChannelHandle channel = kernelArg->channels[i];
        remoteBases[i] = ccu::GetResByChannel<ccu::Variable>(channel, outputResourceId);
        remoteTokens[i] = ccu::GetResByChannel<ccu::Variable>(channel, tokenResourceId);
        BROADCAST_CCU_CHECK(
            ccu::WriteVariableWithNotify(channel, localBase, outputResourceId, ckeIndex, 1U << outputNotifyId));
        BROADCAST_CCU_CHECK(
            ccu::WriteVariableWithNotify(channel, localToken, tokenResourceId, ckeIndex, 1U << tokenNotifyId));
    }
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        BROADCAST_CCU_CHECK(
            ccu::NotifyWait(kernelArg->channels[i], ckeIndex, (1U << outputNotifyId) | (1U << tokenNotifyId)));
    }

    ccu::LocalAddr source;
    source.addr = localBase;
    source.addr += sourceOffset;
    source.token = localToken;
    if (kernelArg->serialized != 0) {
        // 跨机保持既有循环匹配顺序，但每轮同时维持两个完整 D/16 peer Write。
        // 数据不再细分 wave；每批只减少一次 EventWait，不增加 kernel/notify 数量。
        constexpr uint32_t peerBatchSize = 2;
        for (uint32_t batchBegin = 0; batchBegin < kernelArg->channelCount; batchBegin += peerBatchSize) {
            ccu::Event transferDone;
            uint16_t transferMask = 0;
            const uint32_t nextBatch = batchBegin + peerBatchSize;
            const uint32_t batchEnd = nextBatch < kernelArg->channelCount ? nextBatch : kernelArg->channelCount;
            for (uint32_t i = batchBegin; i < batchEnd; ++i) {
                ccu::RemoteAddr destination;
                destination.addr = remoteBases[i];
                destination.addr += sourceOffset;
                destination.token = remoteTokens[i];
                const uint16_t eventBit = static_cast<uint16_t>(1U << (i - batchBegin));
                BROADCAST_CCU_CHECK(ccu::Write(
                    kernelArg->channels[i], destination, source, sourceSize, transferDone, eventBit));
                transferMask = static_cast<uint16_t>(transferMask | eventBit);
            }
            BROADCAST_CCU_CHECK(ccu::EventWait(transferDone, transferMask));
            for (uint32_t i = batchBegin; i < batchEnd; ++i) {
                BROADCAST_CCU_CHECK(
                    ccu::NotifyRecord(kernelArg->channels[i], ckeIndex, 1U << postSyncId));
            }
            if (kernelArg->deferCompletionWait == 0U) {
                for (uint32_t i = batchBegin; i < batchEnd; ++i) {
                    BROADCAST_CCU_CHECK(
                        ccu::NotifyWait(kernelArg->channels[i], ckeIndex, 1U << postSyncId));
                }
            }
        }
        if (kernelArg->deferCompletionWait != 0U) {
            for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
                BROADCAST_CCU_CHECK(ccu::NotifyWait(kernelArg->channels[i], ckeIndex, 1U << postSyncId));
            }
        }
    } else {
        ccu::Event transferDone;
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            ccu::RemoteAddr destination;
            destination.addr = remoteBases[i];
            destination.addr += sourceOffset;
            destination.token = remoteTokens[i];
            BROADCAST_CCU_CHECK(ccu::Write(
                kernelArg->channels[i], destination, source, sourceSize, transferDone, 1U << i));
        }
        BROADCAST_CCU_CHECK(ccu::EventWait(transferDone, static_cast<uint16_t>((1U << kernelArg->channelCount) - 1)));
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            BROADCAST_CCU_CHECK(ccu::NotifyRecord(kernelArg->channels[i], ckeIndex, 1U << postSyncId));
        }
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            BROADCAST_CCU_CHECK(ccu::NotifyWait(kernelArg->channels[i], ckeIndex, 1U << postSyncId));
        }
    }
    return CCU_SUCCESS;
}

} // namespace ops_hccl

#undef BROADCAST_CCU_CHECK
