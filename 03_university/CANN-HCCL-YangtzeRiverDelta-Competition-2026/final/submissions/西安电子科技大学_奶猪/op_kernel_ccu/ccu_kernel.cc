/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include <hcomm/hcomm_primitives.h>

#include <cstdint>
#include <vector>

#include "ccu_kernel.h"
#include "custom.h"
#include "log.h"

namespace ops_hccl {
namespace ccu = ::AscendC::ccu;

#define FOREST_CCU_CHECK(expression) \
    do { \
        const CcuResult forestResult = (expression); \
        if (forestResult != CCU_SUCCESS) { \
            return forestResult; \
        } \
    } while (0)

namespace {
constexpr uint32_t REMOTE_BASE_SLOT = 0;
constexpr uint32_t REMOTE_TOKEN_SLOT = 1;
constexpr uint32_t PARAMETER_NOTIFY = 0;
constexpr uint32_t SEED_NOTIFY = 1;
constexpr uint32_t TREE_NOTIFY = 2;
constexpr uint32_t RING_ACK_NOTIFY = 3;
constexpr uint16_t BASE_READY = 1U << 0;
constexpr uint16_t TOKEN_READY = 1U << 1;
constexpr uint16_t PAYLOAD_READY = 1U;

struct ForestState {
    const ForestKernelConfig *config{nullptr};
    ccu::Variable localBase;
    ccu::Variable localToken;
    ccu::Variable phase;
    ccu::Variable offset;
    ccu::Variable bytes;
    std::vector<ccu::Variable> remoteBases;
    std::vector<ccu::Variable> remoteTokens;
    std::vector<ccu::Variable> ringOffsets;
    std::vector<ccu::Variable> ringBytes;
};

CcuResult LoadState(ForestState &state)
{
    FOREST_CCU_CHECK(ccu::LoadArg(state.localBase, 0));
    FOREST_CCU_CHECK(ccu::LoadArg(state.localToken, 1));
    FOREST_CCU_CHECK(ccu::LoadArg(state.phase, 2));
    state.ringOffsets.reserve(FOREST_RING_COUNT);
    state.ringBytes.reserve(FOREST_RING_COUNT);
    for (uint32_t ring = 0; ring < FOREST_RING_COUNT; ++ring) {
        state.ringOffsets.emplace_back();
        FOREST_CCU_CHECK(ccu::LoadArg(state.ringOffsets.back(), 3 + ring));
    }
    for (uint32_t ring = 0; ring < FOREST_RING_COUNT; ++ring) {
        state.ringBytes.emplace_back();
        FOREST_CCU_CHECK(ccu::LoadArg(state.ringBytes.back(), 3 + FOREST_RING_COUNT + ring));
    }
    state.remoteBases.reserve(state.config->channelCount);
    state.remoteTokens.reserve(state.config->channelCount);
    for (uint32_t channel = 0; channel < state.config->channelCount; ++channel) {
        state.remoteBases.push_back(
            ccu::GetResByChannel<ccu::Variable>(state.config->channels[channel], REMOTE_BASE_SLOT));
        state.remoteTokens.push_back(
            ccu::GetResByChannel<ccu::Variable>(state.config->channels[channel], REMOTE_TOKEN_SLOT));
    }
    return CCU_SUCCESS;
}

CcuResult PublishAddress(ForestState &state, uint32_t channelIndex)
{
    const ChannelHandle channel = state.config->channels[channelIndex];
    FOREST_CCU_CHECK(ccu::WriteVariableWithNotify(
        channel, state.localBase, REMOTE_BASE_SLOT, PARAMETER_NOTIFY, BASE_READY));
    return ccu::WriteVariableWithNotify(
        channel, state.localToken, REMOTE_TOKEN_SLOT, PARAMETER_NOTIFY, TOKEN_READY);
}

CcuResult AwaitAddress(ForestState &state, uint32_t channelIndex)
{
    return ccu::NotifyWait(state.config->channels[channelIndex], PARAMETER_NOTIFY,
        static_cast<uint16_t>(BASE_READY | TOKEN_READY));
}

void SetLocalSource(ForestState &state, uint32_t lane, ccu::LocalAddr &source)
{
    source.addr = state.localBase;
    state.offset = state.config->laneOffsets[lane];
    source.addr += state.offset;
    source.token = state.localToken;
    state.bytes = state.config->laneBytes[lane];
}

void SetRemoteDestination(ForestState &state, uint32_t lane,
    uint32_t channelIndex, ccu::RemoteAddr &destination)
{
    destination.addr = state.remoteBases[channelIndex];
    state.offset = state.config->laneOffsets[lane];
    destination.addr += state.offset;
    destination.token = state.remoteTokens[channelIndex];
}

CcuResult RunDirect(ForestState &state)
{
    const ForestKernelConfig &config = *state.config;
    if (config.rankId == config.rootRank) {
        for (uint32_t channel = 0; channel < config.channelCount; ++channel) {
            FOREST_CCU_CHECK(PublishAddress(state, channel));
        }
        for (uint32_t channel = 0; channel < config.channelCount; ++channel) {
            FOREST_CCU_CHECK(AwaitAddress(state, channel));
        }
        ccu::Event completion;
        uint16_t completionMask = 0;
        for (uint32_t channel = 0; channel < config.channelCount; ++channel) {
            const uint16_t completionBit = static_cast<uint16_t>(1U << channel);
            ccu::LocalAddr source;
            source.addr = state.localBase;
            source.token = state.localToken;
            ccu::RemoteAddr destination;
            destination.addr = state.remoteBases[channel];
            destination.token = state.remoteTokens[channel];
            state.bytes = config.totalBytes;
            FOREST_CCU_CHECK(ccu::Write(config.channels[channel], destination,
                source, state.bytes, completion, completionBit));
            completionMask = static_cast<uint16_t>(completionMask | completionBit);
        }
        FOREST_CCU_CHECK(ccu::EventWait(completion, completionMask));
        for (uint32_t channel = 0; channel < config.channelCount; ++channel) {
            FOREST_CCU_CHECK(ccu::NotifyRecord(
                config.channels[channel], SEED_NOTIFY, PAYLOAD_READY));
        }
        return CCU_SUCCESS;
    }

    for (uint32_t channel = 0; channel < config.channelCount; ++channel) {
        if (config.peerRanks[channel] != config.rootRank) {
            continue;
        }
        FOREST_CCU_CHECK(PublishAddress(state, channel));
        FOREST_CCU_CHECK(AwaitAddress(state, channel));
        return ccu::NotifyWait(config.channels[channel], SEED_NOTIFY, PAYLOAD_READY);
    }
    // A non-root rank may own a second DIE whose channel group does not
    // contain the root.  That kernel is deliberately not launched in direct
    // mode, but registration still builds its static graph and must succeed.
    return CCU_SUCCESS;
}

CcuResult RunSeed(ForestState &state)
{
    const ForestKernelConfig &config = *state.config;
    for (uint32_t channel = 0; channel < config.channelCount; ++channel) {
        FOREST_CCU_CHECK(PublishAddress(state, channel));
    }
    for (uint32_t channel = 0; channel < config.channelCount; ++channel) {
        FOREST_CCU_CHECK(AwaitAddress(state, channel));
    }

    if (config.rankId == config.rootRank) {
        ccu::Event completion;
        uint16_t completionMask = 0;
        for (uint32_t channel = 0; channel < config.channelCount; ++channel) {
            const uint16_t completionBit = static_cast<uint16_t>(1U << channel);
            const uint32_t lane = config.peerRanks[channel];
            ccu::LocalAddr source;
            ccu::RemoteAddr destination;
            SetLocalSource(state, lane, source);
            SetRemoteDestination(state, lane, channel, destination);
            FOREST_CCU_CHECK(ccu::Write(config.channels[channel], destination,
                source, state.bytes, completion, completionBit));
            completionMask = static_cast<uint16_t>(completionMask | completionBit);
        }
        FOREST_CCU_CHECK(ccu::EventWait(completion, completionMask));
        for (uint32_t channel = 0; channel < config.channelCount; ++channel) {
            FOREST_CCU_CHECK(ccu::NotifyRecord(
                config.channels[channel], SEED_NOTIFY, PAYLOAD_READY));
        }
        return CCU_SUCCESS;
    }

    for (uint32_t channel = 0; channel < config.channelCount; ++channel) {
        if (config.peerRanks[channel] == config.rootRank) {
            return ccu::NotifyWait(config.channels[channel], SEED_NOTIFY, PAYLOAD_READY);
        }
    }
    return CCU_SUCCESS;
}

uint32_t CountSends(const ForestKernelConfig &config, uint32_t stage)
{
    uint32_t count = 0;
    for (uint32_t lane = 0; lane < config.rankSize; ++lane) {
        const uint32_t children = config.childRankMasks[stage][lane];
        for (uint32_t channel = 0; channel < config.channelCount; ++channel) {
            if ((children & (1U << config.peerRanks[channel])) != 0) {
                ++count;
            }
        }
    }
    return count;
}

CcuResult RunTreeLevel(ForestState &state, uint32_t stage)
{
    const ForestKernelConfig &config = *state.config;
    const uint32_t sendCount = CountSends(config, stage);
    std::vector<ccu::Event> completions(sendCount);
    uint32_t completionIndex = 0;
    for (uint32_t lane = 0; lane < config.rankSize; ++lane) {
        const uint32_t children = config.childRankMasks[stage][lane];
        for (uint32_t channel = 0; channel < config.channelCount; ++channel) {
            if ((children & (1U << config.peerRanks[channel])) == 0) {
                continue;
            }
            ccu::LocalAddr source;
            ccu::RemoteAddr destination;
            SetLocalSource(state, lane, source);
            SetRemoteDestination(state, lane, channel, destination);
            FOREST_CCU_CHECK(ccu::Write(config.channels[channel], destination,
                source, state.bytes, completions[completionIndex++]));
        }
    }
    for (ccu::Event &event : completions) {
        FOREST_CCU_CHECK(ccu::EventWait(event));
    }

    for (uint32_t channel = 0; channel < config.channelCount; ++channel) {
        uint16_t laneMask = 0;
        const uint32_t peerBit = 1U << config.peerRanks[channel];
        for (uint32_t lane = 0; lane < config.rankSize; ++lane) {
            if ((config.childRankMasks[stage][lane] & peerBit) != 0) {
                laneMask = static_cast<uint16_t>(laneMask | (1U << lane));
            }
        }
        if (laneMask != 0) {
            FOREST_CCU_CHECK(ccu::NotifyRecord(config.channels[channel], TREE_NOTIFY, laneMask));
        }
    }

    for (uint32_t channel = 0; channel < config.channelCount; ++channel) {
        uint16_t laneMask = 0;
        for (uint32_t lane = 0; lane < config.rankSize; ++lane) {
            if (config.parentRanks[stage][lane] == config.peerRanks[channel]) {
                laneMask = static_cast<uint16_t>(laneMask | (1U << lane));
            }
        }
        if (laneMask != 0) {
            FOREST_CCU_CHECK(ccu::NotifyWait(config.channels[channel], TREE_NOTIFY, laneMask));
        }
    }
    return CCU_SUCCESS;
}

CcuResult RunRingStep(ForestState &state)
{
    const ForestKernelConfig &config = *state.config;
    std::vector<uint32_t> sendChannels;
    std::vector<uint32_t> sendRings;
    for (uint32_t ring = 0; ring < FOREST_RING_COUNT; ++ring) {
        for (uint32_t channel = 0; channel < config.channelCount; ++channel) {
            if (config.peerRanks[channel] == config.ringNextRanks[ring]) {
                sendChannels.push_back(channel);
                sendRings.push_back(ring);
                break;
            }
        }
    }

    std::vector<ccu::Event> completions(sendChannels.size());
    for (uint32_t index = 0; index < sendChannels.size(); ++index) {
        const uint32_t channel = sendChannels[index];
        const uint32_t ring = sendRings[index];
        ccu::LocalAddr source;
        source.addr = state.localBase;
        source.addr += state.ringOffsets[ring];
        source.token = state.localToken;
        ccu::RemoteAddr destination;
        destination.addr = state.remoteBases[channel];
        destination.addr += state.ringOffsets[ring];
        destination.token = state.remoteTokens[channel];
        FOREST_CCU_CHECK(ccu::Write(config.channels[channel], destination,
            source, state.ringBytes[ring], completions[index]));
    }
    for (ccu::Event &completion : completions) {
        FOREST_CCU_CHECK(ccu::EventWait(completion));
    }

    for (uint32_t channel = 0; channel < config.channelCount; ++channel) {
        uint16_t ringMask = 0;
        for (uint32_t ring = 0; ring < FOREST_RING_COUNT; ++ring) {
            if (config.ringNextRanks[ring] == config.peerRanks[channel]) {
                ringMask = static_cast<uint16_t>(ringMask | (1U << ring));
            }
        }
        if (ringMask != 0) {
            FOREST_CCU_CHECK(ccu::NotifyRecord(config.channels[channel], TREE_NOTIFY, ringMask));
        }
    }
    for (uint32_t channel = 0; channel < config.channelCount; ++channel) {
        uint16_t ringMask = 0;
        for (uint32_t ring = 0; ring < FOREST_RING_COUNT; ++ring) {
            if (config.ringPreviousRanks[ring] == config.peerRanks[channel]) {
                ringMask = static_cast<uint16_t>(ringMask | (1U << ring));
            }
        }
        if (ringMask != 0) {
            FOREST_CCU_CHECK(ccu::NotifyWait(config.channels[channel], TREE_NOTIFY, ringMask));
        }
    }

    // A notify bit is binary.  Acknowledge its consumption before the sender
    // can reuse the same bit in the next ring step.
    for (uint32_t channel = 0; channel < config.channelCount; ++channel) {
        uint16_t ringMask = 0;
        for (uint32_t ring = 0; ring < FOREST_RING_COUNT; ++ring) {
            if (config.ringPreviousRanks[ring] == config.peerRanks[channel]) {
                ringMask = static_cast<uint16_t>(ringMask | (1U << ring));
            }
        }
        if (ringMask != 0) {
            FOREST_CCU_CHECK(ccu::NotifyRecord(
                config.channels[channel], RING_ACK_NOTIFY, ringMask));
        }
    }
    for (uint32_t channel = 0; channel < config.channelCount; ++channel) {
        uint16_t ringMask = 0;
        for (uint32_t ring = 0; ring < FOREST_RING_COUNT; ++ring) {
            if (config.ringNextRanks[ring] == config.peerRanks[channel]) {
                ringMask = static_cast<uint16_t>(ringMask | (1U << ring));
            }
        }
        if (ringMask != 0) {
            FOREST_CCU_CHECK(ccu::NotifyWait(
                config.channels[channel], RING_ACK_NOTIFY, ringMask));
        }
    }
    return CCU_SUCCESS;
}
} // namespace

CcuResult StripedForestKernel(CcuKernelArg arg)
{
    auto *config = static_cast<ForestKernelConfig *>(arg);
    if (config == nullptr || config->rankSize < 2 || config->rankSize > MAX_RANK_SIZE
        || config->rankId >= config->rankSize || config->rootRank >= config->rankSize
        || config->channelCount == 0 || config->channelCount >= config->rankSize) {
        HCCL_ERROR("[ForestV4] invalid static kernel configuration");
        return CCU_E_PARA;
    }

    ForestState state;
    state.config = config;
    FOREST_CCU_CHECK(LoadState(state));
    if (config->directMode != 0) {
        return RunDirect(state);
    }

    CCU_IF(state.phase == static_cast<uint32_t>(ForestPhase::SEED))
    {
        FOREST_CCU_CHECK(RunSeed(state));
    }
    CCU_IF(state.phase == static_cast<uint32_t>(ForestPhase::LEVEL_ZERO))
    {
        FOREST_CCU_CHECK(RunTreeLevel(state, 0));
    }
    CCU_IF(state.phase == static_cast<uint32_t>(ForestPhase::LEVEL_ONE))
    {
        FOREST_CCU_CHECK(RunTreeLevel(state, 1));
    }
    CCU_IF(state.phase == static_cast<uint32_t>(ForestPhase::RING))
    {
        FOREST_CCU_CHECK(RunRingStep(state));
    }
    return CCU_SUCCESS;
}

#undef FOREST_CCU_CHECK
} // namespace ops_hccl
