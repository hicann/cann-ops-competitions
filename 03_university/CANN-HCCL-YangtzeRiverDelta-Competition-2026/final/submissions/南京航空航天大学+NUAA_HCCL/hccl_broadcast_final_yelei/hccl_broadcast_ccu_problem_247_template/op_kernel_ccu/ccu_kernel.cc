/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

#include <hcomm/hcomm_primitives.h>

#include "ccu_kernel.h"

namespace ccu = ::AscendC::ccu;

#ifndef CCU_CHK_RET
#define CCU_CHK_RET(call)                          \
    do {                                           \
        CcuResult ccuResult = (call);              \
        if (ccuResult != CcuResult::CCU_SUCCESS) { \
            return ccuResult;                      \
        }                                          \
    } while (0)
#endif

namespace ops_hccl {
namespace {

constexpr uint32_t OUTPUT_XN_ID = 1;
constexpr uint32_t TOKEN_XN_ID = 2;
constexpr uint32_t PREPARE_MASK = (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);

bool IsChannelMapValid(const CcuKernelArgBroadcast *arg)
{
    if (arg->rankSize <= 1 || arg->channelCount == 0 || arg->channelCount >= arg->rankSize ||
        arg->ownerCount == 0 || arg->ownerCount > arg->rankSize) {
        return false;
    }
    for (uint32_t index = 0; index < arg->channelCount; ++index) {
        if (arg->peerRanks[index] >= arg->rankSize || arg->peerRanks[index] == arg->rankId) {
            return false;
        }
        for (uint32_t previous = 0; previous < index; ++previous) {
            if (arg->peerRanks[previous] == arg->peerRanks[index]) {
                return false;
            }
        }
    }
    return true;
}

uint32_t FindChannelIndex(const CcuKernelArgBroadcast *arg, uint32_t peerRank)
{
    for (uint32_t index = 0; index < arg->channelCount; ++index) {
        if (arg->peerRanks[index] == peerRank) {
            return index;
        }
    }
    return arg->channelCount;
}

uint32_t FindRankIndex(const uint32_t *ranks, uint32_t rankCount, uint32_t rank)
{
    for (uint32_t index = 0; index < rankCount; ++index) {
        if (ranks[index] == rank) {
            return index;
        }
    }
    return rankCount;
}

uint32_t PathRelativeRank(uint32_t pathId, uint32_t position)
{
    return ((pathId + position - 1) % F2_PATH_COUNT) + 1;
}

uint32_t RankAtPathPosition(const CcuKernelArgBroadcast *arg, uint32_t pathId, uint32_t position)
{
    if (position == 0) {
        return arg->root;
    }
    return (arg->root + PathRelativeRank(pathId, position)) % arg->rankSize;
}

uint32_t PathForPosition(const CcuKernelArgBroadcast *arg, uint32_t position)
{
    const uint32_t relative = (arg->rankId + arg->rankSize - arg->root) % arg->rankSize;
    for (uint32_t path = 0; path < F2_PATH_COUNT; ++path) {
        if (PathRelativeRank(path, position) == relative) {
            return path;
        }
    }
    return F2_PATH_COUNT;
}

bool LocalWritesTo(const CcuKernelArgBroadcast *arg, uint32_t source, uint32_t destination)
{
    if (source == destination) {
        return false;
    }
    if (arg->topologyMode == F2_TOPO_FLAT_FOUR) {
        const uint32_t stripe = FindRankIndex(arg->sourceOwners, arg->ownerCount, source);
        if (stripe >= arg->ownerCount || destination == arg->root) {
            return false;
        }
        return source == arg->root || destination != source;
    }

    const uint32_t sourceStripe = FindRankIndex(arg->sourceOwners, arg->ownerCount, source);
    if (sourceStripe < arg->ownerCount) {
        return destination != arg->root &&
            FindRankIndex(arg->rootServerRanks, arg->rootServerSize, destination) < arg->rootServerSize;
    }
    const uint32_t destinationStripe = FindRankIndex(arg->destinationOwners, arg->ownerCount, source);
    if (destinationStripe < arg->ownerCount) {
        return FindRankIndex(arg->remoteServerRanks, arg->remoteServerSize, destination) < arg->remoteServerSize;
    }
    return false;
}

bool WritesTo(const CcuKernelArgBroadcast *arg, uint32_t source, uint32_t destination)
{
    if (source == destination) {
        return false;
    }
    for (uint32_t stripe = 0; stripe < arg->ownerCount; ++stripe) {
        if (source == arg->root && destination == arg->sourceOwners[stripe] && destination != arg->root) {
            return true;
        }
        if (arg->topologyMode == F2_TOPO_TWO_SERVER && source == arg->root &&
            destination == arg->destinationOwners[stripe]) {
            return true;
        }
    }
    return LocalWritesTo(arg, source, destination);
}

bool HasEarlierWrite(const CcuKernelArgBroadcast *arg, uint32_t source, uint32_t destination)
{
    for (uint32_t stripe = 0; stripe < arg->ownerCount; ++stripe) {
        if (source == arg->root && destination == arg->sourceOwners[stripe] &&
            destination != arg->root) {
            return true;
        }
        if (arg->topologyMode == F2_TOPO_TWO_SERVER && source == arg->root &&
            destination == arg->destinationOwners[stripe]) {
            return true;
        }
    }
    return false;
}

void InitRemoteResources(const CcuKernelArgBroadcast *arg, std::vector<ccu::Variable> &outputs,
    std::vector<ccu::Variable> &tokens)
{
    outputs.resize(arg->rankSize);
    tokens.resize(arg->rankSize);
    for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
        const uint32_t peer = arg->peerRanks[channel];
        outputs[peer] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channel], OUTPUT_XN_ID);
        tokens[peer] = ccu::GetResByChannel<ccu::Variable>(arg->channels[channel], TOKEN_XN_ID);
    }
}

CcuResult LoadLocalResources(ccu::Variable &output, ccu::Variable &token)
{
    uint32_t argIndex = 0;
    CCU_CHK_RET(ccu::LoadArg(output, argIndex++));
    CCU_CHK_RET(ccu::LoadArg(token, argIndex++));
    return CcuResult::CCU_SUCCESS;
}

CcuResult WriteStripe(const CcuKernelArgBroadcast *arg, uint32_t channel, uint32_t peer,
    const ccu::Variable &localOutput, const ccu::Variable &localToken,
    const std::vector<ccu::Variable> &outputs, const std::vector<ccu::Variable> &tokens,
    uint32_t stripe, ccu::Event &completed, uint16_t mask)
{
    ccu::Variable offset;
    ccu::Variable size;
    offset = arg->stripeOffsets[stripe];
    size = arg->stripeSizes[stripe];

    ccu::LocalAddr source;
    source.addr = localOutput;
    source.addr += offset;
    source.token = localToken;

    ccu::RemoteAddr destination;
    destination.addr = outputs[peer];
    destination.addr += offset;
    destination.token = tokens[peer];
    return ccu::Write(arg->channels[channel], destination, source, size, completed, mask);
}

CcuResult PrepareAllChannels(const CcuKernelArgBroadcast *arg,
    const ccu::Variable &localOutput, const ccu::Variable &localToken)
{
    for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[channel],
            localOutput, OUTPUT_XN_ID, F2_PREPARE_NOTIFY, 1U << OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[channel],
            localToken, TOKEN_XN_ID, F2_PREPARE_NOTIFY, 1U << TOKEN_XN_ID));
    }
    for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
        CCU_CHK_RET(ccu::NotifyWait(
            arg->channels[channel], F2_PREPARE_NOTIFY, PREPARE_MASK));
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult WriteRange(const CcuKernelArgBroadcast *arg, uint32_t channel, uint32_t peer,
    const ccu::Variable &localOutput, const ccu::Variable &localToken,
    const std::vector<ccu::Variable> &outputs, const std::vector<ccu::Variable> &tokens,
    uint64_t rawOffset, uint64_t rawSize, ccu::Event &completed, uint16_t mask)
{
    ccu::Variable offset;
    ccu::Variable size;
    offset = rawOffset;
    size = rawSize;

    ccu::LocalAddr source;
    source.addr = localOutput;
    source.addr += offset;
    source.token = localToken;

    ccu::RemoteAddr destination;
    destination.addr = outputs[peer];
    destination.addr += offset;
    destination.token = tokens[peer];
    return ccu::Write(arg->channels[channel], destination, source, size, completed, mask);
}

CcuResult WritePipelineChunk(const CcuKernelArgBroadcast *arg, uint32_t channel, uint32_t peer,
    const ccu::Variable &localOutput, const ccu::Variable &localToken,
    const std::vector<ccu::Variable> &outputs, const std::vector<ccu::Variable> &tokens,
    uint32_t stripe, uint32_t chunk, ccu::Event &completed, uint16_t mask)
{
    return WriteRange(arg, channel, peer, localOutput, localToken, outputs, tokens,
        arg->pipelineOffsets[stripe][chunk], arg->pipelineSizes[stripe][chunk], completed, mask);
}

uint32_t PipelineStageNotify()
{
    return 1U;
}

uint32_t PipelineStageMask(uint32_t chunk)
{
    return 1U << chunk;
}

uint32_t PipelineFinalNotify(uint32_t stripe)
{
    return stripe;
}

uint32_t PipelineFinalMask(uint32_t chunk)
{
    return 1U << chunk;
}

bool IsPipelineChannelMapValid(const CcuKernelArgBroadcast *arg)
{
    if (!IsChannelMapValid(arg) || arg->pipelineRootServerSize < 2 ||
        arg->pipelineRelayServerSize == 0 ||
        arg->pipelineRelayServerSize > F2_PIPELINE_MAX_RELAY_COUNT ||
        arg->pipelineRootServerRanks[0] != arg->root) {
        return false;
    }

    const uint32_t rootIndex = FindRankIndex(
        arg->pipelineRootServerRanks, arg->pipelineRootServerSize, arg->rankId);
    const uint32_t relayIndex = FindRankIndex(
        arg->pipelineRelayServerRanks, arg->pipelineRelayServerSize, arg->rankId);
    if ((rootIndex < arg->pipelineRootServerSize) == (relayIndex < arg->pipelineRelayServerSize)) {
        return false;
    }

    const uint32_t expectedChannels =
        rootIndex < arg->pipelineRootServerSize ? arg->pipelineRelayServerSize : arg->pipelineRootServerSize;
    if (arg->channelCount != expectedChannels) {
        return false;
    }
    const uint32_t *expectedRanks = rootIndex < arg->pipelineRootServerSize ?
        arg->pipelineRelayServerRanks : arg->pipelineRootServerRanks;
    for (uint32_t index = 0; index < expectedChannels; ++index) {
        if (FindChannelIndex(arg, expectedRanks[index]) >= arg->channelCount) {
            return false;
        }
    }
    return true;
}

CcuResult CcuDirectKernelImpl(CcuKernelArg rawArg, bool prepareResources)
{
    auto *arg = static_cast<CcuKernelArgBroadcast *>(rawArg);
    if (!IsChannelMapValid(arg)) {
        return CcuResult::CCU_E_INTERNAL;
    }

    std::vector<ccu::Variable> outputs;
    std::vector<ccu::Variable> tokens;
    InitRemoteResources(arg, outputs, tokens);

    ccu::Variable localOutput;
    ccu::Variable localToken;
    CCU_CHK_RET(LoadLocalResources(localOutput, localToken));
    ccu::Variable offset;
    ccu::Variable dataSize;
    offset = arg->offset;
    dataSize = arg->dataSize;

    if (arg->rankId == arg->root) {
        if (prepareResources) {
            for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
                CCU_CHK_RET(ccu::NotifyWait(arg->channels[channel], F2_PREPARE_NOTIFY, PREPARE_MASK));
            }
        }

        ccu::LocalAddr source;
        source.addr = localOutput;
        source.addr += offset;
        source.token = localToken;

        ccu::Event completed;
        uint16_t totalMask = 0;
        for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
            const uint32_t peer = arg->peerRanks[channel];
            ccu::RemoteAddr destination;
            destination.addr = outputs[peer];
            destination.addr += offset;
            destination.token = tokens[peer];
            const uint16_t mask = static_cast<uint16_t>(1U << channel);
            CCU_CHK_RET(ccu::Write(arg->channels[channel], destination, source, dataSize, completed, mask));
            totalMask = static_cast<uint16_t>(totalMask | mask);
        }
        CCU_CHK_RET(ccu::EventWait(completed, totalMask));
        for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
            CCU_CHK_RET(ccu::NotifyRecord(arg->channels[channel], arg->notifyIndex, 1));
        }
    } else {
        const uint32_t rootChannel = FindChannelIndex(arg, arg->root);
        if (rootChannel >= arg->channelCount) {
            return CcuResult::CCU_SUCCESS;
        }
        if (prepareResources) {
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[rootChannel],
                localOutput, OUTPUT_XN_ID, F2_PREPARE_NOTIFY, 1U << OUTPUT_XN_ID));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[rootChannel],
                localToken, TOKEN_XN_ID, F2_PREPARE_NOTIFY, 1U << TOKEN_XN_ID));
        }
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[rootChannel], arg->notifyIndex, 1));
    }
    return CcuResult::CCU_SUCCESS;
}

} // namespace

CcuResult CcuPrepareKernel(CcuKernelArg rawArg)
{
    auto *arg = static_cast<CcuKernelArgBroadcast *>(rawArg);
    if (!IsChannelMapValid(arg)) {
        return CcuResult::CCU_E_INTERNAL;
    }

    ccu::Variable localOutput;
    ccu::Variable localToken;
    CCU_CHK_RET(LoadLocalResources(localOutput, localToken));

    for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
        const uint32_t peer = arg->peerRanks[channel];
        if (!WritesTo(arg, peer, arg->rankId)) {
            continue;
        }
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[channel],
            localOutput, OUTPUT_XN_ID, F2_PREPARE_NOTIFY, 1U << OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[channel],
            localToken, TOKEN_XN_ID, F2_PREPARE_NOTIFY, 1U << TOKEN_XN_ID));
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuDirectKernel0(CcuKernelArg rawArg)
{
    return CcuDirectKernelImpl(rawArg, true);
}

CcuResult CcuDirectKernel1(CcuKernelArg rawArg)
{
    return CcuDirectKernelImpl(rawArg, false);
}

CcuResult CcuScatterKernel(CcuKernelArg rawArg)
{
    auto *arg = static_cast<CcuKernelArgBroadcast *>(rawArg);
    if (!IsChannelMapValid(arg)) {
        return CcuResult::CCU_E_INTERNAL;
    }

    std::vector<ccu::Variable> outputs;
    std::vector<ccu::Variable> tokens;
    InitRemoteResources(arg, outputs, tokens);
    ccu::Variable localOutput;
    ccu::Variable localToken;
    CCU_CHK_RET(LoadLocalResources(localOutput, localToken));

    for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
        const uint32_t peer = arg->peerRanks[channel];
        if (!WritesTo(arg, peer, arg->rankId)) {
            continue;
        }
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[channel],
            localOutput, OUTPUT_XN_ID, F2_PREPARE_NOTIFY, 1U << OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[channel],
            localToken, TOKEN_XN_ID, F2_PREPARE_NOTIFY, 1U << TOKEN_XN_ID));
    }

    if (arg->rankId == arg->root) {
        const uint32_t groupCount = arg->topologyMode == F2_TOPO_TWO_SERVER ? 2 : 1;
        for (uint32_t group = 0; group < groupCount; ++group) {
            for (uint32_t stripe = 0; stripe < arg->ownerCount; ++stripe) {
                const uint32_t owner = group == 0 ?
                    arg->sourceOwners[stripe] : arg->destinationOwners[stripe];
                if (owner == arg->root) {
                    continue;
                }
                const uint32_t channel = FindChannelIndex(arg, owner);
                if (channel < arg->channelCount) {
                    CCU_CHK_RET(ccu::NotifyWait(
                        arg->channels[channel], F2_PREPARE_NOTIFY, PREPARE_MASK));
                }
            }
        }
        ccu::Event completed;
        uint16_t totalMask = 0;
        for (uint32_t group = 0; group < groupCount; ++group) {
            for (uint32_t stripe = 0; stripe < arg->ownerCount; ++stripe) {
                const uint32_t owner = group == 0 ?
                    arg->sourceOwners[stripe] : arg->destinationOwners[stripe];
                if (owner == arg->root) {
                    continue;
                }
                const uint32_t channel = FindChannelIndex(arg, owner);
                if (channel >= arg->channelCount) {
                    continue;
                }
                const uint16_t mask = static_cast<uint16_t>(1U << channel);
                CCU_CHK_RET(WriteStripe(arg, channel, owner, localOutput, localToken,
                    outputs, tokens, stripe, completed, mask));
                totalMask = static_cast<uint16_t>(totalMask | mask);
            }
        }
        if (totalMask != 0) {
            CCU_CHK_RET(ccu::EventWait(completed, totalMask));
            for (uint32_t group = 0; group < groupCount; ++group) {
                for (uint32_t stripe = 0; stripe < arg->ownerCount; ++stripe) {
                    const uint32_t owner = group == 0 ?
                        arg->sourceOwners[stripe] : arg->destinationOwners[stripe];
                    if (owner == arg->root) {
                        continue;
                    }
                    const uint32_t channel = FindChannelIndex(arg, owner);
                    if (channel < arg->channelCount) {
                        CCU_CHK_RET(ccu::NotifyRecord(arg->channels[channel], F2_SCATTER_NOTIFY, 1));
                    }
                }
            }
        }
    } else {
        const uint32_t sourceStripe = FindRankIndex(arg->sourceOwners, arg->ownerCount, arg->rankId);
        const uint32_t destinationStripe =
            FindRankIndex(arg->destinationOwners, arg->ownerCount, arg->rankId);
        const uint32_t rootChannel = FindChannelIndex(arg, arg->root);
        if ((sourceStripe < arg->ownerCount || destinationStripe < arg->ownerCount) &&
            rootChannel < arg->channelCount) {
            CCU_CHK_RET(ccu::NotifyWait(arg->channels[rootChannel], F2_SCATTER_NOTIFY, 1));
        }
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuCrossKernel(CcuKernelArg rawArg)
{
    auto *arg = static_cast<CcuKernelArgBroadcast *>(rawArg);
    if (arg->topologyMode == F2_TOPO_FLAT_FOUR) {
        return CcuResult::CCU_SUCCESS;
    }
    if (!IsChannelMapValid(arg)) {
        return CcuResult::CCU_E_INTERNAL;
    }

    std::vector<ccu::Variable> outputs;
    std::vector<ccu::Variable> tokens;
    InitRemoteResources(arg, outputs, tokens);
    ccu::Variable localOutput;
    ccu::Variable localToken;
    CCU_CHK_RET(LoadLocalResources(localOutput, localToken));

    const uint32_t sourceStripe = FindRankIndex(arg->sourceOwners, arg->ownerCount, arg->rankId);
    if (sourceStripe < arg->ownerCount) {
        const uint32_t peer = arg->destinationOwners[sourceStripe];
        const uint32_t channel = FindChannelIndex(arg, peer);
        if (channel < arg->channelCount) {
            CCU_CHK_RET(ccu::NotifyWait(
                arg->channels[channel], F2_PREPARE_NOTIFY, PREPARE_MASK));
            ccu::Event completed;
            CCU_CHK_RET(WriteStripe(arg, channel, peer, localOutput, localToken,
                outputs, tokens, sourceStripe, completed, 1));
            CCU_CHK_RET(ccu::EventWait(completed, 1));
            CCU_CHK_RET(ccu::NotifyRecord(arg->channels[channel], F2_CROSS_NOTIFY, 1));
        }
    }

    const uint32_t destinationStripe = FindRankIndex(arg->destinationOwners, arg->ownerCount, arg->rankId);
    if (destinationStripe < arg->ownerCount) {
        const uint32_t peer = arg->sourceOwners[destinationStripe];
        const uint32_t channel = FindChannelIndex(arg, peer);
        if (channel < arg->channelCount) {
            CCU_CHK_RET(ccu::NotifyWait(arg->channels[channel], F2_CROSS_NOTIFY, 1));
        }
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuLocalAllgatherKernel(CcuKernelArg rawArg)
{
    auto *arg = static_cast<CcuKernelArgBroadcast *>(rawArg);
    if (!IsChannelMapValid(arg)) {
        return CcuResult::CCU_E_INTERNAL;
    }

    std::vector<ccu::Variable> outputs;
    std::vector<ccu::Variable> tokens;
    InitRemoteResources(arg, outputs, tokens);
    ccu::Variable localOutput;
    ccu::Variable localToken;
    CCU_CHK_RET(LoadLocalResources(localOutput, localToken));

    uint32_t stripe = FindRankIndex(arg->sourceOwners, arg->ownerCount, arg->rankId);
    if (stripe >= arg->ownerCount) {
        stripe = FindRankIndex(arg->destinationOwners, arg->ownerCount, arg->rankId);
    }

    ccu::Event completed;
    uint16_t totalMask = 0;
    if (stripe < arg->ownerCount) {
        for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
            const uint32_t peer = arg->peerRanks[channel];
            if (LocalWritesTo(arg, arg->rankId, peer) &&
                !HasEarlierWrite(arg, arg->rankId, peer)) {
                CCU_CHK_RET(ccu::NotifyWait(
                    arg->channels[channel], F2_PREPARE_NOTIFY, PREPARE_MASK));
            }
        }
        for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
            const uint32_t peer = arg->peerRanks[channel];
            if (!LocalWritesTo(arg, arg->rankId, peer)) {
                continue;
            }
            const uint16_t mask = static_cast<uint16_t>(1U << channel);
            CCU_CHK_RET(WriteStripe(arg, channel, peer, localOutput, localToken,
                outputs, tokens, stripe, completed, mask));
            totalMask = static_cast<uint16_t>(totalMask | mask);
        }
    }
    if (totalMask != 0) {
        CCU_CHK_RET(ccu::EventWait(completed, totalMask));
        for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
            const uint32_t peer = arg->peerRanks[channel];
            if (LocalWritesTo(arg, arg->rankId, peer)) {
                CCU_CHK_RET(ccu::NotifyRecord(arg->channels[channel], F2_LOCAL_NOTIFY, 1));
            }
        }
    }

    for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
        const uint32_t peer = arg->peerRanks[channel];
        if (LocalWritesTo(arg, peer, arg->rankId)) {
            CCU_CHK_RET(ccu::NotifyWait(arg->channels[channel], F2_LOCAL_NOTIFY, 1));
        }
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuPackedThreePathKernel(CcuKernelArg rawArg)
{
    auto *arg = static_cast<CcuKernelArgBroadcast *>(rawArg);
    if (arg->rankSize != 4 || !IsChannelMapValid(arg) || arg->channelCount != F2_PATH_COUNT) {
        return CcuResult::CCU_E_INTERNAL;
    }

    std::vector<ccu::Variable> outputs;
    std::vector<ccu::Variable> tokens;
    InitRemoteResources(arg, outputs, tokens);

    ccu::Variable localOutput;
    ccu::Variable localToken;
    CCU_CHK_RET(LoadLocalResources(localOutput, localToken));

    if (arg->rankId == arg->root) {
        for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
            CCU_CHK_RET(ccu::NotifyWait(
                arg->channels[channel], F2_PREPARE_NOTIFY, PREPARE_MASK));
        }
    } else {
        const uint32_t relative = (arg->rankId + arg->rankSize - arg->root) % arg->rankSize;
        const uint32_t previousRelative = ((relative + 1) % F2_PATH_COUNT) + 1;
        const uint32_t nextRelative = (relative % F2_PATH_COUNT) + 1;
        const uint32_t previousRank = (arg->root + previousRelative) % arg->rankSize;
        const uint32_t nextRank = (arg->root + nextRelative) % arg->rankSize;
        const uint32_t destinations[] = {arg->root, previousRank};
        for (uint32_t destination : destinations) {
            const uint32_t channel = FindChannelIndex(arg, destination);
            if (channel >= arg->channelCount) {
                return CcuResult::CCU_E_INTERNAL;
            }
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[channel],
                localOutput, OUTPUT_XN_ID, F2_PREPARE_NOTIFY, 1U << OUTPUT_XN_ID));
            CCU_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[channel],
                localToken, TOKEN_XN_ID, F2_PREPARE_NOTIFY, 1U << TOKEN_XN_ID));
        }
        const uint32_t nextChannel = FindChannelIndex(arg, nextRank);
        if (nextChannel >= arg->channelCount) {
            return CcuResult::CCU_E_INTERNAL;
        }
        CCU_CHK_RET(ccu::NotifyWait(arg->channels[nextChannel], F2_PREPARE_NOTIFY, PREPARE_MASK));
    }

    ccu::Variable offsets[F2_PATH_COUNT][F2_CHUNK_COUNT];
    ccu::Variable sizes[F2_PATH_COUNT][F2_CHUNK_COUNT];
    for (uint32_t path = 0; path < F2_PATH_COUNT; ++path) {
        for (uint32_t chunk = 0; chunk < F2_CHUNK_COUNT; ++chunk) {
            offsets[path][chunk] = arg->chunkOffsets[path][chunk];
            sizes[path][chunk] = arg->chunkSizes[path][chunk];
        }
    }

    ccu::Event completed;
    if (arg->rankId == arg->root) {
        for (uint32_t chunk = 0; chunk < F2_CHUNK_COUNT; ++chunk) {
            uint16_t totalMask = 0;
            for (uint32_t path = 0; path < F2_PATH_COUNT; ++path) {
                const uint32_t nextRank = RankAtPathPosition(arg, path, 1);
                const uint32_t channel = FindChannelIndex(arg, nextRank);
                if (channel >= arg->channelCount) {
                    return CcuResult::CCU_E_INTERNAL;
                }
                ccu::LocalAddr source;
                source.addr = localOutput;
                source.addr += offsets[path][chunk];
                source.token = localToken;
                ccu::RemoteAddr destination;
                destination.addr = outputs[nextRank];
                destination.addr += offsets[path][chunk];
                destination.token = tokens[nextRank];
                const uint16_t mask = static_cast<uint16_t>(1U << path);
                CCU_CHK_RET(ccu::Write(
                    arg->channels[channel], destination, source, sizes[path][chunk], completed, mask));
                totalMask = static_cast<uint16_t>(totalMask | mask);
            }
            CCU_CHK_RET(ccu::EventWait(completed, totalMask));
            for (uint32_t path = 0; path < F2_PATH_COUNT; ++path) {
                const uint32_t nextRank = RankAtPathPosition(arg, path, 1);
                const uint32_t channel = FindChannelIndex(arg, nextRank);
                const uint32_t notifyIndex = F2_PATH_NOTIFY_BASE + path;
                CCU_CHK_RET(ccu::NotifyRecord(arg->channels[channel], notifyIndex, 1U << chunk));
            }
        }
    } else {
        for (uint32_t chunk = 0; chunk < F2_CHUNK_COUNT; ++chunk) {
            for (uint32_t position = 1; position <= F2_PATH_COUNT; ++position) {
                const uint32_t path = PathForPosition(arg, position);
                if (path >= F2_PATH_COUNT) {
                    return CcuResult::CCU_E_INTERNAL;
                }
                const uint32_t previousRank = RankAtPathPosition(arg, path, position - 1);
                const uint32_t previousChannel = FindChannelIndex(arg, previousRank);
                if (previousChannel >= arg->channelCount) {
                    return CcuResult::CCU_E_INTERNAL;
                }
                const uint32_t notifyIndex = F2_PATH_NOTIFY_BASE + path;
                CCU_CHK_RET(ccu::NotifyWait(arg->channels[previousChannel], notifyIndex, 1U << chunk));
                if (position < F2_PATH_COUNT) {
                    const uint32_t nextRank = RankAtPathPosition(arg, path, position + 1);
                    const uint32_t nextChannel = FindChannelIndex(arg, nextRank);
                    if (nextChannel >= arg->channelCount) {
                        return CcuResult::CCU_E_INTERNAL;
                    }
                    ccu::LocalAddr source;
                    source.addr = localOutput;
                    source.addr += offsets[path][chunk];
                    source.token = localToken;
                    ccu::RemoteAddr destination;
                    destination.addr = outputs[nextRank];
                    destination.addr += offsets[path][chunk];
                    destination.token = tokens[nextRank];
                    CCU_CHK_RET(ccu::Write(
                        arg->channels[nextChannel], destination, source, sizes[path][chunk], completed, 1));
                    CCU_CHK_RET(ccu::EventWait(completed, 1));
                    CCU_CHK_RET(ccu::NotifyRecord(arg->channels[nextChannel], notifyIndex, 1U << chunk));
                }
            }
        }
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuClosPipelineKernel(CcuKernelArg rawArg)
{
    auto *arg = static_cast<CcuKernelArgBroadcast *>(rawArg);
    if (!IsPipelineChannelMapValid(arg)) {
        return CcuResult::CCU_E_INTERNAL;
    }

    std::vector<ccu::Variable> outputs;
    std::vector<ccu::Variable> tokens;
    InitRemoteResources(arg, outputs, tokens);
    ccu::Variable localOutput;
    ccu::Variable localToken;
    CCU_CHK_RET(LoadLocalResources(localOutput, localToken));
    CCU_CHK_RET(PrepareAllChannels(arg, localOutput, localToken));

    const uint32_t rootIndex = FindRankIndex(
        arg->pipelineRootServerRanks, arg->pipelineRootServerSize, arg->rankId);
    const uint32_t relayIndex = FindRankIndex(
        arg->pipelineRelayServerRanks, arg->pipelineRelayServerSize, arg->rankId);

    if (arg->rankId == arg->root) {
        for (uint32_t chunk = 0; chunk < F2_PIPELINE_CHUNK_COUNT; ++chunk) {
            ccu::Event seedDone;
            uint16_t seedMask = 0;
            for (uint32_t stripe = 0; stripe < arg->pipelineRelayServerSize; ++stripe) {
                const uint32_t peer = arg->pipelineRelayServerRanks[stripe];
                const uint32_t channel = FindChannelIndex(arg, peer);
                if (channel >= arg->channelCount) {
                    return CcuResult::CCU_E_INTERNAL;
                }
                const uint16_t mask = static_cast<uint16_t>(1U << channel);
                CCU_CHK_RET(WritePipelineChunk(arg, channel, peer, localOutput, localToken,
                    outputs, tokens, stripe, chunk, seedDone, mask));
                seedMask = static_cast<uint16_t>(seedMask | mask);
            }
            if (seedMask != 0) {
                CCU_CHK_RET(ccu::EventWait(seedDone, seedMask));
            }
            for (uint32_t stripe = 0; stripe < arg->pipelineRelayServerSize; ++stripe) {
                const uint32_t channel = FindChannelIndex(arg, arg->pipelineRelayServerRanks[stripe]);
                CCU_CHK_RET(ccu::NotifyRecord(
                    arg->channels[channel], PipelineStageNotify(), PipelineStageMask(chunk)));
            }
        }
        return CcuResult::CCU_SUCCESS;
    }

    if (relayIndex < arg->pipelineRelayServerSize) {
        const uint32_t rootChannel = FindChannelIndex(arg, arg->root);
        if (rootChannel >= arg->channelCount) {
            return CcuResult::CCU_E_INTERNAL;
        }
        for (uint32_t chunk = 0; chunk < F2_PIPELINE_CHUNK_COUNT; ++chunk) {
            CCU_CHK_RET(ccu::NotifyWait(
                arg->channels[rootChannel], PipelineStageNotify(), PipelineStageMask(chunk)));

            ccu::Event relayDone;
            uint16_t relayMask = 0;
            for (uint32_t rootPeer = 1; rootPeer < arg->pipelineRootServerSize; ++rootPeer) {
                const uint32_t peer = arg->pipelineRootServerRanks[rootPeer];
                const uint32_t channel = FindChannelIndex(arg, peer);
                if (channel >= arg->channelCount) {
                    return CcuResult::CCU_E_INTERNAL;
                }
                const uint16_t mask = static_cast<uint16_t>(1U << channel);
                CCU_CHK_RET(WritePipelineChunk(arg, channel, peer, localOutput, localToken,
                    outputs, tokens, relayIndex, chunk, relayDone, mask));
                relayMask = static_cast<uint16_t>(relayMask | mask);
            }
            CCU_CHK_RET(ccu::EventWait(relayDone, relayMask));
            for (uint32_t rootPeer = 1; rootPeer < arg->pipelineRootServerSize; ++rootPeer) {
                const uint32_t channel = FindChannelIndex(arg, arg->pipelineRootServerRanks[rootPeer]);
                CCU_CHK_RET(ccu::NotifyRecord(
                    arg->channels[channel], PipelineStageNotify(), PipelineStageMask(chunk)));
            }
        }

        const uint32_t senderCount = arg->pipelineRootServerSize - 1;
        for (uint32_t chunk = 0; chunk < F2_PIPELINE_CHUNK_COUNT; ++chunk) {
            for (uint32_t stripe = 0; stripe < arg->pipelineRelayServerSize; ++stripe) {
                if (stripe == relayIndex) {
                    continue;
                }
                const uint32_t missingPosition =
                    stripe < relayIndex ? stripe : stripe - 1;
                const uint32_t senderSlot = missingPosition % senderCount;
                const uint32_t sender = arg->pipelineRootServerRanks[1 + senderSlot];
                const uint32_t channel = FindChannelIndex(arg, sender);
                if (channel >= arg->channelCount) {
                    return CcuResult::CCU_E_INTERNAL;
                }
                CCU_CHK_RET(ccu::NotifyWait(arg->channels[channel],
                    PipelineFinalNotify(stripe), PipelineFinalMask(chunk)));
            }
        }
        return CcuResult::CCU_SUCCESS;
    }

    if (rootIndex == 0 || rootIndex >= arg->pipelineRootServerSize) {
        return CcuResult::CCU_E_INTERNAL;
    }
    const uint32_t senderCount = arg->pipelineRootServerSize - 1;
    const uint32_t senderSlot = rootIndex - 1;
    for (uint32_t chunk = 0; chunk < F2_PIPELINE_CHUNK_COUNT; ++chunk) {
        for (uint32_t stripe = 0; stripe < arg->pipelineRelayServerSize; ++stripe) {
            const uint32_t owner = arg->pipelineRelayServerRanks[stripe];
            const uint32_t channel = FindChannelIndex(arg, owner);
            if (channel >= arg->channelCount) {
                return CcuResult::CCU_E_INTERNAL;
            }
            CCU_CHK_RET(ccu::NotifyWait(
                arg->channels[channel], PipelineStageNotify(), PipelineStageMask(chunk)));
        }

        if (senderCount >= arg->pipelineRelayServerSize - 1) {
            if (senderSlot >= arg->pipelineRelayServerSize - 1) {
                continue;
            }
            ccu::Event finalDone;
            uint16_t finalMask = 0;
            for (uint32_t destination = 0; destination < arg->pipelineRelayServerSize; ++destination) {
                const uint32_t stripe =
                    senderSlot < destination ? senderSlot : senderSlot + 1;
                const uint32_t peer = arg->pipelineRelayServerRanks[destination];
                const uint32_t channel = FindChannelIndex(arg, peer);
                if (channel >= arg->channelCount) {
                    return CcuResult::CCU_E_INTERNAL;
                }
                const uint16_t mask = static_cast<uint16_t>(1U << channel);
                CCU_CHK_RET(WritePipelineChunk(arg, channel, peer, localOutput, localToken,
                    outputs, tokens, stripe, chunk, finalDone, mask));
                finalMask = static_cast<uint16_t>(finalMask | mask);
            }
            CCU_CHK_RET(ccu::EventWait(finalDone, finalMask));
            for (uint32_t destination = 0; destination < arg->pipelineRelayServerSize; ++destination) {
                const uint32_t stripe =
                    senderSlot < destination ? senderSlot : senderSlot + 1;
                const uint32_t channel = FindChannelIndex(arg, arg->pipelineRelayServerRanks[destination]);
                CCU_CHK_RET(ccu::NotifyRecord(arg->channels[channel],
                    PipelineFinalNotify(stripe), PipelineFinalMask(chunk)));
            }
        } else {
            for (uint32_t destination = 0; destination < arg->pipelineRelayServerSize; ++destination) {
                for (uint32_t stripe = 0; stripe < arg->pipelineRelayServerSize; ++stripe) {
                    if (stripe == destination) {
                        continue;
                    }
                    const uint32_t missingPosition =
                        stripe < destination ? stripe : stripe - 1;
                    if (missingPosition % senderCount != senderSlot) {
                        continue;
                    }
                    const uint32_t peer = arg->pipelineRelayServerRanks[destination];
                    const uint32_t channel = FindChannelIndex(arg, peer);
                    if (channel >= arg->channelCount) {
                        return CcuResult::CCU_E_INTERNAL;
                    }
                    const uint16_t mask = static_cast<uint16_t>(1U << channel);
                    ccu::Event serialDone;
                    CCU_CHK_RET(WritePipelineChunk(arg, channel, peer, localOutput, localToken,
                        outputs, tokens, stripe, chunk, serialDone, mask));
                    CCU_CHK_RET(ccu::EventWait(serialDone, mask));
                    CCU_CHK_RET(ccu::NotifyRecord(arg->channels[channel],
                        PipelineFinalNotify(stripe), PipelineFinalMask(chunk)));
                }
            }
        }
    }
    return CcuResult::CCU_SUCCESS;
}

bool IsMixedForestMapValid(const CcuKernelArgBroadcast *arg)
{
    if (!IsChannelMapValid(arg) || arg->forestTreeCount == 0 ||
        arg->forestTreeCount > F4_FOREST_MAX_TREE_COUNT ||
        arg->dieId >= F2_DIE_COUNT) {
        return false;
    }
    for (uint32_t tree = 0; tree < arg->forestTreeCount; ++tree) {
        const uint32_t parent = arg->forestParents[tree];
        const uint32_t childCount = arg->forestChildCounts[tree];
        const bool active = parent != F4_INVALID_RANK || childCount != 0;
        if (static_cast<bool>(arg->forestActive[tree]) != active ||
            childCount > MAX_RANK_SIZE ||
            arg->forestChildDieMasks[tree] >= (1U << F2_DIE_COUNT)) {
            return false;
        }
        if (parent != F4_INVALID_RANK &&
            (parent >= arg->rankSize ||
                FindChannelIndex(arg, parent) >= arg->channelCount)) {
            return false;
        }
        for (uint32_t child = 0; child < childCount; ++child) {
            const uint32_t peer = arg->forestChildren[tree][child];
            if (peer >= arg->rankSize ||
                FindChannelIndex(arg, peer) >= arg->channelCount) {
                return false;
            }
        }
        const bool localChildren =
            (arg->forestChildDieMasks[tree] & (1U << arg->dieId)) != 0;
        if (localChildren != (childCount != 0)) {
            return false;
        }
    }
    return true;
}

const char *ForestReadyTag(uint32_t tree, uint32_t die)
{
    static constexpr const char *tags[F4_FOREST_MAX_TREE_COUNT][F2_DIE_COUNT] = {
        {"f4_ready_t0_d0", "f4_ready_t0_d1"},
        {"f4_ready_t1_d0", "f4_ready_t1_d1"},
        {"f4_ready_t2_d0", "f4_ready_t2_d1"},
        {"f4_ready_t3_d0", "f4_ready_t3_d1"},
        {"f4_ready_t4_d0", "f4_ready_t4_d1"},
        {"f4_ready_t5_d0", "f4_ready_t5_d1"},
        {"f4_ready_t6_d0", "f4_ready_t6_d1"},
        {"f4_ready_t7_d0", "f4_ready_t7_d1"},
        {"f4_ready_t8_d0", "f4_ready_t8_d1"},
        {"f4_ready_t9_d0", "f4_ready_t9_d1"},
        {"f4_ready_t10_d0", "f4_ready_t10_d1"}
    };
    return tags[tree][die];
}

uint32_t ForestNotify(uint32_t tree)
{
    return tree + 1U;
}

uint16_t ForestMask(uint32_t chunk)
{
    return static_cast<uint16_t>(1U << chunk);
}

CcuResult SendForestChildren(const CcuKernelArgBroadcast *arg,
    const std::vector<ccu::Variable> &outputs,
    const std::vector<ccu::Variable> &tokens,
    const ccu::Variable &localOutput, const ccu::Variable &localToken,
    uint32_t tree, uint32_t chunk, ccu::Event &completed)
{
    const uint32_t childCount = arg->forestChildCounts[tree];
    if (childCount == 0) {
        return CcuResult::CCU_SUCCESS;
    }

    uint16_t completedMask = 0;
    for (uint32_t child = 0; child < childCount; ++child) {
        const uint32_t peer = arg->forestChildren[tree][child];
        const uint32_t channel = FindChannelIndex(arg, peer);
        const uint16_t mask = static_cast<uint16_t>(1U << channel);
        CCU_CHK_RET(WriteRange(arg, channel, peer, localOutput, localToken,
            outputs, tokens, arg->forestOffsets[tree][chunk],
            arg->forestSizes[tree][chunk], completed, mask));
        completedMask = static_cast<uint16_t>(completedMask | mask);
    }
    CCU_CHK_RET(ccu::EventWait(completed, completedMask));
    for (uint32_t child = 0; child < childCount; ++child) {
        const uint32_t channel =
            FindChannelIndex(arg, arg->forestChildren[tree][child]);
        CCU_CHK_RET(ccu::NotifyRecord(
            arg->channels[channel], ForestNotify(tree), ForestMask(chunk)));
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult SeedForestChunk(const CcuKernelArgBroadcast *arg,
    const std::vector<ccu::Variable> &outputs,
    const std::vector<ccu::Variable> &tokens,
    const ccu::Variable &localOutput, const ccu::Variable &localToken,
    uint32_t chunk, ccu::Event &completed)
{
    uint16_t completedMask = 0;
    for (uint32_t tree = 0; tree < arg->forestTreeCount; ++tree) {
        const uint32_t childCount = arg->forestChildCounts[tree];
        for (uint32_t child = 0; child < childCount; ++child) {
            const uint32_t peer = arg->forestChildren[tree][child];
            const uint32_t channel = FindChannelIndex(arg, peer);
            const uint16_t mask = static_cast<uint16_t>(1U << channel);
            if ((completedMask & mask) != 0) {
                return CcuResult::CCU_E_INTERNAL;
            }
            CCU_CHK_RET(WriteRange(arg, channel, peer, localOutput, localToken,
                outputs, tokens, arg->forestOffsets[tree][chunk],
                arg->forestSizes[tree][chunk], completed, mask));
            completedMask = static_cast<uint16_t>(completedMask | mask);
        }
    }
    if (completedMask != 0) {
        CCU_CHK_RET(ccu::EventWait(completed, completedMask));
    }
    for (uint32_t tree = 0; tree < arg->forestTreeCount; ++tree) {
        for (uint32_t child = 0; child < arg->forestChildCounts[tree]; ++child) {
            const uint32_t channel =
                FindChannelIndex(arg, arg->forestChildren[tree][child]);
            CCU_CHK_RET(ccu::NotifyRecord(
                arg->channels[channel], ForestNotify(tree), ForestMask(chunk)));
        }
    }
    return CcuResult::CCU_SUCCESS;
}

namespace {
constexpr uint32_t MESH_BUFFER_ID = 0;
constexpr uint32_t MESH_TOKEN_ID = 1;
constexpr uint32_t MESH_SCATTER_READY_BIT = 4;
constexpr uint32_t MESH_ALLGATHER_DONE_BIT = 5;
constexpr uint32_t MESH_NOTIFY_INDEX = 0;
constexpr uint64_t MESH_PHASE_SCATTER = 0;
constexpr uint64_t MESH_PHASE_ALLGATHER = 1;

struct MeshContext {
    const CcuKernelArgBroadcast *arg = nullptr;
    ccu::Variable localBuffer;
    ccu::Variable localToken;
    ccu::Variable normalSlice;
    ccu::Variable lastSlice;
    ccu::Variable phase;
    std::vector<ccu::Variable> peerBuffers;
    std::vector<ccu::Variable> peerTokens;
    ccu::Event completed;
};

bool IsMeshChannelMapValid(const CcuKernelArgBroadcast *arg)
{
    if (arg->rankSize <= 4 || arg->rankSize > MAX_RANK_SIZE ||
        arg->channelCount == 0 || arg->channelCount >= arg->rankSize ||
        arg->channelCount > 15) {
        return false;
    }
    for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
        const uint32_t peer = arg->peerRanks[channel];
        if (peer >= arg->rankSize || peer == arg->rankId) {
            return false;
        }
        for (uint32_t previous = 0; previous < channel; ++previous) {
            if (arg->peerRanks[previous] == peer) {
                return false;
            }
        }
    }
    return true;
}

CcuResult InitMeshContext(CcuKernelArg rawArg, MeshContext &ctx)
{
    ctx.arg = static_cast<CcuKernelArgBroadcast *>(rawArg);
    if (!IsMeshChannelMapValid(ctx.arg)) {
        return CcuResult::CCU_E_INTERNAL;
    }

    uint32_t argument = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.localBuffer, argument++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localToken, argument++));
    CCU_CHK_RET(ccu::LoadArg(ctx.normalSlice, argument++));
    CCU_CHK_RET(ccu::LoadArg(ctx.lastSlice, argument++));
    CCU_CHK_RET(ccu::LoadArg(ctx.phase, argument++));

    ctx.peerBuffers.resize(ctx.arg->channelCount);
    ctx.peerTokens.resize(ctx.arg->channelCount);
    for (uint32_t channel = 0; channel < ctx.arg->channelCount; ++channel) {
        ctx.peerBuffers[channel] = ccu::GetResByChannel<ccu::Variable>(
            ctx.arg->channels[channel], MESH_BUFFER_ID);
        ctx.peerTokens[channel] = ccu::GetResByChannel<ccu::Variable>(
            ctx.arg->channels[channel], MESH_TOKEN_ID);
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult PrepareMeshChannels(MeshContext &ctx)
{
    const uint16_t bufferBit = static_cast<uint16_t>(1U << MESH_BUFFER_ID);
    const uint16_t tokenBit = static_cast<uint16_t>(1U << MESH_TOKEN_ID);
    const uint16_t readyMask = static_cast<uint16_t>(bufferBit | tokenBit);
    for (uint32_t channel = 0; channel < ctx.arg->channelCount; ++channel) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channel],
            ctx.localBuffer, MESH_BUFFER_ID, MESH_NOTIFY_INDEX, bufferBit));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channel],
            ctx.localToken, MESH_TOKEN_ID, MESH_NOTIFY_INDEX, tokenBit));
    }
    for (uint32_t channel = 0; channel < ctx.arg->channelCount; ++channel) {
        CCU_CHK_RET(ccu::NotifyWait(
            ctx.arg->channels[channel], MESH_NOTIFY_INDEX, readyMask));
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult RunMeshScatter(MeshContext &ctx)
{
    const auto *arg = ctx.arg;
    const uint16_t readyBit =
        static_cast<uint16_t>(1U << MESH_SCATTER_READY_BIT);
    if (arg->rankId == arg->root) {
        ccu::Variable offset;
        offset = 0;
        for (uint32_t rank = 0; rank < arg->rankSize; ++rank) {
            if (rank != 0) {
                offset += ctx.normalSlice;
            }
            const uint32_t channel = FindChannelIndex(arg, rank);
            if (channel >= arg->channelCount) {
                continue;
            }

            ccu::LocalAddr source;
            source.addr = ctx.localBuffer;
            source.addr += offset;
            source.token = ctx.localToken;
            ccu::RemoteAddr destination;
            destination.addr = ctx.peerBuffers[channel];
            destination.addr += offset;
            destination.token = ctx.peerTokens[channel];
            const uint16_t mask = static_cast<uint16_t>(1U << channel);
            auto &size = rank + 1 == arg->rankSize ? ctx.lastSlice : ctx.normalSlice;
            CCU_CHK_RET(ccu::Write(arg->channels[channel], destination,
                source, size, ctx.completed, mask));
        }

        const uint16_t completedMask = static_cast<uint16_t>(
            (1U << arg->channelCount) - 1U);
        CCU_CHK_RET(ccu::EventWait(ctx.completed, completedMask));
        for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
            CCU_CHK_RET(ccu::NotifyRecord(
                arg->channels[channel], MESH_NOTIFY_INDEX, readyBit));
        }
    } else {
        const uint32_t rootChannel = FindChannelIndex(arg, arg->root);
        if (rootChannel < arg->channelCount) {
            CCU_CHK_RET(ccu::NotifyWait(
                arg->channels[rootChannel], MESH_NOTIFY_INDEX, readyBit));
        }
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult RunMeshAllgather(MeshContext &ctx)
{
    const auto *arg = ctx.arg;
    ccu::Variable offset;
    offset = 0;
    for (uint32_t rank = 0; rank < arg->rankId; ++rank) {
        offset += ctx.normalSlice;
    }
    auto &size = arg->rankId + 1 == arg->rankSize ?
        ctx.lastSlice : ctx.normalSlice;

    ccu::LocalAddr source;
    source.addr = ctx.localBuffer;
    source.addr += offset;
    source.token = ctx.localToken;
    uint16_t writtenMask = 0;
    for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
        // Root already holds the full buffer, so writing our slice to it during
        // AllGather is redundant; on 2x8 / 8+4 those writes waste scarce
        // cross-server bandwidth. Skip the data write but keep the doneBit
        // handshake below so the N-way barrier stays symmetric.
        if (arg->peerRanks[channel] == arg->root) {
            continue;
        }
        ccu::RemoteAddr destination;
        destination.addr = ctx.peerBuffers[channel];
        destination.addr += offset;
        destination.token = ctx.peerTokens[channel];
        const uint16_t mask = static_cast<uint16_t>(1U << channel);
        CCU_CHK_RET(ccu::Write(arg->channels[channel], destination,
            source, size, ctx.completed, mask));
        writtenMask = static_cast<uint16_t>(writtenMask | mask);
    }

    if (writtenMask != 0) {
        CCU_CHK_RET(ccu::EventWait(ctx.completed, writtenMask));
    }
    const uint16_t doneBit =
        static_cast<uint16_t>(1U << MESH_ALLGATHER_DONE_BIT);
    for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
        CCU_CHK_RET(ccu::NotifyRecord(
            arg->channels[channel], MESH_NOTIFY_INDEX, doneBit));
    }
    for (uint32_t channel = 0; channel < arg->channelCount; ++channel) {
        CCU_CHK_RET(ccu::NotifyWait(
            arg->channels[channel], MESH_NOTIFY_INDEX, doneBit));
    }
    return CcuResult::CCU_SUCCESS;
}
} // namespace

CcuResult CcuMeshKernel(CcuKernelArg rawArg)
{
    MeshContext ctx;
    CCU_CHK_RET(InitMeshContext(rawArg, ctx));
    CCU_IF(ctx.phase == MESH_PHASE_SCATTER)
    {
        CCU_CHK_RET(PrepareMeshChannels(ctx));
        CCU_CHK_RET(RunMeshScatter(ctx));
    }
    CCU_IF(ctx.phase == MESH_PHASE_ALLGATHER)
    {
        CCU_CHK_RET(RunMeshAllgather(ctx));
    }
    return CcuResult::CCU_SUCCESS;
}
CcuResult CcuMixedForestKernel(CcuKernelArg rawArg)
{
    auto *arg = static_cast<CcuKernelArgBroadcast *>(rawArg);
    if (!IsMixedForestMapValid(arg)) {
        return CcuResult::CCU_E_INTERNAL;
    }

    std::vector<ccu::Variable> outputs;
    std::vector<ccu::Variable> tokens;
    InitRemoteResources(arg, outputs, tokens);
    ccu::Variable localOutput;
    ccu::Variable localToken;
    CCU_CHK_RET(LoadLocalResources(localOutput, localToken));
    CCU_CHK_RET(PrepareAllChannels(arg, localOutput, localToken));

    ccu::Event completed;
    for (uint32_t chunk = 0; chunk < F4_FOREST_CHUNK_COUNT; ++chunk) {
        if (arg->rankId == arg->root) {
            CCU_CHK_RET(SeedForestChunk(arg, outputs, tokens,
                localOutput, localToken, chunk, completed));
            continue;
        }
        for (uint32_t tree = 0; tree < arg->forestTreeCount; ++tree) {
            const uint32_t parent = arg->forestParents[tree];
            if (parent != F4_INVALID_RANK) {
                const uint32_t parentChannel = FindChannelIndex(arg, parent);
                CCU_CHK_RET(ccu::NotifyWait(arg->channels[parentChannel],
                    ForestNotify(tree), ForestMask(chunk)));

                const uint32_t childDieMask =
                    arg->forestChildDieMasks[tree];
                for (uint32_t die = 0; die < F2_DIE_COUNT; ++die) {
                    if ((childDieMask & (1U << die)) != 0) {
                        CCU_CHK_RET(ccu::EventRecord(
                            ForestReadyTag(tree, die), ForestMask(chunk)));
                    }
                }
            }

            if (arg->forestChildCounts[tree] != 0) {
                CCU_CHK_RET(ccu::EventWait(
                    ForestReadyTag(tree, arg->dieId), ForestMask(chunk)));
                CCU_CHK_RET(SendForestChildren(arg, outputs, tokens,
                    localOutput, localToken, tree, chunk, completed));
            }
        }
    }
    return CcuResult::CCU_SUCCESS;
}
} // namespace ops_hccl
