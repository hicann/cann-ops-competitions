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
#include <cstdio>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>
#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>

#include "common.h"
#include "custom.h"
#include "ccu_kernel.h"
#include "exec_op.h"
#include "hccl.h"
#include "log.h"

namespace {
constexpr char RESOURCE_TAG[] = "hccl_custom_allgather_ccu_v143";
constexpr uint32_t CHANNEL_NOTIFY_NUM = 1;
constexpr uint32_t THREAD_NOTIFY_NUM = 2;
constexpr uint32_t MAX_NET_LAYER = 3;
constexpr uint32_t MAX_DIE_COUNT = 2;
constexpr uint32_t MAX_ACQUIRED_CHANNEL_COUNT =
    MAX_DIE_COUNT * (MAX_RANK_SIZE - 1);
constexpr bool ENABLE_RANK12_DUAL_RAIL = false;
constexpr uint64_t GROUP_FANOUT_MIN_BYTES = 1ULL << 20;
constexpr uint64_t RANK12_MESH_FIRST_BYTES = (400ULL << 20) + 4;
constexpr uint64_t FUSED_512_BYTES = 512ULL << 20;
constexpr uint64_t FINE_LOCAL_COPY_MIN_BYTES = 1ULL << 20;
constexpr uint32_t MESH_LINK_LOAD = 4;
constexpr uint32_t CLOS_LINK_LOAD = 1;
constexpr uint8_t CHANNEL_KIND_MESH = 0;
constexpr uint8_t CHANNEL_KIND_CLOS = 1;
constexpr uint32_t RANK12_NHR_STEP_COUNT = 4;

struct ChannelGroup {
    uint32_t dieId = 0;
    uint32_t linkLoad = 0;
    uint32_t closCount = 0;
    bool copyLocal = false;
    std::vector<ChannelHandle> channels;
    std::vector<uint32_t> remoteRanks;
    std::vector<uint8_t> channelKinds;
    std::vector<bool> alternateRails;
};

struct Rank16MixedConfig {
    bool enabled = false;
    uint32_t meshGroupIndex = MAX_DIE_COUNT;
    uint32_t closGroupIndex = MAX_DIE_COUNT;
    uint32_t relayPeerRank = MAX_RANK_SIZE;
};

struct Rank12SplitRelayConfig {
    bool enabled = false;
    bool smallServer = false;
    uint32_t meshGroupIndex = MAX_DIE_COUNT;
    uint32_t closGroupIndex = MAX_DIE_COUNT;
    uint32_t bridgeTargetRank = MAX_RANK_SIZE;
    uint32_t fanoutSourceRanks[2] = {
        MAX_RANK_SIZE,
        MAX_RANK_SIZE,
    };
    uint32_t fanoutSourceCount = 0;
};

uint32_t GetNhrStepCount(uint32_t rankSize)
{
    uint32_t stepCount = 0;
    while ((1U << stepCount) < rankSize) {
        ++stepCount;
    }
    return stepCount;
}

void GetNhrStepRanks(uint32_t rankSize, uint32_t rankId,
    uint32_t stepCount, uint32_t step, uint32_t &toRank,
    uint32_t &fromRank)
{
    uint32_t delta = 1U << (stepCount - 1 - step);
    toRank = (rankId + delta) % rankSize;
    fromRank = (rankId + rankSize - delta) % rankSize;
}

HcclResult ValidateParam(const OpParam &param)
{
    if (param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE ||
        param.myRank >= param.rankSize) {
        HCCL_ERROR("invalid rank info, rank=%u rankSize=%u", param.myRank, param.rankSize);
        return HCCL_E_PARA;
    }
    auto dataTypeIter = SIZE_TABLE.find(param.dataType);
    if (dataTypeIter == SIZE_TABLE.end() || dataTypeIter->second == 0) {
        HCCL_ERROR("unsupported data type=%d", static_cast<int32_t>(param.dataType));
        return HCCL_E_PARA;
    }
    if (param.count > UINT64_MAX / dataTypeIter->second) {
        HCCL_ERROR("input size overflow, count=%llu elementSize=%u",
            static_cast<unsigned long long>(param.count),
            dataTypeIter->second);
        return HCCL_E_PARA;
    }
    uint64_t inputBytes = param.count * dataTypeIter->second;
    if (param.rankSize != 0 && inputBytes > UINT64_MAX / param.rankSize) {
        HCCL_ERROR("output size overflow, inputBytes=%llu rankSize=%u",
            static_cast<unsigned long long>(inputBytes), param.rankSize);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult FindLink(HcclComm comm, uint32_t localRank, uint32_t remoteRank,
    CommLink &selectedLink, uint32_t &selectedLayer)
{
    for (uint32_t layer = 0; layer < MAX_NET_LAYER; ++layer) {
        CommLink *links = nullptr;
        uint32_t linkCount = 0;
        HcclResult ret =
            HcclRankGraphGetLinks(comm, layer, localRank, remoteRank, &links, &linkCount);
        if (ret != HCCL_SUCCESS) {
            continue;
        }

        for (CommProtocol wanted :
            {CommProtocol::COMM_PROTOCOL_UBC_CTP, CommProtocol::COMM_PROTOCOL_UBC_TP}) {
            for (uint32_t index = 0; index < linkCount; ++index) {
                if (links[index].linkAttr.linkProtocol == wanted) {
                    selectedLink = links[index];
                    selectedLayer = layer;
                    return HCCL_SUCCESS;
                }
            }
        }
    }
    HCCL_ERROR("no CCU link found, localRank=%u remoteRank=%u", localRank, remoteRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult GetEndpointDie(HcclComm comm, uint32_t rank,
    EndpointDesc &endpoint, uint32_t &dieId)
{
    EndpointAttrDieId endpointDie{};
    CHK_RET(HcclRankGraphGetEndpointInfo(comm, rank, &endpoint,
        ENDPOINT_ATTR_DIE_ID, sizeof(endpointDie), &endpointDie));
    dieId = endpointDie;
    return HCCL_SUCCESS;
}

uint32_t GetEndpointBwCoeff(HcclComm comm, uint32_t rank,
    EndpointDesc &endpoint)
{
    EndpointAttrBwCoeff bwCoeff{};
    HcclResult ret = HcclRankGraphGetEndpointInfo(comm, rank,
        &endpoint, ENDPOINT_ATTR_BW_COEFF,
        sizeof(bwCoeff), &bwCoeff);
    return ret == HCCL_SUCCESS && bwCoeff != 0 ?
        bwCoeff : 1;
}

HcclResult FindDualLinks(HcclComm comm, uint32_t localRank,
    uint32_t remoteRank, uint32_t layer,
    std::array<CommLink, MAX_DIE_COUNT> &selectedLinks,
    std::array<uint32_t, MAX_DIE_COUNT> &selectedDies)
{
    CommLink *links = nullptr;
    uint32_t linkCount = 0;
    CHK_RET(HcclRankGraphGetLinks(
        comm, layer, localRank, remoteRank, &links, &linkCount));

    for (CommProtocol wanted :
        {CommProtocol::COMM_PROTOCOL_UBC_CTP,
            CommProtocol::COMM_PROTOCOL_UBC_TP}) {
        uint32_t selectedCount = 0;
        for (uint32_t index = 0; index < linkCount; ++index) {
            if (links[index].linkAttr.linkProtocol != wanted) {
                continue;
            }
            uint32_t dieId = 0;
            if (GetEndpointDie(comm, localRank,
                    links[index].srcEndpointDesc, dieId) !=
                HCCL_SUCCESS) {
                continue;
            }
            bool duplicateDie = false;
            for (uint32_t selected = 0;
                 selected < selectedCount; ++selected) {
                duplicateDie |= selectedDies[selected] == dieId;
            }
            if (duplicateDie || selectedCount == MAX_DIE_COUNT) {
                continue;
            }
            selectedLinks[selectedCount] = links[index];
            selectedDies[selectedCount] = dieId;
            ++selectedCount;
        }
        if (selectedCount == MAX_DIE_COUNT) {
            if (selectedDies[0] > selectedDies[1]) {
                std::swap(selectedDies[0], selectedDies[1]);
                std::swap(selectedLinks[0], selectedLinks[1]);
            }
            return HCCL_SUCCESS;
        }
    }
    return HCCL_E_NOT_FOUND;
}

HcclResult BuildChannelPriorities(HcclComm comm, const OpParam &param,
    const std::array<uint32_t, MAX_ACQUIRED_CHANNEL_COUNT> &remoteRanks,
    const std::array<uint32_t, MAX_ACQUIRED_CHANNEL_COUNT> &linkLayers,
    uint32_t channelCount,
    std::array<uint32_t, MAX_RANK_SIZE> &priorities)
{
    priorities.fill(static_cast<uint32_t>(MAX_RANK_SIZE * 2));

    uint32_t *localRanks = nullptr;
    uint32_t localRankCount = 0;
    CHK_RET(HcclRankGraphGetRanksByLayer(
        comm, 0, &localRanks, &localRankCount));
    if (localRanks == nullptr || localRankCount == 0 ||
        localRankCount > MAX_RANK_SIZE) {
        HCCL_ERROR("invalid layer-0 rank group, count=%u", localRankCount);
        return HCCL_E_INTERNAL;
    }

    uint32_t localRankIndex = localRankCount;
    for (uint32_t index = 0; index < localRankCount; ++index) {
        if (localRanks[index] == param.myRank) {
            localRankIndex = index;
            break;
        }
    }
    if (localRankIndex == localRankCount) {
        HCCL_ERROR("local rank %u not found in layer-0 group", param.myRank);
        return HCCL_E_INTERNAL;
    }

    std::vector<uint32_t> closRanks;
    for (uint32_t index = 0; index < channelCount; ++index) {
        if (linkLayers[index] != 0 &&
            std::find(closRanks.begin(), closRanks.end(),
                remoteRanks[index]) == closRanks.end()) {
            closRanks.push_back(remoteRanks[index]);
        }
    }
    std::sort(closRanks.begin(), closRanks.end());

    uint32_t rotationBase = localRankIndex;
    if (localRankCount == 1 && !closRanks.empty()) {
        uint32_t *globalRanks = nullptr;
        uint32_t globalRankCount = 0;
        CHK_RET(HcclRankGraphGetRanksByLayer(
            comm, 1, &globalRanks, &globalRankCount));
        if (globalRanks == nullptr || globalRankCount == 0 ||
            globalRankCount > MAX_RANK_SIZE) {
            HCCL_ERROR("invalid layer-1 rank group, count=%u",
                globalRankCount);
            return HCCL_E_INTERNAL;
        }
        bool rankFound = false;
        for (uint32_t index = 0; index < globalRankCount; ++index) {
            if (globalRanks[index] == param.myRank) {
                rotationBase = index;
                rankFound = true;
                break;
            }
        }
        if (!rankFound) {
            HCCL_ERROR("local rank %u not found in layer-1 group",
                param.myRank);
            return HCCL_E_INTERNAL;
        }
    }

    for (uint32_t order = 0; order < closRanks.size(); ++order) {
        uint32_t rotatedIndex =
            (rotationBase + order) % closRanks.size();
        priorities[closRanks[rotatedIndex]] = order;
    }

    uint32_t nextPriority = static_cast<uint32_t>(closRanks.size());
    for (uint32_t step = 1; step < localRankCount; ++step) {
        uint32_t remoteRank =
            localRanks[(localRankIndex + step) % localRankCount];
        if (remoteRank < MAX_RANK_SIZE && remoteRank != param.myRank) {
            priorities[remoteRank] = nextPriority++;
        }
    }
    for (uint32_t index = 0; index < channelCount; ++index) {
        uint32_t remoteRank = remoteRanks[index];
        if (priorities[remoteRank] >= MAX_RANK_SIZE * 2) {
            priorities[remoteRank] = nextPriority++;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AcquireChannels(HcclComm comm, const OpParam &param,
    uint64_t inputBytes, std::vector<ChannelGroup> &groups,
    bool &rank12DualRail, uint32_t &primaryRailWeight,
    uint32_t &alternateRailWeight)
{
    const uint32_t primaryChannelCount = param.rankSize - 1;
    std::array<CommLink, MAX_RANK_SIZE> primaryLinks{};
    std::array<uint32_t, MAX_RANK_SIZE> primaryLayers{};
    std::array<uint32_t, MAX_RANK_SIZE> primaryRemoteRanks{};
    std::array<std::array<CommLink, MAX_DIE_COUNT>,
        MAX_RANK_SIZE> dualLinks{};
    std::array<std::array<uint32_t, MAX_DIE_COUNT>,
        MAX_RANK_SIZE> dualDies{};
    std::array<std::array<uint32_t, MAX_DIE_COUNT>,
        MAX_RANK_SIZE> dualBwCoefficients{};
    bool dualClosAvailable =
        ENABLE_RANK12_DUAL_RAIL &&
        param.rankSize == 12 &&
        inputBytes >= GROUP_FANOUT_MIN_BYTES;

    uint32_t primaryIndex = 0;
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }

        CommLink link;
        uint32_t linkLayer = 0;
        CHK_RET(FindLink(
            comm, param.myRank, remoteRank, link, linkLayer));
        primaryLinks[primaryIndex] = link;
        primaryLayers[primaryIndex] = linkLayer;
        primaryRemoteRanks[primaryIndex] = remoteRank;
        if (dualClosAvailable && linkLayer != 0 &&
            FindDualLinks(comm, param.myRank, remoteRank,
                linkLayer, dualLinks[primaryIndex],
                dualDies[primaryIndex]) != HCCL_SUCCESS) {
            dualClosAvailable = false;
        } else if (dualClosAvailable && linkLayer != 0) {
            for (uint32_t rail = 0; rail < MAX_DIE_COUNT; ++rail) {
                dualBwCoefficients[primaryIndex][rail] =
                    GetEndpointBwCoeff(comm, param.myRank,
                        dualLinks[primaryIndex][rail].
                            srcEndpointDesc);
            }
        }
        ++primaryIndex;
    }
    if (primaryIndex != primaryChannelCount) {
        HCCL_ERROR("unexpected channel count=%u expected=%u",
            primaryIndex, primaryChannelCount);
        return HCCL_E_INTERNAL;
    }

    uint32_t closPeerCount = 0;
    for (uint32_t index = 0; index < primaryChannelCount; ++index) {
        closPeerCount += primaryLayers[index] != 0 ? 1U : 0U;
    }
    dualClosAvailable =
        dualClosAvailable && closPeerCount != 0;
    uint64_t primaryWeightSum = 0;
    uint64_t alternateWeightSum = 0;
    if (dualClosAvailable) {
        for (uint32_t index = 0;
             index < primaryChannelCount; ++index) {
            if (primaryLayers[index] == 0) {
                continue;
            }
            primaryWeightSum +=
                dualBwCoefficients[index][0];
            alternateWeightSum +=
                dualBwCoefficients[index][1];
        }
        dualClosAvailable =
            primaryWeightSum != 0 &&
            alternateWeightSum != 0 &&
            primaryWeightSum <= UINT32_MAX &&
            alternateWeightSum <= UINT32_MAX;
    }
    uint32_t channelCount =
        primaryChannelCount +
        (dualClosAvailable ? closPeerCount : 0);
    std::array<HcclChannelDesc, MAX_ACQUIRED_CHANNEL_COUNT> descs{};
    std::array<EndpointAttrDieId, MAX_ACQUIRED_CHANNEL_COUNT> dieIds{};
    std::array<uint32_t, MAX_ACQUIRED_CHANNEL_COUNT> remoteRanks{};
    std::array<uint32_t, MAX_ACQUIRED_CHANNEL_COUNT> linkLayers{};
    std::array<bool, MAX_ACQUIRED_CHANNEL_COUNT> alternateRails{};
    CHK_RET(HcclChannelDescInit(descs.data(), channelCount));

    uint32_t channelIndex = 0;
    for (uint32_t index = 0; index < primaryChannelCount; ++index) {
        auto addChannel = [&](const CommLink &link, uint32_t dieId,
                              bool alternateRail) {
            HcclChannelDesc &desc = descs[channelIndex];
            desc.remoteRank = primaryRemoteRanks[index];
            desc.notifyNum = CHANNEL_NOTIFY_NUM;
            desc.channelProtocol = link.linkAttr.linkProtocol;
            desc.localEndpoint = link.srcEndpointDesc;
            desc.remoteEndpoint = link.dstEndpointDesc;
            dieIds[channelIndex] = dieId;
            remoteRanks[channelIndex] = primaryRemoteRanks[index];
            linkLayers[channelIndex] = primaryLayers[index];
            alternateRails[channelIndex] = alternateRail;
            ++channelIndex;
        };

        if (dualClosAvailable && primaryLayers[index] != 0) {
            addChannel(dualLinks[index][0], dualDies[index][0], false);
            addChannel(dualLinks[index][1], dualDies[index][1], true);
        } else {
            uint32_t dieId = 0;
            CHK_RET(GetEndpointDie(comm, param.myRank,
                primaryLinks[index].srcEndpointDesc, dieId));
            addChannel(primaryLinks[index], dieId, false);
        }
    }
    if (channelIndex != channelCount) {
        HCCL_ERROR("unexpected expanded channel count=%u expected=%u",
            channelIndex, channelCount);
        return HCCL_E_INTERNAL;
    }

    std::array<ChannelHandle, MAX_ACQUIRED_CHANNEL_COUNT> channels{};
    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU,
        descs.data(), channelCount, channels.data()));

    std::array<uint32_t, MAX_RANK_SIZE> priorities{};
    CHK_RET(BuildChannelPriorities(comm, param, remoteRanks, linkLayers,
        channelCount, priorities));
    std::array<uint32_t, MAX_ACQUIRED_CHANNEL_COUNT> channelOrder{};
    for (uint32_t index = 0; index < channelCount; ++index) {
        channelOrder[index] = index;
    }
    std::stable_sort(channelOrder.begin(),
        channelOrder.begin() + channelCount,
        [&priorities, &remoteRanks](uint32_t left, uint32_t right) {
            return priorities[remoteRanks[left]] <
                priorities[remoteRanks[right]];
        });

    std::map<uint32_t, ChannelGroup> groupsByDie;
    for (uint32_t order = 0; order < channelCount; ++order) {
        uint32_t index = channelOrder[order];
        ChannelGroup &group = groupsByDie[dieIds[index]];
        group.dieId = dieIds[index];
        group.channels.push_back(channels[index]);
        group.remoteRanks.push_back(remoteRanks[index]);
        group.channelKinds.push_back(linkLayers[index] == 0 ?
            CHANNEL_KIND_MESH : CHANNEL_KIND_CLOS);
        group.alternateRails.push_back(alternateRails[index]);
        if (linkLayers[index] != 0) {
            ++group.closCount;
        }
        group.linkLoad += linkLayers[index] == 0 ?
            MESH_LINK_LOAD : CLOS_LINK_LOAD;
    }

    if (groupsByDie.empty() || groupsByDie.size() > MAX_DIE_COUNT) {
        HCCL_ERROR("unsupported CCU die count=%lu", groupsByDie.size());
        return HCCL_E_NOT_SUPPORT;
    }
    groups.reserve(groupsByDie.size());
    for (auto &item : groupsByDie) {
        if (item.second.channels.size() > MAX_RANK_SIZE - 1) {
            HCCL_ERROR("too many channels on CCU die %u, count=%lu",
                item.first, item.second.channels.size());
            return HCCL_E_NOT_SUPPORT;
        }
        groups.push_back(std::move(item.second));
    }
    rank12DualRail = dualClosAvailable;
    primaryRailWeight = dualClosAvailable ?
        static_cast<uint32_t>(primaryWeightSum) : 0;
    alternateRailWeight = dualClosAvailable ?
        static_cast<uint32_t>(alternateWeightSum) : 0;
    return HCCL_SUCCESS;
}

HcclResult TryAcquireRank12NhrChannels(HcclComm comm,
    const OpParam &param, uint64_t inputBytes,
    std::vector<ChannelGroup> &groups, bool &enabled)
{
    enabled = false;
    if (param.rankSize != 12 ||
        inputBytes < GROUP_FANOUT_MIN_BYTES ||
        inputBytes > FUSED_512_BYTES ||
        GetNhrStepCount(param.rankSize) !=
            RANK12_NHR_STEP_COUNT) {
        return HCCL_SUCCESS;
    }

    std::vector<uint32_t> partnerRanks;
    for (uint32_t step = 0; step < RANK12_NHR_STEP_COUNT;
         ++step) {
        uint32_t toRank = 0;
        uint32_t fromRank = 0;
        GetNhrStepRanks(param.rankSize, param.myRank,
            RANK12_NHR_STEP_COUNT, step, toRank, fromRank);
        for (uint32_t partner : {toRank, fromRank}) {
            if (partner != param.myRank &&
                std::find(partnerRanks.begin(), partnerRanks.end(),
                    partner) == partnerRanks.end()) {
                partnerRanks.push_back(partner);
            }
        }
    }
    if (partnerRanks.empty() ||
        partnerRanks.size() > MAX_RANK_SIZE - 1) {
        return HCCL_SUCCESS;
    }

    std::array<std::array<CommLink, MAX_DIE_COUNT>,
        MAX_RANK_SIZE> partnerLinks{};
    std::array<std::array<uint32_t, MAX_DIE_COUNT>,
        MAX_RANK_SIZE> partnerDies{};
    std::array<uint32_t, MAX_RANK_SIZE> partnerLayers{};
    for (uint32_t index = 0; index < partnerRanks.size();
         ++index) {
        CommLink primaryLink;
        uint32_t layer = 0;
        if (FindLink(comm, param.myRank, partnerRanks[index],
                primaryLink, layer) != HCCL_SUCCESS ||
            FindDualLinks(comm, param.myRank,
                partnerRanks[index], layer,
                partnerLinks[index],
                partnerDies[index]) != HCCL_SUCCESS) {
            return HCCL_SUCCESS;
        }
        partnerLayers[index] = layer;
    }

    uint32_t channelCount = static_cast<uint32_t>(
        partnerRanks.size() * MAX_DIE_COUNT);
    std::array<HcclChannelDesc, MAX_ACQUIRED_CHANNEL_COUNT>
        descs{};
    CHK_RET(HcclChannelDescInit(descs.data(), channelCount));
    std::array<uint32_t, MAX_ACQUIRED_CHANNEL_COUNT> dies{};
    std::array<uint32_t, MAX_ACQUIRED_CHANNEL_COUNT> remotes{};
    std::array<uint32_t, MAX_ACQUIRED_CHANNEL_COUNT> layers{};
    uint32_t channelIndex = 0;
    for (uint32_t partnerIndex = 0;
         partnerIndex < partnerRanks.size(); ++partnerIndex) {
        for (uint32_t rail = 0; rail < MAX_DIE_COUNT; ++rail) {
            HcclChannelDesc &desc = descs[channelIndex];
            const CommLink &link =
                partnerLinks[partnerIndex][rail];
            desc.remoteRank = partnerRanks[partnerIndex];
            desc.notifyNum = CHANNEL_NOTIFY_NUM;
            desc.channelProtocol = link.linkAttr.linkProtocol;
            desc.localEndpoint = link.srcEndpointDesc;
            desc.remoteEndpoint = link.dstEndpointDesc;
            dies[channelIndex] =
                partnerDies[partnerIndex][rail];
            remotes[channelIndex] =
                partnerRanks[partnerIndex];
            layers[channelIndex] =
                partnerLayers[partnerIndex];
            ++channelIndex;
        }
    }

    std::array<ChannelHandle, MAX_ACQUIRED_CHANNEL_COUNT>
        channels{};
    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU,
        descs.data(), channelCount, channels.data()));

    std::map<uint32_t, ChannelGroup> groupsByDie;
    for (uint32_t index = 0; index < channelCount; ++index) {
        ChannelGroup &group = groupsByDie[dies[index]];
        group.dieId = dies[index];
        group.channels.push_back(channels[index]);
        group.remoteRanks.push_back(remotes[index]);
        group.channelKinds.push_back(layers[index] == 0 ?
            CHANNEL_KIND_MESH : CHANNEL_KIND_CLOS);
        group.alternateRails.push_back(false);
        group.closCount += layers[index] != 0 ? 1U : 0U;
        group.linkLoad += GetEndpointBwCoeff(comm, param.myRank,
            descs[index].localEndpoint);
    }
    if (groupsByDie.size() != MAX_DIE_COUNT) {
        HCCL_ERROR("rank12 NHR requires two CCU dies");
        return HCCL_E_NOT_SUPPORT;
    }
    for (auto &item : groupsByDie) {
        if (item.second.channels.size() != partnerRanks.size()) {
            HCCL_ERROR("incomplete rank12 NHR channel set, die=%u "
                       "count=%lu expected=%lu",
                item.first, item.second.channels.size(),
                partnerRanks.size());
            return HCCL_E_INTERNAL;
        }
        groups.push_back(std::move(item.second));
    }
    enabled = true;
    return HCCL_SUCCESS;
}

Rank16MixedConfig DetectRank16MixedTopology(const OpParam &param,
    uint64_t inputBytes, const std::vector<ChannelGroup> &groups)
{
    Rank16MixedConfig config;
    if (param.rankSize != 16 ||
        inputBytes < GROUP_FANOUT_MIN_BYTES ||
        groups.size() != MAX_DIE_COUNT) {
        return config;
    }

    for (uint32_t groupIndex = 0; groupIndex < groups.size();
         ++groupIndex) {
        const ChannelGroup &group = groups[groupIndex];
        bool allMesh =
            group.channels.size() == 7 && group.closCount == 0;
        bool allClos =
            group.channels.size() == 8 && group.closCount == 8;
        for (uint8_t channelKind : group.channelKinds) {
            allMesh = allMesh &&
                channelKind == CHANNEL_KIND_MESH;
            allClos = allClos &&
                channelKind == CHANNEL_KIND_CLOS;
        }
        if (allMesh) {
            if (config.meshGroupIndex != MAX_DIE_COUNT) {
                return Rank16MixedConfig{};
            }
            config.meshGroupIndex = groupIndex;
        }
        if (allClos) {
            if (config.closGroupIndex != MAX_DIE_COUNT) {
                return Rank16MixedConfig{};
            }
            config.closGroupIndex = groupIndex;
        }
    }
    if (config.meshGroupIndex >= groups.size() ||
        config.closGroupIndex >= groups.size() ||
        config.meshGroupIndex == config.closGroupIndex) {
        return Rank16MixedConfig{};
    }

    std::vector<uint32_t> localRanks =
        groups[config.meshGroupIndex].remoteRanks;
    localRanks.push_back(param.myRank);
    std::sort(localRanks.begin(), localRanks.end());
    if (localRanks.size() != 8 ||
        std::adjacent_find(localRanks.begin(), localRanks.end()) !=
            localRanks.end()) {
        return Rank16MixedConfig{};
    }
    auto localRankIter =
        std::find(localRanks.begin(), localRanks.end(), param.myRank);
    if (localRankIter == localRanks.end()) {
        return Rank16MixedConfig{};
    }

    std::vector<uint32_t> remoteRanks =
        groups[config.closGroupIndex].remoteRanks;
    std::sort(remoteRanks.begin(), remoteRanks.end());
    if (remoteRanks.size() != 8 ||
        std::adjacent_find(remoteRanks.begin(), remoteRanks.end()) !=
            remoteRanks.end()) {
        return Rank16MixedConfig{};
    }
    uint32_t localIndex = static_cast<uint32_t>(
        std::distance(localRanks.begin(), localRankIter));
    config.relayPeerRank = remoteRanks[localIndex];
    if (config.relayPeerRank >= param.rankSize ||
        config.relayPeerRank == param.myRank) {
        return Rank16MixedConfig{};
    }
    config.enabled = true;
    return config;
}

Rank12SplitRelayConfig DetectRank12SplitRelayTopology(
    const OpParam &param, uint64_t inputBytes,
    const std::vector<ChannelGroup> &groups)
{
    Rank12SplitRelayConfig config;
    if (param.rankSize != 12 ||
        inputBytes < GROUP_FANOUT_MIN_BYTES ||
        groups.size() != MAX_DIE_COUNT) {
        return config;
    }

    for (uint32_t groupIndex = 0; groupIndex < groups.size();
         ++groupIndex) {
        const ChannelGroup &group = groups[groupIndex];
        bool allMesh =
            (group.channels.size() == 3 ||
                group.channels.size() == 7) &&
            group.closCount == 0;
        bool allClos =
            (group.channels.size() == 4 ||
                group.channels.size() == 8) &&
            group.closCount == group.channels.size();
        for (uint8_t channelKind : group.channelKinds) {
            allMesh = allMesh &&
                channelKind == CHANNEL_KIND_MESH;
            allClos = allClos &&
                channelKind == CHANNEL_KIND_CLOS;
        }
        if (allMesh) {
            if (config.meshGroupIndex != MAX_DIE_COUNT) {
                return Rank12SplitRelayConfig{};
            }
            config.meshGroupIndex = groupIndex;
        }
        if (allClos) {
            if (config.closGroupIndex != MAX_DIE_COUNT) {
                return Rank12SplitRelayConfig{};
            }
            config.closGroupIndex = groupIndex;
        }
    }
    if (config.meshGroupIndex >= groups.size() ||
        config.closGroupIndex >= groups.size() ||
        config.meshGroupIndex == config.closGroupIndex) {
        return Rank12SplitRelayConfig{};
    }

    std::vector<uint32_t> localRanks =
        groups[config.meshGroupIndex].remoteRanks;
    localRanks.push_back(param.myRank);
    std::sort(localRanks.begin(), localRanks.end());
    std::vector<uint32_t> remoteRanks =
        groups[config.closGroupIndex].remoteRanks;
    std::sort(remoteRanks.begin(), remoteRanks.end());
    if (std::adjacent_find(localRanks.begin(), localRanks.end()) !=
            localRanks.end() ||
        std::adjacent_find(remoteRanks.begin(), remoteRanks.end()) !=
            remoteRanks.end()) {
        return Rank12SplitRelayConfig{};
    }

    const bool largeServer =
        localRanks.size() == 8 &&
        remoteRanks.size() == 4;
    const bool smallServer =
        localRanks.size() == 4 &&
        remoteRanks.size() == 8;
    if (!largeServer && !smallServer) {
        return Rank12SplitRelayConfig{};
    }
    for (uint32_t index = 0; index < localRanks.size(); ++index) {
        uint32_t expected = largeServer ? index : index + 8;
        if (localRanks[index] != expected) {
            return Rank12SplitRelayConfig{};
        }
    }
    for (uint32_t index = 0; index < remoteRanks.size(); ++index) {
        uint32_t expected = largeServer ? index + 8 : index;
        if (remoteRanks[index] != expected) {
            return Rank12SplitRelayConfig{};
        }
    }

    auto localRankIter = std::find(
        localRanks.begin(), localRanks.end(), param.myRank);
    if (localRankIter == localRanks.end()) {
        return Rank12SplitRelayConfig{};
    }
    uint32_t localIndex = static_cast<uint32_t>(
        std::distance(localRanks.begin(), localRankIter));
    config.smallServer = smallServer;
    config.bridgeTargetRank =
        smallServer ? remoteRanks[localIndex] :
            remoteRanks[localIndex % remoteRanks.size()];
    if (smallServer) {
        config.fanoutSourceRanks[0] =
            remoteRanks[localIndex];
        config.fanoutSourceRanks[1] =
            remoteRanks[localIndex + localRanks.size()];
        config.fanoutSourceCount = 2;
    } else if (localIndex < remoteRanks.size()) {
        config.fanoutSourceRanks[0] =
            remoteRanks[localIndex];
        config.fanoutSourceCount = 1;
    }
    if (config.bridgeTargetRank >= param.rankSize ||
        config.bridgeTargetRank == param.myRank) {
        return Rank12SplitRelayConfig{};
    }
    config.enabled = true;
    return config;
}

HcclResult RegisterKernels(HcclComm comm, const OpParam &param,
    const std::vector<ChannelGroup> &groups, bool enableRank16Fanout,
    bool preferRank12MeshFirst, bool fuseTwoSlices,
    bool useFineLocalCopy, bool rank12DualRail,
    bool useRank12DirectLocalCopy, bool useRank12DirectReuse,
    bool rank12Nhr,
    const Rank16MixedConfig &rank16Mixed,
    const Rank12SplitRelayConfig &rank12Split,
    AlgResourceCtx &resource)
{
    CcuInsHandle insHandle{0};
    uint32_t insCount = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insCount));
    if (insCount != 1) {
        HCCL_ERROR("unexpected CCU instance count=%u", insCount);
        return HCCL_E_INTERNAL;
    }

    CcuResult ccuRet = HcommCcuKernelRegisterStart(insHandle);
    if (ccuRet != CCU_SUCCESS) {
        HCCL_ERROR("CCU kernel register start failed, ret=%d", ccuRet);
        return ConvertCcuResult(ccuRet);
    }

    resource.kernelCount = static_cast<uint32_t>(groups.size());
    for (uint32_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
        const ChannelGroup &group = groups[groupIndex];
        CcuKernelInfo kernelInfo;
        int nameRet = std::snprintf(kernelInfo.kernelFuncName,
            sizeof(kernelInfo.kernelFuncName), "CustomAllGatherMeshKernelDie%u",
            group.dieId);
        if (nameRet < 0 ||
            static_cast<uint32_t>(nameRet) >= sizeof(kernelInfo.kernelFuncName)) {
            HCCL_ERROR("failed to set CCU kernel name");
            (void)HcommCcuKernelRegisterEnd(insHandle);
            return HCCL_E_INTERNAL;
        }
        kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuKernel);

        auto kernelArg = std::make_shared<ops_hccl::CcuKernelArgAllGather>();
        kernelArg->rankSize = param.rankSize;
        kernelArg->rankId = param.myRank;
        kernelArg->copyLocal =
            !rank12Nhr && group.copyLocal;
        kernelArg->useRank12Nhr = rank12Nhr;
        kernelArg->nhrAxisId = groupIndex;
        kernelArg->useRank12SplitRelay = rank12Split.enabled;
        kernelArg->isRank12MeshKernel =
            rank12Split.enabled &&
            groupIndex == rank12Split.meshGroupIndex;
        kernelArg->rank12SmallServer =
            rank12Split.smallServer;
        kernelArg->rank12FanoutSourceCount =
            rank12Split.fanoutSourceCount;
        kernelArg->useRank16MixedRelay = rank16Mixed.enabled;
        kernelArg->isRank16MeshKernel =
            rank16Mixed.enabled &&
            groupIndex == rank16Mixed.meshGroupIndex;
        kernelArg->relayPeerRank = rank16Mixed.relayPeerRank;
        kernelArg->useGroupBroadcast =
            !rank12Nhr && (rank16Mixed.enabled ?
                !kernelArg->isRank16MeshKernel :
                (!rank12Split.enabled &&
                    enableRank16Fanout &&
                    group.closCount >= 2));
        kernelArg->directMeshFirst =
            !rank12Nhr && !rank12Split.enabled &&
            !rank16Mixed.enabled &&
            preferRank12MeshFirst && groups.size() > 1 &&
            groupIndex == 0;
        kernelArg->fuseTwoSlices =
            !rank12Nhr && !rank12Split.enabled &&
            !rank16Mixed.enabled && fuseTwoSlices;
        kernelArg->useRank12DualRail =
            !rank12Nhr && rank12DualRail;
        kernelArg->useDirectLocalCopy =
            !rank12Nhr && useRank12DirectLocalCopy &&
            group.copyLocal;
        kernelArg->useFineLocalCopy =
            !rank12Nhr && (rank12Split.enabled ?
                !kernelArg->isRank12MeshKernel :
                (!rank16Mixed.enabled &&
                    useFineLocalCopy && group.copyLocal &&
                    !kernelArg->useDirectLocalCopy));
        kernelArg->useRank4DirectReuse =
            param.rankSize == 4 &&
            resource.cachedInputBytes > MAX_DATA_SIZE &&
            !kernelArg->fuseTwoSlices;
        kernelArg->useRank12DirectReuse =
            useRank12DirectReuse;
        kernelArg->rank4FirstOutputOffset =
            static_cast<uint64_t>(param.myRank) *
            resource.cachedInputBytes;
        kernelArg->rank12FirstOutputOffset =
            static_cast<uint64_t>(param.myRank) *
            resource.cachedInputBytes;
        if (rank12Nhr) {
            resource.taskArgCounts[groupIndex] =
                AlgResourceCtx::RANK12_NHR_TASK_ARG_COUNT;
        } else if (rank12DualRail) {
            if (kernelArg->fuseTwoSlices) {
                resource.taskArgCounts[groupIndex] =
                    group.copyLocal ?
                        AlgResourceCtx::
                            DUAL_RAIL_FUSED_FULL_TASK_ARG_COUNT :
                        AlgResourceCtx::
                            DUAL_RAIL_FUSED_BASE_TASK_ARG_COUNT;
            } else {
                resource.taskArgCounts[groupIndex] =
                    group.copyLocal ?
                        AlgResourceCtx::
                            DUAL_RAIL_FULL_TASK_ARG_COUNT :
                        AlgResourceCtx::
                            DUAL_RAIL_BASE_TASK_ARG_COUNT;
            }
        } else if (kernelArg->useDirectLocalCopy) {
            resource.taskArgCounts[groupIndex] =
                kernelArg->fuseTwoSlices ?
                    AlgResourceCtx::
                        FUSED_DIRECT_LOCAL_COPY_TASK_ARG_COUNT :
                    AlgResourceCtx::
                        DIRECT_LOCAL_COPY_TASK_ARG_COUNT;
        } else if (rank12Split.enabled) {
            resource.taskArgCounts[groupIndex] =
                kernelArg->isRank12MeshKernel ?
                    AlgResourceCtx::RANK12_MESH_TASK_ARG_COUNT :
                    AlgResourceCtx::RANK12_CLOS_TASK_ARG_COUNT;
        } else if (rank16Mixed.enabled) {
            resource.taskArgCounts[groupIndex] =
                kernelArg->isRank16MeshKernel ?
                    AlgResourceCtx::RANK16_MESH_TASK_ARG_COUNT :
                    AlgResourceCtx::RANK16_CLOS_TASK_ARG_COUNT;
        } else if (kernelArg->useGroupBroadcast) {
            resource.taskArgCounts[groupIndex] =
                AlgResourceCtx::HYBRID_TASK_ARG_COUNT;
        } else if (kernelArg->fuseTwoSlices) {
            resource.taskArgCounts[groupIndex] = group.copyLocal ?
                AlgResourceCtx::FUSED_FULL_TASK_ARG_COUNT :
                AlgResourceCtx::FUSED_BASE_TASK_ARG_COUNT;
        } else {
            resource.taskArgCounts[groupIndex] = group.copyLocal ?
                AlgResourceCtx::FULL_TASK_ARG_COUNT :
                AlgResourceCtx::BASE_TASK_ARG_COUNT;
        }
        resource.kernelLinkLoads[groupIndex] = group.linkLoad;
        kernelArg->channelCount = static_cast<uint32_t>(group.channels.size());
        for (uint32_t index = 0; index < group.channels.size(); ++index) {
            kernelArg->channels[index] = group.channels[index];
            kernelArg->remoteRanks[index] = group.remoteRanks[index];
            kernelArg->channelKinds[index] = group.channelKinds[index];
            kernelArg->alternateRailChannels[index] =
                group.alternateRails[index];
            if (rank16Mixed.enabled &&
                !kernelArg->isRank16MeshKernel &&
                group.remoteRanks[index] ==
                    rank16Mixed.relayPeerRank) {
                kernelArg->relayChannelIndex = index;
            }
            if (rank12Split.enabled &&
                !kernelArg->isRank12MeshKernel) {
                if (group.remoteRanks[index] ==
                    rank12Split.bridgeTargetRank) {
                    kernelArg->rank12BridgeTargetChannelIndex =
                        index;
                }
                for (uint32_t sourceIndex = 0;
                     sourceIndex <
                         rank12Split.fanoutSourceCount;
                     ++sourceIndex) {
                    if (group.remoteRanks[index] ==
                        rank12Split.fanoutSourceRanks[
                            sourceIndex]) {
                        kernelArg->
                            rank12FanoutSourceChannelIndices[
                                sourceIndex] = index;
                    }
                }
            }
        }
        if (rank12Nhr) {
            kernelArg->nhrStepCount =
                RANK12_NHR_STEP_COUNT;
            for (uint32_t step = 0;
                 step < RANK12_NHR_STEP_COUNT; ++step) {
                uint32_t toRank = 0;
                uint32_t fromRank = 0;
                GetNhrStepRanks(param.rankSize, param.myRank,
                    RANK12_NHR_STEP_COUNT, step,
                    toRank, fromRank);
                auto toIter = std::find(group.remoteRanks.begin(),
                    group.remoteRanks.end(), toRank);
                auto fromIter = std::find(
                    group.remoteRanks.begin(),
                    group.remoteRanks.end(), fromRank);
                if (toIter == group.remoteRanks.end() ||
                    fromIter == group.remoteRanks.end()) {
                    HCCL_ERROR("rank12 NHR step channel missing, "
                               "step=%u to=%u from=%u",
                        step, toRank, fromRank);
                    (void)HcommCcuKernelRegisterEnd(insHandle);
                    return HCCL_E_INTERNAL;
                }
                kernelArg->nhrToChannelIndices[step] =
                    static_cast<uint32_t>(std::distance(
                        group.remoteRanks.begin(), toIter));
                kernelArg->nhrFromChannelIndices[step] =
                    static_cast<uint32_t>(std::distance(
                        group.remoteRanks.begin(), fromIter));

                uint32_t sliceCount =
                    (param.rankSize - 1 +
                        (1U <<
                            (RANK12_NHR_STEP_COUNT - 1 -
                                step))) /
                    (1U <<
                        (RANK12_NHR_STEP_COUNT - step));
                if (sliceCount == 0 ||
                    sliceCount >
                        ops_hccl::NHR_MAX_STEP_SLICE_COUNT) {
                    HCCL_ERROR("invalid rank12 NHR slice count=%u "
                               "step=%u",
                        sliceCount, step);
                    (void)HcommCcuKernelRegisterEnd(insHandle);
                    return HCCL_E_INTERNAL;
                }
                kernelArg->nhrStepSliceCounts[step] =
                    sliceCount;
                uint32_t sliceIndex = param.myRank;
                uint32_t sliceDelta =
                    1U << (RANK12_NHR_STEP_COUNT - step);
                for (uint32_t index = 0;
                     index < sliceCount; ++index) {
                    kernelArg->nhrStepSliceIndices[step]
                        [index] = sliceIndex;
                    sliceIndex =
                        (sliceIndex + param.rankSize -
                            sliceDelta) %
                        param.rankSize;
                }
            }
        }
        if (rank12Split.enabled &&
            !kernelArg->isRank12MeshKernel) {
            if (kernelArg->rank12BridgeTargetChannelIndex >=
                    group.channels.size()) {
                HCCL_ERROR(
                    "rank12 hierarchical target channel missing");
                (void)HcommCcuKernelRegisterEnd(insHandle);
                return HCCL_E_INTERNAL;
            }
            for (uint32_t sourceIndex = 0;
                 sourceIndex < rank12Split.fanoutSourceCount;
                 ++sourceIndex) {
                if (kernelArg->
                        rank12FanoutSourceChannelIndices[
                            sourceIndex] >=
                    group.channels.size()) {
                    HCCL_ERROR(
                        "rank12 hierarchical source channel missing");
                    (void)HcommCcuKernelRegisterEnd(insHandle);
                    return HCCL_E_INTERNAL;
                }
            }
        }
        if (rank16Mixed.enabled &&
            !kernelArg->isRank16MeshKernel &&
            kernelArg->relayChannelIndex >= group.channels.size()) {
            HCCL_ERROR("rank16 relay peer %u is absent from Clos group",
                rank16Mixed.relayPeerRank);
            (void)HcommCcuKernelRegisterEnd(insHandle);
            return HCCL_E_INTERNAL;
        }
        kernelInfo.setKernelArg(kernelArg);

        CcuKernelHandle kernelHandle = 0;
        const void *kernelArgs[] = {kernelInfo.kernelArg};
        ccuRet = HcommCcuKernelRegister(insHandle, group.dieId,
            kernelInfo.kernelFuncName, kernelInfo.kernelFunc, kernelArgs, 1,
            &kernelHandle);
        if (ccuRet != CCU_SUCCESS) {
            HCCL_ERROR("CCU kernel register failed, dieId=%u ret=%d",
                group.dieId, ccuRet);
            (void)HcommCcuKernelRegisterEnd(insHandle);
            return ConvertCcuResult(ccuRet);
        }
        resource.ccuKernels[groupIndex] = kernelHandle;
    }

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    if (ccuRet != CCU_SUCCESS) {
        HCCL_ERROR("CCU kernel register end failed, ret=%d", ccuRet);
        return ConvertCcuResult(ccuRet);
    }
    return HCCL_SUCCESS;
}

HcclResult CacheMemTokens(const OpParam &param, AlgResourceCtx &resource)
{
    auto dataTypeIter = SIZE_TABLE.find(param.dataType);
    if (dataTypeIter == SIZE_TABLE.end() || dataTypeIter->second == 0) {
        return HCCL_E_PARA;
    }
    uint64_t inputBytes = param.count * dataTypeIter->second;
    uint64_t outputBytes = inputBytes * param.rankSize;
    resource.cachedInputAddress =
        reinterpret_cast<uint64_t>(param.inputPtr);
    resource.cachedInputBytes = inputBytes;
    resource.cachedOutputAddress =
        reinterpret_cast<uint64_t>(param.outputPtr);
    resource.cachedOutputBytes = outputBytes;

    CcuResult ccuRet = HcommCcuGetMemToken(resource.cachedInputAddress,
        inputBytes, &resource.cachedInputToken);
    if (ccuRet != CCU_SUCCESS) {
        HCCL_ERROR("failed to cache input token, ret=%d", ccuRet);
        return ConvertCcuResult(ccuRet);
    }
    ccuRet = HcommCcuGetMemToken(resource.cachedOutputAddress,
        outputBytes, &resource.cachedOutputToken);
    if (ccuRet != CCU_SUCCESS) {
        HCCL_ERROR("failed to cache output token, ret=%d", ccuRet);
        return ConvertCcuResult(ccuRet);
    }
    return HCCL_SUCCESS;
}

HcclResult CreateResource(HcclComm comm, aclrtStream stream, OpParam &param,
    AlgResourceCtx &resource)
{
    resource.layoutVersion = 143;
    resource.rankSize = param.rankSize;
    resource.rankId = param.myRank;
    resource.cachedStreamAddress = reinterpret_cast<uint64_t>(stream);
    CHK_RET(CacheMemTokens(param, resource));
    bool enableRank16Fanout =
        param.rankSize == 16 &&
        resource.cachedInputBytes >= GROUP_FANOUT_MIN_BYTES;
    bool preferRank12MeshFirst =
        param.rankSize == 12 &&
        resource.cachedInputBytes == RANK12_MESH_FIRST_BYTES;
    bool fuseTwoSlices =
        (param.rankSize == 4 || param.rankSize == 12) &&
        resource.cachedInputBytes == FUSED_512_BYTES;
    bool useFineLocalCopy =
        param.rankSize == 12 &&
        resource.cachedInputBytes >= FINE_LOCAL_COPY_MIN_BYTES;
    bool useRank12DirectLocalCopy = false;
    bool useSingleRank4LocalCopy =
        param.rankSize == 4 &&
        resource.cachedInputBytes ==
            FUSED_512_BYTES;

    std::vector<ChannelGroup> groups;
    bool rank12Nhr = false;
    bool rank12DualRail = false;
    uint32_t primaryRailWeight = 0;
    uint32_t alternateRailWeight = 0;
    CHK_RET(TryAcquireRank12NhrChannels(
        comm, param, 0, groups, rank12Nhr));
    if (!rank12Nhr) {
        CHK_RET(AcquireChannels(comm, param,
            resource.cachedInputBytes, groups, rank12DualRail,
            primaryRailWeight, alternateRailWeight));
    }

    std::stable_sort(groups.begin(), groups.end(),
        [](const ChannelGroup &left, const ChannelGroup &right) {
            return left.linkLoad > right.linkLoad;
        });
    Rank12SplitRelayConfig rank12Split =
        DetectRank12SplitRelayTopology(param, 0, groups);
    Rank16MixedConfig rank16Mixed =
        DetectRank16MixedTopology(
            param, resource.cachedInputBytes, groups);
    if (rank12Nhr) {
        resource.rank12Nhr = true;
    } else if (rank12DualRail) {
        groups.back().copyLocal = true;
        resource.rank12DualRail = true;
        resource.rank12PrimaryRailWeight =
            primaryRailWeight;
        resource.rank12AlternateRailWeight =
            alternateRailWeight;
        preferRank12MeshFirst = true;
    } else if (rank12Split.enabled) {
        groups[rank12Split.meshGroupIndex].copyLocal = false;
        groups[rank12Split.closGroupIndex].copyLocal = true;
        resource.rank12SplitRelay = true;
        resource.rank12SmallServer =
            rank12Split.smallServer;
        resource.rank12MeshKernelIndex =
            rank12Split.meshGroupIndex;
        resource.rank12ClosKernelIndex =
            rank12Split.closGroupIndex;
        resource.rank12FanoutSourceCount =
            rank12Split.fanoutSourceCount;
        for (uint32_t sourceIndex = 0;
             sourceIndex < rank12Split.fanoutSourceCount;
             ++sourceIndex) {
            resource.rank12FanoutSourceRanks[sourceIndex] =
                rank12Split.fanoutSourceRanks[sourceIndex];
        }
    } else if (rank16Mixed.enabled) {
        groups[rank16Mixed.meshGroupIndex].copyLocal = true;
        groups[rank16Mixed.closGroupIndex].copyLocal = false;
        resource.rank16MixedRelay = true;
        resource.rank16MeshKernelIndex =
            rank16Mixed.meshGroupIndex;
        resource.rank16ClosKernelIndex =
            rank16Mixed.closGroupIndex;
        resource.rank16RelayPeerRank =
            rank16Mixed.relayPeerRank;
    } else if (enableRank16Fanout) {
        auto fanoutGroup = std::find_if(groups.rbegin(), groups.rend(),
            [](const ChannelGroup &group) {
                return group.closCount >= 2;
            });
        if (fanoutGroup != groups.rend()) {
            fanoutGroup->copyLocal = true;
        } else {
            for (ChannelGroup &group : groups) {
                group.copyLocal = true;
            }
        }
    } else if (useSingleRank4LocalCopy) {
        groups.back().copyLocal = true;
    } else {
        for (ChannelGroup &group : groups) {
            group.copyLocal = true;
        }
    }
    resource.rank12DirectLocalCopy =
        useRank12DirectLocalCopy &&
        !rank12DualRail &&
        !rank12Split.enabled;
    resource.rank12DirectReuse =
        param.rankSize == 12 &&
        resource.cachedInputBytes == RANK12_MESH_FIRST_BYTES &&
        !rank12Nhr && !rank12DualRail &&
        !rank12Split.enabled;

    resource.threads[0] = param.cpuThread;
    if (groups.size() > 1) {
        CHK_RET(HcclThreadAcquire(comm, CommEngine::COMM_ENGINE_CCU,
            static_cast<uint32_t>(groups.size() - 1),
            THREAD_NOTIFY_NUM, &resource.threads[1]));
    }
    CHK_RET(RegisterKernels(
        comm, param, groups, enableRank16Fanout,
        preferRank12MeshFirst, fuseTwoSlices,
        useFineLocalCopy, rank12DualRail,
        resource.rank12DirectLocalCopy,
        resource.rank12DirectReuse, rank12Nhr,
        rank16Mixed, rank12Split,
        resource));

    param.ctxSize = sizeof(resource);
    CHK_RET(HcclEngineCtxCreate(comm, param.tag, CommEngine::COMM_ENGINE_CCU,
        param.ctxSize, &param.resCtx));
    CHK_RET(HcclEngineCtxCopy(comm, CommEngine::COMM_ENGINE_CCU, param.tag,
        &resource, sizeof(resource), 0));
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclAllGather(
    void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param;
    int tagRet =
        std::snprintf(param.tag, sizeof(param.tag), "%s", RESOURCE_TAG);
    if (tagRet < 0 || static_cast<uint32_t>(tagRet) >= sizeof(param.tag)) {
        HCCL_ERROR("failed to set resource tag");
        return HCCL_E_INTERNAL;
    }
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = sendCount;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH] = {};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;
    void *ctx = nullptr;
    uint64_t size = 0;
    bool resourceCached =
        HcclEngineCtxGet(comm, param.tag, ccuEngine, &ctx, &size) ==
        HCCL_SUCCESS;
    AlgResourceCtx cachedResource;
    if (resourceCached) {
        CHK_RET(cachedResource.DeSerialize(ctx, size));
        if (cachedResource.layoutVersion != 143 ||
            cachedResource.rankSize == 0 ||
            cachedResource.rankSize > MAX_RANK_SIZE ||
            cachedResource.rankId >= cachedResource.rankSize) {
            HCCL_ERROR("invalid cached CCU resource metadata");
            return HCCL_E_INTERNAL;
        }
        param.myRank = cachedResource.rankId;
        param.rankSize = cachedResource.rankSize;
    } else {
        CHK_RET(HcclGetRankId(comm, &param.myRank));
        CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    }
    CHK_RET(ValidateParam(param));
    if (param.count == 0) {
        return HCCL_SUCCESS;
    }

    uint64_t streamAddress = reinterpret_cast<uint64_t>(stream);
    if (resourceCached &&
        cachedResource.cachedStreamAddress == streamAddress &&
        cachedResource.threads[0] != 0) {
        param.cpuThread = cachedResource.threads[0];
    } else {
        CHK_RET(HcclThreadAcquireWithStream(
            comm, ccuEngine, stream, THREAD_NOTIFY_NUM,
            &param.cpuThread));
    }
    if (param.rankSize == 1) {
        uint64_t inputBytes = param.count * SIZE_TABLE.find(param.dataType)->second;
        return static_cast<HcclResult>(
            HcommLocalCopyOnThread(param.cpuThread, recvBuf, sendBuf, inputBytes));
    }

    if (resourceCached) {
        param.resCtx = ctx;
        param.ctxSize = size;
    } else {
        AlgResourceCtx resCtxHost;
        CHK_RET(CreateResource(comm, stream, param, resCtxHost));
    }

    CHK_RET(ops_hccl::ExecOp(param));
    return HCCL_SUCCESS;
}
