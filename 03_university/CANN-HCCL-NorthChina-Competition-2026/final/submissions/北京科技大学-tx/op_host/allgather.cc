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
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <vector>

#include <ccu/ccu_launch.h>
#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>

#include "common.h"
#include "custom.h"
#include "exec_op.h"
#include "hccl.h"
#include "log.h"

namespace {

constexpr uint32_t LAYER_MESH = 0;
constexpr uint32_t LAYER_CLOS = 1;
constexpr uint32_t CHANNEL_NOTIFY_NUM = 1;
constexpr uint32_t MAX_PEERS_PER_DIE = 8;
constexpr uint32_t MAIN_NOTIFY_COUNT_2X8 = 5;
constexpr uint32_t MAIN_NOTIFY_COUNT_DIRECT = 1;
constexpr uint32_t MAIN_NOTIFY_COUNT_A84_PREFIX = 2;
constexpr uint32_t SLAVE_NOTIFY_COUNT = 1;
constexpr uint64_t PF1_SMALL_DIRECT_THRESHOLD = 1ULL * 1024ULL * 1024ULL;
constexpr uint64_t PF1_DIRECT_WINDOW_SIZE = 2ULL * MAX_DATA_SIZE;
constexpr char REGISTERED_SMALL_MEM_PREFIX[] = "hccl_ag_small_send_";

struct TopologyGroups {
    std::vector<uint32_t> localRanks;
    std::vector<uint32_t> remoteRanks;
};

struct ChannelSet {
    std::vector<uint32_t> peers;
    std::vector<ChannelHandle> handles;
    std::vector<uint64_t> registeredRemoteInputs;
};

struct KernelRegistration {
    const char *name = nullptr;
    const void *function = nullptr;
    std::shared_ptr<void> arg;
};


struct A84PrefixMetadata {
    uint32_t myRank = 0;
    uint32_t localGroupSize = 0;
    uint32_t remoteGroupSize = 0;
    uint32_t outgoingProxyChannelIndex = 0;
    uint32_t incomingProxyMask = 0;
    bool copyLocalOnLayer0 = false;
    std::array<uint32_t, 2> relaySourceRanks{};
    uint32_t relaySourceCount = 0;
    bool enabled = false;
};


std::vector<uint32_t> BuildFourByOneLayer1CyclicOrder(
    const TopologyGroups &groups, const OpParam &param)
{
    if (param.rankSize != 4 || groups.localRanks.size() != 1 ||
        groups.remoteRanks.size() != 3) {
        return groups.remoteRanks;
    }

    std::vector<uint32_t> allRanks = groups.remoteRanks;
    allRanks.push_back(param.myRank);
    std::sort(allRanks.begin(), allRanks.end());
    const auto myIt = std::find(allRanks.begin(), allRanks.end(), param.myRank);
    if (myIt == allRanks.end()) {
        return {};
    }

    const size_t myIndex = static_cast<size_t>(std::distance(allRanks.begin(), myIt));
    std::vector<uint32_t> orderedPeers;
    orderedPeers.reserve(groups.remoteRanks.size());
    for (size_t step = 1; step < allRanks.size(); ++step) {
        orderedPeers.push_back(allRanks[(myIndex + step) % allRanks.size()]);
    }
    return orderedPeers;
}


std::vector<uint32_t> BuildEightPlusFourLayer1BalancedOrder(
    const TopologyGroups &groups, const OpParam &param)
{
    const bool isEightPlusFour = param.rankSize == 12 &&
        ((groups.localRanks.size() == 8 && groups.remoteRanks.size() == 4) ||
         (groups.localRanks.size() == 4 && groups.remoteRanks.size() == 8));
    if (!isEightPlusFour) {
        return groups.remoteRanks;
    }

    const auto myIt =
        std::find(groups.localRanks.begin(), groups.localRanks.end(), param.myRank);
    if (myIt == groups.localRanks.end()) {
        return {};
    }
    const size_t localIndex =
        static_cast<size_t>(std::distance(groups.localRanks.begin(), myIt));
    const size_t phase =
        (localIndex * groups.remoteRanks.size()) / groups.localRanks.size();

    std::vector<uint32_t> orderedPeers;
    orderedPeers.reserve(groups.remoteRanks.size());
    for (size_t step = 0; step < groups.remoteRanks.size(); ++step) {
        orderedPeers.push_back(
            groups.remoteRanks[(phase + step) % groups.remoteRanks.size()]);
    }
    return orderedPeers;
}

bool UsePf1LatinLayer1Order(const OpParam &param)
{
    if (param.rankSize != 16) {
        return false;
    }
    const auto sizeIt = SIZE_TABLE.find(param.dataType);
    if (sizeIt == SIZE_TABLE.end() ||
        param.count > std::numeric_limits<uint64_t>::max() / sizeIt->second) {
        return false;
    }
    const uint64_t dataSize = param.count * sizeIt->second;
    return dataSize > PF1_SMALL_DIRECT_THRESHOLD &&
        dataSize <= PF1_DIRECT_WINDOW_SIZE;
}

bool UseDedicatedSmallKernel(const OpParam &param)
{
    const auto sizeIt = SIZE_TABLE.find(param.dataType);
    if (sizeIt == SIZE_TABLE.end() ||
        param.count > std::numeric_limits<uint64_t>::max() / sizeIt->second) {
        return false;
    }
    const uint64_t dataSize = param.count * sizeIt->second;
    return dataSize != 0 && dataSize <= PF1_SMALL_DIRECT_THRESHOLD;
}

std::vector<uint32_t> BuildTwoStageLayer1LatinOrder(
    const TopologyGroups &groups, const OpParam &param)
{
    if (!UsePf1LatinLayer1Order(param) ||
        groups.localRanks.size() != 8 || groups.remoteRanks.size() != 8) {
        return groups.remoteRanks;
    }

    const auto myIt =
        std::find(groups.localRanks.begin(), groups.localRanks.end(), param.myRank);
    if (myIt == groups.localRanks.end()) {
        return {};
    }
    const size_t localIndex =
        static_cast<size_t>(std::distance(groups.localRanks.begin(), myIt));

    // Latin cyclic order: source local index i emits to destination
    // (i + step) mod 8 at issue slot step.  For every fixed step this is a
    // permutation of all eight remote ranks, so no issue slot concentrates
    // multiple sources on the same Clos destination.
    std::vector<uint32_t> orderedPeers;
    orderedPeers.reserve(groups.remoteRanks.size());
    for (size_t step = 0; step < groups.remoteRanks.size(); ++step) {
        orderedPeers.push_back(
            groups.remoteRanks[(localIndex + step) % groups.remoteRanks.size()]);
    }
    return orderedPeers;
}


std::vector<uint32_t> BuildDirectTopologyLayer1Order(
    const TopologyGroups &groups, const OpParam &param)
{
    if (param.rankSize == 16) {
        return BuildTwoStageLayer1LatinOrder(groups, param);
    }
    if (param.rankSize == 4) {
        return BuildFourByOneLayer1CyclicOrder(groups, param);
    }
    if (param.rankSize == 12) {
        return BuildEightPlusFourLayer1BalancedOrder(groups, param);
    }
    return groups.remoteRanks;
}

HcclResult FindPeerChannelIndex(const ChannelSet &channelSet,
    uint32_t peerRank, uint32_t &channelIndex)
{
    const auto peerIt = std::find(
        channelSet.peers.begin(), channelSet.peers.end(), peerRank);
    CHK_PRT_RET(peerIt == channelSet.peers.end(),
        HCCL_ERROR("Peer rank %u has no acquired Channel", peerRank),
        HCCL_E_INTERNAL);
    channelIndex = static_cast<uint32_t>(
        std::distance(channelSet.peers.begin(), peerIt));
    return HCCL_SUCCESS;
}

std::vector<uint32_t> BuildEightPlusFourLayer0CyclicOrder(
    const TopologyGroups &groups, const OpParam &param)
{
    const bool isEightPlusFour = param.rankSize == 12 &&
        (groups.localRanks.size() == 8 || groups.localRanks.size() == 4);
    if (!isEightPlusFour) {
        std::vector<uint32_t> orderedPeers;
        orderedPeers.reserve(groups.localRanks.size());
        for (uint32_t rank : groups.localRanks) {
            if (rank != param.myRank) {
                orderedPeers.push_back(rank);
            }
        }
        return orderedPeers;
    }

    const auto myIt =
        std::find(groups.localRanks.begin(), groups.localRanks.end(), param.myRank);
    if (myIt == groups.localRanks.end()) {
        return {};
    }

    const size_t myIndex =
        static_cast<size_t>(std::distance(groups.localRanks.begin(), myIt));
    std::vector<uint32_t> orderedPeers;
    orderedPeers.reserve(groups.localRanks.size() - 1);
    for (size_t step = 1; step < groups.localRanks.size(); ++step) {
        orderedPeers.push_back(
            groups.localRanks[(myIndex + step) % groups.localRanks.size()]);
    }
    return orderedPeers;
}

HcclResult BuildA84PrefixMetadata(const TopologyGroups &groups,
    const OpParam &param, const ChannelSet &layer1Channels, A84PrefixMetadata &metadata)
{
    metadata.myRank = param.myRank;
    if (param.rankSize != 12) {
        return HCCL_SUCCESS;
    }

    const bool validTopology =
        (groups.localRanks.size() == 8 && groups.remoteRanks.size() == 4) ||
        (groups.localRanks.size() == 4 && groups.remoteRanks.size() == 8);
    CHK_PRT_RET(!validTopology,
        HCCL_ERROR("A84 Prefix-first requires an 8+4 partition; got local=%zu remote=%zu",
            groups.localRanks.size(), groups.remoteRanks.size()),
        HCCL_E_INTERNAL);

    const auto localIt =
        std::find(groups.localRanks.begin(), groups.localRanks.end(), param.myRank);
    CHK_PRT_RET(localIt == groups.localRanks.end(),
        HCCL_ERROR("Rank %u is absent from its local 8+4 group", param.myRank),
        HCCL_E_INTERNAL);
    const size_t localIndex =
        static_cast<size_t>(std::distance(groups.localRanks.begin(), localIt));

    metadata.enabled = true;
    metadata.localGroupSize = static_cast<uint32_t>(groups.localRanks.size());
    metadata.remoteGroupSize = static_cast<uint32_t>(groups.remoteRanks.size());
    metadata.copyLocalOnLayer0 =
        groups.localRanks.size() - 1 <= groups.remoteRanks.size();

    // Outgoing source data uses exactly one destination proxy.  The mapping is
    // source localIndex modulo the destination-group size.
    const size_t outgoingProxyIndex = localIndex % groups.remoteRanks.size();
    const uint32_t outgoingProxyRank = groups.remoteRanks[outgoingProxyIndex];
    const auto outgoingIt = std::find(layer1Channels.peers.begin(),
        layer1Channels.peers.end(), outgoingProxyRank);
    CHK_PRT_RET(outgoingIt == layer1Channels.peers.end(),
        HCCL_ERROR("Outgoing proxy rank %u has no layer-1 Channel", outgoingProxyRank),
        HCCL_E_INTERNAL);
    metadata.outgoingProxyChannelIndex =
        static_cast<uint32_t>(std::distance(layer1Channels.peers.begin(), outgoingIt));

    // Incoming prefixes are those remote sources mapped to this local proxy.
    for (uint32_t channelIndex = 0; channelIndex < layer1Channels.peers.size(); ++channelIndex) {
        const uint32_t sourceRank = layer1Channels.peers[channelIndex];
        const auto sourceIt =
            std::find(groups.remoteRanks.begin(), groups.remoteRanks.end(), sourceRank);
        CHK_PRT_RET(sourceIt == groups.remoteRanks.end(),
            HCCL_ERROR("Layer-1 peer %u is absent from the sorted remote group", sourceRank),
            HCCL_E_INTERNAL);
        const size_t sourceIndex =
            static_cast<size_t>(std::distance(groups.remoteRanks.begin(), sourceIt));
        if ((sourceIndex % groups.localRanks.size()) != localIndex) {
            continue;
        }

        metadata.incomingProxyMask |= static_cast<uint32_t>(1U << channelIndex);
        CHK_PRT_RET(metadata.relaySourceCount >= metadata.relaySourceRanks.size(),
            HCCL_ERROR("Rank %u proxies more than two remote sources", param.myRank),
            HCCL_E_NOT_SUPPORT);
        metadata.relaySourceRanks[metadata.relaySourceCount++] = sourceRank;
    }

    const uint32_t expectedProxyCount = groups.localRanks.size() == 4 ? 2U :
        (localIndex < groups.remoteRanks.size() ? 1U : 0U);
    CHK_PRT_RET(metadata.relaySourceCount != expectedProxyCount,
        HCCL_ERROR("Rank %u proxy count %u differs from expected %u",
            param.myRank, metadata.relaySourceCount, expectedProxyCount),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult QueryTopologyGroups(HcclComm comm, const OpParam &param, TopologyGroups &groups)
{
    uint32_t *layers = nullptr;
    uint32_t layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layers, &layerCount));
    CHK_PRT_RET(layers == nullptr || layerCount == 0,
        HCCL_ERROR("Rank graph returned no network layers"), HCCL_E_INTERNAL);

    const std::vector<uint32_t> layerList(layers, layers + layerCount);
    const bool hasLayer0 =
        std::find(layerList.begin(), layerList.end(), LAYER_MESH) != layerList.end();
    const bool hasLayer1 =
        std::find(layerList.begin(), layerList.end(), LAYER_CLOS) != layerList.end();
    CHK_PRT_RET(!hasLayer1,
        HCCL_ERROR("Layer-1 is required for the competition topologies"), HCCL_E_NOT_SUPPORT);

    if (hasLayer0) {
        uint32_t *localRanks = nullptr;
        uint32_t localRankCount = 0;
        CHK_RET(HcclRankGraphGetRanksByLayer(
            comm, LAYER_MESH, &localRanks, &localRankCount));
        CHK_PRT_RET(localRanks == nullptr || localRankCount == 0,
            HCCL_ERROR("Layer-0 rank list is empty"), HCCL_E_INTERNAL);
        groups.localRanks.assign(localRanks, localRanks + localRankCount);
    } else {
        groups.localRanks.push_back(param.myRank);
    }

    std::sort(groups.localRanks.begin(), groups.localRanks.end());
    CHK_PRT_RET(std::find(groups.localRanks.begin(), groups.localRanks.end(), param.myRank) ==
            groups.localRanks.end(),
        HCCL_ERROR("Rank %u is absent from its layer-0 instance", param.myRank), HCCL_E_INTERNAL);
    CHK_PRT_RET(groups.localRanks.size() > MAX_PEERS_PER_DIE,
        HCCL_ERROR("Layer-0 instance has %zu ranks", groups.localRanks.size()), HCCL_E_NOT_SUPPORT);

    uint32_t *allRanks = nullptr;
    uint32_t allRankCount = 0;
    CHK_RET(HcclRankGraphGetRanksByLayer(comm, LAYER_CLOS, &allRanks, &allRankCount));
    CHK_PRT_RET(allRanks == nullptr || allRankCount != param.rankSize,
        HCCL_ERROR("Layer-1 rank count %u differs from communicator size %u",
            allRankCount, param.rankSize),
        HCCL_E_INTERNAL);

    for (uint32_t index = 0; index < allRankCount; ++index) {
        const uint32_t rank = allRanks[index];
        if (std::find(groups.localRanks.begin(), groups.localRanks.end(), rank) ==
            groups.localRanks.end()) {
            groups.remoteRanks.push_back(rank);
        }
    }
    std::sort(groups.remoteRanks.begin(), groups.remoteRanks.end());

    CHK_PRT_RET(groups.remoteRanks.empty() ||
            groups.localRanks.size() + groups.remoteRanks.size() != param.rankSize,
        HCCL_ERROR("Layer partition is incomplete: local=%zu remote=%zu rankSize=%u",
            groups.localRanks.size(), groups.remoteRanks.size(), param.rankSize),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(groups.remoteRanks.size() > MAX_PEERS_PER_DIE,
        HCCL_ERROR("Layer-1 requires %zu peer channels", groups.remoteRanks.size()),
        HCCL_E_NOT_SUPPORT);
    return HCCL_SUCCESS;
}

bool IsTwoStage2x8(const TopologyGroups &groups, const OpParam &param)
{
    return param.rankSize == 16 && groups.localRanks.size() == 8 &&
        groups.remoteRanks.size() == 8;
}

HcclResult FillChannelDesc(HcclComm comm, uint32_t myRank, uint32_t remoteRank,
    uint32_t netLayer, HcclChannelDesc &desc)
{
    CommLink *links = nullptr;
    uint32_t linkCount = 0;
    CHK_RET(HcclRankGraphGetLinks(comm, netLayer, myRank, remoteRank, &links, &linkCount));

    constexpr CommProtocol requiredProtocol = CommProtocol::COMM_PROTOCOL_UBC_CTP;
    for (uint32_t index = 0; index < linkCount; ++index) {
        const CommLink &link = links[index];
        if (link.linkAttr.linkProtocol != requiredProtocol) {
            continue;
        }

        desc.remoteRank = remoteRank;
        desc.notifyNum = CHANNEL_NOTIFY_NUM;
        desc.channelProtocol = link.linkAttr.linkProtocol;
        desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
        desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
        desc.localEndpoint.loc = link.srcEndpointDesc.loc;
        desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
        desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
        desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
        return HCCL_SUCCESS;
    }

    HCCL_ERROR("No UBC_CTP link on layer %u between rank %u and rank %u",
        netLayer, myRank, remoteRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult AcquireChannels(HcclComm comm, uint32_t myRank,
    const std::vector<uint32_t> &ranks, uint32_t netLayer,
    HcclMemHandle *registeredInputHandle, uint64_t registeredInputSize,
    ChannelSet &channelSet)
{
    for (uint32_t rank : ranks) {
        if (rank != myRank) {
            channelSet.peers.push_back(rank);
        }
    }
    CHK_PRT_RET(channelSet.peers.empty() || channelSet.peers.size() > MAX_PEERS_PER_DIE,
        HCCL_ERROR("Invalid peer count %zu for layer %u", channelSet.peers.size(), netLayer),
        HCCL_E_PARA);

    std::vector<HcclChannelDesc> descriptions(channelSet.peers.size());
    CHK_RET(HcclChannelDescInit(descriptions.data(), static_cast<uint32_t>(descriptions.size())));
    for (uint32_t index = 0; index < channelSet.peers.size(); ++index) {
        CHK_RET(FillChannelDesc(
            comm, myRank, channelSet.peers[index], netLayer, descriptions[index]));
        if (registeredInputHandle != nullptr) {
            descriptions[index].memHandles = registeredInputHandle;
            descriptions[index].memHandleNum = 1;
        }
    }

    channelSet.handles.resize(channelSet.peers.size());
    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, descriptions.data(),
        static_cast<uint32_t>(descriptions.size()), channelSet.handles.data()));

    if (registeredInputHandle == nullptr) {
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(registeredInputSize == 0,
        HCCL_ERROR("Registered small input size must be non-zero"),
        HCCL_E_PARA);
    channelSet.registeredRemoteInputs.resize(channelSet.handles.size());
    for (uint32_t index = 0; index < channelSet.handles.size(); ++index) {
        uint32_t remoteMemNum = 0;
        CommMem *remoteMems = nullptr;
        char **remoteMemTags = nullptr;
        CHK_RET(HcclChannelGetRemoteMems(comm, channelSet.handles[index],
            &remoteMemNum, &remoteMems, &remoteMemTags));
        CHK_PRT_RET(remoteMemNum == 0 || remoteMems == nullptr ||
                remoteMemTags == nullptr,
            HCCL_ERROR("Registered-small peer %u returned no remote memories",
                channelSet.peers[index]),
            HCCL_E_INTERNAL);

        char expectedPrefix[HCCL_RES_TAG_MAX_LEN]{};
        const int prefixResult = std::snprintf(expectedPrefix,
            sizeof(expectedPrefix), "%sr%u_", REGISTERED_SMALL_MEM_PREFIX,
            channelSet.peers[index]);
        CHK_PRT_RET(prefixResult <= 0 ||
                static_cast<size_t>(prefixResult) >= sizeof(expectedPrefix),
            HCCL_ERROR("Failed to construct registered-small peer prefix"),
            HCCL_E_INTERNAL);
        const size_t prefixLength = static_cast<size_t>(prefixResult);

        uint32_t matchCount = 0;
        uint64_t selectedAddress = 0;
        for (uint32_t memIndex = 0; memIndex < remoteMemNum; ++memIndex) {
            const char *remoteTag = remoteMemTags[memIndex];
            if (remoteTag == nullptr ||
                std::strncmp(remoteTag, expectedPrefix, prefixLength) != 0) {
                continue;
            }

            const CommMem &remoteMem = remoteMems[memIndex];
            CHK_PRT_RET(remoteMem.type != COMM_MEM_TYPE_DEVICE ||
                    remoteMem.addr == nullptr ||
                    remoteMem.size != registeredInputSize,
                HCCL_ERROR("Invalid registered-small memory from peer %u: "
                    "type=%d addr=%p size=%llu expected=%llu",
                    channelSet.peers[index],
                    static_cast<int>(remoteMem.type), remoteMem.addr,
                    static_cast<unsigned long long>(remoteMem.size),
                    static_cast<unsigned long long>(registeredInputSize)),
                HCCL_E_INTERNAL);
            selectedAddress = reinterpret_cast<uint64_t>(remoteMem.addr);
            ++matchCount;
        }

        CHK_PRT_RET(matchCount != 1,
            HCCL_ERROR("Registered-small peer %u returned %u memories, "
                "but %u entries matched tag prefix %s",
                channelSet.peers[index], remoteMemNum, matchCount,
                expectedPrefix),
            HCCL_E_INTERNAL);
        channelSet.registeredRemoteInputs[index] = selectedAddress;
    }
    return HCCL_SUCCESS;
}

std::shared_ptr<CcuAllGatherKernelArg> MakeDirectKernelArg(const ChannelSet &channelSet,
    AllGatherKernelLayer layer, bool directCopyLocal,
    uint32_t rd4Xor1ChannelIndex = 0, uint32_t rd4Xor2ChannelIndex = 0)
{
    auto arg = std::make_shared<CcuAllGatherKernelArg>();
    arg->channelCount = static_cast<uint32_t>(channelSet.handles.size());
    arg->layer = static_cast<uint32_t>(layer);
    arg->directCopyLocal = directCopyLocal ? 1U : 0U;
    arg->columnCopyLocal = 0;
    arg->columnPeerChannelIndex = 0;
    arg->rd4Xor1ChannelIndex = rd4Xor1ChannelIndex;
    arg->rd4Xor2ChannelIndex = rd4Xor2ChannelIndex;
    for (uint32_t index = 0; index < channelSet.handles.size(); ++index) {
        arg->channels[index] = channelSet.handles[index];
    }
    return arg;
}

std::shared_ptr<CcuSmallPullKernelArg> MakeSmallPullKernelArg(
    const ChannelSet &channelSet, bool directCopyLocal)
{
    auto arg = std::make_shared<CcuSmallPullKernelArg>();
    arg->channelCount = static_cast<uint32_t>(channelSet.handles.size());
    arg->directCopyLocal = directCopyLocal ? 1U : 0U;
    for (uint32_t index = 0; index < channelSet.handles.size(); ++index) {
        arg->channels[index] = channelSet.handles[index];
        arg->registeredRemoteInputs[index] =
            channelSet.registeredRemoteInputs[index];
        arg->peerRanks[index] = channelSet.peers[index];
    }
    return arg;
}

std::shared_ptr<CcuM1StageKernelArg> MakeStageKernelArg(
    const ChannelSet &channelSet, uint32_t partnerChannelIndex)
{
    auto arg = std::make_shared<CcuM1StageKernelArg>();
    arg->channelCount = static_cast<uint32_t>(channelSet.handles.size());
    arg->partnerChannelIndex = partnerChannelIndex;
    for (uint32_t index = 0; index < channelSet.handles.size(); ++index) {
        arg->channels[index] = channelSet.handles[index];
    }
    return arg;
}

std::shared_ptr<CcuA84PrefixLayer0KernelArg> MakeA84PrefixLayer0KernelArg(
    const ChannelSet &channelSet, const A84PrefixMetadata &metadata)
{
    auto arg = std::make_shared<CcuA84PrefixLayer0KernelArg>();
    arg->channelCount = static_cast<uint32_t>(channelSet.handles.size());
    arg->myRank = metadata.myRank;
    arg->copyLocal = metadata.copyLocalOnLayer0 ? 1U : 0U;
    arg->relaySourceCount = metadata.relaySourceCount;
    for (uint32_t index = 0; index < metadata.relaySourceRanks.size(); ++index) {
        arg->relaySourceRanks[index] = metadata.relaySourceRanks[index];
    }
    for (uint32_t index = 0; index < channelSet.handles.size(); ++index) {
        arg->channels[index] = channelSet.handles[index];
    }
    return arg;
}

std::shared_ptr<CcuA84PrefixLayer1KernelArg> MakeA84PrefixLayer1KernelArg(
    const ChannelSet &channelSet, const A84PrefixMetadata &metadata)
{
    auto arg = std::make_shared<CcuA84PrefixLayer1KernelArg>();
    arg->channelCount = static_cast<uint32_t>(channelSet.handles.size());
    arg->outgoingProxyChannelIndex = metadata.outgoingProxyChannelIndex;
    arg->incomingProxyMask = metadata.incomingProxyMask;
    arg->copyLocal = metadata.copyLocalOnLayer0 ? 0U : 1U;
    for (uint32_t index = 0; index < channelSet.handles.size(); ++index) {
        arg->channels[index] = channelSet.handles[index];
    }
    return arg;
}

HcclResult RegisterKernels(HcclComm comm,
    const std::vector<KernelRegistration> &registrations, AlgResourceCtx &resourceCtx)
{
    CcuInsHandle insHandle{};
    uint32_t insCount = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insCount));
    CHK_PRT_RET(insCount != 1,
        HCCL_ERROR("Expected one CCU instruction instance, got %u", insCount), HCCL_E_INTERNAL);

    CcuResult result = HcommCcuKernelRegisterStart(insHandle);
    if (result != CCU_SUCCESS) {
        HCCL_ERROR("HcommCcuKernelRegisterStart failed: %d", result);
        return ConvertCcuToHccl(result);
    }

    resourceCtx.ccuKernels.resize(registrations.size());
    for (uint32_t index = 0; index < registrations.size(); ++index) {
        const KernelRegistration &registration = registrations[index];
        const void *registerArgs[] = {registration.arg.get()};
        constexpr uint32_t reservedDieId = 0;
        result = HcommCcuKernelRegister(insHandle, reservedDieId,
            registration.name, registration.function, registerArgs, 1,
            &resourceCtx.ccuKernels[index]);
        if (result != CCU_SUCCESS) {
            HCCL_ERROR("HcommCcuKernelRegister failed for %s: %d",
                registration.name, result);
            const CcuResult endResult = HcommCcuKernelRegisterEnd(insHandle);
            if (endResult != CCU_SUCCESS) {
                HCCL_ERROR("HcommCcuKernelRegisterEnd after %s failure returned %d",
                    registration.name, endResult);
            }
            return ConvertCcuToHccl(result);
        }
    }

    result = HcommCcuKernelRegisterEnd(insHandle);
    if (result != CCU_SUCCESS) {
        HCCL_ERROR("HcommCcuKernelRegisterEnd failed after %zu kernels: %d",
            registrations.size(), result);
        return ConvertCcuToHccl(result);
    }
    return HCCL_SUCCESS;
}

HcclResult AllocateResources(HcclComm comm, const OpParam &param,
    HcclMemHandle *registeredInputHandle, uint64_t registeredInputSize,
    AlgResourceCtx &resourceCtx)
{
    TopologyGroups groups;
    CHK_RET(QueryTopologyGroups(comm, param, groups));

    const bool hasLayer0 = groups.localRanks.size() > 1;
    const bool useTwoStage2x8 = IsTwoStage2x8(groups, param);
    const bool useDedicatedSmallKernel = UseDedicatedSmallKernel(param);
    const bool useAsymmetricPrefix8Plus4 = param.rankSize == 12 &&
        ((groups.localRanks.size() == 8 && groups.remoteRanks.size() == 4) ||
         (groups.localRanks.size() == 4 && groups.remoteRanks.size() == 8));
    resourceCtx.hasLayer0Kernel = hasLayer0 ? 1U : 0U;
    resourceCtx.useTwoStage2x8 = useTwoStage2x8 ? 1U : 0U;
    resourceCtx.partnerRank = INVALID_VALUE_RANKID;
    resourceCtx.useAsymmetricPrefix8Plus4 = useAsymmetricPrefix8Plus4 ? 1U : 0U;
    resourceCtx.localGroupSize = static_cast<uint32_t>(groups.localRanks.size());

    resourceCtx.threads.push_back(param.cpuThread);
    if (hasLayer0) {
        resourceCtx.threads.resize(2);
        constexpr uint32_t extraThreadCount = 1;
        CHK_RET(HcclThreadAcquire(comm, CommEngine::COMM_ENGINE_CCU, extraThreadCount,
            SLAVE_NOTIFY_COUNT, &resourceCtx.threads[1]));
    }

    const std::vector<uint32_t> layer1PeerOrder =
        BuildDirectTopologyLayer1Order(groups, param);
    CHK_PRT_RET(layer1PeerOrder.size() != groups.remoteRanks.size(),
        HCCL_ERROR("Failed to build topology-balanced layer-1 order for rank %u", param.myRank),
        HCCL_E_INTERNAL);

    ChannelSet layer1Channels;
    CHK_RET(AcquireChannels(
        comm, param.myRank, layer1PeerOrder, LAYER_CLOS,
        registeredInputHandle, registeredInputSize, layer1Channels));

    uint32_t rd4Xor1ChannelIndex = 0;
    uint32_t rd4Xor2ChannelIndex = 0;
    if (param.rankSize == 4) {
        CHK_PRT_RET(layer1Channels.peers.size() != 3,
            HCCL_ERROR("RD4 requires exactly three layer-1 peers; got %zu",
                layer1Channels.peers.size()),
            HCCL_E_INTERNAL);
        CHK_RET(FindPeerChannelIndex(
            layer1Channels, param.myRank ^ 1U, rd4Xor1ChannelIndex));
        CHK_RET(FindPeerChannelIndex(
            layer1Channels, param.myRank ^ 2U, rd4Xor2ChannelIndex));
        CHK_PRT_RET(rd4Xor1ChannelIndex == rd4Xor2ChannelIndex,
            HCCL_ERROR("RD4 xor1/xor2 resolved to the same Channel index %u",
                rd4Xor1ChannelIndex),
            HCCL_E_INTERNAL);
    }

    ChannelSet layer0Channels;
    if (hasLayer0) {
        const std::vector<uint32_t> layer0PeerOrder =
            BuildEightPlusFourLayer0CyclicOrder(groups, param);
        CHK_PRT_RET(layer0PeerOrder.size() + 1 != groups.localRanks.size(),
            HCCL_ERROR("Failed to build 8+4 cyclic layer-0 order for rank %u",
                param.myRank),
            HCCL_E_INTERNAL);
        CHK_RET(AcquireChannels(
            comm, param.myRank, layer0PeerOrder, LAYER_MESH,
            registeredInputHandle, registeredInputSize, layer0Channels));
    }

    if (useDedicatedSmallKernel) {
        CHK_PRT_RET(layer1Channels.registeredRemoteInputs.size() !=
                layer1Channels.handles.size() ||
                (hasLayer0 &&
                    layer0Channels.registeredRemoteInputs.size() !=
                        layer0Channels.handles.size()),
            HCCL_ERROR("Small registered remote-address resources are incomplete"),
            HCCL_E_INTERNAL);
    }

    A84PrefixMetadata prefixMetadata;
    CHK_RET(BuildA84PrefixMetadata(groups, param, layer1Channels, prefixMetadata));

    std::vector<KernelRegistration> registrations;
    if (useTwoStage2x8 && !useDedicatedSmallKernel) {
        CHK_PRT_RET(layer0Channels.handles.size() != 7 || layer1Channels.handles.size() != 8,
            HCCL_ERROR("2x8 requires 7 layer-0 and 8 layer-1 peer channels; got %zu/%zu",
                layer0Channels.handles.size(), layer1Channels.handles.size()),
            HCCL_E_INTERNAL);

        const auto localIt =
            std::find(groups.localRanks.begin(), groups.localRanks.end(), param.myRank);
        CHK_PRT_RET(localIt == groups.localRanks.end(),
            HCCL_ERROR("Unable to locate rank %u in sorted layer-0 ranks", param.myRank),
            HCCL_E_INTERNAL);
        const size_t localIndex =
            static_cast<size_t>(std::distance(groups.localRanks.begin(), localIt));
        CHK_PRT_RET(localIndex >= groups.remoteRanks.size(),
            HCCL_ERROR("Local index %zu has no symmetric remote partner", localIndex),
            HCCL_E_INTERNAL);
        resourceCtx.partnerRank = groups.remoteRanks[localIndex];

        const auto partnerIt = std::find(layer1Channels.peers.begin(),
            layer1Channels.peers.end(), resourceCtx.partnerRank);
        CHK_PRT_RET(partnerIt == layer1Channels.peers.end(),
            HCCL_ERROR("2x8 partner rank %u has no layer-1 channel", resourceCtx.partnerRank),
            HCCL_E_INTERNAL);
        const uint32_t partnerChannelIndex =
            static_cast<uint32_t>(std::distance(layer1Channels.peers.begin(), partnerIt));

        registrations.push_back(KernelRegistration{
            "CcuM1Layer0StageKernel",
            reinterpret_cast<const void *>(ops_hccl::CcuM1Layer0StageKernel),
            MakeStageKernelArg(layer0Channels, 0),
        });
        registrations.push_back(KernelRegistration{
            "CcuM1Layer1StageKernel",
            reinterpret_cast<const void *>(ops_hccl::CcuM1Layer1StageKernel),
            MakeStageKernelArg(layer1Channels, partnerChannelIndex),
        });
    } else {
        // Preserve the v6-A direct routes for 4x1 and 8+4. Put the mandatory
        // self-copy on the less loaded direct kernel.
        const bool directCopyOnLayer0 = hasLayer0 &&
            layer0Channels.handles.size() <= layer1Channels.handles.size();

        if (hasLayer0) {
            const char *layer0Name = useDedicatedSmallKernel ?
                "CcuSmallPullLayer0Kernel" : "CcuAllGatherLayer0Kernel";
            const void *layer0Function = useDedicatedSmallKernel ?
                reinterpret_cast<const void *>(ops_hccl::CcuSmallPullLayer0Kernel) :
                reinterpret_cast<const void *>(ops_hccl::CcuAllGatherLayer0Kernel);
            registrations.push_back(KernelRegistration{
                layer0Name,
                layer0Function,
                useDedicatedSmallKernel ?
                    std::static_pointer_cast<void>(
                        MakeSmallPullKernelArg(
                            layer0Channels, directCopyOnLayer0)) :
                    std::static_pointer_cast<void>(
                        MakeDirectKernelArg(layer0Channels,
                            AllGatherKernelLayer::LAYER0,
                            directCopyOnLayer0)),
            });
        }
        const char *layer1Name = useDedicatedSmallKernel ?
            "CcuSmallPullLayer1Kernel" : "CcuAllGatherLayer1Kernel";
        const void *layer1Function = useDedicatedSmallKernel ?
            reinterpret_cast<const void *>(ops_hccl::CcuSmallPullLayer1Kernel) :
            reinterpret_cast<const void *>(ops_hccl::CcuAllGatherLayer1Kernel);
        registrations.push_back(KernelRegistration{
            layer1Name,
            layer1Function,
            useDedicatedSmallKernel ?
                std::static_pointer_cast<void>(
                    MakeSmallPullKernelArg(layer1Channels,
                        !hasLayer0 || !directCopyOnLayer0)) :
                std::static_pointer_cast<void>(
                    MakeDirectKernelArg(layer1Channels,
                        AllGatherKernelLayer::LAYER1,
                        !hasLayer0 || !directCopyOnLayer0,
                        rd4Xor1ChannelIndex, rd4Xor2ChannelIndex)),
        });

        // Register the steady-state pull kernels after the proven cold pair.
        // They reuse the Channel-owned TOKEN_XN_ID state initialized by the
        // cold kernels, but contain no token exchange or self LocalCopy.
        if (useDedicatedSmallKernel) {
            if (hasLayer0) {
                registrations.push_back(KernelRegistration{
                    "CcuSmallPullWarmLayer0Kernel",
                    reinterpret_cast<const void *>(
                        ops_hccl::CcuSmallPullWarmLayer0Kernel),
                    std::static_pointer_cast<void>(
                        MakeSmallPullKernelArg(layer0Channels, false)),
                });
            }
            registrations.push_back(KernelRegistration{
                "CcuSmallPullWarmLayer1Kernel",
                reinterpret_cast<const void *>(
                    ops_hccl::CcuSmallPullWarmLayer1Kernel),
                std::static_pointer_cast<void>(
                    MakeSmallPullKernelArg(layer1Channels, false)),
            });
        }

        if (useAsymmetricPrefix8Plus4 && !useDedicatedSmallKernel) {
            registrations.push_back(KernelRegistration{
                "CcuA84PrefixLayer0Kernel",
                reinterpret_cast<const void *>(ops_hccl::CcuA84PrefixLayer0Kernel),
                MakeA84PrefixLayer0KernelArg(layer0Channels, prefixMetadata),
            });
            registrations.push_back(KernelRegistration{
                "CcuA84PrefixLayer1Kernel",
                reinterpret_cast<const void *>(ops_hccl::CcuA84PrefixLayer1Kernel),
                MakeA84PrefixLayer1KernelArg(layer1Channels, prefixMetadata),
            });
        }
    }

    const size_t expectedKernelCount = useDedicatedSmallKernel ?
        (hasLayer0 ? 4U : 2U) :
        (useTwoStage2x8 ? 2U :
            (useAsymmetricPrefix8Plus4 ? 4U : (hasLayer0 ? 2U : 1U)));
    CHK_PRT_RET(registrations.size() != expectedKernelCount,
        HCCL_ERROR("Static topology produced %zu kernels; expected %zu",
            registrations.size(), expectedKernelCount),
        HCCL_E_INTERNAL);
    CHK_RET(RegisterKernels(comm, registrations, resourceCtx));
    return HCCL_SUCCESS;
}

} // namespace

HcclResult HcclAllGather(void *sendBuf, void *recvBuf, uint64_t sendCount,
    HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    CHK_PRT_RET(SIZE_TABLE.find(dataType) == SIZE_TABLE.end(),
        HCCL_ERROR("Only FP32 is supported"), HCCL_E_NOT_SUPPORT);

    OpParam param;
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = sendCount;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE,
        HCCL_ERROR("Unsupported rank size %u", param.rankSize), HCCL_E_NOT_SUPPORT);

    const uint64_t dataTypeSize = SIZE_TABLE.at(dataType);
    CHK_PRT_RET(sendCount > std::numeric_limits<uint64_t>::max() / dataTypeSize,
        HCCL_ERROR("Input data size overflows uint64_t"), HCCL_E_PARA);
    const uint64_t dataSize = sendCount * dataTypeSize;
    CHK_PRT_RET(dataSize != 0 &&
            static_cast<uint64_t>(param.rankSize) >
                std::numeric_limits<uint64_t>::max() / dataSize,
        HCCL_ERROR("Output data size overflows uint64_t"), HCCL_E_PARA);
    const bool useDedicatedSmallKernel = UseDedicatedSmallKernel(param);
    const bool usePf1LatinLayer1Order = UsePf1LatinLayer1Order(param);
    const uint64_t streamValue = reinterpret_cast<uint64_t>(stream);
    const int tagResult = useDedicatedSmallKernel ?
        std::snprintf(param.tag, sizeof(param.tag),
            "hccl_ag_smallpull_steady_probe_v1_%llx_%llx_%llx_%llx",
            static_cast<unsigned long long>(streamValue),
            static_cast<unsigned long long>(
                reinterpret_cast<uint64_t>(sendBuf)),
            static_cast<unsigned long long>(
                reinterpret_cast<uint64_t>(recvBuf)),
            static_cast<unsigned long long>(dataSize)) :
        (usePf1LatinLayer1Order ?
        std::snprintf(param.tag, sizeof(param.tag),
            "hccl_custom_allgather_f1_latin_%llx",
            static_cast<unsigned long long>(streamValue)) :
        std::snprintf(param.tag, sizeof(param.tag),
            "hccl_custom_allgather_f1_%llx",
            static_cast<unsigned long long>(streamValue)));
    CHK_PRT_RET(tagResult <= 0 || static_cast<size_t>(tagResult) >= sizeof(param.tag),
        HCCL_ERROR("Failed to construct operation tag"), HCCL_E_INTERNAL);

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    constexpr CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;
    void *context = nullptr;
    uint64_t contextSize = 0;
    bool contextExisted = false;
    if (HcclEngineCtxGet(comm, param.tag, ccuEngine, &context, &contextSize) == HCCL_SUCCESS) {
        contextExisted = true;
        param.resCtx = context;
        param.ctxSize = contextSize;
    } else {
        TopologyGroups groups;
        CHK_RET(QueryTopologyGroups(comm, param, groups));
        const uint32_t mainNotifyCount = IsTwoStage2x8(groups, param) ?
            MAIN_NOTIFY_COUNT_2X8 :
            (param.rankSize == 12 ? MAIN_NOTIFY_COUNT_A84_PREFIX : MAIN_NOTIFY_COUNT_DIRECT);
        CHK_RET(HcclThreadAcquireWithStream(
            comm, ccuEngine, stream, mainNotifyCount, &param.cpuThread));

        HcclMemHandle registeredInputHandle = nullptr;
        HcclMemHandle *registeredInputHandlePtr = nullptr;
        if (useDedicatedSmallKernel) {
            char memTag[HCCL_RES_TAG_MAX_LEN]{};
            const int memTagResult = std::snprintf(memTag, sizeof(memTag),
                "%sr%u_r%llx_n%llx",
                REGISTERED_SMALL_MEM_PREFIX, param.myRank,
                static_cast<unsigned long long>(
                    reinterpret_cast<uint64_t>(sendBuf)),
                static_cast<unsigned long long>(dataSize));
            CHK_PRT_RET(memTagResult <= 0 ||
                    static_cast<size_t>(memTagResult) >= sizeof(memTag),
                HCCL_ERROR("Failed to construct registered-small memory tag"),
                HCCL_E_INTERNAL);

            CommMem inputMem{
                COMM_MEM_TYPE_DEVICE, sendBuf, dataSize};
            CHK_RET(HcclCommMemReg(
                comm, memTag, &inputMem, &registeredInputHandle));
            registeredInputHandlePtr = &registeredInputHandle;
        }

        AlgResourceCtx resourceCtx;
        CHK_RET(AllocateResources(comm, param, registeredInputHandlePtr,
            useDedicatedSmallKernel ? dataSize : 0, resourceCtx));

        std::vector<char> sequence = resourceCtx.Serialize();
        param.ctxSize = sequence.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, ccuEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, ccuEngine, param.tag, sequence.data(),
            static_cast<uint64_t>(sequence.size()), 0));
    }

    // Probe contract: on the first small-message call, enqueue the complete
    // cold path followed by the independent warm path.  Later exact context
    // hits execute only warm.  The performance package changes the first-call
    // value from 2 to 0; all five source files otherwise stay identical.
    if (useDedicatedSmallKernel) {
        param.root = contextExisted ? 1U : 0U;
    }

    CHK_RET(ops_hccl::ExecOp(param));
    return HCCL_SUCCESS;
}
