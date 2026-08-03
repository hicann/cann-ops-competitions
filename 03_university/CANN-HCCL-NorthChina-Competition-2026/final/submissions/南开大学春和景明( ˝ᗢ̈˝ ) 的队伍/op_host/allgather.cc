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
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_ccu_res.h>
#include <ccu/ccu_launch.h>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "exec_op.h"
#include "ccu_kernel.h"

namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;
constexpr uint32_t MAX_PHYSICAL_DIE_THREAD_COUNT = 2;
constexpr uint64_t PARALLEL_DIE_THRESHOLD_BYTES = 4ULL * 1024ULL * 1024ULL;
constexpr uint64_t PARALLEL_OVERSIZED_THRESHOLD_BYTES = 2ULL * MAX_DATA_SIZE;

struct ChannelGroup {
    KernelTrafficClass trafficClass = KernelTrafficClass::INTRA_SERVER;
    uint32_t layerOrder = 0;
    uint32_t netLayer = 0;
    uint32_t dieId = 0;
    std::vector<ChannelHandle> channels;
    std::vector<uint32_t> peerRanks;
    std::vector<uint32_t> logicalLocalRanks;
    std::vector<uint32_t> logicalInterRanks;
    std::vector<uint32_t> gatewaySourceRanks;
    uint32_t gatewayTargetRank = INVALID_VALUE_RANKID;
    uint32_t localPeerCount = 0;
    bool hasGatewayTarget = false;
    bool copyLocal = false;
};

uint32_t Gcd(uint32_t lhs, uint32_t rhs)
{
    while (rhs != 0) {
        const uint32_t remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }
    return lhs;
}

HcclResult FillChannelDesc(uint32_t remoteRank, const CommLink &link, HcclChannelDesc &desc)
{
    CHK_RET(HcclChannelDescInit(&desc, 1));
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

HcclResult FindChannelDescOnLayer(HcclComm comm, uint32_t netLayer, uint32_t localRank,
    uint32_t remoteRank, HcclChannelDesc &desc)
{
    CommLink *links = nullptr;
    uint32_t linkCount = 0;
    HcclResult linkRet = HcclRankGraphGetLinks(comm, netLayer, localRank, remoteRank, &links, &linkCount);
    if (linkRet == HCCL_E_NOT_FOUND) {
        return HCCL_E_NOT_FOUND;
    }
    CHK_RET(linkRet);
    if (linkCount == 0) {
        return HCCL_E_NOT_FOUND;
    }
    if (links == nullptr) {
        return HCCL_E_INTERNAL;
    }

    for (uint32_t linkIndex = 0; linkIndex < linkCount; ++linkIndex) {
        if (links[linkIndex].linkAttr.linkProtocol == CommProtocol::COMM_PROTOCOL_UBC_CTP) {
            return FillChannelDesc(remoteRank, links[linkIndex], desc);
        }
    }
    return HCCL_E_NOT_FOUND;
}

HcclResult GetEndpointDieId(HcclComm comm, uint32_t localRank,
    const EndpointDesc &endpoint, uint32_t &dieId)
{
    EndpointAttrDieId endpointDieId{};
    CHK_RET(HcclRankGraphGetEndpointInfo(comm, localRank, &endpoint,
        ENDPOINT_ATTR_DIE_ID, sizeof(endpointDieId), &endpointDieId));
    dieId = endpointDieId;
    return HCCL_SUCCESS;
}

HcclResult GetLinkDieId(HcclComm comm, uint32_t localRank, const CommLink &link, uint32_t &dieId)
{
    return GetEndpointDieId(comm, localRank, link.srcEndpointDesc, dieId);
}

HcclResult CollectUniformDieChannelDescsOnLayer(HcclComm comm, uint32_t netLayer, uint32_t localRank,
    const std::vector<uint32_t> &ranks, std::vector<HcclChannelDesc> &descs,
    std::vector<uint32_t> &peerRanks, uint32_t &selectedDieId)
{
    descs.clear();
    peerRanks.clear();
    selectedDieId = 0;
    std::vector<uint32_t> remoteRanks;
    for (uint32_t remoteRank : ranks) {
        if (remoteRank != localRank) {
            remoteRanks.push_back(remoteRank);
        }
    }
    if (remoteRanks.empty()) {
        return HCCL_E_NOT_FOUND;
    }

    CommLink *firstLinks = nullptr;
    uint32_t firstLinkCount = 0;
    HcclResult firstRet = HcclRankGraphGetLinks(
        comm, netLayer, localRank, remoteRanks.front(), &firstLinks, &firstLinkCount);
    if (firstRet == HCCL_E_NOT_FOUND || firstLinkCount == 0) {
        return HCCL_E_NOT_FOUND;
    }
    CHK_RET(firstRet);
    if (firstLinks == nullptr) {
        return HCCL_E_INTERNAL;
    }

    std::vector<uint32_t> candidateDieIds;
    for (uint32_t linkIndex = 0; linkIndex < firstLinkCount; ++linkIndex) {
        if (firstLinks[linkIndex].linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_UBC_CTP) {
            continue;
        }
        uint32_t dieId = 0;
        CHK_RET(GetLinkDieId(comm, localRank, firstLinks[linkIndex], dieId));
        if (std::find(candidateDieIds.begin(), candidateDieIds.end(), dieId) == candidateDieIds.end()) {
            candidateDieIds.push_back(dieId);
        }
    }
    std::sort(candidateDieIds.begin(), candidateDieIds.end());

    for (uint32_t candidateDieId : candidateDieIds) {
        std::vector<HcclChannelDesc> candidateDescs;
        bool complete = true;
        for (uint32_t remoteRank : remoteRanks) {
            CommLink *links = nullptr;
            uint32_t linkCount = 0;
            HcclResult linkRet = HcclRankGraphGetLinks(
                comm, netLayer, localRank, remoteRank, &links, &linkCount);
            if (linkRet == HCCL_E_NOT_FOUND || linkCount == 0) {
                complete = false;
                break;
            }
            CHK_RET(linkRet);
            if (links == nullptr) {
                return HCCL_E_INTERNAL;
            }

            const CommLink *selectedLink = nullptr;
            for (uint32_t linkIndex = 0; linkIndex < linkCount; ++linkIndex) {
                if (links[linkIndex].linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                    continue;
                }
                uint32_t dieId = 0;
                CHK_RET(GetLinkDieId(comm, localRank, links[linkIndex], dieId));
                if (dieId == candidateDieId) {
                    selectedLink = &links[linkIndex];
                    break;
                }
            }
            if (selectedLink == nullptr) {
                complete = false;
                break;
            }
            HcclChannelDesc desc;
            CHK_RET(FillChannelDesc(remoteRank, *selectedLink, desc));
            candidateDescs.push_back(desc);
        }
        if (complete && candidateDescs.size() == remoteRanks.size()) {
            descs = std::move(candidateDescs);
            peerRanks = std::move(remoteRanks);
            selectedDieId = candidateDieId;
            return HCCL_SUCCESS;
        }
    }
    return HCCL_E_NOT_FOUND;
}

HcclResult CheckGroupsUseUniformDie(HcclComm comm, uint32_t netLayer, uint32_t myRank,
    const std::vector<std::vector<uint32_t>> &groups, bool &supported)
{
    supported = false;
    for (const std::vector<uint32_t> &group : groups) {
        if (std::find(group.begin(), group.end(), myRank) == group.end()) {
            continue;
        }
        std::vector<HcclChannelDesc> descs;
        std::vector<uint32_t> peerRanks;
        uint32_t dieId = 0;
        HcclResult groupRet = CollectUniformDieChannelDescsOnLayer(
            comm, netLayer, myRank, group, descs, peerRanks, dieId);
        if (groupRet == HCCL_E_NOT_FOUND) {
            return HCCL_SUCCESS;
        }
        CHK_RET(groupRet);
        supported = peerRanks.size() + 1 == group.size();
        return HCCL_SUCCESS;
    }
    return HCCL_SUCCESS;
}

HcclResult FindUniformInterLayer(HcclComm comm, const std::vector<uint32_t> &netLayers,
    uint32_t localLayerOrder, uint32_t myRank,
    const std::vector<std::vector<uint32_t>> &interGroups,
    uint32_t &interLayerOrder, uint32_t &interNetLayer)
{
    const uint32_t layerCount = static_cast<uint32_t>(netLayers.size());
    interLayerOrder = layerCount;
    interNetLayer = 0;
    for (uint32_t layerOrder = 0; layerOrder < layerCount; ++layerOrder) {
        if (layerOrder == localLayerOrder) {
            continue;
        }
        bool interGroupsSupported = false;
        CHK_RET(CheckGroupsUseUniformDie(
            comm, netLayers[layerOrder], myRank, interGroups, interGroupsSupported));
        if (interGroupsSupported) {
            interLayerOrder = layerOrder;
            interNetLayer = netLayers[layerOrder];
            return HCCL_SUCCESS;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AcquireChannelGroup(HcclComm comm, uint32_t layerOrder, uint32_t netLayer,
    const std::vector<HcclChannelDesc> &descs, const std::vector<uint32_t> &peerRanks,
    uint32_t dieId, ChannelGroup &group)
{
    if (descs.size() != peerRanks.size()) {
        return HCCL_E_INTERNAL;
    }
    group = ChannelGroup{};
    group.layerOrder = layerOrder;
    group.netLayer = netLayer;
    group.dieId = dieId;
    group.peerRanks = peerRanks;
    for (const HcclChannelDesc &desc : descs) {
        ChannelHandle channel = 0;
        CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, &desc, 1, &channel));
        group.channels.push_back(channel);
    }
    return HCCL_SUCCESS;
}

HcclResult TryBuildPhysicalGridChannelGroups(HcclComm comm, const OpParam &param,
    uint32_t localLayerOrder, uint32_t localNetLayer, uint32_t interLayerOrder,
    uint32_t interNetLayer, const std::vector<uint32_t> &localRanks,
    const std::vector<uint32_t> &interRanks, std::vector<ChannelGroup> &channelGroups,
    bool &built)
{
    built = false;
    if (localRanks.size() < 2 || interRanks.size() < 2 ||
        localRanks.size() * interRanks.size() != param.rankSize) {
        return HCCL_SUCCESS;
    }

    struct PendingDieGroup {
        uint32_t dieId = 0;
        std::vector<HcclChannelDesc> descs;
        std::vector<uint32_t> peerRanks;
    };
    auto collectByDie = [&](uint32_t netLayer, const std::vector<uint32_t> &ranks,
                            std::vector<PendingDieGroup> &groups) -> HcclResult {
        groups.clear();
        for (uint32_t remoteRank : ranks) {
            if (remoteRank == param.myRank) {
                continue;
            }
            HcclChannelDesc desc;
            HcclResult descRet = FindChannelDescOnLayer(
                comm, netLayer, param.myRank, remoteRank, desc);
            if (descRet != HCCL_SUCCESS) {
                return descRet;
            }
            uint32_t dieId = 0;
            CHK_RET(GetEndpointDieId(comm, param.myRank, desc.localEndpoint, dieId));
            auto groupIt = std::find_if(groups.begin(), groups.end(),
                [dieId](const PendingDieGroup &group) { return group.dieId == dieId; });
            if (groupIt == groups.end()) {
                groups.push_back(PendingDieGroup{});
                groupIt = groups.end() - 1;
                groupIt->dieId = dieId;
            }
            groupIt->descs.push_back(desc);
            groupIt->peerRanks.push_back(remoteRank);
        }
        std::sort(groups.begin(), groups.end(),
            [](const PendingDieGroup &lhs, const PendingDieGroup &rhs) {
                if (lhs.peerRanks.size() != rhs.peerRanks.size()) {
                    return lhs.peerRanks.size() < rhs.peerRanks.size();
                }
                return lhs.dieId < rhs.dieId;
            });
        return HCCL_SUCCESS;
    };

    std::vector<PendingDieGroup> intraDieGroups;
    std::vector<PendingDieGroup> interDieGroups;
    HcclResult collectRet = collectByDie(localNetLayer, localRanks, intraDieGroups);
    if (collectRet == HCCL_E_NOT_FOUND) {
        return HCCL_SUCCESS;
    }
    CHK_RET(collectRet);
    collectRet = collectByDie(interNetLayer, interRanks, interDieGroups);
    if (collectRet == HCCL_E_NOT_FOUND) {
        return HCCL_SUCCESS;
    }
    CHK_RET(collectRet);
    if (intraDieGroups.empty() || interDieGroups.empty() ||
        intraDieGroups.size() > 2 || interDieGroups.size() > 2 ||
        intraDieGroups.size() + interDieGroups.size() > AlgResourceCtx::MAX_KERNEL_COUNT) {
        return HCCL_SUCCESS;
    }
    size_t intraPeerCount = 0;
    size_t interPeerCount = 0;
    for (const PendingDieGroup &group : intraDieGroups) {
        intraPeerCount += group.peerRanks.size();
        if (group.peerRanks.size() * interRanks.size() > 16) {
            return HCCL_SUCCESS;
        }
    }
    for (const PendingDieGroup &group : interDieGroups) {
        interPeerCount += group.peerRanks.size();
        if (group.peerRanks.size() * localRanks.size() > 16) {
            return HCCL_SUCCESS;
        }
    }
    if (intraPeerCount + 1 != localRanks.size() ||
        interPeerCount + 1 != interRanks.size()) {
        return HCCL_SUCCESS;
    }

    channelGroups.clear();
    channelGroups.reserve(intraDieGroups.size() + interDieGroups.size());
    // Queue the high-latency Clos work first.  Kernels still run on independent
    // threads, but this lets its launch reach the device before the Mesh work.
    for (const PendingDieGroup &pending : interDieGroups) {
        ChannelGroup group;
        CHK_RET(AcquireChannelGroup(comm, interLayerOrder, interNetLayer,
            pending.descs, pending.peerRanks, pending.dieId, group));
        group.trafficClass = KernelTrafficClass::INTER_SERVER;
        group.logicalLocalRanks = localRanks;
        group.logicalInterRanks = interRanks;
        channelGroups.push_back(std::move(group));
    }
    const size_t firstIntraGroup = channelGroups.size();
    for (const PendingDieGroup &pending : intraDieGroups) {
        ChannelGroup group;
        CHK_RET(AcquireChannelGroup(comm, localLayerOrder, localNetLayer,
            pending.descs, pending.peerRanks, pending.dieId, group));
        group.trafficClass = KernelTrafficClass::INTRA_SERVER;
        group.logicalLocalRanks = localRanks;
        group.logicalInterRanks = interRanks;
        group.localPeerCount = static_cast<uint32_t>(pending.peerRanks.size());
        channelGroups.push_back(std::move(group));
    }
    if (channelGroups.size() != intraDieGroups.size() + interDieGroups.size()) {
        return HCCL_E_INTERNAL;
    }
    // The lightest kernel in each physical class owns that class' local copy.
    channelGroups[0].copyLocal = true;
    channelGroups[firstIntraGroup].copyLocal = true;
    HCCL_INFO("[AllGather] enable pure physical grid, rank[%u], localRanks[%zu], "
        "interRanks[%zu], intraKernels[%zu], interKernels[%zu]", param.myRank,
        localRanks.size(), interRanks.size(), intraDieGroups.size(), interDieGroups.size());
    built = true;
    return HCCL_SUCCESS;
}

HcclResult TryBuildPhysical8Plus4GatewayGroups(HcclComm comm, const OpParam &param,
    uint32_t localLayerOrder, uint32_t localNetLayer, uint32_t interLayerOrder,
    uint32_t interNetLayer, const std::vector<uint32_t> &physicalLocalRanks,
    const std::vector<uint32_t> &globalRanks, std::vector<ChannelGroup> &channelGroups,
    bool &built)
{
    built = false;
    if (param.rankSize != 12 || globalRanks.size() != param.rankSize ||
        (physicalLocalRanks.size() != 4 && physicalLocalRanks.size() != 8) ||
        std::find(physicalLocalRanks.begin(), physicalLocalRanks.end(), param.myRank) ==
            physicalLocalRanks.end()) {
        return HCCL_SUCCESS;
    }

    std::vector<uint32_t> remoteRanks;
    for (uint32_t rank : globalRanks) {
        if (std::find(physicalLocalRanks.begin(), physicalLocalRanks.end(), rank) ==
            physicalLocalRanks.end()) {
            remoteRanks.push_back(rank);
        }
    }
    if (physicalLocalRanks.size() + remoteRanks.size() != param.rankSize ||
        (remoteRanks.size() != 4 && remoteRanks.size() != 8)) {
        return HCCL_SUCCESS;
    }

    const auto myLocalIt = std::find(
        physicalLocalRanks.begin(), physicalLocalRanks.end(), param.myRank);
    const size_t myLocalIndex = static_cast<size_t>(myLocalIt - physicalLocalRanks.begin());
    const uint32_t gatewayTargetRank = remoteRanks[myLocalIndex % remoteRanks.size()];
    std::vector<uint32_t> gatewaySourceRanks;
    for (size_t sourceIndex = 0; sourceIndex < remoteRanks.size(); ++sourceIndex) {
        if (sourceIndex % physicalLocalRanks.size() == myLocalIndex) {
            gatewaySourceRanks.push_back(remoteRanks[sourceIndex]);
        }
    }
    if (gatewaySourceRanks.size() > 2) {
        return HCCL_SUCCESS;
    }

    std::vector<uint32_t> interPeers = gatewaySourceRanks;
    interPeers.push_back(gatewayTargetRank);
    std::sort(interPeers.begin(), interPeers.end());
    interPeers.erase(std::unique(interPeers.begin(), interPeers.end()), interPeers.end());

    struct PendingDieGroup {
        uint32_t dieId = 0;
        std::vector<HcclChannelDesc> descs;
        std::vector<uint32_t> peerRanks;
    };
    auto collectByDie = [&](uint32_t netLayer, const std::vector<uint32_t> &ranks,
                            std::vector<PendingDieGroup> &groups) -> HcclResult {
        groups.clear();
        for (uint32_t remoteRank : ranks) {
            if (remoteRank == param.myRank) {
                continue;
            }
            HcclChannelDesc desc;
            HcclResult descRet = FindChannelDescOnLayer(
                comm, netLayer, param.myRank, remoteRank, desc);
            if (descRet != HCCL_SUCCESS) {
                return descRet;
            }
            uint32_t dieId = 0;
            CHK_RET(GetEndpointDieId(comm, param.myRank, desc.localEndpoint, dieId));
            auto groupIt = std::find_if(groups.begin(), groups.end(),
                [dieId](const PendingDieGroup &group) { return group.dieId == dieId; });
            if (groupIt == groups.end()) {
                groups.push_back(PendingDieGroup{});
                groupIt = groups.end() - 1;
                groupIt->dieId = dieId;
            }
            groupIt->descs.push_back(desc);
            groupIt->peerRanks.push_back(remoteRank);
        }
        std::sort(groups.begin(), groups.end(),
            [](const PendingDieGroup &lhs, const PendingDieGroup &rhs) {
                if (lhs.peerRanks.size() != rhs.peerRanks.size()) {
                    return lhs.peerRanks.size() < rhs.peerRanks.size();
                }
                return lhs.dieId < rhs.dieId;
            });
        return HCCL_SUCCESS;
    };

    std::vector<PendingDieGroup> intraDieGroups;
    std::vector<PendingDieGroup> interDieGroups;
    HcclResult collectRet = collectByDie(localNetLayer, physicalLocalRanks, intraDieGroups);
    if (collectRet == HCCL_E_NOT_FOUND) {
        return HCCL_SUCCESS;
    }
    CHK_RET(collectRet);
    collectRet = collectByDie(interNetLayer, interPeers, interDieGroups);
    if (collectRet == HCCL_E_NOT_FOUND) {
        return HCCL_SUCCESS;
    }
    CHK_RET(collectRet);
    if (intraDieGroups.empty() || interDieGroups.empty() ||
        intraDieGroups.size() > 2 || interDieGroups.size() > 2 ||
        intraDieGroups.size() + interDieGroups.size() > AlgResourceCtx::MAX_KERNEL_COUNT) {
        return HCCL_SUCCESS;
    }
    for (const PendingDieGroup &group : intraDieGroups) {
        if (group.peerRanks.size() * gatewaySourceRanks.size() > 15) {
            return HCCL_SUCCESS;
        }
    }

    channelGroups.clear();
    channelGroups.reserve(intraDieGroups.size() + interDieGroups.size());
    uint32_t targetGroupCount = 0;
    for (const PendingDieGroup &pending : interDieGroups) {
        ChannelGroup group;
        CHK_RET(AcquireChannelGroup(comm, interLayerOrder, interNetLayer,
            pending.descs, pending.peerRanks, pending.dieId, group));
        group.trafficClass = KernelTrafficClass::INTER_SERVER;
        group.logicalLocalRanks = physicalLocalRanks;
        group.logicalInterRanks = remoteRanks;
        group.gatewaySourceRanks = gatewaySourceRanks;
        group.gatewayTargetRank = gatewayTargetRank;
        group.hasGatewayTarget =
            std::find(group.peerRanks.begin(), group.peerRanks.end(), gatewayTargetRank) !=
            group.peerRanks.end();
        targetGroupCount += group.hasGatewayTarget ? 1U : 0U;
        channelGroups.push_back(std::move(group));
    }
    const size_t firstIntraGroup = channelGroups.size();
    for (const PendingDieGroup &pending : intraDieGroups) {
        ChannelGroup group;
        CHK_RET(AcquireChannelGroup(comm, localLayerOrder, localNetLayer,
            pending.descs, pending.peerRanks, pending.dieId, group));
        group.trafficClass = KernelTrafficClass::INTRA_SERVER;
        group.logicalLocalRanks = physicalLocalRanks;
        group.logicalInterRanks = remoteRanks;
        group.gatewaySourceRanks = gatewaySourceRanks;
        group.gatewayTargetRank = gatewayTargetRank;
        group.localPeerCount = static_cast<uint32_t>(group.peerRanks.size());
        channelGroups.push_back(std::move(group));
    }
    if (targetGroupCount != 1 || firstIntraGroup >= channelGroups.size()) {
        channelGroups.clear();
        return HCCL_SUCCESS;
    }
    channelGroups[firstIntraGroup].copyLocal = true;
    HCCL_INFO("[AllGather] enable physical 8+4 gateway hierarchy, rank[%u], local[%zu], "
        "remote[%zu], gatewaySources[%zu], kernels[%zu]", param.myRank,
        physicalLocalRanks.size(), remoteRanks.size(), gatewaySourceRanks.size(),
        channelGroups.size());
    built = true;
    return HCCL_SUCCESS;
}

HcclResult BuildHierarchicalGridGroups(HcclComm comm, const OpParam &param,
    std::vector<ChannelGroup> &channelGroups, bool &hierarchicalGrid,
    bool &regular2x8, bool &purePhysicalGrid, bool &physical8Plus4)
{
    hierarchicalGrid = false;
    regular2x8 = false;
    purePhysicalGrid = false;
    physical8Plus4 = false;
    channelGroups.clear();

    uint32_t *layers = nullptr;
    uint32_t layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layers, &layerCount));
    if (layers == nullptr || layerCount < 2) {
        return HCCL_SUCCESS;
    }
    const std::vector<uint32_t> netLayers(layers, layers + layerCount);

    uint32_t localLayerOrder = layerCount;
    uint32_t localNetLayer = 0;
    std::vector<uint32_t> physicalInstanceSizes;
    bool uniformGrid = false;
    for (uint32_t layerOrder = 0; layerOrder < layerCount; ++layerOrder) {
        CommTopo topoType{};
        HcclResult topoRet = HcclRankGraphGetTopoTypeByLayer(
            comm, netLayers[layerOrder], &topoType);
        if (topoRet != HCCL_SUCCESS || topoType != CommTopo::COMM_TOPO_1DMESH) {
            continue;
        }
        uint32_t *instSizeList = nullptr;
        uint32_t listSize = 0;
        HcclResult sizeRet = HcclRankGraphGetInstSizeListByLayer(
            comm, netLayers[layerOrder], &instSizeList, &listSize);
        if (sizeRet != HCCL_SUCCESS || instSizeList == nullptr || listSize == 0) {
            continue;
        }
        std::vector<uint32_t> sizes(instSizeList, instSizeList + listSize);
        uint32_t totalRankCount = 0;
        bool validSizes = true;
        for (uint32_t size : sizes) {
            if (size == 0 || size > param.rankSize ||
                totalRankCount > param.rankSize - size) {
                validSizes = false;
                break;
            }
            totalRankCount += size;
        }
        if (!validSizes || totalRankCount != param.rankSize) {
            continue;
        }
        std::sort(sizes.begin(), sizes.end());
        const bool supported2x8 = sizes.size() == 2 &&
            sizes[0] == 8 && sizes[1] == 8 && param.rankSize == 16;
        const bool supported8plus4 = sizes.size() == 2 &&
            sizes[0] == 4 && sizes[1] == 8 && param.rankSize == 12;
        const bool supportedUniformGrid = sizes.size() >= 2 && sizes.size() <= 4 &&
            sizes.front() >= 2 &&
            std::all_of(sizes.begin(), sizes.end(),
                [&sizes](uint32_t size) { return size == sizes.front(); });
        if (supportedUniformGrid || supported8plus4) {
            localLayerOrder = layerOrder;
            localNetLayer = netLayers[layerOrder];
            physicalInstanceSizes = sizes;
            regular2x8 = supported2x8;
            uniformGrid = supportedUniformGrid;
            break;
        }
    }
    if (localLayerOrder == layerCount) {
        return HCCL_SUCCESS;
    }

    uint32_t subgroupSize = physicalInstanceSizes[0];
    if (!uniformGrid) {
        for (uint32_t size : physicalInstanceSizes) {
            subgroupSize = Gcd(subgroupSize, size);
        }
    }
    if (subgroupSize <= 1) {
        return HCCL_SUCCESS;
    }

    uint32_t *localInstanceRankList = nullptr;
    uint32_t localInstanceRankCount = 0;
    CHK_RET(HcclRankGraphGetRanksByLayer(
        comm, localNetLayer, &localInstanceRankList, &localInstanceRankCount));
    if (localInstanceRankList == nullptr || localInstanceRankCount < subgroupSize ||
        localInstanceRankCount % subgroupSize != 0) {
        return HCCL_SUCCESS;
    }
    std::vector<uint32_t> localInstanceRanks(
        localInstanceRankList, localInstanceRankList + localInstanceRankCount);
    std::sort(localInstanceRanks.begin(), localInstanceRanks.end());
    auto myLocalIt = std::find(localInstanceRanks.begin(), localInstanceRanks.end(), param.myRank);
    if (myLocalIt == localInstanceRanks.end()) {
        return HCCL_SUCCESS;
    }
    const size_t myLocalIndex = static_cast<size_t>(myLocalIt - localInstanceRanks.begin());
    const size_t localSubgroupStart = (myLocalIndex / subgroupSize) * subgroupSize;
    if (localSubgroupStart + subgroupSize > localInstanceRanks.size()) {
        return HCCL_SUCCESS;
    }
    std::vector<uint32_t> localRanks(
        localInstanceRanks.begin() + localSubgroupStart,
        localInstanceRanks.begin() + localSubgroupStart + subgroupSize);
    std::vector<std::vector<uint32_t>> subgroups{localRanks};

    uint32_t interLayerOrder = layerCount;
    uint32_t interNetLayer = 0;
    std::vector<uint32_t> globalRanks;
    std::vector<uint32_t> interRanks;
    for (uint32_t layerOrder = 0; layerOrder < layerCount; ++layerOrder) {
        if (layerOrder == localLayerOrder) {
            continue;
        }
        CommTopo topoType{};
        HcclResult topoRet = HcclRankGraphGetTopoTypeByLayer(
            comm, netLayers[layerOrder], &topoType);
        if (topoRet != HCCL_SUCCESS || topoType != CommTopo::COMM_TOPO_CLOS) {
            continue;
        }
        uint32_t *candidateRanks = nullptr;
        uint32_t candidateRankCount = 0;
        HcclResult ranksRet = HcclRankGraphGetRanksByLayer(
            comm, netLayers[layerOrder], &candidateRanks, &candidateRankCount);
        if (ranksRet != HCCL_SUCCESS || candidateRanks == nullptr ||
            candidateRankCount != param.rankSize) {
            continue;
        }
        std::vector<uint32_t> candidateGlobalRanks(
            candidateRanks, candidateRanks + candidateRankCount);
        std::sort(candidateGlobalRanks.begin(), candidateGlobalRanks.end());
        auto myGlobalIt = std::find(
            candidateGlobalRanks.begin(), candidateGlobalRanks.end(), param.myRank);
        if (myGlobalIt == candidateGlobalRanks.end()) {
            continue;
        }
        const size_t myGlobalIndex = static_cast<size_t>(
            myGlobalIt - candidateGlobalRanks.begin());
        if (uniformGrid) {
            if (candidateGlobalRanks.size() % subgroupSize != 0) {
                continue;
            }
            const size_t localBlockStart = (myGlobalIndex / subgroupSize) * subgroupSize;
            const std::vector<uint32_t> expectedLocalRanks(
                candidateGlobalRanks.begin() + localBlockStart,
                candidateGlobalRanks.begin() + localBlockStart + subgroupSize);
            if (expectedLocalRanks != localRanks) {
                continue;
            }
            bool meshBlocksValid = true;
            for (size_t blockStart = 0; blockStart < candidateGlobalRanks.size();
                blockStart += subgroupSize) {
                for (size_t srcOffset = 0; srcOffset < subgroupSize && meshBlocksValid; ++srcOffset) {
                    for (size_t dstOffset = 0; dstOffset < subgroupSize; ++dstOffset) {
                        if (srcOffset == dstOffset) {
                            continue;
                        }
                        HcclChannelDesc validationDesc;
                        HcclResult validationRet = FindChannelDescOnLayer(
                            comm, localNetLayer,
                            candidateGlobalRanks[blockStart + srcOffset],
                            candidateGlobalRanks[blockStart + dstOffset], validationDesc);
                        if (validationRet != HCCL_SUCCESS) {
                            meshBlocksValid = false;
                            break;
                        }
                    }
                }
            }
            if (!meshBlocksValid) {
                continue;
            }
        }
        const size_t myColumnIndex = myGlobalIndex % subgroupSize;
        std::vector<uint32_t> candidateInterRanks;
        for (size_t rankIndex = myColumnIndex;
            rankIndex < candidateGlobalRanks.size(); rankIndex += subgroupSize) {
            candidateInterRanks.push_back(candidateGlobalRanks[rankIndex]);
        }
        interLayerOrder = layerOrder;
        interNetLayer = netLayers[layerOrder];
        globalRanks = std::move(candidateGlobalRanks);
        interRanks = std::move(candidateInterRanks);
        break;
    }
    if (interLayerOrder == layerCount) {
        return HCCL_SUCCESS;
    }

    if (!uniformGrid) {
        bool gatewayBuilt = false;
        CHK_RET(TryBuildPhysical8Plus4GatewayGroups(comm, param,
            localLayerOrder, localNetLayer, interLayerOrder, interNetLayer,
            localInstanceRanks, globalRanks, channelGroups, gatewayBuilt));
        if (gatewayBuilt) {
            hierarchicalGrid = true;
            physical8Plus4 = true;
            return HCCL_SUCCESS;
        }
    }

    std::vector<std::vector<uint32_t>> interGroups(subgroupSize);
    for (size_t columnIndex = 0; columnIndex < subgroupSize; ++columnIndex) {
        for (size_t rankIndex = columnIndex;
            rankIndex < globalRanks.size(); rankIndex += subgroupSize) {
            interGroups[columnIndex].push_back(globalRanks[rankIndex]);
        }
    }

    if (uniformGrid) {
        bool multiDieBuilt = false;
        CHK_RET(TryBuildPhysicalGridChannelGroups(comm, param,
            localLayerOrder, localNetLayer, interLayerOrder, interNetLayer,
            localRanks, interRanks, channelGroups, multiDieBuilt));
        if (multiDieBuilt) {
            hierarchicalGrid = true;
            purePhysicalGrid = true;
            return HCCL_SUCCESS;
        }
    } else {
        CHK_RET(FindUniformInterLayer(comm, netLayers, localLayerOrder,
            param.myRank, interGroups, interLayerOrder, interNetLayer));
        if (interLayerOrder == layerCount) {
            return HCCL_SUCCESS;
        }
    }

    bool localGroupsSupported = false;
    CHK_RET(CheckGroupsUseUniformDie(
        comm, localNetLayer, param.myRank, subgroups, localGroupsSupported));
    if (!localGroupsSupported) {
        return HCCL_SUCCESS;
    }
    bool interGroupsUniform = false;
    CHK_RET(CheckGroupsUseUniformDie(
        comm, interNetLayer, param.myRank, interGroups, interGroupsUniform));
    if (!interGroupsUniform) {
        return HCCL_SUCCESS;
    }

    std::vector<HcclChannelDesc> localDescs;
    std::vector<uint32_t> localPeers;
    uint32_t localDieId = 0;
    CHK_RET(CollectUniformDieChannelDescsOnLayer(
        comm, localNetLayer, param.myRank, localRanks, localDescs, localPeers, localDieId));
    std::vector<HcclChannelDesc> interDescs;
    std::vector<uint32_t> interPeers;
    uint32_t interDieId = 0;
    CHK_RET(CollectUniformDieChannelDescsOnLayer(
        comm, interNetLayer, param.myRank, interRanks, interDescs, interPeers, interDieId));

    channelGroups.resize(2);
    CHK_RET(AcquireChannelGroup(
        comm, localLayerOrder, localNetLayer, localDescs, localPeers, localDieId, channelGroups[0]));
    CHK_RET(AcquireChannelGroup(
        comm, interLayerOrder, interNetLayer, interDescs, interPeers, interDieId, channelGroups[1]));
    for (ChannelGroup &group : channelGroups) {
        group.logicalLocalRanks = localRanks;
        group.logicalInterRanks = interRanks;
    }
    channelGroups[0].localPeerCount = static_cast<uint32_t>(localPeers.size());
    channelGroups[0].trafficClass = KernelTrafficClass::INTRA_SERVER;
    channelGroups[1].trafficClass = KernelTrafficClass::INTER_SERVER;
    channelGroups[0].copyLocal = true;
    channelGroups[1].copyLocal = true;
    hierarchicalGrid = true;
    return HCCL_SUCCESS;
}

HcclResult ShouldUseParallelResource(const OpParam &param, bool &useParallel)
{
    if (param.dataType != HCCL_DATA_TYPE_FP32) {
        return HCCL_E_NOT_SUPPORT;
    }
    constexpr uint64_t DATA_TYPE_SIZE = sizeof(float);
    if (param.count > std::numeric_limits<uint64_t>::max() / DATA_TYPE_SIZE) {
        return HCCL_E_PARA;
    }
    useParallel = param.count * DATA_TYPE_SIZE >= PARALLEL_DIE_THRESHOLD_BYTES;
    return HCCL_SUCCESS;
}

HcclResult BuildChannelDesc(HcclComm comm, uint32_t localRank, uint32_t remoteRank,
    HcclChannelDesc &desc, uint32_t &selectedLayerOrder, uint32_t &selectedNetLayer)
{
    uint32_t *layers = nullptr;
    uint32_t layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layers, &layerCount));
    if (layers == nullptr || layerCount == 0) {
        return HCCL_E_NOT_FOUND;
    }
    const std::vector<uint32_t> localLayers(layers, layers + layerCount);

    for (uint32_t layerIndex = 0; layerIndex < layerCount; ++layerIndex) {
        HcclResult layerRet = FindChannelDescOnLayer(
            comm, localLayers[layerIndex], localRank, remoteRank, desc);
        if (layerRet == HCCL_E_NOT_FOUND) {
            continue;
        }
        CHK_RET(layerRet);
        selectedLayerOrder = layerIndex;
        selectedNetLayer = localLayers[layerIndex];
        return HCCL_SUCCESS;
    }

    HCCL_ERROR("No CCU UBC_CTP link from rank %u to rank %u", localRank, remoteRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult TryBuildSingleLayerGroup(HcclComm comm, const OpParam &param,
    std::vector<ChannelGroup> &channelGroups, bool &singleLayerFound)
{
    singleLayerFound = false;
    uint32_t *layers = nullptr;
    uint32_t layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layers, &layerCount));
    if (layers == nullptr || layerCount == 0) {
        return HCCL_SUCCESS;
    }

    std::vector<uint32_t> allRanks;
    allRanks.reserve(param.rankSize);
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        allRanks.push_back(rank);
    }

    for (uint32_t reverseIndex = layerCount; reverseIndex > 0; --reverseIndex) {
        const uint32_t layerOrder = reverseIndex - 1;
        const uint32_t netLayer = layers[layerOrder];
        std::vector<HcclChannelDesc> descs;
        std::vector<uint32_t> peerRanks;
        uint32_t dieId = 0;
        HcclResult groupRet = CollectUniformDieChannelDescsOnLayer(
            comm, netLayer, param.myRank, allRanks, descs, peerRanks, dieId);
        if (groupRet == HCCL_E_NOT_FOUND) {
            continue;
        }
        CHK_RET(groupRet);
        if (peerRanks.size() + 1 != param.rankSize) {
            continue;
        }

        ChannelGroup group;
        CHK_RET(AcquireChannelGroup(
            comm, layerOrder, netLayer, descs, peerRanks, dieId, group));
        channelGroups.clear();
        channelGroups.push_back(std::move(group));
        singleLayerFound = true;
        return HCCL_SUCCESS;
    }
    return HCCL_SUCCESS;
}

HcclResult AcquireChannels(HcclComm comm, const OpParam &param, std::vector<ChannelGroup> &channelGroups)
{
    channelGroups.clear();
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }

        HcclChannelDesc desc;
        uint32_t layerOrder = 0;
        uint32_t netLayer = 0;
        CHK_RET(BuildChannelDesc(comm, param.myRank, remoteRank, desc, layerOrder, netLayer));
        uint32_t dieId = 0;
        CHK_RET(GetEndpointDieId(comm, param.myRank, desc.localEndpoint, dieId));
        ChannelHandle channel = 0;
        CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, &desc, 1, &channel));

        auto groupIt = std::find_if(channelGroups.begin(), channelGroups.end(),
            [layerOrder](const ChannelGroup &group) { return group.layerOrder == layerOrder; });
        if (groupIt == channelGroups.end()) {
            channelGroups.push_back(ChannelGroup{});
            groupIt = channelGroups.end() - 1;
            groupIt->layerOrder = layerOrder;
            groupIt->netLayer = netLayer;
            groupIt->dieId = dieId;
        } else if (groupIt->dieId != dieId) {
            HCCL_ERROR("CCU AllGather layer %u spans multiple local IO dies", layerOrder);
            return HCCL_E_NOT_SUPPORT;
        }
        groupIt->channels.push_back(channel);
        groupIt->peerRanks.push_back(remoteRank);
    }

    std::sort(channelGroups.begin(), channelGroups.end(),
        [](const ChannelGroup &lhs, const ChannelGroup &rhs) { return lhs.layerOrder < rhs.layerOrder; });
    return HCCL_SUCCESS;
}

HcclResult BuildNhrSchedule(const std::vector<uint32_t> &ranks, uint32_t myRank,
    ops_hccl::AllGatherKernelArg &kernelArg)
{
    if (ranks.size() < 2 || ranks.size() > MAX_RANK_SIZE) {
        return HCCL_E_PARA;
    }
    auto rankIt = std::find(ranks.begin(), ranks.end(), myRank);
    if (rankIt == ranks.end()) {
        return HCCL_E_PARA;
    }
    const uint32_t rankSize = static_cast<uint32_t>(ranks.size());
    const uint32_t rankIndex = static_cast<uint32_t>(rankIt - ranks.begin());
    uint32_t stepCount = 0;
    for (uint32_t remaining = rankSize - 1; remaining != 0; remaining >>= 1) {
        ++stepCount;
    }
    if (stepCount == 0 || stepCount > 4) {
        return HCCL_E_PARA;
    }

    auto findChannelIndex = [&kernelArg](uint32_t peerRank, uint32_t &channelIndex) {
        for (uint32_t index = 0; index < kernelArg.channelCount; ++index) {
            if (kernelArg.peerRanks[index] == peerRank) {
                channelIndex = index;
                return true;
            }
        }
        return false;
    };

    uint32_t txOffset = 0;
    for (uint32_t step = 0; step < stepCount; ++step) {
        const uint32_t deltaRank = 1U << (stepCount - 1 - step);
        const uint32_t sendIndex = (rankIndex + deltaRank) % rankSize;
        const uint32_t recvIndex = (rankIndex + rankSize - deltaRank) % rankSize;
        const uint32_t sendRank = ranks[sendIndex];
        const uint32_t recvRank = ranks[recvIndex];
        kernelArg.nhrToRanks[step] = sendRank;
        kernelArg.nhrFromRanks[step] = recvRank;
        if (!findChannelIndex(sendRank, kernelArg.nhrToChannelIndices[step]) ||
            !findChannelIndex(recvRank, kernelArg.nhrFromChannelIndices[step])) {
            return HCCL_E_INTERNAL;
        }

        kernelArg.nhrTxOffsets[step] = txOffset;
        const uint32_t sliceCount = (rankSize - 1 + deltaRank) / (deltaRank << 1);
        const uint32_t deltaSliceIndex = 1U << (stepCount - step);
        uint32_t txSliceIndex = rankIndex;
        for (uint32_t sliceIndex = 0; sliceIndex < sliceCount; ++sliceIndex) {
            if (txOffset >= MAX_RANK_SIZE) {
                return HCCL_E_INTERNAL;
            }
            kernelArg.nhrTxRanks[txOffset++] = ranks[txSliceIndex];
            txSliceIndex = (txSliceIndex + rankSize - deltaSliceIndex) % rankSize;
        }
    }
    kernelArg.nhrTxOffsets[stepCount] = txOffset;
    kernelArg.nhrStepCount = stepCount;
    kernelArg.useNhr = true;
    return HCCL_SUCCESS;
}

HcclResult RegisterKernels(HcclComm comm, const OpParam &param,
    const std::vector<ChannelGroup> &channelGroups,
    const std::vector<uint32_t> &threadDieIds,
    bool hierarchicalGrid, bool hierarchical2x8, bool purePhysicalGrid, bool physical8Plus4,
    bool latencyOptimizedBarrier, bool direct4x1Nhr, AlgResourceCtx &resource)
{
    if (channelGroups.empty() || threadDieIds.empty() ||
        threadDieIds.size() > MAX_PHYSICAL_DIE_THREAD_COUNT) {
        return HCCL_E_INTERNAL;
    }

    CcuInsHandle insHandle{0};
    uint32_t insCount = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insCount));
    if (insCount != 1) {
        HCCL_ERROR("Expected one CCU instruction instance, got %u", insCount);
        return HCCL_E_INTERNAL;
    }

    CcuResult startRet = HcommCcuKernelRegisterStart(insHandle);
    if (startRet != CCU_SUCCESS) {
        return ConvertCcuToHccl(startRet);
    }

    std::vector<uint32_t> localRanks;
    std::vector<uint32_t> interRanks;
    if (hierarchicalGrid) {
        if (!channelGroups[0].logicalLocalRanks.empty() &&
            !channelGroups[0].logicalInterRanks.empty()) {
            localRanks = channelGroups[0].logicalLocalRanks;
            interRanks = channelGroups[0].logicalInterRanks;
        } else {
            localRanks = channelGroups[0].peerRanks;
            localRanks.push_back(param.myRank);
            std::sort(localRanks.begin(), localRanks.end());
            interRanks = channelGroups[1].peerRanks;
            interRanks.push_back(param.myRank);
            std::sort(interRanks.begin(), interRanks.end());
        }
        const bool rankShapeValid = physical8Plus4 ?
            localRanks.size() + interRanks.size() == param.rankSize :
            localRanks.size() * interRanks.size() == param.rankSize;
        if (!rankShapeValid) {
            return HCCL_E_INTERNAL;
        }
    }

    std::vector<CcuKernelHandle> kernelHandles;
    kernelHandles.reserve(channelGroups.size());
    for (uint32_t groupIndex = 0; groupIndex < channelGroups.size(); ++groupIndex) {
        const ChannelGroup &group = channelGroups[groupIndex];
        CcuKernelInfo kernelInfo;
        int written = std::snprintf(
            kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "CcuKernel%u", groupIndex);
        if (written <= 0 || static_cast<size_t>(written) >= sizeof(kernelInfo.kernelFuncName)) {
            return HCCL_E_INTERNAL;
        }
        kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuKernel);

        auto kernelArg = std::make_shared<ops_hccl::AllGatherKernelArg>();
        kernelArg->rankSize = param.rankSize;
        kernelArg->rankId = param.myRank;
        kernelArg->copyLocal = group.copyLocal;
        kernelArg->hierarchical2x8 = hierarchical2x8;
        kernelArg->hierarchicalGrid = hierarchicalGrid;
        kernelArg->purePhysicalGrid = purePhysicalGrid;
        kernelArg->physical8Plus4 = physical8Plus4;
        kernelArg->latencyOptimizedBarrier = latencyOptimizedBarrier;
        if (hierarchicalGrid) {
            kernelArg->role = group.trafficClass == KernelTrafficClass::INTRA_SERVER ?
                ops_hccl::AllGatherKernelRole::INTRA_SERVER :
                ops_hccl::AllGatherKernelRole::INTER_SERVER;
            kernelArg->localRankCount = static_cast<uint32_t>(localRanks.size());
            kernelArg->interRankCount = static_cast<uint32_t>(interRanks.size());
            kernelArg->localPeerCount = group.localPeerCount;
            kernelArg->hasGatewayTarget = group.hasGatewayTarget;
            if (group.gatewaySourceRanks.size() > MAX_RANK_SIZE) {
                return HCCL_E_INTERNAL;
            }
            kernelArg->gatewaySourceCount =
                static_cast<uint32_t>(group.gatewaySourceRanks.size());
            for (uint32_t index = 0; index < kernelArg->gatewaySourceCount; ++index) {
                kernelArg->gatewaySourceRanks[index] = group.gatewaySourceRanks[index];
            }
            if (interRanks.size() == 2 &&
                kernelArg->role == ops_hccl::AllGatherKernelRole::INTER_SERVER) {
                kernelArg->pairedRank = interRanks[0] == param.myRank ? interRanks[1] : interRanks[0];
                auto pairedIt = std::find(
                    group.peerRanks.begin(), group.peerRanks.end(), kernelArg->pairedRank);
                if (pairedIt == group.peerRanks.end()) {
                    return HCCL_E_INTERNAL;
                }
                kernelArg->pairedChannelIndex = static_cast<uint32_t>(
                    pairedIt - group.peerRanks.begin());
            }
            for (uint32_t index = 0; index < kernelArg->localRankCount; ++index) {
                kernelArg->localRanks[index] = localRanks[index];
            }
            for (uint32_t index = 0; index < kernelArg->interRankCount; ++index) {
                kernelArg->interRanks[index] = interRanks[index];
            }
        }
        kernelArg->channelCount = static_cast<uint32_t>(group.channels.size());
        for (uint32_t channelIndex = 0; channelIndex < kernelArg->channelCount; ++channelIndex) {
            kernelArg->channels[channelIndex] = group.channels[channelIndex];
            kernelArg->peerRanks[channelIndex] = group.peerRanks[channelIndex];
        }
        if (group.hasGatewayTarget) {
            auto targetIt = std::find(
                group.peerRanks.begin(), group.peerRanks.end(), group.gatewayTargetRank);
            if (targetIt == group.peerRanks.end()) {
                return HCCL_E_INTERNAL;
            }
            kernelArg->gatewayTargetChannelIndex = static_cast<uint32_t>(
                targetIt - group.peerRanks.begin());
        }
        if (direct4x1Nhr && groupIndex == 0) {
            std::vector<uint32_t> nhrRanks = group.peerRanks;
            nhrRanks.push_back(param.myRank);
            std::sort(nhrRanks.begin(), nhrRanks.end());
            CHK_RET(BuildNhrSchedule(nhrRanks, param.myRank, *kernelArg));
        }
        kernelInfo.setKernelArg(kernelArg);

        CcuKernelHandle kernelHandle = 0;
        const void *kernelArgs[] = {kernelInfo.kernelArg};
        CcuResult registerRet = HcommCcuKernelRegister(insHandle, group.dieId, kernelInfo.kernelFuncName,
            kernelInfo.kernelFunc, kernelArgs, 1, &kernelHandle);
        if (registerRet != CCU_SUCCESS) {
            return ConvertCcuToHccl(registerRet);
        }
        kernelHandles.push_back(kernelHandle);
    }

    CcuResult endRet = HcommCcuKernelRegisterEnd(insHandle);
    if (endRet != CCU_SUCCESS) {
        return ConvertCcuToHccl(endRet);
    }

    if (kernelHandles.size() > AlgResourceCtx::MAX_KERNEL_COUNT) {
        return HCCL_E_INTERNAL;
    }
    resource.kernelCount = static_cast<uint32_t>(kernelHandles.size());
    for (uint32_t index = 0; index < resource.kernelCount; ++index) {
        resource.ccuKernels[index] = kernelHandles[index];
        if (resource.threadCount > 1) {
            auto threadIt = std::find(
                threadDieIds.begin(), threadDieIds.end(), channelGroups[index].dieId);
            if (threadIt == threadDieIds.end() ||
                static_cast<size_t>(threadIt - threadDieIds.begin()) >= resource.threadCount) {
                return HCCL_E_INTERNAL;
            }
            resource.kernelThreadIndex[index] = static_cast<uint8_t>(threadIt - threadDieIds.begin());
        } else {
            resource.kernelThreadIndex[index] = 0;
        }
        resource.kernelTrafficClass[index] = channelGroups[index].trafficClass;
    }
    return HCCL_SUCCESS;
}

HcclResult BuildResource(HcclComm comm, const OpParam &param, AlgResourceCtx &resource)
{
    void *cclBuffer = nullptr;
    uint64_t cclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBuffer, &cclBufferSize));
    resource.localBuffer = CommBuffer{cclBuffer, cclBufferSize};
    resource.ccuThread = param.cpuThread;
    resource.threads[0] = param.cpuThread;
    resource.threadCount = 1;
    resource.rankId = param.myRank;
    resource.rankSize = param.rankSize;

    if (param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    bool useParallel = false;
    CHK_RET(ShouldUseParallelResource(param, useParallel));
    const bool hierarchicalSizeEligible =
        param.count <= PARALLEL_OVERSIZED_THRESHOLD_BYTES / sizeof(float);
    std::vector<ChannelGroup> channelGroups;
    bool hierarchicalGrid = false;
    bool regular2x8 = false;
    bool purePhysicalGrid = false;
    bool physical8Plus4 = false;
    bool singleLayerFound = false;
    const bool useSingleClosSmall = !useParallel;
    if (useSingleClosSmall) {
        CHK_RET(TryBuildSingleLayerGroup(comm, param, channelGroups, singleLayerFound));
    }
    if (useParallel && hierarchicalSizeEligible) {
        CHK_RET(BuildHierarchicalGridGroups(
            comm, param, channelGroups, hierarchicalGrid, regular2x8,
            purePhysicalGrid, physical8Plus4));
    }
    if (!hierarchicalGrid && !singleLayerFound && channelGroups.empty()) {
        CHK_RET(AcquireChannels(comm, param, channelGroups));
    }
    if (channelGroups.size() > AlgResourceCtx::MAX_KERNEL_COUNT) {
        HCCL_ERROR("CCU AllGather supports at most %u kernels, got %zu",
            AlgResourceCtx::MAX_KERNEL_COUNT, channelGroups.size());
        return HCCL_E_NOT_SUPPORT;
    }
    std::vector<uint32_t> threadDieIds;
    for (const ChannelGroup &group : channelGroups) {
        if (std::find(threadDieIds.begin(), threadDieIds.end(), group.dieId) == threadDieIds.end()) {
            threadDieIds.push_back(group.dieId);
        }
    }
    std::sort(threadDieIds.begin(), threadDieIds.end());
    if (threadDieIds.empty() || threadDieIds.size() > MAX_PHYSICAL_DIE_THREAD_COUNT) {
        HCCL_ERROR("CCU AllGather supports at most %u physical IO-die threads, got %zu",
            MAX_PHYSICAL_DIE_THREAD_COUNT, threadDieIds.size());
        return HCCL_E_NOT_SUPPORT;
    }
    if (!hierarchicalGrid && !channelGroups.empty()) {
        channelGroups[0].copyLocal = true;
    }
    bool direct4x1Nhr = false;
    resource.directNhr = direct4x1Nhr;
    resource.hierarchical2x8 = hierarchicalGrid && regular2x8;
    resource.hierarchicalGrid = hierarchicalGrid;
    resource.purePhysicalGrid = hierarchicalGrid && purePhysicalGrid;
    resource.physical8Plus4 = hierarchicalGrid && physical8Plus4;
    // Small messages are latency-bound. Keep all topology kernels on the user
    // stream. Large messages use one thread per physical IO die for overlap.
    if (threadDieIds.size() > 1 && useParallel) {
        const uint32_t slaveThreadCount = static_cast<uint32_t>(threadDieIds.size() - 1);
        ThreadHandle slaveThreads[AlgResourceCtx::MAX_THREAD_COUNT - 1]{};
        CHK_RET(HcclThreadAcquire(comm, CommEngine::COMM_ENGINE_CCU,
            slaveThreadCount, 1, slaveThreads));
        for (uint32_t threadIndex = 0; threadIndex < slaveThreadCount; ++threadIndex) {
            resource.threads[resource.threadCount++] = slaveThreads[threadIndex];
        }
    }
    const bool directSmallFastPath = !useParallel;
    resource.directSmallFastPath = directSmallFastPath;
    return RegisterKernels(comm, param, channelGroups, threadDieIds, resource.hierarchicalGrid,
        resource.hierarchical2x8, resource.purePhysicalGrid, resource.physical8Plus4,
        directSmallFastPath, direct4x1Nhr, resource);
}
} // namespace

HcclResult HcclAllGather(
    void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    // 构造算子参数
    OpParam param;
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = sendCount;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;

    bool useParallel = false;
    CHK_RET(ShouldUseParallelResource(param, useParallel));
    const bool parallelOversized = useParallel &&
        param.count > PARALLEL_OVERSIZED_THRESHOLD_BYTES / sizeof(float);
    const char *contextTag = !useParallel ? "hccl_custom_allgather_ccu_direct_small_topology_hybrid" :
        (parallelOversized ? "hccl_custom_allgather_ccu_parallel_oversized" :
                             "hccl_custom_allgather_ccu_parallel");
    int written = std::snprintf(param.tag, sizeof(param.tag), "%s", contextTag);
    if (written <= 0 || static_cast<size_t>(written) >= sizeof(param.tag)) {
        return HCCL_E_INTERNAL;
    }

    // ==============================================
    // STEP 2: 创建资源
    // ==============================================
    CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;

    // ==============================================
    // STEP 2.1: 申请用于 Host/Device 同步的通信资源
    // ==============================================
    void *ctx = nullptr;
    uint64_t size = 0;
    AlgResourceCtx resource;
    const bool resourceExists = HcclEngineCtxGet(comm, param.tag, ccuEngine, &ctx, &size) == HCCL_SUCCESS;
    if (resourceExists) {
        // CCU 资源已经存在，复用资源
        HCCL_INFO("Engine context already exists");
        if (!resource.DeSerialize(ctx, size)) {
            HCCL_ERROR("CCU AllGather resource context is invalid");
            return HCCL_E_INTERNAL;
        }
    } else {
        CHK_RET(HcclGetCommName(comm, resource.commName));
        resource.commName[COMM_INDENTIFIER_MAX_LENGTH - 1] = '\0';
    }
    if (resource.commName[0] == '\0') {
        return HCCL_E_INTERNAL;
    }

    // 注册本轮算子信息。commName随资源缓存，避免重复查询通信域。
    HcclDfxOpInfo dfxInfo{};
    CHK_RET(HcclDfxRegOpInfoByCommId(
        resource.commName, reinterpret_cast<void *>(&dfxInfo)));

    // 将用户 stream 转为主 CCU thread，并为唯一的 slave 回传预留 notify。
    CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, stream,
        MAX_PHYSICAL_DIE_THREAD_COUNT - 1, &param.cpuThread));
    resource.ccuThread = param.cpuThread;
    resource.threads[0] = param.cpuThread;

    if (!resourceExists) {
        // Device 资源不存在，资源构建
        CHK_RET(HcclGetRankId(comm, &param.myRank));
        CHK_RET(HcclGetRankSize(comm, &param.rankSize));
        if (param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || param.myRank >= param.rankSize) {
            HCCL_ERROR("Unsupported rank configuration: rank %u of %u", param.myRank, param.rankSize);
            return HCCL_E_NOT_SUPPORT;
        }
        CHK_RET(BuildResource(comm, param, resource));

        // ==============================================
        // STEP 2.3: 申请通信引擎上下文
        // ==============================================
        // 申请 CCU 通信引擎上下文，存放 AlgResourceCtx 信息
        std::vector<char> seq = resource.Serialize();
        if (seq.empty()) {
            return HCCL_E_INTERNAL;
        }
        uint64_t seqSize = seq.size();
        param.ctxSize = seqSize;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, ccuEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, ccuEngine, param.tag, seq.data(), seqSize, 0));
    }

    // ==============================================
    // STEP 3: 下发 CCU Kernel
    // ==============================================
    CHK_RET(ops_hccl::ExecOp(param, resource));
    return HCCL_SUCCESS;
}
