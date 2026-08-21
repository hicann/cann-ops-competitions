/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */  

#include <ccu/ccu_launch.h>
#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "ccu_kernel.h"
#include "common.h"
#include "custom.h"
#include "exec_op.h"
#include "hccl.h"
#include "log.h"

namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;
constexpr uint32_t KERNEL_DIE_ID = 0;
constexpr uint32_t SINGLE_KERNEL_NUM = 1;
constexpr uint32_t DUAL_KERNEL_NUM = 3;
constexpr uint32_t PARALLEL_2X8_KERNEL_NUM = 4;
constexpr uint32_t ASYMMETRIC_8X4_KERNEL_NUM = 4;
constexpr uint64_t DUAL_LAYER_MIN_BYTES = 16ULL * 1024ULL * 1024ULL;
constexpr uint64_t LATIN_4X1_MIN_BYTES = 16ULL * 1024ULL * 1024ULL;
constexpr uint32_t LATIN_4X1_RANK_NUM = 4;
constexpr uint64_t PARALLEL_2X8_ALIGNMENT = 4ULL * 1024ULL;
constexpr uint32_t PARALLEL_2X8_SERVER_NUM = 2;
constexpr uint32_t PARALLEL_2X8_LOCAL_RANK_NUM = 8;
constexpr uint32_t ASYMMETRIC_8X4_RANK_NUM = 12;
constexpr uint32_t ASYMMETRIC_LARGE_SERVER_RANK_NUM = 8;
constexpr uint32_t ASYMMETRIC_SMALL_SERVER_RANK_NUM = 4;
constexpr uint32_t LOCAL_MESH_LAYER = 0;
constexpr uint32_t CROSS_SERVER_LAYER = 1;

bool IsCcuProtocol(CommProtocol protocol)
{
    return protocol == CommProtocol::COMM_PROTOCOL_UBC_CTP ||
        protocol == CommProtocol::COMM_PROTOCOL_UBC_TP;
}

struct LinkCandidate {
    CommLink link{};
    EndpointAttrDieId dieId = std::numeric_limits<EndpointAttrDieId>::max();
};

struct ChannelGroup {
    uint32_t netLayer = 0;
    EndpointAttrDieId dieId = std::numeric_limits<EndpointAttrDieId>::max();
    std::vector<uint32_t> peerRanks;
    std::vector<HcclChannelDesc> channelDescs;
};

struct Parallel2x8Topology {
    uint32_t serverIndex = INVALID_VALUE_RANKID;
    uint32_t localIndex = INVALID_VALUE_RANKID;
    uint32_t pairedRank = INVALID_VALUE_RANKID;
    std::array<uint32_t, MAX_RANK_SIZE> topologyRanks{};
};

struct Asymmetric8x4Topology {
    uint32_t serverIndex = INVALID_VALUE_RANKID;
    uint32_t localIndex = INVALID_VALUE_RANKID;
    uint32_t localRankCount = 0;
    uint32_t crossRankCount = 0;
    uint32_t remoteProxyRank = INVALID_VALUE_RANKID;
    std::array<uint32_t, MAX_RANK_SIZE> topologyRanks{};
};

uint64_t AlignDown(uint64_t value, uint64_t alignment)
{
    return value / alignment * alignment;
}

void FillChannelDesc(uint32_t remoteRank, const CommLink &link, HcclChannelDesc &desc)
{
    desc.remoteRank = remoteRank;
    desc.notifyNum = CHANNEL_NOTIFY_NUM;
    desc.channelProtocol = link.linkAttr.linkProtocol;
    desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
    desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
    desc.localEndpoint.loc = link.srcEndpointDesc.loc;
    desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
    desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
    desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
}

HcclResult GetLinkCandidates(HcclComm comm, uint32_t netLayer, uint32_t myRank,
    uint32_t remoteRank, std::vector<LinkCandidate> &candidates)
{
    CommLink *links = nullptr;
    uint32_t linkNum = 0;
    HcclResult ret = HcclRankGraphGetLinks(
        comm, netLayer, myRank, remoteRank, &links, &linkNum);
    if (ret != HCCL_SUCCESS || links == nullptr) {
        return HCCL_E_NOT_FOUND;
    }

    for (uint32_t linkIdx = 0; linkIdx < linkNum; ++linkIdx) {
        const CommLink link = links[linkIdx];
        if (!IsCcuProtocol(link.linkAttr.linkProtocol)) {
            continue;
        }

        EndpointAttrDieId dieId = std::numeric_limits<EndpointAttrDieId>::max();
        ret = HcclRankGraphGetEndpointInfo(comm, myRank, &link.srcEndpointDesc,
            ENDPOINT_ATTR_DIE_ID, sizeof(dieId), &dieId);
        if (ret == HCCL_SUCCESS) {
            candidates.push_back(LinkCandidate{link, dieId});
        }
    }

    return candidates.empty() ? HCCL_E_NOT_FOUND : HCCL_SUCCESS;
}

HcclResult SelectSameDieChannelDescs(HcclComm comm, const OpParam &param, uint32_t netLayer,
    const std::vector<uint32_t> &peerRanks, std::vector<HcclChannelDesc> &channelDescs,
    EndpointAttrDieId &selectedDie)
{
    if (peerRanks.empty() || channelDescs.size() != peerRanks.size()) {
        return HCCL_E_PARA;
    }

    std::vector<std::vector<LinkCandidate>> peerCandidates(peerRanks.size());
    for (uint32_t channelIdx = 0; channelIdx < peerRanks.size(); ++channelIdx) {
        HcclResult ret = GetLinkCandidates(comm, netLayer, param.myRank,
            peerRanks[channelIdx], peerCandidates[channelIdx]);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
    }

    std::vector<EndpointAttrDieId> triedDies;
    for (const LinkCandidate &seed : peerCandidates.front()) {
        if (std::find(triedDies.begin(), triedDies.end(), seed.dieId) != triedDies.end()) {
            continue;
        }
        triedDies.push_back(seed.dieId);

        bool allPeersReachable = true;
        for (uint32_t channelIdx = 0; channelIdx < peerRanks.size(); ++channelIdx) {
            const std::vector<LinkCandidate> &candidates = peerCandidates[channelIdx];
            const auto candidate = std::find_if(candidates.begin(), candidates.end(),
                [&seed](const LinkCandidate &item) { return item.dieId == seed.dieId; });
            if (candidate == candidates.end()) {
                allPeersReachable = false;
                break;
            }
            FillChannelDesc(peerRanks[channelIdx], candidate->link, channelDescs[channelIdx]);
        }

        if (allPeersReachable) {
            selectedDie = seed.dieId;
            return HCCL_SUCCESS;
        }
    }

    return HCCL_E_NOT_FOUND;
}

std::vector<uint32_t> GetAllPeerRanks(const OpParam &param)
{
    std::vector<uint32_t> peerRanks;
    peerRanks.reserve(param.rankSize - 1);
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (rank != param.myRank) {
            peerRanks.push_back(rank);
        }
    }
    return peerRanks;
}

HcclResult GetNetLayers(HcclComm comm, std::vector<uint32_t> &netLayers)
{
    uint32_t *layerData = nullptr;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layerData, &layerNum));
    if (layerData == nullptr || layerNum == 0) {
        HCCL_ERROR("No topology layer found");
        return HCCL_E_NOT_FOUND;
    }
    netLayers.assign(layerData, layerData + layerNum);
    std::sort(netLayers.begin(), netLayers.end());
    return HCCL_SUCCESS;
}

HcclResult PrepareChannelGroup(HcclComm comm, const OpParam &param, uint32_t netLayer,
    const std::vector<uint32_t> &peerRanks, ChannelGroup &group)
{
    if (peerRanks.empty()) {
        return HCCL_E_PARA;
    }

    group = ChannelGroup{};
    group.netLayer = netLayer;
    group.peerRanks = peerRanks;
    group.channelDescs.resize(peerRanks.size());
    CHK_RET(HcclChannelDescInit(
        group.channelDescs.data(), static_cast<uint32_t>(group.channelDescs.size())));
    CHK_RET(SelectSameDieChannelDescs(comm, param, netLayer, group.peerRanks,
        group.channelDescs, group.dieId));
    return HCCL_SUCCESS;
}

HcclResult PrepareSingleLayerGroup(HcclComm comm, const OpParam &param, ChannelGroup &group)
{
    std::vector<uint32_t> netLayers;
    CHK_RET(GetNetLayers(comm, netLayers));
    const std::vector<uint32_t> peerRanks = GetAllPeerRanks(param);

    for (auto layerIt = netLayers.rbegin(); layerIt != netLayers.rend(); ++layerIt) {
        ChannelGroup candidate;
        if (PrepareChannelGroup(comm, param, *layerIt, peerRanks, candidate) != HCCL_SUCCESS) {
            continue;
        }
        group = std::move(candidate);
        return HCCL_SUCCESS;
    }

    HCCL_ERROR("No single topology layer and local die can reach all peers of rank %u",
        param.myRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult PrepareDualLayerGroups(HcclComm comm, const OpParam &param,
    ChannelGroup &localGroup, ChannelGroup &crossGroup)
{
    std::vector<uint32_t> netLayers;
    CHK_RET(GetNetLayers(comm, netLayers));
    if (std::find(netLayers.begin(), netLayers.end(), LOCAL_MESH_LAYER) == netLayers.end() ||
        std::find(netLayers.begin(), netLayers.end(), CROSS_SERVER_LAYER) == netLayers.end()) {
        return HCCL_E_NOT_FOUND;
    }

    std::vector<uint32_t> localPeers;
    std::vector<uint32_t> crossPeers;
    for (uint32_t peerRank : GetAllPeerRanks(param)) {
        std::vector<LinkCandidate> localCandidates;
        if (GetLinkCandidates(
                comm, LOCAL_MESH_LAYER, param.myRank, peerRank, localCandidates) == HCCL_SUCCESS) {
            localPeers.push_back(peerRank);
        } else {
            crossPeers.push_back(peerRank);
        }
    }
    if (localPeers.empty() || crossPeers.empty()) {
        return HCCL_E_NOT_FOUND;
    }

    CHK_RET(PrepareChannelGroup(
        comm, param, LOCAL_MESH_LAYER, localPeers, localGroup));
    CHK_RET(PrepareChannelGroup(
        comm, param, CROSS_SERVER_LAYER, crossPeers, crossGroup));
    if (localGroup.dieId == crossGroup.dieId) {
        HCCL_ERROR("Layer %u and layer %u unexpectedly use the same local die %u",
            LOCAL_MESH_LAYER, CROSS_SERVER_LAYER, localGroup.dieId);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult BuildParallel2x8Topology(const OpParam &param,
    const ChannelGroup &localGroup, const ChannelGroup &crossGroup,
    Parallel2x8Topology &topology)
{
    if (param.rankSize != MAX_RANK_SIZE ||
        localGroup.peerRanks.size() + 1 != PARALLEL_2X8_LOCAL_RANK_NUM ||
        crossGroup.peerRanks.size() != PARALLEL_2X8_LOCAL_RANK_NUM) {
        return HCCL_E_PARA;
    }

    std::vector<uint32_t> localRanks = localGroup.peerRanks;
    localRanks.push_back(param.myRank);
    std::sort(localRanks.begin(), localRanks.end());

    std::vector<uint32_t> remoteRanks = crossGroup.peerRanks;
    std::sort(remoteRanks.begin(), remoteRanks.end());
    if (std::adjacent_find(localRanks.begin(), localRanks.end()) != localRanks.end() ||
        std::adjacent_find(remoteRanks.begin(), remoteRanks.end()) != remoteRanks.end()) {
        return HCCL_E_PARA;
    }

    std::array<uint32_t, MAX_RANK_SIZE> allRanks{};
    std::copy(localRanks.begin(), localRanks.end(), allRanks.begin());
    std::copy(remoteRanks.begin(), remoteRanks.end(),
        allRanks.begin() + PARALLEL_2X8_LOCAL_RANK_NUM);
    std::sort(allRanks.begin(), allRanks.end());
    for (uint32_t rankIndex = 0; rankIndex < param.rankSize; ++rankIndex) {
        if (allRanks[rankIndex] != rankIndex) {
            return HCCL_E_PARA;
        }
    }

    const auto localRankIt =
        std::lower_bound(localRanks.begin(), localRanks.end(), param.myRank);
    if (localRankIt == localRanks.end() || *localRankIt != param.myRank) {
        return HCCL_E_PARA;
    }
    const uint32_t localIndex =
        static_cast<uint32_t>(std::distance(localRanks.begin(), localRankIt));
    const uint32_t pairedRank = remoteRanks[localIndex];
    if (std::count(crossGroup.peerRanks.begin(), crossGroup.peerRanks.end(), pairedRank) != 1) {
        return HCCL_E_PARA;
    }

    topology = Parallel2x8Topology{};
    topology.serverIndex = localRanks.front() < remoteRanks.front() ? 0U : 1U;
    topology.localIndex = localIndex;
    topology.pairedRank = pairedRank;

    const std::vector<uint32_t> &firstServer =
        topology.serverIndex == 0 ? localRanks : remoteRanks;
    const std::vector<uint32_t> &secondServer =
        topology.serverIndex == 0 ? remoteRanks : localRanks;
    std::copy(firstServer.begin(), firstServer.end(), topology.topologyRanks.begin());
    std::copy(secondServer.begin(), secondServer.end(),
        topology.topologyRanks.begin() + PARALLEL_2X8_LOCAL_RANK_NUM);
    return HCCL_SUCCESS;
}

HcclResult BuildAsymmetric8x4Topology(const OpParam &param,
    const ChannelGroup &localGroup, const ChannelGroup &crossGroup,
    Asymmetric8x4Topology &topology)
{
    if (param.rankSize != ASYMMETRIC_8X4_RANK_NUM) {
        return HCCL_E_PARA;
    }

    std::vector<uint32_t> localRanks = localGroup.peerRanks;
    localRanks.push_back(param.myRank);
    std::sort(localRanks.begin(), localRanks.end());
    std::vector<uint32_t> remoteRanks = crossGroup.peerRanks;
    std::sort(remoteRanks.begin(), remoteRanks.end());

    const bool localIsLarge =
        localRanks.size() == ASYMMETRIC_LARGE_SERVER_RANK_NUM &&
        remoteRanks.size() == ASYMMETRIC_SMALL_SERVER_RANK_NUM;
    const bool localIsSmall =
        localRanks.size() == ASYMMETRIC_SMALL_SERVER_RANK_NUM &&
        remoteRanks.size() == ASYMMETRIC_LARGE_SERVER_RANK_NUM;
    if ((!localIsLarge && !localIsSmall) ||
        std::adjacent_find(localRanks.begin(), localRanks.end()) != localRanks.end() ||
        std::adjacent_find(remoteRanks.begin(), remoteRanks.end()) != remoteRanks.end()) {
        return HCCL_E_PARA;
    }

    std::array<uint32_t, ASYMMETRIC_8X4_RANK_NUM> allRanks{};
    std::copy(localRanks.begin(), localRanks.end(), allRanks.begin());
    std::copy(remoteRanks.begin(), remoteRanks.end(), allRanks.begin() + localRanks.size());
    std::sort(allRanks.begin(), allRanks.end());
    for (uint32_t rankIndex = 0; rankIndex < ASYMMETRIC_8X4_RANK_NUM; ++rankIndex) {
        if (allRanks[rankIndex] != rankIndex) {
            return HCCL_E_PARA;
        }
    }

    const auto localRankIt =
        std::lower_bound(localRanks.begin(), localRanks.end(), param.myRank);
    if (localRankIt == localRanks.end() || *localRankIt != param.myRank) {
        return HCCL_E_PARA;
    }

    const std::vector<uint32_t> &largeRanks = localIsLarge ? localRanks : remoteRanks;
    const std::vector<uint32_t> &smallRanks = localIsSmall ? localRanks : remoteRanks;
    topology = Asymmetric8x4Topology{};
    topology.serverIndex = localIsLarge ? 0U : 1U;
    topology.localIndex =
        static_cast<uint32_t>(std::distance(localRanks.begin(), localRankIt));
    topology.localRankCount = static_cast<uint32_t>(localRanks.size());
    topology.crossRankCount = static_cast<uint32_t>(remoteRanks.size());
    topology.remoteProxyRank = localIsLarge
        ? smallRanks[topology.localIndex / 2U]
        : largeRanks[topology.localIndex];
    std::copy(largeRanks.begin(), largeRanks.end(), topology.topologyRanks.begin());
    std::copy(smallRanks.begin(), smallRanks.end(),
        topology.topologyRanks.begin() + ASYMMETRIC_LARGE_SERVER_RANK_NUM);
    return HCCL_SUCCESS;
}

HcclResult PreparePairedCrossGroup(const ChannelGroup &crossGroup,
    uint32_t pairedRank, ChannelGroup &pairedCrossGroup)
{
    if (crossGroup.peerRanks.size() != crossGroup.channelDescs.size()) {
        return HCCL_E_PARA;
    }

    const auto peerIt =
        std::find(crossGroup.peerRanks.begin(), crossGroup.peerRanks.end(), pairedRank);
    if (peerIt == crossGroup.peerRanks.end() ||
        std::count(crossGroup.peerRanks.begin(), crossGroup.peerRanks.end(), pairedRank) != 1) {
        return HCCL_E_PARA;
    }
    const size_t peerIndex =
        static_cast<size_t>(std::distance(crossGroup.peerRanks.begin(), peerIt));

    pairedCrossGroup = ChannelGroup{};
    pairedCrossGroup.netLayer = crossGroup.netLayer;
    pairedCrossGroup.dieId = crossGroup.dieId;
    pairedCrossGroup.peerRanks.push_back(pairedRank);
    pairedCrossGroup.channelDescs.push_back(crossGroup.channelDescs[peerIndex]);
    return HCCL_SUCCESS;
}

bool CanUseParallel2x8(uint64_t recvBytes, uint64_t localBufferSize)
{
    const uint64_t partA = AlignDown(recvBytes / 2, PARALLEL_2X8_ALIGNMENT);
    const uint64_t partB = recvBytes - partA;
    const uint64_t partALane =
        AlignDown(partA / (PARALLEL_2X8_LOCAL_RANK_NUM - 1), PARALLEL_2X8_ALIGNMENT);
    const uint64_t partBLane =
        AlignDown(partB / (PARALLEL_2X8_LOCAL_RANK_NUM - 1), PARALLEL_2X8_ALIGNMENT);
    if (partA == 0 || partB == 0 || partALane == 0 || partBLane == 0 ||
        partA > std::numeric_limits<uint64_t>::max() / PARALLEL_2X8_SERVER_NUM) {
        return false;
    }

    const uint64_t partAWorkspace = PARALLEL_2X8_SERVER_NUM * partA;
    if (partB > (std::numeric_limits<uint64_t>::max() - partAWorkspace) /
            PARALLEL_2X8_LOCAL_RANK_NUM) {
        return false;
    }
    const uint64_t requiredWorkspace =
        partAWorkspace + PARALLEL_2X8_LOCAL_RANK_NUM * partB;
    return requiredWorkspace <= localBufferSize;
}

bool CanUseAsymmetric8x4(uint64_t recvBytes, uint64_t localBufferSize,
    uint32_t localPeerCount, uint32_t crossPeerCount)
{
    const bool validPeerCounts =
        (localPeerCount == ASYMMETRIC_LARGE_SERVER_RANK_NUM - 1U &&
            crossPeerCount == ASYMMETRIC_SMALL_SERVER_RANK_NUM) ||
        (localPeerCount == ASYMMETRIC_SMALL_SERVER_RANK_NUM - 1U &&
            crossPeerCount == ASYMMETRIC_LARGE_SERVER_RANK_NUM);
    if (!validPeerCounts) {
        return false;
    }

    // The four-rank side creates three Mesh-A partials.  Its eight logical
    // cross peers share a rank-side Clos bandwidth of four Mesh links, so the
    // aggregate NHR-B load is 2*B.  Balancing 3*A and 2*B gives A:B = 2:3.
    const uint64_t partA =
        AlignDown(recvBytes * 2U / 5U, PARALLEL_2X8_ALIGNMENT);
    const uint64_t partB = recvBytes - partA;
    const uint64_t aLane =
        AlignDown(partA / localPeerCount, PARALLEL_2X8_ALIGNMENT);
    const uint64_t bCrossLane =
        AlignDown(partB / crossPeerCount, PARALLEL_2X8_ALIGNMENT);
    const uint64_t bMeshLane =
        AlignDown(partB / localPeerCount, PARALLEL_2X8_ALIGNMENT);
    if (partA == 0 || partB == 0 || aLane == 0 ||
        bCrossLane == 0 || bMeshLane == 0 ||
        partA > std::numeric_limits<uint64_t>::max() / ASYMMETRIC_8X4_RANK_NUM) {
        return false;
    }
    return partA * ASYMMETRIC_8X4_RANK_NUM <= localBufferSize;
}

HcclResult AcquireChannelGroup(HcclComm comm, const ChannelGroup &group,
    std::vector<ChannelHandle> &channels)
{
    if (group.channelDescs.empty()) {
        return HCCL_E_PARA;
    }
    channels.resize(group.channelDescs.size());
    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU,
        group.channelDescs.data(), static_cast<uint32_t>(group.channelDescs.size()),
        channels.data()));
    return HCCL_SUCCESS;
}

HcclResult InitKernelInfo(const OpParam &param, const char *kernelName, void *kernelFunc,
    uint32_t hasWorker,
    const std::vector<uint32_t> &peerRanks, const std::vector<ChannelHandle> &channels,
    CcuKernelInfo &kernelInfo)
{
    if (kernelName == nullptr || kernelFunc == nullptr || peerRanks.empty() ||
        peerRanks.size() != channels.size() || peerRanks.size() > MAX_RANK_SIZE) {
        return HCCL_E_PARA;
    }

    const int nameLength = std::snprintf(
        kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "%s", kernelName);
    if (nameLength < 0 || static_cast<size_t>(nameLength) >= sizeof(kernelInfo.kernelFuncName)) {
        HCCL_ERROR("Failed to construct the CCU kernel name");
        return HCCL_E_INTERNAL;
    }
    kernelInfo.kernelFunc = kernelFunc;

    auto kernelArg = std::make_shared<CcuReduceScatterKernelArg>();
    kernelArg->rankSize = param.rankSize;
    kernelArg->rankId = param.myRank;
    kernelArg->hasWorker = hasWorker;
    kernelArg->dataType = param.dataType;
    kernelArg->reduceOp = param.reduceType;
    kernelArg->channelCount = static_cast<uint32_t>(channels.size());
    for (uint32_t channelIdx = 0; channelIdx < kernelArg->channelCount; ++channelIdx) {
        kernelArg->channels[channelIdx] = channels[channelIdx];
        kernelArg->peerRanks[channelIdx] = peerRanks[channelIdx];
    }
    kernelInfo.setKernelArg(kernelArg);
    return HCCL_SUCCESS;
}

HcclResult InitParallel2x8KernelInfo(const OpParam &param, const char *kernelName,
    void *kernelFunc, const Parallel2x8Topology &topology,
    const std::vector<uint32_t> &peerRanks, const std::vector<ChannelHandle> &channels,
    CcuKernelInfo &kernelInfo)
{
    if (kernelName == nullptr || kernelFunc == nullptr || peerRanks.empty() ||
        peerRanks.size() != channels.size() || peerRanks.size() > MAX_RANK_SIZE ||
        topology.serverIndex >= PARALLEL_2X8_SERVER_NUM ||
        topology.localIndex >= PARALLEL_2X8_LOCAL_RANK_NUM ||
        topology.pairedRank >= param.rankSize) {
        return HCCL_E_PARA;
    }

    const int nameLength = std::snprintf(
        kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "%s", kernelName);
    if (nameLength < 0 || static_cast<size_t>(nameLength) >= sizeof(kernelInfo.kernelFuncName)) {
        HCCL_ERROR("Failed to construct the 2D CCU kernel name");
        return HCCL_E_INTERNAL;
    }
    kernelInfo.kernelFunc = kernelFunc;

    auto kernelArg = std::make_shared<CcuReduceScatter2DKernelArg>();
    kernelArg->rankSize = param.rankSize;
    kernelArg->rankId = param.myRank;
    kernelArg->serverIndex = topology.serverIndex;
    kernelArg->localIndex = topology.localIndex;
    kernelArg->dataType = param.dataType;
    kernelArg->reduceOp = param.reduceType;
    std::copy(topology.topologyRanks.begin(), topology.topologyRanks.end(),
        kernelArg->topologyRanks);
    kernelArg->channelCount = static_cast<uint32_t>(channels.size());
    for (uint32_t channelIdx = 0; channelIdx < kernelArg->channelCount; ++channelIdx) {
        kernelArg->channels[channelIdx] = channels[channelIdx];
        kernelArg->peerRanks[channelIdx] = peerRanks[channelIdx];
    }
    kernelInfo.setKernelArg(kernelArg);
    return HCCL_SUCCESS;
}

HcclResult InitAsymmetric8x4KernelInfo(const OpParam &param, const char *kernelName,
    void *kernelFunc, const Asymmetric8x4Topology &topology,
    const std::vector<uint32_t> &peerRanks, const std::vector<ChannelHandle> &channels,
    CcuKernelInfo &kernelInfo)
{
    if (kernelName == nullptr || kernelFunc == nullptr || peerRanks.empty() ||
        peerRanks.size() != channels.size() || peerRanks.size() > MAX_RANK_SIZE ||
        topology.serverIndex >= PARALLEL_2X8_SERVER_NUM ||
        topology.localIndex >= topology.localRankCount ||
        topology.localRankCount + topology.crossRankCount != ASYMMETRIC_8X4_RANK_NUM ||
        topology.remoteProxyRank >= param.rankSize) {
        return HCCL_E_PARA;
    }

    const int nameLength = std::snprintf(
        kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "%s", kernelName);
    if (nameLength < 0 || static_cast<size_t>(nameLength) >= sizeof(kernelInfo.kernelFuncName)) {
        HCCL_ERROR("Failed to construct the asymmetric CCU kernel name");
        return HCCL_E_INTERNAL;
    }
    kernelInfo.kernelFunc = kernelFunc;

    auto kernelArg = std::make_shared<CcuReduceScatterAsymKernelArg>();
    kernelArg->rankSize = param.rankSize;
    kernelArg->rankId = param.myRank;
    kernelArg->serverIndex = topology.serverIndex;
    kernelArg->localIndex = topology.localIndex;
    kernelArg->localRankCount = topology.localRankCount;
    kernelArg->crossRankCount = topology.crossRankCount;
    kernelArg->remoteProxyRank = topology.remoteProxyRank;
    kernelArg->dataType = param.dataType;
    kernelArg->reduceOp = param.reduceType;
    std::copy(topology.topologyRanks.begin(), topology.topologyRanks.end(),
        kernelArg->topologyRanks);
    kernelArg->channelCount = static_cast<uint32_t>(channels.size());
    for (uint32_t channelIdx = 0; channelIdx < kernelArg->channelCount; ++channelIdx) {
        kernelArg->channels[channelIdx] = channels[channelIdx];
        kernelArg->peerRanks[channelIdx] = peerRanks[channelIdx];
    }
    kernelInfo.setKernelArg(kernelArg);
    return HCCL_SUCCESS;
}

HcclResult RegisterKernels(HcclComm comm, std::vector<CcuKernelInfo> &kernelInfos,
    AlgResourceCtx &resCtx)
{
    if (kernelInfos.empty()) {
        return HCCL_E_PARA;
    }

    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    if (insNum != 1) {
        HCCL_ERROR("Expected one CCU instance, actual number is %u", insNum);
        return HCCL_E_INTERNAL;
    }

    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));
    resCtx.ccuKernels.resize(kernelInfos.size());
    for (uint32_t kernelIdx = 0; kernelIdx < kernelInfos.size(); ++kernelIdx) {
        CcuKernelInfo &kernelInfo = kernelInfos[kernelIdx];
        const void *kernelArgs[] = {kernelInfo.kernelArg};
        CcuResult registerRet = HcommCcuKernelRegister(insHandle, KERNEL_DIE_ID,
            kernelInfo.kernelFuncName, kernelInfo.kernelFunc, kernelArgs, 1,
            &resCtx.ccuKernels[kernelIdx]);
        if (registerRet != CCU_SUCCESS) {
            HCCL_ERROR("CCU kernel %s registration failed, ret[%d]",
                kernelInfo.kernelFuncName, registerRet);
            (void)HcommCcuKernelRegisterEnd(insHandle);
            return ConvertCcuToHccl(registerRet);
        }
    }
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
    return HCCL_SUCCESS;
}

HcclResult BuildSingleLayerResources(HcclComm comm, const OpParam &param,
    const ChannelGroup &group, AlgResourceCtx &resCtx)
{
    std::vector<ChannelHandle> channels;
    CHK_RET(AcquireChannelGroup(comm, group, channels));

    std::vector<CcuKernelInfo> kernelInfos(SINGLE_KERNEL_NUM);
    CHK_RET(InitKernelInfo(param, "CcuReduceScatterSingle",
        reinterpret_cast<void *>(ops_hccl::CcuKernel), 0, group.peerRanks, channels,
        kernelInfos[0]));
    resCtx.algorithm = ReduceScatterAlgorithm::SINGLE_LAYER;
    CHK_RET(RegisterKernels(comm, kernelInfos, resCtx));
    HCCL_INFO("Use single-layer ReduceScatter on layer %u, local die %u",
        group.netLayer, group.dieId);
    return HCCL_SUCCESS;
}

HcclResult BuildLatin4x1Resources(HcclComm comm, const OpParam &param,
    const ChannelGroup &group, AlgResourceCtx &resCtx)
{
    if (param.rankSize != LATIN_4X1_RANK_NUM ||
        group.peerRanks.size() + 1 != LATIN_4X1_RANK_NUM) {
        return HCCL_E_PARA;
    }

    std::vector<ChannelHandle> channels;
    CHK_RET(AcquireChannelGroup(comm, group, channels));

    std::vector<CcuKernelInfo> kernelInfos(SINGLE_KERNEL_NUM);
    CHK_RET(InitKernelInfo(param, "CcuReduceScatter4x1Latin",
        reinterpret_cast<void *>(ops_hccl::CcuReduceScatter4x1LatinKernel),
        0, group.peerRanks, channels, kernelInfos[0]));
    resCtx.algorithm = ReduceScatterAlgorithm::LATIN_4X1;
    CHK_RET(RegisterKernels(comm, kernelInfos, resCtx));
    HCCL_INFO("Use 4x1 Latin ReadReduce on layer %u, local die %u",
        group.netLayer, group.dieId);
    return HCCL_SUCCESS;
}

HcclResult BuildDualLayerResources(HcclComm comm, const OpParam &param,
    const ChannelGroup &localGroup, const ChannelGroup &crossGroup, AlgResourceCtx &resCtx)
{
    constexpr uint32_t workerNotifyNum = 1;
    CHK_RET(HcclThreadAcquire(comm, CommEngine::COMM_ENGINE_CCU, 1,
        workerNotifyNum, &resCtx.workerThread));

    std::vector<ChannelHandle> localChannels;
    std::vector<ChannelHandle> crossChannels;
    CHK_RET(AcquireChannelGroup(comm, localGroup, localChannels));
    CHK_RET(AcquireChannelGroup(comm, crossGroup, crossChannels));

    std::vector<CcuKernelInfo> kernelInfos(DUAL_KERNEL_NUM);
    CHK_RET(InitKernelInfo(param, "CcuReduceScatterMain",
        reinterpret_cast<void *>(ops_hccl::CcuReduceScatterMainKernel), 1, localGroup.peerRanks,
        localChannels, kernelInfos[0]));
    CHK_RET(InitKernelInfo(param, "CcuReduceScatterWorker",
        reinterpret_cast<void *>(ops_hccl::CcuReduceScatterWorkerKernel), 1, crossGroup.peerRanks,
        crossChannels, kernelInfos[1]));
    CHK_RET(InitKernelInfo(param, "CcuReduceScatterReduce",
        reinterpret_cast<void *>(ops_hccl::CcuReduceScatterReduceKernel), 1, localGroup.peerRanks,
        localChannels, kernelInfos[2]));
    resCtx.algorithm = ReduceScatterAlgorithm::DUAL_LAYER;
    CHK_RET(RegisterKernels(comm, kernelInfos, resCtx));
    HCCL_INFO("Use dual-layer ReduceScatter: layer %u die %u has %zu peers, "
        "layer %u die %u has %zu peers",
        localGroup.netLayer, localGroup.dieId, localGroup.peerRanks.size(),
        crossGroup.netLayer, crossGroup.dieId, crossGroup.peerRanks.size());
    return HCCL_SUCCESS;
}

HcclResult BuildParallel2x8Resources(HcclComm comm, const OpParam &param,
    const ChannelGroup &localGroup, const ChannelGroup &pairedCrossGroup,
    const Parallel2x8Topology &topology, AlgResourceCtx &resCtx)
{
    constexpr uint32_t workerNotifyNum = 1;
    CHK_RET(HcclThreadAcquire(comm, CommEngine::COMM_ENGINE_CCU, 1,
        workerNotifyNum, &resCtx.workerThread));

    std::vector<ChannelHandle> localChannels;
    std::vector<ChannelHandle> pairedCrossChannels;
    CHK_RET(AcquireChannelGroup(comm, localGroup, localChannels));
    CHK_RET(AcquireChannelGroup(comm, pairedCrossGroup, pairedCrossChannels));

    std::vector<CcuKernelInfo> kernelInfos(PARALLEL_2X8_KERNEL_NUM);
    CHK_RET(InitParallel2x8KernelInfo(param,
        "CcuReduceScatter2DMeshAStage1Kernel",
        reinterpret_cast<void *>(ops_hccl::CcuReduceScatter2DMeshAStage1Kernel),
        topology, localGroup.peerRanks, localChannels, kernelInfos[0]));
    CHK_RET(InitParallel2x8KernelInfo(param,
        "CcuReduceScatter2DNhrBStage1Kernel",
        reinterpret_cast<void *>(ops_hccl::CcuReduceScatter2DNhrBStage1Kernel),
        topology, pairedCrossGroup.peerRanks, pairedCrossChannels, kernelInfos[1]));
    CHK_RET(InitParallel2x8KernelInfo(param,
        "CcuReduceScatter2DMeshBStage2Kernel",
        reinterpret_cast<void *>(ops_hccl::CcuReduceScatter2DMeshBStage2Kernel),
        topology, localGroup.peerRanks, localChannels, kernelInfos[2]));
    CHK_RET(InitParallel2x8KernelInfo(param,
        "CcuReduceScatter2DNhrAStage2Kernel",
        reinterpret_cast<void *>(ops_hccl::CcuReduceScatter2DNhrAStage2Kernel),
        topology, pairedCrossGroup.peerRanks, pairedCrossChannels, kernelInfos[3]));

    resCtx.algorithm = ReduceScatterAlgorithm::PARALLEL_2X8;
    CHK_RET(RegisterKernels(comm, kernelInfos, resCtx));
    HCCL_INFO("Use parallel 2x8 ReduceScatter: local die %u, cross die %u, "
        "server index %u, local index %u, paired rank %u",
        localGroup.dieId, pairedCrossGroup.dieId, topology.serverIndex,
        topology.localIndex, topology.pairedRank);
    return HCCL_SUCCESS;
}

HcclResult BuildAsymmetric8x4Resources(HcclComm comm, const OpParam &param,
    const ChannelGroup &localGroup, const ChannelGroup &crossGroup,
    const Asymmetric8x4Topology &topology, AlgResourceCtx &resCtx)
{
    constexpr uint32_t workerNotifyNum = 1;
    CHK_RET(HcclThreadAcquire(comm, CommEngine::COMM_ENGINE_CCU, 1,
        workerNotifyNum, &resCtx.workerThread));

    std::vector<ChannelHandle> localChannels;
    std::vector<ChannelHandle> crossChannels;
    CHK_RET(AcquireChannelGroup(comm, localGroup, localChannels));
    CHK_RET(AcquireChannelGroup(comm, crossGroup, crossChannels));

    std::vector<CcuKernelInfo> kernelInfos(ASYMMETRIC_8X4_KERNEL_NUM);
    CHK_RET(InitAsymmetric8x4KernelInfo(param,
        "CcuReduceScatterAsymMeshAStage1Kernel",
        reinterpret_cast<void *>(ops_hccl::CcuReduceScatterAsymMeshAStage1Kernel),
        topology, localGroup.peerRanks, localChannels, kernelInfos[0]));
    CHK_RET(InitAsymmetric8x4KernelInfo(param,
        "CcuReduceScatterAsymNhrBStage1Kernel",
        reinterpret_cast<void *>(ops_hccl::CcuReduceScatterAsymNhrBStage1Kernel),
        topology, crossGroup.peerRanks, crossChannels, kernelInfos[1]));
    CHK_RET(InitAsymmetric8x4KernelInfo(param,
        "CcuReduceScatterAsymMeshBStage2Kernel",
        reinterpret_cast<void *>(ops_hccl::CcuReduceScatterAsymMeshBStage2Kernel),
        topology, localGroup.peerRanks, localChannels, kernelInfos[2]));
    CHK_RET(InitAsymmetric8x4KernelInfo(param,
        "CcuReduceScatterAsymNhrAStage2Kernel",
        reinterpret_cast<void *>(ops_hccl::CcuReduceScatterAsymNhrAStage2Kernel),
        topology, crossGroup.peerRanks, crossChannels, kernelInfos[3]));

    resCtx.algorithm = ReduceScatterAlgorithm::ASYMMETRIC_8X4;
    resCtx.localPeerCount = static_cast<uint32_t>(localGroup.peerRanks.size());
    resCtx.crossPeerCount = static_cast<uint32_t>(crossGroup.peerRanks.size());
    CHK_RET(RegisterKernels(comm, kernelInfos, resCtx));
    HCCL_INFO("Use asymmetric 8x4 ReduceScatter: local die %u, cross die %u, "
        "server index %u, local index %u, local ranks %u, cross ranks %u, proxy rank %u",
        localGroup.dieId, crossGroup.dieId, topology.serverIndex, topology.localIndex,
        topology.localRankCount, topology.crossRankCount, topology.remoteProxyRank);
    return HCCL_SUCCESS;
}

bool IsTwoByEightTopology(const OpParam &param,
    const ChannelGroup &localGroup, const ChannelGroup &crossGroup)
{
    return param.rankSize == MAX_RANK_SIZE &&
        localGroup.peerRanks.size() + 1 == PARALLEL_2X8_LOCAL_RANK_NUM &&
        crossGroup.peerRanks.size() == PARALLEL_2X8_LOCAL_RANK_NUM;
}

bool IsOptimizedDualLayerTopology(const OpParam &param,
    const ChannelGroup &localGroup, const ChannelGroup &crossGroup)
{
    const size_t localPeerNum = localGroup.peerRanks.size();
    const size_t crossPeerNum = crossGroup.peerRanks.size();
    const bool isTwoByEight =
        param.rankSize == 16 && localPeerNum == 7 && crossPeerNum == 8;
    const bool isEightPlusFour = param.rankSize == 12 &&
        ((localPeerNum == 7 && crossPeerNum == 4) ||
            (localPeerNum == 3 && crossPeerNum == 8));
    return isTwoByEight || isEightPlusFour;
}

HcclResult BuildResourceContext(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtx)
{
    if (param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    CHK_RET(HcclGetHcclBuffer(comm, &resCtx.localBuffer.addr, &resCtx.localBuffer.size));
    if (resCtx.localBuffer.addr == nullptr ||
        resCtx.localBuffer.size < param.rankSize * sizeof(float)) {
        HCCL_ERROR("HCCL buffer is unavailable or too small, size[%llu]",
            static_cast<unsigned long long>(resCtx.localBuffer.size));
        return HCCL_E_INTERNAL;
    }

    const uint64_t recvBytes = param.count * sizeof(float);
    ChannelGroup localGroup;
    ChannelGroup crossGroup;
    if (recvBytes >= DUAL_LAYER_MIN_BYTES &&
        (param.rankSize == 16 || param.rankSize == 12)) {
        if (PrepareDualLayerGroups(
                comm, param, localGroup, crossGroup) == HCCL_SUCCESS &&
            IsOptimizedDualLayerTopology(param, localGroup, crossGroup)) {
            if (IsTwoByEightTopology(param, localGroup, crossGroup) &&
                CanUseParallel2x8(recvBytes, resCtx.localBuffer.size)) {
                Parallel2x8Topology topology;
                ChannelGroup pairedCrossGroup;
                if (BuildParallel2x8Topology(
                        param, localGroup, crossGroup, topology) == HCCL_SUCCESS &&
                    PreparePairedCrossGroup(
                        crossGroup, topology.pairedRank, pairedCrossGroup) == HCCL_SUCCESS) {
                    return BuildParallel2x8Resources(
                        comm, param, localGroup, pairedCrossGroup, topology, resCtx);
                }
            }
            if (param.rankSize == ASYMMETRIC_8X4_RANK_NUM &&
                CanUseAsymmetric8x4(recvBytes, resCtx.localBuffer.size,
                    static_cast<uint32_t>(localGroup.peerRanks.size()),
                    static_cast<uint32_t>(crossGroup.peerRanks.size()))) {
                Asymmetric8x4Topology topology;
                if (BuildAsymmetric8x4Topology(
                        param, localGroup, crossGroup, topology) == HCCL_SUCCESS) {
                    return BuildAsymmetric8x4Resources(
                        comm, param, localGroup, crossGroup, topology, resCtx);
                }
            }
            return BuildDualLayerResources(
                comm, param, localGroup, crossGroup, resCtx);
        }
    }

    ChannelGroup singleGroup;
    HcclResult singleRet = PrepareSingleLayerGroup(comm, param, singleGroup);
    if (singleRet == HCCL_SUCCESS) {
        if (param.rankSize == LATIN_4X1_RANK_NUM &&
            recvBytes >= LATIN_4X1_MIN_BYTES &&
            singleGroup.peerRanks.size() + 1 == LATIN_4X1_RANK_NUM) {
            return BuildLatin4x1Resources(comm, param, singleGroup, resCtx);
        }
        return BuildSingleLayerResources(comm, param, singleGroup, resCtx);
    }

    return singleRet;
}
} // namespace

HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    if (dataType != HCCL_DATA_TYPE_FP32 || op != HCCL_REDUCE_SUM) {
        HCCL_ERROR("Only FP32 SUM ReduceScatter is supported, dataType[%d], op[%d]",
            static_cast<int>(dataType), static_cast<int>(op));
        return HCCL_E_NOT_SUPPORT;
    }
    if (recvCount > std::numeric_limits<uint64_t>::max() / sizeof(float)) {
        HCCL_ERROR("recvCount[%llu] overflows the FP32 byte size",
            static_cast<unsigned long long>(recvCount));
        return HCCL_E_PARA;
    }

    OpParam param{};
    const int tagLength =
        std::snprintf(param.tag, sizeof(param.tag), "%s", "hccl_custom_reduce_scatter_ccu");
    if (tagLength < 0 || static_cast<size_t>(tagLength) >= sizeof(param.tag)) {
        HCCL_ERROR("Failed to construct the ReduceScatter resource tag");
        return HCCL_E_INTERNAL;
    }
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH] = {};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    if (param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || param.myRank >= param.rankSize) {
        HCCL_ERROR("Invalid rank information, rank[%u], rankSize[%u]", param.myRank, param.rankSize);
        return HCCL_E_PARA;
    }
    const uint64_t recvBytes = recvCount * sizeof(float);
    if (recvBytes != 0 && param.rankSize > std::numeric_limits<uint64_t>::max() / recvBytes) {
        HCCL_ERROR("The ReduceScatter input byte size overflows uint64");
        return HCCL_E_PARA;
    }
    if (recvCount == 0) {
        return HCCL_SUCCESS;
    }

    constexpr uint32_t notifyNum = 1;
    CHK_RET(HcclThreadAcquireWithStream(
        comm, CommEngine::COMM_ENGINE_CCU, stream, notifyNum, &param.cpuThread));

    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    if (HcclEngineCtxGet(
            comm, param.tag, CommEngine::COMM_ENGINE_CCU, &ctx, &ctxSize) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = ctxSize;
    } else {
        AlgResourceCtx resCtxHost{};
        CHK_RET(BuildResourceContext(comm, param, resCtxHost));

        std::vector<char> serializedCtx = resCtxHost.Serialize();
        param.ctxSize = serializedCtx.size();
        CHK_RET(HcclEngineCtxCreate(
            comm, param.tag, CommEngine::COMM_ENGINE_CCU, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, CommEngine::COMM_ENGINE_CCU, param.tag,
            serializedCtx.data(), serializedCtx.size(), 0));
    }

    CHK_RET(ops_hccl::ExecOp(param));
    return HCCL_SUCCESS;
}
