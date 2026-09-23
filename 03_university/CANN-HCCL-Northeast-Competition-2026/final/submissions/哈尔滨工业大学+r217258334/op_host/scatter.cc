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
#include <memory>
#include <utility>
#include <vector>

#include <ccu/ccu_launch.h>
#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>

#include "ccu_kernel.h"
#include "common.h"
#include "custom.h"
#include "exec_op.h"
#include "hccl.h"
#include "log.h"

namespace {

constexpr uint32_t CHANNEL_NOTIFY_NUM_DIRECT = 3;
constexpr uint32_t CHANNEL_NOTIFY_NUM_RELAY = 6;
constexpr uint32_t MAX_DIRECT_GROUP_NUM = 2;
constexpr uint32_t MAX_RELAY_GROUP_NUM = 4;
constexpr uint32_t MAIN_NOTIFY_NUM_NORMAL = 1;
constexpr uint32_t MAIN_NOTIFY_NUM_LARGE = 2;
constexpr uint32_t MAIN_NOTIFY_NUM_RELAY_CANDIDATE = 4;
constexpr uint32_t SLAVE_NOTIFY_NUM = 1;
constexpr uint64_t LARGE_MESSAGE_THRESHOLD = 1ULL * 1024 * 1024;

struct PeerChannelInfo {
    uint32_t peerRank = 0;
    ChannelHandle channel = 0;
    EndpointAttrDieId dieId = 0;
    uint32_t localServerIdx = INVALID_VALUE_RANKID;
    uint32_t remoteServerIdx = INVALID_VALUE_RANKID;
};

struct DieChannelGroup {
    EndpointAttrDieId dieId = 0;
    bool isLocalGroup = false;
    std::vector<PeerChannelInfo> peers;
};

struct RelayTopologyInfo {
    uint32_t numerator = 0;
    uint32_t denominator = 1;
    std::vector<uint32_t> serverIds;
};

bool TryBuildServerIdsFromLayer(HcclComm comm, uint32_t netLayer, uint32_t rankSize,
    std::vector<uint32_t> &serverIds, std::vector<uint32_t> &componentSizes)
{
    serverIds.assign(rankSize, INVALID_VALUE_RANKID);
    componentSizes.clear();

    uint32_t componentId = 0;
    for (uint32_t representative = 0; representative < rankSize; ++representative) {
        if (serverIds[representative] != INVALID_VALUE_RANKID) {
            continue;
        }

        serverIds[representative] = componentId;
        uint32_t componentSize = 1;

        for (uint32_t rank = representative + 1; rank < rankSize; ++rank) {
            if (serverIds[rank] != INVALID_VALUE_RANKID) {
                continue;
            }

            CommLink *linkList = nullptr;
            uint32_t linkNum = 0;
            const HcclResult ret =
                HcclRankGraphGetLinks(comm, netLayer, representative, rank, &linkList, &linkNum);
            if (ret != HCCL_SUCCESS) {
                // Ranks in different topology instances may have no connection
                // on this layer.  Do not reject the entire layer just because a
                // cross-instance pair is unavailable.
                continue;
            }

            // On the intra-Server Full-Mesh layer, ranks in the same physical
            // Server have a link and ranks in different Servers do not.  We do
            // not assume a layer number, instance-size API result, or serverIdx.
            if (linkNum != 0) {
                serverIds[rank] = componentId;
                ++componentSize;
            }
        }

        componentSizes.push_back(componentSize);
        ++componentId;
    }
    return true;
}

bool Detect2x8FromLocalInstance(HcclComm comm, const OpParam &param, RelayTopologyInfo &topology)
{
    topology = RelayTopologyInfo{};
    if (param.rankSize != 16) {
        return false;
    }

    uint32_t *netLayersRaw = nullptr;
    uint32_t netLayerNum = 0;
    const HcclResult layerRet = HcclRankGraphGetLayers(comm, &netLayersRaw, &netLayerNum);
    if (layerRet != HCCL_SUCCESS || netLayersRaw == nullptr || netLayerNum == 0) {
        return false;
    }
    const std::vector<uint32_t> netLayers(netLayersRaw, netLayersRaw + netLayerNum);
    const std::vector<uint32_t> topo2x8 = {8, 8};

    for (uint32_t netLayer : netLayers) {
        // First identify the exact 2x8 Server layer by its global instance sizes.
        uint32_t *instSizesRaw = nullptr;
        uint32_t instNum = 0;
        const HcclResult instRet =
            HcclRankGraphGetInstSizeListByLayer(comm, netLayer, &instSizesRaw, &instNum);
        if (instRet != HCCL_SUCCESS || instSizesRaw == nullptr || instNum == 0) {
            continue;
        }

        std::vector<uint32_t> instSizes(instSizesRaw, instSizesRaw + instNum);
        std::sort(instSizes.begin(), instSizes.end());
        if (instSizes != topo2x8) {
            continue;
        }

        // This API directly returns all ranks in *this rank's* topology
        // instance on the selected layer, avoiding arbitrary cross-instance
        // GetLinks queries that caused the earlier relay gate to fall back.
        uint32_t *localRanksRaw = nullptr;
        uint32_t localRankNum = 0;
        const HcclResult ranksRet =
            HcclRankGraphGetRanksByLayer(comm, netLayer, &localRanksRaw, &localRankNum);
        if (ranksRet != HCCL_SUCCESS || localRanksRaw == nullptr || localRankNum != 8) {
            continue;
        }

        std::vector<uint8_t> seen(param.rankSize, 0);
        std::vector<uint32_t> localRanks;
        localRanks.reserve(localRankNum);
        bool containsMyRank = false;
        bool valid = true;
        for (uint32_t i = 0; i < localRankNum; ++i) {
            const uint32_t rank = localRanksRaw[i];
            if (rank >= param.rankSize || seen[rank] != 0) {
                valid = false;
                break;
            }
            seen[rank] = 1;
            localRanks.push_back(rank);
            containsMyRank = containsMyRank || (rank == param.myRank);
        }
        if (!valid || !containsMyRank) {
            continue;
        }

        std::vector<uint32_t> remoteRanks;
        remoteRanks.reserve(param.rankSize - localRankNum);
        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            if (seen[rank] == 0) {
                remoteRanks.push_back(rank);
            }
        }
        if (remoteRanks.size() != 8) {
            continue;
        }

        std::sort(localRanks.begin(), localRanks.end());
        std::sort(remoteRanks.begin(), remoteRanks.end());

        // Canonicalize the two Server ids so every rank serializes the exact
        // same serverIds vector.  With two disjoint 8-rank instances, comparing
        // the sorted rank lists gives a rank-independent ordering.
        const bool localIsServerZero =
            std::lexicographical_compare(localRanks.begin(), localRanks.end(),
                remoteRanks.begin(), remoteRanks.end());
        const uint32_t localServerId = localIsServerZero ? 0U : 1U;
        const uint32_t remoteServerId = localIsServerZero ? 1U : 0U;

        std::vector<uint32_t> serverIds(param.rankSize, INVALID_VALUE_RANKID);
        for (uint32_t rank : localRanks) {
            serverIds[rank] = localServerId;
        }
        for (uint32_t rank : remoteRanks) {
            serverIds[rank] = remoteServerId;
        }

        topology.numerator = 4;
        topology.denominator = 11;
        topology.serverIds = std::move(serverIds);
        HCCL_INFO("[%s] activate canonical 2x8 relay, layer[%u], localServerId[%u], ratio[4/11]",
            __func__, netLayer, localServerId);
        return true;
    }

    return false;
}


bool Validate4x3ServerPartition(HcclComm comm, uint32_t netLayer,
    const std::vector<uint32_t> &serverIds)
{
    if (serverIds.size() != 12) {
        return false;
    }

    // Validate every pair that is claimed to be inside the same Server.
    // We intentionally do not require cross-Server pairs to have zero links,
    // because some simulated topology layers may still expose a routed link.
    for (uint32_t lhs = 0; lhs < serverIds.size(); ++lhs) {
        for (uint32_t rhs = lhs + 1; rhs < serverIds.size(); ++rhs) {
            if (serverIds[lhs] != serverIds[rhs]) {
                continue;
            }

            CommLink *linkList = nullptr;
            uint32_t linkNum = 0;
            const HcclResult ret =
                HcclRankGraphGetLinks(comm, netLayer, lhs, rhs, &linkList, &linkNum);
            if (ret != HCCL_SUCCESS || linkNum == 0) {
                return false;
            }
        }
    }
    return true;
}

bool Detect4x3FromLocalThreeRankLayer(HcclComm comm, const OpParam &param, RelayTopologyInfo &topology)
{
    topology = RelayTopologyInfo{};
    if (param.rankSize != 12) {
        return false;
    }

    uint32_t *netLayersRaw = nullptr;
    uint32_t netLayerNum = 0;
    const HcclResult layerRet = HcclRankGraphGetLayers(comm, &netLayersRaw, &netLayerNum);
    if (layerRet != HCCL_SUCCESS || netLayersRaw == nullptr || netLayerNum == 0) {
        return false;
    }

    const std::vector<uint32_t> netLayers(netLayersRaw, netLayersRaw + netLayerNum);
    const std::vector<uint32_t> topo4x3 = {3, 3, 3, 3};

    for (uint32_t netLayer : netLayers) {
        // Do NOT gate this on HcclRankGraphGetInstSizeListByLayer().
        // HCCL has LOCAL-view physical levels where the sibling-instance size
        // list is unavailable, while GetRanksByLayer still exposes the current
        // rank's topology instance.
        uint32_t *localRanksRaw = nullptr;
        uint32_t localRankNum = 0;
        const HcclResult ranksRet =
            HcclRankGraphGetRanksByLayer(comm, netLayer, &localRanksRaw, &localRankNum);
        if (ranksRet != HCCL_SUCCESS || localRanksRaw == nullptr || localRankNum != 3) {
            continue;
        }

        std::vector<uint32_t> localRanks(localRanksRaw, localRanksRaw + localRankNum);
        std::sort(localRanks.begin(), localRanks.end());
        if (std::find(localRanks.begin(), localRanks.end(), param.myRank) == localRanks.end()) {
            continue;
        }

        // A 3-rank local topology instance is unique to the 4x3 competition
        // topology.  8+4 can only expose local Server instances of size 8 or 4.
        //
        // First try to reconstruct all four Server components directly on this
        // same layer using arbitrary-rank link queries.
        std::vector<uint32_t> serverIds;
        std::vector<uint32_t> componentSizes;
        if (TryBuildServerIdsFromLayer(comm, netLayer, param.rankSize, serverIds, componentSizes)) {
            std::sort(componentSizes.begin(), componentSizes.end());
            if (componentSizes == topo4x3 &&
                Validate4x3ServerPartition(comm, netLayer, serverIds)) {
                topology.numerator = 5;
                topology.denominator = 6;
                topology.serverIds = std::move(serverIds);
                HCCL_INFO("[%s] activate 4x3 relay from local-3 layer connectivity, layer[%u], ratio[5/6]",
                    __func__, netLayer);
                return true;
            }
        }

        // Fallback A: Server-major communicator ranks:
        // {0,1,2}, {3,4,5}, {6,7,8}, {9,10,11}.
        {
            std::vector<uint32_t> candidate(param.rankSize, 0);
            for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
                candidate[rank] = rank / 3;
            }

            const uint32_t localServer = candidate[param.myRank];
            std::vector<uint32_t> expectedLocal;
            for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
                if (candidate[rank] == localServer) {
                    expectedLocal.push_back(rank);
                }
            }

            if (expectedLocal == localRanks) {
                topology.numerator = 5;
                topology.denominator = 6;
                topology.serverIds = std::move(candidate);
                HCCL_INFO("[%s] activate 4x3 relay from local-3 server-major layout, layer[%u], ratio[5/6]",
                    __func__, netLayer);
                return true;
            }
        }

        // Fallback B: interleaved communicator ranks:
        // {0,4,8}, {1,5,9}, {2,6,10}, {3,7,11}.
        {
            std::vector<uint32_t> candidate(param.rankSize, 0);
            for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
                candidate[rank] = rank % 4;
            }

            const uint32_t localServer = candidate[param.myRank];
            std::vector<uint32_t> expectedLocal;
            for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
                if (candidate[rank] == localServer) {
                    expectedLocal.push_back(rank);
                }
            }

            if (expectedLocal == localRanks) {
                topology.numerator = 5;
                topology.denominator = 6;
                topology.serverIds = std::move(candidate);
                HCCL_INFO("[%s] activate 4x3 relay from local-3 interleaved layout, layer[%u], ratio[5/6]",
                    __func__, netLayer);
                return true;
            }
        }
    }

    return false;
}

bool DetectRelayTopologyFromEndpointServerIdx(const OpParam &param,
    const std::vector<PeerChannelInfo> &peerChannels, RelayTopologyInfo &topology)
{
    topology = RelayTopologyInfo{};
    if ((param.rankSize != 12 && param.rankSize != 16) ||
        peerChannels.size() != static_cast<size_t>(param.rankSize - 1)) {
        return false;
    }

    std::vector<uint32_t> serverIds(param.rankSize, INVALID_VALUE_RANKID);
    uint32_t localServerIdx = INVALID_VALUE_RANKID;
    for (const auto &peer : peerChannels) {
        if (peer.peerRank >= param.rankSize || peer.peerRank == param.myRank ||
            peer.localServerIdx == INVALID_VALUE_RANKID ||
            peer.remoteServerIdx == INVALID_VALUE_RANKID) {
            return false;
        }

        if (localServerIdx == INVALID_VALUE_RANKID) {
            localServerIdx = peer.localServerIdx;
        } else if (peer.localServerIdx != localServerIdx) {
            HCCL_WARNING("[%s] inconsistent local serverIdx[%u/%u]",
                __func__, localServerIdx, peer.localServerIdx);
            return false;
        }
        serverIds[peer.peerRank] = peer.remoteServerIdx;
    }
    if (localServerIdx == INVALID_VALUE_RANKID) {
        return false;
    }
    serverIds[param.myRank] = localServerIdx;

    std::vector<uint32_t> uniqueServerIds;
    std::vector<uint32_t> serverSizes;
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (serverIds[rank] == INVALID_VALUE_RANKID) {
            return false;
        }

        auto it = std::find(uniqueServerIds.begin(), uniqueServerIds.end(), serverIds[rank]);
        if (it == uniqueServerIds.end()) {
            uniqueServerIds.push_back(serverIds[rank]);
            serverSizes.push_back(1);
        } else {
            ++serverSizes[static_cast<size_t>(it - uniqueServerIds.begin())];
        }
    }
    std::sort(serverSizes.begin(), serverSizes.end());

    const std::vector<uint32_t> topo2x8 = {8, 8};
    const std::vector<uint32_t> topo4x3 = {3, 3, 3, 3};
    if (param.rankSize == 16 && serverSizes == topo2x8) {
        topology.numerator = 4;
        topology.denominator = 11;
    } else if (param.rankSize == 12 && serverSizes == topo4x3) {
        topology.numerator = 5;
        topology.denominator = 6;
    } else {
        // 8+4 is {4,8}; keep it on the validated direct path.
        return false;
    }

    topology.serverIds = std::move(serverIds);
    HCCL_INFO("[%s] relay topology activated from endpoint serverIdx, rankSize[%u], ratio[%u/%u]",
        __func__, param.rankSize, topology.numerator, topology.denominator);
    return true;
}

bool DetectRelayTopologyFromConnectivity(HcclComm comm, const OpParam &param, RelayTopologyInfo &topology)
{
    topology = RelayTopologyInfo{};
    if (param.rankSize != 12 && param.rankSize != 16) {
        return false;
    }

    uint32_t *netLayersRaw = nullptr;
    uint32_t netLayerNum = 0;
    const HcclResult layerRet = HcclRankGraphGetLayers(comm, &netLayersRaw, &netLayerNum);
    if (layerRet != HCCL_SUCCESS || netLayersRaw == nullptr || netLayerNum == 0) {
        HCCL_WARNING("[%s] rank graph layers unavailable, ret[%d], layerNum[%u]; use direct path",
            __func__, static_cast<int>(layerRet), netLayerNum);
        return false;
    }

    // Query every layer by observed connectivity.  This deliberately removes
    // both failed gates from earlier versions:
    //   V1.9: GetInstSizeListByLayer must exactly match the expected shape.
    //   V2.0: EndpointLoc.device.serverIdx must expose the physical Server.
    const std::vector<uint32_t> netLayers(netLayersRaw, netLayersRaw + netLayerNum);
    const std::vector<uint32_t> topo2x8 = {8, 8};
    const std::vector<uint32_t> topo4x3 = {3, 3, 3, 3};

    for (uint32_t netLayer : netLayers) {
        std::vector<uint32_t> serverIds;
        std::vector<uint32_t> componentSizes;
        if (!TryBuildServerIdsFromLayer(comm, netLayer, param.rankSize, serverIds, componentSizes)) {
            continue;
        }

        std::sort(componentSizes.begin(), componentSizes.end());
        if (param.rankSize == 16 && componentSizes == topo2x8) {
            topology.numerator = 4;
            topology.denominator = 11;
            topology.serverIds = std::move(serverIds);
            HCCL_INFO("[%s] activate 2x8 relay on layer[%u], ratio[4/11]",
                __func__, netLayer);
            return true;
        }

        if (param.rankSize == 12 && componentSizes == topo4x3) {
            topology.numerator = 5;
            topology.denominator = 6;
            topology.serverIds = std::move(serverIds);
            HCCL_INFO("[%s] activate 4x3 relay on layer[%u], ratio[5/6]",
                __func__, netLayer);
            return true;
        }
    }

    // 8+4, 4x1 and any unexpected RankGraph shape remain on the validated
    // direct path.
    return false;
}

bool EnableLargeCopyOverlap(const OpParam &param)
{
    auto sizeIt = SIZE_TABLE.find(param.dataType);
    if (sizeIt == SIZE_TABLE.end()) {
        return false;
    }
    const uint64_t recvBytes = param.count * static_cast<uint64_t>(sizeIt->second);
    return recvBytes > LARGE_MESSAGE_THRESHOLD;
}

HcclResult AcquireAllPeerChannels(HcclComm comm, const OpParam &param, uint32_t notifyNum,
    std::vector<PeerChannelInfo> &peerChannels)
{
    peerChannels.clear();
    peerChannels.reserve(param.rankSize > 0 ? param.rankSize - 1 : 0);

    uint32_t *netLayersRaw = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayersRaw, &netLayerNum));
    CHK_PRT_RET(netLayersRaw == nullptr || netLayerNum == 0,
        HCCL_ERROR("[%s] no valid rank graph layers", __func__), HCCL_E_NOT_FOUND);

    // RankGraph query results may be invalidated by subsequent topology queries.
    // Copy layer ids before calling HcclRankGraphGetLinks repeatedly.
    std::vector<uint32_t> netLayers(netLayersRaw, netLayersRaw + netLayerNum);

    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }

        HcclChannelDesc desc;
        CHK_RET(HcclChannelDescInit(&desc, 1));

        bool found = false;
        for (uint32_t layerIdx = 0; layerIdx < netLayers.size() && !found; ++layerIdx) {
            uint32_t listSize = 0;
            CommLink *linkList = nullptr;
            CHK_RET(HcclRankGraphGetLinks(
                comm, netLayers[layerIdx], param.myRank, remoteRank, &linkList, &listSize));

            for (uint32_t idx = 0; idx < listSize; ++idx) {
                const CommLink &link = linkList[idx];
                if (link.linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                    continue;
                }

                desc.remoteRank = remoteRank;
                desc.notifyNum = notifyNum;
                desc.channelProtocol = link.linkAttr.linkProtocol;
                desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
                desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
                desc.localEndpoint.loc = link.srcEndpointDesc.loc;
                desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
                desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
                desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
                found = true;
                break;
            }
        }

        CHK_PRT_RET(!found,
            HCCL_ERROR("[%s] UBC_CTP link not found between rank[%u] and rank[%u]",
                __func__, param.myRank, remoteRank),
            HCCL_E_NOT_FOUND);

        EndpointAttrDieId dieId{};
        CHK_RET(HcclRankGraphGetEndpointInfo(comm, param.myRank, &desc.localEndpoint,
            ENDPOINT_ATTR_DIE_ID, sizeof(dieId), &dieId));

        ChannelHandle channel = 0;
        CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, &desc, 1, &channel));
        peerChannels.push_back(PeerChannelInfo{remoteRank, channel, dieId,
            desc.localEndpoint.loc.device.serverIdx, desc.remoteEndpoint.loc.device.serverIdx});
    }

    return HCCL_SUCCESS;
}

HcclResult GroupChannelsByDie(const OpParam &param, const std::vector<PeerChannelInfo> &peerChannels,
    std::vector<DieChannelGroup> &groups)
{
    CHK_PRT_RET(peerChannels.size() != static_cast<size_t>(param.rankSize - 1),
        HCCL_ERROR("[%s] channel number[%zu] does not match rankSize[%u]",
            __func__, peerChannels.size(), param.rankSize),
        HCCL_E_INTERNAL);

    groups.clear();
    for (const auto &peer : peerChannels) {
        auto it = std::find_if(groups.begin(), groups.end(),
            [&peer](const DieChannelGroup &group) { return group.dieId == peer.dieId; });
        if (it == groups.end()) {
            DieChannelGroup group{};
            group.dieId = peer.dieId;
            group.peers.push_back(peer);
            groups.push_back(group);
        } else {
            it->peers.push_back(peer);
        }
    }

    std::sort(groups.begin(), groups.end(),
        [](const DieChannelGroup &lhs, const DieChannelGroup &rhs) { return lhs.dieId < rhs.dieId; });

    CHK_PRT_RET(groups.empty() || groups.size() > MAX_DIRECT_GROUP_NUM,
        HCCL_ERROR("[%s] unsupported die group number[%zu]", __func__, groups.size()), HCCL_E_NOT_SUPPORT);
    return HCCL_SUCCESS;
}

HcclResult GroupRelayChannelsByDieAndScope(const OpParam &param,
    const std::vector<PeerChannelInfo> &peerChannels, const RelayTopologyInfo &topology,
    std::vector<DieChannelGroup> &groups)
{
    CHK_PRT_RET(peerChannels.size() != static_cast<size_t>(param.rankSize - 1) ||
        topology.serverIds.size() != param.rankSize,
        HCCL_ERROR("[%s] invalid channel/topology metadata", __func__), HCCL_E_INTERNAL);

    groups.clear();
    const uint32_t myServerId = topology.serverIds[param.myRank];
    for (const auto &peer : peerChannels) {
        const bool isLocal = topology.serverIds[peer.peerRank] == myServerId;
        auto it = std::find_if(groups.begin(), groups.end(),
            [&peer, isLocal](const DieChannelGroup &group) {
                return group.dieId == peer.dieId && group.isLocalGroup == isLocal;
            });
        if (it == groups.end()) {
            DieChannelGroup group{};
            group.dieId = peer.dieId;
            group.isLocalGroup = isLocal;
            group.peers.push_back(peer);
            groups.push_back(group);
        } else {
            it->peers.push_back(peer);
        }
    }

    std::sort(groups.begin(), groups.end(), [](const DieChannelGroup &lhs, const DieChannelGroup &rhs) {
        if (lhs.dieId != rhs.dieId) {
            return lhs.dieId < rhs.dieId;
        }
        // Keep intra before inter on the same die for deterministic registration.
        return lhs.isLocalGroup && !rhs.isLocalGroup;
    });

    CHK_PRT_RET(groups.empty() || groups.size() > MAX_RELAY_GROUP_NUM,
        HCCL_ERROR("[%s] unsupported relay group number[%zu]", __func__, groups.size()), HCCL_E_NOT_SUPPORT);
    return HCCL_SUCCESS;
}

HcclResult RegisterDirectKernels(HcclComm comm, const OpParam &param, const std::vector<DieChannelGroup> &groups,
    AlgResourceCtx &resCtxHost)
{
    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1,
        HCCL_ERROR("[%s] HcclCommQueryCcuIns returned insNum[%u], expected 1", __func__, insNum),
        HCCL_E_INTERNAL);

    // V1.5 baseline: aggregate all same-local-die channels into one kernel.
    // Ascend 950 has two IO dies, so this keeps registration at one or two kernels
    // while satisfying the same-die restriction of a registered CCU kernel.
    resCtxHost.ccuKernels.clear();
    resCtxHost.ccuKernels.reserve(groups.size());
    resCtxHost.directKernelPeerMasks.clear();
    resCtxHost.directKernelPeerMasks.reserve(groups.size());

    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));

    for (uint32_t groupIdx = 0; groupIdx < groups.size(); ++groupIdx) {
        const auto &group = groups[groupIdx];
        CHK_PRT_RET(group.peers.empty() || group.peers.size() > MAX_RANK_SIZE - 1,
            HCCL_ERROR("[%s] invalid peer number[%zu] in die[%u]", __func__, group.peers.size(), group.dieId),
            HCCL_E_INTERNAL);

        auto kernelArg = std::make_shared<ops_hccl::CcuKernelArgScatterDirect>();
        kernelArg->rankSize = param.rankSize;
        kernelArg->rankId = param.myRank;
        kernelArg->channelCount = static_cast<uint32_t>(group.peers.size());
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            kernelArg->channels[i] = group.peers[i].channel;
            kernelArg->peerRanks[i] = group.peers[i].peerRank;
        }

        CcuKernelInfo kernelInfo{};
        std::snprintf(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName),
            "CcuScatterDirectKernel_die_%u", static_cast<uint32_t>(group.dieId));
        kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuScatterDirectKernel);
        kernelInfo.setKernelArg(kernelArg);

        const void *kernelArgs[] = {kernelInfo.kernelArg};
        constexpr uint32_t kernelArgNum = 1;

        CcuKernelHandle kernelHandle = 0;
        CHK_RET_CCU(HcommCcuKernelRegister(insHandle, static_cast<uint32_t>(group.dieId),
            kernelInfo.kernelFuncName, kernelInfo.kernelFunc, kernelArgs, kernelArgNum, &kernelHandle));
        resCtxHost.ccuKernels.push_back(kernelHandle);

        uint32_t peerMask = 0;
        for (const auto &peer : group.peers) {
            CHK_PRT_RET(peer.peerRank >= 32,
                HCCL_ERROR("[%s] peerRank[%u] exceeds direct mask capacity", __func__, peer.peerRank),
                HCCL_E_INTERNAL);
            peerMask |= (1U << peer.peerRank);
        }
        resCtxHost.directKernelPeerMasks.push_back(peerMask);
    }

    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
    return HCCL_SUCCESS;
}

HcclResult RegisterRelayKernels(HcclComm comm, const OpParam &param, const std::vector<DieChannelGroup> &groups,
    const RelayTopologyInfo &topology, AlgResourceCtx &resCtxHost)
{
    CHK_PRT_RET(topology.numerator == 0 || topology.denominator == 0 ||
        topology.serverIds.size() != param.rankSize,
        HCCL_ERROR("[%s] invalid relay topology metadata", __func__), HCCL_E_INTERNAL);

    std::vector<uint32_t> relayTargetByRank;
    std::vector<uint32_t> relaySourceByRank;
    const uint32_t pairNum = BuildScatterRelayPairs(
        topology.serverIds, param.myRank, relayTargetByRank, relaySourceByRank);
    uint32_t expectedPairNum = 0;
    const uint32_t rootServerId = topology.serverIds[param.myRank];
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (rank != param.myRank && topology.serverIds[rank] == rootServerId) {
            ++expectedPairNum;
        }
    }
    CHK_PRT_RET(pairNum != expectedPairNum,
        HCCL_ERROR("[%s] relay pair number[%u] does not match expected[%u]",
            __func__, pairNum, expectedPairNum),
        HCCL_E_INTERNAL);

    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1,
        HCCL_ERROR("[%s] HcclCommQueryCcuIns returned insNum[%u], expected 1", __func__, insNum),
        HCCL_E_INTERNAL);

    resCtxHost.ccuKernels.clear();
    resCtxHost.ccuKernels.reserve(groups.size());
    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));

    for (uint32_t groupIdx = 0; groupIdx < groups.size(); ++groupIdx) {
        const auto &group = groups[groupIdx];
        CHK_PRT_RET(group.peers.empty() || group.peers.size() > MAX_RANK_SIZE - 1,
            HCCL_ERROR("[%s] invalid peer number[%zu] in die[%u]", __func__, group.peers.size(), group.dieId),
            HCCL_E_INTERNAL);

        auto kernelArg = std::make_shared<ops_hccl::CcuKernelArgScatterRelay>();
        kernelArg->rankSize = param.rankSize;
        kernelArg->rankId = param.myRank;
        kernelArg->channelCount = static_cast<uint32_t>(group.peers.size());
        for (uint32_t rank = 0; rank < MAX_RANK_SIZE; ++rank) {
            kernelArg->rootRelayTargetByRank[rank] = INVALID_VALUE_RANKID;
            kernelArg->rootRelaySourceByRank[rank] = INVALID_VALUE_RANKID;
        }
        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            kernelArg->rootRelayTargetByRank[rank] = relayTargetByRank[rank];
            kernelArg->rootRelaySourceByRank[rank] = relaySourceByRank[rank];
        }
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            const uint32_t peerRank = group.peers[i].peerRank;
            kernelArg->channels[i] = group.peers[i].channel;
            kernelArg->peerRanks[i] = peerRank;
            kernelArg->peerIsLocal[i] =
                topology.serverIds[peerRank] == topology.serverIds[param.myRank] ? 1U : 0U;
        }

        CcuKernelInfo kernelInfo{};
        std::snprintf(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName),
            "CcuScatterRelayKernel_die_%u_%s", static_cast<uint32_t>(group.dieId),
            group.isLocalGroup ? "intra" : "inter");
        kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuScatterRelayKernel);
        kernelInfo.setKernelArg(kernelArg);

        const void *kernelArgs[] = {kernelInfo.kernelArg};
        constexpr uint32_t kernelArgNum = 1;
        CcuKernelHandle kernelHandle = 0;
        CHK_RET_CCU(HcommCcuKernelRegister(insHandle, static_cast<uint32_t>(group.dieId),
            kernelInfo.kernelFuncName, kernelInfo.kernelFunc, kernelArgs, kernelArgNum, &kernelHandle));
        resCtxHost.ccuKernels.push_back(kernelHandle);
    }

    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
    return HCCL_SUCCESS;
}

HcclResult AcquireThreadsForKernels(HcclComm comm, const OpParam &param, uint32_t kernelNum,
    AlgResourceCtx &resCtxHost)
{
    CHK_PRT_RET(kernelNum == 0 || kernelNum > MAX_RELAY_GROUP_NUM,
        HCCL_ERROR("[%s] invalid kernelNum[%u]", __func__, kernelNum), HCCL_E_INTERNAL);

    // V1.6-B: every large-message context gets one extra CCU thread.
    // It is used by root for the local self-copy so the copy can overlap
    // the die-parallel network kernels. Small messages keep the V1.5 layout.
    const uint32_t extraCopyThreadNum = EnableLargeCopyOverlap(param) ? 1U : 0U;
    const uint32_t totalThreadNum = kernelNum + extraCopyThreadNum;

    resCtxHost.ccuThread = param.cpuThread;
    resCtxHost.threads.resize(totalThreadNum);
    resCtxHost.threads[0] = param.cpuThread;
    if (totalThreadNum > 1) {
        CHK_RET(HcclThreadAcquire(comm, CommEngine::COMM_ENGINE_CCU,
            totalThreadNum - 1, SLAVE_NOTIFY_NUM, &resCtxHost.threads[1]));
    }
    return HCCL_SUCCESS;
}

} // namespace

HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param;
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.root = root;
    param.opType = HcclCMDType::HCCL_CMD_SCATTER;

    HcclDfxOpInfo dfxInfo;
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE,
        HCCL_ERROR("[%s] unsupported rankSize[%u]", __func__, param.rankSize), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(root >= param.rankSize,
        HCCL_ERROR("[%s] root[%u] is out of rankSize[%u]", __func__, root, param.rankSize), HCCL_E_PARA);

    auto sizeIt = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(sizeIt == SIZE_TABLE.end(),
        HCCL_ERROR("[%s] unsupported dataType[%d]", __func__, param.dataType), HCCL_E_NOT_SUPPORT);
    const bool enableLargeCopyOverlap = EnableLargeCopyOverlap(param);
    const bool relayCandidate = enableLargeCopyOverlap && (param.rankSize == 12 || param.rankSize == 16);
    std::snprintf(param.tag, sizeof(param.tag), "%s", enableLargeCopyOverlap
        ? "hccl_custom_scatter_ccu_v26c_direct_peer_kernel_prune"
        : "hccl_custom_scatter_ccu_v26c_direct_peer_kernel_prune_small");

    const CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;
    // 512KB stays byte-for-byte on the validated V1.5 resource shape. For
    // 12/16-rank large-message communicators reserve enough main-thread notify
    // slots for up to four split intra/inter relay kernels plus copy overlap.
    const uint32_t mainNotifyNum = !enableLargeCopyOverlap ? MAIN_NOTIFY_NUM_NORMAL
        : (relayCandidate ? MAIN_NOTIFY_NUM_RELAY_CANDIDATE : MAIN_NOTIFY_NUM_LARGE);
    CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, stream, mainNotifyNum, &param.cpuThread));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, ccuEngine, &ctx, &size) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = size;
    } else {
        AlgResourceCtx resCtxHost{};

        void *cclBufferAddr = nullptr;
        uint64_t cclBufferSize = 0;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        // Avoid depending on RankGraph layer numbering/instance shapes here.
        // The selected UBC_CTP links already expose both endpoint Server indices.
        const uint32_t channelNotifyNum = relayCandidate ? CHANNEL_NOTIFY_NUM_RELAY : CHANNEL_NOTIFY_NUM_DIRECT;
        std::vector<PeerChannelInfo> peerChannels;
        CHK_RET(AcquireAllPeerChannels(comm, param, channelNotifyNum, peerChannels));

        RelayTopologyInfo relayTopology;
        bool useRelay = false;
        if (relayCandidate) {
            if (param.rankSize == 16) {
                useRelay = Detect2x8FromLocalInstance(comm, param, relayTopology);
            } else if (param.rankSize == 12) {
                useRelay = Detect4x3FromLocalThreeRankLayer(comm, param, relayTopology);
            }
            if (!useRelay) {
                useRelay = DetectRelayTopologyFromEndpointServerIdx(param, peerChannels, relayTopology);
            }
            if (!useRelay) {
                useRelay = DetectRelayTopologyFromConnectivity(comm, param, relayTopology);
            }
        }

        std::vector<DieChannelGroup> groups;
        if (useRelay) {
            // Competition guidance requires intra-Server and inter-Server
            // communication to be orchestrated in different CCU kernels.
            CHK_RET(GroupRelayChannelsByDieAndScope(param, peerChannels, relayTopology, groups));
            resCtxHost.relayNumerator = relayTopology.numerator;
            resCtxHost.relayDenominator = relayTopology.denominator;
            resCtxHost.relayServerIds = relayTopology.serverIds;
            CHK_RET(RegisterRelayKernels(comm, param, groups, relayTopology, resCtxHost));
        } else {
            CHK_RET(GroupChannelsByDie(param, peerChannels, groups));
            CHK_RET(RegisterDirectKernels(comm, param, groups, resCtxHost));
        }
        CHK_RET(AcquireThreadsForKernels(comm, param, static_cast<uint32_t>(groups.size()), resCtxHost));

        std::vector<char> seq = resCtxHost.Serialize();
        param.ctxSize = seq.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, ccuEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, ccuEngine, param.tag, seq.data(), seq.size(), 0));
    }

    CHK_RET(ops_hccl::ExecOp(param));
    return HCCL_SUCCESS;
}
