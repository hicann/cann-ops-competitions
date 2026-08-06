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
#include <cstdio>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <ccu/ccu_launch.h>
#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>

#include "ccu_kernel.h"
#include "common.h"
#include "custom.h"
#include "exec_op.h"
#include "hccl.h"
#include "log.h"

namespace {

constexpr uint32_t CHANNEL_NOTIFY_NUM = 1U;
constexpr uint32_t THREAD_NOTIFY_NUM = 3U;
constexpr uint32_t SINGLE_DIE_THREAD_NOTIFY_NUM = 1U;
constexpr uint32_t MAX_DIE_NUM = 2U;
constexpr uint32_t MESH_LAYER_ID = 0U;
constexpr uint32_t CLOS_LAYER_ID = 1U;
constexpr uint32_t MESH_LOAD_UNITS = 4U;

struct TopologyView {
    FinalTopology topology{FinalTopology::TOPOLOGY_4X1};
    uint32_t meshLayer{ALLGATHER_INVALID_LAYER};
    CommTopo meshTopoType{COMM_TOPO_RESERVED};
    uint32_t closLayer{ALLGATHER_INVALID_LAYER};
    std::vector<uint32_t> localRanks;
    std::vector<uint32_t> remoteRanks;
};

struct LinkCandidate {
    // 始终按 lowRank -> highRank 保存，保证一对 Rank 在两端使用同一套比较键。
    CommLink link{};
    uint32_t lowRank{0};
    uint32_t highRank{0};
};

struct LayerPeerPlan {
    AllGatherLayerRole role{AllGatherLayerRole::CLOS};
    CommTopo topoType{COMM_TOPO_RESERVED};
    uint32_t layer{ALLGATHER_INVALID_LAYER};
    std::vector<uint32_t> peerRanks;
};

struct SelectedChannel {
    HcclChannelDesc desc{};
    AllGatherLayerRole role{AllGatherLayerRole::CLOS};
    CommTopo topoType{COMM_TOPO_RESERVED};
    uint32_t layer{ALLGATHER_INVALID_LAYER};
    uint32_t peerRank{0};
    uint32_t actualDieId{0};
};

struct KernelGroup {
    AllGatherLayerRole role{AllGatherLayerRole::CLOS};
    CommTopo topoType{COMM_TOPO_RESERVED};
    uint32_t layer{ALLGATHER_INVALID_LAYER};
    uint32_t actualDieId{0};
    bool isPrimary{false};
    std::vector<uint32_t> peerRanks;
    std::vector<ChannelHandle> channels;
};

const char *DataPathName(AllGatherDataPath dataPath)
{
    switch (dataPath) {
        case AllGatherDataPath::GENERIC_DIRECT:
            return "generic-direct";
        case AllGatherDataPath::FIXED_SMALL:
            return "fixed-small";
        case AllGatherDataPath::ROLLING2:
            return "rolling-2";
        case AllGatherDataPath::SINGLE_DMA:
            return "single-dma";
        case AllGatherDataPath::HYBRID_RELAY:
            return "hybrid-relay";
        case AllGatherDataPath::HYBRID_RELAY_CKE:
            return "hybrid-relay-cke";
        case AllGatherDataPath::DISTRIBUTED_ROOT:
            return "distributed-root";
        case AllGatherDataPath::DISTRIBUTED_ROOT_CAPPED:
            return "distributed-root-capped";
        case AllGatherDataPath::DISTRIBUTED_ROOT_MATCHED_SERIAL:
            return "distributed-root-matched-serial";
        case AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT:
            return "bidirectional-half-root";
        case AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_CAPPED:
            return "bidirectional-half-root-capped";
        case AllGatherDataPath::MATCHED_READY:
            return "matched-ready";
        case AllGatherDataPath::MATCHED_SERIAL_ROLLING2:
            return "matched-serial-rolling2";
        case AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_MATCHED:
            return "bidirectional-half-root-matched";
        case AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_SHARED:
            return "bidirectional-half-root-shared";
        case AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_SHARED_5_8:
            return "bidirectional-half-root-shared-5-8";
        case AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_FULL_SEED_4_7:
            return "bidirectional-half-root-full-seed-4-7";
        case AllGatherDataPath::INVALID:
            return "invalid";
        default:
            return "invalid";
    }
}

const char *ProfileName(AllGatherProfile profile)
{
    switch (profile) {
        case AllGatherProfile::P2X8_512K:
            return "P2X8_512K";
        case AllGatherProfile::P2X8_512M:
            return "P2X8_512M";
        case AllGatherProfile::P2X8_400M4B:
            return "P2X8_400M4B";
        case AllGatherProfile::P4X1_512K:
            return "P4X1_512K";
        case AllGatherProfile::P4X1_512M:
            return "P4X1_512M";
        case AllGatherProfile::P4X1_400M4B:
            return "P4X1_400M4B";
        case AllGatherProfile::P8P4_512K:
            return "P8P4_512K";
        case AllGatherProfile::P8P4_512M:
            return "P8P4_512M";
        case AllGatherProfile::P8P4_400M4B:
            return "P8P4_400M4B";
        case AllGatherProfile::GENERIC:
            return "GENERIC";
        default:
            return "unknown";
    }
}

bool GetDataTypeSize(HcclDataType dataType, uint64_t &size)
{
    switch (dataType) {
        case HCCL_DATA_TYPE_INT8:
        case HCCL_DATA_TYPE_UINT8:
        case HCCL_DATA_TYPE_HIF8:
        case HCCL_DATA_TYPE_FP8E4M3:
        case HCCL_DATA_TYPE_FP8E5M2:
        case HCCL_DATA_TYPE_FP8E8M0:
            size = 1U;
            return true;
        case HCCL_DATA_TYPE_INT16:
        case HCCL_DATA_TYPE_UINT16:
        case HCCL_DATA_TYPE_FP16:
        case HCCL_DATA_TYPE_BFP16:
            size = 2U;
            return true;
        case HCCL_DATA_TYPE_INT32:
        case HCCL_DATA_TYPE_UINT32:
        case HCCL_DATA_TYPE_FP32:
            size = 4U;
            return true;
        case HCCL_DATA_TYPE_INT64:
        case HCCL_DATA_TYPE_UINT64:
        case HCCL_DATA_TYPE_FP64:
            size = 8U;
            return true;
        case HCCL_DATA_TYPE_INT128:
            size = 16U;
            return true;
        default:
            return false;
    }
}

HcclResult GetTopologyByRankSize(uint32_t rankSize, FinalTopology &topology)
{
    switch (rankSize) {
        case 4U:
            topology = FinalTopology::TOPOLOGY_4X1;
            return HCCL_SUCCESS;
        case 12U:
            topology = FinalTopology::TOPOLOGY_8X4;
            return HCCL_SUCCESS;
        case 16U:
            topology = FinalTopology::TOPOLOGY_2X8;
            return HCCL_SUCCESS;
        default:
            HCCL_ERROR("[GetTopologyByRankSize] unsupported rank size %u", rankSize);
            return HCCL_E_NOT_SUPPORT;
    }
}

HcclResult GetExpectedTopology(
    uint32_t rankSize, FinalTopology &topology, std::vector<uint32_t> &expectedInstanceSizes)
{
    CHK_RET(GetTopologyByRankSize(rankSize, topology));
    if (topology == FinalTopology::TOPOLOGY_4X1) {
        expectedInstanceSizes = {1U, 1U, 1U, 1U};
    } else if (topology == FinalTopology::TOPOLOGY_8X4) {
        expectedInstanceSizes = {4U, 8U};
    } else {
        expectedInstanceSizes = {8U, 8U};
    }
    return HCCL_SUCCESS;
}

HcclResult GetRankGraphLayers(HcclComm comm, std::vector<uint32_t> &layers)
{
    uint32_t *layerData = nullptr;
    uint32_t layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layerData, &layerCount));
    CHK_PRT_RET(layerData == nullptr || layerCount == 0U,
        HCCL_ERROR("[GetRankGraphLayers] rank graph has no layer"), HCCL_E_INTERNAL);
    // RankGraph 返回的内存由库管理，必须立即复制，避免后续查询使指针失效。
    layers.assign(layerData, layerData + layerCount);
    return HCCL_SUCCESS;
}

HcclResult GetInstanceSizes(HcclComm comm, uint32_t layer, std::vector<uint32_t> &instanceSizes)
{
    uint32_t *sizeData = nullptr;
    uint32_t sizeCount = 0;
    CHK_RET(HcclRankGraphGetInstSizeListByLayer(comm, layer, &sizeData, &sizeCount));
    CHK_PRT_RET(sizeData == nullptr || sizeCount == 0U,
        HCCL_ERROR("[GetInstanceSizes] layer %u has no instance", layer), HCCL_E_INTERNAL);
    instanceSizes.assign(sizeData, sizeData + sizeCount);
    std::sort(instanceSizes.begin(), instanceSizes.end());
    return HCCL_SUCCESS;
}

HcclResult GetRanksInLayer(HcclComm comm, uint32_t layer, uint32_t rankSize, std::vector<uint32_t> &ranks)
{
    uint32_t *rankData = nullptr;
    uint32_t rankCount = 0;
    CHK_RET(HcclRankGraphGetRanksByLayer(comm, layer, &rankData, &rankCount));
    CHK_PRT_RET(rankData == nullptr || rankCount == 0U,
        HCCL_ERROR("[GetRanksInLayer] layer %u has no rank", layer), HCCL_E_INTERNAL);
    ranks.assign(rankData, rankData + rankCount);
    std::sort(ranks.begin(), ranks.end());
    CHK_PRT_RET(std::adjacent_find(ranks.begin(), ranks.end()) != ranks.end(),
        HCCL_ERROR("[GetRanksInLayer] layer %u has duplicate ranks", layer), HCCL_E_INTERNAL);
    CHK_PRT_RET(ranks.back() >= rankSize,
        HCCL_ERROR("[GetRanksInLayer] layer %u has out-of-range rank %u", layer, ranks.back()), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult DetectFinalTopology(HcclComm comm, const OpParam &param, TopologyView &view)
{
    if (param.rankSize == ALLGATHER_P4X1_RANK_COUNT) {
        view.topology = FinalTopology::TOPOLOGY_4X1;
        CommTopo closTopo = COMM_TOPO_RESERVED;
        CHK_RET(HcclRankGraphGetTopoTypeByLayer(comm, CLOS_LAYER_ID, &closTopo));
        CHK_PRT_RET(closTopo != COMM_TOPO_CLOS,
            HCCL_ERROR("[DetectFinalTopology] layer-1 is not Clos"), HCCL_E_NOT_SUPPORT);
        std::vector<uint32_t> closRanks;
        CHK_RET(GetRanksInLayer(comm, CLOS_LAYER_ID, param.rankSize, closRanks));
        CHK_PRT_RET(closRanks.size() != param.rankSize,
            HCCL_ERROR("[DetectFinalTopology] 4x1 Clos rank set is incomplete"), HCCL_E_NOT_SUPPORT);
        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            CHK_PRT_RET(closRanks[rank] != rank,
                HCCL_ERROR("[DetectFinalTopology] 4x1 Clos misses rank %u", rank),
                HCCL_E_NOT_SUPPORT);
            if (rank != param.myRank) {
                view.remoteRanks.push_back(rank);
            }
        }
        view.closLayer = CLOS_LAYER_ID;
        view.localRanks = {param.myRank};
        return HCCL_SUCCESS;
    }

    std::vector<uint32_t> expectedInstanceSizes;
    CHK_RET(GetExpectedTopology(param.rankSize, view.topology, expectedInstanceSizes));

    std::vector<uint32_t> layers;
    CHK_RET(GetRankGraphLayers(comm, layers));
    bool foundMesh = false;
    bool foundClos = false;
    bool hasLayerZero = false;
    for (uint32_t layer : layers) {
        // 本题硬件口径固定为 layer-0 Server 内 Mesh、layer-1 Server 间 Clos。
        // 其余层可能是 Host/RoCE 等附加网络，不能误选或当作重复候选报错。
        if (layer != MESH_LAYER_ID && layer != CLOS_LAYER_ID) {
            continue;
        }
        hasLayerZero = hasLayerZero || layer == MESH_LAYER_ID;
        CommTopo topoType = COMM_TOPO_RESERVED;
        CHK_RET(HcclRankGraphGetTopoTypeByLayer(comm, layer, &topoType));

        std::vector<uint32_t> instanceSizes;
        CHK_RET(GetInstanceSizes(comm, layer, instanceSizes));
        uint64_t coveredRanks = 0;
        for (uint32_t instanceSize : instanceSizes) {
            CHK_PRT_RET(instanceSize == 0U || instanceSize > param.rankSize,
                HCCL_ERROR("[DetectFinalTopology] layer %u has invalid instance size %u", layer, instanceSize),
                HCCL_E_INTERNAL);
            coveredRanks += instanceSize;
        }
        CHK_PRT_RET(coveredRanks != param.rankSize,
            HCCL_ERROR("[DetectFinalTopology] layer %u covers %lu ranks, expected %u",
                layer, coveredRanks, param.rankSize),
            HCCL_E_NOT_SUPPORT);

        std::vector<uint32_t> ranks;
        CHK_RET(GetRanksInLayer(comm, layer, param.rankSize, ranks));
        CHK_PRT_RET(!std::binary_search(ranks.begin(), ranks.end(), param.myRank),
            HCCL_ERROR("[DetectFinalTopology] layer %u does not contain local rank %u", layer, param.myRank),
            HCCL_E_INTERNAL);

        // 同一 Server 的 Full-Mesh 在平台 RankGraph 中可能表示为 1DMESH 或 CUSTOM；
        // 实例形状用于确认物理 Server 分组，实际 topoType 则原样带入 Channel 元数据。
        if (layer == MESH_LAYER_ID
            && (topoType == COMM_TOPO_1DMESH || topoType == COMM_TOPO_CUSTOM)
            && instanceSizes == expectedInstanceSizes) {
            view.meshLayer = layer;
            view.meshTopoType = topoType;
            view.localRanks = ranks;
            foundMesh = true;
        }
        if (layer == CLOS_LAYER_ID && topoType == COMM_TOPO_CLOS && instanceSizes.size() == 1U
            && instanceSizes[0] == param.rankSize) {
            CHK_PRT_RET(ranks.size() != param.rankSize,
                HCCL_ERROR("[DetectFinalTopology] Clos layer %u has incomplete rank set", layer),
                HCCL_E_NOT_SUPPORT);
            for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
                CHK_PRT_RET(ranks[rank] != rank,
                    HCCL_ERROR("[DetectFinalTopology] Clos layer %u misses rank %u", layer, rank),
                    HCCL_E_NOT_SUPPORT);
            }
            view.closLayer = layer;
            foundClos = true;
        }
    }

    CHK_PRT_RET(!foundClos,
        HCCL_ERROR("[DetectFinalTopology] rank graph has no full-rank Clos layer"), HCCL_E_NOT_SUPPORT);
    if (!foundMesh) {
        // 4x1 的本地实例只有本 Rank，部分 RankGraph 会省略不产生通信的 singleton Mesh 层。
        // 若 layer-0 明确存在但形状不匹配，则不能把同 Server 的 1x4 误判成 4x1。
        CHK_PRT_RET(view.topology != FinalTopology::TOPOLOGY_4X1 || hasLayerZero,
            HCCL_ERROR("[DetectFinalTopology] layer-0 Mesh is missing or has an unexpected shape"),
            HCCL_E_NOT_SUPPORT);
        view.localRanks = {param.myRank};
    } else {
        CHK_PRT_RET(view.meshLayer == view.closLayer,
            HCCL_ERROR("[DetectFinalTopology] Mesh and Clos layers are not distinct"), HCCL_E_NOT_SUPPORT);
        CHK_PRT_RET(std::find(expectedInstanceSizes.begin(), expectedInstanceSizes.end(), view.localRanks.size())
                        == expectedInstanceSizes.end(),
            HCCL_ERROR("[DetectFinalTopology] local Mesh instance size %zu is invalid", view.localRanks.size()),
            HCCL_E_NOT_SUPPORT);
    }

    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (!std::binary_search(view.localRanks.begin(), view.localRanks.end(), rank)) {
            view.remoteRanks.push_back(rank);
        }
    }
    CHK_PRT_RET(view.localRanks.size() + view.remoteRanks.size() != param.rankSize,
        HCCL_ERROR("[DetectFinalTopology] local/remote rank partition is incomplete"), HCCL_E_INTERNAL);

    return HCCL_SUCCESS;
}

HcclResult GetEndpointDieId(
    HcclComm comm, uint32_t rank, const EndpointDesc &endpoint, uint32_t &dieId)
{
    EndpointAttrDieId endpointDie{};
    CHK_RET(HcclRankGraphGetEndpointInfo(
        comm, rank, &endpoint, ENDPOINT_ATTR_DIE_ID, sizeof(endpointDie), &endpointDie));
    CHK_PRT_RET(endpointDie >= MAX_DIE_NUM,
        HCCL_ERROR("[GetEndpointDieId] invalid endpoint die %u for rank %u", endpointDie, rank),
        HCCL_E_INTERNAL);
    dieId = endpointDie;
    return HCCL_SUCCESS;
}

HcclResult CollectLinkCandidates(HcclComm comm, const OpParam &param,
    uint32_t layer, uint32_t peerRank, std::vector<LinkCandidate> &candidates)
{
    CHK_PRT_RET(peerRank == param.myRank || peerRank >= param.rankSize,
        HCCL_ERROR("[CollectLinkCandidates] invalid local/peer rank %u/%u", param.myRank, peerRank),
        HCCL_E_PARA);
    CommLink *links = nullptr;
    uint32_t linkCount = 0;
    // 按官方路径只查询 myRank -> peer，避免访问与本 Rank 无关的 RankGraph 边。
    CHK_RET(HcclRankGraphGetLinks(
        comm, layer, param.myRank, peerRank, &links, &linkCount));
    CHK_PRT_RET(linkCount == 0U,
        HCCL_ERROR("[CollectLinkCandidates] no link on layer %u from rank %u to %u",
            layer, param.myRank, peerRank), HCCL_E_NOT_FOUND);
    CHK_PRT_RET(links == nullptr,
        HCCL_ERROR("[CollectLinkCandidates] layer %u returned null links", layer), HCCL_E_INTERNAL);
    // RankGraph 返回的整个数组由库管理，先一次性复制后再筛选。
    const std::vector<CommLink> linkCopies(links, links + linkCount);
    for (const CommLink &link : linkCopies) {
        const CommProtocol protocol = link.linkAttr.linkProtocol;
        if (protocol != COMM_PROTOCOL_UBC_CTP && protocol != COMM_PROTOCOL_UBC_TP) {
            continue;
        }
        LinkCandidate candidate{link, std::min(param.myRank, peerRank),
            std::max(param.myRank, peerRank)};
        if (param.myRank > peerRank) {
            // 两端查询方向相反；高 Rank 交换副本后统一为 lowRank -> highRank。
            std::swap(candidate.link.srcEndpointDesc, candidate.link.dstEndpointDesc);
        }
        candidates.push_back(candidate);
    }
    CHK_PRT_RET(candidates.empty(),
        HCCL_ERROR("[CollectLinkCandidates] no CTP/TP link on layer %u from rank %u to %u",
            layer, param.myRank, peerRank),
        HCCL_E_NOT_FOUND);
    return HCCL_SUCCESS;
}

template <size_t N>
int CompareBytes(const uint8_t (&left)[N], const uint8_t (&right)[N])
{
    if (std::lexicographical_compare(left, left + N, right, right + N)) {
        return -1;
    }
    return std::lexicographical_compare(right, right + N, left, left + N) ? 1 : 0;
}

int CompareCommAddress(const CommAddr &left, const CommAddr &right)
{
    if (left.type != right.type) {
        return static_cast<int32_t>(left.type) < static_cast<int32_t>(right.type) ? -1 : 1;
    }
    switch (left.type) {
        case COMM_ADDR_TYPE_IP_V4:
            if (left.addr.s_addr == right.addr.s_addr) {
                return 0;
            }
            return left.addr.s_addr < right.addr.s_addr ? -1 : 1;
        case COMM_ADDR_TYPE_IP_V6:
            return CompareBytes(left.addr6.s6_addr, right.addr6.s6_addr);
        case COMM_ADDR_TYPE_ID:
            if (left.id == right.id) {
                return 0;
            }
            return left.id < right.id ? -1 : 1;
        case COMM_ADDR_TYPE_EID:
            return CompareBytes(left.eid, right.eid);
        default:
            // 保留类型没有可作为规范键的有效载荷；上层协议和另一端地址继续打破平局。
            return 0;
    }
}

int CompareEndpoint(const EndpointDesc &left, const EndpointDesc &right)
{
    if (left.protocol != right.protocol) {
        return static_cast<int32_t>(left.protocol) < static_cast<int32_t>(right.protocol) ? -1 : 1;
    }
    // loc、superDevId、union 未使用尾部和保留区都不是 Rank 对共享的稳定选路键。
    return CompareCommAddress(left.commAddr, right.commAddr);
}

int CompareLinkCandidate(const LinkCandidate &candidate, const LinkCandidate &selected)
{
    const CommProtocol candidateProtocol = candidate.link.linkAttr.linkProtocol;
    const CommProtocol selectedProtocol = selected.link.linkAttr.linkProtocol;
    if (candidateProtocol != selectedProtocol) {
        return candidateProtocol == COMM_PROTOCOL_UBC_CTP ? -1 : 1;
    }
    if (candidate.link.linkAttr.hop != selected.link.linkAttr.hop) {
        return candidate.link.linkAttr.hop < selected.link.linkAttr.hop ? -1 : 1;
    }
    const int sourceCompare
        = CompareEndpoint(candidate.link.srcEndpointDesc, selected.link.srcEndpointDesc);
    if (sourceCompare != 0) {
        return sourceCompare;
    }
    return CompareEndpoint(candidate.link.dstEndpointDesc, selected.link.dstEndpointDesc);
}

bool IsBetterCandidate(const LinkCandidate &candidate, const LinkCandidate &selected)
{
    // 协议、hop 和 canonical Endpoint 描述可由两个 Rank 完全一致地复现。
    return CompareLinkCandidate(candidate, selected) < 0;
}

const LinkCandidate *SelectBestCandidate(const std::vector<LinkCandidate> &candidates)
{
    const LinkCandidate *selected = nullptr;
    for (const LinkCandidate &candidate : candidates) {
        if (selected == nullptr || IsBetterCandidate(candidate, *selected)) {
            selected = &candidate;
        }
    }
    return selected;
}

HcclResult BuildSelectedChannel(HcclComm comm, const LayerPeerPlan &plan, uint32_t myRank,
    uint32_t peerRank, const LinkCandidate &candidate, SelectedChannel &channel)
{
    CHK_RET(HcclChannelDescInit(&channel.desc, 1U));
    channel.desc.remoteRank = peerRank;
    channel.desc.channelProtocol = candidate.link.linkAttr.linkProtocol;
    const bool localIsLowRank = myRank == candidate.lowRank;
    channel.desc.localEndpoint
        = localIsLowRank ? candidate.link.srcEndpointDesc : candidate.link.dstEndpointDesc;
    channel.desc.remoteEndpoint
        = localIsLowRank ? candidate.link.dstEndpointDesc : candidate.link.srcEndpointDesc;
    channel.desc.notifyNum = CHANNEL_NOTIFY_NUM;
    channel.role = plan.role;
    channel.topoType = plan.topoType;
    channel.layer = plan.layer;
    channel.peerRank = peerRank;
    // Kernel 分组只依赖本 Rank 的 Endpoint Die；带宽属性不参与选链或执行。
    CHK_RET(GetEndpointDieId(
        comm, myRank, channel.desc.localEndpoint, channel.actualDieId));
    return HCCL_SUCCESS;
}

HcclResult SelectChannelsForPlanByPeer(HcclComm comm, const OpParam &param,
    const LayerPeerPlan &plan, std::vector<SelectedChannel> &selected)
{
    CHK_PRT_RET(plan.peerRanks.empty() || plan.layer == ALLGATHER_INVALID_LAYER,
        HCCL_ERROR("[SelectChannelsForPlanByPeer] empty or invalid layer plan"), HCCL_E_PARA);
    const size_t firstSelected = selected.size();
    selected.reserve(selected.size() + plan.peerRanks.size());
    for (uint32_t peerRank : plan.peerRanks) {
        std::vector<LinkCandidate> candidates;
        CHK_RET(CollectLinkCandidates(comm, param, plan.layer, peerRank, candidates));
        // 两端使用完全相同的 lowRank -> highRank 候选和规范键，先确定物理 Link；
        // 随后查询本地 Endpoint 的 Die，验证同一层的网络设备只属于一个 IO Die。
        const LinkCandidate *candidate = SelectBestCandidate(candidates);
        CHK_PRT_RET(candidate == nullptr,
            HCCL_ERROR("[SelectChannelsForPlanByPeer] no usable candidate for peer %u", peerRank),
            HCCL_E_NOT_SUPPORT);
        SelectedChannel channel;
        CHK_RET(BuildSelectedChannel(comm, plan, param.myRank, peerRank, *candidate, channel));
        selected.push_back(channel);
    }
    {
        const uint32_t layerDie = selected[firstSelected].actualDieId;
        const bool spansDies = std::any_of(selected.begin() + firstSelected, selected.end(),
            [layerDie](const SelectedChannel &channel) { return channel.actualDieId != layerDie; });
        CHK_PRT_RET(spansDies,
            HCCL_ERROR("[SelectChannelsForPlanByPeer] one layer spans multiple local Dies"),
            HCCL_E_NOT_SUPPORT);
    }
    return HCCL_SUCCESS;
}

uint32_t GroupLoadUnits(const KernelGroup &group)
{
    // Server 内 Mesh Peer 使用独立链路并行发送，任意非空 Mesh 组的耗时固定按 4 建模；
    // Clos Peer 共享本 NPU 的 4B 总带宽，负载随 Peer 数线性增加。
    return group.role == AllGatherLayerRole::LOCAL_MESH
               ? MESH_LOAD_UNITS
               : static_cast<uint32_t>(group.channels.size());
}

HcclResult PromotePrimaryGroup(std::vector<KernelGroup> &groups)
{
    std::array<uint32_t, MAX_DIE_NUM> dieLoads{};
    std::array<bool, MAX_DIE_NUM> dieActive{};
    for (const KernelGroup &group : groups) {
        CHK_PRT_RET(group.actualDieId >= MAX_DIE_NUM,
            HCCL_ERROR("[PromotePrimaryGroup] invalid Die %u", group.actualDieId), HCCL_E_INTERNAL);
        dieActive[group.actualDieId] = true;
        dieLoads[group.actualDieId] += GroupLoadUnits(group);
    }

    uint32_t primaryDie = MAX_DIE_NUM;
    for (uint32_t dieId = 0; dieId < MAX_DIE_NUM; ++dieId) {
        if (dieActive[dieId]
            && (primaryDie == MAX_DIE_NUM || dieLoads[dieId] < dieLoads[primaryDie])) {
            primaryDie = dieId;
        }
    }
    CHK_PRT_RET(primaryDie == MAX_DIE_NUM,
        HCCL_ERROR("[PromotePrimaryGroup] no active Die"), HCCL_E_INTERNAL);

    // 同一 Die 固定 Mesh -> Clos 下发，因此 primary 必须是该 Die 最先执行的组，
    // 让 out-of-place LocalCopy 尽早与两个 Die 的网络传输重叠。
    auto primary = std::find_if(groups.begin(), groups.end(), [primaryDie](const KernelGroup &group) {
        return group.actualDieId == primaryDie && group.role == AllGatherLayerRole::LOCAL_MESH;
    });
    if (primary == groups.end()) {
        primary = std::find_if(groups.begin(), groups.end(), [primaryDie](const KernelGroup &group) {
            return group.actualDieId == primaryDie && group.role == AllGatherLayerRole::CLOS;
        });
    }
    CHK_PRT_RET(primary == groups.end(),
        HCCL_ERROR("[PromotePrimaryGroup] primary Die has no Kernel group"), HCCL_E_INTERNAL);
    if (primary != groups.begin()) {
        std::iter_swap(groups.begin(), primary);
    }
    groups.front().isPrimary = true;
    return HCCL_SUCCESS;
}

HcclResult BuildLayerPeerPlans(const OpParam &param, const TopologyView &topology,
    const AllGatherProfileSpec &spec, std::vector<LayerPeerPlan> &plans)
{
    if (spec.layerPolicy == AllGatherLayerPolicy::CLOS_ONLY) {
        LayerPeerPlan closPlan;
        closPlan.role = AllGatherLayerRole::CLOS;
        closPlan.topoType = COMM_TOPO_CLOS;
        closPlan.layer = topology.closLayer;
        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            if (rank != param.myRank) {
                closPlan.peerRanks.push_back(rank);
            }
        }
        plans.push_back(std::move(closPlan));
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(spec.layerPolicy != AllGatherLayerPolicy::LOCAL_MESH_REMOTE_CLOS,
        HCCL_ERROR("[BuildLayerPeerPlans] unsupported layer policy"), HCCL_E_NOT_SUPPORT);
    LayerPeerPlan meshPlan;
    meshPlan.role = AllGatherLayerRole::LOCAL_MESH;
    meshPlan.topoType = topology.meshTopoType;
    meshPlan.layer = topology.meshLayer;
    for (uint32_t rank : topology.localRanks) {
        if (rank != param.myRank) {
            meshPlan.peerRanks.push_back(rank);
        }
    }
    LayerPeerPlan closPlan;
    closPlan.role = AllGatherLayerRole::CLOS;
    closPlan.topoType = COMM_TOPO_CLOS;
    closPlan.layer = topology.closLayer;
    closPlan.peerRanks = topology.remoteRanks;
    CHK_PRT_RET(meshPlan.peerRanks.empty() || closPlan.peerRanks.empty(),
        HCCL_ERROR("[BuildLayerPeerPlans] layered Profile requires non-empty Mesh and Clos peer sets"),
        HCCL_E_NOT_SUPPORT);
    // 这里只固定资源选择顺序，主组要在 Channel 申请完成后按实际 Peer 数决定。
    // 这样同 Die 冲突处理仍能明确区分 Mesh/Clos，同时 8+4 两侧可分别选择较轻的组。
    plans.push_back(std::move(meshPlan));
    plans.push_back(std::move(closPlan));
    return HCCL_SUCCESS;
}

HcclResult AuditKernelGroups(const OpParam &param, const TopologyView &topology,
    const AllGatherProfileSpec &spec, const std::vector<KernelGroup> &groups)
{
    CHK_PRT_RET(groups.empty() || groups.size() < spec.minKernelCount
                    || groups.size() > spec.maxKernelCount,
        HCCL_ERROR("[AuditKernelGroups] got %zu groups, Profile expects [%u, %u]",
            groups.size(), spec.minKernelCount, spec.maxKernelCount),
        HCCL_E_INTERNAL);
    uint32_t primaryCount = 0;
    uint32_t meshGroupCount = 0;
    uint32_t closGroupCount = 0;
    uint32_t meshDieMask = 0;
    uint32_t closDieMask = 0;
    std::array<uint32_t, MAX_DIE_NUM> dieLoads{};
    std::array<bool, MAX_DIE_NUM> dieActive{};
    std::vector<uint32_t> peers;
    std::vector<uint32_t> meshPeers;
    std::vector<uint32_t> closPeers;
    for (const KernelGroup &group : groups) {
        CHK_PRT_RET(group.channels.empty() || group.channels.size() != group.peerRanks.size()
                        || group.layer == ALLGATHER_INVALID_LAYER || group.actualDieId >= MAX_DIE_NUM,
            HCCL_ERROR("[AuditKernelGroups] invalid group shape"), HCCL_E_INTERNAL);
        primaryCount += group.isPrimary ? 1U : 0U;
        dieActive[group.actualDieId] = true;
        dieLoads[group.actualDieId] += GroupLoadUnits(group);
        if (group.role == AllGatherLayerRole::LOCAL_MESH) {
            const uint32_t dieBit = 1U << group.actualDieId;
            CHK_PRT_RET((meshDieMask & dieBit) != 0U
                            || group.topoType != topology.meshTopoType
                            || group.layer != topology.meshLayer,
                HCCL_ERROR("[AuditKernelGroups] duplicate or invalid Mesh die group"), HCCL_E_INTERNAL);
            meshDieMask |= dieBit;
            ++meshGroupCount;
            meshPeers.insert(meshPeers.end(), group.peerRanks.begin(), group.peerRanks.end());
        } else if (group.role == AllGatherLayerRole::CLOS) {
            const uint32_t dieBit = 1U << group.actualDieId;
            CHK_PRT_RET((closDieMask & dieBit) != 0U || group.topoType != COMM_TOPO_CLOS
                            || group.layer != topology.closLayer,
                HCCL_ERROR("[AuditKernelGroups] duplicate or invalid Clos group"), HCCL_E_INTERNAL);
            closDieMask |= dieBit;
            ++closGroupCount;
            closPeers.insert(closPeers.end(), group.peerRanks.begin(), group.peerRanks.end());
        } else {
            HCCL_ERROR("[AuditKernelGroups] invalid layer role");
            return HCCL_E_INTERNAL;
        }
        peers.insert(peers.end(), group.peerRanks.begin(), group.peerRanks.end());
    }
    CHK_PRT_RET(primaryCount != 1U || !groups.front().isPrimary,
        HCCL_ERROR("[AuditKernelGroups] primary group is not unique or not group 0"), HCCL_E_INTERNAL);
    const uint32_t primaryDie = groups.front().actualDieId;
    const bool primaryDieHasMesh = std::any_of(groups.begin(), groups.end(), [primaryDie](const KernelGroup &group) {
        return group.actualDieId == primaryDie && group.role == AllGatherLayerRole::LOCAL_MESH;
    });
    bool invalidPrimary = primaryDieHasMesh && groups.front().role != AllGatherLayerRole::LOCAL_MESH;
    for (uint32_t dieId = 0; dieId < MAX_DIE_NUM; ++dieId) {
        invalidPrimary = invalidPrimary || (dieActive[dieId]
            && (dieLoads[dieId] < dieLoads[primaryDie]
                || (dieLoads[dieId] == dieLoads[primaryDie] && dieId < primaryDie)));
    }
    CHK_PRT_RET(invalidPrimary,
        HCCL_ERROR("[AuditKernelGroups] primary is not the first group on the lightest Die"),
        HCCL_E_INTERNAL);
    const uint32_t expectedLoad
        = MESH_LOAD_UNITS * meshGroupCount + static_cast<uint32_t>(closPeers.size());
    CHK_PRT_RET(dieLoads[0] + dieLoads[1] != expectedLoad,
        HCCL_ERROR("[AuditKernelGroups] weighted Die load does not cover all peers"), HCCL_E_INTERNAL);

    std::sort(peers.begin(), peers.end());
    CHK_PRT_RET(peers.size() != param.rankSize - 1U
                    || std::adjacent_find(peers.begin(), peers.end()) != peers.end(),
        HCCL_ERROR("[AuditKernelGroups] peer coverage is duplicated or incomplete"), HCCL_E_INTERNAL);
    for (uint32_t rank = 0, peerIndex = 0; rank < param.rankSize; ++rank) {
        if (rank == param.myRank) {
            continue;
        }
        CHK_PRT_RET(peerIndex >= peers.size() || peers[peerIndex++] != rank,
            HCCL_ERROR("[AuditKernelGroups] peer %u is missing", rank), HCCL_E_INTERNAL);
    }

    if (spec.layerPolicy == AllGatherLayerPolicy::CLOS_ONLY) {
        std::sort(closPeers.begin(), closPeers.end());
        CHK_PRT_RET(meshGroupCount != 0U || closGroupCount != 1U || closPeers != peers,
            HCCL_ERROR("[AuditKernelGroups] Clos-only groups violate Profile"), HCCL_E_INTERNAL);
    } else if (spec.layerPolicy == AllGatherLayerPolicy::LOCAL_MESH_REMOTE_CLOS) {
        std::vector<uint32_t> expectedMeshPeers;
        for (uint32_t rank : topology.localRanks) {
            if (rank != param.myRank) {
                expectedMeshPeers.push_back(rank);
            }
        }
        std::sort(meshPeers.begin(), meshPeers.end());
        std::sort(closPeers.begin(), closPeers.end());
        const bool invalidMeshGroupCount = meshGroupCount != 1U;
        const bool invalidClosGroupCount = closGroupCount != 1U;
        // 赛题双层路径要求两层各自只落在一个 IO Die，具体 Die 编号由 Endpoint 决定。
        const bool invalidDieLayout = meshDieMask == 0U || closDieMask == 0U
                                      || (meshDieMask & closDieMask) != 0U;
        CHK_PRT_RET(invalidMeshGroupCount
                        || invalidClosGroupCount
                        || invalidDieLayout
                        || meshPeers != expectedMeshPeers || closPeers != topology.remoteRanks
                        || topology.meshLayer == topology.closLayer,
            HCCL_ERROR("[AuditKernelGroups] layered groups violate RankGraph peer partition"),
            HCCL_E_INTERNAL);
    } else {
        HCCL_ERROR("[AuditKernelGroups] unsupported layer policy");
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult AcquireChannels(HcclComm comm, const OpParam &param, const TopologyView &topology,
    const AllGatherProfileSpec &spec, std::vector<KernelGroup> &groups)
{
    std::vector<LayerPeerPlan> plans;
    CHK_RET(BuildLayerPeerPlans(param, topology, spec, plans));
    const size_t expectedPlanCount
        = spec.layerPolicy == AllGatherLayerPolicy::CLOS_ONLY ? 1U : 2U;
    CHK_PRT_RET(plans.size() != expectedPlanCount,
        HCCL_ERROR("[AcquireChannels] layer plan count does not match Profile"), HCCL_E_INTERNAL);

    std::vector<SelectedChannel> selected;
    selected.reserve(param.rankSize - 1U);
    // 固定按 layer-0 物理实例划分：同 Server 全部走 Mesh，跨 Server 全部走 Clos。
    // 8+4 因此在 8 卡侧为 7+4，在 4 卡侧为 3+8；每个 Peer 只进入一个 plan。
    for (const LayerPeerPlan &plan : plans) {
        CHK_RET(SelectChannelsForPlanByPeer(comm, param, plan, selected));
    }

    uint32_t selectedPeerMask = 0U;
    for (const SelectedChannel &channel : selected) {
        CHK_PRT_RET(channel.peerRank >= param.rankSize || channel.peerRank == param.myRank
                        || (selectedPeerMask & (1U << channel.peerRank)) != 0U,
            HCCL_ERROR("[AcquireChannels] duplicate or invalid Channel for peer %u", channel.peerRank),
            HCCL_E_INTERNAL);
        selectedPeerMask |= 1U << channel.peerRank;
    }
    const uint32_t expectedPeerMask
        = ((1U << param.rankSize) - 1U) & ~(1U << param.myRank);
    CHK_PRT_RET(selectedPeerMask != expectedPeerMask,
        HCCL_ERROR("[AcquireChannels] Channel plan does not cover every peer exactly once"),
        HCCL_E_INTERNAL);

    std::vector<HcclChannelDesc> descriptors;
    descriptors.reserve(selected.size());
    for (const SelectedChannel &channel : selected) {
        descriptors.push_back(channel.desc);
    }
    CHK_PRT_RET(descriptors.size() != param.rankSize - 1U,
        HCCL_ERROR("[AcquireChannels] selected %zu channels, expected %u",
            descriptors.size(), param.rankSize - 1U),
        HCCL_E_INTERNAL);

    std::vector<ChannelHandle> handles(descriptors.size());
    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, descriptors.data(),
        static_cast<uint32_t>(descriptors.size()), handles.data()));

    groups.reserve(spec.maxKernelCount);
    for (size_t channelIndex = 0; channelIndex < selected.size(); ++channelIndex) {
        const SelectedChannel &channel = selected[channelIndex];
        auto group = std::find_if(groups.begin(), groups.end(), [&channel](const KernelGroup &candidate) {
            return candidate.role == channel.role && candidate.topoType == channel.topoType
                   && candidate.layer == channel.layer && candidate.actualDieId == channel.actualDieId;
        });
        if (group == groups.end()) {
            KernelGroup newGroup;
            newGroup.role = channel.role;
            newGroup.topoType = channel.topoType;
            newGroup.layer = channel.layer;
            newGroup.actualDieId = channel.actualDieId;
            groups.push_back(std::move(newGroup));
            group = groups.end() - 1;
        }
        group->peerRanks.push_back(channel.peerRank);
        group->channels.push_back(handles[channelIndex]);
    }
    CHK_PRT_RET(groups.empty() || groups.size() > AlgResourceCtx::MAX_TRANSFER_KERNEL_COUNT,
        HCCL_ERROR("[AcquireChannels] selected %zu layer/die groups", groups.size()), HCCL_E_NOT_SUPPORT);

    CHK_RET(PromotePrimaryGroup(groups));
    return AuditKernelGroups(param, topology, spec, groups);
}

HcclResult ApplyP4X1MatchingOrder(const OpParam &param, std::vector<KernelGroup> &groups)
{
    CHK_PRT_RET(param.rankSize != ALLGATHER_P4X1_RANK_COUNT
                    || param.myRank >= ALLGATHER_P4X1_RANK_COUNT || groups.size() != 1U,
        HCCL_ERROR("[ApplyP4X1MatchingOrder] invalid rank %u/%u", param.myRank, param.rankSize),
        HCCL_E_INTERNAL);
    KernelGroup &group = groups.front();
    CHK_PRT_RET(group.role != AllGatherLayerRole::CLOS
                    || group.peerRanks.size() != ALLGATHER_P4X1_PEER_COUNT
                    || group.channels.size() != ALLGATHER_P4X1_PEER_COUNT,
        HCCL_ERROR("[ApplyP4X1MatchingOrder] invalid Clos group"), HCCL_E_INTERNAL);
    std::vector<uint32_t> orderedPeers;
    std::vector<ChannelHandle> orderedChannels;
    orderedPeers.reserve(ALLGATHER_P4X1_PEER_COUNT);
    orderedChannels.reserve(ALLGATHER_P4X1_PEER_COUNT);
    for (uint32_t orderIndex = 0U;
         orderIndex < ALLGATHER_P4X1_PEER_COUNT; ++orderIndex) {
        const uint32_t peerRank
            = ALLGATHER_P4X1_MATCHING_ORDER[param.myRank][orderIndex];
        const auto peer = std::find(group.peerRanks.begin(), group.peerRanks.end(), peerRank);
        CHK_PRT_RET(peer == group.peerRanks.end(),
            HCCL_ERROR("[ApplyP4X1MatchingOrder] matching order misses peer %u", peerRank),
            HCCL_E_INTERNAL);
        const size_t channelIndex
            = static_cast<size_t>(std::distance(group.peerRanks.begin(), peer));
        orderedPeers.push_back(peerRank);
        orderedChannels.push_back(group.channels[channelIndex]);
    }
    group.peerRanks.swap(orderedPeers);
    group.channels.swap(orderedChannels);
    return HCCL_SUCCESS;
}

const void *KernelFunction(AllGatherProfile profile, AllGatherDataPath dataPath)
{
    // 每个固定 Profile 只接受预设 dataPath。不存在 default 回退，防止错误配置
    // 静默落到另一个拓扑、数据量或算法的 Kernel。
    switch (profile) {
        case AllGatherProfile::P2X8_512K:
            return dataPath == AllGatherDataPath::FIXED_SMALL
                       ? reinterpret_cast<const void *>(ops_hccl::CcuAllGatherP2X8_512KKernel)
                       : nullptr;
        case AllGatherProfile::P2X8_512M:
            return dataPath == AllGatherDataPath::DISTRIBUTED_ROOT
                       ? reinterpret_cast<const void *>(
                             ops_hccl::CcuAllGatherP2X8_512MDistributedRootPhaseAKernel)
                   : dataPath == AllGatherDataPath::DISTRIBUTED_ROOT_CAPPED
                       ? reinterpret_cast<const void *>(
                             ops_hccl::CcuAllGatherP2X8_512MDistributedRootCappedPhaseAKernel)
                   : dataPath == AllGatherDataPath::DISTRIBUTED_ROOT_MATCHED_SERIAL
                       ? reinterpret_cast<const void *>(
                             ops_hccl::CcuAllGatherP2X8_512MDistributedRootMatchedSerialPhaseAKernel)
                   : dataPath == AllGatherDataPath::HYBRID_RELAY
                       ? reinterpret_cast<const void *>(ops_hccl::CcuAllGatherP2X8_512MHybridPhaseAKernel)
                   : dataPath == AllGatherDataPath::HYBRID_RELAY_CKE
                       ? reinterpret_cast<const void *>(ops_hccl::CcuAllGatherP2X8_512MHybridRelayKernel)
                   : dataPath == AllGatherDataPath::SINGLE_DMA
                       ? reinterpret_cast<const void *>(ops_hccl::CcuAllGatherP2X8_512MSingleDmaKernel)
                   : dataPath == AllGatherDataPath::ROLLING2
                       ? reinterpret_cast<const void *>(ops_hccl::CcuAllGatherP2X8_512MKernel)
                       : nullptr;
        case AllGatherProfile::P2X8_400M4B:
            return dataPath == AllGatherDataPath::DISTRIBUTED_ROOT
                       ? reinterpret_cast<const void *>(
                             ops_hccl::CcuAllGatherP2X8_400M4BDistributedRootPhaseAKernel)
                   : dataPath == AllGatherDataPath::DISTRIBUTED_ROOT_CAPPED
                       ? reinterpret_cast<const void *>(
                             ops_hccl::CcuAllGatherP2X8_400M4BDistributedRootCappedPhaseAKernel)
                   : dataPath == AllGatherDataPath::DISTRIBUTED_ROOT_MATCHED_SERIAL
                       ? reinterpret_cast<const void *>(
                             ops_hccl::CcuAllGatherP2X8_400M4BDistributedRootMatchedSerialPhaseAKernel)
                   : dataPath == AllGatherDataPath::HYBRID_RELAY
                       ? reinterpret_cast<const void *>(ops_hccl::CcuAllGatherP2X8_400M4BHybridPhaseAKernel)
                   : dataPath == AllGatherDataPath::HYBRID_RELAY_CKE
                       ? reinterpret_cast<const void *>(ops_hccl::CcuAllGatherP2X8_400M4BHybridRelayKernel)
                   : dataPath == AllGatherDataPath::SINGLE_DMA
                       ? reinterpret_cast<const void *>(ops_hccl::CcuAllGatherP2X8_400M4BSingleDmaKernel)
                   : dataPath == AllGatherDataPath::ROLLING2
                       ? reinterpret_cast<const void *>(ops_hccl::CcuAllGatherP2X8_400M4BKernel)
                       : nullptr;
        case AllGatherProfile::P4X1_512K:
            return dataPath == AllGatherDataPath::FIXED_SMALL
                       ? reinterpret_cast<const void *>(ops_hccl::CcuAllGatherP4X1_512KKernel)
                       : nullptr;
        case AllGatherProfile::P4X1_512M:
            return dataPath == AllGatherDataPath::MATCHED_SERIAL_ROLLING2
                       ? reinterpret_cast<const void *>(
                             ops_hccl::CcuAllGatherP4X1LargeMatchedSerialRolling2Kernel)
                   : dataPath == AllGatherDataPath::MATCHED_READY
                       ? reinterpret_cast<const void *>(
                             ops_hccl::CcuAllGatherP4X1LargeMatchedReadyKernel)
                   : dataPath == AllGatherDataPath::SINGLE_DMA
                       ? reinterpret_cast<const void *>(ops_hccl::CcuAllGatherP4X1LargeSingleDmaKernel)
                   : dataPath == AllGatherDataPath::ROLLING2
                       ? reinterpret_cast<const void *>(ops_hccl::CcuAllGatherP4X1_512MKernel)
                       : nullptr;
        case AllGatherProfile::P4X1_400M4B:
            return dataPath == AllGatherDataPath::MATCHED_SERIAL_ROLLING2
                       ? reinterpret_cast<const void *>(
                             ops_hccl::CcuAllGatherP4X1LargeMatchedSerialRolling2Kernel)
                   : dataPath == AllGatherDataPath::MATCHED_READY
                       ? reinterpret_cast<const void *>(
                             ops_hccl::CcuAllGatherP4X1LargeMatchedReadyKernel)
                   : dataPath == AllGatherDataPath::SINGLE_DMA
                       ? reinterpret_cast<const void *>(ops_hccl::CcuAllGatherP4X1LargeSingleDmaKernel)
                   : dataPath == AllGatherDataPath::ROLLING2
                       ? reinterpret_cast<const void *>(ops_hccl::CcuAllGatherP4X1_400M4BKernel)
                       : nullptr;
        case AllGatherProfile::P8P4_512K:
            return dataPath == AllGatherDataPath::FIXED_SMALL
                       ? reinterpret_cast<const void *>(ops_hccl::CcuAllGatherP8P4_512KKernel)
                       : nullptr;
        case AllGatherProfile::P8P4_512M:
            return dataPath == AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_FULL_SEED_4_7
                       ? reinterpret_cast<const void *>(
                             ops_hccl::CcuAllGatherP8P4_512MBidirectionalHalfRootFullSeed4Over7PhaseAKernel)
                   : dataPath == AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_SHARED_5_8
                       ? reinterpret_cast<const void *>(
                             ops_hccl::CcuAllGatherP8P4_512MBidirectionalHalfRootShared5Over8PhaseAKernel)
                   : dataPath == AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_SHARED
                       ? reinterpret_cast<const void *>(
                             ops_hccl::CcuAllGatherP8P4_512MBidirectionalHalfRootSharedPhaseAKernel)
                   : dataPath == AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_MATCHED
                       ? reinterpret_cast<const void *>(
                             ops_hccl::CcuAllGatherP8P4_512MBidirectionalHalfRootMatchedPhaseAKernel)
                   : dataPath == AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT
                       ? reinterpret_cast<const void *>(
                             ops_hccl::CcuAllGatherP8P4_512MBidirectionalHalfRootPhaseAKernel)
                   : dataPath == AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_CAPPED
                       ? reinterpret_cast<const void *>(
                             ops_hccl::CcuAllGatherP8P4_512MBidirectionalHalfRootCappedPhaseAKernel)
                   : dataPath == AllGatherDataPath::HYBRID_RELAY
                       ? reinterpret_cast<const void *>(ops_hccl::CcuAllGatherP8P4_512MHybridPhaseAKernel)
                   : dataPath == AllGatherDataPath::SINGLE_DMA
                       ? reinterpret_cast<const void *>(ops_hccl::CcuAllGatherP8P4_512MSingleDmaKernel)
                   : dataPath == AllGatherDataPath::ROLLING2
                       ? reinterpret_cast<const void *>(ops_hccl::CcuAllGatherP8P4_512MKernel)
                       : nullptr;
        case AllGatherProfile::P8P4_400M4B:
            return dataPath == AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_FULL_SEED_4_7
                       ? reinterpret_cast<const void *>(
                             ops_hccl::CcuAllGatherP8P4_400M4BBidirectionalHalfRootFullSeed4Over7PhaseAKernel)
                   : dataPath == AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_SHARED_5_8
                       ? reinterpret_cast<const void *>(
                             ops_hccl::CcuAllGatherP8P4_400M4BBidirectionalHalfRootShared5Over8PhaseAKernel)
                   : dataPath == AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_SHARED
                       ? reinterpret_cast<const void *>(
                             ops_hccl::CcuAllGatherP8P4_400M4BBidirectionalHalfRootSharedPhaseAKernel)
                   : dataPath == AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT
                       ? reinterpret_cast<const void *>(
                             ops_hccl::CcuAllGatherP8P4_400M4BBidirectionalHalfRootPhaseAKernel)
                   : dataPath == AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_CAPPED
                       ? reinterpret_cast<const void *>(
                             ops_hccl::CcuAllGatherP8P4_400M4BBidirectionalHalfRootCappedPhaseAKernel)
                   : dataPath == AllGatherDataPath::HYBRID_RELAY
                       ? reinterpret_cast<const void *>(ops_hccl::CcuAllGatherP8P4_400M4BHybridPhaseAKernel)
                   : dataPath == AllGatherDataPath::SINGLE_DMA
                       ? reinterpret_cast<const void *>(ops_hccl::CcuAllGatherP8P4_400M4BSingleDmaKernel)
                   : dataPath == AllGatherDataPath::ROLLING2
                       ? reinterpret_cast<const void *>(ops_hccl::CcuAllGatherP8P4_400M4BKernel)
                       : nullptr;
        case AllGatherProfile::GENERIC:
            return dataPath == AllGatherDataPath::GENERIC_DIRECT
                       ? reinterpret_cast<const void *>(ops_hccl::CcuAllGatherDirectKernel)
                       : nullptr;
        default:
            return nullptr;
    }
}

const void *HybridPhaseBKernelFunction(AllGatherProfile profile, AllGatherDataPath dataPath)
{
    if (dataPath == AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_FULL_SEED_4_7) {
        switch (profile) {
            case AllGatherProfile::P8P4_512M:
                return reinterpret_cast<const void *>(
                    ops_hccl::CcuAllGatherP8P4_512MBidirectionalHalfRootFullSeed4Over7PhaseBKernel);
            case AllGatherProfile::P8P4_400M4B:
                return reinterpret_cast<const void *>(
                    ops_hccl::CcuAllGatherP8P4_400M4BBidirectionalHalfRootFullSeed4Over7PhaseBKernel);
            default:
                return nullptr;
        }
    }
    if (dataPath == AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_SHARED_5_8) {
        switch (profile) {
            case AllGatherProfile::P8P4_512M:
                return reinterpret_cast<const void *>(
                    ops_hccl::CcuAllGatherP8P4_512MBidirectionalHalfRootShared5Over8PhaseBKernel);
            case AllGatherProfile::P8P4_400M4B:
                return reinterpret_cast<const void *>(
                    ops_hccl::CcuAllGatherP8P4_400M4BBidirectionalHalfRootShared5Over8PhaseBKernel);
            default:
                return nullptr;
        }
    }
    if (dataPath == AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_SHARED) {
        switch (profile) {
            case AllGatherProfile::P8P4_512M:
                return reinterpret_cast<const void *>(
                    ops_hccl::CcuAllGatherP8P4_512MBidirectionalHalfRootSharedPhaseBKernel);
            case AllGatherProfile::P8P4_400M4B:
                return reinterpret_cast<const void *>(
                    ops_hccl::CcuAllGatherP8P4_400M4BBidirectionalHalfRootSharedPhaseBKernel);
            default:
                return nullptr;
        }
    }
    if (dataPath == AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_MATCHED) {
        return profile == AllGatherProfile::P8P4_512M
                   ? reinterpret_cast<const void *>(
                         ops_hccl::CcuAllGatherP8P4_512MBidirectionalHalfRootMatchedPhaseBKernel)
                   : nullptr;
    }
    if (dataPath == AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT) {
        switch (profile) {
            case AllGatherProfile::P8P4_512M:
                return reinterpret_cast<const void *>(
                    ops_hccl::CcuAllGatherP8P4_512MBidirectionalHalfRootPhaseBKernel);
            case AllGatherProfile::P8P4_400M4B:
                return reinterpret_cast<const void *>(
                    ops_hccl::CcuAllGatherP8P4_400M4BBidirectionalHalfRootPhaseBKernel);
            default:
                return nullptr;
        }
    }
    if (dataPath == AllGatherDataPath::BIDIRECTIONAL_HALF_ROOT_CAPPED) {
        switch (profile) {
            case AllGatherProfile::P8P4_512M:
                return reinterpret_cast<const void *>(
                    ops_hccl::CcuAllGatherP8P4_512MBidirectionalHalfRootCappedPhaseBKernel);
            case AllGatherProfile::P8P4_400M4B:
                return reinterpret_cast<const void *>(
                    ops_hccl::CcuAllGatherP8P4_400M4BBidirectionalHalfRootCappedPhaseBKernel);
            default:
                return nullptr;
        }
    }
    if (dataPath == AllGatherDataPath::DISTRIBUTED_ROOT) {
        switch (profile) {
            case AllGatherProfile::P2X8_512M:
                return reinterpret_cast<const void *>(
                    ops_hccl::CcuAllGatherP2X8_512MDistributedRootPhaseBKernel);
            case AllGatherProfile::P2X8_400M4B:
                return reinterpret_cast<const void *>(
                    ops_hccl::CcuAllGatherP2X8_400M4BDistributedRootPhaseBKernel);
            default:
                return nullptr;
        }
    }
    if (dataPath == AllGatherDataPath::DISTRIBUTED_ROOT_CAPPED) {
        switch (profile) {
            case AllGatherProfile::P2X8_512M:
                return reinterpret_cast<const void *>(
                    ops_hccl::CcuAllGatherP2X8_512MDistributedRootCappedPhaseBKernel);
            case AllGatherProfile::P2X8_400M4B:
                return reinterpret_cast<const void *>(
                    ops_hccl::CcuAllGatherP2X8_400M4BDistributedRootCappedPhaseBKernel);
            default:
                return nullptr;
        }
    }
    if (dataPath == AllGatherDataPath::DISTRIBUTED_ROOT_MATCHED_SERIAL) {
        switch (profile) {
            case AllGatherProfile::P2X8_512M:
                return reinterpret_cast<const void *>(
                    ops_hccl::CcuAllGatherP2X8_512MDistributedRootMatchedSerialPhaseBKernel);
            case AllGatherProfile::P2X8_400M4B:
                return reinterpret_cast<const void *>(
                    ops_hccl::CcuAllGatherP2X8_400M4BDistributedRootMatchedSerialPhaseBKernel);
            default:
                return nullptr;
        }
    }
    if (dataPath == AllGatherDataPath::HYBRID_RELAY) {
        switch (profile) {
            case AllGatherProfile::P2X8_512M:
                return reinterpret_cast<const void *>(ops_hccl::CcuAllGatherP2X8_512MHybridPhaseBKernel);
            case AllGatherProfile::P2X8_400M4B:
                return reinterpret_cast<const void *>(ops_hccl::CcuAllGatherP2X8_400M4BHybridPhaseBKernel);
            case AllGatherProfile::P8P4_512M:
                return reinterpret_cast<const void *>(ops_hccl::CcuAllGatherP8P4_512MHybridPhaseBKernel);
            case AllGatherProfile::P8P4_400M4B:
                return reinterpret_cast<const void *>(ops_hccl::CcuAllGatherP8P4_400M4BHybridPhaseBKernel);
            default:
                return nullptr;
        }
    }
    return nullptr;
}

HcclResult RegisterKernels(
    HcclComm comm, const OpParam &param, const std::vector<KernelGroup> &groups, AlgResourceCtx &resource)
{
    const void *kernelFunction = KernelFunction(resource.profile, resource.dataPath);
    const void *hybridPhaseBFunction
        = HybridPhaseBKernelFunction(resource.profile, resource.dataPath);
    CHK_PRT_RET(kernelFunction == nullptr,
        HCCL_ERROR("[RegisterKernels] no exact Kernel for profile=%u dataPath=%u",
            static_cast<uint32_t>(resource.profile), static_cast<uint32_t>(resource.dataPath)),
        HCCL_E_INTERNAL);
    CcuInsHandle insHandle{};
    uint32_t insCount = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insCount));
    CHK_PRT_RET(insCount != 1U,
        HCCL_ERROR("[RegisterKernels] expected one CCU instruction instance, got %u", insCount),
        HCCL_E_INTERNAL);

    CcuResult ccuRet = HcommCcuKernelRegisterStart(insHandle);
    if (ccuRet != CCU_SUCCESS) {
        return ConvertCcuToHccl(ccuRet);
    }

    // 注册参数对象必须存活到 RegisterEnd；Prepare 与 Transfer 也必须处于同一注册批次。
    std::vector<std::shared_ptr<CcuKernelArgBase>> registrationArgs;
    if (resource.sourceMode == AllGatherSourceMode::STAGED_OUTPUT) {
        auto prepareArg = std::make_shared<CcuKernelArgPrepare>();
        const void *args[] = {prepareArg.get()};
        ccuRet = HcommCcuKernelRegister(insHandle, groups.front().actualDieId, "CcuAllGatherPrepare",
            reinterpret_cast<const void *>(ops_hccl::CcuAllGatherPrepareKernel), args, 1U,
            &resource.prepareKernel);
        if (ccuRet != CCU_SUCCESS) {
            (void)HcommCcuKernelRegisterEnd(insHandle);
            return ConvertCcuToHccl(ccuRet);
        }
        registrationArgs.push_back(prepareArg);
    }

    resource.transferKernels.resize(groups.size());
    if (hybridPhaseBFunction != nullptr) {
        resource.hybridPhaseBKernels.resize(groups.size());
    }
    resource.kernelMeta.resize(groups.size());
    for (size_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
        const KernelGroup &group = groups[groupIndex];
        auto transferArg = std::make_shared<CcuKernelArgAllGather>();
        transferArg->rankId = param.myRank;
        transferArg->rankSize = param.rankSize;
        transferArg->channelCount = static_cast<uint32_t>(group.channels.size());
        transferArg->copyLocalSlice = group.isPrimary
                                          && resource.sourceMode == AllGatherSourceMode::DIRECT_INPUT
                                      ? 1U
                                      : 0U;
        transferArg->sourceMode = resource.sourceMode;
        transferArg->profile = resource.profile;
        transferArg->dataPath = resource.dataPath;
        transferArg->layerRole = group.role;
        transferArg->fixedSliceBytes = resource.fixedSliceBytes;
        transferArg->dataTypeSize = resource.dataTypeSize;
        transferArg->cellBytes = resource.cellBytes;
        transferArg->localRankMask = resource.localRankMask;
        for (size_t channelIndex = 0; channelIndex < group.channels.size(); ++channelIndex) {
            transferArg->channels[channelIndex] = group.channels[channelIndex];
            transferArg->peerRanks[channelIndex] = group.peerRanks[channelIndex];
        }

        char kernelName[96]{};
        const int nameRet = sprintf_s(kernelName, sizeof(kernelName),
            "CcuAG_V%u_P%u_K%u_R%u_X%u_L%u_D%u_M%u_S%lu_T%lu_C%lu_H0",
            AlgResourceCtx::VERSION, static_cast<uint32_t>(resource.profile),
            static_cast<uint32_t>(resource.dataPath), param.rankSize,
            static_cast<uint32_t>(group.role), group.layer, group.actualDieId,
            static_cast<uint32_t>(resource.sourceMode), resource.fixedSliceBytes,
            resource.dataTypeSize, resource.cellBytes);
        if (nameRet <= 0) {
            (void)HcommCcuKernelRegisterEnd(insHandle);
            return HCCL_E_INTERNAL;
        }
        const void *args[] = {transferArg.get()};
        // 注册 Die 必须来自该组 Channel 的本地 Endpoint，不能使用固定 Die 编号。
        ccuRet = HcommCcuKernelRegister(insHandle, group.actualDieId, kernelName,
            kernelFunction, args, 1U,
            &resource.transferKernels[groupIndex]);
        if (ccuRet != CCU_SUCCESS) {
            (void)HcommCcuKernelRegisterEnd(insHandle);
            return ConvertCcuToHccl(ccuRet);
        }
        uint32_t peerMask = 0;
        for (uint32_t peerRank : group.peerRanks) {
            peerMask |= 1U << peerRank;
        }
        resource.kernelMeta[groupIndex] = KernelLaunchMeta{group.role, group.topoType, group.layer,
            group.actualDieId, static_cast<uint32_t>(group.channels.size()), peerMask,
            group.isPrimary ? 1U : 0U, transferArg->copyLocalSlice};
        registrationArgs.push_back(transferArg);

        if (hybridPhaseBFunction != nullptr) {
            auto phaseBArg = std::make_shared<CcuKernelArgAllGather>(*transferArg);
            phaseBArg->copyLocalSlice = 0U;
            char phaseBKernelName[96]{};
            const int phaseBNameRet = sprintf_s(phaseBKernelName, sizeof(phaseBKernelName),
                "CcuAG_V%u_P%u_K%u_R%u_X%u_L%u_D%u_M%u_S%lu_T%lu_C%lu_H1",
                AlgResourceCtx::VERSION, static_cast<uint32_t>(resource.profile),
                static_cast<uint32_t>(resource.dataPath), param.rankSize,
                static_cast<uint32_t>(group.role), group.layer, group.actualDieId,
                static_cast<uint32_t>(resource.sourceMode), resource.fixedSliceBytes,
                resource.dataTypeSize, resource.cellBytes);
            if (phaseBNameRet <= 0) {
                (void)HcommCcuKernelRegisterEnd(insHandle);
                return HCCL_E_INTERNAL;
            }
            const void *phaseBArgs[] = {phaseBArg.get()};
            ccuRet = HcommCcuKernelRegister(insHandle, group.actualDieId, phaseBKernelName,
                hybridPhaseBFunction, phaseBArgs, 1U,
                &resource.hybridPhaseBKernels[groupIndex]);
            if (ccuRet != CCU_SUCCESS) {
                (void)HcommCcuKernelRegisterEnd(insHandle);
                return ConvertCcuToHccl(ccuRet);
            }
            registrationArgs.push_back(phaseBArg);
        }
    }

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    if (ccuRet != CCU_SUCCESS) {
        return ConvertCcuToHccl(ccuRet);
    }
    return HCCL_SUCCESS;
}

HcclResult CreateResources(HcclComm comm, const OpParam &param, const TopologyView &topology,
    uint64_t dataTypeSize, uint64_t sliceBytes, AllGatherProfile profile,
    AllGatherSourceMode sourceMode, AlgResourceCtx &resource)
{
    AllGatherProfileSpec spec;
    CHK_PRT_RET(!GetAllGatherProfileSpec(profile, spec),
        HCCL_ERROR("[CreateResources] invalid Profile %u", static_cast<uint32_t>(profile)), HCCL_E_INTERNAL);
    ApplyAllGatherTopologyPolicy(topology.topology, param.rankSize, spec);
    CHK_PRT_RET(profile != AllGatherProfile::GENERIC
                    && (spec.topology != topology.topology || spec.expectedRankSize != param.rankSize),
        HCCL_ERROR("[CreateResources] Profile topology/rank mismatch"), HCCL_E_INTERNAL);
    resource.dataPath = GetConfiguredAllGatherDataPath(profile);
    resource.rankSize = param.rankSize;
    resource.topology = topology.topology;
    resource.profile = profile;
    resource.algorithm = spec.algorithm;
    resource.layerPolicy = spec.layerPolicy;
    resource.sourceMode = sourceMode;
    resource.fixedSliceBytes = profile == AllGatherProfile::GENERIC ? 0U : sliceBytes;
    resource.dataTypeSize = dataTypeSize;
    resource.cellBytes = profile == AllGatherProfile::GENERIC
                             ? 0U
                             : (spec.algorithm == AllGatherAlgorithm::FIXED_DIRECT_SMALL
                                       ? sliceBytes
                                       : spec.cellBytes);
    resource.localRankCount = static_cast<uint32_t>(topology.localRanks.size());
    for (uint32_t rank : topology.localRanks) {
        resource.localRankMask |= 1U << rank;
    }
    resource.meshLayer = topology.meshLayer;
    resource.meshTopoType = topology.meshTopoType;
    resource.closLayer = topology.closLayer;

    if (sourceMode == AllGatherSourceMode::STAGED_OUTPUT) {
        void *buffer = nullptr;
        uint64_t bufferSize = 0;
        CHK_RET(HcclGetHcclBuffer(comm, &buffer, &bufferSize));
        CHK_PRT_RET(buffer == nullptr || bufferSize == 0U,
            HCCL_ERROR("[CreateResources] HCCL buffer is empty"), HCCL_E_MEMORY);
        resource.localBuffer = CommBuffer{buffer, bufferSize};
    }

    std::vector<KernelGroup> groups;
    CHK_RET(AcquireChannels(comm, param, topology, spec, groups));
    if (resource.dataPath == AllGatherDataPath::MATCHED_READY
        || resource.dataPath == AllGatherDataPath::MATCHED_SERIAL_ROLLING2) {
        CHK_RET(ApplyP4X1MatchingOrder(param, groups));
    }
    uint32_t dieMask = 0;
    for (const KernelGroup &group : groups) {
        dieMask |= 1U << group.actualDieId;
    }
    const uint32_t dieCount = static_cast<uint32_t>((dieMask & 1U) != 0U)
                              + static_cast<uint32_t>((dieMask & 2U) != 0U);
    CHK_PRT_RET(dieCount == 0U || dieCount > MAX_DIE_NUM,
        HCCL_ERROR("[CreateResources] invalid active Die count %u", dieCount), HCCL_E_INTERNAL);
    const bool requiresDualDie
        = spec.layerPolicy == AllGatherLayerPolicy::LOCAL_MESH_REMOTE_CLOS;
    CHK_PRT_RET(requiresDualDie
                    && (dieCount != MAX_DIE_NUM || dieMask != ((1U << MAX_DIE_NUM) - 1U)
                        || groups.size() != MAX_DIE_NUM),
        HCCL_ERROR("[CreateResources] profile %s requires one Mesh group and one Clos group "
                   "on distinct Dies, dieMask=0x%x groups=%zu",
            ProfileName(profile), dieMask, groups.size()),
        HCCL_E_NOT_SUPPORT);
    if (dieCount > 1U) {
        resource.extraThreads.resize(dieCount - 1U);
        CHK_RET(HcclThreadAcquire(comm, CommEngine::COMM_ENGINE_CCU,
            static_cast<uint32_t>(resource.extraThreads.size()), THREAD_NOTIFY_NUM,
            resource.extraThreads.data()));
    }
    CHK_PRT_RET(requiresDualDie
                    && (resource.extraThreads.size() != 1U || resource.extraThreads[0] == 0U
                        || resource.extraThreads[0] == param.cpuThread),
        HCCL_ERROR("[CreateResources] dual-Die Profile did not acquire an independent CCU Thread"),
        HCCL_E_INTERNAL);
    CHK_RET(RegisterKernels(comm, param, groups, resource));
    return HCCL_SUCCESS;
}

bool CheckedRange(uintptr_t begin, uint64_t bytes, uintptr_t &end)
{
    if (bytes > static_cast<uint64_t>(std::numeric_limits<uintptr_t>::max() - begin)) {
        return false;
    }
    end = begin + static_cast<uintptr_t>(bytes);
    return true;
}

HcclResult ClassifySourceMode(const void *input, const void *output, uint64_t sliceBytes,
    uint64_t outputBytes, uint64_t localOffset, AllGatherSourceMode &mode)
{
    const uintptr_t inputBegin = reinterpret_cast<uintptr_t>(input);
    const uintptr_t outputBegin = reinterpret_cast<uintptr_t>(output);
    uintptr_t inputEnd = 0;
    uintptr_t outputEnd = 0;
    CHK_PRT_RET(!CheckedRange(inputBegin, sliceBytes, inputEnd)
                    || !CheckedRange(outputBegin, outputBytes, outputEnd)
                    || localOffset > std::numeric_limits<uintptr_t>::max() - outputBegin,
        HCCL_ERROR("[ClassifySourceMode] address range overflows uintptr_t"), HCCL_E_PARA);
    const uintptr_t canonical = outputBegin + static_cast<uintptr_t>(localOffset);
    // 精确命中本 Rank 最终 Offset 是标准 in-place；其余交叠形式必须先 staging。
    if (inputBegin == canonical) {
        mode = AllGatherSourceMode::CANONICAL_OUTPUT;
    } else if (inputBegin < outputEnd && outputBegin < inputEnd) {
        mode = AllGatherSourceMode::STAGED_OUTPUT;
    } else {
        mode = AllGatherSourceMode::DIRECT_INPUT;
    }
    return HCCL_SUCCESS;
}

} // namespace

HcclResult HcclAllGather(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    uint64_t dataTypeSize = 0;
    CHK_PRT_RET(!GetDataTypeSize(dataType, dataTypeSize),
        HCCL_ERROR("[HcclAllGather] unsupported data type %d", dataType), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(sendCount > std::numeric_limits<uint64_t>::max() / dataTypeSize,
        HCCL_ERROR("[HcclAllGather] Slice byte count overflows"), HCCL_E_PARA);
    const uint64_t sliceBytes = sendCount * dataTypeSize;
    if (sliceBytes == 0U) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);

    OpParam param;
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = sendCount;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0U || param.rankSize > MAX_RANK_SIZE || param.myRank >= param.rankSize,
        HCCL_ERROR("[HcclAllGather] invalid rank %u/%u", param.myRank, param.rankSize), HCCL_E_PARA);
    CHK_PRT_RET(sliceBytes > std::numeric_limits<uint64_t>::max() / param.rankSize,
        HCCL_ERROR("[HcclAllGather] Output byte count overflows"), HCCL_E_PARA);
    const uint64_t outputBytes = sliceBytes * param.rankSize;
    const uint64_t localOffset = sliceBytes * param.myRank;

    FinalTopology topologyFromRank = FinalTopology::TOPOLOGY_4X1;
    CHK_RET(GetTopologyByRankSize(param.rankSize, topologyFromRank));
    const AllGatherProfile profile = SelectAllGatherProfile(sendCount, param.rankSize, dataTypeSize);
    if (profile == AllGatherProfile::GENERIC) {
        HCCL_ERROR("[HcclAllGather] Unsupported input: rankSize=%u actualOutputBytes=%lu; "
                   "only 512KiB/512MiB/400MiB+4B on rankSize 4/12/16 are allowed",
            param.rankSize, outputBytes);
        return HCCL_E_INTERNAL;
    }
    AllGatherProfileSpec profileSpec;
    CHK_PRT_RET(!GetAllGatherProfileSpec(profile, profileSpec),
        HCCL_ERROR("[HcclAllGather] invalid Profile selection"), HCCL_E_INTERNAL);
    ApplyAllGatherTopologyPolicy(topologyFromRank, param.rankSize, profileSpec);
    CHK_PRT_RET(profile != AllGatherProfile::GENERIC
                    && (profileSpec.topology != topologyFromRank
                        || profileSpec.expectedRankSize != param.rankSize),
        HCCL_ERROR("[HcclAllGather] Profile does not match rank topology"), HCCL_E_INTERNAL);
    const AllGatherDataPath dataPath = GetConfiguredAllGatherDataPath(profile);
    CHK_PRT_RET(KernelFunction(profile, dataPath) == nullptr,
        HCCL_ERROR("[HcclAllGather] profile=%s has no preset Kernel for dataPath=%s",
            ProfileName(profile), DataPathName(dataPath)),
        HCCL_E_INTERNAL);

    AllGatherSourceMode sourceMode = AllGatherSourceMode::DIRECT_INPUT;
    CHK_RET(ClassifySourceMode(sendBuf, recvBuf, sliceBytes, outputBytes, localOffset, sourceMode));
    const uint64_t fixedSliceBytes = profile == AllGatherProfile::GENERIC ? 0U : sliceBytes;
    const int tagRet = sprintf_s(param.tag, sizeof(param.tag),
        "hccl_custom_ag_dd_v%u_t%u_p%u_a%u_k%u_m%u_s%lu_d%lu_c%lu", AlgResourceCtx::VERSION,
        static_cast<uint32_t>(topologyFromRank), static_cast<uint32_t>(profile),
        static_cast<uint32_t>(profileSpec.algorithm), static_cast<uint32_t>(dataPath),
        static_cast<uint32_t>(sourceMode),
        fixedSliceBytes, dataTypeSize,
        profile == AllGatherProfile::GENERIC
            ? 0U
            : (profileSpec.algorithm == AllGatherAlgorithm::FIXED_DIRECT_SMALL
                      ? sliceBytes
                      : profileSpec.cellBytes));
    CHK_PRT_RET(tagRet <= 0,
        HCCL_ERROR("[HcclAllGather] failed to build Engine Context tag"), HCCL_E_INTERNAL);

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    constexpr CommEngine engine = CommEngine::COMM_ENGINE_CCU;
    const uint32_t mainThreadNotifyNum = topologyFromRank == FinalTopology::TOPOLOGY_4X1
                                             ? SINGLE_DIE_THREAD_NOTIFY_NUM
                                             : THREAD_NOTIFY_NUM;
    CHK_RET(HcclThreadAcquireWithStream(
        comm, engine, stream, mainThreadNotifyNum, &param.cpuThread));

    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    const HcclResult getCtxRet = HcclEngineCtxGet(comm, param.tag, engine, &ctx, &ctxSize);
    if (getCtxRet == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = ctxSize;
    } else {
        CHK_PRT_RET(getCtxRet != HCCL_E_NOT_FOUND,
            HCCL_ERROR("[HcclAllGather] Engine Context lookup failed, ret %d", getCtxRet), getCtxRet);
        TopologyView topology;
        CHK_RET(DetectFinalTopology(comm, param, topology));
        CHK_PRT_RET(topology.topology != topologyFromRank,
            HCCL_ERROR("[HcclAllGather] rank dispatch and detected topology disagree"), HCCL_E_INTERNAL);

        AlgResourceCtx resource;
        // 固定 Profile 的资源规划失败时不能由单个 Rank 私自切换 Generic，否则各 Rank
        // 会建立不同的 Channel 图。缺少全 Rank 一致决策机制时必须 fail closed。
        CHK_RET(CreateResources(
            comm, param, topology, dataTypeSize, sliceBytes, profile, sourceMode, resource));
        return ops_hccl::ExecPreparedOp(param, resource);
    }

    return ops_hccl::ExecOp(param);
}
