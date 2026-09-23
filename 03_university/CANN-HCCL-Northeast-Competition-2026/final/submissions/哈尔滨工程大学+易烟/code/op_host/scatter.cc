/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>
#if __has_include(<hccl/hccl_ccu_res.h>)
#include <hccl/hccl_ccu_res.h>
#endif
#include <ccu/ccu_res.h>
#if __has_include(<ccu/ccu_launch.h>)
#include <ccu/ccu_launch.h>
#endif

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <new>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "exec_op.h"
#include "ccu_kernel.h"

namespace {
constexpr CommEngine DIRECT_ENGINE = CommEngine::COMM_ENGINE_CCU;

bool CopyMemberMask(const uint32_t *ranks, uint32_t count, uint32_t rankSize, uint32_t myRank, uint32_t &mask)
{
    mask = 0;
    if (ranks == nullptr || count == 0 || count > rankSize) {
        return false;
    }
    for (uint32_t i = 0; i < count; ++i) {
        if (ranks[i] >= rankSize || (mask & ops_hccl::RankBit(ranks[i])) != 0) {
            mask = 0;
            return false;
        }
        mask |= ops_hccl::RankBit(ranks[i]);
    }
    if ((mask & ops_hccl::RankBit(myRank)) == 0) {
        mask = 0;
        return false;
    }
    return true;
}

void ReadLayerFacts(HcclComm comm, const OpParam &param, ops_hccl::TopologyLayerFacts &facts)
{
    using namespace ops_hccl;
    facts.typeResult = HcclRankGraphGetTopoTypeByLayer(comm, facts.layer, &facts.type);
    if (facts.typeResult == HCCL_SUCCESS) {
        facts.valid |= TOPO_TYPE_VALID;
    }

    uint32_t *ranks = nullptr;
    uint32_t rankCount = 0;
    facts.membersResult = HcclRankGraphGetRanksByLayer(comm, facts.layer, &ranks, &rankCount);
    if (facts.membersResult == HCCL_SUCCESS) {
        if (CopyMemberMask(ranks, rankCount, param.rankSize, param.myRank, facts.memberMask)) {
            facts.valid |= TOPO_MEMBERS_VALID;
        } else {
            facts.issue = TopologyReason::INVALID_METADATA;
        }
    }

    // Copy each provider-owned result before making another topology call.
    uint32_t *sizes = nullptr;
    uint32_t sizeCount = 0;
    facts.sizesResult = HcclRankGraphGetInstSizeListByLayer(comm, facts.layer, &sizes, &sizeCount);
    if (facts.sizesResult == HCCL_SUCCESS) {
        uint64_t sum = 0;
        bool valid = sizes != nullptr && sizeCount != 0 && sizeCount <= MAX_RANK_SIZE;
        if (valid) {
            for (uint32_t i = 0; i < sizeCount; ++i) {
                if (sizes[i] == 0 || sizes[i] > param.rankSize) {
                    valid = false;
                    break;
                }
                sum += sizes[i];
            }
        }
        if (valid && sum == param.rankSize) {
            facts.instanceCount = sizeCount;
            std::copy(sizes, sizes + sizeCount, facts.instanceSizes);
            facts.valid |= TOPO_SIZES_VALID;
        } else {
            facts.issue = TopologyReason::INVALID_METADATA;
        }
    }
    if (facts.issue == TopologyReason::NOT_QUERIED &&
        (facts.typeResult != HCCL_SUCCESS || facts.membersResult != HCCL_SUCCESS ||
         facts.sizesResult != HCCL_SUCCESS)) {
        facts.issue = TopologyReason::QUERY_FAILED;
    }
    if ((facts.valid & (TOPO_MEMBERS_VALID | TOPO_SIZES_VALID)) == (TOPO_MEMBERS_VALID | TOPO_SIZES_VALID)) {
        bool matched = false;
        const uint32_t localSize = RankMaskSize(facts.memberMask);
        for (uint32_t i = 0; i < facts.instanceCount; ++i) {
            matched = matched || facts.instanceSizes[i] == localSize;
        }
        if (!matched) {
            facts.issue = TopologyReason::INVALID_METADATA;
        }
    }
}

bool HasDirectMeshPaths(HcclComm comm, const OpParam &param, uint32_t layer, uint32_t members)
{
    // Only inspect edges incident to this rank. Official 9.1 may hide paths
    // between two ranks outside our inner instance even on a connected system.
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer == param.myRank || (members & ops_hccl::RankBit(peer)) == 0) {
            continue;
        }
        CommLink *links = nullptr;
        uint32_t count = 0;
        if (HcclRankGraphGetLinks(comm, layer, param.myRank, peer, &links, &count) != HCCL_SUCCESS ||
            links == nullptr || count == 0) {
            return false;
        }
        bool direct = false;
        for (uint32_t i = 0; i < count; ++i) {
            const auto protocol = links[i].linkAttr.linkProtocol;
            if (links[i].linkAttr.hop == 1 &&
                (protocol == COMM_PROTOCOL_UBC_CTP || protocol == COMM_PROTOCOL_UBC_TP) &&
                links[i].srcEndpointDesc.protocol == protocol && links[i].dstEndpointDesc.protocol == protocol) {
                direct = true;
                break;
            }
        }
        if (!direct) {
            return false;
        }
    }
    return true;
}

void MergeMeshCandidate(const OpParam &param, ops_hccl::TopologyFacts &topology,
    ops_hccl::TopologyLayerFacts &layer, uint32_t members, ops_hccl::TopologyEvidence evidence,
    const uint32_t *sortedSizes, uint32_t sizeCount, bool &ambiguous)
{
    using namespace ops_hccl;
    const TopologyKind kind = ShapeFromLocalSize(param.rankSize, RankMaskSize(members));
    if (kind == TopologyKind::UNKNOWN || (members & RankBit(param.myRank)) == 0) {
        return;
    }
    if ((topology.kind != TopologyKind::UNKNOWN &&
         (topology.kind != kind || topology.localMemberMask != members)) ||
        (layer.meshMemberMask != 0 && layer.meshMemberMask != members)) {
        ambiguous = true;
        layer.issue = TopologyReason::AMBIGUOUS;
        return;
    }
    layer.meshMemberMask = members;
    if (topology.kind == TopologyKind::UNKNOWN ||
        (evidence == TopologyEvidence::LAYER_INSTANCES && topology.evidence != evidence)) {
        topology.kind = kind;
        topology.evidence = evidence;
        topology.reason = TopologyReason::RECOGNIZED;
        topology.meshLayer = layer.layer;
        topology.localMemberMask = members;
        topology.serverCount = sizeCount;
        if (sizeCount != 0) {
            std::copy(sortedSizes, sortedSizes + sizeCount, topology.serverSizes);
        }
    }
}

void ReadCustomInstances(HcclComm comm, const OpParam &param, ops_hccl::TopologyFacts &topology,
    ops_hccl::TopologyLayerFacts &layer, bool &ambiguous)
{
    using namespace ops_hccl;
    uint32_t *ids = nullptr;
    uint32_t count = 0;
    layer.detailResult = HcclRankGraphGetTopoInstsByLayer(comm, layer.layer, &ids, &count);
    if (layer.detailResult != HCCL_SUCCESS) {
        layer.issue = TopologyReason::QUERY_FAILED;
        return;
    }
    if (ids == nullptr || count == 0 || count > MAX_TOPOLOGY_SUBINSTANCES) {
        layer.issue = count > MAX_TOPOLOGY_SUBINSTANCES ? TopologyReason::CAPACITY_LIMIT :
            TopologyReason::INVALID_METADATA;
        return;
    }
    uint32_t copiedIds[MAX_TOPOLOGY_SUBINSTANCES]{};
    std::copy(ids, ids + count, copiedIds);
    std::sort(copiedIds, copiedIds + count);
    count = static_cast<uint32_t>(std::unique(copiedIds, copiedIds + count) - copiedIds);
    const uint32_t all = AllRankMask(param.rankSize);
    uint32_t meshCandidate = 0;
    bool complete = true;
    uint32_t sortedSizes[MAX_RANK_SIZE]{};
    std::copy(layer.instanceSizes, layer.instanceSizes + layer.instanceCount, sortedSizes);
    std::sort(sortedSizes, sortedSizes + layer.instanceCount);
    const TopologyKind parentKind = ShapeFromSizes(param.rankSize, sortedSizes, layer.instanceCount);
    // These IDs describe this rank's accessible instances, not all Servers.
    for (uint32_t i = 0; i < count; ++i) {
        CommTopo type = COMM_TOPO_RESERVED;
        HcclResult ret = HcclRankGraphGetTopoType(comm, layer.layer, copiedIds[i], &type);
        if (ret != HCCL_SUCCESS) {
            complete = false;
            layer.detailResult = ret;
            layer.issue = TopologyReason::QUERY_FAILED;
            continue;
        }
        if (type != COMM_TOPO_1DMESH && type != COMM_TOPO_CLOS) {
            continue;
        }
        uint32_t *ranks = nullptr;
        uint32_t rankCount = 0;
        ret = HcclRankGraphGetRanksByTopoInst(comm, layer.layer, copiedIds[i], &ranks, &rankCount);
        uint32_t members = 0;
        if (ret != HCCL_SUCCESS || !CopyMemberMask(ranks, rankCount, param.rankSize, param.myRank, members)) {
            complete = false;
            layer.detailResult = ret;
            layer.issue = ret == HCCL_SUCCESS ? TopologyReason::INVALID_METADATA : TopologyReason::QUERY_FAILED;
            continue;
        }
        if ((layer.valid & TOPO_MEMBERS_VALID) != 0 && (members & ~layer.memberMask) != 0) {
            complete = false;
            layer.issue = TopologyReason::INVALID_METADATA;
            continue;
        }
        if (type == COMM_TOPO_CLOS) {
            if (members == all) {
                layer.closMemberMask = all;
            }
            continue;
        }
        const TopologyKind localKind = ShapeFromLocalSize(param.rankSize, rankCount);
        if (localKind == TopologyKind::UNKNOWN) {
            continue;
        }
        if (!HasDirectMeshPaths(comm, param, layer.layer, members)) {
            complete = false;
            layer.issue = TopologyReason::NO_SERVER_MESH;
            continue;
        }
        if ((parentKind != TopologyKind::UNKNOWN && (parentKind != localKind ||
             ((layer.valid & TOPO_MEMBERS_VALID) != 0 && members != layer.memberMask))) ||
            (meshCandidate != 0 && meshCandidate != members)) {
            ambiguous = true;
            layer.issue = TopologyReason::AMBIGUOUS;
            continue;
        }
        // Do not merge overlapping small Mesh instances to invent a full mesh.
        meshCandidate = members;
    }
    if (complete && meshCandidate != 0) {
        MergeMeshCandidate(param, topology, layer, meshCandidate, TopologyEvidence::LOCAL_MESH_INSTANCE,
            nullptr, 0, ambiguous);
    }
}

void DiscoverTopology(HcclComm comm, const OpParam &param, const std::vector<uint32_t> &layers,
    ops_hccl::TopologyFacts &topology)
{
    using namespace ops_hccl;
    topology.Reset();
    if (param.rankSize != 4 && param.rankSize != 12 && param.rankSize != 16) {
        topology.reason = TopologyReason::UNSUPPORTED_RANK_SIZE;
        return;
    }
    if (layers.size() > MAX_TOPOLOGY_LAYERS) {
        topology.reason = TopologyReason::CAPACITY_LIMIT;
        return;
    }
    bool ambiguous = false;
    topology.layerCount = static_cast<uint32_t>(layers.size());
    topology.reason = TopologyReason::NO_SERVER_MESH;
    for (uint32_t i = 0; i < topology.layerCount; ++i) {
        auto &layer = topology.layers[i];
        layer.layer = layers[i];
        ReadLayerFacts(comm, param, layer);
        if ((layer.valid & TOPO_TYPE_VALID) == 0 || layer.issue == TopologyReason::INVALID_METADATA) {
            continue;
        }
        const uint32_t all = AllRankMask(param.rankSize);
        if (layer.type == COMM_TOPO_CLOS) {
            if ((layer.valid & TOPO_MEMBERS_VALID) != 0 && layer.memberMask == all) {
                layer.closMemberMask = all;
            }
        } else if (layer.type == COMM_TOPO_1DMESH || layer.type == COMM_TOPO_CUSTOM) {
            const uint32_t localSize = RankMaskSize(layer.memberMask);
            const TopologyKind localKind = ShapeFromLocalSize(param.rankSize, localSize);
            uint32_t sortedSizes[MAX_RANK_SIZE]{};
            std::copy(layer.instanceSizes, layer.instanceSizes + layer.instanceCount, sortedSizes);
            std::sort(sortedSizes, sortedSizes + layer.instanceCount);
            const TopologyKind globalKind = ShapeFromSizes(param.rankSize, sortedSizes, layer.instanceCount);
            const bool hasSizes = (layer.valid & TOPO_SIZES_VALID) != 0;
            const bool hasMembers = (layer.valid & TOPO_MEMBERS_VALID) != 0;
            const bool matchingSizes = globalKind != TopologyKind::UNKNOWN && globalKind == localKind;
            if (hasMembers && localKind != TopologyKind::UNKNOWN &&
                ((!hasSizes && layer.type == COMM_TOPO_1DMESH) || matchingSizes) &&
                (layer.type == COMM_TOPO_1DMESH || HasDirectMeshPaths(comm, param, layer.layer, layer.memberMask))) {
                MergeMeshCandidate(param, topology, layer, layer.memberMask,
                    matchingSizes ? TopologyEvidence::LAYER_INSTANCES : TopologyEvidence::LOCAL_MESH_INSTANCE,
                    sortedSizes, matchingSizes ? layer.instanceCount : 0, ambiguous);
            }
            if (layer.type == COMM_TOPO_CUSTOM && layer.meshMemberMask == 0) {
                // A CUSTOM netInstance may cover all N ranks while its Mesh
                // subinstance only contains this rank's physical local peers.
                ReadCustomInstances(comm, param, topology, layer, ambiguous);
            }
        }
        if (layer.closMemberMask == all && topology.closLayer == INVALID_VALUE_RANKID) {
            topology.closLayer = layer.layer;
        }
    }
    if (ambiguous) {
        topology.kind = TopologyKind::UNKNOWN;
        topology.evidence = TopologyEvidence::NONE;
        topology.localMemberMask = 0;
        topology.meshLayer = INVALID_VALUE_RANKID;
        topology.serverCount = 0;
        std::fill(topology.serverSizes, topology.serverSizes + MAX_RANK_SIZE, 0);
        topology.reason = TopologyReason::AMBIGUOUS;
    } else if (topology.kind == TopologyKind::UNKNOWN && param.rankSize == 4 &&
        topology.closLayer != INVALID_VALUE_RANKID) {
        // This inference is confined to the contest's four topology cases.
        // A missing multi-rank Mesh layer must not reject the 4x1 Direct case.
        topology.kind = TopologyKind::FOUR_BY_ONE;
        topology.evidence = TopologyEvidence::SINGLETON_CASE;
        topology.localMemberMask = RankBit(param.myRank);
        topology.reason = TopologyReason::RECOGNIZED;
    }
    if (topology.kind == TopologyKind::UNKNOWN && topology.reason != TopologyReason::AMBIGUOUS) {
        for (uint32_t i = 0; i < topology.layerCount; ++i) {
            if (topology.layers[i].issue != TopologyReason::NOT_QUERIED) {
                topology.reason = topology.layers[i].issue;
                break;
            }
        }
    }
}

void ClassifySelectedPaths(AlgResourceCtx &ctx)
{
    using namespace ops_hccl;
    if (ctx.topology.kind == TopologyKind::UNKNOWN) {
        return;
    }
    for (uint32_t peer = 0; peer < ctx.rankSize; ++peer) {
        auto &link = ctx.peerLinks[peer];
        if (link.valid == 0) {
            continue;
        }
        const bool local = (ctx.topology.localMemberMask & RankBit(peer)) != 0;
        for (uint32_t i = 0; i < ctx.topology.layerCount; ++i) {
            const auto &layer = ctx.topology.layers[i];
            if (layer.layer != link.layer) {
                continue;
            }
            if (local && layer.meshMemberMask == ctx.topology.localMemberMask && link.hop == 1) {
                link.pathClass = PathClass::LOCAL_MESH;
            } else if (!local && (layer.closMemberMask & RankBit(peer)) != 0) {
                link.pathClass = PathClass::REMOTE_CLOS;
            }
            // A selected path that does not match stays UNKNOWN. Never switch
            // endpoints merely to make the mathematical model fit.
            break;
        }
    }
}

int CompareAddress(const CommAddr &left, const CommAddr &right)
{
    if (left.type != right.type) {
        return left.type < right.type ? -1 : 1;
    }
    switch (left.type) {
        case CommAddrType::COMM_ADDR_TYPE_IP_V4:
            return std::memcmp(&left.addr, &right.addr, sizeof(left.addr));
        case CommAddrType::COMM_ADDR_TYPE_IP_V6:
            return std::memcmp(&left.addr6, &right.addr6, sizeof(left.addr6));
        case CommAddrType::COMM_ADDR_TYPE_ID:
            return left.id == right.id ? 0 : (left.id < right.id ? -1 : 1);
        case CommAddrType::COMM_ADDR_TYPE_EID:
            return std::memcmp(left.eid, right.eid, sizeof(left.eid));
        default:
            return 0;
    }
}

bool LinkComesFirst(const CommLink &left, const CommLink &right)
{
    if (left.linkAttr.linkProtocol != right.linkAttr.linkProtocol) {
        return left.linkAttr.linkProtocol < right.linkAttr.linkProtocol;
    }
    const int sourceOrder = CompareAddress(left.srcEndpointDesc.commAddr, right.srcEndpointDesc.commAddr);
    if (sourceOrder != 0) {
        return sourceOrder < 0;
    }
    return CompareAddress(left.dstEndpointDesc.commAddr, right.dstEndpointDesc.commAddr) < 0;
}

HcclResult SelectChannel(HcclComm comm, const OpParam &param, uint32_t peer, const std::vector<uint32_t> &layers,
    HcclChannelDesc &desc, uint32_t &dieId, ops_hccl::PeerLinkInfo &info)
{
    // GetEndpointInfo in 9.1 resolves this communicator's local interface map.
    // Do not reuse this function to query another rank's endpoint attributes.
    const uint32_t myRank = param.myRank;
    // Both ranks use the same direction and hop/layer/protocol/address order.
    // Local topology observations never participate in channel selection.
    const uint32_t sourceRank = std::min(myRank, peer);
    const uint32_t targetRank = std::max(myRank, peer);
    CommLink selected{};
    uint32_t selectedLayer = INVALID_VALUE_RANKID;
    bool found = false;
    for (uint32_t layer : layers) {
        CommLink *links = nullptr;
        uint32_t linkCount = 0;
        CHK_RET(HcclRankGraphGetLinks(comm, layer, sourceRank, targetRank, &links, &linkCount));
        if (linkCount == 0) {
            continue;
        }
        CHK_PTR_NULL(links);
        for (uint32_t i = 0; i < linkCount; ++i) {
            const auto &candidate = links[i];
            const CommProtocol protocol = candidate.linkAttr.linkProtocol;
            // CANN 9.1 uses UBC_CTP; UB_CTP is the name in newer headers.
            if (protocol != CommProtocol::COMM_PROTOCOL_UBC_CTP &&
                protocol != CommProtocol::COMM_PROTOCOL_UBC_TP) {
                continue;
            }
            if (candidate.srcEndpointDesc.protocol != protocol || candidate.dstEndpointDesc.protocol != protocol) {
                continue;
            }
            const bool better = !found || candidate.linkAttr.hop < selected.linkAttr.hop ||
                (candidate.linkAttr.hop == selected.linkAttr.hop &&
                 (layer < selectedLayer || (layer == selectedLayer && LinkComesFirst(candidate, selected))));
            if (better) {
                // Preserve provider-owned endpoint data before querying another layer.
                selected = candidate;
                selectedLayer = layer;
                found = true;
            }
        }
    }
    if (!found) {
        HCCL_ERROR("No supported CCU channel between ranks %u and %u", myRank, peer);
        return HCCL_E_NOT_SUPPORT;
    }
    desc.remoteRank = peer;
    desc.channelProtocol = selected.linkAttr.linkProtocol;
    desc.localEndpoint = myRank == sourceRank ? selected.srcEndpointDesc : selected.dstEndpointDesc;
    desc.remoteEndpoint = myRank == sourceRank ? selected.dstEndpointDesc : selected.srcEndpointDesc;
    desc.notifyNum = ops_hccl::CHANNEL_NOTIFY_COUNT;
    EndpointAttrDieId endpointDie{};
    const HcclResult ret = HcclRankGraphGetEndpointInfo(comm, myRank, &desc.localEndpoint,
        EndpointAttr::ENDPOINT_ATTR_DIE_ID, sizeof(endpointDie), &endpointDie);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("Cannot resolve CCU endpoint die: rank=%u peer=%u layer=%u ret=%d",
            myRank, peer, selectedLayer, static_cast<int>(ret)), ret);
    dieId = endpointDie;
    info.valid = 1;
    info.peer = peer;
    info.layer = selectedLayer;
    info.dieId = dieId;
    info.hop = selected.linkAttr.hop;
    info.localEndpoint = desc.localEndpoint;
    info.remoteEndpoint = desc.remoteEndpoint;
    HCCL_DEBUG("[Scatter v20 channel] rank=%u peer=%u layer=%u hop=%u die=%u protocol=%u",
        myRank, peer, selectedLayer, info.hop, dieId, static_cast<uint32_t>(desc.channelProtocol));
    return HCCL_SUCCESS;
}

HcclResult AcquireChannels(HcclComm comm, const OpParam &param, AlgResourceCtx &ctx)
{
    if (param.rankSize == 1) {
        ctx.topology.reason = ops_hccl::TopologyReason::UNSUPPORTED_RANK_SIZE;
        return HCCL_SUCCESS;
    }
    uint32_t *layerList = nullptr;
    uint32_t layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layerList, &layerCount));
    CHK_PRT_RET(layerCount == 0 || layerList == nullptr,
        HCCL_ERROR("No communication topology layers"), HCCL_E_NOT_SUPPORT);
    std::vector<uint32_t> layers(layerList, layerList + layerCount);
    std::sort(layers.begin(), layers.end());
    layers.erase(std::unique(layers.begin(), layers.end()), layers.end());

    // Optional observations select supplier layouts within the same large-message
    // descriptor protocol. UNKNOWN never changes which handshake a rank executes.
    try {
        DiscoverTopology(comm, param, layers, ctx.topology);
    } catch (const std::exception &) {
        ctx.topology.Reset(ops_hccl::TopologyReason::QUERY_EXCEPTION);
    }
    if (!ctx.topology.Valid(param.rankSize, param.myRank)) {
        ctx.topology.Reset(ops_hccl::TopologyReason::INVALID_METADATA);
    }

    const uint32_t peerCount = param.rankSize - 1;
    std::vector<HcclChannelDesc> descs(peerCount);
    std::vector<ChannelHandle> handles(peerCount);
    std::vector<uint32_t> dieIds(peerCount);
    CHK_RET(HcclChannelDescInit(descs.data(), peerCount));
    uint32_t index = 0;
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer != param.myRank) {
            CHK_RET(SelectChannel(comm, param, peer, layers, descs[index], dieIds[index], ctx.peerLinks[peer]));
            CHK_PRT_RET(dieIds[index] >= ops_hccl::DIRECT_MAX_DIE_GROUPS,
                HCCL_ERROR("Unsupported local CCU die: peer=%u die=%u", peer, dieIds[index]), HCCL_E_NOT_SUPPORT);
            ++index;
        }
    }
    // Cache exactly one channel per peer, independently of the current root.
    CHK_RET(HcclChannelAcquire(comm, DIRECT_ENGINE, descs.data(), peerCount, handles.data()));
    for (uint32_t i = 0; i < peerCount; ++i) {
        CHK_PRT_RET(handles[i] == 0, HCCL_ERROR("Empty CCU channel"), HCCL_E_INTERNAL);
        uint32_t group = 0;
        while (group < ctx.groupCount && ctx.groups[group].dieId != dieIds[i]) {
            ++group;
        }
        if (group == ctx.groupCount) {
            CHK_PRT_RET(ctx.groupCount >= ops_hccl::DIRECT_MAX_DIE_GROUPS,
                HCCL_ERROR("More local channel dies than Ascend 950 supports"), HCCL_E_NOT_SUPPORT);
            ++ctx.groupCount;
            ctx.groups[group].dieId = dieIds[i];
            ctx.groups[group].kernelArg.myRank = param.myRank;
            ctx.groups[group].kernelArg.rankSize = param.rankSize;
        }
        auto &arg = ctx.groups[group].kernelArg;
        arg.peers[arg.peerCount] = descs[i].remoteRank;
        arg.channels[arg.peerCount] = handles[i];
        ++arg.peerCount;
        ctx.peerLinks[descs[i].remoteRank].channel = handles[i];
    }
    ClassifySelectedPaths(ctx);
    HCCL_DEBUG("[Scatter v20 topology] rank=%u N=%u kind=%u evidence=%u reason=%u "
        "localMask=0x%x mesh=%u clos=%u",
        ctx.myRank, ctx.rankSize, static_cast<uint32_t>(ctx.topology.kind),
        static_cast<uint32_t>(ctx.topology.evidence), static_cast<uint32_t>(ctx.topology.reason),
        ctx.topology.localMemberMask, ctx.topology.meshLayer, ctx.topology.closLayer);
    return HCCL_SUCCESS;
}

HcclResult RegisterSerialKernel(CcuInsHandle instance, uint32_t dieId, const char *name,
    const void *function, const void **args, uint32_t argCount, CcuKernelHandle &handle)
{
    handle = 0;
    const CcuResult startRet = HcommCcuKernelRegisterStart(instance);
    if (startRet != CCU_SUCCESS) {
        HCCL_ERROR("CCU register start failed: kernel=%s die=%u ret=%d",
            name, dieId, static_cast<int>(startRet));
        return ConvertCcuToHccl(startRet);
    }

    CcuKernelHandle candidate = 0;
    const CcuResult registerRet = HcommCcuKernelRegister(instance, dieId, name,
        function, args, argCount, &candidate);
    // End every successfully started round, including an aborted registration.
    // Publish the handle only after the round has translated successfully.
    const CcuResult endRet = HcommCcuKernelRegisterEnd(instance);
    if (registerRet != CCU_SUCCESS) {
        HCCL_ERROR("CCU register failed: kernel=%s die=%u ret=%d endRet=%d",
            name, dieId, static_cast<int>(registerRet), static_cast<int>(endRet));
        return ConvertCcuToHccl(registerRet);
    }
    if (endRet != CCU_SUCCESS) {
        HCCL_ERROR("CCU register end failed: kernel=%s die=%u ret=%d",
            name, dieId, static_cast<int>(endRet));
        return ConvertCcuToHccl(endRet);
    }
    CHK_PRT_RET(candidate == 0, HCCL_ERROR("Empty CCU kernel handle: kernel=%s", name), HCCL_E_INTERNAL);
    handle = candidate;
    return HCCL_SUCCESS;
}

HcclResult RegisterParallelGroups(CcuInsHandle instance, AlgResourceCtx &ctx, const char *stage,
    const void *function, CcuKernelHandle ops_hccl::DirectGroup::*member)
{
    if (ctx.groupCount == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(ctx.groupCount > ops_hccl::DIRECT_MAX_DIE_GROUPS,
        HCCL_ERROR("Invalid Scatter v20 parallel group count"), HCCL_E_INTERNAL);
    CcuKernelHandle candidates[ops_hccl::DIRECT_MAX_DIE_GROUPS]{};
    CHK_RET_CCU(HcommCcuKernelRegisterStart(instance));
    CcuResult registerRet = CCU_SUCCESS;
    uint32_t failedGroup = INVALID_VALUE_RANKID;
    for (uint32_t g = 0; g < ctx.groupCount; ++g) {
        auto &group = ctx.groups[g];
        const void *args[] = {&group.kernelArg};
        char name[96];
        std::snprintf(name, sizeof(name), "scatter_%s_v20_r%u_d%u", stage, ctx.myRank, group.dieId);
        registerRet = HcommCcuKernelRegister(instance, group.dieId, name,
            function, args, 1, &candidates[g]);
        if (registerRet != CCU_SUCCESS) {
            failedGroup = g;
            break;
        }
    }
    // Both actual-die groups may run together. End even an aborted round and
    // publish their handles only after the whole concurrent round translates.
    const CcuResult endRet = HcommCcuKernelRegisterEnd(instance);
    if (registerRet != CCU_SUCCESS) {
        HCCL_ERROR("Scatter v20 parallel register failed: stage=%s group=%u die=%u ret=%d endRet=%d",
            stage, failedGroup, ctx.groups[failedGroup].dieId,
            static_cast<int>(registerRet), static_cast<int>(endRet));
        return ConvertCcuToHccl(registerRet);
    }
    CHK_RET_CCU(endRet);
    for (uint32_t g = 0; g < ctx.groupCount; ++g) {
        CHK_PRT_RET(candidates[g] == 0,
            HCCL_ERROR("Empty Scatter v20 parallel kernel: stage=%s group=%u", stage, g), HCCL_E_INTERNAL);
    }
    for (uint32_t g = 0; g < ctx.groupCount; ++g) {
        ctx.groups[g].*member = candidates[g];
        HCCL_DEBUG("[Scatter v20 register] stage=%s root=%u rank=%u die=%u handle=%llu",
            stage, ctx.myRank, ctx.myRank, ctx.groups[g].dieId, static_cast<unsigned long long>(candidates[g]));
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterDirectRootKernels(CcuInsHandle instance, AlgResourceCtx &ctx)
{
    using namespace ops_hccl;
    if (!V13_SMALL_ROOT_SPECIALIZATION || ctx.groupCount == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(ctx.groupCount > DIRECT_MAX_DIE_GROUPS,
        HCCL_ERROR("Invalid Scatter v20 direct root group count"), HCCL_E_INTERNAL);
    struct Variant {
        const char *name;
        const void *function;
        CcuKernelHandle DirectGroup::*member;
    };
    const void *copyFunction = reinterpret_cast<const void *>(CcuDirectRootCopyKernel);
    if (UseFourByOneCompact(ctx.topology, ctx.rankSize)) {
        copyFunction = reinterpret_cast<const void *>(CcuFourByOneDirectRootCopyKernel);
    } else if (UseFourByThreeCompact(ctx.topology, ctx.rankSize)) {
        copyFunction = reinterpret_cast<const void *>(CcuFourByThreeDirectRootCopyKernel);
    }
    const Variant variants[] = {
        {"direct_root", reinterpret_cast<const void *>(CcuDirectRootKernel), &DirectGroup::directRoot},
        {"direct_root_copy", copyFunction, &DirectGroup::directRootCopy}
    };
    constexpr uint32_t variantCount = sizeof(variants) / sizeof(variants[0]);
    CcuKernelHandle candidates[variantCount][DIRECT_MAX_DIE_GROUPS]{};
    CHK_RET_CCU(HcommCcuKernelRegisterStart(instance));
    CcuResult registerRet = CCU_SUCCESS;
    uint32_t failedVariant = INVALID_VALUE_RANKID;
    uint32_t failedGroup = INVALID_VALUE_RANKID;
    for (uint32_t v = 0; v < variantCount && registerRet == CCU_SUCCESS; ++v) {
        for (uint32_t g = 0; g < ctx.groupCount; ++g) {
            const auto &group = ctx.groups[g];
            const void *args[] = {&group.kernelArg};
            char name[96];
            std::snprintf(name, sizeof(name), "scatter_%s_v20_r%u_d%u",
                variants[v].name, ctx.myRank, group.dieId);
            registerRet = HcommCcuKernelRegister(instance, group.dieId, name,
                variants[v].function, args, 1, &candidates[v][g]);
            if (registerRet != CCU_SUCCESS) {
                failedVariant = v;
                failedGroup = g;
                break;
            }
        }
    }
    // Host may pair either die's copy variant with the other die's no-copy
    // variant. All combinations need one resource round; old smallRoot handles
    // belong to a separate round and cannot join this concurrent family.
    const CcuResult endRet = HcommCcuKernelRegisterEnd(instance);
    if (registerRet != CCU_SUCCESS) {
        HCCL_ERROR("Scatter v20 direct root register failed: stage=%s group=%u die=%u ret=%d endRet=%d",
            variants[failedVariant].name, failedGroup, ctx.groups[failedGroup].dieId,
            static_cast<int>(registerRet), static_cast<int>(endRet));
        return ConvertCcuToHccl(registerRet);
    }
    CHK_RET_CCU(endRet);
    for (uint32_t v = 0; v < variantCount; ++v) {
        for (uint32_t g = 0; g < ctx.groupCount; ++g) {
            CHK_PRT_RET(candidates[v][g] == 0,
                HCCL_ERROR("Empty Scatter v20 direct root kernel: stage=%s group=%u", variants[v].name, g),
                HCCL_E_INTERNAL);
        }
    }
    for (uint32_t v = 0; v < variantCount; ++v) {
        for (uint32_t g = 0; g < ctx.groupCount; ++g) {
            ctx.groups[g].*variants[v].member = candidates[v][g];
            HCCL_DEBUG("[Scatter v20 register] stage=%s root=%u rank=%u die=%u handle=%llu",
                variants[v].name, ctx.myRank, ctx.myRank, ctx.groups[g].dieId,
                static_cast<unsigned long long>(candidates[v][g]));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterGroupSerialKernels(CcuInsHandle instance, AlgResourceCtx &ctx)
{
    using namespace ops_hccl;
    const bool rootPhase = UseEightSideRootPhase(ctx.topology, ctx.rankSize, ctx.myRank, ctx.groupCount);
    struct Stage {
        const char *name;
        const void *function;
        CcuKernelHandle DirectGroup::*member;
    };
    const Stage stages[] = {
        {"root_publish", reinterpret_cast<const void *>(CcuRootPublishKernel), &DirectGroup::publish},
        {"root_wait", reinterpret_cast<const void *>(CcuRootWaitKernel), &DirectGroup::rootWait},
        {"root_release", reinterpret_cast<const void *>(CcuRootReleaseKernel), &DirectGroup::rootRelease}
    };
    for (uint32_t g = 0; g < ctx.groupCount; ++g) {
        auto &group = ctx.groups[g];
        const void *args[] = {&group.kernelArg};
        for (const auto &stage : stages) {
            if (rootPhase && stage.member == &DirectGroup::publish) {
                // This slot already holds the jointly registered RPH9 kernel.
                continue;
            }
            char name[96];
            std::snprintf(name, sizeof(name), "scatter_%s_v20_r%u_d%u", stage.name, ctx.myRank, group.dieId);
            CHK_RET(RegisterSerialKernel(instance, group.dieId, name, stage.function,
                args, 1, group.*stage.member));
            HCCL_DEBUG("[Scatter v20 register] stage=%s root=%u rank=%u die=%u handle=%llu",
                stage.name, ctx.myRank, ctx.myRank, group.dieId,
                static_cast<unsigned long long>(group.*stage.member));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterPullGroupKernels(CcuInsHandle instance, AlgResourceCtx &ctx)
{
    using namespace ops_hccl;
    if (ctx.groupCount == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(ctx.groupCount > DIRECT_MAX_DIE_GROUPS,
        HCCL_ERROR("Invalid Scatter v20 pull group count"), HCCL_E_INTERNAL);
    for (uint32_t root = 0; root < ctx.rankSize; ++root) {
        if (root == ctx.myRank) {
            continue;
        }
        const uint32_t localMask = ctx.topology.localMemberMask;
        const uint32_t all = AllRankMask(ctx.rankSize);
        const bool rootLocal = (localMask & RankBit(root)) != 0;
        const uint32_t expectedRootMask = rootLocal ? localMask : (all ^ localMask);
        const bool twoByEight = ctx.rankSize == 16 && ctx.topology.kind == TopologyKind::TWO_BY_EIGHT &&
            (localMask & ~all) == 0 && (localMask & RankBit(ctx.myRank)) != 0 && RankMaskSize(localMask) == 8 &&
            (expectedRootMask & ~all) == 0 && (expectedRootMask & RankBit(root)) != 0 &&
            RankMaskSize(expectedRootMask) == 8;
        const bool fusedRoot8 = RootOnEightSide(ctx.topology, ctx.rankSize, root);
        PullKernelArg kernelArgs[DIRECT_MAX_DIE_GROUPS];
        PrefixRegArg prefixArgs[DIRECT_MAX_DIE_GROUPS];
        CcuKernelHandle candidates[DIRECT_MAX_DIE_GROUPS]{};
        CHK_RET_CCU(HcommCcuKernelRegisterStart(instance));
        CcuResult registerRet = CCU_SUCCESS;
        uint32_t failedGroup = INVALID_VALUE_RANKID;
        for (uint32_t g = 0; g < ctx.groupCount; ++g) {
            const auto &group = ctx.groups[g];
            kernelArgs[g] = group.kernelArg;
            kernelArgs[g].staticRoot = root;
            const bool prefix = twoByEight && group.dieId == ctx.peerLinks[root].dieId;
            const void *function = reinterpret_cast<const void *>(CcuPullGroupKernel);
            const char *stage = "pull_group";
            const void *args[] = {&kernelArgs[g]};
            if (prefix) {
                prefixArgs[g].kernel = kernelArgs[g];
                prefixArgs[g].expectedRootMask = expectedRootMask;
                args[0] = &prefixArgs[g];
                function = reinterpret_cast<const void *>(CcuTwoByEightPrefixPullGroupKernel);
                stage = "pull_2x8_prefix";
            } else if (fusedRoot8) {
                // BOTH actual dies publish zero BYTES with full META before draining peers;
                // this also includes a group whose sole peer is root.
                function = reinterpret_cast<const void *>(CcuEightPlusFourFusedPullGroupKernel);
                stage = "pull_8plus4_fused";
            }
            // Both argument arrays outlive the shared Start/End round. Runtime
            // root headers still authorize specialized reads inside the kernel.
            char name[96];
            std::snprintf(name, sizeof(name), "scatter_%s_v20_r%u_root%u_d%u",
                stage, ctx.myRank, root, group.dieId);
            registerRet = HcommCcuKernelRegister(instance, group.dieId, name,
                function, args, 1, &candidates[g]);
            if (registerRet != CCU_SUCCESS) {
                failedGroup = g;
                break;
            }
        }
        // Each future root has one concurrent die-group round. Baking in root
        // keeps each kernel's root-gap program small; other roots reuse resources.
        const CcuResult endRet = HcommCcuKernelRegisterEnd(instance);
        if (registerRet != CCU_SUCCESS) {
            HCCL_ERROR("Scatter v20 pull register failed: root=%u group=%u die=%u ret=%d endRet=%d",
                root, failedGroup, ctx.groups[failedGroup].dieId,
                static_cast<int>(registerRet), static_cast<int>(endRet));
            return ConvertCcuToHccl(registerRet);
        }
        CHK_RET_CCU(endRet);
        for (uint32_t g = 0; g < ctx.groupCount; ++g) {
            CHK_PRT_RET(candidates[g] == 0,
                HCCL_ERROR("Empty Scatter v20 pull kernel: root=%u group=%u", root, g), HCCL_E_INTERNAL);
        }
        for (uint32_t g = 0; g < ctx.groupCount; ++g) {
            ctx.pullGroups[root][g] = candidates[g];
            HCCL_DEBUG("[Scatter v20 register] stage=pull_group root=%u rank=%u die=%u handle=%llu",
                root, ctx.myRank, ctx.groups[g].dieId, static_cast<unsigned long long>(candidates[g]));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterOfferGroupKernels(CcuInsHandle instance, AlgResourceCtx &ctx)
{
    using namespace ops_hccl;
    if (!V13_BATCH_OFFERS || ctx.groupCount == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(ctx.groupCount > DIRECT_MAX_DIE_GROUPS,
        HCCL_ERROR("Invalid Scatter v20 offer group count"), HCCL_E_INTERNAL);
    for (uint32_t root = 0; root < ctx.rankSize; ++root) {
        if (root == ctx.myRank) {
            continue;
        }
        HelperPreparationPlan preparation;
        if (BuildHelperPreparationPlan(ctx, root, preparation)) {
            PullKernelArg kernelArgs[DIRECT_MAX_DIE_GROUPS];
            CcuKernelHandle candidates[DIRECT_MAX_DIE_GROUPS]{};
            CHK_RET_CCU(HcommCcuKernelRegisterStart(instance));
            CcuResult registerRet = CCU_SUCCESS;
            uint32_t failedGroup = INVALID_VALUE_RANKID;
            for (uint32_t g = 0; g < ctx.groupCount; ++g) {
                const auto &group = ctx.groups[g];
                kernelArgs[g] = group.kernelArg;
                kernelArgs[g].staticRoot = root;
                const void *args[] = {&kernelArgs[g]};
                char name[96];
                std::snprintf(name, sizeof(name), "scatter_helper_prepare_v20_r%u_root%u_d%u",
                    ctx.myRank, root, group.dieId);
                registerRet = HcommCcuKernelRegister(instance, group.dieId, name,
                    reinterpret_cast<const void *>(CcuTwoByEightPrepareKernel), args, 1, &candidates[g]);
                if (registerRet != CCU_SUCCESS) {
                    failedGroup = g;
                    break;
                }
            }
            // Root-only groups still capture META and prefetch. Both live
            // arguments and candidate handles span one concurrent die round.
            const CcuResult endRet = HcommCcuKernelRegisterEnd(instance);
            if (registerRet != CCU_SUCCESS) {
                HCCL_ERROR("Scatter v20 helper prepare register failed: root=%u group=%u ret=%d endRet=%d",
                    root, failedGroup, static_cast<int>(registerRet), static_cast<int>(endRet));
                return ConvertCcuToHccl(registerRet);
            }
            CHK_RET_CCU(endRet);
            for (uint32_t g = 0; g < ctx.groupCount; ++g) {
                CHK_PRT_RET(candidates[g] == 0,
                    HCCL_ERROR("Empty V20 helper prepare kernel: root=%u group=%u", root, g), HCCL_E_INTERNAL);
            }
            for (uint32_t g = 0; g < ctx.groupCount; ++g) {
                ctx.offerGroups[root][g] = candidates[g];
                HCCL_DEBUG("[Scatter v20 register] stage=helper_prepare root=%u rank=%u die=%u handle=%llu",
                    root, ctx.myRank, ctx.groups[g].dieId, static_cast<unsigned long long>(candidates[g]));
            }
            continue;
        }
        for (uint32_t g = 0; g < ctx.groupCount; ++g) {
            const auto &group = ctx.groups[g];
            bool hasOfferPeer = false;
            for (uint32_t i = 0; i < group.kernelArg.peerCount; ++i) {
                hasOfferPeer = hasOfferPeer || group.kernelArg.peers[i] != root;
            }
            if (!hasOfferPeer) {
                // Root receives no offer. Avoid a channel-free kernel when it
                // is this group's only peer; the corresponding handle stays 0.
                continue;
            }
            PullKernelArg kernelArg = group.kernelArg;
            kernelArg.staticRoot = root;
            const void *args[] = {&kernelArg};
            char name[96];
            std::snprintf(name, sizeof(name), "scatter_offer_group_v20_r%u_root%u_d%u",
                ctx.myRank, root, group.dieId);
            // Groups publish in sequence on the main thread. Separate resource
            // rounds are safe here; never launch these handles concurrently.
            CHK_RET(RegisterSerialKernel(instance, group.dieId, name,
                reinterpret_cast<const void *>(CcuOfferGroupKernel), args, 1, ctx.offerGroups[root][g]));
            HCCL_DEBUG("[Scatter v20 register] stage=offer_group root=%u rank=%u die=%u handle=%llu",
                root, ctx.myRank, group.dieId, static_cast<unsigned long long>(ctx.offerGroups[root][g]));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterPeerKernels(CcuInsHandle instance, AlgResourceCtx &ctx)
{
    using namespace ops_hccl;
    struct Stage {
        const char *name;
        const void *function;
        CcuKernelHandle *handles;
    };
    // The same receiver slot serves both Small and every 4x1 Large frame.
    const void *peerFunction = reinterpret_cast<const void *>(CcuSmallPullPeerKernel);
    if (UseFourByOneCompact(ctx.topology, ctx.rankSize)) {
        peerFunction = reinterpret_cast<const void *>(CcuFourByOnePullPeerKernel);
    } else if (UseFourByThreeCompact(ctx.topology, ctx.rankSize)) {
        // N12 Large uses PullGroup, so this replacement serves Small only.
        peerFunction = reinterpret_cast<const void *>(CcuFourByThreePullPeerKernel);
    }
    const Stage stages[] = {
        {"small_peer", peerFunction, ctx.smallPeers},
        {"root_capture", reinterpret_cast<const void *>(CcuRootCaptureKernel), ctx.captures},
        {"prefetch", reinterpret_cast<const void *>(CcuPrefetchKernel), ctx.prefetches},
        {"root_ack", reinterpret_cast<const void *>(CcuRootAckKernel), ctx.rootAcks},
        {"offer_send", reinterpret_cast<const void *>(CcuOfferSendKernel), ctx.offerSends},
        {"offer_ready", reinterpret_cast<const void *>(CcuOfferReadyKernel), ctx.offerReadies}
    };
    // Each peer may become root or receive an offer on a later call. Register
    // these serial roles once using the one already acquired channel.
    for (uint32_t peer = 0; peer < ctx.rankSize; ++peer) {
        if (peer == ctx.myRank) {
            continue;
        }
        const auto &link = ctx.peerLinks[peer];
        PullPeerArg arg{link.channel, ctx.myRank, ctx.rankSize};
        const void *args[] = {&arg};
        for (const auto &stage : stages) {
            char name[96];
            std::snprintf(name, sizeof(name), "scatter_%s_v20_r%u_p%u_d%u",
                stage.name, ctx.myRank, peer, link.dieId);
            // Kernel generation consumes arg synchronously. Each possible
            // root and stage uses a separate resource registration round.
            CHK_RET(RegisterSerialKernel(instance, link.dieId, name, stage.function,
                args, 1, stage.handles[peer]));
            HCCL_DEBUG("[Scatter v20 register] stage=%s root=%u rank=%u die=%u handle=%llu",
                stage.name, peer, ctx.myRank, link.dieId, static_cast<unsigned long long>(stage.handles[peer]));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterKernels(HcclComm comm, AlgResourceCtx &ctx)
{
    using namespace ops_hccl;
    CcuInsHandle instance = 0;
    uint32_t instanceCount = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &instance, &instanceCount));
    CHK_PRT_RET(instanceCount != 1 || instance == 0,
        HCCL_ERROR("Expected one communicator-owned CCU instance"), HCCL_E_INTERNAL);

    // Small/direct roots and large descriptor consumers each run their local
    // die groups concurrently. These two families are mutually exclusive.
    CHK_RET(RegisterParallelGroups(instance, ctx, "small_root",
        reinterpret_cast<const void *>(CcuSmallPullRootKernel), &DirectGroup::smallRoot));
    CHK_RET(RegisterDirectRootKernels(instance, ctx));
    CHK_RET(RegisterPullGroupKernels(instance, ctx));
    if (UseEightSideRootPhase(ctx.topology, ctx.rankSize, ctx.myRank, ctx.groupCount)) {
        CHK_RET(RegisterParallelGroups(instance, ctx, "root_phase",
            reinterpret_cast<const void *>(CcuEightPlusFourRootPhaseKernel), &DirectGroup::publish));
    }
    CHK_RET(RegisterGroupSerialKernels(instance, ctx));
    CHK_RET(RegisterPeerKernels(instance, ctx));
    CHK_RET(RegisterOfferGroupKernels(instance, ctx));
    // No-channel copy selects an enabled die in the 9.1 backend; dieId is only
    // a registration hint. Host never overlaps it with another local kernel.
    const uint32_t copyDie = ctx.groupCount == 0 ? 0 : ctx.groups[0].dieId;
    CHK_RET(RegisterSerialKernel(instance, copyDie, "scatter_copy_v20",
        reinterpret_cast<const void *>(CcuCopyKernel), nullptr, 0, ctx.copyKernel));
    CHK_PRT_RET(!ctx.Valid(), HCCL_ERROR("Invalid registered Scatter v20 context"), HCCL_E_INTERNAL);
    // The instance and all registered kernels remain owned by the communicator.
    return HCCL_SUCCESS;
}

HcclResult PrepareWorker(HcclComm comm, ThreadHandle mainThread, AlgResourceCtx &ctx)
{
    CHK_PRT_RET(ctx.groupCount != ops_hccl::DIRECT_MAX_DIE_GROUPS || ctx.workerReady != 0 || ctx.worker != 0,
        HCCL_ERROR("Invalid initial direct worker state"), HCCL_E_INTERNAL);
    ThreadHandle candidate = 0;
    // Use the legacy acquire entry; its engine validation differs from
    // WithConfig in the 9.1 source. No private thread/stream API is needed.
    CHK_RET(HcclThreadAcquire(comm, DIRECT_ENGINE, 1, ops_hccl::DIRECT_THREAD_NOTIFY_COUNT, &candidate));
    CHK_PRT_RET(candidate == 0 || candidate == mainThread,
        HCCL_ERROR("Direct worker must be distinct from the current main thread"), HCCL_E_INTERNAL);
    ctx.worker = candidate;
    ctx.workerReady = 1;
    return HCCL_SUCCESS;
}

bool ValidRange(const void *address, uint64_t bytes)
{
    const uintptr_t start = reinterpret_cast<uintptr_t>(address);
    return address != nullptr && bytes <= std::numeric_limits<uintptr_t>::max() - start;
}

HcclResult ScatterImpl(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    OpParam param{};
    std::snprintf(param.tag, sizeof(param.tag), "%s", "hccl_scatter_ccu_pull_v20_ctx51");
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.root = root;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_SCATTER;
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || param.myRank >= param.rankSize ||
        root >= param.rankSize || dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("Invalid Scatter rank, root or data type"), HCCL_E_PARA);
    CHK_PRT_RET(recvCount > std::numeric_limits<uint64_t>::max() / ops_hccl::ELEMENT_BYTES / param.rankSize,
        HCCL_ERROR("Scatter byte count overflow"), HCCL_E_PARA);
    if (recvCount == 0) {
        return HCCL_SUCCESS;
    }
    const uint64_t bytes = recvCount * ops_hccl::ELEMENT_BYTES;
    CHK_PTR_NULL(recvBuf);
    CHK_PRT_RET(!ValidRange(recvBuf, bytes), HCCL_ERROR("Invalid Scatter output range"), HCCL_E_PARA);
    if (param.myRank == root) {
        CHK_PTR_NULL(sendBuf);
        CHK_PRT_RET(!ValidRange(sendBuf, bytes * param.rankSize),
            HCCL_ERROR("Invalid Scatter input range"), HCCL_E_PARA);
    }

    CHK_RET(HcclThreadAcquireWithStream(comm, DIRECT_ENGINE, stream,
        ops_hccl::DIRECT_THREAD_NOTIFY_COUNT, &param.cpuThread));
    HcclDfxOpInfo dfxInfo{};
    dfxInfo.opType = static_cast<uint32_t>(param.opType);
    dfxInfo.dataType = static_cast<uint32_t>(dataType);
    dfxInfo.dataCount = recvCount;
    dfxInfo.root = root;
    dfxInfo.engine = DIRECT_ENGINE;
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, &dfxInfo));

    AlgResourceCtx ctx;
    const bool newContext =
        HcclEngineCtxGet(comm, param.tag, DIRECT_ENGINE, &param.resCtx, &param.ctxSize) != HCCL_SUCCESS;
    if (newContext) {
        ctx.myRank = param.myRank;
        ctx.rankSize = param.rankSize;
        CHK_RET(HcclGetHcclBuffer(comm, &ctx.localBuffer.addr, &ctx.localBuffer.size));
        // Round only scratch capacity; user data and the final +4B tail are never truncated.
        ctx.localBuffer.size -= ctx.localBuffer.size % ops_hccl::ELEMENT_BYTES;
        CHK_PRT_RET(!ValidRange(ctx.localBuffer.addr, ctx.localBuffer.size) ||
            ctx.localBuffer.size < ops_hccl::ELEMENT_BYTES,
            HCCL_ERROR("Insufficient HCCL scratch buffer"), HCCL_E_MEMORY);
        CHK_RET(AcquireChannels(comm, param, ctx));
        CHK_RET(RegisterKernels(comm, ctx));
    } else {
        CHK_PRT_RET(!ctx.DeSerialize(param.resCtx, param.ctxSize) || ctx.myRank != param.myRank ||
            ctx.rankSize != param.rankSize,
            HCCL_ERROR("Invalid cached Scatter v20 pull context"), HCCL_E_INTERNAL);
    }

    bool publishContext = newContext;
    const bool needsWorker = ctx.groupCount == ops_hccl::DIRECT_MAX_DIE_GROUPS &&
        (param.myRank == param.root ||
         (ops_hccl::ClassifyMessage(param.count) == ops_hccl::MessageClass::LARGE && param.rankSize != 4));
    if (needsWorker && ctx.workerReady == 0) {
        // Allocate before per-call work is enqueued, including small roots and
        // non-root large descriptor consumers with channels on both dies.
        CHK_RET(PrepareWorker(comm, param.cpuThread, ctx));
        publishContext = true;
    }
    if (publishContext) {
        const std::vector<char> serialized = ctx.Serialize();
        param.ctxSize = serialized.size();
        if (newContext) {
            CHK_RET(HcclEngineCtxCreate(comm, param.tag, DIRECT_ENGINE, param.ctxSize, &param.resCtx));
        }
        const HcclResult copyResult = HcclEngineCtxCopy(comm, DIRECT_ENGINE, param.tag,
            serialized.data(), serialized.size(), 0);
        if (copyResult != HCCL_SUCCESS) {
            // Do not leave a partially created or updated context for reuse.
            (void)HcclEngineCtxDestroy(comm, param.tag, DIRECT_ENGINE);
            return copyResult;
        }
    }
    return ops_hccl::ExecOp(param);
}
} // namespace

HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    try {
        return ScatterImpl(sendBuf, recvBuf, recvCount, dataType, root, comm, stream);
    } catch (const std::bad_alloc &) {
        HCCL_ERROR("Scatter host allocation failed");
        return HCCL_E_MEMORY;
    } catch (const std::exception &) {
        HCCL_ERROR("Scatter resource construction failed");
        return HCCL_E_INTERNAL;
    }
}
