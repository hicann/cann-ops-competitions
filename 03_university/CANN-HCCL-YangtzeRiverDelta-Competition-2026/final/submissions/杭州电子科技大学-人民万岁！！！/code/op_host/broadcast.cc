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
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <vector>

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>
#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_ccu_res.h>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "exec_op.h"
#include "ccu_kernel.h"

namespace {
constexpr uint32_t DEFAULT_CHANNEL_NOTIFY_NUM = 2;
constexpr uint32_t CUT_THROUGH_CHANNEL_NOTIFY_NUM = 3;
constexpr uint32_t MAX_CCU_KERNEL_NUM = 2;
constexpr uint64_t SCATTER_ALLGATHER_THRESHOLD = 16ULL * 1024ULL * 1024ULL;
constexpr uint64_t MEM2MEM_MAX_SIZE = 256ULL * 1024ULL * 1024ULL;
constexpr uint64_t CUT_THROUGH_SIZE_512M = 512ULL * 1024ULL * 1024ULL;
constexpr uint64_t CUT_THROUGH_SIZE_400M_PLUS_4 = 400ULL * 1024ULL * 1024ULL + 4ULL;
constexpr uint64_t ASYMMETRIC_DIRECT_512K_SIZE = 512ULL * 1024ULL;
constexpr uint32_t CUT_THROUGH_READY_NOTIFY_IDX0 = 1;
constexpr uint32_t CUT_THROUGH_READY_NOTIFY_IDX1 = 2;

uint32_t GetTopoPriority(CommTopo topoType)
{
    if (topoType == CommTopo::COMM_TOPO_1DMESH) {
        return 0;
    }
    if (topoType == CommTopo::COMM_TOPO_CLOS) {
        return 1;
    }
    return 2;
}

struct KernelGroupKey {
    uint32_t layerPriority;
    CommTopo topoType;
    uint32_t layer;
    uint32_t dieId;

    bool operator<(const KernelGroupKey &other) const
    {
        if (layerPriority != other.layerPriority) {
            return layerPriority < other.layerPriority;
        }
        const uint32_t priority = GetTopoPriority(topoType);
        const uint32_t otherPriority = GetTopoPriority(other.topoType);
        if (priority != otherPriority) {
            return priority < otherPriority;
        }
        if (topoType != other.topoType) {
            return static_cast<int32_t>(topoType) < static_cast<int32_t>(other.topoType);
        }
        if (layer != other.layer) {
            return layer < other.layer;
        }
        return dieId < other.dieId;
    }
};

struct LinkCandidate {
    uint32_t remoteRank;
    uint32_t layer;
    uint32_t layerPriority;
    CommTopo topoType;
    CommLink link;
    EndpointAttrDieId dieId;
    EndpointAttrBwCoeff bwCoeff;
};

struct NetLayerInfo {
    uint32_t layer;
    uint32_t priority;
    CommTopo topoType;
};

struct ChainRole {
    uint32_t hasPrev;
    uint32_t hasNext;
    uint32_t prevChannelIdx;
    uint32_t nextChannelIdx;
    uint32_t isFirst;
    uint32_t isLast;
    uint32_t isPairFirst;
    uint32_t isPairLast;
    uint32_t readyMask;
    uint32_t cutReadyNotifyIdx0;
    uint32_t cutReadyMask0;
    uint32_t cutReadyNotifyIdx1;
    uint32_t cutReadyMask1;
    uint32_t cutReadyNotifyIdx2;
    uint32_t cutReadyMask2;
    uint32_t cutReadyNotifyIdx3;
    uint32_t cutReadyMask3;
    uint32_t cutReadyNotifyIdx4;
    uint32_t cutReadyMask4;
};

uint32_t GetCutThroughNotifyIdx(uint32_t segmentIdx)
{
    return segmentIdx < CUT_THROUGH_MAX_SEGMENTS / 2 ?
        CUT_THROUGH_READY_NOTIFY_IDX0 : CUT_THROUGH_READY_NOTIFY_IDX1;
}

uint32_t GetCutThroughNotifyMask(uint32_t segmentIdx)
{
    return 1U << (segmentIdx % (CUT_THROUGH_MAX_SEGMENTS / 2));
}

HcclResult GetNetLayers(HcclComm comm, std::vector<NetLayerInfo> &netLayers)
{
    uint32_t *layers = nullptr;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layers, &layerNum));
    CHK_PRT_RET(layerNum == 0 || layers == nullptr, HCCL_ERROR("No network layer found"), HCCL_E_NOT_FOUND);

    const std::vector<uint32_t> layerIds(layers, layers + layerNum);
    netLayers.reserve(layerNum);
    for (uint32_t layerIdx = 0; layerIdx < layerNum; layerIdx++) {
        CommTopo topoType = CommTopo::COMM_TOPO_RESERVED;
        CHK_RET(HcclRankGraphGetTopoTypeByLayer(comm, layerIds[layerIdx], &topoType));
        CHK_PRT_RET(topoType == CommTopo::COMM_TOPO_RESERVED,
            HCCL_ERROR("Unsupported topology type on layer %u", layerIds[layerIdx]), HCCL_E_NOT_SUPPORT);
        netLayers.push_back(NetLayerInfo{layerIds[layerIdx], layerIdx, topoType});
    }
    return HCCL_SUCCESS;
}

HcclResult CollectPeerCandidates(HcclComm comm, uint32_t localRank, uint32_t remoteRank,
    const std::vector<NetLayerInfo> &netLayers, std::vector<LinkCandidate> &candidates)
{
    for (const NetLayerInfo &layerInfo : netLayers) {
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        HcclResult ret = HcclRankGraphGetLinks(
            comm, layerInfo.layer, localRank, remoteRank, &links, &linkNum);
        if (ret != HCCL_SUCCESS || links == nullptr || linkNum == 0) {
            continue;
        }

        for (uint32_t linkIdx = 0; linkIdx < linkNum; linkIdx++) {
            CommLink link = links[linkIdx];
            if (link.linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                continue;
            }

            EndpointAttrDieId dieId = 0;
            CHK_RET(HcclRankGraphGetEndpointInfo(comm, localRank, &link.srcEndpointDesc, ENDPOINT_ATTR_DIE_ID,
                sizeof(dieId), &dieId));

            EndpointAttrBwCoeff bwCoeff = 0;
            ret = HcclRankGraphGetEndpointInfo(comm, localRank, &link.srcEndpointDesc, ENDPOINT_ATTR_BW_COEFF,
                sizeof(bwCoeff), &bwCoeff);
            if (ret != HCCL_SUCCESS) {
                bwCoeff = 0;
            }
            candidates.push_back(
                LinkCandidate{remoteRank, layerInfo.layer, layerInfo.priority, layerInfo.topoType, link, dieId,
                    bwCoeff});
        }
    }

    CHK_PRT_RET(candidates.empty(),
        HCCL_ERROR("No UBC_CTP link found from rank %u to rank %u", localRank, remoteRank), HCCL_E_NOT_FOUND);
    return HCCL_SUCCESS;
}

uint32_t GetPreferredLayer(const std::vector<LinkCandidate> &candidates)
{
    const LinkCandidate *preferred = &candidates[0];
    for (const LinkCandidate &candidate : candidates) {
        if (candidate.layerPriority < preferred->layerPriority) {
            preferred = &candidate;
        }
    }
    return preferred->layer;
}

const LinkCandidate *SelectCandidateForDie(
    const std::vector<LinkCandidate> &candidates, uint32_t preferredLayer, uint32_t dieId)
{
    const LinkCandidate *selected = nullptr;
    for (const LinkCandidate &candidate : candidates) {
        if (candidate.layer != preferredLayer || candidate.dieId != dieId) {
            continue;
        }
        if (selected == nullptr || candidate.bwCoeff > selected->bwCoeff) {
            selected = &candidate;
        }
    }
    return selected;
}

const LinkCandidate *SelectBestCandidate(
    const std::vector<LinkCandidate> &candidates, uint32_t preferredLayer)
{
    const LinkCandidate *selected = nullptr;
    for (const LinkCandidate &candidate : candidates) {
        if (candidate.layer != preferredLayer) {
            continue;
        }
        if (selected == nullptr || candidate.bwCoeff > selected->bwCoeff ||
            (candidate.bwCoeff == selected->bwCoeff && candidate.dieId < selected->dieId)) {
            selected = &candidate;
        }
    }
    return selected;
}

HcclResult SelectOneLinkPerPeer(HcclComm comm, const OpParam &param,
    const std::vector<uint32_t> &peerRanks,
    std::map<KernelGroupKey, std::vector<LinkCandidate>> &selectedByGroup)
{
    CHK_PRT_RET(peerRanks.empty(), HCCL_ERROR("Peer list is empty"), HCCL_E_PARA);

    std::set<uint32_t> requestedPeers;
    for (const uint32_t peerRank : peerRanks) {
        CHK_PRT_RET(peerRank >= param.rankSize || peerRank == param.myRank,
            HCCL_ERROR("Invalid peer %u for rank %u in rank size %u", peerRank, param.myRank, param.rankSize),
            HCCL_E_PARA);
        CHK_PRT_RET(!requestedPeers.insert(peerRank).second,
            HCCL_ERROR("Peer %u is requested more than once", peerRank), HCCL_E_PARA);
    }

    std::vector<NetLayerInfo> netLayers;
    CHK_RET(GetNetLayers(comm, netLayers));

    std::vector<std::vector<LinkCandidate>> candidatesByPeer(param.rankSize);
    std::vector<uint32_t> preferredLayerByPeer(param.rankSize, INVALID_VALUE_RANKID);
    std::map<uint32_t, uint32_t> peerCountByLayer;
    std::map<uint32_t, std::map<uint32_t, uint32_t>> dieCoverageByLayer;
    for (const uint32_t peerRank : peerRanks) {
        CHK_RET(CollectPeerCandidates(
            comm, param.myRank, peerRank, netLayers, candidatesByPeer[peerRank]));
        preferredLayerByPeer[peerRank] = GetPreferredLayer(candidatesByPeer[peerRank]);

        std::set<uint32_t> peerDies;
        for (const LinkCandidate &candidate : candidatesByPeer[peerRank]) {
            if (candidate.layer == preferredLayerByPeer[peerRank]) {
                peerDies.insert(candidate.dieId);
            }
        }
        CHK_PRT_RET(peerDies.empty(),
            HCCL_ERROR("No candidate found on preferred layer %u for peer %u",
                preferredLayerByPeer[peerRank], peerRank), HCCL_E_NOT_FOUND);
        peerCountByLayer[preferredLayerByPeer[peerRank]]++;
        for (const uint32_t dieId : peerDies) {
            dieCoverageByLayer[preferredLayerByPeer[peerRank]][dieId]++;
        }
    }

    const uint32_t peerNum = peerRanks.size();
    std::map<uint32_t, uint32_t> preferredDieByLayer;
    for (const auto &layerCoverage : dieCoverageByLayer) {
        uint32_t preferredDie = INVALID_VALUE_RANKID;
        uint64_t preferredDieScore = 0;
        for (const auto &coverage : layerCoverage.second) {
            if (coverage.second != peerCountByLayer[layerCoverage.first]) {
                continue;
            }

            uint64_t score = 0;
            for (const uint32_t peerRank : peerRanks) {
                if (preferredLayerByPeer[peerRank] != layerCoverage.first) {
                    continue;
                }
                const LinkCandidate *candidate = SelectCandidateForDie(
                    candidatesByPeer[peerRank], layerCoverage.first, coverage.first);
                CHK_PRT_RET(candidate == nullptr, HCCL_ERROR("Candidate selection failed"), HCCL_E_INTERNAL);
                score += candidate->bwCoeff;
            }
            if (preferredDie == INVALID_VALUE_RANKID || score > preferredDieScore) {
                preferredDie = coverage.first;
                preferredDieScore = score;
            }
        }
        preferredDieByLayer[layerCoverage.first] = preferredDie;
    }

    std::set<uint32_t> selectedPeers;
    for (const uint32_t peerRank : peerRanks) {
        const LinkCandidate *selected = nullptr;
        const uint32_t preferredDie = preferredDieByLayer[preferredLayerByPeer[peerRank]];
        if (preferredDie != INVALID_VALUE_RANKID) {
            selected = SelectCandidateForDie(
                candidatesByPeer[peerRank], preferredLayerByPeer[peerRank], preferredDie);
        } else {
            selected = SelectBestCandidate(candidatesByPeer[peerRank], preferredLayerByPeer[peerRank]);
        }
        CHK_PRT_RET(selected == nullptr, HCCL_ERROR("No channel candidate selected"), HCCL_E_INTERNAL);
        CHK_PRT_RET(!selectedPeers.insert(peerRank).second,
            HCCL_ERROR("Peer %u is selected more than once", peerRank), HCCL_E_INTERNAL);

        const KernelGroupKey groupKey{
            selected->layerPriority, selected->topoType, selected->layer, selected->dieId};
        selectedByGroup[groupKey].push_back(*selected);
    }

    CHK_PRT_RET(selectedPeers.size() != peerNum,
        HCCL_ERROR("Selected peer count %zu does not match expected %u", selectedPeers.size(), peerNum),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(selectedByGroup.empty() || selectedByGroup.size() > MAX_CCU_KERNEL_NUM,
        HCCL_ERROR("Unsupported CCU topology/layer/die group count %zu", selectedByGroup.size()),
        HCCL_E_NOT_SUPPORT);
    return HCCL_SUCCESS;
}

HcclResult AcquireChannels(HcclComm comm, CommEngine engine, const OpParam &param,
    const std::map<KernelGroupKey, std::vector<LinkCandidate>> &selectedByGroup,
    std::vector<KernelGroupKey> &kernelGroups, std::vector<std::vector<ChannelHandle>> &channelsPerKernel,
    std::vector<std::vector<uint32_t>> &peersPerKernel, uint32_t channelNotifyNum)
{
    uint32_t kernelIdx = 0;
    for (const auto &group : selectedByGroup) {
        kernelGroups.push_back(group.first);
        channelsPerKernel.emplace_back();
        peersPerKernel.emplace_back();
        std::vector<ChannelHandle> &kernelChannels = channelsPerKernel.back();
        std::vector<uint32_t> &kernelPeers = peersPerKernel.back();
        kernelChannels.resize(group.second.size());
        kernelPeers.reserve(group.second.size());

        HCCL_INFO("rank[%u] kernelGroup[%u] topo[%d] layer[%u] die[%u] peerCount[%zu]", param.myRank,
            kernelIdx, static_cast<int32_t>(group.first.topoType), group.first.layer, group.first.dieId,
            group.second.size());

        for (uint32_t channelIdx = 0; channelIdx < group.second.size(); channelIdx++) {
            const LinkCandidate &candidate = group.second[channelIdx];
            kernelPeers.push_back(candidate.remoteRank);
            HcclChannelDesc desc;
            CHK_RET(HcclChannelDescInit(&desc, 1));
            desc.remoteRank = candidate.remoteRank;
            desc.notifyNum = channelNotifyNum;
            desc.channelProtocol = candidate.link.linkAttr.linkProtocol;
            desc.localEndpoint.protocol = candidate.link.srcEndpointDesc.protocol;
            desc.localEndpoint.commAddr = candidate.link.srcEndpointDesc.commAddr;
            desc.localEndpoint.loc = candidate.link.srcEndpointDesc.loc;
            desc.remoteEndpoint.protocol = candidate.link.dstEndpointDesc.protocol;
            desc.remoteEndpoint.commAddr = candidate.link.dstEndpointDesc.commAddr;
            desc.remoteEndpoint.loc = candidate.link.dstEndpointDesc.loc;

            CHK_RET(HcclChannelAcquire(comm, engine, &desc, 1, &kernelChannels[channelIdx]));
            HCCL_INFO("rank[%u] peer[%u] topo[%d] layer[%u] die[%u] bwCoeff[%u] kernelIdx[%u] channelIdx[%u]",
                param.myRank, candidate.remoteRank, static_cast<int32_t>(candidate.topoType), candidate.layer,
                candidate.dieId, candidate.bwCoeff, kernelIdx, channelIdx);
        }
        kernelIdx++;
    }
    return HCCL_SUCCESS;
}

HcclResult PromoteRootChannelGroup(const OpParam &param, std::vector<KernelGroupKey> &kernelGroups,
    std::vector<std::vector<ChannelHandle>> &channelsPerKernel,
    std::vector<std::vector<uint32_t>> &peersPerKernel)
{
    if (param.myRank == param.root) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(kernelGroups.size() != channelsPerKernel.size() ||
        kernelGroups.size() != peersPerKernel.size() || kernelGroups.empty(),
        HCCL_ERROR("Owner-ready SAG group vectors are inconsistent"), HCCL_E_INTERNAL);

    uint32_t rootGroupIdx = INVALID_VALUE_RANKID;
    uint32_t rootChannelCount = 0;
    for (uint32_t groupIdx = 0; groupIdx < peersPerKernel.size(); groupIdx++) {
        for (const uint32_t peer : peersPerKernel[groupIdx]) {
            if (peer == param.root) {
                rootGroupIdx = groupIdx;
                rootChannelCount++;
            }
        }
    }
    CHK_PRT_RET(rootChannelCount != 1 || rootGroupIdx == INVALID_VALUE_RANKID,
        HCCL_ERROR("Owner-ready SAG found %u root channels for rank %u", rootChannelCount, param.myRank),
        HCCL_E_INTERNAL);

    // Thread 0 is the stream-bound main thread.  Keeping the sole root channel
    // in group 0 lets its Scatter completion release every auxiliary thread
    // with one ordered local-notify edge, without a full local barrier.
    if (rootGroupIdx != 0) {
        std::swap(kernelGroups[0], kernelGroups[rootGroupIdx]);
        std::swap(channelsPerKernel[0], channelsPerKernel[rootGroupIdx]);
        std::swap(peersPerKernel[0], peersPerKernel[rootGroupIdx]);
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterKernelGroup(CcuInsHandle insHandle, const OpParam &param,
    const std::vector<KernelGroupKey> &kernelGroups,
    const std::vector<std::vector<ChannelHandle>> &channelsPerKernel, const char *phaseName,
    const void *kernelFunc, std::vector<CcuKernelHandle> &kernelHandles, const ChainRole *chainRole = nullptr,
    const std::vector<std::vector<uint32_t>> *peersPerKernel = nullptr,
    uint64_t registeredBaseAddr = 0, uint64_t registeredToken = 0)
{
    CHK_PRT_RET(kernelGroups.size() != channelsPerKernel.size(),
        HCCL_ERROR("Kernel group count %zu does not match channel group count %zu", kernelGroups.size(),
            channelsPerKernel.size()), HCCL_E_INTERNAL);
    if (peersPerKernel != nullptr) {
        CHK_PRT_RET(peersPerKernel->size() != channelsPerKernel.size(),
            HCCL_ERROR("Peer group count %zu does not match channel group count %zu",
                peersPerKernel->size(), channelsPerKernel.size()), HCCL_E_INTERNAL);
    }

    uint32_t rootChannelCount = 0;
    if (peersPerKernel != nullptr) {
        for (uint32_t groupIdx = 0; groupIdx < peersPerKernel->size(); groupIdx++) {
            CHK_PRT_RET((*peersPerKernel)[groupIdx].size() != channelsPerKernel[groupIdx].size(),
                HCCL_ERROR("Peer count %zu does not match channel count %zu in group %u",
                    (*peersPerKernel)[groupIdx].size(), channelsPerKernel[groupIdx].size(), groupIdx),
                HCCL_E_INTERNAL);
            for (const uint32_t peer : (*peersPerKernel)[groupIdx]) {
                if (peer == param.root) {
                    rootChannelCount++;
                }
            }
        }
        CHK_PRT_RET((param.myRank == param.root && rootChannelCount != 0) ||
            (param.myRank != param.root && rootChannelCount != 1),
            HCCL_ERROR("Lean-SAG root channel count %u is invalid for rank %u", rootChannelCount, param.myRank),
            HCCL_E_INTERNAL);
    }

    std::vector<std::shared_ptr<CcuKernelArgBroadcast>> kernelArgs(channelsPerKernel.size());
    kernelHandles.resize(channelsPerKernel.size());
    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));

    for (uint32_t kernelIdx = 0; kernelIdx < channelsPerKernel.size(); kernelIdx++) {
        kernelArgs[kernelIdx] = std::make_shared<CcuKernelArgBroadcast>();
        kernelArgs[kernelIdx]->rankId = param.myRank;
        kernelArgs[kernelIdx]->rankSize = param.rankSize;
        kernelArgs[kernelIdx]->rootId = param.root;
        kernelArgs[kernelIdx]->registeredBaseAddr = registeredBaseAddr;
        kernelArgs[kernelIdx]->registeredToken = registeredToken;
        kernelArgs[kernelIdx]->channelCount = channelsPerKernel[kernelIdx].size();
        kernelArgs[kernelIdx]->hasPrev = 0;
        kernelArgs[kernelIdx]->hasNext = 0;
        kernelArgs[kernelIdx]->prevChannelIdx = 0;
        kernelArgs[kernelIdx]->nextChannelIdx = 0;
        kernelArgs[kernelIdx]->chainIsFirst = 0;
        kernelArgs[kernelIdx]->chainIsLast = 0;
        kernelArgs[kernelIdx]->chainPairIsFirst = 0;
        kernelArgs[kernelIdx]->chainPairIsLast = 0;
        kernelArgs[kernelIdx]->chainReadyMask = 0;
        kernelArgs[kernelIdx]->cutReadyNotifyIdx0 = 0;
        kernelArgs[kernelIdx]->cutReadyMask0 = 0;
        kernelArgs[kernelIdx]->cutReadyNotifyIdx1 = 0;
        kernelArgs[kernelIdx]->cutReadyMask1 = 0;
        kernelArgs[kernelIdx]->cutReadyNotifyIdx2 = 0;
        kernelArgs[kernelIdx]->cutReadyMask2 = 0;
        kernelArgs[kernelIdx]->cutReadyNotifyIdx3 = 0;
        kernelArgs[kernelIdx]->cutReadyMask3 = 0;
        kernelArgs[kernelIdx]->cutReadyNotifyIdx4 = 0;
        kernelArgs[kernelIdx]->cutReadyMask4 = 0;
        kernelArgs[kernelIdx]->leanHasRootChannel = 0;
        kernelArgs[kernelIdx]->leanRootChannelIdx = 0;
        if (peersPerKernel != nullptr && param.myRank != param.root) {
            const std::vector<uint32_t> &peers = (*peersPerKernel)[kernelIdx];
            for (uint32_t peerIdx = 0; peerIdx < peers.size(); peerIdx++) {
                if (peers[peerIdx] == param.root) {
                    kernelArgs[kernelIdx]->leanHasRootChannel = 1;
                    kernelArgs[kernelIdx]->leanRootChannelIdx = peerIdx;
                    break;
                }
            }
        }
        if (chainRole != nullptr) {
            kernelArgs[kernelIdx]->hasPrev = chainRole->hasPrev;
            kernelArgs[kernelIdx]->hasNext = chainRole->hasNext;
            kernelArgs[kernelIdx]->prevChannelIdx = chainRole->prevChannelIdx;
            kernelArgs[kernelIdx]->nextChannelIdx = chainRole->nextChannelIdx;
            kernelArgs[kernelIdx]->chainIsFirst = chainRole->isFirst;
            kernelArgs[kernelIdx]->chainIsLast = chainRole->isLast;
            kernelArgs[kernelIdx]->chainPairIsFirst = chainRole->isPairFirst;
            kernelArgs[kernelIdx]->chainPairIsLast = chainRole->isPairLast;
            kernelArgs[kernelIdx]->chainReadyMask = chainRole->readyMask;
            kernelArgs[kernelIdx]->cutReadyNotifyIdx0 = chainRole->cutReadyNotifyIdx0;
            kernelArgs[kernelIdx]->cutReadyMask0 = chainRole->cutReadyMask0;
            kernelArgs[kernelIdx]->cutReadyNotifyIdx1 = chainRole->cutReadyNotifyIdx1;
            kernelArgs[kernelIdx]->cutReadyMask1 = chainRole->cutReadyMask1;
            kernelArgs[kernelIdx]->cutReadyNotifyIdx2 = chainRole->cutReadyNotifyIdx2;
            kernelArgs[kernelIdx]->cutReadyMask2 = chainRole->cutReadyMask2;
            kernelArgs[kernelIdx]->cutReadyNotifyIdx3 = chainRole->cutReadyNotifyIdx3;
            kernelArgs[kernelIdx]->cutReadyMask3 = chainRole->cutReadyMask3;
            kernelArgs[kernelIdx]->cutReadyNotifyIdx4 = chainRole->cutReadyNotifyIdx4;
            kernelArgs[kernelIdx]->cutReadyMask4 = chainRole->cutReadyMask4;
        }
        for (uint32_t channelIdx = 0; channelIdx < channelsPerKernel[kernelIdx].size(); channelIdx++) {
            kernelArgs[kernelIdx]->channels[channelIdx] = channelsPerKernel[kernelIdx][channelIdx];
        }

        char kernelName[64];
        int ret = sprintf_s(kernelName, sizeof(kernelName), "Ccu%sBroadcastT%uL%uD%u", phaseName,
            static_cast<uint32_t>(kernelGroups[kernelIdx].topoType),
            kernelGroups[kernelIdx].layer, kernelGroups[kernelIdx].dieId);
        CHK_PRT_RET(ret <= 0, HCCL_ERROR("Failed to build CCU kernel name"), HCCL_E_INTERNAL);

        const void *registerArgs[] = {kernelArgs[kernelIdx].get()};
        CHK_RET_CCU(HcommCcuKernelRegister(
            insHandle, 0, kernelName, kernelFunc, registerArgs, 1, &kernelHandles[kernelIdx]));
    }

    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
    return HCCL_SUCCESS;
}

HcclResult RegisterKernels(HcclComm comm, const OpParam &param,
    const std::vector<KernelGroupKey> &kernelGroups,
    const std::vector<std::vector<ChannelHandle>> &channelsPerKernel, AlgResourceCtx &resCtx,
    const ChainRole *chainRole = nullptr)
{
    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1, HCCL_ERROR("Expected one CCU instance, but got %u", insNum), HCCL_E_INTERNAL);

    if (resCtx.algorithm == BroadcastAlgorithm::ASYMMETRIC_PULL_512K_2ARG) {
        return RegisterKernelGroup(insHandle, param, kernelGroups, channelsPerKernel,
            "AsymPull512K2Arg",
            reinterpret_cast<const void *>(ops_hccl::CcuAsymmetricPull512K2ArgBroadcastKernel),
            resCtx.ccuKernels);
    }

    if (resCtx.algorithm == BroadcastAlgorithm::ASYMMETRIC_DIRECT_512K_REGISTERED) {
        CHK_RET(RegisterKernelGroup(insHandle, param, kernelGroups, channelsPerKernel,
            "AsymDirect512K2ArgFallback",
            reinterpret_cast<const void *>(ops_hccl::CcuAsymmetricDirect512K2ArgBroadcastKernel),
            resCtx.ccuKernels));
        const void *registeredKernelFunc = param.myRank == param.root ?
            reinterpret_cast<const void *>(ops_hccl::CcuAsymmetricDirect512KRegisteredBroadcastKernel) :
            reinterpret_cast<const void *>(ops_hccl::CcuAsymmetricDirect512KRegisteredNonRootBroadcastKernel);
        return RegisterKernelGroup(insHandle, param, kernelGroups, channelsPerKernel,
            "AsymDirect512KRegistered",
            registeredKernelFunc,
            resCtx.registeredFastKernels, nullptr, nullptr,
            resCtx.registeredBaseAddr, resCtx.registeredToken);
    }

    if (resCtx.algorithm == BroadcastAlgorithm::ASYMMETRIC_DIRECT_512K_2ARG) {
        return RegisterKernelGroup(insHandle, param, kernelGroups, channelsPerKernel, "AsymDirect512K2Arg",
            reinterpret_cast<const void *>(ops_hccl::CcuAsymmetricDirect512K2ArgBroadcastKernel),
            resCtx.ccuKernels);
    }

    if (resCtx.algorithm == BroadcastAlgorithm::ASYMMETRIC_DIRECT) {
        return RegisterKernelGroup(insHandle, param, kernelGroups, channelsPerKernel, "AsymDirect",
            reinterpret_cast<const void *>(ops_hccl::CcuAsymmetricDirectBroadcastKernel),
            resCtx.ccuKernels);
    }

    if (resCtx.algorithm == BroadcastAlgorithm::DIRECT) {
        return RegisterKernelGroup(insHandle, param, kernelGroups, channelsPerKernel, "Direct",
            reinterpret_cast<const void *>(ops_hccl::CcuDirectBroadcastKernel), resCtx.ccuKernels);
    }

    if (resCtx.algorithm == BroadcastAlgorithm::PERSISTENT_ROLLING2_400) {
        CHK_PRT_RET(chainRole == nullptr || kernelGroups.size() != 1 || channelsPerKernel.size() != 1,
            HCCL_ERROR("Persistent Rolling-2 chain requires exactly one kernel group"), HCCL_E_INTERNAL);
        return RegisterKernelGroup(insHandle, param, kernelGroups, channelsPerKernel,
            "PersistentR2_400_8m",
            reinterpret_cast<const void *>(ops_hccl::CcuPersistentRolling2_400BroadcastKernel),
            resCtx.ccuKernels, chainRole);
    }

    if (resCtx.algorithm == BroadcastAlgorithm::PERSISTENT_ROLLING2_512) {
        CHK_PRT_RET(chainRole == nullptr || kernelGroups.size() != 1 || channelsPerKernel.size() != 1,
            HCCL_ERROR("Persistent 512 MiB Rolling-2 chain requires exactly one kernel group"),
            HCCL_E_INTERNAL);
        return RegisterKernelGroup(insHandle, param, kernelGroups, channelsPerKernel,
            "PersistentR2_512_8m",
            reinterpret_cast<const void *>(ops_hccl::CcuPersistentRolling2_512BroadcastKernel),
            resCtx.ccuKernels, chainRole);
    }

    if (resCtx.algorithm == BroadcastAlgorithm::SEGMENT_CUT_THROUGH_CHAIN) {
        CHK_PRT_RET(chainRole == nullptr || kernelGroups.size() != 1 || channelsPerKernel.size() != 1,
            HCCL_ERROR("Cut-through chain requires exactly one kernel group"), HCCL_E_INTERNAL);

        resCtx.ccuKernels.clear();
        resCtx.ccuKernels.reserve(CUT_THROUGH_KERNEL_COUNT);
        for (uint32_t pairSlot = 0; pairSlot < CUT_THROUGH_PAIR_COUNT; pairSlot++) {
            const uint32_t firstSegment = 2 * pairSlot;
            const uint32_t secondSegment = firstSegment + 1;
            ChainRole pairRole = *chainRole;
            pairRole.isFirst = pairSlot == 0 ? 1 : 0;
            pairRole.isLast = 0;
            pairRole.readyMask = 0;
            pairRole.cutReadyNotifyIdx0 = GetCutThroughNotifyIdx(firstSegment);
            pairRole.cutReadyMask0 = GetCutThroughNotifyMask(firstSegment);
            pairRole.cutReadyNotifyIdx1 = GetCutThroughNotifyIdx(secondSegment);
            pairRole.cutReadyMask1 = GetCutThroughNotifyMask(secondSegment);

            char phaseName[64];
            int ret = sprintf_s(phaseName, sizeof(phaseName), "CutThroughPairS%u", pairSlot);
            CHK_PRT_RET(ret <= 0, HCCL_ERROR("Failed to build cut-through pair name"), HCCL_E_INTERNAL);

            std::vector<CcuKernelHandle> pairHandles;
            CHK_RET(RegisterKernelGroup(insHandle, param, kernelGroups, channelsPerKernel, phaseName,
                reinterpret_cast<const void *>(ops_hccl::CcuCutThroughPairBroadcastKernel),
                pairHandles, &pairRole));
            CHK_PRT_RET(pairHandles.size() != 1,
                HCCL_ERROR("Cut-through pair %u registration returned %zu handles",
                    pairSlot, pairHandles.size()), HCCL_E_INTERNAL);
            resCtx.ccuKernels.push_back(pairHandles[0]);
        }

        ChainRole tailRole = *chainRole;
        tailRole.isFirst = 0;
        tailRole.isLast = 0;
        tailRole.readyMask = 0;
        tailRole.cutReadyNotifyIdx0 = GetCutThroughNotifyIdx(CUT_THROUGH_TAIL_SEGMENT);
        tailRole.cutReadyMask0 = GetCutThroughNotifyMask(CUT_THROUGH_TAIL_SEGMENT);
        tailRole.cutReadyNotifyIdx1 = 0;
        tailRole.cutReadyMask1 = 0;
        std::vector<CcuKernelHandle> tailHandles;
        CHK_RET(RegisterKernelGroup(insHandle, param, kernelGroups, channelsPerKernel,
            "CutThroughTailS24", reinterpret_cast<const void *>(ops_hccl::CcuCutThroughTailBroadcastKernel),
            tailHandles, &tailRole));
        CHK_PRT_RET(tailHandles.size() != 1,
            HCCL_ERROR("Cut-through tail registration returned %zu handles", tailHandles.size()),
            HCCL_E_INTERNAL);
        resCtx.ccuKernels.push_back(tailHandles[0]);

        ChainRole drainRole = *chainRole;
        drainRole.isFirst = 0;
        drainRole.isLast = 1;
        drainRole.readyMask = 0;
        drainRole.cutReadyNotifyIdx0 = 0;
        drainRole.cutReadyMask0 = 0;
        drainRole.cutReadyNotifyIdx1 = 0;
        drainRole.cutReadyMask1 = 0;
        std::vector<CcuKernelHandle> drainHandles;
        CHK_RET(RegisterKernelGroup(insHandle, param, kernelGroups, channelsPerKernel,
            "CutThroughDrain", reinterpret_cast<const void *>(ops_hccl::CcuUniqueDrainBroadcastKernel),
            drainHandles, &drainRole));
        CHK_PRT_RET(drainHandles.size() != 1,
            HCCL_ERROR("Cut-through drain registration returned %zu handles", drainHandles.size()),
            HCCL_E_INTERNAL);
        resCtx.ccuKernels.push_back(drainHandles[0]);

        CHK_PRT_RET(resCtx.ccuKernels.size() != CUT_THROUGH_KERNEL_COUNT,
            HCCL_ERROR("Registered %zu cut-through kernels, expected %u",
                resCtx.ccuKernels.size(), CUT_THROUGH_KERNEL_COUNT), HCCL_E_INTERNAL);
        return HCCL_SUCCESS;
    }

    if (resCtx.algorithm == BroadcastAlgorithm::QUAD4_ROLLING2_FINAL5_CHAIN) {
        CHK_PRT_RET(chainRole == nullptr || kernelGroups.size() != 1 || channelsPerKernel.size() != 1,
            HCCL_ERROR("Quad4 Rolling2 chain requires exactly one kernel group"), HCCL_E_INTERNAL);

        resCtx.ccuKernels.clear();
        resCtx.ccuKernels.reserve(QUAD4_RESOURCE_COUNT);
        for (uint32_t quadSlot = 0; quadSlot < QUAD4_KERNEL_COUNT; quadSlot++) {
            const uint32_t firstSegment = QUAD4_SEGMENTS_PER_KERNEL * quadSlot;
            ChainRole quadRole = *chainRole;
            quadRole.isFirst = quadSlot == 0 ? 1 : 0;
            quadRole.isLast = 0;
            quadRole.readyMask = 0;
            quadRole.cutReadyNotifyIdx0 = GetCutThroughNotifyIdx(firstSegment);
            quadRole.cutReadyMask0 = GetCutThroughNotifyMask(firstSegment);
            quadRole.cutReadyNotifyIdx1 = GetCutThroughNotifyIdx(firstSegment + 1);
            quadRole.cutReadyMask1 = GetCutThroughNotifyMask(firstSegment + 1);
            quadRole.cutReadyNotifyIdx2 = GetCutThroughNotifyIdx(firstSegment + 2);
            quadRole.cutReadyMask2 = GetCutThroughNotifyMask(firstSegment + 2);
            quadRole.cutReadyNotifyIdx3 = GetCutThroughNotifyIdx(firstSegment + 3);
            quadRole.cutReadyMask3 = GetCutThroughNotifyMask(firstSegment + 3);

            char phaseName[64];
            int ret = sprintf_s(phaseName, sizeof(phaseName), "Quad4Roll2S%u", quadSlot);
            CHK_PRT_RET(ret <= 0, HCCL_ERROR("Failed to build Quad4 phase name"), HCCL_E_INTERNAL);

            std::vector<CcuKernelHandle> quadHandles;
            CHK_RET(RegisterKernelGroup(insHandle, param, kernelGroups, channelsPerKernel, phaseName,
                reinterpret_cast<const void *>(ops_hccl::CcuQuad4Rolling2BroadcastKernel),
                quadHandles, &quadRole));
            CHK_PRT_RET(quadHandles.size() != 1,
                HCCL_ERROR("Quad4 kernel %u registration returned %zu handles",
                    quadSlot, quadHandles.size()), HCCL_E_INTERNAL);
            resCtx.ccuKernels.push_back(quadHandles[0]);
        }

        constexpr uint32_t finalFirstSegment = CUT_THROUGH_TAIL_SEGMENT -
            (FINAL5_SEGMENTS_PER_KERNEL - 1);
        ChainRole finalRole = *chainRole;
        finalRole.isFirst = 0;
        finalRole.isLast = 1;
        finalRole.readyMask = 0;
        finalRole.cutReadyNotifyIdx0 = GetCutThroughNotifyIdx(finalFirstSegment);
        finalRole.cutReadyMask0 = GetCutThroughNotifyMask(finalFirstSegment);
        finalRole.cutReadyNotifyIdx1 = GetCutThroughNotifyIdx(finalFirstSegment + 1);
        finalRole.cutReadyMask1 = GetCutThroughNotifyMask(finalFirstSegment + 1);
        finalRole.cutReadyNotifyIdx2 = GetCutThroughNotifyIdx(finalFirstSegment + 2);
        finalRole.cutReadyMask2 = GetCutThroughNotifyMask(finalFirstSegment + 2);
        finalRole.cutReadyNotifyIdx3 = GetCutThroughNotifyIdx(finalFirstSegment + 3);
        finalRole.cutReadyMask3 = GetCutThroughNotifyMask(finalFirstSegment + 3);
        finalRole.cutReadyNotifyIdx4 = GetCutThroughNotifyIdx(finalFirstSegment + 4);
        finalRole.cutReadyMask4 = GetCutThroughNotifyMask(finalFirstSegment + 4);
        std::vector<CcuKernelHandle> finalHandles;
        CHK_RET(RegisterKernelGroup(insHandle, param, kernelGroups, channelsPerKernel,
            "FinalRolling5DrainS20", reinterpret_cast<const void *>(
                ops_hccl::CcuFinalRolling5DrainBroadcastKernel), finalHandles, &finalRole));
        CHK_PRT_RET(finalHandles.size() != 1,
            HCCL_ERROR("Final Rolling5+Drain registration returned %zu handles", finalHandles.size()),
            HCCL_E_INTERNAL);
        resCtx.ccuKernels.push_back(finalHandles[0]);

        ChainRole drainRole = *chainRole;
        drainRole.isFirst = 0;
        drainRole.isLast = 1;
        drainRole.readyMask = 0;
        drainRole.cutReadyNotifyIdx0 = 0;
        drainRole.cutReadyMask0 = 0;
        drainRole.cutReadyNotifyIdx1 = 0;
        drainRole.cutReadyMask1 = 0;
        drainRole.cutReadyNotifyIdx2 = 0;
        drainRole.cutReadyMask2 = 0;
        drainRole.cutReadyNotifyIdx3 = 0;
        drainRole.cutReadyMask3 = 0;
        drainRole.cutReadyNotifyIdx4 = 0;
        drainRole.cutReadyMask4 = 0;
        std::vector<CcuKernelHandle> drainHandles;
        CHK_RET(RegisterKernelGroup(insHandle, param, kernelGroups, channelsPerKernel,
            "V12AQuad4Drain", reinterpret_cast<const void *>(ops_hccl::CcuUniqueDrainBroadcastKernel),
            drainHandles, &drainRole));
        CHK_PRT_RET(drainHandles.size() != 1,
            HCCL_ERROR("Quad4 drain registration returned %zu handles", drainHandles.size()), HCCL_E_INTERNAL);
        resCtx.ccuKernels.push_back(drainHandles[0]);

        CHK_PRT_RET(resCtx.ccuKernels.size() != QUAD4_RESOURCE_COUNT,
            HCCL_ERROR("Registered %zu Quad4 kernels, expected %u",
                resCtx.ccuKernels.size(), QUAD4_RESOURCE_COUNT), HCCL_E_INTERNAL);
        return HCCL_SUCCESS;
    }

    if (resCtx.algorithm == BroadcastAlgorithm::PIPELINED_CHAIN) {
        CHK_PRT_RET(chainRole == nullptr || kernelGroups.size() != 1 || channelsPerKernel.size() != 1,
            HCCL_ERROR("Pipelined chain requires exactly one kernel group"), HCCL_E_INTERNAL);

        constexpr uint32_t fallbackCount = 4;
        const char *fallbackPhaseNames[fallbackCount] = {
            "ChainFirst", "ChainMiddle", "ChainLast", "ChainSingle"};
        const void *fallbackKernelFuncs[fallbackCount] = {
            reinterpret_cast<const void *>(ops_hccl::CcuPipelinedChainBroadcastKernel),
            reinterpret_cast<const void *>(ops_hccl::CcuPipelinedChainBroadcastKernel),
            reinterpret_cast<const void *>(ops_hccl::CcuPipelinedChainBroadcastKernel),
            reinterpret_cast<const void *>(ops_hccl::CcuPipelinedChainBroadcastKernel)};
        constexpr uint32_t variantCount = static_cast<uint32_t>(ChainKernelVariant::COUNT);
        resCtx.ccuKernels.clear();
        resCtx.ccuKernels.reserve(variantCount);
        for (uint32_t variantIdx = 0; variantIdx < fallbackCount; variantIdx++) {
            ChainRole variantRole = *chainRole;
            variantRole.isFirst = variantIdx == static_cast<uint32_t>(ChainKernelVariant::FIRST) ||
                variantIdx == static_cast<uint32_t>(ChainKernelVariant::SINGLE);
            variantRole.isLast = variantIdx == static_cast<uint32_t>(ChainKernelVariant::LAST) ||
                variantIdx == static_cast<uint32_t>(ChainKernelVariant::SINGLE);
            variantRole.isPairFirst = 0;
            variantRole.isPairLast = 0;
            variantRole.readyMask = 0;

            std::vector<CcuKernelHandle> variantHandles;
            CHK_RET(RegisterKernelGroup(insHandle, param, kernelGroups, channelsPerKernel,
                fallbackPhaseNames[variantIdx], fallbackKernelFuncs[variantIdx],
                variantHandles, &variantRole));
            CHK_PRT_RET(variantHandles.size() != 1,
                HCCL_ERROR("Chain variant %u registration returned %zu handles",
                    variantIdx, variantHandles.size()), HCCL_E_INTERNAL);
            resCtx.ccuKernels.push_back(variantHandles[0]);
        }

        for (uint32_t slot = 0; slot < CHAIN_UNIQUE_PAIR_SLOT_COUNT; slot++) {
            ChainRole slotRole = *chainRole;
            slotRole.isFirst = slot == 0 ? 1 : 0;
            slotRole.isLast = 0;
            slotRole.isPairFirst = 0;
            slotRole.isPairLast = 0;
            slotRole.readyMask = 1U << slot;

            char phaseName[64];
            int ret = sprintf_s(phaseName, sizeof(phaseName), "ChainUniquePairS%u", slot);
            CHK_PRT_RET(ret <= 0, HCCL_ERROR("Failed to build unique pair phase name"), HCCL_E_INTERNAL);

            std::vector<CcuKernelHandle> slotHandles;
            CHK_RET(RegisterKernelGroup(insHandle, param, kernelGroups, channelsPerKernel, phaseName,
                reinterpret_cast<const void *>(ops_hccl::CcuUniquePairBroadcastKernel),
                slotHandles, &slotRole));
            CHK_PRT_RET(slotHandles.size() != 1,
                HCCL_ERROR("Unique pair slot %u registration returned %zu handles", slot, slotHandles.size()),
                HCCL_E_INTERNAL);
            resCtx.ccuKernels.push_back(slotHandles[0]);
        }

        ChainRole tailRole = *chainRole;
        tailRole.isFirst = 0;
        tailRole.isLast = 0;
        tailRole.isPairFirst = 0;
        tailRole.isPairLast = 0;
        tailRole.readyMask = 1U << CHAIN_UNIQUE_TAIL_SLOT;
        std::vector<CcuKernelHandle> tailHandles;
        CHK_RET(RegisterKernelGroup(insHandle, param, kernelGroups, channelsPerKernel,
            "ChainUniqueTailS12", reinterpret_cast<const void *>(ops_hccl::CcuUniqueTailBroadcastKernel),
            tailHandles, &tailRole));
        CHK_PRT_RET(tailHandles.size() != 1,
            HCCL_ERROR("Unique tail registration returned %zu handles", tailHandles.size()), HCCL_E_INTERNAL);
        resCtx.ccuKernels.push_back(tailHandles[0]);

        ChainRole drainRole = *chainRole;
        drainRole.isFirst = 0;
        drainRole.isLast = 1;
        drainRole.isPairFirst = 0;
        drainRole.isPairLast = 0;
        drainRole.readyMask = 0;
        std::vector<CcuKernelHandle> drainHandles;
        CHK_RET(RegisterKernelGroup(insHandle, param, kernelGroups, channelsPerKernel,
            "ChainUniqueDrain", reinterpret_cast<const void *>(ops_hccl::CcuUniqueDrainBroadcastKernel),
            drainHandles, &drainRole));
        CHK_PRT_RET(drainHandles.size() != 1,
            HCCL_ERROR("Unique drain registration returned %zu handles", drainHandles.size()), HCCL_E_INTERNAL);
        resCtx.ccuKernels.push_back(drainHandles[0]);

        CHK_PRT_RET(resCtx.ccuKernels.size() != variantCount,
            HCCL_ERROR("Registered %zu chain kernels, expected %u", resCtx.ccuKernels.size(), variantCount),
            HCCL_E_INTERNAL);
        return HCCL_SUCCESS;
    }

    if (resCtx.algorithm == BroadcastAlgorithm::LEAN_SCATTER_ALLGATHER ||
        resCtx.algorithm == BroadcastAlgorithm::OWNER_READY_SCATTER_ALLGATHER) {
        const bool validLeanRank = param.rankSize == 12 ||
            (V05B_ENABLE_RANK16_LEAN_SAG && param.rankSize == 16);
        CHK_PRT_RET(!validLeanRank || kernelGroups.empty() ||
            kernelGroups.size() != channelsPerKernel.size() ||
            resCtx.peersPerKernel.size() != channelsPerKernel.size(),
            HCCL_ERROR("Lean-SAG rank/group shape is invalid for rank size %u", param.rankSize),
            HCCL_E_INTERNAL);
        resCtx.ccuKernels.clear();
        resCtx.allGatherKernels.clear();
        const bool ownerReady = resCtx.algorithm == BroadcastAlgorithm::OWNER_READY_SCATTER_ALLGATHER;
        const char *scatterName = ownerReady ? "OwnerScatter" : "LeanScatter";
        const char *allGatherName = ownerReady ? "OwnerAllGather" : "LeanAllGather";
        CHK_RET(RegisterKernelGroup(insHandle, param, kernelGroups, channelsPerKernel, scatterName,
            reinterpret_cast<const void *>(ops_hccl::CcuLeanScatterBroadcastKernel), resCtx.ccuKernels,
            nullptr, &resCtx.peersPerKernel));
        CHK_RET(RegisterKernelGroup(insHandle, param, kernelGroups, channelsPerKernel, allGatherName,
            reinterpret_cast<const void *>(ops_hccl::CcuLeanAllGatherBroadcastKernel), resCtx.allGatherKernels,
            nullptr, &resCtx.peersPerKernel));
        return HCCL_SUCCESS;
    }

    CHK_RET(RegisterKernelGroup(insHandle, param, kernelGroups, channelsPerKernel, "Scatter",
        reinterpret_cast<const void *>(ops_hccl::CcuScatterBroadcastKernel), resCtx.ccuKernels));
    CHK_RET(RegisterKernelGroup(insHandle, param, kernelGroups, channelsPerKernel, "AllGather",
        reinterpret_cast<const void *>(ops_hccl::CcuAllGatherBroadcastKernel), resCtx.allGatherKernels));
    return HCCL_SUCCESS;
}

HcclResult BuildResources(HcclComm comm, CommEngine engine, const OpParam &param,
    BroadcastAlgorithm algorithm, AlgResourceCtx &resCtx)
{
    std::vector<uint32_t> peerRanks;
    const bool useDirectStar = algorithm == BroadcastAlgorithm::ASYMMETRIC_DIRECT ||
        algorithm == BroadcastAlgorithm::ASYMMETRIC_DIRECT_512K_2ARG ||
        algorithm == BroadcastAlgorithm::ASYMMETRIC_DIRECT_512K_REGISTERED ||
        algorithm == BroadcastAlgorithm::ASYMMETRIC_PULL_512K_2ARG ||
        (algorithm == BroadcastAlgorithm::DIRECT && V05B_ENABLE_DIRECT_STAR);
    if (useDirectStar && param.myRank != param.root) {
        peerRanks.push_back(param.root);
    } else {
        peerRanks.reserve(param.rankSize - 1);
        for (uint32_t peerRank = 0; peerRank < param.rankSize; peerRank++) {
            if (peerRank != param.myRank) {
                peerRanks.push_back(peerRank);
            }
        }
    }
    const uint32_t expectedPeerCount = useDirectStar && param.myRank != param.root ? 1 : param.rankSize - 1;
    CHK_PRT_RET(peerRanks.size() != expectedPeerCount,
        HCCL_ERROR("Resource peer count %zu does not match expected %u", peerRanks.size(), expectedPeerCount),
        HCCL_E_INTERNAL);

    std::map<KernelGroupKey, std::vector<LinkCandidate>> selectedByGroup;
    CHK_RET(SelectOneLinkPerPeer(comm, param, peerRanks, selectedByGroup));

    std::vector<KernelGroupKey> kernelGroups;
    std::vector<std::vector<ChannelHandle>> channelsPerKernel;
    CHK_RET(AcquireChannels(
        comm, engine, param, selectedByGroup, kernelGroups, channelsPerKernel, resCtx.peersPerKernel,
        DEFAULT_CHANNEL_NOTIFY_NUM));

    if (algorithm == BroadcastAlgorithm::OWNER_READY_SCATTER_ALLGATHER) {
        CHK_RET(PromoteRootChannelGroup(param, kernelGroups, channelsPerKernel, resCtx.peersPerKernel));
    }

    const uint32_t threadNum = channelsPerKernel.size();
    resCtx.threads.resize(threadNum);
    resCtx.threads[0] = param.cpuThread;
    if (threadNum > 1) {
        const uint32_t notifyNumPerThread =
            algorithm == BroadcastAlgorithm::OWNER_READY_SCATTER_ALLGATHER ? 2 : 1;
        CHK_RET(HcclThreadAcquire(
            comm, engine, threadNum - 1, notifyNumPerThread, &resCtx.threads[1]));
    }

    resCtx.algorithm = algorithm;
    if (algorithm == BroadcastAlgorithm::ASYMMETRIC_DIRECT_512K_REGISTERED) {
        resCtx.registeredBaseAddr = reinterpret_cast<uint64_t>(param.inputPtr);
        CHK_RET_CCU(HcommCcuGetMemToken(
            resCtx.registeredBaseAddr, ASYMMETRIC_DIRECT_512K_SIZE, &resCtx.registeredToken));
    }
    CHK_RET(RegisterKernels(comm, param, kernelGroups, channelsPerKernel, resCtx));
    return HCCL_SUCCESS;
}

HcclResult BuildChainResources(
    HcclComm comm, CommEngine engine, const OpParam &param, BroadcastAlgorithm algorithm,
    uint32_t channelNotifyNum, AlgResourceCtx &resCtx)
{
    CHK_PRT_RET(param.rankSize != 4,
        HCCL_ERROR("Pipelined chain is only enabled for the 4-rank contest topology"), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(algorithm != BroadcastAlgorithm::PIPELINED_CHAIN &&
        algorithm != BroadcastAlgorithm::SEGMENT_CUT_THROUGH_CHAIN &&
        algorithm != BroadcastAlgorithm::QUAD4_ROLLING2_FINAL5_CHAIN &&
        algorithm != BroadcastAlgorithm::PERSISTENT_ROLLING2_400 &&
        algorithm != BroadcastAlgorithm::PERSISTENT_ROLLING2_512,
        HCCL_ERROR("Unsupported linear chain algorithm %u", static_cast<uint32_t>(algorithm)),
        HCCL_E_INTERNAL);

    const uint32_t virtualRank = (param.myRank + param.rankSize - param.root) % param.rankSize;
    const uint32_t hasPrev = virtualRank == 0 ? 0 : 1;
    const uint32_t hasNext = virtualRank + 1 == param.rankSize ? 0 : 1;

    std::vector<uint32_t> peerRanks;
    peerRanks.reserve(hasPrev + hasNext);
    const uint32_t prevRank = (param.myRank + param.rankSize - 1) % param.rankSize;
    const uint32_t nextRank = (param.myRank + 1) % param.rankSize;
    if (hasPrev != 0) {
        peerRanks.push_back(prevRank);
    }
    if (hasNext != 0) {
        peerRanks.push_back(nextRank);
    }

    std::map<KernelGroupKey, std::vector<LinkCandidate>> selectedByGroup;
    CHK_RET(SelectOneLinkPerPeer(comm, param, peerRanks, selectedByGroup));
    CHK_PRT_RET(selectedByGroup.size() != 1,
        HCCL_ERROR("Chain peers span %zu CCU groups; one group is required", selectedByGroup.size()),
        HCCL_E_NOT_SUPPORT);

    std::vector<KernelGroupKey> kernelGroups;
    std::vector<std::vector<ChannelHandle>> channelsPerKernel;
    CHK_RET(AcquireChannels(
        comm, engine, param, selectedByGroup, kernelGroups, channelsPerKernel, resCtx.peersPerKernel,
        channelNotifyNum));
    const uint32_t expectedChannelCount = hasPrev + hasNext;
    CHK_PRT_RET(channelsPerKernel.size() != 1 || resCtx.peersPerKernel.size() != 1 ||
        channelsPerKernel[0].size() != expectedChannelCount ||
        resCtx.peersPerKernel[0].size() != expectedChannelCount,
        HCCL_ERROR("Chain channel grouping is invalid"), HCCL_E_INTERNAL);

    ChainRole chainRole{};
    chainRole.hasPrev = hasPrev;
    chainRole.hasNext = hasNext;
    bool prevFound = hasPrev == 0;
    bool nextFound = hasNext == 0;
    for (uint32_t channelIdx = 0; channelIdx < resCtx.peersPerKernel[0].size(); channelIdx++) {
        const uint32_t peerRank = resCtx.peersPerKernel[0][channelIdx];
        if (hasPrev != 0 && peerRank == prevRank) {
            chainRole.prevChannelIdx = channelIdx;
            prevFound = true;
        }
        if (hasNext != 0 && peerRank == nextRank) {
            chainRole.nextChannelIdx = channelIdx;
            nextFound = true;
        }
    }
    CHK_PRT_RET(!prevFound || !nextFound,
        HCCL_ERROR("Failed to map chain channels for rank %u", param.myRank), HCCL_E_INTERNAL);

    resCtx.algorithm = algorithm;
    resCtx.threads = {param.cpuThread};
    CHK_RET(RegisterKernels(comm, param, kernelGroups, channelsPerKernel, resCtx, &chainRole));
    return HCCL_SUCCESS;
}

BroadcastAlgorithm SelectAlgorithm(uint64_t dataSize, uint64_t count, uint64_t typeSize, uint32_t rankSize)
{
    if (V07A_ENABLE_ASYM_PUSH_DIRECT && typeSize == sizeof(float) &&
        dataSize == ASYMMETRIC_DIRECT_512K_SIZE &&
        (rankSize == 4 || rankSize == 12 || rankSize == 16)) {
        return rankSize == 4 ? BroadcastAlgorithm::ASYMMETRIC_PULL_512K_2ARG
                             : BroadcastAlgorithm::ASYMMETRIC_DIRECT_512K_REGISTERED;
    }

    if (dataSize < SCATTER_ALLGATHER_THRESHOLD) {
        return V07A_ENABLE_ASYM_PUSH_DIRECT ? BroadcastAlgorithm::ASYMMETRIC_DIRECT
                                           : BroadcastAlgorithm::DIRECT;
    }

    const bool useLeanSag = rankSize == 12 || (V05B_ENABLE_RANK16_LEAN_SAG && rankSize == 16);
    if (useLeanSag && typeSize == sizeof(float) &&
        (dataSize == CUT_THROUGH_SIZE_512M || dataSize == CUT_THROUGH_SIZE_400M_PLUS_4)) {
        return V07A_ENABLE_OWNER_READY_SAG ? BroadcastAlgorithm::OWNER_READY_SCATTER_ALLGATHER
                                          : BroadcastAlgorithm::LEAN_SCATTER_ALLGATHER;
    }

    if (rankSize == 4 && typeSize == sizeof(float) &&
        dataSize == CUT_THROUGH_SIZE_400M_PLUS_4 && V13_ENABLE_PERSISTENT_ROLLING2_400) {
        return BroadcastAlgorithm::PERSISTENT_ROLLING2_400;
    }

    if (rankSize == 4 && typeSize == sizeof(float) &&
        dataSize == CUT_THROUGH_SIZE_512M && V18A_ENABLE_PERSISTENT_ROLLING2_512) {
        return BroadcastAlgorithm::PERSISTENT_ROLLING2_512;
    }

    if (rankSize == 4 && typeSize == sizeof(float) &&
        (dataSize == CUT_THROUGH_SIZE_512M || dataSize == CUT_THROUGH_SIZE_400M_PLUS_4)) {
        return V12A_ENABLE_FINAL5_DRAIN_FUSION ? BroadcastAlgorithm::QUAD4_ROLLING2_FINAL5_CHAIN
                                               : BroadcastAlgorithm::SEGMENT_CUT_THROUGH_CHAIN;
    }

    if (rankSize == 4) {
        return BroadcastAlgorithm::PIPELINED_CHAIN;
    }

    const uint64_t maxSliceCount = count / rankSize + ((count % rankSize == 0) ? 0 : 1);
    const uint64_t maxSliceSize = maxSliceCount * typeSize;
    return maxSliceSize < MEM2MEM_MAX_SIZE ? BroadcastAlgorithm::SCATTER_ALLGATHER
                                          : BroadcastAlgorithm::DIRECT;
}
} // namespace

HcclResult HcclBroadcast(
    void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(buf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    // 构造算子参数
    OpParam param{};
    param.inputPtr = buf;
    param.outputPtr = buf;
    param.count = count;
    param.dataType = dataType;
    param.root = root;
    param.opType = HcclCMDType::HCCL_CMD_BROADCAST;

    CHK_PRT_RET(SIZE_TABLE.find(dataType) == SIZE_TABLE.end(), HCCL_ERROR("Unsupported data type %d", dataType),
        HCCL_E_NOT_SUPPORT);
    const uint64_t typeSize = SIZE_TABLE.at(dataType);
    CHK_PRT_RET(count > std::numeric_limits<uint64_t>::max() / typeSize,
        HCCL_ERROR("Broadcast data size overflows uint64"), HCCL_E_PARA);
    const uint64_t dataSize = count * typeSize;

    CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;
    bool smallFastContextMissing = false;
    const bool smallFastCandidate = dataSize == ASYMMETRIC_DIRECT_512K_SIZE &&
        typeSize == sizeof(float);
    if (smallFastCandidate) {
        const int fastTagRet = sprintf_s(param.tag, sizeof(param.tag),
            "hccl_custom_broadcast_ccu_v18_hybrid_512k_f%u_b%u_r%u",
            V07A_FEATURE_MASK, V05B_FEATURE_MASK, root);
        CHK_PRT_RET(fastTagRet <= 0,
                HCCL_ERROR("Failed to build v0.18 hybrid fast operator tag"), HCCL_E_INTERNAL);

        void *fastCtx = nullptr;
        uint64_t fastCtxSize = 0;
        if (HcclEngineCtxGet(comm, param.tag, ccuEngine, &fastCtx, &fastCtxSize) == HCCL_SUCCESS) {
            CHK_PRT_RET(fastCtx == nullptr || fastCtxSize < sizeof(SmallFastContextHeader),
                HCCL_ERROR("v0.16B fast context is truncated, size %llu",
                    static_cast<unsigned long long>(fastCtxSize)), HCCL_E_INTERNAL);

            SmallFastContextHeader fastHeader{};
            std::memcpy(&fastHeader, fastCtx, sizeof(fastHeader));
            const bool validRankSize = fastHeader.rankSize == 4 || fastHeader.rankSize == 12 ||
                fastHeader.rankSize == 16;
            const bool validHybridAlgorithm =
                (fastHeader.rankSize == 4 && fastHeader.algorithm == static_cast<uint32_t>(
                    BroadcastAlgorithm::ASYMMETRIC_PULL_512K_2ARG)) ||
                ((fastHeader.rankSize == 12 || fastHeader.rankSize == 16) &&
                    fastHeader.algorithm == static_cast<uint32_t>(
                        BroadcastAlgorithm::ASYMMETRIC_DIRECT_512K_REGISTERED));
            CHK_PRT_RET(fastHeader.magic != V18_SMALL_FAST_CONTEXT_MAGIC ||
                fastHeader.version != V18_SMALL_FAST_CONTEXT_VERSION || !validHybridAlgorithm ||
                !validRankSize || fastHeader.rankId >= fastHeader.rankSize ||
                fastHeader.mainThread == 0,
                HCCL_ERROR("v0.18 hybrid fast context header is invalid"), HCCL_E_INTERNAL);
            CHK_PRT_RET(root >= fastHeader.rankSize,
                HCCL_ERROR("Invalid root %u for cached rank size %u", root, fastHeader.rankSize),
                HCCL_E_PARA);

            param.myRank = fastHeader.rankId;
            param.rankSize = fastHeader.rankSize;
            param.resCtx = fastCtx;
            param.ctxSize = fastCtxSize;
            const uint64_t currentStreamKey = reinterpret_cast<uint64_t>(stream);
            if (currentStreamKey == fastHeader.streamKey) {
                param.cpuThread = fastHeader.mainThread;
            } else {
                CHK_RET(HcclThreadAcquireWithStream(
                    comm, ccuEngine, stream, 1, &param.cpuThread));
            }
            return ops_hccl::ExecOp(param);
        }
        smallFastContextMissing = true;
    }

    // ==============================================
    // STEP 1: 解析拓扑信息
    // ==============================================
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE,
        HCCL_ERROR("Unsupported rank size %u", param.rankSize), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(root >= param.rankSize, HCCL_ERROR("Invalid root %u for rank size %u", root, param.rankSize),
        HCCL_E_PARA);

    const BroadcastAlgorithm algorithm = SelectAlgorithm(dataSize, count, typeSize, param.rankSize);

    const char *algorithmTag = "sag";
    if (algorithm == BroadcastAlgorithm::DIRECT) {
        algorithmTag = "direct";
    } else if (algorithm == BroadcastAlgorithm::ASYMMETRIC_DIRECT) {
        algorithmTag = "asym_direct";
    } else if (algorithm == BroadcastAlgorithm::ASYMMETRIC_DIRECT_512K_2ARG) {
        algorithmTag = "asym_512k_2arg";
    } else if (algorithm == BroadcastAlgorithm::ASYMMETRIC_DIRECT_512K_REGISTERED) {
        algorithmTag = "asym_512k_registered";
    } else if (algorithm == BroadcastAlgorithm::ASYMMETRIC_PULL_512K_2ARG) {
        algorithmTag = "asym_512k_pull";
    } else if (algorithm == BroadcastAlgorithm::PIPELINED_CHAIN) {
        algorithmTag = "chain";
    } else if (algorithm == BroadcastAlgorithm::SEGMENT_CUT_THROUGH_CHAIN) {
        algorithmTag = "cut";
    } else if (algorithm == BroadcastAlgorithm::QUAD4_ROLLING2_FINAL5_CHAIN) {
        algorithmTag = "quad4_final5";
    } else if (algorithm == BroadcastAlgorithm::PERSISTENT_ROLLING2_400) {
        algorithmTag = "persistent_r2_400_8m";
    } else if (algorithm == BroadcastAlgorithm::PERSISTENT_ROLLING2_512) {
        algorithmTag = "persistent_r2_512_8m";
    } else if (algorithm == BroadcastAlgorithm::LEAN_SCATTER_ALLGATHER) {
        algorithmTag = "lean_sag";
    } else if (algorithm == BroadcastAlgorithm::OWNER_READY_SCATTER_ALLGATHER) {
        algorithmTag = "owner_sag";
    }
    const bool usesV07AResources = algorithm == BroadcastAlgorithm::ASYMMETRIC_DIRECT ||
        algorithm == BroadcastAlgorithm::ASYMMETRIC_DIRECT_512K_2ARG ||
        algorithm == BroadcastAlgorithm::ASYMMETRIC_DIRECT_512K_REGISTERED ||
        algorithm == BroadcastAlgorithm::ASYMMETRIC_PULL_512K_2ARG ||
        algorithm == BroadcastAlgorithm::OWNER_READY_SCATTER_ALLGATHER;
    const bool usesV05BResources = algorithm == BroadcastAlgorithm::LEAN_SCATTER_ALLGATHER ||
        (algorithm == BroadcastAlgorithm::DIRECT && V05B_ENABLE_DIRECT_STAR);
    const bool usesV12AResources = algorithm == BroadcastAlgorithm::QUAD4_ROLLING2_FINAL5_CHAIN;
    const bool usesV13Resources = algorithm == BroadcastAlgorithm::PERSISTENT_ROLLING2_400;
    const bool usesV18APersistentResources = algorithm == BroadcastAlgorithm::PERSISTENT_ROLLING2_512;
    const char *versionTag = algorithm == BroadcastAlgorithm::ASYMMETRIC_PULL_512K_2ARG ? "v16b" :
        (algorithm == BroadcastAlgorithm::ASYMMETRIC_DIRECT_512K_REGISTERED ? "v16a" :
        (algorithm == BroadcastAlgorithm::ASYMMETRIC_DIRECT_512K_2ARG ? "v14a" :
        (usesV18APersistentResources ? "v18a" : (usesV13Resources ? "v13b" :
        (usesV12AResources ? "v12a" :
        (algorithm == BroadcastAlgorithm::SEGMENT_CUT_THROUGH_CHAIN ? "v03f" :
        (usesV07AResources ? "v07a" : (usesV05BResources ? "v05b" : "v03e"))))))));
    int tagRet = 0;
    const bool hybridSmallAlgorithm = algorithm == BroadcastAlgorithm::ASYMMETRIC_PULL_512K_2ARG ||
        algorithm == BroadcastAlgorithm::ASYMMETRIC_DIRECT_512K_REGISTERED;
    if (smallFastCandidate && hybridSmallAlgorithm) {
        tagRet = sprintf_s(param.tag, sizeof(param.tag),
            "hccl_custom_broadcast_ccu_v18_hybrid_512k_f%u_b%u_r%u",
            V07A_FEATURE_MASK, V05B_FEATURE_MASK, root);
    } else if (usesV07AResources) {
        tagRet = sprintf_s(param.tag, sizeof(param.tag),
            "hccl_custom_broadcast_ccu_%s_%s_f%u_b%u_r%u", versionTag, algorithmTag,
            V07A_FEATURE_MASK, V05B_FEATURE_MASK, root);
    } else if (usesV05BResources) {
        tagRet = sprintf_s(param.tag, sizeof(param.tag),
            "hccl_custom_broadcast_ccu_%s_%s_f%u_r%u", versionTag, algorithmTag, V05B_FEATURE_MASK, root);
    } else {
        tagRet = sprintf_s(param.tag, sizeof(param.tag),
            "hccl_custom_broadcast_ccu_%s_%s_r%u", versionTag, algorithmTag, root);
    }
    CHK_PRT_RET(tagRet <= 0, HCCL_ERROR("Failed to build operator tag"), HCCL_E_INTERNAL);

    if (count == 0 || param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    // ==============================================
    // STEP 2: 创建资源
    // ==============================================
    // ==============================================
    // STEP 2.1: 申请用于 Host/Device 同步的通信资源
    // ==============================================
    // 将用户传入的 stream 转换为 CCU 通信引擎中的 thread，并申请 1 个 notify
    CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, stream, 1, &param.cpuThread));

    void *ctx = nullptr;
    uint64_t size = 0;
    bool resourceCreated = false;
    const bool fastMissForSelectedAlgorithm = smallFastContextMissing &&
        hybridSmallAlgorithm;
    if (!fastMissForSelectedAlgorithm &&
        HcclEngineCtxGet(comm, param.tag, ccuEngine, &ctx, &size) == HCCL_SUCCESS) {
        // CCU 资源已经存在，复用资源
        HCCL_INFO("Engine context already exists");
        param.resCtx = ctx;
        param.ctxSize = size;
    } else {
        // Device 资源不存在，资源构建
        AlgResourceCtx resCtxHost;
        if (algorithm == BroadcastAlgorithm::PERSISTENT_ROLLING2_400) {
            CHK_RET(BuildChainResources(comm, ccuEngine, param, algorithm,
                V13_PERSISTENT_CHANNEL_NOTIFY_NUM, resCtxHost));
        } else if (algorithm == BroadcastAlgorithm::PERSISTENT_ROLLING2_512) {
            CHK_RET(BuildChainResources(comm, ccuEngine, param, algorithm,
                V18A_PERSISTENT_512_CHANNEL_NOTIFY_NUM, resCtxHost));
        } else if (algorithm == BroadcastAlgorithm::SEGMENT_CUT_THROUGH_CHAIN ||
            algorithm == BroadcastAlgorithm::QUAD4_ROLLING2_FINAL5_CHAIN) {
            CHK_RET(BuildChainResources(comm, ccuEngine, param, algorithm,
                CUT_THROUGH_CHANNEL_NOTIFY_NUM, resCtxHost));
        } else if (algorithm == BroadcastAlgorithm::PIPELINED_CHAIN) {
            CHK_RET(BuildChainResources(comm, ccuEngine, param, algorithm,
                DEFAULT_CHANNEL_NOTIFY_NUM, resCtxHost));
        } else {
            CHK_RET(BuildResources(comm, ccuEngine, param, algorithm, resCtxHost));
        }

        if (hybridSmallAlgorithm) {
            resCtxHost.smallFastHeader.magic = V18_SMALL_FAST_CONTEXT_MAGIC;
            resCtxHost.smallFastHeader.version = V18_SMALL_FAST_CONTEXT_VERSION;
            resCtxHost.smallFastHeader.algorithm = static_cast<uint32_t>(algorithm);
            resCtxHost.smallFastHeader.rankId = param.myRank;
            resCtxHost.smallFastHeader.rankSize = param.rankSize;
            resCtxHost.smallFastHeader.streamKey = reinterpret_cast<uint64_t>(stream);
            resCtxHost.smallFastHeader.mainThread = param.cpuThread;
        }

        // ==============================================
        // STEP 2.3: 申请通信引擎上下文
        // ==============================================
        // 申请 CCU 通信引擎上下文，存放 AlgResourceCtx 信息
        std::vector<char> seq = resCtxHost.Serialize();
        uint64_t seqSize = seq.size();
        param.ctxSize = seqSize;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, ccuEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, ccuEngine, param.tag, seq.data(), seqSize, 0));
        resourceCreated = true;
    }

    // Preserve a valid DFX object on the first small-message invocation, but
    // avoid recreating the same profiling state on every steady reuse of the
    // Engine Context.  All other algorithms retain the v0.14B1 behavior.
    if (resourceCreated || algorithm != BroadcastAlgorithm::ASYMMETRIC_DIRECT_512K_REGISTERED) {
        HcclDfxOpInfo dfxInfo{};
        char commName[COMM_INDENTIFIER_MAX_LENGTH];
        CHK_RET(HcclGetCommName(comm, commName));
        CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));
    }

    // ==============================================
    // STEP 3: 下发 CCU Kernel
    // ==============================================
    CHK_RET(ops_hccl::ExecOp(param));
    return HCCL_SUCCESS;
}
