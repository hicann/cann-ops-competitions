/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <ccu/ccu_res.h>
#include <hcomm/ccu/ccu_launch.h>
#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "ccu_kernel.h"
#include "common.h"
#include "custom.h"
#include "exec_op.h"
#include "hccl.h"
#include "log.h"

namespace {
constexpr char DIRECT_CONTEXT_TAG[] = "hccl_bcast_oneshot_noack_v6_3";
constexpr char HIERARCHICAL_CONTEXT_TAG[] = "hccl_bcast_endpoint_order_v6_2";
constexpr uint32_t CHANNEL_NOTIFY_NUM = 2;
constexpr uint32_t DIE_COUNT = 2;
constexpr uint64_t SMALL_MESSAGE_BYTES = 512ULL * 1024ULL;

struct LinkChoice {
    CommLink link{};
    uint32_t die = DIE_COUNT;
    uint32_t pass = 0;
};

struct HierarchicalPlan {
    std::vector<uint32_t> sourceGroup;
    std::vector<uint32_t> destinationGroup;
};

bool Contains(const std::vector<uint32_t> &ranks, uint32_t rank)
{
    return std::find(ranks.begin(), ranks.end(), rank) != ranks.end();
}

uint32_t RankIndex(const std::vector<uint32_t> &ranks, uint32_t rank)
{
    return static_cast<uint32_t>(std::distance(ranks.begin(), std::find(ranks.begin(), ranks.end(), rank)));
}

HcclResult GetLayers(HcclComm comm, std::vector<uint32_t> &layers)
{
    uint32_t *layerData = nullptr;
    uint32_t layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layerData, &layerCount));
    layers.assign(layerData, layerData + layerCount);
    CHK_PRT_RET(layers.empty(), HCCL_ERROR("rank graph has no layers"), HCCL_E_NOT_FOUND);
    return HCCL_SUCCESS;
}

HcclResult BuildHierarchicalPlan(HcclComm comm, const OpParam &param, HierarchicalPlan &plan)
{
    if (param.rankSize == 4) {
        plan.sourceGroup = {0, 1, 2, 3};
        return HCCL_SUCCESS;
    }
    std::vector<uint32_t> layers;
    CHK_RET(GetLayers(comm, layers));
    std::vector<uint32_t> local;
    for (uint32_t layer : layers) {
        uint32_t *ranks = nullptr;
        uint32_t rankCount = 0;
        CHK_RET(HcclRankGraphGetRanksByLayer(comm, layer, &ranks, &rankCount));
        if (rankCount > local.size() && rankCount > 1 && rankCount < param.rankSize) {
            local.assign(ranks, ranks + rankCount);
        }
    }
    CHK_PRT_RET(local.empty(), HCCL_ERROR("no server-local group for hierarchical broadcast"), HCCL_E_NOT_SUPPORT);
    std::sort(local.begin(), local.end());
    std::vector<uint32_t> other;
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (!Contains(local, rank)) {
            other.push_back(rank);
        }
    }
    CHK_PRT_RET(other.empty() || local.size() > 8 || other.size() > 8,
        HCCL_ERROR("unsupported hierarchical groups local=%zu remote=%zu", local.size(), other.size()),
        HCCL_E_NOT_SUPPORT);
    if (Contains(local, param.root)) {
        plan.sourceGroup = local;
        plan.destinationGroup = other;
    } else {
        plan.sourceGroup = other;
        plan.destinationGroup = local;
    }
    return HCCL_SUCCESS;
}

void GetChunkRange(uint64_t dataBytes, uint32_t groupSize, uint32_t index, uint64_t &offset, uint64_t &length)
{
    const uint64_t elementCount = dataBytes / sizeof(float);
    const uint64_t normalElements = elementCount / groupSize;
    offset = static_cast<uint64_t>(index) * normalElements * sizeof(float);
    length = index + 1U == groupSize ? dataBytes - offset : normalElements * sizeof(float);
}

HcclResult CollectChoices(HcclComm comm, uint32_t myRank, uint32_t peer, const std::vector<uint32_t> &layers,
    bool preferClos, std::vector<LinkChoice> &choices)
{
    for (uint32_t pass = 0; pass < 2; ++pass) {
        for (uint32_t layer : layers) {
            CommTopo topology = COMM_TOPO_RESERVED;
            CHK_RET(HcclRankGraphGetTopoTypeByLayer(comm, layer, &topology));
            const bool isPreferred = (topology == COMM_TOPO_CLOS) == preferClos;
            if ((pass == 0) != isPreferred) {
                continue;
            }
            CommLink *links = nullptr;
            uint32_t linkCount = 0;
            CHK_RET(HcclRankGraphGetLinks(comm, layer, myRank, peer, &links, &linkCount));
            for (uint32_t index = 0; index < linkCount; ++index) {
                const CommProtocol protocol = links[index].linkAttr.linkProtocol;
                if (protocol != COMM_PROTOCOL_UBC_CTP && protocol != COMM_PROTOCOL_UBC_TP) {
                    continue;
                }
                EndpointAttrDieId die = DIE_COUNT;
                CHK_RET(HcclRankGraphGetEndpointInfo(
                    comm, myRank, &links[index].srcEndpointDesc, ENDPOINT_ATTR_DIE_ID, sizeof(die), &die));
                if (die < DIE_COUNT) {
                    choices.push_back(LinkChoice{links[index], die, pass});
                }
            }
        }
    }
    CHK_PRT_RET(choices.empty(), HCCL_ERROR("no CCU link from rank %u to root %u", myRank, peer), HCCL_E_NOT_FOUND);
    return HCCL_SUCCESS;
}

int CompareEndpoints(const EndpointDesc &left, const EndpointDesc &right)
{
    if (left.protocol != right.protocol) {
        return left.protocol < right.protocol ? -1 : 1;
    }
    const int addressOrder = std::memcmp(&left.commAddr, &right.commAddr, sizeof(CommAddr));
    if (addressOrder != 0) {
        return addressOrder;
    }
    return std::memcmp(&left.loc, &right.loc, sizeof(EndpointLoc));
}

bool RootLinkLess(const LinkChoice &left, const LinkChoice &right, bool isRoot)
{
    const EndpointDesc &leftRoot = isRoot ? left.link.srcEndpointDesc : left.link.dstEndpointDesc;
    const EndpointDesc &rightRoot = isRoot ? right.link.srcEndpointDesc : right.link.dstEndpointDesc;
    const int rootOrder = CompareEndpoints(leftRoot, rightRoot);
    if (rootOrder != 0) {
        return rootOrder < 0;
    }
    const EndpointDesc &leftPeer = isRoot ? left.link.dstEndpointDesc : left.link.srcEndpointDesc;
    const EndpointDesc &rightPeer = isRoot ? right.link.dstEndpointDesc : right.link.srcEndpointDesc;
    return CompareEndpoints(leftPeer, rightPeer) < 0;
}

const LinkChoice &SelectRootAlignedChoice(const OpParam &param, const std::vector<LinkChoice> &choices)
{
    const LinkChoice *selected = &choices.front();
    if (param.rankSize != MAX_RANK_SIZE) {
        return *selected;
    }
    const bool isRoot = param.myRank == param.root;
    const uint32_t preferredPass = choices.front().pass;
    for (const LinkChoice &choice : choices) {
        if (choice.pass == preferredPass && RootLinkLess(choice, *selected, isRoot)) {
            selected = &choice;
        }
    }
    return *selected;
}

HcclResult AcquireDirectChannels(
    HcclComm comm, const OpParam &param, std::vector<uint32_t> &channelDies, std::vector<ChannelHandle> &channels)
{
    std::vector<uint32_t> peers;
    if (param.myRank == param.root) {
        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            if (rank != param.root) {
                peers.push_back(rank);
            }
        }
    } else {
        peers.push_back(param.root);
    }
    CHK_PRT_RET(peers.empty() || peers.size() > BROADCAST_MAX_PEERS,
        HCCL_ERROR("invalid direct broadcast peer count %zu", peers.size()), HCCL_E_PARA);

    std::vector<uint32_t> layers;
    CHK_RET(GetLayers(comm, layers));
    const uint64_t dataBytes = param.count * sizeof(float);
    const bool topologyAwareSmallPath = param.rankSize == MAX_RANK_SIZE && dataBytes <= SMALL_MESSAGE_BYTES;
    std::vector<uint32_t> localRanks;
    if (topologyAwareSmallPath) {
        for (uint32_t layer : layers) {
            uint32_t *ranks = nullptr;
            uint32_t rankCount = 0;
            CHK_RET(HcclRankGraphGetRanksByLayer(comm, layer, &ranks, &rankCount));
            if (rankCount > localRanks.size() && rankCount > 1 && rankCount < param.rankSize) {
                localRanks.assign(ranks, ranks + rankCount);
            }
        }
        CHK_PRT_RET(
            localRanks.empty(), HCCL_ERROR("no server-local group for 2x8 direct broadcast"), HCCL_E_NOT_SUPPORT);
    }
    std::vector<std::vector<LinkChoice>> choices(peers.size());
    for (size_t index = 0; index < peers.size(); ++index) {
        const bool preferClos = !topologyAwareSmallPath || !Contains(localRanks, peers[index]);
        CHK_RET(CollectChoices(comm, param.myRank, peers[index], layers, preferClos, choices[index]));
    }

    std::vector<HcclChannelDesc> descs(peers.size());
    channelDies.resize(peers.size());
    for (size_t index = 0; index < peers.size(); ++index) {
        const LinkChoice &selected = SelectRootAlignedChoice(param, choices[index]);
        CHK_RET(HcclChannelDescInit(&descs[index], 1));
        descs[index].remoteRank = peers[index];
        descs[index].notifyNum = CHANNEL_NOTIFY_NUM;
        descs[index].channelProtocol = selected.link.linkAttr.linkProtocol;
        descs[index].localEndpoint = selected.link.srcEndpointDesc;
        descs[index].remoteEndpoint = selected.link.dstEndpointDesc;
        channelDies[index] = selected.die;
    }
    channels.resize(peers.size());
    return HcclChannelAcquire(
        comm, COMM_ENGINE_CCU, descs.data(), static_cast<uint32_t>(descs.size()), channels.data());
}

void AddPeer(std::vector<uint32_t> &peers, uint32_t peer, uint32_t myRank)
{
    if (peer != myRank && !Contains(peers, peer)) {
        peers.push_back(peer);
    }
}

HcclResult SelectCommonDie(
    const std::vector<std::vector<LinkChoice>> &choices, const std::vector<size_t> &indices, uint32_t &die)
{
    if (indices.empty()) {
        die = 0;
        return HCCL_SUCCESS;
    }
    for (uint32_t candidate = 0; candidate < DIE_COUNT; ++candidate) {
        const bool matches = std::all_of(indices.begin(), indices.end(), [&choices, candidate](size_t index) {
            return std::any_of(choices[index].begin(), choices[index].end(), [candidate](const LinkChoice &choice) {
                return choice.die == candidate;
            });
        });
        if (matches) {
            die = candidate;
            return HCCL_SUCCESS;
        }
    }
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AcquireHierarchicalChannels(HcclComm comm, const OpParam &param, const HierarchicalPlan &plan,
    std::vector<uint32_t> &peers, std::vector<uint32_t> &channelDies, std::vector<ChannelHandle> &channels)
{
    const bool inSource = Contains(plan.sourceGroup, param.myRank);
    const std::vector<uint32_t> &localGroup = inSource ? plan.sourceGroup : plan.destinationGroup;
    std::vector<uint32_t> localPeers;
    for (uint32_t peer : localGroup) {
        AddPeer(localPeers, peer, param.myRank);
    }
    std::vector<uint32_t> crossPeers;
    const uint64_t dataBytes = param.count * sizeof(float);
    if (inSource && !plan.destinationGroup.empty()) {
        const uint32_t sourceIndex = RankIndex(plan.sourceGroup, param.myRank);
        uint64_t sourceOffset = 0;
        uint64_t sourceLength = 0;
        GetChunkRange(
            dataBytes, static_cast<uint32_t>(plan.sourceGroup.size()), sourceIndex, sourceOffset, sourceLength);
        for (uint32_t destinationIndex = 0; destinationIndex < plan.destinationGroup.size(); ++destinationIndex) {
            uint64_t destinationOffset = 0;
            uint64_t destinationLength = 0;
            GetChunkRange(dataBytes, static_cast<uint32_t>(plan.destinationGroup.size()), destinationIndex,
                destinationOffset, destinationLength);
            if (std::max(sourceOffset, destinationOffset)
                < std::min(sourceOffset + sourceLength, destinationOffset + destinationLength)) {
                AddPeer(crossPeers, plan.destinationGroup[destinationIndex], param.myRank);
            }
        }
    } else if (!inSource) {
        const uint32_t destinationIndex = RankIndex(plan.destinationGroup, param.myRank);
        uint64_t destinationOffset = 0;
        uint64_t destinationLength = 0;
        GetChunkRange(dataBytes, static_cast<uint32_t>(plan.destinationGroup.size()), destinationIndex,
            destinationOffset, destinationLength);
        for (uint32_t sourceIndex = 0; sourceIndex < plan.sourceGroup.size(); ++sourceIndex) {
            uint64_t sourceOffset = 0;
            uint64_t sourceLength = 0;
            GetChunkRange(
                dataBytes, static_cast<uint32_t>(plan.sourceGroup.size()), sourceIndex, sourceOffset, sourceLength);
            if (std::max(sourceOffset, destinationOffset)
                < std::min(sourceOffset + sourceLength, destinationOffset + destinationLength)) {
                AddPeer(crossPeers, plan.sourceGroup[sourceIndex], param.myRank);
            }
        }
    }
    peers = localPeers;
    for (uint32_t peer : crossPeers) {
        AddPeer(peers, peer, param.myRank);
    }
    CHK_PRT_RET(peers.empty() || peers.size() > BROADCAST_MAX_PEERS,
        HCCL_ERROR("invalid hierarchical peer count %zu", peers.size()), HCCL_E_PARA);

    std::vector<uint32_t> layers;
    CHK_RET(GetLayers(comm, layers));
    std::vector<std::vector<LinkChoice>> choices(peers.size());
    for (size_t index = 0; index < peers.size(); ++index) {
        CHK_RET(CollectChoices(comm, param.myRank, peers[index], layers, true, choices[index]));
    }
    std::vector<size_t> localIndices;
    std::vector<size_t> crossIndices;
    for (size_t index = 0; index < peers.size(); ++index) {
        if (Contains(localPeers, peers[index])) {
            localIndices.push_back(index);
        } else {
            crossIndices.push_back(index);
        }
    }
    uint32_t localDie = 0;
    uint32_t crossDie = 0;
    CHK_RET(SelectCommonDie(choices, localIndices, localDie));
    CHK_RET(SelectCommonDie(choices, crossIndices, crossDie));

    std::vector<HcclChannelDesc> descs(peers.size());
    channelDies.resize(peers.size());
    for (size_t index = 0; index < peers.size(); ++index) {
        const uint32_t requestedDie = Contains(localPeers, peers[index]) ? localDie : crossDie;
        auto selected = std::find_if(choices[index].begin(), choices[index].end(), [requestedDie](const auto &choice) {
            return choice.die == requestedDie;
        });
        CHK_PRT_RET(selected == choices[index].end(), HCCL_ERROR("missing selected endpoint die"), HCCL_E_INTERNAL);
        CHK_RET(HcclChannelDescInit(&descs[index], 1));
        descs[index].remoteRank = peers[index];
        descs[index].notifyNum = CHANNEL_NOTIFY_NUM;
        descs[index].channelProtocol = selected->link.linkAttr.linkProtocol;
        descs[index].localEndpoint = selected->link.srcEndpointDesc;
        descs[index].remoteEndpoint = selected->link.dstEndpointDesc;
        channelDies[index] = requestedDie;
    }
    channels.resize(peers.size());
    return HcclChannelAcquire(
        comm, COMM_ENGINE_CCU, descs.data(), static_cast<uint32_t>(descs.size()), channels.data());
}

HcclResult RegisterKernel(HcclComm comm, const OpParam &param, const std::vector<uint32_t> &channelDies,
    const std::vector<ChannelHandle> &channels, AlgResourceCtx &resources)
{
    CcuInsHandle instance = 0;
    uint32_t instanceCount = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &instance, &instanceCount));
    CHK_PRT_RET(instanceCount != 1, HCCL_ERROR("expected one CCU instance, got %u", instanceCount), HCCL_E_INTERNAL);

    for (uint32_t die = 0; die < DIE_COUNT; ++die) {
        BroadcastKernelArg kernelArg{};
        kernelArg.root = param.root;
        kernelArg.myRank = param.myRank;
        for (size_t index = 0; index < channels.size(); ++index) {
            if (channelDies[index] == die) {
                kernelArg.channels[kernelArg.channelCount++] = channels[index];
            }
        }
        if (kernelArg.channelCount == 0) {
            continue;
        }
        CHK_PRT_RET(resources.kernelCount >= 2, HCCL_ERROR("too many CCU die groups"), HCCL_E_INTERNAL);
        char name[64]{};
        std::snprintf(name, sizeof(name), "BroadcastDirectV1_%u_r%u_d%u", param.myRank, param.root, die);
        const void *kernelArgs[] = {&kernelArg};
        CHK_RET_CCU(HcommCcuKernelRegisterStart(instance));
        CHK_RET_CCU(
            HcommCcuKernelRegister(instance, die, name, reinterpret_cast<const void *>(ops_hccl::BroadcastKernel),
                kernelArgs, 1, &resources.kernels[resources.kernelCount]));
        CHK_RET_CCU(HcommCcuKernelRegisterEnd(instance));
        ++resources.kernelCount;
    }
    CHK_PRT_RET(resources.kernelCount == 0, HCCL_ERROR("no broadcast kernel was registered"), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult RegisterHierarchicalPhase(CcuInsHandle instance, const OpParam &param, uint32_t phase,
    BroadcastKernelMode mode, uint32_t groupSize, uint32_t myIndex, const std::vector<uint32_t> &activePeers,
    const std::vector<uint64_t> &transferOffsets, const std::vector<uint64_t> &transferLengths, uint64_t dataBytes,
    const std::vector<uint32_t> &peers, const std::vector<uint32_t> &channelDies,
    const std::vector<ChannelHandle> &channels, AlgResourceCtx &resources)
{
    CHK_PRT_RET(activePeers.size() != transferOffsets.size() || activePeers.size() != transferLengths.size(),
        HCCL_ERROR("invalid transfer metadata for phase %u", phase), HCCL_E_INTERNAL);
    HierarchicalBroadcastKernelArg kernelArg{};
    kernelArg.mode = mode;
    kernelArg.groupSize = groupSize;
    kernelArg.myIndex = myIndex;
    kernelArg.dataBytes = dataBytes;
    uint32_t die = 0;
    for (size_t activeIndex = 0; activeIndex < activePeers.size(); ++activeIndex) {
        auto found = std::find(peers.begin(), peers.end(), activePeers[activeIndex]);
        CHK_PRT_RET(
            found == peers.end(), HCCL_ERROR("missing phase peer %u", activePeers[activeIndex]), HCCL_E_INTERNAL);
        const size_t index = static_cast<size_t>(std::distance(peers.begin(), found));
        if (kernelArg.channelCount == 0) {
            die = channelDies[index];
        } else {
            CHK_PRT_RET(die != channelDies[index], HCCL_ERROR("phase peers span CCU dies"), HCCL_E_NOT_SUPPORT);
        }
        kernelArg.channels[kernelArg.channelCount] = channels[index];
        kernelArg.transferOffsets[kernelArg.channelCount] = transferOffsets[activeIndex];
        kernelArg.transferLengths[kernelArg.channelCount] = transferLengths[activeIndex];
        ++kernelArg.channelCount;
    }
    if (mode != BroadcastKernelMode::NOOP) {
        CHK_PRT_RET(kernelArg.channelCount == 0, HCCL_ERROR("invalid hierarchical phase %u", phase), HCCL_E_INTERNAL);
    }
    char name[64]{};
    std::snprintf(name, sizeof(name), "BroadcastParallelSagV4_%u_r%u_p%u", param.myRank, param.root, phase);
    const void *kernelArgs[] = {&kernelArg};
    CHK_RET_CCU(HcommCcuKernelRegisterStart(instance));
    CHK_RET_CCU(HcommCcuKernelRegister(instance, die, name,
        reinterpret_cast<const void *>(ops_hccl::HierarchicalBroadcastKernel), kernelArgs, 1,
        &resources.kernels[phase]));
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(instance));
    return HCCL_SUCCESS;
}

HcclResult RegisterHierarchicalKernels(HcclComm comm, const OpParam &param, const HierarchicalPlan &plan,
    const std::vector<uint32_t> &peers, const std::vector<uint32_t> &channelDies,
    const std::vector<ChannelHandle> &channels, AlgResourceCtx &resources)
{
    CcuInsHandle instance = 0;
    uint32_t instanceCount = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &instance, &instanceCount));
    CHK_PRT_RET(instanceCount != 1, HCCL_ERROR("expected one CCU instance, got %u", instanceCount), HCCL_E_INTERNAL);

    const uint64_t dataBytes = param.count * sizeof(float);
    const bool inSource = Contains(plan.sourceGroup, param.myRank);

    std::vector<uint32_t> phasePeers;
    std::vector<uint64_t> transferOffsets;
    std::vector<uint64_t> transferLengths;
    BroadcastKernelMode mode = BroadcastKernelMode::NOOP;
    uint32_t groupSize = 1;
    uint32_t myIndex = 0;
    if (inSource) {
        groupSize = static_cast<uint32_t>(plan.sourceGroup.size());
        myIndex = RankIndex(plan.sourceGroup, param.myRank);
        if (param.myRank == param.root) {
            mode = BroadcastKernelMode::SCATTER_SEND;
            for (uint32_t rank : plan.sourceGroup) {
                if (rank != param.root) {
                    phasePeers.push_back(rank);
                    uint64_t offset = 0;
                    uint64_t length = 0;
                    GetChunkRange(dataBytes, groupSize, RankIndex(plan.sourceGroup, rank), offset, length);
                    transferOffsets.push_back(offset);
                    transferLengths.push_back(length);
                }
            }
        } else {
            mode = BroadcastKernelMode::SCATTER_RECV;
            phasePeers.push_back(param.root);
            uint64_t offset = 0;
            uint64_t length = 0;
            GetChunkRange(dataBytes, groupSize, myIndex, offset, length);
            transferOffsets.push_back(offset);
            transferLengths.push_back(length);
        }
    }
    CHK_RET(RegisterHierarchicalPhase(instance, param, 0, mode, groupSize, myIndex, phasePeers, transferOffsets,
        transferLengths, dataBytes, peers, channelDies, channels, resources));

    phasePeers.clear();
    transferOffsets.clear();
    transferLengths.clear();
    mode = BroadcastKernelMode::NOOP;
    groupSize = 1;
    myIndex = 0;
    if (!plan.destinationGroup.empty()) {
        uint64_t ownedOffset = 0;
        uint64_t ownedLength = 0;
        if (inSource) {
            const uint32_t sourceIndex = RankIndex(plan.sourceGroup, param.myRank);
            GetChunkRange(
                dataBytes, static_cast<uint32_t>(plan.sourceGroup.size()), sourceIndex, ownedOffset, ownedLength);
            for (uint32_t destinationIndex = 0; destinationIndex < plan.destinationGroup.size(); ++destinationIndex) {
                uint64_t destinationOffset = 0;
                uint64_t destinationLength = 0;
                GetChunkRange(dataBytes, static_cast<uint32_t>(plan.destinationGroup.size()), destinationIndex,
                    destinationOffset, destinationLength);
                const uint64_t begin = std::max(ownedOffset, destinationOffset);
                const uint64_t end = std::min(ownedOffset + ownedLength, destinationOffset + destinationLength);
                if (begin < end) {
                    phasePeers.push_back(plan.destinationGroup[destinationIndex]);
                    transferOffsets.push_back(begin);
                    transferLengths.push_back(end - begin);
                }
            }
            if (!phasePeers.empty()) {
                mode = BroadcastKernelMode::SCATTER_SEND;
            }
        } else {
            mode = BroadcastKernelMode::SCATTER_RECV;
            const uint32_t destinationIndex = RankIndex(plan.destinationGroup, param.myRank);
            GetChunkRange(dataBytes, static_cast<uint32_t>(plan.destinationGroup.size()), destinationIndex, ownedOffset,
                ownedLength);
            for (uint32_t sourceIndex = 0; sourceIndex < plan.sourceGroup.size(); ++sourceIndex) {
                uint64_t sourceOffset = 0;
                uint64_t sourceLength = 0;
                GetChunkRange(
                    dataBytes, static_cast<uint32_t>(plan.sourceGroup.size()), sourceIndex, sourceOffset, sourceLength);
                const uint64_t begin = std::max(ownedOffset, sourceOffset);
                const uint64_t end = std::min(ownedOffset + ownedLength, sourceOffset + sourceLength);
                if (begin < end) {
                    phasePeers.push_back(plan.sourceGroup[sourceIndex]);
                    transferOffsets.push_back(begin);
                    transferLengths.push_back(end - begin);
                }
            }
        }
    }
    CHK_RET(RegisterHierarchicalPhase(instance, param, 1, mode, groupSize, myIndex, phasePeers, transferOffsets,
        transferLengths, dataBytes, peers, channelDies, channels, resources));

    phasePeers.clear();
    transferOffsets.clear();
    transferLengths.clear();
    mode = BroadcastKernelMode::ALLGATHER;
    const std::vector<uint32_t> &localGroup = inSource ? plan.sourceGroup : plan.destinationGroup;
    groupSize = static_cast<uint32_t>(localGroup.size());
    myIndex = RankIndex(localGroup, param.myRank);
    for (uint32_t rank : localGroup) {
        if (rank != param.myRank) {
            phasePeers.push_back(rank);
            transferOffsets.push_back(0);
            transferLengths.push_back(0);
        }
    }
    CHK_RET(RegisterHierarchicalPhase(instance, param, 2, mode, groupSize, myIndex, phasePeers, transferOffsets,
        transferLengths, dataBytes, peers, channelDies, channels, resources));
    resources.kernelCount = BROADCAST_PHASE_COUNT;
    resources.hierarchical = true;
    return HCCL_SUCCESS;
}

} // namespace

HcclResult HcclBroadcast(
    void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(buf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32, HCCL_ERROR("only FP32 broadcast is supported"), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(count > std::numeric_limits<uint64_t>::max() / sizeof(float), HCCL_ERROR("invalid broadcast count"),
        HCCL_E_PARA);

    OpParam param;
    param.inputPtr = buf;
    param.outputPtr = buf;
    param.count = count;
    param.root = root;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_BROADCAST;
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize < 2 || param.rankSize > MAX_RANK_SIZE || root >= param.rankSize,
        HCCL_ERROR("unsupported broadcast rankSize=%u root=%u", param.rankSize, root), HCCL_E_PARA);
    if (count == 0) {
        return HCCL_SUCCESS;
    }
    const uint64_t dataBytes = count * sizeof(float);
    const bool useHierarchicalPath = dataBytes > SMALL_MESSAGE_BYTES && param.rankSize >= 4;
    const char *contextTag = useHierarchicalPath ? HIERARCHICAL_CONTEXT_TAG : DIRECT_CONTEXT_TAG;
    std::snprintf(param.tag, sizeof(param.tag), "%s_%u_r%u_%s_%lu", contextTag, param.rankSize, root,
        useHierarchicalPath ? "hier" : "direct", static_cast<unsigned long>(dataBytes));

    HcclDfxOpInfo dfxInfo;
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclThreadAcquireWithStream(comm, COMM_ENGINE_CCU, stream, 1, &param.cpuThread));
    void *context = nullptr;
    uint64_t contextSize = 0;
    if (HcclEngineCtxGet(comm, param.tag, COMM_ENGINE_CCU, &context, &contextSize) == HCCL_SUCCESS) {
        param.resCtx = context;
        param.ctxSize = contextSize;
    } else {
        AlgResourceCtx resources;
        resources.thread = param.cpuThread;
        std::vector<ChannelHandle> channels;
        std::vector<uint32_t> channelDies;
        if (useHierarchicalPath) {
            HierarchicalPlan plan;
            CHK_RET(BuildHierarchicalPlan(comm, param, plan));
            std::vector<uint32_t> peers;
            CHK_RET(AcquireHierarchicalChannels(comm, param, plan, peers, channelDies, channels));
            CHK_RET(RegisterHierarchicalKernels(comm, param, plan, peers, channelDies, channels, resources));
        } else {
            CHK_RET(AcquireDirectChannels(comm, param, channelDies, channels));
            CHK_RET(RegisterKernel(comm, param, channelDies, channels, resources));
            if (param.rankSize > 4 && resources.kernelCount == DIE_COUNT) {
                CHK_RET(HcclThreadAcquire(comm, COMM_ENGINE_CCU, 1, 1, &resources.auxiliaryThread));
            }
        }
        std::vector<char> serialized = resources.Serialize();
        param.ctxSize = serialized.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, COMM_ENGINE_CCU, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, COMM_ENGINE_CCU, param.tag, serialized.data(), serialized.size(), 0));
    }
    return ops_hccl::ExecOp(param);
}