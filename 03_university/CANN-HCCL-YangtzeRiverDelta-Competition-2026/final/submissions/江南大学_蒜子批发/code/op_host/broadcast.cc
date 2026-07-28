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
#include <hccl/hccl_ccu_res.h>
#include <ccu/ccu_launch.h>

#include <cstdio>
#include <limits>
#include <map>
#include <memory>
#include <vector>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "exec_op.h"
#include "ccu_kernel.h"

namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 1;
constexpr uint32_t PIPELINE_CHANNEL_NOTIFY_NUM = 2;
constexpr uint64_t SMALL_TOPO_SCATTER_THRESHOLD = 4 * 1024 * 1024;
constexpr uint64_t LARGE_TOPO_SCATTER_THRESHOLD = 256 * 1024;

struct ChannelGroup {
    uint32_t dieId = 0;
    BroadcastKernelRole role = BroadcastKernelRole::PUSH_RECEIVER;
    uint32_t relayRankA = MAX_RANK_SIZE;
    uint32_t partnerRank = MAX_RANK_SIZE;
    bool pipelineCrossDie = false;
    std::vector<ChannelHandle> channels;
    std::vector<uint32_t> peerRanks;
};

struct PeerChannel {
    uint32_t peerRank = 0;
    uint32_t dieId = 0;
    ChannelHandle channel = 0;
};

struct CcuLinkSelection {
    uint32_t remoteRank = 0;
    uint32_t dieId = 0;
    CommLink link{};
};

bool RootParticipates(uint32_t rankSize)
{
    return rankSize == 4 || rankSize == 12 || rankSize == 16;
}

bool SupportsTwoDieParallel(const std::vector<ChannelGroup> &groups)
{
    if (groups.size() != 2 || groups[0].dieId == groups[1].dieId) {
        return false;
    }
    const auto isPipelineRole = [](BroadcastKernelRole role) {
        return role == BroadcastKernelRole::PIPELINE_SPLIT_ROOT ||
               role == BroadcastKernelRole::PIPELINE_SPLIT_RELAY ||
               role == BroadcastKernelRole::PIPELINE_SPLIT_TAIL;
    };
    if (isPipelineRole(groups[0].role) && isPipelineRole(groups[1].role)) {
        return true;
    }
    if (groups[0].role != groups[1].role) {
        return false;
    }
    const BroadcastKernelRole role = groups[0].role;
    return role == BroadcastKernelRole::PUSH_SENDER ||
           role == BroadcastKernelRole::SCATTER_RECEIVER ||
           role == BroadcastKernelRole::SCATTER_SENDER ||
           role == BroadcastKernelRole::PULL_ALLGATHER ||
           role == BroadcastKernelRole::PUSH_ALLGATHER ||
           role == BroadcastKernelRole::FUSED_SCATTER_ROOT ||
           role == BroadcastKernelRole::FUSED_PUSH_ALLGATHER;
}

HcclResult SelectCcuLinks(HcclComm comm, uint32_t myRank, uint32_t remoteRank,
    bool selectBothDies, std::vector<CcuLinkSelection> &selections,
    uint32_t preferredDie = std::numeric_limits<uint32_t>::max())
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));

    bool found = false;
    for (uint32_t layerIdx = 0; layerIdx < netLayerNum; ++layerIdx) {
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayers[layerIdx], myRank, remoteRank, &links, &linkNum));
        for (uint32_t linkIdx = 0; linkIdx < linkNum; ++linkIdx) {
            if (links[linkIdx].linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                continue;
            }
            uint32_t dieId = 0;
            CHK_RET(HcclRankGraphGetEndpointInfo(comm, myRank, &links[linkIdx].srcEndpointDesc,
                ENDPOINT_ATTR_DIE_ID, sizeof(dieId), &dieId));
            if (preferredDie != std::numeric_limits<uint32_t>::max() && dieId != preferredDie) {
                continue;
            }
            bool dieAlreadySelected = false;
            for (const auto &selection : selections) {
                if (selection.remoteRank == remoteRank && selection.dieId == dieId) {
                    dieAlreadySelected = true;
                    break;
                }
            }
            if (dieAlreadySelected) {
                continue;
            }
            selections.push_back(CcuLinkSelection{remoteRank, dieId, links[linkIdx]});
            found = true;
            if (!selectBothDies) {
                return HCCL_SUCCESS;
            }
        }
    }

    CHK_PRT_RET(!found,
        HCCL_ERROR("No UBC_CTP link found between rank[%u] and rank[%u]", myRank, remoteRank),
        HCCL_E_NOT_FOUND);
    return HCCL_SUCCESS;
}

HcclResult AcquirePeerChannels(HcclComm comm, CommEngine engine, const OpParam &param, bool acquireAllPeers,
    bool acquireBothDies, std::vector<PeerChannel> &peerChannels,
    uint32_t preferredDie = std::numeric_limits<uint32_t>::max(),
    uint32_t channelNotifyNum = CHANNEL_NOTIFY_NUM)
{
    if (param.rankSize <= 1) {
        return HCCL_SUCCESS;
    }

    std::vector<uint32_t> remoteRanks;
    if (acquireAllPeers || param.myRank == param.root) {
        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            if (rank != param.myRank) {
                remoteRanks.push_back(rank);
            }
        }
    } else {
        remoteRanks.push_back(param.root);
    }

    std::vector<CcuLinkSelection> selections;
    for (const uint32_t remoteRank : remoteRanks) {
        CHK_RET(SelectCcuLinks(
            comm, param.myRank, remoteRank, acquireBothDies, selections, preferredDie));
    }

    const uint32_t channelNum = static_cast<uint32_t>(selections.size());
    std::vector<HcclChannelDesc> descs(channelNum);
    CHK_RET(HcclChannelDescInit(descs.data(), channelNum));
    for (uint32_t channelIdx = 0; channelIdx < channelNum; ++channelIdx) {
        const auto &selection = selections[channelIdx];
        descs[channelIdx].remoteRank = selection.remoteRank;
        descs[channelIdx].notifyNum = channelNotifyNum;
        descs[channelIdx].channelProtocol = selection.link.linkAttr.linkProtocol;
        descs[channelIdx].localEndpoint = selection.link.srcEndpointDesc;
        descs[channelIdx].remoteEndpoint = selection.link.dstEndpointDesc;
    }

    std::vector<ChannelHandle> channels(channelNum);
    CHK_RET(HcclChannelAcquire(comm, engine, descs.data(), channelNum, channels.data()));

    for (uint32_t channelIdx = 0; channelIdx < channelNum; ++channelIdx) {
        peerChannels.push_back(PeerChannel{
            selections[channelIdx].remoteRank, selections[channelIdx].dieId, channels[channelIdx]});
    }
    return HCCL_SUCCESS;
}

const PeerChannel *FindPeerChannel(const std::vector<PeerChannel> &peerChannels, uint32_t peerRank)
{
    for (const auto &peerChannel : peerChannels) {
        if (peerChannel.peerRank == peerRank) {
            return &peerChannel;
        }
    }
    return nullptr;
}

HcclResult SelectPipelineRelays(const OpParam &param,
    uint32_t &relayRankA, uint32_t &relayRankB)
{
    CHK_PRT_RET(param.rankSize != 4,
        HCCL_ERROR("Pipeline relays do not support rankSize[%u]", param.rankSize), HCCL_E_PARA);
    relayRankA = (param.root + 1) % param.rankSize;
    relayRankB = (param.root + 2) % param.rankSize;
    return HCCL_SUCCESS;
}

bool GetRank4RecursiveStep(const OpParam &param, uint32_t phase,
    uint32_t &peerRank, BroadcastKernelRole &role)
{
    const uint32_t relativeRank = (param.myRank + param.rankSize - param.root) % param.rankSize;
    uint32_t peerRelativeRank = 0;
    if (phase == 0) {
        if (relativeRank == 0) {
            peerRelativeRank = 2;
            role = BroadcastKernelRole::RECURSIVE_SCATTER_SENDER;
        } else if (relativeRank == 2) {
            peerRelativeRank = 0;
            role = BroadcastKernelRole::RECURSIVE_SCATTER_RECEIVER;
        } else {
            return false;
        }
    } else if (phase == 1) {
        peerRelativeRank = relativeRank ^ 1U;
        role = (relativeRank & 1U) == 0 ? BroadcastKernelRole::RECURSIVE_SCATTER_SENDER :
            BroadcastKernelRole::RECURSIVE_SCATTER_RECEIVER;
    } else if (phase == 2) {
        peerRelativeRank = relativeRank ^ 2U;
        if (relativeRank == 0) {
            role = BroadcastKernelRole::RECURSIVE_SCATTER_SENDER;
        } else if (relativeRank == 2) {
            role = BroadcastKernelRole::RECURSIVE_SCATTER_RECEIVER;
        } else {
            role = BroadcastKernelRole::RECURSIVE_EXCHANGE;
        }
    } else {
        return false;
    }
    peerRank = (param.root + peerRelativeRank) % param.rankSize;
    return true;
}

HcclResult AddReceiverGroup(const std::vector<PeerChannel> &peerChannels, uint32_t parentRank,
    BroadcastKernelRole role, std::vector<ChannelGroup> &groups)
{
    const PeerChannel *peerChannel = FindPeerChannel(peerChannels, parentRank);
    CHK_PRT_RET(peerChannel == nullptr,
        HCCL_ERROR("Channel to parent rank[%u] was not acquired", parentRank), HCCL_E_NOT_FOUND);

    ChannelGroup group;
    group.dieId = peerChannel->dieId;
    group.role = role;
    group.channels.push_back(peerChannel->channel);
    group.peerRanks.push_back(parentRank);
    groups.push_back(std::move(group));
    return HCCL_SUCCESS;
}

HcclResult AddSenderGroups(const std::vector<PeerChannel> &peerChannels,
    const std::vector<uint32_t> &childRanks, BroadcastKernelRole role, std::vector<ChannelGroup> &groups)
{
    std::map<uint32_t, ChannelGroup> groupsByDie;
    for (const uint32_t childRank : childRanks) {
        const PeerChannel *peerChannel = FindPeerChannel(peerChannels, childRank);
        CHK_PRT_RET(peerChannel == nullptr,
            HCCL_ERROR("Channel to child rank[%u] was not acquired", childRank), HCCL_E_NOT_FOUND);

        auto &group = groupsByDie[peerChannel->dieId];
        group.dieId = peerChannel->dieId;
        group.role = role;
        group.channels.push_back(peerChannel->channel);
        group.peerRanks.push_back(childRank);
    }
    for (auto &entry : groupsByDie) {
        groups.push_back(std::move(entry.second));
    }
    return HCCL_SUCCESS;
}

HcclResult BuildDirectGroups(const OpParam &param, const std::vector<PeerChannel> &peerChannels,
    bool usePull, std::vector<ChannelGroup> &groups)
{
    if (param.myRank != param.root) {
        const BroadcastKernelRole role = usePull ? BroadcastKernelRole::PULL_RECEIVER :
            BroadcastKernelRole::PUSH_RECEIVER;
        return AddReceiverGroup(peerChannels, param.root, role, groups);
    }

    std::vector<uint32_t> childRanks;
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (rank != param.myRank) {
            childRanks.push_back(rank);
        }
    }
    const BroadcastKernelRole role = usePull ? BroadcastKernelRole::PULL_ROOT :
        BroadcastKernelRole::PUSH_SENDER;
    return AddSenderGroups(peerChannels, childRanks, role, groups);
}

HcclResult BuildPipelineSplitGroups(const OpParam &param,
    const std::vector<PeerChannel> &peerChannels, uint32_t relayRankA,
    uint32_t relayRankB, std::vector<ChannelGroup> &groups)
{
    CHK_PRT_RET(param.rankSize != 4 && param.rankSize != 12 && param.rankSize != 16,
        HCCL_ERROR("Pipeline split does not support rankSize[%u]", param.rankSize), HCCL_E_PARA);
    if (param.myRank == param.root) {
        CHK_RET(AddSenderGroups(peerChannels, {relayRankA, relayRankB},
            BroadcastKernelRole::PIPELINE_SPLIT_ROOT, groups));
    } else if (param.myRank == relayRankA || param.myRank == relayRankB) {
        const uint32_t partnerRank =
            param.myRank == relayRankA ? relayRankB : relayRankA;
        std::vector<uint32_t> destinations;
        destinations.reserve(param.rankSize - 1);
        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            if (rank != param.myRank) {
                destinations.push_back(rank);
            }
        }
        CHK_RET(AddSenderGroups(
            peerChannels, destinations, BroadcastKernelRole::PIPELINE_SPLIT_RELAY, groups));
        for (auto &group : groups) {
            group.partnerRank = partnerRank;
        }
        if (groups.size() > 1) {
            for (auto &group : groups) {
                for (const uint32_t peerRank : group.peerRanks) {
                    if (peerRank == param.root) {
                        group.pipelineCrossDie = true;
                    }
                }
            }
        }
    } else {
        CHK_RET(AddSenderGroups(peerChannels, {relayRankA, relayRankB},
            BroadcastKernelRole::PIPELINE_SPLIT_TAIL, groups));
    }
    for (auto &group : groups) {
        group.relayRankA = relayRankA;
    }
    return HCCL_SUCCESS;
}

HcclResult BuildScatterGroups(const OpParam &param, const std::vector<PeerChannel> &peerChannels,
    std::vector<ChannelGroup> &groups)
{
    const bool fuseRootSlice = param.rankSize == 12 || param.rankSize == 16;
    if (param.myRank != param.root) {
        if (fuseRootSlice) {
            const PeerChannel *rootChannel = FindPeerChannel(peerChannels, param.root);
            CHK_PRT_RET(rootChannel == nullptr,
                HCCL_ERROR("Channel to Broadcast root rank[%u] was not acquired", param.root),
                HCCL_E_NOT_FOUND);
            ChannelGroup group;
            group.dieId = rootChannel->dieId;
            group.role = BroadcastKernelRole::FUSED_SCATTER_RECEIVER;
            group.channels.push_back(rootChannel->channel);
            group.peerRanks.push_back(param.root);
            for (const auto &peerChannel : peerChannels) {
                if (peerChannel.dieId == rootChannel->dieId &&
                    peerChannel.peerRank != param.root) {
                    group.channels.push_back(peerChannel.channel);
                    group.peerRanks.push_back(peerChannel.peerRank);
                }
            }
            groups.push_back(std::move(group));
            return HCCL_SUCCESS;
        }
        return AddReceiverGroup(
            peerChannels, param.root, BroadcastKernelRole::SCATTER_RECEIVER, groups);
    }

    std::vector<uint32_t> childRanks;
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (rank != param.myRank) {
            childRanks.push_back(rank);
        }
    }
    const BroadcastKernelRole role = fuseRootSlice ? BroadcastKernelRole::FUSED_SCATTER_ROOT :
                                                     BroadcastKernelRole::SCATTER_SENDER;
    return AddSenderGroups(peerChannels, childRanks, role, groups);
}

HcclResult BuildAllgatherGroups(const OpParam &param, const std::vector<PeerChannel> &peerChannels,
    std::vector<ChannelGroup> &groups)
{
    // Keep the root as a worker so Scatter transfers only (N-1)/N of the
    // message. Four ranks use Pull; twelve and sixteen ranks use Push.
    const bool rootParticipates = RootParticipates(param.rankSize);
    const bool fuseRootSlice = param.rankSize == 12 || param.rankSize == 16;
    if (fuseRootSlice && param.myRank == param.root) {
        return HCCL_SUCCESS;
    }
    if (!rootParticipates && param.myRank == param.root) {
        return HCCL_SUCCESS;
    }

    std::vector<uint32_t> peerRanks;
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (rank != param.myRank && (rootParticipates || rank != param.root)) {
            peerRanks.push_back(rank);
        }
    }
    const bool usePull = param.rankSize == 4;
    const BroadcastKernelRole role = fuseRootSlice ? BroadcastKernelRole::FUSED_PUSH_ALLGATHER :
        (usePull ? BroadcastKernelRole::PULL_ALLGATHER : BroadcastKernelRole::PUSH_ALLGATHER);
    return AddSenderGroups(peerChannels, peerRanks, role, groups);
}

HcclResult SaveGroupPeerRanks(const std::vector<ChannelGroup> &groups,
    uint32_t peerCounts[AlgResourceCtx::MAX_KERNEL_GROUPS],
    uint32_t peerRanks[AlgResourceCtx::MAX_KERNEL_GROUPS][MAX_RANK_SIZE])
{
    CHK_PRT_RET(groups.size() > AlgResourceCtx::MAX_KERNEL_GROUPS,
        HCCL_ERROR("Too many CCU kernel groups[%zu]", groups.size()), HCCL_E_INTERNAL);
    for (uint32_t groupIdx = 0; groupIdx < groups.size(); ++groupIdx) {
        const auto &groupPeers = groups[groupIdx].peerRanks;
        CHK_PRT_RET(groupPeers.size() > MAX_RANK_SIZE,
            HCCL_ERROR("Too many peers[%zu] in CCU kernel group[%u]", groupPeers.size(), groupIdx),
            HCCL_E_INTERNAL);
        peerCounts[groupIdx] = static_cast<uint32_t>(groupPeers.size());
        for (uint32_t peerIdx = 0; peerIdx < groupPeers.size(); ++peerIdx) {
            peerRanks[groupIdx][peerIdx] = groupPeers[peerIdx];
        }
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterBroadcastKernels(HcclComm comm, const OpParam &param,
    const std::vector<ChannelGroup> &stageOneGroups, const std::vector<ChannelGroup> &stageTwoGroups,
    const char *stageOneName, const char *stageTwoName, AlgResourceCtx &resCtx)
{
    CcuInsHandle insHandle = 0;
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1,
        HCCL_ERROR("HcclCommQueryCcuIns returned invalid insNum[%u]", insNum), HCCL_E_INTERNAL);

    CcuResult ccuRet = HcommCcuKernelRegisterStart(insHandle);
    if (ccuRet != CCU_SUCCESS) {
        HCCL_ERROR("CCU kernel register start failed, ret[%d]", ccuRet);
        return CcuResultToHccl(ccuRet);
    }

    CHK_PRT_RET(stageOneGroups.size() > AlgResourceCtx::MAX_KERNEL_GROUPS ||
        stageTwoGroups.size() > AlgResourceCtx::MAX_KERNEL_GROUPS,
        HCCL_ERROR("Too many Broadcast CCU kernel groups, stageOne[%zu], stageTwo[%zu]",
            stageOneGroups.size(), stageTwoGroups.size()), HCCL_E_INTERNAL);
    resCtx.stageOneKernelCount = static_cast<uint32_t>(stageOneGroups.size());
    resCtx.stageTwoKernelCount = static_cast<uint32_t>(stageTwoGroups.size());
    resCtx.stageOneParallel = SupportsTwoDieParallel(stageOneGroups);
    resCtx.stageTwoParallel = SupportsTwoDieParallel(stageTwoGroups);
    CHK_RET(SaveGroupPeerRanks(stageOneGroups, resCtx.stageOnePeerCounts, resCtx.stageOnePeerRanks));
    CHK_RET(SaveGroupPeerRanks(stageTwoGroups, resCtx.stageTwoPeerCounts, resCtx.stageTwoPeerRanks));
    const auto registerGroups = [&](const char *kernelStageName, const std::vector<ChannelGroup> &kernelGroups,
                                    CcuKernelHandle *kernels) -> CcuResult {
        for (uint32_t groupIdx = 0; groupIdx < kernelGroups.size(); ++groupIdx) {
            const ChannelGroup &group = kernelGroups[groupIdx];
            const bool isReceiver = group.role == BroadcastKernelRole::PUSH_RECEIVER ||
                                    group.role == BroadcastKernelRole::SCATTER_RECEIVER ||
                                    group.role == BroadcastKernelRole::PULL_RECEIVER ||
                                    group.role == BroadcastKernelRole::FUSED_SCATTER_RECEIVER;
            const char *roleName = isReceiver ? "Recv" :
                (group.role == BroadcastKernelRole::PULL_ROOT ? "Root" :
                (group.role == BroadcastKernelRole::PULL_ALLGATHER ? "Pull" :
                (group.role == BroadcastKernelRole::PUSH_ALLGATHER ? "Exchange" : "Send")));
            CcuKernelInfo kernelInfo{};
            std::snprintf(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName),
                "CcuBcast%s%sDie%uGroup%u", kernelStageName, roleName, group.dieId, groupIdx);
            kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuKernel);

            auto kernelArg = std::make_shared<CcuKernelArgBroadcast>();
            kernelArg->rankSize = param.rankSize;
            kernelArg->rankId = param.myRank;
            kernelArg->root = param.root;
            kernelArg->role = group.role;
            kernelArg->relayRankA = group.relayRankA;
            kernelArg->partnerRank = group.partnerRank;
            kernelArg->pipelineCrossDie = group.pipelineCrossDie;
            kernelArg->channelCount = static_cast<uint32_t>(group.channels.size());
            for (uint32_t channelIdx = 0; channelIdx < kernelArg->channelCount; ++channelIdx) {
                kernelArg->channels[channelIdx] = group.channels[channelIdx];
                kernelArg->peerRanks[channelIdx] = group.peerRanks[channelIdx];
            }
            kernelInfo.setKernelArg(kernelArg);

            const void *kernelArgs[] = {kernelInfo.kernelArg};
            const CcuResult registerRet = HcommCcuKernelRegister(insHandle, group.dieId, kernelInfo.kernelFuncName,
                kernelInfo.kernelFunc, kernelArgs, 1, &kernels[groupIdx]);
            if (registerRet != CCU_SUCCESS) {
                return registerRet;
            }
        }
        return CCU_SUCCESS;
    };

    ccuRet = registerGroups(stageOneName, stageOneGroups, resCtx.stageOneKernels);
    if (ccuRet == CCU_SUCCESS && !stageTwoGroups.empty()) {
        ccuRet = registerGroups(stageTwoName, stageTwoGroups, resCtx.stageTwoKernels);
    }
    const CcuResult endRet = HcommCcuKernelRegisterEnd(insHandle);
    if (ccuRet != CCU_SUCCESS) {
        HCCL_ERROR("CCU kernel register failed, ret[%d]", ccuRet);
        return CcuResultToHccl(ccuRet);
    }
    if (endRet != CCU_SUCCESS) {
        HCCL_ERROR("CCU kernel register end failed, ret[%d]", endRet);
        return CcuResultToHccl(endRet);
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterRank4RecursiveKernels(HcclComm comm, const OpParam &param,
    const std::vector<PeerChannel> &peerChannels, AlgResourceCtx &resCtx)
{
    CcuInsHandle insHandle = 0;
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1,
        HCCL_ERROR("HcclCommQueryCcuIns returned invalid insNum[%u]", insNum), HCCL_E_INTERNAL);
    CcuResult ccuRet = HcommCcuKernelRegisterStart(insHandle);
    if (ccuRet != CCU_SUCCESS) {
        return CcuResultToHccl(ccuRet);
    }

    std::map<uint32_t, CcuKernelHandle> kernelsByPeer;
    for (uint32_t phase = 0; phase < 3 && ccuRet == CCU_SUCCESS; ++phase) {
        uint32_t peerRank = 0;
        BroadcastKernelRole stepRole = BroadcastKernelRole::RECURSIVE_EXCHANGE;
        if (!GetRank4RecursiveStep(param, phase, peerRank, stepRole)) {
            continue;
        }
        const auto cachedKernel = kernelsByPeer.find(peerRank);
        if (cachedKernel != kernelsByPeer.end()) {
            resCtx.recursiveKernels[phase] = cachedKernel->second;
            resCtx.recursiveKernelMask |= 1U << phase;
            continue;
        }
        const PeerChannel *peerChannel = FindPeerChannel(peerChannels, peerRank);
        if (peerChannel == nullptr) {
            ccuRet = CCU_E_NOT_FOUND;
            break;
        }

        CcuKernelInfo kernelInfo{};
        std::snprintf(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName),
            "CcuBcastRank4RecursivePeer%uDie%u", peerRank, peerChannel->dieId);
        kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuKernel);

        auto kernelArg = std::make_shared<CcuKernelArgBroadcast>();
        kernelArg->rankSize = param.rankSize;
        kernelArg->rankId = param.myRank;
        kernelArg->root = param.root;
        kernelArg->role = BroadcastKernelRole::RECURSIVE_DYNAMIC;
        kernelArg->partnerRank = static_cast<uint32_t>(stepRole);
        kernelArg->channelCount = 1;
        kernelArg->channels[0] = peerChannel->channel;
        kernelArg->peerRanks[0] = peerRank;
        kernelInfo.setKernelArg(kernelArg);

        const void *kernelArgs[] = {kernelInfo.kernelArg};
        ccuRet = HcommCcuKernelRegister(insHandle, peerChannel->dieId, kernelInfo.kernelFuncName,
            kernelInfo.kernelFunc, kernelArgs, 1, &resCtx.recursiveKernels[phase]);
        if (ccuRet == CCU_SUCCESS) {
            kernelsByPeer[peerRank] = resCtx.recursiveKernels[phase];
            resCtx.recursiveKernelMask |= 1U << phase;
        }
    }

    const CcuResult endRet = HcommCcuKernelRegisterEnd(insHandle);
    if (ccuRet != CCU_SUCCESS) {
        HCCL_ERROR("Rank4 recursive kernel register failed, ret[%d]", ccuRet);
        return CcuResultToHccl(ccuRet);
    }
    if (endRet != CCU_SUCCESS) {
        return CcuResultToHccl(endRet);
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclBroadcast(
    void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(buf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    // 构造算子参数
    OpParam param;
    param.inputPtr = buf;
    param.outputPtr = buf;
    param.count = count;
    param.dataType = dataType;
    param.root = root;
    param.opType = HcclCMDType::HCCL_CMD_BROADCAST;

    // 注册算子信息
    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    // ==============================================
    // STEP 1: 解析拓扑信息
    // ==============================================
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE,
        HCCL_ERROR("Invalid rankSize[%u]", param.rankSize), HCCL_E_PARA);
    CHK_PRT_RET(root >= param.rankSize,
        HCCL_ERROR("Invalid root[%u], rankSize[%u]", root, param.rankSize), HCCL_E_PARA);
    CHK_PRT_RET(SIZE_TABLE.find(dataType) == SIZE_TABLE.end(),
        HCCL_ERROR("Unsupported dataType[%d]", static_cast<int32_t>(dataType)), HCCL_E_PARA);
    if (count == 0) {
        return HCCL_SUCCESS;
    }
    const uint64_t dataTypeSize = SIZE_TABLE.at(dataType);
    CHK_PRT_RET(count > std::numeric_limits<uint64_t>::max() / dataTypeSize,
        HCCL_ERROR("Broadcast data size overflow, count[%llu]", static_cast<unsigned long long>(count)),
        HCCL_E_PARA);
    const uint64_t dataSize = count * dataTypeSize;
    const uint64_t workerCount = RootParticipates(param.rankSize) ? param.rankSize :
        (param.rankSize > 1 ? param.rankSize - 1 : 0);
    const uint64_t maxWorkerElements = workerCount == 0 ? count :
        count / workerCount + count % workerCount;
    const uint64_t scatterThreshold =
        (param.rankSize == 12 || param.rankSize == 16) ?
        LARGE_TOPO_SCATTER_THRESHOLD : SMALL_TOPO_SCATTER_THRESHOLD;
    const bool usePipelineSplit = param.rankSize == 4 &&
        dataSize > SMALL_TOPO_SCATTER_THRESHOLD &&
        dataSize <= 2ULL * static_cast<uint64_t>(MAX_DATA_SIZE);
    const bool useScatterAllgather = !usePipelineSplit &&
        param.rankSize > 2 &&
        dataSize > scatterThreshold &&
        maxWorkerElements <= static_cast<uint64_t>(MAX_DATA_SIZE) / dataTypeSize;
    const bool useFusedScatterAllgather = useScatterAllgather &&
        (param.rankSize == 12 || param.rankSize == 16);
    const bool useRank4Recursive = useScatterAllgather && param.rankSize == 4 &&
        dataSize <= 2ULL * static_cast<uint64_t>(MAX_DATA_SIZE);
    const bool useDirectPull = !useScatterAllgather && !usePipelineSplit &&
        param.rankSize == 4;
    const char *algorithmName = "DirectPush";
    if (usePipelineSplit) {
        algorithmName = "Rank4RelayWindow26BalancedTail4K";
    } else if (useRank4Recursive) {
        algorithmName = "Rank4BroadcastRecursiveAddr";
    } else if (useFusedScatterAllgather) {
        algorithmName = "SameDiePrepublishedFusedAllGather";
    } else if (useScatterAllgather) {
        algorithmName = "BestTopologyAllGather";
    } else if (useDirectPull) {
        algorithmName = "DirectPull";
    }
    std::snprintf(param.tag, sizeof(param.tag), "hccl_custom_broadcast_root_%u_%s_v100", root, algorithmName);

    // ==============================================
    // STEP 2: 创建资源
    // ==============================================
    CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;

    // ==============================================
    // STEP 2.1: 申请用于 Host/Device 同步的通信资源
    // ==============================================
    // 主CCU thread绑定用户stream；notify 0用于与第二个die的CCU thread建立阶段栅栏。
    CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, stream, 1, &param.cpuThread));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, ccuEngine, &ctx, &size) == HCCL_SUCCESS) {
        // CCU 资源已经存在，复用资源
        HCCL_INFO("Engine context already exists");
        param.resCtx = ctx;
        param.ctxSize = size;
    } else {
        // Device 资源不存在，资源构建
        AlgResourceCtx resCtxHost{};

        // 从通信域获取 HCCL Buffer（Device上的内存，默认总大小400MB）
        void *cclBufferAddr;
        uint64_t cclBufferSize;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        // ==============================================
        // STEP 2.2: 申请资源：Thread、Channel、CCU Kernel
        // ==============================================

        resCtxHost.ccuThread = param.cpuThread;
        resCtxHost.algorithm = usePipelineSplit ? BroadcastAlgorithm::RANK4_PIPELINE_SPLIT :
            (useRank4Recursive ? BroadcastAlgorithm::RANK4_RECURSIVE :
            (useFusedScatterAllgather ? BroadcastAlgorithm::FUSED_SCATTER_ALLGATHER :
            (useScatterAllgather ? BroadcastAlgorithm::SCATTER_ALLGATHER :
            (useDirectPull ? BroadcastAlgorithm::DIRECT_PULL : BroadcastAlgorithm::DIRECT))));

        if (param.rankSize > 1) {
            std::vector<PeerChannel> peerChannels;
            std::vector<ChannelGroup> stageOneGroups;
            std::vector<ChannelGroup> stageTwoGroups;
            CHK_RET(AcquirePeerChannels(comm, ccuEngine, param,
                usePipelineSplit || useScatterAllgather, false, peerChannels,
                std::numeric_limits<uint32_t>::max(),
                usePipelineSplit ? PIPELINE_CHANNEL_NOTIFY_NUM : CHANNEL_NOTIFY_NUM));
            if (usePipelineSplit) {
                uint32_t relayRankA = 0;
                uint32_t relayRankB = 0;
                CHK_RET(SelectPipelineRelays(param, relayRankA, relayRankB));
                CHK_RET(BuildPipelineSplitGroups(
                    param, peerChannels, relayRankA, relayRankB, stageOneGroups));
                CHK_RET(RegisterBroadcastKernels(comm, param, stageOneGroups, stageTwoGroups,
                    "K4Split26WindowBalanced4K", "Unused", resCtxHost));
            } else if (useRank4Recursive) {
                CHK_RET(RegisterRank4RecursiveKernels(comm, param, peerChannels, resCtxHost));
            } else if (useScatterAllgather) {
                CHK_RET(BuildScatterGroups(param, peerChannels, stageOneGroups));
                CHK_RET(BuildAllgatherGroups(param, peerChannels, stageTwoGroups));
                CHK_RET(RegisterBroadcastKernels(comm, param, stageOneGroups, stageTwoGroups,
                    useFusedScatterAllgather ? "FusedSameDiePrepublish" : "Scatter",
                    param.rankSize == 4 ? "PullAllGather" : "PushAllGather", resCtxHost));
            } else {
                CHK_RET(BuildDirectGroups(param, peerChannels, useDirectPull, stageOneGroups));
                CHK_RET(RegisterBroadcastKernels(comm, param, stageOneGroups, stageTwoGroups,
                    useDirectPull ? "DirectPull" : "DirectPush", "Unused", resCtxHost));
            }
            if (resCtxHost.stageOneParallel || resCtxHost.stageTwoParallel) {
                CHK_RET(HcclThreadAcquire(comm, ccuEngine, 1, 1, &resCtxHost.parallelThread));
            }
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
    }

    // ==============================================
    // STEP 3: 下发 CCU Kernel
    // ==============================================
    CHK_RET(ops_hccl::ExecOp(param));
    return HCCL_SUCCESS;
}
