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

constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;
constexpr uint32_t CCU_DIE_ID = 0;
constexpr uint32_t THREAD_NOTIFY_NUM = 1;
constexpr size_t MAX_CCU_DIE_GROUPS = 2;

struct PeerLink {
    uint32_t rank;
    uint32_t dieId;
    bool isLocalPeer;
    uint32_t transferMode;
    CommLink link;
};

struct TopologyPlan {
    bool useTwoServerEight = false;
    uint32_t intraLayer = 0;
    uint32_t outgoingSeedRank = INVALID_VALUE_RANKID;
    uint32_t incomingRelayRank = INVALID_VALUE_RANKID;
    std::vector<uint32_t> localRanks;
    std::vector<uint32_t> remoteRanks;
};

struct ChannelGroup {
    uint32_t dieId;
    std::vector<PeerLink> peerLinks;
    std::vector<ChannelHandle> channels;
};

HcclResult GetNetworkLayers(HcclComm comm, std::vector<uint32_t> &layers)
{
    uint32_t *layerList = nullptr;
    uint32_t layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layerList, &layerCount));
    CHK_PRT_RET(layerCount == 0 || layerList == nullptr,
        HCCL_ERROR("[AllGather] rank graph has no network layer"), HCCL_E_NOT_FOUND);
    layers.assign(layerList, layerList + layerCount);
    return HCCL_SUCCESS;
}

HcclResult BuildTopologyPlan(HcclComm comm, const OpParam &param,
    const std::vector<uint32_t> &layers, TopologyPlan &plan)
{
    plan = TopologyPlan{};
    if (param.rankSize != 16) {
        return HCCL_SUCCESS;
    }

    constexpr uint32_t intraLayer = 0;
    if (std::count(layers.begin(), layers.end(), intraLayer) != 1) {
        return HCCL_SUCCESS;
    }

    uint32_t *instanceSizes = nullptr;
    uint32_t instanceCount = 0;
    if (HcclRankGraphGetInstSizeListByLayer(
            comm, intraLayer, &instanceSizes, &instanceCount) != HCCL_SUCCESS ||
        instanceSizes == nullptr || instanceCount != 2 ||
        instanceSizes[0] != 8 || instanceSizes[1] != 8) {
        return HCCL_SUCCESS;
    }

    uint32_t *rankList = nullptr;
    uint32_t rankCount = 0;
    if (HcclRankGraphGetRanksByLayer(
            comm, intraLayer, &rankList, &rankCount) != HCCL_SUCCESS ||
        rankList == nullptr || rankCount != 8) {
        return HCCL_SUCCESS;
    }
    std::vector<uint32_t> localRanks(rankList, rankList + rankCount);

    std::vector<bool> seen(param.rankSize, false);
    for (uint32_t rank : localRanks) {
        if (rank >= param.rankSize || seen[rank]) {
            return HCCL_SUCCESS;
        }
        seen[rank] = true;
    }
    if (!seen[param.myRank]) {
        return HCCL_SUCCESS;
    }

    std::vector<uint32_t> remoteRanks;
    remoteRanks.reserve(8);
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (!seen[rank]) {
            remoteRanks.push_back(rank);
        }
    }
    if (remoteRanks.size() != 8) {
        return HCCL_SUCCESS;
    }

    std::sort(localRanks.begin(), localRanks.end());
    std::sort(remoteRanks.begin(), remoteRanks.end());
    const auto selfIter = std::find(localRanks.begin(), localRanks.end(), param.myRank);
    if (selfIter == localRanks.end()) {
        return HCCL_SUCCESS;
    }
    const size_t localOrdinal =
        static_cast<size_t>(std::distance(localRanks.begin(), selfIter));

    plan.useTwoServerEight = true;
    plan.intraLayer = intraLayer;
    plan.outgoingSeedRank = remoteRanks[localOrdinal];
    plan.incomingRelayRank = remoteRanks[localOrdinal];
    plan.localRanks = std::move(localRanks);
    plan.remoteRanks = std::move(remoteRanks);
    return HCCL_SUCCESS;
}

HcclResult FindSupportedLink(HcclComm comm, const std::vector<uint32_t> &layers, uint32_t localRank,
    uint32_t remoteRank, CommLink &selectedLink, bool preferTp = false)
{
    bool foundCtp = false;
    CommLink ctpLink{};
    bool foundTp = false;
    CommLink tpLink{};

    for (uint32_t layer : layers) {
        CommLink *linkList = nullptr;
        uint32_t linkCount = 0;
        HcclResult ret = HcclRankGraphGetLinks(
            comm, layer, localRank, remoteRank, &linkList, &linkCount);
        if (ret != HCCL_SUCCESS || linkList == nullptr) {
            continue;
        }

        for (uint32_t index = 0; index < linkCount; ++index) {
            const CommProtocol protocol = linkList[index].linkAttr.linkProtocol;
            if (protocol == COMM_PROTOCOL_UBC_CTP) {
                if (!preferTp) {
                    selectedLink = linkList[index];
                    return HCCL_SUCCESS;
                }
                if (!foundCtp) {
                    ctpLink = linkList[index];
                    foundCtp = true;
                }
            }
            if (protocol == COMM_PROTOCOL_UBC_TP) {
                if (preferTp) {
                    selectedLink = linkList[index];
                    return HCCL_SUCCESS;
                }
                if (!foundTp) {
                    tpLink = linkList[index];
                    foundTp = true;
                }
            }
        }
    }

    if (foundCtp) {
        selectedLink = ctpLink;
        return HCCL_SUCCESS;
    }
    if (foundTp) {
        selectedLink = tpLink;
        return HCCL_SUCCESS;
    }

    HCCL_ERROR("[AllGather] no CCU link between rank %u and rank %u", localRank, remoteRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult BuildPeerLinks(HcclComm comm, const OpParam &param,
    const std::vector<uint32_t> &layers, const TopologyPlan &plan,
    std::vector<PeerLink> &peerLinks)
{
    peerLinks.clear();
    peerLinks.reserve(param.rankSize - 1);
    if (param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    for (uint32_t peerRank = 0; peerRank < param.rankSize; ++peerRank) {
        if (peerRank == param.myRank) {
            continue;
        }
        PeerLink peerLink{};
        peerLink.rank = peerRank;
        peerLink.isLocalPeer =
            std::find(plan.localRanks.begin(), plan.localRanks.end(), peerRank) !=
            plan.localRanks.end();
        peerLink.transferMode =
            static_cast<uint32_t>(ops_hccl::AllGatherTransferMode::FULL_CHUNK);
        if (plan.useTwoServerEight && !peerLink.isLocalPeer &&
            peerRank != plan.outgoingSeedRank) {
            peerLink.transferMode =
                static_cast<uint32_t>(ops_hccl::AllGatherTransferMode::DIRECT_PREFIX);
        }

        if (plan.useTwoServerEight && peerLink.isLocalPeer) {
            const std::vector<uint32_t> intraLayers = {plan.intraLayer};
            CHK_RET(FindSupportedLink(
                comm, intraLayers, param.myRank, peerRank, peerLink.link, true));
        } else {
            CHK_RET(FindSupportedLink(comm, layers, param.myRank, peerRank, peerLink.link));
        }
        EndpointAttrDieId dieId{};
        CHK_RET(HcclRankGraphGetEndpointInfo(comm, param.myRank, &peerLink.link.srcEndpointDesc,
            ENDPOINT_ATTR_DIE_ID, sizeof(dieId), &dieId));
        peerLink.dieId = dieId;
        peerLinks.push_back(peerLink);
    }
    return HCCL_SUCCESS;
}

HcclResult AcquireChannels(
    HcclComm comm, const std::vector<PeerLink> &peerLinks, std::vector<ChannelHandle> &channels)
{
    channels.clear();
    if (peerLinks.empty()) {
        return HCCL_SUCCESS;
    }

    std::vector<HcclChannelDesc> descs(peerLinks.size());
    CHK_RET(HcclChannelDescInit(descs.data(), static_cast<uint32_t>(descs.size())));
    for (size_t index = 0; index < peerLinks.size(); ++index) {
        const PeerLink &peer = peerLinks[index];
        HcclChannelDesc &desc = descs[index];
        desc.remoteRank = peer.rank;
        desc.notifyNum = CHANNEL_NOTIFY_NUM;
        desc.channelProtocol = peer.link.linkAttr.linkProtocol;
        desc.localEndpoint = peer.link.srcEndpointDesc;
        desc.remoteEndpoint = peer.link.dstEndpointDesc;
    }

    channels.resize(peerLinks.size());
    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, descs.data(),
        static_cast<uint32_t>(descs.size()), channels.data()));
    return HCCL_SUCCESS;
}

HcclResult GroupChannelsByDie(const std::vector<PeerLink> &peerLinks,
    const std::vector<ChannelHandle> &channels, std::vector<ChannelGroup> &groups)
{
    CHK_PRT_RET(peerLinks.size() != channels.size(),
        HCCL_ERROR("[AllGather] peer/channel count mismatch"), HCCL_E_INTERNAL);

    groups.clear();
    for (size_t index = 0; index < peerLinks.size(); ++index) {
        const uint32_t dieId = peerLinks[index].dieId;
        auto groupIter = std::find_if(groups.begin(), groups.end(),
            [dieId](const ChannelGroup &group) { return group.dieId == dieId; });
        if (groupIter == groups.end()) {
            ChannelGroup group{};
            group.dieId = dieId;
            groups.push_back(group);
            groupIter = groups.end() - 1;
        }
        groupIter->peerLinks.push_back(peerLinks[index]);
        groupIter->channels.push_back(channels[index]);
    }

    if (groups.empty()) {
        ChannelGroup localOnlyGroup{};
        localOnlyGroup.dieId = CCU_DIE_ID;
        groups.push_back(localOnlyGroup);
    }
    std::sort(groups.begin(), groups.end(),
        [](const ChannelGroup &left, const ChannelGroup &right) { return left.dieId < right.dieId; });
    CHK_PRT_RET(groups.size() > MAX_CCU_DIE_GROUPS,
        HCCL_ERROR("[AllGather] expected at most %zu CCU die groups, got %zu",
            MAX_CCU_DIE_GROUPS, groups.size()),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult RegisterKernels(HcclComm comm, const OpParam &param,
    const std::vector<ChannelGroup> &phaseOneGroups,
    const std::vector<ChannelGroup> &phaseTwoGroups,
    const TopologyPlan &plan, AlgResourceCtx &resource)
{
    CHK_PRT_RET(phaseOneGroups.empty(),
        HCCL_ERROR("[AllGather] no channel group to register"), HCCL_E_INTERNAL);
    CHK_PRT_RET(plan.useTwoServerEight && phaseTwoGroups.empty(),
        HCCL_ERROR("[AllGather] relay topology has no local channel group"), HCCL_E_INTERNAL);
    CHK_PRT_RET(!plan.useTwoServerEight && !phaseTwoGroups.empty(),
        HCCL_ERROR("[AllGather] direct topology unexpectedly has relay groups"), HCCL_E_INTERNAL);
    for (const ChannelGroup &group : phaseOneGroups) {
        CHK_PRT_RET(group.peerLinks.size() != group.channels.size() ||
                group.channels.size() >= param.rankSize,
            HCCL_ERROR("[AllGather] invalid distribute channel group"), HCCL_E_INTERNAL);
    }
    for (const ChannelGroup &group : phaseTwoGroups) {
        CHK_PRT_RET(group.peerLinks.size() != group.channels.size() ||
                group.channels.empty() || group.channels.size() >= param.rankSize,
            HCCL_ERROR("[AllGather] invalid relay channel group"), HCCL_E_INTERNAL);
    }

    size_t localCopyGroup = 0;
    for (size_t index = 1; index < phaseOneGroups.size(); ++index) {
        if (phaseOneGroups[index].channels.size() <
            phaseOneGroups[localCopyGroup].channels.size()) {
            localCopyGroup = index;
        }
    }

    CcuInsHandle insHandle{};
    uint32_t insCount = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insCount));
    CHK_PRT_RET(insCount != 1,
        HCCL_ERROR("[AllGather] expected one CCU instance, got %u", insCount), HCCL_E_INTERNAL);

    std::vector<std::shared_ptr<ops_hccl::CcuAllGatherKernelArg>> registeredArgs;
    registeredArgs.reserve(phaseOneGroups.size() + phaseTwoGroups.size());
    const uint32_t algorithm = plan.useTwoServerEight
        ? static_cast<uint32_t>(AllGatherAlgorithm::TWO_SERVER_EIGHT)
        : static_cast<uint32_t>(AllGatherAlgorithm::DIRECT);

    for (size_t groupIndex = 0; groupIndex < phaseOneGroups.size(); ++groupIndex) {
        const ChannelGroup &group = phaseOneGroups[groupIndex];
        auto kernelArg = std::make_shared<ops_hccl::CcuAllGatherKernelArg>();
        kernelArg->rankSize = param.rankSize;
        kernelArg->rankId = param.myRank;
        kernelArg->algorithm = algorithm;
        kernelArg->phase =
            static_cast<uint32_t>(ops_hccl::AllGatherKernelPhase::DISTRIBUTE);
        kernelArg->handleLocalCopy = groupIndex == localCopyGroup ? 1U : 0U;
        kernelArg->channelCount = static_cast<uint32_t>(group.channels.size());
        for (size_t channelIndex = 0; channelIndex < group.channels.size(); ++channelIndex) {
            kernelArg->channels[channelIndex] = group.channels[channelIndex];
            kernelArg->peerRanks[channelIndex] = group.peerLinks[channelIndex].rank;
            kernelArg->transferModes[channelIndex] =
                group.peerLinks[channelIndex].transferMode;
        }
        registeredArgs.push_back(kernelArg);
    }

    for (const ChannelGroup &group : phaseTwoGroups) {
        auto kernelArg = std::make_shared<ops_hccl::CcuAllGatherKernelArg>();
        kernelArg->rankSize = param.rankSize;
        kernelArg->rankId = param.myRank;
        kernelArg->algorithm = algorithm;
        kernelArg->phase = static_cast<uint32_t>(ops_hccl::AllGatherKernelPhase::RELAY);
        kernelArg->handleLocalCopy = 0;
        kernelArg->channelCount = static_cast<uint32_t>(group.channels.size());
        for (size_t channelIndex = 0; channelIndex < group.channels.size(); ++channelIndex) {
            kernelArg->channels[channelIndex] = group.channels[channelIndex];
            kernelArg->peerRanks[channelIndex] = group.peerLinks[channelIndex].rank;
            kernelArg->transferModes[channelIndex] =
                static_cast<uint32_t>(ops_hccl::AllGatherTransferMode::FULL_CHUNK);
        }
        registeredArgs.push_back(kernelArg);
    }

    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));
    std::vector<CcuKernelHandle> kernelHandles;
    kernelHandles.reserve(registeredArgs.size());
    for (size_t kernelIndex = 0; kernelIndex < registeredArgs.size(); ++kernelIndex) {
        const auto &kernelArg = registeredArgs[kernelIndex];
        CcuKernelHandle kernelHandle{};
        const void *kernelArgs[] = {kernelArg.get()};
        const char *kernelName =
            kernelArg->phase ==
                    static_cast<uint32_t>(ops_hccl::AllGatherKernelPhase::DISTRIBUTE)
                ? "CcuAllGatherDistributeKernel"
                : "CcuAllGatherRelayKernel";
        CcuResult registerRet = HcommCcuKernelRegister(insHandle, CCU_DIE_ID,
            kernelName, reinterpret_cast<void *>(ops_hccl::CcuKernel),
            kernelArgs, 1, &kernelHandle);
        if (registerRet != CCU_SUCCESS) {
            (void)HcommCcuKernelRegisterEnd(insHandle);
            HCCL_ERROR("[AllGather] kernel register %zu failed, ret %d",
                kernelIndex, registerRet);
            return ConvertCcuToHccl(registerRet);
        }
        kernelHandles.push_back(kernelHandle);
    }
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));

    resource.ccuKernels = kernelHandles;
    resource.phaseOneKernelCount = static_cast<uint32_t>(phaseOneGroups.size());
    resource.phaseTwoKernelCount = static_cast<uint32_t>(phaseTwoGroups.size());
    resource.threads.clear();
    const size_t maxPhaseKernelCount =
        std::max(phaseOneGroups.size(), phaseTwoGroups.size());
    if (maxPhaseKernelCount > 1) {
        resource.threads.resize(maxPhaseKernelCount - 1);
        CHK_RET(HcclThreadAcquire(comm, CommEngine::COMM_ENGINE_CCU,
            static_cast<uint32_t>(resource.threads.size()), THREAD_NOTIFY_NUM,
            resource.threads.data()));
    }
    resource.algorithm = algorithm;
    resource.partnerRank =
        plan.useTwoServerEight ? plan.incomingRelayRank : INVALID_VALUE_RANKID;
    return HCCL_SUCCESS;
}

HcclResult CreatePersistentResources(HcclComm comm, const OpParam &param, AlgResourceCtx &resource)
{
    void *cclBuffer = nullptr;
    uint64_t cclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBuffer, &cclBufferSize));
    resource.localBuffer = CommBuffer{cclBuffer, cclBufferSize};

    std::vector<uint32_t> layers;
    CHK_RET(GetNetworkLayers(comm, layers));
    TopologyPlan plan{};
    CHK_RET(BuildTopologyPlan(comm, param, layers, plan));

    std::vector<PeerLink> peerLinks;
    CHK_RET(BuildPeerLinks(comm, param, layers, plan, peerLinks));

    std::vector<ChannelHandle> channels;
    CHK_RET(AcquireChannels(comm, peerLinks, channels));
    std::vector<ChannelGroup> phaseOneGroups;
    CHK_RET(GroupChannelsByDie(peerLinks, channels, phaseOneGroups));

    std::vector<ChannelGroup> phaseTwoGroups;
    if (plan.useTwoServerEight) {
        std::vector<PeerLink> localPeerLinks;
        std::vector<ChannelHandle> localChannels;
        localPeerLinks.reserve(7);
        localChannels.reserve(7);
        for (size_t index = 0; index < peerLinks.size(); ++index) {
            if (peerLinks[index].isLocalPeer) {
                localPeerLinks.push_back(peerLinks[index]);
                localChannels.push_back(channels[index]);
            }
        }
        CHK_PRT_RET(localPeerLinks.size() != 7,
            HCCL_ERROR("[AllGather] expected seven local peers for 2x8 topology, got %zu",
                localPeerLinks.size()),
            HCCL_E_INTERNAL);
        CHK_RET(GroupChannelsByDie(localPeerLinks, localChannels, phaseTwoGroups));
    }

    CHK_RET(RegisterKernels(
        comm, param, phaseOneGroups, phaseTwoGroups, plan, resource));
    return HCCL_SUCCESS;
}

HcclResult LoadOrCreateContext(HcclComm comm, OpParam &param)
{
    void *context = nullptr;
    uint64_t contextSize = 0;
    if (HcclEngineCtxGet(
            comm, param.tag, CommEngine::COMM_ENGINE_CCU, &context, &contextSize) == HCCL_SUCCESS) {
        CHK_PRT_RET(context == nullptr || contextSize == 0,
            HCCL_ERROR("[AllGather] cached engine context is invalid"), HCCL_E_INTERNAL);
        param.resCtx = context;
        param.ctxSize = contextSize;
        return HCCL_SUCCESS;
    }

    AlgResourceCtx resource{};
    CHK_RET(CreatePersistentResources(comm, param, resource));
    std::vector<char> sequence = resource.Serialize();
    CHK_PRT_RET(sequence.empty(),
        HCCL_ERROR("[AllGather] serialized context is empty"), HCCL_E_INTERNAL);

    param.ctxSize = static_cast<uint64_t>(sequence.size());
    CHK_RET(HcclEngineCtxCreate(
        comm, param.tag, CommEngine::COMM_ENGINE_CCU, param.ctxSize, &param.resCtx));
    CHK_RET(HcclEngineCtxCopy(
        comm, CommEngine::COMM_ENGINE_CCU, param.tag, sequence.data(), param.ctxSize, 0));
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
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("[AllGather] only FP32 is supported"), HCCL_E_NOT_SUPPORT);

    OpParam param{};
    int tagLength = std::snprintf(
        param.tag, sizeof(param.tag), "%s", "hccl_custom_allgather_mixed_v3");
    CHK_PRT_RET(tagLength <= 0 || static_cast<size_t>(tagLength) >= sizeof(param.tag),
        HCCL_ERROR("[AllGather] failed to set operation tag"), HCCL_E_INTERNAL);
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = sendCount;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || param.myRank >= param.rankSize,
        HCCL_ERROR("[AllGather] invalid rank %u of %u", param.myRank, param.rankSize), HCCL_E_PARA);
    CHK_PRT_RET(sendCount > std::numeric_limits<uint64_t>::max() / sizeof(float),
        HCCL_ERROR("[AllGather] input byte size overflows"), HCCL_E_PARA);
    const uint64_t inputBytes = sendCount * sizeof(float);
    CHK_PRT_RET(inputBytes > std::numeric_limits<uint64_t>::max() / param.rankSize,
        HCCL_ERROR("[AllGather] output byte size overflows"), HCCL_E_PARA);
    const uint64_t outputBytes = inputBytes * param.rankSize;

    HcclDfxOpInfo dfxInfo{};
    dfxInfo.opMode = 1;
    dfxInfo.opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER);
    dfxInfo.reduceOp = static_cast<uint32_t>(HcclReduceOp::HCCL_REDUCE_RESERVED);
    dfxInfo.dataType = static_cast<uint32_t>(dataType);
    dfxInfo.dataCount = sendCount;
    dfxInfo.engine = CommEngine::COMM_ENGINE_CCU;
    dfxInfo.inputMemAddr = reinterpret_cast<uint64_t>(sendBuf);
    dfxInfo.inputMemSize = inputBytes;
    dfxInfo.outputMemAddr = reinterpret_cast<uint64_t>(recvBuf);
    dfxInfo.outputMemSize = outputBytes;
    int algTagLength = std::snprintf(
        dfxInfo.algTag, sizeof(dfxInfo.algTag), "%s", "hccl_custom_allgather_mixed_v3");
    CHK_PRT_RET(algTagLength <= 0 || static_cast<size_t>(algTagLength) >= sizeof(dfxInfo.algTag),
        HCCL_ERROR("[AllGather] failed to set DFX algorithm tag"), HCCL_E_INTERNAL);
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    if (sendCount == 0) {
        return HCCL_SUCCESS;
    }

    CHK_RET(HcclThreadAcquireWithStream(
        comm, CommEngine::COMM_ENGINE_CCU, stream, THREAD_NOTIFY_NUM, &param.cpuThread));
    CHK_RET(LoadOrCreateContext(comm, param));
    CHK_RET(ops_hccl::ExecOp(param));
    return HCCL_SUCCESS;
}
