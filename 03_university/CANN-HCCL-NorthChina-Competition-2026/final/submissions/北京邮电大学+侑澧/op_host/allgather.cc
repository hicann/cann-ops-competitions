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
#include <vector>

#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>

#include "ccu_kernel.h"
#include "ccu_launch.h"
#include "hccl_ccu_res.h"

#include "common.h"
#include "custom.h"
#include "exec_op.h"
#include "hccl.h"
#include "log.h"

namespace {

constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;
constexpr uint32_t DIE_NUM = 2;
constexpr uint32_t PIPELINE_RANK_SIZE = 16;
constexpr uint32_t PIPELINE_SERVER_SIZE = 8;
constexpr uint32_t ASYMMETRIC_RANK_SIZE = 12;
constexpr uint32_t ASYMMETRIC_LARGE_SERVER_SIZE = 8;
constexpr uint32_t ASYMMETRIC_SMALL_SERVER_SIZE = 4;
constexpr uint64_t PIPELINE_THRESHOLD = 512ULL * 1024;
constexpr uint64_t PIPELINE_MAX_DATA_SIZE = 2ULL * MAX_DATA_SIZE;

struct ChannelGroup {
    std::vector<ChannelHandle> channels;
};

struct PipelinePlan {
    bool enabled = false;
    uint32_t intraLayer = 0;
    uint32_t interLayer = 0;
    uint32_t pairRank = INVALID_VALUE_RANKID;
    std::vector<uint32_t> localRanks;
    std::vector<uint32_t> intraRanks;
};

struct PipelineChannels {
    std::vector<ChannelHandle> intra;
    std::vector<ChannelHandle> inter;
};

struct AsymmetricPlan {
    bool enabled = false;
    uint32_t intraLayer = 0;
    uint32_t interLayer = 0;
    uint32_t closSendCount = 0;
    uint32_t meshSendCount = 0;
    std::vector<uint32_t> localRanks;
    std::vector<uint32_t> pairRanks;
    std::vector<uint32_t> relayRanks;
};

struct AsymmetricChannels {
    std::vector<ChannelHandle> intra;
    std::vector<ChannelHandle> inter;
};

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

HcclResult BuildChannelDescAtLayer(
    HcclComm comm, uint32_t netLayer, uint32_t myRank, uint32_t remoteRank, HcclChannelDesc &desc)
{
    const CommProtocol protocols[] = {
        CommProtocol::COMM_PROTOCOL_UBC_CTP,
        CommProtocol::COMM_PROTOCOL_UBC_TP,
    };

    CommLink *linkList = nullptr;
    uint32_t listSize = 0;
    HcclResult ret = HcclRankGraphGetLinks(comm, netLayer, myRank, remoteRank, &linkList, &listSize);
    if (ret != HCCL_SUCCESS || linkList == nullptr || listSize == 0) {
        return HCCL_E_NOT_FOUND;
    }
    for (CommProtocol protocol : protocols) {
        for (uint32_t linkIdx = 0; linkIdx < listSize; ++linkIdx) {
            if (linkList[linkIdx].linkAttr.linkProtocol == protocol) {
                CHK_RET(HcclChannelDescInit(&desc, 1));
                FillChannelDesc(remoteRank, linkList[linkIdx], desc);
                return HCCL_SUCCESS;
            }
        }
    }
    return HCCL_E_NOT_FOUND;
}

HcclResult BuildChannelDesc(HcclComm comm, uint32_t myRank, uint32_t remoteRank, HcclChannelDesc &desc)
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    CHK_PRT_RET(netLayers == nullptr || netLayerNum == 0,
        HCCL_ERROR("[BuildChannelDesc] Rank graph has no network layer."),
        HCCL_E_NOT_FOUND);

    for (uint32_t layerIdx = 0; layerIdx < netLayerNum; ++layerIdx) {
        if (BuildChannelDescAtLayer(
                comm, netLayers[layerIdx], myRank, remoteRank, desc) == HCCL_SUCCESS) {
            return HCCL_SUCCESS;
        }
    }

    HCCL_ERROR("[BuildChannelDesc] No CCU link between rank[%u] and rank[%u].", myRank, remoteRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult GetCurrentServerRanks(HcclComm comm, uint32_t intraLayer, uint32_t myRank,
    uint32_t rankSize, std::vector<uint32_t> &serverRanks)
{
    uint32_t *topoInsts = nullptr;
    uint32_t topoInstNum = 0;
    HcclResult ret = HcclRankGraphGetTopoInstsByLayer(
        comm, intraLayer, &topoInsts, &topoInstNum);
    if (ret != HCCL_SUCCESS || topoInsts == nullptr || topoInstNum == 0) {
        return HCCL_E_NOT_FOUND;
    }

    serverRanks.push_back(myRank);
    for (uint32_t instIdx = 0; instIdx < topoInstNum; ++instIdx) {
        uint32_t *ranks = nullptr;
        uint32_t rankNum = 0;
        ret = HcclRankGraphGetRanksByTopoInst(
            comm, intraLayer, topoInsts[instIdx], &ranks, &rankNum);
        if (ret != HCCL_SUCCESS || ranks == nullptr || rankNum == 0) {
            return HCCL_E_NOT_FOUND;
        }
        for (uint32_t rankIdx = 0; rankIdx < rankNum; ++rankIdx) {
            uint32_t rank = ranks[rankIdx];
            if (rank < rankSize &&
                std::find(serverRanks.begin(), serverRanks.end(), rank) == serverRanks.end()) {
                serverRanks.push_back(rank);
            }
        }
    }
    std::sort(serverRanks.begin(), serverRanks.end());
    return HCCL_SUCCESS;
}

HcclResult GetChannelDie(
    HcclComm comm, uint32_t myRank, HcclChannelDesc &desc, uint32_t &dieId)
{
    return HcclRankGraphGetEndpointInfo(comm, myRank, &desc.localEndpoint,
        ENDPOINT_ATTR_DIE_ID, sizeof(dieId), &dieId);
}

HcclResult SelectPipelinePlan(
    HcclComm comm, uint64_t dataSize, const OpParam &param, PipelinePlan &plan)
{
    if (param.rankSize != PIPELINE_RANK_SIZE || dataSize <= PIPELINE_THRESHOLD ||
        dataSize > PIPELINE_MAX_DATA_SIZE) {
        return HCCL_SUCCESS;
    }

    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    if (netLayers == nullptr || netLayerNum < 2 ||
        std::find(netLayers, netLayers + netLayerNum, 0U) == netLayers + netLayerNum) {
        return HCCL_SUCCESS;
    }

    uint32_t *serverSizes = nullptr;
    uint32_t serverCount = 0;
    CHK_RET(HcclRankGraphGetInstSizeListByLayer(
        comm, 0, &serverSizes, &serverCount));
    if (serverSizes == nullptr || serverCount != 2 ||
        serverSizes[0] != PIPELINE_SERVER_SIZE || serverSizes[1] != PIPELINE_SERVER_SIZE) {
        return HCCL_SUCCESS;
    }

    std::vector<uint32_t> localRanks;
    if (GetCurrentServerRanks(comm, 0, param.myRank, param.rankSize, localRanks) != HCCL_SUCCESS ||
        localRanks.size() != PIPELINE_SERVER_SIZE) {
        return HCCL_SUCCESS;
    }
    std::vector<uint32_t> remoteRanks;
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (std::find(localRanks.begin(), localRanks.end(), rank) == localRanks.end()) {
            remoteRanks.push_back(rank);
        }
    }
    if (remoteRanks.size() != PIPELINE_SERVER_SIZE) {
        return HCCL_SUCCESS;
    }

    auto localIt = std::find(localRanks.begin(), localRanks.end(), param.myRank);
    if (localIt == localRanks.end()) {
        return HCCL_SUCCESS;
    }
    uint32_t localIndex = static_cast<uint32_t>(localIt - localRanks.begin());
    uint32_t pairRank = remoteRanks[localIndex];

    for (uint32_t rank : localRanks) {
        if (rank == param.myRank) {
            continue;
        }
        HcclChannelDesc desc;
        if (BuildChannelDescAtLayer(comm, 0, param.myRank, rank, desc) != HCCL_SUCCESS) {
            return HCCL_SUCCESS;
        }
        uint32_t dieId = DIE_NUM;
        CHK_RET(GetChannelDie(comm, param.myRank, desc, dieId));
        if (dieId >= DIE_NUM) {
            return HCCL_SUCCESS;
        }
    }

    uint32_t interLayer = 0;
    uint32_t interDie = DIE_NUM;
    bool foundInter = false;
    for (uint32_t layerIdx = 0; layerIdx < netLayerNum; ++layerIdx) {
        if (netLayers[layerIdx] == 0) {
            continue;
        }
        HcclChannelDesc desc;
        if (BuildChannelDescAtLayer(
                comm, netLayers[layerIdx], param.myRank, pairRank, desc) != HCCL_SUCCESS) {
            continue;
        }
        CHK_RET(GetChannelDie(comm, param.myRank, desc, interDie));
        if (interDie < DIE_NUM) {
            interLayer = netLayers[layerIdx];
            foundInter = true;
            break;
        }
    }
    if (!foundInter) {
        return HCCL_SUCCESS;
    }

    plan.enabled = true;
    plan.intraLayer = 0;
    plan.interLayer = interLayer;
    plan.pairRank = pairRank;
    plan.localRanks = localRanks;
    for (uint32_t rank : localRanks) {
        if (rank != param.myRank) {
            plan.intraRanks.push_back(rank);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult SelectAsymmetricPlan(
    HcclComm comm, uint64_t dataSize, const OpParam &param, AsymmetricPlan &plan)
{
    if (param.rankSize != ASYMMETRIC_RANK_SIZE || dataSize <= PIPELINE_THRESHOLD ||
        dataSize > PIPELINE_MAX_DATA_SIZE) {
        return HCCL_SUCCESS;
    }

    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    if (netLayers == nullptr || netLayerNum < 2 ||
        std::find(netLayers, netLayers + netLayerNum, 0U) == netLayers + netLayerNum) {
        return HCCL_SUCCESS;
    }

    uint32_t *serverSizes = nullptr;
    uint32_t serverCount = 0;
    CHK_RET(HcclRankGraphGetInstSizeListByLayer(comm, 0, &serverSizes, &serverCount));
    if (serverSizes == nullptr || serverCount != 2) {
        return HCCL_SUCCESS;
    }
    uint32_t minServerSize = std::min(serverSizes[0], serverSizes[1]);
    uint32_t maxServerSize = std::max(serverSizes[0], serverSizes[1]);
    if (minServerSize != ASYMMETRIC_SMALL_SERVER_SIZE ||
        maxServerSize != ASYMMETRIC_LARGE_SERVER_SIZE) {
        return HCCL_SUCCESS;
    }

    if (GetCurrentServerRanks(comm, 0, param.myRank, param.rankSize, plan.localRanks) != HCCL_SUCCESS) {
        return HCCL_SUCCESS;
    }
    std::vector<uint32_t> remoteRanks;
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (std::find(plan.localRanks.begin(), plan.localRanks.end(), rank) == plan.localRanks.end()) {
            remoteRanks.push_back(rank);
        }
    }
    uint32_t localSize = static_cast<uint32_t>(plan.localRanks.size());
    uint32_t remoteSize = static_cast<uint32_t>(remoteRanks.size());
    bool sizeMatch = (localSize == ASYMMETRIC_LARGE_SERVER_SIZE &&
                         remoteSize == ASYMMETRIC_SMALL_SERVER_SIZE) ||
        (localSize == ASYMMETRIC_SMALL_SERVER_SIZE &&
            remoteSize == ASYMMETRIC_LARGE_SERVER_SIZE);
    if (!sizeMatch) {
        plan.localRanks.clear();
        return HCCL_SUCCESS;
    }

    auto localIt = std::find(plan.localRanks.begin(), plan.localRanks.end(), param.myRank);
    if (localIt == plan.localRanks.end()) {
        plan.localRanks.clear();
        return HCCL_SUCCESS;
    }
    uint32_t localIndex = static_cast<uint32_t>(localIt - plan.localRanks.begin());
    plan.relayRanks.push_back(param.myRank);
    plan.closSendCount = 1;
    if (localSize == ASYMMETRIC_LARGE_SERVER_SIZE) {
        uint32_t column = localIndex % ASYMMETRIC_SMALL_SERVER_SIZE;
        plan.pairRanks.push_back(remoteRanks[column]);
        if (localIndex < ASYMMETRIC_SMALL_SERVER_SIZE) {
            plan.relayRanks.push_back(remoteRanks[column]);
            plan.meshSendCount = 1;
        }
    } else {
        plan.pairRanks.push_back(remoteRanks[localIndex]);
        plan.pairRanks.push_back(remoteRanks[localIndex + ASYMMETRIC_SMALL_SERVER_SIZE]);
        plan.relayRanks.insert(plan.relayRanks.end(), plan.pairRanks.begin(), plan.pairRanks.end());
        plan.meshSendCount = 2;
    }

    uint32_t intraDie = DIE_NUM;
    for (uint32_t rank : plan.localRanks) {
        if (rank == param.myRank) {
            continue;
        }
        HcclChannelDesc desc;
        if (BuildChannelDescAtLayer(comm, 0, param.myRank, rank, desc) != HCCL_SUCCESS) {
            plan.localRanks.clear();
            return HCCL_SUCCESS;
        }
        uint32_t dieId = DIE_NUM;
        CHK_RET(GetChannelDie(comm, param.myRank, desc, dieId));
        if (dieId >= DIE_NUM || (intraDie < DIE_NUM && dieId != intraDie)) {
            plan.localRanks.clear();
            return HCCL_SUCCESS;
        }
        intraDie = dieId;
    }

    for (uint32_t layerIdx = 0; layerIdx < netLayerNum; ++layerIdx) {
        if (netLayers[layerIdx] == 0) {
            continue;
        }
        uint32_t interDie = DIE_NUM;
        bool layerMatches = true;
        for (uint32_t pairRank : plan.pairRanks) {
            HcclChannelDesc desc;
            if (BuildChannelDescAtLayer(
                    comm, netLayers[layerIdx], param.myRank, pairRank, desc) != HCCL_SUCCESS) {
                layerMatches = false;
                break;
            }
            uint32_t dieId = DIE_NUM;
            CHK_RET(GetChannelDie(comm, param.myRank, desc, dieId));
            if (dieId >= DIE_NUM || (interDie < DIE_NUM && dieId != interDie)) {
                layerMatches = false;
                break;
            }
            interDie = dieId;
        }
        if (layerMatches) {
            plan.interLayer = netLayers[layerIdx];
            plan.enabled = true;
            return HCCL_SUCCESS;
        }
    }

    plan.localRanks.clear();
    plan.pairRanks.clear();
    plan.relayRanks.clear();
    return HCCL_SUCCESS;
}

HcclResult AcquireChannelGroups(
    HcclComm comm, const OpParam &param, CommEngine engine, std::vector<ChannelGroup> &groups)
{
    std::vector<HcclChannelDesc> descs;
    std::vector<uint32_t> dieIds;
    descs.reserve(param.rankSize - 1);
    dieIds.reserve(param.rankSize - 1);
    groups.resize(DIE_NUM);

    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }
        HcclChannelDesc desc;
        CHK_RET(BuildChannelDesc(comm, param.myRank, remoteRank, desc));
        uint32_t dieId = 0;
        CHK_RET(HcclRankGraphGetEndpointInfo(comm, param.myRank, &desc.localEndpoint,
            ENDPOINT_ATTR_DIE_ID, sizeof(dieId), &dieId));
        CHK_PRT_RET(dieId >= DIE_NUM,
            HCCL_ERROR("[AcquireChannelGroups] Invalid dieId[%u] for remote rank[%u].", dieId, remoteRank),
            HCCL_E_INTERNAL);
        descs.push_back(desc);
        dieIds.push_back(dieId);
    }

    std::vector<ChannelHandle> channels(descs.size());
    CHK_RET(HcclChannelAcquire(
        comm, engine, descs.data(), static_cast<uint32_t>(descs.size()), channels.data()));
    uint32_t channelCount = static_cast<uint32_t>(channels.size());
    for (uint32_t channelIdx = 0; channelIdx < channelCount; ++channelIdx) {
        groups[dieIds[channelIdx]].channels.push_back(channels[channelIdx]);
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterCcuKernels(HcclComm comm, const OpParam &param, const std::vector<ChannelGroup> &groups,
    AlgResourceCtx &resCtx)
{
    uint32_t groupCount = static_cast<uint32_t>(groups.size());
    uint32_t selfGroupIdx = groupCount;
    uint32_t minChannelCount = MAX_RANK_SIZE;
    for (uint32_t groupIdx = 0; groupIdx < groupCount; ++groupIdx) {
        uint32_t channelCount = static_cast<uint32_t>(groups[groupIdx].channels.size());
        if (channelCount != 0 && channelCount < minChannelCount) {
            minChannelCount = channelCount;
            selfGroupIdx = groupIdx;
        }
    }
    CHK_PRT_RET(selfGroupIdx == groupCount,
        HCCL_ERROR("[RegisterCcuKernels] No channel group can handle the local copy."),
        HCCL_E_INTERNAL);

    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1,
        HCCL_ERROR("[RegisterCcuKernels] Expected one CCU instance, got [%u].", insNum),
        HCCL_E_INTERNAL);

    CcuResult ccuRet = HcommCcuKernelRegisterStart(insHandle);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS,
        HCCL_ERROR("[RegisterCcuKernels] RegisterStart failed, ret[%d].", ccuRet),
        ConvertCcuToHccl(ccuRet));

    std::vector<std::shared_ptr<ops_hccl::CcuKernelArgAllGather>> kernelArgs;
    for (uint32_t groupIdx = 0; groupIdx < groupCount; ++groupIdx) {
        const ChannelGroup &group = groups[groupIdx];
        if (group.channels.empty()) {
            continue;
        }

        auto kernelArg = std::make_shared<ops_hccl::CcuKernelArgAllGather>();
        kernelArg->rankSize = param.rankSize;
        kernelArg->rankId = param.myRank;
        kernelArg->handleSelf = groupIdx == selfGroupIdx ? 1U : 0U;
        kernelArg->channelCount = static_cast<uint32_t>(group.channels.size());
        for (uint32_t channelIdx = 0; channelIdx < kernelArg->channelCount; ++channelIdx) {
            kernelArg->channels[channelIdx] = group.channels[channelIdx];
        }
        kernelArgs.push_back(kernelArg);

        CcuKernelHandle kernelHandle = 0;
        const void *args[] = {kernelArg.get()};
        ccuRet = HcommCcuKernelRegister(insHandle, 0, "CcuKernel",
            reinterpret_cast<void *>(ops_hccl::CcuKernel), args, 1, &kernelHandle);
        CHK_PRT_RET(ccuRet != CCU_SUCCESS,
            HCCL_ERROR("[RegisterCcuKernels] Register failed, ret[%d].", ccuRet),
            ConvertCcuToHccl(ccuRet));
        resCtx.ccuKernels.push_back(kernelHandle);
    }

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS,
        HCCL_ERROR("[RegisterCcuKernels] RegisterEnd failed, ret[%d].", ccuRet),
        ConvertCcuToHccl(ccuRet));

    CHK_PRT_RET(resCtx.ccuKernels.empty(),
        HCCL_ERROR("[RegisterCcuKernels] No CCU kernel was registered."),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult AcquirePipelineChannels(HcclComm comm, const OpParam &param, const PipelinePlan &plan,
    CommEngine engine, PipelineChannels &pipelineChannels)
{
    std::vector<HcclChannelDesc> descs;
    for (uint32_t rank : plan.intraRanks) {
        HcclChannelDesc desc;
        CHK_RET(BuildChannelDescAtLayer(
            comm, plan.intraLayer, param.myRank, rank, desc));
        descs.push_back(desc);
    }
    uint32_t intraCount = static_cast<uint32_t>(descs.size());

    HcclChannelDesc interDesc;
    CHK_RET(BuildChannelDescAtLayer(
        comm, plan.interLayer, param.myRank, plan.pairRank, interDesc));
    descs.push_back(interDesc);

    std::vector<ChannelHandle> channels(descs.size());
    CHK_RET(HcclChannelAcquire(
        comm, engine, descs.data(), static_cast<uint32_t>(descs.size()), channels.data()));
    pipelineChannels.intra.assign(channels.begin(), channels.begin() + intraCount);
    pipelineChannels.inter.assign(channels.begin() + intraCount, channels.end());
    return HCCL_SUCCESS;
}

HcclResult AcquireAsymmetricChannels(HcclComm comm, const OpParam &param, const AsymmetricPlan &plan,
    CommEngine engine, AsymmetricChannels &channels)
{
    std::vector<HcclChannelDesc> descs;
    for (uint32_t rank : plan.localRanks) {
        if (rank == param.myRank) {
            continue;
        }
        HcclChannelDesc desc;
        CHK_RET(BuildChannelDescAtLayer(comm, plan.intraLayer, param.myRank, rank, desc));
        descs.push_back(desc);
    }
    uint32_t intraCount = static_cast<uint32_t>(descs.size());

    for (uint32_t pairRank : plan.pairRanks) {
        HcclChannelDesc desc;
        CHK_RET(BuildChannelDescAtLayer(comm, plan.interLayer, param.myRank, pairRank, desc));
        descs.push_back(desc);
    }

    std::vector<ChannelHandle> handles(descs.size());
    CHK_RET(HcclChannelAcquire(
        comm, engine, descs.data(), static_cast<uint32_t>(descs.size()), handles.data()));
    channels.intra.assign(handles.begin(), handles.begin() + intraCount);
    channels.inter.assign(handles.begin() + intraCount, handles.end());
    return HCCL_SUCCESS;
}

HcclResult RegisterPipelineKernels(HcclComm comm, const OpParam &param, const PipelinePlan &plan,
    const PipelineChannels &channels, AlgResourceCtx &resCtx)
{
    CHK_PRT_RET(channels.intra.size() != PIPELINE_SERVER_SIZE - 1 || channels.inter.size() != 1,
        HCCL_ERROR("[RegisterPipelineKernels] Invalid channel counts, intra[%u], inter[%u].",
            static_cast<uint32_t>(channels.intra.size()), static_cast<uint32_t>(channels.inter.size())),
        HCCL_E_INTERNAL);

    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1,
        HCCL_ERROR("[RegisterPipelineKernels] Expected one CCU instance, got [%u].", insNum),
        HCCL_E_INTERNAL);

    CcuResult ccuRet = HcommCcuKernelRegisterStart(insHandle);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS,
        HCCL_ERROR("[RegisterPipelineKernels] RegisterStart failed, ret[%d].", ccuRet),
        ConvertCcuToHccl(ccuRet));

    auto intraArg = std::make_shared<ops_hccl::CcuKernelArgPipelineIntra>();
    intraArg->rankSize = param.rankSize;
    intraArg->rankId = param.myRank;
    intraArg->pairRank = plan.pairRank;
    intraArg->channelCount = static_cast<uint32_t>(channels.intra.size());
    for (uint32_t channelIdx = 0; channelIdx < intraArg->channelCount; ++channelIdx) {
        intraArg->channels[channelIdx] = channels.intra[channelIdx];
    }
    CcuKernelHandle intraKernel = 0;
    const void *intraArgs[] = {intraArg.get()};
    ccuRet = HcommCcuKernelRegister(insHandle, 0, "CcuPipelineIntraKernel",
        reinterpret_cast<void *>(ops_hccl::CcuPipelineIntraKernel), intraArgs, 1, &intraKernel);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS,
        HCCL_ERROR("[RegisterPipelineKernels] Intra register failed, ret[%d].", ccuRet),
        ConvertCcuToHccl(ccuRet));
    resCtx.ccuKernels.push_back(intraKernel);

    auto interArg = std::make_shared<ops_hccl::CcuKernelArgPipelineInter>();
    interArg->rankSize = param.rankSize;
    interArg->rankId = param.myRank;
    interArg->localRankCount = static_cast<uint32_t>(plan.localRanks.size());
    interArg->channelCount = 1;
    interArg->channels[0] = channels.inter[0];
    for (uint32_t rankIdx = 0; rankIdx < interArg->localRankCount; ++rankIdx) {
        interArg->localRanks[rankIdx] = plan.localRanks[rankIdx];
    }
    CcuKernelHandle interKernel = 0;
    const void *interArgs[] = {interArg.get()};
    ccuRet = HcommCcuKernelRegister(insHandle, 0, "CcuPipelineInterKernel",
        reinterpret_cast<void *>(ops_hccl::CcuPipelineInterKernel), interArgs, 1, &interKernel);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS,
        HCCL_ERROR("[RegisterPipelineKernels] Inter register failed, ret[%d].", ccuRet),
        ConvertCcuToHccl(ccuRet));
    resCtx.ccuKernels.push_back(interKernel);

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS,
        HCCL_ERROR("[RegisterPipelineKernels] RegisterEnd failed, ret[%d].", ccuRet),
        ConvertCcuToHccl(ccuRet));
    return HCCL_SUCCESS;
}

HcclResult RegisterAsymmetricKernels(HcclComm comm, const OpParam &param, const AsymmetricPlan &plan,
    const AsymmetricChannels &channels, AlgResourceCtx &resCtx)
{
    uint32_t expectedIntraCount = static_cast<uint32_t>(plan.localRanks.size()) - 1;
    CHK_PRT_RET(channels.intra.size() != expectedIntraCount ||
            channels.inter.size() != plan.pairRanks.size(),
        HCCL_ERROR("[RegisterAsymmetricKernels] Invalid channel counts, intra[%u], inter[%u].",
            static_cast<uint32_t>(channels.intra.size()),
            static_cast<uint32_t>(channels.inter.size())),
        HCCL_E_INTERNAL);

    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1,
        HCCL_ERROR("[RegisterAsymmetricKernels] Expected one CCU instance, got [%u].", insNum),
        HCCL_E_INTERNAL);

    CcuResult ccuRet = HcommCcuKernelRegisterStart(insHandle);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS,
        HCCL_ERROR("[RegisterAsymmetricKernels] RegisterStart failed, ret[%d].", ccuRet),
        ConvertCcuToHccl(ccuRet));

    auto intraArg = std::make_shared<ops_hccl::CcuKernelArgAsymmetricIntra>();
    intraArg->rankSize = param.rankSize;
    intraArg->rankId = param.myRank;
    intraArg->relayRankCount = static_cast<uint32_t>(plan.relayRanks.size());
    intraArg->channelCount = static_cast<uint32_t>(channels.intra.size());
    for (uint32_t channelIdx = 0; channelIdx < intraArg->channelCount; ++channelIdx) {
        intraArg->channels[channelIdx] = channels.intra[channelIdx];
    }
    for (uint32_t rankIdx = 0; rankIdx < intraArg->relayRankCount; ++rankIdx) {
        intraArg->relayRanks[rankIdx] = plan.relayRanks[rankIdx];
    }
    CcuKernelHandle intraKernel = 0;
    const void *intraArgs[] = {intraArg.get()};
    ccuRet = HcommCcuKernelRegister(insHandle, 0, "CcuAsymmetricIntraKernel",
        reinterpret_cast<void *>(ops_hccl::CcuAsymmetricIntraKernel), intraArgs, 1, &intraKernel);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS,
        HCCL_ERROR("[RegisterAsymmetricKernels] Intra register failed, ret[%d].", ccuRet),
        ConvertCcuToHccl(ccuRet));
    resCtx.ccuKernels.push_back(intraKernel);

    auto interArg = std::make_shared<ops_hccl::CcuKernelArgAsymmetricInter>();
    interArg->rankSize = param.rankSize;
    interArg->rankId = param.myRank;
    interArg->localRankCount = static_cast<uint32_t>(plan.localRanks.size());
    interArg->closSendCount = plan.closSendCount;
    interArg->meshSendCount = plan.meshSendCount;
    interArg->channelCount = static_cast<uint32_t>(channels.inter.size());
    for (uint32_t channelIdx = 0; channelIdx < interArg->channelCount; ++channelIdx) {
        interArg->channels[channelIdx] = channels.inter[channelIdx];
    }
    for (uint32_t rankIdx = 0; rankIdx < interArg->localRankCount; ++rankIdx) {
        interArg->localRanks[rankIdx] = plan.localRanks[rankIdx];
    }
    CcuKernelHandle interKernel = 0;
    const void *interArgs[] = {interArg.get()};
    ccuRet = HcommCcuKernelRegister(insHandle, 0, "CcuAsymmetricInterKernel",
        reinterpret_cast<void *>(ops_hccl::CcuAsymmetricInterKernel), interArgs, 1, &interKernel);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS,
        HCCL_ERROR("[RegisterAsymmetricKernels] Inter register failed, ret[%d].", ccuRet),
        ConvertCcuToHccl(ccuRet));
    resCtx.ccuKernels.push_back(interKernel);

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS,
        HCCL_ERROR("[RegisterAsymmetricKernels] RegisterEnd failed, ret[%d].", ccuRet),
        ConvertCcuToHccl(ccuRet));
    return HCCL_SUCCESS;
}

HcclResult CreateDirectResources(HcclComm comm, const OpParam &param, CommEngine engine, AlgResourceCtx &resCtx)
{
    void *cclBufferAddr = nullptr;
    uint64_t cclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    resCtx.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};
    resCtx.ccuThread = param.cpuThread;

    if (param.rankSize == 1) {
        resCtx.threads.push_back(param.cpuThread);
        return HCCL_SUCCESS;
    }

    std::vector<ChannelGroup> groups;
    CHK_RET(AcquireChannelGroups(comm, param, engine, groups));

    uint32_t kernelCount = 0;
    for (const ChannelGroup &group : groups) {
        kernelCount += group.channels.empty() ? 0U : 1U;
    }
    CHK_PRT_RET(kernelCount == 0 || kernelCount > DIE_NUM,
        HCCL_ERROR("[CreateCcuResources] Invalid kernel count[%u].", kernelCount),
        HCCL_E_INTERNAL);

    resCtx.threads.resize(kernelCount);
    resCtx.threads[0] = param.cpuThread;
    if (kernelCount > 1) {
        CHK_RET(HcclThreadAcquire(comm, engine, kernelCount - 1, 1, &resCtx.threads[1]));
    }

    CHK_RET(RegisterCcuKernels(comm, param, groups, resCtx));
    return HCCL_SUCCESS;
}

HcclResult CreatePipelineResources(HcclComm comm, const OpParam &param, const PipelinePlan &plan,
    CommEngine engine, AlgResourceCtx &resCtx)
{
    void *cclBufferAddr = nullptr;
    uint64_t cclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    resCtx.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};
    resCtx.ccuThread = param.cpuThread;
    resCtx.pipelineMode = 1;

    PipelineChannels channels;
    CHK_RET(AcquirePipelineChannels(comm, param, plan, engine, channels));
    resCtx.threads.resize(2);
    resCtx.threads[0] = param.cpuThread;
    CHK_RET(HcclThreadAcquire(comm, engine, 1, 1, &resCtx.threads[1]));
    CHK_RET(RegisterPipelineKernels(comm, param, plan, channels, resCtx));
    return HCCL_SUCCESS;
}

HcclResult CreateAsymmetricResources(HcclComm comm, const OpParam &param, const AsymmetricPlan &plan,
    CommEngine engine, AlgResourceCtx &resCtx)
{
    void *cclBufferAddr = nullptr;
    uint64_t cclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    resCtx.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};
    resCtx.ccuThread = param.cpuThread;
    resCtx.pipelineMode = 2;

    AsymmetricChannels channels;
    CHK_RET(AcquireAsymmetricChannels(comm, param, plan, engine, channels));
    resCtx.threads.resize(2);
    resCtx.threads[0] = param.cpuThread;
    CHK_RET(HcclThreadAcquire(comm, engine, 1, 1, &resCtx.threads[1]));
    CHK_RET(RegisterAsymmetricKernels(comm, param, plan, channels, resCtx));
    return HCCL_SUCCESS;
}

HcclResult CreateCcuResources(HcclComm comm, const OpParam &param, const PipelinePlan &plan,
    const AsymmetricPlan &asymmetricPlan, CommEngine engine, AlgResourceCtx &resCtx)
{
    if (plan.enabled) {
        return CreatePipelineResources(comm, param, plan, engine, resCtx);
    }
    if (asymmetricPlan.enabled) {
        return CreateAsymmetricResources(comm, param, asymmetricPlan, engine, resCtx);
    }
    return CreateDirectResources(comm, param, engine, resCtx);
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

    // 注册算子信息
    HcclDfxOpInfo dfxInfo;
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    // ==============================================
    // STEP 1: 解析拓扑信息
    // ==============================================
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE,
        HCCL_ERROR("[HcclAllGather] Unsupported rankSize[%u].", param.rankSize),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(SIZE_TABLE.find(dataType) == SIZE_TABLE.end(),
        HCCL_ERROR("[HcclAllGather] Unsupported dataType[%d].", static_cast<int32_t>(dataType)),
        HCCL_E_NOT_SUPPORT);
    uint64_t dataSize = sendCount * SIZE_TABLE.at(dataType);
    PipelinePlan pipelinePlan;
    CHK_RET(SelectPipelinePlan(comm, dataSize, param, pipelinePlan));
    AsymmetricPlan asymmetricPlan;
    if (!pipelinePlan.enabled) {
        CHK_RET(SelectAsymmetricPlan(comm, dataSize, param, asymmetricPlan));
    }

    const char *tag = "hccl_custom_allgather_v12_direct";
    if (pipelinePlan.enabled) {
        tag = "hccl_custom_allgather_v12_pipeline";
    } else if (asymmetricPlan.enabled) {
        tag = "hccl_custom_allgather_v12_asymmetric";
    }
    int tagRet = std::snprintf(param.tag, sizeof(param.tag), "%s", tag);
    CHK_PRT_RET(tagRet <= 0 || static_cast<size_t>(tagRet) >= sizeof(param.tag),
        HCCL_ERROR("[HcclAllGather] Failed to fill tag."),
        HCCL_E_INTERNAL);

    // ==============================================
    // STEP 2: 创建资源
    // ==============================================
    CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;

    // ==============================================
    // STEP 2.1: 申请用于 Host/Device 同步的通信资源
    // ==============================================
    // 将用户传入的 stream 转换为 CCU 通信引擎中的 thread，并申请 1 个 notify
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
        AlgResourceCtx resCtxHost;

        CHK_RET(CreateCcuResources(
            comm, param, pipelinePlan, asymmetricPlan, ccuEngine, resCtxHost));

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
