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

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <iterator>
#include <limits>
#include <memory>
#include <vector>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "exec_op.h"
#include "ccu_kernel.h"
#include "ccu_launch.h"
#include "hccl_ccu_res.h"

namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 1;
constexpr uint32_t KERNEL_DIE_ID = 0;
constexpr uint32_t MAX_KERNEL_DIE_COUNT = 2;
constexpr uint32_t FIRST_SERVER_SIZE = 8;
constexpr uint32_t FIRST_SERVER_START = 0;
constexpr uint32_t SECOND_SERVER_START = 8;
constexpr uint64_t SMALL_RECV_SIZE_LIMIT = 1024 * 1024;
constexpr const char *ALGORITHM_TAG = "hccl_custom_reducescatter_ccu";
constexpr const char *HIERARCHICAL_ALGORITHM_TAG = "hccl_custom_reducescatter_ccu_hier_small_single_stream";
constexpr const char *KERNEL_NAME = "CcuReduceScatterDeterministicKernel";
constexpr const char *HIER_LOCAL_KERNEL_NAME = "CcuReduceScatterHierLocalKernel";
constexpr const char *HIER_CROSS_KERNEL_NAME = "CcuReduceScatterHierCrossKernel";

struct ChannelGroup {
    uint32_t dieId = INVALID_VALUE_RANKID;
    std::vector<uint32_t> peerRanks;
    std::vector<HcclChannelDesc> channelDescs;
    std::vector<ChannelHandle> channelHandles;
};

struct ServerInfo {
    uint32_t start = 0;
    uint32_t size = 0;
    uint32_t remoteStart = 0;
    uint32_t remoteSize = 0;
};

bool IsSmallHierarchicalParam(const OpParam &param)
{
    // The stable rank-16 path is already competitive.  Limit this experiment
    // to the asymmetric 8+4 topology, whose small-message latency is the clear
    // remaining bottleneck.
    return param.rankSize == 12 && param.count <= SMALL_RECV_SIZE_LIMIT / sizeof(float);
}

HcclResult GetServerInfo(const OpParam &param, ServerInfo &serverInfo)
{
    if (param.rankSize != 12 && param.rankSize != 16) {
        return HCCL_E_NOT_SUPPORT;
    }
    if (param.myRank < SECOND_SERVER_START) {
        serverInfo.start = FIRST_SERVER_START;
        serverInfo.size = FIRST_SERVER_SIZE;
        serverInfo.remoteStart = SECOND_SERVER_START;
        serverInfo.remoteSize = param.rankSize - SECOND_SERVER_START;
    } else {
        serverInfo.start = SECOND_SERVER_START;
        serverInfo.size = param.rankSize - SECOND_SERVER_START;
        serverInfo.remoteStart = FIRST_SERVER_START;
        serverInfo.remoteSize = FIRST_SERVER_SIZE;
    }
    return HCCL_SUCCESS;
}

HcclResult CheckParameters(uint64_t recvCount, HcclDataType dataType, HcclReduceOp op)
{
    if (dataType != HCCL_DATA_TYPE_FP32) {
        HCCL_ERROR("Only float32 is supported, dataType[%d]", static_cast<int32_t>(dataType));
        return HCCL_E_NOT_SUPPORT;
    }
    if (op != HcclReduceOp::HCCL_REDUCE_SUM) {
        HCCL_ERROR("Only sum is supported, reduceOp[%d]", static_cast<int32_t>(op));
        return HCCL_E_NOT_SUPPORT;
    }
    if (recvCount > std::numeric_limits<uint64_t>::max() / sizeof(float)) {
        HCCL_ERROR("recvCount[%" PRIu64 "] causes byte-size overflow", recvCount);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
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

HcclResult QueryNetLayers(HcclComm comm, std::vector<uint32_t> &netLayers)
{
    uint32_t *rawNetLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &rawNetLayers, &netLayerNum));
    if (rawNetLayers == nullptr || netLayerNum == 0) {
        HCCL_ERROR("No network layer found");
        return HCCL_E_NOT_FOUND;
    }
    netLayers.assign(rawNetLayers, rawNetLayers + netLayerNum);
    std::sort(netLayers.begin(), netLayers.end());
    return HCCL_SUCCESS;
}

HcclResult QueryLinkDie(HcclComm comm, uint32_t localRank, const CommLink &link, uint32_t &dieId)
{
    EndpointAttrDieId endpointDie = INVALID_VALUE_RANKID;
    CHK_RET(HcclRankGraphGetEndpointInfo(
        comm, localRank, &link.srcEndpointDesc, EndpointAttr::ENDPOINT_ATTR_DIE_ID, sizeof(endpointDie), &endpointDie));
    if (endpointDie >= MAX_KERNEL_DIE_COUNT) {
        HCCL_ERROR("Invalid local die id[%u] for rank[%u]", endpointDie, localRank);
        return HCCL_E_NOT_SUPPORT;
    }
    dieId = endpointDie;
    return HCCL_SUCCESS;
}

HcclResult FindPreferredLink(HcclComm comm, const std::vector<uint32_t> &netLayers, uint32_t localRank,
    uint32_t remoteRank, CommLink &selectedLink, uint32_t &localDie)
{
    for (uint32_t netLayer : netLayers) {
        CommLink *linkList = nullptr;
        uint32_t linkNum = 0;
        HcclResult getLinkRet = HcclRankGraphGetLinks(comm, netLayer, localRank, remoteRank, &linkList, &linkNum);
        if (getLinkRet != HCCL_SUCCESS) {
            HCCL_DEBUG("No usable link on layer[%u] between rank[%u] and rank[%u], ret[%d]", netLayer, localRank,
                remoteRank, static_cast<int32_t>(getLinkRet));
            continue;
        }
        for (uint32_t linkIndex = 0; linkIndex < linkNum; ++linkIndex) {
            const CommLink link = linkList[linkIndex];
            if (link.linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                continue;
            }
            CHK_RET(QueryLinkDie(comm, localRank, link, localDie));
            selectedLink = link;
            return HCCL_SUCCESS;
        }
    }
    HCCL_ERROR("UBC_CTP link not found between rank[%u] and rank[%u]", localRank, remoteRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult AppendChannel(HcclComm comm, const OpParam &param, const std::vector<uint32_t> &netLayers,
    uint32_t remoteRank, ChannelGroup &group)
{
    CommLink selectedLink;
    uint32_t localDie = INVALID_VALUE_RANKID;
    CHK_RET(FindPreferredLink(comm, netLayers, param.myRank, remoteRank, selectedLink, localDie));
    if (group.dieId == INVALID_VALUE_RANKID) {
        group.dieId = localDie;
    } else if (group.dieId != localDie) {
        HCCL_ERROR("Hierarchical channel group spans dies[%u/%u]", group.dieId, localDie);
        return HCCL_E_NOT_SUPPORT;
    }

    HcclChannelDesc channelDesc;
    CHK_RET(HcclChannelDescInit(&channelDesc, CHANNEL_NOTIFY_NUM));
    FillChannelDesc(remoteRank, selectedLink, channelDesc);
    group.peerRanks.push_back(remoteRank);
    group.channelDescs.push_back(channelDesc);
    return HCCL_SUCCESS;
}

HcclResult AcquireChannelGroup(HcclComm comm, ChannelGroup &group, const char *groupName)
{
    if (group.channelDescs.empty() || group.dieId == INVALID_VALUE_RANKID) {
        HCCL_ERROR("Invalid hierarchical channel group[%s]", groupName);
        return HCCL_E_INTERNAL;
    }
    group.channelHandles.resize(group.channelDescs.size());
    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, group.channelDescs.data(),
        static_cast<uint32_t>(group.channelDescs.size()), group.channelHandles.data()));
    HCCL_INFO(
        "Acquired hierarchical group[%s] on die[%u], count[%zu]", groupName, group.dieId, group.channelHandles.size());
    return HCCL_SUCCESS;
}

HcclResult AcquireHierarchicalChannels(HcclComm comm, const OpParam &param, const std::vector<uint32_t> &netLayers,
    ChannelGroup &crossGroup, ChannelGroup &localGroup, ServerInfo &serverInfo, uint32_t &remoteReadChannelIndex)
{
    CHK_RET(GetServerInfo(param, serverInfo));
    const uint32_t serverEnd = serverInfo.start + serverInfo.size;
    for (uint32_t peerRank = serverInfo.start; peerRank < serverEnd; ++peerRank) {
        if (peerRank != param.myRank) {
            CHK_RET(AppendChannel(comm, param, netLayers, peerRank, localGroup));
        }
    }

    std::vector<uint32_t> crossPeers;
    uint32_t remoteReadPeer = INVALID_VALUE_RANKID;
    if (serverInfo.start == FIRST_SERVER_START) {
        remoteReadPeer = serverInfo.remoteStart + param.myRank % serverInfo.remoteSize;
        crossPeers.push_back(remoteReadPeer);
    } else {
        const uint32_t localIndex = param.myRank - serverInfo.start;
        remoteReadPeer = localIndex;
        for (uint32_t firstServerIndex = localIndex; firstServerIndex < FIRST_SERVER_SIZE;
            firstServerIndex += serverInfo.size) {
            crossPeers.push_back(firstServerIndex);
        }
    }
    std::sort(crossPeers.begin(), crossPeers.end());
    auto remotePeerIter = std::find(crossPeers.begin(), crossPeers.end(), remoteReadPeer);
    if (remotePeerIter == crossPeers.end()) {
        HCCL_ERROR("Remote hierarchical source rank[%u] is not connected", remoteReadPeer);
        return HCCL_E_INTERNAL;
    }
    remoteReadChannelIndex = static_cast<uint32_t>(std::distance(crossPeers.begin(), remotePeerIter));
    for (uint32_t peerRank : crossPeers) {
        CHK_RET(AppendChannel(comm, param, netLayers, peerRank, crossGroup));
    }

    if (localGroup.dieId == crossGroup.dieId) {
        HCCL_ERROR("Hierarchical local/cross groups unexpectedly share die[%u]", localGroup.dieId);
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_RET(AcquireChannelGroup(comm, localGroup, "local"));
    CHK_RET(AcquireChannelGroup(comm, crossGroup, "cross"));
    return HCCL_SUCCESS;
}

HcclResult AcquireChannels(HcclComm comm, const OpParam &param, const std::vector<uint32_t> &netLayers,
    std::vector<ChannelGroup> &channelGroups)
{
    // Every rank must issue the same number of kernel launches and channel synchronizations.
    // Always using host-side chunking keeps that schedule deterministic on asymmetric topologies
    // without querying endpoint attributes that belong to non-local ranks.
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }
        CommLink selectedLink;
        uint32_t localDie = INVALID_VALUE_RANKID;
        CHK_RET(FindPreferredLink(comm, netLayers, param.myRank, remoteRank, selectedLink, localDie));

        auto groupIter
            = std::find_if(channelGroups.begin(), channelGroups.end(), [localDie](const ChannelGroup &group) {
                  return group.dieId == localDie;
              });
        if (groupIter == channelGroups.end()) {
            channelGroups.emplace_back();
            channelGroups.back().dieId = localDie;
            groupIter = channelGroups.end() - 1;
        }
        HcclChannelDesc channelDesc;
        CHK_RET(HcclChannelDescInit(&channelDesc, 1));
        FillChannelDesc(remoteRank, selectedLink, channelDesc);
        groupIter->peerRanks.push_back(remoteRank);
        groupIter->channelDescs.push_back(channelDesc);
    }

    std::sort(channelGroups.begin(), channelGroups.end(), [](const ChannelGroup &left, const ChannelGroup &right) {
        return left.dieId < right.dieId;
    });
    if (channelGroups.empty() || channelGroups.size() > MAX_KERNEL_DIE_COUNT) {
        HCCL_ERROR("Invalid CCU die group count[%zu]", channelGroups.size());
        return HCCL_E_NOT_SUPPORT;
    }

    for (ChannelGroup &group : channelGroups) {
        group.channelHandles.resize(group.channelDescs.size());
        CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, group.channelDescs.data(),
            static_cast<uint32_t>(group.channelDescs.size()), group.channelHandles.data()));
        HCCL_INFO("Acquired CCU channels on die[%u], count[%zu]", group.dieId, group.channelHandles.size());
    }
    // 资源申请保持固定Die顺序。
    // 注册时让通道较少的Die承担本rank输入，平衡两个局部归约树。
    std::sort(channelGroups.begin(), channelGroups.end(), [](const ChannelGroup &left, const ChannelGroup &right) {
        if (left.channelHandles.size() != right.channelHandles.size()) {
            return left.channelHandles.size() < right.channelHandles.size();
        }
        return left.dieId < right.dieId;
    });
    return HCCL_SUCCESS;
}

HcclResult RegisterHierarchicalKernels(HcclComm comm, const OpParam &param, const ChannelGroup &crossGroup,
    const ChannelGroup &localGroup, const ServerInfo &serverInfo, uint32_t remoteReadChannelIndex,
    AlgResourceCtx &resourceCtx)
{
    CcuInsHandle insHandle = 0;
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    if (insNum != 1) {
        HCCL_ERROR("Expected exactly one CCU instance, got[%u]", insNum);
        return HCCL_E_INTERNAL;
    }

    CcuResult ccuRet = HcommCcuKernelRegisterStart(insHandle);
    if (ccuRet != CCU_SUCCESS) {
        HCCL_ERROR("Hierarchical kernel register start failed, ret[%d]", static_cast<int32_t>(ccuRet));
        return ConvertCcuToHccl(ccuRet);
    }

    CcuKernelInfo crossKernelInfo;
    int32_t nameRet = std::snprintf(crossKernelInfo.kernelFuncName, sizeof(crossKernelInfo.kernelFuncName), "%s_die%u",
        HIER_CROSS_KERNEL_NAME, crossGroup.dieId);
    if (nameRet < 0 || static_cast<size_t>(nameRet) >= sizeof(crossKernelInfo.kernelFuncName)) {
        (void)HcommCcuKernelRegisterEnd(insHandle);
        return HCCL_E_INTERNAL;
    }
    crossKernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuReduceScatterHierCrossKernel);
    auto crossKernelArg = std::make_shared<CcuKernelArgReduceScatterHierCross>();
    crossKernelArg->rankSize = param.rankSize;
    crossKernelArg->rankId = param.myRank;
    crossKernelArg->localServerSize = serverInfo.size;
    crossKernelArg->remoteServerSize = serverInfo.remoteSize;
    crossKernelArg->remoteReadChannelIndex = remoteReadChannelIndex;
    crossKernelArg->dataType = param.dataType;
    crossKernelArg->reduceOp = param.reduceType;
    crossKernelArg->channelCount = static_cast<uint32_t>(crossGroup.channelHandles.size());
    for (uint32_t channelIndex = 0; channelIndex < crossKernelArg->channelCount; ++channelIndex) {
        crossKernelArg->channels[channelIndex] = crossGroup.channelHandles[channelIndex];
        crossKernelArg->peerRanks[channelIndex] = crossGroup.peerRanks[channelIndex];
    }
    crossKernelInfo.setKernelArg(crossKernelArg);

    CcuKernelHandle crossKernelHandle = 0;
    const void *crossKernelArgs[] = {crossKernelInfo.kernelArg};
    constexpr uint32_t kernelArgNum = 1;
    ccuRet = HcommCcuKernelRegister(insHandle, KERNEL_DIE_ID, crossKernelInfo.kernelFuncName,
        reinterpret_cast<const void *>(crossKernelInfo.kernelFunc), crossKernelArgs, kernelArgNum, &crossKernelHandle);
    if (ccuRet != CCU_SUCCESS) {
        HCCL_ERROR("Hierarchical cross kernel register failed, ret[%d]", static_cast<int32_t>(ccuRet));
        (void)HcommCcuKernelRegisterEnd(insHandle);
        return ConvertCcuToHccl(ccuRet);
    }
    resourceCtx.ccuKernels.push_back(crossKernelHandle);

    CcuKernelInfo localKernelInfo;
    nameRet = std::snprintf(localKernelInfo.kernelFuncName, sizeof(localKernelInfo.kernelFuncName), "%s_die%u",
        HIER_LOCAL_KERNEL_NAME, localGroup.dieId);
    if (nameRet < 0 || static_cast<size_t>(nameRet) >= sizeof(localKernelInfo.kernelFuncName)) {
        (void)HcommCcuKernelRegisterEnd(insHandle);
        return HCCL_E_INTERNAL;
    }
    localKernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuReduceScatterHierLocalKernel);
    auto localKernelArg = std::make_shared<CcuKernelArgReduceScatterHierLocal>();
    localKernelArg->rankSize = param.rankSize;
    localKernelArg->rankId = param.myRank;
    localKernelArg->serverStart = serverInfo.start;
    localKernelArg->serverSize = serverInfo.size;
    const uint32_t localIndex = param.myRank - serverInfo.start;
    for (uint32_t outputRank = localIndex; outputRank < param.rankSize; outputRank += serverInfo.size) {
        ++localKernelArg->assignedOutputCount;
    }
    localKernelArg->dataType = param.dataType;
    localKernelArg->reduceOp = param.reduceType;
    localKernelArg->channelCount = static_cast<uint32_t>(localGroup.channelHandles.size());
    for (uint32_t channelIndex = 0; channelIndex < localKernelArg->channelCount; ++channelIndex) {
        localKernelArg->channels[channelIndex] = localGroup.channelHandles[channelIndex];
        localKernelArg->peerRanks[channelIndex] = localGroup.peerRanks[channelIndex];
    }
    localKernelInfo.setKernelArg(localKernelArg);

    CcuKernelHandle localKernelHandle = 0;
    const void *localKernelArgs[] = {localKernelInfo.kernelArg};
    ccuRet = HcommCcuKernelRegister(insHandle, KERNEL_DIE_ID, localKernelInfo.kernelFuncName,
        reinterpret_cast<const void *>(localKernelInfo.kernelFunc), localKernelArgs, kernelArgNum, &localKernelHandle);
    if (ccuRet != CCU_SUCCESS) {
        HCCL_ERROR("Hierarchical local kernel register failed, ret[%d]", static_cast<int32_t>(ccuRet));
        (void)HcommCcuKernelRegisterEnd(insHandle);
        return ConvertCcuToHccl(ccuRet);
    }
    resourceCtx.ccuKernels.push_back(localKernelHandle);

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    if (ccuRet != CCU_SUCCESS) {
        HCCL_ERROR("Hierarchical kernel register end failed, ret[%d]", static_cast<int32_t>(ccuRet));
        return ConvertCcuToHccl(ccuRet);
    }
    resourceCtx.partialKernelCount = 2;
    resourceCtx.secondaryScratchStartSlot = 1;
    return HCCL_SUCCESS;
}

HcclResult RegisterKernels(
    HcclComm comm, const OpParam &param, const std::vector<ChannelGroup> &channelGroups, AlgResourceCtx &resourceCtx)
{
    if (param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    CcuInsHandle insHandle = 0;
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    if (insNum != 1) {
        HCCL_ERROR("Expected exactly one CCU instance, got[%u]", insNum);
        return HCCL_E_INTERNAL;
    }

    CcuResult ccuRet = HcommCcuKernelRegisterStart(insHandle);
    if (ccuRet != CCU_SUCCESS) {
        HCCL_ERROR("CCU kernel register start failed, ret[%d]", static_cast<int32_t>(ccuRet));
        return ConvertCcuToHccl(ccuRet);
    }

    uint32_t scratchStartSlot = 0;
    const uint32_t secondaryScratchStartSlot
        = channelGroups.size() > 1 ? static_cast<uint32_t>(channelGroups[0].channelHandles.size()) + 1 : 0;
    for (uint32_t groupIndex = 0; groupIndex < channelGroups.size(); ++groupIndex) {
        const ChannelGroup &group = channelGroups[groupIndex];
        CcuKernelInfo kernelInfo;
        int32_t nameRet = std::snprintf(
            kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "%s_die%u", KERNEL_NAME, group.dieId);
        if (nameRet < 0 || static_cast<size_t>(nameRet) >= sizeof(kernelInfo.kernelFuncName)) {
            HCCL_ERROR("Failed to fill CCU kernel name");
            (void)HcommCcuKernelRegisterEnd(insHandle);
            return HCCL_E_INTERNAL;
        }
        kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuReduceScatterDeterministicKernel);

        auto kernelArg = std::make_shared<CcuKernelArgReduceScatter>();
        kernelArg->rankSize = param.rankSize;
        kernelArg->rankId = param.myRank;
        kernelArg->dataType = param.dataType;
        kernelArg->reduceOp = param.reduceType;
        kernelArg->includeSelf = groupIndex == 0 ? 1 : 0;
        kernelArg->writeOutput = groupIndex == 0 ? 1 : 0;
        kernelArg->scratchStartSlot = scratchStartSlot;
        kernelArg->channelCount = static_cast<uint32_t>(group.channelHandles.size());
        for (uint32_t channelIndex = 0; channelIndex < group.channelHandles.size(); ++channelIndex) {
            kernelArg->channels[channelIndex] = group.channelHandles[channelIndex];
            kernelArg->peerRanks[channelIndex] = group.peerRanks[channelIndex];
        }
        kernelInfo.setKernelArg(kernelArg);

        CcuKernelHandle kernelHandle = 0;
        const void *kernelArgs[] = {kernelInfo.kernelArg};
        constexpr uint32_t kernelArgNum = 1;
        ccuRet = HcommCcuKernelRegister(insHandle, KERNEL_DIE_ID, kernelInfo.kernelFuncName,
            reinterpret_cast<const void *>(kernelInfo.kernelFunc), kernelArgs, kernelArgNum, &kernelHandle);
        if (ccuRet != CCU_SUCCESS) {
            HCCL_ERROR("CCU kernel register failed, ret[%d]", static_cast<int32_t>(ccuRet));
            (void)HcommCcuKernelRegisterEnd(insHandle);
            return ConvertCcuToHccl(ccuRet);
        }
        resourceCtx.ccuKernels.push_back(kernelHandle);
        scratchStartSlot += kernelArg->channelCount + kernelArg->includeSelf;
    }

    if (channelGroups.size() > 1) {
        CcuKernelInfo mergeKernelInfo;
        int32_t nameRet = std::snprintf(
            mergeKernelInfo.kernelFuncName, sizeof(mergeKernelInfo.kernelFuncName), "%s_merge", KERNEL_NAME);
        if (nameRet < 0 || static_cast<size_t>(nameRet) >= sizeof(mergeKernelInfo.kernelFuncName)) {
            HCCL_ERROR("Failed to fill CCU merge kernel name");
            (void)HcommCcuKernelRegisterEnd(insHandle);
            return HCCL_E_INTERNAL;
        }
        mergeKernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuReduceScatterMergeKernel);

        auto mergeKernelArg = std::make_shared<CcuKernelArgReduceScatterMerge>();
        mergeKernelArg->dataType = param.dataType;
        mergeKernelArg->reduceOp = param.reduceType;
        mergeKernelArg->scratchStartSlot = secondaryScratchStartSlot;
        mergeKernelInfo.setKernelArg(mergeKernelArg);

        CcuKernelHandle mergeKernelHandle = 0;
        const void *mergeKernelArgs[] = {mergeKernelInfo.kernelArg};
        constexpr uint32_t kernelArgNum = 1;
        ccuRet = HcommCcuKernelRegister(insHandle, KERNEL_DIE_ID, mergeKernelInfo.kernelFuncName,
            reinterpret_cast<const void *>(mergeKernelInfo.kernelFunc), mergeKernelArgs, kernelArgNum,
            &mergeKernelHandle);
        if (ccuRet != CCU_SUCCESS) {
            HCCL_ERROR("CCU merge kernel register failed, ret[%d]", static_cast<int32_t>(ccuRet));
            (void)HcommCcuKernelRegisterEnd(insHandle);
            return ConvertCcuToHccl(ccuRet);
        }
        resourceCtx.ccuKernels.push_back(mergeKernelHandle);
    }

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    if (ccuRet != CCU_SUCCESS) {
        HCCL_ERROR("CCU kernel register end failed, ret[%d]", static_cast<int32_t>(ccuRet));
        return ConvertCcuToHccl(ccuRet);
    }

    resourceCtx.partialKernelCount = static_cast<uint32_t>(channelGroups.size());
    resourceCtx.secondaryScratchStartSlot = secondaryScratchStartSlot;
    return HCCL_SUCCESS;
}

HcclResult BuildResources(HcclComm comm, const OpParam &param, AlgResourceCtx &resourceCtx)
{
    void *cclBufferAddr = nullptr;
    uint64_t cclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    if (cclBufferAddr == nullptr || cclBufferSize == 0) {
        HCCL_ERROR("Invalid local HCCL buffer");
        return HCCL_E_INTERNAL;
    }
    resourceCtx.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};
    if (param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    std::vector<uint32_t> netLayers;
    CHK_RET(QueryNetLayers(comm, netLayers));

    const bool useHierarchicalPath = IsSmallHierarchicalParam(param);
    std::vector<ChannelGroup> channelGroups;
    ChannelGroup crossGroup;
    ChannelGroup localGroup;
    ServerInfo serverInfo;
    uint32_t remoteReadChannelIndex = 0;
    if (useHierarchicalPath) {
        CHK_RET(AcquireHierarchicalChannels(
            comm, param, netLayers, crossGroup, localGroup, serverInfo, remoteReadChannelIndex));
    } else {
        CHK_RET(AcquireChannels(comm, param, netLayers, channelGroups));
    }
    if (!useHierarchicalPath && channelGroups.size() > 1) {
        // HcommCcuKernelLaunch requires a thread created from a runtime stream.
        // A bare HcclThreadAcquire has no runtime context in the simulator and
        // makes every two-die topology fail with "GetRankIdByCtxId CurrContext: 0".
        // The runtime owns this stream for the lifetime of the cached engine
        // context; the associated thread handle is serialized in resourceCtx.
        aclrtStream secondaryStream = nullptr;
        ACLCHECK(aclrtCreateStream(&secondaryStream));
        HcclResult threadRet = HcclThreadAcquireWithStream(
            comm, CommEngine::COMM_ENGINE_CCU, secondaryStream, 1, &resourceCtx.secondaryThread);
        if (threadRet != HCCL_SUCCESS) {
            aclError destroyRet = aclrtDestroyStream(secondaryStream);
            if (destroyRet != ACL_SUCCESS) {
                HCCL_WARNING("Failed to destroy unused secondary stream, acl ret[%d]", destroyRet);
            }
            HCCL_ERROR("Failed to acquire secondary CCU thread with stream, ret[%d]", static_cast<int32_t>(threadRet));
            return threadRet;
        }
    }
    if (useHierarchicalPath) {
        CHK_RET(RegisterHierarchicalKernels(
            comm, param, crossGroup, localGroup, serverInfo, remoteReadChannelIndex, resourceCtx));
    } else {
        CHK_RET(RegisterKernels(comm, param, channelGroups, resourceCtx));
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    CHK_RET(CheckParameters(recvCount, dataType, op));

    // 构造算子参数
    OpParam param{};
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;

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
    if (param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || param.myRank >= param.rankSize) {
        HCCL_ERROR("Invalid rank information, rank[%u/%u]", param.myRank, param.rankSize);
        return HCCL_E_PARA;
    }
    const uint64_t recvSize = recvCount * sizeof(float);
    if (recvSize != 0 && param.rankSize > std::numeric_limits<uint64_t>::max() / recvSize) {
        HCCL_ERROR("Input byte-size overflow, recvSize[%" PRIu64 "], rankSize[%u]", recvSize, param.rankSize);
        return HCCL_E_PARA;
    }
    const char *algorithmTag = IsSmallHierarchicalParam(param) ? HIERARCHICAL_ALGORITHM_TAG : ALGORITHM_TAG;
    int32_t tagRet = std::snprintf(param.tag, sizeof(param.tag), "%s", algorithmTag);
    if (tagRet < 0 || static_cast<size_t>(tagRet) >= sizeof(param.tag)) {
        HCCL_ERROR("Failed to fill operator tag");
        return HCCL_E_INTERNAL;
    }

    // ==============================================
    // STEP 2: 创建资源
    // ==============================================
    const CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;

    // ==============================================
    // STEP 2.1: 申请用于 Host/Device 同步的通信资源
    // ==============================================
    // 将用户传入的 stream 转换为 CCU 通信引擎中的 thread，并申请 1 个 notify
    CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, stream, 1, &param.cpuThread));

    void *ctx = nullptr;
    uint64_t size = 0;
    HcclResult ctxGetRet = HcclEngineCtxGet(comm, param.tag, ccuEngine, &ctx, &size);
    if (ctxGetRet == HCCL_SUCCESS) {
        // CCU 资源已经存在，复用资源
        HCCL_INFO("Engine context already exists");
        param.resCtx = ctx;
        param.ctxSize = size;
    } else {
        if (ctxGetRet != HCCL_E_NOT_FOUND) {
            HCCL_ERROR("Failed to query engine context, ret[%d]", static_cast<int32_t>(ctxGetRet));
            return ctxGetRet;
        }
        // Device 资源不存在，资源构建
        AlgResourceCtx resCtxHost{};

        // ==============================================
        // STEP 2.2: 申请资源：Thread、Channel、CCU Kernel
        // ==============================================
        CHK_RET(BuildResources(comm, param, resCtxHost));

        // ==============================================
        // STEP 2.3: 申请通信引擎上下文
        // ==============================================
        // 申请 CCU 通信引擎上下文，存放 AlgResourceCtx 信息
        std::vector<char> seq = resCtxHost.Serialize();
        uint64_t seqSize = seq.size();
        param.ctxSize = seqSize;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, ccuEngine, param.ctxSize, &param.resCtx));
        HcclResult ctxCopyRet = HcclEngineCtxCopy(comm, ccuEngine, param.tag, seq.data(), seqSize, 0);
        if (ctxCopyRet != HCCL_SUCCESS) {
            HcclResult ctxDestroyRet = HcclEngineCtxDestroy(comm, param.tag, ccuEngine);
            if (ctxDestroyRet != HCCL_SUCCESS) {
                HCCL_ERROR("Failed to destroy incomplete engine context, ret[%d]", static_cast<int32_t>(ctxDestroyRet));
            }
            return ctxCopyRet;
        }
    }

    // ==============================================
    // STEP 3: 下发 CCU Kernel
    // ==============================================
    CHK_RET(ops_hccl::ExecOp(param));
    return HCCL_SUCCESS;
}
