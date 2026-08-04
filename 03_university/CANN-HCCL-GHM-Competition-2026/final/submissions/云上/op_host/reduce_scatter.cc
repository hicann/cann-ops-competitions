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
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include <ccu/ccu_launch.h>
#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "exec_op.h"
#include "ccu_kernel.h"

namespace {

constexpr uint32_t CHANNEL_NOTIFY_NUM = 1;
constexpr uint32_t MAX_DIE_COUNT = 2;
constexpr uint32_t OMNIPIPE_MAX_MESH_RANKS = 8;
constexpr uint32_t OMNIPIPE_MAX_NHR_RANKS = 4;
constexpr uint64_t PARALLEL_2D_OMNIPIPE_MIN_BYTES = 1UL * 1024 * 1024;
constexpr uint64_t OMNIPIPE_SLICE_ALIGNMENT = 128;
constexpr char KERNEL_NAME[] = "CcuReduceScatterReadReduceKernel";
constexpr char OMNIPIPE_MESH_KERNEL_NAME[] = "CcuReduceScatterOmnipipeMeshKernel";
constexpr char OMNIPIPE_NHR_KERNEL_NAME[] = "CcuReduceScatterOmnipipeNhrKernel";

struct ChannelGroup {
    uint32_t dieId = 0;
    std::vector<ChannelHandle> channels;
    uint32_t kernelKind = ops_hccl::CCU_REDUCE_SCATTER_KERNEL_GENERIC;
    uint32_t axisRankSize = 0;
    uint32_t axisRankIndex = 0;
    std::vector<uint32_t> peerSubRanks;
};

HcclResult ValidateChannelGroup(const ChannelGroup &group)
{
    if (group.dieId >= MAX_DIE_COUNT || group.channels.empty() || group.channels.size() >= MAX_RANK_SIZE) {
        HCCL_ERROR("[ValidateChannelGroup] invalid die[%u] or channel count[%zu]", group.dieId, group.channels.size());
        return HCCL_E_PARA;
    }

    const bool isMeshKernel = group.kernelKind == ops_hccl::CCU_REDUCE_SCATTER_KERNEL_OMNIPIPE_MESH;
    const bool isNhrKernel = group.kernelKind == ops_hccl::CCU_REDUCE_SCATTER_KERNEL_NHR;
    if (group.kernelKind != ops_hccl::CCU_REDUCE_SCATTER_KERNEL_GENERIC && !isMeshKernel && !isNhrKernel) {
        HCCL_ERROR("[ValidateChannelGroup] unsupported kernel kind[%u]", group.kernelKind);
        return HCCL_E_PARA;
    }
    if (!isMeshKernel && !isNhrKernel) {
        return HCCL_SUCCESS;
    }

    std::vector<uint32_t> sortedPeerSubRanks = group.peerSubRanks;
    std::sort(sortedPeerSubRanks.begin(), sortedPeerSubRanks.end());
    if (group.axisRankSize <= 1 || group.axisRankSize > MAX_RANK_SIZE
        || group.axisRankIndex >= group.axisRankSize || group.channels.size() + 1 != group.axisRankSize
        || group.peerSubRanks.size() != group.channels.size() || sortedPeerSubRanks.empty()
        || sortedPeerSubRanks.back() >= group.axisRankSize
        || std::adjacent_find(sortedPeerSubRanks.begin(), sortedPeerSubRanks.end()) != sortedPeerSubRanks.end()
        || std::binary_search(sortedPeerSubRanks.begin(), sortedPeerSubRanks.end(), group.axisRankIndex)) {
        HCCL_ERROR("[ValidateChannelGroup] invalid axis metadata: kind[%u], rankSize[%u], rankIndex[%u], peers[%zu]",
            group.kernelKind, group.axisRankSize, group.axisRankIndex, group.peerSubRanks.size());
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

bool IsSupportedReduceOp(HcclReduceOp op)
{
    return op == HcclReduceOp::HCCL_REDUCE_SUM || op == HcclReduceOp::HCCL_REDUCE_MAX
           || op == HcclReduceOp::HCCL_REDUCE_MIN;
}

HcclResult ValidateOpParam(const OpParam &param)
{
    if (param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || param.myRank >= param.rankSize) {
        HCCL_ERROR("[ValidateOpParam] invalid rank: myRank[%u], rankSize[%u]", param.myRank, param.rankSize);
        return HCCL_E_PARA;
    }
    auto dataTypeIt = SIZE_TABLE.find(param.dataType);
    if (dataTypeIt == SIZE_TABLE.end()) {
        HCCL_ERROR("[ValidateOpParam] unsupported dataType[%d]", param.dataType);
        return HCCL_E_NOT_SUPPORT;
    }
    if (!IsSupportedReduceOp(param.reduceType)) {
        HCCL_ERROR("[ValidateOpParam] unsupported reduce op[%d]", param.reduceType);
        return HCCL_E_NOT_SUPPORT;
    }

    const uint64_t dataTypeSize = dataTypeIt->second;
    if (param.count > std::numeric_limits<uint64_t>::max() / dataTypeSize / param.rankSize) {
        HCCL_ERROR("[ValidateOpParam] input size overflow");
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult GetNetLayers(HcclComm comm, std::vector<uint32_t> &netLayers)
{
    uint32_t *rawNetLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &rawNetLayers, &netLayerNum));
    CHK_PRT_RET(rawNetLayers == nullptr || netLayerNum == 0,
        HCCL_ERROR("[GetNetLayers] communication topology has no network layer"), HCCL_E_NOT_FOUND);

    netLayers.assign(rawNetLayers, rawNetLayers + netLayerNum);
    std::sort(netLayers.begin(), netLayers.end());
    netLayers.erase(std::unique(netLayers.begin(), netLayers.end()), netLayers.end());
    return HCCL_SUCCESS;
}

HcclResult GetEndpointDieId(HcclComm comm, uint32_t rank, const EndpointDesc &endpoint, uint32_t &dieId)
{
    CHK_RET(HcclRankGraphGetEndpointInfo(
        comm, rank, &endpoint, ENDPOINT_ATTR_DIE_ID, sizeof(dieId), static_cast<void *>(&dieId)));
    CHK_PRT_RET(dieId >= MAX_DIE_COUNT, HCCL_ERROR("[GetEndpointDieId] invalid dieId[%u] for rank[%u]", dieId, rank),
        HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult SelectChannelDesc(HcclComm comm, const OpParam &param, const std::vector<uint32_t> &netLayers,
    uint32_t remoteRank, HcclChannelDesc &channelDesc, uint32_t &dieId)
{
    HcclResult firstQueryError = HCCL_SUCCESS;
    for (uint32_t netLayer : netLayers) {
        uint32_t linkNum = 0;
        CommLink *links = nullptr;
        HcclResult queryResult = HcclRankGraphGetLinks(comm, netLayer, param.myRank, remoteRank, &links, &linkNum);
        if (queryResult != HCCL_SUCCESS) {
            if (firstQueryError == HCCL_SUCCESS) {
                firstQueryError = queryResult;
            }
            continue;
        }
        if (linkNum == 0) {
            continue;
        }
        CHK_PRT_RET(links == nullptr,
            HCCL_ERROR("[SelectChannelDesc] null link list on layer[%u] for rank[%u] to rank[%u]", netLayer,
                param.myRank, remoteRank),
            HCCL_E_INTERNAL);

        const CommLink *selectedLink = nullptr;
        for (uint32_t index = 0; index < linkNum; ++index) {
            const CommLink &candidate = links[index];
            if (candidate.linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                continue;
            }
            selectedLink = &candidate;
            break;
        }
        if (selectedLink == nullptr) {
            continue;
        }

        uint32_t selectedLocalDieId = 0;
        CHK_RET(GetEndpointDieId(comm, param.myRank, selectedLink->srcEndpointDesc, selectedLocalDieId));
        CHK_RET(HcclChannelDescInit(&channelDesc, 1));
        channelDesc.remoteRank = remoteRank;
        channelDesc.notifyNum = CHANNEL_NOTIFY_NUM;
        channelDesc.channelProtocol = selectedLink->linkAttr.linkProtocol;
        channelDesc.localEndpoint.protocol = selectedLink->srcEndpointDesc.protocol;
        channelDesc.localEndpoint.commAddr = selectedLink->srcEndpointDesc.commAddr;
        channelDesc.localEndpoint.loc = selectedLink->srcEndpointDesc.loc;
        channelDesc.remoteEndpoint.protocol = selectedLink->dstEndpointDesc.protocol;
        channelDesc.remoteEndpoint.commAddr = selectedLink->dstEndpointDesc.commAddr;
        channelDesc.remoteEndpoint.loc = selectedLink->dstEndpointDesc.loc;
        dieId = selectedLocalDieId;
        return HCCL_SUCCESS;
    }

    HCCL_ERROR("[SelectChannelDesc] UBC_CTP link not found from rank[%u] to rank[%u]", param.myRank, remoteRank);
    return firstQueryError == HCCL_SUCCESS ? HCCL_E_NOT_FOUND : firstQueryError;
}

HcclResult AcquireChannelGroups(HcclComm comm, const OpParam &param, std::vector<ChannelGroup> &groups)
{
    if (param.rankSize <= 1) {
        return HCCL_SUCCESS;
    }

    std::vector<uint32_t> netLayers;
    CHK_RET(GetNetLayers(comm, netLayers));

    std::map<uint32_t, std::vector<ChannelHandle>> channelsByDie;
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }

        HcclChannelDesc channelDesc;
        uint32_t dieId = 0;
        CHK_RET(SelectChannelDesc(comm, param, netLayers, remoteRank, channelDesc, dieId));

        ChannelHandle channel = 0;
        CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, &channelDesc, 1, &channel));
        channelsByDie[dieId].push_back(channel);
    }

    CHK_PRT_RET(channelsByDie.size() > MAX_DIE_COUNT,
        HCCL_ERROR("[AcquireChannelGroups] too many die groups[%zu]", channelsByDie.size()), HCCL_E_INTERNAL);

    groups.reserve(channelsByDie.size());
    for (auto &entry : channelsByDie) {
        ChannelGroup group;
        group.dieId = entry.first;
        group.channels = std::move(entry.second);
        groups.push_back(std::move(group));
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterCcuKernels(HcclComm comm, const OpParam &param, const std::vector<ChannelGroup> &groups,
    AlgResourceCtx &resourceCtx, bool initializeEveryGroup)
{
    if (groups.empty()) {
        return HCCL_SUCCESS;
    }
    for (const ChannelGroup &group : groups) {
        CHK_RET(ValidateChannelGroup(group));
    }

    CcuInsHandle insHandle = 0;
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1, HCCL_ERROR("[RegisterCcuKernels] expected one CCU instruction handle, got[%u]", insNum),
        HCCL_E_INTERNAL);

    CcuResult ccuResult = HcommCcuKernelRegisterStart(insHandle);
    if (ccuResult != CCU_SUCCESS) {
        HCCL_ERROR("[RegisterCcuKernels] register start failed[%d]", ccuResult);
        return ConvertCcuToHccl(ccuResult);
    }

    std::vector<std::shared_ptr<ops_hccl::CcuKernelArgReduceScatter>> kernelArgs;
    kernelArgs.reserve(groups.size());
    resourceCtx.ccuKernels.reserve(groups.size());
    resourceCtx.channelCounts.reserve(groups.size());
    for (uint32_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
        const ChannelGroup &group = groups[groupIndex];

        auto kernelArg = std::make_shared<ops_hccl::CcuKernelArgReduceScatter>();
        kernelArg->channelCount = static_cast<uint32_t>(group.channels.size());
        kernelArg->dataType = param.dataType;
        kernelArg->reduceOp = param.reduceType;
        kernelArg->destinationInitialized = (groupIndex == 0);
        kernelArg->kernelKind = group.kernelKind;
        kernelArg->axisRankSize = group.axisRankSize;
        kernelArg->axisRankIndex = group.axisRankIndex;
        if (initializeEveryGroup) {
            kernelArg->destinationInitialized = true;
        }
        for (uint32_t channelIndex = 0; channelIndex < group.channels.size(); ++channelIndex) {
            kernelArg->channels[channelIndex] = group.channels[channelIndex];
            if (channelIndex < group.peerSubRanks.size()) {
                kernelArg->peerSubRanks[channelIndex] = group.peerSubRanks[channelIndex];
            }
        }

        const void *registerArgs[] = {kernelArg.get()};
        CcuKernelHandle kernelHandle = 0;
        const char *kernelName = KERNEL_NAME;
        void *kernelFunction = reinterpret_cast<void *>(ops_hccl::CcuReduceScatterReadReduceKernel);
        if (group.kernelKind == ops_hccl::CCU_REDUCE_SCATTER_KERNEL_OMNIPIPE_MESH) {
            kernelName = OMNIPIPE_MESH_KERNEL_NAME;
            kernelFunction = reinterpret_cast<void *>(ops_hccl::CcuReduceScatterOmnipipeMeshKernel);
        } else if (group.kernelKind == ops_hccl::CCU_REDUCE_SCATTER_KERNEL_NHR) {
            kernelName = OMNIPIPE_NHR_KERNEL_NAME;
            kernelFunction = reinterpret_cast<void *>(ops_hccl::CcuReduceScatterOmnipipeNhrKernel);
        }
        ccuResult = HcommCcuKernelRegister(
            insHandle, group.dieId, kernelName, kernelFunction, registerArgs, 1, &kernelHandle);
        if (ccuResult != CCU_SUCCESS) {
            HCCL_ERROR("[RegisterCcuKernels] register failed on die[%u], ccuRet[%d]", group.dieId, ccuResult);
            return ConvertCcuToHccl(ccuResult);
        }
        kernelArgs.push_back(std::move(kernelArg));
        resourceCtx.ccuKernels.push_back(kernelHandle);
        resourceCtx.channelCounts.push_back(static_cast<uint32_t>(group.channels.size()));
    }

    ccuResult = HcommCcuKernelRegisterEnd(insHandle);
    if (ccuResult != CCU_SUCCESS) {
        HCCL_ERROR("[RegisterCcuKernels] register end failed[%d]", ccuResult);
        return ConvertCcuToHccl(ccuResult);
    }
    return HCCL_SUCCESS;
}

HcclResult AcquireThreadResources(
    HcclComm comm, const OpParam &param, uint32_t threadCount, AlgResourceCtx &resourceCtx)
{
    CHK_PRT_RET(threadCount == 0, HCCL_ERROR("[AcquireThreadResources] thread count is zero"), HCCL_E_PARA);
    resourceCtx.threads.resize(threadCount);
    resourceCtx.threads[0] = param.cpuThread;
    if (threadCount > 1) {
        CHK_RET(HcclThreadAcquire(comm, CommEngine::COMM_ENGINE_CCU, threadCount - 1, 1, &resourceCtx.threads[1]));
    }
    return HCCL_SUCCESS;
}

bool IsValidRankList(const std::vector<uint32_t> &ranks, uint32_t rankSize)
{
    std::vector<uint32_t> sortedRanks = ranks;
    std::sort(sortedRanks.begin(), sortedRanks.end());
    if (sortedRanks.empty() || sortedRanks.back() >= rankSize
        || std::adjacent_find(sortedRanks.begin(), sortedRanks.end()) != sortedRanks.end()) {
        return false;
    }
    return true;
}

bool HasSameRanks(const std::vector<uint32_t> &left, const std::vector<uint32_t> &right)
{
    if (left.size() != right.size()) {
        return false;
    }
    std::vector<uint32_t> sortedLeft = left;
    std::vector<uint32_t> sortedRight = right;
    std::sort(sortedLeft.begin(), sortedLeft.end());
    std::sort(sortedRight.begin(), sortedRight.end());
    return sortedLeft == sortedRight;
}

HcclResult SelectAxisChannelDescs(HcclComm comm, const OpParam &param, const std::vector<uint32_t> &netLayers,
    const std::vector<uint32_t> &axisRanks, std::vector<HcclChannelDesc> &channelDescs, uint32_t &dieId, bool &found)
{
    found = false;
    channelDescs.clear();
    if (axisRanks.size() <= 1) {
        return HCCL_SUCCESS;
    }
    channelDescs.reserve(axisRanks.size() - 1);

    for (uint32_t remoteRank : axisRanks) {
        if (remoteRank == param.myRank) {
            continue;
        }

        const CommLink *selectedLink = nullptr;
        for (uint32_t netLayer : netLayers) {
            uint32_t linkNum = 0;
            CommLink *links = nullptr;
            const HcclResult result =
                HcclRankGraphGetLinks(comm, netLayer, param.myRank, remoteRank, &links, &linkNum);
            if (result != HCCL_SUCCESS || links == nullptr || linkNum == 0) {
                continue;
            }

            for (uint32_t linkIndex = 0; linkIndex < linkNum; ++linkIndex) {
                const CommLink &candidate = links[linkIndex];
                if (candidate.linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                    continue;
                }
                selectedLink = &candidate;
                break;
            }
            if (selectedLink != nullptr) {
                break;
            }
        }
        if (selectedLink == nullptr) {
            return HCCL_SUCCESS;
        }

        HcclChannelDesc channelDesc;
        CHK_RET(HcclChannelDescInit(&channelDesc, 1));
        channelDesc.remoteRank = remoteRank;
        channelDesc.notifyNum = CHANNEL_NOTIFY_NUM;
        channelDesc.channelProtocol = selectedLink->linkAttr.linkProtocol;
        channelDesc.localEndpoint.protocol = selectedLink->srcEndpointDesc.protocol;
        channelDesc.localEndpoint.commAddr = selectedLink->srcEndpointDesc.commAddr;
        channelDesc.localEndpoint.loc = selectedLink->srcEndpointDesc.loc;
        channelDesc.remoteEndpoint.protocol = selectedLink->dstEndpointDesc.protocol;
        channelDesc.remoteEndpoint.commAddr = selectedLink->dstEndpointDesc.commAddr;
        channelDesc.remoteEndpoint.loc = selectedLink->dstEndpointDesc.loc;
        channelDescs.push_back(channelDesc);
    }

    dieId = 0;
    found = channelDescs.size() + 1 == axisRanks.size();
    return HCCL_SUCCESS;
}

HcclResult TryBuildParallel2dOmnipipeContext(
    HcclComm comm, const OpParam &param, AlgResourceCtx &resourceCtx, bool &built)
{
    built = false;
    if (param.rankSize != 12 && param.rankSize != 16) {
        return HCCL_SUCCESS;
    }

    auto dataTypeIt = SIZE_TABLE.find(param.dataType);
    if (dataTypeIt == SIZE_TABLE.end() || param.count > std::numeric_limits<uint64_t>::max() / dataTypeIt->second) {
        return HCCL_SUCCESS;
    }
    const uint64_t sliceBytes = param.count * dataTypeIt->second;
    if (sliceBytes < PARALLEL_2D_OMNIPIPE_MIN_BYTES || sliceBytes > MAX_DATA_SIZE
        || sliceBytes % OMNIPIPE_SLICE_ALIGNMENT != 0 || resourceCtx.localBuffer.addr == nullptr) {
        return HCCL_SUCCESS;
    }

    std::vector<uint32_t> netLayers;
    HcclResult result = GetNetLayers(comm, netLayers);
    if (result != HCCL_SUCCESS) {
        return HCCL_SUCCESS;
    }

    std::vector<uint32_t> topologyLocalRanks;
    std::vector<uint32_t> globalRanks;
    std::vector<uint32_t> instanceSizes;
    bool foundLocalTopology = false;
    bool foundGlobalTopology = false;
    for (uint32_t netLayer : netLayers) {
        CommTopo topoType = COMM_TOPO_RESERVED;
        result = HcclRankGraphGetTopoTypeByLayer(comm, netLayer, &topoType);
        if (result != HCCL_SUCCESS) {
            continue;
        }

        uint32_t *rawRanks = nullptr;
        uint32_t rankNum = 0;
        result = HcclRankGraphGetRanksByLayer(comm, netLayer, &rawRanks, &rankNum);
        if (result != HCCL_SUCCESS || rawRanks == nullptr || rankNum == 0) {
            continue;
        }
        std::vector<uint32_t> candidateRanks(rawRanks, rawRanks + rankNum);
        if (!IsValidRankList(candidateRanks, param.rankSize)) {
            continue;
        }

        if (topoType == COMM_TOPO_1DMESH && !foundLocalTopology) {
            if (candidateRanks.size() <= 1 || candidateRanks.size() >= param.rankSize
                || std::find(candidateRanks.begin(), candidateRanks.end(), param.myRank) == candidateRanks.end()) {
                continue;
            }

            uint32_t *rawInstanceSizes = nullptr;
            uint32_t instanceCount = 0;
            result = HcclRankGraphGetInstSizeListByLayer(comm, netLayer, &rawInstanceSizes, &instanceCount);
            if (result != HCCL_SUCCESS || rawInstanceSizes == nullptr || instanceCount <= 1) {
                continue;
            }
            std::vector<uint32_t> candidateInstanceSizes(rawInstanceSizes, rawInstanceSizes + instanceCount);
            uint64_t totalRanks = 0;
            bool uniform = true;
            for (uint32_t instanceSize : candidateInstanceSizes) {
                totalRanks += instanceSize;
                uniform = uniform && instanceSize == candidateRanks.size();
            }
            if (!uniform || totalRanks != param.rankSize
                || candidateRanks.size() * instanceCount != param.rankSize) {
                continue;
            }
            topologyLocalRanks = std::move(candidateRanks);
            instanceSizes = std::move(candidateInstanceSizes);
            foundLocalTopology = true;
        } else if (topoType == COMM_TOPO_CLOS && !foundGlobalTopology
            && candidateRanks.size() == param.rankSize) {
            globalRanks = std::move(candidateRanks);
            foundGlobalTopology = true;
        }

        if (foundLocalTopology && foundGlobalTopology) {
            break;
        }
    }
    if (!foundLocalTopology || !foundGlobalTopology) {
        return HCCL_SUCCESS;
    }

    const uint32_t localGroupSize = static_cast<uint32_t>(topologyLocalRanks.size());
    const uint32_t serverGroupCount = static_cast<uint32_t>(instanceSizes.size());
    const uint64_t scratchSlotCount = static_cast<uint64_t>(localGroupSize) - 1;
    if (localGroupSize > OMNIPIPE_MAX_MESH_RANKS || serverGroupCount > OMNIPIPE_MAX_NHR_RANKS
        || scratchSlotCount == 0 || sliceBytes > std::numeric_limits<uint64_t>::max() / scratchSlotCount
        || scratchSlotCount * sliceBytes > resourceCtx.localBuffer.size) {
        return HCCL_SUCCESS;
    }

    std::vector<std::vector<uint32_t>> serverGroups;
    serverGroups.reserve(serverGroupCount);
    uint64_t globalOffset = 0;
    for (uint32_t instanceSize : instanceSizes) {
        if (globalOffset + instanceSize > globalRanks.size()) {
            return HCCL_SUCCESS;
        }
        serverGroups.emplace_back(
            globalRanks.begin() + globalOffset, globalRanks.begin() + globalOffset + instanceSize);
        globalOffset += instanceSize;
    }
    if (globalOffset != globalRanks.size()) {
        return HCCL_SUCCESS;
    }

    uint32_t columnRankIndex = INVALID_VALUE_RANKID;
    for (uint32_t serverIndex = 0; serverIndex < serverGroups.size(); ++serverIndex) {
        if (HasSameRanks(serverGroups[serverIndex], topologyLocalRanks)) {
            if (columnRankIndex != INVALID_VALUE_RANKID) {
                return HCCL_SUCCESS;
            }
            columnRankIndex = serverIndex;
        }
    }
    if (columnRankIndex == INVALID_VALUE_RANKID) {
        return HCCL_SUCCESS;
    }

    std::vector<uint32_t> localRanks = serverGroups[columnRankIndex];
    auto localRankIt = std::find(localRanks.begin(), localRanks.end(), param.myRank);
    if (localRankIt == localRanks.end()) {
        return HCCL_SUCCESS;
    }
    const uint32_t localRankIndex = static_cast<uint32_t>(localRankIt - localRanks.begin());
    std::vector<uint32_t> columnRanks;
    columnRanks.reserve(serverGroupCount);
    for (const auto &serverGroup : serverGroups) {
        if (localRankIndex >= serverGroup.size()) {
            return HCCL_SUCCESS;
        }
        columnRanks.push_back(serverGroup[localRankIndex]);
    }
    if (!IsValidRankList(columnRanks, param.rankSize) || columnRanks[columnRankIndex] != param.myRank) {
        return HCCL_SUCCESS;
    }

    std::vector<HcclChannelDesc> localChannelDescs;
    std::vector<HcclChannelDesc> columnChannelDescs;
    ChannelGroup localGroup;
    ChannelGroup columnGroup;
    bool foundLocalChannels = false;
    bool foundColumnChannels = false;
    CHK_RET(SelectAxisChannelDescs(
        comm, param, netLayers, localRanks, localChannelDescs, localGroup.dieId, foundLocalChannels));
    CHK_RET(SelectAxisChannelDescs(
        comm, param, netLayers, columnRanks, columnChannelDescs, columnGroup.dieId, foundColumnChannels));
    if (!foundLocalChannels || !foundColumnChannels) {
        return HCCL_SUCCESS;
    }

    localGroup.channels.resize(localChannelDescs.size());
    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, localChannelDescs.data(),
        static_cast<uint32_t>(localChannelDescs.size()), localGroup.channels.data()));
    columnGroup.channels.resize(columnChannelDescs.size());
    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, columnChannelDescs.data(),
        static_cast<uint32_t>(columnChannelDescs.size()), columnGroup.channels.data()));

    localGroup.kernelKind = ops_hccl::CCU_REDUCE_SCATTER_KERNEL_OMNIPIPE_MESH;
    localGroup.axisRankSize = localGroupSize;
    localGroup.axisRankIndex = localRankIndex;
    for (const HcclChannelDesc &channelDesc : localChannelDescs) {
        auto rankIt = std::find(localRanks.begin(), localRanks.end(), channelDesc.remoteRank);
        if (rankIt == localRanks.end()) {
            return HCCL_SUCCESS;
        }
        localGroup.peerSubRanks.push_back(static_cast<uint32_t>(rankIt - localRanks.begin()));
    }
    columnGroup.kernelKind = ops_hccl::CCU_REDUCE_SCATTER_KERNEL_NHR;
    columnGroup.axisRankSize = serverGroupCount;
    columnGroup.axisRankIndex = columnRankIndex;
    for (const HcclChannelDesc &channelDesc : columnChannelDescs) {
        auto rankIt = std::find(columnRanks.begin(), columnRanks.end(), channelDesc.remoteRank);
        if (rankIt == columnRanks.end()) {
            return HCCL_SUCCESS;
        }
        columnGroup.peerSubRanks.push_back(static_cast<uint32_t>(rankIt - columnRanks.begin()));
    }

    resourceCtx.algorithm = REDUCE_SCATTER_ALGORITHM_PARALLEL_2D_OMNIPIPE;
    resourceCtx.localRanks = std::move(localRanks);
    resourceCtx.columnRanks = std::move(columnRanks);
    resourceCtx.globalRanks = std::move(globalRanks);
    resourceCtx.localRankIndex = localRankIndex;
    resourceCtx.columnRankIndex = columnRankIndex;
    std::vector<ChannelGroup> groups;
    groups.push_back(std::move(localGroup));
    groups.push_back(std::move(columnGroup));
    CHK_RET(AcquireThreadResources(comm, param, 3, resourceCtx));
    CHK_RET(RegisterCcuKernels(comm, param, groups, resourceCtx, true));
    built = true;
    return HCCL_SUCCESS;
}

HcclResult BuildFullMeshContext(HcclComm comm, const OpParam &param, AlgResourceCtx &resourceCtx)
{
    resourceCtx.algorithm = REDUCE_SCATTER_ALGORITHM_FULL_MESH;
    resourceCtx.localRanks.clear();
    resourceCtx.columnRanks.clear();
    resourceCtx.globalRanks.clear();
    resourceCtx.localRankIndex = INVALID_VALUE_RANKID;
    resourceCtx.columnRankIndex = INVALID_VALUE_RANKID;

    std::vector<ChannelGroup> groups;
    CHK_RET(AcquireChannelGroups(comm, param, groups));
    CHK_RET(AcquireThreadResources(comm, param, groups.empty() ? 1 : groups.size(), resourceCtx));
    return RegisterCcuKernels(comm, param, groups, resourceCtx, false);
}

HcclResult BuildResourceContext(HcclComm comm, const OpParam &param, AlgResourceCtx &resourceCtx)
{
    void *cclBufferAddr = nullptr;
    uint64_t cclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    resourceCtx.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};
    resourceCtx.ccuThread = param.cpuThread;

    bool builtParallel2dOmnipipe = false;
    CHK_RET(TryBuildParallel2dOmnipipeContext(comm, param, resourceCtx, builtParallel2dOmnipipe));
    if (builtParallel2dOmnipipe) {
        return HCCL_SUCCESS;
    }
    return BuildFullMeshContext(comm, param, resourceCtx);
}

} // namespace

HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op,
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
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_RET(ValidateOpParam(param));

    int tagLength = std::snprintf(param.tag, sizeof(param.tag), "hccl_custom_reducescatter_2d_v5_%u_%u_%u_%llu",
        static_cast<uint32_t>(dataType), static_cast<uint32_t>(op), param.rankSize,
        static_cast<unsigned long long>(recvCount));
    CHK_PRT_RET(tagLength < 0 || static_cast<size_t>(tagLength) >= sizeof(param.tag),
        HCCL_ERROR("[HcclReduceScatter] failed to construct resource tag"), HCCL_E_INTERNAL);

    CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;
    CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, stream, 2, &param.cpuThread));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, ccuEngine, &ctx, &size) == HCCL_SUCCESS) {
        HCCL_INFO("[HcclReduceScatter] reuse CCU engine context");
        param.resCtx = ctx;
        param.ctxSize = size;
    } else {
        AlgResourceCtx resCtxHost;
        CHK_RET(BuildResourceContext(comm, param, resCtxHost));
        std::vector<char> seq = resCtxHost.Serialize();
        uint64_t seqSize = seq.size();
        param.ctxSize = seqSize;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, ccuEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, ccuEngine, param.tag, seq.data(), seqSize, 0));
    }

    CHK_RET(ops_hccl::ExecOp(param));
    return HCCL_SUCCESS;
}
