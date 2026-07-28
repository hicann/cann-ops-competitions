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
#include <cstdio>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>

#include "ccu_launch.h"
#include "hccl_ccu_res.h"

#include "common.h"
#include "custom.h"
#include "exec_op.h"
#include "hccl.h"
#include "log.h"
#include "../op_kernel_ccu/ccu_kernel.h"

namespace {
constexpr uint32_t DEFAULT_MAIN_THREAD_NOTIFY_NUM = 0;
constexpr uint32_t HIERARCHY_THREAD_NOTIFY_NUM = 1;
constexpr uint32_t CCU_THREAD_NUM = 1;
constexpr uint32_t CCU_DIE_ID = 0;
constexpr uint32_t CCU_KERNEL_ARG_NUM = 1;
constexpr uint32_t INVALID_DIE_ID = 0xFFFFFFFF;

struct ChannelCandidate {
    HcclChannelDesc desc;
    uint32_t dieId = INVALID_DIE_ID;
    uint32_t bwCoeff = 0;
};

struct BroadcastHierarchyTopology {
    uint32_t layer0 = 0;
    uint32_t layer1 = 0;
    uint32_t laneCount = 0;
    bool isSourceGroup = false;
    std::vector<uint32_t> localRanks;
    std::vector<uint32_t> remoteRanks;
    std::vector<uint32_t> sourceGatewayRanks;
    std::vector<uint32_t> destinationGatewayRanks;
};

bool ContainsRank(const std::vector<uint32_t> &ranks, uint32_t rank)
{
    return std::find(ranks.begin(), ranks.end(), rank) != ranks.end();
}

HcclResult ValidateHierarchyRanks(const OpParam &param, const std::vector<uint32_t> &localRanks)
{
    CHK_PRT_RET(localRanks.empty() || localRanks.size() >= param.rankSize,
        HCCL_ERROR("[ValidateHierarchyRanks] invalid local rank count[%zu]", localRanks.size()),
        HCCL_E_NOT_SUPPORT);
    std::vector<bool> seen(param.rankSize, false);
    for (uint32_t rank : localRanks) {
        CHK_PRT_RET(rank >= param.rankSize || seen[rank],
            HCCL_ERROR("[ValidateHierarchyRanks] invalid or duplicate rank[%u]", rank), HCCL_E_PARA);
        seen[rank] = true;
    }
    CHK_PRT_RET(!seen[param.myRank],
        HCCL_ERROR("[ValidateHierarchyRanks] local group misses myRank[%u]", param.myRank), HCCL_E_PARA);

    const size_t remoteSize = param.rankSize - localRanks.size();
    const bool valid2x8 = param.rankSize == BROADCAST_PARALLEL_2D_RANK_SIZE &&
        localRanks.size() == BROADCAST_PARALLEL_2D_LANE_COUNT &&
        remoteSize == BROADCAST_PARALLEL_2D_LANE_COUNT;
    const size_t smallSize = std::min(localRanks.size(), remoteSize);
    const size_t largeSize = std::max(localRanks.size(), remoteSize);
    const bool valid8p4 = param.rankSize == BROADCAST_ASYMMETRIC_2D_RANK_SIZE &&
        smallSize == BROADCAST_ASYMMETRIC_2D_SMALL_GROUP_SIZE &&
        largeSize == BROADCAST_PARALLEL_2D_LANE_COUNT;
    CHK_PRT_RET(!valid2x8 && !valid8p4,
        HCCL_ERROR("[ValidateHierarchyRanks] unsupported topology rankSize[%u], groupSizes[%zu,%zu]",
            param.rankSize, localRanks.size(), remoteSize), HCCL_E_NOT_SUPPORT);
    return HCCL_SUCCESS;
}

HcclResult BuildBroadcastHierarchyTopology(
    HcclComm comm, const OpParam &param, BroadcastHierarchyTopology &topology)
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    CHK_PRT_RET(netLayers == nullptr || netLayerNum < 2,
        HCCL_ERROR("[BuildBroadcastHierarchyTopology] expected two layers, actual[%u]", netLayerNum),
        HCCL_E_NOT_SUPPORT);
    topology.layer0 = netLayers[0];
    topology.layer1 = netLayers[1];

    uint32_t *localRanks = nullptr;
    uint32_t localRankNum = 0;
    CHK_RET(HcclRankGraphGetRanksByLayer(comm, topology.layer0, &localRanks, &localRankNum));
    CHK_PRT_RET(localRanks == nullptr || localRankNum == 0,
        HCCL_ERROR("[BuildBroadcastHierarchyTopology] empty layer-0 rank list"), HCCL_E_NOT_FOUND);
    topology.localRanks.assign(localRanks, localRanks + localRankNum);
    std::sort(topology.localRanks.begin(), topology.localRanks.end());
    CHK_RET(ValidateHierarchyRanks(param, topology.localRanks));

    std::vector<bool> isLocal(param.rankSize, false);
    for (uint32_t rank : topology.localRanks) {
        isLocal[rank] = true;
    }
    topology.remoteRanks.clear();
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (!isLocal[rank]) {
            topology.remoteRanks.push_back(rank);
        }
    }

    topology.isSourceGroup = ContainsRank(topology.localRanks, param.root);
    const auto &sourceRanks = topology.isSourceGroup ? topology.localRanks : topology.remoteRanks;
    const auto &destinationRanks = topology.isSourceGroup ? topology.remoteRanks : topology.localRanks;
    topology.laneCount = static_cast<uint32_t>(std::max(sourceRanks.size(), destinationRanks.size()));
    topology.sourceGatewayRanks.clear();
    topology.destinationGatewayRanks.clear();
    topology.sourceGatewayRanks.reserve(topology.laneCount);
    topology.destinationGatewayRanks.reserve(topology.laneCount);
    for (uint32_t laneIdx = 0; laneIdx < topology.laneCount; ++laneIdx) {
        // 8+4 拓扑用 8 个 lane；4 卡组的每个 rank 重复承载两个不同 lane。
        topology.sourceGatewayRanks.push_back(sourceRanks[laneIdx % sourceRanks.size()]);
        topology.destinationGatewayRanks.push_back(destinationRanks[laneIdx % destinationRanks.size()]);
    }

    HCCL_INFO("[BuildBroadcastHierarchyTopology] rank[%u] root[%u] layer0[%u] layer1[%u] "
              "localSize[%zu] remoteSize[%zu] laneCount[%u] sourceGroup[%u]",
        param.myRank, param.root, topology.layer0, topology.layer1, topology.localRanks.size(),
        topology.remoteRanks.size(), topology.laneCount, topology.isSourceGroup ? 1U : 0U);
    return HCCL_SUCCESS;
}

HcclResult FillChannelDesc(
    const CommLink &link, uint32_t remoteRank, uint32_t notifyNum, HcclChannelDesc &channelDesc)
{
    CHK_RET(HcclChannelDescInit(&channelDesc, 1));
    channelDesc.remoteRank = remoteRank;
    channelDesc.channelProtocol = link.linkAttr.linkProtocol;
    channelDesc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
    channelDesc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
    channelDesc.localEndpoint.loc = link.srcEndpointDesc.loc;
    channelDesc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
    channelDesc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
    channelDesc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
    channelDesc.notifyNum = notifyNum;
    return HCCL_SUCCESS;
}

HcclResult GetChannelDieId(HcclComm comm, uint32_t localRank, const HcclChannelDesc &channelDesc, uint32_t &dieId)
{
    EndpointAttrDieId endpointDieId = 0;
    CHK_RET(HcclRankGraphGetEndpointInfo(comm, localRank, &channelDesc.localEndpoint, ENDPOINT_ATTR_DIE_ID,
        sizeof(endpointDieId), &endpointDieId));
    dieId = endpointDieId;
    return HCCL_SUCCESS;
}

HcclResult GetChannelBwCoeff(HcclComm comm, uint32_t localRank, const HcclChannelDesc &channelDesc, uint32_t &bwCoeff)
{
    EndpointAttrBwCoeff endpointBwCoeff = 0;
    HcclResult ret = HcclRankGraphGetEndpointInfo(comm, localRank, &channelDesc.localEndpoint, ENDPOINT_ATTR_BW_COEFF,
        sizeof(endpointBwCoeff), &endpointBwCoeff);
    if (ret != HCCL_SUCCESS) {
        bwCoeff = 0;
        return HCCL_SUCCESS;
    }
    bwCoeff = endpointBwCoeff;
    return HCCL_SUCCESS;
}

HcclResult AddCandidateIfNewDie(HcclComm comm, uint32_t localRank, uint32_t remoteRank, const CommLink &link,
    std::vector<ChannelCandidate> &candidates)
{
    HcclChannelDesc desc;
    CHK_RET(FillChannelDesc(link, remoteRank, BROADCAST_KERNEL_NOTIFY_NUM, desc));

    uint32_t dieId = INVALID_DIE_ID;
    CHK_RET(GetChannelDieId(comm, localRank, desc, dieId));
    uint32_t bwCoeff = 0;
    CHK_RET(GetChannelBwCoeff(comm, localRank, desc, bwCoeff));
    for (auto &candidate : candidates) {
        if (candidate.dieId == dieId) {
            // 同一个 die 可能存在多条链路，保留带宽系数更高的候选。
            if (bwCoeff > candidate.bwCoeff) {
                candidate = ChannelCandidate{desc, dieId, bwCoeff};
            }
            return HCCL_SUCCESS;
        }
    }

    candidates.push_back(ChannelCandidate{desc, dieId, bwCoeff});
    return HCCL_SUCCESS;
}

HcclResult CollectChannelCandidatesOnLayers(HcclComm comm, uint32_t localRank, uint32_t remoteRank,
    const std::vector<uint32_t> &netLayers, std::vector<ChannelCandidate> &candidates)
{
    constexpr std::array<CommProtocol, 2> supportProtocols = {
        CommProtocol::COMM_PROTOCOL_UBC_CTP,
        CommProtocol::COMM_PROTOCOL_UBC_TP,
    };

    for (const CommProtocol protocol : supportProtocols) {
        const size_t before = candidates.size();
        for (uint32_t netLayer : netLayers) {
            CommLink *links = nullptr;
            uint32_t linkNum = 0;
            CHK_RET(HcclRankGraphGetLinks(comm, netLayer, localRank, remoteRank, &links, &linkNum));
            for (uint32_t linkIdx = 0; linkIdx < linkNum; ++linkIdx) {
                if (links[linkIdx].linkAttr.linkProtocol == protocol) {
                    CHK_RET(AddCandidateIfNewDie(comm, localRank, remoteRank, links[linkIdx], candidates));
                }
            }
        }
        if (candidates.size() > before) {
            return HCCL_SUCCESS;
        }
    }

    HCCL_ERROR("[BuildChannelDesc] no CCU link between rank %u and rank %u", localRank, remoteRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult GetAllNetworkLayers(HcclComm comm, std::vector<uint32_t> &layers)
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    CHK_PRT_RET(netLayers == nullptr || netLayerNum == 0,
        HCCL_ERROR("[GetAllNetworkLayers] no network layers"), HCCL_E_NOT_FOUND);
    layers.assign(netLayers, netLayers + netLayerNum);
    return HCCL_SUCCESS;
}

HcclResult SelectSameDieChannels(const OpParam &param, const std::vector<std::vector<ChannelCandidate>> &allCandidates,
    std::vector<HcclChannelDesc> &channelDescs, uint32_t &selectedDie)
{
    selectedDie = INVALID_DIE_ID;
    uint64_t selectedBw = 0;
    for (uint32_t dieId = 0; dieId < 2; ++dieId) {
        bool dieCoversAllRanks = true;
        uint64_t totalBw = 0;
        for (const auto &rankCandidates : allCandidates) {
            bool found = false;
            uint32_t bestBw = 0;
            for (const auto &candidate : rankCandidates) {
                if (candidate.dieId == dieId) {
                    found = true;
                    bestBw = std::max(bestBw, candidate.bwCoeff);
                }
            }
            if (!found) {
                dieCoversAllRanks = false;
                break;
            }
            totalBw += bestBw;
        }
        if (dieCoversAllRanks && (selectedDie == INVALID_DIE_ID || totalBw > selectedBw)) {
            selectedDie = dieId;
            selectedBw = totalBw;
        }
    }
    CHK_PRT_RET(selectedDie == INVALID_DIE_ID,
        HCCL_ERROR("[SelectSameDieChannels] cannot find one die covering all peers, myRank[%u]", param.myRank),
        HCCL_E_NOT_FOUND);

    channelDescs.clear();
    channelDescs.reserve(allCandidates.size());
    for (const auto &rankCandidates : allCandidates) {
        const ChannelCandidate *bestCandidate = nullptr;
        for (const auto &candidate : rankCandidates) {
            if (candidate.dieId != selectedDie) {
                continue;
            }
            if (bestCandidate == nullptr || candidate.bwCoeff > bestCandidate->bwCoeff) {
                bestCandidate = &candidate;
            }
        }
        CHK_PRT_RET(bestCandidate == nullptr,
            HCCL_ERROR("[SelectSameDieChannels] selected die[%u] missing candidate", selectedDie), HCCL_E_INTERNAL);
        channelDescs.push_back(bestCandidate->desc);
    }
    HCCL_INFO("[SelectSameDieChannels] rank[%u] selects die[%u], channelNum[%zu]", param.myRank, selectedDie,
        channelDescs.size());
    return HCCL_SUCCESS;
}

HcclResult CreatePeerChannels(HcclComm comm, CommEngine engine, const OpParam &param,
    const std::vector<uint32_t> &peerRanks, const std::vector<uint32_t> &netLayers,
    std::vector<ChannelHandle> &channels, uint32_t &selectedDie)
{
    if (peerRanks.empty()) {
        channels.clear();
        selectedDie = INVALID_DIE_ID;
        return HCCL_SUCCESS;
    }

    std::vector<std::vector<ChannelCandidate>> allCandidates;
    allCandidates.reserve(peerRanks.size());
    std::vector<bool> seen(param.rankSize, false);
    for (uint32_t remoteRank : peerRanks) {
        CHK_PRT_RET(remoteRank >= param.rankSize || remoteRank == param.myRank || seen[remoteRank],
            HCCL_ERROR("[CreatePeerChannels] invalid or duplicate peer[%u]", remoteRank), HCCL_E_PARA);
        seen[remoteRank] = true;
        std::vector<ChannelCandidate> candidates;
        CHK_RET(CollectChannelCandidatesOnLayers(comm, param.myRank, remoteRank, netLayers, candidates));
        allCandidates.push_back(candidates);
    }

    std::vector<HcclChannelDesc> channelDescs;
    CHK_RET(SelectSameDieChannels(param, allCandidates, channelDescs, selectedDie));
    channels.resize(channelDescs.size());
    CHK_RET(HcclChannelAcquire(
        comm, engine, channelDescs.data(), static_cast<uint32_t>(channelDescs.size()), channels.data()));
    return HCCL_SUCCESS;
}

HcclResult CreateAllRankChannels(
    HcclComm comm, CommEngine engine, const OpParam &param, std::vector<ChannelHandle> &channels)
{
    if (param.rankSize <= 1) {
        channels.clear();
        return HCCL_SUCCESS;
    }
    std::vector<uint32_t> peerRanks;
    peerRanks.reserve(param.rankSize - 1);
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank != param.myRank) {
            peerRanks.push_back(remoteRank);
        }
    }
    std::vector<uint32_t> netLayers;
    CHK_RET(GetAllNetworkLayers(comm, netLayers));
    uint32_t selectedDie = INVALID_DIE_ID;
    return CreatePeerChannels(comm, engine, param, peerRanks, netLayers, channels, selectedDie);
}

uint32_t GetNhrStepNum(uint32_t rankSize)
{
    uint32_t stepNum = 0;
    uint32_t coveredRanks = 1;
    while (coveredRanks < rankSize) {
        coveredRanks <<= 1;
        ++stepNum;
    }
    return stepNum;
}

void BuildNhrScatterStep(
    const OpParam &param, uint32_t step, uint32_t stepNum, BroadcastNhrStepInfo &stepInfo)
{
    const uint32_t rankSize = param.rankSize;
    const uint32_t deltaRoot = (param.root + rankSize - param.myRank) % rankSize;
    const uint32_t deltaRankPair = 1U << step;
    const uint32_t sliceNum = (rankSize - 1 + deltaRankPair) / (1U << (step + 1));
    const uint32_t deltaSliceIndex = 1U << (step + 1);
    const bool isPowerOfTwo = (rankSize & (rankSize - 1)) == 0;
    const uint32_t activeRankNum = (!isPowerOfTwo && step + 1 == stepNum) ?
        (rankSize - deltaRankPair) : deltaRankPair;

    stepInfo.step = step;
    stepInfo.toRank = rankSize;
    stepInfo.fromRank = rankSize;
    if (deltaRoot < activeRankNum) {
        stepInfo.toRank = (param.myRank + rankSize - deltaRankPair) % rankSize;
        uint32_t sliceIdx = stepInfo.toRank;
        for (uint32_t idx = 0; idx < sliceNum; ++idx) {
            stepInfo.txSliceIdxs.push_back(sliceIdx);
            sliceIdx = (sliceIdx + rankSize - deltaSliceIndex) % rankSize;
        }
    } else if (deltaRoot >= deltaRankPair && deltaRoot < activeRankNum + deltaRankPair) {
        stepInfo.fromRank = (param.myRank + deltaRankPair) % rankSize;
        uint32_t sliceIdx = param.myRank;
        for (uint32_t idx = 0; idx < sliceNum; ++idx) {
            stepInfo.rxSliceIdxs.push_back(sliceIdx);
            sliceIdx = (sliceIdx + rankSize - deltaSliceIndex) % rankSize;
        }
    }
}

void BuildNhrAllGatherStep(
    const OpParam &param, uint32_t step, uint32_t stepNum, BroadcastNhrStepInfo &stepInfo)
{
    const uint32_t rankSize = param.rankSize;
    const uint32_t deltaRank = 1U << (stepNum - 1 - step);
    const uint32_t sliceNum = (rankSize - 1 + deltaRank) / (1U << (stepNum - step));
    const uint32_t deltaSliceIndex = 1U << (stepNum - step);

    stepInfo.step = step;
    stepInfo.toRank = (param.myRank + deltaRank) % rankSize;
    stepInfo.fromRank = (param.myRank + rankSize - deltaRank) % rankSize;
    uint32_t txSliceIdx = param.myRank;
    uint32_t rxSliceIdx = stepInfo.fromRank;
    for (uint32_t idx = 0; idx < sliceNum; ++idx) {
        stepInfo.txSliceIdxs.push_back(txSliceIdx);
        stepInfo.rxSliceIdxs.push_back(rxSliceIdx);
        txSliceIdx = (txSliceIdx + rankSize - deltaSliceIndex) % rankSize;
        rxSliceIdx = (rxSliceIdx + rankSize - deltaSliceIndex) % rankSize;
    }
}

std::vector<BroadcastNhrStepInfo> BuildNhrStepInfo(const OpParam &param)
{
    // 参考官方 Broadcast NHR：先递归减半 Scatter，再反向递归倍增 AllGather。
    // 步骤在 Host 侧固化进 kernelArg，CCU 执行时无需动态计算拓扑。
    const uint32_t stepNum = GetNhrStepNum(param.rankSize);
    std::vector<BroadcastNhrStepInfo> stepInfoVector;
    stepInfoVector.reserve(stepNum * 2);
    for (uint32_t step = 0; step < stepNum; ++step) {
        BroadcastNhrStepInfo stepInfo;
        BuildNhrScatterStep(param, step, stepNum, stepInfo);
        stepInfoVector.push_back(std::move(stepInfo));
    }
    for (uint32_t step = 0; step < stepNum; ++step) {
        BroadcastNhrStepInfo stepInfo;
        BuildNhrAllGatherStep(param, step, stepNum, stepInfo);
        stepInfoVector.push_back(std::move(stepInfo));
    }
    return stepInfoVector;
}

HcclResult SetKernelInfoName(CcuKernelInfo &kernelInfo, const char *kernelName)
{
    const int nameLen = std::snprintf(
        kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "%s", kernelName);
    CHK_PRT_RET(nameLen < 0 || static_cast<size_t>(nameLen) >= sizeof(kernelInfo.kernelFuncName),
        HCCL_ERROR("[SetKernelInfoName] failed to set kernel name"), HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult RegisterKernelInfos(
    HcclComm comm, const std::vector<CcuKernelInfo> &kernelInfos, AlgResourceCtx &resCtx)
{
    CHK_PRT_RET(kernelInfos.empty(), HCCL_ERROR("[RegisterKernelInfos] no kernels"), HCCL_E_PARA);
    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1,
        HCCL_ERROR("[RegisterKernelInfos] unexpected CCU instance num %u", insNum), HCCL_E_INTERNAL);

    CcuResult regStartRet = HcommCcuKernelRegisterStart(insHandle);
    if (regStartRet != CCU_SUCCESS) {
        HCCL_ERROR("[RegisterKernelInfos] register start failed, ccuRet[%d]", regStartRet);
        return ConvertCcuToHccl(regStartRet);
    }

    resCtx.ccuKernels.clear();
    resCtx.ccuKernels.reserve(kernelInfos.size());
    for (uint32_t kernelIdx = 0; kernelIdx < kernelInfos.size(); ++kernelIdx) {
        CcuKernelHandle kernelHandle = 0;
        const void *kernelArgs[] = {kernelInfos[kernelIdx].kernelArg};
        CcuResult regRet = HcommCcuKernelRegister(insHandle, CCU_DIE_ID,
            kernelInfos[kernelIdx].kernelFuncName, kernelInfos[kernelIdx].kernelFunc,
            kernelArgs, CCU_KERNEL_ARG_NUM, &kernelHandle);
        if (regRet != CCU_SUCCESS) {
            HCCL_ERROR("[RegisterKernelInfos] register kernel[%u], algorithmMode[%u] failed, ccuRet[%d]",
                kernelIdx, resCtx.algorithmMode, regRet);
            return ConvertCcuToHccl(regRet);
        }
        resCtx.ccuKernels.push_back(kernelHandle);
    }

    CcuResult regEndRet = HcommCcuKernelRegisterEnd(insHandle);
    if (regEndRet != CCU_SUCCESS) {
        HCCL_ERROR("[RegisterKernelInfos] register end failed, ccuRet[%d]", regEndRet);
        return ConvertCcuToHccl(regEndRet);
    }
    return HCCL_SUCCESS;
}

void FillCommonKernelArg(const OpParam &param, const std::vector<ChannelHandle> &channels,
    CcuBroadcastMesh1DMem2MemKernelArg &kernelArg)
{
    kernelArg.rankSize = param.rankSize;
    kernelArg.rankId = param.myRank;
    kernelArg.rootId = param.root;
    kernelArg.channelCount = static_cast<uint32_t>(channels.size());
    for (uint32_t channelIdx = 0; channelIdx < kernelArg.channelCount; ++channelIdx) {
        kernelArg.channels[channelIdx] = channels[channelIdx];
    }
}

HcclResult RegisterBroadcastKernel(
    HcclComm comm, const OpParam &param, const std::vector<ChannelHandle> &channels, AlgResourceCtx &resCtx)
{
    CcuKernelInfo kernelInfo{};
    const char *kernelName = nullptr;
    switch (resCtx.algorithmMode) {
        case BROADCAST_ALG_DIRECT:
            kernelName = "CcuBroadcastDirectKernel";
            kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuBroadcastDirectKernel);
            break;
        case BROADCAST_ALG_NHR1D_MEM2MEM:
            kernelName = "CcuBroadcastNhr1DMem2MemKernel";
            kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuBroadcastNhr1DMem2MemKernel);
            break;
        case BROADCAST_ALG_MESH1D_MEM2MEM:
            kernelName = "CcuBroadcastMesh1DMem2MemKernel";
            kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuBroadcastMesh1DMem2MemKernel);
            break;
        default:
            HCCL_ERROR("[RegisterBroadcastKernel] unsupported algorithmMode[%u]", resCtx.algorithmMode);
            return HCCL_E_NOT_SUPPORT;
    }
    CHK_RET(SetKernelInfoName(kernelInfo, kernelName));

    if (resCtx.algorithmMode == BROADCAST_ALG_NHR1D_MEM2MEM) {
        auto kernelArg = std::make_shared<CcuBroadcastNhr1DMem2MemKernelArg>();
        FillCommonKernelArg(param, channels, *kernelArg);
        kernelArg->stepInfoVector = BuildNhrStepInfo(param);
        kernelInfo.setKernelArg(kernelArg);
    } else {
        auto kernelArg = std::make_shared<CcuBroadcastMesh1DMem2MemKernelArg>();
        FillCommonKernelArg(param, channels, *kernelArg);
        kernelInfo.setKernelArg(kernelArg);
    }
    return RegisterKernelInfos(comm, {kernelInfo}, resCtx);
}

HcclResult FillParallelDirectKernelArg(const OpParam &param, const std::vector<uint32_t> &peerRanks,
    const std::vector<ChannelHandle> &channels, CcuBroadcastParallelDirectKernelArg &kernelArg)
{
    CHK_PRT_RET(peerRanks.empty() || peerRanks.size() != channels.size() || peerRanks.size() > MAX_RANK_SIZE,
        HCCL_ERROR("[FillParallelDirectKernelArg] invalid peer/channel count[%zu,%zu]",
            peerRanks.size(), channels.size()), HCCL_E_PARA);
    kernelArg.rankSize = param.rankSize;
    kernelArg.rankId = param.myRank;
    kernelArg.rootId = param.root;
    kernelArg.channelCount = static_cast<uint32_t>(channels.size());
    for (uint32_t channelIdx = 0; channelIdx < channels.size(); ++channelIdx) {
        kernelArg.channels[channelIdx] = channels[channelIdx];
        kernelArg.peerRanks[channelIdx] = peerRanks[channelIdx];
    }
    return HCCL_SUCCESS;
}

HcclResult AppendParallelDirectKernel(HcclComm comm, CommEngine engine, const OpParam &param,
    const std::vector<uint32_t> &peerRanks, uint32_t netLayer, const char *kernelName,
    void *kernelFunc, std::vector<CcuKernelInfo> &kernelInfos, uint32_t &selectedDie)
{
    std::vector<ChannelHandle> channels;
    CHK_RET(CreatePeerChannels(comm, engine, param, peerRanks, {netLayer}, channels, selectedDie));

    CcuKernelInfo kernelInfo{};
    CHK_RET(SetKernelInfoName(kernelInfo, kernelName));
    kernelInfo.kernelFunc = kernelFunc;
    auto kernelArg = std::make_shared<CcuBroadcastParallelDirectKernelArg>();
    CHK_RET(FillParallelDirectKernelArg(param, peerRanks, channels, *kernelArg));
    kernelInfo.setKernelArg(kernelArg);
    kernelInfos.push_back(kernelInfo);
    return HCCL_SUCCESS;
}

HcclResult CreateParallelDirectResource(HcclComm comm, CommEngine engine, const OpParam &param,
    const BroadcastHierarchyTopology &topology, AlgResourceCtx &resCtx)
{
    std::vector<CcuKernelInfo> kernelInfos;
    if (param.myRank == param.root) {
        std::vector<uint32_t> layer0Peers;
        layer0Peers.reserve(topology.localRanks.size() - 1);
        for (uint32_t rank : topology.localRanks) {
            if (rank != param.root) {
                layer0Peers.push_back(rank);
            }
        }
        CHK_PRT_RET(layer0Peers.empty() || topology.remoteRanks.empty(),
            HCCL_ERROR("[CreateParallelDirectResource] root[%u] has an empty layer peer set", param.root),
            HCCL_E_INTERNAL);

        uint32_t layer0Die = INVALID_DIE_ID;
        CHK_RET(AppendParallelDirectKernel(comm, engine, param, layer0Peers, topology.layer0,
            "CcuBroadcastParallelDirectLayer0Kernel",
            reinterpret_cast<void *>(ops_hccl::CcuBroadcastParallelDirectLayer0Kernel),
            kernelInfos, layer0Die));
        uint32_t layer1Die = INVALID_DIE_ID;
        CHK_RET(AppendParallelDirectKernel(comm, engine, param, topology.remoteRanks, topology.layer1,
            "CcuBroadcastParallelDirectLayer1Kernel",
            reinterpret_cast<void *>(ops_hccl::CcuBroadcastParallelDirectLayer1Kernel),
            kernelInfos, layer1Die));
        if (layer0Die == layer1Die) {
            HCCL_WARNING("[CreateParallelDirectResource] layer-0/layer-1 resolved to the same die[%u]",
                layer0Die);
        }
        HCCL_INFO("[CreateParallelDirectResource] root[%u] layer0Peers[%zu] layer1Peers[%zu] dies[%u,%u]",
            param.root, layer0Peers.size(), topology.remoteRanks.size(), layer0Die, layer1Die);
    } else {
        const bool rootIsLocal = ContainsRank(topology.localRanks, param.root);
        const uint32_t netLayer = rootIsLocal ? topology.layer0 : topology.layer1;
        const char *kernelName = rootIsLocal ?
            "CcuBroadcastParallelDirectLayer0Kernel" : "CcuBroadcastParallelDirectLayer1Kernel";
        void *kernelFunc = rootIsLocal ?
            reinterpret_cast<void *>(ops_hccl::CcuBroadcastParallelDirectLayer0Kernel) :
            reinterpret_cast<void *>(ops_hccl::CcuBroadcastParallelDirectLayer1Kernel);
        uint32_t selectedDie = INVALID_DIE_ID;
        CHK_RET(AppendParallelDirectKernel(
            comm, engine, param, {param.root}, netLayer, kernelName, kernelFunc, kernelInfos, selectedDie));
        HCCL_INFO("[CreateParallelDirectResource] rank[%u] root[%u] layer[%u] die[%u]",
            param.myRank, param.root, rootIsLocal ? 0U : 1U, selectedDie);
    }
    return RegisterKernelInfos(comm, kernelInfos, resCtx);
}

std::vector<uint32_t> FindHierarchyCrossPeers(
    const BroadcastHierarchyTopology &topology, uint32_t rankId)
{
    std::vector<uint32_t> peers;
    for (uint32_t laneIdx = 0; laneIdx < topology.laneCount; ++laneIdx) {
        uint32_t peer = INVALID_VALUE_RANKID;
        if (topology.sourceGatewayRanks[laneIdx] == rankId) {
            peer = topology.destinationGatewayRanks[laneIdx];
        } else if (topology.destinationGatewayRanks[laneIdx] == rankId) {
            peer = topology.sourceGatewayRanks[laneIdx];
        }
        if (peer != INVALID_VALUE_RANKID && !ContainsRank(peers, peer)) {
            peers.push_back(peer);
        }
    }
    return peers;
}

HcclResult FillHierarchicalKernelArg(const OpParam &param, const BroadcastHierarchyTopology &topology,
    const std::vector<uint32_t> &peerRanks, const std::vector<ChannelHandle> &channels,
    CcuBroadcastHierarchicalKernelArg &kernelArg)
{
    CHK_PRT_RET(peerRanks.size() != channels.size() || peerRanks.size() > MAX_RANK_SIZE,
        HCCL_ERROR("[FillHierarchicalKernelArg] invalid peer/channel count[%zu,%zu]",
            peerRanks.size(), channels.size()), HCCL_E_PARA);
    kernelArg.rankSize = param.rankSize;
    kernelArg.rankId = param.myRank;
    kernelArg.rootId = param.root;
    kernelArg.localRankCount = static_cast<uint32_t>(topology.localRanks.size());
    kernelArg.laneCount = topology.laneCount;
    kernelArg.isSourceGroup = topology.isSourceGroup ? 1U : 0U;
    kernelArg.channelCount = static_cast<uint32_t>(channels.size());
    for (uint32_t idx = 0; idx < topology.localRanks.size(); ++idx) {
        kernelArg.localRanks[idx] = topology.localRanks[idx];
    }
    for (uint32_t laneIdx = 0; laneIdx < topology.laneCount; ++laneIdx) {
        kernelArg.sourceGatewayRanks[laneIdx] = topology.sourceGatewayRanks[laneIdx];
        kernelArg.destinationGatewayRanks[laneIdx] = topology.destinationGatewayRanks[laneIdx];
    }
    for (uint32_t channelIdx = 0; channelIdx < channels.size(); ++channelIdx) {
        kernelArg.channels[channelIdx] = channels[channelIdx];
        kernelArg.peerRanks[channelIdx] = peerRanks[channelIdx];
    }
    return HCCL_SUCCESS;
}

HcclResult CreateHierarchicalResource(HcclComm comm, CommEngine engine, const OpParam &param,
    const BroadcastHierarchyTopology &topology, AlgResourceCtx &resCtx)
{
    std::vector<uint32_t> layer0Peers;
    layer0Peers.reserve(topology.localRanks.size() - 1);
    for (uint32_t rank : topology.localRanks) {
        if (rank != param.myRank) {
            layer0Peers.push_back(rank);
        }
    }
    std::vector<ChannelHandle> layer0Channels;
    uint32_t layer0Die = INVALID_DIE_ID;
    CHK_RET(CreatePeerChannels(
        comm, engine, param, layer0Peers, {topology.layer0}, layer0Channels, layer0Die));

    std::vector<CcuKernelInfo> kernelInfos;
    CcuKernelInfo layer0KernelInfo{};
    CHK_RET(SetKernelInfoName(layer0KernelInfo, "CcuBroadcastHierarchicalLayer0Kernel"));
    layer0KernelInfo.kernelFunc =
        reinterpret_cast<void *>(ops_hccl::CcuBroadcastHierarchicalLayer0Kernel);
    auto layer0KernelArg = std::make_shared<CcuBroadcastHierarchicalKernelArg>();
    CHK_RET(FillHierarchicalKernelArg(
        param, topology, layer0Peers, layer0Channels, *layer0KernelArg));
    layer0KernelInfo.setKernelArg(layer0KernelArg);
    kernelInfos.push_back(layer0KernelInfo);

    const std::vector<uint32_t> layer1Peers = FindHierarchyCrossPeers(topology, param.myRank);
    CHK_PRT_RET(layer1Peers.empty(),
        HCCL_ERROR("[CreateHierarchicalResource] rank[%u] has no layer-1 peer", param.myRank),
        HCCL_E_INTERNAL);
    resCtx.hierarchyHasLayer1Kernel = 1;
    uint32_t layer1Die = INVALID_DIE_ID;
    std::vector<ChannelHandle> layer1Channels;
    CHK_RET(CreatePeerChannels(
        comm, engine, param, layer1Peers, {topology.layer1}, layer1Channels, layer1Die));
    CcuKernelInfo layer1KernelInfo{};
    CHK_RET(SetKernelInfoName(layer1KernelInfo, "CcuBroadcastHierarchicalLayer1Kernel"));
    layer1KernelInfo.kernelFunc =
        reinterpret_cast<void *>(ops_hccl::CcuBroadcastHierarchicalLayer1Kernel);
    auto layer1KernelArg = std::make_shared<CcuBroadcastHierarchicalKernelArg>();
    CHK_RET(FillHierarchicalKernelArg(
        param, topology, layer1Peers, layer1Channels, *layer1KernelArg));
    layer1KernelInfo.setKernelArg(layer1KernelArg);
    kernelInfos.push_back(layer1KernelInfo);
    if (layer0Die == layer1Die) {
        HCCL_WARNING("[CreateHierarchicalResource] layer-0/layer-1 resolved to the same die[%u]",
            layer0Die);
    }

    resCtx.hierarchyLaneCount = topology.laneCount;
    HCCL_INFO("[CreateHierarchicalResource] rank[%u] layer0Die[%u] layer1Die[%u] kernelNum[%zu]",
        param.myRank, layer0Die, layer1Die, kernelInfos.size());
    return RegisterKernelInfos(comm, kernelInfos, resCtx);
}

HcclResult CreateAlgResource(HcclComm comm, CommEngine engine, const OpParam &param,
    uint32_t algorithmMode, const BroadcastHierarchyTopology *topology, AlgResourceCtx &resCtx)
{
    void *cclBufferAddr = nullptr;
    uint64_t cclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    resCtx.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

    resCtx.algorithmMode = algorithmMode;
    resCtx.ccuThread = param.cpuThread;
    resCtx.threads.resize(CCU_THREAD_NUM);
    resCtx.threads[0] = param.cpuThread;

    if (param.rankSize <= 1) {
        resCtx.ccuKernels.clear();
        return HCCL_SUCCESS;
    }

    if (resCtx.algorithmMode == BROADCAST_ALG_PARALLEL_HIERARCHICAL) {
        CHK_PRT_RET(topology == nullptr,
            HCCL_ERROR("[CreateAlgResource] missing hierarchy topology"), HCCL_E_INTERNAL);
        resCtx.threads.resize(2);
        CHK_RET(HcclThreadAcquire(
            comm, engine, 1, HIERARCHY_THREAD_NOTIFY_NUM, &resCtx.threads[1]));
        CHK_RET(CreateHierarchicalResource(comm, engine, param, *topology, resCtx));
    } else if (resCtx.algorithmMode == BROADCAST_ALG_PARALLEL_DIRECT) {
        CHK_PRT_RET(topology == nullptr,
            HCCL_ERROR("[CreateAlgResource] missing parallel direct topology"), HCCL_E_INTERNAL);
        if (param.myRank == param.root) {
            resCtx.threads.resize(2);
            CHK_RET(HcclThreadAcquire(
                comm, engine, 1, HIERARCHY_THREAD_NOTIFY_NUM, &resCtx.threads[1]));
        }
        CHK_RET(CreateParallelDirectResource(comm, engine, param, *topology, resCtx));
    } else {
        std::vector<ChannelHandle> channels;
        CHK_RET(CreateAllRankChannels(comm, engine, param, channels));
        CHK_RET(RegisterBroadcastKernel(comm, param, channels, resCtx));
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

    OpParam param;
    param.inputPtr = buf;
    param.outputPtr = buf;
    param.count = count;
    param.root = root;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_BROADCAST;

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH] = {};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE,
        HCCL_ERROR("[HcclBroadcast] unsupported rankSize[%u]", param.rankSize), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(root >= param.rankSize,
        HCCL_ERROR("[HcclBroadcast] invalid root[%u], rankSize[%u]", root, param.rankSize), HCCL_E_PARA);

    auto dataTypeIt = SIZE_TABLE.find(dataType);
    CHK_PRT_RET(dataTypeIt == SIZE_TABLE.end(),
        HCCL_ERROR("[HcclBroadcast] unsupported dataType[%d]", static_cast<int>(dataType)), HCCL_E_NOT_SUPPORT);
    const uint64_t dataTypeSize = dataTypeIt->second;
    CHK_PRT_RET(count > std::numeric_limits<uint64_t>::max() / dataTypeSize,
        HCCL_ERROR("[HcclBroadcast] data size overflow, count[%llu]",
            static_cast<unsigned long long>(count)), HCCL_E_PARA);

    const uint64_t dataSize = count * dataTypeSize;
    uint32_t algorithmMode = SelectBroadcastAlgorithmMode(dataSize, param.rankSize);
    BroadcastHierarchyTopology hierarchyTopology;
    const BroadcastHierarchyTopology *hierarchyTopologyPtr = nullptr;
    const bool needsLayerTopology = algorithmMode == BROADCAST_ALG_PARALLEL_HIERARCHICAL ||
        algorithmMode == BROADCAST_ALG_PARALLEL_DIRECT;
    if (needsLayerTopology) {
        const HcclResult topologyRet = BuildBroadcastHierarchyTopology(comm, param, hierarchyTopology);
        if (topologyRet == HCCL_SUCCESS) {
            hierarchyTopologyPtr = &hierarchyTopology;
        } else {
            const uint32_t fallbackMode = algorithmMode == BROADCAST_ALG_PARALLEL_DIRECT ?
                BROADCAST_ALG_DIRECT : BROADCAST_ALG_MESH1D_MEM2MEM;
            HCCL_WARNING("[HcclBroadcast] layer topology unavailable, fallback mode[%u], ret[%d]",
                fallbackMode, topologyRet);
            algorithmMode = fallbackMode;
        }
    }
    // algorithmMode 隔离资源；并行模式按 Layer 拆分 Kernel，root 通过双 Thread 并发 Launch。
    const int tagLen = std::snprintf(param.tag, sizeof(param.tag), "hccl_custom_broadcast_ccu_v2_a%u_r%u_n%u",
        algorithmMode, root, param.rankSize);
    CHK_PRT_RET(tagLen < 0 || static_cast<size_t>(tagLen) >= sizeof(param.tag),
        HCCL_ERROR("[HcclBroadcast] failed to build resource tag"), HCCL_E_PARA);

    CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;
    const bool needsSlaveThread = algorithmMode == BROADCAST_ALG_PARALLEL_HIERARCHICAL ||
        (algorithmMode == BROADCAST_ALG_PARALLEL_DIRECT && param.myRank == param.root);
    const uint32_t mainThreadNotifyNum = needsSlaveThread ?
        HIERARCHY_THREAD_NOTIFY_NUM : DEFAULT_MAIN_THREAD_NOTIFY_NUM;
    CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, stream, mainThreadNotifyNum, &param.cpuThread));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, ccuEngine, &ctx, &size) == HCCL_SUCCESS) {
        HCCL_INFO("[HcclBroadcast] reuse CCU engine context");
        param.resCtx = ctx;
        param.ctxSize = size;
    } else {
        AlgResourceCtx resCtxHost;
        CHK_RET(CreateAlgResource(
            comm, ccuEngine, param, algorithmMode, hierarchyTopologyPtr, resCtxHost));

        std::vector<char> seq = resCtxHost.Serialize();
        param.ctxSize = seq.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, ccuEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, ccuEngine, param.tag, seq.data(), param.ctxSize, 0));
    }

    CHK_RET(ops_hccl::ExecOp(param));
    return HCCL_SUCCESS;
}