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
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include <ccu/ccu_launch.h>
#include <ccu/ccu_res.h>
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

namespace final_small512 {
namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 1;

struct ChannelBinding {
    uint32_t peerRank = INVALID_VALUE_RANKID;
    uint32_t layer = 0;
    uint32_t dieId = 0;
    ChannelHandle handle = 0;
};

using ChannelGroups = std::map<std::pair<uint32_t, uint32_t>, std::vector<ChannelBinding>>;

HcclResult ConvertCcuResult(CcuResult result)
{
    switch (result) {
        case CCU_SUCCESS:
            return HCCL_SUCCESS;
        case CCU_E_PARA:
            return HCCL_E_PARA;
        case CCU_E_PTR:
            return HCCL_E_PTR;
        case CCU_E_NOT_SUPPORT:
            return HCCL_E_NOT_SUPPORT;
        case CCU_E_NOT_FOUND:
            return HCCL_E_NOT_FOUND;
        case CCU_E_UNAVAIL:
            return HCCL_E_UNAVAIL;
        default:
            return HCCL_E_INTERNAL;
    }
}

const char *GetContextTag()
{
    switch (RS2_BUILD_FLAVOR) {
        case Rs2BuildFlavor::HIERARCHICAL:
            return "rs2_ccu_hierarchical";
        case Rs2BuildFlavor::HCCL_TOPOLOGY:
            return "rs2_ccu_hccl_topology";
        case Rs2BuildFlavor::GLOBAL_MESH_TREE:
            return "rs2_ccu_v7_global_tree";
        case Rs2BuildFlavor::ADAPTIVE:
            return "rs2_ccu_adaptive";
        case Rs2BuildFlavor::HCCL_TOPOLOGY_HALF_RING:
            return "rs2_ccu_hccl_topology_41_half_ring";
        case Rs2BuildFlavor::HCCL_TOPOLOGY_512K_OPT:
            return "rs2_ccu_hccl_topology_512k_opt";
        default:
            return "rs2_ccu_unknown";
    }
}

bool IsLargeInput(const OpParam &param)
{
    if (param.rankSize == 0) {
        return false;
    }
    return param.count > SMALL_INPUT_BYTES / sizeof(float) / param.rankSize;
}

bool ContainsRank(const std::vector<uint32_t> &ranks, uint32_t rank)
{
    return std::find(ranks.begin(), ranks.end(), rank) != ranks.end();
}

HcclResult GetTopology(HcclComm comm, const OpParam &param, std::vector<uint32_t> &layers,
    std::vector<uint32_t> &localRanks, Rs2TopologyKind &topologyKind)
{
    uint32_t *layerData = nullptr;
    uint32_t layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layerData, &layerCount));
    layers.assign(layerData, layerData + layerCount);

    if (ContainsRank(layers, 0)) {
        uint32_t *rankData = nullptr;
        uint32_t localRankCount = 0;
        CHK_RET(HcclRankGraphGetRanksByLayer(comm, 0, &rankData, &localRankCount));
        localRanks.assign(rankData, rankData + localRankCount);
    } else {
        localRanks.push_back(param.myRank);
    }
    std::sort(localRanks.begin(), localRanks.end());

    topologyKind = Rs2TopologyKind::UNKNOWN;
    if (param.rankSize == 16 && localRanks.size() == 8) {
        topologyKind = Rs2TopologyKind::TWO_BY_EIGHT;
    } else if (param.rankSize == 4 && localRanks.size() == 1) {
        topologyKind = Rs2TopologyKind::FOUR_BY_ONE;
    } else if (param.rankSize == 12 && (localRanks.size() == 8 || localRanks.size() == 4)) {
        topologyKind = Rs2TopologyKind::EIGHT_PLUS_FOUR;
    }
    return HCCL_SUCCESS;
}

HcclResult FindChannelDesc(HcclComm comm, const OpParam &param, uint32_t peerRank,
    const std::vector<uint32_t> &layers, uint32_t preferredLayer, HcclChannelDesc &desc,
    uint32_t &selectedLayer, uint32_t &dieId)
{
    std::vector<uint32_t> candidateLayers;
    if (ContainsRank(layers, preferredLayer)) {
        candidateLayers.push_back(preferredLayer);
    }
    for (uint32_t layer : layers) {
        if (layer != preferredLayer) {
            candidateLayers.push_back(layer);
        }
    }

    for (uint32_t layer : candidateLayers) {
        CommLink *linkList = nullptr;
        uint32_t linkCount = 0;
        if (HcclRankGraphGetLinks(comm, layer, param.myRank, peerRank, &linkList, &linkCount) != HCCL_SUCCESS) {
            continue;
        }
        for (uint32_t linkIdx = 0; linkIdx < linkCount; ++linkIdx) {
            const CommLink &link = linkList[linkIdx];
            if (link.linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                continue;
            }

            CHK_RET(HcclChannelDescInit(&desc, 1));
            desc.remoteRank = peerRank;
            desc.notifyNum = CHANNEL_NOTIFY_NUM;
            desc.channelProtocol = link.linkAttr.linkProtocol;
            desc.localEndpoint = link.srcEndpointDesc;
            desc.remoteEndpoint = link.dstEndpointDesc;

            EndpointAttrDieId endpointDieId = 0;
            CHK_RET(HcclRankGraphGetEndpointInfo(comm, param.myRank, &desc.localEndpoint,
                ENDPOINT_ATTR_DIE_ID, sizeof(endpointDieId), &endpointDieId));
            selectedLayer = layer;
            dieId = endpointDieId;
            return HCCL_SUCCESS;
        }
    }

    HCCL_ERROR("No UBC_CTP link between rank[%u] and rank[%u]", param.myRank, peerRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult AcquireChannels(HcclComm comm, const OpParam &param, const std::vector<uint32_t> &layers,
    const std::vector<uint32_t> &localRanks, std::vector<ChannelBinding> &bindings)
{
    for (uint32_t peerRank = 0; peerRank < param.rankSize; ++peerRank) {
        if (peerRank == param.myRank) {
            continue;
        }

        const bool isLocalPeer = ContainsRank(localRanks, peerRank);
        const uint32_t preferredLayer = isLocalPeer ? 0 : 1;
        HcclChannelDesc desc;
        uint32_t selectedLayer = 0;
        uint32_t dieId = 0;
        CHK_RET(FindChannelDesc(comm, param, peerRank, layers, preferredLayer, desc, selectedLayer, dieId));

        ChannelHandle channel = 0;
        CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, &desc, 1, &channel));
        bindings.push_back(ChannelBinding{peerRank, selectedLayer, dieId, channel});
    }
    return HCCL_SUCCESS;
}

ChannelGroups GroupChannels(const std::vector<ChannelBinding> &bindings,
    const std::vector<uint32_t> &localRanks, int localFilter)
{
    ChannelGroups groups;
    for (const ChannelBinding &binding : bindings) {
        const bool isLocalPeer = ContainsRank(localRanks, binding.peerRank);
        if ((localFilter == 1 && !isLocalPeer) || (localFilter == 0 && isLocalPeer)) {
            continue;
        }
        groups[{binding.layer, binding.dieId}].push_back(binding);
    }
    return groups;
}

HcclResult RegisterKernelSet(CcuInsHandle insHandle, const ChannelGroups &groups, bool includeLocalFirst,
    bool initializeFirst, bool writeOutput, bool assignScratchSlots, std::vector<CcuKernelHandle> &kernelHandles,
    std::vector<uint32_t> &pieceCounts, std::vector<uint32_t> *kernelDieIds = nullptr,
    std::vector<uint32_t> *resultSlots = nullptr, uint32_t *scratchSlotCount = nullptr)
{
    uint32_t groupIdx = 0;
    for (const auto &groupEntry : groups) {
        const std::vector<ChannelBinding> &group = groupEntry.second;
        if (group.empty() || group.size() >= MAX_RANK_SIZE) {
            return HCCL_E_PARA;
        }

        CcuKernelInfo kernelInfo{};
        const int nameResult = std::snprintf(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName),
            "%s", "CcuReduceScatterTreeKernel");
        if (nameResult <= 0 || static_cast<size_t>(nameResult) >= sizeof(kernelInfo.kernelFuncName)) {
            return HCCL_E_INTERNAL;
        }
        kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuReduceScatterTreeKernel);

        auto kernelArg = std::make_shared<CcuKernelArgReduceScatterTree>();
        kernelArg->channelCount = static_cast<uint32_t>(group.size());
        kernelArg->includeLocalInput = includeLocalFirst && groupIdx == 0 ? 1U : 0U;
        kernelArg->initializeOutput = initializeFirst && groupIdx == 0 ? 1U : 0U;
        kernelArg->scratchSlotBase = assignScratchSlots && scratchSlotCount != nullptr ? *scratchSlotCount : 0U;
        kernelArg->writeOutput = writeOutput ? 1U : 0U;
        for (uint32_t channelIdx = 0; channelIdx < kernelArg->channelCount; ++channelIdx) {
            kernelArg->channels[channelIdx] = group[channelIdx].handle;
        }
        kernelInfo.SetKernelArg(kernelArg);

        CcuKernelHandle kernelHandle = 0;
        const void *kernelArgs[] = {kernelInfo.kernelArg};
        const CcuResult registerResult = HcommCcuKernelRegister(insHandle, groupEntry.first.second,
            kernelInfo.kernelFuncName, kernelInfo.kernelFunc, kernelArgs, 1, &kernelHandle);
        if (registerResult != CCU_SUCCESS) {
            return ConvertCcuResult(registerResult);
        }
        kernelHandles.push_back(kernelHandle);
        const uint32_t pieceCount = kernelArg->channelCount + kernelArg->includeLocalInput;
        pieceCounts.push_back(pieceCount);
        if (assignScratchSlots && kernelDieIds != nullptr && resultSlots != nullptr && scratchSlotCount != nullptr) {
            kernelDieIds->push_back(groupEntry.first.second);
            resultSlots->push_back(kernelArg->scratchSlotBase);
            *scratchSlotCount += pieceCount;
        }
        ++groupIdx;
    }
    return HCCL_SUCCESS;
}

bool IsPowerOfTwo(uint32_t value)
{
    return value != 0 && (value & (value - 1)) == 0;
}

HcclResult ConfigureSmallBarrier(const OpParam &param, const std::vector<ChannelBinding> &group,
    bool includeLocalInput, CcuKernelArgReduceScatterSmall &kernelArg)
{
    kernelArg.barrierRoundCount = 0;
    if (!includeLocalInput) {
        return HCCL_SUCCESS;
    }

    std::vector<uint32_t> memberRanks{param.myRank};
    for (const ChannelBinding &binding : group) {
        memberRanks.push_back(binding.peerRank);
    }
    std::sort(memberRanks.begin(), memberRanks.end());
    if (!IsPowerOfTwo(static_cast<uint32_t>(memberRanks.size()))) {
        return HCCL_SUCCESS;
    }

    const auto myIter = std::find(memberRanks.begin(), memberRanks.end(), param.myRank);
    if (myIter == memberRanks.end()) {
        return HCCL_E_INTERNAL;
    }
    const uint32_t myGroupRank = static_cast<uint32_t>(std::distance(memberRanks.begin(), myIter));
    const uint32_t groupSize = static_cast<uint32_t>(memberRanks.size());
    for (uint32_t distance = 1; distance < groupSize; distance <<= 1) {
        const uint32_t sendRank = memberRanks[(myGroupRank + distance) % groupSize];
        const uint32_t recvRank = memberRanks[(myGroupRank + groupSize - distance) % groupSize];
        uint32_t sendChannelIdx = MAX_RANK_SIZE;
        uint32_t recvChannelIdx = MAX_RANK_SIZE;
        for (uint32_t channelIdx = 0; channelIdx < group.size(); ++channelIdx) {
            if (group[channelIdx].peerRank == sendRank) {
                sendChannelIdx = channelIdx;
            }
            if (group[channelIdx].peerRank == recvRank) {
                recvChannelIdx = channelIdx;
            }
        }
        if (sendChannelIdx == MAX_RANK_SIZE || recvChannelIdx == MAX_RANK_SIZE) {
            return HCCL_E_INTERNAL;
        }
        const uint32_t roundIdx = kernelArg.barrierRoundCount++;
        kernelArg.barrierSendChannelIndices[roundIdx] = sendChannelIdx;
        kernelArg.barrierRecvChannelIndices[roundIdx] = recvChannelIdx;
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterSmallKernelSet(CcuInsHandle insHandle, const OpParam &param,
    const ChannelGroups &groups, AlgResourceCtx &resourceCtx)
{
    if (groups.empty() || groups.size() > 2) {
        return HCCL_E_NOT_SUPPORT;
    }

    uint32_t directKernelIdx = 0;
    uint32_t minPieceCount = MAX_RANK_SIZE + 1;
    uint32_t groupIdx = 0;
    for (const auto &groupEntry : groups) {
        const uint32_t pieceCount = static_cast<uint32_t>(groupEntry.second.size()) + (groupIdx == 0 ? 1U : 0U);
        if (pieceCount < minPieceCount) {
            minPieceCount = pieceCount;
            directKernelIdx = groupIdx;
        }
        ++groupIdx;
    }
    resourceCtx.smallDirectKernelIndex = directKernelIdx;

    groupIdx = 0;
    for (const auto &groupEntry : groups) {
        const std::vector<ChannelBinding> &group = groupEntry.second;
        if (group.empty() || group.size() >= MAX_RANK_SIZE) {
            return HCCL_E_PARA;
        }

        CcuKernelInfo kernelInfo{};
        const int nameResult = std::snprintf(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName),
            "%s", "CcuReduceScatterSmallKernel");
        if (nameResult <= 0 || static_cast<size_t>(nameResult) >= sizeof(kernelInfo.kernelFuncName)) {
            return HCCL_E_INTERNAL;
        }
        kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuReduceScatterSmallKernel);

        auto kernelArg = std::make_shared<CcuKernelArgReduceScatterSmall>();
        kernelArg->channelCount = static_cast<uint32_t>(group.size());
        kernelArg->includeLocalInput = groupIdx == 0 ? 1U : 0U;
        kernelArg->scratchSlotBase = resourceCtx.smallScratchSlotCount;
        kernelArg->writeOutput = groupIdx == directKernelIdx ? 1U : 0U;
        for (uint32_t channelIdx = 0; channelIdx < kernelArg->channelCount; ++channelIdx) {
            kernelArg->channels[channelIdx] = group[channelIdx].handle;
        }
        CHK_RET(ConfigureSmallBarrier(param, group, kernelArg->includeLocalInput != 0, *kernelArg));
        kernelInfo.SetKernelArg(kernelArg);

        CcuKernelHandle kernelHandle = 0;
        const void *kernelArgs[] = {kernelInfo.kernelArg};
        const CcuResult registerResult = HcommCcuKernelRegister(insHandle, groupEntry.first.second,
            kernelInfo.kernelFuncName, kernelInfo.kernelFunc, kernelArgs, 1, &kernelHandle);
        if (registerResult != CCU_SUCCESS) {
            return ConvertCcuResult(registerResult);
        }
        resourceCtx.smallKernels.push_back(kernelHandle);
        resourceCtx.smallKernelDieIds.push_back(groupEntry.first.second);
        resourceCtx.smallResultSlots.push_back(kernelArg->scratchSlotBase);
        resourceCtx.smallScratchSlotCount += kernelArg->channelCount + kernelArg->includeLocalInput;
        ++groupIdx;
    }

    if (groups.size() == 2) {
        CcuKernelInfo kernelInfo{};
        const int nameResult = std::snprintf(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName),
            "%s", "CcuReduceScatterSmallFinalizeKernel");
        if (nameResult <= 0 || static_cast<size_t>(nameResult) >= sizeof(kernelInfo.kernelFuncName)) {
            return HCCL_E_INTERNAL;
        }
        kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuReduceScatterSmallFinalizeKernel);
        auto kernelArg = std::make_shared<CcuKernelArgBase>();
        kernelArg->channelCount = 0;
        kernelInfo.SetKernelArg(kernelArg);
        const void *kernelArgs[] = {kernelInfo.kernelArg};
        const CcuResult registerResult = HcommCcuKernelRegister(insHandle, 0, kernelInfo.kernelFuncName,
            kernelInfo.kernelFunc, kernelArgs, 1, &resourceCtx.smallFinalizeKernel);
        if (registerResult != CCU_SUCCESS) {
            return ConvertCcuResult(registerResult);
        }
    }
    return HCCL_SUCCESS;
}

const ChannelBinding *FindBinding(const std::vector<ChannelBinding> &bindings, uint32_t peerRank)
{
    for (const ChannelBinding &binding : bindings) {
        if (binding.peerRank == peerRank) {
            return &binding;
        }
    }
    return nullptr;
}

HcclResult RegisterHalfRingKernel(CcuInsHandle insHandle, const OpParam &param,
    const std::vector<ChannelBinding> &bindings, CcuKernelHandle &kernelHandle, uint32_t &kernelDieId)
{
    if (param.rankSize != 4) {
        return HCCL_E_NOT_SUPPORT;
    }
    const uint32_t nextRank = (param.myRank + 1) % param.rankSize;
    const uint32_t prevRank = (param.myRank + param.rankSize - 1) % param.rankSize;
    const ChannelBinding *nextBinding = FindBinding(bindings, nextRank);
    const ChannelBinding *prevBinding = FindBinding(bindings, prevRank);
    if (nextBinding == nullptr || prevBinding == nullptr || nextBinding->dieId != prevBinding->dieId) {
        HCCL_ERROR("Ring neighbors must be available on one IO Die, rank[%u]", param.myRank);
        return HCCL_E_NOT_SUPPORT;
    }

    CcuKernelInfo kernelInfo{};
    const int nameResult = std::snprintf(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName),
        "%s", "CcuReduceScatterHalfRingKernel");
    if (nameResult <= 0 || static_cast<size_t>(nameResult) >= sizeof(kernelInfo.kernelFuncName)) {
        return HCCL_E_INTERNAL;
    }
    kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuReduceScatterHalfRingKernel);

    auto kernelArg = std::make_shared<CcuKernelArgReduceScatterRing>();
    kernelArg->channelCount = 2;
    kernelArg->channels[0] = nextBinding->handle;
    kernelArg->channels[1] = prevBinding->handle;
    kernelArg->rankId = param.myRank;
    kernelArg->rankSize = param.rankSize;
    for (uint32_t stage = 0; stage < param.rankSize - 1; ++stage) {
        kernelArg->sendBlockIndices[stage] =
            (param.myRank + param.rankSize - stage - 1) % param.rankSize;
    }
    kernelInfo.SetKernelArg(kernelArg);

    const void *kernelArgs[] = {kernelInfo.kernelArg};
    const CcuResult registerResult = HcommCcuKernelRegister(insHandle, nextBinding->dieId,
        kernelInfo.kernelFuncName, kernelInfo.kernelFunc, kernelArgs, 1, &kernelHandle);
    if (registerResult != CCU_SUCCESS) {
        return ConvertCcuResult(registerResult);
    }
    kernelDieId = nextBinding->dieId;
    return HCCL_SUCCESS;
}

HcclResult RegisterKernels(HcclComm comm, const OpParam &param, const std::vector<ChannelBinding> &bindings,
    const std::vector<uint32_t> &localRanks, AlgResourceCtx &resourceCtx)
{
    CcuInsHandle insHandle = 0;
    uint32_t insCount = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insCount));
    if (insCount == 0) {
        return HCCL_E_INTERNAL;
    }

    const CcuResult startResult = HcommCcuKernelRegisterStart(insHandle);
    if (startResult != CCU_SUCCESS) {
        return ConvertCcuResult(startResult);
    }

    const auto topologyKind = static_cast<Rs2TopologyKind>(resourceCtx.topologyKind);
    const bool useHalfRing = (RS2_BUILD_FLAVOR == Rs2BuildFlavor::HCCL_TOPOLOGY_HALF_RING ||
        RS2_BUILD_FLAVOR == Rs2BuildFlavor::HCCL_TOPOLOGY_512K_OPT) &&
        topologyKind == Rs2TopologyKind::FOUR_BY_ONE &&
        IsLargeInput(param);
    const bool useSmallOpt = RS2_BUILD_FLAVOR == Rs2BuildFlavor::HCCL_TOPOLOGY_512K_OPT &&
        !IsLargeInput(param) && topologyKind != Rs2TopologyKind::UNKNOWN;
    if (RS2_BUILD_FLAVOR == Rs2BuildFlavor::HCCL_TOPOLOGY ||
        RS2_BUILD_FLAVOR == Rs2BuildFlavor::GLOBAL_MESH_TREE ||
        RS2_BUILD_FLAVOR == Rs2BuildFlavor::ADAPTIVE ||
        RS2_BUILD_FLAVOR == Rs2BuildFlavor::HCCL_TOPOLOGY_HALF_RING ||
        RS2_BUILD_FLAVOR == Rs2BuildFlavor::HCCL_TOPOLOGY_512K_OPT) {
        if (useSmallOpt) {
            const ChannelGroups globalGroups = GroupChannels(bindings, localRanks, -1);
            CHK_RET(RegisterSmallKernelSet(insHandle, param, globalGroups, resourceCtx));
        } else if (!useHalfRing) {
            const ChannelGroups globalGroups = GroupChannels(bindings, localRanks, -1);
            CHK_RET(RegisterKernelSet(insHandle, globalGroups, true, false, false, true,
                resourceCtx.globalKernels, resourceCtx.globalPieceCounts, &resourceCtx.globalKernelDieIds,
                &resourceCtx.globalResultSlots, &resourceCtx.globalScratchSlotCount));
        } else {
            resourceCtx.halfRingKernels.resize(1);
            resourceCtx.halfRingKernelDieIds.resize(1);
            CHK_RET(RegisterHalfRingKernel(insHandle, param, bindings,
                resourceCtx.halfRingKernels[0], resourceCtx.halfRingKernelDieIds[0]));
        }
    } else {
        const ChannelGroups localGroups = GroupChannels(bindings, localRanks, 1);
        const ChannelGroups remoteGroups = GroupChannels(bindings, localRanks, 0);
        CHK_RET(RegisterKernelSet(insHandle, localGroups, true, true, true, false,
            resourceCtx.localKernels, resourceCtx.localPieceCounts));
        const bool remoteStartsResult = localGroups.empty();
        CHK_RET(RegisterKernelSet(insHandle, remoteGroups, remoteStartsResult, remoteStartsResult, true, false,
            resourceCtx.remoteKernels, resourceCtx.remotePieceCounts));
    }

    const CcuResult endResult = HcommCcuKernelRegisterEnd(insHandle);
    return ConvertCcuResult(endResult);
}

HcclResult CreateResource(HcclComm comm, OpParam &param, AlgResourceCtx &resourceCtx)
{
    void *cclBuffer = nullptr;
    uint64_t cclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBuffer, &cclBufferSize));
    resourceCtx.localBuffer = CommBuffer{cclBuffer, cclBufferSize};
    resourceCtx.threads.push_back(param.cpuThread);

    std::vector<uint32_t> layers;
    std::vector<uint32_t> localRanks;
    Rs2TopologyKind topologyKind = Rs2TopologyKind::UNKNOWN;
    CHK_RET(GetTopology(comm, param, layers, localRanks, topologyKind));
    resourceCtx.topologyKind = static_cast<uint32_t>(topologyKind);
    resourceCtx.localGroupSize = static_cast<uint32_t>(localRanks.size());

    if (param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    ThreadConfig threadConfig{};
    CHK_RET(static_cast<HcclResult>(ThreadConfigInit(&threadConfig, 1)));
    threadConfig.notifyNumPerThread = 1;
    ThreadHandle slaveThread = 0;
    CHK_RET(HcclThreadAcquireWithConfig(comm, CommEngine::COMM_ENGINE_CPU, 1, THREAD_TYPE_TS,
        &threadConfig, &slaveThread));
    resourceCtx.threads.push_back(slaveThread);

    std::vector<ChannelBinding> bindings;
    CHK_RET(AcquireChannels(comm, param, layers, localRanks, bindings));
    return RegisterKernels(comm, param, bindings, localRanks, resourceCtx);
}
} // namespace

HcclResult Run(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream, uint32_t rankSize)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(stream);
    if (dataType != HCCL_DATA_TYPE_FP32 || op != HCCL_REDUCE_SUM) {
        HCCL_ERROR("rs2-ccu supports FP32 SUM only");
        return HCCL_E_NOT_SUPPORT;
    }

    OpParam param;
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    param.rankSize = rankSize;
    if (param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || param.myRank >= param.rankSize) {
        return HCCL_E_PARA;
    }
    if (recvCount > std::numeric_limits<uint64_t>::max() / sizeof(float) / param.rankSize) {
        return HCCL_E_PARA;
    }
    const char *inputClass = IsLargeInput(param) ? "large" : "small";
    const int tagResult = std::snprintf(param.tag, sizeof(param.tag), "%s_%s", GetContextTag(), inputClass);
    if (tagResult <= 0 || static_cast<size_t>(tagResult) >= sizeof(param.tag)) {
        return HCCL_E_INTERNAL;
    }

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH] = {};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    if (HcclEngineCtxGet(comm, param.tag, CommEngine::COMM_ENGINE_CCU, &ctx, &ctxSize) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = ctxSize;
    } else {
        constexpr uint32_t notifyCount = 1;
        CHK_RET(HcclThreadAcquireWithStream(comm, CommEngine::COMM_ENGINE_CCU, stream, notifyCount,
            &param.cpuThread));
        AlgResourceCtx resourceCtx;
        CHK_RET(CreateResource(comm, param, resourceCtx));
        const std::vector<char> sequence = resourceCtx.Serialize();
        param.ctxSize = sequence.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, CommEngine::COMM_ENGINE_CCU, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, CommEngine::COMM_ENGINE_CCU, param.tag, sequence.data(),
            sequence.size(), 0));
    }

    return ops_hccl::ExecOp(param);
}
} // namespace final_small512

namespace final_scratch_lite {
namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 1;

struct ChannelBinding {
    uint32_t peerRank = INVALID_VALUE_RANKID;
    uint32_t layer = 0;
    uint32_t dieId = 0;
    ChannelHandle handle = 0;
};

using ChannelGroups = std::map<std::pair<uint32_t, uint32_t>, std::vector<ChannelBinding>>;

HcclResult ConvertCcuResult(CcuResult result)
{
    switch (result) {
        case CCU_SUCCESS:
            return HCCL_SUCCESS;
        case CCU_E_PARA:
            return HCCL_E_PARA;
        case CCU_E_PTR:
            return HCCL_E_PTR;
        case CCU_E_NOT_SUPPORT:
            return HCCL_E_NOT_SUPPORT;
        case CCU_E_NOT_FOUND:
            return HCCL_E_NOT_FOUND;
        case CCU_E_UNAVAIL:
            return HCCL_E_UNAVAIL;
        default:
            return HCCL_E_INTERNAL;
    }
}

const char *GetContextTag()
{
    switch (RS2_BUILD_FLAVOR) {
        case Rs2BuildFlavor::HIERARCHICAL:
            return "rs2_ccu_hierarchical";
        case Rs2BuildFlavor::HCCL_TOPOLOGY:
            return "rs2_ccu_hccl_topology";
        case Rs2BuildFlavor::GLOBAL_MESH_TREE:
            return "rs2_ccu_v7_global_tree";
        case Rs2BuildFlavor::ADAPTIVE:
            return "rs2_ccu_adaptive";
        case Rs2BuildFlavor::SCRATCH_LITE:
            return "rs2_ccu_hccl_topology_scratch_lite";
        default:
            return "rs2_ccu_unknown";
    }
}

bool ContainsRank(const std::vector<uint32_t> &ranks, uint32_t rank)
{
    return std::find(ranks.begin(), ranks.end(), rank) != ranks.end();
}

HcclResult GetTopology(HcclComm comm, const OpParam &param, std::vector<uint32_t> &layers,
    std::vector<uint32_t> &localRanks, Rs2TopologyKind &topologyKind)
{
    uint32_t *layerData = nullptr;
    uint32_t layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layerData, &layerCount));
    layers.assign(layerData, layerData + layerCount);

    if (ContainsRank(layers, 0)) {
        uint32_t *rankData = nullptr;
        uint32_t localRankCount = 0;
        CHK_RET(HcclRankGraphGetRanksByLayer(comm, 0, &rankData, &localRankCount));
        localRanks.assign(rankData, rankData + localRankCount);
    } else {
        localRanks.push_back(param.myRank);
    }
    std::sort(localRanks.begin(), localRanks.end());

    topologyKind = Rs2TopologyKind::UNKNOWN;
    if (param.rankSize == 16 && localRanks.size() == 8) {
        topologyKind = Rs2TopologyKind::TWO_BY_EIGHT;
    } else if (param.rankSize == 4 && localRanks.size() == 1) {
        topologyKind = Rs2TopologyKind::FOUR_BY_ONE;
    } else if (param.rankSize == 12 && (localRanks.size() == 8 || localRanks.size() == 4)) {
        topologyKind = Rs2TopologyKind::EIGHT_PLUS_FOUR;
    }
    return HCCL_SUCCESS;
}

HcclResult FindChannelDesc(HcclComm comm, const OpParam &param, uint32_t peerRank,
    const std::vector<uint32_t> &layers, uint32_t preferredLayer, HcclChannelDesc &desc,
    uint32_t &selectedLayer, uint32_t &dieId)
{
    std::vector<uint32_t> candidateLayers;
    if (ContainsRank(layers, preferredLayer)) {
        candidateLayers.push_back(preferredLayer);
    }
    for (uint32_t layer : layers) {
        if (layer != preferredLayer) {
            candidateLayers.push_back(layer);
        }
    }

    for (uint32_t layer : candidateLayers) {
        CommLink *linkList = nullptr;
        uint32_t linkCount = 0;
        if (HcclRankGraphGetLinks(comm, layer, param.myRank, peerRank, &linkList, &linkCount) != HCCL_SUCCESS) {
            continue;
        }
        for (uint32_t linkIdx = 0; linkIdx < linkCount; ++linkIdx) {
            const CommLink &link = linkList[linkIdx];
            if (link.linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                continue;
            }

            CHK_RET(HcclChannelDescInit(&desc, 1));
            desc.remoteRank = peerRank;
            desc.notifyNum = CHANNEL_NOTIFY_NUM;
            desc.channelProtocol = link.linkAttr.linkProtocol;
            desc.localEndpoint = link.srcEndpointDesc;
            desc.remoteEndpoint = link.dstEndpointDesc;

            EndpointAttrDieId endpointDieId = 0;
            CHK_RET(HcclRankGraphGetEndpointInfo(comm, param.myRank, &desc.localEndpoint,
                ENDPOINT_ATTR_DIE_ID, sizeof(endpointDieId), &endpointDieId));
            selectedLayer = layer;
            dieId = endpointDieId;
            return HCCL_SUCCESS;
        }
    }

    HCCL_ERROR("No UBC_CTP link between rank[%u] and rank[%u]", param.myRank, peerRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult AcquireChannels(HcclComm comm, const OpParam &param, const std::vector<uint32_t> &layers,
    const std::vector<uint32_t> &localRanks, std::vector<ChannelBinding> &bindings)
{
    for (uint32_t peerRank = 0; peerRank < param.rankSize; ++peerRank) {
        if (peerRank == param.myRank) {
            continue;
        }

        const bool isLocalPeer = ContainsRank(localRanks, peerRank);
        const uint32_t preferredLayer = isLocalPeer ? 0 : 1;
        HcclChannelDesc desc;
        uint32_t selectedLayer = 0;
        uint32_t dieId = 0;
        CHK_RET(FindChannelDesc(comm, param, peerRank, layers, preferredLayer, desc, selectedLayer, dieId));

        ChannelHandle channel = 0;
        CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, &desc, 1, &channel));
        bindings.push_back(ChannelBinding{peerRank, selectedLayer, dieId, channel});
    }
    return HCCL_SUCCESS;
}

ChannelGroups GroupChannels(const std::vector<ChannelBinding> &bindings,
    const std::vector<uint32_t> &localRanks, int localFilter)
{
    ChannelGroups groups;
    for (const ChannelBinding &binding : bindings) {
        const bool isLocalPeer = ContainsRank(localRanks, binding.peerRank);
        if ((localFilter == 1 && !isLocalPeer) || (localFilter == 0 && isLocalPeer)) {
            continue;
        }
        groups[{binding.layer, binding.dieId}].push_back(binding);
    }
    return groups;
}

HcclResult RegisterKernelSet(CcuInsHandle insHandle, const ChannelGroups &groups, bool includeLocalFirst,
    bool initializeFirst, bool writeOutput, bool assignScratchSlots, std::vector<CcuKernelHandle> &kernelHandles,
    std::vector<uint32_t> &pieceCounts, std::vector<uint32_t> *kernelDieIds = nullptr,
    std::vector<uint32_t> *resultSlots = nullptr, uint32_t *scratchSlotCount = nullptr)
{
    uint32_t groupIdx = 0;
    const bool scratchLite = RS2_BUILD_FLAVOR == Rs2BuildFlavor::SCRATCH_LITE && assignScratchSlots;
    for (const auto &groupEntry : groups) {
        const std::vector<ChannelBinding> &group = groupEntry.second;
        if (group.empty() || group.size() >= MAX_RANK_SIZE) {
            return HCCL_E_PARA;
        }

        CcuKernelInfo kernelInfo{};
        const int nameResult = std::snprintf(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName),
            "%s", "CcuReduceScatterTreeKernel");
        if (nameResult <= 0 || static_cast<size_t>(nameResult) >= sizeof(kernelInfo.kernelFuncName)) {
            return HCCL_E_INTERNAL;
        }
        kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuReduceScatterTreeKernel);

        auto kernelArg = std::make_shared<CcuKernelArgReduceScatterTree>();
        kernelArg->channelCount = static_cast<uint32_t>(group.size());
        kernelArg->includeLocalInput = includeLocalFirst && groupIdx == 0 ? 1U : 0U;
        kernelArg->initializeOutput = (scratchLite || initializeFirst) && groupIdx == 0 ? 1U : 0U;
        kernelArg->scratchSlotBase = assignScratchSlots && scratchSlotCount != nullptr ? *scratchSlotCount : 0U;
        kernelArg->writeOutput = scratchLite ? (groupIdx == 0 ? 1U : 0U) : (writeOutput ? 1U : 0U);
        for (uint32_t channelIdx = 0; channelIdx < kernelArg->channelCount; ++channelIdx) {
            kernelArg->channels[channelIdx] = group[channelIdx].handle;
        }
        kernelInfo.SetKernelArg(kernelArg);

        CcuKernelHandle kernelHandle = 0;
        const void *kernelArgs[] = {kernelInfo.kernelArg};
        const CcuResult registerResult = HcommCcuKernelRegister(insHandle, groupEntry.first.second,
            kernelInfo.kernelFuncName, kernelInfo.kernelFunc, kernelArgs, 1, &kernelHandle);
        if (registerResult != CCU_SUCCESS) {
            return ConvertCcuResult(registerResult);
        }
        kernelHandles.push_back(kernelHandle);
        const uint32_t pieceCount = kernelArg->channelCount + kernelArg->includeLocalInput;
        const uint32_t scratchPieceCount = scratchLite ? kernelArg->channelCount : pieceCount;
        pieceCounts.push_back(scratchPieceCount);
        if (assignScratchSlots && kernelDieIds != nullptr && resultSlots != nullptr && scratchSlotCount != nullptr) {
            kernelDieIds->push_back(groupEntry.first.second);
            if (!scratchLite || groupIdx != 0) {
                resultSlots->push_back(kernelArg->scratchSlotBase);
            }
            *scratchSlotCount += scratchPieceCount;
        }
        ++groupIdx;
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterKernels(HcclComm comm, const OpParam &param, const std::vector<ChannelBinding> &bindings,
    const std::vector<uint32_t> &localRanks, AlgResourceCtx &resourceCtx)
{
    CcuInsHandle insHandle = 0;
    uint32_t insCount = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insCount));
    if (insCount == 0) {
        return HCCL_E_INTERNAL;
    }

    const CcuResult startResult = HcommCcuKernelRegisterStart(insHandle);
    if (startResult != CCU_SUCCESS) {
        return ConvertCcuResult(startResult);
    }

    if (RS2_BUILD_FLAVOR == Rs2BuildFlavor::HCCL_TOPOLOGY ||
        RS2_BUILD_FLAVOR == Rs2BuildFlavor::GLOBAL_MESH_TREE ||
        RS2_BUILD_FLAVOR == Rs2BuildFlavor::ADAPTIVE ||
        RS2_BUILD_FLAVOR == Rs2BuildFlavor::SCRATCH_LITE) {
        const ChannelGroups globalGroups = GroupChannels(bindings, localRanks, -1);
        CHK_RET(RegisterKernelSet(insHandle, globalGroups, true, false, false, true,
            resourceCtx.globalKernels, resourceCtx.globalPieceCounts, &resourceCtx.globalKernelDieIds,
            &resourceCtx.globalResultSlots, &resourceCtx.globalScratchSlotCount));
    } else {
        const ChannelGroups localGroups = GroupChannels(bindings, localRanks, 1);
        const ChannelGroups remoteGroups = GroupChannels(bindings, localRanks, 0);
        CHK_RET(RegisterKernelSet(insHandle, localGroups, true, true, true, false,
            resourceCtx.localKernels, resourceCtx.localPieceCounts));
        const bool remoteStartsResult = localGroups.empty();
        CHK_RET(RegisterKernelSet(insHandle, remoteGroups, remoteStartsResult, remoteStartsResult, true, false,
            resourceCtx.remoteKernels, resourceCtx.remotePieceCounts));
    }

    const CcuResult endResult = HcommCcuKernelRegisterEnd(insHandle);
    return ConvertCcuResult(endResult);
}

HcclResult CreateResource(HcclComm comm, OpParam &param, AlgResourceCtx &resourceCtx)
{
    void *cclBuffer = nullptr;
    uint64_t cclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBuffer, &cclBufferSize));
    resourceCtx.localBuffer = CommBuffer{cclBuffer, cclBufferSize};
    const CcuResult tokenResult = HcommCcuGetMemToken(
        reinterpret_cast<uint64_t>(cclBuffer), cclBufferSize, &resourceCtx.localBufferToken);
    if (tokenResult != CCU_SUCCESS) {
        return ConvertCcuResult(tokenResult);
    }
    resourceCtx.threads.push_back(param.cpuThread);

    std::vector<uint32_t> layers;
    std::vector<uint32_t> localRanks;
    Rs2TopologyKind topologyKind = Rs2TopologyKind::UNKNOWN;
    CHK_RET(GetTopology(comm, param, layers, localRanks, topologyKind));
    resourceCtx.topologyKind = static_cast<uint32_t>(topologyKind);
    resourceCtx.localGroupSize = static_cast<uint32_t>(localRanks.size());

    if (param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    ThreadConfig threadConfig{};
    CHK_RET(static_cast<HcclResult>(ThreadConfigInit(&threadConfig, 1)));
    threadConfig.notifyNumPerThread = 1;
    ThreadHandle slaveThread = 0;
    CHK_RET(HcclThreadAcquireWithConfig(comm, CommEngine::COMM_ENGINE_CPU, 1, THREAD_TYPE_TS,
        &threadConfig, &slaveThread));
    resourceCtx.threads.push_back(slaveThread);

    std::vector<ChannelBinding> bindings;
    CHK_RET(AcquireChannels(comm, param, layers, localRanks, bindings));
    return RegisterKernels(comm, param, bindings, localRanks, resourceCtx);
}
} // namespace

HcclResult Run(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream, uint32_t rankSize)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(stream);
    if (dataType != HCCL_DATA_TYPE_FP32 || op != HCCL_REDUCE_SUM) {
        HCCL_ERROR("rs2-ccu supports FP32 SUM only");
        return HCCL_E_NOT_SUPPORT;
    }

    OpParam param;
    const int tagResult = std::snprintf(param.tag, sizeof(param.tag), "%s", GetContextTag());
    if (tagResult <= 0 || static_cast<size_t>(tagResult) >= sizeof(param.tag)) {
        return HCCL_E_INTERNAL;
    }
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    param.rankSize = rankSize;
    if (param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || param.myRank >= param.rankSize) {
        return HCCL_E_PARA;
    }

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH] = {};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    constexpr uint32_t notifyCount = 1;
    CHK_RET(HcclThreadAcquireWithStream(comm, CommEngine::COMM_ENGINE_CCU, stream, notifyCount,
        &param.cpuThread));

    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    if (HcclEngineCtxGet(comm, param.tag, CommEngine::COMM_ENGINE_CCU, &ctx, &ctxSize) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = ctxSize;
    } else {
        AlgResourceCtx resourceCtx;
        CHK_RET(CreateResource(comm, param, resourceCtx));
        const std::vector<char> sequence = resourceCtx.Serialize();
        param.ctxSize = sequence.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, CommEngine::COMM_ENGINE_CCU, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, CommEngine::COMM_ENGINE_CCU, param.tag, sequence.data(),
            sequence.size(), 0));
    }

    return ops_hccl::ExecOp(param);
}
} // namespace final_scratch_lite

namespace final_epoch_cache {
namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 1;

struct ChannelBinding {
    uint32_t peerRank = INVALID_VALUE_RANKID;
    uint32_t layer = 0;
    uint32_t dieId = 0;
    ChannelHandle handle = 0;
};

using ChannelGroups = std::map<std::pair<uint32_t, uint32_t>, std::vector<ChannelBinding>>;

HcclResult ConvertCcuResult(CcuResult result)
{
    switch (result) {
        case CCU_SUCCESS:
            return HCCL_SUCCESS;
        case CCU_E_PARA:
            return HCCL_E_PARA;
        case CCU_E_PTR:
            return HCCL_E_PTR;
        case CCU_E_NOT_SUPPORT:
            return HCCL_E_NOT_SUPPORT;
        case CCU_E_NOT_FOUND:
            return HCCL_E_NOT_FOUND;
        case CCU_E_UNAVAIL:
            return HCCL_E_UNAVAIL;
        default:
            return HCCL_E_INTERNAL;
    }
}

const char *GetContextTag()
{
    switch (RS2_BUILD_FLAVOR) {
        case Rs2BuildFlavor::HIERARCHICAL:
            return "rs2_ccu_hierarchical";
        case Rs2BuildFlavor::HCCL_TOPOLOGY:
            return "rs2_ccu_hccl_topology";
        case Rs2BuildFlavor::GLOBAL_MESH_TREE:
            return "rs2_ccu_v7_global_tree";
        case Rs2BuildFlavor::ADAPTIVE:
            return "rs2_ccu_adaptive";
        case Rs2BuildFlavor::SCRATCH_LITE:
            return "rs2_ccu_hccl_topology_scratch_lite";
        case Rs2BuildFlavor::EPOCH_CACHE:
            return "rs2_ccu_hccl_topology_epoch_cache";
        default:
            return "rs2_ccu_unknown";
    }
}

bool ContainsRank(const std::vector<uint32_t> &ranks, uint32_t rank)
{
    return std::find(ranks.begin(), ranks.end(), rank) != ranks.end();
}

HcclResult GetTopology(HcclComm comm, const OpParam &param, std::vector<uint32_t> &layers,
    std::vector<uint32_t> &localRanks, Rs2TopologyKind &topologyKind)
{
    uint32_t *layerData = nullptr;
    uint32_t layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layerData, &layerCount));
    layers.assign(layerData, layerData + layerCount);

    if (ContainsRank(layers, 0)) {
        uint32_t *rankData = nullptr;
        uint32_t localRankCount = 0;
        CHK_RET(HcclRankGraphGetRanksByLayer(comm, 0, &rankData, &localRankCount));
        localRanks.assign(rankData, rankData + localRankCount);
    } else {
        localRanks.push_back(param.myRank);
    }
    std::sort(localRanks.begin(), localRanks.end());

    topologyKind = Rs2TopologyKind::UNKNOWN;
    if (param.rankSize == 16 && localRanks.size() == 8) {
        topologyKind = Rs2TopologyKind::TWO_BY_EIGHT;
    } else if (param.rankSize == 4 && localRanks.size() == 1) {
        topologyKind = Rs2TopologyKind::FOUR_BY_ONE;
    } else if (param.rankSize == 12 && (localRanks.size() == 8 || localRanks.size() == 4)) {
        topologyKind = Rs2TopologyKind::EIGHT_PLUS_FOUR;
    }
    return HCCL_SUCCESS;
}

HcclResult FindChannelDesc(HcclComm comm, const OpParam &param, uint32_t peerRank,
    const std::vector<uint32_t> &layers, uint32_t preferredLayer, HcclChannelDesc &desc,
    uint32_t &selectedLayer, uint32_t &dieId)
{
    std::vector<uint32_t> candidateLayers;
    if (ContainsRank(layers, preferredLayer)) {
        candidateLayers.push_back(preferredLayer);
    }
    for (uint32_t layer : layers) {
        if (layer != preferredLayer) {
            candidateLayers.push_back(layer);
        }
    }

    for (uint32_t layer : candidateLayers) {
        CommLink *linkList = nullptr;
        uint32_t linkCount = 0;
        if (HcclRankGraphGetLinks(comm, layer, param.myRank, peerRank, &linkList, &linkCount) != HCCL_SUCCESS) {
            continue;
        }
        for (uint32_t linkIdx = 0; linkIdx < linkCount; ++linkIdx) {
            const CommLink &link = linkList[linkIdx];
            if (link.linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                continue;
            }

            CHK_RET(HcclChannelDescInit(&desc, 1));
            desc.remoteRank = peerRank;
            desc.notifyNum = CHANNEL_NOTIFY_NUM;
            desc.channelProtocol = link.linkAttr.linkProtocol;
            desc.localEndpoint = link.srcEndpointDesc;
            desc.remoteEndpoint = link.dstEndpointDesc;

            EndpointAttrDieId endpointDieId = 0;
            CHK_RET(HcclRankGraphGetEndpointInfo(comm, param.myRank, &desc.localEndpoint,
                ENDPOINT_ATTR_DIE_ID, sizeof(endpointDieId), &endpointDieId));
            selectedLayer = layer;
            dieId = endpointDieId;
            return HCCL_SUCCESS;
        }
    }

    HCCL_ERROR("No UBC_CTP link between rank[%u] and rank[%u]", param.myRank, peerRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult AcquireChannels(HcclComm comm, const OpParam &param, const std::vector<uint32_t> &layers,
    const std::vector<uint32_t> &localRanks, std::vector<ChannelBinding> &bindings)
{
    for (uint32_t peerRank = 0; peerRank < param.rankSize; ++peerRank) {
        if (peerRank == param.myRank) {
            continue;
        }

        const bool isLocalPeer = ContainsRank(localRanks, peerRank);
        const uint32_t preferredLayer = isLocalPeer ? 0 : 1;
        HcclChannelDesc desc;
        uint32_t selectedLayer = 0;
        uint32_t dieId = 0;
        CHK_RET(FindChannelDesc(comm, param, peerRank, layers, preferredLayer, desc, selectedLayer, dieId));

        ChannelHandle channel = 0;
        CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, &desc, 1, &channel));
        bindings.push_back(ChannelBinding{peerRank, selectedLayer, dieId, channel});
    }
    return HCCL_SUCCESS;
}

ChannelGroups GroupChannels(const std::vector<ChannelBinding> &bindings,
    const std::vector<uint32_t> &localRanks, int localFilter)
{
    ChannelGroups groups;
    for (const ChannelBinding &binding : bindings) {
        const bool isLocalPeer = ContainsRank(localRanks, binding.peerRank);
        if ((localFilter == 1 && !isLocalPeer) || (localFilter == 0 && isLocalPeer)) {
            continue;
        }
        groups[{binding.layer, binding.dieId}].push_back(binding);
    }
    return groups;
}

HcclResult RegisterKernelSet(CcuInsHandle insHandle, const ChannelGroups &groups, bool includeLocalFirst,
    bool initializeFirst, bool writeOutput, bool assignScratchSlots, std::vector<CcuKernelHandle> &kernelHandles,
    std::vector<uint32_t> &pieceCounts, std::vector<uint32_t> *kernelDieIds = nullptr,
    std::vector<uint32_t> *resultSlots = nullptr, uint32_t *scratchSlotCount = nullptr)
{
    uint32_t groupIdx = 0;
    const bool scratchLite = (RS2_BUILD_FLAVOR == Rs2BuildFlavor::SCRATCH_LITE ||
        RS2_BUILD_FLAVOR == Rs2BuildFlavor::EPOCH_CACHE) && assignScratchSlots;
    for (const auto &groupEntry : groups) {
        const std::vector<ChannelBinding> &group = groupEntry.second;
        if (group.empty() || group.size() >= MAX_RANK_SIZE) {
            return HCCL_E_PARA;
        }

        CcuKernelInfo kernelInfo{};
        const int nameResult = std::snprintf(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName),
            "%s", "CcuReduceScatterTreeKernel");
        if (nameResult <= 0 || static_cast<size_t>(nameResult) >= sizeof(kernelInfo.kernelFuncName)) {
            return HCCL_E_INTERNAL;
        }
        kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuReduceScatterTreeKernel);

        auto kernelArg = std::make_shared<CcuKernelArgReduceScatterTree>();
        kernelArg->channelCount = static_cast<uint32_t>(group.size());
        kernelArg->includeLocalInput = includeLocalFirst && groupIdx == 0 ? 1U : 0U;
        kernelArg->initializeOutput = (scratchLite || initializeFirst) && groupIdx == 0 ? 1U : 0U;
        kernelArg->scratchSlotBase = assignScratchSlots && scratchSlotCount != nullptr ? *scratchSlotCount : 0U;
        kernelArg->writeOutput = scratchLite ? (groupIdx == 0 ? 1U : 0U) : (writeOutput ? 1U : 0U);
        for (uint32_t channelIdx = 0; channelIdx < kernelArg->channelCount; ++channelIdx) {
            kernelArg->channels[channelIdx] = group[channelIdx].handle;
        }
        kernelInfo.SetKernelArg(kernelArg);

        CcuKernelHandle kernelHandle = 0;
        const void *kernelArgs[] = {kernelInfo.kernelArg};
        const CcuResult registerResult = HcommCcuKernelRegister(insHandle, groupEntry.first.second,
            kernelInfo.kernelFuncName, kernelInfo.kernelFunc, kernelArgs, 1, &kernelHandle);
        if (registerResult != CCU_SUCCESS) {
            return ConvertCcuResult(registerResult);
        }
        kernelHandles.push_back(kernelHandle);
        const uint32_t pieceCount = kernelArg->channelCount + kernelArg->includeLocalInput;
        const uint32_t scratchPieceCount = scratchLite ? kernelArg->channelCount : pieceCount;
        pieceCounts.push_back(scratchPieceCount);
        if (assignScratchSlots && kernelDieIds != nullptr && resultSlots != nullptr && scratchSlotCount != nullptr) {
            kernelDieIds->push_back(groupEntry.first.second);
            if (!scratchLite || groupIdx != 0) {
                resultSlots->push_back(kernelArg->scratchSlotBase);
            }
            *scratchSlotCount += scratchPieceCount;
        }
        ++groupIdx;
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterKernels(HcclComm comm, const OpParam &param, const std::vector<ChannelBinding> &bindings,
    const std::vector<uint32_t> &localRanks, AlgResourceCtx &resourceCtx)
{
    CcuInsHandle insHandle = 0;
    uint32_t insCount = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insCount));
    if (insCount == 0) {
        return HCCL_E_INTERNAL;
    }

    const CcuResult startResult = HcommCcuKernelRegisterStart(insHandle);
    if (startResult != CCU_SUCCESS) {
        return ConvertCcuResult(startResult);
    }

    if (RS2_BUILD_FLAVOR == Rs2BuildFlavor::HCCL_TOPOLOGY ||
        RS2_BUILD_FLAVOR == Rs2BuildFlavor::GLOBAL_MESH_TREE ||
        RS2_BUILD_FLAVOR == Rs2BuildFlavor::ADAPTIVE ||
        RS2_BUILD_FLAVOR == Rs2BuildFlavor::SCRATCH_LITE ||
        RS2_BUILD_FLAVOR == Rs2BuildFlavor::EPOCH_CACHE) {
        const ChannelGroups globalGroups = GroupChannels(bindings, localRanks, -1);
        CHK_RET(RegisterKernelSet(insHandle, globalGroups, true, false, false, true,
            resourceCtx.globalKernels, resourceCtx.globalPieceCounts, &resourceCtx.globalKernelDieIds,
            &resourceCtx.globalResultSlots, &resourceCtx.globalScratchSlotCount));
    } else {
        const ChannelGroups localGroups = GroupChannels(bindings, localRanks, 1);
        const ChannelGroups remoteGroups = GroupChannels(bindings, localRanks, 0);
        CHK_RET(RegisterKernelSet(insHandle, localGroups, true, true, true, false,
            resourceCtx.localKernels, resourceCtx.localPieceCounts));
        const bool remoteStartsResult = localGroups.empty();
        CHK_RET(RegisterKernelSet(insHandle, remoteGroups, remoteStartsResult, remoteStartsResult, true, false,
            resourceCtx.remoteKernels, resourceCtx.remotePieceCounts));
    }

    const CcuResult endResult = HcommCcuKernelRegisterEnd(insHandle);
    return ConvertCcuResult(endResult);
}

HcclResult CreateResource(HcclComm comm, OpParam &param, AlgResourceCtx &resourceCtx)
{
    void *cclBuffer = nullptr;
    uint64_t cclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBuffer, &cclBufferSize));
    resourceCtx.localBuffer = CommBuffer{cclBuffer, cclBufferSize};
    const CcuResult tokenResult = HcommCcuGetMemToken(
        reinterpret_cast<uint64_t>(cclBuffer), cclBufferSize, &resourceCtx.localBufferToken);
    if (tokenResult != CCU_SUCCESS) {
        return ConvertCcuResult(tokenResult);
    }
    if (RS2_BUILD_FLAVOR == Rs2BuildFlavor::EPOCH_CACHE && param.root != 0) {
        const uint64_t recvBytes = param.count * sizeof(float);
        const uint64_t inputBytes = recvBytes * param.rankSize;
        const CcuResult inputTokenResult = HcommCcuGetMemToken(
            reinterpret_cast<uint64_t>(param.inputPtr), inputBytes, &resourceCtx.inputToken);
        if (inputTokenResult != CCU_SUCCESS) {
            return ConvertCcuResult(inputTokenResult);
        }
        const CcuResult outputTokenResult = HcommCcuGetMemToken(
            reinterpret_cast<uint64_t>(param.outputPtr), recvBytes, &resourceCtx.outputToken);
        if (outputTokenResult != CCU_SUCCESS) {
            return ConvertCcuResult(outputTokenResult);
        }
    }
    resourceCtx.threads.push_back(param.cpuThread);

    std::vector<uint32_t> layers;
    std::vector<uint32_t> localRanks;
    Rs2TopologyKind topologyKind = Rs2TopologyKind::UNKNOWN;
    CHK_RET(GetTopology(comm, param, layers, localRanks, topologyKind));
    resourceCtx.topologyKind = static_cast<uint32_t>(topologyKind);
    resourceCtx.localGroupSize = static_cast<uint32_t>(localRanks.size());

    if (param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    ThreadConfig threadConfig{};
    CHK_RET(static_cast<HcclResult>(ThreadConfigInit(&threadConfig, 1)));
    threadConfig.notifyNumPerThread = 1;
    ThreadHandle slaveThread = 0;
    CHK_RET(HcclThreadAcquireWithConfig(comm, CommEngine::COMM_ENGINE_CPU, 1, THREAD_TYPE_TS,
        &threadConfig, &slaveThread));
    resourceCtx.threads.push_back(slaveThread);

    std::vector<ChannelBinding> bindings;
    CHK_RET(AcquireChannels(comm, param, layers, localRanks, bindings));
    return RegisterKernels(comm, param, bindings, localRanks, resourceCtx);
}
} // namespace

HcclResult Run(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream, uint32_t rankSize)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(stream);
    if (dataType != HCCL_DATA_TYPE_FP32 || op != HCCL_REDUCE_SUM) {
        HCCL_ERROR("rs2-ccu supports FP32 SUM only");
        return HCCL_E_NOT_SUPPORT;
    }

    OpParam param;
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    param.rankSize = rankSize;
    if (param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || param.myRank >= param.rankSize) {
        return HCCL_E_PARA;
    }
    constexpr uint64_t smallInputBytes = 1ULL * 1024 * 1024;
    const bool smallInput = recvCount <= smallInputBytes / sizeof(float) / param.rankSize;
    const int tagResult = smallInput ? std::snprintf(param.tag, sizeof(param.tag), "%s_small_%llx_%llx",
        GetContextTag(), static_cast<unsigned long long>(reinterpret_cast<uint64_t>(sendBuf)),
        static_cast<unsigned long long>(reinterpret_cast<uint64_t>(recvBuf))) :
        std::snprintf(param.tag, sizeof(param.tag), "%s_large", GetContextTag());
    if (tagResult <= 0 || static_cast<size_t>(tagResult) >= sizeof(param.tag)) {
        return HCCL_E_INTERNAL;
    }

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH] = {};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    constexpr uint32_t notifyCount = 1;
    CHK_RET(HcclThreadAcquireWithStream(comm, CommEngine::COMM_ENGINE_CCU, stream, notifyCount,
        &param.cpuThread));

    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    if (HcclEngineCtxGet(comm, param.tag, CommEngine::COMM_ENGINE_CCU, &ctx, &ctxSize) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = ctxSize;
        param.root = smallInput ? 1U : 0U;
    } else {
        param.root = smallInput ? 1U : 0U;
        AlgResourceCtx resourceCtx;
        CHK_RET(CreateResource(comm, param, resourceCtx));
        const std::vector<char> sequence = resourceCtx.Serialize();
        param.ctxSize = sequence.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, CommEngine::COMM_ENGINE_CCU, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, CommEngine::COMM_ENGINE_CCU, param.tag, sequence.data(),
            sequence.size(), 0));
    }

    return ops_hccl::ExecOp(param);
}
} // namespace final_epoch_cache

namespace final_direct_pipeline {
namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 1;
constexpr uint64_t SMALL_INPUT_BYTES = 1ULL * 1024 * 1024;

struct ChannelBinding {
    uint32_t peerRank = INVALID_VALUE_RANKID;
    uint32_t layer = 0;
    uint32_t dieId = 0;
    ChannelHandle handle = 0;
};

using ChannelGroups = std::map<std::pair<uint32_t, uint32_t>, std::vector<ChannelBinding>>;

HcclResult ConvertCcuResult(CcuResult result)
{
    switch (result) {
        case CCU_SUCCESS:
            return HCCL_SUCCESS;
        case CCU_E_PARA:
            return HCCL_E_PARA;
        case CCU_E_PTR:
            return HCCL_E_PTR;
        case CCU_E_NOT_SUPPORT:
            return HCCL_E_NOT_SUPPORT;
        case CCU_E_NOT_FOUND:
            return HCCL_E_NOT_FOUND;
        case CCU_E_UNAVAIL:
            return HCCL_E_UNAVAIL;
        default:
            return HCCL_E_INTERNAL;
    }
}

const char *GetContextTag()
{
    switch (RS2_BUILD_FLAVOR) {
        case Rs2BuildFlavor::HIERARCHICAL:
            return "rs2_ccu_hierarchical";
        case Rs2BuildFlavor::HCCL_TOPOLOGY:
            return "rs2_ccu_hccl_topology";
        case Rs2BuildFlavor::GLOBAL_MESH_TREE:
            return "rs2_ccu_v7_global_tree";
        case Rs2BuildFlavor::ADAPTIVE:
            return "rs2_ccu_adaptive";
        case Rs2BuildFlavor::SCRATCH_LITE:
            return "rs2_ccu_hccl_topology_scratch_lite";
        case Rs2BuildFlavor::EPOCH_CACHE:
            return "rs2_ccu_hccl_topology_epoch_cache";
        case Rs2BuildFlavor::DIRECT_PIPELINE:
            return "rs2_ccu_hccl_topology_direct_pipeline";
        default:
            return "rs2_ccu_unknown";
    }
}

bool ContainsRank(const std::vector<uint32_t> &ranks, uint32_t rank)
{
    return std::find(ranks.begin(), ranks.end(), rank) != ranks.end();
}

bool UseFirstLayerPairReady(const OpParam &param, Rs2TopologyKind topologyKind)
{
    if (topologyKind != Rs2TopologyKind::TWO_BY_EIGHT &&
        topologyKind != Rs2TopologyKind::EIGHT_PLUS_FOUR) {
        return false;
    }
    if (param.rankSize == 0 ||
        param.count > std::numeric_limits<uint64_t>::max() / sizeof(float) / param.rankSize) {
        return false;
    }
    return param.count * sizeof(float) * param.rankSize > SMALL_INPUT_BYTES;
}

HcclResult GetTopology(HcclComm comm, const OpParam &param, std::vector<uint32_t> &layers,
    std::vector<uint32_t> &localRanks, Rs2TopologyKind &topologyKind)
{
    uint32_t *layerData = nullptr;
    uint32_t layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layerData, &layerCount));
    layers.assign(layerData, layerData + layerCount);

    if (ContainsRank(layers, 0)) {
        uint32_t *rankData = nullptr;
        uint32_t localRankCount = 0;
        CHK_RET(HcclRankGraphGetRanksByLayer(comm, 0, &rankData, &localRankCount));
        localRanks.assign(rankData, rankData + localRankCount);
    } else {
        localRanks.push_back(param.myRank);
    }
    std::sort(localRanks.begin(), localRanks.end());

    topologyKind = Rs2TopologyKind::UNKNOWN;
    if (param.rankSize == 16 && localRanks.size() == 8) {
        topologyKind = Rs2TopologyKind::TWO_BY_EIGHT;
    } else if (param.rankSize == 4 && localRanks.size() == 1) {
        topologyKind = Rs2TopologyKind::FOUR_BY_ONE;
    } else if (param.rankSize == 12 && (localRanks.size() == 8 || localRanks.size() == 4)) {
        topologyKind = Rs2TopologyKind::EIGHT_PLUS_FOUR;
    }
    return HCCL_SUCCESS;
}

HcclResult FindChannelDesc(HcclComm comm, const OpParam &param, uint32_t peerRank,
    const std::vector<uint32_t> &layers, uint32_t preferredLayer, HcclChannelDesc &desc,
    uint32_t &selectedLayer, uint32_t &dieId)
{
    std::vector<uint32_t> candidateLayers;
    if (ContainsRank(layers, preferredLayer)) {
        candidateLayers.push_back(preferredLayer);
    }
    for (uint32_t layer : layers) {
        if (layer != preferredLayer) {
            candidateLayers.push_back(layer);
        }
    }

    for (uint32_t layer : candidateLayers) {
        CommLink *linkList = nullptr;
        uint32_t linkCount = 0;
        if (HcclRankGraphGetLinks(comm, layer, param.myRank, peerRank, &linkList, &linkCount) != HCCL_SUCCESS) {
            continue;
        }
        for (uint32_t linkIdx = 0; linkIdx < linkCount; ++linkIdx) {
            const CommLink &link = linkList[linkIdx];
            if (link.linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                continue;
            }

            CHK_RET(HcclChannelDescInit(&desc, 1));
            desc.remoteRank = peerRank;
            desc.notifyNum = CHANNEL_NOTIFY_NUM;
            desc.channelProtocol = link.linkAttr.linkProtocol;
            desc.localEndpoint = link.srcEndpointDesc;
            desc.remoteEndpoint = link.dstEndpointDesc;

            EndpointAttrDieId endpointDieId = 0;
            CHK_RET(HcclRankGraphGetEndpointInfo(comm, param.myRank, &desc.localEndpoint,
                ENDPOINT_ATTR_DIE_ID, sizeof(endpointDieId), &endpointDieId));
            selectedLayer = layer;
            dieId = endpointDieId;
            return HCCL_SUCCESS;
        }
    }

    HCCL_ERROR("No UBC_CTP link between rank[%u] and rank[%u]", param.myRank, peerRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult AcquireChannels(HcclComm comm, const OpParam &param, const std::vector<uint32_t> &layers,
    const std::vector<uint32_t> &localRanks, std::vector<ChannelBinding> &bindings)
{
    for (uint32_t peerRank = 0; peerRank < param.rankSize; ++peerRank) {
        if (peerRank == param.myRank) {
            continue;
        }

        const bool isLocalPeer = ContainsRank(localRanks, peerRank);
        const uint32_t preferredLayer = isLocalPeer ? 0 : 1;
        HcclChannelDesc desc;
        uint32_t selectedLayer = 0;
        uint32_t dieId = 0;
        CHK_RET(FindChannelDesc(comm, param, peerRank, layers, preferredLayer, desc, selectedLayer, dieId));

        ChannelHandle channel = 0;
        CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, &desc, 1, &channel));
        bindings.push_back(ChannelBinding{peerRank, selectedLayer, dieId, channel});
    }
    return HCCL_SUCCESS;
}

ChannelGroups GroupChannels(const std::vector<ChannelBinding> &bindings,
    const std::vector<uint32_t> &localRanks, int localFilter)
{
    ChannelGroups groups;
    for (const ChannelBinding &binding : bindings) {
        const bool isLocalPeer = ContainsRank(localRanks, binding.peerRank);
        if ((localFilter == 1 && !isLocalPeer) || (localFilter == 0 && isLocalPeer)) {
            continue;
        }
        groups[{binding.layer, binding.dieId}].push_back(binding);
    }
    return groups;
}

HcclResult RegisterKernelSet(CcuInsHandle insHandle, const ChannelGroups &groups, bool includeLocalFirst,
    bool initializeFirst, bool writeOutput, bool assignScratchSlots, std::vector<CcuKernelHandle> &kernelHandles,
    std::vector<uint32_t> &pieceCounts, std::vector<uint32_t> *kernelDieIds = nullptr,
    std::vector<uint32_t> *resultSlots = nullptr, uint32_t *scratchSlotCount = nullptr,
    bool useFirstLayerPairReady = false)
{
    uint32_t groupIdx = 0;
    const bool scratchLite = (RS2_BUILD_FLAVOR == Rs2BuildFlavor::SCRATCH_LITE ||
        RS2_BUILD_FLAVOR == Rs2BuildFlavor::EPOCH_CACHE ||
        RS2_BUILD_FLAVOR == Rs2BuildFlavor::DIRECT_PIPELINE) && assignScratchSlots;
    for (const auto &groupEntry : groups) {
        const std::vector<ChannelBinding> &group = groupEntry.second;
        if (group.empty() || group.size() >= MAX_RANK_SIZE) {
            return HCCL_E_PARA;
        }

        CcuKernelInfo kernelInfo{};
        const int nameResult = std::snprintf(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName),
            "%s", "CcuReduceScatterTreeKernel");
        if (nameResult <= 0 || static_cast<size_t>(nameResult) >= sizeof(kernelInfo.kernelFuncName)) {
            return HCCL_E_INTERNAL;
        }
        kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuReduceScatterTreeKernel);

        auto kernelArg = std::make_shared<CcuKernelArgReduceScatterTree>();
        kernelArg->channelCount = static_cast<uint32_t>(group.size());
        kernelArg->includeLocalInput = includeLocalFirst && groupIdx == 0 ? 1U : 0U;
        kernelArg->initializeOutput = (scratchLite || initializeFirst) && groupIdx == 0 ? 1U : 0U;
        kernelArg->scratchSlotBase = assignScratchSlots && scratchSlotCount != nullptr ? *scratchSlotCount : 0U;
        kernelArg->writeOutput = scratchLite ? (groupIdx == 0 ? 1U : 0U) : (writeOutput ? 1U : 0U);
        kernelArg->useFirstLayerPairReady = useFirstLayerPairReady ? 1U : 0U;
        for (uint32_t channelIdx = 0; channelIdx < kernelArg->channelCount; ++channelIdx) {
            kernelArg->channels[channelIdx] = group[channelIdx].handle;
        }
        kernelInfo.SetKernelArg(kernelArg);

        CcuKernelHandle kernelHandle = 0;
        const void *kernelArgs[] = {kernelInfo.kernelArg};
        const CcuResult registerResult = HcommCcuKernelRegister(insHandle, groupEntry.first.second,
            kernelInfo.kernelFuncName, kernelInfo.kernelFunc, kernelArgs, 1, &kernelHandle);
        if (registerResult != CCU_SUCCESS) {
            return ConvertCcuResult(registerResult);
        }
        kernelHandles.push_back(kernelHandle);
        const uint32_t pieceCount = kernelArg->channelCount + kernelArg->includeLocalInput;
        const uint32_t scratchPieceCount = scratchLite ? kernelArg->channelCount : pieceCount;
        pieceCounts.push_back(scratchPieceCount);
        if (assignScratchSlots && kernelDieIds != nullptr && resultSlots != nullptr && scratchSlotCount != nullptr) {
            kernelDieIds->push_back(groupEntry.first.second);
            if (!scratchLite || groupIdx != 0) {
                resultSlots->push_back(kernelArg->scratchSlotBase);
            }
            *scratchSlotCount += scratchPieceCount;
        }
        ++groupIdx;
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterKernels(HcclComm comm, const OpParam &param, const std::vector<ChannelBinding> &bindings,
    const std::vector<uint32_t> &localRanks, AlgResourceCtx &resourceCtx)
{
    CcuInsHandle insHandle = 0;
    uint32_t insCount = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insCount));
    if (insCount == 0) {
        return HCCL_E_INTERNAL;
    }

    const CcuResult startResult = HcommCcuKernelRegisterStart(insHandle);
    if (startResult != CCU_SUCCESS) {
        return ConvertCcuResult(startResult);
    }

    if (RS2_BUILD_FLAVOR == Rs2BuildFlavor::HCCL_TOPOLOGY ||
        RS2_BUILD_FLAVOR == Rs2BuildFlavor::GLOBAL_MESH_TREE ||
        RS2_BUILD_FLAVOR == Rs2BuildFlavor::ADAPTIVE ||
        RS2_BUILD_FLAVOR == Rs2BuildFlavor::SCRATCH_LITE ||
        RS2_BUILD_FLAVOR == Rs2BuildFlavor::EPOCH_CACHE ||
        RS2_BUILD_FLAVOR == Rs2BuildFlavor::DIRECT_PIPELINE) {
        const ChannelGroups globalGroups = GroupChannels(bindings, localRanks, -1);
        const bool useFirstLayerPairReady = UseFirstLayerPairReady(
            param, static_cast<Rs2TopologyKind>(resourceCtx.topologyKind));
        CHK_RET(RegisterKernelSet(insHandle, globalGroups, true, false, false, true,
            resourceCtx.globalKernels, resourceCtx.globalPieceCounts, &resourceCtx.globalKernelDieIds,
            &resourceCtx.globalResultSlots, &resourceCtx.globalScratchSlotCount, useFirstLayerPairReady));
    } else {
        const ChannelGroups localGroups = GroupChannels(bindings, localRanks, 1);
        const ChannelGroups remoteGroups = GroupChannels(bindings, localRanks, 0);
        CHK_RET(RegisterKernelSet(insHandle, localGroups, true, true, true, false,
            resourceCtx.localKernels, resourceCtx.localPieceCounts));
        const bool remoteStartsResult = localGroups.empty();
        CHK_RET(RegisterKernelSet(insHandle, remoteGroups, remoteStartsResult, remoteStartsResult, true, false,
            resourceCtx.remoteKernels, resourceCtx.remotePieceCounts));
    }

    const CcuResult endResult = HcommCcuKernelRegisterEnd(insHandle);
    return ConvertCcuResult(endResult);
}

HcclResult CreateResource(HcclComm comm, OpParam &param, AlgResourceCtx &resourceCtx)
{
    void *cclBuffer = nullptr;
    uint64_t cclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBuffer, &cclBufferSize));
    resourceCtx.localBuffer = CommBuffer{cclBuffer, cclBufferSize};
    const CcuResult tokenResult = HcommCcuGetMemToken(
        reinterpret_cast<uint64_t>(cclBuffer), cclBufferSize, &resourceCtx.localBufferToken);
    if (tokenResult != CCU_SUCCESS) {
        return ConvertCcuResult(tokenResult);
    }
    if ((RS2_BUILD_FLAVOR == Rs2BuildFlavor::EPOCH_CACHE ||
        RS2_BUILD_FLAVOR == Rs2BuildFlavor::DIRECT_PIPELINE) && param.root != 0) {
        const uint64_t recvBytes = param.count * sizeof(float);
        const uint64_t inputBytes = recvBytes * param.rankSize;
        const CcuResult inputTokenResult = HcommCcuGetMemToken(
            reinterpret_cast<uint64_t>(param.inputPtr), inputBytes, &resourceCtx.inputToken);
        if (inputTokenResult != CCU_SUCCESS) {
            return ConvertCcuResult(inputTokenResult);
        }
        const CcuResult outputTokenResult = HcommCcuGetMemToken(
            reinterpret_cast<uint64_t>(param.outputPtr), recvBytes, &resourceCtx.outputToken);
        if (outputTokenResult != CCU_SUCCESS) {
            return ConvertCcuResult(outputTokenResult);
        }
    }
    resourceCtx.threads.push_back(param.cpuThread);

    std::vector<uint32_t> layers;
    std::vector<uint32_t> localRanks;
    Rs2TopologyKind topologyKind = Rs2TopologyKind::UNKNOWN;
    CHK_RET(GetTopology(comm, param, layers, localRanks, topologyKind));
    resourceCtx.topologyKind = static_cast<uint32_t>(topologyKind);
    resourceCtx.localGroupSize = static_cast<uint32_t>(localRanks.size());

    if (param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    ThreadConfig threadConfig{};
    CHK_RET(static_cast<HcclResult>(ThreadConfigInit(&threadConfig, 1)));
    threadConfig.notifyNumPerThread = RS2_BUILD_FLAVOR == Rs2BuildFlavor::DIRECT_PIPELINE ?
        DIRECT_PIPELINE_NOTIFY_COUNT : 1;
    ThreadHandle slaveThread = 0;
    CHK_RET(HcclThreadAcquireWithConfig(comm, CommEngine::COMM_ENGINE_CPU, 1, THREAD_TYPE_TS,
        &threadConfig, &slaveThread));
    resourceCtx.threads.push_back(slaveThread);

    std::vector<ChannelBinding> bindings;
    CHK_RET(AcquireChannels(comm, param, layers, localRanks, bindings));
    return RegisterKernels(comm, param, bindings, localRanks, resourceCtx);
}
} // namespace

HcclResult Run(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream, uint32_t rankSize)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(stream);
    if (dataType != HCCL_DATA_TYPE_FP32 || op != HCCL_REDUCE_SUM) {
        HCCL_ERROR("rs2-ccu supports FP32 SUM only");
        return HCCL_E_NOT_SUPPORT;
    }

    OpParam param;
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    param.rankSize = rankSize;
    if (param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || param.myRank >= param.rankSize) {
        return HCCL_E_PARA;
    }
    const bool smallInput = recvCount <= SMALL_INPUT_BYTES / sizeof(float) / param.rankSize;
    const int tagResult = smallInput ? std::snprintf(param.tag, sizeof(param.tag), "%s_small_%llx_%llx",
        GetContextTag(), static_cast<unsigned long long>(reinterpret_cast<uint64_t>(sendBuf)),
        static_cast<unsigned long long>(reinterpret_cast<uint64_t>(recvBuf))) :
        std::snprintf(param.tag, sizeof(param.tag), "%s_large", GetContextTag());
    if (tagResult <= 0 || static_cast<size_t>(tagResult) >= sizeof(param.tag)) {
        return HCCL_E_INTERNAL;
    }

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH] = {};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    constexpr uint32_t notifyCount = RS2_BUILD_FLAVOR == Rs2BuildFlavor::DIRECT_PIPELINE ?
        DIRECT_PIPELINE_NOTIFY_COUNT : 1;
    CHK_RET(HcclThreadAcquireWithStream(comm, CommEngine::COMM_ENGINE_CCU, stream, notifyCount,
        &param.cpuThread));

    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    if (HcclEngineCtxGet(comm, param.tag, CommEngine::COMM_ENGINE_CCU, &ctx, &ctxSize) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = ctxSize;
        param.root = smallInput ? 1U : 0U;
    } else {
        param.root = smallInput ? 1U : 0U;
        AlgResourceCtx resourceCtx;
        CHK_RET(CreateResource(comm, param, resourceCtx));
        const std::vector<char> sequence = resourceCtx.Serialize();
        param.ctxSize = sequence.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, CommEngine::COMM_ENGINE_CCU, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, CommEngine::COMM_ENGINE_CCU, param.tag, sequence.data(),
            sequence.size(), 0));
    }

    return ops_hccl::ExecOp(param);
}
} // namespace final_direct_pipeline

namespace final_adaptive41 {
namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 1;

struct ChannelBinding {
    uint32_t peerRank = INVALID_VALUE_RANKID;
    uint32_t layer = 0;
    uint32_t dieId = 0;
    ChannelHandle handle = 0;
};

using ChannelGroups = std::map<std::pair<uint32_t, uint32_t>, std::vector<ChannelBinding>>;

HcclResult ConvertCcuResult(CcuResult result)
{
    switch (result) {
        case CCU_SUCCESS:
            return HCCL_SUCCESS;
        case CCU_E_PARA:
            return HCCL_E_PARA;
        case CCU_E_PTR:
            return HCCL_E_PTR;
        case CCU_E_NOT_SUPPORT:
            return HCCL_E_NOT_SUPPORT;
        case CCU_E_NOT_FOUND:
            return HCCL_E_NOT_FOUND;
        case CCU_E_UNAVAIL:
            return HCCL_E_UNAVAIL;
        default:
            return HCCL_E_INTERNAL;
    }
}

const char *GetContextTag()
{
    switch (RS2_BUILD_FLAVOR) {
        case Rs2BuildFlavor::HIERARCHICAL:
            return "rs2_ccu_hierarchical";
        case Rs2BuildFlavor::HCCL_TOPOLOGY:
            return "rs2_ccu_hccl_topology";
        case Rs2BuildFlavor::GLOBAL_MESH_TREE:
            return "rs2_ccu_v7_global_tree";
        case Rs2BuildFlavor::ADAPTIVE:
            return "rs2_ccu_adaptive";
        case Rs2BuildFlavor::FOUR_BY_ONE_RING:
            return "adative_41ring";
        default:
            return "rs2_ccu_unknown";
    }
}

bool ContainsRank(const std::vector<uint32_t> &ranks, uint32_t rank)
{
    return std::find(ranks.begin(), ranks.end(), rank) != ranks.end();
}

HcclResult GetTopology(HcclComm comm, const OpParam &param, std::vector<uint32_t> &layers,
    std::vector<uint32_t> &localRanks, Rs2TopologyKind &topologyKind)
{
    uint32_t *layerData = nullptr;
    uint32_t layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layerData, &layerCount));
    layers.assign(layerData, layerData + layerCount);

    if (ContainsRank(layers, 0)) {
        uint32_t *rankData = nullptr;
        uint32_t localRankCount = 0;
        CHK_RET(HcclRankGraphGetRanksByLayer(comm, 0, &rankData, &localRankCount));
        localRanks.assign(rankData, rankData + localRankCount);
    } else {
        localRanks.push_back(param.myRank);
    }
    std::sort(localRanks.begin(), localRanks.end());

    topologyKind = Rs2TopologyKind::UNKNOWN;
    if (param.rankSize == 16 && localRanks.size() == 8) {
        topologyKind = Rs2TopologyKind::TWO_BY_EIGHT;
    } else if (param.rankSize == 4 && localRanks.size() == 1) {
        topologyKind = Rs2TopologyKind::FOUR_BY_ONE;
    } else if (param.rankSize == 12 && (localRanks.size() == 8 || localRanks.size() == 4)) {
        topologyKind = Rs2TopologyKind::EIGHT_PLUS_FOUR;
    }
    return HCCL_SUCCESS;
}

HcclResult FindChannelDesc(HcclComm comm, const OpParam &param, uint32_t peerRank,
    const std::vector<uint32_t> &layers, uint32_t preferredLayer, HcclChannelDesc &desc,
    uint32_t &selectedLayer, uint32_t &dieId)
{
    std::vector<uint32_t> candidateLayers;
    if (ContainsRank(layers, preferredLayer)) {
        candidateLayers.push_back(preferredLayer);
    }
    for (uint32_t layer : layers) {
        if (layer != preferredLayer) {
            candidateLayers.push_back(layer);
        }
    }

    for (uint32_t layer : candidateLayers) {
        CommLink *linkList = nullptr;
        uint32_t linkCount = 0;
        if (HcclRankGraphGetLinks(comm, layer, param.myRank, peerRank, &linkList, &linkCount) != HCCL_SUCCESS) {
            continue;
        }
        for (uint32_t linkIdx = 0; linkIdx < linkCount; ++linkIdx) {
            const CommLink &link = linkList[linkIdx];
            if (link.linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                continue;
            }

            CHK_RET(HcclChannelDescInit(&desc, 1));
            desc.remoteRank = peerRank;
            desc.notifyNum = CHANNEL_NOTIFY_NUM;
            desc.channelProtocol = link.linkAttr.linkProtocol;
            desc.localEndpoint = link.srcEndpointDesc;
            desc.remoteEndpoint = link.dstEndpointDesc;

            EndpointAttrDieId endpointDieId = 0;
            CHK_RET(HcclRankGraphGetEndpointInfo(comm, param.myRank, &desc.localEndpoint,
                ENDPOINT_ATTR_DIE_ID, sizeof(endpointDieId), &endpointDieId));
            selectedLayer = layer;
            dieId = endpointDieId;
            return HCCL_SUCCESS;
        }
    }

    HCCL_ERROR("No UBC_CTP link between rank[%u] and rank[%u]", param.myRank, peerRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult AcquireChannels(HcclComm comm, const OpParam &param, const std::vector<uint32_t> &layers,
    const std::vector<uint32_t> &localRanks, std::vector<ChannelBinding> &bindings)
{
    for (uint32_t peerRank = 0; peerRank < param.rankSize; ++peerRank) {
        if (peerRank == param.myRank) {
            continue;
        }

        const bool isLocalPeer = ContainsRank(localRanks, peerRank);
        const uint32_t preferredLayer = isLocalPeer ? 0 : 1;
        HcclChannelDesc desc;
        uint32_t selectedLayer = 0;
        uint32_t dieId = 0;
        CHK_RET(FindChannelDesc(comm, param, peerRank, layers, preferredLayer, desc, selectedLayer, dieId));

        ChannelHandle channel = 0;
        CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, &desc, 1, &channel));
        bindings.push_back(ChannelBinding{peerRank, selectedLayer, dieId, channel});
    }
    return HCCL_SUCCESS;
}

ChannelGroups GroupChannels(const std::vector<ChannelBinding> &bindings,
    const std::vector<uint32_t> &localRanks, int localFilter)
{
    ChannelGroups groups;
    for (const ChannelBinding &binding : bindings) {
        const bool isLocalPeer = ContainsRank(localRanks, binding.peerRank);
        if ((localFilter == 1 && !isLocalPeer) || (localFilter == 0 && isLocalPeer)) {
            continue;
        }
        groups[{binding.layer, binding.dieId}].push_back(binding);
    }
    return groups;
}

HcclResult RegisterKernelSet(CcuInsHandle insHandle, const ChannelGroups &groups, bool includeLocalFirst,
    bool initializeFirst, bool writeOutput, bool assignScratchSlots, std::vector<CcuKernelHandle> &kernelHandles,
    std::vector<uint32_t> &pieceCounts, std::vector<uint32_t> *kernelDieIds = nullptr,
    std::vector<uint32_t> *resultSlots = nullptr, uint32_t *scratchSlotCount = nullptr)
{
    uint32_t groupIdx = 0;
    for (const auto &groupEntry : groups) {
        const std::vector<ChannelBinding> &group = groupEntry.second;
        if (group.empty() || group.size() >= MAX_RANK_SIZE) {
            return HCCL_E_PARA;
        }

        CcuKernelInfo kernelInfo{};
        const int nameResult = std::snprintf(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName),
            "%s", "CcuReduceScatterTreeKernel");
        if (nameResult <= 0 || static_cast<size_t>(nameResult) >= sizeof(kernelInfo.kernelFuncName)) {
            return HCCL_E_INTERNAL;
        }
        kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuReduceScatterTreeKernel);

        auto kernelArg = std::make_shared<CcuKernelArgReduceScatterTree>();
        kernelArg->channelCount = static_cast<uint32_t>(group.size());
        kernelArg->includeLocalInput = includeLocalFirst && groupIdx == 0 ? 1U : 0U;
        kernelArg->initializeOutput = initializeFirst && groupIdx == 0 ? 1U : 0U;
        kernelArg->scratchSlotBase = assignScratchSlots && scratchSlotCount != nullptr ? *scratchSlotCount : 0U;
        kernelArg->writeOutput = writeOutput ? 1U : 0U;
        for (uint32_t channelIdx = 0; channelIdx < kernelArg->channelCount; ++channelIdx) {
            kernelArg->channels[channelIdx] = group[channelIdx].handle;
        }
        kernelInfo.SetKernelArg(kernelArg);

        CcuKernelHandle kernelHandle = 0;
        const void *kernelArgs[] = {kernelInfo.kernelArg};
        const CcuResult registerResult = HcommCcuKernelRegister(insHandle, groupEntry.first.second,
            kernelInfo.kernelFuncName, kernelInfo.kernelFunc, kernelArgs, 1, &kernelHandle);
        if (registerResult != CCU_SUCCESS) {
            return ConvertCcuResult(registerResult);
        }
        kernelHandles.push_back(kernelHandle);
        const uint32_t pieceCount = kernelArg->channelCount + kernelArg->includeLocalInput;
        pieceCounts.push_back(pieceCount);
        if (assignScratchSlots && kernelDieIds != nullptr && resultSlots != nullptr && scratchSlotCount != nullptr) {
            kernelDieIds->push_back(groupEntry.first.second);
            resultSlots->push_back(kernelArg->scratchSlotBase);
            *scratchSlotCount += pieceCount;
        }
        ++groupIdx;
    }
    return HCCL_SUCCESS;
}

const ChannelBinding *FindBinding(const std::vector<ChannelBinding> &bindings, uint32_t peerRank)
{
    for (const ChannelBinding &binding : bindings) {
        if (binding.peerRank == peerRank) {
            return &binding;
        }
    }
    return nullptr;
}

HcclResult RegisterRingKernel(CcuInsHandle insHandle, const OpParam &param,
    const std::vector<ChannelBinding> &bindings, AlgResourceCtx &resourceCtx)
{
    if (param.rankSize != 4) {
        return HCCL_E_NOT_SUPPORT;
    }
    const uint32_t nextRank = (param.myRank + 1) % param.rankSize;
    const uint32_t prevRank = (param.myRank + param.rankSize - 1) % param.rankSize;
    const ChannelBinding *nextBinding = FindBinding(bindings, nextRank);
    const ChannelBinding *prevBinding = FindBinding(bindings, prevRank);
    if (nextBinding == nullptr || prevBinding == nullptr || nextBinding->dieId != prevBinding->dieId) {
        HCCL_ERROR("Ring neighbors must be available on one IO Die, rank[%u]", param.myRank);
        return HCCL_E_NOT_SUPPORT;
    }

    CcuKernelInfo kernelInfo{};
    const int nameResult = std::snprintf(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName),
        "%s", "CcuReduceScatterRingKernel");
    if (nameResult <= 0 || static_cast<size_t>(nameResult) >= sizeof(kernelInfo.kernelFuncName)) {
        return HCCL_E_INTERNAL;
    }
    kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuReduceScatterRingKernel);

    auto kernelArg = std::make_shared<CcuKernelArgReduceScatterRing>();
    kernelArg->channelCount = 2;
    kernelArg->channels[0] = nextBinding->handle;
    kernelArg->channels[1] = prevBinding->handle;
    kernelArg->rankId = param.myRank;
    kernelArg->rankSize = param.rankSize;
    for (uint32_t stage = 0; stage < param.rankSize - 1; ++stage) {
        kernelArg->sendBlockIndices[stage] =
            (param.myRank + param.rankSize - stage - 1) % param.rankSize;
    }
    kernelInfo.SetKernelArg(kernelArg);

    const void *kernelArgs[] = {kernelInfo.kernelArg};
    const CcuResult registerResult = HcommCcuKernelRegister(insHandle, nextBinding->dieId,
        kernelInfo.kernelFuncName, kernelInfo.kernelFunc, kernelArgs, 1, &resourceCtx.ringKernel);
    if (registerResult != CCU_SUCCESS) {
        return ConvertCcuResult(registerResult);
    }
    resourceCtx.ringKernelDieId = nextBinding->dieId;
    return HCCL_SUCCESS;
}

constexpr uint32_t EIGHT_PLUS_FOUR_GROUP_SIZE_HOST = 4;
constexpr uint32_t EIGHT_PLUS_FOUR_GROUP_COUNT_HOST = 3;

HcclResult RegisterEightPlusFourPreReduceKernel(CcuInsHandle insHandle, const OpParam &param,
    const std::vector<ChannelBinding> &bindings, AlgResourceCtx &resourceCtx)
{
    const uint32_t groupIndex = param.myRank / EIGHT_PLUS_FOUR_GROUP_SIZE_HOST;
    const uint32_t groupRank = param.myRank % EIGHT_PLUS_FOUR_GROUP_SIZE_HOST;
    const uint32_t groupBase = groupIndex * EIGHT_PLUS_FOUR_GROUP_SIZE_HOST;
    std::vector<const ChannelBinding *> groupBindings;
    for (uint32_t rankOffset = 0; rankOffset < EIGHT_PLUS_FOUR_GROUP_SIZE_HOST; ++rankOffset) {
        const uint32_t peerRank = groupBase + rankOffset;
        if (peerRank == param.myRank) {
            continue;
        }
        const ChannelBinding *binding = FindBinding(bindings, peerRank);
        if (binding == nullptr) {
            return HCCL_E_NOT_FOUND;
        }
        groupBindings.push_back(binding);
    }
    if (groupBindings.size() != EIGHT_PLUS_FOUR_GROUP_SIZE_HOST - 1) {
        return HCCL_E_INTERNAL;
    }
    const uint32_t dieId = groupBindings[0]->dieId;
    for (const ChannelBinding *binding : groupBindings) {
        if (binding->dieId != dieId) {
            HCCL_ERROR("8+4 pre-reduce group crosses IO Dies, rank[%u]", param.myRank);
            return HCCL_E_NOT_SUPPORT;
        }
    }

    CcuKernelInfo kernelInfo{};
    const int nameResult = std::snprintf(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName),
        "%s", "CcuReduceScatterEightPlusFourPreReduceKernel");
    if (nameResult <= 0 || static_cast<size_t>(nameResult) >= sizeof(kernelInfo.kernelFuncName)) {
        return HCCL_E_INTERNAL;
    }
    kernelInfo.kernelFunc =
        reinterpret_cast<void *>(ops_hccl::CcuReduceScatterEightPlusFourPreReduceKernel);
    auto kernelArg = std::make_shared<CcuKernelArgEightPlusFourPreReduce>();
    kernelArg->channelCount = static_cast<uint32_t>(groupBindings.size());
    kernelArg->groupRank = groupRank;
    for (uint32_t channelIdx = 0; channelIdx < kernelArg->channelCount; ++channelIdx) {
        kernelArg->channels[channelIdx] = groupBindings[channelIdx]->handle;
    }
    kernelInfo.SetKernelArg(kernelArg);

    const void *kernelArgs[] = {kernelInfo.kernelArg};
    const CcuResult registerResult = HcommCcuKernelRegister(insHandle, dieId, kernelInfo.kernelFuncName,
        kernelInfo.kernelFunc, kernelArgs, 1, &resourceCtx.eightPlusFourPreReduceKernel);
    if (registerResult != CCU_SUCCESS) {
        return ConvertCcuResult(registerResult);
    }
    resourceCtx.eightPlusFourPreReduceKernelDieId = dieId;
    return HCCL_SUCCESS;
}

HcclResult RegisterEightPlusFourCommonKernels(CcuInsHandle insHandle, const OpParam &param,
    const std::vector<ChannelBinding> &bindings, AlgResourceCtx &resourceCtx)
{
    if (param.rankSize != EIGHT_PLUS_FOUR_GROUP_SIZE_HOST * EIGHT_PLUS_FOUR_GROUP_COUNT_HOST) {
        return HCCL_E_NOT_SUPPORT;
    }
    return RegisterEightPlusFourPreReduceKernel(insHandle, param, bindings, resourceCtx);
}
constexpr uint32_t EIGHT_PLUS_FOUR_EDGE_KERNEL_COUNT = 6;

HcclResult RegisterEightPlusFourRingEdgeKernels(CcuInsHandle insHandle, const OpParam &param,
    const std::vector<ChannelBinding> &bindings, AlgResourceCtx &resourceCtx)
{
    const uint32_t groupIndex = param.myRank / EIGHT_PLUS_FOUR_GROUP_SIZE_HOST;
    const uint32_t groupRank = param.myRank % EIGHT_PLUS_FOUR_GROUP_SIZE_HOST;
    constexpr uint32_t edgeGroups[EIGHT_PLUS_FOUR_GROUP_COUNT_HOST][2] = {{0, 1}, {1, 2}, {2, 0}};
    const char *kernelNames[EIGHT_PLUS_FOUR_EDGE_KERNEL_COUNT] = {
        "CcuReduceScatterEightPlusFourEdgeInitKernel",
        "CcuReduceScatterEightPlusFourEdgeSendStageZeroKernel",
        "CcuReduceScatterEightPlusFourEdgeWaitStageZeroKernel",
        "CcuReduceScatterEightPlusFourEdgeSendStageOneKernel",
        "CcuReduceScatterEightPlusFourEdgeWaitStageOneKernel",
        "CcuReduceScatterEightPlusFourEdgeCopyOutputKernel",
    };
    void *kernelFunctions[EIGHT_PLUS_FOUR_EDGE_KERNEL_COUNT] = {
        reinterpret_cast<void *>(ops_hccl::CcuReduceScatterEightPlusFourEdgeInitKernel),
        reinterpret_cast<void *>(ops_hccl::CcuReduceScatterEightPlusFourEdgeSendStageZeroKernel),
        reinterpret_cast<void *>(ops_hccl::CcuReduceScatterEightPlusFourEdgeWaitStageZeroKernel),
        reinterpret_cast<void *>(ops_hccl::CcuReduceScatterEightPlusFourEdgeSendStageOneKernel),
        reinterpret_cast<void *>(ops_hccl::CcuReduceScatterEightPlusFourEdgeWaitStageOneKernel),
        reinterpret_cast<void *>(ops_hccl::CcuReduceScatterEightPlusFourEdgeCopyOutputKernel),
    };
    const uint32_t resourceCount =
        EIGHT_PLUS_FOUR_GROUP_COUNT_HOST * EIGHT_PLUS_FOUR_EDGE_KERNEL_COUNT;
    resourceCtx.eightPlusFourRingEdgeKernels.assign(resourceCount, 0);
    resourceCtx.eightPlusFourRingEdgeKernelDieIds.assign(resourceCount, 0);

    std::vector<const ChannelBinding *> edgeBindings(EIGHT_PLUS_FOUR_GROUP_COUNT_HOST, nullptr);
    for (uint32_t edgeIdx = 0; edgeIdx < EIGHT_PLUS_FOUR_GROUP_COUNT_HOST; ++edgeIdx) {
        uint32_t peerGroup = EIGHT_PLUS_FOUR_GROUP_COUNT_HOST;
        if (groupIndex == edgeGroups[edgeIdx][0]) {
            peerGroup = edgeGroups[edgeIdx][1];
        } else if (groupIndex == edgeGroups[edgeIdx][1]) {
            peerGroup = edgeGroups[edgeIdx][0];
        } else {
            continue;
        }
        const uint32_t peerRank = peerGroup * EIGHT_PLUS_FOUR_GROUP_SIZE_HOST + groupRank;
        edgeBindings[edgeIdx] = FindBinding(bindings, peerRank);
        if (edgeBindings[edgeIdx] == nullptr) {
            return HCCL_E_NOT_FOUND;
        }
    }

    for (uint32_t kernelIdx = 0; kernelIdx < EIGHT_PLUS_FOUR_EDGE_KERNEL_COUNT; ++kernelIdx) {
        const CcuResult startResult = HcommCcuKernelRegisterStart(insHandle);
        if (startResult != CCU_SUCCESS) {
            return ConvertCcuResult(startResult);
        }
        for (uint32_t edgeIdx = 0; edgeIdx < EIGHT_PLUS_FOUR_GROUP_COUNT_HOST; ++edgeIdx) {
            const ChannelBinding *binding = edgeBindings[edgeIdx];
            if (binding == nullptr) {
                continue;
            }
            auto kernelArg = std::make_shared<CcuKernelArgEightPlusFourRingEdge>();
            kernelArg->channelCount = 1;
            kernelArg->channels[0] = binding->handle;
            kernelArg->groupIndex = groupIndex;
            kernelArg->groupRank = groupRank;

            CcuKernelInfo kernelInfo{};
            const int nameResult = std::snprintf(kernelInfo.kernelFuncName,
                sizeof(kernelInfo.kernelFuncName), "%s_edge%u", kernelNames[kernelIdx], edgeIdx);
            if (nameResult <= 0 || static_cast<size_t>(nameResult) >= sizeof(kernelInfo.kernelFuncName)) {
                return HCCL_E_INTERNAL;
            }
            kernelInfo.kernelFunc = kernelFunctions[kernelIdx];
            kernelInfo.SetKernelArg(kernelArg);
            const void *kernelArgs[] = {kernelInfo.kernelArg};
            const uint32_t resourceIdx =
                edgeIdx * EIGHT_PLUS_FOUR_EDGE_KERNEL_COUNT + kernelIdx;
            const CcuResult registerResult = HcommCcuKernelRegister(insHandle, binding->dieId,
                kernelInfo.kernelFuncName, kernelInfo.kernelFunc, kernelArgs, 1,
                &resourceCtx.eightPlusFourRingEdgeKernels[resourceIdx]);
            if (registerResult != CCU_SUCCESS) {
                HCCL_ERROR("8+4 edge kernel register failed, rank[%u], edge[%u], op[%u], result[%d]",
                    param.myRank, edgeIdx, kernelIdx, registerResult);
                return ConvertCcuResult(registerResult);
            }
            resourceCtx.eightPlusFourRingEdgeKernelDieIds[resourceIdx] = binding->dieId;
        }
        const CcuResult endResult = HcommCcuKernelRegisterEnd(insHandle);
        if (endResult != CCU_SUCCESS) {
            return ConvertCcuResult(endResult);
        }
    }
    return HCCL_SUCCESS;
}
HcclResult RegisterKernels(HcclComm comm, const OpParam &param, const std::vector<ChannelBinding> &bindings,
    const std::vector<uint32_t> &localRanks, AlgResourceCtx &resourceCtx)
{
    CcuInsHandle insHandle = 0;
    uint32_t insCount = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insCount));
    if (insCount == 0) {
        return HCCL_E_INTERNAL;
    }
    const bool useEightPlusFourEdgeKernels =
        RS2_BUILD_FLAVOR == Rs2BuildFlavor::FOUR_BY_ONE_RING &&
        static_cast<Rs2TopologyKind>(resourceCtx.topologyKind) == Rs2TopologyKind::EIGHT_PLUS_FOUR;

    const CcuResult startResult = HcommCcuKernelRegisterStart(insHandle);
    if (startResult != CCU_SUCCESS) {
        return ConvertCcuResult(startResult);
    }

    if (RS2_BUILD_FLAVOR == Rs2BuildFlavor::HCCL_TOPOLOGY ||
        RS2_BUILD_FLAVOR == Rs2BuildFlavor::GLOBAL_MESH_TREE ||
        RS2_BUILD_FLAVOR == Rs2BuildFlavor::ADAPTIVE ||
        RS2_BUILD_FLAVOR == Rs2BuildFlavor::FOUR_BY_ONE_RING) {
        const ChannelGroups globalGroups = GroupChannels(bindings, localRanks, -1);
        CHK_RET(RegisterKernelSet(insHandle, globalGroups, true, false, false, true,
            resourceCtx.globalKernels, resourceCtx.globalPieceCounts, &resourceCtx.globalKernelDieIds,
            &resourceCtx.globalResultSlots, &resourceCtx.globalScratchSlotCount));
        if (RS2_BUILD_FLAVOR == Rs2BuildFlavor::FOUR_BY_ONE_RING &&
            static_cast<Rs2TopologyKind>(resourceCtx.topologyKind) == Rs2TopologyKind::FOUR_BY_ONE) {
            CHK_RET(RegisterRingKernel(insHandle, param, bindings, resourceCtx));
        } else if (RS2_BUILD_FLAVOR == Rs2BuildFlavor::FOUR_BY_ONE_RING &&
            static_cast<Rs2TopologyKind>(resourceCtx.topologyKind) == Rs2TopologyKind::EIGHT_PLUS_FOUR) {
            CHK_RET(RegisterEightPlusFourCommonKernels(insHandle, param, bindings, resourceCtx));
        }
    } else {
        const ChannelGroups localGroups = GroupChannels(bindings, localRanks, 1);
        const ChannelGroups remoteGroups = GroupChannels(bindings, localRanks, 0);
        CHK_RET(RegisterKernelSet(insHandle, localGroups, true, true, true, false,
            resourceCtx.localKernels, resourceCtx.localPieceCounts));
        const bool remoteStartsResult = localGroups.empty();
        CHK_RET(RegisterKernelSet(insHandle, remoteGroups, remoteStartsResult, remoteStartsResult, true, false,
            resourceCtx.remoteKernels, resourceCtx.remotePieceCounts));
    }

    const CcuResult endResult = HcommCcuKernelRegisterEnd(insHandle);
    if (endResult != CCU_SUCCESS) {
        return ConvertCcuResult(endResult);
    }
    if (useEightPlusFourEdgeKernels) {
        return RegisterEightPlusFourRingEdgeKernels(insHandle, param, bindings, resourceCtx);
    }
    return HCCL_SUCCESS;
}

HcclResult CreateResource(HcclComm comm, OpParam &param, AlgResourceCtx &resourceCtx)
{
    void *cclBuffer = nullptr;
    uint64_t cclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBuffer, &cclBufferSize));
    resourceCtx.localBuffer = CommBuffer{cclBuffer, cclBufferSize};
    resourceCtx.threads.push_back(param.cpuThread);

    std::vector<uint32_t> layers;
    std::vector<uint32_t> localRanks;
    Rs2TopologyKind topologyKind = Rs2TopologyKind::UNKNOWN;
    CHK_RET(GetTopology(comm, param, layers, localRanks, topologyKind));
    resourceCtx.topologyKind = static_cast<uint32_t>(topologyKind);
    resourceCtx.localGroupSize = static_cast<uint32_t>(localRanks.size());

    if (param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    ThreadConfig threadConfig{};
    CHK_RET(static_cast<HcclResult>(ThreadConfigInit(&threadConfig, 1)));
    threadConfig.notifyNumPerThread = 1;
    ThreadHandle slaveThread = 0;
    CHK_RET(HcclThreadAcquireWithConfig(comm, CommEngine::COMM_ENGINE_CPU, 1, THREAD_TYPE_TS,
        &threadConfig, &slaveThread));
    resourceCtx.threads.push_back(slaveThread);

    std::vector<ChannelBinding> bindings;
    CHK_RET(AcquireChannels(comm, param, layers, localRanks, bindings));
    return RegisterKernels(comm, param, bindings, localRanks, resourceCtx);
}
} // namespace

HcclResult Run(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream, uint32_t rankSize)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(stream);
    if (dataType != HCCL_DATA_TYPE_FP32 || op != HCCL_REDUCE_SUM) {
        HCCL_ERROR("rs2-ccu supports FP32 SUM only");
        return HCCL_E_NOT_SUPPORT;
    }

    OpParam param;
    const int tagResult = std::snprintf(param.tag, sizeof(param.tag), "%s", GetContextTag());
    if (tagResult <= 0 || static_cast<size_t>(tagResult) >= sizeof(param.tag)) {
        return HCCL_E_INTERNAL;
    }
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    param.rankSize = rankSize;
    if (param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || param.myRank >= param.rankSize) {
        return HCCL_E_PARA;
    }

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH] = {};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    constexpr uint32_t notifyCount = 1;
    CHK_RET(HcclThreadAcquireWithStream(comm, CommEngine::COMM_ENGINE_CCU, stream, notifyCount,
        &param.cpuThread));

    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    if (HcclEngineCtxGet(comm, param.tag, CommEngine::COMM_ENGINE_CCU, &ctx, &ctxSize) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = ctxSize;
    } else {
        AlgResourceCtx resourceCtx;
        CHK_RET(CreateResource(comm, param, resourceCtx));
        const std::vector<char> sequence = resourceCtx.Serialize();
        param.ctxSize = sequence.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, CommEngine::COMM_ENGINE_CCU, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, CommEngine::COMM_ENGINE_CCU, param.tag, sequence.data(),
            sequence.size(), 0));
    }

    return ops_hccl::ExecOp(param);
}
} // namespace final_adaptive41

namespace {
constexpr uint64_t MIB = 1024ULL * 1024ULL;
constexpr uint64_t SMALL_INPUT_LIMIT = 1ULL * MIB;
constexpr uint64_t MEDIUM_INPUT_LIMIT = 400ULL * MIB;

bool IsMediumInput(uint64_t inputBytes, uint32_t rankSize)
{
    const uint64_t roundingAllowance = static_cast<uint64_t>(rankSize) * sizeof(float);
    return inputBytes <= MEDIUM_INPUT_LIMIT + roundingAllowance;
}
} // namespace

HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    uint32_t rankSize = 0;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    if (rankSize == 0 || rankSize > MAX_RANK_SIZE ||
        recvCount > std::numeric_limits<uint64_t>::max() / sizeof(float) / rankSize) {
        return HCCL_E_PARA;
    }

    const uint64_t inputBytes = recvCount * sizeof(float) * rankSize;
    if (rankSize == 4) {
        return final_adaptive41::Run(sendBuf, recvBuf, recvCount, dataType, op, comm, stream, rankSize);
    }
    if (rankSize == 16) {
        if (inputBytes <= SMALL_INPUT_LIMIT) {
            return final_small512::Run(sendBuf, recvBuf, recvCount, dataType, op, comm, stream, rankSize);
        }
        if (IsMediumInput(inputBytes, rankSize)) {
            return final_epoch_cache::Run(sendBuf, recvBuf, recvCount, dataType, op, comm, stream, rankSize);
        }
        return final_scratch_lite::Run(sendBuf, recvBuf, recvCount, dataType, op, comm, stream, rankSize);
    }
    if (rankSize == 12) {
        if (inputBytes <= SMALL_INPUT_LIMIT) {
            return final_small512::Run(sendBuf, recvBuf, recvCount, dataType, op, comm, stream, rankSize);
        }
        if (IsMediumInput(inputBytes, rankSize)) {
            return final_direct_pipeline::Run(sendBuf, recvBuf, recvCount, dataType, op, comm, stream, rankSize);
        }
        return final_epoch_cache::Run(sendBuf, recvBuf, recvCount, dataType, op, comm, stream, rankSize);
    }

    HCCL_ERROR("rs2-ccu-final supports only rank sizes 4, 12, and 16, rankSize[%u]", rankSize);
    return HCCL_E_NOT_SUPPORT;
}
