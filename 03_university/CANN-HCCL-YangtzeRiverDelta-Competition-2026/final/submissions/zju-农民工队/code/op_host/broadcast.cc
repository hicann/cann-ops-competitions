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
#include <memory>
#include <vector>

#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>
#include "ccu_launch.h"
#include "hccl_ccu_res.h"

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "exec_op.h"
#include "ccu_kernel.h"

namespace ops_hccl {
CcuResult CcuSmallFlatKernel(CcuKernelArg arg);
CcuResult CcuFourRankPipelineKernel(CcuKernelArg arg);
CcuResult CcuTwelveLayerKernel(CcuKernelArg arg);
CcuResult CcuSixteenScatterKernel(CcuKernelArg arg);
CcuResult CcuSixteenFlatAllGatherKernel(CcuKernelArg arg);
} // namespace ops_hccl

namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;
constexpr uint32_t SIXTEEN_LOCAL_ALLGATHER_MARKER = 7;
constexpr uint32_t SIXTEEN_CROSS_ALLGATHER_MARKER = 8;

struct ChannelPlanResources {
    std::vector<ChannelHandle> layerChannels[2];
    std::vector<ChannelHandle> pairChannels;
    std::vector<uint32_t> pairPeers;
    std::vector<uint32_t> pairLayers;
    std::vector<ChannelHandle> sixteenScatterChannels[2];
    std::vector<uint32_t> sixteenScatterPeers[2];
    std::vector<ChannelHandle> sixteenAllGatherChannels[2];
};

HcclResult AcquireChannel(HcclComm comm, CommEngine engine, uint32_t myRank, uint32_t remoteRank,
    ChannelHandle &channel, uint32_t &selectedLayer)
{
    CommLink *selectedLink = nullptr;
    for (uint32_t netLayer = 0; netLayer <= 1 && selectedLink == nullptr; ++netLayer) {
        uint32_t listSize = 0;
        CommLink *linkList = nullptr;
        HcclResult ret = HcclRankGraphGetLinks(comm, netLayer, myRank, remoteRank, &linkList, &listSize);
        if (ret != HCCL_SUCCESS) {
            continue;
        }
        for (uint32_t i = 0; i < listSize; ++i) {
            if (linkList[i].linkAttr.linkProtocol == CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                selectedLink = &linkList[i];
                selectedLayer = netLayer;
                break;
            }
        }
    }
    CHK_PRT_RET(selectedLink == nullptr,
        HCCL_ERROR("No UBC_CTP link between rank[%u] and rank[%u]", myRank, remoteRank), HCCL_E_NOT_FOUND);

    HcclChannelDesc desc;
    CHK_RET(HcclChannelDescInit(&desc, 1));
    desc.remoteRank = remoteRank;
    desc.notifyNum = CHANNEL_NOTIFY_NUM;
    desc.channelProtocol = selectedLink->linkAttr.linkProtocol;
    desc.localEndpoint.protocol = selectedLink->srcEndpointDesc.protocol;
    desc.localEndpoint.commAddr = selectedLink->srcEndpointDesc.commAddr;
    desc.localEndpoint.loc = selectedLink->srcEndpointDesc.loc;
    desc.remoteEndpoint.protocol = selectedLink->dstEndpointDesc.protocol;
    desc.remoteEndpoint.commAddr = selectedLink->dstEndpointDesc.commAddr;
    desc.remoteEndpoint.loc = selectedLink->dstEndpointDesc.loc;
    CHK_RET(HcclChannelAcquire(comm, engine, &desc, 1, &channel));
    return HCCL_SUCCESS;
}

HcclResult RegisterKernels(HcclComm comm, const OpParam &param, const std::vector<ChannelHandle> (&layerChannels)[2],
    AlgResourceCtx &resCtxHost, bool useSmallFlatKernel = false)
{
    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1, HCCL_ERROR("Unexpected CCU instance count[%u]", insNum), HCCL_E_INTERNAL);

    CcuResult ccuRet = HcommCcuKernelRegisterStart(insHandle);
    CHK_PRT_RET(
        ccuRet != CCU_SUCCESS, HCCL_ERROR("CCU kernel register start failed[%d]", ccuRet), ConvertCcuToHccl(ccuRet));

    std::vector<CcuKernelInfo> kernelInfos(2);
    for (uint32_t layer = 0; layer < 2; ++layer) {
        if (layerChannels[layer].empty()) {
            continue;
        }

        CcuKernelInfo &kernelInfo = kernelInfos[layer];
        if (useSmallFlatKernel) {
            sprintf(kernelInfo.kernelFuncName, "CcuBroadcastSmallRoot_%u_Layer_%u", param.root, layer);
            kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuSmallFlatKernel);
        } else {
            sprintf(kernelInfo.kernelFuncName, "CcuBroadcastRoot_%u_Layer_%u", param.root, layer);
            kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuKernel);
        }
        auto kernelArg = std::make_shared<CcuKernelArgBase>();
        kernelArg->channelCount = static_cast<uint32_t>(layerChannels[layer].size());
        kernelArg->isSender = static_cast<uint32_t>(param.myRank == param.root);
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            kernelArg->channels[i] = layerChannels[layer][i];
        }
        kernelInfo.setKernelArg(kernelArg);

        const void *kernelArgs[] = {kernelInfo.kernelArg};
        constexpr uint32_t dieId = 0;
        constexpr uint32_t kernelArgNum = 1;
        CcuKernelHandle kernelHandle;
        ccuRet = HcommCcuKernelRegister(insHandle, dieId, kernelInfo.kernelFuncName, kernelInfo.kernelFunc, kernelArgs,
            kernelArgNum, &kernelHandle);
        CHK_PRT_RET(ccuRet != CCU_SUCCESS, HCCL_ERROR("CCU kernel register for layer[%u] failed[%d]", layer, ccuRet),
            ConvertCcuToHccl(ccuRet));
        resCtxHost.ccuKernels.push_back(kernelHandle);
        resCtxHost.kernelLayers.push_back(layer);
    }

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    CHK_PRT_RET(
        ccuRet != CCU_SUCCESS, HCCL_ERROR("CCU kernel register end failed[%d]", ccuRet), ConvertCcuToHccl(ccuRet));
    return HCCL_SUCCESS;
}

HcclResult RegisterPairKernels(
    HcclComm comm, const OpParam &param, const ChannelPlanResources &channels, AlgResourceCtx &resCtxHost)
{
    CHK_PRT_RET(channels.pairChannels.size() != channels.pairPeers.size()
                    || channels.pairChannels.size() != channels.pairLayers.size(),
        HCCL_ERROR("Invalid pair channel resources"), HCCL_E_INTERNAL);

    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1, HCCL_ERROR("Unexpected CCU instance count[%u]", insNum), HCCL_E_INTERNAL);

    CcuResult ccuRet = HcommCcuKernelRegisterStart(insHandle);
    CHK_PRT_RET(
        ccuRet != CCU_SUCCESS, HCCL_ERROR("CCU kernel register start failed[%d]", ccuRet), ConvertCcuToHccl(ccuRet));

    std::vector<CcuKernelInfo> kernelInfos(channels.pairChannels.size());
    for (uint32_t index = 0; index < channels.pairChannels.size(); ++index) {
        const uint32_t peer = channels.pairPeers[index];
        const uint32_t lowRank = std::min(param.myRank, peer);
        const uint32_t highRank = std::max(param.myRank, peer);
        CcuKernelInfo &kernelInfo = kernelInfos[index];
        sprintf(kernelInfo.kernelFuncName, "CcuBroadcastRoot_%u_Pair_%u_%u", param.root, lowRank, highRank);
        kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuKernel);

        auto kernelArg = std::make_shared<CcuKernelArgBase>();
        kernelArg->channels[0] = channels.pairChannels[index];
        kernelArg->channelCount = 1;
        kernelInfo.setKernelArg(kernelArg);

        const void *kernelArgs[] = {kernelInfo.kernelArg};
        constexpr uint32_t dieId = 0;
        constexpr uint32_t kernelArgNum = 1;
        CcuKernelHandle kernelHandle;
        ccuRet = HcommCcuKernelRegister(insHandle, dieId, kernelInfo.kernelFuncName, kernelInfo.kernelFunc, kernelArgs,
            kernelArgNum, &kernelHandle);
        CHK_PRT_RET(ccuRet != CCU_SUCCESS, HCCL_ERROR("CCU pair kernel register for peer[%u] failed[%d]", peer, ccuRet),
            ConvertCcuToHccl(ccuRet));
        resCtxHost.ccuKernels.push_back(kernelHandle);
        resCtxHost.kernelLayers.push_back(channels.pairLayers[index]);
        resCtxHost.kernelPeers.push_back(peer);
    }

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    CHK_PRT_RET(
        ccuRet != CCU_SUCCESS, HCCL_ERROR("CCU kernel register end failed[%d]", ccuRet), ConvertCcuToHccl(ccuRet));
    return HCCL_SUCCESS;
}

HcclResult RegisterFourRankPipelineKernel(
    HcclComm comm, const OpParam &param, const ChannelPlanResources &channels, AlgResourceCtx &resCtxHost)
{
    const uint32_t position = (param.myRank + 4U - param.root) % 4U;
    const uint32_t expectedChannels = (position == 0U || position == 3U) ? 1U : 2U;
    CHK_PRT_RET(param.rankSize != 4U || channels.pairChannels.size() != expectedChannels,
        HCCL_ERROR("Invalid four-rank pipeline resources on rank[%u]", param.myRank), HCCL_E_INTERNAL);

    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1, HCCL_ERROR("Unexpected CCU instance count[%u]", insNum), HCCL_E_INTERNAL);
    CcuResult ccuRet = HcommCcuKernelRegisterStart(insHandle);
    CHK_PRT_RET(
        ccuRet != CCU_SUCCESS, HCCL_ERROR("CCU kernel register start failed[%d]", ccuRet), ConvertCcuToHccl(ccuRet));

    CcuKernelInfo kernelInfo{};
    sprintf(kernelInfo.kernelFuncName, "CcuBroadcast4Pipeline_Root_%u_Rank_%u", param.root, param.myRank);
    kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuFourRankPipelineKernel);
    auto kernelArg = std::make_shared<CcuFourRankPipelineKernelArg>();
    kernelArg->position = position;
    kernelArg->channelCount = expectedChannels;
    for (uint32_t index = 0; index < expectedChannels; ++index) {
        kernelArg->channels[index] = channels.pairChannels[index];
    }
    kernelInfo.setKernelArg(kernelArg);

    const void *kernelArgs[] = {kernelInfo.kernelArg};
    CcuKernelHandle kernelHandle;
    ccuRet = HcommCcuKernelRegister(
        insHandle, 0, kernelInfo.kernelFuncName, kernelInfo.kernelFunc, kernelArgs, 1, &kernelHandle);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS, HCCL_ERROR("CCU four-rank pipeline kernel register failed[%d]", ccuRet),
        ConvertCcuToHccl(ccuRet));
    resCtxHost.ccuKernels.push_back(kernelHandle);
    resCtxHost.kernelLayers.push_back(channels.pairLayers[0]);

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    CHK_PRT_RET(
        ccuRet != CCU_SUCCESS, HCCL_ERROR("CCU kernel register end failed[%d]", ccuRet), ConvertCcuToHccl(ccuRet));
    return HCCL_SUCCESS;
}

HcclResult CreateFlatRootChannels(
    HcclComm comm, CommEngine engine, const OpParam &param, std::vector<ChannelHandle> (&layerChannels)[2])
{
    if (param.myRank == param.root) {
        for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
            if (remoteRank == param.myRank) {
                continue;
            }
            ChannelHandle channel{};
            uint32_t layer = 0;
            CHK_RET(AcquireChannel(comm, engine, param.myRank, remoteRank, channel, layer));
            layerChannels[layer].push_back(channel);
        }
    } else {
        ChannelHandle channel{};
        uint32_t layer = 0;
        CHK_RET(AcquireChannel(comm, engine, param.myRank, param.root, channel, layer));
        layerChannels[layer].push_back(channel);
    }
    return HCCL_SUCCESS;
}

HcclResult CreateFourRankTreeChannels(
    HcclComm comm, CommEngine engine, const OpParam &param, ChannelPlanResources &channels)
{
    const uint32_t firstPeer = FourRankFirstPeer(param.root);
    const uint32_t secondPeer = FourRankSecondPeer(param.root);
    const uint32_t forwardPeer = FourRankForwardPeer(param.root);
    std::vector<uint32_t> peers;
    peers.reserve(2);
    if (param.myRank == param.root) {
        peers.push_back(firstPeer);
        peers.push_back(secondPeer);
    } else if (param.myRank == firstPeer) {
        peers.push_back(param.root);
        peers.push_back(forwardPeer);
    } else if (param.myRank == secondPeer) {
        peers.push_back(param.root);
    } else {
        peers.push_back(firstPeer);
    }

    for (uint32_t peer : peers) {
        ChannelHandle channel{};
        uint32_t layer = 0;
        CHK_RET(AcquireChannel(comm, engine, param.myRank, peer, channel, layer));
        channels.pairChannels.push_back(channel);
        channels.pairPeers.push_back(peer);
        channels.pairLayers.push_back(layer);
    }
    return HCCL_SUCCESS;
}

HcclResult CreateFourRankPipelineChannels(
    HcclComm comm, CommEngine engine, const OpParam &param, ChannelPlanResources &channels)
{
    const uint32_t firstPeer = FourRankFirstPeer(param.root);
    const uint32_t secondPeer = FourRankSecondPeer(param.root);
    const uint32_t forwardPeer = FourRankForwardPeer(param.root);
    std::vector<uint32_t> peers;
    peers.reserve(2);
    if (param.myRank == param.root) {
        peers.push_back(firstPeer);
    } else if (param.myRank == firstPeer) {
        peers.push_back(param.root);
        peers.push_back(secondPeer);
    } else if (param.myRank == secondPeer) {
        peers.push_back(firstPeer);
        peers.push_back(forwardPeer);
    } else {
        peers.push_back(secondPeer);
    }

    for (uint32_t peer : peers) {
        ChannelHandle channel{};
        uint32_t layer = 0;
        CHK_RET(AcquireChannel(comm, engine, param.myRank, peer, channel, layer));
        channels.pairChannels.push_back(channel);
        channels.pairPeers.push_back(peer);
        channels.pairLayers.push_back(layer);
    }
    return HCCL_SUCCESS;
}

HcclResult CreateTwelvePipelineChannels(
    HcclComm comm, CommEngine engine, const OpParam &param, ChannelPlanResources &channels)
{
    CHK_PRT_RET(param.rankSize != 12 || param.root >= 8,
        HCCL_ERROR("Twelve-rank pipeline requires rankSize[12] and root on the 8-rank server"), HCCL_E_PARA);

    std::vector<uint32_t> peers;
    if (param.myRank < 8) {
        // A/B 在 8-rank Server 内 Scatter/AllGather。
        for (uint32_t peer = 0; peer < 8; ++peer) {
            if (peer != param.myRank) {
                peers.push_back(peer);
            }
        }
        // 四条跨机 channel 组成 A/B 与 C/D 的两轮全双工交换面。
        for (uint32_t peer = 8; peer < 12; ++peer) {
            peers.push_back(peer);
        }
    } else {
        // C/D 在 4-rank Server 内 AllGather。
        for (uint32_t peer = 8; peer < 12; ++peer) {
            if (peer != param.myRank) {
                peers.push_back(peer);
            }
        }
        for (uint32_t peer = 0; peer < 8; ++peer) {
            peers.push_back(peer);
        }
    }

    std::sort(peers.begin(), peers.end());
    peers.erase(std::unique(peers.begin(), peers.end()), peers.end());
    for (uint32_t peer : peers) {
        ChannelHandle channel{};
        uint32_t layer = 0;
        CHK_RET(AcquireChannel(comm, engine, param.myRank, peer, channel, layer));
        const uint32_t expectedLayer = (param.myRank < 8) == (peer < 8) ? 0 : 1;
        CHK_PRT_RET(layer != expectedLayer,
            HCCL_ERROR("Twelve-rank peer[%u] is on layer[%u], expected[%u]", peer, layer, expectedLayer),
            HCCL_E_INTERNAL);
        channels.pairChannels.push_back(channel);
        channels.pairPeers.push_back(peer);
        channels.pairLayers.push_back(layer);
    }
    return HCCL_SUCCESS;
}
HcclResult CreateSixteenFusedChannels(
    HcclComm comm, CommEngine engine, const OpParam &param, ChannelPlanResources &channels)
{
    CHK_PRT_RET(param.rankSize != 16, HCCL_ERROR("Sixteen-rank fused plan requires rankSize[16]"), HCCL_E_PARA);

    std::array<ChannelHandle, 16> handles{};
    std::array<uint32_t, 16> layers{};
    std::array<bool, 16> acquired{};
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer == param.myRank) {
            continue;
        }
        ChannelHandle channel{};
        uint32_t layer = 0;
        CHK_RET(AcquireChannel(comm, engine, param.myRank, peer, channel, layer));
        handles[peer] = channel;
        layers[peer] = layer;
        acquired[peer] = true;
    }

    if (param.myRank == param.root) {
        for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
            if (peer == param.myRank) {
                continue;
            }
            const uint32_t layer = layers[peer];
            CHK_PRT_RET(layer > 1, HCCL_ERROR("Invalid scatter layer[%u] for peer[%u]", layer, peer), HCCL_E_INTERNAL);
            channels.sixteenScatterChannels[layer].push_back(handles[peer]);
            channels.sixteenScatterPeers[layer].push_back(peer);
        }
    } else {
        CHK_PRT_RET(!acquired[param.root], HCCL_ERROR("Root channel was not acquired"), HCCL_E_INTERNAL);
        const uint32_t layer = layers[param.root];
        CHK_PRT_RET(
            layer > 1, HCCL_ERROR("Invalid scatter layer[%u] for root[%u]", layer, param.root), HCCL_E_INTERNAL);
        channels.sixteenScatterChannels[layer].push_back(handles[param.root]);
        channels.sixteenScatterPeers[layer].push_back(param.root);
    }

    const uint32_t serverBase = (param.myRank / 8U) * 8U;
    for (uint32_t peer = serverBase; peer < serverBase + 8U; ++peer) {
        if (peer == param.myRank) {
            continue;
        }
        CHK_PRT_RET(!acquired[peer] || layers[peer] != 0,
            HCCL_ERROR("Local AllGather peer[%u] is not on layer 0", peer), HCCL_E_INTERNAL);
        channels.sixteenAllGatherChannels[0].push_back(handles[peer]);
    }

    const uint32_t localRank = param.myRank % 8U;
    for (uint32_t round = 0; round < 8U; ++round) {
        const uint32_t peer = param.myRank < 8U ? 8U + (localRank + round) % 8U : (localRank + 8U - round) % 8U;
        CHK_PRT_RET(!acquired[peer] || layers[peer] != 1,
            HCCL_ERROR("Cross AllGather peer[%u] is not on layer 1", peer), HCCL_E_INTERNAL);
        channels.sixteenAllGatherChannels[1].push_back(handles[peer]);
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterSixteenFusedKernels(
    HcclComm comm, const OpParam &param, const BroadcastPlan &plan,
    const ChannelPlanResources &channels, AlgResourceCtx &resCtxHost)
{
    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1, HCCL_ERROR("Unexpected CCU instance count[%u]", insNum), HCCL_E_INTERNAL);

    CcuResult ccuRet = HcommCcuKernelRegisterStart(insHandle);
    CHK_PRT_RET(
        ccuRet != CCU_SUCCESS, HCCL_ERROR("CCU kernel register start failed[%d]", ccuRet), ConvertCcuToHccl(ccuRet));

    std::vector<CcuKernelInfo> kernelInfos(4);
    uint32_t infoIndex = 0;
    for (uint32_t layer = 0; layer < 2; ++layer) {
        if (channels.sixteenScatterChannels[layer].empty()) {
            continue;
        }
        CcuKernelInfo &kernelInfo = kernelInfos[infoIndex++];
        sprintf(kernelInfo.kernelFuncName, "CcuBroadcast16Scatter_Root_%u_Layer_%u", param.root, layer);
        kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuSixteenScatterKernel);
        auto kernelArg = std::make_shared<CcuSixteenScatterKernelArg>();
        kernelArg->channelCount = static_cast<uint32_t>(channels.sixteenScatterChannels[layer].size());
        kernelArg->isSender = static_cast<uint32_t>(param.myRank == param.root);
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            kernelArg->channels[i] = channels.sixteenScatterChannels[layer][i];
            kernelArg->peerSlices[i] = channels.sixteenScatterPeers[layer][i];
        }
        kernelInfo.setKernelArg(kernelArg);
        const void *kernelArgs[] = {kernelInfo.kernelArg};
        CcuKernelHandle kernelHandle;
        ccuRet = HcommCcuKernelRegister(
            insHandle, 0, kernelInfo.kernelFuncName, kernelInfo.kernelFunc, kernelArgs, 1, &kernelHandle);
        CHK_PRT_RET(ccuRet != CCU_SUCCESS, HCCL_ERROR("CCU sixteen-rank scatter kernel register failed[%d]", ccuRet),
            ConvertCcuToHccl(ccuRet));
        resCtxHost.ccuKernels.push_back(kernelHandle);
        resCtxHost.kernelLayers.push_back(layer);
    }

    constexpr uint32_t expectedCounts[2] = {7, 8};
    constexpr uint32_t markers[2] = {SIXTEEN_LOCAL_ALLGATHER_MARKER, SIXTEEN_CROSS_ALLGATHER_MARKER};
    for (uint32_t workflow = 0; workflow < 2; ++workflow) {
        CHK_PRT_RET(channels.sixteenAllGatherChannels[workflow].size() != expectedCounts[workflow],
            HCCL_ERROR("Sixteen-rank flat workflow[%u] requires[%u] channels", workflow, expectedCounts[workflow]),
            HCCL_E_INTERNAL);
        CcuKernelInfo &kernelInfo = kernelInfos[infoIndex++];
        sprintf(kernelInfo.kernelFuncName, "CcuBroadcast16Flat_Root_%u_Workflow_%u", param.root, workflow);
        kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuSixteenFlatAllGatherKernel);
        auto kernelArg = std::make_shared<CcuSixteenFlatKernelArg>();
        kernelArg->channelCount = static_cast<uint32_t>(channels.sixteenAllGatherChannels[workflow].size());
        kernelArg->isSender = 1;
        kernelArg->myRank = param.myRank;
        kernelArg->serialized = workflow;
        kernelArg->deferCompletionWait
            = static_cast<uint32_t>(plan.sixteenDeferCrossWait != 0U && workflow == 1U);
        for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
            kernelArg->channels[i] = channels.sixteenAllGatherChannels[workflow][i];
        }
        kernelInfo.setKernelArg(kernelArg);
        const void *kernelArgs[] = {kernelInfo.kernelArg};
        CcuKernelHandle kernelHandle;
        ccuRet = HcommCcuKernelRegister(
            insHandle, 0, kernelInfo.kernelFuncName, kernelInfo.kernelFunc, kernelArgs, 1, &kernelHandle);
        CHK_PRT_RET(ccuRet != CCU_SUCCESS, HCCL_ERROR("CCU sixteen-rank flat kernel register failed[%d]", ccuRet),
            ConvertCcuToHccl(ccuRet));
        resCtxHost.ccuKernels.push_back(kernelHandle);
        resCtxHost.kernelLayers.push_back(markers[workflow]);
    }

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    CHK_PRT_RET(
        ccuRet != CCU_SUCCESS, HCCL_ERROR("CCU kernel register end failed[%d]", ccuRet), ConvertCcuToHccl(ccuRet));
    return HCCL_SUCCESS;
}

HcclResult RegisterTwelvePipelineKernels(
    HcclComm comm, const OpParam &param, const ChannelPlanResources &channels, AlgResourceCtx &resCtxHost)
{
    CHK_PRT_RET(channels.pairChannels.empty() || channels.pairChannels.size() != channels.pairPeers.size()
                    || channels.pairChannels.size() != channels.pairLayers.size(),
        HCCL_ERROR("Invalid twelve-rank channel resources"), HCCL_E_INTERNAL);

    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1, HCCL_ERROR("Unexpected CCU instance count[%u]", insNum), HCCL_E_INTERNAL);
    CcuResult ccuRet = HcommCcuKernelRegisterStart(insHandle);
    CHK_PRT_RET(
        ccuRet != CCU_SUCCESS, HCCL_ERROR("CCU kernel register start failed[%d]", ccuRet), ConvertCcuToHccl(ccuRet));

    enum class TwelveStage : uint32_t {
        LOCAL_SCATTER_A,
        REMOTE_SCATTER_D,
        LOCAL_SCATTER_B_GATHER_A,
        REMOTE_ALLGATHER_D,
        CROSS_WAVE_AD_C,
        LOCAL_ALLGATHER_B,
        REMOTE_ALLGATHER_C,
        CROSS_WAVE_BC,
    };
    struct Workflow {
        TwelveStage stage;
    };
    std::vector<Workflow> workflows;
    workflows.reserve(6);
    if (param.myRank < 8) {
        workflows.push_back(Workflow{TwelveStage::LOCAL_SCATTER_A});
        workflows.push_back(Workflow{TwelveStage::LOCAL_SCATTER_B_GATHER_A});
        workflows.push_back(Workflow{TwelveStage::LOCAL_ALLGATHER_B});
        if (param.myRank == param.root) {
            workflows.push_back(Workflow{TwelveStage::REMOTE_SCATTER_D});
        }
        workflows.push_back(Workflow{TwelveStage::CROSS_WAVE_AD_C});
        workflows.push_back(Workflow{TwelveStage::CROSS_WAVE_BC});
    } else {
        workflows.push_back(Workflow{TwelveStage::REMOTE_SCATTER_D});
        workflows.push_back(Workflow{TwelveStage::REMOTE_ALLGATHER_D});
        workflows.push_back(Workflow{TwelveStage::CROSS_WAVE_AD_C});
        workflows.push_back(Workflow{TwelveStage::REMOTE_ALLGATHER_C});
        workflows.push_back(Workflow{TwelveStage::CROSS_WAVE_BC});
    }

    std::shared_ptr<CcuTwelveKernelArg> layerArgs[2];
    for (uint32_t layer = 0; layer < 2; ++layer) {
        layerArgs[layer] = std::make_shared<CcuTwelveKernelArg>();
        layerArgs[layer]->myRank = param.myRank;
        layerArgs[layer]->root = param.root;
        for (uint32_t index = 0; index < channels.pairChannels.size(); ++index) {
            if (channels.pairLayers[index] != layer) {
                continue;
            }
            const uint32_t unionIndex = layerArgs[layer]->channelCount++;
            CHK_PRT_RET(unionIndex >= MAX_RANK_SIZE,
                HCCL_ERROR("Too many twelve-rank layer[%u] channels", layer), HCCL_E_INTERNAL);
            layerArgs[layer]->channels[unionIndex] = channels.pairChannels[index];
            layerArgs[layer]->peerRanks[unionIndex] = channels.pairPeers[index];
        }
    }

    for (uint32_t workflowIndex = 0; workflowIndex < workflows.size(); ++workflowIndex) {
        const Workflow workflow = workflows[workflowIndex];
        std::vector<uint32_t> channelIndices;
        for (uint32_t index = 0; index < channels.pairChannels.size(); ++index) {
            const uint32_t peer = channels.pairPeers[index];
            const uint32_t layer = channels.pairLayers[index];
            bool useChannel = false;
            switch (workflow.stage) {
                case TwelveStage::LOCAL_SCATTER_A:
                case TwelveStage::LOCAL_SCATTER_B_GATHER_A:
                case TwelveStage::LOCAL_ALLGATHER_B:
                    useChannel = layer == 0;
                    break;
                case TwelveStage::REMOTE_SCATTER_D:
                    useChannel = param.myRank == param.root ? layer == 1 : peer == param.root;
                    break;
                case TwelveStage::REMOTE_ALLGATHER_D:
                case TwelveStage::REMOTE_ALLGATHER_C:
                    useChannel = layer == 0;
                    break;
                case TwelveStage::CROSS_WAVE_AD_C:
                case TwelveStage::CROSS_WAVE_BC:
                    useChannel = layer == 1;
                    break;
            }
            if (useChannel) {
                channelIndices.push_back(index);
            }
        }

        uint32_t expectedChannels = 1;
        if (workflow.stage == TwelveStage::LOCAL_SCATTER_A
            || workflow.stage == TwelveStage::LOCAL_SCATTER_B_GATHER_A
            || workflow.stage == TwelveStage::LOCAL_ALLGATHER_B) {
            expectedChannels = 7;
        } else if (workflow.stage == TwelveStage::REMOTE_SCATTER_D && param.myRank == param.root) {
            expectedChannels = 4;
        } else if (workflow.stage == TwelveStage::REMOTE_ALLGATHER_D
                   || workflow.stage == TwelveStage::REMOTE_ALLGATHER_C) {
            expectedChannels = 3;
        } else if (workflow.stage == TwelveStage::CROSS_WAVE_AD_C
                   || workflow.stage == TwelveStage::CROSS_WAVE_BC) {
            expectedChannels = param.myRank < 8 ? 4U : 8U;
        }
        CHK_PRT_RET(channelIndices.size() != expectedChannels,
            HCCL_ERROR("Twelve stage[%u] rank[%u] requires[%u] channels, got[%zu]",
                static_cast<uint32_t>(workflow.stage), param.myRank, expectedChannels, channelIndices.size()),
            HCCL_E_INTERNAL);

        const uint32_t expectedLayer
            = workflow.stage == TwelveStage::LOCAL_SCATTER_A
                      || workflow.stage == TwelveStage::LOCAL_SCATTER_B_GATHER_A
                      || workflow.stage == TwelveStage::LOCAL_ALLGATHER_B
                      || workflow.stage == TwelveStage::REMOTE_ALLGATHER_D
                      || workflow.stage == TwelveStage::REMOTE_ALLGATHER_C
                  ? 0U
                  : 1U;
        for (uint32_t index : channelIndices) {
            CHK_PRT_RET(channels.pairLayers[index] != expectedLayer,
                HCCL_ERROR("Twelve stage[%u] mixes channel layers", static_cast<uint32_t>(workflow.stage)),
                HCCL_E_INTERNAL);
        }

        const uint32_t stage = static_cast<uint32_t>(workflow.stage);
        CcuTwelveKernelArg &kernelArg = *layerArgs[expectedLayer];
        kernelArg.stageChannelCount[stage] = static_cast<uint32_t>(channelIndices.size());
        for (uint32_t i = 0; i < channelIndices.size(); ++i) {
            const uint32_t channelIndex = channelIndices[i];
            auto channelIt = std::find(kernelArg.channels, kernelArg.channels + kernelArg.channelCount,
                channels.pairChannels[channelIndex]);
            CHK_PRT_RET(channelIt == kernelArg.channels + kernelArg.channelCount,
                HCCL_ERROR("Twelve stage[%u] channel is absent from layer[%u]", stage, expectedLayer),
                HCCL_E_INTERNAL);
            kernelArg.stageChannelIndices[stage][i] = static_cast<uint32_t>(channelIt - kernelArg.channels);
        }
    }

    std::vector<CcuKernelInfo> kernelInfos(2);
    for (uint32_t layer = 0; layer < 2; ++layer) {
        CHK_PRT_RET(layerArgs[layer]->channelCount == 0,
            HCCL_ERROR("Twelve-rank layer[%u] has no channels", layer), HCCL_E_INTERNAL);
        CcuKernelInfo &kernelInfo = kernelInfos[layer];
        sprintf(kernelInfo.kernelFuncName, "CcuBroadcast12_Layer_%u_Root_%u_Rank_%u",
            layer, param.root, param.myRank);
        kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuTwelveLayerKernel);
        kernelInfo.setKernelArg(layerArgs[layer]);
        const void *kernelArgs[] = {kernelInfo.kernelArg};
        CcuKernelHandle kernelHandle;
        ccuRet = HcommCcuKernelRegister(
            insHandle, 0, kernelInfo.kernelFuncName, kernelInfo.kernelFunc, kernelArgs, 1, &kernelHandle);
        CHK_PRT_RET(ccuRet != CCU_SUCCESS,
            HCCL_ERROR("CCU twelve-rank layer[%u] kernel register failed[%d]", layer, ccuRet),
            ConvertCcuToHccl(ccuRet));
        resCtxHost.ccuKernels.push_back(kernelHandle);
        resCtxHost.kernelLayers.push_back(layer);
    }

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    CHK_PRT_RET(
        ccuRet != CCU_SUCCESS, HCCL_ERROR("CCU kernel register end failed[%d]", ccuRet), ConvertCcuToHccl(ccuRet));
    return HCCL_SUCCESS;
}
HcclResult CreateChannelsForPlan(
    HcclComm comm, CommEngine engine, const OpParam &param, const BroadcastPlan &plan, ChannelPlanResources &channels)
{
    switch (plan.channelPlan) {
        case ChannelPlanType::FLAT_ROOT:
            return CreateFlatRootChannels(comm, engine, param, channels.layerChannels);
        case ChannelPlanType::FOUR_RANK_TREE:
            return CreateFourRankTreeChannels(comm, engine, param, channels);
        case ChannelPlanType::FOUR_RANK_PIPELINE:
            return CreateFourRankPipelineChannels(comm, engine, param, channels);
        case ChannelPlanType::TWELVE_PIPELINE:
            return CreateTwelvePipelineChannels(comm, engine, param, channels);
        case ChannelPlanType::SIXTEEN_FUSED:
            return CreateSixteenFusedChannels(comm, engine, param, channels);
        default:
            HCCL_ERROR("Unsupported channel plan[%u]", static_cast<uint32_t>(plan.channelPlan));
            return HCCL_E_NOT_SUPPORT;
    }
}

HcclResult RegisterKernelsForPlan(HcclComm comm, const OpParam &param, const BroadcastPlan &plan,
    const ChannelPlanResources &channels, AlgResourceCtx &resCtxHost)
{
    switch (plan.kernelPlan) {
        case KernelPlanType::LAYER_PARALLEL:
            return RegisterKernels(comm, param, channels.layerChannels, resCtxHost);
        case KernelPlanType::FOUR_RANK_TREE:
            return RegisterPairKernels(comm, param, channels, resCtxHost);
        case KernelPlanType::FOUR_RANK_PIPELINE:
            return RegisterFourRankPipelineKernel(comm, param, channels, resCtxHost);
        case KernelPlanType::SMALL_FLAT:
            return RegisterKernels(comm, param, channels.layerChannels, resCtxHost, true);
        case KernelPlanType::TWELVE_PIPELINE:
            return RegisterTwelvePipelineKernels(comm, param, channels, resCtxHost);
        case KernelPlanType::SIXTEEN_SCATTER_DOUBLING:
            return RegisterSixteenFusedKernels(comm, param, plan, channels, resCtxHost);
        default:
            HCCL_ERROR("Unsupported kernel plan[%u]", static_cast<uint32_t>(plan.kernelPlan));
            return HCCL_E_NOT_SUPPORT;
    }
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

    // 解析拓扑与数据量。
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || root >= param.rankSize,
        HCCL_ERROR("Invalid rankSize[%u] or root[%u]", param.rankSize, root), HCCL_E_PARA);

    uint64_t totalSize = 0;
    if (param.count != 0) {
        CHK_PRT_RET(param.dataType != HCCL_DATA_TYPE_FP32,
            HCCL_ERROR("Unsupported data type[%d]", static_cast<int>(param.dataType)), HCCL_E_PARA);
        constexpr uint32_t dataTypeSize = sizeof(float);
        CHK_PRT_RET(param.count > ~static_cast<uint64_t>(0) / dataTypeSize, HCCL_ERROR("Broadcast data size overflows"),
            HCCL_E_PARA);
        totalSize = param.count * dataTypeSize;
    }
    BroadcastPlan plan = SelectBroadcastPlan(param.rankSize, totalSize, param.root);
    if (param.count == 0 || param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    // Small 非 root 只有一个 layer kernel，不申请不会使用的 worker/notify。
    uint32_t executionThreadNum = plan.executionThreadNum;
    uint32_t threadNotifyNum = plan.threadNotifyNum;
    if (plan.messageSize == MessageSizeClass::SMALL && param.myRank != param.root) {
        executionThreadNum = 1;
        threadNotifyNum = 0;
    }

    // 4-rank、16-rank 继续使用版本 11 的 v15 资源；仅 8+4 Large 切到版本 15 的 v21 资源。
    constexpr uint32_t baselineResourceVersion = 15;
    const uint32_t resourceVersion
        = plan.kernelPlan == KernelPlanType::TWELVE_PIPELINE ? 21U : baselineResourceVersion;
    if (plan.kernelPlan == KernelPlanType::SIXTEEN_SCATTER_DOUBLING) {
        sprintf(param.tag, "hccl_broadcast_v18_root_%u_channel_%u_kernel_%u_thread_%u_defer_%u_peer_batch_2", root,
            static_cast<uint32_t>(plan.channelPlan), static_cast<uint32_t>(plan.kernelPlan), executionThreadNum,
            plan.sixteenDeferCrossWait);
    } else {
        sprintf(param.tag, "hccl_broadcast_v%u_root_%u_channel_%u_kernel_%u_thread_%u", resourceVersion, root,
            static_cast<uint32_t>(plan.channelPlan), static_cast<uint32_t>(plan.kernelPlan), executionThreadNum);
    }

    CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;

    // 用户 stream 绑定 CCU thread，仅并行计划申请 Thread Notify。
    const uint32_t streamNotifyNum = threadNotifyNum;
    CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, stream, streamNotifyNum, &param.cpuThread));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, ccuEngine, &ctx, &size) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = size;
    } else {
        // 首次调用构建 Thread、Channel 和 Kernel 资源。
        AlgResourceCtx resCtxHost{};

        // 只有确实存在独立并行工作流时才申请 worker thread。
        uint32_t threadNum = executionThreadNum;
        uint32_t notifyNumPerThread = threadNotifyNum;

        CHK_PRT_RET(threadNum == 0 || threadNum > resCtxHost.threads.capacity(),
            HCCL_ERROR("Invalid execution thread count[%u]", threadNum), HCCL_E_PARA);
        resCtxHost.threads.resize(threadNum);
        resCtxHost.threads[0] = param.cpuThread;
        if (threadNum > 1) {
            CHK_RET(HcclThreadAcquire(comm, ccuEngine, threadNum - 1, notifyNumPerThread, &resCtxHost.threads[1]));
        }

        if (param.rankSize > 1) {
            ChannelPlanResources channels;
            CHK_RET(CreateChannelsForPlan(comm, ccuEngine, param, plan, channels));
            CHK_RET(RegisterKernelsForPlan(comm, param, plan, channels, resCtxHost));
        }

        // 固定布局可直接复制到 Engine Context，避免热路径反序列化和堆分配。
        SmallResourceCtx smallCtx;
        void *ctxData = nullptr;
        if (plan.messageSize == MessageSizeClass::SMALL) {
            CHK_PRT_RET(resCtxHost.threads.empty() || resCtxHost.threads.size() > smallCtx.threads.size()
                            || resCtxHost.ccuKernels.empty() || resCtxHost.ccuKernels.size() > smallCtx.kernels.size()
                            || resCtxHost.kernelLayers.size() != resCtxHost.ccuKernels.size(),
                HCCL_ERROR("Invalid Small resource counts"), HCCL_E_INTERNAL);
            smallCtx.threadCount = static_cast<uint32_t>(resCtxHost.threads.size());
            smallCtx.kernelCount = static_cast<uint32_t>(resCtxHost.ccuKernels.size());
            std::copy(resCtxHost.threads.begin(), resCtxHost.threads.end(), smallCtx.threads.begin());
            std::copy(resCtxHost.ccuKernels.begin(), resCtxHost.ccuKernels.end(), smallCtx.kernels.begin());
            std::copy(resCtxHost.kernelLayers.begin(), resCtxHost.kernelLayers.end(), smallCtx.kernelLayers.begin());
            for (uint32_t index = 0; index < smallCtx.kernelCount; ++index) {
                const uint32_t layer = smallCtx.kernelLayers[index];
                CHK_PRT_RET(layer > 1, HCCL_ERROR("Invalid Small kernel layer[%u]", layer), HCCL_E_INTERNAL);
                smallCtx.layerKernels[layer] = index;
            }
            param.ctxSize = sizeof(SmallResourceCtx);
            ctxData = &smallCtx;
        } else {
            CHK_PRT_RET(resCtxHost.ccuKernels.empty()
                            || resCtxHost.ccuKernels.size() != resCtxHost.kernelLayers.size(),
                HCCL_ERROR("Invalid Large resource counts"), HCCL_E_INTERNAL);
            param.ctxSize = sizeof(AlgResourceCtx);
            ctxData = &resCtxHost;
        }
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, ccuEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, ccuEngine, param.tag, ctxData, param.ctxSize, 0));
    }

    CHK_RET(ops_hccl::ExecOp(param));
    return HCCL_SUCCESS;
}
