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
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include <ccu/ccu_res.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>

#include "ccu_launch.h"
#include "ccu_kernel.h"
#include "hccl_ccu_res.h"
#include "common.h"
#include "custom.h"
#include "exec_op.h"
#include "hccl.h"
#include "log.h"

namespace {

constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;
constexpr uint64_t SMALL_512_DATA_SIZE = 512ULL * 1024;
constexpr uint64_t SMALL_512_FP32_COUNT = SMALL_512_DATA_SIZE / sizeof(float);
constexpr uint32_t SMALL_512_FUSED_LAYER = 2;
constexpr bool ENABLE_8PLUS4_SMALL512_FAST = false;

struct PeerChannel {
    uint32_t remoteRank;
    uint32_t netLayer;
    ChannelHandle handle;
};

struct LayerChannelGroup {
    uint32_t netLayer;
    std::vector<PeerChannel> layerChannels;
};

struct Small512ThreadCache {
    HcclComm comm = nullptr;
    aclrtStream stream = nullptr;
    Small512FastResourceCtx *ctx = nullptr;
};

thread_local Small512ThreadCache g_small512ThreadCache;

HcclResult FillChannelDesc(
    HcclComm comm, uint32_t srcRank, uint32_t dstRank, HcclChannelDesc &desc, uint32_t &selectedLayer)
{
    CHK_RET(HcclChannelDescInit(&desc, 1));

    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    for (uint32_t layerIdx = 0; layerIdx < netLayerNum; ++layerIdx) {
        CommLink *linkList = nullptr;
        uint32_t listSize = 0;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayers[layerIdx], srcRank, dstRank, &linkList, &listSize));
        for (uint32_t linkIdx = 0; linkIdx < listSize; ++linkIdx) {
            const CommLink &link = linkList[linkIdx];
            if (link.linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                continue;
            }

            desc.remoteRank = dstRank;
            desc.notifyNum = CHANNEL_NOTIFY_NUM;
            desc.channelProtocol = link.linkAttr.linkProtocol;
            desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
            desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
            desc.localEndpoint.loc = link.srcEndpointDesc.loc;
            desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
            desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
            desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
            selectedLayer = netLayers[layerIdx];
            return HCCL_SUCCESS;
        }
    }

    HCCL_ERROR("[FillChannelDesc] no UBC_CTP link found from rank %u to rank %u", srcRank, dstRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult AcquirePeerChannels(HcclComm comm, const OpParam &param, std::vector<PeerChannel> &peerChannels)
{
    const uint32_t channelNum = param.rankSize - 1;
    std::vector<HcclChannelDesc> channelDescs(channelNum);
    std::vector<uint32_t> remoteRanks(channelNum);
    std::vector<uint32_t> netLayers(channelNum);

    uint32_t channelIdx = 0;
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }
        CHK_RET(FillChannelDesc(comm, param.myRank, remoteRank, channelDescs[channelIdx], netLayers[channelIdx]));
        remoteRanks[channelIdx] = remoteRank;
        ++channelIdx;
    }

    std::vector<ChannelHandle> channelHandles(channelNum);
    CHK_RET(
        HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, channelDescs.data(), channelNum, channelHandles.data()));

    peerChannels.reserve(channelNum);
    for (uint32_t idx = 0; idx < channelNum; ++idx) {
        peerChannels.push_back(PeerChannel{remoteRanks[idx], netLayers[idx], channelHandles[idx]});
    }
    return HCCL_SUCCESS;
}

const char *GetTopologyName(AllGatherTopology topology)
{
    switch (topology) {
        case AllGatherTopology::TWO_SERVER_EIGHT_NPU:
            return "2x8";
        case AllGatherTopology::FOUR_SERVER_ONE_NPU:
            return "4x1";
        case AllGatherTopology::EIGHT_PLUS_FOUR:
            return "8plus4";
        default:
            return "reserved";
    }
}

void *GetKernelFunc(AllGatherTopology topology, AllGatherSizeClass sizeClass)
{
    const bool isSmall = sizeClass == AllGatherSizeClass::SMALL;
    switch (topology) {
        case AllGatherTopology::TWO_SERVER_EIGHT_NPU:
            return isSmall ? reinterpret_cast<void *>(ops_hccl::topo_2x8::CcuAllGatherSmallKernel)
                           : reinterpret_cast<void *>(ops_hccl::topo_2x8::CcuAllGatherLargeKernel);
        case AllGatherTopology::FOUR_SERVER_ONE_NPU:
            return isSmall ? reinterpret_cast<void *>(ops_hccl::topo_4x1::CcuAllGatherSmallKernel)
                           : reinterpret_cast<void *>(ops_hccl::topo_4x1::CcuAllGatherLargeKernel);
        case AllGatherTopology::EIGHT_PLUS_FOUR:
            return isSmall ? reinterpret_cast<void *>(ops_hccl::topo_8plus4::CcuAllGatherSmallKernel)
                           : reinterpret_cast<void *>(ops_hccl::topo_8plus4::CcuAllGatherLargeKernel);
        default:
            return nullptr;
    }
}

HcclResult FillKernelInfo(const OpParam &param, AllGatherTopology topology, AllGatherSizeClass sizeClass,
    const LayerChannelGroup &layerGroup, bool handleSelfRank, CcuKernelInfo &kernelInfo)
{
    const char *sizeName = sizeClass == AllGatherSizeClass::SMALL ? "Small" : "Large";
    const int nameLen = std::snprintf(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName),
        "CcuAllGather%s%sL%uKernel", sizeName, GetTopologyName(topology), layerGroup.netLayer);
    if (nameLen <= 0 || static_cast<size_t>(nameLen) >= sizeof(kernelInfo.kernelFuncName)) {
        HCCL_ERROR("[FillKernelInfo] failed to create kernel name");
        return HCCL_E_INTERNAL;
    }

    kernelInfo.kernelFunc = GetKernelFunc(topology, sizeClass);
    if (kernelInfo.kernelFunc == nullptr) {
        HCCL_ERROR("[FillKernelInfo] no kernel function for topology %u and size class %u",
            static_cast<uint32_t>(topology), static_cast<uint32_t>(sizeClass));
        return HCCL_E_NOT_SUPPORT;
    }

    auto kernelArg = std::make_shared<ops_hccl::CcuKernelArgAllGather>();
    kernelArg->rankSize = param.rankSize;
    kernelArg->rankId = param.myRank;
    kernelArg->netLayer = layerGroup.netLayer;
    kernelArg->handleSelfRank = handleSelfRank ? 1U : 0U;
    kernelArg->small512FastMode = 0;
    kernelArg->topology = topology;
    kernelArg->sizeClass = sizeClass;

    kernelArg->channelCount = static_cast<uint32_t>(layerGroup.layerChannels.size());
    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        kernelArg->channels[idx] = layerGroup.layerChannels[idx].handle;
        kernelArg->remoteRanks[idx] = layerGroup.layerChannels[idx].remoteRank;
    }
    kernelInfo.setKernelArg(kernelArg);
    return HCCL_SUCCESS;
}

std::vector<LayerChannelGroup> GroupChannelsByLayer(const std::vector<PeerChannel> &peerChannels)
{
    std::map<uint32_t, std::vector<PeerChannel>> channelsByLayer;
    for (const PeerChannel &peerChannel : peerChannels) {
        channelsByLayer[peerChannel.netLayer].push_back(peerChannel);
    }

    std::vector<LayerChannelGroup> layerGroups;
    layerGroups.reserve(channelsByLayer.size());
    for (auto &entry : channelsByLayer) {
        layerGroups.push_back(LayerChannelGroup{entry.first, std::move(entry.second)});
    }
    return layerGroups;
}

std::vector<PeerChannel> BuildSmall512FastChannelOrder(
    const OpParam &param, const std::vector<PeerChannel> &peerChannels)
{
    if (param.rankSize == 4) {
        std::vector<PeerChannel> ordered;
        ordered.reserve(peerChannels.size());
        const uint32_t channelCount = static_cast<uint32_t>(peerChannels.size());
        const uint32_t start = param.myRank % channelCount;
        for (uint32_t step = 0; step < channelCount; ++step) {
            ordered.push_back(peerChannels[(start + step) % channelCount]);
        }
        return ordered;
    }

    std::vector<PeerChannel> meshChannels;
    std::vector<PeerChannel> closChannels;
    meshChannels.reserve(7);
    closChannels.reserve(8);
    const bool isServer0 = param.myRank < 8;
    for (const PeerChannel &peer : peerChannels) {
        const bool peerServer0 = peer.remoteRank < 8;
        (peerServer0 == isServer0 ? meshChannels : closChannels).push_back(peer);
    }

    const uint32_t meshStart =
        (isServer0 ? param.myRank : param.myRank - 8) % static_cast<uint32_t>(meshChannels.size());
    const uint32_t closStart = isServer0
        ? param.myRank % static_cast<uint32_t>(closChannels.size())
        : 2 * (param.myRank - 8) % static_cast<uint32_t>(closChannels.size());
    const uint32_t meshCount = static_cast<uint32_t>(meshChannels.size());
    const uint32_t closCount = static_cast<uint32_t>(closChannels.size());

    std::vector<PeerChannel> ordered;
    ordered.reserve(peerChannels.size());
    const uint32_t maxLayerChannels = std::max(meshCount, closCount);
    for (uint32_t step = 0; step < maxLayerChannels; ++step) {
        if (step < closCount) {
            ordered.push_back(closChannels[(closStart + step) % closCount]);
        }
        if (step < meshCount) {
            ordered.push_back(meshChannels[(meshStart + step) % meshCount]);
        }
    }
    return ordered;
}

HcclResult FillSmall512FastKernelInfo(const OpParam &param, AllGatherTopology topology,
    const std::vector<PeerChannel> &orderedChannels, CcuKernelInfo &kernelInfo)
{
    const int nameLen = std::snprintf(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName),
        "CcuAllGatherSmall512Fast%sKernel", GetTopologyName(topology));
    if (nameLen <= 0 || static_cast<size_t>(nameLen) >= sizeof(kernelInfo.kernelFuncName)) {
        HCCL_ERROR("[FillSmall512FastKernelInfo] failed to create kernel name");
        return HCCL_E_INTERNAL;
    }

    kernelInfo.kernelFunc = GetKernelFunc(topology, AllGatherSizeClass::SMALL);
    auto kernelArg = std::make_shared<ops_hccl::CcuKernelArgAllGather>();
    kernelArg->rankSize = param.rankSize;
    kernelArg->rankId = param.myRank;
    kernelArg->netLayer = SMALL_512_FUSED_LAYER;
    kernelArg->handleSelfRank = 1;
    kernelArg->small512FastMode = SMALL_512_FAST_MODE;
    kernelArg->topology = topology;
    kernelArg->sizeClass = AllGatherSizeClass::SMALL;
    kernelArg->channelCount = static_cast<uint32_t>(orderedChannels.size());
    for (uint32_t idx = 0; idx < kernelArg->channelCount; ++idx) {
        kernelArg->channels[idx] = orderedChannels[idx].handle;
        kernelArg->remoteRanks[idx] = orderedChannels[idx].remoteRank;
    }
    kernelInfo.setKernelArg(kernelArg);
    return HCCL_SUCCESS;
}

HcclResult RegisterSmall512FastKernels(HcclComm comm, const OpParam &param,
    AllGatherTopology topology, const std::vector<PeerChannel> &peerChannels,
    Small512FastResourceCtx &fastCtx)
{
    const std::vector<PeerChannel> orderedChannels =
        BuildSmall512FastChannelOrder(param, peerChannels);
    CHK_PRT_RET(orderedChannels.size() != param.rankSize - 1,
        HCCL_ERROR("[RegisterSmall512FastKernels] expected %u channels, got %zu",
            param.rankSize - 1, orderedChannels.size()),
        HCCL_E_INTERNAL);

    CcuKernelInfo kernelInfo;
    CHK_RET(FillSmall512FastKernelInfo(
        param, topology, orderedChannels, kernelInfo));

    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(
        insNum != 1, HCCL_ERROR("[RegisterSmall512FastKernels] expected one CCU instance, got %u", insNum),
        HCCL_E_INTERNAL);

    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));
    constexpr uint32_t dieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    const void *kernelArgs[] = {kernelInfo.kernelArg};
    CHK_RET_CCU(HcommCcuKernelRegister(insHandle, dieId, kernelInfo.kernelFuncName,
        kernelInfo.kernelFunc, kernelArgs, kernelArgNum, &fastCtx.kernels[0]));
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
    return HCCL_SUCCESS;
}

HcclResult GetOrCreateSmall512FastResource(HcclComm comm, aclrtStream stream,
    const OpParam &param, AllGatherTopology topology, Small512FastResourceCtx *&fastCtx)
{
    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    if (HcclEngineCtxGet(comm, param.tag, CommEngine::COMM_ENGINE_CCU, &ctx, &ctxSize) == HCCL_SUCCESS) {
        CHK_PRT_RET(ctxSize != sizeof(Small512FastResourceCtx),
            HCCL_ERROR("[GetOrCreateSmall512FastResource] invalid context size %llu",
                static_cast<unsigned long long>(ctxSize)),
            HCCL_E_INTERNAL);
        fastCtx = static_cast<Small512FastResourceCtx *>(ctx);
        CHK_PRT_RET(fastCtx->magic != SMALL_512_FAST_CTX_MAGIC,
            HCCL_ERROR("[GetOrCreateSmall512FastResource] invalid context magic"),
            HCCL_E_INTERNAL);
        return HCCL_SUCCESS;
    }

    Small512FastResourceCtx hostCtx;
    hostCtx.rankId = param.myRank;
    hostCtx.rankSize = param.rankSize;
    CHK_RET(HcclThreadAcquireWithStream(
        comm, CommEngine::COMM_ENGINE_CCU, stream, 0, &hostCtx.thread));

    std::vector<PeerChannel> peerChannels;
    CHK_RET(AcquirePeerChannels(comm, param, peerChannels));
    CHK_RET(RegisterSmall512FastKernels(
        comm, param, topology, peerChannels, hostCtx));

    CHK_RET(HcclEngineCtxCreate(comm, param.tag, CommEngine::COMM_ENGINE_CCU,
        sizeof(Small512FastResourceCtx), &ctx));
    fastCtx = static_cast<Small512FastResourceCtx *>(ctx);
    *fastCtx = hostCtx;
    return HCCL_SUCCESS;
}

HcclResult AcquireLayerThreads(HcclComm comm, ThreadHandle mainThread, uint32_t layerCount, AlgResourceCtx &resCtxHost)
{
    if (layerCount == 0 || layerCount > 2) {
        HCCL_ERROR("[AcquireLayerThreads] unsupported layer count %u", layerCount);
        return HCCL_E_NOT_SUPPORT;
    }

    resCtxHost.ccuThread = mainThread;
    resCtxHost.threads.push_back(mainThread);
    if (layerCount == 1) {
        return HCCL_SUCCESS;
    }

    const uint32_t slaveThreadNum = layerCount - 1;
    std::vector<ThreadHandle> slaveThreads(slaveThreadNum);
    constexpr uint32_t notifyNumPerSlaveThread = 1;
    CHK_RET(HcclThreadAcquire(
        comm, CommEngine::COMM_ENGINE_CCU, slaveThreadNum, notifyNumPerSlaveThread, slaveThreads.data()));
    resCtxHost.threads.insert(resCtxHost.threads.end(), slaveThreads.begin(), slaveThreads.end());
    return HCCL_SUCCESS;
}

HcclResult RegisterCcuKernels(HcclComm comm, const OpParam &param, AllGatherTopology topology,
    AllGatherSizeClass sizeClass, const std::vector<LayerChannelGroup> &layerGroups, AlgResourceCtx &resCtxHost)
{
    if (layerGroups.empty() || layerGroups.size() != resCtxHost.threads.size()) {
        HCCL_ERROR("[RegisterCcuKernels] layer count %zu does not match thread count %zu", layerGroups.size(),
            resCtxHost.threads.size());
        return HCCL_E_INTERNAL;
    }

    // Keep the local copy in the Mesh kernel so that it overlaps with the
    // fan-out writes on layer-0. A topology without layer-0 (4 x 1) falls
    // back to its only layer.
    const auto selfGroupIt = std::find_if(
        layerGroups.begin(), layerGroups.end(), [](const LayerChannelGroup &group) { return group.netLayer == 0; });
    const uint32_t selfGroupIndex = selfGroupIt == layerGroups.end()
        ? 0U
        : static_cast<uint32_t>(std::distance(layerGroups.begin(), selfGroupIt));

    std::vector<AllGatherSizeClass> sizeClasses = {AllGatherSizeClass::SMALL, AllGatherSizeClass::LARGE};
    if (topology == AllGatherTopology::TWO_SERVER_EIGHT_NPU) {
        sizeClasses = {sizeClass};
    }
    std::vector<CcuKernelInfo> kernelInfos;
    kernelInfos.reserve(sizeClasses.size() * layerGroups.size());
    resCtxHost.ccuKernelMetas.clear();
    resCtxHost.ccuKernelMetas.reserve(sizeClasses.size() * layerGroups.size());
    for (AllGatherSizeClass sizeClass : sizeClasses) {
        for (uint32_t groupIdx = 0; groupIdx < layerGroups.size(); ++groupIdx) {
            kernelInfos.emplace_back();
            CHK_RET(FillKernelInfo(
                param, topology, sizeClass, layerGroups[groupIdx], groupIdx == selfGroupIndex, kernelInfos.back()));
            resCtxHost.ccuKernelMetas.push_back(
                AllGatherKernelMeta{layerGroups[groupIdx].netLayer, sizeClass, groupIdx});
        }
    }

    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(
        insNum != 1, HCCL_ERROR("[RegisterCcuKernels] expected one CCU instance, got %u", insNum), HCCL_E_INTERNAL);

    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));
    resCtxHost.ccuKernels.resize(kernelInfos.size());
    constexpr uint32_t dieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    for (uint32_t idx = 0; idx < kernelInfos.size(); ++idx) {
        const void *kernelArgs[] = {kernelInfos[idx].kernelArg};
        CHK_RET_CCU(HcommCcuKernelRegister(insHandle, dieId, kernelInfos[idx].kernelFuncName,
            kernelInfos[idx].kernelFunc, kernelArgs, kernelArgNum, &resCtxHost.ccuKernels[idx]));
    }
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
    return HCCL_SUCCESS;
}

HcclResult ValidateAllGatherParam(
    uint64_t sendCount, HcclDataType dataType, const OpParam &param, AllGatherRoute &route)
{
    if (dataType != HCCL_DATA_TYPE_FP32) {
        HCCL_ERROR("[ValidateAllGatherParam] only float32 is supported, dataType=%d", static_cast<int32_t>(dataType));
        return HCCL_E_NOT_SUPPORT;
    }
    constexpr uint64_t dataTypeSize = sizeof(float);
    if (sendCount > std::numeric_limits<uint64_t>::max() / dataTypeSize) {
        HCCL_ERROR("[ValidateAllGatherParam] sendCount overflows byte size");
        return HCCL_E_PARA;
    }
    const uint64_t rankDataSize = sendCount * dataTypeSize;
    if (param.rankSize == 0 || rankDataSize > std::numeric_limits<uint64_t>::max() / param.rankSize) {
        HCCL_ERROR("[ValidateAllGatherParam] output size overflows");
        return HCCL_E_PARA;
    }

    return ops_hccl::SelectAllGatherRoute(rankDataSize, param.rankSize, route);
}

} // namespace

HcclResult HcclAllGather(
    void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    const bool isSmall512 =
        dataType == HCCL_DATA_TYPE_FP32 && sendCount == SMALL_512_FP32_COUNT;
    if (ENABLE_8PLUS4_SMALL512_FAST && isSmall512 && g_small512ThreadCache.comm == comm
        && g_small512ThreadCache.stream == stream && g_small512ThreadCache.ctx != nullptr
        && g_small512ThreadCache.ctx->magic == SMALL_512_FAST_CTX_MAGIC
        && g_small512ThreadCache.ctx->rankSize == 12) {
        return ops_hccl::ExecSmall512Fast(
            sendBuf, recvBuf, g_small512ThreadCache.ctx, sizeof(Small512FastResourceCtx));
    }

    OpParam param;
    const int tagLen = std::snprintf(param.tag, sizeof(param.tag), "%s", "hccl_custom_allgather");
    if (tagLen <= 0 || static_cast<size_t>(tagLen) >= sizeof(param.tag)) {
        return HCCL_E_INTERNAL;
    }
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = sendCount;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    AllGatherRoute resourceRoute;
    CHK_RET(ValidateAllGatherParam(sendCount, dataType, param, resourceRoute));
    CHK_PRT_RET(param.myRank >= param.rankSize,
        HCCL_ERROR("[HcclAllGather] rank %u outside rank size %u", param.myRank, param.rankSize), HCCL_E_PARA);

    if (ENABLE_8PLUS4_SMALL512_FAST && isSmall512 && param.rankSize == 12) {
        const int fastTagLen = std::snprintf(param.tag, sizeof(param.tag),
            "hccl_custom_allgather_512_fast_%u_%p", param.rankSize, static_cast<void *>(stream));
        if (fastTagLen <= 0 || static_cast<size_t>(fastTagLen) >= sizeof(param.tag)) {
            return HCCL_E_INTERNAL;
        }

        Small512FastResourceCtx *fastCtx = nullptr;
        CHK_RET(GetOrCreateSmall512FastResource(
            comm, stream, param, resourceRoute.topology, fastCtx));
        g_small512ThreadCache = Small512ThreadCache{comm, stream, fastCtx};
        return ops_hccl::ExecSmall512Fast(
            sendBuf, recvBuf, fastCtx, sizeof(Small512FastResourceCtx));
    }

    if (resourceRoute.topology == AllGatherTopology::TWO_SERVER_EIGHT_NPU) {
        const char *sizeTag = resourceRoute.sizeClass == AllGatherSizeClass::SMALL
            ? "hccl_custom_allgather_2x8_small"
            : "hccl_custom_allgather_2x8_large";
        const int sizeTagLen = std::snprintf(param.tag, sizeof(param.tag), "%s", sizeTag);
        if (sizeTagLen <= 0 || static_cast<size_t>(sizeTagLen) >= sizeof(param.tag)) {
            return HCCL_E_INTERNAL;
        }
    }

    const uint32_t expectedLayerCount = resourceRoute.topology == AllGatherTopology::FOUR_SERVER_ONE_NPU ? 1U : 2U;
    const uint32_t notifyNumOnMainThread = expectedLayerCount - 1;
    CHK_RET(HcclThreadAcquireWithStream(
        comm, CommEngine::COMM_ENGINE_CCU, stream, notifyNumOnMainThread, &param.cpuThread));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, CommEngine::COMM_ENGINE_CCU, &ctx, &size) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = size;
    } else {
        AlgResourceCtx resCtxHost;

        void *cclBufferAddr = nullptr;
        uint64_t cclBufferSize = 0;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        std::vector<PeerChannel> peerChannels;
        CHK_RET(AcquirePeerChannels(comm, param, peerChannels));
        std::vector<LayerChannelGroup> layerGroups = GroupChannelsByLayer(peerChannels);
        CHK_PRT_RET(layerGroups.size() != expectedLayerCount,
            HCCL_ERROR(
                "[HcclAllGather] expected %u net layers, discovered %zu", expectedLayerCount, layerGroups.size()),
            HCCL_E_INTERNAL);
        CHK_RET(AcquireLayerThreads(comm, param.cpuThread, static_cast<uint32_t>(layerGroups.size()), resCtxHost));
        CHK_RET(RegisterCcuKernels(
            comm, param, resourceRoute.topology, resourceRoute.sizeClass, layerGroups, resCtxHost));

        std::vector<char> seq = resCtxHost.Serialize();
        param.ctxSize = seq.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, CommEngine::COMM_ENGINE_CCU, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, CommEngine::COMM_ENGINE_CCU, param.tag, seq.data(), seq.size(), 0));
    }

    CHK_RET(ops_hccl::ExecOp(param));
    return HCCL_SUCCESS;
}
