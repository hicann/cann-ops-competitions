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
#include <memory>
#include <set>
#include <vector>

#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>

#include "ccu_launch.h"
#include "hccl_ccu_res.h"

#include "ccu_kernel.h"
#include "common.h"
#include "custom.h"
#include "exec_op.h"
#include "hccl.h"
#include "log.h"

namespace {

constexpr uint32_t LAYER0 = 0;
constexpr uint32_t LAYER1 = 1;
constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;
constexpr bool ENABLE_PURE_CLOS_MESH = true;

struct KernelBuildInfo {
    CcuKernelInfo kernelInfo;
    BroadcastKernelRole role;
    uint32_t threadIdx;
};

HcclResult FindUbcLink(
    HcclComm comm, uint32_t layer, uint32_t localRank, uint32_t remoteRank, HcclChannelDesc &desc, bool &found)
{
    found = false;
    CommLink *links = nullptr;
    uint32_t linkNum = 0;
    CHK_RET(HcclRankGraphGetLinks(comm, layer, localRank, remoteRank, &links, &linkNum));
    for (uint32_t i = 0; i < linkNum; ++i) {
        const CommLink &link = links[i];
        if (link.linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_UBC_CTP) {
            continue;
        }
        CHK_RET(HcclChannelDescInit(&desc, 1));
        desc.remoteRank = remoteRank;
        desc.notifyNum = CHANNEL_NOTIFY_NUM;
        desc.channelProtocol = link.linkAttr.linkProtocol;
        desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
        desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
        desc.localEndpoint.loc = link.srcEndpointDesc.loc;
        desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
        desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
        desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
        found = true;
        return HCCL_SUCCESS;
    }
    return HCCL_SUCCESS;
}

HcclResult GetPeerLayerAndDesc(
    HcclComm comm, const OpParam &param, uint32_t peerRank, uint32_t &layer, HcclChannelDesc &desc)
{
    bool found = false;
    CHK_RET(FindUbcLink(comm, LAYER0, param.myRank, peerRank, desc, found));
    if (found) {
        layer = LAYER0;
        return HCCL_SUCCESS;
    }
    CHK_RET(FindUbcLink(comm, LAYER1, param.myRank, peerRank, desc, found));
    if (found) {
        layer = LAYER1;
        return HCCL_SUCCESS;
    }
    HCCL_ERROR("[Broadcast] no UBC_CTP link between rank %u and rank %u", param.myRank, peerRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult IsPureClosTopology(HcclComm comm, bool &isPureClos)
{
    uint32_t *layers = nullptr;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layers, &layerNum));
    bool hasLayer1 = false;
    for (uint32_t i = 0; i < layerNum; ++i) {
        hasLayer1 = hasLayer1 || layers[i] == LAYER1;
    }
    CHK_PRT_RET(!hasLayer1, HCCL_ERROR("[Broadcast] layer1 is required"), HCCL_E_NOT_SUPPORT);

    uint32_t *instSizes = nullptr;
    uint32_t instNum = 0;
    CHK_RET(HcclRankGraphGetInstSizeListByLayer(comm, LAYER0, &instSizes, &instNum));
    isPureClos = true;
    for (uint32_t i = 0; i < instNum; ++i) {
        if (instSizes[i] > 1) {
            isPureClos = false;
            break;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AcquirePeerChannels(HcclComm comm, const OpParam &param, const std::vector<uint32_t> &peers,
    std::vector<ChannelHandle> &handles, std::vector<uint32_t> &layers)
{
    handles.clear();
    layers.clear();
    if (peers.empty()) {
        return HCCL_SUCCESS;
    }
    std::vector<HcclChannelDesc> descs(peers.size());
    layers.resize(peers.size());
    for (uint32_t i = 0; i < peers.size(); ++i) {
        CHK_RET(GetPeerLayerAndDesc(comm, param, peers[i], layers[i], descs[i]));
    }
    handles.resize(peers.size());
    CHK_RET(HcclChannelAcquire(
        comm, CommEngine::COMM_ENGINE_CCU, descs.data(), static_cast<uint32_t>(descs.size()), handles.data()));
    return HCCL_SUCCESS;
}

void FillPeerArg(CcuKernelArgPeers &arg, const OpParam &param, const std::vector<uint32_t> &peers,
    const std::map<uint32_t, ChannelHandle> &channelByRank)
{
    arg.rankSize = param.rankSize;
    arg.rankId = param.myRank;
    arg.rootId = param.root;
    arg.channelCount = static_cast<uint32_t>(peers.size());
    for (uint32_t i = 0; i < peers.size(); ++i) {
        arg.peerRanks[i] = peers[i];
        arg.channels[i] = channelByRank.at(peers[i]);
    }
}

void AddPeerKernel(std::vector<KernelBuildInfo> &kernels, const OpParam &param, BroadcastKernelRole role,
    uint32_t threadIdx, const char *name, void *func, const std::vector<uint32_t> &peers,
    const std::map<uint32_t, ChannelHandle> &channelByRank)
{
    auto arg = std::make_shared<CcuKernelArgPeers>();
    FillPeerArg(*arg, param, peers, channelByRank);
    KernelBuildInfo build{};
    std::snprintf(build.kernelInfo.kernelFuncName, sizeof(build.kernelInfo.kernelFuncName), "%s", name);
    build.kernelInfo.kernelFunc = func;
    build.kernelInfo.setKernelArg(arg);
    build.role = role;
    build.threadIdx = threadIdx;
    kernels.push_back(build);
}

uint32_t GetNhrStepNum(uint32_t rankSize)
{
    uint32_t steps = 0;
    for (uint32_t value = rankSize - 1; value != 0; value >>= 1U) {
        ++steps;
    }
    return steps;
}

HcclResult FillNhrSteps(const OpParam &param, CcuKernelArgNhr &arg, std::set<uint32_t> &peerSet)
{
    const uint32_t rankSize = param.rankSize;
    const uint32_t rankIdx = param.myRank;
    const uint32_t rootIdx = param.root;
    const uint32_t stepNum = GetNhrStepNum(rankSize);
    CHK_PRT_RET(stepNum * 2 > BROADCAST_MAX_NHR_STEPS, HCCL_ERROR("[Broadcast] too many NHR steps %u", stepNum * 2),
        HCCL_E_NOT_SUPPORT);
    arg.scatterStepCount = stepNum;
    arg.totalStepCount = stepNum * 2;

    for (uint32_t stepIdx = 0; stepIdx < stepNum; ++stepIdx) {
        BroadcastNhrStep &step = arg.steps[stepIdx];
        const uint32_t deltaRoot = (rootIdx + rankSize - rankIdx) % rankSize;
        const uint32_t deltaRank = 1U << stepIdx;
        const uint32_t sliceCount = (rankSize - 1 + deltaRank) / (1U << (stepIdx + 1));
        const uint32_t sliceDelta = 1U << (stepIdx + 1);
        const bool isPowerOfTwo = (rankSize & (rankSize - 1)) == 0;
        const uint32_t activeRankNum = (!isPowerOfTwo && stepIdx + 1 == stepNum) ? rankSize - deltaRank : deltaRank;

        if (deltaRoot < activeRankNum) {
            const uint32_t sendTo = (rankIdx + rankSize - deltaRank) % rankSize;
            step.toRank = sendTo;
            step.txSliceCount = sliceCount;
            uint32_t sliceIdx = sendTo;
            for (uint32_t i = 0; i < sliceCount; ++i) {
                step.txSliceIdxs[i] = sliceIdx;
                sliceIdx = (sliceIdx + rankSize - sliceDelta) % rankSize;
            }
            peerSet.insert(sendTo);
        } else if (deltaRoot >= deltaRank && deltaRoot < activeRankNum + deltaRank) {
            const uint32_t recvFrom = (rankIdx + deltaRank) % rankSize;
            step.fromRank = recvFrom;
            step.rxSliceCount = sliceCount;
            peerSet.insert(recvFrom);
        }
    }

    for (uint32_t stepIdx = 0; stepIdx < stepNum; ++stepIdx) {
        BroadcastNhrStep &step = arg.steps[stepNum + stepIdx];
        const uint32_t deltaRank = 1U << (stepNum - 1 - stepIdx);
        const uint32_t recvFrom = (rankIdx + rankSize - deltaRank) % rankSize;
        const uint32_t sendTo = (rankIdx + deltaRank) % rankSize;
        const uint32_t sliceCount = (rankSize - 1 + (1U << (stepNum - 1 - stepIdx))) / (1U << (stepNum - stepIdx));
        const uint32_t sliceDelta = 1U << (stepNum - stepIdx);
        step.toRank = sendTo;
        step.fromRank = recvFrom;
        step.txSliceCount = sliceCount;
        step.rxSliceCount = sliceCount;
        uint32_t sliceIdx = rankIdx;
        for (uint32_t i = 0; i < sliceCount; ++i) {
            step.txSliceIdxs[i] = sliceIdx;
            sliceIdx = (sliceIdx + rankSize - sliceDelta) % rankSize;
        }
        peerSet.insert(sendTo);
        peerSet.insert(recvFrom);
    }
    return HCCL_SUCCESS;
}

HcclResult BuildNhrResource(HcclComm comm, const OpParam &param, std::vector<KernelBuildInfo> &kernels)
{
    auto arg = std::make_shared<CcuKernelArgNhr>();
    arg->rankSize = param.rankSize;
    arg->rankId = param.myRank;
    arg->rootId = param.root;
    std::set<uint32_t> peerSet;
    CHK_RET(FillNhrSteps(param, *arg, peerSet));
    std::vector<uint32_t> peers(peerSet.begin(), peerSet.end());
    std::vector<ChannelHandle> handles;
    std::vector<uint32_t> layers;
    CHK_RET(AcquirePeerChannels(comm, param, peers, handles, layers));
    arg->channelCount = static_cast<uint32_t>(peers.size());
    for (uint32_t i = 0; i < peers.size(); ++i) {
        CHK_PRT_RET(layers[i] != LAYER1, HCCL_ERROR("[Broadcast] NHR peer is not on layer1"), HCCL_E_INTERNAL);
        arg->peerRanks[i] = peers[i];
        arg->channels[i] = handles[i];
    }
    KernelBuildInfo build{};
    std::snprintf(
        build.kernelInfo.kernelFuncName, sizeof(build.kernelInfo.kernelFuncName), "%s", "CcuBroadcastNhrLayer1Kernel");
    build.kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuBroadcastNhrLayer1Kernel);
    build.kernelInfo.setKernelArg(arg);
    build.role = BroadcastKernelRole::NHR_LAYER1;
    build.threadIdx = 0;
    kernels.push_back(build);
    return HCCL_SUCCESS;
}

HcclResult BuildLayer1MeshResource(HcclComm comm, const OpParam &param, std::vector<KernelBuildInfo> &kernels)
{
    std::vector<uint32_t> peers;
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer != param.myRank) {
            peers.push_back(peer);
        }
    }

    std::vector<ChannelHandle> handles;
    std::vector<uint32_t> layers;
    CHK_RET(AcquirePeerChannels(comm, param, peers, handles, layers));
    std::map<uint32_t, ChannelHandle> channelByRank;
    for (uint32_t i = 0; i < peers.size(); ++i) {
        CHK_PRT_RET(
            layers[i] != LAYER1, HCCL_ERROR("[Broadcast] pure Clos Mesh peer is not on layer1"), HCCL_E_INTERNAL);
        channelByRank[peers[i]] = handles[i];
    }

    const std::vector<uint32_t> scatterPeers = param.myRank == param.root ? peers : std::vector<uint32_t>{param.root};
    AddPeerKernel(kernels, param, BroadcastKernelRole::SCATTER_LAYER1, 0, "CcuScatterLayer1Kernel",
        reinterpret_cast<void *>(ops_hccl::CcuScatterLayer1Kernel), scatterPeers, channelByRank);
    AddPeerKernel(kernels, param, BroadcastKernelRole::ALLGATHER_LAYER1, 0, "CcuAllGatherLayer1Kernel",
        reinterpret_cast<void *>(ops_hccl::CcuAllGatherLayer1Kernel), peers, channelByRank);
    return HCCL_SUCCESS;
}

HcclResult BuildMixedResource(
    HcclComm comm, const OpParam &param, bool smallData, std::vector<KernelBuildInfo> &kernels)
{
    std::vector<uint32_t> acquirePeers;
    if (smallData && param.myRank != param.root) {
        acquirePeers.push_back(param.root);
    } else {
        for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
            if (peer != param.myRank) {
                acquirePeers.push_back(peer);
            }
        }
    }
    std::vector<ChannelHandle> handles;
    std::vector<uint32_t> layers;
    CHK_RET(AcquirePeerChannels(comm, param, acquirePeers, handles, layers));
    std::map<uint32_t, ChannelHandle> channelByRank;
    std::map<uint32_t, uint32_t> layerByRank;
    for (uint32_t i = 0; i < acquirePeers.size(); ++i) {
        channelByRank[acquirePeers[i]] = handles[i];
        layerByRank[acquirePeers[i]] = layers[i];
    }

    if (smallData) {
        std::vector<uint32_t> layer0Peers;
        std::vector<uint32_t> layer1Peers;
        for (uint32_t peer : acquirePeers) {
            (layerByRank[peer] == LAYER0 ? layer0Peers : layer1Peers).push_back(peer);
        }
        uint32_t nextThread = 0;
        if (!layer0Peers.empty()) {
            AddPeerKernel(kernels, param, BroadcastKernelRole::FLAT_LAYER0, nextThread++,
                "CcuFlatBroadcastLayer0Kernel", reinterpret_cast<void *>(ops_hccl::CcuFlatBroadcastLayer0Kernel),
                layer0Peers, channelByRank);
        }
        if (!layer1Peers.empty()) {
            AddPeerKernel(kernels, param, BroadcastKernelRole::FLAT_LAYER1, nextThread, "CcuFlatBroadcastLayer1Kernel",
                reinterpret_cast<void *>(ops_hccl::CcuFlatBroadcastLayer1Kernel), layer1Peers, channelByRank);
        }
        return HCCL_SUCCESS;
    }

    std::vector<uint32_t> layer0Peers;
    std::vector<uint32_t> layer1Peers;
    for (uint32_t peer : acquirePeers) {
        (layerByRank[peer] == LAYER0 ? layer0Peers : layer1Peers).push_back(peer);
    }
    CHK_PRT_RET(layer0Peers.empty() || layer1Peers.empty(),
        HCCL_ERROR("[Broadcast] mixed topology requires non-empty layer0 and layer1 peers"), HCCL_E_INTERNAL);

    if (param.myRank == param.root) {
        AddPeerKernel(kernels, param, BroadcastKernelRole::SCATTER_LAYER0, 0, "CcuScatterLayer0Kernel",
            reinterpret_cast<void *>(ops_hccl::CcuScatterLayer0Kernel), layer0Peers, channelByRank);
        AddPeerKernel(kernels, param, BroadcastKernelRole::SCATTER_LAYER1, 1, "CcuScatterLayer1Kernel",
            reinterpret_cast<void *>(ops_hccl::CcuScatterLayer1Kernel), layer1Peers, channelByRank);
    } else {
        const uint32_t rootLayer = layerByRank[param.root];
        const BroadcastKernelRole role
            = rootLayer == LAYER0 ? BroadcastKernelRole::SCATTER_LAYER0 : BroadcastKernelRole::SCATTER_LAYER1;
        const uint32_t threadIdx = rootLayer == LAYER0 ? 0 : 1;
        const char *name = rootLayer == LAYER0 ? "CcuScatterLayer0Kernel" : "CcuScatterLayer1Kernel";
        void *func = rootLayer == LAYER0 ? reinterpret_cast<void *>(ops_hccl::CcuScatterLayer0Kernel)
                                         : reinterpret_cast<void *>(ops_hccl::CcuScatterLayer1Kernel);
        AddPeerKernel(kernels, param, role, threadIdx, name, func, {param.root}, channelByRank);
    }
    AddPeerKernel(kernels, param, BroadcastKernelRole::ALLGATHER_LAYER0, 0, "CcuAllGatherLayer0Kernel",
        reinterpret_cast<void *>(ops_hccl::CcuAllGatherLayer0Kernel), layer0Peers, channelByRank);
    AddPeerKernel(kernels, param, BroadcastKernelRole::ALLGATHER_LAYER1, 1, "CcuAllGatherLayer1Kernel",
        reinterpret_cast<void *>(ops_hccl::CcuAllGatherLayer1Kernel), layer1Peers, channelByRank);
    return HCCL_SUCCESS;
}

HcclResult RegisterKernels(HcclComm comm, const std::vector<KernelBuildInfo> &kernels, AlgResourceCtx &resCtx)
{
    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(
        insNum != 1, HCCL_ERROR("[Broadcast] unexpected CCU instruction handle count %u", insNum), HCCL_E_INTERNAL);
    CcuResult ccuRet = HcommCcuKernelRegisterStart(insHandle);
    CHK_PRT_RET(
        ccuRet != CCU_SUCCESS, HCCL_ERROR("[Broadcast] register start failed %d", ccuRet), ConvertCcuToHccl(ccuRet));

    resCtx.ccuKernels.resize(kernels.size());
    resCtx.kernelRoles.resize(kernels.size());
    resCtx.kernelThreadIdx.resize(kernels.size());
    for (uint32_t i = 0; i < kernels.size(); ++i) {
        const KernelBuildInfo &build = kernels[i];
        const void *kernelArgs[] = {build.kernelInfo.kernelArg};
        constexpr uint32_t dieId = 0;
        constexpr uint32_t kernelArgNum = 1;
        ccuRet = HcommCcuKernelRegister(insHandle, dieId, build.kernelInfo.kernelFuncName, build.kernelInfo.kernelFunc,
            kernelArgs, kernelArgNum, &resCtx.ccuKernels[i]);
        CHK_PRT_RET(ccuRet != CCU_SUCCESS,
            HCCL_ERROR("[Broadcast] register kernel %s failed %d", build.kernelInfo.kernelFuncName, ccuRet),
            ConvertCcuToHccl(ccuRet));
        resCtx.kernelRoles[i] = static_cast<uint32_t>(build.role);
        resCtx.kernelThreadIdx[i] = build.threadIdx;
    }
    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    CHK_PRT_RET(
        ccuRet != CCU_SUCCESS, HCCL_ERROR("[Broadcast] register end failed %d", ccuRet), ConvertCcuToHccl(ccuRet));
    return HCCL_SUCCESS;
}

HcclResult BuildResource(HcclComm comm, const OpParam &param, uint64_t dataSize, AlgResourceCtx &resCtx)
{
    void *cclBufferAddr = nullptr;
    uint64_t cclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    resCtx.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

    bool pureClos = false;
    CHK_RET(IsPureClosTopology(comm, pureClos));
    const bool smallData = dataSize <= BROADCAST_SMALL_DATA_SIZE;
    std::vector<KernelBuildInfo> kernels;
    if (pureClos) {
        CHK_PRT_RET(param.rankSize != 4,
            HCCL_ERROR("[Broadcast] pure Clos plan currently expects 4 ranks, got %u", param.rankSize),
            HCCL_E_NOT_SUPPORT);
        if (smallData) {
            // For the 4x1 topology, a single layer-1 flat kernel lets the root
            // write to all three peers concurrently. This avoids the four
            // ordered scatter/allgather steps of NHR for latency-bound data.
            resCtx.algorithm = BroadcastAlgorithm::FLAT_MIXED;
            CHK_RET(BuildMixedResource(comm, param, true, kernels));
        } else if (ENABLE_PURE_CLOS_MESH) {
            resCtx.algorithm = BroadcastAlgorithm::GLOBAL_MESH;
            CHK_RET(BuildLayer1MeshResource(comm, param, kernels));
        } else {
            resCtx.algorithm = BroadcastAlgorithm::NHR_LAYER1;
            CHK_RET(BuildNhrResource(comm, param, kernels));
        }
    } else {
        resCtx.algorithm = smallData ? BroadcastAlgorithm::FLAT_MIXED : BroadcastAlgorithm::GLOBAL_MESH;
        CHK_RET(BuildMixedResource(comm, param, smallData, kernels));
    }

    uint32_t threadNum = 1;
    for (const KernelBuildInfo &kernel : kernels) {
        threadNum = std::max(threadNum, kernel.threadIdx + 1);
    }
    resCtx.threads.resize(threadNum);
    resCtx.threads[0] = param.cpuThread;
    if (threadNum > 1) {
        CHK_RET(HcclThreadAcquire(comm, CommEngine::COMM_ENGINE_CCU, threadNum - 1, 1, &resCtx.threads[1]));
    }
    CHK_RET(RegisterKernels(comm, kernels, resCtx));
    HCCL_INFO("[Broadcast] algorithm %u, kernels %zu, threads %zu", static_cast<uint32_t>(resCtx.algorithm),
        resCtx.ccuKernels.size(), resCtx.threads.size());
    return HCCL_SUCCESS;
}

} // namespace

HcclResult HcclBroadcast(
    void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(buf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("[Broadcast] only float32 is supported, dataType %d", dataType), HCCL_E_NOT_SUPPORT);

    OpParam param;
    param.inputPtr = buf;
    param.outputPtr = buf;
    param.count = count;
    param.root = root;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_BROADCAST;
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE,
        HCCL_ERROR("[Broadcast] unsupported rank size %u", param.rankSize), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(root >= param.rankSize, HCCL_ERROR("[Broadcast] invalid root %u", root), HCCL_E_PARA);
    if (count == 0 || param.rankSize == 1) {
        return HCCL_SUCCESS;
    }
    constexpr uint64_t typeSize = sizeof(float);
    CHK_PRT_RET(count > std::numeric_limits<uint64_t>::max() / typeSize, HCCL_ERROR("[Broadcast] data size overflows"),
        HCCL_E_PARA);
    const uint64_t dataSize = count * typeSize;
    const char *sizeClass = dataSize <= BROADCAST_SMALL_DATA_SIZE ? "small" : "large";
    const int tagRet
        = std::snprintf(param.tag, sizeof(param.tag), "hccl_custom_broadcast_r%u_%s", param.root, sizeClass);
    CHK_PRT_RET(tagRet <= 0 || static_cast<size_t>(tagRet) >= sizeof(param.tag),
        HCCL_ERROR("[Broadcast] failed to build engine tag"), HCCL_E_INTERNAL);

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    constexpr uint32_t mainThreadNotifyNum = 1;
    CHK_RET(
        HcclThreadAcquireWithStream(comm, CommEngine::COMM_ENGINE_CCU, stream, mainThreadNotifyNum, &param.cpuThread));

    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    if (HcclEngineCtxGet(comm, param.tag, CommEngine::COMM_ENGINE_CCU, &ctx, &ctxSize) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = ctxSize;
    } else {
        AlgResourceCtx hostCtx;
        CHK_RET(BuildResource(comm, param, dataSize, hostCtx));
        std::vector<char> serialized = hostCtx.Serialize();
        param.ctxSize = serialized.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, CommEngine::COMM_ENGINE_CCU, param.ctxSize, &param.resCtx));
        CHK_RET(
            HcclEngineCtxCopy(comm, CommEngine::COMM_ENGINE_CCU, param.tag, serialized.data(), serialized.size(), 0));
    }

    return ops_hccl::ExecOp(param);
}
