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
#include <iterator>
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
constexpr uint32_t CCU_CHANNEL_NOTIFY_NUM = 8;
constexpr uint32_t CHANNEL_SEARCH_PASS_COUNT = 3;
constexpr uint64_t SMALL_MESSAGE_BYTES = 512ULL * 1024;
constexpr char BROADCAST_TAG[] = "hccl_custom_broadcast_ccu_aggressive_flat_v3";

struct ChannelCandidate {
    CommLink link;
    uint32_t layer = 0;
    CommTopo topo = COMM_TOPO_RESERVED;
    uint32_t searchPass = 0;
    EndpointAttrDieId dieId = 0;
};

struct KernelSpec {
    std::shared_ptr<CcuKernelArgBroadcast> arg;
    std::vector<uint32_t> peers;
    std::vector<uint32_t> logicalPeers;
    uint32_t die = 0;
};

bool IsSameServer(uint32_t rankSize, uint32_t lhs, uint32_t rhs)
{
    if (rankSize == 16) {
        return lhs / 8 == rhs / 8;
    }
    if (rankSize == 12) {
        return (lhs < 8 && rhs < 8) || (lhs >= 8 && rhs >= 8);
    }
    return false;
}

uint32_t ChannelSearchPass(bool preferClos, uint32_t layer, CommTopo topo)
{
    const bool isClos = topo == COMM_TOPO_CLOS;
    const bool isIntra = layer == 0 && !isClos;
    if (preferClos) {
        return isClos ? 0 : (isIntra ? 1 : 2);
    }
    return isIntra ? 0 : (isClos ? 1 : 2);
}

HcclResult FillChannelDesc(uint32_t remoteRank, const CommLink &link, HcclChannelDesc &desc)
{
    CHK_RET(HcclChannelDescInit(&desc, 1));
    desc.remoteRank = remoteRank;
    desc.notifyNum = CCU_CHANNEL_NOTIFY_NUM;
    desc.channelProtocol = link.linkAttr.linkProtocol;
    desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
    desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
    desc.localEndpoint.loc = link.srcEndpointDesc.loc;
    desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
    desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
    desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
    return HCCL_SUCCESS;
}

HcclResult FindChannelCandidates(HcclComm comm, uint32_t rank, uint32_t rankSize, uint32_t remoteRank,
    std::vector<ChannelCandidate> &candidates)
{
    uint32_t *layers = nullptr;
    uint32_t layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layers, &layerCount));
    const bool preferClos = !IsSameServer(rankSize, rank, remoteRank);
    constexpr CommProtocol protocols[] = {COMM_PROTOCOL_UBC_CTP, COMM_PROTOCOL_UBC_TP};
    for (uint32_t pass = 0; pass < CHANNEL_SEARCH_PASS_COUNT; ++pass) {
        for (uint32_t layerIndex = 0; layerIndex < layerCount; ++layerIndex) {
            CommTopo topo = COMM_TOPO_RESERVED;
            CHK_RET(HcclRankGraphGetTopoTypeByLayer(comm, layers[layerIndex], &topo));
            if (ChannelSearchPass(preferClos, layers[layerIndex], topo) != pass) {
                continue;
            }
            CommLink *links = nullptr;
            uint32_t linkCount = 0;
            CHK_RET(HcclRankGraphGetLinks(comm, layers[layerIndex], rank, remoteRank, &links, &linkCount));
            for (CommProtocol protocol : protocols) {
                for (uint32_t linkIndex = 0; linkIndex < linkCount; ++linkIndex) {
                    if (links[linkIndex].linkAttr.linkProtocol != protocol) {
                        continue;
                    }
                    EndpointAttrDieId dieId = 0;
                    CHK_RET(HcclRankGraphGetEndpointInfo(comm, rank, &links[linkIndex].srcEndpointDesc,
                        ENDPOINT_ATTR_DIE_ID, sizeof(dieId), &dieId));
                    candidates.push_back({links[linkIndex], layers[layerIndex], topo, pass, dieId});
                }
            }
        }
    }
    CHK_PRT_RET(candidates.empty(), HCCL_ERROR("No topology-aware CCU channel between rank[%u] and rank[%u]",
        rank, remoteRank), HCCL_E_NOT_FOUND);
    return HCCL_SUCCESS;
}

bool IsBetterPassCount(const uint32_t (&candidate)[CHANNEL_SEARCH_PASS_COUNT],
    const uint32_t (&current)[CHANNEL_SEARCH_PASS_COUNT])
{
    for (uint32_t pass = 0; pass < CHANNEL_SEARCH_PASS_COUNT; ++pass) {
        if (candidate[pass] != current[pass]) {
            return candidate[pass] > current[pass];
        }
    }
    return false;
}

HcclResult SelectChannelCandidates(const std::vector<std::vector<ChannelCandidate>> &candidatesByPeer,
    std::vector<uint32_t> &selectedIndices, EndpointAttrDieId &selectedDie)
{
    std::vector<EndpointAttrDieId> candidateDies;
    for (const auto &candidates : candidatesByPeer) {
        for (const auto &candidate : candidates) {
            if (std::find(candidateDies.begin(), candidateDies.end(), candidate.dieId) == candidateDies.end()) {
                candidateDies.push_back(candidate.dieId);
            }
        }
    }

    bool found = false;
    uint32_t bestPassCount[CHANNEL_SEARCH_PASS_COUNT] = {};
    for (EndpointAttrDieId dieId : candidateDies) {
        std::vector<uint32_t> indices;
        uint32_t passCount[CHANNEL_SEARCH_PASS_COUNT] = {};
        bool coversAllPeers = true;
        for (const auto &candidates : candidatesByPeer) {
            const auto candidateIt = std::find_if(candidates.begin(), candidates.end(),
                [dieId](const ChannelCandidate &candidate) { return candidate.dieId == dieId; });
            if (candidateIt == candidates.end()) {
                coversAllPeers = false;
                break;
            }
            indices.push_back(static_cast<uint32_t>(candidateIt - candidates.begin()));
            ++passCount[candidateIt->searchPass];
        }
        if (coversAllPeers && (!found || IsBetterPassCount(passCount, bestPassCount))) {
            found = true;
            selectedDie = dieId;
            selectedIndices = std::move(indices);
            std::copy(std::begin(passCount), std::end(passCount), std::begin(bestPassCount));
        }
    }

    if (!found) {
        selectedIndices.clear();
        for (const auto &candidates : candidatesByPeer) {
            CHK_PRT_RET(candidates.empty(), HCCL_ERROR("No CCU channel candidate for global peer"), HCCL_E_NOT_FOUND);
            selectedIndices.push_back(0);
        }
        selectedDie = 0;
    }
    return HCCL_SUCCESS;
}

HcclResult AcquireChannels(HcclComm comm, uint32_t rank, uint32_t rankSize, KernelSpec &spec)
{
    CHK_PRT_RET(!spec.logicalPeers.empty() && spec.logicalPeers.size() != spec.peers.size(),
        HCCL_ERROR("Broadcast global/logical peer mapping size mismatch"), HCCL_E_INTERNAL);
    std::vector<std::vector<ChannelCandidate>> candidatesByPeer(spec.peers.size());
    for (uint32_t index = 0; index < spec.peers.size(); ++index) {
        CHK_RET(FindChannelCandidates(comm, rank, rankSize, spec.peers[index], candidatesByPeer[index]));
    }
    std::vector<uint32_t> selectedIndices;
    EndpointAttrDieId selectedDie = 0;
    CHK_RET(SelectChannelCandidates(candidatesByPeer, selectedIndices, selectedDie));

    std::vector<HcclChannelDesc> descs(spec.peers.size());
    for (uint32_t index = 0; index < spec.peers.size(); ++index) {
        const ChannelCandidate &candidate = candidatesByPeer[index][selectedIndices[index]];
        HCCL_INFO("Select topology-aware channel rank[%u] -> rank[%u], layer[%u], topo[%u], die[%u]",
            rank, spec.peers[index], candidate.layer, candidate.topo, candidate.dieId);
        CHK_RET(FillChannelDesc(spec.peers[index], candidate.link, descs[index]));
    }
    spec.die = selectedDie;
    std::vector<ChannelHandle> handles(spec.peers.size());
    CHK_RET(HcclChannelAcquire(
        comm, CommEngine::COMM_ENGINE_CCU, descs.data(), static_cast<uint32_t>(descs.size()), handles.data()));
    spec.arg->channelCount = static_cast<uint32_t>(handles.size());
    for (uint32_t i = 0; i < handles.size(); ++i) {
        spec.arg->channels[i] = handles[i];
        spec.arg->peerRanks[i] = spec.logicalPeers.empty() ? spec.peers[i] : spec.logicalPeers[i];
    }
    return HCCL_SUCCESS;
}

std::shared_ptr<CcuKernelArgBroadcast> MakeGroupArg(uint32_t rank, uint32_t rankSize, uint32_t root,
    CcuBroadcastPlanKind kind, uint32_t groupRank, uint32_t groupSize, uint32_t groupRoot)
{
    auto arg = std::make_shared<CcuKernelArgBroadcast>();
    arg->rankId = rank;
    arg->rankSize = rankSize;
    arg->root = root;
    arg->planKind = static_cast<uint32_t>(kind);
    arg->groupRank = groupRank;
    arg->groupSize = groupSize;
    arg->groupRoot = groupRoot;
    return arg;
}

KernelSpec MakeGroupSpec(uint32_t rank, uint32_t rankSize, uint32_t root, CcuBroadcastPlanKind kind,
    uint32_t groupBegin, uint32_t groupEnd, uint32_t groupRootRank)
{
    const uint32_t groupRank = rank - groupBegin;
    const uint32_t groupSize = groupEnd - groupBegin;
    KernelSpec spec;
    spec.arg = MakeGroupArg(rank, rankSize, root, kind, groupRank, groupSize, groupRootRank - groupBegin);
    spec.peers.reserve(groupSize - 1);
    spec.logicalPeers.reserve(groupSize - 1);
    for (uint32_t peer = groupBegin; peer < groupEnd; ++peer) {
        if (peer != rank) {
            spec.peers.push_back(peer);
            spec.logicalPeers.push_back(peer - groupBegin);
        }
    }
    return spec;
}

HcclResult BuildKernelSpecs(
    uint32_t rank, uint32_t rankSize, uint32_t root, bool small, std::vector<KernelSpec> &specs)
{
    CHK_PRT_RET(rankSize != 4 && rankSize != 12 && rankSize != 16,
        HCCL_ERROR("Unsupported broadcast rankSize[%u]", rankSize), HCCL_E_NOT_SUPPORT);
    specs.push_back(MakeGroupSpec(rank, rankSize, root,
        small ? CcuBroadcastPlanKind::DIRECT : CcuBroadcastPlanKind::ROOT_EXCLUDED_SCATTER_GATHER,
        0, rankSize, root));
    return HCCL_SUCCESS;
}

HcclResult RegisterKernels(HcclComm comm, const std::vector<KernelSpec> &specs, AlgResourceCtx &resCtx)
{
    CcuInsHandle insHandle = 0;
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1, HCCL_ERROR("Expected one CCU instruction instance, got[%u]", insNum), HCCL_E_INTERNAL);

    CcuResult ret = HcommCcuKernelRegisterStart(insHandle);
    CHK_PRT_RET(ret != CCU_SUCCESS, HCCL_ERROR("CCU register start failed[%d]", ret), ConvertCcuToHccl(ret));
    for (const KernelSpec &spec : specs) {
        const void *args[] = {spec.arg.get()};
        CcuKernelHandle handle = 0;
        ret = HcommCcuKernelRegister(insHandle, spec.die, "CcuBroadcastKernel",
            reinterpret_cast<const void *>(ops_hccl::CcuKernel), args, 1, &handle);
        CHK_PRT_RET(ret != CCU_SUCCESS, HCCL_ERROR("CCU register failed[%d]", ret), ConvertCcuToHccl(ret));
        resCtx.ccuKernels.push_back(handle);
        const auto plan = static_cast<CcuBroadcastPlanKind>(spec.arg->planKind);
        resCtx.kernelGroupSizes.push_back(plan == CcuBroadcastPlanKind::ROOT_EXCLUDED_SCATTER_GATHER
                ? spec.arg->groupSize - 1
                : spec.arg->groupSize);
    }
    ret = HcommCcuKernelRegisterEnd(insHandle);
    CHK_PRT_RET(ret != CCU_SUCCESS, HCCL_ERROR("CCU register end failed[%d]", ret), ConvertCcuToHccl(ret));
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclBroadcast(
    void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(buf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32, HCCL_ERROR("Only FP32 broadcast is supported"), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(count > std::numeric_limits<uint64_t>::max() / sizeof(float),
        HCCL_ERROR("Broadcast byte size overflows"), HCCL_E_PARA);
    const uint64_t totalBytes = count * sizeof(float);
    const bool small = totalBytes <= SMALL_MESSAGE_BYTES;

    OpParam param;
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(
        root >= param.rankSize, HCCL_ERROR("Invalid root[%u], rankSize[%u]", root, param.rankSize), HCCL_E_PARA);
    if (param.rankSize == 1 || count == 0) {
        return HCCL_SUCCESS;
    }
    int tagRet = sprintf_s(param.tag, sizeof(param.tag), "%s_%s_%u", BROADCAST_TAG, small ? "small" : "large", root);
    CHK_PRT_RET(tagRet <= 0, HCCL_ERROR("Failed to construct broadcast tag"), HCCL_E_INTERNAL);
    param.inputPtr = buf;
    param.outputPtr = buf;
    param.count = count;
    param.dataType = dataType;
    param.root = root;
    param.opType = HcclCMDType::HCCL_CMD_BROADCAST;

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH] = {};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;
    CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, stream, 1, &param.cpuThread));
    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    if (HcclEngineCtxGet(comm, param.tag, ccuEngine, &ctx, &ctxSize) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = ctxSize;
    } else {
        AlgResourceCtx resCtxHost;
        void *cclBufferAddr = nullptr;
        uint64_t cclBufferSize = 0;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};
        resCtxHost.ccuThread = param.cpuThread;
        resCtxHost.threads = {param.cpuThread};

        std::vector<KernelSpec> specs;
        CHK_RET(BuildKernelSpecs(param.myRank, param.rankSize, root, small, specs));
        for (KernelSpec &spec : specs) {
            CHK_RET(AcquireChannels(comm, param.myRank, param.rankSize, spec));
        }
        CHK_RET(RegisterKernels(comm, specs, resCtxHost));

        std::vector<char> seq = resCtxHost.Serialize();
        param.ctxSize = seq.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, ccuEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, ccuEngine, param.tag, seq.data(), seq.size(), 0));
    }

    CHK_RET(ops_hccl::ExecOp(param));
    return HCCL_SUCCESS;
}
