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
#include <limits>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_ccu_res.h>
#include <ccu/ccu_launch.h>

#include "log.h"
#include "common.h"
#include "../include/custom.h"
#include "hccl.h"
#include "exec_op.h"
#include "../op_kernel_ccu/ccu_kernel.h"

namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;
constexpr uint32_t INVALID_GROUP = 0xFFFFFFFFu;
HcclResult GetAllLayers(HcclComm comm, std::vector<uint32_t> &layers)
{
    uint32_t *layerPtr = nullptr;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layerPtr, &layerNum));
    CHK_PRT_RET(layerPtr == nullptr || layerNum == 0,
        HCCL_ERROR("[Scatter][GetAllLayers] empty net layers"), HCCL_E_NOT_SUPPORT);
    layers.assign(layerPtr, layerPtr + layerNum);
    std::sort(layers.begin(), layers.end());
    return HCCL_SUCCESS;
}

static bool HasCtpLinkOnLayer(HcclComm comm, uint32_t layer, uint32_t myRank, uint32_t peerRank)
{
    CommLink *linkList = nullptr;
    uint32_t linkNum = 0;
    if (HcclRankGraphGetLinks(comm, layer, myRank, peerRank, &linkList, &linkNum) != HCCL_SUCCESS
        || linkList == nullptr) {
        return false;
    }
    const CommProtocol protocol = CommProtocol::COMM_PROTOCOL_UBC_CTP;
    for (uint32_t idx = 0; idx < linkNum; idx++) {
        if (linkList[idx].linkAttr.linkProtocol == protocol) {
            return true;
        }
    }
    return false;
}

HcclResult FindCtpLinkOnLayer(HcclComm comm, uint32_t layer, uint32_t myRank, uint32_t peerRank, HcclChannelDesc &desc)
{
    CHK_RET(HcclChannelDescInit(&desc, 1));
    const CommProtocol protocol = CommProtocol::COMM_PROTOCOL_UBC_CTP;
    CommLink *linkList = nullptr;
    uint32_t linkNum = 0;
    if (HcclRankGraphGetLinks(comm, layer, myRank, peerRank, &linkList, &linkNum) != HCCL_SUCCESS
        || linkList == nullptr) {
        HCCL_ERROR("[Scatter][FindCtpLinkOnLayer] get links failed, layer %u, rank %u -> rank %u", layer, myRank,
            peerRank);
        return HCCL_E_NOT_FOUND;
    }
    for (uint32_t idx = 0; idx < linkNum; idx++) {
        const CommLink &link = linkList[idx];
        if (link.linkAttr.linkProtocol != protocol) {
            continue;
        }
        desc.remoteRank = peerRank;
        desc.notifyNum = CHANNEL_NOTIFY_NUM;
        desc.channelProtocol = link.linkAttr.linkProtocol;
        desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
        desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
        desc.localEndpoint.loc = link.srcEndpointDesc.loc;
        desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
        desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
        desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
        return HCCL_SUCCESS;
    }
    HCCL_ERROR("[Scatter][FindCtpLinkOnLayer] no UB_CTP link on layer %u between rank %u and rank %u", layer, myRank,
        peerRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult QueryLocalDie(HcclComm comm, uint32_t myRank, const EndpointDesc &localEndpoint, uint32_t &dieId)
{
    EndpointAttrDieId attrDie = 0;
    uint32_t infoLen = sizeof(EndpointAttrDieId);
    CHK_PRT_RET(HcclRankGraphGetEndpointInfo(comm, myRank, &localEndpoint, EndpointAttr::ENDPOINT_ATTR_DIE_ID,
                    infoLen, &attrDie) != HCCL_SUCCESS,
        HCCL_ERROR("[Scatter][QueryLocalDie] query ENDPOINT_ATTR_DIE_ID failed, rank %u", myRank), HCCL_E_NOT_FOUND);
    dieId = static_cast<uint32_t>(attrDie);
    return HCCL_SUCCESS;
}

HcclResult GroupChannelsByLayer(HcclComm comm, const OpParam &param, std::vector<ChannelGroup> &groups)
{
    std::vector<uint32_t> layers;
    CHK_RET(GetAllLayers(comm, layers));

    std::map<uint32_t, std::vector<std::pair<uint32_t, ChannelHandle>>> grouped;
    std::map<uint32_t, uint32_t> layerDie;
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; remoteRank++) {
        if (remoteRank == param.myRank) {
            continue;
        }
        bool found = false;
        for (uint32_t layer : layers) {
            if (!HasCtpLinkOnLayer(comm, layer, param.myRank, remoteRank)) {
                continue;
            }
            HcclChannelDesc desc;
            CHK_RET(FindCtpLinkOnLayer(comm, layer, param.myRank, remoteRank, desc));
            uint32_t dieId = 0;
            CHK_RET(QueryLocalDie(comm, param.myRank, desc.localEndpoint, dieId));
            layerDie[layer] = dieId;
            ChannelHandle channel;
            CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, &desc, 1, &channel));
            grouped[layer].emplace_back(remoteRank, channel);
            found = true;
            break;
        }
        CHK_PRT_RET(!found,
            HCCL_ERROR("[Scatter][GroupChannelsByLayer] no UB_CTP link to rank %u on any layer", remoteRank),
            HCCL_E_NOT_FOUND);
    }
    CHK_PRT_RET(grouped.size() > 2,
        HCCL_ERROR("[Scatter][GroupChannelsByLayer] layer groups [%zu] exceed IO die num 2", grouped.size()),
        HCCL_E_NOT_SUPPORT);
    std::vector<ChannelGroup> built;
    for (auto &entry : grouped) {
        ChannelGroup group;
        group.layer = entry.first;
        group.dieId = layerDie[entry.first];
        group.ranks.reserve(entry.second.size());
        group.channels.reserve(entry.second.size());
        for (auto &item : entry.second) { // 对端按 rank 升序遍历而来，天然有序
            group.ranks.push_back(item.first);
            group.channels.push_back(item.second);
        }
        CHK_PRT_RET(group.dieId >= 2,
            HCCL_ERROR("[Scatter][GroupChannelsByLayer] invalid die %u on layer %u", group.dieId, group.layer),
            HCCL_E_INTERNAL);
        built.push_back(std::move(group));
    }
    size_t meshIdx = 0; // 兜底：层号最低的组
    for (size_t i = 0; i < built.size(); i++) {
        if (built[i].dieId == 1 && built.size() >= 2) { // 只有多组时才用 die 判据（单组拓扑无所谓）
            meshIdx = i;
            break;
        }
    }
    if (meshIdx != 0) {
        std::swap(built[0], built[meshIdx]); // 规范化：mesh 永远排第一
        HCCL_INFO("[Scatter][GroupChannelsByLayer] rank %u normalized mesh group to index 0 (die %u, layer %u)",
            param.myRank, built[0].dieId, built[0].layer);
    }
    for (auto &group : built) {
        HCCL_INFO("[Scatter][GroupChannelsByLayer] rank %u group layer %u die %u, %zu peers", param.myRank,
            group.layer, group.dieId, group.ranks.size());
        groups.push_back(std::move(group));
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterOneScatterKernel(CcuInsHandle insHandle, const char *kernelName, void *kernelFunc,
    const OpParam &param, const ChannelGroup &group, uint32_t groupIndex, bool isMeshGroup, const ScatterTopoView &view,
    const std::vector<uint32_t> &meshPeers, CcuKernelHandle &kernelHandle)
{
    CcuKernelInfo kernelInfo;
    int ret = snprintf(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "%s", kernelName);
    CHK_PRT_RET(ret < 0, HCCL_ERROR("[Scatter] failed to fill kernelFuncName"), HCCL_E_INTERNAL);
    kernelInfo.kernelFunc = kernelFunc;

    auto kernelArg = std::make_shared<CcuKernelArgScatter>();
    kernelArg->rootId = param.root;
    kernelArg->targetRank = view.relayTargetRank;
    for (uint32_t i = 0; i < MAX_RANK_SIZE; ++i) {
        kernelArg->relaySourceRanks[i] = i < meshPeers.size() ? meshPeers[i] : 0xFFFFFFFFu;
    }
    kernelArg->rankSize = param.rankSize;
    kernelArg->rankId = param.myRank;
    kernelArg->groupIndex = groupIndex;
    kernelArg->hasLocalSlice = (groupIndex == 0) ? 1u : 0u;
    kernelArg->isMeshGroup = isMeshGroup ? 1u : 0u;
    kernelArg->relayTopo = view.relayTopo;
    kernelArg->relayCount = view.relayCount;
    kernelArg->firstPeerRank = group.ranks.empty() ? 0u : group.ranks[0];
    for (uint32_t i = 0; i < group.ranks.size() && i < MAX_RANK_SIZE; i++) {
        kernelArg->peerRanks[i] = group.ranks[i];
    }
    kernelInfo.setKernelArg(kernelArg);

    auto *kernelArgBase = static_cast<CcuKernelArgBase *>(kernelInfo.kernelArg);
    CHK_PTR_NULL(kernelArgBase);
    for (uint32_t i = 0; i < group.channels.size() && i < MAX_RANK_SIZE; i++) {
        kernelArgBase->channels[i] = group.channels[i];
    }
    kernelArgBase->channelCount = static_cast<uint32_t>(group.channels.size());

    const uint32_t dieId = group.dieId;
    constexpr uint32_t argNum = 1; // 注册期仅固化 1 个 kernelArg 实例
    const void *kernelArgs[] = {kernelInfo.kernelArg};
    CcuResult ccuRet = HcommCcuKernelRegister(
        insHandle, dieId, kernelInfo.kernelFuncName, kernelInfo.kernelFunc, kernelArgs, argNum, &kernelHandle);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS, HCCL_ERROR("[Scatter] ccu kernel register failed: ccuRet -> %d", ccuRet),
        ConvertCcuToHccl(ccuRet));
    return HCCL_SUCCESS;
}
HcclResult RegisterScatterCcuKernels(HcclComm comm, const OpParam &param, const std::vector<ChannelGroup> &groups,
    AlgResourceCtx &resCtx, const ScatterTopoView &view)
{
    CcuInsHandle insHandle = 0;
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1, HCCL_ERROR("[Scatter] HcclCommQueryCcuIns fail! insNum is [%u]", insNum),
        HCCL_E_INTERNAL);

    CcuResult ccuRet = HcommCcuKernelRegisterStart(insHandle);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS, HCCL_ERROR("[Scatter] ccu kernel register start failed: ccuRet -> %d", ccuRet),
        ConvertCcuToHccl(ccuRet));

    const bool amRoot = (param.myRank == param.root);
    const bool twoGroups = (groups.size() >= 2);
    const bool relayTopo = (view.relayTopo != 0);
    const bool amRelay = (view.amRelay != 0);

    resCtx.ccuKernels.assign(groups.size() * KERNEL_SLOTS_PER_GROUP, 0);
    for (uint32_t g = 0; g < groups.size(); g++) {
        const bool isMeshGroup = twoGroups && (g == 0);
        static const struct {
            void *fn;
            const char *name;
        } ENTRY_TABLE[ENTRY_COUNT] = {
            {reinterpret_cast<void *>(ops_hccl::CcuScatterRootPub), "SRootPub"},
            {reinterpret_cast<void *>(ops_hccl::CcuScatterPeerPull), "SPeerPull"},
            {reinterpret_cast<void *>(ops_hccl::CcuScatterRootPush), "SRootPush"},
            {reinterpret_cast<void *>(ops_hccl::CcuScatterRecvLarge), "SRecvLarge"},
            {reinterpret_cast<void *>(ops_hccl::CcuScatterRelayRead), "SRelayRead"},   // 4
            {reinterpret_cast<void *>(ops_hccl::CcuScatterRelayOwn), "SRelayOwn"},     // 5
            {reinterpret_cast<void *>(ops_hccl::CcuScatterRelayFwd), "SRelayFwd"},     // 6
        };
        ScatterEntry entries[KERNEL_SLOTS_PER_GROUP] = {ENTRY_ROOT_PUB, ENTRY_ROOT_PUB};
        CcuKernelHandle slotHandle[KERNEL_SLOTS_PER_GROUP] = {0};
        for (uint32_t slot = 0; slot < KERNEL_SLOTS_PER_GROUP; slot++) {
            entries[slot] = PickScatterEntry(amRoot, amRelay, isMeshGroup, relayTopo, slot);
            uint32_t dup = KERNEL_SLOTS_PER_GROUP;
            for (uint32_t p = 0; p < slot; p++) {
                if (entries[p] == entries[slot]) {
                    dup = p;
                    break;
                }
            }
            if (dup != KERNEL_SLOTS_PER_GROUP) {
                slotHandle[slot] = slotHandle[dup]; // 同组同入口：复用句柄，不再注册
                continue;
            }
            char kernelName[64];
            int ret = snprintf(kernelName, sizeof(kernelName), "ScatterG%uS%u%s", g, slot,
                ENTRY_TABLE[entries[slot]].name);
            CHK_PRT_RET(ret < 0 || static_cast<size_t>(ret) >= sizeof(kernelName),
                HCCL_ERROR("[Scatter] kernel name build failed"), HCCL_E_INTERNAL);
            HcclResult registered = RegisterOneScatterKernel(insHandle, kernelName, ENTRY_TABLE[entries[slot]].fn,
                param, groups[g], g, isMeshGroup, view, groups[0].ranks, slotHandle[slot]);
            if (registered != HCCL_SUCCESS) {
                (void)HcommCcuKernelRegisterEnd(insHandle);
                return registered;
            }
        }
        for (uint32_t slot = 0; slot < KERNEL_SLOTS_PER_GROUP; slot++) {
            resCtx.ccuKernels[g * KERNEL_SLOTS_PER_GROUP + slot] = slotHandle[slot];
        }
        const char *slotName[KERNEL_SLOTS_PER_GROUP] = {ENTRY_TABLE[entries[0]].name, ENTRY_TABLE[entries[1]].name};
        HCCL_INFO("[Scatter] rank %u register group %u (layer %u die %u, %u peers, mesh=%d): slot0=%s slot1=%s",
            param.myRank, g, groups[g].layer, groups[g].dieId, static_cast<uint32_t>(groups[g].ranks.size()),
            static_cast<int>(isMeshGroup), slotName[0], slotName[1]);
    }

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    CHK_PRT_RET(ccuRet != CCU_SUCCESS, HCCL_ERROR("[Scatter] ccu kernel register end failed: ccuRet -> %d", ccuRet),
        ConvertCcuToHccl(ccuRet));
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    if (recvCount == 0) {
        return HCCL_SUCCESS;
    }
    if (dataType != HCCL_DATA_TYPE_FP32) {
        HCCL_ERROR("[HcclScatter] unsupported dataType [%u]", static_cast<uint32_t>(dataType));
        return HCCL_E_NOT_SUPPORT;
    }

    // 构造算子参数
    OpParam param;
    sprintf(param.tag, "%s", "hccl_custom_scatter");
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_SCATTER;
    param.root = root;

    // 注册算子信息
    HcclDfxOpInfo dfxInfo;
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    if (root >= param.rankSize || param.rankSize > MAX_RANK_SIZE || param.myRank >= param.rankSize) {
        HCCL_ERROR("[HcclScatter] invalid root [%u] or rankSize [%u]", root, param.rankSize);
        return HCCL_E_PARA;
    }
    if (recvCount > std::numeric_limits<uint64_t>::max() / sizeof(float) / param.rankSize) {
        HCCL_ERROR("[HcclScatter] recvCount [%llu] overflow", static_cast<unsigned long long>(recvCount));
        return HCCL_E_PARA;
    }
    CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;
    CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, stream, 3, &param.cpuThread));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, ccuEngine, &ctx, &size) == HCCL_SUCCESS) {
        HCCL_INFO("Engine context already exists");
        param.resCtx = ctx;
        param.ctxSize = size;
    } else {
        AlgResourceCtx resCtxHost;
        void *cclBufferAddr = nullptr;
        uint64_t cclBufferSize = 0;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};
        std::vector<ChannelGroup> groups;
        for (uint32_t i = 0; i < MAX_RANK_SIZE; i++) {
            resCtxHost.groupOfRank[i] = INVALID_GROUP;
        }
        resCtxHost.groupCount = 0;
        if (param.rankSize > 1) {
            CHK_RET(GroupChannelsByLayer(comm, param, groups));
        }
        for (uint32_t g = 0; g < groups.size(); g++) {
            for (uint32_t r : groups[g].ranks) {
                resCtxHost.groupOfRank[r] = g;
            }
        }
        resCtxHost.groupCount = static_cast<uint32_t>(groups.size());
        ScatterTopoView topo;
        if (param.rankSize > 1) {
            topo = DeriveTopo(resCtxHost.groupOfRank, resCtxHost.groupCount, param.rankSize, param.myRank, param.root);
            HCCL_INFO("[Scatter] rank %u topo: M=%u C=%u k=%u relayTopo=%u amRelay=%u target=%u root=%u", param.myRank,
                topo.meshCount, topo.crossCount, topo.relayCount, topo.relayTopo, topo.amRelay, topo.relayTargetRank,
                param.root);
        }
        const uint32_t threadNum = (groups.size() > 1) ? 2 : 1;
        resCtxHost.threads.resize(threadNum);
        resCtxHost.threads[0] = param.cpuThread;
        if (threadNum > 1) {
            CHK_RET(HcclThreadAcquire(comm, ccuEngine, threadNum - 1, 1, &resCtxHost.threads[1]));
        }
        if (param.rankSize > 1) {
            CHK_RET(RegisterScatterCcuKernels(comm, param, groups, resCtxHost, topo));
        }
        std::vector<char> seq = resCtxHost.Serialize();
        uint64_t seqSize = seq.size();
        param.ctxSize = seqSize;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, ccuEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, ccuEngine, param.tag, seq.data(), seqSize, 0));
    }
    CHK_RET(ops_hccl::ExecOp(param));
    return HCCL_SUCCESS;
}
