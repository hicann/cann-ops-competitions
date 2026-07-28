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

#include <ccu/ccu_launch.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>

#include "ccu_kernel.h"
#include "common.h"
#include "custom.h"
#include "exec_op.h"
#include "hccl.h"
#include "log.h"

namespace {

constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;

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

HcclResult BuildChannelDesc(HcclComm comm, uint32_t netLayer,
    uint32_t srcRank, uint32_t dstRank, HcclChannelDesc &desc)
{
    CommLink *linkList = nullptr;
    uint32_t listSize = 0;
    CHK_RET(HcclRankGraphGetLinks(comm, netLayer, srcRank, dstRank, &linkList, &listSize));
    CHK_PRT_RET(listSize == 0,
        HCCL_ERROR("[AllReduce] no link between rank %u and rank %u", srcRank, dstRank),
        HCCL_E_NOT_FOUND);

    CommLink *selected = nullptr;
    for (uint32_t i = 0; i < listSize; ++i) {
        if (linkList[i].linkAttr.linkProtocol == COMM_PROTOCOL_UBC_CTP) {
            selected = &linkList[i];
            break;
        }
    }
    CHK_PRT_RET(selected == nullptr,
        HCCL_ERROR("[AllReduce] UBC_CTP link not found between rank %u and rank %u", srcRank, dstRank),
        HCCL_E_NOT_FOUND);

    CHK_RET(HcclChannelDescInit(&desc, 1));
    desc.remoteRank = dstRank;
    desc.notifyNum = CHANNEL_NOTIFY_NUM;
    desc.channelProtocol = selected->linkAttr.linkProtocol;
    desc.localEndpoint.protocol = selected->srcEndpointDesc.protocol;
    desc.localEndpoint.commAddr = selected->srcEndpointDesc.commAddr;
    desc.localEndpoint.loc = selected->srcEndpointDesc.loc;
    desc.remoteEndpoint.protocol = selected->dstEndpointDesc.protocol;
    desc.remoteEndpoint.commAddr = selected->dstEndpointDesc.commAddr;
    desc.remoteEndpoint.loc = selected->dstEndpointDesc.loc;
    return HCCL_SUCCESS;
}

HcclResult AcquireChannels(HcclComm comm, const OpParam &param,
    std::vector<ChannelHandle> &channels, std::vector<HcclChannelDesc> &descs)
{
    constexpr uint32_t closLayer = 1;
    const uint32_t channelCount = param.rankSize - 1;
    descs.resize(channelCount);
    channels.resize(channelCount);

    uint32_t channelIndex = 0;
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer == param.myRank) {
            continue;
        }
        CHK_RET(BuildChannelDesc(comm, closLayer, param.myRank, peer, descs[channelIndex]));
        ++channelIndex;
    }
    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU,
        descs.data(), channelCount, channels.data()));
    return HCCL_SUCCESS;
}

HcclResult GetDescDieId(HcclComm comm, uint32_t rankId,
    const HcclChannelDesc &desc, uint32_t &dieId)
{
    EndpointAttrDieId endpointDie = 0;
    uint32_t infoLen = sizeof(endpointDie);
    CHK_RET(HcclRankGraphGetEndpointInfo(comm, rankId,
        &desc.localEndpoint, ENDPOINT_ATTR_DIE_ID, infoLen, &endpointDie));
    dieId = static_cast<uint32_t>(endpointDie);
    return HCCL_SUCCESS;
}

struct HierPipelineResources {
    bool available = false;
    uint32_t localRank = 0;
    uint32_t localSize = 0;
    uint32_t laneCount = 0;
    uint32_t serverId = 0;
    bool isGateway = false;
    uint32_t meshDie = 0;
    uint32_t closDie = 0;
    std::vector<ChannelHandle> meshChannels;
    std::vector<ChannelHandle> closChannels;
};

HcclResult AcquireHierPipeline(HcclComm comm, const OpParam &param,
    const std::vector<HcclChannelDesc> &baselineDescs,
    const std::vector<ChannelHandle> &baselineChannels,
    HierPipelineResources &hier)
{
    constexpr uint32_t meshLayer = 0;
    constexpr uint64_t hierarchyMinCount = (512ULL * 1024) / sizeof(float);
    if (param.rankSize != 16 || param.count <= hierarchyMinCount) {
        return HCCL_SUCCESS;
    }

    constexpr uint32_t laneCount = 8U;
    uint32_t *rankList = nullptr;
    uint32_t rankNum = 0;
    HcclResult rankRet = HcclRankGraphGetRanksByLayer(
        comm, meshLayer, &rankList, &rankNum);
    const bool validLocalSize = rankNum == 8U;
    if (rankRet != HCCL_SUCCESS || rankList == nullptr || !validLocalSize) {
        HCCL_INFO("[AllReduce] hierarchy unavailable: layer-0 rank count=%u ret=%d",
            rankNum, rankRet);
        return HCCL_SUCCESS;
    }

    std::vector<uint32_t> localRanks(rankList, rankList + rankNum);
    std::sort(localRanks.begin(), localRanks.end());
    auto localIt = std::find(localRanks.begin(), localRanks.end(), param.myRank);
    if (localIt == localRanks.end()) {
        return HCCL_SUCCESS;
    }
    hier.localRank = static_cast<uint32_t>(localIt - localRanks.begin());
    hier.localSize = rankNum;
    hier.laneCount = laneCount;
    hier.isGateway = true;

    std::vector<uint32_t> remoteRanks;
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (!std::binary_search(localRanks.begin(), localRanks.end(), rank)) {
            remoteRanks.push_back(rank);
        }
    }
    if (remoteRanks.size() + localRanks.size() != param.rankSize ||
        remoteRanks.size() < laneCount) {
        return HCCL_SUCCESS;
    }
    hier.serverId = localRanks.front() < remoteRanks.front() ? 0U : 1U;

    std::vector<HcclChannelDesc> meshDescs(rankNum - 1);
    uint32_t descIndex = 0;
    bool meshDieReady = false;
    for (uint32_t peer : localRanks) {
        if (peer == param.myRank) {
            continue;
        }
        HcclResult descRet = BuildChannelDesc(
            comm, meshLayer, param.myRank, peer, meshDescs[descIndex]);
        if (descRet != HCCL_SUCCESS) {
            HCCL_INFO("[AllReduce] hierarchy unavailable: layer-0 peer=%u ret=%d",
                peer, descRet);
            return HCCL_SUCCESS;
        }
        uint32_t dieId = 0;
        HcclResult dieRet = GetDescDieId(comm, param.myRank, meshDescs[descIndex], dieId);
        if (dieRet != HCCL_SUCCESS || dieId >= 2 ||
            (meshDieReady && dieId != hier.meshDie)) {
            HCCL_INFO("[AllReduce] hierarchy unavailable: Mesh channels span IO dies");
            return HCCL_SUCCESS;
        }
        hier.meshDie = dieId;
        meshDieReady = true;
        ++descIndex;
    }

    if (baselineDescs.empty()) {
        return HCCL_SUCCESS;
    }
    if (baselineChannels.size() != baselineDescs.size()) {
        return HCCL_SUCCESS;
    }
    bool closDieReady = false;
    for (const auto &desc : baselineDescs) {
        uint32_t dieId = 0;
        HcclResult dieRet = GetDescDieId(comm, param.myRank, desc, dieId);
        if (dieRet != HCCL_SUCCESS || dieId >= 2 ||
            (closDieReady && dieId != hier.closDie)) {
            HCCL_INFO("[AllReduce] hierarchy unavailable: Clos channels span IO dies");
            return HCCL_SUCCESS;
        }
        hier.closDie = dieId;
        closDieReady = true;
    }
    if (!meshDieReady || !closDieReady || hier.meshDie == hier.closDie) {
        HCCL_INFO("[AllReduce] hierarchy unavailable: meshDie=%u closDie=%u",
            hier.meshDie, hier.closDie);
        return HCCL_SUCCESS;
    }

    hier.meshChannels.resize(meshDescs.size());
    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU,
        meshDescs.data(), static_cast<uint32_t>(meshDescs.size()),
        hier.meshChannels.data()));

    if (hier.isGateway) {
        const uint32_t lane = hier.localRank % laneCount;
        const uint32_t crossPeer = remoteRanks[lane];
        const uint32_t crossIndex = crossPeer < param.myRank ? crossPeer : crossPeer - 1;
        if (crossIndex >= baselineDescs.size()) {
            return HCCL_SUCCESS;
        }
        uint32_t crossDie = 0;
        HcclResult crossDieRet = GetDescDieId(
            comm, param.myRank, baselineDescs[crossIndex], crossDie);
        if (crossDieRet != HCCL_SUCCESS || crossDie != hier.closDie) {
            HCCL_INFO("[AllReduce] hierarchy unavailable: cross lane is on die=%u, expected=%u",
                crossDie, hier.closDie);
            return HCCL_SUCCESS;
        }
        // The corresponding layer-1 channel is already part of the baseline
        // all-peer set. Reuse its handle instead of consuming a duplicate CCU
        // channel resource; baseline and hierarchy kernels never run together.
        hier.closChannels.push_back(baselineChannels[crossIndex]);
    }

    hier.available = true;
    HCCL_INFO("[AllReduce] hierarchy enabled: rank=%u local=%u/%u lanes=%u server=%u gateway=%u meshDie=%u closDie=%u",
        param.myRank, hier.localRank, hier.localSize, hier.laneCount,
        hier.serverId, hier.isGateway ? 1U : 0U, hier.meshDie, hier.closDie);
    return HCCL_SUCCESS;
}

HcclResult RegisterKernel(HcclComm comm, const OpParam &param,
    const std::vector<ChannelHandle> &channels,
    const HierPipelineResources &hier, AlgResourceCtx &resCtx)
{
    auto kernelArg = std::make_shared<CcuKernelArgAllReduce>();
    kernelArg->rankSize = param.rankSize;
    kernelArg->rankId = param.myRank;
    kernelArg->dataType = param.dataType;
    kernelArg->reduceOp = param.reduceType;
    kernelArg->channelCount = static_cast<uint32_t>(channels.size());
    for (uint32_t i = 0; i < channels.size(); ++i) {
        kernelArg->channels[i] = channels[i];
    }
    CcuInsHandle insHandle{0};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1,
        HCCL_ERROR("[AllReduce] expected one CCU instruction instance, got %u", insNum),
        HCCL_E_INTERNAL);

    CcuResult ccuRet = HcommCcuKernelRegisterStart(insHandle);
    if (ccuRet != CCU_SUCCESS) {
        return ConvertCcuResult(ccuRet);
    }

    const void *baselineArgs[] = {kernelArg.get()};
    const uint32_t baselineDie = hier.available ? hier.closDie : 0U;
    CcuKernelHandle baselineHandle = 0;
    ccuRet = HcommCcuKernelRegister(insHandle, baselineDie,
        "CcuRootAllReduceKernel", reinterpret_cast<void *>(ops_hccl::CcuKernel),
        baselineArgs, 1, &baselineHandle);
    if (ccuRet != CCU_SUCCESS) {
        return ConvertCcuResult(ccuRet);
    }
    resCtx.ccuKernels.push_back(baselineHandle);

    if (hier.available) {
        resCtx.useParallel2x8 = 1;
        resCtx.hierLocalRank = hier.localRank;
        resCtx.hierLocalSize = hier.localSize;
        resCtx.hierLaneCount = hier.laneCount;
        resCtx.hierIsGateway = hier.isGateway ? 1U : 0U;
        resCtx.hierServerId = hier.serverId;

        auto meshArg = std::make_shared<CcuKernelArgAllReduce>();
        meshArg->rankSize = hier.localSize;
        meshArg->rankId = hier.localRank;
        meshArg->dataType = param.dataType;
        meshArg->reduceOp = param.reduceType;
        meshArg->hierLaneCount = hier.laneCount;
        meshArg->hierIsOwner = hier.isGateway ? 1U : 0U;
        meshArg->channelCount = static_cast<uint32_t>(hier.meshChannels.size());
        for (uint32_t i = 0; i < hier.meshChannels.size(); ++i) {
            meshArg->channels[i] = hier.meshChannels[i];
        }
        const void *meshArgs[] = {meshArg.get()};
        CcuKernelHandle meshHandle = 0;
        ccuRet = HcommCcuKernelRegister(insHandle, hier.meshDie,
            "CcuParallel2x8MeshKernel",
            reinterpret_cast<void *>(ops_hccl::CcuParallel2x8MeshKernel),
            meshArgs, 1, &meshHandle);
        if (ccuRet != CCU_SUCCESS) {
            return ConvertCcuResult(ccuRet);
        }
        resCtx.ccuKernels.push_back(meshHandle);

        if (hier.isGateway) {
            auto closArg = std::make_shared<CcuKernelArgAllReduce>();
            closArg->rankSize = 2;
            closArg->rankId = hier.serverId;
            closArg->dataType = param.dataType;
            closArg->reduceOp = param.reduceType;
            closArg->channelCount = 1;
            closArg->channels[0] = hier.closChannels[0];
            const void *closArgs[] = {closArg.get()};
            CcuKernelHandle closHandle = 0;
            ccuRet = HcommCcuKernelRegister(insHandle, hier.closDie,
                "CcuParallel2x8ClosKernel",
                reinterpret_cast<void *>(ops_hccl::CcuParallel2x8ClosKernel),
                closArgs, 1, &closHandle);
            if (ccuRet != CCU_SUCCESS) {
                return ConvertCcuResult(ccuRet);
            }
            resCtx.ccuKernels.push_back(closHandle);
        }
    } else if (param.rankSize == 4 || param.rankSize == 12 || param.rankSize == 16) {
        void *scratchFunc = reinterpret_cast<void *>(ops_hccl::CcuMesh16ScratchKernel);
        const char *scratchName = "CcuMesh16ScratchKernel";
        if (param.rankSize == 12) {
            scratchFunc = reinterpret_cast<void *>(ops_hccl::CcuMesh12ScratchKernel);
            scratchName = "CcuMesh12ScratchKernel";
        } else if (param.rankSize == 4) {
            scratchFunc = reinterpret_cast<void *>(ops_hccl::CcuMesh4ScratchKernel);
            scratchName = "CcuMesh4ScratchKernel";
        }
        CcuKernelHandle scratchHandle = 0;
        ccuRet = HcommCcuKernelRegister(insHandle, 0, scratchName,
            scratchFunc, baselineArgs, 1, &scratchHandle);
        if (ccuRet != CCU_SUCCESS) {
            return ConvertCcuResult(ccuRet);
        }
        resCtx.ccuKernels.push_back(scratchHandle);
    }

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    if (ccuRet != CCU_SUCCESS) {
        return ConvertCcuResult(ccuRet);
    }
    return HCCL_SUCCESS;
}

HcclResult CreateResources(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtx)
{
    void *cclBufferAddr = nullptr;
    uint64_t cclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    resCtx.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

    resCtx.threads.push_back(param.cpuThread);
    if (param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    std::vector<ChannelHandle> channels;
    std::vector<HcclChannelDesc> descs;
    HierPipelineResources hier;
    CHK_RET(AcquireChannels(comm, param, channels, descs));
    CHK_RET(AcquireHierPipeline(comm, param, descs, channels, hier));

    if (hier.available && hier.isGateway) {
        aclrtStream auxiliaryStream = nullptr;
        aclError aclRet = aclrtCreateStream(&auxiliaryStream);
        CHK_PRT_RET(aclRet != ACL_SUCCESS,
            HCCL_ERROR("[AllReduce] failed to create Clos auxiliary stream: %d", aclRet),
            HCCL_E_INTERNAL);
        ThreadHandle auxiliaryThread = 0;
        HcclResult threadRet = HcclThreadAcquireWithStream(
            comm, CommEngine::COMM_ENGINE_CCU, auxiliaryStream, 2, &auxiliaryThread);
        if (threadRet != HCCL_SUCCESS) {
            (void)aclrtDestroyStream(auxiliaryStream);
            return threadRet;
        }
        resCtx.threads.push_back(auxiliaryThread);
        resCtx.auxiliaryStream = reinterpret_cast<uint64_t>(auxiliaryStream);
    }

    CHK_RET(RegisterKernel(comm, param, channels, hier, resCtx));
    return HCCL_SUCCESS;
}

} // namespace

HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("[AllReduce] only float32 is supported"), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(op != HCCL_REDUCE_SUM,
        HCCL_ERROR("[AllReduce] only sum is supported"), HCCL_E_NOT_SUPPORT);

    OpParam param;
    int tagRet = std::snprintf(param.tag, sizeof(param.tag), "%s",
        "hccl_custom_allreduce_ccu_a35_b36_best_v1");
    CHK_PRT_RET(tagRet <= 0 || static_cast<size_t>(tagRet) >= sizeof(param.tag),
        HCCL_ERROR("[AllReduce] failed to set tag"), HCCL_E_INTERNAL);
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = count;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE,
        HCCL_ERROR("[AllReduce] invalid rank size %u", param.rankSize), HCCL_E_PARA);

    constexpr CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;
    constexpr uint64_t hierarchyMinCount = (512ULL * 1024) / sizeof(float);
    const bool hierarchyCandidate = param.rankSize == 16 &&
        param.count > hierarchyMinCount;
    CHK_RET(HcclThreadAcquireWithStream(
        comm, ccuEngine, stream, hierarchyCandidate ? 2U : 1U, &param.cpuThread));

    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    if (HcclEngineCtxGet(comm, param.tag, ccuEngine, &ctx, &ctxSize) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = ctxSize;
    } else {
        AlgResourceCtx resCtx;
        CHK_RET(CreateResources(comm, param, resCtx));

        std::vector<char> serialized = resCtx.Serialize();
        param.ctxSize = serialized.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, ccuEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, ccuEngine, param.tag,
            serialized.data(), serialized.size(), 0));
    }

    CHK_RET(ops_hccl::ExecOp(param));
    return HCCL_SUCCESS;
}