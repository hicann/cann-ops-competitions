/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>
#include <ccu/ccu_launch.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "ccu_kernel.h"
#include "common.h"
#include "custom.h"
#include "exec_op.h"
#include "hccl.h"
#include "log.h"

namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 1;
constexpr uint32_t DIE_COUNT = 2;

struct ChannelGroup {
    uint32_t dieId = 0;
    std::vector<HcclChannelDesc> descs;
    std::vector<uint32_t> peerRanks;
    std::vector<ChannelHandle> handles;
};

const CommLink *SelectCcuLink(CommLink *links, uint32_t linkCount)
{
    if (links == nullptr || linkCount == 0) {
        return nullptr;
    }
    for (uint32_t i = 0; i < linkCount; ++i) {
        if (links[i].linkAttr.linkProtocol == CommProtocol::COMM_PROTOCOL_UBC_CTP) {
            return &links[i];
        }
    }
    return nullptr;
}

HcclResult GetNetLayers(HcclComm comm, std::vector<uint32_t> &layers)
{
    uint32_t *layerList = nullptr;
    uint32_t layerCount = 0;
    HcclResult ret = HcclRankGraphGetLayers(comm, &layerList, &layerCount);
    if (ret == HCCL_SUCCESS && layerList != nullptr && layerCount != 0) {
        layers.assign(layerList, layerList + layerCount);
        std::sort(layers.begin(), layers.end());
        return HCCL_SUCCESS;
    }
    layers.assign(1, 0);
    return HCCL_SUCCESS;
}

HcclResult BuildChannelDesc(HcclComm comm, uint32_t srcRank, uint32_t dstRank,
    const std::vector<uint32_t> &layers, HcclChannelDesc &desc)
{
    HcclResult lastRet = HCCL_E_NOT_FOUND;
    for (uint32_t layer : layers) {
        CommLink *links = nullptr;
        uint32_t linkCount = 0;
        HcclResult ret = HcclRankGraphGetLinks(comm, layer, srcRank, dstRank, &links, &linkCount);
        if (ret != HCCL_SUCCESS) {
            lastRet = ret;
            continue;
        }
        const CommLink *link = SelectCcuLink(links, linkCount);
        if (link == nullptr) {
            continue;
        }

        CHK_RET(HcclChannelDescInit(&desc, 1));
        desc.remoteRank = dstRank;
        desc.notifyNum = CHANNEL_NOTIFY_NUM;
        desc.channelProtocol = link->linkAttr.linkProtocol;
        desc.localEndpoint = link->srcEndpointDesc;
        desc.remoteEndpoint = link->dstEndpointDesc;
        return HCCL_SUCCESS;
    }
    HCCL_ERROR("[BuildChannelDesc] no CCU link between rank %u and rank %u", srcRank, dstRank);
    return lastRet;
}

HcclResult AcquireChannelGroups(HcclComm comm, uint32_t myRank, uint32_t rankSize,
    std::vector<ChannelGroup> &groups)
{
    std::vector<uint32_t> layers;
    CHK_RET(GetNetLayers(comm, layers));
    groups.resize(DIE_COUNT);
    for (uint32_t die = 0; die < DIE_COUNT; ++die) {
        groups[die].dieId = die;
    }

    std::vector<HcclChannelDesc> allDescs;
    std::vector<uint32_t> allPeerRanks;
    std::vector<uint32_t> allDieIds;
    allDescs.reserve(rankSize - 1);
    allPeerRanks.reserve(rankSize - 1);
    allDieIds.reserve(rankSize - 1);
    for (uint32_t remoteRank = 0; remoteRank < rankSize; ++remoteRank) {
        if (remoteRank == myRank) {
            continue;
        }
        HcclChannelDesc desc;
        CHK_RET(BuildChannelDesc(comm, myRank, remoteRank, layers, desc));
        EndpointAttrDieId dieId = 0;
        CHK_RET(HcclRankGraphGetEndpointInfo(comm, myRank, &desc.localEndpoint,
            ENDPOINT_ATTR_DIE_ID, sizeof(dieId), &dieId));
        if (dieId >= DIE_COUNT) {
            HCCL_ERROR("[AcquireChannelGroups] invalid die id %u", dieId);
            return HCCL_E_INTERNAL;
        }
        allDescs.push_back(desc);
        allPeerRanks.push_back(remoteRank);
        allDieIds.push_back(dieId);
    }

    std::vector<ChannelHandle> allHandles(allDescs.size());
    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, allDescs.data(),
        static_cast<uint32_t>(allDescs.size()), allHandles.data()));
    for (size_t i = 0; i < allHandles.size(); ++i) {
        ChannelGroup &group = groups[allDieIds[i]];
        group.descs.push_back(allDescs[i]);
        group.peerRanks.push_back(allPeerRanks[i]);
        group.handles.push_back(allHandles[i]);
    }

    return HCCL_SUCCESS;
}

HcclResult RegisterKernels(HcclComm comm, uint32_t myRank, uint32_t rankSize,
    const std::vector<ChannelGroup> &groups, AlgResourceCtx &resCtx)
{
    CcuInsHandle insHandle = 0;
    uint32_t insCount = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insCount));
    if (insCount != 1 || groups.size() != DIE_COUNT) {
        return HCCL_E_INTERNAL;
    }

    const bool singleGroup = groups[0].handles.empty() || groups[1].handles.empty();
    const uint32_t localDie = groups[0].handles.size() > groups[1].handles.size() ? 1U : 0U;
    const uint32_t secondGroupBase = static_cast<uint32_t>(groups[0].handles.size()) +
        (localDie == 0 ? 1U : 0U);
    CcuResult ret = HcommCcuKernelRegisterStart(insHandle);
    if (ret != CCU_SUCCESS) {
        return ConvertCcuToHccl(ret);
    }

    resCtx.ccuKernels.resize(singleGroup ? 1 : DIE_COUNT + 1);
    uint32_t kernelIndex = 0;
    for (uint32_t die = 0; die < DIE_COUNT; ++die) {
        const ChannelGroup &group = groups[die];
        if (singleGroup && group.handles.empty()) {
            continue;
        }
        auto kernelArg = std::make_shared<AllReduceKernelArg>();
        kernelArg->rankSize = rankSize;
        kernelArg->rankId = myRank;
        kernelArg->dataType = HCCL_DATA_TYPE_FP32;
        kernelArg->reduceOp = HCCL_REDUCE_SUM;
        kernelArg->dieId = die;
        kernelArg->includeLocal = singleGroup || die == localDie;
        kernelArg->directBroadcast = singleGroup;
        kernelArg->writeOutput = true;
        kernelArg->scratchSlotBase = singleGroup || die == 0 ? 0 : secondGroupBase;
        kernelArg->partialSlot[0] = 0;
        kernelArg->partialSlot[1] = secondGroupBase;
        for (size_t i = 0; i < group.handles.size(); ++i) {
            kernelArg->channels[i] = group.handles[i];
            kernelArg->peerRanks[i] = group.peerRanks[i];
        }
        kernelArg->channelCount = static_cast<uint32_t>(group.handles.size());

        CcuKernelInfo kernelInfo{};
        (void)snprintf(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName),
            "CcuKernelDie%u", die);
        kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuKernel);
        kernelInfo.setKernelArg(kernelArg);
        const void *kernelArgs[] = {kernelInfo.kernelArg};
        constexpr uint32_t reservedDieId = 0;
        ret = HcommCcuKernelRegister(insHandle, reservedDieId, kernelInfo.kernelFuncName,
            kernelInfo.kernelFunc, kernelArgs, 1, &resCtx.ccuKernels[kernelIndex]);
        if (ret != CCU_SUCCESS) {
            return ConvertCcuToHccl(ret);
        }
        ++kernelIndex;
    }

    if (singleGroup) {
        ret = HcommCcuKernelRegisterEnd(insHandle);
        return ConvertCcuToHccl(ret);
    }

    auto combineArg = std::make_shared<AllReduceKernelArg>();
    combineArg->rankSize = rankSize;
    combineArg->rankId = myRank;
    combineArg->dataType = HCCL_DATA_TYPE_FP32;
    combineArg->reduceOp = HCCL_REDUCE_SUM;
    combineArg->dieId = 0;
    combineArg->combineOnly = true;
    combineArg->partialSlot[0] = 0;
    combineArg->partialSlot[1] = secondGroupBase;
    for (size_t i = 0; i < groups[0].handles.size(); ++i) {
        combineArg->channels[i] = groups[0].handles[i];
        combineArg->peerRanks[i] = groups[0].peerRanks[i];
    }
    combineArg->channelCount = static_cast<uint32_t>(groups[0].handles.size());

    CcuKernelInfo combineInfo{};
    (void)snprintf(combineInfo.kernelFuncName, sizeof(combineInfo.kernelFuncName),
        "%s", "CcuKernelCombine");
    combineInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuKernel);
    combineInfo.setKernelArg(combineArg);
    const void *combineArgs[] = {combineInfo.kernelArg};
    constexpr uint32_t reservedDieId = 0;
    ret = HcommCcuKernelRegister(insHandle, reservedDieId, combineInfo.kernelFuncName,
        combineInfo.kernelFunc, combineArgs, 1, &resCtx.ccuKernels[DIE_COUNT]);
    if (ret != CCU_SUCCESS) {
        return ConvertCcuToHccl(ret);
    }

    ret = HcommCcuKernelRegisterEnd(insHandle);
    return ConvertCcuToHccl(ret);
}
} // namespace

HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    if (dataType != HCCL_DATA_TYPE_FP32 || op != HCCL_REDUCE_SUM) {
        return HCCL_E_NOT_SUPPORT;
    }
    if (count == 0) {
        return HCCL_SUCCESS;
    }
    if (count > UINT64_MAX / sizeof(float)) {
        return HCCL_E_PARA;
    }

    OpParam param;
    (void)snprintf(param.tag, sizeof(param.tag), "%s", "hccl_custom_allreduce");
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = count;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    if (param.rankSize <= 1 || param.rankSize > MAX_RANK_SIZE || param.myRank >= param.rankSize) {
        return HCCL_E_PARA;
    }

    const CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;
    CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, stream, 1, &param.cpuThread));

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH] = {};
    CHK_RET(HcclGetCommName(comm, commName));
    dfxInfo.opType = static_cast<uint32_t>(param.opType);
    dfxInfo.reduceOp = static_cast<uint32_t>(param.reduceType);
    dfxInfo.dataType = static_cast<uint32_t>(param.dataType);
    dfxInfo.dataCount = param.count;
    dfxInfo.engine = ccuEngine;
    dfxInfo.cpuTsThread = param.cpuThread;
    dfxInfo.cpuWaitAicpuNotifyIdx = 0;
    dfxInfo.inputMemAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    dfxInfo.inputMemSize = count * sizeof(float);
    dfxInfo.outputMemAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    dfxInfo.outputMemSize = count * sizeof(float);
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, ccuEngine, &ctx, &size) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = size;
    } else {
        AlgResourceCtx resCtxHost;
        void *cclBufferAddr = nullptr;
        uint64_t cclBufferSize = 0;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};
        resCtxHost.ccuThread = param.cpuThread;
        std::vector<ChannelGroup> groups;
        CHK_RET(AcquireChannelGroups(comm, param.myRank, param.rankSize, groups));
        const bool singleGroup = groups[0].handles.empty() || groups[1].handles.empty();
        resCtxHost.threads.resize(singleGroup ? 1 : DIE_COUNT);
        resCtxHost.threads[0] = param.cpuThread;
        if (!singleGroup) {
            CHK_RET(HcclThreadAcquire(comm, ccuEngine, 1, 1, &resCtxHost.threads[1]));
        }
        CHK_RET(RegisterKernels(comm, param.myRank, param.rankSize, groups, resCtxHost));

        std::vector<char> sequence = resCtxHost.Serialize();
        param.ctxSize = sequence.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, ccuEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, ccuEngine, param.tag, sequence.data(), sequence.size(), 0));
    }
    return ops_hccl::ExecOp(param);
}
