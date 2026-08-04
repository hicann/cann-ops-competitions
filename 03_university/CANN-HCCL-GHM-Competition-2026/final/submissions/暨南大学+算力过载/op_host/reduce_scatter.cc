/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <ccu/ccu_launch.h>
#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>

#include <cstdio>
#include <utility>

#include "../op_kernel_ccu/ccu_kernel.h"
#include "common.h"
#include "custom.h"
#include "exec_op.h"
#include "hccl.h"
#include "log.h"

namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 1;

HcclResult FindCommonUbcLayer(HcclComm comm, uint32_t myRank, uint32_t rankSize,
    uint32_t &selectedLayer, std::vector<CommLink> &selectedLinks)
{
    uint32_t *layers = nullptr;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layers, &layerNum));
    const std::vector<uint32_t> candidateLayers(layers, layers + layerNum);
    for (const uint32_t layer : candidateLayers) {
        std::vector<CommLink> links;
        bool isCommonLayer = true;
        for (uint32_t remoteRank = 0; remoteRank < rankSize; ++remoteRank) {
            if (remoteRank == myRank) {
                continue;
            }
            CommLink *linkList = nullptr;
            uint32_t linkNum = 0;
            CHK_RET(HcclRankGraphGetLinks(comm, layer, myRank, remoteRank, &linkList, &linkNum));
            const CommLink *ubcLink = nullptr;
            for (uint32_t index = 0; index < linkNum; ++index) {
                if (linkList[index].linkAttr.linkProtocol == COMM_PROTOCOL_UBC_CTP) {
                    ubcLink = &linkList[index];
                    break;
                }
            }
            if (ubcLink == nullptr) {
                isCommonLayer = false;
                break;
            }
            links.push_back(*ubcLink);
        }
        if (isCommonLayer && links.size() == rankSize - 1) {
            selectedLayer = layer;
            selectedLinks = std::move(links);
            return HCCL_SUCCESS;
        }
    }
    HCCL_ERROR("FindCommonUbcLayer: no UBC_CTP layer connects all ranks");
    return HCCL_E_NOT_FOUND;
}

HcclResult AcquireChannels(HcclComm comm, const OpParam &param, const std::vector<CommLink> &links,
    std::vector<ChannelHandle> &handles)
{
    const uint32_t channelNum = param.rankSize - 1;
    CHK_PRT_RET(links.size() != channelNum,
        HCCL_ERROR("AcquireChannels: expected[%u] links, got[%zu]", channelNum, links.size()), HCCL_E_INTERNAL);
    std::vector<HcclChannelDesc> descs(channelNum);
    CHK_RET(HcclChannelDescInit(descs.data(), channelNum));
    uint32_t channelIndex = 0;
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }
        const CommLink &link = links[channelIndex];
        HcclChannelDesc &desc = descs[channelIndex];
        desc.remoteRank = remoteRank;
        desc.notifyNum = CHANNEL_NOTIFY_NUM;
        desc.channelProtocol = link.linkAttr.linkProtocol;
        desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
        desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
        desc.localEndpoint.loc = link.srcEndpointDesc.loc;
        desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
        desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
        desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
        ++channelIndex;
    }
    handles.resize(channelNum);
    return HcclChannelAcquire(
        comm, CommEngine::COMM_ENGINE_CCU, descs.data(), channelNum, handles.data());
}

HcclResult RegisterKernel(HcclComm comm, const OpParam &param, const std::vector<ChannelHandle> &channels,
    AlgResourceCtx &resCtxHost)
{
    CcuKernelInfo kernelInfo;
    snprintf(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "%s", "CcuReduceScatterKernel");
    kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuReduceScatterKernel);
    auto kernelArg = std::make_shared<ops_hccl::ReduceScatterKernelArg>();
    kernelArg->rankSize = param.rankSize;
    kernelArg->rankId = param.myRank;
    kernelArg->channelCount = static_cast<uint32_t>(channels.size());
    for (uint32_t index = 0; index < kernelArg->channelCount; ++index) {
        kernelArg->channels[index] = channels[index];
    }
    kernelInfo.SetKernelArg(kernelArg);

    CcuInsHandle insHandle = 0;
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1, HCCL_ERROR("RegisterKernel: invalid CCU instance count[%u]", insNum), HCCL_E_INTERNAL);

    CcuResult ccuRet = HcommCcuKernelRegisterStart(insHandle);
    if (ccuRet != CCU_SUCCESS) {
        return ConvertCcuToHccl(ccuRet);
    }
    const void *kernelArgs[] = {kernelInfo.kernelArg};
    CcuKernelHandle kernelHandle = 0;
    ccuRet = HcommCcuKernelRegister(insHandle, 0, kernelInfo.kernelFuncName, kernelInfo.kernelFunc,
        kernelArgs, 1, &kernelHandle);
    if (ccuRet != CCU_SUCCESS) {
        return ConvertCcuToHccl(ccuRet);
    }
    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    if (ccuRet != CCU_SUCCESS) {
        return ConvertCcuToHccl(ccuRet);
    }
    resCtxHost.ccuKernels.push_back(kernelHandle);
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param;
    sprintf(param.tag, "%s", "hccl_custom_reduce_scatter");
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;

    HcclDfxOpInfo dfxInfo;
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE,
        HCCL_ERROR("HcclReduceScatter: invalid rank size[%u]", param.rankSize), HCCL_E_PARA);
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("HcclReduceScatter: unsupported data type[%d]", dataType), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(op != HCCL_REDUCE_SUM,
        HCCL_ERROR("HcclReduceScatter: unsupported reduce op[%d]", op), HCCL_E_NOT_SUPPORT);
    CHK_RET(HcclThreadAcquireWithStream(comm, CommEngine::COMM_ENGINE_CCU, stream, 0, &param.cpuThread));

    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    std::vector<char> invocationContext;
    const HcclResult ctxRet = HcclEngineCtxGet(comm, param.tag, CommEngine::COMM_ENGINE_CCU, &ctx, &ctxSize);
    if (ctxRet == HCCL_SUCCESS) {
        invocationContext.assign(static_cast<char *>(ctx), static_cast<char *>(ctx) + ctxSize);
        AlgResourceCtx resCtxHost;
        resCtxHost.DeSerialize(invocationContext);
        CHK_PRT_RET(resCtxHost.threads.empty(),
            HCCL_ERROR("HcclReduceScatter: cached context has no execution thread"), HCCL_E_INTERNAL);
        resCtxHost.threads[0] = param.cpuThread;
        invocationContext = resCtxHost.Serialize();
        param.resCtx = invocationContext.data();
        param.ctxSize = invocationContext.size();
    } else {
        const bool contextMissing = (ctxRet == HCCL_E_PARA || ctxRet == HCCL_E_NOT_FOUND);
        CHK_PRT_RET(!contextMissing,
            HCCL_ERROR("HcclReduceScatter: failed to query engine context[%d]", ctxRet), ctxRet);
        AlgResourceCtx resCtxHost;
        void *cclBufferAddr = nullptr;
        uint64_t cclBufferSize = 0;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};
        resCtxHost.threads.push_back(param.cpuThread);
        if (param.rankSize > 1) {
            uint32_t selectedLayer = 0;
            std::vector<CommLink> selectedLinks;
            CHK_RET(FindCommonUbcLayer(comm, param.myRank, param.rankSize, selectedLayer, selectedLinks));
            (void)selectedLayer;
            std::vector<ChannelHandle> channels;
            CHK_RET(AcquireChannels(comm, param, selectedLinks, channels));
            CHK_RET(RegisterKernel(comm, param, channels, resCtxHost));
        }
        const std::vector<char> sequence = resCtxHost.Serialize();
        param.ctxSize = sequence.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, CommEngine::COMM_ENGINE_CCU, param.ctxSize, &param.resCtx));
        const HcclResult copyRet = HcclEngineCtxCopy(
            comm, CommEngine::COMM_ENGINE_CCU, param.tag, sequence.data(), param.ctxSize, 0);
        if (copyRet != HCCL_SUCCESS) {
            (void)HcclEngineCtxDestroy(comm, param.tag, CommEngine::COMM_ENGINE_CCU);
            return copyRet;
        }
    }
    return ops_hccl::ExecOp(param);
}
