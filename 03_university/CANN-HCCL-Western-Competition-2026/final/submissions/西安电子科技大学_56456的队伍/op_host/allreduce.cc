/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

// ===== 提交版本 v63-final | v60安全基线 + peer就绪流水规约 | 2026-07-24 =====
// v63-final: notify数量和槽位完全保持v60，仅缩短peer就绪后的等待路径
// allreduce.cc: 单槽位模式, CHANNEL_NOTIFY_NUM=3 (实际只用slot 0)

#include <cstdio>
#include <map>
#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_ccu_res.h>
#include <hcomm/hcomm_primitives.h>
#include <ccu/ccu_res.h>
#include <ccu/ccu_launch.h>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "exec_op.h"
#include "ccu_kernel.h"

#ifndef ENDPOINT_ATTR_DIE_ID
#define ENDPOINT_ATTR_DIE_ID 1
#endif

namespace {

// Channel 上 notify 寄存器数量
constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;

HcclResult BuildChannelsByDie(HcclComm comm, const OpParam &param,
    std::map<uint32_t, std::vector<ChannelHandle>> &channelsByDie)
{
    channelsByDie.clear();

    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; remoteRank++) {
        if (remoteRank == param.myRank) {
            continue;
        }

        bool found = false;
        for (uint32_t netLayer = 0; netLayer < 4 && !found; netLayer++) {
            uint32_t listSize = 0;
            CommLink *linkList = nullptr;
            HcclResult ret = HcclRankGraphGetLinks(comm, netLayer, param.myRank, remoteRank,
                &linkList, &listSize);
            if (ret != HCCL_SUCCESS || listSize == 0) {
                continue;
            }

            for (uint32_t idx = 0; idx < listSize; idx++) {
                auto proto = linkList[idx].linkAttr.linkProtocol;
                if (proto == CommProtocol::COMM_PROTOCOL_UBC_CTP ||
                    proto == CommProtocol::COMM_PROTOCOL_UBC_TP) {
                    CommLink &link = linkList[idx];

                    uint32_t dieId = 0;
                    HcclResult dieRet = HcclRankGraphGetEndpointInfo(comm, param.myRank,
                        &link.srcEndpointDesc, static_cast<EndpointAttr>(ENDPOINT_ATTR_DIE_ID),
                        sizeof(uint32_t), &dieId);
                    if (dieRet != HCCL_SUCCESS) {
                        HCCL_ERROR("[BuildChannelsByDie] GetEndpointInfo dieId failed, "
                            "rank %u -> rank %u, ret=%d",
                            param.myRank, remoteRank, dieRet);
                        return dieRet;
                    }

                    HcclChannelDesc desc;
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

                    ChannelHandle ch;
                    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, &desc, 1, &ch));
                    channelsByDie[dieId].push_back(ch);
                    found = true;
                    HCCL_INFO("[BuildChannelsByDie] rank %u -> rank %u via netLayer=%u, "
                        "protocol=%d, dieId=%u",
                        param.myRank, remoteRank, netLayer, static_cast<int>(proto), dieId);
                    break;
                }
            }
        }
        if (!found) {
            HCCL_ERROR("[BuildChannelsByDie] no usable channel from rank %u to rank %u",
                param.myRank, remoteRank);
            return HCCL_E_NOT_FOUND;
        }
    }

    size_t totalChannels = 0;
    for (const auto &[dieId, chs] : channelsByDie) {
        (void)dieId;
        totalChannels += chs.size();
    }
    if (totalChannels != static_cast<size_t>(param.rankSize - 1)) {
        HCCL_ERROR("[BuildChannelsByDie] channel count mismatch: got=%zu expected=%u",
            totalChannels, param.rankSize - 1);
        return HCCL_E_INTERNAL;
    }

    HCCL_INFO("[BuildChannelsByDie] rank %u: %zu die groups (rankSize=%u)",
        param.myRank, channelsByDie.size(), param.rankSize);
    for (const auto &[dieId, chs] : channelsByDie) {
        HCCL_INFO("[BuildChannelsByDie] dieId=%u: %zu channels", dieId, chs.size());
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterCcuKernels(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtxHost,
    const std::map<uint32_t, std::vector<ChannelHandle>> &channelsByDie)
{
    CcuInsHandle insHandle{};
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1,
        HCCL_ERROR("[RegisterCcuKernels] HcclCommQueryCcuIns fail! insNum is [%u]", insNum),
        HCCL_E_INTERNAL);

    constexpr uint32_t kernelArgNum = 1;
    uint32_t phase = 0;

    for (const auto &[dieId, channels] : channelsByDie) {
        if (channels.empty()) {
            continue;
        }
        if (channels.size() > ALLREDUCE_MAX_CHANNELS) {
            HCCL_ERROR("[RegisterCcuKernels] die %u has too many channels: %zu > %u",
                dieId, channels.size(), ALLREDUCE_MAX_CHANNELS);
            return HCCL_E_INTERNAL;
        }

        auto kernelArg = std::make_shared<ops_hccl::CcuKernelArg>();
        kernelArg->rankSize = param.rankSize;
        kernelArg->rankId = param.myRank;
        kernelArg->reduceOp = param.reduceType;
        kernelArg->dataType = param.dataType;
        kernelArg->phase = phase;
        for (size_t i = 0; i < channels.size(); ++i) {
            kernelArg->channels[i] = channels[i];
        }
        kernelArg->channelCount = static_cast<uint32_t>(channels.size());

        const void *kernelArgsArr[] = { kernelArg.get() };
        CcuKernelHandle kernelHandle{};

        CcuResult regStartRet = HcommCcuKernelRegisterStart(insHandle);
        if (regStartRet != CCU_SUCCESS) {
            HCCL_ERROR("[RegisterCcuKernels] die %u register start failed: %d", dieId, regStartRet);
            return ConvertCcuToHccl(regStartRet);
        }

        CcuResult regRet = HcommCcuKernelRegister(insHandle, dieId,
            ops_hccl::CCU_KERNEL_NAME,
            reinterpret_cast<void *>(ops_hccl::CcuKernel),
            kernelArgsArr, kernelArgNum, &kernelHandle);
        if (regRet != CCU_SUCCESS) {
            HCCL_ERROR("[RegisterCcuKernels] die %u register failed: %d", dieId, regRet);
            return ConvertCcuToHccl(regRet);
        }

        CcuResult regEndRet = HcommCcuKernelRegisterEnd(insHandle);
        if (regEndRet != CCU_SUCCESS) {
            HCCL_ERROR("[RegisterCcuKernels] die %u register end failed: %d", dieId, regEndRet);
            return ConvertCcuToHccl(regEndRet);
        }

        resCtxHost.ccuKernels.push_back(kernelHandle);
        HCCL_INFO("[RegisterCcuKernels] die %u (phase=%u) registered, channels=%zu",
            dieId, phase, channels.size());

        phase = 1;
    }

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
    if (dataType != HCCL_DATA_TYPE_FP32 || op != HCCL_REDUCE_SUM) {
        HCCL_ERROR("[HcclAllReduce] only FP32 SUM is supported, dataType=%d op=%d",
            static_cast<int>(dataType), static_cast<int>(op));
        return HCCL_E_NOT_SUPPORT;
    }

    OpParam param;
    int tagRet = snprintf(param.tag, sizeof(param.tag), "hccl_custom_allreduce_%d_%d",
        static_cast<int>(dataType), static_cast<int>(op));
    if (tagRet < 0 || static_cast<size_t>(tagRet) >= sizeof(param.tag)) {
        HCCL_ERROR("[HcclAllReduce] failed to build resource tag");
        return HCCL_E_INTERNAL;
    }
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = count;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    param.reduceType = op;

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));

    const CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;

    AlgResourceCtx resCtxHost;
    void *cclBufferAddr = nullptr;
    uint64_t cclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    resCtxHost.cclMem = CommBuffer{cclBufferAddr, cclBufferSize};

    resCtxHost.notifyNumOnMainThread = 0;
    CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, stream, 0, &param.cpuThread));
    resCtxHost.threads.push_back(param.cpuThread);

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, ccuEngine, &ctx, &size) == HCCL_SUCCESS) {
        HCCL_INFO("Engine context already exists");
        param.resCtx = ctx;
        param.ctxSize = size;
    } else {
        if (param.rankSize > 1) {
            std::map<uint32_t, std::vector<ChannelHandle>> channelsByDie;
            CHK_RET(BuildChannelsByDie(comm, param, channelsByDie));
            CHK_RET(RegisterCcuKernels(comm, param, resCtxHost, channelsByDie));
        } else {
            HCCL_INFO("[HcclAllReduce] RankSize == 1, skip channel and ccu kernel allocation.");
        }

        std::vector<char> seq = resCtxHost.Serialize();
        const uint64_t seqSize = seq.size();
        param.ctxSize = seqSize;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, ccuEngine, param.ctxSize, &param.resCtx));
        if (seqSize > 0) {
            CHK_RET(HcclEngineCtxCopy(comm, ccuEngine, param.tag, seq.data(), seqSize, 0));
        }
    }

    CHK_RET(ops_hccl::ExecOp(param));
    return HCCL_SUCCESS;
}
