/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <vector>

#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>

#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"
#include "log.h"

namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;
constexpr CommProtocol COMM_PROTOCOL_UBOE_COMPAT = static_cast<CommProtocol>(7);
constexpr CommProtocol COMM_PROTOCOL_UBG_COMPAT = static_cast<CommProtocol>(9);

HcclResult FillChannelDescFromLink(uint32_t dstRank, const CommLink &link, HcclChannelDesc &desc)
{
    CHK_RET(HcclChannelDescInit(&desc, 1));
    desc.remoteRank = dstRank;
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

HcclResult FillChannelDesc(HcclComm comm, uint32_t srcRank, uint32_t dstRank, HcclChannelDesc &desc)
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));

    const CommProtocol expectedProtocols[] = {
        CommProtocol::COMM_PROTOCOL_UBC_CTP,
        CommProtocol::COMM_PROTOCOL_UBC_TP,
        CommProtocol::COMM_PROTOCOL_PCIE,
        COMM_PROTOCOL_UBOE_COMPAT,
        COMM_PROTOCOL_UBG_COMPAT,
    };

    for (uint32_t layerIdx = 0; layerIdx < netLayerNum; layerIdx++) {
        uint32_t listSize = 0;
        CommLink *linkList = nullptr;
        HcclResult ret = HcclRankGraphGetLinks(comm, netLayers[layerIdx], srcRank, dstRank, &linkList, &listSize);
        if (ret != HCCL_SUCCESS) {
            HCCL_DEBUG("[FillChannelDesc] no link in layer %u between rank %u and rank %u", netLayers[layerIdx],
                srcRank, dstRank);
            continue;
        }

        for (CommProtocol protocol : expectedProtocols) {
            for (uint32_t idx = 0; idx < listSize; idx++) {
                if (linkList[idx].linkAttr.linkProtocol != protocol) {
                    continue;
                }
                CHK_RET(FillChannelDescFromLink(dstRank, linkList[idx], desc));
                HCCL_INFO("[FillChannelDesc] select protocol %d in layer %u between rank %u and rank %u",
                    static_cast<int>(desc.channelProtocol), netLayers[layerIdx], srcRank, dstRank);
                return HCCL_SUCCESS;
            }
        }
    }

    HCCL_ERROR("[FillChannelDesc] no available AICPU_TS link between rank %u and rank %u", srcRank, dstRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult InitThreads(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtxHost)
{
    uint32_t channelNum = param.rankSize > 0 ? param.rankSize - 1 : 0;
    uint32_t threadNum = channelNum + 1;
    uint32_t notifyNumPerThread = channelNum + 1;

    resCtxHost.threads.resize(threadNum);
    CHK_RET(
        HcclThreadAcquire(comm, CommEngine::COMM_ENGINE_AICPU_TS, threadNum, notifyNumPerThread, resCtxHost.threads.data()));
    resCtxHost.aicpuThread = resCtxHost.threads[0];
    return HCCL_SUCCESS;
}

HcclResult InitChannels(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtxHost)
{
    uint32_t channelNum = param.rankSize > 0 ? param.rankSize - 1 : 0;
    if (channelNum == 0) {
        return HCCL_SUCCESS;
    }

    std::vector<HcclChannelDesc> descs(channelNum);
    std::vector<ChannelHandle> channelHandles(channelNum);
    std::vector<uint32_t> remoteRanks(channelNum);

    uint32_t idx = 0;
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; remoteRank++) {
        if (remoteRank == param.myRank) {
            continue;
        }
        CHK_RET(FillChannelDesc(comm, param.myRank, remoteRank, descs[idx]));
        remoteRanks[idx] = remoteRank;
        idx++;
    }

    CHK_RET(HcclChannelAcquire(
        comm, CommEngine::COMM_ENGINE_AICPU_TS, descs.data(), channelNum, channelHandles.data()));

    resCtxHost.channels.reserve(channelNum);
    for (uint32_t i = 0; i < channelNum; i++) {
        void *remoteCclBufferAddr = nullptr;
        uint64_t remoteCclBufferSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, channelHandles[i], &remoteCclBufferAddr, &remoteCclBufferSize));

        ChannelInfo channel;
        channel.remoteRank = remoteRanks[i];
        channel.notifyNum = CHANNEL_NOTIFY_NUM;
        channel.handle = channelHandles[i];
        channel.remoteCclMem = CommBuffer{remoteCclBufferAddr, remoteCclBufferSize};
        resCtxHost.channels.push_back(channel);
    }
    return HCCL_SUCCESS;
}

HcclResult InitAlgResource(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtxHost)
{
    void *cclBufferAddr = nullptr;
    uint64_t cclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

    CHK_RET(InitThreads(comm, param, resCtxHost));
    CHK_RET(InitChannels(comm, param, resCtxHost));
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclAllGather(
    void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param;
    int ret = std::snprintf(param.tag, sizeof(param.tag), "%s", "hccl_custom_allgather");
    if (ret <= 0 || static_cast<size_t>(ret) >= sizeof(param.tag)) {
        HCCL_ERROR("[HcclAllGather] failed to fill tag");
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

    CommEngine aicpuTsEngine = CommEngine::COMM_ENGINE_AICPU_TS;
    CommEngine cpuTsEngine = CommEngine::COMM_ENGINE_CPU_TS;

    CHK_RET(HcclThreadAcquireWithStream(comm, cpuTsEngine, stream, 1, &param.cpuThread));
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &param.cpuThread, aicpuTsEngine, &param.cpuThreadOnAicpu));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, aicpuTsEngine, &ctx, &size) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = size;

        void *hostCtx = nullptr;
        uint64_t hostCtxSize = sizeof(ThreadHandle);
        CHK_RET(HcclEngineCtxGet(comm, param.tag, cpuTsEngine, &hostCtx, &hostCtxSize));
        ThreadHandle *aicpuThread = static_cast<ThreadHandle *>(hostCtx);
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));
    } else {
        AlgResourceCtx resCtxHost;
        CHK_RET(InitAlgResource(comm, param, resCtxHost));

        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        std::vector<char> seq = resCtxHost.Serialize();
        uint64_t seqSize = seq.size();
        param.ctxSize = seqSize;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, aicpuTsEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, aicpuTsEngine, param.tag, seq.data(), seqSize, 0));

        void *hostCtx = nullptr;
        uint64_t hostCtxSize = sizeof(ThreadHandle);
        const void *aicpuThreadPtr = static_cast<const void *>(&resCtxHost.aicpuThread);
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, cpuTsEngine, hostCtxSize, &hostCtx));
        CHK_RET(HcclEngineCtxCopy(comm, cpuTsEngine, param.tag, aicpuThreadPtr, hostCtxSize, 0));
    }

    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}
