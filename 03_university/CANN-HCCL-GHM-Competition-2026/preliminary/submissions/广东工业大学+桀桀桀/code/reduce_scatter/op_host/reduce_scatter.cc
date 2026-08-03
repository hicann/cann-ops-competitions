/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0.
 */

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
constexpr uint32_t THREAD_NUM = 8;
constexpr uint32_t NOTIFY_NUM_PER_THREAD = 8;

bool IsAicpuTsProtocol(uint32_t protocol)
{
    return protocol == COMM_PROTOCOL_UBOE || protocol == COMM_PROTOCOL_PCIE || protocol == COMM_PROTOCOL_ROCE ||
        protocol == COMM_PROTOCOL_UBC_CTP || protocol == COMM_PROTOCOL_UBC_TP || protocol == COMM_PROTOCOL_HCCS;
}

HcclResult AcquireDesc(HcclComm comm, uint32_t srcRank, uint32_t dstRank, HcclChannelDesc *desc)
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));

    CHK_RET(HcclChannelDescInit(desc, 1));
    for (uint32_t layerIdx = 0; layerIdx < netLayerNum; ++layerIdx) {
        uint32_t listSize = 0;
        CommLink *linkList = nullptr;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayers[layerIdx], srcRank, dstRank, &linkList, &listSize));
        for (uint32_t linkIdx = 0; linkIdx < listSize; ++linkIdx) {
            const CommLink &link = linkList[linkIdx];
            if (!IsAicpuTsProtocol(link.dstEndpointDesc.protocol)) {
                continue;
            }
            desc->remoteRank = dstRank;
            desc->notifyNum = CHANNEL_NOTIFY_NUM;
            desc->channelProtocol = link.linkAttr.linkProtocol;
            desc->localEndpoint.protocol = link.srcEndpointDesc.protocol;
            desc->localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
            desc->localEndpoint.loc = link.srcEndpointDesc.loc;
            desc->remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
            desc->remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
            desc->remoteEndpoint.loc = link.dstEndpointDesc.loc;
            return HCCL_SUCCESS;
        }
    }
    HCCL_ERROR("AcquireDesc: no AICPU_TS link between rank[%u] and rank[%u]", srcRank, dstRank);
    return HCCL_E_INTERNAL;
}

HcclResult AcquireChannels(HcclComm comm, const OpParam &param, AlgResourceCtx &resource)
{
    if (param.rankSize <= 1) {
        return HCCL_SUCCESS;
    }
    std::vector<uint32_t> peers;
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer != param.myRank) {
            peers.push_back(peer);
        }
    }

    const uint32_t channelNum = peers.size();
    std::vector<HcclChannelDesc> desc(channelNum);
    std::vector<ChannelHandle> channels(channelNum);
    for (uint32_t index = 0; index < channelNum; ++index) {
        CHK_RET(AcquireDesc(comm, param.myRank, peers[index], &desc[index]));
    }
    CHK_RET(HcclChannelAcquire(comm, COMM_ENGINE_AICPU_TS, desc.data(), channelNum, channels.data()));
    for (uint32_t i = 0; i < channelNum; ++i) {
        void *cclBuffer = nullptr;
        uint64_t cclBufferSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, channels[i], &cclBuffer, &cclBufferSize));
        resource.channels.push_back(ChannelInfo{desc[i].remoteRank, CHANNEL_NOTIFY_NUM, channels[i],
            CommBuffer{cclBuffer, cclBufferSize}});
    }
    return HCCL_SUCCESS;
}

HcclResult CreateResources(HcclComm comm, OpParam &param, CommEngine aicpuTsEngine, CommEngine cpuTsEngine)
{
    AlgResourceCtx resource;
    void *localBuffer = nullptr;
    uint64_t localBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &localBuffer, &localBufferSize));
    resource.localBuffer = CommBuffer{localBuffer, localBufferSize};

    resource.threads.resize(THREAD_NUM);
    CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, THREAD_NUM, NOTIFY_NUM_PER_THREAD, resource.threads.data()));
    resource.aicpuThread = resource.threads[0];
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resource.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));
    CHK_RET(AcquireChannels(comm, param, resource));

    std::vector<char> serialized = resource.Serialize();
    param.ctxSize = serialized.size();
    CHK_RET(HcclEngineCtxCreate(comm, param.tag, aicpuTsEngine, param.ctxSize, &param.resCtx));
    CHK_RET(HcclEngineCtxCopy(comm, aicpuTsEngine, param.tag, serialized.data(), param.ctxSize, 0));

    void *hostCtx = nullptr;
    const uint64_t hostCtxSize = sizeof(ThreadHandle);
    CHK_RET(HcclEngineCtxCreate(comm, param.tag, cpuTsEngine, hostCtxSize, &hostCtx));
    CHK_RET(HcclEngineCtxCopy(comm, cpuTsEngine, param.tag, &resource.aicpuThread, hostCtxSize, 0));
    return HCCL_SUCCESS;
}
} // namespace

extern "C" HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    CHK_PRT_RET(SIZE_TABLE.find(dataType) == SIZE_TABLE.end(), HCCL_ERROR("unsupported data type[%d]", dataType),
        HCCL_E_PARA);
    CHK_PRT_RET(op == HCCL_REDUCE_RESERVED, HCCL_ERROR("unsupported reduce operation[%d]", op), HCCL_E_PARA);

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
    CHK_PRT_RET(param.myRank >= param.rankSize || param.rankSize == 0,
        HCCL_ERROR("invalid rank[%u] or rank size[%u]", param.myRank, param.rankSize), HCCL_E_INTERNAL);

    const CommEngine aicpuTsEngine = CommEngine::COMM_ENGINE_AICPU_TS;
    const CommEngine cpuTsEngine = CommEngine::COMM_ENGINE_CPU_TS;
    CHK_RET(HcclThreadAcquireWithStream(comm, cpuTsEngine, stream, 1, &param.cpuThread));
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &param.cpuThread, aicpuTsEngine, &param.cpuThreadOnAicpu));

    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    if (HcclEngineCtxGet(comm, param.tag, aicpuTsEngine, &ctx, &ctxSize) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = ctxSize;
        void *hostCtx = nullptr;
        uint64_t hostCtxSize = 0;
        CHK_RET(HcclEngineCtxGet(comm, param.tag, cpuTsEngine, &hostCtx, &hostCtxSize));
        ThreadHandle *aicpuThread = static_cast<ThreadHandle *>(hostCtx);
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));
    } else {
        CHK_RET(CreateResources(comm, param, aicpuTsEngine, cpuTsEngine));
    }
    return ops_hccl::LaunchAICPUKernel(param, stream);
}
