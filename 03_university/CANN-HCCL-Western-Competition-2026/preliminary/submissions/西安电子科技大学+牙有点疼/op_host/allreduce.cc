/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;

bool GetDataTypeSize(HcclDataType dataType, uint32_t &size)
{
    switch (dataType) {
        case HCCL_DATA_TYPE_INT8:
        case HCCL_DATA_TYPE_UINT8:
        case HCCL_DATA_TYPE_HIF8:
        case HCCL_DATA_TYPE_FP8E4M3:
        case HCCL_DATA_TYPE_FP8E5M2:
        case HCCL_DATA_TYPE_FP8E8M0:
            size = 1;
            return true;
        case HCCL_DATA_TYPE_INT16:
        case HCCL_DATA_TYPE_UINT16:
        case HCCL_DATA_TYPE_FP16:
        case HCCL_DATA_TYPE_BFP16:
            size = 2;
            return true;
        case HCCL_DATA_TYPE_INT32:
        case HCCL_DATA_TYPE_UINT32:
        case HCCL_DATA_TYPE_FP32:
            size = 4;
            return true;
        case HCCL_DATA_TYPE_INT64:
        case HCCL_DATA_TYPE_UINT64:
        case HCCL_DATA_TYPE_FP64:
            size = 8;
            return true;
        case HCCL_DATA_TYPE_INT128:
            size = 16;
            return true;
        default:
            size = 0;
            return false;
    }
}

bool IsReduceOpSupported(HcclReduceOp op)
{
    return op == HCCL_REDUCE_SUM || op == HCCL_REDUCE_PROD || op == HCCL_REDUCE_MAX || op == HCCL_REDUCE_MIN;
}

const CommLink *SelectLink(CommLink *linkList, uint32_t listSize)
{
    if (linkList == nullptr || listSize == 0) {
        return nullptr;
    }

    for (uint32_t idx = 0; idx < listSize; idx++) {
        if (linkList[idx].linkAttr.linkProtocol == CommProtocol::COMM_PROTOCOL_UBC_CTP) {
            return &linkList[idx];
        }
    }

    for (uint32_t idx = 0; idx < listSize; idx++) {
        if (linkList[idx].linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_RESERVED) {
            return &linkList[idx];
        }
    }
    return nullptr;
}

HcclResult GetNetLayers(HcclComm comm, std::vector<uint32_t> &netLayers)
{
    uint32_t *layers = nullptr;
    uint32_t layerNum = 0;
    HcclResult ret = HcclRankGraphGetLayers(comm, &layers, &layerNum);
    if (ret == HCCL_SUCCESS && layers != nullptr && layerNum > 0) {
        netLayers.assign(layers, layers + layerNum);
        std::sort(netLayers.begin(), netLayers.end());
        return HCCL_SUCCESS;
    }

    HCCL_WARNING("[GetNetLayers] failed to get rank graph layers, ret[%d], fallback to layer 0", ret);
    netLayers.push_back(0);
    return HCCL_SUCCESS;
}

HcclResult FillChannelDesc(HcclComm comm, uint32_t srcRank, uint32_t dstRank, HcclChannelDesc &desc)
{
    std::vector<uint32_t> netLayers;
    CHK_RET(GetNetLayers(comm, netLayers));

    HcclResult lastRet = HCCL_E_NOT_FOUND;
    for (uint32_t netLayer : netLayers) {
        CommLink *linkList = nullptr;
        uint32_t listSize = 0;
        HcclResult ret = HcclRankGraphGetLinks(comm, netLayer, srcRank, dstRank, &linkList, &listSize);
        if (ret != HCCL_SUCCESS) {
            lastRet = ret;
            continue;
        }

        const CommLink *link = SelectLink(linkList, listSize);
        if (link == nullptr) {
            lastRet = HCCL_E_NOT_FOUND;
            continue;
        }

        CHK_RET(HcclChannelDescInit(&desc, 1));
        desc.remoteRank = dstRank;
        desc.notifyNum = CHANNEL_NOTIFY_NUM;
        desc.channelProtocol = link->linkAttr.linkProtocol;
        desc.localEndpoint.protocol = link->srcEndpointDesc.protocol;
        desc.localEndpoint.commAddr = link->srcEndpointDesc.commAddr;
        desc.localEndpoint.loc = link->srcEndpointDesc.loc;
        desc.remoteEndpoint.protocol = link->dstEndpointDesc.protocol;
        desc.remoteEndpoint.commAddr = link->dstEndpointDesc.commAddr;
        desc.remoteEndpoint.loc = link->dstEndpointDesc.loc;
        return HCCL_SUCCESS;
    }

    HCCL_ERROR("[FillChannelDesc] no link found between rank %u and rank %u", srcRank, dstRank);
    return lastRet;
}

HcclResult AcquireChannels(HcclComm comm, const OpParam &param, CommEngine engine, AlgResourceCtx &resCtx)
{
    if (param.rankSize <= 1) {
        return HCCL_SUCCESS;
    }

    const uint32_t channelNum = param.rankSize - 1;
    std::vector<HcclChannelDesc> descs(channelNum);
    std::vector<ChannelHandle> handles(channelNum);
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

    CHK_RET(HcclChannelAcquire(comm, engine, descs.data(), channelNum, handles.data()));

    resCtx.channels.reserve(channelNum);
    for (uint32_t i = 0; i < channelNum; i++) {
        void *remoteCclBuffer = nullptr;
        uint64_t remoteCclBufferSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, handles[i], &remoteCclBuffer, &remoteCclBufferSize));

        ChannelInfo channel;
        channel.remoteRank = remoteRanks[i];
        channel.notifyNum = CHANNEL_NOTIFY_NUM;
        channel.handle = handles[i];
        channel.remoteCclMem = CommBuffer{remoteCclBuffer, remoteCclBufferSize};
        resCtx.channels.push_back(channel);
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

    uint32_t dataTypeSize = 0;
    if (!GetDataTypeSize(dataType, dataTypeSize)) {
        HCCL_ERROR("[HcclAllReduce] unsupported dataType[%d]", dataType);
        return HCCL_E_NOT_SUPPORT;
    }
    if (!IsReduceOpSupported(op)) {
        HCCL_ERROR("[HcclAllReduce] unsupported reduce op[%d]", op);
        return HCCL_E_NOT_SUPPORT;
    }
    if (dataTypeSize != 0 && count > std::numeric_limits<uint64_t>::max() / dataTypeSize) {
        HCCL_ERROR("[HcclAllReduce] count[%llu] overflows byte size", static_cast<unsigned long long>(count));
        return HCCL_E_PARA;
    }
    if (count == 0) {
        return HCCL_SUCCESS;
    }

    // 构造算子参数
    OpParam param;
    sprintf(param.tag, "%s", "hccl_custom_allreduce");
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = count;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;

    // 注册算子信息
    HcclDfxOpInfo dfxInfo = {};
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));

    // ==============================================
    // STEP 1: 解析拓扑信息
    // ==============================================
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    if (param.rankSize == 0 || param.myRank >= param.rankSize) {
        HCCL_ERROR("[HcclAllReduce] invalid rank info, myRank[%u], rankSize[%u]", param.myRank, param.rankSize);
        return HCCL_E_PARA;
    }

    // ==============================================
    // STEP 2: 创建资源
    // ==============================================
    CommEngine aicpuTsEngine = CommEngine::COMM_ENGINE_AICPU_TS;
    CommEngine cpuTsEngine = CommEngine::COMM_ENGINE_CPU_TS;

    // ==============================================
    // STEP 2.1: 申请用于 Host/Device 同步的通信资源
    // ==============================================
    // 将用户传入的 stream 转换为 thread，并申请 Notify；同时导出为 AICPU 上可用的 thread
    CHK_RET(HcclThreadAcquireWithStream(comm, cpuTsEngine, stream, 1, &param.cpuThread));
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &param.cpuThread, aicpuTsEngine, &param.cpuThreadOnAicpu));
    dfxInfo.opType = static_cast<uint32_t>(param.opType);
    dfxInfo.reduceOp = static_cast<uint32_t>(param.reduceType);
    dfxInfo.dataType = static_cast<uint32_t>(param.dataType);
    dfxInfo.dataCount = param.count;
    dfxInfo.engine = aicpuTsEngine;
    dfxInfo.cpuTsThread = param.cpuThread;
    dfxInfo.cpuWaitAicpuNotifyIdx = 0;
    dfxInfo.inputMemAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    dfxInfo.inputMemSize = count * dataTypeSize;
    dfxInfo.outputMemAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    dfxInfo.outputMemSize = count * dataTypeSize;
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, aicpuTsEngine, &ctx, &size) == HCCL_SUCCESS) {
        // AICPU 资源已经存在，复用资源
        HCCL_INFO("Engine context already exists");
        param.resCtx = ctx;
        param.ctxSize = size;

        // Host 资源已经存在，复用资源
        void *hostCtx = nullptr;
        uint64_t hostCtxSize = 0;
        CHK_RET(HcclEngineCtxGet(comm, param.tag, cpuTsEngine, &hostCtx, &hostCtxSize));
        ThreadHandle *aicpuThread = static_cast<ThreadHandle *>(hostCtx);
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));
    } else {
        // Device 资源不存在，资源构建
        AlgResourceCtx resCtxHost;

        // 从通信域获取 HCCL Buffer（Device上的内存，默认总大小400MB）
        void *cclBufferAddr;
        uint64_t cclBufferSize;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        // ==============================================
        // STEP 2.2: 申请资源Thread和Channel
        // ==============================================

        // 创建 AICPU_TS 通信引擎上的 thread 资源
        uint32_t threadNum = param.rankSize > 1 ? param.rankSize - 1 : 1;
        uint32_t notifyNumPerThread = threadNum;

        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, notifyNumPerThread, resCtxHost.threads.data()));
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        CHK_RET(AcquireChannels(comm, param, aicpuTsEngine, resCtxHost));

        // ==============================================
        // STEP 2.3: 申请通信引擎上下文
        // ==============================================
        // 申请 AICPU 通信引擎上下文，存放 AlgResourceCtx 信息
        std::vector<char> seq = resCtxHost.Serialize();
        uint64_t seqSize = seq.size();
        param.ctxSize = seqSize;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, aicpuTsEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, aicpuTsEngine, param.tag, seq.data(), seqSize, 0));
        // 申请 CPU 通信引擎上下文，存放 aicpuThread 句柄
        void *hostCtx = nullptr;
        uint64_t hostCtxSize = sizeof(ThreadHandle);
        const void *aicpuThreadPtr = static_cast<const void *>(&resCtxHost.aicpuThread);
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, cpuTsEngine, hostCtxSize, &hostCtx));
        CHK_RET(HcclEngineCtxCopy(comm, cpuTsEngine, param.tag, aicpuThreadPtr, hostCtxSize, 0));
    }

    // ==============================================
    // STEP 3: 下发 AICPU Kernel
    // ==============================================
    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}
