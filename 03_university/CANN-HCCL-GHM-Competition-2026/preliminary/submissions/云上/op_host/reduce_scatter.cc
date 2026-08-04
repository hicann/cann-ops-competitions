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

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

// The campus-2026 hcomm library exports this query, but its public toolkit
// header does not declare it yet. Querying the acquired handle is necessary
// because HcclChannelAcquire may reuse a channel created with fewer Notifys.
extern "C" HcommResult HcommChannelGetNotifyNum(ChannelHandle channelHandle, uint32_t *notifyNum);

namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;
constexpr uint32_t UNSUPPORTED_PROTOCOL_PRIORITY = 3;

uint32_t ProtocolPriority(CommProtocol protocol)
{
    switch (protocol) {
        case COMM_PROTOCOL_UBC_CTP:
            return 0;
        case COMM_PROTOCOL_UBC_TP:
            return 1;
        case COMM_PROTOCOL_UBOE:
            return 2;
        default:
            return UNSUPPORTED_PROTOCOL_PRIORITY;
    }
}

bool IsAicpuTsProtocol(CommProtocol protocol)
{
    return ProtocolPriority(protocol) < UNSUPPORTED_PROTOCOL_PRIORITY;
}

bool IsSupportedReduceOp(HcclReduceOp op)
{
    return op == HCCL_REDUCE_SUM || op == HCCL_REDUCE_MAX || op == HCCL_REDUCE_MIN;
}

HcclResult BuildPeerChannelDesc(
    HcclComm comm, const std::vector<uint32_t> &netLayers, uint32_t srcRank, uint32_t dstRank, HcclChannelDesc &desc)
{
    HcclResult firstQueryError = HCCL_SUCCESS;
    for (uint32_t netLayer : netLayers) {
        CommLink *linkList = nullptr;
        uint32_t listSize = 0;
        HcclResult ret = HcclRankGraphGetLinks(comm, netLayer, srcRank, dstRank, &linkList, &listSize);
        if (ret != HCCL_SUCCESS) {
            if (firstQueryError == HCCL_SUCCESS) {
                firstQueryError = ret;
            }
            HCCL_WARNING(
                "GetLinks failed, srcRank[%u], dstRank[%u], netLayer[%u], ret[%d]", srcRank, dstRank, netLayer, ret);
            continue;
        }
        if (listSize == 0) {
            continue;
        }
        CHK_PRT_RET(linkList == nullptr,
            HCCL_ERROR("GetLinks returned null links with listSize[%u], srcRank[%u], dstRank[%u], netLayer[%u]",
                listSize, srcRank, dstRank, netLayer),
            HCCL_E_INTERNAL);

        const CommLink *selectedLink = nullptr;
        uint32_t selectedPriority = UNSUPPORTED_PROTOCOL_PRIORITY;
        for (uint32_t linkIdx = 0; linkIdx < listSize; ++linkIdx) {
            if (!IsAicpuTsProtocol(linkList[linkIdx].linkAttr.linkProtocol)) {
                continue;
            }
            uint32_t priority = ProtocolPriority(linkList[linkIdx].linkAttr.linkProtocol);
            if (priority < selectedPriority) {
                selectedLink = &linkList[linkIdx];
                selectedPriority = priority;
            }
        }
        if (selectedLink != nullptr) {
            const CommLink link = *selectedLink;
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
    }

    HCCL_ERROR("No AICPU_TS-compatible link found, srcRank[%u], dstRank[%u]", srcRank, dstRank);
    return firstQueryError == HCCL_SUCCESS ? HCCL_E_NOT_FOUND : firstQueryError;
}

HcclResult AcquirePeerChannels(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtxHost)
{
    if (param.rankSize <= 1) {
        return HCCL_SUCCESS;
    }

    uint32_t *rawNetLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &rawNetLayers, &netLayerNum));
    CHK_PRT_RET(netLayerNum == 0 || rawNetLayers == nullptr,
        HCCL_ERROR("No topology layer found, rank[%u]", param.myRank), HCCL_E_NOT_FOUND);

    const std::vector<uint32_t> netLayers(rawNetLayers, rawNetLayers + netLayerNum);
    std::vector<HcclChannelDesc> descs;
    descs.reserve(param.rankSize - 1);
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }
        HcclChannelDesc desc;
        CHK_RET(BuildPeerChannelDesc(comm, netLayers, param.myRank, remoteRank, desc));
        descs.push_back(desc);
    }

    CHK_PRT_RET(descs.size() != static_cast<size_t>(param.rankSize - 1),
        HCCL_ERROR("Channel description count mismatch, actual[%llu], expected[%u]",
            static_cast<unsigned long long>(descs.size()), param.rankSize - 1),
        HCCL_E_INTERNAL);

    std::vector<ChannelHandle> handles(descs.size());
    CHK_RET(HcclChannelAcquire(
        comm, COMM_ENGINE_AICPU_TS, descs.data(), static_cast<uint32_t>(descs.size()), handles.data()));

    resCtxHost.channels.reserve(handles.size());
    for (size_t channelIdx = 0; channelIdx < handles.size(); ++channelIdx) {
        uint32_t actualNotifyNum = 0;
        CHK_RET(static_cast<HcclResult>(HcommChannelGetNotifyNum(handles[channelIdx], &actualNotifyNum)));
        CHK_PRT_RET(actualNotifyNum < CHANNEL_NOTIFY_NUM,
            HCCL_ERROR("Insufficient Notify capacity on acquired channel, remoteRank[%u], actual[%u], required[%u]",
                descs[channelIdx].remoteRank, actualNotifyNum, CHANNEL_NOTIFY_NUM),
            HCCL_E_NOT_SUPPORT);

        void *remoteCclBuffer = nullptr;
        uint64_t remoteCclBufferSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, handles[channelIdx], &remoteCclBuffer, &remoteCclBufferSize));
        CHK_PRT_RET(remoteCclBuffer == nullptr || remoteCclBufferSize == 0,
            HCCL_ERROR("Invalid remote HCCL buffer, remoteRank[%u], addr[%p], size[%llu]", descs[channelIdx].remoteRank,
                remoteCclBuffer, static_cast<unsigned long long>(remoteCclBufferSize)),
            HCCL_E_INTERNAL);

        ChannelInfo channel;
        channel.remoteRank = descs[channelIdx].remoteRank;
        channel.notifyNum = actualNotifyNum;
        channel.handle = handles[channelIdx];
        channel.remoteCclMem = CommBuffer{remoteCclBuffer, remoteCclBufferSize};
        resCtxHost.channels.push_back(channel);
    }
    return HCCL_SUCCESS;
}

HcclResult CreateAndCopyEngineContext(
    HcclComm comm, const char *tag, CommEngine engine, const void *data, uint64_t dataSize, void **ctx)
{
    HcclResult ret = HcclEngineCtxCreate(comm, tag, engine, dataSize, ctx);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("Failed to create engine context, engine[%d], ret[%d]", engine, ret);
        return ret;
    }

    ret = HcclEngineCtxCopy(comm, engine, tag, data, dataSize, 0);
    if (ret == HCCL_SUCCESS) {
        return HCCL_SUCCESS;
    }

    HCCL_ERROR("Failed to copy engine context, engine[%d], ret[%d]", engine, ret);
    HcclResult destroyRet = HcclEngineCtxDestroy(comm, tag, engine);
    if (destroyRet != HCCL_SUCCESS) {
        HCCL_ERROR("Failed to roll back engine context, engine[%d], ret[%d]", engine, destroyRet);
    }
    return ret;
}
} // namespace

HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    CHK_PRT_RET(SIZE_TABLE.find(dataType) == SIZE_TABLE.end(),
        HCCL_ERROR("Unsupported data type[%d]", static_cast<int32_t>(dataType)), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(!IsSupportedReduceOp(op), HCCL_ERROR("Unsupported reduce operation[%d]", static_cast<int32_t>(op)),
        HCCL_E_NOT_SUPPORT);

    // 构造算子参数
    OpParam param;
    sprintf(param.tag, "%s", "hccl_custom_reduce_scatter");
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;

    // 注册算子信息
    HcclDfxOpInfo dfxInfo;
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    // ==============================================
    // STEP 1: 解析拓扑信息
    // ==============================================
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.myRank >= param.rankSize,
        HCCL_ERROR("Invalid rank metadata, myRank[%u], rankSize[%u]", param.myRank, param.rankSize), HCCL_E_PARA);

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

    void *ctx = nullptr;
    uint64_t size = 0;
    HcclResult aicpuCtxRet = HcclEngineCtxGet(comm, param.tag, aicpuTsEngine, &ctx, &size);
    if (aicpuCtxRet != HCCL_SUCCESS && aicpuCtxRet != HCCL_E_NOT_FOUND) {
        HCCL_ERROR("Failed to query AICPU context, ret[%d]", aicpuCtxRet);
        return aicpuCtxRet;
    }

    void *hostCtx = nullptr;
    uint64_t hostCtxSize = 0;
    HcclResult cpuCtxRet = HcclEngineCtxGet(comm, param.tag, cpuTsEngine, &hostCtx, &hostCtxSize);
    if (cpuCtxRet != HCCL_SUCCESS && cpuCtxRet != HCCL_E_NOT_FOUND) {
        HCCL_ERROR("Failed to query CPU context, ret[%d]", cpuCtxRet);
        return cpuCtxRet;
    }

    if (aicpuCtxRet == HCCL_SUCCESS && cpuCtxRet == HCCL_SUCCESS) {
        // AICPU 资源已经存在，复用资源
        HCCL_INFO("Engine contexts already exist");
        CHK_PRT_RET(ctx == nullptr || size == 0,
            HCCL_ERROR(
                "Invalid cached AICPU context, addr[%p], size[%llu]", ctx, static_cast<unsigned long long>(size)),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(hostCtx == nullptr || hostCtxSize != sizeof(ThreadHandle),
            HCCL_ERROR("Invalid cached CPU context, addr[%p], size[%llu], expected[%llu]", hostCtx,
                static_cast<unsigned long long>(hostCtxSize), static_cast<unsigned long long>(sizeof(ThreadHandle))),
            HCCL_E_INTERNAL);
        param.resCtx = ctx;
        param.ctxSize = size;

        // Host 资源已经存在，复用资源
        ThreadHandle *aicpuThread = static_cast<ThreadHandle *>(hostCtx);
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));
    } else {
        // 若上次创建仅完成一个 Context，先清理残留再原地重建。
        if (aicpuCtxRet == HCCL_SUCCESS) {
            CHK_RET(HcclEngineCtxDestroy(comm, param.tag, aicpuTsEngine));
        }
        if (cpuCtxRet == HCCL_SUCCESS) {
            CHK_RET(HcclEngineCtxDestroy(comm, param.tag, cpuTsEngine));
        }
        // Device 资源不存在，资源构建
        AlgResourceCtx resCtxHost;

        // 从通信域获取 HCCL Buffer（Device上的内存，默认总大小400MB）
        void *cclBufferAddr;
        uint64_t cclBufferSize;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        CHK_PRT_RET(cclBufferAddr == nullptr || cclBufferSize == 0,
            HCCL_ERROR("Invalid local HCCL buffer, addr[%p], size[%llu]", cclBufferAddr,
                static_cast<unsigned long long>(cclBufferSize)),
            HCCL_E_INTERNAL);
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        // ==============================================
        // STEP 2.2: 申请资源Thread和Channel
        // ==============================================

        // 创建 AICPU_TS 通信引擎上的 thread 资源
        uint32_t threadNum = 1;
        uint32_t notifyNumPerThread = 1;

        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, notifyNumPerThread, resCtxHost.threads.data()));
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        CHK_RET(AcquirePeerChannels(comm, param, resCtxHost));

        // ==============================================
        // STEP 2.3: 申请通信引擎上下文
        // ==============================================
        // 申请 AICPU 通信引擎上下文，存放 AlgResourceCtx 信息
        std::vector<char> seq = resCtxHost.Serialize();
        uint64_t seqSize = seq.size();
        param.ctxSize = seqSize;
        CHK_RET(CreateAndCopyEngineContext(comm, param.tag, aicpuTsEngine, seq.data(), seqSize, &param.resCtx));
        // 申请 CPU 通信引擎上下文，存放 aicpuThread 句柄
        hostCtx = nullptr;
        hostCtxSize = sizeof(ThreadHandle);
        const void *aicpuThreadPtr = static_cast<const void *>(&resCtxHost.aicpuThread);
        HcclResult cpuCreateRet
            = CreateAndCopyEngineContext(comm, param.tag, cpuTsEngine, aicpuThreadPtr, hostCtxSize, &hostCtx);
        if (cpuCreateRet != HCCL_SUCCESS) {
            HcclResult rollbackRet = HcclEngineCtxDestroy(comm, param.tag, aicpuTsEngine);
            if (rollbackRet != HCCL_SUCCESS) {
                HCCL_ERROR("Failed to roll back AICPU context, ret[%d]", rollbackRet);
            }
            return cpuCreateRet;
        }
    }

    // ==============================================
    // STEP 3: 下发 AICPU Kernel
    // ==============================================
    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}
