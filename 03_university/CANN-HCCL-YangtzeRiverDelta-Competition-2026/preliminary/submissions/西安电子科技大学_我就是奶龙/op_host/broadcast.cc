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
#include <vector>

#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

namespace {
constexpr uint32_t INTRA_SERVER_LAYER = 0;
constexpr uint32_t INTER_SERVER_LAYER = 1;
constexpr uint32_t CHANNEL_NOTIFY_NUM = 2;
constexpr uint32_t OFFICIAL_RANK_SIZE = 16;
constexpr uint32_t DIRECT16_THREAD_NUM = OFFICIAL_RANK_SIZE;
constexpr uint32_t DIRECT16_THREAD_NOTIFY_NUM = OFFICIAL_RANK_SIZE - 1;
constexpr char HYBRID_DIRECT16_TAG[] = "hccl_custom_broadcast_direct16_hybrid_v2";

void FillChannelDesc(const CommLink &link, uint32_t remoteRank, HcclChannelDesc &channelDesc)
{
    channelDesc.remoteRank = remoteRank;
    channelDesc.notifyNum = CHANNEL_NOTIFY_NUM;
    channelDesc.channelProtocol = link.linkAttr.linkProtocol;
    channelDesc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
    channelDesc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
    channelDesc.localEndpoint.loc = link.srcEndpointDesc.loc;
    channelDesc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
    channelDesc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
    channelDesc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
}

HcclResult AppendChannelDesc(HcclComm comm, uint32_t myRank, uint32_t remoteRank, uint32_t netLayer,
    uint32_t preferredLinkIndex, std::vector<HcclChannelDesc> &channelDescs)
{
    CommLink *linkList = nullptr;
    uint32_t listSize = 0;
    CHK_RET(HcclRankGraphGetLinks(comm, netLayer, myRank, remoteRank, &linkList, &listSize));
    CHK_PRT_RET(listSize == 0,
        HCCL_ERROR("[AppendChannelDesc] No link between rank[%u] and rank[%u] on layer[%u].", myRank,
            remoteRank, netLayer),
        HCCL_E_INTERNAL);

    CommProtocol selectedProtocol = linkList[0].linkAttr.linkProtocol;
    uint32_t protocolLinkNum = 0;
    for (uint32_t index = 0; index < listSize; ++index) {
        protocolLinkNum += (linkList[index].linkAttr.linkProtocol == selectedProtocol);
    }
    CHK_PRT_RET(protocolLinkNum == 0,
        HCCL_ERROR("[AppendChannelDesc] No usable link to rank[%u] on layer[%u].", remoteRank, netLayer),
        HCCL_E_INTERNAL);

    uint32_t targetLinkIndex = preferredLinkIndex % protocolLinkNum;
    const CommLink *selectedLink = nullptr;
    for (uint32_t index = 0; index < listSize; ++index) {
        if (linkList[index].linkAttr.linkProtocol != selectedProtocol) {
            continue;
        }
        if (targetLinkIndex-- == 0) {
            selectedLink = &linkList[index];
            break;
        }
    }

    CHK_PTR_NULL(selectedLink);
    HcclChannelDesc channelDesc;
    CHK_RET(HcclChannelDescInit(&channelDesc, 1));
    FillChannelDesc(*selectedLink, remoteRank, channelDesc);
    channelDescs.push_back(channelDesc);
    return HCCL_SUCCESS;
}

HcclResult AcquireDirect16Channels(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtxHost)
{
    CHK_PRT_RET(param.rankSize != OFFICIAL_RANK_SIZE,
        HCCL_ERROR("[AcquireDirect16Channels] Invalid rankSize[%u].", param.rankSize), HCCL_E_PARA);

    uint32_t localRankSize = param.rankSize / 2;
    std::vector<HcclChannelDesc> channelDescs;
    channelDescs.reserve(param.rankSize - 1);
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }
        bool isSameServer = (remoteRank / localRankSize) == (param.myRank / localRankSize);
        uint32_t netLayer = isSameServer ? INTRA_SERVER_LAYER : INTER_SERVER_LAYER;
        uint32_t preferredLinkIndex = 0;
        if (!isSameServer) {
            preferredLinkIndex
                = (param.myRank % localRankSize + remoteRank % localRankSize) % localRankSize;
        }
        CHK_RET(AppendChannelDesc(
            comm, param.myRank, remoteRank, netLayer, preferredLinkIndex, channelDescs));
    }

    std::vector<bool> channelSeen(param.rankSize, false);
    for (const auto &channelDesc : channelDescs) {
        CHK_PRT_RET(channelDesc.remoteRank >= param.rankSize || channelDesc.remoteRank == param.myRank,
            HCCL_ERROR("[AcquireDirect16Channels] Invalid remoteRank[%u].", channelDesc.remoteRank),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(channelSeen[channelDesc.remoteRank],
            HCCL_ERROR("[AcquireDirect16Channels] Duplicate channel to rank[%u].", channelDesc.remoteRank),
            HCCL_E_INTERNAL);
        channelSeen[channelDesc.remoteRank] = true;
    }
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        CHK_PRT_RET(remoteRank != param.myRank && !channelSeen[remoteRank],
            HCCL_ERROR("[AcquireDirect16Channels] Channel to rank[%u] is missing.", remoteRank),
            HCCL_E_INTERNAL);
    }

    uint32_t channelNum = static_cast<uint32_t>(channelDescs.size());
    std::vector<ChannelHandle> channelHandles(channelNum);
    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AICPU_TS, channelDescs.data(), channelNum,
        channelHandles.data()));
    for (uint32_t index = 0; index < channelNum; ++index) {
        void *remoteBuffer = nullptr;
        uint64_t remoteBufferSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, channelHandles[index], &remoteBuffer, &remoteBufferSize));

        ChannelInfo channel;
        channel.remoteRank = channelDescs[index].remoteRank;
        channel.notifyNum = CHANNEL_NOTIFY_NUM;
        channel.handle = channelHandles[index];
        channel.remoteCclMem = CommBuffer{remoteBuffer, remoteBufferSize};
        resCtxHost.channels.push_back(channel);
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclBroadcast(
    void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(buf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    // 构造算子参数
    OpParam param;
    sprintf(param.tag, "%s", HYBRID_DIRECT16_TAG);
    param.inputPtr = buf;
    param.outputPtr = buf;
    param.count = count;
    param.dataType = dataType;
    param.root = root;
    param.opType = HcclCMDType::HCCL_CMD_BROADCAST;

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
    CHK_PRT_RET(root >= param.rankSize,
        HCCL_ERROR("[HcclBroadcast] Invalid root[%u], rankSize[%u].", root, param.rankSize), HCCL_E_PARA);

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

        // TODO: 根据通信算法申请 Thread 资源
        // 创建 AICPU_TS 通信引擎上的 thread 资源
        uint32_t threadNum = DIRECT16_THREAD_NUM;
        uint32_t notifyNumPerThread = DIRECT16_THREAD_NOTIFY_NUM;

        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, notifyNumPerThread, resCtxHost.threads.data()));
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        // TODO: 根据通信算法申请 Channel 资源
        // 调用 HcclRankGraphGetLinks()、HcclChannelDescInit()、HcclChannelAcquire() 等接口按需申请 Channel 资源
        if (param.rankSize > 1) {
            CHK_RET(AcquireDirect16Channels(comm, param, resCtxHost));
        }

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