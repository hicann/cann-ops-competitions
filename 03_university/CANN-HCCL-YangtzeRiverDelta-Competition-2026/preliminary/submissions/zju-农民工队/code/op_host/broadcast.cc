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

constexpr uint32_t CHANNEL_NOTIFY_NUM = 4;

HcclResult AcquireDesc(
    HcclComm comm, uint32_t localRank, uint32_t remoteRank, uint32_t netLayer, HcclChannelDesc *desc)
{
    uint32_t listSize = 0;
    CommLink *linkList = nullptr;
    CHK_RET(HcclRankGraphGetLinks(comm, netLayer, localRank, remoteRank, &linkList, &listSize));
    CHK_PRT_RET(listSize == 0,
        HCCL_ERROR("AcquireDesc: no layer-%u link between rank[%u] and rank[%u]", netLayer, localRank, remoteRank),
        HCCL_E_INTERNAL);

    CHK_RET(HcclChannelDescInit(desc, 1));
    CommLink link = linkList[0];
    desc->remoteRank = remoteRank;
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

// 每卡申请7条Server内Mesh channel，以及另一个Server的8条Clos channel。
HcclResult AcquireChannel(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtxHost)
{
    CHK_PRT_RET(param.rankSize != 16, HCCL_ERROR("AcquireChannel: rankSize[%u] is not 16", param.rankSize),
        HCCL_E_PARA);
    constexpr uint32_t serverSize = 8;
    constexpr uint32_t channelNum = 15;
    std::vector<HcclChannelDesc> desc(channelNum);
    std::vector<ChannelHandle> handles(channelNum);
    uint32_t idx = 0;
    uint32_t serverBegin = param.myRank / serverSize * serverSize;
    for (uint32_t rank = serverBegin; rank < serverBegin + serverSize; rank++) {
        if (rank != param.myRank) {
            CHK_RET(AcquireDesc(comm, param.myRank, rank, 0, &desc[idx++]));
        }
    }
    uint32_t remoteServerBegin = param.myRank < serverSize ? serverSize : 0;
    for (uint32_t rank = remoteServerBegin; rank < remoteServerBegin + serverSize; rank++) {
        CHK_RET(AcquireDesc(comm, param.myRank, rank, 1, &desc[idx++]));
    }

    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AICPU_TS, desc.data(), channelNum, handles.data()));
    for (uint32_t i = 0; i < channelNum; i++) {
        ChannelInfo channel;
        channel.remoteRank = desc[i].remoteRank;
        channel.notifyNum = CHANNEL_NOTIFY_NUM;
        channel.handle = handles[i];
        void *remoteBuffer = nullptr;
        uint64_t remoteBufferSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, handles[i], &remoteBuffer, &remoteBufferSize));
        channel.remoteCclMem = CommBuffer{remoteBuffer, remoteBufferSize};
        resCtxHost.channels.push_back(channel);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclBroadcast(
    void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(buf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    // 构造算子参数
    OpParam param;
    sprintf(param.tag, "%s", "hccl_custom_broadcast");
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
        uint32_t threadNum = 6;          // 0:Scatter，1:Fold，2/3:Doubling，4:Final Read，5:Copy
        uint32_t notifyNumPerThread = 6; // 0..2:wave启动，3..5:Copy完成或Buffer释放

        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, notifyNumPerThread, resCtxHost.threads.data()));
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        // TODO: 根据通信算法申请 Channel 资源
        // 调用 HcclRankGraphGetLinks()、HcclChannelDescInit()、HcclChannelAcquire() 等接口按需申请 Channel 资源
        CHK_RET(AcquireChannel(comm, param, resCtxHost));

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
