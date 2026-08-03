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

#include <limits>
#include <vector>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

namespace {

HcclResult FillChannelDesc(HcclComm comm, uint32_t netLayer, uint32_t srcRank, uint32_t dstRank,
    uint32_t notifyNum, HcclChannelDesc &desc)
{
    uint32_t linkNum = 0;
    CommLink *links = nullptr;
    CHK_RET(HcclRankGraphGetLinks(comm, netLayer, srcRank, dstRank, &links, &linkNum));
    CHK_PRT_RET(linkNum == 0,
        HCCL_ERROR("No link on layer[%u] between rank[%u] and rank[%u]", netLayer, srcRank, dstRank),
        HCCL_E_INTERNAL);
    CHK_PTR_NULL(links);

    CHK_RET(HcclChannelDescInit(&desc, 1));
    const CommLink link = links[0];
    desc.remoteRank = dstRank;
    desc.notifyNum = notifyNum;
    desc.channelProtocol = link.linkAttr.linkProtocol;
    desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
    desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
    desc.localEndpoint.loc = link.srcEndpointDesc.loc;
    desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
    desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
    desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
    return HCCL_SUCCESS;
}

HcclResult AcquireChannels(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtx)
{
    const uint32_t channelNum = custom_rs::DIRECT_CHANNEL_NUM;
    const uint32_t channelNotifyNum = custom_rs::DIRECT_CHANNEL_NOTIFY_NUM;
    std::vector<HcclChannelDesc> descs(channelNum);
    std::vector<ChannelHandle> handles(channelNum);

    uint32_t channelIdx = 0;
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }
        const uint32_t netLayer = custom_rs::GetNetLayer(param.myRank, remoteRank);
        CHK_RET(FillChannelDesc(
            comm, netLayer, param.myRank, remoteRank, channelNotifyNum, descs[channelIdx++]));
    }

    CHK_PRT_RET(channelIdx != channelNum,
        HCCL_ERROR("Unexpected channel count[%u], expected[%u]", channelIdx, channelNum),
        HCCL_E_INTERNAL);

    CHK_RET(HcclChannelAcquire(comm, COMM_ENGINE_AICPU_TS, descs.data(), channelNum, handles.data()));
    resCtx.channels.reserve(channelNum);
    for (uint32_t idx = 0; idx < channelNum; ++idx) {
        void *remoteBuffer = nullptr;
        uint64_t remoteBufferSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, handles[idx], &remoteBuffer, &remoteBufferSize));
        CHK_PTR_NULL(remoteBuffer);
        CHK_PRT_RET(remoteBufferSize == 0,
            HCCL_ERROR("Remote HCCL buffer of rank[%u] is empty", descs[idx].remoteRank), HCCL_E_INTERNAL);

        ChannelInfo channel;
        channel.remoteRank = descs[idx].remoteRank;
        channel.notifyNum = channelNotifyNum;
        channel.handle = handles[idx];
        channel.remoteCclMem = CommBuffer{remoteBuffer, remoteBufferSize};
        resCtx.channels.push_back(channel);
    }
    return HCCL_SUCCESS;
}

} // namespace

HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    if (recvCount == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("Only FP32 is supported, dataType[%d]", static_cast<int32_t>(dataType)), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(op != HCCL_REDUCE_SUM,
        HCCL_ERROR("Only SUM is supported, reduceOp[%d]", static_cast<int32_t>(op)), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(recvCount > std::numeric_limits<uint64_t>::max() / sizeof(float) / custom_rs::RANK_SIZE,
        HCCL_ERROR("recvCount[%llu] causes input size overflow", static_cast<unsigned long long>(recvCount)),
        HCCL_E_PARA);

    // 构造算子参数
    OpParam param;
    sprintf(param.tag, "%s", "hccl_custom_reduce_scatter_hybrid");
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
    CHK_PRT_RET(param.rankSize != custom_rs::RANK_SIZE,
        HCCL_ERROR("Only rankSize[%u] is supported, actual[%u]", custom_rs::RANK_SIZE, param.rankSize),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(param.myRank >= param.rankSize,
        HCCL_ERROR("Invalid rank[%u] for rankSize[%u]", param.myRank, param.rankSize), HCCL_E_PARA);

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
        CHK_PTR_NULL(cclBufferAddr);
        CHK_PRT_RET(cclBufferSize == 0, HCCL_ERROR("Local HCCL buffer is empty"), HCCL_E_INTERNAL);
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        // ==============================================
        // STEP 2.2: 申请资源Thread和Channel
        // ==============================================

        const uint32_t threadNum = custom_rs::DIRECT_THREAD_NUM;
        const uint32_t threadNotifyNum = custom_rs::DIRECT_THREAD_NOTIFY_NUM;
        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(
            comm, aicpuTsEngine, threadNum, threadNotifyNum, resCtxHost.threads.data()));
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        CHK_RET(AcquireChannels(comm, param, resCtxHost));

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
