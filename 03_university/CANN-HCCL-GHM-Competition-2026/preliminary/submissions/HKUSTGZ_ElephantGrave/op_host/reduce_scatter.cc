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

#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 2;
constexpr uint32_t COMPETITION_RANK_SIZE = 16;

HcclResult FillChannelDesc(HcclComm comm, const std::vector<uint32_t> &layers, uint32_t srcRank,
    uint32_t dstRank, HcclChannelDesc &desc)
{
    for (uint32_t layer : layers) {
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        HcclResult ret = HcclRankGraphGetLinks(comm, layer, srcRank, dstRank, &links, &linkNum);
        if (ret != HCCL_SUCCESS || links == nullptr || linkNum == 0) {
            continue;
        }

        for (uint32_t linkIdx = 0; linkIdx < linkNum; ++linkIdx) {
            const CommLink &link = links[linkIdx];
            if (link.linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                continue;
            }
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

    HCCL_ERROR("No UBC_CTP link from rank[%u] to rank[%u]", srcRank, dstRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult AcquireChannels(HcclComm comm, const OpParam &param, AlgResourceCtx &resources)
{
    uint32_t *layerList = nullptr;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layerList, &layerNum));
    CHK_PRT_RET(layerList == nullptr || layerNum == 0,
        HCCL_ERROR("The communicator exposes no topology layer"), HCCL_E_INTERNAL);

    const std::vector<uint32_t> layers(layerList, layerList + layerNum);
    const uint32_t channelNum = param.rankSize - 1;
    std::vector<HcclChannelDesc> descs(channelNum);
    CHK_RET(HcclChannelDescInit(descs.data(), channelNum));

    uint32_t channelIdx = 0;
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }
        CHK_RET(FillChannelDesc(comm, layers, param.myRank, remoteRank, descs[channelIdx]));
        ++channelIdx;
    }

    std::vector<ChannelHandle> handles(channelNum);
    CHK_RET(HcclChannelAcquire(
        comm, CommEngine::COMM_ENGINE_AICPU_TS, descs.data(), channelNum, handles.data()));

    resources.channels.reserve(channelNum);
    for (uint32_t index = 0; index < channelNum; ++index) {
        void *remoteBuffer = nullptr;
        uint64_t remoteBufferSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, handles[index], &remoteBuffer, &remoteBufferSize));

        ChannelInfo channel;
        channel.remoteRank = descs[index].remoteRank;
        channel.notifyNum = CHANNEL_NOTIFY_NUM;
        channel.handle = handles[index];
        channel.remoteCclMem = CommBuffer{remoteBuffer, remoteBufferSize};
        resources.channels.push_back(channel);
    }
    return HCCL_SUCCESS;
}

HcclResult AcquireThreads(HcclComm comm, AlgResourceCtx &resources)
{
    // Thread 0 is the main task stream. Threads 1..15 each own exactly one
    // peer channel. Main notify 0 is reserved for Host/AICPU synchronization;
    // main notify 1..15 collect worker completion. Each worker needs only its
    // start token at notify 0.
    const uint32_t threadNum = COMPETITION_RANK_SIZE;
    std::vector<ThreadConfig> configs(threadNum);
    CHK_RET(static_cast<HcclResult>(ThreadConfigInit(configs.data(), threadNum)));
    configs[0].notifyNumPerThread = static_cast<uint16_t>(threadNum);
    for (uint32_t index = 1; index < threadNum; ++index) {
        configs[index].notifyNumPerThread = 1;
    }

    resources.threads.resize(threadNum);
    CHK_RET(HcclThreadAcquireWithConfig(comm, CommEngine::COMM_ENGINE_AICPU, threadNum,
        ThreadType::THREAD_TYPE_TS, configs.data(), resources.threads.data()));
    resources.aicpuThread = resources.threads[0];
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

    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("Unsupported data type[%d]; competition cases require FP32", static_cast<int>(dataType)),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(op != HCCL_REDUCE_SUM,
        HCCL_ERROR("Unsupported reduce operation[%d]; competition cases require SUM", static_cast<int>(op)),
        HCCL_E_NOT_SUPPORT);

    // 构造算子参数
    OpParam param;
    snprintf(param.tag, sizeof(param.tag), "%s", "hccl_custom_reduce_scatter_latin_v1");
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;

    // 注册算子信息
    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    // ==============================================
    // STEP 1: 解析拓扑信息
    // ==============================================
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize != COMPETITION_RANK_SIZE || param.myRank >= param.rankSize,
        HCCL_ERROR("This competition implementation requires rankSize[%u], got myRank[%u], rankSize[%u]",
            COMPETITION_RANK_SIZE, param.myRank, param.rankSize),
        HCCL_E_PARA);

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

        CHK_RET(AcquireThreads(comm, resCtxHost));
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
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
