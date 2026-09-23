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

namespace {
// 每条 channel 的 notify 数。big 双槽流水需每槽独立 ACK/DATA：slot0=idx0/1、slot1=idx2/3，
// 故需 4 个；small 只用 0/1。
constexpr uint32_t CHANNEL_NOTIFY_NUM = 4;
// 固定拓扑常量（初赛固定 2 Server × 8 NPU = 16 ranks）
constexpr uint32_t SERVER_SIZE = 8;
constexpr uint32_t NUM_SERVERS = 2;
// 每线程 notify 数：用于从线程间同步（主/从流握手）
constexpr uint32_t THREAD_NOTIFY_NUM = 4;

HcclResult AcquireDesc(HcclComm comm, uint32_t srcRank, uint32_t dstRank, HcclChannelDesc *desc)
{
    uint32_t *layerPtr = nullptr;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layerPtr, &layerNum));
    std::vector<uint32_t> layers(layerPtr, layerPtr + layerNum);
    for (uint32_t i = 0; i < layers.size(); ++i) {
        uint32_t linkNum = 0;
        CommLink *links = nullptr;
        if (HcclRankGraphGetLinks(comm, layers[i], srcRank, dstRank, &links, &linkNum) != HCCL_SUCCESS) continue;
        if (linkNum == 0) continue;
        CHK_RET(HcclChannelDescInit(desc, 1));
        const CommLink &link = links[0];
        desc->remoteRank = dstRank;
        desc->notifyNum = CHANNEL_NOTIFY_NUM;
        desc->channelProtocol = link.linkAttr.linkProtocol;
        desc->localEndpoint = link.srcEndpointDesc;
        desc->remoteEndpoint = link.dstEndpointDesc;
        return HCCL_SUCCESS;
    }
    return HCCL_E_NOT_FOUND;
}

HcclResult AcquireChannels(HcclComm comm, const OpParam &param, AlgResourceCtx &resource)
{
    const uint32_t serverId = param.myRank / SERVER_SIZE;
    const uint32_t rankInServer = param.myRank % SERVER_SIZE;

    // 本地 CCL buffer（通信域总 hcc buffer）；通讯对象必须落到这里才能 wire。
    // ---- Server 内 Mesh channel（同 Server 的其他 7 个 peer，layer0）----
    std::vector<HcclChannelDesc> meshDescs;
    std::vector<ChannelHandle> meshHandles;
    for (uint32_t k = 0; k < SERVER_SIZE; ++k) {
        if (k == rankInServer) continue;
        HcclChannelDesc desc;
        CHK_RET(AcquireDesc(comm, param.myRank, serverId * SERVER_SIZE + k, &desc));
        meshDescs.push_back(desc);
    }
    if (!meshDescs.empty()) {
        meshHandles.resize(meshDescs.size());
        CHK_RET(HcclChannelAcquire(comm, COMM_ENGINE_AICPU_TS, meshDescs.data(), meshDescs.size(),
            meshHandles.data()));
        for (size_t i = 0; i < meshDescs.size(); ++i) {
            ChannelInfo channel;
            channel.remoteRank = meshDescs[i].remoteRank;
            channel.notifyNum = CHANNEL_NOTIFY_NUM;
            channel.handle = meshHandles[i];
            CHK_RET(HcclChannelGetHcclBuffer(comm, meshHandles[i], &channel.remoteCclMem.addr,
                &channel.remoteCclMem.size));
            resource.meshChannels.push_back(channel);
        }
    }

    // ---- 跨 Server Clos channel（对端 Server 全部 8 个 rank，各 1 条，layer1）----
    // 去中转：root 直发所有 rank，本 Server 走 Mesh、跨 Server 走 Clos，不再经 subRoot 中转。
    if (param.rankSize > SERVER_SIZE) {
        const uint32_t otherServerId = (serverId == 0) ? 1 : 0;
        for (uint32_t k = 0; k < SERVER_SIZE; ++k) {
            const uint32_t peer = otherServerId * SERVER_SIZE + k;
            HcclChannelDesc desc;
            CHK_RET(AcquireDesc(comm, param.myRank, peer, &desc));
            ChannelHandle handle = 0;
            CHK_RET(HcclChannelAcquire(comm, COMM_ENGINE_AICPU_TS, &desc, 1, &handle));
            ChannelInfo channel;
            channel.remoteRank = desc.remoteRank;
            channel.notifyNum = CHANNEL_NOTIFY_NUM;
            channel.handle = handle;
            CHK_RET(HcclChannelGetHcclBuffer(comm, handle, &channel.remoteCclMem.addr,
                &channel.remoteCclMem.size));
            resource.closChannels.push_back(channel);
        }
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    CHK_PRT_RET(SIZE_TABLE.find(dataType) == SIZE_TABLE.end(),
        HCCL_ERROR("Unsupported data type[%d]", dataType), HCCL_E_PARA);

    // 构造算子参数
    OpParam param;
    sprintf(param.tag, "%s", "hccl_custom_scatter");
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_SCATTER;
    param.root = root;

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
    CHK_PRT_RET(param.rankSize == 0 || root >= param.rankSize,
        HCCL_ERROR("Invalid root[%u] for rank size[%u]", root, param.rankSize), HCCL_E_PARA);

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

        // 根据固定 2×8 拓扑申请 Thread：1 主（Host/Device 同步）+ 7 Mesh 从线程 + 8 Clos 从线程。
        // v7：clos 每条独立流提交，让 root 的 7 Mesh + 8 Clos 真并发(裁决 root clos 是否 8 独立物理口)。
        const uint32_t meshPeerNum = SERVER_SIZE - 1;                    // 7
        const uint32_t closPeerNum = SERVER_SIZE;                        // 8
        const uint32_t threadNum = 1 + meshPeerNum + closPeerNum;        // 16
        const uint32_t notifyNumPerThread = THREAD_NOTIFY_NUM;

        std::vector<ThreadHandle> allThreads(threadNum);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, notifyNumPerThread, allThreads.data()));
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = allThreads[0];
        for (uint32_t i = 0; i < meshPeerNum; ++i) {
            resCtxHost.meshThreads.push_back(allThreads[1 + i]);
        }
        for (uint32_t i = 0; i < closPeerNum; ++i) {
            resCtxHost.closThreads.push_back(allThreads[1 + meshPeerNum + i]);
        }
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
