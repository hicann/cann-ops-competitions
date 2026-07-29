/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

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
constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;

HcclResult BuildChannelDesc(
    HcclComm comm,
    uint32_t srcRank,
    uint32_t dstRank,
    HcclChannelDesc *desc)
{
    uint32_t *layerList = nullptr;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layerList, &layerNum));

    CHK_PRT_RET(
        layerList == nullptr || layerNum == 0,
        HCCL_ERROR("Topology layer list is empty"),
        HCCL_E_INTERNAL);

    for (uint32_t i = 0; i < layerNum; ++i) {
        CommLink *links = nullptr;
        uint32_t linkNum = 0;

        HcclResult ret = HcclRankGraphGetLinks(
            comm,
            layerList[i],
            srcRank,
            dstRank,
            &links,
            &linkNum);

        if (ret != HCCL_SUCCESS || links == nullptr || linkNum == 0) {
            continue;
        }

        const CommLink &link = links[0];

        desc->remoteRank = dstRank;
        desc->notifyNum = CHANNEL_NOTIFY_NUM;
        desc->channelProtocol = link.linkAttr.linkProtocol;

        desc->localEndpoint.protocol = link.srcEndpointDesc.protocol;
        desc->localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
        desc->localEndpoint.loc = link.srcEndpointDesc.loc;

        desc->remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
        desc->remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
        desc->remoteEndpoint.loc = link.dstEndpointDesc.loc;

        HCCL_INFO(
            "BuildChannelDesc success: src[%u], dst[%u], layer[%u], protocol[%d]",
            srcRank,
            dstRank,
            layerList[i],
            desc->channelProtocol);

        return HCCL_SUCCESS;
    }

    HCCL_ERROR(
        "No available link from rank[%u] to rank[%u]",
        srcRank,
        dstRank);
    return HCCL_E_INTERNAL;
}

HcclResult AcquireBroadcastStarChannels(
    HcclComm comm,
    CommEngine engine,
    const OpParam &param,
    AlgResourceCtx &resCtxHost)
{
    if (param.rankSize <= 1) {
        return HCCL_SUCCESS;
    }

    constexpr uint32_t ranksPerServer = 8;
    std::vector<uint32_t> remoteRanks;

    if (param.rankSize <= ranksPerServer) {
        // 2卡或8卡单 Server：root 与其他 Rank 直接通信。
        if (param.myRank == param.root) {
            for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
                if (rank != param.root) {
                    remoteRanks.push_back(rank);
                }
            }
        } else {
            remoteRanks.push_back(param.root);
        }
    } else if (param.rankSize == 2 * ranksPerServer) {
        // 2×8卡：
        // root 发送给本 Server 其余 Rank，并通过 Clos 发送给另一个
        // Server 上 local-rank 相同的 bridge；bridge 再完成第二级广播。
        const uint32_t rootServer = param.root / ranksPerServer;
        const uint32_t myServer = param.myRank / ranksPerServer;
        const uint32_t rootServerBase = rootServer * ranksPerServer;
        const uint32_t bridgeRank =
            rootServer == 0 ? param.root + ranksPerServer : param.root - ranksPerServer;
        const uint32_t bridgeServerBase =
            (1 - rootServer) * ranksPerServer;

        if (param.myRank == param.root) {
            // 先服务本 Server，再发送给 bridge。
            for (uint32_t rank = rootServerBase;
                 rank < rootServerBase + ranksPerServer;
                 ++rank) {
                if (rank != param.root) {
                    remoteRanks.push_back(rank);
                }
            }
            remoteRanks.push_back(bridgeRank);
        } else if (param.myRank == bridgeRank) {
            // 先从 root 接收，再服务第二个 Server。
            remoteRanks.push_back(param.root);
            for (uint32_t rank = bridgeServerBase;
                 rank < bridgeServerBase + ranksPerServer;
                 ++rank) {
                if (rank != bridgeRank) {
                    remoteRanks.push_back(rank);
                }
            }
        } else if (myServer == rootServer) {
            remoteRanks.push_back(param.root);
        } else {
            remoteRanks.push_back(bridgeRank);
        }
    } else {
        HCCL_ERROR(
            "Unsupported rank size[%u], only 2/8/16-rank competition topologies are supported",
            param.rankSize);
        return HCCL_E_INTERNAL;
    }

    const uint32_t channelNum =
        static_cast<uint32_t>(remoteRanks.size());

    std::vector<HcclChannelDesc> channelDescs(channelNum);
    std::vector<ChannelHandle> channelHandles(channelNum);

    CHK_RET(HcclChannelDescInit(
        channelDescs.data(),
        channelNum));

    for (uint32_t i = 0; i < channelNum; ++i) {
        CHK_RET(BuildChannelDesc(
            comm,
            param.myRank,
            remoteRanks[i],
            &channelDescs[i]));
    }

    CHK_RET(HcclChannelAcquire(
        comm,
        engine,
        channelDescs.data(),
        channelNum,
        channelHandles.data()));

    resCtxHost.channels.resize(channelNum);

    for (uint32_t i = 0; i < channelNum; ++i) {
        ChannelInfo &channel = resCtxHost.channels[i];

        channel.remoteRank = remoteRanks[i];
        channel.notifyNum = CHANNEL_NOTIFY_NUM;
        channel.handle = channelHandles[i];

        CHK_RET(HcclChannelGetHcclBuffer(
            comm,
            channel.handle,
            &channel.remoteCclMem.addr,
            &channel.remoteCclMem.size));

        CHK_PRT_RET(
            channel.remoteCclMem.addr == nullptr ||
                channel.remoteCclMem.size == 0,
            HCCL_ERROR(
                "Invalid remote buffer for rank[%u]",
                channel.remoteRank),
            HCCL_E_INTERNAL);
    }

    HCCL_INFO(
        "AcquireBroadcastChannels success: rank[%u], root[%u], channelNum[%u]",
        param.myRank,
        param.root,
        channelNum);

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
    sprintf(param.tag, "hccl_custom_broadcast_r%u", root);
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
    CHK_PRT_RET(
        param.root >= param.rankSize,
        HCCL_ERROR("Invalid root[%u], rankSize[%u]", param.root, param.rankSize),
        HCCL_E_INTERNAL);

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

        // 每条通信 Channel 使用一条独立工作 Thread。
        // threads[0] 为控制 Thread，用于 Host/Device 同步以及工作 Thread 汇合。
        CHK_RET(AcquireBroadcastStarChannels(
            comm, aicpuTsEngine, param, resCtxHost));

        const uint32_t workerThreadNum =
            static_cast<uint32_t>(resCtxHost.channels.size());
        const uint32_t threadNum = workerThreadNum + 1;
        const uint32_t notifyNumPerThread =
            workerThreadNum == 0 ? 1 : workerThreadNum;

        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(
            comm,
            aicpuTsEngine,
            threadNum,
            notifyNumPerThread,
            resCtxHost.threads.data()));

        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(
            comm,
            1,
            &resCtxHost.aicpuThread,
            cpuTsEngine,
            &param.aicpuThreadOnCpu));

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
