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

HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

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
    HcclDfxOpInfo dfxInfo;
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    // ==============================================
    // STEP 1: 解析拓扑信息
    // ==============================================
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));

    constexpr uint32_t NUM_SERVERS = 2;
    constexpr uint32_t NUM_NPU_PER_SERVER = 8;
    uint32_t serverId = param.myRank / NUM_NPU_PER_SERVER;
    uint32_t localRankId = param.myRank % NUM_NPU_PER_SERVER;

    HCCL_INFO("AllReduce: myRank=%u, rankSize=%u, serverId=%u, localRankId=%u, count=%lu",
        param.myRank, param.rankSize, serverId, localRankId, count);

    // 构建 Rank 分组信息
    RankGroupInfo rankGroupInfo;
    rankGroupInfo.serverId = serverId;
    rankGroupInfo.localRankId = localRankId;
    rankGroupInfo.globalRankId = param.myRank;
    rankGroupInfo.rankSize = param.rankSize;
    rankGroupInfo.numPerServer = NUM_NPU_PER_SERVER;

    // 解析拓扑链路，确定每个对端 rank 的链路类型
    std::vector<LinkInfo> linkInfos;
    uint32_t numLinks = 0;
    HcclRankLink *rankLinks = nullptr;
    CHK_RET(HcclRankGraphGetLinks(comm, &numLinks, &rankLinks));
    for (uint32_t i = 0; i < numLinks; i++) {
        uint32_t remoteRank = rankLinks[i].remoteRank;
        LinkType linkType;
        // 根据 remoteRank 判断链路类型：同一 Server 内为 Mesh，不同 Server 为 Clos
        if (remoteRank / NUM_NPU_PER_SERVER == serverId) {
            linkType = LinkType::INTRA_SERVER_MESH;
        } else {
            linkType = LinkType::INTER_SERVER_CLOS;
        }
        linkInfos.push_back({remoteRank, linkType});
        HCCL_INFO("Link[%u]: remoteRank=%u, type=%s", i, remoteRank,
            (linkType == LinkType::INTRA_SERVER_MESH) ? "INTRA_MESH" : "INTER_CLOS");
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
        resCtxHost.rankGroupInfo = rankGroupInfo;
        resCtxHost.linkInfos = linkInfos;

        // 从通信域获取 HCCL Buffer（Device上的内存，默认总大小400MB）
        void *cclBufferAddr;
        uint64_t cclBufferSize;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        // ==============================================
        // STEP 2.2: 申请资源 Thread 和 Channel
        // ==============================================

        // 层次化 AllReduce 算法需要：
        //   - 7 个 Intra-server Channel (Full-Mesh 直连其他 7 个 NPU)
        //   - 1 个 Inter-server Channel (Clos 连接对端 Server 对应 NPU)
        //   - 每个 Channel 需要 1 个 Notify 用于收发同步
        //   - 总共最多 numLinks 个 Channel

        // 计算 Intra-server 和 Inter-server 对端数量
        uint32_t intraPeerCount = 0;
        uint32_t interPeerCount = 0;
        for (const auto &link : linkInfos) {
            if (link.linkType == LinkType::INTRA_SERVER_MESH) {
                intraPeerCount++;
            } else {
                interPeerCount++;
            }
        }
        HCCL_INFO("Topology: %u intra-server peers, %u inter-server peers", intraPeerCount, interPeerCount);

        // Thread 数量：至少为 1（用于通信操作下发），额外线程可提升并行度
        // 每个 Channel 配 1 个 Notify，加上 Notify 0 用于 Host/Device 同步
        uint32_t threadNum = 1;
        uint32_t notifyNumPerThread = 1 + intraPeerCount + interPeerCount; // 至少覆盖所有 channel 的同步需求

        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, notifyNumPerThread, resCtxHost.threads.data()));
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        // 为每条拓扑链路申请 Channel 资源
        resCtxHost.channels.clear();
        resCtxHost.channels.reserve(numLinks);
        for (uint32_t i = 0; i < numLinks; i++) {
            uint32_t remoteRank = rankLinks[i].remoteRank;
            LinkType linkType = linkInfos[i].linkType;
            uint32_t channelNotifyNum = i + 1; // notifyIdx 从 1 开始分配（0 留给 Host/Device 同步）

            HcclChannelDesc channelDesc;
            CHK_RET(HcclChannelDescInit(&channelDesc));
            channelDesc.remoteRank = remoteRank;
            channelDesc.notifyIdx = channelNotifyNum;
            // 标记链路类型，便于 Device 侧根据链路类型选择通信策略
            channelDesc.linkType = (linkType == LinkType::INTRA_SERVER_MESH) ?
                HCCL_LINK_TYPE_MESH : HCCL_LINK_TYPE_CLOS;

            ChannelHandle channelHandle = 0;
            CHK_RET(HcclChannelAcquire(comm, aicpuTsEngine, &channelDesc, &channelHandle));

            ChannelInfo channelInfo;
            channelInfo.remoteRank = remoteRank;
            channelInfo.notifyNum = channelNotifyNum;
            channelInfo.handle = channelHandle;
            // remoteCclMem 由框架自动填充
            resCtxHost.channels.push_back(channelInfo);

            HCCL_INFO("Channel acquired: remoteRank=%u, notifyNum=%u, type=%s",
                remoteRank, channelNotifyNum,
                (linkType == LinkType::INTRA_SERVER_MESH) ? "INTRA_MESH" : "INTER_CLOS");
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
