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
constexpr uint32_t CHANNEL_NOTIFY_NUM = 2;
constexpr uint32_t RANK_SIZE_PER_SERVER = 8;
constexpr uint32_t SERVER_NUM = 2;
constexpr uint32_t COMM_THREAD_NUM = RANK_SIZE_PER_SERVER;

HcclResult GetChannelLink(HcclComm comm, uint32_t srcRank, uint32_t dstRank, CommLink &selectedLink)
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));

    for (uint32_t idx = 0; idx < netLayerNum; ++idx) {
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayers[idx], srcRank, dstRank, &links, &linkNum));
        if (links == nullptr) {
            continue;
        }

        // 同一对 rank 可能返回多条候选链路，其中占位链路的协议为 RESERVED(-1)，
        // 不能直接使用 links[0] 创建 Channel。
        for (uint32_t linkIdx = 0; linkIdx < linkNum; ++linkIdx) {
            if (links[linkIdx].linkAttr.linkProtocol == CommProtocol::COMM_PROTOCOL_RESERVED) {
                continue;
            }
            selectedLink = links[linkIdx];
            return HCCL_SUCCESS;
        }
    }

    HCCL_ERROR("No valid link found between rank %u and rank %u", srcRank, dstRank);
    return HCCL_E_NOT_SUPPORT;
}
} // namespace

HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    // 构造算子参数
    OpParam param;
    snprintf(param.tag, sizeof(param.tag), "%s", "hccl_custom_allreduce");
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = count;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;

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
    if (param.rankSize != RANK_SIZE_PER_SERVER * SERVER_NUM) {
        HCCL_ERROR("Only the 2-server x 8-rank topology is supported, rank size is %u", param.rankSize);
        return HCCL_E_NOT_SUPPORT;
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

        // 从通信域获取 HCCL Buffer（Device上的内存，默认总大小400MB）
        void *cclBufferAddr;
        uint64_t cclBufferSize;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        // ==============================================
        // STEP 2.2: 申请资源Thread和Channel
        // ==============================================

        // 创建 AICPU_TS 通信引擎上的 thread 资源
        uint32_t threadNum = COMM_THREAD_NUM;
        uint32_t notifyNumPerThread = COMM_THREAD_NUM;

        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, notifyNumPerThread, resCtxHost.threads.data()));
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        // Server 内连接其余 7 个 rank，Server 间连接相同 local-rank 的对端 rank。
        // 本地 Channel 分配独立通信 Thread，以并行利用 Full-Mesh 链路；跨 Server Channel 使用主 Thread。
        std::vector<uint32_t> peerRanks;
        std::vector<uint32_t> threadIndices;
        uint32_t serverStartRank = (param.myRank / RANK_SIZE_PER_SERVER) * RANK_SIZE_PER_SERVER;
        uint32_t workerThreadIdx = 1;
        for (uint32_t rank = serverStartRank; rank < serverStartRank + RANK_SIZE_PER_SERVER; ++rank) {
            if (rank != param.myRank) {
                peerRanks.push_back(rank);
                threadIndices.push_back(workerThreadIdx++);
            }
        }
        uint32_t crossServerRank = (param.myRank + RANK_SIZE_PER_SERVER) % param.rankSize;
        peerRanks.push_back(crossServerRank);
        threadIndices.push_back(0);

        uint32_t channelNum = peerRanks.size();
        if (channelNum > 0) {
            std::vector<HcclChannelDesc> channelDescs(channelNum);
            CHK_RET(HcclChannelDescInit(channelDescs.data(), channelNum));
            for (uint32_t idx = 0; idx < channelNum; ++idx) {
                CommLink link{};
                CHK_RET(GetChannelLink(comm, param.myRank, peerRanks[idx], link));
                channelDescs[idx].remoteRank = peerRanks[idx];
                channelDescs[idx].channelProtocol = link.linkAttr.linkProtocol;
                channelDescs[idx].localEndpoint = link.srcEndpointDesc;
                channelDescs[idx].remoteEndpoint = link.dstEndpointDesc;
                channelDescs[idx].notifyNum = CHANNEL_NOTIFY_NUM;
            }

            std::vector<ChannelHandle> channelHandles(channelNum);
            CHK_RET(HcclChannelAcquire(
                comm, aicpuTsEngine, channelDescs.data(), channelNum, channelHandles.data()));

            resCtxHost.channels.resize(channelNum);
            for (uint32_t idx = 0; idx < channelNum; ++idx) {
                ChannelInfo &channel = resCtxHost.channels[idx];
                channel.remoteRank = peerRanks[idx];
                channel.notifyNum = CHANNEL_NOTIFY_NUM;
                channel.threadIdx = threadIndices[idx];
                channel.handle = channelHandles[idx];
                CHK_RET(HcclChannelGetHcclBuffer(
                    comm, channel.handle, &channel.remoteCclMem.addr, &channel.remoteCclMem.size));
            }
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
