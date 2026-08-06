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
constexpr uint32_t SERVER_NUM = 2;
constexpr uint32_t RANKS_PER_SERVER = 8;
constexpr uint32_t EXPECTED_RANK_SIZE = SERVER_NUM * RANKS_PER_SERVER;
constexpr uint32_t CHANNEL_NOTIFY_NUM = 2;
constexpr uint32_t MESH_NET_LAYER = 0;
constexpr uint32_t CLOS_NET_LAYER = 1;

HcclResult FillChannelDesc(HcclComm comm, uint32_t localRank, uint32_t remoteRank,
    const uint32_t *netLayers, uint32_t netLayerNum, HcclChannelDesc &desc)
{
    CHK_RET(HcclChannelDescInit(&desc, 1));
    constexpr CommProtocol requiredProtocol = CommProtocol::COMM_PROTOCOL_UBC_CTP;
    const bool isSameServer =
        localRank / RANKS_PER_SERVER == remoteRank / RANKS_PER_SERVER;
    const uint32_t preferredLayer = isSameServer ? MESH_NET_LAYER : CLOS_NET_LAYER;

    for (uint32_t pass = 0; pass < 2; ++pass) {
        for (uint32_t layerIdx = 0; layerIdx < netLayerNum; ++layerIdx) {
            const uint32_t netLayer = netLayers[layerIdx];
            if ((pass == 0 && netLayer != preferredLayer) ||
                (pass == 1 && netLayer == preferredLayer)) {
                continue;
            }

            CommLink *linkList = nullptr;
            uint32_t linkNum = 0;
            CHK_RET(HcclRankGraphGetLinks(
                comm, netLayer, localRank, remoteRank, &linkList, &linkNum));
            for (uint32_t linkIdx = 0; linkIdx < linkNum; ++linkIdx) {
                const CommLink &link = linkList[linkIdx];
                if (link.linkAttr.linkProtocol != requiredProtocol) {
                    continue;
                }
                desc.remoteRank = remoteRank;
                desc.notifyNum = CHANNEL_NOTIFY_NUM;
                desc.channelProtocol = link.linkAttr.linkProtocol;
                desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
                desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
                desc.localEndpoint.loc = link.srcEndpointDesc.loc;
                desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
                desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
                desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
                HCCL_INFO("Use net layer %u between rank %u and rank %u",
                    netLayer, localRank, remoteRank);
                return HCCL_SUCCESS;
            }
        }
    }

    HCCL_ERROR("UBC_CTP link not found in %u net layers between rank %u and rank %u",
        netLayerNum, localRank, remoteRank);
    return HCCL_E_NOT_FOUND;
}

std::vector<uint32_t> BuildRemoteRanks(uint32_t myRank, uint32_t rankSize)
{
    std::vector<uint32_t> remoteRanks;
    if (rankSize == EXPECTED_RANK_SIZE) {
        const uint32_t serverBase = myRank / RANKS_PER_SERVER * RANKS_PER_SERVER;
        for (uint32_t rank = serverBase; rank < serverBase + RANKS_PER_SERVER; ++rank) {
            if (rank != myRank) {
                remoteRanks.push_back(rank);
            }
        }
        // Channel/Thread 7 remains the paired cross-server channel used by
        // the small-message hierarchical algorithm.
        const uint32_t pairedRank =
            (myRank + RANKS_PER_SERVER) % EXPECTED_RANK_SIZE;
        remoteRanks.push_back(pairedRank);

        // Add one unique layer-1 channel for every remaining rank on the
        // remote server. Large messages use all 15 peer channels concurrently.
        const uint32_t remoteServerBase =
            (serverBase + RANKS_PER_SERVER) % EXPECTED_RANK_SIZE;
        for (uint32_t rank = remoteServerBase;
             rank < remoteServerBase + RANKS_PER_SERVER; ++rank) {
            if (rank != pairedRank) {
                remoteRanks.push_back(rank);
            }
        }
        return remoteRanks;
    }

    // 非赛题拓扑使用直接 AllGather，便于单元测试和接口复用。
    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        if (rank != myRank) {
            remoteRanks.push_back(rank);
        }
    }
    return remoteRanks;
}
} // namespace

HcclResult HcclAllGather(
    void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    // 构造算子参数
    OpParam param;
    sprintf(param.tag, "%s", "hccl_custom_allgather_v29_small_2worker_decouple");
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = sendCount;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;

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

        const std::vector<uint32_t> remoteRanks = BuildRemoteRanks(param.myRank, param.rankSize);
        // 每条 Channel 使用独立 Thread，并额外保留一个只执行本地输出复制的Thread。
        uint32_t threadNum = remoteRanks.empty() ? 1 : remoteRanks.size() + 1;
        // 主 Thread 等待其余 Thread，并为分层算法的跨 Server 阶段保留一个 Notify。
        uint32_t notifyNumPerThread = threadNum;

        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, notifyNumPerThread, resCtxHost.threads.data()));
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        if (!remoteRanks.empty()) {
            uint32_t *netLayers = nullptr;
            uint32_t netLayerNum = 0;
            CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
            CHK_PRT_RET(netLayers == nullptr || netLayerNum == 0,
                HCCL_ERROR("No net layer found in the current communication domain"),
                HCCL_E_NOT_FOUND);

            const uint32_t channelNum = remoteRanks.size();
            std::vector<HcclChannelDesc> channelDescs(channelNum);
            std::vector<ChannelHandle> channelHandles(channelNum);
            for (uint32_t idx = 0; idx < channelNum; ++idx) {
                CHK_RET(FillChannelDesc(comm, param.myRank, remoteRanks[idx],
                    netLayers, netLayerNum, channelDescs[idx]));
            }

            CHK_RET(HcclChannelAcquire(
                comm, aicpuTsEngine, channelDescs.data(), channelNum, channelHandles.data()));
            resCtxHost.channels.resize(channelNum);
            for (uint32_t idx = 0; idx < channelNum; ++idx) {
                void *remoteBufferAddr = nullptr;
                uint64_t remoteBufferSize = 0;
                CHK_RET(HcclChannelGetHcclBuffer(
                    comm, channelHandles[idx], &remoteBufferAddr, &remoteBufferSize));
                resCtxHost.channels[idx] =
                    ChannelInfo{remoteRanks[idx], CHANNEL_NOTIFY_NUM, channelHandles[idx],
                        CommBuffer{remoteBufferAddr, remoteBufferSize}};
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
