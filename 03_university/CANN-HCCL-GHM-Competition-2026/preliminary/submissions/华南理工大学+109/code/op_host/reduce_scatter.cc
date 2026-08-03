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

HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    // 构造算子参数
    OpParam param;
    sprintf(param.tag, "%s", "hccl_custom_reduce_scatter_v3");
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

    // ==============================================
    // STEP 2: 创建资源
    // ==============================================
    CommEngine aicpuEngine = CommEngine::COMM_ENGINE_AICPU;
    CommEngine aicpuTsEngine = CommEngine::COMM_ENGINE_AICPU_TS;
    CommEngine cpuTsEngine = CommEngine::COMM_ENGINE_CPU_TS;

    // ==============================================
    // STEP 2.1: 申请用于 Host/Device 同步的通信资源
    // ==============================================
    CHK_RET(HcclThreadAcquireWithStream(comm, cpuTsEngine, stream, 1, &param.cpuThread));
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &param.cpuThread, aicpuTsEngine, &param.cpuThreadOnAicpu));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, aicpuEngine, &ctx, &size) == HCCL_SUCCESS) {
        // AICPU 资源已经存在，复用资源
        HCCL_INFO("Engine context already exists");
        param.resCtx = ctx;
        param.ctxSize = size;

        void *hostCtx = nullptr;
        uint64_t hostCtxSize = 0;
        CHK_RET(HcclEngineCtxGet(comm, param.tag, cpuTsEngine, &hostCtx, &hostCtxSize));
        ThreadHandle *aicpuThread = static_cast<ThreadHandle *>(hostCtx);
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));
    } else {
        // Device 资源不存在，资源构建
        AlgResourceCtx resCtxHost;

        // 从通信域获取 HCCL Buffer
        void *cclBufferAddr;
        uint64_t cclBufferSize;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        // ==============================================
        // STEP 2.2: 申请 Thread 资源
        // ==============================================
        // 1 个主线程归约，15 个从线程分别驱动到其余 rank 的链路。
        // 大包路径同时使用 7 条 Mesh 和 8 条 Clos 链路；小包仍可复用其中的
        // 前 7 个机内 worker。
        constexpr uint32_t kLocalRankNum = 8;
        constexpr uint32_t kRankNum = 16;
        uint32_t threadNum = kRankNum;
        // 主线程: notify 0 用于 Host 控制同步，1 - 15 用于接收从线程完成通知，
        // 16 - 19 用于 Direct 路径的四组局部归约完成通知。
        uint32_t notifyNumPerThread = kRankNum + 4;

        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, notifyNumPerThread, resCtxHost.threads.data()));
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        // ==============================================
        // STEP 2.3: 申请 Channel 资源
        // ==============================================
        // 为其余 15 个 rank 各申请一个 channel。每个对端始终只有一个
        // channel；大包直接使用所有 Clos/Mesh 链路，小包走分层路径。
        // ==============================================

        // 获取拓扑层次信息（layer 0 = intra-server, layer 1 = cross-server）
        uint32_t *netLayers = nullptr;
        uint32_t netLayerNum = 0;
        CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));

        CHK_PRT_RET(param.rankSize != 16 || netLayerNum < 2,
            HCCL_ERROR("This algorithm requires the competition 2x8 topology, rankSize=%u, layers=%u", param.rankSize,
                netLayerNum),
            HCCL_E_PARA);

        uint32_t *localRanks = nullptr;
        uint32_t localRankNum = 0;
        CHK_RET(HcclRankGraphGetRanksByLayer(comm, netLayers[0], &localRanks, &localRankNum));
        CHK_PRT_RET(localRankNum != kLocalRankNum,
            HCCL_ERROR("Expected 8 ranks in the intra-server layer, got %u", localRankNum), HCCL_E_INTERNAL);
        resCtxHost.localRanks.assign(localRanks, localRanks + localRankNum);
        std::sort(resCtxHost.localRanks.begin(), resCtxHost.localRanks.end());

        // 竞赛拓扑由两个连续编号的 8 卡 Server 组成，^8 保持 Server 内位置不变。
        resCtxHost.crossRank = param.myRank ^ kLocalRankNum;
        std::vector<uint32_t> peers;
        peers.reserve(kRankNum - 1);
        for (uint32_t rank = 0; rank < kRankNum; ++rank) {
            if (rank != param.myRank) {
                peers.push_back(rank);
            }
        }

        // Phase 1: 为每个对端准备 Channel 描述符，遍历所有 layer 查找 UBC_CTP 链路
        const uint32_t numChannels = static_cast<uint32_t>(peers.size());
        std::vector<HcclChannelDesc> channelDescs(numChannels);

        for (uint32_t i = 0; i < numChannels; i++) {
            uint32_t remoteRank = peers[i];
            CHK_RET(HcclChannelDescInit(&channelDescs[i], 1));

            bool protocolExists = false;
            for (uint32_t li = 0; li < netLayerNum && !protocolExists; li++) {
                CommLink *linkList = nullptr;
                uint32_t listSize = 0;
                CHK_RET(HcclRankGraphGetLinks(comm, netLayers[li], param.myRank, remoteRank, &linkList, &listSize));

                for (uint32_t idx = 0; idx < listSize; idx++) {
                    CommLink link = linkList[idx];
                    if (link.linkAttr.linkProtocol == CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                        channelDescs[i].remoteRank = remoteRank;
                        // 0 = data ready, 1 = receiver consumed; host/device 同步使用 thread notify。
                        channelDescs[i].notifyNum = 2;
                        channelDescs[i].channelProtocol = link.linkAttr.linkProtocol;
                        channelDescs[i].localEndpoint.protocol = link.srcEndpointDesc.protocol;
                        channelDescs[i].localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
                        channelDescs[i].localEndpoint.loc = link.srcEndpointDesc.loc;
                        channelDescs[i].remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
                        channelDescs[i].remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
                        channelDescs[i].remoteEndpoint.loc = link.dstEndpointDesc.loc;
                        protocolExists = true;
                        break;
                    }
                }
            }
            CHK_PRT_RET(!protocolExists, HCCL_ERROR("No UBC_CTP link found to rank %u in any layer", remoteRank),
                HCCL_E_INTERNAL);
        }

        // Phase 2: 批量创建所有 Channel（单次 HcclChannelAcquire 调用，利于框架集体同步）
        resCtxHost.channels.resize(numChannels);
        std::vector<ChannelHandle> handles(numChannels);
        CHK_RET(HcclChannelAcquire(comm, aicpuEngine, channelDescs.data(), numChannels, handles.data()));

        for (uint32_t i = 0; i < numChannels; i++) {
            resCtxHost.channels[i].handle = handles[i];
            resCtxHost.channels[i].remoteRank = peers[i];
            resCtxHost.channels[i].notifyNum = channelDescs[i].notifyNum;
        }

        // Phase 3: 获取远端 HCCL buffer 地址（所有 channel 创建完毕后统一交换，避免阻塞）
        for (uint32_t i = 0; i < numChannels; i++) {
            void *remoteAddr = nullptr;
            uint64_t remoteSize = 0;
            CHK_RET(HcclChannelGetHcclBuffer(comm, resCtxHost.channels[i].handle, &remoteAddr, &remoteSize));
            resCtxHost.channels[i].remoteCclMem = CommBuffer{remoteAddr, remoteSize};
        }

        // ==============================================
        // STEP 2.4: 申请通信引擎上下文
        // ==============================================
        std::vector<char> seq = resCtxHost.Serialize();
        uint64_t seqSize = seq.size();
        param.ctxSize = seqSize;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, aicpuEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, aicpuEngine, param.tag, seq.data(), seqSize, 0));

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
