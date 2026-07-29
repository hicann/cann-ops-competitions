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

#include <vector>
#include <cstring>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

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
        // 大数据路径为每个远端 rank 分配 1 条 thread，使不同 channel 上的 Write 能并行执行。
        // threads[0] 同时承担 Host/Device 同步：notify 0 保留给 Host，其余 notify 用于等待 slave thread 完成。
        uint32_t threadNum = param.rankSize > 1 ? param.rankSize - 1 : 1;
        uint32_t notifyNumPerThread = threadNum;

        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, notifyNumPerThread, resCtxHost.threads.data()));
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        // 为本 rank 到所有其余 rank 各建 1 条 channel（全连接，对任意 root 通用）。
        // 保留每条 channel 的 layer 信息，便于后续比较直发与分层算法。
        constexpr uint32_t CHANNEL_NOTIFY_NUM = 2; // idx0=ACK, idx1=DATA_SIGNAL (见 common.h)
        std::vector<HcclChannelDesc> channelDescs;
        std::vector<uint32_t> channelLayers; // 与 channelDescs 并行，记录每条 channel 的 layer 编号
        uint32_t *netLayersPtr = nullptr;
        uint32_t netLayerNum = 0;
        CHK_RET(HcclRankGraphGetLayers(comm, &netLayersPtr, &netLayerNum));
        // 立即复制到本地 vector：GetLinks/GetRanksByLayer 等后续调用会使库内指针失效
        // （赛题环境已踩坑：循环里调 GetLinks 后 netLayers[0] 变垃圾值 874986464）
        std::vector<uint32_t> netLayers(netLayersPtr, netLayersPtr + netLayerNum);
        for (uint32_t remoteRank = 0; remoteRank < param.rankSize; remoteRank++) {
            if (remoteRank == param.myRank) {
                continue;
            }
            // 逐层查找本 rank 到 remoteRank 的可用链路（同 server 命中 layer0，跨 server 命中 layer1）
            bool foundLink = false;
            for (uint32_t layerIdx = 0; layerIdx < netLayerNum && !foundLink; layerIdx++) {
                CommLink *linkList = nullptr;
                uint32_t linkNum = 0;
                CHK_RET(HcclRankGraphGetLinks(comm, netLayers[layerIdx], param.myRank, remoteRank,
                                               &linkList, &linkNum));
                for (uint32_t i = 0; i < linkNum && !foundLink; i++) {
                    if (linkList[i].linkAttr.linkProtocol == COMM_PROTOCOL_RESERVED) {
                        continue;
                    }
                    HcclChannelDesc desc;
                    CHK_RET(HcclChannelDescInit(&desc, 1));
                    desc.remoteRank = remoteRank;
                    desc.localEndpoint = linkList[i].srcEndpointDesc;
                    desc.remoteEndpoint = linkList[i].dstEndpointDesc;
                    desc.channelProtocol = linkList[i].linkAttr.linkProtocol;
                    desc.notifyNum = CHANNEL_NOTIFY_NUM;
                    channelDescs.push_back(desc);
                    channelLayers.push_back(netLayers[layerIdx]); // 记录该 channel 走的 layer
                    foundLink = true;
                }
            }
            CHK_PRT_RET(!foundLink,
                        HCCL_ERROR("rank[%u] no available link to remoteRank[%u]", param.myRank, remoteRank),
                        HCCL_E_INTERNAL);
        }

        // 一次性申请所有 channel
        uint32_t channelNum = static_cast<uint32_t>(channelDescs.size());
        std::vector<ChannelHandle> channelHandles(channelNum);
        if (channelNum > 0) {
            CHK_RET(HcclChannelAcquire(comm, aicpuTsEngine, channelDescs.data(), channelNum,
                                       channelHandles.data()));
        }

        // 取每个对端的 cclBuffer 地址，组装 ChannelInfo（含 layer）存入资源上下文
        resCtxHost.channels.resize(channelNum);
        for (uint32_t i = 0; i < channelNum; i++) {
            ChannelInfo &info = resCtxHost.channels[i];
            info.remoteRank = channelDescs[i].remoteRank;
            info.notifyNum = CHANNEL_NOTIFY_NUM;
            info.handle = channelHandles[i];
            info.layer = channelLayers[i];
            void *remoteAddr = nullptr;
            uint64_t remoteSize = 0;
            CHK_RET(HcclChannelGetHcclBuffer(comm, channelHandles[i], &remoteAddr, &remoteSize));
            info.remoteCclMem = CommBuffer{remoteAddr, remoteSize};
        }

        // 查询并保留本 server（layer0）的 rank 列表，供后续分层算法对照实验复用。
        // sameServerRanks = 本 server 8 卡（含自己）；otherServerRanks = 对端 server 8 卡
        uint32_t *layer0Ranks = nullptr;
        uint32_t layer0RankNum = 0;
        CHK_RET(HcclRankGraphGetRanksByLayer(comm, netLayers[0], &layer0Ranks, &layer0RankNum));
        for (uint32_t i = 0; i < layer0RankNum; i++) {
            resCtxHost.sameServerRanks.push_back(layer0Ranks[i]);
        }
        for (uint32_t r = 0; r < param.rankSize; r++) {
            bool inSame = false;
            for (uint32_t i = 0; i < layer0RankNum; i++) {
                if (layer0Ranks[i] == r) {
                    inSame = true;
                    break;
                }
            }
            if (!inSame) {
                resCtxHost.otherServerRanks.push_back(r);
            }
        }
        HCCL_INFO("broadcast stage1: rank[%u] channels=%u sameServer=%zu otherServer=%zu cclBuf=%lu",
                  param.myRank, channelNum, resCtxHost.sameServerRanks.size(),
                  resCtxHost.otherServerRanks.size(), cclBufferSize);

        // ==============================================
        // STEP 2.3: 申请通信引擎上下文
        // ==============================================
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
