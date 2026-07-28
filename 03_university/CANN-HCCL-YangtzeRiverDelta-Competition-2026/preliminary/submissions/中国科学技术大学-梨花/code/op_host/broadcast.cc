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
#include <array>
#include <cstdio>
#include <limits>
#include <vector>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

namespace {
constexpr uint32_t ALGORITHM_FLAT = 0;
constexpr uint32_t ALGORITHM_TWO_DIMENSION = 1;
constexpr uint32_t COMPETITION_RANK_SIZE = 16;
constexpr uint32_t SERVER_RANK_SIZE = 8;
constexpr uint32_t FLAT_CHANNEL_NOTIFY_NUM = 2;
constexpr uint32_t FLAT_THREAD_NUM = 2;
constexpr uint32_t FLAT_THREAD_NOTIFY_NUM = FLAT_THREAD_NUM;
// 400MB+4B场景使用4MB细粒度Tile，单个50MB Slice最多需要13个Tile。
// 每个Tile分别占用StageReady和AllGatherReady两个Notify，额外保留TailReady和ChunkDone。
constexpr uint32_t PIPELINE_MAX_TILE_NUM = 13;
constexpr uint32_t TWO_DIMENSION_CHANNEL_NOTIFY_NUM = PIPELINE_MAX_TILE_NUM * 2 + 2;
constexpr uint32_t TWO_DIMENSION_THREAD_NUM = SERVER_RANK_SIZE;
constexpr uint32_t TWO_DIMENSION_THREAD_NOTIFY_NUM = TWO_DIMENSION_THREAD_NUM;
constexpr uint64_t TWO_DIMENSION_THRESHOLD = 1ULL * 1024 * 1024;

HcclResult FillChannelDesc(
    const CommLink &link, uint32_t remoteRank, uint32_t notifyNum, HcclChannelDesc &channelDesc)
{
    CHK_RET(HcclChannelDescInit(&channelDesc, 1));
    channelDesc.remoteRank = remoteRank;
    channelDesc.channelProtocol = link.linkAttr.linkProtocol;
    channelDesc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
    channelDesc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
    channelDesc.localEndpoint.loc = link.srcEndpointDesc.loc;
    channelDesc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
    channelDesc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
    channelDesc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
    channelDesc.notifyNum = notifyNum;
    return HCCL_SUCCESS;
}

HcclResult BuildChannelDesc(HcclComm comm, uint32_t localRank, uint32_t remoteRank, uint32_t notifyNum,
    HcclChannelDesc &channelDesc)
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));

    // 与 HCCL AICPU_TS 的协议选择顺序保持一致，兼容卡内 Mesh 与跨 Server 链路。
    constexpr std::array<CommProtocol, 4> supportedProtocols = {
        CommProtocol::COMM_PROTOCOL_UBC_CTP,
        CommProtocol::COMM_PROTOCOL_UBC_TP,
        CommProtocol::COMM_PROTOCOL_PCIE,
        CommProtocol::COMM_PROTOCOL_UBOE,
    };
    for (uint32_t layerIdx = 0; layerIdx < netLayerNum; ++layerIdx) {
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayers[layerIdx], localRank, remoteRank, &links, &linkNum));
        for (const CommProtocol protocol : supportedProtocols) {
            for (uint32_t linkIdx = 0; linkIdx < linkNum; ++linkIdx) {
                if (links[linkIdx].linkAttr.linkProtocol == protocol) {
                    return FillChannelDesc(links[linkIdx], remoteRank, notifyNum, channelDesc);
                }
            }
        }
    }

    HCCL_ERROR("[BuildChannelDesc] No AICPU_TS link found between rank %u and rank %u", localRank, remoteRank);
    return HCCL_E_NOT_FOUND;
}

void AddUniqueRank(std::vector<uint32_t> &ranks, uint32_t rank)
{
    for (const uint32_t existingRank : ranks) {
        if (existingRank == rank) {
            return;
        }
    }
    ranks.push_back(rank);
}

void BuildRemoteRanks(const OpParam &param, uint32_t algorithmMode, std::vector<uint32_t> &remoteRanks)
{
    if (algorithmMode == ALGORITHM_TWO_DIMENSION) {
        const uint32_t serverBase = param.myRank / SERVER_RANK_SIZE * SERVER_RANK_SIZE;
        for (uint32_t localIdx = 0; localIdx < SERVER_RANK_SIZE; ++localIdx) {
            const uint32_t rank = serverBase + localIdx;
            if (rank != param.myRank) {
                AddUniqueRank(remoteRanks, rank);
            }
        }
        AddUniqueRank(remoteRanks, param.myRank ^ SERVER_RANK_SIZE);
        return;
    }

    if (param.myRank == param.root) {
        remoteRanks.reserve(param.rankSize - 1);
        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            if (rank != param.root) {
                remoteRanks.push_back(rank);
            }
        }
    } else {
        remoteRanks.push_back(param.root);
    }
}

HcclResult CreateChannels(HcclComm comm, CommEngine engine, const OpParam &param, AlgResourceCtx &resCtx)
{
    std::vector<uint32_t> remoteRanks;
    BuildRemoteRanks(param, resCtx.algorithmMode, remoteRanks);
    const uint32_t notifyNum = resCtx.algorithmMode == ALGORITHM_TWO_DIMENSION ?
        TWO_DIMENSION_CHANNEL_NOTIFY_NUM : FLAT_CHANNEL_NOTIFY_NUM;

    std::vector<HcclChannelDesc> channelDescs(remoteRanks.size());
    std::vector<ChannelHandle> channelHandles(remoteRanks.size());
    for (size_t idx = 0; idx < remoteRanks.size(); ++idx) {
        CHK_RET(BuildChannelDesc(comm, param.myRank, remoteRanks[idx], notifyNum, channelDescs[idx]));
    }
    if (!channelDescs.empty()) {
        CHK_RET(HcclChannelAcquire(
            comm, engine, channelDescs.data(), static_cast<uint32_t>(channelDescs.size()), channelHandles.data()));
    }

    resCtx.channels.reserve(remoteRanks.size());
    for (size_t idx = 0; idx < remoteRanks.size(); ++idx) {
        void *remoteBufferAddr = nullptr;
        uint64_t remoteBufferSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, channelHandles[idx], &remoteBufferAddr, &remoteBufferSize));

        ChannelInfo channel;
        channel.remoteRank = remoteRanks[idx];
        channel.notifyNum = notifyNum;
        channel.handle = channelHandles[idx];
        channel.remoteCclMem = CommBuffer{remoteBufferAddr, remoteBufferSize};
        resCtx.channels.push_back(channel);
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclBroadcast(
    void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(buf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    // 构造算子参数；资源 tag 在解析 rankSize 和算法模式后生成。
    OpParam param;
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
    CHK_PRT_RET(root >= param.rankSize,
        HCCL_ERROR("[HcclBroadcast] Invalid root %u for rank size %u", root, param.rankSize), HCCL_E_PARA);
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("[HcclBroadcast] Only FP32 is supported, data type is %d", static_cast<int>(dataType)),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(count > std::numeric_limits<uint64_t>::max() / sizeof(float),
        HCCL_ERROR("[HcclBroadcast] Data size overflows uint64_t, count is %llu",
            static_cast<unsigned long long>(count)), HCCL_E_PARA);

    const uint64_t dataSize = count * sizeof(float);
    const bool useTwoDimension =
        param.rankSize == COMPETITION_RANK_SIZE && dataSize > TWO_DIMENSION_THRESHOLD;
    const char *algorithmTag = useTwoDimension ? "sag" : "flat";
    const int tagLength = std::snprintf(
        param.tag, sizeof(param.tag), "hccl_custom_broadcast_root_%u_%s", root, algorithmTag);
    CHK_PRT_RET(tagLength < 0 || static_cast<size_t>(tagLength) >= sizeof(param.tag),
        HCCL_ERROR("[HcclBroadcast] Failed to build resource tag"), HCCL_E_PARA);

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

        // 小消息使用 2 个 Thread 并行放行接收端；大消息使用 8 个 Thread 并行处理 8 个 Slice。
        resCtxHost.algorithmMode = useTwoDimension ? ALGORITHM_TWO_DIMENSION : ALGORITHM_FLAT;
        const uint32_t threadNum = useTwoDimension ? TWO_DIMENSION_THREAD_NUM : FLAT_THREAD_NUM;
        const uint32_t notifyNumPerThread =
            useTwoDimension ? TWO_DIMENSION_THREAD_NOTIFY_NUM : FLAT_THREAD_NOTIFY_NUM;

        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, notifyNumPerThread, resCtxHost.threads.data()));
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        if (param.rankSize > 1) {
            CHK_RET(CreateChannels(comm, aicpuTsEngine, param, resCtxHost));
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
