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
#include <cstdio>
#include <limits>
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

HcclResult FillChannelDescByLink(HcclComm comm, uint32_t myRank, uint32_t remoteRank, HcclChannelDesc &channelDesc)
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));

    for (uint32_t layerIdx = 0; layerIdx < netLayerNum; ++layerIdx) {
        CommLink *linkList = nullptr;
        uint32_t listSize = 0;
        HcclResult ret = HcclRankGraphGetLinks(comm, netLayers[layerIdx], myRank, remoteRank, &linkList, &listSize);
        if (ret != HCCL_SUCCESS || linkList == nullptr || listSize == 0) {
            continue;
        }

        CommLink link = linkList[0];
        CHK_RET(HcclChannelDescInit(&channelDesc, 1));
        channelDesc.remoteRank = remoteRank;
        channelDesc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
        channelDesc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
        channelDesc.localEndpoint.loc = link.srcEndpointDesc.loc;
        channelDesc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
        channelDesc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
        channelDesc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
        channelDesc.channelProtocol = link.linkAttr.linkProtocol;
        channelDesc.notifyNum = CHANNEL_NOTIFY_NUM;
        return HCCL_SUCCESS;
    }

    HCCL_ERROR("Failed to find link from rank[%u] to rank[%u]", myRank, remoteRank);
    return HCCL_E_PARA;
}

HcclResult BuildAllPeerChannels(HcclComm comm, const OpParam &param, CommEngine engine, AlgResourceCtx &resCtxHost)
{
    std::vector<HcclChannelDesc> channelDescs;
    std::vector<uint32_t> remoteRanks;
    channelDescs.reserve(param.rankSize > 0 ? param.rankSize - 1 : 0);
    remoteRanks.reserve(param.rankSize > 0 ? param.rankSize - 1 : 0);

    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (rank == param.myRank) {
            continue;
        }

        HcclChannelDesc channelDesc;
        CHK_RET(FillChannelDescByLink(comm, param.myRank, rank, channelDesc));
        channelDescs.push_back(channelDesc);
        remoteRanks.push_back(rank);
    }

    if (channelDescs.empty()) {
        return HCCL_SUCCESS;
    }

    std::vector<ChannelHandle> channelHandles(channelDescs.size());
    CHK_RET(HcclChannelAcquire(
        comm, engine, channelDescs.data(), static_cast<uint32_t>(channelDescs.size()), channelHandles.data()));

    resCtxHost.channels.resize(channelHandles.size());
    for (size_t idx = 0; idx < channelHandles.size(); ++idx) {
        void *remoteBufferAddr = nullptr;
        uint64_t remoteBufferSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, channelHandles[idx], &remoteBufferAddr, &remoteBufferSize));

        resCtxHost.channels[idx].remoteRank = remoteRanks[idx];
        resCtxHost.channels[idx].notifyNum = CHANNEL_NOTIFY_NUM;
        resCtxHost.channels[idx].handle = channelHandles[idx];
        resCtxHost.channels[idx].remoteCclMem = CommBuffer{remoteBufferAddr, remoteBufferSize};
        resCtxHost.maxSliceSize = std::min(resCtxHost.maxSliceSize, remoteBufferSize);
    }

    return HCCL_SUCCESS;
}

HcclResult RegisterBroadcastDfxInfo(const OpParam &param, char *commName, uint64_t dataSize)
{
    HcclDfxOpInfo dfxInfo{};
    dfxInfo.opType = static_cast<uint32_t>(param.opType);
    dfxInfo.dataType = static_cast<uint32_t>(param.dataType);
    dfxInfo.dataCount = param.count;
    dfxInfo.root = param.root;
    dfxInfo.engine = CommEngine::COMM_ENGINE_AICPU_TS;
    dfxInfo.cpuTsThread = param.cpuThread;
    dfxInfo.cpuWaitAicpuNotifyIdx = 0;
    dfxInfo.inputMemAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    dfxInfo.inputMemSize = dataSize;
    dfxInfo.outputMemAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    dfxInfo.outputMemSize = dataSize;
    const int tagLength = std::snprintf(dfxInfo.algTag, sizeof(dfxInfo.algTag), "%s", param.tag);
    if (tagLength < 0 || tagLength >= static_cast<int>(sizeof(dfxInfo.algTag))) {
        HCCL_ERROR("Failed to fill algorithm tag");
        return HCCL_E_INTERNAL;
    }
    return HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo));
}
} // namespace

HcclResult HcclBroadcast(
    void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(buf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    auto sizeIt = SIZE_TABLE.find(dataType);
    CHK_PRT_RET(sizeIt == SIZE_TABLE.end(), HCCL_ERROR("Unsupported data type[%d]", dataType), HCCL_E_PARA);

    // 构造算子参数
    OpParam param;
    const int tagLength = std::snprintf(param.tag, sizeof(param.tag), "%s", "hccl_custom_broadcast");
    CHK_PRT_RET(tagLength < 0 || tagLength >= static_cast<int>(sizeof(param.tag)),
        HCCL_ERROR("Failed to fill resource tag"), HCCL_E_INTERNAL);
    param.inputPtr = buf;
    param.outputPtr = buf;
    param.count = count;
    param.dataType = dataType;
    param.root = root;
    param.opType = HcclCMDType::HCCL_CMD_BROADCAST;

    // 获取通信域标识
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));

    // ==============================================
    // STEP 1: 解析拓扑信息
    // ==============================================
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || root >= param.rankSize,
        HCCL_ERROR("Invalid root[%u], rankSize[%u]", root, param.rankSize), HCCL_E_PARA);
    const uint64_t typeSize = sizeIt->second;
    CHK_PRT_RET(count > std::numeric_limits<uint64_t>::max() / typeSize,
        HCCL_ERROR("Broadcast data size overflow, count[%llu]", static_cast<unsigned long long>(count)), HCCL_E_PARA);
    const uint64_t dataSize = count * typeSize;

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
    CHK_RET(RegisterBroadcastDfxInfo(param, commName, dataSize));

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
        void *cclBufferAddr = nullptr;
        uint64_t cclBufferSize = 0;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};
        resCtxHost.maxSliceSize = cclBufferSize;

        // ==============================================
        // STEP 2.2: 申请资源Thread和Channel
        // ==============================================

        // 创建 AICPU_TS 通信引擎上的 thread 资源
        const uint32_t threadNum = param.rankSize > 1 ? param.rankSize - 1 : 1;
        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, threadNum, resCtxHost.threads.data()));
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        // 为本rank与所有其他rank申请Channel，用于小包Pull和大包Two-Shot。
        CHK_RET(BuildAllPeerChannels(comm, param, aicpuTsEngine, resCtxHost));

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
