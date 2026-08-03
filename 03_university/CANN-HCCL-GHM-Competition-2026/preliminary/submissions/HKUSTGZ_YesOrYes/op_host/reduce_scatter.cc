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

#include <algorithm>
#include <vector>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

namespace {
constexpr uint32_t SMALL_CHANNEL_NOTIFY_NUM = 2;
constexpr uint32_t LARGE_CHANNEL_NOTIFY_NUM = CUSTOM_LARGE_TREE_CHANNEL_NOTIFY_NUM;

bool IsLargeMessage(uint64_t count)
{
    return count > CUSTOM_LARGE_MESSAGE_THRESHOLD_BYTES / sizeof(float);
}

bool IsSmallRecursiveHalving(uint64_t count, uint32_t rankSize)
{
    return rankSize == 2 * CUSTOM_COMPETITION_GROUP_RANK_NUM
        && count == CUSTOM_COMPETITION_512KB_RECV_COUNT;
}

bool IsRollingStripe512MB(uint64_t count, uint32_t rankSize)
{
    return rankSize == 2 * CUSTOM_COMPETITION_GROUP_RANK_NUM
        && count == CUSTOM_COMPETITION_512MB_RECV_COUNT;
}

void SetChannelDesc(uint32_t remoteRank, uint32_t notifyNum, const CommLink &link, HcclChannelDesc &desc)
{
    desc.remoteRank = remoteRank;
    desc.notifyNum = notifyNum;
    desc.channelProtocol = link.linkAttr.linkProtocol;
    desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
    desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
    desc.localEndpoint.loc = link.srcEndpointDesc.loc;
    desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
    desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
    desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
}

HcclResult FillChannelDesc(HcclComm comm, uint32_t localRank, uint32_t remoteRank,
    const std::vector<uint32_t> &netLayers, uint32_t notifyNum, HcclChannelDesc &desc)
{
    CHK_RET(HcclChannelDescInit(&desc, 1));
    CommLink fallbackLink;
    bool hasFallback = false;

    for (uint32_t netLayer : netLayers) {
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayer, localRank, remoteRank, &links, &linkNum));

        for (uint32_t linkIdx = 0; linkIdx < linkNum; ++linkIdx) {
            const CommLink &link = links[linkIdx];
            if (!hasFallback && link.linkAttr.linkProtocol != COMM_PROTOCOL_RESERVED) {
                fallbackLink = link;
                hasFallback = true;
            }
            if (link.linkAttr.linkProtocol == COMM_PROTOCOL_UBC_CTP) {
                SetChannelDesc(remoteRank, notifyNum, link, desc);
                return HCCL_SUCCESS;
            }
        }
    }

    if (hasFallback) {
        SetChannelDesc(remoteRank, notifyNum, fallbackLink, desc);
        return HCCL_SUCCESS;
    }

    HCCL_ERROR("[FillChannelDesc] No link found between rank %u and rank %u", localRank, remoteRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult AcquireChannels(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtx)
{
    if (param.rankSize <= 1) {
        return HCCL_SUCCESS;
    }

    uint32_t *layerList = nullptr;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layerList, &layerNum));
    CHK_PRT_RET(layerList == nullptr || layerNum == 0, HCCL_ERROR("[AcquireChannels] No network layer is available"),
        HCCL_E_NOT_FOUND);
    const std::vector<uint32_t> netLayers(layerList, layerList + layerNum);

    const bool useSmallRecursiveHalving = IsSmallRecursiveHalving(param.count, param.rankSize);
    const bool useRollingStripe512MB = IsRollingStripe512MB(param.count, param.rankSize);
    const uint32_t channelNum = useSmallRecursiveHalving
        ? CUSTOM_SMALL_RH_STEP_NUM
        : param.rankSize - 1;
    const uint32_t notifyNum = useSmallRecursiveHalving
        ? CUSTOM_SMALL_RH_CHANNEL_NOTIFY_NUM
        : (useRollingStripe512MB
            ? CUSTOM_ROLLING_CHANNEL_NOTIFY_NUM
            : (IsLargeMessage(param.count) ? LARGE_CHANNEL_NOTIFY_NUM : SMALL_CHANNEL_NOTIFY_NUM));
    std::vector<HcclChannelDesc> descs(channelNum);
    std::vector<ChannelHandle> handles(channelNum);
    std::vector<uint32_t> remoteRanks(channelNum);

    uint32_t channelIdx = 0;
    if (useSmallRecursiveHalving) {
        for (uint32_t step = 0; step < CUSTOM_SMALL_RH_STEP_NUM; ++step) {
            const uint32_t remoteRank = param.myRank ^ CUSTOM_SMALL_RH_PARTNER_DELTAS[step];
            CHK_RET(FillChannelDesc(comm, param.myRank, remoteRank, netLayers, notifyNum, descs[channelIdx]));
            remoteRanks[channelIdx] = remoteRank;
            ++channelIdx;
        }
    } else {
        for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
            if (remoteRank == param.myRank) {
                continue;
            }
            CHK_RET(FillChannelDesc(comm, param.myRank, remoteRank, netLayers, notifyNum, descs[channelIdx]));
            remoteRanks[channelIdx] = remoteRank;
            ++channelIdx;
        }
    }

    CHK_RET(HcclChannelAcquire(comm, COMM_ENGINE_AICPU_TS, descs.data(), channelNum, handles.data()));
    resCtx.channels.reserve(channelNum);
    for (uint32_t idx = 0; idx < channelNum; ++idx) {
        void *remoteBuffer = nullptr;
        uint64_t remoteBufferSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, handles[idx], &remoteBuffer, &remoteBufferSize));

        ChannelInfo channel;
        channel.remoteRank = remoteRanks[idx];
        channel.notifyNum = notifyNum;
        channel.handle = handles[idx];
        channel.remoteCclMem = CommBuffer{remoteBuffer, remoteBufferSize};
        resCtx.channels.push_back(channel);
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32 || op != HCCL_REDUCE_SUM,
        HCCL_ERROR("Only FP32 SUM is supported, data type %d, reduce op %d", dataType, op), HCCL_E_NOT_SUPPORT);

    // 构造算子参数
    OpParam param;
    const bool useLargeMessageResources = IsLargeMessage(recvCount);
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
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > CUSTOM_MAX_RANK_NUM,
        HCCL_ERROR("Competition kernel supports 1..%u ranks, got %u",
            CUSTOM_MAX_RANK_NUM, param.rankSize),
        HCCL_E_NOT_SUPPORT);
    const bool useSmallRecursiveHalving = IsSmallRecursiveHalving(recvCount, param.rankSize);
    const bool useRollingStripe512MB = IsRollingStripe512MB(recvCount, param.rankSize);
    sprintf(param.tag, "%s", useSmallRecursiveHalving
        ? "hccl_custom_reduce_scatter_512k_rh_v25"
        : (useRollingStripe512MB
            ? "hccl_custom_reduce_scatter_512m_rolling_v25"
            : (useLargeMessageResources
            ? "hccl_custom_reduce_scatter_large_tree_v25"
            : "hccl_custom_reduce_scatter_small")));

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

        // Large rank-16 messages separate 15 communication queues from eight
        // tree-reduction queues. Small messages retain their compact resources.
        const uint32_t remoteRankCount = param.rankSize > 1 ? param.rankSize - 1 : 0;
        const uint32_t smallWorkerNum = std::min(CUSTOM_SMALL_COMM_WORKER_NUM, remoteRankCount);
        uint32_t threadNum = param.rankSize <= 1
            ? 1
            : (useSmallRecursiveHalving
                ? CUSTOM_SMALL_RH_THREAD_NUM
                : (useLargeMessageResources
                    ? CUSTOM_LARGE_TREE_THREAD_NUM
                    : 1 + smallWorkerNum));
        resCtxHost.threads.resize(threadNum);
        const uint32_t threadNotifyNum = useLargeMessageResources
            ? CUSTOM_LARGE_TREE_THREAD_NUM
            : threadNum;
        CHK_RET(HcclThreadAcquire(
            comm, aicpuTsEngine, threadNum, threadNotifyNum, resCtxHost.threads.data()));
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        // 按 remote rank 升序创建全连接 Channel。比赛的 2x8 拓扑中，Server 内优先 UBC CTP，
        // 跨 Server 使用 Clos 链路；每个对端仍严格只申请一个 Channel。
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
