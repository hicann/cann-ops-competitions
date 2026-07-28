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
constexpr uint32_t BASE_CHANNEL_NOTIFY_NUM = 5;
constexpr uint32_t PIPELINE_CHANNEL_NOTIFY_NUM = 9;
constexpr uint32_t MAX_AICPU_THREAD_NUM = custom_allreduce::PEER_SIZE;
constexpr uint32_t PIPELINE_WORKER_THREAD_NUM = 2;
constexpr uint32_t FUSED_ROUND_THREAD_NOTIFY_NUM
    = custom_allreduce::PEER_SIZE + custom_allreduce::LOCAL_RANK_SIZE;
constexpr uint32_t SMALL_ROOT_RANK_SIZE = 16;
constexpr uint64_t SMALL_ROOT_GATHER_MAX_BYTES = 512ULL * 1024ULL;
constexpr uint64_t FP32_BYTES = sizeof(float);
constexpr uint64_t SMALL_512K_COUNT = SMALL_ROOT_GATHER_MAX_BYTES / FP32_BYTES;
constexpr uint64_t MIDDLE_512M_COUNT = 512ULL * 1024ULL * 1024ULL / FP32_BYTES;
constexpr uint64_t LARGE_400M4B_COUNT = 400ULL * 1024ULL * 1024ULL / FP32_BYTES + 1ULL;
constexpr uint32_t TWO_SERVER_RANK_SIZE = 16;
constexpr uint32_t SERVER_CROSS_RANK_MASK = 8;

bool IsPowerOfTwo(uint32_t value)
{
    return value != 0 && (value & (value - 1U)) == 0;
}

void AppendUniqueRank(std::vector<uint32_t> &ranks, uint32_t rank)
{
    if (std::find(ranks.begin(), ranks.end(), rank) == ranks.end()) {
        ranks.push_back(rank);
    }
}

bool ShouldReserveSmallRootChannels(const OpParam &param)
{
    return param.rankSize == SMALL_ROOT_RANK_SIZE && param.count * FP32_BYTES < SMALL_ROOT_GATHER_MAX_BYTES;
}

void AppendTwoServerCrossButterflyRanks(std::vector<uint32_t> &ranks, const OpParam &param)
{
    if (param.rankSize != TWO_SERVER_RANK_SIZE) {
        return;
    }

    AppendUniqueRank(ranks, param.myRank ^ SERVER_CROSS_RANK_MASK);
    AppendUniqueRank(ranks, param.myRank ^ SERVER_CROSS_RANK_MASK ^ 1U);
    AppendUniqueRank(ranks, param.myRank ^ SERVER_CROSS_RANK_MASK ^ 2U);
    AppendUniqueRank(ranks, param.myRank ^ SERVER_CROSS_RANK_MASK ^ 4U);
}

std::vector<uint32_t> GetRequiredRemoteRanks(const OpParam &param)
{
    std::vector<uint32_t> remoteRanks;
    if (param.rankSize <= 1) {
        return remoteRanks;
    }

    if (param.rankSize == TWO_SERVER_RANK_SIZE && param.count == SMALL_512K_COUNT) {
        remoteRanks.reserve(4);
        AppendTwoServerCrossButterflyRanks(remoteRanks, param);
        return remoteRanks;
    }

    if (param.rankSize == custom_allreduce::RANK_SIZE) {
        remoteRanks.reserve(custom_allreduce::PEER_SIZE);
        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            if (rank != param.myRank) {
                remoteRanks.push_back(rank);
            }
        }
        return remoteRanks;
    }

    if (IsPowerOfTwo(param.rankSize)) {
        remoteRanks.reserve(8);
        for (uint32_t stepMask = 1; stepMask < param.rankSize; stepMask <<= 1U) {
            AppendUniqueRank(remoteRanks, param.myRank ^ stepMask);
        }
    } else if (param.myRank == param.root) {
        remoteRanks.reserve(param.rankSize - 1);
        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            if (rank != param.myRank) {
                remoteRanks.push_back(rank);
            }
        }
    } else {
        remoteRanks.push_back(param.root);
    }

    if (ShouldReserveSmallRootChannels(param)) {
        if (param.myRank == param.root) {
            for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
                if (rank != param.myRank) {
                    AppendUniqueRank(remoteRanks, rank);
                }
            }
        } else {
            AppendUniqueRank(remoteRanks, param.root);
        }
    }
    AppendTwoServerCrossButterflyRanks(remoteRanks, param);
    return remoteRanks;
}

HcclResult BuildUbcCtpChannelDesc(HcclComm comm, uint32_t srcRank, uint32_t dstRank, uint32_t notifyNum,
    HcclChannelDesc &desc, uint32_t &selectedNetLayer)
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    CHK_PRT_RET(netLayerNum == 0, HCCL_ERROR("No topology layer found"), HCCL_E_NOT_FOUND);
    CHK_PTR_NULL(netLayers);
    const std::vector<uint32_t> layerList(netLayers, netLayers + netLayerNum);

    CHK_RET(HcclChannelDescInit(&desc, 1));
    for (const uint32_t netLayer : layerList) {
        CommLink *linkList = nullptr;
        uint32_t linkNum = 0;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayer, srcRank, dstRank, &linkList, &linkNum));
        if (linkNum == 0) {
            continue;
        }
        CHK_PTR_NULL(linkList);

        for (uint32_t linkIdx = 0; linkIdx < linkNum; ++linkIdx) {
            const CommLink &link = linkList[linkIdx];
            if (link.linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                continue;
            }

            desc.remoteRank = dstRank;
            desc.notifyNum = notifyNum;
            desc.channelProtocol = link.linkAttr.linkProtocol;
            desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
            desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
            desc.localEndpoint.loc = link.srcEndpointDesc.loc;
            desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
            desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
            desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
            selectedNetLayer = netLayer;
            return HCCL_SUCCESS;
        }
    }

    HCCL_ERROR("UBC_CTP link not found between rank %u and rank %u", srcRank, dstRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult AcquireRequiredChannels(
    HcclComm comm, const OpParam &param, uint32_t notifyNum, AlgResourceCtx &resCtx)
{
    std::vector<uint32_t> remoteRanks = GetRequiredRemoteRanks(param);
    if (remoteRanks.empty()) {
        return HCCL_SUCCESS;
    }

    const uint32_t channelNum = static_cast<uint32_t>(remoteRanks.size());
    std::vector<HcclChannelDesc> channelDescs(channelNum);
    std::vector<uint32_t> channelNetLayers(channelNum, INVALID_VALUE_RANKID);
    for (uint32_t channelIdx = 0; channelIdx < channelNum; ++channelIdx) {
        CHK_RET(BuildUbcCtpChannelDesc(comm, param.myRank, remoteRanks[channelIdx], notifyNum,
            channelDescs[channelIdx], channelNetLayers[channelIdx]));
    }

    std::vector<ChannelHandle> channelHandles(channelNum);
    CHK_RET(HcclChannelAcquire(
        comm, CommEngine::COMM_ENGINE_AICPU_TS, channelDescs.data(), channelNum, channelHandles.data()));

    resCtx.channels.reserve(channelNum);
    for (uint32_t channelIdx = 0; channelIdx < channelNum; ++channelIdx) {
        void *remoteCclAddr = nullptr;
        uint64_t remoteCclSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(
            comm, channelHandles[channelIdx], &remoteCclAddr, &remoteCclSize));
        CHK_PTR_NULL(remoteCclAddr);
        CHK_PRT_RET(remoteCclSize == 0,
            HCCL_ERROR("Remote CCL buffer size is zero for rank %u", remoteRanks[channelIdx]), HCCL_E_MEMORY);

        ChannelInfo channel;
        channel.remoteRank = remoteRanks[channelIdx];
        channel.remoteRankIndex = remoteRanks[channelIdx];
        channel.netLayer = channelNetLayers[channelIdx];
        channel.notifyNum = notifyNum;
        channel.handle = channelHandles[channelIdx];
        channel.remoteCclMem = CommBuffer{remoteCclAddr, remoteCclSize};
        resCtx.channels.push_back(channel);
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("Only FP32 is supported, dataType=%d", static_cast<int32_t>(dataType)), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(op != HCCL_REDUCE_SUM,
        HCCL_ERROR("Only SUM is supported, reduceOp=%d", static_cast<int32_t>(op)), HCCL_E_NOT_SUPPORT);

    // 构造算子参数
    OpParam param;
    const bool use512KCompactResources = count == SMALL_512K_COUNT;
    const bool use512MPipelineResources = count == MIDDLE_512M_COUNT;
    const bool use400M4BPipelineResources = count == LARGE_400M4B_COUNT;
    const bool usePipelineResources = use512MPipelineResources || use400M4BPipelineResources;
    const char *resourceTag = use512KCompactResources ? "hccl_custom_allreduce_v99_512k" :
        (use512MPipelineResources ? "hccl_custom_allreduce_v99_512m" :
            (use400M4BPipelineResources ? "hccl_custom_allreduce_v99_400m4b" :
                "hccl_custom_allreduce_v99_base"));
    sprintf(param.tag, "%s", resourceTag);
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = count;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    const uint64_t totalBytes = count * FP32_BYTES;
    param.root = custom_allreduce::PackKernelConfig(
        totalBytes <= custom_allreduce::SMALL_THRESHOLD_BYTES
            ? custom_allreduce::AlgorithmType::OneShot16
            : custom_allreduce::AlgorithmType::DirectAuto,
        0);

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
    CHK_PRT_RET(param.rankSize == 0 || param.root >= param.rankSize,
        HCCL_ERROR("Invalid rankSize=%u or root=%u", param.rankSize, param.root), HCCL_E_PARA);

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
        resCtxHost.rankIndex = param.myRank;
        resCtxHost.ranks.resize(param.rankSize);
        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            resCtxHost.ranks[rank] = rank;
        }

        // ==============================================
        // STEP 2.2: 申请资源Thread和Channel
        // ==============================================

        // 创建 AICPU_TS 通信引擎上的 thread 资源
        uint32_t threadNum = use512KCompactResources
            ? 1U
            : std::min(MAX_AICPU_THREAD_NUM, param.rankSize - 1);
        uint32_t notifyNumPerThread = use512KCompactResources
            ? 1U
            : (use400M4BPipelineResources ? FUSED_ROUND_THREAD_NOTIFY_NUM : custom_allreduce::PEER_SIZE);

        const uint32_t workerThreadNum = usePipelineResources ? PIPELINE_WORKER_THREAD_NUM : 0U;
        std::vector<ThreadHandle> allThreads(threadNum + workerThreadNum);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, static_cast<uint32_t>(allThreads.size()),
            notifyNumPerThread, allThreads.data()));
        resCtxHost.threads.assign(allThreads.begin(), allThreads.begin() + threadNum);
        resCtxHost.workerThreads.assign(allThreads.begin() + threadNum, allThreads.end());
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        const uint32_t channelNotifyNum = use512KCompactResources
            ? 2U
            : (usePipelineResources ? PIPELINE_CHANNEL_NOTIFY_NUM : BASE_CHANNEL_NOTIFY_NUM);
        CHK_RET(AcquireRequiredChannels(comm, param, channelNotifyNum, resCtxHost));

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
