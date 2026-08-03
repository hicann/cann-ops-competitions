/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_diag.h>

#include <algorithm>
#include <vector>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

namespace {
bool IsPowerOfTwo(uint32_t value)
{
    return value != 0 && (value & (value - 1)) == 0;
}

HcclResult FillChannelDesc(HcclComm comm, uint32_t localRank, uint32_t remoteRank, HcclChannelDesc &desc)
{
    CHK_RET(HcclChannelDescInit(&desc, 1));
    constexpr CommProtocol preferredProtocol = CommProtocol::COMM_PROTOCOL_UBC_CTP;
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    CHK_PRT_RET(netLayers == nullptr || netLayerNum == 0,
        HCCL_ERROR("No communication layers for rank[%u]", localRank), HCCL_E_NOT_FOUND);

    for (uint32_t layerIdx = 0; layerIdx < netLayerNum; ++layerIdx) {
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        if (HcclRankGraphGetLinks(comm, netLayers[layerIdx], localRank, remoteRank, &links, &linkNum) != HCCL_SUCCESS) {
            continue;
        }
        for (uint32_t i = 0; i < linkNum; ++i) {
            if (links[i].linkAttr.linkProtocol != preferredProtocol) {
                continue;
            }
            desc.remoteRank = remoteRank;
            desc.notifyNum = custom_reducescatter::kChannelNotifyNum;
            desc.channelProtocol = links[i].linkAttr.linkProtocol;
            desc.localEndpoint.protocol = links[i].srcEndpointDesc.protocol;
            desc.localEndpoint.commAddr = links[i].srcEndpointDesc.commAddr;
            desc.localEndpoint.loc = links[i].srcEndpointDesc.loc;
            desc.remoteEndpoint.protocol = links[i].dstEndpointDesc.protocol;
            desc.remoteEndpoint.commAddr = links[i].dstEndpointDesc.commAddr;
            desc.remoteEndpoint.loc = links[i].dstEndpointDesc.loc;
            return HCCL_SUCCESS;
        }
    }

    HCCL_ERROR("UBC_CTP link not found between rank[%u] and rank[%u] on any layer", localRank, remoteRank);
    return HCCL_E_NOT_FOUND;
}

void FillFlatRemoteRanks(const OpParam &param, std::vector<uint32_t> &remoteRanks)
{
    remoteRanks.clear();
    remoteRanks.reserve(param.rankSize > 0 ? param.rankSize - 1 : 0);
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (rank != param.myRank) {
            remoteRanks.push_back(rank);
        }
    }
}

HcclResult BuildRemoteRanks(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtx,
    std::vector<uint32_t> &remoteRanks)
{
    FillFlatRemoteRanks(param, remoteRanks);
    if (param.rankSize <= 1) {
        return HCCL_SUCCESS;
    }
    const uint32_t dataTypeSize = SIZE_TABLE.at(param.dataType);
    if (param.count >= custom_reducescatter::kFlatTreeMinBytes / dataTypeSize) {
        return HCCL_SUCCESS;
    }

    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    if (HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum) != HCCL_SUCCESS || netLayers == nullptr ||
        netLayerNum < 2) {
        HCCL_WARNING("Two-level topology is unavailable, use flat ReduceScatter");
        return HCCL_SUCCESS;
    }
    const uint32_t localLayer = netLayers[0];
    std::vector<uint32_t> localRanks{param.myRank};
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (rank == param.myRank) {
            continue;
        }
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        if (HcclRankGraphGetLinks(comm, localLayer, param.myRank, rank, &links, &linkNum) != HCCL_SUCCESS ||
            links == nullptr) {
            continue;
        }
        for (uint32_t i = 0; i < linkNum; ++i) {
            if (links[i].linkAttr.linkProtocol == CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                localRanks.push_back(rank);
                break;
            }
        }
    }
    if (localRanks.size() <= 1 || localRanks.size() * 2 != param.rankSize) {
        HCCL_WARNING("Topology is not a balanced two-server layout, use flat ReduceScatter");
        return HCCL_SUCCESS;
    }
    std::sort(localRanks.begin(), localRanks.end());

    const auto myRankIter = std::find(localRanks.begin(), localRanks.end(), param.myRank);
    if (myRankIter == localRanks.end()) {
        HCCL_WARNING("Rank[%u] is absent from its local topology, use flat ReduceScatter", param.myRank);
        return HCCL_SUCCESS;
    }

    std::vector<uint32_t> otherRanks;
    otherRanks.reserve(localRanks.size());
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (!std::binary_search(localRanks.begin(), localRanks.end(), rank)) {
            otherRanks.push_back(rank);
        }
    }
    if (otherRanks.size() != localRanks.size()) {
        HCCL_WARNING("Failed to split ranks into two equal server groups, use flat ReduceScatter");
        return HCCL_SUCCESS;
    }

    const bool localIsGroup0 = localRanks.front() < otherRanks.front();
    resCtx.group0Ranks = localIsGroup0 ? localRanks : otherRanks;
    resCtx.group1Ranks = localIsGroup0 ? otherRanks : localRanks;
    resCtx.localGroupIndex = localIsGroup0 ? 0 : 1;
    resCtx.localRankIndex = static_cast<uint32_t>(std::distance(localRanks.begin(), myRankIter));
    resCtx.peerRank = otherRanks[resCtx.localRankIndex];

    const bool useSmallNhr = param.count <= custom_reducescatter::kSmallNhrMaxBytes / dataTypeSize &&
        IsPowerOfTwo(param.rankSize) && IsPowerOfTwo(static_cast<uint32_t>(localRanks.size()));
    if (useSmallNhr) {
        resCtx.hierarchyMode = custom_reducescatter::kHierarchySmallNhr;
        const uint32_t localRankSize = static_cast<uint32_t>(localRanks.size());
        const uint32_t algRank = resCtx.localGroupIndex * localRankSize + resCtx.localRankIndex;
        remoteRanks.clear();
        for (uint32_t mask = param.rankSize / 2; mask > 0; mask >>= 1) {
            const uint32_t peerAlgRank = algRank ^ mask;
            const uint32_t peerRank = peerAlgRank < localRankSize ? resCtx.group0Ranks[peerAlgRank] :
                resCtx.group1Ranks[peerAlgRank - localRankSize];
            remoteRanks.push_back(peerRank);
        }
        HCCL_INFO("Use small NHR ReduceScatter: rank[%u], algRank[%u], channelNum[%zu]",
            param.myRank, algRank, remoteRanks.size());
        return HCCL_SUCCESS;
    }

    resCtx.hierarchyMode = custom_reducescatter::kHierarchyTwoServer;

    remoteRanks.clear();
    remoteRanks.reserve(localRanks.size());
    for (uint32_t rank : localRanks) {
        if (rank != param.myRank) {
            remoteRanks.push_back(rank);
        }
    }
    remoteRanks.push_back(resCtx.peerRank);
    HCCL_INFO("Use two-server ReduceScatter: rank[%u], localIndex[%u], peerRank[%u], channelNum[%zu]",
        param.myRank, resCtx.localRankIndex, resCtx.peerRank, remoteRanks.size());
    return HCCL_SUCCESS;
}

HcclResult AcquireChannels(HcclComm comm, const std::vector<uint32_t> &remoteRanks, AlgResourceCtx &resCtx)
{
    if (remoteRanks.empty()) {
        return HCCL_SUCCESS;
    }

    const uint32_t channelNum = static_cast<uint32_t>(remoteRanks.size());
    std::vector<HcclChannelDesc> descs(channelNum);
    std::vector<ChannelHandle> handles(channelNum);
    uint32_t localRank = INVALID_VALUE_RANKID;
    CHK_RET(HcclGetRankId(comm, &localRank));
    for (uint32_t i = 0; i < channelNum; ++i) {
        CHK_PRT_RET(remoteRanks[i] == localRank,
            HCCL_ERROR("Remote rank list contains local rank[%u]", localRank), HCCL_E_PARA);
        CHK_PRT_RET(std::find(remoteRanks.begin(), remoteRanks.begin() + i, remoteRanks[i]) !=
                remoteRanks.begin() + i,
            HCCL_ERROR("Duplicate channel request for remote rank[%u]", remoteRanks[i]), HCCL_E_PARA);
        CHK_RET(FillChannelDesc(comm, localRank, remoteRanks[i], descs[i]));
    }

    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AICPU_TS, descs.data(), channelNum, handles.data()));
    resCtx.channels.reserve(channelNum);
    for (uint32_t i = 0; i < channelNum; ++i) {
        void *remoteBuffer = nullptr;
        uint64_t remoteBufferSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, handles[i], &remoteBuffer, &remoteBufferSize));
        CHK_PTR_NULL(remoteBuffer);

        ChannelInfo channel;
        channel.remoteRank = remoteRanks[i];
        channel.notifyNum = custom_reducescatter::kChannelNotifyNum;
        channel.handle = handles[i];
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
    CHK_PRT_RET(SIZE_TABLE.find(dataType) == SIZE_TABLE.end(), HCCL_ERROR("Unsupported data type[%d]", dataType),
        HCCL_E_NOT_SUPPORT);

    // 构造算子参数
    OpParam param;
    const uint32_t dataTypeSize = SIZE_TABLE.at(dataType);
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    // ==============================================
    // STEP 1: 解析拓扑信息
    // ==============================================
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    const bool useSmallNhr = param.rankSize > 1 && IsPowerOfTwo(param.rankSize) &&
        recvCount <= custom_reducescatter::kSmallNhrMaxBytes / dataTypeSize;
    const bool useFlatTree = recvCount >= custom_reducescatter::kFlatTreeMinBytes / dataTypeSize;
    sprintf(param.tag, "%s", useSmallNhr ? "hccl_custom_rs_small_nhr" :
        (useFlatTree ? "hccl_custom_rs_flat_tree" : "hccl_custom_rs_hierarchy"));

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

        // TODO: 根据通信算法申请 Thread 资源
        // 创建 AICPU_TS 通信引擎上的 thread 资源
        std::vector<uint32_t> remoteRanks;
        CHK_RET(BuildRemoteRanks(comm, param, resCtxHost, remoteRanks));

        uint32_t threadNum = resCtxHost.hierarchyMode == custom_reducescatter::kHierarchySmallNhr ? 1 :
            std::max<uint32_t>(1, static_cast<uint32_t>(remoteRanks.size()));
        uint32_t notifyNumPerThread = threadNum;

        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, notifyNumPerThread, resCtxHost.threads.data()));
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        CHK_RET(AcquireChannels(comm, remoteRanks, resCtxHost));

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
