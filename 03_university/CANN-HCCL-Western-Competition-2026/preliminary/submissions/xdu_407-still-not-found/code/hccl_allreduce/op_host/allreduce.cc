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
#include <cstdint>
#include <limits>
#include <vector>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

namespace {
constexpr uint32_t EXPECTED_RANK_SIZE = 16;
constexpr uint32_t LOCAL_MESH_RANK_SIZE = 8;
constexpr uint32_t CHANNEL_NOTIFY_NUM = 4;
constexpr uint32_t THREAD_NUM = 16;
constexpr uint32_t INVALID_LAYER = std::numeric_limits<uint32_t>::max();
constexpr int UNSUPPORTED_PROTOCOL_PRIORITY = 3;

struct TopologyInfo {
    uint32_t meshLayer = INVALID_LAYER;
    uint32_t closLayer = INVALID_LAYER;
    std::vector<uint32_t> localRanks;
    std::vector<uint32_t> allRanks;
};

bool IsSortedUnique(const std::vector<uint32_t> &values)
{
    return std::adjacent_find(values.begin(), values.end()) == values.end();
}

bool ContainsRank(const std::vector<uint32_t> &ranks, uint32_t rank)
{
    return std::binary_search(ranks.begin(), ranks.end(), rank);
}

HcclResult GetSortedRanksByLayer(HcclComm comm, uint32_t layer, std::vector<uint32_t> &ranks)
{
    uint32_t *rankArray = nullptr;
    uint32_t rankNum = 0;
    CHK_RET(HcclRankGraphGetRanksByLayer(comm, layer, &rankArray, &rankNum));
    if (rankArray == nullptr || rankNum == 0) {
        HCCL_ERROR("Layer [%u] returned an empty rank list", layer);
        return HCCL_E_INTERNAL;
    }

    ranks.assign(rankArray, rankArray + rankNum);
    std::sort(ranks.begin(), ranks.end());
    if (!IsSortedUnique(ranks)) {
        HCCL_ERROR("Layer [%u] returned duplicate ranks", layer);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult DiscoverTopology(HcclComm comm, uint32_t myRank, TopologyInfo &topology)
{
    uint32_t *layerArray = nullptr;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layerArray, &layerNum));
    if (layerArray == nullptr || layerNum == 0) {
        HCCL_ERROR("Rank graph returned no network layers");
        return HCCL_E_INTERNAL;
    }

    std::vector<uint32_t> layers(layerArray, layerArray + layerNum);
    std::sort(layers.begin(), layers.end());
    if (!IsSortedUnique(layers)) {
        HCCL_ERROR("Rank graph returned duplicate network layers");
        return HCCL_E_INTERNAL;
    }

    for (uint32_t layer : layers) {
        CommTopo topo = COMM_TOPO_RESERVED;
        CHK_RET(HcclRankGraphGetTopoTypeByLayer(comm, layer, &topo));
        if (topo != COMM_TOPO_1DMESH && topo != COMM_TOPO_CUSTOM && topo != COMM_TOPO_CLOS) {
            continue;
        }

        std::vector<uint32_t> ranks;
        CHK_RET(GetSortedRanksByLayer(comm, layer, ranks));
        if (!ContainsRank(ranks, myRank)) {
            HCCL_ERROR("Rank [%u] is missing from layer [%u]", myRank, layer);
            return HCCL_E_INTERNAL;
        }

        if ((topo == COMM_TOPO_1DMESH || topo == COMM_TOPO_CUSTOM) && ranks.size() == LOCAL_MESH_RANK_SIZE
            && topology.meshLayer == INVALID_LAYER) {
            topology.meshLayer = layer;
            topology.localRanks = ranks;
        } else if (topo == COMM_TOPO_CLOS && ranks.size() == EXPECTED_RANK_SIZE
                   && topology.closLayer == INVALID_LAYER) {
            topology.closLayer = layer;
            topology.allRanks = ranks;
        }
    }

    if (topology.meshLayer == INVALID_LAYER || topology.closLayer == INVALID_LAYER
        || topology.localRanks.size() != LOCAL_MESH_RANK_SIZE || topology.allRanks.size() != EXPECTED_RANK_SIZE) {
        HCCL_ERROR("Required 8-rank local and 16-rank Clos layers were not found");
        return HCCL_E_NOT_SUPPORT;
    }
    for (uint32_t rank : topology.localRanks) {
        if (!ContainsRank(topology.allRanks, rank)) {
            HCCL_ERROR("Mesh rank [%u] is missing from the all-ranks Clos layer", rank);
            return HCCL_E_INTERNAL;
        }
    }
    return HCCL_SUCCESS;
}

int ProtocolPriority(CommTopo topo, CommProtocol protocol)
{
    if (topo == COMM_TOPO_1DMESH) {
        if (protocol == COMM_PROTOCOL_UBC_CTP) {
            return 0;
        }
        if (protocol == COMM_PROTOCOL_UBC_TP) {
            return 1;
        }
        if (protocol == COMM_PROTOCOL_UBOE) {
            return 2;
        }
    } else if (topo == COMM_TOPO_CLOS) {
        if (protocol == COMM_PROTOCOL_UBOE) {
            return 0;
        }
        if (protocol == COMM_PROTOCOL_UBC_CTP) {
            return 1;
        }
        if (protocol == COMM_PROTOCOL_UBC_TP) {
            return 2;
        }
    }
    return UNSUPPORTED_PROTOCOL_PRIORITY;
}

const CommLink *SelectLink(const CommLink *links, uint32_t linkNum, CommTopo topo)
{
    const CommLink *bestLink = nullptr;
    int bestPriority = UNSUPPORTED_PROTOCOL_PRIORITY;
    uint8_t bestHop = std::numeric_limits<uint8_t>::max();
    for (uint32_t index = 0; index < linkNum; ++index) {
        const CommLink &link = links[index];
        const int priority = ProtocolPriority(topo, link.linkAttr.linkProtocol);
        if (priority == UNSUPPORTED_PROTOCOL_PRIORITY) {
            continue;
        }
        if (bestLink == nullptr || priority < bestPriority
            || (priority == bestPriority && link.linkAttr.hop < bestHop)) {
            bestLink = &link;
            bestPriority = priority;
            bestHop = link.linkAttr.hop;
        }
    }
    return bestLink;
}

HcclResult BuildChannelDescriptions(
    HcclComm comm, uint32_t myRank, const TopologyInfo &topology, std::vector<HcclChannelDesc> &descriptions)
{
    descriptions.reserve(EXPECTED_RANK_SIZE - 1);
    for (uint32_t remoteRank : topology.allRanks) {
        if (remoteRank == myRank) {
            continue;
        }

        const bool isLocalRank = ContainsRank(topology.localRanks, remoteRank);
        const uint32_t layer = isLocalRank ? topology.meshLayer : topology.closLayer;
        const CommTopo topo = isLocalRank ? COMM_TOPO_1DMESH : COMM_TOPO_CLOS;
        CommLink *linkArray = nullptr;
        uint32_t linkNum = 0;
        CHK_RET(HcclRankGraphGetLinks(comm, layer, myRank, remoteRank, &linkArray, &linkNum));
        if (linkArray == nullptr || linkNum == 0) {
            HCCL_ERROR("No link from rank [%u] to rank [%u] in layer [%u]", myRank, remoteRank, layer);
            return HCCL_E_NOT_SUPPORT;
        }

        const CommLink *selectedLink = SelectLink(linkArray, linkNum, topo);
        if (selectedLink == nullptr) {
            HCCL_ERROR(
                "No supported non-RoCE link from rank [%u] to rank [%u] in layer [%u]", myRank, remoteRank, layer);
            return HCCL_E_NOT_SUPPORT;
        }
        const CommLink link = *selectedLink;

        HcclChannelDesc description;
        CHK_RET(HcclChannelDescInit(&description, 1));
        description.remoteRank = remoteRank;
        description.channelProtocol = link.linkAttr.linkProtocol;
        description.localEndpoint = link.srcEndpointDesc;
        description.remoteEndpoint = link.dstEndpointDesc;
        description.notifyNum = CHANNEL_NOTIFY_NUM;
        descriptions.push_back(description);
    }

    if (descriptions.size() != EXPECTED_RANK_SIZE - 1) {
        HCCL_ERROR("Expected 15 channel descriptions, got [%zu]", descriptions.size());
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult AcquireChannels(HcclComm comm, CommEngine engine, const std::vector<HcclChannelDesc> &descriptions,
    std::vector<ChannelInfo> &channels)
{
    std::vector<ChannelHandle> handles(descriptions.size());
    CHK_RET(HcclChannelAcquire(
        comm, engine, descriptions.data(), static_cast<uint32_t>(descriptions.size()), handles.data()));

    channels.resize(descriptions.size());
    for (size_t index = 0; index < descriptions.size(); ++index) {
        void *remoteBufferAddr = nullptr;
        uint64_t remoteBufferSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, handles[index], &remoteBufferAddr, &remoteBufferSize));
        if (remoteBufferAddr == nullptr || remoteBufferSize == 0) {
            HCCL_ERROR("Remote CCL buffer for rank [%u] is invalid", descriptions[index].remoteRank);
            return HCCL_E_INTERNAL;
        }

        channels[index].remoteRank = descriptions[index].remoteRank;
        channels[index].notifyNum = descriptions[index].notifyNum;
        channels[index].handle = handles[index];
        channels[index].remoteCclMem = CommBuffer{remoteBufferAddr, remoteBufferSize};
    }
    return HCCL_SUCCESS;
}

HcclResult ValidateResourceContext(const AlgResourceCtx &resource, uint32_t myRank)
{
    if (resource.rootRank == INVALID_VALUE_RANKID || resource.allRanks.size() != EXPECTED_RANK_SIZE
        || resource.localRanks.size() != LOCAL_MESH_RANK_SIZE || !IsSortedUnique(resource.allRanks)
        || !IsSortedUnique(resource.localRanks) || resource.rootRank != resource.allRanks.front()
        || !ContainsRank(resource.allRanks, myRank) || !ContainsRank(resource.localRanks, myRank)) {
        HCCL_ERROR("Resource topology metadata is invalid for rank [%u]", myRank);
        return HCCL_E_INTERNAL;
    }
    if (resource.localBuffer.addr == nullptr || resource.localBuffer.size == 0) {
        HCCL_ERROR("Local CCL buffer is invalid");
        return HCCL_E_INTERNAL;
    }
    if (resource.threads.size() != THREAD_NUM || resource.aicpuThread != resource.threads[0]) {
        HCCL_ERROR("Expected exactly one AICPU_TS algorithm thread");
        return HCCL_E_INTERNAL;
    }
    if (resource.channels.size() != 15) {
        HCCL_ERROR("Expected 15 channels, got [%zu]", resource.channels.size());
        return HCCL_E_INTERNAL;
    }

    std::vector<uint32_t> remoteRanks;
    remoteRanks.reserve(resource.channels.size());
    for (const ChannelInfo &channel : resource.channels) {
        if (channel.remoteRank == myRank || !ContainsRank(resource.allRanks, channel.remoteRank)
            || channel.notifyNum != CHANNEL_NOTIFY_NUM || channel.remoteCclMem.addr == nullptr
            || channel.remoteCclMem.size == 0) {
            HCCL_ERROR("Channel metadata for remote rank [%u] is invalid", channel.remoteRank);
            return HCCL_E_INTERNAL;
        }
        remoteRanks.push_back(channel.remoteRank);
    }
    std::sort(remoteRanks.begin(), remoteRanks.end());
    if (!IsSortedUnique(remoteRanks)) {
        HCCL_ERROR("Channel metadata contains duplicate remote ranks");
        return HCCL_E_INTERNAL;
    }

    std::vector<uint32_t> expectedRemoteRanks;
    for (uint32_t rank : resource.allRanks) {
        if (rank != myRank) {
            expectedRemoteRanks.push_back(rank);
        }
    }
    if (remoteRanks != expectedRemoteRanks) {
        HCCL_ERROR("Channel metadata does not cover every remote rank");
        return HCCL_E_INTERNAL;
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
    if (dataType != HCCL_DATA_TYPE_FP32 || op != HCCL_REDUCE_SUM) {
        HCCL_ERROR("Only FP32 SUM AllReduce is supported");
        return HCCL_E_NOT_SUPPORT;
    }
    if (count > std::numeric_limits<uint64_t>::max() / sizeof(float)) {
        HCCL_ERROR("AllReduce byte count overflows uint64_t");
        return HCCL_E_PARA;
    }

    uint32_t rankSize = 0;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    if (rankSize != 16) {
        HCCL_ERROR("AllReduce requires exactly 16 ranks, got [%u]", rankSize);
        return HCCL_E_NOT_SUPPORT;
    }
    if (count == 0) {
        return HCCL_SUCCESS;
    }

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
    param.rankSize = rankSize;

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

        TopologyInfo topology;
        CHK_RET(DiscoverTopology(comm, param.myRank, topology));
        resCtxHost.allRanks = topology.allRanks;
        resCtxHost.localRanks = topology.localRanks;
        resCtxHost.rootRank = topology.allRanks.front();

        // ==============================================
        // STEP 2.2: 申请资源Thread和Channel
        // ==============================================

        // 创建 AICPU_TS 通信引擎上的 thread 资源
        resCtxHost.threads.resize(THREAD_NUM);
        CHK_RET(HcclThreadAcquire(
            comm, aicpuTsEngine, THREAD_NUM, HCCL_CUSTOM_THREAD_NOTIFY_NUM, resCtxHost.threads.data()));
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        std::vector<HcclChannelDesc> channelDescriptions;
        CHK_RET(BuildChannelDescriptions(comm, param.myRank, topology, channelDescriptions));
        CHK_RET(AcquireChannels(comm, aicpuTsEngine, channelDescriptions, resCtxHost.channels));
        CHK_RET(ValidateResourceContext(resCtxHost, param.myRank));

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