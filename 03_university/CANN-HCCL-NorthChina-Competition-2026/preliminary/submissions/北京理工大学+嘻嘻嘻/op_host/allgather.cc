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
#include <cstring>
#include <iterator>
#include <limits>
#include <utility>
#include <vector>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 2;
constexpr uint64_t HOST_RESOURCE_CTX_MAGIC = 0x4843434C41474358ULL;
constexpr uint32_t HOST_RESOURCE_CTX_VERSION = 6;
constexpr uint32_t HIERARCHICAL_RANK_SIZE = 16;
constexpr uint32_t HIERARCHICAL_GROUP_SIZE = 8;
constexpr uint32_t HIERARCHICAL_THREAD_NUM = HIERARCHICAL_RANK_SIZE;
constexpr uint32_t HIERARCHICAL_NOTIFY_NUM_PER_THREAD = HIERARCHICAL_THREAD_NUM;

// CPU Context 中保存的资源快照头，用于校验 Host 与 Device 使用同一代 AICPU 资源
struct HostResourceCtxHeader {
    uint64_t magic = HOST_RESOURCE_CTX_MAGIC;
    uint32_t version = HOST_RESOURCE_CTX_VERSION;
    uint32_t headerSize = 0;
    uint64_t serializedSize = 0;
    ThreadHandle aicpuThread = 0;
};

// 2x8 分层 AllGather 的 RankGraph 探测结果
struct HierarchicalPlan {
    bool enabled = false;
    uint32_t intraLayer = 0;
    uint32_t interLayer = 0;
    uint32_t mirrorRank = INVALID_VALUE_RANKID;
    std::vector<uint32_t> intraRanks;
};

// 单个 RankGraph 网络层的规范化拓扑信息
struct LayerTopology {
    uint32_t layer = 0;
    std::vector<uint32_t> ranks;
    std::vector<uint32_t> instSizes;
};

/**
 * @brief 校验排序后的 rank 组是否大小正确且全局 rank 连续
 * @param ranks 已按升序排列的 rank 组
 * @param expectedSize 期望的 rank 数量
 * @return 大小正确且 rank 连续返回 true，否则返回 false
 */
static bool IsContiguousRankGroup(const std::vector<uint32_t> &ranks, uint32_t expectedSize)
{
    if (ranks.size() != expectedSize || ranks.empty()) {
        return false;
    }
    for (uint32_t index = 1; index < expectedSize; ++index) {
        if (ranks[index] != ranks[0] + index) {
            return false;
        }
    }
    return true;
}

/**
 * @brief 从 RankGraph 探测两个连续 8-rank 实例及本 rank 的跨 Server 镜像 peer
 * @param comm 通信域句柄
 * @param param 当前 AllGather 的 rank 参数
 * @param orderedLayers 按编号升序排列的 RankGraph 网络层
 * @param plan 输出的分层 AllGather 计划，拓扑不匹配时保持禁用
 * @return 探测完成或拓扑不匹配返回 HCCL_SUCCESS，接口或拓扑数据非法时返回对应错误码
 */
static HcclResult DetectHierarchicalPlan(
    HcclComm comm, const OpParam &param, const std::vector<uint32_t> &orderedLayers, HierarchicalPlan &plan)
{
    if (param.rankSize != HIERARCHICAL_RANK_SIZE) {
        return HCCL_SUCCESS;
    }

    std::vector<LayerTopology> layerTopologies;
    for (uint32_t layer : orderedLayers) {
        uint32_t *rankList = nullptr;
        uint32_t rankNum = 0;
        CHK_RET(HcclRankGraphGetRanksByLayer(comm, layer, &rankList, &rankNum));
        CHK_PRT_RET(rankNum > 0 && rankList == nullptr,
            HCCL_ERROR("Invalid rank list for network layer %u", layer), HCCL_E_INTERNAL);
        std::vector<uint32_t> intraRanks;
        if (rankNum > 0) {
            intraRanks.assign(rankList, rankList + rankNum);
        }

        uint32_t *instSizeList = nullptr;
        uint32_t instNum = 0;
        CHK_RET(HcclRankGraphGetInstSizeListByLayer(comm, layer, &instSizeList, &instNum));
        CHK_PRT_RET(instNum > 0 && instSizeList == nullptr,
            HCCL_ERROR("Invalid instance size list for network layer %u", layer), HCCL_E_INTERNAL);
        std::vector<uint32_t> instSizes;
        if (instNum > 0) {
            instSizes.assign(instSizeList, instSizeList + instNum);
        }

        std::sort(intraRanks.begin(), intraRanks.end());
        CHK_PRT_RET(std::adjacent_find(intraRanks.begin(), intraRanks.end()) != intraRanks.end(),
            HCCL_ERROR("Duplicate rank in network layer %u", layer), HCCL_E_INTERNAL);
        for (uint32_t rank : intraRanks) {
            CHK_PRT_RET(rank >= param.rankSize,
                HCCL_ERROR("Invalid rank %u in network layer %u", rank, layer), HCCL_E_INTERNAL);
        }
        std::sort(instSizes.begin(), instSizes.end());
        LayerTopology topology;
        topology.layer = layer;
        topology.ranks = std::move(intraRanks);
        topology.instSizes = std::move(instSizes);
        layerTopologies.push_back(std::move(topology));
    }

    size_t intraIndex = layerTopologies.size();
    for (size_t index = 0; index < layerTopologies.size(); ++index) {
        const LayerTopology &topology = layerTopologies[index];
        if (topology.ranks.size() == HIERARCHICAL_GROUP_SIZE && topology.instSizes.size() == 2 &&
            topology.instSizes[0] == HIERARCHICAL_GROUP_SIZE &&
            topology.instSizes[1] == HIERARCHICAL_GROUP_SIZE &&
            std::binary_search(topology.ranks.begin(), topology.ranks.end(), param.myRank) &&
            IsContiguousRankGroup(topology.ranks, HIERARCHICAL_GROUP_SIZE)) {
            intraIndex = index;
            break;
        }
    }
    if (intraIndex == layerTopologies.size()) {
        return HCCL_SUCCESS;
    }

    size_t interIndex = layerTopologies.size();
    for (size_t index = intraIndex + 1; index < layerTopologies.size(); ++index) {
        const LayerTopology &topology = layerTopologies[index];
        if (topology.ranks.size() == param.rankSize && topology.instSizes.size() == 1 &&
            topology.instSizes[0] == param.rankSize && topology.ranks[0] == 0 &&
            IsContiguousRankGroup(topology.ranks, param.rankSize)) {
            interIndex = index;
            break;
        }
    }
    if (interIndex == layerTopologies.size()) {
        return HCCL_SUCCESS;
    }

    const std::vector<uint32_t> &intraRanks = layerTopologies[intraIndex].ranks;
    const std::vector<uint32_t> &fullRanks = layerTopologies[interIndex].ranks;
    std::vector<uint32_t> remoteRanks;
    for (uint32_t rank : fullRanks) {
        if (!std::binary_search(intraRanks.begin(), intraRanks.end(), rank)) {
            remoteRanks.push_back(rank);
        }
    }
    if (!IsContiguousRankGroup(remoteRanks, HIERARCHICAL_GROUP_SIZE)) {
        return HCCL_SUCCESS;
    }

    const uint32_t intraLayer = layerTopologies[intraIndex].layer;
    const uint32_t interLayer = layerTopologies[interIndex].layer;
    const auto localIter = std::lower_bound(intraRanks.begin(), intraRanks.end(), param.myRank);
    const uint32_t localIndex = static_cast<uint32_t>(std::distance(intraRanks.begin(), localIter));
    plan.enabled = true;
    plan.intraLayer = intraLayer;
    plan.interLayer = interLayer;
    plan.mirrorRank = remoteRanks[localIndex];
    plan.intraRanks = intraRanks;
    HCCL_INFO("Enable hierarchical AllGather, rank %u, intra layer %u, group start %u, mirror rank %u",
        param.myRank, plan.intraLayer, plan.intraRanks[0], plan.mirrorRank);
    return HCCL_SUCCESS;
}

/**
 * @brief 按 RankGraph 层级顺序选择 UBC_CTP 链路并填充单个 Channel 描述符
 * @param comm 通信域句柄
 * @param srcRank 本端全局 rank
 * @param dstRank 对端全局 rank
 * @param candidateLayers 按优先级排列的候选网络层
 * @param desc 输出的 Channel 描述符
 * @return 成功返回 HCCL_SUCCESS，接口失败或无可用链路时返回对应错误码
 */
static HcclResult FillChannelDesc(HcclComm comm, uint32_t srcRank, uint32_t dstRank,
    const std::vector<uint32_t> &candidateLayers, HcclChannelDesc &desc)
{
    CHK_PRT_RET(candidateLayers.empty(),
        HCCL_ERROR("No candidate network layer for rank %u and rank %u", srcRank, dstRank), HCCL_E_NOT_FOUND);
    CHK_RET(HcclChannelDescInit(&desc, 1));
    for (uint32_t netLayer : candidateLayers) {
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayer, srcRank, dstRank, &links, &linkNum));
        CHK_PRT_RET(linkNum > 0 && links == nullptr,
            HCCL_ERROR("Invalid link list between rank %u and rank %u", srcRank, dstRank), HCCL_E_INTERNAL);

        for (uint32_t linkIndex = 0; linkIndex < linkNum; ++linkIndex) {
            const CommLink &link = links[linkIndex];
            if (link.linkAttr.linkProtocol != COMM_PROTOCOL_UBC_CTP) {
                continue;
            }

            desc.remoteRank = dstRank;
            desc.channelProtocol = link.linkAttr.linkProtocol;
            desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
            desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
            desc.localEndpoint.loc = link.srcEndpointDesc.loc;
            desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
            desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
            desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
            desc.notifyNum = CHANNEL_NOTIFY_NUM;
            return HCCL_SUCCESS;
        }
    }

    HCCL_ERROR("UBC_CTP link not found between rank %u and rank %u", srcRank, dstRank);
    return HCCL_E_NOT_FOUND;
}

/**
 * @brief 按 RankGraph 计划为所需对端批量申请单 Channel 并记录远端 HCCL buffer
 * @param comm 通信域句柄
 * @param param 当前 AllGather 的 rank 参数
 * @param resCtxHost 输出的 Host 侧算法资源上下文
 * @return 成功返回 HCCL_SUCCESS，描述符构造或资源申请失败时返回对应错误码
 */
static HcclResult AcquirePeerChannels(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtxHost)
{
    if (param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    CHK_PRT_RET(netLayerNum == 0 || netLayers == nullptr,
        HCCL_ERROR("No network layer found for AllGather"), HCCL_E_NOT_FOUND);
    std::vector<uint32_t> orderedLayers(netLayers, netLayers + netLayerNum);
    std::sort(orderedLayers.begin(), orderedLayers.end());
    CHK_PRT_RET(std::adjacent_find(orderedLayers.begin(), orderedLayers.end()) != orderedLayers.end(),
        HCCL_ERROR("Duplicate RankGraph network layer"), HCCL_E_INTERNAL);

    HierarchicalPlan plan;
    CHK_RET(DetectHierarchicalPlan(comm, param, orderedLayers, plan));

    std::vector<HcclChannelDesc> channelDescs;
    if (plan.enabled) {
        const std::vector<uint32_t> intraLayers{plan.intraLayer};
        const std::vector<uint32_t> interLayers{plan.interLayer};
        channelDescs.reserve(param.rankSize - 1);
        for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
            if (remoteRank == param.myRank) {
                continue;
            }
            HcclChannelDesc desc;
            const std::vector<uint32_t> &candidateLayers =
                std::binary_search(plan.intraRanks.begin(), plan.intraRanks.end(), remoteRank) ?
                intraLayers : interLayers;
            CHK_RET(FillChannelDesc(comm, param.myRank, remoteRank, candidateLayers, desc));
            channelDescs.push_back(desc);
        }
        resCtxHost.interPeerRank = plan.mirrorRank;
        resCtxHost.intraGroupStart = plan.intraRanks[0];
    } else {
        channelDescs.reserve(param.rankSize - 1);
        for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
            if (remoteRank == param.myRank) {
                continue;
            }
            HcclChannelDesc desc;
            CHK_RET(FillChannelDesc(comm, param.myRank, remoteRank, orderedLayers, desc));
            channelDescs.push_back(desc);
        }
    }

    const uint32_t channelNum = static_cast<uint32_t>(channelDescs.size());
    const uint32_t expectedChannelNum = param.rankSize - 1;
    CHK_PRT_RET(channelNum != expectedChannelNum,
        HCCL_ERROR("Unexpected Channel descriptor count %u, expected %u", channelNum, expectedChannelNum),
        HCCL_E_INTERNAL);
    std::vector<ChannelHandle> channelHandles(channelNum);

    CHK_RET(HcclChannelAcquire(
        comm, COMM_ENGINE_AICPU_TS, channelDescs.data(), channelNum, channelHandles.data()));

    std::vector<bool> seenRemoteRanks(param.rankSize, false);
    resCtxHost.channels.reserve(channelNum);
    for (uint32_t index = 0; index < channelNum; ++index) {
        const uint32_t remoteRank = channelDescs[index].remoteRank;
        CHK_PRT_RET(remoteRank >= param.rankSize || remoteRank == param.myRank || seenRemoteRanks[remoteRank],
            HCCL_ERROR("Invalid or duplicate remote rank %u", remoteRank), HCCL_E_INTERNAL);
        CHK_PRT_RET(channelHandles[index] == 0,
            HCCL_ERROR("Invalid Channel handle for remote rank %u", remoteRank), HCCL_E_INTERNAL);
        seenRemoteRanks[remoteRank] = true;

        void *remoteBufferAddr = nullptr;
        uint64_t remoteBufferSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, channelHandles[index], &remoteBufferAddr, &remoteBufferSize));
        CHK_PRT_RET(remoteBufferAddr == nullptr || remoteBufferSize == 0,
            HCCL_ERROR("Invalid remote HCCL buffer for rank %u", remoteRank), HCCL_E_INTERNAL);

        ChannelInfo channelInfo;
        channelInfo.remoteRank = remoteRank;
        channelInfo.notifyNum = CHANNEL_NOTIFY_NUM;
        channelInfo.handle = channelHandles[index];
        channelInfo.remoteCclMem = CommBuffer{remoteBufferAddr, remoteBufferSize};
        resCtxHost.channels.push_back(channelInfo);
    }
    return HCCL_SUCCESS;
}

/**
 * @brief 按 CPU_TS 到 AICPU_TS 的顺序销毁指定通信引擎上下文
 * @param comm 通信域句柄
 * @param tag 通信引擎上下文标签
 * @param destroyCpuCtx 是否销毁 CPU_TS 上下文
 * @param destroyAicpuCtx 是否销毁 AICPU_TS 上下文
 * @return 全部指定上下文销毁成功返回 HCCL_SUCCESS，否则返回首个错误码
 */
static HcclResult DestroyEngineContexts(
    HcclComm comm, const char *tag, bool destroyCpuCtx, bool destroyAicpuCtx)
{
    HcclResult firstRet = HCCL_SUCCESS;
    if (destroyCpuCtx) {
        HcclResult ret = HcclEngineCtxDestroy(comm, tag, CommEngine::COMM_ENGINE_CPU_TS);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("Failed to destroy CPU_TS engine context, ret %d", static_cast<int>(ret));
            firstRet = ret;
        }
    }
    if (destroyAicpuCtx) {
        HcclResult ret = HcclEngineCtxDestroy(comm, tag, CommEngine::COMM_ENGINE_AICPU_TS);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("Failed to destroy AICPU_TS engine context, ret %d", static_cast<int>(ret));
            if (firstRet == HCCL_SUCCESS) {
                firstRet = ret;
            }
        }
    }
    return firstRet;
}
} // 匿名命名空间

HcclResult HcclAllGather(
    void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("Only FP32 is supported, data type is %d", static_cast<int>(dataType)), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(sendCount > std::numeric_limits<uint64_t>::max() / sizeof(float),
        HCCL_ERROR("AllGather send count overflows byte size"), HCCL_E_PARA);

    // 构造算子参数
    OpParam param;
    sprintf(param.tag, "%s", "hccl_custom_allgather");
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = sendCount;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;

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
    CHK_PRT_RET(param.rankSize == 0 || param.myRank >= param.rankSize,
        HCCL_ERROR("Invalid rank information, rank %u of %u", param.myRank, param.rankSize), HCCL_E_PARA);

    const uint64_t dataBytes = sendCount * sizeof(float);
    CHK_PRT_RET(dataBytes != 0 && param.rankSize > std::numeric_limits<uint64_t>::max() / dataBytes,
        HCCL_ERROR("AllGather receive size overflows"), HCCL_E_PARA);
    if (sendCount == 0) {
        return HCCL_SUCCESS;
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
    HcclResult aicpuCtxRet = HcclEngineCtxGet(comm, param.tag, aicpuTsEngine, &ctx, &size);
    CHK_PRT_RET(aicpuCtxRet != HCCL_SUCCESS && aicpuCtxRet != HCCL_E_PARA &&
            aicpuCtxRet != HCCL_E_NOT_FOUND,
        HCCL_ERROR("Failed to query AICPU engine context, ret %d", static_cast<int>(aicpuCtxRet)), aicpuCtxRet);

    void *hostCtx = nullptr;
    uint64_t hostCtxSize = 0;
    HcclResult cpuCtxRet = HcclEngineCtxGet(comm, param.tag, cpuTsEngine, &hostCtx, &hostCtxSize);
    CHK_PRT_RET(cpuCtxRet != HCCL_SUCCESS && cpuCtxRet != HCCL_E_PARA && cpuCtxRet != HCCL_E_NOT_FOUND,
        HCCL_ERROR("Failed to query CPU engine context, ret %d", static_cast<int>(cpuCtxRet)), cpuCtxRet);

    const bool aicpuCtxExists = aicpuCtxRet == HCCL_SUCCESS;
    const bool cpuCtxExists = cpuCtxRet == HCCL_SUCCESS;
    bool reuseContexts = false;
    if (aicpuCtxExists && cpuCtxExists && ctx != nullptr && size > 0 && hostCtx != nullptr &&
        hostCtxSize >= sizeof(HostResourceCtxHeader)) {
        HostResourceCtxHeader header;
        std::memcpy(&header, hostCtx, sizeof(header));
        const bool snapshotSizeValid = header.headerSize == sizeof(HostResourceCtxHeader) &&
            header.serializedSize >= sizeof(ThreadHandle) && header.serializedSize == size &&
            header.serializedSize <= std::numeric_limits<uint64_t>::max() - header.headerSize &&
            hostCtxSize == header.headerSize + header.serializedSize;
        if (header.magic == HOST_RESOURCE_CTX_MAGIC && header.version == HOST_RESOURCE_CTX_VERSION &&
            snapshotSizeValid && header.aicpuThread != 0) {
            const char *serializedCtx = static_cast<const char *>(hostCtx) + header.headerSize;
            ThreadHandle serializedAicpuThread = 0;
            std::memcpy(&serializedAicpuThread, serializedCtx, sizeof(serializedAicpuThread));
            if (serializedAicpuThread == header.aicpuThread) {
                HcclResult exportRet = HcclThreadExportToCommEngine(
                    comm, 1, &header.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu);
                if (exportRet == HCCL_SUCCESS) {
                    HcclResult copyRet = HcclEngineCtxCopy(
                        comm, aicpuTsEngine, param.tag, serializedCtx, header.serializedSize, 0);
                    if (copyRet == HCCL_SUCCESS) {
                        param.resCtx = ctx;
                        param.ctxSize = size;
                        reuseContexts = true;
                        HCCL_INFO("Engine context pair already exists");
                    } else {
                        HCCL_ERROR("Failed to restore cached AICPU context, ret %d", static_cast<int>(copyRet));
                    }
                } else {
                    HCCL_ERROR("Failed to export cached AICPU thread, ret %d", static_cast<int>(exportRet));
                }
            } else {
                HCCL_ERROR("Cached AICPU resource snapshot contains inconsistent thread handles");
            }
        }
    }

    if (!reuseContexts && (aicpuCtxExists || cpuCtxExists)) {
        HCCL_WARNING("Engine context pair is incomplete or invalid, rebuild resources");
        HcclResult cleanupRet =
            DestroyEngineContexts(comm, param.tag, cpuCtxExists, aicpuCtxExists);
        CHK_PRT_RET(cleanupRet != HCCL_SUCCESS,
            HCCL_ERROR("Failed to clean incomplete Engine Context pair"), cleanupRet);
    }

    if (!reuseContexts) {
        // Device 资源不存在，资源构建
        AlgResourceCtx resCtxHost;

        // 从通信域获取 HCCL Buffer（Device上的内存，默认总大小400MB）
        void *cclBufferAddr = nullptr;
        uint64_t cclBufferSize = 0;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        CHK_PRT_RET(cclBufferAddr == nullptr || cclBufferSize == 0,
            HCCL_ERROR("Invalid local HCCL buffer"), HCCL_E_INTERNAL);
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        // ==============================================
        // STEP 2.2: 申请资源Thread和Channel
        // ==============================================

        CHK_RET(AcquirePeerChannels(comm, param, resCtxHost));

        // 分层拓扑为控制路径和十五条 peer 链路分别申请独立 thread
        const bool hierarchical = resCtxHost.interPeerRank != INVALID_VALUE_RANKID;
        const uint32_t threadNum = hierarchical ? HIERARCHICAL_THREAD_NUM : 1;
        const uint32_t notifyNumPerThread = hierarchical ? HIERARCHICAL_NOTIFY_NUM_PER_THREAD : 1;

        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, notifyNumPerThread, resCtxHost.threads.data()));
        CHK_PRT_RET(std::find(resCtxHost.threads.begin(), resCtxHost.threads.end(), 0) != resCtxHost.threads.end(),
            HCCL_ERROR("Invalid acquired AICPU communication thread"), HCCL_E_INTERNAL);
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        // ==============================================
        // STEP 2.3: 申请通信引擎上下文
        // ==============================================
        // 申请 AICPU 通信引擎上下文，存放 AlgResourceCtx 信息
        std::vector<char> seq = resCtxHost.Serialize();
        uint64_t seqSize = seq.size();
        CHK_PRT_RET(seqSize < sizeof(ThreadHandle) ||
                seqSize > std::numeric_limits<uint64_t>::max() - sizeof(HostResourceCtxHeader),
            HCCL_ERROR("Invalid serialized resource context size"), HCCL_E_INTERNAL);
        ThreadHandle serializedAicpuThread = 0;
        std::memcpy(&serializedAicpuThread, seq.data(), sizeof(serializedAicpuThread));
        CHK_PRT_RET(serializedAicpuThread != resCtxHost.aicpuThread,
            HCCL_ERROR("Serialized AICPU thread does not match acquired thread"), HCCL_E_INTERNAL);

        HostResourceCtxHeader header;
        header.headerSize = static_cast<uint32_t>(sizeof(HostResourceCtxHeader));
        header.serializedSize = seqSize;
        header.aicpuThread = resCtxHost.aicpuThread;
        std::vector<char> hostResourceSnapshot(sizeof(header) + seq.size());
        std::memcpy(hostResourceSnapshot.data(), &header, sizeof(header));
        std::memcpy(hostResourceSnapshot.data() + sizeof(header), seq.data(), seq.size());

        param.ctxSize = seqSize;
        HcclResult ret = HcclEngineCtxCreate(comm, param.tag, aicpuTsEngine, param.ctxSize, &param.resCtx);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
        if (param.resCtx == nullptr) {
            (void)DestroyEngineContexts(comm, param.tag, false, true);
            HCCL_ERROR("Created AICPU engine context is null");
            return HCCL_E_INTERNAL;
        }

        ret = HcclEngineCtxCopy(comm, aicpuTsEngine, param.tag, seq.data(), seqSize, 0);
        if (ret != HCCL_SUCCESS) {
            (void)DestroyEngineContexts(comm, param.tag, false, true);
            return ret;
        }

        // 申请 CPU 通信引擎上下文，存放 Host 可校验的完整资源快照
        hostCtx = nullptr;
        hostCtxSize = hostResourceSnapshot.size();
        ret = HcclEngineCtxCreate(comm, param.tag, cpuTsEngine, hostCtxSize, &hostCtx);
        if (ret != HCCL_SUCCESS) {
            (void)DestroyEngineContexts(comm, param.tag, false, true);
            return ret;
        }
        if (hostCtx == nullptr) {
            (void)DestroyEngineContexts(comm, param.tag, true, true);
            HCCL_ERROR("Created CPU engine context is null");
            return HCCL_E_INTERNAL;
        }

        ret = HcclEngineCtxCopy(comm, cpuTsEngine, param.tag, hostResourceSnapshot.data(), hostCtxSize, 0);
        if (ret != HCCL_SUCCESS) {
            (void)DestroyEngineContexts(comm, param.tag, true, true);
            return ret;
        }
    }

    // ==============================================
    // STEP 3: 下发 AICPU Kernel
    // ==============================================
    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}
