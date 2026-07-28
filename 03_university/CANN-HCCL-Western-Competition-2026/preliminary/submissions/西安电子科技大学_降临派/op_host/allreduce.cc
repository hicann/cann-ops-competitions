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

#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

namespace {
constexpr uint32_t EXPECTED_RANK_SIZE = 16;
constexpr uint32_t RANKS_PER_SERVER = 8;
constexpr uint32_t CHANNEL_NOTIFY_NUM = 8;
constexpr uint32_t PARALLEL_THREAD_NUM = 24;
constexpr uint32_t PARALLEL_THREAD_NOTIFY_NUM = 26;
constexpr uint32_t CHANNEL_TYPE_NUM = 2;
constexpr uint32_t PORT_INDEX = 5;
constexpr uint8_t AGGREGATE_PORT = 127;

HcclResult FillChannelDesc(uint32_t dstRank, const CommLink &link, HcclChannelDesc &desc)
{
    CHK_RET(HcclChannelDescInit(&desc, 1));
    desc.remoteRank = dstRank;
    desc.notifyNum = CHANNEL_NOTIFY_NUM;
    desc.channelProtocol = link.linkAttr.linkProtocol;
    desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
    desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
    desc.localEndpoint.loc = link.srcEndpointDesc.loc;
    desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
    desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
    desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
    return HCCL_SUCCESS;
}

bool IsAggregateLink(const CommLink &link)
{
    return link.srcEndpointDesc.commAddr.type == COMM_ADDR_TYPE_EID
           && link.dstEndpointDesc.commAddr.type == COMM_ADDR_TYPE_EID
           && link.srcEndpointDesc.commAddr.eid[PORT_INDEX] == AGGREGATE_PORT
           && link.dstEndpointDesc.commAddr.eid[PORT_INDEX] == AGGREGATE_PORT;
}

HcclResult FindChannelDesc(HcclComm comm, uint32_t srcRank, uint32_t dstRank, ChannelType type, HcclChannelDesc &desc)
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));

    constexpr CommProtocol expectedProtocols[] = {CommProtocol::COMM_PROTOCOL_UBC_CTP,
        CommProtocol::COMM_PROTOCOL_UBC_TP, CommProtocol::COMM_PROTOCOL_PCIE, CommProtocol::COMM_PROTOCOL_UBOE};
    const CommLink *fallback = nullptr;
    for (uint32_t layerIdx = 0; layerIdx < netLayerNum; layerIdx++) {
        uint32_t netLayer = netLayers[layerIdx];
        if ((type == ChannelType::MESH && netLayer != 0)
            || (type == ChannelType::NHR && netLayerNum > 1 && netLayer == 0)) {
            continue;
        }

        CommLink *linkList = nullptr;
        uint32_t listSize = 0;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayer, srcRank, dstRank, &linkList, &listSize));
        for (CommProtocol protocol : expectedProtocols) {
            for (uint32_t linkIdx = 0; linkIdx < listSize; linkIdx++) {
                const CommLink &link = linkList[linkIdx];
                if (link.linkAttr.linkProtocol != protocol) {
                    continue;
                }

                if (fallback == nullptr) {
                    fallback = &link;
                }
                if (type == ChannelType::MESH || IsAggregateLink(link)) {
                    CHK_RET(FillChannelDesc(dstRank, link, desc));
                    return HCCL_SUCCESS;
                }
            }
        }
        if (fallback != nullptr) {
            CHK_RET(FillChannelDesc(dstRank, *fallback, desc));
            return HCCL_SUCCESS;
        }
    }

    HCCL_ERROR(
        "No AICPU_TS channel type[%u] between rank[%u] and rank[%u]", static_cast<uint32_t>(type), srcRank, dstRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult AcquireChannels(HcclComm comm, const OpParam &param, bool acquireMesh, AlgResourceCtx &resCtxHost)
{
    if (param.rankSize <= 1) {
        return HCCL_SUCCESS;
    }

    std::vector<uint32_t> meshRanks;
    if (acquireMesh) {
        uint32_t serverBase = param.myRank / RANKS_PER_SERVER * RANKS_PER_SERVER;
        for (uint32_t localRank = 0; localRank < RANKS_PER_SERVER; localRank++) {
            uint32_t remoteRank = serverBase + localRank;
            if (remoteRank != param.myRank) {
                meshRanks.push_back(remoteRank);
            }
        }
    }

    std::vector<uint32_t> nhrRanks;
    uint32_t remoteServerBase = (param.myRank / RANKS_PER_SERVER ^ 1U) * RANKS_PER_SERVER;
    for (uint32_t localRank = 0; localRank < RANKS_PER_SERVER; localRank++) {
        nhrRanks.push_back(remoteServerBase + localRank);
    }

    std::vector<HcclChannelDesc> descs;
    std::vector<ChannelType> types;
    descs.resize(meshRanks.size() + nhrRanks.size());
    types.reserve(descs.size());
    uint32_t descIdx = 0;
    for (uint32_t remoteRank : meshRanks) {
        CHK_RET(FindChannelDesc(comm, param.myRank, remoteRank, ChannelType::MESH, descs[descIdx++]));
        types.push_back(ChannelType::MESH);
    }
    for (uint32_t remoteRank : nhrRanks) {
        CHK_RET(FindChannelDesc(comm, param.myRank, remoteRank, ChannelType::NHR, descs[descIdx++]));
        types.push_back(ChannelType::NHR);
    }

    std::vector<ChannelHandle> handles(descs.size());
    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AICPU_TS, descs.data(), descs.size(), handles.data()));

    resCtxHost.channels.reserve(descs.size());
    resCtxHost.channelIndices.assign(CHANNEL_TYPE_NUM * param.rankSize, INVALID_VALUE_RANKID);
    for (uint32_t idx = 0; idx < descs.size(); idx++) {
        void *remoteCclBuffer = nullptr;
        uint64_t remoteCclBufferSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, handles[idx], &remoteCclBuffer, &remoteCclBufferSize));
        resCtxHost.channels.push_back(ChannelInfo{descs[idx].remoteRank, types[idx], CHANNEL_NOTIFY_NUM, handles[idx],
            {remoteCclBuffer, remoteCclBufferSize}});
        uint32_t typeIdx = static_cast<uint32_t>(types[idx]);
        resCtxHost.channelIndices[typeIdx * param.rankSize + descs[idx].remoteRank] = idx;
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
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32 || op != HCCL_REDUCE_SUM,
        HCCL_ERROR("Only FP32 SUM is supported, dataType[%d], reduceOp[%d]", dataType, op), HCCL_E_NOT_SUPPORT);

    // 构造算子参数
    OpParam param;
    const char *algTag = "hccl_custom_allreduce_final_best_points_v1";
    int tagRet = sprintf_s(param.tag, sizeof(param.tag), "%s", algTag);
    CHK_PRT_RET(tagRet <= 0, HCCL_ERROR("Failed to fill allreduce tag"), HCCL_E_INTERNAL);
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = count;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;

    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));

    // ==============================================
    // STEP 1: 解析拓扑信息
    // ==============================================
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize != 1 && param.rankSize != EXPECTED_RANK_SIZE,
        HCCL_ERROR("Only rankSize 1 or 16 is supported, rankSize[%u]", param.rankSize), HCCL_E_NOT_SUPPORT);

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

        // 统一资源支持同通信域混合调用。
        resCtxHost.threads.resize(PARALLEL_THREAD_NUM);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, PARALLEL_THREAD_NUM, PARALLEL_THREAD_NOTIFY_NUM,
            resCtxHost.threads.data()));
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        CHK_RET(AcquireChannels(comm, param, true, resCtxHost));

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

    HcclDfxOpInfo dfxInfo{};
    dfxInfo.beginTime = HcommGetProfilingSysCycleTime();
    dfxInfo.opType = static_cast<uint32_t>(param.opType);
    dfxInfo.reduceOp = static_cast<uint32_t>(param.reduceType);
    dfxInfo.dataType = static_cast<uint32_t>(param.dataType);
    dfxInfo.dataCount = param.count;
    dfxInfo.engine = aicpuTsEngine;
    dfxInfo.cpuTsThread = param.cpuThread;
    dfxInfo.cpuWaitAicpuNotifyIdx = 0;
    dfxInfo.inputMemAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    dfxInfo.inputMemSize = param.count * sizeof(float);
    dfxInfo.outputMemAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    dfxInfo.outputMemSize = param.count * sizeof(float);
    int algTagRet = sprintf_s(dfxInfo.algTag, sizeof(dfxInfo.algTag), "%s", param.tag);
    CHK_PRT_RET(algTagRet <= 0, HCCL_ERROR("Failed to fill DFX tag"), HCCL_E_INTERNAL);
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    // ==============================================
    // STEP 3: 下发 AICPU Kernel
    // ==============================================
    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}