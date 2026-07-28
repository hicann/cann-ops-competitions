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

#include <cstddef>
#include <limits>
#include <vector>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

namespace {

constexpr uint32_t LOCAL_RANK_SIZE = 8;
constexpr uint32_t EXPECTED_RANK_SIZE = 2 * LOCAL_RANK_SIZE;
constexpr uint32_t CHANNEL_NOTIFY_NUM = 2;
constexpr uint32_t LAYER_0 = 0;
constexpr uint32_t LAYER_1 = 1;

HcclResult CheckNetLayers(HcclComm comm)
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));

    bool hasLayer0 = false;
    bool hasLayer1 = false;
    for (uint32_t i = 0; i < netLayerNum; ++i) {
        hasLayer0 = hasLayer0 || netLayers[i] == LAYER_0;
        hasLayer1 = hasLayer1 || netLayers[i] == LAYER_1;
    }
    CHK_PRT_RET(!hasLayer0 || !hasLayer1,
        HCCL_ERROR("Broadcast requires net layer 0 and layer 1, layer num[%u]", netLayerNum), HCCL_E_NOT_SUPPORT);
    return HCCL_SUCCESS;
}

HcclResult FillChannelDesc(
    HcclComm comm, uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, HcclChannelDesc &desc)
{
    CommLink *linkList = nullptr;
    uint32_t linkNum = 0;
    CHK_RET(HcclRankGraphGetLinks(comm, netLayer, srcRank, dstRank, &linkList, &linkNum));
    CHK_RET(HcclChannelDescInit(&desc, 1));

    constexpr CommProtocol protocol = CommProtocol::COMM_PROTOCOL_UBC_CTP;
    for (uint32_t i = 0; i < linkNum; ++i) {
        const CommLink &link = linkList[i];
        if (link.linkAttr.linkProtocol != protocol) {
            continue;
        }
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

    HCCL_ERROR("UBC_CTP link not found, layer[%u], srcRank[%u], dstRank[%u]", netLayer, srcRank, dstRank);
    return HCCL_E_NOT_FOUND;
}

} // namespace

HcclResult HcclBroadcast(
    void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(buf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    // 构造算子参数
    OpParam param;
    int tagRet = std::snprintf(param.tag, sizeof(param.tag), "%s", "hccl_custom_broadcast");
    CHK_PRT_RET(tagRet < 0 || static_cast<std::size_t>(tagRet) >= sizeof(param.tag),
        HCCL_ERROR("Failed to fill Broadcast tag"), HCCL_E_INTERNAL);
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
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("Broadcast only supports float32, dataType[%d]", dataType), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(root >= param.rankSize,
        HCCL_ERROR("Broadcast root[%u] is out of rankSize[%u]", root, param.rankSize), HCCL_E_PARA);
    CHK_PRT_RET(count > std::numeric_limits<uint64_t>::max() / sizeof(float),
        HCCL_ERROR("Broadcast count[%llu] overflows data size", static_cast<unsigned long long>(count)), HCCL_E_PARA);
    if (count == 0 || param.rankSize == 1) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(param.rankSize != EXPECTED_RANK_SIZE,
        HCCL_ERROR("Broadcast expects rankSize[%u], actual rankSize[%u]", EXPECTED_RANK_SIZE, param.rankSize),
        HCCL_E_NOT_SUPPORT);
    CHK_RET(CheckNetLayers(comm));

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
        resCtxHost.localIndex = param.myRank % LOCAL_RANK_SIZE;
        resCtxHost.localRankSize = LOCAL_RANK_SIZE;

        // ==============================================
        // STEP 2.2: 申请资源Thread和Channel
        // ==============================================

        // 创建 AICPU_TS 通信引擎上的 thread 资源
        // root需要同时使用7条Layer-0链路和8条Layer-1链路，因此每个远端rank分配一个thread。
        uint32_t threadNum = param.rankSize - 1;
        // thread[0]的notify[0]用于Host/AICPU同步，其余notify用于从thread回主thread的同步。
        uint32_t notifyNumPerThread = threadNum;

        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, notifyNumPerThread, resCtxHost.threads.data()));
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        // 为任意root预申请到其余15个rank的Channel。Server内使用Layer-0，Server间使用Layer-1。
        uint32_t channelNum = param.rankSize - 1;
        std::vector<HcclChannelDesc> channelDescs(channelNum);
        std::vector<ChannelHandle> channelHandles(channelNum);
        std::vector<uint32_t> remoteRanks(channelNum);
        std::vector<uint32_t> channelLayers(channelNum);
        uint32_t channelIdx = 0;
        uint32_t localServer = param.myRank / LOCAL_RANK_SIZE;
        for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
            if (remoteRank == param.myRank) {
                continue;
            }
            uint32_t remoteServer = remoteRank / LOCAL_RANK_SIZE;
            uint32_t netLayer = remoteServer == localServer ? LAYER_0 : LAYER_1;
            CHK_RET(FillChannelDesc(comm, netLayer, param.myRank, remoteRank, channelDescs[channelIdx]));
            remoteRanks[channelIdx] = remoteRank;
            channelLayers[channelIdx] = netLayer;
            ++channelIdx;
        }
        CHK_RET(HcclChannelAcquire(
            comm, aicpuTsEngine, channelDescs.data(), channelNum, channelHandles.data()));

        resCtxHost.channels.reserve(channelNum);
        for (uint32_t i = 0; i < channelNum; ++i) {
            void *remoteCclBuffer = nullptr;
            uint64_t remoteCclBufferSize = 0;
            CHK_RET(HcclChannelGetHcclBuffer(
                comm, channelHandles[i], &remoteCclBuffer, &remoteCclBufferSize));

            ChannelInfo channel;
            channel.remoteRank = remoteRanks[i];
            channel.remoteLocalIndex = remoteRanks[i] % LOCAL_RANK_SIZE;
            channel.netLayer = channelLayers[i];
            channel.notifyNum = CHANNEL_NOTIFY_NUM;
            channel.handle = channelHandles[i];
            channel.remoteCclMem = CommBuffer{remoteCclBuffer, remoteCclBufferSize};
            resCtxHost.channels.push_back(channel);
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
