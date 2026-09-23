/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <limits>
#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

namespace {
bool SupportsAicpuTs(CommProtocol protocol)
{
    return protocol == COMM_PROTOCOL_UBC_CTP || protocol == COMM_PROTOCOL_UBOE;
}

HcclResult AcquireScatterChannels(HcclComm comm, const OpParam &param, AlgResourceCtx &resource)
{
    if (param.rankSize == 1) {
        return HCCL_SUCCESS;
    }
    uint32_t *layerList = nullptr;
    uint32_t layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layerList, &layerCount));
    CHK_PRT_RET(layerList == nullptr || layerCount == 0, HCCL_ERROR("No topology layers"), HCCL_E_INTERNAL);
    // Queries may invalidate earlier library-owned lists, so retain a copy.
    std::vector<uint32_t> layers(layerList, layerList + layerCount);
    std::sort(layers.begin(), layers.end());

    if (param.rankSize == 16) {
        uint32_t *instanceSizes = nullptr;
        uint32_t instanceCount = 0;
        CHK_RET(HcclRankGraphGetInstSizeListByLayer(comm, layers.front(), &instanceSizes, &instanceCount));
        // The global shape makes every rank select the same relay/fallback path.
        const bool pairedServers = instanceSizes != nullptr && instanceCount == 2
            && instanceSizes[0] == 8 && instanceSizes[1] == 8;
        if (pairedServers) {
            uint32_t *members = nullptr;
            uint32_t memberCount = 0;
            CHK_RET(HcclRankGraphGetRanksByLayer(comm, layers.front(), &members, &memberCount));
            CHK_PRT_RET(members == nullptr || memberCount != 8,
                HCCL_ERROR("Missing paired server membership"), HCCL_E_INTERNAL);
            resource.serverRanks.assign(members, members + memberCount);
            std::sort(resource.serverRanks.begin(), resource.serverRanks.end());
            CHK_PRT_RET(resource.serverRanks.back() >= param.rankSize
                    || std::adjacent_find(resource.serverRanks.begin(), resource.serverRanks.end()) != resource.serverRanks.end()
                    || !std::binary_search(resource.serverRanks.begin(), resource.serverRanks.end(), param.myRank),
                HCCL_ERROR("Invalid server membership"), HCCL_E_INTERNAL);
        }
    }

    const uint32_t channelCount = param.rankSize - 1;
    std::vector<HcclChannelDesc> descriptions(channelCount);
    std::vector<ChannelHandle> handles(channelCount);
    CHK_RET(HcclChannelDescInit(descriptions.data(), channelCount));
    uint32_t index = 0;
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer == param.myRank) {
            continue;
        }
        auto &description = descriptions[index++];
        bool found = false;
        for (uint32_t layer : layers) {
            CommLink *links = nullptr;
            uint32_t linkCount = 0;
            CHK_RET(HcclRankGraphGetLinks(comm, layer, param.myRank, peer, &links, &linkCount));
            for (uint32_t linkIndex = 0; links != nullptr && linkIndex < linkCount; ++linkIndex) {
                const CommLink &link = links[linkIndex];
                if (!SupportsAicpuTs(link.linkAttr.linkProtocol)) {
                    continue;
                }
                description.remoteRank = peer;
                description.notifyNum = SCATTER_CHANNEL_NOTIFY_NUM;
                description.channelProtocol = link.linkAttr.linkProtocol;
                description.localEndpoint.protocol = link.srcEndpointDesc.protocol;
                description.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
                description.localEndpoint.loc = link.srcEndpointDesc.loc;
                description.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
                description.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
                description.remoteEndpoint.loc = link.dstEndpointDesc.loc;
                found = true;
                break;
            }
            if (found) {
                break;
            }
        }
        CHK_PRT_RET(!found, HCCL_ERROR("No AICPU_TS link from rank %u to %u", param.myRank, peer), HCCL_E_NOT_SUPPORT);
    }
    // One channel per peer; the complete peer set remains valid when root changes.
    CHK_RET(HcclChannelAcquire(comm, COMM_ENGINE_AICPU_TS, descriptions.data(), channelCount, handles.data()));
    resource.channels.resize(channelCount);
    for (uint32_t i = 0; i < channelCount; ++i) {
        auto &channel = resource.channels[i];
        channel.remoteRank = descriptions[i].remoteRank;
        channel.notifyNum = SCATTER_CHANNEL_NOTIFY_NUM;
        channel.handle = handles[i];
        CHK_RET(HcclChannelGetHcclBuffer(comm, channel.handle, &channel.remoteCclMem.addr, &channel.remoteCclMem.size));
        CHK_PTR_NULL(channel.remoteCclMem.addr);
        CHK_PRT_RET(channel.remoteCclMem.size < sizeof(float), HCCL_ERROR("Remote buffer is too small"), HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    // 构造算子参数
    OpParam param;
    snprintf(param.tag, sizeof(param.tag), "%s", "hccl_custom_scatter_tree_v5");
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.root = root;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_SCATTER;

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
    CHK_PRT_RET(param.rankSize == 0 || param.myRank >= param.rankSize || root >= param.rankSize,
        HCCL_ERROR("Invalid Scatter rank or root"), HCCL_E_PARA);
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32, HCCL_ERROR("Scatter supports float32"), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(recvCount > std::numeric_limits<uint64_t>::max() / sizeof(float) / param.rankSize,
        HCCL_ERROR("Scatter input size overflows"), HCCL_E_PARA);
    if (recvCount == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(recvBuf);
    if (param.myRank == root) {
        CHK_PTR_NULL(sendBuf);
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
        if (param.rankSize > 1) {
            CHK_PTR_NULL(cclBufferAddr);
            CHK_PRT_RET(ScatterSlotBytes(cclBufferSize, param.rankSize) == 0,
                HCCL_ERROR("Local buffer cannot hold one element per peer"), HCCL_E_INTERNAL);
        }

        // ==============================================
        // STEP 2.2: 申请资源Thread和Channel
        // ==============================================

        // One control thread plus one sender per peer. Notify 0 on the control
        // thread is reserved for Host/Device synchronization; 1..N-1 join workers.
        uint32_t threadNum = param.rankSize;
        uint32_t notifyNumPerThread = param.rankSize;

        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, notifyNumPerThread, resCtxHost.threads.data()));
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        CHK_RET(AcquireScatterChannels(comm, param, resCtxHost));

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
