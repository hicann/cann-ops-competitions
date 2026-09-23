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
#include <limits>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

namespace {
HcclResult CreateScatterChannels(HcclComm comm, const OpParam &param, AlgResourceCtx &ctx)
{
    if (param.rankSize == 1) {
        return HCCL_SUCCESS;
    }
    uint32_t *layerList = nullptr;
    uint32_t layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layerList, &layerCount));
    if (layerList == nullptr || layerCount == 0) {
        return HCCL_E_NOT_FOUND;
    }
    std::vector<uint32_t> layers(layerList, layerList + layerCount);
    std::sort(layers.begin(), layers.end());
    if (param.rankSize == 16) {
        uint32_t *ranks = nullptr;
        uint32_t count = 0;
        CHK_RET(HcclRankGraphGetRanksByLayer(comm, layers.front(), &ranks, &count));
        if (ranks != nullptr && count == 8) {
            std::vector<uint32_t> group(ranks, ranks + count);
            std::sort(group.begin(), group.end());
            if (group.back() < param.rankSize && std::adjacent_find(group.begin(), group.end()) == group.end()
                && std::binary_search(group.begin(), group.end(), param.myRank)) {
                ctx.localRanks = group;
            }
        }
    }
    std::vector<HcclChannelDesc> descs(param.rankSize - 1);
    std::vector<ChannelHandle> handles(param.rankSize - 1);
    ctx.channels.resize(param.rankSize - 1);
    uint32_t index = 0;
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer == param.myRank) {
            continue;
        }
        auto &desc = descs[index];
        CHK_RET(HcclChannelDescInit(&desc, 1));
        bool found = false;
        for (uint32_t layer : layers) {
            uint32_t *rankList = nullptr;
            uint32_t rankCount = 0;
            CHK_RET(HcclRankGraphGetRanksByLayer(comm, layer, &rankList, &rankCount));
            if (rankList == nullptr || std::find(rankList, rankList + rankCount, peer) == rankList + rankCount) {
                continue;
            }
            CommLink *links = nullptr;
            uint32_t linkCount = 0;
            CHK_RET(HcclRankGraphGetLinks(comm, layer, param.myRank, peer, &links, &linkCount));
            for (uint32_t i = 0; links != nullptr && i < linkCount; ++i) {
                if (links[i].linkAttr.linkProtocol != COMM_PROTOCOL_UBC_CTP) {
                    continue;
                }
                desc.remoteRank = peer;
                desc.notifyNum = SCATTER_CHANNEL_NOTIFY_NUM;
                desc.channelProtocol = links[i].linkAttr.linkProtocol;
                desc.localEndpoint = links[i].srcEndpointDesc;
                desc.remoteEndpoint = links[i].dstEndpointDesc;
                found = true;
                break;
            }
            if (found) {
                break;
            }
        }
        if (!found) {
            HCCL_ERROR("Scatter requires an AICPU_TS UBC_CTP link from rank %u to %u", param.myRank, peer);
            return HCCL_E_NOT_SUPPORT;
        }
        ctx.channels[index].remoteRank = peer;
        ctx.channels[index].notifyNum = SCATTER_CHANNEL_NOTIFY_NUM;
        ++index;
    }
    // A root-independent full mesh allows later calls to change root without creating duplicate channels.
    CHK_RET(HcclChannelAcquire(comm, COMM_ENGINE_AICPU_TS, descs.data(), descs.size(), handles.data()));
    for (uint32_t i = 0; i < handles.size(); ++i) {
        auto &channel = ctx.channels[i];
        channel.handle = handles[i];
        CHK_RET(HcclChannelGetHcclBuffer(comm, handles[i], &channel.remoteCclMem.addr, &channel.remoteCclMem.size));
        if (channel.remoteCclMem.addr == nullptr || ScatterSlotSize(channel.remoteCclMem.size, param.rankSize) == 0) {
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    if (SIZE_TABLE.find(dataType) == SIZE_TABLE.end()) {
        return HCCL_E_PARA;
    }

    // 构造算子参数
    OpParam param{};
    // Keep cached resources separate from older implementations.
    sprintf(param.tag, "%s", "hccl_custom_scatter_tree_v13");
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.root = root;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_SCATTER;

    // ==============================================
    // STEP 1: 解析拓扑信息
    // ==============================================
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    if (param.rankSize == 0 || root >= param.rankSize || param.myRank >= param.rankSize
        || recvCount > std::numeric_limits<uint64_t>::max() / SIZE_TABLE.at(dataType) / param.rankSize) {
        return HCCL_E_PARA;
    }
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

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
    CHK_RET(HcclGetCommName(comm, commName));
    dfxInfo.opType = static_cast<uint32_t>(param.opType);
    dfxInfo.dataType = static_cast<uint32_t>(dataType);
    dfxInfo.dataCount = recvCount;
    dfxInfo.root = root;
    dfxInfo.engine = aicpuTsEngine;
    dfxInfo.cpuTsThread = param.cpuThread;
    dfxInfo.cpuWaitAicpuNotifyIdx = 0;
    dfxInfo.inputMemAddr = reinterpret_cast<uint64_t>(sendBuf);
    dfxInfo.outputMemAddr = reinterpret_cast<uint64_t>(recvBuf);
    dfxInfo.outputMemSize = recvCount * SIZE_TABLE.at(dataType);
    dfxInfo.inputMemSize = param.myRank == root ? dfxInfo.outputMemSize * param.rankSize : 0;
    std::snprintf(dfxInfo.algTag, sizeof(dfxInfo.algTag), "%s", param.tag);
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, &dfxInfo));

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
        if (hostCtx == nullptr || hostCtxSize < sizeof(ThreadHandle)) {
            return HCCL_E_INTERNAL;
        }
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
        if (param.rankSize > 1 && (cclBufferAddr == nullptr || ScatterSlotSize(cclBufferSize, param.rankSize) == 0)) {
            return HCCL_E_PARA;
        }

        // ==============================================
        // STEP 2.2: 申请资源Thread和Channel
        // ==============================================

        // Thread 0 is reserved for the launcher; each peer has one transfer thread.
        uint32_t threadNum = param.rankSize;
        uint32_t notifyNumPerThread = param.rankSize;

        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, notifyNumPerThread, resCtxHost.threads.data()));
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        CHK_RET(CreateScatterChannels(comm, param, resCtxHost));

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
