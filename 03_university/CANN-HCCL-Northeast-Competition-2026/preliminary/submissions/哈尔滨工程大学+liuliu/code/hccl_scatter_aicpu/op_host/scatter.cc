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
// Allocate one channel per peer. Query topology instead of assuming rank/8 is a server ID.
HcclResult AcquireScatterChannels(HcclComm comm, const OpParam &param, AlgResourceCtx &ctx)
{
    if (param.rankSize == 1) {
        return HCCL_SUCCESS;
    }
    uint32_t *layerIds = nullptr;
    uint32_t layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layerIds, &layerCount));
    if (layerIds == nullptr || layerCount == 0) {
        HCCL_ERROR("Scatter: topology has no network layers");
        return HCCL_E_PARA;
    }
    std::vector<uint32_t> layers(layerIds, layerIds + layerCount);
    std::sort(layers.begin(), layers.end());
    std::vector<HcclChannelDesc> descs;
    std::vector<uint32_t> localRanks{param.myRank};
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer == param.myRank) {
            continue;
        }
        bool found = false;
        HcclChannelDesc desc;
        HcclChannelDescInit(&desc, 1);
        for (uint32_t layer : layers) {
            CommLink *links = nullptr;
            uint32_t linkCount = 0;
            CHK_RET(HcclRankGraphGetLinks(comm, layer, param.myRank, peer, &links, &linkCount));
            for (uint32_t i = 0; links != nullptr && i < linkCount; ++i) {
                const auto protocol = links[i].linkAttr.linkProtocol;
                // Preserve the protocol reported by the competition topology.
                // ChannelAcquire validates it against the required AICPU_TS engine.
                desc.remoteRank = peer;
                desc.localEndpoint.protocol = links[i].srcEndpointDesc.protocol;
                desc.localEndpoint.commAddr = links[i].srcEndpointDesc.commAddr;
                desc.localEndpoint.loc = links[i].srcEndpointDesc.loc;
                desc.remoteEndpoint.protocol = links[i].dstEndpointDesc.protocol;
                desc.remoteEndpoint.commAddr = links[i].dstEndpointDesc.commAddr;
                desc.remoteEndpoint.loc = links[i].dstEndpointDesc.loc;
                desc.channelProtocol = protocol;
                desc.notifyNum = SCATTER_CHANNEL_NOTIFY_NUM;
                if (layer == 0) {
                    localRanks.push_back(peer);
                }
                found = true;
                break;
            }
            if (found) {
                break;
            }
        }
        if (!found) {
            HCCL_ERROR("Scatter: topology returned no link from rank %u to rank %u across %u layers",
                param.myRank, peer, layerCount);
            return HCCL_E_PARA;
        }
        descs.push_back(desc);
    }
    // Layer 0 is the server-local Full Mesh in the competition topology.
    // Canonicalize the two groups without assuming contiguous rank numbering.
    constexpr uint32_t relayRankSize = 16;
    constexpr uint32_t ranksPerServer = 8;
    ctx.serverGroup.clear();
    if (param.rankSize == relayRankSize && layers.front() == 0 &&
        localRanks.size() == ranksPerServer) {
        std::sort(localRanks.begin(), localRanks.end());
        const bool localIsGroupZero = std::binary_search(localRanks.begin(), localRanks.end(), 0U);
        ctx.serverGroup.assign(param.rankSize, localIsGroupZero ? 1U : 0U);
        for (uint32_t rank : localRanks) {
            ctx.serverGroup[rank] = localIsGroupZero ? 0U : 1U;
        }
    }
    const uint32_t channelCount = static_cast<uint32_t>(descs.size());
    std::vector<ChannelHandle> handles(channelCount);
    const HcclResult acquireRet = HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AICPU_TS,
        descs.data(), channelCount, handles.data());
    if (acquireRet != HCCL_SUCCESS) {
        for (const HcclChannelDesc &request : descs) {
            HCCL_ERROR("Scatter: channel request rank=%u peer=%u protocol=%d, acquire ret=%d",
                param.myRank, request.remoteRank, static_cast<int>(request.channelProtocol),
                static_cast<int>(acquireRet));
        }
        return acquireRet;
    }
    ctx.channels.resize(channelCount);
    for (uint32_t i = 0; i < channelCount; ++i) {
        ChannelInfo &info = ctx.channels[i];
        info.remoteRank = descs[i].remoteRank;
        info.notifyNum = SCATTER_CHANNEL_NOTIFY_NUM;
        info.handle = handles[i];
        CHK_RET(HcclChannelGetHcclBuffer(comm, info.handle, &info.remoteCclMem.addr, &info.remoteCclMem.size));
        if (info.remoteCclMem.addr == nullptr || info.remoteCclMem.size < sizeof(float)) {
            HCCL_ERROR("Scatter: invalid peer communication buffer, rank %u", info.remoteRank);
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

    // 构造算子参数
    OpParam param{};
    sprintf(param.tag, "%s", "hccl_scatter_staged_v7");
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
    if (param.rankSize == 0 || param.myRank >= param.rankSize || root >= param.rankSize ||
        dataType != HCCL_DATA_TYPE_FP32) {
        HCCL_ERROR("Scatter: invalid rank, root or data type");
        return HCCL_E_PARA;
    }
    const uint64_t maxBytes = std::numeric_limits<uint64_t>::max();
    if (recvCount > maxBytes / sizeof(float) / param.rankSize) {
        HCCL_ERROR("Scatter: input size overflows uint64_t");
        return HCCL_E_PARA;
    }
    if (recvCount != 0) {
        CHK_PTR_NULL(recvBuf);
        if (param.myRank == root) {
            CHK_PTR_NULL(sendBuf);
        }
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
        AlgResourceCtx resCtxHost{};

        // 从通信域获取 HCCL Buffer（Device上的内存，默认总大小400MB）
        void *cclBufferAddr;
        uint64_t cclBufferSize;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        if (cclBufferAddr == nullptr || cclBufferSize < sizeof(float)) {
            return HCCL_E_PARA;
        }
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        // ==============================================
        // STEP 2.2: 申请资源Thread和Channel
        // ==============================================

        // Main: Host notify 0 and one completion notify per producer.
        // For the 16-rank topology, each producer has an independent network thread.
        // 31 AICPU_TS threads use 16 + 15 + 15 = 46 thread notifies in total.
        const uint32_t extraThreads = param.rankSize == 16 ? param.rankSize - 1 : 0;
        resCtxHost.threads.resize(param.rankSize + extraThreads);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, 1, param.rankSize, resCtxHost.threads.data()));
        if (param.rankSize > 1) {
            CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, param.rankSize - 1 + extraThreads, 1,
                resCtxHost.threads.data() + 1));
        }
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        // All peer resources are cached independently of root and message size.
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
