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

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 4;
constexpr uint32_t CONTEST_RANK_SIZE = 16;
constexpr uint32_t SERVER_SIZE = 8;
constexpr uint32_t CONTEST_THREAD_NUM = 15;
constexpr uint32_t CONTEST_THREAD_NOTIFY_NUM = 15;

HcclResult AcquireDesc(HcclComm comm, uint32_t srcRank, uint32_t dstRank, HcclChannelDesc *desc)
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    CHK_PRT_RET(netLayers == nullptr || netLayerNum == 0,
        HCCL_ERROR("Communication topology contains no network layer"), HCCL_E_INTERNAL);
    std::vector<uint32_t> layers(netLayers, netLayers + netLayerNum);

    // Prefer the direct layer-0 link inside a server and the layer-1 Clos
    // link across servers. Keep the remaining layers as a compatibility
    // fallback in case the topology exposes a different layer set.
    uint32_t preferredLayer = srcRank / SERVER_SIZE == dstRank / SERVER_SIZE ? 0 : 1;
    auto preferred = std::find(layers.begin(), layers.end(), preferredLayer);
    if (preferred != layers.end()) {
        std::iter_swap(layers.begin(), preferred);
    }

    for (uint32_t i = 0; i < netLayerNum; ++i) {
        CommLink *linkList = nullptr;
        uint32_t listSize = 0;
        HcclResult ret = HcclRankGraphGetLinks(comm, layers[i], srcRank, dstRank, &linkList, &listSize);
        if (ret != HCCL_SUCCESS || listSize == 0) {
            continue;
        }

        CHK_RET(HcclChannelDescInit(desc, 1));
        const CommLink &link = linkList[0];
        desc->remoteRank = dstRank;
        desc->notifyNum = CHANNEL_NOTIFY_NUM;
        desc->channelProtocol = link.linkAttr.linkProtocol;
        desc->localEndpoint.protocol = link.srcEndpointDesc.protocol;
        desc->localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
        desc->localEndpoint.loc = link.srcEndpointDesc.loc;
        desc->remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
        desc->remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
        desc->remoteEndpoint.loc = link.dstEndpointDesc.loc;
        return HCCL_SUCCESS;
    }

    HCCL_ERROR("No link between rank[%u] and rank[%u]", srcRank, dstRank);
    return HCCL_E_INTERNAL;
}

void AddPeer(std::vector<uint32_t> &peers, uint32_t peer)
{
    if (std::find(peers.begin(), peers.end(), peer) == peers.end()) {
        peers.push_back(peer);
    }
}

HcclResult AcquireChannels(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtxHost)
{
    std::vector<uint32_t> peers;

    // Recursive-doubling partners. This path is used by small messages and as
    // the fallback for power-of-two topologies other than the contest's 2x8.
    for (uint32_t distance = 1; distance < param.rankSize; distance <<= 1U) {
        AddPeer(peers, param.myRank ^ distance);
    }

    // The performance path uses all seven layer-0 Full-Mesh links in the
    // local server and all eight layer-1 Clos links to the other server.
    // Add the complete set even when the first invocation is small because
    // EngineCtx is cached for the lifetime of the communicator.
    if (param.rankSize == CONTEST_RANK_SIZE) {
        uint32_t serverBase = (param.myRank / SERVER_SIZE) * SERVER_SIZE;
        for (uint32_t localPeer = 0; localPeer < SERVER_SIZE; ++localPeer) {
            uint32_t peer = serverBase + localPeer;
            if (peer != param.myRank) {
                AddPeer(peers, peer);
            }
        }
        uint32_t remoteServerBase = serverBase ^ SERVER_SIZE;
        for (uint32_t remoteLocalRank = 0; remoteLocalRank < SERVER_SIZE; ++remoteLocalRank) {
            AddPeer(peers, remoteServerBase + remoteLocalRank);
        }
    }

    if (peers.empty()) {
        return HCCL_SUCCESS;
    }

    std::vector<HcclChannelDesc> descs(peers.size());
    std::vector<ChannelHandle> handles(peers.size());
    for (size_t i = 0; i < peers.size(); ++i) {
        CHK_RET(AcquireDesc(comm, param.myRank, peers[i], &descs[i]));
    }
    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AICPU_TS, descs.data(),
        static_cast<uint32_t>(descs.size()), handles.data()));

    for (size_t i = 0; i < peers.size(); ++i) {
        void *remoteBuffer = nullptr;
        uint64_t remoteBufferSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, handles[i], &remoteBuffer, &remoteBufferSize));

        ChannelInfo channel;
        channel.remoteRank = peers[i];
        channel.notifyNum = CHANNEL_NOTIFY_NUM;
        channel.handle = handles[i];
        channel.remoteCclMem = CommBuffer{remoteBuffer, remoteBufferSize};
        resCtxHost.channels.push_back(channel);
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
        HCCL_ERROR("Only float32 is supported, dataType[%d]", static_cast<int32_t>(dataType)), HCCL_E_PARA);
    CHK_PRT_RET(op != HCCL_REDUCE_SUM,
        HCCL_ERROR("Only sum is supported, reduceOp[%d]", static_cast<int32_t>(op)), HCCL_E_PARA);
    CHK_PRT_RET(count > UINT64_MAX / sizeof(float), HCCL_ERROR("Input byte size overflows uint64"), HCCL_E_PARA);

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
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || (param.rankSize & (param.rankSize - 1)) != 0,
        HCCL_ERROR("This implementation requires a power-of-two rank size, rankSize[%u]", param.rankSize),
        HCCL_E_PARA);

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

        // T0..T6 drive seven layer-0 reads, T7..T13 drive seven layer-1
        // reads, and T14 copies this rank's own result slice. T0 notify index
        // 0 remains reserved for Host/AICPU synchronization; indices 1..14
        // receive worker completion signals.
        uint32_t threadNum = param.rankSize == CONTEST_RANK_SIZE ? CONTEST_THREAD_NUM : 1;
        uint32_t notifyNumPerThread =
            param.rankSize == CONTEST_RANK_SIZE ? CONTEST_THREAD_NOTIFY_NUM : 1;

        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, notifyNumPerThread, resCtxHost.threads.data()));
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

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
