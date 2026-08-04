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

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 2;
constexpr uint32_t RANKS_PER_SERVER = 8;

struct HostResourceCtx {
    aclrtStream stream;
    ThreadHandle cpuThread;
    ThreadHandle cpuThreadOnAicpu;
    ThreadHandle aicpuThreadOnCpu;
    uint32_t myRank;
    uint32_t rankSize;
};

HcclResult FillChannelDesc(
    HcclComm comm, uint32_t myRank, uint32_t remoteRank, HcclChannelDesc &desc)
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    CHK_PRT_RET(netLayers == nullptr || netLayerNum == 0,
        HCCL_ERROR("No network layer found for rank %u", myRank), HCCL_E_NOT_FOUND);

    const bool sameServer = (myRank / RANKS_PER_SERVER) == (remoteRank / RANKS_PER_SERVER);
    const uint32_t preferredLayer = sameServer ? 0U : 1U;
    // Same preference order as the official AICPU_TS channel selector.
    const CommProtocol supportedProtocols[] = {
        CommProtocol::COMM_PROTOCOL_UBC_CTP,
        CommProtocol::COMM_PROTOCOL_UBC_TP,
        CommProtocol::COMM_PROTOCOL_PCIE,
        CommProtocol::COMM_PROTOCOL_UBOE,
    };

    // Pass 0 searches the bandwidth-optimal layer. Pass 1 is a defensive
    // fallback for unexpected RankGraph layouts.
    for (uint32_t pass = 0; pass < 2; ++pass) {
        for (uint32_t layerIdx = 0; layerIdx < netLayerNum; ++layerIdx) {
            const uint32_t netLayer = netLayers[layerIdx];
            const bool isPreferred = netLayer == preferredLayer;
            if ((pass == 0 && !isPreferred) || (pass == 1 && isPreferred)) {
                continue;
            }

            uint32_t listSize = 0;
            CommLink *linkList = nullptr;
            HcclResult ret =
                HcclRankGraphGetLinks(comm, netLayer, myRank, remoteRank, &linkList, &listSize);
            if (ret != HCCL_SUCCESS || linkList == nullptr) {
                continue;
            }

            for (const CommProtocol protocol : supportedProtocols) {
                for (uint32_t linkIdx = 0; linkIdx < listSize; ++linkIdx) {
                    const CommLink &link = linkList[linkIdx];
                    if (link.linkAttr.linkProtocol != protocol) {
                        continue;
                    }

                    CHK_RET(HcclChannelDescInit(&desc, 1));
                    desc.remoteRank = remoteRank;
                    desc.notifyNum = CHANNEL_NOTIFY_NUM;
                    desc.channelProtocol = link.linkAttr.linkProtocol;
                    desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
                    desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
                    desc.localEndpoint.loc = link.srcEndpointDesc.loc;
                    desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
                    desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
                    desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
                    HCCL_INFO("Channel rank %u -> %u uses layer %u protocol %d",
                        myRank, remoteRank, netLayer, static_cast<int>(protocol));
                    return HCCL_SUCCESS;
                }
            }
        }
    }

    HCCL_ERROR("No AICPU_TS link found between rank %u and rank %u", myRank, remoteRank);
    return HCCL_E_NOT_FOUND;
}
} // namespace

HcclResult HcclAllGather(
    void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    // 构造算子参数
    OpParam param;
    sprintf(param.tag, "%s", "hccl_custom_allgather_host_hotpath_cache_exp035");
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = sendCount;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;

    // 注册算子信息
    HcclDfxOpInfo dfxInfo;
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    // ==============================================
    // STEP 2: 创建资源
    // ==============================================
    CommEngine aicpuTsEngine = CommEngine::COMM_ENGINE_AICPU_TS;
    CommEngine cpuTsEngine = CommEngine::COMM_ENGINE_CPU_TS;

    void *ctx = nullptr;
    uint64_t size = 0;
    const HcclResult getAicpuCtxRet =
        HcclEngineCtxGet(comm, param.tag, aicpuTsEngine, &ctx, &size);
    if (getAicpuCtxRet == HCCL_SUCCESS && ctx != nullptr && size != 0) {
        // AICPU 资源已经存在，复用资源
        HCCL_INFO("Engine context already exists");
        param.resCtx = ctx;
        param.ctxSize = size;

        // Host 资源已经存在，复用资源
        void *hostCtx = nullptr;
        uint64_t hostCtxSize = 0;
        CHK_RET(HcclEngineCtxGet(comm, param.tag, cpuTsEngine, &hostCtx, &hostCtxSize));
        CHK_PRT_RET(hostCtx == nullptr || hostCtxSize != sizeof(HostResourceCtx),
            HCCL_ERROR("Invalid Host context, addr=%p size=%llu expected=%zu",
                hostCtx, static_cast<unsigned long long>(hostCtxSize),
                sizeof(HostResourceCtx)),
            HCCL_E_INTERNAL);
        const HostResourceCtx *hostResource =
            static_cast<const HostResourceCtx *>(hostCtx);
        param.myRank = hostResource->myRank;
        param.rankSize = hostResource->rankSize;
        param.aicpuThreadOnCpu = hostResource->aicpuThreadOnCpu;

        // The benchmark reuses one stream. Reuse its already-acquired and
        // already-exported thread handles without re-entering the resource
        // manager. Preserve full multi-stream semantics with the original
        // Acquire/Export path when a different stream is supplied.
        if (hostResource->stream == stream) {
            param.cpuThread = hostResource->cpuThread;
            param.cpuThreadOnAicpu = hostResource->cpuThreadOnAicpu;
        } else {
            CHK_RET(HcclThreadAcquireWithStream(
                comm, cpuTsEngine, stream, 1, &param.cpuThread));
            CHK_RET(HcclThreadExportToCommEngine(
                comm, 1, &param.cpuThread, aicpuTsEngine,
                &param.cpuThreadOnAicpu));
        }
    } else {
        // Device 资源不存在，资源构建
        CHK_RET(HcclGetRankId(comm, &param.myRank));
        CHK_RET(HcclGetRankSize(comm, &param.rankSize));
        CHK_RET(HcclThreadAcquireWithStream(
            comm, cpuTsEngine, stream, 1, &param.cpuThread));
        CHK_RET(HcclThreadExportToCommEngine(
            comm, 1, &param.cpuThread, aicpuTsEngine,
            &param.cpuThreadOnAicpu));

        AlgResourceCtx resCtxHost;

        // 从通信域获取 HCCL Buffer（Device上的内存，默认总大小400MB）
        void *cclBufferAddr;
        uint64_t cclBufferSize;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        // ==============================================
        // STEP 2.2: 申请资源Thread和Channel
        // ==============================================

        // 创建 AICPU_TS 通信引擎上的 thread 资源
        // Thread 0 is the Host/control/staging thread. Threads [1, rankSize-1]
        // each drive exactly one remote Channel. Channel count remains
        // rankSize - 1, so every peer still has only one Channel.
        const uint32_t threadNum = param.rankSize > 1 ? param.rankSize : 1;
        // Notify 0 on worker threads is used by main-to-worker start sync.
        // Main-thread Notify 0 is reserved for Host->AICPU; worker completion
        // therefore uses the unique indices [1, threadNum - 1].
        const uint32_t notifyNumPerThread = threadNum;

        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, notifyNumPerThread, resCtxHost.threads.data()));
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        // 每个对端恰好申请一个Channel。Channel顺序与Thread顺序一一对应。
        if (param.rankSize > 1) {
            const uint32_t channelNum = param.rankSize - 1;
            std::vector<HcclChannelDesc> channelDescs(channelNum);
            std::vector<ChannelHandle> channelHandles(channelNum);
            std::vector<uint32_t> remoteRanks(channelNum);

            uint32_t channelIdx = 0;
            for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
                if (remoteRank == param.myRank) {
                    continue;
                }
                CHK_RET(FillChannelDesc(
                    comm, param.myRank, remoteRank, channelDescs[channelIdx]));
                remoteRanks[channelIdx] = remoteRank;
                ++channelIdx;
            }

            CHK_RET(HcclChannelAcquire(
                comm, aicpuTsEngine, channelDescs.data(), channelNum, channelHandles.data()));

            resCtxHost.channels.resize(channelNum);
            for (uint32_t idx = 0; idx < channelNum; ++idx) {
                void *remoteCclBuffer = nullptr;
                uint64_t remoteCclBufferSize = 0;
                CHK_RET(HcclChannelGetHcclBuffer(
                    comm, channelHandles[idx], &remoteCclBuffer, &remoteCclBufferSize));
                resCtxHost.channels[idx].remoteRank = remoteRanks[idx];
                resCtxHost.channels[idx].notifyNum = CHANNEL_NOTIFY_NUM;
                resCtxHost.channels[idx].handle = channelHandles[idx];
                resCtxHost.channels[idx].remoteCclMem =
                    CommBuffer{remoteCclBuffer, remoteCclBufferSize};
            }
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
        // Cache immutable communicator metadata and stable exported handles
        // in the CPU context. These handles remain owned by the communicator.
        void *hostCtx = nullptr;
        const HostResourceCtx hostResource{
            stream,
            param.cpuThread,
            param.cpuThreadOnAicpu,
            param.aicpuThreadOnCpu,
            param.myRank,
            param.rankSize,
        };
        const uint64_t hostCtxSize = sizeof(HostResourceCtx);
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, cpuTsEngine, hostCtxSize, &hostCtx));
        CHK_RET(HcclEngineCtxCopy(
            comm, cpuTsEngine, param.tag, &hostResource, hostCtxSize, 0));
    }

    // ==============================================
    // STEP 3: 下发 AICPU Kernel
    // ==============================================
    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}
