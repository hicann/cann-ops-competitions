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
constexpr uint32_t CHANNEL_NOTIFY_NUM = 3; // ACK + DATA_SIGNAL (+ reserve)
// 拓扑图中最大扫描层数：覆盖本机 Full-Mesh(0) 与跨机 Clos(>=1)
constexpr uint32_t MAX_NET_LAYERS = 8;

// 构造单条 Channel 描述，从 srcRank -> dstRank 中选取协议为 TS 的链路
// 拓扑：每台server内8卡为Full-Mesh，server间走Clos。
// netLayer=0 仅包含直连Mesh；server间链路在更高层，因此需要逐层扫描
HcclResult FillChannelDesc(HcclComm comm, uint32_t srcRank, uint32_t dstRank, HcclChannelDesc &desc)
{
    CHK_RET(HcclChannelDescInit(&desc, 1));
    const CommProtocol protocol = CommProtocol::COMM_PROTOCOL_UBC_CTP;

    for (uint32_t netLayer = 0; netLayer < MAX_NET_LAYERS; netLayer++) {
        uint32_t listSize = 0;
        CommLink *linkList = nullptr;
        HcclResult ret = HcclRankGraphGetLinks(comm, netLayer, srcRank, dstRank, &linkList, &listSize);
        if (ret != HCCL_SUCCESS || listSize == 0 || linkList == nullptr) {
            continue; // 当前层无直连链路，尝试下一层
        }
        for (uint32_t idx = 0; idx < listSize; idx++) {
            const CommLink &link = linkList[idx];
            if (link.linkAttr.linkProtocol == protocol) {
                desc.remoteRank = dstRank;
                desc.notifyNum = CHANNEL_NOTIFY_NUM;
                desc.channelProtocol = link.linkAttr.linkProtocol;
                desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
                desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
                desc.localEndpoint.loc = link.srcEndpointDesc.loc;
                desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
                desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
                desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
                HCCL_INFO("[FillChannelDesc] rank %u -> %u via netLayer %u", srcRank, dstRank, netLayer);
                return HCCL_SUCCESS;
            }
        }
    }

    HCCL_ERROR("[FillChannelDesc] Protocol %d not found between rank %u and rank %u (scanned %u layers)",
               static_cast<int>(protocol), srcRank, dstRank, MAX_NET_LAYERS);
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
    sprintf(param.tag, "%s", "hccl_custom_allgather");
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
    // STEP 1: 解析拓扑信息
    // ==============================================
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));

    // ==============================================
    // STEP 2: 创建资源
    // ==============================================
    const CommEngine aicpuTsEngine = CommEngine::COMM_ENGINE_AICPU_TS;
    const CommEngine cpuTsEngine = CommEngine::COMM_ENGINE_CPU_TS;

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

        // Host 资源已经存在，复用资源（缓存了主aicpu thread句柄）
        void *hostCtx = nullptr;
        uint64_t hostCtxSize = sizeof(ThreadHandle);
        CHK_RET(HcclEngineCtxGet(comm, param.tag, cpuTsEngine, &hostCtx, &hostCtxSize));
        ThreadHandle *aicpuThread = static_cast<ThreadHandle *>(hostCtx);
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));
    } else {
        // Device 资源不存在，资源构建
        AlgResourceCtx resCtxHost;

        // 从通信域获取 HCCL Buffer（Device上的内存，默认总大小约 400MB）
        void *cclBufferAddr;
        uint64_t cclBufferSize;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        // ==============================================
        // STEP 2.2: 申请资源 Thread 和 Channel
        // ==============================================
        // 一对一线程驱动策略，与参考模板保持一致：
        //   threads[0]         -> 主 aicpu thread，负责 host/device 同步及算法编排
        //   threads[1..rankSize-2] -> 从thread，每个驱动一个channel，与远端rank一一对应
        // channel数量 = rankSize-1，thread数量 = rankSize-1，两者1:1对齐。
        // 主thread需要 slaveThreadNum 个 notify slot（每个从thread用一个），
        // 从thread只需要 1 个 slot（用于汇报）。
        const uint32_t slaveThreadNum = (param.rankSize <= 2) ? 0 : (param.rankSize - 2);
        const uint32_t threadNum = slaveThreadNum + 1;

        resCtxHost.slaveThreadNum = slaveThreadNum;
        resCtxHost.notifyNumOnMainThread = slaveThreadNum;
        resCtxHost.notifyNumPerThread.assign(slaveThreadNum, 1);

        // 每个thread分配 (notifyNumOnMainThread + 1) 个 slot：+1 保留给 host/device 同步
        resCtxHost.threads.resize(threadNum);
        const uint32_t slotsPerThread = (param.rankSize == 1) ? 1 : (resCtxHost.notifyNumOnMainThread + 1);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, slotsPerThread, resCtxHost.threads.data()));

        // 主aicpu thread = threads[0]，同时负责 host/device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine,
                                             &param.aicpuThreadOnCpu));

        // 为每个对端rank申请一个channel（与每个远端仅一条channel进行通信）
        if (param.rankSize > 1) {
            const uint32_t channelNum = param.rankSize - 1;
            std::vector<HcclChannelDesc> descs(channelNum);
            std::vector<ChannelHandle> channels(channelNum);
            std::vector<uint32_t> remoteRanks(channelNum);

            uint32_t idx = 0;
            for (uint32_t remoteRank = 0; remoteRank < param.rankSize; remoteRank++) {
                if (remoteRank == param.myRank) {
                    continue;
                }
                CHK_RET(FillChannelDesc(comm, param.myRank, remoteRank, descs[idx]));
                remoteRanks[idx] = remoteRank;
                idx++;
            }
            CHK_RET(HcclChannelAcquire(comm, aicpuTsEngine, descs.data(), channelNum, channels.data()));

            resCtxHost.channels.resize(channelNum);
            for (uint32_t i = 0; i < channelNum; i++) {
                ChannelInfo info;
                info.remoteRank = remoteRanks[i];
                info.handle = channels[i];
                info.notifyNum = CHANNEL_NOTIFY_NUM;
                void *cclBuf = nullptr;
                uint64_t cclBufSize = 0;
                CHK_RET(HcclChannelGetHcclBuffer(comm, channels[i], &cclBuf, &cclBufSize));
                info.remoteCclMem = CommBuffer{cclBuf, cclBufSize};
                resCtxHost.channels[i] = info;
            }
        }

        // ==============================================
        // STEP 2.3: 申请通信引擎上下文
        // ==============================================
        std::vector<char> seq = resCtxHost.Serialize();
        uint64_t seqSize = seq.size();
        param.ctxSize = seqSize;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, aicpuTsEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, aicpuTsEngine, param.tag, seq.data(), seqSize, 0));

        // 缓存主aicpu thread句柄到cpu引擎上下文，以便host复用
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
