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

// 新增
#include <vector>
// 结束

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

// 新增
namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;
// 新增
constexpr uint32_t DEFAULT_NET_LAYER = 0;
// 结束

HcclResult FillChannelDesc(HcclComm comm, uint32_t srcRank, uint32_t dstRank, HcclChannelDesc &desc)
{
    CHK_RET(HcclChannelDescInit(&desc, 1));

    // 新增
    // 官方自定义 AllGather 示例固定优先查询 netLayer 0。比赛拓扑为一层 MESH_1D_CLOS，
    // 先走该路径可以避免遍历顺序变化时选到其他 plane；若未找到则保留原有全层回退。
    CommLink *defaultLayerLinks = nullptr;
    uint32_t defaultLayerLinkNum = 0;
    HcclResult defaultLayerRet = HcclRankGraphGetLinks(
        comm, DEFAULT_NET_LAYER, srcRank, dstRank, &defaultLayerLinks, &defaultLayerLinkNum);
    if (defaultLayerRet == HCCL_SUCCESS) {
        for (uint32_t linkIdx = 0; linkIdx < defaultLayerLinkNum; ++linkIdx) {
            const CommLink &link = defaultLayerLinks[linkIdx];
            if (link.linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_UBC_CTP) {
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
    }
    // 结束

    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));

    for (uint32_t layerIdx = 0; layerIdx < netLayerNum; ++layerIdx) {
        CommLink *linkList = nullptr;
        uint32_t listSize = 0;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayers[layerIdx], srcRank, dstRank, &linkList, &listSize));
        for (uint32_t linkIdx = 0; linkIdx < listSize; ++linkIdx) {
            const CommLink &link = linkList[linkIdx];
            if (link.linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_UBC_CTP) {
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
    }

    HCCL_ERROR("No UBC_CTP link found between rank %u and rank %u", srcRank, dstRank);
    return HCCL_E_NOT_FOUND;
}
} // namespace
// 结束

HcclResult HcclAllGather(
    void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    // 新增
    CHK_PTR_NULL(recvBuf);
    // 结束
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
        // 删除
        // uint64_t hostCtxSize = 0;
        // 新增
        uint64_t hostCtxSize = sizeof(ThreadHandle);
        // 结束
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

        // 删除
        // // TODO: 根据通信算法申请 Thread 资源
        // // 创建 AICPU_TS 通信引擎上的 thread 资源
        // uint32_t threadNum = 1;          // TODO: 按需修改所申请的 Thread 数量（>=1）
        // uint32_t notifyNumPerThread = 1; // TODO: 按需修改所申请的 Thread 上的 Notify 数量（>=1）

        // 新增
        // 一个控制 Thread，加上每个 peer 独占的一个通信 Thread。
        uint32_t threadNum = param.rankSize;
        uint32_t notifyNumPerThread = param.rankSize;
        // 结束

        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, notifyNumPerThread, resCtxHost.threads.data()));
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        // 删除
        // // TODO: 根据通信算法申请 Channel 资源
        // // 调用 HcclRankGraphGetLinks()、HcclChannelDescInit()、HcclChannelAcquire() 等接口按需申请 Channel 资源

        // 新增
        // 一次性申请所有 peer 的资源超集，确保固定 tag 的缓存上下文可以覆盖全部消息尺寸。
        uint32_t channelNum = param.rankSize > 1 ? param.rankSize - 1 : 0;
        std::vector<HcclChannelDesc> channelDescs(channelNum);
        std::vector<ChannelHandle> channelHandles(channelNum);
        std::vector<uint32_t> remoteRanks(channelNum);

        uint32_t channelIdx = 0;
        for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
            if (remoteRank == param.myRank) {
                continue;
            }
            CHK_RET(FillChannelDesc(comm, param.myRank, remoteRank, channelDescs[channelIdx]));
            remoteRanks[channelIdx] = remoteRank;
            ++channelIdx;
        }

        if (channelNum > 0) {
            CHK_RET(HcclChannelAcquire(comm, aicpuTsEngine, channelDescs.data(), channelNum, channelHandles.data()));
        }

        resCtxHost.channels.reserve(channelNum);
        for (uint32_t idx = 0; idx < channelNum; ++idx) {
            void *remoteCclBufferAddr = nullptr;
            uint64_t remoteCclBufferSize = 0;
            CHK_RET(HcclChannelGetHcclBuffer(
                comm, channelHandles[idx], &remoteCclBufferAddr, &remoteCclBufferSize));

            ChannelInfo channel;
            channel.remoteRank = remoteRanks[idx];
            channel.threadIdx = idx + 1;
            channel.notifyNum = CHANNEL_NOTIFY_NUM;
            channel.handle = channelHandles[idx];
            channel.remoteCclMem = CommBuffer{remoteCclBufferAddr, remoteCclBufferSize};
            resCtxHost.channels.push_back(channel);
        }
        // 结束

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
