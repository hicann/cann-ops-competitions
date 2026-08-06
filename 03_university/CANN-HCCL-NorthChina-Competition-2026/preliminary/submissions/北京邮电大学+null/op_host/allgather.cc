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
#include <array>
#include <limits>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

HcclResult HcclAllGather(
    void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    const auto dataTypeSize = SIZE_TABLE.find(dataType);
    CHK_PRT_RET(dataTypeSize == SIZE_TABLE.end(), HCCL_ERROR("Unsupported data type[%d]", dataType), HCCL_E_PARA);
    CHK_PRT_RET(sendCount > std::numeric_limits<uint64_t>::max() / dataTypeSize->second,
        HCCL_ERROR("Input size overflow, count[%llu]", static_cast<unsigned long long>(sendCount)), HCCL_E_PARA);

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
    const uint64_t inputSize = sendCount * dataTypeSize->second;
    CHK_PRT_RET(param.rankSize == 0 || inputSize > std::numeric_limits<uint64_t>::max() / param.rankSize,
        HCCL_ERROR("Invalid rank size[%u] or output size overflow", param.rankSize), HCCL_E_PARA);

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

        // 创建 AICPU_TS 通信引擎上的 thread 资源
        // threads[0] 只负责 Host/Device 同步和本地拷贝；每个远端 rank 独占一条工作 Thread。
        uint32_t threadNum = param.rankSize;
        uint32_t notifyNumPerThread = threadNum;

        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, notifyNumPerThread, resCtxHost.threads.data()));
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(
            comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        std::vector<HcclChannelDesc> channelDescs;
        channelDescs.reserve(param.rankSize - 1);
        if (param.rankSize > 1) {
            // 每个 rank 与其余所有 rank 建立一条 Channel，充分利用 Mesh 和 Clos 的全连接带宽。
            uint32_t *netLayers = nullptr;
            uint32_t netLayerNum = 0;
            CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
            CHK_PRT_RET(
                netLayers == nullptr || netLayerNum == 0, HCCL_ERROR("No network layer found"), HCCL_E_INTERNAL);
            std::vector<uint32_t> netLayerList(netLayers, netLayers + netLayerNum);
            constexpr std::array<CommProtocol, 4> supportedProtocols = {CommProtocol::COMM_PROTOCOL_UBC_CTP,
                CommProtocol::COMM_PROTOCOL_UBC_TP, CommProtocol::COMM_PROTOCOL_PCIE,
                CommProtocol::COMM_PROTOCOL_UBOE};

            for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
                if (remoteRank == param.myRank) {
                    continue;
                }

                bool linkFound = false;
                for (uint32_t netLayer : netLayerList) {
                    CommLink *links = nullptr;
                    uint32_t linkNum = 0;
                    CHK_RET(HcclRankGraphGetLinks(comm, netLayer, param.myRank, remoteRank, &links, &linkNum));
                    CHK_PRT_RET(linkNum != 0 && links == nullptr,
                        HCCL_ERROR("Invalid links from rank[%u] to rank[%u]", param.myRank, remoteRank),
                        HCCL_E_INTERNAL);
                    for (CommProtocol protocol : supportedProtocols) {
                        for (uint32_t linkIdx = 0; linkIdx < linkNum; ++linkIdx) {
                            if (links[linkIdx].linkAttr.linkProtocol != protocol) {
                                continue;
                            }

                            HcclChannelDesc channelDesc;
                            CHK_RET(HcclChannelDescInit(&channelDesc, 1));
                            channelDesc.remoteRank = remoteRank;
                            channelDesc.channelProtocol = protocol;
                            channelDesc.localEndpoint = links[linkIdx].srcEndpointDesc;
                            channelDesc.remoteEndpoint = links[linkIdx].dstEndpointDesc;
                            channelDesc.notifyNum = 2;
                            channelDescs.push_back(channelDesc);
                            linkFound = true;
                            break;
                        }
                        if (linkFound) {
                            break;
                        }
                    }
                    if (linkFound) {
                        break;
                    }
                }
                CHK_PRT_RET(!linkFound, HCCL_ERROR("No AICPU link from rank[%u] to rank[%u]", param.myRank, remoteRank),
                    HCCL_E_INTERNAL);
            }
        }

        resCtxHost.channels.resize(channelDescs.size());
        if (!channelDescs.empty()) {
            std::vector<ChannelHandle> channelHandles(channelDescs.size());
            CHK_RET(HcclChannelAcquire(
                comm, aicpuTsEngine, channelDescs.data(), channelDescs.size(), channelHandles.data()));
            for (size_t channelIdx = 0; channelIdx < channelDescs.size(); ++channelIdx) {
                ChannelInfo &channel = resCtxHost.channels[channelIdx];
                channel.remoteRank = channelDescs[channelIdx].remoteRank;
                channel.notifyNum = channelDescs[channelIdx].notifyNum;
                channel.handle = channelHandles[channelIdx];
                CHK_RET(HcclChannelGetHcclBuffer(
                    comm, channel.handle, &channel.remoteCclMem.addr, &channel.remoteCclMem.size));
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
