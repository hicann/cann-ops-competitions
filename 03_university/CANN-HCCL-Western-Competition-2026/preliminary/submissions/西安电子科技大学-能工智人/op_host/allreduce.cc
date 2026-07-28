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

HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

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
        // STEP 2.2: 构建 star拓扑
        // ==============================================
        uint32_t myRank = param.myRank;
        uint32_t rankSize = param.rankSize;
        const uint32_t ROOT_RANK = 0;

        resCtxHost.algConfig.rootRank = ROOT_RANK;
        resCtxHost.algConfig.isRoot = (myRank==ROOT_RANK);

         HCCL_INFO("Star topology: myRank=%u, isRoot=%d, rankSize=%u", myRank, resCtxHost.algConfig.isRoot, rankSize);

        // ==============================================
        // STEP 2.3: 申请 Thread 资源
        // ==============================================
        // 根节点需要处理 15 个 Channel 的并发/串行通信，为确保 Notify 充足，
        // 每个 Thread 申请 3 个 Notify（DATA_SIGNAL, ACK, 备用）
        uint32_t threadNum = 1;
        uint32_t notifyNumPerThread = 3;

        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, notifyNumPerThread, resCtxHost.threads.data()));
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        // ==============================================
        // STEP 2.4: 申请 Channel 资源
        // ==============================================
        // 根节点：与所有其他 Rank 建立 Channel（共 rankSize-1 个）
        // 非根节点：仅与根节点建立 1 个 Channel
        if (rankSize > 1) {
            // 查询网络分层
            uint32_t *netLayersRaw = nullptr;
            uint32_t netLayerNum = 0;
            CHK_RET(HcclRankGraphGetLayers(comm, &netLayersRaw, &netLayerNum));
            std::vector<uint32_t> netLayers(netLayersRaw, netLayersRaw + netLayerNum);

            // 确定需要建立 Channel 的目标 Rank 列表
            std::vector<uint32_t> targetRanks;
            if (myRank == ROOT_RANK) {
                // 根节点：连接所有非根节点
                for (uint32_t r = 0; r < rankSize; r++) {
                    if (r != ROOT_RANK) targetRanks.push_back(r);
                }
            } else {
                // 非根节点：只连接根节点
                targetRanks.push_back(ROOT_RANK);
            }

            uint32_t channelNum = targetRanks.size();
            resCtxHost.channels.resize(channelNum);

            // 注册本地 Buffer
            CommMem localCommMem;
            localCommMem.type = COMM_MEM_TYPE_DEVICE;
            localCommMem.addr = cclBufferAddr;
            localCommMem.size = cclBufferSize;

            // 为每个目标 Rank 建立 Channel
            for (uint32_t i = 0; i < channelNum; i++) {
                uint32_t neighbor = targetRanks[i];

                // 查找链路
                CommLink *links = nullptr;
                uint32_t linkNum = 0;
                bool foundLink = false;
                for (uint32_t layer = 0; layer < netLayerNum; layer++) {
                    uint32_t queryLayer = netLayers[layer];
                    CHK_RET(HcclRankGraphGetLinks(comm, queryLayer, myRank, neighbor, &links, &linkNum));
                    if (linkNum > 0 && links != nullptr) {
                        foundLink = true;
                        HCCL_INFO("Found %u link(s) from rank %u to rank %u at layer %u", linkNum, myRank, neighbor, queryLayer);
                        break;
                    }
                }
                CHK_PRT_RET(!foundLink, HCCL_ERROR("[%s] No link found from rank %u to rank %u", __func__, myRank, neighbor),
                    HCCL_E_INTERNAL);

                // 复制链路信息（避免指针失效）
                CommProtocol protocol = links[0].linkAttr.linkProtocol;
                EndpointDesc localEp = links[0].srcEndpointDesc;
                EndpointDesc remoteEp = links[0].dstEndpointDesc;

                // 注册内存
                char memTag[TAG_LENGTH + 32];
                snprintf(memTag, sizeof(memTag), "%s_ch%u", param.tag, i);
                HcclMemHandle memHandle = nullptr;
                CHK_RET(HcclCommMemReg(comm, memTag, &localCommMem, &memHandle));

                // 初始化 Channel 描述符
                HcclChannelDesc channelDesc;
                CHK_RET(HcclChannelDescInit(&channelDesc, 1));
                channelDesc.remoteRank = neighbor;
                channelDesc.channelProtocol = protocol;
                channelDesc.localEndpoint = localEp;
                channelDesc.remoteEndpoint = remoteEp;
                channelDesc.notifyNum = notifyNumPerThread;
                channelDesc.memHandles = &memHandle;
                channelDesc.memHandleNum = 1;

                // 申请 Channel
                CHK_RET(HcclChannelAcquire(comm, aicpuTsEngine, &channelDesc, 1, &resCtxHost.channels[i].handle));

                // 获取远端内存信息
                uint32_t remoteMemNum = 0;
                CommMem *remoteMems = nullptr;
                char **remoteMemTags = nullptr;
                CHK_RET(HcclChannelGetRemoteMems(comm, resCtxHost.channels[i].handle, &remoteMemNum, &remoteMems,
                    &remoteMemTags));
                CHK_PRT_RET(remoteMemNum == 0,
                    HCCL_ERROR("[%s] No remote memory exchanged for channel to rank %u", __func__, neighbor),
                    HCCL_E_INTERNAL);

                resCtxHost.channels[i].remoteRank = neighbor;
                resCtxHost.channels[i].notifyNum = notifyNumPerThread;
                resCtxHost.channels[i].remoteCclMem.addr = remoteMems[0].addr;
                resCtxHost.channels[i].remoteCclMem.size = remoteMems[0].size;

                HCCL_INFO("Channel[%u] created: localRank=%u -> remoteRank=%u, remoteCclMem=[%p, %lu]",
                    i, myRank, neighbor,
                    resCtxHost.channels[i].remoteCclMem.addr,
                    resCtxHost.channels[i].remoteCclMem.size);
            }
        }


        // ==============================================
        // STEP 2.5: 申请通信引擎上下文
        // ==============================================
        std::vector<char> seq = resCtxHost.Serialize();
        uint64_t seqSize = seq.size();
        param.ctxSize = seqSize;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, aicpuTsEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, aicpuTsEngine, param.tag, seq.data(), seqSize, 0));

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
