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
        // STEP 2.2: 构建 Ring 拓扑
        // ==============================================
        uint32_t myRank = param.myRank;
        uint32_t rankSize = param.rankSize;

        // 环形拓扑：前驱与后继 Rank
        uint32_t prevRank = (myRank + rankSize - 1) % rankSize;
        uint32_t nextRank = (myRank + 1) % rankSize;
        resCtxHost.algConfig.prevRank = prevRank;
        resCtxHost.algConfig.nextRank = nextRank;

        HCCL_INFO("Ring topology: myRank=%u, prevRank=%u, nextRank=%u, rankSize=%u", myRank, prevRank, nextRank,
            rankSize);

        // ==============================================
        // STEP 2.3: 申请 Thread 资源
        // ==============================================
        // Ring AllReduce 按顺序执行 send/recv，单 Thread 即可满足；
        // 每个 Thread 申请 2 个 Notify（NOTIFY_IDX_ACK + NOTIFY_IDX_DATA_SIGNAL）
        uint32_t threadNum = 1;
        uint32_t notifyNumPerThread = 2;

        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, notifyNumPerThread, resCtxHost.threads.data()));
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        // ==============================================
        // STEP 2.4: 申请 Channel 资源（仅当 rankSize > 1 时需要）
        // ==============================================
        if (rankSize > 1) {
            // 查询网络分层与 Rank Graph 链路
            // ⚠️ HcclRankGraphGetLayers 返回的 netLayers 指针由库内管理，
            // 后续 HcclRankGraphGetLinks 等调用可能致其失效，必须立即拷贝。
            uint32_t *netLayersRaw = nullptr;
            uint32_t netLayerNum = 0;
            CHK_RET(HcclRankGraphGetLayers(comm, &netLayersRaw, &netLayerNum));
            HCCL_INFO("Net layer count: %u", netLayerNum);
            std::vector<uint32_t> netLayers(netLayersRaw, netLayersRaw + netLayerNum);

            // Ring 算法：rankSize>2 时需要 2 个 Channel（前驱+后继）；
            // rankSize==2 时前驱==后继，仅需 1 个 Channel
            uint32_t channelNum = (rankSize > 2) ? 2 : 1;
            // neighborRanks 按 rank ID 升序排列，确保所有 rank 的 Channel
            // 创建顺序一致，避免环形等待死锁（低 rank pair 先完成握手）
            uint32_t neighborRanks[2];
            if (nextRank < prevRank) {
                neighborRanks[0] = nextRank;
                neighborRanks[1] = prevRank;
            } else {
                neighborRanks[0] = prevRank;
                neighborRanks[1] = nextRank;
            }
            resCtxHost.channels.resize(channelNum);

            // 将本地 Buffer 注册为通信内存句柄（用于 Channel 建立时与远端交换地址）
            CommMem localCommMem;
            localCommMem.type = COMM_MEM_TYPE_DEVICE;
            localCommMem.addr = cclBufferAddr;
            localCommMem.size = cclBufferSize;

            for (uint32_t i = 0; i < channelNum; i++) {
                uint32_t neighbor = neighborRanks[i];

                // 遍历各网络层次，查找到达 neighbor 的 CommLink
                CommLink *links = nullptr;
                uint32_t linkNum = 0;
                bool foundLink = false;
                for (uint32_t layer = 0; layer < netLayerNum; layer++) {
                    uint32_t queryLayer = netLayers[layer];
                    CHK_RET(HcclRankGraphGetLinks(comm, queryLayer, myRank, neighbor, &links, &linkNum));
                    if (linkNum > 0 && links != nullptr) {
                        foundLink = true;
                        HCCL_INFO("Found %u link(s) to rank %u at layer %u", linkNum, neighbor, queryLayer);
                        break;
                    }
                }
                CHK_PRT_RET(!foundLink, HCCL_ERROR("[%s] No link found from rank %u to rank %u in any layer", __func__,
                                                   myRank, neighbor),
                    HCCL_E_INTERNAL);

                // ⚠️ 立即复制 Link 数据到局部变量：
                // HcclRankGraphGetLinks 返回的 links 指针由库内管理，后续调用
                // HcclCommMemReg 等 API 可能导致 links 被内部复用而失效。
                // 必须在调用其他 HCCL API 之前完成数据拷贝。
                CommProtocol protocol = links[0].linkAttr.linkProtocol;
                EndpointDesc localEp = links[0].srcEndpointDesc;
                EndpointDesc remoteEp = links[0].dstEndpointDesc;

                // 注册本地 Buffer 为可交换内存（每 Channel 使用独立 tag 避免冲突）
                char memTag[TAG_LENGTH + 8];
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

                // 获取远端交换过来的内存信息
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

                HCCL_INFO("Channel[%u] created: localRank=%u -> remoteRank=%u, remoteCclMem=[%p, %lu]", i, myRank,
                    neighbor, resCtxHost.channels[i].remoteCclMem.addr, resCtxHost.channels[i].remoteCclMem.size);
            }
        }

        // ==============================================
        // STEP 2.6: 申请通信引擎上下文
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
