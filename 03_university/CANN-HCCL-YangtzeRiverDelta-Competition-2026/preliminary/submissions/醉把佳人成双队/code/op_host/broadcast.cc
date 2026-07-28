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

HcclResult HcclBroadcast(
    void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(buf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param;
    sprintf(param.tag, "%s", "hccl_custom_broadcast");
    param.inputPtr = buf;
    param.outputPtr = buf;
    param.count = count;
    param.dataType = dataType;
    param.root = root;
    param.opType = HcclCMDType::HCCL_CMD_BROADCAST;

    HcclDfxOpInfo dfxInfo;
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));

    CommEngine aicpuTsEngine = CommEngine::COMM_ENGINE_AICPU_TS;
    CommEngine cpuTsEngine = CommEngine::COMM_ENGINE_CPU_TS;

    CHK_RET(HcclThreadAcquireWithStream(comm, cpuTsEngine, stream, 1, &param.cpuThread));
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &param.cpuThread, aicpuTsEngine, &param.cpuThreadOnAicpu));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, aicpuTsEngine, &ctx, &size) == HCCL_SUCCESS) {
        HCCL_INFO("Engine context already exists");
        param.resCtx = ctx;
        param.ctxSize = size;

        void *hostCtx = nullptr;
        uint64_t hostCtxSize = 0;
        CHK_RET(HcclEngineCtxGet(comm, param.tag, cpuTsEngine, &hostCtx, &hostCtxSize));
        ThreadHandle *aicpuThread = static_cast<ThreadHandle *>(hostCtx);
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));
    } else {
        AlgResourceCtx resCtxHost;

        void *cclBufferAddr = nullptr;
        uint64_t cclBufferSize = 0;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        // 1条控制Thread + 15条通信Worker。
        // 小消息使用15路一次直发；大消息使用其中7条Worker和第8条作为Copy Thread。
        constexpr uint32_t threadNum = 16;
        constexpr uint32_t notifyNumPerThread = 8;

        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(
            comm,
            aicpuTsEngine,
            threadNum,
            notifyNumPerThread,
            resCtxHost.threads.data()));

        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(
            comm,
            1,
            &resCtxHost.aicpuThread,
            cpuTsEngine,
            &param.aicpuThreadOnCpu));

        // 稳定基线：每个 rank 与其余所有 rank 建立 Channel。
        if (param.rankSize > 1) {
            const uint32_t channelNum = param.rankSize - 1;

            std::vector<uint32_t> remoteRanks;
            remoteRanks.reserve(channelNum);
            for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
                if (rank != param.myRank) {
                    remoteRanks.push_back(rank);
                }
            }

            std::vector<HcclChannelDesc> channelDescs(channelNum);
            CHK_RET(HcclChannelDescInit(channelDescs.data(), channelNum));

            uint32_t *netLayers = nullptr;
            uint32_t netLayerNum = 0;
            CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));

            // 双缓冲slot各使用4个Channel Notify：
            // ACK、root到远端分片、远端回传分片、Server内AllGather。
            // slot0使用[0..3]，slot1使用[4..7]。
            constexpr uint32_t channelNotifyNum = 8;

            for (uint32_t idx = 0; idx < channelNum; ++idx) {
                const uint32_t remoteRank = remoteRanks[idx];
                bool linkFound = false;

                for (uint32_t layerIdx = 0; layerIdx < netLayerNum; ++layerIdx) {
                    CommLink *links = nullptr;
                    uint32_t linkNum = 0;

                    CHK_RET(HcclRankGraphGetLinks(
                        comm,
                        netLayers[layerIdx],
                        param.myRank,
                        remoteRank,
                        &links,
                        &linkNum));

                    if (linkNum == 0) {
                        continue;
                    }

                    channelDescs[idx].remoteRank = remoteRank;
                    channelDescs[idx].channelProtocol = links[0].linkAttr.linkProtocol;
                    channelDescs[idx].localEndpoint = links[0].srcEndpointDesc;
                    channelDescs[idx].remoteEndpoint = links[0].dstEndpointDesc;
                    channelDescs[idx].notifyNum = channelNotifyNum;
                    linkFound = true;
                    break;
                }

                CHK_PRT_RET(
                    !linkFound,
                    HCCL_ERROR(
                        "No communication link from rank[%u] to rank[%u]",
                        param.myRank,
                        remoteRank),
                    HCCL_E_INTERNAL);
            }

            std::vector<ChannelHandle> channelHandles(channelNum);
            CHK_RET(HcclChannelAcquire(
                comm,
                aicpuTsEngine,
                channelDescs.data(),
                channelNum,
                channelHandles.data()));

            resCtxHost.channels.resize(channelNum);
            for (uint32_t idx = 0; idx < channelNum; ++idx) {
                void *remoteBufferAddr = nullptr;
                uint64_t remoteBufferSize = 0;

                CHK_RET(HcclChannelGetHcclBuffer(
                    comm,
                    channelHandles[idx],
                    &remoteBufferAddr,
                    &remoteBufferSize));

                resCtxHost.channels[idx].remoteRank = remoteRanks[idx];
                resCtxHost.channels[idx].notifyNum = channelNotifyNum;
                resCtxHost.channels[idx].handle = channelHandles[idx];
                resCtxHost.channels[idx].remoteCclMem = CommBuffer{remoteBufferAddr, remoteBufferSize};
            }
        }

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

    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}