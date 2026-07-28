/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root path of the software repository for the full text of the License.
 */

#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

// Fill channel descriptor from link info - try UBC_CTP first, fall back to any
static HcclResult FillChannelDesc(HcclComm comm, uint32_t *netLayers, uint32_t netLayerNum,
                                   uint32_t myRank, uint32_t remoteRank, HcclChannelDesc &desc)
{
    // First pass: try UBC_CTP protocol
    const CommProtocol targetProtocol = CommProtocol::COMM_PROTOCOL_UBC_CTP;
    for (uint32_t layerIdx = 0; layerIdx < netLayerNum; layerIdx++) {
        CommLink *linkList = nullptr;
        uint32_t listSize = 0;
        HcclResult ret = HcclRankGraphGetLinks(comm, netLayers[layerIdx],
            myRank, remoteRank, &linkList, &listSize);
        if (ret != HCCL_SUCCESS || listSize == 0 || linkList == nullptr) {
            continue;
        }
        for (uint32_t idx = 0; idx < listSize; idx++) {
            if (linkList[idx].linkAttr.linkProtocol == targetProtocol) {
                desc.channelProtocol = linkList[idx].linkAttr.linkProtocol;
                desc.localEndpoint.protocol = linkList[idx].srcEndpointDesc.protocol;
                desc.localEndpoint.commAddr = linkList[idx].srcEndpointDesc.commAddr;
                desc.localEndpoint.loc = linkList[idx].srcEndpointDesc.loc;
                desc.remoteEndpoint.protocol = linkList[idx].dstEndpointDesc.protocol;
                desc.remoteEndpoint.commAddr = linkList[idx].dstEndpointDesc.commAddr;
                desc.remoteEndpoint.loc = linkList[idx].dstEndpointDesc.loc;
                HCCL_INFO("Found UBC_CTP link at layer %u: rank %u -> %u", netLayers[layerIdx], myRank, remoteRank);
                return HCCL_SUCCESS;
            }
        }
    }

    // Second pass: fall back to any available link
    for (uint32_t layerIdx = 0; layerIdx < netLayerNum; layerIdx++) {
        CommLink *linkList = nullptr;
        uint32_t listSize = 0;
        HcclResult ret = HcclRankGraphGetLinks(comm, netLayers[layerIdx],
            myRank, remoteRank, &linkList, &listSize);
        if (ret != HCCL_SUCCESS || listSize == 0 || linkList == nullptr) {
            continue;
        }
        desc.channelProtocol = linkList[0].linkAttr.linkProtocol;
        desc.localEndpoint.protocol = linkList[0].srcEndpointDesc.protocol;
        desc.localEndpoint.commAddr = linkList[0].srcEndpointDesc.commAddr;
        desc.localEndpoint.loc = linkList[0].srcEndpointDesc.loc;
        desc.remoteEndpoint.protocol = linkList[0].dstEndpointDesc.protocol;
        desc.remoteEndpoint.commAddr = linkList[0].dstEndpointDesc.commAddr;
        desc.remoteEndpoint.loc = linkList[0].dstEndpointDesc.loc;
        HCCL_INFO("Fallback link at layer %u: rank %u -> %u, protocol=%u",
                  netLayers[layerIdx], myRank, remoteRank, (uint32_t)linkList[0].linkAttr.linkProtocol);
        return HCCL_SUCCESS;
    }

    HCCL_ERROR("No link found for rank %u -> %u", myRank, remoteRank);
    return HCCL_E_INTERNAL;
}

HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    // Build op params
    OpParam param;
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = count;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;

    // Register op info
    HcclDfxOpInfo dfxInfo;
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    // STEP 1: Parse topology
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));

    // Use versioned tag to force context recreation when algorithm changes
    sprintf(param.tag, "%s_%u", "hccl_allreduce_v17", param.rankSize);

    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));

    // STEP 2: Create resources
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

        void *cclBufferAddr;
        uint64_t cclBufferSize;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        resCtxHost.rankSize = param.rankSize;
        resCtxHost.myRank = param.myRank;

        constexpr uint32_t serverSize = 8;
        if (param.rankSize == 16) {
            // ===== Recursive Halving + Butterfly for 2x8 topology =====
            // 4 channels: [0]=XOR^1, [1]=XOR^2, [2]=XOR^4 (within-server), [3]=XOR^8 (cross-server)
            resCtxHost.useRecursiveHalving = true;
            resCtxHost.serverRankSize = serverSize;
            resCtxHost.meshPartnerRanks[0] = param.myRank ^ 1;
            resCtxHost.meshPartnerRanks[1] = param.myRank ^ 2;
            resCtxHost.meshPartnerRanks[2] = param.myRank ^ 4;
            resCtxHost.crossServerPartnerRank = param.myRank ^ 8;
            resCtxHost.crossServerChIdx = 3;

            HCCL_INFO("RecHalving 16: rank=%u, mesh=[%u,%u,%u], cross=%u, chIdx=%u",
                      param.myRank,
                      resCtxHost.meshPartnerRanks[0], resCtxHost.meshPartnerRanks[1],
                      resCtxHost.meshPartnerRanks[2],
                      resCtxHost.crossServerPartnerRank, resCtxHost.crossServerChIdx);

            uint32_t numChannels = 4;
            uint32_t remoteRanks[4] = {
                param.myRank ^ 1, param.myRank ^ 2,
                param.myRank ^ 4, param.myRank ^ 8
            };

            resCtxHost.channels.resize(numChannels);
            constexpr uint32_t channelNotifyNum = 2;
            HcclChannelDesc descs[4];

            for (uint32_t i = 0; i < numChannels; i++) {
                CHK_RET(HcclChannelDescInit(&descs[i], 1));
                descs[i].remoteRank = remoteRanks[i];
                descs[i].notifyNum = channelNotifyNum;
                CHK_RET(FillChannelDesc(comm, netLayers, netLayerNum, param.myRank, remoteRanks[i], descs[i]));
                HCCL_INFO("Channel %u: rank %u -> %u", i, param.myRank, remoteRanks[i]);
            }

            ChannelHandle handles[4];
            CHK_RET(HcclChannelAcquire(comm, aicpuTsEngine, descs, numChannels, handles));

            for (uint32_t i = 0; i < numChannels; i++) {
                resCtxHost.channels[i].handle = handles[i];
                resCtxHost.channels[i].remoteRank = remoteRanks[i];
                resCtxHost.channels[i].notifyNum = channelNotifyNum;
                CHK_RET(HcclChannelGetHcclBuffer(comm, handles[i],
                    &resCtxHost.channels[i].remoteCclMem.addr,
                    &resCtxHost.channels[i].remoteCclMem.size));
            }
        } else {
            // ===== Simple Ring fallback for non-16 ranks =====
            resCtxHost.ringPos = param.myRank;
            resCtxHost.prevRank = (param.myRank + param.rankSize - 1) % param.rankSize;
            resCtxHost.nextRank = (param.myRank + 1) % param.rankSize;
            HCCL_INFO("Default ring: rank=%u, ringPos=%u, prev=%u, next=%u, rankSize=%u",
                      param.myRank, resCtxHost.ringPos, resCtxHost.prevRank, resCtxHost.nextRank, param.rankSize);

            uint32_t numChannels = 2;
            uint32_t remoteRanks[2] = {resCtxHost.prevRank, resCtxHost.nextRank};

            resCtxHost.channels.resize(numChannels);
            constexpr uint32_t channelNotifyNum = 2;
            HcclChannelDesc descs[2];

            for (uint32_t i = 0; i < numChannels; i++) {
                CHK_RET(HcclChannelDescInit(&descs[i], 1));
                descs[i].remoteRank = remoteRanks[i];
                descs[i].notifyNum = channelNotifyNum;
                CHK_RET(FillChannelDesc(comm, netLayers, netLayerNum, param.myRank, remoteRanks[i], descs[i]));
                HCCL_INFO("Channel %u: rank %u -> %u", i, param.myRank, remoteRanks[i]);
            }

            ChannelHandle handles[2];
            CHK_RET(HcclChannelAcquire(comm, aicpuTsEngine, descs, numChannels, handles));

            for (uint32_t i = 0; i < numChannels; i++) {
                resCtxHost.channels[i].handle = handles[i];
                resCtxHost.channels[i].remoteRank = remoteRanks[i];
                resCtxHost.channels[i].notifyNum = channelNotifyNum;
                CHK_RET(HcclChannelGetHcclBuffer(comm, handles[i],
                    &resCtxHost.channels[i].remoteCclMem.addr,
                    &resCtxHost.channels[i].remoteCclMem.size));
            }
        }

        uint32_t threadNum = 1;
        uint32_t notifyNumPerThread = 2;
        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, notifyNumPerThread, resCtxHost.threads.data()));
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        // Create engine context
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

    // STEP 3: Launch AICPU Kernel
    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}
