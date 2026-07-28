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
#include <vector>

#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

namespace {
bool IsSupportedDataType(HcclDataType dataType)
{
    switch (dataType) {
        case HCCL_DATA_TYPE_INT8:
        case HCCL_DATA_TYPE_INT16:
        case HCCL_DATA_TYPE_INT32:
        case HCCL_DATA_TYPE_FP16:
        case HCCL_DATA_TYPE_FP32:
        case HCCL_DATA_TYPE_INT64:
        case HCCL_DATA_TYPE_UINT64:
        case HCCL_DATA_TYPE_UINT8:
        case HCCL_DATA_TYPE_UINT16:
        case HCCL_DATA_TYPE_UINT32:
        case HCCL_DATA_TYPE_FP64:
        case HCCL_DATA_TYPE_BFP16:
        case HCCL_DATA_TYPE_INT128:
        case HCCL_DATA_TYPE_HIF8:
        case HCCL_DATA_TYPE_FP8E4M3:
        case HCCL_DATA_TYPE_FP8E5M2:
        case HCCL_DATA_TYPE_FP8E8M0:
            return true;
        default:
            return false;
    }
}

HcclResult FillChannelDesc(uint32_t dstRank, const CommLink &link, HcclChannelDesc &desc)
{
    CHK_RET(HcclChannelDescInit(&desc, 1));
    desc.remoteRank = dstRank;
    desc.notifyNum = BROADCAST_CHANNEL_NOTIFY_COUNT;
    desc.channelProtocol = link.linkAttr.linkProtocol;
    desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
    desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
    desc.localEndpoint.loc = link.srcEndpointDesc.loc;
    desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
    desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
    desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
    return HCCL_SUCCESS;
}

HcclResult FindChannelDesc(HcclComm comm, uint32_t srcRank, uint32_t dstRank, HcclChannelDesc &desc)
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));

    constexpr CommProtocol protocols[] = {CommProtocol::COMM_PROTOCOL_UBC_CTP, CommProtocol::COMM_PROTOCOL_UBC_TP,
        CommProtocol::COMM_PROTOCOL_PCIE, CommProtocol::COMM_PROTOCOL_UBOE};
    for (uint32_t layerIdx = netLayerNum; layerIdx > 0; layerIdx--) {
        CommLink *linkList = nullptr;
        uint32_t linkNum = 0;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayers[layerIdx - 1], srcRank, dstRank, &linkList, &linkNum));
        for (CommProtocol protocol : protocols) {
            for (uint32_t linkIdx = 0; linkIdx < linkNum; linkIdx++) {
                if (linkList[linkIdx].linkAttr.linkProtocol == protocol) {
                    return FillChannelDesc(dstRank, linkList[linkIdx], desc);
                }
            }
        }
    }

    HCCL_ERROR("No AICPU_TS channel between rank[%u] and rank[%u]", srcRank, dstRank);
    return HCCL_E_NOT_FOUND;
}

void AddUniqueRank(std::vector<uint32_t> &ranks, uint32_t rank)
{
    if (std::find(ranks.begin(), ranks.end(), rank) == ranks.end()) {
        ranks.push_back(rank);
    }
}

HcclResult AcquireChannels(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtxHost)
{
    if (param.rankSize <= 1) {
        return HCCL_SUCCESS;
    }

    std::vector<uint32_t> remoteRanks;
    if (param.rankSize == BROADCAST_RANK_SIZE) {
        for (uint32_t remoteRank = 0; remoteRank < param.rankSize; remoteRank++) {
            if (remoteRank != param.myRank) {
                remoteRanks.push_back(remoteRank);
            }
        }
    } else {
        for (uint32_t distance = 1; distance < param.rankSize; distance <<= 1) {
            AddUniqueRank(remoteRanks, (param.myRank + distance) % param.rankSize);
            AddUniqueRank(remoteRanks, (param.myRank + param.rankSize - distance) % param.rankSize);
            if (distance > param.rankSize / 2) {
                break;
            }
        }
    }

    std::vector<HcclChannelDesc> descs(remoteRanks.size());
    for (uint32_t idx = 0; idx < remoteRanks.size(); idx++) {
        CHK_RET(FindChannelDesc(comm, param.myRank, remoteRanks[idx], descs[idx]));
    }

    std::vector<ChannelHandle> handles(descs.size());
    CHK_RET(HcclChannelAcquire(
        comm, CommEngine::COMM_ENGINE_AICPU_TS, descs.data(), static_cast<uint32_t>(descs.size()), handles.data()));

    resCtxHost.channels.reserve(descs.size());
    for (uint32_t idx = 0; idx < descs.size(); idx++) {
        void *remoteCclBuffer = nullptr;
        uint64_t remoteCclBufferSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, handles[idx], &remoteCclBuffer, &remoteCclBufferSize));
        ChannelInfo channel;
        channel.remoteRank = descs[idx].remoteRank;
        channel.notifyNum = BROADCAST_CHANNEL_NOTIFY_COUNT;
        channel.handle = handles[idx];
        channel.remoteCclMem = CommBuffer{remoteCclBuffer, remoteCclBufferSize};
        resCtxHost.channels.push_back(channel);
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclBroadcast(
    void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    if (count != 0) {
        CHK_PTR_NULL(buf);
    }

    // Keep the operation descriptor independent of the selected data path.
    OpParam param;
    sprintf(param.tag, "%s", "hccl_custom_broadcast");
    param.inputPtr = buf;
    param.outputPtr = buf;
    param.count = count;
    param.dataType = dataType;
    param.root = root;
    param.opType = HcclCMDType::HCCL_CMD_BROADCAST;

    // Register diagnostics before acquiring reusable communication resources.
    HcclDfxOpInfo dfxInfo;
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || root >= param.rankSize,
        HCCL_ERROR("Invalid root[%u] for rankSize[%u]", root, param.rankSize), HCCL_E_PARA);
    CHK_PRT_RET(!IsSupportedDataType(dataType), HCCL_ERROR("Unsupported dataType[%d]", dataType), HCCL_E_PARA);

    CommEngine aicpuTsEngine = CommEngine::COMM_ENGINE_AICPU_TS;
    CommEngine cpuTsEngine = CommEngine::COMM_ENGINE_CPU_TS;

    // The user stream starts and joins the AICPU coordinator.
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

        // Incoming pieces occupy disjoint slots in this shared staging buffer.
        void *cclBufferAddr;
        uint64_t cclBufferSize;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        // Thread 0 is both coordinator and one peer stream; threads 1..14 cover the other peers.
        uint32_t threadNum = param.rankSize == BROADCAST_RANK_SIZE ? BROADCAST_THREAD_COUNT : 1;
        uint32_t notifyNumPerThread =
            param.rankSize == BROADCAST_RANK_SIZE ? BROADCAST_THREAD_NOTIFY_COUNT : 1;

        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, notifyNumPerThread, resCtxHost.threads.data()));
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        CHK_RET(AcquireChannels(comm, param, resCtxHost));

        // Both engine contexts are keyed only by the communication domain tag.
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