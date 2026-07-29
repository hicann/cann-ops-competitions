/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>

#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"
#include "log.h"

#include <vector>

namespace {
constexpr uint32_t RANK_SIZE = 16;
constexpr uint32_t SERVER_RANK_SIZE = 8;
constexpr uint32_t NET_LAYER_INTRA_SERVER = 0;
constexpr uint32_t NET_LAYER_INTER_SERVER = 1;

constexpr uint32_t UNIFIED_CHANNEL_NUM = RANK_SIZE - 1;
constexpr uint32_t UNIFIED_THREAD_NUM = RANK_SIZE;
constexpr uint32_t UNIFIED_THREAD_NOTIFY_NUM = RANK_SIZE - 1;
constexpr uint32_t UNIFIED_CHANNEL_NOTIFY_NUM = 2 * (RANK_SIZE - 1) + 2;
constexpr const char *UNIFIED_RESOURCE_TAG = "hccl_custom_allreduce_v34_cached_min_ccl";

HcclResult TryAcquireDesc(HcclComm comm, uint32_t netLayer, uint32_t srcRank, uint32_t dstRank,
    uint32_t notifyNum, HcclChannelDesc *desc, bool &found)
{
    found = false;
    CommLink *linkList = nullptr;
    uint32_t listSize = 0;
    CHK_RET(HcclRankGraphGetLinks(comm, netLayer, srcRank, dstRank, &linkList, &listSize));
    if (listSize == 0) {
        return HCCL_SUCCESS;
    }

    CHK_RET(HcclChannelDescInit(desc, 1));
    const CommLink &link = linkList[0];
    desc->remoteRank = dstRank;
    desc->notifyNum = notifyNum;
    desc->channelProtocol = link.linkAttr.linkProtocol;
    desc->localEndpoint.protocol = link.srcEndpointDesc.protocol;
    desc->localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
    desc->localEndpoint.loc = link.srcEndpointDesc.loc;
    desc->remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
    desc->remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
    desc->remoteEndpoint.loc = link.dstEndpointDesc.loc;
    found = true;
    return HCCL_SUCCESS;
}

HcclResult AcquireDescByRank(HcclComm comm, uint32_t srcRank, uint32_t dstRank, uint32_t notifyNum,
    HcclChannelDesc *desc, uint32_t &netLayer)
{
    bool found = false;
    netLayer = (srcRank / SERVER_RANK_SIZE == dstRank / SERVER_RANK_SIZE) ? NET_LAYER_INTRA_SERVER :
                                                                            NET_LAYER_INTER_SERVER;
    CHK_RET(TryAcquireDesc(comm, netLayer, srcRank, dstRank, notifyNum, desc, found));
    if (found) {
        return HCCL_SUCCESS;
    }

    uint32_t fallbackLayer = netLayer == NET_LAYER_INTRA_SERVER ? NET_LAYER_INTER_SERVER : NET_LAYER_INTRA_SERVER;
    CHK_RET(TryAcquireDesc(comm, fallbackLayer, srcRank, dstRank, notifyNum, desc, found));
    CHK_PRT_RET(!found, HCCL_ERROR("No link between rank[%u] and rank[%u]", srcRank, dstRank), HCCL_E_INTERNAL);
    netLayer = fallbackLayer;
    return HCCL_SUCCESS;
}

void BuildRemoteRanks(const OpParam &param, std::vector<uint32_t> &remoteRanks)
{
    remoteRanks.clear();
    for (uint32_t rank = 0; rank < param.rankSize; rank++) {
        if (rank != param.myRank) {
            remoteRanks.push_back(rank);
        }
    }
}

HcclResult AcquireChannels(HcclComm comm, const OpParam &param, AlgResourceCtx &resource)
{
    std::vector<uint32_t> remoteRanks;
    BuildRemoteRanks(param, remoteRanks);
    CHK_PRT_RET(remoteRanks.size() > MAX_ALG_CHANNEL_NUM,
        HCCL_ERROR("Too many channels[%zu]", remoteRanks.size()), HCCL_E_INTERNAL);
    std::vector<HcclChannelDesc> descs(remoteRanks.size());
    std::vector<ChannelHandle> handles(remoteRanks.size());
    std::vector<uint32_t> netLayers(remoteRanks.size());
    uint32_t notifyNum = UNIFIED_CHANNEL_NOTIFY_NUM;

    for (uint32_t idx = 0; idx < remoteRanks.size(); idx++) {
        CHK_RET(AcquireDescByRank(
            comm, param.myRank, remoteRanks[idx], notifyNum, &descs[idx], netLayers[idx]));
    }
    CHK_RET(HcclChannelAcquire(
        comm, COMM_ENGINE_AICPU_TS, descs.data(), static_cast<uint32_t>(descs.size()), handles.data()));

    for (uint32_t idx = 0; idx < remoteRanks.size(); idx++) {
        void *remoteBuffer = nullptr;
        uint64_t remoteBufferSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, handles[idx], &remoteBuffer, &remoteBufferSize));
        if (resource.minCclBufferSize == 0 || remoteBufferSize < resource.minCclBufferSize) {
            resource.minCclBufferSize = remoteBufferSize;
        }
        ChannelInfo channel;
        channel.remoteRank = remoteRanks[idx];
        channel.netLayer = netLayers[idx];
        channel.notifyNum = notifyNum;
        channel.handle = handles[idx];
        channel.remoteCclMem = CommBuffer{remoteBuffer, remoteBufferSize};
        resource.channels[resource.channelNum++] = channel;
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
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32, HCCL_ERROR("Only float32 is supported"), HCCL_E_PARA);
    CHK_PRT_RET(op != HCCL_REDUCE_SUM, HCCL_ERROR("Only sum is supported"), HCCL_E_PARA);

    OpParam param;
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = count;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;

    HcclDfxOpInfo dfxInfo;
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize != RANK_SIZE, HCCL_ERROR("Only 16 ranks are supported, rankSize[%u]", param.rankSize),
        HCCL_E_PARA);
    sprintf(param.tag, "%s", UNIFIED_RESOURCE_TAG);

    CommEngine aicpuTsEngine = CommEngine::COMM_ENGINE_AICPU_TS;
    CommEngine cpuTsEngine = CommEngine::COMM_ENGINE_CPU_TS;
    CHK_RET(HcclThreadAcquireWithStream(comm, cpuTsEngine, stream, 1, &param.cpuThread));
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &param.cpuThread, aicpuTsEngine, &param.cpuThreadOnAicpu));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, aicpuTsEngine, &ctx, &size) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = size;

        void *hostCtx = nullptr;
        uint64_t hostCtxSize = 0;
        CHK_RET(HcclEngineCtxGet(comm, param.tag, cpuTsEngine, &hostCtx, &hostCtxSize));
        CHK_PRT_RET(hostCtxSize != sizeof(ThreadHandle), HCCL_ERROR("Invalid cached CPU context size[%llu]",
            static_cast<unsigned long long>(hostCtxSize)), HCCL_E_INTERNAL);
        param.aicpuThreadOnCpu = *static_cast<ThreadHandle *>(hostCtx);
    } else {
        AlgResourceCtx resCtxHost;
        void *cclBufferAddr = nullptr;
        uint64_t cclBufferSize = 0;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};
        resCtxHost.minCclBufferSize = cclBufferSize;

        // Use one resource context for all sizes: one channel per peer and the large-path worker pool. Small data
        // still only touches its four CLOS peers, but avoids creating a second set of peer channels under another tag.
        uint32_t threadNum = UNIFIED_THREAD_NUM;
        uint32_t threadNotifyNum = UNIFIED_THREAD_NOTIFY_NUM;
        CHK_PRT_RET(threadNum > MAX_ALG_THREAD_NUM, HCCL_ERROR("Too many threads[%u]", threadNum), HCCL_E_INTERNAL);
        resCtxHost.threadNum = threadNum;
        CHK_RET(HcclThreadAcquire(
            comm, aicpuTsEngine, threadNum, threadNotifyNum, resCtxHost.threads));
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(
            comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        CHK_RET(AcquireChannels(comm, param, resCtxHost));
        CHK_PRT_RET(resCtxHost.channelNum != UNIFIED_CHANNEL_NUM,
            HCCL_ERROR("Invalid channel count[%u]", resCtxHost.channelNum), HCCL_E_INTERNAL);

        std::vector<char> seq = resCtxHost.Serialize();
        param.ctxSize = seq.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, aicpuTsEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, aicpuTsEngine, param.tag, seq.data(), seq.size(), 0));

        void *hostCtx = nullptr;
        uint64_t hostCtxSize = sizeof(ThreadHandle);
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, cpuTsEngine, hostCtxSize, &hostCtx));
        CHK_RET(HcclEngineCtxCopy(
            comm, cpuTsEngine, param.tag, &param.aicpuThreadOnCpu, hostCtxSize, 0));
    }

    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}
