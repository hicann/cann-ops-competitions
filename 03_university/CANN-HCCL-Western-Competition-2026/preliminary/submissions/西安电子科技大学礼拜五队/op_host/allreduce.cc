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

#include <limits>
#include <vector>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;

int32_t GetProtocolPriority(CommProtocol protocol)
{
    switch (protocol) {
        case COMM_PROTOCOL_UBC_CTP:
            return 0;
        case COMM_PROTOCOL_UBC_TP:
            return 1;
        case COMM_PROTOCOL_UBOE:
            return 2;
        case COMM_PROTOCOL_UB_MEM:
            return 3;
        default:
            return 100;
    }
}

HcclResult CreateChannels(HcclComm comm, CommEngine engine, const OpParam &param, AlgResourceCtx &resCtx)
{
    if (param.rankSize <= 1) {
        return HCCL_SUCCESS;
    }

    uint32_t *layers = nullptr;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layers, &layerNum));
    CHK_PRT_RET(layerNum == 0 || layers == nullptr, HCCL_ERROR("No network layer found"), HCCL_E_INTERNAL);
    std::vector<uint32_t> layerList(layers, layers + layerNum);

    const uint32_t channelNum = param.rankSize - 1;
    std::vector<HcclChannelDesc> channelDescs(channelNum);
    CHK_RET(HcclChannelDescInit(channelDescs.data(), channelNum));
    resCtx.channels.resize(channelNum);

    uint32_t channelIdx = 0;
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }

        bool found = false;
        CommLink selectedLink {};
        int32_t selectedPriority = std::numeric_limits<int32_t>::max();
        uint8_t selectedHop = std::numeric_limits<uint8_t>::max();
        for (uint32_t layer : layerList) {
            CommLink *links = nullptr;
            uint32_t linkNum = 0;
            CHK_RET(HcclRankGraphGetLinks(comm, layer, param.myRank, remoteRank, &links, &linkNum));
            for (uint32_t linkIdx = 0; linkIdx < linkNum; ++linkIdx) {
                const int32_t priority = GetProtocolPriority(links[linkIdx].linkAttr.linkProtocol);
                const uint8_t hop = links[linkIdx].linkAttr.hop;
                if (!found || priority < selectedPriority || (priority == selectedPriority && hop < selectedHop)) {
                    selectedLink = links[linkIdx];
                    selectedPriority = priority;
                    selectedHop = hop;
                    found = true;
                }
            }
        }
        CHK_PRT_RET(!found, HCCL_ERROR("No link from rank[%u] to rank[%u]", param.myRank, remoteRank),
            HCCL_E_INTERNAL);

        HcclChannelDesc &desc = channelDescs[channelIdx];
        desc.remoteRank = remoteRank;
        desc.channelProtocol = selectedLink.linkAttr.linkProtocol;
        desc.localEndpoint = selectedLink.srcEndpointDesc;
        desc.remoteEndpoint = selectedLink.dstEndpointDesc;
        desc.notifyNum = CHANNEL_NOTIFY_NUM;

        resCtx.channels[channelIdx].remoteRank = remoteRank;
        resCtx.channels[channelIdx].notifyNum = CHANNEL_NOTIFY_NUM;
        ++channelIdx;
    }

    std::vector<ChannelHandle> handles(channelNum);
    CHK_RET(HcclChannelAcquire(comm, engine, channelDescs.data(), channelNum, handles.data()));
    for (uint32_t idx = 0; idx < channelNum; ++idx) {
        ChannelInfo &channel = resCtx.channels[idx];
        channel.handle = handles[idx];
        CHK_RET(HcclChannelGetHcclBuffer(
            comm, channel.handle, &channel.remoteCclMem.addr, &channel.remoteCclMem.size));
        CHK_PRT_RET(channel.remoteCclMem.addr == nullptr || channel.remoteCclMem.size == 0,
            HCCL_ERROR("Invalid remote CCL buffer of rank[%u]", channel.remoteRank), HCCL_E_INTERNAL);
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

    OpParam param {};
    (void)snprintf(param.tag, sizeof(param.tag), "%s", "hccl_custom_allreduce");
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = count;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;

    HcclDfxOpInfo dfxInfo {};
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.myRank >= param.rankSize,
        HCCL_ERROR("Invalid rank info: myRank[%u], rankSize[%u]", param.myRank, param.rankSize), HCCL_E_PARA);

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
        ThreadHandle *aicpuThread = static_cast<ThreadHandle *>(hostCtx);
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));
    } else {
        AlgResourceCtx resCtxHost;
        void *cclBufferAddr;
        uint64_t cclBufferSize;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        uint32_t threadNum = param.rankSize > 1 ? param.rankSize - 1 : 1;
        uint32_t notifyNumPerThread = threadNum;

        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, notifyNumPerThread, resCtxHost.threads.data()));
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        CHK_RET(CreateChannels(comm, aicpuTsEngine, param, resCtxHost));

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