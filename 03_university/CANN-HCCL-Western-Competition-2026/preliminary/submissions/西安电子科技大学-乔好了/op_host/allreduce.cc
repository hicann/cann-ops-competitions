/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <cstdio>
#include <vector>

#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>

#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"
#include "log.h"

namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;
constexpr uint32_t ALLREDUCE_RANK_SIZE = 16;
constexpr uint32_t ALLREDUCE_CHANNEL_COUNT = 8;
constexpr uint32_t ALLREDUCE_THREAD_COUNT = 8;
constexpr uint32_t ALLREDUCE_CHANNEL_DELTAS[ALLREDUCE_CHANNEL_COUNT] = {1, 2, 4, 8, 3, 5, 6, 7};

HcclResult FindPeerLink(HcclComm comm, uint32_t srcRank, uint32_t dstRank, CommLink *link)
{
    uint32_t *layers = nullptr;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layers, &layerNum));
    for (uint32_t index = 0; index < layerNum; ++index) {
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        HcclResult result = HcclRankGraphGetLinks(comm, layers[index], srcRank, dstRank, &links, &linkNum);
        if (result == HCCL_SUCCESS && linkNum != 0) {
            *link = links[0];
            return HCCL_SUCCESS;
        }
    }
    HCCL_ERROR("FindPeerLink: no link from rank[%u] to rank[%u]", srcRank, dstRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult AcquirePeerChannels(HcclComm comm, const OpParam &param, AlgResourceCtx &resource)
{
    std::vector<HcclChannelDesc> desc(ALLREDUCE_CHANNEL_COUNT);
    std::vector<ChannelHandle> channels(ALLREDUCE_CHANNEL_COUNT);
    for (uint32_t step = 0; step < ALLREDUCE_CHANNEL_COUNT; ++step) {
        const uint32_t peer = param.myRank ^ ALLREDUCE_CHANNEL_DELTAS[step];
        for (uint32_t prior = 0; prior < step; ++prior) {
            CHK_PRT_RET(desc[prior].remoteRank == peer,
                HCCL_ERROR("AcquirePeerChannels: duplicate remote rank[%u]", peer), HCCL_E_INTERNAL);
        }
        CommLink link;
        CHK_RET(FindPeerLink(comm, param.myRank, peer, &link));
        CHK_RET(HcclChannelDescInit(&desc[step], 1));
        desc[step].remoteRank = peer;
        desc[step].notifyNum = CHANNEL_NOTIFY_NUM;
        desc[step].channelProtocol = link.linkAttr.linkProtocol;
        desc[step].localEndpoint = link.srcEndpointDesc;
        desc[step].remoteEndpoint = link.dstEndpointDesc;
    }
    CHK_RET(HcclChannelAcquire(comm, COMM_ENGINE_AICPU_TS, desc.data(), ALLREDUCE_CHANNEL_COUNT, channels.data()));
    for (uint32_t step = 0; step < ALLREDUCE_CHANNEL_COUNT; ++step) {
        void *remoteBuffer = nullptr;
        uint64_t remoteBufferSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, channels[step], &remoteBuffer, &remoteBufferSize));
        resource.channels.push_back(ChannelInfo{desc[step].remoteRank, desc[step].notifyNum, channels[step],
            CommBuffer{remoteBuffer, remoteBufferSize}});
        if (resource.minCclBufferBytes == 0 || remoteBufferSize < resource.minCclBufferBytes) {
            resource.minCclBufferBytes = remoteBufferSize;
        }
    }
    return HCCL_SUCCESS;
}

bool IsHcclVmProxyActive()
{
    const char *preload = std::getenv("LD_PRELOAD");
    return preload != nullptr && std::strstr(preload, "libhccl_proxy_") != nullptr;
}

HcclResult CheckArguments(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32, HCCL_ERROR("HcclAllReduce only supports fp32"), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(op != HCCL_REDUCE_SUM, HCCL_ERROR("HcclAllReduce only supports sum"), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(count > std::numeric_limits<uint64_t>::max() / sizeof(float),
        HCCL_ERROR("HcclAllReduce count overflows byte size"), HCCL_E_PARA);
    const bool checkAlignment = !IsHcclVmProxyActive();
    CHK_PRT_RET(checkAlignment && (reinterpret_cast<uintptr_t>(sendBuf) & (alignof(float) - 1)) != 0,
        HCCL_ERROR("HcclAllReduce sendBuf is not 4-byte aligned"), HCCL_E_PARA);
    CHK_PRT_RET(checkAlignment && (reinterpret_cast<uintptr_t>(recvBuf) & (alignof(float) - 1)) != 0,
        HCCL_ERROR("HcclAllReduce recvBuf is not 4-byte aligned"), HCCL_E_PARA);
    return HCCL_SUCCESS;
}
}

HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    HcclComm comm, aclrtStream stream)
{
    CHK_RET(CheckArguments(sendBuf, recvBuf, count, dataType, op, comm, stream));

    OpParam param;
    sprintf(param.tag, "%s", "hccl_custom_allreduce_fm8_fused_v2");
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = count;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    param.reduceType = op;

    HcclDfxOpInfo dfxInfo;
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));
    const CommEngine aicpuTsEngine = COMM_ENGINE_AICPU_TS;
    const CommEngine cpuTsEngine = COMM_ENGINE_CPU_TS;

    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    if (HcclEngineCtxGet(comm, param.tag, aicpuTsEngine, &ctx, &ctxSize) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = ctxSize;
        void *baseCtxData = nullptr;
        uint64_t baseCtxSize = 0;
        CHK_RET(HcclEngineCtxGet(comm, param.tag, cpuTsEngine, &baseCtxData, &baseCtxSize));
        CHK_PRT_RET(baseCtxSize != sizeof(HostBaseResourceCtx),
            HCCL_ERROR("HcclAllReduce invalid host base context size[%llu]",
                static_cast<unsigned long long>(baseCtxSize)), HCCL_E_INTERNAL);
        const auto *baseCtx = static_cast<const HostBaseResourceCtx *>(baseCtxData);
        param.myRank = baseCtx->myRank;
        param.rankSize = baseCtx->rankSize;
    } else {
        CHK_RET(HcclGetRankId(comm, &param.myRank));
        CHK_RET(HcclGetRankSize(comm, &param.rankSize));
        CHK_PRT_RET(param.rankSize != ALLREDUCE_RANK_SIZE,
            HCCL_ERROR("HcclAllReduce requires 16 ranks, got[%u]", param.rankSize), HCCL_E_NOT_SUPPORT);
        AlgResourceCtx resource;
        void *cclBuffer = nullptr;
        uint64_t cclBufferSize = 0;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBuffer, &cclBufferSize));
        resource.localBuffer = CommBuffer{cclBuffer, cclBufferSize};
        resource.minCclBufferBytes = cclBufferSize;
        resource.threads.resize(ALLREDUCE_THREAD_COUNT);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, ALLREDUCE_THREAD_COUNT, 1, resource.threads.data()));
        resource.aicpuThread = resource.threads[0];
        CHK_RET(AcquirePeerChannels(comm, param, resource));

        param.ctxSize = sizeof(resource);
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, aicpuTsEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, aicpuTsEngine, param.tag, &resource, param.ctxSize, 0));
        const HostBaseResourceCtx baseCtx = {resource.aicpuThread, param.myRank, param.rankSize};
        void *baseCtxData = nullptr;
        const uint64_t baseCtxSize = sizeof(baseCtx);
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, cpuTsEngine, baseCtxSize, &baseCtxData));
        CHK_RET(HcclEngineCtxCopy(comm, cpuTsEngine, param.tag, &baseCtx, baseCtxSize, 0));
    }

    char streamTag[TAG_LENGTH];
    const int tagLength = std::snprintf(streamTag, sizeof(streamTag),
        "hccl_custom_allreduce_fused_stream_v2_%p", reinterpret_cast<void *>(stream));
    CHK_PRT_RET(tagLength < 0 || static_cast<size_t>(tagLength) >= sizeof(streamTag),
        HCCL_ERROR("HcclAllReduce stream tag overflow"), HCCL_E_INTERNAL);
    void *streamCtxData = nullptr;
    uint64_t streamCtxSize = 0;
    if (HcclEngineCtxGet(comm, streamTag, cpuTsEngine, &streamCtxData, &streamCtxSize) == HCCL_SUCCESS) {
        CHK_PRT_RET(streamCtxSize != sizeof(HostStreamResourceCtx),
            HCCL_ERROR("HcclAllReduce invalid host stream context size[%llu]",
                static_cast<unsigned long long>(streamCtxSize)), HCCL_E_INTERNAL);
        const auto *streamCtx = static_cast<const HostStreamResourceCtx *>(streamCtxData);
        param.cpuThread = streamCtx->cpuThread;
        param.cpuThreadOnAicpu = streamCtx->cpuThreadOnAicpu;
        param.aicpuThreadOnCpu = streamCtx->aicpuThreadOnCpu;
    } else {
        HostStreamResourceCtx streamCtx;
        CHK_RET(HcclThreadAcquireWithStream(comm, cpuTsEngine, stream, 3, &streamCtx.cpuThread));
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &streamCtx.cpuThread,
            aicpuTsEngine, &streamCtx.cpuThreadOnAicpu));
        void *baseCtxData = nullptr;
        uint64_t baseCtxSize = 0;
        CHK_RET(HcclEngineCtxGet(comm, param.tag, cpuTsEngine, &baseCtxData, &baseCtxSize));
        CHK_PRT_RET(baseCtxSize != sizeof(HostBaseResourceCtx),
            HCCL_ERROR("HcclAllReduce invalid host base context size[%llu]",
                static_cast<unsigned long long>(baseCtxSize)), HCCL_E_INTERNAL);
        const auto *baseCtx = static_cast<const HostBaseResourceCtx *>(baseCtxData);
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &baseCtx->aicpuThread,
            cpuTsEngine, &streamCtx.aicpuThreadOnCpu));
        param.cpuThread = streamCtx.cpuThread;
        param.cpuThreadOnAicpu = streamCtx.cpuThreadOnAicpu;
        param.aicpuThreadOnCpu = streamCtx.aicpuThreadOnCpu;
        const uint64_t newStreamCtxSize = sizeof(streamCtx);
        CHK_RET(HcclEngineCtxCreate(comm, streamTag, cpuTsEngine, newStreamCtxSize, &streamCtxData));
        CHK_RET(HcclEngineCtxCopy(comm, cpuTsEngine, streamTag, &streamCtx, newStreamCtxSize, 0));
    }
    return ops_hccl::LaunchAICPUKernel(param, stream);
}
