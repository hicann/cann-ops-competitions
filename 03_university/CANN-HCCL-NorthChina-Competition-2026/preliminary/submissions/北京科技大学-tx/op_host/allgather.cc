/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

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
constexpr uint32_t EXPECTED_RANK_SIZE = 16;
constexpr uint32_t THREAD_NUM = 16;
constexpr uint32_t THREAD_NOTIFY_NUM = 5;
constexpr uint32_t CHANNEL_NUM = 15;
constexpr uint32_t CHANNEL_NOTIFY_NUM = 2;
constexpr char RESOURCE_TAG[] =
    "hccl_custom_allgather_v16_parallel_stage";

HcclResult FillChannelDesc(
    HcclComm comm,
    uint32_t srcRank,
    uint32_t dstRank,
    HcclChannelDesc &desc)
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));

    constexpr CommProtocol REQUIRED_PROTOCOL =
        CommProtocol::COMM_PROTOCOL_UBC_CTP;
    for (uint32_t layerIdx = 0; layerIdx < netLayerNum; ++layerIdx) {
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        CHK_RET(HcclRankGraphGetLinks(
            comm,
            netLayers[layerIdx],
            srcRank,
            dstRank,
            &links,
            &linkNum));

        for (uint32_t linkIdx = 0; linkIdx < linkNum; ++linkIdx) {
            const CommLink &link = links[linkIdx];
            if (link.linkAttr.linkProtocol != REQUIRED_PROTOCOL) {
                continue;
            }

            CHK_RET(HcclChannelDescInit(&desc, 1));
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

    HCCL_ERROR(
        "AllGather found no UBC_CTP link between rank[%u] and rank[%u]",
        srcRank,
        dstRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult AcquireChannels(
    HcclComm comm,
    const OpParam &param,
    AlgResourceCtx &resCtxHost)
{
    std::vector<HcclChannelDesc> descs(CHANNEL_NUM);
    std::vector<ChannelHandle> handles(CHANNEL_NUM);

    uint32_t channelIdx = 0;
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }
        CHK_PRT_RET(channelIdx >= CHANNEL_NUM,
            HCCL_ERROR(
                "AllGather generated too many Channels, rank[%u] index[%u]",
                param.myRank,
                channelIdx),
            HCCL_E_INTERNAL);
        CHK_RET(FillChannelDesc(
            comm,
            param.myRank,
            remoteRank,
            descs[channelIdx]));
        ++channelIdx;
    }

    CHK_PRT_RET(channelIdx != CHANNEL_NUM,
        HCCL_ERROR(
            "AllGather generated Channel count[%u], expected[%u]",
            channelIdx,
            CHANNEL_NUM),
        HCCL_E_INTERNAL);

    CHK_RET(HcclChannelAcquire(
        comm,
        CommEngine::COMM_ENGINE_AICPU_TS,
        descs.data(),
        CHANNEL_NUM,
        handles.data()));

    resCtxHost.channels.reserve(CHANNEL_NUM);
    for (uint32_t idx = 0; idx < CHANNEL_NUM; ++idx) {
        void *remoteCclBuffer = nullptr;
        uint64_t remoteCclBufferSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(
            comm,
            handles[idx],
            &remoteCclBuffer,
            &remoteCclBufferSize));
        CHK_PRT_RET(
            remoteCclBuffer == nullptr || remoteCclBufferSize == 0,
            HCCL_ERROR(
                "AllGather invalid remote CCL buffer, rank[%u] size[%llu]",
                descs[idx].remoteRank,
                static_cast<unsigned long long>(remoteCclBufferSize)),
            HCCL_E_INTERNAL);

        ChannelInfo channel;
        channel.remoteRank = descs[idx].remoteRank;
        channel.notifyNum = CHANNEL_NOTIFY_NUM;
        channel.handle = handles[idx];
        channel.remoteCclMem =
            CommBuffer{remoteCclBuffer, remoteCclBufferSize};
        resCtxHost.channels.push_back(channel);
    }

    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclAllGather(
    void *sendBuf,
    void *recvBuf,
    uint64_t sendCount,
    HcclDataType dataType,
    HcclComm comm,
    aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param{};
    const int tagLength = std::snprintf(
        param.tag,
        sizeof(param.tag),
        "%s",
        RESOURCE_TAG);
    CHK_PRT_RET(
        tagLength < 0 ||
            static_cast<size_t>(tagLength) >= sizeof(param.tag),
        HCCL_ERROR("AllGather failed to construct resource tag"),
        HCCL_E_INTERNAL);
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = sendCount;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(
        commName,
        reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize != EXPECTED_RANK_SIZE,
        HCCL_ERROR(
            "AllGather supports rankSize[%u], actual[%u]",
            EXPECTED_RANK_SIZE,
            param.rankSize),
        HCCL_E_PARA);
    CHK_PRT_RET(param.myRank >= param.rankSize,
        HCCL_ERROR(
            "AllGather invalid rank[%u], rankSize[%u]",
            param.myRank,
            param.rankSize),
        HCCL_E_PARA);
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR(
            "AllGather supports FP32 only, actual type[%d]",
            static_cast<int>(dataType)),
        HCCL_E_PARA);

    const CommEngine aicpuTsEngine =
        CommEngine::COMM_ENGINE_AICPU_TS;
    const CommEngine cpuTsEngine =
        CommEngine::COMM_ENGINE_CPU_TS;

    CHK_RET(HcclThreadAcquireWithStream(
        comm,
        cpuTsEngine,
        stream,
        1,
        &param.cpuThread));
    CHK_RET(HcclThreadExportToCommEngine(
        comm,
        1,
        &param.cpuThread,
        aicpuTsEngine,
        &param.cpuThreadOnAicpu));

    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    if (HcclEngineCtxGet(
            comm,
            param.tag,
            aicpuTsEngine,
            &ctx,
            &ctxSize) == HCCL_SUCCESS) {
        HCCL_INFO(
            "AllGather v16 Parallel Stage EngineCtx already exists");
        param.resCtx = ctx;
        param.ctxSize = ctxSize;

        void *hostCtx = nullptr;
        uint64_t hostCtxSize = 0;
        CHK_RET(HcclEngineCtxGet(
            comm,
            param.tag,
            cpuTsEngine,
            &hostCtx,
            &hostCtxSize));
        CHK_PRT_RET(
            hostCtx == nullptr || hostCtxSize != sizeof(ThreadHandle),
            HCCL_ERROR(
                "AllGather invalid Host EngineCtx size[%llu]",
                static_cast<unsigned long long>(hostCtxSize)),
            HCCL_E_INTERNAL);
        ThreadHandle *aicpuThread =
            static_cast<ThreadHandle *>(hostCtx);
        CHK_RET(HcclThreadExportToCommEngine(
            comm,
            1,
            aicpuThread,
            cpuTsEngine,
            &param.aicpuThreadOnCpu));
    } else {
        AlgResourceCtx resCtxHost;

        void *cclBufferAddr = nullptr;
        uint64_t cclBufferSize = 0;
        CHK_RET(HcclGetHcclBuffer(
            comm,
            &cclBufferAddr,
            &cclBufferSize));
        CHK_PRT_RET(cclBufferAddr == nullptr || cclBufferSize == 0,
            HCCL_ERROR(
                "AllGather invalid local CCL buffer size[%llu]",
                static_cast<unsigned long long>(cclBufferSize)),
            HCCL_E_INTERNAL);
        resCtxHost.localBuffer =
            CommBuffer{cclBufferAddr, cclBufferSize};

        resCtxHost.threads.resize(THREAD_NUM);
        CHK_RET(HcclThreadAcquire(
            comm,
            aicpuTsEngine,
            THREAD_NUM,
            THREAD_NOTIFY_NUM,
            resCtxHost.threads.data()));
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(
            comm,
            1,
            &resCtxHost.aicpuThread,
            cpuTsEngine,
            &param.aicpuThreadOnCpu));

        CHK_RET(AcquireChannels(comm, param, resCtxHost));

        std::vector<char> sequence = resCtxHost.Serialize();
        const uint64_t sequenceSize = sequence.size();
        CHK_PRT_RET(sequenceSize == 0,
            HCCL_ERROR("AllGather serialized an empty resource context"),
            HCCL_E_INTERNAL);
        param.ctxSize = sequenceSize;
        CHK_RET(HcclEngineCtxCreate(
            comm,
            param.tag,
            aicpuTsEngine,
            param.ctxSize,
            &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(
            comm,
            aicpuTsEngine,
            param.tag,
            sequence.data(),
            sequenceSize,
            0));

        void *hostCtx = nullptr;
        const uint64_t hostCtxSize = sizeof(ThreadHandle);
        const void *aicpuThreadPtr =
            static_cast<const void *>(&resCtxHost.aicpuThread);
        CHK_RET(HcclEngineCtxCreate(
            comm,
            param.tag,
            cpuTsEngine,
            hostCtxSize,
            &hostCtx));
        CHK_RET(HcclEngineCtxCopy(
            comm,
            cpuTsEngine,
            param.tag,
            aicpuThreadPtr,
            hostCtxSize,
            0));
    }

    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}
