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

namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 2;
constexpr uint32_t RANKS_PER_SERVER = 8;
constexpr uint32_t COMPETITION_RANK_SIZE = 16;

std::vector<uint32_t> GetChannelPeers(const OpParam &param)
{
    std::vector<uint32_t> peers;

    if (param.rankSize == COMPETITION_RANK_SIZE) {
        const uint32_t serverBase =
            (param.myRank / RANKS_PER_SERVER) * RANKS_PER_SERVER;

        // 申请本服务器内的Full-Mesh Channel。
        for (uint32_t rank = serverBase;
             rank < serverBase + RANKS_PER_SERVER;
             ++rank) {
            if (rank != param.myRank) {
                peers.push_back(rank);
            }
        }

        // 申请连接另一台服务器对应NPU的Clos Channel。
        peers.push_back(param.myRank ^ RANKS_PER_SERVER);
        return peers;
    }

    // 非赛题拓扑的通用回退。
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (rank != param.myRank) {
            peers.push_back(rank);
        }
    }

    return peers;
}

HcclResult FillChannelDesc(
    HcclComm comm,
    uint32_t srcRank,
    uint32_t dstRank,
    HcclChannelDesc &desc)
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));

    CommLink *selectedLink = nullptr;

    // 服务器内Mesh链路和服务器间Clos链路可能位于不同网络层，
    // 因此不能硬编码netLayer=0。
    for (uint32_t i = 0; i < netLayerNum; ++i) {
        CommLink *links = nullptr;
        uint32_t linkNum = 0;

        CHK_RET(HcclRankGraphGetLinks(
            comm,
            netLayers[i],
            srcRank,
            dstRank,
            &links,
            &linkNum));

        if (linkNum > 0) {
            selectedLink = &links[0];
            break;
        }
    }

    CHK_PRT_RET(
        selectedLink == nullptr,
        HCCL_ERROR(
            "No direct link between rank[%u] and rank[%u]",
            srcRank,
            dstRank),
        HCCL_E_INTERNAL);

    desc.remoteRank = dstRank;
    desc.notifyNum = CHANNEL_NOTIFY_NUM;
    desc.channelProtocol = selectedLink->linkAttr.linkProtocol;

    desc.localEndpoint.protocol =
        selectedLink->srcEndpointDesc.protocol;
    desc.localEndpoint.commAddr =
        selectedLink->srcEndpointDesc.commAddr;
    desc.localEndpoint.loc =
        selectedLink->srcEndpointDesc.loc;

    desc.remoteEndpoint.protocol =
        selectedLink->dstEndpointDesc.protocol;
    desc.remoteEndpoint.commAddr =
        selectedLink->dstEndpointDesc.commAddr;
    desc.remoteEndpoint.loc =
        selectedLink->dstEndpointDesc.loc;

    return HCCL_SUCCESS;
}

HcclResult AcquireChannels(
    HcclComm comm,
    const OpParam &param,
    AlgResourceCtx &resCtxHost)
{
    const std::vector<uint32_t> peers = GetChannelPeers(param);

    if (peers.empty()) {
        return HCCL_SUCCESS;
    }

    const uint32_t channelNum =
        static_cast<uint32_t>(peers.size());

    std::vector<HcclChannelDesc> descs(channelNum);
    std::vector<ChannelHandle> handles(channelNum);

    CHK_RET(HcclChannelDescInit(descs.data(), channelNum));

    for (uint32_t i = 0; i < channelNum; ++i) {
        CHK_RET(FillChannelDesc(
            comm,
            param.myRank,
            peers[i],
            descs[i]));
    }

    CHK_RET(HcclChannelAcquire(
        comm,
        CommEngine::COMM_ENGINE_AICPU_TS,
        descs.data(),
        channelNum,
        handles.data()));

    resCtxHost.channels.reserve(channelNum);

    for (uint32_t i = 0; i < channelNum; ++i) {
        void *remoteBuffer = nullptr;
        uint64_t remoteBufferSize = 0;

        CHK_RET(HcclChannelGetHcclBuffer(
            comm,
            handles[i],
            &remoteBuffer,
            &remoteBufferSize));

        ChannelInfo channel;
        channel.remoteRank = peers[i];
        channel.notifyNum = CHANNEL_NOTIFY_NUM;
        channel.handle = handles[i];
        channel.remoteCclMem =
            CommBuffer{remoteBuffer, remoteBufferSize};

        resCtxHost.channels.push_back(channel);
    }

    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclBroadcast(
    void *buf,
    uint64_t count,
    HcclDataType dataType,
    uint32_t root,
    HcclComm comm,
    aclrtStream stream)
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
    CHK_RET(HcclDfxRegOpInfoByCommId(
        commName,
        reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));

    CHK_PRT_RET(
        root >= param.rankSize,
        HCCL_ERROR(
            "Invalid root[%u], rankSize[%u]",
            root,
            param.rankSize),
        HCCL_E_PARA);

    CommEngine aicpuTsEngine =
        CommEngine::COMM_ENGINE_AICPU_TS;
    CommEngine cpuTsEngine =
        CommEngine::COMM_ENGINE_CPU_TS;

    // 用户Stream与AICPU线程同步。
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
    uint64_t size = 0;

    if (HcclEngineCtxGet(
            comm,
            param.tag,
            aicpuTsEngine,
            &ctx,
            &size) == HCCL_SUCCESS) {
        HCCL_INFO("Engine context already exists");

        param.resCtx = ctx;
        param.ctxSize = size;

        void *hostCtx = nullptr;
        uint64_t hostCtxSize = 0;

        CHK_RET(HcclEngineCtxGet(
            comm,
            param.tag,
            cpuTsEngine,
            &hostCtx,
            &hostCtxSize));

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

        resCtxHost.localBuffer =
            CommBuffer{cclBufferAddr, cclBufferSize};

        constexpr uint32_t threadNum = 1;
        constexpr uint32_t notifyNumPerThread = 1;

        resCtxHost.threads.resize(threadNum);

        CHK_RET(HcclThreadAcquire(
            comm,
            aicpuTsEngine,
            threadNum,
            notifyNumPerThread,
            resCtxHost.threads.data()));

        resCtxHost.aicpuThread =
            resCtxHost.threads[0];

        CHK_RET(HcclThreadExportToCommEngine(
            comm,
            1,
            &resCtxHost.aicpuThread,
            cpuTsEngine,
            &param.aicpuThreadOnCpu));

        CHK_RET(AcquireChannels(
            comm,
            param,
            resCtxHost));

        std::vector<char> seq =
            resCtxHost.Serialize();

        uint64_t seqSize = seq.size();
        param.ctxSize = seqSize;

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
            seq.data(),
            seqSize,
            0));

        void *hostCtx = nullptr;
        uint64_t hostCtxSize = sizeof(ThreadHandle);

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