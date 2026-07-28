/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it
 * under the terms and conditions of CANN Open Software License Agreement
 * Version 2.0 (the "License").
 */

#include <cstdio>
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

constexpr uint32_t RANKS_PER_SERVER = 8;
constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;
constexpr uint32_t THREAD_NUM = 16;
constexpr uint32_t THREAD_NOTIFY_NUM = 16;

bool IsSameServer(uint32_t a, uint32_t b)
{
    return a / RANKS_PER_SERVER == b / RANKS_PER_SERVER;
}

bool TryFillDesc(HcclComm comm, uint32_t layer, uint32_t srcRank, uint32_t dstRank, HcclChannelDesc &desc)
{
    CommLink *links = nullptr;
    uint32_t linkNum = 0;

    if (HcclRankGraphGetLinks(comm, layer, srcRank, dstRank, &links, &linkNum) != HCCL_SUCCESS ||
        links == nullptr || linkNum == 0) {
        return false;
    }

    for (uint32_t i = 0; i < linkNum; ++i) {
        CommProtocol protocol = links[i].linkAttr.linkProtocol;

        if (protocol != COMM_PROTOCOL_UBC_CTP && protocol != COMM_PROTOCOL_UBC_TP) {
            continue;
        }

        desc.remoteRank = dstRank;
        desc.notifyNum = CHANNEL_NOTIFY_NUM;
        desc.channelProtocol = protocol;
        desc.localEndpoint = links[i].srcEndpointDesc;
        desc.remoteEndpoint = links[i].dstEndpointDesc;
        return true;
    }

    return false;
}

HcclResult FillDesc(HcclComm comm, uint32_t srcRank, uint32_t dstRank, HcclChannelDesc &desc)
{
    uint32_t firstLayer = IsSameServer(srcRank, dstRank) ? 0U : 1U;
    uint32_t secondLayer = firstLayer ^ 1U;

    if (TryFillDesc(comm, firstLayer, srcRank, dstRank, desc) ||
        TryFillDesc(comm, secondLayer, srcRank, dstRank, desc)) {
        return HCCL_SUCCESS;
    }

    HCCL_ERROR("[Broadcast] no UBC link from rank[%u] to rank[%u]", srcRank, dstRank);
    return HCCL_E_INTERNAL;
}

HcclResult AcquireAllChannels(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtx)
{
    if (param.rankSize <= 1) {
        return HCCL_SUCCESS;
    }

    uint32_t channelNum = param.rankSize - 1;

    std::vector<uint32_t> peerRanks(channelNum);
    std::vector<HcclChannelDesc> descs(channelNum);
    std::vector<ChannelHandle> handles(channelNum);

    CHK_RET(HcclChannelDescInit(descs.data(), channelNum));

    uint32_t index = 0;

    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer == param.myRank) {
            continue;
        }

        peerRanks[index] = peer;
        CHK_RET(FillDesc(comm, param.myRank, peer, descs[index]));
        ++index;
    }

    CHK_RET(HcclChannelAcquire(
        comm,
        COMM_ENGINE_AICPU_TS,
        descs.data(),
        channelNum,
        handles.data()));

    resCtx.channels.resize(channelNum);

    for (uint32_t i = 0; i < channelNum; ++i) {
        void *remoteAddr = nullptr;
        uint64_t remoteSize = 0;

        CHK_RET(HcclChannelGetHcclBuffer(
            comm,
            handles[i],
            &remoteAddr,
            &remoteSize));

        ChannelInfo &channel = resCtx.channels[i];
        channel.remoteRank = peerRanks[i];
        channel.notifyNum = CHANNEL_NOTIFY_NUM;
        channel.handle = handles[i];
        channel.remoteCclMem = CommBuffer{remoteAddr, remoteSize};
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

    (void)snprintf(
        param.tag,
        sizeof(param.tag),
        "hccl_bcast_v11_two_bridge_v8_large");

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

    CHK_PRT_RET(
        param.root >= param.rankSize,
        HCCL_ERROR("invalid root[%u], rankSize[%u]", param.root, param.rankSize),
        HCCL_E_PARA);

    CHK_PRT_RET(
        param.dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("only FP32 is supported"),
        HCCL_E_PARA);

    CHK_RET(HcclThreadAcquireWithStream(
        comm,
        COMM_ENGINE_CPU_TS,
        stream,
        1,
        &param.cpuThread));

    CHK_RET(HcclThreadExportToCommEngine(
        comm,
        1,
        &param.cpuThread,
        COMM_ENGINE_AICPU_TS,
        &param.cpuThreadOnAicpu));

    void *ctx = nullptr;
    uint64_t ctxSize = 0;

    if (HcclEngineCtxGet(
            comm,
            param.tag,
            COMM_ENGINE_AICPU_TS,
            &ctx,
            &ctxSize) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = ctxSize;

        void *hostCtx = nullptr;
        uint64_t hostCtxSize = 0;

        CHK_RET(HcclEngineCtxGet(
            comm,
            param.tag,
            COMM_ENGINE_CPU_TS,
            &hostCtx,
            &hostCtxSize));

        CHK_PRT_RET(
            hostCtx == nullptr || hostCtxSize < sizeof(ThreadHandle),
            HCCL_ERROR("invalid host context"),
            HCCL_E_INTERNAL);

        CHK_RET(HcclThreadExportToCommEngine(
            comm,
            1,
            static_cast<ThreadHandle *>(hostCtx),
            COMM_ENGINE_CPU_TS,
            &param.aicpuThreadOnCpu));
    } else {
        AlgResourceCtx resCtx;

        CHK_RET(HcclGetHcclBuffer(
            comm,
            &resCtx.localBuffer.addr,
            &resCtx.localBuffer.size));

        CHK_PRT_RET(
            resCtx.localBuffer.addr == nullptr || resCtx.localBuffer.size == 0,
            HCCL_ERROR("invalid local HCCL buffer"),
            HCCL_E_INTERNAL);

        resCtx.threads.resize(THREAD_NUM);

        CHK_RET(HcclThreadAcquire(
            comm,
            COMM_ENGINE_AICPU_TS,
            THREAD_NUM,
            THREAD_NOTIFY_NUM,
            resCtx.threads.data()));

        resCtx.aicpuThread = resCtx.threads[0];

        CHK_RET(HcclThreadExportToCommEngine(
            comm,
            1,
            &resCtx.aicpuThread,
            COMM_ENGINE_CPU_TS,
            &param.aicpuThreadOnCpu));

        CHK_RET(AcquireAllChannels(comm, param, resCtx));

        std::vector<char> sequence = resCtx.Serialize();
        param.ctxSize = sequence.size();

        CHK_RET(HcclEngineCtxCreate(
            comm,
            param.tag,
            COMM_ENGINE_AICPU_TS,
            param.ctxSize,
            &param.resCtx));

        CHK_RET(HcclEngineCtxCopy(
            comm,
            COMM_ENGINE_AICPU_TS,
            param.tag,
            sequence.data(),
            param.ctxSize,
            0));

        void *hostCtx = nullptr;

        CHK_RET(HcclEngineCtxCreate(
            comm,
            param.tag,
            COMM_ENGINE_CPU_TS,
            sizeof(ThreadHandle),
            &hostCtx));

        CHK_RET(HcclEngineCtxCopy(
            comm,
            COMM_ENGINE_CPU_TS,
            param.tag,
            &resCtx.aicpuThread,
            sizeof(ThreadHandle),
            0));
    }

    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}
