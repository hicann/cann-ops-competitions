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
#if __has_include(<hccl/hccl_diag.h>)
#include <hccl/hccl_diag.h>
#define SCATTER_HAS_DFX_API 1
#else
#define SCATTER_HAS_DFX_API 0
#endif
#include <vector>
#include "log.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

namespace {
HcclResult RegisterScatterDfx(HcclComm comm, const OpParam &param)
{
#if SCATTER_HAS_DFX_API
    HcclDfxOpInfo info{};
    info.opMode = 1; 
    info.opType = static_cast<uint32_t>(param.opType);
    info.reduceOp = static_cast<uint32_t>(param.reduceType);
    info.dataType = static_cast<uint32_t>(param.dataType);
    info.outputType = static_cast<uint32_t>(param.dataType);
    info.dataCount = param.count;
    info.root = param.root;
    info.engine = COMM_ENGINE_AICPU_TS;
    info.cpuTsThread = param.cpuThread;
    info.cpuWaitAicpuNotifyIdx = 0;
    info.cpuWaitAicpuNotifyId = INVALID_VALUE_RANKID;
    const int len = snprintf(info.algTag, sizeof(info.algTag), "%s", param.tag);
    if (len < 0 || static_cast<size_t>(len) >= sizeof(info.algTag)) {
        return HCCL_E_INTERNAL;
    }
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
    CHK_RET(HcclGetCommName(comm, commName));
    const HcclResult ret = HcclDfxRegOpInfoByCommId(commName, &info);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("Scatter DFX registration failed: rank=%u root=%u ret=%d", param.myRank, param.root, ret);
    }
    return ret;
#else
    (void)comm;
    (void)param;
    HCCL_ERROR("Scatter requires the contest CANN SDK with hccl/hccl_diag.h and HcclDfxRegOpInfoByCommId");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult GetLayers(HcclComm comm, std::vector<uint32_t> &layers)
{
    uint32_t *ptr = nullptr;
    uint32_t count = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &ptr, &count));
    if (ptr == nullptr || count == 0) {
        return HCCL_E_NOT_SUPPORT;
    }
    layers.assign(ptr, ptr + count);
    std::sort(layers.begin(), layers.end());
    return HCCL_SUCCESS;
}

HcclResult GetLocalRanks(HcclComm comm, const std::vector<uint32_t> &layers, AlgResourceCtx &ctx, uint32_t &meshLayer)
{
    for (uint32_t layer : layers) {
        uint32_t *instances = nullptr;
        uint32_t count = 0;
        if (HcclRankGraphGetTopoInstsByLayer(comm, layer, &instances, &count) != HCCL_SUCCESS || instances == nullptr) {
            continue;
        }
        std::vector<uint32_t> ids(instances, instances + count);
        for (uint32_t id : ids) {
            CommTopo type = COMM_TOPO_RESERVED;
            CHK_RET(HcclRankGraphGetTopoType(comm, layer, id, &type));
            if (type != COMM_TOPO_1DMESH) {
                continue;
            }
            uint32_t *ranks = nullptr;
            uint32_t rankNum = 0;
            CHK_RET(HcclRankGraphGetRanksByTopoInst(comm, layer, id, &ranks, &rankNum));
            if (ranks == nullptr || rankNum != SCATTER_WIDTH) {
                continue;
            }
            std::copy(ranks, ranks + rankNum, ctx.localRanks.begin());
            std::sort(ctx.localRanks.begin(), ctx.localRanks.end());
            if (std::find(ctx.localRanks.begin(), ctx.localRanks.end(), ctx.myRank) == ctx.localRanks.end()) {
                continue;
            }
            meshLayer = layer;
            return HCCL_SUCCESS;
        }
    }
    return HCCL_E_NOT_SUPPORT;
}

HcclResult FillChannel(
    HcclComm comm, uint32_t self, uint32_t peer, const std::vector<uint32_t> &layers, HcclChannelDesc &desc)
{
    CHK_RET(HcclChannelDescInit(&desc, 1));
    for (uint32_t layer : layers) {
        CommLink *links = nullptr;
        uint32_t count = 0;
        if (HcclRankGraphGetLinks(comm, layer, self, peer, &links, &count) != HCCL_SUCCESS || links == nullptr) {
            continue;
        }
        for (uint32_t i = 0; i < count; ++i) {
            if (links[i].linkAttr.linkProtocol != COMM_PROTOCOL_UBC_CTP) {
                continue;
            }
            desc.remoteRank = peer;
            desc.notifyNum = 2;
            desc.channelProtocol = links[i].linkAttr.linkProtocol;
            desc.localEndpoint = links[i].srcEndpointDesc;
            desc.remoteEndpoint = links[i].dstEndpointDesc;
            return HCCL_SUCCESS;
        }
    }
    HCCL_ERROR("No CTP link from rank %u to %u", self, peer);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult CreateResources(HcclComm comm, const OpParam &param, AlgResourceCtx &ctx)
{
    ctx.rankSize = param.rankSize;
    ctx.myRank = param.myRank;
    CHK_RET(HcclGetHcclBuffer(comm, &ctx.localBuffer.addr, &ctx.localBuffer.size));
    CHK_PTR_NULL(ctx.localBuffer.addr);
    if (param.rankSize == 1) {
        CHK_RET(HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 1, 1, ctx.threads.data()));
        ctx.aicpuThread = ctx.threads[0];
        return HCCL_SUCCESS;
    }
    std::vector<uint32_t> layers;
    CHK_RET(GetLayers(comm, layers));
    uint32_t meshLayer = 0;
    CHK_RET(GetLocalRanks(comm, layers, ctx, meshLayer));
    std::array<bool, SCATTER_RANKS> present{};
    for (uint32_t rank : ctx.localRanks) {
        if (rank >= SCATTER_RANKS || present[rank]) {
            return HCCL_E_PARA;
        }
        present[rank] = true;
    }
    uint32_t remoteIndex = 0;
    for (uint32_t rank = 0; rank < SCATTER_RANKS; ++rank) {
        if (!present[rank]) {
            ctx.remoteRanks[remoteIndex++] = rank;
        }
    }
    ctx.localIndex = static_cast<uint32_t>(
        std::find(ctx.localRanks.begin(), ctx.localRanks.end(), ctx.myRank) - ctx.localRanks.begin());
    std::array<HcclChannelDesc, SCATTER_CHANNELS> descs{};
    std::array<ChannelHandle, SCATTER_CHANNELS> handles{};
    uint32_t idx = 0;
    for (uint32_t peer = 0; peer < SCATTER_RANKS; ++peer) {
        if (peer != ctx.myRank) {
            const std::vector<uint32_t> peerLayers = present[peer] ? std::vector<uint32_t>{meshLayer} : layers;
            CHK_RET(FillChannel(comm, ctx.myRank, peer, peerLayers, descs[idx++]));
        }
    }
    CHK_RET(HcclChannelAcquire(comm, COMM_ENGINE_AICPU_TS, descs.data(), SCATTER_CHANNELS, handles.data()));
    for (uint32_t i = 0; i < SCATTER_CHANNELS; ++i) {
        auto &channel = ctx.channels[i];
        channel.remoteRank = descs[i].remoteRank;
        channel.handle = handles[i];
        CHK_RET(HcclChannelGetHcclBuffer(comm, handles[i], &channel.remoteCclMem.addr, &channel.remoteCclMem.size));
        CHK_PTR_NULL(channel.remoteCclMem.addr);
        if (channel.remoteCclMem.size != ctx.localBuffer.size) {
            HCCL_ERROR("Unequal HCCL buffer capacities on rank %u and %u", ctx.myRank, channel.remoteRank);
            return HCCL_E_NOT_SUPPORT;
        }
    }
    if (ctx.localBuffer.size < sizeof(float)) {
        return HCCL_E_MEMORY;
    }
    CHK_RET(HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, SCATTER_THREADS, SCATTER_THREADS, ctx.threads.data()));
    ctx.aicpuThread = ctx.threads[0];
    return HCCL_SUCCESS;
}

HcclResult InitContext(HcclComm comm, OpParam &param)
{
    void *device = nullptr;
    uint64_t deviceSize = 0;
    if (HcclEngineCtxGet(comm, param.tag, COMM_ENGINE_AICPU_TS, &device, &deviceSize) == HCCL_SUCCESS) {
        if (device == nullptr || deviceSize != sizeof(AlgResourceCtx)) {
            return HCCL_E_INTERNAL;
        }
        void *host = nullptr;
        uint64_t hostSize = 0;
        CHK_RET(HcclEngineCtxGet(comm, param.tag, COMM_ENGINE_CPU_TS, &host, &hostSize));
        if (host == nullptr || hostSize != sizeof(ThreadHandle)) {
            return HCCL_E_INTERNAL;
        }
        CHK_RET(HcclThreadExportToCommEngine(
            comm, 1, static_cast<ThreadHandle *>(host), COMM_ENGINE_CPU_TS, &param.aicpuThreadOnCpu));
        param.resCtx = device;
        param.ctxSize = deviceSize;
        return HCCL_SUCCESS;
    }
    AlgResourceCtx ctx{};
    CHK_RET(CreateResources(comm, param, ctx));
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &ctx.aicpuThread, COMM_ENGINE_CPU_TS, &param.aicpuThreadOnCpu));
    void *host = nullptr;
    CHK_RET(HcclEngineCtxCreate(comm, param.tag, COMM_ENGINE_CPU_TS, sizeof(ThreadHandle), &host));
    CHK_PTR_NULL(host);
    if (memcpy_s(host, sizeof(ThreadHandle), &ctx.aicpuThread, sizeof(ThreadHandle)) != EOK) {
        return HCCL_E_INTERNAL;
    }
    param.ctxSize = sizeof(ctx);
    CHK_RET(HcclEngineCtxCreate(comm, param.tag, COMM_ENGINE_AICPU_TS, param.ctxSize, &param.resCtx));
    CHK_RET(HcclEngineCtxCopy(comm, COMM_ENGINE_AICPU_TS, param.tag, &ctx, sizeof(ctx), 0));
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    OpParam param{};
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    if (param.rankSize == 0 || root >= param.rankSize || param.myRank >= param.rankSize) {
        return HCCL_E_PARA;
    }
    if (dataType != HCCL_DATA_TYPE_FP32 || (param.rankSize != 1 && param.rankSize != SCATTER_RANKS)) {
        return HCCL_E_NOT_SUPPORT;
    }
    if (recvCount > std::numeric_limits<uint64_t>::max() / sizeof(float) / param.rankSize) {
        return HCCL_E_PARA;
    }
    if (recvCount == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(recvBuf);
    if (param.myRank == root) {
        CHK_PTR_NULL(sendBuf);
    }
    if (snprintf(param.tag, sizeof(param.tag), "%s", "hccl_scatter_gpt_v5") < 0) {
        return HCCL_E_INTERNAL;
    }
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.root = root;
    param.dataType = dataType;
    param.opType = HCCL_CMD_SCATTER;
    CHK_RET(HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, stream, 1, &param.cpuThread));
    CHK_RET(RegisterScatterDfx(comm, param));
    CHK_RET(ops_hccl::PrepareAICPUKernel());
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &param.cpuThread, COMM_ENGINE_AICPU_TS, &param.cpuThreadOnAicpu));
    CHK_RET(InitContext(comm, param));
    return ops_hccl::LaunchAICPUKernel(param, stream);
}
