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

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

namespace {

const CommLink *SelectAicpuTsLink(const CommLink *links, uint32_t linkNum)
{
    constexpr CommProtocol protocolPriority[] = {CommProtocol::COMM_PROTOCOL_UBC_CTP,
        CommProtocol::COMM_PROTOCOL_UBC_TP, CommProtocol::COMM_PROTOCOL_PCIE, CommProtocol::COMM_PROTOCOL_UBOE};
    for (const auto protocol : protocolPriority) {
        for (uint32_t idx = 0; idx < linkNum; ++idx) {
            if (links[idx].linkAttr.linkProtocol == protocol) {
                return &links[idx];
            }
        }
    }
    return nullptr;
}

HcclResult FillChannelDesc(uint32_t remoteRank, const CommLink &link, HcclChannelDesc &desc)
{
    CHK_RET(HcclChannelDescInit(&desc, 1));
    desc.remoteRank = remoteRank;
    desc.notifyNum = SCATTER_CHANNEL_NOTIFY_NUM;
    desc.channelProtocol = link.linkAttr.linkProtocol;
    desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
    desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
    desc.localEndpoint.loc = link.srcEndpointDesc.loc;
    desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
    desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
    desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
    return HCCL_SUCCESS;
}

HcclResult GetChannelDescOnLayer(
    HcclComm comm, uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, HcclChannelDesc &desc)
{
    CommLink *links = nullptr;
    uint32_t linkNum = 0;
    CHK_RET(HcclRankGraphGetLinks(comm, netLayer, srcRank, dstRank, &links, &linkNum));
    CHK_PRT_RET(linkNum == 0,
        HCCL_ERROR("No link on layer[%u] between rank[%u] and rank[%u]", netLayer, srcRank, dstRank),
        HCCL_E_NOT_FOUND);
    const CommLink *selectedLink = SelectAicpuTsLink(links, linkNum);
    CHK_PRT_RET(selectedLink == nullptr,
        HCCL_ERROR("No AICPU_TS link on layer[%u] between rank[%u] and rank[%u]", netLayer, srcRank, dstRank),
        HCCL_E_NOT_FOUND);
    return FillChannelDesc(dstRank, *selectedLink, desc);
}

HcclResult GetInterServerChannelDesc(HcclComm comm, uint32_t srcRank, uint32_t dstRank, HcclChannelDesc &desc)
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    for (uint32_t idx = 0; idx < netLayerNum; ++idx) {
        if (netLayers[idx] == 0) {
            continue;
        }
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayers[idx], srcRank, dstRank, &links, &linkNum));
        if (linkNum != 0) {
            const CommLink *selectedLink = SelectAicpuTsLink(links, linkNum);
            if (selectedLink != nullptr) {
                return FillChannelDesc(dstRank, *selectedLink, desc);
            }
        }
    }
    HCCL_ERROR("No inter-server link between rank[%u] and rank[%u]", srcRank, dstRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult AcquireScatterChannels(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtxHost)
{
    const uint32_t serverIdx = param.myRank / SCATTER_RANKS_PER_SERVER;
    const uint32_t serverBase = serverIdx * SCATTER_RANKS_PER_SERVER;
    std::array<HcclChannelDesc, SCATTER_CHANNEL_COUNT> descs{};
    uint32_t channelIdx = 0;

    // CLOS is a fabric: every cross-server rank pair has a layer-1 link. Keep
    // one channel per peer so the root can deliver remote suffixes directly
    // instead of concentrating them at a same-local-id relay.
    for (uint32_t remoteRank = 0; remoteRank < SCATTER_EXPECTED_RANK_SIZE; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }
        if (remoteRank >= serverBase && remoteRank < serverBase + SCATTER_RANKS_PER_SERVER) {
            CHK_RET(GetChannelDescOnLayer(comm, 0, param.myRank, remoteRank, descs[channelIdx]));
        } else {
            CHK_RET(GetInterServerChannelDesc(comm, param.myRank, remoteRank, descs[channelIdx]));
        }
        ++channelIdx;
    }

    CHK_PRT_RET(channelIdx != SCATTER_CHANNEL_COUNT,
        HCCL_ERROR("Unexpected channel descriptor count[%u], expected[%u]", channelIdx, SCATTER_CHANNEL_COUNT),
        HCCL_E_INTERNAL);
    std::array<ChannelHandle, SCATTER_CHANNEL_COUNT> channelHandles{};
    CHK_RET(HcclChannelAcquire(
        comm, CommEngine::COMM_ENGINE_AICPU_TS, descs.data(), SCATTER_CHANNEL_COUNT, channelHandles.data()));

    for (uint32_t idx = 0; idx < SCATTER_CHANNEL_COUNT; ++idx) {
        void *remoteBuffer = nullptr;
        uint64_t remoteBufferSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, channelHandles[idx], &remoteBuffer, &remoteBufferSize));
        CHK_PRT_RET(remoteBuffer == nullptr || remoteBufferSize < SCATTER_REQUIRED_CCL_BYTES,
            HCCL_ERROR("Remote HCCL buffer for rank[%u] is too small, size[%llu]", descs[idx].remoteRank,
                static_cast<unsigned long long>(remoteBufferSize)),
            HCCL_E_INTERNAL);
        ChannelInfo &channel = resCtxHost.channels[descs[idx].remoteRank];
        channel.remoteRank = descs[idx].remoteRank;
        channel.notifyNum = SCATTER_CHANNEL_NOTIFY_NUM;
        channel.handle = channelHandles[idx];
        channel.remoteCclMem = CommBuffer{remoteBuffer, remoteBufferSize};
    }
    return HCCL_SUCCESS;
}

HcclResult ValidateInvocation(const OpParam &param)
{
    CHK_PRT_RET(param.rankSize != SCATTER_EXPECTED_RANK_SIZE,
        HCCL_ERROR("Scatter direct hybrid requires rankSize[%u], actual[%u]",
            SCATTER_EXPECTED_RANK_SIZE, param.rankSize),
        HCCL_E_PARA);
    CHK_PRT_RET(param.myRank >= param.rankSize || param.root >= param.rankSize,
        HCCL_ERROR("Invalid ranks, myRank[%u] root[%u] rankSize[%u]", param.myRank, param.root, param.rankSize),
        HCCL_E_PARA);
    CHK_PRT_RET(param.dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("Scatter direct hybrid only supports FP32, dataType[%d]", static_cast<int>(param.dataType)),
        HCCL_E_PARA);
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / sizeof(float),
        HCCL_ERROR("recvCount[%llu] overflows byte size", static_cast<unsigned long long>(param.count)), HCCL_E_PARA);
    const uint64_t recvBytes = param.count * sizeof(float);
    CHK_PRT_RET(recvBytes > std::numeric_limits<uint64_t>::max() / param.rankSize,
        HCCL_ERROR("Scatter input size overflows, recvBytes[%llu] rankSize[%u]",
            static_cast<unsigned long long>(recvBytes), param.rankSize),
        HCCL_E_PARA);
    return HCCL_SUCCESS;
}

} // namespace

HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    // 构造算子参数
    OpParam param;
    sprintf(param.tag, "%s", "hccl_custom_scatter_v6");
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.root = root;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_SCATTER;

    // 注册算子信息
    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    // ==============================================
    // STEP 2: 创建资源
    // ==============================================
    CommEngine aicpuTsEngine = CommEngine::COMM_ENGINE_AICPU_TS;
    CommEngine cpuTsEngine = CommEngine::COMM_ENGINE_CPU_TS;

    // ==============================================
    // STEP 2.1: 申请用于 Host/Device 同步的通信资源
    // ==============================================
    // 将用户传入的 stream 转换为 thread，并申请 Notify；同时导出为 AICPU 上可用的 thread
    CHK_RET(HcclThreadAcquireWithStream(comm, cpuTsEngine, stream, 1, &param.cpuThread));
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &param.cpuThread, aicpuTsEngine, &param.cpuThreadOnAicpu));

    void *hostCtx = nullptr;
    uint64_t hostCtxSize = 0;
    if (HcclEngineCtxGet(comm, param.tag, cpuTsEngine, &hostCtx, &hostCtxSize) == HCCL_SUCCESS) {
        CHK_PRT_RET(hostCtx == nullptr || hostCtxSize != sizeof(HostResourceCtx),
            HCCL_ERROR("Invalid host resource context, addr[%p] size[%llu]", hostCtx,
                static_cast<unsigned long long>(hostCtxSize)),
            HCCL_E_INTERNAL);
        const HostResourceCtx &hostResource = *static_cast<const HostResourceCtx *>(hostCtx);
        CHK_PRT_RET(hostResource.magic != SCATTER_HOST_RESOURCE_MAGIC ||
                        hostResource.version != SCATTER_RESOURCE_VERSION || hostResource.deviceResCtx == nullptr ||
                        hostResource.deviceResCtxSize != sizeof(AlgResourceCtx),
            HCCL_ERROR("Invalid cached scatter resource context"), HCCL_E_INTERNAL);
        param.myRank = hostResource.myRank;
        param.rankSize = hostResource.rankSize;
        param.resCtx = hostResource.deviceResCtx;
        param.ctxSize = hostResource.deviceResCtxSize;
        param.aicpuThreadOnCpu = hostResource.aicpuThreadOnCpu;
        CHK_RET(ValidateInvocation(param));
    } else {
        // Device 资源不存在，资源构建
        CHK_RET(HcclGetRankId(comm, &param.myRank));
        CHK_RET(HcclGetRankSize(comm, &param.rankSize));
        CHK_RET(ValidateInvocation(param));
        AlgResourceCtx resCtxHost;

        // 从通信域获取 HCCL Buffer（Device上的内存，默认总大小400MB）
        void *cclBufferAddr;
        uint64_t cclBufferSize;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        CHK_PRT_RET(cclBufferAddr == nullptr || cclBufferSize < SCATTER_REQUIRED_CCL_BYTES,
            HCCL_ERROR("Local HCCL buffer is too small, size[%llu]", static_cast<unsigned long long>(cclBufferSize)),
            HCCL_E_INTERNAL);
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        // ==============================================
        // STEP 2.2: 申请资源Thread和Channel
        // ==============================================

        // threads[0] is the control/Host-sync and root-CLOS thread. The seven
        // workers map one-to-one to root-local Mesh peers. A source helper or
        // a remote target uses threads[1] for its independent second stream.
        uint32_t threadNum = SCATTER_THREAD_COUNT;
        uint32_t notifyNumPerThread = SCATTER_MESH_WORKER_COUNT;

        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, notifyNumPerThread, resCtxHost.threads.data()));
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        // One channel is acquired for every peer. This is still exactly one
        // channel per peer and permits direct CLOS delivery to remote targets.
        CHK_RET(AcquireScatterChannels(comm, param, resCtxHost));

        // ==============================================
        // STEP 2.3: 申请通信引擎上下文
        // ==============================================
        // 申请 AICPU 通信引擎上下文，存放 AlgResourceCtx 信息
        param.ctxSize = sizeof(resCtxHost);
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, aicpuTsEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, aicpuTsEngine, param.tag, &resCtxHost, param.ctxSize, 0));
        // CPU context is directly readable on the Host. Cache immutable rank,
        // Device-context and exported-thread metadata for the warm path.
        HostResourceCtx hostResource;
        hostResource.deviceResCtx = param.resCtx;
        hostResource.deviceResCtxSize = param.ctxSize;
        hostResource.aicpuThreadOnCpu = param.aicpuThreadOnCpu;
        hostResource.myRank = param.myRank;
        hostResource.rankSize = param.rankSize;
        hostCtxSize = sizeof(hostResource);
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, cpuTsEngine, hostCtxSize, &hostCtx));
        CHK_RET(HcclEngineCtxCopy(comm, cpuTsEngine, param.tag, &hostResource, hostCtxSize, 0));
    }

    // ==============================================
    // STEP 3: 下发 AICPU Kernel
    // ==============================================
    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}
