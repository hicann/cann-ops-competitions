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
#include <cstdio>
#include <limits>
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
constexpr uint32_t THREAD_NOTIFY_NUM = 1;
constexpr uint32_t CHANNEL_NOTIFY_NUM = 2;
constexpr uint32_t HOST_DEVICE_NOTIFY_IDX = 0;
constexpr uint64_t FP32_DATA_TYPE_SIZE = sizeof(float);
constexpr char BROADCAST_TAG[] = "hccl_custom_broadcast";
constexpr char BROADCAST_LARGE_TAG[] = "hccl_custom_broadcast_flat16_dedicated_copy";

static_assert(sizeof(float) == 4, "HCCL FP32 must contain four bytes");

const CommProtocol AICPU_TS_PROTOCOLS[] = {
    CommProtocol::COMM_PROTOCOL_UBC_CTP,
    CommProtocol::COMM_PROTOCOL_UBC_TP,
    CommProtocol::COMM_PROTOCOL_PCIE,
    CommProtocol::COMM_PROTOCOL_UBOE,
};

HcclResult FillChannelDesc(HcclComm comm, uint32_t srcRank, uint32_t dstRank,
    const std::vector<uint32_t> &netLayers, HcclChannelDesc &desc, uint32_t &selectedNetLayer)
{
    CHK_RET(HcclChannelDescInit(&desc, 1));
    selectedNetLayer = INVALID_VALUE_RANKID;

    for (uint32_t netLayer : netLayers) {
        CommLink *linkList = nullptr;
        uint32_t linkNum = 0;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayer, srcRank, dstRank, &linkList, &linkNum));
        CHK_PRT_RET(linkNum > 0 && linkList == nullptr,
            HCCL_ERROR("[FillChannelDesc] link list is null, srcRank[%u], dstRank[%u], netLayer[%u]", srcRank,
                dstRank, netLayer),
            HCCL_E_INTERNAL);

        for (CommProtocol expectedProtocol : AICPU_TS_PROTOCOLS) {
            for (uint32_t linkIdx = 0; linkIdx < linkNum; ++linkIdx) {
                const CommLink &link = linkList[linkIdx];
                if (link.linkAttr.linkProtocol != expectedProtocol) {
                    continue;
                }

                desc.remoteRank = dstRank;
                desc.channelProtocol = link.linkAttr.linkProtocol;
                desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
                desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
                desc.localEndpoint.loc = link.srcEndpointDesc.loc;
                desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
                desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
                desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
                desc.notifyNum = CHANNEL_NOTIFY_NUM;
                selectedNetLayer = netLayer;
                HCCL_INFO("[FillChannelDesc] channel found, srcRank[%u], dstRank[%u], netLayer[%u], protocol[%d]",
                    srcRank, dstRank, netLayer, static_cast<int32_t>(expectedProtocol));
                return HCCL_SUCCESS;
            }
        }
    }

    HCCL_ERROR("[FillChannelDesc] no AICPU_TS link found, srcRank[%u], dstRank[%u]", srcRank, dstRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult CreateAllPeerChannels(
    HcclComm comm, const OpParam &param, CommEngine engine, AlgResourceCtx &resCtxHost)
{
    if (param.rankSize == 1) {
        return HCCL_SUCCESS;
    }

    uint32_t *netLayerList = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayerList, &netLayerNum));
    CHK_PRT_RET(netLayerNum == 0 || netLayerList == nullptr,
        HCCL_ERROR("[CreateAllPeerChannels] invalid network layers, netLayerNum[%u]", netLayerNum),
        HCCL_E_NOT_FOUND);
    const std::vector<uint32_t> netLayers(netLayerList, netLayerList + netLayerNum);

    const uint32_t channelNum = param.rankSize - 1;
    std::vector<HcclChannelDesc> channelDescs(channelNum);
    std::vector<ChannelHandle> channelHandles(channelNum);
    std::vector<uint32_t> channelNetLayers(channelNum, INVALID_VALUE_RANKID);

    uint32_t channelIdx = 0;
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank == param.myRank) {
            continue;
        }
        CHK_RET(FillChannelDesc(
            comm, param.myRank, remoteRank, netLayers, channelDescs[channelIdx], channelNetLayers[channelIdx]));
        ++channelIdx;
    }
    CHK_PRT_RET(channelIdx != channelNum,
        HCCL_ERROR("[CreateAllPeerChannels] channel count mismatch, expected[%u], actual[%u]", channelNum, channelIdx),
        HCCL_E_INTERNAL);

    CHK_RET(HcclChannelAcquire(comm, engine, channelDescs.data(), channelNum, channelHandles.data()));
    resCtxHost.channels.resize(channelNum);
    for (uint32_t idx = 0; idx < channelNum; ++idx) {
        void *remoteBufferAddr = nullptr;
        uint64_t remoteBufferSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, channelHandles[idx], &remoteBufferAddr, &remoteBufferSize));
        CHK_PRT_RET(remoteBufferAddr == nullptr || remoteBufferSize == 0,
            HCCL_ERROR("[CreateAllPeerChannels] invalid remote HCCL buffer, remoteRank[%u], addr[%p], size[%llu]",
                channelDescs[idx].remoteRank, remoteBufferAddr,
                static_cast<unsigned long long>(remoteBufferSize)),
            HCCL_E_INTERNAL);

        ChannelInfo &channelInfo = resCtxHost.channels[idx];
        channelInfo.remoteRank = channelDescs[idx].remoteRank;
        channelInfo.notifyNum = CHANNEL_NOTIFY_NUM;
        channelInfo.handle = channelHandles[idx];
        channelInfo.remoteCclMem = CommBuffer{remoteBufferAddr, remoteBufferSize};
        channelInfo.netLayer = channelNetLayers[idx];
        CHK_PRT_RET(channelInfo.netLayer == INVALID_VALUE_RANKID,
            HCCL_ERROR("[CreateAllPeerChannels] invalid network layer, remoteRank[%u]", channelInfo.remoteRank),
            HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterDfxInfo(HcclComm comm, const OpParam &param, CommEngine engine, uint64_t dataSize)
{
    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH] = {};
    CHK_RET(HcclGetCommName(comm, commName));

    const int32_t tagRet = std::snprintf(dfxInfo.algTag, sizeof(dfxInfo.algTag), "%s", param.tag);
    CHK_PRT_RET(tagRet < 0 || static_cast<uint64_t>(tagRet) >= sizeof(dfxInfo.algTag),
        HCCL_ERROR("[RegisterDfxInfo] failed to copy algorithm tag, ret[%d]", tagRet), HCCL_E_INTERNAL);

    dfxInfo.opType = static_cast<uint32_t>(param.opType);
    dfxInfo.reduceOp = static_cast<uint32_t>(param.reduceType);
    dfxInfo.dataType = static_cast<uint32_t>(param.dataType);
    dfxInfo.dataCount = param.count;
    dfxInfo.root = param.root;
    dfxInfo.engine = engine;
    dfxInfo.cpuTsThread = param.cpuThread;
    dfxInfo.cpuWaitAicpuNotifyIdx = HOST_DEVICE_NOTIFY_IDX;
    dfxInfo.inputMemAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    dfxInfo.inputMemSize = dataSize;
    dfxInfo.outputMemAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    dfxInfo.outputMemSize = dataSize;
    return HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo));
}
} // namespace

HcclResult HcclBroadcast(
    void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(buf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    if (count == 0) {
        HCCL_WARNING("[HcclBroadcast] input count is 0, return success");
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("[HcclBroadcast] unsupported data type[%d]", static_cast<int32_t>(dataType)), HCCL_E_PARA);
    CHK_PRT_RET(count > std::numeric_limits<uint64_t>::max() / FP32_DATA_TYPE_SIZE,
        HCCL_ERROR("[HcclBroadcast] count[%llu] overflows byte size", static_cast<unsigned long long>(count)),
        HCCL_E_PARA);
    const uint64_t dataSize = count * FP32_DATA_TYPE_SIZE;

    // 构造算子参数
    OpParam param{};
    const bool useScatterAllGather = dataSize > BROADCAST_SCATTER_ALLGATHER_THRESHOLD;
    const char *algorithmTag = useScatterAllGather ? BROADCAST_LARGE_TAG : BROADCAST_TAG;
    const int32_t tagRet = std::snprintf(param.tag, sizeof(param.tag), "%s", algorithmTag);
    CHK_PRT_RET(tagRet < 0 || static_cast<uint64_t>(tagRet) >= sizeof(param.tag),
        HCCL_ERROR("[HcclBroadcast] failed to fill operation tag, ret[%d]", tagRet), HCCL_E_INTERNAL);
    param.inputPtr = buf;
    param.outputPtr = buf;
    param.count = count;
    param.dataType = dataType;
    param.root = root;
    param.opType = HcclCMDType::HCCL_CMD_BROADCAST;

    // ==============================================
    // STEP 1: 解析拓扑信息
    // ==============================================
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.myRank >= param.rankSize,
        HCCL_ERROR("[HcclBroadcast] invalid rank information, myRank[%u], rankSize[%u]", param.myRank,
            param.rankSize),
        HCCL_E_PARA);
    CHK_PRT_RET(root >= param.rankSize,
        HCCL_ERROR("[HcclBroadcast] invalid root[%u], rankSize[%u]", root, param.rankSize), HCCL_E_PARA);

    // ==============================================
    // STEP 2: 创建资源
    // ==============================================
    CommEngine aicpuTsEngine = CommEngine::COMM_ENGINE_AICPU_TS;
    CommEngine cpuTsEngine = CommEngine::COMM_ENGINE_CPU_TS;

    // ==============================================
    // STEP 2.1: 申请用于 Host/Device 同步的通信资源
    // ==============================================
    // 将用户传入的 stream 转换为 thread，并申请 Notify；同时导出为 AICPU 上可用的 thread
    CHK_RET(HcclThreadAcquireWithStream(comm, cpuTsEngine, stream, THREAD_NOTIFY_NUM, &param.cpuThread));
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &param.cpuThread, aicpuTsEngine, &param.cpuThreadOnAicpu));
    CHK_RET(RegisterDfxInfo(comm, param, aicpuTsEngine, dataSize));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, aicpuTsEngine, &ctx, &size) == HCCL_SUCCESS) {
        // AICPU 资源已经存在，复用资源
        HCCL_INFO("Engine context already exists");
        CHK_PRT_RET(ctx == nullptr || size == 0,
            HCCL_ERROR("[HcclBroadcast] invalid cached AICPU context, ctx[%p], size[%llu]", ctx,
                static_cast<unsigned long long>(size)),
            HCCL_E_INTERNAL);
        param.resCtx = ctx;
        param.ctxSize = size;

        // Host 资源已经存在，复用资源
        void *hostCtx = nullptr;
        uint64_t hostCtxSize = sizeof(ThreadHandle);
        CHK_RET(HcclEngineCtxGet(comm, param.tag, cpuTsEngine, &hostCtx, &hostCtxSize));
        CHK_PRT_RET(hostCtx == nullptr || hostCtxSize < sizeof(ThreadHandle),
            HCCL_ERROR("[HcclBroadcast] invalid cached Host context, ctx[%p], size[%llu]", hostCtx,
                static_cast<unsigned long long>(hostCtxSize)),
            HCCL_E_INTERNAL);
        ThreadHandle *aicpuThread = static_cast<ThreadHandle *>(hostCtx);
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));
    } else {
        // Device 资源不存在，资源构建
        AlgResourceCtx resCtxHost{};

        // 从通信域获取实际的 HCCL Buffer 地址和大小
        void *cclBufferAddr = nullptr;
        uint64_t cclBufferSize = 0;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        CHK_PRT_RET(cclBufferAddr == nullptr || cclBufferSize == 0,
            HCCL_ERROR("[HcclBroadcast] invalid local HCCL buffer, addr[%p], size[%llu]", cclBufferAddr,
                static_cast<unsigned long long>(cclBufferSize)),
            HCCL_E_INTERNAL);
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        // ==============================================
        // STEP 2.2: 申请资源Thread和Channel
        // ==============================================

        // 小消息保持V1单thread资源；大消息用15条data thread覆盖全部peer，并增加1条专用预取thread。
        const uint32_t aicpuThreadNum = useScatterAllGather ? BROADCAST_AICPU_THREAD_NUM : 1;
        const uint32_t notifyNumPerThread =
            useScatterAllGather ? BROADCAST_AICPU_NOTIFY_NUM_PER_THREAD : THREAD_NOTIFY_NUM;
        resCtxHost.threads.resize(aicpuThreadNum);
        CHK_RET(HcclThreadAcquire(
            comm, aicpuTsEngine, aicpuThreadNum, notifyNumPerThread, resCtxHost.threads.data()));
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        // 为全部 peer 创建 Channel，确保固定 context tag 不依赖某一次调用的 root
        CHK_RET(CreateAllPeerChannels(comm, param, aicpuTsEngine, resCtxHost));

        // ==============================================
        // STEP 2.3: 申请通信引擎上下文
        // ==============================================
        // 申请 AICPU 通信引擎上下文，存放 AlgResourceCtx 信息
        std::vector<char> seq = resCtxHost.Serialize();
        CHK_PRT_RET(seq.empty(), HCCL_ERROR("[HcclBroadcast] serialized resource context is empty"), HCCL_E_INTERNAL);
        uint64_t seqSize = static_cast<uint64_t>(seq.size());
        param.ctxSize = seqSize;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, aicpuTsEngine, param.ctxSize, &param.resCtx));
        CHK_PRT_RET(param.resCtx == nullptr,
            HCCL_ERROR("[HcclBroadcast] created AICPU context is null"), HCCL_E_INTERNAL);
        CHK_RET(HcclEngineCtxCopy(comm, aicpuTsEngine, param.tag, seq.data(), seqSize, 0));
        // 申请 CPU 通信引擎上下文，存放 aicpuThread 句柄
        void *hostCtx = nullptr;
        uint64_t hostCtxSize = sizeof(ThreadHandle);
        const void *aicpuThreadPtr = static_cast<const void *>(&resCtxHost.aicpuThread);
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, cpuTsEngine, hostCtxSize, &hostCtx));
        CHK_RET(HcclEngineCtxCopy(comm, cpuTsEngine, param.tag, aicpuThreadPtr, hostCtxSize, 0));
    }

    // ==============================================
    // STEP 3: 下发 AICPU Kernel
    // ==============================================
    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}
