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

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

namespace {
HcclResult AcquireScatterChannels(HcclComm comm, const OpParam &param, AlgResourceCtx &ctx)
{
    if (param.rankSize == 1) {
        return HCCL_SUCCESS;
    }
    uint32_t *layerPtr = nullptr;
    uint32_t layerCount = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layerPtr, &layerCount));
    CHK_PTR_NULL(layerPtr);
    const std::vector<uint32_t> layers(layerPtr, layerPtr + layerCount);
    std::vector<HcclChannelDesc> descs;
    for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
        if (peer == param.myRank) {
            continue;
        }
        HcclChannelDesc desc;
        CHK_RET(HcclChannelDescInit(&desc, 1));
        bool found = false;
        for (uint32_t layer : layers) {
            CommLink *links = nullptr;
            uint32_t linkCount = 0;
            CHK_RET(HcclRankGraphGetLinks(comm, layer, param.myRank, peer, &links, &linkCount));
            for (uint32_t i = 0; i < linkCount; ++i) {
                const auto &link = links[i];
                const auto protocol = link.linkAttr.linkProtocol;
                if (protocol != COMM_PROTOCOL_UBC_CTP && protocol != COMM_PROTOCOL_UBOE) {
                    continue;
                }
                desc.remoteRank = peer;
                desc.notifyNum = SCATTER_CHANNEL_NOTIFIES;
                desc.channelProtocol = protocol;
                desc.localEndpoint = link.srcEndpointDesc;
                desc.remoteEndpoint = link.dstEndpointDesc;
                found = true;
                break;
            }
            if (found) {
                break;
            }
        }
        CHK_PRT_RET(!found, HCCL_ERROR("No AICPU TS link from %u to %u", param.myRank, peer), HCCL_E_NOT_SUPPORT);
        descs.push_back(desc);
        ChannelInfo info;
        info.remoteRank = peer;
        info.notifyNum = SCATTER_CHANNEL_NOTIFIES;
        ctx.channels.push_back(info);
    }
    // Submit all descriptors together; every peer gets exactly one channel.
    std::vector<ChannelHandle> handles(descs.size());
    CHK_RET(HcclChannelAcquire(comm, COMM_ENGINE_AICPU_TS, descs.data(),
        static_cast<uint32_t>(descs.size()), handles.data()));
    for (size_t i = 0; i < handles.size(); ++i) {
        ctx.channels[i].handle = handles[i];
        CHK_RET(HcclChannelGetHcclBuffer(comm, handles[i], &ctx.channels[i].remoteCclMem.addr,
            &ctx.channels[i].remoteCclMem.size));
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    // 构造算子参数
    OpParam param{};
    sprintf(param.tag, "%s", "hccl_custom_scatter");
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
    // STEP 1: 解析拓扑信息
    // ==============================================
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || root >= param.rankSize || dataType != HCCL_DATA_TYPE_FP32,
        HCCL_ERROR("Invalid Scatter rank, root or data type"), HCCL_E_PARA);
    CHK_PRT_RET(recvCount > UINT64_MAX / sizeof(float) / param.rankSize,
        HCCL_ERROR("Scatter byte count overflow"), HCCL_E_PARA);
    if (param.myRank == root) {
        CHK_PTR_NULL(sendBuf);
    }

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

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, aicpuTsEngine, &ctx, &size) == HCCL_SUCCESS) {
        // AICPU 资源已经存在，复用资源
        HCCL_INFO("Engine context already exists");
        param.resCtx = ctx;
        param.ctxSize = size;

        // Host 资源已经存在，复用资源
        void *hostCtx = nullptr;
        uint64_t hostCtxSize = 0;
        CHK_RET(HcclEngineCtxGet(comm, param.tag, cpuTsEngine, &hostCtx, &hostCtxSize));
        ThreadHandle *aicpuThread = static_cast<ThreadHandle *>(hostCtx);
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));
    } else {
        // Device 资源不存在，资源构建
        AlgResourceCtx resCtxHost{};

        // 从通信域获取 HCCL Buffer（Device上的内存，默认总大小400MB）
        void *cclBufferAddr;
        uint64_t cclBufferSize;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        // ==============================================
        // STEP 2.2: 申请资源Thread和Channel
        // ==============================================

        // Main thread plus one independent lane per peer. Notify 0 is reserved
        // for the template's host/device handshake; lane i completes on i+1.
        uint32_t threadNum = param.rankSize;
        uint32_t notifyNumPerThread = param.rankSize;

        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, notifyNumPerThread, resCtxHost.threads.data()));
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        CHK_RET(AcquireScatterChannels(comm, param, resCtxHost));

        // ==============================================
        // STEP 2.3: 申请通信引擎上下文
        // ==============================================
        // 申请 AICPU 通信引擎上下文，存放 AlgResourceCtx 信息
        std::vector<char> seq = resCtxHost.Serialize();
        uint64_t seqSize = seq.size();
        param.ctxSize = seqSize;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, aicpuTsEngine, param.ctxSize, &param.resCtx));
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
