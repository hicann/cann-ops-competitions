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

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 3;
constexpr uint64_t DIRECT_BROADCAST_MAX_SIZE = 512ULL * 1024ULL;

HcclResult SelectLinkForAicpuTs(HcclComm comm, uint32_t myRank, uint32_t remoteRank, HcclChannelDesc &channelDesc)
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
    std::vector<CommProtocol> protocolPriority = {
        CommProtocol::COMM_PROTOCOL_UBC_CTP,
        CommProtocol::COMM_PROTOCOL_UBC_TP,
        CommProtocol::COMM_PROTOCOL_PCIE,
        CommProtocol::COMM_PROTOCOL_UBOE,
    };

    for (uint32_t layerIdx = 0; layerIdx < netLayerNum; ++layerIdx) {
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayers[layerIdx], myRank, remoteRank, &links, &linkNum));
        if (linkNum == 0) {
            continue;
        }

        for (auto protocol : protocolPriority) {
            for (uint32_t linkIdx = 0; linkIdx < linkNum; ++linkIdx) {
                if (links[linkIdx].linkAttr.linkProtocol != protocol) {
                    continue;
                }
                channelDesc.remoteRank = remoteRank;
                channelDesc.channelProtocol = links[linkIdx].linkAttr.linkProtocol;
                channelDesc.localEndpoint = links[linkIdx].srcEndpointDesc;
                channelDesc.remoteEndpoint = links[linkIdx].dstEndpointDesc;
                channelDesc.notifyNum = CHANNEL_NOTIFY_NUM;
                return HCCL_SUCCESS;
            }
        }
    }

    HCCL_ERROR("No AICPU_TS link found, myRank[%u], remoteRank[%u]", myRank, remoteRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult BuildChannelDescs(
    HcclComm comm, const OpParam &param, HcclMemHandle *userMemHandle, bool useTwoShot,
    std::vector<HcclChannelDesc> &channelDescs)
{
    channelDescs.clear();
    if (param.rankSize <= 1) {
        return HCCL_SUCCESS;
    }

    std::vector<uint32_t> remoteRanks;
    if (param.myRank == param.root) {
        for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
            if (rank != param.myRank) {
                remoteRanks.push_back(rank);
            }
        }
    } else {
        // Keep the root channel on threads[0]. It gates the allgather workers
        // until this rank's scatter slice is ready.
        remoteRanks.push_back(param.root);
        if (useTwoShot) {
            for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
                if (rank != param.myRank && rank != param.root) {
                    remoteRanks.push_back(rank);
                }
            }
        }
    }

    channelDescs.resize(remoteRanks.size());
    for (size_t idx = 0; idx < remoteRanks.size(); ++idx) {
        CHK_RET(HcclChannelDescInit(&channelDescs[idx], 1));
        CHK_RET(SelectLinkForAicpuTs(comm, param.myRank, remoteRanks[idx], channelDescs[idx]));
        channelDescs[idx].memHandles = userMemHandle;
        channelDescs[idx].memHandleNum = 1;
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclBroadcast(
    void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(buf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    // 构造算子参数
    OpParam param;
    char opBaseTag[TAG_LENGTH] = {0};
    std::snprintf(opBaseTag, sizeof(opBaseTag), "hccl_custom_broadcast_r%u_c%llu_t%d", root,
        static_cast<unsigned long long>(count), static_cast<int32_t>(dataType));
    std::snprintf(param.tag, sizeof(param.tag), "%s", opBaseTag);
    param.inputPtr = buf;
    param.outputPtr = buf;
    param.count = count;
    param.dataType = dataType;
    param.root = root;
    param.opType = HcclCMDType::HCCL_CMD_BROADCAST;

    // 注册算子信息
    HcclDfxOpInfo dfxInfo;
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    // ==============================================
    // STEP 1: 解析拓扑信息
    // ==============================================
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    std::snprintf(param.tag, sizeof(param.tag), "%s_rank%u_buf%llx", opBaseTag, param.myRank,
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(buf)));
    CHK_PRT_RET(root >= param.rankSize, HCCL_ERROR("Invalid root[%u], rankSize[%u]", root, param.rankSize), HCCL_E_PARA);
    CHK_PRT_RET(SIZE_TABLE.find(dataType) == SIZE_TABLE.end(),
        HCCL_ERROR("Unsupported dataType[%d]", static_cast<int32_t>(dataType)), HCCL_E_PARA);
    if (count == 0 || param.rankSize <= 1) {
        return HCCL_SUCCESS;
    }
    uint32_t typeSize = SIZE_TABLE.at(dataType);
    CHK_PRT_RET(count > UINT64_MAX / typeSize,
        HCCL_ERROR("Data size overflow, count[%llu], typeSize[%u]", static_cast<unsigned long long>(count), typeSize),
        HCCL_E_PARA);
    uint64_t dataSize = count * typeSize;
    bool useTwoShot = dataSize > DIRECT_BROADCAST_MAX_SIZE;

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
        AlgResourceCtx resCtxHost;

        // 从通信域获取 HCCL Buffer（Device上的内存，默认总大小400MB）
        void *cclBufferAddr;
        uint64_t cclBufferSize;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        // ==============================================
        // STEP 2.2: 申请资源Thread和Channel
        // ==============================================

        // TODO: 根据通信算法申请 Thread 资源
        // 创建 AICPU_TS 通信引擎上的 thread 资源
        uint32_t threadNum = (param.myRank == param.root || useTwoShot) ? param.rankSize - 1 : 1;
        // Root uses threads[0] as the main thread. It needs one host-ready notify and
        // one completion notify for every remaining channel thread.
        uint32_t notifyNumPerThread = std::max(2U, threadNum + 1);

        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, notifyNumPerThread, resCtxHost.threads.data()));
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        // TODO: 根据通信算法申请 Channel 资源
        // 调用 HcclRankGraphGetLinks()、HcclChannelDescInit()、HcclChannelAcquire() 等接口按需申请 Channel 资源
        char userMemTag[HCCL_RES_TAG_MAX_LEN] = {0};
        std::snprintf(userMemTag, sizeof(userMemTag), "%s_user_rank%u_%llx", opBaseTag, param.myRank,
            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(buf)));
        CommMem userMem{COMM_MEM_TYPE_DEVICE, buf, dataSize};
        HcclMemHandle userMemHandle = nullptr;
        CHK_RET(HcclCommMemReg(comm, userMemTag, &userMem, &userMemHandle));
        CHK_PTR_NULL(userMemHandle);

        std::vector<HcclChannelDesc> channelDescs;
        CHK_RET(BuildChannelDescs(comm, param, &userMemHandle, useTwoShot, channelDescs));
        resCtxHost.channels.resize(channelDescs.size());
        std::vector<ChannelHandle> channelHandles(channelDescs.size());
        if (!channelDescs.empty()) {
            CHK_RET(HcclChannelAcquire(
                comm, aicpuTsEngine, channelDescs.data(), channelDescs.size(), channelHandles.data()));
        }
        for (size_t idx = 0; idx < channelDescs.size(); ++idx) {
            resCtxHost.channels[idx].remoteRank = channelDescs[idx].remoteRank;
            resCtxHost.channels[idx].notifyNum = channelDescs[idx].notifyNum;
            resCtxHost.channels[idx].handle = channelHandles[idx];
            CHK_RET(HcclChannelGetHcclBuffer(comm, channelHandles[idx], &resCtxHost.channels[idx].remoteCclMem.addr,
                &resCtxHost.channels[idx].remoteCclMem.size));
            uint32_t memNum = 0;
            CommMem *remoteMems = nullptr;
            char **memTags = nullptr;
            CHK_RET(HcclChannelGetRemoteMems(comm, channelHandles[idx], &memNum, &remoteMems, &memTags));
            CHK_PRT_RET(memNum == 0, HCCL_ERROR("No remote user memory, remoteRank[%u]",
                resCtxHost.channels[idx].remoteRank), HCCL_E_INTERNAL);
            CHK_PTR_NULL(remoteMems);
            CHK_PTR_NULL(memTags);
            char remoteMemTagPrefix[HCCL_RES_TAG_MAX_LEN] = {0};
            std::snprintf(remoteMemTagPrefix, sizeof(remoteMemTagPrefix), "%s_user_rank%u_", opBaseTag,
                resCtxHost.channels[idx].remoteRank);
            size_t remoteMemTagPrefixLen = std::strlen(remoteMemTagPrefix);
            bool foundUserMem = false;
            for (uint32_t memIdx = 0; memIdx < memNum; ++memIdx) {
                if (memTags[memIdx] == nullptr ||
                    std::strncmp(memTags[memIdx], remoteMemTagPrefix, remoteMemTagPrefixLen) != 0) {
                    continue;
                }
                resCtxHost.channels[idx].remoteUserMem =
                    CommBuffer{remoteMems[memIdx].addr, remoteMems[memIdx].size};
                foundUserMem = true;
                break;
            }
            CHK_PRT_RET(!foundUserMem,
                HCCL_ERROR("Remote user memory tag not found, remoteRank[%u], tagPrefix[%s]",
                    resCtxHost.channels[idx].remoteRank, remoteMemTagPrefix),
                HCCL_E_INTERNAL);
        }

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
