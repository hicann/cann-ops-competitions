/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <mutex>

#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

namespace {
std::mutex g_allGatherLaunchMutex;
constexpr uint32_t BUFFER_BANK_MASK = 1;
constexpr uint32_t PREVIOUS_FULL_BUFFER_FLAG = 2;

struct HostResourceCtx {
    ThreadHandle aicpuThread;
    uint32_t nextBufferBank;
    aclrtStream boundStream;
    uint32_t epochHealthy;
    uint64_t halfBufferBytes;
    uint32_t previousUsedFullBuffer;
};

HcclResult GetHostDataTypeSize(HcclDataType dataType, uint64_t &dataTypeSize)
{
    switch (dataType) {
        case HCCL_DATA_TYPE_INT8:
        case HCCL_DATA_TYPE_UINT8:
        case HCCL_DATA_TYPE_HIF8:
        case HCCL_DATA_TYPE_FP8E4M3:
        case HCCL_DATA_TYPE_FP8E5M2:
        case HCCL_DATA_TYPE_FP8E8M0:
            dataTypeSize = 1;
            return HCCL_SUCCESS;
        case HCCL_DATA_TYPE_INT16:
        case HCCL_DATA_TYPE_FP16:
        case HCCL_DATA_TYPE_UINT16:
        case HCCL_DATA_TYPE_BFP16:
            dataTypeSize = 2;
            return HCCL_SUCCESS;
        case HCCL_DATA_TYPE_INT32:
        case HCCL_DATA_TYPE_FP32:
        case HCCL_DATA_TYPE_UINT32:
            dataTypeSize = 4;
            return HCCL_SUCCESS;
        case HCCL_DATA_TYPE_INT64:
        case HCCL_DATA_TYPE_UINT64:
        case HCCL_DATA_TYPE_FP64:
            dataTypeSize = 8;
            return HCCL_SUCCESS;
        case HCCL_DATA_TYPE_INT128:
            dataTypeSize = 16;
            return HCCL_SUCCESS;
        default:
            HCCL_ERROR("Unsupported AllGather data type: %d", static_cast<int32_t>(dataType));
            return HCCL_E_PARA;
    }
}

HcclResult UsesFullBuffer(uint64_t count, HcclDataType dataType, uint32_t rankSize,
    uint64_t halfBufferBytes, bool &usesFullBuffer)
{
    if (count == 0) {
        usesFullBuffer = false;
        return HCCL_SUCCESS;
    }
    uint64_t dataTypeSize = 0;
    CHK_RET(GetHostDataTypeSize(dataType, dataTypeSize));
    CHK_PRT_RET(rankSize == 0 || dataTypeSize == 0,
        HCCL_ERROR("Invalid AllGather full-buffer classifier inputs"), HCCL_E_PARA);
    const uint64_t maxHalfBufferCount = halfBufferBytes / dataTypeSize / rankSize;
    usesFullBuffer = count > maxHalfBufferCount;
    return HCCL_SUCCESS;
}

uint32_t SelectInvocationEpoch(HostResourceCtx *hostResource, uint64_t count, bool currentUsesFullBuffer)
{
    const uint32_t previousFullFlag = hostResource->previousUsedFullBuffer != 0 ?
        PREVIOUS_FULL_BUFFER_FLAG : 0;
    if (count == 0) {
        return (__atomic_load_n(&hostResource->nextBufferBank, __ATOMIC_RELAXED) & BUFFER_BANK_MASK) |
            previousFullFlag;
    }
    const uint32_t bufferBank =
        __atomic_fetch_xor(&hostResource->nextBufferBank, 1U, __ATOMIC_RELAXED) & BUFFER_BANK_MASK;
    hostResource->previousUsedFullBuffer = currentUsesFullBuffer ? 1U : 0U;
    return bufferBank | previousFullFlag;
}
} // namespace

HcclResult HcclAllGather(
    void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    std::lock_guard<std::mutex> launchGuard(g_allGatherLaunchMutex);

    // 构造算子参数
    OpParam param;
    sprintf(param.tag, "%s", "hccl_custom_allgather");
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = sendCount;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;

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
    CHK_PRT_RET(param.rankSize == 0 || param.myRank >= param.rankSize,
        HCCL_ERROR("Invalid rank metadata: rank %u of %u", param.myRank, param.rankSize), HCCL_E_PARA);

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
    HostResourceCtx *selectedHostResource = nullptr;
    if (HcclEngineCtxGet(comm, param.tag, aicpuTsEngine, &ctx, &size) == HCCL_SUCCESS) {
        // AICPU 资源已经存在，复用资源
        HCCL_INFO("Engine context already exists");
        param.resCtx = ctx;
        param.ctxSize = size;

        // Host 资源已经存在，复用资源
        void *hostCtx = nullptr;
        uint64_t hostCtxSize = sizeof(HostResourceCtx);
        CHK_RET(HcclEngineCtxGet(comm, param.tag, cpuTsEngine, &hostCtx, &hostCtxSize));
        CHK_PRT_RET(hostCtx == nullptr || hostCtxSize < sizeof(HostResourceCtx),
            HCCL_ERROR("Invalid cached AICPU thread context, size %llu",
                static_cast<unsigned long long>(hostCtxSize)), HCCL_E_PARA);
        HostResourceCtx *hostResource = static_cast<HostResourceCtx *>(hostCtx);
        CHK_PRT_RET(hostResource->epochHealthy == 0,
            HCCL_ERROR("H18 bank epoch is unavailable after a failed kernel launch"), HCCL_E_RUNTIME);
        CHK_PRT_RET(hostResource->boundStream != stream,
            HCCL_ERROR("H18 requires one bound stream per communicator context"), HCCL_E_NOT_SUPPORT);
        CHK_RET(HcclThreadExportToCommEngine(
            comm, 1, &hostResource->aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));
        bool currentUsesFullBuffer = false;
        CHK_RET(UsesFullBuffer(
            param.count, param.dataType, param.rankSize, hostResource->halfBufferBytes, currentUsesFullBuffer));
        param.root = SelectInvocationEpoch(hostResource, param.count, currentUsesFullBuffer);
        selectedHostResource = hostResource;
    } else {
        // Device 资源不存在，资源构建
        AlgResourceCtx resCtxHost;

        // 从通信域获取 HCCL Buffer（Device上的内存，默认总大小400MB）
        void *cclBufferAddr;
        uint64_t cclBufferSize;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        CHK_PRT_RET(cclBufferAddr == nullptr, HCCL_ERROR("Local HCCL buffer is null"), HCCL_E_PTR);
        CHK_PRT_RET(cclBufferSize == 0, HCCL_ERROR("Local HCCL buffer is empty"), HCCL_E_PARA);
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        // ==============================================
        // STEP 2.2: 申请资源Thread和Channel
        // ==============================================

        // 创建 AICPU_TS 通信引擎上的 thread 资源
        const uint32_t peerCount = param.rankSize - 1;
        const uint32_t threadCount = peerCount == 0 ? 1 : peerCount + 1;
        const uint32_t notifyCount = threadCount;
        resCtxHost.threads.resize(threadCount);
        CHK_RET(HcclThreadAcquire(
            comm, aicpuTsEngine, threadCount, notifyCount, resCtxHost.threads.data()));
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        std::vector<uint32_t> netLayers;
        if (peerCount > 0) {
            uint32_t *netLayerList = nullptr;
            uint32_t netLayerCount = 0;
            CHK_RET(HcclRankGraphGetLayers(comm, &netLayerList, &netLayerCount));
            CHK_PRT_RET(netLayerList == nullptr || netLayerCount == 0,
                HCCL_ERROR("No network layers available for direct AllGather"), HCCL_E_NOT_FOUND);
            netLayers.assign(netLayerList, netLayerList + netLayerCount);
        }

        std::vector<HcclChannelDesc> channelDescs(peerCount);
        std::vector<ChannelHandle> channelHandles(peerCount);
        std::vector<uint32_t> selectedNetLayers(peerCount);
        uint32_t channelIndex = 0;
        for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
            if (remoteRank == param.myRank) {
                continue;
            }

            HcclChannelDesc &channelDesc = channelDescs[channelIndex];
            CHK_RET(HcclChannelDescInit(&channelDesc, 1));

            CommLink tpCandidate{};
            CommLink ctpCandidate{};
            uint32_t tpNetLayer = 0;
            uint32_t ctpNetLayer = 0;
            bool tpSelected = false;
            bool ctpSelected = false;
            for (const uint32_t netLayer : netLayers) {
                CommLink *links = nullptr;
                uint32_t linkCount = 0;
                const HcclResult linkResult =
                    HcclRankGraphGetLinks(comm, netLayer, param.myRank, remoteRank, &links, &linkCount);
                if (linkResult == HCCL_E_NOT_FOUND) {
                    continue;
                }
                CHK_RET(linkResult);
                if (links == nullptr || linkCount == 0) {
                    continue;
                }

                for (uint32_t linkIndex = 0; linkIndex < linkCount; ++linkIndex) {
                    const CommLink copiedLink = links[linkIndex];
                    const CommProtocol protocol = copiedLink.linkAttr.linkProtocol;
                    if (!tpSelected && protocol == CommProtocol::COMM_PROTOCOL_UBC_TP) {
                        tpCandidate = copiedLink;
                        tpNetLayer = netLayer;
                        tpSelected = true;
                    } else if (!ctpSelected && protocol == CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                        ctpCandidate = copiedLink;
                        ctpNetLayer = netLayer;
                        ctpSelected = true;
                    }
                }
            }

            CommLink selectedLink{};
            uint32_t selectedNetLayer = 0;
            bool linkSelected = false;
            if (tpSelected) {
                selectedLink = tpCandidate;
                selectedNetLayer = tpNetLayer;
                linkSelected = true;
            } else if (ctpSelected) {
                selectedLink = ctpCandidate;
                selectedNetLayer = ctpNetLayer;
                linkSelected = true;
            }
            CHK_PRT_RET(!linkSelected,
                HCCL_ERROR("UBC link not found between rank %u and rank %u", param.myRank, remoteRank),
                HCCL_E_NOT_FOUND);

            channelDesc.remoteRank = remoteRank;
            channelDesc.channelProtocol = selectedLink.linkAttr.linkProtocol;
            channelDesc.localEndpoint = selectedLink.srcEndpointDesc;
            channelDesc.remoteEndpoint = selectedLink.dstEndpointDesc;
            channelDesc.notifyNum = DIRECT_CHANNEL_NOTIFY_COUNT;
            selectedNetLayers[channelIndex] = selectedNetLayer;
            ++channelIndex;
        }
        CHK_PRT_RET(channelIndex != peerCount,
            HCCL_ERROR("Channel descriptor cardinality mismatch: %u != %u", channelIndex, peerCount),
            HCCL_E_PARA);

        if (peerCount > 0) {
            std::vector<uint32_t> channelOrder(peerCount);
            for (uint32_t index = 0; index < peerCount; ++index) {
                channelOrder[index] = index;
            }
            std::stable_sort(channelOrder.begin(), channelOrder.end(), [&](uint32_t lhs, uint32_t rhs) {
                const bool lhsIsCtp = channelDescs[lhs].channelProtocol == CommProtocol::COMM_PROTOCOL_UBC_CTP;
                const bool rhsIsCtp = channelDescs[rhs].channelProtocol == CommProtocol::COMM_PROTOCOL_UBC_CTP;
                if (lhsIsCtp != rhsIsCtp) {
                    return lhsIsCtp;
                }
                if (selectedNetLayers[lhs] != selectedNetLayers[rhs]) {
                    return selectedNetLayers[lhs] > selectedNetLayers[rhs];
                }
                return channelDescs[lhs].remoteRank < channelDescs[rhs].remoteRank;
            });
            std::vector<HcclChannelDesc> orderedChannelDescs(peerCount);
            for (uint32_t index = 0; index < peerCount; ++index) {
                orderedChannelDescs[index] = channelDescs[channelOrder[index]];
            }
            channelDescs.swap(orderedChannelDescs);

            CHK_RET(HcclChannelAcquire(
                comm, aicpuTsEngine, channelDescs.data(), peerCount, channelHandles.data()));
        }

        resCtxHost.channels.reserve(peerCount);
        for (uint32_t index = 0; index < peerCount; ++index) {
            void *remoteBufferAddr = nullptr;
            uint64_t remoteBufferSize = 0;
            CHK_RET(HcclChannelGetHcclBuffer(
                comm, channelHandles[index], &remoteBufferAddr, &remoteBufferSize));
            CHK_PRT_RET(remoteBufferAddr == nullptr,
                HCCL_ERROR("Remote HCCL buffer for rank %u is null", channelDescs[index].remoteRank), HCCL_E_PTR);
            CHK_PRT_RET(remoteBufferSize < resCtxHost.localBuffer.size,
                HCCL_ERROR("Remote HCCL buffer for rank %u is too small: %llu < %llu",
                    channelDescs[index].remoteRank, static_cast<unsigned long long>(remoteBufferSize),
                    static_cast<unsigned long long>(resCtxHost.localBuffer.size)), HCCL_E_PARA);

            ChannelInfo channel;
            channel.remoteRank = channelDescs[index].remoteRank;
            channel.notifyNum = DIRECT_CHANNEL_NOTIFY_COUNT;
            channel.handle = channelHandles[index];
            channel.remoteCclMem = CommBuffer{remoteBufferAddr, remoteBufferSize};
            resCtxHost.channels.push_back(channel);
        }
        CHK_PRT_RET(resCtxHost.channels.size() != peerCount,
            HCCL_ERROR("Serialized channel cardinality mismatch: %zu != %u", resCtxHost.channels.size(), peerCount),
            HCCL_E_PARA);

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
        uint64_t hostCtxSize = sizeof(HostResourceCtx);
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, cpuTsEngine, hostCtxSize, &hostCtx));
        CHK_PRT_RET(hostCtx == nullptr || hostCtxSize < sizeof(HostResourceCtx),
            HCCL_ERROR("Invalid created AICPU thread context, size %llu",
                static_cast<unsigned long long>(hostCtxSize)), HCCL_E_PARA);
        HostResourceCtx *hostResource = static_cast<HostResourceCtx *>(hostCtx);
        hostResource->aicpuThread = resCtxHost.aicpuThread;
        hostResource->nextBufferBank = 0;
        hostResource->boundStream = stream;
        hostResource->epochHealthy = 1;
        hostResource->halfBufferBytes = resCtxHost.localBuffer.size / 2;
        hostResource->previousUsedFullBuffer = 0;
        bool currentUsesFullBuffer = false;
        CHK_RET(UsesFullBuffer(
            param.count, param.dataType, param.rankSize, hostResource->halfBufferBytes, currentUsesFullBuffer));
        param.root = SelectInvocationEpoch(hostResource, param.count, currentUsesFullBuffer);
        selectedHostResource = hostResource;
    }

    // ==============================================
    // STEP 3: 下发 AICPU Kernel
    // ==============================================
    const HcclResult launchResult = ops_hccl::LaunchAICPUKernel(param, stream);
    if (launchResult != HCCL_SUCCESS) {
        selectedHostResource->epochHealthy = 0;
        HCCL_ERROR("H18 kernel launch failed; bank epoch context is now unavailable");
        return launchResult;
    }
    return HCCL_SUCCESS;
}
