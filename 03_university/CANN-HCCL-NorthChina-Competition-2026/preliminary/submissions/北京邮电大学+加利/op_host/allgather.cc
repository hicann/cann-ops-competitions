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

#include <cstring>
#include <limits>
#include <vector>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

namespace {

constexpr uint32_t kRankSize = 16;
constexpr uint32_t kRanksPerServer = 8;
constexpr uint32_t kChannelNotifyNum = 2;
constexpr uint32_t kDirectThreadNum = kRankSize;
constexpr uint32_t kDirectNotifyNum = 2 * kRankSize;
constexpr const char *kInputMemTag = "hccl_custom_allgather_input";
constexpr const char *kOutputMemTag = "hccl_custom_allgather_output";

bool IsAicpuProtocol(uint32_t protocol)
{
    return protocol == COMM_PROTOCOL_HCCS || protocol == COMM_PROTOCOL_UBOE ||
           protocol == COMM_PROTOCOL_PCIE || protocol == COMM_PROTOCOL_ROCE ||
           protocol == COMM_PROTOCOL_UBC_CTP || protocol == COMM_PROTOCOL_UBC_TP;
}

HcclResult FillChannelDesc(HcclComm comm, uint32_t localRank, uint32_t remoteRank, HcclChannelDesc *desc)
{
    uint32_t *netLayers = nullptr;
    uint32_t netLayerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));

    for (uint32_t layerIdx = 0; layerIdx < netLayerNum; ++layerIdx) {
        CommLink *linkList = nullptr;
        uint32_t linkNum = 0;
        CHK_RET(HcclRankGraphGetLinks(comm, netLayers[layerIdx], localRank, remoteRank, &linkList, &linkNum));
        for (uint32_t linkIdx = 0; linkIdx < linkNum; ++linkIdx) {
            const CommLink &link = linkList[linkIdx];
            if (!IsAicpuProtocol(link.linkAttr.linkProtocol)) {
                continue;
            }
            CHK_RET(HcclChannelDescInit(desc, 1));
            desc->remoteRank = remoteRank;
            desc->notifyNum = kChannelNotifyNum;
            desc->channelProtocol = link.linkAttr.linkProtocol;
            desc->localEndpoint.protocol = link.srcEndpointDesc.protocol;
            desc->localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
            desc->localEndpoint.loc = link.srcEndpointDesc.loc;
            desc->remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
            desc->remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
            desc->remoteEndpoint.loc = link.dstEndpointDesc.loc;
            return HCCL_SUCCESS;
        }
    }
    HCCL_ERROR("No AICPU-compatible channel from rank %u to rank %u", localRank, remoteRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult AcquireAlgorithmResources(HcclComm comm, const OpParam &param, AlgResourceCtx *resCtx)
{
    void *buffer = nullptr;
    uint64_t bufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &buffer, &bufferSize));
    resCtx->localBuffer = CommBuffer{buffer, bufferSize};
    resCtx->localRankCount = kRanksPerServer;

    CHK_PRT_RET(param.dataType != HCCL_DATA_TYPE_FP32,
                HCCL_ERROR("Unsupported data type %u", static_cast<uint32_t>(param.dataType)),
                HCCL_E_PARA);
    constexpr uint64_t typeSize = sizeof(float);
    CHK_PRT_RET(param.count > std::numeric_limits<uint64_t>::max() / typeSize,
                HCCL_ERROR("Input byte size overflows uint64_t"), HCCL_E_PARA);
    const uint64_t inputBytes = param.count * typeSize;
    HcclMemHandle inputMemHandle = nullptr;
    HcclMemHandle outputMemHandle = nullptr;
    std::vector<HcclMemHandle> memHandles;
    memHandles.reserve(2);
    if (inputBytes != 0) {
        CommMem inputMem{COMM_MEM_TYPE_DEVICE, param.inputPtr, inputBytes};
        CHK_RET(HcclCommMemReg(comm, kInputMemTag, &inputMem, &inputMemHandle));
        resCtx->registeredInput = CommBuffer{param.inputPtr, inputBytes};
        memHandles.push_back(inputMemHandle);

        CHK_PRT_RET(inputBytes > std::numeric_limits<uint64_t>::max() / kRankSize,
                    HCCL_ERROR("Output byte size overflows uint64_t"), HCCL_E_PARA);
        const uint64_t outputBytes = inputBytes * kRankSize;
        CommMem outputMem{COMM_MEM_TYPE_DEVICE, param.outputPtr, outputBytes};
        CHK_RET(HcclCommMemReg(comm, kOutputMemTag, &outputMem, &outputMemHandle));
        resCtx->registeredOutput = CommBuffer{param.outputPtr, outputBytes};
        memHandles.push_back(outputMemHandle);
    }

    // The large-message path reads every peer's CCL slice directly into the
    // final output. Threads 0..14 each own one peer channel and thread 15
    // copies this rank's contribution while those reads are in flight.
    static_assert(kDirectThreadNum == ALG_MAX_THREAD_NUM,
                  "AllGather thread resource layout must stay fixed");
    resCtx->threadNum = kDirectThreadNum;
    CHK_RET(HcclThreadAcquire(comm, CommEngine::COMM_ENGINE_AICPU_TS, kDirectThreadNum,
                              kDirectNotifyNum, resCtx->threads));
    resCtx->aicpuThread = resCtx->threads[0];

    const uint32_t serverBegin = (param.myRank / kRanksPerServer) * kRanksPerServer;
    const uint32_t remoteServerBegin =
        ((param.myRank / kRanksPerServer + 1) % 2) * kRanksPerServer;
    const uint32_t correspondingRemoteRank = (param.myRank + kRanksPerServer) % kRankSize;
    std::vector<uint32_t> peers;
    peers.reserve(kRankSize - 1);
    for (uint32_t peer = serverBegin; peer < serverBegin + kRanksPerServer; ++peer) {
        if (peer != param.myRank) {
            peers.push_back(peer);
        }
    }
    // Keep the corresponding Clos peer at index 7 so the small-message fast
    // path retains its original channel and thread layout.
    peers.push_back(correspondingRemoteRank);
    for (uint32_t peer = remoteServerBegin; peer < remoteServerBegin + kRanksPerServer; ++peer) {
        if (peer != correspondingRemoteRank) {
            peers.push_back(peer);
        }
    }

    std::vector<HcclChannelDesc> descs(peers.size());
    std::vector<ChannelHandle> handles(peers.size());
    for (uint32_t idx = 0; idx < peers.size(); ++idx) {
        CHK_RET(FillChannelDesc(comm, param.myRank, peers[idx], &descs[idx]));
        if (!memHandles.empty()) {
            descs[idx].memHandles = memHandles.data();
            descs[idx].memHandleNum = static_cast<uint32_t>(memHandles.size());
        }
    }
    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AICPU_TS, descs.data(),
                               static_cast<uint32_t>(descs.size()), handles.data()));

    CHK_PRT_RET(peers.size() != ALG_MAX_CHANNEL_NUM,
                HCCL_ERROR("Unexpected AllGather peer count %zu", peers.size()), HCCL_E_PARA);
    resCtx->channelNum = static_cast<uint32_t>(peers.size());
    for (uint32_t idx = 0; idx < peers.size(); ++idx) {
        void *remoteBuffer = nullptr;
        uint64_t remoteBufferSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, handles[idx], &remoteBuffer, &remoteBufferSize));
        resCtx->channels[idx].remoteRank = peers[idx];
        resCtx->channels[idx].notifyNum = kChannelNotifyNum;
        resCtx->channels[idx].handle = handles[idx];
        resCtx->channels[idx].remoteCclMem = CommBuffer{remoteBuffer, remoteBufferSize};

        if (inputMemHandle != nullptr) {
            uint32_t remoteMemNum = 0;
            CommMem *remoteMems = nullptr;
            char **remoteMemTags = nullptr;
            CHK_RET(HcclChannelGetRemoteMems(
                comm, handles[idx], &remoteMemNum, &remoteMems, &remoteMemTags));
            for (uint32_t memIdx = 0; memIdx < remoteMemNum; ++memIdx) {
                if (remoteMemTags[memIdx] != nullptr &&
                    std::strcmp(remoteMemTags[memIdx], kInputMemTag) == 0) {
                    resCtx->channels[idx].remoteInputMem =
                        CommBuffer{remoteMems[memIdx].addr, remoteMems[memIdx].size};
                } else if (remoteMemTags[memIdx] != nullptr &&
                           std::strcmp(remoteMemTags[memIdx], kOutputMemTag) == 0) {
                    resCtx->channels[idx].remoteOutputMem =
                        CommBuffer{remoteMems[memIdx].addr, remoteMems[memIdx].size};
                }
            }
            CHK_PRT_RET(resCtx->channels[idx].remoteInputMem.addr == nullptr,
                        HCCL_ERROR("Registered input memory not found for remote rank %u", peers[idx]),
                        HCCL_E_NOT_FOUND);
            CHK_PRT_RET(resCtx->channels[idx].remoteOutputMem.addr == nullptr,
                        HCCL_ERROR("Registered output memory not found for remote rank %u", peers[idx]),
                        HCCL_E_NOT_FOUND);
        }
    }
    return HCCL_SUCCESS;
}

} // namespace

HcclResult HcclAllGather(
    void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    // 构造算子参数
    OpParam param;
    sprintf(param.tag, "%s", "hccl_custom_allgather");
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = sendCount;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;

    // Attach DFX metadata before enqueuing any host or AICPU communication
    // tasks. The runtime profiling teardown expects every task to carry it.
    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(
        commName, reinterpret_cast<void *>(&dfxInfo)));

    // ==============================================
    // STEP 1: 解析拓扑信息
    // ==============================================
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize != kRankSize,
                HCCL_ERROR("This implementation requires the contest 2 x 8 topology, got rankSize=%u", param.rankSize),
                HCCL_E_PARA);

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
        CHK_PRT_RET(size != sizeof(AlgResourceCtx),
                    HCCL_ERROR("Unexpected AllGather engine context size %llu, expected %zu",
                               static_cast<unsigned long long>(size), sizeof(AlgResourceCtx)),
                    HCCL_E_PARA);
        param.resCtx = ctx;
        param.ctxSize = size;

        // Host 资源已经存在，复用资源
        void *hostCtx = nullptr;
        uint64_t hostCtxSize = 0;
        CHK_RET(HcclEngineCtxGet(comm, param.tag, cpuTsEngine, &hostCtx, &hostCtxSize));
        const ThreadHandle *aicpuThreadOnCpu = static_cast<const ThreadHandle *>(hostCtx);
        param.aicpuThreadOnCpu = *aicpuThreadOnCpu;
    } else {
        // Device 资源不存在，资源构建
        AlgResourceCtx resCtxHost{};

        // ==============================================
        // STEP 2.2: 申请资源Thread和Channel
        // ==============================================

        CHK_RET(AcquireAlgorithmResources(comm, param, &resCtxHost));
        // 将主AICPU thread导出为CPU可用的thread，用于Host与Device同步。
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        // ==============================================
        // STEP 2.3: 申请通信引擎上下文
        // ==============================================
        // AlgResourceCtx has a fixed cross-architecture layout, so AICPU can
        // consume the engine context directly on every collective invocation.
        param.ctxSize = sizeof(AlgResourceCtx);
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, aicpuTsEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(
            comm, aicpuTsEngine, param.tag, &resCtxHost, sizeof(resCtxHost), 0));
        // Cache the already-exported CPU-side handle. It is a comm-level
        // invariant, so subsequent collectives do not need to export it again.
        void *hostCtx = nullptr;
        uint64_t hostCtxSize = sizeof(ThreadHandle);
        const void *aicpuThreadOnCpu = static_cast<const void *>(&param.aicpuThreadOnCpu);
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, cpuTsEngine, hostCtxSize, &hostCtx));
        CHK_RET(HcclEngineCtxCopy(comm, cpuTsEngine, param.tag, aicpuThreadOnCpu, hostCtxSize, 0));
    }

    // ==============================================
    // STEP 3: 下发 AICPU Kernel
    // ==============================================
    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}
