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
#include <cstring>
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
bool IsValidLink(const CommLink &link)
{
    return link.linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_RESERVED
        && link.srcEndpointDesc.protocol != CommProtocol::COMM_PROTOCOL_RESERVED
        && link.dstEndpointDesc.protocol != CommProtocol::COMM_PROTOCOL_RESERVED
        && link.srcEndpointDesc.loc.locType != EndpointLocType::ENDPOINT_LOC_TYPE_RESERVED
        && link.dstEndpointDesc.loc.locType != EndpointLocType::ENDPOINT_LOC_TYPE_RESERVED;
}

uint32_t ProtocolPriority(CommProtocol protocol)
{
    if (protocol == CommProtocol::COMM_PROTOCOL_UBC_CTP) {
        return 0;
    }
    if (protocol == CommProtocol::COMM_PROTOCOL_UBC_TP) {
        return 1;
    }
    if (protocol == CommProtocol::COMM_PROTOCOL_PCIE) {
        return 2;
    }
    return 3;
}

uint32_t GetBandwidthCoeff(HcclComm comm, uint32_t rank, const EndpointDesc &endpoint)
{
    uint32_t bandwidthCoeff = 1;
    HcclResult ret = HcclRankGraphGetEndpointInfo(
        comm, rank, &endpoint, ENDPOINT_ATTR_BW_COEFF, sizeof(bandwidthCoeff), &bandwidthCoeff);
    return ret == HCCL_SUCCESS && bandwidthCoeff != 0 ? bandwidthCoeff : 1;
}

HcclResult GetRemoteRegisteredBuffers(HcclComm comm, ChannelHandle handle, const char *inputTag,
    const char *outputTag, CommBuffer &remoteInput, CommBuffer &remoteOutput)
{
    uint32_t memNum = 0;
    CommMem *remoteMems = nullptr;
    char **memTags = nullptr;
    CHK_RET(HcclChannelGetRemoteMems(comm, handle, &memNum, &remoteMems, &memTags));
    if (memNum == 0 || remoteMems == nullptr || memTags == nullptr) {
        return HCCL_E_NOT_FOUND;
    }
    for (uint32_t index = 0; index < memNum; ++index) {
        if (memTags[index] == nullptr) {
            continue;
        }
        if (std::strcmp(memTags[index], inputTag) == 0) {
            remoteInput = CommBuffer{remoteMems[index].addr, remoteMems[index].size};
        } else if (std::strcmp(memTags[index], outputTag) == 0) {
            remoteOutput = CommBuffer{remoteMems[index].addr, remoteMems[index].size};
        }
    }
    if (remoteInput.addr == nullptr || remoteOutput.addr == nullptr) {
        return HCCL_E_NOT_FOUND;
    }
    return HCCL_SUCCESS;
}
}

HcclResult HcclAllGather(
    void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    OpParam param;
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = sendCount;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_ALLGATHER;
    int tagRet = std::snprintf(param.tag, sizeof(param.tag), "hccl_custom_allgather_%llu_%u",
        static_cast<unsigned long long>(sendCount), static_cast<uint32_t>(dataType));
    if (tagRet <= 0 || static_cast<uint64_t>(tagRet) >= sizeof(param.tag)) {
        return HCCL_E_INTERNAL;
    }

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
    if (param.rankSize == 0 || param.myRank >= param.rankSize) {
        HCCL_ERROR("Invalid rank %u or rank size %u", param.myRank, param.rankSize);
        return HCCL_E_PARA;
    }
    auto typeSize = SIZE_TABLE.find(dataType);
    if (typeSize == SIZE_TABLE.end()) {
        return HCCL_E_NOT_SUPPORT;
    }
    if (sendCount > std::numeric_limits<uint64_t>::max() / typeSize->second) {
        return HCCL_E_PARA;
    }
    uint64_t inputSize = sendCount * typeSize->second;
    if (inputSize == 0) {
        return HCCL_SUCCESS;
    }
    if (inputSize > std::numeric_limits<uint64_t>::max() / param.rankSize) {
        return HCCL_E_PARA;
    }
    uint64_t outputSize = inputSize * param.rankSize;

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
        resCtxHost.registeredInput = CommBuffer{sendBuf, inputSize};
        resCtxHost.registeredOutput = CommBuffer{recvBuf, outputSize};
        uint64_t minBufferSize = cclBufferSize;

        if (param.rankSize < 2 || param.rankSize % 2 != 0) {
            HCCL_ERROR("Unsupported rank size %u", param.rankSize);
            return HCCL_E_NOT_SUPPORT;
        }
        resCtxHost.serverRankSize = param.rankSize / 2;
        uint32_t localRankIndex = param.myRank % resCtxHost.serverRankSize;
        uint32_t serverIndex = param.myRank / resCtxHost.serverRankSize;
        uint32_t serverBaseRank = serverIndex * resCtxHost.serverRankSize;
        uint32_t peerServerBaseRank = serverIndex == 0 ? resCtxHost.serverRankSize : 0;
        uint32_t pairedRank = peerServerBaseRank + localRankIndex;

        // ==============================================
        // STEP 2.2: 申请资源Thread和Channel
        // ==============================================

        // 创建 AICPU_TS 通信引擎上的 thread 资源
        uint32_t threadNum = resCtxHost.serverRankSize + 1;
        uint32_t notifyNumPerThread = threadNum;

        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, notifyNumPerThread, resCtxHost.threads.data()));
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        uint32_t channelNum = resCtxHost.serverRankSize;
        uint32_t intraChannelNum = resCtxHost.serverRankSize - 1;
        uint32_t *netLayers = nullptr;
        uint32_t netLayerNum = 0;
        CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum));
        CHK_PTR_NULL(netLayers);

        std::vector<uint32_t> remoteRanks;
        remoteRanks.reserve(channelNum);
        for (uint32_t remoteRank = serverBaseRank; remoteRank < serverBaseRank + resCtxHost.serverRankSize;
            ++remoteRank) {
            if (remoteRank != param.myRank) {
                remoteRanks.push_back(remoteRank);
            }
        }
        remoteRanks.push_back(pairedRank);

        char inputBuffTag[TAG_LENGTH] = {};
        char outputBuffTag[TAG_LENGTH] = {};
        int inputTagRet = std::snprintf(inputBuffTag, sizeof(inputBuffTag), "%s_InputBuffer", param.tag);
        int outputTagRet = std::snprintf(outputBuffTag, sizeof(outputBuffTag), "%s_OutputBuffer", param.tag);
        if (inputTagRet <= 0 || outputTagRet <= 0 || static_cast<uint64_t>(inputTagRet) >= sizeof(inputBuffTag)
            || static_cast<uint64_t>(outputTagRet) >= sizeof(outputBuffTag)) {
            return HCCL_E_INTERNAL;
        }
        HcclMemHandle inputHandle = nullptr;
        HcclMemHandle outputHandle = nullptr;
        CommMem inputMem{COMM_MEM_TYPE_DEVICE, sendBuf, inputSize};
        CommMem outputMem{COMM_MEM_TYPE_DEVICE, recvBuf, outputSize};
        CHK_RET(HcclCommMemReg(comm, inputBuffTag, &inputMem, &inputHandle));
        CHK_RET(HcclCommMemReg(comm, outputBuffTag, &outputMem, &outputHandle));
        CHK_PTR_NULL(inputHandle);
        CHK_PTR_NULL(outputHandle);
        std::vector<HcclMemHandle> memHandles{inputHandle, outputHandle};

        std::vector<HcclChannelDesc> channelDescs(channelNum);
        std::vector<uint32_t> bandwidthCoeffs(channelNum, 1);
        CHK_RET(HcclChannelDescInit(channelDescs.data(), channelNum));
        for (uint32_t channelIndex = 0; channelIndex < channelNum; ++channelIndex) {
            uint32_t remoteRank = remoteRanks[channelIndex];
            bool linkFound = false;
            for (uint32_t layerIndex = 0; layerIndex < netLayerNum; ++layerIndex) {
                CommLink *links = nullptr;
                uint32_t linkNum = 0;
                CHK_RET(HcclRankGraphGetLinks(comm, netLayers[layerIndex], param.myRank, remoteRank, &links, &linkNum));
                uint32_t selectedLink = linkNum;
                uint32_t selectedPriority = INVALID_VALUE_RANKID;
                uint32_t selectedBandwidth = 0;
                for (uint32_t linkIndex = 0; links != nullptr && linkIndex < linkNum; ++linkIndex) {
                    if (!IsValidLink(links[linkIndex])) {
                        continue;
                    }
                    uint32_t priority = ProtocolPriority(links[linkIndex].linkAttr.linkProtocol);
                    uint32_t bandwidth = GetBandwidthCoeff(comm, param.myRank, links[linkIndex].srcEndpointDesc);
                    if (selectedLink == linkNum || priority < selectedPriority
                        || (priority == selectedPriority && bandwidth > selectedBandwidth)) {
                        selectedLink = linkIndex;
                        selectedPriority = priority;
                        selectedBandwidth = bandwidth;
                    }
                }
                if (selectedLink != linkNum) {
                    const CommLink &link = links[selectedLink];
                    channelDescs[channelIndex].localEndpoint = link.srcEndpointDesc;
                    channelDescs[channelIndex].remoteEndpoint = link.dstEndpointDesc;
                    channelDescs[channelIndex].channelProtocol = link.linkAttr.linkProtocol;
                    bandwidthCoeffs[channelIndex] = selectedBandwidth;
                    linkFound = true;
                    break;
                }
            }
            if (!linkFound) {
                HCCL_ERROR("No communication link from rank %u to rank %u", param.myRank, remoteRank);
                return HCCL_E_NOT_FOUND;
            }

            channelDescs[channelIndex].remoteRank = remoteRank;
            channelDescs[channelIndex].notifyNum = 4;
            channelDescs[channelIndex].memHandles = memHandles.data();
            channelDescs[channelIndex].memHandleNum = memHandles.size();
        }

        std::vector<ChannelHandle> channelHandles(channelNum);
        CHK_RET(HcclChannelAcquire(comm, aicpuTsEngine, channelDescs.data(), channelNum, channelHandles.data()));

        resCtxHost.intraChannels.resize(intraChannelNum);
        for (uint32_t index = 0; index < channelNum; ++index) {
            void *remoteBufferAddr = nullptr;
            uint64_t remoteBufferSize = 0;
            CHK_RET(HcclChannelGetHcclBuffer(comm, channelHandles[index], &remoteBufferAddr, &remoteBufferSize));
            CHK_PTR_NULL(remoteBufferAddr);

            ChannelInfo channel;
            channel.remoteRank = remoteRanks[index];
            channel.notifyNum = 4;
            channel.protocol = channelDescs[index].channelProtocol;
            channel.bandwidthCoeff = bandwidthCoeffs[index];
            channel.handle = channelHandles[index];
            channel.remoteCclMem = CommBuffer{remoteBufferAddr, remoteBufferSize};
            CHK_RET(GetRemoteRegisteredBuffers(comm, channel.handle, inputBuffTag, outputBuffTag,
                channel.remoteInput, channel.remoteOutput));
            if (channel.remoteInput.size < inputSize || channel.remoteOutput.size < outputSize) {
                return HCCL_E_MEMORY;
            }
            if (index < intraChannelNum) {
                resCtxHost.intraChannels[index] = channel;
            } else {
                resCtxHost.interChannel = channel;
            }
            if (remoteBufferSize < minBufferSize) {
                minBufferSize = remoteBufferSize;
            }
        }

        resCtxHost.chunkSize = minBufferSize / resCtxHost.serverRankSize;
        resCtxHost.chunkSize -= resCtxHost.chunkSize % sizeof(float);
        if (resCtxHost.chunkSize == 0) {
            HCCL_ERROR("Communication buffer is too small");
            return HCCL_E_MEMORY;
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
