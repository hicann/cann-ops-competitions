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
#include <cstring>
#include <limits>
#include <vector>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

namespace {
constexpr char SCATTER_TAG_FORMAT[] = "hccl_scatter_relay_v14_r%u_c%llu_t%d";
constexpr char SCATTER_OUTPUT_MEM_TAG_FORMAT[] = "hccl_scatter_relaymem_v14_r%u_c%llu_t%d";
constexpr uint32_t CHANNEL_NOTIFY_NUM = 1;
constexpr size_t MEM_TAG_LENGTH = 96;

HcclResult GetDataTypeSize(HcclDataType dataType, uint64_t &dataTypeSize)
{
    auto iter = SIZE_TABLE.find(dataType);
    CHK_PRT_RET(iter == SIZE_TABLE.end(), HCCL_ERROR("Unsupported data type[%d]", dataType), HCCL_E_PARA);
    dataTypeSize = iter->second;
    return HCCL_SUCCESS;
}

bool IsAicpuProtocol(CommProtocol protocol)
{
    return protocol == CommProtocol::COMM_PROTOCOL_UBC_CTP || protocol == CommProtocol::COMM_PROTOCOL_UBC_TP
        || protocol == CommProtocol::COMM_PROTOCOL_UBOE;
}

HcclResult GetNetLayers(HcclComm comm, std::vector<uint32_t> &layers)
{
    uint32_t *layerList = nullptr;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layerList, &layerNum));
    CHK_PRT_RET(layerNum == 0 || layerList == nullptr, HCCL_ERROR("Rank graph has no network layer"),
        HCCL_E_NOT_SUPPORT);
    layers.assign(layerList, layerList + layerNum);
    return HCCL_SUCCESS;
}

HcclResult SelectLink(HcclComm comm, uint32_t srcRank, uint32_t dstRank, const std::vector<uint32_t> &layers,
    CommLink &selectedLink)
{
    // The problem topology defines layer 0 as the intra-server full mesh and
    // layer 1 as the inter-server Clos fabric. Prefer them in that order, but
    // only when the layer is actually reported by the rank graph.
    constexpr uint32_t preferredLayers[] = {0, 1};
    for (uint32_t layer : preferredLayers) {
        if (std::find(layers.begin(), layers.end(), layer) == layers.end()) {
            continue;
        }
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        HcclResult ret = HcclRankGraphGetLinks(comm, layer, srcRank, dstRank, &links, &linkNum);
        if (ret != HCCL_SUCCESS) {
            continue;
        }
        for (uint32_t index = 0; index < linkNum; ++index) {
            if (IsAicpuProtocol(links[index].linkAttr.linkProtocol)) {
                selectedLink = links[index];
                return HCCL_SUCCESS;
            }
        }
    }
    // Keep the implementation topology-adaptive if more layers are exposed.
    for (uint32_t layer : layers) {
        if (layer == preferredLayers[0] || layer == preferredLayers[1]) {
            continue;
        }
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        HcclResult ret = HcclRankGraphGetLinks(comm, layer, srcRank, dstRank, &links, &linkNum);
        if (ret != HCCL_SUCCESS) {
            continue;
        }
        for (uint32_t index = 0; index < linkNum; ++index) {
            if (IsAicpuProtocol(links[index].linkAttr.linkProtocol)) {
                selectedLink = links[index];
                return HCCL_SUCCESS;
            }
        }
    }
    HCCL_ERROR("No AICPU-readable link from rank[%u] to rank[%u]", srcRank, dstRank);
    return HCCL_E_NOT_SUPPORT;
}

// Layer-0 rank lists are local to each server. For the supported 8+8 topology,
// the root's group is either this list or its complement in the communicator.
HcclResult MakeRelayPlan(HcclComm comm, uint32_t ranks, uint32_t root, uint64_t bytes,
    const std::vector<uint32_t> &layers, std::vector<uint32_t> &plan)
{
    if (ranks != 16 || bytes < 4ULL * 1024 * 1024 ||
        std::find(layers.begin(), layers.end(), 0) == layers.end()) return HCCL_SUCCESS;
    uint32_t *list = nullptr, count = 0;
    CHK_RET(HcclRankGraphGetRanksByLayer(comm, 0, &list, &count));
    if (list == nullptr || count != 8) return HCCL_SUCCESS;
    std::vector<uint32_t> group(list, list + count);
    std::sort(group.begin(), group.end());
    CHK_PRT_RET(group.back() >= ranks || std::adjacent_find(group.begin(), group.end()) != group.end(),
        HCCL_ERROR("Invalid layer-0 partition"), HCCL_E_PARA);
    const bool rootIsHere = std::binary_search(group.begin(), group.end(), root);
    std::vector<uint32_t> local, remote;
    for (uint32_t rank = 0; rank < ranks; ++rank) {
        if (rank == root) continue;
        const bool sameAsRoot = std::binary_search(group.begin(), group.end(), rank) == rootIsHere;
        (sameAsRoot ? local : remote).push_back(rank);
    }
    if (local.size() != 7 || remote.size() != 8) return HCCL_SUCCESS;
    plan.assign(ranks, INVALID_VALUE_RANKID);
    for (size_t i = 0; i < remote.size(); ++i) plan[remote[i]] = local[i % local.size()];
    return HCCL_SUCCESS;
}

HcclResult BuildChannelDescs(HcclComm comm, uint32_t myRank, uint32_t rankSize, uint32_t root,
    const std::vector<uint32_t> &layers, HcclMemHandle *memHandles, uint32_t memHandleNum,
    std::vector<HcclChannelDesc> &descs, const std::vector<uint32_t> &plan)
{
    descs.reserve(myRank == root ? (rankSize > 0 ? rankSize - 1 : 0) : (rankSize > 1 ? 1 : 0));
    for (uint32_t remoteRank = 0; remoteRank < rankSize; ++remoteRank) {
        const bool relayEdge = !plan.empty() &&
            (plan[myRank] == remoteRank || plan[remoteRank] == myRank);
        if (remoteRank == myRank || (myRank != root && remoteRank != root && !relayEdge)) {
            continue;
        }
        CommLink link;
        CHK_RET(SelectLink(comm, myRank, remoteRank, layers, link));

        HcclChannelDesc desc;
        CHK_RET(HcclChannelDescInit(&desc, 1));
        desc.remoteRank = remoteRank;
        desc.notifyNum = plan.empty() ? CHANNEL_NOTIFY_NUM : 2;
        desc.channelProtocol = link.linkAttr.linkProtocol;
        desc.localEndpoint = link.srcEndpointDesc;
        desc.remoteEndpoint = link.dstEndpointDesc;
        desc.memHandles = memHandles;
        desc.memHandleNum = memHandleNum;
        descs.push_back(desc);
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

    OpParam param{};
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.root = root;
    param.opType = HcclCMDType::HCCL_CMD_SCATTER;
    int32_t tagLength = std::snprintf(param.tag, sizeof(param.tag), SCATTER_TAG_FORMAT, root,
        static_cast<unsigned long long>(recvCount), static_cast<int>(dataType));
    CHK_PRT_RET(tagLength < 0 || static_cast<size_t>(tagLength) >= sizeof(param.tag),
        HCCL_ERROR("Failed to build Scatter tag"), HCCL_E_INTERNAL);

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.myRank >= param.rankSize || root >= param.rankSize,
        HCCL_ERROR("Invalid rank information: rank[%u], rankSize[%u], root[%u]", param.myRank, param.rankSize, root),
        HCCL_E_PARA);
    if (param.myRank == root) {
        CHK_PTR_NULL(sendBuf);
    }

    uint64_t dataTypeSize = 0;
    CHK_RET(GetDataTypeSize(dataType, dataTypeSize));
    CHK_PRT_RET(recvCount > std::numeric_limits<uint64_t>::max() / dataTypeSize,
        HCCL_ERROR("Scatter receive size overflows"), HCCL_E_PARA);
    uint64_t recvBytes = recvCount * dataTypeSize;
    CHK_PRT_RET(recvBytes > std::numeric_limits<uint64_t>::max() / param.rankSize,
        HCCL_ERROR("Scatter input size overflows"), HCCL_E_PARA);
    uint64_t totalBytes = recvBytes * param.rankSize;

    CommEngine aicpuTsEngine = CommEngine::COMM_ENGINE_AICPU_TS;
    CommEngine cpuTsEngine = CommEngine::COMM_ENGINE_CPU_TS;
    CHK_RET(HcclThreadAcquireWithStream(comm, cpuTsEngine, stream, 1, &param.cpuThread));
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &param.cpuThread, aicpuTsEngine, &param.cpuThreadOnAicpu));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, aicpuTsEngine, &ctx, &size) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = size;
        void *hostCtx = nullptr;
        uint64_t hostCtxSize = 0;
        CHK_RET(HcclEngineCtxGet(comm, param.tag, cpuTsEngine, &hostCtx, &hostCtxSize));
        ThreadHandle *aicpuThread = static_cast<ThreadHandle *>(hostCtx);
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));
    } else {
        AlgResourceCtx resCtxHost;
        void *cclBufferAddr = nullptr;
        uint64_t cclBufferSize = 0;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        std::vector<uint32_t> layers;
        if (param.rankSize > 1) {
            CHK_RET(GetNetLayers(comm, layers));
        }

        CHK_RET(MakeRelayPlan(comm, param.rankSize, root, recvBytes, layers, resCtxHost.helperForRank));
        const bool relay = !resCtxHost.helperForRank.empty();
        const bool helper = relay && std::find(resCtxHost.helperForRank.begin(),
            resCtxHost.helperForRank.end(), param.myRank) != resCtxHost.helperForRank.end();
        CHK_PRT_RET(helper && (cclBufferAddr == nullptr || cclBufferSize < 2 * RELAY_CHUNK_BYTES),
            HCCL_ERROR("Relay requires two scratch slots"), HCCL_E_NOT_SUPPORT);

        char outputMemTag[MEM_TAG_LENGTH] = {};
        int32_t outputTagLength = std::snprintf(outputMemTag, sizeof(outputMemTag),
            SCATTER_OUTPUT_MEM_TAG_FORMAT, root, static_cast<unsigned long long>(recvCount),
            static_cast<int>(dataType));
        CHK_PRT_RET(outputTagLength < 0 || static_cast<size_t>(outputTagLength) >= sizeof(outputMemTag),
            HCCL_ERROR("Failed to build Scatter output memory tag"), HCCL_E_INTERNAL);

        std::vector<HcclMemHandle> outputMemHandles;
        if (param.rankSize > 1 && recvBytes > 0) {
            void *localAddr = param.myRank == root ? sendBuf : recvBuf;
            uint64_t localBytes = param.myRank == root ? totalBytes : recvBytes;
            CommMem outputMem{COMM_MEM_TYPE_DEVICE, localAddr, localBytes};
            HcclMemHandle outputMemHandle = nullptr;
            HcclResult registerRet = HcclCommMemReg(comm, outputMemTag, &outputMem, &outputMemHandle);
            if (registerRet == HCCL_SUCCESS && outputMemHandle != nullptr) {
                outputMemHandles.push_back(outputMemHandle);
                resCtxHost.registeredOutput = CommBuffer{localAddr, localBytes};
            } else {
                HCCL_ERROR("Pull memory registration failed, ret[%d]", registerRet);
                return registerRet == HCCL_SUCCESS ? HCCL_E_INTERNAL : registerRet;
            }
        }

        std::vector<HcclChannelDesc> channelDescs;
        CHK_RET(BuildChannelDescs(comm, param.myRank, param.rankSize, root, layers,
            outputMemHandles.empty() ? nullptr : outputMemHandles.data(),
            static_cast<uint32_t>(outputMemHandles.size()), channelDescs, resCtxHost.helperForRank));
        uint32_t expectedChannelNum = param.myRank == root ? param.rankSize - 1 : (param.rankSize > 1 ? 1 : 0);
        if (relay && param.myRank != root) {
            if (resCtxHost.helperForRank[param.myRank] != INVALID_VALUE_RANKID) ++expectedChannelNum;
            expectedChannelNum += static_cast<uint32_t>(std::count(resCtxHost.helperForRank.begin(),
                resCtxHost.helperForRank.end(), param.myRank));
        }
        CHK_PRT_RET(channelDescs.size() != static_cast<size_t>(expectedChannelNum),
            HCCL_ERROR("Unexpected channel count[%zu]", channelDescs.size()), HCCL_E_INTERNAL);

        // Symmetric resource allocation across ranks; distinct channels have distinct owners.
        const uint32_t threadNum = relay ? 2 : 1;
        resCtxHost.threads.resize(threadNum);
        std::vector<ThreadConfig> threadConfigs(threadNum);
        CHK_PRT_RET(ThreadConfigInit(threadConfigs.data(), threadNum) != 0,
            HCCL_ERROR("Failed to initialize thread configurations"), HCCL_E_INTERNAL);
        for (auto &cfg : threadConfigs) cfg.notifyNumPerThread = relay ? 4 : 1;
        CHK_RET(HcclThreadAcquireWithConfig(comm, CommEngine::COMM_ENGINE_AICPU, threadNum, THREAD_TYPE_TS,
            threadConfigs.data(), resCtxHost.threads.data()));
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(
            comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        uint32_t channelNum = static_cast<uint32_t>(channelDescs.size());
        if (channelNum > 0) {
            std::vector<ChannelHandle> channelHandles(channelNum);
            CHK_RET(HcclChannelAcquire(
                comm, aicpuTsEngine, channelDescs.data(), channelNum, channelHandles.data()));
            resCtxHost.channels.resize(channelNum);
            bool rootInputFound = param.myRank == root || recvBytes == 0;
            for (uint32_t index = 0; index < channelNum; ++index) {
                ChannelInfo &channel = resCtxHost.channels[index];
                channel.remoteRank = channelDescs[index].remoteRank;
                channel.notifyNum = channelDescs[index].notifyNum;
                channel.handle = channelHandles[index];
                CHK_RET(HcclChannelGetHcclBuffer(
                    comm, channel.handle, &channel.remoteCclMem.addr, &channel.remoteCclMem.size));

                // Only a receiver needs the root's registered input. The root
                // never writes receiver memory in the pull algorithm.
                const bool needMem = channel.remoteRank == root ||
                    (helper && resCtxHost.helperForRank[channel.remoteRank] == param.myRank);
                if (param.myRank != root && needMem && !outputMemHandles.empty()) {
                    uint32_t remoteMemNum = 0;
                    CommMem *remoteMems = nullptr;
                    char **remoteMemTags = nullptr;
                    HcclResult remoteMemRet = HcclChannelGetRemoteMems(
                        comm, channel.handle, &remoteMemNum, &remoteMems, &remoteMemTags);
                    bool found = false;
                    if (remoteMemRet == HCCL_SUCCESS && remoteMems != nullptr && remoteMemTags != nullptr) {
                        for (uint32_t memIndex = 0; memIndex < remoteMemNum; ++memIndex) {
                            if (remoteMemTags[memIndex] != nullptr
                                && std::strcmp(remoteMemTags[memIndex], outputMemTag) == 0
                                && remoteMems[memIndex].type == COMM_MEM_TYPE_DEVICE
                                && remoteMems[memIndex].addr != nullptr
                                && remoteMems[memIndex].size >= (channel.remoteRank == root ? totalBytes : recvBytes)) {
                                channel.remoteOutput = CommBuffer{remoteMems[memIndex].addr, remoteMems[memIndex].size};
                                found = true;
                                break;
                            }
                        }
                    }
                    CHK_PRT_RET(!found, HCCL_ERROR("Registered peer buffer missing"), HCCL_E_NOT_SUPPORT);
                    if (channel.remoteRank == root) rootInputFound = true;
                }
            }
            CHK_PRT_RET(recvBytes > 0 && !rootInputFound,
                HCCL_ERROR("Registered root input is missing"), HCCL_E_NOT_SUPPORT);
        }

        std::vector<char> seq = resCtxHost.Serialize();
        param.ctxSize = seq.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, aicpuTsEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, aicpuTsEngine, param.tag, seq.data(), seq.size(), 0));
        void *hostCtx = nullptr;
        uint64_t hostCtxSize = sizeof(ThreadHandle);
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, cpuTsEngine, hostCtxSize, &hostCtx));
        CHK_RET(HcclEngineCtxCopy(comm, cpuTsEngine, param.tag, &resCtxHost.aicpuThread, hostCtxSize, 0));
    }

    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}
