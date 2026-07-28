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
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <limits>
#include <vector>

#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

namespace {
constexpr uint32_t EXPECTED_RANK_SIZE = 16;
constexpr uint32_t LOCAL_RANK_SIZE = 8;
constexpr uint32_t THREAD_NUM = 16;
constexpr uint32_t NOTIFY_NUM_PER_THREAD = 16;
constexpr uint32_t CHANNEL_NOTIFY_NUM = 2;
constexpr uint32_t DFX_OP_MODE_OPBASE = 1;

int32_t ProtocolPriority(CommProtocol protocol)
{
    switch (protocol) {
        case COMM_PROTOCOL_UBC_CTP:
            return 0;
        case COMM_PROTOCOL_UBC_TP:
            return 1;
        case COMM_PROTOCOL_UBOE:
            return 2;
        default:
            return std::numeric_limits<int32_t>::max();
    }
}

bool ContainsRank(const std::vector<uint32_t> &ranks, uint32_t rank)
{
    return std::find(ranks.begin(), ranks.end(), rank) != ranks.end();
}

void DestroyEngineContextBestEffort(
    HcclComm comm, const char *tag, CommEngine engine, uint32_t rank)
{
    const HcclResult ret = HcclEngineCtxDestroy(comm, tag, engine);
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING("Rollback engine context failed, rank[%u], engine[%d], ret[%d]", rank,
            static_cast<int32_t>(engine), static_cast<int32_t>(ret));
    }
}

HcclResult CreateResourceContexts(HcclComm comm, const char *tag, CommEngine aicpuTsEngine,
    CommEngine cpuTsEngine, const std::vector<char> &serializedCtx, ThreadHandle aicpuThread,
    uint32_t rank, void *&aicpuCtx)
{
    bool aicpuCtxCreated = false;
    bool cpuCtxCreated = false;
    auto rollback = [&](HcclResult originalRet) -> HcclResult {
        if (cpuCtxCreated) {
            DestroyEngineContextBestEffort(comm, tag, cpuTsEngine, rank);
        }
        if (aicpuCtxCreated) {
            DestroyEngineContextBestEffort(comm, tag, aicpuTsEngine, rank);
        }
        aicpuCtx = nullptr;
        return originalRet;
    };

    const uint64_t serializedSize = static_cast<uint64_t>(serializedCtx.size());
    HcclResult ret = HcclEngineCtxCreate(comm, tag, aicpuTsEngine, serializedSize, &aicpuCtx);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("Create AICPU context failed, rank[%u], ret[%d]", rank, static_cast<int32_t>(ret));
        return ret;
    }
    aicpuCtxCreated = true;
    if (aicpuCtx == nullptr) {
        HCCL_ERROR("AICPU context creation returned nullptr, rank[%u]", rank);
        return rollback(HCCL_E_INTERNAL);
    }

    ret = HcclEngineCtxCopy(comm, aicpuTsEngine, tag, serializedCtx.data(), serializedSize, 0);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("Copy AICPU context failed, rank[%u], ret[%d]", rank, static_cast<int32_t>(ret));
        return rollback(ret);
    }

    void *cpuCtx = nullptr;
    const uint64_t cpuCtxSize = sizeof(ThreadHandle);
    ret = HcclEngineCtxCreate(comm, tag, cpuTsEngine, cpuCtxSize, &cpuCtx);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("Create CPU context failed, rank[%u], ret[%d]", rank, static_cast<int32_t>(ret));
        return rollback(ret);
    }
    cpuCtxCreated = true;
    if (cpuCtx == nullptr) {
        HCCL_ERROR("CPU context creation returned nullptr, rank[%u]", rank);
        return rollback(HCCL_E_INTERNAL);
    }

    ret = HcclEngineCtxCopy(comm, cpuTsEngine, tag, &aicpuThread, cpuCtxSize, 0);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("Copy CPU context failed, rank[%u], ret[%d]", rank, static_cast<int32_t>(ret));
        return rollback(ret);
    }
    return HCCL_SUCCESS;
}

HcclResult ValidateInput(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);

    if (dataType != HCCL_DATA_TYPE_FP32) {
        HCCL_ERROR("Only FP32 is supported, dataType[%d]", static_cast<int32_t>(dataType));
        return HCCL_E_NOT_SUPPORT;
    }
    if (op != HCCL_REDUCE_SUM) {
        HCCL_ERROR("Only SUM is supported, reduceOp[%d]", static_cast<int32_t>(op));
        return HCCL_E_NOT_SUPPORT;
    }
    if (count > std::numeric_limits<uint64_t>::max() / sizeof(float)) {
        HCCL_ERROR("count[%llu] overflows FP32 byte size", static_cast<unsigned long long>(count));
        return HCCL_E_PARA;
    }

    const uint64_t bytes = count * sizeof(float);
    const uintptr_t sendAddr = reinterpret_cast<uintptr_t>(sendBuf);
    const uintptr_t recvAddr = reinterpret_cast<uintptr_t>(recvBuf);
    if (bytes > 0 && sendAddr != recvAddr) {
        const uintptr_t distance = sendAddr < recvAddr ? recvAddr - sendAddr : sendAddr - recvAddr;
        if (distance < bytes) {
            HCCL_ERROR("Partially overlapping send/recv buffers are not supported, send[%p], recv[%p], bytes[%llu]",
                sendBuf, recvBuf, static_cast<unsigned long long>(bytes));
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult QueryTopology(HcclComm comm, uint32_t myRank, uint32_t rankSize, AlgResourceCtx &resCtx)
{
    uint32_t *layers = nullptr;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layers, &layerNum));
    if (layers == nullptr || layerNum == 0) {
        HCCL_ERROR("Rank graph returned no network layer, rank[%u]", myRank);
        return HCCL_E_NOT_FOUND;
    }
    std::vector<uint32_t> netLayers(layers, layers + layerNum);

    bool foundLocalLayer = false;
    bool foundGlobalLayer = false;
    for (uint32_t netLayer : netLayers) {
        uint32_t *ranks = nullptr;
        uint32_t rankNum = 0;
        CHK_RET(HcclRankGraphGetRanksByLayer(comm, netLayer, &ranks, &rankNum));
        if (rankNum > 0 && ranks == nullptr) {
            HCCL_ERROR("Rank graph returned nullptr ranks, rank[%u], layer[%u], rankNum[%u]", myRank, netLayer,
                rankNum);
            return HCCL_E_INTERNAL;
        }

        std::vector<uint32_t> rankList;
        if (rankNum > 0) {
            rankList.assign(ranks, ranks + rankNum);
        }
        std::sort(rankList.begin(), rankList.end());
        if (std::adjacent_find(rankList.begin(), rankList.end()) != rankList.end()) {
            HCCL_ERROR("Duplicate rank in topology, rank[%u], layer[%u]", myRank, netLayer);
            return HCCL_E_PARA;
        }
        for (uint32_t rank : rankList) {
            if (rank >= rankSize) {
                HCCL_ERROR("Invalid topology rank[%u], myRank[%u], layer[%u], rankSize[%u]", rank, myRank,
                    netLayer, rankSize);
                return HCCL_E_PARA;
            }
        }

        HCCL_INFO("Topology rank[%u], layer[%u], rankNum[%u]", myRank, netLayer, rankNum);
        if (!foundLocalLayer && rankNum == LOCAL_RANK_SIZE && ContainsRank(rankList, myRank)) {
            resCtx.localRanks = rankList;
            resCtx.localLayer = netLayer;
            foundLocalLayer = true;
        }
        if (!foundGlobalLayer && rankNum == rankSize && ContainsRank(rankList, myRank)) {
            resCtx.globalLayer = netLayer;
            foundGlobalLayer = true;
        }
    }

    if (!foundLocalLayer || resCtx.localRanks.size() != LOCAL_RANK_SIZE) {
        HCCL_ERROR("Cannot find the 8-rank local topology, rank[%u]", myRank);
        return HCCL_E_NOT_FOUND;
    }
    if (!foundGlobalLayer) {
        HCCL_ERROR("Cannot find the global topology layer, rank[%u], rankSize[%u]", myRank, rankSize);
        return HCCL_E_NOT_FOUND;
    }

    auto localIt = std::find(resCtx.localRanks.begin(), resCtx.localRanks.end(), myRank);
    if (localIt == resCtx.localRanks.end()) {
        HCCL_ERROR("Local topology does not contain myRank[%u]", myRank);
        return HCCL_E_INTERNAL;
    }
    resCtx.localRankIndex = static_cast<uint32_t>(std::distance(resCtx.localRanks.begin(), localIt));

    std::vector<uint32_t> remoteServerRanks;
    remoteServerRanks.reserve(LOCAL_RANK_SIZE);
    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        if (!ContainsRank(resCtx.localRanks, rank)) {
            remoteServerRanks.push_back(rank);
        }
    }
    if (remoteServerRanks.size() != LOCAL_RANK_SIZE) {
        HCCL_ERROR("Invalid remote server rank count[%zu], rank[%u]", remoteServerRanks.size(), myRank);
        return HCCL_E_PARA;
    }

    resCtx.serverIndex = resCtx.localRanks.front() < remoteServerRanks.front() ? 0 : 1;
    resCtx.isLowerServer = resCtx.serverIndex == 0 ? 1 : 0;
    resCtx.pairRank = remoteServerRanks[resCtx.localRankIndex];

    HCCL_INFO("Topology summary rank[%u], localLayer[%u], globalLayer[%u], localIndex[%u], pairRank[%u], "
              "serverIndex[%u]",
        myRank, resCtx.localLayer, resCtx.globalLayer, resCtx.localRankIndex, resCtx.pairRank, resCtx.serverIndex);
    for (uint32_t index = 0; index < resCtx.localRanks.size(); ++index) {
        HCCL_INFO("Local topology rank[%u], localIndex[%u], localRank[%u]", myRank, index, resCtx.localRanks[index]);
    }
    return HCCL_SUCCESS;
}

HcclResult SelectBestLink(
    HcclComm comm, uint32_t myRank, uint32_t remoteRank, uint32_t netLayer, CommLink &selectedLink)
{
    bool found = false;
    int32_t bestPriority = std::numeric_limits<int32_t>::max();
    uint8_t bestHop = std::numeric_limits<uint8_t>::max();
    CommLink *links = nullptr;
    uint32_t linkNum = 0;
    HcclResult ret = HcclRankGraphGetLinks(comm, netLayer, myRank, remoteRank, &links, &linkNum);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("GetLinks failed, rank[%u], peer[%u], layer[%u], ret[%d]", myRank, remoteRank, netLayer,
            static_cast<int32_t>(ret));
        return ret;
    }
    if (linkNum > 0 && links == nullptr) {
        HCCL_ERROR("GetLinks returned nullptr, rank[%u], peer[%u], layer[%u], linkNum[%u]", myRank, remoteRank,
            netLayer, linkNum);
        return HCCL_E_INTERNAL;
    }

    for (uint32_t linkIndex = 0; linkIndex < linkNum; ++linkIndex) {
        const CommLink &candidate = links[linkIndex];
        const int32_t priority = ProtocolPriority(candidate.linkAttr.linkProtocol);
        HCCL_INFO("Link candidate rank[%u], peer[%u], layer[%u], link[%u], protocol[%d], hop[%u]", myRank,
            remoteRank, netLayer, linkIndex, static_cast<int32_t>(candidate.linkAttr.linkProtocol),
            static_cast<uint32_t>(candidate.linkAttr.hop));
        if (priority == std::numeric_limits<int32_t>::max()) {
            continue;
        }
        if (priority < bestPriority || (priority == bestPriority && candidate.linkAttr.hop < bestHop)) {
            selectedLink = candidate;
            bestPriority = priority;
            bestHop = candidate.linkAttr.hop;
            found = true;
        }
    }

    if (!found) {
        HCCL_ERROR(
            "No supported AICPU_TS write link, rank[%u], peer[%u], layer[%u]", myRank, remoteRank, netLayer);
        return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

HcclResult BuildChannels(HcclComm comm, uint32_t myRank, uint32_t rankSize, AlgResourceCtx &resCtx)
{
    const uint32_t channelNum = rankSize - 1;
    std::vector<HcclChannelDesc> channelDescs(channelNum);
    CHK_RET(HcclChannelDescInit(channelDescs.data(), channelNum));
    resCtx.channels.resize(channelNum);

    uint32_t channelIndex = 0;
    for (uint32_t remoteRank = 0; remoteRank < rankSize; ++remoteRank) {
        if (remoteRank == myRank) {
            continue;
        }

        CommLink selectedLink{};
        const uint32_t selectedLayer =
            ContainsRank(resCtx.localRanks, remoteRank) ? resCtx.localLayer : resCtx.globalLayer;
        CHK_RET(SelectBestLink(comm, myRank, remoteRank, selectedLayer, selectedLink));

        HcclChannelDesc &desc = channelDescs[channelIndex];
        desc.remoteRank = remoteRank;
        desc.channelProtocol = selectedLink.linkAttr.linkProtocol;
        desc.localEndpoint = selectedLink.srcEndpointDesc;
        desc.remoteEndpoint = selectedLink.dstEndpointDesc;
        desc.notifyNum = CHANNEL_NOTIFY_NUM;

        ChannelInfo &channelInfo = resCtx.channels[channelIndex];
        channelInfo.remoteRank = remoteRank;
        channelInfo.notifyNum = CHANNEL_NOTIFY_NUM;
        channelInfo.netLayer = selectedLayer;
        channelInfo.protocol = selectedLink.linkAttr.linkProtocol;

        HCCL_INFO("Select link rank[%u], peer[%u], layer[%u], protocol[%d], hop[%u]", myRank, remoteRank,
            selectedLayer, static_cast<int32_t>(selectedLink.linkAttr.linkProtocol),
            static_cast<uint32_t>(selectedLink.linkAttr.hop));
        ++channelIndex;
    }

    std::vector<ChannelHandle> channelHandles(channelNum);
    CHK_RET(HcclChannelAcquire(
        comm, CommEngine::COMM_ENGINE_AICPU_TS, channelDescs.data(), channelNum, channelHandles.data()));

    uint64_t usableBufferSize = resCtx.localBuffer.size;
    for (uint32_t index = 0; index < channelNum; ++index) {
        ChannelInfo &channelInfo = resCtx.channels[index];
        channelInfo.handle = channelHandles[index];
        if (channelInfo.handle == 0) {
            HCCL_ERROR("Invalid channel handle, rank[%u], peer[%u]", myRank, channelInfo.remoteRank);
            return HCCL_E_UNAVAIL;
        }
        void *remoteBuffer = nullptr;
        uint64_t remoteBufferSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, channelInfo.handle, &remoteBuffer, &remoteBufferSize));
        if (remoteBuffer == nullptr || remoteBufferSize == 0) {
            HCCL_ERROR("Invalid channel resource, rank[%u], peer[%u], handle[%llu], remoteBuffer[%p], size[%llu]",
                myRank, channelInfo.remoteRank, static_cast<unsigned long long>(channelInfo.handle), remoteBuffer,
                static_cast<unsigned long long>(remoteBufferSize));
            return HCCL_E_UNAVAIL;
        }
        channelInfo.remoteCclMem = CommBuffer{remoteBuffer, remoteBufferSize};
        usableBufferSize = std::min(usableBufferSize, remoteBufferSize);
        HCCL_INFO("Channel resource rank[%u], peer[%u], handle[%llu], protocol[%d], remoteBuffer[%p], size[%llu]",
            myRank, channelInfo.remoteRank, static_cast<unsigned long long>(channelInfo.handle),
            static_cast<int32_t>(channelInfo.protocol), remoteBuffer,
            static_cast<unsigned long long>(remoteBufferSize));
    }
    if (usableBufferSize == 0) {
        HCCL_ERROR("No usable HCCL buffer, rank[%u]", myRank);
        return HCCL_E_UNAVAIL;
    }
    resCtx.usableBufferSize = usableBufferSize;
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op,
    HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    CHK_RET(ValidateInput(sendBuf, recvBuf, count, dataType, op));

    char commName[COMM_INDENTIFIER_MAX_LENGTH] = {};
    CHK_RET(HcclGetCommName(comm, commName));

    OpParam param{};
    int32_t tagRet = snprintf(param.tag, sizeof(param.tag), "%s%s", "allreduce_res_v3_", commName);
    if (tagRet < 0 || static_cast<std::size_t>(tagRet) >= sizeof(param.tag)) {
        HCCL_ERROR("Failed to build resource tag, commName[%s], ret[%d]", commName, tagRet);
        return HCCL_E_PARA;
    }
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = count;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;

    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    if (param.rankSize != EXPECTED_RANK_SIZE || param.myRank >= param.rankSize) {
        HCCL_ERROR("This implementation requires 16 ranks, myRank[%u], rankSize[%u]", param.myRank, param.rankSize);
        return HCCL_E_NOT_SUPPORT;
    }

    const CommEngine aicpuTsEngine = CommEngine::COMM_ENGINE_AICPU_TS;
    const CommEngine cpuTsEngine = CommEngine::COMM_ENGINE_CPU_TS;

    CHK_RET(HcclThreadAcquireWithStream(comm, cpuTsEngine, stream, 1, &param.cpuThread));
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &param.cpuThread, aicpuTsEngine, &param.cpuThreadOnAicpu));

    void *ctx = nullptr;
    uint64_t size = 0;
    HcclResult ctxRet = HcclEngineCtxGet(comm, param.tag, aicpuTsEngine, &ctx, &size);
    if (ctxRet == HCCL_SUCCESS) {
        if (ctx == nullptr || size == 0) {
            HCCL_ERROR("Invalid cached AICPU context, rank[%u], ctx[%p], size[%llu]", param.myRank, ctx,
                static_cast<unsigned long long>(size));
            return HCCL_E_INTERNAL;
        }
        param.resCtx = ctx;
        param.ctxSize = size;

        void *hostCtx = nullptr;
        uint64_t hostCtxSize = 0;
        CHK_RET(HcclEngineCtxGet(comm, param.tag, cpuTsEngine, &hostCtx, &hostCtxSize));
        if (hostCtx == nullptr || hostCtxSize != sizeof(ThreadHandle)) {
            HCCL_ERROR("Invalid cached CPU context, rank[%u], ctx[%p], size[%llu]", param.myRank, hostCtx,
                static_cast<unsigned long long>(hostCtxSize));
            return HCCL_E_INTERNAL;
        }
        ThreadHandle aicpuThread = *static_cast<ThreadHandle *>(hostCtx);
        if (aicpuThread == 0) {
            HCCL_ERROR("Invalid cached AICPU thread, rank[%u]", param.myRank);
            return HCCL_E_INTERNAL;
        }
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));
        HCCL_INFO("Reuse engine context, rank[%u], ctxSize[%llu]", param.myRank,
            static_cast<unsigned long long>(param.ctxSize));
    } else {
        if (ctxRet != HCCL_E_NOT_FOUND && ctxRet != HCCL_E_PARA) {
            HCCL_ERROR("Get AICPU context failed, rank[%u], ret[%d]", param.myRank, static_cast<int32_t>(ctxRet));
            return ctxRet;
        }
        if (ctxRet == HCCL_E_PARA) {
            HCCL_WARNING("Treat HCCL_E_PARA as a simulator-compatible context cache miss, rank[%u]", param.myRank);
        }
        AlgResourceCtx resCtxHost{};

        void *cclBufferAddr = nullptr;
        uint64_t cclBufferSize = 0;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        if (cclBufferAddr == nullptr || cclBufferSize == 0) {
            HCCL_ERROR("Invalid local HCCL buffer, rank[%u], addr[%p], size[%llu]", param.myRank, cclBufferAddr,
                static_cast<unsigned long long>(cclBufferSize));
            return HCCL_E_UNAVAIL;
        }
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        CHK_RET(QueryTopology(comm, param.myRank, param.rankSize, resCtxHost));

        resCtxHost.threads.resize(THREAD_NUM);
        CHK_RET(HcclThreadAcquire(
            comm, aicpuTsEngine, THREAD_NUM, NOTIFY_NUM_PER_THREAD, resCtxHost.threads.data()));
        for (uint32_t index = 0; index < THREAD_NUM; ++index) {
            if (resCtxHost.threads[index] == 0) {
                HCCL_ERROR("Invalid AICPU thread handle, rank[%u], threadIndex[%u]", param.myRank, index);
                return HCCL_E_UNAVAIL;
            }
        }
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(
            comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        CHK_RET(BuildChannels(comm, param.myRank, param.rankSize, resCtxHost));
        HCCL_INFO("Resource summary rank[%u], threads[%u], threadNotifys[%u], channels[%zu], channelNotifys[%u], "
                  "localBuffer[%llu], usableBuffer[%llu]",
            param.myRank, THREAD_NUM, NOTIFY_NUM_PER_THREAD, resCtxHost.channels.size(), CHANNEL_NOTIFY_NUM,
            static_cast<unsigned long long>(resCtxHost.localBuffer.size),
            static_cast<unsigned long long>(resCtxHost.usableBufferSize));

        std::vector<char> seq = resCtxHost.Serialize();
        if (seq.empty()) {
            HCCL_ERROR("Serialized resource context is empty, rank[%u]", param.myRank);
            return HCCL_E_INTERNAL;
        }
        const uint64_t seqSize = static_cast<uint64_t>(seq.size());
        param.ctxSize = seqSize;
        CHK_RET(CreateResourceContexts(comm, param.tag, aicpuTsEngine, cpuTsEngine, seq,
            resCtxHost.aicpuThread, param.myRank, param.resCtx));
        HCCL_INFO("Create engine context, rank[%u], ctxSize[%llu]", param.myRank,
            static_cast<unsigned long long>(param.ctxSize));
    }

    const uint64_t dataBytes = count * sizeof(float);
    HcclDfxOpInfo dfxInfo{};
    dfxInfo.opMode = DFX_OP_MODE_OPBASE;
    dfxInfo.opType = static_cast<uint32_t>(param.opType);
    dfxInfo.reduceOp = static_cast<uint32_t>(op);
    dfxInfo.dataType = static_cast<uint32_t>(dataType);
    dfxInfo.outputType = static_cast<uint32_t>(dataType);
    dfxInfo.dataCount = count;
    dfxInfo.root = INVALID_VALUE_RANKID;
    dfxInfo.engine = aicpuTsEngine;
    dfxInfo.cpuTsThread = param.cpuThread;
    dfxInfo.cpuWaitAicpuNotifyIdx = 0;
    dfxInfo.inputMemAddr = reinterpret_cast<uint64_t>(sendBuf);
    dfxInfo.inputMemSize = dataBytes;
    dfxInfo.outputMemAddr = reinterpret_cast<uint64_t>(recvBuf);
    dfxInfo.outputMemSize = dataBytes;
    int32_t dfxTagRet = snprintf(dfxInfo.algTag, sizeof(dfxInfo.algTag), "%s", param.tag);
    if (dfxTagRet < 0 || static_cast<std::size_t>(dfxTagRet) >= sizeof(dfxInfo.algTag)) {
        HCCL_ERROR("Failed to build DFX tag, rank[%u], ret[%d]", param.myRank, dfxTagRet);
        return HCCL_E_PARA;
    }
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    HCCL_INFO("Launch tiled hierarchical AllReduce, rank[%u], count[%llu], inPlace[%u]", param.myRank,
        static_cast<unsigned long long>(param.count), sendBuf == recvBuf ? 1U : 0U);
    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}