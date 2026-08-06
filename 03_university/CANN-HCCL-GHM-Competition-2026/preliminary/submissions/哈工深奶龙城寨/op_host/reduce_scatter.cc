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
constexpr uint32_t CHANNEL_NOTIFY_NUM_PER_BANK = 2;
constexpr uint32_t PARALLEL_BANK_NUM = 2;
constexpr uint32_t CONTROL_THREAD_NUM = 1;
constexpr uint32_t REDUCE_THREAD_NUM = 2;
constexpr uint32_t PAPER_THREAD_NOTIFY_NUM = 1;
constexpr uint64_t SMALL_INPUT_BYTES = 1ULL * 1024 * 1024;
constexpr uint64_t BALANCED_INPUT_THRESHOLD_BYTES = 400ULL * 1024 * 1024;

enum class AlgorithmMode {
    PAPER2_SMALL,
    SINGLE_GLOBAL_TREE,
    BALANCED_GLOBAL_TREE,
};

bool IsPowerOfTwo(uint32_t value)
{
    return value != 0 && (value & (value - 1)) == 0;
}

void AddUniqueRank(std::vector<uint32_t> &ranks, uint32_t rank)
{
    if (std::find(ranks.begin(), ranks.end(), rank) == ranks.end()) {
        ranks.push_back(rank);
    }
}

std::vector<uint32_t> CollectPaperPeers(uint32_t myRank, uint32_t rankSize)
{
    std::vector<uint32_t> peers;
    const uint32_t orderNum = rankSize == 16 ? 2 : 1;
    for (uint32_t orderIdx = 0; orderIdx < orderNum; ++orderIdx) {
        const bool interleaved = orderIdx == 1;
        const uint32_t myLogicalRank = interleaved ? 2 * (myRank % 8) + myRank / 8 : myRank;
        for (uint32_t skip = rankSize / 2; skip > 0; skip /= 2) {
            const uint32_t toLogicalRank = (myLogicalRank + skip) % rankSize;
            const uint32_t fromLogicalRank = (myLogicalRank + rankSize - skip) % rankSize;
            const uint32_t toRank = interleaved ? toLogicalRank / 2 + (toLogicalRank % 2) * 8 : toLogicalRank;
            const uint32_t fromRank =
                interleaved ? fromLogicalRank / 2 + (fromLogicalRank % 2) * 8 : fromLogicalRank;
            AddUniqueRank(peers, toRank);
            AddUniqueRank(peers, fromRank);
        }
    }
    return peers;
}

HcclResult SelectAlgorithmMode(const OpParam &param, AlgorithmMode &mode)
{
    if (param.rankSize == 0 || param.count > std::numeric_limits<uint64_t>::max() / sizeof(float)) {
        return HCCL_E_PARA;
    }
    const uint64_t recvBytes = param.count * sizeof(float);
    if (recvBytes > std::numeric_limits<uint64_t>::max() / param.rankSize) {
        return HCCL_E_PARA;
    }
    const uint64_t inputBytes = recvBytes * param.rankSize;
    if (inputBytes < SMALL_INPUT_BYTES) {
        if (!IsPowerOfTwo(param.rankSize)) {
            HCCL_ERROR("Paper2 small path requires a power-of-two rank size, rankSize[%u]", param.rankSize);
            return HCCL_E_NOT_SUPPORT;
        }
        mode = AlgorithmMode::PAPER2_SMALL;
        return HCCL_SUCCESS;
    }
    mode = inputBytes >= BALANCED_INPUT_THRESHOLD_BYTES ?
        AlgorithmMode::BALANCED_GLOBAL_TREE : AlgorithmMode::SINGLE_GLOBAL_TREE;
    return HCCL_SUCCESS;
}

void FillChannelDescFromLink(
    uint32_t remoteRank, uint32_t notifyNum, const CommLink &link, HcclChannelDesc &desc)
{
    desc.remoteRank = remoteRank;
    desc.notifyNum = notifyNum;
    desc.channelProtocol = link.linkAttr.linkProtocol;
    desc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
    desc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
    desc.localEndpoint.loc = link.srcEndpointDesc.loc;
    desc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
    desc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
    desc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
}

HcclResult BuildChannelDesc(
    HcclComm comm, uint32_t srcRank, uint32_t dstRank, uint32_t notifyNum, HcclChannelDesc &desc)
{
    uint32_t *layers = nullptr;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layers, &layerNum));

    CommLink fallbackLink{};
    bool hasFallback = false;
    for (uint32_t layerIdx = 0; layerIdx < layerNum; ++layerIdx) {
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        HcclResult ret = HcclRankGraphGetLinks(comm, layers[layerIdx], srcRank, dstRank, &links, &linkNum);
        if (ret != HCCL_SUCCESS || links == nullptr) {
            continue;
        }
        for (uint32_t linkIdx = 0; linkIdx < linkNum; ++linkIdx) {
            const CommLink &link = links[linkIdx];
            if (!hasFallback && link.linkAttr.linkProtocol != CommProtocol::COMM_PROTOCOL_RESERVED) {
                fallbackLink = link;
                hasFallback = true;
            }
            if (link.linkAttr.linkProtocol == CommProtocol::COMM_PROTOCOL_UBC_CTP) {
                CHK_RET(HcclChannelDescInit(&desc, 1));
                FillChannelDescFromLink(dstRank, notifyNum, link, desc);
                return HCCL_SUCCESS;
            }
        }
    }

    if (!hasFallback) {
        HCCL_ERROR("No link found from rank %u to rank %u", srcRank, dstRank);
        return HCCL_E_NOT_FOUND;
    }
    CHK_RET(HcclChannelDescInit(&desc, 1));
    FillChannelDescFromLink(dstRank, notifyNum, fallbackLink, desc);
    return HCCL_SUCCESS;
}

HcclResult AcquireAlgorithmResources(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtx)
{
    const uint32_t peerNum = param.rankSize > 1 ? param.rankSize - 1 : 0;
    AlgorithmMode mode = AlgorithmMode::SINGLE_GLOBAL_TREE;
    CHK_RET(SelectAlgorithmMode(param, mode));
    const bool usePaper2Small = mode == AlgorithmMode::PAPER2_SMALL;
    const bool useBalancedOverlap = mode == AlgorithmMode::BALANCED_GLOBAL_TREE;
    const uint32_t bankNum = useBalancedOverlap ? PARALLEL_BANK_NUM : 1;
    std::vector<uint32_t> remoteRanks;
    if (peerNum > 0) {
        if (usePaper2Small) {
            remoteRanks = CollectPaperPeers(param.myRank, param.rankSize);
        } else {
            remoteRanks.reserve(peerNum);
            for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
                if (remoteRank != param.myRank) {
                    remoteRanks.push_back(remoteRank);
                }
            }
        }
    }
    const uint32_t workerNum = bankNum * peerNum;
    const uint32_t channelNotifyNum = usePaper2Small ? CHANNEL_NOTIFY_NUM_PER_BANK :
        bankNum * CHANNEL_NOTIFY_NUM_PER_BANK;
    const uint32_t threadNum = peerNum == 0 ? 1 : usePaper2Small ? 1 :
        (useBalancedOverlap ? CONTROL_THREAD_NUM + REDUCE_THREAD_NUM + workerNum : CONTROL_THREAD_NUM + workerNum);
    const uint32_t notifyNumPerThread = usePaper2Small ? PAPER_THREAD_NOTIFY_NUM : std::max(16U, peerNum);

    resCtx.threads.resize(threadNum);
    CHK_RET(HcclThreadAcquire(comm, CommEngine::COMM_ENGINE_AICPU_TS, threadNum, notifyNumPerThread,
        resCtx.threads.data()));
    resCtx.aicpuThread = resCtx.threads[0];

    void *cclBufferAddr = nullptr;
    uint64_t cclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
    resCtx.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

    if (remoteRanks.empty()) {
        return HCCL_SUCCESS;
    }

    std::vector<HcclChannelDesc> descs(remoteRanks.size());
    std::vector<ChannelHandle> handles(remoteRanks.size());
    for (uint32_t channelIdx = 0; channelIdx < remoteRanks.size(); ++channelIdx) {
        CHK_RET(BuildChannelDesc(
            comm, param.myRank, remoteRanks[channelIdx], channelNotifyNum, descs[channelIdx]));
    }

    CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AICPU_TS, descs.data(), descs.size(),
        handles.data()));
    resCtx.channels.resize(remoteRanks.size());
    for (uint32_t idx = 0; idx < remoteRanks.size(); ++idx) {
        void *remoteBufferAddr = nullptr;
        uint64_t remoteBufferSize = 0;
        CHK_RET(HcclChannelGetHcclBuffer(comm, handles[idx], &remoteBufferAddr, &remoteBufferSize));
        resCtx.channels[idx].remoteRank = remoteRanks[idx];
        resCtx.channels[idx].notifyNum = channelNotifyNum;
        resCtx.channels[idx].handle = handles[idx];
        resCtx.channels[idx].remoteCclMem = CommBuffer{remoteBufferAddr, remoteBufferSize};
    }
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    if (dataType != HCCL_DATA_TYPE_FP32 || op != HCCL_REDUCE_SUM) {
        HCCL_ERROR("Only FP32 SUM is supported, dataType[%d], op[%d]", dataType, op);
        return HCCL_E_NOT_SUPPORT;
    }

    OpParam param;
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    if (param.rankSize == 0) {
        HCCL_ERROR("Invalid rank size 0");
        return HCCL_E_PARA;
    }

    AlgorithmMode mode = AlgorithmMode::SINGLE_GLOBAL_TREE;
    CHK_RET(SelectAlgorithmMode(param, mode));
    const char *tag = mode == AlgorithmMode::PAPER2_SMALL ? "hccl_origin_v7_paper2_small" :
        mode == AlgorithmMode::BALANCED_GLOBAL_TREE ?
        "hccl_origin_v7_balanced_global_tree" : "hccl_origin_v7_single_global_tree";
    int written = std::snprintf(param.tag, sizeof(param.tag), "%s", tag);
    if (written <= 0 || static_cast<size_t>(written) >= sizeof(param.tag)) {
        HCCL_ERROR("Failed to initialize operator tag");
        return HCCL_E_INTERNAL;
    }

    const CommEngine aicpuTsEngine = CommEngine::COMM_ENGINE_AICPU_TS;
    const CommEngine cpuTsEngine = CommEngine::COMM_ENGINE_CPU_TS;
    CHK_RET(HcclThreadAcquireWithStream(comm, cpuTsEngine, stream, 1, &param.cpuThread));
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &param.cpuThread, aicpuTsEngine, &param.cpuThreadOnAicpu));

    void *ctx = nullptr;
    uint64_t ctxSize = 0;
    if (HcclEngineCtxGet(comm, param.tag, aicpuTsEngine, &ctx, &ctxSize) == HCCL_SUCCESS) {
        param.resCtx = ctx;
        param.ctxSize = ctxSize;
        void *hostCtx = nullptr;
        uint64_t hostCtxSize = sizeof(ThreadHandle);
        CHK_RET(HcclEngineCtxGet(comm, param.tag, cpuTsEngine, &hostCtx, &hostCtxSize));
        auto *aicpuThread = static_cast<ThreadHandle *>(hostCtx);
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));
    } else {
        AlgResourceCtx resCtxHost;
        CHK_RET(AcquireAlgorithmResources(comm, param, resCtxHost));
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine,
            &param.aicpuThreadOnCpu));

        std::vector<char> sequence = resCtxHost.Serialize();
        param.ctxSize = sequence.size();
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, aicpuTsEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, aicpuTsEngine, param.tag, sequence.data(), sequence.size(), 0));

        void *hostCtx = nullptr;
        const uint64_t hostCtxSize = sizeof(ThreadHandle);
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, cpuTsEngine, hostCtxSize, &hostCtx));
        CHK_RET(HcclEngineCtxCopy(comm, cpuTsEngine, param.tag, &resCtxHost.aicpuThread, hostCtxSize, 0));
    }

    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}
