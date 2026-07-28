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
#include <array>
#include <cstring>
#include <cstdio>
#include <limits>
#include <map>
#include <utility>
#include <vector>

#include <ccu/ccu_launch.h>
#include <hccl/hccl_ccu_res.h>
#include <hccl/hccl_diag.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_res_expt.h>

#include "ccu_kernel.h"
#include "common.h"
#include "custom.h"
#include "exec_op.h"
#include "hccl.h"
#include "log.h"

namespace {

constexpr uint32_t CHANNEL_NOTIFY_NUM = 1;
constexpr uint64_t SEQREUSE_PROBE_MIN_BYTES = 400ULL * 1024ULL * 1024ULL + 4ULL;
constexpr uint64_t TARGET_2X8_512M_BYTES = 512ULL * 1024ULL * 1024ULL;
constexpr uint64_t SMALL_MESSAGE_SCRATCH_WARMUP_BYTES = 512ULL * 1024ULL;
constexpr uint64_t TARGET_SMALL4_BYTES = 512ULL * 1024ULL;

struct SelectedLink {
    CommLink link{};
    uint32_t layer = 0;
    uint32_t dieId = 0;
};

struct ChannelGroup {
    uint32_t dieId = 0;
    std::vector<ChannelHandle> channels;
    std::vector<uint32_t> remoteRanks;
};

bool IsSequentialScratchReuseRequested(const OpParam &param)
{
    return std::strstr(param.tag, "_seqreuse_") != nullptr;
}

bool IsTargetedSmallMessageBalanceRequested(const OpParam &param)
{
    constexpr uint64_t TARGET_BYTES = 512ULL * 1024ULL;
    constexpr uint64_t TARGET_COUNT = TARGET_BYTES / sizeof(float);
    return (param.rankSize == 12 || param.rankSize == 16) &&
        param.dataType == HCCL_DATA_TYPE_FP32 &&
        param.reduceType == HCCL_REDUCE_SUM &&
        param.count == TARGET_COUNT;
}

bool IsExactSmall4Requested(const OpParam &param)
{
    constexpr uint64_t TARGET_COUNT = TARGET_SMALL4_BYTES / sizeof(float);
    return param.rankSize == 4 &&
        param.dataType == HCCL_DATA_TYPE_FP32 &&
        param.reduceType == HCCL_REDUCE_SUM &&
        param.count == TARGET_COUNT;
}

// 2*8 (rankSize=16) 与 8+4 (rankSize=12) 拓扑下，同Server内的对端默认走
// Layer-0 Mesh直连，跨Server对端走Layer-1 Clos，这会把两组channel分别落在
// 2个不同的IO Die上，需要注册2个CCU kernel并通过CPU线程间notify做
// PUBLISH/PARTIAL/COMBINE/FINISH四轮握手同步。对512KB这种小报文，
// 传输本身的耗时远小于这几轮跨线程同步的固定开销，因此本场景下让全部
// 对端都强制走Layer-1（Clos）单层链路，使所有channel都落在同一个IO Die，
// 从而只注册1个CCU kernel、完全消除跨Die线程同步。
// 仅在512KB且rankSize为12/16时触发；512MB、400MB+4B等其它用例的tag
// 与此不同（见下方tag拼接），resource context彼此独立缓存，互不影响。
bool IsSmallMessageDualDieFastPathRequested(const OpParam &param)
{
    return IsTargetedSmallMessageBalanceRequested(param);
}

HcclResult SelectLinkForPeer(HcclComm comm, uint32_t myRank, uint32_t peerRank,
    const std::vector<uint32_t> &layers, SelectedLink &selection)
{
    const CommProtocol priorities[] = {COMM_PROTOCOL_UBC_CTP, COMM_PROTOCOL_UBC_TP};
    for (const CommProtocol protocol : priorities) {
        for (uint32_t layer : layers) {
            CommLink *links = nullptr;
            uint32_t linkNum = 0;
            CHK_RET(HcclRankGraphGetLinks(comm, layer, myRank, peerRank, &links, &linkNum));
            for (uint32_t linkIdx = 0; linkIdx < linkNum; ++linkIdx) {
                if (links[linkIdx].linkAttr.linkProtocol != protocol) {
                    continue;
                }
                EndpointAttrDieId dieId = 0;
                CHK_RET(HcclRankGraphGetEndpointInfo(comm, myRank,
                    &links[linkIdx].srcEndpointDesc, ENDPOINT_ATTR_DIE_ID,
                    sizeof(dieId), &dieId));
                selection.link = links[linkIdx];
                selection.layer = layer;
                selection.dieId = dieId;
                return HCCL_SUCCESS;
            }
        }
    }
    HCCL_ERROR("No CCU-compatible link between rank %u and rank %u", myRank, peerRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult AcquireChannelGroups(HcclComm comm, const OpParam &param,
    std::vector<ChannelGroup> &groups, std::vector<uint32_t> &channelLayers)
{
    uint32_t *layerList = nullptr;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layerList, &layerNum));
    CHK_PRT_RET(layerList == nullptr || layerNum == 0,
        HCCL_ERROR("Rank graph contains no network layer"), HCCL_E_NOT_FOUND);
    std::vector<uint32_t> layers(layerList, layerList + layerNum);

    // 小报文单Die快速路径：所有对端都只在Clos（数值最大的layer）上选链路，
    // 这样每个peer解析出的dieId都相同，AcquireChannelGroups最终只会产生
    // 1个ChannelGroup，后续RegisterKernels/CreateResourceContext也就只
    // 注册1个CCU kernel，彻底跳过双Die线程握手。
    if (layers.size() > 1 && IsSmallMessageDualDieFastPathRequested(param)) {
        const uint32_t closLayer = *std::max_element(layers.begin(), layers.end());
        layers.assign(1, closLayer);
    }

    std::map<uint32_t, ChannelGroup> groupsByDie;
    channelLayers.reserve(param.rankSize - 1);
    for (uint32_t peerRank = 0; peerRank < param.rankSize; ++peerRank) {
        if (peerRank == param.myRank) {
            continue;
        }
        SelectedLink selection;
        CHK_RET(SelectLinkForPeer(comm, param.myRank, peerRank, layers, selection));

        HcclChannelDesc desc;
        CHK_RET(HcclChannelDescInit(&desc, 1));
        desc.remoteRank = peerRank;
        desc.notifyNum = CHANNEL_NOTIFY_NUM;
        desc.channelProtocol = selection.link.linkAttr.linkProtocol;
        desc.localEndpoint = selection.link.srcEndpointDesc;
        desc.remoteEndpoint = selection.link.dstEndpointDesc;

        ChannelHandle channel = 0;
        CHK_RET(HcclChannelAcquire(
            comm, CommEngine::COMM_ENGINE_CCU, &desc, 1, &channel));
        ChannelGroup &group = groupsByDie[selection.dieId];
        group.dieId = selection.dieId;
        group.channels.push_back(channel);
        group.remoteRanks.push_back(peerRank);
        channelLayers.push_back(selection.layer);
    }

    CHK_PRT_RET(groupsByDie.empty() || groupsByDie.size() > MAX_CCU_DIE_GROUPS,
        HCCL_ERROR("Invalid CCU IO Die group count: %zu", groupsByDie.size()),
        HCCL_E_NOT_SUPPORT);
    groups.reserve(groupsByDie.size());
    for (auto &entry : groupsByDie) {
        groups.push_back(std::move(entry.second));
    }
    if ((IsSequentialScratchReuseRequested(param) ||
        IsTargetedSmallMessageBalanceRequested(param)) &&
        groups.size() == MAX_CCU_DIE_GROUPS &&
        groups[0].channels.size() > groups[1].channels.size()) {
        std::swap(groups[0], groups[1]);
    }
    return HCCL_SUCCESS;
}

HcclResult RegisterSmall4Kernel(HcclComm comm, const OpParam &param,
    const std::vector<ChannelGroup> &groups, Small4ResourceCtx &resourceCtx)
{
    CHK_PRT_RET(groups.size() != 1 || groups[0].channels.size() != 3,
        HCCL_ERROR("4x1 small AllReduce requires one three-peer IO Die group"),
        HCCL_E_NOT_SUPPORT);

    CcuInsHandle insHandle = 0;
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1,
        HCCL_ERROR("Expected exactly one CCU instance, got %u", insNum), HCCL_E_UNAVAIL);

    Small4KernelArg kernelArg{};
    kernelArg.dataType = param.dataType;
    kernelArg.reduceType = param.reduceType;
    kernelArg.inPlace = param.inputPtr == param.outputPtr ? 1 : 0;
    kernelArg.channelCount = 3;
    for (uint32_t idx = 0; idx < kernelArg.channelCount; ++idx) {
        kernelArg.channels[idx] = groups[0].channels[idx];
    }

    resourceCtx.kernelDieIds.resize(1);
    resourceCtx.ccuKernels.resize(1);
    resourceCtx.kernelDieIds[0] = groups[0].dieId;
    const void *kernelArgs[] = {&kernelArg};
    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));
    CHK_RET_CCU(HcommCcuKernelRegister(insHandle, groups[0].dieId,
        "CcuSmall4AllPairsKernel",
        reinterpret_cast<const void *>(ops_hccl::CcuSmall4AllPairsKernel),
        kernelArgs, 1, &resourceCtx.ccuKernels[0]));
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
    return HCCL_SUCCESS;
}

HcclResult CreateSmall4ResourceContext(HcclComm comm, OpParam &param)
{
    Small4ResourceCtx resourceCtx;
    resourceCtx.rankSize = param.rankSize;
    resourceCtx.dataType = param.dataType;
    resourceCtx.reduceType = param.reduceType;

    void *cclBuffer = nullptr;
    uint64_t cclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBuffer, &cclBufferSize));
    constexpr uint64_t requiredScratchBytes = 3 * TARGET_SMALL4_BYTES;
    CHK_PRT_RET(cclBuffer == nullptr || cclBufferSize < requiredScratchBytes,
        HCCL_ERROR("HCCL buffer is too small for 4x1 small path: %llu",
            static_cast<unsigned long long>(cclBufferSize)), HCCL_E_UNAVAIL);
    resourceCtx.localBuffer = CommBuffer{cclBuffer, cclBufferSize};

    std::vector<ChannelGroup> groups;
    std::vector<uint32_t> channelLayers;
    CHK_RET(AcquireChannelGroups(comm, param, groups, channelLayers));
    CHK_RET(RegisterSmall4Kernel(comm, param, groups, resourceCtx));

    const std::vector<char> sequence = resourceCtx.Serialize();
    CHK_PRT_RET(sequence.empty(),
        HCCL_ERROR("Serialized 4x1 small resource context is empty"), HCCL_E_INTERNAL);
    param.ctxSize = sequence.size();
    CHK_RET(HcclEngineCtxCreate(
        comm, param.tag, CommEngine::COMM_ENGINE_CCU, param.ctxSize, &param.resCtx));
    CHK_RET(HcclEngineCtxCopy(
        comm, CommEngine::COMM_ENGINE_CCU, param.tag, sequence.data(), sequence.size(), 0));
    return HCCL_SUCCESS;
}

HcclResult GetOrCreateSmall4ResourceContext(HcclComm comm, OpParam &param)
{
    void *ctx = nullptr;
    uint64_t size = 0;
    const HcclResult getRet = HcclEngineCtxGet(
        comm, param.tag, CommEngine::COMM_ENGINE_CCU, &ctx, &size);
    if (getRet == HCCL_SUCCESS) {
        CHK_PRT_RET(ctx == nullptr || size == 0,
            HCCL_ERROR("Cached 4x1 small engine context is invalid"), HCCL_E_INTERNAL);
        param.resCtx = ctx;
        param.ctxSize = size;
        return HCCL_SUCCESS;
    }
    HCCL_INFO("4x1 small engine context cache miss, HcclEngineCtxGet ret=%d", getRet);
    return CreateSmall4ResourceContext(comm, param);
}

HcclResult RegisterKernels(HcclComm comm, const OpParam &param,
    const std::vector<ChannelGroup> &groups, AlgResourceCtx &resourceCtx)
{
    CcuInsHandle insHandle = 0;
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1,
        HCCL_ERROR("Expected exactly one CCU instance, got %u", insNum), HCCL_E_UNAVAIL);

    std::array<uint32_t, MAX_CCU_DIE_GROUPS> sourceCounts{};
    std::array<uint32_t, MAX_CCU_DIE_GROUPS> scratchBaseSlots{};
    for (uint32_t groupIdx = 0; groupIdx < groups.size(); ++groupIdx) {
        const uint32_t includesLocal = groupIdx == 0 ? 1 : 0;
        const uint32_t sourceCount =
            static_cast<uint32_t>(groups[groupIdx].channels.size()) + includesLocal;
        CHK_PRT_RET(sourceCount == 0 || sourceCount > MAX_RANK_SIZE,
            HCCL_ERROR("Invalid source count %u for CCU Die group %u", sourceCount, groupIdx),
            HCCL_E_INTERNAL);
        sourceCounts[groupIdx] = sourceCount;
    }

    const bool sequentialScratchReuse =
        IsSequentialScratchReuseRequested(param) && groups.size() == MAX_CCU_DIE_GROUPS;
    const uint32_t outputRootUsesLocalInput = param.inputPtr == param.outputPtr ? 1 : 0;
    uint32_t scratchSlotCount = 0;
    if (sequentialScratchReuse) {
        const uint32_t localChannelCount = static_cast<uint32_t>(groups[0].channels.size());
        const uint32_t nonLocalChannelCount = static_cast<uint32_t>(groups[1].channels.size());
        scratchBaseSlots[0] = 1;
        scratchBaseSlots[1] = 0;
        scratchSlotCount = std::max(nonLocalChannelCount, 1U + localChannelCount);
        CHK_PRT_RET(scratchSlotCount < 2 || scratchSlotCount > param.rankSize - 1,
            HCCL_ERROR("Invalid sequential scratch layout: localChannels=%u, "
                "nonLocalChannels=%u, slots=%u, rankSize=%u", localChannelCount,
                nonLocalChannelCount, scratchSlotCount, param.rankSize), HCCL_E_INTERNAL);
    } else {
        uint32_t nextScratchSlot = 0;
        for (uint32_t groupIdx = 0; groupIdx < groups.size(); ++groupIdx) {
            scratchBaseSlots[groupIdx] = nextScratchSlot;
            nextScratchSlot += static_cast<uint32_t>(groups[groupIdx].channels.size());
        }
        CHK_PRT_RET(nextScratchSlot != param.rankSize - 1,
            HCCL_ERROR("Scratch source layout mismatch: slots=%u, rankSize=%u",
                nextScratchSlot, param.rankSize), HCCL_E_INTERNAL);
        scratchSlotCount = nextScratchSlot;
    }
    resourceCtx.scratchSlotCount = scratchSlotCount;
    resourceCtx.sequentialScratchReuse = sequentialScratchReuse ? 1 : 0;

    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));
    resourceCtx.ccuKernels.resize(groups.size());
    resourceCtx.kernelDieIds.resize(groups.size());
    for (uint32_t groupIdx = 0; groupIdx < groups.size(); ++groupIdx) {
        const ChannelGroup &group = groups[groupIdx];
        char kernelName[64]{};
        const int nameLen = std::snprintf(kernelName, sizeof(kernelName),
            "CcuHostSyncTreeAllReduceKernelDie%u", group.dieId);
        CHK_PRT_RET(nameLen < 0 || static_cast<size_t>(nameLen) >= sizeof(kernelName),
            HCCL_ERROR("Failed to build CCU kernel name"), HCCL_E_INTERNAL);

        if (sequentialScratchReuse) {
            SequentialScratchReuseKernelArg kernelArg{};
            AllReduceKernelArg &baseArg = kernelArg;
            baseArg.rankSize = param.rankSize;
            baseArg.rankId = param.myRank;
            baseArg.dataType = param.dataType;
            baseArg.reduceType = param.reduceType;
            baseArg.groupIndex = groupIdx;
            baseArg.groupCount = static_cast<uint32_t>(groups.size());
            baseArg.includesLocalRank = groupIdx == 0 ? 1 : 0;
            baseArg.sourceCount = sourceCounts[groupIdx];
            baseArg.scratchBaseSlot = scratchBaseSlots[groupIdx];
            baseArg.channelCount = static_cast<uint32_t>(group.channels.size());
            for (uint32_t idx = 0; idx < baseArg.channelCount; ++idx) {
                baseArg.channels[idx] = group.channels[idx];
                baseArg.remoteRanks[idx] = group.remoteRanks[idx];
            }
            baseArg.partialSlotIndices[0] = scratchBaseSlots[0];
            baseArg.partialSlotIndices[1] = scratchBaseSlots[1];
            kernelArg.scratchSlotCount = resourceCtx.scratchSlotCount;
            kernelArg.outputRootUsesLocalInput =
                param.inputPtr == param.outputPtr ? 1 : 0;

            const void *kernelArgs[] = {&kernelArg};
            CHK_RET_CCU(HcommCcuKernelRegister(insHandle, group.dieId, kernelName,
                reinterpret_cast<const void *>(
                    ops_hccl::CcuPhasedMeshAllReduceSequentialScratchReuseKernel),
                kernelArgs, 1, &resourceCtx.ccuKernels[groupIdx]));
        } else {
            AllReduceKernelArg kernelArg{};
            kernelArg.rankSize = param.rankSize;
            kernelArg.rankId = param.myRank;
            kernelArg.dataType = param.dataType;
            kernelArg.reduceType = param.reduceType;
            kernelArg.groupIndex = groupIdx;
            kernelArg.groupCount = static_cast<uint32_t>(groups.size());
            kernelArg.includesLocalRank = groupIdx == 0 ? 1 : 0;
            kernelArg.sourceCount = sourceCounts[groupIdx];
            kernelArg.scratchBaseSlot = scratchBaseSlots[groupIdx];
            kernelArg.channelCount = static_cast<uint32_t>(group.channels.size());
            for (uint32_t idx = 0; idx < kernelArg.channelCount; ++idx) {
                kernelArg.channels[idx] = group.channels[idx];
                kernelArg.remoteRanks[idx] = group.remoteRanks[idx];
            }
            for (uint32_t idx = 0; idx < groups.size(); ++idx) {
                kernelArg.partialSlotIndices[idx] = scratchBaseSlots[idx];
            }
            kernelArg.outputRootUsesLocalInput = outputRootUsesLocalInput;

            const void *kernelArgs[] = {&kernelArg};
            CHK_RET_CCU(HcommCcuKernelRegister(insHandle, group.dieId, kernelName,
                reinterpret_cast<const void *>(ops_hccl::CcuPhasedMeshAllReduceKernel),
                kernelArgs, 1, &resourceCtx.ccuKernels[groupIdx]));
        }
        resourceCtx.kernelDieIds[groupIdx] = group.dieId;
    }
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
    return HCCL_SUCCESS;
}

HcclResult CreateResourceContext(HcclComm comm, OpParam &param)
{
    AlgResourceCtx resourceCtx;
    resourceCtx.rankSize = param.rankSize;
    resourceCtx.dataType = param.dataType;
    resourceCtx.reduceType = param.reduceType;

    void *cclBuffer = nullptr;
    uint64_t cclBufferSize = 0;
    CHK_RET(HcclGetHcclBuffer(comm, &cclBuffer, &cclBufferSize));
    CHK_PRT_RET(cclBuffer == nullptr || cclBufferSize == 0,
        HCCL_ERROR("HCCL buffer is empty"), HCCL_E_UNAVAIL);
    resourceCtx.localBuffer = CommBuffer{cclBuffer, cclBufferSize};

    std::vector<ChannelGroup> groups;
    CHK_RET(AcquireChannelGroups(comm, param, groups, resourceCtx.channelLayers));
    if (groups.size() == MAX_CCU_DIE_GROUPS) {
        CHK_RET(HcclThreadAcquire(comm, CommEngine::COMM_ENGINE_CCU, 1,
            CCU_THREAD_NOTIFY_NUM, &resourceCtx.dualDieThread));
        CHK_PRT_RET(resourceCtx.dualDieThread == 0,
            HCCL_ERROR("Failed to acquire the second CCU thread"), HCCL_E_UNAVAIL);
    }
    CHK_RET(RegisterKernels(comm, param, groups, resourceCtx));

    const std::vector<char> sequence = resourceCtx.Serialize();
    CHK_PRT_RET(sequence.empty(),
        HCCL_ERROR("Serialized resource context is empty"), HCCL_E_INTERNAL);
    param.ctxSize = sequence.size();
    CHK_RET(HcclEngineCtxCreate(
        comm, param.tag, CommEngine::COMM_ENGINE_CCU, param.ctxSize, &param.resCtx));
    CHK_RET(HcclEngineCtxCopy(
        comm, CommEngine::COMM_ENGINE_CCU, param.tag, sequence.data(), sequence.size(), 0));
    return HCCL_SUCCESS;
}

HcclResult GetOrCreateResourceContext(HcclComm comm, OpParam &param)
{
    void *ctx = nullptr;
    uint64_t size = 0;
    const HcclResult getRet = HcclEngineCtxGet(
        comm, param.tag, CommEngine::COMM_ENGINE_CCU, &ctx, &size);
    if (getRet == HCCL_SUCCESS) {
        CHK_PRT_RET(ctx == nullptr || size == 0,
            HCCL_ERROR("Cached engine context is invalid"), HCCL_E_INTERNAL);
        param.resCtx = ctx;
        param.ctxSize = size;
        return HCCL_SUCCESS;
    }
    HCCL_INFO("Engine context cache miss, HcclEngineCtxGet ret=%d", getRet);
    return CreateResourceContext(comm, param);
}

} // namespace

HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count,
    HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    if (count == 0) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);

    uint32_t dataTypeSize = 0;
    CHK_PRT_RET(!GetCcuDataTypeSize(dataType, dataTypeSize),
        HCCL_ERROR("CCU AllReduce does not support dataType=%d", dataType), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(!IsCcuReduceOpSupported(op),
        HCCL_ERROR("CCU AllReduce does not support reduceType=%d", op), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(count > std::numeric_limits<uint64_t>::max() / dataTypeSize,
        HCCL_ERROR("AllReduce byte count overflows uint64"), HCCL_E_PARA);

    OpParam param{};
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = count;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE || param.myRank >= param.rankSize,
        HCCL_ERROR("Invalid rank graph: rank=%u, rankSize=%u", param.myRank, param.rankSize),
        HCCL_E_PARA);

    const uint64_t totalBytes = count * dataTypeSize;
    const bool isExactSmall4 = IsExactSmall4Requested(param);
    const bool isTarget2x8_512M =
        param.rankSize == 16 &&
        totalBytes == TARGET_2X8_512M_BYTES &&
        dataType == HCCL_DATA_TYPE_FP32 &&
        op == HCCL_REDUCE_SUM;
    bool requestSequentialScratchReuse = false;
    const bool probeSequentialScratchReuse =
        totalBytes >= SEQREUSE_PROBE_MIN_BYTES && !isTarget2x8_512M;
    const bool warmupSmallMessageScratch =
        totalBytes == SMALL_MESSAGE_SCRATCH_WARMUP_BYTES;
    if (param.rankSize == 16 &&
        (probeSequentialScratchReuse || warmupSmallMessageScratch)) {
        void *cclBuffer = nullptr;
        uint64_t cclBufferSize = 0;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBuffer, &cclBufferSize));
        CHK_PRT_RET(cclBuffer == nullptr || cclBufferSize == 0,
            HCCL_ERROR("HCCL buffer is empty"), HCCL_E_UNAVAIL);
        const uint64_t maxConcurrentSliceCount =
            cclBufferSize / dataTypeSize / (param.rankSize - 1);
        const uint64_t maxConcurrentChunkCount =
            maxConcurrentSliceCount * param.rankSize;
        requestSequentialScratchReuse = probeSequentialScratchReuse &&
            count > maxConcurrentChunkCount;
    }

    int tagLen = 0;
    if (requestSequentialScratchReuse) {
        const uint32_t inPlace = sendBuf == recvBuf ? 1 : 0;
        tagLen = std::snprintf(param.tag, sizeof(param.tag),
            "hccl_custom_allreduce_ccu_hostsync_tree_b_ctx10_seqroot_seqreuse_ip%u_dt%u_op%u_n%u",
            inPlace, static_cast<uint32_t>(dataType), static_cast<uint32_t>(op),
            param.rankSize);
    } else if (isExactSmall4) {
        const uint32_t inPlace = sendBuf == recvBuf ? 1 : 0;
        tagLen = std::snprintf(param.tag, sizeof(param.tag),
            "hccl_custom_allreduce_ccu_small4_1pa_iso_v1_ip%u",
            inPlace);
    } else if (IsSmallMessageDualDieFastPathRequested(param)) {
        // 独立tag，确保该场景创建的（单Die/单kernel）resource context
        // 只被512KB这一个用例命中，不会被同一rankSize/dtype/op下的
        // 512MB、400MB+4B等用例复用，也不会反过来影响它们已有的
        // 双Die resource context缓存。
        const uint32_t inPlace = sendBuf == recvBuf ? 1 : 0;
        tagLen = std::snprintf(param.tag, sizeof(param.tag),
            "hccl_custom_allreduce_ccu_hostsync_tree_b_ctx9_outputroot_smallfast_ip%u_dt%u_op%u_n%u",
            inPlace, static_cast<uint32_t>(dataType), static_cast<uint32_t>(op),
            param.rankSize);
    } else {
        const uint32_t inPlace = sendBuf == recvBuf ? 1 : 0;
        tagLen = std::snprintf(param.tag, sizeof(param.tag),
            "hccl_custom_allreduce_ccu_hostsync_tree_b_ctx9_outputroot_ip%u_dt%u_op%u_n%u",
            inPlace, static_cast<uint32_t>(dataType), static_cast<uint32_t>(op),
            param.rankSize);
    }
    CHK_PRT_RET(tagLen < 0 || static_cast<size_t>(tagLen) >= sizeof(param.tag),
        HCCL_ERROR("Failed to build resource context tag"), HCCL_E_INTERNAL);

    HcclDfxOpInfo dfxInfo{};
    char commName[COMM_INDENTIFIER_MAX_LENGTH]{};
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));
    CHK_RET(HcclThreadAcquireWithStream(
        comm, CommEngine::COMM_ENGINE_CCU, stream,
        CCU_THREAD_NOTIFY_NUM, &param.cpuThread));

    if (param.rankSize == 1) {
        if (sendBuf != recvBuf) {
            CHK_RET(static_cast<HcclResult>(
                HcommLocalCopyOnThread(param.cpuThread, recvBuf, sendBuf, totalBytes)));
        }
        return HCCL_SUCCESS;
    }

    if (isExactSmall4) {
        CHK_RET(GetOrCreateSmall4ResourceContext(comm, param));
        return ops_hccl::ExecSmall4Op(param);
    }

    CHK_RET(GetOrCreateResourceContext(comm, param));
    return ops_hccl::ExecOp(param);
}
