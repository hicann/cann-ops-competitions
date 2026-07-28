/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel.h"
#include "log.h"

#define CCU_KERNEL_CHK_RET(call) \
    do { \
        CcuResult ccuRet = static_cast<CcuResult>(call); \
        if (UNLIKELY(ccuRet != CCU_SUCCESS)) { \
            HCCL_ERROR("[%s] call trace: ccuRet -> %d", __func__, ccuRet); \
            return ccuRet; \
        } \
    } while (0)

namespace ops_hccl {
namespace {
constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0;
constexpr int SCATTER_SYNC_ID = 3;
constexpr int FINISH_SYNC_ID = 4;
constexpr int DIRECT_FINISH_SYNC_ID = 5;
constexpr int NHR_POST_SYNC_ID = 3;
constexpr int NHR_SCATTER_STEP_SYNC_ID = 4;
constexpr int NHR_ALLGATHER_STEP_SYNC_ID = 5;
constexpr uint32_t NHR_EVENT_BIT_NUM = 16;
constexpr int HIERARCHY_SCATTER_READY_ID = 3;
constexpr int HIERARCHY_GROUP_FINISH_ID = 4;
constexpr uint64_t DIRECT_GROUP_DATA_SIZE = 512ULL * 1024;
constexpr uint64_t DIRECT_GROUP_SLICE_SIZE = 4096;
constexpr uint32_t DIRECT_GROUP_LOOP_COUNT = 128;
constexpr uint32_t DIRECT_GROUP_MS_INTERLEAVE = 8;
static_assert(DIRECT_GROUP_SLICE_SIZE * DIRECT_GROUP_LOOP_COUNT == DIRECT_GROUP_DATA_SIZE,
    "direct LoopGroup must cover exactly 512KB");

constexpr uint64_t MakeDirectBitMask(uint16_t end)
{
    return (uint64_t{1} << (end + 1)) - uint64_t{1};
}

constexpr uint64_t MakeDirectLoopParam(uint64_t loopCtxId, uint64_t gsaOffset, uint64_t loopIterNum)
{
    return ((loopCtxId & MakeDirectBitMask(8)) << 45) |
        ((gsaOffset & MakeDirectBitMask(32)) << 13) |
        (loopIterNum & MakeDirectBitMask(13));
}

constexpr uint64_t MakeDirectParallelParam(
    uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum)
{
    return ((repeatNum & MakeDirectBitMask(7)) << 55) |
        ((repeatLoopIndex & MakeDirectBitMask(7)) << 48) |
        ((totalLoopNum & MakeDirectBitMask(7)) << 41);
}

constexpr uint64_t MakeDirectOffsetParam(uint64_t gsaOffset, uint64_t msOffset, uint64_t ckeOffset)
{
    return ((gsaOffset & MakeDirectBitMask(32)) << 21) |
        ((msOffset & MakeDirectBitMask(11)) << 10) |
        (ckeOffset & MakeDirectBitMask(10));
}

struct BroadcastHierarchicalContext {
    const CcuBroadcastHierarchicalKernelArg *arg = nullptr;

    ccu::Variable myInput;
    ccu::Variable myOutput;
    ccu::Variable myToken;
    ccu::Variable firstNormalLaneSize;
    ccu::Variable firstLastLaneSize;
    ccu::Variable firstNormalSourceSize;
    ccu::Variable firstNormalDestinationSize;
    ccu::Variable firstLastSourceSize;
    ccu::Variable firstLastDestinationSize;
    ccu::Variable secondSourceHalfSize;
    ccu::Variable secondDestinationHalfSize;
    ccu::Variable secondSourceNormalLaneSize;
    ccu::Variable secondSourceLastLaneSize;
    ccu::Variable secondDestinationNormalLaneSize;
    ccu::Variable secondDestinationLastLaneSize;
    ccu::Variable secondOffset;
    ccu::Variable phase;
    ccu::Variable zeroOffset;
    std::vector<ccu::Variable> remoteOutput;
    std::vector<ccu::Variable> remoteToken;
    std::vector<ccu::Variable> sliceOffsets;
    ccu::Event event;
    ccu::LocalAddr localSrc;
    std::vector<ccu::LocalAddr> scatterSrc;
    std::vector<ccu::RemoteAddr> remoteDst;
    std::vector<ccu::RemoteAddr> laneRemoteDst;
};

struct BroadcastParallelDirectContext {
    const CcuBroadcastParallelDirectKernelArg *arg = nullptr;
    ccu::Variable myInput;
    ccu::Variable myOutput;
    ccu::Variable myToken;
    ccu::Variable dataSize;
    std::vector<ccu::Variable> remoteOutput;
    std::vector<ccu::Variable> remoteToken;
    ccu::Event event;
    ccu::LocalAddr localSrc;
    std::vector<ccu::RemoteAddr> remoteDst;
};

uint32_t GetChannelIndex(const CcuBroadcastMesh1DMem2MemKernelArg *arg, uint32_t peerRank)
{
    return (peerRank < arg->rankId) ? peerRank : (peerRank - 1);
}

CcuResult InitResource(BroadcastMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    if (arg->rankSize <= 1 || arg->rankSize > MAX_RANK_SIZE) {
        HCCL_ERROR("[InitResource] invalid rankSize[%llu]", static_cast<unsigned long long>(arg->rankSize));
        return CcuResult::CCU_E_PARA;
    }
    if (arg->channelCount != arg->rankSize - 1) {
        HCCL_ERROR("[InitResource] channelCount[%u] does not match rankSize[%llu]", arg->channelCount,
            static_cast<unsigned long long>(arg->rankSize));
        return CcuResult::CCU_E_PARA;
    }

    ctx.remoteOutput.reserve(arg->rankSize - 1);
    ctx.remoteToken.reserve(arg->rankSize - 1);
    for (uint64_t peerId = 0; peerId < arg->rankSize; ++peerId) {
        if (peerId == arg->rankId) {
            continue;
        }
        const uint32_t channelIdx = GetChannelIndex(arg, static_cast<uint32_t>(peerId));
        ctx.remoteOutput.push_back(ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], OUTPUT_XN_ID));
        ctx.remoteToken.push_back(ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID));
    }

    ctx.scatterSrc.resize(arg->rankSize);
    ctx.remoteDst.resize(arg->rankSize);
    ctx.nhrSliceOffset.resize(arg->rankSize);
    return CCU_SUCCESS;
}

CcuResult LoadArgs(BroadcastMesh1DMem2MemContext &ctx)
{
    uint32_t argId = 0;
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.myInput, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.myOutput, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.myToken, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.normalSliceSize, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.lastSliceSize, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.allgatherOffset, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.secondNormalSliceSize, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.secondLastSliceSize, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.secondAllgatherOffset, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.secondOffset, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.secondDataSize, argId++));
    ctx.zeroOffset = 0;
    return CCU_SUCCESS;
}

CcuResult LoadDirectArgs(BroadcastMesh1DMem2MemContext &ctx)
{
    uint32_t argId = 0;
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.myInput, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.myOutput, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.myToken, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.dataSize, argId++));
    return CCU_SUCCESS;
}

CcuResult InitParallelDirectResource(BroadcastParallelDirectContext &ctx)
{
    const auto *arg = ctx.arg;
    if (arg->rankSize <= 1 || arg->rankSize > MAX_RANK_SIZE ||
        arg->rankId >= arg->rankSize || arg->rootId >= arg->rankSize ||
        arg->channelCount == 0 || arg->channelCount > NHR_EVENT_BIT_NUM) {
        HCCL_ERROR("[InitParallelDirectResource] invalid rank/channel info, rankSize[%llu], rank[%u], "
                   "root[%u], channelCount[%u]",
            static_cast<unsigned long long>(arg->rankSize), arg->rankId, arg->rootId,
            arg->channelCount);
        return CcuResult::CCU_E_PARA;
    }
    if (arg->rankId != arg->rootId &&
        (arg->channelCount != 1 || arg->peerRanks[0] != arg->rootId)) {
        HCCL_ERROR("[InitParallelDirectResource] non-root rank[%u] must only connect root[%u]",
            arg->rankId, arg->rootId);
        return CcuResult::CCU_E_PARA;
    }

    bool seen[MAX_RANK_SIZE] = {};
    ctx.remoteOutput.resize(arg->channelCount);
    ctx.remoteToken.resize(arg->channelCount);
    ctx.remoteDst.resize(arg->channelCount);
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        const uint32_t peerRank = arg->peerRanks[channelIdx];
        if (peerRank >= arg->rankSize || peerRank == arg->rankId || seen[peerRank]) {
            HCCL_ERROR("[InitParallelDirectResource] invalid or duplicate peer[%u]", peerRank);
            return CcuResult::CCU_E_PARA;
        }
        seen[peerRank] = true;
        ctx.remoteOutput[channelIdx] =
            ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], OUTPUT_XN_ID);
        ctx.remoteToken[channelIdx] =
            ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID);
    }
    return CCU_SUCCESS;
}

CcuResult LoadParallelDirectArgs(BroadcastParallelDirectContext &ctx)
{
    uint32_t argId = 0;
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.myInput, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.myOutput, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.myToken, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.dataSize, argId++));
    return CCU_SUCCESS;
}

CcuResult ParallelDirectPreSync(BroadcastParallelDirectContext &ctx)
{
    const auto *arg = ctx.arg;
    if (arg->rankId == arg->rootId) {
        constexpr uint32_t allAddrBits = (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
        for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
            CCU_KERNEL_CHK_RET(
                ccu::NotifyWait(arg->channels[channelIdx], CKE_IDX_0, allAddrBits));
        }
        return CCU_SUCCESS;
    }

    CCU_KERNEL_CHK_RET(ccu::WriteVariableWithNotify(
        arg->channels[0], ctx.myOutput, OUTPUT_XN_ID, CKE_IDX_0, 1U << OUTPUT_XN_ID));
    CCU_KERNEL_CHK_RET(ccu::WriteVariableWithNotify(
        arg->channels[0], ctx.myToken, TOKEN_XN_ID, CKE_IDX_0, 1U << TOKEN_XN_ID));
    return CCU_SUCCESS;
}

CcuResult DoParallelDirectBroadcast(BroadcastParallelDirectContext &ctx)
{
    const auto *arg = ctx.arg;
    if (arg->rankId != arg->rootId) {
        return CCU_SUCCESS;
    }

    CCU_IF(ctx.dataSize != 0)
    {
        ctx.localSrc.addr = ctx.myInput;
        ctx.localSrc.token = ctx.myToken;
        for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
            ctx.remoteDst[channelIdx].addr = ctx.remoteOutput[channelIdx];
            ctx.remoteDst[channelIdx].token = ctx.remoteToken[channelIdx];
            CCU_KERNEL_CHK_RET(ccu::Write(arg->channels[channelIdx], ctx.remoteDst[channelIdx],
                ctx.localSrc, ctx.dataSize, ctx.event, static_cast<uint16_t>(1U << channelIdx)));
        }
        const uint16_t allChannelMask =
            static_cast<uint16_t>((1U << arg->channelCount) - 1U);
        CCU_KERNEL_CHK_RET(ccu::EventWait(ctx.event, allChannelMask));
    }
    return CCU_SUCCESS;
}

CcuResult ParallelDirectPostSync(BroadcastParallelDirectContext &ctx)
{
    const auto *arg = ctx.arg;
    if (arg->rankId == arg->rootId) {
        for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
            CCU_KERNEL_CHK_RET(ccu::NotifyRecord(
                arg->channels[channelIdx], CKE_IDX_0, 1U << DIRECT_FINISH_SYNC_ID));
        }
        return CCU_SUCCESS;
    }

    CCU_KERNEL_CHK_RET(
        ccu::NotifyWait(arg->channels[0], CKE_IDX_0, 1U << DIRECT_FINISH_SYNC_ID));
    return CCU_SUCCESS;
}

CcuResult RunParallelDirectKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuBroadcastParallelDirectKernelArg *>(arg);
    BroadcastParallelDirectContext ctx;
    ctx.arg = kernelArg;

    HCCL_INFO("[RunParallelDirectKernel] start rank[%u] root[%u] rankSize[%llu] peers[%u]",
        ctx.arg->rankId, ctx.arg->rootId, static_cast<unsigned long long>(ctx.arg->rankSize),
        ctx.arg->channelCount);
    CCU_KERNEL_CHK_RET(InitParallelDirectResource(ctx));
    CCU_KERNEL_CHK_RET(LoadParallelDirectArgs(ctx));
    CCU_KERNEL_CHK_RET(ParallelDirectPreSync(ctx));
    CCU_KERNEL_CHK_RET(DoParallelDirectBroadcast(ctx));
    CCU_KERNEL_CHK_RET(ParallelDirectPostSync(ctx));
    HCCL_INFO("[RunParallelDirectKernel] end");
    return CCU_SUCCESS;
}

CcuResult PreSync(BroadcastMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        CCU_KERNEL_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[channelIdx], ctx.myOutput, OUTPUT_XN_ID,
            CKE_IDX_0, 1 << OUTPUT_XN_ID));
        CCU_KERNEL_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[channelIdx], ctx.myToken, TOKEN_XN_ID,
            CKE_IDX_0, 1 << TOKEN_XN_ID));
    }

    constexpr uint32_t allAddrBits = (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        CCU_KERNEL_CHK_RET(ccu::NotifyWait(arg->channels[channelIdx], CKE_IDX_0, allAddrBits));
    }
    return CCU_SUCCESS;
}

CcuResult DirectRootPreSync(BroadcastMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    if (arg->rankId == arg->rootId) {
        constexpr uint32_t allAddrBits = (1 << OUTPUT_XN_ID) | (1 << TOKEN_XN_ID);
        for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; ++rankIdx) {
            if (rankIdx == arg->rankId) {
                continue;
            }
            const uint32_t channelIdx = GetChannelIndex(arg, rankIdx);
            CCU_KERNEL_CHK_RET(ccu::NotifyWait(arg->channels[channelIdx], CKE_IDX_0, allAddrBits));
        }
        return CCU_SUCCESS;
    }

    const uint32_t rootChannelIdx = GetChannelIndex(arg, arg->rootId);
    CCU_KERNEL_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[rootChannelIdx], ctx.myOutput, OUTPUT_XN_ID,
        CKE_IDX_0, 1 << OUTPUT_XN_ID));
    CCU_KERNEL_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[rootChannelIdx], ctx.myToken, TOKEN_XN_ID,
        CKE_IDX_0, 1 << TOKEN_XN_ID));
    return CCU_SUCCESS;
}

CcuResult DirectRootPostSync(BroadcastMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    if (arg->rankId == arg->rootId) {
        for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; ++rankIdx) {
            if (rankIdx == arg->rankId) {
                continue;
            }
            const uint32_t channelIdx = GetChannelIndex(arg, rankIdx);
            CCU_KERNEL_CHK_RET(
                ccu::NotifyRecord(arg->channels[channelIdx], CKE_IDX_0, 1 << DIRECT_FINISH_SYNC_ID));
        }
        return CCU_SUCCESS;
    }

    const uint32_t rootChannelIdx = GetChannelIndex(arg, arg->rootId);
    CCU_KERNEL_CHK_RET(ccu::NotifyWait(arg->channels[rootChannelIdx], CKE_IDX_0, 1 << DIRECT_FINISH_SYNC_ID));
    return CCU_SUCCESS;
}

CcuResult RankBarrier(BroadcastMesh1DMem2MemContext &ctx, int signalId)
{
    const auto *arg = ctx.arg;
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        CCU_KERNEL_CHK_RET(ccu::NotifyRecord(arg->channels[channelIdx], CKE_IDX_0, 1 << signalId));
    }
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        CCU_KERNEL_CHK_RET(ccu::NotifyWait(arg->channels[channelIdx], CKE_IDX_0, 1 << signalId));
    }
    return CCU_SUCCESS;
}

CcuResult CalcNhrSliceOffset(BroadcastMesh1DMem2MemContext &ctx, ccu::Variable &normalSliceSize,
    std::vector<ccu::Variable> &sliceOffsets)
{
    if (sliceOffsets.size() != ctx.arg->rankSize) {
        HCCL_ERROR("[CalcNhrSliceOffset] invalid offset count[%zu], rankSize[%llu]", sliceOffsets.size(),
            static_cast<unsigned long long>(ctx.arg->rankSize));
        return CcuResult::CCU_E_PARA;
    }
    ccu::Variable currentOffset;
    currentOffset = 0;
    for (uint32_t sliceIdx = 0; sliceIdx < ctx.arg->rankSize; ++sliceIdx) {
        sliceOffsets[sliceIdx] = currentOffset;
        currentOffset += normalSliceSize;
    }
    return CCU_SUCCESS;
}

// 只编排一个分片的 Write，不等待 Event。paired NHR 会把两个分片的 Write 合并后再等待，
// 让同一 NHR step 中的两段数据尽可能同时在途。
CcuResult IssueNhrStepWrites(BroadcastMesh1DMem2MemContext &ctx, const BroadcastNhrStepInfo &stepInfo,
    ccu::Variable &baseOffset, ccu::Variable &normalSliceSize, ccu::Variable &lastSliceSize,
    const std::vector<ccu::Variable> &sliceOffsets, bool isScatterStep, uint32_t eventBitOffset,
    uint32_t &eventCount)
{
    const auto &sendSliceIdxs = stepInfo.txSliceIdxs;
    eventCount = static_cast<uint32_t>(sendSliceIdxs.size());
    if (sendSliceIdxs.empty()) {
        return CCU_SUCCESS;
    }
    if (sendSliceIdxs.size() > NHR_EVENT_BIT_NUM || eventBitOffset + sendSliceIdxs.size() > NHR_EVENT_BIT_NUM ||
        stepInfo.toRank >= ctx.arg->rankSize) {
        HCCL_ERROR("[IssueNhrStepWrites] invalid send step, sliceNum[%zu], eventBitOffset[%u], toRank[%u]",
            sendSliceIdxs.size(), eventBitOffset, stepInfo.toRank);
        return CcuResult::CCU_E_PARA;
    }

    const uint32_t channelIdx = GetChannelIndex(ctx.arg, stepInfo.toRank);
    for (uint32_t idx = 0; idx < sendSliceIdxs.size(); ++idx) {
        const uint32_t sliceIdx = sendSliceIdxs[idx];
        if (sliceIdx >= ctx.arg->rankSize) {
            HCCL_ERROR("[IssueNhrStepWrites] invalid sliceIdx[%u]", sliceIdx);
            return CcuResult::CCU_E_PARA;
        }

        ctx.localSrc.addr = (isScatterStep && stepInfo.step == 0) ? ctx.myInput : ctx.myOutput;
        ctx.localSrc.addr += baseOffset;
        ctx.localSrc.addr += sliceOffsets[sliceIdx];
        ctx.localSrc.token = ctx.myToken;

        ctx.remoteDst[stepInfo.toRank].addr = ctx.remoteOutput[channelIdx];
        ctx.remoteDst[stepInfo.toRank].addr += baseOffset;
        ctx.remoteDst[stepInfo.toRank].addr += sliceOffsets[sliceIdx];
        ctx.remoteDst[stepInfo.toRank].token = ctx.remoteToken[channelIdx];

        auto &sliceSize = (sliceIdx + 1 == ctx.arg->rankSize) ? lastSliceSize : normalSliceSize;
        const uint16_t eventMask = static_cast<uint16_t>(1U << (eventBitOffset + idx));
        CCU_IF(sliceSize != 0)
        {
            CCU_KERNEL_CHK_RET(ccu::Write(ctx.arg->channels[channelIdx], ctx.remoteDst[stepInfo.toRank],
                ctx.localSrc, sliceSize, ctx.event, eventMask));
        }
        CCU_IF(sliceSize == 0)
        {
            CCU_KERNEL_CHK_RET(ccu::EventRecord(ctx.event, eventMask));
        }
    }
    return CCU_SUCCESS;
}

CcuResult WaitNhrStepWrites(BroadcastMesh1DMem2MemContext &ctx, uint32_t eventCount)
{
    if (eventCount == 0) {
        return CCU_SUCCESS;
    }
    if (eventCount > NHR_EVENT_BIT_NUM) {
        HCCL_ERROR("[WaitNhrStepWrites] invalid event count[%u]", eventCount);
        return CcuResult::CCU_E_PARA;
    }
    const uint16_t allEventMask = static_cast<uint16_t>((1U << eventCount) - 1);
    CCU_KERNEL_CHK_RET(ccu::EventWait(ctx.event, allEventMask));
    return CCU_SUCCESS;
}

CcuResult SyncNhrStep(BroadcastMesh1DMem2MemContext &ctx, const BroadcastNhrStepInfo &stepInfo,
    int signalId, bool needPeerSync)
{
    if (!needPeerSync) {
        return CCU_SUCCESS;
    }
    if (!stepInfo.txSliceIdxs.empty()) {
        if (stepInfo.toRank >= ctx.arg->rankSize) {
            HCCL_ERROR("[SyncNhrStep] invalid toRank[%u]", stepInfo.toRank);
            return CcuResult::CCU_E_PARA;
        }
        const uint32_t channelIdx = GetChannelIndex(ctx.arg, stepInfo.toRank);
        CCU_KERNEL_CHK_RET(ccu::NotifyRecord(
            ctx.arg->channels[channelIdx], CKE_IDX_0, static_cast<uint32_t>(1U << signalId)));
    }

    if (!stepInfo.rxSliceIdxs.empty()) {
        if (stepInfo.fromRank >= ctx.arg->rankSize) {
            HCCL_ERROR("[SyncNhrStep] invalid fromRank[%u]", stepInfo.fromRank);
            return CcuResult::CCU_E_PARA;
        }
        const uint32_t channelIdx = GetChannelIndex(ctx.arg, stepInfo.fromRank);
        CCU_KERNEL_CHK_RET(ccu::NotifyWait(
            ctx.arg->channels[channelIdx], CKE_IDX_0, static_cast<uint32_t>(1U << signalId)));
    }
    return CCU_SUCCESS;
}

CcuResult DoNhrStep(BroadcastMesh1DMem2MemContext &ctx, const BroadcastNhrStepInfo &stepInfo,
    ccu::Variable &baseOffset, ccu::Variable &normalSliceSize, ccu::Variable &lastSliceSize,
    bool isScatterStep, int signalId, bool needPeerSync)
{
    // 每一步只与 NHR 指定的一个发送对端和一个接收对端通信，避免 Mesh AllGather
    // 在大包场景同时向所有 rank 扇出，显著降低跨 Server 链路争用。
    uint32_t eventCount = 0;
    CCU_KERNEL_CHK_RET(IssueNhrStepWrites(ctx, stepInfo, baseOffset, normalSliceSize, lastSliceSize,
        ctx.nhrSliceOffset, isScatterStep, 0, eventCount));
    CCU_KERNEL_CHK_RET(WaitNhrStepWrites(ctx, eventCount));
    CCU_KERNEL_CHK_RET(SyncNhrStep(ctx, stepInfo, signalId, needPeerSync));
    return CCU_SUCCESS;
}

CcuResult DoNhrPairedStep(BroadcastMesh1DMem2MemContext &ctx, const BroadcastNhrStepInfo &stepInfo,
    ccu::Variable &firstBaseOffset, ccu::Variable &firstNormalSliceSize, ccu::Variable &firstLastSliceSize,
    ccu::Variable &secondBaseOffset, ccu::Variable &secondNormalSliceSize,
    ccu::Variable &secondLastSliceSize, const std::vector<ccu::Variable> &secondSliceOffsets,
    bool isScatterStep, int signalId, bool needPeerSync)
{
    uint32_t firstEventCount = 0;
    uint32_t secondEventCount = 0;
    CCU_KERNEL_CHK_RET(IssueNhrStepWrites(ctx, stepInfo, firstBaseOffset, firstNormalSliceSize,
        firstLastSliceSize, ctx.nhrSliceOffset, isScatterStep, 0, firstEventCount));
    // 两个分片使用不重叠的 Event bit；rank4 每步最多 2 个 slice，因此总数不超过 4。
    CCU_KERNEL_CHK_RET(IssueNhrStepWrites(ctx, stepInfo, secondBaseOffset, secondNormalSliceSize,
        secondLastSliceSize, secondSliceOffsets, isScatterStep, firstEventCount, secondEventCount));
    CCU_KERNEL_CHK_RET(WaitNhrStepWrites(ctx, firstEventCount + secondEventCount));
    CCU_KERNEL_CHK_RET(SyncNhrStep(ctx, stepInfo, signalId, needPeerSync));
    return CCU_SUCCESS;
}

CcuResult DoNhrChunk(BroadcastMesh1DMem2MemContext &ctx,
    const CcuBroadcastNhr1DMem2MemKernelArg *kernelArg, ccu::Variable &baseOffset,
    ccu::Variable &normalSliceSize, ccu::Variable &lastSliceSize)
{
    const size_t stepNum = kernelArg->stepInfoVector.size() / 2;
    if (stepNum == 0 || kernelArg->stepInfoVector.size() != stepNum * 2) {
        HCCL_ERROR("[DoNhrChunk] invalid stepInfo size[%zu]", kernelArg->stepInfoVector.size());
        return CcuResult::CCU_E_PARA;
    }

    CCU_KERNEL_CHK_RET(CalcNhrSliceOffset(ctx, normalSliceSize, ctx.nhrSliceOffset));
    for (size_t step = 0; step < stepNum; ++step) {
        CCU_KERNEL_CHK_RET(DoNhrStep(ctx, kernelArg->stepInfoVector[step], baseOffset,
            normalSliceSize, lastSliceSize, true, NHR_SCATTER_STEP_SYNC_ID, true));
    }
    for (size_t step = stepNum; step < kernelArg->stepInfoVector.size(); ++step) {
        // 最后一步的数据完成由 EventWait 保证，随后紧邻的 RankBarrier 负责全 rank 收敛，
        // 无需再做一次点对点 Notify。该依赖关系与官方 Broadcast NHR Kernel 保持一致。
        const bool needPeerSync = step + 1 != kernelArg->stepInfoVector.size();
        CCU_KERNEL_CHK_RET(DoNhrStep(ctx, kernelArg->stepInfoVector[step], baseOffset,
            normalSliceSize, lastSliceSize, false, NHR_ALLGATHER_STEP_SYNC_ID, needPeerSync));
    }
    return CCU_SUCCESS;
}

CcuResult DoNhrPairedChunks(BroadcastMesh1DMem2MemContext &ctx,
    const CcuBroadcastNhr1DMem2MemKernelArg *kernelArg)
{
    const size_t stepNum = kernelArg->stepInfoVector.size() / 2;
    if (stepNum == 0 || kernelArg->stepInfoVector.size() != stepNum * 2) {
        HCCL_ERROR("[DoNhrPairedChunks] invalid stepInfo size[%zu]", kernelArg->stepInfoVector.size());
        return CcuResult::CCU_E_PARA;
    }

    // 只在 NHR Kernel 中申请第二套 offset Variable，避免 Direct/Mesh Kernel 增加静态资源。
    std::vector<ccu::Variable> secondSliceOffsets(ctx.arg->rankSize);
    CCU_KERNEL_CHK_RET(CalcNhrSliceOffset(ctx, ctx.normalSliceSize, ctx.nhrSliceOffset));
    CCU_KERNEL_CHK_RET(CalcNhrSliceOffset(ctx, ctx.secondNormalSliceSize, secondSliceOffsets));

    for (size_t step = 0; step < stepNum; ++step) {
        CCU_KERNEL_CHK_RET(DoNhrPairedStep(ctx, kernelArg->stepInfoVector[step],
            ctx.zeroOffset, ctx.normalSliceSize, ctx.lastSliceSize,
            ctx.secondOffset, ctx.secondNormalSliceSize, ctx.secondLastSliceSize,
            secondSliceOffsets, true, NHR_SCATTER_STEP_SYNC_ID, true));
    }
    for (size_t step = stepNum; step < kernelArg->stepInfoVector.size(); ++step) {
        // 最后一步由一次 RankBarrier 收敛，两个分片不再分别做点对点 Notify。
        const bool needPeerSync = step + 1 != kernelArg->stepInfoVector.size();
        CCU_KERNEL_CHK_RET(DoNhrPairedStep(ctx, kernelArg->stepInfoVector[step],
            ctx.zeroOffset, ctx.normalSliceSize, ctx.lastSliceSize,
            ctx.secondOffset, ctx.secondNormalSliceSize, ctx.secondLastSliceSize,
            secondSliceOffsets, false, NHR_ALLGATHER_STEP_SYNC_ID, needPeerSync));
    }
    return CCU_SUCCESS;
}

CcuResult InitHierarchicalResource(BroadcastHierarchicalContext &ctx)
{
    const auto *arg = ctx.arg;
    const bool valid2x8 = arg->rankSize == BROADCAST_PARALLEL_2D_RANK_SIZE &&
        arg->localRankCount == BROADCAST_PARALLEL_2D_LANE_COUNT;
    const bool valid8p4 = arg->rankSize == BROADCAST_ASYMMETRIC_2D_RANK_SIZE &&
        (arg->localRankCount == BROADCAST_PARALLEL_2D_LANE_COUNT ||
            arg->localRankCount == BROADCAST_ASYMMETRIC_2D_SMALL_GROUP_SIZE);
    if ((!valid2x8 && !valid8p4) || arg->laneCount != BROADCAST_PARALLEL_2D_LANE_COUNT ||
        arg->channelCount > MAX_RANK_SIZE) {
        HCCL_ERROR("[InitHierarchicalResource] invalid topology, rankSize[%llu], localRankCount[%u], laneCount[%u], "
                   "channelCount[%u]",
            static_cast<unsigned long long>(arg->rankSize), arg->localRankCount, arg->laneCount,
            arg->channelCount);
        return CcuResult::CCU_E_PARA;
    }

    bool containsLocalRank = false;
    for (uint32_t localIdx = 0; localIdx < arg->localRankCount; ++localIdx) {
        if (arg->localRanks[localIdx] >= arg->rankSize) {
            HCCL_ERROR("[InitHierarchicalResource] invalid local rank[%u]", arg->localRanks[localIdx]);
            return CcuResult::CCU_E_PARA;
        }
        containsLocalRank = containsLocalRank || arg->localRanks[localIdx] == arg->rankId;
    }
    if (!containsLocalRank) {
        HCCL_ERROR("[InitHierarchicalResource] local group misses rank[%u]", arg->rankId);
        return CcuResult::CCU_E_PARA;
    }

    const uint32_t *laneOwners = arg->isSourceGroup != 0 ?
        arg->sourceGatewayRanks : arg->destinationGatewayRanks;
    bool containsLocalLane = false;
    for (uint32_t laneIdx = 0; laneIdx < arg->laneCount; ++laneIdx) {
        const uint32_t sourceRank = arg->sourceGatewayRanks[laneIdx];
        const uint32_t destinationRank = arg->destinationGatewayRanks[laneIdx];
        if (sourceRank >= arg->rankSize || destinationRank >= arg->rankSize ||
            sourceRank == destinationRank) {
            HCCL_ERROR("[InitHierarchicalResource] invalid lane[%u], pair[%u,%u]",
                laneIdx, sourceRank, destinationRank);
            return CcuResult::CCU_E_PARA;
        }
        containsLocalLane = containsLocalLane || laneOwners[laneIdx] == arg->rankId;
    }
    if (!containsLocalLane) {
        HCCL_ERROR("[InitHierarchicalResource] rank[%u] owns no lane", arg->rankId);
        return CcuResult::CCU_E_PARA;
    }

    ctx.remoteOutput.resize(arg->channelCount);
    ctx.remoteToken.resize(arg->channelCount);
    ctx.remoteDst.resize(arg->channelCount);
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        if (arg->peerRanks[channelIdx] >= arg->rankSize || arg->peerRanks[channelIdx] == arg->rankId) {
            HCCL_ERROR("[InitHierarchicalResource] invalid peer rank[%u]", arg->peerRanks[channelIdx]);
            return CcuResult::CCU_E_PARA;
        }
        ctx.remoteOutput[channelIdx] =
            ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], OUTPUT_XN_ID);
        ctx.remoteToken[channelIdx] =
            ccu::GetResByChannel<ccu::Variable>(arg->channels[channelIdx], TOKEN_XN_ID);
    }
    ctx.scatterSrc.resize(arg->laneCount);
    ctx.sliceOffsets.resize(arg->laneCount);
    return CCU_SUCCESS;
}

CcuResult LoadHierarchicalArgs(BroadcastHierarchicalContext &ctx)
{
    uint32_t argId = 0;
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.myInput, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.myOutput, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.myToken, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.firstNormalLaneSize, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.firstLastLaneSize, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.firstNormalSourceSize, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.firstNormalDestinationSize, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.firstLastSourceSize, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.firstLastDestinationSize, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.secondSourceHalfSize, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.secondDestinationHalfSize, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.secondSourceNormalLaneSize, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.secondSourceLastLaneSize, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.secondDestinationNormalLaneSize, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.secondDestinationLastLaneSize, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.secondOffset, argId++));
    CCU_KERNEL_CHK_RET(ccu::LoadArg(ctx.phase, argId++));
    ctx.zeroOffset = 0;
    return CCU_SUCCESS;
}

uint32_t FindHierarchicalChannelIndex(const CcuBroadcastHierarchicalKernelArg *arg, uint32_t peerRank)
{
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        if (arg->peerRanks[channelIdx] == peerRank) {
            return channelIdx;
        }
    }
    return arg->channelCount;
}

const uint32_t *GetLocalLaneOwners(const CcuBroadcastHierarchicalKernelArg *arg)
{
    return arg->isSourceGroup != 0 ? arg->sourceGatewayRanks : arg->destinationGatewayRanks;
}

uint32_t FindGatewayLane(const CcuBroadcastHierarchicalKernelArg *arg, uint32_t rankId)
{
    const uint32_t *gatewayRanks = arg->isSourceGroup != 0 ?
        arg->sourceGatewayRanks : arg->destinationGatewayRanks;
    for (uint32_t laneIdx = 0; laneIdx < arg->laneCount; ++laneIdx) {
        if (gatewayRanks[laneIdx] == rankId) {
            return laneIdx;
        }
    }
    return arg->laneCount;
}

uint32_t FindSourceGatewayLane(const CcuBroadcastHierarchicalKernelArg *arg, uint32_t rankId)
{
    for (uint32_t laneIdx = 0; laneIdx < arg->laneCount; ++laneIdx) {
        if (arg->sourceGatewayRanks[laneIdx] == rankId) {
            return laneIdx;
        }
    }
    return arg->laneCount;
}

CcuResult HierarchicalPreSync(BroadcastHierarchicalContext &ctx)
{
    const auto *arg = ctx.arg;
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        CCU_KERNEL_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[channelIdx], ctx.myOutput,
            OUTPUT_XN_ID, CKE_IDX_0, 1U << OUTPUT_XN_ID));
        CCU_KERNEL_CHK_RET(ccu::WriteVariableWithNotify(arg->channels[channelIdx], ctx.myToken,
            TOKEN_XN_ID, CKE_IDX_0, 1U << TOKEN_XN_ID));
    }
    constexpr uint32_t allAddrBits = (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        CCU_KERNEL_CHK_RET(ccu::NotifyWait(arg->channels[channelIdx], CKE_IDX_0, allAddrBits));
    }
    return CCU_SUCCESS;
}

CcuResult CalcHierarchicalSliceOffsets(BroadcastHierarchicalContext &ctx, ccu::Variable &normalSliceSize)
{
    ccu::Variable currentOffset;
    currentOffset = 0;
    for (uint32_t laneIdx = 0; laneIdx < ctx.arg->laneCount; ++laneIdx) {
        ctx.sliceOffsets[laneIdx] = currentOffset;
        currentOffset += normalSliceSize;
    }
    return CCU_SUCCESS;
}

CcuResult HierarchicalChannelBarrier(BroadcastHierarchicalContext &ctx)
{
    const auto *arg = ctx.arg;
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        CCU_KERNEL_CHK_RET(ccu::NotifyRecord(
            arg->channels[channelIdx], CKE_IDX_0, 1U << HIERARCHY_GROUP_FINISH_ID));
    }
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        CCU_KERNEL_CHK_RET(ccu::NotifyWait(
            arg->channels[channelIdx], CKE_IDX_0, 1U << HIERARCHY_GROUP_FINISH_ID));
    }
    return CCU_SUCCESS;
}

CcuResult DoHierarchicalLocalScatter(BroadcastHierarchicalContext &ctx, ccu::Variable &baseOffset,
    ccu::Variable &normalSliceSize, ccu::Variable &lastSliceSize, uint32_t localRoot,
    bool sourceIsInput, int readySignalId)
{
    const auto *arg = ctx.arg;
    const uint32_t *laneOwners = GetLocalLaneOwners(arg);
    if (ctx.laneRemoteDst.empty()) {
        ctx.laneRemoteDst.resize(arg->laneCount);
    }
    CCU_KERNEL_CHK_RET(CalcHierarchicalSliceOffsets(ctx, normalSliceSize));
    if (arg->rankId == localRoot) {
        for (uint32_t laneIdx = 0; laneIdx < arg->laneCount; ++laneIdx) {
            const uint32_t destinationRank = laneOwners[laneIdx];
            const uint16_t eventMask = static_cast<uint16_t>(1U << laneIdx);
            auto &sliceSize = laneIdx + 1 == arg->laneCount ? lastSliceSize : normalSliceSize;
            ctx.scatterSrc[laneIdx].addr = sourceIsInput ? ctx.myInput : ctx.myOutput;
            ctx.scatterSrc[laneIdx].addr += baseOffset;
            ctx.scatterSrc[laneIdx].addr += ctx.sliceOffsets[laneIdx];
            ctx.scatterSrc[laneIdx].token = ctx.myToken;

            if (destinationRank == arg->rankId) {
                CCU_KERNEL_CHK_RET(ccu::EventRecord(ctx.event, eventMask));
                continue;
            }
            const uint32_t channelIdx = FindHierarchicalChannelIndex(arg, destinationRank);
            if (channelIdx >= arg->channelCount) {
                HCCL_ERROR("[DoHierarchicalLocalScatter] missing destination channel[%u]", destinationRank);
                return CcuResult::CCU_E_PARA;
            }
            ctx.laneRemoteDst[laneIdx].addr = ctx.remoteOutput[channelIdx];
            ctx.laneRemoteDst[laneIdx].addr += baseOffset;
            ctx.laneRemoteDst[laneIdx].addr += ctx.sliceOffsets[laneIdx];
            ctx.laneRemoteDst[laneIdx].token = ctx.remoteToken[channelIdx];
            CCU_IF(sliceSize != 0)
            {
                CCU_KERNEL_CHK_RET(ccu::Write(arg->channels[channelIdx], ctx.laneRemoteDst[laneIdx],
                    ctx.scatterSrc[laneIdx], sliceSize, ctx.event, eventMask));
            }
            CCU_IF(sliceSize == 0)
            {
                CCU_KERNEL_CHK_RET(ccu::EventRecord(ctx.event, eventMask));
            }
        }

        const uint16_t allLaneMask = static_cast<uint16_t>((1U << arg->laneCount) - 1);
        CCU_KERNEL_CHK_RET(ccu::EventWait(ctx.event, allLaneMask));
        for (uint32_t localIdx = 0; localIdx < arg->localRankCount; ++localIdx) {
            const uint32_t destinationRank = arg->localRanks[localIdx];
            if (destinationRank == arg->rankId) {
                continue;
            }
            const uint32_t channelIdx = FindHierarchicalChannelIndex(arg, destinationRank);
            CCU_KERNEL_CHK_RET(ccu::NotifyRecord(
                arg->channels[channelIdx], CKE_IDX_0, static_cast<uint32_t>(1U << readySignalId)));
        }
        return CCU_SUCCESS;
    }

    const uint32_t rootChannelIdx = FindHierarchicalChannelIndex(arg, localRoot);
    if (rootChannelIdx >= arg->channelCount) {
        HCCL_ERROR("[DoHierarchicalLocalScatter] missing root channel[%u]", localRoot);
        return CcuResult::CCU_E_PARA;
    }
    CCU_KERNEL_CHK_RET(ccu::NotifyWait(
        arg->channels[rootChannelIdx], CKE_IDX_0, static_cast<uint32_t>(1U << readySignalId)));
    return CCU_SUCCESS;
}

CcuResult DoHierarchicalLocalAllGather(BroadcastHierarchicalContext &ctx, ccu::Variable &baseOffset,
    ccu::Variable &normalSliceSize, ccu::Variable &lastSliceSize)
{
    const auto *arg = ctx.arg;
    const uint32_t *laneOwners = GetLocalLaneOwners(arg);
    if (ctx.laneRemoteDst.empty()) {
        ctx.laneRemoteDst.resize(arg->laneCount);
    }
    CCU_KERNEL_CHK_RET(CalcHierarchicalSliceOffsets(ctx, normalSliceSize));

    uint32_t taskIdx = 0;
    for (uint32_t laneIdx = 0; laneIdx < arg->laneCount; ++laneIdx) {
        if (laneOwners[laneIdx] != arg->rankId) {
            continue;
        }
        auto &sliceSize = laneIdx + 1 == arg->laneCount ? lastSliceSize : normalSliceSize;
        ctx.scatterSrc[laneIdx].addr = ctx.myOutput;
        ctx.scatterSrc[laneIdx].addr += baseOffset;
        ctx.scatterSrc[laneIdx].addr += ctx.sliceOffsets[laneIdx];
        ctx.scatterSrc[laneIdx].token = ctx.myToken;

        for (uint32_t localIdx = 0; localIdx < arg->localRankCount; ++localIdx) {
            const uint32_t destinationRank = arg->localRanks[localIdx];
            const uint32_t taskSlot = taskIdx++;
            if (taskSlot >= arg->laneCount) {
                HCCL_ERROR("[DoHierarchicalLocalAllGather] task slot[%u] exceeds lane resource[%u]",
                    taskSlot, arg->laneCount);
                return CcuResult::CCU_E_PARA;
            }
            const uint16_t eventMask = static_cast<uint16_t>(1U << taskSlot);
            if (destinationRank == arg->rankId) {
                CCU_KERNEL_CHK_RET(ccu::EventRecord(ctx.event, eventMask));
                continue;
            }
            const uint32_t channelIdx = FindHierarchicalChannelIndex(arg, destinationRank);
            if (channelIdx >= arg->channelCount) {
                HCCL_ERROR("[DoHierarchicalLocalAllGather] missing destination channel[%u]", destinationRank);
                return CcuResult::CCU_E_PARA;
            }
            ctx.laneRemoteDst[taskSlot].addr = ctx.remoteOutput[channelIdx];
            ctx.laneRemoteDst[taskSlot].addr += baseOffset;
            ctx.laneRemoteDst[taskSlot].addr += ctx.sliceOffsets[laneIdx];
            ctx.laneRemoteDst[taskSlot].token = ctx.remoteToken[channelIdx];
            CCU_IF(sliceSize != 0)
            {
                CCU_KERNEL_CHK_RET(ccu::Write(arg->channels[channelIdx], ctx.laneRemoteDst[taskSlot],
                    ctx.scatterSrc[laneIdx], sliceSize, ctx.event, eventMask));
            }
            CCU_IF(sliceSize == 0)
            {
                CCU_KERNEL_CHK_RET(ccu::EventRecord(ctx.event, eventMask));
            }
        }
    }

    if (taskIdx == 0 || taskIdx > NHR_EVENT_BIT_NUM) {
        HCCL_ERROR("[DoHierarchicalLocalAllGather] invalid task count[%u]", taskIdx);
        return CcuResult::CCU_E_PARA;
    }
    const uint16_t allTaskMask = static_cast<uint16_t>((1U << taskIdx) - 1);
    CCU_KERNEL_CHK_RET(ccu::EventWait(ctx.event, allTaskMask));
    CCU_KERNEL_CHK_RET(HierarchicalChannelBarrier(ctx));
    return CCU_SUCCESS;
}

CcuResult DoHierarchicalCrossScatter(BroadcastHierarchicalContext &ctx, ccu::Variable &sourceOffset,
    ccu::Variable &destinationOffset, ccu::Variable &transferSize, uint32_t senderRank,
    uint32_t receiverRank, int readySignalId)
{
    const auto *arg = ctx.arg;
    if (arg->rankId == senderRank) {
        const uint32_t channelIdx = FindHierarchicalChannelIndex(arg, receiverRank);
        if (channelIdx >= arg->channelCount) {
            HCCL_ERROR("[DoHierarchicalCrossScatter] missing receiver channel[%u]", receiverRank);
            return CcuResult::CCU_E_PARA;
        }
        ctx.localSrc.addr = senderRank == arg->rootId ? ctx.myInput : ctx.myOutput;
        ctx.localSrc.addr += sourceOffset;
        ctx.localSrc.token = ctx.myToken;
        ctx.remoteDst[channelIdx].addr = ctx.remoteOutput[channelIdx];
        ctx.remoteDst[channelIdx].addr += destinationOffset;
        ctx.remoteDst[channelIdx].token = ctx.remoteToken[channelIdx];
        CCU_IF(transferSize != 0)
        {
            CCU_KERNEL_CHK_RET(ccu::Write(arg->channels[channelIdx], ctx.remoteDst[channelIdx],
                ctx.localSrc, transferSize, ctx.event, 1));
        }
        CCU_IF(transferSize == 0)
        {
            CCU_KERNEL_CHK_RET(ccu::EventRecord(ctx.event, 1));
        }
        CCU_KERNEL_CHK_RET(ccu::EventWait(ctx.event, 1));
        CCU_KERNEL_CHK_RET(ccu::NotifyRecord(
            arg->channels[channelIdx], CKE_IDX_0, static_cast<uint32_t>(1U << readySignalId)));
    } else if (arg->rankId == receiverRank) {
        const uint32_t channelIdx = FindHierarchicalChannelIndex(arg, senderRank);
        if (channelIdx >= arg->channelCount) {
            HCCL_ERROR("[DoHierarchicalCrossScatter] missing sender channel[%u]", senderRank);
            return CcuResult::CCU_E_PARA;
        }
        CCU_KERNEL_CHK_RET(ccu::NotifyWait(
            arg->channels[channelIdx], CKE_IDX_0, static_cast<uint32_t>(1U << readySignalId)));
    }
    return CCU_SUCCESS;
}

CcuResult DoHierarchicalCrossScatterAllLanes(BroadcastHierarchicalContext &ctx, int readySignalId)
{
    const auto *arg = ctx.arg;
    CCU_KERNEL_CHK_RET(CalcHierarchicalSliceOffsets(ctx, ctx.firstNormalLaneSize));

    uint32_t sendCount = 0;
    for (uint32_t laneIdx = 0; laneIdx < arg->laneCount; ++laneIdx) {
        const uint32_t sourceRank = arg->sourceGatewayRanks[laneIdx];
        if (arg->rankId != sourceRank) {
            continue;
        }
        const uint32_t destinationRank = arg->destinationGatewayRanks[laneIdx];
        const uint32_t channelIdx = FindHierarchicalChannelIndex(arg, destinationRank);
        if (channelIdx >= arg->channelCount) {
            HCCL_ERROR("[DoHierarchicalCrossScatterAllLanes] missing destination channel[%u]", destinationRank);
            return CcuResult::CCU_E_PARA;
        }
        auto &sourceSize = laneIdx + 1 == arg->laneCount ?
            ctx.firstLastSourceSize : ctx.firstNormalSourceSize;
        auto &destinationSize = laneIdx + 1 == arg->laneCount ?
            ctx.firstLastDestinationSize : ctx.firstNormalDestinationSize;
        ccu::Variable destinationOffset;
        destinationOffset = ctx.sliceOffsets[laneIdx];
        destinationOffset += sourceSize;
        ctx.scatterSrc[laneIdx].addr = sourceRank == arg->rootId ? ctx.myInput : ctx.myOutput;
        ctx.scatterSrc[laneIdx].addr += destinationOffset;
        ctx.scatterSrc[laneIdx].token = ctx.myToken;
        ctx.remoteDst[channelIdx].addr = ctx.remoteOutput[channelIdx];
        ctx.remoteDst[channelIdx].addr += destinationOffset;
        ctx.remoteDst[channelIdx].token = ctx.remoteToken[channelIdx];
        const uint16_t eventMask = static_cast<uint16_t>(1U << sendCount++);
        CCU_IF(destinationSize != 0)
        {
            CCU_KERNEL_CHK_RET(ccu::Write(arg->channels[channelIdx], ctx.remoteDst[channelIdx],
                ctx.scatterSrc[laneIdx], destinationSize, ctx.event, eventMask));
        }
        CCU_IF(destinationSize == 0)
        {
            CCU_KERNEL_CHK_RET(ccu::EventRecord(ctx.event, eventMask));
        }
    }
    if (sendCount != 0) {
        CCU_KERNEL_CHK_RET(ccu::EventWait(ctx.event, static_cast<uint16_t>((1U << sendCount) - 1)));
    }

    for (uint32_t laneIdx = 0; laneIdx < arg->laneCount; ++laneIdx) {
        const uint32_t sourceRank = arg->sourceGatewayRanks[laneIdx];
        const uint32_t destinationRank = arg->destinationGatewayRanks[laneIdx];
        if (arg->rankId == sourceRank) {
            const uint32_t channelIdx = FindHierarchicalChannelIndex(arg, destinationRank);
            CCU_KERNEL_CHK_RET(ccu::NotifyRecord(
                arg->channels[channelIdx], CKE_IDX_0, static_cast<uint32_t>(1U << readySignalId)));
        } else if (arg->rankId == destinationRank) {
            const uint32_t channelIdx = FindHierarchicalChannelIndex(arg, sourceRank);
            if (channelIdx >= arg->channelCount) {
                HCCL_ERROR("[DoHierarchicalCrossScatterAllLanes] missing source channel[%u]", sourceRank);
                return CcuResult::CCU_E_PARA;
            }
            CCU_KERNEL_CHK_RET(ccu::NotifyWait(
                arg->channels[channelIdx], CKE_IDX_0, static_cast<uint32_t>(1U << readySignalId)));
        }
    }
    return CCU_SUCCESS;
}

CcuResult DoHierarchicalCrossAllGatherA(BroadcastHierarchicalContext &ctx)
{
    const auto *arg = ctx.arg;
    CCU_KERNEL_CHK_RET(CalcHierarchicalSliceOffsets(ctx, ctx.firstNormalLaneSize));

    uint32_t sendCount = 0;
    for (uint32_t laneIdx = 0; laneIdx < arg->laneCount; ++laneIdx) {
        const uint32_t sourceRank = arg->sourceGatewayRanks[laneIdx];
        const uint32_t destinationRank = arg->destinationGatewayRanks[laneIdx];
        const bool sendSource = arg->rankId == sourceRank;
        const bool sendDestination = !BROADCAST_SKIP_REDUNDANT_A_RETURN && arg->rankId == destinationRank;
        if (!sendSource && !sendDestination) {
            continue;
        }
        const uint32_t peerRank = sendSource ? destinationRank : sourceRank;
        const uint32_t channelIdx = FindHierarchicalChannelIndex(arg, peerRank);
        if (channelIdx >= arg->channelCount) {
            HCCL_ERROR("[DoHierarchicalCrossAllGatherA] missing peer channel[%u]", peerRank);
            return CcuResult::CCU_E_PARA;
        }
        auto &sourceSize = laneIdx + 1 == arg->laneCount ?
            ctx.firstLastSourceSize : ctx.firstNormalSourceSize;
        auto &destinationSize = laneIdx + 1 == arg->laneCount ?
            ctx.firstLastDestinationSize : ctx.firstNormalDestinationSize;
        ccu::Variable sendOffset;
        sendOffset = ctx.sliceOffsets[laneIdx];
        if (!sendSource) {
            sendOffset += sourceSize;
        }
        auto &sendSize = sendSource ? sourceSize : destinationSize;
        ctx.scatterSrc[laneIdx].addr = arg->rankId == arg->rootId ? ctx.myInput : ctx.myOutput;
        ctx.scatterSrc[laneIdx].addr += sendOffset;
        ctx.scatterSrc[laneIdx].token = ctx.myToken;
        ctx.remoteDst[channelIdx].addr = ctx.remoteOutput[channelIdx];
        ctx.remoteDst[channelIdx].addr += sendOffset;
        ctx.remoteDst[channelIdx].token = ctx.remoteToken[channelIdx];
        const uint16_t eventMask = static_cast<uint16_t>(1U << sendCount++);
        CCU_IF(sendSize != 0)
        {
            CCU_KERNEL_CHK_RET(ccu::Write(arg->channels[channelIdx], ctx.remoteDst[channelIdx],
                ctx.scatterSrc[laneIdx], sendSize, ctx.event, eventMask));
        }
        CCU_IF(sendSize == 0)
        {
            CCU_KERNEL_CHK_RET(ccu::EventRecord(ctx.event, eventMask));
        }
    }
    if (sendCount != 0) {
        CCU_KERNEL_CHK_RET(ccu::EventWait(ctx.event, static_cast<uint16_t>((1U << sendCount) - 1)));
    }
    CCU_KERNEL_CHK_RET(HierarchicalChannelBarrier(ctx));
    return CCU_SUCCESS;
}

bool IsPrimaryCrossLane(const CcuBroadcastHierarchicalKernelArg *arg, uint32_t laneIdx)
{
    const uint32_t *remoteOwners = arg->isSourceGroup != 0 ?
        arg->destinationGatewayRanks : arg->sourceGatewayRanks;
    for (uint32_t previousLane = 0; previousLane < laneIdx; ++previousLane) {
        if (remoteOwners[previousLane] == remoteOwners[laneIdx]) {
            return false;
        }
    }
    return true;
}

CcuResult DoHierarchicalCrossAllGatherB(BroadcastHierarchicalContext &ctx, ccu::Variable &sendOffset,
    ccu::Variable &sendSize)
{
    const auto *arg = ctx.arg;
    const uint32_t *laneOwners = GetLocalLaneOwners(arg);
    const uint32_t *remoteOwners = arg->isSourceGroup != 0 ?
        arg->destinationGatewayRanks : arg->sourceGatewayRanks;
    const bool localIsSmallGroup = arg->localRankCount < arg->laneCount;

    uint32_t sendCount = 0;
    for (uint32_t laneIdx = 0; laneIdx < arg->laneCount; ++laneIdx) {
        if (laneOwners[laneIdx] != arg->rankId ||
            (!localIsSmallGroup && !IsPrimaryCrossLane(arg, laneIdx))) {
            continue;
        }
        // 小组向两个大组 peer 各发送一份；大组只用 primary lane 回传，避免重复覆盖小组半区。
        const uint32_t peerRank = remoteOwners[laneIdx];
        const uint32_t channelIdx = FindHierarchicalChannelIndex(arg, peerRank);
        if (channelIdx >= arg->channelCount) {
            HCCL_ERROR("[DoHierarchicalCrossAllGatherB] missing peer channel[%u]", peerRank);
            return CcuResult::CCU_E_PARA;
        }
        ctx.scatterSrc[laneIdx].addr = ctx.myOutput;
        ctx.scatterSrc[laneIdx].addr += sendOffset;
        ctx.scatterSrc[laneIdx].token = ctx.myToken;
        ctx.remoteDst[channelIdx].addr = ctx.remoteOutput[channelIdx];
        ctx.remoteDst[channelIdx].addr += sendOffset;
        ctx.remoteDst[channelIdx].token = ctx.remoteToken[channelIdx];
        const uint16_t eventMask = static_cast<uint16_t>(1U << sendCount++);
        CCU_IF(sendSize != 0)
        {
            CCU_KERNEL_CHK_RET(ccu::Write(arg->channels[channelIdx], ctx.remoteDst[channelIdx],
                ctx.scatterSrc[laneIdx], sendSize, ctx.event, eventMask));
        }
        CCU_IF(sendSize == 0)
        {
            CCU_KERNEL_CHK_RET(ccu::EventRecord(ctx.event, eventMask));
        }
    }
    if (sendCount != 0) {
        CCU_KERNEL_CHK_RET(ccu::EventWait(ctx.event, static_cast<uint16_t>((1U << sendCount) - 1)));
    }
    CCU_KERNEL_CHK_RET(HierarchicalChannelBarrier(ctx));
    return CCU_SUCCESS;
}

CcuResult DoScatter(BroadcastMesh1DMem2MemContext &ctx, ccu::Variable &baseOffset,
    ccu::Variable &normalSliceSize, ccu::Variable &lastSliceSize)
{
    const auto *arg = ctx.arg;
    if (arg->rankId != arg->rootId) {
        return CCU_SUCCESS;
    }

    ccu::Variable sliceOffset;
    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; ++rankIdx) {
        if (rankIdx == 0) {
            sliceOffset = 0;
        } else {
            sliceOffset += normalSliceSize;
        }

        auto &sliceSize = (rankIdx + 1 == arg->rankSize) ? lastSliceSize : normalSliceSize;
        const uint16_t rankMask = 1 << rankIdx;

        ctx.scatterSrc[rankIdx].addr = ctx.myInput;
        ctx.scatterSrc[rankIdx].addr += baseOffset;
        ctx.scatterSrc[rankIdx].addr += sliceOffset;
        ctx.scatterSrc[rankIdx].token = ctx.myToken;

        CCU_IF(sliceSize != 0)
        {
            if (rankIdx == arg->rankId) {
                CCU_KERNEL_CHK_RET(ccu::EventRecord(ctx.event, rankMask));
            } else {
                const uint32_t vecIdx = GetChannelIndex(arg, rankIdx);
                ctx.remoteDst[rankIdx].addr = ctx.remoteOutput[vecIdx];
                ctx.remoteDst[rankIdx].addr += baseOffset;
                ctx.remoteDst[rankIdx].addr += sliceOffset;
                ctx.remoteDst[rankIdx].token = ctx.remoteToken[vecIdx];
                CCU_KERNEL_CHK_RET(
                    ccu::Write(arg->channels[vecIdx], ctx.remoteDst[rankIdx], ctx.scatterSrc[rankIdx],
                        sliceSize, ctx.event, rankMask));
            }
        }
        CCU_IF(sliceSize == 0)
        {
            CCU_KERNEL_CHK_RET(ccu::EventRecord(ctx.event, rankMask));
        }
    }

    const uint16_t allRankMask = (1 << arg->rankSize) - 1;
    CCU_KERNEL_CHK_RET(ccu::EventWait(ctx.event, allRankMask));
    return CCU_SUCCESS;
}

CcuResult DoAllGather(BroadcastMesh1DMem2MemContext &ctx, ccu::Variable &baseOffset,
    ccu::Variable &normalSliceSize, ccu::Variable &lastSliceSize, ccu::Variable &allgatherOffset)
{
    const auto *arg = ctx.arg;
    auto &sliceSize = (arg->rankId + 1 == arg->rankSize) ? lastSliceSize : normalSliceSize;

    CCU_IF(sliceSize != 0)
    {
        ctx.localSrc.addr = ctx.myOutput;
        ctx.localSrc.addr += baseOffset;
        ctx.localSrc.addr += allgatherOffset;
        ctx.localSrc.token = ctx.myToken;

        for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; ++rankIdx) {
            const uint16_t rankMask = 1 << rankIdx;
            if (rankIdx == arg->rankId) {
                CCU_KERNEL_CHK_RET(ccu::EventRecord(ctx.event, rankMask));
            } else {
                const uint32_t vecIdx = GetChannelIndex(arg, rankIdx);
                ctx.remoteDst[rankIdx].addr = ctx.remoteOutput[vecIdx];
                ctx.remoteDst[rankIdx].addr += baseOffset;
                ctx.remoteDst[rankIdx].addr += allgatherOffset;
                ctx.remoteDst[rankIdx].token = ctx.remoteToken[vecIdx];
                CCU_KERNEL_CHK_RET(ccu::Write(arg->channels[vecIdx], ctx.remoteDst[rankIdx],
                    ctx.localSrc, sliceSize, ctx.event, rankMask));
            }
        }

        const uint16_t allRankMask = (1 << arg->rankSize) - 1;
        CCU_KERNEL_CHK_RET(ccu::EventWait(ctx.event, allRankMask));
    }
    return CCU_SUCCESS;
}

CcuResult DoDirectBroadcast(BroadcastMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    if (arg->rankId != arg->rootId) {
        return CCU_SUCCESS;
    }

    CCU_IF(ctx.dataSize != 0)
    {
        ctx.localSrc.addr = ctx.myInput;
        ctx.localSrc.token = ctx.myToken;

        for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; ++rankIdx) {
            const uint16_t rankMask = 1 << rankIdx;
            if (rankIdx == arg->rankId) {
                CCU_KERNEL_CHK_RET(ccu::EventRecord(ctx.event, rankMask));
            } else {
                const uint32_t vecIdx = GetChannelIndex(arg, rankIdx);
                ctx.remoteDst[rankIdx].addr = ctx.remoteOutput[vecIdx];
                ctx.remoteDst[rankIdx].token = ctx.remoteToken[vecIdx];
                CCU_KERNEL_CHK_RET(ccu::Write(arg->channels[vecIdx], ctx.remoteDst[rankIdx],
                    ctx.localSrc, ctx.dataSize, ctx.event, rankMask));
            }
        }

        const uint16_t allRankMask = (1 << arg->rankSize) - 1;
        CCU_KERNEL_CHK_RET(ccu::EventWait(ctx.event, allRankMask));
    }
    return CCU_SUCCESS;
}

CcuResult DoDirectGroupBroadcast512K(BroadcastMesh1DMem2MemContext &ctx)
{
    const auto *arg = ctx.arg;
    if (arg->rankId != arg->rootId) {
        return CCU_SUCCESS;
    }

    ccu::Array<ccu::Event> completedEvents(DIRECT_GROUP_LOOP_COUNT);
    ccu::Array<ccu::CcuBuffer> ccuBuffers(DIRECT_GROUP_LOOP_COUNT * DIRECT_GROUP_MS_INTERLEAVE);
    ccu::Variable loopConfig;
    ccu::Variable parallelConfig;
    ccu::Variable offsetConfig;
    ccu::Variable sliceSize;
    loopConfig = MakeDirectLoopParam(0, 0, 1);
    parallelConfig = MakeDirectParallelParam(DIRECT_GROUP_LOOP_COUNT - 1, 0, 1);
    offsetConfig = MakeDirectOffsetParam(
        DIRECT_GROUP_SLICE_SIZE, DIRECT_GROUP_MS_INTERLEAVE, 1);
    sliceSize = DIRECT_GROUP_SLICE_SIZE;

    ctx.localSrc.addr = ctx.myInput;
    ctx.localSrc.token = ctx.myToken;
    for (uint32_t rankIdx = 0; rankIdx < arg->rankSize; ++rankIdx) {
        if (rankIdx == arg->rankId) {
            continue;
        }
        const uint32_t channelIdx = GetChannelIndex(arg, rankIdx);
        ctx.remoteDst[channelIdx].addr = ctx.remoteOutput[channelIdx];
        ctx.remoteDst[channelIdx].token = ctx.remoteToken[channelIdx];
    }

    auto &loopEvent = completedEvents[0];
    auto &loopBuffer = ccuBuffers[0];
    ccu::Func loopBody([&]() {
        ccu::LocalCopy(loopBuffer, ctx.localSrc, sliceSize, loopEvent, 1);
        ccu::EventWait(loopEvent, 1);
        for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
            ccu::Write(arg->channels[channelIdx], ctx.remoteDst[channelIdx], loopBuffer,
                sliceSize, loopEvent, static_cast<uint16_t>(1U << channelIdx));
        }
        ccu::EventWait(loopEvent,
            static_cast<uint16_t>((1U << arg->channelCount) - 1));
    });
    ccu::Loop loop(loopConfig, loopBody);
    std::vector<ccu::Loop> loops{loop};
    ccu::LoopGroup group(parallelConfig, offsetConfig, DIRECT_GROUP_LOOP_COUNT, loops);
    return CCU_SUCCESS;
}
} // namespace

CcuResult CcuBroadcastMesh1DMem2MemKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuBroadcastMesh1DMem2MemKernelArg *>(arg);
    BroadcastMesh1DMem2MemContext ctx;
    ctx.arg = kernelArg;

    HCCL_INFO("[CcuBroadcastMesh1DMem2MemKernel] start rank[%u] root[%u] rankSize[%llu]", ctx.arg->rankId,
        ctx.arg->rootId, static_cast<unsigned long long>(ctx.arg->rankSize));

    CCU_KERNEL_CHK_RET(InitResource(ctx));
    CCU_KERNEL_CHK_RET(LoadArgs(ctx));
    CCU_KERNEL_CHK_RET(PreSync(ctx));
    CCU_KERNEL_CHK_RET(DoScatter(ctx, ctx.zeroOffset, ctx.normalSliceSize, ctx.lastSliceSize));
    CCU_KERNEL_CHK_RET(RankBarrier(ctx, SCATTER_SYNC_ID));
    CCU_KERNEL_CHK_RET(DoAllGather(ctx, ctx.zeroOffset, ctx.normalSliceSize, ctx.lastSliceSize, ctx.allgatherOffset));
    CCU_KERNEL_CHK_RET(RankBarrier(ctx, FINISH_SYNC_ID));
    CCU_IF(ctx.secondDataSize != 0)
    {
        CCU_KERNEL_CHK_RET(DoScatter(ctx, ctx.secondOffset, ctx.secondNormalSliceSize, ctx.secondLastSliceSize));
        CCU_KERNEL_CHK_RET(RankBarrier(ctx, SCATTER_SYNC_ID));
        CCU_KERNEL_CHK_RET(DoAllGather(ctx, ctx.secondOffset, ctx.secondNormalSliceSize, ctx.secondLastSliceSize,
            ctx.secondAllgatherOffset));
        CCU_KERNEL_CHK_RET(RankBarrier(ctx, FINISH_SYNC_ID));
    }

    HCCL_INFO("[CcuBroadcastMesh1DMem2MemKernel] end");
    return CCU_SUCCESS;
}

CcuResult CcuBroadcastDirectKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuBroadcastMesh1DMem2MemKernelArg *>(arg);
    BroadcastMesh1DMem2MemContext ctx;
    ctx.arg = kernelArg;

    HCCL_INFO("[CcuBroadcastDirectKernel] start rank[%u] root[%u] rankSize[%llu]", ctx.arg->rankId,
        ctx.arg->rootId, static_cast<unsigned long long>(ctx.arg->rankSize));

    CCU_KERNEL_CHK_RET(InitResource(ctx));
    CCU_KERNEL_CHK_RET(LoadDirectArgs(ctx));
    CCU_KERNEL_CHK_RET(DirectRootPreSync(ctx));
    if (BROADCAST_ENABLE_GROUP_DIRECT_512K) {
        CCU_IF(ctx.dataSize == DIRECT_GROUP_DATA_SIZE)
        {
            CCU_KERNEL_CHK_RET(DoDirectGroupBroadcast512K(ctx));
        }
        CCU_IF(ctx.dataSize != DIRECT_GROUP_DATA_SIZE)
        {
            CCU_KERNEL_CHK_RET(DoDirectBroadcast(ctx));
        }
    } else {
        CCU_KERNEL_CHK_RET(DoDirectBroadcast(ctx));
    }
    CCU_KERNEL_CHK_RET(DirectRootPostSync(ctx));

    HCCL_INFO("[CcuBroadcastDirectKernel] end");
    return CCU_SUCCESS;
}

CcuResult CcuBroadcastParallelDirectLayer0Kernel(CcuKernelArg arg)
{
    return RunParallelDirectKernel(arg);
}

CcuResult CcuBroadcastParallelDirectLayer1Kernel(CcuKernelArg arg)
{
    return RunParallelDirectKernel(arg);
}

CcuResult CcuBroadcastNhr1DMem2MemKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuBroadcastNhr1DMem2MemKernelArg *>(arg);
    BroadcastMesh1DMem2MemContext ctx;
    ctx.arg = kernelArg;

    HCCL_INFO("[CcuBroadcastNhr1DMem2MemKernel] start rank[%u] root[%u] rankSize[%llu] stepNum[%zu]",
        ctx.arg->rankId, ctx.arg->rootId, static_cast<unsigned long long>(ctx.arg->rankSize),
        kernelArg->stepInfoVector.size() / 2);

    CCU_KERNEL_CHK_RET(InitResource(ctx));
    CCU_KERNEL_CHK_RET(LoadArgs(ctx));
    CCU_KERNEL_CHK_RET(PreSync(ctx));
    if (BROADCAST_ENABLE_NHR_PAIRED_CHUNKS) {
        CCU_KERNEL_CHK_RET(DoNhrPairedChunks(ctx, kernelArg));
        CCU_KERNEL_CHK_RET(RankBarrier(ctx, NHR_POST_SYNC_ID));
    } else {
        CCU_KERNEL_CHK_RET(DoNhrChunk(
            ctx, kernelArg, ctx.zeroOffset, ctx.normalSliceSize, ctx.lastSliceSize));
        CCU_KERNEL_CHK_RET(RankBarrier(ctx, NHR_POST_SYNC_ID));
        CCU_IF(ctx.secondDataSize != 0)
        {
            CCU_KERNEL_CHK_RET(DoNhrChunk(ctx, kernelArg, ctx.secondOffset,
                ctx.secondNormalSliceSize, ctx.secondLastSliceSize));
            CCU_KERNEL_CHK_RET(RankBarrier(ctx, NHR_POST_SYNC_ID));
        }
    }

    HCCL_INFO("[CcuBroadcastNhr1DMem2MemKernel] end");
    return CCU_SUCCESS;
}

CcuResult CcuBroadcastHierarchicalLayer0Kernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuBroadcastHierarchicalKernelArg *>(arg);
    BroadcastHierarchicalContext ctx;
    ctx.arg = kernelArg;

    HCCL_INFO("[CcuBroadcastHierarchicalLayer0Kernel] start rank[%u] root[%u] rankSize[%llu]",
        ctx.arg->rankId, ctx.arg->rootId, static_cast<unsigned long long>(ctx.arg->rankSize));

    CCU_KERNEL_CHK_RET(InitHierarchicalResource(ctx));
    CCU_KERNEL_CHK_RET(LoadHierarchicalArgs(ctx));
    if (!BROADCAST_REUSE_HIERARCHY_REMOTE_ADDR) {
        CCU_KERNEL_CHK_RET(HierarchicalPreSync(ctx));
    }
    const uint32_t rootLaneIdx = FindSourceGatewayLane(ctx.arg, ctx.arg->rootId);
    if (rootLaneIdx >= ctx.arg->laneCount) {
        HCCL_ERROR("[CcuBroadcastHierarchicalLayer0Kernel] root[%u] has no source lane", ctx.arg->rootId);
        return CcuResult::CCU_E_PARA;
    }
    const uint32_t destinationRoot = ctx.arg->destinationGatewayRanks[rootLaneIdx];
    ccu::Variable secondDestinationOffset;
    secondDestinationOffset = ctx.secondOffset;
    secondDestinationOffset += ctx.secondSourceHalfSize;

    CCU_IF(ctx.phase == BROADCAST_HIERARCHY_PHASE_SCATTER_0)
    {
        // 四个 phase 的 output/token 不变；首次交换后，后续 Kernel Launch 复用 Channel XN 变量。
        if (BROADCAST_REUSE_HIERARCHY_REMOTE_ADDR) {
            CCU_KERNEL_CHK_RET(HierarchicalPreSync(ctx));
        }
        if (ctx.arg->isSourceGroup != 0) {
            CCU_KERNEL_CHK_RET(DoHierarchicalLocalScatter(ctx, ctx.zeroOffset,
                ctx.firstNormalLaneSize, ctx.firstLastLaneSize, ctx.arg->rootId,
                true, HIERARCHY_SCATTER_READY_ID));
        }
    }
    CCU_IF(ctx.phase == BROADCAST_HIERARCHY_PHASE_SCATTER_1)
    {
        if (ctx.arg->isSourceGroup != 0) {
            CCU_KERNEL_CHK_RET(DoHierarchicalLocalScatter(ctx, ctx.secondOffset,
                ctx.secondSourceNormalLaneSize, ctx.secondSourceLastLaneSize,
                ctx.arg->rootId, true, HIERARCHY_SCATTER_READY_ID));
        } else {
            CCU_KERNEL_CHK_RET(DoHierarchicalLocalScatter(ctx, secondDestinationOffset,
                ctx.secondDestinationNormalLaneSize, ctx.secondDestinationLastLaneSize,
                destinationRoot, false, HIERARCHY_SCATTER_READY_ID));
        }
    }
    CCU_IF(ctx.phase == BROADCAST_HIERARCHY_PHASE_ALLGATHER_0)
    {
        if (ctx.arg->isSourceGroup != 0) {
            CCU_KERNEL_CHK_RET(DoHierarchicalLocalAllGather(ctx, ctx.secondOffset,
                ctx.secondSourceNormalLaneSize, ctx.secondSourceLastLaneSize));
        } else {
            CCU_KERNEL_CHK_RET(DoHierarchicalLocalAllGather(ctx, secondDestinationOffset,
                ctx.secondDestinationNormalLaneSize, ctx.secondDestinationLastLaneSize));
        }
    }
    CCU_IF(ctx.phase == BROADCAST_HIERARCHY_PHASE_ALLGATHER_1)
    {
        CCU_KERNEL_CHK_RET(DoHierarchicalLocalAllGather(
            ctx, ctx.zeroOffset, ctx.firstNormalLaneSize, ctx.firstLastLaneSize));
    }

    HCCL_INFO("[CcuBroadcastHierarchicalLayer0Kernel] end");
    return CCU_SUCCESS;
}

CcuResult CcuBroadcastHierarchicalLayer1Kernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuBroadcastHierarchicalKernelArg *>(arg);
    BroadcastHierarchicalContext ctx;
    ctx.arg = kernelArg;

    HCCL_INFO("[CcuBroadcastHierarchicalLayer1Kernel] start rank[%u] root[%u] rankSize[%llu]",
        ctx.arg->rankId, ctx.arg->rootId, static_cast<unsigned long long>(ctx.arg->rankSize));

    CCU_KERNEL_CHK_RET(InitHierarchicalResource(ctx));
    CCU_KERNEL_CHK_RET(LoadHierarchicalArgs(ctx));
    if (!BROADCAST_REUSE_HIERARCHY_REMOTE_ADDR) {
        CCU_KERNEL_CHK_RET(HierarchicalPreSync(ctx));
    }
    const uint32_t rootLaneIdx = FindSourceGatewayLane(ctx.arg, ctx.arg->rootId);
    const uint32_t rankLaneIdx = FindGatewayLane(ctx.arg, ctx.arg->rankId);
    if (rootLaneIdx >= ctx.arg->laneCount || rankLaneIdx >= ctx.arg->laneCount) {
        HCCL_ERROR("[CcuBroadcastHierarchicalLayer1Kernel] invalid root/rank lane[%u,%u]",
            rootLaneIdx, rankLaneIdx);
        return CcuResult::CCU_E_PARA;
    }
    const uint32_t destinationRoot = ctx.arg->destinationGatewayRanks[rootLaneIdx];
    ccu::Variable secondDestinationOffset;
    secondDestinationOffset = ctx.secondOffset;
    secondDestinationOffset += ctx.secondSourceHalfSize;

    CCU_IF(ctx.phase == BROADCAST_HIERARCHY_PHASE_SCATTER_0)
    {
        if (BROADCAST_REUSE_HIERARCHY_REMOTE_ADDR) {
            CCU_KERNEL_CHK_RET(HierarchicalPreSync(ctx));
        }
        CCU_KERNEL_CHK_RET(DoHierarchicalCrossScatter(ctx, secondDestinationOffset,
            secondDestinationOffset, ctx.secondDestinationHalfSize, ctx.arg->rootId,
            destinationRoot, HIERARCHY_SCATTER_READY_ID));
    }
    CCU_IF(ctx.phase == BROADCAST_HIERARCHY_PHASE_SCATTER_1)
    {
        CCU_KERNEL_CHK_RET(DoHierarchicalCrossScatterAllLanes(
            ctx, HIERARCHY_SCATTER_READY_ID));
    }
    CCU_IF(ctx.phase == BROADCAST_HIERARCHY_PHASE_ALLGATHER_0)
    {
        CCU_KERNEL_CHK_RET(DoHierarchicalCrossAllGatherA(ctx));
    }
    CCU_IF(ctx.phase == BROADCAST_HIERARCHY_PHASE_ALLGATHER_1)
    {
        if (ctx.arg->isSourceGroup != 0) {
            CCU_KERNEL_CHK_RET(DoHierarchicalCrossAllGatherB(
                ctx, ctx.secondOffset, ctx.secondSourceHalfSize));
        } else {
            CCU_KERNEL_CHK_RET(DoHierarchicalCrossAllGatherB(
                ctx, secondDestinationOffset, ctx.secondDestinationHalfSize));
        }
    }

    HCCL_INFO("[CcuBroadcastHierarchicalLayer1Kernel] end");
    return CCU_SUCCESS;
}

} // namespace ops_hccl