/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <hcomm/hcomm_primitives.h>

#include "ccu_kernel.h"
#include "log.h"

#define CCU_CHK_RET(call) \
    do { \
        CcuResult ccuRet = (call); \
        if (ccuRet != CCU_SUCCESS) { \
            return ccuRet; \
        } \
    } while (0)

namespace ops_hccl {

constexpr uint32_t OUTPUT_XN_ID = 1;
constexpr uint32_t TOKEN_XN_ID = 2;
constexpr uint32_t CKE_IDX = 0;
constexpr uint32_t POST_SYNC_ID = 3;

static CcuResult InitResources(CcuKernelContext &ctx)
{
    ctx.output.resize(ctx.arg->channelCount);
    ctx.token.resize(ctx.arg->channelCount);

    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        ctx.output[channelIdx] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], OUTPUT_XN_ID);
        ctx.token[channelIdx] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], TOKEN_XN_ID);
    }
    return CCU_SUCCESS;
}

static CcuResult LoadArgs(CcuKernelContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localOutput, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.currentRankOutputOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.normalSliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.lastSliceSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.repeatNum, argId++));
    return CCU_SUCCESS;
}

static CcuResult PreSync(CcuKernelContext &ctx)
{
    uint32_t variableMask = (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx],
            ctx.localOutput, OUTPUT_XN_ID, CKE_IDX, 1U << OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx],
            ctx.localToken, TOKEN_XN_ID, CKE_IDX, 1U << TOKEN_XN_ID));
    }
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIdx], CKE_IDX, variableMask));
    }
    return CCU_SUCCESS;
}

static CcuResult Transfer(CcuKernelContext &ctx, const ccu::Variable &sliceSize,
    const ccu::Variable &currentOffset)
{
    ccu::LocalAddr src;
    src.addr = ctx.input;
    src.addr += currentOffset;
    src.token = ctx.localToken;

    ccu::LocalAddr localDst;
    localDst.addr = ctx.localOutput;
    localDst.addr += ctx.currentRankOutputOffset;
    localDst.addr += currentOffset;
    localDst.token = ctx.localToken;

    std::vector<ccu::RemoteAddr> remoteDst(ctx.arg->channelCount);
    uint16_t eventMask = 0;
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        uint16_t channelMask = static_cast<uint16_t>(1U << channelIdx);
        remoteDst[channelIdx].addr = ctx.output[channelIdx];
        remoteDst[channelIdx].addr += ctx.currentRankOutputOffset;
        remoteDst[channelIdx].addr += currentOffset;
        remoteDst[channelIdx].token = ctx.token[channelIdx];
        CCU_CHK_RET(ccu::Write(ctx.arg->channels[channelIdx], remoteDst[channelIdx], src,
            sliceSize, ctx.event, channelMask));
        eventMask |= channelMask;
    }

    if (ctx.arg->handleSelf != 0) {
        uint16_t localMask = static_cast<uint16_t>(1U << ctx.arg->channelCount);
        CCU_CHK_RET(ccu::LocalCopy(localDst, src, sliceSize, ctx.event, localMask));
        eventMask |= localMask;
    }

    CCU_CHK_RET(ccu::EventWait(ctx.event, eventMask));
    return CCU_SUCCESS;
}

static CcuResult RepeatTransfer(CcuKernelContext &ctx)
{
    ctx.currentOffset = 0;
    ctx.currentSliceSize = ctx.normalSliceSize;
    ctx.constOne = 1;

    CCU_WHILE(ctx.repeatNum != UINT64_MAX)
    {
        ctx.currentSliceSize = ctx.normalSliceSize;
        CCU_IF(ctx.repeatNum == UINT64_MAX - 1)
        {
            ctx.currentSliceSize = ctx.lastSliceSize;
        }
        ctx.repeatNum += ctx.constOne;
        CCU_CHK_RET(Transfer(ctx, ctx.currentSliceSize, ctx.currentOffset));
        ctx.currentOffset += ctx.normalSliceSize;
    }
    return CCU_SUCCESS;
}

static CcuResult PostSync(CcuKernelContext &ctx)
{
    uint32_t syncMask = 1U << POST_SYNC_ID;
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[channelIdx], CKE_IDX, syncMask));
    }
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIdx], CKE_IDX, syncMask));
    }
    return CCU_SUCCESS;
}

CcuResult CcuKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAllGather *>(arg);
    if (kernelArg == nullptr || kernelArg->rankSize <= 1 || kernelArg->rankSize > MAX_RANK_SIZE ||
        kernelArg->channelCount == 0 || kernelArg->channelCount >= kernelArg->rankSize ||
        kernelArg->handleSelf > 1) {
        HCCL_ERROR("[CcuKernel] Invalid kernel argument.");
        return CCU_E_PARA;
    }

    CcuKernelContext ctx;
    ctx.arg = kernelArg;
    CCU_CHK_RET(InitResources(ctx));
    CCU_CHK_RET(LoadArgs(ctx));
    CCU_CHK_RET(PreSync(ctx));
    CCU_CHK_RET(RepeatTransfer(ctx));
    CCU_CHK_RET(PostSync(ctx));
    return CCU_SUCCESS;
}

static CcuResult InitPipelineResources(CcuPipelineContext &ctx)
{
    ctx.output.resize(ctx.arg->channelCount);
    ctx.token.resize(ctx.arg->channelCount);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        ctx.output[channelIdx] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], OUTPUT_XN_ID);
        ctx.token[channelIdx] =
            ccu::GetResByChannel<ccu::Variable>(ctx.arg->channels[channelIdx], TOKEN_XN_ID);
    }
    return CCU_SUCCESS;
}

static CcuResult LoadPipelineArgs(CcuPipelineContext &ctx)
{
    uint32_t argId = 0;
    CCU_CHK_RET(ccu::LoadArg(ctx.input, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localOutput, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.localToken, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.dataSize, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.meshBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.nhrOffset, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.nhrBytes, argId++));
    CCU_CHK_RET(ccu::LoadArg(ctx.stage, argId++));
    return CCU_SUCCESS;
}

static CcuResult PipelinePreSync(CcuPipelineContext &ctx)
{
    uint32_t variableMask = (1U << OUTPUT_XN_ID) | (1U << TOKEN_XN_ID);
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx],
            ctx.localOutput, OUTPUT_XN_ID, CKE_IDX, 1U << OUTPUT_XN_ID));
        CCU_CHK_RET(ccu::WriteVariableWithNotify(ctx.arg->channels[channelIdx],
            ctx.localToken, TOKEN_XN_ID, CKE_IDX, 1U << TOKEN_XN_ID));
    }
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIdx], CKE_IDX, variableMask));
    }
    return CCU_SUCCESS;
}

static CcuResult PipelinePostSync(CcuPipelineContext &ctx)
{
    uint32_t syncMask = 1U << POST_SYNC_ID;
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        CCU_CHK_RET(ccu::NotifyRecord(ctx.arg->channels[channelIdx], CKE_IDX, syncMask));
    }
    for (uint32_t channelIdx = 0; channelIdx < ctx.arg->channelCount; ++channelIdx) {
        CCU_CHK_RET(ccu::NotifyWait(ctx.arg->channels[channelIdx], CKE_IDX, syncMask));
    }
    return CCU_SUCCESS;
}

static void SetRankOffset(
    ccu::Variable &offset, const ccu::Variable &dataSize, uint32_t rank)
{
    offset = 0;
    for (uint32_t idx = 0; idx < rank; ++idx) {
        offset += dataSize;
    }
}

static CcuResult IntraMeshFirst(CcuPipelineContext &ctx, const CcuKernelArgPipelineIntra *arg)
{
    ccu::LocalAddr src;
    src.addr = ctx.input;
    src.token = ctx.localToken;

    ccu::Variable rankOffset;
    SetRankOffset(rankOffset, ctx.dataSize, arg->rankId);
    uint16_t eventMask = 0;
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        ccu::RemoteAddr dst;
        dst.addr = ctx.output[channelIdx];
        dst.addr += rankOffset;
        dst.token = ctx.token[channelIdx];
        uint16_t mask = static_cast<uint16_t>(1U << channelIdx);
        CCU_CHK_RET(ccu::Write(arg->channels[channelIdx], dst, src, ctx.meshBytes, ctx.event, mask));
        eventMask |= mask;
    }

    ccu::LocalAddr localDst;
    localDst.addr = ctx.localOutput;
    localDst.addr += rankOffset;
    localDst.token = ctx.localToken;
    uint16_t localMask = static_cast<uint16_t>(1U << arg->channelCount);
    CCU_CHK_RET(ccu::LocalCopy(localDst, src, ctx.meshBytes, ctx.event, localMask));
    eventMask |= localMask;
    CCU_CHK_RET(ccu::EventWait(ctx.event, eventMask));
    return CCU_SUCCESS;
}

static CcuResult IntraClosFirst(CcuPipelineContext &ctx, const CcuKernelArgPipelineIntra *arg)
{
    const uint32_t sourceRanks[] = {arg->rankId, arg->pairRank};
    uint16_t eventMask = 0;
    uint32_t eventIdx = 0;

    for (uint32_t sourceIdx = 0; sourceIdx < 2; ++sourceIdx) {
        ccu::Variable rankOffset;
        SetRankOffset(rankOffset, ctx.dataSize, sourceRanks[sourceIdx]);
        ccu::LocalAddr src;
        src.addr = ctx.localOutput;
        src.addr += rankOffset;
        src.addr += ctx.nhrOffset;
        src.token = ctx.localToken;

        for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
            ccu::RemoteAddr dst;
            dst.addr = ctx.output[channelIdx];
            dst.addr += rankOffset;
            dst.addr += ctx.nhrOffset;
            dst.token = ctx.token[channelIdx];
            uint16_t mask = static_cast<uint16_t>(1U << eventIdx);
            CCU_CHK_RET(ccu::Write(arg->channels[channelIdx], dst, src, ctx.nhrBytes, ctx.event, mask));
            eventMask |= mask;
            ++eventIdx;
        }
    }

    CCU_CHK_RET(ccu::EventWait(ctx.event, eventMask));
    return CCU_SUCCESS;
}

CcuResult CcuPipelineIntraKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgPipelineIntra *>(arg);
    if (kernelArg == nullptr || kernelArg->rankSize != MAX_RANK_SIZE ||
        kernelArg->channelCount != 7 || kernelArg->pairRank >= kernelArg->rankSize) {
        HCCL_ERROR("[CcuPipelineIntraKernel] Invalid kernel argument.");
        return CCU_E_PARA;
    }

    CcuPipelineContext ctx;
    ctx.arg = kernelArg;
    CCU_CHK_RET(InitPipelineResources(ctx));
    CCU_CHK_RET(LoadPipelineArgs(ctx));
    CCU_IF(ctx.stage == 0)
    {
        CCU_CHK_RET(PipelinePreSync(ctx));
        CCU_CHK_RET(IntraMeshFirst(ctx, kernelArg));
    }
    CCU_ELSE
    {
        CCU_CHK_RET(IntraClosFirst(ctx, kernelArg));
    }
    CCU_CHK_RET(PipelinePostSync(ctx));
    return CCU_SUCCESS;
}

static CcuResult InterClosFirst(CcuPipelineContext &ctx, const CcuKernelArgPipelineInter *arg)
{
    ccu::Variable rankOffset;
    SetRankOffset(rankOffset, ctx.dataSize, arg->rankId);
    ccu::LocalAddr src;
    src.addr = ctx.input;
    src.addr += ctx.nhrOffset;
    src.token = ctx.localToken;

    ccu::RemoteAddr dst;
    dst.addr = ctx.output[0];
    dst.addr += rankOffset;
    dst.addr += ctx.nhrOffset;
    dst.token = ctx.token[0];
    CCU_CHK_RET(ccu::Write(arg->channels[0], dst, src, ctx.nhrBytes, ctx.event, 1U));

    ccu::LocalAddr localDst;
    localDst.addr = ctx.localOutput;
    localDst.addr += rankOffset;
    localDst.addr += ctx.nhrOffset;
    localDst.token = ctx.localToken;
    CCU_CHK_RET(ccu::LocalCopy(localDst, src, ctx.nhrBytes, ctx.event, 2U));
    CCU_CHK_RET(ccu::EventWait(ctx.event, 3U));
    return CCU_SUCCESS;
}

static CcuResult InterMeshFirst(CcuPipelineContext &ctx, const CcuKernelArgPipelineInter *arg)
{
    uint16_t eventMask = 0;
    for (uint32_t rankIdx = 0; rankIdx < arg->localRankCount; ++rankIdx) {
        ccu::Variable rankOffset;
        SetRankOffset(rankOffset, ctx.dataSize, arg->localRanks[rankIdx]);
        ccu::LocalAddr src;
        src.addr = ctx.localOutput;
        src.addr += rankOffset;
        src.token = ctx.localToken;

        ccu::RemoteAddr dst;
        dst.addr = ctx.output[0];
        dst.addr += rankOffset;
        dst.token = ctx.token[0];
        uint16_t mask = static_cast<uint16_t>(1U << rankIdx);
        CCU_CHK_RET(ccu::Write(arg->channels[0], dst, src, ctx.meshBytes, ctx.event, mask));
        eventMask |= mask;
    }
    CCU_CHK_RET(ccu::EventWait(ctx.event, eventMask));
    return CCU_SUCCESS;
}

CcuResult CcuPipelineInterKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgPipelineInter *>(arg);
    if (kernelArg == nullptr || kernelArg->rankSize != MAX_RANK_SIZE ||
        kernelArg->channelCount != 1 || kernelArg->localRankCount != 8) {
        HCCL_ERROR("[CcuPipelineInterKernel] Invalid kernel argument.");
        return CCU_E_PARA;
    }

    CcuPipelineContext ctx;
    ctx.arg = kernelArg;
    CCU_CHK_RET(InitPipelineResources(ctx));
    CCU_CHK_RET(LoadPipelineArgs(ctx));
    CCU_IF(ctx.stage == 0)
    {
        CCU_CHK_RET(PipelinePreSync(ctx));
        CCU_CHK_RET(InterClosFirst(ctx, kernelArg));
    }
    CCU_ELSE
    {
        CCU_CHK_RET(InterMeshFirst(ctx, kernelArg));
    }
    CCU_CHK_RET(PipelinePostSync(ctx));
    return CCU_SUCCESS;
}

static CcuResult AsymmetricIntraMeshFirst(
    CcuPipelineContext &ctx, const CcuKernelArgAsymmetricIntra *arg)
{
    ccu::LocalAddr src;
    src.addr = ctx.input;
    src.token = ctx.localToken;

    ccu::Variable rankOffset;
    SetRankOffset(rankOffset, ctx.dataSize, arg->rankId);
    uint16_t eventMask = 0;
    for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
        ccu::RemoteAddr dst;
        dst.addr = ctx.output[channelIdx];
        dst.addr += rankOffset;
        dst.token = ctx.token[channelIdx];
        uint16_t mask = static_cast<uint16_t>(1U << channelIdx);
        CCU_CHK_RET(ccu::Write(arg->channels[channelIdx], dst, src, ctx.meshBytes, ctx.event, mask));
        eventMask |= mask;
    }

    ccu::LocalAddr localDst;
    localDst.addr = ctx.localOutput;
    localDst.addr += rankOffset;
    localDst.token = ctx.localToken;
    uint16_t localMask = static_cast<uint16_t>(1U << arg->channelCount);
    CCU_CHK_RET(ccu::LocalCopy(localDst, src, ctx.meshBytes, ctx.event, localMask));
    eventMask |= localMask;
    CCU_CHK_RET(ccu::EventWait(ctx.event, eventMask));
    return CCU_SUCCESS;
}

static CcuResult AsymmetricIntraClosFirst(
    CcuPipelineContext &ctx, const CcuKernelArgAsymmetricIntra *arg)
{
    uint16_t eventMask = 0;
    uint32_t eventIdx = 0;
    for (uint32_t relayIdx = 0; relayIdx < arg->relayRankCount; ++relayIdx) {
        ccu::Variable rankOffset;
        SetRankOffset(rankOffset, ctx.dataSize, arg->relayRanks[relayIdx]);
        ccu::LocalAddr src;
        src.addr = ctx.localOutput;
        src.addr += rankOffset;
        src.addr += ctx.nhrOffset;
        src.token = ctx.localToken;

        for (uint32_t channelIdx = 0; channelIdx < arg->channelCount; ++channelIdx) {
            ccu::RemoteAddr dst;
            dst.addr = ctx.output[channelIdx];
            dst.addr += rankOffset;
            dst.addr += ctx.nhrOffset;
            dst.token = ctx.token[channelIdx];
            uint16_t mask = static_cast<uint16_t>(1U << eventIdx);
            CCU_CHK_RET(ccu::Write(
                arg->channels[channelIdx], dst, src, ctx.nhrBytes, ctx.event, mask));
            eventMask |= mask;
            ++eventIdx;
        }
    }
    CCU_CHK_RET(ccu::EventWait(ctx.event, eventMask));
    return CCU_SUCCESS;
}

CcuResult CcuAsymmetricIntraKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAsymmetricIntra *>(arg);
    if (kernelArg == nullptr || kernelArg->rankSize != 12 ||
        (kernelArg->channelCount != 3 && kernelArg->channelCount != 7) ||
        kernelArg->relayRankCount == 0 || kernelArg->relayRankCount > 3) {
        HCCL_ERROR("[CcuAsymmetricIntraKernel] Invalid kernel argument.");
        return CCU_E_PARA;
    }
    uint32_t eventCount = kernelArg->channelCount * kernelArg->relayRankCount;
    if (eventCount > 15) {
        HCCL_ERROR("[CcuAsymmetricIntraKernel] Too many transfer events.");
        return CCU_E_PARA;
    }

    CcuPipelineContext ctx;
    ctx.arg = kernelArg;
    CCU_CHK_RET(InitPipelineResources(ctx));
    CCU_CHK_RET(LoadPipelineArgs(ctx));
    CCU_IF(ctx.stage == 0)
    {
        CCU_CHK_RET(PipelinePreSync(ctx));
        CCU_CHK_RET(AsymmetricIntraMeshFirst(ctx, kernelArg));
    }
    CCU_ELSE
    {
        CCU_CHK_RET(AsymmetricIntraClosFirst(ctx, kernelArg));
    }
    CCU_CHK_RET(PipelinePostSync(ctx));
    return CCU_SUCCESS;
}

static CcuResult AsymmetricInterClosFirst(
    CcuPipelineContext &ctx, const CcuKernelArgAsymmetricInter *arg)
{
    ccu::Variable rankOffset;
    SetRankOffset(rankOffset, ctx.dataSize, arg->rankId);
    ccu::LocalAddr src;
    src.addr = ctx.input;
    src.addr += ctx.nhrOffset;
    src.token = ctx.localToken;

    uint16_t eventMask = 0;
    for (uint32_t channelIdx = 0; channelIdx < arg->closSendCount; ++channelIdx) {
        ccu::RemoteAddr dst;
        dst.addr = ctx.output[channelIdx];
        dst.addr += rankOffset;
        dst.addr += ctx.nhrOffset;
        dst.token = ctx.token[channelIdx];
        uint16_t mask = static_cast<uint16_t>(1U << channelIdx);
        CCU_CHK_RET(ccu::Write(
            arg->channels[channelIdx], dst, src, ctx.nhrBytes, ctx.event, mask));
        eventMask |= mask;
    }

    ccu::LocalAddr localDst;
    localDst.addr = ctx.localOutput;
    localDst.addr += rankOffset;
    localDst.addr += ctx.nhrOffset;
    localDst.token = ctx.localToken;
    uint16_t localMask = static_cast<uint16_t>(1U << arg->channelCount);
    CCU_CHK_RET(ccu::LocalCopy(localDst, src, ctx.nhrBytes, ctx.event, localMask));
    eventMask |= localMask;
    CCU_CHK_RET(ccu::EventWait(ctx.event, eventMask));
    return CCU_SUCCESS;
}

static CcuResult AsymmetricInterMeshFirst(
    CcuPipelineContext &ctx, const CcuKernelArgAsymmetricInter *arg)
{
    uint16_t eventMask = 0;
    uint32_t eventIdx = 0;
    for (uint32_t channelIdx = 0; channelIdx < arg->meshSendCount; ++channelIdx) {
        for (uint32_t rankIdx = 0; rankIdx < arg->localRankCount; ++rankIdx) {
            ccu::Variable rankOffset;
            SetRankOffset(rankOffset, ctx.dataSize, arg->localRanks[rankIdx]);
            ccu::LocalAddr src;
            src.addr = ctx.localOutput;
            src.addr += rankOffset;
            src.token = ctx.localToken;

            ccu::RemoteAddr dst;
            dst.addr = ctx.output[channelIdx];
            dst.addr += rankOffset;
            dst.token = ctx.token[channelIdx];
            uint16_t mask = static_cast<uint16_t>(1U << eventIdx);
            CCU_CHK_RET(ccu::Write(
                arg->channels[channelIdx], dst, src, ctx.meshBytes, ctx.event, mask));
            eventMask |= mask;
            ++eventIdx;
        }
    }
    if (eventMask != 0) {
        CCU_CHK_RET(ccu::EventWait(ctx.event, eventMask));
    }
    return CCU_SUCCESS;
}

CcuResult CcuAsymmetricInterKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuKernelArgAsymmetricInter *>(arg);
    if (kernelArg == nullptr || kernelArg->rankSize != 12 ||
        (kernelArg->channelCount != 1 && kernelArg->channelCount != 2) ||
        (kernelArg->localRankCount != 4 && kernelArg->localRankCount != 8) ||
        kernelArg->closSendCount != 1 || kernelArg->meshSendCount > kernelArg->channelCount ||
        kernelArg->meshSendCount * kernelArg->localRankCount > 15) {
        HCCL_ERROR("[CcuAsymmetricInterKernel] Invalid kernel argument.");
        return CCU_E_PARA;
    }

    CcuPipelineContext ctx;
    ctx.arg = kernelArg;
    CCU_CHK_RET(InitPipelineResources(ctx));
    CCU_CHK_RET(LoadPipelineArgs(ctx));
    CCU_IF(ctx.stage == 0)
    {
        CCU_CHK_RET(PipelinePreSync(ctx));
        CCU_CHK_RET(AsymmetricInterClosFirst(ctx, kernelArg));
    }
    CCU_ELSE
    {
        CCU_CHK_RET(AsymmetricInterMeshFirst(ctx, kernelArg));
    }
    CCU_CHK_RET(PipelinePostSync(ctx));
    return CCU_SUCCESS;
}

} // namespace ops_hccl
